/*
Copyright 2025  The Hyperlight Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

//! Dynamic FAT mount creation and management for guest code.
//!
//! This module allows guests to dynamically create FAT filesystems in memory
//! at runtime, without requiring the host to pre-configure them.
//!
//! # Usage
//!
//! ```ignore
//! use hyperlight_guest::fs;
//!
//! // Create a 1MB FAT mount at /scratch
//! fs::create_fat_mount("/scratch", 1024 * 1024)?;
//!
//! // Use it like any other FAT mount
//! let mut file = fs::OpenOptions::new()
//!     .write(true)
//!     .create(true)
//!     .open("/scratch/temp.txt")?;
//! file.write_all(b"temporary data")?;
//!
//! // When done, unmount to free memory
//! fs::unmount("/scratch")?;
//! ```
//!
//! # Memory Management
//!
//! Guest-created FAT mounts allocate memory from the guest heap. The minimum
//! size is 64KB (required for FAT12). Memory is freed when `unmount()` is called.
//!
//! # Host Visibility
//!
//! Guest-created mounts use regular heap memory, which is **not** visible to
//! the host via MAP_SHARED. If you need host extraction, use host-created
//! FAT mounts instead.
//!
//! # Limitations
//!
//! - Cannot unmount host-provided mounts (returns `FsError::PermissionDenied`)
//! - Mount paths must be absolute and not conflict with existing mounts

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::UnsafeCell;

use super::error::FsError;
use super::fat::{GuestFat, RawMemoryStorage};
use super::vfs::{Mount, MountBackend};
use super::{vfs, vfs_mut};

/// Minimum FAT image size (64KB for FAT12).
pub const MIN_FAT_SIZE: usize = 64 * 1024;

/// Maximum FAT image size (128MB to prevent excessive memory use).
/// Note: FAT32 requires ~32MB+ so sizes above that will use FAT32.
pub const MAX_FAT_SIZE: usize = 128 * 1024 * 1024;

/// Tracks guest-created mounts for unmount permission checks.
///
/// When a mount is created via `create_fat_mount()`, its path is added here.
/// Only mounts in this list can be unmounted via `unmount()`.
static GUEST_CREATED_MOUNTS: GuestMountsCell = GuestMountsCell(UnsafeCell::new(Vec::new()));

struct GuestMountsCell(UnsafeCell<Vec<GuestMountInfo>>);

// SAFETY: Guest is single-threaded.
unsafe impl Sync for GuestMountsCell {}

/// Information about a guest-created mount.
struct GuestMountInfo {
    /// Mount path (e.g., "/scratch").
    path: String,
    /// Pointer to the allocated memory (for deallocation).
    memory_ptr: *mut u8,
    /// Size of the allocated memory.
    memory_size: usize,
}

impl GuestMountsCell {
    fn add(&self, path: String, ptr: *mut u8, size: usize) {
        // SAFETY: Guest is single-threaded
        unsafe {
            (*self.0.get()).push(GuestMountInfo {
                path,
                memory_ptr: ptr,
                memory_size: size,
            });
        }
    }

    fn contains(&self, path: &str) -> bool {
        // SAFETY: Guest is single-threaded
        unsafe { (*self.0.get()).iter().any(|m| m.path == path) }
    }

    fn remove(&self, path: &str) -> Option<GuestMountInfo> {
        // SAFETY: Guest is single-threaded
        unsafe {
            let mounts = &mut *self.0.get();
            if let Some(pos) = mounts.iter().position(|m| m.path == path) {
                Some(mounts.remove(pos))
            } else {
                None
            }
        }
    }
}

/// Create a new FAT filesystem mount at the specified path.
///
/// This allocates memory from the guest heap, formats it as FAT, and
/// registers it in the VFS at the given mount path.
///
/// # Arguments
///
/// * `mount_path` - Absolute path where the FAT mount will be accessible
///   (e.g., "/scratch"). Must start with "/" and not conflict with existing mounts.
/// * `size` - Size in bytes for the FAT image. Must be between 64KB and 16MB.
///
/// # Errors
///
/// - `FsError::InvalidPath` - `mount_path` doesn't start with "/"
/// - `FsError::InvalidArgument` - `size` is too small (< 64KB) or too large (> 16MB)
/// - `FsError::AlreadyExists` - A mount already exists at this path
/// - `FsError::OutOfMemory` - Failed to allocate memory for the FAT image
/// - `FsError::IoError` - Failed to format the FAT filesystem
/// - `FsError::NotInitialized` - HyperlightFS not initialized
///
/// # Example
///
/// ```ignore
/// use hyperlight_guest::fs;
///
/// // Create a 512KB FAT mount
/// fs::create_fat_mount("/temp", 512 * 1024)?;
///
/// // Now /temp is usable as a FAT filesystem
/// fs::mkdir("/temp/subdir")?;
/// ```
///
/// # Notes
///
/// - The FAT variant (FAT12/FAT16/FAT32) is auto-selected based on size
/// - Memory is allocated from the guest heap and freed on `unmount()`
/// - Guest-created mounts are NOT visible to the host (not MAP_SHARED)
pub fn create_fat_mount(mount_path: &str, size: usize) -> Result<(), FsError> {
    // Validate mount path
    if !mount_path.starts_with('/') {
        return Err(FsError::InvalidPath);
    }

    // Validate size
    if size < MIN_FAT_SIZE {
        return Err(FsError::InvalidArgument);
    }
    if size > MAX_FAT_SIZE {
        return Err(FsError::InvalidArgument);
    }

    // Check VFS is initialized
    let _ = vfs()?;

    // Allocate memory for the FAT image
    // We use a boxed slice to get heap allocation, then leak it to get a raw pointer.
    // The memory will be reclaimed when unmount() is called.
    let memory: Box<[u8]> = alloc::vec![0u8; size].into_boxed_slice();
    let memory_ptr = Box::into_raw(memory) as *mut u8;

    // Format the memory as FAT
    // SAFETY: memory_ptr points to valid, zeroed memory of `size` bytes
    let format_result = unsafe { format_fat_in_memory(memory_ptr, size) };

    if let Err(e) = format_result {
        // Format failed - reclaim the memory
        // SAFETY: memory_ptr was created from Box::into_raw above
        unsafe {
            let _ = Box::from_raw(core::slice::from_raw_parts_mut(memory_ptr, size));
        }
        return Err(e);
    }

    // Create GuestFat from the formatted memory
    // SAFETY: memory_ptr points to valid FAT image of `size` bytes
    let fat = match unsafe { GuestFat::from_memory(memory_ptr, size) } {
        Ok(fat) => fat,
        Err(e) => {
            // Failed to open FAT - reclaim memory
            // SAFETY: memory_ptr was created from Box::into_raw above
            unsafe {
                let _ = Box::from_raw(core::slice::from_raw_parts_mut(memory_ptr, size));
            }
            return Err(e);
        }
    };

    // Create the mount
    let mount = Mount::new(String::from(mount_path), MountBackend::Fat(fat));

    // Add to VFS
    // SAFETY: We're not holding any other VFS references
    let vfs = unsafe { vfs_mut()? };
    if let Err(e) = vfs.add_mount(mount) {
        // Mount failed (likely duplicate) - reclaim memory
        // SAFETY: memory_ptr was created from Box::into_raw above
        unsafe {
            let _ = Box::from_raw(core::slice::from_raw_parts_mut(memory_ptr, size));
        }
        return Err(e);
    }

    // Track this as a guest-created mount
    GUEST_CREATED_MOUNTS.add(String::from(mount_path), memory_ptr, size);

    Ok(())
}

/// Unmount a guest-created FAT mount and free its memory.
///
/// This removes the mount from the VFS and deallocates the memory used
/// by the FAT image. Only mounts created via `create_fat_mount()` can
/// be unmounted; attempting to unmount a host-provided mount will fail.
///
/// # Arguments
///
/// * `mount_path` - The path of the mount to remove (must match exactly
///   what was passed to `create_fat_mount()`).
///
/// # Errors
///
/// - `FsError::NotFound` - No mount exists at this path
/// - `FsError::PermissionDenied` - Mount was created by the host, not the guest
/// - `FsError::FileLocked` - Files are still open on this mount
/// - `FsError::NotInitialized` - HyperlightFS not initialized
///
/// # Example
///
/// ```ignore
/// use hyperlight_guest::fs;
///
/// fs::create_fat_mount("/temp", 512 * 1024)?;
/// // ... use /temp ...
/// fs::unmount("/temp")?; // Memory is freed
/// ```
pub fn unmount(mount_path: &str) -> Result<(), FsError> {
    // Check if this is a guest-created mount
    if !GUEST_CREATED_MOUNTS.contains(mount_path) {
        // Either doesn't exist or was host-created
        // Check VFS to distinguish
        let vfs = vfs()?;
        let mount_exists = vfs.mounts().any(|m| m.path() == mount_path);
        if mount_exists {
            return Err(FsError::PermissionDenied);
        } else {
            return Err(FsError::NotFound);
        }
    }

    // Check for open files on this mount BEFORE modifying any state.
    // This prevents partial state corruption if there are open files.
    if super::fd::has_open_files_on_mount(mount_path) {
        return Err(FsError::FileLocked);
    }

    // Remove from tracking FIRST so we have the memory info for cleanup.
    // If VFS removal fails, we restore the tracking entry (rollback).
    let info = GUEST_CREATED_MOUNTS
        .remove(mount_path)
        .ok_or(FsError::NotFound)?;

    // Remove from VFS
    // SAFETY: We're not holding any other VFS references
    let vfs = unsafe { vfs_mut()? };
    if let Err(e) = vfs.remove_mount(mount_path) {
        // VFS removal failed - put the tracking info back!
        GUEST_CREATED_MOUNTS.add(info.path.clone(), info.memory_ptr, info.memory_size);
        return Err(e);
    }

    // Success - now free the memory
    // SAFETY: info.memory_ptr was created from Box::into_raw in create_fat_mount
    unsafe {
        let _ = Box::from_raw(core::slice::from_raw_parts_mut(
            info.memory_ptr,
            info.memory_size,
        ));
    }

    Ok(())
}

/// Check if a mount path was created by the guest (vs. provided by the host).
///
/// # Returns
///
/// - `true` if the mount was created via `create_fat_mount()`
/// - `false` if the mount was provided by the host or doesn't exist
pub fn is_guest_created_mount(mount_path: &str) -> bool {
    GUEST_CREATED_MOUNTS.contains(mount_path)
}

/// Format a memory region as a FAT filesystem.
///
/// # Safety
///
/// The caller must ensure:
/// - `ptr` points to valid, writable memory of at least `size` bytes
/// - The memory is not accessed by other code during formatting
unsafe fn format_fat_in_memory(ptr: *mut u8, size: usize) -> Result<(), FsError> {
    // Create storage adapter
    // SAFETY: Caller guarantees ptr/size validity
    let mut storage = unsafe { RawMemoryStorage::new(ptr, size) };

    // Format with default options (fatfs auto-selects FAT12/16/32 based on size)
    let options = fatfs::FormatVolumeOptions::new();

    fatfs::format_volume(&mut storage, options).map_err(|_| FsError::IoError)?;

    Ok(())
}

// Note: Unit tests for this module are in the integration test suite
// (hyperlight_fs_test.rs) since they require a full guest environment
// with proper heap allocation and VFS initialization.
