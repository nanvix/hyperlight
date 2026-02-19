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

//! HyperlightFS manifest parsing and inode lookup.
//!
//! This module parses the FlatBuffer manifest from the host and provides:
//! - VFS (Virtual Filesystem) with mount table for path routing
//! - Path-based lookup to resolve guest paths to inodes
//!
//! # Initialization Flow
//!
//! Per spec §5.5.5:
//! 1. Parse FlatBuffer manifest
//! 2. Identify FAT mounts vs RO files by `InodeType`
//! 3. Initialize backends:
//!    - FAT mounts: Create `GuestFat` from memory region
//!    - RO files: Handled via root ReadOnly mount
//! 4. Build VFS mount table (sorted by path length for longest-prefix matching)
//!
//! Note: Initial cwd defaults to "/" (set by `Vfs::new()`).

use alloc::string::String;
use alloc::vec::Vec;
use core::cell::UnsafeCell;

use hyperlight_common::flatbuffer_wrappers::hyperlight_fs::{HyperlightFSData, InodeData};

use super::error::FsError;
use super::fat::GuestFat;
use super::vfs::{Mount, MountBackend, Vfs};

/// A FAT memory region (base address and size).
#[derive(Clone, Copy)]
struct FatRegion {
    base: u64,
    size: u64,
}

/// Global FAT region table for page fault handler queries.
///
/// This is separate from FS_STATE because it needs to be accessible
/// from the page fault handler without going through the full FS lookup.
static FAT_REGIONS: FatRegionCell = FatRegionCell(UnsafeCell::new(Vec::new()));

struct FatRegionCell(UnsafeCell<Vec<FatRegion>>);

// SAFETY: Guest is single-threaded.
unsafe impl Sync for FatRegionCell {}

impl FatRegionCell {
    fn add(&self, base: u64, size: u64) {
        // SAFETY: Guest is single-threaded
        unsafe {
            (*self.0.get()).push(FatRegion { base, size });
        }
    }

    fn contains(&self, addr: u64) -> bool {
        // SAFETY: Guest is single-threaded
        let regions = unsafe { &*self.0.get() };
        regions
            .iter()
            .any(|r| addr >= r.base && addr < r.base + r.size)
    }
}

/// Parsed filesystem state.
struct FsState {
    /// Inode table from the manifest.
    inodes: Vec<InodeData>,
    /// Virtual filesystem with mount table.
    vfs: Vfs,
}

/// Global filesystem state.
///
/// SAFETY: Guest code is single-threaded, so this is safe.
static FS_STATE: FsStateCell = FsStateCell(UnsafeCell::new(None));

struct FsStateCell(UnsafeCell<Option<FsState>>);

// SAFETY: Guest is single-threaded.
unsafe impl Sync for FsStateCell {}

impl FsStateCell {
    fn get(&self) -> Option<&FsState> {
        // SAFETY: Guest is single-threaded.
        unsafe { (*self.0.get()).as_ref() }
    }

    fn is_initialized(&self) -> bool {
        // SAFETY: Guest is single-threaded.
        unsafe { (*self.0.get()).is_some() }
    }

    fn set(&self, state: FsState) {
        // SAFETY: Guest is single-threaded.
        unsafe {
            *self.0.get() = Some(state);
        }
    }

    #[cfg(test)]
    fn reset(&self) {
        // SAFETY: Guest is single-threaded.
        unsafe {
            *self.0.get() = None;
        }
    }
}

/// Initialize the filesystem from a manifest.
///
/// Parses the FlatBuffer manifest and builds the VFS mount table:
/// - FAT mount inodes become `MountBackend::Fat` mounts
/// - A root ReadOnly mount is added for RO file access
///
/// # Arguments
///
/// * `manifest_ptr` - Pointer to the FlatBuffer manifest data
/// * `manifest_len` - Length of the manifest in bytes
///
/// # Safety
///
/// The caller must ensure:
/// - `manifest_ptr` points to valid memory containing the manifest
/// - The memory remains valid for the lifetime of the filesystem
/// - This function is only called once
///
/// # Spec Reference
///
/// See spec §5.5.5 for initialization flow.
pub unsafe fn init(manifest_ptr: *const u8, manifest_len: usize) -> Result<(), FsError> {
    // Check if already initialized - init() must only be called once
    if FS_STATE.is_initialized() {
        return Err(FsError::NotSupported);
    }

    if manifest_ptr.is_null() || manifest_len == 0 {
        return Err(FsError::InvalidManifest);
    }

    // SAFETY: Caller guarantees pointer is valid for manifest_len bytes
    let manifest_bytes = unsafe { core::slice::from_raw_parts(manifest_ptr, manifest_len) };

    let fs_data: HyperlightFSData = manifest_bytes
        .try_into()
        .map_err(|_| FsError::InvalidManifest)?;

    // Build the VFS mount table
    let mut vfs = Vfs::new();

    // Process FAT mount inodes first (they take precedence over RO root)
    for inode in &fs_data.inodes {
        if inode.is_fat_mount() {
            // Validate FAT mount parameters before creating GuestFat
            if inode.guest_address == 0 {
                return Err(FsError::InvalidManifest);
            }
            if inode.size == 0 {
                return Err(FsError::InvalidManifest);
            }

            // Register this FAT region for page fault handler queries.
            // Must be done BEFORE creating GuestFat since accessing the FAT
            // may trigger page faults that need to know this is a FAT region.
            FAT_REGIONS.add(inode.guest_address, inode.size);

            // Create GuestFat from the memory region
            // SAFETY: Host has mapped the FAT image at guest_address with size bytes.
            // We validated guest_address != 0 and size != 0 above.
            let fat = unsafe {
                GuestFat::from_memory(inode.guest_address as *mut u8, inode.size as usize)
            }
            .map_err(|_| FsError::InvalidManifest)?;

            let mount = Mount::new(inode.path.clone(), MountBackend::Fat(fat));
            vfs.add_mount(mount).map_err(|_| FsError::InvalidManifest)?;
        }
    }

    // Add root ReadOnly mount for RO files.
    // Due to longest-prefix matching, this "/" mount has lowest priority and
    // acts as a fallback for paths not covered by more specific FAT mounts.
    // If a FAT mount at "/" already exists, this will return AlreadyExists
    // which we intentionally ignore - the FAT mount takes precedence.
    let ro_root = Mount::new(String::from("/"), MountBackend::ReadOnly);
    let _ = vfs.add_mount(ro_root);

    let state = FsState {
        inodes: fs_data.inodes,
        vfs,
    };

    FS_STATE.set(state);

    Ok(())
}

/// Check if the filesystem is initialized.
pub fn is_initialized() -> bool {
    FS_STATE.get().is_some()
}

/// Check if an address falls within a FAT memory region.
///
/// This is used by the page fault handler to determine whether to create
/// a read-write PTE (for FAT regions) or a read-only PTE (for RO files).
///
/// # Arguments
///
/// * `addr` - The faulting address to check
///
/// # Returns
///
/// `true` if the address is within a registered FAT region, `false` otherwise.
///
/// # Note
///
/// FAT regions are registered during `init()` before the corresponding
/// GuestFat is created. This ensures the page fault handler can correctly
/// identify FAT pages even during initial FAT filesystem setup.
#[inline]
pub fn is_fat_region(addr: u64) -> bool {
    FAT_REGIONS.contains(addr)
}

/// Reset the filesystem state (for testing only).
///
/// This clears all filesystem state, allowing `init()` to be called again.
#[cfg(test)]
pub fn reset() {
    FS_STATE.reset();
}

/// Get a reference to the VFS.
///
/// Returns the VFS mount table for routing file operations to the
/// appropriate backend (ReadOnly or FAT).
///
/// # Errors
///
/// Returns `FsError::NotInitialized` if `init()` has not been called.
pub fn vfs() -> Result<&'static Vfs, FsError> {
    let state = FS_STATE.get().ok_or(FsError::NotInitialized)?;
    Ok(&state.vfs)
}

/// Get a mutable reference to the VFS.
///
/// # Safety
///
/// The caller must ensure that no other references (mutable or immutable)
/// to the VFS exist when calling this function. While the guest is
/// single-threaded, holding a `&Vfs` from `vfs()` while calling `vfs_mut()`
/// would create aliasing mutable references, which is undefined behavior.
///
/// # Errors
///
/// Returns `FsError::NotInitialized` if `init()` has not been called.
///
/// # Example
///
/// ```ignore
/// // WRONG - UB!
/// let vfs_ref = vfs().unwrap();
/// let vfs_mut_ref = unsafe { vfs_mut().unwrap() }; // Aliasing!
///
/// // CORRECT
/// unsafe {
///     let vfs_mut_ref = vfs_mut().unwrap();
///     vfs_mut_ref.set_cwd("/data").unwrap();
/// } // vfs_mut_ref dropped here
/// let vfs_ref = vfs().unwrap(); // Safe now
/// ```
pub unsafe fn vfs_mut() -> Result<&'static mut Vfs, FsError> {
    // SAFETY: Caller guarantees no aliasing references exist
    let state = unsafe { (*FS_STATE.0.get()).as_mut() }.ok_or(FsError::NotInitialized)?;
    Ok(&mut state.vfs)
}

/// Look up an inode by guest path.
///
/// Returns the inode index and a reference to the inode data.
pub fn lookup(path: &str) -> Result<(usize, &'static InodeData), FsError> {
    let state = FS_STATE.get().ok_or(FsError::NotInitialized)?;

    // Normalize path: must start with /
    if !path.starts_with('/') {
        return Err(FsError::InvalidPath);
    }

    // Handle root directory specially
    if path == "/" {
        for (idx, inode) in state.inodes.iter().enumerate() {
            if inode.path == "/" && inode.is_dir() {
                return Ok((idx, inode));
            }
        }
        return Err(FsError::NotFound);
    }

    // Normalize: remove trailing slash, collapse multiple slashes
    let normalized = normalize_path(path);

    // Linear search through inodes
    // TODO: Could build a hashmap for O(1) lookup if performance matters
    for (idx, inode) in state.inodes.iter().enumerate() {
        if inode.path == normalized {
            return Ok((idx, inode));
        }
    }

    Err(FsError::NotFound)
}

/// Look up a file (not directory) by guest path.
pub fn lookup_file(path: &str) -> Result<(usize, &'static InodeData), FsError> {
    let (idx, inode) = lookup(path)?;
    if inode.is_dir() {
        return Err(FsError::NotAFile);
    }
    Ok((idx, inode))
}

/// List children of a directory.
pub fn list_dir(path: &str) -> Result<Vec<&'static InodeData>, FsError> {
    let (parent_idx, parent_inode) = lookup(path)?;
    if parent_inode.is_file() {
        return Err(FsError::NotADirectory);
    }

    let state = FS_STATE.get().ok_or(FsError::NotInitialized)?;

    let children: Vec<_> = state
        .inodes
        .iter()
        .filter(|inode| inode.parent == parent_idx as u32 && inode.path != parent_inode.path)
        .collect();

    Ok(children)
}

/// Normalize a path by removing trailing slashes and collapsing multiple slashes.
fn normalize_path(path: &str) -> String {
    let mut result = String::with_capacity(path.len());
    let mut prev_slash = false;

    for c in path.chars() {
        if c == '/' {
            if !prev_slash {
                result.push(c);
            }
            prev_slash = true;
        } else {
            result.push(c);
            prev_slash = false;
        }
    }

    // Remove trailing slash (unless it's just "/")
    if result.len() > 1 && result.ends_with('/') {
        result.pop();
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_path() {
        assert_eq!(normalize_path("/"), "/");
        assert_eq!(normalize_path("/foo"), "/foo");
        assert_eq!(normalize_path("/foo/"), "/foo");
        assert_eq!(normalize_path("/foo/bar"), "/foo/bar");
        assert_eq!(normalize_path("/foo//bar"), "/foo/bar");
        assert_eq!(normalize_path("//foo//bar//"), "/foo/bar");
    }

    #[test]
    fn test_init_null_ptr() {
        let result = unsafe { init(core::ptr::null(), 0) };
        assert_eq!(result, Err(FsError::InvalidManifest));
    }
}
