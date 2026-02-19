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

//! HyperlightFS: Virtual Filesystem for guest code.
//!
//! This module provides a unified VFS that supports:
//! - **Read-only files**: Memory-mapped from the host, zero-copy access
//! - **FAT filesystems**: Read-write access to FAT12/16/32 images
//!
//! The VFS routes operations to the appropriate backend based on mount points.
//! Paths are resolved using longest-prefix matching.
//!
//! # Usage
//!
//! ## Reading files (read-only, memory-mapped)
//!
//! ```ignore
//! use embedded_io::Read;
//! use hyperlight_guest::fs;
//!
//! // Open and read a file from a read-only mount
//! let mut file = fs::open("/config.json")?;
//! let mut buf = [0u8; 1024];
//! let bytes_read = file.read(&mut buf)?;
//! ```
//!
//! ## Writing files (FAT filesystem)
//!
//! ```ignore
//! use embedded_io::{Read, Write};
//! use hyperlight_guest::fs::{self, OpenOptions};
//!
//! // Create and write to a new file on a FAT mount
//! let mut file = OpenOptions::new()
//!     .write(true)
//!     .create(true)
//!     .open("/data/output.txt")?;
//! file.write_all(b"Hello, FAT!")?;
//! file.flush()?;
//!
//! // Read it back
//! let mut file = fs::open("/data/output.txt")?;
//! let content = file.read_to_vec()?;
//! ```
//!
//! ## Querying metadata
//!
//! ```ignore
//! use hyperlight_guest::fs;
//!
//! // Get file/directory info
//! let info = fs::stat("/config.json")?;
//! println!("Size: {} bytes, is_dir: {}", info.size, info.is_dir);
//!
//! // List directory contents
//! for entry in fs::read_dir("/data")? {
//!     println!("{}: {} bytes", entry.name, entry.size);
//! }
//! ```
//!
//! # Mount Resolution
//!
//! When you access a path like `/data/file.txt`, the VFS finds the mount
//! with the longest matching prefix:
//!
//! - If `/data` is a FAT mount → routes to FAT backend with relative path `file.txt`
//! - If `/` is a read-only mount → routes to RO backend with full path `/data/file.txt`
//!
//! This allows mixing read-only and read-write mounts in the same filesystem tree.
//!
//! ## Guest-Created FAT Mounts
//!
//! Guests can also create FAT filesystems dynamically at runtime:
//!
//! ```ignore
//! use hyperlight_guest::fs;
//!
//! // Create a 512KB FAT mount at /scratch
//! fs::create_fat_mount("/scratch", 512 * 1024)?;
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
//! **Note:** Guest-created mounts use heap memory and are NOT visible to the host.

mod dynamic_fat;
mod error;
mod fat;
pub(crate) mod fd;
mod file;
mod manifest;
pub mod vfs;

pub use dynamic_fat::{create_fat_mount, is_guest_created_mount, unmount};
pub use error::FsError;
pub use fat::GuestFatFile;
pub use fd::{
    FatFdEntry, FdEntry, OpenFile, SharedRoFile, alloc_fat_fd, alloc_ro_fd, dup_fd, dup_fd_to,
    free_fd, get_fat_fd, get_fd_entry, get_ro_fd, has_open_files_on_mount,
};
pub use file::{
    DirEntry, FatFile, File, OpenOptions, RawFd, RoFile, Stat, chdir, cwd, mkdir, open, read_dir,
    rename, rmdir, stat, unlink,
};
pub use manifest::{init, is_fat_region, is_initialized, vfs, vfs_mut};

#[cfg(test)]
mod tests;
