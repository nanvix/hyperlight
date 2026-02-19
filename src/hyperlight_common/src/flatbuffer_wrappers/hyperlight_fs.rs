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

//! HyperlightFS FlatBuffer wrapper types.
//!
//! This module provides Rust wrapper types for the HyperlightFS FlatBuffer schema.
//! These types are used to serialize filesystem metadata for passing between
//! host and guest.
//!
//! # Zero-Copy Design
//!
//! HyperlightFS uses memory mapping (mmap) to avoid copying file data.
//! The manifest (header + inodes) is serialized using FlatBuffers,
//! but file data is mapped directly from the host filesystem.
//!
//! **IMPORTANT:** Files must not be modified on the host while the
//! HyperlightFS image is in use. Modifying files after mapping can cause:
//! - SIGBUS crashes (if file shrinks)
//! - Undefined behavior (if file is partially updated)
//! - Missing data (if file grows, extra data not visible)
//!
//! # Sharing Across Sandboxes
//!
//! A single HyperlightFS image can be shared across multiple sandboxes.
//! The underlying mmaps share physical pages via the OS page cache,
//! so memory usage is O(1) regardless of sandbox count.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use anyhow::{Error, Result, anyhow};
use flatbuffers::FlatBufferBuilder;
#[cfg(feature = "tracing")]
use tracing::{Span, instrument};

use crate::flatbuffers::hyperlight::generated::{
    HyperlightFS as FbHyperlightFS, HyperlightFSArgs as FbHyperlightFSArgs, Inode as FbInode,
    InodeArgs as FbInodeArgs, InodeType as FbInodeType,
};

/// Current version of the HyperlightFS format.
/// Version 2 adds FatMount inode type and mount_id field.
pub const HYPERLIGHT_FS_VERSION: u16 = 2;

/// Type of an inode entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InodeType {
    /// Regular file
    File,
    /// Directory
    Directory,
    /// FAT filesystem mount
    FatMount,
}

impl From<FbInodeType> for InodeType {
    fn from(fb: FbInodeType) -> Self {
        match fb {
            FbInodeType::File => InodeType::File,
            FbInodeType::Directory => InodeType::Directory,
            FbInodeType::FatMount => InodeType::FatMount,
            // Default to File for unknown types (forward compatibility)
            _ => InodeType::File,
        }
    }
}

impl From<InodeType> for FbInodeType {
    fn from(t: InodeType) -> Self {
        match t {
            InodeType::File => FbInodeType::File,
            InodeType::Directory => FbInodeType::Directory,
            InodeType::FatMount => FbInodeType::FatMount,
        }
    }
}

/// An inode entry describing a file or directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InodeData {
    /// Type of this inode (file, directory, or FAT mount)
    pub inode_type: InodeType,
    /// Path of this entry in the guest filesystem
    pub path: String,
    /// Index of the parent directory's inode (0 for root)
    pub parent: u32,
    /// For files/FatMounts: GVA/GPA where data is mapped
    /// For directories: 0
    pub guest_address: u64,
    /// For files/FatMounts: size in bytes
    /// For directories: 0
    pub size: u64,
    /// For FatMounts: unique identifier for extraction
    /// For files/directories: 0
    pub mount_id: u32,
}

impl InodeData {
    /// Create a new file inode.
    pub fn file(path: String, parent: u32, guest_address: u64, size: u64) -> Self {
        Self {
            inode_type: InodeType::File,
            path,
            parent,
            guest_address,
            size,
            mount_id: 0,
        }
    }

    /// Create a new directory inode.
    pub fn directory(path: String, parent: u32) -> Self {
        Self {
            inode_type: InodeType::Directory,
            path,
            parent,
            guest_address: 0,
            size: 0,
            mount_id: 0,
        }
    }

    /// Create a new FAT mount inode.
    pub fn fat_mount(
        path: String,
        parent: u32,
        guest_address: u64,
        size: u64,
        mount_id: u32,
    ) -> Self {
        Self {
            inode_type: InodeType::FatMount,
            path,
            parent,
            guest_address,
            size,
            mount_id,
        }
    }

    /// Check if this inode is a file.
    pub fn is_file(&self) -> bool {
        self.inode_type == InodeType::File
    }

    /// Check if this inode is a directory (or a FAT mount point, which acts as a directory).
    pub fn is_dir(&self) -> bool {
        self.inode_type == InodeType::Directory || self.inode_type == InodeType::FatMount
    }

    /// Check if this inode is a FAT mount.
    pub fn is_fat_mount(&self) -> bool {
        self.inode_type == InodeType::FatMount
    }
}

/// HyperlightFS manifest data.
///
/// This contains the metadata for a HyperlightFS image, serialized using FlatBuffers.
/// The actual file data is not included here - it is memory-mapped separately
/// at the guest addresses specified in each inode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HyperlightFSData {
    /// Format version
    pub version: u16,
    /// Inode entries (files and directories)
    pub inodes: Vec<InodeData>,
}

impl HyperlightFSData {
    /// Create a new HyperlightFS manifest.
    pub fn new(inodes: Vec<InodeData>) -> Self {
        Self {
            version: HYPERLIGHT_FS_VERSION,
            inodes,
        }
    }
}

impl TryFrom<&[u8]> for HyperlightFSData {
    type Error = Error;

    #[cfg_attr(feature = "tracing", instrument(err(Debug), skip_all, parent = Span::current(), level = "Trace"))]
    fn try_from(raw_bytes: &[u8]) -> Result<Self> {
        let fb = flatbuffers::root::<FbHyperlightFS>(raw_bytes)
            .map_err(|e| anyhow!("Error reading HyperlightFS: {:?}", e))?;

        let version = fb.version();

        let inodes = fb
            .inodes()
            .iter()
            .map(|inode| InodeData {
                inode_type: InodeType::from(inode.inode_type()),
                path: inode.path().to_string(),
                parent: inode.parent(),
                guest_address: inode.guest_address(),
                size: inode.size(),
                mount_id: inode.mount_id(),
            })
            .collect();

        Ok(HyperlightFSData { version, inodes })
    }
}

impl TryFrom<&HyperlightFSData> for Vec<u8> {
    type Error = Error;

    #[cfg_attr(feature = "tracing", instrument(err(Debug), skip_all, parent = Span::current(), level = "Trace"))]
    fn try_from(value: &HyperlightFSData) -> Result<Vec<u8>> {
        let mut builder = FlatBufferBuilder::with_capacity(4096);

        // Create inode entries
        let inode_offsets: Vec<_> = value
            .inodes
            .iter()
            .map(|inode| {
                let path = builder.create_string(&inode.path);
                FbInode::create(
                    &mut builder,
                    &FbInodeArgs {
                        inode_type: FbInodeType::from(inode.inode_type),
                        path: Some(path),
                        parent: inode.parent,
                        guest_address: inode.guest_address,
                        size: inode.size,
                        mount_id: inode.mount_id,
                    },
                )
            })
            .collect();

        let inodes_vec = builder.create_vector(&inode_offsets);

        let fs = FbHyperlightFS::create(
            &mut builder,
            &FbHyperlightFSArgs {
                version: value.version,
                inodes: Some(inodes_vec),
            },
        );

        builder.finish(fs, None);
        Ok(builder.finished_data().to_vec())
    }
}

impl TryFrom<HyperlightFSData> for Vec<u8> {
    type Error = Error;

    #[cfg_attr(feature = "tracing", instrument(err(Debug), skip_all, parent = Span::current(), level = "Trace"))]
    fn try_from(value: HyperlightFSData) -> Result<Vec<u8>> {
        (&value).try_into()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    #[test]
    fn test_roundtrip_empty() {
        let data = HyperlightFSData::new(vec![]);
        let bytes: Vec<u8> = (&data).try_into().unwrap();
        let parsed: HyperlightFSData = bytes.as_slice().try_into().unwrap();

        assert_eq!(parsed.version, HYPERLIGHT_FS_VERSION);
        assert!(parsed.inodes.is_empty());
    }

    #[test]
    fn test_roundtrip_with_inodes() {
        let inodes = vec![
            InodeData::directory("/".to_string(), 0),
            InodeData::file("/test.txt".to_string(), 0, 0x1000_0000, 1024),
            InodeData::directory("/data".to_string(), 0),
            InodeData::file("/data/file.json".to_string(), 2, 0x1000_1000, 256),
        ];

        let data = HyperlightFSData::new(inodes.clone());
        let bytes: Vec<u8> = (&data).try_into().unwrap();
        let parsed: HyperlightFSData = bytes.as_slice().try_into().unwrap();

        assert_eq!(parsed.version, HYPERLIGHT_FS_VERSION);
        assert_eq!(parsed.inodes.len(), 4);
        assert_eq!(parsed.inodes[0].path, "/");
        assert!(parsed.inodes[0].is_dir());
        assert_eq!(parsed.inodes[1].path, "/test.txt");
        assert!(parsed.inodes[1].is_file());
        assert_eq!(parsed.inodes[1].size, 1024);
        assert_eq!(parsed.inodes[1].guest_address, 0x1000_0000);
    }

    #[test]
    fn test_inode_type_conversion() {
        assert_eq!(InodeType::from(FbInodeType::File), InodeType::File);
        assert_eq!(
            InodeType::from(FbInodeType::Directory),
            InodeType::Directory
        );
        assert_eq!(InodeType::from(FbInodeType::FatMount), InodeType::FatMount);
        assert_eq!(FbInodeType::from(InodeType::File), FbInodeType::File);
        assert_eq!(
            FbInodeType::from(InodeType::Directory),
            FbInodeType::Directory
        );
        assert_eq!(
            FbInodeType::from(InodeType::FatMount),
            FbInodeType::FatMount
        );
    }

    #[test]
    fn test_fat_mount_roundtrip() {
        let inodes = vec![
            InodeData::directory("/".to_string(), 0),
            InodeData::fat_mount("/data".to_string(), 0, 0x2000_0000, 10 * 1024 * 1024, 1),
            InodeData::file("/config.json".to_string(), 0, 0x1000_0000, 256),
        ];

        let data = HyperlightFSData::new(inodes);
        let bytes: Vec<u8> = (&data).try_into().unwrap();
        let parsed: HyperlightFSData = bytes.as_slice().try_into().unwrap();

        assert_eq!(parsed.version, HYPERLIGHT_FS_VERSION);
        assert_eq!(parsed.inodes.len(), 3);

        // Check FAT mount
        assert_eq!(parsed.inodes[1].path, "/data");
        assert!(parsed.inodes[1].is_fat_mount());
        assert_eq!(parsed.inodes[1].size, 10 * 1024 * 1024);
        assert_eq!(parsed.inodes[1].mount_id, 1);
        assert_eq!(parsed.inodes[1].guest_address, 0x2000_0000);

        // Check file still works
        assert!(parsed.inodes[2].is_file());
        assert_eq!(parsed.inodes[2].mount_id, 0);
    }
}
