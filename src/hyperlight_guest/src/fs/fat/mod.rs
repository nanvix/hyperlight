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

//! FAT filesystem backend for guest memory regions.
//!
//! This module provides:
//! - [`RawMemoryStorage`]: Low-level adapter implementing fatfs I/O traits over raw memory
//! - [`HyperlightTimeProvider`]: TimeProvider returning fixed 1980-01-01 timestamp
//! - [`GuestFat`]: High-level FAT filesystem wrapper for guest operations
//! - [`GuestFatFile`]: File handle for FAT files
//!
//! # Purpose
//!
//! This is the FAT backend for the guest VFS. Given a pointer to a memory region
//! containing a FAT image, these types allow reading and writing files.
//!
//! # Safety
//!
//! The memory region must remain valid for the lifetime of the storage.
//! The caller is responsible for ensuring:
//! - The memory region is properly mapped and accessible
//! - No concurrent access without synchronization
//! - The region is not unmapped while the storage is in use

mod error;
mod file;
mod filesystem;
mod storage;
mod time;

pub use file::GuestFatFile;
pub(crate) use filesystem::GuestFat;
pub(crate) use storage::RawMemoryStorage;
pub(crate) use time::HyperlightTimeProvider;

/// Type alias for the FAT filesystem with our storage and time provider.
pub(crate) type FatFs =
    fatfs::FileSystem<RawMemoryStorage, HyperlightTimeProvider, fatfs::LossyOemCpConverter>;

/// Type alias for a FAT file handle.
pub(crate) type FatFile<'a> =
    fatfs::File<'a, RawMemoryStorage, HyperlightTimeProvider, fatfs::LossyOemCpConverter>;
