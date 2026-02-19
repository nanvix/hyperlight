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

//! HyperlightFS: Zero-copy filesystem passthrough for Hyperlight sandboxes.
//!
//! This module provides a way to map host files into guest virtual machines
//! without copying the file data. Files are memory-mapped (mmap) on the host
//! and the mappings are shared with the guest.
//!
//! # Design Overview
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │ HyperlightFS Image                                                  │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │ Manifest (FlatBuffer-serialized)                                    │
//! │   - Version                                                         │
//! │   - Inodes: [{type, path, parent, guest_address, size}, ...]        │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │ Mapped Files Region (NOT stored in manifest!)                       │
//! │   - Each file is mmap'd separately from the host filesystem         │
//! │   - Data is accessed via memory-mapped pointers                     │
//! │   - Zero-copy: no data is copied, just pointer arithmetic           │
//! │   - guest_address in manifest points into this region               │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Zero-Copy Design
//!
//! HyperlightFS uses memory mapping (`mmap` with `MAP_PRIVATE | PROT_READ`)
//! to avoid copying file data. This is critical for performance with large
//! files (MB to GB scale).
//!
//! The manifest (metadata) is a small FlatBuffer that gets serialized once.
//! File contents are mapped directly from the host filesystem using `mmap`.
//! Multiple sandboxes can share the same mappings through the OS page cache.
//!
//! # Sharing Across Sandboxes
//!
//! A single [`HyperlightFSImage`] can be shared across multiple sandboxes.
//! The underlying mmaps share physical pages via the OS page cache,
//! so memory usage is O(1) regardless of how many sandboxes use the image.
//!
//! # ⚠️ IMPORTANT: File Modification Warning
//!
//! **Files must NOT be modified on the host while a HyperlightFS image is in use.**
//!
//! Modifying files after mapping can cause:
//! - **SIGBUS crashes** if the file shrinks (accessing pages beyond EOF)
//! - **Undefined behavior** if the file is partially updated
//! - **Missing data** if the file grows (extra data is not visible)
//!
//! This is a fundamental limitation of memory-mapped I/O, not specific to
//! HyperlightFS. The caller is responsible for ensuring files are stable
//! during the lifetime of the image.
//!
//! # Usage
//!
//! ```no_run
//! # use hyperlight_host::hyperlight_fs::HyperlightFSBuilder;
//! # fn main() -> hyperlight_host::Result<()> {
//! // Create a builder (empty by default - explicit opt-in)
//! let builder = HyperlightFSBuilder::new();
//!
//! // Add individual files
//! let builder = builder.add_file("/host/path/to/file.txt", "/guest/file.txt")?;
//!
//! // Add a directory with pattern matching
//! let builder = builder
//!     .add_dir("/host/data", "/guest/data")?
//!     .include("**/*.json")
//!     .include("**/*.txt")
//!     .exclude("**/secret/*")
//!     .done()?;
//!
//! // Preview what would be mapped (dry run)
//! let manifest = builder.file_summary()?;
//! println!("Would map {} files", manifest.files.len());
//!
//! // Build the image (creates mmaps)
//! let image = builder.build()?;
//! # Ok(())
//! # }
//! ```
//!
//! # Platform Support
//!
//! - **Linux**: Full support using `mmap(2)`
//! - **Windows**: Not yet supported (returns error)

mod builder;
mod config;
mod fat_image;
mod image;

pub use builder::{BuildManifest, DirectoryBuilder, HyperlightFSBuilder, ManifestEntry};
pub use config::{
    ConfigError, DirectoryMapping, FileMapping as ConfigFileMapping, HyperlightFsConfig,
};
pub use fat_image::{
    FatEntry, FatFileReader, FatFileWriter, FatImage, FatStat, MAX_FAT_IMAGE_SIZE,
    MIN_FAT_IMAGE_SIZE,
};
pub use image::{FileMapping, HyperlightFSImage};
