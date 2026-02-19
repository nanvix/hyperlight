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

//! FAT file handle implementation.
//!
//! # Threading Model
//!
//! Hyperlight guests are **single-threaded by design**. This module assumes
//! single-threaded execution for all file operations. The `GuestFatFile` type
//! is explicitly marked `!Send + !Sync` via a `PhantomData<*const ()>` marker.
//!
//! If guest multi-threading is ever added in the future, attempting to share
//! `GuestFatFile` across threads will cause **compile-time errors**, forcing
//! a review of the concurrency model. This is intentional - file handles have
//! mutable internal state (file position, buffers) that would require
//! synchronization primitives (`Mutex`, `RwLock`) to be thread-safe.
//!
//! The single-threaded assumption allows for simpler, faster code without
//! synchronization overhead, while the `!Send + !Sync` markers ensure we don't
//! accidentally create unsafe conditions if threading is added later.

use core::fmt;
use core::marker::PhantomData;

use fatfs::{Read, Seek, SeekFrom, Write};

use super::FatFile;
use super::error::map_fatfs_error;
use crate::fs::error::FsError;

/// A file handle for a file on a FAT filesystem.
///
/// Wraps a `fatfs::File` and tracks read/write permissions.
///
/// # Thread Safety
///
/// This type is intentionally `!Send + !Sync` because:
/// 1. File handles have mutable internal state (position, buffers)
/// 2. The FAT filesystem itself is not thread-safe
/// 3. Guest code is single-threaded by design
///
/// The `PhantomData<*const ()>` marker ensures that if multi-threading is
/// ever added to guests, code that tries to share file handles across threads
/// will fail to compile, forcing explicit consideration of the threading model.
pub struct GuestFatFile<'a> {
    /// The underlying fatfs file.
    file: FatFile<'a>,
    /// Whether this file is open for reading.
    can_read: bool,
    /// Whether this file is open for writing.
    can_write: bool,
    /// Marker to make this type `!Send + !Sync`.
    ///
    /// Raw pointers are `!Send + !Sync` by default in Rust.
    /// This ensures GuestFatFile cannot be sent across threads or shared
    /// without explicit unsafe code - protecting against future threading mistakes.
    _not_send_sync: PhantomData<*const ()>,
}

impl<'a> GuestFatFile<'a> {
    /// Create a new file handle.
    ///
    /// The returned handle is `!Send + !Sync` - see struct docs for rationale.
    pub(super) fn new(file: FatFile<'a>, can_read: bool, can_write: bool) -> Self {
        Self {
            file,
            can_read,
            can_write,
            _not_send_sync: PhantomData,
        }
    }

    /// Returns true if this file is open for reading.
    #[inline]
    pub fn can_read(&self) -> bool {
        self.can_read
    }

    /// Returns true if this file is open for writing.
    #[inline]
    pub fn can_write(&self) -> bool {
        self.can_write
    }

    /// Get the current file size in bytes.
    ///
    /// # Performance
    ///
    /// This method performs 3 seek operations (current pos, end, restore).
    /// For repeated size queries, store the result rather than calling this
    /// multiple times. If you only need the size once before reading the
    /// entire file, consider using `seek(SeekFrom::End(0))` to get the size
    /// and then `seek(SeekFrom::Start(0))` to rewind (2 seeks instead of 3).
    ///
    /// # Errors
    ///
    /// Returns `FsError::IoError` if seeking fails.
    pub fn len(&mut self) -> Result<u64, FsError> {
        // Save current position
        let current = self
            .file
            .seek(SeekFrom::Current(0))
            .map_err(map_fatfs_error)?;
        // Seek to end to get size
        let size = self.file.seek(SeekFrom::End(0)).map_err(map_fatfs_error)?;
        // Restore position
        self.file
            .seek(SeekFrom::Start(current))
            .map_err(map_fatfs_error)?;
        Ok(size)
    }

    /// Returns true if the file is empty.
    ///
    /// # Errors
    ///
    /// Returns `FsError::IoError` if determining size fails.
    pub fn is_empty(&mut self) -> Result<bool, FsError> {
        Ok(self.len()? == 0)
    }

    /// Read data from the file.
    ///
    /// # Errors
    ///
    /// - `FsError::NotSupported` if file is not open for reading
    /// - `FsError::IoError` on read failure
    pub fn read(&mut self, buf: &mut [u8]) -> Result<usize, FsError> {
        if !self.can_read {
            return Err(FsError::NotSupported);
        }
        self.file.read(buf).map_err(map_fatfs_error)
    }

    /// Write data to the file.
    ///
    /// # Errors
    ///
    /// - `FsError::ReadOnly` if file is not open for writing
    /// - `FsError::NoSpace` if filesystem is full
    /// - `FsError::IoError` on write failure
    pub fn write(&mut self, buf: &[u8]) -> Result<usize, FsError> {
        if !self.can_write {
            return Err(FsError::ReadOnly);
        }
        self.file.write(buf).map_err(map_fatfs_error)
    }

    /// Seek to a position in the file.
    ///
    /// # Errors
    ///
    /// - `FsError::IoError` if seeking to invalid position
    pub fn seek(&mut self, pos: SeekFrom) -> Result<u64, FsError> {
        self.file.seek(pos).map_err(map_fatfs_error)
    }

    /// Seek to a position in the file using raw values.
    ///
    /// This is a convenience method for C API callers who don't have access
    /// to the fatfs::SeekFrom type.
    ///
    /// # Arguments
    /// * `whence` - 0 for SEEK_SET, 1 for SEEK_CUR, 2 for SEEK_END
    /// * `offset` - Offset in bytes (absolute for SET, relative for CUR/END)
    ///
    /// # Errors
    ///
    /// - `FsError::InvalidArgument` if whence is invalid
    /// - `FsError::IoError` if seeking to invalid position
    pub fn seek_raw(&mut self, whence: i32, offset: i64) -> Result<u64, FsError> {
        let pos = match whence {
            0 => {
                // SEEK_SET: offset must be non-negative
                if offset < 0 {
                    return Err(FsError::InvalidArgument);
                }
                SeekFrom::Start(offset as u64)
            }
            1 => SeekFrom::Current(offset), // SEEK_CUR
            2 => SeekFrom::End(offset),     // SEEK_END
            _ => return Err(FsError::InvalidArgument),
        };
        self.seek(pos)
    }

    /// Flush any buffered data to the filesystem.
    ///
    /// # Errors
    ///
    /// - `FsError::IoError` on flush failure
    pub fn flush(&mut self) -> Result<(), FsError> {
        self.file.flush().map_err(map_fatfs_error)
    }

    /// Truncate the file at the current position.
    ///
    /// # Errors
    ///
    /// - `FsError::ReadOnly` if file is not open for writing
    /// - `FsError::IoError` on truncate failure
    pub fn truncate(&mut self) -> Result<(), FsError> {
        if !self.can_write {
            return Err(FsError::ReadOnly);
        }
        self.file.truncate().map_err(map_fatfs_error)
    }
}

impl fmt::Debug for GuestFatFile<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GuestFatFile")
            .field("can_read", &self.can_read)
            .field("can_write", &self.can_write)
            .finish_non_exhaustive()
    }
}
