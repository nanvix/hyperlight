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

//! High-level FAT filesystem wrapper.

use core::fmt;

use fatfs::{Seek, SeekFrom};

use super::FatFs;
use super::error::map_fatfs_error;
use super::file::GuestFatFile;
use super::storage::RawMemoryStorage;
use super::time::HyperlightTimeProvider;
use crate::fs::error::FsError;

/// High-level FAT filesystem wrapper for guest operations.
///
/// Wraps a `fatfs::FileSystem` over `RawMemoryStorage` and provides a cleaner
/// API that returns `FsError` instead of fatfs error types.
///
/// # Example
///
/// ```ignore
/// use hyperlight_guest::fs::fat::GuestFat;
///
/// // Open a FAT filesystem from a memory region
/// let fat = unsafe { GuestFat::from_memory(fat_ptr, fat_size)? };
///
/// // Read a file
/// let mut file = fat.open("config.txt", true, false, false, false)?;
/// let mut buf = [0u8; 256];
/// let n = file.read(&mut buf)?;
///
/// // Create and write a file
/// let mut out = fat.open("output.txt", false, true, true, true)?;
/// out.write(b"hello world")?;
/// out.flush()?;
///
/// // List directory contents
/// for entry in fat.read_dir(".")? {
///     println!("{}: {} bytes", entry.name, entry.size);
/// }
/// ```
///
/// # Spec Reference
///
/// See spec §5.5.2 for design.
pub struct GuestFat {
    /// The underlying fatfs FileSystem.
    fs: FatFs,
}

impl GuestFat {
    /// Open an existing FAT filesystem from a memory region.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to the start of the FAT image in memory
    /// * `size` - Size of the memory region in bytes
    ///
    /// # Safety
    ///
    /// The caller must ensure the memory region is valid, properly aligned,
    /// and remains valid for the lifetime of this `GuestFat`.
    ///
    /// # Errors
    ///
    /// Returns `FsError::IoError` if the FAT image is invalid or corrupted.
    pub unsafe fn from_memory(ptr: *mut u8, size: usize) -> Result<Self, FsError> {
        // SAFETY: Caller guarantees memory region validity
        let storage = unsafe { RawMemoryStorage::new(ptr, size) };
        let options = fatfs::FsOptions::new().time_provider(HyperlightTimeProvider);
        let fs = fatfs::FileSystem::new(storage, options).map_err(map_fatfs_error)?;
        Ok(Self { fs })
    }

    /// Open a file with the specified mode.
    ///
    /// # Arguments
    ///
    /// * `path` - Path relative to the FAT root (e.g., "subdir/file.txt")
    /// * `read` - Open for reading
    /// * `write` - Open for writing
    /// * `create` - Create file if it doesn't exist
    /// * `truncate` - Truncate file to zero length
    ///
    /// # Flag Combinations
    ///
    /// | read | write | create | truncate | Behavior |
    /// |------|-------|--------|----------|----------|
    /// | true | false | false  | false    | Open existing for read |
    /// | true | true  | false  | false    | Open existing for read/write |
    /// | true | true  | true   | false    | Open or create for read/write |
    /// | true | true  | true   | true     | Open or create, truncate to 0 |
    /// | false| true  | true   | true     | Create/truncate, write-only |
    ///
    /// # Errors
    ///
    /// - `FsError::NotFound` if file doesn't exist and `create` is false
    /// - `FsError::IoError` if path refers to a directory (fatfs limitation)
    pub fn open(
        &self,
        path: &str,
        read: bool,
        write: bool,
        create: bool,
        truncate: bool,
    ) -> Result<GuestFatFile<'_>, FsError> {
        let root = self.fs.root_dir();

        // Determine if we need to create or just open
        if create {
            // Try to open first, create if not found
            match root.open_file(path) {
                Ok(mut file) => {
                    // File exists - apply truncate if requested
                    if truncate {
                        file.truncate().map_err(map_fatfs_error)?;
                    }
                    Ok(GuestFatFile::new(file, read, write))
                }
                Err(fatfs::Error::NotFound) => {
                    // Create the file
                    let file = root.create_file(path).map_err(map_fatfs_error)?;
                    Ok(GuestFatFile::new(file, read, write))
                }
                Err(e) => Err(map_fatfs_error(e)),
            }
        } else {
            // Just open existing
            let mut file = root.open_file(path).map_err(map_fatfs_error)?;
            if truncate && write {
                file.truncate().map_err(map_fatfs_error)?;
            }
            Ok(GuestFatFile::new(file, read, write))
        }
    }

    /// Create a new file, failing if it already exists.
    ///
    /// This implements O_CREAT | O_EXCL semantics - atomic "create if not exists".
    /// In the single-threaded guest environment, checking existence then creating
    /// is inherently atomic.
    ///
    /// # Errors
    ///
    /// - `FsError::AlreadyExists` if file already exists
    /// - `FsError::NotFound` if parent directory doesn't exist
    pub fn create_new(&self, path: &str) -> Result<GuestFatFile<'_>, FsError> {
        let root = self.fs.root_dir();

        // fatfs::Dir::create_file does NOT fail if file exists - it opens it!
        // We must explicitly check for existence first.
        // NOTE: In single-threaded guest, this check-then-create is atomic.
        if root.open_file(path).is_ok() {
            return Err(FsError::AlreadyExists);
        }

        let file = root.create_file(path).map_err(map_fatfs_error)?;
        Ok(GuestFatFile::new(file, true, true))
    }

    /// Get file/directory metadata.
    ///
    /// # Errors
    ///
    /// - `FsError::NotFound` if path doesn't exist
    pub fn stat(&self, path: &str) -> Result<FatStat, FsError> {
        let root = self.fs.root_dir();

        // Handle root directory specially
        if path.is_empty() || path == "/" || path == "." {
            return Ok(FatStat {
                size: 0,
                is_dir: true,
                is_readonly: false,
            });
        }

        // Try opening as file first
        if let Ok(mut file) = root.open_file(path) {
            // Get file size by seeking to end
            let size = file.seek(SeekFrom::End(0)).map_err(map_fatfs_error)?;
            return Ok(FatStat {
                size,
                is_dir: false,
                is_readonly: false, // fatfs::File doesn't expose attributes directly
            });
        }

        // Try opening as directory
        if root.open_dir(path).is_ok() {
            return Ok(FatStat {
                size: 0,
                is_dir: true,
                is_readonly: false,
            });
        }

        Err(FsError::NotFound)
    }

    /// Read directory contents.
    ///
    /// # Errors
    ///
    /// - `FsError::NotFound` if directory doesn't exist
    /// - `FsError::IoError` if path is a file (fatfs returns generic error)
    pub fn read_dir(&self, path: &str) -> Result<alloc::vec::Vec<FatDirEntry>, FsError> {
        let root = self.fs.root_dir();

        // Open the directory (root or subdirectory)
        let dir = if path.is_empty() || path == "/" || path == "." {
            root
        } else {
            root.open_dir(path).map_err(map_fatfs_error)?
        };

        let mut entries = alloc::vec::Vec::new();
        for entry in dir.iter() {
            let entry = entry.map_err(map_fatfs_error)?;
            let name = entry.file_name();

            // Skip . and .. entries
            if name == "." || name == ".." {
                continue;
            }

            entries.push(FatDirEntry {
                name,
                is_dir: entry.is_dir(),
                size: entry.len(),
            });
        }

        Ok(entries)
    }

    /// Create a directory.
    ///
    /// # Errors
    ///
    /// - `FsError::AlreadyExists` if directory already exists
    /// - `FsError::NotFound` if parent directory doesn't exist
    pub fn mkdir(&self, path: &str) -> Result<(), FsError> {
        let root = self.fs.root_dir();
        root.create_dir(path).map_err(map_fatfs_error)?;
        Ok(())
    }

    /// Remove an empty directory.
    ///
    /// # Errors
    ///
    /// - `FsError::NotFound` if directory doesn't exist
    /// - `FsError::NotEmpty` if directory is not empty
    /// - `FsError::NotADirectory` if path is a file
    pub fn rmdir(&self, path: &str) -> Result<(), FsError> {
        let root = self.fs.root_dir();

        // Verify it's a directory, not a file
        if root.open_file(path).is_ok() {
            return Err(FsError::NotADirectory);
        }

        // Verify directory exists before removing
        root.open_dir(path).map_err(map_fatfs_error)?;

        root.remove(path).map_err(map_fatfs_error)
    }

    /// Delete a file.
    ///
    /// # Errors
    ///
    /// - `FsError::NotFound` if file doesn't exist
    /// - `FsError::NotAFile` if path is a directory
    pub fn unlink(&self, path: &str) -> Result<(), FsError> {
        let root = self.fs.root_dir();

        // Verify it's a file, not a directory
        if root.open_dir(path).is_ok() {
            return Err(FsError::NotAFile);
        }

        // Verify file exists before removing
        root.open_file(path).map_err(map_fatfs_error)?;

        root.remove(path).map_err(map_fatfs_error)
    }

    /// Rename/move a file or directory.
    ///
    /// # Errors
    ///
    /// - `FsError::NotFound` if source doesn't exist
    /// - `FsError::AlreadyExists` if destination already exists
    pub fn rename(&self, old_path: &str, new_path: &str) -> Result<(), FsError> {
        let root = self.fs.root_dir();
        root.rename(old_path, &root, new_path)
            .map_err(map_fatfs_error)
    }
}

impl fmt::Debug for GuestFat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GuestFat").finish_non_exhaustive()
    }
}

/// File metadata from a FAT filesystem.
///
/// Returned by [`GuestFat::stat()`] to describe a file or directory.
///
/// # Limitations
///
/// The `is_readonly` field is always `false` because the underlying fatfs
/// `File` type does not expose FAT attributes. This may be improved in
/// future versions.
#[derive(Debug, Clone, Copy)]
pub struct FatStat {
    /// File size in bytes (0 for directories).
    pub size: u64,
    /// True if this is a directory.
    pub is_dir: bool,
    /// True if the read-only attribute is set.
    ///
    /// **Note:** Currently always `false` due to fatfs API limitations.
    pub is_readonly: bool,
}

/// Directory entry from a FAT filesystem.
///
/// Returned by [`GuestFat::read_dir()`] for each file or subdirectory.
/// Does not include `.` or `..` pseudo-entries.
#[derive(Debug, Clone)]
pub struct FatDirEntry {
    /// Entry name (filename only, not full path).
    pub name: alloc::string::String,
    /// True if this is a directory.
    pub is_dir: bool,
    /// Size in bytes (0 for directories).
    pub size: u64,
}
