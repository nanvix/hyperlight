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

//! FAT filesystem image management for HyperlightFS.
//!
//! This module provides [`FatImage`], a wrapper around FAT filesystem images
//! that handles:
//! - Exclusive file locking to prevent concurrent access
//! - Memory mapping with `MAP_SHARED` for write persistence
//! - FAT filesystem formatting for new images (FAT12/16/32 auto-selected by size)
//! - Automatic cleanup of temporary files
//!
//! # Zero-Copy Architecture
//!
//! FAT images use `MAP_SHARED` memory mapping, which means:
//! - Guest writes go directly to the kernel's page cache
//! - The same physical pages are shared between host and guest
//! - Durability is ensured by `msync(MS_SYNC)` called during sandbox HLT handling
//!
//! # Exclusive Locking
//!
//! Each FAT image is exclusively locked using `flock(LOCK_EX)` to prevent
//! multiple sandboxes from concurrently accessing the same backing file.
//!
//! # Example
//!
//! ```no_run
//! # use hyperlight_host::hyperlight_fs::FatImage;
//! # use std::io::{Read, Write};
//! # fn main() -> hyperlight_host::Result<()> {
//! // Create a new temporary FAT image (1MB)
//! let mut image = FatImage::create_temp(1024 * 1024)?;
//!
//! // Get the mmap'd region for guest memory setup
//! let ptr = image.as_ptr();
//! let size = image.size();
//!
//! let mut writer = image.create_file("/hello.txt")?;
//! writer.write_all(b"Hello, 1985!")?;
//! drop(writer); // Release borrow
//!
//! let mut reader = image.open_file("/hello.txt")?;
//! let mut contents = Vec::new();
//! reader.read_to_end(&mut contents)?;
//! # Ok(())
//! # }
//! ```

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use chrono::NaiveDateTime;
use fs2::FileExt;
use tracing::{debug, error, info, trace, warn};

use crate::Result;
use crate::error::HyperlightError;

/// Type alias for the I/O wrapper used with fatfs 0.4.0.
///
/// fatfs 0.4 uses custom Read/Write/Seek traits instead of std::io traits.
/// StdIoWrapper adapts std::io types to fatfs's traits.
type FatIo = fatfs::StdIoWrapper<std::io::Cursor<&'static mut [u8]>>;

/// Type alias for the filesystem type with default time provider and OEM converter.
type FatFs = fatfs::FileSystem<FatIo, fatfs::DefaultTimeProvider, fatfs::LossyOemCpConverter>;

/// Type alias for a file within our FAT filesystem.
type FatFile<'a> = fatfs::File<'a, FatIo, fatfs::DefaultTimeProvider, fatfs::LossyOemCpConverter>;

/// A file handle for reading from a FAT image.
///
/// Provides streaming access to file contents without loading the entire file
/// into memory.
///
/// This type implements [`Read`] and [`Seek`], so you can use it with standard
/// Rust I/O idioms like `BufReader`, `io::copy`, etc.
///
/// # Lifetime
///
/// The `FatFileReader` borrows from the [`FatImage`] that created it. You must
/// drop the reader before you can do other operations on the image.
///
/// # Example
///
/// ```no_run
/// # use hyperlight_host::hyperlight_fs::FatImage;
/// # use std::io::{BufRead, BufReader};
/// # fn main() -> hyperlight_host::Result<()> {
/// # let mut image = FatImage::create_temp(1024 * 1024)?;
/// let reader = image.open_file("/config.txt")?;
/// let buf_reader = BufReader::new(reader);
/// for line in buf_reader.lines() {
///     println!("{}", line?);
/// }
/// # Ok(())
/// # }
/// ```
pub struct FatFileReader<'a> {
    file: FatFile<'a>,
    /// Path to the file, used in `Debug` output.
    path: String,
}

impl<'a> Read for FatFileReader<'a> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.file.read(buf)
    }
}

impl<'a> Seek for FatFileReader<'a> {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.file.seek(pos)
    }
}

impl std::fmt::Debug for FatFileReader<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FatFileReader")
            .field("path", &self.path)
            .finish()
    }
}

/// A file handle for writing to a FAT image.
///
/// Provides streaming write access to files without buffering the entire
/// contents in memory.
///
/// This type implements [`Write`] and [`Seek`], so you can use it with standard
/// Rust I/O idioms like `BufWriter`, `io::copy`, etc.
///
/// # Lifetime
///
/// The `FatFileWriter` borrows from the [`FatImage`] that created it. You must
/// drop the writer before you can do other operations on the image.
///
/// # Flush on Drop
///
/// The writer automatically flushes on drop to ensure data is persisted to the
/// mmap'd region. However, for error handling you should call `flush()` explicitly.
///
/// # Example
///
/// ```no_run
/// # use hyperlight_host::hyperlight_fs::FatImage;
/// # use std::io::Write;
/// # fn main() -> hyperlight_host::Result<()> {
/// # let mut image = FatImage::create_temp(1024 * 1024)?;
/// let mut writer = image.create_file("/output.bin")?;
/// writer.write_all(b"Hello, world!")?;
/// writer.flush()?; // Explicit flush for error handling
/// drop(writer);    // Or let it drop, which also flushes
/// # Ok(())
/// # }
/// ```
#[must_use = "writer must be used to write data; dropping immediately writes nothing"]
pub struct FatFileWriter<'a> {
    file: FatFile<'a>,
    /// Path to the file, used in `Debug` output and error logging on drop.
    path: String,
}

impl<'a> Write for FatFileWriter<'a> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.file.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}

impl<'a> Seek for FatFileWriter<'a> {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.file.seek(pos)
    }
}

impl Drop for FatFileWriter<'_> {
    fn drop(&mut self) {
        // Best-effort flush on drop - can't return errors here
        if let Err(e) = self.file.flush() {
            error!(path = %self.path, error = %e, "Failed to flush file on drop");
        }
    }
}

impl std::fmt::Debug for FatFileWriter<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FatFileWriter")
            .field("path", &self.path)
            .finish()
    }
}

/// An entry from a FAT directory listing, representing either a file or subdirectory.
///
/// Returned by [`FatImage::read_dir`].
///
/// # Snapshot Semantics
///
/// This struct represents a **point-in-time snapshot** of the directory entry.
/// If you modify the file (e.g., via [`FatImage::create_file`]) after calling
/// [`FatImage::read_dir`], the metadata in `stat` will be stale. Use
/// [`FatImage::stat`] to get current metadata if needed.
///
/// # Example
///
/// ```no_run
/// # use hyperlight_host::hyperlight_fs::FatImage;
/// # fn main() -> hyperlight_host::Result<()> {
/// # let mut image = FatImage::create_temp(1024 * 1024)?;
/// let entries = image.read_dir("/")?;
/// for entry in entries {
///     if entry.stat.is_dir {
///         println!("Directory: {}", entry.name);
///     } else {
///         println!("File: {} ({} bytes)", entry.name, entry.stat.size);
///     }
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct FatEntry {
    /// Name of the file or directory.
    pub name: String,
    /// Metadata (size, type, timestamps) at the time of the directory listing.
    ///
    /// **Note**: This is a snapshot. If the file is modified after
    /// `read_dir()` is called, these values will be stale.
    pub stat: FatStat,
}

/// Metadata for a file or directory returned by [`FatImage::stat`].
///
/// # FAT Timestamp Limitations
///
/// These are limitations of the FAT filesystem format itself, not our implementation:
/// - **Created/Modified**: 2-second resolution (seconds are always even numbers)
/// - **Accessed**: Date only - FAT stores no time component for last access
///
/// For `accessed`, we return midnight (00:00:00) as the time since FAT doesn't
/// provide one.
#[derive(Debug, Clone, PartialEq)]
pub struct FatStat {
    /// Size in bytes (0 for directories).
    pub size: u64,
    /// Whether this is a directory.
    pub is_dir: bool,
    /// Creation timestamp (2-second resolution).
    pub created: Option<NaiveDateTime>,
    /// Last modification timestamp (2-second resolution).
    pub modified: Option<NaiveDateTime>,
    /// Last access date. Time component is always midnight (00:00:00) because
    /// FAT only stores the date, not the time, for last access.
    pub accessed: Option<NaiveDateTime>,
}

// FatImage requires 64-bit pointers for mmap of large files
#[cfg(not(target_pointer_width = "64"))]
compile_error!("FatImage requires a 64-bit target");

/// Minimum FAT image size: 1 MiB (1,048,576 bytes).
///
/// The `fatfs` crate auto-selects the appropriate FAT variant based on size:
/// - FAT12: < ~4085 clusters (small volumes up to ~16MB)
/// - FAT16: 4085-65524 clusters (medium volumes, 16MB-2GB)
/// - FAT32: >= 65525 clusters (large volumes, requires ~33MB+)
///
/// At 1 MiB, FAT12 or FAT16 will be used. This is sufficient for typical
/// Hyperlight use cases (config files, temp data). Note that FAT12/16 have
/// a fixed root directory limit (~224-512 entries), but this is unlikely
/// to be hit in normal sandbox usage.
pub const MIN_FAT_IMAGE_SIZE: usize = 1_024 * 1_024;

/// Maximum FAT image size: 16 GiB (17,179,869,184 bytes).
///
/// This is a practical limit to avoid excessive mmap usage. FAT itself
/// supports much larger volumes (FAT32 up to ~2TB), but we cap it here for sanity.
///
/// Note: The 4GB-1 limit often cited is for individual *files within* a FAT
/// filesystem, not the volume size.
pub const MAX_FAT_IMAGE_SIZE: usize = 16 * 1_024 * 1_024 * 1_024;

/// A FAT filesystem image backed by an mmap'd host file.
///
/// The backing file is exclusively locked (`flock`) to prevent multiple
/// sandboxes from mapping the same file concurrently. The memory mapping
/// uses `MAP_SHARED` so writes persist to the backing file automatically.
///
/// # Lifecycle
///
/// 1. **Creation**: `create_temp()` or `create_at()` creates a new file,
///    formats it as FAT, and acquires an exclusive lock.
/// 2. **Opening**: `open()` opens an existing file and acquires an exclusive lock.
/// 3. **Usage**: `as_ptr()` and `size()` provide access for guest memory mapping.
/// 4. **Cleanup**: On `drop()`, the lock is released. For temp files, the file is deleted.
///
/// # Locking Model
///
/// This type uses `flock(LOCK_EX | LOCK_NB)` for exclusive access. This is an
/// **advisory** lock, meaning:
///
/// - It prevents multiple Hyperlight sandboxes from accessing the same file
/// - It works across processes (multi-process protection)
/// - It does NOT prevent external processes that don't use `flock()` from accessing the file
///
/// External tools that need to access FAT files while Hyperlight might use them should
/// also use `flock()` to cooperate with this locking scheme. See the HyperlightFS
/// specification (section 3.6) for full details on the advisory locking model.
///
/// # Internal Design: The `'static` Lifetime Lie
///
/// This struct stores a `FileSystem<Cursor<&'static mut [u8]>>` internally to avoid
/// recreating the filesystem on every operation. The `'static` lifetime is a **lie** -
/// the slice actually points to `mmap_ptr`, which is only valid until `drop()` calls
/// `munmap`.
///
/// ## Why this is safe
///
/// 1. **Drop ordering**: `Drop::drop()` sets `self.fs = None` *before* calling `munmap`,
///    ensuring the `FileSystem` is dropped while the memory is still valid.
///
/// 2. **No `mem::forget`**: If someone calls `mem::forget(fat_image)`, the `FileSystem`
///    would hold a dangling pointer. This is a fundamental limitation - `mem::forget`
///    is safe in Rust, but can break invariants like this. Don't do it.
///
/// 3. **Moving is safe**: Moving `FatImage` doesn't invalidate `mmap_ptr` because it
///    points to kernel-managed memory (the mmap region), not data inside the struct.
///
/// ## Why we can't use a correct lifetime
///
/// This is a self-referential struct: `fs` borrows from `mmap_ptr`, but both are fields
/// of the same struct. Rust's borrow checker cannot express "this field borrows from
/// that field" because lifetimes must come from *outside* the struct.
///
/// Alternatives considered:
/// - `FileSystem<Cursor<&'a mut [u8]>>` with `FatImage<'a>` - doesn't work, the slice
///   comes from inside the struct, not from an external source
/// - Recreate `FileSystem` on every operation - works but has overhead (re-parses FAT)
/// - `ouroboros`/`self_cell` crates - use the same `'static` trick, just wrapped in macros
#[cfg(unix)]
pub struct FatImage {
    /// The backing file (holds the exclusive lock).
    /// Kept for its Drop impl which releases the flock.
    _file: File,
    /// Memory-mapped region (MAP_SHARED for write persistence).
    mmap_ptr: *mut u8,
    /// Size of the mapped region.
    mmap_size: usize,
    /// Path to the backing file (for cleanup).
    path: PathBuf,
    /// Whether this is a temp file (delete on drop).
    is_temp: bool,
    /// Cached FAT filesystem handle.
    ///
    /// # Safety
    ///
    /// The `'static` lifetime is a lie. See struct-level documentation.
    /// This field **must** be set to `None` before `munmap` in `Drop::drop()`.
    fs: Option<FatFs>,
}

// Manual Debug impl because fatfs::FileSystem doesn't implement Debug
#[cfg(unix)]
impl std::fmt::Debug for FatImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FatImage")
            .field("mmap_ptr", &self.mmap_ptr)
            .field("mmap_size", &self.mmap_size)
            .field("path", &self.path)
            .field("is_temp", &self.is_temp)
            .field("fs", &self.fs.as_ref().map(|_| "<FileSystem>"))
            .finish()
    }
}

// SAFETY: FatImage is Send because the mmap'd region is process-local
// and the file lock ensures exclusive access.
//
// FatImage is intentionally NOT Sync. While the raw pointer could technically
// support shared reads, FatImage is designed to be owned by a single Sandbox,
// and Sandbox itself is not Sync.
#[cfg(unix)]
unsafe impl Send for FatImage {}

#[cfg(unix)]
impl FatImage {
    /// Open an existing FAT image from a file with exclusive lock.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to an existing FAT image file
    ///
    /// # Returns
    ///
    /// A `FatImage` with an exclusive lock on the file.
    ///
    /// # Errors
    ///
    /// Returns `HyperlightError::Error` if the file doesn't exist, can't be opened,
    /// is not a valid FAT image, or another process holds the lock.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        info!(path = %path.display(), "Opening FAT image");

        // Open file for read/write
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| {
                error!(path = %path.display(), error = %e, "Failed to open FAT image");
                HyperlightError::Error(format!("Failed to open FAT image {:?}: {}", path, e))
            })?;

        // Try to acquire exclusive lock (non-blocking)
        file.try_lock_exclusive().map_err(|e| {
            error!(path = %path.display(), error = %e, "FAT image is locked by another process");
            HyperlightError::Error(format!(
                "File is locked by another process {:?}: {}",
                path, e
            ))
        })?;

        // Get file size
        let metadata = file.metadata().map_err(|e| {
            error!(path = %path.display(), error = %e, "Failed to get FAT image metadata");
            HyperlightError::Error(format!("Failed to get metadata for {:?}: {}", path, e))
        })?;
        let size = metadata.len() as usize;

        Self::validate_size(size)?;

        // Memory map the file with MAP_SHARED
        let mmap_ptr = Self::mmap_file(&file, size)?;

        // Validate it's actually a FAT filesystem
        Self::validate_fat_image(mmap_ptr, size).inspect_err(|_| {
            // Unmap on validation failure. The file lock is released when `file`
            // is dropped at function exit (implicit via the returned Err).
            unsafe { libc::munmap(mmap_ptr as *mut libc::c_void, size) };
        })?;

        info!(path = %path.display(), size, "FAT image opened and locked");

        Ok(Self {
            _file: file,
            mmap_ptr,
            mmap_size: size,
            path,
            is_temp: false,
            fs: None,
        })
    }

    /// Create a new empty FAT image at the specified path.
    ///
    /// Creates the file, extends it to the specified size (sparse if supported),
    /// formats it as FAT, and acquires an exclusive lock.
    ///
    /// # Arguments
    ///
    /// * `path` - Path where the FAT image will be created
    /// * `size_bytes` - Size of the image (must be between 1MB and 16GB)
    ///
    /// # Errors
    ///
    /// - `HyperlightError::Error` if size is out of range or if FAT formatting fails
    /// - `HyperlightError::IOError` if file creation fails
    pub fn create_at<P: AsRef<Path>>(path: P, size_bytes: usize) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        Self::validate_size(size_bytes)?;

        info!(path = %path.display(), size = size_bytes, "Creating FAT image");

        // Create file (fails if it already exists)
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|e| {
                error!(path = %path.display(), error = %e, "Failed to create FAT image");
                HyperlightError::Error(format!("Failed to create FAT image {:?}: {}", path, e))
            })?;

        // Acquire exclusive lock (shouldn't fail on newly created file, but if it does we should not try and delete it since someone else may be using it)
        file.try_lock_exclusive().map_err(|e| {
            error!(path = %path.display(), error = %e, "Failed to lock newly created FAT image");
            HyperlightError::Error(format!(
                "Failed to lock newly created file {:?}: {}",
                path, e
            ))
        })?;

        Self::initialize_image(file, path, size_bytes, false)
    }

    /// Create a new empty FAT image as a temporary file.
    ///
    /// The temp file will be deleted when the `FatImage` is dropped.
    ///
    /// # Arguments
    ///
    /// * `size_bytes` - Size of the image (must be between 1MB and 16GB)
    ///
    /// # Errors
    ///
    /// - `HyperlightError::Error` if size is out of range
    /// - `HyperlightError::IOError` if temp file creation fails
    pub fn create_temp(size_bytes: usize) -> Result<Self> {
        Self::validate_size(size_bytes)?;

        // Create temp file and persist it (so path remains valid)
        let temp_file = tempfile::NamedTempFile::new().map_err(|e| {
            error!(error = %e, "Failed to create temp file for FAT image");
            HyperlightError::Error(format!("Failed to create temp file: {}", e))
        })?;

        let path = temp_file.path().to_path_buf();

        info!(path = %path.display(), size = size_bytes, "Creating temporary FAT image");

        // Why persist() instead of letting NamedTempFile auto-delete?
        //
        // We need the File handle to live in FatImage._file (for flock lifetime),
        // but NamedTempFile would delete the file when dropped at function end.
        // Alternative: make _file an enum or generic to hold NamedTempFile directly,
        // which would auto-delete on drop (no need for is_temp + manual delete).
        // Tradeoff: enum/generic adds type complexity, bool + manual delete is more explicit.
        let file = temp_file.persist(&path).map_err(|e| {
            error!(path = %path.display(), error = %e, "Failed to persist temp file");
            HyperlightError::Error(format!("Failed to persist temp file: {}", e))
        })?;

        // Acquire exclusive lock (shouldn't fail on newly created file, but if it does, leave file for debugging)
        file.try_lock_exclusive().map_err(|e| {
            error!(path = %path.display(), error = %e, "Failed to lock temp FAT image");
            HyperlightError::Error(format!("Failed to lock temp file: {}", e))
        })?;

        Self::initialize_image(file, path, size_bytes, true)
    }

    /// Get a pointer to the mmap'd region.
    ///
    /// This pointer can be used to map the FAT image into guest memory.
    ///
    /// # Safety Note
    ///
    /// The returned pointer is valid only for the lifetime of this `FatImage`.
    /// After the `FatImage` is dropped, the pointer becomes invalid and must
    /// not be dereferenced. The caller is responsible for ensuring the pointer
    /// is not used after the `FatImage` is dropped.
    ///
    /// The pointer points to `size()` bytes of memory.
    pub fn as_ptr(&self) -> *const u8 {
        self.mmap_ptr as *const u8
    }

    /// Get a mutable pointer to the mmap'd region.
    ///
    /// This pointer can be used to map the FAT image into guest memory
    /// with write access.
    ///
    /// # Safety Note
    ///
    /// The returned pointer is valid only for the lifetime of this `FatImage`.
    /// After the `FatImage` is dropped, the pointer becomes invalid and must
    /// not be dereferenced. The caller is responsible for ensuring the pointer
    /// is not used after the `FatImage` is dropped.
    ///
    /// The pointer points to `size()` bytes of memory. Writes through this
    /// pointer will be persisted to the backing file via the `MAP_SHARED` mapping.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.mmap_ptr
    }

    /// Get the raw file descriptor for the backing file.
    ///
    /// This is used by the sandbox to create a `MAP_PRIVATE` CoW mapping
    /// of the FAT image for MSHV v4 compatibility (which requires writable
    /// host pages for `pin_user_pages_fast`).
    pub(crate) fn as_raw_fd(&self) -> std::os::unix::io::RawFd {
        use std::os::unix::io::AsRawFd;
        self._file.as_raw_fd()
    }

    /// Get the size of the FAT image in bytes.
    ///
    /// This is the total size of the mmap'd region, which equals the size
    /// of the backing file.
    pub fn size(&self) -> usize {
        self.mmap_size
    }

    /// Get the path to the backing file.
    ///
    /// For temporary images created with [`create_temp`](Self::create_temp),
    /// this will be a path in the system's temporary directory.
    ///
    /// # Warning
    ///
    /// For temporary images, the file at this path will be deleted when the
    /// `FatImage` is dropped. Do not retain and use this path after dropping
    /// the `FatImage`.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Check if this is a temporary file (will be deleted on drop).
    ///
    /// Returns `true` for images created with [`create_temp`](Self::create_temp),
    /// `false` for images created with [`create_at`](Self::create_at) or
    /// opened with [`open`](Self::open).
    pub fn is_temp(&self) -> bool {
        self.is_temp
    }

    /// Synchronously flush dirty pages to the backing file.
    ///
    /// This calls `msync(MS_SYNC)` on the mmap'd region, ensuring all writes
    /// made by the guest are durably persisted to the backing file before
    /// returning.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the sync succeeded, or an error if `msync` failed.
    ///
    /// # Notes
    ///
    /// - This is called automatically by the sandbox on successful HLT
    /// - The kernel tracks dirty pages; calling this on clean pages is fast
    /// - For temporary images (`is_temp() == true`), this is a no-op since
    ///   the file will be deleted on drop anyway
    pub fn msync(&self) -> Result<()> {
        // Skip sync for temp files - they're deleted on drop anyway
        if self.is_temp {
            trace!(path = %self.path.display(), "Skipping msync for temp FAT image");
            return Ok(());
        }

        trace!(
            path = %self.path.display(),
            size = self.mmap_size,
            "Calling msync(MS_SYNC) on FAT image"
        );

        // SAFETY: mmap_ptr and mmap_size are valid from construction and remain
        // valid until Drop. MS_SYNC is a safe flag that just requests synchronous
        // write-back of dirty pages.
        let result = unsafe {
            libc::msync(
                self.mmap_ptr as *mut libc::c_void,
                self.mmap_size,
                libc::MS_SYNC,
            )
        };

        if result != 0 {
            let err = std::io::Error::last_os_error();
            error!(
                path = %self.path.display(),
                error = %err,
                "msync(MS_SYNC) failed on FAT image"
            );
            return Err(HyperlightError::Error(format!(
                "msync failed on FAT image {:?}: {}",
                self.path, err
            )));
        }

        debug!(path = %self.path.display(), "FAT image synced to disk");
        Ok(())
    }

    // ---- File Operations ----

    /// List the contents of a directory.
    ///
    /// # Arguments
    ///
    /// * `path` - Absolute path to the directory (e.g., "/" or "/subdir")
    ///
    /// # Returns
    ///
    /// A vector of [`FatEntry`] structs, one for each file or subdirectory.
    /// Does not include "." or ".." entries.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use hyperlight_host::hyperlight_fs::FatImage;
    /// # fn main() -> hyperlight_host::Result<()> {
    /// # let mut image = FatImage::create_temp(1024 * 1024)?;
    /// let entries = image.read_dir("/")?;
    /// for entry in &entries {
    ///     if entry.stat.is_dir {
    ///         println!("Directory: {}", entry.name);
    ///     } else {
    ///         println!("File: {} ({} bytes)", entry.name, entry.stat.size);
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the path doesn't exist, is not a directory,
    /// or contains invalid path components (e.g., "..").
    pub fn read_dir(&mut self, path: &str) -> Result<Vec<FatEntry>> {
        Self::validate_path(path)?;
        trace!(path, "Reading directory");

        let raw_entries = self.list_dir_entries(path)?;
        let entries: Vec<FatEntry> = raw_entries
            .into_iter()
            .map(|(name, stat)| FatEntry { name, stat })
            .collect();

        trace!(path, count = entries.len(), "Directory read complete");
        Ok(entries)
    }

    /// Open a file for reading.
    ///
    /// Returns a [`FatFileReader`] that implements [`Read`] and [`Seek`],
    /// providing streaming access to the file contents.
    ///
    /// # Arguments
    ///
    /// * `path` - Absolute path to the file (e.g., "/file.txt")
    ///
    /// # Returns
    ///
    /// A [`FatFileReader`] that borrows from this `FatImage`. The reader implements
    /// standard Rust I/O traits, so you can use it with `BufReader`, `io::copy`, etc.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use hyperlight_host::hyperlight_fs::FatImage;
    /// use std::io::{BufRead, BufReader, Read, Write};
    /// # fn main() -> hyperlight_host::Result<()> {
    /// # let mut image = FatImage::create_temp(1024 * 1024)?;
    /// # image.create_file("/small.txt")?.write_all(b"test")?;
    ///
    /// // Read entire small file
    /// let mut reader = image.open_file("/small.txt")?;
    /// let mut contents = String::new();
    /// reader.read_to_string(&mut contents)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the path doesn't exist, is a directory, or contains
    /// invalid path components.
    pub fn open_file(&mut self, path: &str) -> Result<FatFileReader<'_>> {
        Self::validate_path(path)?;
        trace!(path, "Opening file for reading");

        let fs = self.open_fs()?;
        let root = fs.root_dir();

        let file = root.open_file(path).map_err(|e| {
            error!(path, error = %e, "Failed to open file for reading");
            HyperlightError::Error(format!("Failed to open file '{}': {}", path, e))
        })?;

        trace!(path, "File opened for reading");
        Ok(FatFileReader {
            file,
            path: path.to_string(),
        })
    }

    /// Create or truncate a file for writing.
    ///
    /// Returns a [`FatFileWriter`] that implements [`Write`] and [`Seek`],
    /// allowing you to stream data to the file without buffering everything
    /// in memory.
    ///
    /// If the file already exists, it will be **truncated** (emptied) before
    /// writing. Parent directories must already exist.
    ///
    /// # Arguments
    ///
    /// * `path` - Absolute path to the file (e.g., "/file.txt")
    ///
    /// # Returns
    ///
    /// A [`FatFileWriter`] that borrows from this `FatImage`. The writer implements
    /// standard Rust I/O traits and flushes automatically on drop.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use hyperlight_host::hyperlight_fs::FatImage;
    /// use std::io::Write;
    /// # fn main() -> hyperlight_host::Result<()> {
    /// # let mut image = FatImage::create_temp(1024 * 1024)?;
    ///
    /// // Simple write
    /// let mut writer = image.create_file("/hello.txt")?;
    /// writer.write_all(b"Hello, world!")?;
    /// writer.flush()?; // Explicit flush for error handling
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the path refers to a directory, the parent directory
    /// doesn't exist, or there's insufficient space in the FAT image.
    pub fn create_file(&mut self, path: &str) -> Result<FatFileWriter<'_>> {
        Self::validate_path(path)?;
        trace!(path, "Creating file for writing");

        let fs = self.open_fs()?;
        let root = fs.root_dir();

        let mut file = root.create_file(path).map_err(|e| {
            error!(path, error = %e, "Failed to create file for writing");
            HyperlightError::Error(format!("Failed to create file '{}': {}", path, e))
        })?;

        // Truncate to 0 in case file existed
        file.truncate().map_err(|e| {
            error!(path, error = %e, "Failed to truncate file");
            HyperlightError::Error(format!("Failed to truncate file '{}': {}", path, e))
        })?;

        trace!(path, "File created and truncated for writing");
        Ok(FatFileWriter {
            file,
            path: path.to_string(),
        })
    }

    /// Create a directory.
    ///
    /// Parent directories must already exist.
    ///
    /// # Arguments
    ///
    /// * `path` - Absolute path to the directory to create (e.g., "/newdir")
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use hyperlight_host::hyperlight_fs::FatImage;
    /// # fn main() -> hyperlight_host::Result<()> {
    /// # let mut image = FatImage::create_temp(1024 * 1024)?;
    /// // Create a single directory
    /// image.create_dir("/data")?;
    ///
    /// // Create nested directories (parents must exist first)
    /// image.create_dir("/data/logs")?;
    /// image.create_dir("/data/logs/2025")?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the parent directory doesn't exist or the path
    /// already exists.
    pub fn create_dir(&mut self, path: &str) -> Result<()> {
        Self::validate_path(path)?;
        trace!(path, "Creating directory");

        let fs = self.open_fs()?;
        let root = fs.root_dir();

        root.create_dir(path).map_err(|e| {
            error!(path, error = %e, "Failed to create directory");
            HyperlightError::Error(format!("Failed to create directory '{}': {}", path, e))
        })?;

        trace!(path, "Directory created");
        Ok(())
    }

    /// Delete a file.
    ///
    /// # Arguments
    ///
    /// * `path` - Absolute path to the file to delete
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The path doesn't exist
    /// - The path contains invalid components
    ///
    /// # Note
    ///
    /// The underlying fatfs `remove()` call works on both files and empty
    /// directories. Use [`delete_dir`](Self::delete_dir) for directories to
    /// make your intent clear.
    pub fn delete_file(&mut self, path: &str) -> Result<()> {
        Self::validate_path(path)?;
        trace!(path, "Deleting file");

        let fs = self.open_fs()?;
        let root = fs.root_dir();

        root.remove(path).map_err(|e| {
            error!(path, error = %e, "Failed to delete file");
            HyperlightError::Error(format!("Failed to delete file '{}': {}", path, e))
        })?;

        trace!(path, "File deleted");
        Ok(())
    }

    /// Delete an empty directory.
    ///
    /// # Arguments
    ///
    /// * `path` - Absolute path to the directory to delete
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The path doesn't exist
    /// - The directory is not empty
    /// - The path contains invalid components
    ///
    /// # Note
    ///
    /// The underlying fatfs `remove()` call works on both files and empty
    /// directories. Use [`delete_file`](Self::delete_file) for files to
    /// make your intent clear.
    pub fn delete_dir(&mut self, path: &str) -> Result<()> {
        Self::validate_path(path)?;
        trace!(path, "Deleting directory");

        let fs = self.open_fs()?;
        let root = fs.root_dir();

        root.remove(path).map_err(|e| {
            error!(path, error = %e, "Failed to delete directory");
            HyperlightError::Error(format!("Failed to delete directory '{}': {}", path, e))
        })?;

        trace!(path, "Directory deleted");
        Ok(())
    }

    /// Rename or move a file or directory.
    ///
    /// # Arguments
    ///
    /// * `old_path` - Current absolute path of the file or directory
    /// * `new_path` - New absolute path (can be in a different directory)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use hyperlight_host::hyperlight_fs::FatImage;
    /// # fn main() -> hyperlight_host::Result<()> {
    /// # let mut image = FatImage::create_temp(1024 * 1024)?;
    /// # image.create_file("/old_name.txt")?;
    /// # image.create_dir("/subdir")?;
    /// // Rename a file
    /// image.rename("/old_name.txt", "/new_name.txt")?;
    ///
    /// // Move a file to a subdirectory
    /// image.rename("/new_name.txt", "/subdir/file.txt")?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The source path doesn't exist
    /// - The destination already exists
    /// - The paths contain invalid components (e.g., "..")
    /// - The destination's parent directory doesn't exist
    /// - Attempting to rename the root directory
    pub fn rename(&mut self, old_path: &str, new_path: &str) -> Result<()> {
        Self::validate_path(old_path)?;
        Self::validate_path(new_path)?;
        trace!(old_path, new_path, "Renaming");

        // fatfs rename expects paths relative to the root directory,
        // WITHOUT a leading slash. We strip it here because our API uses
        // absolute paths (with leading /).
        let old_rel = old_path.trim_start_matches('/');
        let new_rel = new_path.trim_start_matches('/');

        // Prevent renaming the root directory itself
        if old_rel.is_empty() || new_rel.is_empty() {
            return Err(HyperlightError::Error(
                "Cannot rename the root directory".to_string(),
            ));
        }

        let fs = self.open_fs()?;
        let root = fs.root_dir();

        root.rename(old_rel, &root, new_rel).map_err(|e| {
            error!(old_path, new_path, error = %e, "Failed to rename");
            HyperlightError::Error(format!(
                "Failed to rename '{}' to '{}': {}",
                old_path, new_path, e
            ))
        })?;

        trace!(old_path, new_path, "Renamed successfully");
        Ok(())
    }

    /// Get metadata for a file or directory.
    ///
    /// # Arguments
    ///
    /// * `path` - Absolute path to the file or directory
    ///
    /// # Returns
    ///
    /// A [`FatStat`] struct containing size, type, and timestamps.
    ///
    /// # Special Case: Root Directory
    ///
    /// For the root directory ("/"), returns `FatStat` with:
    /// - `size: 0`
    /// - `is_dir: true`
    /// - All timestamps: `None`
    ///
    /// This is because FAT filesystems don't store metadata for the root
    /// directory itself.
    ///
    /// # Errors
    ///
    /// Returns an error if the path doesn't exist or contains invalid components.
    pub fn stat(&mut self, path: &str) -> Result<FatStat> {
        Self::validate_path(path)?;
        trace!(path, "Getting file stat");

        // Special case for root directory
        if path == "/" {
            return Ok(FatStat {
                size: 0,
                is_dir: true,
                created: None,
                modified: None,
                accessed: None,
            });
        }

        // Use the dedicated stat helper that captures timestamps
        self.stat_entry(path)
    }

    /// Check if a path exists (file or directory).
    ///
    /// # Arguments
    ///
    /// * `path` - Absolute path to check
    ///
    /// # Returns
    ///
    /// - `Ok(true)` if the path exists
    /// - `Ok(false)` if the path does not exist
    /// - `Err` if the path is invalid or there's an I/O error
    pub fn exists(&mut self, path: &str) -> Result<bool> {
        Self::validate_path(path)?;

        // Root always exists
        if path == "/" {
            return Ok(true);
        }

        let fs = self.open_fs()?;
        let root = fs.root_dir();

        // Try to open as file first, then as directory.
        // fatfs returns fatfs::Error::NotFound if path doesn't exist.
        let file_err = match root.open_file(path) {
            Ok(_) => return Ok(true),
            Err(fatfs::Error::NotFound) => None,
            Err(e) => Some(e), // Save error in case open_dir also fails
        };

        match root.open_dir(path) {
            Ok(_) => Ok(true),
            Err(fatfs::Error::NotFound) => Ok(false),
            Err(dir_err) => {
                // Both failed with non-NotFound errors. Report the directory error,
                // but log both for debugging.
                if let Some(fe) = file_err {
                    trace!(path, file_error = %fe, dir_error = %dir_err,
                           "Both open_file and open_dir failed");
                }
                error!(path, error = %dir_err, "I/O error checking path existence");
                // Convert fatfs::Error to std::io::Error
                let io_err: std::io::Error = dir_err.into();
                Err(HyperlightError::IOError(io_err))
            }
        }
    }

    // ---- Private helpers ----

    /// Validate that a path is absolute and doesn't contain "..".
    ///
    /// We reject any path containing ".." as a substring (not just as a path
    /// component) because:
    /// 1. FAT filesystems use ".." only as a special directory entry, never in filenames
    /// 2. Simplifies security checks - no need to handle edge cases like "foo../bar"
    /// 3. Prevents path traversal attempts
    fn validate_path(path: &str) -> Result<()> {
        if !path.starts_with('/') {
            error!(path, "Path must be absolute");
            return Err(HyperlightError::Error(format!(
                "Path must be absolute (start with '/'): '{}'",
                path
            )));
        }

        if path.contains("..") {
            error!(path, "Path contains '..' which is not allowed");
            return Err(HyperlightError::Error(format!(
                "Path contains '..' which is not allowed: '{}'",
                path
            )));
        }

        // Null bytes can truncate paths in C APIs and cause security issues
        if path.contains('\0') {
            error!(path = %path.escape_default(), "Path contains null byte");
            return Err(HyperlightError::Error(
                "Path contains null byte which is not allowed".to_string(),
            ));
        }

        Ok(())
    }

    /// Split a path into parent directory and filename.
    ///
    /// # Returns
    ///
    /// A tuple of `(parent_path, filename)`. For root, returns `("/", "")`.
    ///
    /// # Examples (conceptual)
    ///
    /// - `"/foo/bar.txt"` → `("/foo", "bar.txt")`
    /// - `"/file.txt"` → `("/", "file.txt")`
    /// - `"/"` → `("/", "")`
    fn split_path(path: &str) -> Result<(&str, &str)> {
        // Handle root specially
        if path == "/" {
            return Ok(("/", ""));
        }

        // Remove trailing slash if present
        let path = path.strip_suffix('/').unwrap_or(path);

        // Find last slash
        if let Some(pos) = path.rfind('/') {
            let parent = if pos == 0 { "/" } else { &path[..pos] };
            let name = &path[pos + 1..];
            Ok((parent, name))
        } else {
            Err(HyperlightError::Error(format!(
                "Invalid path (no parent): '{}'",
                path
            )))
        }
    }

    /// Get or create the cached FAT filesystem handle.
    ///
    /// All file operations (`read_dir`, `open_file`, `create_file`, etc.) go through
    /// this method to access the underlying fatfs library.
    ///
    /// # Why lazy?
    ///
    /// `FatImage::open()` and `create_*()` only validate the FAT boot sector and
    /// set up the mmap. We defer constructing the full `fatfs::FileSystem` until
    /// someone actually performs a file operation. This avoids parsing FAT tables
    /// for images that might only be memory-mapped into a guest without host-side
    /// file operations.
    ///
    /// # Why cached?
    ///
    /// Creating a `FileSystem` parses the boot sector and FAT tables. We cache it
    /// in `self.fs` to avoid re-parsing on every operation.
    ///
    /// # The `'static` lie
    ///
    /// This is where we create the `&'static mut [u8]` slice over the mmap region.
    /// See the struct-level documentation for why this is safe despite the lifetime
    /// not actually being `'static`.
    fn open_fs(&mut self) -> Result<&mut FatFs> {
        if self.fs.is_none() {
            // SAFETY: We're creating a slice with a 'static lifetime, but it actually
            // lives only as long as the mmap region. This is the "'static lie" documented
            // on the struct. Safety is maintained by:
            // 1. Drop::drop() sets self.fs = None before munmap
            // 2. FatImage is not Sync, so no concurrent access
            // 3. We have exclusive access via flock
            let slice: &'static mut [u8] =
                unsafe { std::slice::from_raw_parts_mut(self.mmap_ptr, self.mmap_size) };
            let cursor = std::io::Cursor::new(slice);
            // fatfs 0.4 uses custom I/O traits; StdIoWrapper adapts std::io types
            let io = fatfs::StdIoWrapper::new(cursor);

            let fs = fatfs::FileSystem::new(io, fatfs::FsOptions::new()).map_err(|e| {
                error!(error = %e, "Failed to open FAT filesystem");
                HyperlightError::Error(format!("Failed to open FAT filesystem: {}", e))
            })?;

            self.fs = Some(fs);
        }

        self.fs.as_mut().ok_or_else(|| {
            // This should never happen - we just set self.fs = Some above
            error!("Internal error: FileSystem cache unexpectedly None");
            HyperlightError::Error("Internal error: FileSystem cache unexpectedly None".to_string())
        })
    }

    /// List directory contents, collecting into owned data.
    ///
    /// # Why Vec instead of an iterator?
    ///
    /// The fatfs `Dir::iter()` yields `DirEntry` items that borrow from the
    /// `FileSystem`. If we tried to return an iterator, we'd have lifetime
    /// issues: the iterator would borrow `self.fs` (via `open_fs()`), preventing
    /// any other operations on the `FatImage` until iteration completes.
    ///
    /// By collecting into a `Vec<(String, FatStat)>` with all owned data, we
    /// release the borrow on the filesystem before returning, allowing the
    /// caller to interleave directory listing with other operations.
    ///
    /// Returns tuples of (name, FatStat). Excludes "." and ".." entries.
    fn list_dir_entries(&mut self, path: &str) -> Result<Vec<(String, FatStat)>> {
        let fs = self.open_fs()?;
        let root = fs.root_dir();

        let dir = if path == "/" {
            root
        } else {
            root.open_dir(path).map_err(|e| {
                error!(path, error = %e, "Failed to open directory");
                HyperlightError::Error(format!("Failed to open directory '{}': {}", path, e))
            })?
        };

        let mut entries = Vec::new();
        for entry in dir.iter() {
            let entry = entry.map_err(|e| {
                error!(path, error = %e, "Failed to read directory entry");
                HyperlightError::Error(format!(
                    "Failed to read directory entry in '{}': {}",
                    path, e
                ))
            })?;

            let name = entry.file_name();
            if name == "." || name == ".." {
                continue;
            }

            let stat = FatStat {
                size: entry.len(),
                is_dir: entry.is_dir(),
                created: Self::datetime_to_chrono(entry.created()),
                modified: Self::datetime_to_chrono(entry.modified()),
                accessed: Self::date_to_chrono(entry.accessed()),
            };
            entries.push((name, stat));
        }

        Ok(entries)
    }

    /// Get stat for a specific path by looking it up in its parent directory.
    fn stat_entry(&mut self, path: &str) -> Result<FatStat> {
        let fs = self.open_fs()?;
        let root = fs.root_dir();

        let (parent_path, name) = Self::split_path(path)?;

        let parent_dir = if parent_path == "/" {
            root
        } else {
            root.open_dir(parent_path).map_err(|e| {
                error!(parent_path, error = %e, "Failed to open parent directory for stat");
                HyperlightError::Error(format!(
                    "Failed to open parent directory '{}': {}",
                    parent_path, e
                ))
            })?
        };

        for entry in parent_dir.iter() {
            let entry = entry.map_err(|e| {
                error!(path, error = %e, "Failed to read directory entry during stat");
                HyperlightError::Error(format!("Failed to stat '{}': {}", path, e))
            })?;

            if entry.file_name() == name {
                return Ok(FatStat {
                    size: entry.len(),
                    is_dir: entry.is_dir(),
                    created: Self::datetime_to_chrono(entry.created()),
                    modified: Self::datetime_to_chrono(entry.modified()),
                    accessed: Self::date_to_chrono(entry.accessed()),
                });
            }
        }

        error!(path, "Path not found during stat");
        Err(HyperlightError::Error(format!(
            "Path not found: '{}'",
            path
        )))
    }

    /// Convert fatfs DateTime to chrono NaiveDateTime.
    ///
    /// Returns `None` if the date/time values are invalid (e.g., month 0).
    fn datetime_to_chrono(dt: fatfs::DateTime) -> Option<NaiveDateTime> {
        chrono::NaiveDate::from_ymd_opt(
            dt.date.year as i32,
            dt.date.month as u32,
            dt.date.day as u32,
        )
        .and_then(|date| {
            date.and_hms_opt(dt.time.hour as u32, dt.time.min as u32, dt.time.sec as u32)
        })
    }

    /// Convert fatfs Date to chrono NaiveDateTime.
    ///
    /// The time component is set to midnight (00:00:00) because FAT only
    /// stores the date for last-access timestamps.
    ///
    /// Returns `None` if the date values are invalid.
    fn date_to_chrono(d: fatfs::Date) -> Option<NaiveDateTime> {
        chrono::NaiveDate::from_ymd_opt(d.year as i32, d.month as u32, d.day as u32)
            .and_then(|date| date.and_hms_opt(0, 0, 0))
    }

    /// Initialize a newly created image file: extend, format, and mmap.
    /// Cleans up the file on failure.
    fn initialize_image(file: File, path: PathBuf, size: usize, is_temp: bool) -> Result<Self> {
        // Helper to delete file on error
        let cleanup_on_err = |e| {
            warn!(path = %path.display(), "Deleting incomplete FAT image after initialization failure");
            let _ = std::fs::remove_file(&path);
            e
        };

        // Extend file to size (creates sparse file on Linux)
        Self::extend_file(&file, size).map_err(cleanup_on_err)?;

        // Format as FAT filesystem (auto-selects FAT12/16/32 based on size)
        Self::format_fat(&file, size).map_err(cleanup_on_err)?;

        // Memory map the file
        let mmap_ptr = Self::mmap_file(&file, size).map_err(cleanup_on_err)?;

        debug!(path = %path.display(), size, is_temp, "FAT image initialized");

        Ok(Self {
            _file: file,
            mmap_ptr,
            mmap_size: size,
            path,
            is_temp,
            fs: None,
        })
    }

    /// Validate that the image size is within acceptable bounds.
    ///
    /// # Errors
    ///
    /// Returns an error if `size_bytes` is less than [`MIN_FAT_IMAGE_SIZE`]
    /// or greater than [`MAX_FAT_IMAGE_SIZE`].
    fn validate_size(size_bytes: usize) -> Result<()> {
        if !(MIN_FAT_IMAGE_SIZE..=MAX_FAT_IMAGE_SIZE).contains(&size_bytes) {
            error!(
                size = size_bytes,
                min = MIN_FAT_IMAGE_SIZE,
                max = MAX_FAT_IMAGE_SIZE,
                "FAT image size out of range"
            );
            return Err(HyperlightError::Error(format!(
                "FAT image size {} is out of range ({} - {})",
                size_bytes, MIN_FAT_IMAGE_SIZE, MAX_FAT_IMAGE_SIZE
            )));
        }
        Ok(())
    }

    /// Extend the file to the specified size using ftruncate.
    ///
    /// On Linux, this creates a sparse file (doesn't allocate physical blocks
    /// until written).
    fn extend_file(file: &File, size: usize) -> Result<()> {
        // Use ftruncate to extend the file (creates sparse file on Linux)
        file.set_len(size as u64).map_err(|e| {
            error!(size, error = %e, "Failed to extend FAT image file");
            HyperlightError::Error(format!("Failed to extend file to {} bytes: {}", size, e))
        })?;
        Ok(())
    }

    /// Memory-map the file with MAP_SHARED for write persistence.
    ///
    /// # Returns
    ///
    /// A pointer to the mapped region. The caller is responsible for calling
    /// `munmap` when done.
    ///
    /// # Errors
    ///
    /// Returns an error if the mmap syscall fails.
    fn mmap_file(file: &File, size: usize) -> Result<*mut u8> {
        use std::os::unix::io::AsRawFd;

        // SAFETY:
        // - `file` is a valid, open file descriptor (from OpenOptions or tempfile)
        // - `size` > 0 (validated by validate_size, minimum 1MB)
        // - offset 0 is valid for any file
        // - We have exclusive access via flock, preventing concurrent modification
        // - The file has been extended to `size` bytes (extend_file called first)
        // - The returned pointer is only used while FatImage is alive (Drop does munmap)
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            let err = std::io::Error::last_os_error();
            error!(size, error = %err, "Failed to mmap FAT image");
            return Err(HyperlightError::Error(format!("mmap failed: {}", err)));
        }

        Ok(ptr as *mut u8)
    }

    /// Format the file as a FAT filesystem.
    ///
    /// The `fatfs` crate auto-selects the appropriate FAT variant:
    /// - FAT12 for small volumes (< ~16MB)
    /// - FAT16 for medium volumes (16MB - 2GB)
    /// - FAT32 for large volumes (> ~33MB)
    ///
    /// # Errors
    ///
    /// Returns an error if formatting fails or the sync fails.
    fn format_fat(file: &File, size: usize) -> Result<()> {
        use std::io::{Seek, SeekFrom};

        // fatfs::format_volume requires Read + Write + Seek. We need a mutable
        // reference to the file for formatting. Since File implements these traits
        // via shared reference (&File), we clone the file descriptor to get an
        // owned File that we can pass mutably.
        let mut file_clone = file.try_clone().map_err(|e| {
            error!(error = %e, "Failed to clone file for FAT formatting");
            HyperlightError::Error(format!("Failed to clone file: {}", e))
        })?;

        // Seek to start before formatting
        file_clone.seek(SeekFrom::Start(0)).map_err(|e| {
            error!(error = %e, "Failed to seek to start for FAT formatting");
            HyperlightError::Error(format!("Failed to seek to start: {}", e))
        })?;

        // fatfs 0.4 uses custom I/O traits; StdIoWrapper adapts std::io types
        let mut io = fatfs::StdIoWrapper::new(file_clone);

        // Let fatfs auto-select the appropriate FAT type based on volume size:
        // - FAT12: < ~4085 clusters (small volumes)
        // - FAT16: 4085 - 65524 clusters (medium volumes)
        // - FAT32: >= 65525 clusters (large volumes, requires ~33MB+ with 512-byte sectors)
        let options = fatfs::FormatVolumeOptions::new();
        fatfs::format_volume(&mut io, options).map_err(|e| {
            error!(size, error = %e, "Failed to format FAT volume");
            HyperlightError::Error(format!("Failed to format FAT volume: {}", e))
        })?;

        // Ensure all data is flushed to disk
        file.sync_all().map_err(|e| {
            error!(error = %e, "Failed to sync FAT volume after formatting");
            HyperlightError::Error(format!("Failed to sync after formatting: {}", e))
        })?;

        info!(size, "FAT volume formatted");

        Ok(())
    }

    /// Validate that the mmap'd region contains a valid FAT filesystem.
    fn validate_fat_image(ptr: *mut u8, size: usize) -> Result<()> {
        // SAFETY:
        // - `ptr` is valid for `size` bytes (from successful mmap_file call)
        // - Data is initialized: either formatted by format_fat, or read from existing file
        // - The slice lifetime is bounded by this function scope (no escape)
        // - We use a mutable slice because fatfs::FileSystem requires ReadWriteSeek,
        //   but we only read (validation doesn't modify the filesystem)
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, size) };
        let cursor = std::io::Cursor::new(slice);
        // fatfs 0.4 uses custom I/O traits; StdIoWrapper adapts std::io types
        let io = fatfs::StdIoWrapper::new(cursor);

        // Try to open as a FAT filesystem - this validates the boot sector
        fatfs::FileSystem::new(io, fatfs::FsOptions::new()).map_err(|e| {
            error!(error = %e, "File is not a valid FAT image");
            HyperlightError::Error(format!("Not a valid FAT image: {}", e))
        })?;

        Ok(())
    }
}

#[cfg(unix)]
impl Drop for FatImage {
    fn drop(&mut self) {
        // Note: msync is NOT called here. Durability is ensured by the sandbox's HLT handling
        // which calls msync before returning from sandbox.call(). Calling msync here would be:
        // 1. Redundant in the normal case (HLT already synced)
        // 2. Potentially harmful in error cases (if persisting corrupted data)

        // CRITICAL: Drop the FileSystem BEFORE unmapping the memory it references.
        // The FileSystem holds a &'static mut [u8] that actually points to mmap_ptr.
        // If we munmap first, the FileSystem's drop would access freed memory.
        self.fs = None;

        // Unmap the memory region
        // SAFETY: mmap_ptr and mmap_size are valid from construction, and we've
        // already dropped self.fs above so nothing references this memory.
        let result = unsafe { libc::munmap(self.mmap_ptr as *mut libc::c_void, self.mmap_size) };
        if result != 0 {
            let err = std::io::Error::last_os_error();
            error!(error = %err, "Failed to munmap FAT image");
        }

        // Lock is automatically released when file is dropped

        // Delete temp file if applicable
        if self.is_temp {
            if let Err(e) = std::fs::remove_file(&self.path) {
                error!(path = %self.path.display(), error = %e, "Failed to delete temp FAT image");
            } else {
                debug!(path = %self.path.display(), "Deleted temp FAT image");
            }
        }
    }
}

/// Windows implementation of FatImage using memory-mapped files.
///
/// Uses CreateFileMappingW / MapViewOfFile for memory mapping and fs2 for
/// file locking (same as Unix).
#[cfg(windows)]
pub struct FatImage {
    /// The backing file (holds the exclusive lock via fs2).
    _file: File,
    /// Memory-mapped view pointer.
    mmap_ptr: *mut u8,
    /// Size of the mapped region.
    mmap_size: usize,
    /// Path to the backing file.
    path: PathBuf,
    /// Whether this is a temp file (delete on drop).
    is_temp: bool,
    /// Windows file mapping handle (must be closed after unmapping).
    mapping_handle: windows::Win32::Foundation::HANDLE,
    /// Cached FAT filesystem handle.
    ///
    /// # Safety
    ///
    /// The `'static` lifetime is a lie — see Unix FatImage docs.
    /// This field **must** be set to `None` before unmapping in `Drop::drop()`.
    fs: Option<FatFs>,
}

#[cfg(windows)]
impl std::fmt::Debug for FatImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FatImage")
            .field("mmap_ptr", &self.mmap_ptr)
            .field("mmap_size", &self.mmap_size)
            .field("path", &self.path)
            .field("is_temp", &self.is_temp)
            .field("fs", &self.fs.as_ref().map(|_| "<FileSystem>"))
            .finish()
    }
}

// SAFETY: Same reasoning as Unix — mmap'd region is process-local,
// file lock ensures exclusive access.
#[cfg(windows)]
unsafe impl Send for FatImage {}

#[cfg(windows)]
impl FatImage {
    /// Open an existing FAT image from a file with exclusive lock.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        use std::os::windows::io::AsRawHandle;
        use windows::Win32::Foundation::HANDLE;
        use windows::Win32::System::Memory::*;

        let path = path.as_ref().to_path_buf();

        info!(path = %path.display(), "Opening FAT image");

        // Open file for read/write
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| {
                error!(path = %path.display(), error = %e, "Failed to open FAT image");
                HyperlightError::Error(format!("Failed to open FAT image {:?}: {}", path, e))
            })?;

        // Try to acquire exclusive lock (non-blocking) via fs2
        file.try_lock_exclusive().map_err(|e| {
            error!(path = %path.display(), error = %e, "FAT image is locked by another process");
            HyperlightError::Error(format!(
                "File is locked by another process {:?}: {}",
                path, e
            ))
        })?;

        // Get file size
        let metadata = file.metadata().map_err(|e| {
            error!(path = %path.display(), error = %e, "Failed to get FAT image metadata");
            HyperlightError::Error(format!("Failed to get metadata for {:?}: {}", path, e))
        })?;
        let size = metadata.len() as usize;

        Self::validate_size(size)?;

        // Create a file mapping object
        let file_handle = HANDLE(file.as_raw_handle());
        let mapping_handle = unsafe {
            CreateFileMappingW(
                file_handle,
                None,
                PAGE_READWRITE,
                0,
                0,
                None,
            )
        }
        .map_err(|e| {
            error!(error = %e, "Failed to create file mapping");
            HyperlightError::Error(format!("CreateFileMappingW failed: {}", e))
        })?;

        // Map the entire file into memory
        let ptr = unsafe {
            MapViewOfFile(
                mapping_handle,
                FILE_MAP_READ | FILE_MAP_WRITE,
                0,
                0,
                size,
            )
        };

        if ptr.Value.is_null() {
            let err = std::io::Error::last_os_error();
            unsafe {
                windows::Win32::Foundation::CloseHandle(mapping_handle).ok();
            }
            error!(size, error = %err, "Failed to map FAT image into memory");
            return Err(HyperlightError::Error(format!(
                "MapViewOfFile failed: {}",
                err
            )));
        }

        let mmap_ptr = ptr.Value as *mut u8;

        // Validate it's actually a FAT filesystem
        if let Err(e) = Self::validate_fat_image(mmap_ptr, size) {
            unsafe {
                UnmapViewOfFile(ptr).ok();
                windows::Win32::Foundation::CloseHandle(mapping_handle).ok();
            }
            return Err(e);
        }

        info!(path = %path.display(), size, "FAT image opened and locked");

        Ok(Self {
            _file: file,
            mmap_ptr,
            mmap_size: size,
            path,
            is_temp: false,
            mapping_handle,
            fs: None,
        })
    }

    /// Returns an error - creating FAT images is not supported on Windows.
    pub fn create_at<P: AsRef<Path>>(_path: P, _size_bytes: usize) -> Result<Self> {
        Err(HyperlightError::Error(
            "Creating FAT images is not supported on Windows".to_string(),
        ))
    }

    /// Returns an error - creating FAT images is not supported on Windows.
    pub fn create_temp(_size_bytes: usize) -> Result<Self> {
        Err(HyperlightError::Error(
            "Creating FAT images is not supported on Windows".to_string(),
        ))
    }

    /// Get pointer to the memory-mapped FAT image.
    pub fn as_ptr(&self) -> *const u8 {
        self.mmap_ptr as *const u8
    }

    /// Get mutable pointer to the memory-mapped FAT image.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.mmap_ptr
    }

    /// Get the size of the FAT image.
    pub fn size(&self) -> usize {
        self.mmap_size
    }

    /// Get the path to the backing file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Whether this is a temporary file.
    pub fn is_temp(&self) -> bool {
        self.is_temp
    }

    /// Validate the FAT image size.
    fn validate_size(size_bytes: usize) -> Result<()> {
        if !(MIN_FAT_IMAGE_SIZE..=MAX_FAT_IMAGE_SIZE).contains(&size_bytes) {
            error!(
                size = size_bytes,
                min = MIN_FAT_IMAGE_SIZE,
                max = MAX_FAT_IMAGE_SIZE,
                "FAT image size out of range"
            );
            return Err(HyperlightError::Error(format!(
                "FAT image size {} is out of range ({} - {})",
                size_bytes, MIN_FAT_IMAGE_SIZE, MAX_FAT_IMAGE_SIZE
            )));
        }
        Ok(())
    }

    /// Validate that the mmap'd region contains a valid FAT filesystem.
    fn validate_fat_image(ptr: *mut u8, size: usize) -> Result<()> {
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, size) };
        let cursor = std::io::Cursor::new(slice);
        let io = fatfs::StdIoWrapper::new(cursor);

        fatfs::FileSystem::new(io, fatfs::FsOptions::new()).map_err(|e| {
            error!(error = %e, "File is not a valid FAT image");
            HyperlightError::Error(format!("Not a valid FAT image: {}", e))
        })?;

        Ok(())
    }

    /// Flush memory-mapped changes to disk.
    pub(crate) fn msync(&self) -> Result<()> {
        use windows::Win32::System::Memory::FlushViewOfFile;

        unsafe {
            FlushViewOfFile(self.mmap_ptr as *const std::ffi::c_void, self.mmap_size).map_err(
                |e| {
                    error!(error = %e, "Failed to flush FAT image to disk");
                    HyperlightError::Error(format!("FlushViewOfFile failed: {}", e))
                },
            )?;
        }
        Ok(())
    }

    /// Get or create the cached FAT filesystem handle.
    fn get_fs(&mut self) -> Result<&mut FatFs> {
        if self.fs.is_none() {
            let slice = unsafe { std::slice::from_raw_parts_mut(self.mmap_ptr, self.mmap_size) };
            let cursor = std::io::Cursor::new(slice);
            let io = fatfs::StdIoWrapper::new(cursor);
            let fs = fatfs::FileSystem::new(io, fatfs::FsOptions::new()).map_err(|e| {
                HyperlightError::Error(format!("Failed to open FAT filesystem: {}", e))
            })?;
            self.fs = Some(fs);
        }
        Ok(self.fs.as_mut().unwrap())
    }

    /// List directory contents.
    pub fn read_dir(&mut self, path: &str) -> Result<Vec<FatEntry>> {
        Self::validate_path(path)?;
        trace!(path, "Reading directory");

        let raw_entries = self.list_dir_entries(path)?;
        let entries: Vec<FatEntry> = raw_entries
            .into_iter()
            .map(|(name, stat)| FatEntry { name, stat })
            .collect();

        trace!(path, count = entries.len(), "Directory read complete");
        Ok(entries)
    }

    /// Open a file for reading.
    pub fn open_file(&mut self, path: &str) -> Result<FatFileReader<'_>> {
        Self::validate_path(path)?;
        trace!(path, "Opening file for reading");

        let fs = self.get_fs()?;
        let root = fs.root_dir();

        let file = root.open_file(path).map_err(|e| {
            error!(path, error = %e, "Failed to open file for reading");
            HyperlightError::Error(format!("Failed to open file '{}': {}", path, e))
        })?;

        trace!(path, "File opened for reading");
        Ok(FatFileReader {
            file,
            path: path.to_string(),
        })
    }

    /// Create or truncate a file for writing.
    pub fn create_file(&mut self, path: &str) -> Result<FatFileWriter<'_>> {
        Self::validate_path(path)?;
        trace!(path, "Creating file for writing");

        let fs = self.get_fs()?;
        let root = fs.root_dir();

        let mut file = root.create_file(path).map_err(|e| {
            error!(path, error = %e, "Failed to create file for writing");
            HyperlightError::Error(format!("Failed to create file '{}': {}", path, e))
        })?;

        file.truncate().map_err(|e| {
            error!(path, error = %e, "Failed to truncate file");
            HyperlightError::Error(format!("Failed to truncate file '{}': {}", path, e))
        })?;

        trace!(path, "File created and truncated for writing");
        Ok(FatFileWriter {
            file,
            path: path.to_string(),
        })
    }

    /// Create a directory.
    pub fn create_dir(&mut self, path: &str) -> Result<()> {
        Self::validate_path(path)?;
        trace!(path, "Creating directory");

        let fs = self.get_fs()?;
        let root = fs.root_dir();

        root.create_dir(path).map_err(|e| {
            error!(path, error = %e, "Failed to create directory");
            HyperlightError::Error(format!("Failed to create directory '{}': {}", path, e))
        })?;

        trace!(path, "Directory created");
        Ok(())
    }

    /// Delete a file.
    pub fn delete_file(&mut self, path: &str) -> Result<()> {
        Self::validate_path(path)?;
        trace!(path, "Deleting file");

        let fs = self.get_fs()?;
        let root = fs.root_dir();

        root.remove(path).map_err(|e| {
            error!(path, error = %e, "Failed to delete file");
            HyperlightError::Error(format!("Failed to delete file '{}': {}", path, e))
        })?;

        trace!(path, "File deleted");
        Ok(())
    }

    /// Delete an empty directory.
    pub fn delete_dir(&mut self, path: &str) -> Result<()> {
        Self::validate_path(path)?;
        trace!(path, "Deleting directory");

        let fs = self.get_fs()?;
        let root = fs.root_dir();

        root.remove(path).map_err(|e| {
            error!(path, error = %e, "Failed to delete directory");
            HyperlightError::Error(format!("Failed to delete directory '{}': {}", path, e))
        })?;

        trace!(path, "Directory deleted");
        Ok(())
    }

    /// Rename a file or directory.
    pub fn rename(&mut self, old_path: &str, new_path: &str) -> Result<()> {
        Self::validate_path(old_path)?;
        Self::validate_path(new_path)?;
        trace!(old_path, new_path, "Renaming");

        let old_rel = old_path.trim_start_matches('/');
        let new_rel = new_path.trim_start_matches('/');

        if old_rel.is_empty() || new_rel.is_empty() {
            return Err(HyperlightError::Error(
                "Cannot rename the root directory".to_string(),
            ));
        }

        let fs = self.get_fs()?;
        let root = fs.root_dir();

        root.rename(old_rel, &root, new_rel).map_err(|e| {
            error!(old_path, new_path, error = %e, "Failed to rename");
            HyperlightError::Error(format!(
                "Failed to rename '{}' to '{}': {}",
                old_path, new_path, e
            ))
        })?;

        trace!(old_path, new_path, "Renamed successfully");
        Ok(())
    }

    /// Get metadata for a file or directory.
    pub fn stat(&mut self, path: &str) -> Result<FatStat> {
        Self::validate_path(path)?;
        trace!(path, "Getting file stat");

        if path == "/" {
            return Ok(FatStat {
                size: 0,
                is_dir: true,
                created: None,
                modified: None,
                accessed: None,
            });
        }

        self.stat_entry(path)
    }

    /// Check if a path exists (file or directory).
    pub fn exists(&mut self, path: &str) -> Result<bool> {
        Self::validate_path(path)?;

        if path == "/" {
            return Ok(true);
        }

        let fs = self.get_fs()?;
        let root = fs.root_dir();

        let file_err = match root.open_file(path) {
            Ok(_) => return Ok(true),
            Err(fatfs::Error::NotFound) => None,
            Err(e) => Some(e),
        };

        match root.open_dir(path) {
            Ok(_) => Ok(true),
            Err(fatfs::Error::NotFound) => Ok(false),
            Err(dir_err) => {
                if let Some(fe) = file_err {
                    trace!(path, file_error = %fe, dir_error = %dir_err,
                           "Both open_file and open_dir failed");
                }
                error!(path, error = %dir_err, "I/O error checking path existence");
                let io_err: std::io::Error = dir_err.into();
                Err(HyperlightError::IOError(io_err))
            }
        }
    }

    // ---- Private helpers ----

    /// Validate that a path is absolute and doesn't contain "..".
    fn validate_path(path: &str) -> Result<()> {
        if !path.starts_with('/') {
            error!(path, "Path must be absolute");
            return Err(HyperlightError::Error(format!(
                "Path must be absolute (start with '/'): '{}'",
                path
            )));
        }

        if path.contains("..") {
            error!(path, "Path contains '..' which is not allowed");
            return Err(HyperlightError::Error(format!(
                "Path contains '..' which is not allowed: '{}'",
                path
            )));
        }

        if path.contains('\0') {
            error!(path = %path.escape_default(), "Path contains null byte");
            return Err(HyperlightError::Error(
                "Path contains null byte which is not allowed".to_string(),
            ));
        }

        Ok(())
    }

    /// Split a path into parent directory and filename.
    fn split_path(path: &str) -> Result<(&str, &str)> {
        if path == "/" {
            return Ok(("/", ""));
        }

        let path = path.strip_suffix('/').unwrap_or(path);

        if let Some(pos) = path.rfind('/') {
            let parent = if pos == 0 { "/" } else { &path[..pos] };
            let name = &path[pos + 1..];
            Ok((parent, name))
        } else {
            Err(HyperlightError::Error(format!(
                "Invalid path (no parent): '{}'",
                path
            )))
        }
    }

    /// List directory contents, collecting into owned data.
    fn list_dir_entries(&mut self, path: &str) -> Result<Vec<(String, FatStat)>> {
        let fs = self.get_fs()?;
        let root = fs.root_dir();

        let dir = if path == "/" {
            root
        } else {
            root.open_dir(path).map_err(|e| {
                error!(path, error = %e, "Failed to open directory");
                HyperlightError::Error(format!("Failed to open directory '{}': {}", path, e))
            })?
        };

        let mut entries = Vec::new();
        for entry in dir.iter() {
            let entry = entry.map_err(|e| {
                error!(path, error = %e, "Failed to read directory entry");
                HyperlightError::Error(format!(
                    "Failed to read directory entry in '{}': {}",
                    path, e
                ))
            })?;

            let name = entry.file_name();
            if name == "." || name == ".." {
                continue;
            }

            let stat = FatStat {
                size: entry.len(),
                is_dir: entry.is_dir(),
                created: Self::datetime_to_chrono(entry.created()),
                modified: Self::datetime_to_chrono(entry.modified()),
                accessed: Self::date_to_chrono(entry.accessed()),
            };
            entries.push((name, stat));
        }

        Ok(entries)
    }

    /// Get stat for a specific path by looking it up in its parent directory.
    fn stat_entry(&mut self, path: &str) -> Result<FatStat> {
        let fs = self.get_fs()?;
        let root = fs.root_dir();

        let (parent_path, name) = Self::split_path(path)?;

        let parent_dir = if parent_path == "/" {
            root
        } else {
            root.open_dir(parent_path).map_err(|e| {
                error!(parent_path, error = %e, "Failed to open parent directory for stat");
                HyperlightError::Error(format!(
                    "Failed to open parent directory '{}': {}",
                    parent_path, e
                ))
            })?
        };

        for entry in parent_dir.iter() {
            let entry = entry.map_err(|e| {
                error!(path, error = %e, "Failed to read directory entry during stat");
                HyperlightError::Error(format!("Failed to stat '{}': {}", path, e))
            })?;

            if entry.file_name() == name {
                return Ok(FatStat {
                    size: entry.len(),
                    is_dir: entry.is_dir(),
                    created: Self::datetime_to_chrono(entry.created()),
                    modified: Self::datetime_to_chrono(entry.modified()),
                    accessed: Self::date_to_chrono(entry.accessed()),
                });
            }
        }

        error!(path, "Path not found during stat");
        Err(HyperlightError::Error(format!(
            "Path not found: '{}'",
            path
        )))
    }

    /// Convert fatfs DateTime to chrono NaiveDateTime.
    fn datetime_to_chrono(dt: fatfs::DateTime) -> Option<NaiveDateTime> {
        chrono::NaiveDate::from_ymd_opt(
            dt.date.year as i32,
            dt.date.month as u32,
            dt.date.day as u32,
        )
        .and_then(|date| {
            date.and_hms_opt(dt.time.hour as u32, dt.time.min as u32, dt.time.sec as u32)
        })
    }

    /// Convert fatfs Date to chrono NaiveDateTime.
    fn date_to_chrono(d: fatfs::Date) -> Option<NaiveDateTime> {
        chrono::NaiveDate::from_ymd_opt(d.year as i32, d.month as u32, d.day as u32)
            .and_then(|date| date.and_hms_opt(0, 0, 0))
    }
}

#[cfg(windows)]
impl Drop for FatImage {
    fn drop(&mut self) {
        // CRITICAL: Drop the FileSystem BEFORE unmapping the memory it references.
        self.fs = None;

        // Unmap the view
        let view = windows::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
            Value: self.mmap_ptr as *mut std::ffi::c_void,
        };
        if let Err(e) = unsafe { windows::Win32::System::Memory::UnmapViewOfFile(view) } {
            error!(error = %e, "Failed to unmap FAT image view");
        }

        // Close the mapping handle
        if let Err(e) = unsafe { windows::Win32::Foundation::CloseHandle(self.mapping_handle) } {
            error!(error = %e, "Failed to close file mapping handle");
        }

        // File lock is released when _file is dropped

        // Delete temp file if applicable
        if self.is_temp {
            if let Err(e) = std::fs::remove_file(&self.path) {
                error!(path = %self.path.display(), error = %e, "Failed to delete temp FAT image");
            } else {
                debug!(path = %self.path.display(), "Deleted temp FAT image");
            }
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use tempfile::TempDir;

    use super::*;

    #[test]
    fn test_fat_image_create_temp() {
        let image = FatImage::create_temp(MIN_FAT_IMAGE_SIZE).expect("Failed to create temp image");

        assert_eq!(image.size(), MIN_FAT_IMAGE_SIZE);
        assert!(image.is_temp());
        assert!(!image.as_ptr().is_null());

        // Verify the file exists
        assert!(image.path().exists());

        let path = image.path().to_path_buf();

        // Drop should delete the temp file
        drop(image);
        assert!(!path.exists());
    }

    #[test]
    fn test_fat_image_create_at() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.fat");

        let image =
            FatImage::create_at(&path, MIN_FAT_IMAGE_SIZE).expect("Failed to create image at path");

        assert_eq!(image.size(), MIN_FAT_IMAGE_SIZE);
        assert!(!image.is_temp());
        assert!(path.exists());

        drop(image);

        // File should persist after drop (not temp)
        assert!(path.exists());
    }

    #[test]
    fn test_fat_image_open() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.fat");

        // Create first
        {
            let _image = FatImage::create_at(&path, MIN_FAT_IMAGE_SIZE)
                .expect("Failed to create image at path");
        }

        // Open existing
        let image = FatImage::open(&path).expect("Failed to open existing image");

        assert_eq!(image.size(), MIN_FAT_IMAGE_SIZE);
        assert!(!image.is_temp());
    }

    #[test]
    fn test_fat_image_exclusive_lock() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.fat");

        // Create and hold lock
        let _image1 =
            FatImage::create_at(&path, MIN_FAT_IMAGE_SIZE).expect("Failed to create image");

        // Second open should fail with lock error
        let result = FatImage::open(&path);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("locked"), "Expected lock error, got: {}", err);
    }

    #[test]
    fn test_fat_image_lock_released_on_drop() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.fat");

        // Create and drop
        {
            let _image =
                FatImage::create_at(&path, MIN_FAT_IMAGE_SIZE).expect("Failed to create image");
        }

        // Should be able to open now
        let image = FatImage::open(&path).expect("Failed to open after drop");
        assert_eq!(image.size(), MIN_FAT_IMAGE_SIZE);
    }

    #[test]
    fn test_fat_image_size_validation() {
        // Too small
        let result = FatImage::create_temp(MIN_FAT_IMAGE_SIZE - 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("out of range"));

        // Too large (for open - create a sparse file that reports oversized)
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("oversized.fat");
        let file = std::fs::File::create(&path).unwrap();
        file.set_len(MAX_FAT_IMAGE_SIZE as u64 + 1).unwrap();
        drop(file);

        let result = FatImage::open(&path);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("out of range"),
            "Expected 'out of range' error, got: {}",
            err
        );
    }

    #[test]
    fn test_fat_image_create_at_fails_if_exists() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("existing.fat");

        // Create file first time - should succeed
        {
            let _image = FatImage::create_at(&path, MIN_FAT_IMAGE_SIZE)
                .expect("Failed to create image first time");
        }

        // Create again at same path - should fail
        let result = FatImage::create_at(&path, MIN_FAT_IMAGE_SIZE);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("exists") || err.contains("Failed to create"),
            "Expected 'already exists' error, got: {}",
            err
        );
    }

    #[test]
    fn test_fat_image_open_rejects_non_fat_file() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("not_a_fat.img");

        // Create a file with garbage content (not a FAT image)
        let file = std::fs::File::create(&path).unwrap();
        file.set_len(MIN_FAT_IMAGE_SIZE as u64).unwrap();
        drop(file);

        // Opening should fail with validation error
        let result = FatImage::open(&path);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("valid FAT"),
            "Expected 'valid FAT' error, got: {}",
            err
        );
    }

    // ---- File Operations Tests ----

    #[test]
    fn test_fat_write_read_file() {
        let mut image = FatImage::create_temp(MIN_FAT_IMAGE_SIZE).expect("Failed to create image");

        // Write a file using streaming API
        let data = b"Hello, FAT world!";
        {
            let mut writer = image
                .create_file("/test.txt")
                .expect("Failed to create file");
            writer.write_all(data).expect("Failed to write data");
            writer.flush().expect("Failed to flush");
        }

        // Read it back using streaming API
        let mut contents = Vec::new();
        {
            let mut reader = image.open_file("/test.txt").expect("Failed to open file");
            reader.read_to_end(&mut contents).expect("Failed to read");
        }
        assert_eq!(contents, data);

        // Verify it exists
        assert!(image.exists("/test.txt").expect("exists failed"));

        // Check stat
        let stat = image.stat("/test.txt").expect("Failed to stat file");
        assert_eq!(stat.size, data.len() as u64);
        assert!(!stat.is_dir);
    }

    #[test]
    fn test_fat_overwrite_file() {
        let mut image = FatImage::create_temp(MIN_FAT_IMAGE_SIZE).expect("Failed to create image");

        // Write initial content
        {
            let mut writer = image
                .create_file("/test.txt")
                .expect("Failed to create file");
            writer
                .write_all(b"initial content")
                .expect("Failed to write");
        }

        // Overwrite with different content
        let new_data = b"new content";
        {
            let mut writer = image
                .create_file("/test.txt")
                .expect("Failed to create file");
            writer.write_all(new_data).expect("Failed to write");
        }

        // Read it back - should be new content only
        let mut contents = Vec::new();
        {
            let mut reader = image.open_file("/test.txt").expect("Failed to open file");
            reader.read_to_end(&mut contents).expect("Failed to read");
        }
        assert_eq!(contents, new_data);
    }

    #[test]
    fn test_fat_create_delete_dir() {
        let mut image = FatImage::create_temp(MIN_FAT_IMAGE_SIZE).expect("Failed to create image");

        // Create a directory
        image
            .create_dir("/mydir")
            .expect("Failed to create directory");

        // Verify it exists
        assert!(image.exists("/mydir").expect("exists failed"));

        // Check stat
        let stat = image.stat("/mydir").expect("Failed to stat directory");
        assert!(stat.is_dir);

        // Delete the directory
        image
            .delete_dir("/mydir")
            .expect("Failed to delete directory");

        // Verify it's gone
        assert!(!image.exists("/mydir").expect("exists failed"));
    }

    #[test]
    fn test_fat_list_dir() {
        let mut image = FatImage::create_temp(MIN_FAT_IMAGE_SIZE).expect("Failed to create image");

        // Create some files and directories
        {
            let mut w = image.create_file("/file1.txt").expect("create");
            w.write_all(b"content1").expect("write");
        }
        {
            let mut w = image.create_file("/file2.txt").expect("create");
            w.write_all(b"content2").expect("write");
        }
        image
            .create_dir("/subdir")
            .expect("Failed to create subdir");

        // List root directory
        let entries = image.read_dir("/").expect("Failed to read root dir");

        assert_eq!(entries.len(), 3);

        // Check that all entries are present (order may vary)
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"file1.txt"));
        assert!(names.contains(&"file2.txt"));
        assert!(names.contains(&"subdir"));

        // Verify types
        let file1 = entries.iter().find(|e| e.name == "file1.txt").unwrap();
        assert!(!file1.stat.is_dir);
        assert_eq!(file1.stat.size, 8);

        let subdir = entries.iter().find(|e| e.name == "subdir").unwrap();
        assert!(subdir.stat.is_dir);
    }

    #[test]
    fn test_fat_nested_dirs() {
        let mut image = FatImage::create_temp(MIN_FAT_IMAGE_SIZE).expect("Failed to create image");

        // Create nested directories
        image
            .create_dir("/level1")
            .expect("Failed to create level1");
        image
            .create_dir("/level1/level2")
            .expect("Failed to create level2");
        image
            .create_dir("/level1/level2/level3")
            .expect("Failed to create level3");

        // Write a file deep in the tree
        {
            let mut writer = image
                .create_file("/level1/level2/level3/deep.txt")
                .expect("Failed to create deep file");
            writer.write_all(b"deep content").expect("Failed to write");
        }

        // Verify we can read it back
        let mut contents = Vec::new();
        {
            let mut reader = image
                .open_file("/level1/level2/level3/deep.txt")
                .expect("Failed to open deep file");
            reader.read_to_end(&mut contents).expect("Failed to read");
        }
        assert_eq!(contents, b"deep content");

        // List intermediate directory
        let entries = image
            .read_dir("/level1/level2")
            .expect("Failed to read level2");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "level3");
        assert!(entries[0].stat.is_dir);
    }

    #[test]
    fn test_fat_delete_file() {
        let mut image = FatImage::create_temp(MIN_FAT_IMAGE_SIZE).expect("Failed to create image");

        // Create and then delete a file
        {
            let mut writer = image.create_file("/deleteme.txt").expect("create");
            writer.write_all(b"to be deleted").expect("write");
        }
        assert!(image.exists("/deleteme.txt").expect("exists failed"));

        image
            .delete_file("/deleteme.txt")
            .expect("Failed to delete file");
        assert!(!image.exists("/deleteme.txt").expect("exists failed"));
    }

    #[test]
    fn test_fat_path_validation() {
        let mut image = FatImage::create_temp(MIN_FAT_IMAGE_SIZE).expect("Failed to create image");

        // Relative path should fail
        image
            .read_dir("relative/path")
            .expect_err("relative path should be rejected");

        // Path with .. should fail
        image
            .read_dir("/foo/../bar")
            .expect_err("path with .. should be rejected");

        // exists should also error on invalid paths (not just return false)
        image
            .exists("relative/path")
            .expect_err("exists should reject invalid paths, not return false");
    }

    #[test]
    fn test_fat_path_validation_null_byte() {
        let mut image = FatImage::create_temp(MIN_FAT_IMAGE_SIZE).expect("Failed to create image");

        // Null bytes should fail (security: prevents C API path truncation attacks)
        image
            .read_dir("/foo\0bar")
            .expect_err("null byte in path should be rejected");
        image
            .exists("/test\0.txt")
            .expect_err("null byte should be rejected");
        image
            .create_file("/malicious\0hidden.txt")
            .expect_err("null byte should be rejected");
    }

    #[test]
    fn test_fat_exists_directory() {
        let mut image = FatImage::create_temp(MIN_FAT_IMAGE_SIZE).expect("Failed to create image");

        // Create a directory
        image.create_dir("/mydir").expect("Failed to create dir");

        // exists() should return true for directories
        assert!(
            image.exists("/mydir").expect("exists failed"),
            "exists() should return true for directories"
        );

        // Also test nested directory
        image
            .create_dir("/mydir/subdir")
            .expect("Failed to create subdir");
        assert!(
            image.exists("/mydir/subdir").expect("exists failed"),
            "exists() should return true for nested directories"
        );

        // Non-existent directory should return false
        assert!(
            !image.exists("/nonexistent").expect("exists failed"),
            "exists() should return false for non-existent paths"
        );
    }

    #[test]
    fn test_fat_root_exists_and_stat() {
        let mut image = FatImage::create_temp(MIN_FAT_IMAGE_SIZE).expect("Failed to create image");

        // Root should always exist
        assert!(image.exists("/").expect("exists failed"));

        // Stat on root
        let stat = image.stat("/").expect("Failed to stat root");
        assert!(stat.is_dir);
        assert_eq!(stat.size, 0);
    }

    #[test]
    fn test_fat_read_nonexistent_file() {
        let mut image = FatImage::create_temp(MIN_FAT_IMAGE_SIZE).expect("Failed to create image");

        let result = image.open_file("/nonexistent.txt");
        assert!(result.is_err());
    }

    #[test]
    fn test_fat_timestamps() {
        let mut image = FatImage::create_temp(MIN_FAT_IMAGE_SIZE).expect("Failed to create image");

        // Write a file
        let before = chrono::Utc::now().naive_utc();

        {
            let mut writer = image.create_file("/timestamped.txt").expect("create");
            writer.write_all(b"test content").expect("write");
        }

        let after = chrono::Utc::now().naive_utc();

        // Get stat and verify timestamps are populated
        let stat = image.stat("/timestamped.txt").expect("Failed to stat file");

        // Created and modified should be populated for a newly written file
        let created = stat.created.expect("Created timestamp should be populated");
        let modified = stat
            .modified
            .expect("Modified timestamp should be populated");
        assert!(
            stat.accessed.is_some(),
            "Accessed timestamp should be populated"
        );

        // Verify timestamps are within the test execution window
        // (FAT has 2-second resolution, so allow a few seconds of slack)
        let slack = chrono::Duration::seconds(5);
        assert!(
            created >= before - slack && created <= after + slack,
            "Created timestamp {} should be between {} and {}",
            created,
            before - slack,
            after + slack
        );
        assert!(
            modified >= before - slack && modified <= after + slack,
            "Modified timestamp {} should be between {} and {}",
            modified,
            before - slack,
            after + slack
        );
    }

    #[test]
    fn test_fat_delete_nonempty_dir_fails() {
        let mut image = FatImage::create_temp(MIN_FAT_IMAGE_SIZE).expect("Failed to create image");

        // Create directory with a file in it
        image.create_dir("/nonempty").expect("Failed to create dir");
        {
            let mut writer = image.create_file("/nonempty/file.txt").expect("create");
            writer.write_all(b"content").expect("write");
        }

        // Deleting non-empty directory should fail
        let result = image.delete_dir("/nonempty");
        assert!(result.is_err());
    }

    #[test]
    fn test_fat_streaming_read_partial() {
        let mut image = FatImage::create_temp(MIN_FAT_IMAGE_SIZE).expect("Failed to create image");

        // Write a file with known content
        let data = b"0123456789ABCDEF";
        {
            let mut writer = image.create_file("/partial.txt").expect("create");
            writer.write_all(data).expect("write");
        }

        // Read only first 5 bytes using streaming API
        let mut buf = [0u8; 5];
        let bytes_read = {
            let mut reader = image.open_file("/partial.txt").expect("open");
            reader.read(&mut buf).expect("read")
        };

        assert_eq!(bytes_read, 5);
        assert_eq!(&buf, b"01234");
    }

    #[test]
    fn test_fat_streaming_write_chunks() {
        let mut image = FatImage::create_temp(MIN_FAT_IMAGE_SIZE).expect("Failed to create image");

        // Write in multiple chunks using streaming API
        {
            let mut writer = image.create_file("/chunked.txt").expect("create");
            writer.write_all(b"chunk1").expect("write1");
            writer.write_all(b"chunk2").expect("write2");
            writer.write_all(b"chunk3").expect("write3");
        }

        // Read it back and verify
        let mut contents = Vec::new();
        {
            let mut reader = image.open_file("/chunked.txt").expect("open");
            reader.read_to_end(&mut contents).expect("read");
        }

        assert_eq!(contents, b"chunk1chunk2chunk3");
    }

    #[test]
    fn test_fat_copy_between_files() {
        let mut image = FatImage::create_temp(MIN_FAT_IMAGE_SIZE).expect("Failed to create image");

        // Write source data
        let source_data = b"Data to be copied using idiomatic Rust I/O!";
        {
            let mut writer = image.create_file("/source.txt").expect("create");
            writer.write_all(source_data).expect("write");
        }

        // Read source into memory, then write to dest
        // (Can't have reader + writer open simultaneously - they both borrow &mut image)
        let data_copy = {
            let mut reader = image.open_file("/source.txt").expect("open source");
            let mut buf = Vec::new();
            reader.read_to_end(&mut buf).expect("read");
            buf
        };

        {
            let mut writer = image.create_file("/dest.txt").expect("create dest");
            writer.write_all(&data_copy).expect("write dest");
        }

        // Verify the copy
        let mut contents = Vec::new();
        {
            let mut reader = image.open_file("/dest.txt").expect("open dest");
            reader.read_to_end(&mut contents).expect("read");
        }
        assert_eq!(contents, source_data);
    }
}
