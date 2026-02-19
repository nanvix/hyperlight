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

//! Unified file handle with VFS routing.
//!
//! This module provides a [`File`] type that transparently works with both:
//! - **Read-only files**: Memory-mapped from the host manifest
//! - **FAT files**: Read-write files on FAT filesystem mounts
//!
//! The VFS automatically routes operations to the correct backend based on
//! which mount point the file path resolves to.
//!
//! # Opening Files
//!
//! Use [`open()`] for simple read-only access, or [`OpenOptions`] for more control:
//!
//! ```ignore
//! use hyperlight_guest::fs::{self, OpenOptions};
//!
//! // Simple read-only open (works for both RO and FAT files)
//! let file = fs::open("/config.json")?;
//!
//! // Open with specific options (FAT files support all options)
//! let file = OpenOptions::new()
//!     .read(true)
//!     .write(true)
//!     .create(true)
//!     .open("/data/output.txt")?;
//! ```
//!
//! # File Operations
//!
//! [`File`] implements [`embedded_io::Read`], [`embedded_io::Seek`], and
//! [`embedded_io::Write`]. Write operations return [`FsError::ReadOnly`]
//! for read-only files.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use embedded_io::{ErrorType, Read, Seek, SeekFrom, Write};

use super::error::FsError;
use super::fd::{self, OpenFile};
use super::manifest;

// ============================================================================
// Raw File Descriptor Type
// ============================================================================

/// Raw file descriptor type, matching [`std::os::fd::RawFd`].
///
/// This is an alias for `i32` to match POSIX conventions.
pub type RawFd = i32;

/// Builder for opening files with specific access options.
///
/// This provides a readable, builder-pattern API for specifying file open modes
/// instead of multiple boolean parameters. For FAT files, all options are
/// supported. For read-only files, only `read(true)` is valid.
///
/// # Default Behavior
///
/// If you call `open()` without setting any options, it defaults to read-only
/// mode (equivalent to `.read(true)`).
///
/// # Examples
///
/// ```ignore
/// use hyperlight_guest::fs::OpenOptions;
///
/// // Open for reading (explicit)
/// let file = OpenOptions::new()
///     .read(true)
///     .open("/config.txt")?;
///
/// // Open for reading (implicit - same as above)
/// let file = OpenOptions::new().open("/config.txt")?;
///
/// // Create a new file for writing (FAT only)
/// let file = OpenOptions::new()
///     .write(true)
///     .create(true)
///     .open("/data/output.txt")?;
///
/// // Open existing file for read-write (FAT only)
/// let file = OpenOptions::new()
///     .read(true)
///     .write(true)
///     .open("/data/existing.txt")?;
///
/// // Truncate existing file on open (FAT only)
/// let file = OpenOptions::new()
///     .write(true)
///     .truncate(true)
///     .open("/data/existing.txt")?;
///
/// // Create new file, fail if exists (O_CREAT | O_EXCL equivalent)
/// let file = OpenOptions::new()
///     .write(true)
///     .create_new(true)
///     .open("/data/must_not_exist.txt")?;
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct OpenOptions {
    read: bool,
    write: bool,
    create: bool,
    create_new: bool,
    truncate: bool,
}

impl OpenOptions {
    /// Create a new `OpenOptions` with all options set to false.
    ///
    /// Note: You must set at least `read(true)` or `write(true)` for a valid open.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            read: false,
            write: false,
            create: false,
            create_new: false,
            truncate: false,
        }
    }

    /// Sets the option for read access.
    ///
    /// When true, the file will be openable for reading.
    #[must_use]
    pub const fn read(mut self, read: bool) -> Self {
        self.read = read;
        self
    }

    /// Sets the option for write access.
    ///
    /// When true, the file will be openable for writing.
    /// Only supported for FAT filesystems.
    #[must_use]
    pub const fn write(mut self, write: bool) -> Self {
        self.write = write;
        self
    }

    /// Sets the option to create a new file if it doesn't exist.
    ///
    /// Only supported for FAT filesystems.
    #[must_use]
    pub const fn create(mut self, create: bool) -> Self {
        self.create = create;
        self
    }

    /// Sets the option to truncate the file to zero length on open.
    ///
    /// Requires `write(true)`. Only supported for FAT filesystems.
    #[must_use]
    pub const fn truncate(mut self, truncate: bool) -> Self {
        self.truncate = truncate;
        self
    }

    /// Sets the option to create a new file, failing if it already exists.
    ///
    /// This is equivalent to `O_CREAT | O_EXCL` in POSIX terms.
    ///
    /// # Threading Note
    ///
    /// In traditional POSIX, O_EXCL provides atomicity to prevent race conditions
    /// where two processes both see "file doesn't exist" and both try to create it.
    /// In Hyperlight guests, code execution is **single-threaded by design**, so
    /// the check-then-create sequence is inherently atomic - nothing can interleave
    /// between the existence check and the file creation.
    ///
    /// If guest multi-threading is ever added, the `GuestFatFile` type is marked
    /// `!Send + !Sync` which will cause compile errors, forcing a review of the
    /// concurrency model before any unsafe sharing can occur.
    ///
    /// Only supported for FAT filesystems.
    #[must_use]
    pub const fn create_new(mut self, create_new: bool) -> Self {
        self.create_new = create_new;
        self
    }

    /// Opens the file at the specified path with the configured options.
    ///
    /// If neither `read` nor `write` is set, defaults to `read(true)`.
    ///
    /// # Errors
    ///
    /// - [`FsError::NotInitialized`] if the filesystem hasn't been initialized
    /// - [`FsError::NotFound`] if the path doesn't exist and `create` is false
    /// - [`FsError::NotAFile`] if the path refers to a directory
    /// - [`FsError::ReadOnly`] if write/create/truncate on a read-only mount
    /// - [`FsError::InvalidArgument`] if `truncate` is set without `write`
    /// - [`FsError::AlreadyExists`] if `create_new` is set and file exists
    pub fn open(self, path: &str) -> Result<File, FsError> {
        // Validate: truncate requires write
        if self.truncate && !self.write {
            return Err(FsError::InvalidArgument);
        }

        // Validate: create_new is mutually exclusive with create and truncate
        // (create_new implies create semantics but fails if exists)
        if self.create_new && (self.create || self.truncate) {
            return Err(FsError::InvalidArgument);
        }

        // Default to read if neither read nor write specified
        let read = if !self.read && !self.write {
            true
        } else {
            self.read
        };

        open_with_options(
            path,
            read,
            self.write,
            self.create,
            self.create_new,
            self.truncate,
        )
    }
}
use super::vfs::MountBackend;

// ============================================================================
// Unified File Handle
// ============================================================================

/// An open file handle.
///
/// This is a unified type that works with both read-only memory-mapped files
/// and read-write FAT filesystem files. The VFS routes operations to the
/// appropriate backend based on the file's mount point.
///
/// Implements [`embedded_io::Read`], [`embedded_io::Seek`], and
/// [`embedded_io::Write`] (write only works for FAT files).
///
/// # Lifetime
///
/// The `'static` lifetime on the FAT variant is safe because:
/// - The guest environment is single-threaded
/// - FAT filesystems live for the program's entire lifetime
/// - The backing memory is mapped by the host before guest execution
///
/// # Example
///
/// ```ignore
/// use embedded_io::{Read, Write};
/// use hyperlight_guest::fs::{self, OpenOptions};
///
/// // Read from a read-only file
/// let mut file = fs::open("/config.json")?;
/// let mut buf = [0u8; 256];
/// let bytes_read = file.read(&mut buf)?;
///
/// // Write to a FAT file (if mounted) using OpenOptions builder
/// let mut fat_file = OpenOptions::new()
///     .read(true)
///     .write(true)
///     .create(true)
///     .open("/data/output.txt")?;
/// fat_file.write(b"Hello, FAT!")?;
/// fat_file.flush()?;
/// ```
pub enum File {
    /// Read-only memory-mapped file from the manifest.
    ReadOnly(RoFile),
    /// Read-write file on a FAT filesystem.
    ///
    /// Like [`RoFile`], this holds just a file descriptor. The actual
    /// [`GuestFatFile`](super::fat::GuestFatFile) lives in the fd table.
    Fat(FatFile),
}

impl core::fmt::Debug for File {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            File::ReadOnly(ro) => f.debug_tuple("ReadOnly").field(ro).finish(),
            File::Fat(fat) => f.debug_tuple("Fat").field(fat).finish(),
        }
    }
}

impl File {
    /// Returns true if this file is read-only.
    pub fn is_readonly(&self) -> bool {
        matches!(self, File::ReadOnly(_))
    }

    /// Returns true if this file supports writing.
    pub fn is_writable(&self) -> bool {
        match self {
            File::ReadOnly(_) => false,
            File::Fat(f) => f.can_write().unwrap_or(false),
        }
    }

    /// Get the file descriptor number (borrow, doesn't transfer ownership).
    ///
    /// Both read-only and FAT files now have file descriptors.
    pub fn fd(&self) -> RawFd {
        match self {
            File::ReadOnly(f) => f.fd(),
            File::Fat(f) => f.fd(),
        }
    }

    /// Consume the File and return the raw file descriptor.
    ///
    /// This transfers ownership of the file descriptor to the caller.
    /// The caller becomes responsible for closing it via [`free_fd()`](super::fd::free_fd).
    /// The `Drop` implementation will NOT be called.
    ///
    /// # Returns
    ///
    /// The raw file descriptor number.
    pub fn into_raw_fd(self) -> RawFd {
        match self {
            File::ReadOnly(f) => f.into_raw_fd(),
            File::Fat(f) => f.into_raw_fd(),
        }
    }

    /// Create a File from a raw file descriptor.
    ///
    /// # Safety
    ///
    /// The fd must be a valid file descriptor previously obtained from
    /// [`File::into_raw_fd()`]. The fd type (RO or FAT) must match.
    /// The caller transfers ownership to the returned `File`, which will
    /// close the fd on drop.
    ///
    /// # Arguments
    ///
    /// * `fd` - The raw file descriptor
    /// * `is_fat` - If true, creates a FAT file; if false, creates an RO file
    pub unsafe fn from_raw_fd(fd: RawFd, is_fat: bool) -> Self {
        if is_fat {
            // SAFETY: Caller guarantees fd is valid FAT fd
            File::Fat(unsafe { FatFile::from_raw_fd(fd) })
        } else {
            // SAFETY: Caller guarantees fd is valid RO fd
            File::ReadOnly(unsafe { RoFile::from_raw_fd(fd) })
        }
    }

    /// Get the current position in the file.
    pub fn position(&mut self) -> Result<u64, FsError> {
        match self {
            File::ReadOnly(f) => f.position(),
            File::Fat(f) => f.position(),
        }
    }

    /// Get the size of the file in bytes.
    pub fn size(&mut self) -> Result<u64, FsError> {
        match self {
            File::ReadOnly(f) => f.size(),
            File::Fat(f) => f.size(),
        }
    }

    /// Get the remaining bytes from current position to end of file.
    pub fn remaining(&mut self) -> Result<u64, FsError> {
        match self {
            File::ReadOnly(f) => f.remaining(),
            File::Fat(f) => f.remaining(),
        }
    }

    /// Flush any buffered data to the underlying storage.
    ///
    /// For read-only files, this is a no-op.
    /// For FAT files, this flushes the fatfs buffer.
    pub fn flush(&mut self) -> Result<(), FsError> {
        match self {
            File::ReadOnly(_) => Ok(()), // No-op for RO
            File::Fat(f) => f.flush(),
        }
    }

    /// Read the entire file into a newly allocated Vec.
    ///
    /// Seeks to the beginning first, then reads until EOF.
    pub fn read_to_vec(&mut self) -> Result<Vec<u8>, FsError> {
        use alloc::vec;

        use embedded_io::ReadExactError;

        // Get file size by seeking to end (1 seek)
        let size = self.seek(SeekFrom::End(0))? as usize;

        // Rewind to beginning for reading (1 seek)
        self.rewind()?;

        let mut buf = vec![0u8; size];
        self.read_exact(&mut buf).map_err(|e| match e {
            ReadExactError::UnexpectedEof => FsError::IoError,
            ReadExactError::Other(err) => err,
        })?;
        Ok(buf)
    }
}

impl ErrorType for File {
    type Error = FsError;
}

impl Read for File {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, Self::Error> {
        match self {
            File::ReadOnly(f) => f.read(buf),
            File::Fat(f) => f.read(buf),
        }
    }
}

impl Seek for File {
    fn seek(&mut self, pos: SeekFrom) -> Result<u64, Self::Error> {
        match self {
            File::ReadOnly(f) => f.seek(pos),
            File::Fat(f) => f.seek(pos),
        }
    }
}

impl Write for File {
    fn write(&mut self, buf: &[u8]) -> Result<usize, Self::Error> {
        match self {
            File::ReadOnly(_) => Err(FsError::ReadOnly),
            File::Fat(f) => f.write(buf),
        }
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        File::flush(self)
    }
}

// ============================================================================
// Read-Only File Handle
// ============================================================================

/// A read-only file handle for memory-mapped files.
///
/// This type is exposed through the [`File::ReadOnly`] variant. While you can
/// pattern-match to extract it, prefer using the unified [`File`] API which
/// works transparently with both read-only and FAT files.
///
/// This type must be `pub` because it appears in a public enum variant.
#[derive(Debug, PartialEq, Eq)]
pub struct RoFile {
    /// File descriptor index.
    fd: RawFd,
}

impl RoFile {
    /// Create a new RoFile from a file descriptor.
    pub(crate) fn from_fd(fd: RawFd) -> Self {
        Self { fd }
    }

    /// Get the file descriptor number (borrow, doesn't transfer ownership).
    pub fn fd(&self) -> RawFd {
        self.fd
    }

    /// Consume the RoFile and return the raw file descriptor.
    ///
    /// This transfers ownership of the file descriptor to the caller.
    /// The caller becomes responsible for closing it via [`free_fd()`](super::fd::free_fd).
    /// The `Drop` implementation will NOT be called.
    ///
    /// # Returns
    ///
    /// The raw file descriptor number.
    pub fn into_raw_fd(self) -> RawFd {
        let fd = self.fd;
        core::mem::forget(self); // Skip Drop
        fd
    }

    /// Create an RoFile from a raw file descriptor.
    ///
    /// # Safety
    ///
    /// The fd must be a valid file descriptor previously obtained from
    /// [`RoFile::into_raw_fd()`] or [`RoFile::fd()`]. The caller transfers ownership
    /// to the returned `RoFile`, which will close the fd on drop.
    pub unsafe fn from_raw_fd(fd: RawFd) -> Self {
        Self { fd }
    }

    /// Get the current position in the file.
    pub fn position(&self) -> Result<u64, FsError> {
        let entry = fd::get_ro_fd(self.fd)?;
        Ok(entry.position())
    }

    /// Get the size of the file in bytes.
    pub fn size(&self) -> Result<u64, FsError> {
        let entry = fd::get_ro_fd(self.fd)?;
        Ok(entry.size())
    }

    /// Get the remaining bytes from current position to end of file.
    pub fn remaining(&self) -> Result<u64, FsError> {
        let entry = fd::get_ro_fd(self.fd)?;
        Ok(entry.size().saturating_sub(entry.position()))
    }
}

impl Drop for RoFile {
    fn drop(&mut self) {
        // Ignore errors on close - nothing we can do about them
        let _ = fd::free_fd(self.fd);
    }
}

impl ErrorType for RoFile {
    type Error = FsError;
}

impl Read for RoFile {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, Self::Error> {
        if buf.is_empty() {
            return Ok(0);
        }

        let entry = fd::get_ro_fd(self.fd)?;

        // Calculate how many bytes we can read
        let position = entry.position();
        let size = entry.size();
        let remaining = size.saturating_sub(position) as usize;
        if remaining == 0 {
            return Ok(0); // EOF
        }

        let to_read = buf.len().min(remaining);

        // Calculate the source address in guest memory
        let src_addr = entry.guest_address() + position;

        // SAFETY: The host has mapped file data at guest_address.
        // We trust that the manifest contains valid addresses and sizes.
        // If these are corrupted then the guest may crash or read invalid data but it cannot
        // access data in the host that hasn't been explicitly mapped and we don't allow
        // dynamic allocations so this is safe.
        // The guest is single-threaded so no data races.
        unsafe {
            let src_ptr = src_addr as *const u8;
            core::ptr::copy_nonoverlapping(src_ptr, buf.as_mut_ptr(), to_read);
        }

        // Update position
        entry.set_position(position + to_read as u64);

        Ok(to_read)
    }
}

impl Seek for RoFile {
    fn seek(&mut self, pos: SeekFrom) -> Result<u64, Self::Error> {
        let entry = fd::get_ro_fd(self.fd)?;

        let position = entry.position();
        let size = entry.size();
        let new_pos: i64 = match pos {
            SeekFrom::Start(n) => n as i64,
            SeekFrom::End(n) => size as i64 + n,
            SeekFrom::Current(n) => position as i64 + n,
        };

        if new_pos < 0 {
            return Err(FsError::InvalidSeek);
        }

        // Clamp to file size (seeking past EOF is allowed but reads return 0)
        let new_pos = (new_pos as u64).min(size);
        entry.set_position(new_pos);

        Ok(new_pos)
    }
}

// ============================================================================
// FAT File Handle
// ============================================================================

/// A FAT filesystem file handle.
///
/// This type is exposed through the [`File::Fat`] variant. Like [`RoFile`],
/// it holds just a file descriptor - the actual file state lives in the
/// fd table.
///
/// This type must be `pub` because it appears in a public enum variant.
#[derive(Debug, PartialEq, Eq)]
pub struct FatFile {
    /// File descriptor index.
    fd: RawFd,
}

impl FatFile {
    /// Create a new FatFile from a file descriptor.
    pub(crate) fn from_fd(fd: RawFd) -> Self {
        Self { fd }
    }

    /// Get the file descriptor number (borrow, doesn't transfer ownership).
    pub fn fd(&self) -> RawFd {
        self.fd
    }

    /// Consume the FatFile and return the raw file descriptor.
    ///
    /// This transfers ownership of the file descriptor to the caller.
    /// The caller becomes responsible for closing it via [`free_fd()`](super::fd::free_fd).
    /// The `Drop` implementation will NOT be called.
    ///
    /// # Returns
    ///
    /// The raw file descriptor number.
    pub fn into_raw_fd(self) -> RawFd {
        let fd = self.fd;
        core::mem::forget(self); // Skip Drop
        fd
    }

    /// Create a FatFile from a raw file descriptor.
    ///
    /// # Safety
    ///
    /// The fd must be a valid FAT file descriptor previously obtained from
    /// [`FatFile::into_raw_fd()`] or [`FatFile::fd()`]. The caller transfers ownership
    /// to the returned `FatFile`, which will close the fd on drop.
    pub unsafe fn from_raw_fd(fd: RawFd) -> Self {
        Self { fd }
    }

    /// Returns true if this file is open for writing.
    pub fn can_write(&self) -> Result<bool, FsError> {
        let entry = fd::get_fat_fd(self.fd)?;
        Ok(entry.borrow().file.can_write())
    }

    /// Get the current position in the file.
    pub fn position(&mut self) -> Result<u64, FsError> {
        let entry = fd::get_fat_fd(self.fd)?;
        entry.borrow_mut().file.seek(fatfs::SeekFrom::Current(0))
    }

    /// Get the size of the file in bytes.
    pub fn size(&mut self) -> Result<u64, FsError> {
        let entry = fd::get_fat_fd(self.fd)?;
        entry.borrow_mut().file.len()
    }

    /// Get the remaining bytes from current position to end of file.
    pub fn remaining(&mut self) -> Result<u64, FsError> {
        let entry = fd::get_fat_fd(self.fd)?;
        let mut inner = entry.borrow_mut();
        let pos = inner.file.seek(fatfs::SeekFrom::Current(0))?;
        let size = inner.file.len()?;
        Ok(size.saturating_sub(pos))
    }

    /// Flush any buffered data to the underlying storage.
    pub fn flush(&mut self) -> Result<(), FsError> {
        let entry = fd::get_fat_fd(self.fd)?;
        entry.borrow_mut().file.flush()
    }
}

impl Drop for FatFile {
    fn drop(&mut self) {
        // Ignore errors on close - nothing we can do about them
        let _ = fd::free_fd(self.fd);
    }
}

impl ErrorType for FatFile {
    type Error = FsError;
}

impl Read for FatFile {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, Self::Error> {
        if buf.is_empty() {
            return Ok(0);
        }

        let entry = fd::get_fat_fd(self.fd)?;
        entry.borrow_mut().file.read(buf)
    }
}

impl Seek for FatFile {
    fn seek(&mut self, pos: SeekFrom) -> Result<u64, Self::Error> {
        let entry = fd::get_fat_fd(self.fd)?;

        // Convert embedded_io::SeekFrom to fatfs::SeekFrom
        let fatfs_pos = match pos {
            SeekFrom::Start(n) => fatfs::SeekFrom::Start(n),
            SeekFrom::End(n) => fatfs::SeekFrom::End(n),
            SeekFrom::Current(n) => fatfs::SeekFrom::Current(n),
        };

        entry.borrow_mut().file.seek(fatfs_pos)
    }
}

impl Write for FatFile {
    fn write(&mut self, buf: &[u8]) -> Result<usize, Self::Error> {
        let entry = fd::get_fat_fd(self.fd)?;
        entry.borrow_mut().file.write(buf)
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        FatFile::flush(self)
    }
}

// ============================================================================
// Public API Functions
// ============================================================================

/// Result of resolving a path through the VFS.
enum ResolvedPath {
    /// Path resolves to a read-only mount.
    ReadOnly {
        /// The full path (for inode lookup).
        full_path: String,
    },
    /// Path resolves to a FAT mount.
    Fat {
        /// Index of the mount in the VFS.
        mount_idx: usize,
        /// Path relative to the mount point.
        relative_path: String,
    },
    /// Path is a virtual parent directory of one or more mount points.
    ///
    /// This happens when a mount exists at a deeply nested path and the
    /// requested path is an ancestor. For example, if a FAT is mounted at
    /// `/root/.local/lib/python3.12/site-packages/testdir`, then
    /// `/root/.local/lib/python3.12/site-packages` resolves as VirtualDir.
    VirtualDir {
        /// The normalized absolute path.
        full_path: String,
    },
}

/// Resolve a path through the VFS to determine which backend handles it.
fn resolve_path(path: &str) -> Result<ResolvedPath, FsError> {
    let vfs = manifest::vfs()?;

    // Normalize the path first (handles relative paths, ., ..)
    let normalized = vfs.normalize_path(path)?;

    match vfs.resolve(&normalized) {
        Ok((mount_idx, relative_path)) => {
            let mount = vfs.get_mount(mount_idx).ok_or(FsError::NotFound)?;

            match mount.backend() {
                MountBackend::ReadOnly => {
                    // If the path is a non-empty relative path within a ReadOnly
                    // mount AND it's a parent prefix of a deeper mount point,
                    // treat it as a virtual directory. This handles the case where
                    // "/" catches everything but "/root" is actually a virtual
                    // parent of "/root/.local/lib/.../testdir".
                    if !relative_path.is_empty() && vfs.is_mount_parent(&normalized) {
                        Ok(ResolvedPath::VirtualDir {
                            full_path: normalized,
                        })
                    } else {
                        Ok(ResolvedPath::ReadOnly {
                            full_path: normalized,
                        })
                    }
                }
                MountBackend::Fat(_) => Ok(ResolvedPath::Fat {
                    mount_idx,
                    relative_path,
                }),
            }
        }
        Err(FsError::NotFound) => {
            // Path didn't match any mount directly. Check if it's a parent
            // prefix of a mount point (virtual directory).
            if vfs.is_mount_parent(&normalized) {
                Ok(ResolvedPath::VirtualDir {
                    full_path: normalized,
                })
            } else {
                Err(FsError::NotFound)
            }
        }
        Err(e) => Err(e),
    }
}

/// Open a file by path for reading.
///
/// Routes through the VFS to the appropriate backend (read-only or FAT).
/// For read-only files, opens for reading only. For FAT files, opens for
/// reading only (use [`OpenOptions`] for write access).
///
/// # Errors
///
/// - [`FsError::NotInitialized`] if the filesystem hasn't been initialized
/// - [`FsError::NotFound`] if the path doesn't exist
/// - [`FsError::NotAFile`] if the path refers to a directory
pub fn open(path: &str) -> Result<File, FsError> {
    open_with_options(path, true, false, false, false, false)
}

/// Internal implementation for opening files with specific options.
///
/// Use [`OpenOptions`] for the public API.
///
/// # Arguments
///
/// * `read` - Open for reading
/// * `write` - Open for writing
/// * `create` - Create file if it doesn't exist
/// * `create_new` - Create file, fail if it already exists (O_EXCL)
/// * `truncate` - Truncate file to zero length on open
fn open_with_options(
    path: &str,
    read: bool,
    write: bool,
    create: bool,
    create_new: bool,
    truncate: bool,
) -> Result<File, FsError> {
    match resolve_path(path)? {
        ResolvedPath::ReadOnly { full_path } => {
            // RO only supports read-only access
            if write || create || truncate {
                return Err(FsError::ReadOnly);
            }

            let (_idx, inode) = manifest::lookup_file(&full_path)?;

            let open_file = OpenFile::new(inode.size, inode.guest_address);

            let fd = fd::alloc_ro_fd(open_file);
            Ok(File::ReadOnly(RoFile::from_fd(fd)))
        }
        ResolvedPath::Fat {
            mount_idx,
            relative_path,
        } => {
            // SAFETY: Single-threaded guest, no other VFS refs held.
            // We resolved the path above and now need mutable access to open the file.
            let vfs = unsafe { manifest::vfs_mut()? };
            let mount = vfs.get_mount_mut(mount_idx).ok_or(FsError::NotFound)?;

            // Get mount path for tracking (before we borrow_mut the backend)
            let mount_path = mount.path().to_string();

            if let MountBackend::Fat(fat) = mount.backend_mut() {
                // Handle create_new (O_EXCL) - must fail if file exists
                // NOTE: In single-threaded guest, this check-then-create is atomic.
                // See OpenOptions::create_new() docs for threading considerations.
                let fat_file = if create_new {
                    fat.create_new(&relative_path)?
                } else {
                    fat.open(&relative_path, read, write, create, truncate)?
                };

                // Allocate fd - this moves the GuestFatFile into the fd table
                // Default flags (read-only if !write, read-write if write)
                let flags = if write { 2 } else { 0 }; // O_RDWR or O_RDONLY
                let fd = fd::alloc_fat_fd(fat_file, flags, mount_path);
                Ok(File::Fat(FatFile::from_fd(fd)))
            } else {
                // Should never happen - resolve_path already confirmed it's Fat
                Err(FsError::IoError)
            }
        }
        ResolvedPath::VirtualDir { .. } => {
            // Cannot open a virtual parent directory as a file.
            Err(FsError::NotAFile)
        }
    }
}

/// File metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stat {
    /// Size of the file in bytes (0 for directories).
    pub size: u64,
    /// Whether this is a directory.
    pub is_dir: bool,
}

/// Get file metadata without opening the file.
///
/// Routes through VFS to the appropriate backend.
///
/// # Errors
///
/// - [`FsError::NotInitialized`] if the filesystem hasn't been initialized
/// - [`FsError::NotFound`] if the path doesn't exist
pub fn stat(path: &str) -> Result<Stat, FsError> {
    match resolve_path(path)? {
        ResolvedPath::ReadOnly { full_path } => {
            let (_idx, inode) = manifest::lookup(&full_path)?;
            Ok(Stat {
                size: inode.size,
                is_dir: inode.is_dir(),
            })
        }
        ResolvedPath::Fat {
            mount_idx,
            relative_path,
        } => {
            // Handle root of mount specially
            if relative_path.is_empty() {
                return Ok(Stat {
                    size: 0,
                    is_dir: true,
                });
            }

            let vfs = manifest::vfs()?;
            let mount = vfs.get_mount(mount_idx).ok_or(FsError::NotFound)?;

            if let MountBackend::Fat(fat) = mount.backend() {
                let fat_stat = fat.stat(&relative_path)?;
                Ok(Stat {
                    size: fat_stat.size,
                    is_dir: fat_stat.is_dir,
                })
            } else {
                Err(FsError::IoError)
            }
        }
        ResolvedPath::VirtualDir { .. } => {
            // Virtual parent directory: synthesized from mount point ancestry.
            Ok(Stat {
                size: 0,
                is_dir: true,
            })
        }
    }
}

/// Directory entry returned by [`read_dir`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirEntry {
    /// Name of the entry (just the filename, not full path).
    pub name: String,
    /// Whether this entry is a directory.
    pub is_dir: bool,
    /// Size in bytes (0 for directories).
    pub size: u64,
}

/// Create a directory.
///
/// Only works on FAT mounts. Returns [`FsError::ReadOnly`] for read-only mounts.
///
/// # Errors
///
/// - [`FsError::NotInitialized`] if the filesystem hasn't been initialized
/// - [`FsError::ReadOnly`] if the path is on a read-only mount
/// - [`FsError::AlreadyExists`] if directory already exists
/// - [`FsError::NotFound`] if parent directory doesn't exist
pub fn mkdir(path: &str) -> Result<(), FsError> {
    match resolve_path(path)? {
        ResolvedPath::ReadOnly { .. } => Err(FsError::ReadOnly),
        ResolvedPath::Fat {
            mount_idx,
            relative_path,
        } => {
            // SAFETY: Single-threaded guest, no other VFS refs held.
            let vfs = unsafe { manifest::vfs_mut()? };
            let mount = vfs.get_mount_mut(mount_idx).ok_or(FsError::NotFound)?;

            if let MountBackend::Fat(fat) = mount.backend_mut() {
                fat.mkdir(&relative_path)
            } else {
                Err(FsError::IoError)
            }
        }
        ResolvedPath::VirtualDir { .. } => {
            // Virtual parent directory already exists implicitly.
            Err(FsError::AlreadyExists)
        }
    }
}

/// Remove an empty directory.
///
/// Only works on FAT mounts. Returns [`FsError::ReadOnly`] for read-only mounts.
///
/// # Errors
///
/// - [`FsError::NotInitialized`] if the filesystem hasn't been initialized
/// - [`FsError::ReadOnly`] if the path is on a read-only mount
/// - [`FsError::NotFound`] if directory doesn't exist
/// - [`FsError::NotEmpty`] if directory is not empty
/// - [`FsError::NotADirectory`] if path is a file
pub fn rmdir(path: &str) -> Result<(), FsError> {
    match resolve_path(path)? {
        ResolvedPath::ReadOnly { .. } => Err(FsError::ReadOnly),
        ResolvedPath::Fat {
            mount_idx,
            relative_path,
        } => {
            // SAFETY: Single-threaded guest, no other VFS refs held.
            let vfs = unsafe { manifest::vfs_mut()? };
            let mount = vfs.get_mount_mut(mount_idx).ok_or(FsError::NotFound)?;

            if let MountBackend::Fat(fat) = mount.backend_mut() {
                fat.rmdir(&relative_path)
            } else {
                Err(FsError::IoError)
            }
        }
        ResolvedPath::VirtualDir { .. } => {
            // Cannot remove a virtual parent directory.
            Err(FsError::ReadOnly)
        }
    }
}

/// Delete a file.
///
/// Only works on FAT mounts. Returns [`FsError::ReadOnly`] for read-only mounts.
///
/// # Errors
///
/// - [`FsError::NotInitialized`] if the filesystem hasn't been initialized
/// - [`FsError::ReadOnly`] if the path is on a read-only mount
/// - [`FsError::NotFound`] if file doesn't exist
/// - [`FsError::NotAFile`] if path is a directory
pub fn unlink(path: &str) -> Result<(), FsError> {
    match resolve_path(path)? {
        ResolvedPath::ReadOnly { .. } => Err(FsError::ReadOnly),
        ResolvedPath::Fat {
            mount_idx,
            relative_path,
        } => {
            // SAFETY: Single-threaded guest, no other VFS refs held.
            let vfs = unsafe { manifest::vfs_mut()? };
            let mount = vfs.get_mount_mut(mount_idx).ok_or(FsError::NotFound)?;

            if let MountBackend::Fat(fat) = mount.backend_mut() {
                fat.unlink(&relative_path)
            } else {
                Err(FsError::IoError)
            }
        }
        ResolvedPath::VirtualDir { .. } => {
            // Cannot unlink a virtual parent directory.
            Err(FsError::ReadOnly)
        }
    }
}

/// List the contents of a directory.
///
/// Routes through VFS to the appropriate backend.
/// Returns a vector of directory entries (direct children only, not recursive).
///
/// # Errors
///
/// - [`FsError::NotInitialized`] if the filesystem hasn't been initialized
/// - [`FsError::NotFound`] if the path doesn't exist
/// - [`FsError::NotADirectory`] if the path is a file
pub fn read_dir(path: &str) -> Result<Vec<DirEntry>, FsError> {
    use alloc::string::ToString;

    let resolved = resolve_path(path)?;

    // Determine the normalized full path for virtual children lookup.
    let full_path_for_virtual = match &resolved {
        ResolvedPath::ReadOnly { full_path } => Some(full_path.clone()),
        ResolvedPath::Fat {
            mount_idx,
            relative_path,
        } => {
            if relative_path.is_empty() {
                // At mount root - reconstruct the full path.
                let vfs = manifest::vfs()?;
                vfs.get_mount(*mount_idx).map(|m| m.path().to_string())
            } else {
                None // Inside a FAT mount, virtual children don't apply.
            }
        }
        ResolvedPath::VirtualDir { full_path } => Some(full_path.clone()),
    };

    let mut entries: Vec<DirEntry> = match resolved {
        ResolvedPath::ReadOnly { full_path } => {
            let children = manifest::list_dir(&full_path)?;

            children
                .into_iter()
                .map(|inode| {
                    let name = inode
                        .path
                        .rsplit('/')
                        .next()
                        .unwrap_or(&inode.path)
                        .to_string();

                    DirEntry {
                        name,
                        is_dir: inode.is_dir(),
                        size: inode.size,
                    }
                })
                .collect()
        }
        ResolvedPath::Fat {
            mount_idx,
            relative_path,
        } => {
            let vfs = manifest::vfs()?;
            let mount = vfs.get_mount(mount_idx).ok_or(FsError::NotFound)?;

            if let MountBackend::Fat(fat) = mount.backend() {
                // Use "." for mount root
                let fat_path = if relative_path.is_empty() {
                    "."
                } else {
                    &relative_path
                };

                let fat_entries = fat.read_dir(fat_path)?;

                fat_entries
                    .into_iter()
                    .map(|e| DirEntry {
                        name: e.name,
                        is_dir: e.is_dir,
                        size: e.size,
                    })
                    .collect()
            } else {
                return Err(FsError::IoError);
            }
        }
        ResolvedPath::VirtualDir { full_path } => {
            // Virtual parent directory: list immediate child components
            // that lead toward mount points.
            let vfs = manifest::vfs()?;
            let children = vfs.virtual_children(&full_path);

            children
                .into_iter()
                .map(|name| DirEntry {
                    name,
                    is_dir: true,
                    size: 0,
                })
                .collect()
        }
    };

    // Merge virtual children from deeper mount points.
    // For example, if we're listing "/" and there's a mount at
    // /root/.local/lib/.../testdir, then "root" should appear.
    if let Some(dir_path) = full_path_for_virtual {
        let vfs = manifest::vfs()?;
        let virtual_children = vfs.virtual_children(&dir_path);
        for child_name in virtual_children {
            if !entries.iter().any(|e| e.name == child_name) {
                entries.push(DirEntry {
                    name: child_name,
                    is_dir: true,
                    size: 0,
                });
            }
        }
    }

    Ok(entries)
}

/// Rename a file or directory.
///
/// Moves or renames a file or directory from `old_path` to `new_path`.
/// Both paths must be on the same FAT mount.
///
/// # Errors
///
/// - [`FsError::NotInitialized`] if the filesystem hasn't been initialized
/// - [`FsError::NotFound`] if `old_path` doesn't exist
/// - [`FsError::AlreadyExists`] if `new_path` already exists
/// - [`FsError::ReadOnly`] if the path is on a read-only mount
pub fn rename(old_path: &str, new_path: &str) -> Result<(), FsError> {
    let old_resolved = resolve_path(old_path)?;
    let new_resolved = resolve_path(new_path)?;

    match (old_resolved, new_resolved) {
        (
            ResolvedPath::Fat {
                mount_idx: old_idx,
                relative_path: old_rel,
            },
            ResolvedPath::Fat {
                mount_idx: new_idx,
                relative_path: new_rel,
            },
        ) => {
            // Both must be on same mount
            if old_idx != new_idx {
                return Err(FsError::InvalidPath);
            }

            let vfs = manifest::vfs()?;
            let mount = vfs.get_mount(old_idx).ok_or(FsError::NotFound)?;

            if let MountBackend::Fat(fat) = mount.backend() {
                fat.rename(&old_rel, &new_rel)
            } else {
                Err(FsError::IoError)
            }
        }
        // Read-only and virtual directory mounts don't support rename
        (ResolvedPath::ReadOnly { .. }, _)
        | (_, ResolvedPath::ReadOnly { .. })
        | (ResolvedPath::VirtualDir { .. }, _)
        | (_, ResolvedPath::VirtualDir { .. }) => Err(FsError::ReadOnly),
    }
}

/// Get the current working directory.
///
/// Returns the absolute path of the current working directory.
///
/// # Errors
///
/// - [`FsError::NotInitialized`] if the filesystem hasn't been initialized
pub fn cwd() -> Result<alloc::string::String, FsError> {
    let vfs = manifest::vfs()?;
    Ok(alloc::string::String::from(vfs.cwd()))
}

/// Change the current working directory.
///
/// Changes the current working directory to the specified path.
/// The path can be absolute or relative.
///
/// # Errors
///
/// - [`FsError::NotInitialized`] if the filesystem hasn't been initialized
/// - [`FsError::InvalidPath`] if the path is malformed
/// - [`FsError::NotFound`] if no mount handles this path
pub fn chdir(path: &str) -> Result<(), FsError> {
    // Safety: Called from single-threaded guest context
    let vfs = unsafe { manifest::vfs_mut()? };
    vfs.set_cwd(path)
}
