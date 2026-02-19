/*
Copyright 2025 The Hyperlight Authors.

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

//! C API for HyperlightFS - libc-style file operations.
//!
//! Provides POSIX-like file operations for C guests, supporting both
//! read-only memory-mapped files and read-write FAT filesystem files.
//!
//! # Error Codes
//!
//! Functions return negative values on error:
//! - `-1` (HL_ENOENT): File not found or generic error
//! - `-2` (HL_ENOTSUP): Operation not supported
//! - `-3` (HL_EROFS): Read-only filesystem (write attempted on RO file)
//! - `-4` (HL_ENOSPC): No space left on device
//! - `-5` (HL_EEXIST): File already exists
//! - `-6` (HL_ENOTDIR): Not a directory
//! - `-7` (HL_EISDIR): Is a directory (cannot open directory as file)
//! - `-8` (HL_ENOTEMPTY): Directory not empty
//! - `-9` (HL_EINVAL): Invalid argument
//! - `-11` (HL_EBADF): Bad file descriptor

use core::ffi::c_char;

use hyperlight_guest::fs::{
    self, FdEntry, FsError, OpenOptions, create_fat_mount, dup_fd, dup_fd_to, free_fd, get_fat_fd,
    get_fd_entry, is_guest_created_mount, unmount,
};

// ============================================================================
// Constants
// ============================================================================

/// SEEK_SET - seek relative to beginning of file
pub const HL_SEEK_SET: i32 = 0;
/// SEEK_CUR - seek relative to current position
pub const HL_SEEK_CUR: i32 = 1;
/// SEEK_END - seek relative to end of file
pub const HL_SEEK_END: i32 = 2;

/// O_RDONLY - open for reading only
pub const HL_O_RDONLY: i32 = 0x0000;
/// O_WRONLY - open for writing only
pub const HL_O_WRONLY: i32 = 0x0001;
/// O_RDWR - open for reading and writing
pub const HL_O_RDWR: i32 = 0x0002;
/// O_CREAT - create file if it doesn't exist
pub const HL_O_CREAT: i32 = 0x0040;
/// O_EXCL - fail if file exists (with O_CREAT)
pub const HL_O_EXCL: i32 = 0x0080;
/// O_TRUNC - truncate file to zero length
pub const HL_O_TRUNC: i32 = 0x0200;
/// O_APPEND - append mode (writes always go to end)
pub const HL_O_APPEND: i32 = 0x0400;

// Access mode mask
const O_ACCMODE: i32 = 0x0003;

// ============================================================================
// Error Codes (per spec §6.4)
// ============================================================================

/// Generic error (ENOENT, EACCES, etc.)
pub const HL_ENOENT: i32 = -1;
/// Not implemented (ENOTSUP)
pub const HL_ENOTSUP: i32 = -2;
/// Read-only filesystem (EROFS)
pub const HL_EROFS: i32 = -3;
/// No space left (ENOSPC)
pub const HL_ENOSPC: i32 = -4;
/// File exists (EEXIST)
pub const HL_EEXIST: i32 = -5;
/// Not a directory (ENOTDIR)
pub const HL_ENOTDIR: i32 = -6;
/// Is a directory (EISDIR)
pub const HL_EISDIR: i32 = -7;
/// Directory not empty (ENOTEMPTY)
pub const HL_ENOTEMPTY: i32 = -8;
/// Invalid argument (EINVAL)
pub const HL_EINVAL: i32 = -9;
/// Too many open files (EMFILE)
pub const HL_EMFILE: i32 = -10;
/// Bad file descriptor (EBADF)
pub const HL_EBADF: i32 = -11;
/// Permission denied (EACCES)
pub const HL_EACCES: i32 = -12;

/// Convert FsError to C error code.
fn fs_error_to_code(e: FsError) -> i32 {
    match e {
        FsError::NotFound => HL_ENOENT,
        FsError::NotInitialized => HL_ENOENT,
        FsError::NotAFile => HL_EISDIR,
        FsError::NotADirectory => HL_ENOTDIR,
        FsError::ReadOnly => HL_EROFS,
        FsError::AlreadyExists => HL_EEXIST,
        FsError::NotEmpty => HL_ENOTEMPTY,
        FsError::NoSpace => HL_ENOSPC,
        FsError::InvalidPath => HL_EINVAL,
        FsError::InvalidFd => HL_EBADF,
        FsError::InvalidArgument => HL_EINVAL,
        FsError::NotSupported => HL_ENOTSUP,
        FsError::InvalidSeek => HL_EINVAL,
        FsError::InvalidManifest => HL_ENOENT,
        FsError::TooManyOpenFiles => HL_EMFILE,
        FsError::IoError => HL_ENOENT,
        FsError::OutOfMemory => HL_ENOENT,
        FsError::FileLocked => HL_ENOENT,
        FsError::PlatformNotSupported => HL_ENOTSUP,
        FsError::PermissionDenied => HL_EACCES,
    }
}

// ============================================================================
// Data Structures
// ============================================================================

/// File status information (similar to struct stat).
#[repr(C)]
pub struct Stat {
    /// Size of the file in bytes.
    pub size: u64,
    /// 1 if this is a directory, 0 if a file.
    pub is_dir: i32,
    /// Reserved for future use.
    pub _reserved: [u32; 4],
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Parse a C string to a Rust &str, returning HL_EINVAL on failure.
///
/// # Safety
/// The returned `&str` borrows from the caller's string. The caller must ensure
/// the string remains valid and unmodified for the duration of the borrow.
/// This is safe in practice because guest code is single-threaded and C string
/// literals have static lifetime.
fn parse_path(path: *const c_char) -> Result<&'static str, i32> {
    if path.is_null() {
        return Err(HL_EINVAL);
    }
    // SAFETY: Caller ensures string remains valid. See function docs.
    unsafe { core::ffi::CStr::from_ptr(path).to_str() }.map_err(|_| HL_EINVAL)
}

// ============================================================================
// File Operations
// ============================================================================

/// Check if HyperlightFS is initialized.
///
/// Returns 1 if initialized, 0 otherwise.
#[unsafe(no_mangle)]
pub extern "C" fn hl_fs_initialized() -> i32 {
    if fs::is_initialized() { 1 } else { 0 }
}

/// Open a file.
///
/// # Arguments
/// * `path` - Null-terminated path string
/// * `flags` - Open flags (O_RDONLY, O_WRONLY, O_RDWR, O_CREAT, O_TRUNC, etc.)
///
/// # Returns
/// * File descriptor (>= 0) on success
/// * Negative error code on failure
///
/// # Flags
/// - `O_RDONLY` (0): Open for reading only
/// - `O_WRONLY` (1): Open for writing only (FAT only)
/// - `O_RDWR` (2): Open for reading and writing (FAT only)
/// - `O_CREAT` (0x40): Create file if it doesn't exist (FAT only)
/// - `O_EXCL` (0x80): With O_CREAT, fail if file exists (atomic create-if-not-exists)
/// - `O_TRUNC` (0x200): Truncate file to zero length (FAT only)
/// - `O_APPEND` (0x400): Append mode - file position starts at end (FAT only)
///
/// # Example
/// ```c
/// // Read-only open (works for both RO and FAT files)
/// int fd = hl_fs_open("/config.json", O_RDONLY);
///
/// // Create and write (FAT only)
/// int fd = hl_fs_open("/data/out.txt", O_WRONLY | O_CREAT | O_TRUNC);
///
/// // Create exclusively - fails if file exists
/// int fd = hl_fs_open("/data/new.txt", O_WRONLY | O_CREAT | O_EXCL);
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn hl_fs_open(path: *const c_char, flags: i32) -> i32 {
    let path_str = match parse_path(path) {
        Ok(s) => s,
        Err(e) => return e,
    };

    // Parse access mode
    let access_mode = flags & O_ACCMODE;
    let read = access_mode == HL_O_RDONLY || access_mode == HL_O_RDWR;
    let write = access_mode == HL_O_WRONLY || access_mode == HL_O_RDWR;
    let create = (flags & HL_O_CREAT) != 0;
    let exclusive = (flags & HL_O_EXCL) != 0;
    let truncate = (flags & HL_O_TRUNC) != 0;
    let append = (flags & HL_O_APPEND) != 0;

    // O_EXCL without O_CREAT is undefined per POSIX.
    // We return EINVAL to catch bugs in guest code.
    if exclusive && !create {
        return HL_EINVAL;
    }

    // O_EXCL requires O_CREAT - it means "create exclusively" (fail if exists)
    // In single-threaded guest, this is inherently atomic.
    // See OpenOptions::create_new() for threading considerations.
    let create_new = create && exclusive;

    // Build OpenOptions
    // Note: create_new is mutually exclusive with create+truncate,
    // so we only set one or the other
    let opts = if create_new {
        OpenOptions::new().read(read).write(write).create_new(true)
    } else {
        OpenOptions::new()
            .read(read)
            .write(write)
            .create(create)
            .truncate(truncate)
    };

    match opts.open(path_str) {
        Ok(file) => {
            // Both RO and FAT files now have fds allocated during open.
            // For O_APPEND on FAT files, we need to seek to end.
            // We do this via the fd table since File now just wraps an fd.
            let fd = file.fd();
            let is_fat = !file.is_readonly();

            if is_fat && append {
                // Seek to end for append mode - best effort
                if let Ok(entry) = get_fat_fd(fd) {
                    let _ = entry.borrow_mut().file.seek_raw(2, 0);
                }
            }

            // Transfer ownership to caller - no drop, they must close
            file.into_raw_fd()
        }
        Err(e) => fs_error_to_code(e),
    }
}

/// Close a file descriptor.
///
/// # Arguments
/// * `fd` - File descriptor returned by `hl_fs_open`
///
/// # Returns
/// * 0 on success
/// * Negative error code on failure
#[unsafe(no_mangle)]
pub extern "C" fn hl_fs_close(fd: i32) -> i32 {
    if fd < 0 {
        return HL_EBADF;
    }

    match free_fd(fd) {
        Ok(()) => 0,
        Err(_) => HL_EBADF,
    }
}

/// Read from a file.
///
/// # Arguments
/// * `fd` - File descriptor
/// * `buf` - Buffer to read into
/// * `count` - Maximum number of bytes to read
///
/// # Returns
/// * Number of bytes read (0 at EOF)
/// * Negative error code on failure
#[unsafe(no_mangle)]
pub extern "C" fn hl_fs_read(fd: i32, buf: *mut core::ffi::c_void, count: u64) -> i64 {
    if fd < 0 {
        return HL_EBADF as i64;
    }
    if buf.is_null() {
        return HL_EINVAL as i64;
    }
    if count == 0 {
        return 0;
    }

    // Guard against truncation on 32-bit platforms
    let count_usize: usize = match count.try_into() {
        Ok(n) => n,
        Err(_) => return HL_EINVAL as i64,
    };

    let slice = unsafe { core::slice::from_raw_parts_mut(buf as *mut u8, count_usize) };

    match get_fd_entry(fd) {
        Ok(entry) => match entry {
            FdEntry::ReadOnly(ro_file) => {
                // Read from memory-mapped file
                let position = ro_file.position();
                let size = ro_file.size();
                let available = size.saturating_sub(position);
                let to_read = core::cmp::min(count, available) as usize;

                if to_read == 0 {
                    return 0; // EOF
                }

                // Copy data from guest memory with overflow protection
                let offset = position as usize;
                let src_ptr = ro_file.guest_address() as *const u8;
                // SAFETY: guest_address and position are validated during file open.
                // Using wrapping_add to avoid UB on overflow (would just read wrong data).
                let src =
                    unsafe { core::slice::from_raw_parts(src_ptr.wrapping_add(offset), to_read) };
                slice[..to_read].copy_from_slice(src);
                ro_file.set_position(position + to_read as u64);

                to_read as i64
            }
            FdEntry::Fat(fat_entry) => {
                // Read all requested bytes, handling partial reads
                let mut total_read = 0usize;
                let mut remaining = slice;
                while !remaining.is_empty() {
                    match fat_entry.borrow_mut().file.read(remaining) {
                        Ok(0) => {
                            // EOF reached
                            break;
                        }
                        Ok(n) => {
                            total_read += n;
                            remaining = &mut remaining[n..];
                        }
                        Err(e) => {
                            if total_read > 0 {
                                // Return what we have so far
                                return total_read as i64;
                            }
                            return fs_error_to_code(e) as i64;
                        }
                    }
                }
                total_read as i64
            }
        },
        Err(_) => HL_EBADF as i64,
    }
}

/// Write to a file.
///
/// # Arguments
/// * `fd` - File descriptor
/// * `buf` - Buffer to write from
/// * `count` - Number of bytes to write
///
/// # Returns
/// * Number of bytes written
/// * Negative error code on failure (HL_EBADF if not opened for writing)
///
/// # O_APPEND Semantics
///
/// If the file was opened with O_APPEND, this function seeks to end of file
/// before each write, as required by POSIX. This ensures writes always append
/// even if lseek() was called in between.
#[unsafe(no_mangle)]
pub extern "C" fn hl_fs_write(fd: i32, buf: *const core::ffi::c_void, count: u64) -> i64 {
    if fd < 0 {
        return HL_EBADF as i64;
    }
    if buf.is_null() {
        return HL_EINVAL as i64;
    }
    if count == 0 {
        return 0;
    }

    // Guard against truncation on 32-bit platforms
    let count_usize: usize = match count.try_into() {
        Ok(n) => n,
        Err(_) => return HL_EINVAL as i64,
    };

    let slice = unsafe { core::slice::from_raw_parts(buf as *const u8, count_usize) };

    match get_fd_entry(fd) {
        Ok(entry) => match entry {
            FdEntry::ReadOnly(_) => {
                // Read-only files cannot be written - EBADF per POSIX
                // (fd not open for writing)
                HL_EBADF as i64
            }
            FdEntry::Fat(fat_entry) => {
                if !fat_entry.borrow().file.can_write() {
                    return HL_EBADF as i64; // File not opened for writing
                }

                // POSIX O_APPEND: "Before each write, the file offset shall be
                // set to the end of the file." We must do this on EVERY write,
                // not just on open, to handle lseek() calls in between.
                if fat_entry.is_append() {
                    // Seek to end; ignore errors (best effort)
                    let _ = fat_entry.borrow_mut().file.seek_raw(2, 0); // SEEK_END, offset 0
                }

                // Write all bytes, handling partial writes
                let mut total_written = 0usize;
                let mut remaining = slice;
                while !remaining.is_empty() {
                    match fat_entry.borrow_mut().file.write(remaining) {
                        Ok(0) => {
                            // No progress - probably out of space
                            if total_written > 0 {
                                return total_written as i64;
                            }
                            return HL_ENOSPC as i64;
                        }
                        Ok(n) => {
                            total_written += n;
                            remaining = &remaining[n..];
                        }
                        Err(e) => {
                            if total_written > 0 {
                                return total_written as i64;
                            }
                            return fs_error_to_code(e) as i64;
                        }
                    }
                }
                total_written as i64
            }
        },
        Err(_) => HL_EBADF as i64,
    }
}

/// Seek to a position in the file.
///
/// # Arguments
/// * `fd` - File descriptor
/// * `offset` - Offset in bytes
/// * `whence` - HL_SEEK_SET (0), HL_SEEK_CUR (1), or HL_SEEK_END (2)
///
/// # Returns
/// * New position in file on success
/// * Negative error code on failure
#[unsafe(no_mangle)]
pub extern "C" fn hl_fs_lseek(fd: i32, offset: i64, whence: i32) -> i64 {
    if fd < 0 {
        return HL_EBADF as i64;
    }

    match get_fd_entry(fd) {
        Ok(entry) => match entry {
            FdEntry::ReadOnly(ro_file) => {
                // Calculate new position for read-only files
                let position = ro_file.position();
                let size = ro_file.size();
                let new_pos = match whence {
                    HL_SEEK_SET => offset,
                    HL_SEEK_CUR => position as i64 + offset,
                    HL_SEEK_END => size as i64 + offset,
                    _ => return HL_EINVAL as i64,
                };

                if new_pos < 0 {
                    return HL_EINVAL as i64;
                }

                ro_file.set_position(new_pos as u64);
                new_pos
            }
            FdEntry::Fat(fat_entry) => {
                // Use seek_raw which accepts whence/offset directly
                match fat_entry.borrow_mut().file.seek_raw(whence, offset) {
                    Ok(pos) => pos as i64,
                    Err(e) => fs_error_to_code(e) as i64,
                }
            }
        },
        Err(_) => HL_EBADF as i64,
    }
}

/// Get file status by file descriptor.
///
/// # Arguments
/// * `fd` - File descriptor
/// * `stat` - Pointer to Stat struct to fill
///
/// # Returns
/// * 0 on success
/// * Negative error code on failure
#[unsafe(no_mangle)]
pub extern "C" fn hl_fs_fstat(fd: i32, stat: *mut Stat) -> i32 {
    if fd < 0 {
        return HL_EBADF;
    }
    if stat.is_null() {
        return HL_EINVAL;
    }

    match get_fd_entry(fd) {
        Ok(entry) => {
            let (size, is_dir) = match entry {
                FdEntry::ReadOnly(ro_file) => (ro_file.size(), false),
                FdEntry::Fat(fat_entry) => {
                    match fat_entry.borrow_mut().file.len() {
                        Ok(len) => (len, false), // Open files are always files, not directories
                        Err(e) => return fs_error_to_code(e),
                    }
                }
            };

            unsafe {
                (*stat).size = size;
                (*stat).is_dir = if is_dir { 1 } else { 0 };
                (*stat)._reserved = [0; 4];
            }
            0
        }
        Err(_) => HL_EBADF,
    }
}

/// Get file status by path.
///
/// # Arguments
/// * `path` - Null-terminated path string
/// * `stat` - Pointer to Stat struct to fill
///
/// # Returns
/// * 0 on success
/// * Negative error code on failure
#[unsafe(no_mangle)]
pub extern "C" fn hl_fs_stat(path: *const c_char, stat: *mut Stat) -> i32 {
    if stat.is_null() {
        return HL_EINVAL;
    }

    let path_str = match parse_path(path) {
        Ok(s) => s,
        Err(e) => return e,
    };

    match fs::stat(path_str) {
        Ok(s) => {
            unsafe {
                (*stat).size = s.size;
                (*stat).is_dir = if s.is_dir { 1 } else { 0 };
                (*stat)._reserved = [0; 4];
            }
            0
        }
        Err(e) => fs_error_to_code(e),
    }
}

/// Delete a file.
///
/// # Arguments
/// * `path` - Null-terminated path string
///
/// # Returns
/// * 0 on success
/// * Negative error code on failure (HL_EROFS for read-only mounts)
#[unsafe(no_mangle)]
pub extern "C" fn hl_fs_unlink(path: *const c_char) -> i32 {
    let path_str = match parse_path(path) {
        Ok(s) => s,
        Err(e) => return e,
    };

    match fs::unlink(path_str) {
        Ok(()) => 0,
        Err(e) => fs_error_to_code(e),
    }
}

/// Rename a file or directory.
///
/// # Arguments
/// * `oldpath` - Current path (null-terminated)
/// * `newpath` - New path (null-terminated)
///
/// # Returns
/// * 0 on success
/// * Negative error code on failure
#[unsafe(no_mangle)]
pub extern "C" fn hl_fs_rename(oldpath: *const c_char, newpath: *const c_char) -> i32 {
    let old_str = match parse_path(oldpath) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let new_str = match parse_path(newpath) {
        Ok(s) => s,
        Err(e) => return e,
    };

    match fs::rename(old_str, new_str) {
        Ok(()) => 0,
        Err(e) => fs_error_to_code(e),
    }
}

/// Create a directory.
///
/// # Arguments
/// * `path` - Null-terminated path string
///
/// # Returns
/// * 0 on success
/// * Negative error code on failure
#[unsafe(no_mangle)]
pub extern "C" fn hl_fs_mkdir(path: *const c_char) -> i32 {
    let path_str = match parse_path(path) {
        Ok(s) => s,
        Err(e) => return e,
    };

    match fs::mkdir(path_str) {
        Ok(()) => 0,
        Err(e) => fs_error_to_code(e),
    }
}

/// Remove an empty directory.
///
/// # Arguments
/// * `path` - Null-terminated path string
///
/// # Returns
/// * 0 on success
/// * Negative error code on failure (HL_ENOTEMPTY if not empty)
#[unsafe(no_mangle)]
pub extern "C" fn hl_fs_rmdir(path: *const c_char) -> i32 {
    let path_str = match parse_path(path) {
        Ok(s) => s,
        Err(e) => return e,
    };

    match fs::rmdir(path_str) {
        Ok(()) => 0,
        Err(e) => fs_error_to_code(e),
    }
}

/// Get current working directory.
///
/// # Arguments
/// * `buf` - Buffer to store path
/// * `size` - Size of buffer
///
/// # Returns
/// * Pointer to buf on success
/// * NULL on failure
#[unsafe(no_mangle)]
pub extern "C" fn hl_fs_getcwd(buf: *mut c_char, size: u64) -> *mut c_char {
    if buf.is_null() || size == 0 {
        return core::ptr::null_mut();
    }

    let cwd = match fs::cwd() {
        Ok(s) => s,
        Err(_) => return core::ptr::null_mut(),
    };
    let cwd_bytes = cwd.as_bytes();

    // Need space for path + null terminator
    if cwd_bytes.len() + 1 > size as usize {
        return core::ptr::null_mut();
    }

    unsafe {
        core::ptr::copy_nonoverlapping(cwd_bytes.as_ptr(), buf as *mut u8, cwd_bytes.len());
        *buf.add(cwd_bytes.len()) = 0; // Null terminator
    }

    buf
}

/// Change current working directory.
///
/// # Arguments
/// * `path` - Null-terminated path string
///
/// # Returns
/// * 0 on success
/// * Negative error code on failure
#[unsafe(no_mangle)]
pub extern "C" fn hl_fs_chdir(path: *const c_char) -> i32 {
    let path_str = match parse_path(path) {
        Ok(s) => s,
        Err(e) => return e,
    };

    match fs::chdir(path_str) {
        Ok(()) => 0,
        Err(e) => fs_error_to_code(e),
    }
}

// ============================================================================
// Directory Entry Structure for C
// ============================================================================

/// Maximum filename length in directory entry.
const MAX_DIRENT_NAME_LEN: usize = 256;

/// Directory entry structure for C code.
///
/// This is designed for iterative reading via `hl_fs_readdir_entry`.
#[repr(C)]
pub struct hl_DirEntry {
    /// Filename (null-terminated, max 255 chars + null).
    pub name: [c_char; MAX_DIRENT_NAME_LEN],
    /// Whether this is a directory.
    pub is_dir: bool,
    /// File size in bytes (0 for directories).
    pub size: u64,
}

/// Read all directory entries as newline-separated names.
///
/// This is a simple interface for listing directories in C.
/// Returns a buffer containing entry names separated by newlines.
///
/// # Arguments
/// * `path` - Null-terminated path to directory
/// * `buf` - Buffer to store the result
/// * `buf_size` - Size of buffer
///
/// # Returns
/// * Number of bytes written (not including null terminator) on success
/// * Negative error code on failure
///
/// # Safety
/// Caller must ensure `buf` points to valid memory of at least `buf_size` bytes.
#[unsafe(no_mangle)]
pub extern "C" fn hl_fs_readdir(path: *const c_char, buf: *mut c_char, buf_size: u64) -> i64 {
    let path_str = match parse_path(path) {
        Ok(s) => s,
        Err(e) => return e as i64,
    };

    if buf.is_null() || buf_size == 0 {
        return HL_EINVAL as i64;
    }

    // Guard against truncation on 32-bit platforms
    let buf_size_usize: usize = match buf_size.try_into() {
        Ok(n) => n,
        Err(_) => return HL_EINVAL as i64,
    };

    // Read directory entries
    let entries = match fs::read_dir(path_str) {
        Ok(e) => e,
        Err(e) => return fs_error_to_code(e) as i64,
    };

    // Build newline-separated string
    let mut result = alloc::string::String::new();
    for (i, entry) in entries.iter().enumerate() {
        if i > 0 {
            result.push('\n');
        }
        result.push_str(&entry.name);
    }

    let result_bytes = result.as_bytes();

    // Need space for content + null terminator
    if result_bytes.len() + 1 > buf_size_usize {
        // Buffer too small - write what we can but return error
        // This indicates truncation occurred; caller should use larger buffer
        // or use opendir/readdir_entry for iterative reading.
        let write_len = buf_size_usize - 1;
        unsafe {
            core::ptr::copy_nonoverlapping(result_bytes.as_ptr(), buf as *mut u8, write_len);
            *buf.add(write_len) = 0; // Null terminator
        }
        // Return EINVAL to indicate buffer was too small (data truncated)
        return HL_EINVAL as i64;
    }

    // Write full content
    unsafe {
        core::ptr::copy_nonoverlapping(result_bytes.as_ptr(), buf as *mut u8, result_bytes.len());
        *buf.add(result_bytes.len()) = 0; // Null terminator
    }

    result_bytes.len() as i64
}

// ============================================================================
// Additional Constants (per spec §6.1)
// ============================================================================

/// AT_FDCWD - use current working directory for *at() functions
pub const HL_AT_FDCWD: i32 = -100;

/// F_DUPFD - duplicate file descriptor
pub const HL_F_DUPFD: i32 = 0;
/// F_GETFD - get file descriptor flags
pub const HL_F_GETFD: i32 = 1;
/// F_SETFD - set file descriptor flags
pub const HL_F_SETFD: i32 = 2;
/// F_GETFL - get file status flags
pub const HL_F_GETFL: i32 = 3;
/// F_SETFL - set file status flags
pub const HL_F_SETFL: i32 = 4;

/// R_OK - test for read permission
pub const HL_R_OK: i32 = 4;
/// W_OK - test for write permission
pub const HL_W_OK: i32 = 2;
/// X_OK - test for execute permission (always fails - no exec in HyperlightFS)
pub const HL_X_OK: i32 = 1;
/// F_OK - test for file existence
pub const HL_F_OK: i32 = 0;

/// DT_UNKNOWN - unknown type
pub const HL_DT_UNKNOWN: u8 = 0;
/// DT_DIR - directory
pub const HL_DT_DIR: u8 = 4;
/// DT_REG - regular file
pub const HL_DT_REG: u8 = 8;

// ============================================================================
// Directory Iteration (POSIX-style opendir/readdir/closedir)
// ============================================================================

/// Maximum number of open directory streams.
const MAX_DIR_STREAMS: usize = 16;

/// Synthetic inode number returned in dirent (we don't track real inodes).
const SYNTHETIC_INODE: u64 = 1;

/// Directory entry structure for C code.
#[repr(C)]
pub struct hl_dirent_t {
    /// Inode number (synthetic - always 1).
    pub d_ino: u64,
    /// Entry type (DT_REG or DT_DIR).
    pub d_type: u8,
    /// Entry name (null-terminated).
    pub d_name: [c_char; 256],
}

/// Directory stream state.
struct DirStream {
    entries: alloc::vec::Vec<fs::DirEntry>,
    index: usize,
    current_dirent: hl_dirent_t,
}

/// Directory stream table wrapper.
///
/// SAFETY: Guest code is single-threaded, so this is safe.
struct DirStreamTable(core::cell::UnsafeCell<[Option<DirStream>; MAX_DIR_STREAMS]>);

// SAFETY: Guest is single-threaded.
unsafe impl Sync for DirStreamTable {}

impl DirStreamTable {
    const fn new() -> Self {
        Self(core::cell::UnsafeCell::new(
            [const { None }; MAX_DIR_STREAMS],
        ))
    }

    #[allow(clippy::mut_from_ref)] // SAFETY: Guest is single-threaded, interior mutability is intentional
    fn get(&self) -> &mut [Option<DirStream>; MAX_DIR_STREAMS] {
        // SAFETY: Guest is single-threaded.
        unsafe { &mut *self.0.get() }
    }
}

/// Global directory stream table.
static DIR_STREAMS: DirStreamTable = DirStreamTable::new();

/// Opaque directory stream handle.
pub type hl_DIR = core::ffi::c_void;

/// Open a directory for reading.
///
/// Returns pointer to directory stream on success, NULL on error.
#[unsafe(no_mangle)]
pub extern "C" fn hl_fs_opendir(path: *const c_char) -> *mut hl_DIR {
    let path_str = match parse_path(path) {
        Ok(s) => s,
        Err(_) => return core::ptr::null_mut(),
    };

    // Read all entries
    let entries = match fs::read_dir(path_str) {
        Ok(e) => e,
        Err(_) => return core::ptr::null_mut(),
    };

    // Find a free slot
    let streams = DIR_STREAMS.get();
    for (i, slot) in streams.iter_mut().enumerate() {
        if slot.is_none() {
            *slot = Some(DirStream {
                entries,
                index: 0,
                current_dirent: hl_dirent_t {
                    d_ino: 0,
                    d_type: 0,
                    d_name: [0; 256],
                },
            });
            // Return index + 1 as handle (0 would be NULL)
            return (i + 1) as *mut hl_DIR;
        }
    }

    core::ptr::null_mut() // No free slots
}

/// Read next directory entry.
///
/// Returns pointer to dirent on success, NULL at end or on error.
#[unsafe(no_mangle)]
pub extern "C" fn hl_fs_readdir_entry(dirp: *mut hl_DIR) -> *mut hl_dirent_t {
    if dirp.is_null() {
        return core::ptr::null_mut();
    }

    let handle = dirp as usize;
    if handle == 0 || handle > MAX_DIR_STREAMS {
        return core::ptr::null_mut();
    }

    let streams = DIR_STREAMS.get();
    let stream = match streams[handle - 1].as_mut() {
        Some(s) => s,
        None => return core::ptr::null_mut(),
    };

    if stream.index >= stream.entries.len() {
        return core::ptr::null_mut(); // End of directory
    }

    let entry = &stream.entries[stream.index];
    stream.index += 1;

    // Fill in dirent
    stream.current_dirent.d_ino = SYNTHETIC_INODE;
    stream.current_dirent.d_type = if entry.is_dir { HL_DT_DIR } else { HL_DT_REG };

    // Copy name (leave room for null terminator)
    let name_bytes = entry.name.as_bytes();
    let max_name_len = stream.current_dirent.d_name.len() - 1; // Reserve space for null
    let copy_len = core::cmp::min(name_bytes.len(), max_name_len);
    for (i, &b) in name_bytes[..copy_len].iter().enumerate() {
        stream.current_dirent.d_name[i] = b as c_char;
    }
    stream.current_dirent.d_name[copy_len] = 0; // Null terminator

    &mut stream.current_dirent as *mut hl_dirent_t
}

/// Close a directory stream.
///
/// Returns 0 on success, -1 on error.
#[unsafe(no_mangle)]
pub extern "C" fn hl_fs_closedir(dirp: *mut hl_DIR) -> i32 {
    if dirp.is_null() {
        return HL_EBADF; // POSIX: invalid directory stream
    }

    let handle = dirp as usize;
    if handle == 0 || handle > MAX_DIR_STREAMS {
        return HL_EBADF;
    }

    let streams = DIR_STREAMS.get();
    if streams[handle - 1].is_none() {
        return HL_EBADF;
    }

    streams[handle - 1] = None;
    0
}

// ============================================================================
// openat / mkdirat - Directory-relative operations
// ============================================================================

/// Check if an `*at()` operation can proceed with the given dirfd and path.
///
/// Returns:
/// - `Ok(true)` if path is absolute or dirfd is AT_FDCWD (use relative path)
/// - `Ok(false)` should not happen (reserved for future dirfd support)
/// - `Err(code)` if dirfd is invalid or unsupported
fn check_at_path(dirfd: i32, path_str: &str) -> Result<(), i32> {
    // Absolute paths ignore dirfd entirely
    if path_str.starts_with('/') {
        return Ok(());
    }

    // AT_FDCWD means use current working directory
    if dirfd == HL_AT_FDCWD {
        return Ok(());
    }

    // Other dirfd values would require resolving the directory path from fd.
    // This is not currently supported.
    Err(HL_ENOTSUP)
}

/// Open a file relative to a directory fd.
///
/// # Arguments
/// * `dirfd` - Directory fd or AT_FDCWD for current directory
/// * `path` - Path (absolute or relative to dirfd)
/// * `flags` - Open flags
///
/// # Returns
/// * fd >= 0 on success
/// * Negative error code on failure
#[unsafe(no_mangle)]
pub extern "C" fn hl_fs_openat(dirfd: i32, path: *const c_char, flags: i32) -> i32 {
    let path_str = match parse_path(path) {
        Ok(s) => s,
        Err(e) => return e,
    };

    if let Err(e) = check_at_path(dirfd, path_str) {
        return e;
    }

    hl_fs_open(path, flags)
}

/// Create a directory relative to a directory fd.
///
/// # Arguments
/// * `dirfd` - Directory fd or AT_FDCWD for current directory
/// * `path` - Path (absolute or relative to dirfd)
///
/// # Returns
/// * 0 on success
/// * Negative error code on failure
#[unsafe(no_mangle)]
pub extern "C" fn hl_fs_mkdirat(dirfd: i32, path: *const c_char) -> i32 {
    let path_str = match parse_path(path) {
        Ok(s) => s,
        Err(e) => return e,
    };

    if let Err(e) = check_at_path(dirfd, path_str) {
        return e;
    }

    hl_fs_mkdir(path)
}

// ============================================================================
// fcntl - File descriptor control
// ============================================================================

/// File descriptor control.
///
/// # Arguments
/// * `fd` - File descriptor
/// * `cmd` - Command (F_DUPFD, F_GETFD, F_SETFD, F_GETFL, F_SETFL)
/// * `arg` - Command-specific argument
///
/// # Returns
/// * Command-specific value on success
/// * Negative error code on failure
///
/// # F_GETFL
///
/// Returns the original flags passed to open(), including O_APPEND if set.
/// For RO files, always returns O_RDONLY.
///
/// # F_SETFL
///
/// Per POSIX, only O_APPEND, O_NONBLOCK, and O_ASYNC can be changed.
/// We only support changing O_APPEND. Other flags in arg are ignored.
#[unsafe(no_mangle)]
pub extern "C" fn hl_fs_fcntl(fd: i32, cmd: i32, arg: i32) -> i32 {
    if fd < 0 {
        return HL_EBADF;
    }

    // Get fd entry once and reuse for all commands that need it
    let entry = match get_fd_entry(fd) {
        Ok(e) => e,
        Err(_) => return HL_EBADF,
    };

    match cmd {
        HL_F_DUPFD => {
            // Duplicate fd to lowest available >= arg (POSIX F_DUPFD semantics)
            if arg < 0 {
                return HL_EINVAL;
            }
            match dup_fd_to(fd, None, Some(arg)) {
                Ok(newfd) => newfd,
                Err(e) => fs_error_to_code(e),
            }
        }
        HL_F_GETFD => {
            // Get fd flags (we don't support close-on-exec, return 0)
            0
        }
        HL_F_SETFD => {
            // Set fd flags (ignored - no exec in hyperlight)
            0
        }
        HL_F_GETFL => {
            // Get file status flags - return the original open() flags
            match entry {
                FdEntry::ReadOnly(_) => HL_O_RDONLY,
                FdEntry::Fat(fat_entry) => fat_entry.flags(),
            }
        }
        HL_F_SETFL => {
            // Set file status flags
            // Per POSIX, only O_APPEND, O_NONBLOCK, O_ASYNC can be modified.
            // We only support O_APPEND (0x0400). Other flags are ignored.
            match entry {
                FdEntry::ReadOnly(_) => {
                    // Can't modify flags on read-only files
                    0
                }
                FdEntry::Fat(fat_entry) => {
                    // Update O_APPEND based on arg
                    let new_append = (arg & HL_O_APPEND) != 0;
                    fat_entry.set_append(new_append);
                    0
                }
            }
        }
        _ => HL_EINVAL,
    }
}

// ============================================================================
// dup / dup2 - Duplicate file descriptors
// ============================================================================

/// Duplicate a file descriptor.
///
/// Creates a new file descriptor that refers to the same open file.
/// Both RO and FAT files share the same file position (POSIX semantics).
///
/// # Arguments
/// * `oldfd` - File descriptor to duplicate
///
/// # Returns
/// * New fd on success (lowest available fd >= 3)
/// * Negative error code on failure
#[unsafe(no_mangle)]
pub extern "C" fn hl_fs_dup(oldfd: i32) -> i32 {
    if oldfd < 0 {
        return HL_EBADF;
    }

    match dup_fd(oldfd) {
        Ok(newfd) => newfd,
        Err(e) => fs_error_to_code(e),
    }
}

/// Duplicate a file descriptor to a specific fd.
///
/// If newfd is already open, it is closed first (atomically).
/// Both RO and FAT files share the same file position (POSIX semantics).
///
/// # Arguments
/// * `oldfd` - File descriptor to duplicate
/// * `newfd` - Target file descriptor (must be >= 3)
///
/// # Returns
/// * newfd on success
/// * Negative error code on failure
#[unsafe(no_mangle)]
pub extern "C" fn hl_fs_dup2(oldfd: i32, newfd: i32) -> i32 {
    if oldfd < 0 || newfd < 0 {
        return HL_EBADF;
    }

    // POSIX: if oldfd == newfd, just validate oldfd and return
    if oldfd == newfd {
        if get_fd_entry(oldfd).is_err() {
            return HL_EBADF;
        }
        return newfd;
    }

    match dup_fd_to(oldfd, Some(newfd), None) {
        Ok(fd) => fd,
        Err(e) => fs_error_to_code(e),
    }
}

// ============================================================================
// access - Check file accessibility
// ============================================================================

/// Check file accessibility.
///
/// # Arguments
/// * `path` - Path to check
/// * `mode` - Access mode (R_OK, W_OK, X_OK, F_OK)
///
/// # Returns
/// * 0 if accessible
/// * Negative error code on failure
#[unsafe(no_mangle)]
pub extern "C" fn hl_fs_access(path: *const c_char, mode: i32) -> i32 {
    let path_str = match parse_path(path) {
        Ok(s) => s,
        Err(e) => return e,
    };

    // Get stat to check existence
    let stat_result = fs::stat(path_str);

    match stat_result {
        Ok(_stat) => {
            // F_OK - just existence check
            if mode == HL_F_OK {
                return 0;
            }

            // X_OK - execute permission never granted (no exec in HyperlightFS)
            if (mode & HL_X_OK) != 0 {
                return HL_EACCES;
            }

            // R_OK - read permission (always granted for existing files)
            // W_OK - write permission: try to detect by attempting open
            if (mode & HL_W_OK) != 0 {
                // Try to open for writing - if it fails with EROFS, no write access
                let test_result = fs::OpenOptions::new().write(true).open(path_str);
                match test_result {
                    Ok(file) => {
                        drop(file); // Close the test file to avoid FD leak
                        return 0; // Writable
                    }
                    Err(FsError::ReadOnly) => return HL_EROFS,
                    Err(FsError::NotAFile) => return 0, // Directory - can't write directly anyway
                    Err(_) => return 0,                 // Other errors, assume accessible
                }
            }

            0 // R_OK is always granted for existing files
        }
        Err(e) => fs_error_to_code(e),
    }
}

// ============================================================================
// Guest-Created FAT Mounts - Dynamic filesystem creation
// ============================================================================

/// Create a guest-side FAT filesystem mount.
///
/// Allocates memory from the guest heap and creates a FAT filesystem
/// that can be used for temporary storage. The mount is private to the
/// guest and not accessible to the host.
///
/// # Arguments
/// * `path` - Mount point path (must be absolute, e.g., "/scratch")
/// * `size_bytes` - Size of the FAT filesystem in bytes (min 64KB, max 16MB)
///
/// # Returns
/// * 0 on success
/// * Negative error code on failure:
///   - HL_EINVAL: Invalid path or size
///   - HL_EEXIST: Mount point already exists
///   - HL_ENOSPC: Not enough memory for allocation
///
/// # Example (C)
/// ```c
/// if (hl_fs_create_fat_mount("/scratch", 128 * 1024) == 0) {
///     // Mount created successfully
///     int fd = open("/scratch/temp.txt", O_CREAT | O_RDWR);
///     // ... use the filesystem ...
///     hl_fs_unmount_fat("/scratch");
/// }
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn hl_fs_create_fat_mount(path: *const c_char, size_bytes: i64) -> i32 {
    let path_str = match parse_path(path) {
        Ok(s) => s,
        Err(e) => return e,
    };

    match create_fat_mount(path_str, size_bytes as usize) {
        Ok(()) => 0,
        Err(e) => fs_error_to_code(e),
    }
}

/// Unmount a guest-created FAT filesystem.
///
/// Frees the memory associated with a guest-created FAT mount.
/// Only mounts created by `hl_fs_create_fat_mount` can be unmounted;
/// attempting to unmount a host-provided mount will fail.
///
/// # Arguments
/// * `path` - Mount point path to unmount
///
/// # Returns
/// * 0 on success
/// * Negative error code on failure:
///   - HL_ENOENT: Mount point not found, or files still open on this mount
///   - HL_EACCES: Cannot unmount host-provided mount
///
/// # Example (C)
/// ```c
/// hl_fs_unmount_fat("/scratch");  // Free the mount
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn hl_fs_unmount_fat(path: *const c_char) -> i32 {
    let path_str = match parse_path(path) {
        Ok(s) => s,
        Err(e) => return e,
    };

    match unmount(path_str) {
        Ok(()) => 0,
        Err(e) => fs_error_to_code(e),
    }
}

/// Check if a mount point was created by the guest.
///
/// Returns whether the specified mount point was created dynamically
/// by the guest using `hl_fs_create_fat_mount`, as opposed to being
/// provided by the host.
///
/// # Arguments
/// * `path` - Mount point path to check
///
/// # Returns
/// * 1 if the mount was guest-created
/// * 0 if the mount is host-provided or doesn't exist
///
/// # Example (C)
/// ```c
/// if (hl_fs_is_guest_created_mount("/scratch")) {
///     // Safe to unmount
///     hl_fs_unmount_fat("/scratch");
/// }
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn hl_fs_is_guest_created_mount(path: *const c_char) -> i32 {
    let path_str = match parse_path(path) {
        Ok(s) => s,
        Err(_) => return 0,
    };

    if is_guest_created_mount(path_str) {
        1
    } else {
        0
    }
}
