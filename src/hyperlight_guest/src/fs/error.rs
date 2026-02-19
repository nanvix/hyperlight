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

//! Error types for the HyperlightFS guest VFS.

use core::fmt;

use embedded_io::ErrorKind;

/// Filesystem errors.
///
/// Maps to C error codes per spec §11.2:
/// - `-1`: Generic error (NotFound, IoError, etc.)
/// - `-2`: Not supported (NotSupported, PlatformNotSupported)
/// - `-3`: Read-only (ReadOnly)
/// - `-4`: No space (NoSpace)
/// - `-5`: Already exists (AlreadyExists)
/// - `-6`: Not a directory (NotADirectory)
/// - `-7`: Is a directory (NotAFile)
/// - `-8`: Not empty (NotEmpty)
/// - `-9`: Invalid argument (InvalidPath, InvalidSeek)
/// - `-10`: Too many open files (TooManyOpenFiles)
/// - `-11`: Invalid file descriptor (InvalidFd)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsError {
    /// File or directory not found.
    NotFound,
    /// Path refers to a directory, not a file.
    NotAFile,
    /// Path refers to a file, not a directory.
    NotADirectory,
    /// Invalid file descriptor.
    InvalidFd,
    /// Invalid path (empty, contains null bytes, etc.).
    InvalidPath,
    /// Filesystem not initialized.
    NotInitialized,
    /// Seek to invalid position.
    InvalidSeek,
    /// Manifest parsing failed.
    InvalidManifest,
    /// Path is read-only (cannot write to RO file/mount).
    ReadOnly,
    /// File or directory already exists.
    AlreadyExists,
    /// Directory is not empty.
    NotEmpty,
    /// No space left on device.
    NoSpace,
    /// Too many open files.
    TooManyOpenFiles,
    /// Operation not supported.
    NotSupported,
    /// Invalid argument.
    InvalidArgument,
    /// I/O error.
    IoError,
    /// Out of memory.
    OutOfMemory,
    /// Resource is in use and cannot be freed (e.g., unmount with open files).
    FileLocked,
    /// Platform does not support this operation.
    PlatformNotSupported,
    /// Permission denied (cannot perform operation on this resource).
    PermissionDenied,
}

impl FsError {
    /// Convert to C-style error code per spec §11.2.
    #[inline]
    pub fn to_c_error(self) -> i32 {
        match self {
            FsError::NotFound => -1,
            FsError::NotAFile => -7,             // EISDIR
            FsError::NotADirectory => -6,        // ENOTDIR
            FsError::InvalidFd => -11,           // EBADF
            FsError::InvalidPath => -9,          // EINVAL
            FsError::NotInitialized => -1,       // Generic
            FsError::InvalidSeek => -9,          // EINVAL
            FsError::InvalidManifest => -1,      // Generic (EIO)
            FsError::ReadOnly => -3,             // EROFS
            FsError::AlreadyExists => -5,        // EEXIST
            FsError::NotEmpty => -8,             // ENOTEMPTY
            FsError::NoSpace => -4,              // ENOSPC
            FsError::TooManyOpenFiles => -10,    // EMFILE
            FsError::NotSupported => -2,         // ENOTSUP
            FsError::InvalidArgument => -9,      // EINVAL
            FsError::IoError => -1,              // Generic (EIO)
            FsError::OutOfMemory => -1,          // Generic (ENOMEM)
            FsError::FileLocked => -1,           // Generic (EAGAIN)
            FsError::PlatformNotSupported => -2, // ENOTSUP
            FsError::PermissionDenied => -12,    // EACCES
        }
    }
}

impl fmt::Display for FsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FsError::NotFound => write!(f, "file or directory not found"),
            FsError::NotAFile => write!(f, "path is a directory, not a file"),
            FsError::NotADirectory => write!(f, "path is a file, not a directory"),
            FsError::InvalidFd => write!(f, "invalid file descriptor"),
            FsError::InvalidPath => write!(f, "invalid path"),
            FsError::NotInitialized => write!(f, "filesystem not initialized"),
            FsError::InvalidSeek => write!(f, "invalid seek position"),
            FsError::InvalidManifest => write!(f, "invalid filesystem manifest"),
            FsError::ReadOnly => write!(f, "read-only filesystem"),
            FsError::AlreadyExists => write!(f, "file or directory already exists"),
            FsError::NotEmpty => write!(f, "directory not empty"),
            FsError::NoSpace => write!(f, "no space left on device"),
            FsError::TooManyOpenFiles => write!(f, "too many open files"),
            FsError::NotSupported => write!(f, "operation not supported"),
            FsError::InvalidArgument => write!(f, "invalid argument"),
            FsError::IoError => write!(f, "I/O error"),
            FsError::OutOfMemory => write!(f, "out of memory"),
            FsError::FileLocked => write!(f, "file is locked"),
            FsError::PlatformNotSupported => write!(f, "platform not supported"),
            FsError::PermissionDenied => write!(f, "permission denied"),
        }
    }
}

impl core::error::Error for FsError {}

impl embedded_io::Error for FsError {
    fn kind(&self) -> ErrorKind {
        match self {
            FsError::NotFound => ErrorKind::NotFound,
            FsError::NotAFile => ErrorKind::InvalidInput,
            FsError::NotADirectory => ErrorKind::InvalidInput,
            FsError::InvalidFd => ErrorKind::InvalidInput,
            FsError::InvalidPath => ErrorKind::InvalidInput,
            FsError::NotInitialized => ErrorKind::NotFound,
            FsError::InvalidSeek => ErrorKind::InvalidInput,
            FsError::InvalidManifest => ErrorKind::InvalidData,
            FsError::ReadOnly => ErrorKind::PermissionDenied,
            FsError::AlreadyExists => ErrorKind::AlreadyExists,
            FsError::NotEmpty => ErrorKind::InvalidInput,
            FsError::NoSpace => ErrorKind::OutOfMemory,
            FsError::TooManyOpenFiles => ErrorKind::OutOfMemory,
            FsError::NotSupported => ErrorKind::Unsupported,
            FsError::InvalidArgument => ErrorKind::InvalidInput,
            FsError::IoError => ErrorKind::Other,
            FsError::OutOfMemory => ErrorKind::OutOfMemory,
            FsError::FileLocked => ErrorKind::PermissionDenied,
            FsError::PlatformNotSupported => ErrorKind::Unsupported,
            FsError::PermissionDenied => ErrorKind::PermissionDenied,
        }
    }
}
