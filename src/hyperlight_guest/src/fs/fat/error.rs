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

//! Error types and mapping for FAT filesystem operations.

use core::fmt;

use crate::fs::error::FsError;

/// Error type for memory storage I/O operations.
///
/// This is a minimal error type for no_std environments that implements
/// the `fatfs::IoError` trait.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryIoError {
    /// Attempted to seek beyond the end of the memory region.
    OutOfBounds,
    /// Attempted to seek to a negative position.
    InvalidSeek,
    /// Unexpected end of file (read returned fewer bytes than expected).
    UnexpectedEof,
    /// Write returned zero bytes when more were expected.
    WriteZero,
}

impl fmt::Display for MemoryIoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryIoError::OutOfBounds => write!(f, "seek beyond end of memory region"),
            MemoryIoError::InvalidSeek => write!(f, "invalid seek position"),
            MemoryIoError::UnexpectedEof => write!(f, "unexpected end of file"),
            MemoryIoError::WriteZero => write!(f, "write returned zero bytes"),
        }
    }
}

impl fatfs::IoError for MemoryIoError {
    fn is_interrupted(&self) -> bool {
        // Memory operations don't get interrupted
        false
    }

    fn new_unexpected_eof_error() -> Self {
        MemoryIoError::UnexpectedEof
    }

    fn new_write_zero_error() -> Self {
        MemoryIoError::WriteZero
    }
}

/// Maps a fatfs error to an FsError.
///
/// # Spec Reference
///
/// Error mapping per spec §11.3.
pub fn map_fatfs_error<T>(err: fatfs::Error<T>) -> FsError {
    match err {
        fatfs::Error::Io(_) => FsError::IoError,
        fatfs::Error::UnexpectedEof => FsError::IoError,
        fatfs::Error::WriteZero => FsError::IoError,
        fatfs::Error::InvalidInput => FsError::InvalidPath,
        fatfs::Error::InvalidFileNameLength => FsError::InvalidPath,
        fatfs::Error::UnsupportedFileNameCharacter => FsError::InvalidPath,
        fatfs::Error::DirectoryIsNotEmpty => FsError::NotEmpty,
        fatfs::Error::NotFound => FsError::NotFound,
        fatfs::Error::AlreadyExists => FsError::AlreadyExists,
        fatfs::Error::CorruptedFileSystem => FsError::IoError,
        fatfs::Error::NotEnoughSpace => FsError::NoSpace,
        _ => FsError::IoError,
    }
}

#[cfg(test)]
mod tests {
    use alloc::format;

    use super::*;

    #[test]
    fn test_memory_io_error_display() {
        assert_eq!(
            format!("{}", MemoryIoError::OutOfBounds),
            "seek beyond end of memory region"
        );
        assert_eq!(
            format!("{}", MemoryIoError::InvalidSeek),
            "invalid seek position"
        );
        assert_eq!(
            format!("{}", MemoryIoError::UnexpectedEof),
            "unexpected end of file"
        );
        assert_eq!(
            format!("{}", MemoryIoError::WriteZero),
            "write returned zero bytes"
        );
    }

    #[test]
    fn test_memory_io_error_fatfs_trait() {
        use fatfs::IoError;

        assert!(!MemoryIoError::OutOfBounds.is_interrupted());
        assert_eq!(
            MemoryIoError::new_unexpected_eof_error(),
            MemoryIoError::UnexpectedEof
        );
        assert_eq!(
            MemoryIoError::new_write_zero_error(),
            MemoryIoError::WriteZero
        );
    }

    #[test]
    fn test_map_fatfs_error_not_found() {
        let err: fatfs::Error<MemoryIoError> = fatfs::Error::NotFound;
        assert_eq!(map_fatfs_error(err), FsError::NotFound);
    }

    #[test]
    fn test_map_fatfs_error_already_exists() {
        let err: fatfs::Error<MemoryIoError> = fatfs::Error::AlreadyExists;
        assert_eq!(map_fatfs_error(err), FsError::AlreadyExists);
    }

    #[test]
    fn test_map_fatfs_error_directory_not_empty() {
        let err: fatfs::Error<MemoryIoError> = fatfs::Error::DirectoryIsNotEmpty;
        assert_eq!(map_fatfs_error(err), FsError::NotEmpty);
    }

    #[test]
    fn test_map_fatfs_error_invalid_input() {
        let err: fatfs::Error<MemoryIoError> = fatfs::Error::InvalidInput;
        assert_eq!(map_fatfs_error(err), FsError::InvalidPath);
    }

    #[test]
    fn test_map_fatfs_error_invalid_filename_length() {
        let err: fatfs::Error<MemoryIoError> = fatfs::Error::InvalidFileNameLength;
        assert_eq!(map_fatfs_error(err), FsError::InvalidPath);
    }

    #[test]
    fn test_map_fatfs_error_unsupported_filename_char() {
        let err: fatfs::Error<MemoryIoError> = fatfs::Error::UnsupportedFileNameCharacter;
        assert_eq!(map_fatfs_error(err), FsError::InvalidPath);
    }

    #[test]
    fn test_map_fatfs_error_not_enough_space() {
        let err: fatfs::Error<MemoryIoError> = fatfs::Error::NotEnoughSpace;
        assert_eq!(map_fatfs_error(err), FsError::NoSpace);
    }

    #[test]
    fn test_map_fatfs_error_io_errors() {
        // All I/O related errors map to IoError
        let io_err: fatfs::Error<MemoryIoError> = fatfs::Error::Io(MemoryIoError::OutOfBounds);
        assert_eq!(map_fatfs_error(io_err), FsError::IoError);

        let eof_err: fatfs::Error<MemoryIoError> = fatfs::Error::UnexpectedEof;
        assert_eq!(map_fatfs_error(eof_err), FsError::IoError);

        let write_zero_err: fatfs::Error<MemoryIoError> = fatfs::Error::WriteZero;
        assert_eq!(map_fatfs_error(write_zero_err), FsError::IoError);

        let corrupt_err: fatfs::Error<MemoryIoError> = fatfs::Error::CorruptedFileSystem;
        assert_eq!(map_fatfs_error(corrupt_err), FsError::IoError);
    }
}
