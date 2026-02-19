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

//! Raw memory storage backend for FAT filesystem.

use core::fmt;

use fatfs::{IoBase, Read, Seek, SeekFrom, Write};

use super::error::MemoryIoError;

/// A storage backend backed by a raw memory region.
///
/// Wraps a `(*mut u8, usize)` pair and implements the fatfs I/O traits,
/// allowing `fatfs::FileSystem` to read/write a FAT image in memory.
///
/// # Note
///
/// Most code should use [`super::GuestFat`] instead, which wraps this with a
/// cleaner API. This type is exposed for advanced use cases.
///
/// # Thread Safety
///
/// The guest is single-threaded, so no Send/Sync impls are needed.
pub struct RawMemoryStorage {
    /// Pointer to start of the memory region.
    base: *mut u8,
    /// Size of the memory region in bytes.
    size: usize,
    /// Current read/write position.
    position: usize,
}

impl RawMemoryStorage {
    /// Create a new storage over a memory region.
    ///
    /// # Arguments
    ///
    /// * `base` - Pointer to the start of the FAT image in memory
    /// * `size` - Size of the memory region in bytes
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `base` points to a valid, readable, and writable memory region
    /// - The memory region is at least `size` bytes
    /// - The memory remains valid for the lifetime of this `RawMemoryStorage`
    /// - No other code accesses this memory region concurrently without
    ///   proper synchronization
    ///
    /// # Panics
    ///
    /// Panics if `base` is null or `size` is zero.
    #[inline]
    pub unsafe fn new(base: *mut u8, size: usize) -> Self {
        assert!(
            !base.is_null(),
            "RawMemoryStorage base pointer cannot be null"
        );
        assert!(size > 0, "RawMemoryStorage size must be greater than 0");
        Self {
            base,
            size,
            position: 0,
        }
    }

    /// Returns the size of the memory region in bytes.
    #[inline]
    #[allow(dead_code)] // Will be used in Phase 3+ for diagnostics
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns the current position within the memory region.
    #[inline]
    #[allow(dead_code)] // Will be used in Phase 3+ for diagnostics
    pub fn position(&self) -> usize {
        self.position
    }

    /// Returns the number of bytes remaining from current position to end.
    #[inline]
    fn remaining(&self) -> usize {
        self.size.saturating_sub(self.position)
    }
}

impl fmt::Debug for RawMemoryStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RawMemoryStorage")
            .field("base", &self.base)
            .field("size", &self.size)
            .field("position", &self.position)
            .finish()
    }
}

impl IoBase for RawMemoryStorage {
    type Error = MemoryIoError;
}

impl Read for RawMemoryStorage {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, Self::Error> {
        if buf.is_empty() {
            return Ok(0);
        }

        let to_read = buf.len().min(self.remaining());
        if to_read == 0 {
            // EOF: at or past end of region
            return Ok(0);
        }

        // SAFETY: We've verified position + to_read <= size, and the caller
        // guaranteed the memory region is valid via the unsafe constructor.
        unsafe {
            let src = self.base.add(self.position);
            core::ptr::copy_nonoverlapping(src, buf.as_mut_ptr(), to_read);
        }

        self.position += to_read;
        Ok(to_read)
    }
}

impl Write for RawMemoryStorage {
    fn write(&mut self, buf: &[u8]) -> Result<usize, Self::Error> {
        if buf.is_empty() {
            return Ok(0);
        }

        let to_write = buf.len().min(self.remaining());
        if to_write == 0 {
            // No space left: at or past end of region
            // Return 0 to indicate we couldn't write anything
            return Ok(0);
        }

        // SAFETY: We've verified position + to_write <= size, and the caller
        // guaranteed the memory region is valid via the unsafe constructor.
        unsafe {
            let dst = self.base.add(self.position);
            core::ptr::copy_nonoverlapping(buf.as_ptr(), dst, to_write);
        }

        self.position += to_write;
        Ok(to_write)
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        // Memory writes are immediately visible; nothing to flush.
        // The host is responsible for ensuring durability via msync.
        Ok(())
    }
}

impl Seek for RawMemoryStorage {
    fn seek(&mut self, pos: SeekFrom) -> Result<u64, Self::Error> {
        // Calculate new position. We use i64 arithmetic to handle negative offsets.
        // Note: For extremely large offsets (> i64::MAX), the `as i64` cast will
        // produce a negative value, which is caught by the `new_pos < 0` check below.
        // This is intentional - such positions would be invalid anyway.
        let new_pos = match pos {
            SeekFrom::Start(offset) => {
                // Guard against u64 values that would overflow i64
                if offset > i64::MAX as u64 {
                    return Err(MemoryIoError::OutOfBounds);
                }
                offset as i64
            }
            SeekFrom::End(offset) => self.size as i64 + offset,
            SeekFrom::Current(offset) => self.position as i64 + offset,
        };

        if new_pos < 0 {
            return Err(MemoryIoError::InvalidSeek);
        }

        let new_pos = new_pos as usize;

        // Allow seeking to end (for size queries), but not beyond
        if new_pos > self.size {
            return Err(MemoryIoError::OutOfBounds);
        }

        self.position = new_pos;
        Ok(new_pos as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raw_memory_storage_read() {
        let mut data = [0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut storage = unsafe { RawMemoryStorage::new(data.as_mut_ptr(), data.len()) };

        let mut buf = [0u8; 4];
        assert_eq!(storage.read(&mut buf).unwrap(), 4);
        assert_eq!(buf, [0, 1, 2, 3]);
        assert_eq!(storage.position(), 4);

        assert_eq!(storage.read(&mut buf).unwrap(), 4);
        assert_eq!(buf, [4, 5, 6, 7]);
        assert_eq!(storage.position(), 8);

        // Read past end
        assert_eq!(storage.read(&mut buf).unwrap(), 2);
        assert_eq!(buf[..2], [8, 9]);
        assert_eq!(storage.position(), 10);

        // EOF
        assert_eq!(storage.read(&mut buf).unwrap(), 0);
    }

    #[test]
    fn test_raw_memory_storage_write() {
        let mut data = [0u8; 10];
        let mut storage = unsafe { RawMemoryStorage::new(data.as_mut_ptr(), data.len()) };

        let buf = [1u8, 2, 3, 4];
        assert_eq!(storage.write(&buf).unwrap(), 4);
        assert_eq!(storage.position(), 4);
        assert_eq!(&data[..4], [1, 2, 3, 4]);

        assert_eq!(storage.write(&buf).unwrap(), 4);
        assert_eq!(&data[4..8], [1, 2, 3, 4]);

        // Write past end - partial write
        assert_eq!(storage.write(&buf).unwrap(), 2);
        assert_eq!(&data[8..], [1, 2]);

        // No space left
        assert_eq!(storage.write(&buf).unwrap(), 0);
    }

    #[test]
    fn test_raw_memory_storage_seek() {
        let mut data = [0u8; 100];
        let mut storage = unsafe { RawMemoryStorage::new(data.as_mut_ptr(), data.len()) };

        // Seek from start
        assert_eq!(storage.seek(SeekFrom::Start(50)).unwrap(), 50);
        assert_eq!(storage.position(), 50);

        // Seek from current (forward)
        assert_eq!(storage.seek(SeekFrom::Current(10)).unwrap(), 60);
        assert_eq!(storage.position(), 60);

        // Seek from current (backward)
        assert_eq!(storage.seek(SeekFrom::Current(-20)).unwrap(), 40);
        assert_eq!(storage.position(), 40);

        // Seek from end
        assert_eq!(storage.seek(SeekFrom::End(-10)).unwrap(), 90);
        assert_eq!(storage.position(), 90);

        // Seek to end (valid)
        assert_eq!(storage.seek(SeekFrom::End(0)).unwrap(), 100);
        assert_eq!(storage.position(), 100);

        // Seek past end (invalid)
        assert_eq!(
            storage.seek(SeekFrom::End(1)),
            Err(MemoryIoError::OutOfBounds)
        );

        // Seek to negative (invalid)
        assert_eq!(
            storage
                .seek(SeekFrom::Start(0))
                .and_then(|_| storage.seek(SeekFrom::Current(-1))),
            Err(MemoryIoError::InvalidSeek)
        );

        // Seek with offset > i64::MAX (overflow guard)
        assert_eq!(
            storage.seek(SeekFrom::Start(u64::MAX)),
            Err(MemoryIoError::OutOfBounds)
        );
    }

    #[test]
    fn test_raw_memory_storage_empty_operations() {
        let mut data = [0u8; 10];
        let mut storage = unsafe { RawMemoryStorage::new(data.as_mut_ptr(), data.len()) };

        // Empty read
        let mut buf = [];
        assert_eq!(storage.read(&mut buf).unwrap(), 0);

        // Empty write
        assert_eq!(storage.write(&[]).unwrap(), 0);

        // Flush (no-op)
        assert!(storage.flush().is_ok());
    }

    #[test]
    #[should_panic(expected = "null")]
    fn test_raw_memory_storage_null_pointer() {
        unsafe { RawMemoryStorage::new(core::ptr::null_mut(), 100) };
    }

    #[test]
    #[should_panic(expected = "size must be greater than 0")]
    fn test_raw_memory_storage_zero_size() {
        let mut data = [0u8; 10];
        unsafe { RawMemoryStorage::new(data.as_mut_ptr(), 0) };
    }
}
