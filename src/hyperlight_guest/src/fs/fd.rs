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

//! File descriptor table for HyperlightFS.
//!
//! Manages open files dynamically. Supports two types of files:
//! - **Read-only files**: Memory-mapped from the host manifest
//! - **FAT files**: Read-write files on FAT filesystem mounts
//!
//! This unified table allows the C API to use integer file descriptors for both types.
//!
//! ## dup/dup2 Support (POSIX Compliant)
//!
//! Both RO and FAT file entries use `Rc<RefCell<>>` for shared ownership. When `dup()` is called,
//! both file descriptors share the same underlying file state (position for RO, position+flags for FAT).
//! This matches POSIX semantics where duplicated fds share the file offset.

use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::{RefCell, UnsafeCell};

use super::error::FsError;
use super::fat::GuestFatFile;

/// O_APPEND flag value (matches Linux ABI: 0x0400).
/// Used to track append mode for POSIX-compliant write() behavior.
const O_APPEND: i32 = 0x0400;

/// First file descriptor available for allocation.
/// FDs 0-2 are reserved for stdin/stdout/stderr (POSIX compatibility).
const FIRST_AVAILABLE_FD: usize = 3;

/// Shared read-only file state for dup() support.
///
/// This is wrapped in `Rc<RefCell<>>` so multiple file descriptors
/// can share the same file position (POSIX dup semantics).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedRoFile {
    /// Current read position within the file.
    pub position: u64,
    /// Size of the file (cached from inode).
    pub size: u64,
    /// Guest address where file data is mapped.
    pub guest_address: u64,
}

/// An open read-only file entry (memory-mapped from manifest).
///
/// Uses `Rc<RefCell<>>` for shared ownership to support `dup()`/`dup2()`.
/// Multiple file descriptors can share the same file position,
/// matching POSIX semantics.
pub struct OpenFile {
    /// Shared state containing position.
    /// Multiple FdEntry::ReadOnly entries can share this via Rc::clone().
    inner: Rc<RefCell<SharedRoFile>>,
}

impl OpenFile {
    /// Create a new RO file entry.
    pub fn new(size: u64, guest_address: u64) -> Self {
        Self {
            inner: Rc::new(RefCell::new(SharedRoFile {
                position: 0,
                size,
                guest_address,
            })),
        }
    }

    /// Create a duplicate entry sharing the same underlying file.
    ///
    /// This is used by dup()/dup2() to create entries that share
    /// file position per POSIX semantics.
    pub fn dup(&self) -> Self {
        Self {
            inner: Rc::clone(&self.inner),
        }
    }

    /// Borrow the shared file state immutably.
    #[inline]
    pub fn borrow(&self) -> core::cell::Ref<'_, SharedRoFile> {
        self.inner.borrow()
    }

    /// Borrow the shared file state mutably.
    #[inline]
    pub fn borrow_mut(&self) -> core::cell::RefMut<'_, SharedRoFile> {
        self.inner.borrow_mut()
    }

    /// Get the current position.
    #[inline]
    pub fn position(&self) -> u64 {
        self.inner.borrow().position
    }

    /// Set the position.
    #[inline]
    pub fn set_position(&self, pos: u64) {
        self.inner.borrow_mut().position = pos;
    }

    /// Get the file size.
    #[inline]
    pub fn size(&self) -> u64 {
        self.inner.borrow().size
    }

    /// Get the guest address.
    #[inline]
    pub fn guest_address(&self) -> u64 {
        self.inner.borrow().guest_address
    }
}

impl core::fmt::Debug for OpenFile {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("OpenFile")
            .field("position", &self.position())
            .field("size", &self.size())
            .field("guest_address", &self.guest_address())
            .finish()
    }
}

/// A FAT file entry with its open flags.
///
/// Uses `Rc<RefCell<>>` for shared ownership to support `dup()`/`dup2()`.
/// Multiple file descriptors can share the same underlying file state,
/// matching POSIX semantics where duplicated fds share the file offset.
///
/// Tracks the original open() flags so fcntl F_GETFL can return accurate values
/// and write() can honor O_APPEND semantics.
pub struct FatFdEntry {
    /// Shared state containing the FAT file handle and flags.
    /// Multiple FdEntry::Fat entries can share this via Rc::clone().
    inner: Rc<RefCell<SharedFatFile>>,
}

/// Shared FAT file state for dup() support.
///
/// This is wrapped in `Rc<RefCell<>>` so multiple file descriptors
/// can share the same file position and flags (POSIX dup semantics).
pub struct SharedFatFile {
    /// The underlying FAT file handle.
    pub file: GuestFatFile<'static>,
    /// Original open() flags (O_RDONLY, O_WRONLY, O_RDWR, O_APPEND, etc.).
    /// Updated by fcntl F_SETFL for modifiable flags (O_APPEND).
    pub flags: i32,
    /// Mount path this file belongs to (e.g., "/data").
    /// Used to prevent unmounting while files are still open.
    pub mount_path: String,
}

impl FatFdEntry {
    /// Create a new FAT fd entry with the given file, flags, and mount path.
    pub fn new(file: GuestFatFile<'static>, flags: i32, mount_path: String) -> Self {
        Self {
            inner: Rc::new(RefCell::new(SharedFatFile {
                file,
                flags,
                mount_path,
            })),
        }
    }

    /// Create a duplicate entry sharing the same underlying file.
    ///
    /// This is used by dup()/dup2() to create entries that share
    /// file position and flags per POSIX semantics.
    pub fn dup(&self) -> Self {
        Self {
            inner: Rc::clone(&self.inner),
        }
    }

    /// Get the mount path this file belongs to.
    #[inline]
    pub fn mount_path(&self) -> String {
        self.inner.borrow().mount_path.clone()
    }

    /// Borrow the shared file state immutably.
    #[inline]
    pub fn borrow(&self) -> core::cell::Ref<'_, SharedFatFile> {
        self.inner.borrow()
    }

    /// Borrow the shared file state mutably.
    #[inline]
    pub fn borrow_mut(&self) -> core::cell::RefMut<'_, SharedFatFile> {
        self.inner.borrow_mut()
    }

    /// Returns true if O_APPEND flag is set.
    #[inline]
    pub fn is_append(&self) -> bool {
        (self.inner.borrow().flags & O_APPEND) != 0
    }

    /// Set or clear the O_APPEND flag.
    pub fn set_append(&mut self, append: bool) {
        let mut inner = self.inner.borrow_mut();
        if append {
            inner.flags |= O_APPEND;
        } else {
            inner.flags &= !O_APPEND;
        }
    }

    /// Get the current flags.
    #[inline]
    pub fn flags(&self) -> i32 {
        self.inner.borrow().flags
    }
}

impl core::fmt::Debug for FatFdEntry {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FatFdEntry")
            .field("mount_path", &self.mount_path())
            .field("flags", &self.flags())
            .finish_non_exhaustive()
    }
}

/// Entry in the unified file descriptor table.
///
/// Supports both read-only memory-mapped files and FAT filesystem files.
pub enum FdEntry {
    /// Read-only memory-mapped file from manifest.
    ReadOnly(OpenFile),
    /// FAT filesystem file with open flags.
    /// The `'static` lifetime is valid because FAT filesystems live for
    /// the guest's entire execution.
    Fat(FatFdEntry),
}

impl FdEntry {
    /// Returns true if this is a read-only file.
    pub fn is_readonly(&self) -> bool {
        matches!(self, FdEntry::ReadOnly(_))
    }

    /// Returns true if this is a FAT file.
    pub fn is_fat(&self) -> bool {
        matches!(self, FdEntry::Fat(_))
    }
}

/// File descriptor table.
struct FdTable {
    /// Slots for open files. None = slot is free, Some = open file.
    /// Freed slots are reused on subsequent allocations.
    slots: Vec<Option<FdEntry>>,
}

impl FdTable {
    fn new() -> Self {
        // Pre-allocate reserved slots (0=stdin, 1=stdout, 2=stderr)
        let mut slots = Vec::with_capacity(FIRST_AVAILABLE_FD + 8);
        for _ in 0..FIRST_AVAILABLE_FD {
            slots.push(None);
        }
        Self { slots }
    }
}

/// Global file descriptor table.
///
/// SAFETY: Guest code is single-threaded, so this is safe.
/// Initialized lazily on first access.
static FD_TABLE: FdTableCell = FdTableCell(UnsafeCell::new(None));

struct FdTableCell(UnsafeCell<Option<FdTable>>);

// SAFETY: Guest is single-threaded.
unsafe impl Sync for FdTableCell {}

impl FdTableCell {
    #[allow(clippy::mut_from_ref)] // SAFETY: Guest is single-threaded, interior mutability is intentional
    fn get(&self) -> &mut FdTable {
        // SAFETY: Guest is single-threaded.
        let inner = unsafe { &mut *self.0.get() };
        inner.get_or_insert_with(FdTable::new)
    }
}

/// Allocate a new file descriptor for a read-only file.
///
/// Returns the file descriptor number. Reuses freed slots when available,
/// otherwise appends to the table.
pub fn alloc_ro_fd(open_file: OpenFile) -> i32 {
    alloc_fd_entry(FdEntry::ReadOnly(open_file))
}

/// Allocate a new file descriptor for a FAT file.
///
/// # Arguments
/// * `fat_file` - The FAT file handle
/// * `flags` - Original open() flags (for fcntl F_GETFL and O_APPEND handling)
/// * `mount_path` - The mount path this file belongs to (for unmount tracking)
///
/// Returns the file descriptor number. Reuses freed slots when available,
/// otherwise appends to the table.
pub fn alloc_fat_fd(fat_file: GuestFatFile<'static>, flags: i32, mount_path: String) -> i32 {
    alloc_fd_entry(FdEntry::Fat(FatFdEntry::new(fat_file, flags, mount_path)))
}

/// Allocate a file descriptor for any entry type.
fn alloc_fd_entry(entry: FdEntry) -> i32 {
    let table = FD_TABLE.get();

    // Try to reuse a freed slot (skip reserved 0-2)
    for (idx, slot) in table.slots.iter_mut().enumerate().skip(FIRST_AVAILABLE_FD) {
        if slot.is_none() {
            *slot = Some(entry);
            return idx as i32;
        }
    }

    // No free slot, append new entry
    let fd = table.slots.len() as i32;
    table.slots.push(Some(entry));
    fd
}

/// Get an open file entry by file descriptor.
///
/// Returns a mutable reference to the FdEntry.
pub fn get_fd_entry(fd: i32) -> Result<&'static mut FdEntry, FsError> {
    if (fd as usize) < FIRST_AVAILABLE_FD {
        return Err(FsError::InvalidFd);
    }

    let table = FD_TABLE.get();
    table
        .slots
        .get_mut(fd as usize)
        .and_then(|slot| slot.as_mut())
        .ok_or(FsError::InvalidFd)
}

/// Get a read-only file by file descriptor.
///
/// Returns a mutable reference to the OpenFile entry.
/// Returns `InvalidFd` if fd is invalid or refers to a FAT file.
pub fn get_ro_fd(fd: i32) -> Result<&'static mut OpenFile, FsError> {
    match get_fd_entry(fd)? {
        FdEntry::ReadOnly(file) => Ok(file),
        FdEntry::Fat(_) => Err(FsError::InvalidFd),
    }
}

/// Get a FAT file entry by file descriptor.
///
/// Returns a mutable reference to the FatFdEntry (file + flags).
/// Returns `InvalidFd` if fd is invalid or refers to a read-only file.
pub fn get_fat_fd(fd: i32) -> Result<&'static mut FatFdEntry, FsError> {
    match get_fd_entry(fd)? {
        FdEntry::Fat(entry) => Ok(entry),
        FdEntry::ReadOnly(_) => Err(FsError::InvalidFd),
    }
}

/// Close a file descriptor, freeing the slot for reuse.
///
/// For FAT files, the underlying file is only closed when the last
/// reference is dropped (Rc reference count reaches zero).
pub fn free_fd(fd: i32) -> Result<(), FsError> {
    if (fd as usize) < FIRST_AVAILABLE_FD {
        return Err(FsError::InvalidFd);
    }

    let table = FD_TABLE.get();
    match table.slots.get_mut(fd as usize) {
        Some(slot @ Some(_)) => {
            // Drop the entry (FAT files closed when last Rc dropped)
            *slot = None;
            Ok(())
        }
        _ => Err(FsError::InvalidFd),
    }
}

/// Duplicate a file descriptor.
///
/// Creates a new fd that shares the same underlying file state.
/// Both RO and FAT files share position per POSIX semantics.
///
/// Returns the new file descriptor number (lowest available >= 3).
pub fn dup_fd(oldfd: i32) -> Result<i32, FsError> {
    dup_fd_to(oldfd, None, None)
}

/// Duplicate a file descriptor to a specific fd number or lowest >= min_fd.
///
/// # Arguments
/// * `oldfd` - Source file descriptor to duplicate
/// * `newfd` - If Some, use this exact fd (dup2 semantics, closes if open)
/// * `min_fd` - If newfd is None, find lowest available fd >= min_fd (F_DUPFD semantics)
///
/// Both RO and FAT files share position per POSIX semantics.
///
/// Returns the new file descriptor number.
pub fn dup_fd_to(oldfd: i32, newfd: Option<i32>, min_fd: Option<i32>) -> Result<i32, FsError> {
    if (oldfd as usize) < FIRST_AVAILABLE_FD {
        return Err(FsError::InvalidFd);
    }

    let table = FD_TABLE.get();

    // Get the old entry
    let old_entry = table
        .slots
        .get(oldfd as usize)
        .and_then(|slot| slot.as_ref())
        .ok_or(FsError::InvalidFd)?;

    // Create the new entry - both types use Rc::clone for POSIX shared position
    let new_entry = match old_entry {
        FdEntry::ReadOnly(ro_file) => {
            // Read-only files: share position via Rc::clone (POSIX compliant)
            FdEntry::ReadOnly(ro_file.dup())
        }
        FdEntry::Fat(fat_entry) => {
            // FAT files: share the underlying file via Rc::clone
            FdEntry::Fat(fat_entry.dup())
        }
    };

    match newfd {
        Some(target_fd) => {
            // dup2 semantics: use specific fd
            if (target_fd as usize) < FIRST_AVAILABLE_FD {
                return Err(FsError::InvalidFd);
            }

            // Ensure table is big enough
            while table.slots.len() <= target_fd as usize {
                table.slots.push(None);
            }

            // Close existing fd at target (if any) - ignore errors
            table.slots[target_fd as usize] = None;

            // Assign new entry
            table.slots[target_fd as usize] = Some(new_entry);
            Ok(target_fd)
        }
        None => {
            // dup/F_DUPFD semantics: find lowest available fd >= min_fd
            let start_fd = min_fd
                .map(|m| m.max(FIRST_AVAILABLE_FD as i32) as usize)
                .unwrap_or(FIRST_AVAILABLE_FD);

            // Ensure table is big enough for min_fd
            while table.slots.len() <= start_fd {
                table.slots.push(None);
            }

            for (idx, slot) in table.slots.iter_mut().enumerate().skip(start_fd) {
                if slot.is_none() {
                    *slot = Some(new_entry);
                    return Ok(idx as i32);
                }
            }

            // No free slot, append
            let fd = table.slots.len() as i32;
            table.slots.push(Some(new_entry));
            Ok(fd)
        }
    }
}

/// Check if any FAT files are currently open on the given mount path.
///
/// This is used to prevent unmounting while files are still open,
/// which would cause use-after-free when the underlying FAT memory is freed.
///
/// # Arguments
/// * `mount_path` - The mount path to check (e.g., "/scratch")
///
/// # Returns
/// `true` if any open FAT files belong to this mount, `false` otherwise.
pub fn has_open_files_on_mount(mount_path: &str) -> bool {
    let table = FD_TABLE.get();

    for slot in &table.slots {
        if let Some(FdEntry::Fat(fat_entry)) = slot {
            if fat_entry.mount_path() == mount_path {
                return true;
            }
        }
    }

    false
}

/// Check if a file descriptor is valid (open).
#[cfg(test)]
pub fn is_valid_fd(fd: i32) -> bool {
    if (fd as usize) < FIRST_AVAILABLE_FD {
        return false;
    }

    let table = FD_TABLE.get();
    table
        .slots
        .get(fd as usize)
        .is_some_and(|slot| slot.is_some())
}

/// Check if a file descriptor refers to a read-only file.
#[cfg(test)]
pub fn is_readonly_fd(fd: i32) -> bool {
    get_fd_entry(fd).is_ok_and(|e| e.is_readonly())
}

/// Check if a file descriptor refers to a FAT file.
#[cfg(test)]
pub fn is_fat_fd(fd: i32) -> bool {
    get_fd_entry(fd).is_ok_and(|e| e.is_fat())
}

/// Get the number of currently open files.
#[cfg(test)]
pub fn open_count() -> usize {
    let table = FD_TABLE.get();
    table.slots.iter().filter(|s| s.is_some()).count()
}

/// Reset the file descriptor table (for testing).
#[cfg(test)]
pub fn reset() {
    let table = FD_TABLE.get();
    table.slots.clear();
    // Fill reserved slots (0-2) with None
    for _ in 0..FIRST_AVAILABLE_FD {
        table.slots.push(None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alloc_and_free_readonly() {
        reset();

        let file = OpenFile::new(100, 0x1000);

        // Allocate (first available is fd 3, since 0-2 are reserved)
        let fd = alloc_ro_fd(file);
        assert_eq!(fd, 3);
        assert!(is_valid_fd(fd));
        assert!(is_readonly_fd(fd));
        assert!(!is_fat_fd(fd));
        assert_eq!(open_count(), 1);

        // Get and verify
        let entry = get_ro_fd(fd).unwrap();
        assert_eq!(entry.size(), 100);

        // Free
        free_fd(fd).unwrap();
        assert!(!is_valid_fd(fd));
        assert_eq!(open_count(), 0);

        // Double free should fail
        assert_eq!(free_fd(fd), Err(FsError::InvalidFd));
    }

    #[test]
    fn test_invalid_fd() {
        reset();

        assert!(get_ro_fd(-1).is_err());
        assert!(get_ro_fd(1000).is_err()); // Out of bounds
        assert!(get_ro_fd(0).is_err()); // Reserved (stdin)
        assert!(get_ro_fd(1).is_err()); // Reserved (stdout)
        assert!(get_ro_fd(2).is_err()); // Reserved (stderr)
        assert!(get_ro_fd(3).is_err()); // Not open yet
    }

    #[test]
    fn test_slot_reuse() {
        reset();

        let file = OpenFile::new(0, 0);

        // Allocate three fds (starting from 3 since 0-2 are reserved)
        let fd0 = alloc_ro_fd(file.dup());
        let fd1 = alloc_ro_fd(file.dup());
        let fd2 = alloc_ro_fd(file.dup());
        assert_eq!(fd0, 3);
        assert_eq!(fd1, 4);
        assert_eq!(fd2, 5);
        assert_eq!(open_count(), 3);

        // Free the middle one
        free_fd(fd1).unwrap();
        assert_eq!(open_count(), 2);

        // Next alloc should reuse slot 4
        let fd3 = alloc_ro_fd(file.dup());
        assert_eq!(fd3, 4);
        assert_eq!(open_count(), 3);

        reset();
    }

    #[test]
    fn test_position_update() {
        reset();

        let file = OpenFile::new(100, 0x1000);

        let fd = alloc_ro_fd(file);
        assert_eq!(fd, 3); // First available after reserved

        // Update position
        {
            let entry = get_ro_fd(fd).unwrap();
            entry.set_position(50);
        }

        // Verify update persisted
        {
            let entry = get_ro_fd(fd).unwrap();
            assert_eq!(entry.position(), 50);
        }

        reset();
    }

    #[test]
    fn test_mixed_readonly_and_fat_fds() {
        reset();

        let ro_file = OpenFile::new(100, 0x1000);

        // Allocate RO fd
        let ro_fd = alloc_ro_fd(ro_file);
        assert_eq!(ro_fd, 3);
        assert!(is_readonly_fd(ro_fd));
        assert!(!is_fat_fd(ro_fd));

        // get_ro_fd works for RO
        assert!(get_ro_fd(ro_fd).is_ok());
        // get_fat_fd fails for RO
        assert!(get_fat_fd(ro_fd).is_err());

        // Note: We can't test alloc_fat_fd without a real GuestFatFile,
        // which requires an initialized FAT filesystem. That's tested
        // in the integration tests.

        reset();
    }
}
