# HyperlightFS FAT Integration - Implementation Plan

## Document Status

| Field | Value |
|-------|-------|
| Status | **Complete** |
| Created | 2026-01-19 |
| Last Updated | 2026-01-22 |
| Specification | [hyperlight-fs-fat-spec.md](./hyperlight-fs-fat-spec.md) |

## Progress Summary

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 1: Host-Side Foundation | ✅ Complete | 6/6 |
| Phase 2: Guest-Side Foundation | ✅ Complete | 5/5 |
| Phase 3: Host-Guest Integration | ✅ Complete | 4/4 |
| Phase 4: C API Implementation | ✅ Complete | 4/4 |
| Phase 5: Host Extraction APIs | ✅ Complete | 3/3 |
| Phase 6: Testing & Documentation | ✅ Complete | 4/4 |

**Overall: 26/26 steps complete** 🎉

---

## Design Reference

All architecture, design decisions, and constraints are documented in the **[HyperlightFS Specification](./hyperlight-fs-fat-spec.md)**. Key sections:

- **§2**: Architecture and MAP_SHARED flow (including page fault handling)
- **§3**: File system types (RO vs FAT), comparison, data persistence guarantees
- **§12**: Security considerations and file locking strategy
- **§13**: Limitations, platform support, resource limits

This plan focuses on **implementation tasks only**.

---

## Platform Constraints

**Windows (WHP)**: Not supported - see Spec §13.1. All HyperlightFS methods return `FsError::PlatformNotSupported` on Windows.

---

## Phase 1: Host-Side Foundation

Build the host-side infrastructure for FAT filesystem support.

### Step 1.1: Add fatfs Dependencies

**Status:** ✅ Complete

**Goal:** Add fatfs, fscommon, and file locking crates to hyperlight-host.

**Files to modify:**
- `src/hyperlight_host/Cargo.toml`

**Changes:**
```toml
[dependencies]
fatfs = "0.3"
fscommon = "0.1"
fs2 = "0.4"  # For cross-platform file locking
```

**Acceptance criteria:**
- [ ] Dependencies added to Cargo.toml
- [ ] `cargo check` passes
- [ ] No version conflicts with existing dependencies

**Notes:**
- fscommon provides `BufStream` for buffered I/O over FAT images
- fatfs 0.3.6 is latest stable
- fs2 provides `flock()` equivalent via `FileExt::lock_exclusive()`

---

### Step 1.2: Create FatImage Wrapper Type with Exclusive Locking

**Status:** ✅ Complete

**Goal:** Create a type to manage FAT filesystem images on the host with exclusive file locking.

**Files to create:**
- `src/hyperlight_host/src/hyperlight_fs/fat_image.rs`

**Files to modify:**
- `src/hyperlight_host/src/hyperlight_fs/mod.rs` (add module)

**Types to implement:**
```rust
use std::fs::File;
use fs2::FileExt;

/// A FAT filesystem image backed by an mmap'd host file.
/// 
/// The backing file is exclusively locked (flock) to prevent
/// multiple sandboxes from mapping the same file concurrently.
pub struct FatImage {
    /// The backing file (holds the exclusive lock)
    file: File,
    /// Memory-mapped region (MAP_SHARED for write persistence)
    mmap: memmap2::MmapMut,
    /// Mount point in guest namespace
    mount_point: String,
    /// Whether this is a temp file (delete on drop)
    is_temp: bool,
}

impl FatImage {
    /// Open an existing FAT image from a file with exclusive lock.
    /// 
    /// Returns `FsError::FileLocked` if file is already locked by another process.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self>;
    
    /// Create a new empty FAT image with specified fixed size.
    /// Creates a backing temp file on the host (potentially sparse).
    /// Size cannot be changed after creation - ENOSPC when full.
    pub fn create_temp(size_bytes: usize) -> Result<Self>;
    
    /// Create a new empty FAT image at specified path with exclusive lock.
    pub fn create_at<P: AsRef<Path>>(path: P, size_bytes: usize) -> Result<Self>;
    
    /// Get the mmap'd region for guest memory mapping.
    pub fn as_mut_ptr(&mut self) -> *mut u8;
    
    /// Get size of image.
    pub fn size(&self) -> usize;
}

impl Drop for FatImage {
    fn drop(&mut self) {
        // Lock is automatically released when file is dropped
        // If is_temp, delete the backing file
    }
}
```

**Acceptance criteria:**
- [ ] FatImage struct implemented with mmap + exclusive lock
- [ ] open() acquires lock, returns FileLocked if already held
- [ ] create_temp() creates temp file, formats FAT32, acquires lock
- [ ] create_at() creates file at path, formats FAT32, acquires lock
- [ ] Lock released on drop
- [ ] Temp files deleted on drop
- [ ] Unit tests pass

**Tests to add:**
- `test_fat_image_create_temp` - create temp image, verify formatted
- `test_fat_image_create_at` - create at path, verify persists after drop
- `test_fat_image_open` - open existing image
- `test_fat_image_exclusive_lock` - second open fails with FileLocked
- `test_fat_image_lock_released_on_drop` - drop first, second can open

---

### Step 1.3: Host FAT File Operations

**Status:** ✅ Complete

**Goal:** Add methods to read/write files within a FAT image.

**Files to modify:**
- `src/hyperlight_host/src/hyperlight_fs/fat_image.rs`

**Methods to implement:**
```rust
impl FatImage {
    /// List directory contents.
    pub fn read_dir(&self, path: &str) -> Result<Vec<FatDirEntry>>;
    
    /// Read a file's contents.
    pub fn read_file(&self, path: &str) -> Result<Vec<u8>>;
    
    /// Write data to a file (creates or overwrites).
    pub fn write_file(&mut self, path: &str, data: &[u8]) -> Result<()>;
    
    /// Create a directory.
    pub fn create_dir(&mut self, path: &str) -> Result<()>;
    
    /// Delete a file.
    pub fn delete_file(&mut self, path: &str) -> Result<()>;
    
    /// Delete an empty directory.
    pub fn delete_dir(&mut self, path: &str) -> Result<()>;
    
    /// Get file/directory metadata.
    pub fn stat(&self, path: &str) -> Result<FatStat>;
    
    /// Check if path exists.
    pub fn exists(&self, path: &str) -> bool;
}

pub struct FatDirEntry {
    pub name: String,
    pub is_dir: bool,
    pub size: u64,
}

pub struct FatStat {
    pub size: u64,
    pub is_dir: bool,
    pub created: Option<DateTime<Utc>>,
    pub modified: Option<DateTime<Utc>>,
    pub accessed: Option<DateTime<Utc>>,
}
```

**Acceptance criteria:**
- [ ] All methods implemented
- [ ] Proper error handling for not-found, permission errors
- [ ] Path validation (no `..`, must be absolute)
- [ ] Unit tests pass

**Tests to add:**
- `test_fat_write_read_file`
- `test_fat_create_delete_dir`
- `test_fat_list_dir`
- `test_fat_nested_dirs`
- `test_fat_overwrite_file`

---

### Step 1.4: Extend HyperlightFSBuilder for FAT Mounts

**Status:** ✅ Complete

**Goal:** Add methods to mount FAT images in the builder.

**Files to modify:**
- `src/hyperlight_host/src/hyperlight_fs/builder.rs`

**Methods to add:**
```rust
impl HyperlightFSBuilder {
    /// Mount a FAT image from a host file.
    pub fn add_fat_image<P: AsRef<Path>>(
        self,
        host_path: P,
        mount_point: &str
    ) -> Result<Self>;
    
    /// Create an empty FAT filesystem at a mount point.
    /// 
    /// Creates a backing host temp file of the specified size (sparse)
    /// and formats it as FAT32. Size is fixed - ENOSPC when full.
    /// Temp file is deleted when HyperlightFSImage is dropped.
    pub fn add_empty_fat_mount(
        self,
        mount_point: &str,
        size_bytes: usize
    ) -> Result<Self>;
    
    /// Create an empty FAT filesystem backed by a specified host file.
    /// 
    /// Like add_empty_fat_mount(), but the file persists after drop.
    /// Useful for debugging, inspection, or reuse across runs.
    pub fn add_empty_fat_mount_at<P: AsRef<Path>>(
        self,
        host_path: P,
        mount_point: &str,
        size_bytes: usize
    ) -> Result<Self>;
}
```

**Internal changes:**
- Add `fat_mounts: Vec<FatMountEntry>` to builder state
- Add conflict detection between:
  - FAT mounts and RO files
  - FAT mounts and other FAT mounts
  - Root mount exclusivity check
- Validate mount points (absolute, no `..`, no null bytes)

**Acceptance criteria:**
- [x] add_fat_image() loads and validates FAT image
- [x] add_empty_fat_mount() creates formatted FAT
- [x] add_empty_fat_mount_at() creates FAT at specified path
- [x] Conflict detection works for all cases
- [x] Root mount (`/`) prevents other additions
- [x] Unit tests pass

**Tests to add:**
- [x] `test_builder_add_fat_image`
- [x] `test_builder_add_empty_fat`
- [x] `test_builder_add_empty_fat_at` - verify file persists after drop
- [x] `test_builder_fat_conflict_with_file`
- [x] `test_builder_fat_conflict_with_fat`
- [x] `test_builder_root_fat_exclusive`

**Additional tests added:**
- [x] `test_file_after_fat_mount_conflict` - RO file under existing mount
- [x] `test_file_after_root_fat_mount` - RO file with root mount
- [x] `test_dir_after_fat_mount_conflict` - directory under mount
- [x] `test_dir_after_root_fat_mount` - directory with root mount
- [x] `test_nested_fat_mounts_rejected` - /data then /data/nested
- [x] `test_nested_fat_mounts_reverse_order` - /data/nested then /data
- [x] `test_builder_multiple_fat_mounts` - non-conflicting mounts
- [x] `test_builder_fat_and_ro_files_coexist` - valid coexistence
- [x] `test_builder_invalid_mount_point` - path validation
- [x] `test_builder_root_fat_blocks_files` - files then root mount
- [x] `test_builder_fat_conflict_nested_mounts` - nested mount detection

**Deferred:** FsLimits and shared locks for RO files (see Appendix D)

---

### Step 1.5: TOML Config Support for FAT Mounts

**Status:** ✅ Complete

**Goal:** Extend the TOML configuration parser to support FAT mount declarations.

**Files to modify:**
- `src/hyperlight_host/src/hyperlight_fs/config.rs`

**Config format:**
```toml
# Mount an existing FAT image file
[[fat_image]]
host_path = "/host/path/to/data.fat"
mount_point = "/data"

# Create an empty FAT mount (temp file, deleted on drop)
[[fat_mount]]
mount_point = "/scratch"
size = "10MB"

# Create an empty FAT mount at a specific host path (persists after drop)
[[fat_mount]]
host_path = "/host/path/to/persistent.fat"
mount_point = "/logs"
size = "50MB"
```

**Types to add:**
```rust
/// Configuration for mounting an existing FAT image.
#[derive(Debug, Clone, Deserialize)]
pub struct FatImageConfig {
    /// Path to the FAT image file on the host
    pub host_path: String,
    /// Mount point in the guest filesystem
    pub mount_point: String,
}

/// Configuration for creating an empty FAT mount.
#[derive(Debug, Clone, Deserialize)]
pub struct FatMountConfig {
    /// Optional path for persistent FAT image (temp if omitted)
    pub host_path: Option<String>,
    /// Mount point in the guest filesystem
    pub mount_point: String,
    /// Size of the FAT image (e.g., "10MB", "1GB", or bytes as integer)
    pub size: SizeValue,
}

/// Size value that can be parsed from string ("10MB") or integer (bytes).
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum SizeValue {
    Bytes(usize),
    Human(String),
}
```

**Changes to HyperlightFsConfig:**
```rust
pub struct HyperlightFsConfig {
    #[serde(default)]
    pub file: Vec<FileMapping>,
    #[serde(default)]
    pub directory: Vec<DirectoryMapping>,
    #[serde(default)]
    pub fat_image: Vec<FatImageConfig>,    // NEW
    #[serde(default)]
    pub fat_mount: Vec<FatMountConfig>,    // NEW
}
```

**Changes to from_config():**
- Process `fat_image` entries via `add_fat_image()`
- Process `fat_mount` entries via `add_empty_fat_mount()` or `add_empty_fat_mount_at()`
- Parse human-readable sizes ("10MB", "1GB", etc.)

**Acceptance criteria:**
- [x] `FatImageConfig` struct implemented
- [x] `FatMountConfig` struct implemented  
- [x] `SizeValue` parses both bytes and human-readable sizes
- [x] `from_config()` processes FAT entries
- [x] Conflict detection works via builder
- [x] Unit tests pass

**Tests added:**
- [x] `test_from_config_fat_image`
- [x] `test_from_config_fat_mount_temp`
- [x] `test_from_config_fat_mount_persistent`
- [x] `test_from_config_fat_size_bytes`
- [x] `test_from_config_fat_size_human`
- [x] `test_from_config_fat_conflict_with_file`
- [x] `test_from_config_mixed_ro_and_fat`
- [x] `test_from_config_fat_invalid_size`
- [x] `test_from_config_multiple_fat_mounts`

**Implementation notes:**
- Renamed `host` field to `host_path` in all config structs for clarity
- Added support for size suffixes: B, KB, KiB, MB, MiB, GB, GiB
- Size parsing supports decimal values (e.g., "1.5GB")

---

### Step 1.6: Update FlatBuffer Schema

**Status:** ✅ Complete

**Goal:** Extend the manifest schema to support FAT mount metadata.

**Files to modify:**
- `src/schema/hyperlight_fs.fbs`
- `src/hyperlight_common/src/flatbuffer_wrappers/hyperlight_fs.rs`

**Schema changes:**
```flatbuffers
enum InodeType : ubyte {
    File = 0,
    Directory = 1,
    FatMount = 2,    // NEW
}

table Inode {
    inode_type: InodeType;
    path: string (required);
    parent: uint32;
    guest_address: uint64;
    size: uint64;     // CHANGED: was int32 (~2GB limit), now uint64 (no practical limit)
    mount_id: uint32;  // NEW: identifies mount for extraction
}

table HyperlightFS {
    version: uint16;  // Bump to 2
    inodes: [Inode] (required);
}
```

**Wrapper changes:**
- Add `InodeType::FatMount`
- Add `mount_id` field to `InodeData`
- Change `size` field from `i32` to `u64`
- Add `InodeData::fat_mount()` constructor
- Update `HYPERLIGHT_FS_VERSION` to 2
- Add backward compatibility for v1 manifests
- Remove 2GB file size limit check in builder

**Acceptance criteria:**
- [x] Schema updated
- [x] `just gen-all-fbs-rust-code` regenerates bindings
- [x] Wrapper types updated
- [x] Roundtrip serialization works
- [x] v1 manifests still parseable (via default mount_id=0)

**Tests added:**
- [x] `test_fat_mount_roundtrip` - FAT mount serialization/deserialization
- [x] `test_inode_type_conversion` - updated for FatMount

---

## Phase 2: Guest-Side Foundation

Build the guest-side VFS infrastructure.

### Step 2.1: Add fatfs to Guest

**Status:** ✅ Complete

**Goal:** Add fatfs crate to hyperlight-guest with no_std support.

**Files to modify:**
- `src/hyperlight_guest/Cargo.toml`

**Changes:**
```toml
[dependencies]
fatfs = { git = "https://github.com/rafalh/rust-fatfs", rev = "4eccb50d011146fbed20e133d33b22f3c27292e7", default-features = false, features = ["alloc"] }
```

**Note:** We use the git version (0.4.0 unreleased) because:
- v0.3.6 requires `core_io` crate which only works on nightly Rust
- v0.4.0 on master replaced `core_io` with custom I/O traits that work on stable
- Both host and guest use the same commit hash for consistent FAT formatting

**Acceptance criteria:**
- [x] Dependency added
- [x] `cargo build --target x86_64-unknown-none` succeeds  
- [x] No_std compatibility verified
- [x] Host updated to same fatfs version for consistency

**Implementation notes:**
- Used git dependency pinned to commit `4eccb50d011146fbed20e133d33b22f3c27292e7`
- fatfs 0.4.0 auto-selects FAT12/16/32 based on volume size
- Timestamp handling: see spec §10.3

---

### Step 2.2: Create Guest Memory Block Device

**Status:** ✅ Complete

**Commit:** `db16ac33` - feat(guest): Add RawMemoryStorage for FAT filesystem access

**Goal:** Create a block device adapter for in-memory FAT access.

**Files created:**
- `src/hyperlight_guest/src/fs/fat_backend.rs`

**Types implemented:**
```rust
/// Error type for memory I/O operations.
pub enum MemoryIoError {
    OutOfBounds,
    InvalidSeek,
    UnexpectedEof,
    WriteZero,
}

/// Raw memory storage adapter for fatfs.
/// Wraps a memory region and implements fatfs I/O traits.
pub struct RawMemoryStorage {
    base: *mut u8,
    size: usize,
    position: usize,
}
```

**Implementation notes:**
- Named `RawMemoryStorage` instead of `MemoryBlockDevice` to match fatfs terminology
- Implements fatfs traits: `IoBase`, `Read`, `Write`, `Seek` (not std::io)
- Constructor validates non-null pointer and non-zero size (panics on invalid)
- Seek handles edge cases including `offset > i64::MAX` overflow
- `MemoryIoError` implements `fatfs::IoError` trait

**Acceptance criteria:**
- [x] Implements fatfs I/O traits (IoBase, Read, Write, Seek)
- [x] Proper bounds checking
- [x] Works with fatfs::FileSystem
- [x] Comprehensive test coverage (9 unit tests)

---

### Step 2.3: GuestFat Wrapper and VFS Mount Table

**Status:** ✅ Complete

**Commit:** `405f6997` - feat(guest-fs): Refactor FAT backend into modular structure with VFS layer

**Goal:** Create `GuestFat` wrapper over `RawMemoryStorage` + `fatfs::FileSystem`, and VFS mount table for path routing.

**Files created:**
- `src/hyperlight_guest/src/fs/fat/mod.rs` - module organization
- `src/hyperlight_guest/src/fs/fat/storage.rs` - `RawMemoryStorage` (moved from fat_backend.rs)
- `src/hyperlight_guest/src/fs/fat/time.rs` - `HyperlightTimeProvider`
- `src/hyperlight_guest/src/fs/fat/error.rs` - `MemoryIoError` and error mapping
- `src/hyperlight_guest/src/fs/fat/filesystem.rs` - `GuestFat`, `FatStat`, `FatDirEntry`
- `src/hyperlight_guest/src/fs/fat/file.rs` - `GuestFatFile`
- `src/hyperlight_guest/src/fs/vfs.rs` - `Vfs`, `Mount`, `MountBackend`

**Implementation notes:**
- Refactored flat `fat_backend.rs` into `fat/` module with proper separation
- `HyperlightTimeProvider` returns FAT epoch (1980-01-01 00:00:00) - per spec §10.3
- VFS uses longest-prefix matching with mounts sorted by path length descending
- `GuestFat` provides high-level API over fatfs with `FsError` mapping
- `GuestFatFile` tracks read/write permissions per open mode

**Acceptance criteria:**
- [x] `GuestFat::from_memory` opens fatfs filesystem
- [x] File operations: open, read, write, seek, flush, truncate
- [x] Directory operations: read_dir, mkdir, rmdir, stat
- [x] Error mapping per spec §11.3
- [x] `Vfs::resolve` uses longest-prefix matching
- [x] Mount conflict detection (duplicate path returns `AlreadyExists`)

**Tests added:**
- [x] VFS path normalization tests
- [x] Mount matching tests (exact, prefix, root)
- [x] VFS resolution with multiple mounts
- [x] Error mapping tests for fatfs errors

---

### Step 2.4: Parse FAT Mounts from Manifest

**Status:** ✅ Complete

**Commit:** `3ebe5e50` - feat(guest-fs): Parse FAT mounts from manifest and build VFS

**Goal:** Initialize FAT filesystems from manifest during guest startup.

**Files modified:**
- `src/hyperlight_guest/src/fs/manifest.rs` - VFS integration, accessors, validation
- `src/hyperlight_guest/src/fs/mod.rs` - export vfs, vfs_mut
- `src/hyperlight_guest/src/fs/file.rs` - doc example fix
- `src/hyperlight_guest/src/fs/tests.rs` - test isolation fixes

**Implementation notes:**
- Added `vfs: Vfs` field to `FsState`
- `init()` iterates FAT mount inodes, creates `GuestFat` for each
- Root ReadOnly mount added as fallback (lowest priority due to longest-prefix)
- `vfs()` returns immutable ref, `vfs_mut()` is `unsafe fn` with safety docs
- Added validation: `guest_address != 0` and `size != 0` before GuestFat creation
- Added `is_initialized()` check to prevent double-init (returns `NotSupported`)
- Added `reset()` for test cleanup

**Acceptance criteria:**
- [x] FAT mounts detected in manifest by `InodeType::FatMount`
- [x] `GuestFat` initialized for each FAT mount
- [x] VFS populated with FAT mounts + RO root fallback
- [x] RO files still work as before (unchanged paths)
- [x] Double-init returns error
- [x] Invalid FAT mount params validated

---

### Step 2.5: Unified File Operations

**Status:** ✅ Complete

**Commit:** `7247b498` - feat(guest-fs): unified File API with VFS routing (Step 2.5)

**Goal:** Route file operations through VFS to appropriate backend.

**Files modified:**
- `src/hyperlight_guest/src/fs/file.rs` - Unified `File` enum, `OpenOptions` builder
- `src/hyperlight_guest/src/fs/mod.rs` - Updated exports, module docs
- `src/hyperlight_guest/src/fs/error.rs` - Added `InvalidArgument` variant
- `src/hyperlight_guest/src/fs/fat/file.rs` - Performance docs on `len()`
- `src/hyperlight_guest/src/fs/fat/mod.rs` - Cleaned unused re-exports
- `src/hyperlight_guest/src/fs/fat/storage.rs` - `#[allow(dead_code)]` for future methods
- `src/hyperlight_guest_capi/cbindgen.toml` - Exclude hyperlight-guest from parsing
- `src/hyperlight_guest_capi/src/fs.rs` - Handle FAT files gracefully (returns -1)

**Implementation notes:**
- Created unified `File` enum with `ReadOnly(RoFile)` and `Fat(GuestFatFile<'static>)` variants
- Added `OpenOptions` builder pattern for flexible file open modes (replaces `open_with_options`)
- Implements `embedded_io` traits: `Read`, `Seek`, `Write` on `File`
- VFS resolves paths using longest-prefix matching to determine backend
- `RoFile` must be `pub` because it's in a public enum variant (Rust requirement)
- C API only supports read-only files; FAT support deferred to future phase
- Fixed cbindgen panic by excluding hyperlight-guest from include list

**Acceptance criteria:**
- [x] `open()` routes to correct backend based on VFS resolution
- [x] `read()` works for both RO and FAT backends
- [x] `write()` works for FAT, returns `FsError::ReadOnly` for RO
- [x] `seek()` works for both backends
- [x] `stat()` works for both backends
- [x] `OpenOptions` builder provides clean API
- [x] C API handles FAT files gracefully (deferred full support)

---

## Phase 3: Host-Guest Integration

Wire FAT images into sandbox memory.

### Step 3.1: Update HyperlightFSImage for FAT

**Status:** ✅ Complete

**Goal:** Include FAT image data in the built filesystem image.

**Files to modify:**
- `src/hyperlight_host/src/hyperlight_fs/image.rs`
- `src/hyperlight_host/src/hyperlight_fs/builder.rs` (pass fat_mounts to build_image)

**Acceptance criteria:**
- [x] FAT images stored in HyperlightFSImage
- [x] `manifest_size()` includes FAT mount inodes
- [x] `mapped_files_region_size()` includes FAT images
- [x] `generate_manifest()` produces correct FAT mount entries

**Implementation notes:**
- Added `InodeEntryType` enum (File, Directory, FatMount)
- Added `FatMountStorage` struct for FAT mount metadata
- Memory layout: [RO files region][FAT mounts region]
- Extracted `round_up_to()` as `pub(crate)` utility in `layout.rs`
- Added `get_parent_path()` helper for parent directory lookup
- 13 unit tests for FAT mount functionality

---

### Step 3.2: Map FAT Data into Guest Memory (Linux Only)

**Status:** ✅ Complete

**Goal:** Map FAT image data into guest address space with RW permissions, including guest-side page fault handler updates.

**Files modified:**

*Host side:*
- `src/hyperlight_host/src/hyperlight_fs/builder.rs` - typestate pattern (NoFat/WithFat)
- `src/hyperlight_host/src/hyperlight_fs/image.rs` - cleaned up dead code, added file_summary()

*Guest side:*
- `src/hyperlight_guest/src/fs/manifest.rs` - FAT region tracking with FatRegionCell
- `src/hyperlight_guest/src/fs/mod.rs` - export is_fat_region()
- `src/hyperlight_guest_bin/src/paging.rs` - map_page_readwrite() already present
- `src/hyperlight_guest_bin/src/exceptions/handler.rs` - FAT-aware handling already present

**Key changes implemented:**

*Typestate pattern:*
- `HyperlightFSBuilder<NoFat>` - clonable, `build(&self)` borrows
- `HyperlightFSBuilder<WithFat>` - not clonable, `build(self)` consumes
- Adding FAT mount transforms NoFat → WithFat
- Prevents accidental sharing of FAT images with exclusive locks

*FAT region tracking:*
- `FatRegionCell` stores `Vec<FatRegion>` with base/size pairs
- Registered during `init()` BEFORE creating GuestFat
- `is_fat_region(addr)` checks if address is in FAT memory
- Page fault handler uses this to decide RO vs RW PTEs

*Refactoring:*
- `validate_and_normalize_guest_path()` unified helper
- `FAT_NOT_SUPPORTED_ON_WINDOWS` const for Windows stubs
- Renamed `list()` → `file_summary()` for clarity
- Removed dead accessor methods from image.rs (YAGNI)

**Acceptance criteria:**
- [x] Guest: `is_fat_region()` function added to manifest.rs
- [x] Guest: FAT regions tracked in FatRegionCell during init()
- [x] Guest: is_fat_region exported from fs module
- [x] Host: Typestate pattern prevents FAT mount sharing
- [x] Host: Builder API clean and well-documented
- [x] All tests pass (140 hyperlight_fs tests + full suite)

---

### Step 3.3: Guest FAT Mount Initialization

**Status:** ✅ Complete

**Goal:** Initialize FAT filesystems in guest during startup.

**Files modified:**
- `src/hyperlight_guest/src/fs/manifest.rs`

**Implementation notes:**
- Implemented in Phase 2 as part of Step 2.4 (manifest parsing)
- `init()` iterates FAT mount inodes, creates `GuestFat` for each
- VFS populated with `MountBackend::Fat` entries
- Root ReadOnly mount added as fallback (lowest priority)

**Acceptance criteria:**
- [x] FAT mounts initialized during `fs::init()`
- [x] VFS populated with FAT backends
- [x] Files in FAT mounts accessible

---

### Step 3.4: Guest-Created FAT Mounts

**Status:** ✅ Complete

**Goal:** Allow guests to create new FAT filesystems dynamically (per spec §5.1).

**Implementation:**
- `create_fat_mount()` allocates from guest heap and formats as FAT12/16/32
- New mount registered in VFS with guest-created flag
- `unmount()` removes guest-created mount and frees memory
- `unmount()` fails with `PermissionDenied` for host-provided mounts
- `is_guest_created_mount()` queries mount origin
- `get_fat_mount_data()` returns (address, size) for host extraction

**Files modified:**
- `src/hyperlight_guest/src/fs/mod.rs` - Added public APIs
- `src/hyperlight_guest/src/fs/vfs.rs` - VFS mount management
- `src/hyperlight_guest/src/fs/fat.rs` - FAT formatting with fatfs

**Acceptance criteria:**
- [x] `create_fat_mount()` allocates from heap and formats FAT
- [x] New mount appears in VFS
- [x] `unmount()` works for guest-created mounts
- [x] `unmount()` fails for host-provided mounts

---

## Phase 4: C API Implementation

Implement libc-compatible C API (per spec §6).

### Step 4.1: Core File Operations

**Status:** ✅ Complete

**Goal:** Implement open, close, read, write, lseek for C guests.

**Files modified:**
- `src/hyperlight_guest_capi/src/fs.rs`
- `src/hyperlight_guest/src/fs/fd.rs` - Added FatFdEntry with flags tracking

**Implementation notes:**
- Full O_* flag support: O_RDONLY, O_WRONLY, O_RDWR, O_CREAT, O_EXCL, O_TRUNC, O_APPEND
- O_APPEND seeks to EOF before every write (POSIX compliant)
- O_EXCL without O_CREAT returns EINVAL (catches bugs early)
- FatFdEntry struct tracks original open() flags for fcntl F_GETFL
- Consistent EBADF for writes to read-only file descriptors

**Acceptance criteria:**
- [x] All O_* flags handled per spec §6.1
- [x] Correct errno return codes per spec §11.2
- [x] C guest can read/write files

---

### Step 4.2: Directory Operations

**Status:** ✅ Complete

**Goal:** Implement opendir, readdir, closedir, mkdir, rmdir.

**Files modified:**
- `src/hyperlight_guest_capi/src/fs.rs`

**Implementation notes:**
- DirStream table with MAX_DIR_STREAMS=16 concurrent directory handles
- hl_dirent_t with d_ino (synthetic), d_type (DT_REG/DT_DIR), d_name
- readdir returns newline-separated names; readdir_entry for POSIX-style iteration
- closedir returns EBADF for invalid/NULL handles (POSIX compliant)

**Acceptance criteria:**
- [x] opendir/readdir/closedir work per spec §6.3
- [x] d_type set correctly (DT_REG, DT_DIR) per spec §6.2
- [x] mkdir/rmdir work for FAT mounts
- [x] Error codes per spec §11.2

---

### Step 4.3: Stat and Working Directory

**Status:** ✅ Complete

**Goal:** Implement stat, fstat, getcwd, chdir.

**Files modified:**
- `src/hyperlight_guest_capi/src/fs.rs`

**Implementation notes:**
- hl_stat_t with st_size, st_mode (faked per spec §10.4), timestamps
- fstat works for both RO and FAT file descriptors
- stat works for paths (resolves through VFS)
- getcwd returns current working directory with null terminator
- chdir validates path exists and is a directory

**Acceptance criteria:**
- [x] stat/fstat populate struct per spec §6.3
- [x] st_mode set correctly per spec §10.4
- [x] getcwd/chdir work per spec §5.1

---

### Step 4.4: Advanced Operations

**Status:** ✅ Complete

**Goal:** Implement openat, fcntl, unlink, rename, access, dup/dup2.

**Files modified:**
- `src/hyperlight_guest_capi/src/fs.rs`

**Implementation notes:**
- openat/mkdirat support AT_FDCWD; real dirfd returns ENOTSUP (documented)
- fcntl F_GETFL returns accurate stored flags including O_APPEND
- fcntl F_SETFL can modify O_APPEND (per POSIX, only append/nonblock/async changeable)
- dup/dup2 return ENOTSUP (FAT handles have mutable state, documented limitation)
- access supports R_OK, W_OK, F_OK; X_OK always returns EACCES (no exec)
- unlink/rename work for FAT files and directories

**Acceptance criteria:**
- [x] openat with `AT_FDCWD` works
- [x] fcntl `F_DUPFD`/`F_GETFL`/`F_SETFL` work
- [x] unlink/rename work per spec §6.3
- [x] access checks file permissions
- [x] dup/dup2 return ENOTSUP (documented)

---

## Phase 5: Host Extraction APIs

APIs for host to extract data after sandbox execution.

### Step 5.1: Sandbox File Access APIs

**Status:** ✅ Complete

**Goal:** Allow host to read/write files while sandbox is paused.

**Files modified:**
- `src/hyperlight_host/src/sandbox/initialized_multi_use.rs`

**Methods implemented on `MultiUseSandbox`:**
```rust
// Helper method to eliminate boilerplate
fn resolve_fat_image(&mut self, guest_path: &str) -> Result<(&mut FatImage, String)>;

// Public APIs
pub fn fs_stat(&mut self, guest_path: &str) -> Result<FatStat>;
pub fn fs_read_file(&mut self, guest_path: &str) -> Result<Vec<u8>>;
pub fn fs_read_dir(&mut self, guest_path: &str) -> Result<Vec<FatEntry>>;
pub fn fs_write_file(&mut self, guest_path: &str, data: &[u8]) -> Result<()>;
pub fn fs_mkdir(&mut self, guest_path: &str) -> Result<()>;
pub fn fs_remove_file(&mut self, guest_path: &str) -> Result<()>;
pub fn fs_remove_dir(&mut self, guest_path: &str) -> Result<()>;
pub fn fs_rename(&mut self, old_path: &str, new_path: &str) -> Result<()>;
pub fn fs_exists(&mut self, guest_path: &str) -> Result<bool>;
pub fn fs_open_file(&mut self, guest_path: &str) -> Result<FatFileReader<'_>>;
pub fn fs_create_file(&mut self, guest_path: &str) -> Result<FatFileWriter<'_>>;
```

**Implementation notes:**
- Added `hyperlight_fs: Option<HyperlightFSImage>` field to `MultiUseSandbox`
- `resolve_fat_image()` helper eliminates ~180 lines of duplication
- All methods validate HyperlightFS is configured and path is within FAT mount
- `fs_rename` validates both paths are in the same FAT mount

**Acceptance criteria:**
- [x] Read works for FAT files (`fs_read_file`)
- [x] Write works for FAT files (`fs_write_file`)
- [x] Stat works for FAT files (`fs_stat`)
- [x] Directory listing works (`fs_read_dir`)
- [x] Directory operations work (`fs_mkdir`, `fs_remove_dir`)
- [x] File deletion works (`fs_remove_file`)
- [x] Rename/move works (`fs_rename`)
- [x] Path existence check works (`fs_exists`)
- [x] Streaming read works (`fs_open_file`)
- [x] Streaming write works (`fs_create_file`)
- [x] Error if no HyperlightFS configured
- [x] Error if path not within FAT mount

---

### Step 5.2: Builder Pattern API

**Status:** ✅ Complete

**Goal:** Clean builder pattern for attaching HyperlightFS to sandboxes.

**Files modified:**
- `src/hyperlight_host/src/sandbox/uninitialized.rs`

**Implementation:**
```rust
impl UninitializedSandbox {
    /// Sets the HyperlightFS image using builder pattern.
    #[must_use]
    pub fn with_hyperlight_fs(mut self, fs_image: HyperlightFSImage) -> Self;
}
```

**Acceptance criteria:**
- [x] `with_hyperlight_fs()` returns `Self` for chaining
- [x] `#[must_use]` attribute prevents accidental discard
- [x] Documentation with working `no_run` example
- [x] Unit test verifies builder pattern

---

### Step 5.3: Integration Tests for Host APIs

**Status:** ✅ Complete

**Goal:** Comprehensive tests for host-side FAT file operations.

**Files modified:**
- `src/hyperlight_host/tests/hyperlight_fs_test.rs`

**Tests added:**
- `test_sandbox_fs_stat` - Metadata retrieval
- `test_sandbox_fs_read_file` - File content reading
- `test_sandbox_fs_write_and_read_file` - Write then read back
- `test_sandbox_fs_read_dir` - Directory listing
- `test_sandbox_fs_read_dir_file_sizes` - File sizes in listings
- `test_sandbox_fs_mkdir` - Directory creation
- `test_sandbox_fs_remove_file` - File deletion
- `test_sandbox_fs_remove_dir` - Directory deletion
- `test_sandbox_fs_remove_dir_not_empty_fails` - Non-empty dir rejection
- `test_sandbox_fs_rename_file` - File renaming
- `test_sandbox_fs_rename_directory` - Directory renaming
- `test_sandbox_fs_rename_to_existing_fails` - Destination exists check
- `test_sandbox_fs_rename_move_to_subdir` - Move file to subdirectory
- `test_sandbox_fs_rename_cross_mount_fails` - Cross-mount rejection
- `test_sandbox_fs_rename_root_fails` - Root rename rejection
- `test_sandbox_fs_exists` - Path existence checks
- `test_sandbox_fs_open_file_streaming` - Streaming read
- `test_sandbox_fs_create_file_streaming` - Streaming write
- `test_sandbox_fs_full_workflow` - End-to-end workflow
- `test_sandbox_fs_no_hyperlight_fs_error` - No FS configured
- `test_sandbox_fs_path_not_in_mount_error` - Path validation

**Acceptance criteria:**
- [x] All 21 host API tests pass
- [x] Tests cover error conditions
- [x] Tests verify both Rust and C guests (via existing test infrastructure)

---

## Phase 6: Testing & Documentation

### Step 6.1: Integration Tests

**Status:** ✅ Complete

**Goal:** Comprehensive integration tests.

**Files created:**
- `src/hyperlight_host/tests/hyperlight_fs_test.rs` - 12 FAT integration tests
- `src/tests/rust_guests/simpleguest/src/main.rs` - FAT guest functions

**Tests implemented:**
- `test_guest_fat_crud_operations` - Create, read, update, delete cycle
- `test_guest_fat_existing_image` - Load pre-seeded FAT image
- `test_guest_fat_root_level_minimal` - Root-level writes
- `test_guest_fat_root_after_mkdir_rmdir` - State cleanup
- `test_guest_mixed_ro_and_fat` - Mixed RO and FAT mounts
- `test_guest_fat_file_overwrite` - File content replacement
- `test_guest_fat_nested_dirs` - Deep directory creation
- `test_guest_fat_error_cases` - Error handling
- `test_guest_fat_rename` - File/directory renaming
- `test_guest_fat_stat` - Metadata queries
- `test_guest_fat_large_file` - Large file handling (~10KB)
- `test_guest_fat_cwd_operations` - Working directory support

**Guest functions added:**
- `WriteFatFile`, `ReadFatFile`, `DeleteFatFile`
- `MkdirFat`, `RmdirFat`, `ListDirFat`
- `RenameFat`, `StatFatSize`, `ExistsFat`
- `GetCwd`, `Chdir`

**Acceptance criteria:**
- [x] All 12 integration tests pass
- [x] Tests cover error conditions
- [x] Tests use helper functions to reduce boilerplate

---

### Step 6.2: C Guest Test Code

**Status:** ✅ Complete

**Goal:** C test guest for validating C API with FAT filesystem support.

**Files modified:**
- `src/tests/c_guests/c_simpleguest/main.c` - 24+ C API test functions
- `src/hyperlight_host/tests/hyperlight_fs_test.rs` - 55 integration tests

**Tests implemented:**
- Core operations: open flags (O_RDONLY, O_WRONLY, O_RDWR, O_CREAT, O_EXCL, O_TRUNC, O_APPEND)
- Directory operations: opendir, readdir_entry, closedir, mkdir, rmdir
- Stat operations: stat, fstat on files and directories
- Working directory: getcwd, chdir
- Advanced: fcntl (F_GETFL, F_SETFL), access (R_OK, W_OK, F_OK), openat, mkdirat
- Error handling: EBADF, EINVAL, ENOENT, ENOTSUP for unsupported ops
- POSIX compliance tests: O_APPEND after lseek, fcntl flag accuracy

**Acceptance criteria:**
- [x] C guest builds with FAT support
- [x] All C APIs exercised (open, read, write, close, lseek, stat, mkdir, etc.)
- [x] Integration tests callable from host (55 tests total)
- [x] Matches Rust guest test coverage

---

### Step 6.3: Update Documentation

**Status:** ✅ Complete

**Goal:** Update documentation for FAT support.

**Files modified:**
- `README.md` - Added FAT tooling prerequisites
- `src/hyperlight_guest/src/fs/mod.rs` - Module-level docs
- `src/hyperlight_guest/src/fs/file.rs` - API docs
- `src/hyperlight_host/src/hyperlight_fs/builder.rs` - Builder API docs

**Acceptance criteria:**
- [x] Doc examples use real APIs (verified)
- [x] Fixed broken doc reference (`open_with_options` → `OpenOptions`)
- [x] All doc tests pass

---

### Step 6.4: Validation Tool and Examples

**Status:** ✅ Complete

**Goal:** Update hyperlight_fs_validate for FAT configuration and add demo example.

**Files created:**
- `src/hyperlight_host/examples/hyperlight_fs_demo.rs` - Comprehensive FAT demo
- `src/hyperlight_host/examples/assets/hyperlight-fs-example.toml` - Example config
- `src/hyperlight_host/examples/assets/test_fat.img` - Pre-seeded test image
- `src/hyperlight_host/scripts/create-fat32-image.sh` - FAT image creation tool
- `src/hyperlight_host/scripts/inspect-fat32-image.sh` - FAT image inspection tool

**Files modified:**
- `src/hyperlight_host/examples/hyperlight_fs_validate.rs` - FAT mount display

**Acceptance criteria:**
- [x] hyperlight_fs_validate displays FAT mounts
- [x] hyperlight_fs_demo works end-to-end
- [x] Helper scripts documented
- [x] Example compiles and runs

---

## Deferred Items

Items deferred from initial implementation:

| Item | Spec Reference | Notes |
|------|----------------|-------|
| Host extraction APIs (`fs_fat_info`, `fs_persist_fat_mount`) | §4.5 (TBD) | Security analysis needed for guest-to-host data flow |
| `FsLimits` struct | §13.4 | YAGNI |
| Shared locks for RO files | §12 | Not needed yet |

*End of Plan*
