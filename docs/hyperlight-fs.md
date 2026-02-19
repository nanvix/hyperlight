# HyperlightFS

HyperlightFS is the filesystem subsystem for Hyperlight sandboxes. It provides both **read-only** memory-mapped files and **read-write** FAT filesystems—all with zero-copy efficiency.

## Key Features

### Read-Only Files
- **Zero-copy**: Files are memory-mapped (`mmap`), guests read directly from the host's page cache
- **Kernel-enforced read-only**: `PROT_READ` at mmap level, guests cannot modify host files
- **Demand paging**: File pages loaded on-demand via page faults
- **Shareable**: Multiple sandboxes can share the same RO mappings

### Read-Write FAT Filesystems
- **Full read-write**: Create, write, rename, delete files and directories
- **Zero-copy via MAP_SHARED**: Guest writes go directly to backing file via shared pages
- **Auto-persist**: Changes flushed to disk by the OS—no explicit save needed
- **Multiple options**: Use existing FAT images, create persistent ones, or ephemeral temp storage
- **Guest-created mounts**: Guests can dynamically create FAT filesystems at runtime

### Common Features
- **Unified guest API**: Same functions work for both RO and RW files
- **Explicit mapping**: Nothing exposed unless explicitly configured
- **TOML configuration**: Define mappings in config files or programmatically
- **Pattern matching**: Map directories with gitignore-style include/exclude patterns

## Quick Start

### 1. Build a filesystem image

```rust
use hyperlight_host::hyperlight_fs::HyperlightFSBuilder;

let fs = HyperlightFSBuilder::new()
    // Read-only files (zero-copy from host)
    .add_file("/etc/config.json", "/config.json")?
    .add_dir("/opt/app/assets", "/assets")?
        .include("**/*.json")
        .exclude("**/secret/*")
        .done()?
    // Read-write FAT mount (1MB, persists to host file)
    .add_empty_fat_mount_at("/var/data/storage.fat", "/data", 1024 * 1024)?
    .build()?;
```

### 2. Attach to a sandbox

```rust
use hyperlight_host::sandbox::UninitializedSandbox;

let sandbox: MultiUseSandbox = UninitializedSandbox::new(guest_binary, None)?
    .with_hyperlight_fs(fs)
    .evolve()?;
```

### 3. Access files in the guest

```rust
// In guest code
use hyperlight_guest::fs::{self, OpenOptions};

// Read from RO mount
let mut config = fs::open("/config.json")?;
let content = config.read_to_vec()?;

// Write to FAT mount
let mut output = OpenOptions::new()
    .write(true)
    .create(true)
    .open("/data/output.txt")?;
output.write_all(b"Hello from guest!")?;
```

> 📘 **Full demo**: See [hyperlight_fs_demo.rs](../src/hyperlight_host/examples/hyperlight_fs_demo.rs) for a complete working example.

## How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Host Process                                 │
├─────────────────────────────────────────────────────────────────────┤
│  HyperlightFSBuilder                                                │
│    ├── RO files: mmap(PROT_READ) → read-only pages                  │
│    ├── FAT mounts: mmap(MAP_SHARED) → shared RW pages               │
│    └── Serialize manifest (FlatBuffer)                              │
├─────────────────────────────────────────────────────────────────────┤
│                         Guest VM                                     │
├─────────────────────────────────────────────────────────────────────┤
│  Shared Memory                                                      │
│    ├── Manifest: file paths, sizes, guest addresses, mount types    │
│    ├── RO region: demand-paged, read-only                           │
│    └── FAT region: demand-paged, read-write (same pages as host!)   │
│                                                                     │
│  Guest VFS                                                          │
│    ├── fs::open() → longest-prefix mount matching                   │
│    ├── RO file read → direct memory access                          │
│    └── FAT write → modifies shared pages → OS flushes to disk       │
└─────────────────────────────────────────────────────────────────────┘
```

### Read-Only Flow
1. Host mmaps file with `PROT_READ`
2. Same pages mapped into guest with read-only permissions
3. Guest reads trigger page faults → pages loaded from disk on demand
4. Multiple sandboxes share same physical pages via OS page cache

### FAT Read-Write Flow
1. Host mmaps FAT image with `MAP_SHARED` + acquires `flock(LOCK_EX)`
2. Same physical pages mapped into guest with RW permissions
3. Guest writes go directly to shared pages (zero-copy!)
4. OS flushes dirty pages to backing file asynchronously
5. Lock released when sandbox drops → changes persisted

## Host API

### Read-Only Files

```rust
use hyperlight_host::hyperlight_fs::HyperlightFSBuilder;

let fs = HyperlightFSBuilder::new()
    .add_file("/etc/config.json", "/config.json")?     // Single file
    .add_dir("/opt/app/assets", "/assets")?            // Directory with patterns
        .include("**/*.json")
        .exclude("**/secret/*")
        .done()?
    .build()?;
```

### FAT Mounts

Three ways to add read-write FAT filesystems:

```rust
let fs = HyperlightFSBuilder::new()
    // 1. Use an existing FAT image
    .add_fat_image("/var/data/existing.fat", "/data")?
    
    // 2. Create persistent FAT at specific path (survives sandbox drop)
    .add_empty_fat_mount_at("/var/data/output.fat", "/out", 2 * 1024 * 1024)?
    
    // 3. Create ephemeral FAT (auto-deleted when sandbox drops)
    .add_empty_fat_mount("/tmp", 1024 * 1024)?
    
    .build()?;
```

### Host-Side File Access

While the sandbox is paused, the host can read/write FAT mounts:

```rust
let mut sandbox: MultiUseSandbox = /* ... */;

// Pre-populate before guest runs
sandbox.fs_write_file("/data/input.txt", b"Hello guest!")?;
sandbox.fs_mkdir("/data/logs")?;

// Execute guest
let result: String = sandbox.call("process", ())?;

// Read results after guest completes
let output = sandbox.fs_read_file("/data/output.txt")?;
for entry in sandbox.fs_read_dir("/data/logs")? {
    println!("{}: {} bytes", entry.name, entry.size);
}
```

### Preview Before Building

```rust
let builder = HyperlightFSBuilder::new()
    .add_file("/etc/config.json", "/config.json")?
    .add_dir("/opt/app/data", "/data")?
        .include("**/*")
        .done()?;

let manifest = builder.file_summary()?;
println!("Would map {} files, {} bytes total", 
    manifest.files.len(), 
    manifest.total_size);
```

## Guest API

The guest filesystem API is available in both Rust and C.

### Rust API

```rust
use hyperlight_guest::fs::{self, OpenOptions};

// Read from any mount (RO or FAT)
let mut file = fs::open("/config.json")?;
let content = file.read_to_vec()?;

// Write to FAT mount
let mut output = OpenOptions::new()
    .write(true)
    .create(true)
    .truncate(true)
    .open("/data/output.txt")?;
output.write_all(b"Results...")?;

// Create directories (FAT only)
fs::mkdir("/data/logs")?;

// List directory contents
for entry in fs::read_dir("/data")? {
    println!("{}: {} bytes, dir={}", entry.name, entry.size, entry.is_dir);
}

// Guest can create dynamic FAT mounts at runtime
fs::create_fat_mount("/scratch", 256 * 1024)?;  // 256KB from guest heap
```

### C API

The C API provides libc-compatible functions via `hyperlight_guest.h`:

```c
#include "hyperlight_guest.h"

// Read file
int fd = open("/config.json", O_RDONLY);
char buf[1024];
ssize_t n = read(fd, buf, sizeof(buf));
close(fd);

// Write file (FAT mount)
fd = open("/data/output.txt", O_WRONLY | O_CREAT | O_TRUNC);
write(fd, "Results", 7);
close(fd);

// Create directory
mkdir("/data/logs", 0755);  // mode ignored - FAT has no permissions

// List directory
hl_hl_DIR *dir = opendir("/data");
hl_hl_dirent_t *entry;
while ((entry = readdir(dir)) != NULL) {
    LOG(Info, entry->d_name);  // Use LOG macro for output
}
closedir(dir);

// Guest-created dynamic FAT
hl_fs_create_fat_mount("/scratch", 256 * 1024);
```

**C API Summary:**

| Category | Functions |
|----------|----------|
| File I/O | `open`, `close`, `read`, `write`, `lseek` |
| Metadata | `stat`, `fstat`, `access` |
| Directories | `mkdir`, `rmdir`, `opendir`, `readdir`, `closedir`, `chdir`, `getcwd` |
| File ops | `unlink`, `rename` |
| Advanced | `dup`, `dup2`, `fcntl`, `openat`, `mkdirat` |
| FAT | `hl_fs_create_fat_mount`, `hl_fs_unmount` |
The filesystem is automatically initialized during guest startup if the host configured HyperlightFS.

## TOML Configuration

Define filesystem mappings in a TOML file:

```toml
# hyperlight-fs.toml

# Read-only file mappings
[[file]]
host_path = "/etc/app/config.json"
guest = "/config.json"

# Read-only directory with patterns
[[directory]]
host_path = "/opt/app/assets"
guest = "/assets"
include = ["**/*.json", "**/*.txt"]
exclude = ["**/secret/*"]

# FAT image from existing file
[[fat_image]]
host_path = "/var/lib/app/data.fat"
mount_point = "/data"

# Empty FAT (ephemeral - deleted on drop)
[[fat_mount]]
mount_point = "/tmp"
size = "1MB"  # Supports KB, MB, GB, KiB, MiB, GiB

# Empty FAT (persistent - at specific path)
[[fat_mount]]
host_path = "/var/lib/app/logs.fat"
mount_point = "/logs"
size = "10MB"
```

### Loading Configuration

```rust
use hyperlight_host::hyperlight_fs::{HyperlightFSBuilder, HyperlightFsConfig};

// From file
let fs = HyperlightFSBuilder::from_toml_file("hyperlight-fs.toml")?.build()?;

// Or from string
let fs = HyperlightFSBuilder::from_toml(r#"
[[file]]
host_path = "/etc/config.json"
guest = "/config.json"

[[fat_mount]]
mount_point = "/tmp"
size = "1MB"
"#)?.build()?;
```

### Tools and Examples

**Validation tool** - check config files before deployment:
```bash
cargo run --example hyperlight_fs_validate -- hyperlight-fs.toml
```

**Demo** - full RO + FAT demonstration:
```bash
cargo run --release --example hyperlight_fs_demo
```
See [hyperlight_fs_demo.rs](../src/hyperlight_host/examples/hyperlight_fs_demo.rs) for the source.

**Stress test** - large file random reads:
```bash
cargo run --release --example hyperlight_fs_stress -- --size 512 --guest rust
```

## Path Requirements

### Host Paths

| Requirement | Valid | Invalid |
|-------------|-------|---------|
| Must be absolute | `/etc/config.json` | `config.json` |
| No `..` components | `/opt/app/file.txt` | `/opt/../etc/passwd` |

### Guest Paths

| Requirement | Valid | Invalid |
|-------------|-------|---------|
| Must be absolute | `/config.json` | `config.json` |
| No `..` components | `/data/file.txt` | `/data/../secret` |
| No null bytes | `/file.txt` | `/file\0.txt` |

Guest paths are normalized automatically:
- `//foo//bar` → `/foo/bar`
- `/foo/` → `/foo`
- `/./foo` → `/foo`

## Error Handling

All errors are returned as `HyperlightError`:

```rust
// File doesn't exist
HyperlightFSBuilder::new().add_file("/nonexistent", "/file");
// Error: Cannot add file "/nonexistent": No such file or directory

// Symlink rejected (security)
HyperlightFSBuilder::new().add_file("/path/to/symlink", "/file");
// Error: Cannot add "/path/to/symlink": symlinks are not supported

// Path traversal blocked
HyperlightFSBuilder::new().add_file("/etc/passwd", "/guest/../escape");
// Error: Invalid guest path: '..' components are not allowed

// Duplicate guest path
HyperlightFSBuilder::new()
    .add_file("/file1", "/same")?
    .add_file("/file2", "/same");
// Error: Duplicate guest path "/same": already mapped
```

## Limitations

### Platform Support

| Platform | RO Files | FAT Mounts | Notes |
|----------|----------|------------|-------|
| Linux | ✅ | ✅ | Full support |
| Windows | ❌ | ❌ | TODO - requires surrogate process changes |

### FAT Filesystem Limits

| Limit | Value | Notes |
|-------|-------|-------|
| Min image size | 1 MB | Practical minimum |
| Max image size | 16 GB | Implementation limit |
| Max file size | 4 GB - 1 | FAT32 limit |
| Max filename | 255 chars | Long filenames supported |
| Permissions | None | FAT has no Unix permissions |
| Symlinks | None | Not supported |

### File Types

| Type | Supported |
|------|-----------|
| Regular files | ✅ |
| Directories | ✅ |
| Symlinks | ❌ (rejected for security) |
| Device files, sockets, FIFOs | ❌ |

### Concurrency

- **FAT images have exclusive locks** — One sandbox per backing file
- **RO files have no locking** — External modification during execution risks SIGBUS
- **Guest-created FAT** — Private to that guest, not accessible from host

### SIGBUS Risk

If a host modifies/truncates a mapped RO file while the sandbox runs, accessing invalid regions causes `SIGBUS`.

**Mitigation**: Don't modify mapped files during sandbox execution.

## Security

1. **No implicit mappings** — Nothing exposed unless explicitly added
2. **Path validation** — Absolute paths required, no `..` traversal
3. **Symlink rejection** — Prevents TOCTOU attacks
4. **Kernel-enforced read-only** — RO files use `PROT_READ` at mmap level
5. **FAT exclusive locking** — `flock(LOCK_EX)` prevents multi-sandbox corruption
6. **Guest isolation** — Guests cannot access unmapped host paths

## Testing

Comprehensive tests cover all FAT functionality:

```bash
# Run all HyperlightFS tests
cargo test -p hyperlight-host --test hyperlight_fs_test

# Key test categories (82 tests total):
# - test_guest_fat_* : Guest FAT operations (CRUD, rename, stat, etc.)
# - test_cross_validation_* : Host-guest interoperability via MAP_SHARED
# - test_guest_created_fat_* : Dynamic guest-created FAT mounts
# - test_c_api_* : C API coverage
```

See [hyperlight_fs_test.rs](../src/hyperlight_host/tests/hyperlight_fs_test.rs) for details.

## Further Reading

- **[HyperlightFS Specification](hyperlight-fs-fat-spec.md)** — Complete technical spec with API reference, FlatBuffer schema, memory layout, POSIX compliance notes
- **[hyperlight_fs_demo.rs](../src/hyperlight_host/examples/hyperlight_fs_demo.rs)** — Working example with RO, FAT, and guest-created mounts

## Future Work

- **Windows support** — HyperlightFS not yet implemented on Windows/WHP
- **Shared locks for RO files** — `flock(LOCK_SH)` to prevent SIGBUS from external modification
- **Read-only FAT mounts** — Mount existing FAT images as read-only (shareable)

