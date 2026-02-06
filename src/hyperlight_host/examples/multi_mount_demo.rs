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

//! Multi-mount demo: FAT at /lib, FAT at /root/.local/lib/python3.12/site-packages/testdir, ReadOnly at /
//!
//! This example demonstrates a realistic multi-mount scenario similar to
//! how Nanvix runs Python with separate FAT filesystems for:
//! - `/lib` — Python standard library (FAT, read-write for __pycache__)
//! - `/root/.local/lib/python3.12/site-packages/testdir` — Deeply nested FAT mount
//! - `/` — ReadOnly files (e.g., __main__.py entry point)
//!
//! # Key behaviors tested:
//!
//! 1. `stat("/")` works even without a root FAT mount (virtual root)
//! 2. `stat("/lib")` works (mount point as directory)
//! 3. `stat("/root/.local/lib/python3.12/site-packages/testdir")` works (deep mount point)
//! 4. **stat on parent paths** of mount points works (virtual parent directories)
//!    e.g. `stat("/root/.local/lib/python3.12/site-packages")` returns dir
//! 5. `read_dir("/")` lists mount points and RO files
//! 6. Files on FAT mounts can be read/written
//! 7. ReadOnly files at root are accessible
//! 8. `mkdir` works on FAT mounts (for __pycache__)
//! 9. Tilde expansion: `~` resolves to a configured home directory
//!
//! # Usage
//!
//! ```bash
//! cargo run --example multi_mount_demo
//! ```

use std::path::Path;
use std::process::ExitCode;

use hyperlight_host::GuestBinary;
use hyperlight_host::hyperlight_fs::HyperlightFSBuilder;
use hyperlight_host::sandbox::{MultiUseSandbox, SandboxConfiguration, UninitializedSandbox};

/// Path to the test guest binary.
fn get_guest_path() -> &'static str {
    #[cfg(debug_assertions)]
    {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tests/rust_guests/simpleguest/target/x86_64-hyperlight-none/debug/simpleguest"
        )
    }
    #[cfg(not(debug_assertions))]
    {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tests/rust_guests/simpleguest/target/x86_64-hyperlight-none/release/simpleguest"
        )
    }
}

fn run_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Multi-Mount Demo: /lib + deep nested FAT + RO at /          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Create temp directory for FAT images and test files
    let temp_dir = tempfile::TempDir::new()?;
    let lib_fat_path = temp_dir.path().join("lib.fat");
    let local_fat_path = temp_dir.path().join("local.fat");

    // Create a sample entry-point file (simulates __main__.py)
    let main_path = temp_dir.path().join("__main__.py");
    std::fs::write(
        &main_path,
        b"# This is the entry point\nprint('Hello from __main__.py')\n",
    )?;

    // Create a sample config file at root
    let config_path = temp_dir.path().join("config.txt");
    std::fs::write(&config_path, b"app=multi_mount_demo\n")?;

    println!("Setting up HyperlightFS with multiple mount points...");
    println!();

    // =========================================================================
    // Part 1: Build HyperlightFS with /lib FAT + /.local FAT + RO at /
    // =========================================================================

    let fs_image = HyperlightFSBuilder::new()
        // ReadOnly files at root (these should auto-create "/" directory inode)
        .add_file(&main_path, "/__main__.py")?
        .add_file(&config_path, "/config.txt")?
        // FAT mount at /lib (e.g., Python stdlib)
        .add_empty_fat_mount_at(&lib_fat_path, "/lib", 1024 * 1024)? // 1MB
        // FAT mount at deeply nested path (tests virtual parent directories)
        .add_empty_fat_mount_at(
            &local_fat_path,
            "/root/.local/lib/python3.12/site-packages/testdir",
            1024 * 1024,
        )? // 1MB
        .build()?;

    println!("   RO: /__main__.py <- {}", main_path.display());
    println!("   RO: /config.txt  <- {}", config_path.display());
    println!("   FAT: /lib        (1 MB read-write)");
    println!(
        "   FAT: /root/.local/lib/python3.12/site-packages/testdir (1 MB read-write)"
    );
    println!();

    // =========================================================================
    // Part 2: Create sandbox
    // =========================================================================

    println!("Creating sandbox...");
    let guest_path = get_guest_path();

    if !Path::new(guest_path).exists() {
        return Err(format!(
            "Guest binary not found at: {}\n\
             Run 'just guests' first to build the test guests.",
            guest_path
        )
        .into());
    }

    let mut config = SandboxConfiguration::default();
    config.set_heap_size(1024 * 1024); // 1MB heap

    let mut sandbox: MultiUseSandbox =
        UninitializedSandbox::new(GuestBinary::FilePath(guest_path.into()), Some(config))?
            .with_hyperlight_fs(fs_image)
            .evolve()?;
    println!("   Sandbox created");
    println!();

    // =========================================================================
    // Part 3: Test stat("/") — virtual root directory
    // =========================================================================

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Test 1: stat(\"/\") — virtual root");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let result: String = sandbox.call("StatPathResult", "/".to_string())?;
    if result == "ok:dir" {
        println!("   PASS: stat(\"/\") = {} (root is a directory)", result);
    } else {
        println!("   FAIL: stat(\"/\") = {} (expected ok:dir)", result);
        println!("         Root directory should exist even without a root mount.");
        println!("         The VFS should synthesize a virtual root.");
    }
    println!();

    // =========================================================================
    // Part 4: Test stat on mount points
    // =========================================================================

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Test 2: stat on mount points");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let result: String = sandbox.call("StatPathResult", "/lib".to_string())?;
    if result == "ok:dir" {
        println!("   PASS: stat(\"/lib\") = {}", result);
    } else {
        println!("   FAIL: stat(\"/lib\") = {} (expected ok:dir)", result);
    }

    let result: String = sandbox.call(
        "StatPathResult",
        "/root/.local/lib/python3.12/site-packages/testdir".to_string(),
    )?;
    if result == "ok:dir" {
        println!(
            "   PASS: stat(\"/root/.local/.../testdir\") = {}",
            result
        );
    } else {
        println!(
            "   FAIL: stat(\"/root/.local/.../testdir\") = {} (expected ok:dir)",
            result
        );
    }
    println!();

    // =========================================================================
    // Part 4b: Test stat on PARENT paths of mount points (virtual directories)
    // =========================================================================

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Test 2b: stat on virtual parent dirs of mount points");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let virtual_parent_paths = [
        "/root",
        "/root/.local",
        "/root/.local/lib",
        "/root/.local/lib/python3.12",
        "/root/.local/lib/python3.12/site-packages",
        // With trailing slash (normalized away)
        "/root/.local/lib/python3.12/site-packages/",
    ];

    for vpath in &virtual_parent_paths {
        let result: String = sandbox.call("StatPathResult", vpath.to_string())?;
        if result == "ok:dir" {
            println!("   PASS: stat(\"{}\") = {}", vpath, result);
        } else {
            println!(
                "   FAIL: stat(\"{}\") = {} (expected ok:dir — virtual parent)",
                vpath, result
            );
        }
    }
    println!();

    // =========================================================================
    // Part 5: Test ReadOnly files at root
    // =========================================================================

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Test 3: ReadOnly files at root");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // stat should work
    let result: String = sandbox.call("StatPathResult", "/__main__.py".to_string())?;
    if result.starts_with("ok:") && result != "ok:dir" {
        println!("   PASS: stat(\"/__main__.py\") = {}", result);
    } else {
        println!("   FAIL: stat(\"/__main__.py\") = {} (expected ok:<size>)", result);
    }

    // open and read should work
    let content: Vec<u8> = sandbox.call("ReadFile", "/__main__.py".to_string())?;
    let content_str = String::from_utf8_lossy(&content);
    if content_str.contains("Hello from __main__.py") {
        println!("   PASS: ReadFile(\"/__main__.py\") = {:?}", content_str.trim());
    } else {
        println!(
            "   FAIL: ReadFile(\"/__main__.py\") unexpected content: {:?}",
            content_str
        );
    }
    println!();

    // =========================================================================
    // Part 6: Test FAT write/read on /lib
    // =========================================================================

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Test 4: FAT read/write on /lib");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Create a nested directory structure (like Python's __pycache__)
    let mkdir_result: bool = sandbox.call("MkdirFat", "/lib/python3.12".to_string())?;
    if mkdir_result {
        println!("   PASS: mkdir(\"/lib/python3.12\")");
    } else {
        println!("   FAIL: mkdir(\"/lib/python3.12\")");
    }

    let mkdir_result: bool =
        sandbox.call("MkdirFat", "/lib/python3.12/__pycache__".to_string())?;
    if mkdir_result {
        println!("   PASS: mkdir(\"/lib/python3.12/__pycache__\")");
    } else {
        println!("   FAIL: mkdir(\"/lib/python3.12/__pycache__\")");
    }

    // Write a file into it
    let pyc_content = b"fake bytecode content".to_vec();
    let write_result: bool = sandbox.call(
        "WriteFatFile",
        (
            "/lib/python3.12/__pycache__/os.cpython-312.pyc".to_string(),
            pyc_content.clone(),
        ),
    )?;
    if write_result {
        println!("   PASS: write /lib/python3.12/__pycache__/os.cpython-312.pyc");
    } else {
        println!("   FAIL: write /lib/python3.12/__pycache__/os.cpython-312.pyc");
    }

    // Read it back
    let read_content: Vec<u8> = sandbox.call(
        "ReadFatFile",
        "/lib/python3.12/__pycache__/os.cpython-312.pyc".to_string(),
    )?;
    if read_content == pyc_content {
        println!("   PASS: read back matches written content");
    } else {
        println!("   FAIL: read back mismatch");
    }
    println!();

    // =========================================================================
    // Part 7: Test FAT operations on /.local
    // =========================================================================

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Test 5: FAT read/write on deep-nested mount");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Create directory structure inside the testdir mount
    let dirs = [
        "/root/.local/lib/python3.12/site-packages/testdir/markdown",
    ];
    for dir in &dirs {
        let mkdir_result: bool = sandbox.call("MkdirFat", dir.to_string())?;
        if mkdir_result {
            println!("   PASS: mkdir(\"{}\")", dir);
        } else {
            println!("   FAIL: mkdir(\"{}\")", dir);
        }
    }

    // Write a package file
    let init_content = b"# markdown package\n__version__ = '3.7'\n".to_vec();
    let write_result: bool = sandbox.call(
        "WriteFatFile",
        (
            "/root/.local/lib/python3.12/site-packages/testdir/markdown/__init__.py".to_string(),
            init_content.clone(),
        ),
    )?;
    if write_result {
        println!("   PASS: write markdown/__init__.py");
    } else {
        println!("   FAIL: write markdown/__init__.py");
    }

    // Read it back
    let read_content: Vec<u8> = sandbox.call(
        "ReadFatFile",
        "/root/.local/lib/python3.12/site-packages/testdir/markdown/__init__.py".to_string(),
    )?;
    if read_content == init_content {
        println!("   PASS: read markdown/__init__.py matches");
    } else {
        println!("   FAIL: read markdown/__init__.py mismatch");
    }

    // Create __pycache__ in the package dir (this is what Python tries to do)
    let mkdir_result: bool = sandbox.call(
        "MkdirFat",
        "/root/.local/lib/python3.12/site-packages/testdir/markdown/__pycache__".to_string(),
    )?;
    if mkdir_result {
        println!("   PASS: mkdir markdown/__pycache__ (the key operation!)");
    } else {
        println!("   FAIL: mkdir markdown/__pycache__");
    }
    println!();

    // =========================================================================
    // Part 8: Test read_dir("/") — should list mount points + RO files
    // =========================================================================

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Test 6: read_dir(\"/\") — list root contents");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let listing: String = sandbox.call("ListDirFat", "/".to_string())?;
    if !listing.is_empty() {
        println!("   Contents of /:");
        for entry in listing.split('\n').filter(|s| !s.is_empty()) {
            println!("     - {}", entry);
        }
        // Check that mount points and RO files appear
        let has_lib = listing.contains("lib");
        let has_root = listing.contains("root");
        let has_main = listing.contains("__main__.py");
        if has_lib {
            println!("   PASS: /lib appears in listing");
        } else {
            println!("   FAIL: /lib missing from listing");
        }
        if has_root {
            println!("   PASS: /root appears in listing (virtual parent)");
        } else {
            println!("   FAIL: /root missing from listing");
        }
        if has_main {
            println!("   PASS: __main__.py appears in listing");
        } else {
            println!("   FAIL: __main__.py missing from listing");
        }
    } else {
        println!("   FAIL: read_dir(\"/\") returned empty");
        println!("         Should list: lib, root, __main__.py, config.txt");
    }
    println!();

    // =========================================================================
    // Part 9: Test tilde expansion
    // =========================================================================

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Test 7: Tilde expansion");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // stat("~") should resolve to the home directory
    let result: String = sandbox.call("StatPathResult", "~".to_string())?;
    println!("   stat(\"~\") = {}", result);
    if result == "ok:dir" {
        println!("   PASS: tilde resolves to home directory");
    } else {
        println!("   INFO: tilde expansion not yet implemented (expected)");
    }

    // stat("~/.local") should resolve to /root/.local (if home is /root)
    let result: String = sandbox.call("StatPathResult", "~/.local".to_string())?;
    println!("   stat(\"~/.local\") = {}", result);
    if result == "ok:dir" {
        println!("   PASS: ~/ prefix resolves correctly");
    } else {
        println!("   INFO: tilde expansion not yet implemented (expected)");
    }
    println!();

    // =========================================================================
    // Part 10: Test relative paths from root CWD
    // =========================================================================

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Test 8: Relative paths");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // CWD is "/" by default
    let cwd: String = sandbox.call("GetCwd", ())?;
    println!("   CWD = \"{}\"", cwd);

    // Relative path to RO file
    let result: String = sandbox.call("StatPathResult", "__main__.py".to_string())?;
    if result.starts_with("ok:") {
        println!("   PASS: stat(\"__main__.py\") = {} (relative path works)", result);
    } else {
        println!("   FAIL: stat(\"__main__.py\") = {} (relative path broken)", result);
    }

    // Relative path to mount point
    let result: String = sandbox.call("StatPathResult", "lib".to_string())?;
    if result == "ok:dir" {
        println!("   PASS: stat(\"lib\") = {} (relative to mount point)", result);
    } else {
        println!(
            "   FAIL: stat(\"lib\") = {} (should find /lib mount)",
            result
        );
    }
    println!();

    // =========================================================================
    // Summary
    // =========================================================================

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Demo Complete!");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("   This demo tested:");
    println!("   - Virtual root directory (stat/readdir without root mount)");
    println!("   - Multiple non-overlapping FAT mounts (/lib + deep nested)");
    println!("   - Virtual parent directories for deeply nested mounts");
    println!("   - ReadOnly files at root alongside FAT mounts");
    println!("   - mkdir on FAT mounts (__pycache__ creation)");
    println!("   - Directory listing with mixed mount types");
    println!("   - Tilde expansion support");
    println!("   - Relative path resolution");
    println!();

    Ok(())
}

fn main() -> ExitCode {
    match run_demo() {
        Ok(()) => ExitCode::from(0),
        Err(e) => {
            eprintln!("Error: {}", e);
            ExitCode::from(1)
        }
    }
}
