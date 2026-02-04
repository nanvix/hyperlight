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

//! HyperlightFS demonstration - ReadOnly and FAT filesystem features.
//!
//! This example demonstrates the full capabilities of HyperlightFS:
//!
//! 1. **ReadOnly mounts**: Static files mapped from host into guest
//! 2. **FAT mounts**: Read-write filesystem for dynamic data
//! 3. **Guest-created mounts**: Dynamic FAT creation at runtime
//!
//! # Usage
//!
//! ```bash
//! cargo run --example hyperlight_fs_demo
//! ```
//!
//! # What This Demo Shows
//!
//! - Creating a HyperlightFS image with both RO and FAT mounts
//! - Guest reading static config files (ReadOnly)
//! - Guest creating, writing, reading files (FAT)
//! - Guest creating directories (FAT)
//! - Listing directory contents (FAT)
//! - Host-Guest interoperability via MAP_SHARED (zero-copy verification)
//! - Guest dynamically creating FAT mounts at runtime
//! - Guest unmounting filesystems

use std::path::Path;
use std::process::ExitCode;

use hyperlight_host::GuestBinary;
use hyperlight_host::hyperlight_fs::HyperlightFSBuilder;
use hyperlight_host::sandbox::{MultiUseSandbox, SandboxConfiguration, UninitializedSandbox};

/// Path to the test guest binary.
fn get_guest_path() -> &'static str {
    // Use the simpleguest binary which has all the FAT guest functions
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
    println!("║           HyperlightFS Demo - ReadOnly + FAT                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Create a temporary directory for our demo files
    let temp_dir = tempfile::TempDir::new()?;
    let fat_path = temp_dir.path().join("demo.fat");

    // Create a sample config file to demonstrate ReadOnly mount
    let config_path = temp_dir.path().join("config.txt");
    std::fs::write(&config_path, b"app_name=HyperlightDemo\nversion=1.0\n")?;

    println!("📁 Setting up HyperlightFS...");
    println!();

    // =========================================================================
    // Part 1: Build HyperlightFS with ReadOnly + FAT
    // =========================================================================

    let fs_image = HyperlightFSBuilder::new()
        // Add a ReadOnly file mapping
        .add_file(&config_path, "/config.txt")?
        // Add an empty FAT mount for read-write operations
        .add_empty_fat_mount_at(&fat_path, "/data", 1024 * 1024)? // 1MB FAT
        .build()?;

    println!(
        "   ✓ ReadOnly mount: /config.txt <- {}",
        config_path.display()
    );
    println!("   ✓ FAT mount: /data (1 MB read-write)");
    println!();

    // =========================================================================
    // Part 2: Create sandbox with HyperlightFS
    // =========================================================================

    println!("🚀 Creating sandbox with guest binary...");
    let guest_path = get_guest_path();

    if !Path::new(guest_path).exists() {
        return Err(format!(
            "Guest binary not found at: {}\n\
             Run 'just guests' first to build the test guests.",
            guest_path
        )
        .into());
    }

    // Configure with larger heap for guest-created FAT demo (Part 6)
    let mut config = SandboxConfiguration::default();
    config.set_heap_size(1024 * 1024); // 1MB heap

    let mut sandbox: MultiUseSandbox =
        UninitializedSandbox::new(GuestBinary::FilePath(guest_path.into()), Some(config))?
            .with_hyperlight_fs(fs_image)
            .evolve()?;
    println!("   ✓ Sandbox created and initialized");
    println!();

    // =========================================================================
    // Part 3: Demonstrate ReadOnly access
    // =========================================================================

    println!("═══════════════════════════════════════════════════════════════");
    println!("  📖 ReadOnly Filesystem Demo");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Read the config file from guest using the generic ReadFile function
    // which works with the ReadOnly VFS mount
    println!("   Reading /config.txt from guest...");
    let config_content: Vec<u8> = sandbox.call("ReadFile", "/config.txt".to_string())?;
    let config_str = String::from_utf8_lossy(&config_content);
    println!("   ✓ Content:");
    for line in config_str.lines() {
        println!("     │ {}", line);
    }
    println!();

    // =========================================================================
    // Part 3.5: Relative Path Resolution with Mixed Mounts
    // =========================================================================

    println!("═══════════════════════════════════════════════════════════════");
    println!("  🔍 Relative Path Resolution Demo (FAT at /data, RO at /)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("   This section verifies that relative paths work correctly when:");
    println!("   - FAT is mounted at /data (not at root)");
    println!("   - ReadOnly is mounted at / (fallback mount)");
    println!("   - CWD is / (the default)");
    println!();

    // Test 1: Relative path to existing ReadOnly file should work
    println!("   1️⃣  Testing relative path to ReadOnly file...");
    println!("      open(\"config.txt\") should find /config.txt");
    let result: String = sandbox.call("OpenPathResult", "config.txt".to_string())?;
    if result.starts_with("ok:") {
        println!("      ✓ Result: {} (file found via relative path)", result);
    } else {
        println!("      ✗ FAILED: Expected ok:<size>, got: {}", result);
        return Err(format!("Relative path to RO file failed: {}", result).into());
    }
    println!();

    // Test 2: Relative path to non-existent file should return NotFound (NOT InvalidPath)
    println!("   2️⃣  Testing relative path to non-existent file...");
    println!("      stat(\"pyvenv.cfg\") should return NotFound, not InvalidPath");
    let result: String = sandbox.call("StatPathResult", "pyvenv.cfg".to_string())?;
    if result == "not_found" {
        println!("      ✓ Result: {} (correct error for missing file)", result);
    } else if result == "invalid_path" {
        println!("      ✗ FAILED: Got InvalidPath instead of NotFound!");
        println!("        This is the bug we fixed - relative paths were not being normalized.");
        return Err("Relative path incorrectly returned InvalidPath".into());
    } else {
        println!("      ? Unexpected result: {}", result);
    }
    println!();

    // Test 3: Absolute path to FAT mount should work
    println!("   3️⃣  Testing absolute path to FAT mount...");
    println!("      stat(\"/data\") should find the FAT mount directory");
    let result: String = sandbox.call("StatPathResult", "/data".to_string())?;
    if result == "ok:dir" {
        println!("      ✓ Result: {} (FAT mount point found)", result);
    } else {
        println!("      ✗ FAILED: Expected ok:dir, got: {}", result);
        return Err(format!("FAT mount point not found: {}", result).into());
    }
    println!();

    // Test 4: Relative path after chdir should resolve to new cwd
    println!("   4️⃣  Testing relative path after chdir to /data...");
    let chdir_ok: bool = sandbox.call("Chdir", "/data".to_string())?;
    if !chdir_ok {
        return Err("chdir to /data failed".into());
    }
    println!("      chdir(\"/data\") succeeded");
    
    // Now relative path should resolve within /data
    println!("      stat(\"testfile.txt\") should return NotFound (not InvalidPath)");
    let result: String = sandbox.call("StatPathResult", "testfile.txt".to_string())?;
    if result == "not_found" {
        println!("      ✓ Result: {} (correct - file doesn't exist yet)", result);
    } else if result == "invalid_path" {
        println!("      ✗ FAILED: Got InvalidPath instead of NotFound!");
        return Err("Relative path in FAT mount returned InvalidPath".into());
    } else {
        println!("      ? Unexpected result: {}", result);
    }
    println!();

    // Test 5: Create file via relative path and verify
    println!("   5️⃣  Creating file via relative path in /data...");
    let write_ok: bool = sandbox.call(
        "WriteFatFile",
        ("testfile.txt".to_string(), b"Created via relative path!".to_vec()),
    )?;
    if write_ok {
        println!("      ✓ WriteFatFile(\"testfile.txt\") succeeded");
    } else {
        println!("      ✗ WriteFatFile failed");
    }

    // Verify the file exists
    let result: String = sandbox.call("StatPathResult", "testfile.txt".to_string())?;
    if result.starts_with("ok:") {
        println!("      ✓ stat(\"testfile.txt\") = {} (file created)", result);
    } else {
        println!("      ✗ File not found after creation: {}", result);
    }
    println!();

    // Restore CWD to root
    let _: bool = sandbox.call("Chdir", "/".to_string())?;
    println!("      chdir(\"/\") - restored CWD to root");
    println!();

    println!("   ═══════════════════════════════════════════════════════════");
    println!("   🎉 RELATIVE PATH RESOLUTION VERIFIED!");
    println!("   ═══════════════════════════════════════════════════════════");
    println!();

    // =========================================================================
    // Part 4: Demonstrate FAT read-write operations
    // =========================================================================

    println!("═══════════════════════════════════════════════════════════════");
    println!("  📝 FAT Read-Write Filesystem Demo");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Create a directory
    println!("   Creating directory /data/logs...");
    let mkdir_result: bool = sandbox.call("MkdirFat", "/data/logs".to_string())?;
    if mkdir_result {
        println!("   ✓ Directory created");
    } else {
        println!("   ✗ Failed to create directory");
    }
    println!();

    // Write a file
    println!("   Writing /data/logs/app.log...");
    let log_content = b"2026-01-21 10:30:00 INFO Application started\n\
                        2026-01-21 10:30:01 INFO Processing request #1\n\
                        2026-01-21 10:30:02 INFO Request #1 completed\n"
        .to_vec();
    let write_result: bool = sandbox.call(
        "WriteFatFile",
        ("/data/logs/app.log".to_string(), log_content.clone()),
    )?;
    if write_result {
        println!("   ✓ File written ({} bytes)", log_content.len());
    } else {
        println!("   ✗ Failed to write file");
    }
    println!();

    // Read the file back
    println!("   Reading /data/logs/app.log...");
    let read_content: Vec<u8> = sandbox.call("ReadFatFile", "/data/logs/app.log".to_string())?;
    println!("   ✓ Content:");
    for line in String::from_utf8_lossy(&read_content).lines() {
        println!("     │ {}", line);
    }
    println!();

    // Write another file
    println!("   Writing /data/status.json...");
    let status_content = br#"{"status": "running", "uptime_seconds": 42}"#.to_vec();
    let _: bool = sandbox.call(
        "WriteFatFile",
        ("/data/status.json".to_string(), status_content),
    )?;
    println!("   ✓ File written");
    println!();

    // List directory contents
    println!("   Listing /data...");
    let listing: String = sandbox.call("ListDirFat", "/data".to_string())?;
    println!("   ✓ Contents:");
    for entry in listing.split(',').filter(|s| !s.is_empty()) {
        let icon = if entry.ends_with('/') { "📁" } else { "📄" };
        println!("     {} {}", icon, entry);
    }
    println!();

    // List the logs subdirectory
    println!("   Listing /data/logs...");
    let logs_listing: String = sandbox.call("ListDirFat", "/data/logs".to_string())?;
    println!("   ✓ Contents:");
    for entry in logs_listing.split(',').filter(|s| !s.is_empty()) {
        let icon = if entry.ends_with('/') { "📁" } else { "📄" };
        println!("     {} {}", icon, entry);
    }
    println!();

    // Get file stats
    println!("   Getting stats for /data/logs/app.log...");
    let size: i64 = sandbox.call("StatFatSize", "/data/logs/app.log".to_string())?;
    println!("   ✓ File size: {} bytes", size);
    println!();

    // Delete a file
    println!("   Deleting /data/status.json...");
    let delete_result: bool = sandbox.call("DeleteFatFile", "/data/status.json".to_string())?;
    if delete_result {
        println!("   ✓ File deleted");
    } else {
        println!("   ✗ Failed to delete file");
    }
    println!();

    // Verify deletion
    println!("   Verifying deletion...");
    let exists: i32 = sandbox.call("ExistsFat", "/data/status.json".to_string())?;
    if exists == 0 {
        println!("   ✓ File no longer exists");
    } else {
        println!("   ✗ File still exists!");
    }
    println!();

    // =========================================================================
    // Part 5: Host-Guest Interoperability (MAP_SHARED validation)
    // =========================================================================

    println!("═══════════════════════════════════════════════════════════════");
    println!("  🔄 Host-Guest Interoperability Demo (MAP_SHARED)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("   This section demonstrates that HOST and GUEST share the same");
    println!("   memory via MAP_SHARED - writes from one are immediately visible");
    println!("   to the other without any copying.");
    println!();

    // Host writes a file using Sandbox API
    println!("   1️⃣  Host writes /data/from_host.txt using sandbox.fs_write_file()...");
    let host_message = "Hello from the HOST side!";
    sandbox.fs_write_file("/data/from_host.txt", host_message.as_bytes())?;
    println!("      ✓ Host wrote {} bytes", host_message.len());
    println!();

    // Guest reads that same file via its C API
    println!("   2️⃣  Guest reads /data/from_host.txt using ReadFatFile()...");
    let guest_read: Vec<u8> = sandbox.call("ReadFatFile", "/data/from_host.txt".to_string())?;
    let guest_read_str = String::from_utf8_lossy(&guest_read);
    println!("      ✓ Guest read: \"{}\"", guest_read_str);
    if guest_read != host_message.as_bytes() {
        return Err("guest did not read what host wrote!".into());
    }
    println!("      ✓ Content matches!");
    println!();

    // Guest writes a different file via its C API
    println!("   3️⃣  Guest writes /data/from_guest.txt using WriteFatFile()...");
    let guest_message = b"Greetings from the GUEST side!".to_vec();
    let _: bool = sandbox.call(
        "WriteFatFile",
        ("/data/from_guest.txt".to_string(), guest_message.clone()),
    )?;
    println!("      ✓ Guest wrote {} bytes", guest_message.len());
    println!();

    // Host reads that file using Sandbox API
    println!("   4️⃣  Host reads /data/from_guest.txt using sandbox.fs_read_file()...");
    let host_read = sandbox.fs_read_file("/data/from_guest.txt")?;
    let host_read_str = String::from_utf8_lossy(&host_read);
    println!("      ✓ Host read: \"{}\"", host_read_str);
    if host_read != guest_message {
        return Err("host did not read what guest wrote!".into());
    }
    println!("      ✓ Content matches!");
    println!();

    println!("   ═══════════════════════════════════════════════════════════");
    println!("   🎉 MAP_SHARED VERIFIED: Same memory, zero copies!");
    println!("   ═══════════════════════════════════════════════════════════");
    println!();

    // =========================================================================
    // Part 6: Guest-Created Dynamic FAT Mounts
    // =========================================================================

    println!("═══════════════════════════════════════════════════════════════");
    println!("  🔧 Guest-Created Dynamic FAT Mount Demo");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("   Guests can dynamically create FAT filesystems at runtime,");
    println!("   allocating memory from their heap. These mounts are private");
    println!("   to the guest and can be unmounted when no longer needed.");
    println!();

    // Check that existing host-provided mount is not guest-created
    println!("   1️⃣  Checking /data mount origin...");
    let is_guest_created: bool = sandbox.call("IsGuestCreatedMount", "/data".to_string())?;
    if is_guest_created {
        println!("      ✗ /data incorrectly reported as guest-created!");
    } else {
        println!("      ✓ /data correctly identified as host-provided");
    }
    println!();

    // Guest creates a new dynamic FAT mount
    println!("   2️⃣  Guest creates /scratch mount (128 KB dynamic FAT)...");
    let create_result: bool =
        sandbox.call("CreateFatMount", ("/scratch".to_string(), 128 * 1024i64))?;
    if create_result {
        println!("      ✓ /scratch mount created by guest");
    } else {
        println!("      ✗ Failed to create /scratch mount");
        return Err("guest failed to create dynamic FAT mount".into());
    }
    println!();

    // Verify it's marked as guest-created
    println!("   3️⃣  Verifying /scratch is guest-created...");
    let is_scratch_guest: bool = sandbox.call("IsGuestCreatedMount", "/scratch".to_string())?;
    if is_scratch_guest {
        println!("      ✓ /scratch correctly identified as guest-created");
    } else {
        println!("      ✗ /scratch incorrectly reported as host-provided!");
    }
    println!();

    // Write a file to the guest-created mount
    println!("   4️⃣  Guest writes /scratch/temp.txt...");
    let temp_content = b"This file is on a guest-created filesystem!".to_vec();
    let write_scratch: bool = sandbox.call(
        "WriteFatFile",
        ("/scratch/temp.txt".to_string(), temp_content.clone()),
    )?;
    if write_scratch {
        println!("      ✓ File written to guest-created mount");
    } else {
        println!("      ✗ Failed to write to /scratch");
    }
    println!();

    // Read it back
    println!("   5️⃣  Guest reads /scratch/temp.txt...");
    let read_scratch: Vec<u8> = sandbox.call("ReadFatFile", "/scratch/temp.txt".to_string())?;
    if read_scratch == temp_content {
        println!(
            "      ✓ Content: \"{}\"",
            String::from_utf8_lossy(&read_scratch)
        );
    } else {
        println!("      ✗ Content mismatch!");
    }
    println!();

    // Unmount the guest-created filesystem
    println!("   6️⃣  Guest unmounts /scratch...");
    let unmount_result: bool = sandbox.call("UnmountFat", "/scratch".to_string())?;
    if unmount_result {
        println!("      ✓ /scratch unmounted successfully");
    } else {
        println!("      ✗ Failed to unmount /scratch");
    }
    println!();

    // Verify it's gone (trying to read should fail)
    println!("   7️⃣  Verifying /scratch is unmounted...");
    let exists_after: i32 = sandbox.call("ExistsFat", "/scratch/temp.txt".to_string())?;
    if exists_after == 0 {
        println!("      ✓ /scratch no longer accessible (correctly unmounted)");
    } else {
        println!("      ✗ /scratch still accessible after unmount!");
    }
    println!();

    // Verify we can't unmount host-provided mounts
    println!("   8️⃣  Attempting to unmount host-provided /data (should fail)...");
    let unmount_host: bool = sandbox.call("UnmountFat", "/data".to_string())?;
    if !unmount_host {
        println!("      ✓ Correctly refused to unmount host-provided mount");
    } else {
        println!("      ✗ Incorrectly allowed unmounting host-provided mount!");
    }
    println!();

    // =========================================================================
    // Summary
    // =========================================================================

    println!("═══════════════════════════════════════════════════════════════");
    println!("  ✨ Demo Complete!");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("   This demo showed:");
    println!("   • ReadOnly file mapping (host -> guest)");
    println!("   • FAT filesystem creation and mounting");
    println!("   • Creating directories (mkdir)");
    println!("   • Writing files");
    println!("   • Reading files");
    println!("   • Listing directory contents");
    println!("   • Getting file statistics");
    println!("   • Deleting files");
    println!("   • Host-Guest interoperability via MAP_SHARED");
    println!("   • Guest-created dynamic FAT mounts");
    println!("   • Unmounting guest-created filesystems");
    println!();
    println!("   FAT image location: {}", fat_path.display());
    println!();

    Ok(())
}

fn main() -> ExitCode {
    match run_demo() {
        Ok(()) => ExitCode::from(0),
        Err(e) => {
            eprintln!("✗ Error: {}", e);
            ExitCode::from(1)
        }
    }
}
