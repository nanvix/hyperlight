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

//! HyperlightFS stress test.
//!
//! Creates a large file with patterned data, maps it into a sandbox, and verifies
//! that random reads from the guest match the host data.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example hyperlight_fs_stress
//! cargo run --example hyperlight_fs_stress -- --size 512  # 512 MB file
//! ```

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use clap::Parser;
use hyperlight_host::hyperlight_fs::HyperlightFSBuilder;
use hyperlight_host::sandbox::UninitializedSandbox;
use hyperlight_host::{GuestBinary, MultiUseSandbox};
use rand::RngCore;

/// HyperlightFS stress test.
///
/// Creates a large file with patterned data, maps it into a sandbox, and verifies
/// that random reads from the guest match the host data. The guest uses hardware
/// RDRAND for true random offset selection.
#[derive(Parser, Debug)]
#[command(name = "hyperlight_fs_stress")]
#[command(about = "Stress test HyperlightFS with large file random reads")]
#[command(version)]
struct Args {
    /// Size of the test file in megabytes.
    #[arg(short, long, default_value = "1024")]
    size: u64,

    /// Guest type to use: "rust" or "c".
    #[arg(short, long, default_value = "rust")]
    guest: String,

    /// Keep the test file after completion (for debugging).
    #[arg(short, long)]
    keep: bool,

    /// Show verbose output including each chunk verification.
    #[arg(short, long)]
    verbose: bool,
}

const NUM_SAMPLES: usize = 10;
const CHUNK_SIZE: usize = 256;
const SAMPLE_SIZE: usize = 8 + CHUNK_SIZE; // offset (u64) + data

/// Format a duration in human-readable form with appropriate units.
fn format_duration(d: std::time::Duration) -> String {
    let micros = d.as_micros();
    if micros >= 1_000_000 {
        format!("{:.2}s", d.as_secs_f64())
    } else if micros >= 1_000 {
        format!("{:.2}ms", micros as f64 / 1_000.0)
    } else {
        format!("{}µs", micros)
    }
}

/// Format a byte size in human-readable form.
fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

fn run(args: Args) -> Result<(), String> {
    let file_size = args.size * 1024 * 1024; // Convert MB to bytes

    // Validate guest type
    if args.guest != "rust" && args.guest != "c" {
        return Err(format!(
            "Invalid guest type '{}', use 'rust' or 'c'",
            args.guest
        ));
    }

    println!("HyperlightFS Stress Test");
    println!("========================");
    println!("Guest type: {}", args.guest);
    println!(
        "File size: {} ({} bytes)",
        format_size(file_size),
        file_size
    );
    println!("Random source: Guest RDTSC-seeded LCG");
    println!("File content: Random data");
    println!();

    // Create temporary directory for test file
    let temp_dir = tempfile::tempdir().map_err(|e| format!("Failed to create temp dir: {}", e))?;
    let test_file_path: PathBuf = temp_dir.path().join("stress_test.bin");

    // Step 1: Create test file with random data
    println!("Creating test file with random data...");
    let start = Instant::now();

    {
        let mut file =
            File::create(&test_file_path).map_err(|e| format!("Failed to create file: {}", e))?;

        // Write in 1 MB chunks for efficiency
        const WRITE_CHUNK: usize = 1024 * 1024;
        let mut buf = vec![0u8; WRITE_CHUNK];
        let mut offset: u64 = 0;
        let mut rng = rand::rng();

        while offset < file_size {
            let to_write = ((file_size - offset) as usize).min(WRITE_CHUNK);

            // Fill buffer with random data
            rng.fill_bytes(&mut buf[..to_write]);

            file.write_all(&buf[..to_write])
                .map_err(|e| format!("Failed to write: {}", e))?;
            offset += to_write as u64;
        }

        file.flush()
            .map_err(|e| format!("Failed to flush: {}", e))?;
    }

    println!(
        "  Created {} in {}",
        format_size(file_size),
        format_duration(start.elapsed())
    );
    println!();

    // Step 2: Build HyperlightFS image
    println!("Building HyperlightFS image...");
    let start = Instant::now();

    let fs_image = HyperlightFSBuilder::new()
        .add_file(
            test_file_path.to_str().ok_or("Invalid path")?,
            "/stress_test.bin",
        )
        .map_err(|e| format!("Failed to add file: {}", e))?
        .build()
        .map_err(|e| format!("Failed to build FS: {}", e))?;

    println!("  Built FS image in {}", format_duration(start.elapsed()));
    println!();

    // Step 3: Create sandbox with HyperlightFS
    println!("Creating sandbox...");
    let start = Instant::now();

    let guest_path = if args.guest == "c" {
        hyperlight_testing::c_simple_guest_as_string()
    } else {
        hyperlight_testing::simple_guest_as_string()
    }
    .map_err(|e| format!("Guest not found: {}", e))?;

    let mut sandbox: MultiUseSandbox =
        UninitializedSandbox::new(GuestBinary::FilePath(guest_path), None)
            .map_err(|e| format!("Failed to create sandbox: {}", e))?
            .with_hyperlight_fs(fs_image)
            .evolve()
            .map_err(|e| format!("Failed to evolve sandbox: {}", e))?;

    println!("  Sandbox created in {}", format_duration(start.elapsed()));
    println!();

    // Step 4: Call guest function to read random chunks
    println!(
        "Calling guest to read {} random chunks (using RDTSC-seeded LCG)...",
        NUM_SAMPLES
    );
    let start = Instant::now();

    let result: Vec<u8> = sandbox
        .call("RandomReadChunks", "/stress_test.bin".to_string())
        .map_err(|e| format!("Guest call failed: {}", e))?;

    println!(
        "  Guest returned {} bytes in {}",
        result.len(),
        format_duration(start.elapsed())
    );
    println!();

    // Step 5: Verify results
    println!("Verifying results...");

    if result.is_empty() {
        return Err("Guest returned empty result - file may not have been found".into());
    }

    let expected_len = NUM_SAMPLES * SAMPLE_SIZE;
    if result.len() != expected_len {
        return Err(format!(
            "Wrong result size: expected {} bytes, got {}",
            expected_len,
            result.len()
        ));
    }

    // Open the file for verification
    let mut host_file =
        File::open(&test_file_path).map_err(|e| format!("Failed to open file: {}", e))?;

    let mut host_buf = [0u8; CHUNK_SIZE];
    let mut all_ok = true;

    for i in 0..NUM_SAMPLES {
        let sample_start = i * SAMPLE_SIZE;

        // Extract offset from result (guest chose this randomly via RDRAND)
        let offset_bytes: [u8; 8] = result[sample_start..sample_start + 8]
            .try_into()
            .map_err(|_| "Failed to extract offset")?;
        let offset = u64::from_le_bytes(offset_bytes);

        // Extract data from result
        let data_start = sample_start + 8;
        let guest_data = &result[data_start..data_start + CHUNK_SIZE];

        // Read from host file at same offset
        host_file
            .seek(SeekFrom::Start(offset))
            .map_err(|e| format!("Failed to seek: {}", e))?;
        host_file
            .read_exact(&mut host_buf)
            .map_err(|e| format!("Failed to read: {}", e))?;

        // Compare
        let matches = guest_data == host_buf;
        if args.verbose || !matches {
            println!(
                "  Chunk {}: offset=0x{:012X} ({}) {}",
                i,
                offset,
                format_size(offset),
                if matches { "✓" } else { "✗ MISMATCH!" }
            );
        }

        if !matches {
            all_ok = false;
            if args.verbose {
                println!("    Expected: {:02X?}...", &host_buf[..16]);
                println!("    Got:      {:02X?}...", &guest_data[..16]);
            }
        }
    }

    println!();

    // Cleanup (unless --keep)
    if !args.keep {
        drop(temp_dir); // This deletes the temp directory
    } else {
        println!("Test file kept at: {}", test_file_path.display());
        // Prevent temp_dir from being dropped
        let _ = temp_dir.keep();
    }

    if all_ok {
        println!("✓ All {} chunks verified successfully!", NUM_SAMPLES);
        Ok(())
    } else {
        Err("Some chunks failed verification".into())
    }
}

fn main() -> ExitCode {
    let args = Args::parse();

    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {}", e);
            ExitCode::FAILURE
        }
    }
}
