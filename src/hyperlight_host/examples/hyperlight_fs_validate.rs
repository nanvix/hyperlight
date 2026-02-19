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

//! HyperlightFS configuration validator.
//!
//! This tool validates a HyperlightFS TOML configuration file and displays
//! what files would be mapped into the guest filesystem, including FAT mounts.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example hyperlight_fs_validate -- /path/to/hyperlight-fs.toml
//! ```
//!
//! # Example Output
//!
//! ```text
//! ✓ Config valid: hyperlight-fs.toml
//!
//! ReadOnly Files:
//!   /config.json (1.2 KB) <- /etc/app/config.json
//!   /assets/logo.png (45.0 KB) <- /opt/app/assets/logo.png
//!
//! FAT Mounts:
//!   /data (10 MB, read-write) <- /tmp/data.fat
//!   /logs (50 MB, read-write, temporary)
//!
//! Summary:
//!   ReadOnly files: 2
//!   ReadOnly directories: 1
//!   FAT mounts: 2
//!   Total ReadOnly size: 46.2 KB
//!   Total FAT size: 60 MB
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use hyperlight_host::hyperlight_fs::{HyperlightFSBuilder, HyperlightFsConfig};

/// HyperlightFS configuration validator.
///
/// Validates a HyperlightFS TOML configuration file and displays
/// what files would be mapped into the guest filesystem.
#[derive(Parser, Debug)]
#[command(name = "hyperlight_fs_validate")]
#[command(about = "Validate a HyperlightFS TOML configuration file")]
#[command(version)]
struct Args {
    /// Path to the HyperlightFS TOML configuration file.
    #[arg(value_name = "CONFIG")]
    config: PathBuf,

    /// Show only errors, no file listing.
    #[arg(short, long)]
    quiet: bool,

    /// Show detailed information including directory entries.
    #[arg(short, long)]
    verbose: bool,
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
    let config_path = args.config.display().to_string();

    // Step 1: Load and parse the TOML config
    if !args.quiet {
        println!("Loading config: {}", config_path);
        println!();
    }

    let config = HyperlightFsConfig::from_toml_file(&config_path)
        .map_err(|e| format!("Failed to load config: {}", e))?;

    // Check if config has any content
    let has_readonly = !config.file.is_empty() || !config.directory.is_empty();
    let has_fat = !config.fat_image.is_empty() || !config.fat_mount.is_empty();

    if !has_readonly && !has_fat {
        return Err(
            "Config is empty - no file mappings, directory mappings, or FAT mounts defined"
                .to_string(),
        );
    }

    if !args.quiet {
        let mut parts = Vec::new();
        if !config.file.is_empty() {
            parts.push(format!("{} file mapping(s)", config.file.len()));
        }
        if !config.directory.is_empty() {
            parts.push(format!("{} directory mapping(s)", config.directory.len()));
        }
        if !config.fat_image.is_empty() {
            parts.push(format!("{} FAT image(s)", config.fat_image.len()));
        }
        if !config.fat_mount.is_empty() {
            parts.push(format!("{} FAT mount(s)", config.fat_mount.len()));
        }
        println!("✓ Config parsed: {}", parts.join(", "));
        println!();
    }

    // Step 2: Build the manifest (dry-run) - only if we have readonly content
    let manifest = if has_readonly {
        let image = HyperlightFSBuilder::from_config(&config)
            .map_err(|e| format!("Failed to process config: {}", e))?;
        Some(image.file_summary())
    } else {
        None
    };

    // Step 3: Display ReadOnly results
    if !args.quiet && has_readonly {
        let manifest = manifest.as_ref().unwrap();
        let files: Vec<_> = manifest.files.iter().filter(|f| !f.is_dir).collect();
        let dirs: Vec<_> = manifest.files.iter().filter(|f| f.is_dir).collect();

        println!("═══════════════════════════════════════════════════════════════");
        println!("  📖 ReadOnly Mappings");
        println!("═══════════════════════════════════════════════════════════════");
        println!();

        if files.is_empty() {
            println!("  ⚠ Warning: No files matched the configuration patterns");
            println!();
        } else {
            println!("  Files:");
            for entry in &files {
                println!(
                    "    {} ({}) <- {}",
                    entry.guest_path,
                    format_size(entry.size),
                    entry.host_path.display()
                );
            }
            println!();
        }

        if args.verbose && !dirs.is_empty() {
            println!("  Directories:");
            for entry in &dirs {
                println!("    {}", entry.guest_path);
            }
            println!();
        }
    }

    // Step 4: Display FAT mount information
    let mut fat_total_size: u64 = 0;

    if !args.quiet && has_fat {
        println!("═══════════════════════════════════════════════════════════════");
        println!("  📝 FAT Mounts (Read-Write)");
        println!("═══════════════════════════════════════════════════════════════");
        println!();

        // FAT images (existing files)
        for fat_img in &config.fat_image {
            // Try to get the file size from the host filesystem
            let size_str = std::fs::metadata(&fat_img.host_path)
                .map(|m| {
                    let size = m.len();
                    fat_total_size += size;
                    format_size(size)
                })
                .unwrap_or_else(|_| "file not found".to_string());

            println!(
                "    {} ({}) <- {} [existing image]",
                fat_img.mount_point, size_str, fat_img.host_path
            );
        }

        // FAT mounts (to be created)
        for fat_mount in &config.fat_mount {
            let size = fat_mount
                .size
                .to_bytes()
                .map_err(|e| format!("Invalid FAT mount size: {}", e))?;

            fat_total_size += size as u64;

            let location = fat_mount
                .host_path
                .as_ref()
                .map(|p| format!("<- {} [persistent]", p))
                .unwrap_or_else(|| "[temporary]".to_string());

            println!(
                "    {} ({}) {}",
                fat_mount.mount_point,
                format_size(size as u64),
                location
            );
        }
        println!();
    }

    // Step 5: Summary
    if !args.quiet {
        println!("═══════════════════════════════════════════════════════════════");
        println!("  Summary");
        println!("═══════════════════════════════════════════════════════════════");
        println!();

        if has_readonly && let Some(m) = &manifest {
            let file_count = m.files.iter().filter(|f| !f.is_dir).count();
            let dir_count = m.files.iter().filter(|f| f.is_dir).count();
            println!("    ReadOnly files: {}", file_count);
            println!("    ReadOnly directories: {}", dir_count);
            println!("    ReadOnly total size: {}", format_size(m.total_size));
        }
        if has_fat {
            let fat_count = config.fat_image.len() + config.fat_mount.len();
            println!("    FAT mounts: {}", fat_count);
            println!("    FAT total size: {}", format_size(fat_total_size));
        }
        println!();
    }

    println!("✓ Configuration is valid!");

    Ok(())
}

fn main() -> ExitCode {
    let args = Args::parse();

    match run(args) {
        Ok(()) => ExitCode::from(0),
        Err(e) => {
            eprintln!("✗ Error: {}", e);
            ExitCode::from(1)
        }
    }
}
