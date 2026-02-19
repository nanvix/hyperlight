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

//! TOML configuration support for HyperlightFS.
//!
//! This module provides serde-deserializable configuration structures for
//! defining filesystem mappings in TOML format.
//!
//! # Example TOML Configuration
//!
//! ```toml
//! # Single file mappings (read-only)
//! [[file]]
//! host_path = "/etc/app/config.json"
//! guest = "/config.json"
//!
//! [[file]]
//! host_path = "/var/data/model.bin"
//! guest = "/models/model.bin"
//!
//! # Directory mapping with glob patterns (read-only)
//! [[directory]]
//! host_path = "/opt/app/assets"
//! guest = "/assets"
//! include = ["**/*.json", "**/*.txt"]
//! exclude = ["**/secret/*", "**/.git/**"]
//!
//! # Mount an existing FAT image (read-write)
//! [[fat_image]]
//! host_path = "/host/path/to/existing.fat"
//! mount_point = "/data"
//!
//! # Create an empty temporary FAT mount (deleted on drop)
//! [[fat_mount]]
//! mount_point = "/tmp"
//! size = "10MB"
//!
//! # Create an empty FAT mount at specific host path (persists after drop)
//! [[fat_mount]]
//! host_path = "/host/path/to/persistent.fat"
//! mount_point = "/logs"
//! size = "50MB"
//! ```
//!
//! # Usage
//!
//! ## From a TOML file on disk
//!
//! ```no_run
//! # use hyperlight_host::hyperlight_fs::{HyperlightFSBuilder, HyperlightFsConfig};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Load config from a file
//! let config = HyperlightFsConfig::from_toml_file("/path/to/hyperlight-fs.toml")?;
//! let fs = HyperlightFSBuilder::from_config(&config)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## From a TOML string
//!
//! ```no_run
//! # use hyperlight_host::hyperlight_fs::{HyperlightFSBuilder, HyperlightFsConfig};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let toml_content = r#"
//! [[file]]
//! host_path = "/etc/app/config.json"
//! guest = "/config.json"
//! "#;
//!
//! let config = HyperlightFsConfig::from_toml(toml_content)?;
//! let fs = HyperlightFSBuilder::from_config(&config)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Size Values
//!
//! FAT mount sizes can be specified as:
//! - Integer bytes: `size = 1048576`
//! - Human-readable strings: `size = "10MB"` or `size = "1GiB"`
//!
//! Supported suffixes:
//! - `B` - bytes
//! - `KB` - kilobytes (1000 bytes)
//! - `KiB` - kibibytes (1024 bytes)
//! - `MB` - megabytes (1000² bytes)
//! - `MiB` - mebibytes (1024² bytes)
//! - `GB` - gigabytes (1000³ bytes)
//! - `GiB` - gibibytes (1024³ bytes)

use serde::Deserialize;

/// Root configuration structure for HyperlightFS TOML files.
///
/// Contains lists of individual file mappings, directory mappings,
/// and FAT filesystem mounts.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HyperlightFsConfig {
    /// Individual file mappings (read-only).
    #[serde(default)]
    pub file: Vec<FileMapping>,

    /// Directory mappings with optional glob patterns (read-only).
    #[serde(default)]
    pub directory: Vec<DirectoryMapping>,

    /// Existing FAT image mounts (read-write).
    #[serde(default)]
    pub fat_image: Vec<FatImageConfig>,

    /// Empty FAT mounts to create (read-write).
    #[serde(default)]
    pub fat_mount: Vec<FatMountConfig>,
}

/// A single file mapping from host to guest path.
///
/// # Example
///
/// ```toml
/// [[file]]
/// host_path = "/etc/app/config.json"
/// guest = "/config.json"
/// ```
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileMapping {
    /// Absolute path to the file on the host filesystem.
    pub host_path: String,

    /// Path where the file will appear in the guest's virtual filesystem.
    pub guest: String,
}

/// A directory mapping with optional include/exclude glob patterns.
///
/// # Example
///
/// ```toml
/// [[directory]]
/// host_path = "/opt/app/assets"
/// guest = "/assets"
/// include = ["**/*.json", "**/*.txt"]
/// exclude = ["**/secret/*"]
/// ```
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DirectoryMapping {
    /// Absolute path to the directory on the host filesystem.
    pub host_path: String,

    /// Path where the directory will appear in the guest's virtual filesystem.
    pub guest: String,

    /// Glob patterns for files to include (default: `["**/*"]` - all files).
    #[serde(default)]
    pub include: Vec<String>,

    /// Glob patterns for files to exclude (default: empty - exclude nothing).
    #[serde(default)]
    pub exclude: Vec<String>,
}

/// Configuration for mounting an existing FAT image file.
///
/// The FAT image must already exist and be formatted as FAT32. It will be
/// opened with an exclusive lock and mapped read-write into the guest.
///
/// # Example
///
/// ```toml
/// [[fat_image]]
/// host_path = "/host/path/to/existing.fat"
/// mount_point = "/data"
/// ```
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FatImageConfig {
    /// Absolute path to the FAT image file on the host filesystem.
    pub host_path: String,

    /// Mount point in the guest filesystem where the FAT image will be accessible.
    pub mount_point: String,
}

/// Configuration for creating an empty FAT mount.
///
/// If `host_path` is specified, the FAT image will be created at that path and
/// persist after the sandbox is dropped. If `host_path` is omitted, a temporary
/// file will be created and deleted when the sandbox is dropped.
///
/// # Examples
///
/// Temporary FAT mount (deleted on drop):
/// ```toml
/// [[fat_mount]]
/// mount_point = "/tmp"
/// size = "10MB"
/// ```
///
/// Persistent FAT mount (file persists after drop):
/// ```toml
/// [[fat_mount]]
/// host_path = "/host/path/to/persistent.fat"
/// mount_point = "/logs"
/// size = "50MB"
/// ```
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FatMountConfig {
    /// Optional path for the FAT image file on the host.
    ///
    /// If specified, the FAT image will be created at this path and persist
    /// after the sandbox is dropped. If omitted, a temporary file will be
    /// created and automatically deleted.
    #[serde(default)]
    pub host_path: Option<String>,

    /// Mount point in the guest filesystem where the FAT mount will be accessible.
    pub mount_point: String,

    /// Size of the FAT image.
    ///
    /// Can be specified as:
    /// - An integer number of bytes: `1048576`
    /// - A human-readable string: `"10MB"`, `"1GiB"`
    pub size: SizeValue,
}

/// Size value that can be parsed from a string ("10MB") or integer (bytes).
///
/// This enum uses serde's `untagged` representation to allow either format
/// in TOML configuration files.
///
/// # Examples
///
/// ```toml
/// # As integer bytes
/// size = 10485760
///
/// # As human-readable string
/// size = "10MB"
/// size = "1GiB"
/// ```
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum SizeValue {
    /// Size specified as raw bytes.
    Bytes(usize),
    /// Size specified as a human-readable string (e.g., "10MB", "1GiB").
    Human(String),
}

impl SizeValue {
    /// Convert the size value to bytes.
    ///
    /// # Supported Suffixes
    ///
    /// - `B` - bytes (multiplier: 1)
    /// - `KB` - kilobytes (multiplier: 1,000)
    /// - `KiB` - kibibytes (multiplier: 1,024)
    /// - `MB` - megabytes (multiplier: 1,000,000)
    /// - `MiB` - mebibytes (multiplier: 1,048,576)
    /// - `GB` - gigabytes (multiplier: 1,000,000,000)
    /// - `GiB` - gibibytes (multiplier: 1,073,741,824)
    ///
    /// Numbers without a suffix are treated as bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the string cannot be parsed or contains an
    /// unrecognized suffix.
    pub fn to_bytes(&self) -> Result<usize, String> {
        match self {
            SizeValue::Bytes(b) => Ok(*b),
            SizeValue::Human(s) => parse_size_string(s),
        }
    }
}

/// Parse a human-readable size string into bytes.
///
/// Accepts formats like "10MB", "1GiB", "500KB", or just "1024" (bytes).
fn parse_size_string(s: &str) -> Result<usize, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("empty size string".to_string());
    }

    // Find where the numeric part ends
    let num_end = s
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(s.len());

    let (num_str, suffix) = s.split_at(num_end);
    let suffix = suffix.trim();

    // Parse the numeric part
    let num: f64 = num_str
        .parse()
        .map_err(|_| format!("invalid number in size string: '{}'", num_str))?;

    // Determine the multiplier based on suffix
    let multiplier: u64 = match suffix.to_uppercase().as_str() {
        "" | "B" => 1,
        "KB" => 1_000,
        "KIB" => 1_024,
        "MB" => 1_000_000,
        "MIB" => 1_024 * 1_024,
        "GB" => 1_000_000_000,
        "GIB" => 1_024 * 1_024 * 1_024,
        _ => return Err(format!("unrecognized size suffix: '{}'", suffix)),
    };

    let bytes = (num * multiplier as f64) as usize;
    Ok(bytes)
}

impl HyperlightFsConfig {
    /// Parse a TOML configuration string into a `HyperlightFsConfig`.
    ///
    /// # Errors
    ///
    /// Returns an error if the TOML is malformed or contains unknown fields.
    ///
    /// # Example
    ///
    /// ```
    /// use hyperlight_host::hyperlight_fs::HyperlightFsConfig;
    ///
    /// let toml = r#"
    /// [[file]]
    /// host_path = "/etc/config.json"
    /// guest = "/config.json"
    /// "#;
    ///
    /// let config = HyperlightFsConfig::from_toml(toml).unwrap();
    /// assert_eq!(config.file.len(), 1);
    /// ```
    pub fn from_toml(toml_content: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(toml_content)
    }

    /// Load a TOML configuration from a file path.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be read
    /// - The TOML is malformed or contains unknown fields
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use hyperlight_host::hyperlight_fs::HyperlightFsConfig;
    /// # fn main() -> Result<(), hyperlight_host::hyperlight_fs::ConfigError> {
    /// let config = HyperlightFsConfig::from_toml_file("/path/to/hyperlight-fs.toml")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_toml_file(path: &str) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path).map_err(|e| ConfigError::IoError {
            path: path.to_string(),
            source: e,
        })?;
        Self::from_toml(&content).map_err(|e| ConfigError::ParseError {
            path: path.to_string(),
            source: e,
        })
    }

    /// Returns the total number of mappings (files + directories + FAT mounts).
    pub fn mapping_count(&self) -> usize {
        self.file.len() + self.directory.len() + self.fat_image.len() + self.fat_mount.len()
    }

    /// Returns true if the configuration has no mappings.
    pub fn is_empty(&self) -> bool {
        self.file.is_empty()
            && self.directory.is_empty()
            && self.fat_image.is_empty()
            && self.fat_mount.is_empty()
    }
}

/// Errors that can occur when loading a TOML configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// Failed to read the configuration file.
    #[error("failed to read config file '{path}': {source}")]
    IoError {
        /// The path that could not be read.
        path: String,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// Failed to parse the TOML content.
    #[error("failed to parse config file '{path}': {source}")]
    ParseError {
        /// The path that could not be parsed.
        path: String,
        /// The underlying TOML parse error.
        #[source]
        source: toml::de::Error,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_empty_config() {
        let config = HyperlightFsConfig::from_toml("").unwrap();
        assert!(config.is_empty());
        assert_eq!(config.mapping_count(), 0);
    }

    #[test]
    fn test_parse_single_file() {
        let toml = r#"
[[file]]
host_path = "/etc/config.json"
guest = "/config.json"
"#;
        let config = HyperlightFsConfig::from_toml(toml).unwrap();
        assert_eq!(config.file.len(), 1);
        assert_eq!(config.file[0].host_path, "/etc/config.json");
        assert_eq!(config.file[0].guest, "/config.json");
        assert_eq!(config.mapping_count(), 1);
    }

    #[test]
    fn test_parse_multiple_files() {
        let toml = r#"
[[file]]
host_path = "/etc/config.json"
guest = "/config.json"

[[file]]
host_path = "/var/data/model.bin"
guest = "/models/model.bin"
"#;
        let config = HyperlightFsConfig::from_toml(toml).unwrap();
        assert_eq!(config.file.len(), 2);
        assert_eq!(config.file[1].guest, "/models/model.bin");
    }

    #[test]
    fn test_parse_directory_minimal() {
        let toml = r#"
[[directory]]
host_path = "/opt/app/assets"
guest = "/assets"
"#;
        let config = HyperlightFsConfig::from_toml(toml).unwrap();
        assert_eq!(config.directory.len(), 1);
        assert_eq!(config.directory[0].host_path, "/opt/app/assets");
        assert_eq!(config.directory[0].guest, "/assets");
        assert!(config.directory[0].include.is_empty());
        assert!(config.directory[0].exclude.is_empty());
    }

    #[test]
    fn test_parse_directory_with_patterns() {
        let toml = r#"
[[directory]]
host_path = "/opt/app/assets"
guest = "/assets"
include = ["**/*.json", "**/*.txt"]
exclude = ["**/secret/*", "**/.git/**"]
"#;
        let config = HyperlightFsConfig::from_toml(toml).unwrap();
        assert_eq!(config.directory[0].include.len(), 2);
        assert_eq!(config.directory[0].exclude.len(), 2);
        assert_eq!(config.directory[0].include[0], "**/*.json");
        assert_eq!(config.directory[0].exclude[0], "**/secret/*");
    }

    #[test]
    fn test_parse_mixed_config() {
        let toml = r#"
[[file]]
host_path = "/etc/config.json"
guest = "/config.json"

[[directory]]
host_path = "/opt/assets"
guest = "/assets"
include = ["**/*.png"]

[[file]]
host_path = "/var/data.bin"
guest = "/data.bin"
"#;
        let config = HyperlightFsConfig::from_toml(toml).unwrap();
        assert_eq!(config.file.len(), 2);
        assert_eq!(config.directory.len(), 1);
        assert_eq!(config.mapping_count(), 3);
    }

    #[test]
    fn test_reject_unknown_fields() {
        let toml = r#"
[[file]]
host_path = "/etc/config.json"
guest = "/config.json"
unknown_field = "bad"
"#;
        let result = HyperlightFsConfig::from_toml(toml);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_unknown_top_level() {
        let toml = r#"
[[unknown_section]]
foo = "bar"
"#;
        let result = HyperlightFsConfig::from_toml(toml);
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_required_fields() {
        // Missing 'guest' field
        let toml = r#"
[[file]]
host_path = "/etc/config.json"
"#;
        let result = HyperlightFsConfig::from_toml(toml);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_toml_file_not_found() {
        let result = HyperlightFsConfig::from_toml_file("/nonexistent/path/config.toml");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ConfigError::IoError { .. }));
    }

    // ==================== FAT Config Tests ====================

    #[test]
    fn test_parse_fat_image() {
        let toml = r#"
[[fat_image]]
host_path = "/host/path/to/existing.fat"
mount_point = "/data"
"#;
        let config = HyperlightFsConfig::from_toml(toml).unwrap();
        assert_eq!(config.fat_image.len(), 1);
        assert_eq!(config.fat_image[0].host_path, "/host/path/to/existing.fat");
        assert_eq!(config.fat_image[0].mount_point, "/data");
        assert_eq!(config.mapping_count(), 1);
    }

    #[test]
    fn test_parse_fat_mount_temp() {
        let toml = r#"
[[fat_mount]]
mount_point = "/tmp"
size = "10MB"
"#;
        let config = HyperlightFsConfig::from_toml(toml).unwrap();
        assert_eq!(config.fat_mount.len(), 1);
        assert!(config.fat_mount[0].host_path.is_none());
        assert_eq!(config.fat_mount[0].mount_point, "/tmp");
    }

    #[test]
    fn test_parse_fat_mount_persistent() {
        let toml = r#"
[[fat_mount]]
host_path = "/host/path/to/persistent.fat"
mount_point = "/logs"
size = "50MB"
"#;
        let config = HyperlightFsConfig::from_toml(toml).unwrap();
        assert_eq!(config.fat_mount.len(), 1);
        assert_eq!(
            config.fat_mount[0].host_path.as_deref(),
            Some("/host/path/to/persistent.fat")
        );
        assert_eq!(config.fat_mount[0].mount_point, "/logs");
    }

    #[test]
    fn test_parse_multiple_fat_mounts() {
        let toml = r#"
[[fat_image]]
host_path = "/data.fat"
mount_point = "/data"

[[fat_mount]]
mount_point = "/tmp"
size = 1048576

[[fat_mount]]
host_path = "/logs.fat"
mount_point = "/logs"
size = "100MB"
"#;
        let config = HyperlightFsConfig::from_toml(toml).unwrap();
        assert_eq!(config.fat_image.len(), 1);
        assert_eq!(config.fat_mount.len(), 2);
        assert_eq!(config.mapping_count(), 3);
    }

    #[test]
    fn test_parse_mixed_ro_and_fat() {
        let toml = r#"
[[file]]
host_path = "/etc/config.json"
guest = "/config.json"

[[fat_mount]]
mount_point = "/data"
size = "10MB"

[[directory]]
host_path = "/assets"
guest = "/assets"
"#;
        let config = HyperlightFsConfig::from_toml(toml).unwrap();
        assert_eq!(config.file.len(), 1);
        assert_eq!(config.directory.len(), 1);
        assert_eq!(config.fat_mount.len(), 1);
        assert_eq!(config.mapping_count(), 3);
    }

    #[test]
    fn test_fat_image_reject_unknown_fields() {
        let toml = r#"
[[fat_image]]
host_path = "/data.fat"
mount_point = "/data"
unknown = "bad"
"#;
        let result = HyperlightFsConfig::from_toml(toml);
        assert!(result.is_err());
    }

    #[test]
    fn test_fat_mount_reject_unknown_fields() {
        let toml = r#"
[[fat_mount]]
mount_point = "/tmp"
size = "10MB"
unknown = "bad"
"#;
        let result = HyperlightFsConfig::from_toml(toml);
        assert!(result.is_err());
    }

    #[test]
    fn test_fat_mount_missing_size() {
        let toml = r#"
[[fat_mount]]
mount_point = "/tmp"
"#;
        let result = HyperlightFsConfig::from_toml(toml);
        assert!(result.is_err());
    }

    // ==================== SizeValue Tests ====================

    #[test]
    fn test_size_value_bytes() {
        let toml = r#"
[[fat_mount]]
mount_point = "/tmp"
size = 1048576
"#;
        let config = HyperlightFsConfig::from_toml(toml).unwrap();
        let size = config.fat_mount[0].size.to_bytes().unwrap();
        assert_eq!(size, 1048576);
    }

    #[test]
    fn test_size_value_human_bytes() {
        let toml = r#"
[[fat_mount]]
mount_point = "/tmp"
size = "1024B"
"#;
        let config = HyperlightFsConfig::from_toml(toml).unwrap();
        let size = config.fat_mount[0].size.to_bytes().unwrap();
        assert_eq!(size, 1024);
    }

    #[test]
    fn test_size_value_human_kb() {
        let toml = r#"
[[fat_mount]]
mount_point = "/tmp"
size = "10KB"
"#;
        let config = HyperlightFsConfig::from_toml(toml).unwrap();
        let size = config.fat_mount[0].size.to_bytes().unwrap();
        assert_eq!(size, 10_000);
    }

    #[test]
    fn test_size_value_human_kib() {
        let toml = r#"
[[fat_mount]]
mount_point = "/tmp"
size = "10KiB"
"#;
        let config = HyperlightFsConfig::from_toml(toml).unwrap();
        let size = config.fat_mount[0].size.to_bytes().unwrap();
        assert_eq!(size, 10 * 1024);
    }

    #[test]
    fn test_size_value_human_mb() {
        let toml = r#"
[[fat_mount]]
mount_point = "/tmp"
size = "10MB"
"#;
        let config = HyperlightFsConfig::from_toml(toml).unwrap();
        let size = config.fat_mount[0].size.to_bytes().unwrap();
        assert_eq!(size, 10_000_000);
    }

    #[test]
    fn test_size_value_human_mib() {
        let toml = r#"
[[fat_mount]]
mount_point = "/tmp"
size = "10MiB"
"#;
        let config = HyperlightFsConfig::from_toml(toml).unwrap();
        let size = config.fat_mount[0].size.to_bytes().unwrap();
        assert_eq!(size, 10 * 1024 * 1024);
    }

    #[test]
    fn test_size_value_human_gb() {
        let toml = r#"
[[fat_mount]]
mount_point = "/tmp"
size = "1GB"
"#;
        let config = HyperlightFsConfig::from_toml(toml).unwrap();
        let size = config.fat_mount[0].size.to_bytes().unwrap();
        assert_eq!(size, 1_000_000_000);
    }

    #[test]
    fn test_size_value_human_gib() {
        let toml = r#"
[[fat_mount]]
mount_point = "/tmp"
size = "1GiB"
"#;
        let config = HyperlightFsConfig::from_toml(toml).unwrap();
        let size = config.fat_mount[0].size.to_bytes().unwrap();
        assert_eq!(size, 1024 * 1024 * 1024);
    }

    #[test]
    fn test_size_value_human_lowercase() {
        let toml = r#"
[[fat_mount]]
mount_point = "/tmp"
size = "10mb"
"#;
        let config = HyperlightFsConfig::from_toml(toml).unwrap();
        let size = config.fat_mount[0].size.to_bytes().unwrap();
        assert_eq!(size, 10_000_000);
    }

    #[test]
    fn test_size_value_human_with_decimal() {
        let toml = r#"
[[fat_mount]]
mount_point = "/tmp"
size = "1.5GB"
"#;
        let config = HyperlightFsConfig::from_toml(toml).unwrap();
        let size = config.fat_mount[0].size.to_bytes().unwrap();
        assert_eq!(size, 1_500_000_000);
    }

    #[test]
    fn test_size_value_invalid_suffix() {
        let size = SizeValue::Human("10TB".to_string());
        let result = size.to_bytes();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unrecognized size suffix"));
    }

    #[test]
    fn test_size_value_invalid_number() {
        let size = SizeValue::Human("abcMB".to_string());
        let result = size.to_bytes();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid number"));
    }

    #[test]
    fn test_size_value_empty_string() {
        let size = SizeValue::Human("".to_string());
        let result = size.to_bytes();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }
}
