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

//! HyperlightFS image with file metadata for guest memory setup.
//!
//! This module provides the runtime representation of a HyperlightFS image.
//! File mappings store host paths and metadata; actual memory mapping is done
//! at sandbox initialization time using `map_file_cow`.
//!
//! # Guest Address Assignment
//!
//! Guest addresses (GVA/GPA) are not known until sandbox creation time.
//! This image computes file offsets on-demand and generates the final FlatBuffer
//! manifest with actual guest addresses when the mapped files region base is known.

use std::path::PathBuf;

use hyperlight_common::flatbuffer_wrappers::hyperlight_fs::{HyperlightFSData, InodeData};
use hyperlight_common::mem::PAGE_SIZE_USIZE;
use tracing::info;

use super::builder::MappedFile;
use crate::Result;
use crate::error::HyperlightError;
use crate::mem::layout::round_up_to;

/// Get the parent path of a guest path.
///
/// For "/test.txt", returns "/". For "/foo/bar.txt", returns "/foo".
/// For "/" (root), returns "/" (root is its own parent).
fn get_parent_path(path: &str) -> String {
    path.rsplit_once('/')
        .map(|(p, _)| {
            if p.is_empty() {
                "/".to_string()
            } else {
                p.to_string()
            }
        })
        .unwrap_or_else(|| "/".to_string())
}

/// File metadata for guest memory setup.
///
/// This struct stores the host path and metadata for a file that will be
/// mapped into guest memory at sandbox initialization time.
pub struct FileMapping {
    /// Guest path of the file
    guest_path: String,
    /// Host path of the file (used by `map_file_cow` at sandbox init)
    host_path: PathBuf,
    /// Size of the file in bytes
    size: u64,
}

impl FileMapping {
    /// Get the guest path of this file.
    pub fn guest_path(&self) -> &str {
        &self.guest_path
    }

    /// Get the host path of this file.
    pub fn host_path(&self) -> &PathBuf {
        &self.host_path
    }

    /// Get the size of the file in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }
}

/// Type of an internal inode entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InodeEntryType {
    /// Regular read-only file
    File,
    /// Directory
    Directory,
    /// FAT mount point
    FatMount,
}

/// Internal representation of an inode before guest addresses are assigned.
#[derive(Debug, Clone)]
struct InodeEntry {
    /// Type of inode
    entry_type: InodeEntryType,
    /// Guest path
    path: String,
    /// Parent inode index
    parent: u32,
    /// For files/FAT mounts: size in bytes. For directories: 0
    size: u64,
    /// For FAT mounts: mount ID (1-based index). For others: 0
    mount_id: u32,
}

/// Storage for a FAT mount in the image.
///
/// This stores the FAT image itself along with mount point.
/// The FatImage must stay alive for the sandbox lifetime.
pub(crate) struct FatMountStorage {
    /// The FAT image (owns the mmap'd region)
    image: super::fat_image::FatImage,
    /// Mount point path in the guest filesystem
    mount_point: String,
}

impl FatMountStorage {
    /// Create a new FAT mount storage.
    pub(super) fn new(image: super::fat_image::FatImage, mount_point: String) -> Self {
        Self { image, mount_point }
    }

    /// Get a reference to the FAT image.
    pub(crate) fn image(&self) -> &super::fat_image::FatImage {
        &self.image
    }

    /// Get a mutable reference to the FAT image.
    ///
    /// Used by sandbox extraction APIs to read/write files in FAT mounts.
    pub(crate) fn image_mut(&mut self) -> &mut super::fat_image::FatImage {
        &mut self.image
    }

    /// Get the mount point path.
    pub(crate) fn mount_point(&self) -> &str {
        &self.mount_point
    }
}

/// A built HyperlightFS image with zero-copy file mappings.
///
/// The manifest (metadata) is generated on-demand when guest addresses are known.
/// File data is memory-mapped directly from the host filesystem using mmap.
/// This allows mapping large files (GB scale) without copying data.
///
/// # Sharing
///
/// This struct is `Send + Sync` and can be shared across multiple sandboxes.
/// The underlying mmaps share physical pages via the OS page cache.
pub struct HyperlightFSImage {
    /// Inode entries (files, directories, and FAT mounts)
    inode_entries: Vec<InodeEntry>,
    /// Memory-mapped files with metadata (read-only)
    file_mappings: Vec<FileMapping>,
    /// FAT mount storage info (actual FatImages handled separately)
    fat_mounts: Vec<FatMountStorage>,
    /// Total size of the read-only mapped files region (page-aligned)
    ro_files_region_size: usize,
    /// Total size of the FAT mounts region (page-aligned)
    fat_region_size: usize,
}

impl HyperlightFSImage {
    /// Generate a manifest with final guest addresses.
    ///
    /// The `mapped_files_region_base` is the GVA/GPA where the combined mapped region starts.
    /// Layout: [RO files region][FAT mounts region]
    ///
    /// Each entry's guest_address is computed by accumulating page-aligned sizes:
    /// - RO files start at mapped_files_region_base
    /// - FAT mounts start at mapped_files_region_base + ro_files_region_size
    ///
    /// # Arguments
    ///
    /// * `mapped_files_region_base` - The guest address where the mapped region starts
    ///
    /// # Returns
    ///
    /// A FlatBuffer-serialized manifest with guest addresses.
    pub(crate) fn generate_manifest(&self, mapped_files_region_base: u64) -> Result<Vec<u8>> {
        // Compute offsets on-the-fly by accumulating page-aligned sizes
        // RO files are placed first, then FAT mounts
        let mut ro_offset: u64 = 0;
        let fat_region_base = mapped_files_region_base + self.ro_files_region_size as u64;
        let mut fat_offset: u64 = 0;

        let inodes: Vec<InodeData> = self
            .inode_entries
            .iter()
            .map(|entry| {
                match entry.entry_type {
                    InodeEntryType::Directory => {
                        InodeData::directory(entry.path.clone(), entry.parent)
                    }
                    InodeEntryType::File => {
                        // Compute the final guest address from current RO offset
                        let guest_address = mapped_files_region_base + ro_offset;
                        // Advance offset by page-aligned size for next file
                        ro_offset += round_up_to(entry.size as usize, PAGE_SIZE_USIZE) as u64;
                        InodeData::file(entry.path.clone(), entry.parent, guest_address, entry.size)
                    }
                    InodeEntryType::FatMount => {
                        // Compute the final guest address from FAT region base
                        let guest_address = fat_region_base + fat_offset;
                        // Advance FAT offset by page-aligned size
                        fat_offset += round_up_to(entry.size as usize, PAGE_SIZE_USIZE) as u64;
                        InodeData::fat_mount(
                            entry.path.clone(),
                            entry.parent,
                            guest_address,
                            entry.size,
                            entry.mount_id,
                        )
                    }
                }
            })
            .collect();

        let fs_data = HyperlightFSData::new(inodes);

        let manifest: Vec<u8> = (&fs_data).try_into().map_err(|e| {
            HyperlightError::Error(format!("Failed to serialize HyperlightFS manifest: {}", e))
        })?;

        Ok(manifest)
    }

    /// Get the total size of the mapped files region (RO files + FAT mounts).
    pub(crate) fn mapped_files_region_size(&self) -> usize {
        self.ro_files_region_size + self.fat_region_size
    }

    /// Estimate the size of the manifest in bytes (page-aligned).
    ///
    /// This is used to compute where file data should be placed in guest memory.
    /// The estimate is conservative (may be slightly larger than actual).
    pub(crate) fn estimate_manifest_size(&self) -> usize {
        // FlatBuffer overhead estimate:
        // - Root table: ~32 bytes (header, version, vector offset)
        // - Per inode: ~48 bytes base + path length + alignment padding
        const ROOT_OVERHEAD: usize = 64;
        const PER_INODE_OVERHEAD: usize = 64;

        let inodes_size: usize = self
            .inode_entries
            .iter()
            .map(|entry| PER_INODE_OVERHEAD + entry.path.len())
            .sum();

        let total = ROOT_OVERHEAD + inodes_size;

        // Round up to page size
        round_up_to(total, PAGE_SIZE_USIZE)
    }

    /// Get the file mappings for wiring into guest memory.
    pub(crate) fn file_mappings(&self) -> &[FileMapping] {
        &self.file_mappings
    }

    /// Get the FAT mounts for wiring into guest memory.
    pub(crate) fn fat_mounts(&self) -> &[FatMountStorage] {
        &self.fat_mounts
    }

    /// Get mutable access to FAT mounts.
    ///
    /// This is used by the sandbox extraction APIs to read/write files
    /// in FAT mounts while the VM is paused.
    pub(crate) fn fat_mounts_mut(&mut self) -> &mut [FatMountStorage] {
        &mut self.fat_mounts
    }

    /// Get a mutable reference to a FAT image by its mount point.
    ///
    /// This allows writing files to a FAT mount before the sandbox is created.
    ///
    /// # Arguments
    ///
    /// * `mount_point` - The mount point path (e.g., "/" or "/data")
    ///
    /// # Returns
    ///
    /// A mutable reference to the `FatImage` if found, or `None` if no mount
    /// exists at the given path.
    pub fn fat_mount_mut(&mut self, mount_point: &str) -> Option<&mut super::fat_image::FatImage> {
        for mount in &mut self.fat_mounts {
            if mount.mount_point() == mount_point {
                return Some(mount.image_mut());
            }
        }
        None
    }

    /// Synchronously flush all FAT mounts to their backing files.
    ///
    /// This calls `msync(MS_SYNC)` on each FAT mount's mmap'd region,
    /// ensuring all writes made by the guest are durably persisted.
    ///
    /// Called automatically by the sandbox on successful VM halt (HLT).
    ///
    /// # Returns
    ///
    /// `Ok(())` if all syncs succeeded. On error, returns the first error
    /// encountered but still attempts to sync remaining mounts.
    ///
    /// # Notes
    ///
    /// - This is a no-op if there are no FAT mounts
    /// - Temporary FAT images (created with `add_empty_fat_mount`) are skipped
    ///   since they will be deleted on drop
    /// - The kernel tracks dirty pages automatically; syncing clean pages is fast
    pub(crate) fn msync_fat_mounts(&self) -> crate::Result<()> {
        if self.fat_mounts.is_empty() {
            return Ok(());
        }

        tracing::debug!(
            count = self.fat_mounts.len(),
            "Syncing FAT mounts to backing files"
        );

        let mut first_error: Option<crate::HyperlightError> = None;

        for mount in &self.fat_mounts {
            if let Err(e) = mount.image().msync() {
                tracing::warn!(
                    mount_point = mount.mount_point(),
                    error = %e,
                    "Failed to sync FAT mount"
                );
                if first_error.is_none() {
                    first_error = Some(e);
                }
                // Continue syncing other mounts
            }
        }

        match first_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Find a FAT mount that contains the given guest path.
    ///
    /// Returns the index of the mount and the path relative to the mount point.
    ///
    /// # Arguments
    ///
    /// * `guest_path` - Absolute guest path (e.g., "/mnt/fat/subdir/file.txt")
    ///
    /// # Returns
    ///
    /// `Some((mount_index, relative_path))` if the path is within a FAT mount,
    /// `None` if no mount contains this path.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // With mount point "/mnt/fat" (internal API)
    /// let result = image.find_fat_mount("/mnt/fat/subdir/file.txt");
    /// // Returns Some((0, "/subdir/file.txt"))
    ///
    /// let result = image.find_fat_mount("/other/path");
    /// // Returns None
    /// ```
    pub(crate) fn find_fat_mount(&self, guest_path: &str) -> Option<(usize, String)> {
        // Normalize the path: ensure it starts with "/" and doesn't have trailing "/"
        let normalized_path = if guest_path.starts_with('/') {
            guest_path.trim_end_matches('/')
        } else {
            return None; // Require absolute paths
        };

        for (idx, mount) in self.fat_mounts.iter().enumerate() {
            let mount_point = mount.mount_point().trim_end_matches('/');

            // Check if path is within this mount
            if normalized_path == mount_point {
                // Path is exactly the mount point (root of the FAT)
                return Some((idx, "/".to_string()));
            } else if normalized_path.starts_with(mount_point)
                && normalized_path[mount_point.len()..].starts_with('/')
            {
                // Path is under the mount point
                let relative = &normalized_path[mount_point.len()..];
                return Some((idx, relative.to_string()));
            }
        }

        None
    }

    /// Get a summary of all entries in this image.
    ///
    /// Returns information about all files, directories, and FAT mounts
    /// stored in this image. Useful for validation tools and debugging.
    pub fn file_summary(&self) -> super::builder::BuildManifest {
        use std::path::PathBuf;

        use super::builder::ManifestEntry;

        let mut entries = Vec::new();
        let mut total_size = 0u64;

        // Map inodes to manifest entries, using host_path from file_mappings
        // where available (for files)
        let mut file_mapping_idx = 0;

        for entry in &self.inode_entries {
            let is_dir = entry.entry_type == InodeEntryType::Directory;
            let is_fat = entry.entry_type == InodeEntryType::FatMount;

            // Get host path from file_mappings for regular files
            let host_path = if entry.entry_type == InodeEntryType::File
                && file_mapping_idx < self.file_mappings.len()
            {
                let path = self.file_mappings[file_mapping_idx].host_path().clone();
                file_mapping_idx += 1;
                path
            } else {
                // Directories and FAT mounts don't have a direct host path
                PathBuf::from("")
            };

            // FAT mounts show as directories in the listing
            let is_dir_or_fat = is_dir || is_fat;

            if !is_dir && !is_fat {
                total_size += entry.size;
            }

            entries.push(ManifestEntry {
                host_path,
                guest_path: entry.path.clone(),
                size: entry.size,
                is_dir: is_dir_or_fat,
            });
        }

        super::builder::BuildManifest {
            files: entries,
            total_size,
        }
    }
}

impl std::fmt::Debug for HyperlightFSImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HyperlightFSImage")
            .field("inode_count", &self.inode_entries.len())
            .field("ro_files_region_size", &self.ro_files_region_size)
            .field("fat_region_size", &self.fat_region_size)
            .field("file_mappings_count", &self.file_mappings.len())
            .field("fat_mounts_count", &self.fat_mounts.len())
            .finish()
    }
}

/// Build a HyperlightFS image with file metadata.
///
/// # Arguments
///
/// * `files` - Read-only file mappings from the builder
/// * `fat_mounts` - FAT mount entries with images and mount points
pub(super) fn build_image(
    mut files: Vec<MappedFile>,
    fat_mounts: Vec<super::builder::FatMountEntry>,
) -> Result<HyperlightFSImage> {
    use std::collections::HashMap;

    // Sort files by guest path for consistent ordering
    files.sort_by(|a, b| a.guest_path.cmp(&b.guest_path));

    // Calculate RO files region size (sum of page-aligned file sizes)
    let ro_files_region_size: usize = files
        .iter()
        .filter(|f| !f.is_dir)
        .map(|f| round_up_to(f.size as usize, PAGE_SIZE_USIZE))
        .sum();

    // Calculate FAT region size (sum of page-aligned FAT image sizes)
    let fat_region_size: usize = fat_mounts
        .iter()
        .map(|m| round_up_to(m.image.size(), PAGE_SIZE_USIZE))
        .sum();

    // Build parent index map for directories only
    // FAT mounts are not added - they're leaf nodes, not containers for other inodes
    let mut parent_map: HashMap<String, u32> = HashMap::new();
    for (idx, file) in files.iter().enumerate() {
        parent_map.insert(file.guest_path.clone(), idx as u32);
    }

    // Create inode entries for RO files (guest addresses assigned later)
    let mut inode_entries: Vec<InodeEntry> = files
        .iter()
        .map(|file| {
            // Find parent directory
            let parent_path = get_parent_path(&file.guest_path);
            let parent_idx = parent_map.get(&parent_path).copied().unwrap_or(0);

            InodeEntry {
                entry_type: if file.is_dir {
                    InodeEntryType::Directory
                } else {
                    InodeEntryType::File
                },
                path: file.guest_path.clone(),
                parent: parent_idx,
                size: file.size,
                mount_id: 0,
            }
        })
        .collect();

    let mut fat_mounts_storage: Vec<FatMountStorage> = Vec::new();
    for (mount_idx, fat_mount) in fat_mounts.into_iter().enumerate() {
        // Find parent directory for the mount point
        let parent_path = get_parent_path(&fat_mount.mount_point);
        let parent_idx = parent_map.get(&parent_path).copied().unwrap_or(0);

        // Mount IDs are 1-based
        let mount_id = (mount_idx + 1) as u32;
        let size = fat_mount.image.size();

        inode_entries.push(InodeEntry {
            entry_type: InodeEntryType::FatMount,
            path: fat_mount.mount_point.clone(),
            parent: parent_idx,
            size: size as u64,
            mount_id,
        });

        fat_mounts_storage.push(FatMountStorage::new(fat_mount.image, fat_mount.mount_point));
    }

    // Collect file mappings (actual mmap happens at sandbox init via map_file_cow)
    let mut file_mappings: Vec<FileMapping> = Vec::new();

    for file in &files {
        if !file.is_dir && !file.host_path.as_os_str().is_empty() {
            // Verify file exists and get size
            let metadata = std::fs::metadata(&file.host_path).map_err(|e| {
                HyperlightError::Error(format!("Failed to stat {:?}: {}", file.host_path, e))
            })?;

            let file_size = metadata.len();
            if file_size == 0 {
                // Can't mmap empty files, skip
                info!(
                    guest = %file.guest_path,
                    "Skipping empty file (cannot map)"
                );
                continue;
            }

            info!(
                guest = %file.guest_path,
                host = %file.host_path.display(),
                size = file_size,
                "Added file mapping (will be mapped at sandbox init)"
            );

            file_mappings.push(FileMapping {
                guest_path: file.guest_path.clone(),
                host_path: file.host_path.clone(),
                size: file_size,
            });
        }
    }

    info!(
        inode_count = inode_entries.len(),
        file_count = file_mappings.len(),
        ro_files_region_size = ro_files_region_size,
        fat_region_size = fat_region_size,
        fat_mount_count = fat_mounts_storage.len(),
        "HyperlightFS image built"
    );

    Ok(HyperlightFSImage {
        inode_entries,
        file_mappings,
        fat_mounts: fat_mounts_storage,
        ro_files_region_size,
        fat_region_size,
    })
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::super::builder::FatMountEntry;
    use super::super::fat_image::{FatImage, MIN_FAT_IMAGE_SIZE};
    use super::*;

    /// Helper to create a test FAT mount entry
    fn make_fat_mount(mount_point: &str) -> FatMountEntry {
        FatMountEntry {
            image: FatImage::create_temp(MIN_FAT_IMAGE_SIZE).expect("Failed to create temp FAT"),
            mount_point: mount_point.to_string(),
        }
    }

    // Tests for helper functions

    #[test]
    fn test_get_parent_path() {
        // Root's parent is itself
        assert_eq!(get_parent_path("/"), "/");

        // Top-level file's parent is root
        assert_eq!(get_parent_path("/test.txt"), "/");
        assert_eq!(get_parent_path("/data"), "/");

        // Nested file's parent is its directory
        assert_eq!(get_parent_path("/foo/bar.txt"), "/foo");
        assert_eq!(get_parent_path("/a/b/c"), "/a/b");
        assert_eq!(
            get_parent_path("/deep/nested/path/file.txt"),
            "/deep/nested/path"
        );
    }

    #[test]
    fn test_parent_index_for_root_level_file() {
        // Verify that a file at "/test.txt" has parent index pointing to "/"
        let files = vec![
            MappedFile {
                host_path: std::path::PathBuf::new(),
                guest_path: "/".to_string(),
                size: 0,
                is_dir: true,
            },
            MappedFile {
                host_path: std::path::PathBuf::new(),
                guest_path: "/test.txt".to_string(),
                size: 100,
                is_dir: false,
            },
        ];

        let image = build_image(files, vec![]).expect("Failed to build image");

        // Root dir should be at index 0
        let root = &image.inode_entries[0];
        assert_eq!(root.path, "/");
        assert_eq!(root.entry_type, InodeEntryType::Directory);

        // File should have parent index 0 (pointing to root)
        let file = &image.inode_entries[1];
        assert_eq!(file.path, "/test.txt");
        assert_eq!(
            file.parent, 0,
            "File at /test.txt should have parent index 0 (root)"
        );
    }

    #[test]
    fn test_fat_mount_parent_is_root() {
        // FAT mount at "/data" should have parent "/" (index 0)
        let files = vec![MappedFile {
            host_path: std::path::PathBuf::new(),
            guest_path: "/".to_string(),
            size: 0,
            is_dir: true,
        }];

        let fat_mounts = vec![make_fat_mount("/data")];
        let image = build_image(files, fat_mounts).expect("Failed to build image");

        let fat_entry = image
            .inode_entries
            .iter()
            .find(|e| e.entry_type == InodeEntryType::FatMount)
            .expect("FAT mount not found");

        assert_eq!(fat_entry.path, "/data");
        assert_eq!(
            fat_entry.parent, 0,
            "FAT mount at /data should have parent index 0 (root)"
        );
    }

    #[test]
    fn test_nested_file_parent_index() {
        // File at "/subdir/file.txt" should have parent "/subdir"
        let files = vec![
            MappedFile {
                host_path: std::path::PathBuf::new(),
                guest_path: "/".to_string(),
                size: 0,
                is_dir: true,
            },
            MappedFile {
                host_path: std::path::PathBuf::new(),
                guest_path: "/subdir".to_string(),
                size: 0,
                is_dir: true,
            },
            MappedFile {
                host_path: std::path::PathBuf::new(),
                guest_path: "/subdir/file.txt".to_string(),
                size: 100,
                is_dir: false,
            },
        ];

        let image = build_image(files, vec![]).expect("Failed to build image");

        // Find subdir index
        let subdir_idx = image
            .inode_entries
            .iter()
            .position(|e| e.path == "/subdir")
            .expect("subdir not found");

        // File should have parent index pointing to subdir
        let file = image
            .inode_entries
            .iter()
            .find(|e| e.path == "/subdir/file.txt")
            .expect("file not found");

        assert_eq!(
            file.parent, subdir_idx as u32,
            "File at /subdir/file.txt should have parent index {} (subdir)",
            subdir_idx
        );
    }

    // Original tests

    #[test]
    fn test_build_empty_image() {
        let files = vec![MappedFile {
            host_path: std::path::PathBuf::new(),
            guest_path: String::new(),
            size: 0,
            is_dir: true,
        }];

        let image = build_image(files, vec![]).expect("Failed to build image");

        assert_eq!(image.inode_entries.len(), 1);
        assert_eq!(image.file_mappings.len(), 0); // Root dir only, no files
    }

    #[test]
    fn test_generate_manifest_with_guest_address() {
        let files = vec![
            MappedFile {
                host_path: std::path::PathBuf::new(),
                guest_path: "/".to_string(),
                size: 0,
                is_dir: true,
            },
            MappedFile {
                host_path: std::path::PathBuf::new(), // Empty path = no mmap
                guest_path: "/test.txt".to_string(),
                size: 100,
                is_dir: false,
            },
        ];

        let image = build_image(files, vec![]).expect("Failed to build image");

        // Generate manifest with a specific base address
        let mapped_files_region_base: u64 = 0x1000_0000;
        let manifest = image
            .generate_manifest(mapped_files_region_base)
            .expect("Failed to generate manifest");

        // Parse the manifest back and verify guest_address
        let parsed: HyperlightFSData = manifest.as_slice().try_into().unwrap();
        assert_eq!(parsed.inodes.len(), 2);

        // Find the file inode
        let file_inode = parsed.inodes.iter().find(|i| i.is_file()).unwrap();
        assert_eq!(file_inode.guest_address, mapped_files_region_base); // First file at offset 0
        assert_eq!(file_inode.size, 100);
    }

    #[cfg(unix)]
    #[test]
    fn test_build_with_file() {
        let tmp = TempDir::new().unwrap();
        let file_path = tmp.path().join("test.txt");
        std::fs::write(&file_path, b"hello world").unwrap();

        let files = vec![MappedFile {
            host_path: file_path.clone(),
            guest_path: "/test.txt".to_string(),
            size: 11,
            is_dir: false,
        }];

        let image = build_image(files, vec![]).expect("Failed to build image");

        assert_eq!(image.file_mappings().len(), 1);
        assert_eq!(image.file_mappings()[0].guest_path(), "/test.txt");
        assert_eq!(image.file_mappings()[0].size(), 11);

        // Verify host_path is stored correctly
        assert_eq!(image.file_mappings()[0].host_path(), &file_path);

        // Generate manifest and verify
        let manifest = image.generate_manifest(0x2000_0000).unwrap();
        let parsed: HyperlightFSData = manifest.as_slice().try_into().unwrap();
        let file_inode = parsed.inodes.iter().find(|i| i.is_file()).unwrap();
        assert_eq!(file_inode.guest_address, 0x2000_0000);
    }

    #[test]
    fn test_build_with_fat_mount() {
        // Build image with a single FAT mount
        let files = vec![MappedFile {
            host_path: std::path::PathBuf::new(),
            guest_path: "/".to_string(),
            size: 0,
            is_dir: true,
        }];

        let fat_mounts = vec![make_fat_mount("/data")];
        let fat_size = MIN_FAT_IMAGE_SIZE;

        let image = build_image(files, fat_mounts).expect("Failed to build image");

        // Should have 2 inodes: root dir + FAT mount
        assert_eq!(image.inode_entries.len(), 2);

        // Check FAT mount entry
        let fat_entry = image
            .inode_entries
            .iter()
            .find(|e| e.entry_type == InodeEntryType::FatMount)
            .expect("FAT mount entry not found");
        assert_eq!(fat_entry.path, "/data");
        assert_eq!(fat_entry.size, fat_size as u64);
        assert_eq!(fat_entry.mount_id, 1); // First mount = ID 1

        // Check FAT mount storage
        assert_eq!(image.fat_mounts.len(), 1);
        assert_eq!(image.fat_mounts[0].mount_point, "/data");

        // Check total region size (no RO files, just FAT)
        assert_eq!(image.mapped_files_region_size(), fat_size);
    }

    #[test]
    fn test_generate_manifest_with_fat_mount() {
        let files = vec![
            MappedFile {
                host_path: std::path::PathBuf::new(),
                guest_path: "/".to_string(),
                size: 0,
                is_dir: true,
            },
            MappedFile {
                host_path: std::path::PathBuf::new(),
                guest_path: "/readme.txt".to_string(),
                size: 100,
                is_dir: false,
            },
        ];

        let fat_mounts = vec![make_fat_mount("/mnt/fat")];
        let fat_size = MIN_FAT_IMAGE_SIZE;

        let image = build_image(files, fat_mounts).expect("Failed to build image");

        // Generate manifest
        let base_addr: u64 = 0x1000_0000;
        let manifest = image.generate_manifest(base_addr).unwrap();
        let parsed: HyperlightFSData = manifest.as_slice().try_into().unwrap();

        // Should have 3 inodes: root, file, FAT mount
        assert_eq!(parsed.inodes.len(), 3);

        // Find the file inode - should be at base_addr (RO region starts first)
        let file_inode = parsed.inodes.iter().find(|i| i.is_file()).unwrap();
        assert_eq!(file_inode.guest_address, base_addr);

        // Find the FAT mount inode - should be after RO region
        let fat_inode = parsed.inodes.iter().find(|i| i.is_fat_mount()).unwrap();
        // RO file is 100 bytes, page-aligned = 4096 bytes
        let expected_fat_addr = base_addr + 4096; // After page-aligned RO file
        assert_eq!(fat_inode.guest_address, expected_fat_addr);
        assert_eq!(fat_inode.size, fat_size as u64);
        assert_eq!(fat_inode.mount_id, 1);
        assert_eq!(fat_inode.path, "/mnt/fat");
    }

    #[test]
    fn test_multiple_fat_mounts() {
        let files = vec![MappedFile {
            host_path: std::path::PathBuf::new(),
            guest_path: "/".to_string(),
            size: 0,
            is_dir: true,
        }];

        // Two FAT mounts (both MIN_FAT_IMAGE_SIZE)
        let fat_mounts = vec![make_fat_mount("/data"), make_fat_mount("/scratch")];
        let fat_size = MIN_FAT_IMAGE_SIZE;

        let image = build_image(files, fat_mounts).expect("Failed to build image");

        // Should have 3 inodes: root + 2 FAT mounts
        assert_eq!(image.inode_entries.len(), 3);

        // Check mount IDs are sequential
        let fat_entries: Vec<_> = image
            .inode_entries
            .iter()
            .filter(|e| e.entry_type == InodeEntryType::FatMount)
            .collect();
        assert_eq!(fat_entries.len(), 2);
        assert_eq!(fat_entries[0].mount_id, 1);
        assert_eq!(fat_entries[1].mount_id, 2);

        // Check total region size includes both FAT mounts
        assert_eq!(image.mapped_files_region_size(), fat_size * 2);

        // Generate manifest and verify addresses
        let base_addr: u64 = 0x2000_0000;
        let manifest = image.generate_manifest(base_addr).unwrap();
        let parsed: HyperlightFSData = manifest.as_slice().try_into().unwrap();

        let fat_inodes: Vec<_> = parsed.inodes.iter().filter(|i| i.is_fat_mount()).collect();
        assert_eq!(fat_inodes.len(), 2);

        // First FAT mount at base (no RO files)
        assert_eq!(fat_inodes[0].guest_address, base_addr);
        assert_eq!(fat_inodes[0].mount_id, 1);

        // Second FAT mount after first (page-aligned)
        assert_eq!(fat_inodes[1].guest_address, base_addr + fat_size as u64);
        assert_eq!(fat_inodes[1].mount_id, 2);
    }

    #[test]
    fn test_estimate_manifest_size_includes_fat_mounts() {
        let files = vec![MappedFile {
            host_path: std::path::PathBuf::new(),
            guest_path: "/".to_string(),
            size: 0,
            is_dir: true,
        }];

        // Build without FAT mounts
        let image_no_fat = build_image(files.clone(), vec![]).unwrap();
        let size_no_fat = image_no_fat.estimate_manifest_size();

        // Build with FAT mount
        let fat_mounts = vec![make_fat_mount("/data")];
        let image_with_fat = build_image(files, fat_mounts).unwrap();
        let size_with_fat = image_with_fat.estimate_manifest_size();

        // Size with FAT should be larger (more inodes)
        assert!(
            size_with_fat >= size_no_fat,
            "Manifest size with FAT ({}) should be >= without FAT ({})",
            size_with_fat,
            size_no_fat
        );
    }

    #[test]
    fn test_mixed_ro_files_and_fat_region_layout() {
        // Verify that RO files come before FAT mounts in memory layout
        let files = vec![
            MappedFile {
                host_path: std::path::PathBuf::new(),
                guest_path: "/".to_string(),
                size: 0,
                is_dir: true,
            },
            MappedFile {
                host_path: std::path::PathBuf::new(),
                guest_path: "/file1.txt".to_string(),
                size: 1000,
                is_dir: false,
            },
            MappedFile {
                host_path: std::path::PathBuf::new(),
                guest_path: "/file2.txt".to_string(),
                size: 2000,
                is_dir: false,
            },
        ];

        let fat_mounts = vec![make_fat_mount("/data")];
        let fat_size = MIN_FAT_IMAGE_SIZE;

        let image = build_image(files, fat_mounts).unwrap();

        // Total region: RO (8192 bytes page-aligned) + FAT (fat_size)
        // RO region: 1000 bytes -> 4096 (page-aligned) + 2000 -> 4096 = 8192 total
        assert_eq!(image.mapped_files_region_size(), 8192 + fat_size);

        // Generate manifest
        let base_addr: u64 = 0x1000_0000;
        let manifest = image.generate_manifest(base_addr).unwrap();
        let parsed: HyperlightFSData = manifest.as_slice().try_into().unwrap();

        // Files should be in RO region (starting at base)
        let file_inodes: Vec<_> = parsed.inodes.iter().filter(|i| i.is_file()).collect();
        assert_eq!(file_inodes[0].guest_address, base_addr); // file1 at base
        assert_eq!(file_inodes[1].guest_address, base_addr + 4096); // file2 at +4096

        // FAT mount should be after RO region
        let fat_inode = parsed.inodes.iter().find(|i| i.is_fat_mount()).unwrap();
        assert_eq!(fat_inode.guest_address, base_addr + 8192); // After RO region
    }

    // ---- Tests for find_fat_mount path resolution ----

    #[test]
    fn test_find_fat_mount_exact_match() {
        // Build image with FAT mount at /data
        let files = vec![MappedFile {
            host_path: std::path::PathBuf::new(),
            guest_path: "/".to_string(),
            size: 0,
            is_dir: true,
        }];
        let fat_mounts = vec![make_fat_mount("/data")];
        let image = build_image(files, fat_mounts).unwrap();

        // Exact mount point should return "/"
        let result = image.find_fat_mount("/data");
        assert_eq!(result, Some((0, "/".to_string())));
    }

    #[test]
    fn test_find_fat_mount_nested_path() {
        let files = vec![MappedFile {
            host_path: std::path::PathBuf::new(),
            guest_path: "/".to_string(),
            size: 0,
            is_dir: true,
        }];
        let fat_mounts = vec![make_fat_mount("/mnt/fat")];
        let image = build_image(files, fat_mounts).unwrap();

        // Path within mount
        let result = image.find_fat_mount("/mnt/fat/subdir/file.txt");
        assert_eq!(result, Some((0, "/subdir/file.txt".to_string())));

        // Immediate child
        let result = image.find_fat_mount("/mnt/fat/file.txt");
        assert_eq!(result, Some((0, "/file.txt".to_string())));
    }

    #[test]
    fn test_find_fat_mount_not_in_mount() {
        let files = vec![MappedFile {
            host_path: std::path::PathBuf::new(),
            guest_path: "/".to_string(),
            size: 0,
            is_dir: true,
        }];
        let fat_mounts = vec![make_fat_mount("/mnt/fat")];
        let image = build_image(files, fat_mounts).unwrap();

        // Path not in any mount
        assert_eq!(image.find_fat_mount("/other/path"), None);
        assert_eq!(image.find_fat_mount("/mnt/other"), None);

        // Path that starts with mount point prefix but isn't inside it
        // e.g., "/mnt/fatty" should NOT match "/mnt/fat"
        assert_eq!(image.find_fat_mount("/mnt/fatty/file.txt"), None);
    }

    #[test]
    fn test_find_fat_mount_multiple_mounts() {
        let files = vec![MappedFile {
            host_path: std::path::PathBuf::new(),
            guest_path: "/".to_string(),
            size: 0,
            is_dir: true,
        }];
        let fat_mounts = vec![make_fat_mount("/data"), make_fat_mount("/output")];
        let image = build_image(files, fat_mounts).unwrap();

        // Should find correct mount
        let result = image.find_fat_mount("/data/file.txt");
        assert_eq!(result, Some((0, "/file.txt".to_string())));

        let result = image.find_fat_mount("/output/results/test.log");
        assert_eq!(result, Some((1, "/results/test.log".to_string())));
    }

    #[test]
    fn test_find_fat_mount_handles_trailing_slash() {
        let files = vec![MappedFile {
            host_path: std::path::PathBuf::new(),
            guest_path: "/".to_string(),
            size: 0,
            is_dir: true,
        }];
        let fat_mounts = vec![make_fat_mount("/data")];
        let image = build_image(files, fat_mounts).unwrap();

        // Path with trailing slash
        let result = image.find_fat_mount("/data/");
        assert_eq!(result, Some((0, "/".to_string())));

        let result = image.find_fat_mount("/data/subdir/");
        assert_eq!(result, Some((0, "/subdir".to_string())));
    }

    #[test]
    fn test_find_fat_mount_requires_absolute_path() {
        let files = vec![MappedFile {
            host_path: std::path::PathBuf::new(),
            guest_path: "/".to_string(),
            size: 0,
            is_dir: true,
        }];
        let fat_mounts = vec![make_fat_mount("/data")];
        let image = build_image(files, fat_mounts).unwrap();

        // Relative path should fail
        assert_eq!(image.find_fat_mount("data/file.txt"), None);
        assert_eq!(image.find_fat_mount(""), None);
    }
}
