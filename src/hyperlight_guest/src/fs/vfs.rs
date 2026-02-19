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

//! Virtual filesystem layer for the guest.
//!
//! This module provides the VFS mount table and path resolution logic that
//! routes filesystem operations to the appropriate backend (read-only files
//! or FAT mounts).
//!
//! # Architecture
//!
//! The VFS maintains:
//! - A mount table mapping paths to backends (sorted by path length for
//!   longest-prefix matching)
//! - A current working directory for relative path resolution
//!
//! # Spec Reference
//!
//! See spec §5.5.1 for VFS Mount Table design.

use alloc::string::String;
use alloc::vec::Vec;

use super::error::FsError;
use super::fat::GuestFat;

// ============================================================================
// Mount Backend
// ============================================================================

/// Storage backend types for mount points.
///
/// Each mount point is backed by one of these storage types.
pub enum MountBackend {
    /// Read-only memory-mapped file/directory from the manifest.
    ReadOnly,

    /// Read-write FAT filesystem.
    Fat(GuestFat),
}

impl MountBackend {
    /// Returns true if this backend is read-only.
    #[inline]
    pub fn is_readonly(&self) -> bool {
        matches!(self, MountBackend::ReadOnly)
    }
}

// ============================================================================
// Mount
// ============================================================================

/// A single mount point in the VFS.
///
/// Associates an absolute path with a storage backend.
pub struct Mount {
    /// Absolute path where this mount is rooted (e.g., "/data").
    ///
    /// Always starts with "/" and never ends with "/" (except for root "/").
    path: String,

    /// The storage backend for this mount.
    backend: MountBackend,
}

impl Mount {
    /// Create a new mount point.
    ///
    /// # Arguments
    ///
    /// * `path` - Absolute mount path (must start with "/")
    /// * `backend` - The storage backend
    ///
    /// # Panics
    ///
    /// Panics if `path` doesn't start with "/".
    pub fn new(path: String, backend: MountBackend) -> Self {
        assert!(path.starts_with('/'), "mount path must be absolute");
        Self { path, backend }
    }

    /// Returns the mount path.
    #[inline]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Returns a reference to the backend.
    #[inline]
    pub fn backend(&self) -> &MountBackend {
        &self.backend
    }

    /// Returns a mutable reference to the backend.
    #[inline]
    pub fn backend_mut(&mut self) -> &mut MountBackend {
        &mut self.backend
    }

    /// Check if the given path is under this mount point.
    ///
    /// Returns `Some(relative_path)` if the path matches, `None` otherwise.
    /// The returned reference has the same lifetime as the `path` argument.
    pub fn matches<'a>(&self, path: &'a str) -> Option<&'a str> {
        if self.path == "/" {
            // Root mount matches everything
            Some(path.strip_prefix('/').unwrap_or(path))
        } else if path == self.path {
            // Exact match - relative path is empty (root of mount)
            Some("")
        } else if path.starts_with(&self.path) {
            // Check for proper prefix (must be followed by "/" or end)
            let rest = &path[self.path.len()..];
            if let Some(stripped) = rest.strip_prefix('/') {
                Some(stripped)
            } else {
                None // Not a proper prefix match (e.g., "/data" doesn't match "/datafile")
            }
        } else {
            None
        }
    }
}

// ============================================================================
// VFS
// ============================================================================

/// Virtual filesystem managing mounts and path resolution.
///
/// # Path Resolution
///
/// 1. Normalize the path (resolve `.`, `..`, make absolute using cwd)
/// 2. Search mounts in order (sorted by path length descending)
/// 3. Return first mount where path starts with mount.path
/// 4. Extract relative path by stripping mount prefix
///
/// # Spec Reference
///
/// See spec §5.5.1 for design and §7.3 for path resolution details.
pub struct Vfs {
    /// Mount table, sorted by path length descending for longest-prefix matching.
    mounts: Vec<Mount>,

    /// Current working directory (always absolute, always ends without "/").
    cwd: String,

    /// Home directory for tilde expansion (always absolute, no trailing slash).
    /// Defaults to "/". When a path starts with "~" or "~/", it is replaced
    /// with this directory.
    home_dir: String,
}

impl Vfs {
    /// Create a new empty VFS with cwd set to "/" and home_dir set to "/".
    pub fn new() -> Self {
        Self {
            mounts: Vec::new(),
            cwd: String::from("/"),
            home_dir: String::from("/"),
        }
    }

    /// Set the home directory for tilde expansion.
    ///
    /// # Arguments
    ///
    /// * `home` - Absolute path to use as home directory (e.g., "/root", "/home/user")
    ///
    /// # Errors
    ///
    /// Returns `FsError::InvalidPath` if `home` is not absolute.
    pub fn set_home_dir(&mut self, home: &str) -> Result<(), FsError> {
        if !home.starts_with('/') {
            return Err(FsError::InvalidPath);
        }
        // Normalize: remove trailing slash (unless root)
        let normalized = if home.len() > 1 && home.ends_with('/') {
            &home[..home.len() - 1]
        } else {
            home
        };
        self.home_dir = String::from(normalized);
        Ok(())
    }

    /// Get the home directory.
    #[inline]
    pub fn home_dir(&self) -> &str {
        &self.home_dir
    }

    /// Add a mount point.
    ///
    /// The mount is inserted at the correct position to maintain
    /// descending path length order.
    ///
    /// # Errors
    ///
    /// Returns `FsError::AlreadyExists` if a mount already exists at this path.
    pub fn add_mount(&mut self, mount: Mount) -> Result<(), FsError> {
        // Check for duplicate mount point
        if self.mounts.iter().any(|m| m.path == mount.path) {
            return Err(FsError::AlreadyExists);
        }

        // Insert at correct position (sorted by path length descending)
        let pos = self
            .mounts
            .iter()
            .position(|m| m.path.len() < mount.path.len())
            .unwrap_or(self.mounts.len());

        self.mounts.insert(pos, mount);
        Ok(())
    }

    /// Remove a mount point.
    ///
    /// # Errors
    ///
    /// Returns `FsError::NotFound` if no mount exists at this path.
    pub fn remove_mount(&mut self, path: &str) -> Result<Mount, FsError> {
        let pos = self
            .mounts
            .iter()
            .position(|m| m.path == path)
            .ok_or(FsError::NotFound)?;

        Ok(self.mounts.remove(pos))
    }

    /// Get the current working directory.
    #[inline]
    pub fn cwd(&self) -> &str {
        &self.cwd
    }

    /// Change the current working directory.
    ///
    /// Note: This validates that the path resolves to a mount or is a parent
    /// of a mount point, but does not verify that the directory actually exists
    /// on the filesystem. Full validation requires going through the mount
    /// backend which will happen when operations are performed in this directory.
    ///
    /// # Errors
    ///
    /// Returns `FsError::InvalidPath` if the path is malformed.
    /// Returns `FsError::NotFound` if no mount handles this path and the path
    /// is not a parent of any mount point.
    pub fn set_cwd(&mut self, path: &str) -> Result<(), FsError> {
        let normalized = self.normalize_path(path)?;

        // Verify the path resolves to a valid mount point or is a parent of one.
        // This doesn't verify the directory exists on disk, but ensures
        // the path is at least within or above a mounted filesystem.
        if !normalized.is_empty() && normalized != "/" {
            // Check that a mount handles this path, or the path is a parent
            // of a mount point (virtual directory).
            if self.resolve(&normalized).is_err() && !self.is_mount_parent(&normalized) {
                return Err(FsError::NotFound);
            }
        }

        self.cwd = normalized;
        Ok(())
    }

    /// Normalize a path to an absolute path.
    ///
    /// - Resolves `.` (current directory)
    /// - Resolves `..` (parent directory)
    /// - Makes relative paths absolute using cwd
    /// - Removes trailing slashes (except for root)
    ///
    /// # Examples
    ///
    /// ```text
    /// Input (cwd="/")      -> Output
    /// "/foo/bar"           -> "/foo/bar"
    /// "/foo//bar"          -> "/foo/bar"     (double slash removed)
    /// "/foo/./bar"         -> "/foo/bar"     (. resolved)
    /// "/foo/bar/.."        -> "/foo"         (.. resolved)
    /// "foo"                -> "/foo"         (relative to cwd)
    /// "/foo/"              -> "/foo"         (trailing slash removed)
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `FsError::InvalidPath` if the path is empty or contains
    /// invalid sequences (e.g., too many `..`).
    pub fn normalize_path(&self, path: &str) -> Result<String, FsError> {
        if path.is_empty() {
            return Err(FsError::InvalidPath);
        }

        // Expand tilde: "~" -> home_dir, "~/foo" -> home_dir + "/foo"
        let expanded = if path == "~" {
            self.home_dir.clone()
        } else if let Some(rest) = path.strip_prefix("~/") {
            if self.home_dir == "/" {
                alloc::format!("/{}", rest)
            } else {
                alloc::format!("{}/{}", self.home_dir, rest)
            }
        } else {
            String::from(path)
        };

        // Start with absolute path
        let abs_path = if expanded.starts_with('/') {
            expanded
        } else {
            // Make absolute by prepending cwd
            if self.cwd == "/" {
                alloc::format!("/{}", expanded)
            } else {
                alloc::format!("{}/{}", self.cwd, expanded)
            }
        };

        // Split into components and resolve . and ..
        let mut components: Vec<&str> = Vec::new();

        for component in abs_path.split('/') {
            match component {
                "" | "." => {
                    // Skip empty (from "//") and current directory
                }
                ".." => {
                    // Go up one level (if possible)
                    if components.pop().is_none() {
                        // Can't go above root
                        return Err(FsError::InvalidPath);
                    }
                }
                other => {
                    components.push(other);
                }
            }
        }

        // Rebuild the path
        if components.is_empty() {
            Ok(String::from("/"))
        } else {
            let mut result = String::new();
            for component in components {
                result.push('/');
                result.push_str(component);
            }
            Ok(result)
        }
    }

    /// Resolve a path to a mount and relative path within that mount.
    ///
    /// Uses longest-prefix matching to find the best mount.
    ///
    /// # Returns
    ///
    /// A tuple of (mount_index, relative_path) where:
    /// - `mount_index` is the index into `self.mounts`
    /// - `relative_path` is the path relative to the mount point
    ///
    /// # Errors
    ///
    /// Returns `FsError::NotFound` if no mount matches the path.
    pub fn resolve(&self, path: &str) -> Result<(usize, String), FsError> {
        let normalized = self.normalize_path(path)?;

        // Search mounts (already sorted by path length descending)
        for (idx, mount) in self.mounts.iter().enumerate() {
            if let Some(relative) = mount.matches(&normalized) {
                return Ok((idx, String::from(relative)));
            }
        }

        Err(FsError::NotFound)
    }

    /// Get the mount path for a given file path.
    ///
    /// Uses longest-prefix matching to find the mount point.
    ///
    /// # Returns
    ///
    /// The mount path (e.g., "/data" for path "/data/file.txt").
    ///
    /// # Errors
    ///
    /// Returns `FsError::NotFound` if no mount matches the path.
    pub fn get_mount_path(&self, path: &str) -> Result<String, FsError> {
        let normalized = self.normalize_path(path)?;

        // Search mounts (already sorted by path length descending)
        for mount in self.mounts.iter() {
            if mount.matches(&normalized).is_some() {
                return Ok(mount.path.clone());
            }
        }

        Err(FsError::NotFound)
    }

    /// Get a reference to a mount by index.
    #[inline]
    pub fn get_mount(&self, index: usize) -> Option<&Mount> {
        self.mounts.get(index)
    }

    /// Get a mutable reference to a mount by index.
    #[inline]
    pub fn get_mount_mut(&mut self, index: usize) -> Option<&mut Mount> {
        self.mounts.get_mut(index)
    }

    /// Returns the number of mounts.
    #[inline]
    pub fn mount_count(&self) -> usize {
        self.mounts.len()
    }

    /// Iterate over all mounts.
    pub fn mounts(&self) -> impl Iterator<Item = &Mount> {
        self.mounts.iter()
    }

    /// Check if a path is a parent prefix of any mount point.
    ///
    /// This is used to synthesize virtual directories for paths that don't
    /// directly resolve to any mount, but are ancestors of one or more mount
    /// points. For example, if a FAT is mounted at
    /// `/root/.local/lib/python3.12/site-packages/testdir`, then
    /// `/root/.local/lib/python3.12/site-packages` is a virtual parent.
    ///
    /// # Arguments
    ///
    /// * `path` - Normalized absolute path (no trailing slash unless root).
    ///
    /// # Returns
    ///
    /// `true` if at least one mount point starts with `path + "/"`.
    pub fn is_mount_parent(&self, path: &str) -> bool {
        if path == "/" {
            // Root is parent of all mounts (but usually handled elsewhere).
            return !self.mounts.is_empty();
        }

        let prefix_with_slash = alloc::format!("{}/", path);
        self.mounts
            .iter()
            .any(|m| m.path.starts_with(&prefix_with_slash))
    }

    /// List the immediate virtual children of a path that is a mount parent.
    ///
    /// For example, given mounts at:
    ///   - `/root/.local/lib/python3.12/site-packages/testdir`
    ///   - `/root/.local/lib/python3.12/site-packages/other`
    ///
    /// Calling `virtual_children("/root/.local/lib/python3.12/site-packages")`
    /// returns `["testdir", "other"]`.
    ///
    /// Calling `virtual_children("/root")` returns `[".local"]`.
    ///
    /// # Arguments
    ///
    /// * `path` - Normalized absolute path.
    ///
    /// # Returns
    ///
    /// A deduplicated list of immediate child directory names.
    pub fn virtual_children(&self, path: &str) -> Vec<String> {
        let mut children: Vec<String> = Vec::new();

        let prefix = if path == "/" {
            String::from("/")
        } else {
            alloc::format!("{}/", path)
        };

        for mount in &self.mounts {
            if let Some(rest) = mount.path.strip_prefix(prefix.as_str()) {
                // rest is the portion after the parent prefix.
                // The immediate child is the first path component.
                let child = rest.split('/').next().unwrap_or(rest);
                if !child.is_empty() {
                    let child_str = String::from(child);
                    if !children.contains(&child_str) {
                        children.push(child_str);
                    }
                }
            }
        }

        children
    }
}

impl Default for Vfs {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    #[test]
    fn test_normalize_path_absolute() {
        let vfs = Vfs::new();

        assert_eq!(vfs.normalize_path("/").unwrap(), "/");
        assert_eq!(vfs.normalize_path("/foo").unwrap(), "/foo");
        assert_eq!(vfs.normalize_path("/foo/bar").unwrap(), "/foo/bar");
        assert_eq!(vfs.normalize_path("/foo/").unwrap(), "/foo");
        assert_eq!(vfs.normalize_path("/foo//bar").unwrap(), "/foo/bar");
    }

    #[test]
    fn test_normalize_path_relative() {
        let mut vfs = Vfs::new();

        // From root
        assert_eq!(vfs.normalize_path("foo").unwrap(), "/foo");
        assert_eq!(vfs.normalize_path("foo/bar").unwrap(), "/foo/bar");

        // From subdirectory
        vfs.cwd = String::from("/data");
        assert_eq!(vfs.normalize_path("foo").unwrap(), "/data/foo");
        assert_eq!(vfs.normalize_path("foo/bar").unwrap(), "/data/foo/bar");
    }

    #[test]
    fn test_normalize_path_dots() {
        let vfs = Vfs::new();

        // Current directory
        assert_eq!(vfs.normalize_path("/foo/./bar").unwrap(), "/foo/bar");
        assert_eq!(vfs.normalize_path("/./foo").unwrap(), "/foo");

        // Parent directory
        assert_eq!(vfs.normalize_path("/foo/bar/..").unwrap(), "/foo");
        assert_eq!(vfs.normalize_path("/foo/../bar").unwrap(), "/bar");
        assert_eq!(vfs.normalize_path("/foo/bar/../baz").unwrap(), "/foo/baz");
    }

    #[test]
    fn test_normalize_path_invalid() {
        let vfs = Vfs::new();

        // Empty path
        assert!(vfs.normalize_path("").is_err());

        // Too many parent refs
        assert!(vfs.normalize_path("/..").is_err());
        assert!(vfs.normalize_path("/foo/../..").is_err());
    }

    #[test]
    fn test_mount_matches() {
        let mount = Mount::new(String::from("/data"), MountBackend::ReadOnly);

        // Exact match
        assert_eq!(mount.matches("/data"), Some(""));

        // Subpath match
        assert_eq!(mount.matches("/data/file.txt"), Some("file.txt"));
        assert_eq!(mount.matches("/data/sub/file.txt"), Some("sub/file.txt"));

        // No match
        assert_eq!(mount.matches("/other"), None);
        assert_eq!(mount.matches("/datafile"), None); // Not a proper prefix
    }

    #[test]
    fn test_mount_matches_root() {
        let mount = Mount::new(String::from("/"), MountBackend::ReadOnly);

        // Root matches everything
        assert_eq!(mount.matches("/"), Some(""));
        assert_eq!(mount.matches("/foo"), Some("foo"));
        assert_eq!(mount.matches("/foo/bar"), Some("foo/bar"));
    }

    #[test]
    fn test_vfs_add_mount_ordering() {
        let mut vfs = Vfs::new();

        // Add mounts in random order
        vfs.add_mount(Mount::new(String::from("/"), MountBackend::ReadOnly))
            .unwrap();

        vfs.add_mount(Mount::new(
            String::from("/data/cache"),
            MountBackend::ReadOnly,
        ))
        .unwrap();

        vfs.add_mount(Mount::new(String::from("/data"), MountBackend::ReadOnly))
            .unwrap();

        // Should be sorted by path length descending
        assert_eq!(vfs.mounts[0].path, "/data/cache");
        assert_eq!(vfs.mounts[1].path, "/data");
        assert_eq!(vfs.mounts[2].path, "/");
    }

    #[test]
    fn test_vfs_resolve() {
        let mut vfs = Vfs::new();

        vfs.add_mount(Mount::new(String::from("/"), MountBackend::ReadOnly))
            .unwrap();

        vfs.add_mount(Mount::new(String::from("/data"), MountBackend::ReadOnly))
            .unwrap();

        // Should resolve to longest matching mount
        let (idx, rel) = vfs.resolve("/data/file.txt").unwrap();
        assert_eq!(vfs.mounts[idx].path, "/data");
        assert_eq!(rel, "file.txt");

        let (idx, rel) = vfs.resolve("/other/file.txt").unwrap();
        assert_eq!(vfs.mounts[idx].path, "/");
        assert_eq!(rel, "other/file.txt");
    }

    #[test]
    fn test_vfs_duplicate_mount() {
        let mut vfs = Vfs::new();

        vfs.add_mount(Mount::new(String::from("/data"), MountBackend::ReadOnly))
            .unwrap();

        // Duplicate should fail
        let result = vfs.add_mount(Mount::new(String::from("/data"), MountBackend::ReadOnly));

        assert!(matches!(result, Err(FsError::AlreadyExists)));
    }

    #[test]
    fn test_tilde_expansion_default_home() {
        let vfs = Vfs::new();

        // Default home is "/"
        assert_eq!(vfs.home_dir(), "/");

        // "~" -> "/"
        assert_eq!(vfs.normalize_path("~").unwrap(), "/");

        // "~/foo" -> "/foo"
        assert_eq!(vfs.normalize_path("~/foo").unwrap(), "/foo");

        // "~/foo/bar" -> "/foo/bar"
        assert_eq!(vfs.normalize_path("~/foo/bar").unwrap(), "/foo/bar");
    }

    #[test]
    fn test_tilde_expansion_custom_home() {
        let mut vfs = Vfs::new();
        vfs.set_home_dir("/root").unwrap();

        assert_eq!(vfs.home_dir(), "/root");

        // "~" -> "/root"
        assert_eq!(vfs.normalize_path("~").unwrap(), "/root");

        // "~/foo" -> "/root/foo"
        assert_eq!(vfs.normalize_path("~/foo").unwrap(), "/root/foo");

        // "~/.local" -> "/root/.local"
        assert_eq!(vfs.normalize_path("~/.local").unwrap(), "/root/.local");
    }

    #[test]
    fn test_tilde_expansion_no_expansion_for_non_tilde() {
        let mut vfs = Vfs::new();
        vfs.set_home_dir("/root").unwrap();

        // Absolute paths should not be affected
        assert_eq!(vfs.normalize_path("/foo").unwrap(), "/foo");

        // Relative paths starting with ~ in the middle should not expand
        assert_eq!(vfs.normalize_path("/a/~/b").unwrap(), "/a/~/b");
    }

    #[test]
    fn test_set_home_dir_validation() {
        let mut vfs = Vfs::new();

        // Relative paths should fail
        assert!(vfs.set_home_dir("root").is_err());

        // Absolute paths should succeed
        assert!(vfs.set_home_dir("/root").is_ok());
        assert!(vfs.set_home_dir("/").is_ok());

        // Trailing slash should be stripped
        vfs.set_home_dir("/root/").unwrap();
        assert_eq!(vfs.home_dir(), "/root");
    }

    #[test]
    fn test_is_mount_parent() {
        let mut vfs = Vfs::new();
        vfs.add_mount(Mount::new(
            String::from("/root/.local/lib/python3.12/site-packages/testdir"),
            MountBackend::ReadOnly,
        ))
        .unwrap();

        // Direct parents of the mount path should be recognized.
        assert!(vfs.is_mount_parent("/root"));
        assert!(vfs.is_mount_parent("/root/.local"));
        assert!(vfs.is_mount_parent("/root/.local/lib"));
        assert!(vfs.is_mount_parent("/root/.local/lib/python3.12"));
        assert!(vfs.is_mount_parent("/root/.local/lib/python3.12/site-packages"));

        // The mount path itself is NOT a parent (it's the mount itself).
        assert!(!vfs.is_mount_parent(
            "/root/.local/lib/python3.12/site-packages/testdir"
        ));

        // Unrelated paths should not match.
        assert!(!vfs.is_mount_parent("/other"));
        assert!(!vfs.is_mount_parent("/root/.localx"));
    }

    #[test]
    fn test_virtual_children() {
        let mut vfs = Vfs::new();
        vfs.add_mount(Mount::new(
            String::from("/root/.local/lib/python3.12/site-packages/testdir"),
            MountBackend::ReadOnly,
        ))
        .unwrap();
        vfs.add_mount(Mount::new(
            String::from("/root/.local/lib/python3.12/site-packages/other"),
            MountBackend::ReadOnly,
        ))
        .unwrap();

        // Children of site-packages should list both mount children.
        let children =
            vfs.virtual_children("/root/.local/lib/python3.12/site-packages");
        assert_eq!(children.len(), 2);
        assert!(children.contains(&String::from("testdir")));
        assert!(children.contains(&String::from("other")));

        // Children of /root should be ".local".
        let children = vfs.virtual_children("/root");
        assert_eq!(children, vec![String::from(".local")]);

        // Children of /root/.local/lib should be "python3.12".
        let children = vfs.virtual_children("/root/.local/lib");
        assert_eq!(children, vec![String::from("python3.12")]);
    }

    #[test]
    fn test_set_cwd_to_mount_parent() {
        let mut vfs = Vfs::new();
        vfs.add_mount(Mount::new(
            String::from("/root/.local/lib/python3.12/site-packages/testdir"),
            MountBackend::ReadOnly,
        ))
        .unwrap();

        // Setting cwd to a parent of a mount should succeed.
        assert!(vfs.set_cwd("/root/.local/lib/python3.12/site-packages").is_ok());
        assert_eq!(vfs.cwd(), "/root/.local/lib/python3.12/site-packages");

        // Setting cwd to an unrelated path should fail.
        assert!(vfs.set_cwd("/nowhere").is_err());
    }
}
