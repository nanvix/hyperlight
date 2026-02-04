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

//! Integration tests for HyperlightFS guest VFS.
//!
//! These tests use mocked memory to simulate the host-mapped file regions.
//!
//! **IMPORTANT**: These tests share global state (FS_STATE, FD_TABLE, TEST_FILE_DATA)
//! and must be run single-threaded:
//! ```sh
//! cargo test --package hyperlight-guest --lib fs:: -- --test-threads=1
//! ```

use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use core::cell::UnsafeCell;

use embedded_io::{Read, Seek, SeekFrom};
use hyperlight_common::flatbuffer_wrappers::hyperlight_fs::{HyperlightFSData, InodeData};

use super::*;

/// Test file data buffer wrapped for safe static access.
struct TestBuffer(UnsafeCell<[u8; 1024]>);

// SAFETY: Tests run single-threaded
unsafe impl Sync for TestBuffer {}

impl TestBuffer {
    const fn new() -> Self {
        Self(UnsafeCell::new([0u8; 1024]))
    }

    fn write(&self, content: &[u8]) {
        // SAFETY: Tests are single-threaded
        unsafe {
            (&mut (*self.0.get()))[..content.len()].copy_from_slice(content);
        }
    }

    fn as_ptr(&self) -> u64 {
        self.0.get() as u64
    }
}

static TEST_FILE_DATA: TestBuffer = TestBuffer::new();

/// Helper to reset the filesystem state between tests.
fn reset_fs() {
    // Reset the fd table
    fd::reset();
    // Reset the filesystem state so init() can be called again
    manifest::reset();
}

/// Create a test manifest with the given inodes.
fn create_manifest(inodes: Vec<InodeData>) -> Vec<u8> {
    let data = HyperlightFSData::new(inodes);
    (&data).try_into().expect("Failed to serialize manifest")
}

/// Initialize the filesystem with a test manifest.
///
/// # Safety
/// Must only be called once per test or after proper reset.
unsafe fn init_test_fs(manifest: &[u8]) {
    // SAFETY: This is test code and we ensure the manifest is valid.
    unsafe {
        manifest::init(manifest.as_ptr(), manifest.len()).expect("Failed to init fs");
    }
}

#[test]
fn test_init_and_is_initialized() {
    reset_fs();

    // Create minimal manifest with just root
    let manifest = create_manifest(vec![InodeData::directory("/".to_string(), 0)]);

    unsafe {
        init_test_fs(&manifest);
    }

    assert!(is_initialized());
}

#[test]
fn test_stat_directory() {
    reset_fs();

    let manifest = create_manifest(vec![
        InodeData::directory("/".to_string(), 0),
        InodeData::directory("/data".to_string(), 0),
    ]);

    unsafe {
        init_test_fs(&manifest);
    }

    let st = stat("/").expect("stat / failed");
    assert!(st.is_dir);
    assert_eq!(st.size, 0);

    let st = stat("/data").expect("stat /data failed");
    assert!(st.is_dir);
}

#[test]
fn test_stat_file() {
    reset_fs();

    // Set up test file data
    let file_content = b"Hello, HyperlightFS!";
    TEST_FILE_DATA.write(file_content);

    let guest_address = TEST_FILE_DATA.as_ptr();

    let manifest = create_manifest(vec![
        InodeData::directory("/".to_string(), 0),
        InodeData::file(
            "/test.txt".to_string(),
            0,
            guest_address,
            file_content.len() as u64,
        ),
    ]);

    unsafe {
        init_test_fs(&manifest);
    }

    let st = stat("/test.txt").expect("stat /test.txt failed");
    assert!(!st.is_dir);
    assert_eq!(st.size, file_content.len() as u64);
}

#[test]
fn test_stat_not_found() {
    reset_fs();

    let manifest = create_manifest(vec![InodeData::directory("/".to_string(), 0)]);

    unsafe {
        init_test_fs(&manifest);
    }

    let result = stat("/nonexistent");
    assert_eq!(result, Err(FsError::NotFound));
}

#[test]
fn test_open_and_read() {
    reset_fs();

    let file_content = b"Test file content for reading";
    TEST_FILE_DATA.write(file_content);

    let guest_address = TEST_FILE_DATA.as_ptr();

    let manifest = create_manifest(vec![
        InodeData::directory("/".to_string(), 0),
        InodeData::file(
            "/readme.txt".to_string(),
            0,
            guest_address,
            file_content.len() as u64,
        ),
    ]);

    unsafe {
        init_test_fs(&manifest);
    }

    // Open the file
    let mut file = open("/readme.txt").expect("open failed");

    // Check initial state
    assert_eq!(file.size().unwrap(), file_content.len() as u64);
    assert_eq!(file.position().unwrap(), 0);
    assert_eq!(file.remaining().unwrap(), file_content.len() as u64);

    // Read entire file
    let mut buf = [0u8; 64];
    let n = file.read(&mut buf).expect("read failed");
    assert_eq!(n, file_content.len());
    assert_eq!(&buf[..n], file_content);

    // Position should have advanced
    assert_eq!(file.position().unwrap(), file_content.len() as u64);
    assert_eq!(file.remaining().unwrap(), 0);

    // Reading at EOF returns 0
    let n = file.read(&mut buf).expect("read at eof failed");
    assert_eq!(n, 0);
}

#[test]
fn test_read_partial() {
    reset_fs();

    let file_content = b"ABCDEFGHIJ";
    TEST_FILE_DATA.write(file_content);

    let guest_address = TEST_FILE_DATA.as_ptr();

    let manifest = create_manifest(vec![
        InodeData::directory("/".to_string(), 0),
        InodeData::file(
            "/letters.txt".to_string(),
            0,
            guest_address,
            file_content.len() as u64,
        ),
    ]);

    unsafe {
        init_test_fs(&manifest);
    }

    let mut file = open("/letters.txt").expect("open failed");

    // Read 3 bytes
    let mut buf = [0u8; 3];
    let n = file.read(&mut buf).expect("read failed");
    assert_eq!(n, 3);
    assert_eq!(&buf, b"ABC");

    // Read 3 more bytes
    let n = file.read(&mut buf).expect("read failed");
    assert_eq!(n, 3);
    assert_eq!(&buf, b"DEF");

    // Position should be 6
    assert_eq!(file.position().unwrap(), 6);
}

#[test]
fn test_seek() {
    reset_fs();

    let file_content = b"0123456789";
    TEST_FILE_DATA.write(file_content);

    let guest_address = TEST_FILE_DATA.as_ptr();

    let manifest = create_manifest(vec![
        InodeData::directory("/".to_string(), 0),
        InodeData::file(
            "/digits.txt".to_string(),
            0,
            guest_address,
            file_content.len() as u64,
        ),
    ]);

    unsafe {
        init_test_fs(&manifest);
    }

    let mut file = open("/digits.txt").expect("open failed");

    // Seek to position 5
    let pos = file.seek(SeekFrom::Start(5)).expect("seek failed");
    assert_eq!(pos, 5);

    // Read from position 5
    let mut buf = [0u8; 3];
    let n = file.read(&mut buf).expect("read failed");
    assert_eq!(n, 3);
    assert_eq!(&buf, b"567");

    // Seek relative
    let pos = file.seek(SeekFrom::Current(-3)).expect("seek failed");
    assert_eq!(pos, 5);

    // Seek from end
    let pos = file.seek(SeekFrom::End(-2)).expect("seek failed");
    assert_eq!(pos, 8);

    let n = file.read(&mut buf).expect("read failed");
    assert_eq!(n, 2);
    assert_eq!(&buf[..2], b"89");

    // Rewind
    file.rewind().expect("rewind failed");
    assert_eq!(file.position().unwrap(), 0);
}

#[test]
fn test_seek_invalid() {
    reset_fs();

    let file_content = b"test";
    TEST_FILE_DATA.write(file_content);

    let guest_address = TEST_FILE_DATA.as_ptr();

    let manifest = create_manifest(vec![
        InodeData::directory("/".to_string(), 0),
        InodeData::file(
            "/small.txt".to_string(),
            0,
            guest_address,
            file_content.len() as u64,
        ),
    ]);

    unsafe {
        init_test_fs(&manifest);
    }

    let mut file = open("/small.txt").expect("open failed");

    // Seek to negative position should fail
    let result = file.seek(SeekFrom::Current(-100));
    assert_eq!(result, Err(FsError::InvalidSeek));
}

#[test]
fn test_open_directory_fails() {
    reset_fs();

    let manifest = create_manifest(vec![
        InodeData::directory("/".to_string(), 0),
        InodeData::directory("/mydir".to_string(), 0),
    ]);

    unsafe {
        init_test_fs(&manifest);
    }

    let result = open("/mydir");
    assert!(matches!(result, Err(FsError::NotAFile)));
}

#[test]
fn test_open_not_found() {
    reset_fs();

    let manifest = create_manifest(vec![InodeData::directory("/".to_string(), 0)]);

    unsafe {
        init_test_fs(&manifest);
    }

    let result = open("/missing.txt");
    assert!(matches!(result, Err(FsError::NotFound)));
}

#[test]
fn test_read_dir() {
    reset_fs();

    let file_content = b"x";
    TEST_FILE_DATA.write(file_content);

    let guest_address = TEST_FILE_DATA.as_ptr();

    let manifest = create_manifest(vec![
        InodeData::directory("/".to_string(), 0),
        InodeData::directory("/subdir".to_string(), 0),
        InodeData::file("/file1.txt".to_string(), 0, guest_address, 1),
        InodeData::file("/file2.txt".to_string(), 0, guest_address, 1),
    ]);

    unsafe {
        init_test_fs(&manifest);
    }

    let entries = read_dir("/").expect("read_dir failed");

    // Should have 3 entries: subdir, file1.txt, file2.txt
    assert_eq!(entries.len(), 3);

    let names: Vec<_> = entries.iter().map(|e| e.name.as_str()).collect();
    assert!(names.contains(&"subdir"));
    assert!(names.contains(&"file1.txt"));
    assert!(names.contains(&"file2.txt"));

    // Check that subdir is marked as directory
    let subdir = entries.iter().find(|e| e.name == "subdir").unwrap();
    assert!(subdir.is_dir);
}

#[test]
fn test_read_dir_not_a_directory() {
    reset_fs();

    let file_content = b"x";
    TEST_FILE_DATA.write(file_content);

    let guest_address = TEST_FILE_DATA.as_ptr();

    let manifest = create_manifest(vec![
        InodeData::directory("/".to_string(), 0),
        InodeData::file("/file.txt".to_string(), 0, guest_address, 1),
    ]);

    unsafe {
        init_test_fs(&manifest);
    }

    let result = read_dir("/file.txt");
    assert_eq!(result, Err(FsError::NotADirectory));
}

#[test]
fn test_file_drop_closes_fd() {
    reset_fs();

    let file_content = b"test";
    TEST_FILE_DATA.write(file_content);

    let guest_address = TEST_FILE_DATA.as_ptr();

    let manifest = create_manifest(vec![
        InodeData::directory("/".to_string(), 0),
        InodeData::file(
            "/test.txt".to_string(),
            0,
            guest_address,
            file_content.len() as u64,
        ),
    ]);

    unsafe {
        init_test_fs(&manifest);
    }

    // Open file in a scope
    {
        let _file = open("/test.txt").expect("open failed");
        assert_eq!(fd::open_count(), 1);
    }

    // File should be closed after drop
    assert_eq!(fd::open_count(), 0);
}

#[test]
fn test_read_to_vec() {
    reset_fs();

    let file_content = b"Complete file contents";
    TEST_FILE_DATA.write(file_content);

    let guest_address = TEST_FILE_DATA.as_ptr();

    let manifest = create_manifest(vec![
        InodeData::directory("/".to_string(), 0),
        InodeData::file(
            "/full.txt".to_string(),
            0,
            guest_address,
            file_content.len() as u64,
        ),
    ]);

    unsafe {
        init_test_fs(&manifest);
    }

    let mut file = open("/full.txt").expect("open failed");

    // Read partway first
    let mut buf = [0u8; 5];
    file.read(&mut buf).unwrap();

    // read_to_vec should rewind and read everything
    let contents = file.read_to_vec().expect("read_to_vec failed");
    assert_eq!(contents.as_slice(), file_content);
}

// ============================================================================
// File Descriptor Ownership API Tests
// ============================================================================

#[test]
fn test_rofile_into_raw_fd_skips_drop() {
    reset_fs();

    let file_content = b"test data";
    TEST_FILE_DATA.write(file_content);

    let guest_address = TEST_FILE_DATA.as_ptr();

    let manifest = create_manifest(vec![
        InodeData::directory("/".to_string(), 0),
        InodeData::file(
            "/test.txt".to_string(),
            0,
            guest_address,
            file_content.len() as u64,
        ),
    ]);

    unsafe {
        init_test_fs(&manifest);
    }

    // Open file and extract fd
    let raw_fd = {
        let file = open("/test.txt").expect("open failed");
        assert_eq!(fd::open_count(), 1);

        // Extract RoFile and call into_raw_fd
        match file {
            File::ReadOnly(ro) => ro.into_raw_fd(),
            _ => panic!("Expected ReadOnly file"),
        }
    };

    // Drop happened but into_raw_fd skipped closing - fd should still be valid
    assert_eq!(fd::open_count(), 1);

    // Manual cleanup
    let _ = fd::free_fd(raw_fd);
    assert_eq!(fd::open_count(), 0);
}

#[test]
fn test_rofile_from_raw_fd_owns_fd() {
    reset_fs();

    let file_content = b"test data";
    TEST_FILE_DATA.write(file_content);

    let guest_address = TEST_FILE_DATA.as_ptr();

    let manifest = create_manifest(vec![
        InodeData::directory("/".to_string(), 0),
        InodeData::file(
            "/test.txt".to_string(),
            0,
            guest_address,
            file_content.len() as u64,
        ),
    ]);

    unsafe {
        init_test_fs(&manifest);
    }

    // Open file, transfer out, then transfer back in
    let raw_fd = {
        let file = open("/test.txt").expect("open failed");
        match file {
            File::ReadOnly(ro) => ro.into_raw_fd(),
            _ => panic!("Expected ReadOnly file"),
        }
    };

    assert_eq!(fd::open_count(), 1);

    // Reconstruct RoFile from raw fd
    {
        // SAFETY: We know this is a valid RO fd we just extracted
        let _ro = unsafe { RoFile::from_raw_fd(raw_fd) };
        assert_eq!(fd::open_count(), 1);
        // RoFile drops here
    }

    // from_raw_fd took ownership, Drop closed it
    assert_eq!(fd::open_count(), 0);
}

#[test]
fn test_file_into_raw_fd_preserves_fd() {
    reset_fs();

    let file_content = b"test data";
    TEST_FILE_DATA.write(file_content);

    let guest_address = TEST_FILE_DATA.as_ptr();

    let manifest = create_manifest(vec![
        InodeData::directory("/".to_string(), 0),
        InodeData::file(
            "/test.txt".to_string(),
            0,
            guest_address,
            file_content.len() as u64,
        ),
    ]);

    unsafe {
        init_test_fs(&manifest);
    }

    let raw_fd = {
        let file = open("/test.txt").expect("open failed");
        // Use File::into_raw_fd (not extracting inner type)
        file.into_raw_fd()
    };

    // Fd not closed by drop
    assert_eq!(fd::open_count(), 1);

    // Clean up
    let _ = fd::free_fd(raw_fd);
    assert_eq!(fd::open_count(), 0);
}

#[test]
fn test_file_fd_returns_correct_value() {
    reset_fs();

    let file_content = b"test data";
    TEST_FILE_DATA.write(file_content);

    let guest_address = TEST_FILE_DATA.as_ptr();

    let manifest = create_manifest(vec![
        InodeData::directory("/".to_string(), 0),
        InodeData::file(
            "/test.txt".to_string(),
            0,
            guest_address,
            file_content.len() as u64,
        ),
    ]);

    unsafe {
        init_test_fs(&manifest);
    }

    let file = open("/test.txt").expect("open failed");

    // fd() should return a valid non-negative fd
    let fd = file.fd();
    assert!(fd >= 0, "fd should be non-negative");

    // Opening another file should give a different fd
    let file2 = open("/test.txt").expect("second open failed");
    let fd2 = file2.fd();
    assert_ne!(fd, fd2, "Different opens should give different fds");
}

// ==================== Relative Path Tests ====================

#[test]
fn test_open_relative_path_not_found_returns_not_found() {
    // Regression test: opening a relative path for a non-existent file should
    // return NotFound, not InvalidPath. The VFS should normalize the relative
    // path to an absolute path before looking up the file.
    reset_fs();

    let manifest = create_manifest(vec![InodeData::directory("/".to_string(), 0)]);

    unsafe {
        init_test_fs(&manifest);
    }

    // Opening a relative path that doesn't exist should return NotFound
    let result = open("nonexistent.txt");
    match result {
        Err(FsError::NotFound) => (), // Expected
        Err(e) => panic!(
            "Opening non-existent relative path should return NotFound, got {:?}",
            e
        ),
        Ok(_) => panic!("Opening non-existent file should fail"),
    }
}

#[test]
fn test_stat_relative_path_not_found_returns_not_found() {
    // Regression test: stat on a relative path for a non-existent file should
    // return NotFound, not InvalidPath.
    reset_fs();

    let manifest = create_manifest(vec![InodeData::directory("/".to_string(), 0)]);

    unsafe {
        init_test_fs(&manifest);
    }

    // stat on a relative path that doesn't exist should return NotFound
    let result = stat("pyvenv.cfg");
    assert_eq!(
        result,
        Err(FsError::NotFound),
        "stat on non-existent relative path should return NotFound, not InvalidPath"
    );
}

#[test]
fn test_open_relative_path_finds_file() {
    // Test that opening a file with a relative path works when the file exists.
    reset_fs();

    let file_content = b"Hello from relative path!";
    TEST_FILE_DATA.write(file_content);

    let guest_address = TEST_FILE_DATA.as_ptr();

    let manifest = create_manifest(vec![
        InodeData::directory("/".to_string(), 0),
        InodeData::file(
            "/config.txt".to_string(),
            0,
            guest_address,
            file_content.len() as u64,
        ),
    ]);

    unsafe {
        init_test_fs(&manifest);
    }

    // CWD is "/" by default, so "config.txt" should resolve to "/config.txt"
    let mut file = open("config.txt").expect("open with relative path should succeed");

    // Verify we can read the file
    let mut buf = [0u8; 64];
    let n = file.read(&mut buf).expect("read failed");
    assert_eq!(n, file_content.len());
    assert_eq!(&buf[..n], file_content);
}

#[test]
fn test_stat_relative_path_finds_file() {
    // Test that stat with a relative path works when the file exists.
    reset_fs();

    let file_content = b"Stats test";
    TEST_FILE_DATA.write(file_content);

    let guest_address = TEST_FILE_DATA.as_ptr();

    let manifest = create_manifest(vec![
        InodeData::directory("/".to_string(), 0),
        InodeData::file(
            "/myfile.txt".to_string(),
            0,
            guest_address,
            file_content.len() as u64,
        ),
    ]);

    unsafe {
        init_test_fs(&manifest);
    }

    // CWD is "/" by default, so "myfile.txt" should resolve to "/myfile.txt"
    let st = stat("myfile.txt").expect("stat with relative path should succeed");
    assert!(!st.is_dir);
    assert_eq!(st.size, file_content.len() as u64);
}

#[test]
fn test_relative_path_with_nested_directory_structure() {
    // Test that relative paths work correctly with a nested directory structure
    // that mimics a FAT mount at /lib/python with a ReadOnly mount at /.
    // This validates that:
    // 1. Relative paths from "/" resolve to files in the root ReadOnly mount
    // 2. Files in /lib/python are accessible via their full path
    reset_fs();

    let root_file_content = b"root config";
    let python_file_content = b"python config";
    TEST_FILE_DATA.write(root_file_content);

    let guest_address = TEST_FILE_DATA.as_ptr();

    // Create a manifest with:
    // - / (root directory)
    // - /pyvenv.cfg (file at root - like Python looking for venv config)
    // - /lib (directory)
    // - /lib/python (directory - would be FAT mount in real scenario)
    // - /lib/python/config.py (file in nested directory)
    let manifest = create_manifest(vec![
        InodeData::directory("/".to_string(), 0),
        InodeData::file(
            "/pyvenv.cfg".to_string(),
            0,
            guest_address,
            root_file_content.len() as u64,
        ),
        InodeData::directory("/lib".to_string(), 0),
        InodeData::directory("/lib/python".to_string(), 0),
        InodeData::file(
            "/lib/python/config.py".to_string(),
            0,
            guest_address,
            python_file_content.len() as u64,
        ),
    ]);

    unsafe {
        init_test_fs(&manifest);
    }

    // Test 1: Relative path from root should find root file
    let st = stat("pyvenv.cfg").expect("stat pyvenv.cfg from root should succeed");
    assert!(!st.is_dir);

    // Test 2: Absolute path to nested file should work
    let st = stat("/lib/python/config.py").expect("stat absolute nested path should succeed");
    assert!(!st.is_dir);

    // Test 3: Relative path to non-existent file should return NotFound, not InvalidPath
    let result = stat("nonexistent.cfg");
    assert_eq!(
        result,
        Err(FsError::NotFound),
        "Non-existent relative path should return NotFound"
    );
}

#[test]
fn test_relative_path_not_found_with_lib_python_structure() {
    // Regression test: When /lib/python exists (like a FAT mount would be),
    // a relative path lookup at root for a non-existent file should return
    // NotFound, not InvalidPath.
    //
    // This simulates the Python startup scenario where:
    // - FAT is mounted at /lib/python with the stdlib
    // - Python opens "pyvenv.cfg" (relative path) from root
    // - The file doesn't exist, so it should get NotFound (not InvalidPath)
    reset_fs();

    let file_content = b"some file";
    TEST_FILE_DATA.write(file_content);

    let guest_address = TEST_FILE_DATA.as_ptr();

    // Create manifest with /lib/python structure but no pyvenv.cfg at root
    let manifest = create_manifest(vec![
        InodeData::directory("/".to_string(), 0),
        InodeData::directory("/lib".to_string(), 0),
        InodeData::directory("/lib/python".to_string(), 0),
        InodeData::file(
            "/lib/python/os.py".to_string(),
            0,
            guest_address,
            file_content.len() as u64,
        ),
    ]);

    unsafe {
        init_test_fs(&manifest);
    }

    // Relative path "pyvenv.cfg" from root should normalize to "/pyvenv.cfg"
    // and return NotFound (not InvalidPath)
    let result = stat("pyvenv.cfg");
    assert_eq!(
        result,
        Err(FsError::NotFound),
        "stat('pyvenv.cfg') should return NotFound, not InvalidPath"
    );

    let result = open("pyvenv.cfg");
    match result {
        Err(FsError::NotFound) => (), // Expected
        Err(e) => panic!(
            "open('pyvenv.cfg') should return NotFound, got {:?}",
            e
        ),
        Ok(_) => panic!("open('pyvenv.cfg') should fail for non-existent file"),
    }

    // But absolute path to existing file should work
    let st = stat("/lib/python/os.py").expect("stat /lib/python/os.py should succeed");
    assert!(!st.is_dir);
}
