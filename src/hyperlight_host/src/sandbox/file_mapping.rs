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

//! Host-side file mapping preparation for [`map_file_cow`].
//!
//! This module splits the file mapping operation into two phases:
//! - **Prepare** ([`prepare_file_cow`]): performs host-side OS calls
//!   (open file, create mapping) without requiring a VM.
//! - **Apply**: performed by the caller (either [`MultiUseSandbox::map_file_cow`]
//!   or [`evolve_impl_multi_use`]) to map the prepared region into
//!   the guest via [`HyperlightVm::map_region`].
//!
//! This separation allows [`UninitializedSandbox`] to accept
//! `map_file_cow` calls before the VM exists, deferring the VM-side
//! work until [`evolve()`].

use std::os::raw::c_void;
use std::path::Path;

use tracing::{Span, instrument};

#[cfg(target_os = "windows")]
use crate::HyperlightError;
#[cfg(target_os = "windows")]
use crate::hypervisor::wrappers::HandleWrapper;
#[cfg(target_os = "windows")]
use crate::mem::memory_region::{HostRegionBase, MemoryRegionKind};
use crate::mem::memory_region::{MemoryRegion, MemoryRegionFlags, MemoryRegionType};
use crate::{Result, log_then_return};

/// A prepared (host-side) file mapping ready to be applied to a VM.
///
/// Created by [`prepare_file_cow`]. The host-side OS resources (file
/// mapping handle + view on Windows, mmap on Linux) are held here
/// until consumed by the VM-side apply step.
///
/// If dropped without being consumed, the `Drop` impl releases all
/// host-side resources — preventing leaks when an
/// [`UninitializedSandbox`] is dropped without evolving or when
/// apply fails.
pub(crate) struct PreparedFileMapping {
    /// The guest address where this file should be mapped.
    pub(crate) guest_base: u64,
    /// The page-aligned size of the mapping in bytes.
    pub(crate) size: usize,
    /// Host-side OS resources. `None` after successful consumption
    /// by the apply step (ownership transferred to the VM layer).
    pub(crate) host_resources: Option<HostFileResources>,
}

/// Platform-specific host-side file mapping resources.
pub(crate) enum HostFileResources {
    /// Windows: `CreateFileMappingW` handle + `MapViewOfFile` view.
    #[cfg(target_os = "windows")]
    Windows {
        mapping_handle: HandleWrapper,
        view_base: *mut c_void,
    },
    /// Linux: `mmap` base pointer.
    #[cfg(unix)]
    Unix {
        mmap_base: *mut c_void,
        mmap_size: usize,
    },
}

impl Drop for PreparedFileMapping {
    fn drop(&mut self) {
        // Clean up host resources if they haven't been consumed.
        if let Some(resources) = self.host_resources.take() {
            match resources {
                #[cfg(target_os = "windows")]
                HostFileResources::Windows {
                    mapping_handle,
                    view_base,
                } => unsafe {
                    use windows::Win32::Foundation::CloseHandle;
                    use windows::Win32::System::Memory::{
                        MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile,
                    };
                    if let Err(e) = UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view_base })
                    {
                        tracing::error!(
                            "PreparedFileMapping::drop: UnmapViewOfFile failed: {:?}",
                            e
                        );
                    }
                    if let Err(e) = CloseHandle(mapping_handle.into()) {
                        tracing::error!("PreparedFileMapping::drop: CloseHandle failed: {:?}", e);
                    }
                },
                #[cfg(unix)]
                HostFileResources::Unix {
                    mmap_base,
                    mmap_size,
                } => unsafe {
                    libc::munmap(mmap_base, mmap_size);
                },
            }
        }
    }
}

// SAFETY: The host resources are kernel objects (Windows handles, mmap
// pointers) that are safe to transfer between threads. The
// cleanup APIs (UnmapViewOfFile, CloseHandle, munmap) are thread-safe.
unsafe impl Send for PreparedFileMapping {}

impl PreparedFileMapping {
    /// Build the [`MemoryRegion`] that describes this mapping for the
    /// VM layer. The host resources must still be present (not yet
    /// consumed).
    pub(crate) fn to_memory_region(&self) -> Result<MemoryRegion> {
        let resources = self.host_resources.as_ref().ok_or_else(|| {
            crate::HyperlightError::Error(
                "PreparedFileMapping resources already consumed".to_string(),
            )
        })?;

        match resources {
            #[cfg(target_os = "windows")]
            HostFileResources::Windows {
                mapping_handle,
                view_base,
            } => {
                let host_base = HostRegionBase {
                    from_handle: *mapping_handle,
                    handle_base: *view_base as usize,
                    handle_size: self.size,
                    offset: 0,
                };
                let host_end =
                    <crate::mem::memory_region::HostGuestMemoryRegion as MemoryRegionKind>::add(
                        host_base, self.size,
                    );
                Ok(MemoryRegion {
                    host_region: host_base..host_end,
                    guest_region: self.guest_base as usize..self.guest_base as usize + self.size,
                    flags: MemoryRegionFlags::READ | MemoryRegionFlags::EXECUTE,
                    region_type: MemoryRegionType::MappedFile,
                })
            }
            #[cfg(unix)]
            HostFileResources::Unix {
                mmap_base,
                mmap_size,
            } => Ok(MemoryRegion {
                host_region: *mmap_base as usize..(*mmap_base as usize).wrapping_add(*mmap_size),
                guest_region: self.guest_base as usize..self.guest_base as usize + self.size,
                flags: MemoryRegionFlags::READ | MemoryRegionFlags::EXECUTE,
                region_type: MemoryRegionType::MappedFile,
            }),
        }
    }

    /// Mark the host resources as consumed (ownership transferred to
    /// the VM layer). After this call, `Drop` will not release them.
    pub(crate) fn consume(&mut self) {
        self.host_resources = None;
    }
}

/// Perform host-side file mapping preparation without requiring a VM.
///
/// Opens the file, creates a read-only mapping in the host process,
/// and returns a [`PreparedFileMapping`] that can be applied to the
/// VM later.
///
/// # Errors
///
/// Returns an error if the file cannot be opened, is empty, or the
/// OS mapping calls fail.
#[instrument(err(Debug), skip(file_path, guest_base), parent = Span::current())]
pub(crate) fn prepare_file_cow(file_path: &Path, guest_base: u64) -> Result<PreparedFileMapping> {
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;

        use windows::Win32::Foundation::HANDLE;
        use windows::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_READ, MapViewOfFile, PAGE_READONLY,
        };

        let file = std::fs::File::options().read(true).open(file_path)?;
        let file_size = file.metadata()?.len();
        if file_size == 0 {
            log_then_return!("map_file_cow: cannot map an empty file: {:?}", file_path);
        }
        let page_size = page_size::get();
        let size = usize::try_from(file_size).map_err(|_| {
            HyperlightError::Error(format!(
                "File size {file_size} exceeds addressable range on this platform"
            ))
        })?;
        let size = size.div_ceil(page_size) * page_size;

        let file_handle = HANDLE(file.as_raw_handle());

        // Create a read-only file mapping object backed by the actual file.
        // Pass 0,0 for size to use the file's actual size — Windows will
        // NOT extend a read-only file, so requesting page-aligned size
        // would fail for files smaller than one page.
        let mapping_handle =
            unsafe { CreateFileMappingW(file_handle, None, PAGE_READONLY, 0, 0, None) }
                .map_err(|e| HyperlightError::Error(format!("CreateFileMappingW failed: {e}")))?;

        // Map a read-only view into the host process.
        // Passing 0 for dwNumberOfBytesToMap maps the entire file; the OS
        // rounds up to the next page boundary and zero-fills the remainder.
        let view = unsafe { MapViewOfFile(mapping_handle, FILE_MAP_READ, 0, 0, 0) };
        if view.Value.is_null() {
            unsafe {
                let _ = windows::Win32::Foundation::CloseHandle(mapping_handle);
            }
            log_then_return!(
                "MapViewOfFile failed: {:?}",
                std::io::Error::last_os_error()
            );
        }

        Ok(PreparedFileMapping {
            guest_base,
            size,
            host_resources: Some(HostFileResources::Windows {
                mapping_handle: HandleWrapper::from(mapping_handle),
                view_base: view.Value,
            }),
        })
    }
    #[cfg(unix)]
    {
        use std::os::fd::AsRawFd;
        use std::os::linux::fs::MetadataExt;

        let file = std::fs::File::options()
            .read(true)
            .write(true)
            .open(file_path)?;
        let file_size = file.metadata()?.st_size();
        if file_size == 0 {
            log_then_return!("map_file_cow: cannot map an empty file: {:?}", file_path);
        }
        let page_size = page_size::get();
        let size = (file_size as usize).div_ceil(page_size) * page_size;
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if base == libc::MAP_FAILED {
            log_then_return!("mmap error: {:?}", std::io::Error::last_os_error());
        }

        Ok(PreparedFileMapping {
            guest_base,
            size,
            host_resources: Some(HostFileResources::Unix {
                mmap_base: base,
                mmap_size: size,
            }),
        })
    }
}
