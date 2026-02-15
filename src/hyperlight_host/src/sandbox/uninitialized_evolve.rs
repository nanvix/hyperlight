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

use std::path::Path;
#[cfg(gdb)]
use std::sync::{Arc, Mutex};

use rand::Rng;
use tracing::{Span, instrument};

use super::SandboxConfiguration;
#[cfg(any(crashdump, gdb))]
use super::uninitialized::SandboxRuntimeConfig;
use crate::hypervisor::hyperlight_vm::HyperlightVm;
use crate::mem::exe::LoadInfo;
use crate::mem::memory_region::MemoryRegionFlags;
use crate::mem::mgr::SandboxMemoryManager;
use crate::mem::ptr::{GuestPtr, RawPtr};
use crate::mem::ptr_offset::Offset;
use crate::mem::shared_mem::GuestSharedMemory;
#[cfg(gdb)]
use crate::sandbox::config::DebugInfo;
#[cfg(feature = "mem_profile")]
use crate::sandbox::trace::MemTraceInfo;
#[cfg(target_os = "linux")]
use crate::signal_handlers::setup_signal_handlers;
use crate::{MultiUseSandbox, Result, UninitializedSandbox, new_error};

#[instrument(err(Debug), skip_all, parent = Span::current(), level = "Trace")]
pub(super) fn evolve_impl_multi_use(u_sbox: UninitializedSandbox) -> Result<MultiUseSandbox> {
    let t_evolve_total = std::time::Instant::now();

    let t0 = std::time::Instant::now();
    let (mut hshm, mut gshm) = u_sbox.mgr.build();
    crate::timing!("[TIMING] evolve: mgr.build(): {:?}", t0.elapsed());

    let t1 = std::time::Instant::now();
    let mut vm = set_up_hypervisor_partition(
        &mut gshm,
        &u_sbox.config,
        #[cfg(any(crashdump, gdb))]
        &u_sbox.rt_cfg,
        u_sbox.load_info,
    )?;
    crate::timing!("[TIMING] evolve: set_up_hypervisor_partition(): {:?}", t1.elapsed());

    // Wire up HyperlightFS BEFORE vm.initialise() so the guest entrypoint
    // can read the manifest from the PEB during initialization
    let t2 = std::time::Instant::now();
    if let Some(fs_image) = &u_sbox.hyperlight_fs {
        wire_hyperlight_fs(&mut vm, &gshm.layout, &mut hshm.shared_mem, fs_image)?;
    }
    crate::timing!("[TIMING] evolve: wire_hyperlight_fs(): {:?}", t2.elapsed());

    let seed = {
        let mut rng = rand::rng();
        rng.random::<u64>()
    };
    let peb_addr = {
        let peb_u64 = u64::try_from(gshm.layout.peb_address)?;
        RawPtr::from(peb_u64)
    };

    let page_size = u32::try_from(page_size::get())?;

    #[cfg(gdb)]
    let dbg_mem_access_hdl = Arc::new(Mutex::new(hshm.clone()));

    #[cfg(target_os = "linux")]
    setup_signal_handlers(&u_sbox.config)?;

    let t3 = std::time::Instant::now();
    vm.initialise(
        peb_addr,
        seed,
        page_size,
        &mut hshm,
        &u_sbox.host_funcs,
        u_sbox.max_guest_log_level,
        #[cfg(gdb)]
        dbg_mem_access_hdl,
    )?;
    crate::timing!("[TIMING] evolve: vm.initialise(): {:?}", t3.elapsed());

    crate::timing!("[TIMING] evolve total: {:?}", t_evolve_total.elapsed());

    let dispatch_function_addr = hshm.get_pointer_to_dispatch_function()?;
    if dispatch_function_addr == 0 {
        return Err(new_error!("Dispatch function address is null"));
    }

    let dispatch_ptr = RawPtr::from(dispatch_function_addr);

    #[cfg(gdb)]
    let dbg_mem_wrapper = Arc::new(Mutex::new(hshm.clone()));

    let sandbox = MultiUseSandbox::from_uninit(
        u_sbox.host_funcs,
        hshm,
        vm,
        dispatch_ptr,
        #[cfg(gdb)]
        dbg_mem_wrapper,
        u_sbox.hyperlight_fs,
    );

    Ok(sandbox)
}

/// Wire up HyperlightFS by mapping files into guest memory and updating the PEB.
///
/// This function:
/// 1. Computes the manifest address (page-aligned, after contiguous memory)
/// 2. Allocates and maps the manifest into guest memory
/// 3. Computes the files region address (after manifest)
/// 4. Maps each file using `map_file_cow` with READ-only permissions
/// 5. Writes pointers and sizes of `guest_fs_manifest` and `guest_fs_region` to the PEB
///
/// # Memory Layout (after contiguous shared memory)
///
/// ```text
/// [manifest]  at fs_manifest_addr (page-aligned)
/// [file1]     at fs_files_addr = fs_manifest_addr + page_align(manifest_size)
/// [file2]     ...
/// ```
///
/// # Security
///
/// Files and manifest are mapped READ-only (no EXECUTE)
///
/// # Important
///
/// This function must be called BEFORE `vm.initialise()` so that the guest
/// entrypoint can read the manifest from the PEB during initialization.
#[cfg(unix)]
#[instrument(err(Debug), skip_all, parent = Span::current(), level = "Trace")]
fn wire_hyperlight_fs(
    vm: &mut HyperlightVm,
    layout: &crate::mem::layout::SandboxMemoryLayout,
    shared_mem: &mut crate::mem::shared_mem::HostSharedMemory,
    fs_image: &crate::hyperlight_fs::HyperlightFSImage,
) -> Result<()> {
    use tracing::info;

    use crate::mem::layout::SandboxMemoryLayout;
    use crate::mem::memory_region::{MemoryRegion, MemoryRegionType};

    let page_size = page_size::get() as u64;

    // Compute the manifest base address (page-aligned, after contiguous memory)
    let contiguous_mem_size = layout.get_memory_size()?;
    let fs_manifest_addr =
        ((SandboxMemoryLayout::BASE_ADDRESS + contiguous_mem_size) as u64 + page_size - 1)
            & !(page_size - 1);

    // Estimate manifest size to compute where files will go
    let manifest_size_estimate = fs_image.estimate_manifest_size() as u64;
    let fs_files_addr = fs_manifest_addr + manifest_size_estimate;

    info!(
        fs_manifest_addr = format_args!("{:#x}", fs_manifest_addr),
        manifest_size_estimate,
        fs_files_addr = format_args!("{:#x}", fs_files_addr),
        file_count = fs_image.file_mappings().len(),
        total_files_size = fs_image.mapped_files_region_size(),
        "Wiring HyperlightFS into sandbox"
    );

    // Generate the manifest with the computed files base address
    let manifest_data = fs_image.generate_manifest(fs_files_addr)?;
    let manifest_len = manifest_data.len();

    // Verify our estimate was sufficient (manifest fits in estimated space)
    let manifest_len_aligned = ((manifest_len as u64 + page_size - 1) & !(page_size - 1)) as usize;
    if manifest_len_aligned > manifest_size_estimate as usize {
        return Err(crate::HyperlightError::Error(format!(
            "Manifest size {} exceeds estimate {}",
            manifest_len_aligned, manifest_size_estimate
        )));
    }

    // Allocate host memory for manifest and copy data into it
    let manifest_host_ptr = unsafe {
        let ptr = libc::mmap(
            std::ptr::null_mut(),
            manifest_len_aligned,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        );
        if ptr == libc::MAP_FAILED {
            return Err(crate::HyperlightError::Error(format!(
                "Failed to allocate manifest memory: {:?}",
                std::io::Error::last_os_error()
            )));
        }
        // Copy manifest data
        std::ptr::copy_nonoverlapping(manifest_data.as_ptr(), ptr as *mut u8, manifest_len);
        // Make read-only
        if libc::mprotect(ptr, manifest_len_aligned, libc::PROT_READ) != 0 {
            libc::munmap(ptr, manifest_len_aligned);
            return Err(crate::HyperlightError::Error(format!(
                "Failed to set manifest memory read-only: {:?}",
                std::io::Error::last_os_error()
            )));
        }
        ptr
    };

    // Map manifest into guest memory
    if let Err(err) = unsafe {
        vm.map_region(&MemoryRegion {
            host_region: manifest_host_ptr as usize
                ..manifest_host_ptr as usize + manifest_len_aligned,
            guest_region: fs_manifest_addr as usize
                ..fs_manifest_addr as usize + manifest_len_aligned,
            flags: MemoryRegionFlags::READ,
            region_type: MemoryRegionType::HyperlightFS,
        })
    } {
        unsafe { libc::munmap(manifest_host_ptr, manifest_len_aligned) };
        return Err(err);
    }

    info!(
        guest_addr = format_args!("{:#x}", fs_manifest_addr),
        size = manifest_len,
        "Mapped manifest into guest"
    );

    // Map each file into guest memory with READ-only permissions
    let mut current_addr = fs_files_addr;
    for mapping in fs_image.file_mappings() {
        let mapped_size = map_file_cow_to_vm(
            vm,
            mapping.host_path(),
            current_addr,
            MemoryRegionFlags::READ,
        )?;

        info!(
            guest_path = %mapping.guest_path(),
            guest_addr = format_args!("{:#x}", current_addr),
            size = mapped_size,
            "Mapped file into guest"
        );

        current_addr += mapped_size;
    }

    // Map FAT mounts into guest memory with READ|WRITE permissions.
    // Unlike read-only file mappings above, FAT mounts are writable filesystems -
    // the guest can create, modify, and delete files within them, and changes
    // persist to the backing storage (file or anonymous memory).
    for fat_mount in fs_image.fat_mounts() {
        let fat_ptr = fat_mount.image().as_ptr();
        let fat_size = fat_mount.image().size();
        let page_size = page_size::get();
        let fat_size_aligned = (fat_size + page_size - 1) & !(page_size - 1);

        unsafe {
            vm.map_region(&MemoryRegion {
                host_region: fat_ptr as usize..fat_ptr as usize + fat_size_aligned,
                guest_region: current_addr as usize..current_addr as usize + fat_size_aligned,
                flags: MemoryRegionFlags::READ | MemoryRegionFlags::WRITE,
                region_type: MemoryRegionType::HyperlightFS,
            })?;
        }

        info!(
            mount_point = %fat_mount.mount_point(),
            guest_addr = format_args!("{:#x}", current_addr),
            size = fat_size,
            "Mapped FAT mount into guest (READ|WRITE)"
        );

        current_addr += fat_size_aligned as u64;
    }

    // Calculate the total files region size (RO + FAT)
    let fs_files_region_size = current_addr - fs_files_addr;

    // Write manifest location to PEB
    layout.set_guest_fs_manifest(shared_mem, fs_manifest_addr, manifest_len as u64)?;

    // Write files region location to PEB
    layout.set_guest_fs_region(shared_mem, fs_files_addr, fs_files_region_size)?;

    info!(
        manifest_addr = format_args!("{:#x}", fs_manifest_addr),
        manifest_size = manifest_len,
        files_addr = format_args!("{:#x}", fs_files_addr),
        files_size = fs_files_region_size,
        "HyperlightFS wired into sandbox PEB"
    );

    Ok(())
}

/// Helper function to map a file into guest memory via the VM directly.
/// Used by wire_hyperlight_fs before the MultiUseSandbox is created.
#[cfg(unix)]
fn map_file_cow_to_vm(
    vm: &mut HyperlightVm,
    fp: &Path,
    guest_base: u64,
    flags: MemoryRegionFlags,
) -> Result<u64> {
    use std::os::unix::fs::MetadataExt;
    use std::os::unix::io::AsRawFd;

    use crate::mem::memory_region::{MemoryRegion, MemoryRegionType};

    unsafe {
        // Determine host mmap protection based on guest flags
        let mut prot = libc::PROT_READ;
        if flags.contains(MemoryRegionFlags::WRITE) {
            prot |= libc::PROT_WRITE;
        }
        if flags.contains(MemoryRegionFlags::EXECUTE) {
            prot |= libc::PROT_EXEC;
        }

        let file = std::fs::File::options().read(true).open(fp)?;
        let file_size = file.metadata()?.size();
        let page_size = page_size::get();
        let size = (file_size as usize).div_ceil(page_size) * page_size;
        let base = libc::mmap(
            std::ptr::null_mut(),
            size,
            prot,
            libc::MAP_PRIVATE,
            file.as_raw_fd(),
            0,
        );
        if base == libc::MAP_FAILED {
            return Err(crate::HyperlightError::Error(format!(
                "mmap error: {:?}",
                std::io::Error::last_os_error()
            )));
        }

        if let Err(err) = vm.map_region(&MemoryRegion {
            host_region: base as usize..base.wrapping_add(size) as usize,
            guest_region: guest_base as usize..guest_base as usize + size,
            flags,
            region_type: MemoryRegionType::HyperlightFS,
        }) {
            libc::munmap(base, size);
            return Err(err);
        };

        Ok(size as u64)
    }
}

/// Windows implementation of wire_hyperlight_fs.
///
/// Uses VirtualAlloc for manifest memory and CreateFileMappingW/MapViewOfFile
/// for mapping files into the host address space, then maps everything into
/// the guest via vm.map_region (WHP).
#[cfg(windows)]
#[instrument(err(Debug), skip_all, parent = Span::current(), level = "Trace")]
fn wire_hyperlight_fs(
    vm: &mut HyperlightVm,
    layout: &crate::mem::layout::SandboxMemoryLayout,
    shared_mem: &mut crate::mem::shared_mem::HostSharedMemory,
    fs_image: &crate::hyperlight_fs::HyperlightFSImage,
) -> Result<()> {
    use tracing::info;
    use windows::Win32::System::Memory::*;

    use crate::mem::layout::SandboxMemoryLayout;
    use crate::mem::memory_region::{MemoryRegion, MemoryRegionType};

    let page_size = page_size::get() as u64;

    // Compute the manifest base address (page-aligned, after contiguous memory)
    let contiguous_mem_size = layout.get_memory_size()?;
    let fs_manifest_addr =
        ((SandboxMemoryLayout::BASE_ADDRESS + contiguous_mem_size) as u64 + page_size - 1)
            & !(page_size - 1);

    // Estimate manifest size to compute where files will go
    let manifest_size_estimate = fs_image.estimate_manifest_size() as u64;
    let fs_files_addr = fs_manifest_addr + manifest_size_estimate;

    info!(
        fs_manifest_addr = format_args!("{:#x}", fs_manifest_addr),
        manifest_size_estimate,
        fs_files_addr = format_args!("{:#x}", fs_files_addr),
        file_count = fs_image.file_mappings().len(),
        total_files_size = fs_image.mapped_files_region_size(),
        "Wiring HyperlightFS into sandbox"
    );

    // Generate the manifest with the computed files base address
    let manifest_data = fs_image.generate_manifest(fs_files_addr)?;
    let manifest_len = manifest_data.len();

    // Verify our estimate was sufficient (manifest fits in estimated space)
    let manifest_len_aligned = ((manifest_len as u64 + page_size - 1) & !(page_size - 1)) as usize;
    if manifest_len_aligned > manifest_size_estimate as usize {
        return Err(crate::HyperlightError::Error(format!(
            "Manifest size {} exceeds estimate {}",
            manifest_len_aligned, manifest_size_estimate
        )));
    }

    // Allocate host memory for manifest using VirtualAlloc
    let manifest_host_ptr = unsafe {
        let ptr = VirtualAlloc(
            None,
            manifest_len_aligned,
            MEM_COMMIT | MEM_RESERVE,
            PAGE_READWRITE,
        );
        if ptr.is_null() {
            return Err(crate::HyperlightError::Error(format!(
                "Failed to allocate manifest memory: {:?}",
                std::io::Error::last_os_error()
            )));
        }
        // Copy manifest data
        std::ptr::copy_nonoverlapping(manifest_data.as_ptr(), ptr as *mut u8, manifest_len);
        // Make read-only
        let mut old_protect = PAGE_PROTECTION_FLAGS(0);
        if let Err(e) = VirtualProtect(ptr, manifest_len_aligned, PAGE_READONLY, &mut old_protect) {
            let _ = VirtualFree(ptr, 0, MEM_RELEASE);
            return Err(crate::HyperlightError::Error(format!(
                "Failed to set manifest memory read-only: {}",
                e
            )));
        }
        ptr
    };

    // Map manifest into guest memory
    if let Err(err) = unsafe {
        vm.map_region(&MemoryRegion {
            host_region: manifest_host_ptr as usize
                ..manifest_host_ptr as usize + manifest_len_aligned,
            guest_region: fs_manifest_addr as usize
                ..fs_manifest_addr as usize + manifest_len_aligned,
            flags: MemoryRegionFlags::READ,
            region_type: MemoryRegionType::HyperlightFS,
        })
    } {
        unsafe {
            let _ = VirtualFree(manifest_host_ptr, 0, MEM_RELEASE);
        }
        return Err(err);
    }

    info!(
        guest_addr = format_args!("{:#x}", fs_manifest_addr),
        size = manifest_len,
        "Mapped manifest into guest"
    );

    // Map each file into guest memory with READ-only permissions
    let mut current_addr = fs_files_addr;
    for mapping in fs_image.file_mappings() {
        let mapped_size = map_file_cow_to_vm(
            vm,
            mapping.host_path(),
            current_addr,
            MemoryRegionFlags::READ,
        )?;

        info!(
            guest_path = %mapping.guest_path(),
            guest_addr = format_args!("{:#x}", current_addr),
            size = mapped_size,
            "Mapped file into guest"
        );

        current_addr += mapped_size;
    }

    // Map FAT mounts into guest memory with READ|WRITE permissions.
    for fat_mount in fs_image.fat_mounts() {
        let fat_ptr = fat_mount.image().as_ptr();
        let fat_size = fat_mount.image().size();
        let page_size = page_size::get();
        let fat_size_aligned = (fat_size + page_size - 1) & !(page_size - 1);

        unsafe {
            vm.map_region(&MemoryRegion {
                host_region: fat_ptr as usize..fat_ptr as usize + fat_size_aligned,
                guest_region: current_addr as usize..current_addr as usize + fat_size_aligned,
                flags: MemoryRegionFlags::READ | MemoryRegionFlags::WRITE,
                region_type: MemoryRegionType::HyperlightFS,
            })?;
        }

        info!(
            mount_point = %fat_mount.mount_point(),
            guest_addr = format_args!("{:#x}", current_addr),
            size = fat_size,
            "Mapped FAT mount into guest (READ|WRITE)"
        );

        current_addr += fat_size_aligned as u64;
    }

    // Calculate the total files region size (RO + FAT)
    let fs_files_region_size = current_addr - fs_files_addr;

    // Write manifest location to PEB
    layout.set_guest_fs_manifest(shared_mem, fs_manifest_addr, manifest_len as u64)?;

    // Write files region location to PEB
    layout.set_guest_fs_region(shared_mem, fs_files_addr, fs_files_region_size)?;

    info!(
        manifest_addr = format_args!("{:#x}", fs_manifest_addr),
        manifest_size = manifest_len,
        files_addr = format_args!("{:#x}", fs_files_addr),
        files_size = fs_files_region_size,
        "HyperlightFS wired into sandbox PEB"
    );

    Ok(())
}

/// Windows helper to map a file into guest memory via the VM directly.
/// Allocates page-aligned memory, reads the file into it, and maps into guest.
#[cfg(windows)]
fn map_file_cow_to_vm(
    vm: &mut HyperlightVm,
    fp: &Path,
    guest_base: u64,
    flags: MemoryRegionFlags,
) -> Result<u64> {
    use std::io::Read;

    use windows::Win32::System::Memory::*;

    use crate::mem::memory_region::{MemoryRegion, MemoryRegionType};

    let mut file = std::fs::File::options().read(true).open(fp)?;
    let file_size = file.metadata()?.len() as usize;
    let page_size = page_size::get();
    let size = file_size.div_ceil(page_size) * page_size;

    unsafe {
        // Allocate page-aligned memory
        let ptr = VirtualAlloc(
            None,
            size,
            MEM_COMMIT | MEM_RESERVE,
            PAGE_READWRITE,
        );
        if ptr.is_null() {
            return Err(crate::HyperlightError::Error(format!(
                "VirtualAlloc failed for {:?}: {:?}",
                fp,
                std::io::Error::last_os_error()
            )));
        }

        // Read file contents into the allocated memory
        let buf = std::slice::from_raw_parts_mut(ptr as *mut u8, file_size);
        file.read_exact(buf).map_err(|e| {
            let _ = VirtualFree(ptr, 0, MEM_RELEASE);
            crate::HyperlightError::Error(format!("Failed to read {:?}: {}", fp, e))
        })?;

        // Make read-only (matching the requested flags)
        if !flags.contains(MemoryRegionFlags::WRITE) {
            let mut old_protect = PAGE_PROTECTION_FLAGS(0);
            if let Err(e) = VirtualProtect(ptr, size, PAGE_READONLY, &mut old_protect) {
                let _ = VirtualFree(ptr, 0, MEM_RELEASE);
                return Err(crate::HyperlightError::Error(format!(
                    "VirtualProtect failed for {:?}: {}",
                    fp, e
                )));
            }
        }

        let base = ptr as usize;

        if let Err(err) = vm.map_region(&MemoryRegion {
            host_region: base..base + size,
            guest_region: guest_base as usize..guest_base as usize + size,
            flags,
            region_type: MemoryRegionType::HyperlightFS,
        }) {
            let _ = VirtualFree(ptr, 0, MEM_RELEASE);
            return Err(err);
        }

        Ok(size as u64)
    }
}

pub(crate) fn set_up_hypervisor_partition(
    mgr: &mut SandboxMemoryManager<GuestSharedMemory>,
    #[cfg_attr(target_os = "windows", allow(unused_variables))] config: &SandboxConfiguration,
    #[cfg(any(crashdump, gdb))] rt_cfg: &SandboxRuntimeConfig,
    _load_info: LoadInfo,
) -> Result<HyperlightVm> {
    let base_ptr = GuestPtr::try_from(Offset::from(0))?;
    #[cfg(feature = "init-paging")]
    let rsp_ptr = {
        let rsp_offset_u64 = mgr.layout.get_rsp_offset() as u64;
        base_ptr + Offset::from(rsp_offset_u64)
    };

    #[cfg(not(feature = "init-paging"))]
    let rsp_ptr = GuestPtr::try_from(Offset::from(0))?;

    let regions = mgr.layout.get_memory_regions(&mgr.shared_mem)?;

    let pml4_ptr = {
        let pml4_offset_u64 = mgr.layout.get_pt_offset() as u64;
        base_ptr + Offset::from(pml4_offset_u64)
    };
    let entrypoint_ptr = mgr
        .entrypoint_offset
        .ok_or_else(|| new_error!("Entrypoint offset is None"))
        .and_then(|x| {
            let entrypoint_total_offset = mgr.load_addr.clone() + x;
            GuestPtr::try_from(entrypoint_total_offset)
        })?;

    // Create gdb thread if gdb is enabled and the configuration is provided
    #[cfg(gdb)]
    let gdb_conn = if let Some(DebugInfo { port }) = rt_cfg.debug_info {
        use crate::hypervisor::gdb::create_gdb_thread;

        let gdb_conn = create_gdb_thread(port);

        // in case the gdb thread creation fails, we still want to continue
        // without gdb
        match gdb_conn {
            Ok(gdb_conn) => Some(gdb_conn),
            Err(e) => {
                log::error!("Could not create gdb connection: {:#}", e);

                None
            }
        }
    } else {
        None
    };

    #[cfg(feature = "mem_profile")]
    let trace_info = MemTraceInfo::new(_load_info.info)?;

    HyperlightVm::new(
        regions,
        pml4_ptr.absolute()?,
        entrypoint_ptr.absolute()?,
        rsp_ptr.absolute()?,
        config,
        #[cfg(target_os = "windows")]
        {
            use crate::hypervisor::wrappers::HandleWrapper;
            use crate::mem::shared_mem::SharedMemory;
            HandleWrapper::from(
                mgr.shared_mem
                    .with_exclusivity(|s| s.get_mmap_file_handle())?,
            )
        },
        #[cfg(target_os = "windows")]
        {
            use crate::mem::shared_mem::SharedMemory;
            mgr.shared_mem.raw_mem_size()
        },
        #[cfg(gdb)]
        gdb_conn,
        #[cfg(crashdump)]
        rt_cfg.clone(),
        #[cfg(feature = "mem_profile")]
        trace_info,
    )
}

#[cfg(test)]
mod tests {
    use hyperlight_testing::{simple_guest_as_string, simple_guest_by_env};

    use super::evolve_impl_multi_use;
    use crate::sandbox::uninitialized::GuestBinary;
    use crate::{MultiUseSandbox, UninitializedSandbox};

    #[test]
    fn test_evolve() {
        let guest_bin_paths = vec![simple_guest_as_string().unwrap()];
        for guest_bin_path in guest_bin_paths {
            let u_sbox =
                UninitializedSandbox::new(GuestBinary::FilePath(guest_bin_path.clone()), None)
                    .unwrap();
            evolve_impl_multi_use(u_sbox).unwrap();
        }
    }

    /// Test that HyperlightFS is wired correctly during sandbox evolution.
    ///
    /// This test:
    /// 1. Creates a HyperlightFS image with a test file
    /// 2. Sets it on the UninitializedSandbox
    /// 3. Evolves the sandbox
    /// 4. Verifies the PEB has the correct FS manifest and region addresses/sizes
    #[cfg(unix)]
    #[test]
    fn test_evolve_with_hyperlight_fs() {
        use std::io::Write;

        use tempfile::TempDir;

        use crate::hyperlight_fs::HyperlightFSBuilder;
        use crate::mem::layout::SandboxMemoryLayout;
        use crate::mem::shared_mem::SharedMemory;

        // Create a test file
        let temp_dir = TempDir::new().unwrap();
        let test_file_path = temp_dir.path().join("test.txt");
        let test_content = b"Hello from HyperlightFS!";
        {
            let mut f = std::fs::File::create(&test_file_path).unwrap();
            f.write_all(test_content).unwrap();
        }

        // Build the FS image
        let fs_image = HyperlightFSBuilder::new()
            .add_file(&test_file_path, "/test.txt")
            .unwrap()
            .build()
            .unwrap();

        // Create sandbox with HyperlightFS
        let guest_bin_path = simple_guest_as_string().unwrap();
        let sandbox = UninitializedSandbox::new(GuestBinary::FilePath(guest_bin_path), None)
            .unwrap()
            .with_hyperlight_fs(fs_image)
            .evolve()
            .unwrap();

        // Verify the PEB has the FS region and manifest set
        let fs_region_ptr_offset = sandbox.mem_mgr.layout.get_guest_fs_region_pointer_offset();
        let fs_region_size_offset = fs_region_ptr_offset - std::mem::size_of::<u64>();
        let manifest_ptr_offset = sandbox
            .mem_mgr
            .layout
            .get_guest_fs_manifest_pointer_offset();
        let manifest_size_offset = sandbox.mem_mgr.layout.get_guest_fs_manifest_size_offset();

        let (fs_ptr, fs_size, manifest_ptr, manifest_size) = sandbox
            .mem_mgr
            .shared_mem
            .clone()
            .with_exclusivity(|excl| {
                let ptr = excl.read_u64(fs_region_ptr_offset).unwrap();
                let size = excl.read_u64(fs_region_size_offset).unwrap();
                let m_ptr = excl.read_u64(manifest_ptr_offset).unwrap();
                let m_size = excl.read_u64(manifest_size_offset).unwrap();
                (ptr, size, m_ptr, m_size)
            })
            .unwrap();

        // The FS region should be after the contiguous memory
        let contiguous_mem_size = sandbox.mem_mgr.layout.get_memory_size().unwrap();
        let expected_fs_base = SandboxMemoryLayout::BASE_ADDRESS + contiguous_mem_size;

        // Manifest should come first, then files
        assert!(
            manifest_ptr >= expected_fs_base as u64,
            "Manifest should be at or after contiguous memory end"
        );
        assert!(manifest_size > 0, "Manifest size should be non-zero");
        assert!(
            fs_ptr > manifest_ptr,
            "Files region should be after manifest"
        );

        // The file region size should be page-aligned and at least as big as the file
        let page_size = page_size::get() as u64;
        assert!(fs_size >= test_content.len() as u64);
        assert_eq!(
            fs_size % page_size,
            0,
            "FS region size should be page-aligned"
        );
    }

    /// Integration test: Guest reads file content from HyperlightFS.
    ///
    /// This test verifies the full end-to-end pipeline:
    /// 1. Host creates a HyperlightFS image with test files
    /// 2. Sandbox is evolved with the FS image
    /// 3. Guest entrypoint initializes the FS from the PEB manifest
    /// 4. Guest function reads file content via `hyperlight_guest::fs`
    /// 5. Host verifies returned content matches original
    ///
    /// Works with both Rust and C guests (set GUEST=c for C guest).
    #[cfg(unix)]
    #[test]
    fn test_guest_reads_file_from_hyperlight_fs() {
        use std::io::Write;

        use tempfile::TempDir;

        use crate::hyperlight_fs::HyperlightFSBuilder;

        // Create test files with known content
        let temp_dir = TempDir::new().unwrap();

        let file1_path = temp_dir.path().join("hello.txt");
        let file1_content = b"Hello from HyperlightFS!";
        {
            let mut f = std::fs::File::create(&file1_path).unwrap();
            f.write_all(file1_content).unwrap();
        }

        let file2_path = temp_dir.path().join("data.bin");
        let file2_content: Vec<u8> = (0..=255u8).collect();
        {
            let mut f = std::fs::File::create(&file2_path).unwrap();
            f.write_all(&file2_content).unwrap();
        }

        // Build the FS image with both files
        let fs_image = HyperlightFSBuilder::new()
            .add_file(&file1_path, "/hello.txt")
            .unwrap()
            .add_file(&file2_path, "/data.bin")
            .unwrap()
            .build()
            .unwrap();

        // Create and evolve sandbox with HyperlightFS (supports both Rust and C guests)
        let guest_bin_path = simple_guest_by_env();
        let mut sandbox: MultiUseSandbox =
            UninitializedSandbox::new(GuestBinary::FilePath(guest_bin_path), None)
                .unwrap()
                .with_hyperlight_fs(fs_image)
                .evolve()
                .unwrap();

        // Check that FS is initialized in guest
        let is_init: i32 = sandbox.call("IsFsInitialized", ()).unwrap();
        assert_eq!(is_init, 1, "HyperlightFS should be initialized in guest");

        // Read file1 and verify content matches
        let read_content: Vec<u8> = sandbox.call("ReadFile", "/hello.txt".to_string()).unwrap();
        assert_eq!(
            read_content, file1_content,
            "File content should match original"
        );

        // Read file2 (binary data) and verify
        let read_binary: Vec<u8> = sandbox.call("ReadFile", "/data.bin".to_string()).unwrap();
        assert_eq!(
            read_binary, file2_content,
            "Binary file content should match original"
        );

        // Reading non-existent file should return empty vec
        let not_found: Vec<u8> = sandbox
            .call("ReadFile", "/nonexistent.txt".to_string())
            .unwrap();
        assert!(
            not_found.is_empty(),
            "Non-existent file should return empty vec"
        );
    }

    /// Test that sandbox without HyperlightFS reports FS as not initialized.
    ///
    /// Works with both Rust and C guests (set GUEST=c for C guest).
    #[test]
    fn test_guest_fs_not_initialized_without_hyperlight_fs() {
        let guest_bin_path = simple_guest_by_env();
        let u_sbox =
            UninitializedSandbox::new(GuestBinary::FilePath(guest_bin_path), None).unwrap();
        let mut sandbox = evolve_impl_multi_use(u_sbox).unwrap();

        // FS should not be initialized when no HyperlightFS image was set
        let is_init: i32 = sandbox.call("IsFsInitialized", ()).unwrap();
        assert_eq!(
            is_init, 0,
            "HyperlightFS should NOT be initialized without an image"
        );

        // ReadFile should return empty vec when FS is not initialized
        let read_result: Vec<u8> = sandbox.call("ReadFile", "/any/path".to_string()).unwrap();
        assert!(
            read_result.is_empty(),
            "ReadFile should return empty vec when FS not initialized"
        );
    }
}
