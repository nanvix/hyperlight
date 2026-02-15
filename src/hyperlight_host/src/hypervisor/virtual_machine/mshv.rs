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

#[cfg(gdb)]
use std::fmt::Debug;
use std::sync::LazyLock;
#[cfg(feature = "hw-interrupts")]
use std::time::Instant;

#[cfg(gdb)]
use mshv_bindings::{DebugRegisters, hv_message_type_HVMSG_X64_EXCEPTION_INTERCEPT};
use mshv_bindings::{
    FloatingPointUnit, SpecialRegisters, StandardRegisters, hv_message_type,
    hv_message_type_HVMSG_GPA_INTERCEPT, hv_message_type_HVMSG_UNMAPPED_GPA,
    hv_message_type_HVMSG_X64_HALT, hv_message_type_HVMSG_X64_IO_PORT_INTERCEPT,
    hv_partition_property_code_HV_PARTITION_PROPERTY_SYNTHETIC_PROC_FEATURES,
    hv_partition_synthetic_processor_features, hv_register_assoc,
    hv_register_name_HV_X64_REGISTER_RIP, hv_register_value, mshv_user_mem_region,
};
#[cfg(feature = "hw-interrupts")]
use mshv_bindings::{
    hv_register_name_HV_REGISTER_PENDING_INTERRUPTION, hv_register_name_HV_X64_REGISTER_RAX,
};
use mshv_ioctls::{Mshv, VcpuFd, VmFd};
use tracing::{Span, instrument};

#[cfg(gdb)]
use crate::hypervisor::gdb::DebuggableVm;
use crate::hypervisor::regs::{CommonFpu, CommonRegisters, CommonSpecialRegisters};
use crate::hypervisor::virtual_machine::{VirtualMachine, VmExit};
#[cfg(feature = "hw-interrupts")]
use crate::hypervisor::pic_pit::{Pic, Pit};
use crate::mem::memory_region::{MemoryRegion, MemoryRegionFlags};
use crate::{Result, new_error};

/// Determine whether the HyperV for Linux hypervisor API is present
/// and functional.
#[instrument(skip_all, parent = Span::current(), level = "Trace")]
pub(crate) fn is_hypervisor_present() -> bool {
    match Mshv::new() {
        Ok(_) => true,
        Err(_) => {
            log::info!("MSHV is not available on this system");
            false
        }
    }
}

/// A MSHV implementation of a single-vcpu VM
#[derive(Debug)]
pub(crate) struct MshvVm {
    vm_fd: VmFd,
    vcpu_fd: VcpuFd,
    #[cfg(feature = "hw-interrupts")]
    pic: Pic,
    #[cfg(feature = "hw-interrupts")]
    pit: Pit,
    #[cfg(feature = "hw-interrupts")]
    last_tick: Instant,
}

static MSHV: LazyLock<Result<Mshv>> =
    LazyLock::new(|| Mshv::new().map_err(|e| new_error!("Failed to open /dev/mshv: {}", e)));

impl MshvVm {
    /// Create a new instance of a MshvVm
    #[instrument(skip_all, parent = Span::current(), level = "Trace")]
    pub(crate) fn new() -> Result<Self> {
        let mshv = MSHV
            .as_ref()
            .map_err(|e| new_error!("Failed to create MSHV instance: {}", e))?;
        let pr = Default::default();
        let vm_fd = {
            // It's important to avoid create_vm() and explicitly use
            // create_vm_with_args() with an empty arguments structure
            // here, because otherwise the partition is set up with a SynIC.

            let vm_fd = mshv.create_vm_with_args(&pr)?;
            let features: hv_partition_synthetic_processor_features = Default::default();
            vm_fd.set_partition_property(
                hv_partition_property_code_HV_PARTITION_PROPERTY_SYNTHETIC_PROC_FEATURES,
                unsafe { features.as_uint64[0] },
            )?;
            vm_fd.initialize()?;
            vm_fd
        };

        let vcpu_fd = vm_fd.create_vcpu(0)?;

        Ok(Self {
            vm_fd,
            vcpu_fd,
            #[cfg(feature = "hw-interrupts")]
            pic: Pic::new(),
            #[cfg(feature = "hw-interrupts")]
            pit: Pit::new(),
            #[cfg(feature = "hw-interrupts")]
            last_tick: Instant::now(),
        })
    }
}

impl VirtualMachine for MshvVm {
    unsafe fn map_memory(&mut self, (_slot, region): (u32, &MemoryRegion)) -> Result<()> {
        let mshv_region: mshv_user_mem_region = region.into();
        self.vm_fd.map_user_memory(mshv_region)?;
        Ok(())
    }

    fn unmap_memory(&mut self, (_slot, region): (u32, &MemoryRegion)) -> Result<()> {
        let mshv_region: mshv_user_mem_region = region.into();
        self.vm_fd.unmap_user_memory(mshv_region)?;
        Ok(())
    }

    fn run_vcpu(&mut self) -> Result<VmExit> {
        const HALT_MESSAGE: hv_message_type = hv_message_type_HVMSG_X64_HALT;
        const IO_PORT_INTERCEPT_MESSAGE: hv_message_type =
            hv_message_type_HVMSG_X64_IO_PORT_INTERCEPT;
        const UNMAPPED_GPA_MESSAGE: hv_message_type = hv_message_type_HVMSG_UNMAPPED_GPA;
        const INVALID_GPA_ACCESS_MESSAGE: hv_message_type = hv_message_type_HVMSG_GPA_INTERCEPT;
        #[cfg(gdb)]
        const EXCEPTION_INTERCEPT: hv_message_type = hv_message_type_HVMSG_X64_EXCEPTION_INTERCEPT;

        loop {
            // --- Timer injection (hw-interrupts only) ---
            // Before each vCPU entry, check if a timer tick is due and inject
            // an interrupt via HV_REGISTER_PENDING_INTERRUPTION.
            #[cfg(feature = "hw-interrupts")]
            if let Some(period) = self.pit.period() {
                let elapsed = self.last_tick.elapsed();
                if elapsed >= period {
                    self.last_tick = Instant::now();
                    let vector = self.pic.master_vector_base() as u64;
                    // Format: bit 31 = valid, bits 10:8 = type (0 = external), bits 7:0 = vector
                    let pending = vector | (1u64 << 31);
                    self.vcpu_fd.set_reg(&[hv_register_assoc {
                        name: hv_register_name_HV_REGISTER_PENDING_INTERRUPTION,
                        value: hv_register_value { reg64: pending },
                        ..Default::default()
                    }])?;
                }
            }

            let exit_reason = self.vcpu_fd.run();

            match exit_reason {
                Ok(m) => match m.header.message_type {
                    HALT_MESSAGE => {
                        // When hw-interrupts are enabled and the PIT is configured,
                        // HLT means "wait for the next interrupt". Sleep until the
                        // next timer tick, inject the interrupt, and re-enter.
                        #[cfg(feature = "hw-interrupts")]
                        if let Some(period) = self.pit.period() {
                            let elapsed = self.last_tick.elapsed();
                            if elapsed < period {
                                std::thread::sleep(period - elapsed);
                            }
                            self.last_tick = Instant::now();
                            let vector = self.pic.master_vector_base() as u64;
                            let pending = vector | (1u64 << 31);
                            self.vcpu_fd.set_reg(&[hv_register_assoc {
                                name: hv_register_name_HV_REGISTER_PENDING_INTERRUPTION,
                                value: hv_register_value { reg64: pending },
                                ..Default::default()
                            }])?;
                            continue; // re-enter vCPU
                        }
                        return Ok(VmExit::Halt());
                    }
                    IO_PORT_INTERCEPT_MESSAGE => {
                        let io_message =
                            m.to_ioport_info().map_err(mshv_ioctls::MshvError::from)?;
                        let port = io_message.port_number;
                        let rip = io_message.header.rip;
                        let rax = io_message.rax;
                        let instruction_length =
                            io_message.header.instruction_length() as u64;

                        // Extract access_size from the access_info bitfield (lower 3 bits)
                        let access_size =
                            (unsafe { io_message.access_info.as_uint8 } & 0x07) as usize;

                        // Distinguish IO IN (read, type=0) from IO OUT (write, type=1)
                        let is_io_in = io_message.header.intercept_access_type == 0;

                        // mshv, unlike kvm, does not automatically increment RIP
                        self.vcpu_fd.set_reg(&[hv_register_assoc {
                            name: hv_register_name_HV_X64_REGISTER_RIP,
                            value: hv_register_value {
                                reg64: rip + instruction_length,
                            },
                            ..Default::default()
                        }])?;

                        if is_io_in {
                            // === IO IN ===
                            #[cfg(feature = "hw-interrupts")]
                            if let Some(val) = self.pic.handle_io_in(port) {
                                // Set RAX with response value, preserving upper bits
                                let mask = match access_size {
                                    1 => 0xFFu64,
                                    2 => 0xFFFFu64,
                                    4 => 0xFFFF_FFFFu64,
                                    _ => u64::MAX,
                                };
                                let new_rax = (rax & !mask) | (val as u64 & mask);
                                self.vcpu_fd.set_reg(&[hv_register_assoc {
                                    name: hv_register_name_HV_X64_REGISTER_RAX,
                                    value: hv_register_value { reg64: new_rax },
                                    ..Default::default()
                                }])?;
                                continue; // re-enter vCPU
                            }
                            return Ok(VmExit::IoIn(port, access_size as u8));
                        } else {
                            // === IO OUT ===
                            // Mask rax by access_size
                            let mask = match access_size {
                                1 => 0xFFu64,
                                2 => 0xFFFFu64,
                                4 => 0xFFFF_FFFFu64,
                                _ => u64::MAX,
                            };
                            let data_val = rax & mask;

                            #[cfg(feature = "hw-interrupts")]
                            if self.handle_hw_io_out(port, data_val as u8) {
                                continue; // re-enter vCPU
                            }

                            return Ok(VmExit::IoOut(
                                port,
                                data_val.to_le_bytes()[..access_size.max(1)].to_vec(),
                            ));
                        }
                    }
                    UNMAPPED_GPA_MESSAGE => {
                        let mimo_message =
                            m.to_memory_info().map_err(mshv_ioctls::MshvError::from)?;
                        let addr = mimo_message.guest_physical_address;
                        return match MemoryRegionFlags::try_from(mimo_message)? {
                            MemoryRegionFlags::READ => Ok(VmExit::MmioRead(addr)),
                            MemoryRegionFlags::WRITE => Ok(VmExit::MmioWrite(addr)),
                            _ => Ok(VmExit::Unknown(
                                "Unknown MMIO access".to_string(),
                            )),
                        };
                    }
                    INVALID_GPA_ACCESS_MESSAGE => {
                        let mimo_message =
                            m.to_memory_info().map_err(mshv_ioctls::MshvError::from)?;
                        let gpa = mimo_message.guest_physical_address;
                        let access_info = MemoryRegionFlags::try_from(mimo_message)?;
                        return match access_info {
                            MemoryRegionFlags::READ => Ok(VmExit::MmioRead(gpa)),
                            MemoryRegionFlags::WRITE => Ok(VmExit::MmioWrite(gpa)),
                            _ => Ok(VmExit::Unknown(
                                "Unknown MMIO access".to_string(),
                            )),
                        };
                    }
                    #[cfg(gdb)]
                    EXCEPTION_INTERCEPT => {
                        let ex_info = m
                            .to_exception_info()
                            .map_err(mshv_ioctls::MshvError::from)?;
                        let DebugRegisters { dr6, .. } = self.vcpu_fd.get_debug_regs()?;
                        return Ok(VmExit::Debug {
                            dr6,
                            exception: ex_info.exception_vector as u32,
                        });
                    }
                    other => {
                        return Ok(VmExit::Unknown(format!(
                            "Unknown MSHV VCPU exit: {:?}",
                            other
                        )))
                    }
                },
                Err(e) => match e.errno() {
                    // InterruptHandle::kill() sends a signal (SIGRTMIN+offset) to interrupt the vcpu, which causes EINTR
                    libc::EINTR => return Ok(VmExit::Cancelled()),
                    libc::EAGAIN => continue,
                    _ => {
                        return Ok(VmExit::Unknown(format!(
                            "Unknown MSHV VCPU error: {}",
                            e
                        )))
                    }
                },
            }
        }
    }

    fn regs(&self) -> Result<CommonRegisters> {
        let mshv_regs = self.vcpu_fd.get_regs()?;
        Ok((&mshv_regs).into())
    }

    fn set_regs(&self, regs: &CommonRegisters) -> Result<()> {
        let mshv_regs: StandardRegisters = regs.into();
        self.vcpu_fd.set_regs(&mshv_regs)?;
        Ok(())
    }

    fn fpu(&self) -> Result<CommonFpu> {
        let mshv_fpu = self.vcpu_fd.get_fpu()?;
        Ok((&mshv_fpu).into())
    }

    fn set_fpu(&self, fpu: &CommonFpu) -> Result<()> {
        let mshv_fpu: FloatingPointUnit = fpu.into();
        self.vcpu_fd.set_fpu(&mshv_fpu)?;
        Ok(())
    }

    fn sregs(&self) -> Result<CommonSpecialRegisters> {
        let mshv_sregs = self.vcpu_fd.get_sregs()?;
        Ok((&mshv_sregs).into())
    }

    fn set_sregs(&self, sregs: &CommonSpecialRegisters) -> Result<()> {
        let mshv_sregs: SpecialRegisters = sregs.into();
        self.vcpu_fd.set_sregs(&mshv_sregs)?;
        Ok(())
    }

    #[cfg(crashdump)]
    fn xsave(&self) -> Result<Vec<u8>> {
        let xsave = self.vcpu_fd.get_xsave()?;
        Ok(xsave.buffer.to_vec())
    }
}

#[cfg(feature = "hw-interrupts")]
impl MshvVm {
    /// Handle an IO OUT to a hardware emulation port (PIC, PIT, speaker, IO wait).
    /// Returns true if the port was handled internally.
    fn handle_hw_io_out(&mut self, port: u16, val: u8) -> bool {
        // PIC ports
        if self.pic.handle_io_out(port, val) {
            return true;
        }

        // PIT command register
        if port == 0x43 {
            self.pit.handle_command(val);
            return true;
        }

        // PIT channel 0 data
        if port == 0x40 {
            self.pit.handle_data(val);
            return true;
        }

        // Speaker port (0x61) -- silently ignore
        // Equivalent to KVM's KVM_PIT_SPEAKER_DUMMY
        if port == 0x61 {
            return true;
        }

        // IO wait port (0x80) -- silently ignore (timing delay)
        if port == 0x80 {
            return true;
        }

        false
    }
}

#[cfg(gdb)]
impl DebuggableVm for MshvVm {
    fn translate_gva(&self, gva: u64) -> Result<u64> {
        use mshv_bindings::{HV_TRANSLATE_GVA_VALIDATE_READ, HV_TRANSLATE_GVA_VALIDATE_WRITE};

        use crate::HyperlightError;

        let flags = (HV_TRANSLATE_GVA_VALIDATE_READ | HV_TRANSLATE_GVA_VALIDATE_WRITE) as u64;
        let (addr, _) = self
            .vcpu_fd
            .translate_gva(gva, flags)
            .map_err(|_| HyperlightError::TranslateGuestAddress(gva))?;

        Ok(addr)
    }

    fn set_debug(&mut self, enabled: bool) -> Result<()> {
        use mshv_bindings::{
            HV_INTERCEPT_ACCESS_MASK_EXECUTE, HV_INTERCEPT_ACCESS_MASK_NONE,
            hv_intercept_parameters, hv_intercept_type_HV_INTERCEPT_TYPE_EXCEPTION,
            mshv_install_intercept,
        };

        use crate::hypervisor::gdb::arch::{BP_EX_ID, DB_EX_ID};

        let access_type_mask = if enabled {
            HV_INTERCEPT_ACCESS_MASK_EXECUTE
        } else {
            HV_INTERCEPT_ACCESS_MASK_NONE
        };

        for vector in [DB_EX_ID, BP_EX_ID] {
            self.vm_fd
                .install_intercept(mshv_install_intercept {
                    access_type_mask,
                    intercept_type: hv_intercept_type_HV_INTERCEPT_TYPE_EXCEPTION,
                    intercept_parameter: hv_intercept_parameters {
                        exception_vector: vector as u16,
                    },
                })
                .map_err(|e| {
                    new_error!(
                        "Cannot {} exception intercept for vector {}: {}",
                        if enabled { "install" } else { "remove" },
                        vector,
                        e
                    )
                })?;
        }
        Ok(())
    }

    fn set_single_step(&mut self, enable: bool) -> Result<()> {
        let mut regs = self.regs()?;
        if enable {
            regs.rflags |= 1 << 8;
        } else {
            regs.rflags &= !(1 << 8);
        }
        self.set_regs(&regs)?;
        Ok(())
    }

    fn add_hw_breakpoint(&mut self, addr: u64) -> Result<()> {
        use crate::hypervisor::gdb::arch::MAX_NO_OF_HW_BP;

        let mut debug_regs = self.vcpu_fd.get_debug_regs()?;

        // Check if breakpoint already exists
        if [
            debug_regs.dr0,
            debug_regs.dr1,
            debug_regs.dr2,
            debug_regs.dr3,
        ]
        .contains(&addr)
        {
            return Ok(());
        }

        // Find the first available LOCAL (L0–L3) slot
        let i = (0..MAX_NO_OF_HW_BP)
            .position(|i| debug_regs.dr7 & (1 << (i * 2)) == 0)
            .ok_or_else(|| new_error!("Tried to add more than 4 hardware breakpoints"))?;

        // Assign to corresponding debug register
        *[
            &mut debug_regs.dr0,
            &mut debug_regs.dr1,
            &mut debug_regs.dr2,
            &mut debug_regs.dr3,
        ][i] = addr;

        // Enable LOCAL bit
        debug_regs.dr7 |= 1 << (i * 2);

        self.vcpu_fd.set_debug_regs(&debug_regs)?;
        Ok(())
    }

    fn remove_hw_breakpoint(&mut self, addr: u64) -> Result<()> {
        let mut debug_regs = self.vcpu_fd.get_debug_regs()?;

        let regs = [
            &mut debug_regs.dr0,
            &mut debug_regs.dr1,
            &mut debug_regs.dr2,
            &mut debug_regs.dr3,
        ];

        if let Some(i) = regs.iter().position(|&&mut reg| reg == addr) {
            // Clear the address
            *regs[i] = 0;
            // Disable LOCAL bit
            debug_regs.dr7 &= !(1 << (i * 2));
            self.vcpu_fd.set_debug_regs(&debug_regs)?;
            Ok(())
        } else {
            Err(new_error!("Tried to remove non-existing hw-breakpoint"))
        }
    }
}
