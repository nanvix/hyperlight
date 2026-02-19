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

//! Minimal userspace PIC (8259A) and PIT (8253/8254) emulation for MSHV.
//!
//! KVM provides in-kernel PIC/PIT via `create_irq_chip()` and `create_pit2()`.
//! MSHV has no equivalent, so we emulate just enough to support Nanvix's
//! timer interrupt requirements.
//!
//! Timer interrupt injection is done synchronously in the vCPU run loop
//! via `HV_REGISTER_PENDING_INTERRUPTION`, rather than from a background
//! thread (which would require `request_virtual_interrupt` / ROOT_HVCALL
//! permissions that are not available in non-root partitions).

use std::time::Duration;

// PIC port constants
const PIC_MASTER_CMD: u16 = 0x20;
const PIC_MASTER_DATA: u16 = 0x21;
const PIC_SLAVE_CMD: u16 = 0xA0;
const PIC_SLAVE_DATA: u16 = 0xA1;

// PIT port constants (used by hyperv_linux.rs for port matching)
#[allow(dead_code)]
const PIT_CHANNEL_0: u16 = 0x40;
#[allow(dead_code)]
const PIT_COMMAND: u16 = 0x43;

// PIT oscillator base frequency in Hz
const PIT_BASE_FREQ: u64 = 1_193_181;

// --- PIC ---

#[derive(Debug, Clone, Copy, PartialEq)]
enum PicInitState {
    Ready,
    Icw2,
    Icw3,
    Icw4,
}

/// Minimal 8259A PIC emulation.
///
/// Tracks the ICW1-4 initialization sequence and interrupt mask register (IMR)
/// for both master and slave PICs. Only enough state to satisfy Nanvix's
/// PIC initialization and EOI writes.
#[derive(Debug)]
pub(crate) struct Pic {
    master_init_state: PicInitState,
    master_vector_base: u8,
    master_imr: u8,
    slave_init_state: PicInitState,
    slave_vector_base: u8,
    slave_imr: u8,
}

impl Pic {
    pub(crate) fn new() -> Self {
        Self {
            master_init_state: PicInitState::Ready,
            master_vector_base: 0,
            master_imr: 0xFF,
            slave_init_state: PicInitState::Ready,
            slave_vector_base: 0,
            slave_imr: 0xFF,
        }
    }

    /// Handle an IO OUT to a PIC port. Returns true if the port was handled.
    pub(crate) fn handle_io_out(&mut self, port: u16, val: u8) -> bool {
        match port {
            PIC_MASTER_CMD => {
                if val & 0x10 != 0 {
                    // ICW1: initialization command word 1
                    self.master_init_state = PicInitState::Icw2;
                }
                // 0x20 = non-specific EOI, silently accept
                // Any other command byte is also silently accepted
                true
            }
            PIC_MASTER_DATA => {
                match self.master_init_state {
                    PicInitState::Icw2 => {
                        self.master_vector_base = val;
                        self.master_init_state = PicInitState::Icw3;
                    }
                    PicInitState::Icw3 => {
                        // ICW3: cascade configuration, just advance state
                        self.master_init_state = PicInitState::Icw4;
                    }
                    PicInitState::Icw4 => {
                        // ICW4: mode configuration, just advance to ready
                        self.master_init_state = PicInitState::Ready;
                    }
                    PicInitState::Ready => {
                        // OCW1: interrupt mask register write
                        self.master_imr = val;
                    }
                }
                true
            }
            PIC_SLAVE_CMD => {
                if val & 0x10 != 0 {
                    self.slave_init_state = PicInitState::Icw2;
                }
                true
            }
            PIC_SLAVE_DATA => {
                match self.slave_init_state {
                    PicInitState::Icw2 => {
                        self.slave_vector_base = val;
                        self.slave_init_state = PicInitState::Icw3;
                    }
                    PicInitState::Icw3 => {
                        self.slave_init_state = PicInitState::Icw4;
                    }
                    PicInitState::Icw4 => {
                        self.slave_init_state = PicInitState::Ready;
                    }
                    PicInitState::Ready => {
                        self.slave_imr = val;
                    }
                }
                true
            }
            _ => false,
        }
    }

    /// Handle an IO IN from a PIC port. Returns Some(value) if the port was handled.
    pub(crate) fn handle_io_in(&self, port: u16) -> Option<u8> {
        match port {
            PIC_MASTER_DATA => Some(self.master_imr),
            PIC_SLAVE_DATA => Some(self.slave_imr),
            _ => None,
        }
    }

    /// Returns the master PIC vector base (IRQ0 maps to this vector).
    pub(crate) fn master_vector_base(&self) -> u8 {
        self.master_vector_base
    }
}

// --- PIT ---

#[derive(Debug, Clone, Copy, PartialEq)]
enum PitByteSelect {
    LoByte,
    HiByte,
}

/// Minimal 8253/8254 PIT emulation.
///
/// Parses channel 0 command and data writes to determine the timer divisor
/// and period. Does NOT spawn a background thread -- timer interrupt injection
/// is performed synchronously in the vCPU run loop by checking elapsed time
/// and writing `HV_REGISTER_PENDING_INTERRUPTION` before each vCPU entry.
#[derive(Debug)]
pub(crate) struct Pit {
    byte_select: PitByteSelect,
    divisor_lo: u8,
    divisor: u16,
    period: Option<Duration>,
}

impl Pit {
    pub(crate) fn new() -> Self {
        Self {
            byte_select: PitByteSelect::LoByte,
            divisor_lo: 0,
            divisor: 0,
            period: None,
        }
    }

    /// Handle a command write to port 0x43.
    pub(crate) fn handle_command(&mut self, _val: u8) {
        // Reset byte select for the next data write sequence.
        // We only support channel 0, lo/hi byte access mode (mode 2).
        self.byte_select = PitByteSelect::LoByte;
    }

    /// Handle a data write to port 0x40 (channel 0).
    /// On the second byte (HiByte), computes the divisor and period.
    pub(crate) fn handle_data(&mut self, val: u8) {
        match self.byte_select {
            PitByteSelect::LoByte => {
                self.divisor_lo = val;
                self.byte_select = PitByteSelect::HiByte;
            }
            PitByteSelect::HiByte => {
                self.divisor = u16::from_le_bytes([self.divisor_lo, val]);
                self.byte_select = PitByteSelect::LoByte;

                let divisor = self.divisor as u64;
                if divisor > 0 {
                    let period_ns = divisor * 1_000_000_000 / PIT_BASE_FREQ;
                    self.period = Some(Duration::from_nanos(period_ns));
                }
            }
        }
    }

    /// Returns the programmed timer period, or None if the PIT has not been configured.
    pub(crate) fn period(&self) -> Option<Duration> {
        self.period
    }
}
