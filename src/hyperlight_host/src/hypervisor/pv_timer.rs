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

//! Paravirtualized timer for guest-host timer communication.
//!
//! Instead of emulating a hardware PIT (8253/8254), the guest writes the
//! desired timer period in microseconds to a custom I/O port (0x500).
//! The host then injects timer interrupts at that rate using the
//! backend-specific interrupt injection mechanism.
//!
//! PIC emulation is still required for interrupt vector routing and EOI.

use std::time::{Duration, Instant};

/// Custom I/O port for PV timer configuration.
/// The guest writes a 32-bit little-endian value representing the
/// desired timer period in microseconds. A value of 0 disables the timer.
pub(crate) const PV_TIMER_PORT: u16 = 0x500;

/// Paravirtualized timer state.
///
/// Tracks the guest-requested timer period and the time of the last
/// injected tick. Timer injection is performed synchronously in the
/// vCPU run loop (same as the existing PIT emulation on MSHV/WHP).
#[derive(Debug)]
pub(crate) struct PvTimer {
    period: Duration,
    last_tick: Instant,
}

impl PvTimer {
    /// Create a new PvTimer with the given period in microseconds.
    /// Returns None if period_us is 0 (timer disabled).
    pub(crate) fn new(period_us: u32) -> Option<Self> {
        if period_us == 0 {
            return None;
        }

        Some(Self {
            period: Duration::from_micros(period_us as u64),
            last_tick: Instant::now(),
        })
    }

    /// Check if a timer tick is due. If so, reset the last tick time
    /// and return true.
    pub(crate) fn check_tick(&mut self) -> bool {
        let elapsed = self.last_tick.elapsed();
        if elapsed >= self.period {
            self.last_tick = Instant::now();
            true
        } else {
            false
        }
    }

    /// Sleep until the next timer tick is due, then mark the tick.
    /// Used for HLT handling: the guest halted waiting for an interrupt,
    /// so we sleep until the next tick rather than busy-waiting.
    pub(crate) fn sleep_until_tick(&mut self) {
        let elapsed = self.last_tick.elapsed();
        if elapsed < self.period {
            std::thread::sleep(self.period - elapsed);
        }
        self.last_tick = Instant::now();
    }

    /// Returns the configured timer period.
    pub(crate) fn period(&self) -> Duration {
        self.period
    }
}
