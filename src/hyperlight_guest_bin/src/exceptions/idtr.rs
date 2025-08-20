/*
Copyright 2025 The Hyperlight Authors.

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

use core::mem::size_of;
use core::ptr::addr_of;

use crate::exceptions::idt::{IDT, IdtEntry, init_idt};

#[repr(C, packed)]
pub struct Idtr {
    pub limit: u16,
    pub base: u64,
}

static mut IDTR: Idtr = Idtr { limit: 0, base: 0 };

impl Idtr {
    pub unsafe fn init(&mut self, base: u64, size: u16) {
        self.limit = size - 1;
        self.base = base;
    }

    pub unsafe fn load(&self) {
        unsafe {
            core::arch::asm!("lidt [{}]", in(reg) self, options(readonly, nostack, preserves_flags));
        }
    }
}

#[cfg_attr(target_arch = "x86_64", hyperlight_guest_tracing::trace_function)]
pub(crate) unsafe fn load_idt() {
    unsafe {
        init_idt();

        let idt_size = 256 * size_of::<IdtEntry>();
        let expected_base = addr_of!(IDT) as *const _ as u64;

        // Use &raw mut to get a mutable raw pointer, then dereference it
        // this is to avoid the clippy warning "shared reference to mutable static"
        #[allow(clippy::deref_addrof)]
        let idtr = &mut *(&raw mut IDTR);
        idtr.init(expected_base, idt_size as u16);
        idtr.load();
    }
}
