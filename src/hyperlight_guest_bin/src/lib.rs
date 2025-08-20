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
#![no_std]

// === Dependencies ===
extern crate alloc;

use alloc::string::ToString;

use buddy_system_allocator::LockedHeap;
#[cfg(target_arch = "x86_64")]
use exceptions::{gdt::load_gdt, idtr::load_idt};
use guest_function::call::dispatch_function;
use guest_function::register::GuestFunctionRegister;
use guest_logger::init_logger;
use hyperlight_common::flatbuffer_wrappers::guest_error::ErrorCode;
use hyperlight_common::mem::HyperlightPEB;
#[cfg(feature = "mem_profile")]
use hyperlight_common::outb::OutBAction;
use hyperlight_guest::exit::{abort_with_code_and_message, halt};
use hyperlight_guest::guest_handle::handle::GuestHandle;
#[cfg(target_arch = "x86_64")]
use hyperlight_guest_tracing::{trace, trace_function};
use log::LevelFilter;
use spin::Once;

// === Modules ===
#[cfg(target_arch = "x86_64")]
pub mod exceptions {
    pub(super) mod gdt;
    pub mod handler;
    mod idt;
    pub(super) mod idtr;
    mod interrupt_entry;
}
pub mod guest_err;
pub mod guest_function {
    pub(super) mod call;
    pub mod definition;
    pub mod register;
}

pub mod guest_logger;
pub mod host_comm;
pub mod memory;
pub mod paging;

// Globals
#[cfg(feature = "mem_profile")]
struct ProfiledLockedHeap<const ORDER: usize>(LockedHeap<ORDER>);
#[cfg(feature = "mem_profile")]
unsafe impl<const ORDER: usize> alloc::alloc::GlobalAlloc for ProfiledLockedHeap<ORDER> {
    unsafe fn alloc(&self, layout: core::alloc::Layout) -> *mut u8 {
        let addr = unsafe { self.0.alloc(layout) };
        unsafe {
            core::arch::asm!("out dx, al",
                in("dx") OutBAction::TraceMemoryAlloc as u16,
                in("rax") layout.size() as u64,
                in("rcx") addr as u64);
        }
        addr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: core::alloc::Layout) {
        unsafe {
            core::arch::asm!("out dx, al",
                in("dx") OutBAction::TraceMemoryFree as u16,
                in("rax") layout.size() as u64,
                in("rcx") ptr as u64);
            self.0.dealloc(ptr, layout)
        }
    }
    unsafe fn alloc_zeroed(&self, layout: core::alloc::Layout) -> *mut u8 {
        let addr = unsafe { self.0.alloc_zeroed(layout) };
        unsafe {
            core::arch::asm!("out dx, al",
                in("dx") OutBAction::TraceMemoryAlloc as u16,
                in("rax") layout.size() as u64,
                in("rcx") addr as u64);
        }
        addr
    }
    unsafe fn realloc(
        &self,
        ptr: *mut u8,
        layout: core::alloc::Layout,
        new_size: usize,
    ) -> *mut u8 {
        let new_ptr = unsafe { self.0.realloc(ptr, layout, new_size) };
        unsafe {
            core::arch::asm!("out dx, al",
                in("dx") OutBAction::TraceMemoryFree as u16,
                in("rax") layout.size() as u64,
                in("rcx") ptr);
            core::arch::asm!("out dx, al",
                in("dx") OutBAction::TraceMemoryAlloc as u16,
                in("rax") new_size as u64,
                in("rcx") new_ptr);
        }
        new_ptr
    }
}

// === Globals ===
#[cfg(not(feature = "mem_profile"))]
#[global_allocator]
pub(crate) static HEAP_ALLOCATOR: LockedHeap<32> = LockedHeap::<32>::empty();
#[cfg(feature = "mem_profile")]
#[global_allocator]
pub(crate) static HEAP_ALLOCATOR: ProfiledLockedHeap<32> =
    ProfiledLockedHeap(LockedHeap::<32>::empty());

pub(crate) static mut GUEST_HANDLE: GuestHandle = GuestHandle::new();
pub(crate) static mut REGISTERED_GUEST_FUNCTIONS: GuestFunctionRegister =
    GuestFunctionRegister::new();

pub static mut MIN_STACK_ADDRESS: u64 = 0;

pub static mut OS_PAGE_SIZE: u32 = 0;

// === Panic Handler ===
// It looks like rust-analyzer doesn't correctly manage no_std crates,
// and so it displays an error about a duplicate panic_handler.
// See more here: https://github.com/rust-lang/rust-analyzer/issues/4490
// The cfg_attr attribute is used to avoid clippy failures as test pulls in std which pulls in a panic handler
#[cfg_attr(not(test), panic_handler)]
#[allow(clippy::panic)]
// to satisfy the clippy when cfg == test
#[allow(dead_code)]
fn panic(info: &core::panic::PanicInfo) -> ! {
    let msg = info.to_string();
    let c_string = alloc::ffi::CString::new(msg)
        .unwrap_or_else(|_| alloc::ffi::CString::new("panic (invalid utf8)").unwrap());

    unsafe { abort_with_code_and_message(&[ErrorCode::UnknownError as u8], c_string.as_ptr()) }
}

// === Entrypoint ===

unsafe extern "C" {
    fn hyperlight_main();
    fn srand(seed: u32);
}

static INIT: Once = Once::new();

#[unsafe(no_mangle)]
#[cfg_attr(target_arch = "x86_64", trace_function)]
pub extern "C" fn entrypoint(peb_address: u64, seed: u64, ops: u64, max_log_level: u64) {
    if peb_address == 0 {
        panic!("PEB address is null");
    }

    INIT.call_once(|| {
        unsafe {
            GUEST_HANDLE = GuestHandle::init(peb_address as *mut HyperlightPEB);
            #[allow(static_mut_refs)]
            let peb_ptr = GUEST_HANDLE.peb().unwrap();

            let srand_seed = (((peb_address << 8) ^ (seed >> 4)) >> 32) as u32;

            // Set the seed for the random number generator for C code using rand;
            srand(srand_seed);

            // This static is to make it easier to implement the __chkstk function in assembly.
            // It also means that should we change the layout of the struct in the future, we
            // don't have to change the assembly code.
            MIN_STACK_ADDRESS = (*peb_ptr).guest_stack.min_user_stack_address;

            #[cfg(target_arch = "x86_64")]
            {
                // Setup GDT and IDT
                load_gdt();
                load_idt();
            }

            let heap_start = (*peb_ptr).guest_heap.ptr as usize;
            let heap_size = (*peb_ptr).guest_heap.size as usize;
            #[cfg(not(feature = "mem_profile"))]
            let heap_allocator = &HEAP_ALLOCATOR;
            #[cfg(feature = "mem_profile")]
            let heap_allocator = &HEAP_ALLOCATOR.0;
            heap_allocator
                .try_lock()
                .expect("Failed to access HEAP_ALLOCATOR")
                .init(heap_start, heap_size);

            OS_PAGE_SIZE = ops as u32;

            (*peb_ptr).guest_function_dispatch_ptr = dispatch_function as usize as u64;

            // set up the logger
            let max_log_level = LevelFilter::iter()
                .nth(max_log_level as usize)
                .expect("Invalid log level");
            init_logger(max_log_level);

            #[cfg(target_arch = "x86_64")]
            {
                trace!("hyperlight_main",
                    hyperlight_main();
                );
            }

            #[cfg(not(target_arch = "x86_64"))]
            {
                hyperlight_main();
            }
        }
    });

    halt();
}
