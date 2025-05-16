/*
Copyright 2024 The Hyperlight Authors.

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
// Deps

use buddy_system_allocator::LockedHeap;
use guest_function_register::GuestFunctionRegister;
use hyperlight_common::mem::HyperlightPEB;

extern crate alloc;

// Modules
pub mod entrypoint;
pub mod shared_input_data;
pub mod shared_output_data;

pub mod guest_error;
pub mod guest_function_call;
pub mod guest_function_definition;
pub mod guest_function_register;

pub mod host_function_call;

pub(crate) mod guest_logger;
pub mod memory;
pub mod print;

pub mod error;
#[cfg(target_arch = "x86_64")]
pub mod exceptions {
    pub mod gdt;
    pub mod handlers;
    pub mod idt;
    pub mod idtr;
    pub mod interrupt_entry;
}
pub mod logging;

// Globals
#[global_allocator]
pub(crate) static HEAP_ALLOCATOR: LockedHeap<32> = LockedHeap::<32>::empty();

pub static mut P_PEB: Option<*mut HyperlightPEB> = None;
pub static mut MIN_STACK_ADDRESS: u64 = 0;

pub static mut OS_PAGE_SIZE: u32 = 0;

pub(crate) static mut REGISTERED_GUEST_FUNCTIONS: GuestFunctionRegister =
    GuestFunctionRegister::new();
