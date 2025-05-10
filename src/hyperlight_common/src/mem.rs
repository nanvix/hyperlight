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

#![allow(non_snake_case)]

pub const PAGE_SHIFT: u64 = 12;
pub const PAGE_SIZE: u64 = 1 << 12;
pub const PAGE_SIZE_USIZE: usize = 1 << 12;
pub const PAGE_TABLE_SHIFT: u64 = 22;
pub const PAGE_TABLE_SIZE_USIZE: usize = 1 << 22;

use core::ffi::{c_char, c_void};

#[repr(C)]
pub struct HostFunctionDefinitions {
    pub fbHostFunctionDetailsSize: u64,
    pub fbHostFunctionDetails: *mut c_void,
}

#[repr(C)]
pub struct GuestErrorData {
    pub guestErrorSize: u64,
    pub guestErrorBuffer: *mut c_void,
}

#[repr(u64)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum RunMode {
    None = 0,
    Hypervisor = 1,
    InProcessWindows = 2,
    InProcessLinux = 3,
    Invalid = 4,
}

#[repr(C)]
pub struct InputData {
    pub inputDataSize: u64,
    pub inputDataBuffer: *mut c_void,
}

#[repr(C)]
pub struct OutputData {
    pub outputDataSize: u64,
    pub outputDataBuffer: *mut c_void,
}

#[repr(C)]
pub struct GuestHeapData {
    pub guestHeapSize: u64,
    pub guestHeapBuffer: *mut c_void,
}

#[repr(C)]
pub struct GuestStackData {
    /// This is the top of the user stack
    pub minUserStackAddress: u64,
    /// This is the user stack pointer
    pub userStackAddress: u64,
    /// This is the stack pointer for the kernel mode stack
    pub kernelStackAddress: u64,
    /// This is the initial stack pointer when init is called its used before the TSS is set up
    pub bootStackAddress: u64,
}

#[repr(C)]
pub struct HyperlightPEB {
    pub security_cookie_seed: u64,
    pub guest_function_dispatch_ptr: u64,
    pub pCode: *mut c_char,
    pub pOutb: *mut c_void,
    pub pOutbContext: *mut c_void,
    pub runMode: RunMode,
    pub inputdata: InputData,
    pub outputdata: OutputData,
    pub guestheapData: GuestHeapData,
    pub gueststackData: GuestStackData,
}
