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

pub const PAGE_SHIFT: u64 = 12;
pub const PAGE_SIZE: u64 = 1 << 12;
pub const PAGE_SIZE_USIZE: usize = 1 << 12;
pub const PAGE_TABLE_SHIFT: u64 = 22;
pub const PAGE_TABLE_SIZE_USIZE: usize = 1 << 22;

#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
pub struct InputData {
    pub size: u64,
    pub ptr: u64,
}

#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
pub struct OutputData {
    pub size: u64,
    pub ptr: u64,
}

#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
pub struct GuestHeapData {
    pub size: u64,
    pub ptr: u64,
}

#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
pub struct GuestStackData {
    /// This is the top of the user stack
    pub min_user_stack_ptr: u64,
    /// This is the user stack pointer
    pub user_stack_ptr: u64,
}

#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
pub struct HyperlightPEB {
    pub security_cookie_seed: u64,
    pub guest_function_dispatch_ptr: u64,
    pub code_ptr: u64,
    pub input_data: InputData,
    pub output_data: OutputData,
    pub guest_heap_data: GuestHeapData,
    pub guest_stack_data: GuestStackData,
}
