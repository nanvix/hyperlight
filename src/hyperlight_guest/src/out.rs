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

use core::arch::asm;
use core::ffi::{c_char, CStr};

use hyperlight_common::outb::OutBAction;

/// Halts the CPU by executing the `hlt` instruction.
#[inline(never)]
pub fn halt() {
    unsafe { asm!("hlt", options(nostack)) }
}

/// Performs an out operation to the specified port with the given 32-bit value.
pub(crate) unsafe fn out32(port: u16, val: u32) {
    asm!("out dx, eax", in("dx") port, in("eax") val, options(preserves_flags, nomem, nostack));
}

/// Performs an out operation to the specified port with the given data.
pub fn outb(port: u16, data: &[u8]) {
    unsafe {
        let mut i = 0;
        while i < data.len() {
            let remaining = data.len() - i;
            let chunk_len = remaining.min(3);
            let mut chunk = [0u8; 4];
            chunk[0] = chunk_len as u8;
            chunk[1..1 + chunk_len].copy_from_slice(&data[i..i + chunk_len]);
            let val = u32::from_le_bytes(chunk);
            out32(port, val);
            i += chunk_len;
        }
    }
}

/// Prints a message using `OutBAction::DebugPrint`. It transmits bytes of a message
/// through several VMExists and, with such, it is slower than
/// `print_output_with_host_print`.
///
/// This function should be used in debug mode only. This function does not
/// require memory to be setup to be used.
pub fn debug_print(msg: &str) {
    outb(OutBAction::DebugPrint as u16, msg.as_bytes());
}

/// Aborts the program with a code.
pub fn abort_with_code(code: &[u8]) -> ! {
    outb(OutBAction::Abort as u16, code);
    outb(OutBAction::Abort as u16, &[0xFF]); // send abort terminator (if not included in code)
    unreachable!()
}

/// Aborts the program with a code and a message.
///
/// # Safety
/// This function is unsafe because it dereferences a raw pointer.
pub unsafe fn abort_with_code_and_message(code: &[u8], message_ptr: *const c_char) -> ! {
    // Step 1: Send abort code (typically 1 byte, but `code` allows flexibility)
    outb(OutBAction::Abort as u16, code);

    // Step 2: Convert the C string to bytes
    let message_bytes = CStr::from_ptr(message_ptr).to_bytes(); // excludes null terminator

    // Step 3: Send the message itself in chunks
    outb(OutBAction::Abort as u16, message_bytes);

    // Step 4: Send abort terminator to signal completion (e.g., 0xFF)
    outb(OutBAction::Abort as u16, &[0xFF]);

    // This function never returns
    unreachable!()
}
