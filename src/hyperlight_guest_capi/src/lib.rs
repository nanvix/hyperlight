#![no_std]
#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use hyperlight_common::flatbuffer_wrappers::guest_error::ErrorCode;
use hyperlight_guest::entrypoint::abort_with_code_and_message;

use crate::alloc::string::ToString;

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

extern crate alloc;

pub mod dispatch;
pub mod error;
pub mod flatbuffer;
pub mod logging;
pub mod types;
