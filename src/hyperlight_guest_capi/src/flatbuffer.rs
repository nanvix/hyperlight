use alloc::boxed::Box;
use core::ffi::{c_char, CStr};

use hyperlight_common::flatbuffer_wrappers::util::{
    get_flatbuffer_result_from_double, get_flatbuffer_result_from_float,
    get_flatbuffer_result_from_int, get_flatbuffer_result_from_long,
    get_flatbuffer_result_from_string, get_flatbuffer_result_from_uint,
    get_flatbuffer_result_from_ulong, get_flatbuffer_result_from_vec,
    get_flatbuffer_result_from_void,
};
use hyperlight_guest::host_function_call::{
    get_host_value_return_as_int, get_host_value_return_as_long, get_host_value_return_as_uint,
    get_host_value_return_as_ulong,
};

use crate::types::FfiVec;

// The reason for the capitalized type in the function names below
// is to match the names of the variants in hl_ReturnType,
// which is used in the C macros in macro.h

#[no_mangle]
pub extern "C" fn hl_flatbuffer_result_from_Int(value: i32) -> Box<FfiVec> {
    let vec = get_flatbuffer_result_from_int(value);

    Box::new(unsafe { FfiVec::from_vec(vec) })
}

#[no_mangle]
pub extern "C" fn hl_flatbuffer_result_from_UInt(value: u32) -> Box<FfiVec> {
    let vec = get_flatbuffer_result_from_uint(value);

    Box::new(unsafe { FfiVec::from_vec(vec) })
}

#[no_mangle]
pub extern "C" fn hl_flatbuffer_result_from_Long(value: i64) -> Box<FfiVec> {
    let vec = get_flatbuffer_result_from_long(value);

    Box::new(unsafe { FfiVec::from_vec(vec) })
}

#[no_mangle]
pub extern "C" fn hl_flatbuffer_result_from_ULong(value: u64) -> Box<FfiVec> {
    let vec = get_flatbuffer_result_from_ulong(value);

    Box::new(unsafe { FfiVec::from_vec(vec) })
}

#[no_mangle]
pub extern "C" fn hl_flatbuffer_result_from_Float(value: f32) -> Box<FfiVec> {
    let vec = get_flatbuffer_result_from_float(value);

    Box::new(unsafe { FfiVec::from_vec(vec) })
}

#[no_mangle]
pub extern "C" fn hl_flatbuffer_result_from_Double(value: f64) -> Box<FfiVec> {
    let vec = get_flatbuffer_result_from_double(value);

    Box::new(unsafe { FfiVec::from_vec(vec) })
}

#[no_mangle]
pub extern "C" fn hl_flatbuffer_result_from_Void() -> Box<FfiVec> {
    let vec = get_flatbuffer_result_from_void();

    Box::new(unsafe { FfiVec::from_vec(vec) })
}

#[no_mangle]
pub extern "C" fn hl_flatbuffer_result_from_String(value: *const c_char) -> Box<FfiVec> {
    let str = unsafe { CStr::from_ptr(value) };
    let vec = get_flatbuffer_result_from_string(str.to_string_lossy().as_ref());

    Box::new(unsafe { FfiVec::from_vec(vec) })
}

#[no_mangle]
pub extern "C" fn hl_flatbuffer_result_from_Bytes(data: *const u8, len: usize) -> Box<FfiVec> {
    let slice = unsafe { core::slice::from_raw_parts(data, len) };

    let vec = get_flatbuffer_result_from_vec(slice);

    Box::new(unsafe { FfiVec::from_vec(vec) })
}

//--- Functions for getting values returned by host functions calls

#[no_mangle]
pub extern "C" fn hl_get_host_return_value_as_Int() -> i32 {
    get_host_value_return_as_int().expect("Unable to get host return value as int")
}

#[no_mangle]
pub extern "C" fn hl_get_host_return_value_as_UInt() -> u32 {
    get_host_value_return_as_uint().expect("Unable to get host return value as uint")
}

// the same for long, ulong
#[no_mangle]
pub extern "C" fn hl_get_host_return_value_as_Long() -> i64 {
    get_host_value_return_as_long().expect("Unable to get host return value as long")
}

#[no_mangle]
pub extern "C" fn hl_get_host_return_value_as_ULong() -> u64 {
    get_host_value_return_as_ulong().expect("Unable to get host return value as ulong")
}

// TODO add bool, float, double, string, vecbytes
