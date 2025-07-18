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
#![allow(clippy::disallowed_macros)]
extern crate hyperlight_host;

use std::sync::{Arc, Barrier};

use hyperlight_host::sandbox::Callable;
use hyperlight_host::sandbox::uninitialized::UninitializedSandbox;
use hyperlight_host::sandbox_state::sandbox::EvolvableSandbox;
use hyperlight_host::sandbox_state::transition::Noop;
use hyperlight_host::{GuestBinary, MultiUseSandbox, Result};
use hyperlight_testing::simple_guest_as_string;

fn fn_writer(_msg: String) -> Result<i32> {
    Ok(0)
}

// This example demonstrates how to use the env_logger crate to emit log messages from hyperlight. As no tracing subscriber is set up any trace events that are created
// by Hyperlight will also be emitted as log messages.

fn main() -> Result<()> {
    env_logger::builder()
        .parse_filters("none,hyperlight=info")
        .init();
    // Get the path to a simple guest binary.
    let hyperlight_guest_path =
        simple_guest_as_string().expect("Cannot find the guest binary at the expected location.");

    for _ in 0..20 {
        let path = hyperlight_guest_path.clone();
        let res: Result<()> = {
            // Create a new sandbox.
            let mut usandbox = UninitializedSandbox::new(GuestBinary::FilePath(path), None)?;
            usandbox.register_print(fn_writer)?;

            // Initialize the sandbox.

            let no_op = Noop::<UninitializedSandbox, MultiUseSandbox>::default();

            let mut multiuse_sandbox = usandbox.evolve(no_op)?;

            // Call a guest function 5 times to generate some log entries.
            for _ in 0..5 {
                multiuse_sandbox
                    .call_guest_function_by_name::<String>("Echo", "a".to_string())
                    .unwrap();
            }

            // Define a message to send to the guest.

            let msg = "Hello, World!!\n".to_string();

            // Call a guest function that calls the HostPrint host function 5 times to generate some log entries.
            for _ in 0..5 {
                multiuse_sandbox
                    .call_guest_function_by_name::<i32>("PrintOutput", msg.clone())
                    .unwrap();
            }
            Ok(())
        };

        res.unwrap()
    }

    // Create a new sandbox.
    let usandbox =
        UninitializedSandbox::new(GuestBinary::FilePath(hyperlight_guest_path.clone()), None)?;

    // Initialize the sandbox.

    let no_op = Noop::<UninitializedSandbox, MultiUseSandbox>::default();

    let mut multiuse_sandbox = usandbox.evolve(no_op)?;
    let interrupt_handle = multiuse_sandbox.interrupt_handle();
    let barrier = Arc::new(Barrier::new(2));
    let barrier2 = barrier.clone();
    const NUM_CALLS: i32 = 5;
    let thread = std::thread::spawn(move || {
        for _ in 0..NUM_CALLS {
            barrier2.wait();
            // Sleep for a short time to allow the guest function to run.
            std::thread::sleep(std::time::Duration::from_millis(500));
            // Cancel the host function call.
            interrupt_handle.kill();
        }
    });

    // Call a function that gets cancelled by the host function 5 times to generate some log entries.

    for _ in 0..NUM_CALLS {
        let mut ctx = multiuse_sandbox.new_call_context();
        barrier.wait();
        ctx.call::<()>("Spin", ()).unwrap_err();
        multiuse_sandbox = ctx.finish().unwrap();
    }
    thread.join().unwrap();

    Ok(())
}
