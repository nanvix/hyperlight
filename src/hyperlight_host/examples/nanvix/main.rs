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

use ::hyperlight_host::sandbox::SandboxConfiguration;
use ::hyperlight_host::sandbox_state::sandbox::EvolvableSandbox;
use ::hyperlight_host::sandbox_state::transition::Noop;
use ::hyperlight_host::{GuestBinary, MultiUseSandbox, Result, UninitializedSandbox};
use ::std::env;
use ::std::io::{IsTerminal, Write};
use ::std::sync::{Arc, Mutex};
use ::std::time::Instant;
use ::termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};

fn main() -> Result<()> {
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <kernel-image>", args[0]);
        std::process::exit(1);
    }
    let guest_binary_path = &args[1];

    let current_time = Instant::now();

    let mut config: SandboxConfiguration = SandboxConfiguration::default();
    config.set_heap_size(4 * 1024 * 1024);
    config.set_stack_size(4 * 1024);

    // Create an uninitialized sandbox with a guest binary
    let sandbox = UninitializedSandbox::new(
        None,
        GuestBinary::FilePath(guest_binary_path.to_string()),
        Some(config),
        None, // Use default run options.
        Some(&Arc::new(Mutex::new(writer_fn))),
    )?;

    // Initialize sandbox to be able to call host functions
    let mut _multi_use_sandbox: MultiUseSandbox = sandbox.evolve(Noop::default())?;

    let elapsed = current_time.elapsed();
    println!("Boot time: {:?}.", elapsed);

    Ok(())
}

/// The default writer function is to write to stdout with green text.
pub fn writer_fn(s: String) -> Result<i32> {
    match std::io::stdout().is_terminal() {
        false => {
            print!("{}", s);
        }
        true => {
            let mut stdout = StandardStream::stdout(ColorChoice::Auto);
            let mut color_spec = ColorSpec::new();
            color_spec.set_fg(Some(Color::White));
            stdout.set_color(&color_spec)?;
            stdout.write_all(s.as_bytes())?;
            stdout.reset()?;
        }
    }
    Ok(s.len() as i32)
}
