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

use std::fmt::Debug;
use std::option::Option;
use std::path::Path;
use std::sync::{Arc, Mutex};

use log::LevelFilter;
use tracing::{Span, instrument};

use super::host_funcs::{FunctionRegistry, default_writer_func};
use super::mem_mgr::MemMgrWrapper;
use super::uninitialized_evolve::evolve_impl_multi_use;
use crate::func::host_functions::{HostFunction, register_host_function};
use crate::func::{ParameterTuple, SupportedReturnType};
#[cfg(feature = "build-metadata")]
use crate::log_build_details;
use crate::mem::exe::ExeInfo;
use crate::mem::memory_region::{DEFAULT_GUEST_BLOB_MEM_FLAGS, MemoryRegionFlags};
use crate::mem::mgr::{STACK_COOKIE_LEN, SandboxMemoryManager};
use crate::mem::shared_mem::ExclusiveSharedMemory;
use crate::sandbox::SandboxConfiguration;
use crate::sandbox::uninitialized_evolve::evolve_impl_with_prebuilt_mem;
use crate::sandbox_state::sandbox::EvolvableSandbox;
use crate::sandbox_state::transition::{Noop, PreBuiltMemoryManagers};
use crate::{MultiUseSandbox, Result, log_then_return, new_error};

#[cfg(all(target_os = "linux", feature = "seccomp"))]
const EXTRA_ALLOWED_SYSCALLS_FOR_WRITER_FUNC: &[super::ExtraAllowedSyscall] = &[
    // Fuzzing fails without `mmap` being an allowed syscall on our seccomp filter.
    // All fuzzing does is call `PrintOutput` (which calls `HostPrint` ). Thing is, `println!`
    // is designed to be thread-safe in Rust and the std lib ensures this by using
    // buffered I/O, which I think relies on `mmap`. This gets surfaced in fuzzing with an
    // OOM error, which I think is happening because `println!` is not being able to allocate
    // more memory for its buffers for the fuzzer's huge inputs.
    libc::SYS_mmap,
    libc::SYS_brk,
    libc::SYS_mprotect,
    #[cfg(mshv)]
    libc::SYS_close,
];

#[cfg(any(crashdump, gdb))]
#[derive(Clone, Debug, Default)]
pub(crate) struct SandboxRuntimeConfig {
    #[cfg(crashdump)]
    pub(crate) binary_path: Option<String>,
    #[cfg(gdb)]
    pub(crate) debug_info: Option<super::config::DebugInfo>,
    #[cfg(crashdump)]
    pub(crate) guest_core_dump: bool,
}

/// A preliminary `Sandbox`, not yet ready to execute guest code.
///
/// Prior to initializing a full-fledged `Sandbox`, you must create one of
/// these `UninitializedSandbox`es with the `new` function, register all the
/// host-implemented functions you need to be available to the guest, then
/// call  `evolve` to transform your
/// `UninitializedSandbox` into an initialized `Sandbox`.
pub struct UninitializedSandbox {
    /// Registered host functions
    pub(crate) host_funcs: Arc<Mutex<FunctionRegistry>>,
    /// The memory manager for the sandbox.
    pub mgr: MemMgrWrapper<ExclusiveSharedMemory>,
    pub(crate) max_guest_log_level: Option<LevelFilter>,
    pub(crate) config: SandboxConfiguration,
    #[cfg(any(crashdump, gdb))]
    pub(crate) rt_cfg: SandboxRuntimeConfig,
}

impl crate::sandbox_state::sandbox::UninitializedSandbox for UninitializedSandbox {
    #[instrument(skip_all, parent = Span::current(), level = "Trace")]
    fn get_uninitialized_sandbox(&self) -> &crate::sandbox::UninitializedSandbox {
        self
    }

    #[instrument(skip_all, parent = Span::current(), level = "Trace")]
    fn get_uninitialized_sandbox_mut(&mut self) -> &mut crate::sandbox::UninitializedSandbox {
        self
    }
}

impl Debug for UninitializedSandbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UninitializedSandbox")
            .field("memory_layout", &self.mgr.unwrap_mgr().layout)
            .finish()
    }
}

impl crate::sandbox_state::sandbox::Sandbox for UninitializedSandbox {
    fn check_stack_guard(&self) -> Result<bool> {
        log_then_return!(
            "Checking the stack cookie before the sandbox is initialized is unsupported"
        );
    }
}

impl
    EvolvableSandbox<
        UninitializedSandbox,
        MultiUseSandbox,
        Noop<UninitializedSandbox, MultiUseSandbox>,
    > for UninitializedSandbox
{
    /// Evolve `self` to a `MultiUseSandbox` without any additional metadata.
    #[instrument(err(Debug), skip_all, parent = Span::current(), level = "Trace")]
    fn evolve(self, _: Noop<UninitializedSandbox, MultiUseSandbox>) -> Result<MultiUseSandbox> {
        evolve_impl_multi_use(self)
    }
}

impl
    EvolvableSandbox<
        UninitializedSandbox,
        MultiUseSandbox,
        PreBuiltMemoryManagers<UninitializedSandbox, MultiUseSandbox>,
    > for UninitializedSandbox
{
    /// Evolve `self` to a `MultiUseSandbox` using pre-built memory managers.
    #[instrument(err(Debug), skip_all, parent = Span::current(), level = "Trace")]
    fn evolve(
        self,
        metadata: PreBuiltMemoryManagers<UninitializedSandbox, MultiUseSandbox>,
    ) -> Result<MultiUseSandbox> {
        evolve_impl_with_prebuilt_mem(
            self,
            metadata.hshm,
            metadata.gshm,
            |hf, mut hshm, vm, out_hdl, mem_hdl, dispatch_ptr| {
                {
                    hshm.as_mut().push_state()?;
                }
                Ok(MultiUseSandbox::from_uninit(
                    hf,
                    hshm.clone(),
                    vm,
                    out_hdl,
                    mem_hdl,
                    dispatch_ptr,
                    #[cfg(gdb)]
                    dbg_mem_access_handler_wrapper(hshm),
                ))
            },
        )
    }
}

/// A `GuestBinary` is either a buffer or the file path to some data (e.g., a guest binary).
#[derive(Debug)]
pub enum GuestBinary<'a> {
    /// A buffer containing the GuestBinary
    Buffer(&'a [u8]),
    /// A path to the GuestBinary
    FilePath(String),
}

/// A `GuestBlob` containing data and the permissions for its use.
#[derive(Debug)]
pub struct GuestBlob<'a> {
    /// The data contained in the blob.
    pub data: &'a [u8],
    /// The permissions for the blob in memory.
    /// By default, it's READ
    pub permissions: MemoryRegionFlags,
}

impl<'a> From<&'a [u8]> for GuestBlob<'a> {
    fn from(data: &'a [u8]) -> Self {
        GuestBlob {
            data,
            permissions: DEFAULT_GUEST_BLOB_MEM_FLAGS,
        }
    }
}

/// A `GuestEnvironment` is a structure that contains the guest binary and an optional GuestBinary.
#[derive(Debug)]
pub struct GuestEnvironment<'a, 'b> {
    /// The guest binary, which can be a file path or a buffer.
    pub guest_binary: GuestBinary<'a>,
    /// An optional guest blob, which can be used to provide additional data to the guest.
    pub init_data: Option<GuestBlob<'b>>,
}

impl<'a, 'b> GuestEnvironment<'a, 'b> {
    /// Creates a new `GuestEnvironment` with the given guest binary and an optional guest blob.
    pub fn new(guest_binary: GuestBinary<'a>, init_data: Option<&'b [u8]>) -> Self {
        GuestEnvironment {
            guest_binary,
            init_data: init_data.map(GuestBlob::from),
        }
    }
}

impl<'a> From<GuestBinary<'a>> for GuestEnvironment<'a, '_> {
    fn from(guest_binary: GuestBinary<'a>) -> Self {
        GuestEnvironment {
            guest_binary,
            init_data: None,
        }
    }
}

impl UninitializedSandbox {
    /// Create a new sandbox configured to run the binary at path
    /// `bin_path`.
    ///
    /// The instrument attribute is used to generate tracing spans and also to emit an error should the Result be an error.
    /// The skip attribute is used to skip the guest binary from being printed in the tracing span.
    /// The name attribute is used to name the tracing span.
    /// The err attribute is used to emit an error should the Result be an error, it uses the std::`fmt::Debug trait` to print the error.
    #[instrument(
        err(Debug),
        skip(env),
        parent = Span::current()
    )]
    pub fn new<'a, 'b>(
        env: impl Into<GuestEnvironment<'a, 'b>>,
        cfg: Option<SandboxConfiguration>,
    ) -> Result<Self> {
        #[cfg(feature = "build-metadata")]
        log_build_details();

        // hyperlight is only supported on Windows 11 and Windows Server 2022 and later
        #[cfg(target_os = "windows")]
        check_windows_version()?;

        let env: GuestEnvironment<'_, '_> = env.into();
        let guest_binary = env.guest_binary;
        let guest_blob = env.init_data;

        // If the guest binary is a file make sure it exists
        let guest_binary = match guest_binary {
            GuestBinary::FilePath(binary_path) => {
                let path = Path::new(&binary_path)
                    .canonicalize()
                    .map_err(|e| new_error!("GuestBinary not found: '{}': {}", binary_path, e))?
                    .into_os_string()
                    .into_string()
                    .map_err(|e| new_error!("Error converting OsString to String: {:?}", e))?;

                GuestBinary::FilePath(path)
            }
            buffer @ GuestBinary::Buffer(_) => buffer,
        };

        let sandbox_cfg = cfg.unwrap_or_default();

        #[cfg(any(crashdump, gdb))]
        let rt_cfg = {
            #[cfg(crashdump)]
            let guest_core_dump = sandbox_cfg.get_guest_core_dump();

            #[cfg(gdb)]
            let debug_info = sandbox_cfg.get_guest_debug_info();

            #[cfg(crashdump)]
            let binary_path = if let GuestBinary::FilePath(ref path) = guest_binary {
                Some(path.clone())
            } else {
                None
            };

            SandboxRuntimeConfig {
                #[cfg(crashdump)]
                binary_path,
                #[cfg(gdb)]
                debug_info,
                #[cfg(crashdump)]
                guest_core_dump,
            }
        };

        let mut mem_mgr_wrapper = {
            let mut mgr = UninitializedSandbox::load_guest_binary(
                sandbox_cfg,
                &guest_binary,
                guest_blob.as_ref(),
            )?;

            let stack_guard = Self::create_stack_guard();
            mgr.set_stack_guard(&stack_guard)?;
            MemMgrWrapper::new(mgr, stack_guard)
        };

        mem_mgr_wrapper.write_memory_layout()?;

        // if env has a guest blob, load it into shared mem
        if let Some(blob) = guest_blob {
            mem_mgr_wrapper.write_init_data(blob.data)?;
        }

        let host_funcs = Arc::new(Mutex::new(FunctionRegistry::default()));

        let mut sandbox = Self {
            host_funcs,
            mgr: mem_mgr_wrapper,
            max_guest_log_level: None,
            config: sandbox_cfg,
            #[cfg(any(crashdump, gdb))]
            rt_cfg,
        };

        // If we were passed a writer for host print register it otherwise use the default.
        sandbox.register_print(default_writer_func)?;

        crate::debug!("Sandbox created:  {:#?}", sandbox);

        Ok(sandbox)
    }

    #[instrument(skip_all, parent = Span::current(), level = "Trace")]
    fn create_stack_guard() -> [u8; STACK_COOKIE_LEN] {
        rand::random::<[u8; STACK_COOKIE_LEN]>()
    }

    /// Load the file at `bin_path_str` into a PE file, then attempt to
    /// load the PE file into a `SandboxMemoryManager` and return it.
    ///
    /// If `run_from_guest_binary` is passed as `true`, and this code is
    /// running on windows, this function will call
    /// `SandboxMemoryManager::load_guest_binary_using_load_library` to
    /// create the new `SandboxMemoryManager`. If `run_from_guest_binary` is
    /// passed as `true` and we're not running on windows, this function will
    /// return an `Err`. Otherwise, if `run_from_guest_binary` is passed
    /// as `false`, this function calls `SandboxMemoryManager::load_guest_binary_into_memory`.
    #[instrument(err(Debug), skip_all, parent = Span::current(), level = "Trace")]
    pub(super) fn load_guest_binary(
        cfg: SandboxConfiguration,
        guest_binary: &GuestBinary,
        guest_blob: Option<&GuestBlob>,
    ) -> Result<SandboxMemoryManager<ExclusiveSharedMemory>> {
        let mut exe_info = match guest_binary {
            GuestBinary::FilePath(bin_path_str) => ExeInfo::from_file(bin_path_str)?,
            GuestBinary::Buffer(buffer) => ExeInfo::from_buf(buffer)?,
        };

        SandboxMemoryManager::load_guest_binary_into_memory(cfg, &mut exe_info, guest_blob)
    }

    /// Set the max log level to be used by the guest.
    /// If this is not set then the log level will be determined by parsing the RUST_LOG environment variable.
    /// If the RUST_LOG environment variable is not set then the max log level will be set to `LevelFilter::Error`.
    pub fn set_max_guest_log_level(&mut self, log_level: LevelFilter) {
        self.max_guest_log_level = Some(log_level);
    }

    /// Register a host function with the given name in the sandbox.
    pub fn register<Args: ParameterTuple, Output: SupportedReturnType>(
        &mut self,
        name: impl AsRef<str>,
        host_func: impl Into<HostFunction<Output, Args>>,
    ) -> Result<()> {
        register_host_function(host_func, self, name.as_ref(), None)
    }

    /// Register the host function with the given name in the sandbox.
    /// Unlike `register`, this variant takes a list of extra syscalls that will
    /// allowed during the execution of the function handler.
    #[cfg(all(feature = "seccomp", target_os = "linux"))]
    pub fn register_with_extra_allowed_syscalls<
        Args: ParameterTuple,
        Output: SupportedReturnType,
    >(
        &mut self,
        name: impl AsRef<str>,
        host_func: impl Into<HostFunction<Output, Args>>,
        extra_allowed_syscalls: impl IntoIterator<Item = crate::sandbox::ExtraAllowedSyscall>,
    ) -> Result<()> {
        let extra_allowed_syscalls: Vec<_> = extra_allowed_syscalls.into_iter().collect();
        register_host_function(host_func, self, name.as_ref(), Some(extra_allowed_syscalls))
    }

    /// Register a host function named "HostPrint" that will be called by the guest
    /// when it wants to print to the console.
    /// The "HostPrint" host function is kind of special, as we expect it to have the
    /// `FnMut(String) -> i32` signature.
    pub fn register_print(
        &mut self,
        print_func: impl Into<HostFunction<i32, (String,)>>,
    ) -> Result<()> {
        #[cfg(not(all(target_os = "linux", feature = "seccomp")))]
        self.register("HostPrint", print_func)?;

        #[cfg(all(target_os = "linux", feature = "seccomp"))]
        self.register_with_extra_allowed_syscalls(
            "HostPrint",
            print_func,
            EXTRA_ALLOWED_SYSCALLS_FOR_WRITER_FUNC.iter().copied(),
        )?;

        Ok(())
    }

    /// Register a host function named "HostPrint" that will be called by the guest
    /// when it wants to print to the console.
    /// The "HostPrint" host function is kind of special, as we expect it to have the
    /// `FnMut(String) -> i32` signature.
    /// Unlike `register_print`, this variant takes a list of extra syscalls that will
    /// allowed during the execution of the function handler.
    #[cfg(all(feature = "seccomp", target_os = "linux"))]
    pub fn register_print_with_extra_allowed_syscalls(
        &mut self,
        print_func: impl Into<HostFunction<i32, (String,)>>,
        extra_allowed_syscalls: impl IntoIterator<Item = crate::sandbox::ExtraAllowedSyscall>,
    ) -> Result<()> {
        #[cfg(all(target_os = "linux", feature = "seccomp"))]
        self.register_with_extra_allowed_syscalls(
            "HostPrint",
            print_func,
            EXTRA_ALLOWED_SYSCALLS_FOR_WRITER_FUNC
                .iter()
                .copied()
                .chain(extra_allowed_syscalls),
        )?;

        Ok(())
    }
}
// Check to see if the current version of Windows is supported
// Hyperlight is only supported on Windows 11 and Windows Server 2022 and later
#[cfg(target_os = "windows")]
fn check_windows_version() -> Result<()> {
    use windows_version::{OsVersion, is_server};
    const WINDOWS_MAJOR: u32 = 10;
    const WINDOWS_MINOR: u32 = 0;
    const WINDOWS_PACK: u32 = 0;

    // Windows Server 2022 has version numbers 10.0.20348 or greater
    if is_server() {
        if OsVersion::current() < OsVersion::new(WINDOWS_MAJOR, WINDOWS_MINOR, WINDOWS_PACK, 20348)
        {
            return Err(new_error!(
                "Hyperlight Requires Windows Server 2022 or newer"
            ));
        }
    } else if OsVersion::current()
        < OsVersion::new(WINDOWS_MAJOR, WINDOWS_MINOR, WINDOWS_PACK, 22000)
    {
        return Err(new_error!("Hyperlight Requires Windows 11 or newer"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::mpsc::channel;
    use std::{fs, thread};

    use crossbeam_queue::ArrayQueue;
    use hyperlight_common::flatbuffer_wrappers::function_types::{ParameterValue, ReturnValue};
    use hyperlight_testing::simple_guest_as_string;

    use crate::sandbox::SandboxConfiguration;
    use crate::sandbox::uninitialized::{GuestBinary, GuestEnvironment};
    use crate::sandbox_state::sandbox::EvolvableSandbox;
    use crate::sandbox_state::transition::Noop;
    use crate::{MultiUseSandbox, Result, UninitializedSandbox, new_error};

    #[test]
    fn test_load_extra_blob() {
        let binary_path = simple_guest_as_string().unwrap();
        let buffer = [0xde, 0xad, 0xbe, 0xef];
        let guest_env =
            GuestEnvironment::new(GuestBinary::FilePath(binary_path.clone()), Some(&buffer));

        let uninitialized_sandbox = UninitializedSandbox::new(guest_env, None).unwrap();
        let mut sandbox: MultiUseSandbox = uninitialized_sandbox.evolve(Noop::default()).unwrap();

        let res = sandbox
            .call_guest_function_by_name::<Vec<u8>>("ReadFromUserMemory", (4u64, buffer.to_vec()))
            .expect("Failed to call ReadFromUserMemory");

        assert_eq!(res, buffer.to_vec());
    }

    #[test]
    fn test_new_sandbox() {
        // Guest Binary exists at path

        let binary_path = simple_guest_as_string().unwrap();
        let sandbox = UninitializedSandbox::new(GuestBinary::FilePath(binary_path.clone()), None);
        assert!(sandbox.is_ok());

        // Guest Binary does not exist at path

        let mut binary_path_does_not_exist = binary_path.clone();
        binary_path_does_not_exist.push_str(".nonexistent");
        let uninitialized_sandbox =
            UninitializedSandbox::new(GuestBinary::FilePath(binary_path_does_not_exist), None);
        assert!(uninitialized_sandbox.is_err());

        // Non default memory configuration
        let cfg = {
            let mut cfg = SandboxConfiguration::default();
            cfg.set_input_data_size(0x1000);
            cfg.set_output_data_size(0x1000);
            cfg.set_stack_size(0x1000);
            cfg.set_heap_size(0x1000);
            Some(cfg)
        };

        let uninitialized_sandbox =
            UninitializedSandbox::new(GuestBinary::FilePath(binary_path.clone()), cfg);
        assert!(uninitialized_sandbox.is_ok());

        let uninitialized_sandbox =
            UninitializedSandbox::new(GuestBinary::FilePath(binary_path), None).unwrap();

        // Get a Sandbox from an uninitialized sandbox without a call back function

        let _sandbox: MultiUseSandbox = uninitialized_sandbox.evolve(Noop::default()).unwrap();

        // Test with a valid guest binary buffer

        let binary_path = simple_guest_as_string().unwrap();
        let sandbox =
            UninitializedSandbox::new(GuestBinary::Buffer(&fs::read(binary_path).unwrap()), None);
        assert!(sandbox.is_ok());

        // Test with a invalid guest binary buffer

        let binary_path = simple_guest_as_string().unwrap();
        let mut bytes = fs::read(binary_path).unwrap();
        let _ = bytes.split_off(100);
        let sandbox = UninitializedSandbox::new(GuestBinary::Buffer(&bytes), None);
        assert!(sandbox.is_err());
    }

    #[test]
    fn test_load_guest_binary_manual() {
        let cfg = SandboxConfiguration::default();

        let simple_guest_path = simple_guest_as_string().unwrap();

        UninitializedSandbox::load_guest_binary(
            cfg,
            &GuestBinary::FilePath(simple_guest_path),
            None.as_ref(),
        )
        .unwrap();
    }

    #[test]
    fn test_host_functions() {
        let uninitialized_sandbox = || {
            UninitializedSandbox::new(
                GuestBinary::FilePath(simple_guest_as_string().expect("Guest Binary Missing")),
                None,
            )
            .unwrap()
        };

        // simple register + call
        {
            let mut usbox = uninitialized_sandbox();

            usbox.register("test0", |arg: i32| Ok(arg + 1)).unwrap();

            let sandbox: Result<MultiUseSandbox> = usbox.evolve(Noop::default());
            assert!(sandbox.is_ok());
            let sandbox = sandbox.unwrap();

            let host_funcs = sandbox
                ._host_funcs
                .try_lock()
                .map_err(|_| new_error!("Error locking"));

            assert!(host_funcs.is_ok());

            let res = host_funcs
                .unwrap()
                .call_host_function("test0", vec![ParameterValue::Int(1)])
                .unwrap();

            assert_eq!(res, ReturnValue::Int(2));
        }

        // multiple parameters register + call
        {
            let mut usbox = uninitialized_sandbox();

            usbox.register("test1", |a: i32, b: i32| Ok(a + b)).unwrap();

            let sandbox: Result<MultiUseSandbox> = usbox.evolve(Noop::default());
            assert!(sandbox.is_ok());
            let sandbox = sandbox.unwrap();

            let host_funcs = sandbox
                ._host_funcs
                .try_lock()
                .map_err(|_| new_error!("Error locking"));

            assert!(host_funcs.is_ok());

            let res = host_funcs
                .unwrap()
                .call_host_function(
                    "test1",
                    vec![ParameterValue::Int(1), ParameterValue::Int(2)],
                )
                .unwrap();

            assert_eq!(res, ReturnValue::Int(3));
        }

        // incorrect arguments register + call
        {
            let mut usbox = uninitialized_sandbox();

            usbox
                .register("test2", |msg: String| {
                    println!("test2 called: {}", msg);
                    Ok(())
                })
                .unwrap();

            let sandbox: Result<MultiUseSandbox> = usbox.evolve(Noop::default());
            assert!(sandbox.is_ok());
            let sandbox = sandbox.unwrap();

            let host_funcs = sandbox
                ._host_funcs
                .try_lock()
                .map_err(|_| new_error!("Error locking"));

            assert!(host_funcs.is_ok());

            let res = host_funcs.unwrap().call_host_function("test2", vec![]);
            assert!(res.is_err());
        }

        // calling a function that doesn't exist
        {
            let usbox = uninitialized_sandbox();
            let sandbox: Result<MultiUseSandbox> = usbox.evolve(Noop::default());
            assert!(sandbox.is_ok());
            let sandbox = sandbox.unwrap();

            let host_funcs = sandbox
                ._host_funcs
                .try_lock()
                .map_err(|_| new_error!("Error locking"));

            assert!(host_funcs.is_ok());

            let res = host_funcs.unwrap().call_host_function("test4", vec![]);
            assert!(res.is_err());
        }
    }

    #[test]
    fn test_host_print() {
        // writer as a FnMut closure mutating a captured variable and then trying to access the captured variable
        // after the Sandbox instance has been dropped
        // this example is fairly contrived but we should still support such an approach.

        let (tx, rx) = channel();

        let writer = move |msg| {
            let _ = tx.send(msg);
            Ok(0)
        };

        let mut sandbox = UninitializedSandbox::new(
            GuestBinary::FilePath(simple_guest_as_string().expect("Guest Binary Missing")),
            None,
        )
        .expect("Failed to create sandbox");

        sandbox
            .register_print(writer)
            .expect("Failed to register host print function");

        let host_funcs = sandbox
            .host_funcs
            .try_lock()
            .map_err(|_| new_error!("Error locking"));

        assert!(host_funcs.is_ok());

        host_funcs.unwrap().host_print("test".to_string()).unwrap();

        drop(sandbox);

        let received_msgs: Vec<_> = rx.into_iter().collect();
        assert_eq!(received_msgs, ["test"]);

        // There may be cases where a mutable reference to the captured variable is not required to be used outside the closure
        // e.g. if the function is writing to a file or a socket etc.

        // writer as a FnMut closure mutating a captured variable but not trying to access the captured variable

        // This seems more realistic as the client is creating a file to be written to in the closure
        // and then accessing the file a different handle.
        // The problem is that captured_file still needs static lifetime so even though we can access the data through the second file handle
        // this still does not work as the captured_file is dropped at the end of the function

        // TODO: Currently, we block any writes that are not to
        // the stdout/stderr file handles, so this code is commented
        // out until we can register writer functions like any other
        // host functions with their own set of extra allowed syscalls.
        // In particular, this code should be brought back once we have addressed the issue

        // let captured_file = Arc::new(Mutex::new(NamedTempFile::new().unwrap()));
        // let capture_file_clone = captured_file.clone();
        //
        // let capture_file_lock = captured_file
        //     .try_lock()
        //     .map_err(|_| new_error!("Error locking"))
        //     .unwrap();
        // let mut file = capture_file_lock.reopen().unwrap();
        // drop(capture_file_lock);
        //
        // let writer = move |msg: String| -> Result<i32> {
        //     let mut captured_file = capture_file_clone
        //         .try_lock()
        //         .map_err(|_| new_error!("Error locking"))
        //         .unwrap();
        //     captured_file.write_all(msg.as_bytes()).unwrap();
        //     Ok(0)
        // };
        //
        // let writer_func = Arc::new(Mutex::new(writer));
        //
        // let sandbox = UninitializedSandbox::new(
        //     GuestBinary::FilePath(simple_guest_as_string().expect("Guest Binary Missing")),
        //     None,
        //     None,
        //     Some(&writer_func),
        // )
        // .expect("Failed to create sandbox");
        //
        // let host_funcs = sandbox
        //     .host_funcs
        //     .try_lock()
        //     .map_err(|_| new_error!("Error locking"));
        //
        // assert!(host_funcs.is_ok());
        //
        // host_funcs.unwrap().host_print("test2".to_string()).unwrap();
        //
        // let mut buffer = String::new();
        // file.read_to_string(&mut buffer).unwrap();
        // assert_eq!(buffer, "test2");

        // writer as a function

        fn fn_writer(msg: String) -> Result<i32> {
            assert_eq!(msg, "test2");
            Ok(0)
        }

        let mut sandbox = UninitializedSandbox::new(
            GuestBinary::FilePath(simple_guest_as_string().expect("Guest Binary Missing")),
            None,
        )
        .expect("Failed to create sandbox");

        sandbox
            .register_print(fn_writer)
            .expect("Failed to register host print function");

        let host_funcs = sandbox
            .host_funcs
            .try_lock()
            .map_err(|_| new_error!("Error locking"));

        assert!(host_funcs.is_ok());

        host_funcs.unwrap().host_print("test2".to_string()).unwrap();

        // writer as a method

        let mut test_host_print = TestHostPrint::new();

        // create a closure over the struct method

        let writer_closure = move |s| test_host_print.write(s);

        let mut sandbox = UninitializedSandbox::new(
            GuestBinary::FilePath(simple_guest_as_string().expect("Guest Binary Missing")),
            None,
        )
        .expect("Failed to create sandbox");

        sandbox
            .register_print(writer_closure)
            .expect("Failed to register host print function");

        let host_funcs = sandbox
            .host_funcs
            .try_lock()
            .map_err(|_| new_error!("Error locking"));

        assert!(host_funcs.is_ok());

        host_funcs.unwrap().host_print("test3".to_string()).unwrap();
    }

    struct TestHostPrint {}

    impl TestHostPrint {
        fn new() -> Self {
            TestHostPrint {}
        }

        fn write(&mut self, msg: String) -> Result<i32> {
            assert_eq!(msg, "test3");
            Ok(0)
        }
    }

    #[test]
    fn check_create_and_use_sandbox_on_different_threads() {
        let unintializedsandbox_queue = Arc::new(ArrayQueue::<UninitializedSandbox>::new(10));
        let sandbox_queue = Arc::new(ArrayQueue::<MultiUseSandbox>::new(10));

        for i in 0..10 {
            let simple_guest_path = simple_guest_as_string().expect("Guest Binary Missing");
            let unintializedsandbox = {
                let err_string = format!("failed to create UninitializedSandbox {i}");
                let err_str = err_string.as_str();
                UninitializedSandbox::new(GuestBinary::FilePath(simple_guest_path), None)
                    .expect(err_str)
            };

            {
                let err_string = format!("Failed to push UninitializedSandbox {i}");
                let err_str = err_string.as_str();

                unintializedsandbox_queue
                    .push(unintializedsandbox)
                    .expect(err_str);
            }
        }

        let thread_handles = (0..10)
            .map(|i| {
                let uq = unintializedsandbox_queue.clone();
                let sq = sandbox_queue.clone();
                thread::spawn(move || {
                    let uninitialized_sandbox = uq.pop().unwrap_or_else(|| {
                        panic!("Failed to pop UninitializedSandbox thread {}", i)
                    });

                    let host_funcs = uninitialized_sandbox
                        .host_funcs
                        .try_lock()
                        .map_err(|_| new_error!("Error locking"));

                    assert!(host_funcs.is_ok());

                    host_funcs
                        .unwrap()
                        .host_print(format!("Print from UninitializedSandbox on Thread {}\n", i))
                        .unwrap();

                    let sandbox = uninitialized_sandbox
                        .evolve(Noop::default())
                        .unwrap_or_else(|_| {
                            panic!("Failed to initialize UninitializedSandbox thread {}", i)
                        });

                    sq.push(sandbox).unwrap_or_else(|_| {
                        panic!("Failed to push UninitializedSandbox thread {}", i)
                    })
                })
            })
            .collect::<Vec<_>>();

        for handle in thread_handles {
            handle.join().unwrap();
        }

        let thread_handles = (0..10)
            .map(|i| {
                let sq = sandbox_queue.clone();
                thread::spawn(move || {
                    let sandbox = sq
                        .pop()
                        .unwrap_or_else(|| panic!("Failed to pop Sandbox thread {}", i));

                    let host_funcs = sandbox
                        ._host_funcs
                        .try_lock()
                        .map_err(|_| new_error!("Error locking"));

                    assert!(host_funcs.is_ok());

                    host_funcs
                        .unwrap()
                        .host_print(format!("Print from Sandbox on Thread {}\n", i))
                        .unwrap();
                })
            })
            .collect::<Vec<_>>();

        for handle in thread_handles {
            handle.join().unwrap();
        }
    }

    #[test]
    // Tests that trace data are emitted when a trace subscriber is set
    // this test is ignored because it is incompatible with other tests , specifically those which require a logger for tracing
    // marking  this test as ignored means that running `cargo test` will not run this test but will allow a developer who runs that command
    // from their workstation to be successful without needed to know about test interdependencies
    // this test will be run explicitly as a part of the CI pipeline
    #[ignore]
    #[cfg(feature = "build-metadata")]
    fn test_trace_trace() {
        use hyperlight_testing::logger::Logger as TestLogger;
        use hyperlight_testing::tracing_subscriber::TracingSubscriber as TestSubscriber;
        use serde_json::{Map, Value};
        use tracing::Level as tracing_level;
        use tracing_core::Subscriber;
        use tracing_core::callsite::rebuild_interest_cache;
        use uuid::Uuid;

        use crate::testing::log_values::build_metadata_testing::try_to_strings;
        use crate::testing::log_values::test_value_as_str;

        TestLogger::initialize_log_tracer();
        rebuild_interest_cache();
        let subscriber = TestSubscriber::new(tracing_level::TRACE);
        tracing::subscriber::with_default(subscriber.clone(), || {
            let correlation_id = Uuid::new_v4().as_hyphenated().to_string();
            let span = tracing::error_span!("test_trace_logs", correlation_id).entered();

            // We should be in span 1

            let current_span = subscriber.current_span();
            assert!(current_span.is_known(), "Current span is unknown");
            let current_span_metadata = current_span.into_inner().unwrap();
            assert_eq!(
                current_span_metadata.0.into_u64(),
                1,
                "Current span is not span 1"
            );
            assert_eq!(current_span_metadata.1.name(), "test_trace_logs");

            // Get the span data and check the correlation id

            let span_data = subscriber.get_span(1);
            let span_attributes: &Map<String, Value> = span_data
                .get("span")
                .unwrap()
                .get("attributes")
                .unwrap()
                .as_object()
                .unwrap();

            test_value_as_str(span_attributes, "correlation_id", correlation_id.as_str());

            let mut binary_path = simple_guest_as_string().unwrap();
            binary_path.push_str("does_not_exist");

            let sbox = UninitializedSandbox::new(GuestBinary::FilePath(binary_path), None);
            assert!(sbox.is_err());

            // Now we should still be in span 1 but span 2 should be created (we created entered and exited span 2 when we called UninitializedSandbox::new)

            let current_span = subscriber.current_span();
            assert!(current_span.is_known(), "Current span is unknown");
            let current_span_metadata = current_span.into_inner().unwrap();
            assert_eq!(
                current_span_metadata.0.into_u64(),
                1,
                "Current span is not span 1"
            );

            let span_metadata = subscriber.get_span_metadata(2);
            assert_eq!(span_metadata.name(), "new");

            // There should be one event for the error that the binary path does not exist plus 14 info events for the logging of the crate info

            let events = subscriber.get_events();
            assert_eq!(events.len(), 15);

            let mut count_matching_events = 0;

            for json_value in events {
                let event_values = json_value.as_object().unwrap().get("event").unwrap();
                let metadata_values_map =
                    event_values.get("metadata").unwrap().as_object().unwrap();
                let event_values_map = event_values.as_object().unwrap();

                let expected_error_start = "Error(\"GuestBinary not found:";

                let err_vals_res = try_to_strings([
                    (metadata_values_map, "level"),
                    (event_values_map, "error"),
                    (metadata_values_map, "module_path"),
                    (metadata_values_map, "target"),
                ]);
                if let Ok(err_vals) = err_vals_res {
                    if err_vals[0] == "ERROR"
                        && err_vals[1].starts_with(expected_error_start)
                        && err_vals[2] == "hyperlight_host::sandbox::uninitialized"
                        && err_vals[3] == "hyperlight_host::sandbox::uninitialized"
                    {
                        count_matching_events += 1;
                    }
                }
            }
            assert!(
                count_matching_events == 1,
                "Unexpected number of matching events {}",
                count_matching_events
            );
            span.exit();
            subscriber.clear();
        });
    }

    #[test]
    #[ignore]
    // Tests that traces are emitted as log records when there is no trace
    // subscriber configured.
    #[cfg(feature = "build-metadata")]
    fn test_log_trace() {
        use std::path::PathBuf;

        use hyperlight_testing::logger::{LOGGER as TEST_LOGGER, Logger as TestLogger};
        use log::Level;
        use tracing_core::callsite::rebuild_interest_cache;

        {
            TestLogger::initialize_test_logger();
            TEST_LOGGER.set_max_level(log::LevelFilter::Trace);

            // This makes sure that the metadata interest cache is rebuilt so that
            // the log records are emitted for the trace records

            rebuild_interest_cache();

            let mut invalid_binary_path = simple_guest_as_string().unwrap();
            invalid_binary_path.push_str("does_not_exist");

            let sbox = UninitializedSandbox::new(GuestBinary::FilePath(invalid_binary_path), None);
            assert!(sbox.is_err());

            // When tracing is creating log records it will create a log
            // record for the creation of the span (from the instrument
            // attribute), and will then create a log record for the entry to
            // and exit from the span.
            //
            // It also creates a log record for the span being dropped.
            //
            // In addition there are 14 info log records created for build information
            //
            // So we expect 19 log records for this test, four for the span and
            // then one for the error as the file that we are attempting to
            // load into the sandbox does not exist, plus the 14 info log records

            let num_calls = TEST_LOGGER.num_log_calls();
            assert_eq!(19, num_calls);

            // Log record 1

            let logcall = TEST_LOGGER.get_log_call(0).unwrap();
            assert_eq!(Level::Info, logcall.level);

            assert!(logcall.args.starts_with("new; cfg"));
            assert_eq!("hyperlight_host::sandbox::uninitialized", logcall.target);

            // Log record 2

            let logcall = TEST_LOGGER.get_log_call(1).unwrap();
            assert_eq!(Level::Trace, logcall.level);
            assert_eq!(logcall.args, "-> new;");
            assert_eq!("tracing::span::active", logcall.target);

            // Log record 17

            let logcall = TEST_LOGGER.get_log_call(16).unwrap();
            assert_eq!(Level::Error, logcall.level);
            assert!(
                logcall
                    .args
                    .starts_with("error=Error(\"GuestBinary not found:")
            );
            assert_eq!("hyperlight_host::sandbox::uninitialized", logcall.target);

            // Log record 18

            let logcall = TEST_LOGGER.get_log_call(17).unwrap();
            assert_eq!(Level::Trace, logcall.level);
            assert_eq!(logcall.args, "<- new;");
            assert_eq!("tracing::span::active", logcall.target);

            // Log record 19

            let logcall = TEST_LOGGER.get_log_call(18).unwrap();
            assert_eq!(Level::Trace, logcall.level);
            assert_eq!(logcall.args, "-- new;");
            assert_eq!("tracing::span", logcall.target);
        }
        {
            // test to ensure an invalid binary logs & traces properly
            TEST_LOGGER.clear_log_calls();
            TEST_LOGGER.set_max_level(log::LevelFilter::Info);

            let mut valid_binary_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            valid_binary_path.push("src");
            valid_binary_path.push("sandbox");
            valid_binary_path.push("initialized.rs");

            let sbox = UninitializedSandbox::new(
                GuestBinary::FilePath(valid_binary_path.into_os_string().into_string().unwrap()),
                None,
            );
            assert!(sbox.is_err());

            // There should be 2 calls this time when we change to the log
            // LevelFilter to Info.
            let num_calls = TEST_LOGGER.num_log_calls();
            assert_eq!(2, num_calls);

            // Log record 1

            let logcall = TEST_LOGGER.get_log_call(0).unwrap();
            assert_eq!(Level::Info, logcall.level);

            assert!(logcall.args.starts_with("new; cfg"));
            assert_eq!("hyperlight_host::sandbox::uninitialized", logcall.target);

            // Log record 2

            let logcall = TEST_LOGGER.get_log_call(1).unwrap();
            assert_eq!(Level::Error, logcall.level);
            assert!(
                logcall
                    .args
                    .starts_with("error=Error(\"GuestBinary not found:")
            );
            assert_eq!("hyperlight_host::sandbox::uninitialized", logcall.target);
        }
        {
            TEST_LOGGER.clear_log_calls();
            TEST_LOGGER.set_max_level(log::LevelFilter::Error);

            let sbox = {
                let res = UninitializedSandbox::new(
                    GuestBinary::FilePath(simple_guest_as_string().unwrap()),
                    None,
                );
                res.unwrap()
            };
            let _: Result<MultiUseSandbox> = sbox.evolve(Noop::default());

            let num_calls = TEST_LOGGER.num_log_calls();

            assert_eq!(0, num_calls);
        }
    }

    #[test]
    fn test_invalid_path() {
        let invalid_path = "some/path/that/does/not/exist";
        let sbox = UninitializedSandbox::new(GuestBinary::FilePath(invalid_path.to_string()), None);
        println!("{:?}", sbox);
        #[cfg(target_os = "windows")]
        assert!(
            matches!(sbox, Err(e) if e.to_string().contains("GuestBinary not found: 'some/path/that/does/not/exist': The system cannot find the path specified. (os error 3)"))
        );
        #[cfg(target_os = "linux")]
        assert!(
            matches!(sbox, Err(e) if e.to_string().contains("GuestBinary not found: 'some/path/that/does/not/exist': No such file or directory (os error 2)"))
        );
    }
}
