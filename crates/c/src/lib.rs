//! The runtime of `libkore.so`: libkore's C ABI as one shared object, and the
//! host of an independent C program on Kore.
//!
//! A C program a compiler wrote on Kore (`tcc` at the shell) links
//! `libkore.so` with `-lkore` and includes `kore.h`. From one image the whole
//! system shares, it then has:
//!
//! - `kore.h`'s functions (`kore-ffi`), and the standard C names that
//!   `kore/stdlib.h` and `kore/pthread.h` alias, as real symbols (`malloc`
//!   to `free`, `pthread_create` to `pthread_exit`), over the process heap this
//!   library installs;
//! - a host: the program's `main` runs on a main thread with a C stack, while
//!   thread 0 serves its filesystem session ([`api::kore_fs_run`], declared
//!   in `kore/program.h`) and ends the process when any thread exits.
//!
//! # The shared object
//!
//! A distribution builds the image from a `cdylib` crate whose `[lib]` is
//! named `kore` and whose source names this crate (`use kore_c as _;`); Kore's
//! is `user/runtime/libkore`. Its link carries the contract consumers rely on,
//! which `tools/check-program.py` checks for every ISA:
//!
//! - `--soname=libkore.so`, the name consumers record as `DT_NEEDED`;
//! - `--no-undefined`: the library binds nothing from the program it links into;
//! - a shared object with no entry point, so it registers as a library;
//! - static initial-exec TLS only, which the Kore user targets select;
//! - exactly the declared ABI exported (`kore.h`, `kore/program.h`, the C names
//!   `kore/stdlib.h` and `kore/pthread.h` alias, and the C library's two
//!   lifecycle entry points).
//!
//! It is a C program's runtime, never a Rust program's: a Rust executable links
//! `kore-std` with its own entry, and two runtimes in one process would grow one
//! heap supply behind two cursors.
#![cfg_attr(target_os = "none", no_std)]
#![cfg_attr(target_os = "none", feature(thread_local, c_variadic))]
#![deny(unsafe_op_in_unsafe_fn)]

// The C ABI's exports are this image's reason to exist; naming the crate keeps
// them in the link.
use kore_ffi as _;
// Kore's C library, exported whole from this image.
#[cfg(target_os = "none")]
use kore_c_shim as _;

pub mod api;
// Off Kore only the host suite reads the vector's storage: the page it is
// read from and the hook that answers it are the Kore program's alone.
#[cfg(any(test, target_os = "none"))]
#[cfg_attr(not(target_os = "none"), allow(dead_code))]
mod args;
// Off Kore only the descriptor table's rows and the host suite read this:
// the face is the Kore program's.
#[cfg_attr(not(target_os = "none"), allow(dead_code))]
mod device;
#[cfg(any(test, target_os = "none"))]
mod exit;
// Off Kore only `kore_fd_describe` and the host suite read these two: the
// names, the verbs and most refusals are the Kore program's alone.
#[cfg_attr(not(target_os = "none"), allow(dead_code))]
mod fd;
#[cfg(target_os = "none")]
mod lock;
// Off Kore only the host suite and the device's mapping rules read this.
#[cfg_attr(not(target_os = "none"), allow(dead_code))]
mod mmap;
#[cfg(target_os = "none")]
mod posix;
#[cfg(target_os = "none")]
mod program;
#[cfg(any(test, target_os = "none"))]
mod stdio;

/// The process heap over the founder's `role::HEAP` supply, grown on demand.
/// The program's allocations (`malloc`, `kore_alloc_*`) and this library's
/// own draw from it; with no supply every allocation answers null.
#[cfg(target_os = "none")]
#[global_allocator]
static PROCESS_HEAP: kore_std::runtime::heap::Heap<kore_std::runtime::heap::HeapSupplySource> =
    kore_std::runtime::heap::Heap::new_in(kore_std::runtime::heap::HeapSupplySource);

/// A panic in this library ends the program, recorded; never a silent park.
#[cfg(target_os = "none")]
#[panic_handler]
fn panic(info: &core::panic::PanicInfo<'_>) -> ! {
    program::panicked(info)
}
