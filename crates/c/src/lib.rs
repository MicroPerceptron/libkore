//! `libkore.so`: libkore's C ABI as a shared object, and the host of an
//! independent C program on Kore.
//!
//! A C program a compiler wrote on Kore (`tcc` at the shell) links this
//! library with `-lkore` and includes `kore.h`. From one image the whole
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
//! This is a C program's runtime, never a Rust program's: a Rust executable
//! links `kore-std` with its own entry, and two runtimes in one process would
//! grow one heap supply behind two cursors.
#![cfg_attr(target_os = "none", no_std)]
#![cfg_attr(target_os = "none", feature(thread_local))]
#![deny(unsafe_op_in_unsafe_fn)]

// The C ABI's exports are this image's reason to exist; naming the crate keeps
// them in the link.
use kore_ffi as _;

pub mod api;
#[cfg(any(test, target_os = "none"))]
mod exit;
#[cfg(target_os = "none")]
mod program;

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
