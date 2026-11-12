//! The C symbols a vendored C library imports on a kore user target.
//!
//! A C library built by `tools/kore-cc` is freestanding: Zig's musl triple
//! supplies ISO C headers and nothing links musl's runtime. What such a
//! library still calls is defined here, over the task's own heap and Rust's
//! Unicode tables, so each consumer does not grow its own copy:
//!
//! - allocation (`malloc`, `calloc`, `realloc`, `free`) and `abort`, each
//!   behind its own feature for a C runtime that exports its own;
//! - the `<string.h>` routines beyond `compiler_builtins`' `memcpy`,
//!   `memmove`, `memset`, `memcmp` and `strlen` (its `mem` feature, which
//!   every kore user build enables);
//! - with `wide-classes` (a default), the wide character classes;
//! - the `<stdio.h>` writes: the `printf` family over one conversion engine
//!   ([`format`]), every float conversion exact for `double` and both
//!   `long double` formats, whose `va_arg` each ABI's own rule reads
//!   (`va_long_double`); and `stdout`/`stderr` plus consumer-built streams
//!   ([`stdio`]);
//! - the `<stdlib.h>` parsers (`strtol` … `strtod`), `qsort`, `strerror`,
//!   and a per-thread `errno`;
//! - with `long-double`, the byte-level halves of `strtold` and `ldexpl`
//!   ([`long_double`]) a consumer's C layer wraps, since only C can spell
//!   the type.
//!
//! The set began as tree-sitter's (its `wasm-stdlib/imports.txt`) and grew
//! with TCC, the second C consumer (`res/TCC.md` §2.2), and with the C
//! programs libkore.so runs, the third: their `printf` is this crate's.
//!
//! Linking this crate is what supplies the symbols: a consumer names it
//! (`use kore_c_shim as _;`) so the linker keeps it. On the host the crate's
//! symbols are absent — the host's libc serves host tests, and a second
//! `malloc` there would replace the process allocator — while the pure
//! engines behind them ([`format`], [`number`], [`sort`], [`long_double`])
//! are compiled for the host suite, so it tests the code the target runs.
//!
//! The string routines and wide character classes are pure, so a shared
//! library may carry its own (`libkore_grammars.so` takes the classes alone). Everything that touches the process — the
//! standard streams, `errno` (and with it the parsers that set it), and with
//! the default `malloc` and `abort` features the heap's C names and `abort` —
//! sits behind the default `runtime` feature, and a shared library built
//! without it imports those from the executable it links into
//! (`libkore_grammars.so` does), so the heap stays the task's one heap. A C
//! runtime that is itself the process's C library (libkore.so) takes
//! `runtime` alone and exports its own allocator and `abort`.
#![no_std]
#![cfg_attr(all(target_os = "none", feature = "runtime"), feature(c_variadic))]

#[cfg(test)]
extern crate std;

#[cfg(any(test, all(target_os = "none", feature = "runtime")))]
pub mod format;
#[cfg(any(test, all(target_os = "none", feature = "long-double")))]
pub mod long_double;
#[cfg(any(test, all(target_os = "none", feature = "runtime")))]
pub mod number;
#[cfg(any(test, all(target_os = "none", feature = "runtime")))]
pub mod sort;

#[cfg(all(target_os = "none", feature = "malloc"))]
mod alloc_symbols;
#[cfg(all(target_os = "none", feature = "runtime"))]
pub mod errno;
#[cfg(all(target_os = "none", feature = "runtime"))]
mod format_symbols;
#[cfg(all(target_os = "none", feature = "runtime"))]
pub mod stdio;
#[cfg(all(target_os = "none", feature = "runtime"))]
mod stdlib_symbols;
#[cfg(target_os = "none")]
mod string_symbols;
#[cfg(all(target_os = "none", feature = "runtime"))]
mod va_long_double;
#[cfg(all(target_os = "none", feature = "wide-classes"))]
mod wide_symbols;

/// The C `abort`: the library hit a state it cannot leave. The task ends
/// with the aborted status, which its controller reports.
#[cfg(all(target_os = "none", feature = "abort"))]
#[unsafe(no_mangle)]
extern "C" fn abort() -> ! {
    kore_rt::thread::exit(kore_rt::ExitCode::ABORTED)
}
