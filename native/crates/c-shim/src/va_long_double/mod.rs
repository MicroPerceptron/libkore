//! `va_arg(ap, long double)`, which Rust's [`VaList`] cannot spell: Rust has
//! no type for x87's 80-bit format and no `VaArgSafe` binary128, so each ABI's
//! own rule reads the argument's bytes and the `printf` engine decodes them
//! ([`Float`]).
//!
//! [`VaList`] is laid out as the platform's `va_list`, the struct or pointer
//! its procedure-call standard defines, and the rule for a 16-byte,
//! 16-aligned argument comes from that same standard. One file per ABI,
//! selected here; the engine never names one.

use core::ffi::VaList;

use crate::format::Float;

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "riscv64")]
mod riscv64;
#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(target_arch = "aarch64")]
pub(crate) use aarch64::next;
#[cfg(target_arch = "riscv64")]
pub(crate) use riscv64::next;
#[cfg(target_arch = "x86_64")]
pub(crate) use x86_64::next;

#[cfg(not(any(
    target_arch = "aarch64",
    target_arch = "riscv64",
    target_arch = "x86_64"
)))]
compile_error!("kore's C ABIs are x86_64, aarch64 and riscv64: name this one's long double rule");

/// Every kore C ABI's `long double`: sixteen bytes, aligned to sixteen (x87's
/// ten padded out, or binary128's own).
const BYTES: usize = 16;
const ALIGN: usize = 16;

/// The 16-aligned address at or above `at`.
fn align_up(at: *mut u8) -> *mut u8 {
    at.wrapping_add(at.addr().wrapping_neg() & (ALIGN - 1))
}

/// The `long double` stored at `at`, as its bits.
///
/// # Safety
/// `at` is the address of a `long double` argument the caller passed.
unsafe fn read(at: *const u8) -> u128 {
    // SAFETY: the caller's contract; unaligned for no reason but uniformity,
    // since every rule aligns its slot.
    u128::from_le_bytes(unsafe { at.cast::<[u8; BYTES]>().read_unaligned() })
}
