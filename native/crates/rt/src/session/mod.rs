//! # `session` — staging batched syscalls
//!
//! The write-side complement of the kernel's descriptor decode
//! (`runtime/session/prepare.rs` reads the frozen [`kore_abi::session`] layout
//! zero-copy) — the same house pattern as
//! [`BootBlock`](kore_abi::task::BootBlock) ⇄
//! [`Bootstrap::of`](kore_abi::task::Bootstrap::of): kore-rt writes the layout,
//! the frozen reader accepts them.
//!
//! A kore syscall is a **session commit**: one trap carries one staged
//! batch `[ι₁,…,ιₙ]` (`res/SYSCALL.md` — up to [`MAX_INVOCATIONS`](kore_abi::session::MAX_INVOCATIONS), SSA
//! result slots, one closure pass).  What is several crossings on a
//! conventional kernel is one here, and these APIs are where that shows:
//!
//! - [`flows::spawn_thread`] — SYSCALL.md §7 Flow 1 verbatim: view + stack
//!   + cpu share, **3 invocations, 1 crossing**.
//! - [`flows::share_with`] — the peer-share arrangement: lease + own
//!   window + rendezvous transfer, **1 crossing** (five syscalls and two
//!   round trips on Linux: `memfd_create`/`ftruncate`/`mmap`/`SCM_RIGHTS`/
//!   peer `mmap`).
//! - [`flows::device_setup`] — the driver bring-up's canonical fold: BAR
//!   claim + window + DMA buffer + CPU/IOVA shares + interrupt route — the
//!   **whole driver bring-up in one trap**.
//! - [`flows::revoke_batch`] — batch teardown (a cluster reap, and the
//!   precursor of the swap page-out: eviction is `unshare×N` in one
//!   crossing once the pager unshare verb lands).
//!
//! **Zero-alloc by construction**: the builder is inline fixed-capacity
//! (the I9 heapless bound) and [`finish`](SessionBuilder::finish) writes
//! into a caller buffer — ultimately the task's own arena descriptor
//! scratch.  A swap server evicting under memory pressure can stage its
//! page-out without allocating a byte.
//!
//! [`SessionBuilder::commit`] writes these bytes into the caller's arena
//! scratch and crosses through the selected platform transport. Host tests
//! substitute a byte-capturing transport at that explicit seam; production
//! runtime builds select the architecture syscall backend.

mod builder;
pub mod flows;
pub mod slots;

#[cfg(test)]
mod tests;

pub use builder::{BuildError, DESCRIPTOR_MAX, Deliver, MAX_SESSION_FOUNDS, SessionBuilder, Staged};
pub use slots::SlotAlloc;
