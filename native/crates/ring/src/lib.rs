//! # kore-ring — ring protocols over shared windows
//!
//! The ring family of `res/USERLAND.md` §3.2, all over mm-shared windows
//! (the sharing, revocation, and W^X stories are mm's, already landed);
//! **no kernel ring *machinery* exists or ever will** (#72 retired, §3.5)
//! — no kernel ring object, no ring syscall, no kernel-owned ring memory.
//!
//! **Dated carve-out (#487, 2026-08-14): the kernel links this crate as a
//! ring *endpoint*.**  A vCPU's run loop is the producer of its exit ring
//! and the consumer of its control ring, both living in a
//! hypervisor-owned leased window (`kore_abi::guest`).  That makes
//! `SPSC_ABI` **kernel ABI**: this crate is a contract crate now,
//! effectively un-removable, and the spsc wire format is frozen at
//! kernel-ABI grade.  The guardrails that keep this from becoming a
//! casually-cited precedent are stated at the kernel's one endpoint
//! (`src/runtime/sched/guest.rs`): endpoints only inside the vCPU facet,
//! over hypervisor-owned leases; no kernel-allocated ring windows; no
//! ring syscalls; no other kernel subsystem grows an endpoint without its
//! own decision ticket; the kernel never blocks on a ring; `push_lossy`
//! and the executor-facing [`spsc::Consumer::next`] are forbidden in
//! ring 0.  Dependency tripwire: this crate's dep set stays pinned to
//! `{kore-abi, zerocopy}` — the day it wants a third, the spsc core
//! splits into a crate the kernel links.
//!
//! - [`spsc`] — the typed single-producer/single-consumer queue: fixed
//!   capacity, cache-line-separated head/tail atomics, a `dropped`
//!   counter, a magic+ABI header validated on attach.  The #299 "KINP"
//!   input queue is this shape and survives as its first production
//!   instance (§5.8).  The wake edge is the doorbell mux (§3.1): the
//!   producer kicks with one session-committed notice-mode send **after**
//!   its batch — the kick is the caller's, never a ring op.
//! - [`uring`] — the io_uring-byte-compatible SQE/CQE pair over
//!   `kore_abi::uring`, kept for the **device data plane** (the `kore-abi`
//!   crate header pins T1 byte compatibility with Linux io_uring; a CQE
//!   carries a slot index + status, never a capability).
//! - [`transfer`] — the peer-transfer layer over [`uring`]: a
//!   [`PeerTransfer`](transfer::PeerTransfer) chunks one window-to-window
//!   copy (`op::PEER_COPY`, the `MmioRegion` peer-leaf data plane) into
//!   batched submissions — batch-then-one-kick for bandwidth, a
//!   single-SQE path for latency.  It stages and accounts; the ring, the
//!   doorbell, and the completion reap stay the caller's.
//!
//! **No capability ever rides a ring** (standing doctrine): capabilities
//! move by rendezvous; rings move bytes and indices.  Ring *negotiation*
//! needs no new service — the two parties share the backing lease and the
//! doorbell send over the existing verbs (§3.2).
//!
//! (The `swap::SwapBackend` trait that once lived here stays deleted, as
//! designed — the swap vocabulary is `kore-svc-swap`'s.)

#![no_std]

#[cfg(test)]
extern crate std;

pub mod event;
pub mod spsc;
pub mod transfer;
pub mod uring;

// The io_uring surface stays reachable at the crate root.
pub use uring::{AttachError, Completion, Cqe, Entry, Full, IoUring, Submission, opcode, types};
