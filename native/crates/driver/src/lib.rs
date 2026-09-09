//! # kore-driver — the kore driver substrate
//!
//! The mechanics **every native kore driver whose control plane is an async
//! SQ/CQ ring pair** shares, factored out of [`kore_nvme`] so the second such
//! driver ([`kore_xhci`]) — and every one after — composes them instead of
//! re-deriving them.  NVMe *qpairs* and xHCI *command/event/transfer rings*
//! are the same two inventions twice:
//!
//! - a **producer ring** ([`SubRing`]) — a single writer stages entries and
//!   publishes a whole batch with **one doorbell** (a release fence pairs the
//!   staged writes with the doorbell the device polls);
//! - a **consumer ring** ([`CompRing`]) — the device is the single writer and
//!   flips a per-entry **validity bit** on each wrap (NVMe calls it the
//!   *phase* bit, xHCI the *cycle* bit — the identical lock-free
//!   single-writer handshake); the reader polls that bit, takes an acquire
//!   fence, and drains.
//!
//! plus [`CookieTable`] (correlate a completion back to the submitter's
//! `u64`) and [`RingStats`] (the honest `submitted`/`completed`/`errors`
//! books).  The doorbell/acknowledge *register writes* stay caller-supplied
//! closures — NVMe writes a tail index to a stride-computed doorbell, xHCI
//! writes an endpoint target to a per-slot doorbell array and a 64-bit ERDP;
//! the ring mechanic is one mechanic, the register poke is not.
//!
//! Deliberately **not** here: a universal bring-up sequence. Bring-up is
//! genuinely per-device (NVMe CAP/CC/CSTS vs. xHCI handoff/CONFIG/DCBAA).
//! The userland `host` module may standardize capability intake, bound ownership,
//! quiesce, and teardown around a class adapter, but it must never turn the
//! device protocols themselves into a `probe`/`init` god trait.
//!
//! The [`seam`] ([`Regs`](seam::Regs) + [`DmaPool`](seam::DmaPool)) lived in
//! `kore-nvme` first; it never contained anything NVMe-specific, so it moves here
//! and `kore_nvme::seam` re-exports it (source-compatible).

#![no_std]
#![cfg_attr(feature = "service", feature(allocator_api))]

#[cfg(any(test, feature = "alloc"))]
extern crate alloc;

/// The pluggable device-MMU library (page-table walker + format plugins) —
/// shared machinery for any driver whose device embeds its own MMU behind the
/// platform IOMMU (SMMUv3-in-device first).  See the module docs.
pub mod block;
pub mod common;
pub mod cookie;
#[cfg(any(test, feature = "host-double"))]
pub mod host_double;
/// Read-only GPT discovery — how a block driver learns which blocks are
/// its (`res/DEVICE_SERVICE.md` §8.1).  Parsing only; the caller does the
/// I/O, so it is host-testable and device-free.
pub mod gpt;
pub mod net;
pub mod ring;
pub mod seam;
pub mod stats;
/// The completion-wait seam — a `Completion` probe plus an explicitly bounded
/// spin for bring-up and kernel-internal tests. Userland waits on IRQ routes.
pub mod wait;

/// The userland driver-process host (the #101 device-cap intake) — see the
/// module docs.  Feature-gated on `service` (an optional `kore-rt` dep), so the
/// kernel's default-feature build never pulls it.
#[cfg(feature = "service")]
pub mod host;

pub use cookie::{CookieTable, MAX_SLOTS};
pub use ring::{CompRing, CqEntry, SqEntry, SubRing, Wrap};
pub use seam::{Dma, DmaPool, DmaRead, DmaWrite, OutOfRange, Regs, RegsNarrow};
pub use stats::RingStats;
pub use wait::{Ack, Budget, Completion, handshake, handshake_field, handshake_until, spin_wait};
