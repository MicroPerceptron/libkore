//! # kore-nvme — the kore NVMe driver core (SPDK-shaped)
//!
//! The first native kore hardware driver, and the **template every future
//! driver follows**: a pure driver *core* generic over a small platform
//! [`seam`] ([`Regs`](seam::Regs) + [`DmaPool`](seam::DmaPool)), so the
//! same code can run in a capability-backed userland owner, a kernel-internal
//! first-light test, and `cfg(test)` hardware doubles. Those contexts do not
//! carry equal acceptance weight: only the capability-backed task is a
//! production driver service, and this crate does not yet assemble one.
//!
//! **SPDK-shaped, not os0-shaped** (architect direction): where the sibling
//! `os0` driver is a central block *server* (nameserver + per-queue worker
//! threads mediating every I/O — a scheduling hop SPDK exists to reject),
//! kore-nvme is **per-core lockless queue pairs** ([`Qpair`]), kernel off the
//! data path, polled completions, zero-copy DMA into caller buffers.  A
//! qpair *is* a capability; the concepts mirror `spdk_nvme_*` so code
//! written against SPDK ports by renaming.  The spec-fact layers
//! ([`regs`], [`commands`], [`prp`]) are ported from os0 (a working
//! driver); the *architecture* is rewritten.
//!
//! **Current scope**: the raw controller/qpair/namespace API + submit/poll,
//! plus a feature-gated `NvmeSwap` face.  The production service task is
//! `kore-svc-swap-nvme` (`services/swap-nvme`): it claims the controller
//! from the device broker (`res/DEVICE_SERVICE.md` §10), brings this core
//! up over the capability seam, and serves the swap protocol.  The
//! page datapath moves real bytes: they arrive through the client's
//! staging window and transit a driver-owned bounce page on their way to
//! and from the device (§8.1).

#![no_std]
// Typed portable SIMD for the PRP list fill (the doctrine's "typed over
// hand-rolled" for the one hot scan in the core).
#![feature(portable_simd)]

#[cfg(any(test, feature = "alloc"))]
extern crate alloc;

pub mod block;
pub mod commands;
pub mod prp;
pub mod qpair;
pub mod regs;

/// The platform seam, re-exported from [`kore_driver`] — it never contained
/// anything NVMe-specific, so it lives in the shared substrate now and
/// `kore_nvme::seam` is source-compatible with where it used to be defined.
pub use kore_driver::seam;

mod controller;
mod namespace;

#[cfg(feature = "service")]
pub mod swap;

pub use controller::{Controller, DeviceInfo, PreparedController, StartError};
pub use namespace::Namespace;
pub use qpair::{Completion, MAX_QDEPTH, Qpair, QpairError};
pub use regs::Capabilities;
pub use seam::{Dma, DmaPool, Regs};

/// The qpair's visible books — the [`kore_driver::RingStats`] type under its
/// NVMe-facing name (the field vocabulary is shared: `submitted` /
/// `completed` / `errors`).
pub use kore_driver::RingStats as QpairStats;

#[cfg(test)]
mod mock;

/// A driver-core failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// A DMA buffer could not be allocated from the pool.
    DmaAlloc,
    /// The controller did not reach the expected `CSTS.RDY` state in time
    /// (or reported `CSTS.CFS`, a fatal fault).
    NotReady,
    /// An admin command timed out with no completion.
    Timeout,
    /// An admin command completed with a non-success status.
    Status {
        /// Status Code Type.
        sct: u8,
        /// Status Code.
        sc: u8,
    },
    /// A qpair operation refused.
    Qpair(qpair::QpairError),
    /// A PRP build refused.
    Prp(prp::PrpError),
    /// A transfer exceeds the controller's MDTS (split it).
    ExceedsMdts,
    /// The namespace's identify-reported geometry is one this consumer
    /// cannot lawfully do arithmetic on (e.g. an LBA format that cannot
    /// carry the swap page — see `kore_nvme::swap::blocks_per_page`).
    /// Identify data is device input; refusing it is bring-up's job.
    Geometry,
}

impl From<qpair::QpairError> for Error {
    fn from(e: qpair::QpairError) -> Self {
        Error::Qpair(e)
    }
}

impl From<prp::PrpError> for Error {
    fn from(e: prp::PrpError) -> Self {
        Error::Prp(e)
    }
}

/// The driver-core result type.
pub type Result<T> = core::result::Result<T, Error>;
