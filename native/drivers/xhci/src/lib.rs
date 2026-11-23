//! # kore-xhci — the kore xHCI driver core
//!
//! The **second** native kore driver, and the proof the [`kore_driver`] substrate
//! generalizes beyond NVMe: an xHCI USB host controller's command, event,
//! and transfer rings are the same [`SubRing`](kore_driver::SubRing) /
//! [`CompRing`](kore_driver::CompRing) mechanics kore-nvme's qpair uses — the cycle bit
//! *is* the phase bit, the doorbell discipline is one discipline.  What
//! differs is above the rings (a bus to enumerate, not a block device to
//! read), and that lives here.
//!
//! **Spec facts come from the vendored [`xhci`] crate, à la carte**: its
//! typed `ring::trb` (all 33 TRB layouts, validated decode, the cycle bit at
//! a known position) and `context` (32/64-byte slot/endpoint/input) modules
//! are the high-error-surface layouts, pre-typed.  Its register *accessors*
//! we do **not** use — they perform their own volatile MMIO, which would
//! bypass the [`Regs`](kore_driver::seam::Regs) seam; [`regs`] hand-builds the
//! small register surface on that seam instead.
//!
//! Same template as kore-nvme: a pure core over the seam. The production input
//! task now reaches controller `Running` through capability-backed
//! `CapRegs`/`CapPool`; the in-crate hardware double is `cfg(test)` only, and
//! the kernel first-light seam remains a kernel-internal integration test.
//!
//! **Current scope**: controller bring-up, ports/enumeration, HID boot devices,
//! report decoding, and halt/reset quiesce exist in the core. The production
//! input class binds BAR0, DMA, and one required MSI-family route through
//! `kore_driver::host::DeviceBinding`, then drives IRQ-bounded report drain in
//! its service loop (`res/DEVICE_SERVICE.md` §8.1).

#![no_std]

// The `mock` feature needs `alloc` too: the emulated wire queues frames.
#[cfg(any(test, feature = "alloc", feature = "mock"))]
extern crate alloc;

pub mod descriptor;
pub mod device;
pub mod hid;
pub mod input;
pub mod port;
pub mod regs;
pub mod trb;

/// The platform seam, re-exported from [`kore_driver`] — the kernel first-light
/// boot-test and the userland driver task each provide `Regs` / `Dma` /
/// `DmaPool` impls, and reach the trait through `kore_xhci::seam` (mirroring
/// `kore_nvme::seam`).
pub use kore_driver::seam;

mod controller;

pub use controller::{
    Controller, DEVICE_DMA_PAGES, DbcHandoff, Handoff, InterruptEvent, InterruptProbe, Interrupter,
    MAX_SCRATCHPADS, PointerRegisterAccess, PreparedController, ProbePoll, bring_up_dma_pages,
    pointer_access_for_pci,
};
pub use descriptor::{ConfigWalk, DeviceDescriptor, EndpointDescriptor, InterfaceDescriptor};
pub use device::{ControlProgress, ControlSubmission, Device, Speed};
pub use hid::{KeyboardReport, MouseReport};
#[cfg(any(test, feature = "alloc"))]
pub use input::{DiscoveryCensus, HidSource, InputDrain, InputDriver, InputSource};
pub use input::{LedSync, ReportKind, UsbIdentity, drain_reports, feed_report};
pub use port::{Port, PortState};
pub use regs::Caps;

/// The in-process hardware double.
///
/// `cfg(test)` for this crate's own suite, and behind an off-by-default
/// `mock` feature so a sibling driver core can drive a fake controller in
/// *its* tests — the `kore-uart` precedent. Off by default is what keeps
/// `res/DEVICE_SERVICE.md`'s standing bar true: no audited mock module is
/// reachable in a production build.
#[cfg(any(test, feature = "mock"))]
pub mod mock;

/// A driver-core failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// A DMA buffer could not be allocated from the pool.
    DmaAlloc,
    /// The controller did not halt / reset / become ready in time (CNR
    /// stuck, or controller-fatal `USBSTS.HCE`; QEMU may also raise HCE for
    /// a failed controller DMA access).
    NotReady,
    /// A command produced no completion event within the poll budget (the
    /// command ring was aborted via `CRCR.CA`).
    Timeout,
    /// A command completed with a non-success completion code.
    Completion(u8),
    /// The controller reports a geometry the core cannot serve (no slots,
    /// no ports, an unsupported page size).
    Unsupported,
}

/// The driver-core result type.
pub type Result<T> = core::result::Result<T, Error>;
