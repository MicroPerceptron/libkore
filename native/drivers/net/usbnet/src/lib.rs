//! # kore-usbnet — the CDC Ethernet (ECM) driver core
//!
//! A [`NicBackend`](kore_driver::net::NicBackend) over a USB Ethernet
//! adaptor, reached through the xHCI core's bulk and interrupt pipes.
//!
//! ## Why USB, on a machine with no Ethernet port
//!
//! Neither kore metal target has a wired port, and the peer link (#387) needs
//! an L2 frame service before it needs a radio. A USB-C Ethernet adaptor is a
//! plain USB device — no Thunderbolt involved — so it reaches the same frame
//! service through the xHCI stack that is already live. And CDC-ECM is a
//! *standard*, not one part's glue: nothing written here is specific to a
//! chip, and a device that speaks it works whoever made it.
//!
//! ## The layers
//!
//! | module | what it owns |
//! |---|---|
//! | [`cdc`] | the class's wire facts — constants, request encodings, notification layout |
//! | [`probe`] | descriptor bytes → an [`EcmPlan`](probe::EcmPlan), or a named refusal |
//! | [`device`] | the bring-up ladder and the `NicBackend` impl |
//!
//! `probe` is a pure function over bytes precisely because it is where a
//! device's own claims become addresses this driver will poke — the layer
//! most worth testing against hand-written adversarial descriptors.
//!
//! ## What this crate is not
//!
//! It is a **library**, like `kore-nvme` and `kore-xhci`: no service, no
//! runtime, no capability host. The production composition — claiming a
//! controller from the device broker, the capability intake, the serve loop —
//! is `kore-svc-net`'s, and the in-crate hardware double is `cfg(test)` only.

#![no_std]

#[cfg(test)]
extern crate alloc;

pub mod cdc;
pub mod device;
pub mod probe;

pub use device::EcmDevice;
/// The platform seam, re-exported from [`kore_driver`] — mirroring
/// `kore_nvme::seam` and `kore_xhci::seam` so the driver template is uniform.
pub use kore_driver::seam;
pub use probe::{EcmPlan, ProbeError};

/// A driver-core failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// The descriptor walk found no usable CDC-ECM function.
    Probe(ProbeError),
    /// A control transfer, endpoint configuration, or bulk submission was
    /// refused by the xHCI core.
    Usb(kore_xhci::Error),
    /// The device's MAC address string is not twelve hex digits.
    ///
    /// Refused rather than substituted: a made-up address would put a station
    /// on the wire under an identity nobody chose.
    BadMacString,
    /// A DMA buffer could not be drawn from the pool.
    DmaAlloc,
    /// No receive buffer is free to post, or none is filled to read.
    Exhausted,
}

impl From<kore_xhci::Error> for Error {
    fn from(e: kore_xhci::Error) -> Self {
        Self::Usb(e)
    }
}

impl From<ProbeError> for Error {
    fn from(e: ProbeError) -> Self {
        Self::Probe(e)
    }
}

/// The crate's result alias.
pub type Result<T> = core::result::Result<T, Error>;
