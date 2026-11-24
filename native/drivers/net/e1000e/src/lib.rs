//! # kore-e1000e — the Intel 82574L (e1000e) driver core
//!
//! A [`NicBackend`](kore_driver::net::NicBackend) over a PCI NIC: two
//! descriptor rings in host memory, a tail-pointer doorbell per ring, the
//! device writing a done bit back into each descriptor, and one MSI-family
//! route — the NVMe-shaped half of every network part, wired or wireless,
//! and the half `res/NETWORK_SERVICE.md` §7 named as unproven for a frame
//! workload until a PCI backend existed.
//!
//! ## Why this part
//!
//! The 82574L is what QEMU's `-device e1000e` emulates, both kore QEMU
//! profiles already carry one on root port 0, and its datasheet is public:
//! it is the part that proves the PCI frame path today. It is **not** a
//! metal part — neither kore metal target has a wired port — so what this
//! crate establishes is the class engine (`res/PEER_LINK.md` §4 P2), which
//! the Wi-Fi part reuses once #388 D0 names it.
//!
//! ## The layers
//!
//! | module | what it owns |
//! |---|---|
//! | [`regs`] | the register map — generated from the datasheet transcription — and the compositions the core programs |
//! | [`product`] | the one runtime selection point: probed PCI identity → generation (#809 §G) |
//! | [`desc`] | the legacy 16-byte descriptors, and what makes them ring entries |
//! | [`nic`] | the bring-up ladder, the ring engine, and the `NicBackend` impl |
//!
//! ## What this crate is not
//!
//! It is a **library**, like `kore-usbnet` and `kore-nvme`: no service, no
//! runtime, no capability host. The production composition — the broker
//! claim, the capability intake, the serve loop — is `kore-svc-net`'s, and
//! the in-crate hardware double is `cfg(test)` only.

#![no_std]

#[cfg(test)]
extern crate alloc;
#[cfg(test)]
extern crate std;

pub mod desc;
#[cfg(test)]
mod mock;
pub mod nic;
pub mod product;
pub mod regs;

/// The platform seam, re-exported from [`kore_driver`] — mirroring
/// `kore_usbnet::seam` and `kore_nvme::seam` so the driver template is
/// uniform.
pub use kore_driver::seam;
pub use nic::{E1000e, E1000eNic, Prepared, RX_BUF_BYTES, RX_RING, TX_RING};
pub use product::{PCI_DEVICE_82574L, PCI_VENDOR_INTEL, Product};

/// A driver-core failure — each bring-up step a distinct refusal, so a
/// machine with no serial console still says which rung it fell off.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// The BAR does not decode as a live controller (too short, or the
    /// control registers read all-ones: a function that is unrouted or
    /// power-gated).
    NotPresent,
    /// `CTRL.RST` never self-cleared within its budget.
    ResetTimeout,
    /// The NVM auto-read never completed, so the station address the
    /// controller loads from it cannot be trusted.
    NvmNotLoaded,
    /// The receive-address entry is not valid, or holds an address no
    /// station may use. Refused rather than substituted: a made-up address
    /// would put a station on the wire under an identity nobody chose.
    NoStationAddress,
    /// A DMA buffer could not be drawn from the pool.
    DmaAlloc,
    /// The transmit ring is full — serving-time only; the engine retries the
    /// descriptor on its next sweep.
    TxFull,
}

impl Error {
    /// This crate's rung on the bring-up ladder, `1..=6` in declaration
    /// order. The row surfaces it as `PCI_BRING_UP_BASE + rung` so one
    /// decoder ring reads every backend (`drivers/net/README.md`).
    /// Renumber nothing; append only.
    pub const fn rung(self) -> u8 {
        match self {
            Self::NotPresent => 1,
            Self::ResetTimeout => 2,
            Self::NvmNotLoaded => 3,
            Self::NoStationAddress => 4,
            Self::DmaAlloc => 5,
            Self::TxFull => 6,
        }
    }
}

/// The crate's result alias.
pub type Result<T> = core::result::Result<T, Error>;
