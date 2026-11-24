//! # kore-rtw89 — the Realtek RTL8922AE (rtw89, BE generation) driver core
//!
//! The Wi-Fi 7 part in the metal x86 target (Gigatron, #388 D0:
//! `56:00.0 [10ec:8922]`), brought up over PCI to **firmware alive**: the
//! MAC powered on, the efuse read, the pinned firmware verified, downloaded
//! and started, and one command answered by the running firmware. That is
//! #388's D1 — the go/no-go for a native driver — and everything this
//! crate does today.
//!
//! ## What the part is
//!
//! A mac80211-class SoftMAC: Linux's `rtw89` keeps authentication,
//! association and the 4-way handshake on the host and offloads scanning,
//! rate adaptation and the baseband/RF tables to firmware. The BE
//! generation carries those tables *inside* the firmware file, so the
//! chip-specific host code is a power-on sequence, an efuse map, a
//! download protocol and a mailbox — which is this crate. The frames (D2)
//! and the second interface (D3) are `res/PEER_LINK.md`'s later slices.
//!
//! Registers live in BAR2 (BAR0 is legacy port I/O the kernel never maps);
//! the interrupt is one plain MSI message, realized as the same
//! `InterruptBlock::SINGLE` route the 82574L takes. Several control
//! registers are byte- and halfword-wide, so the core takes the seam's
//! [`RegsNarrow`](kore_driver::seam::RegsNarrow) contract rather than
//! synthesizing narrow accesses from dword read-modify-writes.
//!
//! ## The layers
//!
//! | module | what it owns |
//! |---|---|
//! | [`regs`] | the register face: the generated transcription and the hand-written compositions with their derivations |
//! | [`product`] | the one runtime selection point: probed PCI identity → generation (#809 §G) |
//! | [`desc`] | the BE PCI descriptors and command/reply headers, as bytes |
//! | [`fw`] | the pinned trust decision, the container, the download plan |
//! | [`efuse`] | the physical dump, the RF block, secure-boot recognition |
//! | [`mac`] | power on/off as data, the DMA engine's pre-init |
//! | [`fwdl`] | the firmware-command ring and the download protocol |
//! | [`mailbox`] | the register mailbox and the `PHY_CAP` witness |
//! | [`nic`] | the bring-up ladder: `Prepared` → `Rtw89`, one typed refusal per rung |
//!
//! ## What this crate is not
//!
//! It is a **library**, like `kore-e1000e`: no service, no runtime, no
//! capability host. The production composition — the broker claim, the
//! firmware lease, the capability intake — is `kore-svc-net`'s, and the
//! in-crate hardware double is `cfg(test)` only.

#![no_std]

#[cfg(test)]
extern crate alloc;
#[cfg(test)]
extern crate std;

mod control;
pub mod desc;
pub mod efuse;
pub mod fw;
pub mod fwdl;
pub mod mac;
pub mod mailbox;
#[cfg(test)]
mod mock;
pub mod nic;
mod phy;
pub mod product;
pub mod regs;

use kore_driver::seam::{Regs, RegsNarrow};
use kore_driver::{Budget, handshake_field, handshake_until};

/// The platform seam, re-exported from [`kore_driver`] — mirroring
/// `kore_e1000e::seam` so the driver template is uniform.
pub use kore_driver::seam;
pub use nic::{Prepared, RX_BUF_BYTES, RX_RING, Rtw89, TX_RING};
pub use product::{PCI_DEVICE_RTL8922AE, PCI_VENDOR_REALTEK, Product};

/// A driver-core failure — each bring-up step a distinct refusal, so a
/// machine with no serial console still says which rung it fell off.
///
/// Declared in the order the ladder climbs: the read-only probe, the
/// firmware's trust decision (no hardware side effect yet), the radio's
/// power, its efuse, the DMA carve and engine, the download, the reply.
/// One exception is documented on its variant: the container's *header*
/// is parsed after the efuse, because the secure-boot bytes decide which
/// key the plan carries, so [`Error::BadFirmware`] can surface after
/// [`Error::Efuse`] as well as before [`Error::PowerOn`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// BAR2 does not decode a live controller (too short, or the first
    /// registers read all-ones: a function that is unrouted or
    /// power-gated).
    NotPresent,
    /// No firmware blob was handed to the ladder — the boot carried no
    /// `kore-rtw89-firmware` module. A provisioning note, not a fault.
    NoFirmware,
    /// The blob is not the pinned firmware — wrong size or BLAKE3 digest,
    /// no image for this cut — or, after the efuse, a header this loader
    /// refuses. Refused before a byte of it reaches the device.
    BadFirmware,
    /// A MAC power sequence (on, or off at quiesce) did not complete
    /// within its budget.
    PowerOn,
    /// The efuse could not be read, or does not parse as this part's.
    Efuse,
    /// A DMA buffer could not be drawn from the pool, or sits beyond the
    /// part's 36-bit reach.
    DmaAlloc,
    /// The DMA engine's pre-init did not complete: a DLE never reported
    /// ready, a channel stayed busy, or the BD-RAM bound never reset.
    DmaInit,
    /// The firmware download did not complete: a path never became
    /// ready, the ring stalled, or the WCPU rejected the image.
    FwDownload,
    /// The running firmware never answered the first command.
    NoAck,
    /// Normal packet-memory, scheduler or CAM initialization did not finish.
    MacInit,
    /// PHY register programming or RF command retirement did not finish.
    PhyInit,
    /// MAC feature command completion was refused or did not arrive.
    MacCommand,
}

impl Error {
    /// This crate's rung on the bring-up ladder, `1..=12` in declaration
    /// order. The row surfaces it as `WIFI_BRING_UP_BASE + rung` so one
    /// decoder ring reads every backend (`drivers/net/README.md`).
    /// Renumber nothing; append only.
    pub const fn rung(self) -> u8 {
        match self {
            Self::NotPresent => 1,
            Self::NoFirmware => 2,
            Self::BadFirmware => 3,
            Self::PowerOn => 4,
            Self::Efuse => 5,
            Self::DmaAlloc => 6,
            Self::DmaInit => 7,
            Self::FwDownload => 8,
            Self::NoAck => 9,
            Self::MacInit => 10,
            Self::PhyInit => 11,
            Self::MacCommand => 12,
        }
    }
}

/// The crate's result alias.
pub type Result<T> = core::result::Result<T, Error>;

/// The unclocked fallback for a budget of `ns`: one probe per
/// microsecond, the ratio `kore-e1000e` uses (40 ms → 40 000 probes). On a
/// machine whose clock cannot answer a probe count is the only honest
/// bound, and it conveys nothing about the device on expiry — the row's
/// founding runs after the boot counter is calibrated, so metal never
/// takes this path; the host mock, whose registers answer synchronously,
/// never approaches it.
pub const fn spins_for(ns: u64) -> u32 {
    let probes = ns / 1_000;
    if probes == 0 {
        1
    } else if probes > u32::MAX as u64 {
        u32::MAX
    } else {
        probes as u32
    }
}

/// The wait tier's two hooks, carried together through the ladder: the
/// clock every budget is taken from, and the relaxation between probes.
/// Function pointers rather than generics, because every wait in the
/// crate takes the same pair and the composition passes them once.
#[derive(Clone, Copy)]
pub struct Waits {
    /// Monotonic nanoseconds, or `None` before the counter is calibrated.
    pub clock: fn() -> Option<u64>,
    /// One relaxation between probes (`kore_driver::host::relax` on
    /// metal; a no-op in the mock).
    pub pause: fn(),
}

impl Waits {
    /// A budget of `ns`, clocked when the clock answers.
    pub fn budget(&self, ns: u64) -> Budget {
        Budget::new(ns, spins_for(ns), self.clock)
    }

    /// A fixed delay (upstream's `mdelay`/`fsleep`): a budget waited out
    /// with nothing to probe.
    pub fn delay(&self, ns: u64) {
        let mut budget = self.budget(ns);
        while budget.may_wait() {
            (self.pause)();
        }
    }

    /// Poll register `r` until `done` accepts its value, within `ns`.
    pub fn poll_until<R: Regs + ?Sized>(
        &self,
        regs: &R,
        r: u32,
        ns: u64,
        mut done: impl FnMut(u32) -> bool,
    ) -> bool {
        handshake_until(regs, self.budget(ns), self.pause, |x: &R| {
            done(x.read32(regs::reg(r)))
        })
        .is_acked()
    }

    /// Until any bit of `mask` is set.
    pub fn poll_set<R: Regs + ?Sized>(&self, regs: &R, r: u32, mask: u32, ns: u64) -> bool {
        self.poll_until(regs, r, ns, |v| v & mask != 0)
    }

    /// Until every bit of `mask` is clear.
    pub fn poll_clear<R: Regs + ?Sized>(&self, regs: &R, r: u32, mask: u32, ns: u64) -> bool {
        self.poll_until(regs, r, ns, |v| v & mask == 0)
    }

    /// Until `mask`'s field reads `expected` (shifted down to its zero).
    pub fn poll_field<R: Regs + ?Sized>(
        &self,
        regs: &R,
        r: u32,
        mask: u32,
        expected: u32,
        ns: u64,
    ) -> bool {
        handshake_field(
            regs,
            regs::reg(r),
            mask,
            expected,
            self.budget(ns),
            self.pause,
        )
        .is_acked()
    }

    /// Until any bit of `mask` in the byte at `r` is set.
    pub fn poll_set8<R: RegsNarrow + ?Sized>(&self, regs: &R, r: u32, mask: u8, ns: u64) -> bool {
        handshake_until(regs, self.budget(ns), self.pause, |x: &R| {
            x.read8(regs::reg(r)) & mask != 0
        })
        .is_acked()
    }

    /// Until every bit of `mask` in the byte at `r` is clear.
    pub fn poll_clear8<R: RegsNarrow + ?Sized>(&self, regs: &R, r: u32, mask: u8, ns: u64) -> bool {
        handshake_until(regs, self.budget(ns), self.pause, |x: &R| {
            x.read8(regs::reg(r)) & mask == 0
        })
        .is_acked()
    }
}
