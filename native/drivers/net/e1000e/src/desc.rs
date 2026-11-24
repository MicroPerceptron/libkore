//! **The legacy descriptors** — 16 bytes each, the 82574 datasheet §7.1.4
//! (receive) and §7.2.3 (transmit) — and what makes them entries of the
//! substrate's rings.
//!
//! Legacy rather than extended: the extended formats exist for offloads
//! this core does not use, and the format is a *mode* of the controller
//! (`RFCTL.EXTEN`, `RCTL.DTYP`, `TXD.CMD.DEXT`) that must agree with what
//! the driver writes — so the core pins all three to the legacy value and
//! the mock refuses a descriptor that asks for anything else.
//!
//! # The done bit is not a phase bit
//!
//! [`CompRing`](kore_driver::CompRing) models a completion ring whose
//! validity bit alternates each wrap. An in-place descriptor ring is
//! different: the same slot the driver posted gets its `DD` status bit set
//! by the device, and `DD` does not alternate — the driver zeroes it when it
//! re-posts the slot. So [`RxDesc`] implements `valid` as *the done bit is
//! set* and ignores the phase the ring tracks; the ring's acquire fence and
//! head discipline are reused verbatim, and the re-zero on refill (before
//! the tail moves) is what keeps a stale `DD` from being read twice.

use core::mem::{offset_of, size_of};

use kore_driver::{CqEntry, SqEntry};

use crate::regs::i82574::*;

/// A legacy receive descriptor.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RxDesc {
    /// Device address of the receive buffer.
    pub addr: u64,
    /// Bytes the device wrote (excluding the FCS with `RCTL.SECRC`).
    pub length: u16,
    /// Packet checksum, unused here.
    pub checksum: u16,
    /// `E1000_RXD_STAT_*` — `DD` and `EOP` are what the core reads.
    pub status: u8,
    /// `E1000_RXD_ERR_*` — non-zero means the frame is not delivered.
    pub errors: u8,
    /// VLAN tag, unused here.
    pub special: u16,
}

/// A legacy transmit descriptor.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TxDesc {
    /// Device address of the frame bytes — the client's own slot.
    pub addr: u64,
    /// Frame length in bytes.
    pub length: u16,
    /// Checksum offset, unused (no offload).
    pub cso: u8,
    /// `E1000_TXD_CMD_*` — `EOP | IFCS | RS`, and never `DEXT`.
    pub cmd: u8,
    /// `E1000_TXD_STAT_*` — the device writes `DD` (and `EC`/`LC`) here.
    pub status: u8,
    /// Checksum start, unused.
    pub css: u8,
    /// VLAN tag, unused.
    pub special: u16,
}

/// Bytes per descriptor, both kinds.
pub const DESC_BYTES: usize = 16;
/// Byte offset of `TxDesc::status` inside its descriptor — the one byte the
/// reaper reads through the shared-DMA seam.
pub const TXD_STATUS_AT: usize = offset_of!(TxDesc, status);

const _: () = assert!(size_of::<RxDesc>() == DESC_BYTES);
const _: () = assert!(size_of::<TxDesc>() == DESC_BYTES);
const _: () = assert!(TXD_STATUS_AT == 12);
const _: () = assert!(offset_of!(RxDesc, status) == 12 && offset_of!(RxDesc, length) == 8);

/// The command a transmitted frame carries: end of packet, insert the FCS,
/// report status (write `DD` back — the reaper depends on it).
pub const TXD_CMD_FRAME: u8 =
    E1000_TXD_CMD_EOP as u8 | E1000_TXD_CMD_IFCS as u8 | E1000_TXD_CMD_RS as u8;
const _: () = assert!(
    TXD_CMD_FRAME & E1000_TXD_CMD_DEXT as u8 == 0,
    "legacy, never extended"
);

impl TxDesc {
    /// A descriptor for `length` bytes at device address `addr`.
    pub const fn frame(addr: u64, length: u16) -> Self {
        Self {
            addr,
            length,
            cso: 0,
            cmd: TXD_CMD_FRAME,
            status: 0,
            css: 0,
            special: 0,
        }
    }

    /// Whether the device reported the frame written back.
    pub const fn done(&self) -> bool {
        self.status & E1000_TXD_STAT_DD as u8 != 0
    }

    /// Whether the write-back reported a transmit failure (excess or late
    /// collisions — impossible on a full-duplex link, reported honestly if
    /// the link is not one).
    pub const fn failed(&self) -> bool {
        self.status & (E1000_TXD_STAT_EC as u8 | E1000_TXD_STAT_LC as u8) != 0
    }
}

impl RxDesc {
    /// An empty descriptor pointing at `addr`, ready to post: every other
    /// field zero, `DD` clear.
    pub const fn empty(addr: u64) -> Self {
        Self {
            addr,
            length: 0,
            checksum: 0,
            status: 0,
            errors: 0,
            special: 0,
        }
    }

    /// Whether the device reported a whole, error-free frame.
    pub const fn delivers(&self) -> bool {
        self.status & E1000_RXD_STAT_EOP as u8 != 0 && self.errors == 0
    }
}

/// No in-entry cycle bit: the tail pointer is the publish signal.
impl SqEntry for TxDesc {}

/// Valid when the device set `DD`; the ring's phase is not consulted (see
/// the module doc).
impl CqEntry for RxDesc {
    fn valid(&self, _phase: bool) -> bool {
        self.status & E1000_RXD_STAT_DD as u8 != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_layouts_are_the_datasheets() {
        assert_eq!(size_of::<RxDesc>(), 16);
        assert_eq!(size_of::<TxDesc>(), 16);
        assert_eq!(offset_of!(TxDesc, cmd), 11);
        assert_eq!(offset_of!(RxDesc, errors), 13);
        assert_eq!(TxDesc::frame(0x1000, 60).cmd, 0x0b);
    }

    #[test]
    fn validity_is_the_done_bit_whatever_the_phase() {
        let mut d = RxDesc::empty(0x2000);
        assert!(!d.valid(true) && !d.valid(false));
        d.status = E1000_RXD_STAT_DD as u8 | E1000_RXD_STAT_EOP as u8;
        assert!(d.valid(true) && d.valid(false));
        assert!(d.delivers());
        d.errors = E1000_RXD_ERR_CE as u8;
        assert!(d.valid(true) && !d.delivers(), "done, but not deliverable");
    }
}
