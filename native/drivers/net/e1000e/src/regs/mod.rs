//! The 82574L register face.
//!
//! [`i82574`] is **generated** from `tools/i82574.h`, the datasheet
//! transcription — do not hand-edit it; change the header and regenerate
//! (the command is in the header's comment). This file is the hand-written
//! half: the compositions the core programs, each with the derivation the
//! number came from, and the pool-free decode probe.

pub mod i82574;

use kore_driver::seam::Regs;

use i82574::*;

/// The queue this core drives. The 82574L has two; steering into the
/// second is `res/PEER_LINK.md` P6's work and stays unadvertised until the
/// frame engine consumes it.
pub const QUEUE: u32 = 0;

/// Bytes of BAR0 the core reads or writes: the highest register it touches
/// is `RAH(0)` at `0x5404`, rounded to the 8 KiB block boundary above it so a
/// window that decodes the whole CSR range this driver uses is the test,
/// not a window that merely exists. The 82574L's BAR0 is 128 KiB.
pub const REGS_SPAN: usize = 0x6000;

/// A generated `u32` offset as the `usize` the [`Regs`] seam takes.
#[inline]
pub const fn reg(offset: u32) -> usize {
    offset as usize
}

/// Every cause the core enables routed to MSI-X table entry 0 — the one
/// message `kore_driver::host::InterruptBlock::SINGLE` realizes (the host
/// module is behind `kore-driver`'s `service` feature, which the row
/// enables and this core does not) — with a valid bit per field, plus a
/// transmit interrupt on every write-back (bit 31). A field left without
/// its valid bit is a cause that never fires.
pub const IVAR_ALL_TO_ENTRY0: u32 = (E1000_IVAR_INT_ALLOC_VALID << E1000_IVAR_RXQ0_SHIFT)
    | (E1000_IVAR_INT_ALLOC_VALID << E1000_IVAR_TXQ0_SHIFT)
    | (E1000_IVAR_INT_ALLOC_VALID << E1000_IVAR_OTHER_SHIFT)
    | E1000_IVAR_TX_INT_EVERY_WB;

/// The interrupt mask the core arms: the MSI-X queue causes **and** the
/// legacy causes, because a `PreferMsiX` claim may realize plain MSI, in
/// which mode the controller ignores `IVAR` and raises only the legacy
/// bits. Arming both costs nothing (a cause the mode never raises is a bit
/// that never fires) and keeps one code path for both realizations.
pub const IMS_ARMED: u32 = E1000_ICR_RXQ0
    | E1000_ICR_TXQ0
    | E1000_ICR_OTHER
    | E1000_ICR_RXT0
    | E1000_ICR_TXDW
    | E1000_ICR_LSC
    | E1000_ICR_RXO;

/// The causes the controller auto-clears from `ICR` (and, with `IAME`
/// clear, auto-masks from `IMS`) when it delivers their MSI-X message.
/// Every drive re-arms `IMS`, so the auto-mask is a race-free acknowledge
/// rather than a lost wake.
pub const EIAC_ARMED: u32 = E1000_ICR_RXQ0 | E1000_ICR_TXQ0 | E1000_ICR_OTHER;
const _: () = assert!(
    EIAC_ARMED & !E1000_EIAC_MASK_82574 == 0,
    "only the queue and other causes may auto-clear on the 82574"
);

/// The interrupt-moderation interval the core programs, in nanoseconds:
/// the controller raises at most one message per interval however many
/// causes latch inside it (the causes stay latched in `ICR`, and the next
/// message reports them all).
///
/// Derived from the sweep, not tuned: at line rate the smallest frames
/// arrive one per ≈ 0.67 µs, so 50 µs of them is ≈ 75 descriptors — above
/// the engine's `RX_BATCH` (64), so a wake never finds a batch it cannot
/// fill under load, and under a fifth of the 255 posted receive slots, so
/// a moderated wake cannot arrive to an overrun. It is also where Linux's
/// e1000e starts its dynamic ladder (20 000 interrupts/s, "low latency"),
/// which is a second, independent reading of the same trade. Moving it up
/// buys fewer interrupts and costs latency on a quiet link; moving it down
/// past ≈ 40 µs lets a saturated link wake the row for part-batches.
pub const ITR_INTERVAL_NS: u32 = 50_000;
/// [`ITR_INTERVAL_NS`] in the register's 256 ns units — 195 — written to
/// both `ITR` (legacy/MSI delivery) and `EITR[0]` (the one MSI-X vector the
/// core routes everything to), so whichever mode the claim realized is
/// moderated by the same interval.
pub const ITR_INTERVAL: u32 = ITR_INTERVAL_NS / 256;
const _: () = assert!(
    ITR_INTERVAL != 0 && ITR_INTERVAL & !E1000_ITR_INTERVAL_MASK == 0,
    "the moderation interval is non-zero and fits its 16-bit field"
);

/// `TCTL` as the core runs it: enabled, short packets padded, re-transmit
/// on late collision, and the datasheet's default collision threshold and
/// distance — every field written explicitly so the register is the same
/// on every boot rather than whatever the previous owner left.
pub const TCTL_RUN: u32 = E1000_TCTL_EN
    | E1000_TCTL_PSP
    | E1000_TCTL_RTLC
    | (E1000_COLLISION_THRESHOLD << E1000_CT_SHIFT)
    | (E1000_COLLISION_DISTANCE << E1000_COLD_SHIFT);
const _: () = assert!(
    (E1000_COLLISION_THRESHOLD << E1000_CT_SHIFT) & !E1000_TCTL_CT_MASK == 0
        && (E1000_COLLISION_DISTANCE << E1000_COLD_SHIFT) & !E1000_TCTL_COLD_MASK == 0,
    "the collision defaults fit their fields"
);

/// `RCTL` as the core runs it, before the packet-filter bits are added:
/// enabled, CRC stripped (so a 1514-byte frame arrives as 1514 bytes and
/// not 1518 — the classifier refuses anything over `FRAME_MAX`), legacy
/// descriptors (`DTYP` = 0), 2048-byte buffers (`BSIZE` = 0 with `BSEX`
/// clear, which is [`RX_BUF_BYTES`](crate::nic::RX_BUF_BYTES)).
pub const RCTL_RUN: u32 = E1000_RCTL_EN | E1000_RCTL_SECRC;

/// Descriptor write-back after every descriptor, thresholds in descriptors:
/// a single received frame must land immediately, not after a batch the
/// controller is still waiting to fill.
pub const RXDCTL_RUN: u32 = E1000_RXDCTL_GRAN | E1000_RXDCTL_WTHRESH_1;
/// The transmit twin: a completion is written back per descriptor, which
/// is what "the head is the release" needs.
pub const TXDCTL_RUN: u32 = E1000_TXDCTL_GRAN | E1000_TXDCTL_WTHRESH_1;

/// Does this BAR hold a live, decodable controller? The **pool-free
/// candidate probe** (`res/DEVICE_SERVICE.md` §10): decided from two
/// register reads over a temporary mapping, before the row spends its one
/// DMA supply on the function.
///
/// A power-gated or unrouted function reads all-ones. Zero is *not*
/// rejected here: `STATUS` is legitimately zero on a link-down part before
/// bring-up, and the exact PCI identity the row's selector demands is what
/// rules out an absent decode.
pub fn controller_present(regs: &impl Regs, bar_len: usize) -> bool {
    if bar_len < REGS_SPAN {
        return false;
    }
    let ctrl = regs.read32(reg(E1000_CTRL));
    let status = regs.read32(reg(E1000_STATUS));
    ctrl != u32::MAX && status != u32::MAX
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A register double answering fixed words at fixed offsets — the
    /// cheapest possible probe test.
    struct Fixed {
        ctrl: u32,
        status: u32,
    }

    impl Regs for Fixed {
        fn read32(&self, off: usize) -> u32 {
            match off {
                0x0 => self.ctrl,
                0x8 => self.status,
                _ => 0,
            }
        }
        fn write32(&self, _off: usize, _val: u32) {}
    }

    #[test]
    fn a_decoding_bar_is_recognized_and_an_all_ones_or_short_one_is_refused() {
        let live = Fixed {
            ctrl: 0x0000_0040,
            status: 0x0000_0083,
        };
        assert!(controller_present(&live, REGS_SPAN));
        assert!(controller_present(&live, 0x20000), "the whole 128 KiB BAR");
        assert!(
            !controller_present(&live, REGS_SPAN - 4),
            "a window short of the last register the core touches"
        );
        let gated = Fixed {
            ctrl: u32::MAX,
            status: u32::MAX,
        };
        assert!(
            !controller_present(&gated, REGS_SPAN),
            "all-ones is no decode"
        );
        let dark = Fixed { ctrl: 0, status: 0 };
        assert!(
            controller_present(&dark, REGS_SPAN),
            "zero status is a link-down part, not an absent one"
        );
    }

    #[test]
    fn the_compositions_stay_inside_their_fields() {
        assert_eq!(IVAR_ALL_TO_ENTRY0 & 0x000F_FFFF, 0x0008_0808);
        assert_ne!(IVAR_ALL_TO_ENTRY0 & E1000_IVAR_TX_INT_EVERY_WB, 0);
        assert_eq!(TCTL_RUN & E1000_TCTL_CT_MASK, 15 << 4);
        assert_eq!(TCTL_RUN & E1000_TCTL_COLD_MASK, 63 << 12);
        assert_eq!(RCTL_RUN & E1000_RCTL_BSIZE_MASK, 0, "2048-byte buffers");
        assert_eq!(RCTL_RUN & E1000_RCTL_BSEX, 0);
        assert_eq!(RCTL_RUN & E1000_RCTL_DTYP_MASK, 0, "legacy descriptors");
        assert_eq!(reg(e1000_tdt(QUEUE)), 0x3818);
        assert_eq!(
            reg(e1000_rdt(1)),
            0x2918,
            "the second queue is one stride up"
        );
        assert_eq!(reg(e1000_rah(0)), 0x5404);
        assert_eq!(
            reg(e1000_eitr_82574(0)),
            0xe8,
            "vector 0's throttle register"
        );
        assert_eq!(ITR_INTERVAL, 195, "50 µs in 256 ns units");
    }
}
