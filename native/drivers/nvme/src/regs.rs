//! NVMe controller register layout + `CAP` parse — **spec facts** (NVMe
//! base spec §3.1), the stable core every NVMe driver shares.  Ported from
//! `os0/user/dev/block/nvme/controller.rs` (the same constants; the
//! reference is a working driver).
//!
//! BAR0 layout:
//!
//! | Offset  | Reg   | Description                          |
//! |---------|-------|--------------------------------------|
//! | 0x00    | CAP   | Controller Capabilities (64-bit)     |
//! | 0x08    | VS    | Version                              |
//! | 0x14    | CC    | Controller Configuration             |
//! | 0x1C    | CSTS  | Controller Status                    |
//! | 0x24    | AQA   | Admin Queue Attributes               |
//! | 0x28    | ASQ   | Admin SQ Base Address (64-bit)       |
//! | 0x30    | ACQ   | Admin CQ Base Address (64-bit)       |
//! | 0x1000+ | DBs   | SQ/CQ doorbells (stride 4 << DSTRD)  |

/// Controller Capabilities (64-bit).
pub const REG_CAP: usize = 0x00;
/// Version.
pub const REG_VS: usize = 0x08;
/// Controller Configuration.
pub const REG_CC: usize = 0x14;
/// Controller Status.
pub const REG_CSTS: usize = 0x1C;
/// Admin Queue Attributes.
pub const REG_AQA: usize = 0x24;
/// Admin SQ Base Address (64-bit).
pub const REG_ASQ: usize = 0x28;
/// Admin CQ Base Address (64-bit).
pub const REG_ACQ: usize = 0x30;
/// Base of the doorbell region.
pub const REG_DOORBELL_BASE: usize = 0x1000;

// ── Controller Configuration (CC) fields ────────────────────────────────

/// Enable.
pub const CC_EN: u32 = 1 << 0;
/// I/O Command Set: NVM.
pub const CC_CSS_NVM: u32 = 0 << 4;
/// Memory Page Size 4 KiB (`2^(12+MPS)`, MPS = 0).
pub const CC_MPS_4K: u32 = 0 << 7;
/// Arbitration: round-robin.
pub const CC_AMS_RR: u32 = 0 << 11;
/// I/O SQ entry size 64 B (`2^6`).
pub const CC_IOSQES_64: u32 = 6 << 16;
/// I/O CQ entry size 16 B (`2^4`).
pub const CC_IOCQES_16: u32 = 4 << 20;

/// The enable word the core writes to `CC` for a configured `mps`
/// (`MPS` field, bits 10:7): NVM command set, round-robin, the spec
/// entry sizes, enabled.  `mps` must lie within the controller's
/// advertised `[MPSMIN, MPSMAX]` — the core passes `MPSMIN` (the
/// smallest page the controller accepts; a `MPSMIN > 0` controller is
/// thereby handled correctly, where a hardcoded 4 KiB would be an
/// illegal configuration).
pub const fn cc_enable(mps: u8) -> u32 {
    CC_EN | CC_CSS_NVM | ((mps as u32) << 7) | CC_AMS_RR | CC_IOSQES_64 | CC_IOCQES_16
}

/// The `mps = 0` (4 KiB) enable word — the spec-floor configuration.
pub const CC_ENABLE: u32 = cc_enable(0);

// ── Shutdown Notification (CC bits 15:14) ───────────────────────────────

/// Shutdown Notification field mask.
pub const CC_SHN_MASK: u32 = 0b11 << 14;
/// Normal shutdown notification (flush, then report SHST complete).
pub const CC_SHN_NORMAL: u32 = 0b01 << 14;

// ── Controller Status (CSTS) fields ─────────────────────────────────────

/// Ready.
pub const CSTS_RDY: u32 = 1 << 0;
/// Controller Fatal Status.
pub const CSTS_CFS: u32 = 1 << 1;
/// Shutdown Status field mask (bits 3:2).
pub const CSTS_SHST_MASK: u32 = 0b11 << 2;
/// Shutdown processing complete.
pub const CSTS_SHST_COMPLETE: u32 = 0b10 << 2;

/// Parsed `CAP` register — the controller's advertised limits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capabilities {
    /// Maximum Queue Entries Supported (0-based; real max is `+1`).
    pub mqes: u16,
    /// Contiguous Queues Required.
    pub cqr: bool,
    /// Controller ready timeout, in the register's 500 ms units.
    pub timeout_500ms: u8,
    /// Doorbell Stride (`4 << dstrd` bytes).
    pub dstrd: u8,
    /// NVM Subsystem Reset Supported.
    pub nssrs: bool,
    /// Memory Page Size Minimum (`2^(12+mpsmin)`).
    pub mpsmin: u8,
    /// Memory Page Size Maximum (`2^(12+mpsmax)`).
    pub mpsmax: u8,
}

impl Capabilities {
    /// Parse from a raw `CAP` value.
    pub const fn from_cap(cap: u64) -> Self {
        Self {
            mqes: (cap & 0xFFFF) as u16,
            cqr: (cap >> 16) & 1 != 0,
            timeout_500ms: ((cap >> 24) & 0xFF) as u8,
            dstrd: ((cap >> 32) & 0xF) as u8,
            nssrs: (cap >> 36) & 1 != 0,
            mpsmin: ((cap >> 48) & 0xF) as u8,
            mpsmax: ((cap >> 52) & 0xF) as u8,
        }
    }

    /// Doorbell stride in bytes.
    pub const fn doorbell_stride(&self) -> usize {
        4 << self.dstrd
    }

    /// `CAP.TO` as nanoseconds. The NVMe base specification defines the raw
    /// field directly in 500 ms units; this is the worst-case `CSTS.RDY`
    /// transition time after changing `CC.EN`.
    pub const fn ready_timeout_ns(&self) -> u64 {
        self.timeout_500ms as u64 * 500_000_000
    }

    /// Byte offset of the SQ tail doorbell for queue `qid`.
    pub const fn sq_doorbell(&self, qid: u16) -> usize {
        REG_DOORBELL_BASE + (2 * qid as usize) * self.doorbell_stride()
    }

    /// Byte offset of the CQ head doorbell for queue `qid`.
    pub const fn cq_doorbell(&self, qid: u16) -> usize {
        REG_DOORBELL_BASE + (2 * qid as usize + 1) * self.doorbell_stride()
    }

    /// The real maximum queue depth (`mqes + 1`).
    ///
    /// Returns `u32` because the answer does not fit in `u16`: `CAP.MQES` is
    /// 0-based, so its maximum legal value `0xFFFF` means **65536** entries.
    /// Use [`Self::max_queue_depth`] to clamp it against a queue depth —
    /// casting this to `u16` wraps the largest legal controller to zero.
    pub const fn max_queue_entries(&self) -> u32 {
        self.mqes as u32 + 1
    }

    /// The controller's queue-entry ceiling, saturated into the `u16` a queue
    /// depth is expressed in.
    ///
    /// `mqes + 1` is exactly one larger than `u16::MAX` on a controller
    /// advertising the maximal legal `CAP.MQES`, so `max_queue_entries() as
    /// u16` wraps it to `0` — and every queue constructor then refuses that
    /// depth as [`QpairError::BadDepth`](crate::QpairError::BadDepth). A real
    /// disk shipped that value (Micron 2500, `1344:5425`), which no emulated
    /// controller in the test matrix advertises, so the wrap only ever
    /// appeared on metal: the drive was rejected for being *too capable*.
    ///
    /// Saturating is not a compromise here — a depth is always clamped
    /// against a driver maximum orders of magnitude below 65535, so the
    /// distinction between 65535 and 65536 available entries cannot change
    /// any caller's result.
    pub const fn max_queue_depth(&self) -> u16 {
        // Deliberately not `max_queue_entries() as u16`: that is the bug.
        if self.mqes == u16::MAX {
            u16::MAX
        } else {
            self.mqes + 1
        }
    }
}

/// Does this BAR hold a live, decodable NVMe controller?  The pool-free
/// **candidate probe** (`res/DEVICE_SERVICE.md` §10) — the disk sibling of
/// `kore_xhci::regs::connected_ports`: a claiming driver runs it over a
/// [`DeviceProbe`](../seam) BAR before spending its one DMA supply on
/// bring-up, so a dead function is released back to the broker's ledger
/// instead of failing the whole founding.
///
/// A power-gated (D3) or unrouted function reads all-ones; an absent decode
/// reads zeros.  Both are rejected, as is a BAR too short to reach the
/// doorbell region every operable controller needs, and a `CAP` whose
/// queue-depth floor (`MQES`) the spec forbids.
pub fn controller_present(regs: &impl crate::seam::Regs, bar_len: usize) -> bool {
    // The register file through ACQ plus the first doorbell pair is the
    // minimum any operable controller decodes.
    if bar_len < REG_DOORBELL_BASE + 8 {
        return false;
    }
    let vs = regs.read32(REG_VS);
    if vs == 0 || vs == u32::MAX {
        return false;
    }
    let cap = regs.read64(REG_CAP);
    if cap == 0 || cap == u64::MAX {
        return false;
    }
    // MQES is 0-based with a spec minimum of 1 (two entries); a zero here
    // is a decode artifact, not a controller.
    Capabilities::from_cap(cap).mqes >= 1
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A controller advertising the maximal legal `CAP.MQES` must not be
    /// rejected for it. `mqes + 1` is 65536 there — one past `u16` — so the
    /// old `max_queue_entries() as u16` wrapped to a depth of `0`, which
    /// `Qpair::new` refuses as `BadDepth`. That is the whole of the Micron
    /// 2500 (`1344:5425`) metal bring-up failure: the drive was refused for
    /// advertising the largest queue the spec allows, and no emulated
    /// controller in the matrix advertises it, so only metal ever saw it.
    #[test]
    fn a_maximal_mqes_does_not_wrap_to_a_zero_queue_depth() {
        let c = Capabilities::from_cap(0x0000_0000_0000_FFFF);
        assert_eq!(c.mqes, 0xFFFF);
        // The exact count still does not fit a u16, and still must not be cast.
        assert_eq!(c.max_queue_entries(), 65_536);
        assert_eq!(c.max_queue_depth(), u16::MAX);
        // What the admin queue actually computes.
        assert_eq!(32u16.min(c.max_queue_depth()), 32);
        // The bug, pinned: this is what the two callsites used to do.
        assert_eq!(c.max_queue_entries() as u16, 0);
    }

    /// Saturation must not disturb ordinary controllers, including the
    /// boundary one entry below the wrap.
    #[test]
    fn max_queue_depth_matches_the_entry_count_below_the_wrap() {
        for mqes in [0u16, 1, 0x3F, 0x7FF, 0xFFFE] {
            let c = Capabilities::from_cap(u64::from(mqes));
            assert_eq!(u32::from(c.max_queue_depth()), c.max_queue_entries());
        }
    }

    #[test]
    fn cap_parses_and_doorbells_stride_correctly() {
        // MQES=0x3F (64 entries), CQR=1 (bit 16), DSTRD=0.
        let cap = 0x0000_0000_0001_003F;
        let c = Capabilities::from_cap(cap);
        assert_eq!(c.mqes, 0x3F);
        assert!(c.cqr);
        assert_eq!(c.timeout_500ms, 0);
        assert_eq!(c.max_queue_entries(), 64);
        assert_eq!(c.doorbell_stride(), 4);
        // qid 0 (admin): SQ at 0x1000, CQ at 0x1004.
        assert_eq!(c.sq_doorbell(0), 0x1000);
        assert_eq!(c.cq_doorbell(0), 0x1004);
        // qid 1 (first I/O): SQ at 0x1008, CQ at 0x100C.
        assert_eq!(c.sq_doorbell(1), 0x1008);
        assert_eq!(c.cq_doorbell(1), 0x100C);

        // A wider stride (DSTRD=2 ⇒ 16-byte stride) spaces doorbells out.
        let wide = Capabilities::from_cap(0x0000_0002_0000_0000);
        assert_eq!(wide.doorbell_stride(), 16);
        assert_eq!(wide.cq_doorbell(0), 0x1000 + 16);

        let timed = Capabilities::from_cap(30u64 << 24);
        assert_eq!(timed.timeout_500ms, 30);
        assert_eq!(timed.ready_timeout_ns(), 15_000_000_000);
    }

    #[test]
    fn cc_enable_places_mps_in_bits_10_7() {
        assert_eq!(cc_enable(0), CC_ENABLE);
        assert_eq!(cc_enable(1) & (0xF << 7), 1 << 7, "MPS=1 (8 KiB)");
        assert_eq!(cc_enable(1) & CC_EN, CC_EN, "still enabled");
    }

    /// The candidate probe's three rejection tiers: a D3/unrouted function
    /// (all-ones), an undecoded BAR (zeros), and a window too short for the
    /// doorbell region — against the one accepting shape.
    #[test]
    fn controller_present_rejects_dead_and_short_candidates() {
        struct Fixed {
            vs: u32,
            cap: u64,
        }
        impl crate::seam::Regs for Fixed {
            fn read32(&self, off: usize) -> u32 {
                match off {
                    REG_VS => self.vs,
                    REG_CAP => self.cap as u32,
                    x if x == REG_CAP + 4 => (self.cap >> 32) as u32,
                    _ => 0,
                }
            }
            fn write32(&self, _: usize, _: u32) {}
        }

        // MQES=0x3F, a real version — the live shape.
        let live = Fixed {
            vs: 0x0001_0400,
            cap: 0x0000_0000_0001_003F,
        };
        assert!(controller_present(&live, 0x2000));
        assert!(
            !controller_present(&live, REG_DOORBELL_BASE),
            "a BAR without the first doorbell pair refuses"
        );

        let dead = Fixed {
            vs: u32::MAX,
            cap: u64::MAX,
        };
        assert!(
            !controller_present(&dead, 0x2000),
            "all-ones is D3, not a device"
        );

        let vacant = Fixed { vs: 0, cap: 0 };
        assert!(!controller_present(&vacant, 0x2000), "zeros decode nothing");

        let zero_mqes = Fixed {
            vs: 0x0001_0400,
            cap: 0x0000_0000_0001_0000,
        };
        assert!(
            !controller_present(&zero_mqes, 0x2000),
            "a spec-illegal MQES floor is a decode artifact"
        );
    }
}
