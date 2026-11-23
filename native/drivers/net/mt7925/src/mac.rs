//! **M2-regd, MAC init** — `mt7925_mac_init` (`mt7925/init.c:78`) and the
//! two `mt792x_mac_init_band` calls it makes (`mt792x_mac.c:290`).
//!
//! Register writes only; the firmware is not asked anything. Every address
//! and value here is pinned against the 2026-11-20 capture of `mt7925e` on
//! the Framework 13 (`tools/mt7925_capture.bt`): the test seeds each
//! register with the value Linux *read* from it and asserts the writes
//! Linux *made*, in Linux's order, through this driver's own routing.

use kore_driver::seam::Regs;
use kore_driver::wait::Budget;

use crate::d0::Core;

// ── Registers (`mt7925/regs.h`, `mt792x_regs.h`) ─────────────────────────

/// `MT_MDP_DCR0` (`mt7925/regs.h:12`).
pub const MT_MDP_DCR0: u32 = 0x820c_c800;
/// `MT_MDP_DCR0_DAMSDU_EN` — hardware A-MSDU de-aggregation.
pub const MT_MDP_DCR0_DAMSDU_EN: u32 = 1 << 15;
/// `MT_MDP_DCR1` (`mt7925/regs.h:16`).
pub const MT_MDP_DCR1: u32 = 0x820c_c804;
/// `MT_MDP_DCR1_MAX_RX_LEN` — `GENMASK(15, 3)`.
pub const MT_MDP_DCR1_MAX_RX_LEN: u32 = 0x0000_fff8;

/// `MT7925_WTBL_UPDATE` = `MT_WTBLON_TOP(0x380)` (`mt7925/regs.h:106`).
pub const MT7925_WTBL_UPDATE: u32 = 0x820d_4380;
/// `MT_WTBL_UPDATE_WLAN_IDX` — `GENMASK(11, 0)`.
pub const MT_WTBL_UPDATE_WLAN_IDX: u32 = 0x0000_0fff;
/// `MT_WTBL_UPDATE_ADM_COUNT_CLEAR` — `BIT(14)`.
pub const MT_WTBL_UPDATE_ADM_COUNT_CLEAR: u32 = 1 << 14;
/// `MT_WTBL_UPDATE_BUSY` — `BIT(31)` (`mt792x_regs.h:162`).
pub const MT_WTBL_UPDATE_BUSY: u32 = 1 << 31;
/// `MT792x_WTBL_SIZE` (`mt792x.h:18`): entries whose ADM counters clear.
pub const WTBL_SIZE: u32 = 20;

/// `MT_WTBL_ITCR` / `ITDR0` / `ITDR1` (`mt792x_regs.h:164-168`).
pub const MT_WTBL_ITCR: u32 = 0x820d_43b0;
pub const MT_WTBL_ITDR0: u32 = 0x820d_43b8;
pub const MT_WTBL_ITDR1: u32 = 0x820d_43bc;
pub const MT_WTBL_ITCR_WR: u32 = 1 << 16;
pub const MT_WTBL_ITCR_EXEC: u32 = 1 << 31;
/// `MT_WTBL_SPE_IDX_SEL` — "use wtbl spe idx".
pub const MT_WTBL_SPE_IDX_SEL: u32 = 1 << 6;

/// **Where the basic-rate table starts: 11, as this laptop's kernel wrote
/// it** (`ITCR = 0x8001000b` for the first rate).
///
/// Current upstream says 14 (`MT792x_BASIC_RATES_TBL`, `mt792x.h:44`). The
/// capture is of the kernel that drives this part today and works, so the
/// measured value wins, and this note is the place a reader finds out why
/// the two disagree.
pub const BASIC_RATES_TBL: u32 = 11;

/// `mt76_rates` (`mac80211.c`) as `mt7925_mac_init_basic_rates` encodes
/// them: `MT_TX_RATE_MODE` (bits 9:6) from the PHY type, `MT_TX_RATE_IDX`
/// from the hardware index. Four CCK (mode 0), then eight OFDM (mode 1) in
/// mt76's order: 6, 9, 12, 18, 24, 36, 48, 54 Mb/s.
pub const BASIC_RATES: [u32; 12] = [
    0x00, 0x01, 0x02, 0x03, // CCK 1, 2, 5.5, 11
    0x4b, 0x4f, 0x4a, 0x4e, 0x49, 0x4d, 0x48, 0x4c, // OFDM
];

/// The per-band LMAC blocks (`mt792x_regs.h`), band 0 then band 1.
const fn tmac(band: u32) -> u32 {
    if band == 0 { 0x820e_4000 } else { 0x820f_4000 }
}
const fn rmac(band: u32) -> u32 {
    if band == 0 { 0x820e_5000 } else { 0x820f_5000 }
}
const fn wf_dma(band: u32) -> u32 {
    if band == 0 { 0x820e_7000 } else { 0x820f_7000 }
}
const fn wtbloff(band: u32) -> u32 {
    if band == 0 { 0x820e_9000 } else { 0x820f_9000 }
}
const fn mib(band: u32) -> u32 {
    if band == 0 { 0x820e_d000 } else { 0x820f_d000 }
}

/// How long one WTBL update may stay busy: `mt76_poll(..., 5000)` µs.
pub const WTBL_BUDGET_NS: u64 = 5_000_000;
/// The unclocked fallback for [`WTBL_BUDGET_NS`].
pub const WTBL_SPINS_UNCLOCKED: u32 = 5_000;

/// Why MAC init stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MacRefusal {
    /// A WTBL update never cleared its busy bit.
    ///
    /// **Stricter than upstream**, which ignores `mt76_poll`'s result here.
    /// A table entry still being written when the next write lands is a
    /// state nothing in the driver can name, so this stops instead.
    WtblBusy { idx: u32, reg: u32 },
}

impl<R: Regs + ?Sized> Core<'_, R> {
    /// `mt76_rmw`: read, replace `mask` with `set`, write — always written,
    /// even when unchanged, as upstream's is and as the capture shows.
    fn rmw(&self, addr: u32, mask: u32, set: u32) {
        let v = self.read(addr);
        self.write(addr, (v & !mask) | set);
    }

    /// **The MAC's own setup, before any channel exists.**
    ///
    /// `mt7925_mac_init`, in its order:
    /// 1. MDP: maximum receive length 1536, hardware de-A-MSDU on.
    /// 2. Clear the admission counters of all twenty WTBL entries, waiting
    ///    out each update's busy bit.
    /// 3. `mt792x_mac_init_band` for band 0 and band 1.
    /// 4. The twelve basic rates into the fixed-rate table.
    pub fn mac_init(&self, mut delay: impl FnMut(), now_ns: fn() -> Option<u64>) -> Result<(), MacRefusal> {
        self.rmw(MT_MDP_DCR1, MT_MDP_DCR1_MAX_RX_LEN, 1536 << 3);
        self.rmw(MT_MDP_DCR0, 0, MT_MDP_DCR0_DAMSDU_EN);

        for idx in 0..WTBL_SIZE {
            self.rmw(
                MT7925_WTBL_UPDATE,
                MT_WTBL_UPDATE_WLAN_IDX,
                idx | MT_WTBL_UPDATE_ADM_COUNT_CLEAR,
            );
            let mut budget = Budget::new(WTBL_BUDGET_NS, WTBL_SPINS_UNCLOCKED, now_ns);
            loop {
                let reg = self.read(MT7925_WTBL_UPDATE);
                if reg & MT_WTBL_UPDATE_BUSY == 0 {
                    break;
                }
                if !budget.may_wait() {
                    return Err(MacRefusal::WtblBusy { idx, reg });
                }
                delay();
            }
        }

        for band in 0..2 {
            self.mac_init_band(band);
        }

        for (i, &rate) in BASIC_RATES.iter().enumerate() {
            self.write(MT_WTBL_ITDR0, rate);
            self.write(MT_WTBL_ITDR1, MT_WTBL_SPE_IDX_SEL);
            self.write(MT_WTBL_ITCR, MT_WTBL_ITCR_WR | MT_WTBL_ITCR_EXEC | (BASIC_RATES_TBL + i as u32));
        }
        Ok(())
    }

    /// `mt792x_mac_init_band` (`mt792x_mac.c:290-316`).
    fn mac_init_band(&self, band: u32) {
        // TMAC_CTCR0: DDLMT reference time 0x3f, then DDLMT + VHT SMPDU on.
        self.rmw(tmac(band) + 0x0f4, 0x3f, 0x3f);
        self.rmw(tmac(band) + 0x0f4, 0, (1 << 18) | (1 << 17));
        // RMAC MIB time and airtime: receive time accounting on.
        self.rmw(rmac(band) + 0x3c4, 0, 1 << 30);
        self.rmw(rmac(band) + 0x380, 0, 1 << 30);
        // MIB_SCR1: TX then RX duration reporting.
        self.rmw(mib(band) + 0x004, 0, 1 << 8);
        self.rmw(mib(band) + 0x004, 0, 1 << 9);
        // DMA_DCR0: maximum receive length 1536; the RX rate report off
        // ("due to hw issues", upstream's words).
        self.rmw(wf_dma(band), 0x0000_fff8, 1536 << 3);
        self.rmw(wf_dma(band), 1 << 23, 0);
        // WTBLOFF_TOP_RSCR: RCPI mode 0, parameter 3.
        self.rmw(wtbloff(band) + 0x008, (0b11 << 30) | (0b11 << 24), 3 << 24);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MockMt7925;
    use crate::regs::{DEFAULT_FIXED_MAP, RegMap};
    use alloc::vec;
    use alloc::vec::Vec;

    /// The chip address a captured BAR offset names, through upstream's
    /// fixed map: the capture logs BAR offsets, the mock logs chip
    /// addresses, and this is the one table both agree on.
    fn chip(bar: u32) -> u32 {
        for &RegMap { phys, maps, size } in DEFAULT_FIXED_MAP {
            if bar >= maps && bar - maps < size {
                return phys + (bar - maps);
            }
        }
        panic!("{bar:#x} is in no fixed block");
    }

    /// `(register, value Linux read)` for each register MAC init touches,
    /// from the capture.
    const SEEDS: &[(u32, u32)] = &[
        (0x0e804, 0x0001_3400),
        (0x0e800, 0x000b_c200),
        (0x34380, 0x0500_0013),
        (0x210f4, 0x00ef_3210),
        (0x217c4, 0x0000_0000),
        (0x21780, 0x000a_0000),
        (0x24804, 0x00f8_c310),
        (0x21e00, 0x0277_3400),
        (0x23408, 0x0000_0001),
        (0xa10f4, 0x00ef_3210),
        (0xa17c4, 0x0000_0000),
        (0xa1780, 0x000a_0000),
        (0xa4804, 0x00f8_c310),
        (0xa1e00, 0x0277_3400),
        (0xa3408, 0x0000_0001),
    ];

    /// **The writes Linux made, in Linux's order** — `mt7925_mac_init` on
    /// this laptop, 2026-11-20, interrupt bookkeeping removed.
    fn captured() -> Vec<(u32, u32)> {
        let mut w = vec![(0x0e804, 0x0001_3000), (0x0e800, 0x000b_c200)];
        for idx in 0..20u32 {
            w.push((0x34380, 0x0500_4000 | idx));
        }
        for base in [0x0002_0000u32, 0x000a_0000] {
            w.extend([
                (base + 0x10f4, 0x00ef_323f),
                (base + 0x10f4, 0x00ef_323f),
                (base + 0x17c4, 0x4000_0000),
                (base + 0x1780, 0x400a_0000),
                (base + 0x4804, 0x00f8_c310),
                (base + 0x4804, 0x00f8_c310),
                (base + 0x1e00, 0x0277_3000),
                (base + 0x1e00, 0x0277_3000),
                (base + 0x3408, 0x0300_0001),
            ]);
        }
        let rates = [0x00, 0x01, 0x02, 0x03, 0x4b, 0x4f, 0x4a, 0x4e, 0x49, 0x4d, 0x48, 0x4c];
        for (i, r) in rates.into_iter().enumerate() {
            w.extend([(0x343b8, r), (0x343bc, 0x40), (0x343b0, 0x8001_000b + i as u32)]);
        }
        w.into_iter().map(|(bar, v)| (chip(bar), v)).collect()
    }

    #[test]
    fn mac_init_writes_what_linux_wrote() {
        let dev = MockMt7925::new();
        for &(bar, v) in SEEDS {
            dev.poke(dev.cell_of(chip(bar)), v);
        }
        dev.writes.borrow_mut().clear();
        Core::new(&dev).mac_init(|| {}, || None).unwrap();
        let got: Vec<(u32, u32)> = dev.writes.borrow().clone();
        assert_eq!(got, captured());
    }

    /// A WTBL entry that never stops being busy refuses, naming the entry,
    /// rather than writing the next one over it.
    #[test]
    fn a_stuck_wtbl_update_refuses() {
        let dev = MockMt7925::new();
        dev.poke(dev.cell_of(MT7925_WTBL_UPDATE), MT_WTBL_UPDATE_BUSY);
        let out = Core::new(&dev).mac_init(|| {}, || None);
        assert!(matches!(out, Err(MacRefusal::WtblBusy { idx: 0, .. })), "{out:?}");
        let wtbl_writes = dev.writes.borrow().iter().filter(|(a, _)| *a == dev.cell_of(MT7925_WTBL_UPDATE)).count();
        assert_eq!(wtbl_writes, 1, "a second entry was written over a busy one");
    }
}
