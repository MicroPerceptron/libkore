//! **D0 — the part answers.**
//!
//! The first rung of `res/NETWORK_RESTART.md`'s lane C/M ladder, and the
//! cheapest one there is: claim BAR0, drive the L1 remap window, read the
//! chip identity back, take driver ownership, and read the firmware-ready
//! state. No firmware moves here — that is D1.
//!
//! # What a pass proves
//!
//! Reading `0x7925` out of [`MT_HW_CHIPID`] proves **four** things at once,
//! which is why it is the rung worth having first on a machine with no serial
//! port and one flash per experiment:
//!
//! 1. the part is powered and decoding its BAR;
//! 2. BAR0 is the register window (`res/MT7925_EVIDENCE.md` §2's open
//!    question);
//! 3. the L1 remap window works — the identity is *not* directly addressable,
//!    so a correct read is a correct translation;
//! 4. the value matches what `mt7925e` printed on this same silicon
//!    (`res/MT7925_EVIDENCE.md` §11), so it is checked against a measurement
//!    rather than against a hope.

use kore_driver::seam::Regs;
use kore_driver::wait::{Budget, handshake_until};

use crate::regs::mt7925::*;
use crate::regs::{Route, l1_window_offset, l1_window_word, route};

/// The identity this part must report.
///
/// `mt7925/pci.c:723` composes the revision as
/// `(rr(MT_HW_CHIPID) << 16) | (rr(MT_HW_REV) & 0xff)`, and on the lab's
/// Framework 13 that printed `0x79250000` — so [`MT_HW_CHIPID`] reads
/// `0x7925`. Measured, not assumed: `res/MT7925_EVIDENCE.md` §11.
pub const CHIP_ID_MT7925: u32 = 0x7925;

/// How long to wait for the part to release ownership.
///
/// Upstream polls `MT_CONN_ON_LPCTL` for `OWN_SYNC` to clear on a 50-tick,
/// 1 ms-per-tick budget (`mt792x_core.c:1050`), so **50 ms** is the vendor's
/// own number rather than one chosen here. The unclocked fallback is 50
/// probes, one per tick upstream would have taken.
pub const OWN_BUDGET_NS: u64 = 50_000_000;
/// Probes to take when the machine's clock cannot answer. See
/// [`OWN_BUDGET_NS`].
pub const OWN_SPINS_UNCLOCKED: u32 = 50;

/// What D0 could not establish.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum D0Refusal {
    /// The identity register answered, but not with this part's identity.
    ///
    /// Carries what was read, because "not an MT7925" and "BAR0 is not
    /// mapped" are different failures and the value tells them apart: a
    /// floating or unmapped window reads all-ones or all-zeroes.
    NotThisPart {
        /// What [`MT_HW_CHIPID`] actually returned.
        read: u32,
    },
    /// The part did not release ownership inside [`OWN_BUDGET_NS`].
    OwnershipTimedOut,
}

/// The D0 face over a mapped BAR0.
pub struct Core<'r, R: Regs + ?Sized> {
    regs: &'r R,
}

impl<'r, R: Regs + ?Sized> Core<'r, R> {
    /// Take the D0 face over an already-claimed BAR0 window.
    pub fn new(regs: &'r R) -> Self {
        Self { regs }
    }

    /// The window this face was built over.
    ///
    /// `pub(crate)` on purpose: the D1 rungs are sibling modules over the
    /// same BAR and need the raw window for a budgeted handshake, but
    /// nothing outside this crate should reach past `read`/`write` and the
    /// routing they carry.
    pub(crate) const fn regs(&self) -> &'r R {
        self.regs
    }

    /// Read one chip-space address, programming whatever window it needs.
    ///
    /// The window programming is upstream's (`mt7925_reg_map_l1`,
    /// `mt7925/pci.c:142`), including the read-back that pushes the write:
    /// the window register and the access that follows it are two
    /// transactions on the same bus, and without the read-back the access can
    /// overtake the window it depends on.
    pub fn read(&self, addr: u32) -> u32 {
        match route(addr) {
            Route::Direct(off) | Route::Fixed(off) => self.regs.read32(off as usize),
            Route::L1 { base, offset } => {
                self.set_l1_window(base);
                self.regs.read32(l1_window_offset(offset) as usize)
            }
            Route::L2(addr) => {
                self.set_l1_window((MT_HIF_REMAP_BASE_L2 >> 16) as u16);
                self.regs.write32(MT_HIF_REMAP_L2 as usize, addr);
                // Read to push the write, exactly as `mt7925_reg_map_l2` does.
                let _ = self.regs.read32(MT_HIF_REMAP_L1 as usize);
                self.regs.read32(MT_HIF_REMAP_BASE_L1 as usize)
            }
        }
    }

    /// Write one chip-space address. Window handling is [`Self::read`]'s.
    pub fn write(&self, addr: u32, val: u32) {
        match route(addr) {
            Route::Direct(off) | Route::Fixed(off) => self.regs.write32(off as usize, val),
            Route::L1 { base, offset } => {
                self.set_l1_window(base);
                self.regs.write32(l1_window_offset(offset) as usize, val);
            }
            Route::L2(addr) => {
                self.set_l1_window((MT_HIF_REMAP_BASE_L2 >> 16) as u16);
                self.regs.write32(MT_HIF_REMAP_L2 as usize, addr);
                let _ = self.regs.read32(MT_HIF_REMAP_L1 as usize);
                self.regs.write32(MT_HIF_REMAP_BASE_L1 as usize, val);
            }
        }
    }

    /// Point the L1 window at `base` and push the write.
    fn set_l1_window(&self, base: u16) {
        let current = self.regs.read32(MT_HIF_REMAP_L1 as usize);
        self.regs
            .write32(MT_HIF_REMAP_L1 as usize, l1_window_word(current, base));
        // "use read to push write" — mt7925/pci.c:154.
        let _ = self.regs.read32(MT_HIF_REMAP_L1 as usize);
    }

    /// The part's identity, as `mt7925e` composes it.
    pub fn revision(&self) -> u32 {
        (self.read(MT_HW_CHIPID) << 16) | (self.read(MT_HW_REV) & 0xff)
    }

    /// Prove this is an MT7925.
    pub fn identify(&self) -> Result<u32, D0Refusal> {
        let id = self.read(MT_HW_CHIPID);
        if id != CHIP_ID_MT7925 {
            return Err(D0Refusal::NotThisPart { read: id });
        }
        Ok(self.revision())
    }

    /// Take driver ownership, as `mt792xe_mcu_drv_pmctrl` does.
    ///
    /// Writes `CLR_OWN` and waits for `OWN_SYNC` to read back clear
    /// (`mt792x_core.c:1045-1051`). The budget is the vendor's own 50 ms.
    pub fn take_ownership(&self, delay: impl FnMut(), now_ns: fn() -> Option<u64>) -> Result<(), D0Refusal> {
        self.write(MT_CONN_ON_LPCTL, PCIE_LPCR_HOST_CLR_OWN);
        let budget = Budget::new(OWN_BUDGET_NS, OWN_SPINS_UNCLOCKED, now_ns);
        // The ownership register is a `Fixed` route, so each probe is one
        // MMIO read and the budget is not quietly multiplied by a window
        // write — which is why `route` is checked in the test below.
        let Route::Fixed(off) = route(MT_CONN_ON_LPCTL) else {
            // Unreachable for this constant; the assertion lives in the test
            // rather than as a panic on a device path.
            return Err(D0Refusal::OwnershipTimedOut);
        };
        handshake_until(self.regs, budget, delay, |r| {
            r.read32(off as usize) & PCIE_LPCR_HOST_OWN_SYNC == 0
        })
        .ok_or(D0Refusal::OwnershipTimedOut, D0Refusal::OwnershipTimedOut)
    }

    /// Hand the part back to its firmware, as `mt792x_mcu_fw_pmctrl` does.
    ///
    /// The other half of the ownership handshake: write `SET_OWN` and wait
    /// for `OWN_SYNC` to read back **set**, which is the part saying it has
    /// taken itself.
    ///
    /// Used for the round trip `mt7925e_mcu_init` performs immediately
    /// before the first MCU command — hand back, then take again. That the
    /// re-take exists at all is easy to miss: `mt7925_pci_probe` takes
    /// ownership once near the top and the Wi-Fi subsystem reset happens
    /// after it, so reading only that function suggests ownership survives
    /// the reset. It does not; `mt7925e_mcu_init` re-establishes it, in a
    /// different file.
    pub fn hand_back(
        &self,
        delay: impl FnMut(),
        now_ns: fn() -> Option<u64>,
    ) -> Result<(), D0Refusal> {
        self.write(MT_CONN_ON_LPCTL, PCIE_LPCR_HOST_SET_OWN);
        let budget = Budget::new(OWN_BUDGET_NS, OWN_SPINS_UNCLOCKED, now_ns);
        let Route::Fixed(off) = route(MT_CONN_ON_LPCTL) else {
            return Err(D0Refusal::OwnershipTimedOut);
        };
        handshake_until(self.regs, budget, delay, |r| {
            r.read32(off as usize) & PCIE_LPCR_HOST_OWN_SYNC != 0
        })
        .ok_or(D0Refusal::OwnershipTimedOut, D0Refusal::OwnershipTimedOut)
    }

    /// Whether the firmware's N9 core reports itself powered and up.
    ///
    /// `MT_TOP_MISC2_FW_N9_RDY` is the `FW_PWR_ON | FW_N9_ON` pair
    /// (`mt792x_regs.h:508`). Before D1 this is expected to be **false**;
    /// it is read here so D0 records the part's state on entry rather than
    /// discovering it in D1.
    pub fn firmware_ready(&self) -> bool {
        self.read(MT_CONN_ON_MISC) & MT_TOP_MISC2_FW_N9_RDY == MT_TOP_MISC2_FW_N9_RDY
    }

    /// Whether the firmware state field reads **power-on** —
    /// `mt792x_load_firmware`'s condition after the restart
    /// (`mt792x_core.c:1103-1104`): `MT_TOP_MISC_FW_STATE` (bits 2..0)
    /// equal to `MT_TOP_MISC2_FW_PWR_ON`. A field compare, not a bit
    /// test: the running firmware reads `0x3` here, which has the
    /// power-on bit set and is not this state.
    pub fn powered_on(&self) -> bool {
        self.read(MT_CONN_ON_MISC) & MT_TOP_MISC_FW_STATE == MT_TOP_MISC2_FW_PWR_ON
    }

    /// The raw `MT_CONN_ON_MISC` value behind [`Self::firmware_ready`].
    ///
    /// **The distinction is the diagnosis.** `firmware_ready` collapses
    /// this register to a yes or no, and the two ways of saying no are
    /// different faults: `MT_TOP_MISC2_FW_N9_RDY` is `genmask(1, 0)`, so
    /// `0x0` is a core that never took power and `0x1` is one that powered
    /// and did not come up. The capture from this silicon has it going
    /// `0x1` -> `0x3`, so a C2c timeout still reading `0x1` is firmware
    /// that started and died rather than firmware that never began.
    pub fn firmware_state(&self) -> u32 {
        self.read(MT_CONN_ON_MISC)
    }
}
