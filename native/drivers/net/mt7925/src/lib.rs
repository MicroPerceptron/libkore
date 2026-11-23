//! # kore-mt7925 — the MediaTek MT7925 (RZ717) driver core
//!
//! The Wi-Fi 7 part in the Framework Laptop 13 (`c0:00.0 [14c3:0717]`), and
//! **the only network part the lab can bind**: `res/NETWORK_RESTART.md` §6
//! settles route 3 after Gigatron left, which leaves `kore-rtw89` and
//! `kore-e1000e` holding their code and losing their machine.
//!
//! Today this crate is **D0** — the part answers. Claim BAR0, drive the L1
//! remap window, read the chip identity back, take driver ownership, read the
//! firmware-ready state. D1 (firmware alive) is the next rung and is
//! deliberately not started: §3 rule 2 finishes a rung before opening the
//! next.
//!
//! ## What makes this part different from the Realtek one
//!
//! `kore-rtw89`'s registers are BAR offsets. **These are not.** The names in
//! `tools/mt7925.h` are addresses in the chip's own 32-bit space, reached
//! through BAR0 by a translation with four routes — see [`regs`]. The
//! translation is the single highest-risk thing in the crate and therefore
//! the thing [`mock`] models: a flat register file would let a wrong route
//! pass, so the double keeps *chip* space and resolves BAR0 offsets through
//! the real windows.
//!
//! ## What was read off the metal before any of this was written
//!
//! `res/MT7925_EVIDENCE.md`, and three facts from it are load-bearing here:
//!
//! - **No MSI-X**, and the MSI capability is `Count=1/32` — the part offers
//!   32 messages and Linux enables one (§10). A `PreferMsiX` policy has
//!   nothing to prefer; the binding must admit an MSI-only function.
//! - The firmware is **two blobs**, 1,580,280 bytes, shipped zstd-compressed
//!   and pinned by sha256 over the *uncompressed* bytes (§5).
//! - `mt7925e` printed `ASIC revision: 79250000` on this exact machine
//!   (§11), which is why [`d0::CHIP_ID_MT7925`] is a measurement and not a
//!   guess.
//!
//! ## The layers
//!
//! | module | what it owns |
//! |---|---|
//! | [`regs`] | the address translation, and the generated register map |
//! | [`d0`] | the D0 ladder over the [`Regs`](kore_driver::seam::Regs) seam |
//! | [`fw`] | the two firmware containers, parsed and accounted for |
//! | [`dma`] | the WFDMA geometry: the part's reach, one descriptor, three queues |
//! | [`mcu`] | the MCU command framing, and how an answer is recognised |
//! | [`mock`] | the chip-space double the ladder is proven against |

#![no_std]

#[cfg(test)]
extern crate alloc;

pub mod d0;
pub mod d1;
/// M1 — the configuration plane under the unified command class.
pub mod d2;
pub mod c2;
pub mod clc;
pub mod dma;
pub mod fw;
pub mod mac;
pub mod mcu;
pub mod product;
pub mod regs;

use kore_driver::seam::Regs;

pub use product::{DEVICE_MT7925, Product, VENDOR_MEDIATEK};

/// The wait tier's hooks, handed in by whoever composes this core.
///
/// Mirrors `kore_rtw89::Waits` deliberately: the two drivers should be
/// composed the same way, and a second spelling of the same two function
/// pointers is a difference a reader would have to check is not meaningful.
#[derive(Clone, Copy)]
pub struct Waits {
    /// Monotonic nanoseconds, or `None` before the counter is calibrated.
    pub clock: fn() -> Option<u64>,
    /// One relaxation between probes.
    pub pause: fn(),
}

/// What the bring-up could not establish, as a ladder.
///
/// The rung is what the row notes when the class bind refuses, so the number
/// is the diagnosis on a machine whose only witness channel is a photograph
/// of the panel. **Only D0's rungs exist today** — numbering is left open
/// above them rather than compacted, so D1's rungs can be added without
/// renumbering what a photograph already recorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// The BAR the registers live in was not mapped.
    NotPresent,
    /// The identity register answered with something that is not this part.
    NotThisPart,
    /// The part did not release ownership inside the vendor's own budget.
    Ownership,
}

impl Error {
    /// This error's rung, for the founding note.
    pub const fn rung(self) -> u8 {
        match self {
            Self::NotPresent => 1,
            Self::NotThisPart => 2,
            Self::Ownership => 3,
        }
    }
}

impl From<d0::D0Refusal> for Error {
    fn from(r: d0::D0Refusal) -> Self {
        match r {
            d0::D0Refusal::NotThisPart { .. } => Self::NotThisPart,
            d0::D0Refusal::OwnershipTimedOut => Self::Ownership,
        }
    }
}

/// The crate's result alias.
pub type Result<T> = core::result::Result<T, Error>;

/// **D0 passed**: the part answered with its own identity and handed the
/// driver ownership.
///
/// Carries what was read rather than merely the fact that it was read —
/// the revision is the number a panel photograph has to show, and the row
/// cannot report what it did not keep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mt7925 {
    product: Product,
    revision: u32,
    firmware_ready: bool,
}

impl Mt7925 {
    /// Run D0 against a mapped register window.
    ///
    /// Three steps, in the order the vendor driver takes them: read the
    /// identity back through the L1 remap window, take driver ownership, and
    /// record whether firmware was already up. The third is **not** a gate —
    /// a cold part reports not-ready and that is the expected answer before
    /// D1 — it is recorded so D1 knows what state it inherited.
    pub fn bring_up<R: Regs + ?Sized>(regs: &R, product: Product, waits: Waits) -> Result<Self> {
        let core = d0::Core::new(regs);
        let read = core.read(regs::mt7925::MT_HW_CHIPID);
        if read != product.chip_id() {
            return Err(Error::NotThisPart);
        }
        let revision = core.revision();
        core.take_ownership(waits.pause, waits.clock)?;
        Ok(Self {
            product,
            revision,
            firmware_ready: core.firmware_ready(),
        })
    }

    /// The part.
    pub const fn product(self) -> Product {
        self.product
    }

    /// The composed revision, as `mt7925e` prints it — `0x79250000` on the
    /// lab's machine (`res/MT7925_EVIDENCE.md` §11).
    pub const fn revision(self) -> u32 {
        self.revision
    }

    /// Whether the part's N9 core reported itself up when D0 ran. Expected
    /// `false` before D1.
    pub const fn firmware_ready(self) -> bool {
        self.firmware_ready
    }
}

#[cfg(test)]
pub mod mock;

#[cfg(test)]
mod tests {
    use super::d0::*;
    use super::{Error, Mt7925, Product, Waits};
    use super::mock::MockMt7925;
    use super::regs::mt7925::*;
    use super::regs::*;

    fn unclocked() -> Option<u64> {
        None
    }

    /// The host double's wait tier: no clock, no relaxation needed.
    fn test_waits() -> Waits {
        Waits {
            clock: unclocked,
            pause: || {},
        }
    }

    /// The identity is **not** directly addressable, and the exact window the
    /// part needs is the one upstream's ranges imply.
    #[test]
    fn the_chip_id_takes_the_l1_window() {
        assert_eq!(
            route(MT_HW_CHIPID),
            Route::L1 {
                base: 0x7001,
                offset: 0x0200
            }
        );
    }

    /// Ownership is in a statically mapped block, so each poll of it is one
    /// MMIO read rather than a window write plus a read. `take_ownership`
    /// relies on this to keep its budget honest.
    #[test]
    fn ownership_is_a_fixed_mapping_not_a_window() {
        // 0x7c060000 -> BAR0 0x0e0000 (mt7925/pci.c:227), + 0x10.
        assert_eq!(route(MT_CONN_ON_LPCTL), Route::Fixed(0x0e_0010));
        assert_eq!(route(MT_CONN_ON_MISC), Route::Fixed(0x0e_00f0));
    }

    /// Everything under 2 MiB is already a BAR0 offset — including the
    /// window register itself, which is what makes the whole scheme
    /// bootstrappable.
    #[test]
    fn the_window_register_is_reachable_without_the_window() {
        assert_eq!(route(MT_HIF_REMAP_L1), Route::Direct(MT_HIF_REMAP_L1));
        assert!(MT_HIF_REMAP_L1 < DIRECT_LIMIT);
    }

    /// The part answers with its own identity, through the window, composed
    /// the way `mt7925e` composes it — and the answer is the one the lab's
    /// machine actually printed.
    #[test]
    fn d0_identifies_the_part() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        assert_eq!(core.identify(), Ok(0x7925_0000));
        // The read really went to the chip address, not to a BAR0 offset
        // that happened to hold the value.
        assert!(dev.reads.borrow().contains(&MT_HW_CHIPID));
    }

    /// A different part in the same socket is refused, and the refusal
    /// carries what was read so "wrong part" and "dead window" are
    /// distinguishable.
    #[test]
    fn a_foreign_part_is_refused_with_its_value() {
        let dev = MockMt7925::new();
        dev.poke(MT_HW_CHIPID, 0x7961);
        let core = Core::new(&dev);
        assert_eq!(
            core.identify(),
            Err(D0Refusal::NotThisPart { read: 0x7961 })
        );
    }

    /// An unmapped or floating window reads all-ones, and that is a refusal
    /// rather than a silent misidentification.
    #[test]
    fn a_dead_window_is_refused() {
        let dev = MockMt7925::new();
        dev.poke(MT_HW_CHIPID, 0xffff_ffff);
        let core = Core::new(&dev);
        assert_eq!(
            core.identify(),
            Err(D0Refusal::NotThisPart { read: 0xffff_ffff })
        );
    }

    /// The part releases ownership and the handshake sees it.
    #[test]
    fn ownership_is_taken_when_the_part_clears_sync() {
        let dev = MockMt7925::new();
        // A cold part is the firmware's: OWN_SYNC reads set before we ask.
        assert_ne!(dev.peek(MT_CONN_ON_LPCTL) & PCIE_LPCR_HOST_OWN_SYNC, 0);
        let core = Core::new(&dev);
        assert_eq!(core.take_ownership(|| {}, unclocked), Ok(()));
        // The driver asked for it, and the part answered.
        assert_eq!(
            dev.peek(MT_CONN_ON_LPCTL) & PCIE_LPCR_HOST_CLR_OWN,
            PCIE_LPCR_HOST_CLR_OWN
        );
        assert_eq!(dev.peek(MT_CONN_ON_LPCTL) & PCIE_LPCR_HOST_OWN_SYNC, 0);
    }

    /// A part that never clears `OWN_SYNC` is refused inside the vendor's own
    /// 50 ms budget rather than spun on forever.
    #[test]
    fn ownership_that_never_syncs_is_refused() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        // The rung's own fault switch: a part that takes the strobe and
        // never answers it.
        dev.own_stuck.set(true);
        let probes = core::cell::Cell::new(0u32);
        let r = core.take_ownership(
            || {
                probes.set(probes.get() + 1);
            },
            unclocked,
        );
        assert_eq!(r, Err(D0Refusal::OwnershipTimedOut));
        // Bounded: the unclocked fallback took its probes and stopped.
        assert!(probes.get() <= OWN_SPINS_UNCLOCKED + 1, "probes: {}", probes.get());
    }

    /// Before D1 the firmware is not up, and D0 records that rather than
    /// assuming it.
    #[test]
    fn firmware_is_not_ready_before_d1() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        assert!(!core.firmware_ready());
        dev.poke(MT_CONN_ON_MISC, MT_TOP_MISC2_FW_N9_RDY);
        assert!(core.firmware_ready());
    }

    /// **The four addresses upstream misroutes, pinned.**
    ///
    /// `__mt7925_reg_addr` admits `ofs == size` (`mt7925/pci.c:341`), so a
    /// block whose base is an earlier block's one-past-end matches the
    /// earlier entry. Four blocks in this table are built that way. `route`
    /// deviates — see the table in `regs::route` — and this test is what
    /// keeps the deviation honest: each address must resolve to **its own**
    /// block's BAR0 base, not to the earlier block's end.
    #[test]
    fn upstream_blocks_overlap_by_one_word() {
        // WF_SEC — the key-CAM block M6 programs. Upstream: 0x010000, which
        // is the PCIe MAC window.
        assert_eq!(route(0x820c_e000), Route::Fixed(0x02_1c00));
        // WF_LMAC_TOP BN0 (WF_MUCOP). Upstream: 0x00e000.
        assert_eq!(route(0x820c_a000), Route::Fixed(0x02_6000));
        // WF_LMAC_TOP (WF_WTBLON). Upstream: 0x023000.
        assert_eq!(route(0x820d_0000), Route::Fixed(0x03_0000));
        // WF_LMAC_TOP BN1 (WF_MUCOP). Upstream: 0x00c000.
        assert_eq!(route(0x820c_4000), Route::Fixed(0x0a_8000));
        // One word in, upstream and this agree — the collision really is a
        // single address per boundary.
        assert_eq!(route(0x820c_e004), Route::Fixed(0x02_1c04));
    }

    /// The lab's part is recognised; nothing else is.
    #[test]
    fn only_the_mt7925_is_claimed() {
        assert_eq!(Product::from_pci(0x14c3, 0x0717), Some(Product::Mt7925));
        // The MT7927/MT7928 siblings take a different fixed map upstream, so
        // admitting them would resolve their registers to wrong offsets.
        assert_eq!(Product::from_pci(0x14c3, 0x7927), None);
        assert_eq!(Product::from_pci(0x14c3, 0x7928), None);
        // Another vendor's network function with the same class.
        assert_eq!(Product::from_pci(0x10ec, 0x8922), None);
    }

    /// **D0 end to end against the double**: identity through the window,
    /// ownership taken, firmware state recorded.
    #[test]
    fn bring_up_passes_d0() {
        let dev = MockMt7925::new();
        let part = Mt7925::bring_up(&dev, Product::Mt7925, test_waits()).expect("D0");
        assert_eq!(part.revision(), 0x7925_0000, "what mt7925e printed");
        assert_eq!(part.product(), Product::Mt7925);
        assert!(!part.firmware_ready(), "a cold part has no firmware up");
        // Ownership really was taken, not assumed.
        assert_eq!(dev.peek(MT_CONN_ON_LPCTL) & PCIE_LPCR_HOST_OWN_SYNC, 0);
    }

    /// A foreign part in the socket is refused at D0's first step, before
    /// ownership is touched.
    #[test]
    fn bring_up_refuses_a_foreign_part() {
        let dev = MockMt7925::new();
        dev.poke(MT_HW_CHIPID, 0x7961);
        assert_eq!(
            Mt7925::bring_up(&dev, Product::Mt7925, test_waits()),
            Err(Error::NotThisPart)
        );
        assert_ne!(
            dev.peek(MT_CONN_ON_LPCTL) & PCIE_LPCR_HOST_OWN_SYNC,
            0,
            "ownership was never asked for"
        );
    }

    /// A part that keeps ownership refuses at its own rung, bounded.
    #[test]
    fn bring_up_refuses_a_part_that_keeps_ownership() {
        let dev = MockMt7925::new();
        dev.own_stuck.set(true);
        assert_eq!(
            Mt7925::bring_up(&dev, Product::Mt7925, test_waits()),
            Err(Error::Ownership)
        );
    }

    /// The rungs are distinct, so a note read off a photograph names one
    /// failure and not a class of them.
    #[test]
    fn every_rung_is_distinct() {
        let rungs = [Error::NotPresent, Error::NotThisPart, Error::Ownership]
            .map(Error::rung);
        for (i, a) in rungs.iter().enumerate() {
            assert_ne!(*a, 0, "rung 0 is 'no refusal'");
            for b in &rungs[i + 1..] {
                assert_ne!(a, b, "two refusals share a rung");
            }
        }
    }

    /// Every statically mapped block's base round-trips: `route` forward and
    /// the mock's independently written inverse agree.
    ///
    /// With `route`'s exclusive bound no block is shadowed any more, so the
    /// count asserted at the end is zero: a future table change that
    /// reintroduces an overlap is caught here.
    #[test]
    fn every_fixed_block_round_trips() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        let mut shadowed = 0usize;
        for (i, m) in DEFAULT_FIXED_MAP.iter().enumerate() {
            let addr = m.phys;
            let Route::Fixed(off) = route(addr) else {
                shadowed += 1;
                continue;
            };
            // Under the inclusive bound an earlier block can claim this
            // base; the forward route then names a different chip address
            // than the block's own.
            if off != m.maps {
                shadowed += 1;
                continue;
            }
            let witness = 0xc0de_0000 | i as u32;
            core.write(addr, witness);
            assert_eq!(
                dev.peek(addr),
                witness,
                "block {i} at {:#x} -> BAR0 {off:#x} did not land in chip space",
                m.phys
            );
        }
        assert_eq!(
            shadowed, 0,
            "no block should be shadowed under route()'s exclusive bound"
        );
    }
}
