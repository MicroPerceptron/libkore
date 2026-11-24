//! The MT7925 register face.
//!
//! [`mt7925`] is **generated** from `tools/mt7925.h`, the cited transcription
//! of the upstream mt76 headers — do not hand-edit it; change the header and
//! regenerate (the command is in the header's comment). This file is the
//! hand-written half: the address translation, which upstream expresses as
//! code rather than as defines and which therefore cannot be transcribed.
//!
//! # Why this part needs a translation at all
//!
//! The RTL8922AE's registers are BAR offsets. The MT7925's are **not**: the
//! names in `tools/mt7925.h` are addresses in the chip's own 32-bit space
//! (`MT_HW_CHIPID` is `0x70010200`), and BAR0 is 2 MiB. Three routes reach
//! them, and `route` is the whole of the difference:
//!
//! | route | when | cost |
//! |---|---|---|
//! | [`Route::Direct`] | `addr < 0x200000` — it already *is* a BAR0 offset | free |
//! | [`Route::Fixed`] | the address falls in one of 51 statically mapped blocks | free |
//! | [`Route::L1`] | `0x18000000..0x18c00000`, `0x70000000..0x78000000`, `0x7c000000..0x7c400000` | **a window write before every access** |
//! | [`Route::L2`] | anything else | a window write and an address write |
//!
//! A `Direct` or `Fixed` access is one MMIO read. An `L1` access is a
//! read-modify-write of the window register, a read to push it, and only then
//! the access — four transactions for one register. **That is why `route` is
//! a function and not a constant**: a ladder that reaches for an `L1` address
//! inside a polling loop pays four times the bus traffic per probe, and the
//! D0 ladder is written knowing which of its registers are which.

pub mod mt7925;

use mt7925::*;

/// One statically mapped block: where the chip addresses it, where BAR0
/// carries it, and how far it runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegMap {
    /// The chip-space base of the block.
    pub phys: u32,
    /// The BAR0 offset the block is mapped at.
    pub maps: u32,
    /// The block's length.
    pub size: u32,
}

/// How a chip-space address is reached through BAR0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// Already a BAR0 offset; use it as is.
    Direct(u32),
    /// Statically mapped; this is the BAR0 offset.
    Fixed(u32),
    /// Needs the L1 window: program `base` into [`MT_HIF_REMAP_L1`]'s
    /// `MT_HIF_REMAP_L1_MASK` field, then access
    /// `MT_HIF_REMAP_BASE_L1 + offset`.
    L1 {
        /// The window base — the address's top 16 bits.
        base: u16,
        /// The offset within the window — the address's low 16 bits.
        offset: u16,
    },
    /// Needs the L2 window: program [`MT_HIF_REMAP_BASE_L2`]'s base into the
    /// L1 window, write the full address to [`MT_HIF_REMAP_L2`], then access
    /// [`MT_HIF_REMAP_BASE_L1`].
    L2(u32),
}

/// The address below which a chip-space address IS a BAR0 offset.
///
/// `__mt7925_reg_addr` returns `addr` unchanged under this
/// (`mt7925/pci.c:317`), and it is exactly BAR0's 2 MiB span — the two facts
/// are the same fact, and `res/MT7925_EVIDENCE.md` §2's open question "which
/// BAR carries registers" is answered by it together with
/// `pcim_iomap_regions(pdev, BIT(0), ...)` at `mt7925/pci.c:629`: **BAR0, and
/// only BAR0.** The driver never maps BAR2.
pub const DIRECT_LIMIT: u32 = 0x20_0000;

/// The statically mapped blocks, in upstream's order.
///
/// **The order is semantic, not cosmetic.** `__mt7925_reg_addr` returns the
/// first entry that contains the address, and upstream notes the table "was
/// ordered from highest to lowest frequency to improve lookup efficiency"
/// (`mt7925/pci.c:231`). Re-sorting it would change which block an address in
/// two blocks resolves to, so this is a verbatim transcription including
/// sequence.
pub const DEFAULT_FIXED_MAP: &[RegMap] = &[
    RegMap { phys: 0x830c0000, maps: 0x000000, size: 0x0001000 }, // :179 WF_MCU_BUS_CR_REMAP
    RegMap { phys: 0x54000000, maps: 0x002000, size: 0x0001000 }, // :180 WFDMA PCIE0 MCU DMA0
    RegMap { phys: 0x55000000, maps: 0x003000, size: 0x0001000 }, // :181 WFDMA PCIE0 MCU DMA1
    RegMap { phys: 0x56000000, maps: 0x004000, size: 0x0001000 }, // :182 WFDMA reserved
    RegMap { phys: 0x57000000, maps: 0x005000, size: 0x0001000 }, // :183 WFDMA MCU wrap CR
    RegMap { phys: 0x58000000, maps: 0x006000, size: 0x0001000 }, // :184 WFDMA PCIE1 MCU DMA0 (MEM_DMA)
    RegMap { phys: 0x59000000, maps: 0x007000, size: 0x0001000 }, // :185 WFDMA PCIE1 MCU DMA1
    RegMap { phys: 0x820c0000, maps: 0x008000, size: 0x0004000 }, // :186 WF_UMAC_TOP (PLE)
    RegMap { phys: 0x820c8000, maps: 0x00c000, size: 0x0002000 }, // :187 WF_UMAC_TOP (PSE)
    RegMap { phys: 0x820cc000, maps: 0x00e000, size: 0x0002000 }, // :188 WF_UMAC_TOP (PP)
    RegMap { phys: 0x74030000, maps: 0x010000, size: 0x0001000 }, // :189 PCIe MAC
    RegMap { phys: 0x820e0000, maps: 0x020000, size: 0x0000400 }, // :190 WF_LMAC_TOP BN0 (WF_CFG)
    RegMap { phys: 0x820e1000, maps: 0x020400, size: 0x0000200 }, // :191 WF_LMAC_TOP BN0 (WF_TRB)
    RegMap { phys: 0x820e2000, maps: 0x020800, size: 0x0000400 }, // :192 WF_LMAC_TOP BN0 (WF_AGG)
    RegMap { phys: 0x820e3000, maps: 0x020c00, size: 0x0000400 }, // :193 WF_LMAC_TOP BN0 (WF_ARB)
    RegMap { phys: 0x820e4000, maps: 0x021000, size: 0x0000400 }, // :194 WF_LMAC_TOP BN0 (WF_TMAC)
    RegMap { phys: 0x820e5000, maps: 0x021400, size: 0x0000800 }, // :195 WF_LMAC_TOP BN0 (WF_RMAC)
    RegMap { phys: 0x820ce000, maps: 0x021c00, size: 0x0000200 }, // :196 WF_LMAC_TOP (WF_SEC)
    RegMap { phys: 0x820e7000, maps: 0x021e00, size: 0x0000200 }, // :197 WF_LMAC_TOP BN0 (WF_DMA)
    RegMap { phys: 0x820cf000, maps: 0x022000, size: 0x0001000 }, // :198 WF_LMAC_TOP (WF_PF)
    RegMap { phys: 0x820e9000, maps: 0x023400, size: 0x0000200 }, // :199 WF_LMAC_TOP BN0 (WF_WTBLOFF)
    RegMap { phys: 0x820ea000, maps: 0x024000, size: 0x0000200 }, // :200 WF_LMAC_TOP BN0 (WF_ETBF)
    RegMap { phys: 0x820eb000, maps: 0x024200, size: 0x0000400 }, // :201 WF_LMAC_TOP BN0 (WF_LPON)
    RegMap { phys: 0x820ec000, maps: 0x024600, size: 0x0000200 }, // :202 WF_LMAC_TOP BN0 (WF_INT)
    RegMap { phys: 0x820ed000, maps: 0x024800, size: 0x0000800 }, // :203 WF_LMAC_TOP BN0 (WF_MIB)
    RegMap { phys: 0x820ca000, maps: 0x026000, size: 0x0002000 }, // :204 WF_LMAC_TOP BN0 (WF_MUCOP)
    RegMap { phys: 0x820d0000, maps: 0x030000, size: 0x0010000 }, // :205 WF_LMAC_TOP (WF_WTBLON)
    RegMap { phys: 0x40000000, maps: 0x070000, size: 0x0010000 }, // :206 WF_UMAC_SYSRAM
    RegMap { phys: 0x00400000, maps: 0x080000, size: 0x0010000 }, // :207 WF_MCU_SYSRAM
    RegMap { phys: 0x00410000, maps: 0x090000, size: 0x0010000 }, // :208 WF_MCU_SYSRAM (configure register)
    RegMap { phys: 0x820f0000, maps: 0x0a0000, size: 0x0000400 }, // :209 WF_LMAC_TOP BN1 (WF_CFG)
    RegMap { phys: 0x820f1000, maps: 0x0a0600, size: 0x0000200 }, // :210 WF_LMAC_TOP BN1 (WF_TRB)
    RegMap { phys: 0x820f2000, maps: 0x0a0800, size: 0x0000400 }, // :211 WF_LMAC_TOP BN1 (WF_AGG)
    RegMap { phys: 0x820f3000, maps: 0x0a0c00, size: 0x0000400 }, // :212 WF_LMAC_TOP BN1 (WF_ARB)
    RegMap { phys: 0x820f4000, maps: 0x0a1000, size: 0x0000400 }, // :213 WF_LMAC_TOP BN1 (WF_TMAC)
    RegMap { phys: 0x820f5000, maps: 0x0a1400, size: 0x0000800 }, // :214 WF_LMAC_TOP BN1 (WF_RMAC)
    RegMap { phys: 0x820f7000, maps: 0x0a1e00, size: 0x0000200 }, // :215 WF_LMAC_TOP BN1 (WF_DMA)
    RegMap { phys: 0x820f9000, maps: 0x0a3400, size: 0x0000200 }, // :216 WF_LMAC_TOP BN1 (WF_WTBLOFF)
    RegMap { phys: 0x820fa000, maps: 0x0a4000, size: 0x0000200 }, // :217 WF_LMAC_TOP BN1 (WF_ETBF)
    RegMap { phys: 0x820fb000, maps: 0x0a4200, size: 0x0000400 }, // :218 WF_LMAC_TOP BN1 (WF_LPON)
    RegMap { phys: 0x820fc000, maps: 0x0a4600, size: 0x0000200 }, // :219 WF_LMAC_TOP BN1 (WF_INT)
    RegMap { phys: 0x820fd000, maps: 0x0a4800, size: 0x0000800 }, // :220 WF_LMAC_TOP BN1 (WF_MIB)
    RegMap { phys: 0x820c4000, maps: 0x0a8000, size: 0x0004000 }, // :221 WF_LMAC_TOP BN1 (WF_MUCOP)
    RegMap { phys: 0x820b0000, maps: 0x0ae000, size: 0x0001000 }, // :222 [APB2] WFSYS_ON
    RegMap { phys: 0x80020000, maps: 0x0b0000, size: 0x0010000 }, // :223 WF_TOP_MISC_OFF
    RegMap { phys: 0x81020000, maps: 0x0c0000, size: 0x0010000 }, // :224 WF_TOP_MISC_ON
    RegMap { phys: 0x7c020000, maps: 0x0d0000, size: 0x0010000 }, // :225 CONN_INFRA, wfdma
    RegMap { phys: 0x7c060000, maps: 0x0e0000, size: 0x0010000 }, // :226 CONN_INFRA, conn_host_csr_top
    RegMap { phys: 0x7c000000, maps: 0x0f0000, size: 0x0010000 }, // :227 CONN_INFRA
    RegMap { phys: 0x70020000, maps: 0x1f0000, size: 0x0010000 }, // :228 Reserved for CBTOP, can't switch
    RegMap { phys: 0x7c500000, maps: 0x060000, size: 0x0200000 }, // :229 remap
];

/// Resolve a chip-space address to the route that reaches it.
///
/// This is `__mt7925_reg_addr` (`mt7925/pci.c:316-353`) with the window
/// programming lifted out: upstream performs the MMIO writes inside the
/// lookup, which makes the translation untestable without a device. Here the
/// lookup is pure and the caller performs the writes, so every row of the
/// table is provable on the host.
pub const fn route(addr: u32) -> Route {
    if addr < DIRECT_LIMIT {
        return Route::Direct(addr);
    }
    let mut i = 0;
    while i < DEFAULT_FIXED_MAP.len() {
        let m = DEFAULT_FIXED_MAP[i];
        if addr >= m.phys {
            let ofs = addr - m.phys;
            // **A deliberate deviation from upstream, and the only one in
            // this file.** `__mt7925_reg_addr` rejects a candidate with
            // `ofs > size` (`mt7925/pci.c:341`), which *admits* `ofs == size`
            // — one word past the block. Four blocks in this table begin
            // exactly where an earlier block ends, so under upstream's bound
            // their base address matches the earlier entry and resolves to
            // the wrong BAR0 offset:
            //
            // | block | base | upstream sends it to | should be |
            // |---|---|---|---|
            // | WF_LMAC_TOP (WF_SEC) | `0x820ce000` | `0x010000` — the **PCIe MAC** window | `0x021c00` |
            // | WF_LMAC_TOP BN0 (WF_MUCOP) | `0x820ca000` | `0x00e000` | `0x026000` |
            // | WF_LMAC_TOP (WF_WTBLON) | `0x820d0000` | `0x023000` | `0x030000` |
            // | WF_LMAC_TOP BN1 (WF_MUCOP) | `0x820c4000` | `0x00c000` | `0x0a8000` |
            //
            // It is exactly one address per boundary — `base + 4` already
            // resolves correctly — so the vendor driver gets away with it
            // for as long as nothing reads a block's first word.
            //
            // We do not reproduce it. `WF_SEC` is the key-CAM block that M6
            // programs, and silently sending its base register to the PCIe
            // MAC window is not a compatibility property worth preserving.
            // `lib.rs`'s `upstream_blocks_overlap_by_one_word` pins all four
            // addresses so the deviation stays visible and intentional.
            if ofs < m.size {
                return Route::Fixed(m.maps + ofs);
            }
        }
        i += 1;
    }
    if (addr >= 0x1800_0000 && addr < 0x18c0_0000)
        || (addr >= 0x7000_0000 && addr < 0x7800_0000)
        || (addr >= 0x7c00_0000 && addr < 0x7c40_0000)
    {
        return Route::L1 {
            base: (addr >> 16) as u16,
            offset: (addr & 0xffff) as u16,
        };
    }
    Route::L2(addr)
}

/// The BAR0 offset an [`Route::L1`] access lands on once its window is set.
#[inline]
pub const fn l1_window_offset(offset: u16) -> u32 {
    MT_HIF_REMAP_BASE_L1 + offset as u32
}

/// `MT_HIF_REMAP_L1` with its base field replaced by `base`.
#[inline]
pub const fn l1_window_word(current: u32, base: u16) -> u32 {
    (current & !MT_HIF_REMAP_L1_MASK)
        | ((base as u32) << MT_HIF_REMAP_L1_MASK.trailing_zeros())
}

/// Bytes of BAR0 the D0 ladder reads or writes.
///
/// The highest offset it touches is the L1 window itself at `0x155024`, and
/// the window it reads through runs from `0x130000`. The part's BAR0 is 2 MiB
/// and [`DIRECT_LIMIT`] says the whole of it is addressable, so the span is
/// the BAR: a shorter window is a function that is not this part.
pub const REGS_SPAN: usize = DIRECT_LIMIT as usize;
const _: () = assert!(MT_HIF_REMAP_L1 as usize + 4 <= REGS_SPAN);
const _: () = assert!(l1_window_offset(u16::MAX) as usize + 4 <= REGS_SPAN);
