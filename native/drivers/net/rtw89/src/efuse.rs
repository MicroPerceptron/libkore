//! **The efuse** — the part's one-time-programmable record: the physical
//! dump through the `R_BE_EFUSE_CTRL` window, the logical parse of the RF
//! block (`rtw89_eeprom_parser_be`), and the four secure-boot bytes the
//! firmware loader's key selection keys on.
//!
//! Three facts come out of the RF block and are the regulatory root the
//! founding note records (`rtw8922a_read_efuse_rf`, rtw8922a.c:619-634):
//! the country code, the crystal trim and the RF front-end type. The
//! station address is not in the logical map on PCIe — it is read as
//! three halfwords from MMIO `0x3104` (`rtw8922a_read_efuse_mac_addr`) —
//! and belongs to D2, when there is a frame to put it on.

use kore_driver::seam::RegsNarrow;

use crate::fw::SecureBoot;
use crate::regs::be::*;
use crate::regs::{CHIP_CAV, encode, get, reg};
use crate::{Waits, spins_for};

/// `chip->physical_efuse_size` (rtw8922a.c:2917): the whole physical map.
pub const PHYSICAL_SIZE: usize = 0x1300;
/// `chip->sec_ctrl_efuse_size` (rtw8922a.c:2916): the bytes before the
/// first record.
pub const SEC_CTRL_SIZE: usize = 4;
/// The RF block (`rtw8922a_efuse_blocks[RTW89_EFUSE_BLOCK_RF]`,
/// rtw8922a.c:228): logical page 1, offset 0, `0x240` bytes.
pub const RF_BLOCK_PAGE: u32 = 1;
/// …its offset inside the page.
pub const RF_BLOCK_OFFSET: usize = 0;
/// …and its length.
pub const RF_BLOCK_SIZE: usize = 0x240;
/// `struct rtw8922a_efuse` (rtw8922a.h:40-69), the three members the
/// bring-up consumes, as byte offsets from the RF block's start: the
/// dossier's §D.4 table, `country_code[2]` first, `xtal_k` after
/// `channel_plan` at `0xB8`, `rfe_type` after `board_info` and its
/// reserve.
const COUNTRY_CODE_AT: usize = 0x000;
const XTAL_K_AT: usize = 0x0B9;
const RFE_TYPE_AT: usize = 0x0CA;
const _: () = assert!(RFE_TYPE_AT < RF_BLOCK_SIZE);

/// Nanoseconds one physical word may take to become readable
/// (`read_poll_timeout_atomic(…, 1, 1000000, …)`, efuse_be.c:84-86): 1 s.
const WORD_BUDGET_NS: u64 = 1_000_000_000;
/// Nanoseconds the system power state may take to report active after the
/// BT-disable bit is dropped (`rtw89_cnv_efuse_state_be`, efuse_be.c:154-
/// 157): 5 ms.
const SYS_ACT_BUDGET_NS: u64 = 5_000_000;
/// The `mdelay(1)` between the two A-die isolation steps (efuse_be.c:36,
/// :59).
const APHY_SETTLE_NS: u64 = 1_000_000;

/// The regulatory root: what the RF block says about this unit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RfEfuse {
    /// Two ASCII letters, or `0xFF 0xFF` when unprogrammed.
    pub country_code: [u8; 2],
    /// The crystal trim (`xtal_k`).
    pub xtal_cap: u8,
    /// The RF front-end type the firmware's tables are indexed by.
    pub rfe_type: u8,
    /// Whether the MAC's own autoload from the efuse completed
    /// (`R_BE_SYS_WL_EFUSE_CTRL.B_BE_AUTOLOAD_SUS`, efuse_be.c:350).
    pub autoload: bool,
}

/// Why the efuse could not be read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EfuseError {
    /// A word never became readable within its budget.
    NotReady,
    /// A record names a block index outside the map, or runs past its
    /// end.
    BadRecord,
}

/// `rtw89_cnv_efuse_state_be` (efuse_be.c:143-161): with `idle == false`,
/// drop the BT-disable bit and wait for the whole system's power state to
/// read active; with `idle == true`, set it back. Upstream only warns when
/// the state never reads active and goes on to the read — whose own
/// ready poll is the definitive check — so this does the same and reports
/// the expiry to the caller as a fact, not a refusal.
pub fn cnv_efuse_state<R: RegsNarrow>(regs: &R, waits: &Waits, idle: bool) -> bool {
    if idle {
        let v = regs.read32(reg(R_BE_WL_BT_PWR_CTRL));
        regs.write32(reg(R_BE_WL_BT_PWR_CTRL), v | B_BE_BT_DISN_EN);
        return true;
    }
    let v = regs.read32(reg(R_BE_WL_BT_PWR_CTRL));
    regs.write32(reg(R_BE_WL_BT_PWR_CTRL), v & !B_BE_BT_DISN_EN);
    waits.poll_field(
        regs,
        R_BE_IC_PWR_STATE,
        B_BE_WHOLE_SYS_PWR_STE_MASK,
        MAC_AX_SYS_ACT,
        SYS_ACT_BUDGET_NS,
    )
}

/// `rtw89_enable_efuse_pwr_cut_ddv_be` (efuse_be.c:23-42): power the
/// efuse macro for a read. The A-die isolation patch is skipped on the
/// A-cut.
fn enable_pwr_cut<R: RegsNarrow>(regs: &R, waits: &Waits, cv: u8) {
    let v = regs.read8(reg(R_BE_PMC_DBG_CTRL2));
    regs.write8(
        reg(R_BE_PMC_DBG_CTRL2),
        v | B_BE_SYSON_DIS_PMCR_BE_WRMSK as u8,
    );
    if cv != CHIP_CAV {
        let v = regs.read16(reg(R_BE_SYS_ISO_CTRL));
        regs.write16(reg(R_BE_SYS_ISO_CTRL), v | B_BE_PWC_EV2EF_S as u16);
        waits.delay(APHY_SETTLE_NS);
        let v = regs.read16(reg(R_BE_SYS_ISO_CTRL));
        regs.write16(reg(R_BE_SYS_ISO_CTRL), v | B_BE_PWC_EV2EF_B as u16);
        let v = regs.read16(reg(R_BE_SYS_ISO_CTRL));
        regs.write16(reg(R_BE_SYS_ISO_CTRL), v & !(B_BE_ISO_EB2CORE as u16));
    }
    let v = regs.read32(reg(R_BE_EFUSE_CTRL_2_V1));
    regs.write32(reg(R_BE_EFUSE_CTRL_2_V1), v | B_BE_EF_BURST);
}

/// `rtw89_disable_efuse_pwr_cut_ddv_be` (efuse_be.c:44-62): the mirror
/// image, with the same settle between the two isolation steps.
fn disable_pwr_cut<R: RegsNarrow>(regs: &R, waits: &Waits, cv: u8) {
    if cv != CHIP_CAV {
        let v = regs.read16(reg(R_BE_SYS_ISO_CTRL));
        regs.write16(reg(R_BE_SYS_ISO_CTRL), v | B_BE_ISO_EB2CORE as u16);
        let v = regs.read16(reg(R_BE_SYS_ISO_CTRL));
        regs.write16(reg(R_BE_SYS_ISO_CTRL), v & !(B_BE_PWC_EV2EF_B as u16));
        waits.delay(APHY_SETTLE_NS);
        let v = regs.read16(reg(R_BE_SYS_ISO_CTRL));
        regs.write16(reg(R_BE_SYS_ISO_CTRL), v & !(B_BE_PWC_EV2EF_S as u16));
    }
    let v = regs.read8(reg(R_BE_PMC_DBG_CTRL2));
    regs.write8(
        reg(R_BE_PMC_DBG_CTRL2),
        v & !(B_BE_SYSON_DIS_PMCR_BE_WRMSK as u8),
    );
    let v = regs.read32(reg(R_BE_EFUSE_CTRL_2_V1));
    regs.write32(reg(R_BE_EFUSE_CTRL_2_V1), v & !B_BE_EF_BURST);
}

/// `rtw89_dump_physical_efuse_map_be` over the DDV path
/// (`rtw89_dump_physical_efuse_map_ddv_be`, efuse_be.c:64-97): `out.len()`
/// bytes from physical address `addr`, both 4-aligned, one dword per
/// `R_BE_EFUSE_CTRL` command. Bracketed by the efuse-state conversion and
/// the power cut. Where upstream returns on a ready timeout with the
/// power cut left enabled, this disables it on the way out — the same
/// writes a successful read ends with, so the part is left as a read
/// leaves it.
pub fn read_physical<R: RegsNarrow>(
    regs: &R,
    waits: &Waits,
    cv: u8,
    addr: u32,
    out: &mut [u8],
) -> Result<(), EfuseError> {
    assert!(addr.is_multiple_of(4) && out.len().is_multiple_of(4));
    cnv_efuse_state(regs, waits, false);
    enable_pwr_cut(regs, waits, cv);
    let mut result = Ok(());
    for (i, word) in out.chunks_exact_mut(4).enumerate() {
        let at = addr + 4 * i as u32;
        regs.write32(
            reg(R_BE_EFUSE_CTRL),
            encode(at, B_BE_EF_ADDR_MASK) & !B_BE_EF_RDY,
        );
        if !waits.poll_set(regs, R_BE_EFUSE_CTRL, B_BE_EF_RDY, WORD_BUDGET_NS) {
            result = Err(EfuseError::NotReady);
            break;
        }
        word.copy_from_slice(&regs.read32(reg(R_BE_EFUSE_CTRL_1_V1)).to_le_bytes());
    }
    disable_pwr_cut(regs, waits, cv);
    cnv_efuse_state(regs, waits, true);
    result
}

/// `rtw89_eeprom_parser_be` (efuse_be.c:215-311) for the RF block: walk
/// the physical map's records — a 3-byte header of page `[19:17]`, block
/// index `[16:4]` and word-enable `[3:0]`, then one 2-byte word per
/// *clear* word-enable bit, landing at logical index `(block << 3) + (i
/// << 1)` — and keep the words that fall in page 1's first `0x240` bytes.
/// Unwritten bytes read `0xFF`.
pub fn parse_rf_block(phy: &[u8]) -> Result<RfEfuse, EfuseError> {
    let min = RF_BLOCK_OFFSET & !1;
    let max = (RF_BLOCK_OFFSET + RF_BLOCK_SIZE).next_multiple_of(2);
    let mut log = [0xFFu8; RF_BLOCK_SIZE];
    let mut phy_idx = SEC_CTRL_SIZE;
    while phy_idx < phy.len() {
        let Some(&[h1, h2, h3]) = phy.get(phy_idx..phy_idx + 3) else {
            break;
        };
        if h1 == 0xFF || h2 == 0xFF || h3 == 0xFF {
            break;
        }
        phy_idx += 3;
        let hdr = (h1 as u32) << 16 | (h2 as u32) << 8 | h3 as u32;
        let blk_page = get(hdr, EFUSE_HDR_PAGE_MASK);
        let blk_idx = get(hdr, EFUSE_HDR_OFFSET_MASK) as usize;
        let word_en = get(hdr, EFUSE_HDR_WORD_EN_MASK);
        if blk_idx >= (RTW89_EFUSE_MAX_BLOCK_SIZE as usize) >> 3 {
            return Err(EfuseError::BadRecord);
        }
        for i in 0..4usize {
            if word_en & (1 << i) != 0 {
                continue;
            }
            if phy_idx >= phy.len() - 1 {
                return Err(EfuseError::BadRecord);
            }
            let log_idx = (blk_idx << 3) + (i << 1);
            if blk_page == RF_BLOCK_PAGE && log_idx >= min && log_idx < max {
                let (v0, v1) = (phy[phy_idx], phy[phy_idx + 1]);
                let at = log_idx - RF_BLOCK_OFFSET;
                if log_idx == min && RF_BLOCK_OFFSET > min {
                    log[at + 1] = v1;
                } else if log_idx + 2 == max && RF_BLOCK_OFFSET + RF_BLOCK_SIZE < max {
                    log[at] = v0;
                } else {
                    log[at] = v0;
                    log[at + 1] = v1;
                }
            }
            phy_idx += 2;
        }
    }
    Ok(RfEfuse {
        country_code: [log[COUNTRY_CODE_AT], log[COUNTRY_CODE_AT + 1]],
        xtal_cap: log[XTAL_K_AT],
        rfe_type: log[RFE_TYPE_AT],
        autoload: false,
    })
}

/// `rtw89_parse_efuse_map_be` (efuse_be.c:341-399) for the RF block: the
/// autoload flag, the whole physical dump, the parse.
pub fn read_rf<R: RegsNarrow>(regs: &R, waits: &Waits, cv: u8) -> Result<RfEfuse, EfuseError> {
    let autoload = regs.read16(reg(R_BE_SYS_WL_EFUSE_CTRL)) & B_BE_AUTOLOAD_SUS as u16 != 0;
    let mut phy = [0u8; PHYSICAL_SIZE];
    read_physical(regs, waits, cv, 0, &mut phy)?;
    let mut rf = parse_rf_block(&phy)?;
    rf.autoload = autoload;
    Ok(rf)
}

/// `get_sb_cryp_sel_idx` (efuse_be.c:437-466): the key-management
/// selector index encoded in the `sb_cryp_sel` halfword, or `u16::MAX`
/// when its redundancy checks fail.
fn sb_cryp_sel_idx(sb_cryp_sel: u16) -> u16 {
    let sel_form = (sb_cryp_sel & 0xFF) as u8;
    let sel_idx = (sb_cryp_sel >> 8) as u8;
    let mut cnt_zero = 0;
    let mut v: u16 = 0;
    for idx in 0..4u8 {
        let low = sel_form & (1 << idx) != 0;
        let high = sel_form & (1 << (7 - idx)) != 0;
        if low != high {
            return u16::MAX;
        }
        if low {
            continue;
        }
        cnt_zero += 1;
        if cnt_zero == 1 {
            v = idx as u16 * 16;
        } else {
            return u16::MAX;
        }
    }
    let low = sel_idx & 0x0F;
    let high = sel_idx >> 4;
    if low ^ high != 0xF {
        return u16::MAX;
    }
    v + low as u16
}

/// `sb_sel_mgn[]` (efuse_be.c:19-21).
const SB_SEL_MGN: [u32; 2] = [0x0800_0100, 0x0C00_0180];

/// `get_mss_dev_type_idx` (efuse.c:385-408): the efuse's device-type code
/// to the pool's index, or the invalid marker.
fn mss_dev_type_idx(code: u8) -> u8 {
    match code as u32 {
        MSS_DEV_TYPE_FWSEC_WINLIN_INBOX => 0x0,
        MSS_DEV_TYPE_FWSEC_NONLIN_INBOX_NON_COB => 0x1,
        MSS_DEV_TYPE_FWSEC_NONLIN_INBOX_COB => 0x2,
        MSS_DEV_TYPE_FWSEC_NONWIN_INBOX => 0x3,
        MSS_DEV_TYPE_FWSEC_DEF => RTW89_FW_MSS_DEV_TYPE_FWSEC_DEF as u8,
        _ => RTW89_FW_MSS_DEV_TYPE_FWSEC_INV as u8,
    }
}

/// `rtw89_efuse_read_fw_secure_be` (efuse_be.c:468-510) on the four
/// bytes at `EFUSE_SEC_BE_START`, with `rtw89_efuse_recognize_mss_info_v1`
/// (efuse.c:412-441; the 8852B special case does not apply): secure boot
/// is on only when the selector word is programmed, decodes, and the
/// device type is one the key pool indexes.
pub fn secure_boot_from(sec: [u8; EFUSE_SEC_BE_SIZE as usize]) -> SecureBoot {
    let base = EFUSE_SEC_BE_START as usize;
    let at = |a: u32| a as usize - base;
    let sb_cryp_sel = u16::from_le_bytes([
        sec[at(EFUSE_SB_CRYP_SEL_ADDR)],
        sec[at(EFUSE_SB_CRYP_SEL_ADDR) + 1],
    ]);
    if sb_cryp_sel == EFUSE_SB_CRYP_SEL_DEFAULT as u16 {
        return SecureBoot::OFF;
    }
    let idx = sb_cryp_sel_idx(sb_cryp_sel);
    let Some(&sb_sel_mgn) = SB_SEL_MGN.get(idx as usize) else {
        return SecureBoot::OFF;
    };
    let b1 = sec[at(EFUSE_EXTERNALPN_ADDR_BE)];
    let b2 = sec[at(EFUSE_SERIALNUM_ADDR_BE)];
    let dev_type = mss_dev_type_idx(get(b1 as u32, EFUSE_B1_MSSDEVTYPE_MASK) as u8);
    if dev_type == RTW89_FW_MSS_DEV_TYPE_FWSEC_INV as u8 {
        return SecureBoot::OFF;
    }
    let cust_idx = 0x1F
        - (get(b1 as u32, EFUSE_B1_MSSCUSTIDX0_MASK)
            | get(b2 as u32, EFUSE_B2_MSSCUSTIDX1_MASK) << 4) as u8;
    let key_num = 0xF - get(b2 as u32, EFUSE_B2_MSSKEYNUM_MASK) as u8;
    SecureBoot {
        secure_boot: true,
        sb_sel_mgn,
        mss_dev_type: dev_type,
        mss_cust_idx: cust_idx,
        mss_key_num: key_num,
        mss_idx: 0,
    }
}

/// Read and decode the secure-boot bytes.
pub fn read_secure_boot<R: RegsNarrow>(
    regs: &R,
    waits: &Waits,
    cv: u8,
) -> Result<SecureBoot, EfuseError> {
    let mut sec = [0u8; EFUSE_SEC_BE_SIZE as usize];
    read_physical(regs, waits, cv, EFUSE_SEC_BE_START, &mut sec)?;
    Ok(secure_boot_from(sec))
}

/// One probe per microsecond of budget, the unclocked fallback the whole
/// ladder uses (see [`spins_for`]).
const _: () = assert!(spins_for(WORD_BUDGET_NS) == 1_000_000);

#[cfg(test)]
pub(crate) mod build {
    //! A physical map builder for the host tests, byte-exact to the
    //! parser's record format.

    use super::*;

    /// One record: page, block index, the four words (`None` = word-enable
    /// bit set, word absent).
    pub fn record(
        out: &mut [u8],
        at: &mut usize,
        page: u32,
        blk_idx: u32,
        words: [Option<[u8; 2]>; 4],
    ) {
        let mut word_en = 0u32;
        for (i, w) in words.iter().enumerate() {
            if w.is_none() {
                word_en |= 1 << i;
            }
        }
        let hdr = encode(page, EFUSE_HDR_PAGE_MASK)
            | encode(blk_idx, EFUSE_HDR_OFFSET_MASK)
            | encode(word_en, EFUSE_HDR_WORD_EN_MASK);
        out[*at] = (hdr >> 16) as u8;
        out[*at + 1] = (hdr >> 8) as u8;
        out[*at + 2] = hdr as u8;
        *at += 3;
        for w in words.into_iter().flatten() {
            out[*at] = w[0];
            out[*at + 1] = w[1];
            *at += 2;
        }
    }

    /// A map carrying the three RF facts at their offsets, everything
    /// else unprogrammed.
    pub fn rf_map(country: [u8; 2], xtal_k: u8, rfe_type: u8) -> [u8; PHYSICAL_SIZE] {
        let mut phy = [0xFFu8; PHYSICAL_SIZE];
        let mut at = SEC_CTRL_SIZE;
        // Block 0 of page 1: word 0 is the country code.
        record(
            &mut phy,
            &mut at,
            RF_BLOCK_PAGE,
            0,
            [Some(country), None, None, None],
        );
        // xtal_k at 0xB9 = block 0x17 (0xB8), word 0, high byte.
        record(
            &mut phy,
            &mut at,
            RF_BLOCK_PAGE,
            0x17,
            [Some([0x7F, xtal_k]), None, None, None],
        );
        // rfe_type at 0xCA = block 0x19 (0xC8), word 1, low byte.
        record(
            &mut phy,
            &mut at,
            RF_BLOCK_PAGE,
            0x19,
            [None, Some([rfe_type, 0x00]), None, None],
        );
        // A page-2 record that must be ignored, and a full four-word one.
        record(
            &mut phy,
            &mut at,
            2,
            0,
            [Some([1, 2]), Some([3, 4]), Some([5, 6]), Some([7, 8])],
        );
        phy
    }
}

#[cfg(test)]
mod tests {
    use super::build::*;
    use super::*;

    #[test]
    fn the_rf_block_parse_lands_the_three_facts_and_skips_other_pages() {
        let phy = rf_map(*b"US", 0x2A, 3);
        let rf = parse_rf_block(&phy).unwrap();
        assert_eq!(rf.country_code, *b"US");
        assert_eq!(rf.xtal_cap, 0x2A);
        assert_eq!(rf.rfe_type, 3);
        let blank = [0xFFu8; PHYSICAL_SIZE];
        assert_eq!(parse_rf_block(&blank).unwrap().rfe_type, 0xFF);
        // A record whose word would run past the map's end: the stream
        // is sequential, so it is the first record of a map cut short.
        let hdr = encode(1, EFUSE_HDR_PAGE_MASK) | encode(0b1110, EFUSE_HDR_WORD_EN_MASK);
        let truncated = [
            0xFF,
            0xFF,
            0xFF,
            0xFF,
            (hdr >> 16) as u8,
            (hdr >> 8) as u8,
            hdr as u8,
        ];
        assert_eq!(parse_rf_block(&truncated), Err(EfuseError::BadRecord));
        // A header cut off by the end is the end.
        let mut cut = blank;
        cut[PHYSICAL_SIZE - 2] = 0x00;
        assert!(parse_rf_block(&cut).is_ok());
    }

    #[test]
    fn secure_boot_is_off_unless_the_selector_word_is_programmed_and_valid() {
        assert_eq!(secure_boot_from([0xFF; 4]), SecureBoot::OFF);
        // Selector form 0b1110_0111: bit 3 (and its mirror bit 4) clear →
        // index base 3 * 16; index nibble 0x5 with complement 0xA.
        let sel: u16 = 0xA5 << 8 | 0b1110_0111;
        assert_eq!(sb_cryp_sel_idx(sel), 48 + 5);
        assert_eq!(
            secure_boot_from([0x0C, 0x00, sel as u8, (sel >> 8) as u8]),
            SecureBoot::OFF,
            "index 53 is past the table"
        );
        // Selector index 1: bit 0 clear, mirror bit 7 clear, nibble 1 / 0xE.
        let sel: u16 = 0xE1 << 8 | 0b0111_1110;
        assert_eq!(sb_cryp_sel_idx(sel), 1);
        let b1 = 0x0C | 0x10; // WINLIN inbox, cust idx low nibble 1
        let b2 = 0x03 | 0x40; // key num 3, cust idx high bit
        let on = secure_boot_from([b1, b2, sel as u8, (sel >> 8) as u8]);
        assert!(on.secure_boot);
        assert_eq!(on.sb_sel_mgn, 0x0C00_0180);
        assert_eq!(on.mss_dev_type, 0);
        assert_eq!(on.mss_cust_idx, 0x1F - 0x11);
        assert_eq!(on.mss_key_num, 0xF - 3);
        // An unknown device type turns secure boot off.
        assert_eq!(
            secure_boot_from([0x05, b2, sel as u8, (sel >> 8) as u8]),
            SecureBoot::OFF
        );
        // A selector whose mirror bits disagree is invalid.
        assert_eq!(sb_cryp_sel_idx(0xE1 << 8 | 0b0111_1111), u16::MAX);
    }
}
