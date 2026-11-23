//! RTL8922AE PHY0 table execution over the existing held MMIO/DMA seams.
//! Register facts: Linux v6.17 phy.c:1069-1132,1149-1179,1420-1527;
//! phy_be.c:993; rtw8922a.c:1566-1599,2867; fw.c:5980-6007.
//! This establishes register configuration, not calibration or MCU liveness.
use crate::{
    Waits,
    desc::{FwcmdKind, h2c_header},
    fw::phy::{PhyTables, RfPath, TableKind},
    fwdl::FwcmdQueue,
    mac::{self, Op},
    regs::{be::*, encode, reg, replace},
};
use kore_driver::seam::{Dma, RegsNarrow};

const PHY_BASE: u32 = 0x20000;
const RF_MASK: u32 = 0xfffff;
const RF_BUSY_NS: u64 = 3_800_000;
const RF_PAGE_WORDS: usize = 500;
// Normal H2C has 32 precharged 64-byte WDE pages. A full RF page, its
// command header and descriptor must fit without relying on truncation.
const _: () = assert!(
    RF_PAGE_WORDS * 4 + crate::desc::H2C_HDR_BYTES + crate::desc::FWCMD_DESC_BYTES <= 32 * 64
);
const BYPASS: u32 = 0xbabecafe; // core.h:40
const fn phy(offset: u32) -> u32 {
    PHY_BASE + offset
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PhyError {
    RfBusy,
    Command,
    PostInit,
}

/// Only a completely validated plan can reach this executor. The binding
/// already checked the whole BB-MCU aperture, which also encloses PHY0.
/// Data/interrupt admission stays with the later channel and frame owners.
pub(crate) fn initialize<R: RegsNarrow, B: Dma>(
    regs: &R,
    waits: &Waits,
    cut: u8,
    tables: &PhyTables<'_>,
    queue: &mut FwcmdQueue<'_, B>,
    control: &mut crate::control::Control,
    rx: &B,
) -> Result<(), PhyError> {
    for r in tables.baseband.records() {
        if let Some(ns) = bb_delay(r.address) {
            waits.delay(ns);
        } else if r.value != BYPASS {
            regs.write32(reg(phy(r.address)), r.value);
        }
    }
    mac::run(regs, waits, cut, POST_INIT).map_err(|_| PhyError::PostInit)?;
    // Gain data remains borrowed in the radio owner for W2's per-channel
    // decode. Upstream also stores it in host state; it is never raw MMIO.
    for table in &tables.radio {
        let path = match table.kind() {
            TableKind::Radio(RfPath::A) => 0,
            TableKind::Radio(RfPath::B) => 1,
            TableKind::Baseband => unreachable!("the plan contains exactly two RF tables"),
        };
        for r in table.records() {
            if r.address & 0x10000 != 0 {
                let offset = phy([0xe000, 0xf000][path] + (r.address & 0xff) * 4);
                regs.write32(
                    reg(offset),
                    replace(regs.read32(reg(offset)), RF_MASK, r.value),
                );
                waits.delay(1_000);
            } else {
                rf_idle(regs, waits, path)?;
                regs.write32(
                    reg(phy([0x2ae0, 0x2be0][path])),
                    encode(r.address, B_HWSI_DATA_ADDR) | encode(r.value, B_HWSI_DATA_VAL),
                );
            }
        }
        // The last analog write must finish too; a following direct write
        // does not establish HWSI retirement.
        rf_idle(regs, waits, path)?;
        // The reference applies all writes before notifying this path's FW
        // class. Re-walking immutable bytes avoids its 6 KiB allocated copy.
        let mut words = table.records().filter(|r| r.address >= 0x100);
        let mut payload = [0u8; RF_PAGE_WORDS * 4];
        for page in 0..3 {
            let mut len = 0;
            for r in words.by_ref().take(RF_PAGE_WORDS) {
                payload[len..len + 4].copy_from_slice(&((r.address << 20) | r.value).to_le_bytes());
                len += 4;
            }
            if len == 0 {
                break;
            }
            // Normal H2C requests receipt every fourth sequence, even when
            // its caller requests no ack (fw.c:1440). No done ack is defined
            // for RF register pages; TX retirement is not an RF MCU witness.
            let seq = control.next_sequence();
            let header = h2c_header(
                2,
                if path == 0 { 8 } else { 9 },
                page,
                seq,
                len as u16,
                seq % 4 == 0,
                false,
            );
            queue
                .post(
                    regs,
                    waits,
                    FwcmdKind::H2c,
                    Some(header),
                    &payload[..len],
                    None,
                )
                .map_err(|_| PhyError::Command)?;
            control.posted();
        }
        control
            .retire(regs, waits, queue, rx)
            .map_err(|_| PhyError::Command)?;
    }
    Ok(())
}
fn rf_idle<R: RegsNarrow>(regs: &R, waits: &Waits, path: usize) -> Result<(), PhyError> {
    if waits.poll_clear(regs, phy([0x2c24, 0x2d24][path]), 1 << 29, RF_BUSY_NS) {
        Ok(())
    } else {
        Err(PhyError::RfBusy)
    }
}
fn bb_delay(address: u32) -> Option<u64> {
    match address {
        0xfe => Some(50_000_000),
        0xfd => Some(5_000_000),
        0xfc => Some(1_000_000),
        0xfb => Some(50_000),
        0xfa => Some(5_000),
        0xf9 => Some(1_000),
        _ => None,
    }
}
const POST_INIT: &[Op] = &[
    Op::Set32(R_BE_FEN_RST_ENABLE, B_BE_BOOT_RDY0),
    Op::Set32(R_BE_FEN_RST_ENABLE, B_BE_FEN_BBPLAT_RSTB),
    Op::Set32(phy(R_BBCLK), B_CLK_640M),
    Op::Clr32(phy(R_TXSCALE), B_TXFCTR_EN),
    Op::Field32(phy(R_TXFCTR), B_TXFCTR_THD, 0x200),
    Op::Field32(phy(R_SLOPE), B_EHT_RATE_TH, 0xa),
    Op::Field32(phy(R_BEDGE), B_HE_RATE_TH, 0xa),
    Op::Field32(phy(R_BEDGE2), B_HT_VHT_TH, 0xaaa),
    Op::Field32(phy(R_BEDGE), B_EHT_MCS14, 1),
    Op::Field32(phy(R_BEDGE2), B_EHT_MCS15, 1),
    Op::Field32(phy(R_BEDGE3), B_EHTTB_EN, 0),
    Op::Field32(phy(R_BEDGE3), B_HEERSU_EN, 0),
    Op::Field32(phy(R_BEDGE3), B_HEMU_EN, 0),
    Op::Field32(phy(R_BEDGE3), B_TB_EN, 0),
    Op::Field32(phy(R_SU_PUNC), B_SU_PUNC_EN, 1),
    Op::Field32(phy(R_BEDGE5), B_HWGEN_EN, 1),
    Op::Field32(phy(R_BEDGE5), B_PWROFST_COMP, 1),
    Op::Field32(phy(R_MAG_AB), B_BY_SLOPE, 1),
    Op::Field32(phy(R_MAG_A), B_MGA_AEND, 0xe0),
    Op::Field32(phy(R_MAG_AB), B_MAG_AB, 0xe0c000),
    Op::Field32(phy(R_SLOPE), B_SLOPE_A, 0x3fe0),
    Op::Field32(phy(R_SLOPE), B_SLOPE_B, 0x3fe0),
    Op::Field32(phy(R_SC_CORNER), B_SC_CORNER, 0x200),
    Op::Field32(phy(R_UDP_COEEF), B_UDP_COEEF, 0),
    Op::Field32(phy(R_UDP_COEEF), B_UDP_COEEF, 1),
];
