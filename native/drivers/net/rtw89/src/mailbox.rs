//! **The register mailbox** — four H2C data words, four C2H data words, a
//! trigger byte each way and a two-nibble counter byte
//! (`rtw89_fw_write_h2c_reg` / `rtw89_fw_read_c2h_reg`, fw.c:6790-6862).
//! Independent of the DMA rings, which is what makes it the D1 witness:
//! a reply here is the WCPU running *this* firmware and parsing *this*
//! host's words, with no ring or descriptor in the loop to be wrong.
//!
//! The exchange is the one upstream runs at probe, `GET_FEATURE` part 0 →
//! `PHY_CAP` (`rtw89_mac_read_phycap`, mac.c:2949-2991), bracketed by the
//! efuse-state conversion.

use kore_driver::seam::RegsNarrow;

use crate::Waits;
use crate::efuse::cnv_efuse_state;
use crate::regs::be::*;
use crate::regs::{
    C2H_COUNTER_MASK, C2HREG_DATA, H2C_COUNTER_MASK, H2CREG_DATA, MAILBOX_COUNTER_BYTE, encode,
    get, reg, replace,
};

/// The firmware's own idle poll before a command may be written
/// (`read_poll_timeout(rtw89_read8, val, val == 0, 1000, 5000, …)`): 5 ms.
const H2C_IDLE_NS: u64 = 5_000_000;
/// `RTW89_C2H_TIMEOUT` (fw.h:117), in microseconds: 1 s for a reply.
const C2H_REPLY_NS: u64 = RTW89_C2H_TIMEOUT as u64 * 1_000;

/// Why the mailbox exchange did not complete.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MailboxError {
    /// The firmware never released the H2C trigger from the last command.
    Busy,
    /// No reply within the budget.
    NoReply,
    /// A reply with another function code than the one asked for.
    WrongReply(u8),
}

/// The mailbox's running counters (`fw_info->h2c_counter` /
/// `c2h_counter`), reset to zero by the download.
#[derive(Clone, Copy, Debug, Default)]
pub struct Counters {
    h2c: u8,
    c2h: u8,
}

/// What the firmware answers to `GET_FEATURE` part 0
/// (`struct rtw89_c2hreg_phycap`, fw.h:31-50): the PHY's shape as the
/// firmware sees it — recorded as D1's evidence and the input D2's
/// capability advertisement is built from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PhyCap {
    /// Receive spatial streams.
    pub rx_nss: u8,
    /// The widest bandwidth.
    pub bw: u8,
    /// Transmit spatial streams.
    pub tx_nss: u8,
    /// Protocol capability.
    pub prot: u8,
    /// NIC type.
    pub nic: u8,
    /// WLAN function bits.
    pub wl_func: u8,
    /// Hardware type.
    pub hw_type: u8,
    /// Transmit antennas.
    pub ant_tx_num: u8,
    /// Receive antennas.
    pub ant_rx_num: u8,
    /// Band selection.
    pub band_sel: u8,
}

/// `rtw89_fw_write_h2c_reg`: wait for the trigger to be clear, write the
/// four words with the function and dword count stamped into word 0,
/// bump the H2C nibble, pull the trigger.
fn write_h2c<R: RegsNarrow>(
    regs: &R,
    waits: &Waits,
    counters: &mut Counters,
    func: u8,
    content_len: u8,
    mut words: [u32; RTW89_H2CREG_MAX as usize],
) -> Result<(), MailboxError> {
    if !waits.poll_clear8(regs, R_BE_H2CREG_CTRL, u8::MAX, H2C_IDLE_NS) {
        return Err(MailboxError::Busy);
    }
    let dwords = (content_len as u32 + RTW89_H2CREG_HDR_LEN).div_ceil(4);
    words[0] = replace(words[0], RTW89_H2CREG_HDR_FUNC_MASK, func as u32);
    words[0] = replace(words[0], RTW89_H2CREG_HDR_LEN_MASK, dwords);
    for (r, w) in H2CREG_DATA.iter().zip(words.iter()) {
        regs.write32(reg(*r), *w);
    }
    counters.h2c = counters.h2c.wrapping_add(1);
    let byte = regs.read8(reg(MAILBOX_COUNTER_BYTE));
    regs.write8(
        reg(MAILBOX_COUNTER_BYTE),
        (byte & !H2C_COUNTER_MASK)
            | (counters.h2c << H2C_COUNTER_MASK.trailing_zeros()) & H2C_COUNTER_MASK,
    );
    regs.write8(reg(R_BE_H2CREG_CTRL), B_AX_H2CREG_TRIGGER as u8);
    Ok(())
}

/// `rtw89_fw_read_c2h_reg`: wait for the reply flag, read the four words,
/// drop the flag, bump the C2H nibble. Returns the function code and the
/// words.
fn read_c2h<R: RegsNarrow>(
    regs: &R,
    waits: &Waits,
    counters: &mut Counters,
) -> Result<(u8, [u32; RTW89_C2HREG_MAX as usize]), MailboxError> {
    if !waits.poll_set8(regs, R_BE_C2HREG_CTRL, u8::MAX, C2H_REPLY_NS) {
        return Err(MailboxError::NoReply);
    }
    let mut words = [0u32; RTW89_C2HREG_MAX as usize];
    for (r, w) in C2HREG_DATA.iter().zip(words.iter_mut()) {
        *w = regs.read32(reg(*r));
    }
    regs.write8(reg(R_BE_C2HREG_CTRL), 0);
    let func = get(words[0], RTW89_C2HREG_HDR_FUNC_MASK) as u8;
    counters.c2h = counters.c2h.wrapping_add(1);
    let byte = regs.read8(reg(MAILBOX_COUNTER_BYTE));
    regs.write8(
        reg(MAILBOX_COUNTER_BYTE),
        (byte & !C2H_COUNTER_MASK)
            | (counters.c2h << C2H_COUNTER_MASK.trailing_zeros()) & C2H_COUNTER_MASK,
    );
    Ok((func, words))
}

/// `rtw89_mac_read_phycap(part_num = 0)`: the whole exchange, bracketed
/// by the efuse-state conversion, expecting `PHY_CAP` back.
pub fn get_phy_cap<R: RegsNarrow>(
    regs: &R,
    waits: &Waits,
    counters: &mut Counters,
) -> Result<PhyCap, MailboxError> {
    /// `content_len` on a BE part (mac.c:2959-2962).
    const CONTENT_LEN: u8 = 2;
    /// `GET_FEATURE` part 0.
    const PART_NUM: u32 = 0;
    cnv_efuse_state(regs, waits, false);
    let words = [encode(PART_NUM, RTW89_H2CREG_GET_FEATURE_PART_NUM), 0, 0, 0];
    let result = write_h2c(
        regs,
        waits,
        counters,
        RTW89_FWCMD_H2CREG_FUNC_GET_FEATURE as u8,
        CONTENT_LEN,
        words,
    )
    .and_then(|()| read_c2h(regs, waits, counters))
    .and_then(|(func, w)| {
        if func != RTW89_FWCMD_C2HREG_FUNC_PHY_CAP as u8 {
            return Err(MailboxError::WrongReply(func));
        }
        Ok(PhyCap {
            rx_nss: get(w[0], RTW89_C2HREG_PHYCAP_W0_RX_NSS) as u8,
            bw: get(w[0], RTW89_C2HREG_PHYCAP_W0_BW) as u8,
            tx_nss: get(w[1], RTW89_C2HREG_PHYCAP_W1_TX_NSS) as u8,
            prot: get(w[1], RTW89_C2HREG_PHYCAP_W1_PROT) as u8,
            nic: get(w[1], RTW89_C2HREG_PHYCAP_W1_NIC) as u8,
            wl_func: get(w[1], RTW89_C2HREG_PHYCAP_W1_WL_FUNC) as u8,
            hw_type: get(w[2], RTW89_C2HREG_PHYCAP_W2_HW_TYPE) as u8,
            ant_tx_num: get(w[3], RTW89_C2HREG_PHYCAP_W3_ANT_TX_NUM) as u8,
            ant_rx_num: get(w[3], RTW89_C2HREG_PHYCAP_W3_ANT_RX_NUM) as u8,
            band_sel: get(w[3], RTW89_C2HREG_PHYCAP_W3_BAND_SEL) as u8,
        })
    });
    cnv_efuse_state(regs, waits, true);
    result
}
