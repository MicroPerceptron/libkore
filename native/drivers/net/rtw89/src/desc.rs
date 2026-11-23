//! **The BE PCI descriptors** — the buffer descriptors both rings are made
//! of, the firmware-command descriptor that heads every packet on channel
//! 12, the header the firmware reads a command through, and the receive
//! descriptor the firmware writes a reply through. Each layout cites the
//! upstream struct it transcribes; every encoder returns bytes, never a
//! Rust struct the device would have to share a representation with.
//!
//! Endianness is explicit little-endian throughout: the device reads
//! these words with its own CPU, and the core must encode identically on
//! every host ISA (the `DmaWrite` rule).

use crate::regs::be::*;
use crate::regs::{encode, get};

/// `struct rtw89_pci_tx_bd_32` (pci.h:1379-1385): `__le16 length; __le16
/// opt; __le32 dma;`.
pub const TX_BD_BYTES: usize = 8;
/// `struct rtw89_pci_rx_bd_32` (pci.h:1433-1438): `__le16 buf_size;
/// __le16 opt; __le32 dma;`.
pub const RX_BD_BYTES: usize = 8;
/// `struct rtw89_pci_rxbd_info` (pci.h:1445-1447): one `__le32` the
/// hardware writes at the head of every receive buffer.
pub const RXBD_INFO_BYTES: usize = 4;
/// `chip->h2c_desc_size` = `sizeof(struct rtw89_rxdesc_short_v2)`
/// (rtw8922a.c:2946, core.h:1113-1120): the 24-byte descriptor that heads
/// every firmware-command packet. The transmit side reuses the receive
/// short-descriptor layout; only dword 0 is written
/// (`rtw89_core_fill_txdesc_fwcmd_v2`, core.c:1577-1585).
pub const FWCMD_DESC_BYTES: usize = 24;
/// `struct rtw89_rxdesc_short_v2` — 6 dwords (core.h:1113-1120).
pub const RXDESC_SHORT_BYTES: usize = 24;
/// `struct rtw89_rxdesc_long_v2` — 10 dwords (core.h:1133-1144).
pub const RXDESC_LONG_BYTES: usize = 40;
/// `struct fwcmd_hdr` (fw.h:4112-4115): two `__le32` in front of an H2C
/// payload (`H2C_HEADER_LEN`, fw.h:4151).
pub const H2C_HDR_BYTES: usize = H2C_HEADER_LEN as usize;
/// `struct rtw89_c2h_hdr` (fw.h:3560-3568): two `__le32` in front of a
/// C2H payload.
pub const C2H_HDR_BYTES: usize = 8;

/// Address bits a transmit or receive buffer descriptor can carry: the
/// low dword plus `RTW89_PCI_TXBD_OPT_DMA_HI` / `RXBD_OPT_DMA_HI`
/// (`GENMASK(13, 6)`, eight bits) = bits 39..32. The part itself asks the
/// host for a 36-bit mask (`dma_set_mask_and_coherent(…, DMA_BIT_MASK(36))`,
/// pci.c:3151), which is the narrower and therefore the binding bound.
pub const DMA_ADDR_BITS: u32 = 36;
const _: () = assert!(DMA_ADDR_BITS <= 32 + (RTW89_PCI_TXBD_OPT_DMA_HI >> 6).count_ones());

/// Whether a device address fits the part's DMA reach (`DMA_ADDR_BITS`).
///
/// The arithmetic is the substrate's — `kore_mt7925` asks the same question
/// of a 32-bit part, so the second caller is where the body was lifted
/// rather than copied. The *width* stays here, because it is a fact about
/// this silicon.
#[inline]
pub const fn addressable(iova: u64, len: u64) -> bool {
    kore_driver::within_dma_reach(iova, len, DMA_ADDR_BITS)
}

/// The high-address byte of a descriptor: bits 39..32 of `iova` in the
/// `opt` word's `DMA_HI` field. Callers prove [`addressable`] first.
#[inline]
const fn dma_hi(iova: u64) -> u16 {
    encode((iova >> 32) as u32, RTW89_PCI_TXBD_OPT_DMA_HI) as u16
}
const _: () = assert!(RTW89_PCI_TXBD_OPT_DMA_HI == RTW89_PCI_RXBD_OPT_DMA_HI);

/// One transmit buffer descriptor: `length` bytes at `iova`, always the
/// last segment (`RTW89_PCI_TXBD_OPT_LS`; channel 12 never chains — the
/// buffer *is* the packet, `rtw89_pci_fwcmd_submit`, pci.c:1447-1451).
#[inline]
pub const fn tx_bd(length: u16, iova: u64) -> [u8; TX_BD_BYTES] {
    let opt = RTW89_PCI_TXBD_OPT_LS as u16 | dma_hi(iova);
    let len = length.to_le_bytes();
    let opt = opt.to_le_bytes();
    let dma = (iova as u32).to_le_bytes();
    [
        len[0], len[1], opt[0], opt[1], dma[0], dma[1], dma[2], dma[3],
    ]
}

/// One receive buffer descriptor: a `buf_size`-byte buffer at `iova`
/// (`rtw89_pci_init_rx_bd`, pci.c:3295).
#[inline]
pub const fn rx_bd(buf_size: u16, iova: u64) -> [u8; RX_BD_BYTES] {
    let size = buf_size.to_le_bytes();
    let opt = dma_hi(iova).to_le_bytes();
    let dma = (iova as u32).to_le_bytes();
    [
        size[0], size[1], opt[0], opt[1], dma[0], dma[1], dma[2], dma[3],
    ]
}

/// The kind of packet a channel-12 descriptor announces
/// (`RTW89_CORE_RX_TYPE_H2C` / `_FWDL`, core.h:201-202).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FwcmdKind {
    /// A command with a [`h2c_header`] in front of its payload.
    H2c,
    /// A raw firmware chunk with no header.
    Fwdl,
}

impl FwcmdKind {
    const fn rpkt_type(self) -> u32 {
        match self {
            Self::H2c => RTW89_CORE_RX_TYPE_H2C,
            Self::Fwdl => RTW89_CORE_RX_TYPE_FWDL,
        }
    }
}

/// The 24-byte descriptor heading a channel-12 packet of `pkt_len` bytes
/// (the bytes *after* this descriptor): dword 0 = `BE_RXD_RPKT_LEN_MASK` |
/// `BE_RXD_RPKT_TYPE_MASK`, the other five zero.
#[inline]
pub const fn fwcmd_desc(pkt_len: u16, kind: FwcmdKind) -> [u8; FWCMD_DESC_BYTES] {
    let dword0 = encode(pkt_len as u32, BE_RXD_RPKT_LEN_MASK)
        | encode(kind.rpkt_type(), BE_RXD_RPKT_TYPE_MASK);
    let w = dword0.to_le_bytes();
    let mut out = [0u8; FWCMD_DESC_BYTES];
    out[0] = w[0];
    out[1] = w[1];
    out[2] = w[2];
    out[3] = w[3];
    out
}
const _: () = assert!(
    BE_RXD_RPKT_LEN_MASK == 0x3FFF,
    "a 16-bit length always encodes"
);

/// The `fwcmd_hdr` in front of an H2C payload of `payload_len` bytes
/// (`rtw89_h2c_pkt_set_hdr` / `_fwdl`, fw.c:1432-1473): `hdr0` = category
/// `[1:0]`, class `[7:2]`, function `[15:8]`, delivery type `[19:16]`
/// (`FWCMD_TYPE_H2C` = 0), sequence `[31:24]`; `hdr1` = total length
/// `[13:0]` (payload + 8) and the two acknowledgement requests.
#[inline]
pub const fn h2c_header(
    cat: u32,
    class: u32,
    func: u32,
    seq: u8,
    payload_len: u16,
    rec_ack: bool,
    done_ack: bool,
) -> [u8; H2C_HDR_BYTES] {
    let hdr0 = encode(FWCMD_TYPE_H2C, H2C_HDR_DEL_TYPE)
        | encode(cat, H2C_HDR_CAT)
        | encode(class, H2C_HDR_CLASS)
        | encode(func, H2C_HDR_FUNC)
        | encode(seq as u32, H2C_HDR_H2C_SEQ);
    let mut hdr1 = encode(payload_len as u32 + H2C_HEADER_LEN, H2C_HDR_TOTAL_LEN);
    if rec_ack {
        hdr1 |= H2C_HDR_REC_ACK;
    }
    if done_ack {
        hdr1 |= H2C_HDR_DONE_ACK;
    }
    let a = hdr0.to_le_bytes();
    let b = hdr1.to_le_bytes();
    [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]
}

/// The hardware's word at the head of a receive buffer
/// (`rtw89_pci_rxbd_info_update`, pci.c:183; masks pci.h:1440-1443).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RxbdInfo {
    /// First segment of a packet.
    pub fs: bool,
    /// Last segment of a packet.
    pub ls: bool,
    /// Bytes the device wrote, including this information word.
    pub write_size: u16,
    /// The sequence tag (`check_rx_tag`: 1 on the first buffer, then +1).
    pub tag: u16,
}

impl RxbdInfo {
    /// Decode the little-endian word.
    pub const fn decode(word: u32) -> Self {
        Self {
            fs: word & RTW89_PCI_RXBD_FS != 0,
            ls: word & RTW89_PCI_RXBD_LS != 0,
            write_size: get(word, RTW89_PCI_RXBD_WRITE_SIZE) as u16,
            tag: get(word, RTW89_PCI_RXBD_TAG) as u16,
        }
    }
}

/// Dword 0 of a BE receive descriptor (`rtw89_core_query_rxdesc_v2`,
/// core.c:2708-2715).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RxDesc {
    /// Payload bytes.
    pub pkt_size: u16,
    /// `RTW89_CORE_RX_TYPE_*`: 10 is a C2H message.
    pub pkt_type: u8,
    /// Whether the descriptor is the 40-byte long form.
    pub long: bool,
    /// Bytes between the descriptor and the payload: shift (2-byte
    /// units), driver info (8), PHY report (8), header conversion (16)
    /// (core.c:2737-2742).
    pub prefix: u16,
}

impl RxDesc {
    /// Decode dword 0.
    pub const fn decode(dword0: u32) -> Self {
        let shift = get(dword0, BE_RXD_SHIFT_MASK) << 1;
        let drv_info = get(dword0, BE_RXD_DRV_INFO_SZ_MASK) << 3;
        let phy_rpt = get(dword0, BE_RXD_PHY_RPT_SZ_MASK) << 3;
        let hdr_cnv = get(dword0, BE_RXD_HDR_CNV_SZ_MASK) << 4;
        Self {
            pkt_size: get(dword0, BE_RXD_RPKT_LEN_MASK) as u16,
            pkt_type: get(dword0, BE_RXD_RPKT_TYPE_MASK) as u8,
            long: dword0 & BE_RXD_LONG_RXD != 0,
            prefix: (shift + drv_info + phy_rpt + hdr_cnv) as u16,
        }
    }

    /// Bytes of descriptor: long or short form.
    pub const fn len(&self) -> usize {
        if self.long {
            RXDESC_LONG_BYTES
        } else {
            RXDESC_SHORT_BYTES
        }
    }

    /// Where the payload starts, from the descriptor's own first byte
    /// (`desc_info->offset + desc_info->rxd_len` with the 4-byte
    /// `rxbd_info` already stripped, pci.c:376).
    pub const fn payload_at(&self) -> usize {
        self.prefix as usize + self.len()
    }
}

/// A C2H header (fw.h:3565-3568).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct C2hHeader {
    /// `w0[1:0]`.
    pub category: u8,
    /// `w0[7:2]`.
    pub class: u8,
    /// `w0[15:8]`.
    pub func: u8,
    /// `w1[13:0]`: the whole message including this header.
    pub len: u16,
}

impl C2hHeader {
    /// Decode the two little-endian words.
    pub const fn decode(w0: u32, w1: u32) -> Self {
        Self {
            category: get(w0, RTW89_C2H_HDR_W0_CATEGORY) as u8,
            class: get(w0, RTW89_C2H_HDR_W0_CLASS) as u8,
            func: get(w0, RTW89_C2H_HDR_W0_FUNC) as u8,
            len: get(w1, RTW89_C2H_HDR_W1_LEN) as u16,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_descriptors_carry_forty_address_bits_little_endian() {
        let bd = tx_bd(0x1234, 0x0A_8000_0010);
        assert_eq!(&bd[0..2], &[0x34, 0x12]);
        // opt: LS (bit 14) | 0x0A << 6.
        let opt = u16::from_le_bytes([bd[2], bd[3]]);
        assert_eq!(opt, 0x4000 | (0x0A << 6));
        assert_eq!(
            u32::from_le_bytes([bd[4], bd[5], bd[6], bd[7]]),
            0x8000_0010
        );
        let rx = rx_bd(11498, 0x0000_1000);
        assert_eq!(u16::from_le_bytes([rx[0], rx[1]]), 11498);
        assert_eq!(u16::from_le_bytes([rx[2], rx[3]]), 0);
        assert!(addressable((1 << 36) - 4096, 4096));
        assert!(!addressable((1 << 36) - 4095, 4096));
    }

    #[test]
    fn the_fwcmd_descriptor_and_header_match_the_dossier_wire_bytes() {
        // Dossier §F.7: a 16-byte H2C → dword0 = 16 | 13 << 24.
        let d = fwcmd_desc(16, FwcmdKind::H2c);
        assert_eq!(u32::from_le_bytes([d[0], d[1], d[2], d[3]]), 0x0D00_0010);
        assert!(d[4..].iter().all(|&b| b == 0));
        let f = fwcmd_desc(2020, FwcmdKind::Fwdl);
        assert_eq!(u32::from_le_bytes([f[0], f[1], f[2], f[3]]), 0x0E00_07E4);
        // hdr0 = CAT(1) | CLASS(9 << 2) | FUNC(0x14 << 8) | seq << 24;
        // hdr1 = 0x8010 with done-ack, 0xC010 with both.
        let h = h2c_header(1, 9, 0x14, 7, 8, true, true);
        assert_eq!(u32::from_le_bytes([h[0], h[1], h[2], h[3]]), 0x0700_1425);
        assert_eq!(u32::from_le_bytes([h[4], h[5], h[6], h[7]]), 0xC010);
        let hdr = h2c_header(1, 3, 0, 0, 64, false, false);
        assert_eq!(u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]), 72);
    }

    #[test]
    fn receive_decoders_locate_a_plain_c2h_at_offset_28() {
        let info = RxbdInfo::decode((1 << 16) | (1 << 15) | (1 << 14) | 40);
        assert_eq!(
            info,
            RxbdInfo {
                fs: true,
                ls: true,
                write_size: 40,
                tag: 1
            }
        );
        let rxd = RxDesc::decode((10 << 24) | 12);
        assert_eq!(rxd.pkt_size, 12);
        assert_eq!(rxd.pkt_type, 10);
        assert!(!rxd.long);
        assert_eq!(RXBD_INFO_BYTES + rxd.payload_at(), 28);
        let long = RxDesc::decode((1 << 31) | (1 << 18) | (2 << 14) | 5);
        assert_eq!(long.len(), 40);
        assert_eq!(long.prefix, 8 + 4);
        let c2h = C2hHeader::decode((1 << 8) | (0 << 2) | 1, 12);
        assert_eq!(c2h.category, 1);
        assert_eq!(c2h.func, 1);
        assert_eq!(c2h.len, 12);
    }
}
