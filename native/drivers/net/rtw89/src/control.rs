//! Startup MAC command completion over the already leased RXQ0 buffers.
//! Wire/ring facts: Linux v6.17 pci.c:61-90,183-246,282-322,414-436;
//! rtw8922ae.c:34-36; fw.h:1754-1756,3586-3606; fw.c:2475,4364.
//! Polling is bounded and init-only. It grants no on-air RX or IRQ authority.
use crate::{
    Waits,
    desc::{C2hHeader, FwcmdKind, RxDesc, RxbdInfo, h2c_header},
    fwdl::FwcmdQueue,
    nic::{RX_BUF_BYTES, RX_RING},
    regs::{RXQ0, be::*, get, reg},
};
use core::sync::atomic::{Ordering, fence};
use kore_driver::seam::{Dma, DmaRead, DmaShared, RegsNarrow};

const COMMAND_NS: u64 = 1_000_000_000;
const MAX_MESSAGES: usize = 128;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ControlError {
    Tx,
    Timeout,
    Shape,
    Index,
    Rejected(u8),
    Flood,
}

/// Move-only progress: a live command/RX ring must never restart at zero.
pub(crate) struct Control {
    sequence: u8,
    next: u16,
    tag: u16,
    features: bool,
}
impl Control {
    pub(crate) const fn new() -> Self {
        Self {
            sequence: 0,
            next: 0,
            tag: 1,
            features: false,
        }
    }
    pub(crate) fn next_sequence(&self) -> u8 {
        self.sequence
    }
    pub(crate) fn posted(&mut self) {
        self.sequence = self.sequence.wrapping_add(1);
    }
    pub(crate) fn features_ready(&self) -> bool {
        self.features
    }

    pub(crate) fn initialize<R: RegsNarrow, B: Dma>(
        &mut self,
        regs: &R,
        waits: &Waits,
        tx: &mut FwcmdQueue<'_, B>,
        rx: &B,
    ) -> Result<(), ControlError> {
        // BACAM_V1: one 1024-bit BA user per band, four entries apart.
        // Reserving users does not install an association, BA session or key.
        self.command(regs, waits, tx, rx, 12, 2, &0x0000_0001u32.to_le_bytes())?;
        self.command(regs, waits, tx, rx, 12, 2, &0x0100_4001u32.to_le_bytes())?;
        self.command(regs, waits, tx, rx, 9, 0x14, &[9, 0, 0, 0, 0x5e, 0, 0, 0])?;
        self.features = true;
        Ok(())
    }

    fn command<R: RegsNarrow, B: Dma>(
        &mut self,
        regs: &R,
        waits: &Waits,
        tx: &mut FwcmdQueue<'_, B>,
        rx: &B,
        class: u8,
        func: u8,
        payload: &[u8],
    ) -> Result<(), ControlError> {
        let seq = self.sequence;
        let identity = 1 | u32::from(class) << 2 | u32::from(func) << 8;
        let header = h2c_header(
            1,
            u32::from(class),
            u32::from(func),
            seq,
            payload.len() as u16,
            seq % 4 == 0,
            true,
        );
        tx.post(regs, waits, FwcmdKind::H2c, Some(header), payload, None)
            .map_err(|_| ControlError::Tx)?;
        self.posted();
        self.wait(regs, waits, rx, Some((identity, seq)), || {
            tx.poll_retired(regs).map_err(|_| ControlError::Tx)
        })
    }

    /// RF notifications have no done-ack, but RX must still make progress
    /// while TX retires. This consumes only unsolicited startup C2Hs.
    pub(crate) fn retire<R: RegsNarrow, B: Dma>(
        &mut self,
        regs: &R,
        waits: &Waits,
        tx: &mut FwcmdQueue<'_, B>,
        rx: &B,
    ) -> Result<(), ControlError> {
        self.wait(regs, waits, rx, None, || {
            tx.poll_retired(regs).map_err(|_| ControlError::Tx)
        })
    }

    fn wait<R: RegsNarrow, B: Dma>(
        &mut self,
        regs: &R,
        waits: &Waits,
        rx: &B,
        expected: Option<(u32, u8)>,
        mut retired: impl FnMut() -> Result<bool, ControlError>,
    ) -> Result<(), ControlError> {
        let mut budget = waits.budget(COMMAND_NS);
        let mut messages = 0;
        let mut done = expected.is_none();
        loop {
            let tx_done = retired()?;
            if let Some(ack) = self.poll(regs, rx)? {
                messages += 1;
                if let (Some(word), Some((identity, seq))) = (ack, expected) {
                    if word & 0xffff == identity && (word >> 24) as u8 == seq {
                        let status = (word >> 16) as u8;
                        if status != 0 {
                            return Err(ControlError::Rejected(status));
                        }
                        done = true;
                    }
                }
                if messages >= MAX_MESSAGES {
                    return Err(ControlError::Flood);
                }
            }
            // C2H success and DMA retirement are independent conditions.
            if done && tx_done {
                return Ok(());
            }
            if !budget.may_wait() {
                return Err(ControlError::Timeout);
            }
            (waits.pause)();
        }
    }

    /// None = no completed slot; Some(None) = an unrelated C2H was retired.
    /// Only a complete, bounded done-ack yields its identity/status word.
    fn poll<R: RegsNarrow, B: Dma>(
        &mut self,
        regs: &R,
        rx: &B,
    ) -> Result<Option<Option<u32>>, ControlError> {
        let hw = get(regs.read32(reg(RXQ0.idx)), TXBD_HW_IDX_MASK) as u16;
        if hw >= RX_RING {
            return Err(ControlError::Index);
        }
        if hw == self.next {
            return Ok(None);
        }
        let at = self.next as usize * RX_BUF_BYTES;
        let shared = DmaShared::new(rx);
        let info = RxbdInfo::decode(shared.read_u32_at(at).map_err(|_| ControlError::Shape)?);
        // The hardware index can precede the last writeback. A stale tag
        // neither exposes payload bytes nor returns this buffer to the NIC.
        if info.tag != self.tag {
            return Ok(None);
        }
        fence(Ordering::Acquire);
        let size = usize::from(info.write_size);
        // 8922AE has no FS bit. Startup C2Hs fit one buffer and require LS;
        // segmented/data traffic requires the later frame-path owner.
        if !info.ls || !(4 + 24..=RX_BUF_BYTES).contains(&size) {
            return Err(ControlError::Shape);
        }
        let desc = RxDesc::decode(rx.read_u32_at(at + 4).map_err(|_| ControlError::Shape)?);
        let payload = 4 + desc.payload_at();
        let end = payload + usize::from(desc.pkt_size);
        if desc.pkt_type != RTW89_CORE_RX_TYPE_C2H as u8 || desc.pkt_size < 8 || end > size {
            return Err(ControlError::Shape);
        }
        let w0 = rx
            .read_u32_at(at + payload)
            .map_err(|_| ControlError::Shape)?;
        let w1 = rx
            .read_u32_at(at + payload + 4)
            .map_err(|_| ControlError::Shape)?;
        let h = C2hHeader::decode(w0, w1);
        if h.len != desc.pkt_size {
            return Err(ControlError::Shape);
        }
        let ack = if h.category == 1 && h.class == 0 && h.func <= 1 {
            if h.len != 12 {
                return Err(ControlError::Shape);
            }
            if h.func == 1 {
                Some(
                    rx.read_u32_at(at + payload + 8)
                        .map_err(|_| ControlError::Shape)?,
                )
            } else {
                None
            } // Receipt is not completion; its seq uses other bits.
        } else {
            None
        }; // Startup log/unsolicited C2Hs carry no command authority.
        // No payload borrow survives returning the slot. RXBD itself keeps
        // its original buffer address. A release orders reads before reuse.
        fence(Ordering::Release);
        regs.write16(reg(RXQ0.idx), self.next);
        self.next = (self.next + 1) % RX_RING;
        self.tag = if self.tag == 0x1fff { 1 } else { self.tag + 1 };
        Ok(Some(ack))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::{Faults, MockRtw89};
    use kore_driver::{
        host_double::{HostBuf, HostPool},
        seam::{DmaPool, DmaWrite, Regs},
    };
    fn setup() -> (MockRtw89, HostBuf, Control) {
        (
            MockRtw89::new(&[0xff; crate::efuse::PHYSICAL_SIZE], Faults::default()),
            HostPool.alloc(RX_RING as usize * RX_BUF_BYTES).unwrap(),
            Control::new(),
        )
    }
    fn packet(rx: &HostBuf, slot: u16, tag: u16, func: u32, ack: u32) {
        let at = slot as usize * RX_BUF_BYTES;
        for (i, word) in [
            40 | RTW89_PCI_RXBD_LS | u32::from(tag) << 16,
            12 | RTW89_CORE_RX_TYPE_C2H << 24,
            0,
            0,
            0,
            0,
            0,
            1 | func << 8,
            12,
            ack,
        ]
        .into_iter()
        .enumerate()
        {
            rx.write_u32_at(at + i * 4, word).unwrap();
        }
    }
    fn ready(dev: &MockRtw89, next: u16) {
        dev.write32(
            reg(RXQ0.idx),
            u32::from((next + RX_RING - 1) % RX_RING) | u32::from((next + 1) % RX_RING) << 16,
        );
    }
    #[test]
    fn index_and_tag_both_precede_payload_ownership() {
        let (dev, rx, mut c) = setup();
        packet(&rx, 0, 0, 1, 0x02000231);
        ready(&dev, 0);
        let idx = dev.word(RXQ0.idx);
        assert_eq!(c.poll(&dev, &rx), Ok(None));
        assert_eq!(dev.word(RXQ0.idx), idx);
        packet(&rx, 0, 1, 1, 0x02000231);
        assert_eq!(c.poll(&dev, &rx), Ok(Some(Some(0x02000231))));
        assert_eq!(dev.word(RXQ0.idx), 1 << 16);
        assert_eq!(c.poll(&dev, &rx), Ok(None));
    }
    #[test]
    fn receive_cursor_and_tag_wrap_without_replaying_a_slot() {
        let (dev, rx, mut c) = setup();
        for n in 0..8200u32 {
            let slot = (n % u32::from(RX_RING)) as u16;
            let tag = (n % 8191 + 1) as u16;
            packet(&rx, slot, tag, 1, n);
            ready(&dev, slot);
            assert_eq!(c.poll(&dev, &rx), Ok(Some(Some(n))));
            assert_eq!(c.poll(&dev, &rx), Ok(None));
        }
        assert_eq!(c.tag, 10);
    }
    #[test]
    fn malformed_completed_envelopes_never_return_a_buffer() {
        for (off, value) in [
            (0, 39 | RTW89_PCI_RXBD_LS | 1 << 16),
            (0, 40 | 1 << 16),
            (0, 0x3fff | RTW89_PCI_RXBD_LS | 1 << 16),
            (4, 12),
            (4, 7 | RTW89_CORE_RX_TYPE_C2H << 24),
            (4, 13 | RTW89_CORE_RX_TYPE_C2H << 24),
            (32, 11),
            (4, 12 | RTW89_CORE_RX_TYPE_C2H << 24 | BE_RXD_LONG_RXD),
        ] {
            let (dev, rx, mut c) = setup();
            packet(&rx, 0, 1, 1, 0);
            rx.write_u32_at(off, value).unwrap();
            ready(&dev, 0);
            let idx = dev.word(RXQ0.idx);
            assert_eq!(c.poll(&dev, &rx), Err(ControlError::Shape));
            assert_eq!(dev.word(RXQ0.idx), idx);
        }
        let (dev, rx, mut c) = setup();
        dev.write32(reg(RXQ0.idx), u32::from(RX_RING) << 16);
        assert_eq!(c.poll(&dev, &rx), Err(ControlError::Index));
    }
    #[test]
    fn long_descriptor_and_all_prefix_units_locate_the_same_ack() {
        let (dev, rx, mut c) = setup();
        let prefix = 2 + 8 + 8 + 16;
        let payload = 4 + 40 + prefix;
        rx.write_u32_at(0, (payload + 12) as u32 | RTW89_PCI_RXBD_LS | 1 << 16)
            .unwrap();
        let d = 12
            | RTW89_CORE_RX_TYPE_C2H << 24
            | BE_RXD_LONG_RXD
            | crate::regs::encode(1, BE_RXD_SHIFT_MASK)
            | crate::regs::encode(1, BE_RXD_DRV_INFO_SZ_MASK)
            | crate::regs::encode(1, BE_RXD_PHY_RPT_SZ_MASK)
            | crate::regs::encode(1, BE_RXD_HDR_CNV_SZ_MASK);
        rx.write_u32_at(4, d).unwrap();
        rx.write_u32_at(payload, 0x101).unwrap();
        rx.write_u32_at(payload + 4, 12).unwrap();
        rx.write_u32_at(payload + 8, 0x04021425).unwrap();
        ready(&dev, 0);
        assert_eq!(c.poll(&dev, &rx), Ok(Some(Some(0x04021425))));
    }
    #[test]
    fn successful_ack_is_latched_until_tx_retires() {
        let (dev, rx, mut c) = setup();
        packet(&rx, 0, 1, 1, 0x02000231);
        ready(&dev, 0);
        let mut probes = 0;
        let waits = Waits {
            clock: || None,
            pause: || {},
        };
        assert_eq!(
            c.wait(&dev, &waits, &rx, Some((0x231, 2)), || {
                probes += 1;
                if probes > 1 {
                    assert_eq!(dev.word(RXQ0.idx), 1 << 16);
                }
                Ok(probes == 3)
            }),
            Ok(())
        );
        assert_eq!(probes, 3);
    }

    #[test]
    fn continuous_unsolicited_messages_do_not_renew_the_budget() {
        let (dev, rx, mut c) = setup();
        let waits = Waits {
            clock: || Some(0),
            pause: || {},
        };
        let mut n = 0;
        let result = c.wait(&dev, &waits, &rx, Some((0x231, 2)), || {
            let slot = (n % usize::from(RX_RING)) as u16;
            packet(&rx, slot, (n + 1) as u16, 2, 0);
            ready(&dev, slot);
            n += 1;
            Ok(true)
        });
        assert_eq!(result, Err(ControlError::Flood));
        assert_eq!(n, MAX_MESSAGES);
    }

    #[test]
    fn receipts_and_wrong_command_identities_cannot_complete_a_wait() {
        let waits = Waits {
            clock: || None,
            pause: || {},
        };
        for (func, word) in [
            (0, 0x02000231),
            (1, 0x03000231),
            (1, 0x02000331),
            (1, 0x02000235),
            (1, 0x02000230),
        ] {
            let (dev, rx, mut c) = setup();
            packet(&rx, 0, 1, func, word);
            ready(&dev, 0);
            assert_eq!(
                c.wait(&dev, &waits, &rx, Some((0x231, 2)), || Ok(true)),
                Err(ControlError::Timeout)
            );
        }
        let (dev, rx, mut c) = setup();
        packet(&rx, 0, 1, 1, 0x02070231);
        ready(&dev, 0);
        assert_eq!(
            c.wait(&dev, &waits, &rx, Some((0x231, 2)), || Ok(true)),
            Err(ControlError::Rejected(7))
        );
    }
}
