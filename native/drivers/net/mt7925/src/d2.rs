//! **M1 — the configuration plane.**
//!
//! C2 left the part with running firmware and nothing else. This rung asks
//! it two questions and tells it one thing, and every byte of that rides
//! the *other* command class — see [`crate::mcu::MCU_UNI_TXD_BYTES`] for
//! why that is a separate encoder rather than a flag.
//!
//! What it is for is **not** the three commands. It is that the answers
//! carry facts no image on this machine contains:
//!
//! - the part's own MAC address, out of its efuse, which is in no blob
//!   this driver ships and in no constant in this tree;
//! - a block of that efuse read back directly;
//! - a capability reply whose tagged records account for the body exactly.
//!
//! A stub that replayed a C2 answer fails the triage [`crate::d1`] added;
//! one that returns a canned header fails the walk; one that invents a MAC
//! fails against a value `res/MT7925_EVIDENCE.md` §1 read off this laptop
//! under Linux.
//!
//! # What this rung deliberately does not do
//!
//! **No `mac_init`.** Upstream writes a dozen registers across two bands
//! before frames can arrive. They prepare the *receive path*, nothing here
//! can observe them, and a boot that changed them as well as this would
//! have a note that could not say which spoke. They belong with M3.
//!
//! **No CLC match.** The efuse's `HW_TYPE` byte selects which regulatory
//! record applies, and checking it means walking the CLC region's own
//! record format. That is M2's work; this rung records the byte.
//!
//! **No firmware log.** `fw_log_2_host` pushes unsolicited events into the
//! very ring an exchange polls. It is the last thing upstream enables, not
//! a prerequisite, and turning it on here would make this rung's failures
//! unreadable.

use kore_driver::seam::{Dma, Regs};

use crate::d0::Core;
use crate::d1::{D1Refusal, Rings};
use crate::mcu::{self, NicCapability, UniCommand};

/// What the part said about itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhyReport {
    /// The capability reply, walked.
    pub capability: NicCapability,
    /// The efuse byte at [`mcu::MT_EE_HW_TYPE`], whole.
    ///
    /// Its low two bits (`MT_EE_HW_TYPE_ENCAP`) select a regulatory
    /// record; the rest is carried because this rung has no source for
    /// what the other six bits mean and will not pretend otherwise.
    pub hw_type: u8,
}

/// How far M1 got, and what refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhyRefusal {
    /// A command did not complete a round trip.
    Transport {
        /// Which of the three. See [`PhyStep`].
        step: PhyStep,
        /// The transport refusal underneath, whole.
        inner: D1Refusal,
        /// What the part said about its own firmware, read the moment the
        /// command failed. See [`FirmwareState`]. `None` only for a refusal
        /// raised before any register was in reach.
        after: Option<FirmwareState>,
    },
    /// The capability reply did not parse.
    Capability(mcu::CapaError),
    /// The reply carried no `MT_NIC_CAP_MAC_ADDR`.
    ///
    /// Distinct from a parse failure: the walk succeeded and the part
    /// chose not to tell us who it is, which is a different fault from a
    /// reply this driver could not read.
    NoMacAddress {
        /// How many records the walk did consume.
        ///
        /// Carried because without it an empty reply and a full one
        /// missing this single record read identically on the only
        /// screen that can report either — and they are different
        /// faults, one in the transport and one in the firmware.
        elements: u16,
    },
    /// The reply carried no `MT_NIC_CAP_PHY`, so there is no stream count
    /// and no band list.
    NoPhyCapability {
        /// How many records the walk did consume, for the reason
        /// [`Self::NoMacAddress`] carries it.
        elements: u16,
    },
    /// The efuse reply was shorter than the block it should carry.
    EfuseShort {
        /// How many body bytes arrived.
        got: usize,
    },
    /// The caller's scratch buffer could not hold a reply.
    ScratchTooSmall {
        /// What it would have taken.
        need: usize,
        /// What was offered.
        have: usize,
    },
}

/// **The firmware's own state, read when a command goes unanswered.**
///
/// M1b's second metal photograph (2026-11-18) proved the capability query
/// left the host — the command ring's engine index reached the doorbell —
/// and that nothing came back on the event ring. That leaves the firmware:
/// it died, it raised an alarm, or it is running and ignored the command.
/// These two registers separate those three, and neither is written here.
///
/// Carried raw, not interpreted: what a bit *means* on this part is a
/// belief, and the photograph is where it gets tested.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FirmwareState {
    /// `MT_CONN_ON_MISC` — the register C2's last rung polled for
    /// `MT_TOP_MISC2_FW_N9_RDY`. Still ready after the silence says the
    /// firmware did not die of the command.
    pub misc: u32,
    /// `MT_MCU_CMD` — upstream's `MCU2HOST_SW_INT_STA`, where the firmware
    /// raises `STOP_DMA`, `RESET_DONE`, `RECOVERY_DONE` and their kin
    /// (`MT_MCU_CMD_ERROR_MASK`, `mt792x_regs.h:274-281`). Upstream's IRQ
    /// handler reads it and writes it back to acknowledge
    /// (`mt792x_dma.c:49-51`); **this only reads**, so a flag the
    /// firmware raised is still standing for the photograph.
    pub mcu_cmd: u32,
}

impl FirmwareState {
    /// The one byte a note has room for.
    ///
    /// ```text
    /// bit 0     N9 still ready  (misc & N9_RDY == N9_RDY)
    /// bits 1–5  MT_MCU_CMD bits 1–5, verbatim (the error mask)
    /// bit 6     MT_MCU_CMD bit 0, WAKE_RX_PCIE
    /// bit 7     always set: the registers were read
    /// ```
    ///
    /// Bit 7 exists because every other bit can honestly be zero, and a
    /// zero byte must never be mistaken for "not read".
    pub const fn byte(self) -> u8 {
        use crate::regs::mt7925::{
            MT_MCU_CMD_ERROR_MASK, MT_MCU_CMD_WAKE_RX_PCIE, MT_TOP_MISC2_FW_N9_RDY,
        };
        let ready = (self.misc & MT_TOP_MISC2_FW_N9_RDY == MT_TOP_MISC2_FW_N9_RDY) as u8;
        let errors = (self.mcu_cmd & MT_MCU_CMD_ERROR_MASK) as u8;
        let wake = ((self.mcu_cmd & MT_MCU_CMD_WAKE_RX_PCIE != 0) as u8) << 6;
        0x80 | wake | errors | ready
    }
}

/// Which command of the three a refusal belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhyStep {
    /// `UNI_CHIP_CONFIG_NIC_CAPA` — describe yourself.
    Capability,
    /// `UNI_EFUSE_ACCESS` — read the block holding `MT_EE_HW_TYPE`.
    EfuseRead,
    /// `UNI_EFUSE_BUFFER_MODE` — the efuse is the EEPROM.
    BufferMode,
}

impl PhyStep {
    /// The rung this step reports, continuing the C2 ladder's numbering so
    /// one note band still spans the whole bring-up.
    pub const fn rung(self) -> u8 {
        match self {
            Self::Capability => 22,
            Self::EfuseRead => 23,
            Self::BufferMode => 24,
        }
    }
}

impl PhyRefusal {
    /// The rung this refusal reports.
    pub const fn rung(self) -> u8 {
        match self {
            Self::Transport { step, .. } => step.rung(),
            Self::Capability(_) => 25,
            Self::NoMacAddress { .. } => 26,
            Self::NoPhyCapability { .. } => 27,
            Self::EfuseShort { .. } => 28,
            Self::ScratchTooSmall { .. } => 29,
        }
    }
}

impl<'r, R: Regs + ?Sized> Core<'r, R> {
    /// Read [`FirmwareState`]. Two reads, no writes — `misc` through the
    /// same reader C2's last rung uses, so both rungs mean one register.
    pub fn firmware_snapshot(&self) -> FirmwareState {
        FirmwareState {
            misc: self.firmware_state(),
            mcu_cmd: self.read(crate::regs::mt7925::MT_MCU_CMD),
        }
    }

    /// **M1** — ask the part to describe itself, read a block of its
    /// efuse, and name the efuse as the EEPROM.
    ///
    /// `mt7925_run_firmware` (`mt7925/mcu.c:1190-1209`) and
    /// `__mt7925_init_hardware` (`mt7925/init.c:100-131`), minus
    /// everything this rung cannot observe — see the module docs.
    ///
    /// `scratch` receives each reply's body. A capability reply is a few
    /// hundred bytes on this part, but the size is the firmware's choice,
    /// so the buffer is the caller's and too small is a refusal carrying
    /// both numbers rather than a silent truncation.
    pub fn bring_up_phy<D: Dma + ?Sized>(
        &self,
        rings: &mut Rings<'_, D>,
        scratch: &mut [u8],
        mut delay: impl FnMut() + Copy,
        now_ns: fn() -> Option<u64>,
    ) -> Result<PhyReport, PhyRefusal> {
        // 1. "Describe yourself."
        //
        // **This command asks for no acknowledgement and is waited on
        // anyway** — `CHIP_CONFIG` clears `MCU_CMD_ACK`
        // (`mt7925/mcu.c:3810-3812`) while `get_nic_capability` passes
        // `wait_resp = true`. That is upstream's own behaviour and it
        // evidently works; nothing in the source says by what mechanism.
        // So the budget is what stands between this and a hang.
        let (response, len) = self
            .exchange_uni_into(
                &mut *rings,
                UniCommand::ChipConfig,
                &mcu::nic_capability_body(),
                false,
                scratch,
                delay,
                now_ns,
            )
            .map_err(|inner| PhyRefusal::Transport {
                step: PhyStep::Capability,
                inner,
                after: Some(self.firmware_snapshot()),
            })?;
        let capability =
            NicCapability::parse(&scratch[..len], response.eid).map_err(PhyRefusal::Capability)?;
        if capability.mac.is_none() {
            return Err(PhyRefusal::NoMacAddress {
                elements: capability.elements,
            });
        }
        if capability.nss.is_none() || capability.hw_path.is_none() {
            return Err(PhyRefusal::NoPhyCapability {
                elements: capability.elements,
            });
        }

        // 2. Read the efuse block holding `MT_EE_HW_TYPE`.
        //
        // The one answer in this rung that can only come from silicon:
        // the capability reply is the firmware's to compose, and this is
        // a read of the physical fuse behind it.
        let (_, len) = self
            .exchange_uni_into(
                &mut *rings,
                UniCommand::EfuseCtrl,
                &mcu::eeprom_read_body(mcu::MT_EE_HW_TYPE),
                true,
                scratch,
                delay,
                now_ns,
            )
            .map_err(|inner| PhyRefusal::Transport {
                step: PhyStep::EfuseRead,
                inner,
                after: Some(self.firmware_snapshot()),
            })?;
        let block = mcu::eeprom_reply_block(&scratch[..len])
            .ok_or(PhyRefusal::EfuseShort { got: len })?;
        let hw_type = block[mcu::eeprom_block_offset(mcu::MT_EE_HW_TYPE)];

        // 3. "The efuse is the EEPROM, and read it whole."
        //
        // Upstream sends this unconditionally on every init
        // (`mt7925/init.c:112`). Nothing later works from a part whose
        // EEPROM source is undefined, and it is sent last here for the
        // same reason upstream sends it after the capability query: the
        // MAC is already ours by then, so a failure at this step does not
        // cost the rung its evidence.
        self.exchange_uni(
            &mut *rings,
            UniCommand::EfuseCtrl,
            &mcu::eeprom_buffer_mode_body(),
            false,
            delay,
            now_ns,
        )
        .map_err(|inner| PhyRefusal::Transport {
            step: PhyStep::BufferMode,
            inner,
            after: Some(self.firmware_snapshot()),
        })?;

        let _ = &mut delay;
        Ok(PhyReport {
            capability,
            hw_type,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::d1::Conversation;
    use crate::dma::{self, Descriptor};
    use crate::mcu::{MT_EE_HW_TYPE, uni_tlv_header};
    use crate::regs::mt7925::{MT_NIC_CAP_CHIP_CAP, MT_NIC_CAP_MAC_ADDR, MT_NIC_CAP_PHY};
    use crate::mock::{MockMt7925, ReachablePool};
    use kore_driver::seam::{DmaShared, DmaWrite};

    /// The MAC `res/MT7925_EVIDENCE.md` §1 read off this laptop's sysfs
    /// under Linux. **The witness rests on this value**: it is in no blob
    /// this driver ships and in no constant outside this test.
    const LAB_MAC: [u8; 6] = [0xac, 0xf2, 0x3c, 0xaf, 0x0c, 0xe9];

    const SLOT: usize = 2048;

    /// Stage one reply: a 44-byte header carrying `seq`, then `body`.
    fn stage(
        rx_table: &crate::mock::ReachableBuf,
        rx_buffers: &crate::mock::ReachableBuf,
        slot: usize,
        seq: u8,
        body: &[u8],
    ) {
        let at = slot * SLOT;
        let w = DmaShared::new(rx_buffers);
        w.write_u8_at(at + 37, seq).unwrap();
        rx_buffers
            .write_bytes_at(at + crate::mcu::MCU_RXD_BYTES, body)
            .unwrap();
        let len = (crate::mcu::MCU_RXD_BYTES + body.len()) as u32;
        let posted =
            Descriptor::receive(rx_buffers.iova().0 + at as u64, len).expect("reachable");
        DmaShared::new(rx_table)
            .write_u32_at(
                Descriptor::ctrl_at(slot),
                posted.ctrl | crate::regs::mt7925::MT_DMA_CTL_DMA_DONE,
            )
            .unwrap();
    }

    /// A capability reply body: element count, then tagged records.
    fn capa(records: &[(u16, &[u8])]) -> alloc::vec::Vec<u8> {
        let mut b = alloc::vec::Vec::new();
        b.extend_from_slice(&(records.len() as u16).to_le_bytes());
        b.extend_from_slice(&[0, 0]);
        for (tag, data) in records {
            b.extend_from_slice(&uni_tlv_header(*tag, data.len() as u16));
            b.extend_from_slice(data);
        }
        b
    }

    /// An efuse reply body: upstream's `struct evt` up to `data[16]`.
    fn efuse(block: [u8; 16]) -> alloc::vec::Vec<u8> {
        let mut b = alloc::vec![0u8; 31];
        b.extend_from_slice(&block);
        b
    }

    /// **M1 end to end: the part describes itself and its efuse answers.**
    ///
    /// Three exchanges in order — capability, efuse read, buffer mode —
    /// each with the sequence the driver stamps, so a reply that belonged
    /// to another would be stepped over by the triage rather than taken.
    #[test]
    fn the_part_describes_itself_and_its_efuse_answers() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        let pool = ReachablePool::new();
        let tx_table = pool.alloc(dma::DESC_BYTES * 4);
        let tx_payload = pool.alloc(256);
        let rx_table = pool.alloc(dma::DESC_BYTES * 8);
        let rx_buffers = pool.alloc(SLOT * 4);

        let mut phy = [0u8; 13];
        phy[4] = 2; // nss
        phy[10] = 0b11; // hw_path: 2.4 and 5 GHz
        let mut block = [0u8; 16];
        block[1] = 0x06; // MT_EE_HW_TYPE is 0xa71 — offset 1 of block 0xa70
        stage(
            &rx_table,
            &rx_buffers,
            0,
            1,
            &capa(&[
                (MT_NIC_CAP_MAC_ADDR, &LAB_MAC[..]),
                (MT_NIC_CAP_PHY, &phy[..]),
                (MT_NIC_CAP_CHIP_CAP, &0x0001_0030u64.to_le_bytes()[..]),
            ]),
        );
        stage(&rx_table, &rx_buffers, 1, 2, &efuse(block));
        stage(&rx_table, &rx_buffers, 2, 3, &[]);

        let mut conv = Conversation::new();
        let mut rings = Rings {
            tx_table: &tx_table,
            fwdl_table: &tx_table,
            fwdl_payload: &tx_payload,
            fwdl_chunk_bytes: tx_payload.len(),
            tx_payload: &tx_payload,
            rx_table: &rx_table,
            rx_buffers: &rx_buffers,
            rx_buf_bytes: SLOT,
            data: None,
            conv: &mut conv,
        };
        let mut scratch = [0u8; 512];
        let report = core
            .bring_up_phy(&mut rings, &mut scratch, || {}, || None)
            .expect("the staged answers");

        assert_eq!(report.capability.mac, Some(LAB_MAC), "the part's own MAC");
        assert_eq!(report.capability.nss, Some(2));
        assert!(report.capability.both_bands());
        assert_eq!(report.capability.chip_cap, Some(0x0001_0030));
        // `MT_EE_HW_TYPE` is 0xa71, which lives at offset 1 of the block.
        assert_eq!(report.hw_type, 0x06, "the efuse byte, from its offset");
        assert_eq!(
            mcu::eeprom_block_offset(MT_EE_HW_TYPE),
            1,
            "reading offset zero would report the wrong fuse",
        );
        // All three commands went out, so the conversation advanced three
        // sequences — a rung that sent two and reported success would be
        // the unreachable-seam defect again.
        assert_eq!(rings.conv.tx_next, 3, "three commands crossed");
    }

    /// **A reply with no MAC is refused, and says so by name.**
    ///
    /// The walk succeeded and the part chose not to say who it is. That
    /// is a different fault from a reply this driver could not read, and
    /// collapsing them would send a reader to the framing for a firmware
    /// that answered a different question.
    #[test]
    fn a_capability_reply_without_a_mac_refuses_by_name() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        let pool = ReachablePool::new();
        let tx_table = pool.alloc(dma::DESC_BYTES * 4);
        let tx_payload = pool.alloc(256);
        let rx_table = pool.alloc(dma::DESC_BYTES * 8);
        let rx_buffers = pool.alloc(SLOT * 4);

        let mut phy = [0u8; 13];
        phy[4] = 2;
        phy[10] = 0b11;
        stage(&rx_table, &rx_buffers, 0, 1, &capa(&[(MT_NIC_CAP_PHY, &phy[..])]));

        let mut conv = Conversation::new();
        let mut rings = Rings {
            tx_table: &tx_table,
            fwdl_table: &tx_table,
            fwdl_payload: &tx_payload,
            fwdl_chunk_bytes: tx_payload.len(),
            tx_payload: &tx_payload,
            rx_table: &rx_table,
            rx_buffers: &rx_buffers,
            rx_buf_bytes: SLOT,
            data: None,
            conv: &mut conv,
        };
        let mut scratch = [0u8; 512];
        // One record was staged and walked, so the refusal carries 1 —
        // which is what separates "the part answered and omitted the
        // MAC" from "the reply was empty". Both are refusals; only one
        // of them is the firmware's fault.
        assert_eq!(
            core.bring_up_phy(&mut rings, &mut scratch, || {}, || None),
            Err(PhyRefusal::NoMacAddress { elements: 1 }),
        );
    }

    /// A scratch buffer too small refuses with both numbers rather than
    /// truncating — a short body handed to the walk would report
    /// `TrailingBytes`, naming the framing for a sizing mistake.
    /// **A silent part's refusal carries the firmware's own word, read
    /// and not acknowledged.**
    ///
    /// The second M1b photograph proved the command left the host and
    /// nothing came back. What is left to ask is the firmware, through the
    /// two registers [`FirmwareState`] reads — and the alert register must
    /// still hold its flags afterwards, because upstream acknowledges by
    /// writing the value back (`mt792x_dma.c:49-51`), and a driver that did
    /// that here would erase the very thing the photograph is for.
    #[test]
    fn a_silent_part_reports_its_firmware_state_unacknowledged() {
        use crate::regs::mt7925::{MT_CONN_ON_MISC, MT_MCU_CMD};

        let dev = MockMt7925::new();
        // N9 ready (`GENMASK(1, 0)`, `mt792x_regs.h:508`); STOP_DMA (bit 2,
        // `:277`) standing in the alert register.
        dev.poke(dev.cell_of(MT_CONN_ON_MISC), 0x3);
        dev.poke(dev.cell_of(MT_MCU_CMD), 0x4);
        let core = Core::new(&dev);
        let pool = ReachablePool::new();
        let tx_table = pool.alloc(dma::DESC_BYTES * 4);
        let tx_payload = pool.alloc(256);
        let rx_table = pool.alloc(dma::DESC_BYTES * 8);
        let rx_buffers = pool.alloc(SLOT * 4);
        let mut conv = Conversation::new();
        let mut rings = Rings {
            tx_table: &tx_table,
            fwdl_table: &tx_table,
            fwdl_payload: &tx_payload,
            fwdl_chunk_bytes: tx_payload.len(),
            tx_payload: &tx_payload,
            rx_table: &rx_table,
            rx_buffers: &rx_buffers,
            rx_buf_bytes: SLOT,
            data: None,
            conv: &mut conv,
        };
        let mut scratch = [0u8; SLOT];
        let refusal = core
            .bring_up_phy(&mut rings, &mut scratch, || {}, || None)
            .expect_err("nothing was staged, so nothing answers");
        match refusal {
            PhyRefusal::Transport {
                step: PhyStep::Capability,
                inner: D1Refusal::NoAnswer { .. },
                after,
            } => assert_eq!(
                after,
                Some(FirmwareState {
                    misc: 0x3,
                    mcu_cmd: 0x4
                }),
                "the snapshot read the wrong registers",
            ),
            other => panic!("expected a silent capability query, got {other:?}"),
        }
        let cell = dev.cell_of(MT_MCU_CMD);
        assert_eq!(dev.peek(cell), 0x4, "the alert was acknowledged away");
        assert!(
            !dev.writes.borrow().iter().any(|&(addr, _)| addr == cell),
            "the alert register was written",
        );
    }

    #[test]
    fn a_scratch_too_small_refuses_with_both_numbers() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        let pool = ReachablePool::new();
        let tx_table = pool.alloc(dma::DESC_BYTES * 4);
        let tx_payload = pool.alloc(256);
        let rx_table = pool.alloc(dma::DESC_BYTES * 8);
        let rx_buffers = pool.alloc(SLOT * 4);
        stage(
            &rx_table,
            &rx_buffers,
            0,
            1,
            &capa(&[(MT_NIC_CAP_MAC_ADDR, &LAB_MAC[..])]),
        );
        let mut conv = Conversation::new();
        let mut rings = Rings {
            tx_table: &tx_table,
            fwdl_table: &tx_table,
            fwdl_payload: &tx_payload,
            fwdl_chunk_bytes: tx_payload.len(),
            tx_payload: &tx_payload,
            rx_table: &rx_table,
            rx_buffers: &rx_buffers,
            rx_buf_bytes: SLOT,
            data: None,
            conv: &mut conv,
        };
        let mut scratch = [0u8; 4];
        let refusal = core
            .bring_up_phy(&mut rings, &mut scratch, || {}, || None)
            .expect_err("four bytes cannot hold a capability reply");
        assert!(
            matches!(
                refusal,
                PhyRefusal::Transport {
                    step: PhyStep::Capability,
                    inner: D1Refusal::RingTooSmall { have: 4, .. },
                    ..
                }
            ),
            "{refusal:?}",
        );
    }
}
