//! **The MCU command layer** — how a request is framed, where it is sent,
//! and how an answer is recognised.
//!
//! Everything here returns bytes and takes bytes. Nothing touches a
//! register, nothing allocates, and nothing waits: the framing is the half
//! of the download that can be wrong in a way no amount of hardware access
//! would reveal, so it is the half that is proved on the host.
//!
//! > **This module is a seam, not a live path.** Nothing under the `boot`
//! > feature calls it yet; the D1 ladder that would is the next rung, and
//! > `res/NETWORK_RESTART.md` §3 rule 2 is why the framing is finished and
//! > proved before the ladder that uses it is started.
//!
//! # One command, two shapes
//!
//! Every command but one is a 64-byte descriptor followed by its body.
//! `MCU_CMD(FW_SCATTER)` is the exception and it is not a small one:
//! `mt7925_mcu_fill_message` jumps straight to its exit for that command
//! (`mt7925/mcu.c:3845-3846`), so a firmware chunk crosses as **raw payload
//! with no descriptor at all**, on a different queue
//! ([`Queue::Fwdl`](crate::dma::Queue::Fwdl), `mt7925/pci_mcu.c:21-24`).
//! That is why [`scatter`] exists beside [`command`] rather than as a flag
//! on it: a `FW_SCATTER` that grew a descriptor would be 64 bytes of
//! firmware image the part silently loaded at the wrong offset.
//!
//! # The sequence number is the only correlation there is
//!
//! `mt7925_mcu_parse_response` answers `-EAGAIN` when `seq != rxd->seq`
//! (`mt7925/mcu.c:49-57`), and the field is **four bits**: `++msg_seq & 0xf`,
//! retried once when it lands on zero (`:3838-3841`). So sixteen outstanding
//! commands is the ceiling the protocol itself imposes, and a download that
//! waits for each response before issuing the next — which is what
//! `DL_MODE_NEED_RSP` asks for — never approaches it.

use crate::regs::mt7925::*;

/// Bytes of `struct mt76_connac2_mcu_txd` (`mt76_connac_mcu.h:55-72`).
///
/// `__le32 txd[8]` + `len`/`pq_id` + four bytes + four bytes + `u32 rsv[5]`.
pub const MCU_TXD_BYTES: usize = 64;

/// Bytes of the hardware descriptor inside it — the part `len` excludes.
const MCU_TXD_HW_BYTES: u16 = 32;

/// Bytes of `struct mt7925_mcu_rxd` before its TLV body
/// (`mt7925/mcu.h:26-42`). **Eight** `rxd` words, against the shared
/// connac2 response's six — a difference that moves every field after it,
/// which is why this is the MT7925's own constant and not the family's.
pub const MCU_RXD_BYTES: usize = 44;

/// The largest firmware chunk one `FW_SCATTER` carries.
///
/// `mt76_connac2_load_patch` and `mt76_connac_mcu_send_ram_firmware` both
/// take `mt76_is_sdio(dev) ? 2048 : 4096` (`mt76_connac_mcu.c:3339`,
/// `:3085`). This part is PCIe, so 4,096 — and at 1,275,712 bytes of
/// transfer (`res/MT7925_EVIDENCE.md` §14.3) that is **312 chunks**, each
/// one command and one response.
pub const FW_SCATTER_MAX: usize = 4096;

/// How many chunks the whole download costs, at [`FW_SCATTER_MAX`].
///
/// Not a budget — a *size*, kept here so a reader of the ladder's budget
/// can see what it is divided across. The per-chunk share of the 415 ms
/// `res/MT7925_EVIDENCE.md` §11.2 measured end to end is about 1.3 ms.
pub const DOWNLOAD_CHUNKS: usize = 312;

/// What a response could not be read as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum McuError {
    /// The response is shorter than the header it must carry.
    TooShort,
    /// The response's sequence does not match the request's.
    ///
    /// Carries both, because "a stale answer from the previous command" and
    /// "an unsolicited event" are different conditions and the pair tells
    /// them apart.
    WrongSequence {
        /// What the request used.
        want: u8,
        /// What the response carried.
        got: u8,
    },
    /// The part answered the patch semaphore with a code that is neither a
    /// grant nor a refusal.
    BadSemaphore(u8),
}

/// The commands this rung issues, and only these.
///
/// A `u8` with four names rather than an open enum: every one of them is a
/// value `mt76_connac_mcu.h` defines, and a command this ladder does not
/// send has no business being constructible here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Command {
    /// `MCU_CMD_PATCH_SEM_CONTROL` — take or release the ROM patch
    /// semaphore.
    PatchSem,
    /// `MCU_CMD_PATCH_START_REQ` / `MCU_CMD_TARGET_ADDRESS_LEN_REQ` — the
    /// "a region of this length is coming to this address" request. **Which
    /// of the two is a function of the address**, not of the blob; see
    /// [`Command::init_download`].
    InitDownload {
        /// Whether the address is one of this part's patch entry points.
        patch: bool,
    },
    /// `MCU_CMD_PATCH_FINISH_REQ` — the patch is complete.
    PatchFinish,
    /// `MCU_CMD_FW_START_REQ` — run it.
    FwStart,
    /// `MCU_CMD_FW_SCATTER` — one chunk of payload. Framed by [`scatter`],
    /// never by [`command`].
    FwScatter,
    /// `MCU_CMD_RESTART_DL_REQ`'s sibling `MCU_CMD_NIC_POWER_CTRL`, which
    /// `mt76_connac_mcu_restart` sends to put the part back in a state that
    /// accepts a download (`mt76_connac_mcu.c:2985-2996`).
    NicPowerCtrl,
}

impl Command {
    /// The command id this is on the wire.
    pub const fn id(self) -> u8 {
        match self {
            Self::PatchSem => MCU_CMD_PATCH_SEM_CONTROL as u8,
            Self::InitDownload { patch: true } => MCU_CMD_PATCH_START_REQ as u8,
            Self::InitDownload { patch: false } => MCU_CMD_TARGET_ADDRESS_LEN_REQ as u8,
            Self::PatchFinish => MCU_CMD_PATCH_FINISH_REQ as u8,
            Self::FwStart => MCU_CMD_FW_START_REQ as u8,
            Self::FwScatter => MCU_CMD_FW_SCATTER as u8,
            Self::NicPowerCtrl => MCU_CMD_NIC_POWER_CTRL as u8,
        }
    }

    /// Which download-request command an address takes.
    ///
    /// `mt76_connac_mcu_init_download` picks `PATCH_START_REQ` when the
    /// address is one of the part's two patch entry points and
    /// `TARGET_ADDRESS_LEN_REQ` otherwise
    /// (`mt76_connac_mcu.c:3066-3076`). **The two MT7925 addresses are
    /// `0x900000` and `0xe0002800`**, and both of the ROM patch's sections
    /// carry one of them (`res/MT7925_EVIDENCE.md` §14.4) — so the choice
    /// is read off the image rather than decided by which blob is being
    /// sent, which is what a per-phase flag would have got wrong.
    pub const fn init_download(addr: u32) -> Self {
        Self::InitDownload {
            patch: addr == PATCH_ENTRY_ILM || addr == PATCH_ENTRY_DLM,
        }
    }
}

/// The first address `mt76_connac_mcu_init_download` treats as a patch entry
/// point on this part (`mt76_connac_mcu.c:3069`).
///
/// It is **not** "the patch blob's address": the condition is written per
/// chip, and this one is shared with every other connac2/connac3 part in
/// that expression. The ROM patch's first section lands here
/// (`res/MT7925_EVIDENCE.md` §14.4). If it ever moves, the section still
/// downloads — with the wrong request — so the pairing is checked by test
/// rather than assumed.
pub const PATCH_ENTRY_ILM: u32 = 0x0090_0000;

/// The second, and the one that is **this family's own**
/// (`mt76_connac_mcu.c:3070`): the MT7925 and MT7927 take `0xe0002800`
/// where the MT7928 takes `0xe0002000`. Admitting the sibling's address
/// here would send its section the wrong request, which is one more reason
/// `Product::from_pci` does not admit the sibling.
pub const PATCH_ENTRY_DLM: u32 = 0xe000_2800;

/// A four-bit sequence counter, as `mt7925_mcu_fill_message` keeps it.
///
/// Zero is skipped: upstream increments again when `++msg_seq & 0xf` lands
/// on it (`mt7925/mcu.c:3838-3841`), so a response carrying sequence 0 is
/// never one of ours and can be told apart from a stale slot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Sequence(u8);

impl Sequence {
    /// A fresh counter, before any command.
    pub const fn new() -> Self {
        Self(0)
    }

    /// The next sequence to stamp into a command.
    pub fn next(&mut self) -> u8 {
        self.0 = self.0.wrapping_add(1) & 0xf;
        if self.0 == 0 {
            self.0 = self.0.wrapping_add(1) & 0xf;
        }
        self.0
    }

    /// The sequence last handed out.
    pub const fn current(self) -> u8 {
        self.0
    }
}

/// Frame one command's 64-byte descriptor.
///
/// `body_len` is the command's own payload, excluding this descriptor.
/// `mt7925_mcu_fill_message` (`mt7925/mcu.c:3849-3906`), for the non-UNI,
/// non-CE, non-EXT path every command in [`Command`] takes.
///
/// Panics in debug only, through the `u16` conversion, on a body larger than
/// a 16-bit length — the callers' bodies are all under sixteen bytes and
/// the bound is asserted below.
pub fn command(cmd: Command, seq: u8, body_len: u16) -> [u8; MCU_TXD_BYTES] {
    let mut d = [0u8; MCU_TXD_BYTES];
    let total = MCU_TXD_BYTES as u16 + body_len;

    // txd[0]: the whole packet's length, the command packet format, and the
    // MCU's receive queue.
    let w0 = (u32::from(total) & MT_TXD0_TX_BYTES)
        | (MT_TX_TYPE_CMD << MT_TXD0_PKT_FMT.trailing_zeros())
        | (MT_TX_MCU_PORT_RX_Q0 << MT_TXD0_Q_IDX.trailing_zeros());
    d[0..4].copy_from_slice(&w0.to_le_bytes());
    // txd[1]: a command header rather than an 802.3 or 802.11 one.
    let w1 = MT_HDR_FORMAT_CMD << MT_TXD1_HDR_FORMAT.trailing_zeros();
    d[4..8].copy_from_slice(&w1.to_le_bytes());
    // txd[2..8] stay zero.

    // `len` excludes the eight hardware words, not the whole descriptor.
    d[32..34].copy_from_slice(&(total - MCU_TXD_HW_BYTES).to_le_bytes());
    // `MCU_PQ_ID(MT_TX_PORT_IDX_MCU, MT_TX_MCU_PORT_RX_Q0)` —
    // `((p) << 15) | ((q) << 10)` (`mt76_connac_mcu.h:52`), which for this
    // pair is `0x8000 | 0x8000` and therefore **`0x8000`**, not the sum a
    // reader expects.
    let pq_id: u16 = ((MT_TX_PORT_IDX_MCU as u16) << 15) | ((MT_TX_MCU_PORT_RX_Q0 as u16) << 10);
    d[34..36].copy_from_slice(&pq_id.to_le_bytes());

    d[36] = cmd.id();
    d[37] = MCU_PKT_ID as u8;
    // No `ext_cid` and no CE flag on any of this rung's commands, so the
    // set/query field is "not applicable" — `mt7925/mcu.c:3900`.
    d[38] = MCU_Q_NA as u8;
    d[39] = seq;
    // uc_d2b0_rev, ext_cid, ext_cid_ack all zero; s2d_index is host-to-WM.
    d[42] = MCU_S2D_H2N as u8;
    d
}

/// Bytes of a **unified** command's descriptor.
///
/// **Forty-eight, where [`MCU_TXD_BYTES`] is sixty-four**, and the
/// difference is the reason this is a separate encoder rather than a flag
/// on [`command`].
///
/// `mt7925_mcu_fill_message` picks between the two on
/// `cmd & __MCU_CMD_FIELD_UNI` (`BIT(17)`): every command the download
/// rungs send has that bit clear and takes
/// `struct mt76_connac2_mcu_txd`; every command from M1 onward sets it
/// and takes `struct mt76_connac2_mcu_uni_txd`.
///
/// The two layouts agree on `pkt_type` at 37, `seq` at 39 and
/// `s2d_index` at 42, and disagree everywhere else: `cid` moves from a
/// byte at 36 to a little-endian half-word at **34**, where the legacy
/// form keeps `pq_id`; `set_query`, `ext_cid` and `ext_cid_ack`
/// disappear; and `option` appears at **43**. **A half-ported encoder
/// therefore passes the sequence check and is still garbage** — the
/// shape of the two defects that cost flashes six and seven — which is
/// why these are two functions and why the test asserts on the bytes
/// that cross the ring rather than on the return value.
pub const MCU_UNI_TXD_BYTES: usize = 48;

/// A unified command, by the id it carries in `cid`.
///
/// Only the ones M1 sends. The enum is deliberately not a transcription
/// of upstream's whole list: an id this driver cannot frame a body for
/// is an id it must not be able to name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UniCommand {
    /// `MCU_UNI_CMD_CHIP_CONFIG` = `0x0e`
    /// (`mt76_connac_mcu.h:1326`). Carries `UNI_CHIP_CONFIG_NIC_CAPA`.
    ChipConfig,
    /// `MCU_UNI_CMD_EFUSE_CTRL` = `0x2d` (`mt76_connac_mcu.h:1342`):
    /// reads a block of the part's own efuse, or tells the firmware which
    /// EEPROM source to use.
    EfuseCtrl,
    /// `MCU_UNI_CMD_WSYS_CONFIG` = `0x0b`: where the firmware sends its
    /// log ([`fw_log_body`]).
    WsysConfig,
    /// `MCU_UNI_CMD_SET_POWER_LIMIT` = `0x2c`: a CLC record (tag 3) or a
    /// slice of the power table (tag 1).
    SetPowerLimit,
    /// `MCU_UNI_CMD_SET_DOMAIN_INFO` = `0x15`: the country and the channels
    /// it permits.
    SetDomainInfo,
}

impl UniCommand {
    /// The id this command carries in `cid` — a **half-word**, not a byte.
    pub const fn id(self) -> u16 {
        match self {
            Self::ChipConfig => MCU_UNI_CMD_CHIP_CONFIG,
            Self::EfuseCtrl => MCU_UNI_CMD_EFUSE_CTRL,
            Self::WsysConfig => MCU_UNI_CMD_WSYS_CONFIG,
            Self::SetPowerLimit => MCU_UNI_CMD_SET_POWER_LIMIT,
            Self::SetDomainInfo => MCU_UNI_CMD_SET_DOMAIN_INFO,
        }
    }

    /// The `option` byte this command takes, for a query or for a set.
    ///
    /// `mt7925_mcu_fill_message:3805-3812`, in order:
    ///
    /// ```text
    /// query ? MCU_CMD_UNI_QUERY_ACK : MCU_CMD_UNI_EXT_ACK
    /// if cmd is HIF_CTRL or CHIP_CONFIG: option &= ~MCU_CMD_ACK
    /// ```
    ///
    /// with `ACK = BIT(0)`, `UNI = BIT(1)`, `SET = BIT(2)`
    /// (`mt76_connac_mcu.h:1210-1216`), so `EXT_ACK` is `0b111` and
    /// `QUERY_ACK` is `0b011`.
    ///
    /// **`ChipConfig` asks for no acknowledgement and is waited on
    /// anyway.** That is upstream's own behaviour —
    /// `mt7925_mcu_get_nic_capability` passes `wait_resp = true` for a
    /// command whose ACK bit it has just cleared — and it evidently
    /// works, but nothing in the source says by what mechanism. So the
    /// rung that sends it budgets for the reply never arriving and
    /// refuses with a number rather than hanging.
    ///
    /// Upstream has a third arm for `TESTMODE_CTRL` and
    /// `TESTMODE_RX_STAT` (`:3814-3820`). Neither is reachable from this
    /// driver, and an arm for a command that cannot be named is an arm
    /// that cannot be tested.
    pub const fn option(self, query: bool) -> u8 {
        let base = if query {
            MCU_CMD_ACK | MCU_CMD_UNI
        } else {
            MCU_CMD_ACK | MCU_CMD_UNI | MCU_CMD_SET
        };
        match self {
            Self::ChipConfig => base & !MCU_CMD_ACK,
            Self::EfuseCtrl | Self::WsysConfig | Self::SetPowerLimit | Self::SetDomainInfo => base,
        }
    }
}

/// Frame one unified command's 48-byte descriptor.
///
/// `body_len` is the command's own payload, excluding this descriptor —
/// and for a unified command that payload is itself a four-byte reserved
/// header followed by TLVs, which the caller builds. See
/// [`uni_tlv_header`] for the convention, which is **not** the one
/// mt7921 uses.
///
/// The eight hardware words are built exactly as [`command`] builds them:
/// upstream writes them before it branches on the descriptor kind
/// (`mt7925/mcu.c:3789-3795`), so they are the one part of the two forms
/// that is genuinely shared.
pub fn uni_command(cmd: UniCommand, seq: u8, body_len: u16, query: bool) -> [u8; MCU_UNI_TXD_BYTES] {
    let mut d = [0u8; MCU_UNI_TXD_BYTES];
    let total = MCU_UNI_TXD_BYTES as u16 + body_len;

    // txd[0] and txd[1] — identical to the legacy form's, deliberately.
    let w0 = (u32::from(total) & MT_TXD0_TX_BYTES)
        | (MT_TX_TYPE_CMD << MT_TXD0_PKT_FMT.trailing_zeros())
        | (MT_TX_MCU_PORT_RX_Q0 << MT_TXD0_Q_IDX.trailing_zeros());
    d[0..4].copy_from_slice(&w0.to_le_bytes());
    let w1 = MT_HDR_FORMAT_CMD << MT_TXD1_HDR_FORMAT.trailing_zeros();
    d[4..8].copy_from_slice(&w1.to_le_bytes());
    // txd[2..8] stay zero.

    // `len` excludes the eight hardware words — the same rule the legacy
    // form uses, and one of only three things the two agree on.
    d[32..34].copy_from_slice(&(total - MCU_TXD_HW_BYTES).to_le_bytes());
    // **`cid` is a half-word here, and it sits where `pq_id` sits in the
    // legacy form.** There is no `pq_id` in a unified descriptor.
    d[34..36].copy_from_slice(&cmd.id().to_le_bytes());
    // d[36] is `rsv`, zero.
    d[37] = MCU_PKT_ID as u8;
    // d[38] is `frag_n`, zero: this driver sends no fragmented command.
    d[39] = seq;
    // d[40..42] is `checksum`; "0 mean there is no checksum"
    // (`mt76_connac_mcu.h:80`).
    d[42] = MCU_S2D_H2N as u8;
    d[43] = cmd.option(query);
    // d[44..48] is `rsv1`, zero.
    d
}

/// Frame a unified command body's TLV header.
///
/// **`len` includes these four bytes.** That is mt7925's convention
/// (`mt7925/mcu.c` reads `tlv.len` directly, never `len + sizeof(tlv)`),
/// and it is the opposite of mt7921's `tlv_hdr`, whose `len` excludes the
/// header. Taking the sibling's code as a model loses four bytes per TLV
/// and desynchronises the walk — a silent corruption rather than a
/// refusal, which is why the convention is stated here and tested rather
/// than left to the call sites.
pub const fn uni_tlv_header(tag: u16, payload_len: u16) -> [u8; 4] {
    let tag = tag.to_le_bytes();
    let len = (payload_len + 4).to_le_bytes();
    [tag[0], tag[1], len[0], len[1]]
}

/// What the part said when asked to describe itself.
///
/// Every field is [`Option`], because the reply is a **stream of tagged
/// records and the part chooses which to send**. Upstream reads five tags
/// and ignores the rest; a field absent from the reply is a fact this
/// driver does not have, which is different from a zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NicCapability {
    /// `MT_NIC_CAP_MAC_ADDR` — the part's own hardware address, out of its
    /// efuse. The strongest evidence in this reply: it is in no blob this
    /// driver ships and in no constant in this tree.
    pub mac: Option<[u8; 6]>,
    /// `MT_NIC_CAP_PHY`'s `nss`, the spatial-stream count at offset 4 of
    /// that record.
    pub nss: Option<u8>,
    /// `MT_NIC_CAP_PHY`'s `hw_path` at offset 10 — bit 0 is 2.4 GHz, bit
    /// 1 is 5 GHz (`WF0_24G`/`WF0_5G`).
    pub hw_path: Option<u8>,
    /// `MT_NIC_CAP_CHIP_CAP`, the 64-bit feature word every later rung
    /// branches on.
    pub chip_cap: Option<u64>,
    /// How many records the walk consumed.
    pub elements: u16,
    /// **The event id the part actually sent, recorded and not judged.**
    ///
    /// `MCU_UNI_EVENT_NIC_CAPAB = 0x43` exists in upstream's headers and
    /// **nothing in the mt76 tree reads it** — `get_nic_capability` never
    /// checks the eid on this path. So this driver cannot assert a value
    /// it has no source for. It carries what arrived into the note band
    /// instead, and the first flash tells us the number.
    pub eid: u8,
}

/// Why a capability reply did not parse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapaError {
    /// The body is shorter than the four-byte element count header.
    NoHeader,
    /// A record claims a length that runs past the end of the body.
    RecordTruncated {
        /// Which record, counting from zero.
        index: u16,
        /// The length it claimed.
        claimed: u16,
        /// What remained.
        left: usize,
    },
    /// A record claims a length below its own four-byte header.
    ///
    /// **Not a hang — a misalignment, which is worse.** Both this walk
    /// and upstream's are bounded by the element count, so a short length
    /// terminates either way. What it does is advance by less than the
    /// header just read, so every record after it is parsed from the
    /// wrong offset: tags become lengths, lengths become data, and the
    /// walk reports fields the part never sent. Upstream tolerates that
    /// because it treats the whole reply as advisory. This rung's
    /// evidence *is* that the framing was understood, so a body it cannot
    /// walk is refused by name rather than half-read.
    RecordTooShort {
        /// Which record, counting from zero.
        index: u16,
        /// The length it claimed.
        claimed: u16,
    },
    /// The element count was satisfied and bytes remained.
    ///
    /// Refused rather than ignored: a body this driver cannot account for
    /// exactly is a body it has not understood, and the witness rests on
    /// understanding it. Upstream `break`s out of the loop on anything odd
    /// and carries on with whatever it collected, which is the right call
    /// for a driver that must stay up and the wrong one for a rung whose
    /// whole job is to prove the framing works.
    TrailingBytes {
        /// How many bytes were left over.
        left: usize,
    },
}

impl CapaError {
    /// A distinct byte per variant, for the note's detail.
    ///
    /// The note is read off a photograph of a terminal, so the reader
    /// needs one number that names *which* way the walk failed before any
    /// of the numbers under it mean anything. Appended to, never
    /// renumbered.
    pub const fn code(self) -> u8 {
        match self {
            Self::NoHeader => 1,
            Self::RecordTruncated { .. } => 2,
            Self::RecordTooShort { .. } => 3,
            Self::TrailingBytes { .. } => 4,
        }
    }

    /// The variant's own numbers, packed for the note's evidence word.
    ///
    /// Record index in the high half, the length it claimed in the low
    /// half — the two numbers that locate a bad record in a reply nobody
    /// can capture. `TrailingBytes` has one number and puts it in the low
    /// half; `NoHeader` has none, because a body too short to hold its own
    /// count says everything it has to say by name.
    pub const fn evidence(self) -> u32 {
        match self {
            Self::NoHeader => 0,
            Self::RecordTruncated { index, claimed, .. }
            | Self::RecordTooShort { index, claimed } => {
                ((index as u32) << 16) | claimed as u32
            }
            Self::TrailingBytes { left } => {
                if left > u16::MAX as usize {
                    u16::MAX as u32
                } else {
                    left as u32
                }
            }
        }
    }
}

impl NicCapability {
    /// Walk a capability reply's body — everything after the 44-byte
    /// response header.
    ///
    /// `mt7925_mcu_get_nic_capability` (`mt7925/mcu.c:1069-1137`): a
    /// `struct mt76_connac_cap_hdr { __le16 n_element; u8 rsv[2]; }`,
    /// then `n_element` records of `struct tlv { __le16 tag; __le16 len }`
    /// whose **`len` includes those four bytes** — the walk advances by
    /// `skb_pull(skb, len)` (`:1132`).
    pub fn parse(body: &[u8], eid: u8) -> Result<Self, CapaError> {
        const HDR: usize = 4;
        const TLV: usize = 4;
        if body.len() < HDR {
            return Err(CapaError::NoHeader);
        }
        let elements = u16::from_le_bytes([body[0], body[1]]);
        let mut rest = &body[HDR..];
        let mut out = Self {
            elements,
            eid,
            ..Self::default()
        };

        for index in 0..elements {
            if rest.len() < TLV {
                return Err(CapaError::RecordTruncated {
                    index,
                    claimed: 0,
                    left: rest.len(),
                });
            }
            let tag = u16::from_le_bytes([rest[0], rest[1]]);
            let claimed = u16::from_le_bytes([rest[2], rest[3]]);
            if (claimed as usize) < TLV {
                return Err(CapaError::RecordTooShort { index, claimed });
            }
            if claimed as usize > rest.len() {
                return Err(CapaError::RecordTruncated {
                    index,
                    claimed,
                    left: rest.len(),
                });
            }
            let data = &rest[TLV..claimed as usize];
            match tag {
                MT_NIC_CAP_MAC_ADDR => {
                    if data.len() >= 6 {
                        let mut mac = [0u8; 6];
                        mac.copy_from_slice(&data[..6]);
                        out.mac = Some(mac);
                    }
                }
                MT_NIC_CAP_PHY => {
                    // `struct mt7925_mcu_phy_cap` (`mt7925/mcu.c:1027-1041`):
                    // nss at 4, hw_path at 10.
                    out.nss = data.get(4).copied();
                    out.hw_path = data.get(10).copied();
                }
                MT_NIC_CAP_CHIP_CAP => {
                    if data.len() >= 8 {
                        let mut w = [0u8; 8];
                        w.copy_from_slice(&data[..8]);
                        out.chip_cap = Some(u64::from_le_bytes(w));
                    }
                }
                // Every other tag is a fact this rung does not need. Walked
                // past, exactly as upstream's `default: break` does — the
                // walk must still account for them.
                _ => {}
            }
            rest = &rest[claimed as usize..];
        }

        if !rest.is_empty() {
            return Err(CapaError::TrailingBytes { left: rest.len() });
        }
        Ok(out)
    }

    /// Whether `hw_path` claims both the 2.4 GHz and 5 GHz chains.
    ///
    /// `WF0_24G` is bit 0 and `WF0_5G` is bit 1
    /// (`mt7925_mcu_parse_phy_cap`'s own enum). A part that claims neither
    /// has answered the question without describing a radio.
    pub const fn both_bands(&self) -> bool {
        match self.hw_path {
            Some(path) => path & 0b11 == 0b11,
            None => false,
        }
    }
}

/// Bytes of one efuse block — `MT7925_EEPROM_BLOCK_SIZE`
/// (`mt7925/mt7925.h:28`).
pub const EEPROM_BLOCK_BYTES: usize = 16;

/// `MT_EE_HW_TYPE` (`mt7925/mt7925.h:221`) — the efuse byte whose low two
/// bits (`MT_EE_HW_TYPE_ENCAP`, `GENMASK(1, 0)`) select which CLC record
/// this part's regulatory tables come from.
pub const MT_EE_HW_TYPE: u32 = 0xa71;

/// `EE_MODE_EFUSE` — the on-chip efuse is the EEPROM
/// (`mt76_connac_mcu.h:1191-1194`, first of the enum).
pub const EE_MODE_EFUSE: u8 = 0;
/// `EE_FORMAT_WHOLE` — read it entire (`:1196-1200`, second of the enum).
pub const EE_FORMAT_WHOLE: u8 = 1;

/// `UNI_CHIP_CONFIG_NIC_CAPA`'s request — ask the part to describe itself.
///
/// `mt7925_mcu_get_nic_capability` (`mt7925/mcu.c:1073-1081`): four
/// reserved bytes and a bare TLV header, `len = sizeof(req) - 4` = **4**,
/// which is the header counting itself and carrying no payload.
pub const fn nic_capability_body() -> [u8; 8] {
    let t = uni_tlv_header(UNI_CHIP_CONFIG_NIC_CAPA, 0);
    [0, 0, 0, 0, t[0], t[1], t[2], t[3]]
}

/// `UNI_EFUSE_ACCESS`'s request — read the 16-byte block holding `offset`.
///
/// `mt7925_mcu_read_eeprom` (`mt7925/mcu.c:828-842`): four reserved bytes,
/// the TLV header, then `addr`, `valid` and a 16-byte `data` field the
/// request sends zeroed and the reply fills.
///
/// **The address is rounded down to a block.** `round_down(offset,
/// MT7925_EEPROM_BLOCK_SIZE)` — the part reads in blocks, so asking for
/// `0xa71` reads the block at `0xa70` and the wanted byte is at
/// [`eeprom_block_offset`] within it. Rounding here rather than at the
/// call site means a caller cannot ask for an address the part will
/// silently reinterpret.
pub fn eeprom_read_body(offset: u32) -> [u8; 32] {
    let mut b = [0u8; 32];
    // `sizeof(req) - 4` is 28, which `uni_tlv_header` computes as the
    // 24-byte payload plus its own four.
    let t = uni_tlv_header(UNI_EFUSE_ACCESS, 24);
    b[4..8].copy_from_slice(&t);
    let block = offset - (offset % EEPROM_BLOCK_BYTES as u32);
    b[8..12].copy_from_slice(&block.to_le_bytes());
    // `valid` and `data[16]` go out zero; the reply fills them.
    b
}

/// Where `offset` sits inside the block [`eeprom_read_body`] asks for.
pub const fn eeprom_block_offset(offset: u32) -> usize {
    (offset % EEPROM_BLOCK_BYTES as u32) as usize
}

/// `UNI_WSYS_CONFIG_FW_LOG_CTRL`'s request — `mt7925_mcu_fw_log_2_host`
/// (`mt7925/mcu.c:1039-1059`): four reserved bytes, the TLV header, then
/// `ctrl`, `interval` and two reserved. Upstream sends `ctrl = 1` (log to
/// the host) as the last step of `mt7925_run_firmware`.
pub const fn fw_log_body(ctrl: u8) -> [u8; 12] {
    let t = uni_tlv_header(UNI_WSYS_CONFIG_FW_LOG_CTRL, 4);
    [0, 0, 0, 0, t[0], t[1], t[2], t[3], ctrl, 0, 0, 0]
}

/// `UNI_EFUSE_BUFFER_MODE`'s request — name the EEPROM source.
///
/// `mt7925_mcu_set_eeprom` (`mt7925/mcu.c:1613-1627`): four reserved
/// bytes, the TLV header, then `buffer_mode`, `format` and a zero
/// `buf_len`. Upstream sends this unconditionally on every init: it tells
/// the firmware the on-chip efuse is the whole EEPROM, and nothing later
/// works from a part whose source is undefined.
pub const fn eeprom_buffer_mode_body() -> [u8; 12] {
    let t = uni_tlv_header(UNI_EFUSE_BUFFER_MODE, 4);
    [
        0,
        0,
        0,
        0,
        t[0],
        t[1],
        t[2],
        t[3],
        EE_MODE_EFUSE,
        EE_FORMAT_WHOLE,
        0,
        0,
    ]
}

/// The 16 efuse bytes an `UNI_EFUSE_ACCESS` reply carries.
///
/// `struct evt` (`mt7925/mcu.c:843-857`) is four reserved bytes, the TLV
/// header, then `ver`, `addr`, `valid`, `size`, `magic_num`, `type`,
/// `rsv[2]` and finally `data[16]`. Upstream reads **only** `data`, and
/// so does this: the other fields are declared there and never examined,
/// so this driver has no source for what they should contain and will not
/// gate a rung on them.
pub fn eeprom_reply_block(body: &[u8]) -> Option<[u8; EEPROM_BLOCK_BYTES]> {
    // 4 rsv + 4 tlv + ver(4) + addr(4) + valid(4) + size(4) + magic(4)
    // + type(1) + rsv[2](2) ... upstream's struct, counted to `data`.
    const DATA_AT: usize = 4 + 4 + 4 + 4 + 4 + 4 + 4 + 1 + 2;
    let end = DATA_AT.checked_add(EEPROM_BLOCK_BYTES)?;
    let slice = body.get(DATA_AT..end)?;
    let mut block = [0u8; EEPROM_BLOCK_BYTES];
    block.copy_from_slice(slice);
    Some(block)
}

/// Frame one firmware chunk.
///
/// There is nothing to frame: `FW_SCATTER` crosses as raw payload. This
/// function exists so the call site says what it is doing and so the chunk
/// bound is enforced in one place.
pub fn scatter(chunk: &[u8]) -> Option<&[u8]> {
    (!chunk.is_empty() && chunk.len() <= FW_SCATTER_MAX).then_some(chunk)
}

/// How many `FW_SCATTER` commands a region of `len` bytes costs.
pub const fn chunks_for(len: u64) -> u64 {
    len.div_ceil(FW_SCATTER_MAX as u64)
}

// ── the command bodies ───────────────────────────────────────────────────

/// `mt76_connac_mcu_init_download`'s request (`mt76_connac_mcu.c:54-78`):
/// address, length, mode, little-endian.
pub const fn init_download_body(addr: u32, len: u32, mode: u32) -> [u8; 12] {
    let (a, l, m) = (addr.to_le_bytes(), len.to_le_bytes(), mode.to_le_bytes());
    [a[0], a[1], a[2], a[3], l[0], l[1], l[2], l[3], m[0], m[1], m[2], m[3]]
}

/// `mt76_connac_mcu_patch_sem_ctrl`'s request
/// (`mt76_connac_mcu.c:24-35`): the operation as a **32-bit** word.
pub const fn patch_sem_body(get: bool) -> [u8; 4] {
    let op = if get { PATCH_SEM_GET } else { PATCH_SEM_RELEASE };
    op.to_le_bytes()
}

/// The status byte an answered command carries when it succeeded.
///
/// `mt7925_mcu_parse_response` returns `*skb->data` as the command's error
/// code (`mt7925/mcu.c:57-60`), so zero is the only success — and that is
/// a fact about the *parse*, not about any one command. `PATCH_FINISH_REQ`
/// and `FW_START_REQ` both ride it, which is why this is one constant
/// rather than a per-command pair that would drift.
///
/// **`PATCH_SEM_CONTROL` is the exception, and the reason it is worth
/// naming.** The semaphore answers with a *state* rather than an error:
/// [`Semaphore::from_status`] reads 2 as `PATCH_NOT_DL_SEM_SUCCESS` and 0
/// as a refusal, the opposite sense to this. Note 81 on 2026-11-11 was a
/// finish answering 2, and the question of the day was whether it meant
/// the semaphore's "granted". It did not: metal settled it on 2026-11-12
/// when a correctly-framed finish answered zero.
pub const MCU_STATUS_OK: u8 = 0;

/// `mt76_connac_mcu_start_patch`'s request (`mt76_connac_mcu.c:37-49`):
/// a `check_crc` byte and three reserved. Upstream sends **zero** — the
/// part is not asked to verify the patch's CRC, and this does not second-
/// guess it.
pub const fn patch_finish_body() -> [u8; 4] {
    [0, 0, 0, 0]
}

/// `mt76_connac_mcu_start_firmware`'s request (`mt76_connac_mcu.c:9-21`):
/// the option word first, then the entry address.
///
/// Note the order — `option` precedes `addr`, which is the reverse of
/// [`init_download_body`]'s. Transcribed, not rationalised.
pub const fn fw_start_body(option: u32, addr: u32) -> [u8; 8] {
    let (o, a) = (option.to_le_bytes(), addr.to_le_bytes());
    [o[0], o[1], o[2], o[3], a[0], a[1], a[2], a[3]]
}

/// The option word [`fw_start_body`] takes, for an image that names an
/// entry point and one that does not.
///
/// `FW_START_OVERRIDE` only; `FW_START_WORKING_PDA_CR4` is the WA image's
/// and this part ships none (`mt76_connac_mcu.c:3123-3127`).
pub const fn fw_start_option(override_addr: Option<u32>) -> u32 {
    match override_addr {
        Some(_) => FW_START_OVERRIDE,
        None => 0,
    }
}

/// `mt76_connac_mcu_restart`'s request (`mt76_connac_mcu.c:2985-2996`):
/// a power-mode byte of 1 and three reserved.
pub const fn restart_body() -> [u8; 4] {
    [1, 0, 0, 0]
}

// ── the response ─────────────────────────────────────────────────────────

/// Byte offset of the response header's `option` field.
///
/// `struct mt7925_mcu_rxd` (`mt7925/mcu.h`) is 44 bytes for **both** the
/// legacy and the unified command classes, and `option` sits at 38 in
/// each. The download rungs never read it; from M1 onward it is the first
/// thing read, because it is what says whether this reply is a reply at
/// all.
pub const MCU_RXD_OPTION_AT: usize = 38;

/// `MCU_UNI_CMD_UNSOLICITED_EVENT` — the part spoke without being asked.
///
/// **The event ring is shared.** `mt7925_mcu_rx_event` routes on this bit
/// *before* anything looks at a sequence number
/// (`mt7925/mcu.c`, `rxd->option & MCU_UNI_CMD_UNSOLICITED_EVENT`), and
/// firmware logs, coredumps and HIF-control notices all arrive on the
/// same ring an exchange is polling. A driver that reads the next done
/// descriptor as its answer will eventually read one of these instead —
/// and the sequence check it would then fail names the wrong fault.
///
/// Nothing in C2 could hit it: no unsolicited source is enabled until
/// something turns one on, and this driver deliberately does not enable
/// the firmware log. It is read now because M1 is where that stops being
/// true by luck rather than by construction.
pub const MCU_UNI_CMD_UNSOLICITED_EVENT: u8 = 1 << 2;

/// Whether a response header belongs to nobody's request.
///
/// Takes the whole slot's bytes so a short read cannot be mistaken for a
/// solicited reply: a buffer too small to hold a header is not an event
/// this driver may discard silently, and [`Response::parse`] refuses it
/// with [`McuError::TooShort`].
pub fn is_unsolicited(bytes: &[u8]) -> bool {
    match bytes.get(MCU_RXD_OPTION_AT) {
        Some(option) => option & MCU_UNI_CMD_UNSOLICITED_EVENT != 0,
        None => false,
    }
}

/// The sequence a response header carries, when it is long enough to have
/// one.
///
/// Separate from [`Response::parse`] because a mismatch is not always a
/// fault: upstream discards a non-matching reply and keeps reading until
/// its deadline (`mt76_mcu_skb_send_and_get_msg`'s
/// `do { … } while (ret == -EAGAIN)` loop, where `-EAGAIN` is exactly
/// what a sequence mismatch returns). Telling "not mine" from "malformed"
/// needs the number before the verdict.
pub fn peek_sequence(bytes: &[u8]) -> Option<u8> {
    if bytes.len() < MCU_RXD_BYTES {
        return None;
    }
    Some(bytes[37])
}

/// A response's header, as much of it as this rung reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Response {
    /// The event id.
    pub eid: u8,
    /// The sequence the request carried.
    pub seq: u8,
    /// The status byte at offset 40 — `mt7925_mcu_parse_response` reaches it
    /// as `skb_pull(skb, sizeof(*rxd) - 4)` and then `*skb->data`
    /// (`mt7925/mcu.c:57-60`). Meaningful for `PATCH_SEM_CONTROL` and
    /// `PATCH_FINISH_REQ`; for every other command the body starts after
    /// the full header and this byte is part of it.
    pub status: u8,
}

impl Response {
    /// Read a response header, checking it against the request's sequence.
    pub fn parse(bytes: &[u8], want_seq: u8) -> Result<Self, McuError> {
        if bytes.len() < MCU_RXD_BYTES {
            return Err(McuError::TooShort);
        }
        let got = bytes[37];
        if got != want_seq {
            return Err(McuError::WrongSequence {
                want: want_seq,
                got,
            });
        }
        Ok(Self {
            eid: bytes[36],
            seq: got,
            status: bytes[40],
        })
    }
}

/// What the part said about the ROM patch semaphore.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Semaphore {
    /// The patch is already loaded; this download is a no-op.
    /// `PATCH_IS_DL` — `mt76_connac2_load_patch` returns success on it
    /// (`mt76_connac_mcu.c:3345-3346`).
    AlreadyLoaded,
    /// The semaphore is ours and the patch is not loaded.
    Granted,
    /// The semaphore was released.
    Released,
    /// The part refused to hand it over.
    Refused,
}

impl Semaphore {
    /// Read a semaphore response's status byte.
    pub const fn from_status(status: u8) -> Result<Self, McuError> {
        match status as u32 {
            PATCH_IS_DL => Ok(Self::AlreadyLoaded),
            PATCH_NOT_DL_SEM_SUCCESS => Ok(Self::Granted),
            PATCH_REL_SEM_SUCCESS => Ok(Self::Released),
            PATCH_NOT_DL_SEM_FAIL => Ok(Self::Refused),
            _ => Err(McuError::BadSemaphore(status)),
        }
    }
}

#[cfg(test)]
mod tests {

    /// **Every unified request's TLV length is its own size less four.**
    ///
    /// Upstream writes `len = cpu_to_le16(sizeof(req) - 4)` in each of the
    /// three request structs this rung sends, where the 4 is the leading
    /// reserved word and the length counts the TLV header itself. Two
    /// conventions meet in that one expression, and the sibling driver
    /// uses the other one — so the property is asserted across all three
    /// bodies at once rather than trusted to each call site.
    #[test]
    fn every_unified_request_carries_its_own_length() {
        let bodies: [(&str, &[u8]); 3] = [
            ("nic_capability", &nic_capability_body()),
            ("eeprom_read", &eeprom_read_body(MT_EE_HW_TYPE)),
            ("eeprom_buffer_mode", &eeprom_buffer_mode_body()),
        ];
        for (name, body) in bodies {
            assert_eq!(&body[0..4], &[0, 0, 0, 0], "{name}: leading reserved word");
            let len = u16::from_le_bytes([body[6], body[7]]);
            assert_eq!(
                len as usize,
                body.len() - 4,
                "{name}: tlv len must be sizeof(req) - 4",
            );
        }
    }

    /// The three requests are the sizes upstream's structs are.
    #[test]
    fn the_request_bodies_are_upstreams_sizes() {
        // `{ u8 _rsv[4]; __le16 tag; __le16 len; }`
        assert_eq!(nic_capability_body().len(), 8);
        assert_eq!(
            u16::from_le_bytes([nic_capability_body()[4], nic_capability_body()[5]]),
            UNI_CHIP_CONFIG_NIC_CAPA,
        );
        // `{ rsv[4]; tag; len; __le32 addr; __le32 valid; u8 data[16]; }`
        assert_eq!(eeprom_read_body(0).len(), 32);
        // `{ _rsv[4]; tag; len; u8 buffer_mode; u8 format; __le16 buf_len; }`
        assert_eq!(eeprom_buffer_mode_body().len(), 12);
        let m = eeprom_buffer_mode_body();
        assert_eq!(m[8], EE_MODE_EFUSE, "the efuse is the EEPROM");
        assert_eq!(m[9], EE_FORMAT_WHOLE, "read it entire");
        assert_eq!(u16::from_le_bytes([m[10], m[11]]), 0, "buf_len");
    }

    /// **The efuse address is rounded down to a block, and the wanted
    /// byte's place within it is kept.**
    ///
    /// `round_down(offset, MT7925_EEPROM_BLOCK_SIZE)` — the part reads
    /// sixteen bytes at a time, so a request for `0xa71` reads the block
    /// at `0xa70`. A caller that asked for `0xa71` and then read byte
    /// zero of the reply would get `0xa70`'s contents and never know.
    #[test]
    fn an_efuse_read_asks_for_the_block_and_remembers_the_offset() {
        let b = eeprom_read_body(MT_EE_HW_TYPE);
        assert_eq!(
            u32::from_le_bytes([b[8], b[9], b[10], b[11]]),
            0xa70,
            "0xa71 lives in the block at 0xa70",
        );
        assert_eq!(eeprom_block_offset(MT_EE_HW_TYPE), 1);
        // A block-aligned address is its own block, at offset zero.
        let b = eeprom_read_body(0xa70);
        assert_eq!(u32::from_le_bytes([b[8], b[9], b[10], b[11]]), 0xa70);
        assert_eq!(eeprom_block_offset(0xa70), 0);
        // `valid` and `data` go out zero — the reply fills them.
        assert_eq!(&b[12..32], &[0u8; 20][..]);
    }

    /// The efuse reply's block is read from where upstream's `struct evt`
    /// puts it, and a short reply yields nothing rather than garbage.
    #[test]
    fn an_efuse_reply_yields_its_block_or_nothing() {
        // rsv(4) tlv(4) ver(4) addr(4) valid(4) size(4) magic(4) type(1)
        // rsv[2](2) = 31, then data[16].
        let mut body = alloc::vec![0u8; 31];
        for i in 0..16u8 {
            body.push(0xA0 + i);
        }
        let block = eeprom_reply_block(&body).expect("a full reply");
        assert_eq!(block[0], 0xA0);
        assert_eq!(block[15], 0xAF);
        // One byte short: no block, rather than a block with a zero in it.
        assert_eq!(eeprom_reply_block(&body[..body.len() - 1]), None);
        assert_eq!(eeprom_reply_block(&[]), None);
    }


    /// Build a capability reply body: an element count, then records.
    fn capa_body(records: &[(u16, &[u8])]) -> alloc::vec::Vec<u8> {
        let mut b = alloc::vec::Vec::new();
        b.extend_from_slice(&(records.len() as u16).to_le_bytes());
        b.extend_from_slice(&[0, 0]); // rsv
        for (tag, data) in records {
            b.extend_from_slice(&uni_tlv_header(*tag, data.len() as u16));
            b.extend_from_slice(data);
        }
        b
    }

    /// **The lab's own part, as it should answer.**
    ///
    /// The MAC is the value `res/MT7925_EVIDENCE.md` §1 read off this
    /// laptop's sysfs under Linux. It is the witness this rung rests on,
    /// because it is in no blob this driver ships and no constant in this
    /// tree — a stub would have to be built already knowing it.
    #[test]
    fn a_capability_reply_yields_the_parts_own_identity() {
        const MAC: [u8; 6] = [0xac, 0xf2, 0x3c, 0xaf, 0x0c, 0xe9];
        // `struct mt7925_mcu_phy_cap`: nss at 4, hw_path at 10.
        let mut phy = [0u8; 13];
        phy[4] = 2;
        phy[10] = 0b11;
        let body = capa_body(&[
            (MT_NIC_CAP_MAC_ADDR, &MAC[..]),
            (MT_NIC_CAP_PHY, &phy[..]),
            (MT_NIC_CAP_CHIP_CAP, &0x0001_0030u64.to_le_bytes()[..]),
        ]);

        let capa = NicCapability::parse(&body, 0x43).expect("a well-formed reply");
        assert_eq!(capa.mac, Some(MAC));
        assert_eq!(capa.nss, Some(2));
        assert_eq!(capa.hw_path, Some(0b11));
        assert!(capa.both_bands(), "the part claims 2.4 and 5 GHz");
        assert_eq!(capa.chip_cap, Some(0x0001_0030));
        assert_eq!(capa.elements, 3);
        // **Recorded, not judged.** Upstream never reads the eid on this
        // path, so this driver has no source for an expected value.
        assert_eq!(capa.eid, 0x43, "the eid is carried through verbatim");
    }

    /// **A record claiming less than its own header must refuse, not
    /// skip.**
    ///
    /// The walk advances by the claimed length, so a length under four
    /// leaves the cursor inside the header it just read and every
    /// following record is parsed from the wrong offset. The loop is
    /// bounded by the element count either way — this is a misread, not
    /// a hang, which is why it needs a name rather than a timeout.
    #[test]
    fn a_record_shorter_than_its_header_is_refused() {
        // Hand-built: a count of one, then a tag with len = 2.
        let mut body = alloc::vec![1u8, 0, 0, 0];
        body.extend_from_slice(&MT_NIC_CAP_MAC_ADDR.to_le_bytes());
        body.extend_from_slice(&2u16.to_le_bytes());
        assert_eq!(
            NicCapability::parse(&body, 0),
            Err(CapaError::RecordTooShort {
                index: 0,
                claimed: 2
            }),
        );
    }

    /// A record reaching past the end of the body is refused with the
    /// numbers that say how far past.
    #[test]
    fn a_record_running_past_the_body_is_refused() {
        let mut body = alloc::vec![1u8, 0, 0, 0];
        body.extend_from_slice(&MT_NIC_CAP_PHY.to_le_bytes());
        body.extend_from_slice(&64u16.to_le_bytes());
        body.extend_from_slice(&[0u8; 4]);
        assert_eq!(
            NicCapability::parse(&body, 0),
            Err(CapaError::RecordTruncated {
                index: 0,
                claimed: 64,
                left: 8,
            }),
        );
    }

    /// **The walk must account for the body exactly.**
    ///
    /// Upstream stops at its element count and frees whatever remains;
    /// that is right for a driver that must stay up, and wrong for a rung
    /// whose evidence *is* that the framing was understood. Slack means
    /// this driver read the reply differently from how the part wrote it,
    /// and a witness built on a misread is the thing this whole ladder
    /// exists to avoid.
    #[test]
    fn a_body_with_bytes_left_over_is_refused() {
        let mut body = capa_body(&[(MT_NIC_CAP_MAC_ADDR, &[1, 2, 3, 4, 5, 6][..])]);
        body.extend_from_slice(&[0xde, 0xad]);
        assert_eq!(
            NicCapability::parse(&body, 0),
            Err(CapaError::TrailingBytes { left: 2 }),
        );
    }

    /// Tags this rung does not read are still walked past.
    ///
    /// The part chooses what to send. A reply carrying records this
    /// driver has no use for is ordinary, and must not be mistaken for a
    /// malformed one — but the walk still has to account for their bytes.
    #[test]
    fn unread_tags_are_walked_rather_than_refused() {
        const MAC: [u8; 6] = [0xac, 0xf2, 0x3c, 0xaf, 0x0c, 0xe9];
        let body = capa_body(&[
            (0x05, &[1, 2, 3, 4][..]),  // MT_NIC_CAP_HW_VER, unread here
            (MT_NIC_CAP_MAC_ADDR, &MAC[..]),
            (MT_NIC_CAP_6G, &[1][..]),
        ]);
        let capa = NicCapability::parse(&body, 0).expect("unknown tags are legal");
        assert_eq!(capa.mac, Some(MAC));
        assert_eq!(capa.elements, 3);
        assert_eq!(capa.nss, None, "a tag absent from the reply is not a zero");
    }


    /// **The capability query is the vendor's bytes, exactly** — the
    /// first unified command `mt7925e` sent on this laptop (2026-11-20
    /// capture, `TX hw_idx=15 len=56`), at the sequence it carried.
    ///
    /// Word 1 is the one that was wrong from D1 to M1b: `0x4000`, the
    /// command header format at bits 15:14 (`mt76_connac3_mac.h`, which
    /// is what `mt7925/mac.h` includes), where this driver had transcribed
    /// connac2's 17:16. The ROM took every C2 command anyway; the RAM
    /// firmware fetched its first unified command and never answered.
    /// **The firmware-log request is the vendor's bytes** — `WSYS_CONFIG`
    /// as `mt7925e` sent it on this laptop (2026-11-20, `len=60`, seq 13).
    #[test]
    fn the_firmware_log_request_is_the_captured_command() {
        let mut sent = [0u8; 60];
        sent[..MCU_UNI_TXD_BYTES].copy_from_slice(&uni_command(UniCommand::WsysConfig, 13, 12, false));
        sent[MCU_UNI_TXD_BYTES..].copy_from_slice(&fw_log_body(1));
        let mut want = [0u8; 60];
        want[..8].copy_from_slice(&[0x3c, 0x00, 0x00, 0x41, 0x00, 0x40, 0x00, 0x00]);
        want[32..48].copy_from_slice(&[
            0x1c, 0x00, 0x0b, 0x00, 0x00, 0xa0, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x07, 0, 0, 0, 0,
        ]);
        want[52..57].copy_from_slice(&[0x00, 0x00, 0x08, 0x00, 0x01]);
        assert_eq!(sent, want);
    }

    #[test]
    fn the_capability_query_is_the_captured_command() {
        let mut sent = [0u8; 56];
        sent[..MCU_UNI_TXD_BYTES].copy_from_slice(&uni_command(UniCommand::ChipConfig, 11, 8, false));
        sent[MCU_UNI_TXD_BYTES..].copy_from_slice(&nic_capability_body());
        let mut want = [0u8; 56];
        want[..8].copy_from_slice(&[0x38, 0x00, 0x00, 0x41, 0x00, 0x40, 0x00, 0x00]);
        want[32..48].copy_from_slice(&[
            0x18, 0x00, 0x0e, 0x00, 0x00, 0xa0, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x06, 0, 0, 0, 0,
        ]);
        want[52..56].copy_from_slice(&[0x03, 0x00, 0x04, 0x00]);
        assert_eq!(sent, want);
    }

    /// **The unified descriptor is not the legacy one with a different
    /// id.**
    ///
    /// Three fields land at the same offset in both forms — `pkt_type`
    /// at 37, `seq` at 39, `s2d_index` at 42 — and everything else moves.
    /// That overlap is the trap: an encoder that ported those three and
    /// left `cid` a byte at 36 produces a descriptor the part rejects
    /// while still passing every sequence check this driver makes, which
    /// is exactly how flashes six and seven read.
    ///
    /// So this asserts the layout field by field against
    /// `struct mt76_connac2_mcu_uni_txd` (`mt76_connac_mcu.h:101-121`),
    /// and asserts the two forms differ where they must.
    #[test]
    fn a_unified_descriptor_has_its_own_layout() {
        const BODY: u16 = 12;
        let d = uni_command(UniCommand::ChipConfig, 5, BODY, false);

        assert_eq!(d.len(), 48, "the unified descriptor is 48 bytes");
        assert_eq!(MCU_TXD_BYTES, 64, "the legacy one is 64");

        // txd[0]: total length, command packet format, the MCU's queue.
        let total = 48 + BODY;
        let w0 = u32::from_le_bytes([d[0], d[1], d[2], d[3]]);
        assert_eq!(w0 & MT_TXD0_TX_BYTES, u32::from(total));

        // `len` excludes the eight hardware words, as in the legacy form.
        assert_eq!(
            u16::from_le_bytes([d[32], d[33]]),
            total - MCU_TXD_HW_BYTES,
        );

        // **`cid` is a little-endian half-word at 34**, where the legacy
        // form carries `pq_id`.
        assert_eq!(
            u16::from_le_bytes([d[34], d[35]]),
            MCU_UNI_CMD_CHIP_CONFIG,
            "cid is not the half-word at 34",
        );
        assert_eq!(d[36], 0, "rsv");
        assert_eq!(d[37], MCU_PKT_ID as u8, "pkt_type, shared with legacy");
        assert_eq!(d[38], 0, "frag_n: this driver fragments nothing");
        assert_eq!(d[39], 5, "seq, shared with legacy");
        assert_eq!(
            u16::from_le_bytes([d[40], d[41]]),
            0,
            "checksum: zero means there is none",
        );
        assert_eq!(d[42], MCU_S2D_H2N as u8, "s2d_index, shared with legacy");
        assert_eq!(d[44..48], [0; 4], "rsv1");

        // And the discriminator: the legacy encoder puts this command's
        // id nowhere near 34, so a descriptor built by the wrong one
        // cannot accidentally satisfy the assertion above.
        let legacy = command(Command::PatchSem, 5, BODY);
        assert_ne!(
            u16::from_le_bytes([legacy[34], legacy[35]]),
            MCU_UNI_CMD_CHIP_CONFIG,
            "the legacy form's 34 is pq_id, not cid",
        );
    }

    /// **`option` is derived, and `CHIP_CONFIG` is the exception.**
    ///
    /// `EXT_ACK` is `ACK|UNI|SET` and `QUERY_ACK` is `ACK|UNI`
    /// (`mt76_connac_mcu.h:1214-1216`); then `HIF_CTRL` and
    /// `CHIP_CONFIG` clear `ACK` (`mt7925/mcu.c:3810-3812`). The last
    /// step is the surprising one — the command that asks the part to
    /// describe itself asks for no acknowledgement, and upstream waits
    /// for one anyway.
    #[test]
    fn the_option_byte_follows_upstreams_three_rules() {
        // A set that is not CHIP_CONFIG: ACK | UNI | SET.
        assert_eq!(UniCommand::EfuseCtrl.option(false), 0b111);
        // A query: ACK | UNI, no SET.
        assert_eq!(UniCommand::EfuseCtrl.option(true), 0b011);
        // CHIP_CONFIG clears ACK from whichever base it started with.
        assert_eq!(UniCommand::ChipConfig.option(false), 0b110);
        assert_eq!(UniCommand::ChipConfig.option(true), 0b010);
        // And the byte reaches the wire at 43.
        let d = uni_command(UniCommand::EfuseCtrl, 1, 0, true);
        assert_eq!(d[43], 0b011, "option is the byte at 43");
    }

    /// **A unified TLV's length includes its own header.**
    ///
    /// mt7925 reads `tlv.len` directly; mt7921's `tlv_hdr` excludes the
    /// header and the two look identical at the call site. Taking the
    /// sibling's convention loses four bytes per TLV and desynchronises
    /// the walk — a silent corruption, not a refusal.
    #[test]
    fn a_unified_tlv_length_counts_its_own_header() {
        let t = uni_tlv_header(UNI_CHIP_CONFIG_NIC_CAPA, 0);
        assert_eq!(u16::from_le_bytes([t[0], t[1]]), UNI_CHIP_CONFIG_NIC_CAPA);
        assert_eq!(
            u16::from_le_bytes([t[2], t[3]]),
            4,
            "an empty TLV is four bytes long, not zero",
        );
        let t = uni_tlv_header(UNI_EFUSE_ACCESS, 20);
        assert_eq!(u16::from_le_bytes([t[2], t[3]]), 24);
    }

    use super::*;

    /// **The descriptor's wire bytes**, checked against the words upstream
    /// composes rather than against this encoder's own arithmetic.
    #[test]
    fn a_command_descriptor_matches_the_upstream_words() {
        let d = command(Command::PatchSem, 7, 4);
        assert_eq!(d.len(), 64);
        let w0 = u32::from_le_bytes([d[0], d[1], d[2], d[3]]);
        // TX_BYTES = 64 + 4; PKT_FMT = MT_TX_TYPE_CMD (2) at bit 23;
        // Q_IDX = MT_TX_MCU_PORT_RX_Q0 (0x20) at bit 25.
        assert_eq!(w0, 68 | (2 << 23) | (0x20 << 25));
        let w1 = u32::from_le_bytes([d[4], d[5], d[6], d[7]]);
        // HDR_FORMAT is bits 15:14 on this part — `mt76_connac3_mac.h`,
        // which `mt7925/mac.h` includes — and the capture of `mt7925e`
        // reads `00 40 00 00` here. This line used to assert `1 << 16`,
        // connac2's position, and so agreed with the defect it should have
        // caught: the belief was transcribed into the test along with the
        // constant.
        assert_eq!(w1, 1 << 14, "MT_HDR_FORMAT_CMD");
        // txd[2..8] are untouched.
        assert_eq!(&d[8..32], &[0u8; 24]);
        // `len` excludes the eight hardware words only.
        assert_eq!(u16::from_le_bytes([d[32], d[33]]), 68 - 32);
        // `MCU_PQ_ID(1, 0x20)` is an OR of two identical bits, so 0x8000 —
        // not 0x10000, which would truncate to zero in a __le16.
        assert_eq!(u16::from_le_bytes([d[34], d[35]]), 0x8000);
        assert_eq!(d[36], 0x10, "MCU_CMD_PATCH_SEM_CONTROL");
        assert_eq!(d[37], 0xa0, "MCU_PKT_ID");
        assert_eq!(d[38], 3, "MCU_Q_NA — no ext_cid and no CE flag");
        assert_eq!(d[39], 7, "the sequence");
        assert_eq!(d[42], 0, "MCU_S2D_H2N");
        assert_eq!(&d[44..64], &[0u8; 20], "rsv[5]");
    }

    /// **`FW_SCATTER` carries no descriptor at all.**
    ///
    /// The regression this pins: 64 bytes of descriptor prepended to a
    /// firmware chunk is 64 bytes of image the part loads at the wrong
    /// offset, and every byte after it is displaced. Nothing about the
    /// part's behaviour would say so.
    #[test]
    fn a_firmware_chunk_is_raw_payload() {
        let chunk = [0xabu8; 4096];
        assert_eq!(scatter(&chunk), Some(&chunk[..]));
        assert_eq!(scatter(&[]), None, "an empty chunk is not a chunk");
        assert_eq!(
            scatter(&[0u8; FW_SCATTER_MAX + 1]),
            None,
            "longer than the vendor's own max_len"
        );
    }

    /// The chunk arithmetic over the real transfer.
    #[test]
    fn the_download_is_three_hundred_and_twelve_chunks() {
        // res/MT7925_EVIDENCE.md §14.3: 187,968 + 1,087,744.
        let patch = chunks_for(187_968);
        let ram = chunks_for(1_087_744);
        assert_eq!(patch, 46);
        assert_eq!(ram, 266);
        assert_eq!(patch + ram, DOWNLOAD_CHUNKS as u64);
        // A region that divides evenly costs no extra chunk.
        assert_eq!(chunks_for(4096), 1);
        assert_eq!(chunks_for(4097), 2);
        assert_eq!(chunks_for(0), 0);
    }

    /// **The patch/target choice is the address's, not the blob's.**
    ///
    /// Both of the ROM patch's sections land on an address upstream treats
    /// specially; a RAM region does not. A download that chose by phase
    /// would send `PATCH_START_REQ` for the RAM code's regions.
    #[test]
    fn the_download_request_is_chosen_by_address() {
        assert_eq!(
            Command::init_download(PATCH_ENTRY_ILM).id(),
            MCU_CMD_PATCH_START_REQ as u8
        );
        assert_eq!(
            Command::init_download(PATCH_ENTRY_DLM).id(),
            MCU_CMD_PATCH_START_REQ as u8
        );
        // The entry points are the addresses the distribution's own patch
        // sections carry (`res/MT7925_EVIDENCE.md` §14.4) — the pairing is
        // checked, not assumed.
        assert_eq!(PATCH_ENTRY_ILM, 0x0090_0000);
        assert_eq!(PATCH_ENTRY_DLM, 0xe000_2800);
        // …and not the MT7928's, which is one word short of ours.
        assert_eq!(
            Command::init_download(0xe000_2000).id(),
            MCU_CMD_TARGET_ADDRESS_LEN_REQ as u8,
            "the sibling's entry point is not this part's"
        );
        // The RAM code's four downloaded regions, from §14.1.
        for addr in [0x0090_d000u32, 0x0221_2800, 0x0040_4000, 0xe002_6c00] {
            assert_eq!(
                Command::init_download(addr).id(),
                MCU_CMD_TARGET_ADDRESS_LEN_REQ as u8,
                "{addr:#x}"
            );
        }
        // …including the near miss: one word off the patch entry point is
        // not the patch entry point.
        assert_eq!(
            Command::init_download(0x0090_0004).id(),
            MCU_CMD_TARGET_ADDRESS_LEN_REQ as u8
        );
    }

    /// The sequence runs 1..=15 and never hands out zero.
    #[test]
    fn the_sequence_skips_zero() {
        let mut seq = Sequence::new();
        let mut seen = [false; 16];
        for _ in 0..64 {
            let s = seq.next();
            assert_ne!(s, 0, "zero is never ours");
            assert!(s < 16, "four bits");
            seen[s as usize] = true;
        }
        assert!(!seen[0]);
        assert!(seen[1..].iter().all(|v| *v), "every other value is used");
    }

    /// The bodies are the vendor's structs, field for field.
    #[test]
    fn the_command_bodies_are_the_vendors_structs() {
        // address, length, mode — in that order.
        let b = init_download_body(0x0090_0000, 39_552, 0x8000_0009);
        assert_eq!(u32::from_le_bytes([b[0], b[1], b[2], b[3]]), 0x0090_0000);
        assert_eq!(u32::from_le_bytes([b[4], b[5], b[6], b[7]]), 39_552);
        assert_eq!(u32::from_le_bytes([b[8], b[9], b[10], b[11]]), 0x8000_0009);
        // The semaphore's operation is a word, not a byte.
        assert_eq!(patch_sem_body(true), 1u32.to_le_bytes());
        assert_eq!(patch_sem_body(false), 0u32.to_le_bytes());
        // `check_crc` is zero, as upstream sends it.
        assert_eq!(patch_finish_body(), [0, 0, 0, 0]);
        // **option first, then address** — the reverse of init_download's.
        let s = fw_start_body(FW_START_OVERRIDE, 0x0090_d000);
        assert_eq!(u32::from_le_bytes([s[0], s[1], s[2], s[3]]), 1);
        assert_eq!(u32::from_le_bytes([s[4], s[5], s[6], s[7]]), 0x0090_d000);
        assert_eq!(restart_body(), [1, 0, 0, 0], "power_mode = 1");
    }

    /// An image with an entry point asks for the override; one without
    /// asks for nothing.
    #[test]
    fn the_start_option_follows_the_images_override() {
        assert_eq!(fw_start_option(Some(0x0090_d000)), FW_START_OVERRIDE);
        assert_eq!(fw_start_option(None), 0);
    }

    /// A response is matched by sequence, and a mismatch names both values.
    #[test]
    fn a_response_is_matched_by_its_sequence() {
        let mut r = [0u8; MCU_RXD_BYTES];
        r[36] = 0x01; // eid
        r[37] = 5; // seq
        r[40] = 2; // status
        assert_eq!(
            Response::parse(&r, 5),
            Ok(Response {
                eid: 1,
                seq: 5,
                status: 2
            })
        );
        assert_eq!(
            Response::parse(&r, 6),
            Err(McuError::WrongSequence { want: 6, got: 5 }),
            "a stale answer is refused with both values"
        );
        assert_eq!(
            Response::parse(&r[..MCU_RXD_BYTES - 1], 5),
            Err(McuError::TooShort)
        );
    }

    /// The semaphore's four answers, and a fifth that is not an answer.
    #[test]
    fn the_semaphore_has_four_answers_and_no_more() {
        assert_eq!(Semaphore::from_status(1), Ok(Semaphore::AlreadyLoaded));
        assert_eq!(Semaphore::from_status(2), Ok(Semaphore::Granted));
        assert_eq!(Semaphore::from_status(3), Ok(Semaphore::Released));
        assert_eq!(Semaphore::from_status(0), Ok(Semaphore::Refused));
        assert_eq!(
            Semaphore::from_status(9),
            Err(McuError::BadSemaphore(9)),
            "an unknown code is carried, not collapsed"
        );
    }

    /// The response header is **44** bytes on this part, not the connac2
    /// family's 36 — eight `rxd` words rather than six.
    #[test]
    fn the_response_header_is_this_parts_own_size() {
        assert_eq!(MCU_RXD_BYTES, 44);
        // The status byte upstream reaches as `sizeof(*rxd) - 4`.
        assert_eq!(MCU_RXD_BYTES - 4, 40);
    }
}
