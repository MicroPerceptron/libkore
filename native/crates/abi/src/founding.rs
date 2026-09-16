//! `founding` — the **founding record**: the boot chain's testimony as a
//! frozen boot-page block, so a machine whose console scrolled away (or
//! never had one) can still be asked *which rows came up and why not*.
//!
//! The founder already renders each row's outcome — ready, NACK rung,
//! beacon note — into one console line and drops it.  On metal that line
//! is typically invisible: the testimony scrolls past before the shell
//! exists, and the boot gate's log is a build-host artifact.  This block
//! is the same testimony made durable: the kernel keeps one ledger,
//! updates it as each row settles, and stamps a consistent snapshot into
//! every task's read-only boot page after the clock block — ambient data
//! in the [`ClockPage`](crate::clock::ClockPage) sense. Shell diagnostics
//! read their own boot page and print the relevant rows on demand; no service
//! face, no kick, no new authority.
//!
//! **Seqlock discipline.**  Founding is one-shot on the boot chain, but
//! rows can also be founded post-boot (the exec serve step), so a spawn
//! on another CPU may copy the ledger while a founder writes it.  The
//! kernel bumps [`seq`](FoundingBlock::seq) to odd, writes, bumps to
//! even; the stamping copy retries until it observes a stable even
//! `seq`.  A stamped page therefore carries `seq` even, and in-task
//! readers need no loop — their copy is immutable.
//!
//! Native-endian, naturally aligned, read zero-copy like the other boot
//! blocks; additive evolution only (append rows, claim reserved bytes —
//! never renumber, never resize).

use zerocopy::byteorder::little_endian::{U16, U32, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// Layout version.  A zeroed block reads `abi == 0` and exposes nothing
/// (the pre-block pages fail closed, like the geometry and clock blocks).
pub const FOUNDING_ABI: u16 = 1;

/// Ledger capacity. The manifest founds 25 rows after the Lunar Lake Xe
/// validation row. The next power-of-two ceiling leaves seven bounded slots
/// for later platform rows; the founder's compile-time gate still refuses any
/// manifest that would make `record_row` drop testimony silently.
///
/// Raising this resizes a frozen boot-page block and so shifts
/// `MACHINE_BLOCK_AT` for every image that reads the boot page: legal only
/// in lockstep with a full service-ELF rebuild, never on its own.
pub const MAX_FOUNDING_ROWS: usize = 32;

/// Bytes reserved for a row's service name (NUL-padded).  Longer names
/// are truncated at record time — the record is testimony, not an
/// [`ImageName`](crate::image::ImageName) (which stays 64 bytes).
pub const FOUNDING_NAME_BYTES: usize = 32;

/// Row flag: the slot describes a founded row (zeroed slots are absent).
pub const FOUNDING_PRESENT: u16 = 1 << 0;
/// Row flag: the row reached ring 3 and parked (the executor edge).
pub const FOUNDING_READY: u16 = 1 << 1;
/// Row flag: the row's capability-channel beacon was drained (the
/// service's own serving edge; readiness proper is READY ∧ BEACONED).
pub const FOUNDING_BEACONED: u16 = 1 << 2;
/// Row flag: the row NACKed founding; [`nack`](FoundingRow::nack) says
/// which rung.
pub const FOUNDING_NACKED: u16 = 1 << 3;

// ---------------------------------------------------------------------------
// Note vocabulary: rung, detail, and testimony.
//
// A row's [`note`](FoundingRow::note) is `[7:0]` rung, `[63:8]` rung-specific
// detail — one convention, so a reader comparing a noted row against a refused
// one need not learn two.  The TOP bits are carved out of the detail half for a
// third thing: **testimony**, a fact a row proved about itself that is not a
// complaint.  Testimony is not a note; a row carrying only testimony is fully
// up, and every reader that asks "is this row degraded?" must ask
// [`note_report`](FoundingRow::note_report), never the raw field.
//
// These live HERE, beside the decoder ring, rather than with the beacon cookies
// in `kore-rt`: once the founder stops masking them out (so a machine with no
// serial cable can still read the proof off its boot page), they are boot-page
// vocabulary, and the block that stores a value owns the words for reading it.
// `kore-rt` re-exports them, so the services that SET them are unchanged.

/// Testimony bit: the frame service's founding self-test transmitted a frame
/// and received one back — bulk OUT, the wire, bulk IN, the transfer event, the
/// interrupt route, the classifier, and the vif's own ring, end to end.
pub const NET_FRAME_RT_READY: u64 = 1 << 62;

/// Testimony bit: while that frame landed in the vif whose filter accepted it,
/// a second vif filtered to another protocol stayed **empty**, with neither vif
/// dropping anything.
///
/// The structural half of per-vif isolation, proven on the real datapath.  It
/// deliberately does not claim the *authority* half — both vifs belong to one
/// client here — which is proven in host tests against the per-pair policy and
/// ownership tables.
pub const NET_VIF_ISOLATION_READY: u64 = 1 << 61;

/// Testimony bit: the frame that round-tripped for [`NET_FRAME_RT_READY`] did
/// so through the **PCI** host — descriptor rings, tail doorbells, the
/// device's done-bit write-back and its MSI-family route — rather than the
/// USB one.  Derived, never declared: set only beside `NET_FRAME_RT_READY`,
/// and only when the claim ladder's PCI rung was the host that earned it.
/// Its absence on a boot with no PCI NIC is the honest answer, which is
/// what lets one gate require it and its sibling require it gone.
pub const NET_PCI_FRAME_READY: u64 = 1 << 60;

/// Testimony bit: the frame row claimed the machine's **Wi-Fi** function
/// (the Realtek RTL8922AE, `drivers/net/rtw89`), powered it on, verified the
/// pinned firmware's digest, downloaded it over the firmware-command DMA
/// ring, and had the running firmware answer `GET_FEATURE` with a `PHY_CAP`
/// naming a stream each way — the #388 D1 go/no-go, observed from a founded
/// ring-3 task.  Derived from the claim ladder's Wi-Fi rung having bound the
/// device and from what its firmware said, never declared.  Metal-only: no
/// QEMU profile carries the part, so a boot with no radio honestly lacks
/// the bit.
///
/// Bit 59 is the top of the accel-xe row's engine-evidence field
/// (`[59:44]` of its refusal notes) and 63 is INIT's control-plane bit, so
/// this bit is the one that made the carve **row-scoped**: see
/// [`SERVING_WITNESS_BITS`].
pub const NET_WIFI_FW_ALIVE: u64 = 1 << 59;

/// Testimony bit: after the single round trip, the frame service's founding
/// self-test **loaded** the datapath — a full transmit ring of unicast
/// frames to whoever answered the first one, every completion reaped as
/// sent, every reply landed in the vif that asked, nothing dropped on
/// either vif and the other vif still empty — with the receive ring
/// drained as a client would drain it.  What the single frame cannot prove:
/// descriptor batching behind one doorbell per sweep, completions in ring
/// order across sweeps, refill under a burst, moderated interrupts reaping
/// whole batches, and isolation under pressure rather than at rest.
/// Derived from the counters, never declared.  A network that answers one
/// probe but not a burst honestly lacks the bit and the row still serves.
pub const NET_LOAD_READY: u64 = 1 << 58;

/// Every testimony bit a readiness note may carry, so a reader masks them in
/// one place and a new one cannot be forgotten.
///
/// **The carve is the frame row's** ([`ROW_NAME_NET`]): every bit in it is
/// a fact only that row establishes, and the 64-bit note's upper bits are
/// other rows' rung detail — the accel-xe row packs evidence up to bit 59,
/// the NPU row a stage byte at `[63:56]`, INIT its control-plane bit at 63.
/// So readers apply the carve *by row*, through [`witness_of`] and
/// [`note_report_of`] (or the [`FoundingRow`] methods over them), never
/// to an arbitrary note: on any other row these bits are detail, kept
/// whole in its report and never printed as a proof.
pub const SERVING_WITNESS_BITS: u64 = NET_FRAME_RT_READY
    | NET_VIF_ISOLATION_READY
    | NET_PCI_FRAME_READY
    | NET_WIFI_FW_ALIVE
    | NET_LOAD_READY;

/// Whether the row named `name` carries testimony in its note's carve.
pub fn row_carries_testimony(name: &[u8]) -> bool {
    name == ROW_NAME_NET
}

/// The testimony a row's readiness payload carries: the carve, on the row
/// that owns it; nothing on any other row.
pub fn witness_of(name: &[u8], payload: u64) -> u64 {
    if row_carries_testimony(name) {
        payload & SERVING_WITNESS_BITS
    } else {
        0
    }
}

/// A row's readiness note proper — rung and detail, with the testimony
/// masked off where the row carries any.  **This is what "is this row
/// degraded?" means**: `0` is fully up, nonzero names a rung.
pub fn note_report_of(name: &[u8], payload: u64) -> u64 {
    payload & !witness_of(name, payload)
}

/// Testimony must never reach into the rung byte: a bit that did would make a
/// proven row decode as a refusal rung, which is the exact inversion this
/// vocabulary exists to prevent.  The detail half is 56 bits wide and the
/// testimony carve grows downward from the top, so this holds with room — and
/// says so at build time rather than at the one boot nobody can re-run.
const _: () = assert!(
    SERVING_WITNESS_BITS & 0xFF == 0,
    "a testimony bit landed in the note's rung byte"
);

/// Manifest name of the frame-service row, whose note vocabulary
/// [`FoundingRow::note_meaning`] decodes and whose testimony is
/// [`SERVING_WITNESS_BITS`].  Named here so the decoder and the readers that
/// pick the row out of the ledger cannot drift apart on a literal.
pub const ROW_NAME_NET: &[u8] = b"kore-svc-net";
/// The Lunar Lake Xe validation row, whose post-claim note tail differs from
/// the iVPU accelerator's bring-up ladder.
pub const ROW_NAME_ACCEL_XE: &[u8] = b"kore-svc-accel-xe";

/// Manifest-name prefix every accelerator row shares (the NPU row today, a
/// future GPU row) — matching it catches the whole family without a per-device
/// list.
pub const ROW_NAME_PREFIX_ACCEL: &[u8] = b"kore-svc-accel";

/// One row's founding outcome.
#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct FoundingRow {
    /// Service name, NUL-padded ([`FOUNDING_NAME_BYTES`]).
    pub name: [u8; FOUNDING_NAME_BYTES],
    /// The row's `COOKIE_SERVING` beacon payload — the readiness note
    /// (`0` = fully up; each row owns its rung vocabulary, e.g. the
    /// accel row's `kore-svc-accel-ivpu::note`), plus any
    /// [`SERVING_WITNESS_BITS`] testimony in the top bits.
    ///
    /// Read it through [`note_report`](Self::note_report) and
    /// [`witness`](Self::witness) rather than raw: the two halves answer
    /// different questions, and a row that is fully up may still have a
    /// nonzero field.
    pub note: U64,
    /// [`FOUNDING_PRESENT`] | [`FOUNDING_READY`] | [`FOUNDING_BEACONED`]
    /// | [`FOUNDING_NACKED`].
    pub flags: U16,
    /// NACK rung when [`FOUNDING_NACKED`]; `0` otherwise.
    pub nack: U16,
    /// Exact PCI vendor selected by this row, or zero when the row has no
    /// single exact PCI identity. This is descriptive manifest metadata, not
    /// proof that a function was present or claimed; the founding verdict and
    /// note retain those facts.
    pub pci_vendor: U16,
    /// Exact PCI device selected by this row; paired with [`pci_vendor`](Self::pci_vendor).
    pub pci_device: U16,
}

impl FoundingRow {
    /// The absent row — all zero, [`FOUNDING_PRESENT`] clear.
    pub const EMPTY: Self = Self {
        name: [0; FOUNDING_NAME_BYTES],
        note: U64::ZERO,
        flags: U16::ZERO,
        nack: U16::ZERO,
        pci_vendor: U16::ZERO,
        pci_device: U16::ZERO,
    };

    /// The recorded name as bytes up to the first NUL.
    pub fn name_bytes(&self) -> &[u8] {
        let len = self
            .name
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(FOUNDING_NAME_BYTES);
        &self.name[..len]
    }

    /// Whether this slot records a row at all.
    pub fn present(&self) -> bool {
        self.flags.get() & FOUNDING_PRESENT != 0
    }

    /// Readiness proper: ring-3 park **and** the drained beacon.
    pub fn serving(&self) -> bool {
        let f = self.flags.get();
        f & FOUNDING_READY != 0 && f & FOUNDING_BEACONED != 0
    }

    /// The NACK rung, if the row refused founding.
    pub fn nacked(&self) -> Option<u16> {
        (self.flags.get() & FOUNDING_NACKED != 0).then(|| self.nack.get())
    }

    /// The readiness note proper — rung and detail, with any testimony
    /// masked off.  **This is what "is this row degraded?" means**: `0` is
    /// fully up, nonzero names a rung.  A reader that tests the raw
    /// [`note`](Self::note) field instead reports a row that proved
    /// something about itself as degraded with rung `0`, which reads as
    /// "degraded, note 0: up" — nonsense, and alarming nonsense on the
    /// glance surface a serial-less machine depends on.
    pub fn note_report(&self) -> u64 {
        note_report_of(self.name_bytes(), self.note.get())
    }

    /// The testimony this row carried on its serving beacon — the facts only
    /// it could establish, proven before it beaconed.  See
    /// [`SERVING_WITNESS_BITS`].
    pub fn witness(&self) -> u64 {
        witness_of(self.name_bytes(), self.note.get())
    }

    /// The row's one exact PCI selector, if it declares one.
    ///
    /// This identifies what the driver row targets. It deliberately carries
    /// no BDF and grants no inventory or device authority; whether a matching
    /// function was found is reported by the row's ordinary verdict.
    pub fn pci_identity(&self) -> Option<(u16, u16)> {
        let pair = (self.pci_vendor.get(), self.pci_device.get());
        (pair != (0, 0)).then_some(pair)
    }

    /// The readiness note's rung byte (`note[7:0]`) decoded to English — the
    /// shell-side decoder ring for the founding-beacon note vocabulary, shared
    /// by health readers such as `status`. The detail bits (`note[63:8]`) carry
    /// rung-specific evidence; a glance surface decodes the common evidence
    /// bands and otherwise pairs this line with the raw note.
    ///
    /// **The vocabulary is per row above the shared ladder**, so this dispatches
    /// on the row's own name.  Rungs 1–5 are the driver-claim ladder every
    /// device row speaks (owned by `kore-svc-accel-ivpu::note`); from 11 up each
    /// row numbers its own refusals, and reading one row's tail with another's
    /// map produces a confident, wrong answer — `kore-svc-net`'s rung 11 is "no
    /// Ethernet adaptor on this controller", which the accel map would render as
    /// "no firmware boot module (rebuild the boot media)".  A new row with a tail
    /// vocabulary gets an arm here.  Append-only, like the maps it decodes.
    ///
    /// Masks to the rung byte, so it is unaffected by [`SERVING_WITNESS_BITS`];
    /// pair it with [`note_report`](Self::note_report) to decide whether to
    /// print it at all.
    pub fn note_meaning(&self) -> &'static [u8] {
        let rung = self.note.get() & 0xFF;
        // The shared claimant ladder: same numbering, same words, every device
        // row.  Rung 4 is deliberately NOT here — each row names the register
        // window that failed to decode, and accel's is its buttress BAR.
        match rung {
            0 => return b"up",
            1 => return b"no matching device on this machine",
            2 => return b"every matching function already claimed",
            3 => return b"device broker refused the claim",
            5 => return b"claim realized no resources (BAR/DMA-window refusal)",
            _ => {}
        }
        if self.name_bytes() == ROW_NAME_NET {
            return Self::net_note_meaning(rung);
        }
        if self.name_bytes() == ROW_NAME_ACCEL_XE {
            return Self::xe_note_meaning(rung, (self.note.get() >> 32) as u8);
        }
        Self::accel_note_meaning(rung)
    }

    /// The frame service's own tail (`kore-svc-net::note`).  A note there is a
    /// *degraded-but-serving* report, never a failure — the row founds and
    /// beacons either way, and says here what it could not do.
    fn net_note_meaning(rung: u64) -> &'static [u8] {
        match rung {
            4 => b"the claimed controller's registers do not decode",
            11 => b"no Ethernet adaptor on the claimed controller (move the cable to the other port group)",
            12 => b"adaptor up, but the founding self-test moved no frame in its budget",
            13 => b"could not open its own vif: a window carve or ring attach refused",
            // `BRING_UP_BASE + r` for USB core rung `r` (`kore-svc-net::note`
            // over `metal::usbnet_rung`).  A note in this band means the
            // adaptor IS plugged in and refused — never "move the cable".
            17 => b"USB bring-up: the descriptors do not describe a CDC Ethernet function",
            18 => b"USB bring-up: a control transfer or endpoint setup was refused",
            19 => b"USB bring-up: the adaptor's MAC address string does not decode",
            20 => b"USB bring-up: no DMA buffer could be drawn for the bulk rings",
            21 => b"USB bring-up: a bounded USB resource was exhausted",
            // `PCI_BRING_UP_BASE + r` for `kore-e1000e` rung `r`
            // (`kore-svc-net::note` over `kore_e1000e::Error::rung`).  The
            // function was claimed and decoded; the ladder stopped here.
            33 => b"PCI bring-up: the NIC's BAR does not decode a live controller",
            34 => b"PCI bring-up: the controller reset never self-cleared",
            35 => b"PCI bring-up: the NVM auto-read never completed",
            36 => b"PCI bring-up: no valid station address in the receive-address registers",
            37 => b"PCI bring-up: no DMA buffer could be drawn for the rings",
            38 => b"PCI bring-up: the transmit ring is full",
            // `WIFI_BRING_UP_BASE + r` for `kore-rtw89` rung `r`
            // (`kore-svc-net::note` over `kore_rtw89::Error::rung`).  The
            // Wi-Fi function was claimed; the D1 ladder stopped here.
            49 => b"Wi-Fi bring-up: the radio's BAR2 does not decode a live controller",
            50 => b"Wi-Fi bring-up: no firmware module in this boot (run `make fetch-realtek-firmware`)",
            51 => b"Wi-Fi bring-up: the firmware module is not the pinned blob (size, digest, cut or header)",
            52 => b"Wi-Fi bring-up: a MAC power sequence did not complete",
            53 => b"Wi-Fi bring-up: the efuse could not be read or does not parse",
            54 => b"Wi-Fi bring-up: no DMA buffer could be drawn, or one lies beyond the part's 36-bit reach",
            55 => b"Wi-Fi bring-up: the DMA engine's pre-init did not complete (DLE, busy channel, BD-RAM bound)",
            56 => b"Wi-Fi bring-up: the firmware download did not complete (path, ring stall, or the image was rejected)",
            57 => b"Wi-Fi bring-up: the running firmware never answered the first command",
            58 => b"Wi-Fi bring-up: normal MAC initialization did not complete (packet memory, scheduler or CAM)",
            59 => b"Wi-Fi bring-up: PHY register configuration did not complete (RF busy or command retirement)",
            60 => b"Wi-Fi bring-up: MAC feature command completion failed",
            _ => b"(unmapped note; see the row's note map)",
        }
    }

    /// The Xe row's tail (`kore-svc-accel-xe::note`). Rungs 1-15
    /// are the read-only validation ladder; 16 is the GuC bring-up ladder's
    /// completion (deliberately still degraded: submission is not
    /// implemented); 17+ are `16 + kore-xe::xe::bring_up` rungs. The
    /// note's bits 40/41 carry the warm-boot facts (GuC live at claim /
    /// WOPCM locked at claim), which this glance-level decoder leaves to the
    /// row's note map.
    fn xe_note_meaning(rung: u64, detail_ordinal: u8) -> &'static [u8] {
        match rung {
            4 => b"graphics GMD_ID reads all-ones: the function is not live behind BAR0",
            6 => b"graphics identity exists, but boot ownership was not positively proven",
            7 => b"graphics identity exists, but catalog BAR0 is not user-mappable",
            8 => b"graphics identity and BAR0 exist, but no IOMMU binding covers the driver",
            9 => b"graphics inventory facts satisfy the selector, but the broker refused it",
            10 => b"no Intel graphics function of the display class was enumerated",
            11 => b"no firmware boot module (provision the generation's GuC image and rebuild the boot media)",
            12 => b"firmware module is not the pinned GuC image for this generation",
            13 => b"graphics or media GMD identity register could not be read",
            14 => b"PCI identity + GMD versions select no implemented generation (detail: refusal ordinal)",
            15 => b"PCI/BAR/GMD/firmware proved; Xe2 MCR/PAT/MOCS submission owner not implemented",
            16 => b"GuC READY with HWConfig retained, PAT/MOCS verified; native submission epoch not implemented",
            51 => b"generation selected but Frozen: bring-up withheld (promote the sibling to Supported to run it)",
            17 => b"bring-up rung 1: MCR steering semaphore/selector probe refused",
            18 => b"bring-up rung 2: PAT program or all-replica readback refused",
            19 => match detail_ordinal {
                1 => b"bring-up rung 3: no vacant GGTT span for resident allocation",
                19 => b"bring-up rung 3: firmware display layout cannot be safely retained",
                20 => b"bring-up rung 3: running firmware GuC prevents GGTT handoff",
                21 => b"bring-up rung 3: firmware GGTT aperture reads all ones",
                2 | 4 | 5 | 6 | 7 | 17 | 18 => b"bring-up rung 3: invalid GGTT allocation geometry",
                3 => b"bring-up rung 3: GSM aperture is too small",
                8 | 9 | 10 => b"bring-up rung 3: scratch DMA address is not representable",
                11 | 13 => b"bring-up rung 3: scratch DMA backing is too small",
                12 => b"bring-up rung 3: scratch PAT does not match the generation",
                14 | 15 => b"bring-up rung 3: retained GGTT mapping conflicts with allocation",
                16 => b"bring-up rung 3: GGTT write-ordering register is outside the MMIO view",
                _ => b"bring-up rung 3: GGTT window/scratch initialization refused",
            },
            20 => b"bring-up rung 4: resident firmware GGTT map refused",
            21 => b"bring-up rung 5: minimal ADS build refused",
            22 => b"bring-up rung 6: GuC load parameter block refused",
            23 => b"bring-up rung 7: GT force-wake acquire or release refused",
            24 => b"bring-up rung 8: live-GuC reset workaround timed out (warm-boot quiesce)",
            25 => b"bring-up rung 9: GuC-domain reset did not complete",
            26 => b"bring-up rung 10: WOPCM partition refused",
            27 => b"bring-up rung 11: parameter/RSA staging refused",
            28 => b"bring-up rung 12: firmware DMA transfer did not complete",
            29 => b"bring-up rung 13: GuC never reached READY (status word in detail)",
            30 => b"bring-up rung 14: GET_HWCONFIG query or fetch refused",
            31 => b"bring-up rung 15: MOCS program or readback refused",
            32 => b"bring-up rung 16: DMA pool exhausted (bytes in detail)",
            33 => b"bring-up rung 17: native-submission WA/parameter block refused",
            34 => b"bring-up rung 18: full-ADS inputs or build refused",
            35 => b"bring-up rung 19: submission-epoch GuC boot refused",
            36 => b"bring-up rung 20: CTB backing/GGTT map refused",
            37 => b"bring-up rung 21: CTB self-config KLVs refused",
            38 => b"bring-up rung 22: CONTROL_CTB enable refused",
            39 => b"bring-up rung 23: golden-recording admission refused (context/recorder/destination)",
            40 => b"bring-up rung 24: golden-LRC hardware recording refused or reset-required",
            41 => b"bring-up rung 25: queue PPGTT build or witness map refused",
            42 => b"bring-up rung 26: PPGTT TLB-invalidation round trip refused",
            43 => b"bring-up rung 27: witness storage, NOP job, or compute-LRC build refused",
            44 => b"bring-up rung 28: memirq page or tracked control plane refused",
            45 => b"bring-up rung 29: REGISTER_CONTEXT or default policy did not complete",
            46 => b"bring-up rung 30: first submit or MODE_SET enable event did not complete",
            47 => b"bring-up rung 31: NOP-job completion evidence did not arrive",
            48 => b"bring-up rung 32: CCS0 ring enable refused (RCU_MODE/RING_MODE/HWSP readback)",
            50 => {
                b"recording epoch alive: submission GuC READY, CT enabled; golden recording and native queue not implemented"
            }
            _ => b"(unmapped note; see the row's note map)",
        }
    }

    /// The accelerator tail (`kore-svc-accel-ivpu::note` and
    /// `kore-ivpu::accel::bring_up_rung`, the latter at `16 + backend rung`).
    /// Also the historical default for any row without a tail of its own, which
    /// is what every reader got for every row before the dispatch above existed.
    fn accel_note_meaning(rung: u64) -> &'static [u8] {
        match rung {
            4 => b"buttress (BAR4) reads all-ones: unrouted or parked beyond the claim wake",
            6 => b"generation selected but Frozen: bring-up withheld (promote the sibling to Supported to run it)",
            11 => b"no firmware boot module (provision the blob, rebuild the boot media)",
            12 => b"firmware module is not the pinned blob (short or digest mismatch)",
            13 => b"bring-up completed but the async NOP job path did not prove",
            14 => b"buttress window (BAR4) would not realize: BAR unassigned by firmware?",
            15 => b"Intel NPU selects no implemented generation (detail: refusal ordinal, device id)",
            17 => b"bring-up rung 1: buttress power-up refused",
            18 => b"bring-up rung 2: DMA pool exhausted building SMMU structures",
            19 => b"bring-up rung 3: stage-1 walker build or map refused",
            20 => b"bring-up rung 4: SMMU reset/context install refused",
            21 => b"bring-up rung 5: firmware verification or image plan refused",
            22 => b"bring-up rung 6: BOOT handshake never posted",
            23 => b"bring-up rung 7: firmware up but heartbeat unanswered",
            24 => b"bring-up rung 8: HWS scheduler or queue lifecycle refused",
            _ => b"(unmapped note; see the row's note map)",
        }
    }
}

/// The founding record block — the fifth frozen boot-page block, after
/// the clock block (`kore-rt::entry::FOUNDING_BLOCK_AT`).
#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct FoundingBlock {
    /// Seqlock sequence — **odd** while the kernel is mid-update.  Even
    /// (and typically nonzero) in every stamped page.
    pub seq: U32,
    /// Layout version ([`FOUNDING_ABI`]); `0` = no record (fail closed).
    pub abi: U16,
    /// Occupied row slots (`rows[..count]`; presence is still per-row —
    /// a mid-founding snapshot may carry settled rows past unsettled
    /// ones, and absent slots read [`FoundingRow::present`] false).
    pub count: U16,
    /// Post-readiness terminal-row bitmap. Bit `i` says row `i` exited after
    /// its settled founding verdict. This claims the first reserved v1 word
    /// without moving the frozen block; older readers lawfully ignore it.
    /// Read through [`row_terminal`](Self::row_terminal).
    ///
    /// The second word is the corresponding reaped-row bitmap: teardown of
    /// the founder-held task handle completed after the terminal witness was
    /// recorded. Read through [`row_reaped`](Self::row_reaped).
    pub reserved: [U32; 2],
    /// The rows, in manifest order.
    pub rows: [FoundingRow; MAX_FOUNDING_ROWS],
}

impl FoundingBlock {
    /// The empty ledger (`abi` stamped, no rows yet).
    pub const EMPTY: Self = Self {
        seq: U32::ZERO,
        abi: U16::from_bytes(FOUNDING_ABI.to_le_bytes()),
        count: U16::ZERO,
        reserved: [U32::ZERO; 2],
        rows: [FoundingRow::EMPTY; MAX_FOUNDING_ROWS],
    };

    /// Whether the block carries a record this reader understands.
    pub fn validate(&self) -> bool {
        self.abi.get() == FOUNDING_ABI && (self.count.get() as usize) <= MAX_FOUNDING_ROWS
    }

    /// The recorded rows with their stable manifest indices (present slots
    /// only). Enumerating before filtering is load-bearing: a mid-founding
    /// snapshot can contain a vacant slot before a settled later row, and the
    /// terminal/reaped bitmaps are keyed by the original slot, not by a
    /// compacted visible-row ordinal.
    pub fn present_rows_indexed(&self) -> impl Iterator<Item = (usize, &FoundingRow)> {
        let count = (self.count.get() as usize).min(MAX_FOUNDING_ROWS);
        self.rows[..count]
            .iter()
            .enumerate()
            .filter(|(_, row)| row.present())
    }

    /// The recorded rows (present slots only).
    pub fn present_rows(&self) -> impl Iterator<Item = &FoundingRow> {
        self.present_rows_indexed().map(|(_, row)| row)
    }

    /// Find a row by exact service name.
    pub fn row_named(&self, name: &[u8]) -> Option<&FoundingRow> {
        self.present_rows().find(|r| r.name_bytes() == name)
    }

    /// Whether manifest row `idx` later exited after its founding verdict.
    pub fn row_terminal(&self, idx: usize) -> bool {
        idx < MAX_FOUNDING_ROWS && self.reserved[0].get() & (1u32 << idx) != 0
    }

    /// Whether manifest row `idx` was reaped after its terminal witness was
    /// committed. A terminal row can remain unreaped when its founder handle
    /// is unavailable; that is an observable lifecycle fault, not success.
    pub fn row_reaped(&self, idx: usize) -> bool {
        idx < MAX_FOUNDING_ROWS && self.reserved[1].get() & (1u32 << idx) != 0
    }
}

const _: () = assert!(core::mem::size_of::<FoundingRow>() == 48);
const _: () = assert!(
    core::mem::size_of::<FoundingBlock>() == 16 + 48 * MAX_FOUNDING_ROWS,
    "the founding block is frozen ABI"
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_block_validates_and_exposes_no_rows() {
        let b = FoundingBlock::EMPTY;
        assert!(b.validate());
        assert_eq!(b.present_rows().count(), 0);
    }

    #[test]
    fn a_zeroed_block_fails_closed() {
        let b =
            FoundingBlock::read_from_bytes(&[0u8; core::mem::size_of::<FoundingBlock>()]).unwrap();
        assert!(!b.validate(), "abi 0 must expose nothing");
    }

    #[test]
    fn a_recorded_row_round_trips() {
        let mut b = FoundingBlock::EMPTY;
        let mut row = FoundingRow::EMPTY;
        row.name[..14].copy_from_slice(b"kore-svc-accel");
        row.note = U64::from_bytes(4u64.to_le_bytes());
        row.flags =
            U16::from_bytes((FOUNDING_PRESENT | FOUNDING_READY | FOUNDING_BEACONED).to_le_bytes());
        row.pci_vendor = U16::from_bytes(0x8086u16.to_le_bytes());
        row.pci_device = U16::from_bytes(0x643eu16.to_le_bytes());
        b.rows[12] = row;
        b.count = U16::from_bytes(13u16.to_le_bytes());
        let found = b.row_named(b"kore-svc-accel").expect("recorded");
        assert!(found.serving());
        assert_eq!(found.note.get(), 4);
        assert_eq!(found.nacked(), None);
        assert_eq!(found.pci_identity(), Some((0x8086, 0x643e)));
        assert!(b.row_named(b"kore-svc-none").is_none());
        assert!(!b.row_terminal(12));
        assert!(!b.row_reaped(12));
        b.reserved[0] = (1u32 << 12).into();
        b.reserved[1] = (1u32 << 12).into();
        assert!(b.row_terminal(12));
        assert!(b.row_reaped(12));
        assert!(!b.row_terminal(MAX_FOUNDING_ROWS));
        assert!(!b.row_reaped(MAX_FOUNDING_ROWS));
    }

    #[test]
    fn indexed_rows_keep_manifest_slots_across_gaps() {
        let mut b = FoundingBlock::EMPTY;
        let mut early = FoundingRow::EMPTY;
        early.name[..5].copy_from_slice(b"early");
        early.flags = FOUNDING_PRESENT.into();
        let mut late = FoundingRow::EMPTY;
        late.name[..4].copy_from_slice(b"late");
        late.flags = FOUNDING_PRESENT.into();
        b.rows[1] = early;
        b.rows[12] = late;
        b.count = 13u16.into();

        let mut indexed = b.present_rows_indexed();
        let (idx, row) = indexed.next().expect("early row");
        assert_eq!(idx, 1);
        assert_eq!(row.name_bytes(), b"early");
        let (idx, row) = indexed.next().expect("late row");
        assert_eq!(idx, 12);
        assert_eq!(row.name_bytes(), b"late");
        assert!(indexed.next().is_none());
    }

    /// Build a serving row under `name` carrying `note`.
    fn serving_row(name: &[u8], note: u64) -> FoundingRow {
        let mut row = FoundingRow::EMPTY;
        row.name[..name.len()].copy_from_slice(name);
        row.note = U64::from_bytes(note.to_le_bytes());
        row.flags =
            U16::from_bytes((FOUNDING_PRESENT | FOUNDING_READY | FOUNDING_BEACONED).to_le_bytes());
        row
    }

    /// **A row that proved something about itself is UP, not degraded.**
    ///
    /// Testimony and the rung note share one `u64`, so every "is this row
    /// healthy?" reader has to ask [`FoundingRow::note_report`].  A reader
    /// that tests the raw field sees nonzero and renders `degraded, note 0:
    /// up` — the exact nonsense the glance surface exists to prevent, on the
    /// one row whose whole point is to say the datapath worked.
    #[test]
    fn testimony_alone_is_not_a_degraded_note() {
        let row = serving_row(ROW_NAME_NET, SERVING_WITNESS_BITS);
        assert!(row.serving());
        // Another row's note is detail all the way up: the accel-xe row's
        // engine evidence reaches bit 59, where the frame row keeps its
        // Wi-Fi testimony.
        let xe = serving_row(ROW_NAME_ACCEL_XE, NET_WIFI_FW_ALIVE | 0x11);
        assert_eq!(xe.witness(), 0);
        assert_eq!(xe.note_report(), NET_WIFI_FW_ALIVE | 0x11);
        assert_eq!(row.note_report(), 0, "testimony is not a rung note");
        assert_eq!(row.witness(), SERVING_WITNESS_BITS);
        assert_eq!(row.note_meaning(), b"up");
        assert_ne!(
            row.note.get(),
            0,
            "the raw field IS nonzero — that is the trap"
        );
    }

    /// Testimony and a rung coexist: the frame service proves reachability
    /// and isolation separately, so a row can attest one and still note what
    /// it could not do.  Neither half may eat the other.
    #[test]
    fn testimony_and_a_rung_decode_independently() {
        let note = NET_FRAME_RT_READY | 12 | (3 << 8);
        let row = serving_row(ROW_NAME_NET, note);
        assert_eq!(row.witness(), NET_FRAME_RT_READY);
        assert_eq!(
            row.note_report(),
            12 | (3 << 8),
            "rung and detail survive whole"
        );
        assert_eq!(
            row.note_meaning(),
            b"adaptor up, but the founding self-test moved no frame in its budget",
        );
    }

    /// **One rung, two vocabularies.**  Above the shared claim ladder each row
    /// numbers its own refusals, so the decoder must dispatch on the row's
    /// name.  Rung 11 is the case that matters at a bench: on the frame
    /// service it means "move the cable to the other controller", and the
    /// accelerator map would send someone to rebuild their boot media.
    #[test]
    fn rung_eleven_reads_differently_per_row() {
        let net = serving_row(ROW_NAME_NET, 11);
        let accel = serving_row(b"kore-svc-accel", 11);
        assert_ne!(net.note_meaning(), accel.note_meaning());
        assert!(net.note_meaning().starts_with(b"no Ethernet adaptor"));
        assert!(accel.note_meaning().starts_with(b"no firmware boot module"));
    }

    #[test]
    fn xe_validation_completion_is_visible_without_claiming_job_readiness() {
        let xe = serving_row(ROW_NAME_ACCEL_XE, 15);
        assert_eq!(xe.note_report(), 15);
        assert!(
            xe.note_meaning()
                .starts_with(b"PCI/BAR/GMD/firmware proved")
        );
        assert!(
            xe.note_meaning()
                .ends_with(b"submission owner not implemented")
        );
    }

    #[test]
    fn xe_claim_diagnosis_names_the_missing_catalog_fact() {
        let expected: &[(u64, &[u8])] = &[
            (
                6,
                b"graphics identity exists, but boot ownership was not positively proven",
            ),
            (
                7,
                b"graphics identity exists, but catalog BAR0 is not user-mappable",
            ),
            (
                8,
                b"graphics identity and BAR0 exist, but no IOMMU binding covers the driver",
            ),
            (
                9,
                b"graphics inventory facts satisfy the selector, but the broker refused it",
            ),
            (
                10,
                b"no Intel graphics function of the display class was enumerated",
            ),
        ];
        for &(rung, meaning) in expected {
            assert_eq!(serving_row(ROW_NAME_ACCEL_XE, rung).note_meaning(), meaning);
        }
    }

    /// The claim ladder (rungs 1-5, minus the row-specific register-decode
    /// rung 4) is shared vocabulary: same numbering, same words, every device
    /// row.  A reader must not have to know which row it is holding to read
    /// them.
    #[test]
    fn the_shared_claim_ladder_reads_alike_on_every_row() {
        for rung in [1u64, 2, 3, 5] {
            assert_eq!(
                serving_row(ROW_NAME_NET, rung).note_meaning(),
                serving_row(b"kore-svc-accel", rung).note_meaning(),
                "rung {rung} is the shared claim ladder",
            );
        }
        assert_ne!(
            serving_row(ROW_NAME_NET, 4).note_meaning(),
            serving_row(b"kore-svc-accel", 4).note_meaning(),
            "rung 4 names the register window that failed, which is per row",
        );
    }

    /// **"No adaptor" and "the adaptor refused" must not read alike.**  Rung
    /// 11 sends an operator to move a cable; the bring-up band says the cable
    /// is already right and names where the ladder stopped.  A decoder that
    /// collapsed them would undo the distinction the walk exists to keep.
    #[test]
    fn an_absent_adaptor_and_a_refused_one_read_differently() {
        let absent = serving_row(ROW_NAME_NET, 11);
        let refused = serving_row(ROW_NAME_NET, 19 | (3 << 8));
        assert!(absent.note_meaning().starts_with(b"no Ethernet adaptor"));
        assert!(refused.note_meaning().starts_with(b"USB bring-up"));
        assert_eq!(
            refused.note_report() >> 8,
            3,
            "the port rides the detail bits, so the note says WHICH port refused",
        );
    }

    /// The PCI band sits above the USB band and reads as its own ladder:
    /// a note there means a NIC was claimed and decoded, never "move the
    /// cable", and carries no port because a PCI function has none.
    #[test]
    fn pci_bring_up_notes_decode_and_differ_from_the_usb_band() {
        for rung in 33u64..=38 {
            let row = serving_row(ROW_NAME_NET, rung);
            assert!(
                row.note_meaning().starts_with(b"PCI bring-up"),
                "rung {rung} is the PCI band",
            );
            assert_eq!(row.note_report() >> 8, 0, "no detail bits on PCI");
        }
        assert_ne!(
            serving_row(ROW_NAME_NET, 33).note_meaning(),
            serving_row(ROW_NAME_NET, 17).note_meaning(),
            "the first rung of each band names a different fact",
        );
        assert!(
            serving_row(ROW_NAME_NET, 32)
                .note_meaning()
                .starts_with(b"(unmapped"),
            "the base itself is not a rung",
        );
    }

    /// A row with nothing to attest reports no testimony and no note — the
    /// state of every row in the manifest but one, and the reason adding this
    /// vocabulary changes no existing output.
    #[test]
    fn a_row_with_nothing_to_say_is_unchanged() {
        let row = serving_row(b"kore-svc-dagfs", 0);
        assert_eq!(row.witness(), 0);
        assert_eq!(row.note_report(), 0);
        assert_eq!(row.note_meaning(), b"up");
    }
}
