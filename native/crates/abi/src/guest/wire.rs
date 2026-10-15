//! Architecture-neutral guest window records and frozen semantic roles.
use zerocopy::{FromBytes, Immutable, IntoBytes};

use crate::cap::Rights;

/// Quadwords in the guest register image — sized once, for every
/// architecture, by the largest image any of them publishes: AArch64's
/// thirty-one general registers, two stack pointers, the program counter
/// and PSTATE make thirty-five, so forty leaves headroom.  x86-64 fills
/// eighteen and zero-fills the tail.  Frozen with the record layout: a
/// smaller image never shrinks it, a larger one is a wire break.
pub const GUEST_STATE_QWORDS: usize = 40;

/// The **semantic roles** the guest protocol cites in the register
/// image — the only register vocabulary this agnostic object carries.
/// Every architecture's wire order (the kernel's `state_to_wire` seam)
/// is required to place these roles at these indices; which physical
/// register realizes a role is that seam's private business.
pub mod state_index {
    /// The **programmed-I/O datum** register: its low byte carries the
    /// byte of a trapped one-byte port write, and a completing read
    /// emulation deposits the delivered byte here before resume.
    pub const IO_DATUM: usize = 0;
    /// The **instruction pointer** — what a completing emulator advances
    /// past the trapped instruction before `CTRL_KIND_WRITE_STATE` +
    /// `CTRL_KIND_RESUME`.
    pub const IP: usize = 15;

    const _: () = {
        assert!(IO_DATUM < super::GUEST_STATE_QWORDS);
        assert!(IP < super::GUEST_STATE_QWORDS);
    };
}

/// Bytes of one record slot, both directions — frozen.  Kind, sequence,
/// two payload words, the full register image and the decode-assist tail
/// fit with headroom; the pad is reserved-zero on write, ignored on read.
///
/// **Why 512, and what would reopen it (#765).**  The slot was 256 bytes
/// while the image was x86-64's eighteen quadwords (168 bytes used).  The
/// image is now sized for the largest architecture any port will publish
/// — AArch64's thirty-five ([`GUEST_STATE_QWORDS`]) — and the 24-byte
/// head + 320-byte image + 32-byte assist tail does not fit 256, so the
/// slot doubled: one wire break taken now, in place of one on the day an
/// Arm arm lands.  The cost is paid per forwarded exit and per
/// `WriteState`, about four more cache lines each way, invisible beside
/// the world switch itself; both rings still fit the one-page window
/// (3456 of 4096 bytes).  Weighed and declined: a role-indexed *partial*
/// image (every consumer would then carry a per-arch register map, which
/// is the leak this object exists to prevent) and a per-arch slot size
/// (the window geometry would stop being one frozen vocabulary).  Revisit
/// only against a measured exit-path cost on metal, and shrink the
/// *return* copy first — a lighter "advance the instruction pointer by
/// `n`" control kind ([`CTRL_KIND_WRITE_STATE`]'s note) spares most
/// completions the full image write-back — before touching the slot.
pub const GUEST_RECORD_BYTES: usize = 512;

/// The longest instruction any supported ISA can trap on — x86's fifteen
/// bytes; a fixed 4-byte Arm or RISC-V word fits with room.  Sizes the
/// exit record's raw-bytes assist form.
pub const GUEST_INSN_BYTES_MAX: usize = 15;

/// Exit-ring capacity — **2 by the #487 §5 proof** (publish-then-park
/// bounds ordinary exits to one; the terminal record is the only
/// possible cohabitant).  Not a tunable: revisiting it means reopening
/// the publish-then-park decision, not resizing a constant.
pub const GUEST_EXIT_RING_CAPACITY: u32 = 2;

/// Control-ring capacity — the largest sane hypervisor batch, drained to
/// empty at the vCPU's safe point.
pub const GUEST_CTRL_RING_CAPACITY: u32 = 4;

/// Doorbell cookie carried by the kernel's best-effort exit notice.  The
/// notice is only a level-triggered hint: `payload` is the latest published
/// sequence, while the exit ring remains the source of truth and must always
/// be drained completely after any wake.
pub const NOTICE_COOKIE_GUEST_EXIT: u64 = 0x6B72_5F67_4558_4954; // "kr_gEXIT"

/// Doorbell cookie carried after userland publishes control records. The
/// control ring is the source of truth; this notice only wakes the parked
/// vCPU so it can drain that ring.
pub const NOTICE_COOKIE_GUEST_CONTROL: u64 = 0x6B72_5F67_4354_524C; // "kr_gCTRL"

/// Userland's architecture-neutral first-entry values for one guest vCPU.
/// The kernel converts the guest-physical numbers to its strongly typed
/// address wrappers only after the founding request is authenticated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GuestVcpuConfig {
    pub entry: u64,
    pub stack: u64,
    pub page_table_root: u64,
}

/// Exact lifecycle rights on a successfully founded guest-vCPU handle.
pub const GUEST_VCPU_HANDLE_RIGHTS: Rights = Rights::ACCESS
    .union(Rights::MODIFY)
    .union(Rights::DERIVE)
    .union(Rights::REVOKE);

// ── Exit-record kinds (kernel → hypervisor) ─────────────────────────────
//
// A new vocabulary, deliberately separate from the session report-kind
// namespace (`session::*_REPORT_KIND`, the `0xFFxx` family): these are
// ring records, not bind-inert session reports.  Distinctness is proven
// at compile time below, the report-roster discipline.

/// Guest touched an address the hypervisor models as a device.
/// **Producer-less, deliberately — still, after #765** (#481): device-
/// vs-RAM is knowledge only the hypervisor has (it chose the device
/// windows), so the kernel forwards [`EXIT_KIND_STAGE_FAULT`] *with the
/// decode-assist tail attached* and the hypervisor's device-window table
/// classifies.  The number stays reserved for a kernel-classified fast
/// path — a fault the kernel's own stage-2 ledger already places outside
/// guest RAM — should one ever pay for itself.
pub const EXIT_KIND_MMIO: u16 = 0x4701;
/// A programmed-I/O access trapped (x86-64's port `in`/`out` family;
/// an architecture without a port space never raises it): payload `a`
/// = port, `b` packs
/// write/size.
pub const EXIT_KIND_PORT_IO: u16 = 0x4702;
/// CPUID (or an intercepted ID-register read).  No payload.
pub const EXIT_KIND_CPUID: u16 = 0x4703;
/// Model-specific register access: `a` = index, `b` = write flag.
pub const EXIT_KIND_MSR: u16 = 0x4704;
/// Guest idled (HLT/WFI).  Idleness is hypervisor-serviced: only an
/// injected event can make the guest runnable again.  No payload.
pub const EXIT_KIND_HLT: u16 = 0x4705;
/// Stage-2 fault: `a` = GPA; `b[7:0]` = the access (the paging
/// `AccessFlags` wire value), `b[15:8]` = the fault qualifier — `1` the
/// translation was present and denied the access, `2` the access was the
/// CPU's own walk of the guest's page tables (an accessed/dirty update
/// against a read-only mapping reads as WRITE plus this bit), `4` a
/// reserved bit refused the walk.  The one fault the kernel forwards
/// (#481).  The record's decode-assist tail (module docs, [`assist`])
/// carries what the hardware knew about the trapped instruction, so a
/// device-window fault can be completed without a second exit.
pub const EXIT_KIND_STAGE_FAULT: u16 = 0x4706;
/// Arch-specific exit token the shared taxonomy does not name: `a` =
/// the raw token, decoded by the owning backend's vocabulary.
pub const EXIT_KIND_ARCH: u16 = 0x4707;
/// **Terminal**: the guest retired (shutdown, an unrecoverable
/// machine fault, a served
/// retire order, or a poisoned-window/refusal teardown).  The last
/// record a vCPU ever publishes (#480), doubling as the waiter-release
/// edge.  `a` carries a [`retire_reason`] discriminant, `b` a
/// [`retire_detail`] site code saying which refusal decided it.
pub const EXIT_KIND_RETIRED: u16 = 0x4708;
/// A control record was refused at the apply leg (#481): `a[7:0]` = the
/// zero-based index of the offending record within the drained batch,
/// `a[15:8]` = for a refused `WriteState`, the kernel's probe of *why*
/// (`[15:14]` the gate: 1 foreign scheduler owner, 2 no facility on this
/// CPU, 3 the driver refused; `[13:8]` the driver's last control-block
/// refusal, on Intel `[13:12]` kind, `[11:10]` first-dword class, `[9:8]`
/// op — zero when none), `b` = a refusal discriminant (e.g. double-inject
/// while the pending slot is occupied, or a resume before the exit ring
/// was drained).
/// Publish-then-park holds: the refusal *replaces* the consumed exit as
/// the one ordinary outstanding record.
pub const EXIT_KIND_OP_REFUSED: u16 = 0x4709;

/// The exit-kind roster; compile-time-distinct (below).
const EXIT_KINDS: [u16; 9] = [
    EXIT_KIND_MMIO,
    EXIT_KIND_PORT_IO,
    EXIT_KIND_CPUID,
    EXIT_KIND_MSR,
    EXIT_KIND_HLT,
    EXIT_KIND_STAGE_FAULT,
    EXIT_KIND_ARCH,
    EXIT_KIND_RETIRED,
    EXIT_KIND_OP_REFUSED,
];

// ── Control-record kinds (hypervisor → kernel) ──────────────────────────

/// Install the carried register image before the next entry.
///
/// Every completion today ships the **full** image back through this
/// record, even when only the instruction pointer and one datum register
/// changed (an MMIO or port completion).  A lighter "advance by `n`"
/// control kind is the obvious optimization once exit latency is measured
/// on metal (the rationale under [`GUEST_RECORD_BYTES`]); it takes the
/// next free `CTRL_KIND_*` number and never renumbers this one.
pub const CTRL_KIND_WRITE_STATE: u16 = 0x4801;
/// Queue an interrupt/exception: `a` packs vector/kind, `b` packs the
/// optional error code (see [`GuestCtrlRecord`] docs).
pub const CTRL_KIND_INJECT: u16 = 0x4802;
/// Re-enter the guest.  Explicit, so "resume unchanged" and "resume
/// after edits" are one shape and an empty wake is a protocol error —
/// and **checkable**: the vCPU refuses to re-enter until the exit ring's
/// consumer cursor has passed its own published tail
/// (resume-implies-drained, enforced not assumed — #481 §3).
pub const CTRL_KIND_RESUME: u16 = 0x4803;
/// Retire the guest — #480's retire order, on its concrete carrier.
pub const CTRL_KIND_RETIRE: u16 = 0x4804;

/// The control-kind roster; compile-time-distinct (below).
const CTRL_KINDS: [u16; 4] = [
    CTRL_KIND_WRITE_STATE,
    CTRL_KIND_INJECT,
    CTRL_KIND_RESUME,
    CTRL_KIND_RETIRE,
];

const _: () = {
    // Every kind distinct within its roster, and the two rosters disjoint
    // — a collision frozen into wire vocabulary is permanent (#524's
    // lesson, proven rather than reviewed-for).
    let mut i = 0;
    while i < EXIT_KINDS.len() {
        let mut j = i + 1;
        while j < EXIT_KINDS.len() {
            assert!(EXIT_KINDS[i] != EXIT_KINDS[j], "exit-kind collision");
            j += 1;
        }
        let mut k = 0;
        while k < CTRL_KINDS.len() {
            assert!(EXIT_KINDS[i] != CTRL_KINDS[k], "exit/ctrl kind collision");
            k += 1;
        }
        i += 1;
    }
    let mut i = 0;
    while i < CTRL_KINDS.len() {
        let mut j = i + 1;
        while j < CTRL_KINDS.len() {
            assert!(CTRL_KINDS[i] != CTRL_KINDS[j], "ctrl-kind collision");
            j += 1;
        }
        i += 1;
    }
};

/// The **decode-assist form** an exit record's tail carries — the
/// [`GuestExitRecord::assist`] discriminant.  Plain values, any-bit-valid
/// on read: an unknown form is "no help" to the consumer, never a trap.
///
/// The three shapes mirror what hardware actually provides (module docs):
/// nothing, the raw instruction bytes, or the access already decoded.  A
/// producing architecture fills the richest form it has; a consumer lifts
/// whichever arrives into one decoded access, running an ISA decoder only
/// over [`INSN`](Self::INSN).
pub mod assist {
    /// No assist: `insn`, `insn_len`, `access` and `data` are zero and
    /// meaningless.  The consumer may still refuse the fault fail-closed
    /// or fetch and decode by other means.
    pub const NONE: u8 = 0;
    /// The trapped instruction's raw bytes are in `insn[..insn_len]`
    /// (`1..=GUEST_INSN_BYTES_MAX`), fetched by the hardware (AMD decode
    /// assists) or by the kernel through the guest's own page tables
    /// (Intel).  `access` and `data` are zero.
    pub const INSN: u8 = 1;
    /// The access is already decoded in `access` (and `data`), by the
    /// hardware's own syndrome (`ESR_EL2.ISS` on Arm, `htinst` on
    /// RISC-V).  `insn` is zero.  **Reserved — no producer yet**: the
    /// packing below is fixed now so the first architecture that
    /// decodes in hardware fills a slot rather than reshaping the record.
    ///
    /// ```text
    /// access[3:0]    width of the memory access in bytes: 1, 2, 4, 8
    /// access[4]      1 = the guest wrote (data flows guest → device)
    /// access[5]      1 = a read sign-extends into the register
    /// access[15:8]   wire-image index of the datum register; 0xff = none
    /// access[23:16]  bytes of the register a read overwrites: 8 = all,
    ///                4 = the low dword with the upper cleared, 1 or 2 =
    ///                merged into the low bytes; zero when `[15:8]` is 0xff
    /// access[63:24]  reserved, zero
    /// data           the value written when no register carries it
    /// ```
    pub const ACCESS: u8 = 2;
}

/// Why a guest retired — the [`EXIT_KIND_RETIRED`] record's `a` payload.
/// A plain discriminant (any-bit-valid on read; unknown values are
/// "unspecified" to the consumer, never a trap).
pub mod retire_reason {
    /// The guest requested shutdown or hit an unrecoverable machine
    /// fault.
    pub const SHUTDOWN: u64 = 1;
    /// A hypervisor `Retire` control record was served.
    pub const ORDERED: u64 = 2;
    /// The window was poisoned (a `Full` exit push — the #487 §5
    /// protocol-violation witness) or the loop's own step refused
    /// (revoked authority, quarantined space).
    pub const POISONED: u64 = 3;
    /// A kernel **retire order** was served (#480 Phase R): the owner's
    /// authority teardown retired this guest — distinct from the
    /// hypervisor's own control-ring retire ([`ORDERED`]) and from the
    /// guest's self-termination ([`SHUTDOWN`]), so a waiter on the
    /// window can tell "I asked for this" from "the kernel revoked it".
    pub const REVOKED: u64 = 4;
}

/// Where a retire was decided — the [`EXIT_KIND_RETIRED`] record's `b`
/// payload.  `a` says *that* the loop retired; this says *which* refusal
/// it met, so a guest that ends on its own reads as more than "poisoned"
/// on the hypervisor's status line.  Plain discriminants, read like
/// [`retire_reason`]: unknown values are "unspecified", never a trap.
pub mod retire_detail {
    /// No further detail: an ordered or revoked retire, or a guest
    /// shutdown.
    pub const NONE: u64 = 0;
    /// The vCPU's private resume post could not be re-armed before entry.
    pub const RESUME_POST_LOST: u64 = 1;
    /// Parking on the standing resume post was refused.
    pub const PARK_REFUSED: u64 = 2;
    /// The exit ring refused the record (`Full`, or a window that no
    /// longer binds) — the #487 §5 poisoned-window rule.
    pub const EXIT_RING_FULL: u64 = 3;
    /// The apply leg found the control ring poisoned.
    pub const CONTROL_POISONED: u64 = 4;
    /// The exit ring's header no longer parsed at publish (magic, ABI,
    /// record size or capacity) — the window page was overwritten.
    pub const EXIT_RING_HEADER: u64 = 5;
    /// The vCPU no longer had a bound window page to publish into.
    pub const EXIT_RING_UNBOUND: u64 = 6;
    /// The guest register image could not be read for the record.
    ///
    /// **No longer produced by the publish path**, and kept so the
    /// vocabulary's numbering stays stable for a reader decoding an older
    /// word: the image is now captured in the stint that produced the exit,
    /// so a refused read fails the *step* instead ([`STEP_REFUSED`]) and the
    /// publish never touches the vCPU.
    pub const STATE_READ_REFUSED: u64 = 7;
    /// `8` is **vacant**: it was never assigned, and it stays so for the
    /// same reason [`STATE_READ_REFUSED`] is kept — a reader decoding an
    /// older word must never see a renumbered meaning.  A future detail
    /// takes the next free code, not this hole.
    /// The push failed but the ring had room when re-probed: a consumer
    /// cursor moved between the two — a producer race the SPSC contract
    /// forbids.
    pub const EXIT_RING_RACED: u64 = 9;
    /// The world switch itself was refused; the low nibble is one of
    /// [`step_refusal`]'s codes.
    pub const STEP_REFUSED: u64 = 0x10;

    /// Why a world switch was refused — `STEP_REFUSED | code`.  The
    /// first thirteen mirror the kernel's neutral virtualization error;
    /// the last two split its "invalid state" into the two gates a
    /// healthy loop can actually hit.
    pub mod step_refusal {
        /// No usable virtualization facility, or an unsupported operation.
        pub const UNSUPPORTED: u64 = 1;
        /// The facility is not online on the CPU the vCPU was placed on.
        pub const FACILITY_UNAVAILABLE: u64 = 2;
        /// The build has no driver for the requested mode.
        pub const FEATURE_UNAVAILABLE: u64 = 3;
        /// The current task carries no guest image.
        pub const INVALID_VCPU: u64 = 4;
        /// The guest-physical space is not valid for this facility.
        pub const INVALID_GUEST_SPACE: u64 = 5;
        /// A state gate refused that neither probe below explains.
        pub const INVALID_STATE: u64 = 6;
        /// The pending-event slot was occupied.
        pub const GUEST_BLOCKED: u64 = 7;
        /// The vCPU registry had no free slot.
        pub const REGISTRY_FULL: u64 = 8;
        /// The guest-space table had no free slot.
        pub const GUEST_SPACE_FULL: u64 = 9;
        /// Guest-physical address arithmetic overflowed.
        pub const ADDRESS_OVERFLOW: u64 = 10;
        /// A stage-2 shootdown round failed.
        pub const SHOOTDOWN_FAILED: u64 = 11;
        /// A vCPU kill is still pending on its owning CPU.
        pub const KILL_PENDING: u64 = 12;
        /// A stage-2 page-table operation failed.
        pub const PAGE_TABLE: u64 = 13;
        /// The vCPU task no longer holds a live share of the CPU it woke on.
        pub const NO_CPU_SHARE: u64 = 14;
        /// The guest's stage-2 space was quarantined underneath it.
        pub const SPACE_QUARANTINED: u64 = 15;
    }
}

/// Refusal discriminants — the [`EXIT_KIND_OP_REFUSED`] record's `b`
/// payload.
pub mod refusal {
    /// `Inject` while the single pending-event slot was occupied.
    pub const EVENT_SLOT_OCCUPIED: u64 = 1;
    /// `Resume` before the exit ring was drained (the head-check).
    pub const RESUME_BEFORE_DRAIN: u64 = 2;
    /// A record kind this kernel does not recognize.
    pub const UNKNOWN_KIND: u64 = 3;
    /// A `WriteState`/`Inject` payload the backend rejected.
    pub const BAD_PAYLOAD: u64 = 4;
}

/// Bytes of an exit record ahead of its reserved pad: the 24-byte head,
/// the register image, then the decode-assist tail (`insn` + `insn_len`
/// = 16, `access` + `data` = 16).
const GUEST_EXIT_RECORD_USED: usize = 24 + 8 * GUEST_STATE_QWORDS + 16 + 16;

/// Bytes of a control record ahead of its reserved pad: the 24-byte head
/// and the register image.
const GUEST_CTRL_RECORD_USED: usize = 24 + 8 * GUEST_STATE_QWORDS;

/// One **exit record** (kernel → hypervisor): the exit, the guest's full
/// register image, the producer's sequence, and the decode-assist tail —
/// everything the hypervisor needs, shipped once, so the `read_state`
/// round trip #484 measured is dead and an MMIO completion needs no
/// second exit.  Fixed 512-byte slot, `#[repr(C)]`, frozen.
#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, Immutable)]
pub struct GuestExitRecord {
    /// An `EXIT_KIND_*` discriminant.  Unknown kinds refuse at the
    /// consumer.
    pub kind: u16,
    /// Length in bytes of the instruction that caused the exit, when the
    /// backend exposes a decoded next-instruction length;
    /// `0` = not provided.  Filled for the intercepts whose length the
    /// hardware reports (port I/O on both x86 vendors); for a stage-2
    /// fault the decoder over the assist bytes is authoritative and this
    /// is at most a cross-check.
    pub instr_len: u8,
    /// The decode-assist form the tail carries — an [`assist`] value.
    pub assist: u8,
    /// The producer's free-running publish count — lets the consumer name
    /// a record (e.g. in an `OpRefused` conversation) and detect its own
    /// bookkeeping drift.
    pub sequence: u32,
    /// First payload word — meaning per kind (GPA, port, MSR index, raw
    /// arch token, retire reason, refused index).
    pub a: u64,
    /// Second payload word — meaning per kind (access bits, write/size
    /// packing, refusal discriminant).
    pub b: u64,
    /// The full register image in the architecture's wire order (see
    /// module docs); the tail past the architecture's image is zero.
    pub state: [u64; GUEST_STATE_QWORDS],
    /// [`assist::INSN`]: the trapped instruction's raw bytes, the first
    /// `insn_len` meaningful, the rest zero.  Zero for every other form.
    pub insn: [u8; GUEST_INSN_BYTES_MAX],
    /// [`assist::INSN`]: how many of `insn` are the instruction
    /// (`1..=GUEST_INSN_BYTES_MAX`).  Zero for every other form.
    pub insn_len: u8,
    /// [`assist::ACCESS`]: the decoded access word (packing in [`assist`]).
    /// Zero for every other form.
    pub access: u64,
    /// [`assist::ACCESS`]: the written value when no register carries it.
    /// Zero for every other form.
    pub data: u64,
    /// Reserved, zero on write, ignored on read.
    pub _pad: [u8; GUEST_RECORD_BYTES - GUEST_EXIT_RECORD_USED],
}

/// One **control record** (hypervisor → kernel).  Same frozen slot size;
/// the vCPU's run loop drains these in order at its safe point (#481).
///
/// `Inject` packing: `a = vector | (kind << 8)` where kind is
/// `0 = external, 1 = NMI, 2 = exception`; `b = error_code | (1 << 32)`
/// when an error code is present, `0` otherwise.  Anything else refuses
/// with [`refusal::BAD_PAYLOAD`].
#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, Immutable)]
pub struct GuestCtrlRecord {
    /// A `CTRL_KIND_*` discriminant.  Unknown kinds refuse (the record
    /// is skipped and reported, never silently dropped).
    pub kind: u16,
    /// Reserved, zero.
    pub _rsvd: [u8; 6],
    /// First payload word (see kind docs).
    pub a: u64,
    /// Second payload word (see kind docs).
    pub b: u64,
    /// The register image for `WriteState`; ignored otherwise.
    pub state: [u64; GUEST_STATE_QWORDS],
    /// Reserved, zero on write, ignored on read.
    pub _pad: [u8; GUEST_RECORD_BYTES - GUEST_CTRL_RECORD_USED],
}

const _: () = {
    assert!(size_of::<GuestExitRecord>() == GUEST_RECORD_BYTES);
    assert!(size_of::<GuestCtrlRecord>() == GUEST_RECORD_BYTES);
    assert!(align_of::<GuestExitRecord>() <= 64);
    assert!(align_of::<GuestCtrlRecord>() <= 64);
    // The frozen offsets a peer written against this layout relies on:
    // the head is 24 bytes on both records, the image follows it, and the
    // exit record's assist tail sits right behind the image, its two
    // words naturally aligned.
    assert!(core::mem::offset_of!(GuestExitRecord, state) == 24);
    assert!(core::mem::offset_of!(GuestCtrlRecord, state) == 24);
    assert!(core::mem::offset_of!(GuestExitRecord, insn) == 24 + 8 * GUEST_STATE_QWORDS);
    assert!(core::mem::offset_of!(GuestExitRecord, insn_len) == 24 + 8 * GUEST_STATE_QWORDS + 15);
    assert!(core::mem::offset_of!(GuestExitRecord, access) == 24 + 8 * GUEST_STATE_QWORDS + 16);
    assert!(core::mem::offset_of!(GuestExitRecord, data) == 24 + 8 * GUEST_STATE_QWORDS + 24);
    assert!(GUEST_INSN_BYTES_MAX + 1 == 16);
};

/// Byte offset of the **exit ring** within the per-vCPU window.
pub const GUEST_WINDOW_EXIT_AT: usize = 0;

/// Bytes the exit ring occupies (`kore-ring`'s frozen geometry: 192-byte
/// header block + capacity × slot, rounded to the 64-byte layout unit).
/// Mirrored here as layout *fact* so the window offsets are one frozen
/// vocabulary; `kore-ring::spsc::region_bytes` remains the computing
/// authority and the kernel asserts the two agree at attach.
pub const GUEST_WINDOW_EXIT_BYTES: usize =
    192 + GUEST_EXIT_RING_CAPACITY as usize * GUEST_RECORD_BYTES;

/// Byte offset of the **control ring** within the per-vCPU window.
pub const GUEST_WINDOW_CTRL_AT: usize = GUEST_WINDOW_EXIT_AT + GUEST_WINDOW_EXIT_BYTES;

/// Bytes the control ring occupies.
pub const GUEST_WINDOW_CTRL_BYTES: usize =
    192 + GUEST_CTRL_RING_CAPACITY as usize * GUEST_RECORD_BYTES;

/// Total bytes of the per-vCPU window — one page, frozen.  Both rings
/// fit with room to spare; the tail is reserved.
pub const GUEST_WINDOW_BYTES: usize = 4096;

const _: () = {
    assert!(GUEST_WINDOW_EXIT_AT % 64 == 0);
    assert!(GUEST_WINDOW_CTRL_AT % 64 == 0);
    assert!(GUEST_WINDOW_CTRL_AT + GUEST_WINDOW_CTRL_BYTES <= GUEST_WINDOW_BYTES);
};
