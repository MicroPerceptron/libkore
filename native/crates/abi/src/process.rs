//! Bounded process arguments/results and foreground streams shared by a
//! launcher and child.
//!
//! This is the program-I/O counterpart of [`crate::task::BootBlock`]: one
//! page with a frozen, self-validating layout.  The launcher retains the root,
//! initializes its local mapping, transfers a fresh sublease-backed sibling
//! view to the child, then stops touching the page until the child has exited.
//! The child reads the argument tail and writes a
//! bounded output record. An optional immutable launcher-context prefix reuses
//! the output payload before START; the child may interpret it and then replace
//! it with output. No copied value in this byte plane confers capability
//! authority. An optional second page carries the live foreground stream as
//! two fixed SPSC ring regions: input launcher → child, ordered output events
//! child → launcher. The output-side [`ProcessOutputKind::Closed`] record is
//! the whole-stream terminal: it is legal only with no unanswered
//! [`ProcessOutputKind::InputReady`], after which neither direction is used and
//! the launcher may block on task lifecycle.
//!
//! The input-side [`ProcessInputKind::Interrupt`] record is the one launcher
//! record that is not input: it is legal at any time, answers no
//! `InputReady`, and asks the child to stop at its next safe point and exit
//! in order (the terminal's Ctrl-C). A launcher sends it once, then allows a
//! bounded grace before it revokes the task; a child that never reads its
//! input simply meets the revoke (`res/LIFECYCLE_HARDENING.md` §3).

use zerocopy::byteorder::little_endian::{U16, U32};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

/// Boot-role used by the generic process-I/O view.
///
/// Caller-defined roles begin at `USER_BASE`; reserving the first coordinate
/// here gives every launcher and executable one common spelling without
/// growing the kernel's privileged role vocabulary.
pub const PROCESS_IO_ROLE: u16 = crate::task::role::USER_BASE;
/// Process-I/O role instance.
pub const PROCESS_IO_INDEX: u16 = 0;

/// Boot-role used by the optional foreground-process stream view.
///
/// The stream is data only: the sibling view that carries it is the
/// authority, while records copied through it never confer authority.
pub const PROCESS_STDIO_ROLE: u16 = crate::task::role::USER_BASE;
/// Foreground-process stream role instance. Index zero remains the bounded
/// launch/result [`ProcessIo`] page.
pub const PROCESS_STDIO_INDEX: u16 = 1;

/// Boot-role used by the child-held receive end of the dedicated stdin wake
/// channel. The launcher retains the matching send and its channel root.
pub const PROCESS_STDIN_WAKE_ROLE: u16 = crate::task::role::USER_BASE;
/// Child boot-role instance for the stdin-wake `ChanRecv`.
pub const PROCESS_STDIN_WAKE_INDEX: u16 = 2;

/// Boot-role used by the child-held send end of the dedicated output wake
/// channel. The launcher retains the matching receive and channel root.
pub const PROCESS_OUTPUT_WAKE_ROLE: u16 = crate::task::role::USER_BASE;
/// Child boot-role instance for the output-wake `ChanSend`.
pub const PROCESS_OUTPUT_WAKE_INDEX: u16 = 3;

/// Boot-role used by a **stream host**: a task that holds the *parent* end of
/// another process's stream on its launcher's behalf (an emulator window,
/// `res/SHELL_TERMINAL.md`). The launcher formats the page and keeps the
/// lifecycle pivots (the child's task handle and the channel roots). The host
/// has no output receive: the child's output wake is a send into the host's
/// own executor doorbell, so every host park is one executor park.
pub const PROCESS_STDIO_HOST_ROLE: u16 = crate::task::role::USER_BASE;
/// Host boot-role instance for the stream view. Distinct from every child
/// index above, so one block can never carry both ends' meanings at once.
pub const PROCESS_STDIO_HOST_INDEX: u16 = 4;
/// Host boot-role instance for the **host channel** `ChanRecv`: everything
/// else the launcher hands the host arrives here as capabilities — first the
/// stdin-wake send that completes the parent end, then a delegated
/// `ACCESS | REVOKE` handle for each foreground task the host may end
/// (`res/SHELL_TERMINAL.md` §9).
pub const PROCESS_STDIO_HOST_CHANNEL_INDEX: u16 = 5;
/// Exact rights of a foreground slot: a delegated task handle that may
/// observe and end its task, and nothing else.
pub const FOREGROUND_SLOT_RIGHTS: crate::cap::Rights =
    crate::cap::Rights::ACCESS.union(crate::cap::Rights::REVOKE);

/// Exact authority delivered on either child-held stream-wake endpoint.
///
/// `ACCESS` is the only right needed to rendezvous. `REVOKE` lets task
/// teardown close the delivered holding, while deliberately omitting
/// `DERIVE` prevents a child from proliferating wake authority. The launcher
/// retains each channel root as the remote-lifecycle pivot.
pub const PROCESS_STDIO_WAKE_RIGHTS: crate::cap::Rights =
    crate::cap::Rights::ACCESS.union(crate::cap::Rights::REVOKE);

/// Notice cookie for a process-stream state change visible to the launcher on
/// the output wake channel.
/// The payload is the post-change directional sequence value and remains a
/// hint: the receiver drains and rechecks shared memory in every case.
pub const PROCESS_OUTPUT_WAKE_COOKIE: u64 = 0x6B72_5F70_6F75_7450; // "kr_poutP"
/// Notice cookie for a process-stream state change visible to the child on the
/// stdin wake channel.
pub const PROCESS_STDIN_WAKE_COOKIE: u64 = 0x6B72_5F70_696E_7043; // "kr_pinpC"

/// One page, matching the smallest portable kore mapping granule.
pub const PROCESS_STDIO_BYTES: usize = 4096;
/// Record capacity of each process-stream SPSC ring.
pub const PROCESS_STDIO_RING_CAPACITY: u32 = 32;
/// Frozen byte size of either ring region: the SPSC v1 192-byte control area
/// plus 32 32-byte records, rounded to a 64-byte layout unit.
///
/// `kore-abi` deliberately does not depend on `kore-ring`; this pin is checked
/// against `spsc::region_bytes` by `kore-rt` where the rings are formatted.
pub const PROCESS_STDIO_RING_BYTES: usize = 1216;
/// Byte offset of the launcher-to-child input ring.
pub const PROCESS_STDIN_RING_OFFSET: usize = 0;
/// Byte offset of the child-to-launcher output-event ring.
pub const PROCESS_OUTPUT_RING_OFFSET: usize = PROCESS_STDIN_RING_OFFSET + PROCESS_STDIO_RING_BYTES;
/// End of the two ring regions.
pub const PROCESS_STDIO_RINGS_BYTES: usize = PROCESS_OUTPUT_RING_OFFSET + PROCESS_STDIO_RING_BYTES;
/// Sequence incremented by the launcher after either publishing stdin or
/// consuming output. The child snapshots and rechecks it around its executor
/// park, closing the notify-before-park race.
pub const PROCESS_STDIO_TO_CHILD_SEQUENCE_OFFSET: usize = PROCESS_STDIO_RINGS_BYTES;
/// Child-owned waiting flag paired with
/// [`PROCESS_STDIO_TO_CHILD_SEQUENCE_OFFSET`]. The launcher sends on the
/// stdin-wake channel only after atomically observing this as set.
pub const PROCESS_STDIO_CHILD_WAITING_OFFSET: usize = PROCESS_STDIO_TO_CHILD_SEQUENCE_OFFSET + 4;
/// Sequence incremented by the child after either publishing output or
/// consuming stdin. It occupies its own cache line so the two hot publication
/// sequences never share a line. The parent-written waiting flag intentionally
/// joins this coordinated line and changes only across the parent's final
/// recheck and executor park.
pub const PROCESS_STDIO_TO_PARENT_SEQUENCE_OFFSET: usize =
    PROCESS_STDIO_TO_CHILD_SEQUENCE_OFFSET + 64;
/// Launcher-owned waiting flag paired with
/// [`PROCESS_STDIO_TO_PARENT_SEQUENCE_OFFSET`]. The child sends on the
/// output-wake channel only after atomically observing this as set.
pub const PROCESS_STDIO_PARENT_WAITING_OFFSET: usize = PROCESS_STDIO_TO_PARENT_SEQUENCE_OFFSET + 4;
/// End of the live process-stream layout. Each directional coordination tuple
/// occupies a complete 64-byte line, so the two hot publication sequences do
/// not share a line. The opposite-writer waiting flag intentionally joins its
/// sequence and is touched only across park transitions. The rest of the page
/// is reserved and zeroed when the launcher formats it.
pub const PROCESS_STDIO_LAYOUT_BYTES: usize = PROCESS_STDIO_TO_PARENT_SEQUENCE_OFFSET + 64;

/// Maximum bytes carried by one input record.
pub const PROCESS_STDIN_CHUNK_BYTES: usize = 24;
/// Maximum bytes carried by one stdout/stderr event.
pub const PROCESS_OUTPUT_CHUNK_BYTES: usize = 24;

/// Why a process-stream record could not be constructed or decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessStreamError {
    /// The supplied byte chunk does not fit in one record.
    PayloadTooLong,
    /// The record is not one of the canonical wire encodings.
    Invalid,
}

const PROCESS_INPUT_DATA: u16 = 1;
const PROCESS_INPUT_CLOSED: u16 = 2;
const PROCESS_INPUT_INTERRUPT: u16 = 3;

/// Decoded launcher-to-child input record kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessInputKind {
    /// A non-empty canonical-input byte chunk.
    Data,
    /// The launcher has closed the child's input stream.
    Closed,
    /// The launcher asks the child to stop at its next safe point: release
    /// what it holds and exit, or (a REPL) abandon the current line. Not
    /// input: legal at any time, including with no unanswered `InputReady`,
    /// and it answers none. A child reading a line when it arrives keeps that
    /// request outstanding, so it prompts again without a second
    /// `InputReady`, and if it exits instead it exits without `Closed`. The
    /// launcher revokes the task if the child leaves the record unread past
    /// its grace.
    Interrupt,
}

/// One launcher-to-child foreground-input record.
///
/// Every bit pattern is safe to copy out of a shared ring, but only the three
/// canonical encodings pass [`Self::validate`]. In particular, unused bytes
/// must remain zero so malformed peers cannot smuggle ambiguous state through
/// padding or an inactive payload suffix.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct ProcessInputChunk {
    tag: U16,
    len: U16,
    reserved: U32,
    payload: [u8; PROCESS_STDIN_CHUNK_BYTES],
}

impl ProcessInputChunk {
    /// Encode a non-empty input byte chunk.
    pub fn data(payload: &[u8]) -> Result<Self, ProcessStreamError> {
        if payload.len() > PROCESS_STDIN_CHUNK_BYTES {
            return Err(ProcessStreamError::PayloadTooLong);
        }
        if payload.is_empty() {
            return Err(ProcessStreamError::Invalid);
        }
        let mut record = Self {
            tag: U16::new(PROCESS_INPUT_DATA),
            len: U16::new(payload.len() as u16),
            reserved: U32::new(0),
            payload: [0; PROCESS_STDIN_CHUNK_BYTES],
        };
        record.payload[..payload.len()].copy_from_slice(payload);
        Ok(record)
    }

    /// Encode the single canonical input-closed record.
    pub const fn closed() -> Self {
        Self {
            tag: U16::new(PROCESS_INPUT_CLOSED),
            len: U16::new(0),
            reserved: U32::new(0),
            payload: [0; PROCESS_STDIN_CHUNK_BYTES],
        }
    }

    /// Encode the single canonical interrupt record.
    pub const fn interrupt() -> Self {
        Self {
            tag: U16::new(PROCESS_INPUT_INTERRUPT),
            len: U16::new(0),
            reserved: U32::new(0),
            payload: [0; PROCESS_STDIN_CHUNK_BYTES],
        }
    }

    /// Validate and decode this record's kind.
    pub fn kind(&self) -> Result<ProcessInputKind, ProcessStreamError> {
        if self.reserved.get() != 0 {
            return Err(ProcessStreamError::Invalid);
        }
        let len = usize::from(self.len.get());
        match self.tag.get() {
            PROCESS_INPUT_DATA
                if (1..=PROCESS_STDIN_CHUNK_BYTES).contains(&len)
                    && self.payload[len..].iter().all(|byte| *byte == 0) =>
            {
                Ok(ProcessInputKind::Data)
            }
            PROCESS_INPUT_CLOSED if len == 0 && self.payload.iter().all(|byte| *byte == 0) => {
                Ok(ProcessInputKind::Closed)
            }
            PROCESS_INPUT_INTERRUPT if len == 0 && self.payload.iter().all(|byte| *byte == 0) => {
                Ok(ProcessInputKind::Interrupt)
            }
            _ => Err(ProcessStreamError::Invalid),
        }
    }

    /// Whether this is one of the three canonical input encodings.
    pub fn validate(&self) -> bool {
        self.kind().is_ok()
    }

    /// Validate the record and return its live byte prefix. Closed and
    /// interrupt records return an empty slice; callers that distinguish them
    /// should inspect [`Self::kind`] first.
    pub fn bytes(&self) -> Result<&[u8], ProcessStreamError> {
        self.kind()?;
        Ok(&self.payload[..usize::from(self.len.get())])
    }
}

const PROCESS_OUTPUT_STDOUT: u16 = 1;
const PROCESS_OUTPUT_STDERR: u16 = 2;
const PROCESS_OUTPUT_INPUT_READY: u16 = 3;
const PROCESS_OUTPUT_CLOSED: u16 = 4;

/// Decoded child-to-launcher output event kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessOutputKind {
    /// A non-empty stdout byte chunk.
    Stdout,
    /// A non-empty stderr byte chunk.
    Stderr,
    /// The child is ready for another canonical input line.
    InputReady,
    /// The child has no unanswered input request, will consume no further
    /// input, and will publish no more output events. The launcher may stop
    /// servicing both rings and wait for terminal task lifecycle after this
    /// record.
    Closed,
}

/// The shared ordering law for one child-to-launcher output event.
///
/// `input_requested` means a preceding [`ProcessOutputKind::InputReady`] has
/// not yet been answered with one complete canonical input line. Output bytes
/// may still precede that answer, but another request would overlap it and
/// [`ProcessOutputKind::Closed`] cannot abandon it: `Closed` is the
/// bidirectional stream terminal, not merely an output-half marker. Once
/// `output_closed` is true every later event is invalid.
pub const fn process_output_event_is_lawful(
    output_closed: bool,
    input_requested: bool,
    kind: ProcessOutputKind,
) -> bool {
    if output_closed {
        return false;
    }
    match kind {
        ProcessOutputKind::Stdout | ProcessOutputKind::Stderr => true,
        ProcessOutputKind::InputReady | ProcessOutputKind::Closed => !input_requested,
    }
}

/// One child-to-launcher foreground-output event.
///
/// `InputReady` is the ordering barrier that lets a launcher drain prompts
/// before soliciting the next terminal line. `Closed` is the in-band terminal
/// for both stream directions; task WAIT remains the authoritative lifecycle
/// witness and proves the shared mapping quiescent.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct ProcessOutputEvent {
    tag: U16,
    len: U16,
    reserved: U32,
    payload: [u8; PROCESS_OUTPUT_CHUNK_BYTES],
}

impl ProcessOutputEvent {
    fn bytes_event(tag: u16, payload: &[u8]) -> Result<Self, ProcessStreamError> {
        if payload.len() > PROCESS_OUTPUT_CHUNK_BYTES {
            return Err(ProcessStreamError::PayloadTooLong);
        }
        if payload.is_empty() {
            return Err(ProcessStreamError::Invalid);
        }
        let mut record = Self {
            tag: U16::new(tag),
            len: U16::new(payload.len() as u16),
            reserved: U32::new(0),
            payload: [0; PROCESS_OUTPUT_CHUNK_BYTES],
        };
        record.payload[..payload.len()].copy_from_slice(payload);
        Ok(record)
    }

    /// Encode a non-empty stdout chunk.
    pub fn stdout(payload: &[u8]) -> Result<Self, ProcessStreamError> {
        Self::bytes_event(PROCESS_OUTPUT_STDOUT, payload)
    }

    /// Encode a non-empty stderr chunk.
    pub fn stderr(payload: &[u8]) -> Result<Self, ProcessStreamError> {
        Self::bytes_event(PROCESS_OUTPUT_STDERR, payload)
    }

    /// Encode the child's readiness for another canonical input line.
    pub const fn input_ready() -> Self {
        Self {
            tag: U16::new(PROCESS_OUTPUT_INPUT_READY),
            len: U16::new(0),
            reserved: U32::new(0),
            payload: [0; PROCESS_OUTPUT_CHUNK_BYTES],
        }
    }

    /// Encode the in-band end of both process-stream directions. The child must
    /// have consumed the line for its latest `InputReady` before publishing it.
    pub const fn closed() -> Self {
        Self {
            tag: U16::new(PROCESS_OUTPUT_CLOSED),
            len: U16::new(0),
            reserved: U32::new(0),
            payload: [0; PROCESS_OUTPUT_CHUNK_BYTES],
        }
    }

    /// Validate and decode this event's kind.
    pub fn kind(&self) -> Result<ProcessOutputKind, ProcessStreamError> {
        if self.reserved.get() != 0 {
            return Err(ProcessStreamError::Invalid);
        }
        let len = usize::from(self.len.get());
        let suffix_is_zero =
            len <= PROCESS_OUTPUT_CHUNK_BYTES && self.payload[len..].iter().all(|byte| *byte == 0);
        match self.tag.get() {
            PROCESS_OUTPUT_STDOUT
                if (1..=PROCESS_OUTPUT_CHUNK_BYTES).contains(&len) && suffix_is_zero =>
            {
                Ok(ProcessOutputKind::Stdout)
            }
            PROCESS_OUTPUT_STDERR
                if (1..=PROCESS_OUTPUT_CHUNK_BYTES).contains(&len) && suffix_is_zero =>
            {
                Ok(ProcessOutputKind::Stderr)
            }
            PROCESS_OUTPUT_INPUT_READY
                if len == 0 && self.payload.iter().all(|byte| *byte == 0) =>
            {
                Ok(ProcessOutputKind::InputReady)
            }
            PROCESS_OUTPUT_CLOSED if len == 0 && self.payload.iter().all(|byte| *byte == 0) => {
                Ok(ProcessOutputKind::Closed)
            }
            _ => Err(ProcessStreamError::Invalid),
        }
    }

    /// Whether this is one of the four canonical event encodings.
    pub fn validate(&self) -> bool {
        self.kind().is_ok()
    }

    /// Validate the event and return its live byte prefix. Control events
    /// return an empty slice.
    pub fn bytes(&self) -> Result<&[u8], ProcessStreamError> {
        self.kind()?;
        Ok(&self.payload[..usize::from(self.len.get())])
    }
}

const _: () = {
    assert!(core::mem::size_of::<ProcessInputChunk>() == 32);
    assert!(core::mem::align_of::<ProcessInputChunk>() == 1);
    assert!(core::mem::size_of::<ProcessOutputEvent>() == 32);
    assert!(core::mem::align_of::<ProcessOutputEvent>() == 1);
    assert!(PROCESS_STDIN_RING_OFFSET % 64 == 0);
    assert!(PROCESS_OUTPUT_RING_OFFSET % 64 == 0);
    assert!(PROCESS_STDIO_TO_CHILD_SEQUENCE_OFFSET % 64 == 0);
    assert!(PROCESS_STDIO_TO_PARENT_SEQUENCE_OFFSET % 64 == 0);
    assert!(PROCESS_STDIO_CHILD_WAITING_OFFSET == PROCESS_STDIO_TO_CHILD_SEQUENCE_OFFSET + 4);
    assert!(PROCESS_STDIO_PARENT_WAITING_OFFSET == PROCESS_STDIO_TO_PARENT_SEQUENCE_OFFSET + 4);
    assert!(PROCESS_STDIO_LAYOUT_BYTES <= PROCESS_STDIO_BYTES);
};

/// One page, matching the smallest portable kore mapping granule.
pub const PROCESS_IO_BYTES: usize = 4096;
/// Maximum raw command-tail bytes.  Arguments remain opaque to the launcher;
/// each executable owns its own parser.
/// Argument bytes. **296, sized to what a launcher can compose**: the
/// shell's whole command line is 256 bytes (`LINE_CAP`, asserted against
/// this constant in the shell), the files app bounds a path word at 255, and
/// init's launches pass short constants. The 200 bytes above that which the
/// buffer once held are the page's reservoir: 64 went to the launcher's
/// [`NetSnapshot`](crate::monitor::NetSnapshot) when the link's and the
/// radio's facts joined the page, 24 more when the device's counters did,
/// and 112 when the scheduler timer's arm census joined the system snapshot
/// — 16 for the census itself and 96 so the widest bounded report could
/// grow its `timer arms` row without giving up another. Each joined the
/// page without taking a byte from command output, which every status view
/// already fills to within a few bytes at its maximum topology; 16 went to
/// the frame-class census and 24 to the DRAM stream table the same way, and
/// 16 bytes remain above the longest command line.
pub const PROCESS_ARG_BYTES: usize = 272;
/// Maximum immutable launcher context bytes. Context and child output reuse
/// the payload at opposite sides of the task-lifecycle handoff; they are never
/// simultaneously live.
///
/// So this bound is policy, not geometry: the shared payload is
/// [`PROCESS_OUTPUT_BYTES`] wide, and a context may use any of it without
/// growing the page or touching the wire. What sets the number is the
/// largest context anyone builds, which is the bootstrap command catalogue
/// `help` reads — [`MAX_COMMAND_SNAPSHOT_ENTRIES`](crate::command::MAX_COMMAND_SNAPSHOT_ENTRIES)
/// rows of name, usage and summary plus four framing bytes each. The
/// catalogue has 40 rows, including the delegated Lua worker. Its complete
/// snapshot no longer fits 2560 bytes. 3072 carries the current catalogue
/// and remains within the existing 3208-byte alternating payload; no page
/// layout or command-output capacity changes.
pub const PROCESS_CONTEXT_BYTES: usize = 3072;
const _: () = assert!(
    PROCESS_CONTEXT_BYTES <= PROCESS_OUTPUT_BYTES,
    "context borrows the output payload and cannot outgrow it"
);
/// Header bytes before the argument and alternating payload arrays: 8 fixed
/// (`abi`/`flags`/`arg_len`/`output_len`) + the embedded [`TelemetrySnapshot`]
/// (240 — the IOMMU and timer censuses plus the manifold's own state) + the launcher's
/// [`FilesystemSnapshot`] (236, including physical storage geometry, pressure,
/// health, bounded reclamation outcomes and the derived completion-latency
/// moments) + the launcher's [`NetSnapshot`](crate::monitor::NetSnapshot) (76:
/// the link's facts, the device's counters, and the bound radio's) + 16
/// reserved.
///
/// The header's historical ceiling came from the widest bounded status reply:
/// the former per-core `status sys <first-core>` report needed 3099 output bytes
/// (and `status input i2c` 3072), so with the 540-byte header of ABI 11 the
/// output had 33 bytes to give.  ABI 12 takes 20 of them for the filesystem's
/// latency moments; 13 remain, and the next byte past them costs that report
/// a row.  ABI 13's timer census therefore came out of the argument buffer,
/// not the output — and so did the 96 bytes its `status` row needs, which
/// takes the output to 3208 with 19 bytes under that report. The current
/// compact `status sys` leaves this output allocation ABI-stable. ABI 13's
/// frame-class census (16 bytes in the system snapshot) likewise comes out
/// of the argument buffer: 312 → 296, still clearing the shell's 256-byte
/// line and the files app's 255-byte path word, and the output stays 3208.
/// The DRAM stream table took 24 more the same way (296 → 272).
///
/// The page is a fixed [`PROCESS_IO_BYTES`], so this constant and
/// [`PROCESS_OUTPUT_BYTES`] move together: every byte the header gains is a
/// byte of command output given up, and `size_of::<ProcessIo>()` asserts the
/// two stay in step.
const HEADER_BYTES: usize = 616;
/// Maximum bytes a foreground command can return to its launcher.
pub const PROCESS_OUTPUT_BYTES: usize = PROCESS_IO_BYTES - HEADER_BYTES - PROCESS_ARG_BYTES;
/// Layout version.  Zeroed/uninitialized pages fail closed.
///
/// `8`: versions are burned, never reissued — `4` shipped the retired
/// action-array layout (its output at a different offset would validate
/// under a reused stamp and read shifted), `5` was the pre-context layout,
/// and `6` preceded the IOMMU census fields the federated
/// [`TelemetrySnapshot`] now embeds; `8` added the filesystem physical-storage
/// aggregate, including its health sample. `9` carries the filesystem's four
/// bounded-failure counters into the launch snapshot. `10` adds the launcher's
/// frame-service snapshot (the link's facts and the bound radio's); `11`
/// widens it by the device's counters, paid for out of the argument buffer
/// again. `12` carries the filesystem's derived completion-latency moments —
/// 20 bytes of the output's remaining slack — and gives the word held in
/// reserve beside them a meaning, the read-ahead hit share: an `11` reader
/// would take that word for zero-or-corrupt. Each change shifts an offset or
/// a meaning and therefore burns a version.  13: the system snapshot's
/// scheduler-timer arm census, paid from the argument buffer together with
/// the output bytes its status row needs, so no report lost a row.
pub const PROCESS_IO_ABI: u16 = 13;

/// One launch-time snapshot of exec's bounded process table.
///
/// Exec returns this in the already-required `Spawned` reply. The launcher
/// copies it into the process page before START, so the exec-owned portion
/// adds no request or trap and every executable receives the same generic
/// process context. Profiled telemetry may add provider round trips before
/// this reply is published.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct ProcessSnapshot {
    flags: U32,
    capacity: U32,
    live: U32,
    zombies: U32,
    retiring: U32,
    spawned: U32,
    compensated: U32,
    reclaimed: U32,
}

impl ProcessSnapshot {
    const AVAILABLE: u32 = 1;

    /// No snapshot was available. This is still a valid process-page value.
    pub const EMPTY: Self = Self {
        flags: U32::new(0),
        capacity: U32::new(0),
        live: U32::new(0),
        zombies: U32::new(0),
        retiring: U32::new(0),
        spawned: U32::new(0),
        compensated: U32::new(0),
        reclaimed: U32::new(0),
    };

    /// Construct an available snapshot from exec's native counters.
    pub const fn new(
        capacity: u32,
        live: u32,
        zombies: u32,
        retiring: u32,
        spawned: u32,
        compensated: u32,
        reclaimed: u32,
    ) -> Self {
        Self {
            flags: U32::new(Self::AVAILABLE),
            capacity: U32::new(capacity),
            live: U32::new(live),
            zombies: U32::new(zombies),
            retiring: U32::new(retiring),
            spawned: U32::new(spawned),
            compensated: U32::new(compensated),
            reclaimed: U32::new(reclaimed),
        }
    }

    /// Whether exec supplied the snapshot.
    pub fn available(self) -> bool {
        self.flags.get() == Self::AVAILABLE
    }

    /// Whether the wire value is one of the two defined postures.
    pub fn validate(self) -> bool {
        self.available()
            || (self.flags.get() == 0
                && self.capacity.get() == 0
                && self.live.get() == 0
                && self.zombies.get() == 0
                && self.retiring.get() == 0
                && self.spawned.get() == 0
                && self.compensated.get() == 0
                && self.reclaimed.get() == 0)
    }

    /// Maximum simultaneously occupied exec ledger rows.
    pub fn capacity(self) -> u32 {
        self.capacity.get()
    }
    /// Live rows at snapshot time, including this newly founded child.
    pub fn live(self) -> u32 {
        self.live.get()
    }
    /// Exited rows still awaiting external handle revocation.
    pub fn zombies(self) -> u32 {
        self.zombies.get()
    }
    /// Closed rows whose resource cleanup is retrying.
    pub fn retiring(self) -> u32 {
        self.retiring.get()
    }
    /// Total occupied rows.
    pub fn tasks(self) -> u32 {
        self.live()
            .saturating_add(self.zombies())
            .saturating_add(self.retiring())
    }
    /// Successful foundings since exec bound.
    pub fn spawned(self) -> u32 {
        self.spawned.get()
    }
    /// Failed or ownerless foundings cleaned up by compensation.
    pub fn compensated(self) -> u32 {
        self.compensated.get()
    }
    /// Rows reclaimed after their external owner revoked the handle.
    pub fn reclaimed(self) -> u32 {
        self.reclaimed.get()
    }
}

/// Why a process-I/O page could not be initialized or read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessIoError {
    ArgumentsTooLong,
    ContextTooLong,
    OutputTooLong,
    Invalid,
}

/// Launcher↔child process-I/O page.
///
/// The page is deliberately not a ring: its ownership alternates at the task
/// lifecycle boundary.  The launcher writes arguments before START, the child
/// writes output before EXIT, and the launcher's successful WAIT is the
/// completion witness before it reads output. Concurrent foreground traffic
/// belongs on the separate process-stream page described by
/// [`ProcessInputChunk`] and [`ProcessOutputEvent`].
#[repr(C, align(64))]
#[derive(FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct ProcessIo {
    abi: U16,
    flags: U16,
    arg_len: U16,
    output_len: U16,
    telemetry: crate::monitor::TelemetrySnapshot,
    /// The launcher's filesystem-service snapshot. Deliberately *not* part of
    /// [`TelemetrySnapshot`]: that record is what the monitor assembles and
    /// what its reply wire carries, while a filesystem face is held only by
    /// the launcher. Two producers, two records, one page.
    filesystem: crate::monitor::FilesystemSnapshot,
    /// The launcher's frame-service snapshot: what the row answered to
    /// `Query` and `Radio` at this launch. A third producer on the same
    /// page, for the same reason as the filesystem's: only the launcher
    /// holds the frame service's diagnostic face.
    net: crate::monitor::NetSnapshot,
    // `reserved[0]` was intentionally available for an additive ABI field.
    // It now carries the immutable launcher-context live length; the remaining
    // words stay reserved and must be zero.
    reserved: [U32; 4],
    args: [u8; PROCESS_ARG_BYTES],
    /// Alternating payload: launcher context before START, child output after
    /// execution. The task lifecycle is the exclusive-ownership boundary.
    payload: [u8; PROCESS_OUTPUT_BYTES],
}

impl ProcessIo {
    /// Initialize a freshly zeroed or retired page for one launch.
    pub fn initialize(
        &mut self,
        args: &[u8],
        telemetry: crate::monitor::TelemetrySnapshot,
    ) -> Result<(), ProcessIoError> {
        self.initialize_with_context(args, &[], telemetry)
    }

    /// Initialize the page with opaque arguments and one immutable, bounded
    /// launcher-context record. Ownership transfers to the child before either
    /// prefix becomes observable there.
    pub fn initialize_with_context(
        &mut self,
        args: &[u8],
        context: &[u8],
        telemetry: crate::monitor::TelemetrySnapshot,
    ) -> Result<(), ProcessIoError> {
        if args.len() > PROCESS_ARG_BYTES {
            return Err(ProcessIoError::ArgumentsTooLong);
        }
        if context.len() > PROCESS_CONTEXT_BYTES {
            return Err(ProcessIoError::ContextTooLong);
        }
        if !telemetry.validate() {
            return Err(ProcessIoError::Invalid);
        }
        self.abi.set(PROCESS_IO_ABI);
        self.flags.set(0);
        self.arg_len.set(args.len() as u16);
        self.output_len.set(0);
        self.telemetry = telemetry;
        self.filesystem = crate::monitor::FilesystemSnapshot::EMPTY;
        self.net = crate::monitor::NetSnapshot::EMPTY;
        self.reserved = [U32::new(0); 4];
        self.reserved[0].set(context.len() as u32);
        self.args.fill(0);
        self.args[..args.len()].copy_from_slice(args);
        self.payload.fill(0);
        self.payload[..context.len()].copy_from_slice(context);
        Ok(())
    }

    /// Validate the frozen header and both bounded live prefixes.
    pub fn validate(&self) -> bool {
        self.abi.get() == PROCESS_IO_ABI
            && self.flags.get() == 0
            && self.telemetry.validate()
            && self.filesystem.validate()
            && self.reserved[1..].iter().all(|v| v.get() == 0)
            && usize::from(self.arg_len.get()) <= PROCESS_ARG_BYTES
            && usize::try_from(self.reserved[0].get()).is_ok_and(|len| len <= PROCESS_CONTEXT_BYTES)
            && usize::from(self.output_len.get()) <= PROCESS_OUTPUT_BYTES
            && (self.reserved[0].get() == 0 || self.output_len.get() == 0)
    }

    /// Raw argument tail supplied by the launcher.
    pub fn args(&self) -> Result<&[u8], ProcessIoError> {
        if !self.validate() {
            return Err(ProcessIoError::Invalid);
        }
        Ok(&self.args[..usize::from(self.arg_len.get())])
    }

    /// Immutable launcher context for this invocation. It is data, never an
    /// authority selector; a program must still hold every capability needed
    /// for the operation it describes.
    pub fn context(&self) -> Result<&[u8], ProcessIoError> {
        if !self.validate() {
            return Err(ProcessIoError::Invalid);
        }
        Ok(&self.payload[..self.reserved[0].get() as usize])
    }

    /// Exec's process-table snapshot captured by this launch.
    pub fn snapshot(&self) -> Result<ProcessSnapshot, ProcessIoError> {
        if !self.validate() {
            return Err(ProcessIoError::Invalid);
        }
        Ok(self.telemetry.exec)
    }

    /// Full federated launch snapshot.
    pub fn telemetry(&self) -> Result<crate::monitor::TelemetrySnapshot, ProcessIoError> {
        if !self.validate() {
            return Err(ProcessIoError::Invalid);
        }
        Ok(self.telemetry)
    }

    /// The launcher's filesystem-service snapshot for this launch.
    pub fn filesystem(&self) -> Result<crate::monitor::FilesystemSnapshot, ProcessIoError> {
        if !self.validate() {
            return Err(ProcessIoError::Invalid);
        }
        Ok(self.filesystem)
    }

    /// Publish the filesystem snapshot before the child starts. Separate from
    /// [`Self::write_snapshot`] because the two records have different
    /// producers and either may be unavailable on its own.
    pub fn write_filesystem(
        &mut self,
        snapshot: crate::monitor::FilesystemSnapshot,
    ) -> Result<(), ProcessIoError> {
        if !self.validate() || !snapshot.validate() {
            return Err(ProcessIoError::Invalid);
        }
        self.filesystem = snapshot;
        Ok(())
    }

    /// The launcher's frame-service snapshot for this launch.
    pub fn net(&self) -> Result<crate::monitor::NetSnapshot, ProcessIoError> {
        if !self.validate() {
            return Err(ProcessIoError::Invalid);
        }
        Ok(self.net)
    }

    /// Publish the frame-service snapshot before the child starts. Its own
    /// producer and its own record, like the filesystem's; either may be
    /// unavailable on its own.
    pub fn write_net(
        &mut self,
        snapshot: crate::monitor::NetSnapshot,
    ) -> Result<(), ProcessIoError> {
        if !self.validate() || !snapshot.validate() {
            return Err(ProcessIoError::Invalid);
        }
        self.net = snapshot;
        Ok(())
    }

    /// Publish the `Spawned` reply's snapshot before the child starts.
    pub fn write_snapshot(
        &mut self,
        snapshot: crate::monitor::TelemetrySnapshot,
    ) -> Result<(), ProcessIoError> {
        if !self.validate() || !snapshot.validate() {
            return Err(ProcessIoError::Invalid);
        }
        self.telemetry = snapshot;
        Ok(())
    }

    /// Replace the bounded result bytes.
    pub fn write_output(&mut self, output: &[u8]) -> Result<(), ProcessIoError> {
        if !self.validate() {
            return Err(ProcessIoError::Invalid);
        }
        if output.len() > PROCESS_OUTPUT_BYTES {
            return Err(ProcessIoError::OutputTooLong);
        }
        self.payload.fill(0);
        self.payload[..output.len()].copy_from_slice(output);
        self.reserved[0].set(0);
        self.output_len.set(output.len() as u16);
        Ok(())
    }

    /// Result bytes published by the exited child.
    pub fn output(&self) -> Result<&[u8], ProcessIoError> {
        if !self.validate() {
            return Err(ProcessIoError::Invalid);
        }
        Ok(&self.payload[..usize::from(self.output_len.get())])
    }
}

const _: () = {
    assert!(core::mem::size_of::<ProcessIo>() == PROCESS_IO_BYTES);
    assert!(core::mem::align_of::<ProcessIo>() == 64);
};

#[cfg(test)]
mod tests {
    use super::*;
    use zerocopy::FromZeros;

    #[test]
    fn process_stream_layout_and_records_are_frozen() {
        assert_eq!(PROCESS_STDIO_ROLE, crate::task::role::USER_BASE);
        assert_eq!(PROCESS_STDIO_INDEX, 1);
        assert_eq!(PROCESS_STDIN_WAKE_ROLE, crate::task::role::USER_BASE);
        assert_eq!(PROCESS_STDIN_WAKE_INDEX, 2);
        assert_eq!(PROCESS_OUTPUT_WAKE_ROLE, crate::task::role::USER_BASE);
        assert_eq!(PROCESS_OUTPUT_WAKE_INDEX, 3);
        assert_eq!(
            PROCESS_STDIO_WAKE_RIGHTS,
            crate::cap::Rights::ACCESS.union(crate::cap::Rights::REVOKE)
        );
        assert!(!PROCESS_STDIO_WAKE_RIGHTS.contains(crate::cap::Rights::DERIVE));
        assert_eq!(PROCESS_STDIO_BYTES, 4096);
        assert_eq!(PROCESS_STDIO_RING_CAPACITY, 32);
        assert_eq!(PROCESS_STDIO_RING_BYTES, 1216);
        assert_eq!(PROCESS_STDIN_RING_OFFSET, 0);
        assert_eq!(PROCESS_OUTPUT_RING_OFFSET, 1216);
        assert_eq!(PROCESS_STDIO_RINGS_BYTES, 2432);
        assert_eq!(PROCESS_STDIO_TO_CHILD_SEQUENCE_OFFSET, 2432);
        assert_eq!(PROCESS_STDIO_CHILD_WAITING_OFFSET, 2436);
        assert_eq!(PROCESS_STDIO_TO_PARENT_SEQUENCE_OFFSET, 2496);
        assert_eq!(PROCESS_STDIO_PARENT_WAITING_OFFSET, 2500);
        assert_eq!(PROCESS_STDIO_LAYOUT_BYTES, 2560);
        assert_eq!(core::mem::size_of::<ProcessInputChunk>(), 32);
        assert_eq!(core::mem::size_of::<ProcessOutputEvent>(), 32);

        let input = ProcessInputChunk::data(b"print(6 * 7)\n").unwrap();
        assert!(input.validate());
        assert_eq!(input.kind(), Ok(ProcessInputKind::Data));
        assert_eq!(input.bytes(), Ok(b"print(6 * 7)\n".as_slice()));
        let input_closed = ProcessInputChunk::closed();
        assert_eq!(input_closed.kind(), Ok(ProcessInputKind::Closed));
        assert_eq!(input_closed.bytes(), Ok([].as_slice()));
        let interrupt = ProcessInputChunk::interrupt();
        assert_eq!(interrupt.kind(), Ok(ProcessInputKind::Interrupt));
        assert_eq!(interrupt.bytes(), Ok([].as_slice()));
        assert_ne!(interrupt, input_closed);

        let stdout = ProcessOutputEvent::stdout(b"42\n").unwrap();
        assert!(stdout.validate());
        assert_eq!(stdout.kind(), Ok(ProcessOutputKind::Stdout));
        assert_eq!(stdout.bytes(), Ok(b"42\n".as_slice()));
        let stderr = ProcessOutputEvent::stderr(b"syntax error\n").unwrap();
        assert_eq!(stderr.kind(), Ok(ProcessOutputKind::Stderr));
        assert_eq!(
            ProcessOutputEvent::input_ready().kind(),
            Ok(ProcessOutputKind::InputReady)
        );
        assert_eq!(
            ProcessOutputEvent::closed().kind(),
            Ok(ProcessOutputKind::Closed)
        );

        for kind in [ProcessOutputKind::Stdout, ProcessOutputKind::Stderr] {
            assert!(process_output_event_is_lawful(false, false, kind));
            assert!(process_output_event_is_lawful(false, true, kind));
            assert!(!process_output_event_is_lawful(true, false, kind));
        }
        assert!(process_output_event_is_lawful(
            false,
            false,
            ProcessOutputKind::InputReady
        ));
        assert!(!process_output_event_is_lawful(
            false,
            true,
            ProcessOutputKind::InputReady
        ));
        assert!(process_output_event_is_lawful(
            false,
            false,
            ProcessOutputKind::Closed
        ));
        assert!(!process_output_event_is_lawful(
            false,
            true,
            ProcessOutputKind::Closed
        ));
        assert!(!process_output_event_is_lawful(
            true,
            false,
            ProcessOutputKind::Closed
        ));

        assert_eq!(
            ProcessInputChunk::data(&[b'x'; PROCESS_STDIN_CHUNK_BYTES + 1]),
            Err(ProcessStreamError::PayloadTooLong)
        );
        assert_eq!(
            ProcessOutputEvent::stdout(&[b'x'; PROCESS_OUTPUT_CHUNK_BYTES + 1]),
            Err(ProcessStreamError::PayloadTooLong)
        );
    }

    #[test]
    fn malformed_process_stream_records_fail_closed() {
        let mut input = ProcessInputChunk::data(b"x").unwrap();
        input.reserved.set(1);
        assert_eq!(input.kind(), Err(ProcessStreamError::Invalid));
        assert_eq!(input.bytes(), Err(ProcessStreamError::Invalid));

        let mut input = ProcessInputChunk::data(b"x").unwrap();
        input.payload[1] = 0xff;
        assert_eq!(input.kind(), Err(ProcessStreamError::Invalid));

        let mut input = ProcessInputChunk::closed();
        input.tag.set(u16::MAX);
        assert_eq!(input.kind(), Err(ProcessStreamError::Invalid));

        // An interrupt carries nothing: a length or payload byte is a record
        // this launcher did not write.
        let mut input = ProcessInputChunk::interrupt();
        input.len.set(1);
        assert_eq!(input.kind(), Err(ProcessStreamError::Invalid));
        let mut input = ProcessInputChunk::interrupt();
        input.payload[0] = 3;
        assert_eq!(input.kind(), Err(ProcessStreamError::Invalid));

        let mut event = ProcessOutputEvent::stdout(b"x").unwrap();
        event.len.set(0);
        assert_eq!(event.kind(), Err(ProcessStreamError::Invalid));
        assert_eq!(event.bytes(), Err(ProcessStreamError::Invalid));

        let mut event = ProcessOutputEvent::input_ready();
        event.payload[0] = 1;
        assert_eq!(event.kind(), Err(ProcessStreamError::Invalid));

        let mut event = ProcessOutputEvent::closed();
        event.tag.set(u16::MAX);
        assert_eq!(event.kind(), Err(ProcessStreamError::Invalid));
    }

    #[test]
    fn process_page_round_trips_bounded_arguments_and_output() {
        let mut page = ProcessIo::new_zeroed();
        let snapshot = ProcessSnapshot::new(8, 3, 1, 0, 12, 1, 8);
        page.initialize_with_context(
            b"125 --verbose",
            b"inert task-local metadata",
            crate::monitor::TelemetrySnapshot::exec_only(snapshot),
        )
        .unwrap();
        assert_eq!(page.args().unwrap(), b"125 --verbose");
        assert_eq!(page.context().unwrap(), b"inert task-local metadata");
        assert_eq!(page.snapshot().unwrap(), snapshot);
        page.write_output(b"slept 125 ms\n").unwrap();
        assert_eq!(page.context().unwrap(), b"");
        assert_eq!(page.output().unwrap(), b"slept 125 ms\n");
        // Pinned: the page is a fixed PROCESS_IO_BYTES, so every byte the
        // header gains is a byte of command output given up. 3160 -> 3120 was
        // the 40-byte ManifoldSnapshot joining TelemetrySnapshot; 3120 -> 3132
        // is the launcher's 52-byte NetSnapshot joining the header while the
        // argument buffer gave back the 64 bytes it held above any launcher's
        // longest command line; the snapshot's 24-byte counters block came
        // out of the argument buffer the same way, so the output stayed put.
        // 3132 -> 3112 is the filesystem snapshot's 20-byte completion-latency
        // moments, leaving 13 bytes under the widest bounded report (`status
        // sys <first-core>` at 3099; `status input i2c` needs 3072). 3112 ->
        // 3208 at ABI 13 is the argument buffer's reservoir (424 -> 312, still
        // clearing the shell's 256-byte line and the files app's 255-byte
        // path word) paying both the timer census's 16 header bytes and the
        // 96 its `timer arms` row adds to that widest report, now 3189.
        // The frame-class census (16 header bytes) came out of the argument
        // buffer too (312 -> 296, then 296 -> 272 for the DRAM stream
        // table), so the output still holds at 3208.
        assert_eq!(PROCESS_OUTPUT_BYTES, 3208);
    }

    /// The two snapshots have different producers — the monitor federates
    /// The frame-service snapshot is the page's third producer: absent until
    /// the launcher publishes it, and the radio-only posture (#388 D1: a
    /// bound radio, no frame path) and the link-only posture (a wired NIC)
    /// both survive the round trip with the other half honestly zero.
    #[test]
    fn the_net_snapshot_publishes_independently_of_the_others() {
        use crate::monitor::NetSnapshot;
        use crate::svc::NetRadioWire;

        let mut page = ProcessIo::new_zeroed();
        page.initialize(b"net", crate::monitor::TelemetrySnapshot::EMPTY)
            .unwrap();
        assert_eq!(page.net().unwrap(), NetSnapshot::EMPTY);
        assert!(!page.net().unwrap().available());
        assert!(NetSnapshot::EMPTY.validate());

        let mut radio = NetRadioWire::EMPTY;
        radio.cut = 1;
        radio.country = *b"US";
        radio.flags = NetRadioWire::RADIO_SECURE_BOOT;
        let only_radio = NetSnapshot::new(None, Some(radio));
        assert!(only_radio.validate());
        page.write_net(only_radio).unwrap();
        let read = page.net().unwrap();
        assert!(read.available() && read.radio_present() && !read.link_info());
        assert_eq!(read.radio().country, *b"US");
        assert_eq!(read.mac(), [0; 6]);

        let only_link = NetSnapshot::new(
            Some(crate::monitor::NetLinkFacts {
                up: true,
                mbps: 1000,
                mac: [1, 2, 3, 4, 5, 6],
                counters: crate::svc::NetCountersWire::EMPTY,
            }),
            None,
        );
        assert!(only_link.validate());
        page.write_net(only_link).unwrap();
        let read = page.net().unwrap();
        assert!(read.link_info() && read.link_up() && !read.radio_present());
        assert_eq!((read.link_mbps(), read.mac()), (1000, [1, 2, 3, 4, 5, 6]));
        assert_eq!(read.radio(), NetRadioWire::EMPTY);

        // A malformed record is refused and changes nothing.
        assert!(page.write_net(NetSnapshot::lying_for_test()).is_err());
        assert_eq!(
            page.net().unwrap(),
            only_link,
            "a refused write changes nothing"
        );
        // And the filesystem record is untouched throughout.
        assert_eq!(
            page.filesystem().unwrap(),
            crate::monitor::FilesystemSnapshot::EMPTY
        );
    }

    /// system/memory/exec, the launcher captures the filesystem — so each
    /// must survive the other being absent.
    #[test]
    fn the_filesystem_snapshot_publishes_independently_of_telemetry() {
        use crate::monitor::{
            FilesystemHealthSnapshot, FilesystemSnapshot, FilesystemStorageSnapshot,
        };

        let mut page = ProcessIo::new_zeroed();
        page.initialize(b"fs", crate::monitor::TelemetrySnapshot::EMPTY)
            .unwrap();
        // Absent until the launcher publishes one, and absent is a posture,
        // not an error.
        assert_eq!(page.filesystem().unwrap(), FilesystemSnapshot::EMPTY);
        assert!(!page.filesystem().unwrap().available());

        let fs = FilesystemSnapshot::new(
            FilesystemSnapshot::DURABLE
                | FilesystemSnapshot::ROUTE_HELD
                | FilesystemSnapshot::CHANNEL_LOST,
            5,
            1,
            2,
            1,
            1,
            FilesystemSnapshot::LIFECYCLE_CURRENT,
            FilesystemSnapshot::LIFECYCLE_REBUILT,
            4,
            3,
            0,
            FilesystemStorageSnapshot::new(
                4096,
                1 << 30,
                512 << 20,
                16 << 20,
                256 << 10,
                false,
                true,
                true,
            )
            .with_health(FilesystemHealthSnapshot::new(
                0, 36, 98, 10, 3, 7, 123, 2, 0,
            )),
            7,
            [2, 3, 2, 0, 0],
            [9, 9, 0],
            [4, 3, 2, 1],
            crate::monitor::FilesystemLatencySnapshot::new(8, 39, 39 * 3_000, 20_000, 39 * 4, {
                let mut b = [0u32; crate::monitor::FilesystemLatencySnapshot::BUCKETS];
                b[1] = 38;
                b[3] = 1;
                b
            }),
            u64::from(u16::MAX) + 1,
            [3, 1],
        );
        page.write_filesystem(fs).unwrap();
        let read = page.filesystem().unwrap();
        assert!(read.available() && read.durable() && read.route_held() && read.channel_lost());
        assert_eq!(read.user_lifecycle(), FilesystemSnapshot::LIFECYCLE_CURRENT);
        assert_eq!(
            read.system_lifecycle(),
            FilesystemSnapshot::LIFECYCLE_REBUILT
        );
        assert_eq!(read.storage().physical_bytes(), 1 << 30);
        assert_eq!(read.storage().health().temperature_celsius(), 36);
        assert_eq!(read.commits(), 7);
        assert_eq!(read.readahead_permille(), Some(750));
        assert_eq!(FilesystemSnapshot::EMPTY.readahead_permille(), None);
        assert_eq!((read.parks(), read.fires(), read.spins()), (9, 9, 0));
        assert_eq!(
            (
                read.expired_waits(),
                read.slow_waits(),
                read.leaked_placements(),
                read.abandoned_reclaims(),
            ),
            (4, 3, 2, 1)
        );
        // The census the page used to drop, and the latency histogram, ride
        // the same publish.
        assert_eq!(read.max_outstanding(), 8);
        assert_eq!(
            read.in_flight(),
            u16::MAX,
            "a narrow counter pins, never wraps"
        );
        assert!(read.latency_measured());
        assert_eq!(read.latency_samples(), 39);
        assert_eq!(read.latency_mean_ns(), Some(3_000));
        assert_eq!(read.latency_max_ns(), 20_000);
        assert_eq!(read.mean_depth_milli(), Some(4_000));
        // The telemetry half is untouched by a filesystem publish.
        assert_eq!(
            page.telemetry().unwrap(),
            crate::monitor::TelemetrySnapshot::EMPTY
        );
        assert_eq!(page.args().unwrap(), b"fs");
    }

    #[test]
    fn malformed_lengths_fail_closed() {
        let mut page = ProcessIo::new_zeroed();
        page.initialize(b"", crate::monitor::TelemetrySnapshot::EMPTY)
            .unwrap();
        page.output_len.set(u16::MAX);
        assert_eq!(page.output(), Err(ProcessIoError::Invalid));
    }

    #[test]
    fn malformed_context_length_fails_closed() {
        let mut page = ProcessIo::new_zeroed();
        page.initialize(b"", crate::monitor::TelemetrySnapshot::EMPTY)
            .unwrap();
        page.reserved[0].set((PROCESS_CONTEXT_BYTES + 1) as u32);
        assert_eq!(page.context(), Err(ProcessIoError::Invalid));
    }
}
