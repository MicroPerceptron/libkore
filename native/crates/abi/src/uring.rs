//! io_uring-compatible **data-plane** wire contract (issue #148).
//!
//! This module is the userland-facing *I/O veneer*: the byte layouts of
//! Linux's `io_uring` submission/completion rings, so userland tooling —
//! drivers, the inference engine, the RPC server — ports to kore by
//! *mechanical substitution* rather than a rewrite. It realizes
//! `res/SYSCALL.md` §5 D2 tier 2 (the SQ/CQ completion ring, #72) as an
//! io_uring-shaped surface.
//!
//! # The plane split — this is the *data* plane only
//!
//! kore's **authority** path (`derive`/`revoke`/`rendezvous`) stays the
//! native transactional session in [`crate::session`] — synchronous,
//! all-or-nothing, `Σ_min`. `res/SYSCALL.md` §3.3 is explicit that an async
//! ring in the *authority* path is the "long IPC" L4 abandoned, and that
//! stands. What lives here is the *data/completion* path, where io_uring's
//! async, independent-completion semantics are faithful:
//!
//! - **A completion (CQE) carries a slot index and a status, never a
//!   capability** (the res/SYSCALL.md §3 mandate): caps land in stamped
//!   [arena slots](crate::cap::CapRecord) exactly as today; a CQE only says
//!   "slot *n* delivered".
//! - Nothing here touches `Σ_min`, `prepare`/`commit`, or the arena bind.
//!
//! # "Compatible" means *byte layout*, not *Linux binary* (#148 L2, not L3)
//!
//! These structs are **byte-identical** to `include/uapi/linux/io_uring.h`
//! on every kore target, so source written against `liburing` ports by
//! header-swap. kore does **not** accept a raw Linux `io_uring_setup(2)` /
//! `io_uring_enter(2)` binary (no fds, no POSIX, different syscall numbers);
//! fd→capability lowering lives in the userland POSIX-lite shim, above
//! kore-abi.
//!
//! # Layout discipline (why these differ from [`crate::session`])
//!
//! The session descriptor is [`Unaligned`](zerocopy::Unaligned) because the
//! kernel reads it at an arbitrary byte offset in the arena. The io_uring
//! ring is different: its geometry guarantees natural alignment (a
//! page-aligned base, 64-byte SQEs, 16-byte CQEs), and the io_uring ABI is
//! **native-endian** (plain `__u32`, not an explicitly-little-endian wire
//! field). So these types use plain aligned integers and are read with
//! alignment-checked [`zerocopy`] accessors — matching Linux exactly on the
//! (little-endian) kore targets. The C unions, which kore-abi cannot express
//! under `#![forbid(unsafe_code)]`, are flattened to their primary member
//! (byte-identical); each field's doc lists the aliases.

use bitflags::bitflags;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// A submission-queue entry — byte-compatible with `struct io_uring_sqe`
/// (64 bytes). Userland stages one of these per I/O; the device service or
/// userspace driver consumes and validates it, then posts an [`IoUringCqe`] on
/// completion.
///
/// Every C union is flattened to its primary member (kore-abi is
/// `#![forbid(unsafe_code)]`, so no Rust unions); the layout is identical.
/// The op-specific reinterpretation of a union field is the reader's job —
/// the SDK layer (#95) provides the per-opcode accessors.
///
/// `FromBytes` + `IntoBytes`: every field admits any bit pattern and the
/// `#[repr(C)]` layout has no implicit padding, so a ring peer reinterprets a
/// slot in place and userland serializes one without a hand-rolled encoder.
#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct IoUringSqe {
    /// The operation — an [`op`] code.
    pub opcode: u8,
    /// Per-submission flags ([`IoSqeFlags`]).
    pub flags: u8,
    /// I/O priority for the request.
    pub ioprio: u16,
    /// File descriptor to operate on. In kore's data plane this is a
    /// **fixed-file index** into the driver's registered device/channel
    /// caps (the arena is the fixed-file/registered-buffer table) — never a
    /// POSIX fd.
    pub fd: i32,
    /// Union `off` / `addr2` / `{cmd_op, __pad1}` — the operation's offset
    /// operand (file offset, second address, or command sub-op).
    pub off: u64,
    /// Union `addr` / `splice_off_in` / `{level, optname}` — the primary
    /// buffer/iovec address operand.
    pub addr: u64,
    /// Buffer size, or number of iovecs.
    pub len: u32,
    /// Union of every per-op flags word (`rw_flags`, `poll_events`,
    /// `timeout_flags`, `msg_flags`, …) — meaning is selected by [`opcode`].
    ///
    /// [`opcode`]: IoUringSqe::opcode
    pub op_flags: u32,
    /// Opaque correlation token echoed verbatim into the completion's
    /// [`IoUringCqe::user_data`]. This is how a caller matches a completion
    /// to its submission.
    pub user_data: u64,
    /// Union `buf_index` / `buf_group` — the registered-buffer index (or the
    /// provided-buffer group for `BUFFER_SELECT`).
    pub buf_index: u16,
    /// Credentials/personality index, if used.
    pub personality: u16,
    /// Union `splice_fd_in` / `file_index` / `optlen` / `{addr_len, …}` —
    /// the secondary fd/index operand.
    pub splice_fd_in: i32,
    /// Union tail, high word: `addr3` / `attr_ptr` / `optval` — a third
    /// address operand.
    pub addr3: u64,
    /// Union tail, low word: `__pad2[1]` / `attr_type_mask`. Reserved unless
    /// an opcode names it.
    pub addr3_pad: u64,
}

const _: () = {
    use core::mem::{align_of, offset_of, size_of};
    // Byte-for-byte with `struct io_uring_sqe` — the whole point of the
    // veneer. A wrong offset here silently breaks every ported program, so
    // these are machine-checked (the "copy from the spec, never guess" rule).
    assert!(size_of::<IoUringSqe>() == 64);
    assert!(align_of::<IoUringSqe>() == 8);
    assert!(offset_of!(IoUringSqe, opcode) == 0);
    assert!(offset_of!(IoUringSqe, flags) == 1);
    assert!(offset_of!(IoUringSqe, ioprio) == 2);
    assert!(offset_of!(IoUringSqe, fd) == 4);
    assert!(offset_of!(IoUringSqe, off) == 8);
    assert!(offset_of!(IoUringSqe, addr) == 16);
    assert!(offset_of!(IoUringSqe, len) == 24);
    assert!(offset_of!(IoUringSqe, op_flags) == 28);
    assert!(offset_of!(IoUringSqe, user_data) == 32);
    assert!(offset_of!(IoUringSqe, buf_index) == 40);
    assert!(offset_of!(IoUringSqe, personality) == 42);
    assert!(offset_of!(IoUringSqe, splice_fd_in) == 44);
    assert!(offset_of!(IoUringSqe, addr3) == 48);
    assert!(offset_of!(IoUringSqe, addr3_pad) == 56);
};

/// A completion-queue entry — byte-compatible with `struct io_uring_cqe`
/// (16 bytes). The device service or userspace driver is the single writer;
/// its client reads it.
///
/// **This is the whole "no capability on the ring" contract**: `res` and
/// `flags` carry a status and a slot index (per #72's
/// `(source kind, channel, slot, status)`), and `user_data` echoes the
/// submission's token. A delivered capability is written to a stamped arena
/// slot — the CQE only announces *that* slot `n` is ready.
///
/// The `IORING_SETUP_CQE32` 32-byte variant (a trailing `big_cqe[2]`) is a
/// future extension; this is the base 16-byte CQE.
#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct IoUringCqe {
    /// The submission's [`IoUringSqe::user_data`], echoed verbatim.
    pub user_data: u64,
    /// Operation result: `>= 0` on success (kore data-plane: the delivered
    /// slot index / byte count), `< 0` is a negated error status.
    pub res: i32,
    /// Completion flags (e.g. `IORING_CQE_F_MORE` for multishot).
    pub flags: u32,
}

const _: () = {
    use core::mem::{align_of, offset_of, size_of};
    assert!(size_of::<IoUringCqe>() == 16);
    assert!(align_of::<IoUringCqe>() == 8);
    assert!(offset_of!(IoUringCqe, user_data) == 0);
    assert!(offset_of!(IoUringCqe, res) == 8);
    assert!(offset_of!(IoUringCqe, flags) == 12);
};

/// The submission-ring field offsets — byte-compatible with
/// `struct io_sqring_offsets`. Reported by ring setup so userland can find
/// the `head`/`tail`/`array`/… words within the mmap'd SQ region.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct IoSqringOffsets {
    /// Offset of the ring head (consumer-updated).
    pub head: u32,
    /// Offset of the ring tail (producer index; userland-updated).
    pub tail: u32,
    /// Offset of the ring mask (`entries - 1`).
    pub ring_mask: u32,
    /// Offset of the entry count.
    pub ring_entries: u32,
    /// Offset of the ring flags word (`IORING_SQ_*`).
    pub flags: u32,
    /// Offset of the dropped-submission counter.
    pub dropped: u32,
    /// Offset of the SQE-index array (the indirection into the SQE table).
    pub array: u32,
    /// Reserved.
    pub resv1: u32,
    /// User address of the ring (with `IORING_SETUP_NO_MMAP`).
    pub user_addr: u64,
}

const _: () = {
    use core::mem::size_of;
    assert!(size_of::<IoSqringOffsets>() == 40);
};

/// The completion-ring field offsets — byte-compatible with
/// `struct io_cqring_offsets`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct IoCqringOffsets {
    /// Offset of the ring head (consumer index; userland-updated).
    pub head: u32,
    /// Offset of the ring tail (producer-updated).
    pub tail: u32,
    /// Offset of the ring mask (`entries - 1`).
    pub ring_mask: u32,
    /// Offset of the entry count.
    pub ring_entries: u32,
    /// Offset of the overflow counter.
    pub overflow: u32,
    /// Offset of the CQE array.
    pub cqes: u32,
    /// Offset of the ring flags word (`IORING_CQ_*`).
    pub flags: u32,
    /// Reserved.
    pub resv1: u32,
    /// User address of the ring (with `IORING_SETUP_NO_MMAP`).
    pub user_addr: u64,
}

const _: () = {
    use core::mem::size_of;
    assert!(size_of::<IoCqringOffsets>() == 40);
};

/// Ring setup parameters — byte-compatible with `struct io_uring_params`
/// (120 bytes). The in/out descriptor of a ring: userland requests the
/// geometry and flags; the pair's setup protocol fills the feature set and
/// ring offsets.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct IoUringParams {
    /// Requested submission-queue entry count (power of two).
    pub sq_entries: u32,
    /// Completion-queue entry count (with `IORING_SETUP_CQSIZE`).
    pub cq_entries: u32,
    /// Setup flags ([`IoSetupFlags`]).
    pub flags: u32,
    /// SQ-poll kernel-thread CPU (with `IORING_SETUP_SQ_AFF`).
    pub sq_thread_cpu: u32,
    /// SQ-poll idle timeout, milliseconds.
    pub sq_thread_idle: u32,
    /// Features the ring peer supports ([`IoFeatFlags`]) — setup-filled.
    pub features: u32,
    /// Work-queue fd to share (with `IORING_SETUP_ATTACH_WQ`).
    pub wq_fd: u32,
    /// Reserved.
    pub resv: [u32; 3],
    /// Submission-ring offsets — setup-filled.
    pub sq_off: IoSqringOffsets,
    /// Completion-ring offsets — setup-filled.
    pub cq_off: IoCqringOffsets,
}

const _: () = {
    use core::mem::{align_of, offset_of, size_of};
    assert!(size_of::<IoUringParams>() == 120);
    assert!(align_of::<IoUringParams>() == 8);
    assert!(offset_of!(IoUringParams, features) == 20);
    assert!(offset_of!(IoUringParams, sq_off) == 40);
    assert!(offset_of!(IoUringParams, cq_off) == 80);
};

bitflags! {
    /// Per-submission flags — the [`IoUringSqe::flags`] byte
    /// (`IOSQE_*`). `IO_LINK` / `IO_HARDLINK` (ordered chains) and `IO_DRAIN`
    /// (a barrier) are the closest io_uring has to a kore session's ordering
    /// — see #148 on why they are *not* a substitute for transactional
    /// atomicity.
    #[repr(transparent)]
    #[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
    pub struct IoSqeFlags: u8 {
        /// `fd` is a registered (fixed-file) index, not a raw descriptor.
        const FIXED_FILE = 1 << 0;
        /// Drain: issue only after all prior submissions complete (barrier).
        const IO_DRAIN = 1 << 1;
        /// Link: the next submission depends on this one (ordered chain,
        /// cancel-remaining-on-failure).
        const IO_LINK = 1 << 2;
        /// Like `IO_LINK`, but the chain is not severed by a short result.
        const IO_HARDLINK = 1 << 3;
        /// Hint that the operation will block; run it async from the start.
        const ASYNC = 1 << 4;
        /// Select the buffer from a provided-buffer group at completion.
        const BUFFER_SELECT = 1 << 5;
        /// Do not post a CQE on success (only on failure).
        const CQE_SKIP_SUCCESS = 1 << 6;
    }
}

bitflags! {
    /// Ring setup flags — [`IoUringParams::flags`] (`IORING_SETUP_*`).
    ///
    /// This is the Linux vocabulary, frozen for source-compatibility; kore
    /// honors a subset and refuses an unrecognized bit at setup. Many bits
    /// (`SQPOLL`, `COOP_TASKRUN`, `DEFER_TASKRUN`, …) name Linux
    /// task-running strategies that kore's data-plane ring does not
    /// replicate.
    #[repr(transparent)]
    #[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
    pub struct IoSetupFlags: u32 {
        /// Busy-poll for completions instead of interrupt-driven.
        const IOPOLL = 1 << 0;
        /// A kernel thread polls the submission queue (no `enter` to submit).
        const SQPOLL = 1 << 1;
        /// Pin the SQ-poll thread to `sq_thread_cpu`.
        const SQ_AFF = 1 << 2;
        /// `cq_entries` sizes the completion queue independently.
        const CQSIZE = 1 << 3;
        /// Clamp SQ/CQ sizes to the kernel maximum instead of failing.
        const CLAMP = 1 << 4;
        /// Share the async work-queue of the ring named by `wq_fd`.
        const ATTACH_WQ = 1 << 5;
        /// Start the ring disabled; enable later via register.
        const R_DISABLED = 1 << 6;
        /// Continue submitting after a submission errors.
        const SUBMIT_ALL = 1 << 7;
        /// Cooperative task-running (Linux scheduling detail).
        const COOP_TASKRUN = 1 << 8;
        /// Set a ring flag when task work is pending.
        const TASKRUN_FLAG = 1 << 9;
        /// 128-byte SQEs.
        const SQE128 = 1 << 10;
        /// 32-byte CQEs.
        const CQE32 = 1 << 11;
        /// Only a single task ever submits (an optimization hint).
        const SINGLE_ISSUER = 1 << 12;
        /// Defer task work to the next `enter(GETEVENTS)`.
        const DEFER_TASKRUN = 1 << 13;
        /// Userland provides the ring memory (no kernel mmap).
        const NO_MMAP = 1 << 14;
        /// The ring fd is usable only as a registered ring.
        const REGISTERED_FD_ONLY = 1 << 15;
        /// No SQE-index indirection array (submit in SQE order).
        const NO_SQARRAY = 1 << 16;
    }
}

bitflags! {
    /// `enter`-time flags — the argument to the ring's `enter` doorbell
    /// (`IORING_ENTER_*`). `GETEVENTS` with a non-zero min-complete is the
    /// blocking wait that rides kore's park-after-commit seam (#71).
    #[repr(transparent)]
    #[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
    pub struct IoEnterFlags: u32 {
        /// Wait for at least the requested number of completions.
        const GETEVENTS = 1 << 0;
        /// Wake the SQ-poll thread if it has gone idle.
        const SQ_WAKEUP = 1 << 1;
        /// Wait until the SQ has drained.
        const SQ_WAIT = 1 << 2;
        /// The argument is an extended `io_uring_getevents_arg`.
        const EXT_ARG = 1 << 3;
        /// The ring is named by a registered index, not an fd.
        const REGISTERED_RING = 1 << 4;
    }
}

bitflags! {
    /// Feature flags the kernel reports in [`IoUringParams::features`]
    /// (`IORING_FEAT_*`) — the capabilities userland can rely on.
    #[repr(transparent)]
    #[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
    pub struct IoFeatFlags: u32 {
        /// SQ and CQ rings share one mmap.
        const SINGLE_MMAP = 1 << 0;
        /// Completions are never dropped (the CQ cannot overflow silently).
        const NODROP = 1 << 1;
        /// Submission state is stable once `enter` returns.
        const SUBMIT_STABLE = 1 << 2;
        /// Read/write honor the current file position.
        const RW_CUR_POS = 1 << 3;
        /// Operations assume the submitter's credentials.
        const CUR_PERSONALITY = 1 << 4;
        /// Poll-driven fast path for pollable files.
        const FAST_POLL = 1 << 5;
        /// 32-bit poll event masks.
        const POLL_32BITS = 1 << 6;
    }
}

bitflags! {
    /// Per-op flags for a **timeout** submission — the [`IoUringSqe::op_flags`]
    /// word when [`opcode`](IoUringSqe::opcode) is [`op::TIMEOUT`],
    /// [`op::LINK_TIMEOUT`], or [`op::TIMEOUT_REMOVE`] (`IORING_TIMEOUT_*`,
    /// transcribed verbatim from `include/uapi/linux/io_uring.h`). They select
    /// absolute-vs-relative, the clock domain, and one-shot-vs-multishot.
    ///
    /// **kore operand divergence (I5).** Linux points the SQE's
    /// [`addr`](IoUringSqe::addr) at a `struct __kernel_timespec`; kore's ring
    /// carries **no host pointer** (`res/SYSCALL.md` I5), so the interval /
    /// deadline rides **inline** in `addr` as an unsigned **nanosecond** count —
    /// a relative `core::time::Duration`, or an absolute nanos-since-boot
    /// (monotonic) / nanos-since-UNIX-epoch (realtime) value when [`ABS`] is
    /// set. The flag *bits* are unchanged, so a ported program's timeout-flag
    /// constants still resolve against kore-abi; only the operand vehicle changes.
    /// The typed builder is [`crate::clock::Timeout`].
    ///
    /// [`ABS`]: IoTimeoutFlags::ABS
    #[repr(transparent)]
    #[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
    pub struct IoTimeoutFlags: u32 {
        /// The operand is an **absolute** deadline, not a relative duration.
        const ABS = 1 << 0;
        /// Update an armed timeout's expiration in place (`TIMEOUT_REMOVE`).
        const UPDATE = 1 << 1;
        /// Measure against the boot clock (counts across suspend) rather than
        /// `CLOCK_MONOTONIC`.
        const BOOTTIME = 1 << 2;
        /// Measure against the **wall clock** (`CLOCK_REALTIME`) — honors clock
        /// discipline; the absolute-[`UnixTime`](crate::clock::UnixTime) domain.
        const REALTIME = 1 << 3;
        /// Update a **linked** timeout's expiration in place (`TIMEOUT_REMOVE`).
        const LINK_UPDATE = 1 << 4;
        /// Report a fired timeout as success (`res = 0`) rather than `-ETIME`.
        const ETIME_SUCCESS = 1 << 5;
        /// Re-arm after each expiry, posting a fresh completion (with
        /// [`IoCqeFlags::MORE`] set) every period until cancelled — the
        /// **periodic** timer, pairing with
        /// [`TimerMode::Periodic`](crate::clock::TimerMode).
        const MULTISHOT = 1 << 6;
    }
}

bitflags! {
    /// Completion flags — the low bits of [`IoUringCqe::flags`]
    /// (`IORING_CQE_F_*`). The kernel is the single writer.
    #[repr(transparent)]
    #[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
    pub struct IoCqeFlags: u32 {
        /// The buffer id is in the upper 16 bits (a `BUFFER_SELECT` result).
        const BUFFER = 1 << 0;
        /// **More completions follow** for this submission — the multishot
        /// continuation bit. A [periodic timer](IoTimeoutFlags::MULTISHOT)
        /// posts each expiry with this set; the final (cancelled) one clears it.
        const MORE = 1 << 1;
        /// A zero-copy send's buffer is still in use (notification pending).
        const SOCK_NONEMPTY = 1 << 2;
        /// This is a zero-copy notification CQE.
        const NOTIF = 1 << 3;
    }
}

/// The `IORING_OP_*` opcode vocabulary — the [`IoUringSqe::opcode`] byte.
///
/// These are **Linux's opcode numbers**, transcribed verbatim so a ported
/// program's opcode resolves against kore-abi (single-source-of-truth). The
/// presence of a constant is *not* a claim that kore implements that op:
/// kore's supported data-plane subset and each op's mapping onto a kore
/// effect land with the ring engine (#72) and the design note
/// (`res/SYSCALL.md` D8, per #148). Unimplemented ops are refused with a
/// completion error, never silently accepted.
///
/// The authority verbs (`derive`/`revoke`/`rendezvous`) are deliberately
/// **absent**: they are not io_uring ops — they cross via the native
/// transactional session ([`crate::session`]). A future bridge opcode that
/// carries a native session descriptor by reference (#148) would be a
/// kore-specific addition in a reserved range, not one of these.
pub mod op {
    /// No-op — completes immediately; the porting/liveness smoke test.
    pub const NOP: u8 = 0;
    /// Vectored read.
    pub const READV: u8 = 1;
    /// Vectored write.
    pub const WRITEV: u8 = 2;
    /// File sync.
    pub const FSYNC: u8 = 3;
    /// Read into a registered buffer.
    pub const READ_FIXED: u8 = 4;
    /// Write from a registered buffer.
    pub const WRITE_FIXED: u8 = 5;
    /// Add a poll on a file.
    pub const POLL_ADD: u8 = 6;
    /// Remove a pending poll.
    pub const POLL_REMOVE: u8 = 7;
    /// Sync a file range.
    pub const SYNC_FILE_RANGE: u8 = 8;
    /// Send a message on a socket.
    pub const SENDMSG: u8 = 9;
    /// Receive a message from a socket.
    pub const RECVMSG: u8 = 10;
    /// Arm a timeout.
    pub const TIMEOUT: u8 = 11;
    /// Remove/update a timeout.
    pub const TIMEOUT_REMOVE: u8 = 12;
    /// Accept a connection.
    pub const ACCEPT: u8 = 13;
    /// Cancel an in-flight request.
    pub const ASYNC_CANCEL: u8 = 14;
    /// A timeout linked to the previous request.
    pub const LINK_TIMEOUT: u8 = 15;
    /// Connect a socket.
    pub const CONNECT: u8 = 16;
    /// Preallocate file space.
    pub const FALLOCATE: u8 = 17;
    /// Open a file relative to a directory.
    pub const OPENAT: u8 = 18;
    /// Close a descriptor.
    pub const CLOSE: u8 = 19;
    /// Update the registered-files set.
    pub const FILES_UPDATE: u8 = 20;
    /// Extended file status.
    pub const STATX: u8 = 21;
    /// Read (non-vectored).
    pub const READ: u8 = 22;
    /// Write (non-vectored).
    pub const WRITE: u8 = 23;
    /// File access advice.
    pub const FADVISE: u8 = 24;
    /// Memory access advice.
    pub const MADVISE: u8 = 25;
    /// Send on a socket.
    pub const SEND: u8 = 26;
    /// Receive on a socket.
    pub const RECV: u8 = 27;
    /// Open with `how` flags.
    pub const OPENAT2: u8 = 28;
    /// epoll control.
    pub const EPOLL_CTL: u8 = 29;
    /// Splice between descriptors.
    pub const SPLICE: u8 = 30;
    /// Provide buffers to a group.
    pub const PROVIDE_BUFFERS: u8 = 31;
    /// Remove provided buffers.
    pub const REMOVE_BUFFERS: u8 = 32;
    /// Duplicate a pipe's data.
    pub const TEE: u8 = 33;
    /// Shut down a socket.
    pub const SHUTDOWN: u8 = 34;
    /// Rename at.
    pub const RENAMEAT: u8 = 35;
    /// Unlink at.
    pub const UNLINKAT: u8 = 36;
    /// Make a directory at.
    pub const MKDIRAT: u8 = 37;
    /// Symlink at.
    pub const SYMLINKAT: u8 = 38;
    /// Hard-link at.
    pub const LINKAT: u8 = 39;
    /// Send a CQE to another ring.
    pub const MSG_RING: u8 = 40;
    /// Non-vectored read into a registered buffer (multishot-capable read).
    pub const READ_MULTISHOT: u8 = 49;
    /// Zero-copy send.
    pub const SEND_ZC: u8 = 47;
    /// Socket creation.
    pub const SOCKET: u8 = 45;
    /// A device/file-specific command passthrough (the NVMe-passthrough
    /// vehicle — a natural fit for a kore userland block driver).
    pub const URING_CMD: u8 = 46;
    /// One past the last Linux opcode at the header revision this was
    /// transcribed from. Values `>=` this are unassigned by Linux; a
    /// kore-specific bridge opcode (#148), if built, takes a reserved value
    /// well clear of the growing Linux range.
    pub const LINUX_LAST: u8 = 65;

    // ── kore extensions (the swap data path) ────────────────────────────────
    // The kore-reserved opcode band, `0x80..`, well clear of `LINUX_LAST`
    // (the note above): values a ported Linux program never emits, so the
    // ring stays a strict Linux superset.  The swap ops run the ring
    // *inverted* — the kernel pager submits, a userland `SwapBackend`
    // (the SwapBackend contract) consumes (the "device" role) and posts the CQE.

    /// Base of the kore extension opcode band.
    pub const KORE_BASE: u8 = 0x80;
    /// Page-out (the SwapBackend contract): persist a lessor's evicted pages to backing
    /// store; the swap token rides back in the CQE result.  A kore
    /// extension — not a Linux io_uring op.
    pub const SWAP_OUT: u8 = KORE_BASE;
    /// Page-in (the SwapBackend contract): restore a swap token's bytes into a fresh
    /// frame the lessor has shared in.
    pub const SWAP_IN: u8 = KORE_BASE + 1;
    /// **Peer copy** — a device-to-device bulk transfer between two *registered
    /// buffers* (shared windows), the data-plane primitive for the `MmioRegion`
    /// peer leaf (an NVMe controller-memory-buffer, an accelerator aperture).
    /// A kore extension — not a Linux io_uring op.
    ///
    /// The operands are registered indices, never host addresses (I5), and no
    /// capability rides the ring — the windows are arranged by lease
    /// beforehand and named here by index.  The field reuse is fixed:
    ///
    /// | `IoUringSqe` field | meaning for `PEER_COPY`                  |
    /// |--------------------|------------------------------------------|
    /// | `buf_index`        | **source** registered-buffer index       |
    /// | `splice_fd_in`     | **destination** registered-buffer index  |
    /// | `off`              | byte offset within the source buffer     |
    /// | `addr`             | byte offset within the destination buffer|
    /// | `len`              | bytes to copy                            |
    /// | `user_data`        | correlation token, echoed in the CQE     |
    ///
    /// The CQE result is the bytes transferred (`>= 0`) or a negated status,
    /// exactly as every other completion — it carries bytes and indices, never
    /// a capability. The windows carry asymmetric authority established
    /// before registration: the source must be READ-admitted (including a
    /// proven completion route), while the destination must admit WRITE. A
    /// write-only destination may use the posted-write fabric proof; a
    /// read-write destination necessarily passed the stronger read proof. The
    /// SQE carries indices only and cannot supply a missing topology proof.
    pub const PEER_COPY: u8 = KORE_BASE + 2;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The wire sizes are the ABI: if any of these move, a ported program's
    /// ring math breaks. (The `const _` asserts above already fail the build
    /// on a bad offset; this is the runtime-visible restatement, the
    /// `session.rs` test shape.)
    #[test]
    fn wire_sizes_match_linux() {
        assert_eq!(core::mem::size_of::<IoUringSqe>(), 64);
        assert_eq!(core::mem::size_of::<IoUringCqe>(), 16);
        assert_eq!(core::mem::size_of::<IoSqringOffsets>(), 40);
        assert_eq!(core::mem::size_of::<IoCqringOffsets>(), 40);
        assert_eq!(core::mem::size_of::<IoUringParams>(), 120);
    }

    /// An SQE round-trips through bytes: build one, serialize, reinterpret,
    /// and read the correlation token + operands back. This exercises the
    /// `FromBytes`/`IntoBytes` contract the kernel ring relies on.
    #[test]
    fn sqe_round_trips_through_bytes() {
        let sqe = IoUringSqe {
            opcode: op::READ,
            flags: IoSqeFlags::FIXED_FILE.bits(),
            ioprio: 0,
            fd: 3,
            off: 0x1000,
            addr: 0xdead_beef,
            len: 512,
            op_flags: 0,
            user_data: 0xabcd_1234_5678_9f00,
            buf_index: 7,
            personality: 0,
            splice_fd_in: 0,
            addr3: 0,
            addr3_pad: 0,
        };

        let bytes = sqe.as_bytes();
        assert_eq!(bytes.len(), 64);
        // user_data sits at offset 32 — a ported reaper reads it there.
        assert_eq!(&bytes[32..40], &0xabcd_1234_5678_9f00u64.to_le_bytes());

        let (back, _) = IoUringSqe::read_from_prefix(bytes).expect("64 bytes reinterpret");
        assert_eq!(back.user_data, sqe.user_data);
        assert_eq!(back.opcode, op::READ);
        assert_eq!(back.fd, 3);
        assert_eq!(back.len, 512);
        assert_eq!(back.buf_index, 7);
    }

    /// A CQE carries a status + correlation token and nothing capability-
    /// shaped — the "no cap on the ring" invariant is structural: there is
    /// no field wide enough for a [`crate::cap::CapRecord`], by design.
    #[test]
    fn cqe_carries_status_not_capability() {
        let cqe = IoUringCqe {
            user_data: 0x99,
            res: 512,
            flags: 0,
        };
        let bytes = cqe.as_bytes();
        assert_eq!(bytes.len(), 16);
        let (back, _) = IoUringCqe::read_from_prefix(bytes).expect("16 bytes reinterpret");
        assert_eq!(back.user_data, 0x99);
        assert_eq!(back.res, 512);
    }

    /// The submission-flag bits are the Linux values (a wrong bit silently
    /// changes the meaning of a ported submission).
    #[test]
    fn sqe_flag_bits_match_linux() {
        assert_eq!(IoSqeFlags::FIXED_FILE.bits(), 1 << 0);
        assert_eq!(IoSqeFlags::IO_LINK.bits(), 1 << 2);
        assert_eq!(IoSqeFlags::IO_HARDLINK.bits(), 1 << 3);
        assert_eq!(IoSqeFlags::CQE_SKIP_SUCCESS.bits(), 1 << 6);
        assert_eq!(IoEnterFlags::GETEVENTS.bits(), 1 << 0);
    }

    /// The timeout + completion flag bits are the Linux `IORING_TIMEOUT_*` /
    /// `IORING_CQE_F_*` values (a wrong bit silently changes a ported timeout's
    /// clock domain or a multishot's continuation signal).
    #[test]
    fn timeout_and_cqe_flag_bits_match_linux() {
        assert_eq!(IoTimeoutFlags::ABS.bits(), 1 << 0);
        assert_eq!(IoTimeoutFlags::UPDATE.bits(), 1 << 1);
        assert_eq!(IoTimeoutFlags::BOOTTIME.bits(), 1 << 2);
        assert_eq!(IoTimeoutFlags::REALTIME.bits(), 1 << 3);
        assert_eq!(IoTimeoutFlags::LINK_UPDATE.bits(), 1 << 4);
        assert_eq!(IoTimeoutFlags::ETIME_SUCCESS.bits(), 1 << 5);
        assert_eq!(IoTimeoutFlags::MULTISHOT.bits(), 1 << 6);
        assert_eq!(IoCqeFlags::BUFFER.bits(), 1 << 0);
        assert_eq!(IoCqeFlags::MORE.bits(), 1 << 1);
        assert_eq!(IoCqeFlags::NOTIF.bits(), 1 << 3);
    }
}
