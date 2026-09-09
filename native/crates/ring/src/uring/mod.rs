//! `uring` — kore's io_uring-shaped device data plane
//! (`res/USERLAND.md` §3.2; the `kore-abi` crate header pins T1 byte
//! compatibility).
//!
//! The API mirrors the familiar Rust io-uring shape over
//! [`kore_abi::uring`] wire records: submit typed operations, reap
//! completions, and correlate them by `user_data`. T1 pins byte
//! compatibility; a CQE carries bytes and indices, never a capability.
//!
//! ## Two kore adaptations
//!
//! - **Registered handles, not raw pointers.** Operations name a registered
//!   file ([`types::Fixed`]) and registered buffer ([`types::Buf`]); the ring
//!   carries no host pointer (`res/SYSCALL.md` I5).
//! - **`push` is safe.** Operands are typed handles and the receiving
//!   service/device must validate every untrusted record.
//!
//! ## Shared windows, not a kernel or mock engine
//!
//! [`IoUring`] attaches the client side of two caller-provided
//! [`spsc`](crate::spsc) windows: the submission producer and completion
//! consumer. It allocates nothing, owns no backing memory, executes no
//! operation, and manufactures no completion. The peer service or device
//! consumes SQEs and produces CQEs; the notification edge is a separately
//! possessed doorbell capability. This keeps authority in sessions and bytes
//! in rings, with no kernel RPC shortcut or production mock backend.

use kore_abi::uring::{IoUringCqe, IoUringSqe};

use crate::spsc::{self, Consumer, Producer, RingError};

pub mod opcode;
pub mod types;

/// A built submission entry — the `squeue::Entry` analog, wrapping one wire
/// [`IoUringSqe`]. Attach a correlation token with
/// [`user_data`](Entry::user_data).
#[derive(Clone, Copy, Debug)]
pub struct Entry(IoUringSqe);

impl Entry {
    /// Attach the opaque token echoed back on the matching completion.
    #[must_use]
    pub fn user_data(mut self, user_data: u64) -> Self {
        self.0.user_data = user_data;
        self
    }
}

/// A completion entry — the `cqueue::Entry` analog, wrapping one wire
/// [`IoUringCqe`].
#[derive(Clone, Copy, Debug)]
pub struct Cqe(IoUringCqe);

impl Cqe {
    /// The `user_data` of the submission this completes.
    pub fn user_data(&self) -> u64 {
        self.0.user_data
    }

    /// The operation result: `>= 0` is success (bytes transferred / slot),
    /// `< 0` is a negated error status.
    pub fn result(&self) -> i32 {
        self.0.res
    }

    /// Completion flags published by the peer.
    pub fn flags(&self) -> u32 {
        self.0.flags
    }
}

/// Why the client halves could not attach to their shared windows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AttachError {
    /// The submission window is not a valid SQE ring.
    Submission(RingError),
    /// The completion window is not a valid CQE ring.
    Completion(RingError),
    /// Both rings are valid but have different depths.
    MismatchedCapacity,
}

/// The submission queue was full. The entry was not enqueued.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Full;

/// The userland/client side of an io_uring pair over shared windows.
///
/// Construction only attaches existing, capability-arranged mappings. Ring
/// initialization belongs to the party that creates the shared leases; wake
/// delivery belongs to the caller's separately possessed doorbell endpoint.
pub struct IoUring<'a> {
    sq: Producer<'a, IoUringSqe>,
    cq: Consumer<'a, IoUringCqe>,
}

impl<'a> IoUring<'a> {
    /// Attach the submission producer and completion consumer to two
    /// initialized shared-window rings.
    pub fn attach(
        submission_window: &'a mut [u8],
        completion_window: &'a mut [u8],
    ) -> Result<Self, AttachError> {
        let sq = spsc::producer(submission_window).map_err(AttachError::Submission)?;
        let cq = spsc::consumer(completion_window).map_err(AttachError::Completion)?;
        Self::from_halves(sq, cq)
    }

    /// Assemble already-attached client halves.
    ///
    /// Useful when a service setup flow has attached each mapping separately;
    /// the equal-depth check prevents asymmetric queue geometry.
    pub fn from_halves(
        sq: Producer<'a, IoUringSqe>,
        cq: Consumer<'a, IoUringCqe>,
    ) -> Result<Self, AttachError> {
        if sq.capacity() != cq.capacity() {
            return Err(AttachError::MismatchedCapacity);
        }
        Ok(Self { sq, cq })
    }

    /// The common SQ/CQ depth.
    pub fn entries(&self) -> u32 {
        self.sq.capacity()
    }

    /// Borrow the submission queue to [`push`](Submission::push) entries.
    pub fn submission(&mut self) -> Submission<'_, 'a> {
        Submission {
            inner: &mut self.sq,
        }
    }

    /// Borrow the completion queue to drain entries.
    pub fn completion(&mut self) -> Completion<'_, 'a> {
        Completion {
            inner: &mut self.cq,
        }
    }
}

/// The submission-queue handle (`SubmissionQueue` analog).
pub struct Submission<'q, 'w> {
    inner: &'q mut Producer<'w, IoUringSqe>,
}

impl Submission<'_, '_> {
    /// Stage one built [`Entry`]. Safe because ring operands are registered
    /// handles, not raw pointers. Returns [`Full`] if no slot is free.
    pub fn push(&mut self, entry: &Entry) -> Result<(), Full> {
        self.inner.push(entry.0).map_err(|spsc::Full| Full)
    }

    /// Free submission slots at this instant.
    pub fn capacity_left(&self) -> usize {
        self.inner.capacity_left() as usize
    }
}

/// The completion-queue handle (`CompletionQueue` analog).
pub struct Completion<'q, 'w> {
    inner: &'q mut Consumer<'w, IoUringCqe>,
}

impl Iterator for Completion<'_, '_> {
    type Item = Cqe;

    fn next(&mut self) -> Option<Cqe> {
        self.inner.pop().map(Cqe)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[repr(C, align(64))]
    struct Window<const N: usize>([u8; N]);

    impl<const N: usize> Window<N> {
        fn new() -> Self {
            Self([0; N])
        }
    }

    const DEPTH: u32 = 4;
    const SQ_BYTES: usize = spsc::region_bytes::<IoUringSqe>(DEPTH);
    const CQ_BYTES: usize = spsc::region_bytes::<IoUringCqe>(DEPTH);

    fn complete(sqe: IoUringSqe, result: i32) -> IoUringCqe {
        IoUringCqe {
            user_data: sqe.user_data,
            res: result,
            flags: 0,
        }
    }

    #[test]
    fn shared_peer_round_trip_preserves_correlation() {
        let mut sq_window = Window::<SQ_BYTES>::new();
        let mut cq_window = Window::<CQ_BYTES>::new();
        let (client_sq, mut device_sq) = spsc::split(&mut sq_window.0, DEPTH).expect("SQ split");
        let (mut device_cq, client_cq) = spsc::split(&mut cq_window.0, DEPTH).expect("CQ split");
        let mut ring = IoUring::from_halves(client_sq, client_cq).expect("matching rings");

        let read = opcode::Read::new(types::Fixed(3), types::Buf(0), 1024)
            .offset(0)
            .build()
            .user_data(0x42);
        ring.submission().push(&read).expect("SQ has room");

        let sqe = device_sq.pop().expect("device receives submission");
        assert_eq!(sqe.user_data, 0x42);
        assert_eq!(sqe.len, 1024);
        device_cq.push(complete(sqe, 1024)).expect("CQ has room");

        let cqe = ring
            .completion()
            .next()
            .expect("client receives completion");
        assert_eq!(cqe.user_data(), 0x42);
        assert_eq!(cqe.result(), 1024);
        assert_eq!(cqe.flags(), 0);
    }

    #[test]
    fn submission_backpressure_tracks_peer_consumption() {
        let mut sq_window = Window::<SQ_BYTES>::new();
        let mut cq_window = Window::<CQ_BYTES>::new();
        let (client_sq, mut device_sq) = spsc::split(&mut sq_window.0, DEPTH).expect("SQ split");
        let (_device_cq, client_cq) = spsc::split(&mut cq_window.0, DEPTH).expect("CQ split");
        let mut ring = IoUring::from_halves(client_sq, client_cq).expect("matching rings");

        for cookie in 0..DEPTH as u64 {
            ring.submission()
                .push(&opcode::Nop::new().build().user_data(cookie))
                .expect("SQ slot");
        }
        assert_eq!(ring.submission().capacity_left(), 0);
        assert_eq!(
            ring.submission()
                .push(&opcode::Nop::new().build().user_data(99)),
            Err(Full)
        );

        assert!(device_sq.pop().is_some());
        assert_eq!(ring.submission().capacity_left(), 1);
        ring.submission()
            .push(&opcode::Nop::new().build().user_data(99))
            .expect("peer consumption frees one slot");
    }

    #[test]
    fn attach_validates_both_windows_and_equal_depth() {
        let mut raw_sq = Window::<SQ_BYTES>::new();
        let mut raw_cq = Window::<CQ_BYTES>::new();
        assert_eq!(
            IoUring::attach(&mut raw_sq.0, &mut raw_cq.0).err(),
            Some(AttachError::Submission(RingError::BadHeader))
        );

        spsc::init::<IoUringSqe>(&mut raw_sq.0, DEPTH).expect("SQ init");
        assert_eq!(
            IoUring::attach(&mut raw_sq.0, &mut raw_cq.0).err(),
            Some(AttachError::Completion(RingError::BadHeader))
        );

        const SMALL_DEPTH: u32 = 2;
        let mut small_cq = Window::<{ spsc::region_bytes::<IoUringCqe>(SMALL_DEPTH) }>::new();
        spsc::init::<IoUringCqe>(&mut small_cq.0, SMALL_DEPTH).expect("CQ init");
        assert_eq!(
            IoUring::attach(&mut raw_sq.0, &mut small_cq.0).err(),
            Some(AttachError::MismatchedCapacity)
        );
    }

    #[test]
    fn completion_order_and_cursor_wrap_come_from_shared_protocol() {
        let mut sq_window = Window::<SQ_BYTES>::new();
        let mut cq_window = Window::<CQ_BYTES>::new();
        let (client_sq, mut device_sq) = spsc::split(&mut sq_window.0, DEPTH).expect("SQ split");
        let (mut device_cq, client_cq) = spsc::split(&mut cq_window.0, DEPTH).expect("CQ split");
        let mut ring = IoUring::from_halves(client_sq, client_cq).expect("matching rings");

        for round in 0..10u64 {
            ring.submission()
                .push(&opcode::Nop::new().build().user_data(round))
                .expect("SQ push");
            let sqe = device_sq.pop().expect("SQ pop");
            device_cq.push(complete(sqe, 0)).expect("CQ push");
            assert_eq!(ring.completion().next().map(|c| c.user_data()), Some(round));
        }
        assert!(ring.completion().next().is_none());
    }
}
