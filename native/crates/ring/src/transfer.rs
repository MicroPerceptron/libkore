//! `transfer` — bandwidth- and latency-optimized peer transfers over the
//! [`uring`](crate::uring) data plane.
//!
//! A [`PeerTransfer`] chunks one large window-to-window copy into
//! [`PEER_COPY`](kore_abi::uring::op::PEER_COPY) submissions across two
//! registered device-leaf windows (the `MmioRegion` peer leaf — an NVMe
//! controller-memory-buffer, an accelerator aperture — reached zero-copy
//! through proven fabric paths). The source window is READ-admitted with a
//! completion route; the destination admits WRITE, optionally through the
//! posted-write-only proof. It is the ergonomic layer over
//! [`opcode::PeerCopy`](crate::uring::opcode::PeerCopy):
//!
//! - **Bandwidth**: [`pump`](PeerTransfer::pump) stages as many chunks as the
//!   submission queue will hold, so the caller rings the doorbell **once** per
//!   pump and amortizes the wake edge over the whole batch. A large transfer
//!   drains at line rate with one kick per SQ-full, not one per chunk.
//! - **Latency**: [`single`](PeerTransfer::single) issues the whole transfer
//!   as one SQE for a copy that fits, so the caller pushes and kicks
//!   immediately — lowest time-to-first-byte, no batching.
//!
//! ## What this layer does *not* do (the doctrine boundary)
//!
//! It stages submissions and accounts progress; it owns **no** ring memory,
//! rings **no** doorbell, and reaps **no** completions. The window leases, the
//! [`IoUring`](crate::uring::IoUring), the doorbell kick, and the completion
//! reap loop all stay with the caller — a ring is a data structure in a shared
//! window, never a kernel object, and the kick is the caller's, never a ring
//! op. No capability rides the ring: operands are registered-buffer indices,
//! and completions are demultiplexed to transfers by `user_data`. This keeps
//! the transfer framework a pure composition of the already-landed verbs
//! (lease + doorbell + peer-routability admission).
//!
//! ## Correlation
//!
//! Each staged chunk carries `user_data = cookie_base + chunk_index`. The
//! caller sets a distinct [`cookie_base`](PeerTransfer::cookie_base) per
//! in-flight transfer, reaps CQEs from its ring, maps each `user_data` back to
//! its transfer, and feeds the bytes to [`credit`](PeerTransfer::credit).

use crate::uring::opcode::PeerCopy;
use crate::uring::types::Buf;
use crate::uring::{Entry, Full, Submission};

/// A bulk peer transfer: `len` bytes from registered buffer `src` (at
/// `src_base`) to registered buffer `dst` (at `dst_base`), chunked into
/// `PEER_COPY` submissions.
///
/// Cheap and `Copy`; holds only cursors, no borrows and no ring memory. Build
/// with [`new`](Self::new) plus the `with_*`/offset setters, then drive it
/// against a caller-owned [`Submission`] with [`pump`](Self::pump), crediting
/// completions with [`credit`](Self::credit) until [`is_done`](Self::is_done).
#[derive(Clone, Copy, Debug)]
pub struct PeerTransfer {
    src: Buf,
    dst: Buf,
    src_base: u64,
    dst_base: u64,
    len: u64,
    chunk: u32,
    cookie_base: u64,
    queued: u64,
    completed: u64,
    chunks_queued: u64,
}

impl PeerTransfer {
    /// Default per-SQE chunk (256 KiB) — a reasonable DMA burst. Tune to the
    /// device's optimal transfer size with [`with_chunk`](Self::with_chunk).
    pub const DEFAULT_CHUNK: u32 = 1 << 18;

    /// A transfer of `len` bytes from `src` to `dst`, both at offset 0, in
    /// [`DEFAULT_CHUNK`](Self::DEFAULT_CHUNK)-sized pieces.
    pub fn new(src: Buf, dst: Buf, len: u64) -> Self {
        Self {
            src,
            dst,
            src_base: 0,
            dst_base: 0,
            len,
            chunk: Self::DEFAULT_CHUNK,
            cookie_base: 0,
            queued: 0,
            completed: 0,
            chunks_queued: 0,
        }
    }

    /// Byte offset the copy starts at within the **source** buffer (default 0).
    #[must_use]
    pub fn src_base(mut self, offset: u64) -> Self {
        self.src_base = offset;
        self
    }

    /// Byte offset the copy starts at within the **destination** buffer
    /// (default 0).
    #[must_use]
    pub fn dst_base(mut self, offset: u64) -> Self {
        self.dst_base = offset;
        self
    }

    /// Maximum bytes per submitted chunk. Clamped to at least 1 so a zero can
    /// never stall [`pump`](Self::pump).
    #[must_use]
    pub fn with_chunk(mut self, chunk: u32) -> Self {
        self.chunk = chunk.max(1);
        self
    }

    /// The base correlation token; chunk *i* is submitted with
    /// `cookie_base + i`. Set a distinct base per in-flight transfer so its
    /// completions demultiplex cleanly.
    #[must_use]
    pub fn cookie_base(mut self, base: u64) -> Self {
        self.cookie_base = base;
        self
    }

    /// Total bytes this transfer moves.
    pub fn total(&self) -> u64 {
        self.len
    }

    /// Bytes handed to the submission queue so far.
    pub fn queued(&self) -> u64 {
        self.queued
    }

    /// Bytes the peer has acknowledged complete (via [`credit`](Self::credit)).
    pub fn completed(&self) -> u64 {
        self.completed
    }

    /// Bytes not yet staged into the submission queue.
    pub fn unqueued(&self) -> u64 {
        self.len - self.queued
    }

    /// Every byte has been staged into the SQ (though not necessarily
    /// completed).
    pub fn is_fully_queued(&self) -> bool {
        self.queued >= self.len
    }

    /// Every byte has been acknowledged complete.
    pub fn is_done(&self) -> bool {
        self.completed >= self.len
    }

    /// **Bandwidth path.** Stage as many chunks as the submission queue will
    /// hold, advancing the cursor, and return how many SQEs were staged. The
    /// caller rings the doorbell **once** after a pump — the kick is the
    /// caller's, never a ring op — so one wake edge covers the whole batch.
    ///
    /// Drive to completion by pumping whenever the SQ has room (after the peer
    /// drains it), crediting reaped completions, until [`is_done`](Self::is_done).
    /// Safe and allocation-free; `push`'s own [`Full`] is the single authority
    /// on room, so no stale capacity estimate can over- or under-fill.
    pub fn pump(&mut self, sq: &mut Submission<'_, '_>) -> usize {
        let mut staged = 0;
        while !self.is_fully_queued() {
            let off = self.queued;
            // The tail chunk is the remainder; every other is `chunk`.
            let this = core::cmp::min(self.chunk as u64, self.len - off) as u32;
            let entry = PeerCopy::new(self.src, self.dst, this)
                .src_offset(self.src_base + off)
                .dst_offset(self.dst_base + off)
                .build()
                .user_data(self.cookie_base.wrapping_add(self.chunks_queued));
            match sq.push(&entry) {
                Ok(()) => {
                    self.queued += this as u64;
                    self.chunks_queued += 1;
                    staged += 1;
                }
                Err(Full) => break,
            }
        }
        staged
    }

    /// **Latency path.** The whole transfer as a single `PEER_COPY`, for a copy
    /// that fits one SQE (`len <= u32::MAX`). The caller pushes it and kicks
    /// immediately — no batching. Returns `None` when the transfer is larger
    /// than one SQE can carry; use [`pump`](Self::pump) then.
    ///
    /// This does **not** advance the cursor — it is a distinct issue path, not
    /// a step of the batched one. Use one or the other for a given transfer.
    pub fn single(&self) -> Option<Entry> {
        let len = u32::try_from(self.len).ok()?;
        Some(
            PeerCopy::new(self.src, self.dst, len)
                .src_offset(self.src_base)
                .dst_offset(self.dst_base)
                .build()
                .user_data(self.cookie_base),
        )
    }

    /// Account `bytes` the peer reported complete for one of this transfer's
    /// chunks (a CQE with a non-negative result). Saturates at
    /// [`total`](Self::total), so a duplicated or over-large credit can never
    /// push [`completed`](Self::completed) past the transfer size.
    pub fn credit(&mut self, bytes: u32) {
        self.completed = (self.completed + bytes as u64).min(self.len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spsc;
    use crate::uring::{IoUring, types::Buf};
    use kore_abi::uring::{IoUringCqe, IoUringSqe, op};

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

    /// A staged `PEER_COPY` carries the operands on the pinned wire fields, so
    /// a device consumer reads src/dst/offsets/len back exactly.
    #[test]
    fn peer_copy_maps_operands_onto_the_pinned_wire_fields() {
        let mut sq_window = Window::<SQ_BYTES>::new();
        let mut cq_window = Window::<CQ_BYTES>::new();
        let (client_sq, mut device_sq) = spsc::split(&mut sq_window.0, DEPTH).expect("SQ split");
        let (_device_cq, client_cq) = spsc::split(&mut cq_window.0, DEPTH).expect("CQ split");
        let mut ring = IoUring::from_halves(client_sq, client_cq).expect("matching rings");

        let entry = PeerCopy::new(Buf(2), Buf(5), 4096)
            .src_offset(0x1000)
            .dst_offset(0x2000)
            .build()
            .user_data(0x99);
        ring.submission().push(&entry).expect("SQ has room");

        let sqe = device_sq.pop().expect("device receives the copy");
        assert_eq!(sqe.opcode, op::PEER_COPY);
        assert_eq!(sqe.buf_index, 2, "source registered buffer");
        assert_eq!(sqe.splice_fd_in, 5, "destination registered buffer");
        assert_eq!(sqe.off, 0x1000, "source offset");
        assert_eq!(sqe.addr, 0x2000, "destination offset");
        assert_eq!(sqe.len, 4096);
        assert_eq!(sqe.user_data, 0x99);
    }

    /// `pump` fills the SQ, stops at [`Full`] without losing the cursor, and
    /// resumes exactly where it left off once the peer drains — the bandwidth
    /// contract, chunked to smaller-than-SQ pieces.
    #[test]
    fn pump_fills_the_queue_then_resumes_after_the_peer_drains() {
        let mut sq_window = Window::<SQ_BYTES>::new();
        let mut cq_window = Window::<CQ_BYTES>::new();
        let (client_sq, mut device_sq) = spsc::split(&mut sq_window.0, DEPTH).expect("SQ split");
        let (_device_cq, client_cq) = spsc::split(&mut cq_window.0, DEPTH).expect("CQ split");
        let mut ring = IoUring::from_halves(client_sq, client_cq).expect("matching rings");

        // 10 chunks of 1 KiB; the SQ holds 4.
        let mut xfer = PeerTransfer::new(Buf(0), Buf(1), 10 * 1024)
            .with_chunk(1024)
            .cookie_base(0x1000);

        assert_eq!(
            xfer.pump(&mut ring.submission()),
            4,
            "first pump fills the SQ"
        );
        assert_eq!(xfer.queued(), 4 * 1024);
        assert!(!xfer.is_fully_queued());
        // A second pump with no room stages nothing and keeps the cursor.
        assert_eq!(xfer.pump(&mut ring.submission()), 0);
        assert_eq!(xfer.queued(), 4 * 1024);

        // Peer drains two; two slots free up.
        assert!(device_sq.pop().is_some());
        assert!(device_sq.pop().is_some());
        assert_eq!(xfer.pump(&mut ring.submission()), 2);
        assert_eq!(xfer.queued(), 6 * 1024);

        // Drain the rest and finish staging the transfer.
        while device_sq.pop().is_some() {}
        assert_eq!(xfer.pump(&mut ring.submission()), 4);
        while device_sq.pop().is_some() {}
        assert_eq!(
            xfer.pump(&mut ring.submission()),
            0,
            "10th chunk was the last"
        );
        assert!(xfer.is_fully_queued());
        assert_eq!(xfer.queued(), 10 * 1024);
    }

    /// Correlation tokens run `cookie_base + chunk_index`, and the tail chunk
    /// is the remainder — the whole transfer stages exactly once.
    #[test]
    fn chunks_correlate_and_the_tail_is_the_remainder() {
        let mut sq_window = Window::<SQ_BYTES>::new();
        let mut cq_window = Window::<CQ_BYTES>::new();
        let (client_sq, mut device_sq) = spsc::split(&mut sq_window.0, DEPTH).expect("SQ split");
        let (_device_cq, client_cq) = spsc::split(&mut cq_window.0, DEPTH).expect("CQ split");
        let mut ring = IoUring::from_halves(client_sq, client_cq).expect("matching rings");

        // 2500 bytes in 1000-byte chunks → 1000, 1000, 500.
        let mut xfer = PeerTransfer::new(Buf(3), Buf(4), 2500)
            .with_chunk(1000)
            .cookie_base(0x40);

        assert_eq!(xfer.pump(&mut ring.submission()), 3);
        assert!(xfer.is_fully_queued());

        let first = device_sq.pop().expect("chunk 0");
        assert_eq!(first.user_data, 0x40);
        assert_eq!(first.len, 1000);
        assert_eq!(first.off, 0);
        let second = device_sq.pop().expect("chunk 1");
        assert_eq!(second.user_data, 0x41);
        assert_eq!(second.len, 1000);
        assert_eq!(second.off, 1000);
        let tail = device_sq.pop().expect("chunk 2");
        assert_eq!(tail.user_data, 0x42);
        assert_eq!(tail.len, 500, "tail is the remainder");
        assert_eq!(tail.off, 2000);
    }

    /// Completion credit tracks progress and saturates at the total.
    #[test]
    fn credit_tracks_completion_and_saturates() {
        let mut xfer = PeerTransfer::new(Buf(0), Buf(1), 3000).with_chunk(1000);
        assert!(!xfer.is_done());
        xfer.credit(1000);
        xfer.credit(1000);
        assert_eq!(xfer.completed(), 2000);
        assert!(!xfer.is_done());
        // An over-large / duplicated credit cannot run past the total.
        xfer.credit(5000);
        assert_eq!(xfer.completed(), 3000);
        assert!(xfer.is_done());
    }

    /// The latency path is one SQE for a fits-in-one transfer, and refuses
    /// (rather than truncating) a transfer too large for a single copy.
    #[test]
    fn single_is_one_sqe_and_refuses_the_oversized() {
        let small = PeerTransfer::new(Buf(7), Buf(8), 4096)
            .src_base(0x10)
            .dst_base(0x20)
            .cookie_base(0xabc);
        let entry = small.single().expect("fits one SQE");
        // Round-trip the operands through a ring to confirm the wire mapping.
        let mut sq_window = Window::<SQ_BYTES>::new();
        let mut cq_window = Window::<CQ_BYTES>::new();
        let (client_sq, mut device_sq) = spsc::split(&mut sq_window.0, DEPTH).expect("SQ split");
        let (_device_cq, client_cq) = spsc::split(&mut cq_window.0, DEPTH).expect("CQ split");
        let mut ring = IoUring::from_halves(client_sq, client_cq).expect("matching rings");
        ring.submission().push(&entry).expect("room");
        let sqe = device_sq.pop().expect("single copy");
        assert_eq!(sqe.opcode, op::PEER_COPY);
        assert_eq!(sqe.len, 4096);
        assert_eq!(sqe.off, 0x10);
        assert_eq!(sqe.addr, 0x20);
        assert_eq!(sqe.user_data, 0xabc);

        let huge = PeerTransfer::new(Buf(0), Buf(1), u64::from(u32::MAX) + 1);
        assert!(huge.single().is_none(), "too large for one SQE");
    }
}
