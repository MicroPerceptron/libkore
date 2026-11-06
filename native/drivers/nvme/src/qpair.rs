//! The **queue pair** — an NVMe SQ/CQ pair, the SPDK `spdk_nvme_qpair`
//! shape: per-qpair, **lockless** (no shared state, no server mediating —
//! the kore no-global-locks doctrine, which SPDK arrived at independently),
//! polled completions, batched doorbell.  The admin queue is just qpair 0.
//!
//! Since X0 the ring *mechanics* live in [`kore_driver`]: an NVMe qpair is a
//! [`SubRing`]`<NvmeSqe>` (Modulo wrap — the SQ tail doorbell is the publish
//! signal, SQEs carry no validity bit) + a [`CompRing`]`<NvmeCqe>` (the CQ
//! Phase bit is the validity bit) + a [`CookieTable`].  This file is what's
//! left once the shared parts are factored out: the **NVMe-specific** cid
//! stamping, cookie correlation, doorbell-offset math, and the books — plus
//! the [`SqEntry`]/[`CqEntry`] impls that teach the substrate what an NVMe
//! entry's validity bit is (SQE: none; CQE: Phase).
//!
//! **The hot path is here and it is short**: [`submit`](Qpair::submit)
//! stages one SQE and records its cookie (no MMIO); [`ring`](Qpair::ring)
//! writes the doorbell **once** for a whole batch; [`poll`](Qpair::poll)
//! reads the CQ phase bit and correlates the completion to its cookie.  No
//! allocation, no lock, no kernel — the device and this qpair share only the
//! ring frames.

use core::sync::atomic::AtomicU32;

use kore_abi::device::IovaAddr;

use kore_driver::ring::{CompRing, CqEntry, SqEntry, SubRing, Wrap};

use crate::commands::{NvmeCqe, NvmeSqe};
use crate::regs::Capabilities;
use crate::seam::{Dma, Regs};

/// The **heapless** per-qpair depth ceiling — the fixed inline cookie
/// table's size when the `alloc` feature is off (the kernel boot-test
/// posture).  With `alloc` (userland — the global allocator is the
/// mm-backed heap), the table is sized **exactly** to the negotiated
/// depth instead and this ceiling does not apply.  (The substrate's
/// [`kore_driver::MAX_SLOTS`] under its NVMe-facing name.)
pub use kore_driver::MAX_SLOTS as MAX_QDEPTH;

/// The qpair's visible books — [`kore_driver::RingStats`] under its NVMe-facing
/// name.
pub use kore_driver::RingStats as QpairStats;

/// An NVMe SQE carries **no in-entry validity bit** — the SQ tail doorbell
/// is the sole publish signal — so cycle stamping is the no-op default.
impl SqEntry for NvmeSqe {}

/// An NVMe CQE's validity bit is its status **Phase** bit.
impl CqEntry for NvmeCqe {
    fn valid(&self, phase: bool) -> bool {
        self.phase() == phase
    }
}

/// Why a submit/poll refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QpairError {
    /// The submission ring is full — reap completions, then retry.
    Full,
    /// A requested depth exceeds [`MAX_QDEPTH`] or the controller's MQES.
    BadDepth,
}

/// A reaped completion, correlated to the cookie its submission carried.
#[derive(Clone, Copy, Debug)]
pub struct Completion {
    /// The command id the device echoed.
    pub cid: u16,
    /// The cookie the submitter attached (kore-ring-shaped correlation).
    pub cookie: u64,
    /// The full completion entry (status, result, …).
    pub cqe: NvmeCqe,
}

impl Completion {
    /// Whether the command succeeded.
    pub fn is_success(&self) -> bool {
        self.cqe.is_success()
    }
}

/// One submission queue of a [`Qpair`]: its ring, doorbell, command-id
/// table and books. CIDs are unique per submission queue, as the spec
/// requires; a completion names its queue (`sq_id`) and its CID.
struct SubQueue<S: Dma> {
    qid: u16,
    ring: SubRing<NvmeSqe, S>,
    doorbell_off: usize,
    /// cid → cookie; an id is claimed on submit and released only by its
    /// own completion.
    cookies: kore_driver::CookieTable,
    /// Submitted, not yet completed.
    inflight: u16,
    /// Tail doorbell writes since construction.
    doorbells: u64,
    /// The head the controller reported in this queue's last completion.
    last_head: u16,
}

/// NVMe submission queues over one completion queue — a **queue pair**
/// (`SQS = 1`, the admin queue and every queue before classes existed) or a
/// **queue group**: several submission queues bound to one CQ and one
/// interrupt vector, so each service class has its own hardware queue
/// (`res/STORAGE_TRANSPORT.md` §3).
///
/// Every single-queue method (`submit`, `ring`, `sq_tail`, …) acts on
/// submission queue 0; the `_on` forms take a queue index. A group the
/// controller granted fewer queues than `SQS` is built with the queues it
/// has, and an index past them folds onto the last live queue — degraded
/// isolation, never a refusal.
pub struct Qpair<S: Dma, C: Dma, const SQS: usize = 1> {
    /// The completion queue's id (a pair's SQ shares it).
    qid: u16,
    sqs: [Option<SubQueue<S>>; SQS],
    /// Submission queues actually created (`1..=SQS`).
    live: usize,
    cq: CompRing<NvmeCqe, C>,
    /// Each submission queue's depth.
    depth: u16,
    cq_doorbell_off: usize,
    /// The books, across every submission queue.
    stats: QpairStats,
}

impl<S: Dma, C: Dma> Qpair<S, C, 1> {
    /// Build a qpair over pre-allocated `sq`/`cq` buffers.  Doorbell
    /// offsets come from [`Capabilities`].  Heapless posture: `depth ≤
    /// `[`MAX_QDEPTH`]; with `alloc`, the cookie table is sized exactly
    /// to `depth` (any depth the controller's MQES admits).
    pub fn new(
        qid: u16,
        sq: S,
        cq: C,
        depth: u16,
        caps: &Capabilities,
    ) -> Result<Self, QpairError> {
        Self::new_group(qid, &[qid], [Some(sq)], cq, depth, depth, caps)
    }
}

impl<S: Dma, C: Dma, const SQS: usize> Qpair<S, C, SQS> {
    /// Build a queue group: submission queues `sqids[i]` over `sq_bufs[i]`
    /// (the first `sqids.len()` present), all bound to completion queue
    /// `cqid` over `cq`. The CQ must hold every command the queues can have
    /// in flight — `cq_depth ≥ live × depth` — or a full set of
    /// completions could overrun it.
    pub fn new_group(
        cqid: u16,
        sqids: &[u16],
        sq_bufs: [Option<S>; SQS],
        cq: C,
        depth: u16,
        cq_depth: u16,
        caps: &Capabilities,
    ) -> Result<Self, QpairError> {
        let live = sqids.len();
        if depth == 0 || live == 0 || live > SQS {
            return Err(QpairError::BadDepth);
        }
        #[cfg(not(feature = "alloc"))]
        if depth as usize > MAX_QDEPTH {
            return Err(QpairError::BadDepth);
        }
        if (cq_depth as usize) < live * depth as usize {
            return Err(QpairError::BadDepth);
        }
        let mut sq_bufs = sq_bufs;
        let mut sqs: [Option<SubQueue<S>>; SQS] = core::array::from_fn(|_| None);
        for (index, &qid) in sqids.iter().enumerate() {
            let ring = sq_bufs[index].take().ok_or(QpairError::BadDepth)?;
            sqs[index] = Some(SubQueue {
                qid,
                ring: SubRing::new(ring, depth, Wrap::Modulo),
                doorbell_off: caps.sq_doorbell(qid),
                cookies: kore_driver::CookieTable::new(depth),
                inflight: 0,
                doorbells: 0,
                last_head: 0,
            });
        }
        Ok(Self {
            qid: cqid,
            sqs,
            live,
            cq: CompRing::new(cq, cq_depth),
            depth,
            cq_doorbell_off: caps.cq_doorbell(cqid),
            stats: QpairStats::default(),
        })
    }

    /// Submission queues actually created.
    pub fn live_sqs(&self) -> usize {
        self.live
    }

    /// The live queue index `sq` folds onto.
    fn index(&self, sq: usize) -> usize {
        sq.min(self.live - 1)
    }

    fn queue(&self, sq: usize) -> &SubQueue<S> {
        self.sqs[self.index(sq)]
            .as_ref()
            .expect("every index below live is built")
    }

    fn queue_mut(&mut self, sq: usize) -> &mut SubQueue<S> {
        let index = self.index(sq);
        self.sqs[index]
            .as_mut()
            .expect("every index below live is built")
    }

    /// SQ tail doorbell writes on queue 0 since construction.
    pub fn doorbells(&self) -> u64 {
        self.queue(0).doorbells
    }

    /// The software submission tail of queue 0 — the value the last
    /// doorbell carried, or would carry.
    pub fn sq_tail(&self) -> u16 {
        self.queue(0).ring.tail()
    }

    /// The submission head the controller reported in queue 0's last
    /// completion. Against [`sq_tail`](Self::sq_tail) it says whether the
    /// controller has consumed what was rung in.
    pub fn last_sq_head(&self) -> u16 {
        self.queue(0).last_head
    }

    /// The raw bytes of queue 0's entry `back` slots behind the tail (0 =
    /// the last staged), as the ring holds them — what the controller
    /// fetches, for a command it will not fetch.  `None` past the ring.
    pub fn sq_entry_raw(&self, back: u16) -> Option<[u8; 64]> {
        const _: () = assert!(size_of::<NvmeSqe>() == 64);
        if back >= self.depth {
            return None;
        }
        let queue = self.queue(0);
        let slot = (queue.ring.tail() + self.depth - 1 - back) % self.depth;
        // SAFETY: the ring is `depth` entries of 64 device-visible bytes and
        // `slot < depth`; a volatile read, since the device may be reading
        // the same bytes.
        Some(unsafe {
            queue
                .ring
                .cpu()
                .add(usize::from(slot) * 64)
                .cast::<[u8; 64]>()
                .read_volatile()
        })
    }

    /// The books, across every submission queue.
    pub fn stats(&self) -> QpairStats {
        self.stats
    }

    /// The completion queue's id.
    pub fn qid(&self) -> u16 {
        self.qid
    }

    /// Queue 0's SQ base IOVA (what `ASQ` / Create-SQ point at).
    pub fn sq_iova(&self) -> IovaAddr {
        self.queue(0).ring.iova()
    }

    /// `(qid, cid)` of the command in flight on queue `sq` (folded as
    /// submission folds it) carrying `cookie` — what an Abort names.
    pub fn locate(&self, sq: usize, cookie: u64) -> Option<(u16, u16)> {
        let queue = self.sqs.get(self.index(sq))?.as_ref()?;
        Some((queue.qid, queue.cookies.find(cookie)?))
    }

    /// Submission queue `sq`'s id and base IOVA — what its Create I/O SQ
    /// names. `None` past the live queues.
    pub fn sq_of(&self, sq: usize) -> Option<(u16, IovaAddr)> {
        self.sqs
            .get(sq)
            .and_then(Option::as_ref)
            .map(|queue| (queue.qid, queue.ring.iova()))
    }

    /// CQ base IOVA (what `ACQ` / Create-CQ point at).
    pub fn cq_iova(&self) -> IovaAddr {
        self.cq.iova()
    }

    /// Each submission queue's depth.
    pub fn depth(&self) -> u16 {
        self.depth
    }

    /// The completion queue's depth.
    pub fn cq_depth(&self) -> u16 {
        self.cq.capacity()
    }

    /// Forget every queue's position and every command in flight, as the
    /// controller did when it reset: rings back to slot 0, the completion
    /// ring zeroed, command ids all free. The memory stays; the queues are
    /// created again over it. Answers how many commands were lost.
    pub fn reset(&mut self) -> u32 {
        let mut lost = 0u32;
        for queue in self.sqs.iter_mut().flatten() {
            queue.ring.reset();
            lost += u32::from(queue.cookies.clear());
            queue.inflight = 0;
            queue.last_head = 0;
        }
        self.cq.reset();
        lost
    }

    /// Whether `n` more submissions fit on queue 0.
    pub fn has_space(&self, n: u16) -> bool {
        self.has_space_on(0, n)
    }

    /// Whether `n` more submissions fit on queue `sq`.
    pub fn has_space_on(&self, sq: usize, n: u16) -> bool {
        self.queue(sq).inflight + n < self.depth
    }

    /// Commands queue `sq` can still take.
    pub fn room_on(&self, sq: usize) -> u16 {
        (self.depth - 1).saturating_sub(self.queue(sq).inflight)
    }

    /// **Stage** one command on queue 0 — see [`submit_on`](Self::submit_on).
    pub fn submit(&mut self, cmd: NvmeSqe, cookie: u64) -> Result<u16, QpairError> {
        self.submit_on(0, cmd, cookie)
    }

    /// **Stage** one command on queue `sq` with a correlation `cookie` —
    /// writes the SQE, stamps a fresh cid, advances the tail.  Does **not**
    /// ring the doorbell (batch first, [`ring_on`](Self::ring_on) once).
    /// Returns the cid.
    pub fn submit_on(&mut self, sq: usize, mut cmd: NvmeSqe, cookie: u64) -> Result<u16, QpairError> {
        if !self.has_space_on(sq, 1) {
            return Err(QpairError::Full);
        }
        let queue = self.queue_mut(sq);
        // The cid is claimed, not counted: a command completes out of order
        // on a drive serving random reads, so an id is free again only when
        // its own completion was reaped, never because the ring wrapped.
        let Some(cid) = queue.cookies.claim(cookie) else {
            return Err(QpairError::Full);
        };
        cmd.set_cid(cid);
        queue.ring.stage(cmd);
        queue.inflight += 1;
        self.stats.submitted += 1;
        Ok(cid)
    }

    /// **Ring queue 0's SQ doorbell** — see [`ring_on`](Self::ring_on).
    pub fn ring(&mut self, regs: &impl Regs) {
        self.ring_on(0, regs);
    }

    /// **Ring queue `sq`'s SQ doorbell** — publish its staged batch (release
    /// fence, then one MMIO write of the tail).
    pub fn ring_on(&mut self, sq: usize, regs: &impl Regs) {
        let queue = self.queue_mut(sq);
        let off = queue.doorbell_off;
        queue.ring.ring(|tail| regs.write32(off, tail as u32));
        queue.doorbells += 1;
    }

    /// Submit + ring in one call — the convenience for a lone command.
    pub fn submit_one(
        &mut self,
        cmd: NvmeSqe,
        cookie: u64,
        regs: &impl Regs,
    ) -> Result<u16, QpairError> {
        let cid = self.submit(cmd, cookie)?;
        self.ring(regs);
        Ok(cid)
    }

    /// **Poll one completion** (non-blocking): if the CQ head holds an
    /// entry with the expected phase, consume it, advance the head, ring
    /// the CQ doorbell, and return it correlated to its cookie.
    pub fn poll(&mut self, regs: &impl Regs) -> Option<Completion> {
        let completion = self.reap_one()?;
        regs.write32(self.cq_doorbell_off, self.cq.head() as u32);
        Some(completion)
    }

    /// **The CQ head slot's status dword** — the word a completion's arrival
    /// stores to, for a caller that wants to sleep its core on that store.
    ///
    /// The controller publishes a CQE by DMA-writing all sixteen of its bytes,
    /// the phase bit among them. So the dword carrying that bit is exactly the
    /// line an address monitor
    /// ([`kore_rt::hint::wait_on`](../../kore_rt/hint/fn.wait_on.html)) should
    /// watch: the wake is the completion itself, needing no interrupt, no
    /// route, and no waker — which is what makes it usable on the route-less
    /// posture where nothing exists that could ever deliver a wake.
    ///
    /// **This crate deliberately does not do the sleeping.** `kore-rt` is an
    /// optional `service`-feature dependency precisely so the driver core stays
    /// out of the kernel binary, and the monitored wait is a runtime mechanism.
    /// So the split is by what each layer knows: this crate knows *which* word
    /// carries the validity bit (it defines [`NvmeCqe`]), and its caller knows
    /// how long a core may be held. Publishing the word rather than the wait
    /// keeps [`CompRing`] hardware-neutral — no NVMe layout crosses into
    /// `kore_driver` — while costing that crate nothing.
    ///
    /// The caller snapshots this word, then sleeps while it still reads that
    /// snapshot. It must **not** treat a wake as a completion: a neighbour in
    /// the same line, an interrupt, or the cap all end the sleep, so the poll
    /// after it is the only truth. That is the same discipline
    /// [`CompRing::poll`]'s phase check already is.
    ///
    /// `None` when the ring base is not `u32`-aligned, which no DMA pool in
    /// this tree produces (a CQ is page-aligned) but which cannot be *proved*
    /// here — and an unaligned atomic is undefined behaviour, not a wrong
    /// answer. Refusing costs the caller its fast tier and nothing else.
    pub fn completion_word(&self) -> Option<&AtomicU32> {
        /// Byte offset of the dword holding the CQE's phase bit.
        ///
        /// **Derived, not written down.** `status` is a `u16` at offset 14 of
        /// a `repr(C)` [`NvmeCqe`], so the enclosing aligned dword starts at
        /// 12 — where `cid` also lives. Masking the field offset down to its
        /// dword means a layout change moves this with it instead of leaving a
        /// stale constant pointing at `sq_head`.
        const PHASE_DWORD: usize = core::mem::offset_of!(NvmeCqe, status) & !3;
        const _: () = assert!(PHASE_DWORD == core::mem::offset_of!(NvmeCqe, cid));
        const _: () = assert!(PHASE_DWORD + 4 <= core::mem::size_of::<NvmeCqe>());

        let base = self.cq.cpu();
        if !base.cast::<u32>().is_aligned() {
            return None;
        }
        let offset = self.cq.head() as usize * core::mem::size_of::<NvmeCqe>() + PHASE_DWORD;
        // SAFETY: `cq` is `capacity * size_of::<NvmeCqe>()` device-visible
        // bytes and `head < capacity`, so `offset .. offset + 4` is inside the
        // ring. The base is `u32`-aligned (checked above) and both addends are
        // multiples of 4, so the result is too. The device writes this word
        // concurrently, which is why it is handed out as an `AtomicU32` rather
        // than a `&u32`: every CPU-side access is then a single atomic load,
        // the same posture `CompRing::poll`'s `read_volatile` of the whole
        // entry already takes against the same writer.
        Some(unsafe { &*base.add(offset).cast::<AtomicU32>() })
    }

    /// Reap up to `out.len()` ready completions and acknowledge the final CQ
    /// head with one MMIO write. Output slots are overwritten from the
    /// beginning; the returned count names the initialized prefix.
    pub fn poll_batch(&mut self, regs: &impl Regs, out: &mut [Option<Completion>]) -> usize {
        let mut count = 0;
        for slot in out {
            let Some(completion) = self.reap_one() else {
                break;
            };
            *slot = Some(completion);
            count += 1;
        }
        if count != 0 {
            regs.write32(self.cq_doorbell_off, self.cq.head() as u32);
        }
        count
    }

    /// Consume one ready CQE without ringing the head doorbell. Callers must
    /// acknowledge the final head after their bounded drain.
    fn reap_one(&mut self) -> Option<Completion> {
        loop {
            let cqe = self.cq.poll()?;
            // The completion names its submission queue; a CID is unique
            // only within one. A queue id this group does not hold, or a cid
            // naming no command in flight on it, is a completion the device
            // owes nobody: consumed and counted, never delivered, so it can
            // neither free an id twice nor answer a request with another
            // command's data.
            //
            // A lone queue takes every completion, exactly as a queue pair
            // always did — its behaviour never depends on the `sq_id` field.
            // Only a group routes by it, as it must.
            let live = self.live;
            let found = if live == 1 {
                self.sqs[0].as_mut()
            } else {
                self.sqs[..live]
                    .iter_mut()
                    .flatten()
                    .find(|queue| queue.qid == cqe.sq_id)
            };
            let Some(queue) = found else {
                self.stats.stray += 1;
                continue;
            };
            queue.last_head = cqe.sq_head;
            let Some(cookie) = queue.cookies.release(cqe.cid) else {
                self.stats.stray += 1;
                continue;
            };
            queue.inflight = queue.inflight.saturating_sub(1);
            self.stats.completed += 1;
            if !cqe.is_success() {
                self.stats.errors += 1;
            }
            return Some(Completion {
                cid: cqe.cid,
                cookie,
                cqe,
            });
        }
    }

    /// Drain every ready completion, calling `f` per entry; returns the
    /// count (`spdk_nvme_qpair_process_completions`).
    pub fn process_completions(&mut self, regs: &impl Regs, mut f: impl FnMut(Completion)) -> u32 {
        let mut n = 0;
        while let Some(c) = self.reap_one() {
            f(c);
            n += 1;
        }
        if n != 0 {
            regs.write32(self.cq_doorbell_off, self.cq.head() as u32);
        }
        n
    }

    /// Submit one command and **poll to completion** — the admin-queue
    /// convenience (a blocking op on a synchronous queue).  Spins up to
    /// `max_polls` times; between polls the caller-supplied `pause` runs
    /// (a `spin_loop`, or a real inter-poll delay on hardware).  `None`
    /// on timeout.
    pub fn submit_and_wait(
        &mut self,
        cmd: NvmeSqe,
        regs: &impl Regs,
        max_polls: u32,
        pause: impl FnMut(),
    ) -> Option<Completion> {
        let mut pacer = kore_driver::block::CountedPacer::new(max_polls, pause);
        self.submit_and_wait_paced(cmd, regs, &mut pacer)
    }

    /// Submit one admin command and wait through the shared completion-pacing
    /// seam. Production uses an elapsed-time pacer; counted test callers keep
    /// [`submit_and_wait`](Self::submit_and_wait).
    pub fn submit_and_wait_paced(
        &mut self,
        cmd: NvmeSqe,
        regs: &impl Regs,
        pacer: &mut dyn kore_driver::block::CompletionPacer,
    ) -> Option<Completion> {
        let cid = self.submit_one(cmd, 0, regs).ok()?;
        loop {
            if let Some(c) = self.poll(regs)
                && c.cid == cid
            {
                return Some(c);
            }
            if !pacer.pace() {
                return self.poll(regs).filter(|completion| completion.cid == cid);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::Ordering;

    /// A CQ backing store the qpair really owns.
    ///
    /// [`StubBuf`] hands out a dangling pointer, which is enough for the
    /// constructor's geometry bookkeeping and useless here: `completion_word`
    /// returns a pointer *into* the ring, and these tests store through it.
    ///
    /// `skew` shifts the base off its natural alignment. It is the only way to
    /// reach the refusal path from a test, since every DMA pool in this tree
    /// page-aligns a completion queue.
    struct RingBuf<const N: usize> {
        slots: [NvmeCqe; N],
        skew: usize,
    }

    impl<const N: usize> RingBuf<N> {
        const fn new(skew: usize) -> Self {
            Self {
                slots: [NvmeCqe {
                    result: 0,
                    rsvd: 0,
                    sq_head: 0,
                    sq_id: 0,
                    cid: 0,
                    status: 0,
                }; N],
                skew,
            }
        }
    }

    impl<const N: usize> Dma for RingBuf<N> {
        fn cpu(&self) -> *mut u8 {
            // SAFETY: `skew` is 0 or 1 and the array is `N * 16` bytes with
            // `N > 0`, so the result stays inside the allocation. Handing out
            // `*mut` from `&self` is `Dma::cpu`'s contract — the posture
            // `NvmeBuffer` already takes against the same concurrent writer.
            unsafe { self.slots.as_ptr().cast::<u8>().cast_mut().add(self.skew) }
        }
        fn iova(&self) -> IovaAddr {
            IovaAddr(0x1000)
        }
        fn len(&self) -> usize {
            N * core::mem::size_of::<NvmeCqe>() - self.skew
        }
    }

    /// Doorbell writes go nowhere: these tests are about the ring, not MMIO.
    struct NullRegs;
    impl Regs for NullRegs {
        fn read32(&self, _off: usize) -> u32 {
            0
        }
        fn write32(&self, _off: usize, _val: u32) {}
    }

    /// The published dword, as the controller's DMA would leave it.
    ///
    /// `cid` is the low half and `status` the high half of one little-endian
    /// dword (every target in this tree is little-endian), and the phase bit
    /// is `status` bit 0. `CompRing` expects phase `1` on its first pass.
    const fn published(cid: u16) -> u32 {
        const _: () = assert!(cfg!(target_endian = "little"));
        (1u32 << 16) | cid as u32
    }

    /// **The word `completion_word` hands out is the one a completion lands
    /// in** — proven end to end through the real reap path rather than by
    /// comparing addresses.
    ///
    /// The failure this guards is quiet and expensive. Aliasing the wrong
    /// dword — `sq_head`, say, which is what a hardcoded offset invites when
    /// the struct changes — leaves a monitored wait sleeping on a word the
    /// controller never stores to. Every turn would then run to its cap and
    /// wake having learned nothing, and the tier would read as present and
    /// working while delivering strictly worse latency than the spin it
    /// replaced.
    #[test]
    fn the_monitored_word_is_where_a_completion_lands() {
        let mut qp =
            Qpair::new(1, SqBuf::<4>::new(), RingBuf::<4>::new(0), 4, &caps()).expect("depth fits");
        // A completion answers a command in flight; one is submitted so the
        // published cid names it (an unowned cid is dropped as stray).
        let cid = qp.submit(NvmeSqe::zeroed(), 1).expect("one fits");

        let word = qp
            .completion_word()
            .expect("an aligned ring can be monitored");
        assert_eq!(word.load(Ordering::Relaxed), 0, "a fresh slot reads stale");
        word.store(published(cid), Ordering::Relaxed);

        let completion = qp.poll(&NullRegs).expect("the published entry reaps");
        assert_eq!(completion.cid, cid, "the store landed in the head CQE");
        assert!(completion.cqe.phase(), "the phase bit shares that dword");
    }

    /// The watched word tracks the head, so the turn after a completion
    /// watches the NEXT slot and not the one already consumed.
    ///
    /// A word cached across completions would watch a slot the controller has
    /// finished with — the same silent never-fires failure as above, reached
    /// by a different route.
    #[test]
    fn the_monitored_word_follows_the_head() {
        let mut qp =
            Qpair::new(1, SqBuf::<4>::new(), RingBuf::<4>::new(0), 4, &caps()).expect("depth fits");
        let cid = qp.submit(NvmeSqe::zeroed(), 1).expect("one fits");

        let first = qp.completion_word().expect("aligned").as_ptr();
        qp.completion_word()
            .expect("aligned")
            .store(published(cid), Ordering::Relaxed);
        qp.poll(&NullRegs).expect("the published entry reaps");
        let second = qp.completion_word().expect("aligned").as_ptr();

        assert_eq!(
            second as usize - first as usize,
            core::mem::size_of::<NvmeCqe>(),
            "the watch must advance exactly one slot"
        );
    }

    /// An unaligned ring base declines the monitor rather than handing out a
    /// misaligned atomic — which is undefined behaviour, not a wrong answer.
    /// The caller keeps its ordinary pacing and loses only the fast tier.
    #[test]
    fn an_unaligned_ring_declines_the_monitor() {
        let qp = Qpair::new(1, StubBuf, RingBuf::<4>::new(1), 4, &caps()).expect("depth fits");
        assert!(qp.completion_word().is_none());
    }

    /// A submission ring the qpair really stages into.
    struct SqBuf<const N: usize> {
        slots: [NvmeSqe; N],
    }

    impl<const N: usize> SqBuf<N> {
        const fn new() -> Self {
            Self {
                slots: [NvmeSqe::zeroed(); N],
            }
        }
    }

    impl<const N: usize> Dma for SqBuf<N> {
        fn cpu(&self) -> *mut u8 {
            // SAFETY: as `RingBuf::cpu` — the staged entries land inside
            // the array, which is `N * 64` bytes.
            self.slots.as_ptr().cast::<u8>().cast_mut()
        }
        fn iova(&self) -> IovaAddr {
            IovaAddr(0x2000)
        }
        fn len(&self) -> usize {
            N * core::mem::size_of::<NvmeSqe>()
        }
    }

    /// Publish a completion for `cid` at the head and reap it.
    fn complete(qp: &mut Qpair<SqBuf<8>, RingBuf<8>>, cid: u16) -> Option<Completion> {
        qp.completion_word()
            .expect("aligned")
            .store(published(cid), Ordering::Relaxed);
        qp.poll(&NullRegs)
    }

    /// **An id is never reissued while its command is in flight.**  With
    /// every id but one claimed, the one completion that arrives — out of
    /// order, from the middle of the batch — frees exactly that id, and the
    /// next submit takes it; nothing else is free until its own completion.
    /// Round-robin issue would have handed out the oldest command's id here
    /// while that command was still in flight.
    #[test]
    fn an_id_is_reissued_only_after_its_own_completion() {
        let mut qp =
            Qpair::new(1, SqBuf::<8>::new(), RingBuf::<8>::new(0), 8, &caps()).expect("depth fits");
        let mut cids = [0u16; 7];
        for (n, cid) in cids.iter_mut().enumerate() {
            *cid = qp
                .submit(NvmeSqe::zeroed(), 100 + n as u64)
                .expect("seven fit in a ring of eight");
        }
        assert!(matches!(
            qp.submit(NvmeSqe::zeroed(), 0),
            Err(QpairError::Full)
        ));
        let middle = cids[3];
        let done = complete(&mut qp, middle).expect("a live completion reaps");
        assert_eq!(done.cookie, 103, "the completion answers its own submit");
        let again = qp.submit(NvmeSqe::zeroed(), 200).expect("one id is free");
        assert_eq!(again, middle, "the freed id, not the oldest command's");
        assert!(
            matches!(qp.submit(NvmeSqe::zeroed(), 0), Err(QpairError::Full)),
            "the six others still hold their ids"
        );
        let oldest = complete(&mut qp, cids[0]).expect("live");
        assert_eq!(oldest.cookie, 100);
        assert_eq!(qp.submit(NvmeSqe::zeroed(), 201), Ok(cids[0]));
    }

    /// A completion naming an id no command holds is consumed and counted,
    /// never delivered, and frees nothing.
    #[test]
    fn a_completion_for_no_command_is_dropped_and_counted() {
        let mut qp =
            Qpair::new(1, SqBuf::<8>::new(), RingBuf::<8>::new(0), 8, &caps()).expect("depth fits");
        let a = qp.submit(NvmeSqe::zeroed(), 7).expect("submit");
        assert!(
            complete(&mut qp, 5).is_none(),
            "nothing in flight has cid 5"
        );
        assert_eq!(qp.stats().stray, 1);
        assert_eq!(qp.stats().completed, 0);
        let done = complete(&mut qp, a).expect("the real one still reaps");
        assert_eq!(done.cookie, 7);
        assert!(complete(&mut qp, a).is_none(), "a second answer is stray");
        assert_eq!(qp.stats().stray, 2);
        assert_eq!(qp.stats().completed, 1);
    }

    /// A stub DMA buffer — the qpair constructor only records geometry.
    struct StubBuf;
    impl Dma for StubBuf {
        fn cpu(&self) -> *mut u8 {
            core::ptr::NonNull::dangling().as_ptr()
        }
        fn iova(&self) -> IovaAddr {
            IovaAddr(0x1000)
        }
        fn len(&self) -> usize {
            0x1000
        }
    }

    fn caps() -> Capabilities {
        Capabilities::from_cap(0x0000_0000_0001_003F)
    }

    /// The `alloc` posture sizes the cookie table to the *negotiated*
    /// depth through the global allocator (in kore-rt, the mm-backed heap) —
    /// a depth beyond the heapless inline ceiling is fine.  Without
    /// `alloc`, the same depth is a `BadDepth` refusal (the fixed table
    /// is the ceiling).
    #[test]
    fn depth_beyond_the_inline_ceiling_needs_alloc() {
        let deep = (MAX_QDEPTH + 256) as u16;
        let r = Qpair::new(1, StubBuf, StubBuf, deep, &caps());
        #[cfg(feature = "alloc")]
        assert!(r.is_ok(), "alloc posture sizes exactly to depth");
        #[cfg(not(feature = "alloc"))]
        assert!(
            matches!(r, Err(QpairError::BadDepth)),
            "heapless posture refuses past the inline ceiling"
        );
    }

    /// Zero depth refuses in every posture.
    #[test]
    fn zero_depth_refuses() {
        assert!(matches!(
            Qpair::new(1, StubBuf, StubBuf, 0, &caps()),
            Err(QpairError::BadDepth)
        ));
    }
}
