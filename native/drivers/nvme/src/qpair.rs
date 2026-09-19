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

/// One SQ/CQ pair over device-visible ring buffers `S`/`C`.
pub struct Qpair<S: Dma, C: Dma> {
    qid: u16,
    sq: SubRing<NvmeSqe, S>,
    cq: CompRing<NvmeCqe, C>,
    depth: u16,
    sq_doorbell_off: usize,
    cq_doorbell_off: usize,
    next_cid: u16,
    /// cid → cookie, indexed by `cid % table-len` (cids reused round-robin
    /// over the table).
    cookies: kore_driver::CookieTable,
    /// In-flight count (submitted, not yet completed).
    inflight: u16,
    /// The books.
    stats: QpairStats,
}

impl<S: Dma, C: Dma> Qpair<S, C> {
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
        if depth == 0 {
            return Err(QpairError::BadDepth);
        }
        #[cfg(not(feature = "alloc"))]
        if depth as usize > MAX_QDEPTH {
            return Err(QpairError::BadDepth);
        }
        Ok(Self {
            qid,
            sq: SubRing::new(sq, depth, Wrap::Modulo),
            cq: CompRing::new(cq, depth),
            depth,
            sq_doorbell_off: caps.sq_doorbell(qid),
            cq_doorbell_off: caps.cq_doorbell(qid),
            next_cid: 0,
            cookies: kore_driver::CookieTable::new(depth),
            inflight: 0,
            stats: QpairStats::default(),
        })
    }

    /// The books.
    pub fn stats(&self) -> QpairStats {
        self.stats
    }

    /// Queue id.
    pub fn qid(&self) -> u16 {
        self.qid
    }

    /// SQ base IOVA (what `ASQ` / Create-SQ point at).
    pub fn sq_iova(&self) -> IovaAddr {
        self.sq.iova()
    }

    /// CQ base IOVA (what `ACQ` / Create-CQ point at).
    pub fn cq_iova(&self) -> IovaAddr {
        self.cq.iova()
    }

    /// Queue depth.
    pub fn depth(&self) -> u16 {
        self.depth
    }

    /// Whether `n` more submissions fit without overrunning the ring.
    pub fn has_space(&self, n: u16) -> bool {
        self.inflight + n < self.depth
    }

    /// **Stage** one command with a correlation `cookie` — writes the SQE,
    /// stamps a fresh cid, advances the tail.  Does **not** ring the
    /// doorbell (batch first, [`ring`](Self::ring) once).  Returns the cid.
    pub fn submit(&mut self, mut cmd: NvmeSqe, cookie: u64) -> Result<u16, QpairError> {
        if !self.has_space(1) {
            return Err(QpairError::Full);
        }
        let cid = self.next_cid;
        cmd.set_cid(cid);
        self.cookies.put(cid, cookie);

        // Stage into the SQ ring (volatile write at the tail, advance).
        self.sq.stage(cmd);
        self.next_cid = self.next_cid.wrapping_add(1) % self.cookies.len() as u16;
        self.inflight += 1;
        self.stats.submitted += 1;
        Ok(cid)
    }

    /// **Ring the SQ doorbell** — publish the staged batch to the device
    /// (release fence, then one MMIO write of the tail).
    pub fn ring(&self, regs: &impl Regs) {
        self.sq
            .ring(|tail| regs.write32(self.sq_doorbell_off, tail as u32));
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
        let cqe = self.cq.poll()?;
        // A completion was consumed (the acquire fence is inside `poll`).
        self.inflight = self.inflight.saturating_sub(1);
        self.stats.completed += 1;
        if !cqe.is_success() {
            self.stats.errors += 1;
        }
        Some(Completion {
            cid: cqe.cid,
            cookie: self.cookies.get(cqe.cid),
            cqe,
        })
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
        let mut qp = Qpair::new(1, StubBuf, RingBuf::<4>::new(0), 4, &caps()).expect("depth fits");
        let cid = 0x1234u16;

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
        let mut qp = Qpair::new(1, StubBuf, RingBuf::<4>::new(0), 4, &caps()).expect("depth fits");

        let first = qp.completion_word().expect("aligned").as_ptr();
        qp.completion_word()
            .expect("aligned")
            .store(published(1), Ordering::Relaxed);
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
