//! **Selective refresh** of a read-mostly region (`res/PACK_STREAMING.md` §7).
//!
//! Flash read speed has two slow states, and only one of them is curable.
//!
//! 1. **Folded.** A client SSD writes new data into a pseudo-SLC cache and,
//!    when idle, folds it into native TLC/QLC. Native reads are slower: on
//!    Biggertron's Phison E29T a 4 KiB read takes 47 µs from the cache and
//!    113 µs folded, and folded data streams at 1.6–2.8 GB/s against 5 GB/s
//!    cached (`res/PACK_STREAMING.md` §7). Rewriting folded data only puts it
//!    back in the cache until the drive folds it again, about an hour of
//!    idle later. Nothing a host does changes this state.
//! 2. **Decayed.** Charge leaks out of cells over months. The controller
//!    then needs read-retry and soft-decode to recover the data, and every
//!    retry costs another native read time, so reads slow by whole
//!    multiples and keep slowing. Only a rewrite cures it, and most client
//!    firmware never rewrites on its own.
//!
//! [`RegionRefresh`] tells the two apart and rewrites only the second.
//!
//! - **Measure.** A scan splits the region into buckets and times
//!   [`PROBES`] single 4 KiB reads per bucket, one at a time, issued only
//!   while the device is otherwise idle, so a sample is the read's own
//!   latency rather than a queue's. It makes [`PASSES`] passes over the same
//!   pages and keeps each bucket's fastest pass: slowness that comes and goes
//!   (a cold mapping cache, the drive's own background work) does not
//!   survive the minimum, and decay, a property of the data, shows in every
//!   pass.
//! - **Classify.** A bucket is stale when its median is at least
//!   [`STALE_RATIO`] times the region's 25th percentile. The quartile, not
//!   the median, anchors the comparison so a region that has mostly decayed
//!   is still judged against its healthy part; the ratio sits above the
//!   folded-versus-cached spread so a folded bucket never qualifies.
//! - **Refresh.** Each chunk of a stale bucket is read twice into the job's
//!   own buffers; only bytes the two reads agree on are written back to the
//!   same blocks, and a third read must match them before the chunk counts.
//!   A bucket is flushed when its last chunk lands.
//!
//! **The content invariant.** A refresh never changes a byte. The write's
//! source is the job's own buffer, filled by the device from exactly the
//! blocks it is written back to (a chunk's extent is fixed for its whole
//! read → confirm → write → check cycle; [`Chunk::command`] derives every
//! command from it), and no other party can reach that buffer. A reader
//! racing the rewrite sees, block by block, the old copy or the new one —
//! the same bytes either way — so a refresh needs no exclusion against the
//! region's readers and changes nothing they can observe. A chunk that
//! reads as zeros is skipped: zeros are most likely never-written blocks,
//! and writing them would allocate flash for nothing.
//!
//! **The cost bound.** Each bucket is rewritten at most once per job
//! lifetime (per boot, for a row that keeps one job), forced or not, so the
//! endurance a refresh can spend is bounded by the region's size per boot.
//!
//! **Driving.** The job never waits. [`RegionRefresh::advance`] reaps what
//! completed, stages what is next, publishes once and returns; the owner
//! calls it on every wake, and the job's own completions are wakes.

use super::{BlockBuffer, BlockError, BlockGeometry, DeviceCommand, DeviceCompletion, DirectBlockDevice};

/// Probes per bucket. A median of sixteen single reads is stable against a
/// stray slow sample (Biggertron's idle 4 KiB QD1 p99 is 1.3× its median)
/// while a whole scan of a 40 GiB pack in 1024 buckets stays near two
/// seconds of idle device time (16 384 reads of about 120 µs).
pub const PROBES: usize = 16;

/// Passes a scan makes over the same probe pages; a bucket's estimate is its
/// fastest pass. On Biggertron (2026-10-16) a scan straight after boot read
/// p25 264 µs, p90 922 µs and judged 65 of 1024 buckets stale, while a scan
/// minutes later read p25 138 µs, p90 153 µs and none — and a bucket the
/// first scan put at 1617 µs, never rewritten, read 137 µs. That slowness
/// came and went; decay does not. Under the minimum, a false verdict needs
/// the same bucket slowed in every pass, and the first pass also warms the
/// drive's mapping cache for the pages the later ones read.
pub const PASSES: u32 = 3;

/// A bucket is stale at this multiple of the region's 25th-percentile
/// median. Healthy buckets differ by where the drive keeps them: Biggertron
/// read 47 µs from its cache and 113 µs folded, a 2.4× spread, so the ratio
/// must clear 2.4 with margin or a cached quartile would condemn every
/// folded bucket. Decay costs whole extra read times per read (a read-retry
/// is another full native read), so a decayed bucket crosses 4× of a folded
/// quartile after a few retries — and of a cached one after one.
pub const STALE_RATIO: u64 = 4;

/// Mapped buckets a scan needs before it judges any of them: below four a
/// quartile is one bucket, and a region that small is not worth judging.
pub const MIN_BASELINE: u32 = 4;

/// Chunks a refresh keeps in flight. One chunk's cycle (read, confirm read,
/// write, check read) is serial; Biggertron's folded-data rate stops rising
/// past eight 128 KiB commands in flight, and four overlapping chunks keep
/// at least that many on the device once the engine splits each at the
/// transfer limit. Each slot holds two buffers of up to [`CHUNK_CEILING`].
pub const SLOTS: usize = 4;

/// The largest chunk, so the job's DMA memory stays bounded at
/// `SLOTS × 2 × 1 MiB` on a device whose transfer limit is larger. A
/// smaller limit (Biggertron's MDTS is 256 KiB) sizes the chunk instead.
pub const CHUNK_CEILING: usize = 1 << 20;

/// One probe reads this much: one 4 KiB flash page, the unit a read-retry
/// is paid in. A device whose blocks are larger probes one block.
const PROBE_BYTES: usize = 4096;

/// Attempts at one step before the chunk is given up: one retry. A second
/// failure of the same step on the same blocks is the device's answer.
const ATTEMPTS: u8 = 2;

/// A cookie is the job's generation in its high half and a tag in its low
/// half: a slot index, or one of the two tags beside them.
const TAG_BITS: u32 = 32;
const TAG_MASK: u64 = (1 << TAG_BITS) - 1;
const PROBE_TAG: u64 = SLOTS as u64;
const FLUSH_TAG: u64 = SLOTS as u64 + 1;

/// One bucket's books.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Bucket {
    /// The last scan's estimate: the fastest of its passes' medians of
    /// non-zero probes, in nanoseconds (saturating); zero before the bucket
    /// was scanned.
    pub median_ns: u32,
    /// Probes of the last scan that read non-zero data.
    pub mapped_probes: u8,
    /// Probes of the last scan the device failed.
    pub failed_probes: u8,
    /// The last scan judged this bucket stale and it has not been
    /// refreshed since.
    pub stale: bool,
    /// Rewritten during this job's lifetime: never again before the next.
    pub refreshed: bool,
    /// A chunk of this bucket could not be refreshed (a failed or unstable
    /// read): the rest of the bucket was, and the chunk is untouched.
    pub incomplete: bool,
}

impl Bucket {
    /// Whether the last scan found mapped data here: most probes read
    /// something other than zeros.
    pub const fn mapped(&self) -> bool {
        2 * self.mapped_probes as usize >= PROBES
    }
}

/// What the job is doing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Idle,
    Scanning,
    Refreshing,
    /// Cancelled or faulted: no new commands, waiting for the device to
    /// return the ones it holds.
    Draining,
}

/// How the last job ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// No job has run.
    None,
    Scanned,
    Refreshed,
    Cancelled,
    /// A write could not be made to stick: the device refused it, or a
    /// read after it disagreed with what was written, twice. The bucket was
    /// left as the device has it, flushed or not; the data buffer still
    /// held the agreed bytes when the job stopped.
    Faulted(BlockError),
}

/// Why a request to start was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// A job is running.
    Busy,
    /// No scan has completed: nothing is classified yet.
    NotScanned,
    /// The bucket range is empty or outside the region.
    Range,
    /// The device reads no clock, so nothing can be timed.
    NoClock,
}

/// What [`RegionRefresh::advance`] left behind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Progress {
    /// No job, nothing in flight.
    Idle,
    /// Commands are in flight: their completion is the next wake.
    InFlight,
    /// Work is waiting with nothing of the job's in flight — the device was
    /// busy for a probe, or refused a command. Another party's completion
    /// is the wake.
    Waiting,
}

/// A scan's verdict.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScanSummary {
    /// The last scan ran to its end.
    pub complete: bool,
    /// Buckets probed in the running pass, or by the last scan once it is
    /// complete.
    pub scanned: u32,
    /// The running pass, from one; [`PASSES`] once the scan is complete.
    pub pass: u32,
    /// Of those, buckets whose probes read mapped data.
    pub mapped: u32,
    /// Buckets judged stale.
    pub stale: u32,
    /// Percentiles of the mapped buckets' medians, in nanoseconds.
    pub p25_ns: u32,
    pub p50_ns: u32,
    pub p90_ns: u32,
    /// Probes the device failed.
    pub failed_probes: u32,
    /// Scans completed by this job.
    pub scans: u32,
}

/// A refresh's books. `buckets_*` describe the last (or running) refresh;
/// the byte and error counters accumulate over the job's lifetime.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RefreshSummary {
    /// Buckets the last refresh set out to rewrite.
    pub buckets_planned: u32,
    /// Of those, rewritten and flushed.
    pub buckets_done: u32,
    /// Asked for but skipped: already rewritten this boot.
    pub buckets_skipped: u32,
    /// Buckets rewritten over the job's lifetime.
    pub buckets_lifetime: u32,
    /// Bytes rewritten and verified.
    pub bytes: u64,
    /// Chunks left alone because they read as zeros.
    pub zero_chunks: u32,
    /// Reads the device failed (a failed chunk read is left untouched).
    pub read_errors: u32,
    /// Chunks whose two reads disagreed twice: left untouched.
    pub unstable_chunks: u32,
    /// Writes retried after a failure or a disagreeing check.
    pub write_retries: u32,
    /// Checks that disagreed with the bytes just written.
    pub mismatches: u32,
}

/// The region's shape as the job divides it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub block_bytes: u32,
    pub region_blocks: u64,
    pub bucket_blocks: u64,
    pub buckets: u32,
    pub chunk_blocks: u32,
    pub probe_blocks: u32,
}

impl Shape {
    fn new(geometry: BlockGeometry, max_buffer_bytes: usize, max_buckets: usize) -> Result<Self, BlockError> {
        let block = geometry.block_size as usize;
        if block == 0 || max_buckets == 0 || geometry.block_count == 0 {
            return Err(BlockError::Geometry);
        }
        let probe_blocks = if block >= PROBE_BYTES {
            1
        } else if PROBE_BYTES % block == 0 {
            PROBE_BYTES / block
        } else {
            return Err(BlockError::Geometry);
        };
        let probe_bytes = probe_blocks * block;
        let chunk_bytes = max_buffer_bytes.min(CHUNK_CEILING) / probe_bytes * probe_bytes;
        if chunk_bytes == 0 {
            return Err(BlockError::TooLarge);
        }
        let chunk_blocks = (chunk_bytes / block) as u64;
        let floor = (PROBES * probe_blocks) as u64;
        let wanted = geometry
            .block_count
            .div_ceil(max_buckets as u64)
            .max(chunk_blocks)
            .max(floor);
        let bucket_blocks = wanted.div_ceil(chunk_blocks) * chunk_blocks;
        let buckets = geometry.block_count.div_ceil(bucket_blocks);
        Ok(Self {
            block_bytes: geometry.block_size,
            region_blocks: geometry.block_count,
            bucket_blocks,
            buckets: u32::try_from(buckets).map_err(|_| BlockError::Geometry)?,
            chunk_blocks: chunk_blocks as u32,
            probe_blocks: probe_blocks as u32,
        })
    }

    /// Bucket `index` as `(first block, blocks)`; the last may be short.
    pub fn bucket(&self, index: u32) -> (u64, u64) {
        let start = u64::from(index) * self.bucket_blocks;
        (start, self.bucket_blocks.min(self.region_blocks - start))
    }

    /// Bytes in one full bucket.
    pub fn bucket_bytes(&self) -> u64 {
        self.bucket_blocks * u64::from(self.block_bytes)
    }

    /// Bytes in one full chunk.
    pub fn chunk_bytes(&self) -> u32 {
        self.chunk_blocks * self.block_bytes
    }

    /// Where probe `probe` of bucket `index` reads during scan `scan`: an
    /// even stride with a jitter inside it, so successive scans sample
    /// different pages and no two probes of one scan share one.
    fn probe_lba(&self, index: u32, probe: usize, scan: u32) -> u64 {
        let (start, blocks) = self.bucket(index);
        let pages = blocks / u64::from(self.probe_blocks);
        let stride = (pages / PROBES as u64).max(1);
        let slot = (probe as u64 * pages / PROBES as u64).min(pages.saturating_sub(1));
        let room = stride.min(pages - slot);
        let jitter = mix(u64::from(scan) << 32 | u64::from(index) << 8 | probe as u64) % room;
        start + (slot + jitter) * u64::from(self.probe_blocks)
    }
}

/// SplitMix64's finalizer: a deterministic spread, no state.
const fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Which step of its cycle a chunk is at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    /// Read the blocks into the data buffer.
    Read,
    /// Read them again into the check buffer; the two must agree.
    Confirm,
    /// Write the data buffer back to the same blocks.
    Write,
    /// Read them into the check buffer; it must match the data buffer.
    Check,
}

/// One chunk's cycle: a fixed extent and where it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Chunk {
    bucket: u32,
    lba: u64,
    blocks: u32,
    step: Step,
    attempts: u8,
    in_flight: bool,
}

impl Chunk {
    /// The command this step issues, and whether it lands in the check
    /// buffer. Every step names the same blocks: the extent is fixed when
    /// the chunk is taken, so a write can only return bytes to the blocks
    /// they were read from.
    const fn command(&self) -> (DeviceCommand, bool) {
        let (lba, blocks, buffer_offset) = (self.lba, self.blocks, 0);
        match self.step {
            Step::Read => (DeviceCommand::Read { lba, blocks, buffer_offset }, false),
            Step::Confirm | Step::Check => (DeviceCommand::Read { lba, blocks, buffer_offset }, true),
            Step::Write => (DeviceCommand::Write { lba, blocks, buffer_offset }, false),
        }
    }
}

struct Slot<B> {
    data: B,
    check: B,
    chunk: Option<Chunk>,
}

/// What a refresh rewrites.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Plan {
    /// Buckets the last scan judged stale.
    Stale,
    /// Buckets `first..end`, judged or not.
    Range { first: u32, end: u32 },
}

#[derive(Clone, Copy, Debug)]
struct ScanCursor {
    pass: u32,
    bucket: u32,
    probe: usize,
    staged_at: Option<u64>,
    samples: [u32; PROBES],
    mapped: u8,
    failed: u8,
}

#[derive(Clone, Copy, Debug)]
struct RefreshCursor {
    plan: Plan,
    /// Next bucket to consider.
    next: u32,
    /// The bucket being rewritten, its next chunk's block and its end.
    current: Option<(u32, u64, u64)>,
    /// The bucket's flush is in flight.
    flushing: bool,
}

#[derive(Clone, Copy, Debug)]
enum Work {
    None,
    Scan(ScanCursor),
    Refresh(RefreshCursor),
}

/// The scan-and-refresh job over one region. See the module docs.
pub struct RegionRefresh<'b, D: DirectBlockDevice> {
    device: D,
    shape: Shape,
    slots: [Slot<D::Buffer>; SLOTS],
    buckets: &'b mut [Bucket],
    work: Work,
    /// No new commands: the job is draining toward `outcome`.
    stopping: Option<Outcome>,
    in_flight: u32,
    /// Bumped per job: a completion carrying another generation is not
    /// this job's, and is counted rather than believed.
    generation: u32,
    stale_completions: u32,
    outcome: Outcome,
    scan: ScanSummary,
    refresh: RefreshSummary,
}

impl<'b, D: DirectBlockDevice> RegionRefresh<'b, D> {
    /// A job over `device`'s whole extent, keeping one [`Bucket`] per entry
    /// of `buckets` at most. Allocates its `2 × SLOTS` chunk buffers now,
    /// once: a refusal comes back with the device.
    pub fn new(mut device: D, buckets: &'b mut [Bucket]) -> Result<Self, (D, BlockError)> {
        let shape = match Shape::new(device.geometry(), device.max_buffer_bytes(), buckets.len()) {
            Ok(shape) => shape,
            Err(error) => return Err((device, error)),
        };
        let bytes = shape.chunk_bytes() as usize;
        let mut built: [Option<Slot<D::Buffer>>; SLOTS] = core::array::from_fn(|_| None);
        for slot in &mut built {
            let data = device.allocate_buffer(bytes);
            let check = device.allocate_buffer(bytes);
            match (data, check) {
                (Ok(data), Ok(check)) => {
                    *slot = Some(Slot {
                        data,
                        check,
                        chunk: None,
                    })
                }
                (Err(error), _) | (_, Err(error)) => return Err((device, error)),
            }
        }
        buckets.fill(Bucket::default());
        Ok(Self {
            device,
            shape,
            // Every slot was built above or the constructor returned.
            slots: built.map(|slot| slot.expect("every slot was built")),
            buckets,
            work: Work::None,
            stopping: None,
            in_flight: 0,
            generation: 0,
            stale_completions: 0,
            outcome: Outcome::None,
            scan: ScanSummary::default(),
            refresh: RefreshSummary::default(),
        })
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn phase(&self) -> Phase {
        match (self.stopping, self.work) {
            (Some(_), _) => Phase::Draining,
            (None, Work::None) => Phase::Idle,
            (None, Work::Scan(_)) => Phase::Scanning,
            (None, Work::Refresh(_)) => Phase::Refreshing,
        }
    }

    pub fn outcome(&self) -> Outcome {
        self.outcome
    }

    pub fn scan_summary(&self) -> ScanSummary {
        self.scan
    }

    pub fn refresh_summary(&self) -> RefreshSummary {
        self.refresh
    }

    /// The region's buckets, in order.
    pub fn buckets(&self) -> &[Bucket] {
        &self.buckets[..self.shape.buckets as usize]
    }

    /// Commands of the job's on the device.
    pub fn in_flight(&self) -> u32 {
        self.in_flight
    }

    /// Completions that named no command of the running job.
    pub fn stale_completions(&self) -> u32 {
        self.stale_completions
    }

    /// The device, for the owner's control plane.
    pub fn device(&self) -> &D {
        &self.device
    }

    /// Start a scan. Refused while a job runs or without a clock.
    pub fn scan(&mut self) -> Result<(), Refusal> {
        self.idle_or_busy()?;
        if self.device.clock_ns().is_none() {
            return Err(Refusal::NoClock);
        }
        self.begin();
        self.scan = ScanSummary {
            scans: self.scan.scans,
            pass: 1,
            ..ScanSummary::default()
        };
        for bucket in self.buckets.iter_mut() {
            bucket.stale = false;
        }
        self.work = Work::Scan(ScanCursor {
            pass: 0,
            bucket: 0,
            probe: 0,
            staged_at: None,
            samples: [0; PROBES],
            mapped: 0,
            failed: 0,
        });
        Ok(())
    }

    /// Rewrite every bucket the last complete scan judged stale.
    pub fn refresh_stale(&mut self) -> Result<(), Refusal> {
        self.idle_or_busy()?;
        if !self.scan.complete {
            return Err(Refusal::NotScanned);
        }
        self.start_refresh(Plan::Stale);
        Ok(())
    }

    /// Rewrite buckets `first..first + count`, judged or not — the forced
    /// form, for an operator who knows better than the scan and for the
    /// gate that proves the rewrite. Still at most once per bucket.
    pub fn refresh_range(&mut self, first: u32, count: u32) -> Result<(), Refusal> {
        self.idle_or_busy()?;
        let end = first.checked_add(count).ok_or(Refusal::Range)?;
        if count == 0 || end > self.shape.buckets {
            return Err(Refusal::Range);
        }
        self.start_refresh(Plan::Range { first, end });
        Ok(())
    }

    /// Stop the running job: nothing new is issued, and it ends
    /// [`Outcome::Cancelled`] once the device returned what it holds.
    pub fn cancel(&mut self) {
        if !matches!(self.work, Work::None) && self.stopping.is_none() {
            self.stopping = Some(Outcome::Cancelled);
        }
    }

    fn idle_or_busy(&self) -> Result<(), Refusal> {
        match self.phase() {
            Phase::Idle => Ok(()),
            _ => Err(Refusal::Busy),
        }
    }

    fn begin(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        self.stopping = None;
    }

    fn start_refresh(&mut self, plan: Plan) {
        self.begin();
        let planned = (0..self.shape.buckets).filter(|&index| self.wanted(plan, index)).count() as u32;
        let asked = match plan {
            Plan::Stale => planned,
            Plan::Range { first, end } => end - first,
        };
        self.refresh.buckets_planned = planned;
        self.refresh.buckets_done = 0;
        self.refresh.buckets_skipped = asked - planned;
        let next = match plan {
            Plan::Stale => 0,
            Plan::Range { first, .. } => first,
        };
        self.work = Work::Refresh(RefreshCursor {
            plan,
            next,
            current: None,
            flushing: false,
        });
    }

    fn wanted(&self, plan: Plan, index: u32) -> bool {
        let bucket = &self.buckets[index as usize];
        if bucket.refreshed {
            return false;
        }
        match plan {
            Plan::Stale => bucket.stale,
            Plan::Range { first, end } => (first..end).contains(&index),
        }
    }

    fn cookie(&self, tag: u64) -> u64 {
        u64::from(self.generation) << TAG_BITS | tag
    }

    /// Reap what completed, stage what is next, publish. Never waits.
    /// `idle`: nothing but this job is on the device — the owner's word,
    /// since only it sees the other parties. Probes are issued only then.
    pub fn advance(&mut self, idle: bool) -> Progress {
        while self.in_flight != 0 {
            let Some(completion) = self.device.poll() else {
                break;
            };
            self.complete(completion);
        }
        let staged = if self.stopping.is_some() {
            false
        } else {
            match self.work {
                Work::None => false,
                Work::Scan(_) => self.stage_probe(idle),
                Work::Refresh(_) => self.stage_refresh(),
            }
        };
        if staged {
            self.device.publish();
        }
        if self.in_flight == 0 {
            self.settle();
        }
        match (self.in_flight, self.work) {
            (0, Work::None) => Progress::Idle,
            (0, _) => Progress::Waiting,
            _ => Progress::InFlight,
        }
    }

    /// With nothing in flight: end a stopped job, or a finished one.
    fn settle(&mut self) {
        if let Some(outcome) = self.stopping.take() {
            self.finish(outcome);
            return;
        }
        match self.work {
            Work::Scan(cursor) if cursor.bucket >= self.shape.buckets => {
                self.judge();
                self.finish(Outcome::Scanned);
            }
            Work::Refresh(cursor)
                if cursor.current.is_none() && !cursor.flushing && cursor.next >= self.shape.buckets =>
            {
                self.finish(Outcome::Refreshed);
            }
            _ => {}
        }
    }

    fn finish(&mut self, outcome: Outcome) {
        for slot in &mut self.slots {
            slot.chunk = None;
        }
        self.work = Work::None;
        self.outcome = outcome;
    }

    fn fault(&mut self, error: BlockError) {
        if !matches!(self.stopping, Some(Outcome::Faulted(_))) {
            self.stopping = Some(Outcome::Faulted(error));
        }
    }

    fn stage(&mut self, command: DeviceCommand, slot: Option<(usize, bool)>, tag: u64) -> bool {
        let cookie = self.cookie(tag);
        let buffer = slot.map(|(index, check)| {
            let slot = &mut self.slots[index];
            if check { &mut slot.check } else { &mut slot.data }
        });
        match self.device.stage(command, buffer, cookie) {
            Ok(()) => {
                self.in_flight += 1;
                true
            }
            // Refused before it reached the device: try again on a later
            // wake, when what fills the queue has drained.
            Err(_) => false,
        }
    }

    // ── scan ────────────────────────────────────────────────────────────

    fn stage_probe(&mut self, idle: bool) -> bool {
        let Work::Scan(mut cursor) = self.work else {
            return false;
        };
        if cursor.staged_at.is_some() || cursor.bucket >= self.shape.buckets || !idle {
            return false;
        }
        let Some(now) = self.device.clock_ns() else {
            self.fault(BlockError::Unsupported);
            return false;
        };
        let lba = self.shape.probe_lba(cursor.bucket, cursor.probe, self.scan.scans);
        let command = DeviceCommand::Read {
            lba,
            blocks: self.shape.probe_blocks,
            buffer_offset: 0,
        };
        if !self.stage(command, Some((0, false)), PROBE_TAG) {
            return false;
        }
        cursor.staged_at = Some(now);
        self.work = Work::Scan(cursor);
        true
    }

    fn probed(&mut self, result: Result<(), BlockError>) {
        let Work::Scan(mut cursor) = self.work else {
            return;
        };
        let Some(staged_at) = cursor.staged_at.take() else {
            return;
        };
        match result {
            Ok(()) => {
                let bytes = (self.shape.probe_blocks * self.shape.block_bytes) as usize;
                if !is_zero(&self.slots[0].data.bytes()[..bytes]) {
                    let elapsed = self.device.clock_ns().unwrap_or(staged_at).saturating_sub(staged_at);
                    cursor.samples[cursor.mapped as usize] = u32::try_from(elapsed).unwrap_or(u32::MAX);
                    cursor.mapped += 1;
                }
            }
            Err(_) => {
                cursor.failed += 1;
                self.scan.failed_probes += 1;
            }
        }
        cursor.probe += 1;
        if cursor.probe == PROBES {
            let samples = &mut cursor.samples[..cursor.mapped as usize];
            samples.sort_unstable();
            let median = samples.get(samples.len() / 2).copied();
            let bucket = &mut self.buckets[cursor.bucket as usize];
            // The first pass sets the books; a later pass can only lower the
            // estimate. A pass whose probes all read zeros or failed has no
            // median and leaves it where it was.
            if cursor.pass == 0 {
                bucket.median_ns = median.unwrap_or(0);
                bucket.mapped_probes = cursor.mapped;
                bucket.failed_probes = cursor.failed;
            } else {
                if let Some(median) = median {
                    bucket.median_ns = if bucket.median_ns == 0 { median } else { bucket.median_ns.min(median) };
                }
                bucket.mapped_probes = bucket.mapped_probes.max(cursor.mapped);
                bucket.failed_probes = bucket.failed_probes.max(cursor.failed);
            }
            self.scan.scanned += 1;
            let (pass, bucket) = if cursor.bucket + 1 < self.shape.buckets || cursor.pass + 1 == PASSES {
                (cursor.pass, cursor.bucket + 1)
            } else {
                self.scan.pass += 1;
                self.scan.scanned = 0;
                (cursor.pass + 1, 0)
            };
            cursor = ScanCursor {
                pass,
                bucket,
                probe: 0,
                staged_at: None,
                samples: [0; PROBES],
                mapped: 0,
                failed: 0,
            };
        }
        self.work = Work::Scan(cursor);
    }

    /// The verdict on a finished scan: percentiles of the mapped buckets
    /// and the stale marks.
    fn judge(&mut self) {
        let count = self.shape.buckets as usize;
        let mapped = self.buckets[..count].iter().filter(|bucket| bucket.mapped()).count() as u32;
        self.scan.mapped = mapped;
        self.scan.complete = true;
        self.scan.scans += 1;
        if mapped == 0 {
            return;
        }
        let p25 = self.percentile(25);
        self.scan.p25_ns = p25;
        self.scan.p50_ns = self.percentile(50);
        self.scan.p90_ns = self.percentile(90);
        if mapped < MIN_BASELINE {
            return;
        }
        let threshold = u64::from(p25) * STALE_RATIO;
        let mut stale = 0;
        for bucket in self.buckets[..count].iter_mut() {
            bucket.stale = bucket.mapped() && !bucket.refreshed && u64::from(bucket.median_ns) >= threshold;
            stale += u32::from(bucket.stale);
        }
        self.scan.stale = stale;
    }

    /// The smallest mapped median with at least `percent` of the mapped
    /// medians at or below it. A binary search over the value range rather
    /// than a sort: the books stay in place and no bucket-sized scratch
    /// array lands on a stack the owner may have little of.
    fn percentile(&self, percent: u64) -> u32 {
        let medians = || {
            self.buckets[..self.shape.buckets as usize]
                .iter()
                .filter(|bucket| bucket.mapped())
                .map(|bucket| bucket.median_ns)
        };
        let total = medians().count() as u64;
        let need = (total * percent).div_ceil(100).max(1);
        let (mut low, mut high) = (0u32, u32::MAX);
        while low < high {
            let mid = low + (high - low) / 2;
            if medians().filter(|&median| median <= mid).count() as u64 >= need {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        low
    }

    // ── refresh ─────────────────────────────────────────────────────────

    fn stage_refresh(&mut self) -> bool {
        let Work::Refresh(mut cursor) = self.work else {
            return false;
        };
        let mut staged = false;
        // Restage chunks whose last step completed (or was refused).
        for index in 0..SLOTS {
            if let Some(chunk) = self.slots[index].chunk
                && !chunk.in_flight
            {
                staged |= self.stage_chunk(index, chunk);
            }
        }
        if cursor.flushing {
            return staged;
        }
        // Take new chunks: the current bucket's next, else the next bucket.
        loop {
            if cursor.current.is_none() {
                while cursor.next < self.shape.buckets && !self.wanted(cursor.plan, cursor.next) {
                    cursor.next += 1;
                }
                if cursor.next >= self.shape.buckets {
                    break;
                }
                let (start, blocks) = self.shape.bucket(cursor.next);
                cursor.current = Some((cursor.next, start, start + blocks));
                cursor.next += 1;
            }
            let Some((bucket, lba, end)) = cursor.current else {
                break;
            };
            if lba >= end {
                // Every chunk is taken. Once the last one lands, the flush.
                if self.slots.iter().any(|slot| slot.chunk.is_some()) {
                    break;
                }
                if self.stage(DeviceCommand::Flush, None, FLUSH_TAG) {
                    cursor.flushing = true;
                    staged = true;
                }
                break;
            }
            let Some(free) = self.slots.iter().position(|slot| slot.chunk.is_none()) else {
                break;
            };
            let blocks = (end - lba).min(u64::from(self.shape.chunk_blocks)) as u32;
            let chunk = Chunk {
                bucket,
                lba,
                blocks,
                step: Step::Read,
                attempts: 0,
                in_flight: false,
            };
            self.slots[free].chunk = Some(chunk);
            cursor.current = Some((bucket, lba + u64::from(blocks), end));
            staged |= self.stage_chunk(free, chunk);
        }
        self.work = Work::Refresh(cursor);
        staged
    }

    fn stage_chunk(&mut self, index: usize, mut chunk: Chunk) -> bool {
        let (command, into_check) = chunk.command();
        if !self.stage(command, Some((index, into_check)), index as u64) {
            return false;
        }
        chunk.in_flight = true;
        self.slots[index].chunk = Some(chunk);
        true
    }

    fn flushed(&mut self, result: Result<(), BlockError>) {
        let Work::Refresh(mut cursor) = self.work else {
            return;
        };
        cursor.flushing = false;
        if let Err(error) = result {
            self.work = Work::Refresh(cursor);
            self.fault(error);
            return;
        }
        if let Some((bucket, _, _)) = cursor.current.take() {
            let bucket = &mut self.buckets[bucket as usize];
            bucket.refreshed = true;
            bucket.stale = false;
            self.refresh.buckets_done += 1;
            self.refresh.buckets_lifetime += 1;
        }
        self.work = Work::Refresh(cursor);
    }

    fn chunk_done(&mut self, index: usize, result: Result<(), BlockError>) {
        let Some(mut chunk) = self.slots[index].chunk else {
            return;
        };
        chunk.in_flight = false;
        let slot = &self.slots[index];
        let bytes = (chunk.blocks * self.shape.block_bytes) as usize;
        let next = match (chunk.step, result) {
            (Step::Read, Ok(())) if is_zero(&slot.data.bytes()[..bytes]) => {
                self.refresh.zero_chunks += 1;
                None
            }
            (Step::Read, Ok(())) => Some((Step::Confirm, chunk.attempts)),
            (Step::Read | Step::Confirm, Err(_)) => {
                self.refresh.read_errors += 1;
                self.buckets[chunk.bucket as usize].incomplete = true;
                None
            }
            (Step::Confirm, Ok(())) if slot.data.bytes()[..bytes] == slot.check.bytes()[..bytes] => {
                Some((Step::Write, 0))
            }
            // Two reads of the same blocks disagreed: the device is not
            // giving one answer, so there is nothing safe to write. Read
            // both again once; then leave the chunk exactly as it is.
            (Step::Confirm, Ok(())) if chunk.attempts + 1 < ATTEMPTS => Some((Step::Read, chunk.attempts + 1)),
            (Step::Confirm, Ok(())) => {
                self.refresh.unstable_chunks += 1;
                self.buckets[chunk.bucket as usize].incomplete = true;
                None
            }
            (Step::Write, Ok(())) => Some((Step::Check, chunk.attempts)),
            (Step::Check, Ok(())) if slot.data.bytes()[..bytes] == slot.check.bytes()[..bytes] => {
                self.refresh.bytes += bytes as u64;
                None
            }
            // The write did not stick, or a check after it disagreed. The
            // data buffer still holds the agreed bytes: write them again,
            // once.
            (Step::Write | Step::Check, outcome) if chunk.attempts + 1 < ATTEMPTS => {
                if outcome.is_ok() {
                    self.refresh.mismatches += 1;
                }
                self.refresh.write_retries += 1;
                Some((Step::Write, chunk.attempts + 1))
            }
            (Step::Write | Step::Check, outcome) => {
                if outcome.is_ok() {
                    self.refresh.mismatches += 1;
                }
                self.slots[index].chunk = None;
                self.fault(outcome.err().unwrap_or(BlockError::Integrity));
                return;
            }
        };
        self.slots[index].chunk = next.map(|(step, attempts)| Chunk { step, attempts, ..chunk });
    }

    fn complete(&mut self, completion: DeviceCompletion) {
        if completion.cookie >> TAG_BITS != u64::from(self.generation) {
            self.stale_completions += 1;
            return;
        }
        self.in_flight = self.in_flight.saturating_sub(1);
        match completion.cookie & TAG_MASK {
            PROBE_TAG => self.probed(completion.result),
            FLUSH_TAG => self.flushed(completion.result),
            tag if (tag as usize) < SLOTS => self.chunk_done(tag as usize, completion.result),
            _ => self.stale_completions += 1,
        }
    }

    // ── retirement ──────────────────────────────────────────────────────

    /// Cancel and wait out what the device holds, through the device's own
    /// bounded [`wait`](DirectBlockDevice::wait). `true` once nothing of
    /// the job's is in flight.
    pub fn drain(&mut self) -> bool {
        self.cancel();
        while self.in_flight != 0 {
            let Some(completion) = self.device.wait() else {
                break;
            };
            self.complete(completion);
        }
        if self.in_flight == 0 {
            self.settle();
        }
        self.in_flight == 0
    }

    /// The device back, once [`drain`](Self::drain) has emptied the job.
    /// Refused, the job handed back, while a command is in flight.
    pub fn into_device(self) -> Result<D, Self> {
        if self.in_flight != 0 {
            return Err(self);
        }
        Ok(self.device)
    }
}

/// Whether every byte is zero: an OR-fold, which vectorizes, rather than an
/// early-exit scan, which does not.
fn is_zero(bytes: &[u8]) -> bool {
    // SAFETY: every bit pattern is a valid `u64`; `align_to` hands back
    // only whole, aligned words between the unaligned head and tail.
    let (head, words, tail) = unsafe { bytes.align_to::<u64>() };
    let fold = words.iter().fold(0u64, |acc, word| acc | word);
    fold == 0 && head.iter().chain(tail).all(|byte| *byte == 0)
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::boxed::Box;
    use std::collections::VecDeque;
    use std::vec::Vec;

    use super::*;
    use crate::block::memory::{MemoryBlockDevice, MemoryBuffer};

    const BLOCK: u32 = 512;
    /// 512 KiB region of 512-byte blocks; 8 KiB buffers, so 8 KiB chunks.
    const DISK: usize = 512 << 10;
    const BUF: usize = 8 << 10;
    const BLOCKS: u64 = (DISK / BLOCK as usize) as u64;
    /// The shape these sizes give: eight buckets of 64 KiB (the floor of
    /// sixteen 4 KiB probe pages), eight chunks each.
    const BUCKETS: u32 = 8;
    const BUCKET_BLOCKS: u64 = 128;
    const CHUNK_BLOCKS: u64 = 16;
    const CACHED_NS: u64 = 47_000;
    const FOLDED_NS: u64 = 113_000;
    const DECAYED_NS: u64 = 900_000;
    /// What a block inside `cold` costs on its first read only: the cold
    /// mapping cache Biggertron's first scan after boot paid, and anything
    /// else that slows a read once and not again.
    const COLD_NS: u64 = 1_500_000;

    /// A drive model over the memory device: every block starts folded;
    /// blocks inside `decayed` read slower; a block written since the model
    /// was built reads from the cache. The clock advances by each
    /// completion's latency as it is polled.
    struct Drive {
        inner: Box<MemoryBlockDevice<DISK, BUF>>,
        now: u64,
        pending: VecDeque<u64>,
        decayed: core::ops::Range<u64>,
        cold: core::ops::Range<u64>,
        touched: Vec<bool>,
        written: Vec<bool>,
        /// Reads starting at this block come back with their first byte
        /// changed — differently each time — this many more times.
        flaky: Option<(u64, u32)>,
        flips: u8,
        /// Writes starting at this block land with a byte flipped on the
        /// medium, this many more times.
        corrupt: Option<(u64, u32)>,
        /// Writes starting at this block fail, this many more times.
        refuse: Option<(u64, u32)>,
        flushes: u64,
    }

    fn take(target: &mut Option<(u64, u32)>, lba: u64) -> bool {
        match target {
            Some((at, left)) if *at == lba && *left > 0 => {
                *left -= 1;
                true
            }
            _ => false,
        }
    }

    impl Drive {
        fn new() -> Self {
            Self {
                inner: Box::new(MemoryBlockDevice::new(BLOCK, 1024).unwrap()),
                now: 1_000_000,
                pending: VecDeque::new(),
                decayed: 0..0,
                cold: 0..0,
                touched: std::vec![false; BLOCKS as usize],
                written: std::vec![false; BLOCKS as usize],
                flaky: None,
                flips: 0,
                corrupt: None,
                refuse: None,
                flushes: 0,
            }
        }

        /// Fill blocks `[0, end)` with a pattern of non-zero bytes; the
        /// rest stays zero, the never-written posture.
        fn fill(&mut self, end: u64) {
            let mut buffer = self.inner.allocate_buffer(BLOCK as usize).unwrap();
            for lba in 0..end.min(BLOCKS) {
                for (i, byte) in buffer.bytes_mut().iter_mut().enumerate() {
                    *byte = (mix(lba * 512 + i as u64) as u8) | 1;
                }
                let write = DeviceCommand::Write { lba, blocks: 1, buffer_offset: 0 };
                self.inner.submit(write, Some(&mut buffer), 0).unwrap();
                self.inner.poll().unwrap().result.unwrap();
            }
        }

        /// The medium's bytes, read around the model.
        fn image(&mut self) -> Vec<u8> {
            let mut out = Vec::with_capacity(DISK);
            let mut buffer = self.inner.allocate_buffer(BUF).unwrap();
            let per = (BUF / BLOCK as usize) as u32;
            for lba in (0..BLOCKS).step_by(per as usize) {
                let read = DeviceCommand::Read { lba, blocks: per, buffer_offset: 0 };
                self.inner.submit(read, Some(&mut buffer), 0).unwrap();
                self.inner.poll().unwrap().result.unwrap();
                out.extend_from_slice(buffer.bytes());
            }
            out
        }

        fn latency(&mut self, lba: u64, blocks: u32) -> u64 {
            let end = lba + u64::from(blocks);
            let first = !self.touched[lba as usize];
            self.touched[lba as usize..end as usize].fill(true);
            let cold = if first && self.cold.contains(&lba) { COLD_NS } else { 0 };
            cold + if self.written[lba as usize..end as usize].iter().all(|written| *written) {
                CACHED_NS
            } else if self.decayed.contains(&lba) {
                DECAYED_NS
            } else {
                FOLDED_NS
            }
        }

        fn written_in(&self, range: core::ops::Range<u64>) -> usize {
            self.written[range.start as usize..range.end as usize].iter().filter(|w| **w).count()
        }
    }

    impl DirectBlockDevice for Drive {
        type Buffer = MemoryBuffer<BUF>;
        const INDEPENDENT_PROGRESS: bool = true;

        fn geometry(&self) -> BlockGeometry {
            self.inner.geometry()
        }
        fn max_buffer_bytes(&self) -> usize {
            BUF
        }
        fn clock_ns(&self) -> Option<u64> {
            Some(self.now)
        }
        fn allocate_buffer(&mut self, bytes: usize) -> Result<Self::Buffer, BlockError> {
            self.inner.allocate_buffer(bytes)
        }
        fn submit(
            &mut self,
            command: DeviceCommand,
            mut buffer: Option<&mut Self::Buffer>,
            cookie: u64,
        ) -> Result<(), BlockError> {
            let latency = match command {
                DeviceCommand::Read { lba, blocks, .. } => self.latency(lba, blocks),
                _ => CACHED_NS,
            };
            match command {
                DeviceCommand::Write { lba, .. } if take(&mut self.refuse, lba) => {
                    self.inner.inject_completion(cookie, Err(BlockError::Media))?;
                }
                _ => self.inner.submit(command, buffer.as_deref_mut(), cookie)?,
            }
            match command {
                DeviceCommand::Read { lba, buffer_offset, .. } if take(&mut self.flaky, lba) => {
                    self.flips = self.flips.wrapping_add(1).max(1);
                    buffer.unwrap().bytes_mut()[buffer_offset] ^= self.flips;
                }
                DeviceCommand::Write { lba, blocks, .. } => {
                    for block in lba..lba + u64::from(blocks) {
                        self.written[block as usize] = true;
                    }
                    if take(&mut self.corrupt, lba) {
                        self.inner.corrupt_byte((lba * u64::from(BLOCK)) as usize, 0x5a).unwrap();
                    }
                }
                DeviceCommand::Flush => self.flushes += 1,
                _ => {}
            }
            self.pending.push_back(latency);
            Ok(())
        }
        fn poll(&mut self) -> Option<DeviceCompletion> {
            let completion = self.inner.poll()?;
            self.now += self.pending.pop_front().unwrap_or(0);
            Some(completion)
        }
        fn quiesce(&mut self) -> Result<(), BlockError> {
            self.inner.quiesce()
        }
    }

    /// The drive model's medium and the job's chunk buffers are inline
    /// arrays; a debug build moves them by value more than once, past a
    /// test thread's default stack.
    fn on_big_stack(body: fn()) {
        std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn(body)
            .expect("spawn")
            .join()
            .expect("the test body passed");
    }

    fn run(job: &mut RegionRefresh<'_, Drive>) {
        for _ in 0..100_000 {
            if job.advance(true) == Progress::Idle {
                return;
            }
        }
        panic!("the job never went idle");
    }

    fn table() -> Vec<Bucket> {
        std::vec![Bucket::default(); 16]
    }

    fn build(drive: Drive, books: &mut [Bucket]) -> RegionRefresh<'_, Drive> {
        RegionRefresh::new(drive, books).ok().expect("the job builds")
    }

    fn marked(job: &RegionRefresh<'_, Drive>, pick: impl Fn(&Bucket) -> bool) -> Vec<usize> {
        job.buckets().iter().enumerate().filter(|(_, b)| pick(b)).map(|(i, _)| i).collect()
    }

    fn bucket(index: u64) -> core::ops::Range<u64> {
        index * BUCKET_BLOCKS..(index + 1) * BUCKET_BLOCKS
    }

    #[test]
    fn shape_buckets_the_region_in_whole_chunks() {
        on_big_stack(shape_buckets_the_region_in_whole_chunks_body);
    }

    fn shape_buckets_the_region_in_whole_chunks_body() {
        let mut books = table();
        let job = build(Drive::new(), &mut books);
        let shape = job.shape();
        assert_eq!(shape.probe_blocks, 8);
        assert_eq!(u64::from(shape.chunk_blocks), CHUNK_BLOCKS);
        assert_eq!(shape.bucket_blocks, BUCKET_BLOCKS);
        assert_eq!(shape.buckets, BUCKETS);
        // Every probe lands inside its own bucket, on a page boundary, and
        // no two probes of one scan share a page.
        for index in 0..shape.buckets {
            let (start, blocks) = shape.bucket(index);
            let mut seen = Vec::new();
            for probe in 0..PROBES {
                let lba = shape.probe_lba(index, probe, 3);
                assert!(lba >= start && lba + u64::from(shape.probe_blocks) <= start + blocks);
                assert_eq!(lba % u64::from(shape.probe_blocks), 0);
                assert!(!seen.contains(&lba));
                seen.push(lba);
            }
        }
    }

    #[test]
    fn a_short_region_keeps_a_short_last_bucket() {
        on_big_stack(a_short_region_keeps_a_short_last_bucket_body);
    }

    fn a_short_region_keeps_a_short_last_bucket_body() {
        let geometry = BlockGeometry {
            block_count: 1000,
            ..Drive::new().geometry()
        };
        let shape = Shape::new(geometry, BUF, 4).unwrap();
        assert_eq!(shape.bucket_blocks, 256);
        assert_eq!(shape.buckets, 4);
        let (start, blocks) = shape.bucket(3);
        assert_eq!((start, blocks), (768, 232));
        // A bucket table smaller than the region asks for wider buckets.
        let wide = Shape::new(geometry, BUF, 2).unwrap();
        assert_eq!((wide.bucket_blocks, wide.buckets), (512, 2));
        // A probe page larger than the buffer cannot be chunked.
        assert_eq!(Shape::new(geometry, 2048, 4), Err(BlockError::TooLarge));
    }

    #[test]
    fn folded_buckets_are_healthy_and_decayed_ones_are_stale() {
        on_big_stack(folded_buckets_are_healthy_and_decayed_ones_are_stale_body);
    }

    fn folded_buckets_are_healthy_and_decayed_ones_are_stale_body() {
        let mut drive = Drive::new();
        drive.fill(BLOCKS);
        drive.decayed = bucket(5).start..bucket(6).end;
        let mut books = table();
        let mut job = build(drive, &mut books);
        job.scan().unwrap();
        run(&mut job);
        assert_eq!(job.outcome(), Outcome::Scanned);
        let scan = job.scan_summary();
        assert_eq!((scan.scanned, scan.mapped, scan.stale, scan.scans), (8, 8, 2, 1));
        assert_eq!(u64::from(scan.p25_ns), FOLDED_NS);
        assert_eq!(u64::from(scan.p90_ns), DECAYED_NS);
        assert_eq!(marked(&job, |b| b.stale), [5, 6]);
    }

    #[test]
    fn slowness_that_does_not_repeat_is_not_decay() {
        on_big_stack(slowness_that_does_not_repeat_is_not_decay_body);
    }

    fn slowness_that_does_not_repeat_is_not_decay_body() {
        let mut drive = Drive::new();
        drive.fill(BLOCKS);
        // Buckets 1 and 4 read slowly once, as Biggertron's did right after
        // boot; bucket 6 has decayed and reads slowly every time.
        drive.cold = bucket(1).start..bucket(5).end;
        drive.decayed = bucket(6);
        let mut books = table();
        let mut job = build(drive, &mut books);
        job.scan().unwrap();
        run(&mut job);
        let scan = job.scan_summary();
        assert_eq!((scan.pass, scan.scanned), (PASSES, BUCKETS));
        assert_eq!(marked(&job, |b| b.stale), [6]);
        // Every cold bucket is judged by its warm passes.
        assert!(job.buckets()[1..5].iter().all(|b| u64::from(b.median_ns) == FOLDED_NS));
        assert_eq!(u64::from(scan.p25_ns), FOLDED_NS);
    }

    #[test]
    fn a_cached_quartile_does_not_condemn_folded_buckets() {
        on_big_stack(a_cached_quartile_does_not_condemn_folded_buckets_body);
    }

    fn a_cached_quartile_does_not_condemn_folded_buckets_body() {
        let mut drive = Drive::new();
        drive.fill(BLOCKS);
        // Three buckets sit in the cache: the quartile is cached.
        drive.written[..3 * BUCKET_BLOCKS as usize].fill(true);
        let mut books = table();
        let mut job = build(drive, &mut books);
        job.scan().unwrap();
        run(&mut job);
        let scan = job.scan_summary();
        assert_eq!(u64::from(scan.p25_ns), CACHED_NS);
        assert_eq!(scan.stale, 0, "folded is 2.4x cached, under the 4x ratio");
    }

    #[test]
    fn never_written_buckets_are_unmapped_and_never_judged() {
        on_big_stack(never_written_buckets_are_unmapped_and_never_judged_body);
    }

    fn never_written_buckets_are_unmapped_and_never_judged_body() {
        let mut drive = Drive::new();
        drive.fill(6 * BUCKET_BLOCKS);
        drive.decayed = bucket(2);
        let mut books = table();
        let mut job = build(drive, &mut books);
        job.scan().unwrap();
        run(&mut job);
        let scan = job.scan_summary();
        assert_eq!(scan.mapped, 6);
        assert!(job.buckets()[6..].iter().all(|b| !b.mapped() && !b.stale));
        assert_eq!(marked(&job, |b| b.stale), [2]);
    }

    #[test]
    fn too_few_mapped_buckets_judge_nothing() {
        on_big_stack(too_few_mapped_buckets_judge_nothing_body);
    }

    fn too_few_mapped_buckets_judge_nothing_body() {
        let mut drive = Drive::new();
        drive.fill(3 * BUCKET_BLOCKS);
        drive.decayed = bucket(0);
        let mut books = table();
        let mut job = build(drive, &mut books);
        job.scan().unwrap();
        run(&mut job);
        assert_eq!(job.scan_summary().mapped, 3);
        assert_eq!(job.scan_summary().stale, 0);
    }

    #[test]
    fn refresh_rewrites_only_stale_buckets_and_preserves_every_byte() {
        on_big_stack(refresh_rewrites_only_stale_buckets_and_preserves_every_byte_body);
    }

    fn refresh_rewrites_only_stale_buckets_and_preserves_every_byte_body() {
        let mut drive = Drive::new();
        drive.fill(7 * BUCKET_BLOCKS);
        drive.decayed = bucket(3).start..bucket(4).end;
        let before = drive.image();
        let mut books = table();
        let mut job = build(drive, &mut books);
        assert_eq!(job.refresh_stale(), Err(Refusal::NotScanned));
        job.scan().unwrap();
        run(&mut job);
        job.refresh_stale().unwrap();
        run(&mut job);
        assert_eq!(job.outcome(), Outcome::Refreshed);
        let refresh = job.refresh_summary();
        assert_eq!((refresh.buckets_planned, refresh.buckets_done), (2, 2));
        assert_eq!(refresh.bytes, 2 * BUCKET_BLOCKS * u64::from(BLOCK));
        assert_eq!(refresh.mismatches + refresh.unstable_chunks + refresh.read_errors, 0);
        assert_eq!(marked(&job, |b| b.refreshed), [3, 4]);
        assert!(marked(&job, |b| b.stale).is_empty());
        let mut drive = job.into_device().ok().unwrap();
        // One flush per bucket; nothing outside the two buckets written.
        assert_eq!(drive.flushes, 2);
        assert_eq!(drive.written_in(0..bucket(3).start) + drive.written_in(bucket(5).start..BLOCKS), 0);
        assert_eq!(drive.written_in(bucket(3).start..bucket(5).start), 2 * BUCKET_BLOCKS as usize);
        assert!(drive.image() == before, "a refresh changed a byte");
    }

    #[test]
    fn a_rescan_after_refresh_finds_the_bucket_cached() {
        on_big_stack(a_rescan_after_refresh_finds_the_bucket_cached_body);
    }

    fn a_rescan_after_refresh_finds_the_bucket_cached_body() {
        let mut drive = Drive::new();
        drive.fill(BLOCKS);
        drive.decayed = bucket(0);
        let mut books = table();
        let mut job = build(drive, &mut books);
        job.scan().unwrap();
        run(&mut job);
        job.refresh_stale().unwrap();
        run(&mut job);
        job.scan().unwrap();
        run(&mut job);
        assert_eq!(u64::from(job.buckets()[0].median_ns), CACHED_NS);
        assert_eq!(job.scan_summary().stale, 0);
        assert_eq!(job.scan_summary().scans, 2);
    }

    #[test]
    fn a_bucket_is_rewritten_at_most_once_per_job() {
        on_big_stack(a_bucket_is_rewritten_at_most_once_per_job_body);
    }

    fn a_bucket_is_rewritten_at_most_once_per_job_body() {
        let mut drive = Drive::new();
        drive.fill(BLOCKS);
        let mut books = table();
        let mut job = build(drive, &mut books);
        job.refresh_range(2, 3).unwrap();
        run(&mut job);
        assert_eq!(job.refresh_summary().buckets_done, 3);
        job.refresh_range(0, 6).unwrap();
        run(&mut job);
        let refresh = job.refresh_summary();
        assert_eq!((refresh.buckets_planned, refresh.buckets_done, refresh.buckets_skipped), (3, 3, 3));
        assert_eq!(refresh.buckets_lifetime, 6);
        assert_eq!(job.refresh_range(7, 2), Err(Refusal::Range));
        assert_eq!(job.refresh_range(0, 0), Err(Refusal::Range));
        assert_eq!(job.refresh_range(u32::MAX, 2), Err(Refusal::Range));
    }

    #[test]
    fn zero_chunks_are_never_written() {
        on_big_stack(zero_chunks_are_never_written_body);
    }

    fn zero_chunks_are_never_written_body() {
        let mut drive = Drive::new();
        // Bucket 3 is half data, half zeros.
        drive.fill(bucket(3).start + BUCKET_BLOCKS / 2);
        let mut books = table();
        let mut job = build(drive, &mut books);
        job.refresh_range(3, 1).unwrap();
        run(&mut job);
        let refresh = job.refresh_summary();
        assert_eq!(refresh.zero_chunks, 4);
        assert_eq!(refresh.bytes, BUCKET_BLOCKS / 2 * u64::from(BLOCK));
        let drive = job.into_device().ok().unwrap();
        assert_eq!(drive.written_in(bucket(3).start + BUCKET_BLOCKS / 2..bucket(3).end), 0);
    }

    #[test]
    fn a_read_that_disagrees_once_is_read_again() {
        on_big_stack(a_read_that_disagrees_once_is_read_again_body);
    }

    fn a_read_that_disagrees_once_is_read_again_body() {
        let mut drive = Drive::new();
        drive.fill(BLOCKS);
        drive.flaky = Some((0, 1));
        let before = drive.image();
        let mut books = table();
        let mut job = build(drive, &mut books);
        job.refresh_range(0, 1).unwrap();
        run(&mut job);
        assert_eq!(job.refresh_summary().unstable_chunks, 0);
        assert_eq!(job.refresh_summary().bytes, BUCKET_BLOCKS * u64::from(BLOCK));
        let mut drive = job.into_device().ok().unwrap();
        assert!(drive.image() == before);
    }

    #[test]
    fn reads_that_never_agree_leave_the_chunk_untouched() {
        on_big_stack(reads_that_never_agree_leave_the_chunk_untouched_body);
    }

    fn reads_that_never_agree_leave_the_chunk_untouched_body() {
        let mut drive = Drive::new();
        drive.fill(BLOCKS);
        let first = bucket(1).start;
        drive.flaky = Some((first, u32::MAX));
        let before = drive.image();
        let mut books = table();
        let mut job = build(drive, &mut books);
        job.refresh_range(1, 1).unwrap();
        run(&mut job);
        let refresh = job.refresh_summary();
        assert_eq!(refresh.unstable_chunks, 1);
        assert!(job.buckets()[1].incomplete && job.buckets()[1].refreshed);
        assert_eq!(refresh.bytes, (BUCKET_BLOCKS - CHUNK_BLOCKS) * u64::from(BLOCK));
        let mut drive = job.into_device().ok().unwrap();
        assert_eq!(drive.written_in(first..first + CHUNK_BLOCKS), 0, "an unstable chunk was written");
        assert!(drive.image() == before);
    }

    #[test]
    fn a_write_that_does_not_stick_is_written_again() {
        on_big_stack(a_write_that_does_not_stick_is_written_again_body);
    }

    fn a_write_that_does_not_stick_is_written_again_body() {
        let mut drive = Drive::new();
        drive.fill(BLOCKS);
        drive.corrupt = Some((0, 1));
        let before = drive.image();
        let mut books = table();
        let mut job = build(drive, &mut books);
        job.refresh_range(0, 1).unwrap();
        run(&mut job);
        let refresh = job.refresh_summary();
        assert_eq!(job.outcome(), Outcome::Refreshed);
        assert_eq!((refresh.mismatches, refresh.write_retries), (1, 1));
        let mut drive = job.into_device().ok().unwrap();
        assert!(drive.image() == before, "the second write restored the agreed bytes");
    }

    #[test]
    fn a_write_that_never_sticks_faults_the_job() {
        on_big_stack(a_write_that_never_sticks_faults_the_job_body);
    }

    fn a_write_that_never_sticks_faults_the_job_body() {
        let mut drive = Drive::new();
        drive.fill(BLOCKS);
        drive.corrupt = Some((bucket(1).start, 2));
        let mut books = table();
        let mut job = build(drive, &mut books);
        job.refresh_range(1, 1).unwrap();
        run(&mut job);
        assert_eq!(job.outcome(), Outcome::Faulted(BlockError::Integrity));
        assert!(!job.buckets()[1].refreshed);
        assert_eq!(job.refresh_summary().mismatches, 2);
    }

    #[test]
    fn a_refused_write_is_retried_then_faults() {
        on_big_stack(a_refused_write_is_retried_then_faults_body);
    }

    fn a_refused_write_is_retried_then_faults_body() {
        let mut drive = Drive::new();
        drive.fill(BLOCKS);
        drive.refuse = Some((0, 1));
        let mut books = table();
        let mut job = build(drive, &mut books);
        job.refresh_range(0, 1).unwrap();
        run(&mut job);
        assert_eq!(job.outcome(), Outcome::Refreshed);
        assert_eq!(job.refresh_summary().write_retries, 1);

        let mut drive = job.into_device().ok().unwrap();
        drive.refuse = Some((bucket(2).start, 2));
        let mut books = table();
        let mut job = build(drive, &mut books);
        job.refresh_range(2, 1).unwrap();
        run(&mut job);
        assert_eq!(job.outcome(), Outcome::Faulted(BlockError::Media));
        assert!(!job.buckets()[2].refreshed);
    }

    #[test]
    fn a_failed_read_leaves_its_chunk_and_the_bucket_completes() {
        on_big_stack(a_failed_read_leaves_its_chunk_and_the_bucket_completes_body);
    }

    fn a_failed_read_leaves_its_chunk_and_the_bucket_completes_body() {
        let mut drive = Drive::new();
        drive.fill(BLOCKS);
        // The job's first command is chunk zero's read.
        drive.inner.inject(BlockError::Media);
        let mut books = table();
        let mut job = build(drive, &mut books);
        job.refresh_range(0, 1).unwrap();
        run(&mut job);
        assert_eq!(job.outcome(), Outcome::Refreshed);
        assert_eq!(job.refresh_summary().read_errors, 1);
        assert!(job.buckets()[0].incomplete && job.buckets()[0].refreshed);
        let drive = job.into_device().ok().unwrap();
        assert_eq!(drive.written_in(0..CHUNK_BLOCKS), 0);
    }

    #[test]
    fn probes_wait_for_an_idle_device_and_cancel_drains() {
        on_big_stack(probes_wait_for_an_idle_device_and_cancel_drains_body);
    }

    fn probes_wait_for_an_idle_device_and_cancel_drains_body() {
        let mut drive = Drive::new();
        drive.fill(BLOCKS);
        let mut books = table();
        let mut job = build(drive, &mut books);
        job.scan().unwrap();
        assert_eq!(job.advance(false), Progress::Waiting);
        assert_eq!(job.in_flight(), 0);
        assert_eq!(job.advance(true), Progress::InFlight);
        assert_eq!(job.scan(), Err(Refusal::Busy));
        job.cancel();
        assert_eq!(job.phase(), Phase::Draining);
        assert_eq!(job.advance(true), Progress::Idle);
        assert_eq!(job.outcome(), Outcome::Cancelled);
        assert!(!job.scan_summary().complete);
        // A cancelled job drains the device and hands it back.
        job.refresh_range(0, 2).unwrap();
        assert_eq!(job.advance(true), Progress::InFlight);
        assert!(job.drain());
        assert_eq!(job.outcome(), Outcome::Cancelled);
        assert!(job.into_device().is_ok());
    }

    #[test]
    fn is_zero_folds_every_alignment() {
        let mut bytes = [0u8; 67];
        for start in 0..8 {
            assert!(is_zero(&bytes[start..]));
        }
        for at in [0, 3, 9, 33, 66] {
            bytes[at] = 1;
            assert!(!is_zero(&bytes));
            bytes[at] = 0;
        }
    }
}
