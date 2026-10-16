//! Hardware-neutral, affine direct block channels.
//!
//! A [`BlockChannel`] is the data-plane authority provisioned by the storage
//! control service.  It owns one backend queue, its registered buffers, and
//! its bounded request/correlation table.  It is deliberately neither
//! `Clone` nor `Copy`, and it is not `Sync`: one worker owns and drives one
//! channel, matching the SPDK qpair contract without a lock.
//!
//! No PCI, NVMe, MMIO, IOVA, or raw queue-entry type appears in this module.
//! Hardware drivers implement [`DirectBlockDevice`] below the boundary.
//!
//! This is a **driver class contract**, so it lives in the driver substrate
//! beside the seam it is written against rather than as a sibling crate.  It
//! used to be `kore-block`, which the substrate then had to depend on for
//! [`CompletionPacer`] — a cycle between two halves of one thing.  Folding it
//! in removed that, and put its tests inside a suite `make test` actually
//! runs.

use core::{cell::Cell, marker::PhantomData};

use kore_abi::monitor::FilesystemLatencySnapshot;

pub mod memory;
pub mod region;

pub use region::RegionDevice;

/// Maximum buffers registered in one direct channel by default.
pub const DEFAULT_BUFFERS: usize = 8;
/// Maximum requests in flight in one direct channel by default.
pub const DEFAULT_OUTSTANDING: usize = 32;

/// Immutable geometry and durability facts established while provisioning.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlockIoHints {
    /// Preferred write granularity in logical blocks (0 = unspecified).
    pub preferred_write_blocks: u32,
    /// Preferred write alignment in logical blocks (0 = unspecified).
    pub write_alignment_blocks: u32,
    /// Optimal write command size in logical blocks (0 = unspecified).
    pub optimal_write_blocks: u32,
    /// Optimal I/O boundary in logical blocks (0 = unspecified).
    pub optimal_boundary_blocks: u32,
    /// Every logical block size the device could be formatted to, as a
    /// mask: bit `k` = `512 << k` bytes (0 = unknown).  Descriptive only —
    /// changing the format is a destructive operator action, never a
    /// hint the channel acts on.
    pub supported_block_sizes: u32,
}

impl BlockIoHints {
    pub const NONE: Self = Self {
        preferred_write_blocks: 0,
        write_alignment_blocks: 0,
        optimal_write_blocks: 0,
        optimal_boundary_blocks: 0,
        supported_block_sizes: 0,
    };
}

/// Immutable geometry, durability facts, and optional performance guidance
/// established while provisioning. Hint counts are normalized to actual
/// logical blocks at the device boundary; zero consistently means absent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockGeometry {
    pub block_size: u32,
    pub block_count: u64,
    pub max_transfer_blocks: u32,
    pub atomic_write_blocks: u32,
    pub flush: bool,
    pub discard: bool,
    pub read_only: bool,
    pub io_hints: BlockIoHints,
}

/// Protocol-neutral physical-device health sampled on the control plane.
///
/// This is deliberately not part of [`BlockGeometry`]: geometry is immutable
/// provisioning state and safe to cache, while health is a fallible live
/// device query. Counters saturate at the low 64 bits when a protocol carries
/// wider values; the status surface is diagnostic, not an accounting ledger.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlockHealth {
    pub critical_warning: u8,
    pub temperature_celsius: i16,
    pub available_spare_percent: u8,
    pub spare_threshold_percent: u8,
    pub percentage_used: u8,
    pub power_cycles: u64,
    pub power_on_hours: u64,
    pub unsafe_shutdowns: u64,
    pub media_errors: u64,
}

impl BlockGeometry {
    /// Validate device-provided geometry before any range arithmetic uses it.
    pub const fn validate(self) -> Result<Self, BlockError> {
        if self.block_size == 0
            || !self.block_size.is_power_of_two()
            || self.block_count == 0
            || self.max_transfer_blocks == 0
            || self.atomic_write_blocks == 0
            || self.atomic_write_blocks > self.max_transfer_blocks
        {
            return Err(BlockError::Geometry);
        }
        if self
            .block_count
            .checked_mul(self.block_size as u64)
            .is_none()
        {
            return Err(BlockError::Geometry);
        }
        Ok(self)
    }

    pub const fn bytes(self) -> u64 {
        self.block_count * self.block_size as u64
    }
}

/// One registered buffer's generational, channel-local name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferId {
    slot: u16,
    generation: u16,
}

impl BufferId {
    pub const fn to_bits(self) -> u32 {
        ((self.generation as u32) << 16) | self.slot as u32
    }

    /// The inverse of [`to_bits`](Self::to_bits), for a name that crossed a
    /// process boundary as data.  It conveys nothing by itself: every channel
    /// operation re-checks the slot's live generation, so a forged or stale
    /// word answers `UnknownBuffer`.
    pub const fn from_bits(bits: u32) -> Self {
        Self {
            slot: bits as u16,
            generation: (bits >> 16) as u16,
        }
    }
}

/// Caller-selected correlation identity, unique while outstanding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestId(pub u64);

/// Channel lifecycle.  New work is accepted only while `Online`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelState {
    Online,
    ReadOnly,
    Draining,
    Lost,
    Revoked,
}

/// Completion pacing selected when the channel is provisioned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompletionMode {
    Poll,
    Interrupt,
}

/// Hardware-neutral operation submitted to a direct backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceCommand {
    Read {
        lba: u64,
        blocks: u32,
        buffer_offset: usize,
    },
    Write {
        lba: u64,
        blocks: u32,
        buffer_offset: usize,
    },
    Flush,
    Discard {
        lba: u64,
        blocks: u32,
    },
}

/// Typed data-plane failures.  No filesystem or device-protocol vocabulary
/// leaks through this boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockError {
    Geometry,
    Range,
    Alignment,
    TooLarge,
    ReadOnly,
    Backpressure,
    UnknownBuffer,
    BufferBusy,
    UnknownRequest,
    StaleCompletion,
    Cancelled,
    Revoked,
    Draining,
    Lost,
    Media,
    Transport,
    Integrity,
    Unsupported,
}

/// What a backend can say about its own liveness **without issuing a
/// command**.
///
/// This is deliberately not [`BlockHealth`]: reading SMART data requires the
/// command path to be working, which is precisely what is in doubt when a
/// completion has not arrived.  A liveness probe has to be an out-of-band
/// read — a status register, a link state — or it cannot answer the question
/// being asked.
///
/// It exists because **a wait that expired is not evidence a device is
/// dead.**  A completion can be late because the device failed, but also
/// because its queue is deep, because it is garbage-collecting or thermally
/// throttled, or because the host descheduled the waiter.  Only the first
/// justifies retiring a channel, and only the device can distinguish them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Liveness {
    /// The backend reports itself functional out of band.  A late completion
    /// here means slow, not dead.
    Healthy,
    /// The backend reports a fatal fault — positive evidence, the only kind
    /// that justifies declaring a channel lost.
    Failed,
    /// The backend has no out-of-band liveness signal, so nothing can be
    /// concluded either way.  Callers must keep whatever conservative
    /// posture they had before asking.
    Unknown,
}

/// One device completion.  The cookie is opaque to the backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceCompletion {
    pub cookie: u64,
    pub result: Result<(), BlockError>,
}

/// A completion returned to the channel owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Completion {
    pub request: RequestId,
    pub result: Result<(), BlockError>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChannelStats {
    pub submissions: u64,
    pub completions: u64,
    pub cancellations: u64,
    pub backpressure: u64,
    pub poll_calls: u64,
    /// Waits that reached the backend's pacing seam after an empty fast-path
    /// poll. Not itself a park count — see the backend's own route
    /// statistics for whether the interrupt actually delivered.
    pub device_waits: u64,
    pub max_outstanding: u16,
    pub registered_bytes: usize,
    pub peak_registered_bytes: usize,
    /// Completion latency at the depth the channel drove — measured only
    /// when the backend publishes a clock ([`DirectBlockDevice::clock_ns`]).
    pub latency: LatencyBooks,
}

/// Completion-latency books: a base-2 histogram over nanoseconds plus the
/// moments that turn it into "how fast is this device at the depth we
/// drive it" — the channel-side twin of
/// `kore_abi::monitor::FilesystemLatencySnapshot`, whose bucket geometry
/// this shares so the two never disagree about what a bucket means.
///
/// Every field saturates: a pinned bucket says "many", never a wrong
/// number.  Zero `samples` is the unmeasured posture — a backend with no
/// clock records nothing rather than guessing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LatencyBooks {
    /// Completions whose latency was measured.
    pub samples: u64,
    /// Sum of measured latencies, nanoseconds.
    pub sum_ns: u64,
    /// Slowest measured completion, nanoseconds.
    pub max_ns: u64,
    /// Sum over samples of the commands already in flight when each was
    /// submitted — `depth_sum / samples` is the mean queue depth the
    /// latencies were measured at.
    pub depth_sum: u64,
    /// `buckets[i]` counts latencies below the bucket's upper bound; the
    /// last bucket is open-ended.  Geometry:
    /// [`FilesystemLatencySnapshot::bucket_index`].
    pub buckets: [u32; FilesystemLatencySnapshot::BUCKETS],
}

impl Default for LatencyBooks {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl LatencyBooks {
    pub const EMPTY: Self = Self {
        samples: 0,
        sum_ns: 0,
        max_ns: 0,
        depth_sum: 0,
        buckets: [0; FilesystemLatencySnapshot::BUCKETS],
    };

    /// Book one completion that took `latency_ns`, submitted while `depth`
    /// other commands were already in flight.
    pub fn record(&mut self, latency_ns: u64, depth: u16) {
        self.samples = self.samples.saturating_add(1);
        self.sum_ns = self.sum_ns.saturating_add(latency_ns);
        self.max_ns = self.max_ns.max(latency_ns);
        self.depth_sum = self.depth_sum.saturating_add(u64::from(depth));
        let bucket = &mut self.buckets[FilesystemLatencySnapshot::bucket_index(latency_ns)];
        *bucket = bucket.saturating_add(1);
    }

    /// The wire form, with the channel's deepest queue beside it.
    pub fn snapshot(&self, max_outstanding: u16) -> FilesystemLatencySnapshot {
        FilesystemLatencySnapshot::new(
            max_outstanding,
            self.samples,
            self.sum_ns,
            self.max_ns,
            self.depth_sum,
            self.buckets,
        )
    }
}

/// A DMA-suitable registered buffer, abstract above the device boundary.
pub trait BlockBuffer {
    fn len(&self) -> usize;
    fn bytes(&self) -> &[u8];
    fn bytes_mut(&mut self) -> &mut [u8];
}

/// Spin hints one [`SpinPacer`] issues before declaring its budget spent.
///
/// **This is a count of retired instructions, not a duration.** What it costs
/// in wall-clock time moves with core frequency, SMT contention, and whether
/// a hypervisor descheduled the vCPU mid-loop, so it cannot express a chosen
/// tolerance. See [`SpinPacer`] for why that restricts where it may be used.
///
/// # Where the number comes from
///
/// Not a round number chosen for looking reasonable — the derivation, and its
/// assumptions, so a future reader can re-derive it when they stop holding:
///
/// - **Assumed** a spin hint costs order-100 cycles. `core::hint::spin_loop`
///   lowers to `PAUSE` on x86-64, whose latency is tens of cycles on older
///   parts and around 140 on Skylake and later; on aarch64 it lowers to
///   `YIELD`, which is far cheaper and may be a no-op. These are
///   order-of-magnitude assumptions, not measurements on this tree.
/// - **Assumed** a core between 1 and 5 GHz.
///
/// So 1M hints is very roughly 20–100 ms of x86 spinning, and potentially
/// far less on aarch64. That sits above the microseconds a healthy NVMe
/// command needs and below the one-second budget the clocked pacers use — the
/// band a last-resort bound wants.
///
/// The width of that estimate is the point, and is why this bound may only be
/// used where **nothing is concluded from its expiry** (see [`SpinPacer`]).
/// A quantity known only to within a factor of five, varying by architecture,
/// cannot be the basis of a verdict.
pub const MAX_POLLS: u32 = 1_000_000;

/// **The completion wait is a seam.**  Code that has submitted a command
/// polls for its completion, and between empty polls waits however the
/// composed pacer says: the bounded spin ([`SpinPacer`] — host tests, and
/// the polled fallback on a machine that realized no interrupt route), or a
/// real park on the task's held delivery route with an absolute deadline
/// (`kore_driver::host::RoutePacer` over
/// `kore_rt::cap::verbs::wait_interrupt_until` — the CPU is yielded while
/// the command is in flight, and the wake is the device's own vector).
///
/// The contract: `pace` returns `false` when no further waiting is allowed
/// (budget spent, deadline passed, route revoked, park refused).  The wait
/// path then polls **once more** — a completion that landed without a
/// delivered interrupt is still a completion, and it is exactly what a
/// delivery-dead machine has to show for itself — and only then refuses.
pub trait CompletionPacer {
    /// Wait for the next completion-poll opportunity; `false` = budget
    /// exhausted (poll once more, then refuse).
    fn pace(&mut self) -> bool;
}

/// A deterministic counted pacer with a caller-supplied delay hook.
///
/// Driver-core tests use a no-op or scripted hook; early synchronous hardware
/// paths use `spin_loop`. Production paths that make liveness decisions should
/// use an elapsed-time pacer instead.
pub struct CountedPacer<F> {
    remaining: u32,
    delay: F,
}

impl<F> CountedPacer<F> {
    pub const fn new(remaining: u32, delay: F) -> Self {
        Self { remaining, delay }
    }
}

impl<F: FnMut()> CompletionPacer for CountedPacer<F> {
    fn pace(&mut self) -> bool {
        if self.remaining == 0 {
            return false;
        }
        self.remaining -= 1;
        (self.delay)();
        true
    }
}

/// The bounded spin pacer — [`MAX_POLLS`] spin hints, then refusal.
///
/// **Its budget is retired instructions, not time**, so it must not be the
/// input to a liveness decision. A caller that treats an expired wait as
/// evidence its device is dead (as `BlockProvider` does — an expired wait
/// becomes `mark_lost`) would be answering "the device was slower than N spin
/// hints on this core, at this frequency, under this contention" as though it
/// meant "the device is gone". Anything making that judgement needs a clock:
/// `kore_driver::host::RoutePacer` when a route is held, or
/// `kore_driver::host::DeadlinePacer` when one is not.
///
/// Which leaves this pacer two honest homes: the host tests (mock devices
/// complete synchronously, so the budget is never approached) and the
/// last-resort posture on a machine with **no readable clock at all**, where
/// no time bound is expressible and a spin count is the only thing left.
pub struct SpinPacer {
    remaining: u32,
}

impl SpinPacer {
    /// A pacer with `budget` spins left.
    pub const fn new(budget: u32) -> Self {
        Self { remaining: budget }
    }
}

impl Default for SpinPacer {
    fn default() -> Self {
        Self::new(MAX_POLLS)
    }
}

impl CompletionPacer for SpinPacer {
    fn pace(&mut self) -> bool {
        if self.remaining == 0 {
            return false;
        }
        self.remaining -= 1;
        core::hint::spin_loop();
        true
    }
}

/// One passive direct queue implementation.
///
/// Implementations execute only when the owner calls `submit`, `poll`, or
/// `quiesce`; they spawn no thread and take no global lock.
pub trait DirectBlockDevice {
    type Buffer: BlockBuffer;

    /// Whether this backend's queue advances without an executor-aware
    /// [`progress`](Self::progress) step — true of a hardware queue that
    /// makes progress on its own, false of a remote endpoint that must await
    /// a correlated reply.
    ///
    /// The conservative default is `false`, so a backend only claims
    /// independence deliberately.  A composition that shares one backend
    /// between several drivers needs it: a parking `progress` future would
    /// have to hold the shared borrow across an `await`.
    const INDEPENDENT_PROGRESS: bool = false;

    fn geometry(&self) -> BlockGeometry;
    /// Where this device's block zero sits in the namespace beneath it — the
    /// coordinates a storage-transport grant names (`crate::stor`). Zero for
    /// a whole device; a region clamp adds its first block to its backend's.
    fn namespace_lba(&self) -> u64 {
        0
    }
    /// Largest single registered buffer this backend can accept. The default
    /// is the command-transfer ceiling; bounded test/transport backends may
    /// advertise a smaller allocation limit.
    fn max_buffer_bytes(&self) -> usize {
        usize::try_from(
            u64::from(self.geometry().max_transfer_blocks)
                .saturating_mul(u64::from(self.geometry().block_size)),
        )
        .unwrap_or(usize::MAX)
    }
    /// Ask the backend whether it is alive, **without issuing a command**.
    ///
    /// Called when a completion wait has expired, to decide whether that
    /// expiry is evidence of death or merely of slowness.  The conservative
    /// default is [`Liveness::Unknown`]: a backend with no out-of-band signal
    /// says nothing rather than vouching for itself.
    ///
    /// Implementations must not submit a command, allocate, or block — the
    /// command path is the thing in doubt.  A status-register read is the
    /// shape this expects.
    fn liveness(&mut self) -> Liveness {
        Liveness::Unknown
    }

    /// A monotonic clock reading in nanoseconds, if this backend's task can
    /// read one — what the channel stamps each submission with so a
    /// completion can be timed ([`ChannelStats::latency`]).
    ///
    /// The default is `None`: a backend that cannot read a clock measures
    /// nothing rather than fabricating a sample, and the books stay at the
    /// unmeasured posture.  A founded backend answers from the
    /// founder-published clock page (`kore_rt::clock::uptime`); a test
    /// backend answers from a value the test sets.  Must not block or
    /// allocate — it runs on the submit and completion paths.
    fn clock_ns(&self) -> Option<u64> {
        None
    }

    /// Ask the device to abandon the submitted command `cookie` — the
    /// escalation for one that outlived its deadline. `true` when the device
    /// agreed: the command's own completion still follows, failed, through
    /// the ordinary poll, and only then is its buffer free. `false` (the
    /// default: a backend with no abort) changes nothing; the command may
    /// still complete, late. Control plane only, and it may wait on the
    /// device's own answer.
    fn abort(&mut self, _cookie: u64) -> bool {
        false
    }

    /// Reset the device and bring its queues back — the escalation for a
    /// command it will not give up (an abort it refused). Every submitted
    /// command is lost: none of their completions ever arrives, and their
    /// buffers are free the moment this returns `true`. `false` (the
    /// default: a backend with no reset, or a reset that failed) changes
    /// nothing the caller can rely on. Control plane only.
    fn reset(&mut self) -> bool {
        false
    }

    /// Sample physical-device health on the infrequent control plane.
    /// Backends without a health protocol report `Unsupported`; callers must
    /// keep geometry/status available rather than fabricate a healthy sample.
    fn health(&mut self) -> Result<BlockHealth, BlockError> {
        Err(BlockError::Unsupported)
    }
    fn allocate_buffer(&mut self, bytes: usize) -> Result<Self::Buffer, BlockError>;
    fn submit(
        &mut self,
        command: DeviceCommand,
        buffer: Option<&mut Self::Buffer>,
        cookie: u64,
    ) -> Result<(), BlockError>;
    /// Stage one command for a later [`publish`](Self::publish). Backends
    /// without a doorbell-batching primitive may use the eager default;
    /// callers must never rely on staged commands being invisible.
    fn stage(
        &mut self,
        command: DeviceCommand,
        buffer: Option<&mut Self::Buffer>,
        cookie: u64,
    ) -> Result<(), BlockError> {
        self.submit(command, buffer, cookie)
    }
    /// Publish every command staged since the previous publish. Eager
    /// backends use the no-op default.
    fn publish(&mut self) {}
    /// Drive backend progress that may require an executor-aware transport.
    /// Hardware queues normally make progress independently and use this
    /// ready default; a remote service endpoint awaits its correlated reply
    /// here instead of fabricating a waker or busy-spinning inside `submit`.
    async fn progress(&mut self) -> Result<(), BlockError> {
        Ok(())
    }
    fn poll(&mut self) -> Option<DeviceCompletion>;
    /// Drain a bounded ready prefix. Hardware queues override this to move
    /// the completion-head doorbell once for the entire prefix.
    fn poll_batch(&mut self, out: &mut [Option<DeviceCompletion>]) -> usize {
        let mut count = 0;
        for slot in out {
            let Some(completion) = self.poll() else {
                break;
            };
            *slot = Some(completion);
            count += 1;
        }
        count
    }
    /// Apply completion-side data to the registered buffer named by the
    /// channel's pending record. Hardware DMA has already done so and uses the
    /// no-op default; copied transports finish a read at this exact ownership
    /// boundary.
    fn finish(&mut self, _buffer: Option<&mut Self::Buffer>) -> Result<(), BlockError> {
        Ok(())
    }
    fn quiesce(&mut self) -> Result<(), BlockError>;
    /// Executor-aware retirement twin for remote endpoints.
    async fn quiesce_async(&mut self) -> Result<(), BlockError> {
        self.quiesce()
    }

    /// One bounded wait for the next completion after an accepted
    /// submission — the completion wait's pacing seam.
    ///
    /// The default is the polled posture: poll between bounded spin hints
    /// ([`SpinPacer`]), then one final poll.  A backend whose task holds a
    /// delivery route overrides this to park on it with an absolute
    /// deadline instead, so the CPU is yielded while the command is in
    /// flight.  `None` means the wait budget was spent with nothing
    /// completed; the caller decides whether that is loss.
    fn wait(&mut self) -> Option<DeviceCompletion> {
        let mut pacer = SpinPacer::default();
        loop {
            if let Some(completion) = self.poll() {
                return Some(completion);
            }
            if !pacer.pace() {
                return self.poll();
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Pending {
    id: RequestId,
    buffer: Option<BufferId>,
    cancelled: bool,
    /// The backend's clock at submission, when it had one — the latency
    /// sample's start.
    submitted_ns: Option<u64>,
    /// Commands already in flight when this one was submitted.
    depth: u16,
}

struct Registered<B> {
    generation: u16,
    /// Commands in flight against this buffer.  An ordinary registered
    /// buffer admits exactly one; a streaming target admits up to
    /// `OUTSTANDING`.  CPU access through the channel requires zero.
    inflight: u16,
    /// Adopted through [`BlockChannel::adopt_target`]: caller-funded, read
    /// destination only, many disjoint commands in flight at once.
    target: bool,
    buffer: B,
}

/// One affine direct data-plane channel.
///
/// A channel is intentionally not `Sync`; one worker moves and drives it.
///
/// ```compile_fail
/// use kore_driver::block::{BlockChannel, memory::MemoryBlockDevice};
/// fn requires_sync<T: Sync>() {}
/// requires_sync::<BlockChannel<MemoryBlockDevice<4096, 512>>>();
/// ```
pub struct BlockChannel<
    D: DirectBlockDevice,
    const BUFFERS: usize = DEFAULT_BUFFERS,
    const OUTSTANDING: usize = DEFAULT_OUTSTANDING,
> {
    device: D,
    geometry: BlockGeometry,
    mode: CompletionMode,
    state: ChannelState,
    buffers: [Option<Registered<D::Buffer>>; BUFFERS],
    generations: [u16; BUFFERS],
    pending: [Option<Pending>; OUTSTANDING],
    next_request: u64,
    late_completions: u64,
    stats: ChannelStats,
    // One qpair has one worker owner. Cell is Send but not Sync, making
    // concurrent shared driving unrepresentable without unsafe code.
    _owner: PhantomData<Cell<()>>,
}

impl<D: DirectBlockDevice, const BUFFERS: usize, const OUTSTANDING: usize>
    BlockChannel<D, BUFFERS, OUTSTANDING>
{
    pub fn new(device: D, mode: CompletionMode) -> Result<Self, BlockError> {
        // Both tables are indexed by `u16` (buffer slots, and a target's
        // in-flight count, which the pending table bounds).
        if BUFFERS == 0
            || OUTSTANDING == 0
            || BUFFERS > u16::MAX as usize
            || OUTSTANDING > u16::MAX as usize
        {
            return Err(BlockError::Geometry);
        }
        let geometry = device.geometry().validate()?;
        Ok(Self {
            device,
            geometry,
            mode,
            state: if geometry.read_only {
                ChannelState::ReadOnly
            } else {
                ChannelState::Online
            },
            buffers: core::array::from_fn(|_| None),
            generations: [1; BUFFERS],
            pending: [None; OUTSTANDING],
            next_request: 1,
            late_completions: 0,
            stats: ChannelStats::default(),
            _owner: PhantomData,
        })
    }

    pub const fn geometry(&self) -> BlockGeometry {
        self.geometry
    }

    /// Where the device's block zero sits in its namespace
    /// ([`DirectBlockDevice::namespace_lba`]).
    pub fn namespace_lba(&self) -> u64 {
        self.device.namespace_lba()
    }

    pub fn max_buffer_bytes(&self) -> usize {
        self.device.max_buffer_bytes()
    }

    /// Ask the backend whether it is alive, without issuing a command — what
    /// an owner consults before deciding an expired wait means death rather
    /// than slowness.  See [`Liveness`].
    pub fn liveness(&mut self) -> Liveness {
        self.device.liveness()
    }

    pub const fn mode(&self) -> CompletionMode {
        self.mode
    }

    pub const fn state(&self) -> ChannelState {
        self.state
    }

    pub fn outstanding(&self) -> usize {
        self.pending.iter().flatten().count()
    }

    pub const fn late_completions(&self) -> u64 {
        self.late_completions
    }

    pub const fn stats(&self) -> ChannelStats {
        self.stats
    }

    pub fn register_buffer(&mut self, bytes: usize) -> Result<BufferId, BlockError> {
        self.accepts_control()?;
        let slot = self
            .buffers
            .iter()
            .position(Option::is_none)
            .ok_or(BlockError::Backpressure)?;
        let buffer = self.device.allocate_buffer(bytes)?;
        if buffer.len() < bytes {
            return Err(BlockError::Integrity);
        }
        let id = BufferId {
            slot: slot as u16,
            generation: self.generations[slot],
        };
        self.buffers[slot] = Some(Registered {
            generation: id.generation,
            inflight: 0,
            target: false,
            buffer,
        });
        self.stats.registered_bytes = self.stats.registered_bytes.saturating_add(bytes);
        self.stats.peak_registered_bytes = self
            .stats
            .peak_registered_bytes
            .max(self.stats.registered_bytes);
        Ok(id)
    }

    pub fn unregister_buffer(&mut self, id: BufferId) -> Result<(), BlockError> {
        let slot = self.buffer_slot(id)?;
        if self.buffers[slot].as_ref().is_some_and(|b| b.inflight != 0) {
            return Err(BlockError::BufferBusy);
        }
        if let Some(registered) = self.buffers[slot].take()
            && !registered.target
        {
            self.stats.registered_bytes = self
                .stats
                .registered_bytes
                .saturating_sub(registered.buffer.len());
        }
        self.generations[slot] = self.generations[slot].wrapping_add(1).max(1);
        Ok(())
    }

    /// Adopt a backend buffer over memory the **caller** funded — a device
    /// mapping of a client's lease, never the backend's own pool — as a
    /// **streaming target**.
    ///
    /// A target differs from a registered buffer in exactly two ways, both
    /// following from who owns its bytes:
    ///
    /// - It admits many commands in flight at once, each into its own
    ///   block-aligned range: a multi-GiB destination is one buffer, not one
    ///   buffer per command.  The channel does not police overlap between
    ///   in-flight ranges — the bytes are the caller's, so two reads racing
    ///   into one range can only hand the caller its own data in an
    ///   unspecified order.  What the channel still guarantees is the
    ///   exclusion that protects *other* memory: every range is checked
    ///   against the target's length, and CPU access through the channel is
    ///   refused while anything is in flight.
    /// - It is a read destination only.  A write sourcing caller memory is
    ///   not something a streaming target is for, and refusing it keeps the
    ///   lane's contract one direction wide.
    ///
    /// Target bytes are not the channel's allocation, so they are not
    /// booked in [`ChannelStats::registered_bytes`].  A refused adoption
    /// returns the buffer: it is affine custody of a device mapping, and
    /// dropping it here would strand that mapping without its owner's
    /// quiescent release.
    pub fn adopt_target(&mut self, buffer: D::Buffer) -> Result<BufferId, (D::Buffer, BlockError)> {
        if let Err(error) = self.accepts_control() {
            return Err((buffer, error));
        }
        if buffer.len() == 0 || buffer.len() % self.geometry.block_size as usize != 0 {
            return Err((buffer, BlockError::Alignment));
        }
        let Some(slot) = self.buffers.iter().position(Option::is_none) else {
            return Err((buffer, BlockError::Backpressure));
        };
        let id = BufferId {
            slot: slot as u16,
            generation: self.generations[slot],
        };
        self.buffers[slot] = Some(Registered {
            generation: id.generation,
            inflight: 0,
            target: true,
            buffer,
        });
        Ok(id)
    }

    /// Remove an idle buffer and return it to its owner — the release twin
    /// of [`adopt_target`](Self::adopt_target), for a caller that must
    /// retire the backend mapping itself rather than drop it.  Works for any
    /// registered buffer; refused while a command is in flight against it.
    pub fn release_buffer(&mut self, id: BufferId) -> Result<D::Buffer, BlockError> {
        let slot = self.buffer_slot(id)?;
        if self.buffers[slot].as_ref().is_some_and(|b| b.inflight != 0) {
            return Err(BlockError::BufferBusy);
        }
        let registered = self.buffers[slot].take().ok_or(BlockError::UnknownBuffer)?;
        if !registered.target {
            self.stats.registered_bytes = self
                .stats
                .registered_bytes
                .saturating_sub(registered.buffer.len());
        }
        self.generations[slot] = self.generations[slot].wrapping_add(1).max(1);
        Ok(registered.buffer)
    }

    /// Commands in flight against one buffer, or `None` for a stale id.
    pub fn buffer_inflight(&self, id: BufferId) -> Option<u16> {
        let slot = self.buffer_slot(id).ok()?;
        self.buffers[slot].as_ref().map(|b| b.inflight)
    }

    pub fn write_buffer(
        &mut self,
        id: BufferId,
        offset: usize,
        src: &[u8],
    ) -> Result<(), BlockError> {
        let slot = self.buffer_slot(id)?;
        let registered = self.buffers[slot]
            .as_mut()
            .ok_or(BlockError::UnknownBuffer)?;
        if registered.inflight != 0 {
            return Err(BlockError::BufferBusy);
        }
        // A streaming target is the controller's to write and its owner's to
        // read: the channel may hold no CPU view of it at all (a device-only
        // import), so it never touches one through the CPU.
        if registered.target {
            return Err(BlockError::Range);
        }
        let end = offset.checked_add(src.len()).ok_or(BlockError::Range)?;
        let dst = registered
            .buffer
            .bytes_mut()
            .get_mut(offset..end)
            .ok_or(BlockError::Range)?;
        dst.copy_from_slice(src);
        Ok(())
    }

    pub fn read_buffer(
        &self,
        id: BufferId,
        offset: usize,
        dst: &mut [u8],
    ) -> Result<(), BlockError> {
        let slot = self.buffer_slot(id)?;
        let registered = self.buffers[slot]
            .as_ref()
            .ok_or(BlockError::UnknownBuffer)?;
        if registered.inflight != 0 {
            return Err(BlockError::BufferBusy);
        }
        // A streaming target is the controller's to write and its owner's to
        // read: the channel may hold no CPU view of it at all (a device-only
        // import), so it never touches one through the CPU.
        if registered.target {
            return Err(BlockError::Range);
        }
        let end = offset.checked_add(dst.len()).ok_or(BlockError::Range)?;
        let src = registered
            .buffer
            .bytes()
            .get(offset..end)
            .ok_or(BlockError::Range)?;
        dst.copy_from_slice(src);
        Ok(())
    }

    /// Fill one idle registered buffer without a stack-sized intermediary.
    pub fn fill_buffer(&mut self, id: BufferId, value: u8) -> Result<(), BlockError> {
        let slot = self.buffer_slot(id)?;
        let registered = self.buffers[slot]
            .as_mut()
            .ok_or(BlockError::UnknownBuffer)?;
        if registered.inflight != 0 {
            return Err(BlockError::BufferBusy);
        }
        // A streaming target is the controller's to write and its owner's to
        // read: the channel may hold no CPU view of it at all (a device-only
        // import), so it never touches one through the CPU.
        if registered.target {
            return Err(BlockError::Range);
        }
        registered.buffer.bytes_mut().fill(value);
        Ok(())
    }

    /// Begin a bounded submission batch. Dropping the batch publishes any
    /// successfully staged prefix, so an early validation/backpressure error
    /// cannot strand busy buffers behind an un-rung hardware tail.
    pub fn batch(&mut self) -> BlockBatch<'_, D, BUFFERS, OUTSTANDING> {
        BlockBatch {
            channel: self,
            published: false,
        }
    }

    pub fn submit_read(
        &mut self,
        lba: u64,
        blocks: u32,
        buffer: BufferId,
        buffer_offset: usize,
    ) -> Result<RequestId, BlockError> {
        self.submit(
            DeviceCommand::Read {
                lba,
                blocks,
                buffer_offset,
            },
            Some(buffer),
        )
    }

    pub fn submit_write(
        &mut self,
        lba: u64,
        blocks: u32,
        buffer: BufferId,
        buffer_offset: usize,
    ) -> Result<RequestId, BlockError> {
        if matches!(self.state, ChannelState::ReadOnly) || self.geometry.read_only {
            return Err(BlockError::ReadOnly);
        }
        self.submit(
            DeviceCommand::Write {
                lba,
                blocks,
                buffer_offset,
            },
            Some(buffer),
        )
    }

    pub fn submit_flush(&mut self) -> Result<RequestId, BlockError> {
        if !self.geometry.flush {
            return Err(BlockError::Unsupported);
        }
        self.submit(DeviceCommand::Flush, None)
    }

    pub fn submit_discard(&mut self, lba: u64, blocks: u32) -> Result<RequestId, BlockError> {
        if !self.geometry.discard {
            return Err(BlockError::Unsupported);
        }
        if matches!(self.state, ChannelState::ReadOnly) || self.geometry.read_only {
            return Err(BlockError::ReadOnly);
        }
        self.submit(DeviceCommand::Discard { lba, blocks }, None)
    }

    pub fn cancel(&mut self, id: RequestId) -> Result<(), BlockError> {
        let pending = self
            .pending
            .iter_mut()
            .flatten()
            .find(|pending| pending.id == id)
            .ok_or(BlockError::UnknownRequest)?;
        pending.cancelled = true;
        self.stats.cancellations = self.stats.cancellations.saturating_add(1);
        Ok(())
    }

    /// Process one completion. Unknown/late cookies are reported distinctly
    /// and can never complete a different request.
    pub fn poll(&mut self) -> Option<Completion> {
        self.stats.poll_calls = self.stats.poll_calls.saturating_add(1);
        let completion = self.device.poll()?;
        Some(self.admit(completion))
    }

    /// Drain up to `N` ready completions. Device completion ownership is
    /// acquired before each request is admitted and its buffer becomes idle;
    /// the returned array contains a dense `Some` prefix.
    pub fn poll_batch<const N: usize>(&mut self) -> [Option<Completion>; N] {
        self.stats.poll_calls = self.stats.poll_calls.saturating_add(1);
        let mut raw = [None; N];
        let count = self.device.poll_batch(&mut raw);
        let mut admitted = [None; N];
        for index in 0..count {
            admitted[index] = raw[index].take().map(|completion| self.admit(completion));
        }
        admitted
    }

    /// Process one completion, waiting as the backend's pacing seam allows.
    ///
    /// The fast path is one [`poll`](Self::poll).  When it is empty and a
    /// request is outstanding, the wait delegates to
    /// [`DirectBlockDevice::wait`]: the bounded spin by default, or the
    /// backend's own delivery park when its task holds a route.  `None`
    /// means nothing was outstanding, or the backend's wait budget was
    /// spent with nothing completed — the caller decides whether that is
    /// loss ([`mark_lost`](Self::mark_lost)).
    pub fn wait(&mut self) -> Option<Completion> {
        if let Some(completion) = self.poll() {
            return Some(completion);
        }
        if self.outstanding() == 0 {
            // Nothing in flight can complete; never spend a wait budget
            // (or park on a route) against an idle queue.
            return None;
        }
        // Past the fast-path poll the backend's own seam decides how to
        // wait — a delivery park when its task holds a route, the bounded
        // spin otherwise. Counted here, where a wait genuinely happens: an
        // empty poll against an idle queue is not a park, and counting it as
        // one made a busy-spinning machine read as healthy.
        self.stats.device_waits = self.stats.device_waits.saturating_add(1);
        let completion = self.device.wait()?;
        Some(self.admit(completion))
    }

    /// Correlate one backend completion with its pending request.
    fn admit(&mut self, completion: DeviceCompletion) -> Completion {
        let Some(slot) = self
            .pending
            .iter()
            .position(|entry| entry.is_some_and(|p| p.id.0 == completion.cookie))
        else {
            self.late_completions = self.late_completions.saturating_add(1);
            return Completion {
                request: RequestId(completion.cookie),
                result: Err(BlockError::StaleCompletion),
            };
        };
        let pending = self.pending[slot].take().expect("located pending request");
        self.stats.completions = self.stats.completions.saturating_add(1);
        // A sample needs both ends read off the same clock; a backend that
        // answered at submit but not now (or the reverse) books nothing.
        if let (Some(start), Some(now)) = (pending.submitted_ns, self.device.clock_ns()) {
            self.stats
                .latency
                .record(now.saturating_sub(start), pending.depth);
        }
        let finish = if let Some(buffer) = pending.buffer {
            match self.buffer_slot(buffer) {
                Ok(buffer_slot) => self
                    .device
                    .finish(self.buffers[buffer_slot].as_mut().map(|r| &mut r.buffer)),
                Err(error) => Err(error),
            }
        } else {
            self.device.finish(None)
        };
        if let Some(buffer) = pending.buffer
            && let Ok(buffer_slot) = self.buffer_slot(buffer)
            && let Some(registered) = self.buffers[buffer_slot].as_mut()
        {
            registered.inflight = registered.inflight.saturating_sub(1);
        }
        Completion {
            request: pending.id,
            result: if pending.cancelled {
                Err(BlockError::Cancelled)
            } else {
                completion.result.and(finish)
            },
        }
    }

    /// Await one backend progress step. Callers then use [`poll`](Self::poll)
    /// to correlate and settle any completion it produced.
    pub async fn progress(&mut self) -> Result<(), BlockError> {
        self.device.progress().await
    }

    /// Stop accepting work. The owner polls until `outstanding() == 0`, then
    /// calls this again to retire the backend child-before-parent.
    pub fn quiesce(&mut self) -> Result<(), BlockError> {
        if self.state == ChannelState::Revoked {
            return Err(BlockError::Revoked);
        }
        self.state = ChannelState::Draining;
        if self.outstanding() != 0 {
            return Err(BlockError::Draining);
        }
        self.device.quiesce()?;
        self.state = ChannelState::Revoked;
        Ok(())
    }

    /// Async retirement for a service-backed device. The same state and
    /// child-before-parent rules as [`quiesce`](Self::quiesce) apply.
    pub async fn quiesce_async(&mut self) -> Result<(), BlockError> {
        if self.state == ChannelState::Revoked {
            return Err(BlockError::Revoked);
        }
        self.state = ChannelState::Draining;
        if self.outstanding() != 0 {
            return Err(BlockError::Draining);
        }
        self.device.quiesce_async().await?;
        self.state = ChannelState::Revoked;
        Ok(())
    }

    pub fn mark_lost(&mut self) {
        self.state = ChannelState::Lost;
    }

    /// Recover the backend from a channel that has nothing in flight and no
    /// registered buffers — the re-provisioning path: a provider that has
    /// probed a whole device and discovered a designated region rebuilds its
    /// channel over the same backend, narrowed.
    ///
    /// Registered buffers are the backend's DMA children; requiring them
    /// gone first means this recovery cannot reorder containment.  A refused
    /// recovery returns the intact affine channel to its owner.
    pub fn into_device(self) -> Result<D, (Self, BlockError)> {
        if self.outstanding() != 0 || self.buffers.iter().any(Option::is_some) {
            return Err((self, BlockError::BufferBusy));
        }
        Ok(self.device)
    }

    /// Run an infrequent owner/control-plane operation against the backend.
    /// Refuses while data requests are live, so a test fault injector or a
    /// lifecycle/status probe cannot race the qpair owner.
    pub fn control<T>(&mut self, f: impl FnOnce(&mut D) -> T) -> Result<T, BlockError> {
        self.accepts_control()?;
        if self.outstanding() != 0 {
            return Err(BlockError::BufferBusy);
        }
        Ok(f(&mut self.device))
    }

    fn accepts_control(&self) -> Result<(), BlockError> {
        match self.state {
            ChannelState::Online | ChannelState::ReadOnly => Ok(()),
            ChannelState::Draining => Err(BlockError::Draining),
            ChannelState::Lost => Err(BlockError::Lost),
            ChannelState::Revoked => Err(BlockError::Revoked),
        }
    }

    fn accepts_io(&self) -> Result<(), BlockError> {
        self.accepts_control()
    }

    fn buffer_slot(&self, id: BufferId) -> Result<usize, BlockError> {
        let slot = id.slot as usize;
        let registered = self
            .buffers
            .get(slot)
            .and_then(Option::as_ref)
            .ok_or(BlockError::UnknownBuffer)?;
        if registered.generation != id.generation {
            return Err(BlockError::UnknownBuffer);
        }
        Ok(slot)
    }

    fn validate_range(
        &self,
        command: DeviceCommand,
        buffer: Option<BufferId>,
    ) -> Result<(), BlockError> {
        let (lba, blocks, buffer_offset, needs_buffer) = match command {
            DeviceCommand::Read {
                lba,
                blocks,
                buffer_offset,
            }
            | DeviceCommand::Write {
                lba,
                blocks,
                buffer_offset,
            } => (lba, blocks, buffer_offset, true),
            DeviceCommand::Discard { lba, blocks } => (lba, blocks, 0, false),
            DeviceCommand::Flush => return Ok(()),
        };
        if blocks == 0 || blocks > self.geometry.max_transfer_blocks {
            return Err(if blocks == 0 {
                BlockError::Range
            } else {
                BlockError::TooLarge
            });
        }
        let end = lba.checked_add(blocks as u64).ok_or(BlockError::Range)?;
        if end > self.geometry.block_count {
            return Err(BlockError::Range);
        }
        if needs_buffer {
            let id = buffer.ok_or(BlockError::UnknownBuffer)?;
            let slot = self.buffer_slot(id)?;
            let bytes = (blocks as usize)
                .checked_mul(self.geometry.block_size as usize)
                .ok_or(BlockError::Range)?;
            if buffer_offset % self.geometry.block_size as usize != 0 {
                return Err(BlockError::Alignment);
            }
            let end = buffer_offset.checked_add(bytes).ok_or(BlockError::Range)?;
            if end
                > self.buffers[slot]
                    .as_ref()
                    .expect("validated buffer")
                    .buffer
                    .len()
            {
                return Err(BlockError::Range);
            }
        }
        Ok(())
    }

    fn submit(
        &mut self,
        command: DeviceCommand,
        buffer: Option<BufferId>,
    ) -> Result<RequestId, BlockError> {
        self.submit_inner(command, buffer, false)
    }

    fn stage(
        &mut self,
        command: DeviceCommand,
        buffer: Option<BufferId>,
    ) -> Result<RequestId, BlockError> {
        self.submit_inner(command, buffer, true)
    }

    fn submit_inner(
        &mut self,
        command: DeviceCommand,
        buffer: Option<BufferId>,
        deferred: bool,
    ) -> Result<RequestId, BlockError> {
        self.accepts_io()?;
        self.validate_range(command, buffer)?;
        let Some(request_slot) = self.pending.iter().position(Option::is_none) else {
            self.stats.backpressure = self.stats.backpressure.saturating_add(1);
            return Err(BlockError::Backpressure);
        };
        let id = RequestId(self.next_request);
        self.next_request = self.next_request.wrapping_add(1).max(1);

        let buffer_slot = match buffer {
            Some(id) => Some(self.buffer_slot(id)?),
            None => None,
        };
        if let Some(slot) = buffer_slot {
            let registered = self.buffers[slot].as_mut().expect("validated buffer");
            if registered.target {
                // A target is a read destination; its pending table bounds
                // its depth, so the count cannot pass `OUTSTANDING`.
                if !matches!(command, DeviceCommand::Read { .. }) {
                    return Err(BlockError::ReadOnly);
                }
            } else if registered.inflight != 0 {
                return Err(BlockError::BufferBusy);
            }
            registered.inflight += 1;
        }

        let outcome = match buffer_slot {
            Some(slot) => {
                let buffer = Some(
                    &mut self.buffers[slot]
                        .as_mut()
                        .expect("validated buffer")
                        .buffer,
                );
                if deferred {
                    self.device.stage(command, buffer, id.0)
                } else {
                    self.device.submit(command, buffer, id.0)
                }
            }
            None if deferred => self.device.stage(command, None, id.0),
            None => self.device.submit(command, None, id.0),
        };
        if let Err(error) = outcome {
            if let Some(slot) = buffer_slot {
                let registered = self.buffers[slot].as_mut().expect("validated buffer");
                registered.inflight -= 1;
            }
            return Err(error);
        }
        // Depth is what was in flight *before* this command — the queue it
        // will wait behind — read before the slot is filled.
        let depth = self.outstanding() as u16;
        self.pending[request_slot] = Some(Pending {
            id,
            buffer,
            cancelled: false,
            submitted_ns: self.device.clock_ns(),
            depth,
        });
        self.stats.submissions = self.stats.submissions.saturating_add(1);
        self.stats.max_outstanding = self.stats.max_outstanding.max(self.outstanding() as u16);
        Ok(id)
    }
}

/// One affine borrow of a channel's submission tail. It stages through the
/// same range, buffer-busy, and pending-table checks as ordinary submission.
pub struct BlockBatch<'a, D: DirectBlockDevice, const BUFFERS: usize, const OUTSTANDING: usize> {
    channel: &'a mut BlockChannel<D, BUFFERS, OUTSTANDING>,
    published: bool,
}

impl<'a, D: DirectBlockDevice, const B: usize, const O: usize> BlockBatch<'a, D, B, O> {
    pub fn stage_read(
        &mut self,
        lba: u64,
        blocks: u32,
        buffer: BufferId,
        buffer_offset: usize,
    ) -> Result<RequestId, BlockError> {
        self.channel.stage(
            DeviceCommand::Read {
                lba,
                blocks,
                buffer_offset,
            },
            Some(buffer),
        )
    }

    pub fn stage_write(
        &mut self,
        lba: u64,
        blocks: u32,
        buffer: BufferId,
        buffer_offset: usize,
    ) -> Result<RequestId, BlockError> {
        if matches!(self.channel.state, ChannelState::ReadOnly) || self.channel.geometry.read_only {
            return Err(BlockError::ReadOnly);
        }
        self.channel.stage(
            DeviceCommand::Write {
                lba,
                blocks,
                buffer_offset,
            },
            Some(buffer),
        )
    }

    pub fn stage_flush(&mut self) -> Result<RequestId, BlockError> {
        if !self.channel.geometry.flush {
            return Err(BlockError::Unsupported);
        }
        self.channel.stage(DeviceCommand::Flush, None)
    }

    pub fn stage_discard(&mut self, lba: u64, blocks: u32) -> Result<RequestId, BlockError> {
        if !self.channel.geometry.discard {
            return Err(BlockError::Unsupported);
        }
        if matches!(self.channel.state, ChannelState::ReadOnly) || self.channel.geometry.read_only {
            return Err(BlockError::ReadOnly);
        }
        self.channel
            .stage(DeviceCommand::Discard { lba, blocks }, None)
    }

    pub fn publish(mut self) {
        self.channel.device.publish();
        self.published = true;
    }

    /// Release a batch that staged nothing without a doorbell: a poll that
    /// found no free slot or no queued work should not ring the device.
    pub fn skip_publish(mut self) {
        self.published = true;
    }
}

impl<D: DirectBlockDevice, const B: usize, const O: usize> Drop for BlockBatch<'_, D, B, O> {
    fn drop(&mut self) {
        if !self.published {
            self.channel.device.publish();
        }
    }
}
