//! Frozen, provider-owned telemetry snapshots.
//!
//! Each producer defines the meaning of its counters; the monitor service
//! merely assembles one bounded launch-time view.  Zero is the unavailable
//! posture, never a fabricated sample.

use zerocopy::byteorder::little_endian::{I16, U16, U32, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

use crate::process::ProcessSnapshot;

const AVAILABLE: u32 = 1;
const AVAILABLE_U16: u16 = 1;

/// Number of bytes represented by one extent unit in the fixed monitor ABI.
const FABRIC_EXTENT_UNIT_BYTES: u64 = 1 << 10;

/// Kernel-owned system snapshot: scheduler time, physical memory and heap.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct SystemSnapshot {
    flags: U32,
    cpu_count: U32,
    /// Packed memory-fabric domain posture; see the accessors below.
    fabric_domains: U32,
    /// Aggregate registered extent in KiB, saturating at `u32::MAX`.
    fabric_extent_kib: U32,
    captured_ns: U64,
    cpu_busy_ns: U64,
    phys_total: U64,
    phys_free: U64,
    heap_capacity: U64,
    heap_used: U64,
    heap_largest: U64,
    /// Cumulative IOMMU translation-fault records drained since boot.
    iommu_faults: U64,
    /// The LAST fault record, packed: `[15:0]` source-id, `[23:16]`
    /// raw reason, `[24]` write, `[63:25]` faulting-IOVA page.
    iommu_last: U64,
    /// Scheduler-timer hardware arms since boot, every CPU summed.
    timer_arms: U64,
    /// `[63:32]` arms that needed a second try (a deadline write that read
    /// back unarmed, or a one-shot count clamped up from zero), `[31:0]` arms
    /// refused after the second deadline write also failed
    /// (`SystemMetric::TimerArms` auxiliary).  A timer that loses arms in
    /// steady state is otherwise invisible on a machine with no serial.
    timer_arm_faults: U64,
}

impl SystemSnapshot {
    pub const EMPTY: Self = Self {
        flags: U32::new(0),
        cpu_count: U32::new(0),
        fabric_domains: U32::new(0),
        fabric_extent_kib: U32::new(0),
        captured_ns: U64::new(0),
        cpu_busy_ns: U64::new(0),
        phys_total: U64::new(0),
        phys_free: U64::new(0),
        heap_capacity: U64::new(0),
        heap_used: U64::new(0),
        heap_largest: U64::new(0),
        iommu_faults: U64::new(0),
        iommu_last: U64::new(0),
        timer_arms: U64::new(0),
        timer_arm_faults: U64::new(0),
    };

    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        cpu_count: u32,
        captured_ns: u64,
        cpu_busy_ns: u64,
        phys_total: u64,
        phys_free: u64,
        heap_capacity: u64,
        heap_used: u64,
        heap_largest: u64,
        iommu_faults: u64,
        iommu_last: u64,
        fabric_domains: u32,
        timer_arms: u64,
        timer_arm_faults: u64,
    ) -> Self {
        let fabric_domains = if fabric_domains > u8::MAX as u32 {
            u8::MAX as u32
        } else {
            fabric_domains
        };
        Self {
            flags: U32::new(AVAILABLE),
            cpu_count: U32::new(cpu_count),
            fabric_domains: U32::new(fabric_domains),
            fabric_extent_kib: U32::new(0),
            captured_ns: U64::new(captured_ns),
            cpu_busy_ns: U64::new(cpu_busy_ns),
            phys_total: U64::new(phys_total),
            phys_free: U64::new(phys_free),
            heap_capacity: U64::new(heap_capacity),
            heap_used: U64::new(heap_used),
            heap_largest: U64::new(heap_largest),
            iommu_faults: U64::new(iommu_faults),
            iommu_last: U64::new(iommu_last),
            timer_arms: U64::new(timer_arms),
            timer_arm_faults: U64::new(timer_arm_faults),
        }
    }

    pub fn available(self) -> bool {
        self.flags.get() == AVAILABLE
    }
    pub fn validate(self) -> bool {
        if !self.available() {
            return self == Self::EMPTY;
        }
        let domains = self.fabric_domains();
        let capacity = self.fabric_capacity();
        (capacity == 0 || domains <= capacity)
            && self.fabric_lendable_domains() <= domains
            && self.fabric_fence_required_domains() <= domains
    }
    pub fn cpu_count(self) -> u32 {
        self.cpu_count.get()
    }
    pub fn captured_ns(self) -> u64 {
        self.captured_ns.get()
    }
    pub fn cpu_busy_ns(self) -> u64 {
        self.cpu_busy_ns.get()
    }
    pub fn phys_total(self) -> u64 {
        self.phys_total.get()
    }
    pub fn phys_free(self) -> u64 {
        self.phys_free.get()
    }
    pub fn heap_capacity(self) -> u64 {
        self.heap_capacity.get()
    }
    pub fn heap_used(self) -> u64 {
        self.heap_used.get()
    }
    pub fn heap_largest(self) -> u64 {
        self.heap_largest.get()
    }
    pub fn iommu_faults(self) -> u64 {
        self.iommu_faults.get()
    }
    pub fn iommu_last(self) -> u64 {
        self.iommu_last.get()
    }
    /// Scheduler-timer hardware arms since boot, every CPU summed.
    pub fn timer_arms(self) -> u64 {
        self.timer_arms.get()
    }
    /// Arms that needed a second try: a deadline write that read back
    /// unarmed, or a one-shot count clamped up from zero.
    pub fn timer_arms_retried(self) -> u64 {
        self.timer_arm_faults.get() >> 32
    }
    /// Arms refused after the second write also read back as unarmed.
    pub fn timer_arms_dead(self) -> u64 {
        self.timer_arm_faults.get() & 0xffff_ffff
    }
    /// Live memory-fabric domain registrations (#697). Zero until a producer
    /// registers a domain — the registry health surface for `status fab`.
    pub fn fabric_domains(self) -> u32 {
        self.fabric_domains.get() & 0xff
    }
    /// Bounded registry slots. Zero is the legacy/no-detail posture.
    pub fn fabric_capacity(self) -> u32 {
        (self.fabric_domains.get() >> 8) & 0xff
    }
    /// Domains advertising at least one lendable window.
    pub fn fabric_lendable_domains(self) -> u32 {
        (self.fabric_domains.get() >> 16) & 0xff
    }
    /// Domains whose visibility contract requires an explicit fence.
    pub fn fabric_fence_required_domains(self) -> u32 {
        self.fabric_domains.get() >> 24
    }
    /// Aggregate registered extent, rounded down to KiB and saturating at the
    /// wire ceiling. Registration extents are page-granular in production.
    pub fn fabric_extent_bytes(self) -> u64 {
        u64::from(self.fabric_extent_kib.get()) * FABRIC_EXTENT_UNIT_BYTES
    }

    /// Attach the complete fabric census to a newly assembled system sample.
    pub const fn with_fabric(
        mut self,
        domains: u32,
        capacity: u32,
        lendable_domains: u32,
        fence_required_domains: u32,
        extent_bytes: u64,
    ) -> Self {
        let domains = if domains > u8::MAX as u32 {
            u8::MAX as u32
        } else {
            domains
        };
        let capacity = if capacity > u8::MAX as u32 {
            u8::MAX as u32
        } else {
            capacity
        };
        let lendable = if lendable_domains > u8::MAX as u32 {
            u8::MAX as u32
        } else {
            lendable_domains
        };
        let fences = if fence_required_domains > u8::MAX as u32 {
            u8::MAX as u32
        } else {
            fence_required_domains
        };
        let extent_kib = extent_bytes / FABRIC_EXTENT_UNIT_BYTES;
        let extent_kib = if extent_kib > u32::MAX as u64 {
            u32::MAX
        } else {
            extent_kib as u32
        };
        self.fabric_domains =
            U32::new(domains | (capacity << 8) | (lendable << 16) | (fences << 24));
        self.fabric_extent_kib = U32::new(extent_kib);
        self
    }
}

/// Memory-manager-owned allocator and child-supply snapshot.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct MemorySnapshot {
    flags: U32,
    extents: U32,
    supplies: U32,
    reserved: U32,
    granted: U64,
    released: U64,
    reused: U64,
    leaked: U64,
    remaining: U64,
    supply_bytes: U64,
}

impl MemorySnapshot {
    pub const EMPTY: Self = Self {
        flags: U32::new(0),
        extents: U32::new(0),
        supplies: U32::new(0),
        reserved: U32::new(0),
        granted: U64::new(0),
        released: U64::new(0),
        reused: U64::new(0),
        leaked: U64::new(0),
        remaining: U64::new(0),
        supply_bytes: U64::new(0),
    };

    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        extents: u32,
        supplies: u32,
        granted: u64,
        released: u64,
        reused: u64,
        leaked: u64,
        remaining: u64,
        supply_bytes: u64,
    ) -> Self {
        Self {
            flags: U32::new(AVAILABLE),
            extents: U32::new(extents),
            supplies: U32::new(supplies),
            reserved: U32::new(0),
            granted: U64::new(granted),
            released: U64::new(released),
            reused: U64::new(reused),
            leaked: U64::new(leaked),
            remaining: U64::new(remaining),
            supply_bytes: U64::new(supply_bytes),
        }
    }

    pub fn available(self) -> bool {
        self.flags.get() == AVAILABLE
    }
    pub fn validate(self) -> bool {
        self.reserved.get() == 0 && (self.available() || self == Self::EMPTY)
    }
    pub fn extents(self) -> u32 {
        self.extents.get()
    }
    pub fn supplies(self) -> u32 {
        self.supplies.get()
    }
    pub fn granted(self) -> u64 {
        self.granted.get()
    }
    pub fn released(self) -> u64 {
        self.released.get()
    }
    pub fn reused(self) -> u64 {
        self.reused.get()
    }
    pub fn leaked(self) -> u64 {
        self.leaked.get()
    }
    pub fn remaining(self) -> u64 {
        self.remaining.get()
    }
    pub fn supply_bytes(self) -> u64 {
        self.supply_bytes.get()
    }
}

/// Physical-device health sampled through the block control plane.
///
/// Zero is unavailable, never "healthy": callers must check
/// [`available`](Self::available) before interpreting the other fields.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct FilesystemHealthSnapshot {
    flags: U16,
    temperature_celsius: I16,
    critical_warning: u8,
    available_spare_percent: u8,
    spare_threshold_percent: u8,
    percentage_used: u8,
    power_cycles: U64,
    power_on_hours: U64,
    unsafe_shutdowns: U64,
    media_errors: U64,
}

impl FilesystemHealthSnapshot {
    const AVAILABLE: u16 = 1;

    pub const EMPTY: Self = Self {
        flags: U16::new(0),
        temperature_celsius: I16::new(0),
        critical_warning: 0,
        available_spare_percent: 0,
        spare_threshold_percent: 0,
        percentage_used: 0,
        power_cycles: U64::new(0),
        power_on_hours: U64::new(0),
        unsafe_shutdowns: U64::new(0),
        media_errors: U64::new(0),
    };

    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        critical_warning: u8,
        temperature_celsius: i16,
        available_spare_percent: u8,
        spare_threshold_percent: u8,
        percentage_used: u8,
        power_cycles: u64,
        power_on_hours: u64,
        unsafe_shutdowns: u64,
        media_errors: u64,
    ) -> Self {
        Self {
            flags: U16::new(Self::AVAILABLE),
            temperature_celsius: I16::new(temperature_celsius),
            critical_warning,
            available_spare_percent,
            spare_threshold_percent,
            percentage_used,
            power_cycles: U64::new(power_cycles),
            power_on_hours: U64::new(power_on_hours),
            unsafe_shutdowns: U64::new(unsafe_shutdowns),
            media_errors: U64::new(media_errors),
        }
    }

    pub fn available(self) -> bool {
        self.flags.get() & Self::AVAILABLE != 0
    }

    pub fn validate(self) -> bool {
        self == Self::EMPTY
            || (self.flags.get() == Self::AVAILABLE
                && self.available_spare_percent <= 100
                && self.spare_threshold_percent <= 100)
    }

    pub fn critical_warning(self) -> u8 {
        self.critical_warning
    }
    pub fn temperature_celsius(self) -> i16 {
        self.temperature_celsius.get()
    }
    pub fn available_spare_percent(self) -> u8 {
        self.available_spare_percent
    }
    pub fn spare_threshold_percent(self) -> u8 {
        self.spare_threshold_percent
    }
    pub fn percentage_used(self) -> u8 {
        self.percentage_used
    }
    pub fn power_cycles(self) -> u64 {
        self.power_cycles.get()
    }
    pub fn power_on_hours(self) -> u64 {
        self.power_on_hours.get()
    }
    pub fn unsafe_shutdowns(self) -> u64 {
        self.unsafe_shutdowns.get()
    }
    pub fn media_errors(self) -> u64 {
        self.media_errors.get()
    }
}

impl Default for FilesystemHealthSnapshot {
    fn default() -> Self {
        Self::EMPTY
    }
}

/// Physical storage facts behind one filesystem snapshot.
///
/// The three byte counts deliberately stay distinct: `physical_bytes` is the
/// whole block namespace, `volume_bytes` is the region the filesystem was
/// authorized to mount, and `occupied_bytes` includes live, retired, fallback,
/// and metadata extents that the allocator may not reuse.  Collapsing any two
/// would either overstate the filesystem's authority on a partitioned disk or
/// understate its actual allocation pressure.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct FilesystemStorageSnapshot {
    flags: U32,
    logical_block_bytes: U32,
    physical_bytes: U64,
    volume_bytes: U64,
    occupied_bytes: U64,
    max_transfer_bytes: U64,
    health: FilesystemHealthSnapshot,
}

impl FilesystemStorageSnapshot {
    const AVAILABLE: u32 = 1 << 0;
    const READ_ONLY: u32 = 1 << 1;
    const FLUSH: u32 = 1 << 2;
    const DISCARD: u32 = 1 << 3;
    /// Bits 8..16: every logical block size the device could be formatted
    /// to, bit `8 + k` = `512 << k` bytes.  Zero = unknown.
    const FORMATS_SHIFT: u32 = 8;
    const FORMATS_MASK: u32 = 0xff << Self::FORMATS_SHIFT;
    /// Bits 16..21: `log2` of the device's preferred write granularity in
    /// bytes, zero = unspecified.  Bits 21..26: the same for its optimal
    /// write size.  Packed into the flag word because the launch page that
    /// carries this record has no byte to spare (`process.rs`).
    const PREFERRED_SHIFT: u32 = 16;
    const OPTIMAL_SHIFT: u32 = 21;
    const LOG2_MASK: u32 = 0x1f;
    const LAYOUT_MASK: u32 = Self::FORMATS_MASK
        | (Self::LOG2_MASK << Self::PREFERRED_SHIFT)
        | (Self::LOG2_MASK << Self::OPTIMAL_SHIFT);
    const KNOWN_FLAGS: u32 =
        Self::AVAILABLE | Self::READ_ONLY | Self::FLUSH | Self::DISCARD | Self::LAYOUT_MASK;

    pub const EMPTY: Self = Self {
        flags: U32::new(0),
        logical_block_bytes: U32::new(0),
        physical_bytes: U64::new(0),
        volume_bytes: U64::new(0),
        occupied_bytes: U64::new(0),
        max_transfer_bytes: U64::new(0),
        health: FilesystemHealthSnapshot::EMPTY,
    };

    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        logical_block_bytes: u32,
        physical_bytes: u64,
        volume_bytes: u64,
        occupied_bytes: u64,
        max_transfer_bytes: u64,
        read_only: bool,
        flush: bool,
        discard: bool,
    ) -> Self {
        let mut flags = Self::AVAILABLE;
        if read_only {
            flags |= Self::READ_ONLY;
        }
        if flush {
            flags |= Self::FLUSH;
        }
        if discard {
            flags |= Self::DISCARD;
        }
        Self {
            flags: U32::new(flags),
            logical_block_bytes: U32::new(logical_block_bytes),
            physical_bytes: U64::new(physical_bytes),
            volume_bytes: U64::new(volume_bytes),
            occupied_bytes: U64::new(occupied_bytes),
            max_transfer_bytes: U64::new(max_transfer_bytes),
            health: FilesystemHealthSnapshot::EMPTY,
        }
    }

    /// Attach a validated live health sample to otherwise immutable geometry.
    pub const fn with_health(mut self, health: FilesystemHealthSnapshot) -> Self {
        self.health = health;
        self
    }

    /// Attach the device's layout facts: the block sizes it could be
    /// formatted to (bit `k` = `512 << k`), and its preferred and optimal
    /// write sizes in bytes (zero = unspecified).  A size that is not a
    /// power of two in `[2, 2^31]` is recorded as unspecified rather than
    /// rounded.
    pub const fn with_layout(
        mut self,
        supported_block_sizes: u32,
        preferred_write_bytes: u64,
        optimal_write_bytes: u64,
    ) -> Self {
        const fn log2_field(bytes: u64) -> u32 {
            if bytes >= 2 && bytes.is_power_of_two() && bytes <= (1 << 31) {
                bytes.trailing_zeros()
            } else {
                0
            }
        }
        let mut flags = self.flags.get() & !Self::LAYOUT_MASK;
        flags |= (supported_block_sizes & 0xff) << Self::FORMATS_SHIFT;
        flags |= log2_field(preferred_write_bytes) << Self::PREFERRED_SHIFT;
        flags |= log2_field(optimal_write_bytes) << Self::OPTIMAL_SHIFT;
        self.flags = U32::new(flags);
        self
    }

    /// Copy another record's layout facts — the router's fold, where two
    /// regions share one physical namespace.
    pub const fn with_layout_of(mut self, other: Self) -> Self {
        let flags =
            (self.flags.get() & !Self::LAYOUT_MASK) | (other.flags.get() & Self::LAYOUT_MASK);
        self.flags = U32::new(flags);
        self
    }

    /// Every block size the device could be formatted to, as a mask: bit
    /// `k` = `512 << k` bytes.  Zero = the device did not say.
    pub fn supported_block_sizes(self) -> u32 {
        (self.flags.get() & Self::FORMATS_MASK) >> Self::FORMATS_SHIFT
    }
    /// The device's preferred write granularity in bytes, if it stated one.
    pub fn preferred_write_bytes(self) -> Option<u64> {
        let log2 = (self.flags.get() >> Self::PREFERRED_SHIFT) & Self::LOG2_MASK;
        (log2 != 0).then(|| 1u64 << log2)
    }
    /// The device's optimal write size in bytes, if it stated one.
    pub fn optimal_write_bytes(self) -> Option<u64> {
        let log2 = (self.flags.get() >> Self::OPTIMAL_SHIFT) & Self::LOG2_MASK;
        (log2 != 0).then(|| 1u64 << log2)
    }

    pub fn available(self) -> bool {
        self.flags.get() & Self::AVAILABLE != 0
    }

    pub fn validate(self) -> bool {
        if self == Self::EMPTY {
            return true;
        }
        let flags = self.flags.get();
        let block = self.logical_block_bytes.get();
        flags & Self::AVAILABLE != 0
            && flags & !Self::KNOWN_FLAGS == 0
            && block != 0
            && block.is_power_of_two()
            && self.physical_bytes.get() >= self.volume_bytes.get()
            && self.volume_bytes.get() >= self.occupied_bytes.get()
            && self.max_transfer_bytes.get() >= u64::from(block)
            && self.max_transfer_bytes.get() % u64::from(block) == 0
            && self.health.validate()
    }

    pub fn read_only(self) -> bool {
        self.flags.get() & Self::READ_ONLY != 0
    }
    pub fn flush(self) -> bool {
        self.flags.get() & Self::FLUSH != 0
    }
    pub fn discard(self) -> bool {
        self.flags.get() & Self::DISCARD != 0
    }
    pub fn logical_block_bytes(self) -> u32 {
        self.logical_block_bytes.get()
    }
    pub fn physical_bytes(self) -> u64 {
        self.physical_bytes.get()
    }
    pub fn volume_bytes(self) -> u64 {
        self.volume_bytes.get()
    }
    pub fn occupied_bytes(self) -> u64 {
        self.occupied_bytes.get()
    }
    pub fn max_transfer_bytes(self) -> u64 {
        self.max_transfer_bytes.get()
    }
    pub fn health(self) -> FilesystemHealthSnapshot {
        self.health
    }
}

impl Default for FilesystemStorageSnapshot {
    fn default() -> Self {
        Self::EMPTY
    }
}

/// Block-channel completion latency and queue depth, as the channel that
/// submitted the commands measured them — a base-4 histogram over
/// nanoseconds plus the moments a reader needs to turn it into "how fast is
/// this disk at the depth we drive it".
///
/// The first measurement `res/RESIDENCY.md` §8 asks for is NVMe read latency
/// **at queue depth**, not at depth 1: `depth_sum / samples` is the mean
/// number of commands in flight when each sampled command was submitted, so
/// the latency figures beside it are already "at depth".  Every field is a
/// saturating count; a bucket that pins at `u32::MAX` says "many", never a
/// wrong number.  Zero samples is the unmeasured posture — a channel whose
/// backend publishes no clock records nothing rather than guessing.
///
/// This record rides the filesystem status wire (`FsSystemStatusWire`).  It
/// does **not** ride the launch page: the page's frozen header has 21 bytes
/// of headroom under the widest bounded reply, so the page carries only the
/// derived moments ([`FilesystemSnapshot::latency_mean_ns`] and kin) and the
/// histogram waits for a launcher that can hand `status fs` the whole wire.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct FilesystemLatencySnapshot {
    /// Deepest the channel's pending table ever got — the queue depth the
    /// device was actually driven at.  A channel's table is bounded far
    /// below this width.
    max_outstanding: U16,
    /// `buckets[i]` counts completions with latency below
    /// [`bucket_upper_ns(i)`](Self::bucket_upper_ns) and at or above the
    /// previous bound; the last bucket is open-ended.
    buckets: [U32; Self::BUCKETS],
    /// Completions whose latency was measured.
    samples: U64,
    /// Sum of measured latencies, nanoseconds.
    sum_ns: U64,
    /// Slowest measured completion, nanoseconds.
    max_ns: U64,
    /// Sum over samples of the commands in flight when each was submitted.
    depth_sum: U64,
}

impl FilesystemLatencySnapshot {
    /// Histogram width.  Eight base-4 buckets from 2 µs to 8 ms cover a
    /// healthy NVMe command (single-digit µs) through a controller stalled
    /// in garbage collection, with the open last bucket for anything worse.
    pub const BUCKETS: usize = 8;
    /// `log2` of the first bucket's upper bound in nanoseconds (`2^11` =
    /// 2048 ns).  Chosen so the fastest bucket still separates a ~2 µs NVMe
    /// read from the 8 µs bucket above it.
    pub const FIRST_UPPER_LOG2_NS: u32 = 11;
    /// `log2` of the ratio between neighbouring bucket bounds (base 4).
    pub const STEP_LOG2: u32 = 2;

    pub const EMPTY: Self = Self {
        max_outstanding: U16::new(0),
        buckets: [U32::new(0); Self::BUCKETS],
        samples: U64::new(0),
        sum_ns: U64::new(0),
        max_ns: U64::new(0),
        depth_sum: U64::new(0),
    };

    pub const fn new(
        max_outstanding: u16,
        samples: u64,
        sum_ns: u64,
        max_ns: u64,
        depth_sum: u64,
        buckets: [u32; Self::BUCKETS],
    ) -> Self {
        let mut wire = [U32::new(0); Self::BUCKETS];
        let mut i = 0;
        while i < Self::BUCKETS {
            wire[i] = U32::new(buckets[i]);
            i += 1;
        }
        Self {
            max_outstanding: U16::new(max_outstanding),
            buckets: wire,
            samples: U64::new(samples),
            sum_ns: U64::new(sum_ns),
            max_ns: U64::new(max_ns),
            depth_sum: U64::new(depth_sum),
        }
    }

    /// The bucket a latency of `ns` lands in.
    pub const fn bucket_index(ns: u64) -> usize {
        if ns < (1u64 << Self::FIRST_UPPER_LOG2_NS) {
            return 0;
        }
        let index = ((ns.ilog2() - Self::FIRST_UPPER_LOG2_NS) / Self::STEP_LOG2 + 1) as usize;
        if index >= Self::BUCKETS {
            Self::BUCKETS - 1
        } else {
            index
        }
    }

    /// The exclusive upper bound of bucket `index`, nanoseconds — `None` for
    /// the open last bucket (and for an index past the histogram).
    pub const fn bucket_upper_ns(index: usize) -> Option<u64> {
        if index + 1 >= Self::BUCKETS {
            None
        } else {
            Some(1u64 << (Self::FIRST_UPPER_LOG2_NS + Self::STEP_LOG2 * index as u32))
        }
    }

    /// Fold the sum of two histograms — the router's per-region fold.
    /// Depth takes the maximum (one shared controller queue, two logical
    /// regions); every other figure adds, saturating.
    pub fn folded_with(self, other: Self) -> Self {
        let mut buckets = [0u32; Self::BUCKETS];
        for (i, slot) in buckets.iter_mut().enumerate() {
            *slot = self.buckets[i].get().saturating_add(other.buckets[i].get());
        }
        Self::new(
            self.max_outstanding.get().max(other.max_outstanding.get()),
            self.samples.get().saturating_add(other.samples.get()),
            self.sum_ns.get().saturating_add(other.sum_ns.get()),
            self.max_ns.get().max(other.max_ns.get()),
            self.depth_sum.get().saturating_add(other.depth_sum.get()),
            buckets,
        )
    }

    /// Whether anything was measured.
    pub fn measured(self) -> bool {
        self.samples.get() != 0
    }
    pub fn max_outstanding(self) -> u16 {
        self.max_outstanding.get()
    }
    pub fn samples(self) -> u64 {
        self.samples.get()
    }
    pub fn sum_ns(self) -> u64 {
        self.sum_ns.get()
    }
    pub fn max_ns(self) -> u64 {
        self.max_ns.get()
    }
    pub fn depth_sum(self) -> u64 {
        self.depth_sum.get()
    }
    pub fn bucket(self, index: usize) -> u32 {
        self.buckets.get(index).map_or(0, |b| b.get())
    }
    /// Mean latency, nanoseconds — `None` unmeasured.
    pub fn mean_ns(self) -> Option<u64> {
        let samples = self.samples.get();
        (samples != 0).then(|| self.sum_ns.get() / samples)
    }
    /// Mean in-flight depth at submission, in thousandths — `None` unmeasured.
    pub fn mean_depth_milli(self) -> Option<u64> {
        let samples = self.samples.get();
        (samples != 0).then(|| self.depth_sum.get().saturating_mul(1000) / samples)
    }
    /// The exclusive upper bound, nanoseconds, of the bucket that holds the
    /// `permille`-th percentile sample — `Some(bound)` when that bucket is
    /// closed, `None` when nothing was measured or the percentile falls in
    /// the open last bucket ("worse than the histogram resolves").
    pub fn percentile_upper_ns(self, permille: u64) -> Option<u64> {
        let samples = self.samples.get();
        if samples == 0 {
            return None;
        }
        // The rank of the percentile sample, 1-based, rounding up so the
        // 100th percentile is the last sample and the 0th is the first.
        let rank = samples
            .saturating_mul(permille.min(1000))
            .div_ceil(1000)
            .max(1);
        let mut seen = 0u64;
        for index in 0..Self::BUCKETS {
            seen = seen.saturating_add(u64::from(self.buckets[index].get()));
            if seen >= rank {
                return Self::bucket_upper_ns(index);
            }
        }
        None
    }
}

impl Default for FilesystemLatencySnapshot {
    fn default() -> Self {
        Self::EMPTY
    }
}

/// Filesystem-service posture and load, as the launcher captured it.
///
/// Aggregates only: no name, path, scope, or per-session figure — the record
/// says how the service is doing, never what anyone is doing with it. The
/// `posture` bits mirror `FsSystemStatusWire`'s, which is where this is
/// lifted from.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct FilesystemSnapshot {
    flags: U16,
    posture: U16,
    sessions: U16,
    peers: U16,
    views_private: U16,
    views_pinned: U16,
    windows: U16,
    user_lifecycle: u8,
    system_lifecycle: u8,
    objects: U32,
    edges: U32,
    bytes: U64,
    storage: FilesystemStorageSnapshot,
    commits: U64,
    reads: U64,
    writes: U64,
    flushes: U64,
    rollbacks: U64,
    late: U64,
    parks: U64,
    fires: U64,
    spins: U64,
    expired_waits: U64,
    slow_waits: U64,
    leaked_placements: U64,
    abandoned_reclaims: U64,
    // ── ABI 10: the completion-latency moments, derived by the launcher
    // from the status wire, in the 20 bytes the launch page's header has
    // left under its widest bounded reply.  Every word saturates.  The
    // histogram and the channel census stay on the wire.
    /// Completions whose latency was measured (saturating).
    latency_samples: U32,
    /// Mean completion latency, nanoseconds (saturating at ~4.29 s).
    latency_mean_ns: U32,
    /// Slowest measured completion, nanoseconds (saturating at ~4.29 s).
    latency_max_ns: U32,
    /// Mean commands in flight at submission, in thousandths (saturating).
    mean_depth_milli: U16,
    /// Deepest the channel's pending table ever got.
    max_outstanding: U16,
    /// Commands submitted and not yet completed at capture (saturating).
    in_flight: U16,
    // ── ABI 11: the read-ahead hit share, in the one word ABI 10 held in
    // reserve.  The counts themselves ride the wire.
    /// Payload reads served from the provider's held read-ahead window, per
    /// thousand payload reads that reached the provider, stored plus one so
    /// that zero — the page's "nothing published" value — means no payload
    /// read has been served yet.
    readahead_share: U16,
}

impl FilesystemSnapshot {
    /// A durable volume is mounted (unset: the RAM posture).
    pub const DURABLE: u16 = 1 << 0;
    /// The durable mount recovered degraded and is read-only.
    pub const DEGRADED_READ_ONLY: u16 = 1 << 1;
    /// The realized interrupt route is held (unset: completions are polled).
    pub const ROUTE_HELD: u16 = 1 << 2;
    /// The administration workspace anchor stands.
    pub const WORKSPACE: u16 = 1 << 3;
    /// The pinned system invocation view stands.
    pub const SYSTEM_VIEW: u16 = 1 << 4;
    /// The durable block channel crossed its one-way lost boundary.
    pub const CHANNEL_LOST: u16 = 1 << 5;
    /// The controller holds a host memory buffer for its mapping tables.
    pub const HOST_MEMORY_BUFFER: u16 = 1 << 6;

    pub const LIFECYCLE_UNKNOWN: u8 = crate::svc::FsSystemStatusWire::LIFECYCLE_UNKNOWN as u8;
    pub const LIFECYCLE_UNDESIGNATED: u8 =
        crate::svc::FsSystemStatusWire::LIFECYCLE_UNDESIGNATED as u8;
    pub const LIFECYCLE_REFUSED: u8 = crate::svc::FsSystemStatusWire::LIFECYCLE_REFUSED as u8;
    pub const LIFECYCLE_REFORMATTING: u8 =
        crate::svc::FsSystemStatusWire::LIFECYCLE_REFORMATTING as u8;
    pub const LIFECYCLE_RESEEDING: u8 = crate::svc::FsSystemStatusWire::LIFECYCLE_RESEEDING as u8;
    pub const LIFECYCLE_UNAVAILABLE: u8 =
        crate::svc::FsSystemStatusWire::LIFECYCLE_UNAVAILABLE as u8;
    pub const LIFECYCLE_REBUILT: u8 = crate::svc::FsSystemStatusWire::LIFECYCLE_REBUILT as u8;
    pub const LIFECYCLE_CURRENT: u8 = crate::svc::FsSystemStatusWire::LIFECYCLE_CURRENT as u8;

    pub const EMPTY: Self = Self {
        flags: U16::new(0),
        posture: U16::new(0),
        sessions: U16::new(0),
        peers: U16::new(0),
        views_private: U16::new(0),
        views_pinned: U16::new(0),
        windows: U16::new(0),
        user_lifecycle: 0,
        system_lifecycle: 0,
        objects: U32::new(0),
        edges: U32::new(0),
        bytes: U64::new(0),
        storage: FilesystemStorageSnapshot::EMPTY,
        commits: U64::new(0),
        reads: U64::new(0),
        writes: U64::new(0),
        flushes: U64::new(0),
        rollbacks: U64::new(0),
        late: U64::new(0),
        parks: U64::new(0),
        fires: U64::new(0),
        spins: U64::new(0),
        expired_waits: U64::new(0),
        slow_waits: U64::new(0),
        leaked_placements: U64::new(0),
        abandoned_reclaims: U64::new(0),
        latency_samples: U32::new(0),
        latency_mean_ns: U32::new(0),
        latency_max_ns: U32::new(0),
        mean_depth_milli: U16::new(0),
        max_outstanding: U16::new(0),
        in_flight: U16::new(0),
        readahead_share: U16::new(0),
    };

    /// `io` = reads, writes, flushes, rollbacks, late; `pacing` = parks,
    /// fires, spins; `pressure` = expired, slow, leaked, abandoned;
    /// `latency` = the wire histogram, of which the page keeps the derived
    /// moments; `in_flight` = submissions − completions at capture;
    /// `readahead` = window hits, window-filling device reads.
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        posture: u16,
        sessions: u16,
        peers: u16,
        views_private: u16,
        views_pinned: u16,
        windows: u16,
        user_lifecycle: u8,
        system_lifecycle: u8,
        objects: u32,
        edges: u32,
        bytes: u64,
        storage: FilesystemStorageSnapshot,
        commits: u64,
        io: [u64; 5],
        pacing: [u64; 3],
        pressure: [u64; 4],
        latency: FilesystemLatencySnapshot,
        in_flight: u64,
        readahead: [u64; 2],
    ) -> Self {
        const fn narrow32(value: u64) -> U32 {
            U32::new(if value > u32::MAX as u64 {
                u32::MAX
            } else {
                value as u32
            })
        }
        const fn narrow16(value: u64) -> U16 {
            U16::new(if value > u16::MAX as u64 {
                u16::MAX
            } else {
                value as u16
            })
        }
        let served = readahead[0].saturating_add(readahead[1]);
        let readahead_share = if served == 0 {
            0
        } else {
            // hits ≤ served, so the quotient is at most 1000.
            (readahead[0].saturating_mul(1000) / served) as u16 + 1
        };
        let samples = latency.samples.get();
        let (mean_ns, mean_depth_milli) = if samples == 0 {
            (0, 0)
        } else {
            (
                latency.sum_ns.get() / samples,
                latency.depth_sum.get().saturating_mul(1000) / samples,
            )
        };
        Self {
            flags: U16::new(AVAILABLE_U16),
            posture: U16::new(posture),
            sessions: U16::new(sessions),
            peers: U16::new(peers),
            views_private: U16::new(views_private),
            views_pinned: U16::new(views_pinned),
            windows: U16::new(windows),
            user_lifecycle,
            system_lifecycle,
            objects: U32::new(objects),
            edges: U32::new(edges),
            bytes: U64::new(bytes),
            storage,
            commits: U64::new(commits),
            reads: U64::new(io[0]),
            writes: U64::new(io[1]),
            flushes: U64::new(io[2]),
            rollbacks: U64::new(io[3]),
            late: U64::new(io[4]),
            parks: U64::new(pacing[0]),
            fires: U64::new(pacing[1]),
            spins: U64::new(pacing[2]),
            expired_waits: U64::new(pressure[0]),
            slow_waits: U64::new(pressure[1]),
            leaked_placements: U64::new(pressure[2]),
            abandoned_reclaims: U64::new(pressure[3]),
            latency_samples: narrow32(samples),
            latency_mean_ns: narrow32(mean_ns),
            latency_max_ns: narrow32(latency.max_ns.get()),
            mean_depth_milli: narrow16(mean_depth_milli),
            max_outstanding: latency.max_outstanding,
            in_flight: narrow16(in_flight),
            readahead_share: U16::new(readahead_share),
        }
    }

    pub fn available(self) -> bool {
        self.flags.get() == AVAILABLE_U16
    }
    pub fn validate(self) -> bool {
        self.readahead_share.get() <= 1001
            && (self == Self::EMPTY
                || (self.available()
                    && self.storage.validate()
                    && self.durable() == self.storage.available()))
    }
    pub fn durable(self) -> bool {
        self.posture.get() & Self::DURABLE != 0
    }
    pub fn degraded_read_only(self) -> bool {
        self.posture.get() & Self::DEGRADED_READ_ONLY != 0
    }
    pub fn route_held(self) -> bool {
        self.posture.get() & Self::ROUTE_HELD != 0
    }
    pub fn workspace(self) -> bool {
        self.posture.get() & Self::WORKSPACE != 0
    }
    pub fn system_view(self) -> bool {
        self.posture.get() & Self::SYSTEM_VIEW != 0
    }
    pub fn channel_lost(self) -> bool {
        self.posture.get() & Self::CHANNEL_LOST != 0
    }
    pub fn host_memory_buffer(self) -> bool {
        self.posture.get() & Self::HOST_MEMORY_BUFFER != 0
    }
    pub fn sessions(self) -> u16 {
        self.sessions.get()
    }
    pub fn peers(self) -> u16 {
        self.peers.get()
    }
    pub fn views_private(self) -> u16 {
        self.views_private.get()
    }
    pub fn views_pinned(self) -> u16 {
        self.views_pinned.get()
    }
    pub fn windows(self) -> u16 {
        self.windows.get()
    }
    pub fn user_lifecycle(self) -> u8 {
        self.user_lifecycle
    }
    pub fn system_lifecycle(self) -> u8 {
        self.system_lifecycle
    }
    pub fn objects(self) -> u32 {
        self.objects.get()
    }
    pub fn edges(self) -> u32 {
        self.edges.get()
    }
    pub fn bytes(self) -> u64 {
        self.bytes.get()
    }
    pub fn storage(self) -> FilesystemStorageSnapshot {
        self.storage
    }
    pub fn commits(self) -> u64 {
        self.commits.get()
    }
    pub fn reads(self) -> u64 {
        self.reads.get()
    }
    pub fn writes(self) -> u64 {
        self.writes.get()
    }
    pub fn flushes(self) -> u64 {
        self.flushes.get()
    }
    pub fn rollbacks(self) -> u64 {
        self.rollbacks.get()
    }
    pub fn late(self) -> u64 {
        self.late.get()
    }
    pub fn parks(self) -> u64 {
        self.parks.get()
    }
    /// Route parks the interrupt actually woke. A gap against
    /// [`parks`](Self::parks) is delivery the route did not make.
    pub fn fires(self) -> u64 {
        self.fires.get()
    }
    /// Waits that took the bounded spin instead of a route park.
    pub fn spins(self) -> u64 {
        self.spins.get()
    }
    /// Completion waits that ran one backend budget out empty — one per
    /// round, so a command that was slow for several budgets counts each.
    pub fn expired_waits(self) -> u64 {
        self.expired_waits.get()
    }
    /// Of those, the rounds where the device still reported itself healthy
    /// and was therefore waited out (up to the provider's stall ceiling)
    /// rather than condemned; `expired - slow` is the condemned count.
    pub fn slow_waits(self) -> u64 {
        self.slow_waits.get()
    }
    /// Provider extents made unreachable but not released.
    pub fn leaked_placements(self) -> u64 {
        self.leaked_placements.get()
    }
    /// Namespace reclaim walks abandoned after a structural or contention bound.
    pub fn abandoned_reclaims(self) -> u64 {
        self.abandoned_reclaims.get()
    }
    /// Whether any completion latency was measured (the backend published
    /// a clock).
    pub fn latency_measured(self) -> bool {
        self.latency_samples.get() != 0
    }
    pub fn latency_samples(self) -> u32 {
        self.latency_samples.get()
    }
    /// Mean completion latency, nanoseconds — `None` unmeasured.
    pub fn latency_mean_ns(self) -> Option<u32> {
        self.latency_measured().then(|| self.latency_mean_ns.get())
    }
    /// Slowest measured completion, nanoseconds.
    pub fn latency_max_ns(self) -> u32 {
        self.latency_max_ns.get()
    }
    /// Mean commands in flight at submission, in thousandths — `None`
    /// unmeasured.  The depth the latency figures were measured at.
    pub fn mean_depth_milli(self) -> Option<u16> {
        self.latency_measured().then(|| self.mean_depth_milli.get())
    }
    /// Deepest the channel's pending table ever got.
    pub fn max_outstanding(self) -> u16 {
        self.max_outstanding.get()
    }
    /// Commands in flight when the snapshot was taken.
    pub fn in_flight(self) -> u16 {
        self.in_flight.get()
    }
    /// Share of payload reads the provider answered from its held
    /// read-ahead window, per thousand — `None` until a payload read has
    /// been served.
    pub fn readahead_permille(self) -> Option<u16> {
        self.readahead_share.get().checked_sub(1)
    }
}

/// Kernel-owned manifold snapshot (MANIFOLD §3, §6, §14.4) — the policy layer's
/// own testimony, and on a GUI-only machine the only way to see it.
///
/// Every quantity is integer-scaled: the wire carries no floats, and the
/// renderer needs no FP.  Scales are chosen so the interesting range is legible
/// as an integer — α to a millionth (theory values are exactly 1.0 and 2.0),
/// utilisation in parts-per-million, pressure in thousandths.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct ManifoldSnapshot {
    flags: U32,
    /// Barrier exponent `α_cpu` × 10⁶ (MANIFOLD §6.3).
    alpha_cpu_micro: U32,
    /// Barrier exponent `α_mem` × 10⁶.
    alpha_mem_micro: U32,
    /// The busiest domain slot's `Re(Z_cpu)` × 10⁶ — sustained CPU
    /// utilisation (§5).  Swept across slots, because a UMA machine's node
    /// lives in the dedicated *final* slot, not slot 0.
    node_re_cpu_ppm: U32,
    /// The busiest domain slot's `Re(Z_mem)` × 10⁶.
    node_re_mem_ppm: U32,
    /// Tasks the last sweep saw carrying volatility (`Im ≠ 0`) — the §4
    /// behavioural chain's liveness witness.
    volatile_tasks: U32,
    /// Tasks the last sweep sampled at all, so `volatile` has a denominator.
    sampled_tasks: U32,
    /// Highest per-task Markov transition rate × 10⁶ (§3.3's `q`).
    max_rate_ppm: U32,
    /// The worst domain slot's joint pressure `𝒫` × 1000, **saturating**.  Pressure spans
    /// tens of orders of magnitude near the wall, so saturation is the honest
    /// encoding; [`pressure_finite`](Self::pressure_finite) is what says
    /// whether the §6.1 barrier held at all.
    node_pressure_milli: U64,
}

impl ManifoldSnapshot {
    /// `α_cpu` was measured by boot calibration rather than taken from theory.
    const ALPHA_CPU_MEASURED: u32 = 1 << 1;
    /// `α_mem` was measured by boot calibration rather than taken from theory.
    const ALPHA_MEM_MEASURED: u32 = 1 << 2;
    /// Every domain slot's joint pressure is a finite number.
    ///
    /// This bit is the regression witness for the §6.1 guard: the clamp exists
    /// so the barrier stays finite, and when the denominator underflowed to
    /// zero the pressure came back `∞` at exactly the load the clamp was meant
    /// to survive.  Clear here means that is happening again.
    const PRESSURE_FINITE: u32 = 1 << 3;

    pub const EMPTY: Self = Self {
        flags: U32::new(0),
        alpha_cpu_micro: U32::new(0),
        alpha_mem_micro: U32::new(0),
        node_re_cpu_ppm: U32::new(0),
        node_re_mem_ppm: U32::new(0),
        volatile_tasks: U32::new(0),
        sampled_tasks: U32::new(0),
        max_rate_ppm: U32::new(0),
        node_pressure_milli: U64::new(0),
    };

    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        alpha_cpu_micro: u32,
        alpha_mem_micro: u32,
        node_re_cpu_ppm: u32,
        node_re_mem_ppm: u32,
        volatile_tasks: u32,
        sampled_tasks: u32,
        max_rate_ppm: u32,
        node_pressure_milli: u64,
        detail: u32,
    ) -> Self {
        Self {
            flags: U32::new(AVAILABLE | detail),
            alpha_cpu_micro: U32::new(alpha_cpu_micro),
            alpha_mem_micro: U32::new(alpha_mem_micro),
            node_re_cpu_ppm: U32::new(node_re_cpu_ppm),
            node_re_mem_ppm: U32::new(node_re_mem_ppm),
            volatile_tasks: U32::new(volatile_tasks),
            sampled_tasks: U32::new(sampled_tasks),
            max_rate_ppm: U32::new(max_rate_ppm),
            node_pressure_milli: U64::new(node_pressure_milli),
        }
    }

    pub fn available(self) -> bool {
        self.flags.get() & AVAILABLE == AVAILABLE
    }
    pub fn validate(self) -> bool {
        self.available() || self == Self::EMPTY
    }
    pub fn alpha_cpu_micro(self) -> u32 {
        self.alpha_cpu_micro.get()
    }
    pub fn alpha_mem_micro(self) -> u32 {
        self.alpha_mem_micro.get()
    }
    pub fn alpha_cpu_measured(self) -> bool {
        self.flags.get() & Self::ALPHA_CPU_MEASURED != 0
    }
    pub fn alpha_mem_measured(self) -> bool {
        self.flags.get() & Self::ALPHA_MEM_MEASURED != 0
    }
    pub fn node_re_cpu_ppm(self) -> u32 {
        self.node_re_cpu_ppm.get()
    }
    pub fn node_re_mem_ppm(self) -> u32 {
        self.node_re_mem_ppm.get()
    }
    pub fn volatile_tasks(self) -> u32 {
        self.volatile_tasks.get()
    }
    pub fn sampled_tasks(self) -> u32 {
        self.sampled_tasks.get()
    }
    pub fn max_rate_ppm(self) -> u32 {
        self.max_rate_ppm.get()
    }
    pub fn node_pressure_milli(self) -> u64 {
        self.node_pressure_milli.get()
    }
    pub fn pressure_finite(self) -> bool {
        self.flags.get() & Self::PRESSURE_FINITE != 0
    }

    /// Assemble from the three `SystemMetric` observations, which pack two
    /// scaled `u32`s per `value` word.
    pub const fn from_metrics(
        barrier: (u64, u64),
        field: (u64, u64),
        volatility: (u64, u64),
    ) -> Self {
        Self::new(
            barrier.0 as u32,
            (barrier.0 >> 32) as u32,
            field.0 as u32,
            (field.0 >> 32) as u32,
            volatility.0 as u32,
            (volatility.0 >> 32) as u32,
            volatility.1 as u32,
            field.1,
            barrier.1 as u32,
        )
    }
}

/// Frame-service posture, as the launcher captured it: the link's facts
/// (what `Query` answers) and the bound radio's (what `Radio` answers),
/// each present only when the row answered that verb.
///
/// Facts only — no vif, no client, no counter that names anyone, and no
/// protocol constant either (the frame size and window geometry `Query`
/// also answers are the service's compile-time contract, not the machine's
/// state). The counters and radio blocks are [`crate::svc::NetCountersWire`]
/// and [`crate::svc::NetRadioWire`] verbatim: one definition of those
/// fields, shared with the reply that carried them, so the snapshot cannot
/// drift from the wire.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct NetSnapshot {
    flags: U16,
    posture: U16,
    mac: [u8; 6],
    reserved: [u8; 2],
    link_mbps: U32,
    counters: crate::svc::NetCountersWire,
    radio: crate::svc::NetRadioWire,
}

/// What `Query` answered, as [`NetSnapshot::new`] takes it: the link's
/// facts and the device's counters, together because the reply carries
/// them together.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NetLinkFacts {
    /// The link has carrier.
    pub up: bool,
    /// Negotiated rate in megabits per second; 0 when unknown.
    pub mbps: u32,
    /// The device's own hardware address.
    pub mac: [u8; 6],
    /// The device's counters at the time of the answer.
    pub counters: crate::svc::NetCountersWire,
}

impl NetSnapshot {
    /// The row answered `Query`: `mac`, `link_mbps` and
    /// [`LINK_UP`](Self::LINK_UP) are facts. Unset: the row serves but
    /// lends no frame path (a radio at #388 D1, or a row with no device).
    pub const LINK_INFO: u16 = 1 << 0;
    /// The link has carrier.
    pub const LINK_UP: u16 = 1 << 1;
    /// The row answered `Radio`: the radio block is a fact.
    pub const RADIO: u16 = 1 << 2;
    const KNOWN: u16 = Self::LINK_INFO | Self::LINK_UP | Self::RADIO;

    pub const EMPTY: Self = Self {
        flags: U16::new(0),
        posture: U16::new(0),
        mac: [0; 6],
        reserved: [0; 2],
        link_mbps: U32::new(0),
        counters: crate::svc::NetCountersWire::EMPTY,
        radio: crate::svc::NetRadioWire::EMPTY,
    };

    /// The row answered. `link` is `Query`'s answer when it was `Info`
    /// (link up, rate, address, counters); `radio` is `Radio`'s when it was
    /// a report.
    pub const fn new(link: Option<NetLinkFacts>, radio: Option<crate::svc::NetRadioWire>) -> Self {
        let mut posture = 0;
        let (mut mac, mut link_mbps) = ([0; 6], 0);
        let mut counters = crate::svc::NetCountersWire::EMPTY;
        if let Some(facts) = link {
            posture |= Self::LINK_INFO;
            if facts.up {
                posture |= Self::LINK_UP;
            }
            mac = facts.mac;
            link_mbps = facts.mbps;
            counters = facts.counters;
        }
        let radio = match radio {
            Some(block) => {
                posture |= Self::RADIO;
                block
            }
            None => crate::svc::NetRadioWire::EMPTY,
        };
        Self {
            flags: U16::new(AVAILABLE_U16),
            posture: U16::new(posture),
            mac,
            reserved: [0; 2],
            link_mbps: U32::new(link_mbps),
            counters,
            radio,
        }
    }

    pub fn available(self) -> bool {
        self.flags.get() == AVAILABLE_U16
    }
    /// Empty, or available with only known posture bits, no reserved byte
    /// set, and every fact zero whose posture bit says it is not a fact.
    pub fn validate(self) -> bool {
        if self == Self::EMPTY {
            return true;
        }
        let posture = self.posture.get();
        self.available()
            && posture & !Self::KNOWN == 0
            && self.reserved == [0; 2]
            && (self.link_info()
                || (self.mac == [0; 6]
                    && self.link_mbps.get() == 0
                    && !self.link_up()
                    && self.counters == crate::svc::NetCountersWire::EMPTY))
            && (self.radio_present() || self.radio == crate::svc::NetRadioWire::EMPTY)
    }
    pub fn link_info(self) -> bool {
        self.posture.get() & Self::LINK_INFO != 0
    }
    pub fn link_up(self) -> bool {
        self.posture.get() & Self::LINK_UP != 0
    }
    pub fn link_mbps(self) -> u32 {
        self.link_mbps.get()
    }
    pub fn mac(self) -> [u8; 6] {
        self.mac
    }
    /// The device's counters; meaningful only when [`link_info`](Self::link_info).
    pub fn counters(self) -> crate::svc::NetCountersWire {
        self.counters
    }
    pub fn radio_present(self) -> bool {
        self.posture.get() & Self::RADIO != 0
    }
    /// The radio block; meaningful only when [`radio_present`](Self::radio_present).
    pub fn radio(self) -> crate::svc::NetRadioWire {
        self.radio
    }

    /// A record whose posture says the link is not a fact while its address
    /// is set — the shape [`validate`](Self::validate) exists to refuse.
    #[cfg(test)]
    pub(crate) fn lying_for_test() -> Self {
        let mut lying = Self::new(None, Some(crate::svc::NetRadioWire::EMPTY));
        lying.mac = [1; 6];
        lying
    }
}

/// One federated snapshot assembled by the monitor service.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct TelemetrySnapshot {
    pub system: SystemSnapshot,
    pub memory: MemorySnapshot,
    pub exec: ProcessSnapshot,
    pub manifold: ManifoldSnapshot,
}

impl TelemetrySnapshot {
    pub const EMPTY: Self = Self {
        system: SystemSnapshot::EMPTY,
        memory: MemorySnapshot::EMPTY,
        exec: ProcessSnapshot::EMPTY,
        manifold: ManifoldSnapshot::EMPTY,
    };

    pub const fn exec_only(exec: ProcessSnapshot) -> Self {
        Self {
            system: SystemSnapshot::EMPTY,
            memory: MemorySnapshot::EMPTY,
            exec,
            manifold: ManifoldSnapshot::EMPTY,
        }
    }

    pub const fn new(
        system: SystemSnapshot,
        memory: MemorySnapshot,
        exec: ProcessSnapshot,
        manifold: ManifoldSnapshot,
    ) -> Self {
        Self {
            system,
            memory,
            exec,
            manifold,
        }
    }

    pub fn validate(self) -> bool {
        self.system.validate()
            && self.memory.validate()
            && self.exec.validate()
            && self.manifold.validate()
    }
}

const _: () = {
    assert!(core::mem::size_of::<SystemSnapshot>() == 104);
    assert!(core::mem::size_of::<MemorySnapshot>() == 64);
    assert!(core::mem::size_of::<FilesystemHealthSnapshot>() == 40);
    assert!(core::mem::size_of::<FilesystemStorageSnapshot>() == 80);
    assert!(core::mem::size_of::<FilesystemLatencySnapshot>() == 2 + 8 * 4 + 4 * 8); // 66
    // 216 through `abandoned_reclaims`, then three 32-bit latency moments
    // and four 16-bit depth words.
    assert!(core::mem::size_of::<FilesystemSnapshot>() == 216 + 3 * 4 + 4 * 2); // 236
    // 16-byte link head (2 reserved) ‖ 24-byte counters ‖ 36-byte radio block.
    assert!(core::mem::size_of::<NetSnapshot>() == 76);
    assert!(core::mem::size_of::<ManifoldSnapshot>() == 40);
    assert!(core::mem::size_of::<TelemetrySnapshot>() == 240);
};

#[cfg(test)]
mod tests {
    use super::*;

    /// Every fact the posture bits disown must be zero, every reserved byte
    /// must be zero, and no unknown posture bit may be set.
    #[test]
    fn the_net_snapshot_validates_its_posture() {
        use crate::svc::NetRadioWire;
        assert!(NetSnapshot::EMPTY.validate());
        assert!(!NetSnapshot::EMPTY.available());
        let radio_only = NetSnapshot::new(None, Some(NetRadioWire::EMPTY));
        assert!(radio_only.validate() && radio_only.radio_present() && !radio_only.link_info());
        assert!(!NetSnapshot::lying_for_test().validate());

        let mut stray = NetSnapshot::new(
            Some(NetLinkFacts {
                up: false,
                mbps: 0,
                mac: [0; 6],
                counters: crate::svc::NetCountersWire::EMPTY,
            }),
            None,
        );
        assert!(stray.validate() && stray.link_info() && !stray.link_up());
        stray.reserved[1] = 1;
        assert!(!stray.validate(), "a reserved byte");
        let mut unknown = NetSnapshot::new(None, None);
        assert!(unknown.validate(), "answered, but neither verb was a fact");
        unknown.posture = U16::new(1 << 15);
        assert!(!unknown.validate(), "an unknown posture bit");
        let mut ghost = NetSnapshot::new(None, None);
        ghost.radio.cut = 3;
        assert!(!ghost.validate(), "a radio byte with no radio");
        let mut counted = NetSnapshot::new(None, None);
        counted.counters.irqs = U64::new(1);
        assert!(!counted.validate(), "a counter with no link facts");
        let facts = NetLinkFacts {
            up: true,
            mbps: 1000,
            mac: [1, 2, 3, 4, 5, 6],
            counters: crate::svc::NetCountersWire {
                irqs: U64::new(77),
                link_changes: U32::new(1),
                overruns: U32::new(0),
                rx_errors: U32::new(2),
                dropped_notices: U32::new(0),
            },
        };
        let live = NetSnapshot::new(Some(facts), None);
        assert!(live.validate());
        assert_eq!(
            live.counters(),
            facts.counters,
            "the counters ride the link facts"
        );
    }

    #[test]
    fn storage_layout_facts_pack_into_the_flag_word() {
        let base =
            FilesystemStorageSnapshot::new(512, 1 << 30, 1 << 20, 0, 256 << 10, false, true, true);
        assert_eq!(base.supported_block_sizes(), 0);
        assert_eq!(base.preferred_write_bytes(), None);
        let laid = base.with_layout((1 << 0) | (1 << 3), 4096, 128 << 10);
        assert!(laid.validate());
        assert_eq!(laid.supported_block_sizes(), 0b1001);
        assert_eq!(laid.preferred_write_bytes(), Some(4096));
        assert_eq!(laid.optimal_write_bytes(), Some(128 << 10));
        assert!(laid.flush() && laid.discard() && !laid.read_only());
        // Not a power of two, or out of range: unspecified, never rounded.
        let odd = base.with_layout(0, 3000, 1 << 40);
        assert_eq!(odd.preferred_write_bytes(), None);
        assert_eq!(odd.optimal_write_bytes(), None);
        // The fold copies layout without touching the rest.
        let other = FilesystemStorageSnapshot::new(512, 1, 1, 1, 512, true, false, false)
            .with_layout_of(laid);
        assert!(other.read_only() && !other.flush());
        assert_eq!(other.optimal_write_bytes(), Some(128 << 10));
        assert!(other.validate());
    }

    #[test]
    fn latency_buckets_are_base_four_from_two_microseconds() {
        type L = FilesystemLatencySnapshot;
        assert_eq!(L::bucket_index(0), 0);
        assert_eq!(L::bucket_index(2_047), 0);
        assert_eq!(L::bucket_index(2_048), 1);
        assert_eq!(L::bucket_index(8_191), 1);
        assert_eq!(L::bucket_index(8_192), 2);
        assert_eq!(L::bucket_index((1 << 21) - 1), 5);
        assert_eq!(L::bucket_index(1 << 21), 6);
        assert_eq!(L::bucket_index(1 << 23), L::BUCKETS - 1);
        assert_eq!(L::bucket_index(u64::MAX), L::BUCKETS - 1);
        assert_eq!(L::bucket_upper_ns(0), Some(2_048));
        assert_eq!(L::bucket_upper_ns(1), Some(8_192));
        assert_eq!(L::bucket_upper_ns(6), Some(1 << 23));
        assert_eq!(L::bucket_upper_ns(7), None);
        assert_eq!(L::bucket_upper_ns(99), None);
        // Every bound is the previous one times four, so the index and the
        // bound agree on every edge.
        for i in 0..L::BUCKETS - 1 {
            let upper = L::bucket_upper_ns(i).unwrap();
            assert_eq!(L::bucket_index(upper - 1), i);
            assert_eq!(L::bucket_index(upper), i + 1);
        }
    }

    #[test]
    fn latency_percentiles_and_folds_read_off_the_histogram() {
        type L = FilesystemLatencySnapshot;
        assert_eq!(L::EMPTY.percentile_upper_ns(500), None);
        assert_eq!(L::EMPTY.mean_ns(), None);
        let mut buckets = [0u32; L::BUCKETS];
        buckets[0] = 90; // < 2 µs
        buckets[2] = 9; // < 32 µs
        buckets[7] = 1; // open: worse than 8 ms
        let a = L::new(4, 100, 100 * 1_500, 40_000_000, 250, buckets);
        assert_eq!(a.percentile_upper_ns(0), Some(2_048));
        assert_eq!(a.percentile_upper_ns(500), Some(2_048));
        assert_eq!(a.percentile_upper_ns(900), Some(2_048));
        assert_eq!(a.percentile_upper_ns(990), Some(32_768));
        assert_eq!(
            a.percentile_upper_ns(1000),
            None,
            "the slowest sample is off the scale"
        );
        assert_eq!(a.mean_depth_milli(), Some(2_500));
        // Folding two regions: depth takes the max, everything else adds,
        // and a bucket pins rather than wrapping.
        let mut hot = [u32::MAX; L::BUCKETS];
        hot[3] = 1;
        let b = L::new(7, 3, 30, 20, 9, hot);
        let folded = a.folded_with(b);
        assert_eq!(folded.max_outstanding(), 7);
        assert_eq!(folded.samples(), 103);
        assert_eq!(folded.max_ns(), 40_000_000);
        assert_eq!(folded.bucket(0), u32::MAX);
        assert_eq!(folded.bucket(3), 1);
        assert!(FilesystemSnapshot::EMPTY.validate());
    }

    #[test]
    fn provider_snapshots_validate_only_defined_postures() {
        let system = SystemSnapshot::new(
            4, 1_000, 500, 16_384, 8_192, 4_096, 2_048, 1_024, 0, 0, 3, 0, 0,
        );
        assert_eq!(
            system.fabric_domains(),
            3,
            "the fabric domain count round-trips"
        );
        const TEST_EXTENT_BYTES: u64 = 96 << 20;
        let system = system.with_fabric(3, 128, 2, 1, TEST_EXTENT_BYTES);
        assert_eq!(system.fabric_capacity(), 128);
        assert_eq!(system.fabric_lendable_domains(), 2);
        assert_eq!(system.fabric_fence_required_domains(), 1);
        assert_eq!(system.fabric_extent_bytes(), TEST_EXTENT_BYTES);
        assert!(system.validate());
        assert!(
            !SystemSnapshot::new(
                4, 1_000, 500, 16_384, 8_192, 4_096, 2_048, 1_024, 0, 0, 0, 0, 0
            )
            .with_fabric(3, 128, 4, 1, 0)
            .validate(),
            "the snapshot builder must not hide an impossible fabric census"
        );
        let memory = MemorySnapshot::new(8, 3, 10, 7, 2, 0, 1, 12_288);
        let exec = ProcessSnapshot::new(32, 3, 1, 0, 9, 2, 6);

        assert!(TelemetrySnapshot::new(system, memory, exec, ManifoldSnapshot::EMPTY).validate());
        assert!(TelemetrySnapshot::exec_only(exec).validate());

        let mut corrupt_system = system;
        corrupt_system.flags = U32::new(2);
        assert!(
            !TelemetrySnapshot::new(corrupt_system, memory, exec, ManifoldSnapshot::EMPTY)
                .validate()
        );

        let mut corrupt_memory = memory;
        corrupt_memory.reserved = U32::new(1);
        assert!(
            !TelemetrySnapshot::new(system, corrupt_memory, exec, ManifoldSnapshot::EMPTY)
                .validate()
        );
    }

    #[test]
    fn filesystem_storage_keeps_namespace_volume_and_allocation_distinct() {
        let health = FilesystemHealthSnapshot::new(0, 36, 98, 10, 3, 7, 123, 2, 0);
        let storage = FilesystemStorageSnapshot::new(
            4096,
            1 << 30,
            512 << 20,
            128 << 20,
            256 << 10,
            false,
            true,
            true,
        )
        .with_health(health);
        assert!(storage.validate());
        assert_eq!(storage.physical_bytes(), 1 << 30);
        assert_eq!(storage.volume_bytes(), 512 << 20);
        assert_eq!(storage.occupied_bytes(), 128 << 20);
        assert!(storage.flush() && storage.discard() && !storage.read_only());
        assert!(storage.health().available());
        assert_eq!(storage.health().temperature_celsius(), 36);
        assert_eq!(storage.health().power_on_hours(), 123);

        let invalid = FilesystemStorageSnapshot::new(
            4096,
            128 << 20,
            256 << 20,
            0,
            256 << 10,
            false,
            true,
            false,
        );
        assert!(!invalid.validate());
    }
}
