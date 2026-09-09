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

/// Kernel-owned system snapshot: scheduler time, physical memory and heap.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct SystemSnapshot {
    flags: U32,
    cpu_count: U32,
    /// Live memory-fabric domain registrations (#697) — repurposes one of the
    /// two former reserved words as the `status fab` health surface.
    fabric_domains: U32,
    reserved: U32,
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
}

impl SystemSnapshot {
    pub const EMPTY: Self = Self {
        flags: U32::new(0),
        cpu_count: U32::new(0),
        fabric_domains: U32::new(0),
        reserved: U32::new(0),
        captured_ns: U64::new(0),
        cpu_busy_ns: U64::new(0),
        phys_total: U64::new(0),
        phys_free: U64::new(0),
        heap_capacity: U64::new(0),
        heap_used: U64::new(0),
        heap_largest: U64::new(0),
        iommu_faults: U64::new(0),
        iommu_last: U64::new(0),
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
    ) -> Self {
        Self {
            flags: U32::new(AVAILABLE),
            cpu_count: U32::new(cpu_count),
            fabric_domains: U32::new(fabric_domains),
            reserved: U32::new(0),
            captured_ns: U64::new(captured_ns),
            cpu_busy_ns: U64::new(cpu_busy_ns),
            phys_total: U64::new(phys_total),
            phys_free: U64::new(phys_free),
            heap_capacity: U64::new(heap_capacity),
            heap_used: U64::new(heap_used),
            heap_largest: U64::new(heap_largest),
            iommu_faults: U64::new(iommu_faults),
            iommu_last: U64::new(iommu_last),
        }
    }

    pub fn available(self) -> bool {
        self.flags.get() == AVAILABLE
    }
    pub fn validate(self) -> bool {
        self.reserved.get() == 0 && (self.available() || self == Self::EMPTY)
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
    /// Live memory-fabric domain registrations (#697). Zero until a producer
    /// registers a domain — the registry health surface for `status fab`.
    pub fn fabric_domains(self) -> u32 {
        self.fabric_domains.get()
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
    const KNOWN_FLAGS: u32 = Self::AVAILABLE | Self::READ_ONLY | Self::FLUSH | Self::DISCARD;

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

    pub const LIFECYCLE_UNKNOWN: u8 = crate::svc::FsSystemStatusWire::LIFECYCLE_UNKNOWN as u8;
    pub const LIFECYCLE_UNDESIGNATED: u8 =
        crate::svc::FsSystemStatusWire::LIFECYCLE_UNDESIGNATED as u8;
    pub const LIFECYCLE_REFUSED: u8 =
        crate::svc::FsSystemStatusWire::LIFECYCLE_REFUSED as u8;
    pub const LIFECYCLE_REFORMATTING: u8 =
        crate::svc::FsSystemStatusWire::LIFECYCLE_REFORMATTING as u8;
    pub const LIFECYCLE_RESEEDING: u8 =
        crate::svc::FsSystemStatusWire::LIFECYCLE_RESEEDING as u8;
    pub const LIFECYCLE_UNAVAILABLE: u8 =
        crate::svc::FsSystemStatusWire::LIFECYCLE_UNAVAILABLE as u8;
    pub const LIFECYCLE_REBUILT: u8 =
        crate::svc::FsSystemStatusWire::LIFECYCLE_REBUILT as u8;
    pub const LIFECYCLE_CURRENT: u8 =
        crate::svc::FsSystemStatusWire::LIFECYCLE_CURRENT as u8;

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
    };

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
    ) -> Self {
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
        }
    }

    pub fn available(self) -> bool {
        self.flags.get() == AVAILABLE_U16
    }
    pub fn validate(self) -> bool {
        self == Self::EMPTY
            || (self.available()
                && self.storage.validate()
                && self.durable() == self.storage.available())
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
    /// Completion waits that exhausted their full provider budget.
    pub fn expired_waits(self) -> u64 {
        self.expired_waits.get()
    }
    /// Expired waits where the device still reported itself healthy.
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
    pub const fn from_metrics(barrier: (u64, u64), field: (u64, u64), volatility: (u64, u64)) -> Self {
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
    assert!(core::mem::size_of::<SystemSnapshot>() == 88);
    assert!(core::mem::size_of::<MemorySnapshot>() == 64);
    assert!(core::mem::size_of::<FilesystemHealthSnapshot>() == 40);
    assert!(core::mem::size_of::<FilesystemStorageSnapshot>() == 80);
    assert!(core::mem::size_of::<FilesystemSnapshot>() == 216);
    // 16-byte link head (2 reserved) ‖ 24-byte counters ‖ 36-byte radio block.
    assert!(core::mem::size_of::<NetSnapshot>() == 76);
    assert!(core::mem::size_of::<ManifoldSnapshot>() == 40);
    assert!(core::mem::size_of::<TelemetrySnapshot>() == 224);
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
        assert_eq!(live.counters(), facts.counters, "the counters ride the link facts");
    }

    #[test]
    fn provider_snapshots_validate_only_defined_postures() {
        let system = SystemSnapshot::new(4, 1_000, 500, 16_384, 8_192, 4_096, 2_048, 1_024, 0, 0, 3);
        assert_eq!(system.fabric_domains(), 3, "the fabric domain count round-trips");
        let memory = MemorySnapshot::new(8, 3, 10, 7, 2, 0, 1, 12_288);
        let exec = ProcessSnapshot::new(32, 3, 1, 0, 9, 2, 6);

        assert!(TelemetrySnapshot::new(system, memory, exec, ManifoldSnapshot::EMPTY).validate());
        assert!(TelemetrySnapshot::exec_only(exec).validate());

        let mut corrupt_system = system;
        corrupt_system.flags = U32::new(2);
        assert!(!TelemetrySnapshot::new(corrupt_system, memory, exec, ManifoldSnapshot::EMPTY).validate());

        let mut corrupt_memory = memory;
        corrupt_memory.reserved = U32::new(1);
        assert!(!TelemetrySnapshot::new(system, corrupt_memory, exec, ManifoldSnapshot::EMPTY).validate());
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
