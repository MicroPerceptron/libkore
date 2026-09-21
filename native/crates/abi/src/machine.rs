//! **The machine-facts block** — the sixth frozen boot-page block, after
//! the founding record (`kore-rt::entry::MACHINE_BLOCK_AT`).
//!
//! What the runtime needs to size a launch to the machine without an IPC
//! round trip: how many logical CPUs it has, and how many cores of the
//! founding node's best performance class a fitting SPMD region would be
//! granted exclusively (the G6c reservation population — `res/SPMD.md`
//! §7).  A dev otherwise hardcodes a width literal and guesses; with
//! this, [`kore_rt::spmd::dispatch`] reads the machine and picks.
//!
//! **Kernel testimony, like the founding block and unlike the clock and
//! geometry blocks.**  The clock and geometry blocks are relayed by the
//! exec service into a child's page; this one, like the founding record,
//! is written by the KERNEL at the founding site — because it is a fact
//! the runtime must trust and, for `best_lane`, one that depends on the
//! *founding node* (`current_cpu().numa()`), which only the per-spawn
//! kernel path knows.  A userland relayer could neither be trusted to
//! author it nor know the founding node.
//!
//! No seqlock (the founding block's odd/even `seq`): these facts are
//! immutable after CPU bringup, so the kernel writes the block with a
//! plain copy, the same discipline the immutable clock calibration uses.
//! A zeroed region reads `abi == 0` and exposes nothing — every pre-block
//! founder and every unmapped page fails closed.

use zerocopy::byteorder::little_endian::{U16, U32};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// Layout version.  A zeroed block reads `abi == 0` and exposes nothing
/// (the pre-block pages fail closed, like the geometry, clock, and
/// founding blocks).
pub const MACHINE_ABI: u16 = 1;

/// The machine-facts block — kernel-stamped per spawn at the founding
/// site.  20 bytes, frozen ABI (`#81`): additive-only evolution claims
/// the reserved words, never a reorder or resize.
#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct MachineBlock {
    /// Layout version ([`MACHINE_ABI`]); `0` = no record (fail closed).
    pub abi: U16,
    /// Machine capability flags — see [`MACHINE_FLAG_CPU_ENTROPY`].  An
    /// additive claim on what was reserved space: an unset bit reads the
    /// same as the zero a pre-flags founder wrote, so every flag must be
    /// phrased so that "absent" is the safe answer.
    pub flags: U16,
    /// Logical CPU count — `ResourceCatalog::cpu_count()` at the stamp
    /// site.  Machine-global (every founding sees the same value).
    pub logical_cpus: U32,
    /// The **best-lane population**: how many cores of the founding
    /// node's best performance class an SPMD region would be granted
    /// exclusively (the G6c batch reservation count — P-cores on a
    /// hybrid box, all cores on a homogeneous one).  Node-relative, so
    /// stamped per spawn: two foundings on different nodes of a
    /// heterogeneous machine legitimately see different values.  `0`
    /// only on an empty catalog.
    pub best_lane: U32,
    /// The founding NUMA node id (`current_cpu().numa()`) — the node
    /// [`best_lane`](Self::best_lane) counts.  Context for a reader that
    /// wants to know which node's class it was told about.
    pub node: U32,
    /// The **core-class census** of the whole machine, packed: `[7:0]`
    /// performance cores (Intel P / AMD Zen / Arm big), `[15:8]` efficiency
    /// cores (Intel E / AMD Zen-c / Arm LITTLE), `[23:16]` standard cores
    /// (a homogeneous part reports all of them here), `[31:24]` the LP-E
    /// SUBSET of the efficiency count. Old readers still count every CPU;
    /// new readers subtract that subset to display E and LP-E separately.  Machine-global like `logical_cpus`, and the answer to the
    /// question `res/SPMD.md` left open — "is the hybrid classification
    /// trustworthy?" — readable on a serial-less box through `status cpu`
    /// and eyeballed against the spec sheet.  Claimed additively from the
    /// former reserved word: a pre-census founder wrote `0`, which reads
    /// as "no census" (every count zero), the safe answer.
    pub class_census: U32,
}

impl MachineBlock {
    /// The empty block (`abi` stamped, no facts) — the value a founder
    /// that predates this block leaves, indistinguishable from a zeroed
    /// region only in that both `validate()` paths agree it exposes
    /// nothing useful (a zeroed region has `abi == 0`).
    pub const EMPTY: Self = Self {
        abi: U16::from_bytes(MACHINE_ABI.to_le_bytes()),
        flags: U16::ZERO,
        logical_cpus: U32::ZERO,
        best_lane: U32::ZERO,
        node: U32::ZERO,
        class_census: U32::ZERO,
    };

    /// Build a stamped block from the founding-site facts.  `class_census`
    /// is the packed [`class_census`](Self::class_census) word —
    /// [`pack_class_census`] builds it from the three counts.
    pub const fn new(
        logical_cpus: u32,
        best_lane: u32,
        node: u32,
        flags: u16,
        class_census: u32,
    ) -> Self {
        Self {
            abi: U16::from_bytes(MACHINE_ABI.to_le_bytes()),
            flags: U16::from_bytes(flags.to_le_bytes()),
            logical_cpus: U32::from_bytes(logical_cpus.to_le_bytes()),
            best_lane: U32::from_bytes(best_lane.to_le_bytes()),
            node: U32::from_bytes(node.to_le_bytes()),
            class_census: U32::from_bytes(class_census.to_le_bytes()),
        }
    }

    /// Whether the block carries a record this reader understands.
    pub fn validate(&self) -> bool {
        self.abi.get() == MACHINE_ABI
    }

    /// The logical CPU count, or `None` on an unstamped block.
    pub fn logical_cpus(&self) -> Option<u32> {
        self.validate().then(|| self.logical_cpus.get())
    }

    /// The best-lane population (the founding node's best-class core
    /// count — what a fitting SPMD region gets exclusively), or `None`
    /// on an unstamped block.
    pub fn best_lane(&self) -> Option<u32> {
        self.validate().then(|| self.best_lane.get())
    }

    /// Whether this CPU has an entropy instruction a task may execute at
    /// user privilege (`RDRAND` / `RNDR` / a U-mode-readable Zkr `seed`).
    ///
    /// Fails **closed**: an unstamped block, or one from a founder that
    /// predates the flag, reads `false`.  That is the safe direction —
    /// `FEAT_RNG` is optional on Arm and executing `RNDR` without it is an
    /// undefined-instruction trap, so a task that guessed "present" and
    /// guessed wrong does not get a refusal, it gets killed.
    pub fn cpu_entropy(&self) -> bool {
        self.validate() && self.flags.get() & MACHINE_FLAG_CPU_ENTROPY != 0
    }

    /// Whether the kernel's placement coefficient `c_i` is being steered by a
    /// **hardware feedback** source (Intel HFI today) rather than the boot
    /// streaming measurement alone.  Fails closed like every flag: an
    /// unstamped or pre-flag block reads `false`.
    pub fn hardware_feedback(&self) -> bool {
        self.validate() && self.flags.get() & MACHINE_FLAG_HW_FEEDBACK != 0
    }

    /// The machine's core-class census `(performance, efficient, standard)`,
    /// or `None` on an unstamped block.  All three zero on a pre-census
    /// founder — "no census", not "no cores".
    pub fn class_census(&self) -> Option<(u8, u8, u8)> {
        self.validate().then(|| {
            let packed = self.class_census.get();
            (packed as u8, (packed >> 8) as u8, (packed >> 16) as u8)
        })
    }
    /// Low-power subset of the efficiency count; zero on pre-island founders.
    /// Clamp malformed input rather than underflowing a reader's E count.
    pub fn low_power_cores(&self) -> Option<u8> {
        self.class_census()
            .map(|(_, efficient, _)| ((self.class_census.get() >> 24) as u8).min(efficient))
    }
}

/// Extend the existing census additively. The legacy efficiency byte remains
/// inclusive, so old readers do not lose CPUs when the reserved byte is used.
pub const fn pack_class_census_with_island(p: usize, e: usize, standard: usize, lp: usize) -> u32 {
    pack_class_census(p, e.saturating_add(lp), standard)
        | ((if lp > 255 { 255 } else { lp }) as u32) << 24
}

/// Pack the three class counts into the [`MachineBlock::class_census`]
/// word (each saturating at 255 — a count that wide is a census of a machine
/// this block was never sized for, and the reader still sees "many").
pub const fn pack_class_census(performance: usize, efficient: usize, standard: usize) -> u32 {
    const fn byte(count: usize) -> u32 {
        if count > u8::MAX as usize {
            u8::MAX as u32
        } else {
            count as u32
        }
    }
    byte(performance) | byte(efficient) << 8 | byte(standard) << 16
}

/// [`MachineBlock::flags`]: this CPU has a user-executable entropy
/// instruction.  Set by the kernel, which is the only party that can read
/// the ID registers the answer lives in (`ID_AA64ISAR0_EL1` is EL1-only;
/// x86 `CPUID` is readable at ring 3 but the kernel already knows).
pub const MACHINE_FLAG_CPU_ENTROPY: u16 = 1 << 0;

/// [`MachineBlock::flags`]: a hardware feedback source (Intel HFI) is live
/// and steering the kernel's per-CPU capacity coefficient (#597 inc. 3).
/// Kernel testimony — only the kernel touches the feedback MSRs; a task has
/// no way to ask the silicon itself.  Absent = the boot measurement alone.
pub const MACHINE_FLAG_HW_FEEDBACK: u16 = 1 << 1;

const _: () = assert!(
    core::mem::size_of::<MachineBlock>() == 20,
    "the machine block is frozen ABI"
);

#[cfg(test)]
mod tests {
    use super::*;
    use zerocopy::FromBytes;

    #[test]
    fn island_census_preserves_legacy_totals() {
        let b = MachineBlock::new(16, 4, 0, 0, pack_class_census_with_island(4, 8, 0, 4));
        assert_eq!(b.class_census(), Some((4, 12, 0)));
        assert_eq!(b.low_power_cores(), Some(4));
        let old = MachineBlock::new(8, 4, 0, 0, pack_class_census(4, 4, 0));
        assert_eq!(old.low_power_cores(), Some(0));
    }

    #[test]
    fn an_empty_block_validates_but_exposes_zero_facts() {
        let b = MachineBlock::EMPTY;
        assert!(b.validate());
        assert_eq!(b.logical_cpus(), Some(0));
        assert_eq!(b.best_lane(), Some(0));
    }

    #[test]
    fn a_zeroed_block_fails_closed() {
        let b =
            MachineBlock::read_from_bytes(&[0u8; core::mem::size_of::<MachineBlock>()]).unwrap();
        assert!(!b.validate(), "abi 0 must expose nothing");
        assert_eq!(b.logical_cpus(), None);
        assert_eq!(b.best_lane(), None);
    }

    #[test]
    fn a_stamped_block_round_trips() {
        let b = MachineBlock::new(8, 4, 0, 0, 0);
        assert!(b.validate());
        assert_eq!(b.logical_cpus(), Some(8));
        assert_eq!(b.best_lane(), Some(4));
        assert_eq!(b.node.get(), 0);
        assert!(!b.hardware_feedback());
        assert_eq!(
            b.class_census(),
            Some((0, 0, 0)),
            "a pre-census founder: no census"
        );
    }

    /// The Lunar Lake dev host's shape: 4 P + 4 E, feedback live.
    #[test]
    fn the_census_and_feedback_flag_round_trip() {
        let census = pack_class_census(4, 4, 0);
        let b = MachineBlock::new(8, 4, 0, MACHINE_FLAG_HW_FEEDBACK, census);
        assert!(b.hardware_feedback());
        assert!(!b.cpu_entropy(), "flags are independent bits");
        assert_eq!(b.class_census(), Some((4, 4, 0)));
        // A homogeneous 16-thread box reports everything as standard.
        let b = MachineBlock::new(16, 16, 0, 0, pack_class_census(0, 0, 16));
        assert_eq!(b.class_census(), Some((0, 0, 16)));
        // Counts saturate rather than wrap into a neighbour's byte.
        assert_eq!(pack_class_census(300, 1, 2), 0xFF | 1 << 8 | 2 << 16);
        // An unstamped block exposes neither.
        let z = MachineBlock::read_from_bytes(&[0u8; 20]).unwrap();
        assert!(!z.hardware_feedback());
        assert_eq!(z.class_census(), None);
    }
}
