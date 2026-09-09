//! **The bring-up pause that gives the CPU back.**
//!
//! Driver bring-up is a ladder of bounded handshakes: write a control bit,
//! then probe a status until the hardware agrees — `attempts` probes with a
//! caller-supplied pause between them (`kore_driver::wait::handshake`).
//! Every one of those pauses in this tree was `core::hint::spin_loop`,
//! which is exactly right for the handshake that answers in microseconds
//! and a *freeze* for the one that does not: a bare spin makes no kernel
//! entry at all, so for the whole budget nothing else on that CPU runs.
//!
//! # Why that is the boot, not merely rude
//!
//! During the founding the founder drives each row from its own loop
//! (`Sched::service_dispatch_await_park`, then `drive_serving_ready`), and
//! everything the boot still owes the machine runs *between* that loop's
//! yields into ring 3: the wake drain, the manifold service, the deferred
//! bottom halves — and the **boot splash's only pump**
//! (`Sched::service_kernel_duties` → `splash::pump_frame`). A row that
//! spins out one half-second handshake holds the CPU for that half second,
//! and the only sign of life a serial-less machine has stops for exactly as
//! long. A row that spins out several freezes the splash for seconds, and
//! the founder's own phase line — repainted from that same loop — freezes
//! with it, so the machine cannot even say which row is busy.
//!
//! The kernel side of this lesson is already in the tree: AML `Sleep` used
//! to lower to a raw stall and starved the founding drive for tens of
//! milliseconds at a time, until it was made to service the run-loop duties
//! in place (`platform::firmware::handler`'s sleep servicer). This is the
//! ring-3 half of the same rule, and in ring 3 there is a better answer than
//! performing the duties: **park**, and the founder resumes and performs
//! them itself.
//!
//! # What [`ceding`] does
//!
//! Each ordinary probe takes one monitored [`PROBE_PAUSE`], which degrades to
//! the old spin hint when this machine cannot monitor or read its clock.  A
//! long wait additionally parks briefly once every [`CEDE_STRIDE`] probes so
//! the founder can run. The caller's elapsed-time/probe budget remains the
//! authority that ends the wait; this type only chooses how to relax between
//! probes.
//!
//! Cost is bounded by construction: the widest handshake in the tree is
//! xHCI's `MAX_POLLS` (500 000 probes), which cedes ~122 times at
//! [`CEDE_NS`] apiece — about 122 ms of scheduler cedes at most. Healthy
//! handshakes ordinarily finish after a handful of monitored probes.
//!
//! Not to be confused with [`hint`](crate::hint), which sleeps the *core*
//! on a cache line and deliberately does **not** release it to the
//! scheduler. This module is the other side of that note: milliseconds and
//! up, park.

/// Monitored probes taken before a pace cedes — and the stride between cedes
/// thereafter.
///
/// Chosen so the two properties that matter are both satisfied by one
/// number. Low enough that a slow handshake yields often: even at the full
/// 4 us pause, 4096 probes is about 16 ms, near the splash's frame cadence.
/// High enough that scheduler parking remains exceptional and does not add a
/// millisecond to every healthy short handshake.
pub const CEDE_STRIDE: u32 = 4096;

/// Maximum hardware-monitored relaxation between adjacent register probes.
///
/// Kept equal to the driver host's probe pause.  Four microseconds is short
/// enough that a ready bit is observed promptly, while a non-answering MMIO
/// handshake stops burning an SMT sibling between its elapsed-time checks.
pub const PROBE_PAUSE: core::time::Duration = core::time::Duration::from_micros(4);

/// How far ahead a cede names its wake.
///
/// Short on purpose. The point is to be *descheduled*, not to sleep: the
/// value is the price of one cede, paid up to `attempts / CEDE_STRIDE`
/// times, and the founder needs only the gap — not a long one.
pub const CEDE_NS: u64 = 1_000_000;

/// A bring-up pause that yields the CPU periodically.
///
/// Held by the handshake it paces; [`ceding`] is the closure form every
/// call site actually wants.
#[derive(Clone, Copy, Debug)]
pub struct Pace {
    taken: u32,
    ceded: u32,
}

impl Pace {
    /// A fresh pace — the first cede stride ahead of it.
    pub const fn new() -> Self {
        Self { taken: 0, ceded: 0 }
    }

    /// Whether the `taken`-th probe of a pace is one that cedes.
    ///
    /// The stride as pure arithmetic, because the park itself cannot run
    /// off the metal: this is the half a host test can hold to account.
    pub const fn cedes_at(taken: u32) -> bool {
        taken != 0 && taken.is_multiple_of(CEDE_STRIDE)
    }

    /// Take one probe's pause.
    #[inline]
    pub fn probe(&mut self) {
        // Wrapping, not saturating: a pace that outlived `u32::MAX` probes
        // must keep its stride rather than latch at a value `cedes_at`
        // answers `false` for forever, which would silently restore the
        // bare spin on the longest wait of all.
        self.taken = self.taken.wrapping_add(1);
        if Self::cedes_at(self.taken) && cede() {
            self.ceded = self.ceded.saturating_add(1);
            return;
        }
        let _ = crate::hint::pause_until(PROBE_PAUSE);
    }

    /// How many probes this pace has paced (wrapping, as [`probe`] counts).
    ///
    /// [`probe`]: Self::probe
    pub const fn taken(&self) -> u32 {
        self.taken
    }

    /// How many of them actually parked. Zero on a build or a thread that
    /// cannot park, where every probe was the spin it always was.
    pub const fn ceded(&self) -> u32 {
        self.ceded
    }
}

impl Default for Pace {
    fn default() -> Self {
        Self::new()
    }
}

/// **The pause a bring-up should be handed** — a drop-in replacement for
/// `core::hint::spin_loop` wherever a handshake's `delay` is asked for.
///
/// ```ignore
/// let controller = Controller::prepare(regs, pool, kore_rt::pace::ceding())?;
/// ```
pub fn ceding() -> impl FnMut() {
    let mut pace = Pace::new();
    move || pace.probe()
}

/// Park briefly so the scheduler gets this CPU back.
///
/// `false` means this build or this thread cannot park at all — a host test,
/// or a thread whose transport refused — and the caller takes the ordinary
/// monitored probe pause instead. The caller's own attempt or elapsed-time
/// budget remains authoritative in either posture.
fn cede() -> bool {
    // The deadline must be strictly in the future, or this is not a park at
    // all: the sleep verb answers an already-past deadline by resolving at
    // once *without descheduling*, which would leave the founder exactly as
    // starved as the spin does and one trap poorer.
    //
    // The live counter sample, deliberately, and not `Executor::now` — that
    // anchor only refreshes on parks the executor itself owns, so a loop
    // parking through the verb directly would re-derive the same stale
    // `now` every pass and name a past deadline every time (the
    // back-off-into-hot-spin failure `executor::time` warns about).
    if let Some(now) = crate::clock::monotonic_nanos() {
        return crate::cap::verbs::sleep_until(now.saturating_add(CEDE_NS)).is_some();
    }
    // No calibrated user clock (a legacy target, or a thread founded before
    // the clock page): the trap-return stamp is the fallback the executor
    // uses too. It is millisecond-grained, so the next whole millisecond is
    // the nearest instant that is strictly future for every `now` inside
    // this one — the same reasoning, at the resolution actually available.
    let Some(word) = crate::cap::verbs::sleep_until(1) else {
        return false;
    };
    let ms = word >> 32;
    crate::cap::verbs::sleep_until(ms.saturating_add(1).saturating_mul(1_000_000)).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_probe_stride_is_taken_whole_before_the_first_cede() {
        for taken in 0..=CEDE_STRIDE {
            assert!(
                !Pace::cedes_at(taken) || taken == CEDE_STRIDE,
                "probe {taken} ceded inside the stride",
            );
        }
        assert!(Pace::cedes_at(CEDE_STRIDE), "the stride's last probe cedes");
    }

    #[test]
    fn cedes_keep_their_stride_after_the_first() {
        assert!(Pace::cedes_at(CEDE_STRIDE * 2));
        assert!(Pace::cedes_at(CEDE_STRIDE * 3));
        assert!(!Pace::cedes_at(CEDE_STRIDE * 2 - 1));
        assert!(!Pace::cedes_at(CEDE_STRIDE * 2 + 1));
    }

    #[test]
    fn the_widest_handshake_in_the_tree_cedes_a_bounded_number_of_times() {
        // xHCI's `MAX_POLLS`, the widest budget any bring-up in this tree
        // hands a pause. The cost of adopting this module is exactly this
        // count times `CEDE_NS`, and it has to stay small against the
        // second the spin it replaces already took.
        const WIDEST: u32 = 500_000;
        let cedes = u64::from(WIDEST / CEDE_STRIDE);
        assert_eq!(cedes, 122);
        assert!(
            cedes * CEDE_NS < 200_000_000,
            "the widest handshake must not pay more than ~200 ms in cedes",
        );
    }

    #[test]
    fn a_pace_that_cannot_park_still_paces_and_reports_no_cedes() {
        // Off the metal `cede` refuses. The monitored pause itself may still
        // be available on the host, but it never becomes a task park.
        let mut pace = Pace::new();
        for _ in 0..(CEDE_STRIDE * 2 + 3) {
            pace.probe();
        }
        assert_eq!(pace.taken(), CEDE_STRIDE * 2 + 3);
        assert_eq!(pace.ceded(), 0, "no host build can park");
    }

    #[test]
    fn the_counter_wraps_rather_than_latching_out_of_stride() {
        // A pace that saturated would answer `cedes_at` false forever after
        // `u32::MAX` probes, silently restoring the bare spin on the one
        // wait long enough to reach that count.
        let mut pace = Pace::new();
        pace.taken = u32::MAX;
        pace.probe();
        assert_eq!(pace.taken(), 0);
        pace.probe();
        assert_eq!(pace.taken(), 1);
    }
}
