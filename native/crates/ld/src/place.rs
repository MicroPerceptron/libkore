//! Placement — **the world's half of a link**, factored out so it can be
//! tested.
//!
//! [`LinkWorld::choose_base`] is where a real linker's `mmap(NULL, …)`
//! would be: the engine states a size and an alignment, and the world
//! answers *where*. Its doc is explicit that this is policy, not
//! mechanism — "per-task ASLR lives there, the engine only states
//! size/alignment" — and that the engine "re-checks the alignment and
//! otherwise trusts nothing else about the value".
//!
//! That makes placement the one part of the metal adapter with no
//! capability in it: arithmetic over a window, answering an address. The
//! rest of the adapter is verb calls whose behaviour is only observable
//! on metal (`cap::verbs` answers a typed refusal off the userland
//! target, by the 2026-07-24 ruling that deleted the mock cap layer), so
//! this is the piece a host suite can actually hold to account — and the
//! piece whose bugs are silent rather than loud: an overlapping answer
//! does not refuse, it corrupts whichever mapping was there first.
//!
//! Lives in `kore-ld` for the same reason [`crate::stub`] does: `user/`
//! is a separate workspace `make test` never reaches, so logic left in
//! the stub binary would be the part nothing exercises.
//!
//! [`LinkWorld::choose_base`]: crate::link::LinkWorld::choose_base

/// A bump placer over one contiguous window of a task's address space.
///
/// Hands out non-overlapping, alignment-satisfying spans until the
/// window is exhausted, then refuses. **Monotone and never reused**: a
/// linker maps each module once and nothing here is freed, so the
/// simplest allocator that cannot alias is the correct one — a free list
/// would add a way to hand out a live address in exchange for a
/// generality the caller does not have.
///
/// Placement is deliberately *not* randomized yet. ASLR belongs here and
/// the trait's doc says so, but entropy the stub cannot obtain honestly
/// would be worse than none: it runs before any runtime, so it has no
/// clock and no entropy source, and a "random" base derived from a
/// predictable seed reads as a defence while providing none. The seam is
/// where it goes when the child can be handed real entropy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placer {
    /// Next free address in the window.
    next: u64,
    /// One past the window's last byte.
    end: u64,
}

/// Why a placement refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaceError {
    /// `align` was not a power of two (`0` included) — a caller bug, and
    /// the one input that would otherwise silently produce a misaligned
    /// answer the engine then rejects with a less specific error.
    BadAlign,
    /// A zero-length span. Refused rather than answered, because the
    /// address would be indistinguishable from the next real placement's
    /// and two modules would believe they own it.
    ZeroLen,
    /// The window has no room for `len` at `align`.
    Exhausted,
}

impl Placer {
    /// A placer over `[base, base + len)`.
    pub const fn new(base: u64, len: u64) -> Self {
        Self {
            next: base,
            end: base.saturating_add(len),
        }
    }

    /// Choose a span of `len` bytes at an address that is a multiple of
    /// `align`, and consume it.
    ///
    /// Every arithmetic step is checked: this runs in a stub with no
    /// runtime, where a wrap would answer a *low* address that is very
    /// likely mapped by something else, and the caller has no way to tell
    /// that from a legitimate answer.
    pub fn choose(&mut self, len: u64, align: u64) -> Result<u64, PlaceError> {
        let (at, after) = bump(self.next, self.end, len, align)?;
        self.next = after;
        Ok(at)
    }

    /// Bytes still placeable, ignoring alignment loss — diagnostics, not
    /// a promise: a subsequent [`choose`](Self::choose) may still refuse
    /// after rounding up.
    pub const fn remaining(&self) -> u64 {
        self.end.saturating_sub(self.next)
    }

    /// One past the highest address handed out so far — the placement
    /// high-water mark a world reports through
    /// `LinkWorld::placed_high_water`, so a later placer over the same
    /// window can start above everything this one placed.
    pub const fn cursor(&self) -> u64 {
        self.next
    }
}

/// The one checked bump both placers use: round `next` up to `align`,
/// fit `len` below `end`, and answer `(at, after)`.
///
/// Factored out so the [`Placer`] and [`SharedPlacer`] cannot drift: the
/// failure mode of a second copy is not a compile error but an
/// *overlapping address*, which this module's own doc calls out as
/// silent rather than loud.
const fn bump(next: u64, end: u64, len: u64, align: u64) -> Result<(u64, u64), PlaceError> {
    if !align.is_power_of_two() {
        return Err(PlaceError::BadAlign);
    }
    if len == 0 {
        return Err(PlaceError::ZeroLen);
    }
    // Round up to `align`, without wrapping.
    let Some(bumped) = next.checked_add(align - 1) else {
        return Err(PlaceError::Exhausted);
    };
    let at = bumped & !(align - 1);
    let Some(after) = at.checked_add(len) else {
        return Err(PlaceError::Exhausted);
    };
    if after > end {
        return Err(PlaceError::Exhausted);
    }
    Ok((at, after))
}

/// The `next` value meaning "no window published yet". Not a plausible
/// cursor: a window ending at `u64::MAX` is already refused by
/// [`Placer::new`]'s saturation, so this cannot collide with a live base.
const UNARMED: u64 = u64::MAX;

/// A [`Placer`] several consumers share, without a lock.
///
/// The owned `Placer` is `&mut` by design: one linker, one window, one
/// caller. A task that *also* hands its link window to a second consumer
/// — a wasm engine placing guest memory beside the late loader's modules
/// — cannot use two `Placer` values over one window, because each would
/// hand out the other's addresses. Two cursors over one window is the
/// bug this type exists to make unrepresentable.
///
/// Monotone and lock-free, mimicking `kore_rt::heap`'s supply cursors: a
/// checked `try_update` means a losing racer retries against the winner's
/// mark rather than serialising behind it, and arithmetic that would wrap
/// refuses instead of answering a low, probably-mapped address.
/// Nothing is ever freed, so a compare-and-set that only advances is the
/// whole of the concurrency story.
pub struct SharedPlacer {
    next: core::sync::atomic::AtomicU64,
    /// One past the window's last byte, or `0` for *unarmed* — a placer
    /// whose window is not known yet and which therefore refuses every
    /// request. Fail-closed is the only safe default for a `static`: a
    /// consumer that places before the window is published must get a
    /// typed refusal, never address zero.
    end: core::sync::atomic::AtomicU64,
}

impl SharedPlacer {
    /// A shared placer over `[base, base + len)`.
    pub const fn new(base: u64, len: u64) -> Self {
        Self {
            next: core::sync::atomic::AtomicU64::new(base),
            end: core::sync::atomic::AtomicU64::new(base.saturating_add(len)),
        }
    }

    /// An unarmed placer, for a `static` whose window is only known once
    /// the task can read its own geometry. Refuses until [`arm`](Self::arm).
    pub const fn unarmed() -> Self {
        Self {
            next: core::sync::atomic::AtomicU64::new(UNARMED),
            end: core::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Publish the window, once. A second call refuses and leaves the
    /// first window **and its cursor** intact, so a second consumer cannot
    /// silently re-base a cursor that earlier placements were already
    /// derived from — every one of those addresses would become reusable.
    ///
    /// The claim is the compare-and-set on `next` away from [`UNARMED`],
    /// so a losing arm writes nothing at all. `end` is released only after
    /// that claim is won, and [`choose`](Self::choose) acquires `end`
    /// first: observing an armed `end` therefore guarantees observing the
    /// matching `next`. Publishing `next` first (the obvious order) is
    /// wrong in both directions — a refused arm would clobber a live
    /// cursor, and a racing `choose` could read the new base against the
    /// old window.
    pub fn arm(&self, base: u64, len: u64) -> Result<(), PlaceError> {
        use core::sync::atomic::Ordering;
        let end = base.saturating_add(len);
        if end <= base {
            return Err(PlaceError::ZeroLen);
        }
        if base == UNARMED {
            return Err(PlaceError::Exhausted);
        }
        self.next
            .compare_exchange(UNARMED, base, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| PlaceError::Exhausted)?;
        self.end.store(end, Ordering::Release);
        Ok(())
    }

    /// Whether a window has been published.
    pub fn is_armed(&self) -> bool {
        self.end.load(core::sync::atomic::Ordering::Acquire) != 0
    }

    /// Choose and consume a span, as [`Placer::choose`], from any thread.
    ///
    /// The closure re-runs [`bump`] against whatever mark won the race, so
    /// a concurrent caller never receives an address derived from a stale
    /// cursor.
    pub fn choose(&self, len: u64, align: u64) -> Result<u64, PlaceError> {
        use core::sync::atomic::Ordering;
        let end = self.end.load(Ordering::Acquire);
        if end == 0 {
            return Err(PlaceError::Exhausted);
        }
        let mut refusal = PlaceError::Exhausted;
        self.next
            .try_update(Ordering::AcqRel, Ordering::Acquire, |next| {
                match bump(next, end, len, align) {
                    Ok((_, after)) => Some(after),
                    Err(error) => {
                        refusal = error;
                        None
                    }
                }
            })
            .map_err(|_| refusal)
            .and_then(|previous| bump(previous, end, len, align).map(|(at, _)| at))
    }

    /// Bytes still placeable, ignoring alignment loss — diagnostics only,
    /// and immediately stale under concurrency. An unarmed placer has none.
    pub fn remaining(&self) -> u64 {
        use core::sync::atomic::Ordering;
        self.end
            .load(Ordering::Acquire)
            .saturating_sub(self.next.load(Ordering::Acquire))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The property everything else rests on: successive placements never
    /// overlap, and each satisfies the alignment it asked for.
    #[test]
    fn placements_are_aligned_and_disjoint() {
        let mut p = Placer::new(0x1000_0000, 0x10_0000);
        let a = p.choose(0x1234, 0x1000).expect("fits");
        let b = p.choose(0x1, 0x10_000).expect("fits");
        let c = p.choose(0x2000, 0x1000).expect("fits");

        assert_eq!(a % 0x1000, 0);
        assert_eq!(b % 0x10_000, 0);
        assert_eq!(c % 0x1000, 0);
        assert!(a + 0x1234 <= b, "a and b are disjoint");
        assert!(b + 0x1 <= c, "b and c are disjoint");
    }

    /// A window hands out what it has and then refuses — it does not wrap
    /// to its base or answer past its end.
    #[test]
    fn exhaustion_refuses_rather_than_overrunning() {
        let mut p = Placer::new(0x2000_0000, 0x2000);
        assert!(p.choose(0x1000, 0x1000).is_ok());
        assert!(p.choose(0x1000, 0x1000).is_ok());
        assert_eq!(p.remaining(), 0);
        assert_eq!(p.choose(0x1, 0x1), Err(PlaceError::Exhausted));
    }

    /// Alignment loss is charged against the window, so a placement that
    /// only fits *unaligned* refuses instead of straddling the end.
    #[test]
    fn alignment_padding_cannot_push_a_span_past_the_end() {
        let mut p = Placer::new(0x3000_0000, 0x3000);
        assert!(p.choose(0x1, 0x1000).is_ok(), "consumes into page 0");
        // 0x2000 left, but a 2 MiB alignment lands past the window.
        assert_eq!(
            p.choose(0x1000, 0x20_0000),
            Err(PlaceError::Exhausted),
            "rounding up must not answer outside the window"
        );
    }

    /// A non-power-of-two alignment is a caller bug, named as one.  The
    /// engine re-checks alignment, so answering something plausible here
    /// would surface as a vaguer error further away.
    #[test]
    fn a_bad_alignment_is_refused_by_name() {
        let mut p = Placer::new(0x4000_0000, 0x1_0000);
        assert_eq!(p.choose(0x1000, 3), Err(PlaceError::BadAlign));
        assert_eq!(p.choose(0x1000, 0), Err(PlaceError::BadAlign));
        // ...and refusing consumed nothing.
        assert_eq!(p.remaining(), 0x1_0000);
    }

    /// **A zero-length span refuses.**  Answered, it would hand two
    /// modules the same address — the next placement starts exactly where
    /// the empty one did — and both would believe they own it.
    #[test]
    fn a_zero_length_span_refuses_rather_than_aliasing() {
        let mut p = Placer::new(0x5000_0000, 0x1_0000);
        assert_eq!(p.choose(0, 0x1000), Err(PlaceError::ZeroLen));
        let a = p.choose(0x1000, 0x1000).expect("fits");
        assert_eq!(a, 0x5000_0000, "the refusal consumed nothing");
    }

    /// A window at the top of the address space must not wrap into a low
    /// address — the failure mode with no symptom, since a low answer is
    /// very likely mapped by something else and writing there corrupts
    /// it.
    #[test]
    fn a_window_at_the_address_ceiling_cannot_wrap() {
        let mut p = Placer::new(u64::MAX - 0x1000, 0x1000);
        assert_eq!(
            p.choose(0x2000, 0x1000),
            Err(PlaceError::Exhausted),
            "too large for the window"
        );
        // An alignment round-up that would overflow is equally refused.
        let mut q = Placer::new(u64::MAX - 0x10, 0x10);
        assert_eq!(q.choose(0x8, 0x1000), Err(PlaceError::Exhausted));
    }

    /// `Placer::new` saturates rather than wrapping when base + len would
    /// overflow, so a malformed window yields an empty placer instead of
    /// one spanning the whole address space.
    #[test]
    fn an_overflowing_window_is_empty_not_unbounded() {
        let mut p = Placer::new(u64::MAX - 0x10, u64::MAX);
        assert_eq!(p.remaining(), 0x10);
        assert_eq!(p.choose(0x100, 0x1), Err(PlaceError::Exhausted));
    }
    /// The drift guard the shared placer exists for: for one request
    /// sequence it must answer exactly what the owned placer answers.
    /// If these ever diverge, two consumers of one window overlap, and
    /// this module's doc says that failure is silent rather than loud.
    #[test]
    fn shared_and_owned_placers_agree() {
        let requests = [(0x1234, 0x1000), (0x1, 0x1_0000), (0x2000, 0x1000), (0x8, 0x8)];
        let mut owned = Placer::new(0x1000_0000, 0x10_0000);
        let shared = SharedPlacer::new(0x1000_0000, 0x10_0000);
        for (len, align) in requests {
            assert_eq!(owned.choose(len, align), shared.choose(len, align));
        }
        assert_eq!(owned.remaining(), shared.remaining());
    }

    /// Refusals agree too — a shared placer must not turn a caller bug
    /// into a different (or absent) error than the owned one.
    #[test]
    fn shared_placer_refuses_identically() {
        let mut owned = Placer::new(0x2000_0000, 0x2000);
        let shared = SharedPlacer::new(0x2000_0000, 0x2000);
        for (len, align) in [(0x1000u64, 0x3u64), (0, 0x1000), (0x4000, 0x1000)] {
            assert_eq!(owned.choose(len, align), shared.choose(len, align));
        }
        // A refused choose consumes nothing, on either.
        assert_eq!(owned.remaining(), 0x2000);
        assert_eq!(shared.remaining(), 0x2000);
    }

    /// The property that makes it shareable: concurrent callers receive
    /// disjoint, aligned spans and none is handed out twice.
    #[test]
    fn concurrent_placements_never_overlap() {
        use std::vec::Vec;

        const THREADS: usize = 8;
        const EACH: usize = 64;
        const LEN: u64 = 0x1000;

        let placer = std::sync::Arc::new(SharedPlacer::new(0x4000_0000, 0x100_0000));
        let mut handles = Vec::new();
        for _ in 0..THREADS {
            let placer = std::sync::Arc::clone(&placer);
            handles.push(std::thread::spawn(move || {
                (0..EACH)
                    .map(|_| placer.choose(LEN, LEN).expect("window is large enough"))
                    .collect::<Vec<u64>>()
            }));
        }
        let mut spans: Vec<u64> = handles
            .into_iter()
            .flat_map(|h| h.join().expect("no placer thread panics"))
            .collect();

        assert_eq!(spans.len(), THREADS * EACH);
        spans.sort_unstable();
        spans.dedup();
        assert_eq!(spans.len(), THREADS * EACH, "an address was handed out twice");
        for pair in spans.windows(2) {
            assert!(pair[0] + LEN <= pair[1], "spans overlap");
        }
        assert!(spans.iter().all(|at| at % LEN == 0), "a span was misaligned");
    }
    /// Fail-closed: a `static` placer refuses until its window is known,
    /// rather than answering an address in an unpublished window.
    #[test]
    fn an_unarmed_placer_refuses_everything() {
        let p = SharedPlacer::unarmed();
        assert!(!p.is_armed());
        assert_eq!(p.remaining(), 0);
        assert_eq!(p.choose(0x1000, 0x1000), Err(PlaceError::Exhausted));
    }

    /// Arming publishes the window exactly once. A second arm refuses and
    /// leaves the first window — otherwise a late consumer could re-base a
    /// cursor that earlier placements were already derived from, and every
    /// one of those addresses would silently become reusable.
    #[test]
    fn arming_is_once_and_the_first_window_wins() {
        let p = SharedPlacer::unarmed();
        assert_eq!(p.arm(0x5000_0000, 0x10_0000), Ok(()));
        assert!(p.is_armed());
        let first = p.choose(0x1000, 0x1000).expect("fits");

        assert_eq!(p.arm(0x9000_0000, 0x10_0000), Err(PlaceError::Exhausted));
        let second = p.choose(0x1000, 0x1000).expect("fits");
        assert!(
            (0x5000_0000..0x5010_0000).contains(&second),
            "a refused re-arm must not move the cursor into the new window"
        );
        assert!(second >= first + 0x1000, "the cursor kept advancing");
    }

    /// A degenerate window is refused at arm time rather than producing a
    /// placer that answers its own base.
    #[test]
    fn arming_an_empty_window_refuses() {
        let p = SharedPlacer::unarmed();
        assert_eq!(p.arm(0x5000_0000, 0), Err(PlaceError::ZeroLen));
        assert!(!p.is_armed());
    }
}
