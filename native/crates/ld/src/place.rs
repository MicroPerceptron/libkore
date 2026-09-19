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
        if !align.is_power_of_two() {
            return Err(PlaceError::BadAlign);
        }
        if len == 0 {
            return Err(PlaceError::ZeroLen);
        }
        // Round up to `align`, without wrapping.
        let Some(bumped) = self.next.checked_add(align - 1) else {
            return Err(PlaceError::Exhausted);
        };
        let at = bumped & !(align - 1);
        let Some(after) = at.checked_add(len) else {
            return Err(PlaceError::Exhausted);
        };
        if after > self.end {
            return Err(PlaceError::Exhausted);
        }
        self.next = after;
        Ok(at)
    }

    /// Bytes still placeable, ignoring alignment loss — diagnostics, not
    /// a promise: a subsequent [`choose`](Self::choose) may still refuse
    /// after rounding up.
    pub const fn remaining(&self) -> u64 {
        self.end.saturating_sub(self.next)
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
}
