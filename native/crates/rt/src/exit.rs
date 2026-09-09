//! **Typed exit statuses** — the one vocabulary a kore program ends with.
//!
//! [`thread::exit`](crate::thread::exit) takes an [`ExitCode`], never a bare
//! `i32`.  The kernel's exit verb still carries a signed 32-bit word (the
//! frozen [`ExitReport`](kore_abi::session::ExitReport) field is untouched by
//! this module); what the type adds is a *name at the callsite* and a band
//! discipline the compiler can check, so `exit(123)` cannot silently mean one
//! thing in one program and something else in the next.
//!
//! ## The bands
//!
//! The tree already partitioned the status space; this module writes that
//! partition down and enforces it.
//!
//! | Range | Band | Who owns it |
//! |---|---|---|
//! | `0` | [`OK`](ExitCode::OK) | success, everywhere |
//! | `1..=124` | **program** | the program's own vocabulary — [`ExitCode::app`] |
//! | `125..=127` | **runtime** | the program never really started — kore-rt |
//! | `128..=255` | **abnormal** | terminated rather than returned — kore-rt |
//! | negative | **substrate** | the *foundation* failed — [`ExitCode::link`], [`service_fault`](ExitCode::service_fault), [`PANIC`](ExitCode::PANIC) |
//!
//! Two program-band codes are canonical because seven programs had already
//! converged on them independently: [`REFUSED`](ExitCode::REFUSED) (`1`, the
//! program ran and declined, having reported why) and [`IO`](ExitCode::IO)
//! (`2`, the result could not be delivered).  Everything else in `1..=124`
//! belongs to the program that declares it.
//!
//! The negative band is what the shell prints with a leading `-` and what
//! `res/USERLAND.md` §5.4a means by a launch that never reached the program's
//! own logic: a relocation resolved wrong, the linker stub refused, or the
//! panic handler committed the exit itself.
//!
//! ## Extending the set
//!
//! A program declares its own codes with [`exit_codes!`](crate::exit_codes) —
//! one block, checked at compile time:
//!
//! ```
//! kore_rt::exit_codes! {
//!     /// The clock was unreachable.
//!     const CLOCK = 3;
//!     /// The deadline park did not resolve.
//!     const PARK = 4;
//! }
//! ```
//!
//! The macro rejects a value outside the program band and rejects two codes
//! in the same block sharing a value — the two ways a hand-numbered `const
//! EXIT_*: i32` list rots.  Both failures are `const` evaluation errors, so
//! they cost nothing at runtime and cannot reach a boot.

use crate::entry::RUNG_PANIC;

/// A thread's terminal status, as [`thread::exit`](crate::thread::exit) takes
/// it and a joiner reads it back.
///
/// `Default` is [`OK`](Self::OK): a program that finished has nothing to say.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct ExitCode(i32);

impl ExitCode {
    /// First status in the typed resident-service fault band. The low eight
    /// bits above this base carry `COOKIE_REFUSED`'s runtime rung; keeping it
    /// away from both
    /// [`link`](Self::link)'s near-zero negatives and [`PANIC`](Self::PANIC)
    /// lets a supervisor distinguish all three without guessing.
    const SERVICE_FAULT_BASE: i32 = i32::MIN + 0x100;

    /// Lossy sentinel for a link rung whose direct negative encoding is not
    /// available. It sits below the typed service-fault band, so every value
    /// [`link`](Self::link) accepts remains distinguishable from a resident
    /// service fault even when the original rung cannot be retained exactly.
    const LINK_SATURATED: i32 = -i32::MAX;

    /// The highest status a program may claim for its own vocabulary.
    /// Above this the runtime band begins.
    pub const PROGRAM_MAX: u8 = 124;

    /// **The work completed.**
    pub const OK: Self = Self(0);

    /// **The program ran and declined**, having already reported the reason
    /// on its output.  Not an error in the program — an answer of "no".
    pub const REFUSED: Self = Self(1);

    /// **The result could not be delivered** — the output hand-back, the
    /// terminal write, or the display path failed after the work was done.
    pub const IO: Self = Self(2);

    /// **No process I/O page.**  The founder did not delegate the
    /// [`process`](crate::process) window, so the program has no lawful way
    /// to answer at all.  Nothing of the program's own logic ran.
    pub const NO_PROCESS_IO: Self = Self(125);

    /// **The argument block was unusable** — unreadable, or longer than the
    /// bounded command line the program reserves.
    pub const BAD_ARGS: Self = Self(126);

    /// **Startup did not complete** — a resource the program needs before it
    /// can begin (an interpreter VM, a service attachment) was refused.
    pub const STARTUP: Self = Self(127);

    /// **Terminated, not returned** — a hosted C runtime reached `abort()`.
    /// `128 + SIGABRT`, the value a POSIX shell would report.
    pub const ABORTED: Self = Self(134);

    /// **The panic handler committed this exit.**  An exec-founded task whose
    /// panic handler can still present its own `role::SELF` authority ends
    /// here, so a supervisor reads a typed panic rather than guessing at an
    /// application code ([`entry::PANIC_EXIT_STATUS`](crate::entry::PANIC_EXIT_STATUS)).
    pub const PANIC: Self = Self(i32::MIN + RUNG_PANIC as i32);

    /// **A resident service refused at `payload`'s rung.**
    ///
    /// `COOKIE_REFUSED` carries `[7:0]` rung and `[63:8]` detail, while the
    /// task-exit ABI has one signed 32-bit word. The stable attribution needed
    /// after the readiness channel is gone is the low-byte rung, so this typed
    /// band retains exactly that value. Detail remains best-effort on the
    /// startup notice; it must never be mistaken for part of the rung.
    #[inline]
    #[must_use]
    pub const fn service_fault(payload: u64) -> Self {
        let retained = (payload & u8::MAX as u64) as u8;
        Self(Self::SERVICE_FAULT_BASE + retained as i32)
    }

    /// Decode a status produced by [`service_fault`](Self::service_fault).
    /// Other negative status vocabularies return `None`.
    #[inline]
    #[must_use]
    pub const fn service_fault_rung(status: i32) -> Option<u16> {
        if status >= Self::SERVICE_FAULT_BASE && status <= Self::SERVICE_FAULT_BASE + u8::MAX as i32
        {
            Some((status - Self::SERVICE_FAULT_BASE) as u16)
        } else {
            None
        }
    }

    /// **Declare a program-band code** (`1..=`[`PROGRAM_MAX`](Self::PROGRAM_MAX)).
    ///
    /// `const`-evaluated at every intended callsite, so a code outside the
    /// band is a compile error rather than a status the runtime band will
    /// later be confused by.  Prefer [`exit_codes!`](crate::exit_codes),
    /// which calls this and additionally proves the block's codes distinct.
    ///
    /// # Panics
    ///
    /// If `code` is `0` or above [`PROGRAM_MAX`](Self::PROGRAM_MAX).
    #[inline]
    #[must_use]
    pub const fn app(code: u8) -> Self {
        assert!(
            code != 0 && code <= Self::PROGRAM_MAX,
            "a program exit code must lie in 1..=ExitCode::PROGRAM_MAX; \
             0 is OK and the band above is the runtime's"
        );
        Self(code as i32)
    }

    /// **The substrate failed**, naming the rung that broke: the negative
    /// band, as the linker stub and the dynamic-link witness already used it.
    ///
    /// `rung` is the same refusal ordinal the founder-NACK path posts, so a
    /// foreground joiner and a boot-founded parent read the same number with
    /// opposite signs.  `0` would collide with [`OK`](Self::OK) and is
    /// refused; a rung beyond `i32::MAX` saturates. Typed negative runtime
    /// statuses are reserved too: a rung whose direct negation would equal
    /// [`PANIC`](Self::PANIC) or enter the resident-service fault band uses the
    /// same lossy saturation sentinel. Ordinary link rungs retain their
    /// historical `-rung` encoding.
    ///
    /// # Panics
    ///
    /// If `rung` is `0`.
    #[inline]
    #[must_use]
    pub const fn link(rung: u32) -> Self {
        assert!(rung != 0, "a link-failure rung must be non-zero; 0 is OK");
        let rung = if rung > i32::MAX as u32 {
            i32::MAX
        } else {
            rung as i32
        };
        let status = -rung;
        if status == Self::PANIC.get() || Self::service_fault_rung(status).is_some() {
            Self(Self::LINK_SATURATED)
        } else {
            Self(status)
        }
    }

    /// **Adopt a status produced elsewhere**: a joined child's, a hosted
    /// interpreter's, or a program's own composite encoding (`spmd-witness`
    /// reports stage×10 + fault).  No band is enforced — the caller owns the
    /// meaning of the number it computed.
    #[inline]
    #[must_use]
    pub const fn from_raw(status: i32) -> Self {
        Self(status)
    }

    /// The signed word the kernel's exit verb carries.
    #[inline]
    #[must_use]
    pub const fn get(self) -> i32 {
        self.0
    }

    /// Whether this is [`OK`](Self::OK).
    #[inline]
    #[must_use]
    pub const fn is_ok(self) -> bool {
        self.0 == 0
    }
}

impl From<ExitCode> for i32 {
    #[inline]
    fn from(code: ExitCode) -> Self {
        code.get()
    }
}

/// Prove a block's codes pairwise distinct, at compile time.
///
/// The `const fn` half of [`exit_codes!`](crate::exit_codes); not meant to be
/// called directly.  Quadratic over a list that is never longer than a
/// program's own vocabulary, and evaluated once per crate.
#[doc(hidden)]
pub const fn assert_distinct(codes: &[u8]) {
    let mut i = 0;
    while i < codes.len() {
        let mut j = i + 1;
        while j < codes.len() {
            assert!(
                codes[i] != codes[j],
                "two exit codes in the same `exit_codes!` block share a value"
            );
            j += 1;
        }
        i += 1;
    }
}

/// **Declare a program's own exit codes.**
///
/// Expands to one [`ExitCode`] constant per row, each built through
/// [`ExitCode::app`] — so a value outside the program band is a compile
/// error — plus a single `const` assertion that no two rows in the block
/// share a value.
///
/// ```
/// kore_rt::exit_codes! {
///     /// The entropy source was empty.
///     pub const NO_ENTROPY = 1;
///     /// The clock was unreachable.
///     pub const CLOCK = 3;
/// }
/// ```
///
/// Reach for the canonical [`ExitCode::OK`] / [`ExitCode::REFUSED`] /
/// [`ExitCode::IO`] before minting a program code that would mean the same
/// thing; the point of the shared three is that a reader of a *different*
/// program already knows them.
#[macro_export]
macro_rules! exit_codes {
    ($(
        $(#[$attr:meta])*
        $vis:vis const $name:ident = $value:expr;
    )+) => {
        $(
            $(#[$attr])*
            $vis const $name: $crate::exit::ExitCode = $crate::exit::ExitCode::app($value);
        )+
        const _: () = $crate::exit::assert_distinct(&[$($value),+]);
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_canonical_codes_keep_the_values_the_tree_converged_on() {
        // Seven programs had independently picked these before the type
        // existed; the consolidation is only lawful if it preserves them.
        assert_eq!(ExitCode::OK.get(), 0);
        assert_eq!(ExitCode::REFUSED.get(), 1);
        assert_eq!(ExitCode::IO.get(), 2);
        assert_eq!(ExitCode::NO_PROCESS_IO.get(), 125);
        assert_eq!(ExitCode::BAD_ARGS.get(), 126);
        assert_eq!(ExitCode::STARTUP.get(), 127);
        assert_eq!(ExitCode::ABORTED.get(), 134);
    }

    #[test]
    fn the_bands_do_not_overlap() {
        assert!(ExitCode::app(ExitCode::PROGRAM_MAX).get() < ExitCode::NO_PROCESS_IO.get());
        assert!(ExitCode::NO_PROCESS_IO.get() < ExitCode::ABORTED.get());
        assert!(ExitCode::PANIC.get() < 0);
        assert!(ExitCode::PANIC.get() < ExitCode::service_fault(0).get());
        assert!(ExitCode::link(1).get() < 0);
        assert_eq!(ExitCode::service_fault_rung(ExitCode::PANIC.get()), None);
        assert_eq!(ExitCode::service_fault_rung(ExitCode::link(1).get()), None);
    }

    #[test]
    fn service_fault_retains_the_refusal_rung_and_rejects_other_bands() {
        for rung in [0u16, 1, 0x60, 0x68, 0x7f, 0xff] {
            let status = ExitCode::service_fault(u64::from(rung)).get();
            assert_eq!(ExitCode::service_fault_rung(status), Some(rung));
        }
        assert_eq!(
            ExitCode::service_fault_rung(ExitCode::service_fault(0x12_3456).get()),
            Some(0x56),
            "COOKIE_REFUSED detail must not become part of the exit rung",
        );
        assert_eq!(
            ExitCode::service_fault_rung(ExitCode::service_fault(u64::MAX).get()),
            Some(0xff),
        );
        assert_eq!(ExitCode::service_fault_rung(ExitCode::OK.get()), None);
    }

    #[test]
    fn app_spans_the_program_band() {
        assert_eq!(ExitCode::app(1).get(), 1);
        assert_eq!(ExitCode::app(124).get(), 124);
    }

    #[test]
    fn link_negates_its_rung_and_saturates() {
        assert_eq!(ExitCode::link(3).get(), -3);
        assert_eq!(ExitCode::link(u32::MAX).get(), -i32::MAX);
    }

    #[test]
    fn every_would_be_service_fault_collision_saturates_outside_the_band() {
        for retained in 0..=u8::MAX {
            let service_status = ExitCode::service_fault(u64::from(retained)).get();
            let colliding_rung = service_status.unsigned_abs();
            let link_status = ExitCode::link(colliding_rung).get();
            assert_eq!(link_status, ExitCode::LINK_SATURATED);
            assert_eq!(
                ExitCode::service_fault_rung(link_status),
                None,
                "link rung {colliding_rung} decoded as service fault {retained}",
            );
        }
    }

    #[test]
    fn panic_status_cannot_be_forged_by_link_and_neighbors_keep_their_encoding() {
        let panic_rung = ExitCode::PANIC.get().unsigned_abs();
        assert_eq!(ExitCode::link(panic_rung).get(), ExitCode::LINK_SATURATED);
        assert_ne!(ExitCode::link(panic_rung).get(), ExitCode::PANIC.get());

        for rung in [panic_rung - 1, panic_rung + 1] {
            assert_eq!(ExitCode::link(rung).get(), -(rung as i32));
            assert_ne!(ExitCode::link(rung).get(), ExitCode::PANIC.get());
            assert_eq!(
                ExitCode::service_fault_rung(ExitCode::link(rung).get()),
                None,
            );
        }
    }

    #[test]
    fn link_preserves_values_immediately_outside_the_reserved_fault_band() {
        let fault_low = ExitCode::service_fault(0).get();
        let fault_high = ExitCode::service_fault(u64::from(u8::MAX)).get();
        let rung_before_band = fault_high.unsigned_abs() - 1;
        let rung_after_band = fault_low.unsigned_abs() + 1;

        assert_eq!(
            ExitCode::link(rung_before_band).get(),
            -(rung_before_band as i32),
        );
        assert_eq!(
            ExitCode::link(rung_after_band).get(),
            -(rung_after_band as i32),
        );
        assert_eq!(
            ExitCode::service_fault_rung(ExitCode::link(rung_before_band).get()),
            None,
        );
        assert_eq!(
            ExitCode::service_fault_rung(ExitCode::link(rung_after_band).get()),
            None,
        );
        for rung in [i32::MAX as u32, i32::MAX as u32 + 1, u32::MAX] {
            assert_eq!(ExitCode::link(rung).get(), ExitCode::LINK_SATURATED);
            assert_eq!(
                ExitCode::service_fault_rung(ExitCode::link(rung).get()),
                None,
            );
        }
    }

    #[test]
    fn a_default_code_is_success() {
        assert!(ExitCode::default().is_ok());
        assert!(!ExitCode::REFUSED.is_ok());
    }

    #[test]
    fn distinctness_accepts_a_well_formed_block() {
        assert_distinct(&[1, 2, 3, 90]);
        assert_distinct(&[]);
    }

    #[test]
    #[should_panic(expected = "share a value")]
    fn distinctness_rejects_a_repeated_code() {
        assert_distinct(&[1, 2, 1]);
    }

    #[test]
    #[should_panic(expected = "1..=ExitCode::PROGRAM_MAX")]
    fn app_rejects_the_runtime_band() {
        let _ = ExitCode::app(125);
    }

    #[test]
    #[should_panic(expected = "1..=ExitCode::PROGRAM_MAX")]
    fn app_rejects_zero() {
        let _ = ExitCode::app(0);
    }

    #[test]
    #[should_panic(expected = "non-zero")]
    fn link_rejects_a_zero_rung() {
        let _ = ExitCode::link(0);
    }
}
