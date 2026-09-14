//! The closed vocabulary for native Rust task exits.
//!
//! Native callers select a standard outcome or one of sixteen application
//! slots. There is no integer constructor, numeric conversion into this type,
//! or public extension trait. Application aliases name existing slots:
//!
//! ```
//! use kore_rt::ExitCode;
//! kore_rt::exit_codes! {
//!     /// The parser rejected the document.
//!     const BAD_DOCUMENT = ExitCode::APPLICATION_0;
//! }
//! assert_eq!(BAD_DOCUMENT.get(), 256);
//! ```
//!
//! Values not listed here remain reserved for future standard outcomes. The
//! application slots are nonzero signed 32-bit statuses, not POSIX low-byte
//! statuses: APPLICATION_0 (256) is a failure, never success.
//!
//! The wire ABI remains i32, including statuses from foreign programs. Reading
//! an exit report does not authorize constructing a new native ExitCode from
//! its integer. Detailed diagnostic payloads belong on the program's output
//! or the runtime's readiness protocol, separately from the shared outcome.
//!
//! ```compile_fail
//! let code = kore_rt::ExitCode::from_raw(256);
//! ```
//! ```compile_fail
//! let code = kore_rt::ExitCode::app(3);
//! ```
//! ```compile_fail
//! let code: kore_rt::ExitCode = 256.into();
//! ```
//! ```compile_fail
//! let code = kore_rt::ExitCode(256);
//! ```
//! ```compile_fail
//! kore_rt::exit_codes! { const AD_HOC = 256; }
//! ```
//! ```compile_fail
//! let code = kore_rt::ExitCode::link(3);
//! ```
//! ```compile_fail
//! let code = kore_rt::ExitCode::service_fault(3);
//! ```
//! ```compile_fail
//! kore_rt::thread::exit(256);
//! ```
//! ```compile_fail
//! use kore_rt::session::SessionBuilder;
//! use kore_abi::session::SlotRef;
//! SessionBuilder::new().exit(SlotRef::from_wire(1), 256).unwrap();
//! ```

use crate::entry::RUNG_PANIC;

/// A named native task outcome. Its representation is private so safe callers
/// cannot turn an integer into an exit code. Default means success.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct ExitCode(i32);

impl ExitCode {
    const SERVICE_FAULT_BASE: i32 = i32::MIN + 0x100;
    /// The work completed.
    pub const OK: Self = Self(0);

    /// The program ran and declined, having reported why.
    pub const REFUSED: Self = Self(1);

    /// Input or output failed.
    pub const IO: Self = Self(2);

    /// Input or a result did not satisfy its data contract.
    pub const INVALID_DATA: Self = Self(3);

    /// A required resource or provider was unavailable.
    pub const UNAVAILABLE: Self = Self(4);

    /// The command or arguments were invalid.
    pub const USAGE: Self = Self(5);

    /// The requested operation is not implemented.
    pub const UNSUPPORTED: Self = Self(6);

    /// The requested object does not exist.
    pub const NOT_FOUND: Self = Self(7);

    /// The caller lacks the required authority.
    pub const DENIED: Self = Self(8);

    /// The operation exhausted its resource budget.
    pub const EXHAUSTED: Self = Self(9);

    /// The operation did not complete by its deadline.
    pub const TIMED_OUT: Self = Self(10);

    /// The operation was cancelled.
    pub const CANCELLED: Self = Self(11);

    /// A peer violated the protocol.
    pub const PROTOCOL: Self = Self(12);

    /// An internal invariant failed.
    pub const INTERNAL: Self = Self(13);

    /// Startup did not supply process I/O.
    pub const NO_PROCESS_IO: Self = Self(125);

    /// The startup argument block was unusable.
    pub const BAD_ARGS: Self = Self(126);

    /// The runtime could not start the program.
    pub const STARTUP: Self = Self(127);

    /// The task deliberately aborted.
    pub const ABORTED: Self = Self(134);

    /// The dynamic link or its validation failed.
    pub const LINK_FAILED: Self = Self(-1);

    /// Thread-local initialization or isolation failed.
    pub const TLS_FAILED: Self = Self(-2);

    /// The Rust panic handler ended the task.
    pub const PANIC: Self = Self(i32::MIN + RUNG_PANIC as i32);

    /// Application-reserved slot 0 (256); its meaning belongs to the executable.
    pub const APPLICATION_0: Self = Self(256);

    /// Application-reserved slot 1 (257); its meaning belongs to the executable.
    pub const APPLICATION_1: Self = Self(257);

    /// Application-reserved slot 2 (258); its meaning belongs to the executable.
    pub const APPLICATION_2: Self = Self(258);

    /// Application-reserved slot 3 (259); its meaning belongs to the executable.
    pub const APPLICATION_3: Self = Self(259);

    /// Application-reserved slot 4 (260); its meaning belongs to the executable.
    pub const APPLICATION_4: Self = Self(260);

    /// Application-reserved slot 5 (261); its meaning belongs to the executable.
    pub const APPLICATION_5: Self = Self(261);

    /// Application-reserved slot 6 (262); its meaning belongs to the executable.
    pub const APPLICATION_6: Self = Self(262);

    /// Application-reserved slot 7 (263); its meaning belongs to the executable.
    pub const APPLICATION_7: Self = Self(263);

    /// Application-reserved slot 8 (264); its meaning belongs to the executable.
    pub const APPLICATION_8: Self = Self(264);

    /// Application-reserved slot 9 (265); its meaning belongs to the executable.
    pub const APPLICATION_9: Self = Self(265);

    /// Application-reserved slot 10 (266); its meaning belongs to the executable.
    pub const APPLICATION_10: Self = Self(266);

    /// Application-reserved slot 11 (267); its meaning belongs to the executable.
    pub const APPLICATION_11: Self = Self(267);

    /// Application-reserved slot 12 (268); its meaning belongs to the executable.
    pub const APPLICATION_12: Self = Self(268);

    /// Application-reserved slot 13 (269); its meaning belongs to the executable.
    pub const APPLICATION_13: Self = Self(269);

    /// Application-reserved slot 14 (270); its meaning belongs to the executable.
    pub const APPLICATION_14: Self = Self(270);

    /// Application-reserved slot 15 (271); its meaning belongs to the executable.
    pub const APPLICATION_15: Self = Self(271);

    /// **A resident service refused at `payload`'s rung.**
    ///
    /// `COOKIE_REFUSED` carries `[7:0]` rung and `[63:8]` detail, while the
    /// task-exit ABI has one signed 32-bit word. The stable attribution needed
    /// after the readiness channel is gone is the low-byte rung, so this typed
    /// band retains exactly that value. Detail remains best-effort on the
    /// startup notice; it must never be mistaken for part of the rung.
    #[inline]
    #[must_use]
    pub(crate) const fn service_fault(payload: u64) -> Self {
        let retained = (payload & u8::MAX as u64) as u8;
        Self(Self::SERVICE_FAULT_BASE + retained as i32)
    }

    /// Decode a status produced by the runtime's private service-fault encoder.
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

/// Check that application aliases in one declaration are distinct.
#[doc(hidden)]
pub const fn assert_distinct(codes: &[ExitCode]) {
    let mut i = 0;
    while i < codes.len() {
        let mut j = i + 1;
        while j < codes.len() {
            assert!(
                codes[i].get() != codes[j].get(),
                "two exit codes in the same `exit_codes!` block share a value"
            );
            j += 1;
        }
        i += 1;
    }
}

/// Declare names for existing standard outcomes or application-reserved slots.
/// Integer expressions are rejected by the type checker. Aliases in one block
/// must have distinct values.
#[macro_export]
macro_rules! exit_codes {
    ($(
        $(#[$attr:meta])*
        $vis:vis const $name:ident = $value:expr;
    )+) => {
        $(
            $(#[$attr])*
            $vis const $name: $crate::exit::ExitCode = $value;
        )+
        const _: () = $crate::exit::assert_distinct(&[$($name),+]);
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_and_reserved_outcomes_are_full_width_and_disjoint() {
        assert_eq!(ExitCode::OK.get(), 0);
        assert_eq!(ExitCode::NO_PROCESS_IO.get(), 125);
        assert_eq!(ExitCode::BAD_ARGS.get(), 126);
        assert_eq!(ExitCode::STARTUP.get(), 127);
        assert_eq!(ExitCode::ABORTED.get(), 134);
        assert!(ExitCode::default().is_ok());
        assert!(!ExitCode::APPLICATION_0.is_ok());
        assert_eq!(ExitCode::APPLICATION_0.get(), 256);
        assert_eq!(ExitCode::APPLICATION_1.get(), 257);
        assert_eq!(ExitCode::APPLICATION_15.get(), 271);
        assert_distinct(&[
            ExitCode::OK,
            ExitCode::REFUSED,
            ExitCode::IO,
            ExitCode::INVALID_DATA,
            ExitCode::UNAVAILABLE,
            ExitCode::USAGE,
            ExitCode::UNSUPPORTED,
            ExitCode::NOT_FOUND,
            ExitCode::DENIED,
            ExitCode::EXHAUSTED,
            ExitCode::TIMED_OUT,
            ExitCode::CANCELLED,
            ExitCode::PROTOCOL,
            ExitCode::INTERNAL,
            ExitCode::NO_PROCESS_IO,
            ExitCode::BAD_ARGS,
            ExitCode::STARTUP,
            ExitCode::ABORTED,
            ExitCode::LINK_FAILED,
            ExitCode::TLS_FAILED,
            ExitCode::PANIC,
            ExitCode::APPLICATION_0,
            ExitCode::APPLICATION_1,
            ExitCode::APPLICATION_2,
            ExitCode::APPLICATION_3,
            ExitCode::APPLICATION_4,
            ExitCode::APPLICATION_5,
            ExitCode::APPLICATION_6,
            ExitCode::APPLICATION_7,
            ExitCode::APPLICATION_8,
            ExitCode::APPLICATION_9,
            ExitCode::APPLICATION_10,
            ExitCode::APPLICATION_11,
            ExitCode::APPLICATION_12,
            ExitCode::APPLICATION_13,
            ExitCode::APPLICATION_14,
            ExitCode::APPLICATION_15,
        ]);
    }

    #[test]
    fn runtime_service_fault_keeps_only_the_readiness_rung() {
        for rung in 0..=u8::MAX {
            let code = ExitCode::service_fault(0x12_3400 | u64::from(rung));
            assert_eq!(
                ExitCode::service_fault_rung(code.get()),
                Some(u16::from(rung))
            );
            assert!(!code.is_ok());
        }
        for code in [
            ExitCode::PANIC,
            ExitCode::LINK_FAILED,
            ExitCode::TLS_FAILED,
            ExitCode::OK,
            ExitCode::APPLICATION_0,
        ] {
            assert_eq!(ExitCode::service_fault_rung(code.get()), None);
        }
    }

    #[test]
    #[should_panic(expected = "share a value")]
    fn duplicate_aliases_are_rejected() {
        assert_distinct(&[ExitCode::APPLICATION_0, ExitCode::APPLICATION_0]);
    }
}
