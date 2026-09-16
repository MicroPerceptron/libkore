//! Typed outcomes for a returning application entry point.
//!
//! Returning an integer is deliberately unsupported. Application-specific
//! names still refer to the reserved slots in the native [`ExitCode`]
//! vocabulary; implementing this trait cannot manufacture another code.

use crate::ExitCode;

/// Convert a normal return from `main` into the native exit vocabulary.
///
/// This conversion performs no teardown. The entry runtime owns the single
/// subsequent process-completion path, after the main frame has returned
/// normally and dropped its local values.
pub trait Termination {
    fn report(self) -> ExitCode;
}

impl Termination for () {
    fn report(self) -> ExitCode {
        ExitCode::OK
    }
}

impl Termination for ExitCode {
    fn report(self) -> ExitCode {
        self
    }
}

impl Termination for ! {
    fn report(self) -> ExitCode {
        self
    }
}

impl Termination for core::convert::Infallible {
    fn report(self) -> ExitCode {
        match self {}
    }
}

/// The error arm carries an explicit named outcome, rather than selecting an
/// integer from an error's formatting or silently flattening it to one byte.
impl<T: Termination> Termination for Result<T, ExitCode> {
    fn report(self) -> ExitCode {
        match self {
            Ok(value) => value.report(),
            Err(code) => code,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn return_conversion_preserves_native_outcomes() {
        assert_eq!(().report(), ExitCode::OK);
        for code in [
            ExitCode::OK,
            ExitCode::APPLICATION_0,
            ExitCode::APPLICATION_1,
            ExitCode::APPLICATION_15,
            ExitCode::LINK_FAILED,
            ExitCode::PANIC,
        ] {
            assert_eq!(code.report(), code);
            assert_eq!(Ok::<_, ExitCode>(code).report(), code);
            assert_eq!(Err::<(), _>(code).report(), code);
        }
    }
}
