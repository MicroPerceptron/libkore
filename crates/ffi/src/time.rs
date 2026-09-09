use super::KoreStatus;
use kore::time::TimerError;

/// Clock-domain errors preserve the native distinction; no zero-time fallback.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KoreClockError {
    None = 0,
    Unsupported = 1,
    Unconfigured = 2,
    Uncalibrated = 3,
    WrongCpu = 4,
    ControllerUnavailable = 5,
    OutOfRange = 6,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KoreClockReading {
    pub nanoseconds: u64,
    pub error: KoreClockError,
    pub reserved: u32,
}

unsafe fn reading(out: *mut KoreClockReading, result: Result<u64, TimerError>) -> KoreStatus {
    if out.is_null() {
        return KoreStatus::InvalidArgument;
    }
    let (nanoseconds, error, status) = match result {
        Ok(time) => (time, KoreClockError::None, KoreStatus::Ok),
        Err(error) => (
            0,
            match error {
                TimerError::Unsupported => KoreClockError::Unsupported,
                TimerError::Unconfigured => KoreClockError::Unconfigured,
                TimerError::Uncalibrated => KoreClockError::Uncalibrated,
                TimerError::WrongCpu => KoreClockError::WrongCpu,
                TimerError::ControllerUnavailable => KoreClockError::ControllerUnavailable,
                TimerError::OutOfRange => KoreClockError::OutOfRange,
            },
            KoreStatus::Refused,
        ),
    };
    // SAFETY: the exported function requires aligned writable output storage.
    unsafe {
        out.write(KoreClockReading {
            nanoseconds,
            error,
            reserved: 0,
        });
    }
    status
}

/// Read the native monotonic domain, in nanoseconds since boot.
/// # Safety
/// out is null or aligned writable storage for one KoreClockReading.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_time_uptime(out: *mut KoreClockReading) -> KoreStatus {
    unsafe { reading(out, kore::time::uptime().map(|time| time.as_nanos())) }
}

/// Read the native wall domain, in nanoseconds since the Unix epoch.
/// # Safety
/// out is null or aligned writable storage for one KoreClockReading.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_time_wall(out: *mut KoreClockReading) -> KoreStatus {
    unsafe { reading(out, kore::time::wall().map(|time| time.as_nanos())) }
}

/// Suspend this callback program for relative nanoseconds on the native executor.
/// Submission is asynchronous; return from the callback after Queued.
/// # Safety
/// ctx is this callback's exclusive live filesystem context.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_time_sleep_for(
    ctx: *mut crate::KoreFsContext,
    nanoseconds: u64,
) -> KoreStatus {
    unsafe { queue_time(ctx, crate::KORE_OP_SLEEP_FOR, nanoseconds) }
}
/// Suspend until absolute monotonic nanoseconds; never pass wall-clock time.
/// # Safety
/// ctx is this callback's exclusive live filesystem context.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_time_sleep_until(
    ctx: *mut crate::KoreFsContext,
    nanoseconds: u64,
) -> KoreStatus {
    unsafe { queue_time(ctx, crate::KORE_OP_SLEEP_UNTIL, nanoseconds) }
}
/// Cooperatively yield this callback program to the native executor.
/// # Safety
/// ctx is this callback's exclusive live filesystem context.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_time_yield(ctx: *mut crate::KoreFsContext) -> KoreStatus {
    unsafe { queue_time(ctx, crate::KORE_OP_YIELD, 0) }
}
unsafe fn queue_time(ctx: *mut crate::KoreFsContext, op: u32, ns: u64) -> KoreStatus {
    // SAFETY: exported calls require exclusive callback-local access.
    let ctx = match unsafe { crate::context(ctx) } {
        Ok(ctx) => ctx,
        Err(e) => return e,
    };
    let mut command = crate::Command::new(op);
    command.offset = ns;
    ctx.queue(command)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_clock_refusal_is_not_reported_as_zero_time_success() {
        let mut out = KoreClockReading {
            nanoseconds: 91,
            error: KoreClockError::None,
            reserved: 99,
        };
        // SAFETY: disjoint aligned local output storage.
        assert_eq!(
            unsafe { reading(&mut out, Err(TimerError::Uncalibrated)) },
            KoreStatus::Refused
        );
        assert_eq!(out.error, KoreClockError::Uncalibrated);
        assert_eq!(out.nanoseconds, 0);
        assert_eq!(out.reserved, 0);
    }
}
