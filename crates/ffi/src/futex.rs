use core::{sync::atomic::AtomicU32, time::Duration};
use kore::futex;

/// Pass as timeout_ns to wait indefinitely. All other values are relative ns.
// A literal also translates through Zig 0.16 freestanding stdint headers.
pub const KORE_FUTEX_INFINITE: u64 = 0xffff_ffff_ffff_ffff;
/// Wake argument was null or misaligned; no native wake was attempted.
pub const KORE_FUTEX_INVALID_ARGUMENT: i32 = -1;

/// Native wait outcomes, with an additional adapter argument refusal.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KoreFutexOutcome {
    /// The word differed from expected; re-check the caller's condition.
    Retry = 0,
    /// A wake may be spurious; re-check the caller's condition.
    Woken = 1,
    TimedOut = 2,
    /// This native thread cannot park. Do not retry in a busy loop.
    Unsupported = 3,
    InvalidArgument = 4,
}

impl From<futex::WaitOutcome> for KoreFutexOutcome {
    fn from(value: futex::WaitOutcome) -> Self {
        match value {
            futex::WaitOutcome::Retry => Self::Retry,
            futex::WaitOutcome::Woken => Self::Woken,
            futex::WaitOutcome::TimedOut => Self::TimedOut,
            futex::WaitOutcome::Unsupported => Self::Unsupported,
        }
    }
}

unsafe fn word<'a>(ptr: *mut u32) -> Option<&'a AtomicU32> {
    if ptr.is_null()
        || !ptr
            .addr()
            .is_multiple_of(core::mem::align_of::<AtomicU32>())
    {
        return None;
    }
    // SAFETY: the foreign caller guarantees live atomic-compatible storage.
    Some(unsafe { AtomicU32::from_ptr(ptr) })
}

/// Whether the current native thread can wait (1=yes, 0=no).
#[unsafe(no_mangle)]
pub extern "C" fn kore_futex_available() -> u32 {
    u32::from(futex::available())
}

/// Block this native thread only while word equals expected. timeout_ns is
/// relative; KORE_FUTEX_INFINITE means unbounded. Always re-check the condition
/// after Retry/Woken; Unsupported is a permanent refusal for this context.
/// Process-local only. Do not call from executor callbacks: Rust tasks should
/// use kore::futex::wait_async instead.
/// # Safety
/// ptr is null, misaligned, or points to a live writable 32-bit word aligned
/// to at least 4 bytes. Valid storage must stay at the same address throughout
/// all waits/wakes. All concurrent accesses must be compatible 32-bit atomic
/// operations (C atomic builtins or Zig atomics); never mix ordinary accesses
/// or differently sized atomics. Managed-thread waits are cancellation points:
/// they may exit without stack unwinding, as specified by kore_thread_testcancel.
/// Publish the condition with a sequentially consistent atomic store BEFORE waking.
/// No address-space sharing is implied.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_futex_wait(
    ptr: *mut u32,
    expected: u32,
    timeout_ns: u64,
) -> KoreFutexOutcome {
    let Some(word) = (unsafe { word(ptr) }) else {
        return KoreFutexOutcome::InvalidArgument;
    };
    let timeout = (timeout_ns != KORE_FUTEX_INFINITE).then(|| Duration::from_nanos(timeout_ns));
    if !crate::thread_local::cancellable() {
        return futex::wait(word, expected, timeout).into();
    }
    // A bounded native park makes cancellation observable even without a peer
    // kick capability. Recompute against one deadline; slices never restart it.
    unsafe { crate::kore_thread_testcancel() };
    let start = match kore::time::uptime() {
        Ok(time) => time.as_nanos(),
        Err(_) => return KoreFutexOutcome::Unsupported,
    };
    let deadline = timeout.map(|_| start.saturating_add(timeout_ns));
    loop {
        unsafe { crate::kore_thread_testcancel() };
        let now = match kore::time::uptime() {
            Ok(time) => time.as_nanos(),
            Err(_) => return KoreFutexOutcome::Unsupported,
        };
        let slice = deadline.map_or(1_000_000, |end| end.saturating_sub(now).min(1_000_000));
        let outcome = futex::wait(word, expected, Some(Duration::from_nanos(slice)));
        unsafe { crate::kore_thread_testcancel() };
        if outcome != futex::WaitOutcome::TimedOut {
            return outcome.into();
        }
        if let Some(end) = deadline {
            match kore::time::uptime() {
                Ok(time) if time.as_nanos() < end => {}
                Ok(_) => return KoreFutexOutcome::TimedOut,
                Err(_) => return KoreFutexOutcome::Unsupported,
            }
        }
    }
}

/// Wake one waiter. Returns 1 if a direct or relay notice was posted, 0 if no
/// waiter was reached, or KORE_FUTEX_INVALID_ARGUMENT. Zero does not prove no
/// waiter exists; one does not prove a condition is satisfied. Not a fair queue.
/// # Safety
/// Same atomic storage/lifetime and store-before-wake contract as kore_futex_wait.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_futex_wake(ptr: *mut u32) -> i32 {
    match unsafe { word(ptr) } {
        Some(word) => i32::from(futex::wake(word)),
        None => KORE_FUTEX_INVALID_ARGUMENT,
    }
}

/// Wake all registered waiters. Returns posted-notice count, including at most
/// one relay notice, or KORE_FUTEX_INVALID_ARGUMENT. The count is telemetry,
/// not a count of resumed threads or a synchronization guarantee.
/// # Safety
/// Same atomic storage/lifetime and store-before-wake contract as kore_futex_wait.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_futex_wake_all(ptr: *mut u32) -> i32 {
    match unsafe { word(ptr) } {
        // The native registry is bounded by MAX_THREADS (32).
        Some(word) => futex::wake_all(word) as i32,
        None => KORE_FUTEX_INVALID_ARGUMENT,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preserves_every_native_outcome() {
        for (native, foreign) in [
            (futex::WaitOutcome::Retry, KoreFutexOutcome::Retry),
            (futex::WaitOutcome::Woken, KoreFutexOutcome::Woken),
            (futex::WaitOutcome::TimedOut, KoreFutexOutcome::TimedOut),
            (
                futex::WaitOutcome::Unsupported,
                KoreFutexOutcome::Unsupported,
            ),
        ] {
            assert_eq!(KoreFutexOutcome::from(native), foreign);
        }
    }
    #[test]
    fn invalid_storage_refuses_and_host_wait_is_not_a_fake_timeout() {
        let atomic = AtomicU32::new(7);
        let mut bytes = [0u32; 2];
        let misaligned = bytes.as_mut_ptr().cast::<u8>().wrapping_add(1).cast();
        unsafe {
            for ptr in [core::ptr::null_mut(), misaligned] {
                assert_eq!(
                    kore_futex_wait(ptr, 7, 0),
                    KoreFutexOutcome::InvalidArgument
                );
                assert_eq!(kore_futex_wake(ptr), KORE_FUTEX_INVALID_ARGUMENT);
                assert_eq!(kore_futex_wake_all(ptr), KORE_FUTEX_INVALID_ARGUMENT);
            }
            assert_eq!(kore_futex_available(), 0);
            assert_eq!(
                kore_futex_wait(atomic.as_ptr(), 7, 0),
                KoreFutexOutcome::Unsupported
            );
            assert_eq!(kore_futex_wake(atomic.as_ptr()), 0);
            assert_eq!(kore_futex_wake_all(atomic.as_ptr()), 0);
        }
    }
}
