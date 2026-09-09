use super::UserCounter;

pub(super) struct Counter;

impl UserCounter for Counter {
    #[inline]
    fn ticks() -> u64 {
        // SAFETY: kernel per-CPU chrono bring-up clears CR4.TSD before user
        // dispatch, and the boot block advertises readability only with a
        // calibrated source.
        unsafe { core::arch::x86_64::_rdtsc() }
    }
}
