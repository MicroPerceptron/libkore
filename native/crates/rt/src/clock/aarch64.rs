use super::UserCounter;

pub(super) struct Counter;

impl UserCounter for Counter {
    #[inline]
    fn ticks() -> u64 {
        let ticks: u64;
        // SAFETY: kernel per-CPU chrono bring-up sets CNTKCTL_EL1.EL0VCTEN.
        // ISB orders the sample after preceding architectural effects.
        unsafe {
            core::arch::asm!(
                "isb",
                "mrs {ticks}, cntvct_el0",
                ticks = out(reg) ticks,
                options(nomem, nostack, preserves_flags)
            );
        }
        ticks
    }
}
