use super::UserCounter;

pub(super) struct Counter;

impl UserCounter for Counter {
    #[inline]
    fn ticks() -> u64 {
        let ticks: u64;
        // SAFETY: kernel per-hart chrono bring-up sets scounteren.TM before
        // user dispatch.
        unsafe {
            core::arch::asm!(
                "rdtime {ticks}",
                ticks = out(reg) ticks,
                options(nomem, nostack, preserves_flags)
            );
        }
        ticks
    }
}
