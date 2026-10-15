//! Ctrl-C for an interactive child (`res/LIFECYCLE_HARDENING.md` §3):
//! notice, bounded grace, revoke. This is the one policy every holder of a
//! process stream's parent end follows — the shell for its foreground child,
//! a `term` window for the program it hosts (`res/SHELL_TERMINAL.md` §7.3).
//!
//! The first press asks the child to stop: the stream's parent sends one
//! [`ProcessInputKind::Interrupt`] record, and a child that reads it tears
//! down what it holds (a device queue, DMA windows another service mapped)
//! before it exits. Revoking the task instead cuts those windows out from
//! under a device that may still be using them, which is what sealed the NPU
//! and GPU after a Ctrl-C. The parent still revokes, but only once the child
//! is plainly not listening: a second press while the record is still unread,
//! or the grace running out with it unread. A child that read the record is
//! handling it (a REPL abandons its line and prompts again; `inference`
//! releases its devices and exits), so reading it ends the grace.
//!
//! The notice is a record the child reads at its own safe points, never a
//! handler run at a distance: no code of the child's runs because of it until
//! the child looks. The authority that ends an unheeding child stays a
//! capability (the parent's task handle, or a delegated foreground slot).
//!
//! [`ProcessInputKind::Interrupt`]: kore_abi::process::ProcessInputKind::Interrupt

/// How long an unread interrupt waits before the parent revokes the task. Long
/// enough for a child to reach its next safe point from inside a prompt
/// prefill; a second Ctrl-C revokes at once.
pub const INTERRUPT_GRACE_NS: u64 = 60_000_000_000;

/// What one Ctrl-C press asks of the stream's parent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Press {
    /// Send the child an interrupt record and start the grace.
    Forward,
    /// The child left an earlier interrupt unread: revoke it now.
    Revoke,
}

/// The one interrupt a stream's parent may have outstanding.
#[derive(Clone, Copy, Debug, Default)]
pub struct Interrupt {
    /// When the outstanding record was sent, until the child reads it.
    sent_ns: Option<u64>,
}

impl Interrupt {
    pub const fn new() -> Self {
        Self { sent_ns: None }
    }

    /// One Ctrl-C press. `unread` is how many input records the child has
    /// not consumed; the interrupt, sent last, is read last.
    pub fn press(&mut self, now_ns: u64, unread: u32) -> Press {
        self.observe(unread);
        if self.sent_ns.is_some() {
            return Press::Revoke;
        }
        self.sent_ns = Some(now_ns);
        Press::Forward
    }

    /// Whether the grace ran out with the interrupt still unread. Reading it
    /// ends the grace.
    pub fn expired(&mut self, now_ns: u64, unread: u32) -> bool {
        self.observe(unread);
        self.sent_ns
            .is_some_and(|sent| now_ns.saturating_sub(sent) >= INTERRUPT_GRACE_NS)
    }

    /// Whether an interrupt is outstanding, as last observed.
    pub fn outstanding(&self) -> bool {
        self.sent_ns.is_some()
    }

    /// The longest the parent may park before it must look at the grace
    /// again, or `None` with no interrupt outstanding.
    pub fn park_ns(&self, now_ns: u64) -> Option<u64> {
        self.sent_ns.map(|sent| {
            INTERRUPT_GRACE_NS
                .saturating_sub(now_ns.saturating_sub(sent))
                .max(1)
        })
    }

    fn observe(&mut self, unread: u32) {
        if unread == 0 {
            self.sent_ns = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_first_press_forwards_and_a_second_unread_one_revokes() {
        let mut interrupt = Interrupt::new();
        assert_eq!(interrupt.press(10, 0), Press::Forward);
        // The record is in the ring and unread.
        assert_eq!(interrupt.press(20, 1), Press::Revoke);
    }

    #[test]
    fn a_read_interrupt_ends_the_grace_and_the_next_press_forwards_again() {
        let mut interrupt = Interrupt::new();
        assert_eq!(interrupt.press(0, 0), Press::Forward);
        assert!(!interrupt.expired(1, 1));
        // The child read it (a REPL clearing its line): no kill pending.
        assert!(!interrupt.expired(2, 0));
        assert_eq!(interrupt.park_ns(2), None);
        assert!(!interrupt.expired(INTERRUPT_GRACE_NS * 2, 0));
        // Another press is a fresh request, not a revoke.
        assert_eq!(interrupt.press(3, 0), Press::Forward);
    }

    #[test]
    fn an_unread_interrupt_expires_after_the_grace() {
        let mut interrupt = Interrupt::new();
        assert_eq!(interrupt.press(100, 2), Press::Forward);
        assert!(!interrupt.expired(100 + INTERRUPT_GRACE_NS - 1, 3));
        assert!(interrupt.expired(100 + INTERRUPT_GRACE_NS, 3));
    }

    #[test]
    fn the_park_bound_counts_down_to_the_grace() {
        let mut interrupt = Interrupt::new();
        assert_eq!(interrupt.park_ns(0), None);
        let _ = interrupt.press(1_000, 0);
        assert_eq!(interrupt.park_ns(1_000), Some(INTERRUPT_GRACE_NS));
        assert_eq!(interrupt.park_ns(1_000 + INTERRUPT_GRACE_NS), Some(1));
    }

    #[test]
    fn earlier_unread_input_holds_the_interrupt_behind_it() {
        // Two data records the child never read, then the interrupt: all
        // three must drain before the grace ends.
        let mut interrupt = Interrupt::new();
        assert_eq!(interrupt.press(0, 2), Press::Forward);
        assert!(!interrupt.expired(5, 1));
        assert_eq!(interrupt.press(6, 1), Press::Revoke);
    }
}
