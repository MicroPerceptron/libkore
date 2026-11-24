//! The keyboard's **lock indicators** — the PS/2 "Set/Reset Status
//! Indicators" command (`0xED`), as a state machine driven by the
//! keyboard's replies.
//!
//! Lighting an LED is a two-byte exchange on the same port the keys arrive
//! on: the host writes `0xED`, the keyboard acknowledges (`0xFA`), the host
//! writes the indicator byte, the keyboard acknowledges again. A `0xFE`
//! asks for the last byte again. Nothing about it may block the key path:
//! keys keep arriving between the bytes, so the exchange is a state machine
//! the relay advances with each [`Reply`] the decoder hands it, and with
//! time — a keyboard that never answers (a KVM, a keyboard without LEDs)
//! must cost a bounded number of attempts, not a wedge.
//!
//! The keyboard's indicator byte is **not** the HID LED order: bit 0 is
//! Scroll Lock, bit 1 Num Lock, bit 2 Caps Lock (the PS/2 keyboard
//! protocol; Linux 7.2.11 `atkbd_set_leds` builds the same byte).

use kore_abi::keyboard::Locks;

use crate::decode::Reply;

/// The "Set/Reset Status Indicators" command byte.
pub const SET_LEDS: u8 = 0xED;

/// The "Enable" command byte: resume scanning (and end any wait for an
/// option byte). Linux 7.2.11 `atkbd_activate` sends the same byte.
pub const ENABLE: u8 = 0xF4;

/// How long a written byte may go unacknowledged before the exchange is
/// abandoned. A keyboard answers within a millisecond; Linux allows
/// 200 ms for a command byte, and so does this.
pub const ACK_TIMEOUT_NS: u64 = 200_000_000;

/// Without a clock, how many relay wakes may pass with a byte unanswered
/// before the exchange is abandoned — the bound a machine with no
/// calibrated counter still needs, or an unanswered byte would leave the
/// machine waiting forever. Wakes, not time: without a clock the relay
/// parks on the line alone, so each is a byte the keyboard sent meanwhile.
const ACK_TIMEOUT_WAKES: u8 = 4;

/// Resends honoured per byte before the exchange is abandoned.
const MAX_RESENDS: u8 = 3;

/// Abandoned exchanges for one target state before giving up on it until
/// the target changes — so a keyboard without indicators costs three tries,
/// not one per wake forever. The same bound applies to the `Enable` that
/// ends an abandoned exchange.
const MAX_FAILURES: u8 = 3;

/// The keyboard's indicator byte for a lock state.
pub const fn indicator_byte(locks: Locks) -> u8 {
    (if locks.contains(Locks::SCROLL) { 1 << 0 } else { 0 })
        | (if locks.contains(Locks::NUM) { 1 << 1 } else { 0 })
        | (if locks.contains(Locks::CAPS) { 1 << 2 } else { 0 })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Idle,
    /// `0xED` written; its acknowledgement is owed.
    Command,
    /// The indicator byte written; its acknowledgement is owed.
    Data,
    /// `0xF4` written to end an abandoned exchange; its acknowledgement is
    /// owed.
    Enable,
}

/// One keyboard's indicator state and the exchange that converges it.
///
/// **No abandoned exchange may leave the keyboard deaf.** A keyboard that
/// has accepted `0xED` stops scanning until it receives the option byte
/// (IBM PS/2 keyboard protocol), and an exchange can be abandoned exactly
/// there: the indicator byte refused past its resends, never acknowledged,
/// or never written. So every abandoned exchange is ended with `Enable`,
/// which any keyboard takes as a command — ending a wait for an option
/// byte — and which a scanning keyboard merely acknowledges. Until that
/// lands (or the keyboard proves it is scanning by sending a key, or
/// resets), it takes precedence over any new exchange and over the
/// per-state failure bound.
#[derive(Clone, Copy, Debug)]
pub struct Indicators {
    stage: Stage,
    /// The state being written by the exchange in flight.
    target: Locks,
    /// What the keyboard is known to show; `None` until the first
    /// exchange completes (or after one is abandoned).
    shown: Option<Locks>,
    /// When the byte in flight was written (monotonic ns), when known.
    sent_at: Option<u64>,
    /// Wakes that passed with the byte in flight unanswered and no clock
    /// to time it.
    waited: u8,
    resends: u8,
    /// Consecutive abandoned exchanges for [`failed_target`](Self).
    failures: u8,
    failed_target: Locks,
    /// An exchange was abandoned and the keyboard may be waiting for an
    /// option byte: the next byte written is `Enable`.
    needs_enable: bool,
    /// Consecutive `Enable`s that went unanswered.
    enable_failures: u8,
}

impl Default for Indicators {
    fn default() -> Self {
        Self::new()
    }
}

impl Indicators {
    pub const fn new() -> Self {
        Self {
            stage: Stage::Idle,
            target: Locks::NONE,
            shown: None,
            sent_at: None,
            waited: 0,
            resends: 0,
            failures: 0,
            failed_target: Locks::NONE,
            needs_enable: false,
            enable_failures: 0,
        }
    }

    /// An acknowledgement or resend request belongs to this exchange.
    pub const fn awaiting(&self) -> bool {
        !matches!(self.stage, Stage::Idle)
    }

    /// What the keyboard is known to show.
    pub const fn shown(&self) -> Option<Locks> {
        self.shown
    }

    /// Start converging on `desired` if the keyboard may not show it and no
    /// exchange is in flight: the byte to write now, if any. An abandoned
    /// exchange's `Enable` goes first.
    pub fn start(&mut self, desired: Locks, now: Option<u64>) -> Option<u8> {
        if self.awaiting() {
            return None;
        }
        if self.needs_enable && self.enable_failures < MAX_FAILURES {
            self.stage = Stage::Enable;
            return Some(self.sent(ENABLE, now));
        }
        if self.shown == Some(desired) {
            return None;
        }
        if self.failures >= MAX_FAILURES && self.failed_target == desired {
            return None;
        }
        self.stage = Stage::Command;
        self.target = desired;
        Some(self.sent(SET_LEDS, now))
    }

    /// Advance on a reply; the byte to write next, if any. A reply that
    /// arrives while no exchange is in flight is not this machine's (a
    /// stray acknowledgement) and changes nothing.
    pub fn reply(&mut self, reply: Reply, now: Option<u64>) -> Option<u8> {
        match (self.stage, reply) {
            // A reset is news whenever it arrives: the keyboard went dark.
            (_, Reply::SelfTestPassed) => {
                self.keyboard_reset();
                None
            }
            (Stage::Idle, _) => None,
            (Stage::Command, Reply::Ack) => {
                self.stage = Stage::Data;
                Some(self.sent(indicator_byte(self.target), now))
            }
            (Stage::Data, Reply::Ack) => {
                self.stage = Stage::Idle;
                self.shown = Some(self.target);
                self.failures = 0;
                None
            }
            (Stage::Enable, Reply::Ack) => {
                self.stage = Stage::Idle;
                self.needs_enable = false;
                self.enable_failures = 0;
                None
            }
            (stage, Reply::Resend) if self.resends < MAX_RESENDS => {
                let resends = self.resends + 1;
                let byte = match stage {
                    Stage::Command => SET_LEDS,
                    Stage::Enable => ENABLE,
                    _ => indicator_byte(self.target),
                };
                let byte = self.sent(byte, now);
                self.resends = resends;
                Some(byte)
            }
            // Out of resends, or the keyboard erred mid-exchange: the
            // exchange is over and the keyboard's state is unknown.
            (_, Reply::Resend | Reply::Error) => {
                self.abandon();
                None
            }
        }
    }

    /// The keyboard sent a key: it is scanning, so it is not waiting for an
    /// option byte, whatever an abandoned exchange left uncertain.
    pub fn scanned(&mut self) {
        if !self.awaiting() {
            self.needs_enable = false;
            self.enable_failures = 0;
        }
    }

    /// The keyboard reset itself (its self-test reply): every indicator is
    /// dark, it is scanning, and any exchange in flight is void.
    pub fn keyboard_reset(&mut self) {
        self.stage = Stage::Idle;
        self.shown = Some(Locks::NONE);
        self.failures = 0;
        self.needs_enable = false;
        self.enable_failures = 0;
    }

    /// Abandon an exchange whose acknowledgement is overdue at `now` — by
    /// the clock when there is one, by relay wakes when there is not.
    pub fn expire(&mut self, now: Option<u64>) {
        if !self.awaiting() {
            return;
        }
        let overdue = match (self.sent_at, now) {
            (Some(sent), Some(now)) => now.saturating_sub(sent) >= ACK_TIMEOUT_NS,
            _ => {
                self.waited = self.waited.saturating_add(1);
                self.waited >= ACK_TIMEOUT_WAKES
            }
        };
        if overdue {
            self.abandon();
        }
    }

    /// The byte this machine asked for could not be written (the
    /// controller never took it): the exchange is abandoned now rather
    /// than left to time out.
    pub fn write_failed(&mut self) {
        if self.awaiting() {
            self.abandon();
        }
    }

    /// Record a byte as written at `now`: its acknowledgement is owed from
    /// here, with a fresh resend and wake budget.
    fn sent(&mut self, byte: u8, now: Option<u64>) -> u8 {
        self.sent_at = now;
        self.waited = 0;
        self.resends = 0;
        byte
    }

    fn abandon(&mut self) {
        let stage = core::mem::replace(&mut self.stage, Stage::Idle);
        if stage == Stage::Enable {
            self.enable_failures = self.enable_failures.saturating_add(1);
            return;
        }
        // The keyboard may have accepted `0xED` and now be waiting for an
        // option byte, not scanning: end that wait before anything else.
        self.needs_enable = true;
        self.shown = None;
        if self.failed_target == self.target {
            self.failures = self.failures.saturating_add(1);
        } else {
            self.failed_target = self.target;
            self.failures = 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Settle an abandoned exchange's `Enable`, as a scanning keyboard does.
    fn enabled(leds: &mut Indicators, now: u64) {
        assert_eq!(leds.start(Locks::NONE, Some(now)), Some(ENABLE));
        assert_eq!(leds.reply(Reply::Ack, Some(now)), None);
    }

    #[test]
    fn the_indicator_byte_is_the_ps2_order_not_the_hid_one() {
        assert_eq!(indicator_byte(Locks::SCROLL), 0b001);
        assert_eq!(indicator_byte(Locks::NUM), 0b010);
        assert_eq!(indicator_byte(Locks::CAPS), 0b100);
        assert_eq!(indicator_byte(Locks::CAPS.toggled(Locks::NUM)), 0b110);
    }

    #[test]
    fn an_exchange_writes_the_command_then_the_byte_and_converges() {
        let mut leds = Indicators::new();
        assert_eq!(leds.start(Locks::CAPS, Some(0)), Some(SET_LEDS));
        assert!(leds.awaiting());
        // A second request while one is in flight waits its turn.
        assert_eq!(leds.start(Locks::NUM, Some(1)), None);
        assert_eq!(leds.reply(Reply::Ack, Some(2)), Some(0b100));
        assert_eq!(leds.reply(Reply::Ack, Some(3)), None);
        assert_eq!(leds.shown(), Some(Locks::CAPS));
        assert!(!leds.awaiting());
        // Converged: nothing more to write until the state changes.
        assert_eq!(leds.start(Locks::CAPS, Some(4)), None);
        assert_eq!(leds.start(Locks::NUM, Some(5)), Some(SET_LEDS));
    }

    #[test]
    fn a_resend_repeats_the_byte_in_flight_a_bounded_number_of_times() {
        let mut leds = Indicators::new();
        leds.start(Locks::CAPS, Some(0));
        assert_eq!(leds.reply(Reply::Resend, Some(1)), Some(SET_LEDS));
        assert_eq!(leds.reply(Reply::Ack, Some(2)), Some(0b100));
        for _ in 0..MAX_RESENDS {
            assert_eq!(leds.reply(Reply::Resend, Some(3)), Some(0b100));
        }
        assert_eq!(leds.reply(Reply::Resend, Some(4)), None, "out of resends");
        assert!(!leds.awaiting());
        assert_eq!(leds.shown(), None);
    }

    /// The keyboard acknowledged `0xED`, so it stopped scanning to wait for
    /// the option byte — and the exchange died there. Whatever the failure
    /// bound says about the indicators, the keyboard is told `Enable`.
    #[test]
    fn an_exchange_abandoned_after_its_command_ends_with_enable() {
        let mut leds = Indicators::new();
        for attempt in 0..MAX_FAILURES {
            let t = u64::from(attempt) * 10;
            assert_eq!(leds.start(Locks::CAPS, Some(t)), Some(SET_LEDS));
            assert_eq!(leds.reply(Reply::Ack, Some(t)), Some(0b100));
            // The option byte never lands.
            leds.write_failed();
            assert!(!leds.awaiting());
            // The next byte is Enable, before any retry of the indicators.
            assert_eq!(leds.start(Locks::CAPS, Some(t + 1)), Some(ENABLE));
            assert_eq!(leds.reply(Reply::Ack, Some(t + 2)), None);
        }
        // The indicators are given up on; the keyboard is not left deaf.
        assert_eq!(leds.start(Locks::CAPS, Some(100)), None);
        assert_eq!(leds.start(Locks::NUM, Some(100)), Some(SET_LEDS));
    }

    /// Enable outranks the failure bound, and is bounded itself: a keyboard
    /// that answers nothing at all costs a few bytes, not one per wake.
    #[test]
    fn enable_is_retried_a_bounded_number_of_times_and_a_key_settles_it() {
        let mut leds = Indicators::new();
        leds.start(Locks::CAPS, Some(0));
        leds.expire(Some(ACK_TIMEOUT_NS));
        for attempt in 0..MAX_FAILURES {
            let t = (u64::from(attempt) + 2) * ACK_TIMEOUT_NS;
            assert_eq!(leds.start(Locks::CAPS, Some(t)), Some(ENABLE));
            leds.expire(Some(t + ACK_TIMEOUT_NS));
        }
        assert_ne!(leds.start(Locks::CAPS, Some(u64::MAX / 2)), Some(ENABLE));

        // A key proves the keyboard is scanning: no Enable is owed.
        let mut leds = Indicators::new();
        leds.start(Locks::CAPS, Some(0));
        leds.write_failed();
        leds.scanned();
        assert_eq!(leds.start(Locks::CAPS, Some(1)), Some(SET_LEDS));
    }

    #[test]
    fn a_silent_keyboard_costs_bounded_attempts_per_target() {
        let mut leds = Indicators::new();
        for attempt in 0..MAX_FAILURES {
            let t = u64::from(attempt) * ACK_TIMEOUT_NS * 4;
            assert_eq!(leds.start(Locks::CAPS, Some(t)), Some(SET_LEDS));
            leds.expire(Some(t + ACK_TIMEOUT_NS - 1));
            assert!(leds.awaiting(), "not yet overdue");
            leds.expire(Some(t + ACK_TIMEOUT_NS));
            assert!(!leds.awaiting());
            enabled(&mut leds, t + 2 * ACK_TIMEOUT_NS);
        }
        assert_eq!(leds.start(Locks::CAPS, Some(u64::MAX / 2)), None, "given up on CAPS");
        // A new target is a new chance.
        assert_eq!(leds.start(Locks::NUM, Some(u64::MAX / 2)), Some(SET_LEDS));
    }

    /// Without a calibrated clock an unanswered byte still expires — by the
    /// relay's wakes — instead of leaving the machine waiting forever.
    #[test]
    fn without_a_clock_an_exchange_expires_by_wakes() {
        let mut leds = Indicators::new();
        assert_eq!(leds.start(Locks::CAPS, None), Some(SET_LEDS));
        for _ in 1..ACK_TIMEOUT_WAKES {
            leds.expire(None);
            assert!(leds.awaiting());
        }
        leds.expire(None);
        assert!(!leds.awaiting());
        assert_eq!(leds.start(Locks::CAPS, None), Some(ENABLE));
        // A reply resets the wake budget for the next byte.
        assert_eq!(leds.reply(Reply::Ack, None), None);
        assert_eq!(leds.start(Locks::CAPS, None), Some(SET_LEDS));
        for _ in 1..ACK_TIMEOUT_WAKES {
            leds.expire(None);
        }
        assert_eq!(leds.reply(Reply::Ack, None), Some(0b100));
        leds.expire(None);
        assert!(leds.awaiting(), "a fresh budget for the option byte");
    }

    #[test]
    fn stray_replies_change_nothing_and_a_reset_darkens_every_indicator() {
        let mut leds = Indicators::new();
        assert_eq!(leds.reply(Reply::Ack, Some(0)), None);
        assert_eq!(leds.shown(), None);
        leds.start(Locks::SCROLL, Some(0));
        leds.reply(Reply::Ack, Some(1));
        leds.reply(Reply::Ack, Some(2));
        assert_eq!(leds.shown(), Some(Locks::SCROLL));
        assert_eq!(leds.reply(Reply::SelfTestPassed, Some(3)), None);
        assert_eq!(leds.shown(), Some(Locks::NONE));
        // A replugged keyboard is re-lit with the seat's state.
        assert_eq!(leds.start(Locks::SCROLL, Some(4)), Some(SET_LEDS));
        // A reset mid-exchange also ends any owed Enable: it is scanning.
        leds.write_failed();
        assert_eq!(leds.reply(Reply::SelfTestPassed, Some(5)), None);
        assert_eq!(leds.start(Locks::SCROLL, Some(6)), Some(SET_LEDS));
    }
}
