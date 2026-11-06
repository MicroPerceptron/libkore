//! # `keyboard` — what a keyboard [`InputReport`](crate::svc::InputReport) says
//!
//! A keyboard snapshot is HID boot-protocol **state**: a modifier bitmap, up
//! to six held usages of the Keyboard/Keypad page, and — stamped by the
//! keymap owner, never by a device — the seat's [`Locks`]. This module is
//! the vocabulary for those three fields, shared by every producer (the
//! PS/2, USB and I2C drivers) and every consumer (the terminal discipline,
//! the compositor's window clients), so no layer keeps a private copy of a
//! spec table that could drift from the others.
//!
//! What a key *means* in a given layout is policy, not ABI, and lives with
//! the keymap; this module only names the keys, the modifier bits and the
//! lock bits — facts of the HID specification and of the report layout.

#[rustfmt::skip]
pub mod usage;

/// The modifier bitmap — boot report byte 0, a keyboard report's `state`.
///
/// Bit *n* is usage `0xE0 + n` (HID 1.11 Appendix B.1): the eight modifier
/// keys never occupy the six-key array, they ride this byte instead.
pub mod modifier {
    use super::usage;

    pub const LEFT_CTRL: u8 = 1 << 0;
    pub const LEFT_SHIFT: u8 = 1 << 1;
    pub const LEFT_ALT: u8 = 1 << 2;
    pub const LEFT_GUI: u8 = 1 << 3;
    pub const RIGHT_CTRL: u8 = 1 << 4;
    pub const RIGHT_SHIFT: u8 = 1 << 5;
    pub const RIGHT_ALT: u8 = 1 << 6;
    pub const RIGHT_GUI: u8 = 1 << 7;

    /// Either Control.
    pub const CTRL: u8 = LEFT_CTRL | RIGHT_CTRL;
    /// Either Shift.
    pub const SHIFT: u8 = LEFT_SHIFT | RIGHT_SHIFT;
    /// Either Alt.
    pub const ALT: u8 = LEFT_ALT | RIGHT_ALT;
    /// Either GUI (Super / Windows / Command).
    pub const GUI: u8 = LEFT_GUI | RIGHT_GUI;

    /// The bitmap bit a modifier usage occupies; `None` for every other
    /// usage.
    pub const fn bit(key: u8) -> Option<u8> {
        if key >= usage::LEFT_CTRL && key <= usage::RIGHT_GUI {
            Some(1 << (key - usage::LEFT_CTRL))
        } else {
            None
        }
    }
}

/// The lock state a keyboard report carries: which of Num Lock, Caps Lock
/// and Scroll Lock are engaged.
///
/// The bit order is the HID LED page's (usages 1–3, bit = usage − 1),
/// which is also the boot keyboard's LED output report (HID 1.11 Appendix
/// B.1) — so the byte a report carries is, unchanged, the byte a USB
/// keyboard's indicators are set with. Only the three toggling locks are
/// representable: a byte with any other bit is not a lock state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Locks(u8);

impl Locks {
    /// No lock engaged.
    pub const NONE: Self = Self(0);
    /// Num Lock (LED usage 0x01).
    pub const NUM: Self = Self(1 << 0);
    /// Caps Lock (LED usage 0x02).
    pub const CAPS: Self = Self(1 << 1);
    /// Scroll Lock (LED usage 0x03).
    pub const SCROLL: Self = Self(1 << 2);
    /// Every representable lock bit.
    pub const MASK: u8 = Self::NUM.0 | Self::CAPS.0 | Self::SCROLL.0;

    /// A lock state from its wire byte; `None` when any bit outside
    /// [`MASK`](Self::MASK) is set.
    pub const fn from_bits(bits: u8) -> Option<Self> {
        if bits & !Self::MASK == 0 {
            Some(Self(bits))
        } else {
            None
        }
    }

    /// The wire byte (also the boot keyboard LED output report byte).
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Every lock in `other` is engaged here.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// This state with the locks in `other` flipped.
    #[must_use]
    pub const fn toggled(self, other: Self) -> Self {
        Self(self.0 ^ other.0)
    }

    /// The lock a momentary lock key toggles on its press edge, `None` for
    /// every other key.
    pub const fn toggled_by(key: u8) -> Option<Self> {
        match key {
            usage::CAPS_LOCK => Some(Self::CAPS),
            usage::NUM_LOCK => Some(Self::NUM),
            usage::SCROLL_LOCK => Some(Self::SCROLL),
            _ => None,
        }
    }

    /// The lock a **latching** lock key holds, `None` for every other key.
    ///
    /// HUT 1.5 §10 defines Locking Caps/Num/Scroll Lock (0x82–0x84) as keys
    /// "implemented as a locking key; sent as a toggle button": the key is
    /// reported held for as long as it is mechanically latched down. Each of
    /// its edges — latching and unlatching — is therefore one toggle, where
    /// a momentary key's release is none.
    pub const fn latched_by(key: u8) -> Option<Self> {
        match key {
            usage::LOCKING_CAPS_LOCK => Some(Self::CAPS),
            usage::LOCKING_NUM_LOCK => Some(Self::NUM),
            usage::LOCKING_SCROLL_LOCK => Some(Self::SCROLL),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modifier_bits_follow_the_usage_order() {
        assert_eq!(modifier::bit(usage::LEFT_CTRL), Some(modifier::LEFT_CTRL));
        assert_eq!(modifier::bit(usage::LEFT_GUI), Some(modifier::LEFT_GUI));
        assert_eq!(modifier::bit(usage::RIGHT_GUI), Some(modifier::RIGHT_GUI));
        assert_eq!(modifier::bit(usage::A), None);
        assert_eq!(modifier::bit(0xE8), None);
    }

    #[test]
    fn locks_reject_unrepresentable_bits_and_toggle_per_key() {
        assert_eq!(Locks::from_bits(0b111), Some(Locks(0b111)));
        assert_eq!(Locks::from_bits(1 << 3), None, "Compose is an LED, not a lock");
        let caps = Locks::NONE.toggled(Locks::toggled_by(usage::CAPS_LOCK).unwrap());
        assert!(caps.contains(Locks::CAPS) && !caps.contains(Locks::NUM));
        assert_eq!(caps.toggled(Locks::CAPS), Locks::NONE);
        assert_eq!(Locks::toggled_by(usage::LOCKING_NUM_LOCK), None);
        assert_eq!(Locks::latched_by(usage::LOCKING_NUM_LOCK), Some(Locks::NUM));
        assert_eq!(Locks::latched_by(usage::NUM_LOCK), None);
        assert_eq!(Locks::toggled_by(usage::KP_1), None);
    }

    /// The bit order is the boot keyboard's LED output report — the byte a
    /// report carries goes to a USB keyboard unchanged.
    #[test]
    fn lock_bits_are_the_boot_led_report() {
        assert_eq!(Locks::NUM.bits(), 0x01);
        assert_eq!(Locks::CAPS.bits(), 0x02);
        assert_eq!(Locks::SCROLL.bits(), 0x04);
    }
}
