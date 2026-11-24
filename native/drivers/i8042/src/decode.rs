//! Scancode decoding — bytes from the controller's data port to key
//! events and device replies (#419, PS/2 epic #414).
//!
//! **A pure function over bytes**, which is why it needs no hardware
//! model of any kind: the input is a byte sequence the specification
//! defines, and the output is a key event or the device's reply to a
//! command. Every case below is exercised with real sequences.
//!
//! # Two sets, and why the driver must ask rather than assume
//!
//! An i8042 can hand up **set 1** (the XT-derived codes, what firmware
//! usually leaves enabled through the controller's translation bit) or
//! **set 2** (what AT keyboards actually emit). They are different
//! encodings of the same keys, and the single most common way a PS/2
//! driver goes wrong is assuming one. The bring-up ladder reads the
//! active set and the translation bit and tells this decoder which it is
//! looking at; nothing here guesses.
//!
//! The tables ([`crate::scancode`]) name every key by its HID
//! Keyboard/Keypad usage — the identity the rest of the system speaks — and
//! set 1 is set 2 seen through the controller's translation, so the two
//! sets decode every key identically *by construction*, not by care.
//!
//! # The escapes
//!
//! - `0xE0` prefixes an *extended* key (the arrows, the right-hand
//!   modifiers, the keypad's divide and Enter, the GUI keys). The prefix
//!   and the code together are one key.
//! - `0xE1` prefixes a two-code sequence whose *first* code is inert and
//!   whose second reads as extended — Linux's `atkbd` rule. It is how the
//!   Pause key is spelled: `E1 1D 45` / `E1 14 77` is its make, and the
//!   break bits (or `F0`s) on the second half its break, so Pause decodes
//!   as an ordinary press followed by an ordinary release instead of a key
//!   that is pressed and never let go.
//! - `0xE0 0x2A` / `0xE0 0x12` and their right-shift twins are the "fake
//!   shifts" a keyboard wraps around its navigation keys; they decode as
//!   nothing.
//!
//! # Replies share the byte stream
//!
//! The keyboard answers a command (the LED update, the only one sent after
//! bring-up) on the same port its keys arrive on. In set 2 the reply bytes
//! (`FA`, `FE`, `AA`, `00`/`FF`) are not key codes and are recognised as
//! such. In set 1 several of them *are* also break codes (`AA` releases
//! Left Shift), so the decoder follows Linux's translated-mode rule: such a
//! byte is a break only while its make is held, and a reply otherwise.

use crate::scancode;

/// Which scancode set the controller is currently handing up — read from
/// the device at bring-up, never assumed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScancodeSet {
    /// XT-derived codes: break codes are the make code with bit 7 set.
    Set1,
    /// AT codes: a break is announced by a leading `0xF0`.
    Set2,
}

/// A key, by its identity rather than its wire encoding.
///
/// A scancode is a wire encoding, and letting one leak into the input
/// service would make the service's meaning depend on which set the
/// controller happened to be in; a HID usage is the key itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Key {
    /// A key of the HID Keyboard/Keypad page, by usage
    /// ([`kore_abi::keyboard::usage`]).
    Usage(u8),
    /// A code no table names — carried through as its raw code so a
    /// consumer can count it, and so an unknown key is *observable* rather
    /// than silently dropped.
    Unknown {
        /// The code arrived behind `0xE0`.
        extended: bool,
        /// The make code (break bit stripped in set 1).
        code: u8,
    },
}

/// A decoded key action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyEvent {
    /// A key went down.
    Press(Key),
    /// A key came up.
    Release(Key),
    /// A key that sends no break code (Hanja, Hangeul): pressed and
    /// released in the same instant.
    Tap(Key),
}

/// The keyboard's answer to a host command, or its unsolicited status.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reply {
    /// `0xFA`: the last byte written was accepted.
    Ack,
    /// `0xFE`: the last byte written was garbled; send it again.
    Resend,
    /// `0xAA`: the keyboard completed its basic assurance test — it was
    /// reset (or plugged in), so every key it held is up and its
    /// indicators are dark.
    SelfTestPassed,
    /// `0x00` / `0xFF`: key detection error or internal buffer overrun —
    /// keystrokes were lost inside the keyboard.
    Error,
}

/// One byte's worth of meaning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decoded {
    /// A key changed state.
    Key(KeyEvent),
    /// The device replied.
    Reply(Reply),
}

/// Set-1 bytes that are a protocol reply unless they are the break of a
/// held key, with the make code each would otherwise release (Linux's
/// `xl_table`). Indexed by bit in [`Decoder::xl_held`].
const XL: [u8; 6] = [0xAA, 0xFF, 0xFA, 0xFE, 0xF1, 0xF2];

/// Incremental decoder state: scancodes arrive one byte per interrupt, so
/// the decoder is a small state machine rather than a function over a
/// buffer.
#[derive(Clone, Copy, Debug)]
pub struct Decoder {
    set: ScancodeSet,
    /// Pending escape: 1 after `0xE0` (the next code is extended), 2 after
    /// `0xE1` (the next code is inert, the one after that extended).
    escape: u8,
    /// A `0xF0` break announcement is pending (set 2 only).
    break_pending: bool,
    /// Set 1: which [`XL`] bytes would currently be a break (their make is
    /// held), one bit per entry.
    xl_held: u8,
}

impl Decoder {
    /// A decoder for the set the controller reported.
    pub const fn new(set: ScancodeSet) -> Self {
        Self {
            set,
            escape: 0,
            break_pending: false,
            xl_held: 0,
        }
    }

    /// The set this decoder was told to expect.
    pub const fn set(&self) -> ScancodeSet {
        self.set
    }

    /// Feed one byte. `None` means the byte was a prefix, part of a
    /// multi-byte sequence, or a code that decodes as nothing (a fake
    /// shift, the inert half of an `0xE1` sequence).
    pub fn feed(&mut self, byte: u8) -> Option<Decoded> {
        match self.set {
            ScancodeSet::Set1 => self.feed_set1(byte),
            ScancodeSet::Set2 => self.feed_set2(byte),
        }
    }

    fn feed_set2(&mut self, byte: u8) -> Option<Decoded> {
        match byte {
            0xE0 => {
                self.escape = 1;
                return None;
            }
            0xE1 => {
                self.escape = 2;
                return None;
            }
            0xF0 => {
                self.break_pending = true;
                return None;
            }
            // Not key codes in set 2: the device talking about itself.
            0xFA => return self.reply(Reply::Ack),
            0xFE => return self.reply(Reply::Resend),
            0xAA => return self.reply(Reply::SelfTestPassed),
            0x00 | 0xFF => return self.reply(Reply::Error),
            _ => {}
        }
        if let Some(tap) = tap(byte) {
            self.escape = 0;
            self.break_pending = false;
            return Some(Decoded::Key(KeyEvent::Tap(tap)));
        }
        let released = core::mem::replace(&mut self.break_pending, false);
        let extended = self.take_escape()?;
        let key = if extended {
            if scancode::SET2_EXTENDED_IGNORED.contains(&byte) {
                return None;
            }
            lookup(&scancode::SET2_EXTENDED, true, byte)
        } else {
            lookup(&scancode::SET2, false, byte)
        };
        Some(Decoded::Key(event(key, released)))
    }

    fn feed_set1(&mut self, byte: u8) -> Option<Decoded> {
        if byte == 0xE0 {
            self.escape = 1;
            return None;
        }
        if byte == 0xE1 {
            self.escape = 2;
            return None;
        }
        // Linux's translated-mode rule: a byte that collides with a reply
        // is a break only while its make is held — and only outside an
        // escape, where every byte is a code.
        let xl = XL.iter().position(|&x| x == byte);
        let as_reply = self.escape == 0 && xl.is_some_and(|i| self.xl_held & (1 << i) == 0);
        if self.escape == 0 {
            if let Some(i) = XL.iter().position(|&x| (x ^ byte) & 0x7F == 0) {
                if byte & 0x80 != 0 {
                    self.xl_held &= !(1 << i);
                } else {
                    self.xl_held |= 1 << i;
                }
            }
        }
        if as_reply {
            return match byte {
                0xFA => self.reply(Reply::Ack),
                0xFE => self.reply(Reply::Resend),
                0xAA => self.reply(Reply::SelfTestPassed),
                0xFF => self.reply(Reply::Error),
                // 0xF1 / 0xF2: Hanja / Hangeul, which send no break.
                other => tap(other).map(|key| Decoded::Key(KeyEvent::Tap(key))),
            };
        }
        let released = byte & 0x80 != 0;
        let code = byte & 0x7F;
        let extended = self.take_escape()?;
        let key = if extended {
            if scancode::SET1_EXTENDED_IGNORED.contains(&code) {
                return None;
            }
            lookup(&scancode::SET1_EXTENDED, true, code)
        } else {
            lookup(&scancode::SET1, false, code)
        };
        Some(Decoded::Key(event(key, released)))
    }

    /// Consume the pending escape for one code: `Some(extended)`, or
    /// `None` for the inert first code of an `0xE1` sequence.
    fn take_escape(&mut self) -> Option<bool> {
        match self.escape {
            0 => Some(false),
            1 => {
                self.escape = 0;
                Some(true)
            }
            _ => {
                self.escape = 1;
                None
            }
        }
    }

    /// A reply ends any sequence in flight: the keyboard does not
    /// interleave a reply inside a key's bytes, so a pending escape or
    /// break announcement is stale.
    fn reply(&mut self, reply: Reply) -> Option<Decoded> {
        self.escape = 0;
        self.break_pending = false;
        if reply == Reply::SelfTestPassed {
            // The keyboard reset: every key is up, so no reply-colliding
            // byte can be a break until its make is seen again.
            self.xl_held = 0;
        }
        Some(Decoded::Reply(reply))
    }
}

fn tap(byte: u8) -> Option<Key> {
    scancode::TAPS
        .iter()
        .find(|(code, _)| *code == byte)
        .map(|(_, usage)| Key::Usage(*usage))
}

fn lookup(table: &[u8], extended: bool, code: u8) -> Key {
    match table.get(usize::from(code)).copied().unwrap_or(0) {
        0 => Key::Unknown { extended, code },
        usage => Key::Usage(usage),
    }
}

const fn event(key: Key, released: bool) -> KeyEvent {
    if released {
        KeyEvent::Release(key)
    } else {
        KeyEvent::Press(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kore_abi::keyboard::usage;

    fn feed_all(d: &mut Decoder, bytes: &[u8]) -> alloc::vec::Vec<Decoded> {
        bytes.iter().filter_map(|b| d.feed(*b)).collect()
    }

    fn press(key: u8) -> Decoded {
        Decoded::Key(KeyEvent::Press(Key::Usage(key)))
    }

    fn release(key: u8) -> Decoded {
        Decoded::Key(KeyEvent::Release(Key::Usage(key)))
    }

    /// The typing path both sets must agree on: the SAME key produces the
    /// same event from two different wire encodings. This is the property
    /// that matters — a consumer must never be able to tell which set the
    /// controller was in.
    #[test]
    fn both_sets_decode_the_same_key_identically() {
        let mut s1 = Decoder::new(ScancodeSet::Set1);
        let mut s2 = Decoder::new(ScancodeSet::Set2);
        assert_eq!(feed_all(&mut s1, &[0x1e, 0x9e]), [press(usage::A), release(usage::A)]);
        assert_eq!(
            feed_all(&mut s2, &[0x1c, 0xf0, 0x1c]),
            [press(usage::A), release(usage::A)]
        );
    }

    /// Every set-1 code decodes to what its set-2 twin decodes to, across
    /// the whole table — the property the generated tables are built for.
    #[test]
    fn set1_is_set2_through_the_translation_for_every_named_key() {
        // The controller's translation, set 2 → set 1, for a sample that
        // spans every region of the table (main block, F-keys, keypad,
        // the locks, the extended block).
        for (set2, set1, extended) in [
            (0x58u8, 0x3au8, false), // Caps Lock
            (0x77, 0x45, false),     // Num Lock
            (0x7e, 0x46, false),     // Scroll Lock
            (0x05, 0x3b, false),     // F1
            (0x07, 0x58, false),     // F12
            (0x6c, 0x47, false),     // KP 7
            (0x7c, 0x37, false),     // KP *
            (0x5a, 0x1c, true),      // KP Enter
            (0x4a, 0x35, true),      // KP /
            (0x1f, 0x5b, true),      // Left GUI
            (0x2f, 0x5d, true),      // Application
            (0x7d, 0x49, true),      // Page Up
            (0x70, 0x52, true),      // Insert
        ] {
            let mut s1 = Decoder::new(ScancodeSet::Set1);
            let mut s2 = Decoder::new(ScancodeSet::Set2);
            let wire1: &[u8] = if extended { &[0xe0, set1] } else { &[set1] };
            let wire2: &[u8] = if extended { &[0xe0, set2] } else { &[set2] };
            let a = feed_all(&mut s1, wire1);
            let b = feed_all(&mut s2, wire2);
            assert_eq!(a, b, "set-1 {set1:#04x} vs set-2 {set2:#04x}");
            assert!(matches!(a[..], [Decoded::Key(KeyEvent::Press(Key::Usage(_)))]));
        }
    }

    /// The keys this decoder used to drop: function keys, the navigation
    /// block, the keypad with its own identity, the locks, the GUI keys.
    #[test]
    fn the_whole_pc_keyboard_has_an_identity() {
        let mut d = Decoder::new(ScancodeSet::Set2);
        for (wire, key) in [
            (&[0x58][..], usage::CAPS_LOCK),
            (&[0x77], usage::NUM_LOCK),
            (&[0x7e], usage::SCROLL_LOCK),
            (&[0x05], usage::F1),
            (&[0x78], usage::F11),
            (&[0x07], usage::F12),
            (&[0x83], usage::F7),
            (&[0xe0, 0x70], usage::INSERT),
            (&[0xe0, 0x7d], usage::PAGE_UP),
            (&[0xe0, 0x7a], usage::PAGE_DOWN),
            (&[0xe0, 0x4a], usage::KP_DIVIDE),
            (&[0xe0, 0x5a], usage::KP_ENTER),
            (&[0x69], usage::KP_1),
            (&[0x71], usage::KP_PERIOD),
            (&[0x7c], usage::KP_MULTIPLY),
            (&[0x7b], usage::KP_SUBTRACT),
            (&[0x79], usage::KP_ADD),
            (&[0xe0, 0x1f], usage::LEFT_GUI),
            (&[0xe0, 0x27], usage::RIGHT_GUI),
            (&[0xe0, 0x2f], usage::APPLICATION),
            (&[0x61], usage::NON_US_BACKSLASH),
            (&[0xe0, 0x23], usage::MUTE),
        ] {
            assert_eq!(feed_all(&mut d, wire), [press(key)], "{wire:02x?}");
        }
    }

    /// `0xE0` makes a *different* key out of the same code — the reason
    /// the prefix cannot be dropped. Set-1 `0x48` is keypad 8 plain and Up
    /// extended; both must decode correctly in sequence, so the prefix
    /// state cannot leak into the next key.
    #[test]
    fn the_extended_prefix_selects_a_different_key_and_does_not_leak() {
        let mut d = Decoder::new(ScancodeSet::Set1);
        assert_eq!(
            feed_all(&mut d, &[0x48, 0xe0, 0x48, 0x48]),
            [press(usage::KP_8), press(usage::UP), press(usage::KP_8)],
            "the prefix must apply to exactly one code"
        );
    }

    /// An extended RELEASE: prefix, then a code with the break bit — both
    /// escapes at once, which is where a naive decoder reports a press.
    #[test]
    fn an_extended_release_is_a_release() {
        let mut d = Decoder::new(ScancodeSet::Set1);
        assert_eq!(feed_all(&mut d, &[0xe0, 0xc8]), [release(usage::UP)]);

        let mut d = Decoder::new(ScancodeSet::Set2);
        assert_eq!(feed_all(&mut d, &[0xe0, 0xf0, 0x75]), [release(usage::UP)]);
    }

    /// Pause's make and break are both in its one sequence: it decodes as
    /// a press then a release — never a key held forever — and its inert
    /// bytes (which alias Ctrl and Num Lock) invent nothing.
    #[test]
    fn pause_is_a_press_and_a_release() {
        let mut d = Decoder::new(ScancodeSet::Set1);
        assert_eq!(
            feed_all(&mut d, &[0xe1, 0x1d, 0x45, 0xe1, 0x9d, 0xc5]),
            [press(usage::PAUSE), release(usage::PAUSE)],
        );
        assert_eq!(feed_all(&mut d, &[0x1e]), [press(usage::A)], "clean afterwards");

        let mut d = Decoder::new(ScancodeSet::Set2);
        assert_eq!(
            feed_all(&mut d, &[0xe1, 0x14, 0x77, 0xe1, 0xf0, 0x14, 0xf0, 0x77]),
            [press(usage::PAUSE), release(usage::PAUSE)],
        );
        assert_eq!(feed_all(&mut d, &[0x1c]), [press(usage::A)], "clean afterwards");
        // Ctrl+Pause (Break) is `E0 7E` in set 2 and `E0 46` in set 1.
        assert_eq!(
            feed_all(&mut d, &[0xe0, 0x7e, 0xe0, 0xf0, 0x7e]),
            [press(usage::PAUSE), release(usage::PAUSE)]
        );
    }

    /// PrintScreen arrives wrapped in a fake shift; only the key survives.
    /// Alt+PrintScreen (SysRq) is the same key.
    #[test]
    fn print_screen_sheds_its_fake_shift() {
        let mut d = Decoder::new(ScancodeSet::Set2);
        assert_eq!(
            feed_all(&mut d, &[0xe0, 0x12, 0xe0, 0x7c, 0xe0, 0xf0, 0x7c, 0xe0, 0xf0, 0x12]),
            [press(usage::PRINT_SCREEN), release(usage::PRINT_SCREEN)]
        );
        assert_eq!(feed_all(&mut d, &[0x84]), [press(usage::PRINT_SCREEN)]);

        let mut d = Decoder::new(ScancodeSet::Set1);
        assert_eq!(
            feed_all(&mut d, &[0xe0, 0x2a, 0xe0, 0x37, 0xe0, 0xb7, 0xe0, 0xaa]),
            [press(usage::PRINT_SCREEN), release(usage::PRINT_SCREEN)]
        );
        // The fake shift's break is not a Left Shift release.
        assert_eq!(feed_all(&mut d, &[0x54]), [press(usage::PRINT_SCREEN)]);
    }

    /// Hanja and Hangeul send one byte and no break: a tap.
    #[test]
    fn keys_without_a_break_are_taps() {
        for set in [ScancodeSet::Set1, ScancodeSet::Set2] {
            let mut d = Decoder::new(set);
            assert_eq!(
                feed_all(&mut d, &[0xf2, 0xf1]),
                [
                    Decoded::Key(KeyEvent::Tap(Key::Usage(usage::LANG_1))),
                    Decoded::Key(KeyEvent::Tap(Key::Usage(usage::LANG_2))),
                ]
            );
        }
    }

    /// The keyboard's replies are not keys. In set 2 they are out of the
    /// code range; in set 1 they double as breaks, and are a reply only
    /// when the make they would release is not held.
    #[test]
    fn replies_are_told_apart_from_breaks() {
        let mut d = Decoder::new(ScancodeSet::Set2);
        assert_eq!(
            feed_all(&mut d, &[0xfa, 0xfe, 0xaa, 0x00, 0xff]),
            [
                Decoded::Reply(Reply::Ack),
                Decoded::Reply(Reply::Resend),
                Decoded::Reply(Reply::SelfTestPassed),
                Decoded::Reply(Reply::Error),
                Decoded::Reply(Reply::Error),
            ]
        );

        let mut d = Decoder::new(ScancodeSet::Set1);
        // `AA` with Left Shift up is the self-test reply...
        assert_eq!(feed_all(&mut d, &[0xaa]), [Decoded::Reply(Reply::SelfTestPassed)]);
        // ...and with Left Shift down, its release.
        assert_eq!(
            feed_all(&mut d, &[0x2a, 0xaa]),
            [press(usage::LEFT_SHIFT), release(usage::LEFT_SHIFT)]
        );
        assert_eq!(feed_all(&mut d, &[0xfa]), [Decoded::Reply(Reply::Ack)]);
        assert_eq!(feed_all(&mut d, &[0xfe]), [Decoded::Reply(Reply::Resend)]);
        // Keypad comma's make is 0x7E; while it is held, FE is its break.
        assert_eq!(
            feed_all(&mut d, &[0x7e, 0xfe]),
            [press(usage::KP_COMMA), release(usage::KP_COMMA)]
        );
        // A reset releases every key: a make seen before it no longer turns
        // the colliding reply into that key's break.
        assert_eq!(
            feed_all(&mut d, &[0x7e, 0xaa, 0xfe]),
            [
                press(usage::KP_COMMA),
                Decoded::Reply(Reply::SelfTestPassed),
                Decoded::Reply(Reply::Resend),
            ]
        );
    }

    /// A reply cancels a half-received sequence: the next key is clean.
    #[test]
    fn a_reply_does_not_leave_a_stale_prefix() {
        let mut d = Decoder::new(ScancodeSet::Set2);
        assert_eq!(
            feed_all(&mut d, &[0xe0, 0xfa, 0x1c]),
            [Decoded::Reply(Reply::Ack), press(usage::A)]
        );
        let mut d = Decoder::new(ScancodeSet::Set2);
        assert_eq!(
            feed_all(&mut d, &[0xf0, 0xfa, 0x1c]),
            [Decoded::Reply(Reply::Ack), press(usage::A)]
        );
    }

    /// An unnamed key is carried through as its code rather than dropped,
    /// so a missing table entry is observable instead of silent.
    #[test]
    fn an_unknown_code_is_reported_not_dropped() {
        let mut d = Decoder::new(ScancodeSet::Set2);
        assert_eq!(
            feed_all(&mut d, &[0xe0, 0x3f]),
            [Decoded::Key(KeyEvent::Press(Key::Unknown {
                extended: true,
                code: 0x3f
            }))],
            "set-2 E0 3F is Sleep, a Generic Desktop key, not a keyboard-page one"
        );
    }

    /// Modifiers decode as themselves in both sets — the terminal's line
    /// discipline needs them distinguishable from printable keys.
    #[test]
    fn modifiers_are_named_in_both_sets() {
        let mut s1 = Decoder::new(ScancodeSet::Set1);
        assert_eq!(feed_all(&mut s1, &[0x2a]), [press(usage::LEFT_SHIFT)]);
        assert_eq!(feed_all(&mut s1, &[0xaa]), [release(usage::LEFT_SHIFT)]);
        assert_eq!(feed_all(&mut s1, &[0xe0, 0x38]), [press(usage::RIGHT_ALT)]);

        let mut s2 = Decoder::new(ScancodeSet::Set2);
        assert_eq!(feed_all(&mut s2, &[0x12]), [press(usage::LEFT_SHIFT)]);
        assert_eq!(feed_all(&mut s2, &[0xf0, 0x12]), [release(usage::LEFT_SHIFT)]);
        assert_eq!(feed_all(&mut s2, &[0xe0, 0x14]), [press(usage::RIGHT_CTRL)]);
    }
}
