//! [`Key`] → **HID boot report contribution**, so a PS/2 keystroke becomes
//! the same report a USB one does.
//!
//! The input service's data plane speaks HID boot-protocol reports
//! (`InputReport`: a modifier bitmap plus up to six held usage codes)
//! because that is what the xHCI source produces and what every keyboard
//! consumer decodes. A second source that spoke its own vocabulary would
//! fork those consumers in two; translating here keeps one decoder
//! downstream and makes the keyboards genuinely interchangeable rather than
//! merely coexistent.
//!
//! The translation belongs to the **driver**, not the service: which usage
//! a key carries is a fact about keyboards, and the service's job is
//! routing. The decoder already names keys by usage ([`crate::scancode`]),
//! so all that is left here is the boot report's own split: the eight
//! modifiers ride the bitmap, everything else the six-key array.

use kore_abi::keyboard::modifier;

use crate::decode::Key;

/// What one [`Key`] contributes to a HID boot report.
///
/// The split is the boot report's own: modifiers live in the bitmap and
/// are never members of the six-key array, ordinary keys are the
/// reverse, and a key with no HID meaning contributes nothing at all
/// (which the caller still *counts* — see [`Key::Unknown`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Contribution {
    /// A modifier bit in the report's `state` byte.
    Modifier(u8),
    /// An ordinary key, as its usage code in the six-key array.
    Usage(u8),
    /// No HID meaning — carried no further.
    None,
}

impl Key {
    /// What this key contributes to a HID boot report.
    pub const fn hid(self) -> Contribution {
        match self {
            Key::Usage(usage) => match modifier::bit(usage) {
                Some(bit) => Contribution::Modifier(bit),
                None => Contribution::Usage(usage),
            },
            // A key the tables do not name has no usage code to invent.
            // It is still not *lost*: the decoder produced it, and the
            // relay counts it, so an unmapped key on real hardware is a
            // number in a status row rather than silence.
            Key::Unknown { .. } => Contribution::None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kore_abi::keyboard::usage;

    #[test]
    fn modifiers_ride_the_bitmap_not_the_key_array() {
        assert_eq!(
            Key::Usage(usage::LEFT_SHIFT).hid(),
            Contribution::Modifier(modifier::LEFT_SHIFT)
        );
        assert_eq!(
            Key::Usage(usage::RIGHT_ALT).hid(),
            Contribution::Modifier(modifier::RIGHT_ALT)
        );
        // The GUI keys are modifiers too — the bitmap has had their bits
        // all along; the PS/2 path simply never produced them.
        assert_eq!(
            Key::Usage(usage::LEFT_GUI).hid(),
            Contribution::Modifier(modifier::LEFT_GUI)
        );
        // Left and right are distinct bits: a terminal that treats them
        // as one still works, one that distinguishes them can.
        assert_ne!(
            Key::Usage(usage::LEFT_CTRL).hid(),
            Key::Usage(usage::RIGHT_CTRL).hid()
        );
    }

    #[test]
    fn ordinary_keys_occupy_the_array_under_their_own_usage() {
        assert_eq!(Key::Usage(usage::A).hid(), Contribution::Usage(usage::A));
        assert_eq!(
            Key::Usage(usage::CAPS_LOCK).hid(),
            Contribution::Usage(usage::CAPS_LOCK)
        );
        assert_eq!(Key::Usage(usage::KP_ENTER).hid(), Contribution::Usage(usage::KP_ENTER));
    }

    #[test]
    fn an_unnamed_key_contributes_nothing_but_is_still_a_key() {
        // The decoder refuses to drop it; the mapping refuses to invent
        // a usage for it.  Both are true at once, which is the point.
        assert_eq!(
            Key::Unknown {
                extended: true,
                code: 0x3f
            }
            .hid(),
            Contribution::None
        );
    }
}
