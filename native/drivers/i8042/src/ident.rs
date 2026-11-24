//! Who this driver is for: the ACPI `_HID` a firmware-described i8042
//! keyboard controller carries.
//!
//! A driver row names the device it wants by `_HID`, and the value is a
//! *compressed EISA identifier* — three letters and four hex digits
//! packed into a little-endian dword. Deriving it beats writing the
//! constant down: `0x0303D041` is four bytes with no visible relation to
//! `"PNP0303"`, so a transposition in a hand-written literal reads
//! exactly like a correct one. (That is not hypothetical — a transposed
//! digit in a *test's expectation* for this same encoder was caught by
//! the encoder disagreeing with it.)
//!
//! ## Why this is a second copy
//!
//! The kernel has its own encoder for the same encoding, and that
//! duplication is deliberate rather than an oversight: the driver tree's
//! hardware encoders are kept independent of the kernel-side copies (the
//! standing ruling that closed the SMMUv3 descriptor de-duplication as
//! not-planned). A driver crate that reached into kernel internals for a
//! spec constant would couple the two across a boundary that exists
//! precisely to be crossable only by capabilities. Both copies are
//! pinned against the same published encodings by their own tests.

/// Encode a `"LLLdddd"` ACPI `_HID` string as its compressed EISA dword.
///
/// `None` for anything that is not exactly three uppercase letters
/// followed by four hex digits — the encoding cannot represent it, and
/// guessing would produce a value that matches some *other* device.
pub const fn eisa_id(id: &str) -> Option<u32> {
    let b = id.as_bytes();
    if b.len() != 7 {
        return None;
    }
    // Letters: 5 bits each, 'A' == 1.
    let mut letters = [0u8; 3];
    let mut i = 0;
    while i < 3 {
        let c = b[i];
        if c < b'A' || c > b'Z' {
            return None;
        }
        letters[i] = c - 0x40;
        i += 1;
    }
    // Digits: four hex nibbles, most significant first.
    let mut nibbles = [0u8; 4];
    let mut j = 0;
    while j < 4 {
        let c = b[3 + j];
        nibbles[j] = match c {
            b'0'..=b'9' => c - b'0',
            b'A'..=b'F' => c - b'A' + 10,
            _ => return None,
        };
        j += 1;
    }
    // The packing: two bytes of letters (5+5+5 bits, high bit clear),
    // then the four nibbles as two bytes — read back as a LITTLE-endian
    // dword, which is why the byte order below looks reversed.
    let b0 = (letters[0] << 2) | (letters[1] >> 3);
    let b1 = ((letters[1] & 0x07) << 5) | letters[2];
    let b2 = (nibbles[0] << 4) | nibbles[1];
    let b3 = (nibbles[2] << 4) | nibbles[3];
    Some(u32::from_le_bytes([b0, b1, b2, b3]))
}

/// `PNP0303` — the PC/AT enhanced keyboard controller.
pub const HID_KEYBOARD: u32 = match eisa_id("PNP0303") {
    Some(id) => id,
    None => panic!("PNP0303 is a well-formed _HID"),
};

/// `PNP0F13` — the PS/2 mouse port on the same controller. Named for
/// the pointer seam; this driver decodes keyboards today.
pub const HID_MOUSE: u32 = match eisa_id("PNP0F13") {
    Some(id) => id,
    None => panic!("PNP0F13 is a well-formed _HID"),
};

#[cfg(test)]
mod tests {
    use super::*;

    /// Against the published values, not against the encoder's own
    /// output — the whole point of a transcription check.
    #[test]
    fn the_known_ids_encode_to_their_known_dwords() {
        assert_eq!(eisa_id("PNP0303"), Some(0x0303_d041), "PS/2 keyboard");
        assert_eq!(eisa_id("PNP0F13"), Some(0x130f_d041), "PS/2 mouse");
        assert_eq!(HID_KEYBOARD, 0x0303_d041);
    }

    #[test]
    fn a_malformed_id_encodes_to_nothing() {
        assert_eq!(eisa_id("PNP030"), None, "too short");
        assert_eq!(eisa_id("PNP03031"), None, "too long");
        assert_eq!(eisa_id("pnp0303"), None, "lowercase letters");
        assert_eq!(eisa_id("PNP030G"), None, "non-hex digit");
    }

    /// Two different devices must not collide — the failure a
    /// hand-written literal produces silently.
    #[test]
    fn distinct_ids_encode_distinctly() {
        assert_ne!(eisa_id("PNP0303"), eisa_id("PNP0304"));
        assert_ne!(eisa_id("PNP0303"), eisa_id("PNQ0303"));
    }
}
