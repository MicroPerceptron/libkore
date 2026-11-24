//! Input-report framing — the two-byte length prefix, and the three ways
//! it is not a report.
//!
//! Every read from the input register begins with a little-endian length
//! **that includes the prefix itself**. Three values are not data:
//!
//! - `0x0000` — the device has nothing to say, or is acknowledging a
//!   reset. Which of the two it is depends on driver state, not on the
//!   bytes, so this layer reports the shape and lets the bring-up state
//!   machine decide.
//! - `0xFFFF` — seen from devices that have lost power or are not driving
//!   the bus. Discarded rather than treated as an enormous report.
//! - anything above the descriptor's `wMaxInputLength` — the device has
//!   contradicted its own descriptor, and trusting it would read past the
//!   buffer that descriptor sized.

/// What a read from the input register turned out to be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Frame<'a> {
    /// A report, with the length prefix already stripped.
    Report(&'a [u8]),
    /// A zero-length frame: an empty poll, or a reset acknowledgement.
    Empty,
    /// The frame was refused.
    Refused(FramingError),
}

/// Why a frame was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FramingError {
    /// Fewer than the two prefix bytes arrived.
    Truncated {
        /// How many bytes arrived.
        got: usize,
    },
    /// The all-ones length a powered-down device produces.
    Bogus,
    /// The device claimed a length above its own `wMaxInputLength`.
    TooLong {
        /// What it claimed.
        claimed: u16,
        /// What its descriptor promised.
        max: u16,
    },
    /// The claimed length exceeded what was actually read — a truncated
    /// transfer, not a malformed device.
    Incomplete {
        /// What it claimed.
        claimed: u16,
        /// How many bytes are in hand.
        got: usize,
    },
    /// A length of one: the prefix says the frame is smaller than the
    /// prefix, which cannot be true.
    Impossible(u16),
}

/// The sentinel a device that is not driving the bus produces.
pub const BOGUS_LENGTH: u16 = 0xFFFF;

/// Split a raw input-register read into its frame.
pub fn decode(raw: &[u8], max_input_length: u16) -> Frame<'_> {
    if raw.len() < 2 {
        return Frame::Refused(FramingError::Truncated { got: raw.len() });
    }
    let claimed = u16::from_le_bytes([raw[0], raw[1]]);

    if claimed == 0 {
        return Frame::Empty;
    }
    if claimed == BOGUS_LENGTH {
        return Frame::Refused(FramingError::Bogus);
    }
    if claimed == 1 {
        return Frame::Refused(FramingError::Impossible(claimed));
    }
    if claimed > max_input_length {
        return Frame::Refused(FramingError::TooLong {
            claimed,
            max: max_input_length,
        });
    }
    if claimed as usize > raw.len() {
        return Frame::Refused(FramingError::Incomplete {
            claimed,
            got: raw.len(),
        });
    }
    Frame::Report(&raw[2..claimed as usize])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_report_has_its_prefix_stripped_and_its_length_honoured() {
        // Length 6 = prefix + 4 payload bytes. Trailing bytes beyond the
        // claimed length are not part of the report, even though they
        // were read.
        let raw = [6, 0, 0xAA, 0xBB, 0xCC, 0xDD, 0x99, 0x99];
        assert_eq!(decode(&raw, 64), Frame::Report(&[0xAA, 0xBB, 0xCC, 0xDD]));
    }

    #[test]
    fn a_report_exactly_filling_the_buffer_is_accepted() {
        let raw = [4, 0, 1, 2];
        assert_eq!(decode(&raw, 64), Frame::Report(&[1, 2]));
        // And one exactly at the descriptor's maximum.
        let raw = [4, 0, 1, 2];
        assert_eq!(decode(&raw, 4), Frame::Report(&[1, 2]));
    }

    #[test]
    fn a_zero_length_frame_is_empty_not_an_error() {
        // The same bytes mean "nothing to report" during steady state and
        // "reset complete" during bring-up. This layer must not guess.
        assert_eq!(decode(&[0, 0], 64), Frame::Empty);
        assert_eq!(decode(&[0, 0, 0xff, 0xff], 64), Frame::Empty);
    }

    #[test]
    fn the_powered_down_sentinel_is_discarded() {
        assert_eq!(
            decode(&[0xff, 0xff, 1, 2], 64),
            Frame::Refused(FramingError::Bogus)
        );
    }

    #[test]
    fn a_device_that_contradicts_its_own_descriptor_is_refused() {
        assert_eq!(
            decode(&[100, 0, 1, 2], 64),
            Frame::Refused(FramingError::TooLong {
                claimed: 100,
                max: 64
            })
        );
    }

    #[test]
    fn a_length_of_one_is_impossible_since_the_prefix_is_two() {
        assert_eq!(
            decode(&[1, 0], 64),
            Frame::Refused(FramingError::Impossible(1))
        );
    }

    #[test]
    fn a_truncated_transfer_is_distinguished_from_a_lying_device() {
        // Claimed 8, only 4 read: the bus fell short, the device did not
        // misbehave. Conflating the two would blame the wrong component.
        assert_eq!(
            decode(&[8, 0, 1, 2], 64),
            Frame::Refused(FramingError::Incomplete { claimed: 8, got: 4 })
        );
        for n in 0..2 {
            assert_eq!(
                decode(&[0u8; 2][..n], 64),
                Frame::Refused(FramingError::Truncated { got: n })
            );
        }
    }
}
