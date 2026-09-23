//! Authenticated updates to one still-active frontend intent. These records
//! carry user text and result correlations, never native authority selectors.
use crate::{FRONTEND_TEXT_MAX, FrontendIntentId};
use zerocopy::byteorder::little_endian::{U16, U32};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

const UPDATE: u32 = u32::from_le_bytes(*b"KFU1");
const RESULT: u32 = u32::from_le_bytes(*b"KFA1");
pub const FRONTEND_UPDATE_HEADER_LEN: usize = 32;

#[derive(Clone, Copy, FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned)]
#[repr(C)]
struct Header {
    magic: U32,
    version: U16,
    kind: u8,
    reserved: u8,
    intent: [u8; 16],
    revision: U32,
    length: U16,
    reserved2: [u8; 2],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrontendUpdate<'a> {
    pub intent_id: FrontendIntentId,
    pub revision: u32,
    pub text: &'a str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum FrontendUpdateOutcome {
    Applied = 1,
    Refused = 2,
    Failed = 3,
    Cancelled = 4,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrontendUpdateResult<'a> {
    pub intent_id: FrontendIntentId,
    pub revision: u32,
    pub outcome: FrontendUpdateOutcome,
    pub text: &'a str,
}

impl FrontendUpdate<'_> {
    pub fn encode(self, output: &mut [u8]) -> Option<usize> {
        if self.text.is_empty() {
            return None;
        }
        encode(UPDATE, 0, self.intent_id, self.revision, self.text, output)
    }
}

impl FrontendUpdateResult<'_> {
    pub fn encode(self, output: &mut [u8]) -> Option<usize> {
        encode(
            RESULT,
            self.outcome as u8,
            self.intent_id,
            self.revision,
            self.text,
            output,
        )
    }
}

fn encode(
    magic: u32,
    kind: u8,
    intent_id: FrontendIntentId,
    revision: u32,
    text: &str,
    output: &mut [u8],
) -> Option<usize> {
    if intent_id.is_zero()
        || revision == 0
        || text.len() > FRONTEND_TEXT_MAX
        || text.as_bytes().contains(&0)
    {
        return None;
    }
    let len = FRONTEND_UPDATE_HEADER_LEN + text.len();
    let output = output.get_mut(..len)?;
    output[..FRONTEND_UPDATE_HEADER_LEN].copy_from_slice(
        Header {
            magic: U32::new(magic),
            version: U16::new(1),
            kind,
            reserved: 0,
            intent: intent_id.0,
            revision: U32::new(revision),
            length: U16::new(text.len() as u16),
            reserved2: [0; 2],
        }
        .as_bytes(),
    );
    output[FRONTEND_UPDATE_HEADER_LEN..].copy_from_slice(text.as_bytes());
    Some(len)
}

fn decode(bytes: &[u8], magic: u32) -> Option<(Header, &str)> {
    let (header, text) = Header::read_from_prefix(bytes).ok()?;
    if header.magic.get() != magic
        || header.version.get() != 1
        || header.reserved != 0
        || header.reserved2 != [0; 2]
        || FrontendIntentId(header.intent).is_zero()
        || header.revision.get() == 0
        || usize::from(header.length.get()) != text.len()
        || text.len() > FRONTEND_TEXT_MAX
        || text.contains(&0)
    {
        return None;
    }
    Some((header, core::str::from_utf8(text).ok()?))
}

pub fn decode_frontend_update(bytes: &[u8]) -> Option<FrontendUpdate<'_>> {
    let (header, text) = decode(bytes, UPDATE)?;
    (header.kind == 0 && !text.is_empty()).then_some(FrontendUpdate {
        intent_id: FrontendIntentId(header.intent),
        revision: header.revision.get(),
        text,
    })
}

pub fn decode_frontend_update_result(bytes: &[u8]) -> Option<FrontendUpdateResult<'_>> {
    let (header, text) = decode(bytes, RESULT)?;
    let outcome = match header.kind {
        1 => FrontendUpdateOutcome::Applied,
        2 => FrontendUpdateOutcome::Refused,
        3 => FrontendUpdateOutcome::Failed,
        4 => FrontendUpdateOutcome::Cancelled,
        _ => return None,
    };
    Some(FrontendUpdateResult {
        intent_id: FrontendIntentId(header.intent),
        revision: header.revision.get(),
        outcome,
        text,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn frontend_update_records_are_exact_bounded_and_distinct_from_prompt_results() {
        let request = FrontendUpdate {
            intent_id: FrontendIntentId([1; 16]),
            revision: 9,
            text: "Revoke the input view.",
        };
        let mut bytes = [0; crate::INFERENCE_PART_MAX + 1];
        let len = request.encode(&mut bytes).unwrap();
        assert_eq!(decode_frontend_update(&bytes[..len]), Some(request));
        assert!(crate::decode_frontend_intent(&bytes[..len]).is_none());
        assert!(crate::decode_frontend_result(&bytes[..len]).is_none());
        assert!(decode_frontend_update_result(&bytes[..len]).is_none());
        for end in 0..len {
            assert!(decode_frontend_update(&bytes[..end]).is_none());
        }
        assert!(decode_frontend_update(&bytes[..len + 1]).is_none());
        for offset in [4, 6, 7, 24, 30, 31] {
            let mut malformed = bytes;
            if offset == 24 {
                malformed[24..28].fill(0);
            } else {
                malformed[offset] = 99;
            }
            assert!(decode_frontend_update(&malformed[..len]).is_none());
        }
        for outcome in [
            FrontendUpdateOutcome::Applied,
            FrontendUpdateOutcome::Refused,
            FrontendUpdateOutcome::Failed,
            FrontendUpdateOutcome::Cancelled,
        ] {
            let result = FrontendUpdateResult {
                intent_id: request.intent_id,
                revision: request.revision,
                outcome,
                text: "",
            };
            let len = result.encode(&mut bytes).unwrap();
            assert_eq!(decode_frontend_update_result(&bytes[..len]), Some(result));
            assert!(decode_frontend_update(&bytes[..len]).is_none());
            assert!(crate::decode_frontend_result(&bytes[..len]).is_none());
        }
        for invalid in [
            FrontendUpdate {
                revision: 0,
                ..request
            },
            FrontendUpdate {
                intent_id: FrontendIntentId::ZERO,
                ..request
            },
            FrontendUpdate {
                text: "",
                ..request
            },
            FrontendUpdate {
                text: "x\0y",
                ..request
            },
        ] {
            assert!(invalid.encode(&mut bytes).is_none());
        }
        let text = [b'x'; FRONTEND_TEXT_MAX + 1];
        let max = FrontendUpdate {
            text: core::str::from_utf8(&text[..FRONTEND_TEXT_MAX]).unwrap(),
            ..request
        };
        assert_eq!(max.encode(&mut bytes), Some(crate::INFERENCE_PART_MAX));
        assert!(
            FrontendUpdate {
                text: core::str::from_utf8(&text).unwrap(),
                ..request
            }
            .encode(&mut bytes)
            .is_none()
        );
    }
}
