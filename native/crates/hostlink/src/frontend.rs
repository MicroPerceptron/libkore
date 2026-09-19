//! Bounded authenticated-intent handoff between a host frontend and Kore.
//!
//! The host never pushes an unsolicited message into a guest route. The
//! resident broker first polls for the next intent, then the founded agent
//! claims the same intent by its host-issued opaque identifier. A result travels
//! back on that agent's generation-bound route.

use core::str;

use zerocopy::byteorder::little_endian::{U16, U32};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

use crate::INFERENCE_PART_MAX;

pub const FRONTEND_POLL_MAGIC: u32 = u32::from_le_bytes(*b"KFP1");
pub const FRONTEND_INTENT_MAGIC: u32 = u32::from_le_bytes(*b"KFI1");
pub const FRONTEND_RESULT_MAGIC: u32 = u32::from_le_bytes(*b"KFR1");
pub const FRONTEND_SCHEMA_V1: u16 = 1;
pub const FRONTEND_RECORD_HEADER_LEN: usize = 32;
pub const FRONTEND_TEXT_MAX: usize = INFERENCE_PART_MAX - FRONTEND_RECORD_HEADER_LEN;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct FrontendIntentId(pub [u8; 16]);

impl FrontendIntentId {
    pub const ZERO: Self = Self([0; 16]);

    #[must_use]
    pub const fn from_sequence(sequence: u128) -> Option<Self> {
        if sequence == 0 {
            None
        } else {
            Some(Self(sequence.to_le_bytes()))
        }
    }

    #[must_use]
    pub fn is_zero(self) -> bool {
        self.0.iter().all(|byte| *byte == 0)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum FrontendPollPhase {
    BrokerNext = 1,
    AgentClaim = 2,
    /// The claiming broker confirms that all native resources for this intent
    /// are retired (including a failed partial founding), or none existed.
    /// Sent after terminal publication and before returning to BrokerNext.
    BrokerRetired = 3,
    /// Query this intent's host control state on its claiming broker face.
    BrokerControl = 4,
    /// This particular native lane is ready after receiving the retirement
    /// receipt for intent_id, and may claim its next intent. Other lanes'
    /// completions are unaffected by this acknowledgement.
    BrokerReadyNext = 5,
}

impl FrontendPollPhase {
    const fn from_wire(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::BrokerNext),
            2 => Some(Self::AgentClaim),
            3 => Some(Self::BrokerRetired),
            4 => Some(Self::BrokerControl),
            5 => Some(Self::BrokerReadyNext),
            _ => None,
        }
    }
}

#[derive(
    Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq, Unaligned,
)]
#[repr(C)]
struct FrontendPollV1 {
    magic: U32,
    version: U16,
    phase: u8,
    reserved0: u8,
    intent_id: [u8; 16],
    reserved1: [u8; 8],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrontendPoll {
    pub phase: FrontendPollPhase,
    pub intent_id: FrontendIntentId,
}

impl FrontendPoll {
    pub fn encode(self, output: &mut [u8]) -> Option<usize> {
        let valid_id = match self.phase {
            FrontendPollPhase::BrokerNext => self.intent_id.is_zero(),
            FrontendPollPhase::AgentClaim
            | FrontendPollPhase::BrokerReadyNext
            | FrontendPollPhase::BrokerRetired
            | FrontendPollPhase::BrokerControl => !self.intent_id.is_zero(),
        };
        if !valid_id {
            return None;
        }
        let output = output.get_mut(..FRONTEND_RECORD_HEADER_LEN)?;
        output.copy_from_slice(
            FrontendPollV1 {
                magic: U32::new(FRONTEND_POLL_MAGIC),
                version: U16::new(FRONTEND_SCHEMA_V1),
                phase: self.phase as u8,
                reserved0: 0,
                intent_id: self.intent_id.0,
                reserved1: [0; 8],
            }
            .as_bytes(),
        );
        Some(FRONTEND_RECORD_HEADER_LEN)
    }
}

#[must_use]
pub fn decode_frontend_poll(input: &[u8]) -> Option<FrontendPoll> {
    let record = FrontendPollV1::read_from_bytes(input).ok()?;
    if record.magic.get() != FRONTEND_POLL_MAGIC
        || record.version.get() != FRONTEND_SCHEMA_V1
        || record.reserved0 != 0
        || record.reserved1 != [0; 8]
    {
        return None;
    }
    let poll = FrontendPoll {
        phase: FrontendPollPhase::from_wire(record.phase)?,
        intent_id: FrontendIntentId(record.intent_id),
    };
    match poll.phase {
        FrontendPollPhase::BrokerNext if poll.intent_id.is_zero() => Some(poll),
        FrontendPollPhase::AgentClaim
        | FrontendPollPhase::BrokerReadyNext
        | FrontendPollPhase::BrokerRetired
        | FrontendPollPhase::BrokerControl
            if !poll.intent_id.is_zero() =>
        {
            Some(poll)
        }
        _ => None,
    }
}

#[derive(
    Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq, Unaligned,
)]
#[repr(C)]
struct FrontendTextHeaderV1 {
    magic: U32,
    version: U16,
    kind: u8,
    reserved0: u8,
    intent_id: [u8; 16],
    text_len: U16,
    reserved1: [u8; 6],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrontendIntent<'a> {
    pub intent_id: FrontendIntentId,
    pub text: &'a str,
}

/// Exact one-byte terminal reply to BrokerControl. This is an authenticated
/// frontend decision, never model text or a grant of worker authority.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum FrontendControl {
    Active = 0,
    CancelRequested = 1,
    Settled = 2,
}

impl FrontendControl {
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        match bytes {
            [0] => Some(Self::Active),
            [1] => Some(Self::CancelRequested),
            [2] => Some(Self::Settled),
            _ => None,
        }
    }
}

impl FrontendIntent<'_> {
    pub fn encode(self, output: &mut [u8]) -> Option<usize> {
        if self.intent_id.is_zero() {
            return None;
        }
        encode_text_record(FRONTEND_INTENT_MAGIC, 1, self.intent_id, self.text, output)
    }
}

#[must_use]
pub fn decode_frontend_intent(input: &[u8]) -> Option<FrontendIntent<'_>> {
    let (kind, intent_id, text) = decode_text_record(FRONTEND_INTENT_MAGIC, input)?;
    (kind == 1 && !intent_id.is_zero()).then_some(FrontendIntent { intent_id, text })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum FrontendResultKind {
    Completed = 1,
    Refused = 2,
    Failed = 3,
    Cancelled = 4,
}

impl FrontendResultKind {
    const fn from_wire(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::Completed),
            2 => Some(Self::Refused),
            3 => Some(Self::Failed),
            4 => Some(Self::Cancelled),
            _ => None,
        }
    }
}

/// A terminal proposal for a claimed intent. The first accepted authorized
/// proposal wins. If an agent result races its owning broker's Failed fallback, the
/// fallback may join that result's publication receipt without replacing it.
/// A receipt certifies publication of the intent's terminal outcome, not that
/// the sender's proposed text/kind won. Other routes and generations cannot
/// join; an additional worker result or repeated broker fallback is refused.
/// A host cancellation accepted before any result reserves a Cancelled outcome;
/// later worker/broker proposals acknowledge that outcome instead of undoing it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrontendResult<'a> {
    pub intent_id: FrontendIntentId,
    pub kind: FrontendResultKind,
    pub text: &'a str,
}

impl FrontendResult<'_> {
    pub fn encode(self, output: &mut [u8]) -> Option<usize> {
        if self.intent_id.is_zero() {
            return None;
        }
        encode_text_record(
            FRONTEND_RESULT_MAGIC,
            self.kind as u8,
            self.intent_id,
            self.text,
            output,
        )
    }
}

#[must_use]
pub fn decode_frontend_result(input: &[u8]) -> Option<FrontendResult<'_>> {
    let (kind, intent_id, text) = decode_text_record(FRONTEND_RESULT_MAGIC, input)?;
    (!intent_id.is_zero()).then_some(FrontendResult {
        intent_id,
        kind: FrontendResultKind::from_wire(kind)?,
        text,
    })
}

fn encode_text_record(
    magic: u32,
    kind: u8,
    intent_id: FrontendIntentId,
    text: &str,
    output: &mut [u8],
) -> Option<usize> {
    if text.is_empty() || text.len() > FRONTEND_TEXT_MAX || text.as_bytes().contains(&0) {
        return None;
    }
    let length = FRONTEND_RECORD_HEADER_LEN.checked_add(text.len())?;
    let output = output.get_mut(..length)?;
    output[..FRONTEND_RECORD_HEADER_LEN].copy_from_slice(
        FrontendTextHeaderV1 {
            magic: U32::new(magic),
            version: U16::new(FRONTEND_SCHEMA_V1),
            kind,
            reserved0: 0,
            intent_id: intent_id.0,
            text_len: U16::new(text.len() as u16),
            reserved1: [0; 6],
        }
        .as_bytes(),
    );
    output[FRONTEND_RECORD_HEADER_LEN..].copy_from_slice(text.as_bytes());
    Some(length)
}

fn decode_text_record(magic: u32, input: &[u8]) -> Option<(u8, FrontendIntentId, &str)> {
    let header =
        FrontendTextHeaderV1::read_from_bytes(input.get(..FRONTEND_RECORD_HEADER_LEN)?).ok()?;
    let text_len = usize::from(header.text_len.get());
    if header.magic.get() != magic
        || header.version.get() != FRONTEND_SCHEMA_V1
        || header.kind == 0
        || header.reserved0 != 0
        || header.reserved1 != [0; 6]
        || text_len == 0
        || text_len > FRONTEND_TEXT_MAX
        || input.len() != FRONTEND_RECORD_HEADER_LEN.checked_add(text_len)?
    {
        return None;
    }
    let text = str::from_utf8(&input[FRONTEND_RECORD_HEADER_LEN..]).ok()?;
    (!text.as_bytes().contains(&0)).then_some((
        header.kind,
        FrontendIntentId(header.intent_id),
        text,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concurrent_lane_readiness_names_exactly_one_prior_intent() {
        let mut bytes = [0; FRONTEND_RECORD_HEADER_LEN];
        let poll = FrontendPoll {
            phase: FrontendPollPhase::BrokerReadyNext,
            intent_id: FrontendIntentId::from_sequence(9).unwrap(),
        };
        let len = poll.encode(&mut bytes).unwrap();
        assert_eq!(decode_frontend_poll(&bytes[..len]), Some(poll));
        bytes[8..24].fill(0);
        assert!(decode_frontend_poll(&bytes).is_none());
        assert!(
            FrontendPoll {
                intent_id: FrontendIntentId::ZERO,
                ..poll
            }
            .encode(&mut bytes)
            .is_none()
        );
    }

    #[test]
    fn broker_and_agent_claims_are_distinct_and_bounded() {
        let id = FrontendIntentId::from_sequence(1).unwrap();
        let mut bytes = [0; INFERENCE_PART_MAX];
        let broker = FrontendPoll {
            phase: FrontendPollPhase::BrokerNext,
            intent_id: FrontendIntentId::ZERO,
        };
        let len = broker.encode(&mut bytes).unwrap();
        assert_eq!(decode_frontend_poll(&bytes[..len]), Some(broker));
        let agent = FrontendPoll {
            phase: FrontendPollPhase::AgentClaim,
            intent_id: id,
        };
        let len = agent.encode(&mut bytes).unwrap();
        assert_eq!(decode_frontend_poll(&bytes[..len]), Some(agent));
        assert!(
            FrontendPoll {
                phase: FrontendPollPhase::BrokerNext,
                intent_id: id,
            }
            .encode(&mut bytes)
            .is_none()
        );
    }

    #[test]
    fn retirement_poll_requires_an_exact_nonzero_intent() {
        let mut bytes = [0; FRONTEND_RECORD_HEADER_LEN];
        let poll = FrontendPoll {
            phase: FrontendPollPhase::BrokerRetired,
            intent_id: FrontendIntentId::from_sequence(7).unwrap(),
        };
        let len = poll.encode(&mut bytes).unwrap();
        assert_eq!(decode_frontend_poll(&bytes[..len]), Some(poll));
        bytes[8..24].fill(0);
        assert!(decode_frontend_poll(&bytes).is_none());
        assert!(
            FrontendPoll {
                intent_id: FrontendIntentId::ZERO,
                ..poll
            }
            .encode(&mut bytes)
            .is_none()
        );
    }

    #[test]
    fn control_queries_and_replies_are_bounded_typed_records() {
        let poll = FrontendPoll {
            phase: FrontendPollPhase::BrokerControl,
            intent_id: FrontendIntentId::from_sequence(8).unwrap(),
        };
        let mut bytes = [0; FRONTEND_RECORD_HEADER_LEN];
        let len = poll.encode(&mut bytes).unwrap();
        assert_eq!(decode_frontend_poll(&bytes[..len]), Some(poll));
        assert!(
            FrontendPoll {
                intent_id: FrontendIntentId::ZERO,
                ..poll
            }
            .encode(&mut bytes)
            .is_none()
        );
        for state in [
            FrontendControl::Active,
            FrontendControl::CancelRequested,
            FrontendControl::Settled,
        ] {
            assert_eq!(FrontendControl::decode(&[state as u8]), Some(state));
        }
        for bytes in [&[][..], &[3][..], &[1, 0][..], b"cancel"] {
            assert_eq!(FrontendControl::decode(bytes), None);
        }
    }

    #[test]
    fn intent_carries_an_opaque_id_and_exact_text() {
        let id = FrontendIntentId::from_sequence(1).unwrap();
        let mut bytes = [0; INFERENCE_PART_MAX];
        let len = FrontendIntent {
            intent_id: id,
            text: "read X",
        }
        .encode(&mut bytes)
        .unwrap();
        assert_eq!(
            decode_frontend_intent(&bytes[..len]),
            Some(FrontendIntent {
                intent_id: id,
                text: "read X"
            })
        );
        bytes[len - 1] = b'Y';
        assert_eq!(
            decode_frontend_intent(&bytes[..len]),
            Some(FrontendIntent {
                intent_id: id,
                text: "read Y"
            })
        );
    }

    #[test]
    fn result_round_trips_without_conferring_authority() {
        let id = FrontendIntentId::from_sequence(1).unwrap();
        let mut bytes = [0; INFERENCE_PART_MAX];
        let len = FrontendResult {
            intent_id: id,
            kind: FrontendResultKind::Completed,
            text: "ordinary X content",
        }
        .encode(&mut bytes)
        .unwrap();
        assert_eq!(
            decode_frontend_result(&bytes[..len]),
            Some(FrontendResult {
                intent_id: id,
                kind: FrontendResultKind::Completed,
                text: "ordinary X content"
            })
        );
    }
}
