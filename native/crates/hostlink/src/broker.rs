//! Bounded inner protocol for prompt-scoped broker planning.
//!
//! KIR1/KIE1 remain the transport and inference envelopes. KBP1 contains an
//! authenticated frontend intent plus an inert catalog assembled by the
//! resident broker. KBR1 contains only proposal-local selectors. In
//! particular, neither record has a representation for a capability, CID,
//! endpoint, task pointer, host descriptor, or ambient host path.

use core::str;

use zerocopy::byteorder::little_endian::{U16, U32, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

pub const BROKER_REQUEST_MAGIC: u32 = u32::from_le_bytes(*b"KBP1");
pub const BROKER_REPLY_MAGIC: u32 = u32::from_le_bytes(*b"KBR1");
pub const BROKER_SCHEMA_V1: u16 = 1;
pub const TASK_INTENT_MAX: usize = 128;
pub const BROKER_LABEL_MAX: usize = 64;
/// Complete bounded catalog visible to one broker-model request. Sixty-four
/// rows cover Kore's frozen 23-command bootstrap catalog plus 41 exact
/// workspace objects. Catalog construction refuses overflow rather than
/// silently hiding authority from the planner.
pub const BROKER_OFFERS_MAX: usize = 64;
pub const BROKER_ATTACHMENTS_MAX: usize = 4;
pub const BROKER_SELECTORS_MAX: usize = 8;
pub const BROKER_REQUEST_HEADER_LEN: usize = 120;
pub const BROKER_OFFER_HEADER_LEN: usize = 36;
pub const BROKER_ATTACHMENT_LEN: usize = 20;
pub const BROKER_REPLY_HEADER_LEN: usize = 164;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TaskBudget {
    pub cpu_share: u64,
    pub cpu_time_ns: u64,
    pub memory_bytes: u64,
    pub scratch_bytes: u64,
    pub storage_bytes: u64,
    pub prompt_bytes: u64,
    pub result_bytes: u64,
    pub tool_calls: u64,
    pub worker_jobs: u64,
    pub outstanding_io: u64,
    pub wall_duration_ns: u64,
}

impl TaskBudget {
    pub const ZERO: Self = Self {
        cpu_share: 0,
        cpu_time_ns: 0,
        memory_bytes: 0,
        scratch_bytes: 0,
        storage_bytes: 0,
        prompt_bytes: 0,
        result_bytes: 0,
        tool_calls: 0,
        worker_jobs: 0,
        outstanding_io: 0,
        wall_duration_ns: 0,
    };

    #[must_use]
    pub const fn fits_within(self, ceiling: Self) -> bool {
        self.cpu_share <= ceiling.cpu_share
            && self.cpu_time_ns <= ceiling.cpu_time_ns
            && self.memory_bytes <= ceiling.memory_bytes
            && self.scratch_bytes <= ceiling.scratch_bytes
            && self.storage_bytes <= ceiling.storage_bytes
            && self.prompt_bytes <= ceiling.prompt_bytes
            && self.result_bytes <= ceiling.result_bytes
            && self.tool_calls <= ceiling.tool_calls
            && self.worker_jobs <= ceiling.worker_jobs
            && self.outstanding_io <= ceiling.outstanding_io
            && self.wall_duration_ns <= ceiling.wall_duration_ns
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum BrokerOfferKind {
    View = 1,
    Object = 2,
    Executable = 3,
    Budget = 4,
    ExistingAgent = 5,
    ExistingGrant = 6,
}

impl BrokerOfferKind {
    const fn from_wire(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::View),
            2 => Some(Self::Object),
            3 => Some(Self::Executable),
            4 => Some(Self::Budget),
            5 => Some(Self::ExistingAgent),
            6 => Some(Self::ExistingGrant),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BrokerOffer<'a> {
    pub reference: [u8; 16],
    pub kind: BrokerOfferKind,
    pub permitted_ops: u64,
    pub max_uses: u64,
    pub label: &'a str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceAttachment {
    /// A frontend-selected offer, never a path or authority identifier.
    pub offer: [u8; 16],
    pub rights_ceiling: u32,
}

/// The authenticated text is accepted only on the dedicated frontend-to-
/// broker endpoint. Model, runner, tool-result, and workspace endpoints do
/// not construct this type and cannot feed this state transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TaskIntentRequest<'a> {
    pub plan: [u8; 16],
    pub intent: &'a str,
    pub ceiling: TaskBudget,
    pub offers: &'a [BrokerOffer<'a>],
    pub attachments: &'a [ResourceAttachment],
}

#[derive(
    Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq, Unaligned,
)]
#[repr(C)]
struct BrokerRequestHeaderV1 {
    magic: U32,
    version: U16,
    flags: U16,
    plan: [u8; 16],
    intent_len: U16,
    offer_count: u8,
    attachment_count: u8,
    budget: [U64; 11],
    reserved: U32,
}

#[derive(
    Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq, Unaligned,
)]
#[repr(C)]
struct BrokerOfferHeaderV1 {
    reference: [u8; 16],
    kind: u8,
    flags: u8,
    label_len: U16,
    permitted_ops: U64,
    max_uses: U64,
}

#[derive(
    Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq, Unaligned,
)]
#[repr(C)]
struct BrokerAttachmentV1 {
    offer: [u8; 16],
    rights_ceiling: U32,
}

impl TaskIntentRequest<'_> {
    pub fn encode(&self, output: &mut [u8]) -> Option<usize> {
        if !valid_text(self.intent, TASK_INTENT_MAX)
            || self.offers.len() > BROKER_OFFERS_MAX
            || self.attachments.len() > BROKER_ATTACHMENTS_MAX
        {
            return None;
        }
        for (index, offer) in self.offers.iter().enumerate() {
            if offer.reference == [0; 16]
                || !valid_text(offer.label, BROKER_LABEL_MAX)
                || self.offers[..index]
                    .iter()
                    .any(|other| other.reference == offer.reference)
            {
                return None;
            }
        }
        if self
            .attachments
            .iter()
            .enumerate()
            .any(|(index, attachment)| {
                attachment.offer == [0; 16]
                    || self.attachments[..index]
                        .iter()
                        .any(|other| other.offer == attachment.offer)
                    || !self
                        .offers
                        .iter()
                        .any(|offer| offer.reference == attachment.offer)
            })
        {
            return None;
        }
        let mut length = BROKER_REQUEST_HEADER_LEN.checked_add(self.intent.len())?;
        for offer in self.offers {
            length = length
                .checked_add(BROKER_OFFER_HEADER_LEN)?
                .checked_add(offer.label.len())?;
        }
        length = length.checked_add(BROKER_ATTACHMENT_LEN.checked_mul(self.attachments.len())?)?;
        let output = output.get_mut(..length)?;
        let header = BrokerRequestHeaderV1 {
            magic: U32::new(BROKER_REQUEST_MAGIC),
            version: U16::new(BROKER_SCHEMA_V1),
            flags: U16::ZERO,
            plan: self.plan,
            intent_len: U16::new(self.intent.len() as u16),
            offer_count: self.offers.len() as u8,
            attachment_count: self.attachments.len() as u8,
            budget: budget_to_wire(self.ceiling),
            reserved: U32::ZERO,
        };
        output[..BROKER_REQUEST_HEADER_LEN].copy_from_slice(header.as_bytes());
        let mut cursor = BROKER_REQUEST_HEADER_LEN;
        output[cursor..cursor + self.intent.len()].copy_from_slice(self.intent.as_bytes());
        cursor += self.intent.len();
        for offer in self.offers {
            let header = BrokerOfferHeaderV1 {
                reference: offer.reference,
                kind: offer.kind as u8,
                flags: 0,
                label_len: U16::new(offer.label.len() as u16),
                permitted_ops: U64::new(offer.permitted_ops),
                max_uses: U64::new(offer.max_uses),
            };
            output[cursor..cursor + BROKER_OFFER_HEADER_LEN].copy_from_slice(header.as_bytes());
            cursor += BROKER_OFFER_HEADER_LEN;
            output[cursor..cursor + offer.label.len()].copy_from_slice(offer.label.as_bytes());
            cursor += offer.label.len();
        }
        for attachment in self.attachments {
            let wire = BrokerAttachmentV1 {
                offer: attachment.offer,
                rights_ceiling: U32::new(attachment.rights_ceiling),
            };
            output[cursor..cursor + BROKER_ATTACHMENT_LEN].copy_from_slice(wire.as_bytes());
            cursor += BROKER_ATTACHMENT_LEN;
        }
        Some(cursor)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodedTaskIntent<'a> {
    pub plan: [u8; 16],
    pub intent: &'a str,
    pub ceiling: TaskBudget,
    offer_count: u8,
    attachment_count: u8,
    records: &'a [u8],
}

impl<'a> DecodedTaskIntent<'a> {
    pub fn offers(&self) -> BrokerOffers<'a> {
        BrokerOffers {
            remaining: self.offer_count,
            input: self.records,
        }
    }

    pub fn attachments(&self) -> BrokerAttachments<'a> {
        let mut offers = self.offers();
        for _ in 0..self.offer_count {
            let _ = offers.next();
        }
        BrokerAttachments {
            remaining: self.attachment_count,
            input: offers.input,
        }
    }
}

pub struct BrokerOffers<'a> {
    remaining: u8,
    input: &'a [u8],
}

impl<'a> Iterator for BrokerOffers<'a> {
    type Item = BrokerOffer<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let header =
            BrokerOfferHeaderV1::read_from_bytes(self.input.get(..BROKER_OFFER_HEADER_LEN)?)
                .ok()?;
        let label_len = header.label_len.get() as usize;
        let end = BROKER_OFFER_HEADER_LEN.checked_add(label_len)?;
        let label = str::from_utf8(self.input.get(BROKER_OFFER_HEADER_LEN..end)?).ok()?;
        self.input = &self.input[end..];
        self.remaining -= 1;
        Some(BrokerOffer {
            reference: header.reference,
            kind: BrokerOfferKind::from_wire(header.kind)?,
            permitted_ops: header.permitted_ops.get(),
            max_uses: header.max_uses.get(),
            label,
        })
    }
}

pub struct BrokerAttachments<'a> {
    remaining: u8,
    input: &'a [u8],
}

impl Iterator for BrokerAttachments<'_> {
    type Item = ResourceAttachment;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let wire =
            BrokerAttachmentV1::read_from_bytes(self.input.get(..BROKER_ATTACHMENT_LEN)?).ok()?;
        self.input = &self.input[BROKER_ATTACHMENT_LEN..];
        self.remaining -= 1;
        Some(ResourceAttachment {
            offer: wire.offer,
            rights_ceiling: wire.rights_ceiling.get(),
        })
    }
}

#[must_use]
pub fn decode_task_intent(input: &[u8]) -> Option<DecodedTaskIntent<'_>> {
    let header =
        BrokerRequestHeaderV1::read_from_bytes(input.get(..BROKER_REQUEST_HEADER_LEN)?).ok()?;
    let intent_len = header.intent_len.get() as usize;
    if header.magic.get() != BROKER_REQUEST_MAGIC
        || header.version.get() != BROKER_SCHEMA_V1
        || header.flags.get() != 0
        || header.reserved.get() != 0
        || header.plan == [0; 16]
        || intent_len == 0
        || intent_len > TASK_INTENT_MAX
        || usize::from(header.offer_count) > BROKER_OFFERS_MAX
        || usize::from(header.attachment_count) > BROKER_ATTACHMENTS_MAX
    {
        return None;
    }
    let intent_end = BROKER_REQUEST_HEADER_LEN.checked_add(intent_len)?;
    let intent = str::from_utf8(input.get(BROKER_REQUEST_HEADER_LEN..intent_end)?).ok()?;
    if !valid_text(intent, TASK_INTENT_MAX) {
        return None;
    }
    let decoded = DecodedTaskIntent {
        plan: header.plan,
        intent,
        ceiling: budget_from_wire(header.budget),
        offer_count: header.offer_count,
        attachment_count: header.attachment_count,
        records: input.get(intent_end..)?,
    };
    let mut seen = [[0_u8; 16]; BROKER_OFFERS_MAX];
    let mut offer_count = 0;
    let mut offers = decoded.offers();
    while let Some(offer) = offers.next() {
        if offer.reference == [0; 16]
            || !valid_text(offer.label, BROKER_LABEL_MAX)
            || seen[..offer_count].contains(&offer.reference)
        {
            return None;
        }
        seen[offer_count] = offer.reference;
        offer_count += 1;
    }
    if offer_count != usize::from(header.offer_count) {
        return None;
    }
    let mut attachment_count = 0;
    let mut attachments = BrokerAttachments {
        remaining: header.attachment_count,
        input: offers.input,
    };
    let mut attached = [[0_u8; 16]; BROKER_ATTACHMENTS_MAX];
    while let Some(attachment) = attachments.next() {
        if attachment.offer == [0; 16]
            || !seen[..offer_count].contains(&attachment.offer)
            || attached[..attachment_count].contains(&attachment.offer)
        {
            return None;
        }
        attached[attachment_count] = attachment.offer;
        attachment_count += 1;
    }
    if attachment_count != usize::from(header.attachment_count) || !attachments.input.is_empty() {
        return None;
    }
    Some(decoded)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum BrokerActionKind {
    CreateAgent = 1,
    GrantView = 2,
    GrantObject = 3,
    GrantExecutable = 4,
    RevokeGrant = 5,
    SetBudget = 6,
    CancelAgent = 7,
    InspectAgent = 8,
    DestroyAgent = 9,
    Refused = 255,
}

impl BrokerActionKind {
    const fn from_wire(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::CreateAgent),
            2 => Some(Self::GrantView),
            3 => Some(Self::GrantObject),
            4 => Some(Self::GrantExecutable),
            5 => Some(Self::RevokeGrant),
            6 => Some(Self::SetBudget),
            7 => Some(Self::CancelAgent),
            8 => Some(Self::InspectAgent),
            9 => Some(Self::DestroyAgent),
            255 => Some(Self::Refused),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BrokerPlanningReply<'a> {
    pub plan: [u8; 16],
    pub action: BrokerActionKind,
    pub target: [u8; 16],
    pub subject: [u8; 16],
    pub budget: TaskBudget,
    pub rights: u64,
    pub limit: u32,
    pub reason: u8,
    pub label: &'a str,
    pub selectors: &'a [[u8; 16]],
}

#[derive(
    Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq, Unaligned,
)]
#[repr(C)]
struct BrokerReplyHeaderV1 {
    magic: U32,
    version: U16,
    action: u8,
    flags: u8,
    plan: [u8; 16],
    target: [u8; 16],
    subject: [u8; 16],
    budget: [U64; 11],
    rights: U64,
    limit: U32,
    label_len: U16,
    selector_count: u8,
    reason: u8,
    reserved: U32,
}

impl BrokerPlanningReply<'_> {
    pub fn encode(&self, output: &mut [u8]) -> Option<usize> {
        if self.plan == [0; 16]
            || self.selectors.len() > BROKER_SELECTORS_MAX
            || (!self.label.is_empty() && !valid_text(self.label, BROKER_LABEL_MAX))
            || !valid_reply_shape(self)
            || self.selectors.iter().enumerate().any(|(index, selector)| {
                *selector == [0; 16] || self.selectors[..index].contains(selector)
            })
        {
            return None;
        }
        let length = BROKER_REPLY_HEADER_LEN
            .checked_add(self.label.len())?
            .checked_add(16_usize.checked_mul(self.selectors.len())?)?;
        let output = output.get_mut(..length)?;
        let header = BrokerReplyHeaderV1 {
            magic: U32::new(BROKER_REPLY_MAGIC),
            version: U16::new(BROKER_SCHEMA_V1),
            action: self.action as u8,
            flags: 0,
            plan: self.plan,
            target: self.target,
            subject: self.subject,
            budget: budget_to_wire(self.budget),
            rights: U64::new(self.rights),
            limit: U32::new(self.limit),
            label_len: U16::new(self.label.len() as u16),
            selector_count: self.selectors.len() as u8,
            reason: self.reason,
            reserved: U32::ZERO,
        };
        output[..BROKER_REPLY_HEADER_LEN].copy_from_slice(header.as_bytes());
        let mut cursor = BROKER_REPLY_HEADER_LEN;
        output[cursor..cursor + self.label.len()].copy_from_slice(self.label.as_bytes());
        cursor += self.label.len();
        for selector in self.selectors {
            output[cursor..cursor + 16].copy_from_slice(selector);
            cursor += 16;
        }
        Some(cursor)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodedBrokerReply<'a> {
    pub plan: [u8; 16],
    pub action: BrokerActionKind,
    pub target: [u8; 16],
    pub subject: [u8; 16],
    pub budget: TaskBudget,
    pub rights: u64,
    pub limit: u32,
    pub reason: u8,
    pub label: &'a str,
    selectors: &'a [u8],
}

impl DecodedBrokerReply<'_> {
    pub fn selectors(&self) -> impl Iterator<Item = [u8; 16]> + '_ {
        self.selectors.chunks_exact(16).map(|bytes| {
            let mut selector = [0; 16];
            selector.copy_from_slice(bytes);
            selector
        })
    }
}

#[must_use]
pub fn decode_broker_reply(input: &[u8]) -> Option<DecodedBrokerReply<'_>> {
    let header =
        BrokerReplyHeaderV1::read_from_bytes(input.get(..BROKER_REPLY_HEADER_LEN)?).ok()?;
    let label_len = header.label_len.get() as usize;
    let selector_count = header.selector_count as usize;
    let selectors_len = selector_count.checked_mul(16)?;
    let label_end = BROKER_REPLY_HEADER_LEN.checked_add(label_len)?;
    let end = label_end.checked_add(selectors_len)?;
    if header.magic.get() != BROKER_REPLY_MAGIC
        || header.version.get() != BROKER_SCHEMA_V1
        || header.flags != 0
        || header.reserved.get() != 0
        || header.plan == [0; 16]
        || label_len > BROKER_LABEL_MAX
        || selector_count > BROKER_SELECTORS_MAX
        || input.len() != end
    {
        return None;
    }
    let label = str::from_utf8(input.get(BROKER_REPLY_HEADER_LEN..label_end)?).ok()?;
    let decoded = DecodedBrokerReply {
        plan: header.plan,
        action: BrokerActionKind::from_wire(header.action)?,
        target: header.target,
        subject: header.subject,
        budget: budget_from_wire(header.budget),
        rights: header.rights.get(),
        limit: header.limit.get(),
        reason: header.reason,
        label,
        selectors: &input[label_end..end],
    };
    let mut seen = [[0_u8; 16]; BROKER_SELECTORS_MAX];
    for (index, selector) in decoded.selectors().enumerate() {
        if selector == [0; 16] || seen[..index].contains(&selector) {
            return None;
        }
        seen[index] = selector;
    }
    let borrowed = BrokerPlanningReply {
        plan: decoded.plan,
        action: decoded.action,
        target: decoded.target,
        subject: decoded.subject,
        budget: decoded.budget,
        rights: decoded.rights,
        limit: decoded.limit,
        reason: decoded.reason,
        label: decoded.label,
        selectors: &seen[..selector_count],
    };
    valid_reply_shape(&borrowed).then_some(decoded)
}

fn valid_reply_shape(reply: &BrokerPlanningReply<'_>) -> bool {
    let target = reply.target != [0; 16];
    let subject = reply.subject != [0; 16];
    match reply.action {
        BrokerActionKind::CreateAgent => {
            !target
                && !subject
                && !reply.label.is_empty()
                && reply.rights == 0
                && reply.limit == 0
                && reply.reason == 0
        }
        BrokerActionKind::GrantView | BrokerActionKind::GrantObject => {
            target
                && subject
                && reply.label.is_empty()
                && reply.selectors.is_empty()
                && reply.rights != 0
                && reply.limit == 0
                && reply.reason == 0
                && reply.budget == TaskBudget::ZERO
        }
        BrokerActionKind::GrantExecutable => {
            target
                && subject
                && reply.label.is_empty()
                && reply.selectors.is_empty()
                && reply.rights == 0
                && reply.limit != 0
                && reply.reason == 0
                && reply.budget == TaskBudget::ZERO
        }
        BrokerActionKind::RevokeGrant => {
            target
                && subject
                && reply.label.is_empty()
                && reply.selectors.is_empty()
                && reply.rights == 0
                && reply.limit == 0
                && reply.reason == 0
                && reply.budget == TaskBudget::ZERO
        }
        BrokerActionKind::SetBudget => {
            target
                && !subject
                && reply.label.is_empty()
                && reply.selectors.is_empty()
                && reply.rights == 0
                && reply.limit == 0
                && reply.reason == 0
        }
        BrokerActionKind::CancelAgent => {
            target
                && !subject
                && reply.label.is_empty()
                && reply.selectors.is_empty()
                && reply.rights == 0
                && reply.limit == 0
                && reply.reason != 0
                && reply.budget == TaskBudget::ZERO
        }
        BrokerActionKind::InspectAgent => {
            target
                && !subject
                && reply.label.is_empty()
                && reply.selectors.is_empty()
                && reply.rights != 0
                && reply.limit == 0
                && reply.reason == 0
                && reply.budget == TaskBudget::ZERO
        }
        BrokerActionKind::DestroyAgent => {
            target
                && !subject
                && reply.label.is_empty()
                && reply.selectors.is_empty()
                && reply.rights == 0
                && reply.limit == 0
                && reply.reason == 0
                && reply.budget == TaskBudget::ZERO
        }
        BrokerActionKind::Refused => {
            !target
                && !subject
                && reply.label.is_empty()
                && reply.selectors.is_empty()
                && reply.rights == 0
                && reply.limit == 0
                && reply.reason != 0
                && reply.budget == TaskBudget::ZERO
        }
    }
}

fn valid_text(value: &str, max: usize) -> bool {
    !value.is_empty() && value.len() <= max && !value.as_bytes().contains(&0)
}

fn budget_to_wire(budget: TaskBudget) -> [U64; 11] {
    [
        U64::new(budget.cpu_share),
        U64::new(budget.cpu_time_ns),
        U64::new(budget.memory_bytes),
        U64::new(budget.scratch_bytes),
        U64::new(budget.storage_bytes),
        U64::new(budget.prompt_bytes),
        U64::new(budget.result_bytes),
        U64::new(budget.tool_calls),
        U64::new(budget.worker_jobs),
        U64::new(budget.outstanding_io),
        U64::new(budget.wall_duration_ns),
    ]
}

fn budget_from_wire(wire: [U64; 11]) -> TaskBudget {
    TaskBudget {
        cpu_share: wire[0].get(),
        cpu_time_ns: wire[1].get(),
        memory_bytes: wire[2].get(),
        scratch_bytes: wire[3].get(),
        storage_bytes: wire[4].get(),
        prompt_bytes: wire[5].get(),
        result_bytes: wire[6].get(),
        tool_calls: wire[7].get(),
        worker_jobs: wire[8].get(),
        outstanding_io: wire[9].get(),
        wall_duration_ns: wire[10].get(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const X: [u8; 16] = [1; 16];
    const CAT: [u8; 16] = [2; 16];
    const PLAN: [u8; 16] = *b"prompt-plan-0001";

    #[test]
    fn task_intent_round_trip_is_bounded_and_exact() {
        let offers = [
            BrokerOffer {
                reference: X,
                kind: BrokerOfferKind::Object,
                permitted_ops: 1,
                max_uses: 0,
                label: "workspace/X",
            },
            BrokerOffer {
                reference: CAT,
                kind: BrokerOfferKind::Executable,
                permitted_ops: 1,
                max_uses: 3,
                label: "cat",
            },
        ];
        let attachments = [ResourceAttachment {
            offer: X,
            rights_ceiling: 1,
        }];
        let request = TaskIntentRequest {
            plan: PLAN,
            intent: "read X",
            ceiling: TaskBudget {
                tool_calls: 3,
                ..TaskBudget::ZERO
            },
            offers: &offers,
            attachments: &attachments,
        };
        let mut wire = [0_u8; crate::INFERENCE_PART_MAX];
        let len = request.encode(&mut wire).unwrap();
        let decoded = decode_task_intent(&wire[..len]).unwrap();
        assert_eq!(decoded.plan, PLAN);
        assert_eq!(decoded.intent, "read X");
        assert_eq!(
            decoded.offers().collect::<heapless::Vec<_, 2>>().as_slice(),
            &offers
        );
        assert_eq!(
            decoded
                .attachments()
                .collect::<heapless::Vec<_, 1>>()
                .as_slice(),
            &attachments
        );
        assert!(decode_task_intent(&wire[..len - 1]).is_none());
        wire[len] = 1;
        assert!(decode_task_intent(&wire[..len + 1]).is_none());
    }

    #[test]
    fn task_intent_and_agent_proposal_allow_zero_capabilities() {
        let request = TaskIntentRequest {
            plan: PLAN,
            intent: "Explain the difference between two concepts.",
            ceiling: TaskBudget::ZERO,
            offers: &[],
            attachments: &[],
        };
        let mut wire = [0_u8; crate::INFERENCE_PART_MAX];
        let len = request.encode(&mut wire).unwrap();
        let decoded = decode_task_intent(&wire[..len]).unwrap();
        assert_eq!(decoded.offers().count(), 0);

        let reply = BrokerPlanningReply {
            plan: PLAN,
            action: BrokerActionKind::CreateAgent,
            target: [0; 16],
            subject: [0; 16],
            budget: TaskBudget::ZERO,
            rights: 0,
            limit: 0,
            reason: 0,
            label: "conversation",
            selectors: &[],
        };
        let len = reply.encode(&mut wire).unwrap();
        let decoded = decode_broker_reply(&wire[..len]).unwrap();
        assert_eq!(decoded.action, BrokerActionKind::CreateAgent);
        assert_eq!(decoded.selectors().count(), 0);
    }

    #[test]
    fn proposal_round_trip_rejects_duplicate_and_reserved_selectors() {
        let selectors = [CAT, X];
        let reply = BrokerPlanningReply {
            plan: PLAN,
            action: BrokerActionKind::CreateAgent,
            target: [0; 16],
            subject: [0; 16],
            budget: TaskBudget {
                tool_calls: 3,
                ..TaskBudget::ZERO
            },
            rights: 0,
            limit: 0,
            reason: 0,
            label: "agent/prompt",
            selectors: &selectors,
        };
        let mut wire = [0_u8; crate::INFERENCE_PART_MAX];
        let len = reply.encode(&mut wire).unwrap();
        let decoded = decode_broker_reply(&wire[..len]).unwrap();
        assert_eq!(decoded.action, BrokerActionKind::CreateAgent);
        assert_eq!(
            decoded
                .selectors()
                .collect::<heapless::Vec<_, 2>>()
                .as_slice(),
            &selectors
        );
        wire[BROKER_REPLY_HEADER_LEN - 1] = 1;
        assert!(decode_broker_reply(&wire[..len]).is_none());
        wire[BROKER_REPLY_HEADER_LEN - 1] = 0;
        let second = BROKER_REPLY_HEADER_LEN + reply.label.len() + 16;
        wire.copy_within(second - 16..second, second);
        assert!(decode_broker_reply(&wire[..len]).is_none());
    }
}
