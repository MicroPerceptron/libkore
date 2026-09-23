//! Bounded authority-state attestation from the resident guest broker.
//!
//! This is an inert post-commit observation. It deliberately has no field for
//! a capability, CID, slot handle, endpoint, task pointer, host fd, or source
//! path. The structurally broker-only HostLink route supplies provenance.

use core::str;
use zerocopy::byteorder::little_endian::{U16, U32, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

pub const AUTHORITY_REPORT_MAGIC: u32 = u32::from_le_bytes(*b"KAU1");
pub const AUTHORITY_REPORT_VERSION: u16 = 1;
pub const AUTHORITY_REPORT_HEADER_LEN: usize = 28;
pub const AUTHORITY_LABEL_MAX: usize = 48;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum AuthorityState {
    Installed = 1,
    Revoked = 2,
    Teardown = 3,
}

impl AuthorityState {
    fn from_wire(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::Installed),
            2 => Some(Self::Revoked),
            3 => Some(Self::Teardown),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum AuthorityKind {
    Object = 1,
    View = 2,
    Executable = 3,
    Budget = 4,
    Agent = 5,
}

impl AuthorityKind {
    fn from_wire(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::Object),
            2 => Some(Self::View),
            3 => Some(Self::Executable),
            4 => Some(Self::Budget),
            5 => Some(Self::Agent),
            _ => None,
        }
    }
}

#[derive(
    Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq, Unaligned,
)]
#[repr(C)]
struct AuthorityReportHeaderV1 {
    magic: U32,
    version: U16,
    state: u8,
    kind: u8,
    agent_slot: U16,
    reserved0: U16,
    generation: U64,
    rights: U32,
    label_len: U16,
    reserved1: U16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AuthorityReportView<'a> {
    pub state: AuthorityState,
    pub kind: AuthorityKind,
    pub agent_slot: u16,
    pub generation: u64,
    pub rights: u32,
    pub label: &'a str,
}

impl AuthorityReportView<'_> {
    pub fn encode(&self, output: &mut [u8]) -> Option<usize> {
        if self.generation == 0
            || self.label.is_empty()
            || self.label.len() > AUTHORITY_LABEL_MAX
            || self.label.as_bytes().contains(&0)
        {
            return None;
        }
        let length = AUTHORITY_REPORT_HEADER_LEN.checked_add(self.label.len())?;
        let output = output.get_mut(..length)?;
        let header = AuthorityReportHeaderV1 {
            magic: U32::new(AUTHORITY_REPORT_MAGIC),
            version: U16::new(AUTHORITY_REPORT_VERSION),
            state: self.state as u8,
            kind: self.kind as u8,
            agent_slot: U16::new(self.agent_slot),
            reserved0: U16::ZERO,
            generation: U64::new(self.generation),
            rights: U32::new(self.rights),
            label_len: U16::new(self.label.len() as u16),
            reserved1: U16::ZERO,
        };
        output[..AUTHORITY_REPORT_HEADER_LEN].copy_from_slice(header.as_bytes());
        output[AUTHORITY_REPORT_HEADER_LEN..].copy_from_slice(self.label.as_bytes());
        Some(length)
    }
}

pub fn decode_authority_report(input: &[u8]) -> Option<AuthorityReportView<'_>> {
    let header =
        AuthorityReportHeaderV1::read_from_bytes(input.get(..AUTHORITY_REPORT_HEADER_LEN)?).ok()?;
    let label_len = header.label_len.get() as usize;
    if header.magic.get() != AUTHORITY_REPORT_MAGIC
        || header.version.get() != AUTHORITY_REPORT_VERSION
        || header.reserved0.get() != 0
        || header.reserved1.get() != 0
        || header.generation.get() == 0
        || label_len == 0
        || label_len > AUTHORITY_LABEL_MAX
        || input.len() != AUTHORITY_REPORT_HEADER_LEN.checked_add(label_len)?
    {
        return None;
    }
    let label = str::from_utf8(&input[AUTHORITY_REPORT_HEADER_LEN..]).ok()?;
    if label.as_bytes().contains(&0) {
        return None;
    }
    Some(AuthorityReportView {
        state: AuthorityState::from_wire(header.state)?,
        kind: AuthorityKind::from_wire(header.kind)?,
        agent_slot: header.agent_slot.get(),
        generation: header.generation.get(),
        rights: header.rights.get(),
        label,
    })
}

const _: () =
    assert!(core::mem::size_of::<AuthorityReportHeaderV1>() == AUTHORITY_REPORT_HEADER_LEN);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_is_bounded_and_has_no_authority_identifier() {
        let report = AuthorityReportView {
            state: AuthorityState::Installed,
            kind: AuthorityKind::Object,
            agent_slot: 2,
            generation: 9,
            rights: 1,
            label: "workspace/docs-architecture.md",
        };
        let mut wire = [0; AUTHORITY_REPORT_HEADER_LEN + AUTHORITY_LABEL_MAX];
        let len = report.encode(&mut wire).unwrap();
        assert_eq!(decode_authority_report(&wire[..len]), Some(report));
        wire[10] = 1;
        assert!(decode_authority_report(&wire[..len]).is_none());
    }
}
