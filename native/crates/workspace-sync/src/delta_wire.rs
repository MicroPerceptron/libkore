//! Bounded bulk encoding for one complete canonical topology delta.

use heapless::Vec;
use zerocopy::{FromBytes, IntoBytes};

use crate::{
    Change, ChangeKind, ChangeSetBuilder, DeltaError, DeltaOp, Entry, TopologyDelta,
    bounds::{FORMAT_VERSION, MAX_CHANGES, MAX_PATH_BYTES},
    wire::EntryHeader,
};

pub const DELTA_BULK_MAGIC: [u8; 4] = *b"KWDB";
pub const DELTA_OP_BULK_MAGIC: [u8; 4] = *b"KWDO";
pub const MAX_DELTA_WIRE_BYTES: usize = 64 * 1024;
const HEAD_BYTES: usize = 148;
const OP_HEAD_BYTES: usize = 176;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeltaWireError {
    TooSmall,
    TooLarge,
    Malformed,
    UnsupportedVersion,
    Semantic(DeltaError),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeltaOpPage {
    pub sequence: u64,
    pub parent: [u8; 32],
    pub next: [u8; 32],
    pub delta_id: [u8; 32],
    pub origin_transaction: [u8; 32],
    pub index: u16,
    pub count: u16,
    pub op: DeltaOp,
}

/// Bounded receiver for the per-operation delta pages used by HostLink.
///
/// A delta is not visible until every page has arrived in canonical order and
/// the recomputed transaction identity matches the header repeated by every
/// page.  Keeping the receiver here makes the serial and shared-memory paths
/// use exactly the same gap/replay rules.
pub struct DeltaAssembler {
    sequence: u64,
    parent: [u8; 32],
    next: [u8; 32],
    delta_id: [u8; 32],
    origin_transaction: [u8; 32],
    count: u16,
    ops: Vec<DeltaOp, MAX_CHANGES>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeltaAssembleError {
    Malformed,
    Gap,
    Replay,
    TooMany,
    Semantic(DeltaError),
}

impl DeltaAssembler {
    pub fn begin(page: DeltaOpPage) -> Result<Self, DeltaAssembleError> {
        if page.index != 0 || page.count == 0 || usize::from(page.count) > MAX_CHANGES {
            return Err(DeltaAssembleError::Gap);
        }
        let mut ops = Vec::new();
        ops.push(page.op).map_err(|_| DeltaAssembleError::TooMany)?;
        Ok(Self {
            sequence: page.sequence,
            parent: page.parent,
            next: page.next,
            delta_id: page.delta_id,
            origin_transaction: page.origin_transaction,
            count: page.count,
            ops,
        })
    }

    pub fn push(&mut self, page: DeltaOpPage) -> Result<(), DeltaAssembleError> {
        if page.sequence != self.sequence
            || page.parent != self.parent
            || page.next != self.next
            || page.delta_id != self.delta_id
            || page.origin_transaction != self.origin_transaction
            || page.count != self.count
        {
            return Err(DeltaAssembleError::Replay);
        }
        let expected = u16::try_from(self.ops.len()).map_err(|_| DeltaAssembleError::TooMany)?;
        if page.index != expected {
            return Err(if page.index < expected {
                DeltaAssembleError::Replay
            } else {
                DeltaAssembleError::Gap
            });
        }
        self.ops
            .push(page.op)
            .map_err(|_| DeltaAssembleError::TooMany)
    }

    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.ops.len() == usize::from(self.count)
    }

    pub fn finish(self) -> Result<TopologyDelta, DeltaAssembleError> {
        if !self.is_complete() {
            return Err(DeltaAssembleError::Gap);
        }
        let mut changes = ChangeSetBuilder::new(self.parent);
        for op in &self.ops {
            changes
                .push(op.change.clone())
                .map_err(|_| DeltaAssembleError::Malformed)?;
        }
        let changes = changes
            .finish()
            .map_err(|_| DeltaAssembleError::Malformed)?;
        let delta = TopologyDelta::new(
            self.sequence,
            &changes,
            self.next,
            self.origin_transaction,
            self.ops.as_slice(),
        )
        .map_err(DeltaAssembleError::Semantic)?;
        if delta.id != self.delta_id {
            return Err(DeltaAssembleError::Replay);
        }
        Ok(delta)
    }
}

pub fn encode_delta_op(
    delta: &TopologyDelta,
    index: u16,
    output: &mut [u8],
) -> Result<usize, DeltaWireError> {
    let op = delta
        .ops()
        .get(usize::from(index))
        .ok_or(DeltaWireError::Malformed)?;
    let mut cursor = 0usize;
    put(output, &mut cursor, &DELTA_OP_BULK_MAGIC)?;
    put(output, &mut cursor, &FORMAT_VERSION.to_le_bytes())?;
    put(output, &mut cursor, &[0; 2])?;
    put(output, &mut cursor, &delta.sequence.to_le_bytes())?;
    put(output, &mut cursor, &delta.parent)?;
    put(output, &mut cursor, &delta.next)?;
    put(output, &mut cursor, &delta.id)?;
    put(output, &mut cursor, &delta.origin_transaction)?;
    put(output, &mut cursor, &index.to_le_bytes())?;
    put(
        output,
        &mut cursor,
        &(delta.ops().len() as u16).to_le_bytes(),
    )?;
    encode_op(op, output, &mut cursor)?;
    Ok(cursor)
}

pub fn decode_delta_op(input: &[u8]) -> Result<DeltaOpPage, DeltaWireError> {
    let mut cursor = 0usize;
    if input.len() < HEAD_BYTES + OP_HEAD_BYTES || input.len() > 1024 {
        return Err(DeltaWireError::TooLarge);
    }
    if take(input, &mut cursor, 4)? != DELTA_OP_BULK_MAGIC {
        return Err(DeltaWireError::Malformed);
    }
    if u16_at(take(input, &mut cursor, 2)?) != FORMAT_VERSION {
        return Err(DeltaWireError::UnsupportedVersion);
    }
    if take(input, &mut cursor, 2)? != [0; 2] {
        return Err(DeltaWireError::Malformed);
    }
    let sequence = u64_at(take(input, &mut cursor, 8)?);
    let parent = array(take(input, &mut cursor, 32)?)?;
    let next = array(take(input, &mut cursor, 32)?)?;
    let delta_id = array(take(input, &mut cursor, 32)?)?;
    let origin_transaction = array(take(input, &mut cursor, 32)?)?;
    let index = u16_at(take(input, &mut cursor, 2)?);
    let count = u16_at(take(input, &mut cursor, 2)?);
    if sequence == 0
        || delta_id == [0; 32]
        || count == 0
        || index >= count
        || usize::from(count) > MAX_CHANGES
    {
        return Err(DeltaWireError::Malformed);
    }
    let op = decode_op(input, &mut cursor)?;
    if cursor != input.len() {
        return Err(DeltaWireError::Malformed);
    }
    Ok(DeltaOpPage {
        sequence,
        parent,
        next,
        delta_id,
        origin_transaction,
        index,
        count,
        op,
    })
}

pub fn encode_delta(delta: &TopologyDelta, output: &mut [u8]) -> Result<usize, DeltaWireError> {
    if output.len() < HEAD_BYTES {
        return Err(DeltaWireError::TooSmall);
    }
    let mut cursor = 0usize;
    put(output, &mut cursor, &DELTA_BULK_MAGIC)?;
    put(output, &mut cursor, &FORMAT_VERSION.to_le_bytes())?;
    put(output, &mut cursor, &[0; 2])?;
    put(output, &mut cursor, &delta.sequence.to_le_bytes())?;
    put(output, &mut cursor, &delta.parent)?;
    put(output, &mut cursor, &delta.next)?;
    put(output, &mut cursor, &delta.id)?;
    put(output, &mut cursor, &delta.origin_transaction)?;
    put(
        output,
        &mut cursor,
        &(delta.ops().len() as u16).to_le_bytes(),
    )?;
    put(output, &mut cursor, &[0; 2])?;
    for op in delta.ops() {
        encode_op(op, output, &mut cursor)?;
    }
    Ok(cursor)
}

fn encode_op(op: &DeltaOp, output: &mut [u8], cursor: &mut usize) -> Result<(), DeltaWireError> {
    let path = op.change.path.as_bytes();
    let target = op
        .change
        .target
        .as_ref()
        .map_or(&[][..], |path| path.as_bytes());
    let replacement = op.replacement.as_ref();
    put(
        output,
        cursor,
        &[
            op.change.kind as u8,
            op.change.executable as u8,
            replacement.is_some() as u8,
            0,
        ],
    )?;
    put(output, cursor, &(path.len() as u16).to_le_bytes())?;
    put(output, cursor, &(target.len() as u16).to_le_bytes())?;
    put(output, cursor, &op.change.replacement_len.to_le_bytes())?;
    put(output, cursor, &op.change.expected)?;
    put(output, cursor, &op.change.replacement)?;
    let mut entry = [0; core::mem::size_of::<EntryHeader>()];
    if let Some(replacement) = replacement {
        entry.copy_from_slice(replacement.header(0).as_bytes());
    }
    put(output, cursor, &entry)?;
    put(output, cursor, path)?;
    put(output, cursor, target)
}

fn decode_op(input: &[u8], cursor: &mut usize) -> Result<DeltaOp, DeltaWireError> {
    let flags = take(input, cursor, 4)?;
    if flags[1] > 1 || flags[2] > 1 || flags[3] != 0 {
        return Err(DeltaWireError::Malformed);
    }
    let kind = match flags[0] {
        1 => ChangeKind::CreateDirectory,
        2 => ChangeKind::CreateFile,
        3 => ChangeKind::ModifyFile,
        4 => ChangeKind::Delete,
        5 => ChangeKind::Rename,
        _ => return Err(DeltaWireError::Malformed),
    };
    let path_len = usize::from(u16_at(take(input, cursor, 2)?));
    let target_len = usize::from(u16_at(take(input, cursor, 2)?));
    if path_len == 0 || path_len > MAX_PATH_BYTES || target_len > MAX_PATH_BYTES {
        return Err(DeltaWireError::Malformed);
    }
    let replacement_len = u32_at(take(input, cursor, 4)?);
    let expected = array(take(input, cursor, 32)?)?;
    let replacement_digest = array(take(input, cursor, 32)?)?;
    let entry_bytes = take(input, cursor, core::mem::size_of::<EntryHeader>())?;
    let path = crate::CanonicalPath::new(take(input, cursor, path_len)?)
        .map_err(|_| DeltaWireError::Malformed)?;
    let target = if target_len == 0 {
        None
    } else {
        Some(
            crate::CanonicalPath::new(take(input, cursor, target_len)?)
                .map_err(|_| DeltaWireError::Malformed)?,
        )
    };
    let change = Change {
        kind,
        path: path.clone(),
        target,
        expected,
        replacement: replacement_digest,
        replacement_len,
        executable: flags[1] != 0,
    };
    let replacement = if flags[2] == 0 {
        if entry_bytes.iter().any(|byte| *byte != 0) {
            return Err(DeltaWireError::Malformed);
        }
        None
    } else {
        let header =
            EntryHeader::read_from_bytes(entry_bytes).map_err(|_| DeltaWireError::Malformed)?;
        Some(Entry::from_wire(&header, path.as_bytes()).map_err(|_| DeltaWireError::Malformed)?)
    };
    Ok(DeltaOp {
        change,
        replacement,
    })
}

pub fn decode_delta(input: &[u8]) -> Result<TopologyDelta, DeltaWireError> {
    if input.len() < HEAD_BYTES || input.len() > MAX_DELTA_WIRE_BYTES {
        return Err(DeltaWireError::TooLarge);
    }
    let mut cursor = 0usize;
    if take(input, &mut cursor, 4)? != DELTA_BULK_MAGIC {
        return Err(DeltaWireError::Malformed);
    }
    if u16_at(take(input, &mut cursor, 2)?) != FORMAT_VERSION {
        return Err(DeltaWireError::UnsupportedVersion);
    }
    if take(input, &mut cursor, 2)? != [0; 2] {
        return Err(DeltaWireError::Malformed);
    }
    let sequence = u64_at(take(input, &mut cursor, 8)?);
    let parent = array(take(input, &mut cursor, 32)?)?;
    let next = array(take(input, &mut cursor, 32)?)?;
    let claimed_id = array(take(input, &mut cursor, 32)?)?;
    let origin = array(take(input, &mut cursor, 32)?)?;
    let count = usize::from(u16_at(take(input, &mut cursor, 2)?));
    if take(input, &mut cursor, 2)? != [0; 2] || count > MAX_CHANGES {
        return Err(DeltaWireError::Malformed);
    }
    let mut ops = Vec::<DeltaOp, MAX_CHANGES>::new();
    let mut changes = ChangeSetBuilder::new(parent);
    for _ in 0..count {
        if input.len().saturating_sub(cursor) < OP_HEAD_BYTES {
            return Err(DeltaWireError::TooSmall);
        }
        let flags = take(input, &mut cursor, 4)?;
        if flags[1] > 1 || flags[2] > 1 || flags[3] != 0 {
            return Err(DeltaWireError::Malformed);
        }
        let kind = match flags[0] {
            1 => ChangeKind::CreateDirectory,
            2 => ChangeKind::CreateFile,
            3 => ChangeKind::ModifyFile,
            4 => ChangeKind::Delete,
            5 => ChangeKind::Rename,
            _ => return Err(DeltaWireError::Malformed),
        };
        let path_len = usize::from(u16_at(take(input, &mut cursor, 2)?));
        let target_len = usize::from(u16_at(take(input, &mut cursor, 2)?));
        if path_len == 0 || path_len > MAX_PATH_BYTES || target_len > MAX_PATH_BYTES {
            return Err(DeltaWireError::Malformed);
        }
        let replacement_len = u32_at(take(input, &mut cursor, 4)?);
        let expected = array(take(input, &mut cursor, 32)?)?;
        let replacement_digest = array(take(input, &mut cursor, 32)?)?;
        let entry_bytes = take(input, &mut cursor, core::mem::size_of::<EntryHeader>())?;
        let path = crate::CanonicalPath::new(take(input, &mut cursor, path_len)?)
            .map_err(|_| DeltaWireError::Malformed)?;
        let target = if target_len == 0 {
            None
        } else {
            Some(
                crate::CanonicalPath::new(take(input, &mut cursor, target_len)?)
                    .map_err(|_| DeltaWireError::Malformed)?,
            )
        };
        let change = Change {
            kind,
            path: path.clone(),
            target,
            expected,
            replacement: replacement_digest,
            replacement_len,
            executable: flags[1] != 0,
        };
        changes
            .push(change.clone())
            .map_err(|_| DeltaWireError::Malformed)?;
        let replacement = if flags[2] == 0 {
            if entry_bytes.iter().any(|byte| *byte != 0) {
                return Err(DeltaWireError::Malformed);
            }
            None
        } else {
            let header =
                EntryHeader::read_from_bytes(entry_bytes).map_err(|_| DeltaWireError::Malformed)?;
            // Replacement entries share the change path, so it is encoded
            // only once and cannot disagree across two fields.
            Some(
                Entry::from_wire(&header, path.as_bytes())
                    .map_err(|_| DeltaWireError::Malformed)?,
            )
        };
        ops.push(DeltaOp {
            change,
            replacement,
        })
        .map_err(|_| DeltaWireError::TooLarge)?;
    }
    if cursor != input.len() {
        return Err(DeltaWireError::Malformed);
    }
    let changes = changes.finish().map_err(|_| DeltaWireError::Malformed)?;
    let delta = TopologyDelta::new(sequence, &changes, next, origin, ops.as_slice())
        .map_err(DeltaWireError::Semantic)?;
    if delta.id != claimed_id {
        return Err(DeltaWireError::Malformed);
    }
    Ok(delta)
}

fn put(output: &mut [u8], cursor: &mut usize, bytes: &[u8]) -> Result<(), DeltaWireError> {
    let end = cursor
        .checked_add(bytes.len())
        .ok_or(DeltaWireError::TooLarge)?;
    let destination = output
        .get_mut(*cursor..end)
        .ok_or(DeltaWireError::TooSmall)?;
    destination.copy_from_slice(bytes);
    *cursor = end;
    Ok(())
}

fn take<'a>(input: &'a [u8], cursor: &mut usize, len: usize) -> Result<&'a [u8], DeltaWireError> {
    let end = cursor.checked_add(len).ok_or(DeltaWireError::TooLarge)?;
    let bytes = input.get(*cursor..end).ok_or(DeltaWireError::TooSmall)?;
    *cursor = end;
    Ok(bytes)
}

fn u16_at(bytes: &[u8]) -> u16 {
    u16::from_le_bytes([bytes[0], bytes[1]])
}

fn u32_at(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().expect("length checked by caller"))
}

fn u64_at(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().expect("length checked by caller"))
}

fn array<const N: usize>(bytes: &[u8]) -> Result<[u8; N], DeltaWireError> {
    bytes.try_into().map_err(|_| DeltaWireError::Malformed)
}

const _: () = assert!(
    HEAD_BYTES + MAX_CHANGES * (OP_HEAD_BYTES + 2 * MAX_PATH_BYTES) <= MAX_DELTA_WIRE_BYTES
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CanonicalPath, ChangeSetBuilder, Entry, Policy, TopologyDelta};

    #[test]
    fn complete_delta_round_trips_and_trailing_or_mutated_bytes_fail() {
        let path = CanonicalPath::new(b"file").unwrap();
        let replacement = Entry::file(path.clone(), false, 3, [7; 32], [8; 16]).unwrap();
        let change = Change {
            kind: ChangeKind::CreateFile,
            path,
            target: None,
            expected: [0; 32],
            replacement: [7; 32],
            replacement_len: 3,
            executable: false,
        };
        let mut builder = ChangeSetBuilder::new([1; 32]);
        builder.push(change.clone()).unwrap();
        let changes = builder.finish().unwrap();
        let delta = TopologyDelta::new(
            1,
            &changes,
            [2; 32],
            [0; 32],
            &[DeltaOp {
                change,
                replacement: Some(replacement),
            }],
        )
        .unwrap();
        let mut bytes = [0; MAX_DELTA_WIRE_BYTES];
        let len = encode_delta(&delta, &mut bytes).unwrap();
        let decoded = decode_delta(&bytes[..len]).unwrap();
        assert_eq!(decoded.id, delta.id);
        assert_eq!(decoded.ops()[0].replacement, delta.ops()[0].replacement);
        assert!(matches!(
            decode_delta(&bytes[..len + 1]),
            Err(DeltaWireError::Malformed)
        ));
        bytes[12] ^= 1;
        assert!(decode_delta(&bytes[..len]).is_err());
        let _ = Policy::default();
    }

    #[test]
    fn paged_receiver_refuses_gaps_replay_and_header_equivocation() {
        std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(paged_receiver_case)
            .unwrap()
            .join()
            .unwrap();
    }

    fn paged_receiver_case() {
        let path = CanonicalPath::new(b"file").unwrap();
        let replacement = Entry::file(path.clone(), false, 3, [7; 32], [8; 16]).unwrap();
        let change = Change {
            kind: ChangeKind::CreateFile,
            path,
            target: None,
            expected: [0; 32],
            replacement: [7; 32],
            replacement_len: 3,
            executable: false,
        };
        let mut builder = ChangeSetBuilder::new([1; 32]);
        builder.push(change.clone()).unwrap();
        let changes = builder.finish().unwrap();
        let delta = TopologyDelta::new(
            1,
            &changes,
            [2; 32],
            [0; 32],
            &[DeltaOp {
                change,
                replacement: Some(replacement),
            }],
        )
        .unwrap();
        let mut bytes = [0; 1024];
        let len = encode_delta_op(&delta, 0, &mut bytes).unwrap();
        let page = decode_delta_op(&bytes[..len]).unwrap();
        let assembler = DeltaAssembler::begin(page.clone()).unwrap();
        assert_eq!(assembler.finish().unwrap().id, delta.id);

        let mut replay = DeltaAssembler::begin(page.clone()).unwrap();
        assert_eq!(replay.push(page.clone()), Err(DeltaAssembleError::Replay));
        let mut gap = page.clone();
        gap.index = 1;
        assert!(matches!(
            DeltaAssembler::begin(gap),
            Err(DeltaAssembleError::Gap)
        ));
        let mut equivocation = page;
        equivocation.delta_id[0] ^= 1;
        assert!(matches!(
            DeltaAssembler::begin(equivocation).unwrap().finish(),
            Err(DeltaAssembleError::Replay)
        ));
    }
}
