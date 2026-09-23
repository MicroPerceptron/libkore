//! Bounded metadata wire for automatic copy-on-write transactions.
//!
//! File bytes travel as separate digest-addressed bulk objects so an exact
//! 1 MiB file still fits the founded HostLink profile. This record contains
//! only canonical kore paths, preconditions, and replacement digests.

use crate::{
    Change, ChangeKind, ChangeSet, ChangeSetBuilder,
    bounds::{FORMAT_VERSION, MAX_CHANGES, MAX_PATH_BYTES},
};

pub const WRITE_TRANSACTION_MAGIC: [u8; 4] = *b"KWTX";
pub const WORKSPACE_BLOB_CONTENT: u16 = 0x5701;
pub const WORKSPACE_TRANSACTION_CONTENT: u16 = 0x5702;
pub const MAX_WRITE_TRANSACTION_BYTES: usize = 64 * 1024;
const HEAD_BYTES: usize = 124;
const OP_HEAD_BYTES: usize = 76;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WriteTransaction {
    pub id: [u8; 32],
    pub grant: [u8; 16],
    pub changes: ChangeSet,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WriteWireError {
    TooSmall,
    TooLarge,
    Malformed,
    UnsupportedVersion,
}

pub fn encode_write_transaction(
    transaction: &WriteTransaction,
    output: &mut [u8],
) -> Result<usize, WriteWireError> {
    transaction
        .changes
        .verify()
        .map_err(|_| WriteWireError::Malformed)?;
    if transaction.id == [0; 32] || transaction.grant == [0; 16] {
        return Err(WriteWireError::Malformed);
    }
    let mut cursor = 0usize;
    put(output, &mut cursor, &WRITE_TRANSACTION_MAGIC)?;
    put(output, &mut cursor, &FORMAT_VERSION.to_le_bytes())?;
    put(output, &mut cursor, &[0; 2])?;
    put(output, &mut cursor, &transaction.id)?;
    put(output, &mut cursor, &transaction.grant)?;
    put(output, &mut cursor, &transaction.changes.base)?;
    put(output, &mut cursor, &transaction.changes.id)?;
    put(
        output,
        &mut cursor,
        &(transaction.changes.changes().len() as u16).to_le_bytes(),
    )?;
    put(output, &mut cursor, &[0; 2])?;
    for change in transaction.changes.changes() {
        let path = change.path.as_bytes();
        let target = change
            .target
            .as_ref()
            .map_or(&[][..], |path| path.as_bytes());
        put(
            output,
            &mut cursor,
            &[change.kind as u8, change.executable as u8],
        )?;
        put(output, &mut cursor, &(path.len() as u16).to_le_bytes())?;
        put(output, &mut cursor, &(target.len() as u16).to_le_bytes())?;
        put(output, &mut cursor, &change.replacement_len.to_le_bytes())?;
        put(output, &mut cursor, &change.expected)?;
        put(output, &mut cursor, &change.replacement)?;
        put(output, &mut cursor, path)?;
        put(output, &mut cursor, target)?;
    }
    Ok(cursor)
}

pub fn decode_write_transaction(input: &[u8]) -> Result<WriteTransaction, WriteWireError> {
    if input.len() < HEAD_BYTES || input.len() > MAX_WRITE_TRANSACTION_BYTES {
        return Err(WriteWireError::TooLarge);
    }
    let mut cursor = 0usize;
    if take(input, &mut cursor, 4)? != WRITE_TRANSACTION_MAGIC {
        return Err(WriteWireError::Malformed);
    }
    if u16_at(take(input, &mut cursor, 2)?) != FORMAT_VERSION {
        return Err(WriteWireError::UnsupportedVersion);
    }
    if take(input, &mut cursor, 2)? != [0; 2] {
        return Err(WriteWireError::Malformed);
    }
    let id = array(take(input, &mut cursor, 32)?)?;
    let grant = array(take(input, &mut cursor, 16)?)?;
    let base = array(take(input, &mut cursor, 32)?)?;
    let claimed_id = array(take(input, &mut cursor, 32)?)?;
    let count = usize::from(u16_at(take(input, &mut cursor, 2)?));
    if take(input, &mut cursor, 2)? != [0; 2]
        || id == [0; 32]
        || grant == [0; 16]
        || base == [0; 32]
        || claimed_id == [0; 32]
        || count == 0
        || count > MAX_CHANGES
    {
        return Err(WriteWireError::Malformed);
    }
    let mut builder = ChangeSetBuilder::new(base);
    for _ in 0..count {
        let flags = take(input, &mut cursor, 2)?;
        if flags[1] > 1 {
            return Err(WriteWireError::Malformed);
        }
        let kind = match flags[0] {
            1 => ChangeKind::CreateDirectory,
            2 => ChangeKind::CreateFile,
            3 => ChangeKind::ModifyFile,
            4 => ChangeKind::Delete,
            5 => ChangeKind::Rename,
            _ => return Err(WriteWireError::Malformed),
        };
        let path_len = usize::from(u16_at(take(input, &mut cursor, 2)?));
        let target_len = usize::from(u16_at(take(input, &mut cursor, 2)?));
        if path_len == 0 || path_len > MAX_PATH_BYTES || target_len > MAX_PATH_BYTES {
            return Err(WriteWireError::Malformed);
        }
        let replacement_len = u32_at(take(input, &mut cursor, 4)?);
        let expected = array(take(input, &mut cursor, 32)?)?;
        let replacement = array(take(input, &mut cursor, 32)?)?;
        let path = crate::CanonicalPath::new(take(input, &mut cursor, path_len)?)
            .map_err(|_| WriteWireError::Malformed)?;
        let target = if target_len == 0 {
            None
        } else {
            Some(
                crate::CanonicalPath::new(take(input, &mut cursor, target_len)?)
                    .map_err(|_| WriteWireError::Malformed)?,
            )
        };
        builder
            .push(Change {
                kind,
                path,
                target,
                expected,
                replacement,
                replacement_len,
                executable: flags[1] != 0,
            })
            .map_err(|_| WriteWireError::Malformed)?;
    }
    if cursor != input.len() {
        return Err(WriteWireError::Malformed);
    }
    let changes = builder.finish().map_err(|_| WriteWireError::Malformed)?;
    if changes.id != claimed_id {
        return Err(WriteWireError::Malformed);
    }
    Ok(WriteTransaction { id, grant, changes })
}

fn put(output: &mut [u8], cursor: &mut usize, bytes: &[u8]) -> Result<(), WriteWireError> {
    let end = cursor
        .checked_add(bytes.len())
        .ok_or(WriteWireError::TooLarge)?;
    output
        .get_mut(*cursor..end)
        .ok_or(WriteWireError::TooSmall)?
        .copy_from_slice(bytes);
    *cursor = end;
    Ok(())
}

fn take<'a>(input: &'a [u8], cursor: &mut usize, len: usize) -> Result<&'a [u8], WriteWireError> {
    let end = cursor.checked_add(len).ok_or(WriteWireError::TooLarge)?;
    let bytes = input.get(*cursor..end).ok_or(WriteWireError::TooSmall)?;
    *cursor = end;
    Ok(bytes)
}

fn u16_at(bytes: &[u8]) -> u16 {
    u16::from_le_bytes([bytes[0], bytes[1]])
}

fn u32_at(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().expect("length checked by caller"))
}

fn array<const N: usize>(bytes: &[u8]) -> Result<[u8; N], WriteWireError> {
    bytes.try_into().map_err(|_| WriteWireError::Malformed)
}

const _: () = assert!(
    HEAD_BYTES + MAX_CHANGES * (OP_HEAD_BYTES + 2 * MAX_PATH_BYTES) <= MAX_WRITE_TRANSACTION_BYTES
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_metadata_round_trips_and_commits_preconditions() {
        let mut builder = ChangeSetBuilder::new([1; 32]);
        builder
            .push(Change {
                kind: ChangeKind::Rename,
                path: crate::CanonicalPath::new(b"from").unwrap(),
                target: Some(crate::CanonicalPath::new(b"to").unwrap()),
                expected: [2; 32],
                replacement: [0; 32],
                replacement_len: 0,
                executable: false,
            })
            .unwrap();
        let transaction = WriteTransaction {
            id: [3; 32],
            grant: [4; 16],
            changes: builder.finish().unwrap(),
        };
        let mut bytes = [0; MAX_WRITE_TRANSACTION_BYTES];
        let len = encode_write_transaction(&transaction, &mut bytes).unwrap();
        assert_eq!(decode_write_transaction(&bytes[..len]), Ok(transaction));
        bytes[len - 1] ^= 1;
        assert!(decode_write_transaction(&bytes[..len]).is_err());
    }
}
