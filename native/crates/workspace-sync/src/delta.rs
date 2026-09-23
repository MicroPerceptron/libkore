//! Ordered canonical topology transactions for the live backend.

use heapless::Vec;

use crate::{
    Change, ChangeKind, ChangeSet, Entry, EntryKind, Manifest, ManifestBuilder, Policy,
    bounds::{MAX_CHANGES, MAX_DELTA_HISTORY},
    wire::{ContentDigest, SnapshotId},
};

const DELTA_DOMAIN: &[u8] = b"kore.workspace.delta.v1\0";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeltaOp {
    pub change: Change,
    /// Required for create/modify, absent for delete/rename.
    pub replacement: Option<Entry>,
}

#[derive(Clone, Debug)]
pub struct TopologyDelta {
    pub sequence: u64,
    pub parent: SnapshotId,
    pub next: SnapshotId,
    pub id: ContentDigest,
    pub origin_transaction: ContentDigest,
    ops: Vec<DeltaOp, MAX_CHANGES>,
}

impl TopologyDelta {
    pub fn new(
        sequence: u64,
        changes: &ChangeSet,
        next: SnapshotId,
        origin_transaction: ContentDigest,
        ops: &[DeltaOp],
    ) -> Result<Self, DeltaError> {
        if sequence == 0
            || changes.base == [0; 32]
            || next == [0; 32]
            || ops.len() != changes.changes().len()
        {
            return Err(DeltaError::Malformed);
        }
        for (op, change) in ops.iter().zip(changes.changes()) {
            if &op.change != change || !replacement_shape(op) {
                return Err(DeltaError::Malformed);
            }
        }
        let ops = Vec::from_slice(ops).map_err(|_| DeltaError::TooMany)?;
        let id = delta_id(sequence, changes.id, next, origin_transaction);
        Ok(Self {
            sequence,
            parent: changes.base,
            next,
            id,
            origin_transaction,
            ops,
        })
    }

    #[must_use]
    pub fn ops(&self) -> &[DeltaOp] {
        self.ops.as_slice()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeltaError {
    Malformed,
    TooMany,
    Gap,
    ReplayConflict,
    WrongParent,
    Preconditions,
    DigestMismatch,
    Echo,
    HistoryFull,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeltaApply {
    Applied,
    Duplicate,
}

pub struct TopologyState {
    current: Manifest,
    policy: Policy,
    next_sequence: u64,
    applied: Vec<(u64, ContentDigest), MAX_DELTA_HISTORY>,
    local_transactions: Vec<ContentDigest, MAX_DELTA_HISTORY>,
}

impl TopologyState {
    pub fn new(current: Manifest, policy: Policy, next_sequence: u64) -> Result<Self, DeltaError> {
        current.verify(policy).map_err(|_| DeltaError::Malformed)?;
        if next_sequence == 0 {
            return Err(DeltaError::Malformed);
        }
        Ok(Self {
            current,
            policy,
            next_sequence,
            applied: Vec::new(),
            local_transactions: Vec::new(),
        })
    }

    #[must_use]
    pub const fn manifest(&self) -> &Manifest {
        &self.current
    }

    #[must_use]
    pub const fn next_sequence(&self) -> u64 {
        self.next_sequence
    }

    pub fn mark_local_transaction(&mut self, id: ContentDigest) -> Result<(), DeltaError> {
        if id == [0; 32] || self.local_transactions.contains(&id) {
            return Ok(());
        }
        if self.local_transactions.len() == MAX_DELTA_HISTORY {
            self.local_transactions.remove(0);
        }
        self.local_transactions
            .push(id)
            .map_err(|_| DeltaError::HistoryFull)
    }

    pub fn apply(&mut self, delta: &TopologyDelta) -> Result<DeltaApply, DeltaError> {
        if let Some((_, known)) = self
            .applied
            .iter()
            .find(|(sequence, _)| *sequence == delta.sequence)
        {
            return if *known == delta.id {
                Ok(DeltaApply::Duplicate)
            } else {
                Err(DeltaError::ReplayConflict)
            };
        }
        if delta.sequence != self.next_sequence {
            return Err(DeltaError::Gap);
        }
        if delta.parent != self.current.snapshot_id() {
            return Err(DeltaError::WrongParent);
        }
        if delta.origin_transaction != [0; 32]
            && self.local_transactions.contains(&delta.origin_transaction)
        {
            return Err(DeltaError::Echo);
        }
        let rebuilt = apply_ops(&self.current, self.policy, delta.ops())?;
        if rebuilt.snapshot_id() != delta.next {
            return Err(DeltaError::DigestMismatch);
        }
        self.current = rebuilt;
        if self.applied.len() == MAX_DELTA_HISTORY {
            self.applied.remove(0);
        }
        self.applied
            .push((delta.sequence, delta.id))
            .map_err(|_| DeltaError::HistoryFull)?;
        self.next_sequence = self.next_sequence.checked_add(1).ok_or(DeltaError::Gap)?;
        Ok(DeltaApply::Applied)
    }
}

fn replacement_shape(op: &DeltaOp) -> bool {
    match op.change.kind {
        ChangeKind::CreateDirectory => op.replacement.as_ref().is_some_and(|entry| {
            entry.kind == EntryKind::Directory && entry.path == op.change.path
        }),
        ChangeKind::CreateFile | ChangeKind::ModifyFile => {
            op.replacement.as_ref().is_some_and(|entry| {
                entry.kind == EntryKind::File
                    && entry.path == op.change.path
                    && entry.digest == op.change.replacement
                    && entry.content_len == op.change.replacement_len
                    && entry.flags.executable() == op.change.executable
            })
        }
        ChangeKind::Delete | ChangeKind::Rename => op.replacement.is_none(),
    }
}

fn apply_ops(current: &Manifest, policy: Policy, ops: &[DeltaOp]) -> Result<Manifest, DeltaError> {
    let mut entries = Vec::<Entry, { crate::bounds::MAX_ENTRIES }>::from_slice(current.entries())
        .map_err(|_| DeltaError::TooMany)?;
    for op in ops {
        let at = entries
            .iter()
            .position(|entry| entry.path == op.change.path);
        match op.change.kind {
            ChangeKind::CreateDirectory | ChangeKind::CreateFile => {
                if at.is_some() {
                    return Err(DeltaError::Preconditions);
                }
                entries
                    .push(op.replacement.clone().ok_or(DeltaError::Malformed)?)
                    .map_err(|_| DeltaError::TooMany)?;
            }
            ChangeKind::ModifyFile => {
                let index = at.ok_or(DeltaError::Preconditions)?;
                if entries[index].kind != EntryKind::File
                    || entries[index].digest != op.change.expected
                {
                    return Err(DeltaError::Preconditions);
                }
                entries[index] = op.replacement.clone().ok_or(DeltaError::Malformed)?;
            }
            ChangeKind::Delete => {
                let index = at.ok_or(DeltaError::Preconditions)?;
                if entries[index].digest != op.change.expected {
                    return Err(DeltaError::Preconditions);
                }
                entries.swap_remove(index);
            }
            ChangeKind::Rename => {
                let index = at.ok_or(DeltaError::Preconditions)?;
                if entries[index].digest != op.change.expected {
                    return Err(DeltaError::Preconditions);
                }
                let target = op.change.target.clone().ok_or(DeltaError::Malformed)?;
                if entries.iter().any(|entry| entry.path == target) {
                    return Err(DeltaError::Preconditions);
                }
                entries[index].path = target;
            }
        }
    }
    let mut builder = ManifestBuilder::new(current.snapshot_id(), policy);
    for entry in entries {
        builder.push(entry).map_err(|_| DeltaError::Malformed)?;
    }
    builder.finish().map_err(|_| DeltaError::Malformed)
}

fn delta_id(
    sequence: u64,
    changes: ContentDigest,
    next: SnapshotId,
    origin: ContentDigest,
) -> ContentDigest {
    let mut hash = blake3::Hasher::new();
    hash.update(DELTA_DOMAIN);
    hash.update(&sequence.to_le_bytes());
    hash.update(&changes);
    hash.update(&next);
    hash.update(&origin);
    *hash.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CanonicalPath, ChangeSetBuilder, EntryFlags};

    fn base() -> Manifest {
        let mut builder = ManifestBuilder::new([0; 32], Policy::default());
        builder
            .push(
                Entry::file(
                    CanonicalPath::new(b"a").unwrap(),
                    false,
                    1,
                    [1; 32],
                    [1; 16],
                )
                .unwrap(),
            )
            .unwrap();
        builder.finish().unwrap()
    }

    fn modify_delta(base: &Manifest, sequence: u64, origin: [u8; 32]) -> TopologyDelta {
        let replacement = Entry {
            path: CanonicalPath::new(b"a").unwrap(),
            kind: EntryKind::File,
            flags: EntryFlags::default(),
            content_len: 2,
            digest: [2; 32],
            range_root: [3; 32],
            source: [2; 16],
        };
        let change = Change {
            kind: ChangeKind::ModifyFile,
            path: replacement.path.clone(),
            target: None,
            expected: [1; 32],
            replacement: [2; 32],
            replacement_len: 2,
            executable: false,
        };
        let mut changes = ChangeSetBuilder::new(base.snapshot_id());
        changes.push(change.clone()).unwrap();
        let changes = changes.finish().unwrap();
        let next = apply_ops(
            base,
            Policy::default(),
            &[DeltaOp {
                change: change.clone(),
                replacement: Some(replacement.clone()),
            }],
        )
        .unwrap();
        TopologyDelta::new(
            sequence,
            &changes,
            next.snapshot_id(),
            origin,
            &[DeltaOp {
                change,
                replacement: Some(replacement),
            }],
        )
        .unwrap()
    }

    #[test]
    fn ordered_apply_duplicate_and_conflicting_replay_are_distinct() {
        let base = base();
        let delta = modify_delta(&base, 1, [0; 32]);
        let mut state = TopologyState::new(base, Policy::default(), 1).unwrap();
        assert_eq!(state.apply(&delta), Ok(DeltaApply::Applied));
        assert_eq!(state.apply(&delta), Ok(DeltaApply::Duplicate));
        let mut forged = delta.clone();
        forged.id[0] ^= 1;
        assert_eq!(state.apply(&forged), Err(DeltaError::ReplayConflict));
    }

    #[test]
    fn gaps_and_echoed_local_transactions_fail_closed() {
        let base = base();
        let gap = modify_delta(&base, 2, [0; 32]);
        let mut state = TopologyState::new(base.clone(), Policy::default(), 1).unwrap();
        assert_eq!(state.apply(&gap), Err(DeltaError::Gap));
        state.mark_local_transaction([8; 32]).unwrap();
        let echo = modify_delta(&base, 1, [8; 32]);
        assert_eq!(state.apply(&echo), Err(DeltaError::Echo));
    }
}
