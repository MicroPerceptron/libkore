//! Bounded, preconditioned export change sets.

use heapless::Vec;

use crate::{
    bounds::MAX_CHANGES,
    path::CanonicalPath,
    wire::{ContentDigest, SnapshotId},
};

const CHANGESET_DOMAIN: &[u8] = b"kore.workspace.changes.v1\0";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ChangeKind {
    CreateDirectory = 1,
    CreateFile = 2,
    ModifyFile = 3,
    Delete = 4,
    Rename = 5,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Change {
    pub kind: ChangeKind,
    pub path: CanonicalPath,
    pub target: Option<CanonicalPath>,
    pub expected: ContentDigest,
    pub replacement: ContentDigest,
    pub replacement_len: u32,
    pub executable: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChangeSetError {
    Full,
    DuplicateTarget,
    PortableCollision,
    InvalidShape,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChangeSet {
    pub base: SnapshotId,
    pub id: ContentDigest,
    changes: Vec<Change, MAX_CHANGES>,
}

impl ChangeSet {
    pub fn changes(&self) -> &[Change] {
        self.changes.as_slice()
    }

    pub fn verify(&self) -> Result<(), ChangeSetError> {
        let mut builder = ChangeSetBuilder::new(self.base);
        for change in &self.changes {
            builder.push(change.clone())?;
        }
        let rebuilt = builder.finish()?;
        (rebuilt.id == self.id)
            .then_some(())
            .ok_or(ChangeSetError::InvalidShape)
    }
}

pub struct ChangeSetBuilder {
    base: SnapshotId,
    changes: Vec<Change, MAX_CHANGES>,
}

impl ChangeSetBuilder {
    pub const fn new(base: SnapshotId) -> Self {
        Self {
            base,
            changes: Vec::new(),
        }
    }

    pub fn push(&mut self, change: Change) -> Result<(), ChangeSetError> {
        validate_change(&change)?;
        for earlier in &self.changes {
            if earlier.path == change.path
                || earlier.target.as_ref() == Some(&change.path)
                || change.target.as_ref() == Some(&earlier.path)
                || earlier
                    .target
                    .as_ref()
                    .is_some_and(|left| change.target.as_ref().is_some_and(|right| left == right))
            {
                return Err(ChangeSetError::DuplicateTarget);
            }
            if earlier.path.portable_eq(&change.path)
                || earlier
                    .target
                    .as_ref()
                    .is_some_and(|target| target.portable_eq(&change.path))
                || change
                    .target
                    .as_ref()
                    .is_some_and(|target| target.portable_eq(&earlier.path))
                || earlier.target.as_ref().is_some_and(|left| {
                    change
                        .target
                        .as_ref()
                        .is_some_and(|right| left.portable_eq(right))
                })
            {
                return Err(ChangeSetError::PortableCollision);
            }
        }
        self.changes.push(change).map_err(|_| ChangeSetError::Full)
    }

    pub fn finish(mut self) -> Result<ChangeSet, ChangeSetError> {
        self.changes
            .as_mut_slice()
            .sort_unstable_by(|left, right| left.path.cmp(&right.path));
        let mut hash = blake3::Hasher::new();
        hash.update(CHANGESET_DOMAIN);
        hash.update(&self.base);
        hash.update(&(self.changes.len() as u16).to_le_bytes());
        for change in &self.changes {
            hash.update(&[change.kind as u8, change.executable as u8]);
            hash.update(&(change.path.as_bytes().len() as u16).to_le_bytes());
            hash.update(change.path.as_bytes());
            let target = change
                .target
                .as_ref()
                .map_or(&[][..], |path| path.as_bytes());
            hash.update(&(target.len() as u16).to_le_bytes());
            hash.update(target);
            hash.update(&change.expected);
            hash.update(&change.replacement);
            hash.update(&change.replacement_len.to_le_bytes());
        }
        Ok(ChangeSet {
            base: self.base,
            id: *hash.finalize().as_bytes(),
            changes: self.changes,
        })
    }
}

fn validate_change(change: &Change) -> Result<(), ChangeSetError> {
    let zero = [0; 32];
    match change.kind {
        ChangeKind::Rename if change.target.is_none() => Err(ChangeSetError::InvalidShape),
        ChangeKind::Rename if change.replacement_len != 0 => Err(ChangeSetError::InvalidShape),
        ChangeKind::Delete | ChangeKind::CreateDirectory
            if change.target.is_some() || change.replacement_len != 0 =>
        {
            Err(ChangeSetError::InvalidShape)
        }
        ChangeKind::CreateFile | ChangeKind::ModifyFile if change.target.is_some() => {
            Err(ChangeSetError::InvalidShape)
        }
        ChangeKind::CreateDirectory
            if change.expected != zero || change.replacement != zero || change.executable =>
        {
            Err(ChangeSetError::InvalidShape)
        }
        ChangeKind::CreateFile if change.expected != zero => Err(ChangeSetError::InvalidShape),
        ChangeKind::ModifyFile if change.expected == zero => Err(ChangeSetError::InvalidShape),
        ChangeKind::Delete
            if change.expected == zero || change.replacement != zero || change.executable =>
        {
            Err(ChangeSetError::InvalidShape)
        }
        ChangeKind::Rename
            if change.expected == zero || change.replacement != zero || change.executable =>
        {
            Err(ChangeSetError::InvalidShape)
        }
        ChangeKind::Rename
            if change
                .target
                .as_ref()
                .is_some_and(|target| target == &change.path) =>
        {
            Err(ChangeSetError::InvalidShape)
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn change_identity_commits_preconditions_and_targets() {
        let path = CanonicalPath::new(b"a.txt").unwrap();
        let mut first = ChangeSetBuilder::new([1; 32]);
        first
            .push(Change {
                kind: ChangeKind::ModifyFile,
                path: path.clone(),
                target: None,
                expected: [2; 32],
                replacement: [3; 32],
                replacement_len: 4,
                executable: false,
            })
            .unwrap();
        let a = first.finish().unwrap();

        let mut second = ChangeSetBuilder::new([1; 32]);
        second
            .push(Change {
                expected: [9; 32],
                ..a.changes()[0].clone()
            })
            .unwrap();
        assert_ne!(a.id, second.finish().unwrap().id);
    }
}
