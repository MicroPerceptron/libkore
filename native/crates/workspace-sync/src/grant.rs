//! Standing automatic-writeback authority for one exact workspace lineage.

use heapless::Vec;

use crate::{
    CanonicalPath, ChangeKind, ChangeSet,
    bounds::{MAX_CHANGES, MAX_GRANT_EXCLUSIONS, MAX_OVERLAY_BYTES},
    wire::SnapshotId,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GrantOperations(u8);

impl GrantOperations {
    pub const CREATE: u8 = 1 << 0;
    pub const MODIFY: u8 = 1 << 1;
    pub const RENAME: u8 = 1 << 2;
    pub const DELETE: u8 = 1 << 3;
    pub const ALL: Self = Self(Self::CREATE | Self::MODIFY | Self::RENAME | Self::DELETE);

    #[must_use]
    pub const fn from_bits(bits: u8) -> Option<Self> {
        if bits & !Self::ALL.0 == 0 {
            Some(Self(bits))
        } else {
            None
        }
    }

    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    fn permits(self, kind: ChangeKind) -> bool {
        let required = match kind {
            ChangeKind::CreateDirectory | ChangeKind::CreateFile => Self::CREATE,
            ChangeKind::ModifyFile => Self::MODIFY,
            ChangeKind::Rename => Self::RENAME,
            ChangeKind::Delete => Self::DELETE,
        };
        self.0 & required != 0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GrantError {
    Invalid,
    Revoked,
    Expired,
    WrongRoot,
    OperationDenied,
    Excluded,
    TooMany,
    TooLarge,
    ArithmeticOverflow,
}

#[derive(Clone, Debug)]
pub struct StandingGrant {
    id: [u8; 16],
    root: SnapshotId,
    operations: GrantOperations,
    max_operations: u16,
    max_bytes: u64,
    expires_at: u64,
    exclusions: Vec<CanonicalPath, MAX_GRANT_EXCLUSIONS>,
    revoked: bool,
}

impl StandingGrant {
    pub fn new(
        id: [u8; 16],
        root: SnapshotId,
        operations: GrantOperations,
        max_operations: u16,
        max_bytes: u64,
        expires_at: u64,
        exclusions: &[CanonicalPath],
    ) -> Result<Self, GrantError> {
        if id == [0; 16]
            || root == [0; 32]
            || max_operations == 0
            || usize::from(max_operations) > MAX_CHANGES
            || max_bytes == 0
            || max_bytes > MAX_OVERLAY_BYTES
        {
            return Err(GrantError::Invalid);
        }
        let exclusions = Vec::from_slice(exclusions).map_err(|_| GrantError::TooMany)?;
        Ok(Self {
            id,
            root,
            operations,
            max_operations,
            max_bytes,
            expires_at,
            exclusions,
            revoked: false,
        })
    }

    #[must_use]
    pub const fn id(&self) -> [u8; 16] {
        self.id
    }

    #[must_use]
    pub const fn root(&self) -> SnapshotId {
        self.root
    }

    #[must_use]
    pub const fn operations(&self) -> GrantOperations {
        self.operations
    }

    #[must_use]
    pub const fn max_operations(&self) -> u16 {
        self.max_operations
    }

    #[must_use]
    pub const fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    #[must_use]
    pub const fn expires_at(&self) -> u64 {
        self.expires_at
    }

    pub fn revoke(&mut self) {
        self.revoked = true;
    }

    /// Advance the exact synchronized lineage after one fully applied,
    /// preconditioned transaction. This cannot change the operation mask,
    /// exclusions, limits, expiry, or revocation state.
    pub fn advance(&mut self, expected: SnapshotId, next: SnapshotId) -> Result<(), GrantError> {
        if self.revoked {
            return Err(GrantError::Revoked);
        }
        if self.root != expected || next == [0; 32] || next == expected {
            return Err(GrantError::WrongRoot);
        }
        self.root = next;
        Ok(())
    }

    pub fn authorize(&self, changes: &ChangeSet, now: u64) -> Result<(), GrantError> {
        if self.revoked {
            return Err(GrantError::Revoked);
        }
        if self.expires_at != 0 && now >= self.expires_at {
            return Err(GrantError::Expired);
        }
        if changes.base != self.root {
            return Err(GrantError::WrongRoot);
        }
        if changes.changes().len() > usize::from(self.max_operations) {
            return Err(GrantError::TooMany);
        }
        let mut bytes = 0u64;
        for change in changes.changes() {
            if !self.operations.permits(change.kind) {
                return Err(GrantError::OperationDenied);
            }
            if self
                .exclusions
                .iter()
                .any(|excluded| change.path.is_within(excluded))
                || change.target.as_ref().is_some_and(|target| {
                    self.exclusions
                        .iter()
                        .any(|excluded| target.is_within(excluded))
                })
            {
                return Err(GrantError::Excluded);
            }
            bytes = bytes
                .checked_add(u64::from(change.replacement_len))
                .ok_or(GrantError::ArithmeticOverflow)?;
            if bytes > self.max_bytes {
                return Err(GrantError::TooLarge);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Change, ChangeSetBuilder};

    fn modify(path: &str, bytes: u32) -> ChangeSet {
        let mut builder = ChangeSetBuilder::new([1; 32]);
        builder
            .push(Change {
                kind: ChangeKind::ModifyFile,
                path: CanonicalPath::new(path.as_bytes()).unwrap(),
                target: None,
                expected: [2; 32],
                replacement: [3; 32],
                replacement_len: bytes,
                executable: false,
            })
            .unwrap();
        builder.finish().unwrap()
    }

    #[test]
    fn exact_root_operation_exclusion_expiry_and_revocation_are_enforced() {
        let excluded = CanonicalPath::new(b"private").unwrap();
        let mut grant = StandingGrant::new(
            [9; 16],
            [1; 32],
            GrantOperations::ALL,
            4,
            4096,
            10,
            &[excluded],
        )
        .unwrap();
        assert_eq!(grant.authorize(&modify("public/a", 7), 9), Ok(()));
        assert_eq!(
            grant.authorize(&modify("private/a", 7), 9),
            Err(GrantError::Excluded)
        );
        assert_eq!(
            grant.authorize(&modify("public/a", 7), 10),
            Err(GrantError::Expired)
        );
        grant.revoke();
        assert_eq!(
            grant.authorize(&modify("public/a", 7), 0),
            Err(GrantError::Revoked)
        );
    }

    #[test]
    fn grant_does_not_confuse_a_prefix_with_a_subtree() {
        let grant = StandingGrant::new(
            [9; 16],
            [1; 32],
            GrantOperations::ALL,
            4,
            4096,
            0,
            &[CanonicalPath::new(b"private").unwrap()],
        )
        .unwrap();
        assert_eq!(grant.authorize(&modify("private2/a", 7), 0), Ok(()));
    }
}
