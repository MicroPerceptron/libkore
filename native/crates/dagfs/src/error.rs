//! The filesystem contract's refusal vocabulary.
//!
//! The service engine and every process-side adapter share these exact values.
//! Provider-shaped facts are folded through [`From<ProviderError>`] once, so
//! a full medium cannot become an ambiguous generic device failure at a
//! client boundary.

/// Why the filesystem refused a request.
///
/// This enum is deliberately exhaustive: adding a refusal is a wire-protocol
/// change, so the service codec must fail to compile until it assigns the new
/// state an exact status instead of collapsing it through a wildcard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FsError {
    /// No containment edge of that name beneath the named directory.
    NotFound,
    /// A containment edge of that name already exists beneath the directory.
    AlreadyExists,
    /// The operation needs a directory and the object is a file.
    NotDirectory,
    /// The operation needs a file and the object is a directory.
    IsDirectory,
    /// Empty, over-long, invalid UTF-8, reserved, or otherwise illegal name.
    InvalidName,
    /// The handle is out of range, stale, or belongs to another session.
    InvalidHandle,
    /// The route is live but conveys insufficient rights.
    AccessDenied,
    /// An edge the handle's proof depends on is gone or retired.
    AccessLost,
    /// The expected revision did not match the current file revision.
    Conflict,
    /// The operation would have made the containment graph cyclic.
    Cycle,
    /// Compatibility unlink of a directory that still has children.
    NotEmpty,
    /// The object arena is full.
    OutOfObjects,
    /// A containment-edge or authority-edge arena is full.
    OutOfEdges,
    /// The session's handle table is full.
    OutOfHandles,
    /// The medium itself has no room for the requested bytes.
    OutOfStorage,
    /// A bounded placement/extent table is full while the medium has room.
    OutOfPlacements,
    /// Every bounded immutable-version lease slot is held.
    OutOfVersionLeases,
    /// The bounded derived-result table is full.
    OutOfIndexSlots,
    /// An offset, length, cursor, or depth was invalid or overflowed.
    Range,
    /// The explicit work budget was exhausted before completion.
    BudgetExceeded,
    /// A content query required a result not indexed at the requested policy.
    IndexIncomplete,
    /// A well-formed request this version deliberately does not implement.
    Unsupported,
    /// A bulk verb arrived before a data window was attached.
    NoWindow,
    /// Another request currently owns one of this operation's exact claims.
    Busy,
    /// The storage backend failed after preserving engine metadata consistency.
    ProviderFailure(ProviderError),
}

/// Storage-shaped failures reported below the filesystem engine.
///
/// Exhaustive for the same reason as [`FsError`]: the one provider-to-client
/// classification must be updated consciously when a new backend state exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProviderError {
    /// The medium has no room left to satisfy the request.
    Exhausted,
    /// The bounded placement/extent table is full while the medium has room.
    PlacementsFull,
    /// The key names no live placement.
    UnknownKey,
    /// Offset/length outside the placement, or an overflowing computation.
    Range,
    /// The transfer exceeds the provider's declared maximum.
    TooLarge,
    /// The backing medium failed.
    Media,
    /// Controller/link/queue transport failed.
    Transport,
    /// Device or durable-format integrity check failed.
    Integrity,
    /// The request was cancelled before its completion was consumed.
    Cancelled,
    /// The channel lineage was revoked.
    Revoked,
    /// The provider permanently lost its channel/device.
    Lost,
    /// The provider is quiescing and accepts no new work.
    Draining,
    /// The backend is available only for reads.
    ReadOnly,
    /// A bounded queue, tracker, buffer, or placement table is full.
    Backpressure,
    /// A DMA window an earlier client lent has not retired from the device,
    /// and the requester-wide retirement that would clear it failed, so the
    /// controller admits no new client window until a retry succeeds
    /// (`res/LIFECYCLE_HARDENING.md` §6). Existing lanes keep serving.
    Retiring,
    /// A completion exceeded its budget while the backend remained healthy.
    Timeout,
}

impl From<ProviderError> for FsError {
    #[inline]
    fn from(error: ProviderError) -> Self {
        match error {
            ProviderError::Exhausted | ProviderError::TooLarge => Self::OutOfStorage,
            ProviderError::PlacementsFull => Self::OutOfPlacements,
            ProviderError::Range => Self::Range,
            ProviderError::UnknownKey
            | ProviderError::Media
            | ProviderError::Transport
            | ProviderError::Integrity
            | ProviderError::Cancelled
            | ProviderError::Revoked
            | ProviderError::Lost
            | ProviderError::Draining
            | ProviderError::ReadOnly
            | ProviderError::Backpressure
            | ProviderError::Retiring
            | ProviderError::Timeout => Self::ProviderFailure(error),
        }
    }
}

/// Filesystem result with the one contract refusal vocabulary.
pub type FsResult<T> = Result<T, FsError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_refusals_are_not_reported_as_provider_failures() {
        assert_eq!(
            FsError::from(ProviderError::Exhausted),
            FsError::OutOfStorage
        );
        assert_eq!(
            FsError::from(ProviderError::TooLarge),
            FsError::OutOfStorage
        );
        assert_eq!(
            FsError::from(ProviderError::PlacementsFull),
            FsError::OutOfPlacements
        );
        assert_eq!(FsError::from(ProviderError::Range), FsError::Range);
    }

    #[test]
    fn real_provider_faults_keep_their_cause() {
        for cause in [
            ProviderError::UnknownKey,
            ProviderError::Media,
            ProviderError::Transport,
            ProviderError::Integrity,
            ProviderError::Cancelled,
            ProviderError::Revoked,
            ProviderError::Lost,
            ProviderError::Draining,
            ProviderError::ReadOnly,
            ProviderError::Backpressure,
            ProviderError::Retiring,
            ProviderError::Timeout,
        ] {
            assert_eq!(FsError::from(cause), FsError::ProviderFailure(cause));
        }
    }

    #[test]
    fn denied_and_lost_are_distinct() {
        assert_ne!(FsError::AccessDenied, FsError::AccessLost);
    }
}
