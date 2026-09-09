//! What a client tells its caller instead of a result.
//!
//! Two vocabularies, deliberately distinct: [`Refusal`] is *why the
//! filesystem did not do it* — the service's own answer, an unreachable
//! pair, or a line that did not name what a verb needs — and [`Stop`] adds
//! the one thing that is not a filesystem event at all: the display refusing
//! the answer. A command prints a `Refusal` and returns to its prompt; a
//! `Display` stop is the end of its ability to say anything.

use crate::FsError;

/// Why an executable stopped.
///
/// The two arms are not the same kind of event and the caller treats them
/// differently: a filesystem refusal is a line on the screen and the prompt
/// comes back, while a display refusal is the end of the shell's ability to
/// say anything at all.
#[derive(Debug)]
pub enum Stop {
    /// The service refused, the pair did not answer, or the line did not
    /// name what the verb needs. Printed; the shell continues.
    Refused(Refusal),
    /// The **display** refused a write — terminal, exactly as a failed
    /// output write is terminal on every other path.
    Display,
}

/// What the shell has to tell the user instead of a result.
#[derive(Debug)]
pub enum Refusal {
    /// The filesystem's own answer, in its own vocabulary.
    Fs(FsError),
    /// The pair did not complete a valid round trip. The request may already
    /// have been published or committed; this is not a service refusal and
    /// does not authorize replaying a mutation.
    Pair,
    /// The command line did not carry the arguments the verb needs.
    Usage(&'static [u8]),
}

impl From<Refusal> for Stop {
    fn from(refusal: Refusal) -> Self {
        Self::Refused(refusal)
    }
}

impl Refusal {
    /// Whether this refusal means a derived index could not grow.
    ///
    /// **This predicate is shared on purpose.** It began as a local helper
    /// in `find`, and that locality is exactly why the same bug reappeared
    /// twice: an index that ran out of room is not a failure of the
    /// operation that triggered the indexing. The first instance wedged the
    /// boot (`find --reindex` treated a full index as boot-fatal, #688); the
    /// second told a user `touch` and `write` were "out of storage" when the
    /// file had been created and the bytes committed and both were already
    /// durable — the index alone could not be extended.
    ///
    /// The index is **auxiliary** to the authoritative write. A version that
    /// could not be indexed stays unindexed, and a later `CurrentOnly` search
    /// reports `IndexIncomplete` — that is the designed signal, and it is the
    /// honest one. Retroactively failing a committed write is not.
    ///
    /// A named family rather than a bare `OutOfStorage`, so a sibling
    /// capacity class cannot quietly fall outside it. Every arm here is "a
    /// bound was reached", never "the volume is broken" or "the request was
    /// malformed": those stay command failures.
    pub fn is_capacity_exhausted(&self) -> bool {
        matches!(
            self,
            Refusal::Fs(
                FsError::OutOfStorage
                    | FsError::OutOfPlacements
                    | FsError::OutOfVersionLeases
                    | FsError::OutOfIndexSlots
                    | FsError::OutOfObjects
                    | FsError::OutOfEdges
                    | FsError::OutOfHandles
            )
        )
    }
}

/// The refusal in the user's words — one arm per [`FsError`], so a reader
/// can look the cause up in the refusal vocabulary without guessing.
pub fn describe(err: FsError) -> &'static [u8] {
    match err {
        FsError::NotFound => &b"no such name"[..],
        FsError::AlreadyExists => &b"already exists"[..],
        FsError::NotDirectory => &b"not a directory"[..],
        FsError::IsDirectory => &b"is a directory"[..],
        FsError::InvalidName => &b"invalid name"[..],
        FsError::InvalidHandle => &b"invalid handle"[..],
        FsError::AccessDenied => &b"access denied"[..],
        FsError::AccessLost => &b"access lost"[..],
        FsError::Conflict => &b"changed underneath"[..],
        FsError::Cycle => &b"would make a cycle"[..],
        FsError::NotEmpty => &b"not empty"[..],
        FsError::OutOfObjects => &b"out of objects"[..],
        FsError::OutOfEdges => &b"out of edges"[..],
        FsError::OutOfHandles => &b"out of handles"[..],
        FsError::OutOfStorage => &b"out of storage"[..],
        FsError::Range => &b"out of range"[..],
        FsError::BudgetExceeded => &b"search budget exceeded"[..],
        FsError::IndexIncomplete => &b"search index incomplete"[..],
        FsError::Unsupported => &b"unsupported"[..],
        // The one refusal the user cannot act on and should not read as
        // their mistake: it names a missing transport leg, not a bad
        // request (plan §12d).
        FsError::NoWindow => &b"no data window"[..],
        FsError::ProviderFailure(_) => &b"storage failed"[..],
        // `FsError` is `#[non_exhaustive]`. A variant the service gains and
        // this list has not yet named must still print something, rather
        // than take the shell out of the boot image over a missing arm.
        _ => &b"refused"[..],
    }
}
