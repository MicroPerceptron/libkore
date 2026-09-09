//! Publishing the plain-text index batch for a version a client just wrote.
//!
//! Index publication is **auxiliary** to the authoritative write. This
//! deliberately reuses the #476 immutable-version seam: the writing
//! application interprets its own bytes, while FS only validates and
//! associates normalized facts with the version it lent. A version that
//! could not be indexed stays unindexed, and a later `CurrentOnly` search
//! reports `IndexIncomplete` — the designed, honest signal. Retroactively
//! failing a committed write is not, which is what [`classify`] enforces.

use kore_svc_extract::{
    Confidence, Fact, FactKind, FactValue, InputIdentity, NormalizedResult, ParserProvenance,
    SourceRange, encode_result,
    registry::{PLAIN_TEXT, Registry},
};
use kore_svc_extract_worker_rust::parse_plain_text;

use crate::{
    Fs, FsError, FsRequest, FsResponse, Refusal, VersionLease, View, bounds::MAX_TRANSFER,
    read_version_identity,
};

/// Whether the batch for the version just written was published.
///
/// Carried rather than discarded: a write whose index did not land is a
/// successful write over a *stale* index, and the only honest report says so.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Indexed {
    Published,
    Incomplete,
}

impl Indexed {
    /// The suffix a success line appends. Empty when the batch landed.
    pub fn note(self) -> &'static [u8] {
        match self {
            Self::Published => b"",
            Self::Incomplete => b" (index incomplete)",
        }
    }
}

/// Fold a publication outcome into the success line's fact.
///
/// Unsupported bytes stay unindexed, and **a full index is the same kind of
/// fact as unsupported bytes**: only `Unsupported` used to be tolerated
/// here, so a capacity refusal retroactively failed a write whose bytes were
/// already committed AND durable. That is what put `write: out of storage`
/// on screen directly above a `cat` showing the new contents. The write is
/// authoritative; the index is not, and it does not get to overrule it.
/// Transport and protocol failures remain command failures.
pub fn classify(published: Result<(), Refusal>) -> Result<Indexed, Refusal> {
    match published {
        Ok(()) => Ok(Indexed::Published),
        Err(Refusal::Fs(FsError::Unsupported)) => Ok(Indexed::Incomplete),
        Err(ref error) if error.is_capacity_exhausted() => Ok(Indexed::Incomplete),
        Err(error) => Err(error),
    }
}

/// Publish bounded application-owned tokens for the exact version of `file`
/// visible now. `supplied` is the bytes the caller just wrote (so nothing is
/// re-read); `None` reads the version back through its lease when it fits
/// one transfer, and otherwise leaves it unindexed.
pub async fn publish(
    fs: &mut Fs<'_, '_>,
    file: &View,
    supplied: Option<&[u8]>,
) -> Result<(), Refusal> {
    let FsResponse::VersionLent { lease } = fs
        .checked(&FsRequest::LendVersion {
            handle: file.handle(),
            window_offset: 0,
        })
        .await?
    else {
        return Err(Refusal::Pair);
    };

    // Everything after lending is fallible, but none of it may strand the
    // affine snapshot in FS.  Stage the result first, then release the lease
    // after either a commit answer or a local encoding refusal.
    let staged = match read_version_identity(&fs.window[..], 0).map_err(Refusal::Fs) {
        Ok(identity) => {
            let mut source = [0u8; MAX_TRANSFER];
            let bytes = if let Some(bytes) = supplied {
                Some(bytes)
            } else if identity.len <= source.len() as u64 {
                match fs
                    .checked(&FsRequest::ReadVersion {
                        lease,
                        offset: 0,
                        window_offset: 0,
                        length: identity.len as u32,
                    })
                    .await
                {
                    Ok(FsResponse::Transferred { count }) if u64::from(count) == identity.len => {
                        source[..count as usize].copy_from_slice(&fs.window[..count as usize]);
                        Some(&source[..count as usize])
                    }
                    Ok(_) => {
                        let _ = release(fs, lease).await;
                        return Err(Refusal::Pair);
                    }
                    Err(error) => {
                        let _ = release(fs, lease).await;
                        return Err(error);
                    }
                }
            } else {
                None
            };
            normalized_index_result(identity, bytes)
                .map_err(Refusal::Fs)
                .and_then(|result| {
                    let length = encode_result(&result, &mut fs.window[..])
                        .map_err(|_| Refusal::Fs(FsError::Unsupported))?;
                    u32::try_from(length).map_err(|_| Refusal::Fs(FsError::Range))
                })
        }
        Err(error) => Err(error),
    };
    let committed = match staged {
        Ok(length) => match fs
            .checked(&FsRequest::CommitDerived {
                lease,
                window_offset: 0,
                length,
                output_window_bytes: 0,
            })
            .await
        {
            Ok(FsResponse::DerivedCommitted { .. }) => Ok(()),
            Ok(_) => Err(Refusal::Pair),
            Err(error) => Err(error),
        },
        Err(error) => Err(error),
    };
    let released = release(fs, lease).await;
    match (committed, released) {
        (Err(error), _) | (Ok(()), Err(error)) => Err(error),
        (Ok(()), Ok(())) => Ok(()),
    }
}

async fn release(fs: &mut Fs<'_, '_>, lease: VersionLease) -> Result<(), Refusal> {
    match fs.checked(&FsRequest::ReleaseVersion { lease }).await? {
        FsResponse::Done => Ok(()),
        _ => Err(Refusal::Pair),
    }
}

/// Produce a complete batch only from the exact immutable bytes it describes.
///
/// A zero-length version is the sole exception to ordinary plain-text
/// parsing, whose format contract rejects empty input.  Even that batch is
/// admitted only after its length and digest match.  Missing bytes, parser
/// limits, malformed text, and identity/hash mismatches stay typed refusals;
/// committing an empty fallback would turn those indexing failures into
/// false-negative `CurrentOnly` search results.
fn normalized_index_result(
    identity: InputIdentity,
    bytes: Option<&[u8]>,
) -> Result<NormalizedResult, FsError> {
    let bytes = bytes.ok_or(FsError::Unsupported)?;
    if !bytes.is_empty() {
        return token_result(identity, bytes);
    }
    if InputIdentity::of(identity.source, identity.version, bytes) != identity {
        return Err(FsError::Unsupported);
    }
    let parser = Registry::get(PLAIN_TEXT).expect("the built-in plain-text declaration exists");
    Ok(NormalizedResult::empty(ParserProvenance::deterministic(
        identity,
        parser.id,
        parser.version,
    )))
}

/// Normalize distinct bounded words as custom plain-text facts.  A token is
/// both key and UTF-8 value: fact-key search names it, full-text search reads
/// it, and the semantic plane scores the same authorized vocabulary.
fn token_result(identity: InputIdentity, text: &[u8]) -> Result<NormalizedResult, FsError> {
    let mut result = parse_plain_text(identity, text).map_err(|_| FsError::Unsupported)?;
    let mut at = 0usize;
    while at < text.len() {
        while at < text.len() && !token_byte(text[at]) {
            at += 1;
        }
        let start = at;
        while at < text.len() && token_byte(text[at]) {
            at += 1;
        }
        let token = &text[start..at];
        if token.is_empty() || token.len() > kore_svc_extract::schema::MAX_KEY_BYTES {
            continue;
        }
        let token = core::str::from_utf8(token).map_err(|_| FsError::Unsupported)?;
        if result
            .facts
            .iter()
            .any(|fact| fact.key.as_str().eq_ignore_ascii_case(token))
        {
            continue;
        }
        let key = token.try_into().map_err(|_| FsError::Range)?;
        let value = token.try_into().map_err(|_| FsError::Range)?;
        if result.facts.is_full() {
            break;
        }
        result
            .facts
            .push(Fact {
                kind: FactKind::Custom,
                key,
                value: FactValue::Utf8(value),
                confidence: Confidence::CERTAIN,
                source: Some(SourceRange::new(start as u64, (at - start) as u64)),
            })
            .map_err(|_| FsError::Range)?;
    }
    Ok(result)
}

const fn token_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-' || !byte.is_ascii()
}

#[cfg(test)]
mod tests {
    use super::*;
    use kore_svc_extract::decode_result;

    #[test]
    fn shell_text_produces_distinct_searchable_tokens() {
        let bytes = b"Test test another";
        let result = token_result(InputIdentity::of(7, 1, bytes), bytes).unwrap();
        let mut custom = result
            .facts
            .iter()
            .filter(|fact| fact.kind == FactKind::Custom);
        assert_eq!(custom.next().unwrap().key.as_str(), "Test");
        assert_eq!(custom.next().unwrap().key.as_str(), "another");
        assert!(custom.next().is_none());
        assert_eq!(result.text[0].text.as_str(), "Test test another");

        let mut wire = [0u8; 1024];
        let used = encode_result(&result, &mut wire).unwrap();
        let decoded = decode_result(&wire[..used]).unwrap();
        Registry::validate(&decoded, result.provenance.input, 0).unwrap();
    }

    #[test]
    fn only_a_verified_zero_byte_version_produces_an_empty_batch() {
        let identity = InputIdentity::of(7, 1, b"");
        let result = normalized_index_result(identity, Some(b"")).unwrap();
        assert!(result.facts.is_empty());
        assert!(result.text.is_empty());
        assert!(result.embedded.is_empty());
        assert!(result.references.is_empty());
        assert!(result.symbols.is_empty());
        assert!(result.thumbnails.is_empty());
        assert!(result.diagnostics.is_empty());

        let wrong_identity = InputIdentity::of(7, 1, b"not empty");
        assert!(matches!(
            normalized_index_result(wrong_identity, Some(b"")),
            Err(FsError::Unsupported)
        ));
    }

    #[test]
    fn a_three_hundred_byte_line_is_refused_instead_of_committed_empty() {
        let bytes = [b'x'; 300];
        let identity = InputIdentity::of(7, 1, &bytes);
        assert!(matches!(
            normalized_index_result(identity, Some(&bytes)),
            Err(FsError::Unsupported)
        ));
    }

    #[test]
    fn invalid_utf8_is_an_explicit_unindexed_version() {
        let bytes = [0xff];
        let identity = InputIdentity::of(7, 1, &bytes);
        assert!(matches!(
            normalized_index_result(identity, Some(&bytes)),
            Err(FsError::Unsupported)
        ));
    }

    #[test]
    fn a_version_over_the_reindex_transfer_bound_is_refused_not_committed_empty() {
        let bytes = [b'x'; MAX_TRANSFER + 1];
        let identity = InputIdentity::of(7, 1, &bytes);
        // `publish` has no complete byte slice for this block-backed
        // version, so it must leave the version unindexed for CurrentOnly to
        // report `IndexIncomplete`.
        assert!(matches!(
            normalized_index_result(identity, None),
            Err(FsError::Unsupported)
        ));
    }

    #[test]
    fn unsupported_indexing_does_not_retroactively_fail_a_write() {
        assert_eq!(
            classify(Err(Refusal::Fs(FsError::Unsupported))).ok(),
            Some(Indexed::Incomplete)
        );
        assert!(classify(Err(Refusal::Pair)).is_err());
    }

    /// The reported metal failure: `write` printed "out of storage" and exit 1
    /// with the bytes already committed and durable, because a full index was
    /// allowed to overrule an authoritative write.
    #[test]
    fn a_full_index_does_not_retroactively_fail_a_committed_write() {
        for error in [
            FsError::OutOfStorage,
            FsError::OutOfPlacements,
            FsError::OutOfObjects,
            FsError::OutOfEdges,
            FsError::OutOfHandles,
        ] {
            assert_eq!(
                classify(Err(Refusal::Fs(error))).ok(),
                Some(Indexed::Incomplete),
                "{error:?} must not fail a write that already committed"
            );
        }
    }

    /// Succeeding quietly would hide that search is now stale, so the
    /// successful line has to carry the fact.
    #[test]
    fn an_incomplete_index_is_reported_on_the_success_line() {
        assert!(Indexed::Published.note().is_empty());
        assert_eq!(Indexed::Incomplete.note(), b" (index incomplete)");
    }

    /// A real fault is still a command failure — tolerance covers bounds
    /// being reached, never the provider or the pair failing.
    #[test]
    fn a_genuine_fault_still_fails_the_command() {
        for error in [
            FsError::ProviderFailure(crate::ProviderError::Media),
            FsError::AccessDenied,
            FsError::InvalidHandle,
        ] {
            assert!(
                classify(Err(Refusal::Fs(error))).is_err(),
                "{error:?} must remain a command failure"
            );
        }
    }
}
