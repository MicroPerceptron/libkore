//! Revisions, conflict detection, overlapping writes, atomic append, and what
//! a reader can observe.
//!
//! Covers plan §2.5 and the concurrency rows of §10.1.  The engine is driven
//! by one owner at a time — that exclusive borrow *is* the per-file mutation
//! gate in Phase 1 — so what these tests pin down is the **request-level**
//! contract every caller depends on: one request either commits completely or
//! has no visible effect, revisions are monotonic, and a stale expectation is
//! refused rather than silently applied.

mod common;

use common::{create_file, engine_with_one, engine_with_sessions, name, read_all};
use kore_svc_dagfs::{ANY_REVISION, FsError, ProviderError, Rights, block_on, provider::Fault};

// ── revisions ───────────────────────────────────────────────────────────────

#[test]
fn revisions_advance_monotonically_and_are_never_reused() {
    let (mut engine, session, root) = engine_with_one();
    let file = create_file(&mut engine, session, root, "f", b"");

    let mut seen = vec![engine.stat(session, file).unwrap().revision];
    for _ in 0..5 {
        let revision = block_on(engine.write_at(session, file, 0, b"x", ANY_REVISION)).unwrap();
        assert!(
            revision > *seen.last().unwrap(),
            "each committed write must advance the revision"
        );
        seen.push(revision);
    }
    // Shrinking is a mutation too, so it advances rather than rewinds.
    let after_resize = block_on(engine.resize(session, file, 0, ANY_REVISION)).unwrap();
    assert!(after_resize > *seen.last().unwrap());

    seen.push(after_resize);
    let mut unique = seen.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), seen.len(), "no revision is reused");
}

#[test]
fn an_exact_revision_write_commits_when_current() {
    let (mut engine, session, root) = engine_with_one();
    let file = create_file(&mut engine, session, root, "f", b"");

    let current = engine.stat(session, file).unwrap().revision;
    let next = block_on(engine.write_at(session, file, 0, b"abcd", current)).unwrap();
    assert!(next > current);
    assert_eq!(read_all(&mut engine, session, file).unwrap(), b"abcd");
}

#[test]
fn a_stale_exact_revision_write_is_refused_and_changes_nothing() {
    let (mut engine, session, root) = engine_with_one();
    let file = create_file(&mut engine, session, root, "f", b"first");
    let stale = engine.stat(session, file).unwrap().revision;

    // Someone else moves the file on.
    let current = block_on(engine.write_at(session, file, 0, b"secon", ANY_REVISION)).unwrap();
    assert_ne!(stale, current);

    // The stale expectation is refused…
    assert_eq!(
        block_on(engine.write_at(session, file, 0, b"third", stale)).unwrap_err(),
        FsError::Conflict
    );
    // …and left no trace at all: bytes, length, and revision are untouched.
    assert_eq!(read_all(&mut engine, session, file).unwrap(), b"secon");
    let info = engine.stat(session, file).unwrap();
    assert_eq!(info.revision, current);
    assert_eq!(info.len, 5);

    // Retrying with the current revision succeeds.
    assert!(block_on(engine.write_at(session, file, 0, b"third", current)).is_ok());
    assert_eq!(read_all(&mut engine, session, file).unwrap(), b"third");
}

#[test]
fn a_revision_from_the_future_is_also_refused() {
    let (mut engine, session, root) = engine_with_one();
    let file = create_file(&mut engine, session, root, "f", b"x");
    let current = engine.stat(session, file).unwrap().revision;
    assert_eq!(
        block_on(engine.write_at(session, file, 0, b"y", current + 1)).unwrap_err(),
        FsError::Conflict
    );
}

#[test]
fn any_revision_writes_are_unconditional() {
    let (mut engine, session, root) = engine_with_one();
    let file = create_file(&mut engine, session, root, "f", b"one");

    // Several unconditional writes in a row, with no revision bookkeeping by
    // the caller at all.
    for payload in [b"two", b"six", b"ten"] {
        assert!(block_on(engine.write_at(session, file, 0, payload, ANY_REVISION)).is_ok());
    }
    assert_eq!(read_all(&mut engine, session, file).unwrap(), b"ten");
}

#[test]
fn conflict_detection_is_per_file() {
    let (mut engine, session, root) = engine_with_one();
    let a = create_file(&mut engine, session, root, "a", b"a");
    let b = create_file(&mut engine, session, root, "b", b"b");

    let b_revision = engine.stat(session, b).unwrap().revision;
    // Moving `a` on must not invalidate a caller's expectation about `b`.
    block_on(engine.write_at(session, a, 0, b"A", ANY_REVISION)).unwrap();
    assert!(block_on(engine.write_at(session, b, 0, b"B", b_revision)).is_ok());
    assert_eq!(read_all(&mut engine, session, b).unwrap(), b"B");
}

// ── overlapping and disjoint writes ─────────────────────────────────────────

#[test]
fn non_overlapping_unconditional_writes_both_remain_present() {
    let (mut engine, session, root) = engine_with_one();
    let file = create_file(&mut engine, session, root, "f", b"");

    block_on(engine.write_at(session, file, 0, b"head", ANY_REVISION)).unwrap();
    block_on(engine.write_at(session, file, 8, b"tail", ANY_REVISION)).unwrap();

    let bytes = read_all(&mut engine, session, file).unwrap();
    assert_eq!(bytes.len(), 12);
    assert_eq!(&bytes[..4], b"head");
    assert_eq!(&bytes[4..8], &[0u8; 4], "the gap between them is zeros");
    assert_eq!(&bytes[8..], b"tail");
}

#[test]
fn overlapping_unconditional_writes_serialize_and_the_later_one_wins() {
    let (mut engine, session, root) = engine_with_one();
    let file = create_file(&mut engine, session, root, "f", b"");

    block_on(engine.write_at(session, file, 0, b"aaaaaaaa", ANY_REVISION)).unwrap();
    // Overlaps bytes 4..8 of the first write and extends past it.
    block_on(engine.write_at(session, file, 4, b"bbbbbbbb", ANY_REVISION)).unwrap();

    let bytes = read_all(&mut engine, session, file).unwrap();
    assert_eq!(bytes.len(), 12);
    assert_eq!(&bytes[..4], b"aaaa", "the non-overlapping prefix survives");
    assert_eq!(
        &bytes[4..],
        b"bbbbbbbb",
        "the later committed request determines the overlapping bytes"
    );
}

#[test]
fn a_shorter_write_does_not_truncate_the_file() {
    let (mut engine, session, root) = engine_with_one();
    let file = create_file(&mut engine, session, root, "f", b"abcdefgh");
    // Writing in the middle changes only those bytes; length is a maximum,
    // not a restatement.
    block_on(engine.write_at(session, file, 2, b"XY", ANY_REVISION)).unwrap();
    assert_eq!(read_all(&mut engine, session, file).unwrap(), b"abXYefgh");
    assert_eq!(engine.stat(session, file).unwrap().len, 8);
}

// ── append ──────────────────────────────────────────────────────────────────

#[test]
fn append_chooses_the_end_and_writes_as_one_step() {
    let (mut engine, session, root) = engine_with_one();
    let file = create_file(&mut engine, session, root, "log", b"");

    let (first_offset, first_revision) =
        block_on(engine.append(session, file, b"one", ANY_REVISION)).unwrap();
    let (second_offset, second_revision) =
        block_on(engine.append(session, file, b"two", ANY_REVISION)).unwrap();

    assert_eq!(first_offset, 0);
    assert_eq!(
        second_offset, 3,
        "the second append saw the first one's end"
    );
    assert!(second_revision > first_revision);
    assert_eq!(read_all(&mut engine, session, file).unwrap(), b"onetwo");
}

#[test]
fn appends_never_land_on_the_same_offset() {
    let (mut engine, session, root) = engine_with_one();
    let file = create_file(&mut engine, session, root, "log", b"");

    let mut offsets = Vec::new();
    for _ in 0..8 {
        let (offset, _) = block_on(engine.append(session, file, b"abc", ANY_REVISION)).unwrap();
        offsets.push(offset);
    }
    let mut unique = offsets.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(
        unique.len(),
        offsets.len(),
        "every append got its own space"
    );
    assert_eq!(offsets, vec![0, 3, 6, 9, 12, 15, 18, 21]);
    // No append overwrote another: the file is exactly the concatenation.
    assert_eq!(
        read_all(&mut engine, session, file).unwrap(),
        b"abcabcabcabcabcabcabcabc"
    );
}

#[test]
fn append_onto_a_non_empty_file_starts_at_its_end() {
    let (mut engine, session, root) = engine_with_one();
    let file = create_file(&mut engine, session, root, "f", b"prefix");
    let (offset, _) = block_on(engine.append(session, file, b"-suffix", ANY_REVISION)).unwrap();
    assert_eq!(offset, 6);
    assert_eq!(
        read_all(&mut engine, session, file).unwrap(),
        b"prefix-suffix"
    );
}

#[test]
fn a_conflicted_append_consumes_no_position() {
    let (mut engine, session, root) = engine_with_one();
    let file = create_file(&mut engine, session, root, "log", b"");
    let (_, revision) = block_on(engine.append(session, file, b"one", ANY_REVISION)).unwrap();

    // A stale expectation must be refused *before* an offset is claimed…
    assert_eq!(
        block_on(engine.append(session, file, b"bad", revision - 1)).unwrap_err(),
        FsError::Conflict
    );
    // …so the next legitimate append still lands immediately after "one".
    let (offset, _) = block_on(engine.append(session, file, b"two", revision)).unwrap();
    assert_eq!(offset, 3);
    assert_eq!(read_all(&mut engine, session, file).unwrap(), b"onetwo");
    assert_eq!(engine.stat(session, file).unwrap().len, 6);
}

#[test]
fn append_needs_write_rights() {
    let (mut engine, session, root) = engine_with_one();
    create_file(&mut engine, session, root, "f", b"x");
    let reader = engine
        .open_at(session, root, &name("f"), Rights::READ)
        .unwrap();
    assert_eq!(
        block_on(engine.append(session, reader.handle, b"y", ANY_REVISION)).unwrap_err(),
        FsError::AccessDenied
    );
}

// ── what a reader can observe ───────────────────────────────────────────────

#[test]
fn a_reader_sees_the_previous_complete_revision_when_a_write_fails() {
    let (mut engine, session, root) = engine_with_one();
    let file = create_file(&mut engine, session, root, "f", b"committed");
    let committed = engine.stat(session, file).unwrap().revision;

    // A write that dies inside the provider, after capacity was reserved.
    engine.provider().inject(Fault::Write(ProviderError::Media));
    assert_eq!(
        block_on(engine.write_at(session, file, 0, b"DESTROYED", ANY_REVISION)).unwrap_err(),
        FsError::ProviderFailure(ProviderError::Media)
    );

    // The reader observes the whole previous revision — not a mixture, and not
    // a longer file with a partly-written tail.
    let info = engine.stat(session, file).unwrap();
    assert_eq!(info.revision, committed);
    assert_eq!(info.len, 9);
    assert_eq!(read_all(&mut engine, session, file).unwrap(), b"committed");
}

#[test]
fn a_failed_extending_write_does_not_expose_a_longer_file() {
    let (mut engine, session, root) = engine_with_one();
    let file = create_file(&mut engine, session, root, "f", b"short");

    engine.provider().inject(Fault::Write(ProviderError::Media));
    assert!(block_on(engine.write_at(session, file, 100, b"far away", ANY_REVISION)).is_err());

    // Capacity may have been reserved, but length is what a reader sees, and
    // it never moved.
    assert_eq!(engine.stat(session, file).unwrap().len, 5);
    assert_eq!(read_all(&mut engine, session, file).unwrap(), b"short");
}

#[test]
fn a_reader_sees_the_whole_of_a_committed_write() {
    let (mut engine, session, root) = engine_with_one();
    let file = create_file(&mut engine, session, root, "f", b"");
    let payload: Vec<u8> = (0..600u32).map(|i| (i % 251) as u8).collect();

    let revision = block_on(engine.write_at(session, file, 0, &payload, ANY_REVISION)).unwrap();

    // A write spanning several provider chunks is visible in full, or not at
    // all — never chunk-by-chunk.
    let info = engine.stat(session, file).unwrap();
    assert_eq!(info.revision, revision);
    assert_eq!(info.len, payload.len() as u64);
    assert_eq!(read_all(&mut engine, session, file).unwrap(), payload);
}

// ── two clients on one shared file ──────────────────────────────────────────

#[test]
fn two_clients_interleaving_writes_on_a_shared_file_stay_consistent() {
    let (mut engine, clients) = engine_with_sessions(2);
    let (a, root_a) = clients[0];
    let (b, _root_b) = clients[1];

    let file = create_file(&mut engine, a, root_a, "shared", b"");
    let writable = Rights::READ.union(Rights::WRITE);
    let grant = engine.grant(a, file, b, writable, None).unwrap();
    let b_view = engine.open_grant(b, grant, writable).unwrap();

    // Alternate appends from the two sessions.
    let mut revisions = Vec::new();
    for round in 0..4 {
        let (_, r1) = block_on(engine.append(a, file, b"A", ANY_REVISION)).unwrap();
        let (_, r2) = block_on(engine.append(b, b_view.handle, b"B", ANY_REVISION)).unwrap();
        revisions.push(r1);
        revisions.push(r2);
        assert_eq!(
            engine.stat(a, file).unwrap().len,
            (round + 1) * 2,
            "each append added exactly one byte"
        );
    }

    // One shared revision line, strictly increasing across both writers.
    assert!(revisions.windows(2).all(|pair| pair[0] < pair[1]));
    // Nothing was lost or overwritten.
    assert_eq!(read_all(&mut engine, a, file).unwrap(), b"ABABABAB");
    // And both see the identical object.
    assert_eq!(
        engine.stat(a, file).unwrap().object,
        engine.stat(b, b_view.handle).unwrap().object
    );
}

#[test]
fn one_clients_conditional_write_detects_the_others_change() {
    let (mut engine, clients) = engine_with_sessions(2);
    let (a, root_a) = clients[0];
    let (b, _root_b) = clients[1];

    let file = create_file(&mut engine, a, root_a, "shared", b"start");
    let writable = Rights::READ.union(Rights::WRITE);
    let grant = engine.grant(a, file, b, writable, None).unwrap();
    let b_view = engine.open_grant(b, grant, writable).unwrap();

    // Both read the same revision, then B commits first.
    let observed = engine.stat(a, file).unwrap().revision;
    assert_eq!(engine.stat(b, b_view.handle).unwrap().revision, observed);
    block_on(engine.write_at(b, b_view.handle, 0, b"by-B!", observed)).unwrap();

    // A's conditional write on the now-stale revision is refused, which is the
    // whole point of carrying an expectation.
    assert_eq!(
        block_on(engine.write_at(a, file, 0, b"by-A!", observed)).unwrap_err(),
        FsError::Conflict
    );
    assert_eq!(read_all(&mut engine, a, file).unwrap(), b"by-B!");

    // A re-reads and retries, and now succeeds.
    let fresh = engine.stat(a, file).unwrap().revision;
    assert!(block_on(engine.write_at(a, file, 0, b"by-A!", fresh)).is_ok());
    assert_eq!(read_all(&mut engine, b, b_view.handle).unwrap(), b"by-A!");
}

#[test]
fn a_revoked_writer_stops_being_able_to_write_immediately() {
    let (mut engine, clients) = engine_with_sessions(2);
    let (a, root_a) = clients[0];
    let (b, _root_b) = clients[1];

    let file = create_file(&mut engine, a, root_a, "shared", b"");
    let writable = Rights::READ.union(Rights::WRITE);
    let grant = engine.grant(a, file, b, writable, None).unwrap();
    let b_view = engine.open_grant(b, grant, writable).unwrap();
    block_on(engine.append(b, b_view.handle, b"before", ANY_REVISION)).unwrap();

    block_on(engine.revoke(a, grant)).unwrap();

    assert_eq!(
        block_on(engine.append(b, b_view.handle, b"after", ANY_REVISION)).unwrap_err(),
        FsError::AccessLost
    );
    // Everything B wrote while authorized is still there.
    assert_eq!(read_all(&mut engine, a, file).unwrap(), b"before");
}
