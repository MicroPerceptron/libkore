//! Authority behaviour: private roots, sharing, attenuation, and immediate
//! loss of access.
//!
//! Covers the authority half of the plan's §10.1 matrix.  These are the tests
//! the MVP exists to pass: if any of them regresses, the filesystem's
//! capability story is broken even if every byte still round-trips.

mod common;

use common::{
    create_dir, create_file, engine_with_one, engine_with_sessions, list_names, name, read_all,
};
use kore_svc_dagfs::{
    ANY_REVISION, AuthorityKind, FsError, GrantId, Handle, ObjectKind, Rights, SessionId, block_on,
    bounds::MAX_HANDLES_PER_SESSION,
};

// ── private roots ───────────────────────────────────────────────────────────

#[test]
fn every_session_starts_with_its_own_empty_root() {
    let (engine, clients) = engine_with_sessions(3);
    for (session, root) in clients.iter().copied() {
        let info = engine.stat(session, root).unwrap();
        assert_eq!(info.kind, ObjectKind::Directory);
        assert!(list_names(&engine, session, root).unwrap().is_empty());
    }
    // Three sessions, three distinct root objects — there is no shared `/`.
    let mut roots: Vec<_> = clients
        .iter()
        .map(|(session, root)| engine.stat(*session, *root).unwrap().object.to_bits())
        .collect();
    roots.sort();
    roots.dedup();
    assert_eq!(roots.len(), 3, "no two sessions share a root object");
}

#[test]
fn the_same_pathname_in_two_private_roots_resolves_to_different_objects() {
    let (mut engine, clients) = engine_with_sessions(2);
    let (a, root_a) = clients[0];
    let (b, root_b) = clients[1];

    let file_a = create_file(&mut engine, a, root_a, "notes", b"belongs to A");
    let file_b = create_file(&mut engine, b, root_b, "notes", b"belongs to B");

    let id_a = engine.stat(a, file_a).unwrap().object;
    let id_b = engine.stat(b, file_b).unwrap().object;
    assert_ne!(id_a, id_b, "same name, different objects");

    // And each sees only its own bytes.
    assert_eq!(read_all(&mut engine, a, file_a).unwrap(), b"belongs to A");
    assert_eq!(read_all(&mut engine, b, file_b).unwrap(), b"belongs to B");
}

#[test]
fn a_session_cannot_see_anothers_names() {
    let (mut engine, clients) = engine_with_sessions(2);
    let (a, root_a) = clients[0];
    let (b, root_b) = clients[1];

    create_file(&mut engine, a, root_a, "secret", b"private");

    // B's root simply has no such name; there is no route to attempt.
    assert_eq!(
        engine
            .open_at(b, root_b, &name("secret"), Rights::READ)
            .unwrap_err(),
        FsError::NotFound
    );
    assert!(list_names(&engine, b, root_b).unwrap().is_empty());
}

#[test]
fn a_task_with_no_session_has_no_filesystem_access() {
    let (mut engine, _clients) = engine_with_sessions(1);
    // Filesystem access is not ambient: a session that was never attached has
    // no root to open (plan §12.7).
    assert_eq!(
        engine
            .open_root(SessionId::new(2), Rights::ALL)
            .unwrap_err(),
        FsError::InvalidHandle
    );
    assert!(engine.grants(SessionId::new(2)).unwrap_err() == FsError::InvalidHandle);
}

// ── handle isolation ────────────────────────────────────────────────────────

#[test]
fn one_sessions_handle_is_invalid_in_another_session() {
    let (mut engine, clients) = engine_with_sessions(2);
    let (a, root_a) = clients[0];
    let (b, _root_b) = clients[1];

    let file_a = create_file(&mut engine, a, root_a, "mine", b"A's bytes");

    // B presenting A's handle bits gets nothing — not A's object, not its own.
    assert_eq!(engine.stat(b, file_a).unwrap_err(), FsError::InvalidHandle);
    assert_eq!(
        block_on(engine.read(b, file_a, 0, &mut [0u8; 4])).unwrap_err(),
        FsError::InvalidHandle
    );
    assert_eq!(engine.close(b, file_a).unwrap_err(), FsError::InvalidHandle);
    // A's handle is untouched by the attempts.
    assert_eq!(read_all(&mut engine, a, file_a).unwrap(), b"A's bytes");
}

#[test]
fn forged_and_stale_handles_fail_closed() {
    let (mut engine, session, root) = engine_with_one();
    let file = create_file(&mut engine, session, root, "f", b"bytes");

    // Fabricated bit patterns: the never-issued sentinel, a wrong generation
    // on a slot that *is* live, and out-of-range slots.  (Deliberately not
    // "any small integer" — a session's own live handles have small bit
    // patterns, and re-presenting your own handle is not forgery.)
    let live = engine.stat(session, root).unwrap();
    assert_eq!(live.kind, ObjectKind::Directory);
    let wrong_generation = Handle::from_bits(root.to_bits() + 1);
    for candidate in [
        Handle::from_bits(0),
        wrong_generation,
        Handle::from_bits(u64::MAX),
        Handle::from_bits(0xdead_beef_dead_beef),
    ] {
        assert_eq!(
            engine.stat(session, candidate).unwrap_err(),
            FsError::InvalidHandle,
            "forged handle {:#x} must fail closed",
            candidate.to_bits()
        );
    }

    // A closed handle is stale, even though its slot may be reused.
    engine.close(session, file).unwrap();
    assert_eq!(
        engine.stat(session, file).unwrap_err(),
        FsError::InvalidHandle
    );
    let reopened = engine
        .open_at(session, root, &name("f"), Rights::ALL)
        .unwrap();
    // The reused slot must not resurrect the old handle.
    assert_eq!(
        engine.stat(session, file).unwrap_err(),
        FsError::InvalidHandle
    );
    assert!(engine.stat(session, reopened.handle).is_ok());
}

#[test]
fn a_forged_grant_id_is_inert() {
    let (mut engine, session, _root) = engine_with_one();
    // Derive a wrong-generation variant of the session's own real root grant,
    // so the test forges something plausible rather than something that
    // happens to be live.
    let real = engine
        .grants(session)
        .unwrap()
        .iter()
        .find(|info| info.kind == AuthorityKind::PrivateRoot)
        .expect("its private root is listed")
        .grant;
    for forged in [
        GrantId::from_bits(0),
        GrantId::from_bits(real.to_bits() + 1),
        GrantId::from_bits(u64::MAX),
        GrantId::from_bits(0xdead_beef_dead_beef),
    ] {
        assert!(
            engine.open_grant(session, forged, Rights::READ).is_err(),
            "forged grant {:#x} must not open",
            forged.to_bits()
        );
    }
    // The real one does open, so the test is not passing by refusing
    // everything.
    assert!(engine.open_grant(session, real, Rights::READ).is_ok());
}

// ── sharing one file ────────────────────────────────────────────────────────

#[test]
fn a_read_only_file_grant_lets_the_recipient_read_and_nothing_else() {
    let (mut engine, clients) = engine_with_sessions(2);
    let (a, root_a) = clients[0];
    let (b, root_b) = clients[1];

    let file = create_file(&mut engine, a, root_a, "shared", b"from A");
    let grant = engine
        .grant(a, file, b, Rights::READ, Some(name("gift")))
        .unwrap();

    let opened = engine.open_grant(b, grant, Rights::READ).unwrap();
    assert_eq!(opened.kind, ObjectKind::File);
    assert_eq!(read_all(&mut engine, b, opened.handle).unwrap(), b"from A");

    // Read-only means read-only: no write, no share, no naming it elsewhere.
    assert_eq!(
        block_on(engine.write_at(b, opened.handle, 0, b"no", ANY_REVISION)).unwrap_err(),
        FsError::AccessDenied
    );
    assert_eq!(
        block_on(engine.resize(b, opened.handle, 0, ANY_REVISION)).unwrap_err(),
        FsError::AccessDenied
    );
    assert_eq!(
        engine
            .grant(b, opened.handle, a, Rights::READ, None)
            .unwrap_err(),
        FsError::AccessDenied
    );
    assert_eq!(
        engine
            .link(b, root_b, &name("copy"), opened.handle)
            .unwrap_err(),
        FsError::AccessDenied
    );

    // A's own bytes are unchanged after all of that.
    assert_eq!(read_all(&mut engine, a, file).unwrap(), b"from A");
}

#[test]
fn a_recipient_cannot_open_a_grant_at_more_rights_than_it_carries() {
    let (mut engine, clients) = engine_with_sessions(2);
    let (a, root_a) = clients[0];
    let (b, _root_b) = clients[1];

    let file = create_file(&mut engine, a, root_a, "shared", b"x");
    let grant = engine.grant(a, file, b, Rights::READ, None).unwrap();

    assert_eq!(
        engine.open_grant(b, grant, Rights::WRITE).unwrap_err(),
        FsError::AccessDenied
    );
    assert_eq!(
        engine.open_grant(b, grant, Rights::ALL).unwrap_err(),
        FsError::AccessDenied
    );
    // Exactly what it carries is fine.
    assert!(engine.open_grant(b, grant, Rights::READ).is_ok());
}

#[test]
fn a_grant_cannot_convey_more_than_the_issuer_holds() {
    let (mut engine, clients) = engine_with_sessions(2);
    let (a, root_a) = clients[0];
    let (b, _root_b) = clients[1];

    create_file(&mut engine, a, root_a, "f", b"x");
    // A opens its own file read-only, then tries to share write access.
    let read_only = engine.open_at(a, root_a, &name("f"), Rights::READ).unwrap();
    assert_eq!(
        engine
            .grant(a, read_only.handle, b, Rights::WRITE, None)
            .unwrap_err(),
        // Sharing itself needs SHARE, which a read-only route does not carry.
        FsError::AccessDenied
    );
}

#[test]
fn revoking_an_ancestor_invalidates_redelegated_handles() {
    let (mut engine, clients) = engine_with_sessions(3);
    let (a, root_a) = clients[0];
    let (b, _root_b) = clients[1];
    let (c, _root_c) = clients[2];

    let file = create_file(&mut engine, a, root_a, "shared", b"lineage");
    let read_share = Rights::READ.union(Rights::SHARE);
    let to_b = engine.grant(a, file, b, read_share, None).unwrap();
    let b_view = engine.open_grant(b, to_b, read_share).unwrap();
    let to_c = engine
        .grant(b, b_view.handle, c, Rights::READ, None)
        .unwrap();
    let c_view = engine.open_grant(c, to_c, Rights::READ).unwrap();
    assert_eq!(read_all(&mut engine, c, c_view.handle).unwrap(), b"lineage");

    block_on(engine.revoke(a, to_b)).unwrap();
    assert_eq!(
        engine.stat(b, b_view.handle).unwrap_err(),
        FsError::AccessLost
    );
    assert_eq!(
        engine.stat(c, c_view.handle).unwrap_err(),
        FsError::AccessLost
    );
    assert!(engine.open_grant(c, to_c, Rights::READ).is_err());
    // The object remains reachable through A's independent namespace edge.
    assert_eq!(read_all(&mut engine, a, file).unwrap(), b"lineage");
}

#[test]
fn detaching_an_intermediate_holder_invalidates_its_redelegation() {
    let (mut engine, clients) = engine_with_sessions(3);
    let (a, root_a) = clients[0];
    let (b, _root_b) = clients[1];
    let (c, _root_c) = clients[2];

    let file = create_file(&mut engine, a, root_a, "shared", b"lineage");
    let read_share = Rights::READ.union(Rights::SHARE);
    let to_b = engine.grant(a, file, b, read_share, None).unwrap();
    let b_view = engine.open_grant(b, to_b, read_share).unwrap();
    let to_c = engine
        .grant(b, b_view.handle, c, Rights::READ, None)
        .unwrap();
    let c_view = engine.open_grant(c, to_c, Rights::READ).unwrap();

    block_on(engine.detach_session(b)).unwrap();
    assert_eq!(
        engine.stat(c, c_view.handle).unwrap_err(),
        FsError::AccessLost
    );
    assert!(engine.open_grant(c, to_c, Rights::READ).is_err());
    assert_eq!(read_all(&mut engine, a, file).unwrap(), b"lineage");
}

#[test]
fn a_third_session_cannot_open_someone_elses_grant() {
    let (mut engine, clients) = engine_with_sessions(3);
    let (a, root_a) = clients[0];
    let (b, _root_b) = clients[1];
    let (c, _root_c) = clients[2];

    let file = create_file(&mut engine, a, root_a, "shared", b"x");
    let grant = engine.grant(a, file, b, Rights::READ, None).unwrap();

    // A GrantId is not bearer authority — C holding the bits gets nothing.
    assert_eq!(
        engine.open_grant(c, grant, Rights::READ).unwrap_err(),
        FsError::InvalidHandle
    );
    assert!(engine.open_grant(b, grant, Rights::READ).is_ok());
}

#[test]
fn a_recipient_discovers_its_grants_by_listing_them() {
    let (mut engine, clients) = engine_with_sessions(2);
    let (a, root_a) = clients[0];
    let (b, _root_b) = clients[1];

    let file = create_file(&mut engine, a, root_a, "f", b"x");
    engine
        .grant(a, file, b, Rights::READ, Some(name("from-a")))
        .unwrap();

    let listed = engine.grants(b).unwrap();
    assert_eq!(listed.len(), 2, "its private root plus the new grant");
    let attachment = listed
        .iter()
        .find(|info| info.kind == AuthorityKind::Grant)
        .expect("the grant is listed");
    assert_eq!(attachment.rights, Rights::READ);
    assert_eq!(attachment.display_name.unwrap().as_str(), "from-a");

    // And the listed id is the one that opens.
    assert!(engine.open_grant(b, attachment.grant, Rights::READ).is_ok());
}

#[test]
fn a_sender_cannot_overwrite_a_name_in_the_recipients_view() {
    let (mut engine, clients) = engine_with_sessions(3);
    let (a, root_a) = clients[0];
    let (b, _root_b) = clients[1];
    let (c, root_c) = clients[2];

    let from_a = create_file(&mut engine, a, root_a, "f", b"a");
    let from_c = create_file(&mut engine, c, root_c, "f", b"c");
    engine
        .grant(a, from_a, b, Rights::READ, Some(name("inbox")))
        .unwrap();

    // C cannot claim the attachment name B already uses (plan §12.6).
    assert_eq!(
        engine
            .grant(c, from_c, b, Rights::READ, Some(name("inbox")))
            .unwrap_err(),
        FsError::AlreadyExists
    );
    // Without a name, or with a free one, it is fine.
    assert!(engine.grant(c, from_c, b, Rights::READ, None).is_ok());
    assert!(
        engine
            .grant(c, from_c, b, Rights::READ, Some(name("other")))
            .is_ok()
    );
}

// ── sharing a subtree ───────────────────────────────────────────────────────

#[test]
fn a_subtree_grant_exposes_descendants_and_nothing_above_it() {
    let (mut engine, clients) = engine_with_sessions(2);
    let (a, root_a) = clients[0];
    let (b, _root_b) = clients[1];

    // A builds root/{above.txt, shared/{inside.txt, deeper/}}
    create_file(&mut engine, a, root_a, "above.txt", b"not shared");
    let shared = create_dir(&mut engine, a, root_a, "shared");
    create_file(&mut engine, a, shared, "inside.txt", b"shared bytes");
    let deeper = create_dir(&mut engine, a, shared, "deeper");
    create_file(&mut engine, a, deeper, "deep.txt", b"deep bytes");

    let grant = engine
        .grant(a, shared, b, Rights::READ, Some(name("attached")))
        .unwrap();
    let view = engine.open_grant(b, grant, Rights::READ).unwrap();

    // Everything at and below the granted root is reachable.
    let mut listed = list_names(&engine, b, view.handle).unwrap();
    listed.sort();
    assert_eq!(listed, vec!["deeper", "inside.txt"]);
    let inside = engine
        .open_at(b, view.handle, &name("inside.txt"), Rights::READ)
        .unwrap();
    assert_eq!(
        read_all(&mut engine, b, inside.handle).unwrap(),
        b"shared bytes"
    );
    let deeper_view = engine
        .open_at(b, view.handle, &name("deeper"), Rights::READ)
        .unwrap();
    let deep = engine
        .open_at(b, deeper_view.handle, &name("deep.txt"), Rights::READ)
        .unwrap();
    assert_eq!(
        read_all(&mut engine, b, deep.handle).unwrap(),
        b"deep bytes"
    );

    // Nothing above it is: the sibling is not a child of the granted root…
    assert_eq!(
        engine
            .open_at(b, view.handle, &name("above.txt"), Rights::READ)
            .unwrap_err(),
        FsError::NotFound
    );
    // …and there is no `..` to climb with, by construction.
    assert_eq!(
        kore_svc_dagfs::Name::new("..").unwrap_err(),
        FsError::InvalidName
    );
    // B's own root is still empty — a grant is an attachment, not a copy.
    assert!(list_names(&engine, b, clients[1].1).unwrap().is_empty());
}

#[test]
fn a_read_only_subtree_grant_cannot_restructure_it() {
    // Plan §10.1 requires that a read-only grant refuse `unlink` as well as
    // write, link, and re-share — the namespace verbs are gated at the
    // directory, so a read-only view of one cannot rearrange it.
    let (mut engine, clients) = engine_with_sessions(2);
    let (a, root_a) = clients[0];
    let (b, _root_b) = clients[1];

    let shared = create_dir(&mut engine, a, root_a, "shared");
    create_file(&mut engine, a, shared, "f", b"A's bytes");
    let grant = engine
        .grant(a, shared, b, Rights::READ, Some(name("ro")))
        .unwrap();
    let view = engine.open_grant(b, grant, Rights::READ).unwrap();

    // B can look, and that is the whole of it.
    assert_eq!(list_names(&engine, b, view.handle).unwrap(), vec!["f"]);
    assert_eq!(
        block_on(engine.unlink(b, view.handle, &name("f"))).unwrap_err(),
        FsError::AccessDenied
    );
    assert_eq!(
        engine
            .rename(b, view.handle, &name("f"), &name("g"))
            .unwrap_err(),
        FsError::AccessDenied
    );
    assert_eq!(
        block_on(engine.create(b, view.handle, &name("new"), ObjectKind::File)).unwrap_err(),
        FsError::AccessDenied
    );
    assert_eq!(
        engine
            .relocate(b, view.handle, &name("f"), view.handle, &name("g"))
            .unwrap_err(),
        FsError::AccessDenied
    );

    // A's subtree is exactly as it was.
    assert_eq!(list_names(&engine, a, shared).unwrap(), vec!["f"]);
}

#[test]
fn a_writable_subtree_grant_can_be_used_to_write_within_it() {
    let (mut engine, clients) = engine_with_sessions(2);
    let (a, root_a) = clients[0];
    let (b, _root_b) = clients[1];

    let shared = create_dir(&mut engine, a, root_a, "shared");
    let grant = engine
        .grant(
            a,
            shared,
            b,
            Rights::READ
                .union(Rights::WRITE)
                .union(Rights::MUTATE_NAMESPACE),
            None,
        )
        .unwrap();
    let view = engine
        .open_grant(
            b,
            grant,
            Rights::READ
                .union(Rights::WRITE)
                .union(Rights::MUTATE_NAMESPACE),
        )
        .unwrap();

    // B creates inside the granted subtree, and A sees it.
    let made = block_on(engine.create(b, view.handle, &name("b-made"), ObjectKind::File)).unwrap();
    block_on(engine.write_at(b, made.handle, 0, b"by B", ANY_REVISION)).unwrap();
    assert_eq!(list_names(&engine, a, shared).unwrap(), vec!["b-made"]);
    let seen_by_a = engine
        .open_at(a, shared, &name("b-made"), Rights::READ)
        .unwrap();
    assert_eq!(read_all(&mut engine, a, seen_by_a.handle).unwrap(), b"by B");

    // But B still cannot re-share: SHARE was not delegated.
    assert_eq!(
        engine
            .grant(b, view.handle, a, Rights::READ, None)
            .unwrap_err(),
        FsError::AccessDenied
    );
}

// ── no laundering authority across attachments ──────────────────────────────

#[test]
fn a_granted_object_cannot_be_linked_into_the_recipients_own_root() {
    // The escalation this refuses: a containment edge carries no rights, so if
    // B could name A's read-only file inside B's own full-rights root, B would
    // afterwards reach it with write access that also outlives A's revoke.
    let (mut engine, clients) = engine_with_sessions(2);
    let (a, root_a) = clients[0];
    let (b, root_b) = clients[1];

    let file = create_file(&mut engine, a, root_a, "shared", b"A's bytes");
    let grant = engine.grant(a, file, b, Rights::READ, None).unwrap();
    let view = engine.open_grant(b, grant, Rights::READ).unwrap();

    assert_eq!(
        engine
            .link(b, root_b, &name("laundered"), view.handle)
            .unwrap_err(),
        FsError::AccessDenied
    );
    // Even a fully-rights grant cannot be laundered this way — the point is
    // the attachment boundary, not the rights on it.
    let full = engine
        .grant(a, file, b, Rights::ALL, Some(name("full")))
        .unwrap();
    let full_view = engine.open_grant(b, full, Rights::ALL).unwrap();
    assert_eq!(
        engine
            .link(b, root_b, &name("laundered"), full_view.handle)
            .unwrap_err(),
        FsError::AccessDenied
    );
    assert!(list_names(&engine, b, root_b).unwrap().is_empty());
}

#[test]
fn an_object_cannot_be_moved_out_of_a_granted_subtree_into_the_recipients_root() {
    let (mut engine, clients) = engine_with_sessions(2);
    let (a, root_a) = clients[0];
    let (b, root_b) = clients[1];

    let shared = create_dir(&mut engine, a, root_a, "shared");
    create_file(&mut engine, a, shared, "f", b"A's bytes");
    let delegated = Rights::READ
        .union(Rights::WRITE)
        .union(Rights::MUTATE_NAMESPACE);
    let grant = engine.grant(a, shared, b, delegated, None).unwrap();
    let view = engine.open_grant(b, grant, delegated).unwrap();

    // B may restructure *within* the subtree it was given…
    assert!(
        engine
            .relocate(b, view.handle, &name("f"), view.handle, &name("renamed"))
            .is_ok()
    );
    // …but not carry an object across the attachment boundary into its own
    // root, which would relabel it with B's own root authority.
    assert_eq!(
        engine
            .relocate(b, view.handle, &name("renamed"), root_b, &name("stolen"))
            .unwrap_err(),
        FsError::AccessDenied
    );
    assert!(list_names(&engine, b, root_b).unwrap().is_empty());
    assert_eq!(list_names(&engine, a, shared).unwrap(), vec!["renamed"]);
}

#[test]
fn linking_and_moving_within_one_authority_edge_still_works() {
    // The rule above must not break the ordinary case: a session restructuring
    // its own tree uses one authority edge throughout.
    let (mut engine, session, root) = engine_with_one();
    let dir = create_dir(&mut engine, session, root, "dir");
    let file = create_file(&mut engine, session, root, "f", b"x");

    // A second handle on the same private-root edge is still the same edge.
    let root_again = engine.open_root(session, Rights::ALL).unwrap();
    assert!(
        engine
            .link(session, root_again.handle, &name("second-name"), file)
            .is_ok()
    );
    assert!(
        engine
            .relocate(session, root, &name("f"), dir, &name("f"))
            .is_ok()
    );
}

// ── immediate loss of access ────────────────────────────────────────────────

#[test]
fn retiring_a_grant_makes_the_next_operation_on_its_handle_fail() {
    let (mut engine, clients) = engine_with_sessions(2);
    let (a, root_a) = clients[0];
    let (b, _root_b) = clients[1];

    let file = create_file(&mut engine, a, root_a, "shared", b"payload");
    let grant = engine.grant(a, file, b, Rights::READ, None).unwrap();
    let view = engine.open_grant(b, grant, Rights::READ).unwrap();
    assert_eq!(read_all(&mut engine, b, view.handle).unwrap(), b"payload");

    block_on(engine.revoke(a, grant)).unwrap();

    // Immediately — B still holds the same numeric handle.
    assert_eq!(
        block_on(engine.read(b, view.handle, 0, &mut [0u8; 8])).unwrap_err(),
        FsError::AccessLost
    );
    assert_eq!(
        engine.stat(b, view.handle).unwrap_err(),
        FsError::AccessLost
    );
    // And it cannot be opened afresh either.
    assert_eq!(
        engine.open_grant(b, grant, Rights::READ).unwrap_err(),
        FsError::AccessLost
    );
    // A is unaffected: it still has its own name for the object.
    assert_eq!(read_all(&mut engine, a, file).unwrap(), b"payload");
}

#[test]
fn a_recipient_may_hand_back_a_grant_itself() {
    let (mut engine, clients) = engine_with_sessions(2);
    let (a, root_a) = clients[0];
    let (b, _root_b) = clients[1];

    let file = create_file(&mut engine, a, root_a, "shared", b"x");
    let grant = engine.grant(a, file, b, Rights::READ, None).unwrap();
    let view = engine.open_grant(b, grant, Rights::READ).unwrap();

    block_on(engine.revoke(b, grant)).unwrap();
    assert_eq!(
        engine.stat(b, view.handle).unwrap_err(),
        FsError::AccessLost
    );
}

#[test]
fn a_third_session_cannot_revoke_a_grant_it_is_not_party_to() {
    let (mut engine, clients) = engine_with_sessions(3);
    let (a, root_a) = clients[0];
    let (b, _root_b) = clients[1];
    let (c, _root_c) = clients[2];

    let file = create_file(&mut engine, a, root_a, "shared", b"x");
    let grant = engine.grant(a, file, b, Rights::READ, None).unwrap();
    let view = engine.open_grant(b, grant, Rights::READ).unwrap();

    assert_eq!(
        block_on(engine.revoke(c, grant)).unwrap_err(),
        FsError::AccessDenied
    );
    // B's access survived the attempt.
    assert!(engine.stat(b, view.handle).is_ok());
}

#[test]
fn removing_an_intermediate_edge_invalidates_handles_opened_through_it() {
    let (mut engine, session, root) = engine_with_one();
    let mid = create_dir(&mut engine, session, root, "mid");
    let leaf = create_file(&mut engine, session, mid, "leaf", b"leaf bytes");

    // The leaf handle proved through root -> mid -> leaf.
    assert_eq!(read_all(&mut engine, session, leaf).unwrap(), b"leaf bytes");

    // Dismantle the chain from the middle.
    block_on(engine.unlink(session, mid, &name("leaf"))).unwrap();
    assert_eq!(
        block_on(engine.read(session, leaf, 0, &mut [0u8; 4])).unwrap_err(),
        FsError::AccessLost
    );

    // And removing `mid` itself invalidates a handle to `mid`.
    let mid_again = engine
        .open_at(session, root, &name("mid"), Rights::ALL)
        .unwrap();
    block_on(engine.unlink(session, root, &name("mid"))).unwrap();
    assert_eq!(
        engine.stat(session, mid_again.handle).unwrap_err(),
        FsError::AccessLost
    );
}

#[test]
fn losing_one_route_leaves_a_second_live_route_working() {
    let (mut engine, session, root) = engine_with_one();
    let dir = create_dir(&mut engine, session, root, "dir");
    let file = create_file(&mut engine, session, dir, "f", b"contents");

    // A second name for the same object, from a different parent.
    engine.link(session, root, &name("also-f"), file).unwrap();
    let via_root = engine
        .open_at(session, root, &name("also-f"), Rights::ALL)
        .unwrap();

    // Drop the route the first handle proved through.
    block_on(engine.unlink(session, dir, &name("f"))).unwrap();

    assert_eq!(
        block_on(engine.read(session, file, 0, &mut [0u8; 8])).unwrap_err(),
        FsError::AccessLost
    );
    // The other route still reaches the very same object and bytes.
    assert_eq!(
        read_all(&mut engine, session, via_root.handle).unwrap(),
        b"contents"
    );
    // And the object can be re-opened through the surviving name.
    let reopened = engine
        .open_at(session, root, &name("also-f"), Rights::ALL)
        .unwrap();
    assert_eq!(
        read_all(&mut engine, session, reopened.handle).unwrap(),
        b"contents"
    );
}

#[test]
fn a_grant_keeps_an_object_alive_after_its_last_name_is_gone() {
    let (mut engine, clients) = engine_with_sessions(2);
    let (a, root_a) = clients[0];
    let (b, _root_b) = clients[1];

    let file = create_file(&mut engine, a, root_a, "shared", b"kept alive");
    let grant = engine.grant(a, file, b, Rights::READ, None).unwrap();
    let view = engine.open_grant(b, grant, Rights::READ).unwrap();

    // A removes its only name for the object; B's grant is still a live route.
    block_on(engine.unlink(a, root_a, &name("shared"))).unwrap();
    assert_eq!(
        read_all(&mut engine, b, view.handle).unwrap(),
        b"kept alive",
        "the grant is a route, so the object survives"
    );
    // A's own handle went with the edge it proved through.
    assert_eq!(engine.stat(a, file).unwrap_err(), FsError::AccessLost);

    // When the grant goes too, nothing reaches the object and it is reclaimed.
    block_on(engine.revoke(b, grant)).unwrap();
    assert_eq!(engine.provider().placements(), 0);
    assert_eq!(engine.provider().chunks_in_use(), 0);
}

#[test]
fn a_grant_pins_an_uncontained_directory_graph_until_revocation() {
    let (mut engine, clients) = engine_with_sessions(2);
    let (a, root_a) = clients[0];
    let (b, _root_b) = clients[1];

    let shared = create_dir(&mut engine, a, root_a, "shared");
    create_file(&mut engine, a, shared, "leaf", b"still reachable");
    let grant = engine
        .grant(a, shared, b, Rights::READ, Some(name("shared")))
        .unwrap();
    let view = engine.open_grant(b, grant, Rights::READ).unwrap();

    // Native detach may remove a populated directory edge. The authority
    // edge still reaches the whole subgraph, so neither object is reclaimed.
    block_on(engine.detach_edge(a, root_a, &name("shared"))).unwrap();
    let leaf = engine
        .open_at(b, view.handle, &name("leaf"), Rights::READ)
        .unwrap();
    assert_eq!(
        read_all(&mut engine, b, leaf.handle).unwrap(),
        b"still reachable"
    );

    // The final authority route retires the directory and its newly
    // unreachable descendant exactly once.
    block_on(engine.revoke(b, grant)).unwrap();
    assert_eq!(
        engine.object_count(),
        2,
        "only the two session roots remain"
    );
    assert_eq!(engine.provider().placements(), 0);
    assert_eq!(engine.provider().chunks_in_use(), 0);
}

#[test]
fn moving_through_a_shared_area_invalidates_only_the_moved_route() {
    let (mut engine, session, root) = engine_with_one();
    let src = create_dir(&mut engine, session, root, "src");
    let dst = create_dir(&mut engine, session, root, "dst");
    let file = create_file(&mut engine, session, src, "f", b"moving");
    engine.link(session, root, &name("pinned"), file).unwrap();
    let pinned = engine
        .open_at(session, root, &name("pinned"), Rights::ALL)
        .unwrap();

    engine
        .relocate(session, src, &name("f"), dst, &name("f"))
        .unwrap();

    // The handle that proved through the source edge is gone…
    assert_eq!(engine.stat(session, file).unwrap_err(), FsError::AccessLost);
    // …the parallel link is untouched…
    assert_eq!(
        read_all(&mut engine, session, pinned.handle).unwrap(),
        b"moving"
    );
    // …and the object is reachable at its new place.
    let moved = engine
        .open_at(session, dst, &name("f"), Rights::ALL)
        .unwrap();
    assert_eq!(
        engine.stat(session, moved.handle).unwrap().object,
        engine.stat(session, pinned.handle).unwrap().object
    );
}

// ── teardown ────────────────────────────────────────────────────────────────

#[test]
fn detaching_a_session_releases_its_tree_and_leaves_peers_alone() {
    let (mut engine, clients) = engine_with_sessions(2);
    let (a, root_a) = clients[0];
    let (b, root_b) = clients[1];

    let dir_a = create_dir(&mut engine, a, root_a, "d");
    create_file(&mut engine, a, dir_a, "f", b"A's bytes");
    let file_b = create_file(&mut engine, b, root_b, "keep", b"B's bytes");

    let objects_before = engine.object_count();
    block_on(engine.detach_session(a)).unwrap();

    // A's root, directory, and file are all gone.
    assert_eq!(objects_before - engine.object_count(), 3);
    assert_eq!(engine.stat(a, root_a).unwrap_err(), FsError::InvalidHandle);
    // B is entirely unaffected.
    assert_eq!(read_all(&mut engine, b, file_b).unwrap(), b"B's bytes");
    assert_eq!(list_names(&engine, b, root_b).unwrap(), vec!["keep"]);
}

#[test]
fn detaching_a_recipient_does_not_disturb_the_issuer() {
    let (mut engine, clients) = engine_with_sessions(2);
    let (a, root_a) = clients[0];
    let (b, _root_b) = clients[1];

    let file = create_file(&mut engine, a, root_a, "shared", b"A's");
    let grant = engine.grant(a, file, b, Rights::READ, None).unwrap();
    engine.open_grant(b, grant, Rights::READ).unwrap();

    let edges_before = engine.authority_edge_count();
    block_on(engine.detach_session(b)).unwrap();

    // The grant went with its holder — B's private root and the grant it held
    // are both retired, and the object no longer counts a grant against it.
    assert_eq!(edges_before - engine.authority_edge_count(), 2);
    assert_eq!(engine.stat(a, file).unwrap().grants, 0);
    // A still has its file, bytes and all.
    assert_eq!(read_all(&mut engine, a, file).unwrap(), b"A's");
    // And the retired grant cannot be reopened by anyone.
    assert!(engine.open_grant(b, grant, Rights::READ).is_err());
}

#[test]
fn a_sessions_handle_ceiling_does_not_starve_its_peers() {
    let (mut engine, clients) = engine_with_sessions(2);
    let (a, root_a) = clients[0];
    let (b, root_b) = clients[1];

    create_file(&mut engine, a, root_a, "f", b"x");
    // A already holds two handles (its root and the created file); open the
    // rest of its allowance against the same name.
    while engine.live_handles(a) < MAX_HANDLES_PER_SESSION {
        engine
            .open_at(a, root_a, &name("f"), Rights::READ)
            .expect("within A's allowance");
    }
    assert_eq!(
        engine
            .open_at(a, root_a, &name("f"), Rights::READ)
            .unwrap_err(),
        FsError::OutOfHandles
    );

    // B is unaffected by A having spent its own allowance.
    assert!(engine.open_root(b, Rights::ALL).is_ok());
    assert!(list_names(&engine, b, root_b).unwrap().is_empty());
}
