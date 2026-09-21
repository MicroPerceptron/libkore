//! Namespace behaviour: the containment DAG, names, links, moves, cycles, and
//! reclamation, driven end-to-end through the engine's public API.
//!
//! Covers the namespace half of the plan's §10.1 matrix.

mod common;

use common::{create_dir, create_file, engine_with_one, list_names, name, read_all};
use kore_svc_dagfs::{
    ANY_REVISION, FsError, ObjectKind, Rights, block_on,
    bounds::{MAX_DIR_ENTRIES, MAX_OBJECTS},
};

#[test]
fn a_created_object_is_immediately_usable_by_its_creator() {
    let (mut engine, session, root) = engine_with_one();
    let file = create_file(&mut engine, session, root, "hello", b"kore");
    // No re-open, no second lookup: the create handed back working authority.
    assert_eq!(read_all(&mut engine, session, file).unwrap(), b"kore");
    let info = engine.stat(session, file).unwrap();
    assert_eq!(info.kind, ObjectKind::File);
    assert_eq!(info.len, 4);
    assert_eq!(info.links, 1);
}

#[test]
fn create_list_read_write_round_trip() {
    let (mut engine, session, root) = engine_with_one();
    let dir = create_dir(&mut engine, session, root, "demo");
    let file = create_file(&mut engine, session, dir, "hello", b"kore");

    assert_eq!(list_names(&engine, session, root).unwrap(), vec!["demo"]);
    assert_eq!(list_names(&engine, session, dir).unwrap(), vec!["hello"]);

    // Overwrite in place and read back.
    block_on(engine.write_at(session, file, 0, b"KORE", ANY_REVISION)).unwrap();
    assert_eq!(read_all(&mut engine, session, file).unwrap(), b"KORE");
}

#[test]
fn two_names_identify_one_object_and_share_its_bytes() {
    let (mut engine, session, root) = engine_with_one();
    let projects = create_dir(&mut engine, session, root, "projects");
    let file = create_file(&mut engine, session, projects, "main.rs", b"fn main() {}");

    // A second containment edge to the same object — not a copy.
    engine.link(session, root, &name("notes.rs"), file).unwrap();

    let via_projects = engine
        .open_at(session, projects, &name("main.rs"), Rights::ALL)
        .unwrap();
    let via_root = engine
        .open_at(session, root, &name("notes.rs"), Rights::ALL)
        .unwrap();

    let a = engine.stat(session, via_projects.handle).unwrap();
    let b = engine.stat(session, via_root.handle).unwrap();
    assert_eq!(a.object, b.object, "one object, two names");
    assert_eq!(a.links, 2);

    // A write through one name is visible through the other, which is what
    // "not copied" means observationally.
    block_on(engine.write_at(session, via_projects.handle, 0, b"CHANGED", ANY_REVISION)).unwrap();
    assert_eq!(
        &read_all(&mut engine, session, via_root.handle).unwrap()[..7],
        b"CHANGED"
    );
}

#[test]
fn unlinking_one_name_preserves_the_other_and_the_object() {
    let (mut engine, session, root) = engine_with_one();
    let file = create_file(&mut engine, session, root, "one", b"payload");
    engine.link(session, root, &name("two"), file).unwrap();

    block_on(engine.unlink(session, root, &name("one"))).unwrap();

    assert_eq!(
        engine
            .open_at(session, root, &name("one"), Rights::ALL)
            .unwrap_err(),
        FsError::NotFound
    );
    let survivor = engine
        .open_at(session, root, &name("two"), Rights::ALL)
        .unwrap();
    assert_eq!(
        read_all(&mut engine, session, survivor.handle).unwrap(),
        b"payload"
    );
    assert_eq!(engine.stat(session, survivor.handle).unwrap().links, 1);
}

#[test]
fn rename_relabels_without_changing_the_object() {
    let (mut engine, session, root) = engine_with_one();
    let file = create_file(&mut engine, session, root, "before", b"same bytes");
    let before = engine.stat(session, file).unwrap().object;

    engine
        .rename(session, root, &name("before"), &name("after"))
        .unwrap();

    assert_eq!(list_names(&engine, session, root).unwrap(), vec!["after"]);
    // The handle opened before the rename still works: rename keeps the edge.
    assert_eq!(engine.stat(session, file).unwrap().object, before);
    assert_eq!(read_all(&mut engine, session, file).unwrap(), b"same bytes");
}

#[test]
fn rename_onto_a_taken_name_is_refused_and_changes_nothing() {
    let (mut engine, session, root) = engine_with_one();
    create_file(&mut engine, session, root, "a", b"a");
    create_file(&mut engine, session, root, "b", b"b");

    assert_eq!(
        engine
            .rename(session, root, &name("a"), &name("b"))
            .unwrap_err(),
        FsError::AlreadyExists
    );
    assert_eq!(list_names(&engine, session, root).unwrap(), vec!["a", "b"]);
}

#[test]
fn moving_removes_only_the_source_edge_and_preserves_the_object() {
    let (mut engine, session, root) = engine_with_one();
    let dst = create_dir(&mut engine, session, root, "dst");
    let file = create_file(&mut engine, session, root, "f", b"contents");
    let identity = engine.stat(session, file).unwrap().object;

    engine
        .relocate(session, root, &name("f"), dst, &name("moved"))
        .unwrap();

    assert_eq!(
        engine
            .open_at(session, root, &name("f"), Rights::ALL)
            .unwrap_err(),
        FsError::NotFound
    );
    let moved = engine
        .open_at(session, dst, &name("moved"), Rights::ALL)
        .unwrap();
    let info = engine.stat(session, moved.handle).unwrap();
    assert_eq!(info.object, identity, "the object kept its identity");
    assert_eq!(info.links, 1, "one edge out, one edge in");
    assert_eq!(
        read_all(&mut engine, session, moved.handle).unwrap(),
        b"contents"
    );
}

#[test]
fn moving_needs_namespace_rights_at_both_parents() {
    let (mut engine, session, root) = engine_with_one();
    let dst = create_dir(&mut engine, session, root, "dst");
    create_file(&mut engine, session, root, "f", b"x");

    // A read-only handle on the destination cannot receive a move.
    let dst_read_only = engine
        .open_at(session, root, &name("dst"), Rights::READ)
        .unwrap();
    assert_eq!(
        engine
            .relocate(session, root, &name("f"), dst_read_only.handle, &name("m"))
            .unwrap_err(),
        FsError::AccessDenied
    );

    // A read-only handle on the source cannot give one up either.
    let root_read_only = engine.open_root(session, Rights::READ).unwrap();
    assert_eq!(
        engine
            .relocate(session, root_read_only.handle, &name("f"), dst, &name("m"))
            .unwrap_err(),
        FsError::AccessDenied
    );

    // The file never moved.
    assert!(
        engine
            .open_at(session, root, &name("f"), Rights::ALL)
            .is_ok()
    );
}

#[test]
fn directory_diamonds_keep_route_identity_and_detach_exactly() {
    let (mut engine, session, root) = engine_with_one();
    let left = create_dir(&mut engine, session, root, "left");
    let right = create_dir(&mut engine, session, root, "right");
    let shared = create_dir(&mut engine, session, left, "shared");
    engine
        .link(session, right, &name("shared"), shared)
        .unwrap();
    let through_right = engine
        .open_at(session, right, &name("shared"), Rights::ALL)
        .unwrap();
    assert_eq!(engine.stat(session, shared).unwrap().links, 2);

    block_on(engine.detach_edge(session, left, &name("shared"))).unwrap();
    assert_eq!(engine.stat(session, shared), Err(FsError::AccessLost));
    assert_eq!(engine.stat(session, through_right.handle).unwrap().links, 1);
    assert_eq!(list_names(&engine, session, right).unwrap(), vec!["shared"]);
}

#[test]
fn directory_cycles_are_refused() {
    let (mut engine, session, root) = engine_with_one();
    let outer = create_dir(&mut engine, session, root, "outer");
    let inner = create_dir(&mut engine, session, outer, "inner");
    let deeper = create_dir(&mut engine, session, inner, "deeper");

    assert_eq!(
        engine
            .link(session, deeper, &name("back"), outer)
            .unwrap_err(),
        FsError::Cycle
    );

    // Into itself.
    assert_eq!(
        engine
            .relocate(session, root, &name("outer"), outer, &name("self"))
            .unwrap_err(),
        FsError::Cycle
    );
    // Into its own descendant.
    assert_eq!(
        engine
            .relocate(session, root, &name("outer"), deeper, &name("loop"))
            .unwrap_err(),
        FsError::Cycle
    );

    // The tree survived both refusals intact.
    assert_eq!(list_names(&engine, session, root).unwrap(), vec!["outer"]);
    assert_eq!(list_names(&engine, session, outer).unwrap(), vec!["inner"]);
    assert_eq!(list_names(&engine, session, inner).unwrap(), vec!["deeper"]);
}

#[test]
fn unlinking_a_non_empty_directory_is_refused_until_it_is_emptied() {
    let (mut engine, session, root) = engine_with_one();
    let dir = create_dir(&mut engine, session, root, "d");
    create_file(&mut engine, session, dir, "f", b"x");

    assert_eq!(
        block_on(engine.unlink(session, root, &name("d"))).unwrap_err(),
        FsError::NotEmpty
    );
    block_on(engine.unlink(session, dir, &name("f"))).unwrap();
    block_on(engine.unlink(session, root, &name("d"))).unwrap();
    assert!(list_names(&engine, session, root).unwrap().is_empty());
}

#[test]
fn invalid_names_are_refused_at_every_door() {
    let (mut engine, session, root) = engine_with_one();
    create_file(&mut engine, session, root, "real", b"x");

    for bad in ["", ".", "..", "a/b", "a\0b"] {
        assert_eq!(
            kore_svc_dagfs::Name::new(bad).unwrap_err(),
            FsError::InvalidName,
            "{bad:?} must never become a name"
        );
    }
    // Over-long is refused by length, not truncated into a colliding name.
    let too_long = "x".repeat(kore_svc_dagfs::bounds::MAX_NAME_LEN + 1);
    assert_eq!(
        kore_svc_dagfs::Name::new(&too_long).unwrap_err(),
        FsError::InvalidName
    );
}

#[test]
fn listing_pages_through_a_full_directory_without_loss() {
    let (mut engine, session, root) = engine_with_one();
    // Fill the edge arena without also exhausting the smaller whole-namespace
    // object arena: many names lawfully identify one object in this DAG.
    let file = create_file(&mut engine, session, root, "f0", b"x");
    let mut expected = vec![String::from("f0")];
    for index in 1..MAX_DIR_ENTRIES {
        let file_name = format!("f{index}");
        engine.link(session, root, &name(&file_name), file).unwrap();
        expected.push(file_name);
    }
    engine.close(session, file).unwrap();
    expected.sort();

    let listed = list_names(&engine, session, root).unwrap();
    assert_eq!(listed, expected);

    // And the directory is now full, as a typed refusal.
    assert_eq!(
        block_on(engine.create(session, root, &name("one-too-many"), ObjectKind::File))
            .unwrap_err(),
        FsError::OutOfEdges
    );
}

#[test]
fn a_deleted_name_can_be_reused() {
    let (mut engine, session, root) = engine_with_one();
    create_file(&mut engine, session, root, "temp", b"first");
    block_on(engine.unlink(session, root, &name("temp"))).unwrap();

    // Same name, genuinely new object.
    let second = create_file(&mut engine, session, root, "temp", b"second");
    assert_eq!(read_all(&mut engine, session, second).unwrap(), b"second");
}

#[test]
fn reclamation_frees_the_object_and_its_storage_exactly_once() {
    let (mut engine, session, root) = engine_with_one();
    let objects_before = engine.object_count();
    let chunks_before = engine.provider().chunks_in_use();

    create_file(&mut engine, session, root, "doomed", b"some bytes");
    assert!(engine.provider().chunks_in_use() > chunks_before);

    block_on(engine.unlink(session, root, &name("doomed"))).unwrap();

    assert_eq!(engine.object_count(), objects_before);
    assert_eq!(engine.provider().chunks_in_use(), chunks_before);
    assert_eq!(engine.provider().placements(), 0);
    assert_eq!(engine.leaked_placements(), 0);
}

#[test]
fn a_linked_object_survives_until_its_last_name_goes() {
    let (mut engine, session, root) = engine_with_one();
    let file = create_file(&mut engine, session, root, "one", b"bytes");
    engine.link(session, root, &name("two"), file).unwrap();
    let placements = engine.provider().placements();

    block_on(engine.unlink(session, root, &name("one"))).unwrap();
    assert_eq!(
        engine.provider().placements(),
        placements,
        "still named, so still placed"
    );

    block_on(engine.unlink(session, root, &name("two"))).unwrap();
    assert_eq!(engine.provider().placements(), 0);
    assert_eq!(engine.provider().chunks_in_use(), 0);
}

#[test]
fn removing_a_subtree_releases_every_object_beneath_it() {
    let (mut engine, session, root) = engine_with_one();
    let objects_before = engine.object_count();
    let dirs_before = engine.directory_count();

    let a = create_dir(&mut engine, session, root, "a");
    let b = create_dir(&mut engine, session, a, "b");
    create_file(&mut engine, session, b, "deep", b"deep bytes");

    // Recursive removal is a client-side loop, not one transaction — so walk
    // it back out the way a shell's `rm -r` would.
    block_on(engine.unlink(session, b, &name("deep"))).unwrap();
    block_on(engine.unlink(session, a, &name("b"))).unwrap();
    block_on(engine.unlink(session, root, &name("a"))).unwrap();

    assert_eq!(engine.object_count(), objects_before);
    assert_eq!(engine.directory_count(), dirs_before);
    assert_eq!(engine.provider().chunks_in_use(), 0);
}

#[test]
fn the_object_arena_refuses_when_full_without_corrupting_the_namespace() {
    let (mut engine, session, root) = engine_with_one();
    // Filling the object arena now needs more than one directory: an object
    // bound of MAX_OBJECTS against MAX_DIR_ENTRIES entries per directory means
    // one directory runs out of edges long before the arena runs out of slots,
    // and that refusal is a different (also correct) one. So take a handful of
    // directories up front — the root can only hold MAX_DIR_ENTRIES of them —
    // and fill them in turn, which keeps this test about the arena it names.
    //
    // Handles are closed as we go where possible, so the refusal we observe is
    // the namespace running out of room rather than the session running out of
    // handles.
    const SPILLS: usize = 8;
    assert!(
        SPILLS * (MAX_DIR_ENTRIES - 1) >= MAX_OBJECTS,
        "the spill directories must be able to hold the whole object arena"
    );
    let mut created = 0usize;
    let mut buckets = Vec::new();
    for index in 0..SPILLS {
        let made = block_on(engine.create(
            session,
            root,
            &name(&format!("d{index}")),
            ObjectKind::Directory,
        ))
        .expect("a spill directory beneath the root");
        created += 1;
        buckets.push(made.handle);
    }

    let mut at = 0usize;
    let refusal = loop {
        let file_name = format!("f{created}");
        match block_on(engine.create(session, buckets[at], &name(&file_name), ObjectKind::File)) {
            Ok(made) => {
                engine.close(session, made.handle).unwrap();
                created += 1;
            }
            // This bucket is full; move to the next one.
            Err(FsError::OutOfEdges) if at + 1 < buckets.len() => at += 1,
            Err(err) => break err,
        }
    };
    assert_eq!(refusal, FsError::OutOfObjects);
    // The private root already owns one object slot; everything else created,
    // files and spill directories alike, accounts for the rest of the arena.
    assert_eq!(created, MAX_OBJECTS - 1);
    // Everything created is still correctly there and readable.
    assert_eq!(list_names(&engine, session, root).unwrap().len(), SPILLS);
    let probe = engine
        .open_at(
            session,
            buckets[0],
            &name(&format!("f{SPILLS}")),
            Rights::ALL,
        )
        .unwrap();
    assert_eq!(
        engine.stat(session, probe.handle).unwrap().kind,
        ObjectKind::File
    );
}
