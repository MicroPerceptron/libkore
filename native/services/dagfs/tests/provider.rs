//! Provider bounds, failure propagation, and placement isolation.
//!
//! Covers the provider rows of the plan's §10.1 matrix, and doubles as the
//! start of the conformance evidence Phase 5 needs: everything here is stated
//! in terms of the engine's client-visible behaviour and the provider's
//! *declared* limits, never in terms of RAM specifics, so the same
//! expectations should hold over a block-backed provider.

mod common;

use common::{create_dir, create_file, create_file_closed, engine_with_one, name, read_all};
use kore_svc_dagfs::{
    ANY_REVISION, FsError, ObjectKind, ProviderError, Rights, StorageProvider, block_on,
    provider::Fault,
};

// ── declared limits are the contract ────────────────────────────────────────

#[test]
fn the_engine_honours_the_providers_declared_object_ceiling() {
    let (mut engine, session, root) = engine_with_one();
    let ceiling = engine.provider().limits().max_object_bytes;
    let file = create_file(&mut engine, session, root, "f", b"");

    // Exactly at the ceiling is allowed…
    assert!(block_on(engine.resize(session, file, ceiling, ANY_REVISION)).is_ok());
    assert_eq!(engine.stat(session, file).unwrap().len, ceiling);

    // …one byte past it is a typed capacity refusal, and changes nothing.
    assert_eq!(
        block_on(engine.resize(session, file, ceiling + 1, ANY_REVISION)).unwrap_err(),
        FsError::OutOfStorage
    );
    assert_eq!(engine.stat(session, file).unwrap().len, ceiling);
}

#[test]
fn a_write_that_would_exceed_the_object_ceiling_is_refused() {
    let (mut engine, session, root) = engine_with_one();
    let ceiling = engine.provider().limits().max_object_bytes;
    let file = create_file(&mut engine, session, root, "f", b"");

    assert_eq!(
        block_on(engine.write_at(session, file, ceiling, b"over", ANY_REVISION)).unwrap_err(),
        FsError::OutOfStorage
    );
    assert_eq!(engine.stat(session, file).unwrap().len, 0);
}

#[test]
fn a_transfer_larger_than_the_declared_maximum_is_refused() {
    let (mut engine, session, root) = engine_with_one();
    let max = engine.provider().limits().max_transfer;
    let file = create_file(&mut engine, session, root, "f", b"");

    let oversized = vec![0u8; max + 1];
    assert_eq!(
        block_on(engine.write_at(session, file, 0, &oversized, ANY_REVISION)).unwrap_err(),
        FsError::Range
    );
    assert_eq!(engine.stat(session, file).unwrap().len, 0);
}

#[test]
fn capacity_is_charged_at_the_declared_granularity() {
    let (mut engine, session, root) = engine_with_one();
    let granularity = engine.provider().limits().granularity as u64;
    let file = create_file(&mut engine, session, root, "f", b"");
    assert_eq!(engine.provider().bytes_in_use(), 0);

    // One byte costs one unit of granularity, not one byte.
    block_on(engine.write_at(session, file, 0, b"x", ANY_REVISION)).unwrap();
    assert_eq!(engine.provider().bytes_in_use(), granularity);
    // And the file's *visible* length is still one byte.
    assert_eq!(engine.stat(session, file).unwrap().len, 1);
}

// ── exhaustion ──────────────────────────────────────────────────────────────

#[test]
fn filling_the_pool_refuses_further_bytes_without_damaging_the_namespace() {
    let (mut engine, session, root) = engine_with_one();
    let ceiling = engine.provider().limits().max_object_bytes;
    let payload = vec![0xa5u8; ceiling as usize];

    // Fill the pool with maximum-size files.
    let mut filled = 0usize;
    loop {
        let file_name = format!("big{filled}");
        let made =
            block_on(engine.create(session, root, &name(&file_name), ObjectKind::File)).unwrap();
        match block_on(engine.write_at(session, made.handle, 0, &payload, ANY_REVISION)) {
            Ok(_) => {
                engine.close(session, made.handle).unwrap();
                filled += 1;
            }
            Err(err) => {
                assert_eq!(err, FsError::OutOfStorage, "exhaustion must be typed");
                // The object exists but holds nothing — the refusal did not
                // half-write it.
                assert_eq!(engine.stat(session, made.handle).unwrap().len, 0);
                engine.close(session, made.handle).unwrap();
                break;
            }
        }
    }
    assert!(filled > 0, "the pool must hold at least one maximum object");

    // Everything that did land is still intact and readable.
    for index in 0..filled {
        let opened = engine
            .open_at(session, root, &name(&format!("big{index}")), Rights::READ)
            .unwrap();
        assert_eq!(engine.stat(session, opened.handle).unwrap().len, ceiling);
        let mut head = [0u8; 8];
        assert_eq!(
            block_on(engine.read(session, opened.handle, 0, &mut head)),
            Ok(8)
        );
        assert_eq!(head, [0xa5u8; 8]);
        engine.close(session, opened.handle).unwrap();
    }
}

#[test]
fn freeing_a_file_makes_room_for_another() {
    let (mut engine, session, root) = engine_with_one();
    let ceiling = engine.provider().limits().max_object_bytes;
    let payload = vec![0x5au8; ceiling as usize];

    let mut names = Vec::new();
    loop {
        let file_name = format!("big{}", names.len());
        let made =
            block_on(engine.create(session, root, &name(&file_name), ObjectKind::File)).unwrap();
        let outcome = block_on(engine.write_at(session, made.handle, 0, &payload, ANY_REVISION));
        engine.close(session, made.handle).unwrap();
        if outcome.is_err() {
            block_on(engine.unlink(session, root, &name(&file_name))).unwrap();
            break;
        }
        names.push(file_name);
    }
    let full = engine.provider().chunks_in_use();

    // Release one, and exactly that much room comes back.
    block_on(engine.unlink(session, root, &name(&names[0]))).unwrap();
    assert!(engine.provider().chunks_in_use() < full);

    // And the space is genuinely reusable.
    let replacement =
        block_on(engine.create(session, root, &name("replacement"), ObjectKind::File)).unwrap();
    assert!(
        block_on(engine.write_at(session, replacement.handle, 0, &payload, ANY_REVISION)).is_ok()
    );
    assert_eq!(engine.provider().chunks_in_use(), full);
}

#[test]
fn running_out_of_placements_is_a_storage_refusal() {
    let (mut engine, session, root) = engine_with_one();
    // Zero-length writes take a placement but no capacity, so the placement
    // table is the ceiling reached first.
    let mut made = 0usize;
    let mut refused = None;
    'outer: for dir_index in 0..4 {
        let dir = create_dir(&mut engine, session, root, &format!("d{dir_index}"));
        for file_index in 0..kore_svc_dagfs::bounds::MAX_DIR_ENTRIES {
            let file = block_on(engine.create(
                session,
                dir,
                &name(&format!("f{file_index}")),
                ObjectKind::File,
            ));
            let Ok(file) = file else { break };
            let outcome = block_on(engine.write_at(session, file.handle, 0, b"", ANY_REVISION));
            engine.close(session, file.handle).unwrap();
            match outcome {
                Ok(_) => made += 1,
                Err(err) => {
                    refused = Some(err);
                    break 'outer;
                }
            }
        }
        engine.close(session, dir).unwrap();
    }
    assert_eq!(
        refused,
        Some(FsError::OutOfStorage),
        "placement exhaustion must be a typed storage refusal, not a panic"
    );
    assert_eq!(made, kore_svc_dagfs::bounds::MAX_PLACEMENTS);
}

// ── placement isolation ─────────────────────────────────────────────────────

#[test]
fn two_files_bytes_never_bleed_into_each_other() {
    let (mut engine, session, root) = engine_with_one();
    let a = create_file(&mut engine, session, root, "a", &[0xaa; 300]);
    let b = create_file(&mut engine, session, root, "b", &[0xbb; 300]);

    assert_eq!(read_all(&mut engine, session, a).unwrap(), vec![0xaa; 300]);
    assert_eq!(read_all(&mut engine, session, b).unwrap(), vec![0xbb; 300]);

    // Rewriting one leaves the other untouched, including across the chunk
    // boundary where a mis-mapped placement would show up.
    block_on(engine.write_at(session, a, 0, &[0xcc; 300], ANY_REVISION)).unwrap();
    assert_eq!(read_all(&mut engine, session, b).unwrap(), vec![0xbb; 300]);
}

#[test]
fn releasing_one_file_leaves_its_neighbours_intact() {
    let (mut engine, session, root) = engine_with_one();
    create_file_closed(&mut engine, session, root, "doomed", &[0x11; 500]);
    let keeper = create_file(&mut engine, session, root, "keeper", &[0x22; 500]);

    block_on(engine.unlink(session, root, &name("doomed"))).unwrap();

    assert_eq!(
        read_all(&mut engine, session, keeper).unwrap(),
        vec![0x22; 500]
    );
}

#[test]
fn a_new_file_never_sees_a_released_files_bytes() {
    let (mut engine, session, root) = engine_with_one();
    create_file_closed(&mut engine, session, root, "secret", &[0xde; 400]);
    block_on(engine.unlink(session, root, &name("secret"))).unwrap();
    assert_eq!(engine.provider().chunks_in_use(), 0);

    // A fresh file over the same reclaimed chunks must read as zeros — an
    // information leak between clients would show up right here.
    let fresh = create_file(&mut engine, session, root, "fresh", b"");
    block_on(engine.resize(session, fresh, 400, ANY_REVISION)).unwrap();
    let bytes = read_all(&mut engine, session, fresh).unwrap();
    assert_eq!(bytes.len(), 400);
    assert!(
        bytes.iter().all(|byte| *byte == 0),
        "reclaimed storage must be zeroed before reuse"
    );
}

#[test]
fn reclamation_releases_storage_exactly_once() {
    let (mut engine, session, root) = engine_with_one();
    let baseline_chunks = engine.provider().chunks_in_use();
    let baseline_placements = engine.provider().placements();

    // A subtree with bytes at two levels, plus a second name for one file.
    let dir = create_dir(&mut engine, session, root, "d");
    let shared = create_file(&mut engine, session, dir, "shared", &[1u8; 300]);
    create_file_closed(&mut engine, session, dir, "solo", &[2u8; 300]);
    engine
        .link(session, dir, &name("shared-again"), shared)
        .unwrap();
    assert!(engine.provider().chunks_in_use() > baseline_chunks);

    // Tear it all down.
    block_on(engine.unlink(session, dir, &name("shared"))).unwrap();
    block_on(engine.unlink(session, dir, &name("shared-again"))).unwrap();
    block_on(engine.unlink(session, dir, &name("solo"))).unwrap();
    block_on(engine.unlink(session, root, &name("d"))).unwrap();

    assert_eq!(engine.provider().chunks_in_use(), baseline_chunks);
    assert_eq!(engine.provider().placements(), baseline_placements);
    // A second release of any placement would have failed with UnknownKey and
    // been counted here.
    assert_eq!(engine.leaked_placements(), 0);
}

// ── failure propagation ─────────────────────────────────────────────────────

#[test]
fn an_injected_allocate_failure_leaves_the_object_consistent_and_reusable() {
    let (mut engine, session, root) = engine_with_one();
    let file = create_file(&mut engine, session, root, "f", b"");
    let before = engine.stat(session, file).unwrap();

    engine
        .provider()
        .inject(Fault::Allocate(ProviderError::Media));
    assert_eq!(
        block_on(engine.write_at(session, file, 0, b"bytes", ANY_REVISION)).unwrap_err(),
        FsError::ProviderFailure(ProviderError::Media)
    );

    // Metadata is exactly as it was, and nothing was placed.
    let after = engine.stat(session, file).unwrap();
    assert_eq!(after, before);
    assert_eq!(engine.provider().placements(), 0);

    // And the file works once the provider recovers — the failure left no
    // poisoned state behind.
    assert!(block_on(engine.write_at(session, file, 0, b"bytes", ANY_REVISION)).is_ok());
    assert_eq!(read_all(&mut engine, session, file).unwrap(), b"bytes");
}

#[test]
fn an_injected_resize_failure_leaves_length_and_revision_untouched() {
    let (mut engine, session, root) = engine_with_one();
    let file = create_file(&mut engine, session, root, "f", b"original");
    let before = engine.stat(session, file).unwrap();

    engine
        .provider()
        .inject(Fault::Resize(ProviderError::Media));
    assert_eq!(
        block_on(engine.resize(session, file, 1000, ANY_REVISION)).unwrap_err(),
        FsError::ProviderFailure(ProviderError::Media)
    );

    assert_eq!(engine.stat(session, file).unwrap(), before);
    assert_eq!(read_all(&mut engine, session, file).unwrap(), b"original");
}

#[test]
fn an_injected_read_failure_does_not_disturb_the_file() {
    let (mut engine, session, root) = engine_with_one();
    let file = create_file(&mut engine, session, root, "f", b"intact");
    let before = engine.stat(session, file).unwrap();

    engine.provider().inject(Fault::Read(ProviderError::Media));
    let mut buf = [0u8; 6];
    assert_eq!(
        block_on(engine.read(session, file, 0, &mut buf)).unwrap_err(),
        FsError::ProviderFailure(ProviderError::Media)
    );

    assert_eq!(engine.stat(session, file).unwrap(), before);
    assert_eq!(read_all(&mut engine, session, file).unwrap(), b"intact");
}

#[test]
fn an_injected_sync_failure_is_reported_rather_than_swallowed() {
    let (mut engine, session, root) = engine_with_one();
    let file = create_file(&mut engine, session, root, "f", b"bytes");

    engine.provider().inject(Fault::Sync(ProviderError::Media));
    // A durability request that did not cross the contract must say so — this
    // is the one place silence would be a lie about persistence.
    assert_eq!(
        block_on(engine.sync(session, file)).unwrap_err(),
        FsError::ProviderFailure(ProviderError::Media)
    );
    // Recovered.
    assert!(block_on(engine.sync(session, file)).is_ok());
}

#[test]
fn a_refused_release_keeps_the_namespace_correct_and_is_accounted_for() {
    let (mut engine, session, root) = engine_with_one();
    create_file_closed(&mut engine, session, root, "f", b"bytes");

    engine
        .provider()
        .inject(Fault::Release(ProviderError::Media));
    // The unlink genuinely happened, so it is reported as success…
    assert!(block_on(engine.unlink(session, root, &name("f"))).is_ok());

    // …the namespace reflects that…
    assert_eq!(
        engine
            .open_at(session, root, &name("f"), Rights::READ)
            .unwrap_err(),
        FsError::NotFound
    );
    assert!(
        common::list_names(&engine, session, root)
            .unwrap()
            .is_empty()
    );
    // …and the placement the provider would not take back is counted, not
    // quietly forgotten.
    assert_eq!(engine.leaked_placements(), 1);
    assert_eq!(engine.provider().placements(), 1);
}

#[test]
fn a_single_injected_fault_does_not_persist() {
    let (mut engine, session, root) = engine_with_one();
    let file = create_file(&mut engine, session, root, "f", b"");

    engine.provider().inject(Fault::Write(ProviderError::Media));
    assert!(block_on(engine.write_at(session, file, 0, b"one", ANY_REVISION)).is_err());
    // One shot: the very next request goes through.
    assert!(block_on(engine.write_at(session, file, 0, b"two", ANY_REVISION)).is_ok());
    assert_eq!(read_all(&mut engine, session, file).unwrap(), b"two");
}
