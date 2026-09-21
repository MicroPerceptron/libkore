use super::flows::{self, DeviceSetup};
use super::*;
use kore_abi::cap::Rights;
use kore_abi::session::{
    DESCRIPTOR_ABI, Invocation, InvocationWire, MAX_INVOCATIONS, OpCode, SessionFlags,
    SessionHeader, SlotRef,
};
use zerocopy::FromBytes;

/// Parse a finished descriptor exactly the way the kernel decode does
/// (`prepare::decode`): header off the front, `count` invocations
/// reinterpreted in place, args = the rest.
fn parse(bytes: &[u8]) -> (SessionHeader, &[InvocationWire], &[u8]) {
    let (hdr, rest) = SessionHeader::read_from_prefix(bytes).expect("header");
    let (invs, args) =
        <[InvocationWire]>::ref_from_prefix_with_elems(rest, hdr.count.get() as usize)
            .expect("invocations");
    (hdr, invs, args)
}

#[test]
fn view_placement_requests_only_a_window_not_more_authority() {
    use kore_abi::session::{DERIVE_KIND_VIEW_MAP, ViewMapReq, ViewMapWindow};
    use zerocopy::TryFromBytes;

    for window in [
        ViewMapWindow::Map,
        ViewMapWindow::Surfaces,
        ViewMapWindow::Heap,
    ] {
        let mut builder = SessionBuilder::new();
        builder
            .map_view_in(SlotRef::from_wire(2), window, SlotRef::from_wire(3))
            .unwrap();
        let mut out = [0u8; DESCRIPTOR_MAX];
        let len = builder.finish(&mut out).unwrap();
        let (_, invs, args) = parse(&out[..len]);
        assert_eq!(invs.len(), 1);
        assert_eq!(invs[0].op.get(), OpCode::Derive as u16);
        assert_eq!(invs[0].kind, DERIVE_KIND_VIEW_MAP);
        assert_eq!(invs[0].requested, 0);
        assert_eq!(invs[0].subject.get(), 2);
        assert_eq!(invs[0].result.get(), 3);
        assert_eq!(args.len(), ViewMapReq::WIRE);
        let (req, rest) = ViewMapReq::try_ref_from_prefix(args).unwrap();
        assert!(rest.is_empty());
        assert_eq!(req.placement(), Some(window));
        assert_eq!(req.lane, 0);
        assert_eq!(req.reserved, [0; 5]);
    }

    let mut builder = SessionBuilder::new();
    builder
        .map_view_surface_lane(
            SlotRef::from_wire(2),
            kore_abi::session::SurfaceMapLane::Second,
            SlotRef::from_wire(3),
        )
        .unwrap();
    let mut out = [0u8; DESCRIPTOR_MAX];
    let len = builder.finish(&mut out).unwrap();
    let (_, _, args) = parse(&out[..len]);
    let (req, _) = ViewMapReq::try_ref_from_prefix(args).unwrap();
    assert_eq!(req.placement(), Some(ViewMapWindow::Surfaces));
    assert_eq!(
        req.surface_lane(),
        Some(kore_abi::session::SurfaceMapLane::Second)
    );
}

#[test]
fn finish_writes_the_frozen_layout() {
    let mut b = SessionBuilder::new();
    let lease = b
        .lease(
            SlotRef::from_wire(3),
            Rights::ACCESS | Rights::DERIVE,
            8192,
            SlotRef::from_wire(10),
        )
        .unwrap();
    b.revoke(lease.slot()).unwrap();
    b.wait_delivery();

    let mut out = [0u8; DESCRIPTOR_MAX];
    let n = b.finish(&mut out).unwrap();
    assert_eq!(n, 8 + 2 * 16 + 16, "header + 2 invocations + LeaseReq");

    let (hdr, invs, args) = parse(&out[..n]);
    assert_eq!(hdr.abi.get(), DESCRIPTOR_ABI);
    assert_eq!(hdr.count.get(), 2);
    assert_eq!(hdr.flags.get(), SessionFlags::WAIT_DELIVERY.bits());
    assert_eq!(hdr.reserved.get(), 0);

    assert_eq!(invs[0].op.get(), OpCode::Derive as u16);
    assert_eq!(invs[0].subject.get(), 3);
    assert_eq!(invs[0].result.get(), 10);
    assert_eq!(invs[0].arg.get(), 0, "first payload at args offset 0");
    assert_eq!(
        u64::from_le_bytes(args[0..8].try_into().unwrap()),
        8192,
        "LeaseReq.size"
    );

    assert_eq!(invs[1].op.get(), OpCode::Revoke as u16);
    assert_eq!(invs[1].subject.get(), 10, "revokes the staged result");
    assert_eq!(invs[1].arg.get(), Invocation::NO_ARG);
}

#[test]
fn executable_finalization_is_a_no_argument_consuming_revoke() {
    let source = SlotRef::from_wire(12);
    let result = SlotRef::from_wire(13);
    let mut builder = SessionBuilder::new();
    let staged = builder
        .finalize_exec(source, result)
        .expect("exec finalization stages");
    assert_eq!(staged.slot(), result);

    let mut out = [0u8; DESCRIPTOR_MAX];
    let len = builder.finish(&mut out).expect("descriptor finishes");
    assert_eq!(len, 8 + 16);
    let (header, invocations, args) = parse(&out[..len]);
    assert_eq!(header.count.get(), 1);
    assert_eq!(invocations[0].op.get(), OpCode::Revoke as u16);
    assert_eq!(invocations[0].kind, kore_abi::session::REVOKE_KIND_EXEC);
    assert_eq!(invocations[0].requested, 0);
    assert_eq!(invocations[0].subject.get(), source.index());
    assert_eq!(invocations[0].result.get(), result.index());
    assert_eq!(invocations[0].arg.get(), Invocation::NO_ARG);
    assert!(args.is_empty());
}

/// The return is the finalization's exact wire twin: one no-argument
/// consuming revoke, differing only in the kind byte. Pinning both shapes
/// is what keeps the cycle symmetric on the wire as well as in the kernel.
#[test]
fn the_executable_return_is_the_finalizations_wire_twin() {
    let source = SlotRef::from_wire(12);
    let result = SlotRef::from_wire(13);
    let mut builder = SessionBuilder::new();
    let staged = builder
        .return_writable(source, result)
        .expect("the return stages");
    assert_eq!(staged.slot(), result);

    let mut out = [0u8; DESCRIPTOR_MAX];
    let len = builder.finish(&mut out).expect("descriptor finishes");
    assert_eq!(len, 8 + 16, "the same one-invocation descriptor");
    let (header, invocations, args) = parse(&out[..len]);
    assert_eq!(header.count.get(), 1);
    assert_eq!(invocations[0].op.get(), OpCode::Revoke as u16);
    assert_eq!(
        invocations[0].kind,
        kore_abi::session::REVOKE_KIND_EXEC_RETURN
    );
    assert_eq!(invocations[0].requested, 0, "rights come from the lease");
    assert_eq!(invocations[0].subject.get(), source.index());
    assert_eq!(invocations[0].result.get(), result.index());
    assert_eq!(invocations[0].arg.get(), Invocation::NO_ARG);
    assert!(args.is_empty(), "the return names no second capability");

    // The two kinds are distinct and adjacent: a caller cannot reach one by
    // typo'ing the other into a neighbouring valid selector.
    assert_ne!(
        kore_abi::session::REVOKE_KIND_EXEC,
        kore_abi::session::REVOKE_KIND_EXEC_RETURN
    );
}

/// Length selection happens before the transport chooses registers or
/// scratch, but both carriers must receive byte-identical output from the one
/// encoder. `finish` writes exactly that prefix and leaves spare carrier bytes
/// untouched.
#[test]
fn encoded_len_matches_the_exact_finished_prefix() {
    let mut b = SessionBuilder::new();
    b.revoke(SlotRef::from_wire(9)).unwrap();
    let need = b.encoded_len().expect("ordinary descriptor has a length");
    assert_eq!(need, 8 + 16);

    let mut exact = [0u8; 24];
    assert_eq!(b.finish(&mut exact).unwrap(), need);

    let mut roomy = [0xA5u8; 64];
    assert_eq!(b.finish(&mut roomy).unwrap(), need);
    assert_eq!(&roomy[..need], &exact);
    assert!(roomy[need..].iter().all(|byte| *byte == 0xA5));
}

#[test]
fn exact_rendezvous_move_has_no_requested_rights_or_argument() {
    let mut b = SessionBuilder::new();
    b.send_exact(SlotRef::from_wire(4), SlotRef::from_wire(9))
        .expect("exact move stages");

    let mut out = [0u8; DESCRIPTOR_MAX];
    let n = b.finish(&mut out).expect("exact move finishes");
    let (hdr, invs, args) = parse(&out[..n]);
    assert_eq!(hdr.count.get(), 1);
    assert_eq!(invs[0].op.get(), OpCode::Rendezvous as u16);
    assert_eq!(invs[0].kind, kore_abi::session::RENDEZVOUS_KIND_EXACT_MOVE);
    assert_eq!(invs[0].requested, 0);
    assert_eq!(invs[0].subject.get(), 4);
    assert_eq!(invs[0].result.get(), 9);
    assert_eq!(invs[0].arg.get(), Invocation::NO_ARG);
    assert!(args.is_empty());
}

#[test]
fn guest_share_names_the_space_and_freezes_the_wire_shape() {
    use kore_abi::cap::ResourceKind;
    use kore_abi::session::{GuestShareReq, SHARE_ACCESS_R, SHARE_ACCESS_W};

    let lease = SlotRef::from_wire(4);
    let space = SlotRef::from_wire(11);
    let result = SlotRef::from_wire(19);
    let rights = Rights::ACCESS | Rights::MODIFY | Rights::REVOKE;
    let mut b = SessionBuilder::new();
    b.share_guest(
        lease,
        rights,
        space,
        0x20_0000,
        0x40_0000,
        SHARE_ACCESS_R | SHARE_ACCESS_W,
        result,
    )
    .expect("guest share stages");

    let mut out = [0u8; DESCRIPTOR_MAX];
    let n = b.finish(&mut out).expect("guest share finishes");
    assert_eq!(n, 8 + 16 + GuestShareReq::WIRE);
    let (hdr, invs, args) = parse(&out[..n]);
    assert_eq!(hdr.count.get(), 1);
    assert_eq!(invs[0].op.get(), OpCode::Derive as u16);
    assert_eq!(invs[0].kind, ResourceKind::GuestRegion as u8);
    assert_eq!(invs[0].requested, rights.bits());
    assert_eq!(invs[0].subject.get(), lease.index());
    assert_eq!(invs[0].result.get(), result.index());
    assert_eq!(invs[0].arg.get(), 0);
    assert_eq!(
        u32::from_le_bytes(args[0..4].try_into().unwrap()),
        space.index()
    );
    assert_eq!(&args[4..12], &0x20_0000u64.to_le_bytes());
    assert_eq!(&args[12..20], &0x40_0000u64.to_le_bytes());
    assert_eq!(args[20], SHARE_ACCESS_R | SHARE_ACCESS_W);
    assert_eq!(&args[21..24], &[0; 3]);
}

#[test]
fn task_start_is_one_bind_inert_invocation() {
    use kore_abi::session::DERIVE_KIND_TASK_START;

    let handle = SlotRef::from_wire(17);
    let mut b = SessionBuilder::new();
    b.start(handle).expect("START stages");

    let mut out = [0u8; DESCRIPTOR_MAX];
    let n = b.finish(&mut out).expect("START finishes");
    let (hdr, invs, args) = parse(&out[..n]);
    assert_eq!(hdr.count.get(), 1);
    assert_eq!(hdr.flags.get(), 0);
    assert_eq!(invs[0].op.get(), OpCode::Derive as u16);
    assert_eq!(invs[0].kind, DERIVE_KIND_TASK_START);
    assert_eq!(invs[0].requested, 0);
    assert_eq!(invs[0].subject.get(), handle.index());
    assert_eq!(
        invs[0].result.get(),
        handle.index(),
        "the effect's result name is inert"
    );
    assert_eq!(invs[0].arg.get(), Invocation::NO_ARG);
    assert!(args.is_empty());

    assert_eq!(
        b.start(handle),
        Err(BuildError::EffectMustStandAlone),
        "a retry-reporting effect cannot hide a sibling commit"
    );

    let mut followed = SessionBuilder::new();
    followed.start(handle).expect("first START stages");
    assert_eq!(
        followed.revoke(handle),
        Err(BuildError::EffectMustStandAlone),
        "no invocation may be appended after START"
    );

    let mut flagged = SessionBuilder::new();
    flagged.start(handle).expect("START stages before flags");
    flagged.wait_sleep(1);
    assert_eq!(
        flagged.finish(&mut out),
        Err(BuildError::EffectMustStandAlone),
        "parking flags cannot be appended after START"
    );
}

#[test]
fn device_quiesce_assigns_the_report_slot_and_stands_alone() {
    use kore_abi::session::DERIVE_KIND_DEVICE_QUIESCE;

    let device = SlotRef::from_wire(17);
    let report = SlotRef::from_wire(3);
    let mut b = SessionBuilder::new();
    b.quiesce_device(device, report)
        .expect("DEVICE_QUIESCE stages");

    let mut out = [0u8; DESCRIPTOR_MAX];
    let n = b.finish(&mut out).expect("DEVICE_QUIESCE finishes");
    assert_eq!(n, 8 + 16, "header + invocation, no request bytes");
    let (hdr, invs, args) = parse(&out[..n]);
    assert_eq!(hdr.count.get(), 1);
    assert_eq!(hdr.flags.get(), 0);
    assert!(args.is_empty());
    assert_eq!(invs[0].op.get(), OpCode::Derive as u16);
    assert_eq!(invs[0].kind, DERIVE_KIND_DEVICE_QUIESCE);
    assert_eq!(invs[0].requested, 0, "an effect mints nothing");
    assert_eq!(invs[0].subject.get(), device.index());
    assert_eq!(
        invs[0].result.get(),
        report.index(),
        "the verdict lands in the caller's report slot"
    );
    assert_eq!(invs[0].arg.get(), Invocation::NO_ARG, "no payload");

    assert_eq!(
        b.quiesce_device(device, report),
        Err(BuildError::EffectMustStandAlone),
        "the rendezvous cannot be staged twice"
    );
    assert_eq!(
        b.revoke(device),
        Err(BuildError::EffectMustStandAlone),
        "no invocation may follow the rendezvous"
    );
}

#[test]
fn task_observer_uses_the_high_flavor_not_terminal_found() {
    use kore_abi::session::DERIVE_KIND_TASK_OBSERVER;

    let owner = SlotRef::from_wire(17);
    let observer = SlotRef::from_wire(23);
    let mut b = SessionBuilder::new();
    b.derive_task_observer(owner, observer)
        .expect("task observer stages");

    let mut out = [0u8; DESCRIPTOR_MAX];
    let n = b.finish(&mut out).expect("task observer finishes");
    let (hdr, invs, args) = parse(&out[..n]);
    assert_eq!(hdr.count.get(), 1);
    assert_eq!(invs[0].op.get(), OpCode::Derive as u16);
    assert_eq!(invs[0].kind, DERIVE_KIND_TASK_OBSERVER);
    assert_ne!(
        invs[0].kind,
        kore_abi::cap::ResourceKind::Task as u8,
        "low-band Task is the terminal Found operation"
    );
    assert_eq!(
        invs[0].requested,
        kore_abi::cap::Rights::ACCESS.bits(),
        "the sibling waiter receives observation only"
    );
    assert_eq!(invs[0].subject.get(), owner.index());
    assert_eq!(invs[0].result.get(), observer.index());
    assert_eq!(invs[0].arg.get(), Invocation::NO_ARG);
    assert!(args.is_empty());
}

#[test]
fn device_activation_names_the_exact_dma_lease_and_stands_alone() {
    use kore_abi::session::DERIVE_KIND_DEVICE_ACTIVATE;

    let device = SlotRef::from_wire(17);
    let lease = SlotRef::from_wire(23);
    let mut b = SessionBuilder::new();
    b.activate_device(device, lease)
        .expect("DEVICE_ACTIVATE stages");

    let mut out = [0u8; DESCRIPTOR_MAX];
    let n = b.finish(&mut out).expect("DEVICE_ACTIVATE finishes");
    assert_eq!(n, 8 + 16 + 8, "header + invocation + request");
    let (hdr, invs, args) = parse(&out[..n]);
    assert_eq!(hdr.count.get(), 1);
    assert_eq!(hdr.flags.get(), 0);
    assert_eq!(invs[0].op.get(), OpCode::Derive as u16);
    assert_eq!(invs[0].kind, DERIVE_KIND_DEVICE_ACTIVATE);
    assert_eq!(invs[0].requested, 0);
    assert_eq!(invs[0].subject.get(), device.index());
    assert_eq!(
        invs[0].result.get(),
        device.index(),
        "the effect's result name is inert"
    );
    assert_eq!(invs[0].arg.get(), 0);
    assert_eq!(
        u32::from_le_bytes(args[0..4].try_into().unwrap()),
        lease.index()
    );
    assert_eq!(&args[4..8], &[0; 4], "reserved bytes stay zero");

    assert_eq!(
        b.activate_device(device, lease),
        Err(BuildError::EffectMustStandAlone),
        "activation cannot be staged twice"
    );
    assert_eq!(
        b.revoke(device),
        Err(BuildError::EffectMustStandAlone),
        "no invocation may follow activation"
    );

    let mut preceded = SessionBuilder::new();
    preceded.revoke(device).expect("ordinary invocation stages");
    assert_eq!(
        preceded.activate_device(device, lease),
        Err(BuildError::EffectMustStandAlone),
        "activation cannot hide an earlier committed mutation"
    );

    let mut flagged = SessionBuilder::new();
    flagged
        .activate_device(device, lease)
        .expect("activation stages before flags");
    flagged.wait_sleep(1);
    assert_eq!(
        flagged.finish(&mut out),
        Err(BuildError::EffectMustStandAlone),
        "activation cannot carry parking flags"
    );
}

#[test]
fn typed_recv_post_stages_kind_and_minimum_rights() {
    use kore_abi::cap::ResourceKind;
    use kore_abi::session::RecvPolicyReq;
    use zerocopy::TryFromBytes;

    let mut b = SessionBuilder::new();
    b.post_recv_typed(
        SlotRef::from_wire(4),
        SlotRef::from_wire(9),
        ResourceKind::LeasedRegion,
        Rights::ACCESS | Rights::MODIFY,
    )
    .expect("typed post stages");

    let mut out = [0u8; DESCRIPTOR_MAX];
    let n = b.finish(&mut out).expect("typed post finishes");
    let (_, invs, args) = parse(&out[..n]);
    assert_eq!(invs.len(), 1);
    assert_eq!(invs[0].op.get(), OpCode::Rendezvous as u16);
    assert_eq!((invs[0].subject.get(), invs[0].result.get()), (4, 9));
    assert_eq!(invs[0].requested, 0);
    assert_eq!(invs[0].arg.get(), 0);

    let (req, _) = RecvPolicyReq::try_ref_from_prefix(args).expect("typed post policy");
    assert_eq!(req.kind.get(), ResourceKind::LeasedRegion as u16);
    assert_eq!(req.minimum_rights, (Rights::ACCESS | Rights::MODIFY).bits());
    assert_eq!(req.reserved, [0u8; 5]);

    let mut untyped = SessionBuilder::new();
    untyped
        .post_recv(SlotRef::from_wire(4), SlotRef::from_wire(9))
        .unwrap();
    let n = untyped.finish(&mut out).unwrap();
    let (_, invs, args) = parse(&out[..n]);
    assert_eq!(invs[0].arg.get(), Invocation::NO_ARG);
    assert!(
        args.is_empty(),
        "untyped post remains byte-for-byte unchanged"
    );
}

#[test]
fn inventory_queries_preserve_record_and_count_modes() {
    use kore_abi::session::{
        DERIVE_KIND_DEVICE_RECORD, INVENTORY_QUERY_COUNT, INVENTORY_QUERY_RECORD, InventoryReq,
    };
    use zerocopy::TryFromBytes;

    let mut b = SessionBuilder::new();
    b.read_device_record(SlotRef::from_wire(3), 17, SlotRef::from_wire(8))
        .expect("record query stages");
    b.read_inventory_len(SlotRef::from_wire(3), SlotRef::from_wire(9))
        .expect("count query stages");

    let mut out = [0u8; DESCRIPTOR_MAX];
    let n = b.finish(&mut out).expect("queries finish");
    let (_, invs, args) = parse(&out[..n]);
    assert_eq!(invs.len(), 2);
    assert_eq!(invs[0].kind, DERIVE_KIND_DEVICE_RECORD);
    assert_eq!(invs[1].kind, DERIVE_KIND_DEVICE_RECORD);

    let (record, tail) = InventoryReq::try_ref_from_prefix(args).expect("record request");
    let (count, _) = InventoryReq::try_ref_from_prefix(tail).expect("count request");
    assert_eq!(record.index.get(), 17);
    assert_eq!(record.query, INVENTORY_QUERY_RECORD);
    assert_eq!(record.reserved, [0; 3]);
    assert_eq!(count.index.get(), 0);
    assert_eq!(count.query, INVENTORY_QUERY_COUNT);
    assert_eq!(count.reserved, [0; 3]);
}

#[test]
fn device_claim_query_is_an_access_only_root_observation() {
    let mut builder = SessionBuilder::new();
    builder
        .read_device_claim(SlotRef::from_wire(3), 17, SlotRef::from_wire(8))
        .unwrap();
    let mut out = [0u8; DESCRIPTOR_MAX];
    let n = builder.finish(&mut out).unwrap();
    let (_, invs, args) = parse(&out[..n]);
    assert_eq!(invs.len(), 1);
    assert_eq!(invs[0].kind, kore_abi::session::DERIVE_KIND_DEVICE_RECORD);
    assert_eq!(invs[0].requested, 0);
    assert_eq!(
        args,
        &[
            17,
            0,
            0,
            0,
            kore_abi::session::INVENTORY_QUERY_DEVICE_CLAIM,
            0,
            0,
            0
        ]
    );
}

/// The park block (Phase A, `res/USERLAND.md` §3.4.2/3): `park_standing`
/// + `wait_sleep` emit `ParkReq ‖ SleepReq` at the argument region's
/// FRONT with the flags set, and every staged invocation's `arg` offset
/// is relocated past the block — exactly the kernel's flag-gated decode
/// convention (`kore-abi::session`'s table).
#[test]
fn finish_emits_the_park_block_and_relocates_args() {
    use kore_abi::session::{NoticeReq, ParkReq, SleepReq};
    use zerocopy::TryFromBytes;

    let mut b = SessionBuilder::new();
    // A staged arg payload BEFORE the park methods — relocation must
    // carry it past the block.
    b.notice(SlotRef::from_wire(2), 0xAB, 0xCD).unwrap();
    b.park_standing(7, 5, 0x1234).wait_sleep(9_000);

    let mut out = [0u8; DESCRIPTOR_MAX];
    let n = b.finish(&mut out).unwrap();
    assert_eq!(
        n,
        8 + 16 + (16 + 8) + 16,
        "header + 1 inv + park block + NoticeReq"
    );

    let (hdr, invs, args) = parse(&out[..n]);
    assert_eq!(
        hdr.flags.get(),
        (SessionFlags::WAIT_DELIVERY | SessionFlags::WAIT_SLEEP).bits()
    );

    // The park block heads the args: ParkReq first, SleepReq adjacent.
    let (park, rest) = ParkReq::try_ref_from_prefix(args).expect("ParkReq at args[0]");
    assert_eq!(
        (park.chan.get(), park.slot.get(), park.recv_tag.get()),
        (7, 5, 0x1234)
    );
    assert_eq!(park.reserved, [0u8; 6]);
    let (sleep, _) = SleepReq::try_ref_from_prefix(rest).expect("SleepReq after ParkReq");
    assert_eq!(sleep.deadline_ns.get(), 9_000);

    // The staged NoticeReq was relocated past the 24-byte block.
    assert_eq!(invs[0].arg.get(), 24, "arg offset shifted by the block");
    let (req, _) = NoticeReq::try_ref_from_prefix(&args[invs[0].arg.get() as usize..])
        .expect("NoticeReq where the relocated arg points");
    assert_eq!((req.cookie.get(), req.payload.get()), (0xAB, 0xCD));
}

/// The bounded receive-any form fits the smallest architecture carrier:
/// header + two standing posts + optional deadline = 48 bytes on x86-64.
#[test]
fn alternate_standing_post_stays_in_the_x86_gpr_carrier() {
    use kore_abi::session::{ParkReq, SleepReq};
    use zerocopy::TryFromBytes;

    let mut b = SessionBuilder::new();
    b.park_standing(7, 5, 0x1234)
        .park_standing_alt(9, 6, 0x5678)
        .wait_sleep(11_000);
    assert_eq!(b.encoded_len(), Ok(48));

    let mut out = [0u8; DESCRIPTOR_MAX];
    let n = b.finish(&mut out).expect("two-post park finishes");
    assert_eq!(n, 48);
    let (hdr, invs, args) = parse(&out[..n]);
    assert!(invs.is_empty());
    assert_eq!(
        hdr.flags.get(),
        (SessionFlags::WAIT_DELIVERY | SessionFlags::WAIT_DELIVERY_ALT | SessionFlags::WAIT_SLEEP)
            .bits()
    );
    let (first, rest) = ParkReq::try_ref_from_prefix(args).expect("first post");
    let (second, rest) = ParkReq::try_ref_from_prefix(rest).expect("second post");
    let (sleep, _) = SleepReq::try_ref_from_prefix(rest).expect("deadline");
    assert_eq!(
        (first.chan.get(), first.slot.get(), first.recv_tag.get()),
        (7, 5, 0x1234)
    );
    assert_eq!(
        (second.chan.get(), second.slot.get(), second.recv_tag.get()),
        (9, 6, 0x5678)
    );
    assert_eq!(sleep.deadline_ns.get(), 11_000);

    let mut no_primary = SessionBuilder::new();
    no_primary.park_standing_alt(9, 6, 0x5678);
    assert_eq!(no_primary.finish(&mut out), Err(BuildError::BadParkShape));

    let mut aliased = SessionBuilder::new();
    aliased
        .park_standing(7, 5, 0x1234)
        .park_standing_alt(7, 6, 0x5678);
    assert_eq!(aliased.finish(&mut out), Err(BuildError::BadParkShape));

    let mut nonempty = SessionBuilder::new();
    nonempty
        .park_standing(7, 5, 0x1234)
        .park_standing_alt(9, 6, 0x5678);
    nonempty.revoke(SlotRef::from_wire(3)).unwrap();
    assert_eq!(nonempty.finish(&mut out), Err(BuildError::BadParkShape));
}

/// The L4-style request crossing used by exec is one capability send followed
/// by a park on the caller's already-standing reply post. Header + invocation
/// + `ParkReq` is 40 bytes, below x86-64's six-word (48-byte) carrier, so the
/// hottest architecture never falls back to shared descriptor scratch.
#[test]
fn rendezvous_send_and_wait_stays_in_the_x86_gpr_carrier() {
    use kore_abi::session::ParkReq;
    use zerocopy::TryFromBytes;

    let mut b = SessionBuilder::new();
    b.send(
        SlotRef::from_wire(7),
        SlotRef::from_wire(8),
        Rights::ACCESS | Rights::REVOKE,
    )
    .expect("one rendezvous send fits");
    b.park_standing(11, 9, 0x1234_5678);
    assert_eq!(b.encoded_len(), Ok(40));

    let mut out = [0u8; DESCRIPTOR_MAX];
    let n = b.finish(&mut out).expect("send-and-wait finishes");
    assert_eq!(n, 40);
    let (hdr, invs, args) = parse(&out[..n]);
    assert_eq!(hdr.flags.get(), SessionFlags::WAIT_DELIVERY.bits());
    assert_eq!(invs.len(), 1);
    assert_eq!(invs[0].op.get(), OpCode::Rendezvous as u16);
    assert_eq!(invs[0].subject.get(), 7);
    assert_eq!(invs[0].result.get(), 8);
    assert_eq!(invs[0].arg.get(), Invocation::NO_ARG);
    let (park, rest) = ParkReq::try_ref_from_prefix(args).expect("standing reply post");
    assert!(rest.is_empty());
    assert_eq!(
        (park.chan.get(), park.slot.get(), park.recv_tag.get()),
        (11, 9, 0x1234_5678)
    );
}

/// The standardized request-ring kick uses the compact rendezvous flavor:
/// alone it is 24 bytes, and folded with a reply park it matches the 40-byte
/// send-and-wait shape instead of spilling a generic NoticeReq to scratch.
#[test]
fn request_kick_and_wait_stays_in_the_x86_gpr_carrier() {
    use kore_abi::session::ParkReq;
    use zerocopy::TryFromBytes;

    let mut b = SessionBuilder::new();
    b.request_kick(SlotRef::from_wire(7))
        .expect("one compact request kick fits");
    assert_eq!(b.encoded_len(), Ok(24));
    b.park_standing(11, 9, 0x1234_5678);
    assert_eq!(b.encoded_len(), Ok(40));

    let mut out = [0u8; DESCRIPTOR_MAX];
    let n = b.finish(&mut out).expect("kick-and-wait finishes");
    let (hdr, invs, args) = parse(&out[..n]);
    assert_eq!(hdr.flags.get(), SessionFlags::WAIT_DELIVERY.bits());
    assert_eq!(invs.len(), 1);
    assert_eq!(invs[0].op.get(), OpCode::Rendezvous as u16);
    assert_eq!(
        invs[0].kind,
        kore_abi::session::RENDEZVOUS_KIND_REQUEST_KICK
    );
    assert_eq!(invs[0].subject.get(), 7);
    assert_eq!(invs[0].result.get(), 0);
    assert_eq!(invs[0].arg.get(), Invocation::NO_ARG);
    let (park, rest) = ParkReq::try_ref_from_prefix(args).expect("standing reply post");
    assert!(rest.is_empty());
    assert_eq!(
        (park.chan.get(), park.slot.get(), park.recv_tag.get()),
        (11, 9, 0x1234_5678)
    );
}

#[test]
fn spawn_thread_is_the_unified_task_model_shape() {
    // #407: two thread views off the caller's own ADDRESS_SPACE slot, the
    // pre-written boot page delegated into the staged view, and a Found
    // carrying SHARE_SPACE + the NO_AUTHORITY thread sentinel.
    let mut b = SessionBuilder::new();
    let self_space = SlotRef::from_wire(1);
    flows::spawn_thread(
        &mut b,
        self_space,
        0x4000_1000, // entry (the trampoline)
        0x7fff_fd01_fff0,
        false,                  // an ordinary worker, no SPMD pin
        SlotRef::from_wire(10), // the pre-written boot Region
        0x6a00_1000,
        4096,
        64,
        SlotRef::from_wire(20),
        SlotRef::from_wire(21),
        SlotRef::from_wire(22),
        SlotRef::from_wire(23),
        1,
        2,
        &[],
    )
    .expect("stages");
    let mut out = [0u8; DESCRIPTOR_MAX];
    let n = b.finish(&mut out).expect("finishes");
    let (_, invs, args) = parse(&out[..n]);
    // view + view + delegate + found.
    assert_eq!(invs.len(), 4);
    assert_eq!(invs[0].subject.get(), self_space.index(), "view off self");
    // The second view derives off the STAGED ROOT (its result slot), not
    // the caller's view — the T6 lineage the child's adopt demands.
    assert_eq!(
        invs[1].subject.get(),
        20,
        "second view derives off the staged root"
    );
    let found = &invs[3];
    let at = found.arg.get() as usize;
    let (req, _) = kore_abi::session::FoundReq::ref_from_prefix(&args[at..])
        .expect("a FoundReq rides the found invocation");
    assert_eq!(
        req.authority.get(),
        kore_abi::session::NO_AUTHORITY,
        "a thread founding carries the sentinel, not an ExecAuthority slot"
    );
    let flags = kore_abi::task::SpawnFlags::from_bits(req.spawn.flags.get()).expect("valid flags");
    assert!(
        flags.contains(kore_abi::task::SpawnFlags::SHARE_SPACE),
        "a thread shares its parent's space"
    );
    assert_eq!(
        req.deliver_count.get(),
        2,
        "the child's self-view and its own task handle both deliver"
    );
    // The SELF handle rides the sentinel, not an SSA name: the capability
    // it asks for is minted after the run-table intake, past anything this
    // session can name.
    assert_eq!(
        req.deliver[1].from.get(),
        kore_abi::session::DELIVER_FROM_SELF_TASK,
        "the thread's own handle comes from the founding, not a staged child"
    );
    assert_eq!(
        req.deliver[1].to.get(),
        2,
        "into the child's role::SELF slot"
    );
}

#[test]
fn an_spmd_worker_founding_carries_the_pin_flag() {
    // G6 (`res/SPMD.md` §7): the SPMD launch rides one appended
    // SpawnFlags bit — SHARE_SPACE stays set (the dispatch gate refuses
    // SPMD_WORKER on anything but a thread founding), and the kernel's
    // class-homogeneous single-core pin keys on the new bit.
    let mut b = SessionBuilder::new();
    flows::spawn_thread(
        &mut b,
        SlotRef::from_wire(1),
        0x4000_1000,
        0x7fff_fd01_fff0,
        true, // an SPMD worker: pin me
        SlotRef::from_wire(10),
        0x6a00_1000,
        4096,
        64,
        SlotRef::from_wire(20),
        SlotRef::from_wire(21),
        SlotRef::from_wire(22),
        SlotRef::from_wire(23),
        1,
        2,
        &[],
    )
    .expect("stages");
    let mut out = [0u8; DESCRIPTOR_MAX];
    let n = b.finish(&mut out).expect("finishes");
    let (_, invs, args) = parse(&out[..n]);
    let found = &invs[3];
    let at = found.arg.get() as usize;
    let (req, _) = kore_abi::session::FoundReq::ref_from_prefix(&args[at..])
        .expect("a FoundReq rides the found invocation");
    let flags = kore_abi::task::SpawnFlags::from_bits(req.spawn.flags.get()).expect("valid flags");
    assert!(
        flags.contains(kore_abi::task::SpawnFlags::SHARE_SPACE),
        "an SPMD worker is still a thread founding"
    );
    assert!(
        flags.contains(kore_abi::task::SpawnFlags::SPMD_WORKER),
        "the pin flag rides the wire"
    );
}

#[test]
fn a_gang_founding_stages_legs_then_the_found_tail() {
    // G5 (`res/SPMD.md` §7): N workers, one session — every non-Found
    // invocation first (two views + one delegate per worker, in worker
    // order), then the Found tail: one FoundReq per worker, each with
    // its own delivery map, its own stack, and the pin flag.
    let mut b = SessionBuilder::new();
    let self_space = SlotRef::from_wire(1);
    let mk = |k: u32| flows::GangWorker {
        entry: 0x4000_1000,
        stack: 0x7fff_fd01_fff0 - u64::from(k) * 0x1_0000,
        space_slot: SlotRef::from_wire(20 + 8 * k),
        child_view_slot: SlotRef::from_wire(21 + 8 * k),
        boot_region: SlotRef::from_wire(10),
        boot_va: 0x6a00_1000,
        boot_bytes: 4096,
        boot_result: SlotRef::from_wire(22 + 8 * k),
        handle_slot: SlotRef::from_wire(23 + 8 * k),
        endow: &[],
    };
    let workers = [mk(0), mk(1), mk(2)];
    flows::spawn_gang(&mut b, self_space, true, 64, 1, 2, &workers).expect("stages");
    let mut out = [0u8; DESCRIPTOR_MAX];
    let n = b.finish(&mut out).expect("finishes");
    let (_, invs, args) = parse(&out[..n]);
    // Three legs per worker, then the three-Found tail.
    assert_eq!(invs.len(), 12, "3 workers × 3 legs, then 3 Founds");
    for k in 0..3usize {
        assert_eq!(
            invs[3 * k].subject.get(),
            1,
            "leg run k opens with a view off the caller's space"
        );
    }
    for k in 0..3usize {
        let found = &invs[9 + k];
        let at = found.arg.get() as usize;
        let (req, _) = kore_abi::session::FoundReq::ref_from_prefix(&args[at..])
            .expect("a FoundReq rides each tail invocation");
        let flags =
            kore_abi::task::SpawnFlags::from_bits(req.spawn.flags.get()).expect("valid flags");
        assert!(
            flags.contains(kore_abi::task::SpawnFlags::SHARE_SPACE),
            "every gang worker is a thread founding"
        );
        assert!(
            flags.contains(kore_abi::task::SpawnFlags::SPMD_WORKER),
            "every gang worker rides the pin flag"
        );
        assert_eq!(
            req.authority.get(),
            kore_abi::session::NO_AUTHORITY,
            "a thread founding carries the sentinel"
        );
        assert_eq!(
            req.deliver_count.get(),
            2,
            "each map delivers the self-view and the worker's own handle"
        );
        assert_eq!(
            req.deliver[1].from.get(),
            kore_abi::session::DELIVER_FROM_SELF_TASK,
            "the worker's own handle comes from ITS founding"
        );
        assert_eq!(
            req.spawn.stack.get(),
            0x7fff_fd01_fff0 - (k as u64) * 0x1_0000,
            "each Found carries its own worker's stack"
        );
    }
}

#[test]
fn the_thirteenth_found_refuses_typed() {
    // `MAX_SESSION_FOUNDS` is the descriptor's exact founding capacity
    // (the two-page scratch window); the typed refusal names the remedy
    // — chunk the launch — rather than surfacing as a bare overflow.
    let mut b = SessionBuilder::new();
    let mk = |k: u32| flows::GangWorker {
        entry: 0x4000_1000,
        stack: 0x7fff_fd01_fff0,
        space_slot: SlotRef::from_wire(20 + 8 * k),
        child_view_slot: SlotRef::from_wire(21 + 8 * k),
        boot_region: SlotRef::from_wire(10),
        boot_va: 0x6a00_1000,
        boot_bytes: 4096,
        boot_result: SlotRef::from_wire(22 + 8 * k),
        handle_slot: SlotRef::from_wire(23 + 8 * k),
        endow: &[],
    };
    let workers: [flows::GangWorker<'_>; 13] = core::array::from_fn(|k| mk(k as u32));
    assert_eq!(
        flows::spawn_gang(&mut b, SlotRef::from_wire(1), true, 64, 1, 2, &workers),
        Err(BuildError::TooManyFounds),
        "the thirteenth Found refuses with the chunking remedy named"
    );
}

#[test]
fn an_endowed_thread_founding_derives_into_the_child_arena() {
    // A relay worker holds its own device: sharing an address space
    // shares memory, never authority, so the parent stages one derive
    // per endowment and delivers the result into the child's arena.
    let mut b = SessionBuilder::new();
    let self_space = SlotRef::from_wire(1);
    let endow = [flows::ThreadEndow {
        subject: SlotRef::from_wire(7), // the parent's platform device
        rights: kore_abi::cap::Rights::ACCESS
            | kore_abi::cap::Rights::MODIFY
            | kore_abi::cap::Rights::DERIVE,
        kind: kore_abi::cap::ResourceKind::PlatformDevice as u16 as u8,
        result: SlotRef::from_wire(24),
        child_slot: 3,
    }];
    flows::spawn_thread(
        &mut b,
        self_space,
        0x4000_1000,
        0x7fff_fd01_fff0,
        false, // an ordinary worker, no SPMD pin
        SlotRef::from_wire(10),
        0x6a00_1000,
        4096,
        64,
        SlotRef::from_wire(20),
        SlotRef::from_wire(21),
        SlotRef::from_wire(22),
        SlotRef::from_wire(23),
        1,
        2,
        &endow,
    )
    .expect("stages");
    let mut out = [0u8; DESCRIPTOR_MAX];
    let n = b.finish(&mut out).expect("finishes");
    let (_, invs, args) = parse(&out[..n]);
    // view + view + delegate + the endowment derive + found.
    assert_eq!(invs.len(), 5);
    assert_eq!(
        invs[3].subject.get(),
        7,
        "the endowment derives off the PARENT's held subject, not the staged root"
    );
    let found = &invs[4];
    let at = found.arg.get() as usize;
    let (req, _) = kore_abi::session::FoundReq::ref_from_prefix(&args[at..])
        .expect("a FoundReq rides the found invocation");
    assert_eq!(
        req.deliver_count.get(),
        3,
        "the self-view, the SELF handle, and the endowment all deliver"
    );
}

#[test]
fn a_thread_founding_refuses_more_endowments_than_it_can_carry() {
    // Bounded, and a refusal rather than a truncation: a worker founded
    // without the capability it asked for would fail later, elsewhere,
    // for a reason that no longer names this decision.
    let mut b = SessionBuilder::new();
    let one = flows::ThreadEndow {
        subject: SlotRef::from_wire(7),
        rights: kore_abi::cap::Rights::ACCESS,
        kind: kore_abi::cap::ResourceKind::PlatformDevice as u16 as u8,
        result: SlotRef::from_wire(24),
        child_slot: 2,
    };
    let too_many = [one; flows::MAX_THREAD_ENDOW + 1];
    assert!(
        flows::spawn_thread(
            &mut b,
            SlotRef::from_wire(1),
            0x4000_1000,
            0x7fff_fd01_fff0,
            false, // an ordinary worker, no SPMD pin
            SlotRef::from_wire(10),
            0x6a00_1000,
            4096,
            64,
            SlotRef::from_wire(20),
            SlotRef::from_wire(21),
            SlotRef::from_wire(22),
            SlotRef::from_wire(23),
            1,
            2,
            &too_many,
        )
        .is_err()
    );
}

#[test]
fn device_setup_is_the_whole_bring_up_in_one_session() {
    let mut b = SessionBuilder::new();
    let parts = flows::device_setup(
        &mut b,
        DeviceSetup {
            device: SlotRef::from_wire(4), // role::DEVICE
            region: SlotRef::from_wire(6), // role::DMA_POOL
            bar: 0,
            regs_va: 0x4000_0000,
            regs_bytes: 0x4000,
            buf_size: 1 << 20,
            buf_va: 0x5000_0000,
            iova: 0x8000_0000,
            dma_write: true,
        },
        core::array::from_fn(|i| SlotRef::from_wire(30 + i as u32)),
    )
    .unwrap();

    let mut out = [0u8; DESCRIPTOR_MAX];
    let n = b.finish(&mut out).unwrap();
    let (hdr, invs, args) = parse(&out[..n]);
    assert_eq!(hdr.count.get(), 6, "6 invocations, 1 trap");

    // Both shares hang off the lease (RUNG 2: one root, two windows).
    assert_eq!(invs[3].subject.get(), parts.lease.slot().index());
    assert_eq!(invs[4].subject.get(), parts.lease.slot().index());
    // The device share is kind-disambiguated to the IOVA leaf.
    assert_eq!(
        invs[4].kind,
        (kore_abi::cap::ResourceKind::IovaRegion as u16) as u8
    );
    // Its DeviceShareReq names the device slot as a read edge.
    let off = invs[4].arg.get() as usize;
    assert_eq!(
        u32::from_le_bytes(args[off..off + 4].try_into().unwrap()),
        4
    );
    assert_eq!(
        args[off + 20],
        kore_abi::session::DeviceShareReq::ACCESS_READ_WRITE,
    );
    // The IRQ route is kind-selected and carries the default block request.
    assert_eq!(
        invs[5].kind,
        (kore_abi::cap::ResourceKind::IrqRoute as u16) as u8
    );
    let irq = invs[5].arg.get() as usize;
    assert_eq!(
        u16::from_le_bytes(args[irq..irq + 2].try_into().unwrap()),
        0
    );
    assert_eq!(
        u16::from_le_bytes(args[irq + 2..irq + 4].try_into().unwrap()),
        1
    );
    assert_eq!(
        u32::from_le_bytes(args[irq + 4..irq + 8].try_into().unwrap()),
        kore_abi::session::IRQ_AFFINITY_ANY
    );
}

#[test]
fn write_only_device_share_is_explicit_and_v4() {
    use kore_abi::session::DeviceShareReq;
    use zerocopy::TryFromBytes;

    let mut b = SessionBuilder::new();
    b.share_device_write_only(
        SlotRef::from_wire(3),
        Rights::ACCESS | Rights::MODIFY,
        SlotRef::from_wire(4),
        0x8000_0000,
        0x2000,
        SlotRef::from_wire(9),
    )
    .unwrap();

    let mut out = [0u8; DESCRIPTOR_MAX];
    let n = b.finish(&mut out).unwrap();
    let (header, invs, args) = parse(&out[..n]);
    assert_eq!(header.abi.get(), 4);
    assert_eq!(invs.len(), 1);
    assert_eq!(
        invs[0].kind,
        (kore_abi::cap::ResourceKind::IovaRegion as u16) as u8,
    );
    let off = invs[0].arg.get() as usize;
    let (req, _) = DeviceShareReq::try_ref_from_prefix(&args[off..]).expect("DeviceShareReq");
    assert_eq!(req.access, DeviceShareReq::ACCESS_WRITE);
    assert_eq!(req.reserved, [0; 3]);
}

#[test]
fn irq_route_request_carries_block_and_affinity() {
    let mut b = SessionBuilder::new();
    b.derive_irq_route_with(
        SlotRef::from_wire(4),
        Rights::ACCESS | Rights::REVOKE,
        SlotRef::from_wire(9),
        8,
        4,
        Some(3),
    )
    .unwrap();
    let mut out = [0u8; DESCRIPTOR_MAX];
    let n = b.finish(&mut out).unwrap();
    let (_, invs, args) = parse(&out[..n]);
    let off = invs[0].arg.get() as usize;
    assert_eq!(
        u16::from_le_bytes(args[off..off + 2].try_into().unwrap()),
        8
    );
    assert_eq!(
        u16::from_le_bytes(args[off + 2..off + 4].try_into().unwrap()),
        4
    );
    assert_eq!(
        u32::from_le_bytes(args[off + 4..off + 8].try_into().unwrap()),
        3
    );
}

#[test]
fn pci_firmware_line_route_is_arg_free_and_preserves_authority() {
    let mut builder = SessionBuilder::new();
    let device = SlotRef::from_wire(7);
    let result = SlotRef::from_wire(11);
    let rights = Rights::ACCESS | Rights::REVOKE;
    builder
        .derive_pci_line_route(device, rights, result)
        .unwrap();
    let mut out = [0; DESCRIPTOR_MAX];
    let len = builder.finish(&mut out).unwrap();
    let (_, invocations, args) = parse(&out[..len]);
    assert_eq!(invocations.len(), 1);
    assert_eq!(invocations[0].subject.get(), device.index());
    assert_eq!(invocations[0].result.get(), result.index());
    assert_eq!(invocations[0].requested, rights.bits());
    assert_eq!(
        invocations[0].kind,
        kore_abi::session::DERIVE_KIND_IRQ_ROUTE_LINE
    );
    assert_eq!(invocations[0].arg.get(), u32::MAX);
    assert!(args.is_empty(), "no caller-supplied GSI, polarity, or GPIO");
}

#[test]
fn i2c_child_queries_preserve_the_exact_controller_and_canonical_selector() {
    use kore_abi::session::{
        DERIVE_KIND_I2C_HID_RECORD, I2C_HID_QUERY_COUNT, I2C_HID_QUERY_RECORD,
    };
    let mut builder = SessionBuilder::new();
    let controller = SlotRef::from_wire(7);
    builder
        .read_i2c_hid_len(controller, SlotRef::from_wire(8))
        .unwrap();
    builder
        .read_i2c_hid_record(controller, 3, SlotRef::from_wire(9))
        .unwrap();
    let mut out = [0; DESCRIPTOR_MAX];
    let len = builder.finish(&mut out).unwrap();
    let (_, invocations, args) = parse(&out[..len]);
    assert_eq!(invocations.len(), 2);
    for (invocation, index, query) in [
        (&invocations[0], 0u32, I2C_HID_QUERY_COUNT),
        (&invocations[1], 3u32, I2C_HID_QUERY_RECORD),
    ] {
        assert_eq!(invocation.kind, DERIVE_KIND_I2C_HID_RECORD);
        assert_eq!(invocation.subject.get(), controller.index());
        assert_eq!(invocation.requested, Rights::empty().bits());
        let offset = invocation.arg.get() as usize;
        assert_eq!(&args[offset..offset + 4], &index.to_le_bytes());
        assert_eq!(args[offset + 4], query);
        assert_eq!(&args[offset + 5..offset + 8], &[0; 3]);
    }
}

#[test]
fn strict_msix_route_preserves_the_frozen_request_and_changes_only_the_flavor() {
    let mut b = SessionBuilder::new();
    b.derive_msix_route_with(
        SlotRef::from_wire(4),
        Rights::ACCESS | Rights::REVOKE,
        SlotRef::from_wire(9),
        0,
        1,
        None,
    )
    .unwrap();
    let mut out = [0u8; DESCRIPTOR_MAX];
    let n = b.finish(&mut out).unwrap();
    let (_, invs, args) = parse(&out[..n]);
    assert_eq!(invs[0].kind, kore_abi::session::DERIVE_KIND_IRQ_ROUTE_MSIX);
    let off = invs[0].arg.get() as usize;
    assert_eq!(&args[off..off + 2], &0u16.to_le_bytes());
    assert_eq!(&args[off + 2..off + 4], &1u16.to_le_bytes());
    assert_eq!(
        &args[off + 4..off + 8],
        &kore_abi::session::IRQ_AFFINITY_ANY.to_le_bytes()
    );
}

#[test]
fn share_with_stages_lease_window_and_transfer() {
    let mut b = SessionBuilder::new();
    let parts = flows::share_with(
        &mut b,
        SlotRef::from_wire(5),
        SlotRef::from_wire(7), // peer channel send end
        4096,
        0x6000_0000,
        Rights::ACCESS, // peer gets a read-only child
        [SlotRef::from_wire(40), SlotRef::from_wire(41)],
    )
    .unwrap();

    let mut out = [0u8; DESCRIPTOR_MAX];
    let n = b.finish(&mut out).unwrap();
    let (_, invs, _) = parse(&out[..n]);

    // The transfer re-derives the lease, attenuated — never the bytes.
    assert_eq!(invs[2].op.get(), OpCode::Rendezvous as u16);
    assert_eq!(invs[2].subject.get(), 7, "channel send end");
    assert_eq!(invs[2].result.get(), parts.lease.slot().index());
    assert_eq!(invs[2].requested, Rights::ACCESS.bits());
}

#[test]
fn sessions_are_bounded_and_buffers_checked() {
    let mut b = SessionBuilder::new();
    for i in 0..MAX_INVOCATIONS {
        b.revoke(SlotRef::from_wire(i as u32)).unwrap();
    }
    assert_eq!(
        b.revoke(SlotRef::from_wire(999)),
        Err(BuildError::TooManyInvocations),
        "I9: split the batch, don't grow it"
    );

    let mut tiny = [0u8; 16];
    assert_eq!(b.finish(&mut tiny), Err(BuildError::OutTooSmall));
}

#[test]
fn dma_buffers_stages_lease_cpu_iova_triples_per_buffer() {
    use flows::{BufSpec, DmaWindows, INVS_PER_BUFFER, dma_buffers};
    let mut b = SessionBuilder::new();
    let specs = [
        BufSpec {
            size: 4096,
            cpu_va: 0x1000,
            iova: 0x8000,
            write: true,
        },
        BufSpec {
            size: 8192,
            cpu_va: 0x2000,
            iova: 0x9000,
            write: false,
        },
    ];
    let results: [SlotRef; 6] = core::array::from_fn(|i| SlotRef::from_wire(20 + i as u32));
    let mut out = [DmaWindows {
        lease: Staged(SlotRef::from_wire(0)),
        cpu: Staged(SlotRef::from_wire(0)),
        iova: Staged(SlotRef::from_wire(0)),
    }; 2];
    let n = dma_buffers(
        &mut b,
        SlotRef::from_wire(6),
        SlotRef::from_wire(4),
        &specs,
        &results,
        &mut out,
    )
    .unwrap();
    assert_eq!(n, 2);
    assert_eq!(
        b.staged(),
        2 * INVS_PER_BUFFER,
        "3 invocations per buffer, one crossing"
    );

    let mut buf = [0u8; DESCRIPTOR_MAX];
    let len = b.finish(&mut buf).unwrap();
    let (hdr, invs, _) = parse(&buf[..len]);
    assert_eq!(hdr.count.get(), 6);

    // Buffer 0: lease (inv 0), then its CPU + IOVA windows hang off it.
    assert_eq!(invs[0].subject.get(), 6, "lease of the region");
    assert_eq!(
        invs[1].subject.get(),
        out[0].lease.slot().index(),
        "CPU window off the lease"
    );
    assert_eq!(
        invs[2].subject.get(),
        out[0].lease.slot().index(),
        "IOVA window off the same lease"
    );
    // The device share is the IOVA leaf, kind-disambiguated.
    assert_eq!(
        invs[2].kind,
        (kore_abi::cap::ResourceKind::IovaRegion as u16) as u8
    );
    // Buffer 1 is disjoint (its own lease).
    assert_eq!(invs[3].subject.get(), 6);
    assert_eq!(invs[4].subject.get(), out[1].lease.slot().index());
    assert_ne!(out[0].lease.slot().index(), out[1].lease.slot().index());
}

#[test]
fn dma_buffers_refuses_short_slices() {
    use flows::{BufSpec, DmaWindows, dma_buffers};
    let mut b = SessionBuilder::new();
    let specs = [BufSpec {
        size: 4096,
        cpu_va: 0,
        iova: 0,
        write: true,
    }; 2];
    let short: [SlotRef; 3] = core::array::from_fn(|i| SlotRef::from_wire(i as u32)); // need 6
    let mut out = [DmaWindows {
        lease: Staged(SlotRef::from_wire(0)),
        cpu: Staged(SlotRef::from_wire(0)),
        iova: Staged(SlotRef::from_wire(0)),
    }; 2];
    assert_eq!(
        dma_buffers(
            &mut b,
            SlotRef::from_wire(6),
            SlotRef::from_wire(4),
            &specs,
            &short,
            &mut out
        ),
        Err(BuildError::TooManyBuffers)
    );
    assert_eq!(b.staged(), 0, "a refusal stages nothing");
}

#[test]
fn nvme_bringup_is_the_whole_controller_setup_in_one_crossing() {
    use flows::{BufSpec, NvmeBringupParams, nvme_bringup};
    let mut b = SessionBuilder::new();
    let admin = |iova| BufSpec {
        size: 4096,
        cpu_va: 0x3000,
        iova,
        write: true,
    };
    let parts = nvme_bringup(
        &mut b,
        NvmeBringupParams {
            device: SlotRef::from_wire(4),
            region: SlotRef::from_wire(6),
            bar: 0,
            regs_va: 0x4000_0000,
            regs_bytes: 0x2000,
            admin_sq: admin(0x8000),
            admin_cq: admin(0x9000),
        },
        SlotRef::from_wire(30),
        SlotRef::from_wire(31),
        &core::array::from_fn(|i| SlotRef::from_wire(40 + i as u32)),
        SlotRef::from_wire(50),
    )
    .unwrap();

    let mut buf = [0u8; DESCRIPTOR_MAX];
    let len = b.finish(&mut buf).unwrap();
    let (hdr, invs, _) = parse(&buf[..len]);
    assert_eq!(
        hdr.count.get(),
        9,
        "bar + regs + 2 buffers×3 + irq, one trap"
    );

    // The register window maps off the claimed BAR (SSA).
    assert_eq!(invs[0].op.get(), OpCode::Derive as u16, "claim_bar");
    assert_eq!(
        invs[0].kind,
        kore_abi::cap::ResourceKind::MmioRegion as u8,
        "new BAR claims name their typed result instead of the legacy zero selector"
    );
    assert_eq!(
        invs[1].subject.get(),
        parts.bar.slot().index(),
        "regs map off the BAR"
    );
    // Both admin-SQ windows hang off the admin-SQ lease (RUNG 2).
    assert_eq!(invs[3].subject.get(), parts.admin_sq.lease.slot().index());
    assert_eq!(invs[4].subject.get(), parts.admin_sq.lease.slot().index());
    // The IRQ route is the kind-selected last leg.
    assert_eq!(
        invs[8].kind,
        (kore_abi::cap::ResourceKind::IrqRoute as u16) as u8
    );
    assert_eq!(invs[8].result.get(), parts.irq.slot().index());
}

#[test]
fn nvme_qpair_is_six_invocations_one_crossing() {
    use flows::{BufSpec, nvme_qpair};
    let mut b = SessionBuilder::new();
    let spec = |iova| BufSpec {
        size: 16384,
        cpu_va: 0x5000,
        iova,
        write: true,
    };
    let qp = nvme_qpair(
        &mut b,
        SlotRef::from_wire(6),
        SlotRef::from_wire(4),
        spec(0xA000),
        spec(0xB000),
        &core::array::from_fn(|i| SlotRef::from_wire(60 + i as u32)),
    )
    .unwrap();

    let mut buf = [0u8; DESCRIPTOR_MAX];
    let len = b.finish(&mut buf).unwrap();
    let (hdr, invs, _) = parse(&buf[..len]);
    assert_eq!(hdr.count.get(), 6, "SQ + CQ, each lease+cpu+iova");
    // SQ lease then its two windows; CQ is a disjoint lease.
    assert_eq!(invs[1].subject.get(), qp.sq.lease.slot().index());
    assert_eq!(invs[2].subject.get(), qp.sq.lease.slot().index());
    assert_ne!(qp.sq.lease.slot().index(), qp.cq.lease.slot().index());
}

#[test]
fn xhci_bringup_is_fifteen_invocations_one_crossing() {
    use flows::{BufSpec, XhciBringupParams, xhci_bringup};
    let mut b = SessionBuilder::new();
    let buf = |iova| BufSpec {
        size: 4096,
        cpu_va: 0x3000,
        iova,
        write: true,
    };
    let parts = xhci_bringup(
        &mut b,
        XhciBringupParams {
            device: SlotRef::from_wire(4),
            region: SlotRef::from_wire(6),
            bar: 0,
            regs_va: 0x4000_0000,
            regs_bytes: 0x4000,
            dcbaa: buf(0x8000),
            cmd_ring: buf(0x9000),
            event_ring: buf(0xA000),
            erst: buf(0xB000),
        },
        SlotRef::from_wire(30),
        SlotRef::from_wire(31),
        &core::array::from_fn(|i| SlotRef::from_wire(40 + i as u32)),
        SlotRef::from_wire(60),
    )
    .unwrap();

    let mut buf_out = [0u8; DESCRIPTOR_MAX];
    let len = b.finish(&mut buf_out).unwrap();
    let (hdr, invs, _) = parse(&buf_out[..len]);
    assert_eq!(
        hdr.count.get(),
        15,
        "bar + regs + 4 buffers×3 + irq, one trap"
    );
    assert_eq!(invs[0].op.get(), OpCode::Derive as u16, "claim_bar");
    assert_eq!(
        invs[0].kind,
        kore_abi::cap::ResourceKind::MmioRegion as u8,
        "new BAR claims name their typed result instead of the legacy zero selector"
    );
    assert_eq!(
        invs[1].subject.get(),
        parts.bar.slot().index(),
        "regs map off the BAR"
    );
    // The event-ring windows hang off the event-ring lease (RUNG 2).
    assert_eq!(invs[9].subject.get(), parts.event_ring.lease.slot().index());
    // The IRQ route is the kind-selected last leg.
    assert_eq!(
        invs[14].kind,
        (kore_abi::cap::ResourceKind::IrqRoute as u16) as u8
    );
    assert_eq!(invs[14].result.get(), parts.irq.slot().index());
}

#[test]
fn xhci_device_is_nine_invocations_one_crossing() {
    use flows::{BufSpec, xhci_device};
    let mut b = SessionBuilder::new();
    let buf = |iova| BufSpec {
        size: 4096,
        cpu_va: 0x5000,
        iova,
        write: true,
    };
    let d = xhci_device(
        &mut b,
        SlotRef::from_wire(6),
        SlotRef::from_wire(4),
        buf(0xC000),
        buf(0xD000),
        buf(0xE000),
        &core::array::from_fn(|i| SlotRef::from_wire(70 + i as u32)),
    )
    .unwrap();

    let mut buf_out = [0u8; DESCRIPTOR_MAX];
    let len = b.finish(&mut buf_out).unwrap();
    let (hdr, invs, _) = parse(&buf_out[..len]);
    assert_eq!(
        hdr.count.get(),
        9,
        "ctx + EP0 ring + data, each lease+cpu+iova"
    );
    // The context page's two windows hang off its lease; the three
    // buffers are disjoint leases (one crossing, three tenants).
    assert_eq!(invs[1].subject.get(), d.ctx.lease.slot().index());
    assert_eq!(invs[2].subject.get(), d.ctx.lease.slot().index());
    assert_ne!(d.ctx.lease.slot().index(), d.ep0_ring.lease.slot().index());
    assert_ne!(d.ep0_ring.lease.slot().index(), d.data.lease.slot().index());
}

#[test]
fn share_cpu_access_writes_the_explicit_bitfield() {
    use kore_abi::session::{SHARE_ACCESS_R, SHARE_ACCESS_X, ShareReq};
    use zerocopy::TryFromBytes;

    let mut b = SessionBuilder::new();
    b.share_cpu_access(
        SlotRef::from_wire(3),
        Rights::ACCESS,
        0x7000_0000,
        4096,
        SHARE_ACCESS_R | SHARE_ACCESS_X,
        SlotRef::from_wire(9),
    )
    .unwrap();

    let mut out = [0u8; DESCRIPTOR_MAX];
    let n = b.finish(&mut out).unwrap();
    let (_, invs, args) = parse(&out[..n]);
    let off = invs[0].arg.get() as usize;
    let (req, _) = ShareReq::try_ref_from_prefix(&args[off..]).expect("ShareReq layout");
    assert_eq!(req.va.get(), 0x7000_0000);
    assert_eq!(req.bytes.get(), 4096);
    assert_eq!(req.write, 0, "legacy byte stays zero");
    assert_eq!(req.access, SHARE_ACCESS_R | SHARE_ACCESS_X);
    assert_eq!(req.reserved, [0u8; 6]);
}

#[test]
fn peer_iova_verbs_encode_both_authority_inputs_and_exact_parent() {
    use kore_abi::cap::ResourceKind;
    use kore_abi::session::{
        DERIVE_KIND_PEER_IOVA, DEVICE_MMU_RIGHTS, PeerDeviceShareReq, REVOKE_KIND_UNSHARE_IOVA,
        UnshareReq,
    };
    use zerocopy::TryFromBytes;

    let bar = SlotRef::from_wire(2);
    let device = SlotRef::from_wire(3);
    let mmu_slot = SlotRef::from_wire(4);
    let window = SlotRef::from_wire(5);
    let mut b = SessionBuilder::new();
    let mmu = b
        .attest_device_mmu(device, mmu_slot)
        .expect("device-MMU attestation stages");
    b.share_peer_device(
        bar,
        Rights::ACCESS | Rights::MODIFY | Rights::REVOKE,
        device,
        mmu.slot(),
        0x20_0000,
        0x3000,
        0x2000,
        PeerDeviceShareReq::ACCESS_READ_WRITE,
        window,
    )
    .expect("peer window stages");
    b.unshare_iova(window, bar)
        .expect("exact peer retirement stages");

    let mut out = [0u8; DESCRIPTOR_MAX];
    let n = b.finish(&mut out).expect("peer descriptor finishes");
    let (header, invs, args) = parse(&out[..n]);
    assert_eq!(header.count.get(), 3);

    assert_eq!(invs[0].op.get(), OpCode::Derive as u16);
    assert_eq!(invs[0].kind, ResourceKind::DeviceMmu as u8);
    assert_eq!(invs[0].requested, DEVICE_MMU_RIGHTS.bits());
    assert_eq!(invs[0].subject.get(), device.index());
    assert_eq!(invs[0].result.get(), mmu_slot.index());
    assert_eq!(invs[0].arg.get(), Invocation::NO_ARG);

    assert_eq!(invs[1].op.get(), OpCode::Derive as u16);
    assert_eq!(invs[1].kind, DERIVE_KIND_PEER_IOVA);
    assert_eq!(invs[1].subject.get(), bar.index());
    assert_eq!(invs[1].result.get(), window.index());
    let peer_off = invs[1].arg.get() as usize;
    let (peer, _) = PeerDeviceShareReq::try_ref_from_prefix(&args[peer_off..])
        .expect("PeerDeviceShareReq layout");
    assert_eq!(peer.device.get(), device.index());
    assert_eq!(peer.sponsor.get(), mmu_slot.index());
    assert_eq!(peer.iova.get(), 0x20_0000);
    assert_eq!(peer.offset.get(), 0x3000);
    assert_eq!(peer.bytes.get(), 0x2000);
    assert_eq!(peer.access, PeerDeviceShareReq::ACCESS_READ_WRITE);
    assert_eq!(peer.reserved, [0; 7]);

    assert_eq!(invs[2].op.get(), OpCode::Revoke as u16);
    assert_eq!(invs[2].kind, REVOKE_KIND_UNSHARE_IOVA);
    assert_eq!(invs[2].subject.get(), window.index());
    assert_eq!(invs[2].result.get(), 0);
    let unshare_off = invs[2].arg.get() as usize;
    let (unshare, _) =
        UnshareReq::try_ref_from_prefix(&args[unshare_off..]).expect("UnshareReq layout");
    assert_eq!(unshare.lease.get(), bar.index());
    assert_eq!(unshare.reserved, [0; 4]);
}

#[test]
fn share_to_names_the_dest_space_and_kind() {
    use kore_abi::session::ShareToReq;
    use zerocopy::TryFromBytes;

    let mut b = SessionBuilder::new();
    b.share_to(
        SlotRef::from_wire(4), // the held lease
        Rights::ACCESS,
        SlotRef::from_wire(6), // the child space
        0x5000_0000,
        8192,
        false,
        kore_abi::session::SHARE_ACCESS_R,
        SlotRef::from_wire(10),
    )
    .unwrap();

    let mut out = [0u8; DESCRIPTOR_MAX];
    let n = b.finish(&mut out).unwrap();
    let (_, invs, args) = parse(&out[..n]);
    assert_eq!(invs[0].op.get(), OpCode::Derive as u16);
    assert_eq!(
        invs[0].kind,
        (kore_abi::cap::ResourceKind::AddrSpace as u16) as u8,
        "the explicit-dest share is kind-disambiguated on AddrSpace"
    );
    let off = invs[0].arg.get() as usize;
    let (req, _) = ShareToReq::try_ref_from_prefix(&args[off..]).expect("ShareToReq layout");
    assert_eq!(req.dest_space.get(), 6);
    assert_eq!(req.va.get(), 0x5000_0000);
    assert_eq!(req.bytes.get(), 8192);
    assert_eq!(req.access, kore_abi::session::SHARE_ACCESS_R);
    assert_eq!(req.consume, 0, "the plain form never consumes its name");
    assert_eq!(req.reserved, [0u8; 1]);
}

/// The consuming share differs from the plain one in **exactly** the
/// consume byte — same op, same kind, same geometry, same wire size.  A
/// spawner picks it per share, so a drift that widened the payload (or
/// moved `consume` onto a byte the kernel reads as something else) would
/// silently either stop freeing slots or corrupt an access decode.
#[test]
fn share_to_consuming_sets_only_the_consume_byte() {
    use kore_abi::session::ShareToReq;
    use zerocopy::TryFromBytes;

    let encode = |consuming: bool| {
        let mut b = SessionBuilder::new();
        let args = (
            SlotRef::from_wire(4),
            Rights::ACCESS,
            SlotRef::from_wire(6),
            0x5000_0000u64,
            8192u64,
            false,
            kore_abi::session::SHARE_ACCESS_R,
            SlotRef::from_wire(10),
        );
        if consuming {
            b.share_to_consuming(
                args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7,
            )
        } else {
            b.share_to(
                args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7,
            )
        }
        .unwrap();
        let mut out = [0u8; DESCRIPTOR_MAX];
        let n = b.finish(&mut out).unwrap();
        out[..n].to_vec()
    };

    let plain = encode(false);
    let consuming = encode(true);
    assert_eq!(
        plain.len(),
        consuming.len(),
        "consuming must not change the descriptor size"
    );
    let differing: alloc::vec::Vec<usize> = plain
        .iter()
        .zip(consuming.iter())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        differing.len(),
        1,
        "exactly one byte differs, got {differing:?}"
    );

    let (_, invs, args) = parse(&consuming);
    assert_eq!(invs[0].op.get(), OpCode::Derive as u16);
    assert_eq!(
        invs[0].kind,
        (kore_abi::cap::ResourceKind::AddrSpace as u16) as u8,
        "consuming rides the same kind disambiguator"
    );
    let off = invs[0].arg.get() as usize;
    let (req, _) = ShareToReq::try_ref_from_prefix(&args[off..]).expect("ShareToReq layout");
    assert_eq!(req.consume, 1);
    assert_eq!(req.dest_space.get(), 6);
    assert_eq!(req.va.get(), 0x5000_0000);
    assert_eq!(req.bytes.get(), 8192);
    assert_eq!(req.access, kore_abi::session::SHARE_ACCESS_R);
    assert_eq!(req.reserved, [0u8; 1], "the spare byte stays reserved");
}

#[test]
fn delegate_and_unmap_stage_the_83_verbs() {
    use kore_abi::session::{DelegateReq, REVOKE_KIND_UNMAP};
    use zerocopy::TryFromBytes;

    let mut b = SessionBuilder::new();
    let moved = b
        .delegate(
            SlotRef::from_wire(7),
            SlotRef::from_wire(2),
            0x4000_0000,
            4096,
            true,
            0,
            SlotRef::from_wire(11),
        )
        .unwrap();
    let extent = b.unmap(moved, SlotRef::from_wire(12)).unwrap();
    assert_eq!(extent.slot(), SlotRef::from_wire(12));

    let mut out = [0u8; DESCRIPTOR_MAX];
    let n = b.finish(&mut out).unwrap();
    let (hdr, invs, args) = parse(&out[..n]);
    assert_eq!(hdr.count.get(), 2);

    // The delegate: kind = Region, empty rights, DelegateReq payload.
    assert_eq!(invs[0].op.get(), OpCode::Derive as u16);
    assert_eq!(
        invs[0].kind,
        (kore_abi::cap::ResourceKind::Region as u16) as u8
    );
    assert_eq!(invs[0].requested, 0, "a delegate carries no rights request");
    let off = invs[0].arg.get() as usize;
    let (req, _) = DelegateReq::try_ref_from_prefix(&args[off..]).expect("DelegateReq layout");
    assert_eq!(req.dest_space.get(), 2);
    assert_eq!(req.va.get(), 0x4000_0000);
    assert_eq!(req.bytes.get(), 4096);
    assert_eq!(req.write, 1);
    assert_eq!(req.access, 0);

    // The unmap: a kind-flavored revoke naming its result slot (SSA:
    // chains on the delegate's staged result), no payload.
    assert_eq!(invs[1].op.get(), OpCode::Revoke as u16);
    assert_eq!(invs[1].kind, REVOKE_KIND_UNMAP);
    assert_eq!(invs[1].subject.get(), 11, "unmaps the delegated region");
    assert_eq!(invs[1].result.get(), 12, "the extent's landing slot");
    assert_eq!(invs[1].arg.get(), Invocation::NO_ARG);
}

#[test]
fn revoke_batch_is_one_crossing() {
    let mut b = SessionBuilder::new();
    let caps: [SlotRef; 8] = core::array::from_fn(|i| SlotRef::from_wire(50 + i as u32));
    flows::revoke_batch(&mut b, &caps).unwrap();
    let mut out = [0u8; DESCRIPTOR_MAX];
    let n = b.finish(&mut out).unwrap();
    let (hdr, invs, _) = parse(&out[..n]);
    assert_eq!(hdr.count.get(), 8);
    assert!(invs.iter().all(|i| i.op.get() == OpCode::Revoke as u16));
}

/// **U1 — the v3 `FoundReq` delivery map on the wire**: `found_deliver`
/// emits the 248-byte req with the live entries at offset 32 and a
/// reserved-zero tail, read back through the same zerocopy face the
/// kernel decode uses; a map past `MAX_FOUND_DELIVERIES` refuses at
/// staging (`TooManyDeliveries`), and the delivery-less `found` emits
/// count 0.
#[test]
fn found_deliver_emits_the_v3_delivery_map() {
    use kore_abi::session::{FoundReq, MAX_FOUND_DELIVERIES};
    use kore_abi::task::SpawnFlags;
    use zerocopy::TryFromBytes;

    let mut b = SessionBuilder::new();
    let chan = b
        .derive_channel(
            SlotRef::from_wire(2),
            Rights::ACCESS | Rights::DERIVE | Rights::REVOKE,
            SlotRef::from_wire(10),
        )
        .unwrap();
    let recv = b
        .derive_chan_recv(
            chan,
            Rights::ACCESS | Rights::REVOKE,
            SlotRef::from_wire(11),
        )
        .unwrap();
    let space = b
        .derive_fresh_space(
            SlotRef::from_wire(1),
            Rights::ACCESS | Rights::MODIFY | Rights::DERIVE | Rights::REVOKE,
            SlotRef::from_wire(12),
        )
        .unwrap();
    b.found_deliver(
        space,
        0x40_0000,
        0x7fff_0000,
        None,
        SpawnFlags::empty(),
        8,
        SlotRef::from_wire(9),
        SlotRef::from_wire(13),
        &[Deliver::new(recv, 3)],
    )
    .unwrap();

    let mut out = [0u8; DESCRIPTOR_MAX];
    let n = b.finish(&mut out).unwrap();
    let (_, invs, args) = parse(&out[..n]);
    let (req, _) = FoundReq::try_ref_from_prefix(&args[invs[3].arg.get() as usize..])
        .expect("a v3 FoundReq decodes");
    assert_eq!(req.arena_slots.get(), 8);
    assert_eq!(
        req.authority.get(),
        9,
        "the founder's exec-authority slot rides the FoundReq"
    );
    assert_eq!(req.deliver_count.get(), 1);
    assert_eq!(req.deliver[0].from.get(), 11, "the staged recv's SSA slot");
    assert_eq!(req.deliver[0].to.get(), 3, "the child-arena destination");
    assert!(
        req.deliver[1..]
            .iter()
            .all(|e| e.from.get() == 0 && e.to.get() == 0),
        "the tail stays reserved-zero"
    );

    // The delivery-less `found` emits count 0 (v2 semantics on the v3 form).
    let mut b2 = SessionBuilder::new();
    let space = b2
        .derive_fresh_space(
            SlotRef::from_wire(1),
            Rights::ACCESS | Rights::MODIFY | Rights::DERIVE | Rights::REVOKE,
            SlotRef::from_wire(12),
        )
        .unwrap();
    b2.found(
        space,
        0,
        0,
        None,
        SpawnFlags::empty(),
        8,
        SlotRef::from_wire(9),
        SlotRef::from_wire(13),
    )
    .unwrap();
    let n = b2.finish(&mut out).unwrap();
    let (_, invs, args) = parse(&out[..n]);
    let (req, _) = FoundReq::try_ref_from_prefix(&args[invs[1].arg.get() as usize..]).unwrap();
    assert_eq!(req.deliver_count.get(), 0);

    // An over-bound map refuses at staging, before any bytes exist.
    let mut b3 = SessionBuilder::new();
    let space = b3
        .derive_fresh_space(
            SlotRef::from_wire(1),
            Rights::ACCESS | Rights::MODIFY | Rights::DERIVE | Rights::REVOKE,
            SlotRef::from_wire(12),
        )
        .unwrap();
    let too_many: std::vec::Vec<Deliver> = (0..MAX_FOUND_DELIVERIES + 1)
        .map(|k| Deliver::new(SlotRef::from_wire(k as u32), k as u16))
        .collect();
    assert_eq!(
        b3.found_deliver(
            space,
            0,
            0,
            None,
            SpawnFlags::empty(),
            32,
            SlotRef::from_wire(9),
            SlotRef::from_wire(13),
            &too_many,
        )
        .unwrap_err(),
        BuildError::TooManyDeliveries
    );
}

/// **U1 — the typed channel-mint verbs** (`res/USERLAND.md` §5.2): the
/// three-derive mint (`ChannelSpace → Channel → ChanSend + ChanRecv`)
/// stages byte-for-byte the kernel boottest's `channel_mint_descriptor`
/// shape — kind-routed, arg-free, SSA-chained on the staged channel.
#[test]
fn channel_mint_verbs_stage_the_three_derive_shape() {
    use kore_abi::cap::ResourceKind;

    let mut b = SessionBuilder::new();
    let chan = b
        .derive_channel(
            SlotRef::from_wire(0),
            Rights::ACCESS | Rights::DERIVE | Rights::REVOKE,
            SlotRef::from_wire(1),
        )
        .unwrap();
    b.derive_chan_send(chan, Rights::ACCESS | Rights::DERIVE, SlotRef::from_wire(2))
        .unwrap();
    b.derive_chan_recv(chan, Rights::ACCESS | Rights::REVOKE, SlotRef::from_wire(3))
        .unwrap();

    let mut out = [0u8; DESCRIPTOR_MAX];
    let n = b.finish(&mut out).unwrap();
    let (hdr, invs, _) = parse(&out[..n]);
    assert_eq!(
        hdr.count.get(),
        3,
        "a full pair is three derives, one crossing"
    );

    assert_eq!(invs[0].op.get(), OpCode::Derive as u16);
    assert_eq!(
        invs[0].kind,
        ResourceKind::Channel as u16 as u8,
        "the mint is kind-routed"
    );
    assert_eq!(invs[0].subject.get(), 0, "off the held ChannelSpace record");
    assert_eq!(invs[0].result.get(), 1);
    assert_eq!(
        invs[0].arg.get(),
        Invocation::NO_ARG,
        "a channel is fully named by its subject"
    );

    assert_eq!(invs[1].kind, ResourceKind::ChanSend as u16 as u8);
    assert_eq!(
        invs[1].subject.get(),
        1,
        "carved off the staged channel (SSA)"
    );
    assert_eq!(invs[1].result.get(), 2);
    assert_eq!(invs[1].arg.get(), Invocation::NO_ARG);

    assert_eq!(invs[2].kind, ResourceKind::ChanRecv as u16 as u8);
    assert_eq!(invs[2].subject.get(), 1);
    assert_eq!(invs[2].result.get(), 3);
    assert_eq!(invs[2].arg.get(), Invocation::NO_ARG);
}

/// A launcher endowment is a self-kind child of `ChannelSpace`, not the
/// primordial allocator and not an eagerly minted `Channel`.
#[test]
fn channel_space_grant_stages_the_self_kind_derive() {
    use kore_abi::cap::ResourceKind;

    let mut b = SessionBuilder::new();
    b.derive_channel_space_grant(
        SlotRef::from_wire(4),
        Rights::ACCESS | Rights::DERIVE | Rights::REVOKE,
        SlotRef::from_wire(5),
    )
    .unwrap();

    let mut out = [0u8; DESCRIPTOR_MAX];
    let n = b.finish(&mut out).unwrap();
    let (hdr, invs, _) = parse(&out[..n]);
    assert_eq!(hdr.count.get(), 1);
    assert_eq!(invs[0].op.get(), OpCode::Derive as u16);
    assert_eq!(invs[0].kind, ResourceKind::ChannelSpace as u16 as u8);
    assert_eq!(invs[0].subject.get(), 4);
    assert_eq!(invs[0].result.get(), 5);
    assert_eq!(invs[0].arg.get(), Invocation::NO_ARG);
}

/// The witness decodes exactly the kernel's two verdicts and refuses a
/// vocabulary it does not know — a newer kernel's third verdict must never
/// read as a proof on an older runtime.
#[test]
fn quiesced_witness_decodes_only_known_verdicts() {
    use crate::cap::{FabricVerdict, Quiesced};
    use kore_abi::session::{
        QUIESCE_FLAG_ABORT_BOUND, QUIESCE_FLAG_IRQ_MASKED, QUIESCE_VERDICT_CONTAINED,
        QUIESCE_VERDICT_PROVEN, QuiesceReport,
    };

    let proven = Quiesced::minted(QuiesceReport {
        rid: 0x2a,
        verdict: QUIESCE_VERDICT_PROVEN,
        flags: QUIESCE_FLAG_IRQ_MASKED,
    })
    .expect("proven decodes");
    assert_eq!(proven.rid(), 0x2a);
    assert_eq!(proven.verdict(), FabricVerdict::Proven);
    assert!(!proven.abort_bound(), "no IOMMU behind this function");

    let contained = Quiesced::minted(QuiesceReport {
        rid: 7,
        verdict: QUIESCE_VERDICT_CONTAINED,
        flags: QUIESCE_FLAG_IRQ_MASKED | QUIESCE_FLAG_ABORT_BOUND,
    })
    .expect("contained decodes");
    assert_eq!(contained.verdict(), FabricVerdict::Contained);
    assert!(contained.abort_bound());

    assert_eq!(
        Quiesced::minted(QuiesceReport {
            rid: 7,
            verdict: 2,
            flags: 0,
        }),
        None,
        "an unknown verdict is not a witness"
    );
}

#[test]
fn selected_irq_is_encoded_after_deadline_and_preserves_argument_offsets() {
    use kore_abi::session::{IrqParkReq, NoticeReq, ParkReq, SleepReq};
    use zerocopy::TryFromBytes;
    for slot in [7, 19] {
        let mut b = SessionBuilder::new();
        b.notice(SlotRef::from_wire(2), 0xab, 0xcd).unwrap();
        // Method order must not change the pinned park-block order.
        b.wait_irq_on(SlotRef::from_wire(slot));
        b.park_standing(11, 4, 23).wait_sleep(9_000);
        let mut bytes = [0u8; DESCRIPTOR_MAX];
        let n = b.finish(&mut bytes).unwrap();
        let (header, invs, args) = parse(&bytes[..n]);
        assert_eq!(
            header.flags.get(),
            (SessionFlags::WAIT_IRQ
                | SessionFlags::WAIT_IRQ_SELECTED
                | SessionFlags::WAIT_DELIVERY
                | SessionFlags::WAIT_SLEEP)
                .bits()
        );
        let (_, tail) = ParkReq::try_ref_from_prefix(args).unwrap();
        let (sleep, tail) = SleepReq::try_ref_from_prefix(tail).unwrap();
        assert_eq!(sleep.deadline_ns.get(), 9_000);
        let (irq, _) = IrqParkReq::try_ref_from_prefix(tail).unwrap();
        assert_eq!(irq.slot.get(), slot);
        assert_eq!(irq.reserved.get(), 0);
        assert_eq!(invs[0].arg.get(), 32);
        let (notice, _) = NoticeReq::try_ref_from_prefix(&args[32..]).unwrap();
        assert_eq!((notice.cookie.get(), notice.payload.get()), (0xab, 0xcd));
    }
}

#[test]
fn selected_irq_only_and_deadline_waits_fit_the_inline_carrier() {
    for deadline in [None, Some(42)] {
        let mut b = SessionBuilder::new();
        b.wait_irq_on(SlotRef::from_wire(19));
        if let Some(deadline) = deadline {
            b.wait_sleep(deadline);
        }
        let mut bytes = [0u8; 48];
        let n = b.finish(&mut bytes).unwrap();
        assert_eq!(n, if deadline.is_some() { 24 } else { 16 });
        let (_, invs, args) = parse(&bytes[..n]);
        assert!(invs.is_empty());
        let off = if deadline.is_some() { 8 } else { 0 };
        assert_eq!(
            u32::from_le_bytes(args[off..off + 4].try_into().unwrap()),
            19
        );
    }
}

#[test]
fn virtio_pci_layout_queries_are_inert_and_shape_tight() {
    use kore_abi::session::{
        DERIVE_KIND_VIRTIO_PCI_LAYOUT, VIRTIO_PCI_LAYOUT_QUERY_COUNT,
        VIRTIO_PCI_LAYOUT_QUERY_RECORD, VirtioPciLayoutReq,
    };
    use zerocopy::TryFromBytes;

    let mut builder = SessionBuilder::new();
    builder
        .read_virtio_pci_cap_count(SlotRef::from_wire(4), SlotRef::from_wire(8))
        .expect("count query stages");
    builder
        .read_virtio_pci_cap(SlotRef::from_wire(4), 3, SlotRef::from_wire(9))
        .expect("record query stages");

    let mut out = [0u8; DESCRIPTOR_MAX];
    let len = builder.finish(&mut out).expect("queries finish");
    let (_, invocations, args) = parse(&out[..len]);
    assert_eq!(invocations.len(), 2);
    for invocation in invocations {
        assert_eq!(invocation.kind, DERIVE_KIND_VIRTIO_PCI_LAYOUT);
        assert_eq!(invocation.requested, 0);
    }
    let (count, tail) = VirtioPciLayoutReq::try_ref_from_prefix(args).expect("count request");
    let (record, _) = VirtioPciLayoutReq::try_ref_from_prefix(tail).expect("record request");
    assert_eq!(
        (count.index, count.query),
        (0, VIRTIO_PCI_LAYOUT_QUERY_COUNT)
    );
    assert_eq!(count.reserved, [0; 6]);
    assert_eq!(
        (record.index, record.query),
        (3, VIRTIO_PCI_LAYOUT_QUERY_RECORD)
    );
    assert_eq!(record.reserved, [0; 6]);
}

#[test]
fn typed_exit_serializes_every_status_bit_and_raw_exit_cannot_bypass_it() {
    use crate::ExitCode;
    use kore_abi::session::{DERIVE_KIND_TASK_EXIT, ExitReq};
    use zerocopy::TryFromBytes;
    let slot = SlotRef::from_wire(2);
    for code in [
        ExitCode::OK,
        ExitCode::APPLICATION_0,
        ExitCode::APPLICATION_1,
        ExitCode::APPLICATION_15,
        ExitCode::LINK_FAILED,
        ExitCode::PANIC,
    ] {
        let mut builder = SessionBuilder::new();
        builder.exit(slot, code).unwrap();
        let mut out = [0u8; DESCRIPTOR_MAX];
        let len = builder.finish(&mut out).unwrap();
        let (_, invs, args) = parse(&out[..len]);
        assert_eq!(invs.len(), 1);
        assert_eq!(invs[0].kind, DERIVE_KIND_TASK_EXIT);
        assert_eq!(invs[0].subject.get(), 2);
        let (req, rest) = ExitReq::try_ref_from_prefix(args).unwrap();
        assert!(rest.is_empty());
        assert_eq!(req.status.get(), code.get());
        assert_eq!(req.reserved, [0; 4]);
    }
    let mut builder = SessionBuilder::new();
    assert_eq!(
        builder.derive_raw(
            slot,
            Rights::empty(),
            DERIVE_KIND_TASK_EXIT,
            slot,
            Some(&[0; 8])
        ),
        Err(BuildError::UntypedExit)
    );
    builder.exit(slot, ExitCode::APPLICATION_1).unwrap();
    let mut out = [0u8; DESCRIPTOR_MAX];
    let len = builder.finish(&mut out).unwrap();
    let (_, invs, args) = parse(&out[..len]);
    assert_eq!(
        invs.len(),
        1,
        "rejected raw exit must not stage an invocation"
    );
    assert_eq!(args.len(), ExitReq::WIRE, "nor consume argument storage");
}

#[test]
fn process_drain_is_a_standalone_authenticated_self_operation() {
    use kore_abi::session::DERIVE_KIND_PROCESS_DRAIN;
    let mut b = SessionBuilder::new();
    b.drain_process(SlotRef::from_wire(3), SlotRef::from_wire(5))
        .unwrap();
    let mut bytes = [0u8; DESCRIPTOR_MAX];
    let length = b.finish(&mut bytes).unwrap();
    let (header, invocations, args) = parse(&bytes[..length]);
    assert_eq!(header.flags.get(), 0);
    assert_eq!(invocations.len(), 1);
    assert_eq!(invocations[0].op.get(), OpCode::Derive as u16);
    assert_eq!(invocations[0].kind, DERIVE_KIND_PROCESS_DRAIN);
    assert_eq!(invocations[0].subject.get(), 3);
    assert_eq!(invocations[0].result.get(), 5);
    assert_eq!(invocations[0].requested, 0);
    assert_eq!(invocations[0].arg.get(), Invocation::NO_ARG);
    assert!(args.is_empty());
    assert!(b.start(SlotRef::from_wire(3)).is_err());
    let mut b = SessionBuilder::new();
    b.start(SlotRef::from_wire(3)).unwrap();
    assert!(
        b.drain_process(SlotRef::from_wire(3), SlotRef::from_wire(5))
            .is_err()
    );
    assert!(kore_abi::session::derive_kind_known(
        DERIVE_KIND_PROCESS_DRAIN
    ));
}

#[test]
fn crash_read_and_panic_are_standalone_and_preserve_their_payloads() {
    use kore_abi::session::{DERIVE_KIND_CRASH_READ, DERIVE_KIND_PANIC_REPORT};
    let task = SlotRef::from_wire(3);
    let report = SlotRef::from_wire(5);
    let mut b = SessionBuilder::new();
    b.read_crash(task, report, 63).unwrap();
    assert!(b.start(task).is_err());
    let mut bytes = [0; DESCRIPTOR_MAX];
    let length = b.finish(&mut bytes).unwrap();
    let (_, invocations, args) = parse(&bytes[..length]);
    assert_eq!(invocations.len(), 1);
    assert_eq!(invocations[0].kind, DERIVE_KIND_CRASH_READ);
    assert_eq!(invocations[0].subject.get(), 3);
    assert_eq!(invocations[0].result.get(), 5);
    assert_eq!(args, &[63, 0, 0, 0, 0, 0, 0, 0]);

    let mut b = SessionBuilder::new();
    b.report_panic(task, "panic \u{20ac}").unwrap();
    assert!(b.read_crash(task, report, 0).is_err());
    let length = b.finish(&mut bytes).unwrap();
    let (_, invocations, args) = parse(&bytes[..length]);
    assert_eq!(invocations.len(), 1);
    assert_eq!(invocations[0].kind, DERIVE_KIND_PANIC_REPORT);
    assert_eq!(invocations[0].subject.get(), 3);
    assert_eq!(u32::from_le_bytes(args[..4].try_into().unwrap()), 9);
    assert_eq!(&args[4..8], &[0; 4]);
    assert_eq!(&args[8..], "panic \u{20ac}".as_bytes());
    let mut b = SessionBuilder::new();
    b.start(task).unwrap();
    assert!(b.report_panic(task, "too late").is_err());
}

#[test]
fn crash_binding_is_standalone_and_uses_held_memory_coordinates() {
    let task = SlotRef::from_wire(3);
    let lease = SlotRef::from_wire(7);
    let mut b = SessionBuilder::new();
    b.bind_crash(task, lease, 8192, 4096).unwrap();
    assert!(b.start(task).is_err());
    let mut bytes = [0; DESCRIPTOR_MAX];
    let length = b.finish(&mut bytes).unwrap();
    let (_, invocations, args) = parse(&bytes[..length]);
    assert_eq!(invocations.len(), 1);
    assert_eq!(
        invocations[0].kind,
        kore_abi::session::DERIVE_KIND_CRASH_BIND
    );
    assert_eq!(args.len(), 24);
    assert_eq!(u32::from_le_bytes(args[..4].try_into().unwrap()), 7);
    assert_eq!(&args[4..8], &[0; 4]);
    assert_eq!(u64::from_le_bytes(args[8..16].try_into().unwrap()), 8192);
    assert_eq!(u64::from_le_bytes(args[16..24].try_into().unwrap()), 4096);
}

#[test]
fn task_name_is_bounded_copied_and_standalone() {
    let mut b = SessionBuilder::new();
    let name = kore_abi::task::TaskName::new("hello/worker-2").unwrap();
    b.name_task(SlotRef::from_wire(7), name).unwrap();
    assert_eq!(
        b.name_task(SlotRef::from_wire(8), name),
        Err(BuildError::EffectMustStandAlone)
    );
    let mut bytes = [0; DESCRIPTOR_MAX];
    let n = b.finish(&mut bytes).unwrap();
    let (_, invs, args) = parse(&bytes[..n]);
    assert_eq!(invs.len(), 1);
    assert_eq!(invs[0].kind, kore_abi::session::DERIVE_KIND_TASK_NAME);
    let payload = &args[invs[0].arg.get() as usize..];
    assert_eq!(payload[0] as usize, name.as_str().len());
    assert_eq!(
        &payload[1..1 + payload[0] as usize],
        name.as_str().as_bytes()
    );
}

#[test]
fn task_identity_query_is_standalone_and_has_no_arguments() {
    let mut b = SessionBuilder::new();
    let task = SlotRef::from_wire(7);
    let result = SlotRef::from_wire(8);
    b.task_identity(task, result).unwrap();
    assert_eq!(
        b.task_identity(task, result),
        Err(BuildError::EffectMustStandAlone)
    );
    let mut bytes = [0; DESCRIPTOR_MAX];
    let n = b.finish(&mut bytes).unwrap();
    let (_, invs, args) = parse(&bytes[..n]);
    assert_eq!(invs.len(), 1);
    assert_eq!(invs[0].kind, kore_abi::session::DERIVE_KIND_TASK_IDENTITY);
    assert_eq!(invs[0].arg.get(), u32::MAX);
    assert!(args.is_empty());
}
