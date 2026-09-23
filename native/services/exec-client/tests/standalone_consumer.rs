//! A consumer of the exec launch surface that links **only** this crate.
//!
//! Kore #996 and libkore #18 both ask for "one minimal non-vkore guest consumer
//! of the shared launch surface, so reuse is demonstrated rather than asserted".
//! This is that consumer, in the one form that cannot be faked: an integration
//! test links the crate under test and its declared dev-dependencies and
//! nothing else, so if any of the surface below still needed `kore-svc-exec`,
//! this file would not compile.
//!
//! What it exercises is the whole caller-side arc short of possessing a founded
//! face: name an executable in a local namespace, resolve it back, build the
//! founding record, put it on the wire, and read the service's answers — refusal
//! included. The affine `ExecutableCapability` is stood in for by a plain token,
//! which is exactly what `CapabilityNamespace<C, N>` is generic over and why it
//! is host-testable at all.

use kore_abi::image::ImageName;
use kore_abi::svc::{
    MAX_SPAWN_DATA, MAX_SPAWN_SEGMENTS, MAX_SPAWN_WINDOWS, MAX_SPAWN_WIRING, SpawnData,
    SpawnSegment, SpawnWindow, SpawnWire,
};
use zerocopy::FromZeros;
use kore_exec_client::{
    CapabilityNamespace, ExecProtocol, ExecutableContext, ExecutableMetadata, SpawnSpec,
    TaskRefusal, TaskRequest, TaskResponse,
};
use kore_rt::svc::SvcProtocol;

/// Stands in for the affine `ExecutableCapability` a real caller would hold.
#[derive(Debug, PartialEq, Eq)]
struct HeldExecutable(u32);

fn metadata(name: &str) -> ExecutableMetadata {
    ExecutableMetadata {
        local_name: ImageName::new(name).unwrap(),
        display_name: ImageName::new(name).unwrap(),
        usage: b"",
        summary: b"",
        context: ExecutableContext::None,
    }
}

fn founding_record() -> SpawnSpec {
    SpawnSpec {
        // The decoder requires both of these together: a task nobody holds a
        // handle to is unreapable, and the startup linker is not optional.
        return_handle: true,
        delegable_handle: false,
        return_shared_wake: false,
        linked: true,
        entry: 0x1000,
        stack: 0x2000,
        arena_slots: 8,
        boot_va: 0,
        // Distinct convey entries on purpose: the codec refuses a record that
        // names one slot as both the boot page's writer and its share source.
        boot_write_idx: 0,
        boot_share_idx: 1,
        seg_count: 0,
        wiring_count: 0,
        window_count: 0,
        data_count: 0,
        // Every count above is zero, so the tables are inert. They are frozen
        // wire records without a `Default`, which is the point: a caller fills
        // exactly the rows it declares and the service validates the counts.
        segments: [SpawnSegment::new_zeroed(); MAX_SPAWN_SEGMENTS],
        wiring: [SpawnWire::new_zeroed(); MAX_SPAWN_WIRING],
        windows: [SpawnWindow::new_zeroed(); MAX_SPAWN_WINDOWS],
        data: [SpawnData::new_zeroed(); MAX_SPAWN_DATA],
    }
}

#[test]
fn a_consumer_names_resolves_and_retires_an_executable_without_the_service() {
    let mut namespace = CapabilityNamespace::<HeldExecutable, 4>::new();
    namespace.bind(metadata("read"), HeldExecutable(7)).unwrap();

    // The local name is the consumer's own; authority is the bound capability.
    assert_eq!(
        namespace.resolve(ImageName::new("read").unwrap()),
        Some(&HeldExecutable(7))
    );
    assert!(namespace.resolve(ImageName::new("write").unwrap()).is_none());

    // Rebinding a live name is refused and gives the caller its capability back
    // rather than dropping an affine owner on the floor.
    let (_, returned) = namespace
        .bind(metadata("read"), HeldExecutable(9))
        .unwrap_err();
    assert_eq!(returned, HeldExecutable(9));

    // Unbinding removes local reachability and returns the owner. It is not
    // proof of kernel revocation, which is why it hands the capability back.
    assert_eq!(
        namespace.unbind(ImageName::new("read").unwrap()),
        Some(HeldExecutable(7))
    );
    assert!(namespace.resolve(ImageName::new("read").unwrap()).is_none());
}

#[test]
fn a_consumer_encodes_a_founding_record_and_reads_it_back() {
    let spec = founding_record();
    let wire = ExecProtocol::encode_req(&TaskRequest::Invoke(spec));
    let decoded = ExecProtocol::decode_req(&wire).expect("the service's own decoder accepts it");
    match decoded {
        TaskRequest::Invoke(round_tripped) => {
            assert_eq!(round_tripped.entry, spec.entry);
            assert_eq!(round_tripped.stack, spec.stack);
            assert_eq!(round_tripped.arena_slots, spec.arena_slots);
            assert!(round_tripped.return_handle);
            assert!(!round_tripped.delegable_handle);
        }
        other => panic!("request changed shape on the wire: {other:?}"),
    }
}

#[test]
fn the_codec_refuses_a_malformed_founding_before_it_reaches_the_service() {
    // Same record, one field wrong: the boot page's writer and its share source
    // must be different convey entries. A consumer finds that out from the
    // protocol, not from a round trip to the spawn engine.
    let mut spec = founding_record();
    spec.boot_share_idx = spec.boot_write_idx;
    let wire = ExecProtocol::encode_req(&TaskRequest::Invoke(spec));
    assert!(ExecProtocol::decode_req(&wire).is_none());

    // And a founding that asks for no parent handle is unreapable, so it is
    // refused for shape rather than accepted and leaked.
    let mut spec = founding_record();
    spec.return_handle = false;
    let wire = ExecProtocol::encode_req(&TaskRequest::Invoke(spec));
    assert!(ExecProtocol::decode_req(&wire).is_none());
}

#[test]
fn a_consumer_reads_both_a_founding_and_a_refusal() {
    let request = ExecProtocol::encode_req(&TaskRequest::Invoke(founding_record()));

    // A founding carries its authority out of band; the reply is data plus a
    // grant count, and the protocol is what says whether the two agree.
    let refusal = TaskResponse::Refused(TaskRefusal::Contended);
    let encoded = ExecProtocol::encode_rep(&refusal, 0);
    let (decoded, grants) = ExecProtocol::decode_rep(&encoded).expect("refusals decode");
    assert_eq!(decoded, refusal);
    assert_eq!(grants, 0);
    assert!(ExecProtocol::reply_matches(&request, &decoded, 0));

    // A refusal that claims to carry authority is not a reply to this request.
    assert!(!ExecProtocol::reply_matches(&request, &decoded, 1));
}

#[test]
fn the_grant_ceiling_is_the_protocols_to_state_not_the_servers() {
    // A caller pre-pins this many grant channels before it may speak at all, so
    // it has to be able to read the number without linking the service.
    assert_eq!(
        <ExecProtocol as SvcProtocol>::REPLY_GRANTS,
        kore_exec_client::EXEC_REPLY_GRANTS
    );
    assert!(kore_exec_client::EXEC_REPLY_GRANTS > 0);
}
