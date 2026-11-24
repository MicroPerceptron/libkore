//! The frozen semantic conformance suite, running in-tree — the accel
//! oracle.
//!
//! The suite exercises any [`virtio_accel_core::Accelerator`] — exactly
//! the seam every kore accel backend implements
//! (`kore_svc_accel::contract`), so keeping it green here means: the
//! published contracts kore builds against are alive and self-
//! consistent, and each hardware backend (ivpu next) has its acceptance
//! harness waiting.  The reference targets cover both the mock's in-place xor
//! and its disjoint input/output copy program, including v0.3.2's multi-binding
//! fixture contract.

use virtio_accel_conformance::{
    BindingFixture, CaseStatus, ConformanceHooks, ProgramFixture, ResourceCounts,
    SubmissionPathDiagnostics, TargetDescription, run,
};
use virtio_accel_core::{AccessMode, BackendError, MemoryDomain};
use virtio_accel_mock::fault::{FaultAccelerator, FaultEvent, FaultScript, ResourceState};
use virtio_accel_mock::{MockAccelerator, MockEvent, reference};

/// The reference execution target: the xor-0x5a program over one
/// read-write binding in provider-shared memory (upstream's own
/// fixture, restated).
fn xor_target() -> TargetDescription {
    let initial = vec![0x00, 0x11, 0x7f, 0x80, 0xa5, 0xff, 0x3c, 0xc3];
    let expected = initial.iter().map(|byte| byte ^ 0x5a).collect::<Vec<_>>();
    TargetDescription::new(
        ProgramFixture::new(
            reference::ARTIFACT_FORMAT,
            reference::TARGET_IDENTITY,
            reference::ReferenceArtifact::xor(7, 0x5a).as_bytes(),
            reference::RESIDENT_BYTES,
        )
        .unwrap(),
        BindingFixture::new(
            7,
            AccessMode::ReadWrite,
            MemoryDomain::Shared,
            16,
            initial,
            expected,
        )
        .unwrap(),
    )
}

/// A true two-binding target: immutable slot 2 is copied into distinct writable
/// slot 7.  `with_bindings` and `read_only` are the v0.3.2 integration surface
/// that proves disjoint program inputs and outputs without a hidden bounce.
fn copy_target() -> TargetDescription {
    let input = vec![0x00, 0x11, 0x7f, 0x80, 0xa5, 0xff, 0x3c, 0xc3];
    let output = vec![0; input.len()];
    let artifact = reference::ReferenceArtifact::copy(2, 7).unwrap();
    TargetDescription::with_bindings(
        ProgramFixture::new(
            reference::ARTIFACT_FORMAT,
            reference::TARGET_IDENTITY,
            artifact.as_bytes(),
            reference::RESIDENT_BYTES,
        )
        .unwrap(),
        vec![
            BindingFixture::read_only(2, MemoryDomain::Shared, 16, input.clone()).unwrap(),
            BindingFixture::new(
                7,
                AccessMode::Write,
                MemoryDomain::Shared,
                16,
                output,
                input,
            )
            .unwrap(),
        ],
    )
    .unwrap()
}

struct MockHooks;

impl ConformanceHooks<MockAccelerator> for MockHooks {
    fn complete_event(
        &self,
        backend: &MockAccelerator,
        event: &MockEvent,
    ) -> Result<(), BackendError> {
        backend.complete(event)
    }

    fn submission_path_diagnostics(
        &self,
        backend: &MockAccelerator,
    ) -> Option<SubmissionPathDiagnostics> {
        Some(SubmissionPathDiagnostics {
            direct_bindings: backend.direct_binding_admissions(),
            ..SubmissionPathDiagnostics::default()
        })
    }
}

struct TrackedHooks;

impl ConformanceHooks<FaultAccelerator<MockAccelerator>> for TrackedHooks {
    fn complete_event(
        &self,
        backend: &FaultAccelerator<MockAccelerator>,
        event: &FaultEvent<MockEvent>,
    ) -> Result<(), BackendError> {
        backend.inner().complete(event.inner())
    }

    fn resource_counts(
        &self,
        backend: &FaultAccelerator<MockAccelerator>,
    ) -> Option<ResourceCounts> {
        let snapshot = backend.control().snapshot();
        let live = snapshot.resources_in(ResourceState::Live);
        let unknown = snapshot.resources_in(ResourceState::Indeterminate);
        Some(ResourceCounts {
            contexts: u64::from(live.contexts + unknown.contexts),
            buffers: u64::from(live.buffers + unknown.buffers),
            programs: u64::from(live.programs + unknown.programs),
            queues: u64::from(live.queues + unknown.queues),
            events: u64::from(live.events + unknown.events),
        })
    }

    fn submission_path_diagnostics(
        &self,
        backend: &FaultAccelerator<MockAccelerator>,
    ) -> Option<SubmissionPathDiagnostics> {
        Some(SubmissionPathDiagnostics {
            direct_bindings: backend.inner().direct_binding_admissions(),
            ..SubmissionPathDiagnostics::default()
        })
    }
}

/// The frozen suite is green against the reference backend — the
/// vendored-turned-published stack is alive in-tree.
#[test]
fn the_upstream_conformance_suite_passes_in_tree() {
    for target in [xor_target(), copy_target()] {
        let report = run(MockAccelerator::default, &target, &MockHooks);
        report.assert_conformant();
        assert!(
            report
                .cases()
                .iter()
                .filter(|case| case.id != "accounting.resource-lifecycle")
                .all(|case| case.status == CaseStatus::Passed),
            "every non-accounting case passes: {:?}",
            report.failures().collect::<Vec<_>>(),
        );
    }
}

/// Provider ownership stays balanced across every case when the
/// accounting adapter watches — the leak detector the ivpu backend will
/// run under.
#[test]
fn ownership_accounting_is_balanced_across_the_suite() {
    for target in [xor_target(), copy_target()] {
        let report = run(
            || FaultAccelerator::new(MockAccelerator::default(), FaultScript::default()),
            &target,
            &TrackedHooks,
        );
        report.assert_conformant();
        assert!(
            report
                .cases()
                .iter()
                .all(|case| case.status == CaseStatus::Passed)
        );
    }
}

/// The native opt-in adapter preserves the same borrowed handles and semantics.
impl ConformanceHooks<kore_driver::accel::ArtifactBackend<MockAccelerator>> for MockHooks {
    fn complete_event(
        &self,
        backend: &kore_driver::accel::ArtifactBackend<MockAccelerator>,
        event: &MockEvent,
    ) -> Result<(), BackendError> {
        backend.complete(event)
    }

    fn submission_path_diagnostics(
        &self,
        backend: &kore_driver::accel::ArtifactBackend<MockAccelerator>,
    ) -> Option<SubmissionPathDiagnostics> {
        Some(SubmissionPathDiagnostics {
            direct_bindings: backend.direct_binding_admissions(),
            ..SubmissionPathDiagnostics::default()
        })
    }
}

#[test]
fn native_artifact_adapter_preserves_upstream_conformance() {
    for target in [xor_target(), copy_target()] {
        let report = run(
            kore_driver::accel::ArtifactBackend::<MockAccelerator>::default,
            &target,
            &MockHooks,
        );
        report.assert_conformant();
        assert!(
            report
                .cases()
                .iter()
                .filter(|case| case.id != "accounting.resource-lifecycle")
                .all(|case| case.status == CaseStatus::Passed)
        );
    }
}
