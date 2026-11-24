#![cfg(feature = "xdna")]

use axnn_vaccel::xdna::{BF16_IDENTITY_LINE_ELEMENTS, bf16_identity, unsupported_fp32_identity};
use axnn_vaccel::{AxTosaAccelerator, BackendError, Session};
use virtio_accel_xdna::{AdmitError, CompilerSpec, XDNA_TOSA_TARGET, admit};

#[test]
fn xdna_line_bf16_probe_is_admitted_precisely() {
    let artifact = bf16_identity(BF16_IDENTITY_LINE_ELEMENTS).expect("author BF16 identity");
    assert_eq!(
        admit(&artifact, XDNA_TOSA_TARGET),
        Ok(CompilerSpec::Identity {
            elements: BF16_IDENTITY_LINE_ELEMENTS,
        })
    );
}

#[test]
fn fp32_identity_is_a_fail_loud_negative_control() {
    let artifact =
        unsupported_fp32_identity(BF16_IDENTITY_LINE_ELEMENTS).expect("author valid FP32 identity");
    assert_eq!(
        admit(&artifact, XDNA_TOSA_TARGET),
        Err(AdmitError::Unsupported)
    );
}

#[cfg(not(axnn_vaccel_xdna_runtime))]
#[test]
fn unavailable_native_runtime_is_reported_explicitly() {
    assert!(matches!(
        virtio_accel_xdna::XdnaAccelerator::new(),
        Err(virtio_accel_xdna::InitError::RuntimeUnavailable)
    ));
}

#[test]
fn session_alignment_is_explicit_and_validated() {
    let session = Session::new_with_buffer_alignment(
        AxTosaAccelerator::new(),
        AxTosaAccelerator::target(),
        4 * 1024,
    )
    .expect("construct page-aligned session");
    assert_eq!(session.buffer_alignment(), 4 * 1024);
    assert!(matches!(
        Session::new_with_buffer_alignment(
            AxTosaAccelerator::new(),
            AxTosaAccelerator::target(),
            3,
        ),
        Err(BackendError::InvalidArgument)
    ));
}
