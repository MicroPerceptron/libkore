//! Mirror the OpenVINO provider's native-runtime decision for axrd examples.
//!
//! A dependency's private `va_openvino` cfg is not visible here. The provider
//! deliberately exposes only a placeholder when `libopenvino_c` is absent, so
//! code naming it as an `Accelerator` must use the same host-runtime probe.

#![forbid(unsafe_code)]

fn main() {
    println!("cargo::rustc-check-cfg=cfg(axrd_openvino_runtime)");
    println!("cargo:rerun-if-env-changed=VIRTIO_ACCEL_OPENVINO");
    println!("cargo:rerun-if-env-changed=VIRTIO_ACCEL_OPENVINO_LIB_DIR");

    if std::env::var_os("CARGO_FEATURE_OPENVINO").is_none() {
        return;
    }

    let forced = match std::env::var("VIRTIO_ACCEL_OPENVINO") {
        Ok(value) if value == "0" => return,
        Ok(value) if value == "1" => true,
        Ok(other) => panic!("VIRTIO_ACCEL_OPENVINO must be \"0\" or \"1\", not {other:?}"),
        Err(_) => false,
    };

    let found = if std::env::var("VIRTIO_ACCEL_OPENVINO_LIB_DIR").is_ok() {
        true
    } else {
        pkg_config::Config::new()
            .cargo_metadata(false)
            .probe("openvino")
            .is_ok()
    };
    assert!(
        found || !forced,
        "VIRTIO_ACCEL_OPENVINO=1 but no OpenVINO runtime was found"
    );
    if found {
        println!("cargo::rustc-cfg=axrd_openvino_runtime");
    }
}
