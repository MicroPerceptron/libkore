//! Optional native provider cfgs, shared by the adapter and workspace games.

pub fn probe_openvino() {
    println!("cargo::rustc-check-cfg=cfg(axnn_vaccel_openvino_runtime)");
    println!("cargo:rerun-if-env-changed=VIRTIO_ACCEL_OPENVINO");
    println!("cargo:rerun-if-env-changed=VIRTIO_ACCEL_OPENVINO_LIB_DIR");

    // Do not probe or interpret an OpenVINO override unless this package's
    // optional provider is actually selected. This mirrors Cargo's optional
    // dependency boundary and keeps ordinary CPU/Core ML builds independent
    // of an unrelated host installation.
    if std::env::var_os("CARGO_FEATURE_OPENVINO").is_none() {
        return;
    }

    let forced = match std::env::var("VIRTIO_ACCEL_OPENVINO") {
        Ok(value) if value == "0" => return,
        Ok(value) if value == "1" => true,
        Ok(other) => panic!("VIRTIO_ACCEL_OPENVINO must be \"0\" or \"1\", not {other:?}"),
        Err(_) => false,
    };

    // Keep this probe behavior aligned with virtio-accel-openvino 0.3.9.
    // Linking remains the provider crate's responsibility; we only publish a
    // cfg that prevents typechecking its unsupported-runtime placeholder as an
    // `Accelerator`.
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
        println!("cargo::rustc-cfg=axnn_vaccel_openvino_runtime");
    }
}

pub fn probe_xdna() {
    println!("cargo::rustc-check-cfg=cfg(axnn_vaccel_xdna_runtime)");
    for key in [
        "VIRTIO_ACCEL_XDNA",
        "VIRTIO_ACCEL_HRX_DIR",
        "HRX_DIR",
        "VIRTIO_ACCEL_HRX_LIB_DIR",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    if std::env::var_os("CARGO_FEATURE_XDNA").is_none() {
        return;
    }
    let forced = match std::env::var("VIRTIO_ACCEL_XDNA") {
        Ok(value) if value == "0" => return,
        Ok(value) if value == "1" => true,
        Ok(value) => panic!("VIRTIO_ACCEL_XDNA must be \"0\" or \"1\", not {value:?}"),
        Err(_) => false,
    };
    // Mirror virtio-accel-xdna 0.3.9 exactly; only the provider links libhrx.
    let bare_library = std::env::var("VIRTIO_ACCEL_HRX_LIB_DIR").is_ok();
    let prefix = std::env::var_os("VIRTIO_ACCEL_HRX_DIR")
        .or_else(|| std::env::var_os("HRX_DIR"))
        .map(std::path::PathBuf::from);
    let complete = prefix.is_some_and(|prefix| {
        let header = prefix.join("include/hrx/hrx_amdxdna.h");
        let runtime = prefix.join("include/hrx/hrx_runtime.h");
        println!("cargo:rerun-if-changed={}", header.display());
        println!("cargo:rerun-if-changed={}", runtime.display());
        runtime.is_file()
            && prefix.join("lib/libhrx.so").is_file()
            && std::fs::read_to_string(header)
                .is_ok_and(|text| text.contains("hrx_amdxdna_executable_create"))
    });
    assert!(
        bare_library || complete || !forced,
        "VIRTIO_ACCEL_XDNA=1 requires a complete HRX prefix or VIRTIO_ACCEL_HRX_LIB_DIR"
    );
    if bare_library || complete {
        println!("cargo::rustc-cfg=axnn_vaccel_xdna_runtime");
    }
}

/// Host target operating systems on which virtio-accel-vulkan binds a Vulkan
/// loader at run time. Mirror virtio-accel-vulkan 0.3.9 `build.rs` exactly:
/// there is nothing to probe on disk, so the decision is the target OS plus
/// the same three-state `VIRTIO_ACCEL_VULKAN` control. Loader or device
/// absence is a run-time `InitError`, never a build-time fact.
const VULKAN_LOADER_TARGETS: &[&str] = &["android", "linux", "macos", "windows"];

pub fn probe_vulkan() {
    println!("cargo::rustc-check-cfg=cfg(axnn_vaccel_vulkan_runtime)");
    println!("cargo:rerun-if-env-changed=VIRTIO_ACCEL_VULKAN");
    if std::env::var_os("CARGO_FEATURE_VULKAN").is_none() {
        return;
    }
    let forced = match std::env::var("VIRTIO_ACCEL_VULKAN") {
        Ok(value) if value == "0" => return,
        Ok(value) if value == "1" => true,
        Ok(other) => panic!("VIRTIO_ACCEL_VULKAN must be \"0\" or \"1\", not {other:?}"),
        Err(_) => false,
    };
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let supported = VULKAN_LOADER_TARGETS.contains(&os.as_str());
    assert!(
        supported || !forced,
        "VIRTIO_ACCEL_VULKAN=1 but target OS {os:?} is not a Vulkan loader host target"
    );
    if supported {
        println!("cargo::rustc-cfg=axnn_vaccel_vulkan_runtime");
    }
}
