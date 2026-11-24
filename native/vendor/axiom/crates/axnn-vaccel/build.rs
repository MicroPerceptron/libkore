//! Mirror native provider build decisions for local cfg use.
//!
//! Cargo does not expose dependencies' private `va_openvino` / `va_xdna` /
//! `va_vulkan` cfgs to their dependents. The providers deliberately compile placeholders
//! without `Accelerator` implementations when their native runtimes are
//! absent, so examples and integration tests must make the same decision
//! before naming those types in a generic `Session` or `AxHetero`.

#![forbid(unsafe_code)]

#[path = "build/providers.rs"]
mod providers;

fn main() {
    providers::probe_xdna();
    providers::probe_openvino();
    providers::probe_vulkan();
    #[cfg(feature = "metal")]
    build_metal();
}

#[cfg(feature = "metal")]
fn build_metal() {
    println!("cargo:rerun-if-changed=src/metal/bridge.mm");
    println!("cargo:rerun-if-changed=src/metal/kernels.metal");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    let mut build = cc::Build::new();
    // GNU ar can appear first on a Homebrew PATH but emit Mach-O members with
    // alignment rejected by Apple's linker. Prefer the SDK's native archiver
    // unless the caller deliberately selected one through cc's AR variables.
    let target = std::env::var("TARGET").expect("Cargo TARGET is available");
    let explicit_ar = [
        "AR".to_owned(),
        "HOST_AR".to_owned(),
        "TARGET_AR".to_owned(),
        format!("AR_{target}"),
        format!("AR_{}", target.replace('-', "_")),
    ];
    for key in &explicit_ar {
        println!("cargo:rerun-if-env-changed={key}");
    }
    if cfg!(target_os = "macos")
        && explicit_ar
            .iter()
            .all(|key| std::env::var_os(key).is_none())
    {
        let found = std::process::Command::new("/usr/bin/xcrun")
            .args(["--find", "ar"])
            .output()
            .expect("Metal bridge requires the macOS command line tools");
        assert!(
            found.status.success(),
            "xcrun could not find the native archiver"
        );
        let archiver =
            std::str::from_utf8(&found.stdout).expect("xcrun returned an invalid tool path");
        build.archiver(archiver.trim());
    }
    build
        .cpp(true)
        .std("c++17")
        .file("src/metal/bridge.mm")
        .flag("-fobjc-arc")
        .flag("-fobjc-arc-exceptions")
        .flag("-fobjc-exceptions")
        .flag("-ffp-contract=off")
        .flag("-fno-fast-math")
        .compile("axiom_metal_field");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
}
