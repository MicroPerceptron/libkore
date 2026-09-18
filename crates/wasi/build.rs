//! Build the stock guest fixture (`guests/hello`) for `wasm32-wasip2`, when
//! the `fixture` feature asks for it.
//!
//! One build path for its two consumers: this crate's host test (run with
//! `--features fixture`), which resolves it against the real linker, and
//! Kore's `wasm-witness`, which runs it on Kore. It builds in its own target directory under
//! `OUT_DIR`, so the outer build's lock and target are untouched, and with the
//! outer build's flags cleared — they name kore's target, not wasm's.

use std::{env, path::PathBuf, process::Command};

fn main() {
    // Only a consumer asking for the fixture pays for a wasm toolchain.
    if env::var_os("CARGO_FEATURE_FIXTURE").is_none() {
        return;
    }
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"))
        .join("guests/hello");
    println!("cargo:rerun-if-changed={}", manifest.join("src").display());
    println!(
        "cargo:rerun-if-changed={}",
        manifest.join("Cargo.toml").display()
    );

    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let target_dir = out.join("guest-target");
    let status = Command::new(env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
        .args(["build", "--quiet", "--release", "--target", "wasm32-wasip2"])
        .arg("--manifest-path")
        .arg(manifest.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(&target_dir)
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env_remove("CARGO_BUILD_TARGET")
        .env_remove("CARGO_TARGET_DIR")
        .status()
        .expect("run cargo for the wasm32-wasip2 guest fixture");
    assert!(
        status.success(),
        "guests/hello did not build for wasm32-wasip2 (rust-toolchain.toml lists the target)"
    );
    std::fs::copy(
        target_dir.join("wasm32-wasip2/release/wasi-hello.wasm"),
        out.join("wasi-hello.wasm"),
    )
    .expect("copy the guest fixture");
}
