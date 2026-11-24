//! Compile-time tier auto-detection.
//!
//! The best kernel tier the *target* allows is selected automatically
//! from `CARGO_CFG_TARGET_FEATURE` — a plain `cargo build` gets the
//! fastest sound kernels with no cargo features and no RUSTFLAGS
//! tuning. The principle: if the compiler is already permitted to emit
//! an instruction set (it's in the target feature set), selecting the
//! tier that uses it is always sound; no runtime detection is needed.
//!
//! The cargo features (`neon`, `avx2`, …) remain as *explicit
//! overrides* for cross-builds where the builder asserts a CPU
//! contract the target triple doesn't carry; `max` is retained as a
//! no-op for compatibility (auto-detection is now unconditional).
//!
//! Exception: Intel AMX stays opt-in (`feature = "amx"`) — tile
//! configuration depends on OS state (XTILEDATA permission), which the
//! target feature set cannot promise.

fn main() {
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    match target_arch.as_str() {
        "x86_64" => detect_x86_64(),
        "aarch64" => detect_aarch64(),
        _ => {} // portable fallback
    }
}

fn detect_x86_64() {
    let features = std::env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let has = |f: &str| features.split(',').any(|feat| feat == f);

    // The AVX2 float-override tier assumes FMA + F16C (true on every
    // real AVX2 CPU, but the auto path only engages when the target
    // feature set actually says so).
    let avx2_full = has("avx2") && has("fma") && has("f16c");

    // Float/FP8 kernels and qdot are separate dispatch axes.  Emit the
    // AVX2 base tier cumulatively so AVX-VNNI/AVX-512 targets still get
    // the AVX2 float and F16C kernels while qdot selects the higher tier.
    if avx2_full {
        println!("cargo:rustc-cfg=feature=\"avx2\"");
    }

    if has("avx512bw") && avx2_full {
        println!("cargo:rustc-cfg=feature=\"avx512\"");
    }
    // NVFP4 W4A8 is a separate choice from legacy qdot. Some AVX-512 VNNI
    // CPUs lack the distinct 256-bit AVX-VNNI instruction set.
    if has("avxvnni") && avx2_full {
        println!("cargo:rustc-cfg=feature=\"avx_vnni\"");
    }
    if (has("avxvnni") || (has("avx512vnni") && has("avx512bw"))) && avx2_full {
        println!("cargo:rustc-cfg=feature=\"nvfp4_vnni\"");
    }
    if !avx2_full && has("sse4.2") {
        println!("cargo:rustc-cfg=feature=\"sse42\"");
    } else if !avx2_full && has("sse2") {
        println!("cargo:rustc-cfg=feature=\"sse2\"");
    }
}

fn detect_aarch64() {
    // NEON is architecturally mandatory on AArch64.
    println!("cargo:rustc-cfg=feature=\"neon\"");
}
