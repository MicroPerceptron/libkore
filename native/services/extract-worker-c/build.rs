fn main() {
    let vendor = "../extract/vendor/jsmn";
    println!("cargo:rerun-if-changed={vendor}/jsmn.h");
    println!("cargo:rerun-if-changed={vendor}/kore_jsmn.c");
    let mut build = cc::Build::new();
    let host = std::env::var("HOST").unwrap_or_default();
    let target = std::env::var("TARGET").unwrap_or_default();

    // cc-rs has no packaged cross compiler for kore's custom user triples,
    // so it falls back to the single-target host `cc`; an x86_64 host then
    // silently emits x86_64 objects into the aarch64/riscv worker link and
    // LLD correctly refuses the mix.  Clang is an all-targets driver and
    // cc-rs hands it the exact `--target` triple, so prefer it for every
    // freestanding cross build.  An explicit CC/KORE_LLVM_CLANG remains
    // authoritative.
    //
    // On Darwin the PATH clang is Xcode's, deliberately built with only
    // Apple's native targets: it can compile the aarch64 worker by accident
    // on Apple Silicon but fails as soon as the same ordinary image is
    // checked for RISC-V, so a complete LLVM toolchain is required there —
    // the two fallback paths are Homebrew's documented Apple Silicon and
    // Intel prefixes.  Every other host keeps its ordinary PATH clang.
    if host != target && !target.contains("apple") {
        let explicit_cc = std::env::var_os("CC").is_some()
            || std::env::var_os(format!("CC_{}", target.replace('-', "_"))).is_some()
            || std::env::var_os(format!("CC_{target}")).is_some();
        if !explicit_cc {
            let clang = std::env::var_os("KORE_LLVM_CLANG")
                .map(std::path::PathBuf::from)
                .or_else(|| {
                    host.contains("apple-darwin")
                        .then(|| {
                            [
                                "/opt/homebrew/opt/llvm/bin/clang",
                                "/usr/local/opt/llvm/bin/clang",
                            ]
                            .into_iter()
                            .map(std::path::PathBuf::from)
                            .find(|path| path.is_file())
                        })
                        .flatten()
                })
                .unwrap_or_else(|| {
                    if host.contains("apple-darwin") {
                        panic!(
                            "freestanding C worker for {target} needs a complete LLVM clang; \
                             install Homebrew llvm or set KORE_LLVM_CLANG/CC"
                        )
                    }
                    std::path::PathBuf::from("clang")
                });
            let llvm_ar = clang.with_file_name("llvm-ar");
            build.compiler(clang);
            if llvm_ar.is_file() {
                build.archiver(llvm_ar);
            }
        }
    }
    // Keep the C object's ELF ABI identical to the custom Rust target.  Clang
    // otherwise defaults bare riscv64 to lp64, while kore's rv64gc user ABI is
    // explicitly lp64d; LLD correctly refuses to mix the two.
    if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("riscv64") {
        build.flag("-march=rv64gc").flag("-mabi=lp64d");
    }
    // A developer may have GNU binutils ahead of Xcode in PATH. GNU `ar`
    // emits a valid generic archive but does not preserve the 8-byte Mach-O
    // member placement required by Apple's linker. Select the platform shim
    // for Apple targets; every non-Apple and freestanding build keeps cc-rs's
    // target-aware archiver discovery.
    if std::env::var("CARGO_CFG_TARGET_VENDOR").as_deref() == Ok("apple") {
        build.archiver("/usr/bin/ar");
    }
    build
        .file(format!("{vendor}/kore_jsmn.c"))
        .include(vendor)
        .define("JSMN_STRICT", None)
        // The worker is an ordinary ET_DYN generation.  Compile the vendored
        // adapter as PIC as well; otherwise local jump-table/rodata references
        // become absolute relocations that the generic PIE linker correctly
        // rejects.
        .pic(true)
        .flag_if_supported("-ffreestanding")
        // The kore user ABI has no SysV red zone and this deliberately
        // minimal parser worker receives no FPU/SIMD authority. Keep the C
        // object on the same integer-only, kernel-safe profile as its Rust
        // host image; otherwise clang may issue SSE before the task owns an
        // FPU ticket and wedge in the #NM path instead of reporting failure.
        .flag_if_supported("-mno-red-zone")
        .flag_if_supported("-mno-sse")
        .flag_if_supported("-mno-sse2")
        .flag_if_supported("-mno-avx")
        .flag_if_supported("-mno-avx2")
        .flag_if_supported("-mno-mmx")
        // res/SPMD.md §13.6 (R7): the x87/MMX bank is reserved as the
        // lane tier's register room, so no profile may emit it — this
        // worker's soft-float already avoids x87 arithmetic; the flag
        // makes the reservation explicit rather than incidental.
        .flag_if_supported("-mno-x87")
        .flag_if_supported("-msoft-float")
        .flag_if_supported("-fno-stack-protector")
        .warnings(true)
        .compile("kore_jsmn");
}
