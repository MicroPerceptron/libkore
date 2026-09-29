use std::{
    env,
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
    process::{Command, ExitStatus},
};

const SUPPORTED_ZIG_MAJOR: u64 = 0;
const SUPPORTED_ZIG_MINOR: u64 = 16;

fn zig_release_line(version: &str) -> Option<(u64, u64)> {
    let release = version
        .split_once('-')
        .map_or(version, |(release, _)| release);
    let release = release
        .split_once('+')
        .map_or(release, |(release, _)| release);
    let mut parts = release.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let _patch: u64 = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some((major, minor))
}

fn zig_version_supported(version: &str) -> bool {
    zig_release_line(version) == Some((SUPPORTED_ZIG_MAJOR, SUPPORTED_ZIG_MINOR))
}

fn run(mut command: Command, what: &str) -> ExitStatus {
    let status = command
        .status()
        .unwrap_or_else(|error| panic!("could not run {what}: {error}"));
    if !status.success() {
        panic!("{what} failed with {status}");
    }
    status
}

fn zig_command(zig: &OsStr, out: &Path) -> Command {
    let mut command = Command::new(zig);
    command
        .env("ZIG_LOCAL_CACHE_DIR", out.join("zig-cache"))
        .env("ZIG_GLOBAL_CACHE_DIR", out.join("zig-global-cache"));
    command
}

/// Whether a C compiler answers at all.
///
/// Deliberately `--version` and not a probe compile: the question is
/// "is there a toolchain here", and a compiler that exists but cannot
/// build the engine must fail loudly rather than be skipped.
fn compiler_present(cc: &OsStr) -> bool {
    Command::new(cc)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// Build and run the C smoke tests with a host C compiler.
///
/// Answers `false` when no C compiler is available, which is a skip and
/// not a failure: an ordinary host `cargo check`/`cargo test` of this
/// workspace must not require a C toolchain. A compiler that is present
/// but fails still panics — a broken smoke is a real regression.
fn host_smoke(out: &Path, host: &str) -> bool {
    let host_cc = env::var_os("KORE_HOST_CC").unwrap_or_else(|| {
        if host.contains("apple-darwin") {
            "/usr/bin/clang".into()
        } else {
            "cc".into()
        }
    });
    if !compiler_present(&host_cc) {
        return false;
    }
    let host_smoke = out.join("lua-host-smoke");
    let mut smoke_cc = Command::new(&host_cc);
    smoke_cc
        .args(["-std=c11", "-O2", "-Ivendor/lua"])
        .args(["src/kore_lua.c", "tests/host_smoke.c", "-lm", "-o"])
        .arg(&host_smoke);
    run(smoke_cc, "native Lua protected-call smoke compilation");
    run(
        Command::new(&host_smoke),
        "native Lua protected-call smoke execution",
    );

    // Compile the C engine a second time without its libc numeric oracle, then
    // link it beneath the target's exact Rust numeric source. This exercises
    // the real C/Rust ABI and Lua two-pass caller, not merely each half alone.
    let target_engine_object = out.join("kore-lua-target-engine.o");
    let mut target_engine_cc = Command::new(&host_cc);
    target_engine_cc
        .args([
            "-std=c11",
            "-O2",
            "-fPIC",
            "-Ivendor/lua",
            "-c",
            "src/kore_lua.c",
            "-o",
        ])
        .arg(&target_engine_object);
    run(
        target_engine_cc,
        "native Lua target-engine object compilation",
    );

    let target_smoke_object = out.join("kore-lua-target-smoke.o");
    let mut target_smoke_cc = Command::new(&host_cc);
    target_smoke_cc
        .args([
            "-std=c11",
            "-O2",
            "-fPIC",
            "-DKORE_LUA_USE_TARGET_NUMERIC=1",
            "-Ivendor/lua",
            "-c",
            "tests/host_smoke.c",
            "-o",
        ])
        .arg(&target_smoke_object);
    run(
        target_smoke_cc,
        "native Lua target-smoke object compilation",
    );

    // The same harness also checks the bounded ABI directly and compares both
    // %g passes against host libc over a deterministic random corpus.
    let target_shim_smoke = out.join("lua-target-shim-smoke");
    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let mut shim_rustc = Command::new(rustc);
    shim_rustc
        .args([
            "--edition=2024",
            "--cfg",
            "kore_lua_c_engine",
            "-O",
            "tests/target_shim.rs",
            "-o",
        ])
        .arg(&target_shim_smoke)
        .arg("-C")
        .arg(format!("link-arg={}", target_engine_object.display()))
        .arg("-C")
        .arg(format!("link-arg={}", target_smoke_object.display()))
        .args(["-C", "link-arg=-lm"]);
    run(shim_rustc, "native Lua target-shim smoke compilation");
    run(
        Command::new(&target_shim_smoke),
        "native Lua target-shim smoke execution",
    );

    true
}

fn main() {
    println!("cargo:rerun-if-env-changed=KORE_ZIG");
    println!("cargo:rerun-if-env-changed=KORE_HOST_CC");
    println!("cargo:rerun-if-changed=src/numeric.rs");
    println!("cargo:rerun-if-changed=src/kore_lua.c");
    println!("cargo:rerun-if-changed=tests/host_smoke.c");
    println!("cargo:rerun-if-changed=tests/host_fs.c");
    println!("cargo:rerun-if-changed=tests/target_shim.rs");
    println!("cargo:rerun-if-changed=vendor/lua");
    println!("cargo:rerun-if-changed=vendor/musl-setjmp");

    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    fs::create_dir_all(&out).expect("create Lua build output");

    // The C smoke tests need only a host C compiler, which is worth having
    // on every build: they run the whole engine and the whole `io`/`os`/
    // `kore.fs` surface against the in-memory store in `tests/host_fs.c`,
    // and they are the only place that happens without a machine.
    //
    // They are skipped rather than fatal when no C compiler is present:
    // `cargo check`/`cargo test` of this workspace must work for a
    // consumer who has neither toolchain, which is what the README
    // promises and AGENTS.md requires. `make cross` builds for the Kore
    // targets and therefore always runs them.
    let host = env::var("HOST").expect("host triple");
    if !host_smoke(&out, &host) {
        println!(
            "cargo:warning=kore-lua: no host C compiler; skipped the engine smoke tests. \
             Set KORE_HOST_CC to run them."
        );
    }

    // The engine itself exists only where `src/lib.rs` compiles its Rust
    // half: the amalgamation, the worker thread and the doorbell are all
    // `target_os = "none"` surfaces. A host build carries `files`' logic
    // and none of the C, so there is no object for it to link and no
    // reason to require Zig.
    if env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() != "none" {
        return;
    }

    let arch = env::var("CARGO_CFG_TARGET_ARCH").expect("target architecture");
    let c_target = match arch.as_str() {
        "x86_64" => "x86_64-linux-musl",
        "aarch64" => "aarch64-linux-musl",
        "riscv64" => "riscv64-linux-musl",
        other => panic!("Lua has no kore C ABI mapping for {other}"),
    };
    let zig = env::var_os("KORE_ZIG").unwrap_or_else(|| "zig".into());
    let zig_version = Command::new(&zig)
        .arg("version")
        .output()
        .expect("could not query the supported Zig toolchain");
    assert!(zig_version.status.success(), "`zig version` failed");
    let zig_version = std::str::from_utf8(&zig_version.stdout)
        .expect("`zig version` returned non-UTF-8 output")
        .trim();
    assert!(
        zig_version_supported(zig_version),
        "Lua requires Zig {SUPPORTED_ZIG_MAJOR}.{SUPPORTED_ZIG_MINOR}.x; found {zig_version}. \
         The target C build separately verifies the musl jmp_buf ABI used by the vendored assembly"
    );
    let object = out.join("kore_lua.o");
    let archive = out.join("libkore_lua.a");
    let mut cc = zig_command(&zig, &out);
    cc.arg("cc")
        // Zig's musl target supplies ISO C headers only. The resulting object
        // is still freestanding and links no musl runtime into the kore image.
        .args(["-target", c_target, "-std=c11", "-O2"])
        .args([
            "-ffreestanding",
            "-fno-stack-protector",
            "-fno-unwind-tables",
            "-fno-asynchronous-unwind-tables",
            "-ffunction-sections",
            "-fdata-sections",
            "-fPIC",
            "-fvisibility=hidden",
            "-DKORE_LUA_ZIG_MUSL_ABI=1",
            "-Wall",
            "-Wextra",
            "-Werror=implicit-function-declaration",
        ])
        .arg("-Ivendor/lua");
    match arch.as_str() {
        "x86_64" => {
            cc.args(["-mno-red-zone", "-mno-mmx"]);
        }
        "riscv64" => {
            // Zig's riscv64-musl baseline is rv64gc. The explicit ABI keeps
            // the object identical to kore's custom Rust target.
            cc.arg("-mabi=lp64d");
        }
        _ => {}
    }
    cc.args(["-c", "src/kore_lua.c", "-o"]).arg(&object);
    run(cc, "Zig Lua C compilation");

    let asm_extension = if arch == "riscv64" { "S" } else { "s" };
    let mut asm_objects = Vec::new();
    for stem in ["setjmp", "longjmp"] {
        let source = format!("vendor/musl-setjmp/{arch}/{stem}.{asm_extension}");
        let output = out.join(format!("{stem}.o"));
        let mut assembler = zig_command(&zig, &out);
        assembler.args(["cc", "-target", c_target, "-fPIC"]);
        if arch == "riscv64" {
            assembler.arg("-mabi=lp64d");
        } else if arch == "x86_64" {
            assembler.arg("-mno-red-zone");
        }
        assembler.args(["-c", &source, "-o"]).arg(&output);
        run(assembler, "Zig setjmp assembly");
        asm_objects.push(output);
    }

    let mut ar = zig_command(&zig, &out);
    ar.args(["ar", "rcs"]).arg(&archive).arg(&object);
    for object in &asm_objects {
        ar.arg(object);
    }
    run(ar, "Zig Lua archive creation");

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=kore_lua");
}
