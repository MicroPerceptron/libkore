use std::{
    env,
    path::PathBuf,
    process::{Command, Stdio},
};
fn main() {
    println!("cargo:rerun-if-env-changed=ZIG");
    println!("cargo:rerun-if-changed=tests/consumer.c");
    println!("cargo:rerun-if-changed=tests/hash.c");
    println!("cargo:rerun-if-changed=tests/alloc.c");
    println!("cargo:rerun-if-changed=../../extern/zig/src/allocator.zig");
    println!("cargo:rerun-if-changed=../../extern/c/include/kore/stdlib.h");
    println!("cargo:rerun-if-changed=tests/consumer.zig");
    println!("cargo:rerun-if-changed=../../extern/c/include/kore.h");
    println!("cargo:rerun-if-changed=../../extern/zig/src/root.zig");
    if env::var_os("CARGO_FEATURE_TEST_FOREIGN").is_none()
        || env::var_os("CARGO_CFG_MIRI").is_some()
        || env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("none")
    {
        return;
    }
    let zig = env::var("ZIG").unwrap_or_else(|_| "zig".into());
    let version = Command::new(&zig)
        .arg("version")
        .output()
        .expect("Zig 0.16.0 is required for test-foreign");
    assert!(
        version.status.success() && version.stdout == b"0.16.0\n",
        "Zig 0.16.0 is required"
    );
    let output = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let translated = output.join("kore-c.zig");
    // Write translation directly to its artifact, as the binding checker does.
    // Zig 0.16 on macOS can spin in fcopyfile when stdout is a captured pipe.
    let result = Command::new(&zig)
        .args(["translate-c", "../../extern/c/include/kore.h"])
        .stdout(Stdio::from(
            std::fs::File::create(&translated).expect("create C translation"),
        ))
        .output()
        .expect("run Zig translator");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let object = output.join("zig-consumer.o");
    let status = Command::new(&zig)
        .args([
            "build-obj",
            // Rust host tests are PIE on Linux; foreign objects must support
            // relocation without weakening the executable's link policy.
            "-fPIC",
            "-O",
            "ReleaseFast",
            "--dep",
            "kore",
            "-Mroot=tests/consumer.zig",
            "--dep",
            "kore-c",
            "-Mkore=../../extern/zig/src/root.zig",
        ])
        .arg(format!("-Mkore-c={}", translated.display()))
        .arg(format!("-femit-bin={}", object.display()))
        .status()
        .unwrap();
    assert!(status.success(), "compile Zig consumer");
    cc::Build::new()
        .file("tests/consumer.c")
        .file("tests/hash.c")
        .file("tests/alloc.c")
        .include("../../extern/c/include")
        .flag("-std=c11")
        .warnings_into_errors(true)
        .object(object)
        .compile("foreign_consumers");
}
