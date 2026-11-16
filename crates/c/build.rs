//! Publish the C headers `libkore.so` implements to whoever links this crate
//! (`links = "kore"`): a dependent's build script reads `DEP_KORE_INCLUDE`, the
//! directory holding `kore.h` and `kore/`, from this exact source revision, so
//! the headers a distribution ships cannot drift from the image it builds.
use std::path::PathBuf;

fn main() {
    let manifest =
        PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let include = manifest.join("../../extern/c/include");
    let include = include
        .canonicalize()
        .unwrap_or_else(|error| panic!("{}: {error}", include.display()));
    for header in [
        "kore.h",
        "kore/stdlib.h",
        "kore/pthread.h",
        "kore/program.h",
    ] {
        let path = include.join(header);
        assert!(
            path.is_file(),
            "missing generated header {}",
            path.display()
        );
        println!("cargo:rerun-if-changed={}", path.display());
    }
    println!("cargo:include={}", include.display());
}
