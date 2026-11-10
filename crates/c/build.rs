//! `libkore.so`'s link contract on a Kore target, carried by the crate so
//! every builder produces the same image: the soname its consumers record as
//! `DT_NEEDED`, and no undefined symbol, since the library binds nothing from
//! the program it links into.
fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("none") {
        println!("cargo:rustc-cdylib-link-arg=--soname=libkore.so");
        println!("cargo:rustc-cdylib-link-arg=--no-undefined");
    }
}
