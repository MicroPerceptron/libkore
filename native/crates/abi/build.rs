//! Which image's generated sources this build includes
//! (`res/IMAGE_PROFILES.md` §7: the default image's are committed beside the
//! code; any other image's are generated per build under
//! `target/images/<name>/gen/`, which `KORE_IMAGE_GEN` names).
fn main() {
    println!("cargo:rerun-if-env-changed=KORE_IMAGE_GEN");
    println!("cargo::rustc-check-cfg=cfg(kore_image_gen)");
    if std::env::var_os("KORE_IMAGE_GEN").is_some() {
        println!("cargo:rustc-cfg=kore_image_gen");
    }
}
