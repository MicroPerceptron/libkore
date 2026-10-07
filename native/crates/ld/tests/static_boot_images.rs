//! Validate the exact static images a boot target is about to ship.
//!
//! `KORE_STATIC_BOOT_ARCH` selects the existing relocation contract used by
//! the loader. It defaults to `x86_64` for older callers and accepts
//! `aarch64` for the corresponding production-media gate.

use std::path::{Path, PathBuf};

use kore_abi::image::seg;
use kore_ld::check::check_image;
use kore_ld::reloc::{AArch64, RelocArch, X86_64};

fn check_configured_images<A: RelocArch>(images: &str) {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("kore-ld must live under <workspace>/crates/ld")
        .to_owned();
    for path in images.split(':').filter(|path| !path.is_empty()) {
        let resolved = workspace.join(path);
        let bytes = std::fs::read(&resolved)
            .unwrap_or_else(|err| panic!("{}: read failed: {err}", Path::new(path).display()));
        let image = check_image::<A>(&bytes).unwrap_or_else(|err| {
            panic!(
                "{}: startup linker refusal: {err:?}",
                Path::new(path).display()
            )
        });
        assert!(
            image
                .descriptor
                .live_segs()
                .iter()
                .filter(|s| s.kind != seg::KIND_TLS)
                .all(|s| s.vaddr.get() % 4096 == 0 && s.memsz.get() % 4096 == 0),
            "{}: Register emitted a non-page-granular mapping",
            Path::new(path).display(),
        );
    }
}

#[test]
fn configured_static_boot_images_are_loader_compatible() {
    let Ok(images) = std::env::var("KORE_STATIC_BOOT_IMAGES") else {
        return;
    };
    match std::env::var("KORE_STATIC_BOOT_ARCH").as_deref() {
        Ok("aarch64") => check_configured_images::<AArch64>(&images),
        Ok("x86_64") | Err(_) => check_configured_images::<X86_64>(&images),
        Ok(other) => {
            panic!("KORE_STATIC_BOOT_ARCH={other:?} is unsupported; expected x86_64 or aarch64")
        }
    }
}
