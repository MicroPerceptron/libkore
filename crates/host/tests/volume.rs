//! The exported volume procedure as a host program consumes it from libkore:
//! the distribution's own closure and lockfile, not Kore's workspace.

use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU32, Ordering},
};

use kore_host::volume::{self, Kind, Listing, VolError};

/// A fresh scratch directory under Cargo's per-target temporary area.
fn scratch(test: &str) -> PathBuf {
    static SEQUENCE: AtomicU32 = AtomicU32::new(0);
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "kore-host-{test}-{}-{}",
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// A zeroed (sparse) raw image: blank roots, as `qemu-img create` makes one.
fn blank_image(dir: &Path, bytes: u64) -> PathBuf {
    let path = dir.join("disk.raw");
    fs::File::create(&path).unwrap().set_len(bytes).unwrap();
    path
}

#[test]
fn a_host_tree_installs_and_reads_back_through_the_distribution() {
    let dir = scratch("roundtrip");
    let image = blank_image(&dir, 32 << 20);
    let source = dir.join("shots");
    fs::create_dir_all(source.join("2026-11")).unwrap();
    fs::write(source.join("a.png"), [0x89, b'P', b'N', b'G']).unwrap();
    fs::write(source.join("2026-11/b.log"), b"boot: ok\n".repeat(1000)).unwrap();

    // Blank roots are the first boot's to format, unless asked.
    assert!(matches!(
        volume::read(&image, None, |_| Ok(())),
        Err(VolError::Blank)
    ));
    let formatted = volume::format(&image, None).unwrap();
    let installed = volume::put(&image, None, 0, &source, "pictures/screenshots", false).unwrap();
    assert_eq!((installed.files, installed.directories), (2, 2));
    assert!(installed.after > formatted.generation);

    let (listing, mut walked, log) = volume::read(&image, None, |mount| {
        let listing = mount.list(0, Some("pictures/screenshots"))?;
        let mut walked = Vec::new();
        mount.walk(0, "pictures", &mut |entry| {
            walked.push((entry.path.to_owned(), entry.kind, entry.len));
            Ok(())
        })?;
        let mut log = Vec::new();
        mount.read_to(0, "pictures/screenshots/2026-11/b.log", &mut log)?;
        Ok((listing, walked, log))
    })
    .unwrap();
    let Listing::Directory(mut entries) = listing else {
        panic!("the screenshots directory lists as a directory");
    };
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    assert_eq!(
        entries
            .iter()
            .map(|entry| (entry.name.as_str(), entry.kind, entry.len))
            .collect::<Vec<_>>(),
        [("2026-11", Kind::Directory, 0), ("a.png", Kind::File, 4)]
    );
    walked.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(
        walked,
        [
            ("screenshots".into(), Kind::Directory, 0),
            ("screenshots/2026-11".into(), Kind::Directory, 0),
            ("screenshots/2026-11/b.log".into(), Kind::File, 9000),
            ("screenshots/a.png".into(), Kind::File, 4),
        ]
    );
    assert_eq!(log, b"boot: ok\n".repeat(1000));

    let removed = volume::rm(&image, None, 0, "pictures/screenshots/a.png").unwrap();
    assert!(removed.after > removed.before);
    let gone = volume::read(&image, None, |mount| {
        mount.list(0, Some("pictures/screenshots/a.png"))
    });
    assert!(matches!(gone, Err(VolError::NotFound(_))));
}

#[test]
fn what_kore_cannot_name_is_refused_before_the_image_is_written() {
    let dir = scratch("names");
    let image = blank_image(&dir, 16 << 20);
    volume::format(&image, None).unwrap();
    let before = fs::read(&image).unwrap();
    let source = dir.join("too-long");
    fs::create_dir_all(&source).unwrap();
    fs::write(source.join("a-name-of-thirty-three-bytes.txt0"), b"x").unwrap();
    assert!(volume::Name::new("a-name-of-thirty-three-bytes.txt0").is_err());
    let refused = volume::put(&image, None, 0, &source, "x", false);
    assert!(matches!(refused, Err(VolError::Host(message)) if message.contains("not a kore name")));
    assert!(fs::read(&image).unwrap() == before);
    assert!(volume::is_face(0));
    const { assert!(volume::FORMAT_VERSION >= 5) };
}
