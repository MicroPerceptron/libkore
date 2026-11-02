//! `kore_vol` as a host program drives it: the same procedure the binary
//! prints over, answered as data — what a VM supervisor or a volume browser
//! reads back off a disk a machine left.

mod common;

use std::{collections::BTreeMap, fs, path::Path};

use common::*;
use kore_vol::{Kind, Listing, VolError};

/// Every file under `root`, by its `/`-separated path below it.
fn host_files(root: &Path) -> BTreeMap<String, Vec<u8>> {
    fn visit(root: &Path, at: &Path, files: &mut BTreeMap<String, Vec<u8>>) {
        for entry in fs::read_dir(at).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                visit(root, &path, files);
            } else {
                let relative = path.strip_prefix(root).unwrap().to_str().unwrap();
                files.insert(relative.replace('\\', "/"), fs::read(&path).unwrap());
            }
        }
    }
    let mut files = BTreeMap::new();
    visit(root, root, &mut files);
    files
}

#[test]
fn a_tree_put_through_the_library_reads_back_whole_through_a_mount() {
    let dir = scratch("library-roundtrip");
    let image = blank_image(&dir, 64 * MIB);
    let source = bundle(&dir, 7);

    let formatted = kore_vol::format(&image, None).unwrap();
    let installed = kore_vol::put(&image, None, 0, &source, "js/hello", false).unwrap();
    assert_eq!(installed.files, 7);
    assert_eq!(installed.directories, 3);
    assert!(installed.before >= formatted.generation);
    assert!(installed.after > installed.before);

    let expected = host_files(&source);
    let (listing, files, walked_dirs) = kore_vol::read(&image, None, |mount| {
        assert_eq!(mount.status().generation, installed.after);
        assert!(!mount.status().used_fallback_root);
        let listing = mount.list(0, Some("js/hello"))?;
        let mut files = BTreeMap::new();
        let mut dirs = Vec::new();
        let mut paths = Vec::new();
        mount.walk(0, "js/hello", &mut |walked| {
            paths.push(walked.path.to_owned());
            assert_eq!(walked.depth, walked.path.split('/').count());
            match walked.kind {
                Kind::Directory => dirs.push(walked.path.to_owned()),
                Kind::File => {
                    let mut bytes = Vec::new();
                    let len = mount.read_to(0, &format!("js/hello/{}", walked.path), &mut bytes)?;
                    assert_eq!(len, walked.len);
                    files.insert(walked.path.to_owned(), bytes);
                }
            }
            Ok(())
        })?;
        // Each directory is visited before what it holds.
        for dir in &dirs {
            let at = paths.iter().position(|path| path == dir).unwrap();
            assert!(
                paths[..at]
                    .iter()
                    .all(|path| !path.starts_with(&format!("{dir}/")))
            );
        }
        Ok((listing, files, dirs))
    })
    .unwrap();

    let Listing::Directory(entries) = listing else {
        panic!("js/hello is a directory");
    };
    let mut names = entries
        .iter()
        .map(|entry| (entry.name.as_str(), entry.kind))
        .collect::<Vec<_>>();
    names.sort_by_key(|(name, _)| *name);
    assert_eq!(
        names,
        [
            ("assets", Kind::Directory),
            ("index.html", Kind::File),
            ("vite.svg", Kind::File)
        ]
    );
    assert_eq!(files, expected);
    let mut walked_dirs = walked_dirs;
    walked_dirs.sort();
    assert_eq!(walked_dirs, ["assets", "assets/fonts"]);
}

#[test]
fn a_mount_answers_files_homes_and_the_volume_root_as_data() {
    let dir = scratch("library-listing");
    let image = blank_image(&dir, 32 * MIB);
    let source = dir.join("notes.txt");
    fs::write(&source, b"kore\n").unwrap();
    kore_vol::format(&image, None).unwrap();

    // A home nothing has written to is empty, not missing.
    let home = kore_vol::read(&image, None, |mount| mount.list(1, None)).unwrap();
    assert_eq!(home, Listing::EmptyHome);

    kore_vol::put(&image, None, 1, &source, "notes.txt", false).unwrap();
    kore_vol::read(&image, None, |mount| {
        assert_eq!(mount.list(1, Some("notes.txt"))?, Listing::File { len: 5 });
        let Listing::Directory(root) = mount.list(0, Some("/"))? else {
            panic!("the volume's root is a directory");
        };
        assert!(root.iter().any(|entry| entry.name == "work"));
        let Listing::Directory(faces) = mount.list(0, Some("/work"))? else {
            panic!("work is a directory");
        };
        assert!(faces.iter().any(|entry| entry.name == "face-1"));
        // What is not there is said to be not there, and a file is not a
        // directory to walk nor a directory a file to read.
        assert!(matches!(
            mount.list(1, Some("missing")),
            Err(VolError::NotFound(_))
        ));
        assert!(matches!(
            mount.walk(1, "notes.txt", &mut |_| Ok(())),
            Err(VolError::Fs(_, _))
        ));
        assert!(matches!(
            mount.read_to(0, "/work", &mut Vec::new()),
            Err(VolError::Fs(_, _))
        ));
        Ok(())
    })
    .unwrap();
}

#[test]
fn a_visit_that_refuses_stops_the_walk_with_its_refusal() {
    let dir = scratch("library-refusal");
    let image = blank_image(&dir, 64 * MIB);
    let source = bundle(&dir, 9);
    kore_vol::format(&image, None).unwrap();
    kore_vol::put(&image, None, 0, &source, "app", false).unwrap();
    let mut seen = 0;
    let outcome = kore_vol::read(&image, None, |mount| {
        mount.walk(0, "app", &mut |_| {
            seen += 1;
            Err(VolError::Host("enough".into()))
        })
    });
    assert!(matches!(outcome, Err(VolError::Host(message)) if message == "enough"));
    assert_eq!(seen, 1);
}

#[test]
fn a_blank_image_is_refused_to_read_and_the_caller_needs_no_stack_of_its_own() {
    let dir = scratch("library-stack");
    let image = blank_image(&dir, 32 * MIB);
    // The engine recovers on the library's own thread: a caller on a small
    // stack is as good as any.
    let small = std::thread::Builder::new()
        .stack_size(128 * 1024)
        .spawn(move || {
            let blank = kore_vol::read(&image, None, |_| Ok(()));
            assert!(matches!(blank, Err(VolError::Blank)));
            kore_vol::format(&image, None).unwrap();
            kore_vol::read(&image, None, |mount| Ok(mount.status().generation)).unwrap()
        })
        .unwrap();
    assert!(small.join().unwrap() >= 1);
}

#[test]
fn a_sink_that_fails_is_the_output_s_refusal_not_the_volume_s() {
    struct Closed;
    impl std::io::Write for Closed {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::ErrorKind::BrokenPipe.into())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let dir = scratch("library-sink");
    let image = blank_image(&dir, 32 * MIB);
    let source = dir.join("blob");
    fs::write(&source, pattern(3, 70_000)).unwrap();
    kore_vol::format(&image, None).unwrap();
    kore_vol::put(&image, None, 0, &source, "blob", false).unwrap();
    let outcome = kore_vol::read(&image, None, |mount| mount.read_to(0, "blob", &mut Closed));
    assert!(matches!(outcome, Err(VolError::Output(_))));
    // The volume is as it was.
    let mut bytes = Vec::new();
    kore_vol::read(&image, None, |mount| mount.read_to(0, "blob", &mut bytes)).unwrap();
    assert_eq!(bytes, pattern(3, 70_000));
}

#[test]
fn a_mount_releases_its_handles_whatever_a_read_or_a_listing_meets() {
    /// Takes one byte, then has had enough: a preview's bound.
    struct Enough(usize);
    impl std::io::Write for Enough {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            if self.0 == 0 {
                return Err(std::io::Error::other("enough"));
            }
            self.0 -= 1;
            Ok(buf.len().min(1))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let dir = scratch("library-handles");
    let image = blank_image(&dir, 32 * MIB);
    let source = dir.join("blob");
    fs::write(&source, pattern(5, 70_000)).unwrap();
    kore_vol::format(&image, None).unwrap();
    kore_vol::put(&image, None, 0, &source, "shots/blob", false).unwrap();
    // A session holds at most `MAX_HANDLES_PER_SESSION` (32) handles; each
    // refusal below, repeated well past that, must hand back what it opened.
    kore_vol::read(&image, None, |mount| {
        for _ in 0..100 {
            assert!(matches!(
                mount.read_to(0, "shots/blob", &mut Enough(1)),
                Err(VolError::Output(_))
            ));
            assert!(matches!(
                mount.read_to(0, "shots/missing", &mut Vec::new()),
                Err(VolError::NotFound(_))
            ));
            assert!(matches!(
                mount.read_to(0, "shots", &mut Vec::new()),
                Err(VolError::Fs(_, _))
            ));
            assert!(matches!(
                mount.list(0, Some("shots/missing")),
                Err(VolError::NotFound(_))
            ));
            assert!(matches!(
                mount.walk(0, "shots/blob", &mut |_| Ok(())),
                Err(VolError::Fs(_, _))
            ));
        }
        // And the session is still whole.
        let Listing::Directory(entries) = mount.list(0, Some("shots"))? else {
            panic!("shots is a directory");
        };
        assert_eq!(entries.len(), 1);
        let mut bytes = Vec::new();
        assert_eq!(mount.read_to(0, "shots/blob", &mut bytes)?, 70_000);
        assert_eq!(bytes, pattern(5, 70_000));
        Ok(())
    })
    .unwrap();
}
