//! `kore-vol` as its user runs it: the built binary, on disk images made
//! here, read back through the binary itself and, where the claim is about
//! bytes, through the image file.

mod common;

use std::{
    fs::{self, File, OpenOptions},
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
};

use common::*;
use kore_driver::gpt::{KORE_FS_TYPE, synthetic};

#[test]
fn a_blank_image_is_formatted_then_takes_an_install_its_replacement_and_its_removal() {
    let dir = scratch("lifecycle");
    let image = blank_image(&dir, 64 * MIB);
    let image = image.to_str().unwrap();
    let first = bundle(&dir, 1);
    let second = bundle(&dir, 2);
    fs::remove_file(second.join("vite.svg")).unwrap();
    let (first, second) = (first.to_str().unwrap(), second.to_str().unwrap());

    // Blank roots: kore formats them at its first boot, and nothing else
    // does unless asked.
    refused(&[image, "put", first, "js/hello"], "blank");
    refused(&[image, "ls"], "blank");
    assert!(
        read(Path::new(image), 0, 2 * BLOCK as usize)
            .iter()
            .all(|&b| b == 0)
    );
    assert!(ok(&[image, "format"]).contains("formatted"));
    refused(&[image, "format"], "kore's already");

    // A home nothing was ever written to is empty, not missing.
    assert_eq!(ok(&[image, "ls"]), "");

    let installed = ok(&[image, "put", first, "js/hello"]);
    assert!(installed.contains("7 files, 3 directories"), "{installed}");
    assert!(installed.contains("read back identical"), "{installed}");
    // A healthy volume read says nothing beside what it lists: opened to
    // read, it mounts read-only, which is not kore's fallback posture.
    let quiet = kore_vol(&[image, "ls", "js/hello"]);
    assert_eq!((quiet.code, quiet.stderr.as_str()), (0, ""));
    let listed = ok(&[image, "ls", "js/hello"]);
    assert_eq!(
        listed.lines().collect::<Vec<_>>(),
        [
            "dir                assets/",
            "file          431  index.html",
            "file            1  vite.svg",
        ]
    );
    let fonts = ok(&[image, "ls", "js/hello/assets/fonts"]);
    assert!(fonts.contains("file            0  .keep"), "{fonts}");
    assert!(fonts.contains("file       300001  inter.woff2"), "{fonts}");
    assert!(ok(&[image, "ls", "/work"]).contains("face-0/"));

    // Installing over an install is a decision, not an accident.
    refused(&[image, "put", second, "js/hello"], "exists");
    assert!(ok(&[image, "ls", "js/hello"]).contains("vite.svg"));
    ok(&[image, "put", "--replace", second, "js/hello"]);
    let replaced = ok(&[image, "ls", "js/hello"]);
    assert!(!replaced.contains("vite.svg"), "{replaced}");
    assert!(replaced.contains("index.html"), "{replaced}");

    // Another terminal's home is its own.
    ok(&["--face", "1", image, "put", first, "notes"]);
    assert_eq!(ok(&[image, "ls", "/work"]).lines().count(), 2);
    refused(&[image, "ls", "notes"], "not found");

    ok(&[image, "rm", "js/hello"]);
    refused(&[image, "ls", "js/hello"], "not found");
    assert_eq!(ok(&[image, "ls", "js"]), "");
    refused(&[image, "rm", "js/hello"], "not found");
}

#[test]
fn the_install_lands_inside_the_designated_partition_and_nowhere_else() {
    let dir = scratch("gpt");
    let image = blank_image(&dir, 32 * MIB);
    let blocks = 32 * MIB / BLOCK;
    let (start, end) = (2048, blocks - 2048);
    let mut mbr = [0u8; 512];
    mbr[510] = 0x55;
    mbr[511] = 0xaa;
    let (header, entries) = synthetic::table(KORE_FS_TYPE, start, end, blocks - 34);
    {
        let file = OpenOptions::new().write(true).open(&image).unwrap();
        file.write_all_at(&mbr, 0).unwrap();
        file.write_all_at(&header, BLOCK).unwrap();
        file.write_all_at(&entries, 2 * BLOCK).unwrap();
    }
    let before = read(&image, 0, (start * BLOCK) as usize);
    let tail = (blocks - end - 1) * BLOCK;
    let image_path = image.to_str().unwrap();
    let source = bundle(&dir, 3);

    ok(&[image_path, "format"]);
    ok(&[image_path, "put", source.to_str().unwrap(), "app"]);
    assert!(ok(&[image_path, "ls", "app/assets"]).contains("index.js"));

    // The table and everything ahead of the partition is as it was; so is
    // everything after it; the partition's own first block holds a root.
    assert_eq!(read(&image, 0, (start * BLOCK) as usize), before);
    assert!(
        read(&image, (end + 1) * BLOCK, tail as usize)
            .iter()
            .all(|&b| b == 0)
    );
    assert_eq!(&read(&image, start * BLOCK, 8)[..], b"KOREROOT");
}

#[test]
fn a_foreign_disk_is_refused_untouched() {
    let dir = scratch("foreign");
    let image = blank_image(&dir, 16 * MIB);
    OpenOptions::new()
        .write(true)
        .open(&image)
        .unwrap()
        .write_all_at(&pattern(9, 4096), 0)
        .unwrap();
    let before = digest(&image);
    let image_path = image.to_str().unwrap();
    refused(&[image_path, "format"], "no kore volume");
    refused(&[image_path, "ls"], "no kore volume");
    assert_eq!(digest(&image), before);
}

#[test]
fn an_image_another_process_holds_is_refused() {
    use rustix::fs::{FlockOperation, fcntl_lock};
    let dir = scratch("lock");
    let image = blank_image(&dir, 16 * MIB);
    let image_path = image.to_str().unwrap();
    ok(&[image_path, "format"]);
    // As QEMU holds the images it runs: a record lock on the file.
    let held = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&image)
        .unwrap();
    fcntl_lock(&held, FlockOperation::NonBlockingLockExclusive).unwrap();
    refused(&[image_path, "ls"], "in use");
    refused(&[image_path, "put", dir.to_str().unwrap(), "x"], "in use");
    drop(held);
    ok(&[image_path, "ls"]);
}

#[test]
fn what_the_volume_cannot_hold_is_refused_before_the_image_is_opened() {
    let dir = scratch("names");
    let image = blank_image(&dir, 16 * MIB);
    let image_path = image.to_str().unwrap();
    ok(&[image_path, "format"]);
    let before = digest(&image);

    let long = dir.join("long");
    fs::create_dir_all(&long).unwrap();
    fs::write(long.join("a-name-of-thirty-three-bytes.txt0"), b"x").unwrap();
    refused(
        &[image_path, "put", long.to_str().unwrap(), "x"],
        "not a kore name",
    );

    let wide = dir.join("wide");
    fs::create_dir_all(&wide).unwrap();
    for index in 0..129 {
        fs::write(wide.join(format!("{index}")), b"").unwrap();
    }
    refused(
        &[image_path, "put", wide.to_str().unwrap(), "x"],
        "at most 128",
    );

    #[cfg(unix)]
    {
        let linked = dir.join("linked");
        fs::create_dir_all(&linked).unwrap();
        std::os::unix::fs::symlink("/etc/hostname", linked.join("hostname")).unwrap();
        refused(&[image_path, "put", linked.to_str().unwrap(), "x"], "links");
    }

    let small = bundle(&dir, 4);
    refused(
        &[image_path, "put", small.to_str().unwrap(), "shared/x"],
        "store's grant",
    );
    refused(
        &[image_path, "put", small.to_str().unwrap(), "/abs"],
        "under the terminal's home",
    );
    assert_eq!(digest(&image), before);
}

/// A chain of `levels` directories under `root` with a file at the bottom:
/// a tree `levels + 1` deep, counting the destination as one.
fn chain(root: &Path, levels: usize) -> PathBuf {
    let mut at = root.to_path_buf();
    for level in 1..levels {
        at = at.join(format!("{level}"));
    }
    fs::create_dir_all(&at).unwrap();
    fs::write(at.join("bottom"), b"deepest").unwrap();
    root.to_path_buf()
}

#[test]
fn a_tree_as_deep_as_kore_allows_installs_and_one_deeper_is_refused_first() {
    let dir = scratch("depth");
    let image = blank_image(&dir, 16 * MIB);
    let image_path = image.to_str().unwrap();
    ok(&[image_path, "format"]);
    // The home is two levels below the volume's root (`work/face-0`), and
    // kore allows 32: thirty below the home, the destination among them.
    let deepest = chain(&dir.join("deepest"), 29);
    ok(&[image_path, "put", deepest.to_str().unwrap(), "d"]);
    let mut path = String::from("d");
    for level in 1..29 {
        path.push_str(&format!("/{level}"));
    }
    path.push_str("/bottom");
    assert_eq!(ok(&[image_path, "cat", &path]), "deepest");

    let before = digest(&image);
    let deeper = chain(&dir.join("deeper"), 30);
    refused(
        &[image_path, "put", deeper.to_str().unwrap(), "e"],
        "33 levels",
    );
    assert_eq!(digest(&image), before);
}

#[test]
fn an_install_that_fails_part_way_leaves_the_volume_as_it_was() {
    let dir = scratch("atomic");
    let image = blank_image(&dir, 16 * MIB);
    let image_path = image.to_str().unwrap();
    ok(&[image_path, "format"]);
    let small = bundle(&dir, 5);
    ok(&[image_path, "put", small.to_str().unwrap(), "kept"]);
    let home = ok(&[image_path, "ls"]);

    // Three files in name order; the second cannot fit, after the first
    // was written.
    let big = dir.join("big");
    fs::create_dir_all(&big).unwrap();
    fs::write(big.join("a.bin"), pattern(6, 200_000)).unwrap();
    File::create(big.join("b.bin"))
        .unwrap()
        .set_len(32 * MIB)
        .unwrap();
    fs::write(big.join("c.bin"), pattern(7, 10)).unwrap();
    refused(
        &[image_path, "put", big.to_str().unwrap(), "big"],
        "no contiguous room",
    );

    assert_eq!(ok(&[image_path, "ls"]), home);
    refused(&[image_path, "ls", "big"], "not found");
    assert!(ok(&[image_path, "ls", "kept"]).contains("index.html"));
}

#[test]
fn both_roots_name_the_install() {
    let dir = scratch("roots");
    let image = blank_image(&dir, 16 * MIB);
    let image_path = image.to_str().unwrap();
    ok(&[image_path, "format"]);
    let source = bundle(&dir, 8);
    let installed = ok(&[image_path, "put", source.to_str().unwrap(), "js/app"]);
    // Formatting published generation 1; the install, 2 and 3. The newest
    // root is generation 3's, in slot 3 & 1: block 1.
    assert!(installed.contains("generation 1 -> 3"), "{installed}");
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&image)
        .unwrap();
    let mut root = [0u8; 512];
    file.read_exact_at(&mut root, BLOCK).unwrap();
    root[200] ^= 0xff;
    file.write_all_at(&root, BLOCK).unwrap();
    drop(file);

    // Recovery falls back to generation 2, which names the install too.
    let run = kore_vol(&[image_path, "ls", "js/app"]);
    assert_eq!(run.code, 0, "{}", run.stderr);
    assert!(
        run.stderr.contains("fallback root (generation 2)"),
        "{}",
        run.stderr
    );
    assert!(run.stdout.contains("index.html"), "{}", run.stdout);
    // And kore would not write a volume it mounts from its fallback.
    refused(&[image_path, "rm", "js/app"], "fallback root");
}

#[test]
fn usage_errors_exit_two() {
    for args in [
        &[][..],
        &["disk.raw"][..],
        &["disk.raw", "put", "dist"][..],
        &["disk.raw", "ls", "a", "b"][..],
        &["--face", "10", "disk.raw", "ls"][..],
        &["--block-size", "1024", "disk.raw", "ls"][..],
        &["--replace", "disk.raw", "ls"][..],
    ] {
        assert_eq!(kore_vol(args).code, 2, "{args:?}");
    }
}
