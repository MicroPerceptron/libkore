//! What the suites share: the built binary run as its user runs it, the
//! scratch it works in, and bundles shaped like a built application.

#![allow(dead_code)]

use std::{
    fs::{self, File},
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    process::Command,
};

pub const MIB: u64 = 1 << 20;
pub const BLOCK: u64 = 512;

pub struct Run {
    pub code: i32,
    pub stdout: String,
    pub stderr: String,
}

pub fn kore_vol(args: &[&str]) -> Run {
    let output = Command::new(env!("CARGO_BIN_EXE_kore-vol"))
        .args(args)
        .output()
        .expect("kore-vol runs");
    Run {
        code: output.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

/// `kore-vol` with `args`, which must succeed; its standard output.
pub fn ok(args: &[&str]) -> String {
    let run = kore_vol(args);
    assert_eq!(
        run.code, 0,
        "kore-vol {args:?} failed: {}{}",
        run.stdout, run.stderr
    );
    run.stdout
}

/// `kore-vol` with `args`, which must refuse with a message naming `why`.
pub fn refused(args: &[&str], why: &str) {
    let run = kore_vol(args);
    assert_eq!(
        run.code, 1,
        "kore-vol {args:?} should refuse: {}",
        run.stdout
    );
    assert!(
        run.stderr.contains(why),
        "kore-vol {args:?}: expected {why:?} in {:?}",
        run.stderr
    );
}

/// A fresh directory of this test's own under cargo's per-target scratch.
pub fn scratch(test: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("kore-vol-{test}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// A zeroed (sparse) image: blank roots, as `qemu-img create` makes one.
pub fn blank_image(dir: &Path, bytes: u64) -> PathBuf {
    let path = dir.join("disk.raw");
    File::create(&path).unwrap().set_len(bytes).unwrap();
    path
}

/// Deterministic, position-dependent bytes: a misplaced or truncated chunk
/// cannot read back equal.
pub fn pattern(seed: u8, len: usize) -> Vec<u8> {
    (0..len)
        .map(|at| (at as u32).wrapping_mul(2_654_435_761).to_le_bytes()[1] ^ seed)
        .collect()
}

/// A built application's shape: a document, assets beside it, a nested
/// directory, an empty file, and files either side of the 64 KiB write
/// chunk.
pub fn bundle(dir: &Path, seed: u8) -> PathBuf {
    let root = dir.join(format!("dist-{seed}"));
    fs::create_dir_all(root.join("assets/fonts")).unwrap();
    fs::write(root.join("index.html"), pattern(seed, 431)).unwrap();
    fs::write(root.join("vite.svg"), pattern(seed ^ 1, 1)).unwrap();
    fs::write(root.join("assets/index.js"), pattern(seed ^ 2, 65_535)).unwrap();
    fs::write(root.join("assets/index.css"), pattern(seed ^ 3, 65_536)).unwrap();
    fs::write(root.join("assets/chunk.js"), pattern(seed ^ 4, 65_537)).unwrap();
    fs::write(
        root.join("assets/fonts/inter.woff2"),
        pattern(seed ^ 5, 300_001),
    )
    .unwrap();
    fs::write(root.join("assets/fonts/.keep"), b"").unwrap();
    root
}

pub fn read(path: &Path, at: u64, len: usize) -> Vec<u8> {
    let mut bytes = vec![0; len];
    File::open(path)
        .unwrap()
        .read_exact_at(&mut bytes, at)
        .unwrap();
    bytes
}

pub fn digest(path: &Path) -> Vec<u8> {
    let bytes = fs::read(path).unwrap();
    // Not a security digest: equality of two reads of one file.
    let mut sum = vec![0u8; 32];
    for (at, byte) in bytes.iter().enumerate() {
        let slot = &mut sum[at % 32];
        *slot = slot.wrapping_mul(31).wrapping_add(*byte);
    }
    sum.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    sum
}
