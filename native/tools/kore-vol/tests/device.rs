//! `kore-vol` on a machine's disk, as its user runs it on Linux: the built
//! binary against loop devices over disk images made here. The partitions a
//! disk's table describes are listed as a kernel lists them: read from the
//! table by a kernel that reads GPT, else added as the kernel adds them
//! (BLKPG, by util-linux's `addpart`). What holds a disk in the field (a
//! mounted filesystem's claim, QEMU's record locks, a partitioner's BSD
//! lock) the test process holds here, while the binary runs as another.
//!
//! Needs root, loop devices and util-linux:
//!
//! ```text
//! sudo -E cargo test -p kore-vol --test device -- --ignored --test-threads=1
//! ```

#![cfg(target_os = "linux")]

mod common;

use std::{
    fs::{File, OpenOptions},
    os::{fd::OwnedFd, unix::fs::FileExt},
    path::{Path, PathBuf},
    process::Command,
    thread,
    time::{Duration, Instant},
};

use common::*;
use kore_driver::gpt::{KORE_FS_TYPE, KORE_SYSTEM_TYPE, synthetic};
use rustix::fs::{FlockOperation, Mode, OFlags, fcntl_lock, flock, open};

/// The disk the suite builds: a user partition, a gap, a system partition.
const DISK_BYTES: u64 = 96 * MIB;
const USER: (u64, u64) = (2048, 133_119);
const SYSTEM: (u64, u64) = (135_168, 176_127);

/// A disk image with a GPT holding a user and a system partition, its user
/// volume formatted as the machine's first boot would.
fn gpt_disk(dir: &Path) -> PathBuf {
    let image = blank_image(dir, DISK_BYTES);
    let mut mbr = [0u8; 512];
    mbr[510] = 0x55;
    mbr[511] = 0xaa;
    let (header, entries) = synthetic::table_pair(
        (KORE_FS_TYPE, USER.0, USER.1),
        Some((KORE_SYSTEM_TYPE, SYSTEM.0, SYSTEM.1)),
        DISK_BYTES / BLOCK - 34,
    );
    let file = OpenOptions::new().write(true).open(&image).unwrap();
    file.write_all_at(&mbr, 0).unwrap();
    file.write_all_at(&header, BLOCK).unwrap();
    file.write_all_at(&entries, 2 * BLOCK).unwrap();
    drop(file);
    ok(&[image.to_str().unwrap(), "format"]);
    image
}

/// Everything on the image outside the user partition.
fn outside_user(image: &Path) -> (Vec<u8>, Vec<u8>) {
    let tail = DISK_BYTES - (USER.1 + 1) * BLOCK;
    (
        read(image, 0, (USER.0 * BLOCK) as usize),
        read(image, (USER.1 + 1) * BLOCK, tail as usize),
    )
}

/// A loop device over a disk image, detached when dropped, listing the
/// partitions given (the module docs).
struct Disk {
    node: PathBuf,
}

impl Disk {
    fn attach(image: &Path, sector_size: u32, partitions: &[(u64, u64)]) -> Self {
        let mut losetup = Command::new("losetup");
        losetup.args([
            "--find",
            "--show",
            "--sector-size",
            &sector_size.to_string(),
        ]);
        if !partitions.is_empty() {
            losetup.arg("--partscan");
        }
        let output = losetup.arg(image).output().expect("losetup runs");
        assert!(
            output.status.success(),
            "losetup: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let disk = Self {
            node: PathBuf::from(String::from_utf8(output.stdout).unwrap().trim()),
        };
        for (index, &(first, last)) in partitions.iter().enumerate() {
            let node = disk.partition(index + 1);
            if !node.exists() {
                let added = Command::new("addpart")
                    .arg(&disk.node)
                    .arg((index + 1).to_string())
                    .arg(first.to_string())
                    .arg((last - first + 1).to_string())
                    .status()
                    .expect("addpart runs");
                assert!(added.success(), "addpart {}", node.display());
            }
            let sysfs = Path::new("/sys/class/block").join(node.file_name().unwrap());
            let number = |name: &str| {
                std::fs::read_to_string(sysfs.join(name))
                    .unwrap()
                    .trim()
                    .parse::<u64>()
                    .unwrap()
            };
            assert_eq!((number("start"), number("size")), (first, last - first + 1));
        }
        disk
    }

    fn partition(&self, number: usize) -> PathBuf {
        PathBuf::from(format!("{}p{number}", self.node.display()))
    }

    fn path(&self) -> &str {
        self.node.to_str().unwrap()
    }
}

impl Drop for Disk {
    fn drop(&mut self) {
        let _ = Command::new("losetup")
            .arg("--detach")
            .arg(&self.node)
            .status();
    }
}

/// An exclusive claim on `node`, as a mounted filesystem holds one.
fn mounted(node: &Path) -> OwnedFd {
    open(
        node,
        OFlags::RDONLY | OFlags::EXCL | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .unwrap()
}

/// A record lock on `node`, as QEMU holds the nodes it runs a machine on.
fn machine(node: &Path) -> File {
    let file = File::open(node).unwrap();
    fcntl_lock(&file, FlockOperation::NonBlockingLockShared).unwrap();
    file
}

/// The disk's BSD lock, as a partitioner holds it (exclusive) or udev
/// does around each event (shared).
fn bsd_lock(node: &Path, operation: FlockOperation) -> File {
    let file = File::open(node).unwrap();
    flock(&file, operation).unwrap();
    file
}

#[test]
#[ignore = "needs root, loop devices and util-linux: the module docs"]
fn a_disk_is_written_whole_inside_its_user_partition_alone() {
    let dir = scratch("device-whole");
    let image = gpt_disk(&dir);
    let before = outside_user(&image);
    let source = bundle(&dir, 11);
    {
        let disk = Disk::attach(&image, 512, &[]);
        let installed = ok(&[disk.path(), "put", source.to_str().unwrap(), "js/app"]);
        assert!(installed.contains("7 files, 3 directories"), "{installed}");
        assert!(installed.contains("read back identical"), "{installed}");
        let listed = kore_vol(&[disk.path(), "ls", "js/app/assets"]);
        assert_eq!((listed.code, listed.stderr.as_str()), (0, ""));
        assert!(listed.stdout.contains("index.js"), "{}", listed.stdout);
        refused(&[disk.path(), "format"], "never formats");
        refused(
            &["--block-size", "4096", disk.path(), "ls"],
            "logical blocks are 512 bytes",
        );
    }
    // What the next boot mounts: the disk read as an image.
    assert!(ok(&[image.to_str().unwrap(), "ls", "js/app"]).contains("index.html"));
    assert_eq!(outside_user(&image), before);
    assert_eq!(&read(&image, USER.0 * BLOCK, 8)[..], b"KOREROOT");
}

#[test]
#[ignore = "needs root, loop devices and util-linux: the module docs"]
fn a_disk_with_a_partition_in_use_is_written_through_its_user_partition() {
    let dir = scratch("device-partition");
    let image = gpt_disk(&dir);
    let before = outside_user(&image);
    let source = bundle(&dir, 12);
    let source = source.to_str().unwrap();
    let disk = Disk::attach(&image, 512, &[USER, SYSTEM]);
    let (user, system) = (disk.partition(1), disk.partition(2));
    let user_path = user.to_str().unwrap();

    // Naming the system partition names the disk, and the wrong partition.
    let elsewhere = format!("which is {user_path}");
    refused(
        &[system.to_str().unwrap(), "put", source, "never"],
        &elsewhere,
    );
    refused(&[system.to_str().unwrap(), "ls"], &elsewhere);

    {
        // Something mounted from the system partition: the disk cannot be
        // claimed whole, and the user partition is claimed alone.
        let _system = mounted(&system);
        let run = kore_vol(&[disk.path(), "put", source, "through"]);
        assert_eq!(run.code, 0, "{}", run.stderr);
        assert!(
            run.stderr
                .contains(&format!("writing its user volume through {user_path}")),
            "{}",
            run.stderr
        );
        // A machine run on the user partition is refused.
        let _machine = machine(&user);
        refused(&[disk.path(), "put", source, "never"], "in use");
    }
    {
        // The user partition itself in use: nothing to write through.
        let _user = mounted(&user);
        refused(&[disk.path(), "put", source, "never"], "refused to claim");
    }
    // The disk free: claimed whole, whichever of its nodes is named, and a
    // machine run on the user partition is refused all the same.
    ok(&[user_path, "put", source, "whole"]);
    {
        let _machine = machine(&user);
        refused(&[disk.path(), "put", source, "never"], "in use");
    }
    let home = ok(&[user_path, "ls"]);
    assert!(
        home.contains("through/") && home.contains("whole/") && !home.contains("never"),
        "{home}"
    );
    drop(disk);
    assert_eq!(outside_user(&image), before);
}

#[test]
#[ignore = "needs root, loop devices and util-linux: the module docs"]
fn a_disk_a_machine_or_a_partitioner_holds_is_refused() {
    let dir = scratch("device-held");
    let image = gpt_disk(&dir);
    let source = bundle(&dir, 13);
    let source = source.to_str().unwrap();
    let disk = Disk::attach(&image, 512, &[]);
    {
        let _machine = machine(&disk.node);
        refused(&[disk.path(), "put", source, "never"], "in use");
        // Read beside it, as an image QEMU runs is.
        ok(&[disk.path(), "ls"]);
    }
    {
        let _partitioner = bsd_lock(&disk.node, FlockOperation::LockExclusive);
        let started = Instant::now();
        refused(&[disk.path(), "put", source, "never"], "BSD lock");
        assert!(started.elapsed() >= Duration::from_secs(5));
    }
    {
        // udev's shared hold for one event is waited out.
        let udev = bsd_lock(&disk.node, FlockOperation::LockShared);
        let release = thread::spawn(move || {
            thread::sleep(Duration::from_millis(300));
            drop(udev);
        });
        ok(&[disk.path(), "put", source, "after"]);
        release.join().unwrap();
    }
    let home = ok(&[disk.path(), "ls"]);
    assert!(home.contains("after/") && !home.contains("never"), "{home}");
}

#[test]
#[ignore = "needs root, loop devices and util-linux: the module docs"]
fn a_disk_is_read_and_written_at_its_own_block_size() {
    let dir = scratch("device-4k");
    let image = blank_image(&dir, 32 * MIB);
    let image_path = image.to_str().unwrap();
    ok(&["--block-size", "4096", image_path, "format"]);
    let source = bundle(&dir, 14);
    {
        let disk = Disk::attach(&image, 4096, &[]);
        ok(&[disk.path(), "put", source.to_str().unwrap(), "app"]);
        assert!(ok(&["--block-size", "4096", disk.path(), "ls", "app"]).contains("index.html"));
        refused(
            &["--block-size", "512", disk.path(), "ls"],
            "logical blocks are 4096 bytes",
        );
    }
    assert!(ok(&["--block-size", "4096", image_path, "ls", "app"]).contains("index.html"));
}
