//! A machine's disk, from the host (Linux): which of its nodes the user
//! volume is read and written through, and the disk held while it is.
//!
//! The disk is attached to the host running the tool, with the machine that
//! boots it shut down: a live system on that machine, or the disk moved to
//! an enclosure. Its user volume is found as the filesystem row finds it at
//! boot, by `DurableService::locate` over the whole disk, so naming one of
//! its partitions names the disk and asserts that the volume is in that
//! partition.
//!
//! - **Reading** goes through the disk's node, narrowed as the boot narrows
//!   it, unbuffered and under a shared record lock.
//! - **Writing** first holds the disk as Linux asks of a program that
//!   writes one (systemd's "Locking Block Device Access"): an exclusive BSD
//!   lock on the whole disk's node, which udev takes shared before it probes
//!   the disk or any partition of it, and partitioning tools take before
//!   they write a table. A disk a machine is running on is refused: QEMU
//!   locks the nodes it runs, as it locks images. The tool then claims the
//!   whole disk (`O_EXCL`) and writes through the boot's own narrowed view,
//!   a claim the kernel grants only while nothing on the disk is mounted or
//!   claimed. A disk with something mounted from another partition (a Linux
//!   installed beside kore, an EFI system partition) cannot be claimed
//!   whole; the tool then claims the partition that holds the volume, alone,
//!   through its own node, a claim the kernel grants while nothing mounts or
//!   claims that partition.
//! - **Never formatted.** A device's blank volume is formatted by the
//!   machine's own first boot, and partitioning a disk is
//!   `tools/provision_fs.py`'s.

use std::{
    fs::{self, File},
    io,
    os::unix::fs::{FileTypeExt, MetadataExt},
    path::{Path, PathBuf},
    thread,
    time::{Duration, Instant},
};

use kore_driver::block::{
    DirectBlockDevice,
    image::{Access, ImageError, ImageFile},
};
use kore_svc_dagfs::durable::Located;
use rustix::{
    fs::{FlockOperation, fcntl_lock, flock, major, minor},
    process::{Flock, FlockOffsetType, FlockType, fcntl_getlk},
};

use crate::volume::{Found, VolError, classify};

/// The unit sysfs gives a partition's start and size in: 512-byte sectors,
/// whatever the disk's logical block size (`Documentation/ABI/stable/
/// sysfs-block`).
const SYSFS_SECTOR: u64 = 512;

/// Where the kernel names its block device nodes.
const DEV: &str = "/dev";

/// Where sysfs links each block device's number to its directory.
const SYS_DEV_BLOCK: &str = "/sys/dev/block";

/// How long the disk's BSD lock is waited for. udev holds it shared through
/// one run of its rules for each event on the disk, and closing a node that
/// was written raises one, so a tool started straight after another can meet
/// it held for a moment; a holder past this is a partitioner or another
/// writer, which is a conflict.
const HOLD_WAIT: Duration = Duration::from_secs(5);

/// Between attempts on the held lock.
const HOLD_RETRY: Duration = Duration::from_millis(20);

/// The disk held while a region of it is written (the module docs), until
/// the tool is done with the volume: bound before the volume, so that it is
/// dropped after it.
///
/// The volume is written through one node, claimed and under its exclusive
/// record lock. The hold keeps a shared record lock on the other node a
/// machine could be run on, the whole disk's or the volume's partition's,
/// which a QEMU started there meets and refuses. A record lock belongs to
/// the process and is dropped when *any* of its descriptors of that node
/// closes, so a guard is taken last, once nothing else of its node is open,
/// and nothing opens that node again while the hold lives.
pub(crate) struct DiskHold {
    /// The whole disk's node, under its BSD lock.
    disk: File,
    /// The volume's partition's node, guarded while the disk is written
    /// whole.
    partition: Option<File>,
}

impl DiskHold {
    /// Hold the disk at `node`: its BSD lock, and no machine running on it.
    fn take(node: &Path) -> Result<Self, VolError> {
        let disk = File::open(node).map_err(|error| io_error(node, error))?;
        let deadline = Instant::now() + HOLD_WAIT;
        loop {
            match flock(&disk, FlockOperation::NonBlockingLockExclusive) {
                Ok(()) => break,
                Err(rustix::io::Errno::WOULDBLOCK) if Instant::now() < deadline => {
                    thread::sleep(HOLD_RETRY);
                }
                Err(rustix::io::Errno::WOULDBLOCK) => {
                    return Err(VolError::Host(format!(
                        "{}: another program holds the disk (its BSD lock, which \
                         partitioning tools take) and kept it for {} s",
                        node.display(),
                        HOLD_WAIT.as_secs()
                    )));
                }
                Err(error) => return Err(io_error(node, error.into())),
            }
        }
        refuse_running(&disk, node)?;
        Ok(Self {
            disk,
            partition: None,
        })
    }

    /// While the disk is written whole: guard the volume's partition's own
    /// node, refusing a machine already running on it.
    fn guard_partition(&mut self, node: &Path) -> Result<(), VolError> {
        let partition = File::open(node).map_err(|error| io_error(node, error))?;
        refuse_running(&partition, node)?;
        share(&partition, node)?;
        self.partition = Some(partition);
        Ok(())
    }

    /// While the volume is written through its partition's node: guard the
    /// whole disk's.
    fn guard_disk(&self, node: &Path) -> Result<(), VolError> {
        share(&self.disk, node)
    }
}

/// Refuse a node another process holds a record lock on, as QEMU holds the
/// nodes it runs a machine on. Asked with `F_GETLK`, which reading the node
/// is enough for.
fn refuse_running(file: &File, node: &Path) -> Result<(), VolError> {
    let whole = Flock {
        start: 0,
        length: 0,
        pid: None,
        typ: FlockType::WriteLock,
        offset_type: FlockOffsetType::Set,
    };
    match fcntl_getlk(file, &whole) {
        Ok(None) => Ok(()),
        Ok(Some(_)) => Err(VolError::Device(node.to_path_buf(), ImageError::InUse)),
        Err(error) => Err(io_error(node, error.into())),
    }
}

/// A shared record lock on the whole of `file`, without waiting.
fn share(file: &File, node: &Path) -> Result<(), VolError> {
    match fcntl_lock(file, FlockOperation::NonBlockingLockShared) {
        Ok(()) => Ok(()),
        // POSIX lets a conflicting `F_SETLK` answer either.
        Err(rustix::io::Errno::AGAIN | rustix::io::Errno::ACCESS) => {
            Err(VolError::Device(node.to_path_buf(), ImageError::InUse))
        }
        Err(error) => Err(io_error(node, error.into())),
    }
}

fn io_error(node: &Path, error: io::Error) -> VolError {
    VolError::Device(node.to_path_buf(), ImageError::Io(error))
}

/// A partition of the disk, as sysfs gives it.
struct Partition {
    node: PathBuf,
    /// Its first sector and its length, in [`SYSFS_SECTOR`]s.
    start: u64,
    sectors: u64,
}

/// The disk a named node belongs to.
struct Disk {
    /// The whole disk's node.
    node: PathBuf,
    /// Its sysfs directory, which holds a directory for each partition.
    dir: PathBuf,
    /// The partition the node named, when it named one.
    named: Option<Partition>,
}

/// The disk the node at `path` belongs to, as sysfs describes it: itself,
/// when it is a whole disk.
fn disk_of(path: &Path) -> Result<Disk, VolError> {
    let meta = fs::metadata(path).map_err(|error| io_error(path, error))?;
    if !meta.file_type().is_block_device() {
        return Err(VolError::Device(path.to_path_buf(), ImageError::NotDevice));
    }
    let link =
        Path::new(SYS_DEV_BLOCK).join(format!("{}:{}", major(meta.rdev()), minor(meta.rdev())));
    let dir = fs::canonicalize(&link).map_err(|error| sysfs_error(&link, error))?;
    if !dir.join("partition").is_file() {
        return Ok(Disk {
            node: path.to_path_buf(),
            dir,
            named: None,
        });
    }
    let disk = dir
        .parent()
        .ok_or_else(|| VolError::Host(format!("{}: a partition of no disk", dir.display())))?
        .to_path_buf();
    Ok(Disk {
        node: node_of(&disk)?,
        dir: disk,
        named: Some(Partition {
            node: path.to_path_buf(),
            start: read_number(&dir.join("start"))?,
            sectors: read_number(&dir.join("size"))?,
        }),
    })
}

/// The node of the block device sysfs describes at `dir`: the name its
/// `uevent` gives, under [`DEV`], checked to be that device.
fn node_of(dir: &Path) -> Result<PathBuf, VolError> {
    let uevent = dir.join("uevent");
    let text = fs::read_to_string(&uevent).map_err(|error| sysfs_error(&uevent, error))?;
    let name = text
        .lines()
        .find_map(|line| line.strip_prefix("DEVNAME="))
        .ok_or_else(|| VolError::Host(format!("{}: no DEVNAME", uevent.display())))?;
    let node = Path::new(DEV).join(name);
    let numbers = dir.join("dev");
    let want = fs::read_to_string(&numbers).map_err(|error| sysfs_error(&numbers, error))?;
    let rdev = fs::metadata(&node)
        .map_err(|error| io_error(&node, error))?
        .rdev();
    if want.trim() != format!("{}:{}", major(rdev), minor(rdev)) {
        return Err(VolError::Host(format!(
            "{}: not the device sysfs names {} ({})",
            node.display(),
            dir.display(),
            want.trim()
        )));
    }
    Ok(node)
}

/// The node of the disk's partition at exactly `start` for `sectors`.
fn partition_at(disk: &Disk, start: u64, sectors: u64) -> Result<Option<PathBuf>, VolError> {
    let entries = fs::read_dir(&disk.dir).map_err(|error| sysfs_error(&disk.dir, error))?;
    for entry in entries {
        let dir = entry.map_err(|error| sysfs_error(&disk.dir, error))?.path();
        if !dir.join("partition").is_file() {
            continue;
        }
        if read_number(&dir.join("start"))? == start && read_number(&dir.join("size"))? == sectors {
            return node_of(&dir).map(Some);
        }
    }
    Ok(None)
}

fn sysfs_error(path: &Path, error: io::Error) -> VolError {
    VolError::Host(format!("{}: {error}", path.display()))
}

fn read_number(path: &Path) -> Result<u64, VolError> {
    let text = fs::read_to_string(path).map_err(|error| sysfs_error(path, error))?;
    text.trim().parse().map_err(|_| {
        VolError::Host(format!(
            "{}: not a number: {:?}",
            path.display(),
            text.trim()
        ))
    })
}

/// Open a node of the disk at the device's own logical block size, which a
/// block size the user named must agree with.
fn open(node: &Path, access: Access, block_size: Option<u32>) -> Result<ImageFile, VolError> {
    let device = ImageFile::open_device(node, access)
        .map_err(|error| VolError::Device(node.to_path_buf(), error))?;
    let own = device.geometry().block_size;
    match block_size {
        Some(asked) if asked != own => Err(VolError::Host(format!(
            "--block-size {asked}: {}'s logical blocks are {own} bytes",
            node.display()
        ))),
        _ => Ok(device),
    }
}

/// Where a volume to recover lies on the medium it was found through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Region {
    /// Its first block and its length, in the medium's blocks.
    first: u64,
    blocks: u64,
    /// Whether that is the whole medium.
    whole: bool,
    block_size: u32,
}

/// The region of a found volume; `None` for anything but a volume to
/// recover.
fn region(found: &Found) -> Result<Option<Region>, VolError> {
    let Located::Recover(provider) = found else {
        return Ok(None);
    };
    let block_size = provider.geometry().block_size;
    provider
        .control(|device| {
            let (first, blocks) = device.region();
            Region {
                first,
                blocks,
                whole: device.is_whole(),
                block_size,
            }
        })
        .map(Some)
        .map_err(|error| VolError::Host(format!("the volume's view: {error:?}")))
}

/// Find the user volume on the disk `path` names, as the boot does, and
/// open it to read or to write (the module docs). The hold, when writing,
/// must outlive the volume.
pub(crate) fn locate(
    path: &Path,
    access: Access,
    block_size: Option<u32>,
) -> Result<(Found, Option<DiskHold>), VolError> {
    let disk = disk_of(path)?;
    let hold = match access {
        Access::ReadWrite => Some(DiskHold::take(&disk.node)?),
        Access::ReadOnly => None,
    };
    // The boot's decision, over the whole disk.
    let found = classify(open(&disk.node, Access::ReadOnly, block_size)?)?;
    let Some(decided) = region(&found)? else {
        // Blank, or no kore volume: the caller says which.
        return Ok((found, hold));
    };
    let per = u64::from(decided.block_size) / SYSFS_SECTOR;
    let (start, sectors) = (decided.first * per, decided.blocks * per);
    let at = if decided.whole {
        Some(disk.node.clone())
    } else {
        partition_at(&disk, start, sectors)?
    };
    if let Some(named) = &disk.named
        && (decided.whole || named.start != start || named.sectors != sectors)
    {
        return Err(VolError::Host(format!(
            "{}: not where {}'s user volume is, which is {}",
            named.node.display(),
            disk.node.display(),
            describe(at.as_deref(), &disk, decided)
        )));
    }
    let Some(mut hold) = hold else {
        return Ok((found, None));
    };
    // The decision's descriptor, and its record lock, close before anything
    // is claimed.
    drop(found);
    let block_size = Some(decided.block_size);
    match open(&disk.node, Access::ReadWrite, block_size) {
        Ok(device) => {
            let found = classify(device)?;
            if region(&found)? != Some(decided) {
                return Err(changed(&disk.node));
            }
            // A machine run on the volume's own partition holds no lock on
            // the disk's node.
            if let Some(partition) = at.as_deref().filter(|node| *node != disk.node) {
                hold.guard_partition(partition)?;
            }
            return Ok((found, Some(hold)));
        }
        Err(VolError::Device(_, ImageError::Claimed)) if !decided.whole => {}
        Err(error) => return Err(error),
    }
    // Something on the disk is mounted or claimed: claim the volume's
    // partition alone.
    let node = at.ok_or_else(|| {
        VolError::Host(format!(
            "{}: in use (something on it is mounted or claimed), and the kernel lists no \
             node for {} to write through alone",
            disk.node.display(),
            describe(None, &disk, decided)
        ))
    })?;
    let found = classify(open(&node, Access::ReadWrite, block_size)?)?;
    // Through the partition's own node, the volume is the whole medium.
    let alone = Region {
        first: 0,
        whole: true,
        ..decided
    };
    if region(&found)? != Some(alone) {
        return Err(changed(&node));
    }
    // Decided again, with the partition claimed.
    let again = classify(open(&disk.node, Access::ReadOnly, block_size)?)?;
    if region(&again)? != Some(decided) {
        return Err(changed(&disk.node));
    }
    drop(again);
    hold.guard_disk(&disk.node)?;
    eprintln!(
        "kore-vol: {} is in use: writing its user volume through {}",
        disk.node.display(),
        node.display()
    );
    Ok((found, Some(hold)))
}

fn describe(node: Option<&Path>, disk: &Disk, region: Region) -> String {
    match node {
        Some(node) if node == disk.node => format!("the whole disk, {}", node.display()),
        Some(node) => node.display().to_string(),
        None => format!(
            "its partition at blocks {}..={} of {}",
            region.first,
            region.first + region.blocks - 1,
            disk.node.display()
        ),
    }
}

fn changed(node: &Path) -> VolError {
    VolError::Host(format!(
        "{}: the disk changed while kore-vol read it; nothing was written",
        node.display()
    ))
}
