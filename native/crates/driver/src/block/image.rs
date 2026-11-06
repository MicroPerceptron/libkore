//! A disk's blocks, from the host, as a [`DirectBlockDevice`]: the medium a
//! host tool reads or writes a kore volume on while no machine is running
//! it. Either a regular file holding a disk's image (the disk a QEMU run
//! boots from, as raw bytes, or one copied off a machine) or, on Linux, a
//! machine's disk itself through a block device node.
//!
//! - **A file or a disk's node, presented and nothing more.**
//!   [`ImageFile::open`] takes a regular file, at the block size its caller
//!   names, since an image carries none of its own.
//!   [`ImageFile::open_device`] takes a block device node, a whole disk or
//!   one partition of it, at the logical block size the kernel reports for
//!   it. Which blocks of a disk may be written is its caller's to decide
//!   (`tools/kore-vol` asks the boot's own `locate`), and partitioning one is
//!   provisioning (`tools/provision_fs.py`), which nothing here does.
//! - **Never beside a running machine.** A mounted volume keeps its free
//!   extents and its next root slot in memory, so a second writer would
//!   allocate over blocks the first believes free, and payload blocks carry
//!   no checksum that would catch it. Both opens take a whole-file record
//!   lock, exclusive to write and shared to read, and refuse a file or node
//!   another process holds: QEMU locks the images and devices it runs, so
//!   the one lock answers both "is a machine running on this?" and "is
//!   another tool writing it?". The lock is a POSIX record lock, which
//!   belongs to the process and is dropped when *any* descriptor of the file
//!   closes: a process holds one `ImageFile` per file or node and closes no
//!   other descriptor of it while that one lives. A device opened to write
//!   is also claimed exclusively (`O_EXCL`), which Linux refuses while a
//!   filesystem on it, or on a partition of the same disk, is mounted, or
//!   another program has claimed one of them.
//! - **No cache between a device and its medium.** A device is read and
//!   written with `O_DIRECT`. Linux caches each node's blocks separately, so
//!   a disk and each of its partitions hold their own copies of the same
//!   blocks, and a buffered read through one node can answer, stale, what
//!   was written through another. Direct transfers take memory aligned to
//!   the logical block size, so every buffer begins on a [`BUFFER_ALIGN`]
//!   boundary (or on the block size, where that is larger) and a transfer
//!   begins a whole number of blocks into it, as the block channel already
//!   requires of its callers.
//! - **Synchronous.** A command runs to completion inside `submit` and its
//!   completion waits in a queue for `poll`: the asynchronous contract kept,
//!   with nothing ever in flight. A flush is `fdatasync`, since an image's
//!   length never changes and its metadata has nothing to make durable; on a
//!   device it also flushes the device's volatile write cache.

extern crate std;

use std::{
    collections::VecDeque,
    fs::{File, OpenOptions},
    io,
    os::unix::fs::FileExt,
    path::Path,
    vec,
    vec::Vec,
};

use super::{
    BlockBuffer, BlockError, BlockGeometry, BlockIoHints, DeviceCommand, DeviceCompletion,
    DirectBlockDevice,
};

/// Completions queued and not yet polled. A channel keeps no more commands
/// outstanding than its own bound (32 in the providers built on this), so
/// the queue is never the limit in practice; past it `submit` answers
/// [`BlockError::Backpressure`], as a full hardware queue would.
const QUEUE_DEPTH: usize = 64;

/// The largest single transfer, and so the largest buffer: 256 KiB. The
/// filesystem provider's own transfers are at most 64 KiB of payload and
/// its 128 KiB metadata images; anything larger is refused rather than
/// allocated.
const MAX_TRANSFER_BYTES: u32 = 256 * 1024;

/// Where every buffer begins: on a 4 KiB boundary, the largest logical
/// block kore formats a volume with (an NVMe namespace's 4 KiB format), and
/// so at least the alignment a direct transfer to such a device takes. A
/// device whose logical blocks are larger aligns its buffers to its block
/// size instead.
pub const BUFFER_ALIGN: usize = 4096;

/// How an image is opened: read-only images are opened, locked and
/// presented read-only all the way down.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    ReadOnly,
    ReadWrite,
}

/// Why an image could not be opened.
#[derive(Debug)]
pub enum ImageError {
    /// The operating system's answer, for a path that could not be opened,
    /// read or locked.
    Io(io::Error),
    /// The path names something other than a regular file
    /// ([`ImageFile::open`]).
    NotRegular,
    /// The path names something other than a block device
    /// ([`ImageFile::open_device`]).
    NotDevice,
    /// The kernel refused the exclusive claim a device is written under: a
    /// filesystem on it, or on a partition of the same disk, is mounted, or
    /// another program has claimed one of them.
    Claimed,
    /// The file's length is not a whole number of blocks.
    Unaligned { len: u64, block_size: u32 },
    /// The geometry the file would present is not a valid one.
    Geometry(BlockError),
    /// Another process holds the image: a machine running on it, or another
    /// tool.
    InUse,
}

impl From<io::Error> for ImageError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// A registered buffer: heap memory, since nothing here is DMA, whose first
/// byte sits on the alignment a direct transfer takes (the module docs).
/// The vector is never resized, so `start` stays where it was measured.
pub struct ImageBuffer {
    bytes: Vec<u8>,
    start: usize,
    len: usize,
}

impl ImageBuffer {
    /// `len` zeroed bytes beginning on an `align` boundary, a power of two.
    fn new(len: usize, align: usize) -> Self {
        let bytes = vec![0; len + align - 1];
        let at = bytes.as_ptr().addr();
        let start = at.next_multiple_of(align) - at;
        Self { bytes, start, len }
    }
}

impl BlockBuffer for ImageBuffer {
    fn len(&self) -> usize {
        self.len
    }
    fn bytes(&self) -> &[u8] {
        &self.bytes[self.start..self.start + self.len]
    }
    fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.bytes[self.start..self.start + self.len]
    }
}

pub struct ImageFile {
    file: File,
    geometry: BlockGeometry,
    ready: VecDeque<DeviceCompletion>,
}

impl ImageFile {
    /// Open, lock and size `path` as a device of `block_size`-byte logical
    /// blocks. The block size is the caller's: an image carries none of its
    /// own, and a kore volume refuses to recover under any but the one it
    /// was formatted with.
    pub fn open(path: &Path, access: Access, block_size: u32) -> Result<Self, ImageError> {
        let writable = access == Access::ReadWrite;
        let file = OpenOptions::new().read(true).write(writable).open(path)?;
        if !file.metadata()?.file_type().is_file() {
            return Err(ImageError::NotRegular);
        }
        lock(&file, writable)?;
        // Measured under the lock: nothing that honours it resizes the file
        // from here on.
        let len = file.metadata()?.len();
        Self::over(file, len, block_size, writable)
    }

    /// Open the block device node at `path` (a whole disk, or a partition
    /// of one), unbuffered, claimed exclusively when writing, locked as
    /// [`open`](Self::open) locks a file, and sized as the kernel reports
    /// it: its capacity and its logical block size, the size a kore volume
    /// on it was formatted with by the machine that first booted it.
    #[cfg(target_os = "linux")]
    pub fn open_device(path: &Path, access: Access) -> Result<Self, ImageError> {
        use rustix::fs::{Mode, OFlags};
        use std::{
            io::{Seek, SeekFrom},
            os::unix::fs::FileTypeExt,
        };
        // Refused before it is opened: what is not a device is neither
        // claimed nor opened unbuffered. Checked again on what was opened.
        if !std::fs::metadata(path)?.file_type().is_block_device() {
            return Err(ImageError::NotDevice);
        }
        let writable = access == Access::ReadWrite;
        let mode = if writable {
            OFlags::RDWR | OFlags::EXCL
        } else {
            OFlags::RDONLY
        };
        let file =
            match rustix::fs::open(path, mode | OFlags::DIRECT | OFlags::CLOEXEC, Mode::empty()) {
                Ok(fd) => File::from(fd),
                Err(rustix::io::Errno::BUSY) => return Err(ImageError::Claimed),
                Err(error) => return Err(ImageError::Io(error.into())),
            };
        if !file.metadata()?.file_type().is_block_device() {
            return Err(ImageError::NotDevice);
        }
        lock(&file, writable)?;
        let block_size =
            rustix::fs::ioctl_blksszget(&file).map_err(|error| ImageError::Io(error.into()))?;
        // A device's end is its capacity.
        let len = (&file).seek(SeekFrom::End(0))?;
        Self::over(file, len, block_size, writable)
    }

    /// The device over an opened, locked file of `len` bytes.
    fn over(file: File, len: u64, block_size: u32, writable: bool) -> Result<Self, ImageError> {
        if block_size == 0 || !block_size.is_power_of_two() || len % u64::from(block_size) != 0 {
            return Err(ImageError::Unaligned { len, block_size });
        }
        let geometry = BlockGeometry {
            block_size,
            block_count: len / u64::from(block_size),
            max_transfer_blocks: MAX_TRANSFER_BYTES / block_size,
            atomic_write_blocks: 1,
            flush: true,
            discard: false,
            read_only: !writable,
            io_hints: BlockIoHints::NONE,
        }
        .validate()
        .map_err(ImageError::Geometry)?;
        Ok(Self {
            file,
            geometry,
            ready: VecDeque::with_capacity(QUEUE_DEPTH),
        })
    }

    /// Run one command against the file, synchronously.
    fn execute(
        &self,
        command: DeviceCommand,
        buffer: Option<&mut ImageBuffer>,
    ) -> Result<(), BlockError> {
        let (lba, blocks, offset, write) = match command {
            DeviceCommand::Flush => return self.file.sync_data().map_err(|_| BlockError::Media),
            DeviceCommand::Discard { .. } => return Err(BlockError::Unsupported),
            DeviceCommand::Read {
                lba,
                blocks,
                buffer_offset,
            } => (lba, blocks, buffer_offset, false),
            DeviceCommand::Write {
                lba,
                blocks,
                buffer_offset,
            } => (lba, blocks, buffer_offset, true),
        };
        if write && self.geometry.read_only {
            return Err(BlockError::ReadOnly);
        }
        if blocks > self.geometry.max_transfer_blocks {
            return Err(BlockError::TooLarge);
        }
        if lba
            .checked_add(u64::from(blocks))
            .is_none_or(|end| end > self.geometry.block_count)
        {
            return Err(BlockError::Range);
        }
        // A transfer begins a whole number of blocks into its buffer, so on
        // a block boundary in memory, which a direct transfer needs.
        if !offset.is_multiple_of(self.geometry.block_size as usize) {
            return Err(BlockError::Alignment);
        }
        // Bounded by the transfer ceiling above, so neither product overflows.
        let len = blocks as usize * self.geometry.block_size as usize;
        let at = lba * u64::from(self.geometry.block_size);
        let buffer = buffer.ok_or(BlockError::UnknownBuffer)?;
        let bytes = offset
            .checked_add(len)
            .and_then(|end| buffer.bytes_mut().get_mut(offset..end))
            .ok_or(BlockError::Range)?;
        if write {
            self.file.write_all_at(bytes, at)
        } else {
            self.file.read_exact_at(bytes, at)
        }
        .map_err(|_| BlockError::Media)
    }
}

/// Take the whole-file record lock, without waiting for it.
fn lock(file: &File, exclusive: bool) -> Result<(), ImageError> {
    use rustix::fs::{FlockOperation, fcntl_lock};
    let operation = if exclusive {
        FlockOperation::NonBlockingLockExclusive
    } else {
        FlockOperation::NonBlockingLockShared
    };
    match fcntl_lock(file, operation) {
        Ok(()) => Ok(()),
        // POSIX lets a conflicting `F_SETLK` answer either.
        Err(rustix::io::Errno::AGAIN | rustix::io::Errno::ACCESS) => Err(ImageError::InUse),
        Err(error) => Err(ImageError::Io(error.into())),
    }
}

impl DirectBlockDevice for ImageFile {
    type Buffer = ImageBuffer;
    const INDEPENDENT_PROGRESS: bool = true;

    fn geometry(&self) -> BlockGeometry {
        self.geometry
    }
    fn allocate_buffer(&mut self, bytes: usize) -> Result<ImageBuffer, BlockError> {
        if bytes > self.max_buffer_bytes() {
            return Err(BlockError::TooLarge);
        }
        let align = BUFFER_ALIGN.max(self.geometry.block_size as usize);
        Ok(ImageBuffer::new(bytes, align))
    }
    fn submit(
        &mut self,
        command: DeviceCommand,
        buffer: Option<&mut ImageBuffer>,
        cookie: u64,
    ) -> Result<(), BlockError> {
        if self.ready.len() >= QUEUE_DEPTH {
            return Err(BlockError::Backpressure);
        }
        let result = self.execute(command, buffer);
        self.ready.push_back(DeviceCompletion { cookie, result });
        Ok(())
    }
    fn poll(&mut self) -> Option<DeviceCompletion> {
        self.ready.pop_front()
    }
    fn quiesce(&mut self) -> Result<(), BlockError> {
        if self.geometry.read_only {
            return Ok(());
        }
        self.file.sync_data().map_err(|_| BlockError::Media)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        path::PathBuf,
        string::{String, ToString},
    };

    /// A scratch path of this test's own beside the test binary's.
    fn scratch(name: &str, bytes: Option<u64>) -> PathBuf {
        let path = std::env::temp_dir().join(std::format!(
            "kore-driver-image-{}-{name}",
            std::process::id()
        ));
        let _ = fs::remove_file(&path);
        if let Some(bytes) = bytes {
            File::create(&path).unwrap().set_len(bytes).unwrap();
        }
        path
    }

    fn run(
        image: &mut ImageFile,
        command: DeviceCommand,
        buffer: Option<&mut ImageBuffer>,
    ) -> Result<(), BlockError> {
        image.submit(command, buffer, 7)?;
        let completion = image.poll().expect("a submitted command completes at once");
        assert_eq!(completion.cookie, 7);
        assert!(image.poll().is_none());
        completion.result
    }

    #[test]
    fn a_written_block_reads_back_at_its_offset() {
        let path = scratch("round-trip", Some(64 * 512));
        let mut image = ImageFile::open(&path, Access::ReadWrite, 512).unwrap();
        assert_eq!(image.geometry().block_count, 64);
        let mut buffer = image.allocate_buffer(3 * 512).unwrap();
        for (at, byte) in buffer.bytes_mut().iter_mut().enumerate() {
            *byte = (at % 251) as u8;
        }
        let write = DeviceCommand::Write {
            lba: 5,
            blocks: 2,
            buffer_offset: 512,
        };
        run(&mut image, write, Some(&mut buffer)).unwrap();
        run(&mut image, DeviceCommand::Flush, None).unwrap();
        let on_disk = fs::read(&path).unwrap();
        assert_eq!(&on_disk[5 * 512..7 * 512], &buffer.bytes()[512..]);
        assert!(on_disk[..5 * 512].iter().all(|&b| b == 0));
        let mut back = image.allocate_buffer(2 * 512).unwrap();
        let read = DeviceCommand::Read {
            lba: 5,
            blocks: 2,
            buffer_offset: 0,
        };
        run(&mut image, read, Some(&mut back)).unwrap();
        assert_eq!(back.bytes(), &buffer.bytes()[512..]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn what_leaves_the_image_or_the_buffer_is_refused() {
        let path = scratch("bounds", Some(8 * 512));
        let mut image = ImageFile::open(&path, Access::ReadWrite, 512).unwrap();
        let mut buffer = image.allocate_buffer(512).unwrap();
        let past_end = DeviceCommand::Read {
            lba: 8,
            blocks: 1,
            buffer_offset: 0,
        };
        assert_eq!(
            run(&mut image, past_end, Some(&mut buffer)),
            Err(BlockError::Range)
        );
        let past_buffer = DeviceCommand::Read {
            lba: 0,
            blocks: 2,
            buffer_offset: 0,
        };
        assert_eq!(
            run(&mut image, past_buffer, Some(&mut buffer)),
            Err(BlockError::Range)
        );
        let no_buffer = DeviceCommand::Write {
            lba: 0,
            blocks: 1,
            buffer_offset: 0,
        };
        assert_eq!(
            run(&mut image, no_buffer, None),
            Err(BlockError::UnknownBuffer)
        );
        let discard = DeviceCommand::Discard { lba: 0, blocks: 1 };
        assert_eq!(run(&mut image, discard, None), Err(BlockError::Unsupported));
        assert_eq!(
            image.allocate_buffer(MAX_TRANSFER_BYTES as usize + 1).err(),
            Some(BlockError::TooLarge)
        );
        drop(image);

        let mut read_only = ImageFile::open(&path, Access::ReadOnly, 512).unwrap();
        assert!(read_only.geometry().read_only);
        let mut buffer = read_only.allocate_buffer(512).unwrap();
        let write = DeviceCommand::Write {
            lba: 0,
            blocks: 1,
            buffer_offset: 0,
        };
        assert_eq!(
            run(&mut read_only, write, Some(&mut buffer)),
            Err(BlockError::ReadOnly)
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn only_a_regular_whole_number_of_blocks_opens() {
        let directory = std::env::temp_dir();
        assert!(matches!(
            ImageFile::open(&directory, Access::ReadOnly, 512),
            Err(ImageError::NotRegular)
        ));
        let ragged = scratch("ragged", Some(4096 + 100));
        assert!(matches!(
            ImageFile::open(&ragged, Access::ReadOnly, 512),
            Err(ImageError::Unaligned {
                len: 4196,
                block_size: 512
            })
        ));
        // 4 KiB logical blocks: a whole number of them, or not.
        let sectors = scratch("sectors", Some(3 * 512));
        assert!(ImageFile::open(&sectors, Access::ReadOnly, 512).is_ok());
        assert!(matches!(
            ImageFile::open(&sectors, Access::ReadOnly, 4096),
            Err(ImageError::Unaligned { .. })
        ));
        let missing = scratch("missing", None);
        assert!(matches!(
            ImageFile::open(&missing, Access::ReadOnly, 512),
            Err(ImageError::Io(_))
        ));
        let _ = fs::remove_file(ragged);
        let _ = fs::remove_file(sectors);
    }

    #[test]
    fn every_buffer_begins_where_a_direct_transfer_can_reach_it() {
        for block_size in [512, 4096] {
            let path = scratch("aligned", Some(64 * 4096));
            let mut image = ImageFile::open(&path, Access::ReadOnly, block_size).unwrap();
            for bytes in [0, 1, 511, 512, 4096, 65_536, MAX_TRANSFER_BYTES as usize] {
                let buffer = image.allocate_buffer(bytes).unwrap();
                assert_eq!(buffer.len(), bytes);
                assert_eq!(buffer.bytes().len(), bytes);
                assert_eq!(buffer.bytes().as_ptr().addr() % BUFFER_ALIGN, 0);
                assert!(buffer.bytes().iter().all(|&byte| byte == 0));
            }
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn a_transfer_begins_on_a_block_boundary_in_its_buffer() {
        let path = scratch("boundary", Some(8 * 512));
        let mut image = ImageFile::open(&path, Access::ReadWrite, 512).unwrap();
        let mut buffer = image.allocate_buffer(3 * 512).unwrap();
        let read = DeviceCommand::Read {
            lba: 0,
            blocks: 1,
            buffer_offset: 100,
        };
        assert_eq!(
            run(&mut image, read, Some(&mut buffer)),
            Err(BlockError::Alignment)
        );
        fs::remove_file(path).unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn only_a_block_device_opens_as_one() {
        let file = scratch("not-a-device", Some(8 * 512));
        for access in [Access::ReadOnly, Access::ReadWrite] {
            assert!(matches!(
                ImageFile::open_device(&file, access),
                Err(ImageError::NotDevice)
            ));
            assert!(matches!(
                ImageFile::open_device(&std::env::temp_dir(), access),
                Err(ImageError::NotDevice)
            ));
        }
        let missing = scratch("missing-device", None);
        assert!(matches!(
            ImageFile::open_device(&missing, Access::ReadOnly),
            Err(ImageError::Io(_))
        ));
        fs::remove_file(file).unwrap();
    }

    /// A loop device over a scratch file, detached when dropped.
    #[cfg(target_os = "linux")]
    struct Loop(PathBuf);

    #[cfg(target_os = "linux")]
    impl Loop {
        fn attach(file: &Path, sector_size: u32) -> Self {
            let output = std::process::Command::new("losetup")
                .args(["--find", "--show", "--sector-size"])
                .arg(sector_size.to_string())
                .arg(file)
                .output()
                .expect("losetup runs");
            assert!(
                output.status.success(),
                "losetup: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let node = String::from_utf8(output.stdout).unwrap();
            Self(PathBuf::from(node.trim()))
        }
    }

    #[cfg(target_os = "linux")]
    impl Drop for Loop {
        fn drop(&mut self) {
            let _ = std::process::Command::new("losetup")
                .arg("--detach")
                .arg(&self.0)
                .status();
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "needs root and loop devices: sudo -E cargo test -p kore-driver --features image -- --ignored"]
    fn a_device_is_read_and_written_unbuffered_at_its_own_block_size() {
        for sector_size in [512, 4096] {
            let file = scratch("loop", Some(64 * 4096));
            let device = Loop::attach(&file, sector_size);
            let mut disk = ImageFile::open_device(&device.0, Access::ReadWrite).unwrap();
            let geometry = disk.geometry();
            assert_eq!(geometry.block_size, sector_size);
            assert_eq!(geometry.block_count, 64 * 4096 / u64::from(sector_size));
            let block = sector_size as usize;
            let mut buffer = disk.allocate_buffer(3 * block).unwrap();
            for (at, byte) in buffer.bytes_mut().iter_mut().enumerate() {
                *byte = (at % 251) as u8 ^ 0x5a;
            }
            let write = DeviceCommand::Write {
                lba: 5,
                blocks: 2,
                buffer_offset: block,
            };
            run(&mut disk, write, Some(&mut buffer)).unwrap();
            run(&mut disk, DeviceCommand::Flush, None).unwrap();
            // Through the medium, not a cache: the file behind it holds it.
            let behind = fs::read(&file).unwrap();
            assert_eq!(&behind[5 * block..7 * block], &buffer.bytes()[block..]);
            let mut back = disk.allocate_buffer(2 * block).unwrap();
            let read = DeviceCommand::Read {
                lba: 5,
                blocks: 2,
                buffer_offset: 0,
            };
            run(&mut disk, read, Some(&mut back)).unwrap();
            assert_eq!(back.bytes(), &buffer.bytes()[block..]);

            // Claimed while it is written: a second writer is refused by the
            // kernel; a reader is not.
            assert!(matches!(
                ImageFile::open_device(&device.0, Access::ReadWrite),
                Err(ImageError::Claimed)
            ));
            drop(disk);
            let reader = ImageFile::open_device(&device.0, Access::ReadOnly).unwrap();
            assert!(reader.geometry().read_only);
            drop(reader);
            drop(device);
            fs::remove_file(file).unwrap();
        }
    }
}
