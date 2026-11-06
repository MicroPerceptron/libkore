//! Host-only reader: one directory or one file of an offline kore volume
//! image, read through the real DAGFS provider and durable codec.
//!
//! ```text
//! volume_read IMAGE ls  PATH     # the names in a directory, one per line
//! volume_read IMAGE cat PATH     # a file's bytes, to stdout
//! ```
//!
//! `PATH` is from the volume's root (terminal N's home is `work/face-N`).
//! The image is a raw copy (`qemu-img convert -O raw`) of a disk no machine
//! is running: it is opened read-only, recovered from its newest root, and
//! never written. A volume that mounts only from its fallback root is
//! refused, as is anything that is not a kore volume. The screenshot
//! keytest reads the PNGs a guest saved with it (`make
//! keytest-screenshot-x86`); `kore-vol` (MicroPerceptron/kore#1391) is the
//! general tool, reading and writing a terminal's home.
use kore_driver::block::{
    BlockBuffer, BlockChannel, BlockError, BlockGeometry, BlockIoHints, CompletionMode,
    DeviceCommand, DeviceCompletion, DirectBlockDevice,
};
use kore_svc_dagfs::{
    Name, Rights, SessionId, block_on, persistence::PersistentFs, provider::BlockProvider,
};
use std::{
    collections::VecDeque,
    fs::File,
    io::Write,
    os::unix::fs::FileExt,
    path::Path,
};

const SECTOR: u64 = 512;

struct Buffer(Vec<u8>);
impl BlockBuffer for Buffer {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn bytes(&self) -> &[u8] {
        &self.0
    }
    fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

/// A raw image as a read-only block device: writes, flushes and discards
/// are refused, so nothing here can change the volume.
struct Image {
    file: File,
    geometry: BlockGeometry,
    ready: VecDeque<DeviceCompletion>,
}

impl Image {
    fn open(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(err)?;
        let metadata = file.metadata().map_err(err)?;
        if !metadata.is_file() || metadata.len() % SECTOR != 0 {
            return Err("a regular, sector-aligned raw image is required".into());
        }
        let geometry = BlockGeometry {
            block_size: SECTOR as u32,
            block_count: metadata.len() / SECTOR,
            max_transfer_blocks: 512,
            atomic_write_blocks: 1,
            flush: false,
            discard: false,
            read_only: true,
            io_hints: BlockIoHints::NONE,
        };
        geometry.validate().map_err(err)?;
        Ok(Self {
            file,
            geometry,
            ready: VecDeque::new(),
        })
    }
}

impl DirectBlockDevice for Image {
    type Buffer = Buffer;
    const INDEPENDENT_PROGRESS: bool = true;
    fn geometry(&self) -> BlockGeometry {
        self.geometry
    }
    fn allocate_buffer(&mut self, n: usize) -> Result<Buffer, BlockError> {
        if n > 256 * 1024 {
            return Err(BlockError::TooLarge);
        }
        Ok(Buffer(vec![0; n]))
    }
    fn submit(
        &mut self,
        command: DeviceCommand,
        buffer: Option<&mut Buffer>,
        cookie: u64,
    ) -> Result<(), BlockError> {
        if self.ready.len() >= 32 {
            return Err(BlockError::Backpressure);
        }
        let result = (|| {
            let DeviceCommand::Read {
                lba,
                blocks,
                buffer_offset,
            } = command
            else {
                return Err(BlockError::ReadOnly);
            };
            if lba
                .checked_add(u64::from(blocks))
                .is_none_or(|end| end > self.geometry.block_count)
            {
                return Err(BlockError::Range);
            }
            let len = blocks as usize * SECTOR as usize;
            let buffer = buffer.ok_or(BlockError::UnknownBuffer)?;
            let bytes = buffer
                .0
                .get_mut(buffer_offset..buffer_offset.checked_add(len).ok_or(BlockError::Range)?)
                .ok_or(BlockError::Range)?;
            self.file
                .read_exact_at(bytes, lba * SECTOR)
                .map_err(|_| BlockError::Media)
        })();
        self.ready.push_back(DeviceCompletion { cookie, result });
        Ok(())
    }
    fn poll(&mut self) -> Option<DeviceCompletion> {
        self.ready.pop_front()
    }
    fn quiesce(&mut self) -> Result<(), BlockError> {
        Ok(())
    }
}

fn err(e: impl std::fmt::Debug) -> String {
    format!("{e:?}")
}

type Volume = PersistentFs<Image, 8, 32>;

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let [_, image, verb, path] = args.as_slice() else {
        return Err("usage: volume_read IMAGE ls|cat PATH".into());
    };
    let components: Vec<&str> = path.split('/').filter(|c| !c.is_empty()).collect();
    let channel = BlockChannel::new(Image::open(Path::new(image))?, CompletionMode::Poll).map_err(err)?;
    let provider = BlockProvider::new(channel).map_err(err)?;
    let session = SessionId::new(0);
    let volume = block_on(Volume::recover(provider, session)).map_err(err)?;
    if volume.status().used_fallback_root {
        return Err("the volume mounts only from its fallback root".into());
    }
    let engine = volume.engine();
    let mut at = engine.open_root(session, Rights::READ).map_err(err)?.handle;
    for component in &components {
        let next = engine
            .open_at(session, at, &Name::new(component).map_err(err)?, Rights::READ)
            .map_err(|e| format!("{path}: {component}: {e:?}"))?
            .handle;
        engine.close(session, at).map_err(err)?;
        at = next;
    }
    let mut out = std::io::stdout().lock();
    match verb.as_str() {
        "ls" => {
            let mut cursor = Some(0);
            while let Some(from) = cursor {
                let page = engine.list(session, at, from).map_err(err)?;
                for entry in &page.entries {
                    writeln!(out, "{}", entry.name.as_str()).map_err(err)?;
                }
                cursor = page.next;
            }
        }
        "cat" => {
            let len = engine.stat(session, at).map_err(err)?.len;
            let mut buffer = vec![0; 64 * 1024];
            let mut offset = 0;
            while offset < len {
                let n = block_on(engine.read(session, at, offset, &mut buffer)).map_err(err)?;
                if n == 0 {
                    return Err(format!("{path}: short read at {offset} of {len}"));
                }
                out.write_all(&buffer[..n]).map_err(err)?;
                offset += n as u64;
            }
        }
        other => return Err(format!("unknown verb {other:?}: ls or cat")),
    }
    out.flush().map_err(err)?;
    engine.close(session, at).map_err(err)
}

fn main() {
    // Recovery's image scratch and the engine's tables are deep; the main
    // thread's stack is not.
    let result = std::thread::Builder::new()
        .stack_size(16 << 20)
        .spawn(run)
        .unwrap()
        .join()
        .unwrap();
    if let Err(e) = result {
        eprintln!("volume_read: {e}");
        std::process::exit(1);
    }
}
