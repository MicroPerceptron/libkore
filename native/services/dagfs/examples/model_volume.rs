//! Host-only builder/verifier. Uses the real DAGFS provider and durable codec
//! over the image-file device (`kore_driver::block::image`), which never
//! opens a block device; destination creation is exclusive.
use kore_driver::block::{
    BlockChannel, CompletionMode,
    image::{Access, ImageFile},
};
use kore_svc_dagfs::{
    ANY_REVISION, Name, ObjectKind, Rights, SYSTEM_WORKSPACE_DIR, SessionId, block_on,
    face_home_label, persistence::PersistentFs, provider::BlockProvider,
};
use std::{
    fs::{File, OpenOptions},
    io::Read,
    path::Path,
};

/// The logical block size of the disks this example's volumes are spliced
/// into: QEMU's.
const BLOCK_SIZE: u32 = 512;

fn err(e: impl std::fmt::Debug) -> String {
    format!("{e:?}")
}
type Volume = PersistentFs<ImageFile, 8, 32>;
fn provider(path: &Path, writable: bool) -> Result<BlockProvider<ImageFile>, String> {
    let access = if writable {
        Access::ReadWrite
    } else {
        Access::ReadOnly
    };
    let image = ImageFile::open(path, access, BLOCK_SIZE).map_err(err)?;
    BlockProvider::new(BlockChannel::new(image, CompletionMode::Poll).map_err(err)?).map_err(err)
}
/// `work/face-0/models/<member>`: terminal 0's home, where the model a
/// keytest's terminal opens lives.
fn components() -> [String; 4] {
    let member = match std::env::var("MODEL_VOLUME_MEMBER").as_deref() {
        Ok("witness.model") => "witness.model",
        Ok("lfm25.model") | Err(_) => "lfm25.model",
        _ => panic!("unsupported model member"),
    };
    let face = face_home_label(0).expect("terminal 0 has a home");
    [
        SYSTEM_WORKSPACE_DIR.into(),
        String::from_utf8_lossy(&face).into_owned(),
        "models".into(),
        member.into(),
    ]
}
fn build(image: &Path, source: &Path, bytes: u64) -> Result<(), String> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(image)
        .map_err(err)?;
    if bytes < 1024 * 1024 || !bytes.is_multiple_of(u64::from(BLOCK_SIZE)) {
        return Err("invalid volume size".into());
    }
    file.set_len(bytes).map_err(err)?;
    drop(file);
    let session = SessionId::new(0);
    let mut volume = block_on(Volume::format(provider(image, true)?, session)).map_err(err)?;
    let engine = volume.engine();
    let mut at = engine.open_root(session, Rights::ALL).map_err(err)?.handle;
    for component in &components()[..3] {
        let next = block_on(engine.create(
            session,
            at,
            &Name::new(component).map_err(err)?,
            ObjectKind::Directory,
        ))
        .map_err(err)?
        .handle;
        engine.close(session, at).map_err(err)?;
        at = next;
    }
    let handle = block_on(engine.create(
        session,
        at,
        &Name::new(&components()[3]).map_err(err)?,
        ObjectKind::File,
    ))
    .map_err(err)?
    .handle;
    let mut source = File::open(source).map_err(err)?;
    let len = source.metadata().map_err(err)?.len();
    block_on(engine.resize(session, handle, len, ANY_REVISION)).map_err(err)?;
    let mut buffer = vec![0; 64 * 1024];
    let mut offset = 0;
    while offset < len {
        let want = buffer.len().min((len - offset) as usize);
        source.read_exact(&mut buffer[..want]).map_err(err)?;
        block_on(engine.write_at(session, handle, offset, &buffer[..want], ANY_REVISION))
            .map_err(err)?;
        offset += want as u64;
        if offset % (256 << 20) == 0 {
            eprintln!("imported {offset}/{len} bytes");
        }
    }
    engine.close(session, handle).map_err(err)?;
    engine.close(session, at).map_err(err)?;
    // Both roots must name the imported file, rather than a blank fallback.
    block_on(volume.commit(session)).map_err(err)?;
    block_on(volume.commit(session)).map_err(err)?;
    Ok(())
}
fn verify(image: &Path, source: &Path) -> Result<(), String> {
    let session = SessionId::new(0);
    let volume = block_on(Volume::recover(provider(image, false)?, session)).map_err(err)?;
    if volume.status().used_fallback_root {
        return Err("unexpected fallback root".into());
    }
    let engine = volume.engine();
    let mut at = engine.open_root(session, Rights::READ).map_err(err)?.handle;
    for component in components() {
        let next = engine
            .open_at(
                session,
                at,
                &Name::new(&component).map_err(err)?,
                Rights::READ,
            )
            .map_err(err)?
            .handle;
        engine.close(session, at).map_err(err)?;
        at = next;
    }
    let mut source = File::open(source).map_err(err)?;
    let len = source.metadata().map_err(err)?.len();
    if engine.stat(session, at).map_err(err)?.len != len {
        return Err("length mismatch".into());
    }
    let mut expected = vec![0; 64 * 1024];
    let mut actual = vec![0; 64 * 1024];
    let mut offset = 0;
    let mut digest = blake3::Hasher::new();
    while offset < len {
        let n = block_on(engine.read(session, at, offset, &mut actual)).map_err(err)?;
        if n == 0 {
            return Err("short model read".into());
        }
        source.read_exact(&mut expected[..n]).map_err(err)?;
        if actual[..n] != expected[..n] {
            return Err(format!("model mismatch at {offset}"));
        }
        digest.update(&actual[..n]);
        offset += n as u64;
    }
    println!(
        "verified: path={} bytes={len} generation={} blake3={}",
        components().join("/"),
        volume.status().generation,
        digest.finalize()
    );
    Ok(())
}
fn probe_write(image: &Path) -> Result<(), String> {
    let session = SessionId::new(0);
    let mut volume = block_on(Volume::recover(provider(image, true)?, session)).map_err(err)?;
    eprintln!("recovered: {:?}", volume.status());
    let engine = volume.engine();
    let root = engine.open_root(session, Rights::ALL).map_err(err)?.handle;
    let file = block_on(engine.create(
        session,
        root,
        &Name::new("write-probe").map_err(err)?,
        ObjectKind::File,
    ))
    .map_err(err)?;
    engine.close(session, file.handle).map_err(err)?;
    engine.close(session, root).map_err(err)?;
    block_on(volume.commit(session)).map_err(err)?;
    println!("cold recovery write: PASS");
    Ok(())
}
fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 4 {
        return Err("model_volume build|verify|probe-write IMAGE SOURCE [VOLUME_BYTES]".into());
    }
    let image = Path::new(&args[2]);
    let source = Path::new(&args[3]);
    match args[1].as_str() {
        "build" if args.len() == 5 => {
            build(image, source, args[4].parse().map_err(err)?)?;
            verify(image, source)
        }
        "verify" if args.len() == 4 => verify(image, source),
        "probe-write" if args.len() == 4 => probe_write(image),
        _ => Err("invalid arguments".into()),
    }
}
fn main() {
    let result = std::thread::Builder::new()
        .stack_size(16 << 20)
        .spawn(run)
        .unwrap()
        .join()
        .unwrap();
    if let Err(e) = result {
        eprintln!("model_volume: {e}");
        std::process::exit(1);
    }
}
