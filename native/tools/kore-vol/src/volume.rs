//! The volume side: found, mounted, changed and read back as the filesystem
//! row would.

use std::{
    fmt,
    fs::{self, File},
    io::{self, Read, Write},
    os::unix::fs::FileTypeExt,
    path::{Path, PathBuf},
};

use kore_driver::block::{
    CompletionMode, RegionDevice,
    image::{Access, ImageError, ImageFile},
};
use kore_svc_dagfs::{
    ANY_REVISION, DurableService, FsError, Handle, Name, ObjectKind, OpenedObject, Rights,
    SYSTEM_APPS_VIEW, SYSTEM_LIBS_VIEW, SYSTEM_SHARE_VIEW, SYSTEM_WORKSPACE_DIR, SessionId,
    block_on,
    bounds::{MAX_OBJECTS, MAX_PATH_DEPTH},
    durable::{Located, note},
    face_home_label,
    persistence::{MountMode, PersistenceError, PersistentFs},
};

use crate::{
    on_volume_stack,
    source::{self, Entry},
};

/// The provider's buffer and request bounds: the filesystem row's own
/// (`Located<SharedLane, 8, 32>` in `durable.rs`), so the volume is driven
/// as the row drives it.
const BUFFERS: usize = 8;
const OUTSTANDING: usize = 32;

type Volume = PersistentFs<RegionDevice<ImageFile>, BUFFERS, OUTSTANDING>;
pub(crate) type Found = Located<ImageFile, BUFFERS, OUTSTANDING>;

/// What keeps a machine's disk held while its volume is written
/// (`device::DiskHold`); nothing holds an image beyond its own lock.
#[cfg(target_os = "linux")]
type Hold = crate::device::DiskHold;
#[cfg(not(target_os = "linux"))]
type Hold = std::convert::Infallible;

/// The session the tool acts in: the row's own boot session, rooted at the
/// persisted root.
const SESSION: SessionId = SessionId::new(0);

/// One payload transfer: the provider's largest single write.
const CHUNK: usize = 64 * 1024;

pub enum VolError {
    /// The host side: a source that cannot be read, a usage the volume
    /// cannot honour.
    Host(String),
    Image(ImageError),
    /// A node of a machine's disk, and what opening it met.
    Device(PathBuf, ImageError),
    Blank,
    NoVolume(u64),
    Degraded {
        generation: u64,
    },
    Persistence(&'static str, PersistenceError),
    Fs(String, FsError),
    Exists(String),
    NotFound(String),
    Verify(String),
    /// The caller's sink refused a file's bytes ([`Mount::read_to`]): the
    /// volume is untouched, the output short.
    Output(io::Error),
}

impl fmt::Display for VolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Host(message) => f.write_str(message),
            Self::Image(ImageError::InUse) => f.write_str(
                "the image is in use: a machine is running on it, or another tool holds it",
            ),
            Self::Image(ImageError::NotRegular | ImageError::NotDevice) => f.write_str(
                "neither a disk image nor a disk: kore-vol takes a regular file, or on Linux \
                 a disk's block device",
            ),
            Self::Image(ImageError::Claimed) => f.write_str("the image is claimed"),
            Self::Image(ImageError::Unaligned { len, block_size }) => write!(
                f,
                "{len} bytes is not a whole number of {block_size}-byte blocks"
            ),
            Self::Image(ImageError::Geometry(error)) => write!(f, "image geometry: {error:?}"),
            Self::Image(ImageError::Io(error)) => write!(f, "image: {error}"),
            Self::Device(node, error) => {
                write!(f, "{}: ", node.display())?;
                match error {
                    ImageError::InUse => {
                        f.write_str("in use: a machine is running on it, or another tool holds it")
                    }
                    ImageError::Claimed => f.write_str(
                        "the kernel refused to claim it exclusively: something on it, or on \
                         another partition of the same disk, is mounted, or another program \
                         has claimed one of them",
                    ),
                    ImageError::NotDevice | ImageError::NotRegular => {
                        f.write_str("not a block device")
                    }
                    ImageError::Unaligned { len, block_size } => write!(
                        f,
                        "{len} bytes is not a whole number of {block_size}-byte blocks"
                    ),
                    ImageError::Geometry(error) => write!(f, "geometry: {error:?}"),
                    ImageError::Io(error) => write!(f, "{error}"),
                }
            }
            Self::Blank => f.write_str(
                "the volume is blank: the machine formats it at its first boot (and on an \
                 image, `kore-vol <image> format` formats it now as that boot would)",
            ),
            Self::NoVolume(note::VOLATILE_UNDESIGNATED) => f.write_str(
                "no kore volume: the disk is not kore's, and its partition table designates \
                 no kore filesystem partition",
            ),
            Self::NoVolume(_) => f.write_str(
                "no kore volume: the designated kore filesystem partition holds foreign data, \
                 or is too small for one",
            ),
            Self::Degraded { generation } => write!(
                f,
                "the volume mounts only from its fallback root (generation {generation}), \
                 so kore keeps it read-only, and so does kore-vol"
            ),
            Self::Persistence(stage, PersistenceError::Incompatible) => write!(
                f,
                "{stage}: the volume's format or geometry is not this build's (another kore \
                 version, another block size, or a resized image?)"
            ),
            Self::Persistence(stage, error) => write!(f, "{stage}: {error:?}"),
            Self::Fs(path, error) => write!(f, "{path}: {}", describe(*error)),
            Self::Exists(path) => write!(f, "{path} exists (put --replace replaces it)"),
            Self::NotFound(path) => write!(f, "{path}: not found"),
            Self::Verify(message) => write!(f, "read back after recovery: {message}"),
            Self::Output(error) => write!(f, "the output: {error}"),
        }
    }
}

impl VolError {
    /// Whether the image or disk is held by a running machine or another
    /// tool: QEMU's record lock on the images it runs, or another writer's.
    pub fn is_in_use(&self) -> bool {
        matches!(
            self,
            Self::Image(ImageError::InUse) | Self::Device(_, ImageError::InUse)
        )
    }
}

impl fmt::Debug for VolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl std::error::Error for VolError {}

impl From<ImageError> for VolError {
    fn from(error: ImageError) -> Self {
        Self::Image(error)
    }
}

/// The engine's refusals a host install meets, in its terms.
fn describe(error: FsError) -> String {
    match error {
        FsError::NotFound => "not found".into(),
        FsError::NotDirectory => "not a directory".into(),
        FsError::IsDirectory => "a directory".into(),
        FsError::AlreadyExists => "exists".into(),
        FsError::OutOfObjects => format!("the volume holds at most {MAX_OBJECTS} objects"),
        FsError::OutOfEdges => "a directory or the volume is out of entries".into(),
        FsError::OutOfStorage => "the volume has no contiguous room for it".into(),
        FsError::Range => format!("deeper than kore's {MAX_PATH_DEPTH} levels"),
        other => format!("{other:?}"),
    }
}

fn name(part: &str, path: &str) -> Result<Name, VolError> {
    Name::new(part).map_err(|_| {
        VolError::Host(format!(
            "{path}: {part:?} is not a kore name (UTF-8, at most 32 bytes, no NUL)"
        ))
    })
}

/// A path as terminal N types it: components under its home. The first
/// component may not be a name the terminal's view already gives the
/// store's read-only grants: a real child of the home by that name would
/// hide the grant (`view_name_taken`).
fn components(path: &str) -> Result<Vec<Name>, VolError> {
    let trimmed = path.trim_end_matches('/');
    if trimmed.is_empty() || trimmed.starts_with('/') {
        return Err(VolError::Host(format!(
            "{path}: a path under the terminal's home, like js/hello"
        )));
    }
    let names = trimmed
        .split('/')
        .map(|part| name(part, path))
        .collect::<Result<Vec<_>, _>>()?;
    let first = names[0].as_str();
    if [SYSTEM_APPS_VIEW, SYSTEM_LIBS_VIEW, SYSTEM_SHARE_VIEW].contains(&first) {
        return Err(VolError::Host(format!(
            "{path}: {first} names the store's grant in a terminal's view"
        )));
    }
    Ok(names)
}

fn display(names: &[Name]) -> String {
    names.iter().map(Name::as_str).collect::<Vec<_>>().join("/")
}

/// `work/face-N`.
fn home(face: u8) -> Result<[Name; 2], VolError> {
    let label = face_home_label(face)
        .ok_or_else(|| VolError::Host(format!("face {face}: not a terminal's")))?;
    let work = Name::new(SYSTEM_WORKSPACE_DIR).map_err(|_| VolError::Host("work".into()))?;
    let face = Name::from_bytes(&label).map_err(|_| VolError::Host("face".into()))?;
    Ok([work, face])
}

/// Whether `image` names a block device: a machine's disk, or a partition
/// of one, rather than an image of one.
fn is_device(image: &Path) -> Result<bool, VolError> {
    fs::metadata(image)
        .map(|meta| meta.file_type().is_block_device())
        .map_err(|error| VolError::Image(ImageError::Io(error)))
}

/// Find the user volume as the filesystem row does at boot, on an image or,
/// on Linux, a machine's disk (`device::locate`), and what holds the disk
/// while it is written: bound before the volume, so that it outlives it.
fn locate(
    image: &Path,
    access: Access,
    block_size: Option<u32>,
) -> Result<(Found, Option<Hold>), VolError> {
    if is_device(image)? {
        #[cfg(target_os = "linux")]
        return crate::device::locate(image, access, block_size);
        #[cfg(not(target_os = "linux"))]
        return Err(VolError::Host(format!(
            "{}: a disk's block device, which kore-vol reads and writes on Linux alone",
            image.display()
        )));
    }
    let device = ImageFile::open(
        image,
        access,
        block_size.unwrap_or(crate::DEFAULT_BLOCK_SIZE),
    )?;
    Ok((classify(device)?, None))
}

/// Find the user volume on `device` as the filesystem row does at boot: the
/// same composition, then the same `locate`.
pub(crate) fn classify(device: ImageFile) -> Result<Found, VolError> {
    let provider =
        DurableService::whole_medium_provider(device, CompletionMode::Poll).map_err(|failure| {
            VolError::Host(format!("the medium refused composition: {failure:?}"))
        })?;
    block_on(DurableService::locate(provider, CompletionMode::Poll))
        .map_err(|failure| VolError::Host(format!("the medium refused discovery: {failure:?}")))
}

/// Recover what [`locate`] found, refusing a volume kore itself would only
/// read when `writing`.
fn recover(found: Found, writing: bool) -> Result<Volume, VolError> {
    let provider = match found {
        Located::Recover(provider) => provider,
        Located::Format { .. } => return Err(VolError::Blank),
        Located::Volatile(why) => return Err(VolError::NoVolume(why)),
        // The system region's policy: `locate` never answers it.
        Located::Reformat { .. } => {
            return Err(VolError::NoVolume(note::VOLATILE_REFUSED_REGION));
        }
    };
    let volume = block_on(Volume::recover(provider, SESSION))
        .map_err(|error| VolError::Persistence("recover", error))?;
    let status = volume.status();
    if writing && (status.used_fallback_root || status.mode != MountMode::ReadWrite) {
        return Err(VolError::Degraded {
            generation: status.generation,
        });
    }
    Ok(volume)
}

/// Publish the change, then mount again from what is on the image: both
/// roots re-read and the namespace rebuilt from them, as the next boot will.
///
/// The change is committed twice. The first root publication makes it
/// durable; the second overwrites the other root, which still names the
/// volume as it was, so recovery has no pre-change fallback to choose.
fn publish(mut volume: Volume) -> Result<Volume, VolError> {
    block_on(volume.commit(SESSION)).map_err(|error| VolError::Persistence("commit", error))?;
    let generation =
        block_on(volume.commit(SESSION)).map_err(|error| VolError::Persistence("commit", error))?;
    let volume = block_on(volume.restart(SESSION))
        .map_err(|error| VolError::Persistence("recover after commit", error))?;
    let status = volume.status();
    if status.generation != generation || status.used_fallback_root {
        return Err(VolError::Verify(format!(
            "recovery chose generation {}{}, not the {generation} just committed",
            status.generation,
            if status.used_fallback_root {
                " (the fallback root)"
            } else {
                ""
            }
        )));
    }
    Ok(volume)
}

fn fs<T>(shown: &str, result: Result<T, FsError>) -> Result<T, VolError> {
    result.map_err(|error| VolError::Fs(shown.to_string(), error))
}

/// Walk from the persisted root along `names`, every step a directory,
/// holding one handle at a time. Missing directories are created when
/// `create`, else the walk answers `None`.
///
/// Every handle it opens is either answered or closed, whatever a step
/// meets: a session holds few, and a [`Mount`] lives as long as its caller
/// browses.
fn walk(
    volume: &Volume,
    names: &[Name],
    rights: Rights,
    create: bool,
) -> Result<Option<Handle>, VolError> {
    let engine = volume.engine();
    let mut at = fs("/", engine.open_root(SESSION, rights))?.handle;
    for (depth, part) in names.iter().enumerate() {
        let shown = display(&names[..=depth]);
        let next = match engine.open_at(SESSION, at, part, rights) {
            Ok(opened) => Ok(Some(opened.handle)),
            Err(FsError::NotFound) if create => fs(
                &shown,
                block_on(engine.create(SESSION, at, part, ObjectKind::Directory)),
            )
            .map(|created| Some(created.handle)),
            Err(FsError::NotFound) => Ok(None),
            Err(error) => Err(VolError::Fs(shown.clone(), error)),
        };
        let closed = fs(&shown, engine.close(SESSION, at));
        match (next, closed) {
            (Ok(Some(next)), Ok(())) => at = next,
            (Ok(None), Ok(())) => return Ok(None),
            (Ok(Some(next)), Err(error)) => {
                let _ = engine.close(SESSION, next);
                return Err(error);
            }
            (Ok(None), Err(error)) | (Err(error), _) => return Err(error),
        }
    }
    Ok(Some(at))
}

// ── reading: a mount and what it answers ────────────────────────────────

/// What recovery found when the volume was mounted to read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MountStatus {
    /// The root generation the mount reads.
    pub generation: u64,
    /// Recovery had to take the fallback root: kore itself mounts this
    /// volume read-only, and nothing writes it.
    pub used_fallback_root: bool,
}

/// What a listed object is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    File,
    Directory,
}

/// One entry of a listed directory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Listed {
    pub name: String,
    pub kind: Kind,
    /// A file's length in bytes; 0 for a directory.
    pub len: u64,
}

/// What [`Mount::list`] found at a path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Listing {
    /// A directory's entries, in the volume's order.
    Directory(Vec<Listed>),
    /// A file, and its length in bytes.
    File { len: u64 },
    /// The terminal's home itself, which nothing has written to yet: the
    /// row creates it on the boot that first roots a terminal there.
    EmptyHome,
}

/// One object beneath a walked directory ([`Mount::walk`]).
#[derive(Clone, Copy, Debug)]
pub struct Walked<'a> {
    /// Its path below the walked directory, `/`-separated.
    pub path: &'a str,
    pub kind: Kind,
    /// A file's length in bytes; 0 for a directory.
    pub len: u64,
    /// Levels below the walked directory: 1 for its own entries.
    pub depth: usize,
}

/// A volume recovered read-only, as [`read`] hands it to its caller.
///
/// It lives on the volume's own thread for the length of the call; what the
/// caller takes away is data.
pub struct Mount {
    volume: Volume,
}

/// Mount `image` read-only and run `session` against it. Kore keeps a volume
/// whose recovery took the fallback root read-only; the mount reads it, and
/// says so in [`Mount::status`].
pub fn read<T: Send>(
    image: &Path,
    block_size: Option<u32>,
    session: impl FnOnce(&Mount) -> Result<T, VolError> + Send,
) -> Result<T, VolError> {
    on_volume_stack(|| {
        // Nothing holds a disk that is only read.
        let (found, _) = locate(image, Access::ReadOnly, block_size)?;
        let mount = Mount {
            volume: recover(found, false)?,
        };
        session(&mount)
    })
}

/// A path to read: under terminal `face`'s home, or from the volume's root
/// when it starts with `/`. The names, and the path as given.
fn readable(face: u8, path: Option<&str>) -> Result<(Vec<Name>, String), VolError> {
    let mut names = Vec::new();
    let shown = match path {
        Some(path) if path.starts_with('/') => {
            for part in path.split('/').filter(|part| !part.is_empty()) {
                names.push(name(part, path)?);
            }
            format!("/{}", display(&names))
        }
        Some(path) => {
            names.extend(home(face)?);
            names.extend(components(path)?);
            path.to_string()
        }
        None => {
            names.extend(home(face)?);
            String::from(".")
        }
    };
    Ok((names, shown))
}

impl Mount {
    /// The mount's root generation, and whether recovery took the fallback
    /// root.
    pub fn status(&self) -> MountStatus {
        let status = self.volume.status();
        MountStatus {
            generation: status.generation,
            used_fallback_root: status.used_fallback_root,
        }
    }

    /// What terminal `face` sees at `path` (its home when `None`).
    pub fn list(&self, face: u8, path: Option<&str>) -> Result<Listing, VolError> {
        let (names, shown) = readable(face, path)?;
        let engine = self.volume.engine();
        let Some(at) = walk(&self.volume, &names, Rights::READ, false)? else {
            if path.is_none() {
                return Ok(Listing::EmptyHome);
            }
            return Err(VolError::NotFound(shown));
        };
        // The handle closes whatever the listing meets.
        let listing = self.listing(at, &shown);
        let closed = fs(&shown, engine.close(SESSION, at));
        let listing = listing?;
        closed?;
        Ok(listing)
    }

    fn listing(&self, at: Handle, shown: &str) -> Result<Listing, VolError> {
        let engine = self.volume.engine();
        let stat = fs(shown, engine.stat(SESSION, at))?;
        if !stat.kind.is_dir() {
            return Ok(Listing::File { len: stat.len });
        }
        let mut entries = Vec::new();
        let mut cursor = 0;
        loop {
            let page = fs(shown, engine.list(SESSION, at, cursor))?;
            for entry in &page.entries {
                if entry.kind.is_dir() {
                    entries.push(Listed {
                        name: entry.name.as_str().to_owned(),
                        kind: Kind::Directory,
                        len: 0,
                    });
                    continue;
                }
                let opened = fs(
                    shown,
                    engine.open_at(SESSION, at, &entry.name, Rights::READ),
                )?;
                fs(shown, engine.close(SESSION, opened.handle))?;
                entries.push(Listed {
                    name: entry.name.as_str().to_owned(),
                    kind: Kind::File,
                    len: opened.len,
                });
            }
            match page.next {
                Some(next) => cursor = next,
                None => return Ok(Listing::Directory(entries)),
            }
        }
    }

    /// Copy the file terminal `face` sees at `path` into `out`, and answer
    /// how many bytes it held.
    pub fn read_to<W: Write + ?Sized>(
        &self,
        face: u8,
        path: &str,
        out: &mut W,
    ) -> Result<u64, VolError> {
        let (names, shown) = readable(face, Some(path))?;
        let engine = self.volume.engine();
        let (last, parents) = names
            .split_last()
            .ok_or_else(|| VolError::NotFound(shown.clone()))?;
        let parent = walk(&self.volume, parents, Rights::READ, false)?
            .ok_or_else(|| VolError::NotFound(shown.clone()))?;
        let opened = engine.open_at(SESSION, parent, last, Rights::READ);
        let closed = fs(&shown, engine.close(SESSION, parent));
        let file = match opened {
            Ok(opened) => opened,
            Err(FsError::NotFound) => return Err(VolError::NotFound(shown)),
            Err(error) => return Err(VolError::Fs(shown, error)),
        };
        // The file's handle closes whatever the copy meets: a sink that has
        // had enough (a preview's bound) is an everyday end, not a leak.
        let copied = match closed {
            Ok(()) if file.kind.is_dir() => Err(VolError::Fs(shown.clone(), FsError::IsDirectory)),
            Ok(()) => self.copy(&file, &shown, out),
            Err(error) => Err(error),
        };
        let released = fs(&shown, engine.close(SESSION, file.handle));
        let len = copied?;
        released?;
        Ok(len)
    }

    fn copy<W: Write + ?Sized>(
        &self,
        file: &OpenedObject,
        shown: &str,
        out: &mut W,
    ) -> Result<u64, VolError> {
        let engine = self.volume.engine();
        let mut chunk = vec![0; CHUNK];
        let mut offset = 0;
        while offset < file.len {
            let read = fs(
                shown,
                block_on(engine.read(SESSION, file.handle, offset, &mut chunk)),
            )?;
            if read == 0 {
                break;
            }
            out.write_all(&chunk[..read]).map_err(VolError::Output)?;
            offset += read as u64;
        }
        out.flush().map_err(VolError::Output)?;
        Ok(offset)
    }

    /// Visit every object beneath the directory terminal `face` sees at
    /// `path`, depth first, each directory before what it holds, in the
    /// volume's order. The walk holds one handle per level, so it is bounded
    /// by kore's [`MAX_PATH_DEPTH`]; `visit` refusing stops it with that
    /// refusal.
    pub fn walk(
        &self,
        face: u8,
        path: &str,
        visit: &mut dyn FnMut(Walked<'_>) -> Result<(), VolError>,
    ) -> Result<(), VolError> {
        let (names, shown) = readable(face, Some(path))?;
        let engine = self.volume.engine();
        let at = walk(&self.volume, &names, Rights::READ, false)?
            .ok_or_else(|| VolError::NotFound(shown.clone()))?;
        let walked = match fs(&shown, engine.stat(SESSION, at)) {
            Ok(stat) if stat.kind.is_dir() => {
                self.descend(at, &shown, &mut String::new(), 1, visit)
            }
            Ok(_) => Err(VolError::Fs(shown.clone(), FsError::NotDirectory)),
            Err(error) => Err(error),
        };
        let closed = fs(&shown, engine.close(SESSION, at));
        walked.and(closed)
    }

    fn descend(
        &self,
        at: Handle,
        shown: &str,
        prefix: &mut String,
        depth: usize,
        visit: &mut dyn FnMut(Walked<'_>) -> Result<(), VolError>,
    ) -> Result<(), VolError> {
        let engine = self.volume.engine();
        let mut cursor = 0;
        loop {
            let page = fs(shown, engine.list(SESSION, at, cursor))?;
            for entry in &page.entries {
                let mark = prefix.len();
                if !prefix.is_empty() {
                    prefix.push('/');
                }
                prefix.push_str(entry.name.as_str());
                let opened = match engine.open_at(SESSION, at, &entry.name, Rights::READ) {
                    Ok(opened) => opened,
                    Err(error) => return Err(VolError::Fs(format!("{shown}/{prefix}"), error)),
                };
                let visited = if opened.kind.is_dir() {
                    visit(Walked {
                        path: prefix,
                        kind: Kind::Directory,
                        len: 0,
                        depth,
                    })
                    .and_then(|()| self.descend(opened.handle, shown, prefix, depth + 1, visit))
                } else {
                    visit(Walked {
                        path: prefix,
                        kind: Kind::File,
                        len: opened.len,
                        depth,
                    })
                };
                let closed = fs(shown, engine.close(SESSION, opened.handle));
                prefix.truncate(mark);
                visited?;
                closed?;
            }
            match page.next {
                Some(next) => cursor = next,
                None => return Ok(()),
            }
        }
    }
}

// ── put ─────────────────────────────────────────────────────────────────

/// What [`put`] installed, and the publication it made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Installed {
    pub files: usize,
    pub directories: usize,
    pub bytes: u64,
    /// The root generation before the install, and the one that names it.
    pub before: u64,
    pub after: u64,
}

/// A file as written: what its read-back must match.
struct Written {
    path: Vec<Name>,
    len: u64,
    digest: blake3::Hash,
}

/// Install `source` (a regular file or a directory) at `path` under
/// terminal `face`'s home; with `replace`, in place of what is there.
/// Symbolic links and special files are refused, not followed.
pub fn put(
    image: &Path,
    block_size: Option<u32>,
    face: u8,
    source: &Path,
    path: &str,
    replace: bool,
) -> Result<Installed, VolError> {
    on_volume_stack(|| install(image, block_size, face, source, path, replace))
}

fn install(
    image: &Path,
    block_size: Option<u32>,
    face: u8,
    source: &Path,
    path: &str,
    replace: bool,
) -> Result<Installed, VolError> {
    let target = components(path)?;
    let (last, parents) = target.split_last().expect("a path has a component");
    let plan = source::plan(source, *last).map_err(VolError::Host)?;
    let mut route = home(face)?.to_vec();
    route.extend_from_slice(parents);
    // Depth as the engine counts it: edges from the persisted root.
    let deepest = route.len() + plan.depth;
    if deepest > MAX_PATH_DEPTH {
        return Err(VolError::Host(format!(
            "{path}: its deepest object would sit {deepest} levels below the volume's root, \
             past kore's {MAX_PATH_DEPTH}"
        )));
    }
    let (found, _disk) = locate(image, Access::ReadWrite, block_size)?;
    let volume = recover(found, true)?;
    let before = volume.status().generation;
    let parent = walk(&volume, &route, Rights::ALL, true)?.ok_or_else(|| {
        VolError::Host(format!("{path}: a directory created on the way vanished"))
    })?;
    let engine = volume.engine();
    match engine.open_at(SESSION, parent, last, Rights::READ) {
        Ok(existing) => {
            fs(path, engine.close(SESSION, existing.handle))?;
            if !replace {
                return Err(VolError::Exists(path.to_string()));
            }
            // The old tree's blocks stay as they are until the commit that
            // stops naming them has itself been superseded: the provider
            // retires a detached extent only after two root publications.
            fs(path, block_on(engine.detach_edge(SESSION, parent, last)))?;
        }
        Err(FsError::NotFound) => {}
        Err(error) => return Err(VolError::Fs(path.to_string(), error)),
    }
    let room = MAX_OBJECTS.saturating_sub(engine.object_count());
    if plan.objects() > room {
        return Err(VolError::Host(format!(
            "{path}: {} objects, and the volume has room for {room} more of its {MAX_OBJECTS}",
            plan.objects()
        )));
    }
    let mut written = Vec::with_capacity(plan.files);
    let mut at = target.clone();
    create(&volume, parent, &plan.root, &mut at, &mut written)?;
    fs(path, volume.engine().close(SESSION, parent))?;
    let volume = publish(volume)?;
    verify(&volume, face, &written)?;
    Ok(Installed {
        files: plan.files,
        directories: plan.directories,
        bytes: plan.bytes,
        before,
        after: volume.status().generation,
    })
}

/// Create `entry` under `parent`, depth first, holding one handle per level
/// beneath it.
fn create(
    volume: &Volume,
    parent: Handle,
    entry: &Entry,
    path: &mut Vec<Name>,
    written: &mut Vec<Written>,
) -> Result<(), VolError> {
    let engine = volume.engine();
    let shown = display(path);
    match entry {
        Entry::Directory { name, entries } => {
            let handle = fs(
                &shown,
                block_on(engine.create(SESSION, parent, name, ObjectKind::Directory)),
            )?
            .handle;
            for child in entries {
                path.push(*child.name());
                create(volume, handle, child, path, written)?;
                path.pop();
            }
            fs(&shown, engine.close(SESSION, handle))
        }
        Entry::File {
            name,
            path: source,
            len,
        } => {
            let handle = fs(
                &shown,
                block_on(engine.create(SESSION, parent, name, ObjectKind::File)),
            )?
            .handle;
            fs(
                &shown,
                block_on(engine.resize(SESSION, handle, *len, ANY_REVISION)),
            )?;
            let mut file = File::open(source)
                .map_err(|error| VolError::Host(format!("{}: {error}", source.display())))?;
            let mut hasher = blake3::Hasher::new();
            let mut chunk = vec![0; CHUNK];
            let mut offset = 0;
            while offset < *len {
                let want = CHUNK.min((*len - offset) as usize);
                file.read_exact(&mut chunk[..want]).map_err(|error| {
                    VolError::Host(format!("{}: {error} while copying", source.display()))
                })?;
                hasher.update(&chunk[..want]);
                fs(
                    &shown,
                    block_on(engine.write_at(
                        SESSION,
                        handle,
                        offset,
                        &chunk[..want],
                        ANY_REVISION,
                    )),
                )?;
                offset += want as u64;
            }
            // A file that grew after it was measured would be copied short.
            let more = file
                .read(&mut chunk[..1])
                .map_err(|error| VolError::Host(format!("{}: {error}", source.display())))?;
            if more != 0 {
                return Err(VolError::Host(format!(
                    "{}: changed while it was copied",
                    source.display()
                )));
            }
            fs(&shown, engine.close(SESSION, handle))?;
            written.push(Written {
                path: path.clone(),
                len: *len,
                digest: hasher.finalize(),
            });
            Ok(())
        }
    }
}

/// Read every written file back through the recovered volume and compare
/// it with what was written.
fn verify(volume: &Volume, face: u8, written: &[Written]) -> Result<(), VolError> {
    let engine = volume.engine();
    let mut chunk = vec![0; CHUNK];
    for file in written {
        let shown = display(&file.path);
        let mut route = home(face)?.to_vec();
        route.extend_from_slice(&file.path);
        let handle = walk(volume, &route, Rights::READ, false)?
            .ok_or_else(|| VolError::Verify(format!("{shown} is missing")))?;
        let len = fs(&shown, engine.stat(SESSION, handle))?.len;
        if len != file.len {
            return Err(VolError::Verify(format!(
                "{shown} holds {len} bytes, not {}",
                file.len
            )));
        }
        let mut hasher = blake3::Hasher::new();
        let mut offset = 0;
        while offset < len {
            let read = fs(
                &shown,
                block_on(engine.read(SESSION, handle, offset, &mut chunk)),
            )?;
            if read == 0 {
                return Err(VolError::Verify(format!(
                    "{shown} ends at {offset} of its {len} bytes"
                )));
            }
            hasher.update(&chunk[..read]);
            offset += read as u64;
        }
        fs(&shown, engine.close(SESSION, handle))?;
        if hasher.finalize() != file.digest {
            return Err(VolError::Verify(format!("{shown} differs from its source")));
        }
    }
    Ok(())
}

// ── rm ──────────────────────────────────────────────────────────────────

/// What [`rm`] removed: the publication that stopped naming it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Removed {
    pub before: u64,
    pub after: u64,
}

/// Remove `path` (and what it holds) from terminal `face`'s home.
pub fn rm(
    image: &Path,
    block_size: Option<u32>,
    face: u8,
    path: &str,
) -> Result<Removed, VolError> {
    on_volume_stack(|| remove(image, block_size, face, path))
}

fn remove(
    image: &Path,
    block_size: Option<u32>,
    face: u8,
    path: &str,
) -> Result<Removed, VolError> {
    let target = components(path)?;
    let (last, parents) = target.split_last().expect("a path has a component");
    let (found, _disk) = locate(image, Access::ReadWrite, block_size)?;
    let volume = recover(found, true)?;
    let before = volume.status().generation;
    let mut route = home(face)?.to_vec();
    route.extend_from_slice(parents);
    let parent = walk(&volume, &route, Rights::ALL, false)?
        .ok_or_else(|| VolError::NotFound(path.to_string()))?;
    let engine = volume.engine();
    match block_on(engine.detach_edge(SESSION, parent, last)) {
        Ok(()) => {}
        Err(FsError::NotFound) => return Err(VolError::NotFound(path.to_string())),
        Err(error) => return Err(VolError::Fs(path.to_string(), error)),
    }
    fs(path, engine.close(SESSION, parent))?;
    let volume = publish(volume)?;
    let mut full = home(face)?.to_vec();
    full.extend_from_slice(&target);
    if walk(&volume, &full, Rights::READ, false)?.is_some() {
        return Err(VolError::Verify(format!("{path} is still there")));
    }
    Ok(Removed {
        before,
        after: volume.status().generation,
    })
}

// ── format ──────────────────────────────────────────────────────────────

/// What [`format()`] made: a volume at its first generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Formatted {
    pub generation: u64,
}

/// Format a blank image as kore's first boot would. Nothing else is ever
/// formatted: a volume kore already wrote, foreign data and any disk are
/// refused.
pub fn format(image: &Path, block_size: Option<u32>) -> Result<Formatted, VolError> {
    on_volume_stack(|| format_blank(image, block_size))
}

fn format_blank(image: &Path, block_size: Option<u32>) -> Result<Formatted, VolError> {
    if is_device(image)? {
        return Err(VolError::Host(format!(
            "{}: a disk, which kore-vol never formats: the machine formats a blank volume at \
             its first boot, and partitioning a disk is tools/provision_fs.py's",
            image.display()
        )));
    }
    let (found, _) = locate(image, Access::ReadWrite, block_size)?;
    let (provider, era) = match found {
        Located::Format { provider, era } => (provider, era),
        Located::Recover(_) => {
            return Err(VolError::Host(
                "the volume is kore's already: nothing formatted".into(),
            ));
        }
        Located::Volatile(why) => return Err(VolError::NoVolume(why)),
        Located::Reformat { .. } => {
            return Err(VolError::NoVolume(note::VOLATILE_REFUSED_REGION));
        }
    };
    let volume = block_on(Volume::format_with_era(provider, SESSION, era))
        .map_err(|error| VolError::Persistence("format", error))?;
    let volume = block_on(volume.restart(SESSION))
        .map_err(|error| VolError::Persistence("recover after format", error))?;
    Ok(Formatted {
        generation: volume.status().generation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_held_image_or_disk_is_in_use_and_nothing_else_is() {
        assert!(VolError::Image(ImageError::InUse).is_in_use());
        assert!(VolError::Device(PathBuf::from("/dev/nvme0n1"), ImageError::InUse).is_in_use());
        assert!(!VolError::Image(ImageError::Claimed).is_in_use());
        assert!(!VolError::Blank.is_in_use());
        assert!(!VolError::Host("in use".into()).is_in_use());
    }
}
