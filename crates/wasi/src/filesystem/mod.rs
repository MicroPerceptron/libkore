//! `wasi:filesystem` — the task's own filesystem view, as one preopen.
//!
//! A guest gets exactly one preopen, `.`: the task's root view (on Kore,
//! the directory it was launched in), with the rights that view conveys.
//! Every path resolves beneath a descriptor the guest holds, one component
//! at a time from that descriptor's own view; there is no `..`,
//! no absolute path, and never the task's named grants. A task launched
//! without the filesystem profile has no [`Filesystem`], and
//! `get-directories` is empty.
//!
//! Every import is async: a request is a round trip to the filesystem
//! service, parked on the Kore executor while the guest's fiber waits. The
//! one pair carries one request at a time, and the host keeps that custody
//! across cancelled polls and dropped descriptors.
//!
//! **Implemented**: `read-via-stream`, `write-via-stream`,
//! `append-via-stream`, `read`, `write`, `get-type`, `get-flags`, `stat`,
//! `stat-at`, `open-at` (create, directory, exclusive, truncate),
//! `read-directory`, `create-directory-at`, `unlink-file-at`,
//! `remove-directory-at`, `rename-at`, `set-size`, `sync`, `sync-data`,
//! `metadata-hash`, `metadata-hash-at`, `is-same-object`, `advise` (advice
//! is a hint; taking none is a valid answer), `filesystem-error-code`, and
//! `preopens.get-directories`.
//!
//! **`unsupported`**: `set-times`, `set-times-at` (the graph keeps no
//! timestamps; `stat` reports none), `link-at`, `symlink-at`, `readlink-at`.
//!
//! Semantics a guest may notice: `rename-at` onto an existing name is
//! refused (`exist`) rather than replacing it; `open-at` of `.` reopens the
//! descriptor's directory by the path it was opened with, from the preopen.

mod host;
mod stream;
mod walk;

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use kore::fs::dag::bounds::MAX_DIR_ENTRIES;
use kore::fs::dag::{GrantId, ObjectKind, ProviderError};
use kore::fs::{DirEntry, FsError as Refused, Name, Refusal, Rights, Stat, View};
use kore::runtime::Pinned;
use wasmtime::component::{Resource, ResourceTableError};
use wasmtime_wasi_io::streams::{DynInputStream, DynOutputStream};

use crate::WasiCtxView;
use crate::bindings::wasi::filesystem::{preopens, types};
use host::{Custody, Face, FsHost, Owned, Shared};
use stream::{FileInput, FileOutput, StreamFailure};
use types::{
    Advice, DescriptorFlags, DescriptorStat, DescriptorType, DirectoryEntry, ErrorCode, Filesize,
    MetadataHashValue, NewTimestamp, OpenFlags, PathFlags,
};

pub use host::Filesystem;

/// A `wasi:filesystem` failure: a WASI error code the guest sees, or a trap.
pub struct FsError(Failure);

enum Failure {
    Code(ErrorCode),
    Trap(wasmtime::Error),
}

impl FsError {
    /// Trap the guest.
    pub fn trap(error: wasmtime::Error) -> Self {
        Self(Failure::Trap(error))
    }

    /// The error code, when this is one rather than a trap.
    pub fn code(&self) -> Option<ErrorCode> {
        match &self.0 {
            Failure::Code(code) => Some(*code),
            Failure::Trap(_) => None,
        }
    }

    /// As a trap: an error code where the interface has no error to answer
    /// with becomes one naming the code.
    fn into_trap(self) -> wasmtime::Error {
        match self.0 {
            Failure::Code(code) => wasmtime::format_err!("{code:?}"),
            Failure::Trap(error) => error,
        }
    }
}

impl From<ErrorCode> for FsError {
    fn from(code: ErrorCode) -> Self {
        Self(Failure::Code(code))
    }
}

impl From<ResourceTableError> for FsError {
    fn from(error: ResourceTableError) -> Self {
        Self::trap(error.into())
    }
}

impl From<wasmtime::Error> for FsError {
    fn from(error: wasmtime::Error) -> Self {
        Self::trap(error)
    }
}

impl fmt::Debug for FsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            Failure::Code(code) => write!(f, "{code:?}"),
            Failure::Trap(error) => write!(f, "trap: {error:?}"),
        }
    }
}

type FsResult<T> = Result<T, FsError>;

/// Kore's refusal as the WASI error code a guest acts on — one row per
/// refusal, so a new one is a compile error here, not a silent `io`.
pub(crate) fn code(refusal: Refusal) -> ErrorCode {
    let error = match refusal {
        Refusal::Fs(error) => error,
        // The request may have been published; the host publishes nothing
        // after it. Not a filesystem answer.
        Refusal::Pair => return ErrorCode::Io,
        Refusal::Usage(_) => return ErrorCode::Invalid,
    };
    match error {
        Refused::NotFound => ErrorCode::NoEntry,
        Refused::AlreadyExists => ErrorCode::Exist,
        Refused::NotDirectory => ErrorCode::NotDirectory,
        Refused::IsDirectory => ErrorCode::IsDirectory,
        Refused::InvalidName => ErrorCode::Invalid,
        Refused::InvalidHandle => ErrorCode::BadDescriptor,
        Refused::AccessDenied => ErrorCode::Access,
        // The route the handle was proven by is gone: the name no longer
        // leads here.
        Refused::AccessLost => ErrorCode::NoEntry,
        Refused::Conflict => ErrorCode::Busy,
        Refused::Cycle => ErrorCode::Loop,
        Refused::NotEmpty => ErrorCode::NotEmpty,
        Refused::OutOfObjects
        | Refused::OutOfEdges
        | Refused::OutOfStorage
        | Refused::OutOfPlacements
        | Refused::OutOfVersionLeases
        | Refused::OutOfIndexSlots => ErrorCode::InsufficientSpace,
        // The session's handle budget travels with its authority edge.
        Refused::OutOfHandles => ErrorCode::Quota,
        Refused::Range => ErrorCode::Overflow,
        Refused::BudgetExceeded | Refused::Busy => ErrorCode::Busy,
        Refused::IndexIncomplete | Refused::NoWindow => ErrorCode::Io,
        Refused::Unsupported => ErrorCode::Unsupported,
        Refused::ProviderFailure(ProviderError::ReadOnly) => ErrorCode::ReadOnly,
        Refused::ProviderFailure(_) => ErrorCode::Io,
    }
}

fn off_thread() -> FsError {
    FsError::trap(wasmtime::format_err!(
        "a filesystem resource was used off the thread that owns it"
    ))
}

/// Pin a Kore future so a `Send` host future can await it (see
/// [`Pinned`]).
fn pinned<F: Future>(future: F) -> FsResult<Pinned<F>> {
    Pinned::new(future).map_err(|_| off_thread())
}

fn kind_type(kind: ObjectKind) -> DescriptorType {
    match kind {
        ObjectKind::File => DescriptorType::RegularFile,
        ObjectKind::Directory => DescriptorType::Directory,
    }
}

fn stat_of(stat: Stat) -> DescriptorStat {
    DescriptorStat {
        type_: kind_type(stat.kind),
        link_count: u64::from(stat.links),
        size: stat.len,
        data_access_timestamp: None,
        data_modification_timestamp: None,
        status_change_timestamp: None,
    }
}

fn hash_of(stat: Stat) -> MetadataHashValue {
    // The durable identity and the committed revision: equal exactly when
    // the object is the same and unchanged.
    MetadataHashValue {
        lower: stat.stable_object.get(),
        upper: stat.revision,
    }
}

/// A preopen's flags: what its route conveys.
fn preopen_flags(rights: Rights) -> DescriptorFlags {
    let mut flags = DescriptorFlags::READ;
    if rights.contains(Rights::WRITE) {
        flags |= DescriptorFlags::WRITE;
    }
    if rights.contains(Rights::MUTATE_NAMESPACE) {
        flags |= DescriptorFlags::MUTATE_DIRECTORY;
    }
    flags
}

/// An open file or directory: a Kore view the guest holds, with the path
/// it was opened by (from its preopen) and the flags it was opened with.
pub struct Descriptor(Pinned<Open>);

/// The preopen a descriptor descends from: what its path is relative to,
/// and what reopening it by that path starts from. A view has no
/// duplicate verb, so `open-at` of `.` reopens the directory from here.
#[derive(Clone, Copy)]
enum Origin {
    /// `.`, the task's root view.
    Root,
    /// One of the task's named grants the embedder offered
    /// ([`Filesystem::with_named_grants`]).
    Grant(GrantId),
}

struct Open {
    /// Shared with the streams and listings made from this descriptor.
    file: Shared,
    host: Rc<FsHost>,
    origin: Origin,
    /// From `origin`; empty for the preopen itself.
    path: Vec<Name>,
    flags: DescriptorFlags,
}

impl Descriptor {
    fn new(
        host: Rc<FsHost>,
        file: Shared,
        origin: Origin,
        path: Vec<Name>,
        flags: DescriptorFlags,
    ) -> Option<Self> {
        let open = Open {
            file,
            host,
            origin,
            path,
            flags,
        };
        Pinned::new(open).ok().map(Self)
    }

    fn open(&self) -> FsResult<&Open> {
        self.0.get().ok_or_else(off_thread)
    }

    /// Close the view, unless a stream made from it still holds it (then
    /// the last holder's drop queues it). The op still runs if this future
    /// is dropped after it started; the view is queued if it is dropped
    /// before.
    async fn close(self) {
        let Ok(open) = self.0.into_inner() else {
            return;
        };
        let Open { file, host, .. } = open;
        if let Some(owned) = Rc::into_inner(file) {
            host.exec(async move |pair: &mut Custody| {
                if let Some(view) = owned.into_view() {
                    pair.close(view).await;
                }
            })
            .await;
        }
    }
}

/// Close `shared` now if nothing else holds it; otherwise the last holder's
/// drop queues it.
async fn release(pair: &mut Custody, shared: Option<Shared>) {
    if let Some(view) = shared.and_then(Rc::into_inner).and_then(Owned::into_view) {
        pair.close(view).await;
    }
}

/// The directory `names` leads to beneath `base` — `None` for `base`
/// itself — owned as soon as it is opened.
async fn parent(
    pair: &mut Custody,
    base: &View,
    names: &[Name],
    rights: Rights,
) -> Result<Option<Shared>, Refusal> {
    if names.is_empty() {
        return Ok(None);
    }
    let dir = walk::walk(pair, base, names, rights).await?;
    let dir = pair.own(dir);
    if !dir.view()?.is_directory() {
        release(pair, Some(dir)).await;
        return Err(Refusal::Fs(Refused::NotDirectory));
    }
    Ok(Some(dir))
}

/// The view a [`parent`] result names.
fn at<'a>(walked: &'a Option<Shared>, base: &'a View) -> Result<&'a View, Refusal> {
    match walked {
        Some(dir) => dir.view(),
        None => Ok(base),
    }
}

impl Open {
    fn view(&self) -> FsResult<&View> {
        self.file.view().map_err(|refusal| code(refusal).into())
    }

    /// `flag` or `refusal`.
    fn need(&self, flag: DescriptorFlags, refusal: ErrorCode) -> FsResult<()> {
        if self.flags.contains(flag) {
            Ok(())
        } else {
            Err(refusal.into())
        }
    }

    /// A file's view, for byte I/O.
    fn file(&self) -> FsResult<Shared> {
        if self.view()?.is_directory() {
            return Err(ErrorCode::IsDirectory.into());
        }
        Ok(self.file.clone())
    }

    /// A directory's view, for namespace verbs.
    fn dir(&self) -> FsResult<Shared> {
        if !self.view()?.is_directory() {
            return Err(ErrorCode::NotDirectory.into());
        }
        Ok(self.file.clone())
    }

    /// Run `op` on the pair and fold its refusal into a WASI code.
    async fn run<R: 'static>(
        &self,
        op: impl AsyncFnOnce(&mut Custody) -> Result<R, Refusal> + 'static,
    ) -> FsResult<R> {
        match self.host.exec(op).await {
            Some(Ok(value)) => Ok(value),
            Some(Err(refusal)) => Err(code(refusal).into()),
            None => Err(ErrorCode::Io.into()),
        }
    }

    async fn stat_at(&self, path: &str) -> FsResult<Stat> {
        let names = walk::components(path)?;
        let base = self.file.clone();
        self.run(async move |pair: &mut Custody| {
            let base = base.view()?;
            if names.is_empty() {
                return pair.call(async |fs: &mut Face| fs.stat(base).await).await;
            }
            let target = walk::walk(pair, base, &names, Rights::READ).await?;
            let stat = pair
                .call(async |fs: &mut Face| fs.stat(&target).await)
                .await;
            pair.close(target).await;
            stat
        })
        .await
    }

    async fn open_at(
        &self,
        path: &str,
        oflags: OpenFlags,
        flags: DescriptorFlags,
    ) -> FsResult<Descriptor> {
        let base = self.dir()?;
        if path.is_empty() {
            return Err(ErrorCode::NoEntry.into());
        }
        let names = walk::components(path)?;
        let create = oflags.contains(OpenFlags::CREATE);
        let exclusive = oflags.contains(OpenFlags::EXCLUSIVE);
        let truncate = oflags.contains(OpenFlags::TRUNCATE);
        let directory = oflags.contains(OpenFlags::DIRECTORY);
        if directory && (create || exclusive || truncate) {
            return Err(ErrorCode::Invalid.into());
        }
        // Nothing past what this descriptor conveys.
        self.need(DescriptorFlags::READ, ErrorCode::NotPermitted)?;
        if create || truncate {
            self.need(DescriptorFlags::MUTATE_DIRECTORY, ErrorCode::NotPermitted)?;
        }
        if flags.contains(DescriptorFlags::WRITE) || truncate {
            self.need(DescriptorFlags::WRITE, ErrorCode::NotPermitted)?;
        }
        if flags.contains(DescriptorFlags::MUTATE_DIRECTORY) {
            self.need(DescriptorFlags::MUTATE_DIRECTORY, ErrorCode::NotPermitted)?;
        }

        let mut rights = Rights::READ;
        if flags.contains(DescriptorFlags::WRITE) || truncate {
            rights = rights.union(Rights::WRITE);
        }
        if flags.contains(DescriptorFlags::MUTATE_DIRECTORY) {
            rights = rights.union(Rights::MUTATE_NAMESPACE);
        }
        // A created object carries its parent's rights, so the parent is
        // walked with what the new file needs beside the mutation.
        let parent_rights = if create {
            rights.union(Rights::MUTATE_NAMESPACE)
        } else {
            rights
        };
        let origin = self.origin;
        let from_preopen = self.path.clone();
        let mut path_of = self.path.clone();
        path_of.extend_from_slice(&names);

        let file = self
            .run(async move |pair: &mut Custody| {
                let base = base.view()?;
                // Owned as soon as it is opened: a close still to be
                // awaited must not leave it a bare view.
                let file = match names.split_last() {
                    // `.`: this directory again, reopened by its own path
                    // from its preopen (a view has no duplicate verb).
                    None => {
                        if create && exclusive {
                            return Err(Refusal::Fs(Refused::AlreadyExists));
                        }
                        let root = pair
                            .call(async |fs: &mut Face| match origin {
                                Origin::Root => fs.open_root(rights).await,
                                Origin::Grant(grant) => fs.open_grant(grant, rights).await,
                            })
                            .await?;
                        let root = pair.own(root);
                        if from_preopen.is_empty() {
                            root
                        } else {
                            let dir = walk::walk(pair, root.view()?, &from_preopen, rights).await;
                            let dir = dir.map(|dir| pair.own(dir));
                            release(pair, Some(root)).await;
                            dir?
                        }
                    }
                    Some((&last, parents)) => {
                        let walked = parent(pair, base, parents, parent_rights).await?;
                        let dir = at(&walked, base)?;
                        let opened = pair
                            .call(async |fs: &mut Face| fs.open_at(dir, last, rights).await)
                            .await;
                        let found = match opened {
                            Ok(existing) if create && exclusive => {
                                pair.close(existing).await;
                                Err(Refusal::Fs(Refused::AlreadyExists))
                            }
                            Err(Refusal::Fs(Refused::NotFound)) if create => {
                                pair.call(async |fs: &mut Face| {
                                    fs.create(dir, last, ObjectKind::File).await
                                })
                                .await
                            }
                            other => other,
                        };
                        let found = found.map(|view| pair.own(view));
                        release(pair, walked).await;
                        found?
                    }
                };
                let is_dir = file.view()?.is_directory();
                if directory && !is_dir {
                    return Err(Refusal::Fs(Refused::NotDirectory));
                }
                if is_dir && (rights.contains(Rights::WRITE) || truncate) {
                    return Err(Refusal::Fs(Refused::IsDirectory));
                }
                if truncate {
                    let view = file.view()?;
                    pair.call(async |fs: &mut Face| fs.resize(view, 0).await)
                        .await?;
                }
                Ok(file)
            })
            .await?;
        Descriptor::new(self.host.clone(), file, origin, path_of, flags).ok_or_else(off_thread)
    }

    /// The directory holding `path`'s last component, and that component.
    fn edge(&self, path: &str) -> FsResult<(Shared, Vec<Name>, Name)> {
        let base = self.dir()?;
        self.need(DescriptorFlags::MUTATE_DIRECTORY, ErrorCode::NotPermitted)?;
        let mut names = walk::components(path)?;
        let last = names.pop().ok_or(ErrorCode::Invalid)?;
        Ok((base, names, last))
    }

    /// Remove `path`'s edge once its target is `kind` — `wrong` otherwise.
    async fn remove_at(&self, path: &str, kind: ObjectKind, wrong: Refused) -> FsResult<()> {
        let (base, parents, last) = self.edge(path)?;
        self.run(async move |pair: &mut Custody| {
            let base = base.view()?;
            let walked = parent(pair, base, &parents, MUTATE).await?;
            let dir = at(&walked, base)?;
            let target = pair
                .call(async |fs: &mut Face| fs.open_at(dir, last, Rights::READ).await)
                .await;
            let removed = match target {
                Ok(target) => {
                    let found = target.kind();
                    pair.close(target).await;
                    if found == kind {
                        pair.call(async |fs: &mut Face| fs.unlink(dir, last).await)
                            .await
                    } else {
                        Err(Refusal::Fs(wrong))
                    }
                }
                Err(error) => Err(error),
            };
            release(pair, walked).await;
            removed
        })
        .await
    }
}

/// The rights a namespace verb walks its parent with.
const MUTATE: Rights = Rights::READ.union(Rights::MUTATE_NAMESPACE);

/// A directory's entries, a page at a time.
pub struct DirectoryEntryStream(Pinned<Listing>);

struct Listing {
    host: Rc<FsHost>,
    dir: Shared,
    /// Where the next page starts; `None` once the last page is in.
    cursor: Option<u32>,
    page: VecDeque<DirEntry>,
}

impl Listing {
    async fn next(&mut self) -> FsResult<Option<DirectoryEntry>> {
        // Each page either yields an entry or moves the cursor forward.
        for _ in 0..=MAX_DIR_ENTRIES {
            if let Some(entry) = self.page.pop_front() {
                return Ok(Some(DirectoryEntry {
                    type_: kind_type(entry.kind),
                    name: String::from(entry.name.as_str()),
                }));
            }
            let Some(cursor) = self.cursor else {
                return Ok(None);
            };
            let dir = self.dir.clone();
            let listed = self
                .host
                .exec(async move |pair: &mut Custody| {
                    pair.call(async |fs: &mut Face| {
                        // Each entry is copied out of the window before the
                        // next request can reuse it.
                        let (entries, next) = fs.list_page(dir.view()?, cursor).await?;
                        let mut page = VecDeque::new();
                        for index in 0..entries as usize {
                            page.push_back(fs.dir_entry(index)?);
                        }
                        Ok((page, next))
                    })
                    .await
                })
                .await;
            let (page, next) = match listed {
                Some(Ok(page)) => page,
                Some(Err(refusal)) => return Err(code(refusal).into()),
                None => return Err(ErrorCode::Io.into()),
            };
            self.page = page;
            self.cursor = next;
        }
        Err(ErrorCode::Io.into())
    }
}

impl types::Host for WasiCtxView<'_> {
    fn convert_error_code(&mut self, error: FsError) -> wasmtime::Result<ErrorCode> {
        match error.0 {
            Failure::Code(code) => Ok(code),
            Failure::Trap(error) => Err(error),
        }
    }

    async fn filesystem_error_code(
        &mut self,
        error: Resource<types::Error>,
    ) -> wasmtime::Result<Option<ErrorCode>> {
        let error = self.table.get(&error)?;
        Ok(error
            .downcast_ref::<StreamFailure>()
            .map(|failure| failure.0))
    }
}

impl types::HostDescriptor for WasiCtxView<'_> {
    async fn read_via_stream(
        &mut self,
        fd: Resource<Descriptor>,
        offset: Filesize,
    ) -> FsResult<Resource<DynInputStream>> {
        let open = self.table.get(&fd)?.open()?;
        open.need(DescriptorFlags::READ, ErrorCode::BadDescriptor)?;
        let file = open.file()?;
        let stream = FileInput::new(open.host.clone(), file, offset).ok_or_else(off_thread)?;
        let stream: DynInputStream = Box::new(stream);
        Ok(self.table.push(stream)?)
    }

    async fn write_via_stream(
        &mut self,
        fd: Resource<Descriptor>,
        offset: Filesize,
    ) -> FsResult<Resource<DynOutputStream>> {
        let open = self.table.get(&fd)?.open()?;
        open.need(DescriptorFlags::WRITE, ErrorCode::BadDescriptor)?;
        let file = open.file()?;
        let stream =
            FileOutput::new(open.host.clone(), file, Some(offset)).ok_or_else(off_thread)?;
        let stream: DynOutputStream = Box::new(stream);
        Ok(self.table.push(stream)?)
    }

    async fn append_via_stream(
        &mut self,
        fd: Resource<Descriptor>,
    ) -> FsResult<Resource<DynOutputStream>> {
        let open = self.table.get(&fd)?.open()?;
        open.need(DescriptorFlags::WRITE, ErrorCode::BadDescriptor)?;
        let file = open.file()?;
        let stream = FileOutput::new(open.host.clone(), file, None).ok_or_else(off_thread)?;
        let stream: DynOutputStream = Box::new(stream);
        Ok(self.table.push(stream)?)
    }

    /// Advice is a hint; taking none is a valid answer.
    async fn advise(
        &mut self,
        fd: Resource<Descriptor>,
        _offset: Filesize,
        _len: Filesize,
        _advice: Advice,
    ) -> FsResult<()> {
        self.table.get(&fd)?.open()?.file()?;
        Ok(())
    }

    async fn sync_data(&mut self, fd: Resource<Descriptor>) -> FsResult<()> {
        self.sync(fd).await
    }

    async fn get_flags(&mut self, fd: Resource<Descriptor>) -> FsResult<DescriptorFlags> {
        Ok(self.table.get(&fd)?.open()?.flags)
    }

    async fn get_type(&mut self, fd: Resource<Descriptor>) -> FsResult<DescriptorType> {
        Ok(kind_type(self.table.get(&fd)?.open()?.view()?.kind()))
    }

    async fn set_size(&mut self, fd: Resource<Descriptor>, size: Filesize) -> FsResult<()> {
        let fd = self.table.get(&fd)?;
        pinned(async move {
            let open = fd.open()?;
            open.need(DescriptorFlags::WRITE, ErrorCode::BadDescriptor)?;
            let file = open.file()?;
            open.run(async move |pair: &mut Custody| {
                let file = file.view()?;
                pair.call(async |fs: &mut Face| fs.resize(file, size).await)
                    .await
                    .map(|_| ())
            })
            .await
        })?
        .await
    }

    async fn set_times(
        &mut self,
        _fd: Resource<Descriptor>,
        _atime: NewTimestamp,
        _mtime: NewTimestamp,
    ) -> FsResult<()> {
        Err(ErrorCode::Unsupported.into())
    }

    async fn read(
        &mut self,
        fd: Resource<Descriptor>,
        len: Filesize,
        offset: Filesize,
    ) -> FsResult<(Vec<u8>, bool)> {
        let fd = self.table.get(&fd)?;
        pinned(async move {
            let open = fd.open()?;
            open.need(DescriptorFlags::READ, ErrorCode::BadDescriptor)?;
            let file = open.file()?;
            let wanted = usize::try_from(len).unwrap_or(usize::MAX);
            let bytes = open
                .run(async move |pair: &mut Custody| {
                    let file = file.view()?;
                    pair.call(async |fs: &mut Face| {
                        // At most one transfer; a short read is not EOF,
                        // only an empty one is.
                        let mut bytes = vec![0; wanted.min(fs.chunk())];
                        if bytes.is_empty() && wanted > 0 {
                            return Err(Refusal::Fs(Refused::NoWindow));
                        }
                        let count = fs.read_at(file, offset, &mut bytes).await?;
                        bytes.truncate(count);
                        Ok(bytes)
                    })
                    .await
                })
                .await?;
            let end = bytes.is_empty() && len > 0;
            Ok((bytes, end))
        })?
        .await
    }

    async fn write(
        &mut self,
        fd: Resource<Descriptor>,
        buffer: Vec<u8>,
        offset: Filesize,
    ) -> FsResult<Filesize> {
        let fd = self.table.get(&fd)?;
        pinned(async move {
            let open = fd.open()?;
            open.need(DescriptorFlags::WRITE, ErrorCode::BadDescriptor)?;
            let file = open.file()?;
            open.run(async move |pair: &mut Custody| {
                let file = file.view()?;
                // Transfer by transfer; a refusal after some committed is
                // the short write it is, not a lost count.
                let mut done = 0usize;
                while done < buffer.len() {
                    let at = offset
                        .checked_add(done as u64)
                        .ok_or(Refusal::Fs(Refused::Range))?;
                    let rest = &buffer[done..];
                    match pair
                        .call(async |fs: &mut Face| fs.write_at(file, at, rest).await)
                        .await
                    {
                        Ok(0) => break,
                        Ok(count) => done += count,
                        Err(error) if done == 0 => return Err(error),
                        Err(_) => break,
                    }
                }
                Ok(done as Filesize)
            })
            .await
        })?
        .await
    }

    async fn read_directory(
        &mut self,
        fd: Resource<Descriptor>,
    ) -> FsResult<Resource<DirectoryEntryStream>> {
        let open = self.table.get(&fd)?.open()?;
        open.need(DescriptorFlags::READ, ErrorCode::BadDescriptor)?;
        let dir = open.dir()?;
        let listing = Listing {
            host: open.host.clone(),
            dir,
            cursor: Some(0),
            page: VecDeque::new(),
        };
        let stream = DirectoryEntryStream(Pinned::new(listing).map_err(|_| off_thread())?);
        Ok(self.table.push(stream)?)
    }

    async fn sync(&mut self, fd: Resource<Descriptor>) -> FsResult<()> {
        let fd = self.table.get(&fd)?;
        pinned(async move {
            let open = fd.open()?;
            let file = open.file.clone();
            open.run(async move |pair: &mut Custody| {
                let file = file.view()?;
                pair.call(async |fs: &mut Face| fs.sync(file).await).await
            })
            .await
        })?
        .await
    }

    async fn create_directory_at(
        &mut self,
        fd: Resource<Descriptor>,
        path: String,
    ) -> FsResult<()> {
        let fd = self.table.get(&fd)?;
        pinned(async move {
            let open = fd.open()?;
            let (base, parents, last) = open.edge(&path)?;
            open.run(async move |pair: &mut Custody| {
                let base = base.view()?;
                let walked = parent(pair, base, &parents, MUTATE).await?;
                let dir = at(&walked, base)?;
                let created = pair
                    .call(async |fs: &mut Face| fs.create(dir, last, ObjectKind::Directory).await)
                    .await;
                let created = match created {
                    Ok(view) => {
                        pair.close(view).await;
                        Ok(())
                    }
                    Err(error) => Err(error),
                };
                release(pair, walked).await;
                created
            })
            .await
        })?
        .await
    }

    async fn stat(&mut self, fd: Resource<Descriptor>) -> FsResult<DescriptorStat> {
        let fd = self.table.get(&fd)?;
        pinned(async move { Ok(stat_of(fd.open()?.stat_at(".").await?)) })?.await
    }

    async fn stat_at(
        &mut self,
        fd: Resource<Descriptor>,
        _path_flags: PathFlags,
        path: String,
    ) -> FsResult<DescriptorStat> {
        let fd = self.table.get(&fd)?;
        pinned(async move {
            let open = fd.open()?;
            open.dir()?;
            Ok(stat_of(open.stat_at(&path).await?))
        })?
        .await
    }

    async fn set_times_at(
        &mut self,
        _fd: Resource<Descriptor>,
        _path_flags: PathFlags,
        _path: String,
        _atime: NewTimestamp,
        _mtime: NewTimestamp,
    ) -> FsResult<()> {
        Err(ErrorCode::Unsupported.into())
    }

    async fn link_at(
        &mut self,
        _fd: Resource<Descriptor>,
        _old_path_flags: PathFlags,
        _old_path: String,
        _new_descriptor: Resource<Descriptor>,
        _new_path: String,
    ) -> FsResult<()> {
        Err(ErrorCode::Unsupported.into())
    }

    /// Kore has no symbolic links; `symlink-follow` changes nothing.
    async fn open_at(
        &mut self,
        fd: Resource<Descriptor>,
        _path_flags: PathFlags,
        path: String,
        oflags: OpenFlags,
        flags: DescriptorFlags,
    ) -> FsResult<Resource<Descriptor>> {
        let base = self.table.get(&fd)?;
        let opened =
            pinned(async move { base.open()?.open_at(&path, oflags, flags).await })?.await?;
        Ok(self.table.push(opened)?)
    }

    async fn readlink_at(&mut self, _fd: Resource<Descriptor>, _path: String) -> FsResult<String> {
        Err(ErrorCode::Unsupported.into())
    }

    async fn remove_directory_at(
        &mut self,
        fd: Resource<Descriptor>,
        path: String,
    ) -> FsResult<()> {
        let fd = self.table.get(&fd)?;
        pinned(async move {
            fd.open()?
                .remove_at(&path, ObjectKind::Directory, Refused::NotDirectory)
                .await
        })?
        .await
    }

    async fn rename_at(
        &mut self,
        fd: Resource<Descriptor>,
        old_path: String,
        new_fd: Resource<Descriptor>,
        new_path: String,
    ) -> FsResult<()> {
        let from = self.table.get(&fd)?;
        let to = self.table.get(&new_fd)?;
        pinned(async move {
            let (from, to) = (from.open()?, to.open()?);
            if !Rc::ptr_eq(&from.host, &to.host) {
                return Err(ErrorCode::CrossDevice.into());
            }
            let (src, src_parents, old) = from.edge(&old_path)?;
            let (dst, dst_parents, new) = to.edge(&new_path)?;
            from.run(async move |pair: &mut Custody| {
                let (src, dst) = (src.view()?, dst.view()?);
                let src_walked = parent(pair, src, &src_parents, MUTATE).await?;
                let dst_walked = match parent(pair, dst, &dst_parents, MUTATE).await {
                    Ok(walked) => walked,
                    Err(error) => {
                        release(pair, src_walked).await;
                        return Err(error);
                    }
                };
                let moved = match (at(&src_walked, src), at(&dst_walked, dst)) {
                    (Ok(from), Ok(to)) if from.stable() == to.stable() => {
                        pair.call(async |fs: &mut Face| fs.rename(from, old, new).await)
                            .await
                    }
                    (Ok(from), Ok(to)) => {
                        pair.call(async |fs: &mut Face| fs.relocate(from, old, to, new).await)
                            .await
                    }
                    (Err(error), _) | (_, Err(error)) => Err(error),
                };
                release(pair, src_walked).await;
                release(pair, dst_walked).await;
                moved
            })
            .await
        })?
        .await
    }

    async fn symlink_at(
        &mut self,
        _fd: Resource<Descriptor>,
        _old_path: String,
        _new_path: String,
    ) -> FsResult<()> {
        Err(ErrorCode::Unsupported.into())
    }

    async fn unlink_file_at(&mut self, fd: Resource<Descriptor>, path: String) -> FsResult<()> {
        let fd = self.table.get(&fd)?;
        pinned(async move {
            fd.open()?
                .remove_at(&path, ObjectKind::File, Refused::IsDirectory)
                .await
        })?
        .await
    }

    async fn is_same_object(
        &mut self,
        fd: Resource<Descriptor>,
        other: Resource<Descriptor>,
    ) -> wasmtime::Result<bool> {
        let (Ok(a), Ok(b)) = (
            self.table.get(&fd)?.open().and_then(Open::node_identity),
            self.table.get(&other)?.open().and_then(Open::node_identity),
        ) else {
            return Ok(false);
        };
        Ok(a == b)
    }

    async fn metadata_hash(&mut self, fd: Resource<Descriptor>) -> FsResult<MetadataHashValue> {
        let fd = self.table.get(&fd)?;
        pinned(async move { Ok(hash_of(fd.open()?.stat_at(".").await?)) })?.await
    }

    async fn metadata_hash_at(
        &mut self,
        fd: Resource<Descriptor>,
        _path_flags: PathFlags,
        path: String,
    ) -> FsResult<MetadataHashValue> {
        let fd = self.table.get(&fd)?;
        pinned(async move {
            let open = fd.open()?;
            open.dir()?;
            Ok(hash_of(open.stat_at(&path).await?))
        })?
        .await
    }

    async fn drop(&mut self, fd: Resource<Descriptor>) -> wasmtime::Result<()> {
        let fd = self.table.delete(fd)?;
        // Off the runtime the descriptor is dropped here, its view leaked
        // to the session's teardown (see `Pinned`).
        let close = match Pinned::new(fd.close()) {
            Ok(close) => close,
            Err(_) => return Ok(()),
        };
        close.await;
        Ok(())
    }
}

impl Open {
    /// The object's durable identity, and which filesystem it is in.
    fn node_identity(&self) -> FsResult<(usize, u64)> {
        let stable = self.view()?.stable().get();
        Ok((Rc::as_ptr(&self.host) as usize, stable))
    }
}

impl types::HostDirectoryEntryStream for WasiCtxView<'_> {
    async fn read_directory_entry(
        &mut self,
        stream: Resource<DirectoryEntryStream>,
    ) -> FsResult<Option<DirectoryEntry>> {
        let stream = self.table.get_mut(&stream)?;
        pinned(async move {
            let listing = stream.0.get_mut().ok_or_else(off_thread)?;
            listing.next().await
        })?
        .await
    }

    async fn drop(&mut self, stream: Resource<DirectoryEntryStream>) -> wasmtime::Result<()> {
        self.table.delete(stream)?;
        Ok(())
    }
}

/// The trap a refused preopen is answered with: `get-directories` has no
/// error to return, and an empty list would read as "no files" where the
/// service said why.
fn refused_preopens(refused: host::PreopensRefused) -> wasmtime::Error {
    use host::{GrantsRefused, PreopensRefused};
    match refused {
        PreopensRefused::Root(refusal) => {
            wasmtime::format_err!("the task's root could not be opened for the guest: {refusal:?}")
        }
        PreopensRefused::Grants(GrantsRefused::Listing(refusal)) => wasmtime::format_err!(
            "the task's named grants could not be listed for the guest: {refusal:?}"
        ),
        PreopensRefused::Grants(GrantsRefused::Grant(name, refusal)) => wasmtime::format_err!(
            "the task's grant `{}` could not be opened for the guest: {refusal:?}",
            name.as_str()
        ),
    }
}

impl preopens::Host for WasiCtxView<'_> {
    /// `.` — the task's root view — then, when the embedder offers them,
    /// the task's named grants under their names
    /// ([`Filesystem::with_named_grants`]); nothing for a task without a
    /// filesystem; a trap naming the refusal when one could not be opened.
    /// Every call answers descriptors on the views the first call opened.
    async fn get_directories(&mut self) -> wasmtime::Result<Vec<(Resource<Descriptor>, String)>> {
        let Some(filesystem) = self.ctx.filesystem.as_ref() else {
            return Ok(Vec::new());
        };
        // Bound by a `match`, not awaited in one: the scrutinee's `Err` side
        // is the unpinned future, which must not live across the await.
        let opened = match Pinned::new(async move {
            let host = filesystem.host().ok_or_else(off_thread)?;
            let preopens = host
                .preopens()
                .await
                .map_err(|refused| FsError::trap(refused_preopens(refused)))?;
            let mut directories = Vec::with_capacity(1 + preopens.grants.len());
            let (root, rights) = &preopens.root;
            let root = Descriptor::new(
                host.clone(),
                root.clone(),
                Origin::Root,
                Vec::new(),
                preopen_flags(*rights),
            )
            .ok_or_else(off_thread)?;
            directories.push((String::from("."), root));
            for (name, grant, view, rights) in &preopens.grants {
                let descriptor = Descriptor::new(
                    host.clone(),
                    view.clone(),
                    Origin::Grant(*grant),
                    Vec::new(),
                    preopen_flags(*rights),
                )
                .ok_or_else(off_thread)?;
                directories.push((String::from(name.as_str()), descriptor));
            }
            Ok::<_, FsError>(directories)
        }) {
            Ok(opened) => opened,
            Err(_) => return Ok(Vec::new()),
        };
        let mut directories = Vec::new();
        for (name, descriptor) in opened.await.map_err(FsError::into_trap)? {
            directories.push((self.table.push(descriptor)?, name));
        }
        Ok(directories)
    }
}
