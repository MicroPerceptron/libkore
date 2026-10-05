//! `kore.fs`, and everything `io` and `os` are spellings of.
//!
//! `kore.fs`'s thread-0 half: the open-file table, the positional verbs,
//! and the error vocabulary. Public because the vocabulary is part of the
//! contract — an embedder whose own library can fail a filesystem failure
//! renders it with [`code`] rather than spelling one of its own.
//!
//! The filesystem session lives on thread 0 and a [`View`]
//! is affine, so Lua never holds one. It holds a *handle* — an index and a
//! generation into the [`Files`] table below — and every verb here
//! resolves that handle back to the view before touching the store. A
//! handle whose slot was recycled names a generation that no longer
//! matches, so a use after close is
//! [`FsError::InvalidHandle`] rather
//! than a read of somebody else's file.
//!
//! Every verb here is *positional*. DagFS addresses bytes absolutely
//! (`read_at` / `write_at`), and a stream position is a POSIX idea that
//! belongs to the file object a script holds — which is in C, because
//! `io.open` and `kore.fs.open` answer the *same* object and that object
//! also owns the read-ahead buffer. Keeping a second cursor here would be
//! two authorities for one number, and they would drift the first time a
//! buffered read looked further ahead than the script had consumed.
//!
//! Every verb answers in the three shapes [`Answer`] names,
//! and an expected failure — a missing file, a denied right — is a
//! `Refused` carrying the kebab-case spelling of the
//! [`FsError`] that caused it, never a raise.
//! One vocabulary, two spellings: `kore.fs` renders it as a table and `io`
//! as a message and an errno (`res/SYSTEM_LIBRARY.md`, the six rules).

use alloc::{string::String, vec::Vec};

use kore::fs::{Fs, FsError, Name, Refusal, Rights, View, dag::ObjectKind};

use crate::Answer;

/// How many files one VM may hold open at once.
///
/// A bound, not a policy: the table is a `Vec` that grows to this and no
/// further, so a script that leaks handles is refused at a stated ceiling
/// instead of consuming the task's heap until something else fails. The
/// store's own session handle table is the real limit; this one exists so
/// the refusal names the script's mistake.
const MAX_OPEN_FILES: usize = 256;

/// The largest single read a verb will answer.
///
/// A positional read that wants more asks again, so this bounds one reply
/// and not one file: the reply crosses the bridge as one buffer, and an
/// unbounded one would be an unbounded allocation on thread 0 driven by a
/// script.
const MAX_READ_BYTES: usize = 1 << 20;

/// The largest whole file [`read_file`] will answer.
///
/// [`MAX_READ_BYTES`] bounds a verb a script can call again; this bounds
/// one that has no "again", so it is the ceiling on a single Lua string
/// the store will produce in one call. Past it the answer is
/// [`FsError::Range`]: the file is real and readable, and it is having all
/// of it at once that is refused.
const MAX_FILE_BYTES: usize = 16 << 20;

/// The destination size at which the native facade streams a read through
/// this launch's storage ring instead of copying it a window at a time.
///
/// Not used here — the decision is made in C, where the read-ahead buffer
/// is (`KORE_LUA_STREAM_MIN` in `src/kore_lua.c`) — but pinned here
/// because this is the side that knows what the facade promises. A read
/// below it is a window copy however large the file is, so a surface that
/// only ever asks for a page can never stream.
///
/// The C half pins the same number as `KORE_LUA_FACADE_STREAM_MIN` and its
/// harness asserts the engine compiled it, so a drift on either side fails
/// a build rather than quietly costing every read its ring.
pub const FACADE_STREAM_BYTES: u64 = 256 * 1024;

// A ceiling below the threshold would mean no read this surface can
// express is ever large enough to stream — the defect this path exists to
// remove, and an easy one to reintroduce by lowering a cap for an
// unrelated reason. Compile-time, because a cap is a compile-time choice
// and a test would only catch it after the build that shipped it.
const _: () = assert!(MAX_READ_BYTES as u64 >= FACADE_STREAM_BYTES);
const _: () = assert!(MAX_FILE_BYTES as u64 >= FACADE_STREAM_BYTES);
// A whole file is the verb with no "again", so its ceiling must not be the
// stricter of the two.
const _: () = assert!(MAX_FILE_BYTES >= MAX_READ_BYTES);

/// One open file the VM holds a handle to.
struct OpenFile {
    view: View,
    /// Bumped when the slot is reused, so a handle from a closed file is
    /// detectably stale rather than an alias of whatever took its place.
    generation: u32,
    readable: bool,
    writable: bool,
    /// Every write goes to the current end, whatever offset the caller
    /// passed — `io.open`'s `"a"` modes.
    append: bool,
}

/// The VM's open files.
///
/// Lives in `Services`, so it is created with the filesystem session and
/// retired with it: a handle cannot outlive the session that could close
/// it.
#[derive(Default)]
pub struct Files {
    slots: Vec<Option<OpenFile>>,
    /// Monotonic across the whole table, so a generation is unique to one
    /// open and not merely to one slot's history.
    next_generation: u32,
}

/// A Lua-visible handle: the slot index and the generation that slot held
/// when the file was opened.
///
/// One word, because the transport carries words. The generation is the
/// high half so a handle printed in a diagnostic is legible as a pair.
fn encode(index: usize, generation: u32) -> u64 {
    (u64::from(generation) << 32) | index as u64
}

fn decode(handle: u64) -> (usize, u32) {
    ((handle & 0xffff_ffff) as usize, (handle >> 32) as u32)
}

impl Files {
    fn resolve(&mut self, handle: u64) -> Result<&mut OpenFile, FsError> {
        let (index, generation) = decode(handle);
        match self.slots.get_mut(index) {
            Some(Some(file)) if file.generation == generation => Ok(file),
            _ => Err(FsError::InvalidHandle),
        }
    }

    /// Take the view into a free slot and answer its handle.
    ///
    /// `Err` hands the view **back**, because it is affine: the store still
    /// believes the handle is open, and a refusal that dropped it here
    /// would retire it locally and leak it remotely. The caller owes it a
    /// `close` either way, which is the whole reason this cannot answer a
    /// bare `Option`.
    fn insert(
        &mut self,
        view: View,
        readable: bool,
        writable: bool,
        append: bool,
    ) -> Result<u64, View> {
        let index = match self.slots.iter().position(Option::is_none) {
            Some(index) => index,
            None if self.slots.len() < MAX_OPEN_FILES => {
                self.slots.push(None);
                self.slots.len() - 1
            }
            None => return Err(view),
        };
        let generation = self.next_generation;
        self.slots[index] = Some(OpenFile {
            view,
            generation,
            readable,
            writable,
            append,
        });
        self.next_generation = self.next_generation.wrapping_add(1);
        Ok(encode(index, generation))
    }

    /// Take the file out of the table, leaving the slot free.
    ///
    /// The caller still owes the store a `close`: the view is affine and
    /// dropping it would retire the handle locally while the session still
    /// believes it is open.
    fn take(&mut self, handle: u64) -> Result<View, FsError> {
        let (index, generation) = decode(handle);
        match self.slots.get_mut(index) {
            Some(slot @ Some(_)) if slot.as_ref().is_some_and(|f| f.generation == generation) => {
                Ok(slot.take().expect("just matched Some").view)
            }
            _ => Err(FsError::InvalidHandle),
        }
    }

    /// Whether any file is still open — the REPL's teardown asks, so a
    /// script that forgot `:close()` is closed for it rather than leaking
    /// a session handle.
    pub fn is_empty(&self) -> bool {
        self.slots.iter().all(Option::is_none)
    }

    /// Every live handle, for teardown.
    pub fn drain(&mut self) -> Vec<View> {
        self.slots
            .iter_mut()
            .filter_map(|slot| slot.take().map(|file| file.view))
            .collect()
    }
}

/// The kebab-case spelling of a filesystem error.
///
/// Derived from `FsError`'s own variants rather than invented, which is
/// rule 2: the same word appears in a `kore.fs` error table, in an `io`
/// message, and in the C ABI's `KoreFsError`. The match is exhaustive on
/// purpose — a new variant upstream must be spelled here consciously.
pub fn code(error: FsError) -> &'static str {
    match error {
        FsError::NotFound => "not-found",
        FsError::AlreadyExists => "already-exists",
        FsError::NotDirectory => "not-directory",
        FsError::IsDirectory => "is-directory",
        FsError::InvalidName => "invalid-name",
        FsError::InvalidHandle => "invalid-handle",
        FsError::AccessDenied => "access-denied",
        FsError::AccessLost => "access-lost",
        FsError::Conflict => "conflict",
        FsError::Cycle => "cycle",
        FsError::NotEmpty => "not-empty",
        FsError::OutOfObjects => "out-of-objects",
        FsError::OutOfEdges => "out-of-edges",
        FsError::OutOfHandles => "out-of-handles",
        FsError::OutOfStorage => "out-of-storage",
        FsError::OutOfPlacements => "out-of-placements",
        FsError::OutOfVersionLeases => "out-of-version-leases",
        FsError::OutOfIndexSlots => "out-of-index-slots",
        FsError::Range => "range",
        FsError::BudgetExceeded => "budget-exceeded",
        FsError::IndexIncomplete => "index-incomplete",
        FsError::Unsupported => "unsupported",
        FsError::NoWindow => "no-window",
        FsError::Busy => "busy",
        FsError::ProviderFailure(_) => "provider-failure",
    }
}

/// A refusal in the one vocabulary, whatever layer produced it.
///
/// A transport fault is not an `FsError` — the request may even have
/// committed — so it keeps its own word rather than being folded into the
/// nearest filesystem code, which would tell a caller something untrue
/// about what happened to its write.
fn refusal(error: Refusal) -> Answer {
    Answer::Refused(String::from(match error {
        Refusal::Fs(error) => code(error),
        Refusal::Pair => "transport",
        Refusal::Usage(_) => "usage",
    }))
}

fn refused(error: FsError) -> Answer {
    Answer::Refused(String::from(code(error)))
}

/// The open modes, as the C side packs them into `arg0`.
///
/// Bits rather than a string because the transport carries words; the
/// spelling (`"r"`, `"w+"`, `"a"`) is parsed in C where the Lua argument
/// already is.
pub mod mode {
    pub const READ: u64 = 1 << 0;
    pub const WRITE: u64 = 1 << 1;
    /// Create the file when it does not exist.
    pub const CREATE: u64 = 1 << 2;
    /// Empty it when it does.
    pub const TRUNCATE: u64 = 1 << 3;
    /// Every write goes to the end.
    pub const APPEND: u64 = 1 << 4;
    /// Refuse when it already exists (`os.tmpname`'s guarantee).
    pub const EXCLUSIVE: u64 = 1 << 5;
    /// Make a directory rather than a file.
    pub const DIRECTORY: u64 = 1 << 6;
}

/// How many ASCII letters of a rights spelling one argument word carries.
///
/// `rights_label` spells four bits, so eight is already twice what the
/// store's vocabulary can need. The spelling crosses as bytes rather than
/// as a bit pattern so that [`rights_parse`] stays the one thing that
/// decides what a letter means — the C half packs characters it does not
/// interpret.
///
/// [`rights_parse`]: kore::fs::dag::rights_parse
const RIGHTS_SPELLING_BYTES: usize = 8;

/// The route rights a spelling asks for.
///
/// `None` is a spelling this surface will not open a route with, which the
/// caller renders as an argument error rather than a refusal: both a
/// malformed spelling and an unreachable right are mistakes in the script,
/// not conditions of the store.
///
/// Unreachable, for now, is `MUTATE_NAMESPACE` and `SHARE`. A route
/// carrying either is useless while no verb here can link or share, and
/// accepting the letter would claim a reach this surface does not have.
/// They unlock with the verbs, not before (`docs/kore-fs-vocabulary.md`).
fn route_rights(word: u64) -> Option<Rights> {
    let bytes = word.to_le_bytes();
    let spelling = match bytes.iter().position(|byte| *byte == 0) {
        Some(end) => &bytes[..end],
        None => &bytes[..RIGHTS_SPELLING_BYTES],
    };
    let rights = kore::fs::dag::rights_parse(spelling)?;
    let unreachable = Rights::MUTATE_NAMESPACE.union(Rights::SHARE);
    if rights.intersection(unreachable) != Rights::NONE {
        return None;
    }
    Some(rights)
}

/// Split a path into its parent and final component.
///
/// `None` when the path names the root itself, which has no parent to
/// mutate — refusing that is what stops `os.remove("/")`.
fn split(path: &[u8]) -> Option<(&[u8], &[u8])> {
    let trimmed = path.strip_suffix(b"/").unwrap_or(path);
    if trimmed.is_empty() {
        return None;
    }
    match trimmed.iter().rposition(|byte| *byte == b'/') {
        Some(0) => Some((b"/", &trimmed[1..])),
        Some(cut) => Some((&trimmed[..cut], &trimmed[cut + 1..])),
        None => Some((b".", trimmed)),
    }
}

fn name(bytes: &[u8]) -> Result<Name, FsError> {
    Name::from_bytes(bytes)
}

/// Open — the one verb that creates a handle.
pub async fn open(
    fs: &mut Fs<'_, '_>,
    files: &mut Files,
    path: &[u8],
    flags: u64,
    spelling: u64,
) -> Answer {
    let readable = flags & mode::READ != 0;
    let writable = flags & mode::WRITE != 0;
    let append = flags & mode::APPEND != 0;
    // Two questions, not one. `rights` is what the *route* may do, from the
    // caller's spelling; `readable`/`writable` are what the *handle* may do,
    // from the mode or the same spelling. A write route must be able to
    // read to take a step, which is why `io.open(p, "w")` opens a readable
    // route and still answers an object that refuses `f:read`.
    let Some(rights) = route_rights(spelling) else {
        return Answer::Refused(String::from("usage"));
    };
    // A directory is made, never opened for bytes: `kore.fs.mkdir` is this
    // verb with DIRECTORY set, so one arm creates both kinds and the
    // already-exists answer is identical for each.
    if flags & mode::DIRECTORY != 0 {
        return match create(fs, path, ObjectKind::Directory).await {
            Ok(view) => {
                fs.close(view).await;
                Answer::Value(0, 0)
            }
            Err(error) => refusal(error),
        };
    }
    let existing = fs.open(path, rights).await;
    let view = match existing {
        Ok(view) if flags & mode::EXCLUSIVE != 0 => {
            // The file is there and the caller demanded it not be. Close
            // the view before answering: the refusal must not also leak a
            // session handle.
            fs.close(view).await;
            return refused(FsError::AlreadyExists);
        }
        Ok(view) => view,
        Err(Refusal::Fs(FsError::NotFound)) if flags & mode::CREATE != 0 => {
            match create(fs, path, ObjectKind::File).await {
                Ok(view) => view,
                Err(error) => return refusal(error),
            }
        }
        Err(error) => return refusal(error),
    };
    if view.is_directory() {
        fs.close(view).await;
        return refused(FsError::IsDirectory);
    }
    if flags & mode::TRUNCATE != 0
        && let Err(error) = fs.resize(&view, 0).await
    {
        fs.close(view).await;
        return refusal(error);
    }
    let length = view.len();
    match files.insert(view, readable, writable, append) {
        // The length rides back beside the handle: an append stream starts
        // its position at the end, and the object in C would otherwise
        // have to stat immediately just to learn where it is.
        Ok(handle) => Answer::Value(
            handle,
            if flags & mode::TRUNCATE != 0 {
                0
            } else {
                length
            },
        ),
        Err(view) => {
            // The table is full. Give the store its handle back rather
            // than holding one the VM can no longer name.
            fs.close(view).await;
            refused(FsError::OutOfHandles)
        }
    }
}

/// Create the final component of `path` beneath its parent.
async fn create(fs: &mut Fs<'_, '_>, path: &[u8], kind: ObjectKind) -> Result<View, Refusal> {
    let Some((parent, leaf)) = split(path) else {
        return Err(Refusal::Fs(FsError::AlreadyExists));
    };
    let leaf = name(leaf).map_err(Refusal::Fs)?;
    let parent = fs
        .open(parent, Rights::READ.union(Rights::MUTATE_NAMESPACE))
        .await?;
    let made = fs.create(&parent, leaf, kind).await;
    fs.close(parent).await;
    made
}

pub async fn close(fs: &mut Fs<'_, '_>, files: &mut Files, handle: u64) -> Answer {
    match files.take(handle) {
        Ok(view) => {
            // `close_checked` so a refused close is reported rather than
            // silently dropping a view the session still holds.
            match fs.close_checked(view).await {
                Ok(()) => Answer::Value(0, 0),
                Err(failure) => {
                    // The store kept it. Drop ours — retrying from Lua
                    // would need a handle we have already retired.
                    fs.close(failure.view).await;
                    refusal(failure.error)
                }
            }
        }
        Err(error) => refused(error),
    }
}

/// Where the positional verbs move their bytes.
///
/// A seam, not an abstraction: the only implementation outside this file's
/// tests is [`Session`], which is the task's filesystem face and one handle
/// in the table above. It exists because the rule the two loops below
/// encode — what a refusal answers once some bytes have *already* moved —
/// is reachable only from a session with a device under it, and it is
/// exactly the rule worth pinning.
// One caller, one thread, never spawned: a `Send` bound is not wanted here,
// which is the whole of what this lint is for.
#[allow(async_fn_in_trait)]
trait Transfer {
    /// Read at most one transfer's worth into `dst`, answering how many
    /// bytes arrived. Zero is end of file.
    async fn read_at(&mut self, offset: u64, dst: &mut [u8]) -> Result<usize, Refusal>;

    /// Write at most one transfer's worth, answering how many bytes the
    /// store acknowledged.
    async fn write_at(&mut self, offset: u64, src: &[u8]) -> Result<usize, Refusal>;
}

/// The real one: the filesystem face, and one handle resolved per
/// transfer.
///
/// Per transfer rather than once across a loop because the store's window
/// bounds each one anyway, so the call is already a loop and holding the
/// borrow would only forbid the resolve.
struct Session<'a, 'w, 't> {
    fs: &'a mut Fs<'w, 't>,
    files: &'a mut Files,
    handle: u64,
}

impl Transfer for Session<'_, '_, '_> {
    async fn read_at(&mut self, offset: u64, dst: &mut [u8]) -> Result<usize, Refusal> {
        let view = &self.files.resolve(self.handle).map_err(Refusal::Fs)?.view;
        self.fs.read_at(view, offset, dst).await
    }

    async fn write_at(&mut self, offset: u64, src: &[u8]) -> Result<usize, Refusal> {
        let view = &self.files.resolve(self.handle).map_err(Refusal::Fs)?.view;
        self.fs.write_at(view, offset, src).await
    }
}

/// Fill `dst` from `offset`, answering the prefix that arrived and the
/// refusal that stopped it, if one did.
///
/// **A refusal is reported beside the prefix, never instead of it.** Bytes
/// the store already moved are the caller's; answering only the refusal
/// would throw them away. Nothing is hidden by that — the caller's next
/// call starts at a zero prefix and meets the same refusal — it is only
/// deferred past the bytes that are real.
///
/// The prefix is what was *acknowledged*, which is not the same as what
/// was initialized: a read large enough to stream through the storage ring
/// writes into `dst` as it lands, so a refusal part way can leave bytes
/// past the prefix clobbered. It is a count, never a replay offset.
async fn fill<T: Transfer>(
    transfer: &mut T,
    offset: u64,
    dst: &mut [u8],
) -> (usize, Option<Refusal>) {
    let mut filled = 0;
    while filled < dst.len() {
        match transfer
            .read_at(offset + filled as u64, &mut dst[filled..])
            .await
        {
            Ok(0) => break,
            Ok(count) => filled += count,
            Err(error) => return (filled, Some(error)),
        }
    }
    (filled, None)
}

/// Write all of `src` at `offset`, answering the acknowledged prefix and
/// the refusal that stopped it, if one did.
///
/// [`fill`]'s rule, outward — with one asymmetry that matters: on a
/// transport failure the current unacknowledged chunk may *also* have
/// committed, so the prefix is a lower bound on what changed, not a
/// description of it.
async fn drain<T: Transfer>(transfer: &mut T, offset: u64, src: &[u8]) -> (usize, Option<Refusal>) {
    let mut written = 0;
    while written < src.len() {
        match transfer
            .write_at(offset + written as u64, &src[written..])
            .await
        {
            Ok(0) => break,
            Ok(count) => written += count,
            Err(error) => return (written, Some(error)),
        }
    }
    (written, None)
}

/// Read up to `count` bytes from `offset`.
///
/// Short is not an error: a read that reaches the end answers the bytes it
/// got, and a read entirely past the end answers zero of them. `results[0]`
/// states the count, so a caller tells "read nothing" from "at end of file"
/// without a second call.
pub async fn read(
    fs: &mut Fs<'_, '_>,
    files: &mut Files,
    handle: u64,
    offset: u64,
    count: u64,
) -> Answer {
    let file = match files.resolve(handle) {
        Ok(file) => file,
        Err(error) => return refused(error),
    };
    if !file.readable {
        return refused(FsError::AccessDenied);
    }
    let count = (count as usize).min(MAX_READ_BYTES);
    // `try_reserve`, not `vec![0; count]`: the size is a script's argument
    // and the refusal belongs to the script, not to the task. An abort
    // here would take thread 0 with it.
    let mut buffer = Vec::new();
    if buffer.try_reserve_exact(count).is_err() {
        return refused(FsError::OutOfStorage);
    }
    buffer.resize(count, 0);
    let mut session = Session { fs, files, handle };
    let (filled, stopped) = fill(&mut session, offset, &mut buffer).await;
    if filled == 0
        && let Some(error) = stopped
    {
        return refusal(error);
    }
    buffer.truncate(filled);
    let length = buffer.len() as u64;
    Answer::Bytes(length, buffer)
}

/// A whole file by path, in one call.
///
/// Not a convenience over [`read`]. `Fs::read_file` opens, streams and
/// closes under **one pinned version**, so the answer is one revision of
/// the file rather than a seam of several; and a file of at least
/// [`FACADE_STREAM_BYTES`] crosses by DMA through this launch's storage
/// ring instead of a window at a time. Neither is reachable by looping
/// [`read`] — each call there pins its own version, and a page-sized
/// destination is never large enough to stream.
///
/// Bounded by `MAX_FILE_BYTES` (16 MiB), refused as `range` rather than
/// grown without a ceiling.
pub async fn read_file(fs: &mut Fs<'_, '_>, path: &[u8]) -> Answer {
    let mut bytes: Vec<u8> = Vec::new();
    let mut overflow = false;
    let read = fs
        .read_file(path, |chunk| {
            if bytes.len() + chunk.len() > MAX_FILE_BYTES {
                overflow = true;
                return Err(Refusal::Fs(FsError::Range));
            }
            // The size is the file's, not this task's choice, so a heap
            // that cannot hold it refuses the script rather than aborting
            // thread 0 — `read`'s rule, at a larger scale.
            bytes
                .try_reserve(chunk.len())
                .map_err(|_| Refusal::Fs(FsError::OutOfStorage))?;
            bytes.extend_from_slice(chunk);
            Ok(())
        })
        .await;
    match read {
        _ if overflow => refused(FsError::Range),
        Ok(()) => {
            let length = bytes.len() as u64;
            Answer::Bytes(length, bytes)
        }
        Err(error) => refusal(error),
    }
}

/// Write `bytes` at `offset`, answering how many were acknowledged and
/// where that leaves the end of the written run.
///
/// An append file ignores `offset` entirely and writes at the current end,
/// re-read now rather than remembered from open: another writer may have
/// extended the file since, and an append that overwrote their bytes would
/// not be an append.
pub async fn write(
    fs: &mut Fs<'_, '_>,
    files: &mut Files,
    handle: u64,
    offset: u64,
    bytes: &[u8],
) -> Answer {
    let file = match files.resolve(handle) {
        Ok(file) => file,
        Err(error) => return refused(error),
    };
    if !file.writable {
        return refused(FsError::AccessDenied);
    }
    let start = if file.append {
        match fs.stat(&file.view).await {
            Ok(stat) => stat.len,
            Err(error) => return refusal(error),
        }
    } else {
        offset
    };
    finish_write(fs, files, handle, start, bytes).await
}

async fn finish_write(
    fs: &mut Fs<'_, '_>,
    files: &mut Files,
    handle: u64,
    start: u64,
    bytes: &[u8],
) -> Answer {
    let mut session = Session { fs, files, handle };
    let (written, stopped) = drain(&mut session, start, bytes).await;
    if written == 0
        && let Some(error) = stopped
    {
        return refusal(error);
    }
    Answer::Value(written as u64, start + written as u64)
}

/// Replace an open file's whole contents in one transaction.
///
/// Not a loop of [`write`](fn@write) with a truncation in front. `Fs::write_bytes`
/// opens the store's own replacement (`BeginReplace`), and the hold around
/// it is what makes every window-sized chunk land under **one** commit
/// instead of one commit each — the difference between a save costing one
/// image/flush/root/flush chain and one costing that per window.
///
/// A refused hold is not a failed save: the chunks still commit one at a
/// time, exactly as they did before, so the hold is attempted and its
/// refusal left unreported. The release, though, *is* the commit — a save
/// whose bytes all landed but whose release was refused is not saved, and
/// says so.
///
/// Nothing is bounded here and nothing is allocated: `bytes` is borrowed
/// from the Lua frame the parked worker is holding open and the store
/// chunks it, so the ceiling [`read_file`] needs has no counterpart —
/// there is no reply to build.
///
/// `kore_dagfs::Fs::replace_file` additionally publishes the index for what
/// it wrote. That verb is behind the crate's `index` feature, which this
/// build does not carry, so a save here leaves the index stale. A stated
/// gap, not an oversight: see `docs/pack-backed-reads.md`.
pub async fn save(fs: &mut Fs<'_, '_>, files: &mut Files, handle: u64, bytes: &[u8]) -> Answer {
    let writable = match files.resolve(handle) {
        Ok(file) => file.writable,
        Err(error) => return refused(error),
    };
    if !writable {
        return refused(FsError::AccessDenied);
    }
    let held = fs.hold_commits(true).await.is_ok();
    let view = match files.resolve(handle) {
        Ok(file) => &file.view,
        Err(error) => return refused(error),
    };
    let written = fs.write_bytes(view, bytes).await;
    // Released whatever the write did, so a refused replacement cannot
    // leave this session holding commits for every later writer.
    let released = if held {
        fs.hold_commits(false).await
    } else {
        Ok(())
    };
    if let Err(error) = written {
        return refusal(error);
    }
    if let Err(error) = released {
        return refusal(error);
    }
    let length = bytes.len() as u64;
    Answer::Value(length, length)
}

pub async fn sync(fs: &mut Fs<'_, '_>, files: &mut Files, handle: u64) -> Answer {
    let view = match files.resolve(handle) {
        Ok(file) => &file.view,
        Err(error) => return refused(error),
    };
    match fs.sync(view).await {
        Ok(()) => Answer::Value(0, 0),
        Err(error) => refusal(error),
    }
}

pub async fn resize(fs: &mut Fs<'_, '_>, files: &mut Files, handle: u64, length: u64) -> Answer {
    let file = match files.resolve(handle) {
        Ok(file) => file,
        Err(error) => return refused(error),
    };
    if !file.writable {
        return refused(FsError::AccessDenied);
    }
    match fs.resize(&file.view, length).await {
        Ok(_) => Answer::Value(length, 0),
        Err(error) => refusal(error),
    }
}

/// The fixed record [`stat`] answers, little-endian:
///
/// ```text
///   0  u64  len          the object's size in bytes
///   8  u64  revision     monotonic, committed; 0 is the any-revision sentinel
///  16  u64  stable       volume-stable identity, 0 when the store has none
///  24  u32  links        names pointing at this object
///  28  u32  grants       grants outstanding on it
///  32  u8   kind         0 file, 1 directory
///  33  u8   subtype      carried verbatim, known to this build or not
/// ```
///
/// A record rather than result words because there are eight facts and the
/// transport carries two. The C half checks the length before decoding, so
/// a bridge whose halves disagree refuses rather than reading a field that
/// moved.
const STAT_RECORD_BYTES: usize = 34;

/// The store's own answer, whole.
///
/// `Stat` carries eight facts and this surface used to answer two of them,
/// which is the sense in which it hid the graph: a script could not see
/// the revision it raced, the links that kept an object alive, or the
/// grants outstanding on it. None of this is authority — `StableObjectId`
/// "conveys no authority" and a subtype "gates nothing" — so reporting it
/// widens what a script can know without widening what it can do.
fn stat_record(stat: &kore::fs::Stat) -> Vec<u8> {
    let mut record = Vec::with_capacity(STAT_RECORD_BYTES);
    record.extend_from_slice(&stat.len.to_le_bytes());
    record.extend_from_slice(&stat.revision.to_le_bytes());
    record.extend_from_slice(&stat.stable_object.get().to_le_bytes());
    record.extend_from_slice(&stat.links.to_le_bytes());
    record.extend_from_slice(&stat.grants.to_le_bytes());
    record.push(u8::from(stat.kind.is_dir()));
    record.push(stat.subtype.to_bits());
    debug_assert_eq!(record.len(), STAT_RECORD_BYTES);
    record
}

/// `stat`, by open handle when `path` is empty and by path otherwise.
///
/// Both spellings are needed and neither is redundant: `f:seek("end")` has
/// a handle and must not re-walk a path that may since have been renamed,
/// while `os.rename` wants to know a destination exists without opening it.
///
/// Answers the length and a kind word: 0 file, 1 directory.
pub async fn stat(fs: &mut Fs<'_, '_>, files: &mut Files, handle: u64, path: &[u8]) -> Answer {
    if path.is_empty() {
        let view = match files.resolve(handle) {
            Ok(file) => &file.view,
            Err(error) => return refused(error),
        };
        return match fs.stat(view).await {
            Ok(stat) => Answer::Bytes(stat.len, stat_record(&stat)),
            Err(error) => refusal(error),
        };
    }
    let view = match fs.open(path, Rights::READ).await {
        Ok(view) => view,
        Err(error) => return refusal(error),
    };
    let answer = match fs.stat(&view).await {
        Ok(stat) => Answer::Bytes(stat.len, stat_record(&stat)),
        Err(error) => refusal(error),
    };
    fs.close(view).await;
    answer
}

pub async fn remove(fs: &mut Fs<'_, '_>, path: &[u8]) -> Answer {
    let Some((parent, leaf)) = split(path) else {
        return refused(FsError::AccessDenied);
    };
    let leaf = match name(leaf) {
        Ok(leaf) => leaf,
        Err(error) => return refused(error),
    };
    let parent = match fs
        .open(parent, Rights::READ.union(Rights::MUTATE_NAMESPACE))
        .await
    {
        Ok(view) => view,
        Err(error) => return refusal(error),
    };
    let answer = match fs.unlink(&parent, leaf).await {
        Ok(()) => Answer::Value(0, 0),
        Err(error) => refusal(error),
    };
    fs.close(parent).await;
    answer
}

/// `os.rename`, including across directories — `move_path` is the store's
/// own verb for that, so this is not a copy-and-unlink and it does not
/// lose the object's identity or its grants.
pub async fn rename(fs: &mut Fs<'_, '_>, from: &[u8], to: &[u8]) -> Answer {
    match fs.move_path(from, to).await {
        Ok(_) => Answer::Value(0, 0),
        Err(error) => refusal(error),
    }
}

/// A directory's edges, with what each one points at.
///
/// [`list`] answers the *namespace* — the names a directory holds — and is
/// the right question most of the time. This answers the *graph*: each
/// edge with its target's class and subtype, so a script can tell a
/// directory from a file, or a native module from a plain file, without
/// opening and statting every name in turn.
///
/// One record per entry: a kind byte, a subtype byte, then the name and a
/// NUL. A name is UTF-8 and may contain neither NUL nor `/`, so the
/// terminator is unambiguous.
///
/// The target's `ObjectId` is deliberately **not** here. It is a
/// service-local slot key — "possessing this naming value conveys no
/// authority", and a stale one is a refusal rather than a mistake — but it
/// is also not stable across a restart, and a script that stored one as an
/// identity would be wrong in a way nothing would tell it. The stable
/// identity is [`stat`]'s, where it is named as such.
pub async fn entries(fs: &mut Fs<'_, '_>, path: &[u8]) -> Answer {
    let mut records: Vec<u8> = Vec::new();
    let mut overflow = false;
    let listed = fs
        .list_directory(path, |entry| {
            let name = entry.name.as_bytes();
            if records.len() + name.len() + 3 > MAX_READ_BYTES {
                overflow = true;
                return Err(Refusal::Fs(FsError::Range));
            }
            records.push(u8::from(entry.kind.is_dir()));
            records.push(entry.subtype.to_bits());
            records.extend_from_slice(name);
            records.push(0);
            Ok(())
        })
        .await;
    match listed {
        _ if overflow => refused(FsError::Range),
        Ok(()) => {
            let count = records.iter().filter(|byte| **byte == 0).count() as u64;
            Answer::Bytes(count, records)
        }
        Err(error) => refusal(error),
    }
}

/// The rights this task actually holds on `path`.
///
/// Answered as the store's own compact spelling — a subset of `rwns`, the
/// vocabulary `kore_dagfs::rights_label` prints and `rights_parse` reads —
/// rather than a set of booleans invented here. That matters beyond
/// tidiness: what a listing prints is what a `share` accepts, so a script
/// that reads rights here and hands them back later is speaking the same
/// language the store does. The letters are the contract; the booleans the
/// C half derives from them are a convenience over it.
///
/// This reports authority, it never confers any: `effective_rights` walks
/// to the object with the rights the task already holds, and answers what
/// survived the walk.
pub async fn rights(fs: &mut Fs<'_, '_>, path: &[u8]) -> Answer {
    match fs.effective_rights(path).await {
        Ok(rights) => Answer::Bytes(
            u64::from(rights.bits()),
            kore::fs::dag::rights_label(rights).to_vec(),
        ),
        Err(error) => refusal(error),
    }
}

/// List a directory as NUL-separated names.
///
/// One buffer rather than a cursor per call: a directory listing is the
/// kind of thing a script iterates once, and paging it would put a second
/// piece of mutable state in the handle table for no gain at these sizes.
pub async fn list(fs: &mut Fs<'_, '_>, path: &[u8]) -> Answer {
    let mut names: Vec<u8> = Vec::new();
    let mut overflow = false;
    let listed = fs
        .list_directory(path, |entry| {
            // Stop at the cap rather than growing without bound; the
            // caller is told by `Range` that the listing was not complete.
            if names.len() + entry.name.len() + 1 > MAX_READ_BYTES {
                overflow = true;
                return Err(Refusal::Fs(FsError::Range));
            }
            names.extend_from_slice(entry.name.as_bytes());
            names.push(0);
            Ok(())
        })
        .await;
    match listed {
        _ if overflow => refused(FsError::Range),
        Ok(()) => {
            let count = names.iter().filter(|byte| **byte == 0).count() as u64;
            Answer::Bytes(count, names)
        }
        Err(error) => refusal(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use alloc::vec;
    use core::future::Future;
    use core::task::{Context, Poll, Waker};

    /// Drive a future that never pends.
    ///
    /// [`Scripted`] answers from a list, so one poll always finishes. A
    /// `Pending` here would mean the test grew a real await and needs a
    /// real executor — worth a panic rather than a hang.
    fn answer<F: Future>(future: F) -> F::Output {
        let mut future = core::pin::pin!(future);
        match future
            .as_mut()
            .poll(&mut Context::from_waker(Waker::noop()))
        {
            Poll::Ready(value) => value,
            Poll::Pending => panic!("the scripted transfer never pends"),
        }
    }

    /// A [`Transfer`] that answers a written script.
    ///
    /// The point is the states a real session cannot be put into on
    /// demand: a refusal that lands *after* some bytes have moved, a store
    /// that accepts nothing without refusing, a window that answers short
    /// twice running. Each step is one transfer's answer, in order.
    struct Scripted {
        steps: Vec<Result<usize, Refusal>>,
        /// Stamped into a read's destination, so a test can tell bytes
        /// that were delivered from bytes that were only zeroed.
        mark: u8,
        /// What each transfer was actually asked for: `(offset, len)`.
        asked: Vec<(u64, usize)>,
    }

    impl Scripted {
        fn new(steps: Vec<Result<usize, Refusal>>) -> Self {
            Self {
                steps,
                mark: 0xab,
                asked: Vec::new(),
            }
        }

        fn step(&mut self, offset: u64, len: usize) -> Result<usize, Refusal> {
            assert!(!self.steps.is_empty(), "the script ran out of answers");
            self.asked.push((offset, len));
            match self.steps.remove(0) {
                // Never more than was asked for: a store that overran the
                // destination would be a different bug, tested elsewhere.
                Ok(count) => Ok(count.min(len)),
                Err(error) => Err(error),
            }
        }
    }

    impl Transfer for Scripted {
        async fn read_at(&mut self, offset: u64, dst: &mut [u8]) -> Result<usize, Refusal> {
            let count = self.step(offset, dst.len())?;
            dst[..count].fill(self.mark);
            Ok(count)
        }

        async fn write_at(&mut self, offset: u64, src: &[u8]) -> Result<usize, Refusal> {
            self.step(offset, src.len())
        }
    }

    fn lost() -> Refusal {
        Refusal::Fs(FsError::AccessLost)
    }

    /// The rule S-L3 exists for, in the direction it was wrong.
    #[test]
    fn a_read_refused_after_some_bytes_answers_the_prefix_beside_the_refusal() {
        let mut source = Scripted::new(vec![Ok(4), Err(lost())]);
        let mut dst = [0u8; 8];
        let (filled, stopped) = answer(fill(&mut source, 0, &mut dst));
        assert_eq!(filled, 4);
        assert!(stopped.is_some());
        // The prefix is the caller's: it must still be in the destination,
        // not discarded along with the refusal.
        assert_eq!(&dst[..4], &[0xab; 4]);
        assert_eq!(&dst[4..], &[0; 4]);
    }

    #[test]
    fn a_read_refused_before_anything_moved_answers_only_the_refusal() {
        let mut source = Scripted::new(vec![Err(lost())]);
        let mut dst = [0u8; 8];
        let (filled, stopped) = answer(fill(&mut source, 0, &mut dst));
        // Zero and a reason is what `read` turns back into a refusal; a
        // prefix of zero with a reason is the only case where it may.
        assert_eq!(filled, 0);
        assert!(stopped.is_some());
        assert_eq!(dst, [0; 8]);
    }

    #[test]
    fn a_read_stops_at_end_of_file_without_a_refusal() {
        let mut source = Scripted::new(vec![Ok(3), Ok(0)]);
        let mut dst = [0u8; 8];
        let (filled, stopped) = answer(fill(&mut source, 0, &mut dst));
        assert_eq!(filled, 3);
        // Short is not an error. A caller that cannot tell these apart
        // reports end of file as a failure.
        assert!(stopped.is_none());
    }

    #[test]
    fn a_short_read_resumes_where_the_prefix_left_off() {
        let mut source = Scripted::new(vec![Ok(2), Ok(3), Ok(3)]);
        let mut dst = [0u8; 8];
        let (filled, stopped) = answer(fill(&mut source, 100, &mut dst));
        assert_eq!((filled, stopped.is_none()), (8, true));
        // Each transfer asks from the caller's offset plus what has
        // already arrived, for only what is left.
        assert_eq!(source.asked, vec![(100, 8), (102, 6), (105, 3)]);
    }

    /// The same rule outward. A short write is a fact, not a detail.
    #[test]
    fn a_write_refused_after_some_bytes_answers_the_acknowledged_prefix() {
        let mut sink = Scripted::new(vec![Ok(5), Err(lost())]);
        let (written, stopped) = answer(drain(&mut sink, 0, &[7; 9]));
        assert_eq!(written, 5);
        assert!(stopped.is_some());
    }

    #[test]
    fn a_write_refused_before_anything_committed_answers_only_the_refusal() {
        let mut sink = Scripted::new(vec![Err(lost())]);
        let (written, stopped) = answer(drain(&mut sink, 0, &[7; 9]));
        assert_eq!(written, 0);
        assert!(stopped.is_some());
    }

    #[test]
    fn a_store_that_takes_nothing_stops_rather_than_spinning() {
        // Zero acknowledged and no refusal: the loop must end, or a store
        // under pressure becomes a hang instead of a short write.
        let mut sink = Scripted::new(vec![Ok(4), Ok(0)]);
        let (written, stopped) = answer(drain(&mut sink, 0, &[7; 9]));
        assert_eq!(written, 4);
        assert!(stopped.is_none());
    }

    #[test]
    fn a_write_resumes_where_the_acknowledged_prefix_left_off() {
        let mut sink = Scripted::new(vec![Ok(2), Ok(7)]);
        let (written, stopped) = answer(drain(&mut sink, 64, &[7; 9]));
        assert_eq!((written, stopped.is_none()), (9, true));
        assert_eq!(sink.asked, vec![(64, 9), (66, 7)]);
    }

    #[test]
    fn an_empty_transfer_asks_the_store_nothing() {
        let mut source = Scripted::new(Vec::new());
        let (filled, stopped) = answer(fill(&mut source, 0, &mut []));
        assert_eq!((filled, stopped.is_none()), (0, true));
        let mut sink = Scripted::new(Vec::new());
        let (written, stopped) = answer(drain(&mut sink, 0, &[]));
        assert_eq!((written, stopped.is_none()), (0, true));
        // `Scripted` panics when asked past its script, so reaching here
        // is the assertion: neither loop made a round trip.
    }

    /// Pack a spelling the way the C half does, so these tests exercise
    /// the same word the bridge actually carries.
    fn word(spelling: &str) -> u64 {
        let mut packed = 0u64;
        for (index, byte) in spelling.bytes().enumerate() {
            packed |= u64::from(byte) << (8 * index);
        }
        packed
    }

    #[test]
    fn a_spelling_means_what_the_store_says_it_means() {
        assert_eq!(route_rights(word("r")), Some(Rights::READ));
        assert_eq!(
            route_rights(word("rw")),
            Some(Rights::READ.union(Rights::WRITE))
        );
        // `rights_parse` implies READ — a route that cannot read cannot
        // take a step — so `w` alone is a readable write route. That rule
        // is the store's, which is the whole reason this does not parse
        // the letters itself.
        assert_eq!(
            route_rights(word("w")),
            Some(Rights::READ.union(Rights::WRITE))
        );
    }

    #[test]
    fn an_empty_spelling_is_a_read_only_route() {
        // Zero is what an all-NUL word unpacks to, and it must mean the
        // least authority rather than the most.
        assert_eq!(route_rights(0), Some(Rights::READ));
    }

    #[test]
    fn a_right_this_surface_cannot_exercise_is_refused() {
        // The rule the whole slice turns on: a route carrying SHARE is
        // useless while nothing here can share, and accepting the letter
        // would claim a reach this surface does not have. These unlock
        // with their verbs, not before.
        assert_eq!(route_rights(word("rn")), None);
        assert_eq!(route_rights(word("rs")), None);
        assert_eq!(route_rights(word("rwns")), None);
    }

    #[test]
    fn a_posix_mode_is_not_a_rights_spelling() {
        // Each of these is a mode string someone might reach for out of
        // habit. None may quietly open something: `+`, `b` and `a` are not
        // rights letters, so the caller is told rather than surprised.
        for spelling in ["w+", "r+", "a", "rb", "a+"] {
            assert_eq!(route_rights(word(spelling)), None, "{spelling}");
        }
    }

    #[test]
    fn a_letter_may_not_repeat_and_a_stray_one_is_not_guessed() {
        assert_eq!(route_rights(word("rr")), None);
        assert_eq!(route_rights(word("x")), None);
    }

    #[test]
    fn a_full_word_without_a_terminator_is_read_whole() {
        // Eight letters and no NUL: the unpack must stop at the word's end
        // rather than run past it. Eight distinct rights letters do not
        // exist, so this can only be a refusal — the point is that it is
        // a refusal and not a read of whatever followed.
        assert_eq!(route_rights(u64::from_le_bytes(*b"rwrwrwrw")), None);
    }

    #[test]
    fn a_path_splits_into_a_parent_and_a_leaf() {
        assert_eq!(split(b"/etc/hosts"), Some((&b"/etc"[..], &b"hosts"[..])));
        assert_eq!(split(b"/hosts"), Some((&b"/"[..], &b"hosts"[..])));
        assert_eq!(split(b"hosts"), Some((&b"."[..], &b"hosts"[..])));
        // A trailing slash names the same directory, not an empty leaf.
        assert_eq!(split(b"/etc/"), Some((&b"/"[..], &b"etc"[..])));
    }

    #[test]
    fn the_root_has_no_parent_to_mutate() {
        // This is what refuses `os.remove("/")` before it reaches the store.
        assert_eq!(split(b"/"), None);
        assert_eq!(split(b""), None);
    }

    #[test]
    fn a_handle_survives_a_round_trip_and_a_recycled_slot_does_not_alias() {
        let handle = encode(3, 7);
        assert_eq!(decode(handle), (3, 7));
        // Same slot, later generation: the old handle must not resolve.
        assert_ne!(encode(3, 8), handle);
    }

    #[test]
    fn every_filesystem_error_has_a_kebab_case_spelling() {
        // Rule 2: derived, not invented. A code with an underscore or a
        // capital would mean someone spelled one by hand.
        for error in [
            FsError::NotFound,
            FsError::AlreadyExists,
            FsError::InvalidHandle,
            FsError::OutOfVersionLeases,
        ] {
            let spelling = code(error);
            assert!(!spelling.is_empty());
            assert!(
                spelling
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte == b'-')
            );
        }
    }
}
