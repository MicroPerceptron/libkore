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
/// `f:read("a")` on a big file loops rather than asking for all of it at
/// once: the reply crosses the bridge as one buffer, and an unbounded one
/// would be an unbounded allocation on thread 0 driven by a script.
const MAX_READ_BYTES: usize = 1 << 20;

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
pub async fn open(fs: &mut Fs<'_, '_>, files: &mut Files, path: &[u8], flags: u64) -> Answer {
    let readable = flags & mode::READ != 0;
    let writable = flags & mode::WRITE != 0;
    let append = flags & mode::APPEND != 0;
    let mut rights = Rights::NONE;
    if readable {
        rights = rights.union(Rights::READ);
    }
    if writable {
        rights = rights.union(Rights::WRITE);
    }
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
    let existing = fs.open(path, rights.union(Rights::READ)).await;
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
    let mut filled = 0;
    while filled < count {
        match fs
            .read_at(&file.view, offset + filled as u64, &mut buffer[filled..])
            .await
        {
            Ok(0) => break,
            Ok(got) => filled += got,
            Err(error) => return refusal(error),
        }
    }
    buffer.truncate(filled);
    let length = buffer.len() as u64;
    Answer::Bytes(length, buffer)
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
    let mut written = 0;
    while written < bytes.len() {
        let view = match files.resolve(handle) {
            Ok(file) => &file.view,
            Err(error) => return refused(error),
        };
        // The view is borrowed per chunk rather than across the loop: the
        // store's window bounds each transfer, so the call is already a
        // loop and holding the borrow would only forbid the resolve.
        match fs
            .write_at(view, start + written as u64, &bytes[written..])
            .await
        {
            Ok(0) => break,
            Ok(count) => written += count,
            Err(error) => {
                // A short write is a fact, not a detail: report the
                // acknowledged prefix so a caller knows what committed.
                if written == 0 {
                    return refusal(error);
                }
                break;
            }
        }
    }
    Answer::Value(written as u64, start + written as u64)
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
        let directory = view.is_directory();
        return match fs.stat(view).await {
            Ok(stat) => Answer::Value(stat.len, u64::from(directory)),
            Err(error) => refusal(error),
        };
    }
    let view = match fs.open(path, Rights::READ).await {
        Ok(view) => view,
        Err(error) => return refusal(error),
    };
    let directory = view.is_directory();
    let answer = match fs.stat(&view).await {
        Ok(stat) => Answer::Value(stat.len, u64::from(directory)),
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
