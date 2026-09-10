//! C ABI over native runtime APIs and caller-allocated hash tables.
//!
//! The host supplies an attached `Fs`; C cannot manufacture one from a slot.
//! Each callback queues at most one operation and returns to the native async
//! driver. Arguments are copied before returning to C. No C stack pointer is
//! retained while a request is pending. Hash tables retain their caller-provided
//! allocator and entry pointers until explicitly released. See `Session` and
//! the generated header for the separate ownership contracts.
#![no_std]
#![feature(type_alias_impl_trait, allocator_api, thread_local)]
#![deny(unsafe_op_in_unsafe_fn)]

use core::{
    ffi::c_void,
    sync::atomic::{AtomicU64, Ordering},
};
use kore::fs::dag::{self, FsError, Name, ObjectKind, Refusal, Rights};
use kore::fs::{Fs, View};

mod alloc;
pub use alloc::*;
mod futex;
pub use futex::*;
mod hash;
pub use hash::*;
mod parallel;
mod random;
mod session;
mod task;
mod thread;
mod thread_local;
pub use thread_local::*;
mod time;
pub use parallel::*;
pub use random::*;
pub use session::{RunError, Session};
pub use task::*;
pub use thread::*;
pub use time::*;

pub const KORE_ABI_VERSION: u32 = 1;
pub const KORE_FS_HANDLE_CAPACITY: usize = 8;
pub const KORE_FS_BUFFER_CAPACITY: usize = 4096;
pub const KORE_FS_PATH_CAPACITY: usize = 512;
pub const KORE_FS_NAME_CAPACITY: usize = 32;
pub const KORE_FS_ENTRY_CAPACITY: usize = 128;
pub const KORE_FS_READ: u32 = 1;
pub const KORE_FS_WRITE: u32 = 2;
pub const KORE_FS_MUTATE_NAMESPACE: u32 = 4;
pub const KORE_FS_SHARE: u32 = 8;
pub const KORE_FS_FILE: u32 = 0;
pub const KORE_FS_DIRECTORY: u32 = 1;
pub const KORE_OP_START: u32 = 0;
pub const KORE_OP_OPEN: u32 = 1;
pub const KORE_OP_CREATE: u32 = 2;
pub const KORE_OP_READ: u32 = 3;
pub const KORE_OP_WRITE: u32 = 4;
pub const KORE_OP_STAT: u32 = 5;
pub const KORE_OP_RESIZE: u32 = 6;
pub const KORE_OP_SYNC: u32 = 7;
pub const KORE_OP_CLOSE: u32 = 8;
pub const KORE_OP_UNLINK: u32 = 9;
pub const KORE_OP_LIST: u32 = 10;
pub const KORE_OP_DETACH: u32 = 11;
pub const KORE_OP_LINK: u32 = 12;
pub const KORE_OP_SLEEP_FOR: u32 = 13;
pub const KORE_OP_SLEEP_UNTIL: u32 = 14;
pub const KORE_OP_YIELD: u32 = 15;

const _: () =
    assert!(KORE_FS_HANDLE_CAPACITY + dag::PEAK_HANDLES <= dag::bounds::MAX_HANDLES_PER_SESSION);
const _: () = assert!(KORE_FS_NAME_CAPACITY == dag::bounds::MAX_NAME_LEN);
const _: () = assert!(KORE_FS_ENTRY_CAPACITY >= dag::bounds::MAX_DIR_ENTRIES);
const _: () = assert!(KORE_FS_READ == Rights::READ.bits() as u32);
const _: () = assert!(KORE_FS_WRITE == Rights::WRITE.bits() as u32);
const _: () = assert!(KORE_FS_MUTATE_NAMESPACE == Rights::MUTATE_NAMESPACE.bits() as u32);
const _: () = assert!(KORE_FS_SHARE == Rights::SHARE.bits() as u32);

/// Submission status is distinct from the later operation's completion status.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KoreStatus {
    #[default]
    Ok = 0,
    Queued = 1,
    InvalidArgument = 2,
    Busy = 3,
    Capacity = 4,
    StaleHandle = 5,
    Quarantined = 6,
    Refused = 7,
    /// The request may have committed. Retire the session; never replay it.
    Indeterminate = 8,
    BufferTooSmall = 9,
}

/// Versioned foreign error vocabulary, independent of Rust enum layout.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KoreFsError {
    #[default]
    None = 0,
    NotFound = 1,
    AlreadyExists = 2,
    NotDirectory = 3,
    IsDirectory = 4,
    InvalidName = 5,
    InvalidHandle = 6,
    AccessDenied = 7,
    AccessLost = 8,
    Conflict = 9,
    Cycle = 10,
    NotEmpty = 11,
    OutOfObjects = 12,
    OutOfEdges = 13,
    OutOfHandles = 14,
    OutOfStorage = 15,
    OutOfPlacements = 16,
    OutOfVersionLeases = 17,
    OutOfIndexSlots = 18,
    Range = 19,
    BudgetExceeded = 20,
    IndexIncomplete = 21,
    Unsupported = 22,
    NoWindow = 23,
    Busy = 24,
    ProviderFailure = 25,
    Unknown = 255,
}

/// Provider failures retain their native diagnostic class.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KoreProviderError {
    #[default]
    None = 0,
    Exhausted = 1,
    PlacementsFull = 2,
    UnknownKey = 3,
    Range = 4,
    TooLarge = 5,
    Media = 6,
    Transport = 7,
    Integrity = 8,
    Cancelled = 9,
    Revoked = 10,
    Lost = 11,
    Draining = 12,
    ReadOnly = 13,
    Backpressure = 14,
    Timeout = 15,
}

/// Scoped generational name. Copying its bits does not duplicate native ownership.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KoreFsHandle {
    pub owner: u64,
    pub slot: u32,
    pub generation: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct KoreFsCompletion {
    pub abi_version: u32,
    pub operation: u32,
    pub status: KoreStatus,
    pub error: KoreFsError,
    pub handle: KoreFsHandle,
    /// Confirmed bytes for reads/writes; entry count for list.
    pub count: u64,
    pub length: u64,
    pub revision: u64,
    /// Correlation data only, never authority.
    pub stable_object: u64,
    pub kind: u32,
    pub next_cursor: u32,
    pub has_next: u32,
    pub provider_error: KoreProviderError,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct KoreFsEntry {
    pub name: [u8; KORE_FS_NAME_CAPACITY],
    pub name_length: u32,
    pub kind: u32,
}

const _: () = assert!(core::mem::size_of::<KoreFsHandle>() == 16);
const _: () = assert!(core::mem::size_of::<KoreFsCompletion>() == 80);
const _: () = assert!(core::mem::offset_of!(KoreFsCompletion, count) == 32);

/// Called only on the native task, outside any outstanding filesystem request.
/// Queue one operation and return; queue none to finish and close owned views.
/// Context, completion and user pointers must not escape the callback or be
/// accessed concurrently. Do not unwind/longjmp across this boundary.
pub type KoreFsStep =
    unsafe extern "C" fn(*mut KoreFsContext, *const KoreFsCompletion, *mut c_void);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Callback,
    Queued,
    InFlight,
    Finished,
    Quarantined,
}

#[derive(Clone, Copy)]
struct Slot {
    occupied: bool,
    generation: u32,
}
const EMPTY_SLOT: Slot = Slot {
    occupied: false,
    generation: 1,
};

#[derive(Clone, Copy)]
struct Command {
    op: u32,
    slot: usize,
    destination: usize,
    offset: u64,
    length: usize,
    rights: Rights,
    name: Option<Name>,
    kind: ObjectKind,
}
impl Command {
    fn new(op: u32) -> Self {
        Self {
            op,
            slot: 0,
            destination: 0,
            offset: 0,
            length: 0,
            rights: Rights::READ,
            name: None,
            kind: ObjectKind::File,
        }
    }
}

/// Opaque callback-local access. Its representation is not a foreign ABI.
/// Only Session creates it from an actual attached native filesystem owner.
pub struct KoreFsContext {
    owner: u64,
    phase: Phase,
    slots: [Slot; KORE_FS_HANDLE_CAPACITY],
    command: Option<Command>,
    path: [u8; KORE_FS_PATH_CAPACITY],
    data: [u8; KORE_FS_BUFFER_CAPACITY],
    entries: [KoreFsEntry; KORE_FS_ENTRY_CAPACITY],
    completion: KoreFsCompletion,
}

static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

impl KoreFsContext {
    fn new() -> Option<Self> {
        // Exhaustion is permanent; no owner identifier can wrap onto a live one.
        let owner = NEXT_OWNER
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .ok()?;
        Some(Self {
            owner,
            phase: Phase::Finished,
            slots: [EMPTY_SLOT; KORE_FS_HANDLE_CAPACITY],
            command: None,
            path: [0; KORE_FS_PATH_CAPACITY],
            data: [0; KORE_FS_BUFFER_CAPACITY],
            entries: [KoreFsEntry::default(); KORE_FS_ENTRY_CAPACITY],
            completion: KoreFsCompletion {
                abi_version: KORE_ABI_VERSION,
                ..Default::default()
            },
        })
    }
    fn ready(&self) -> Result<(), KoreStatus> {
        match self.phase {
            Phase::Callback => Ok(()),
            Phase::Queued => Err(KoreStatus::Busy),
            _ => Err(KoreStatus::Quarantined),
        }
    }
    fn index(&self, handle: KoreFsHandle) -> Result<usize, KoreStatus> {
        let slot = self
            .slots
            .get(handle.slot as usize)
            .ok_or(KoreStatus::StaleHandle)?;
        if handle.owner != self.owner || !slot.occupied || slot.generation != handle.generation {
            return Err(KoreStatus::StaleHandle);
        }
        Ok(handle.slot as usize)
    }
    fn vacant(&self) -> Result<usize, KoreStatus> {
        self.slots
            .iter()
            .position(|s| !s.occupied && s.generation != 0)
            .ok_or(KoreStatus::Capacity)
    }
    fn handle(&self, slot: usize) -> KoreFsHandle {
        KoreFsHandle {
            owner: self.owner,
            slot: slot as u32,
            generation: self.slots[slot].generation,
        }
    }
    fn retire(&mut self, slot: usize) {
        self.slots[slot].occupied = false;
        self.slots[slot].generation = self.slots[slot].generation.checked_add(1).unwrap_or(0);
    }
    fn queue(&mut self, command: Command) -> KoreStatus {
        self.command = Some(command);
        self.phase = Phase::Queued;
        KoreStatus::Queued
    }
}

// FFI calls are serialized by the callback contract. These helpers check nulls
// and bounds; validity of non-null foreign pointers remains the caller's duty.
unsafe fn context<'a>(ctx: *mut KoreFsContext) -> Result<&'a mut KoreFsContext, KoreStatus> {
    // SAFETY: every exported call requires a live, exclusive callback context.
    let ctx = unsafe { ctx.as_mut() }.ok_or(KoreStatus::InvalidArgument)?;
    ctx.ready()?;
    Ok(ctx)
}
unsafe fn bytes<'a>(ptr: *const u8, len: usize) -> Result<&'a [u8], KoreStatus> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() || len > isize::MAX as usize {
        return Err(KoreStatus::InvalidArgument);
    }
    // SAFETY: the exported call requires len readable bytes, held for this call.
    Ok(unsafe { core::slice::from_raw_parts(ptr, len) })
}
macro_rules! take {
    ($e:expr) => {
        match $e {
            Ok(value) => value,
            Err(error) => return error,
        }
    };
}

/// Open a path within this task's endowed namespace; empty path means its root.
/// # Safety
/// ctx is the live exclusive callback context; path names length readable bytes disjoint from ctx.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_fs_open(
    ctx: *mut KoreFsContext,
    path: *const u8,
    length: usize,
    rights: u32,
) -> KoreStatus {
    let ctx = take!(unsafe { context(ctx) });
    if length > KORE_FS_PATH_CAPACITY || rights > u8::MAX as u32 {
        return KoreStatus::InvalidArgument;
    }
    let Some(rights) = Rights::from_bits_exact(rights as u8) else {
        return KoreStatus::InvalidArgument;
    };
    let destination = take!(ctx.vacant());
    let path = take!(unsafe { bytes(path, length) });
    ctx.path[..length].copy_from_slice(path);
    let mut command = Command::new(KORE_OP_OPEN);
    command.length = length;
    command.rights = rights;
    command.destination = destination;
    ctx.queue(command)
}

/// Create a file or directory beneath an owned directory view.
/// # Safety
/// ctx is the live exclusive callback context; name names length readable bytes disjoint from ctx.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_fs_create(
    ctx: *mut KoreFsContext,
    parent: KoreFsHandle,
    name: *const u8,
    length: usize,
    kind: u32,
) -> KoreStatus {
    let ctx = take!(unsafe { context(ctx) });
    let slot = take!(ctx.index(parent));
    let destination = take!(ctx.vacant());
    if length > KORE_FS_NAME_CAPACITY {
        return KoreStatus::InvalidArgument;
    }
    let name = take!(unsafe { bytes(name, length) });
    let Ok(name) = Name::from_bytes(name) else {
        return KoreStatus::InvalidArgument;
    };
    let kind = match kind {
        KORE_FS_FILE => ObjectKind::File,
        KORE_FS_DIRECTORY => ObjectKind::Directory,
        _ => return KoreStatus::InvalidArgument,
    };
    let mut command = Command::new(KORE_OP_CREATE);
    command.slot = slot;
    command.destination = destination;
    command.name = Some(name);
    command.kind = kind;
    ctx.queue(command)
}

/// Read one bounded transfer; retrieve acknowledged bytes in the next callback.
/// # Safety
/// ctx is the live exclusive callback context.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_fs_read(
    ctx: *mut KoreFsContext,
    handle: KoreFsHandle,
    offset: u64,
    length: usize,
) -> KoreStatus {
    let ctx = take!(unsafe { context(ctx) });
    let slot = take!(ctx.index(handle));
    if length > KORE_FS_BUFFER_CAPACITY || offset.checked_add(length as u64).is_none() {
        return KoreStatus::InvalidArgument;
    }
    let mut command = Command::new(KORE_OP_READ);
    command.slot = slot;
    command.offset = offset;
    command.length = length;
    ctx.queue(command)
}

/// Copy and queue one bounded write. Does not truncate, sync, or retry.
/// # Safety
/// ctx is the live exclusive callback context; data names length readable bytes disjoint from ctx.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_fs_write(
    ctx: *mut KoreFsContext,
    handle: KoreFsHandle,
    offset: u64,
    data: *const u8,
    length: usize,
) -> KoreStatus {
    let ctx = take!(unsafe { context(ctx) });
    let slot = take!(ctx.index(handle));
    if length > KORE_FS_BUFFER_CAPACITY || offset.checked_add(length as u64).is_none() {
        return KoreStatus::InvalidArgument;
    }
    let data = take!(unsafe { bytes(data, length) });
    ctx.data[..length].copy_from_slice(data);
    let mut command = Command::new(KORE_OP_WRITE);
    command.slot = slot;
    command.offset = offset;
    command.length = length;
    ctx.queue(command)
}

/// Copy a completed read. Too-small output is refused without a partial copy.
/// # Safety
/// ctx is the live exclusive callback context; dst names capacity writable bytes
/// disjoint from the context. No buffer is retained after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_fs_read_data(
    ctx: *mut KoreFsContext,
    dst: *mut u8,
    capacity: usize,
) -> KoreStatus {
    let ctx = take!(unsafe { context(ctx) });
    if ctx.completion.operation != KORE_OP_READ || ctx.completion.status != KoreStatus::Ok {
        return KoreStatus::InvalidArgument;
    }
    let count = ctx.completion.count as usize;
    if capacity < count {
        return KoreStatus::BufferTooSmall;
    }
    if count == 0 {
        return KoreStatus::Ok;
    }
    if dst.is_null() {
        return KoreStatus::InvalidArgument;
    }
    // SAFETY: caller provides count writable bytes disjoint from owner storage.
    unsafe {
        core::ptr::copy_nonoverlapping(ctx.data.as_ptr(), dst, count);
    }
    KoreStatus::Ok
}

unsafe fn simple(
    ctx: *mut KoreFsContext,
    handle: KoreFsHandle,
    op: u32,
    argument: u64,
) -> KoreStatus {
    let ctx = take!(unsafe { context(ctx) });
    let mut command = Command::new(op);
    command.slot = take!(ctx.index(handle));
    command.offset = argument;
    ctx.queue(command)
}

/// Refresh metadata from the service.
/// # Safety
/// ctx is the live exclusive callback context.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_fs_stat(ctx: *mut KoreFsContext, handle: KoreFsHandle) -> KoreStatus {
    unsafe { simple(ctx, handle, KORE_OP_STAT, 0) }
}
/// Explicitly set the file length.
/// # Safety
/// ctx is the live exclusive callback context.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_fs_resize(
    ctx: *mut KoreFsContext,
    handle: KoreFsHandle,
    length: u64,
) -> KoreStatus {
    unsafe { simple(ctx, handle, KORE_OP_RESIZE, length) }
}
/// Request the native sync operation.
/// # Safety
/// ctx is the live exclusive callback context.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_fs_sync(ctx: *mut KoreFsContext, handle: KoreFsHandle) -> KoreStatus {
    unsafe { simple(ctx, handle, KORE_OP_SYNC, 0) }
}
/// Close only invalidates the foreign handle after the native acknowledgement.
/// # Safety
/// ctx is the live exclusive callback context.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_fs_close(
    ctx: *mut KoreFsContext,
    handle: KoreFsHandle,
) -> KoreStatus {
    unsafe { simple(ctx, handle, KORE_OP_CLOSE, 0) }
}
/// List a native page; inspect each entry before queuing the next operation.
/// # Safety
/// ctx is the live exclusive callback context.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_fs_list(
    ctx: *mut KoreFsContext,
    handle: KoreFsHandle,
    cursor: u32,
) -> KoreStatus {
    unsafe { simple(ctx, handle, KORE_OP_LIST, cursor as u64) }
}
/// Remove an edge using native compatibility unlink semantics.
/// # Safety
/// ctx is the live exclusive callback context; name names length readable bytes disjoint from ctx.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_fs_unlink(
    ctx: *mut KoreFsContext,
    parent: KoreFsHandle,
    name: *const u8,
    length: usize,
) -> KoreStatus {
    let ctx = take!(unsafe { context(ctx) });
    let slot = take!(ctx.index(parent));
    if length > KORE_FS_NAME_CAPACITY {
        return KoreStatus::InvalidArgument;
    }
    let Ok(name) = Name::from_bytes(take!(unsafe { bytes(name, length) })) else {
        return KoreStatus::InvalidArgument;
    };
    let mut command = Command::new(KORE_OP_UNLINK);
    command.slot = slot;
    command.name = Some(name);
    ctx.queue(command)
}
/// Detach a native DAG edge, including a nonempty directory. Other routes survive.
/// # Safety
/// ctx is the live exclusive callback context; name names length readable bytes
/// disjoint from ctx.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_dag_detach(
    ctx: *mut KoreFsContext,
    parent: KoreFsHandle,
    name: *const u8,
    length: usize,
) -> KoreStatus {
    let ctx = take!(unsafe { context(ctx) });
    let slot = take!(ctx.index(parent));
    if length > KORE_FS_NAME_CAPACITY {
        return KoreStatus::InvalidArgument;
    }
    let Ok(name) = Name::from_bytes(take!(unsafe { bytes(name, length) })) else {
        return KoreStatus::InvalidArgument;
    };
    let mut command = Command::new(KORE_OP_DETACH);
    command.slot = slot;
    command.name = Some(name);
    ctx.queue(command)
}

/// Project another native name for an owned child; never copies file bytes.
/// # Safety
/// ctx is the live exclusive callback context; name names length readable bytes
/// disjoint from ctx.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_dag_link(
    ctx: *mut KoreFsContext,
    parent: KoreFsHandle,
    name: *const u8,
    length: usize,
    child: KoreFsHandle,
) -> KoreStatus {
    let ctx = take!(unsafe { context(ctx) });
    let slot = take!(ctx.index(parent));
    let destination = take!(ctx.index(child));
    if length > KORE_FS_NAME_CAPACITY {
        return KoreStatus::InvalidArgument;
    }
    let Ok(name) = Name::from_bytes(take!(unsafe { bytes(name, length) })) else {
        return KoreStatus::InvalidArgument;
    };
    let mut command = Command::new(KORE_OP_LINK);
    command.slot = slot;
    command.destination = destination;
    command.name = Some(name);
    ctx.queue(command)
}

/// Copy an entry from the last successfully completed list page.
/// # Safety
/// ctx is the live exclusive callback context; out is aligned writable storage
/// disjoint from ctx and remains valid for this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_fs_entry(
    ctx: *mut KoreFsContext,
    index: u32,
    out: *mut KoreFsEntry,
) -> KoreStatus {
    let ctx = take!(unsafe { context(ctx) });
    if out.is_null()
        || ctx.completion.operation != KORE_OP_LIST
        || ctx.completion.status != KoreStatus::Ok
        || u64::from(index) >= ctx.completion.count
    {
        return KoreStatus::InvalidArgument;
    }
    // SAFETY: caller guarantees aligned writable storage, disjoint from ctx.
    unsafe {
        out.write(ctx.entries[index as usize]);
    }
    KoreStatus::Ok
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opened() -> KoreFsContext {
        let mut ctx = KoreFsContext::new().unwrap();
        ctx.phase = Phase::Callback;
        // Exercise foreign naming/admission independently of native transport.
        // Only Session::install can publish these slots in production.
        ctx.slots[0].occupied = true;
        ctx
    }

    #[test]
    fn foreign_names_are_scoped_and_retired_generations_never_recur() {
        let mut one = opened();
        let two = opened();
        let handle = one.handle(0);
        assert_eq!(two.index(handle), Err(KoreStatus::StaleHandle));
        one.retire(0);
        assert_eq!(one.index(handle), Err(KoreStatus::StaleHandle));
        one.slots[0].occupied = true;
        assert_ne!(one.handle(0), handle);
        one.slots[0].generation = u32::MAX;
        one.retire(0);
        assert_eq!(one.vacant(), Ok(1));
    }

    #[test]
    fn a_queued_write_owns_its_copy_and_rejects_replacement_before_touching_bytes() {
        let mut ctx = opened();
        let handle = ctx.handle(0);
        let mut input = [1, 2, 3];
        // SAFETY: exclusive live context and disjoint readable local arrays.
        unsafe {
            assert_eq!(
                kore_fs_write(&mut ctx, handle, 42, input.as_ptr(), 3),
                KoreStatus::Queued
            );
            input.fill(9);
            assert_eq!(
                kore_fs_write(&mut ctx, handle, 7, input.as_ptr(), 3),
                KoreStatus::Busy
            );
            assert_eq!(kore_fs_close(&mut ctx, handle), KoreStatus::Busy);
        }
        assert_eq!(&ctx.data[..3], &[1, 2, 3]);
        assert_eq!(ctx.command.unwrap().offset, 42);
        assert_eq!(ctx.index(handle), Ok(0)); // queueing a close did not consume it
    }

    #[test]
    fn bad_ranges_rights_names_and_capacity_publish_nothing() {
        let mut ctx = opened();
        let handle = ctx.handle(0);
        // SAFETY: buffers are local and disjoint; zero length permits null.
        unsafe {
            assert_eq!(
                kore_fs_read(&mut ctx, handle, u64::MAX, 1),
                KoreStatus::InvalidArgument
            );
            assert_eq!(
                kore_fs_open(&mut ctx, core::ptr::null(), 0, 256),
                KoreStatus::InvalidArgument
            );
            assert_eq!(
                kore_fs_create(&mut ctx, handle, b"..".as_ptr(), 2, KORE_FS_FILE),
                KoreStatus::InvalidArgument
            );
            assert_eq!(
                kore_fs_write(&mut ctx, handle, 0, core::ptr::null(), 1),
                KoreStatus::InvalidArgument
            );
            ctx.slots.fill(Slot {
                occupied: true,
                generation: 1,
            });
            assert_eq!(
                kore_fs_open(&mut ctx, core::ptr::null(), 0, KORE_FS_READ),
                KoreStatus::Capacity
            );
        }
        assert!(ctx.command.is_none());
        assert_eq!(ctx.phase, Phase::Callback);
    }

    #[test]
    fn graph_projection_cannot_smuggle_a_handle_from_another_owner() {
        let mut ctx = opened();
        let other = opened();
        let parent = ctx.handle(0);
        // SAFETY: live context, disjoint static name; foreign handle is data
        // whose ownership check must fail before native publication.
        assert_eq!(
            unsafe { kore_dag_link(&mut ctx, parent, b"link".as_ptr(), 4, other.handle(0)) },
            KoreStatus::StaleHandle
        );
        assert!(ctx.command.is_none());
    }

    #[test]
    fn result_buffers_are_readable_only_before_the_next_submission() {
        let mut ctx = opened();
        ctx.completion.operation = KORE_OP_READ;
        ctx.completion.count = 3;
        ctx.data[..3].copy_from_slice(&[3, 4, 5]);
        let handle = ctx.handle(0);
        let mut output = [99; 3];
        // SAFETY: writable disjoint array, valid live context.
        unsafe {
            assert_eq!(
                kore_fs_read_data(&mut ctx, output.as_mut_ptr(), 2),
                KoreStatus::BufferTooSmall
            );
            assert_eq!(output, [99; 3]);
            assert_eq!(
                kore_fs_read_data(&mut ctx, output.as_mut_ptr(), 3),
                KoreStatus::Ok
            );
            assert_eq!(output, [3, 4, 5]);
            assert_eq!(kore_fs_stat(&mut ctx, handle), KoreStatus::Queued);
            assert_eq!(
                kore_fs_read_data(&mut ctx, output.as_mut_ptr(), 3),
                KoreStatus::Busy
            );
        }
    }

    #[test]
    fn in_flight_and_quarantined_contexts_cannot_publish_another_request() {
        for phase in [Phase::InFlight, Phase::Quarantined] {
            let mut ctx = opened();
            ctx.phase = phase;
            let handle = ctx.handle(0);
            // SAFETY: the context remains valid; its state must deny admission.
            assert_eq!(
                unsafe { kore_fs_close(&mut ctx, handle) },
                KoreStatus::Quarantined
            );
            assert!(ctx.command.is_none());
        }
    }

    #[cfg(all(feature = "test-foreign", not(miri), not(target_os = "none")))]
    #[test]
    fn compiled_c_and_translated_zig_cross_the_actual_export_boundary() {
        // C only sees an incomplete struct and exchanges thin pointers. Its
        // pointee layout is deliberately private; no Rust-layout value crosses.
        #[allow(improper_ctypes)]
        unsafe extern "C" {
            fn c_submit_write(ctx: *mut KoreFsContext, handle: KoreFsHandle) -> u32;
            fn c_read_completion(ctx: *mut KoreFsContext, result: *const KoreFsCompletion) -> u32;
            fn c_sleep_and_random(ctx: *mut KoreFsContext) -> u32;
            fn zig_check_random() -> u32;
            fn c_check_teardown_api() -> u32;
            fn zig_check_teardown_api() -> u32;
            fn c_check_futex() -> u32;
            fn zig_check_futex() -> u32;
            fn c_check_thread_refusal() -> u32;
            fn zig_check_thread_refusal() -> u32;
            fn zig_submit_create(ctx: *mut KoreFsContext, handle: KoreFsHandle) -> u32;
        }
        let mut ctx = opened();
        let handle = ctx.handle(0);
        // SAFETY: the compiled fixtures obey the callback ABI and use bounded
        // local buffers; completion is copied outside mutable context storage.
        unsafe {
            assert_eq!(c_submit_write(&mut ctx, handle), 0);
            assert_eq!(&ctx.data[..4], &[17, 23, 31, 47]);
            ctx.phase = Phase::Callback;
            ctx.command = None;
            ctx.completion.operation = KORE_OP_READ;
            ctx.completion.count = 4;
            let completion = ctx.completion;
            assert_eq!(c_read_completion(&mut ctx, &completion), 0);
            assert_eq!(zig_submit_create(&mut ctx, handle), 0);
            let command = ctx.command.unwrap();
            assert_eq!(command.op, KORE_OP_CREATE);
            assert_eq!(command.destination, 1);
            assert_eq!(command.name.unwrap().as_bytes(), b"zig-file");
            ctx.phase = Phase::Callback;
            ctx.command = None;
            assert_eq!(c_sleep_and_random(&mut ctx), 0);
            assert_eq!(ctx.command.unwrap().op, KORE_OP_SLEEP_FOR);
            assert_eq!(ctx.command.unwrap().offset, 1_250_000);
            assert_eq!(zig_check_random(), 0);
            assert_eq!(c_check_teardown_api(), 0);
            assert_eq!(zig_check_teardown_api(), 0);
            assert_eq!(c_check_futex(), 0);
            assert_eq!(zig_check_futex(), 0);
            assert_eq!(c_check_thread_refusal(), 0);
            assert_eq!(zig_check_thread_refusal(), 0);
        }
    }
}
