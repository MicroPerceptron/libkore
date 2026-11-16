//! `FILE` streams: the standard pair, and the streams a consumer builds over
//! its own file layer.
//!
//! A [`File`] is a write operation and its state. C only ever holds a
//! `FILE *` to one (musl's `FILE` is incomplete in the headers every kore C
//! build uses), so the layout is this crate's own.
//!
//! - **`stdout` and `stderr`** go to the [`StandardSink`] the process
//!   installs once ([`install_standard_sink`]) — its terminal, or a result
//!   buffer it reports when the task ends. Until one is installed a write
//!   reports `EOF` and goes nowhere, which is what every consumer that never
//!   installs one has always seen.
//! - **Any other stream** is built by the consumer that owns a file layer
//!   (`fdopen` over its own descriptors) with [`File::new`] and its own
//!   [`FileOps`]. This crate never opens a file: kore has no ambient
//!   filesystem for it to open one in.

use core::ffi::c_void;
use core::ptr;
use core::sync::atomic::{AtomicPtr, Ordering};

/// A C `FILE`.
#[repr(C)]
pub struct File {
    ops: &'static FileOps,
    state: *mut c_void,
}

// SAFETY: the standard streams' state is an inert tag, never dereferenced;
// a consumer that places its own stream in a `static` owns that stream's
// synchronization, as C's own `FILE` locking would.
unsafe impl Sync for File {}

/// What a stream does with its bytes.
pub struct FileOps {
    /// Accept `bytes` whole. `false` is a write error: C's `EOF`.
    pub write: unsafe fn(state: *mut c_void, bytes: &[u8]) -> bool,
    /// Push anything buffered onward. `false` is an error.
    pub flush: unsafe fn(state: *mut c_void) -> bool,
    /// `fclose`'s last act: flush and release the stream, the `File` itself
    /// included when its owner allocated it. `false` is an error; the stream
    /// is closed either way, as C's `fclose` leaves it.
    pub close: unsafe fn(file: *mut File) -> bool,
}

impl File {
    /// A stream over `ops`, carrying the consumer's `state`.
    pub const fn new(ops: &'static FileOps, state: *mut c_void) -> Self {
        Self { ops, state }
    }

    /// The consumer's state, for its [`FileOps`].
    pub const fn state(&self) -> *mut c_void {
        self.state
    }
}

/// Write `bytes` to the stream at `file`.
///
/// # Safety
/// `file` is a live `FILE *` this crate or a consumer constructed.
pub(crate) unsafe fn write(file: *mut File, bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return true;
    }
    // SAFETY: the caller's contract; only the fields are read.
    unsafe { ((*file).ops.write)((*file).state, bytes) }
}

/// # Safety
/// As [`write`].
pub(crate) unsafe fn flush(file: *mut File) -> bool {
    // SAFETY: the caller's contract.
    unsafe { ((*file).ops.flush)((*file).state) }
}

/// # Safety
/// As [`write`]; `file` is not used again.
pub(crate) unsafe fn close(file: *mut File) -> bool {
    // SAFETY: the caller's contract.
    unsafe { ((*file).ops.close)(file) }
}

/// The two standard output streams.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Standard {
    /// `stdout`.
    Output,
    /// `stderr`.
    Error,
}

/// Where the standard streams' bytes go, and the lock one stream call holds
/// across every write it makes.
///
/// POSIX has each stdio call lock its stream (`flockfile`), so one call's
/// output reaches the stream together even when the call writes it in
/// pieces, as `vfprintf`'s batches and `puts`' line and newline are. The
/// process decides what the lock is: a sink only one thread writes takes
/// none.
pub struct StandardSink {
    /// Write `bytes` to the stream whole. `false` reports a write error.
    pub write: fn(Standard, &[u8]) -> bool,
    /// Take the standard streams' lock for one call. A sink reached with
    /// the lock already held by the calling thread takes it again: the lock
    /// is the sink's to make reentrant.
    pub lock: fn(),
    /// Release one [`lock`](Self::lock).
    pub unlock: fn(),
}

/// The installed sink; null until a process installs one.
static SINK: AtomicPtr<StandardSink> = AtomicPtr::new(ptr::null_mut());

/// Send `stdout` and `stderr` to `sink` from now on. A later install
/// replaces an earlier one; a call in flight finishes on the sink it read.
pub fn install_standard_sink(sink: &'static StandardSink) {
    SINK.store(ptr::from_ref(sink).cast_mut(), Ordering::Release);
}

fn installed() -> Option<&'static StandardSink> {
    // SAFETY: only `install_standard_sink` stores here, and it stores a
    // `&'static StandardSink`.
    unsafe { SINK.load(Ordering::Acquire).as_ref() }
}

/// Run one stream call on `file` holding its lock: around every write it
/// makes, the standard streams' sink's lock; nothing for a consumer's
/// stream, whose owner synchronizes it.
///
/// # Safety
/// `file` is a live `FILE *` this crate or a consumer constructed.
pub(crate) unsafe fn locked<R>(file: *mut File, call: impl FnOnce() -> R) -> R {
    // SAFETY: the caller's contract.
    let standard = ptr::eq(unsafe { (*file).ops }, &STANDARD_OPS);
    let sink = if standard { installed() } else { None };
    if let Some(sink) = sink {
        (sink.lock)();
    }
    let result = call();
    if let Some(sink) = sink {
        (sink.unlock)();
    }
    result
}

/// The standard streams' state: an inert tag naming which one.
const OUTPUT_TAG: *mut c_void = ptr::without_provenance_mut(1);
const ERROR_TAG: *mut c_void = ptr::without_provenance_mut(2);

unsafe fn standard_write(state: *mut c_void, bytes: &[u8]) -> bool {
    let Some(sink) = installed() else {
        return false;
    };
    let stream = if state == OUTPUT_TAG {
        Standard::Output
    } else {
        Standard::Error
    };
    (sink.write)(stream, bytes)
}

unsafe fn standard_flush(_: *mut c_void) -> bool {
    // The sink receives every write whole; there is nothing buffered here.
    true
}

unsafe fn standard_close(_: *mut File) -> bool {
    // C lets a program close `stdout`; the stream is static, so closing it
    // only ends this program's use of it.
    true
}

static STANDARD_OPS: FileOps = FileOps {
    write: standard_write,
    flush: standard_flush,
    close: standard_close,
};

static STDOUT: File = File::new(&STANDARD_OPS, OUTPUT_TAG);
static STDERR: File = File::new(&STANDARD_OPS, ERROR_TAG);

/// A `FILE *` value: musl declares `extern FILE *const stdout`, so the
/// symbol holds a pointer to the stream, not the stream.
#[repr(transparent)]
pub struct Stream(*mut File);

// SAFETY: a pointer to an immutable static; nothing writes through it.
unsafe impl Sync for Stream {}

impl Stream {
    pub(crate) fn file(&self) -> *mut File {
        self.0
    }
}

/// The `FILE *` of a standard stream — what a consumer's `fdopen` answers
/// for descriptor 1 or 2.
pub fn standard_stream(which: Standard) -> *mut File {
    match which {
        Standard::Output => stdout.file(),
        Standard::Error => stderr.file(),
    }
}

#[unsafe(no_mangle)]
pub static stdout: Stream = Stream(&STDOUT as *const File as *mut File);

#[unsafe(no_mangle)]
pub static stderr: Stream = Stream(&STDERR as *const File as *mut File);
