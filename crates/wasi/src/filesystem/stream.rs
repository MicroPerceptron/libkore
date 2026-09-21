//! `read-via-stream`, `write-via-stream` and `append-via-stream`: a file as
//! `wasmtime-wasi-io` streams.
//!
//! `wasi:io` reads and writes are synchronous and `ready` is where a stream
//! waits, so a file stream does its I/O in `ready`: an input stream reads
//! the next transfer into its buffer, an output stream publishes the bytes
//! the guest wrote. That I/O is an op on the host's slot ([`FsHost::exec`]),
//! which is what makes it safe for `wasi:io/poll` to drop a `ready` half
//! way: the op finishes on the next turn and applies its own result to the
//! stream's state — an offset advanced, a buffer filled — whether or not
//! the `ready` that started it is still waiting.
//!
//! A stream shares its descriptor's view: the file stays open until the
//! descriptor and every stream made from it are gone.

use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::fmt;

use kore::runtime::Pinned;
use wasmtime_wasi_io::bytes::Bytes;
use wasmtime_wasi_io::poll::Pollable;
use wasmtime_wasi_io::streams::{InputStream, OutputStream, StreamError, StreamResult};

use super::code;
use super::host::{Custody, Face, FsHost, Shared};
use crate::bindings::wasi::filesystem::types::ErrorCode;

/// A file operation's refusal, carried as a stream's `error` so
/// `filesystem-error-code` can answer it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StreamFailure(pub(crate) ErrorCode);

impl fmt::Display for StreamFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "file operation failed: {:?}", self.0)
    }
}

impl core::error::Error for StreamFailure {}

fn failed(error: ErrorCode) -> StreamError {
    StreamError::LastOperationFailed(wasmtime::Error::new(StreamFailure(error)))
}

/// What a stream's `ready` does next.
enum Next {
    Ready,
    /// This stream's op is in the slot: drive it.
    Settle,
    Issue,
}

struct ReadState {
    position: u64,
    buffer: Bytes,
    end: bool,
    failure: Option<ErrorCode>,
    in_flight: bool,
}

struct Reader {
    host: Rc<FsHost>,
    file: Shared,
    state: Rc<RefCell<ReadState>>,
}

impl ReadState {
    fn next(&self) -> Next {
        if !self.buffer.is_empty() || self.end || self.failure.is_some() {
            Next::Ready
        } else if self.in_flight {
            Next::Settle
        } else {
            Next::Issue
        }
    }
}

impl Reader {
    fn next(&self) -> Next {
        self.state.try_borrow().map_or(Next::Settle, |s| s.next())
    }

    /// Wait until bytes are buffered, the file ended, or the read failed.
    /// Each round finishes an op, and a finished read op always leaves one
    /// of the three, so this takes at most two rounds.
    async fn fill(&self) {
        for _ in 0..2 {
            match self.next() {
                Next::Ready => return,
                Next::Settle => self.host.settle().await,
                Next::Issue => {
                    let (state, file) = (self.state.clone(), self.file.clone());
                    self.host
                        .exec(async move |pair: &mut Custody| {
                            let position = {
                                let mut s = state.borrow_mut();
                                s.in_flight = true;
                                s.position
                            };
                            let read = pair
                                .call(async |fs: &mut Face| {
                                    // One transfer into the window, copied
                                    // out before the next request reuses it.
                                    let count = fs.read_chunk(file.view()?, position).await?;
                                    Ok(Vec::from(&fs.window()[..count]))
                                })
                                .await;
                            let mut s = state.borrow_mut();
                            s.in_flight = false;
                            match read {
                                Ok(bytes) if bytes.is_empty() => s.end = true,
                                Ok(bytes) => {
                                    s.position += bytes.len() as u64;
                                    s.buffer = bytes.into();
                                }
                                Err(refusal) => s.failure = Some(code(refusal)),
                            }
                        })
                        .await;
                }
            }
        }
        // Still waiting after an op settled: the op is gone. Report it
        // rather than let the guest wait on nothing.
        if let Ok(mut s) = self.state.try_borrow_mut()
            && !matches!(s.next(), Next::Ready)
        {
            s.in_flight = false;
            s.failure = Some(ErrorCode::Io);
        }
    }
}

/// `read-via-stream`: the file from an offset, one transfer at a time.
pub(crate) struct FileInput(Pinned<Reader>);

impl FileInput {
    pub(crate) fn new(host: Rc<FsHost>, file: Shared, position: u64) -> Option<Self> {
        let reader = Reader {
            host,
            file,
            state: Rc::new(RefCell::new(ReadState {
                position,
                buffer: Bytes::new(),
                end: false,
                failure: None,
                in_flight: false,
            })),
        };
        Pinned::new(reader).ok().map(Self)
    }
}

#[wasmtime_wasi_io::async_trait]
impl Pollable for FileInput {
    async fn ready(&mut self) {
        let reader = &self.0;
        // Bound by a `match`, not an `if let`: the scrutinee's `Err` side
        // is the unpinned future, which must not live across the await.
        let wait = match Pinned::new(async move {
            if let Some(reader) = reader.get() {
                reader.fill().await;
            }
        }) {
            Ok(wait) => wait,
            Err(_) => return,
        };
        wait.await
    }
}

impl InputStream for FileInput {
    fn read(&mut self, size: usize) -> StreamResult<Bytes> {
        let reader = self.0.get().ok_or(StreamError::Closed)?;
        let mut s = reader
            .state
            .try_borrow_mut()
            .map_err(|_| StreamError::trap("file stream state busy"))?;
        if !s.buffer.is_empty() {
            let take = size.min(s.buffer.len());
            return Ok(s.buffer.split_to(take));
        }
        if let Some(error) = s.failure.take() {
            s.end = true;
            return Err(failed(error));
        }
        if s.end {
            return Err(StreamError::Closed);
        }
        // Nothing yet: `ready` reads the next transfer.
        Ok(Bytes::new())
    }
}

struct WriteState {
    /// `None`: every write appends.
    position: Option<u64>,
    pending: Bytes,
    failure: Option<ErrorCode>,
    closed: bool,
    in_flight: bool,
}

struct Writer {
    host: Rc<FsHost>,
    file: Shared,
    state: Rc<RefCell<WriteState>>,
}

impl WriteState {
    fn next(&self) -> Next {
        if self.in_flight {
            Next::Settle
        } else if self.pending.is_empty() || self.failure.is_some() || self.closed {
            Next::Ready
        } else {
            Next::Issue
        }
    }
}

impl Writer {
    fn next(&self) -> Next {
        self.state.try_borrow().map_or(Next::Settle, |s| s.next())
    }

    /// Publish what the guest wrote. `check-write` admits one transfer at a
    /// time, so one op carries it. Each round finishes an op, and a
    /// finished write op either commits at least one byte or fails, so the
    /// rounds are bounded by the bytes pending.
    async fn drain(&self) {
        let rounds = self.state.try_borrow().map_or(0, |s| s.pending.len()) + 2;
        for _ in 0..rounds {
            match self.next() {
                Next::Ready => return,
                Next::Settle => self.host.settle().await,
                Next::Issue => {
                    let (state, file) = (self.state.clone(), self.file.clone());
                    self.host
                        .exec(async move |pair: &mut Custody| {
                            let (position, bytes) = {
                                let mut s = state.borrow_mut();
                                s.in_flight = true;
                                (s.position, s.pending.clone())
                            };
                            let wrote = pair
                                .call(async |fs: &mut Face| {
                                    let file = file.view()?;
                                    match position {
                                        Some(offset) => fs.write_at(file, offset, &bytes).await,
                                        // Atomic per request; `check-write`
                                        // admitted at most one transfer.
                                        None => {
                                            fs.append_bytes(file, &bytes).await?;
                                            Ok(bytes.len())
                                        }
                                    }
                                })
                                .await;
                            let mut s = state.borrow_mut();
                            s.in_flight = false;
                            match wrote {
                                // A commit that moves nothing would retry
                                // forever: refuse it as the I/O failure it is.
                                Ok(0) => s.failure = Some(ErrorCode::Io),
                                Ok(count) => {
                                    let count = count.min(s.pending.len());
                                    let _ = s.pending.split_to(count);
                                    if let Some(offset) = s.position.as_mut() {
                                        *offset += count as u64;
                                    }
                                }
                                Err(refusal) => s.failure = Some(code(refusal)),
                            }
                        })
                        .await;
                }
            }
        }
        if let Ok(mut s) = self.state.try_borrow_mut()
            && !matches!(s.next(), Next::Ready)
        {
            s.in_flight = false;
            s.failure = Some(ErrorCode::Io);
        }
    }
}

/// `write-via-stream` (positioned) or `append-via-stream`.
pub(crate) struct FileOutput(Pinned<Writer>);

impl FileOutput {
    pub(crate) fn new(host: Rc<FsHost>, file: Shared, position: Option<u64>) -> Option<Self> {
        let writer = Writer {
            host,
            file,
            state: Rc::new(RefCell::new(WriteState {
                position,
                pending: Bytes::new(),
                failure: None,
                closed: false,
                in_flight: false,
            })),
        };
        Pinned::new(writer).ok().map(Self)
    }

    fn state(&self) -> StreamResult<(&Writer, core::cell::RefMut<'_, WriteState>)> {
        let writer = self.0.get().ok_or(StreamError::Closed)?;
        let mut s = writer
            .state
            .try_borrow_mut()
            .map_err(|_| StreamError::trap("file stream state busy"))?;
        if let Some(error) = s.failure.take() {
            s.closed = true;
            return Err(failed(error));
        }
        if s.closed {
            return Err(StreamError::Closed);
        }
        Ok((writer, s))
    }
}

#[wasmtime_wasi_io::async_trait]
impl Pollable for FileOutput {
    async fn ready(&mut self) {
        let writer = &self.0;
        // Bound by a `match`, not an `if let`: the scrutinee's `Err` side
        // is the unpinned future, which must not live across the await.
        let wait = match Pinned::new(async move {
            if let Some(writer) = writer.get() {
                writer.drain().await;
            }
        }) {
            Ok(wait) => wait,
            Err(_) => return,
        };
        wait.await
    }
}

impl OutputStream for FileOutput {
    fn write(&mut self, bytes: Bytes) -> StreamResult<()> {
        let (writer, mut s) = self.state()?;
        // `check-write` promised this much room; more is the guest's fault.
        if !s.pending.is_empty() || s.in_flight || bytes.len() > writer.host.chunk() {
            return Err(StreamError::trap("write exceeded check-write"));
        }
        s.pending = bytes;
        Ok(())
    }

    /// Bytes are published by `ready`; until they are, `check-write`
    /// answers zero, which is what a flush waits out.
    fn flush(&mut self) -> StreamResult<()> {
        self.state().map(|_| ())
    }

    fn check_write(&mut self) -> StreamResult<usize> {
        let (writer, s) = self.state()?;
        // No data window: nothing could ever be admitted, so say so rather
        // than answer zero forever.
        if writer.host.chunk() == 0 {
            return Err(failed(ErrorCode::Io));
        }
        Ok(if s.pending.is_empty() && !s.in_flight {
            writer.host.chunk()
        } else {
            0
        })
    }
}
