//! **WASI's standard streams, over the native process stream**:
//! `wasmtime-wasi-io`'s `InputStream`, `OutputStream` and `Pollable`
//! implemented on Kore's own endpoint.
//!
//! A WASI host hands a guest three stream resources — stdin, stdout,
//! stderr — and `wasi:io/poll` may wait on several at once. Here all three
//! are views of **one** native [`ProcessStdioChild`], shared on its thread;
//! no pump task sits between the guest and the rings and nothing is copied
//! twice:
//!
//! - **Reads and writes are synchronous ring operations** — the endpoint's
//!   `*_now` forms, which publish and notify before returning, as WASI's
//!   non-blocking `read`/`write`/`check-write` require.
//! - **`ready` is the endpoint's own park**, split at its wait
//!   ([`ProcessStdioParkStep`]): the endpoint is borrowed only to prepare
//!   and to settle, never across the wait, so several pollables share it.
//! - **The line discipline is the launcher's protocol**
//!   ([`kore::abi::process::ProcessOutputKind`]): the guest wanting input
//!   publishes one `InputReady`, the launcher answers with one canonical
//!   line (a trailing `\n`, or EOT for end of input), and [`close`]
//!   publishes the stream's `Closed` only once any requested line has
//!   arrived.
//!
//! The endpoint is thread-bound (its slots are), and the traits demand
//! `Send`: each resource holds the shared endpoint in a
//! [`Pinned`], and each `ready` future is pinned too.
//!
//! [`close`]: ProcessStreams::close

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::rc::Rc;
use core::cell::RefCell;

use kore::abi::process::{PROCESS_OUTPUT_CHUNK_BYTES, ProcessInputKind, ProcessOutputEvent};
use kore::process::{
    ProcessStdioChild, ProcessStdioParkStep, ProcessStdioWaitFor, ProcessStdioWake,
};
use kore::runtime::Pinned;
use kore::runtime::transport::Transport;
use wasmtime_wasi_io::bytes::Bytes;
use wasmtime_wasi_io::poll::Pollable;
use wasmtime_wasi_io::streams::{
    DynInputStream, DynOutputStream, InputStream, OutputStream, StreamError, StreamResult,
};

/// Bytes of one output record.
const RECORD: usize = PROCESS_OUTPUT_CHUNK_BYTES;
/// Canonical-mode end of input (Ctrl-D), as the terminal delivers it.
const EOT: u8 = 0x04;
/// How long a park waits out a rotating notice receiver's handoff before
/// preparing again — the native endpoint's own figure (`park_after_drain`).
const HANDOFF_NS: u64 = 100_000;

/// Why the process stream stopped carrying bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StdioLost {
    /// The launcher stopped answering: a notice was unreachable, or a park
    /// refused.
    Peer,
    /// The launcher sent a record the protocol does not allow.
    Protocol,
}

/// The endpoint and the guest-side protocol state the three streams share.
struct Shared<T: Transport + 'static> {
    endpoint: ProcessStdioChild<'static, 'static, T>,
    /// Bytes of the current line the guest has not read yet.
    input: VecDeque<u8>,
    /// An `InputReady` is published and its line has not fully arrived.
    requested: bool,
    /// End of input: a `Closed` record, or EOT.
    input_closed: bool,
    lost: Option<StdioLost>,
}

impl<T: Transport + 'static> Shared<T> {
    fn note(&mut self, wake: ProcessStdioWake) {
        if matches!(wake, ProcessStdioWake::Unreached(_)) {
            self.lost.get_or_insert(StdioLost::Peer);
        }
    }

    /// Take every stdin record that has landed.
    fn receive(&mut self) {
        while self.lost.is_none() {
            let Some((record, wake)) = self.endpoint.recv_stdin_now() else {
                return;
            };
            self.note(wake);
            match (record.kind(), record.bytes()) {
                (Ok(ProcessInputKind::Data), Ok(bytes)) => {
                    match bytes.iter().position(|&b| b == EOT) {
                        Some(end) => {
                            self.input.extend(&bytes[..end]);
                            self.input_closed = true;
                        }
                        None => self.input.extend(bytes),
                    }
                    if bytes.contains(&b'\n') || self.input_closed {
                        self.requested = false;
                    }
                }
                (Ok(ProcessInputKind::Closed), _) => {
                    self.input_closed = true;
                    self.requested = false;
                }
                _ => self.lost = Some(StdioLost::Protocol),
            }
        }
    }

    /// Ask the launcher for the next line, once. A full output ring defers
    /// the request to the next `ready` (which then also waits for capacity).
    fn request_line(&mut self) {
        if self.requested || self.input_closed || self.lost.is_some() {
            return;
        }
        if let Ok(wake) = self
            .endpoint
            .send_output_now(ProcessOutputEvent::input_ready())
        {
            self.requested = true;
            self.note(wake);
        }
    }

    /// What an input `ready` still waits for (`None`: ready).
    fn input_wait(&mut self) -> Option<ProcessStdioWaitFor> {
        self.receive();
        if !self.input.is_empty() || self.input_closed || self.lost.is_some() {
            return None;
        }
        self.request_line();
        Some(if self.requested {
            ProcessStdioWaitFor::Incoming
        } else {
            ProcessStdioWaitFor::IncomingOrCapacity
        })
    }

    /// What an output `ready` still waits for (`None`: ready).
    fn output_wait(&mut self) -> Option<ProcessStdioWaitFor> {
        self.receive();
        if self.lost.is_some() || self.endpoint.output_capacity_left() != 0 {
            return None;
        }
        Some(ProcessStdioWaitFor::IncomingOrCapacity)
    }

    /// What closing still waits for: the line an outstanding request owes.
    fn close_wait(&mut self) -> Option<ProcessStdioWaitFor> {
        self.receive();
        self.input.clear();
        (self.requested && !self.input_closed && self.lost.is_none())
            .then_some(ProcessStdioWaitFor::Incoming)
    }
}

/// Park until `wait` answers `None`, holding the endpoint only to prepare
/// and settle each park — never across the wait.
async fn park<T: Transport + 'static>(
    shared: &RefCell<Shared<T>>,
    mut wait: impl FnMut(&mut Shared<T>) -> Option<ProcessStdioWaitFor>,
) {
    loop {
        let step = {
            let mut s = shared.borrow_mut();
            // The drain snapshot comes before the state check, so a record
            // that lands between them is seen by the park's recheck.
            let drain = s.endpoint.begin_drain();
            let Some(wait_for) = wait(&mut s) else {
                return;
            };
            match s.endpoint.prepare_park(&drain, wait_for) {
                Ok(step) => step,
                Err(_) => {
                    s.lost.get_or_insert(StdioLost::Peer);
                    return;
                }
            }
        };
        match step {
            ProcessStdioParkStep::Changed => {}
            ProcessStdioParkStep::Handoff => {
                kore::runtime::executor::next_wake_for(HANDOFF_NS).await;
            }
            ProcessStdioParkStep::Park(ticket) => {
                let reason = kore::runtime::executor::next_wake_on(ticket.post()).await;
                let mut s = shared.borrow_mut();
                if s.endpoint.settle_park(ticket, reason).is_err() {
                    s.lost.get_or_insert(StdioLost::Peer);
                    return;
                }
            }
        }
    }
}

/// A process stream shared as WASI's three standard streams. `!Send`, like
/// the endpoint it owns; the streams it hands out are `Send` through
/// [`Pinned`].
pub struct ProcessStreams<T: Transport + 'static> {
    shared: Rc<RefCell<Shared<T>>>,
}

impl<T: Transport + 'static> ProcessStreams<T> {
    /// Share `endpoint` as the standard streams.
    pub fn new(endpoint: ProcessStdioChild<'static, 'static, T>) -> Self {
        Self {
            shared: Rc::new(RefCell::new(Shared {
                endpoint,
                input: VecDeque::new(),
                requested: false,
                input_closed: false,
                lost: None,
            })),
        }
    }

    fn pinned(&self) -> Option<Pinned<Rc<RefCell<Shared<T>>>>> {
        Pinned::new(self.shared.clone()).ok()
    }

    /// `wasi:cli/stdin`. `None` only before the runtime is up.
    pub fn stdin(&self) -> Option<DynInputStream> {
        Some(Box::new(Stdin(self.pinned()?)))
    }

    /// `wasi:cli/stdout`.
    pub fn stdout(&self) -> Option<DynOutputStream> {
        Some(Box::new(Output {
            shared: self.pinned()?,
            stderr: false,
        }))
    }

    /// `wasi:cli/stderr`.
    pub fn stderr(&self) -> Option<DynOutputStream> {
        Some(Box::new(Output {
            shared: self.pinned()?,
            stderr: true,
        }))
    }

    /// Why the stream stopped, if it did.
    pub fn lost(&self) -> Option<StdioLost> {
        self.shared.borrow().lost
    }

    /// End the stream: wait for a line an outstanding `InputReady` still
    /// owes (the protocol forbids abandoning it), then publish `Closed` —
    /// waiting for ring capacity if the launcher is behind.
    pub async fn close(self) -> Result<(), StdioLost> {
        park(&self.shared, Shared::close_wait).await;
        loop {
            {
                let mut s = self.shared.borrow_mut();
                if let Some(lost) = s.lost {
                    return Err(lost);
                }
                if let Ok(wake) = s.endpoint.send_output_now(ProcessOutputEvent::closed()) {
                    s.note(wake);
                    return s.lost.map_or(Ok(()), Err);
                }
            }
            park(&self.shared, Shared::output_wait).await;
        }
    }
}

/// `wasi:cli/stdin` over the shared endpoint.
struct Stdin<T: Transport + 'static>(Pinned<Rc<RefCell<Shared<T>>>>);

#[wasmtime_wasi_io::async_trait]
impl<T: Transport + 'static> Pollable for Stdin<T> {
    async fn ready(&mut self) {
        let shared = &self.0;
        // Bound by a `match`, not an `if let`: the scrutinee's `Err` side is
        // the unpinned future, and an `if let` would keep it alive across
        // the await.
        let wait = match Pinned::new(async move {
            if let Some(shared) = shared.get() {
                park(shared, Shared::input_wait).await;
            }
        }) {
            Ok(wait) => wait,
            Err(_) => return,
        };
        wait.await
    }
}

impl<T: Transport + 'static> InputStream for Stdin<T> {
    fn read(&mut self, size: usize) -> StreamResult<Bytes> {
        let shared = self.0.get().ok_or(StreamError::Closed)?;
        let mut s = shared.borrow_mut();
        s.receive();
        if s.lost.is_some() {
            return Err(StreamError::Closed);
        }
        if s.input.is_empty() {
            if s.input_closed {
                return Err(StreamError::Closed);
            }
            // Nothing yet: ask for the next line; `ready` waits for it.
            s.request_line();
            return Ok(Bytes::new());
        }
        let take = size.min(s.input.len());
        Ok(s.input
            .drain(..take)
            .collect::<alloc::vec::Vec<u8>>()
            .into())
    }
}

/// `wasi:cli/stdout` or `wasi:cli/stderr` over the shared endpoint. Both
/// write into the one output ring, so the launcher sees them interleaved
/// exactly as the guest wrote them.
struct Output<T: Transport + 'static> {
    shared: Pinned<Rc<RefCell<Shared<T>>>>,
    stderr: bool,
}

#[wasmtime_wasi_io::async_trait]
impl<T: Transport + 'static> Pollable for Output<T> {
    async fn ready(&mut self) {
        let shared = &self.shared;
        // Bound by a `match`, not an `if let`: the scrutinee's `Err` side is
        // the unpinned future, and an `if let` would keep it alive across
        // the await.
        let wait = match Pinned::new(async move {
            if let Some(shared) = shared.get() {
                park(shared, Shared::output_wait).await;
            }
        }) {
            Ok(wait) => wait,
            Err(_) => return,
        };
        wait.await
    }
}

impl<T: Transport + 'static> OutputStream for Output<T> {
    fn write(&mut self, bytes: Bytes) -> StreamResult<()> {
        let shared = self.shared.get().ok_or(StreamError::Closed)?;
        let mut s = shared.borrow_mut();
        if s.lost.is_some() {
            return Err(StreamError::Closed);
        }
        // `check-write` promised this much room; more is the guest's fault.
        if bytes.len().div_ceil(RECORD) > s.endpoint.output_capacity_left() as usize {
            return Err(StreamError::trap("write exceeded check-write"));
        }
        for chunk in bytes.chunks(RECORD) {
            let event = if self.stderr {
                ProcessOutputEvent::stderr(chunk)
            } else {
                ProcessOutputEvent::stdout(chunk)
            }
            .map_err(|_| StreamError::trap("unencodable output record"))?;
            let wake = s
                .endpoint
                .send_output_now(event)
                .map_err(|_| StreamError::trap("output ring filled under check-write"))?;
            s.note(wake);
        }
        if s.lost.is_some() {
            return Err(StreamError::Closed);
        }
        Ok(())
    }

    /// Every write is already published to the launcher's ring; there is
    /// nothing held back to flush.
    fn flush(&mut self) -> StreamResult<()> {
        let shared = self.shared.get().ok_or(StreamError::Closed)?;
        match shared.borrow().lost {
            Some(_) => Err(StreamError::Closed),
            None => Ok(()),
        }
    }

    fn check_write(&mut self) -> StreamResult<usize> {
        let shared = self.shared.get().ok_or(StreamError::Closed)?;
        let mut s = shared.borrow_mut();
        s.receive();
        match s.lost {
            Some(_) => Err(StreamError::Closed),
            None => Ok(s.endpoint.output_capacity_left() as usize * RECORD),
        }
    }
}
