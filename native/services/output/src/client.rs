//! `client` — the output client face (`res/USERLAND.md` §5.9).
//!
//! [`OutputClient`] is the app-facing contract, born async: `write`
//! makes forward progress or awaits queue space (never a silent drop),
//! `flush` confirms the backend consumed a prefix.  [`RingOutput`] is
//! the production shape: the request ring carries [`OutputRequest`]s,
//! the data plane is the delivered queue window's
//! `ring::spsc<OutChunk>` producer half, and every reply pushes the
//! consumed counter back through the control plane.

use kore_abi::svc::OutChunk;

use crate::service::{OutputRequest, OutputResponse};

/// Why a client-side operation failed (a wire refusal or a full queue
/// the caller chose not to retry against yet).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputError {
    /// The service refused the attach/verb.
    Refused,
    /// The queue is full at this instant — the honest answer of the v1
    /// `write`, which is **non-blocking** w.r.t. ring capacity: the
    /// caller's loop (or a `try_write` wrapper, when one lands) decides
    /// when to retry and drive a flush.
    Full,
    /// The backend is not attached (no queue delivered yet).
    NotAttached,
}

/// The app-facing output contract (`res/USERLAND.md` §5.9).  Empty
/// buffers complete immediately; a partial write is a lawful answer;
/// `write_all` loops belong in client code, not the protocol.
#[allow(async_fn_in_trait)] // clients are generic over the impl (§4)
pub trait OutputClient {
    /// Write a prefix of `src`; **non-blocking in v1** — a full queue
    /// answers [`OutputError::Full`] rather than parking, so the caller
    /// drives a flush and retries (the await-space convenience belongs to
    /// the client library, per `res/USERLAND.md` §5.9).
    async fn write(&mut self, src: &[u8]) -> Result<usize, OutputError>;

    /// Confirm all bytes accepted before this call reached the backend.
    async fn flush(&mut self) -> Result<(), OutputError>;
}

/// The production client: a ring producer over the delivered queue
/// window plus the request path (`AttachTx`/`Flush`/`DetachTx`).
pub struct RingOutput<C> {
    producer: kore_ring::spsc::Producer<'static, OutChunk>,
    control: C,
    /// Bytes this client has pushed (the flush watermark).
    pushed: u64,
}

impl<C> RingOutput<C> {
    /// Attach a producer half + the control channel implementation.
    pub fn new(producer: kore_ring::spsc::Producer<'static, OutChunk>, control: C) -> Self {
        Self {
            producer,
            control,
            pushed: 0,
        }
    }

    /// The raw producer half (a driver that chunks itself).
    pub fn producer(&mut self) -> &mut kore_ring::spsc::Producer<'static, OutChunk> {
        &mut self.producer
    }

    /// Push one chunk's worth from `src`; `None` when the queue is full.
    fn push_chunk(&mut self, src: &[u8]) -> Option<usize> {
        let n = src.len().min(OutChunk::CHUNK_DATA);
        let mut chunk = OutChunk {
            len: n as u8,
            flags: 0,
            reserved: [0; 6],
            bytes: [0; 24],
        };
        chunk.bytes[..n].copy_from_slice(&src[..n]);
        self.producer.push(chunk).ok()?;
        self.pushed += n as u64;
        Some(n)
    }
}

impl<C, Fut> OutputClient for RingOutput<C>
where
    C: FnMut(OutputRequest) -> Fut,
    Fut: core::future::Future<Output = Option<OutputResponse>>,
{
    async fn write(&mut self, src: &[u8]) -> Result<usize, OutputError> {
        if src.is_empty() {
            return Ok(0);
        }
        match self.push_chunk(src) {
            Some(n) => Ok(n),
            // Full: drive the backend (a flush is what consumes) and
            // report honestly — awaiting space is the caller's loop.
            None => Err(OutputError::Full),
        }
    }

    async fn flush(&mut self) -> Result<(), OutputError> {
        let reply = (self.control)(OutputRequest::Flush {
            through: self.pushed,
        })
        .await;
        match reply {
            Some(OutputResponse::Flushed { through }) if through >= self.pushed => Ok(()),
            Some(OutputResponse::Flushed { .. }) => Err(OutputError::Refused),
            Some(OutputResponse::Refused(_)) => Err(OutputError::NotAttached),
            _ => Err(OutputError::Refused),
        }
    }
}
