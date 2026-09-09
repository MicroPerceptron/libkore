//! Bytes in and out of a file, through the pair's bounded data window.
//!
//! Nothing here allocates: a reader consumes each chunk in place from the
//! window (or copies it into storage it owns), and a writer stages each
//! chunk into the window before the request names it. Every transfer is
//! bounded by the window and by [`MAX_TRANSFER`], and a reply may never
//! claim more bytes than were asked for.

use crate::{
    ANY_REVISION, Fs, FsError, FsRequest, FsResponse, Refusal, View, WALK_READ,
    bounds::MAX_TRANSFER,
};

#[cfg(feature = "index")]
use crate::{ObjectKind, Rights, WALK_MUTATE, WALK_WRITE};

/// Validate a read reply before any byte in the shared data window is
/// exposed to a consumer.
pub(crate) fn read_transfer_count(reply: FsResponse, requested: usize) -> Result<usize, Refusal> {
    let FsResponse::Transferred { count } = reply else {
        return Err(Refusal::Pair);
    };
    let count = usize::try_from(count).map_err(|_| Refusal::Pair)?;
    if count > requested {
        return Err(Refusal::Pair);
    }
    Ok(count)
}

/// The request must fit both the actual window and the frozen wire limit.
fn transfer_len(offset: u64, bytes: usize, window: usize) -> Result<usize, Refusal> {
    let count = bytes.min(window).min(MAX_TRANSFER);
    if count == 0 {
        return Err(Refusal::Fs(FsError::NoWindow));
    }
    offset
        .checked_add(count as u64)
        .ok_or(Refusal::Fs(FsError::Range))?;
    Ok(count)
}

fn committed_count(reply: FsResponse, requested: usize) -> Result<usize, Refusal> {
    match reply {
        FsResponse::Committed { .. } => Ok(requested),
        _ => Err(Refusal::Pair),
    }
}

/// Result of replacing an authoritative file version. The bytes are stored
/// in both cases; the second variant tells a UI that the auxiliary text
/// index did not accept the new version.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SaveOutcome {
    Indexed,
    IndexIncomplete,
}

impl<'w, 't> Fs<'w, 't> {
    /// The largest single transfer this face can carry: the data window,
    /// capped at the protocol's [`MAX_TRANSFER`]. Zero means no window was
    /// attached, which every bulk verb refuses as [`FsError::NoWindow`].
    pub fn chunk(&self) -> usize {
        self.window.len().min(MAX_TRANSFER)
    }

    /// Read at most one transfer window into caller-owned storage.
    ///
    /// A short successful read is not necessarily EOF; only zero is EOF.
    /// The destination is unchanged on refusal. No allocation or retained borrow.
    pub async fn read_at(
        &mut self,
        file: &View,
        offset: u64,
        dst: &mut [u8],
    ) -> Result<usize, Refusal> {
        if dst.is_empty() {
            return Ok(0);
        }
        let requested = transfer_len(offset, dst.len(), self.chunk())?;
        let reply = self
            .checked(&FsRequest::Read {
                handle: file.handle(),
                offset,
                window_offset: 0,
                length: requested as u32,
            })
            .await?;
        let count = read_transfer_count(reply, requested)?;
        dst[..count].copy_from_slice(&self.window[..count]);
        Ok(count)
    }

    /// Write at most one transfer window without truncating the file's tail.
    ///
    /// `Committed` acknowledges this whole bounded chunk. A pair failure may
    /// occur after publication: it is not proof that no bytes were written and
    /// must not be automatically retried. An outstanding request keeps custody
    /// of the data window, so reject before staging any replacement bytes.
    pub async fn write_at(
        &mut self,
        file: &View,
        offset: u64,
        src: &[u8],
    ) -> Result<usize, Refusal> {
        if src.is_empty() {
            return Ok(0);
        }
        let requested = transfer_len(offset, src.len(), self.chunk())?;
        if !self.client.is_idle() {
            return Err(Refusal::Pair);
        }
        self.window[..requested].copy_from_slice(&src[..requested]);
        let reply = self
            .checked(&FsRequest::WriteAt {
                handle: file.handle(),
                offset,
                window_offset: 0,
                length: requested as u32,
                expected: ANY_REVISION,
            })
            .await?;
        committed_count(reply, requested)
    }

    /// Set a file's length; unlike an offset write this explicitly truncates.
    pub async fn resize(&mut self, file: &View, length: u64) -> Result<crate::Revision, Refusal> {
        match self
            .checked(&FsRequest::Resize {
                handle: file.handle(),
                length,
                expected: ANY_REVISION,
            })
            .await?
        {
            FsResponse::Committed { revision } => Ok(revision),
            _ => Err(Refusal::Pair),
        }
    }

    /// Read one chunk of `file` at `offset` into the window; the bytes are
    /// `self.window()[..n]` until the next request. `Ok(0)` is end of file.
    pub async fn read_chunk(&mut self, file: &View, offset: u64) -> Result<usize, Refusal> {
        let chunk = self.chunk();
        if chunk == 0 {
            return Err(Refusal::Fs(FsError::NoWindow));
        }
        let length = u32::try_from(chunk).map_err(|_| Refusal::Fs(FsError::Range))?;
        let reply = self
            .checked(&FsRequest::Read {
                handle: file.handle(),
                offset,
                window_offset: 0,
                length,
            })
            .await?;
        // A provider may impose a smaller transfer ceiling; only a zero-byte
        // reply proves EOF. It may never claim bytes beyond the requested
        // window.
        read_transfer_count(reply, chunk)
    }

    /// Read the complete file at `path`, presenting each bounded chunk to
    /// `consume` before the data window is reused.
    ///
    /// The callback is deliberately synchronous and the client allocates
    /// nothing: a caller that wants to retain the bytes chooses and owns the
    /// storage, while a streaming caller can consume them in place. Each
    /// slice is borrowed from this process's filesystem data window and is
    /// valid only for the duration of that callback invocation.
    ///
    /// The opened view is released after success, a filesystem or pair
    /// refusal, a malformed transfer count, offset overflow, or a callback
    /// refusal. A callback must return an error rather than unwind if it
    /// needs that cleanup guarantee.
    pub async fn read_file<F>(&mut self, path: &[u8], mut consume: F) -> Result<(), Refusal>
    where
        F: FnMut(&[u8]) -> Result<(), Refusal>,
    {
        let file = self.open(path, WALK_READ).await?;
        let mut offset = 0u64;
        let result = loop {
            let count = match self.read_chunk(&file, offset).await {
                Ok(count) => count,
                Err(error) => break Err(error),
            };
            if count == 0 {
                break Ok(());
            }
            let Some(next) = offset.checked_add(count as u64) else {
                break Err(Refusal::Fs(FsError::Range));
            };
            if let Err(error) = consume(&self.window[..count]) {
                break Err(error);
            }
            offset = next;
        };
        self.close(file).await;
        result
    }

    /// Replace `file`'s contents with `bytes`: written chunk by chunk from
    /// offset zero, then resized to exactly `bytes.len()`. Unconditional
    /// (`ANY_REVISION`): the caller holds the route, and a concurrent writer
    /// is the service's `Conflict` to report, not this client's to guess.
    pub async fn write_bytes(&mut self, file: &View, bytes: &[u8]) -> Result<(), Refusal> {
        let mut offset = 0usize;
        while offset < bytes.len() {
            let count = self.write_at(file, offset as u64, &bytes[offset..]).await?;
            offset += count;
        }
        self.resize(file, bytes.len() as u64).await?;
        Ok(())
    }

    /// Append `bytes` to `file` in one bounded transfer; yields the offset
    /// the service committed them at. Bounded by one chunk on purpose: an
    /// append is atomic per request, and splitting a payload across several
    /// would interleave with another appender.
    pub async fn append_bytes(&mut self, file: &View, bytes: &[u8]) -> Result<u64, Refusal> {
        let chunk = self.chunk();
        if chunk == 0 {
            return Err(Refusal::Fs(FsError::NoWindow));
        }
        if bytes.len() > chunk {
            return Err(Refusal::Fs(FsError::Range));
        }
        if !self.client.is_idle() {
            return Err(Refusal::Pair);
        }
        self.window[..bytes.len()].copy_from_slice(bytes);
        match self
            .checked(&FsRequest::Append {
                handle: file.handle(),
                window_offset: 0,
                length: bytes.len() as u32,
                expected: ANY_REVISION,
            })
            .await?
        {
            FsResponse::Appended { offset, .. } => Ok(offset),
            _ => Err(Refusal::Pair),
        }
    }

    /// Replace a complete text file, creating its final edge when absent
    /// — the editor's save. Streams `bytes` through the bounded window,
    /// truncates the old tail, then publishes the same plain-text index
    /// batch the `write` verb does; every view is closed on both paths.
    #[cfg(feature = "index")]
    pub async fn replace_file(
        &mut self,
        path: &[u8],
        bytes: &[u8],
    ) -> Result<SaveOutcome, Refusal> {
        let file = match self.open(path, WALK_WRITE).await {
            Ok(file) => file,
            Err(Refusal::Fs(FsError::NotFound)) => {
                // `Create` hands back a view attenuated to the parent's
                // rights, and the replacement writes through it at once,
                // so the parent walk carries WRITE beside the mutation.
                let (parent, name) = self
                    .open_parent(path, WALK_MUTATE.union(Rights::WRITE))
                    .await?;
                let created = self.create(&parent, name, ObjectKind::File).await;
                self.close(parent).await;
                created?
            }
            Err(error) => return Err(error),
        };
        let written = self.write_bytes(&file, bytes).await;
        let outcome = match written {
            Ok(()) => crate::index::classify(crate::index::publish(self, &file, Some(bytes)).await),
            Err(error) => Err(error),
        };
        self.close(file).await;
        outcome.map(|indexed| match indexed {
            crate::index::Indexed::Published => SaveOutcome::Indexed,
            crate::index::Indexed::Incomplete => SaveOutcome::IndexIncomplete,
        })
    }

    /// Ask the provider to make `file`'s committed state durable.
    pub async fn sync(&mut self, file: &View) -> Result<(), Refusal> {
        match self
            .checked(&FsRequest::Sync {
                handle: file.handle(),
            })
            .await?
        {
            FsResponse::Done => Ok(()),
            _ => Err(Refusal::Pair),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_offset_requests_reject_overflow_before_publication() {
        assert_eq!(transfer_len(5, 17, 8).unwrap(), 8);
        assert_eq!(
            transfer_len(0, usize::MAX, usize::MAX).unwrap(),
            MAX_TRANSFER
        );
        assert!(matches!(
            transfer_len(0, 1, 0),
            Err(Refusal::Fs(FsError::NoWindow))
        ));
        assert!(matches!(
            transfer_len(u64::MAX, 1, 8),
            Err(Refusal::Fs(FsError::Range))
        ));
        assert_eq!(transfer_len(u64::MAX - 8, 9, 8).unwrap(), 8);
    }

    #[test]
    fn only_a_write_commit_acknowledges_the_staged_chunk() {
        assert_eq!(
            committed_count(FsResponse::Committed { revision: 1 }, 7).unwrap(),
            7
        );
        for reply in [FsResponse::Done, FsResponse::Transferred { count: 7 }] {
            assert!(matches!(committed_count(reply, 7), Err(Refusal::Pair)));
        }
    }

    #[test]
    fn read_count_cannot_exceed_the_shared_window_request() {
        assert!(matches!(
            read_transfer_count(FsResponse::Transferred { count: 17 }, 16),
            Err(Refusal::Pair)
        ));
    }

    #[test]
    fn read_count_rejects_the_wrong_reply_shape() {
        assert!(matches!(
            read_transfer_count(FsResponse::Done, MAX_TRANSFER),
            Err(Refusal::Pair)
        ));
    }

    #[test]
    fn read_count_accepts_eof_and_the_exact_bound() {
        assert!(matches!(
            read_transfer_count(FsResponse::Transferred { count: 0 }, MAX_TRANSFER),
            Ok(0)
        ));
        assert!(matches!(
            read_transfer_count(
                FsResponse::Transferred {
                    count: MAX_TRANSFER as u32,
                },
                MAX_TRANSFER,
            ),
            Ok(MAX_TRANSFER)
        ));
    }
}
