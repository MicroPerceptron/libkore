//! Ordinary filesystem I/O over the native DagFS owner.
//!
//! Paths resolve inside the task's endowed view; this module adds no namespace,
//! permission cache or handle owner. Helpers borrow the real client and view.
//! Explicit graph operations remain available under [`dag`].
//!
//! Native views retain their affine ownership:
//! ```compile_fail
//! use kore_std::fs::View;
//! fn duplicate(view: View) {
//!     let first = view;
//!     let second = view;
//!     drop((first, second));
//! }
//! ```

pub use kore_dagfs::{CloseFailure, DirEntry, Fs, FsError, Name, Refusal, Rights, Stat, View};

/// Native graph semantics: projections, grants, revisions and query vocabulary.
pub mod dag {
    pub use kore_dagfs::*;
}

/// Why a multi-transfer operation stopped.
#[derive(Debug)]
pub enum TransferStop {
    /// A read reached EOF before filling the caller's destination.
    UnexpectedEof,
    /// The operation was refused or its reply could not be established.
    Refused(Refusal),
}

/// An acknowledged prefix followed by a failure.
///
/// On a transport failure the current write may also have committed. `completed`
/// is a confirmed prefix, not a safe replay offset. No automatic retries occur.
#[derive(Debug)]
pub struct TransferError {
    pub completed: usize,
    pub reason: TransferStop,
}

fn checked_end(offset: u64, count: usize) -> Result<(), TransferError> {
    let count = u64::try_from(count).map_err(|_| range())?;
    offset.checked_add(count).ok_or_else(range)?;
    Ok(())
}

fn range() -> TransferError {
    TransferError {
        completed: 0,
        reason: TransferStop::Refused(Refusal::Fs(FsError::Range)),
    }
}

/// Fill `dst` from `offset`, reporting the initialized prefix on early EOF/error.
/// Allocation-free; each round trip uses the owner's existing bounded window.
/// Cancellation can leave the current request outstanding; keep its owner alive.
pub async fn read_exact_at(
    fs: &mut Fs<'_, '_>,
    file: &View,
    offset: u64,
    dst: &mut [u8],
) -> Result<(), TransferError> {
    checked_end(offset, dst.len())?;
    let mut completed = 0;
    while completed < dst.len() {
        match fs
            .read_at(file, offset + completed as u64, &mut dst[completed..])
            .await
        {
            Ok(0) => {
                return Err(TransferError {
                    completed,
                    reason: TransferStop::UnexpectedEof,
                });
            }
            Ok(count) => completed += count,
            Err(error) => {
                return Err(TransferError {
                    completed,
                    reason: TransferStop::Refused(error),
                });
            }
        }
    }
    Ok(())
}

/// Write all bytes at `offset` without truncation, atomic-save or durability claims.
/// The existing service acknowledges each chunk before the next is staged.
pub async fn write_all_at(
    fs: &mut Fs<'_, '_>,
    file: &View,
    offset: u64,
    src: &[u8],
) -> Result<(), TransferError> {
    checked_end(offset, src.len())?;
    let mut completed = 0;
    while completed < src.len() {
        match fs
            .write_at(file, offset + completed as u64, &src[completed..])
            .await
        {
            Ok(count) => completed += count,
            Err(error) => {
                return Err(TransferError {
                    completed,
                    reason: TransferStop::Refused(error),
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_is_checked_before_any_partial_write() {
        assert!(checked_end(u64::MAX, 0).is_ok());
        assert!(checked_end(u64::MAX - 7, 7).is_ok());
        let error = checked_end(u64::MAX - 7, 8).unwrap_err();
        assert_eq!(error.completed, 0);
        assert!(matches!(
            error.reason,
            TransferStop::Refused(Refusal::Fs(FsError::Range))
        ));
    }
}
