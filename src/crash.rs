//! Fatal-process evidence and explicit parent-side persistence.
//!
//! Prepare the file while the parent is healthy; collect the bounded report,
//! reap the child, and only then await storage. A filesystem failure never
//! participates in stopping the process. The file view is caller-owned.
pub use kore_abi::crash::CrashRecord;
pub use kore_rt::crash::*;

/// A borrowed destination whose mandatory record extent has been initialized.
/// Preparation is explicit: it overwrites the first 1024 bytes with an invalid
/// record so a previous crash cannot be mistaken for the next process's report.
#[cfg(feature = "fs")]
pub struct PreparedDump<'a> {
    file: &'a crate::fs::View,
}

#[cfg(feature = "fs")]
impl<'a> PreparedDump<'a> {
    pub async fn prepare(
        fs: &mut crate::fs::Fs<'_, '_>,
        file: &'a crate::fs::View,
    ) -> Result<Self, crate::fs::TransferError> {
        crate::fs::write_all_at(fs, file, 0, &[0; kore_abi::crash::MIN_BUFFER_BYTES]).await?;
        fs.sync(file)
            .await
            .map_err(|error| crate::fs::TransferError {
                completed: kore_abi::crash::MIN_BUFFER_BYTES,
                reason: crate::fs::TransferStop::Refused(error),
            })?;
        Ok(Self { file })
    }

    /// Consume the prepared destination and acknowledge persistence only after
    /// the filesystem provider's explicit sync succeeds. Transport errors may
    /// follow a committed write: never automatically replay this operation.
    /// The caller must reap the child before calling this asynchronous method.
    pub async fn persist(
        self,
        fs: &mut crate::fs::Fs<'_, '_>,
        captured: &CrashBuffer<'_>,
    ) -> Result<(), PersistError> {
        let bytes = captured.captured_bytes().ok_or(PersistError::NoRecord)?;
        // prepare() durably invalidated the magic. Keep it invalid across
        // short writes and an interrupted body sync. Only publish the magic
        // after every other byte is durable; a partial magic remains invalid.
        let (magic, body) = bytes.split_at(8);
        crate::fs::write_all_at(fs, self.file, 8, body)
            .await
            .map_err(PersistError::BodyWrite)?;
        fs.sync(self.file).await.map_err(PersistError::BodySync)?;
        crate::fs::write_all_at(fs, self.file, 0, magic)
            .await
            .map_err(PersistError::PublishWrite)?;
        fs.sync(self.file).await.map_err(PersistError::PublishSync)
    }
}

#[cfg(feature = "fs")]
#[derive(Debug)]
pub enum PersistError {
    NoRecord,
    BodyWrite(crate::fs::TransferError),
    BodySync(crate::fs::Refusal),
    PublishWrite(crate::fs::TransferError),
    PublishSync(crate::fs::Refusal),
}
