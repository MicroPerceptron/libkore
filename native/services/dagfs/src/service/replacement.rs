//! Session-owned replacement staging shared by native and mounted dispatch.
use super::*;
use crate::Revision;

impl<P: StorageProvider> FsService<P> {
    pub(crate) async fn replacement_request(
        &mut self,
        session: SessionId,
        request: &FsRequest,
        window: Option<&mut DataWindow<'_>>,
    ) -> FsResult<FsResponse> {
        if request.needs_window() && self.attachment(session).is_none() {
            return Err(FsError::NoWindow);
        }
        Ok(match *request {
            FsRequest::BeginReplace {
                handle,
                length,
                expected,
            } => {
                let cell = self
                    .replacements
                    .get_mut(session.index() as usize)
                    .ok_or(FsError::AccessLost)?;
                if cell.is_some() {
                    return Err(FsError::Busy);
                }
                *cell = Some(
                    self.engine
                        .begin_replace(session, handle, length, expected)
                        .await?,
                );
                FsResponse::Done
            }
            FsRequest::WriteReplacement {
                handle,
                offset,
                window_offset,
                length,
            } => {
                let stage = self
                    .replacements
                    .get_mut(session.index() as usize)
                    .and_then(Option::as_mut)
                    .filter(|stage| stage.handle == handle)
                    .ok_or(FsError::InvalidHandle)?;
                let src = window
                    .ok_or(FsError::NoWindow)?
                    .source(window_offset, length)?;
                self.engine
                    .write_replacement(session, stage, offset, src)
                    .await?;
                FsResponse::Done
            }
            FsRequest::CommitReplace { handle } => {
                let cell = self
                    .replacements
                    .get_mut(session.index() as usize)
                    .ok_or(FsError::AccessLost)?;
                let stage = cell
                    .as_ref()
                    .filter(|stage| stage.handle == handle)
                    .ok_or(FsError::InvalidHandle)?;
                let revision = self.engine.commit_replace(session, stage).await?;
                // The object owns the placement from this point onward.
                *cell = None;
                FsResponse::Committed { revision }
            }
            FsRequest::AbortReplace { handle } => {
                let cell = self
                    .replacements
                    .get_mut(session.index() as usize)
                    .ok_or(FsError::AccessLost)?;
                if let Some(stage) = cell.as_ref() {
                    if stage.handle != handle {
                        return Err(FsError::InvalidHandle);
                    }
                    self.engine.abort_replace(stage).await?;
                    *cell = None;
                }
                FsResponse::Done
            }
            _ => return Err(FsError::Unsupported),
        })
    }

    pub(crate) async fn commit_host_replacement<C>(
        &mut self,
        session: SessionId,
        handle: Handle,
        commit_host: C,
    ) -> FsResult<Revision>
    where
        C: AsyncFnOnce(&[u8]) -> FsResult<()>,
    {
        let cell = self
            .replacements
            .get_mut(session.index() as usize)
            .ok_or(FsError::AccessLost)?;
        let stage = cell
            .as_ref()
            .filter(|stage| stage.handle == handle)
            .ok_or(FsError::InvalidHandle)?;
        let revision = self
            .engine
            .commit_host_replacement(session, stage, commit_host)
            .await?;
        *cell = None;
        Ok(revision)
    }
}
