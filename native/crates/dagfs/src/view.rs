//! Task views and authority edges: the cwd binding, sharing a route with
//! another task, and the named shares this task holds.
//!
//! Both capability-moving verbs here consume a one-shot endpoint the
//! launcher delivered with the founding: a cwd binding for an executable
//! whose profile permits it, a grant target for one that may share. A
//! refusal that never moved the endpoint hands it back, so the verb can be
//! retried; one that moved it cannot.

use kore_rt::svc::ReturnCallError;

use crate::{
    Fs, FsError, FsRequest, FsResponse, GrantEntry, Name, Refusal, Rights, Unconfirmed,
    read_grant_entry,
};

impl<'w, 't> Fs<'w, 't> {
    /// Resolve and structurally publish a replacement cwd view. The target
    /// path and handle stay between this executable and FS; the launcher
    /// learns only whether the operation succeeded and retains the
    /// corresponding immutable root it minted before launch.
    pub async fn bind_cwd(&mut self, path: &[u8]) -> Result<(), Refusal> {
        let rights = self.effective_rights(path).await?;
        let view = self.open(path, rights).await?;
        if !view.is_directory() {
            self.close(view).await;
            return Err(Refusal::Fs(FsError::NotDirectory));
        }
        let Some(binding) = self.view_binding.take() else {
            self.close(view).await;
            return Err(Refusal::Fs(FsError::AccessDenied));
        };
        let moved = self.client.bind_view(view.handle(), rights, binding).await;
        self.close(view).await;
        match moved {
            Ok(FsResponse::Done) => Ok(()),
            Ok(FsResponse::Refused(error)) => Err(Refusal::Fs(error)),
            Ok(_) => Err(Refusal::Pair),
            Err(ReturnCallError::NotMoved { capability, .. }) => {
                self.view_binding = Some(capability);
                Err(Refusal::Pair)
            }
            Err(ReturnCallError::AfterTransfer(_)) => Err(Refusal::Pair),
        }
    }

    /// Confirm that `position` — the route from this task view's root that
    /// the launcher spelled this line's paths from — still names the
    /// directory its cwd view is bound to, before the verb acts on any of
    /// them. The DAG keeps no parent edges, so a launcher folds `..` and
    /// rooted paths onto its remembered route; a rename of that directory or
    /// of one above it keeps the cwd view working but leaves the route
    /// naming whatever took the name since, and acting on it would act
    /// there. The anchor makes the service the judge ([`FsRequest::MatchView`]).
    ///
    /// A launch with no anchor has nothing to confirm, and a position must
    /// never arrive without one: unconfirmed, it would mean nothing.
    pub async fn confirm_position(&mut self, position: &[u8]) -> Result<(), Unconfirmed> {
        let refused = |refusal| Err(Unconfirmed::Refused(refusal));
        if self.view_anchor.is_none() {
            return if position.is_empty() {
                Ok(())
            } else {
                refused(Refusal::Fs(FsError::AccessDenied))
            };
        }
        let view = match self.open(position, Rights::READ).await {
            Ok(view) => view,
            // The remembered route now reaches nothing, or a file: the cwd
            // left it.
            Err(Refusal::Fs(FsError::NotFound | FsError::NotDirectory)) => {
                return Err(Unconfirmed::Moved);
            }
            Err(error) => return refused(error),
        };
        let Some(anchor) = self.view_anchor.take() else {
            self.close(view).await;
            return refused(Refusal::Pair);
        };
        let matched = self.client.match_view(view.handle(), anchor).await;
        self.close(view).await;
        match matched {
            Ok(FsResponse::Done) => Ok(()),
            // Bound elsewhere, or the view's own route was retired by a move.
            Ok(FsResponse::Refused(FsError::Conflict | FsError::AccessLost)) => {
                Err(Unconfirmed::Moved)
            }
            Ok(FsResponse::Refused(error)) => refused(Refusal::Fs(error)),
            Ok(_) => refused(Refusal::Pair),
            Err(ReturnCallError::NotMoved { capability, .. }) => {
                self.view_anchor = Some(capability);
                refused(Refusal::Pair)
            }
            Err(ReturnCallError::AfterTransfer(_)) => refused(Refusal::Pair),
        }
    }

    /// Share the object at `path` into the recipient task view the
    /// launcher selected, displayed there as `display_name` with `rights`.
    /// The route is opened with `SHARE` in addition to `rights`, so a route
    /// that may not be re-shared refuses before anything moves.
    pub async fn share(
        &mut self,
        path: &[u8],
        display_name: Name,
        rights: Rights,
    ) -> Result<(), Refusal> {
        let view = self.open(path, rights.union(Rights::SHARE)).await?;
        let Some(target) = self.grant_target.take() else {
            self.close(view).await;
            return Err(Refusal::Fs(FsError::AccessDenied));
        };
        let result = self
            .client
            .grant_view(view.handle(), rights, display_name, target)
            .await;
        self.close(view).await;
        match result {
            Ok(FsResponse::Granted { .. }) => Ok(()),
            Ok(FsResponse::Refused(error)) => Err(Refusal::Fs(error)),
            Ok(_) => Err(Refusal::Pair),
            Err(ReturnCallError::NotMoved { capability, .. }) => {
                self.grant_target = Some(capability);
                Err(Refusal::Pair)
            }
            Err(ReturnCallError::AfterTransfer(_)) => Err(Refusal::Pair),
        }
    }

    /// Retire the named share this task holds.
    pub async fn unshare(&mut self, name: &Name) -> Result<(), Refusal> {
        let Some(grant) = self.named_grant(name).await? else {
            return Err(Refusal::Fs(FsError::NotFound));
        };
        self.checked(&FsRequest::Revoke { grant }).await?;
        Ok(())
    }

    /// List this task's authority edges into the window; yields the entry
    /// count. Read each with [`Fs::share_entry`] before the next request.
    pub async fn shares(&mut self) -> Result<u32, Refusal> {
        match self
            .checked(&FsRequest::ListGrants { window_offset: 0 })
            .await?
        {
            FsResponse::GrantsListed { entries } => Ok(entries),
            _ => Err(Refusal::Pair),
        }
    }

    /// The `index`th authority edge of the page the last [`Fs::shares`]
    /// landed. The private root is among them (`is_private_root`).
    pub fn share_entry(&self, index: usize) -> Result<GrantEntry, Refusal> {
        read_grant_entry(&self.window[..], 0, index).map_err(Refusal::Fs)
    }
}
