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
    Fs, FsError, FsRequest, FsResponse, GrantEntry, GrantId, Name, Refusal, Rights,
    read_grant_entry,
};

impl<'w, 't> Fs<'w, 't> {
    /// Resolve and structurally publish a replacement cwd view. The target
    /// path and handle stay between this executable and FS; the launcher
    /// learns only whether the operation succeeded and retains the
    /// corresponding immutable root it minted before launch.
    pub async fn bind_cwd(&mut self, path: &[u8]) -> Result<(), Refusal> {
        let rights = self.effective_rights(path).await?;
        self.bind_cwd_with_rights(path, rights).await
    }

    /// Bind exactly the requested rights on a directory. Opening the route
    /// with SHARE enforces delegation authority before the one-shot endpoint moves; a
    /// request for rights the source does not hold refuses rather than
    /// silently publishing a weaker or broader view.
    pub async fn bind_cwd_with_rights(
        &mut self,
        path: &[u8],
        rights: Rights,
    ) -> Result<(), Refusal> {
        self.bind_cwd_mode(path, rights, false).await
    }

    /// Bind a directory without exposing the issuer's other named grants.
    pub async fn bind_cwd_isolated(&mut self, path: &[u8], rights: Rights) -> Result<(), Refusal> {
        self.bind_cwd_mode(path, rights, true).await
    }

    /// Bind one exact non-directory object as an isolated invocation root.
    /// The recipient opens `/` to use that object; no parent, sibling or
    /// issuer's named grants accompany it. Object identity is pinned by the
    /// native handle before this command exits, not re-resolved by pathname.
    pub async fn bind_object_isolated(
        &mut self,
        path: &[u8],
        rights: Rights,
    ) -> Result<(), Refusal> {
        if !rights.contains(Rights::READ) || rights.contains(Rights::MUTATE_NAMESPACE) {
            return Err(Refusal::Fs(FsError::AccessDenied));
        }
        let view = self.open(path, rights.union(Rights::SHARE)).await?;
        if view.is_directory() {
            self.close(view).await;
            return Err(Refusal::Fs(FsError::IsDirectory));
        }
        self.bind_selected(view, rights, true).await
    }

    /// Start a private read-only namespace containing one explicitly named
    /// resource. Further resources require ordinary target-authenticated grants.
    pub async fn bind_resource_view(
        &mut self,
        path: &[u8],
        rights: Rights,
        name: Name,
    ) -> Result<(), Refusal> {
        if !rights.contains(Rights::READ) || !rights.is_subset_of(Rights::HOME) {
            return Err(Refusal::Fs(FsError::AccessDenied));
        }
        let view = self.open(path, rights.union(Rights::SHARE)).await?;
        self.bind_selected_mode(view, rights, crate::ViewBindingMode::Resources(name))
            .await
    }

    /// Bind only the explicitly named grant, without a same-named containment
    /// entry taking precedence. Parent admission uses this for imported roots.
    pub async fn bind_named_grant_isolated(
        &mut self,
        name: &Name,
        rights: Rights,
    ) -> Result<(), Refusal> {
        if !rights.contains(Rights::READ) {
            return Err(Refusal::Fs(FsError::AccessDenied));
        }
        let grant = self
            .named_grant(name)
            .await?
            .ok_or(Refusal::Fs(FsError::NotFound))?;
        let view = self.open_grant(grant, rights.union(Rights::SHARE)).await?;
        self.bind_directory(view, rights, true).await
    }

    async fn bind_cwd_mode(
        &mut self,
        path: &[u8],
        rights: Rights,
        isolated: bool,
    ) -> Result<(), Refusal> {
        if !rights.contains(Rights::READ) {
            return Err(Refusal::Fs(FsError::AccessDenied));
        }
        let view = self.open(path, rights.union(Rights::SHARE)).await?;
        self.bind_directory(view, rights, isolated).await
    }

    async fn bind_directory(
        &mut self,
        view: crate::View,
        rights: Rights,
        isolated: bool,
    ) -> Result<(), Refusal> {
        if !view.is_directory() {
            self.close(view).await;
            return Err(Refusal::Fs(FsError::NotDirectory));
        }
        self.bind_selected(view, rights, isolated).await
    }

    async fn bind_selected(
        &mut self,
        view: crate::View,
        rights: Rights,
        isolated: bool,
    ) -> Result<(), Refusal> {
        let mode = if isolated {
            crate::ViewBindingMode::Isolated
        } else {
            crate::ViewBindingMode::Inherited
        };
        self.bind_selected_mode(view, rights, mode).await
    }

    async fn bind_selected_mode(
        &mut self,
        view: crate::View,
        rights: Rights,
        mode: crate::ViewBindingMode,
    ) -> Result<(), Refusal> {
        let Some(binding) = self.view_binding.take() else {
            self.close(view).await;
            return Err(Refusal::Fs(FsError::AccessDenied));
        };
        let moved = self
            .client
            .bind_view_mode(view.handle(), rights, mode, binding)
            .await;
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
        self.share_grant(path, display_name, rights)
            .await
            .map(|_| ())
    }

    /// Preserve the exact edge returned by the service for native parent
    /// bookkeeping. Receiving its ID does not confer authority over that edge.
    pub async fn share_grant(
        &mut self,
        path: &[u8],
        display_name: Name,
        rights: Rights,
    ) -> Result<GrantId, Refusal> {
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
            Ok(FsResponse::Granted { grant }) => Ok(grant),
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

    /// Refuse replacement-by-name. The final Revoke still checks the exact
    /// generational ID if the namespace changes after this lookup.
    pub async fn unshare_exact(&mut self, name: &Name, expected: GrantId) -> Result<(), Refusal> {
        if self.named_grant(name).await? != Some(expected) {
            return Err(Refusal::Fs(FsError::NotFound));
        }
        self.checked(&FsRequest::Revoke { grant: expected }).await?;
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
