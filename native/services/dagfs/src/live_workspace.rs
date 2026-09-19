//! The user-store composition of the verified HostLink client and native
//! DagFS authority. The mounted store owns the affine channel across retries;
//! client requests carry only their ordinary session-bound handles.

mod create;
mod namespace;
mod writeback;

use alloc::boxed::Box;
use heapless::Vec;
use kore_workspace_sync::{CanonicalPath, EntryKind, Policy, bounds::MAX_ENTRIES};

use crate::{
    FsError, FsRequest, FsResponse, FsResult, Handle, ObjectKind, SessionId, StorageProvider,
    host_workspace::{HostWorkspace, HostWorkspaceError, WorkspaceChannel, stable_live_object},
    service::{FsService, HostNodeRecord},
    window::DataWindow,
    workspace::WorkspaceStore,
};

pub(crate) struct WorkspaceMount<C: WorkspaceChannel> {
    channel: Option<C>,
    live: Option<Box<HostWorkspace<C>>>,
    topology_dirty: bool,
    // Set before a mutating async operation can publish to the host. It is
    // retained even when that future is cancelled before returning a result.
    reconcile_required: bool,
    // Creation has committed, but its final snapshot refresh has not yet
    // returned the handle to the client. Retain exact custody across cancel.
    unreported_creation: Option<(SessionId, Handle)>,
}

impl<C: WorkspaceChannel> WorkspaceMount<C> {
    pub(crate) const fn empty() -> Self {
        Self {
            channel: None,
            live: None,
            topology_dirty: false,
            reconcile_required: false,
            unreported_creation: None,
        }
    }

    pub(crate) fn new(channel: C) -> Self {
        Self {
            channel: Some(channel),
            live: None,
            topology_dirty: false,
            reconcile_required: false,
            unreported_creation: None,
        }
    }

    pub(crate) fn topology_dirty(&self) -> bool {
        self.topology_dirty
    }

    pub(crate) fn topology_committed(&mut self) {
        self.topology_dirty = false;
    }

    /// Retry without losing the only founded workspace face. A failed
    /// refresh invalidates the verified range cache and returns a refusal;
    /// callers may continue unrelated native operations, but never satisfy
    /// a host read from its empty local topology placeholder.
    pub(crate) async fn refresh<P: StorageProvider>(
        &mut self,
        fs: &mut FsService<P>,
    ) -> FsResult<()> {
        if self.reconcile_required {
            if let Some(live) = self.live.take() {
                self.channel = Some(live.into_channel());
            }
            self.reconcile_required = false;
        }
        if self.live.is_some() {
            self.reconcile_required = true;
            match self
                .live
                .as_mut()
                .expect("attached workspace")
                .poll_delta()
                .await
            {
                Ok(false) => {
                    self.reconcile_required = false;
                    return Ok(());
                }
                Ok(true) => {}
                Err(error) => {
                    self.channel =
                        Some(self.live.take().expect("attached workspace").into_channel());
                    return Err(workspace_error(error));
                }
            }
        } else {
            if self.channel.is_none() {
                return Ok(());
            }
            self.reconcile_required = true;
            let live = HostWorkspace::attach_in_slot(&mut self.channel, Policy::default())
                .await
                .map_err(workspace_error)?;
            self.live = Some(Box::new(live));
        }
        let result = Self::install(fs, self.live.as_mut().expect("attached workspace")).await;
        if result.is_ok() {
            self.topology_dirty = true;
            self.reconcile_required = false;
        } else {
            self.channel = Some(self.live.take().expect("attached workspace").into_channel());
        }
        result
    }

    async fn install<P: StorageProvider>(
        fs: &mut FsService<P>,
        live: &mut HostWorkspace<C>,
    ) -> FsResult<()> {
        // A write acknowledgement changes the host snapshot without changing
        // native object identities. Rebind when the entire verified topology
        // still matches; an external topology change requires a new import.
        if fs.host_topology_matches(live.manifest()) {
            WorkspaceStore::new(
                fs.engine_ref(),
                SessionId::new(0),
                SessionId::new(crate::service::contract::ADMIN_VIA),
            )
            .record_live_snapshot(live.manifest(), fs.host_root_id().ok_or(FsError::Conflict)?)
            .await?;
            live.bind_kore_objects(fs.host_file_bindings()?.as_slice())
                .map_err(workspace_error)?;
            return Ok(());
        }
        let importer = SessionId::new(0);
        let admin = SessionId::new(crate::service::contract::ADMIN_VIA);
        if !fs.ensure_session(importer) || !fs.ensure_session(admin) {
            return Err(FsError::OutOfObjects);
        }
        let imported = WorkspaceStore::new(fs.engine_ref(), importer, admin)
            .import_live_topology(live.manifest(), Policy::default())
            .await?;
        let mut bindings = Vec::<(CanonicalPath, [u8; 32]), MAX_ENTRIES>::new();
        let mut records = Vec::<HostNodeRecord, MAX_ENTRIES>::new();
        for node in imported.nodes {
            let kind = match node.kind {
                EntryKind::File => ObjectKind::File,
                EntryKind::Directory => ObjectKind::Directory,
            };
            if kind == ObjectKind::File {
                bindings
                    .push((node.path.clone(), stable_live_object(node.stable_object)))
                    .map_err(|_| FsError::OutOfObjects)?;
            }
            records
                .push(HostNodeRecord {
                    stable: node.stable_object,
                    path: node.path,
                    kind,
                    len: u64::from(node.len),
                    digest: node.digest,
                    executable: node.executable,
                    materialized: false,
                    revoked: false,
                })
                .map_err(|_| FsError::OutOfObjects)?;
        }
        live.bind_kore_objects(bindings.as_slice())
            .map_err(workspace_error)?;
        fs.install_imported_workspace(imported.commit.broker_root)?;
        fs.replace_host_nodes(imported.root_stable_object, records.as_slice())?;
        Ok(())
    }

    /// Return `None` only for an ordinary native read. The engine rechecks
    /// READ authority before any verified cache or HostLink access. A numeric
    /// handle from another session cannot select a cached host range.
    pub(crate) async fn read<P: StorageProvider>(
        &mut self,
        fs: &FsService<P>,
        session: SessionId,
        handle: Handle,
        offset: u64,
        window_offset: u32,
        length: u32,
        window: Option<&mut DataWindow<'_>>,
    ) -> FsResult<Option<FsResponse>> {
        let Some(node) = fs.host_node_for_handle(session, handle)? else {
            return Ok(None);
        };
        if self.reconcile_required {
            return Err(if self.live.is_some() {
                FsError::Busy
            } else {
                FsError::ProviderFailure(crate::ProviderError::Lost)
            });
        }
        if node.revoked {
            return Err(FsError::AccessLost);
        }
        if node.kind != ObjectKind::File {
            return Err(FsError::IsDirectory);
        }
        if node.materialized {
            return Ok(None);
        }
        fs.engine_ref().stat(session, handle)?;
        let live = self
            .live
            .as_mut()
            .ok_or(FsError::ProviderFailure(crate::ProviderError::Lost))?;
        if fs.attachment(session).is_none() {
            return Err(FsError::NoWindow);
        }
        let dst = window
            .ok_or(FsError::NoWindow)?
            .destination(window_offset, length)?;
        offset
            .checked_add(u64::from(length))
            .ok_or(FsError::Range)?;
        if offset >= node.len || dst.is_empty() {
            return Ok(Some(FsResponse::Transferred { count: 0 }));
        }
        let count = dst
            .len()
            .min((node.len - offset) as usize)
            .min(fs.transfer_limit() as usize);
        let offset = u32::try_from(offset).map_err(|_| FsError::Range)?;
        match live
            .read(stable_live_object(node.stable), offset, &mut dst[..count])
            .await
        {
            Ok(count) => Ok(Some(FsResponse::Transferred {
                count: count as u32,
            })),
            Err(error) => {
                // Uncertain source state retires the cache before another
                // request can hit it. The affine channel remains retryable.
                if let Some(live) = self.live.take() {
                    self.channel = Some(live.into_channel());
                }
                Err(workspace_error(error))
            }
        }
    }

    /// Obtain complete verified bytes for an ordinary native immutable
    /// version. Hydration has its own unpublished placement and preserves the
    /// source revision, so the engine's existing snapshot/COW path stays the
    /// sole owner of version leases.
    pub(crate) async fn materialize<P: StorageProvider>(
        &mut self,
        fs: &mut FsService<P>,
        session: SessionId,
        handle: Handle,
        expected: crate::Revision,
        required: crate::Rights,
    ) -> FsResult<()> {
        let Some(mut node) = fs.host_node_for_handle(session, handle)? else {
            return Ok(());
        };
        if self.reconcile_required {
            return Err(if self.live.is_some() {
                FsError::Busy
            } else {
                FsError::ProviderFailure(crate::ProviderError::Lost)
            });
        }
        if node.revoked {
            return Err(FsError::AccessLost);
        }
        if node.kind != ObjectKind::File {
            return Err(FsError::IsDirectory);
        }
        if node.materialized {
            return Ok(());
        }
        let live = self
            .live
            .as_mut()
            .ok_or(FsError::ProviderFailure(crate::ProviderError::Lost))?;
        let source = stable_live_object(node.stable);
        let result = fs
            .engine_ref()
            .hydrate_host_file(
                session,
                handle,
                crate::engine::Hydration {
                    len: node.len,
                    digest: node.digest,
                    expected,
                    required,
                },
                async |offset, out: &mut [u8]| {
                    let offset = u32::try_from(offset).map_err(|_| FsError::Range)?;
                    live.read(source, offset, out)
                        .await
                        .map_err(workspace_error)
                },
            )
            .await;
        match result {
            Ok(_) => {
                node.materialized = true;
                fs.update_host_node(node)?;
                self.topology_dirty = true;
                Ok(())
            }
            Err(error) => {
                // Native authority/quota/stale-revision refusals do not
                // invalidate a verified source. A source/provider failure
                // does: reattach before another read can use its cache.
                if matches!(error, FsError::ProviderFailure(_) | FsError::Range) {
                    if let Some(live) = self.live.take() {
                        self.channel = Some(live.into_channel());
                    }
                }
                Err(error)
            }
        }
    }

    /// Used by both mounted provider variants after user/system routing. The
    /// primary service retains the mapping owner throughout this scoped read.
    pub(crate) async fn before_request<P: StorageProvider>(
        &mut self,
        fs: &mut FsService<P>,
        session: SessionId,
        request: &FsRequest,
    ) -> FsResult<Option<FsResponse>> {
        if let Some((owner, handle)) = self.unreported_creation {
            match fs.engine_ref().close(owner, handle) {
                Ok(()) | Err(FsError::InvalidHandle) => self.unreported_creation = None,
                Err(error) => return Err(error),
            }
        }
        let touches_host = fs.request_touches_host(session, request)?;
        if self.reconcile_required
            || matches!(
                request,
                FsRequest::OpenRoot { .. }
                    | FsRequest::OpenGrant { .. }
                    | FsRequest::OpenAt { .. }
                    | FsRequest::List { .. }
                    | FsRequest::Stat { .. }
                    | FsRequest::Read { .. }
                    | FsRequest::Search { .. }
                    | FsRequest::LendVersion { .. }
                    | FsRequest::LendExecutable { .. }
                    | FsRequest::WriteAt { .. }
                    | FsRequest::Append { .. }
                    | FsRequest::Resize { .. }
                    | FsRequest::Rename { .. }
                    | FsRequest::Relocate { .. }
                    | FsRequest::Unlink { .. }
                    | FsRequest::DetachEdge { .. }
                    | FsRequest::Create { .. }
            )
        {
            let result = self.refresh(fs).await;
            if touches_host {
                result?;
            }
        }
        if matches!(request, FsRequest::Create { .. }) {
            let response = self.create(fs, session, request).await?;
            if let Some(FsResponse::Opened {
                handle,
                stable_object,
                ..
            }) = response
            {
                self.unreported_creation = Some((session, handle));
                if self.reconcile_required {
                    self.refresh(fs).await?;
                }
                if fs.engine_ref().stable_object_for_handle(session, handle)? != stable_object {
                    return Err(FsError::AccessLost);
                }
                self.unreported_creation = None;
            }
            return Ok(response);
        }
        if matches!(
            request,
            FsRequest::Rename { .. }
                | FsRequest::Relocate { .. }
                | FsRequest::Unlink { .. }
                | FsRequest::DetachEdge { .. }
        ) {
            let response = self.write_namespace(fs, session, request).await?;
            if response.is_some() && self.reconcile_required {
                self.refresh(fs).await?;
            }
            return Ok(response);
        }
        if matches!(
            request,
            FsRequest::WriteAt { .. } | FsRequest::Append { .. } | FsRequest::Resize { .. }
        ) {
            let mut window = fs.mapped_window_span(session).map(|span| {
                // SAFETY: the affine mapping owner remains in fs, exclusively
                // borrowed for this request. Copy its input before awaiting.
                DataWindow::new(unsafe {
                    core::slice::from_raw_parts_mut(span.base.as_ptr(), span.len)
                })
            });
            let response = self
                .write_file(fs, session, request, window.as_mut())
                .await?;
            // A successful reply must persist the acknowledged host snapshot
            // with its native topology/bytes, not the preceding snapshot's
            // recovery record. The mounted owner commits topology_dirty before
            // publishing this response to the client.
            if response.is_some() && self.reconcile_required {
                self.refresh(fs).await?;
            }
            return Ok(response);
        }
        match *request {
            FsRequest::LendVersion { handle, .. } => {
                self.materialize(
                    fs,
                    session,
                    handle,
                    crate::ANY_REVISION,
                    crate::Rights::READ,
                )
                .await?;
            }
            FsRequest::LendExecutable { handle, .. } => {
                self.materialize(
                    fs,
                    session,
                    handle,
                    crate::ANY_REVISION,
                    crate::Rights::READ.union(crate::Rights::BIND_EXECUTABLE),
                )
                .await?;
            }
            _ => {}
        }
        if let FsRequest::Read {
            handle,
            offset,
            window_offset,
            length,
        } = *request
        {
            let span = fs.mapped_window_span(session);
            let mut window = span.map(|span| {
                // SAFETY: the affine mapped capability stays in `fs` and this
                // service borrow spans the read. No second window is created
                // or request dispatched until this borrow has ended.
                DataWindow::new(unsafe {
                    core::slice::from_raw_parts_mut(span.base.as_ptr(), span.len)
                })
            });
            return self
                .read(
                    fs,
                    session,
                    handle,
                    offset,
                    window_offset,
                    length,
                    window.as_mut(),
                )
                .await;
        }
        Ok(None)
    }
}

fn workspace_error<E>(error: HostWorkspaceError<E>) -> FsError {
    match error {
        HostWorkspaceError::Conflict | HostWorkspaceError::Stale => FsError::Conflict,
        HostWorkspaceError::Refused | HostWorkspaceError::Revoked => FsError::AccessLost,
        HostWorkspaceError::Integrity => FsError::ProviderFailure(crate::ProviderError::Integrity),
        HostWorkspaceError::Bounds | HostWorkspaceError::Protocol => FsError::Range,
        HostWorkspaceError::Backpressure => FsError::Busy,
        HostWorkspaceError::Transport(_) | HostWorkspaceError::Unavailable => {
            FsError::ProviderFailure(crate::ProviderError::Lost)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Name, RamProvider, Rights, block_on, host_workspace::tests::FakeChannel};

    fn with_mount(
        test: impl FnOnce(
            WorkspaceMount<FakeChannel>,
            crate::service::RamService,
            SessionId,
            crate::OpenedObject,
        ) + Send
        + 'static,
    ) {
        with_provider(RamProvider::default, test);
    }

    fn with_provider<P: StorageProvider + 'static>(
        provider: impl FnOnce() -> P + Send + 'static,
        test: impl FnOnce(WorkspaceMount<FakeChannel>, FsService<P>, SessionId, crate::OpenedObject)
        + Send
        + 'static,
    ) {
        // This unoptimized fixture owns both the engine and the bounded
        // attach/import futures by value. It is not a guest stack estimate.
        std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
                let mut mount = WorkspaceMount::new(FakeChannel::new());
                let mut fs = FsService::new(crate::Engine::new(provider()));
                block_on(mount.refresh(&mut fs)).unwrap();
                assert!(mount.topology_dirty());
                let session = SessionId::new(crate::service::contract::ADMIN_VIA);
                let commit = block_on(
                    WorkspaceStore::new(fs.engine_ref(), SessionId::new(0), session)
                        .recover_current(),
                )
                .unwrap()
                .unwrap();
                let root = fs
                    .engine_ref()
                    .open_grant(session, commit.broker_root, Rights::READ_ONLY)
                    .unwrap();
                let src = fs
                    .engine_ref()
                    .open_at(
                        session,
                        root.handle,
                        &Name::new("src").unwrap(),
                        Rights::READ_ONLY,
                    )
                    .unwrap();
                let file = fs
                    .engine_ref()
                    .open_at(
                        session,
                        src.handle,
                        &Name::new("main.rs").unwrap(),
                        Rights::READ_ONLY,
                    )
                    .unwrap();
                fs.dispatch(
                    session,
                    &FsRequest::AttachWindow { slot: 0, bytes: 64 },
                    None,
                );
                test(mount, fs, session, file);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn live_workspace_mount_reads_verified_bytes_through_native_handle_and_window() {
        with_mount(|mut mount, mut fs, session, file| {
            assert_eq!(fs.engine_ref().stat(session, file.handle).unwrap().len, 0);
            assert!(matches!(
                fs.dispatch(
                    session,
                    &FsRequest::Stat {
                        handle: file.handle
                    },
                    None
                ),
                FsResponse::Stat { len: 4127, .. }
            ));
            let mut bytes = [0xa5; 64];
            let mut window = DataWindow::new(&mut bytes);
            assert_eq!(
                block_on(mount.read(&fs, session, file.handle, 16, 8, 32, Some(&mut window))),
                Ok(Some(FsResponse::Transferred { count: 32 }))
            );
            assert_eq!(&bytes[..8], &[0xa5; 8]);
            assert_eq!(&bytes[8..40], &(16..48).collect::<std::vec::Vec<u8>>());
            assert_eq!(&bytes[40..], &[0xa5; 24]);
            assert_eq!(fs.engine_ref().stat(session, file.handle).unwrap().len, 0);
        });
    }

    #[test]
    fn live_workspace_mount_revalidates_authority_before_cached_reads() {
        with_mount(|mut mount, mut fs, session, file| {
            let mut bytes = [0; 64];
            let mut window = DataWindow::new(&mut bytes);
            block_on(mount.read(&fs, session, file.handle, 0, 0, 32, Some(&mut window))).unwrap();
            // Closing the handle retires its native proof even with a verified
            // range cached. Neither its old bits nor another session select it.
            fs.engine().close(session, file.handle).unwrap();
            assert_eq!(
                block_on(mount.read(&fs, session, file.handle, 0, 0, 32, Some(&mut window))),
                Err(FsError::InvalidHandle)
            );
            let other = SessionId::new(2);
            assert!(fs.ensure_session(other));
            assert_eq!(
                block_on(mount.read(&fs, other, file.handle, 0, 0, 32, Some(&mut window))),
                Err(FsError::InvalidHandle)
            );
        });
    }

    #[test]
    fn live_workspace_mount_refuses_window_overflow_and_lost_source_without_empty_eof() {
        with_mount(|mut mount, mut fs, session, file| {
            let mut bytes = [0xa5; 64];
            let mut window = DataWindow::new(&mut bytes);
            assert_eq!(
                block_on(mount.read(&fs, session, file.handle, 0, 63, 2, Some(&mut window))),
                Err(FsError::Range)
            );
            block_on(mount.read(&fs, session, file.handle, 0, 0, 32, Some(&mut window))).unwrap();
            // This peer fixture refuses delta polling. A refresh failure must
            // discard its earlier verified cache, while retaining the channel.
            assert_eq!(
                block_on(mount.refresh(&mut fs)),
                Err(FsError::ProviderFailure(crate::ProviderError::Lost))
            );
            assert!(mount.live.is_none());
            assert!(mount.channel.is_some());
            assert_eq!(
                block_on(mount.read(&fs, session, file.handle, 0, 0, 32, Some(&mut window))),
                Err(FsError::ProviderFailure(crate::ProviderError::Lost))
            );
            assert_eq!(
                block_on(mount.read(&fs, session, file.handle, u64::MAX, 0, 1, Some(&mut window))),
                Err(FsError::ProviderFailure(crate::ProviderError::Lost))
            );
            block_on(mount.refresh(&mut fs)).unwrap();
            assert_eq!(
                block_on(mount.read(&fs, session, file.handle, 4096, 0, 32, Some(&mut window))),
                Ok(Some(FsResponse::Transferred { count: 31 }))
            );
        });
    }
    #[test]
    fn live_workspace_materialization_lends_a_native_version_without_advancing_revision() {
        // The native RAM provider deliberately caps each file at 4 KiB.
        // Use the actual block provider for this multi-range (4127-byte)
        // snapshot, rather than shrinking the protocol fixture to fit RAM.
        with_provider(
            || {
                use kore_driver::block::{BlockChannel, CompletionMode, memory::MemoryBlockDevice};
                let device = MemoryBlockDevice::<327680, 8192>::new(512, 16).unwrap();
                let channel = BlockChannel::<_, 3, 8>::new(device, CompletionMode::Poll).unwrap();
                let start = crate::persistence::bootstrap_data_start(channel.geometry().block_size)
                    .unwrap();
                crate::BlockProvider::new_with_data_start(channel, start).unwrap()
            },
            |mut mount, mut fs, session, file| {
                block_on(mount.materialize(
                    &mut fs,
                    session,
                    file.handle,
                    file.revision,
                    Rights::READ,
                ))
                .unwrap();
                let info = fs.engine_ref().stat(session, file.handle).unwrap();
                assert_eq!(info.len, 4127);
                assert_eq!(info.revision, file.revision);
                let mut bytes = [0; 64];
                let mut window = DataWindow::new(&mut bytes);
                let FsResponse::VersionLent { lease } = fs.dispatch(
                    session,
                    &FsRequest::LendVersion {
                        handle: file.handle,
                        window_offset: 0,
                    },
                    Some(&mut window),
                ) else {
                    panic!("native immutable version must be reachable after hydration");
                };
                assert_eq!(
                    fs.dispatch(
                        session,
                        &FsRequest::ReadVersion {
                            lease,
                            offset: 4096,
                            window_offset: 0,
                            length: 32
                        },
                        Some(&mut window)
                    ),
                    FsResponse::Transferred { count: 31 }
                );
                assert_eq!(
                    fs.dispatch(session, &FsRequest::ReleaseVersion { lease }, None),
                    FsResponse::Done
                );
                assert_eq!(&bytes[..31], &(0..31).collect::<std::vec::Vec<u8>>());
            },
        );
    }

    #[test]
    fn live_workspace_executable_materialization_requires_native_bind_authority() {
        with_mount(|mut mount, mut fs, session, file| {
            assert_eq!(
                block_on(mount.materialize(
                    &mut fs,
                    session,
                    file.handle,
                    file.revision,
                    Rights::READ.union(Rights::BIND_EXECUTABLE)
                )),
                Err(FsError::AccessDenied)
            );
            assert_eq!(fs.engine_ref().stat(session, file.handle).unwrap().len, 0);
            assert!(
                !fs.host_node_for_handle(session, file.handle)
                    .unwrap()
                    .unwrap()
                    .materialized
            );
        });
    }
    #[test]
    fn live_workspace_materialization_preserves_the_ram_provider_limit() {
        with_mount(|mut mount, mut fs, session, file| {
            assert_eq!(
                block_on(mount.materialize(
                    &mut fs,
                    session,
                    file.handle,
                    file.revision,
                    Rights::READ
                )),
                Err(FsError::OutOfStorage)
            );
            assert_eq!(fs.engine_ref().stat(session, file.handle).unwrap().len, 0);
            assert!(
                !fs.host_node_for_handle(session, file.handle)
                    .unwrap()
                    .unwrap()
                    .materialized
            );
        });
    }
}
