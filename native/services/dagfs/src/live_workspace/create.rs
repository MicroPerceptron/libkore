//! Host creation uses the native engine's unpublished object/edge/handle set.
use super::*;
use kore_workspace_sync::{Change, ChangeKind, Entry};

impl<C: WorkspaceChannel> WorkspaceMount<C> {
    pub(crate) async fn create<P: StorageProvider>(
        &mut self,
        fs: &mut FsService<P>,
        session: SessionId,
        request: &FsRequest,
    ) -> FsResult<Option<FsResponse>> {
        let FsRequest::Create {
            parent,
            name,
            kind,
            subtype,
        } = *request
        else {
            return Ok(None);
        };
        let Some(path) = fs.host_path_under_handle(session, parent, &name)? else {
            return Ok(None);
        };
        if fs.is_deferring(session) {
            return Err(FsError::Busy);
        }
        // The host manifest currently records plain files/directories only.
        if !subtype.is_plain() {
            return Err(FsError::Unsupported);
        }
        let live = self
            .live
            .as_mut()
            .ok_or(FsError::ProviderFailure(crate::ProviderError::Lost))?;
        if !live.writeback_enabled() {
            return Err(FsError::AccessDenied);
        }
        let digest = match kind {
            ObjectKind::File => *blake3::hash(&[]).as_bytes(),
            ObjectKind::Directory => Entry::directory(path.clone(), [0; 16]).digest,
        };
        let change = Change {
            kind: if kind.is_dir() {
                ChangeKind::CreateDirectory
            } else {
                ChangeKind::CreateFile
            },
            path: path.clone(),
            target: None,
            expected: [0; 32],
            replacement: if kind.is_dir() { [0; 32] } else { digest },
            replacement_len: 0,
            executable: false,
        };
        let mut candidate = None;
        self.reconcile_required = true;
        let result = fs
            .engine_ref()
            .create_with_commit(session, parent, &name, kind, subtype, async |stable| {
                candidate = Some(fs.prepare_host_create(HostNodeRecord {
                    stable,
                    path,
                    kind,
                    len: 0,
                    digest,
                    executable: false,
                    materialized: false,
                    revoked: false,
                })?);
                live.writeback(change, (kind == ObjectKind::File).then_some(&[][..]))
                    .await
                    .map(|_| ())
                    .map_err(workspace_error)
            })
            .await;
        if let Some(live) = self.live.take() {
            self.channel = Some(live.into_channel());
        }
        let opened = result?;
        fs.publish_host_nodes(
            candidate.expect("successful host creation has preflighted metadata"),
        );
        self.topology_dirty = true;
        Ok(Some(FsResponse::Opened {
            handle: opened.handle,
            stable_object: opened.stable_object,
            kind,
            subtype,
            len: opened.len,
            revision: opened.revision,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::live_workspace::namespace::tests::with_workspace;
    use crate::{Name, ObjectSubtype, Rights, block_on};
    use core::future::Future;

    fn request(parent: Handle, name: &str, kind: ObjectKind) -> FsRequest {
        FsRequest::Create {
            parent,
            name: Name::new(name).unwrap(),
            kind,
            subtype: ObjectSubtype::PLAIN,
        }
    }

    #[test]
    fn cancelled_creation_refresh_retains_the_channel_and_closes_the_unreported_handle() {
        with_workspace(true, |mount, fs, session, root, _, _| {
            mount
                .live
                .as_mut()
                .unwrap()
                .channel_mut()
                .pause_attach_after_write = true;
            let request = request(root, "new", ObjectKind::File);
            let mut future = std::boxed::Box::pin(mount.before_request(fs, session, &request));
            let mut cx = core::task::Context::from_waker(core::task::Waker::noop());
            assert!(future.as_mut().poll(&mut cx).is_pending());
            drop(future);
            let (owner, unreported) = mount.unreported_creation.unwrap();
            assert_eq!(owner, session);
            assert!(fs.engine_ref().stat(session, unreported).is_ok());
            assert!(
                mount.channel.is_some(),
                "the handshake borrowed the retained transport"
            );
            assert!(mount.reconcile_required);
            mount.channel.as_mut().unwrap().pause_attach_after_write = false;
            assert_eq!(
                block_on(mount.before_request(fs, session, &FsRequest::Stat { handle: root })),
                Ok(None)
            );
            assert!(mount.unreported_creation.is_none());
            assert_eq!(
                fs.engine_ref().stat(session, unreported),
                Err(FsError::InvalidHandle)
            );
            assert!(
                fs.engine_ref()
                    .open_at(session, root, &Name::new("new").unwrap(), Rights::READ)
                    .is_ok()
            );
        });
    }

    #[test]
    fn cancelled_delta_poll_keeps_transport_and_requires_fresh_attachment() {
        with_workspace(true, |mount, fs, session, root, _, file| {
            mount.live.as_mut().unwrap().channel_mut().pause_delta = true;
            let mut future = std::boxed::Box::pin(mount.refresh(fs));
            let mut cx = core::task::Context::from_waker(core::task::Waker::noop());
            assert!(future.as_mut().poll(&mut cx).is_pending());
            drop(future);
            assert!(mount.live.is_some());
            assert!(mount.reconcile_required);
            assert_eq!(
                block_on(mount.read(fs, session, file.handle, 0, 0, 0, None)),
                Err(FsError::Busy)
            );
            mount.live.as_mut().unwrap().channel_mut().pause_delta = false;
            block_on(mount.refresh(fs)).unwrap();
            assert!(!mount.reconcile_required);
            assert!(fs.engine_ref().stat(session, root).is_ok());
        });
    }

    #[test]
    fn mounted_creation_returns_live_handles_and_records_the_acknowledged_snapshot() {
        with_workspace(true, |mount, fs, session, root, _, original| {
            let Some(FsResponse::Opened {
                handle: directory, ..
            }) = block_on(mount.before_request(
                fs,
                session,
                &request(root, "new", ObjectKind::Directory),
            ))
            .unwrap()
            else {
                panic!("directory response")
            };
            let Some(FsResponse::Opened {
                handle,
                stable_object,
                ..
            }) = block_on(mount.before_request(
                fs,
                session,
                &request(directory, "file", ObjectKind::File),
            ))
            .unwrap()
            else {
                panic!("file response")
            };
            assert_eq!(
                fs.engine_ref().stat(session, handle).unwrap().stable_object,
                stable_object
            );
            assert_eq!(
                fs.host_node_for_handle(session, handle)
                    .unwrap()
                    .unwrap()
                    .path
                    .as_str(),
                "new/file"
            );
            let recorded = block_on(
                WorkspaceStore::new(fs.engine_ref(), SessionId::new(0), session).recover_current(),
            )
            .unwrap()
            .unwrap();
            assert_eq!(
                recorded.snapshot,
                mount.live.as_ref().unwrap().manifest().snapshot_id()
            );
            fs.dispatch(
                session,
                &FsRequest::AttachWindow { slot: 0, bytes: 3 },
                None,
            );
            let mut bytes = *b"new";
            block_on(mount.write_file(
                fs,
                session,
                &FsRequest::WriteAt {
                    handle,
                    expected: crate::ANY_REVISION,
                    offset: 0,
                    window_offset: 0,
                    length: 3,
                },
                Some(&mut DataWindow::new(&mut bytes)),
            ))
            .unwrap();
            block_on(mount.refresh(fs)).unwrap();
            bytes.fill(0);
            assert_eq!(
                block_on(mount.read(
                    fs,
                    session,
                    handle,
                    0,
                    0,
                    3,
                    Some(&mut DataWindow::new(&mut bytes))
                )),
                Ok(None)
            );
            assert_eq!(
                fs.dispatch(
                    session,
                    &FsRequest::Read {
                        handle,
                        offset: 0,
                        window_offset: 0,
                        length: 3
                    },
                    Some(&mut DataWindow::new(&mut bytes))
                ),
                FsResponse::Transferred { count: 3 }
            );
            assert_eq!(&bytes, b"new");
            assert_eq!(
                block_on(mount.read(
                    fs,
                    session,
                    original.handle,
                    0,
                    0,
                    3,
                    Some(&mut DataWindow::new(&mut bytes))
                )),
                Ok(Some(FsResponse::Transferred { count: 3 }))
            );
            assert_eq!(bytes, [0, 1, 2]);
        });
    }

    #[test]
    fn mounted_creation_refuses_disabled_writeback_and_insufficient_native_rights() {
        with_workspace(false, |mount, fs, session, root, _, _| {
            assert_eq!(
                block_on(mount.before_request(
                    fs,
                    session,
                    &request(root, "new", ObjectKind::File)
                )),
                Err(FsError::AccessDenied)
            );
        });
        with_workspace(true, |mount, fs, session, root, _, _| {
            let readonly = fs
                .engine_ref()
                .open_at(session, root, &Name::new("src").unwrap(), Rights::READ)
                .unwrap();
            assert_eq!(
                block_on(mount.create(
                    fs,
                    session,
                    &request(readonly.handle, "new", ObjectKind::File)
                )),
                Err(FsError::AccessDenied)
            );
            assert_eq!(mount.channel.as_ref().unwrap().transactions, 0);
            assert!(!mount.channel.as_ref().unwrap().has_path(b"src/new"));
        });
    }

    #[test]
    fn mounted_creation_conflict_rolls_back_then_retries_without_lost_capacity() {
        with_workspace(true, |mount, fs, session, root, _, _| {
            mount.live.as_mut().unwrap().channel_mut().write_status =
                kore_workspace_sync::WorkspaceStatus::Conflict;
            let before = fs.engine_ref().object_count();
            assert_eq!(
                block_on(mount.create(fs, session, &request(root, "new", ObjectKind::Directory))),
                Err(FsError::Conflict)
            );
            assert_eq!(fs.engine_ref().object_count(), before);
            assert_eq!(
                fs.engine_ref()
                    .open_at(session, root, &Name::new("new").unwrap(), Rights::READ),
                Err(FsError::NotFound)
            );
            mount.channel.as_mut().unwrap().write_status = kore_workspace_sync::WorkspaceStatus::Ok;
            block_on(mount.refresh(fs)).unwrap();
            assert!(matches!(
                block_on(mount.before_request(
                    fs,
                    session,
                    &request(root, "new", ObjectKind::Directory)
                )),
                Ok(Some(FsResponse::Opened { .. }))
            ));
        });
    }

    #[test]
    fn cancelled_host_creation_requires_reimport_before_reusing_the_old_view() {
        with_workspace(true, |mount, fs, session, root, _, _| {
            mount.live.as_mut().unwrap().channel_mut().pause_after_write = true;
            let before = fs.engine_ref().object_count();
            let request = request(root, "new", ObjectKind::File);
            let mut future = std::boxed::Box::pin(mount.create(fs, session, &request));
            let mut cx = core::task::Context::from_waker(core::task::Waker::noop());
            assert!(future.as_mut().poll(&mut cx).is_pending());
            drop(future);
            assert_eq!(fs.engine_ref().object_count(), before);
            assert!(mount.reconcile_required);
            assert_eq!(
                fs.engine_ref()
                    .open_at(session, root, &Name::new("new").unwrap(), Rights::READ),
                Err(FsError::NotFound)
            );
            mount.live.as_mut().unwrap().channel_mut().pause_after_write = false;
            block_on(mount.refresh(fs)).unwrap();
            assert_eq!(
                fs.engine_ref().stat(session, root),
                Err(FsError::AccessLost)
            );
            let committed = block_on(
                WorkspaceStore::new(fs.engine_ref(), SessionId::new(0), session).recover_current(),
            )
            .unwrap()
            .unwrap();
            let root = fs
                .engine_ref()
                .open_grant(session, committed.broker_root, Rights::HOME)
                .unwrap();
            let created = fs
                .engine_ref()
                .open_at(
                    session,
                    root.handle,
                    &Name::new("new").unwrap(),
                    Rights::HOME,
                )
                .unwrap();
            assert_eq!(
                fs.host_node_for_handle(session, created.handle)
                    .unwrap()
                    .unwrap()
                    .len,
                0
            );
        });
    }
}
