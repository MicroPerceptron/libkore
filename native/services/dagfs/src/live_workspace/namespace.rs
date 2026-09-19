//! Host namespace transactions through native per-directory preparation.
use super::*;
use kore_workspace_sync::{Change, ChangeKind};

impl<C: WorkspaceChannel> WorkspaceMount<C> {
    pub(crate) async fn write_namespace<P: StorageProvider>(
        &mut self,
        fs: &mut FsService<P>,
        session: SessionId,
        request: &FsRequest,
    ) -> FsResult<Option<FsResponse>> {
        let (parent, name, destination) = match *request {
            FsRequest::Rename { parent, old, new } => (parent, old, Some((parent, new))),
            FsRequest::Relocate {
                src_parent,
                name,
                dst_parent,
                new_name,
            } => (src_parent, name, Some((dst_parent, new_name))),
            FsRequest::Unlink { parent, name } | FsRequest::DetachEdge { parent, name } => {
                (parent, name, None)
            }
            _ => return Ok(None),
        };
        let source = fs.host_path_under_handle(session, parent, &name)?;
        let target = destination
            .map(|(parent, name)| fs.host_path_under_handle(session, parent, &name))
            .transpose()?
            .flatten();
        let Some(path) = source else {
            return if target.is_some() {
                Err(FsError::AccessDenied)
            } else {
                Ok(None)
            };
        };
        if destination.is_some() != target.is_some() {
            return Err(FsError::AccessDenied);
        }
        let node = fs.host_node_by_path(&path).ok_or(FsError::NotFound)?;
        if fs.is_deferring(session) {
            return Err(FsError::Busy);
        }
        // No-op still needs the ordinary native name/rights validation, but
        // never publishes an invalid same-path host rename transaction.
        if target.as_ref() == Some(&path) {
            match *request {
                FsRequest::Rename { parent, old, new } => {
                    fs.engine_ref().rename(session, parent, &old, &new)?
                }
                FsRequest::Relocate {
                    src_parent,
                    name,
                    dst_parent,
                    new_name,
                } => fs
                    .engine_ref()
                    .relocate(session, src_parent, &name, dst_parent, &new_name)?,
                _ => unreachable!(),
            }
            return Ok(Some(FsResponse::Done));
        }
        let renamed = target
            .as_ref()
            .map(|target| fs.prepare_host_rename(&path, target))
            .transpose()?;
        let live = self
            .live
            .as_mut()
            .ok_or(FsError::ProviderFailure(crate::ProviderError::Lost))?;
        if !live.writeback_enabled() {
            return Err(FsError::AccessDenied);
        }
        let change = Change {
            kind: if target.is_some() {
                ChangeKind::Rename
            } else {
                ChangeKind::Delete
            },
            path: path.clone(),
            target,
            expected: node.digest,
            replacement: [0; 32],
            replacement_len: 0,
            executable: false,
        };
        let mut changes = kore_workspace_sync::ChangeSetBuilder::new(live.manifest().snapshot_id());
        if matches!(request, FsRequest::DetachEdge { .. }) {
            for change in fs.host_delete_changes(&path)? {
                changes.push(change).map_err(|_| FsError::Range)?;
            }
        } else {
            changes.push(change).map_err(|_| FsError::Range)?;
        }
        let changes = changes.finish().map_err(|_| FsError::Range)?;
        self.reconcile_required = true;
        let commit = async || {
            live.writeback_changes(changes, None)
                .await
                .map(|_| ())
                .map_err(workspace_error)
        };
        let result = match *request {
            FsRequest::Rename { parent, old, new } => {
                fs.engine_ref()
                    .rename_host_entry(session, parent, &old, &new, node.stable, commit)
                    .await
            }
            FsRequest::Relocate {
                src_parent,
                name,
                dst_parent,
                new_name,
            } => {
                fs.engine_ref()
                    .relocate_host_entry(
                        session,
                        src_parent,
                        &name,
                        dst_parent,
                        &new_name,
                        node.stable,
                        commit,
                    )
                    .await
            }
            FsRequest::Unlink { .. } | FsRequest::DetachEdge { .. } => {
                fs.engine_ref()
                    .remove_host_entry(
                        session,
                        parent,
                        &name,
                        node.stable,
                        matches!(request, FsRequest::Unlink { .. }),
                        commit,
                    )
                    .await
            }
            _ => unreachable!(),
        };
        if let Some(live) = self.live.take() {
            self.channel = Some(live.into_channel());
        }
        result?;
        if let Some(renamed) = renamed {
            fs.publish_host_nodes(renamed);
        } else {
            fs.retire_host_subtree(&path);
        }
        self.topology_dirty = true;
        Ok(Some(FsResponse::Done))
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::{Name, Rights, block_on, host_workspace::tests::FakeChannel};
    use core::future::Future;

    fn name(value: &str) -> Name {
        Name::new(value).unwrap()
    }

    pub(in crate::live_workspace) fn with_workspace(
        enabled: bool,
        test: impl FnOnce(
            &mut WorkspaceMount<FakeChannel>,
            &mut crate::service::RamService,
            SessionId,
            Handle,
            Handle,
            crate::OpenedObject,
        ) + Send
        + 'static,
    ) {
        std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
                let mut peer = FakeChannel::new().applying_writes();
                if enabled {
                    peer = peer.with_writeback()
                }
                let mut mount = WorkspaceMount::new(peer);
                let mut fs = crate::service::RamService::default();
                block_on(mount.refresh(&mut fs)).unwrap();
                let session = SessionId::new(crate::service::contract::ADMIN_VIA);
                let committed = block_on(
                    WorkspaceStore::new(fs.engine_ref(), SessionId::new(0), session)
                        .recover_current(),
                )
                .unwrap()
                .unwrap();
                let root = fs
                    .engine_ref()
                    .open_grant(session, committed.broker_root, Rights::HOME)
                    .unwrap()
                    .handle;
                let src = fs
                    .engine_ref()
                    .open_at(session, root, &name("src"), Rights::HOME)
                    .unwrap()
                    .handle;
                let file = fs
                    .engine_ref()
                    .open_at(session, src, &name("main.rs"), Rights::HOME)
                    .unwrap();
                test(&mut mount, &mut fs, session, root, src, file);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn mounted_namespace_response_records_the_acknowledged_snapshot_before_persistence() {
        std::thread::Builder::new()
            .stack_size(128 * 1024 * 1024)
            .spawn(|| {
                use kore_driver::block::{BlockChannel, CompletionMode, memory::MemoryBlockDevice};
                let device = MemoryBlockDevice::<327680, 8192>::new(512, 16).unwrap();
                let channel = BlockChannel::<_, 3, 8>::new(device, CompletionMode::Poll).unwrap();
                let start = crate::persistence::bootstrap_data_start(channel.geometry().block_size)
                    .unwrap();
                let provider = crate::BlockProvider::new_with_data_start(channel, start).unwrap();
                let volume = block_on(crate::persistence::PersistentFs::format(
                    provider,
                    SessionId::new(0),
                ))
                .unwrap();
                let (engine, mut durability) = volume.into_parts();
                let mut fs = FsService::new(engine);
                let mut mount =
                    WorkspaceMount::new(FakeChannel::new().applying_writes().with_writeback());
                block_on(mount.refresh(&mut fs)).unwrap();
                let session = SessionId::new(crate::service::contract::ADMIN_VIA);
                let before = block_on(
                    WorkspaceStore::new(fs.engine_ref(), SessionId::new(0), session)
                        .recover_current(),
                )
                .unwrap()
                .unwrap();
                let root = fs
                    .engine_ref()
                    .open_grant(session, before.broker_root, Rights::HOME)
                    .unwrap();
                let request = FsRequest::Rename {
                    parent: root.handle,
                    old: name("src"),
                    new: name("lib"),
                };
                assert_eq!(
                    block_on(mount.before_request(&mut fs, session, &request)),
                    Ok(Some(FsResponse::Done))
                );
                let Some(FsResponse::Opened {
                    stable_object: created_stable,
                    ..
                }) = block_on(mount.before_request(
                    &mut fs,
                    session,
                    &FsRequest::Create {
                        parent: root.handle,
                        name: name("created"),
                        kind: ObjectKind::File,
                        subtype: crate::ObjectSubtype::PLAIN,
                    },
                ))
                .unwrap()
                else {
                    panic!("create response")
                };
                let snapshot = mount.live.as_ref().unwrap().manifest().snapshot_id();
                assert_ne!(snapshot, before.snapshot);
                let committed = block_on(
                    WorkspaceStore::new(fs.engine_ref(), SessionId::new(0), session)
                        .recover_current(),
                )
                .unwrap()
                .unwrap();
                assert_eq!(committed.snapshot, snapshot);
                assert_eq!(committed.broker_root, before.broker_root);
                assert!(mount.topology_dirty());
                block_on(crate::persistence::commit_engine(
                    fs.engine(),
                    &mut durability,
                    SessionId::new(0),
                    None,
                ))
                .unwrap();
                let provider = fs
                    .into_engine()
                    .into_provider()
                    .restart_with_data_start(start)
                    .unwrap();
                let recovered = block_on(crate::persistence::PersistentFs::recover(
                    provider,
                    SessionId::new(7),
                ))
                .unwrap();
                let engine = recovered.engine();
                block_on(engine.attach_session(SessionId::new(6))).unwrap();
                let record = block_on(
                    WorkspaceStore::new(engine, SessionId::new(7), SessionId::new(6))
                        .recover_current(),
                )
                .unwrap()
                .unwrap();
                assert_eq!(record.snapshot, snapshot);
                let root = engine
                    .open_grant(SessionId::new(6), record.broker_root, Rights::READ)
                    .unwrap();
                let directory = engine
                    .open_at(SessionId::new(6), root.handle, &name("lib"), Rights::READ)
                    .unwrap();
                assert_eq!(
                    engine
                        .open_at(
                            SessionId::new(6),
                            root.handle,
                            &name("created"),
                            Rights::READ
                        )
                        .unwrap()
                        .stable_object,
                    created_stable
                );
                assert!(
                    engine
                        .open_at(
                            SessionId::new(6),
                            directory.handle,
                            &name("main.rs"),
                            Rights::READ
                        )
                        .is_ok()
                );
                assert_eq!(
                    engine.open_at(SessionId::new(6), root.handle, &name("src"), Rights::READ),
                    Err(FsError::NotFound)
                );
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn mounted_directory_rename_preserves_native_identity_and_verified_content() {
        with_workspace(true, |mount, fs, session, root, _, file| {
            let request = FsRequest::Rename {
                parent: root,
                old: name("src"),
                new: name("lib"),
            };
            assert_eq!(
                block_on(mount.write_namespace(fs, session, &request)),
                Ok(Some(FsResponse::Done))
            );
            let peer = mount.channel.as_ref().unwrap();
            assert!(peer.has_path(b"lib/main.rs"));
            assert!(!peer.has_path(b"src/main.rs"));
            assert_eq!(peer.transactions, 1);
            assert_eq!(
                fs.host_node_for_handle(session, file.handle)
                    .unwrap()
                    .unwrap()
                    .path
                    .as_bytes(),
                b"lib/main.rs"
            );
            block_on(mount.refresh(fs)).unwrap();
            assert_eq!(
                fs.engine_ref()
                    .stat(session, file.handle)
                    .unwrap()
                    .stable_object,
                file.stable_object
            );
            fs.dispatch(
                session,
                &FsRequest::AttachWindow { slot: 0, bytes: 16 },
                None,
            );
            let mut bytes = [0; 16];
            assert_eq!(
                block_on(mount.read(
                    fs,
                    session,
                    file.handle,
                    0,
                    0,
                    16,
                    Some(&mut DataWindow::new(&mut bytes))
                )),
                Ok(Some(FsResponse::Transferred { count: 16 }))
            );
            assert_eq!(bytes, core::array::from_fn(|index| index as u8));
        });
    }

    #[test]
    fn mounted_move_and_remove_commit_host_changes_before_reporting_done() {
        with_workspace(true, |mount, fs, session, root, src, file| {
            let request = FsRequest::Relocate {
                src_parent: src,
                name: name("main.rs"),
                dst_parent: root,
                new_name: name("output.rs"),
            };
            assert_eq!(
                block_on(mount.write_namespace(fs, session, &request)),
                Ok(Some(FsResponse::Done))
            );
            assert!(mount.channel.as_ref().unwrap().has_path(b"output.rs"));
            assert_eq!(
                fs.engine_ref().stat(session, file.handle),
                Err(FsError::AccessLost)
            );
            block_on(mount.refresh(fs)).unwrap();
            let moved = fs
                .engine_ref()
                .open_at(session, root, &name("output.rs"), Rights::HOME)
                .unwrap();
            assert_eq!(moved.stable_object, file.stable_object);
            assert_eq!(
                block_on(mount.write_namespace(
                    fs,
                    session,
                    &FsRequest::Unlink {
                        parent: root,
                        name: name("output.rs")
                    }
                )),
                Ok(Some(FsResponse::Done))
            );
            assert!(!mount.channel.as_ref().unwrap().has_path(b"output.rs"));
            block_on(mount.refresh(fs)).unwrap();
            assert_eq!(
                block_on(mount.write_namespace(
                    fs,
                    session,
                    &FsRequest::Unlink {
                        parent: root,
                        name: name("src")
                    }
                )),
                Ok(Some(FsResponse::Done))
            );
            assert!(!mount.channel.as_ref().unwrap().has_path(b"src"));
            block_on(mount.refresh(fs)).unwrap();
            assert_eq!(
                fs.engine_ref()
                    .open_at(session, root, &name("src"), Rights::READ),
                Err(FsError::NotFound)
            );
        });
    }

    #[test]
    fn mounted_namespace_disabled_and_nonempty_refusals_have_no_host_effects() {
        with_workspace(false, |mount, fs, session, root, _, file| {
            for request in [
                FsRequest::Rename {
                    parent: root,
                    old: name("src"),
                    new: name("lib"),
                },
                FsRequest::DetachEdge {
                    parent: root,
                    name: name("src"),
                },
            ] {
                assert_eq!(
                    block_on(mount.write_namespace(fs, session, &request)),
                    Err(FsError::AccessDenied)
                );
            }
            assert!(fs.engine_ref().stat(session, file.handle).is_ok());
            assert_eq!(
                mount.live.as_ref().unwrap().manifest().entries()[0]
                    .path
                    .as_bytes(),
                b"src"
            );
        });
        with_workspace(true, |mount, fs, session, root, _, file| {
            assert_eq!(
                block_on(mount.write_namespace(
                    fs,
                    session,
                    &FsRequest::Unlink {
                        parent: root,
                        name: name("src")
                    }
                )),
                Err(FsError::NotEmpty)
            );
            assert_eq!(mount.channel.as_ref().unwrap().transactions, 0);
            assert!(fs.engine_ref().stat(session, file.handle).is_ok());
            block_on(mount.refresh(fs)).unwrap();
            assert_eq!(
                block_on(mount.write_namespace(
                    fs,
                    session,
                    &FsRequest::DetachEdge {
                        parent: root,
                        name: name("src")
                    }
                )),
                Ok(Some(FsResponse::Done))
            );
            assert!(!mount.channel.as_ref().unwrap().has_path(b"src/main.rs"));
        });
    }

    #[test]
    fn mounted_move_cannot_cross_between_host_and_private_native_roots() {
        with_workspace(true, |mount, fs, session, root, src, file| {
            let private = fs.engine_ref().open_root(session, Rights::HOME).unwrap();
            let local = block_on(fs.engine_ref().create(
                session,
                private.handle,
                &name("local"),
                ObjectKind::File,
            ))
            .unwrap();
            for request in [
                FsRequest::Relocate {
                    src_parent: src,
                    name: name("main.rs"),
                    dst_parent: private.handle,
                    new_name: name("escape"),
                },
                FsRequest::Relocate {
                    src_parent: private.handle,
                    name: name("local"),
                    dst_parent: root,
                    new_name: name("injected"),
                },
            ] {
                assert_eq!(
                    block_on(mount.write_namespace(fs, session, &request)),
                    Err(FsError::AccessDenied)
                );
            }
            // The source integration also refuses links involving the host
            // tree. Preserve that boundary instead of inventing hard-link
            // identity semantics absent from the portable manifest contract.
            for (parent, child) in [
                (root, file.handle),
                (private.handle, file.handle),
                (root, local.handle),
            ] {
                let request = FsRequest::Link {
                    parent,
                    child,
                    name: name("alias"),
                };
                assert_eq!(
                    block_on(mount.before_request(fs, session, &request)),
                    Ok(None)
                );
                assert_eq!(
                    fs.dispatch(session, &request, None),
                    FsResponse::Refused(FsError::AccessDenied)
                );
            }
            assert!(fs.engine_ref().stat(session, file.handle).is_ok());
            assert_eq!(mount.live.take().unwrap().into_channel().transactions, 0);
        });
    }

    #[test]
    fn cancelled_file_resize_also_reconciles_before_reusing_cached_bytes() {
        with_workspace(true, |mount, fs, session, _, _, file| {
            let mut peer = mount.live.take().unwrap().into_channel();
            peer.pause_after_write = true;
            mount.channel = Some(peer);
            block_on(mount.refresh(fs)).unwrap();
            let request = FsRequest::Resize {
                handle: file.handle,
                length: 0,
                expected: file.revision,
            };
            {
                let future = mount.write_file(fs, session, &request, None);
                let mut future = core::pin::pin!(future);
                let mut cx = core::task::Context::from_waker(core::task::Waker::noop());
                assert!(future.as_mut().poll(&mut cx).is_pending());
            }
            assert!(mount.reconcile_required);
            assert_eq!(
                fs.engine_ref().stat(session, file.handle).unwrap().revision,
                file.revision
            );
            assert_eq!(
                block_on(mount.read(fs, session, file.handle, 0, 0, 0, None)),
                Err(FsError::Busy)
            );
            let other = SessionId::new(2);
            assert!(fs.ensure_session(other));
            let root = fs.engine_ref().open_root(other, Rights::HOME).unwrap();
            let native = block_on(fs.engine_ref().create(
                other,
                root.handle,
                &name("ordinary"),
                ObjectKind::File,
            ))
            .unwrap();
            assert_eq!(
                block_on(mount.read(fs, other, native.handle, 0, 0, 0, None)),
                Ok(None)
            );
            block_on(mount.refresh(fs)).unwrap();
            assert_eq!(
                fs.engine_ref().stat(session, file.handle),
                Err(FsError::AccessLost)
            );
            let file = mount
                .live
                .as_ref()
                .unwrap()
                .manifest()
                .entries()
                .iter()
                .find(|entry| entry.kind == EntryKind::File)
                .unwrap();
            assert_eq!(file.content_len, 0);
            assert_eq!(file.digest, *blake3::hash(b"").as_bytes());
        });
    }

    #[test]
    fn mounted_namespace_conflict_preserves_local_state_and_allows_retry() {
        with_workspace(true, |mount, fs, session, root, _, file| {
            let mut peer = mount.live.take().unwrap().into_channel();
            peer.write_status = kore_workspace_sync::WorkspaceStatus::Conflict;
            mount.channel = Some(peer);
            block_on(mount.refresh(fs)).unwrap();
            let request = FsRequest::Rename {
                parent: root,
                old: name("src"),
                new: name("lib"),
            };
            assert_eq!(
                block_on(mount.write_namespace(fs, session, &request)),
                Err(FsError::Conflict)
            );
            assert!(mount.channel.as_ref().unwrap().has_path(b"src/main.rs"));
            assert!(fs.engine_ref().stat(session, file.handle).is_ok());
            mount.channel.as_mut().unwrap().write_status = kore_workspace_sync::WorkspaceStatus::Ok;
            block_on(mount.refresh(fs)).unwrap();
            assert_eq!(
                block_on(mount.write_namespace(fs, session, &request)),
                Ok(Some(FsResponse::Done))
            );
            assert_eq!(mount.channel.as_ref().unwrap().transactions, 2);
        });
    }

    #[test]
    fn lost_namespace_receipt_reimports_host_truth_and_retires_old_authority() {
        with_workspace(true, |mount, fs, session, root, _, file| {
            let mut peer = mount.live.take().unwrap().into_channel();
            peer.lose_next_receipt = true;
            mount.channel = Some(peer);
            block_on(mount.refresh(fs)).unwrap();
            let request = FsRequest::Rename {
                parent: root,
                old: name("src"),
                new: name("lib"),
            };
            assert_eq!(
                block_on(mount.write_namespace(fs, session, &request)),
                Err(FsError::ProviderFailure(crate::ProviderError::Lost))
            );
            assert!(mount.channel.as_ref().unwrap().has_path(b"lib/main.rs"));
            assert!(
                fs.engine_ref()
                    .open_at(session, root, &name("src"), Rights::READ)
                    .is_ok()
            );
            block_on(mount.refresh(fs)).unwrap();
            assert_eq!(
                fs.engine_ref().stat(session, file.handle),
                Err(FsError::AccessLost)
            );
            let current = block_on(
                WorkspaceStore::new(fs.engine_ref(), SessionId::new(0), session).recover_current(),
            )
            .unwrap()
            .unwrap();
            let root = fs
                .engine_ref()
                .open_grant(session, current.broker_root, Rights::READ)
                .unwrap();
            assert!(
                fs.engine_ref()
                    .open_at(session, root.handle, &name("lib"), Rights::READ)
                    .is_ok()
            );
        });
    }

    #[test]
    fn cancelled_namespace_write_forces_reconciliation_before_the_next_read() {
        with_workspace(true, |mount, fs, session, root, _, file| {
            let mut peer = mount.live.take().unwrap().into_channel();
            peer.pause_after_write = true;
            mount.channel = Some(peer);
            block_on(mount.refresh(fs)).unwrap();
            let request = FsRequest::Rename {
                parent: root,
                old: name("src"),
                new: name("lib"),
            };
            {
                let future = mount.write_namespace(fs, session, &request);
                let mut future = core::pin::pin!(future);
                let mut cx = core::task::Context::from_waker(core::task::Waker::noop());
                assert!(future.as_mut().poll(&mut cx).is_pending());
            }
            assert!(mount.reconcile_required);
            assert!(fs.engine_ref().stat(session, file.handle).is_ok());
            assert_eq!(
                block_on(mount.read(fs, session, file.handle, 0, 0, 0, None)),
                Err(FsError::Busy)
            );
            block_on(mount.refresh(fs)).unwrap();
            assert!(!mount.reconcile_required);
            assert_eq!(
                fs.engine_ref().stat(session, file.handle),
                Err(FsError::AccessLost)
            );
            assert!(
                mount
                    .live
                    .as_ref()
                    .unwrap()
                    .manifest()
                    .entries()
                    .iter()
                    .any(|entry| entry.path.as_bytes() == b"lib/main.rs")
            );
        });
    }
}
