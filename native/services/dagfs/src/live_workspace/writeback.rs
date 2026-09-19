//! File writeback composes native authority and staged provider publication
//! with the existing preconditioned, journaled host transaction protocol.

use super::*;
use crate::{Revision, Rights};
use kore_workspace_sync::{Change, ChangeKind, bounds::MAX_FILE_BYTES};

impl<C: WorkspaceChannel> WorkspaceMount<C> {
    pub(crate) async fn write_file<P: StorageProvider>(
        &mut self,
        fs: &mut FsService<P>,
        session: SessionId,
        request: &FsRequest,
        window: Option<&mut DataWindow<'_>>,
    ) -> FsResult<Option<FsResponse>> {
        let (handle, expected) = match *request {
            FsRequest::WriteAt {
                handle, expected, ..
            }
            | FsRequest::Append {
                handle, expected, ..
            }
            | FsRequest::Resize {
                handle, expected, ..
            } => (handle, expected),
            _ => return Ok(None),
        };
        let Some(mut node) = fs.host_node_for_handle(session, handle)? else {
            return Ok(None);
        };
        if node.revoked {
            return Err(FsError::AccessLost);
        }
        if node.kind != ObjectKind::File {
            return Err(FsError::IsDirectory);
        }
        let stat = fs
            .engine_ref()
            .stat_with_rights(session, handle, Rights::WRITE)?;
        if expected != crate::ANY_REVISION && expected != stat.revision {
            return Err(FsError::Conflict);
        }
        if fs.is_deferring(session) {
            return Err(FsError::Busy);
        }
        let live = self
            .live
            .as_mut()
            .ok_or(FsError::ProviderFailure(crate::ProviderError::Lost))?;
        if !live.writeback_enabled() {
            return Err(FsError::AccessDenied);
        }
        self.reconcile_required = true;
        let mut fragment = heapless::Vec::<u8, { crate::bounds::MAX_TRANSFER }>::new();
        let (offset, len, append) = match *request {
            FsRequest::WriteAt {
                length,
                window_offset,
                ..
            }
            | FsRequest::Append {
                length,
                window_offset,
                ..
            } => {
                if fs.attachment(session).is_none() {
                    return Err(FsError::NoWindow);
                }
                if length > fs.transfer_limit() {
                    return Err(FsError::Range);
                }
                fragment
                    .extend_from_slice(
                        window
                            .ok_or(FsError::NoWindow)?
                            .source(window_offset, length)?,
                    )
                    .map_err(|_| FsError::Range)?;
                let append = matches!(request, FsRequest::Append { .. });
                let offset = match *request {
                    FsRequest::WriteAt { offset, .. } => offset,
                    _ => node.len,
                };
                let end = offset
                    .checked_add(u64::from(length))
                    .ok_or(FsError::Range)?;
                (offset, node.len.max(end), append)
            }
            FsRequest::Resize { length, .. } => (0, length, false),
            _ => unreachable!(),
        };
        if len > MAX_FILE_BYTES || len > fs.engine_ref().provider().limits().max_object_bytes {
            return Err(FsError::OutOfStorage);
        }
        let mut content = alloc::vec::Vec::new_in(crate::service_alloc());
        content
            .try_reserve_exact(len as usize)
            .map_err(|_| FsError::OutOfStorage)?;
        content.resize(len as usize, 0);
        let result = async {
            let source = stable_live_object(node.stable);
            let keep = node.len.min(len) as usize;
            let mut cursor = 0;
            while cursor < keep {
                let end = keep.min(cursor + kore_workspace_sync::bounds::RANGE_BYTES);
                let count = live
                    .read(source, cursor as u32, &mut content[cursor..end])
                    .await
                    .map_err(workspace_error)?;
                if count == 0 || count > end - cursor {
                    return Err(FsError::ProviderFailure(crate::ProviderError::Integrity));
                }
                cursor += count;
            }
            if !fragment.is_empty() {
                content[offset as usize..offset as usize + fragment.len()]
                    .copy_from_slice(&fragment);
            }
            let digest = *blake3::hash(&content).as_bytes();
            if len == node.len && digest == node.digest {
                let current = fs
                    .engine_ref()
                    .stat_with_rights(session, handle, Rights::WRITE)?;
                if current.revision != stat.revision {
                    return Err(FsError::Conflict);
                }
                return Ok((stat.revision, digest, false));
            }
            let change = Change {
                kind: ChangeKind::ModifyFile,
                path: node.path.clone(),
                target: None,
                expected: node.digest,
                replacement: digest,
                replacement_len: len as u32,
                executable: node.executable,
            };
            // Freeze the exact native revision observed before source reads,
            // even for ANY_REVISION, so a concurrent change cannot be lost.
            let revision = fs
                .engine_ref()
                .replace_host_file(session, handle, stat.revision, &content, async || {
                    live.writeback(change, Some(&content))
                        .await
                        .map(|_| ())
                        .map_err(workspace_error)
                })
                .await?;
            Ok((revision, digest, true))
        }
        .await;
        match result {
            Ok((revision, digest, changed)) => {
                if changed {
                    node.len = len;
                    node.digest = digest;
                    node.materialized = true;
                    fs.update_host_node(node)?;
                    self.topology_dirty = true;
                    // The acknowledged snapshot must be attached before any
                    // subsequent cache access. Preserve the channel on retry.
                    if let Some(live) = self.live.take() {
                        self.channel = Some(live.into_channel());
                    }
                } else {
                    self.reconcile_required = false;
                }
                Ok(Some(file_reply(append, offset, revision)))
            }
            Err(error) => {
                // A conflict or uncertain terminal result must be reconciled
                // with a fresh host attachment, never the pre-write cache.
                if let Some(live) = self.live.take() {
                    self.channel = Some(live.into_channel());
                }
                Err(error)
            }
        }
    }
}

fn file_reply(append: bool, offset: u64, revision: Revision) -> FsResponse {
    if append {
        FsResponse::Appended { offset, revision }
    } else {
        FsResponse::Committed { revision }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Name, block_on, host_workspace::tests::FakeChannel};
    use kore_workspace_sync::WorkspaceStatus;

    fn exercise(enabled: bool, status: WorkspaceStatus, noop: bool, lose_receipt: bool) {
        std::thread::Builder::new()
            // Combined unoptimized format/import/write/recover futures own the
            // block device by value. This is not the guest stack requirement.
            .stack_size(128 * 1024 * 1024)
            .spawn(move || {
                use kore_driver::block::{BlockChannel, CompletionMode, memory::MemoryBlockDevice};
                let device = MemoryBlockDevice::<327680, 8192>::new(512, 16).unwrap();
                let channel = BlockChannel::<_, 3, 8>::new(device, CompletionMode::Poll).unwrap();
                let start = crate::persistence::bootstrap_data_start(channel.geometry().block_size)
                    .unwrap();
                let provider = crate::BlockProvider::new_with_data_start(channel, start).unwrap();
                let mut peer = FakeChannel::new().applying_writes();
                if enabled {
                    peer = peer.with_writeback();
                }
                peer.write_status = status;
                peer.lose_next_receipt = lose_receipt;
                let mut mount = WorkspaceMount::new(peer);
                let volume = block_on(crate::persistence::PersistentFs::format(
                    provider,
                    SessionId::new(0),
                ))
                .unwrap();
                let (engine, mut durability) = volume.into_parts();
                let mut fs = FsService::new(engine);
                block_on(mount.refresh(&mut fs)).unwrap();
                let session = SessionId::new(crate::service::contract::ADMIN_VIA);
                let commit = block_on(
                    WorkspaceStore::new(fs.engine_ref(), SessionId::new(0), session)
                        .recover_current(),
                )
                .unwrap()
                .unwrap();
                let root = fs
                    .engine_ref()
                    .open_grant(session, commit.broker_root, Rights::HOME)
                    .unwrap();
                let dir = fs
                    .engine_ref()
                    .open_at(
                        session,
                        root.handle,
                        &Name::new("src").unwrap(),
                        Rights::HOME,
                    )
                    .unwrap();
                let file = fs
                    .engine_ref()
                    .open_at(
                        session,
                        dir.handle,
                        &Name::new("main.rs").unwrap(),
                        Rights::HOME,
                    )
                    .unwrap();
                fs.dispatch(
                    session,
                    &FsRequest::AttachWindow { slot: 0, bytes: 3 },
                    None,
                );
                let mut bytes = if noop { [0, 1, 2] } else { *b"NEW" };
                let request = FsRequest::WriteAt {
                    handle: file.handle,
                    expected: file.revision,
                    offset: 0,
                    window_offset: 0,
                    length: 3,
                };
                let result = block_on(mount.write_file(
                    &mut fs,
                    session,
                    &request,
                    Some(&mut DataWindow::new(&mut bytes)),
                ));
                let stat = fs.engine_ref().stat(session, file.handle).unwrap();
                if !enabled {
                    assert_eq!(result, Err(FsError::AccessDenied));
                    assert_eq!((stat.len, stat.revision), (0, file.revision));
                } else if lose_receipt {
                    assert_eq!(
                        result,
                        Err(FsError::ProviderFailure(crate::ProviderError::Lost))
                    );
                    assert_eq!((stat.len, stat.revision), (0, file.revision));
                    assert_eq!(&mount.channel.as_ref().unwrap().content()[..3], b"NEW");
                    block_on(mount.refresh(&mut fs)).unwrap();
                    assert_eq!(
                        fs.dispatch(
                            session,
                            &FsRequest::Stat {
                                handle: file.handle
                            },
                            None
                        ),
                        FsResponse::Refused(FsError::AccessLost)
                    );
                    let current = block_on(
                        WorkspaceStore::new(fs.engine_ref(), SessionId::new(0), session)
                            .recover_current(),
                    )
                    .unwrap()
                    .unwrap();
                    let root = fs
                        .engine_ref()
                        .open_grant(session, current.broker_root, Rights::HOME)
                        .unwrap();
                    let dir = fs
                        .engine_ref()
                        .open_at(
                            session,
                            root.handle,
                            &Name::new("src").unwrap(),
                            Rights::HOME,
                        )
                        .unwrap();
                    let replacement = fs
                        .engine_ref()
                        .open_at(
                            session,
                            dir.handle,
                            &Name::new("main.rs").unwrap(),
                            Rights::HOME,
                        )
                        .unwrap();
                    block_on(mount.materialize(
                        &mut fs,
                        session,
                        replacement.handle,
                        replacement.revision,
                        Rights::READ,
                    ))
                    .unwrap();
                } else if status == WorkspaceStatus::Conflict {
                    assert_eq!(result, Err(FsError::Conflict));
                    assert_eq!((stat.len, stat.revision), (0, file.revision));
                    assert_ne!(&mount.channel.as_ref().unwrap().content()[..3], b"NEW");
                    mount.channel.as_mut().unwrap().write_status = WorkspaceStatus::Ok;
                    block_on(mount.refresh(&mut fs)).unwrap();
                    assert_eq!(
                        block_on(mount.write_file(
                            &mut fs,
                            session,
                            &request,
                            Some(&mut DataWindow::new(&mut bytes))
                        )),
                        Ok(Some(FsResponse::Committed {
                            revision: file.revision + 1
                        }))
                    );
                } else if noop {
                    assert_eq!(
                        result,
                        Ok(Some(FsResponse::Committed {
                            revision: file.revision
                        }))
                    );
                    assert_eq!((stat.len, stat.revision), (0, file.revision));
                } else {
                    assert_eq!(
                        result,
                        Ok(Some(FsResponse::Committed {
                            revision: file.revision + 1
                        }))
                    );
                    assert_eq!(stat.len, 4127);
                    let mut selected = [0; 3];
                    block_on(fs.engine_ref().read(session, file.handle, 0, &mut selected)).unwrap();
                    assert_eq!(&selected, b"NEW");
                }
                if enabled && !noop && !lose_receipt {
                    let peer = mount.channel.as_ref().unwrap();
                    assert_eq!(&peer.content()[..3], b"NEW");
                    let snapshot = peer.snapshot();
                    assert_ne!(snapshot, commit.snapshot);
                    block_on(mount.refresh(&mut fs)).unwrap();
                    assert_eq!(
                        fs.engine_ref()
                            .stat(session, file.handle)
                            .unwrap()
                            .stable_object,
                        file.stable_object
                    );
                    let rebound = block_on(
                        WorkspaceStore::new(fs.engine_ref(), SessionId::new(0), session)
                            .recover_current(),
                    )
                    .unwrap()
                    .unwrap();
                    assert_eq!(rebound.snapshot, snapshot);
                    assert_eq!(rebound.broker_root, commit.broker_root);
                }
                let peer = match mount.live.take() {
                    Some(live) => live.into_channel(),
                    None => mount.channel.take().unwrap(),
                };
                if enabled && !noop {
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
                    let recovered_importer = SessionId::new(7);
                    let recovered_broker = SessionId::new(6);
                    let volume = block_on(crate::persistence::PersistentFs::recover(
                        provider,
                        recovered_importer,
                    ))
                    .unwrap();
                    let engine = volume.engine();
                    block_on(engine.attach_session(recovered_broker)).unwrap();
                    let recovered = block_on(
                        WorkspaceStore::new(engine, recovered_importer, recovered_broker)
                            .recover_current(),
                    )
                    .unwrap()
                    .unwrap();
                    assert_eq!(recovered.snapshot, peer.snapshot());
                    assert!(engine.stat(session, file.handle).is_err());
                    let root = engine
                        .open_grant(recovered_broker, recovered.broker_root, Rights::READ_ONLY)
                        .unwrap();
                    let dir = engine
                        .open_at(
                            recovered_broker,
                            root.handle,
                            &Name::new("src").unwrap(),
                            Rights::READ_ONLY,
                        )
                        .unwrap();
                    let file = engine
                        .open_at(
                            recovered_broker,
                            dir.handle,
                            &Name::new("main.rs").unwrap(),
                            Rights::READ_ONLY,
                        )
                        .unwrap();
                    let mut recovered_bytes = std::vec![0; peer.content().len()];
                    let count = block_on(engine.read(
                        recovered_broker,
                        file.handle,
                        0,
                        &mut recovered_bytes,
                    ))
                    .unwrap();
                    assert_eq!(count, recovered_bytes.len());
                    assert_eq!(recovered_bytes, peer.content());
                }
                assert_eq!(
                    peer.transactions,
                    usize::from(enabled && !noop)
                        + usize::from(status == WorkspaceStatus::Conflict)
                );
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn writeback_file_commits_one_native_revision_after_host_receipt() {
        exercise(true, WorkspaceStatus::Ok, false, false);
    }
    #[test]
    fn writeback_disabled_cannot_mutate_the_native_file() {
        exercise(false, WorkspaceStatus::Ok, false, false);
    }
    #[test]
    fn writeback_conflict_preserves_native_file_and_revision() {
        exercise(true, WorkspaceStatus::Conflict, false, false);
    }
    #[test]
    fn writeback_noop_sends_no_transaction_and_keeps_revision() {
        exercise(true, WorkspaceStatus::Ok, true, false);
    }
    #[test]
    fn writeback_lost_receipt_resynchronizes_to_applied_host_bytes() {
        exercise(true, WorkspaceStatus::Ok, false, true);
    }

    #[test]
    fn writeback_append_resize_and_sparse_write_preserve_write_only_authority() {
        std::thread::Builder::new()
            .stack_size(128 * 1024 * 1024)
            .spawn(|| {
                use kore_driver::block::{BlockChannel, CompletionMode, memory::MemoryBlockDevice};
                let device = MemoryBlockDevice::<327680, 8192>::new(512, 16).unwrap();
                let channel = BlockChannel::<_, 3, 8>::new(device, CompletionMode::Poll).unwrap();
                let provider = crate::BlockProvider::new(channel).unwrap();
                let mut fs = FsService::new(crate::Engine::new(provider));
                let peer = FakeChannel::new().with_writeback().applying_writes();
                let mut expected_bytes = peer.content().to_vec();
                let mut mount = WorkspaceMount::new(peer);
                block_on(mount.refresh(&mut fs)).unwrap();
                let session = SessionId::new(crate::service::contract::ADMIN_VIA);
                let commit = block_on(
                    WorkspaceStore::new(fs.engine_ref(), SessionId::new(0), session)
                        .recover_current(),
                )
                .unwrap()
                .unwrap();
                let root = fs
                    .engine_ref()
                    .open_grant(session, commit.broker_root, Rights::HOME)
                    .unwrap();
                let dir = fs
                    .engine_ref()
                    .open_at(
                        session,
                        root.handle,
                        &Name::new("src").unwrap(),
                        Rights::HOME,
                    )
                    .unwrap();
                let writer = fs
                    .engine_ref()
                    .open_at(
                        session,
                        dir.handle,
                        &Name::new("main.rs").unwrap(),
                        Rights::WRITE,
                    )
                    .unwrap();
                let reader = fs
                    .engine_ref()
                    .open_at(
                        session,
                        dir.handle,
                        &Name::new("main.rs").unwrap(),
                        Rights::READ,
                    )
                    .unwrap();
                let mut revision = writer.revision;
                fs.dispatch(
                    session,
                    &FsRequest::AttachWindow { slot: 0, bytes: 3 },
                    None,
                );
                let mut bytes = *b"END";

                // No range, window, authority or revision refusal may reach
                // the host transaction or publish native bytes.
                for (request, refusal) in [
                    (
                        FsRequest::Append {
                            handle: reader.handle,
                            expected: revision,
                            window_offset: 0,
                            length: 3,
                        },
                        FsError::AccessDenied,
                    ),
                    (
                        FsRequest::Append {
                            handle: writer.handle,
                            expected: revision + 1,
                            window_offset: 0,
                            length: 3,
                        },
                        FsError::Conflict,
                    ),
                    (
                        FsRequest::Append {
                            handle: writer.handle,
                            expected: revision,
                            window_offset: 1,
                            length: 3,
                        },
                        FsError::Range,
                    ),
                    (
                        FsRequest::WriteAt {
                            handle: writer.handle,
                            expected: revision,
                            offset: u64::MAX,
                            window_offset: 0,
                            length: 3,
                        },
                        FsError::Range,
                    ),
                    (
                        FsRequest::Resize {
                            handle: writer.handle,
                            expected: revision,
                            length: MAX_FILE_BYTES + 1,
                        },
                        FsError::OutOfStorage,
                    ),
                ] {
                    assert_eq!(
                        block_on(mount.write_file(
                            &mut fs,
                            session,
                            &request,
                            Some(&mut DataWindow::new(&mut bytes))
                        )),
                        Err(refusal)
                    );
                    assert_eq!(
                        fs.engine_ref()
                            .stat(session, reader.handle)
                            .unwrap()
                            .revision,
                        revision
                    );
                }
                let append_offset = expected_bytes.len() as u64;
                assert_eq!(
                    block_on(mount.write_file(
                        &mut fs,
                        session,
                        &FsRequest::Append {
                            handle: writer.handle,
                            expected: revision,
                            window_offset: 0,
                            length: 3,
                        },
                        Some(&mut DataWindow::new(&mut bytes))
                    )),
                    Ok(Some(FsResponse::Appended {
                        offset: append_offset,
                        revision: revision + 1
                    }))
                );
                revision += 1;
                expected_bytes.extend_from_slice(b"END");
                assert_eq!(mount.channel.as_ref().unwrap().content(), expected_bytes);
                assert_eq!(mount.channel.as_ref().unwrap().transactions, 1);
                block_on(mount.refresh(&mut fs)).unwrap();

                // Grow, shrink and erase through the same native write-only
                // route. Growth must expose zeros, never the discarded tail.
                for length in [4135usize, 2, 7, 0] {
                    assert_eq!(
                        block_on(mount.write_file(
                            &mut fs,
                            session,
                            &FsRequest::Resize {
                                handle: writer.handle,
                                expected: revision,
                                length: length as u64,
                            },
                            None
                        )),
                        Ok(Some(FsResponse::Committed {
                            revision: revision + 1
                        }))
                    );
                    revision += 1;
                    expected_bytes.resize(length, 0);
                    assert_eq!(mount.channel.as_ref().unwrap().content(), expected_bytes);
                    block_on(mount.refresh(&mut fs)).unwrap();
                    let mut selected = std::vec![0xa5; length + 1];
                    let count =
                        block_on(
                            fs.engine_ref()
                                .read(session, reader.handle, 0, &mut selected),
                        )
                        .unwrap();
                    assert_eq!(count, length);
                    assert_eq!(&selected[..count], expected_bytes);
                    assert_eq!(selected[count], 0xa5);
                }

                assert_eq!(
                    block_on(mount.write_file(
                        &mut fs,
                        session,
                        &FsRequest::WriteAt {
                            handle: writer.handle,
                            expected: revision,
                            offset: 4,
                            window_offset: 0,
                            length: 3,
                        },
                        Some(&mut DataWindow::new(&mut bytes))
                    )),
                    Ok(Some(FsResponse::Committed {
                        revision: revision + 1
                    }))
                );
                assert_eq!(mount.channel.as_ref().unwrap().content(), b"\0\0\0\0END");
                assert_eq!(mount.channel.as_ref().unwrap().transactions, 6);
                block_on(mount.refresh(&mut fs)).unwrap();
                assert_eq!(
                    fs.engine_ref().stat(session, writer.handle),
                    Err(FsError::AccessDenied)
                );
                let mut sentinel = [0xa5; 3];
                // Materialized reads use the ordinary service path, whose
                // authority check must still refuse the write-only handle.
                assert_eq!(
                    block_on(mount.read(
                        &fs,
                        session,
                        writer.handle,
                        0,
                        0,
                        3,
                        Some(&mut DataWindow::new(&mut sentinel))
                    )),
                    Ok(None)
                );
                assert_eq!(
                    fs.dispatch(
                        session,
                        &FsRequest::Read {
                            handle: writer.handle,
                            offset: 0,
                            window_offset: 0,
                            length: 3
                        },
                        Some(&mut DataWindow::new(&mut sentinel))
                    ),
                    FsResponse::Refused(FsError::AccessDenied)
                );
                assert_eq!(sentinel, [0xa5; 3]);
                assert_eq!(
                    fs.engine_ref()
                        .stat(session, reader.handle)
                        .unwrap()
                        .stable_object,
                    reader.stable_object
                );
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
