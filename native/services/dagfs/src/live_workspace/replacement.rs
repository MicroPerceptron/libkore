//! Atomic replacement is staged by the existing session owner; only its
//! terminal publication is composed with the preconditioned host transaction.
use super::*;
use crate::Rights;
use kore_workspace_sync::{Change, ChangeKind, bounds::MAX_FILE_BYTES};

impl<C: WorkspaceChannel> WorkspaceMount<C> {
    pub(crate) async fn replace_file<P: StorageProvider>(
        &mut self,
        fs: &mut FsService<P>,
        session: SessionId,
        request: &FsRequest,
        window: Option<&mut DataWindow<'_>>,
    ) -> FsResult<Option<FsResponse>> {
        let handle = match *request {
            FsRequest::BeginReplace { handle, .. }
            | FsRequest::WriteReplacement { handle, .. }
            | FsRequest::CommitReplace { handle }
            | FsRequest::AbortReplace { handle } => handle,
            _ => return Ok(None),
        };
        let Some(mut node) = fs.host_node_for_handle(session, handle)? else {
            return Ok(None);
        };
        // Cleanup needs no live host or standing write grant.
        if matches!(request, FsRequest::AbortReplace { .. }) {
            return fs
                .replacement_request(session, request, window)
                .await
                .map(Some);
        }
        if node.revoked {
            return Err(FsError::AccessLost);
        }
        fs.engine_ref()
            .stat_with_rights(session, handle, Rights::WRITE)?;
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
        if let FsRequest::BeginReplace { length, .. } = *request {
            if length > MAX_FILE_BYTES {
                return Err(FsError::OutOfStorage);
            }
        }
        if !matches!(request, FsRequest::CommitReplace { .. }) {
            return fs
                .replacement_request(session, request, window)
                .await
                .map(Some);
        }
        self.reconcile_required = true;
        let result = fs
            .commit_host_replacement(session, handle, async |content: &[u8]| {
                let digest = *blake3::hash(content).as_bytes();
                live.writeback(
                    Change {
                        kind: ChangeKind::ModifyFile,
                        path: node.path.clone(),
                        target: None,
                        expected: node.digest,
                        replacement: digest,
                        replacement_len: content.len() as u32,
                        executable: node.executable,
                    },
                    Some(content),
                )
                .await
                .map_err(workspace_error)?;
                node.len = content.len() as u64;
                node.digest = digest;
                node.materialized = true;
                Ok(())
            })
            .await;
        if let Some(live) = self.live.take() {
            self.channel = Some(live.into_channel());
        }
        let revision = result?;
        fs.update_host_node(node)?;
        self.topology_dirty = true;
        Ok(Some(FsResponse::Committed { revision }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{block_on, live_workspace::namespace::tests::with_workspace};
    use kore_workspace_sync::WorkspaceStatus;

    fn stage(
        mount: &mut WorkspaceMount<crate::host_workspace::tests::FakeChannel>,
        fs: &mut crate::service::RamService,
        session: SessionId,
        handle: Handle,
        bytes: &[u8],
    ) {
        assert_eq!(
            block_on(mount.before_request(
                fs,
                session,
                &FsRequest::BeginReplace {
                    handle,
                    length: bytes.len() as u64,
                    expected: crate::ANY_REVISION,
                }
            )),
            Ok(Some(FsResponse::Done))
        );
        fs.dispatch(
            session,
            &FsRequest::AttachWindow {
                slot: 0,
                bytes: bytes.len() as u32,
            },
            None,
        );
        let mut input = bytes.to_vec();
        assert_eq!(
            block_on(mount.replace_file(
                fs,
                session,
                &FsRequest::WriteReplacement {
                    handle,
                    offset: 0,
                    window_offset: 0,
                    length: input.len() as u32,
                },
                Some(&mut DataWindow::new(&mut input))
            )),
            Ok(Some(FsResponse::Done))
        );
    }

    #[test]
    fn mounted_atomic_replacement_publishes_host_bytes_and_snapshot() {
        with_workspace(true, |mount, fs, session, _, src, file| {
            let writer = fs
                .engine_ref()
                .open_at(
                    session,
                    src,
                    &crate::Name::new("main.rs").unwrap(),
                    Rights::WRITE,
                )
                .unwrap();
            let before = fs.engine_ref().stat(session, file.handle).unwrap().revision;
            stage(mount, fs, session, writer.handle, b"hello");
            assert_eq!(mount.live.as_mut().unwrap().channel_mut().transactions, 0);
            assert_eq!(
                fs.engine_ref().stat(session, file.handle).unwrap().revision,
                before
            );
            assert!(matches!(
                block_on(mount.before_request(
                    fs,
                    session,
                    &FsRequest::CommitReplace {
                        handle: writer.handle
                    }
                )),
                Ok(Some(FsResponse::Committed { .. }))
            ));
            let peer = &mount.live.as_mut().unwrap().channel_mut();
            assert_eq!(peer.content(), b"hello");
            assert_eq!(peer.transactions, 1);
            let recorded = block_on(
                WorkspaceStore::new(fs.engine_ref(), SessionId::new(0), session).recover_current(),
            )
            .unwrap()
            .unwrap();
            assert_eq!(recorded.snapshot, peer.snapshot());
            let mut bytes = [0; 5];
            assert_eq!(
                block_on(fs.engine_ref().read(session, file.handle, 0, &mut bytes)),
                Ok(5)
            );
            assert_eq!(&bytes, b"hello");
            assert_eq!(
                fs.engine_ref().stat(session, writer.handle),
                Err(FsError::AccessDenied)
            );
            assert_eq!(
                block_on(mount.before_request(
                    fs,
                    session,
                    &FsRequest::CommitReplace {
                        handle: writer.handle
                    }
                )),
                Err(FsError::InvalidHandle)
            );
        });
    }

    #[test]
    fn mounted_atomic_replacement_host_refusal_preserves_bytes_and_abort_custody() {
        with_workspace(true, |mount, fs, session, _, _, file| {
            stage(mount, fs, session, file.handle, b"old");
            block_on(mount.before_request(
                fs,
                session,
                &FsRequest::CommitReplace {
                    handle: file.handle,
                },
            ))
            .unwrap();
            let old = mount
                .live
                .as_mut()
                .unwrap()
                .channel_mut()
                .content()
                .to_vec();
            let before = fs.engine_ref().stat(session, file.handle).unwrap().revision;
            stage(mount, fs, session, file.handle, b"hello");
            mount.live.as_mut().unwrap().channel_mut().write_status = WorkspaceStatus::Conflict;
            assert!(
                block_on(mount.before_request(
                    fs,
                    session,
                    &FsRequest::CommitReplace {
                        handle: file.handle
                    }
                ))
                .is_err()
            );
            assert_eq!(
                fs.engine_ref().stat(session, file.handle).unwrap().revision,
                before
            );
            let mut bytes = std::vec![0; old.len()];
            block_on(fs.engine_ref().read(session, file.handle, 0, &mut bytes)).unwrap();
            assert_eq!(bytes, old);
            assert_eq!(mount.channel.as_ref().unwrap().content(), old);
            assert_eq!(
                block_on(mount.before_request(
                    fs,
                    session,
                    &FsRequest::AbortReplace {
                        handle: file.handle
                    }
                )),
                Ok(Some(FsResponse::Done))
            );
            stage(mount, fs, session, file.handle, b"retry");
            assert_eq!(
                fs.dispatch(
                    session,
                    &FsRequest::Close {
                        handle: file.handle
                    },
                    None
                ),
                FsResponse::Done
            );
        });
    }

    #[test]
    fn mounted_atomic_replacement_denies_disabled_and_unmounted_writeback() {
        with_workspace(false, |mount, fs, session, _, _, file| {
            let request = FsRequest::BeginReplace {
                handle: file.handle,
                length: 5,
                expected: crate::ANY_REVISION,
            };
            assert_eq!(
                block_on(mount.before_request(fs, session, &request)),
                Err(FsError::AccessDenied)
            );
            assert_eq!(
                fs.dispatch(session, &request, None),
                FsResponse::Refused(FsError::AccessDenied)
            );
            assert_eq!(mount.live.as_mut().unwrap().channel_mut().transactions, 0);
        });
    }

    #[test]
    fn mounted_atomic_replacement_incomplete_stage_never_publishes() {
        with_workspace(true, |mount, fs, session, _, _, file| {
            assert_eq!(
                block_on(mount.before_request(
                    fs,
                    session,
                    &FsRequest::BeginReplace {
                        handle: file.handle,
                        length: 5,
                        expected: crate::ANY_REVISION
                    }
                )),
                Ok(Some(FsResponse::Done))
            );
            assert_eq!(
                block_on(mount.before_request(
                    fs,
                    session,
                    &FsRequest::CommitReplace {
                        handle: file.handle
                    }
                )),
                Err(FsError::Range)
            );
            assert_eq!(mount.channel.as_ref().unwrap().transactions, 0);
            assert_eq!(
                fs.dispatch(
                    session,
                    &FsRequest::Close {
                        handle: file.handle
                    },
                    None
                ),
                FsResponse::Done
            );
        });
    }
    #[test]
    fn mounted_atomic_replacement_lost_receipt_reconciles_and_close_reclaims_stage() {
        with_workspace(true, |mount, fs, session, _, _, file| {
            stage(mount, fs, session, file.handle, b"hello");
            mount.live.as_mut().unwrap().channel_mut().lose_next_receipt = true;
            assert!(
                block_on(mount.before_request(
                    fs,
                    session,
                    &FsRequest::CommitReplace {
                        handle: file.handle
                    }
                ))
                .is_err()
            );
            assert_eq!(mount.channel.as_ref().unwrap().content(), b"hello");
            assert!(mount.reconcile_required);
            block_on(mount.refresh(fs)).unwrap();
            assert_eq!(
                mount.live.as_mut().unwrap().channel_mut().content(),
                b"hello"
            );
            assert_eq!(
                fs.dispatch(
                    session,
                    &FsRequest::Close {
                        handle: file.handle
                    },
                    None
                ),
                FsResponse::Done
            );
        });
    }

    #[test]
    fn mounted_atomic_replacement_read_only_refuses_and_empty_replacement_commits() {
        with_workspace(true, |mount, fs, session, _, src, file| {
            let reader = fs
                .engine_ref()
                .open_at(
                    session,
                    src,
                    &crate::Name::new("main.rs").unwrap(),
                    Rights::READ,
                )
                .unwrap();
            assert_eq!(
                block_on(mount.before_request(
                    fs,
                    session,
                    &FsRequest::BeginReplace {
                        handle: reader.handle,
                        length: 0,
                        expected: crate::ANY_REVISION,
                    }
                )),
                Err(FsError::AccessDenied)
            );
            assert_eq!(
                block_on(mount.before_request(
                    fs,
                    session,
                    &FsRequest::BeginReplace {
                        handle: file.handle,
                        length: 0,
                        expected: crate::ANY_REVISION,
                    }
                )),
                Ok(Some(FsResponse::Done))
            );
            assert!(matches!(
                block_on(mount.before_request(
                    fs,
                    session,
                    &FsRequest::CommitReplace {
                        handle: file.handle
                    }
                )),
                Ok(Some(FsResponse::Committed { .. }))
            ));
            assert_eq!(mount.live.as_mut().unwrap().channel_mut().content(), b"");
            assert_eq!(fs.engine_ref().stat(session, file.handle).unwrap().len, 0);
        });
    }
    #[test]
    fn mounted_atomic_replacement_chunks_remain_private_and_stale_commit_refuses() {
        with_workspace(true, |mount, fs, session, _, _, file| {
            stage(mount, fs, session, file.handle, b"old");
            block_on(mount.before_request(
                fs,
                session,
                &FsRequest::CommitReplace {
                    handle: file.handle,
                },
            ))
            .unwrap();
            let request = FsRequest::BeginReplace {
                handle: file.handle,
                length: 5,
                expected: crate::ANY_REVISION,
            };
            assert_eq!(
                block_on(mount.before_request(fs, session, &request)),
                Ok(Some(FsResponse::Done))
            );
            fs.dispatch(
                session,
                &FsRequest::AttachWindow { slot: 0, bytes: 3 },
                None,
            );
            for (offset, input) in [(0, &b"he"[..]), (2, &b"llo"[..])] {
                let mut bytes = input.to_vec();
                assert_eq!(
                    block_on(mount.replace_file(
                        fs,
                        session,
                        &FsRequest::WriteReplacement {
                            handle: file.handle,
                            offset,
                            window_offset: 0,
                            length: bytes.len() as u32,
                        },
                        Some(&mut DataWindow::new(&mut bytes))
                    )),
                    Ok(Some(FsResponse::Done))
                );
                assert_eq!(mount.live.as_mut().unwrap().channel_mut().content(), b"old");
            }
            let mut winner = *b"new";
            block_on(mount.write_file(
                fs,
                session,
                &FsRequest::WriteAt {
                    handle: file.handle,
                    offset: 0,
                    window_offset: 0,
                    length: 3,
                    expected: crate::ANY_REVISION,
                },
                Some(&mut DataWindow::new(&mut winner)),
            ))
            .unwrap();
            block_on(mount.refresh(fs)).unwrap();
            assert_eq!(
                block_on(mount.before_request(
                    fs,
                    session,
                    &FsRequest::CommitReplace {
                        handle: file.handle
                    }
                )),
                Err(FsError::Conflict)
            );
            assert_eq!(mount.channel.as_ref().unwrap().content(), b"new");
            assert_eq!(mount.channel.as_ref().unwrap().transactions, 2);
            assert_eq!(
                fs.dispatch(
                    session,
                    &FsRequest::Close {
                        handle: file.handle
                    },
                    None
                ),
                FsResponse::Done
            );
        });
    }
}
