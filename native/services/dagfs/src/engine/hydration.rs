//! Trusted source materialization through native per-object claims, quota,
//! placement ownership and immutable-version lifetimes. No wire verb exposes
//! this: the workspace owner supplies the verified source identity.

use super::*;

pub(crate) struct Hydration {
    pub len: u64,
    pub digest: [u8; 32],
    pub expected: Revision,
    /// The authority needed by the requesting operation: a read or version
    /// lend needs READ, an executable lend also BIND_EXECUTABLE, and a write
    /// may merge existing bytes without disclosing them to a write-only caller.
    pub required: Rights,
}

impl<P: StorageProvider> Engine<P> {
    /// Fill an unpublished placement and select it only after the complete
    /// digest is verified. Filling a placeholder or fetching identical bytes
    /// preserves the native revision. Replacing an already selected, different
    /// source advances it so immutable/derived identities cannot alias two
    /// contents. The caller keeps its host record unmaterialized until success.
    pub(crate) async fn hydrate_host_file<F>(
        &self,
        session: SessionId,
        handle: Handle,
        hydration: Hydration,
        read: F,
    ) -> FsResult<Revision>
    where
        F: for<'a> AsyncFnMut(u64, &'a mut [u8]) -> FsResult<usize>,
    {
        self.select_verified_file(session, handle, hydration, read, false, async || Ok(()))
            .await
    }

    /// Stage the complete replacement before asking the host to commit it.
    /// A refused host transaction leaves native contents and revision intact.
    pub(crate) async fn replace_host_file<C>(
        &self,
        session: SessionId,
        handle: Handle,
        expected: Revision,
        bytes: &[u8],
        commit_host: C,
    ) -> FsResult<Revision>
    where
        C: AsyncFnOnce() -> FsResult<()>,
    {
        self.select_verified_file(
            session,
            handle,
            Hydration {
                len: bytes.len() as u64,
                digest: *blake3::hash(bytes).as_bytes(),
                expected,
                required: Rights::WRITE,
            },
            async |offset, out: &mut [u8]| {
                let start = offset as usize;
                out.copy_from_slice(&bytes[start..start + out.len()]);
                Ok(out.len())
            },
            true,
            commit_host,
        )
        .await
    }

    async fn select_verified_file<F, C>(
        &self,
        session: SessionId,
        handle: Handle,
        hydration: Hydration,
        mut read: F,
        mutation: bool,
        commit_host: C,
    ) -> FsResult<Revision>
    where
        F: for<'a> AsyncFnMut(u64, &'a mut [u8]) -> FsResult<usize>,
        C: AsyncFnOnce() -> FsResult<()>,
    {
        let route = self.route(session, handle)?;
        route.rights.require(hydration.required)?;
        if hydration.len > self.provider.limits().max_object_bytes {
            return Err(FsError::OutOfStorage);
        }
        let mut object = self.ns.object_mut(route.object)?;
        if !object.kind().is_file() {
            return Err(FsError::IsDirectory);
        }
        if !object.revision_admits(hydration.expected) {
            return Err(FsError::Conflict);
        }
        let _quota = self.require_growth_with_claimed(
            route,
            &object,
            0,
            0,
            hydration.len.saturating_sub(object.len()),
        )?;
        let key = match object.staged_storage() {
            Some(key) => {
                self.provider.resize(key, hydration.len).await?;
                key
            }
            None => {
                let key = self
                    .provider
                    .allocate(ObjectKind::File, hydration.len)
                    .await?;
                // No await separates allocation completion from its owner.
                object.set_staged_storage(Some(key));
                key
            }
        };
        let mut digest = blake3::Hasher::new();
        let mut bytes = [0u8; MAX_TRANSFER];
        let limit = self.provider.limits().max_transfer.min(bytes.len());
        if limit == 0 {
            return Err(FsError::Range);
        }
        let mut offset = 0;
        while offset < hydration.len {
            let want = (hydration.len - offset).min(limit as u64) as usize;
            let count = read(offset, &mut bytes[..want]).await?;
            if count == 0 || count > want {
                return Err(FsError::ProviderFailure(ProviderError::Integrity));
            }
            self.provider.write(key, offset, &bytes[..count]).await?;
            digest.update(&bytes[..count]);
            offset += count as u64;
        }
        if digest.finalize().as_bytes() != &hydration.digest {
            return Err(FsError::ProviderFailure(ProviderError::Integrity));
        }
        self.provider.sync(key).await?;
        // An endpoint can be retired while source/provider I/O is pending.
        // Revalidate the original proof before publishing a storage change.
        let current = self.route(session, handle)?;
        current.rights.require(hydration.required)?;
        if current.object != route.object {
            return Err(FsError::AccessLost);
        }
        let old = object.storage();
        let preserve_revision = !mutation
            && match old {
                None => true, // A topology placeholder has no selected bytes yet.
                Some(old) if object.len() == hydration.len => {
                    let mut previous = blake3::Hasher::new();
                    let mut offset = 0;
                    while offset < object.len() {
                        let want = (object.len() - offset).min(limit as u64) as usize;
                        let count = self.provider.read(old, offset, &mut bytes[..want]).await?;
                        if count != want {
                            return Err(FsError::ProviderFailure(ProviderError::Integrity));
                        }
                        previous.update(&bytes[..count]);
                        offset += count as u64;
                    }
                    previous.finalize().as_bytes() == &hydration.digest
                }
                Some(_) => false,
            };
        // The selected-source comparison may also await provider I/O.
        self.route(session, handle)?
            .rights
            .require(hydration.required)?;
        let old_borrowed = match old {
            Some(old) => self.extraction.storage_borrowers(old, None)? != 0,
            None => false,
        };
        // All fallible native preparation is finished before the host sees
        // a mutation. The object and its quota remain claimed during the RPC.
        commit_host().await?;
        self.route(session, handle)?
            .rights
            .require(hydration.required)?;
        object.set_storage(key);
        // A borrowed old placement passes to the native lease owner. An
        // unborrowed one remains this object's one reusable staging buffer.
        object.set_staged_storage(if old_borrowed { None } else { old });
        Ok(if preserve_revision {
            self.ns
                .hydrate_file_len(route.object, &mut object, hydration.len)
        } else {
            self.ns
                .commit_file_len(route.object, &mut object, hydration.len)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{block_on, namespace::Name};

    fn fixture() -> (RamEngine, SessionId, Handle, crate::OpenedObject) {
        let engine = RamEngine::default();
        let session = SessionId::new(0);
        block_on(engine.attach_session(session)).unwrap();
        let root = engine.open_root(session, Rights::ALL).unwrap().handle;
        let file =
            block_on(engine.create(session, root, &Name::new("host").unwrap(), ObjectKind::File))
                .unwrap();
        (engine, session, root, file)
    }

    fn source(bytes: &[u8], expected: Revision, required: Rights) -> Hydration {
        Hydration {
            len: bytes.len() as u64,
            digest: *blake3::hash(bytes).as_bytes(),
            expected,
            required,
        }
    }

    #[test]
    fn hydration_preserves_revision_for_the_first_conditional_write() {
        let (engine, session, _, file) = fixture();
        let bytes = b"verified host bytes";
        let revision = block_on(engine.hydrate_host_file(
            session,
            file.handle,
            source(bytes, file.revision, Rights::WRITE),
            async |offset, out: &mut [u8]| {
                let count = out.len().min(bytes.len() - offset as usize);
                out[..count].copy_from_slice(&bytes[offset as usize..offset as usize + count]);
                Ok(count)
            },
        ))
        .unwrap();
        assert_eq!(revision, file.revision);
        assert_eq!(
            engine.stat(session, file.handle).unwrap().revision,
            file.revision
        );
        assert_eq!(engine.system_tree_usage().unwrap().2, bytes.len() as u64);
        assert_eq!(
            block_on(engine.write_at(session, file.handle, 0, b"V", file.revision)),
            Ok(file.revision + 1)
        );
        let mut out = [0; 19];
        assert_eq!(
            block_on(engine.read(session, file.handle, 0, &mut out)).unwrap(),
            bytes.len()
        );
        assert_eq!(&out, b"Verified host bytes");
    }

    #[test]
    fn hydration_rejects_digest_before_selecting_any_bytes_and_reuses_staging() {
        let (engine, session, root, file) = fixture();
        block_on(engine.write_at(session, file.handle, 0, b"old", file.revision)).unwrap();
        let revision = file.revision + 1;
        let wrong = b"corrupt";
        assert_eq!(
            block_on(engine.hydrate_host_file(
                session,
                file.handle,
                source(b"trusted", revision, Rights::READ),
                async |_, out: &mut [u8]| {
                    out.copy_from_slice(wrong);
                    Ok(wrong.len())
                }
            )),
            Err(FsError::ProviderFailure(ProviderError::Integrity))
        );
        assert_eq!(
            engine.stat(session, file.handle).unwrap().revision,
            revision
        );
        assert_eq!(engine.stat(session, file.handle).unwrap().len, 3);
        let mut out = [0; 7];
        assert_eq!(
            block_on(engine.read(session, file.handle, 0, &mut out)).unwrap(),
            3
        );
        assert_eq!(&out[..3], b"old");
        let route = engine.route(session, file.handle).unwrap();
        let staged = engine
            .ns
            .object(route.object)
            .unwrap()
            .staged_storage()
            .unwrap();
        block_on(engine.hydrate_host_file(
            session,
            file.handle,
            source(b"trusted", revision, Rights::READ),
            async |_, out: &mut [u8]| {
                out.copy_from_slice(b"trusted");
                Ok(out.len())
            },
        ))
        .unwrap();
        assert_eq!(
            engine.ns.object(route.object).unwrap().storage(),
            Some(staged)
        );
        block_on(engine.read(session, file.handle, 0, &mut out)).unwrap();
        assert_eq!(&out, b"trusted");
        // The selected and reusable staging placements both belong to the
        // object and must be reclaimed together.
        block_on(engine.unlink(session, root, &Name::new("host").unwrap())).unwrap();
        assert_eq!(engine.provider().chunks_in_use(), 0);
    }

    #[test]
    fn hydration_cancelled_during_source_io_keeps_staging_owned_until_reclamation() {
        use core::{
            future::Future,
            task::{Context, Poll, Waker},
        };
        let (engine, session, root, file) = fixture();
        let before = engine.provider().chunks_in_use();
        let mut future = std::boxed::Box::pin(engine.hydrate_host_file(
            session,
            file.handle,
            source(b"pending", file.revision, Rights::READ),
            async |_, _: &mut [u8]| core::future::pending::<FsResult<usize>>().await,
        ));
        assert!(matches!(
            future
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop())),
            Poll::Pending
        ));
        drop(future);
        assert_eq!(engine.stat(session, file.handle).unwrap().len, 0);
        assert!(engine.provider().chunks_in_use() > before);
        block_on(engine.unlink(session, root, &Name::new("host").unwrap())).unwrap();
        assert_eq!(engine.provider().chunks_in_use(), before);
        assert_eq!(engine.leaked_placements(), 0);
    }

    #[test]
    fn hydration_keeps_a_previously_lent_version_immutable() {
        let (engine, session, _, file) = fixture();
        block_on(engine.write_at(session, file.handle, 0, b"old", file.revision)).unwrap();
        let owner = ExtractionOwner::new(17);
        let (lease, _) = block_on(engine.lend_version(owner, session, file.handle)).unwrap();
        block_on(engine.hydrate_host_file(
            session,
            file.handle,
            source(b"new", file.revision + 1, Rights::READ),
            async |_, out: &mut [u8]| {
                out.copy_from_slice(b"new");
                Ok(out.len())
            },
        ))
        .unwrap();
        assert_eq!(
            engine.stat(session, file.handle).unwrap().revision,
            file.revision + 2
        );
        let mut out = [0; 3];
        block_on(engine.read(session, file.handle, 0, &mut out)).unwrap();
        assert_eq!(&out, b"new");
        block_on(engine.read_lent_version(owner, lease, 0, &mut out)).unwrap();
        assert_eq!(&out, b"old");
        block_on(engine.release_lent_version(owner, lease)).unwrap();
    }

    #[test]
    fn hydration_checks_rights_and_expected_revision_before_source_io() {
        let (engine, session, root, file) = fixture();
        let read_only = engine
            .open_at(session, root, &Name::new("host").unwrap(), Rights::READ)
            .unwrap();
        assert_eq!(
            block_on(engine.hydrate_host_file(
                session,
                read_only.handle,
                source(b"a", file.revision, Rights::WRITE),
                async |_, _: &mut [u8]| { panic!("unauthorized source access") }
            )),
            Err(FsError::AccessDenied)
        );
        assert_eq!(
            block_on(engine.hydrate_host_file(
                session,
                file.handle,
                source(b"a", file.revision + 1, Rights::READ),
                async |_, _: &mut [u8]| { panic!("stale source access") }
            )),
            Err(FsError::Conflict)
        );
    }
    #[test]
    fn hydration_revalidates_a_handle_closed_during_source_io() {
        let (engine, session, root, file) = fixture();
        assert_eq!(
            block_on(engine.hydrate_host_file(
                session,
                file.handle,
                source(b"new", file.revision, Rights::READ),
                async |_, out: &mut [u8]| {
                    engine.close(session, file.handle).unwrap();
                    out.copy_from_slice(b"new");
                    Ok(out.len())
                }
            )),
            Err(FsError::InvalidHandle)
        );
        let reopened = engine
            .open_at(session, root, &Name::new("host").unwrap(), Rights::READ)
            .unwrap();
        assert_eq!(reopened.len, 0);
        assert_eq!(reopened.revision, file.revision);
        block_on(engine.unlink(session, root, &Name::new("host").unwrap())).unwrap();
        assert_eq!(engine.provider().chunks_in_use(), 0);
    }

    #[test]
    fn hydration_reserves_growth_against_the_exact_granted_quota() {
        let (engine, source_session, _, file) = fixture();
        let target = SessionId::new(1);
        block_on(engine.attach_session(target)).unwrap();
        let grant = engine
            .grant_scoped(
                source_session,
                file.handle,
                target,
                Rights::READ,
                AuthorityQuota::new(1, 0, 2, 1),
                None,
            )
            .unwrap();
        let file = engine.open_grant(target, grant, Rights::READ).unwrap();
        assert_eq!(
            block_on(engine.hydrate_host_file(
                target,
                file.handle,
                source(b"too large", file.revision, Rights::READ),
                async |_, _: &mut [u8]| { panic!("quota must be checked before fetching") }
            )),
            Err(FsError::OutOfStorage)
        );
        assert_eq!(engine.provider().chunks_in_use(), 0);
    }

    #[test]
    fn hydration_claims_only_its_file_and_preserves_selected_bytes_on_provider_failure() {
        let (engine, session, root, file) = fixture();
        let other = block_on(engine.create(
            session,
            root,
            &Name::new("other").unwrap(),
            ObjectKind::File,
        ))
        .unwrap();
        block_on(engine.write_at(session, file.handle, 0, b"old", file.revision)).unwrap();
        let expected = file.revision + 1;
        assert_eq!(
            block_on(engine.hydrate_host_file(
                session,
                file.handle,
                source(b"new", expected, Rights::READ),
                async |_, out: &mut [u8]| {
                    assert_eq!(
                        engine
                            .write_at(session, file.handle, 0, b"!", expected)
                            .await,
                        Err(FsError::Busy)
                    );
                    engine
                        .write_at(session, other.handle, 0, b"ok", other.revision)
                        .await
                        .unwrap();
                    engine
                        .provider()
                        .inject(crate::provider::Fault::Write(ProviderError::Media));
                    out.copy_from_slice(b"new");
                    Ok(out.len())
                }
            )),
            Err(FsError::ProviderFailure(ProviderError::Media))
        );
        let mut bytes = [0; 3];
        block_on(engine.read(session, file.handle, 0, &mut bytes)).unwrap();
        assert_eq!(&bytes, b"old");
        assert_eq!(
            engine.stat(session, file.handle).unwrap().revision,
            expected
        );
    }
    #[test]
    fn hydration_of_identical_selected_bytes_keeps_the_source_revision() {
        let (engine, session, _, file) = fixture();
        block_on(engine.write_at(session, file.handle, 0, b"same", file.revision)).unwrap();
        let expected = file.revision + 1;
        assert_eq!(
            block_on(engine.hydrate_host_file(
                session,
                file.handle,
                source(b"same", expected, Rights::READ),
                async |_, out: &mut [u8]| {
                    out.copy_from_slice(b"same");
                    Ok(out.len())
                }
            )),
            Ok(expected)
        );
    }
}
