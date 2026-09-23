//! Native namespace claims retained across the workspace owner's transaction.
//! No wire request supplies this callback or the host's change description.
use super::*;

impl<P: StorageProvider> Engine<P> {
    pub(crate) async fn create_with_commit<C>(
        &self,
        session: SessionId,
        parent: Handle,
        name: &Name,
        kind: ObjectKind,
        subtype: ObjectSubtype,
        commit_host: C,
    ) -> FsResult<OpenedObject>
    where
        C: AsyncFnOnce(StableObjectId) -> FsResult<()>,
    {
        let route = self.route(session, parent)?;
        route.rights.require(Rights::MUTATE_NAMESPACE)?;
        if !subtype.is_plain() && (kind.is_dir() || !self.durable_subtypes) {
            return Err(FsError::Unsupported);
        }
        if !self.ns.kind(route.object)?.is_dir() {
            return Err(FsError::NotDirectory);
        }
        let _view = self.claim_view_if_root(session, parent)?;
        self.require_view_name_free(session, parent, name)?;
        let (grant, proof) = {
            let record = self.auth.handle(session, parent)?;
            (record.grant(), record.proof().clone())
        };
        if proof.depth() >= MAX_PATH_DEPTH {
            return Err(FsError::Range);
        }
        let _quota = self.require_growth(route, 1, 1, 0)?;
        let prepared = self
            .ns
            .prepare_creation(route.object, name, kind, subtype)?;
        let handle = self.auth.prepare_handle(
            session,
            prepared.child(),
            grant,
            proof.extend(prepared.edge())?,
            route.rights,
        )?;
        let stable_object = prepared.stable();
        let revision = prepared.revision();
        commit_host(stable_object).await?;
        self.route(session, parent)?
            .rights
            .require(Rights::MUTATE_NAMESPACE)?;
        prepared.commit();
        Ok(OpenedObject {
            handle: handle.commit(),
            stable_object,
            kind,
            subtype,
            len: 0,
            revision,
        })
    }

    pub(crate) async fn relocate_host_entry<C>(
        &self,
        session: SessionId,
        src_parent: Handle,
        name: &Name,
        dst_parent: Handle,
        new_name: &Name,
        expected_child: StableObjectId,
        commit_host: C,
    ) -> FsResult<()>
    where
        C: AsyncFnOnce() -> FsResult<()>,
    {
        let src = self.route(session, src_parent)?;
        let dst = self.route(session, dst_parent)?;
        src.rights.require(Rights::MUTATE_NAMESPACE)?;
        dst.rights.require(Rights::MUTATE_NAMESPACE)?;
        self.require_same_authority(session, src_parent, dst_parent)?;
        let _view = self.claim_view_if_root(session, dst_parent)?;
        self.require_view_name_free(session, dst_parent, new_name)?;
        let prepared = self
            .ns
            .prepare_relocate(src.object, name, dst.object, new_name)?;
        if self.ns.object(prepared.child())?.stable() != expected_child {
            return Err(FsError::Conflict);
        }
        commit_host().await?;
        for parent in [src_parent, dst_parent] {
            self.auth
                .revalidate_using(session, parent, &self.ns, |edge| {
                    prepared
                        .edge_child(edge)
                        .unwrap_or_else(|| self.ns.edge_child(edge))
                })?
                .rights
                .require(Rights::MUTATE_NAMESPACE)?;
        }
        prepared.commit().map(|_| ())
    }

    pub(crate) async fn rename_host_entry<C>(
        &self,
        session: SessionId,
        parent: Handle,
        old: &Name,
        new: &Name,
        expected_child: StableObjectId,
        commit_host: C,
    ) -> FsResult<()>
    where
        C: AsyncFnOnce() -> FsResult<()>,
    {
        let route = self.route(session, parent)?;
        route.rights.require(Rights::MUTATE_NAMESPACE)?;
        let _view = self.claim_view_if_root(session, parent)?;
        self.require_view_name_free(session, parent, new)?;
        let prepared = self.ns.prepare_rename(route.object, old, new)?;
        if self.ns.object(prepared.child())?.stable() != expected_child {
            return Err(FsError::Conflict);
        }
        commit_host().await?;
        // The parent proof does not traverse its own child list. Revalidate
        // it while retaining the exact directory claim, before publication.
        self.route(session, parent)?
            .rights
            .require(Rights::MUTATE_NAMESPACE)?;
        prepared.commit().map(|_| ())
    }

    pub(crate) async fn remove_host_entry<C>(
        &self,
        session: SessionId,
        parent: Handle,
        name: &Name,
        expected_child: StableObjectId,
        require_empty: bool,
        commit_host: C,
    ) -> FsResult<()>
    where
        C: AsyncFnOnce() -> FsResult<()>,
    {
        let route = self.route(session, parent)?;
        route.rights.require(Rights::MUTATE_NAMESPACE)?;
        let prepared = self.ns.prepare_removal(route.object, name, require_empty)?;
        if prepared.stable() != expected_child {
            return Err(FsError::Conflict);
        }
        commit_host().await?;
        self.route(session, parent)?
            .rights
            .require(Rights::MUTATE_NAMESPACE)?;
        let (_, child) = prepared.commit()?;
        self.reclaim(child).await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_on;
    use core::{cell::Cell, future::Future};

    fn name(value: &str) -> Name {
        Name::new(value).unwrap()
    }

    fn fixture() -> (
        RamEngine,
        SessionId,
        Handle,
        crate::OpenedObject,
        crate::OpenedObject,
    ) {
        let engine = RamEngine::default();
        let session = SessionId::new(0);
        block_on(engine.attach_session(session)).unwrap();
        let root = engine.open_root(session, Rights::ALL).unwrap().handle;
        let directory =
            block_on(engine.create(session, root, &name("dir"), ObjectKind::Directory)).unwrap();
        let file = block_on(engine.create(session, root, &name("file"), ObjectKind::File)).unwrap();
        (engine, session, root, directory, file)
    }

    #[test]
    fn cancelled_creation_releases_objects_directories_handles_and_quota() {
        let (engine, session, root, directory, file) = fixture();
        let before = engine.object_count();
        let handles = engine.auth.live_handles(session);
        // More aborted directories than the arena capacity proves their
        // child-list reservations are returned, not just the object slots.
        for _ in 0..crate::bounds::MAX_DIRECTORIES + 2 {
            let fresh = name("fresh");
            let mut future = std::boxed::Box::pin(engine.create_with_commit(
                session,
                directory.handle,
                &fresh,
                ObjectKind::Directory,
                ObjectSubtype::PLAIN,
                async |_| core::future::pending::<FsResult<()>>().await,
            ));
            let mut cx = core::task::Context::from_waker(core::task::Waker::noop());
            assert!(future.as_mut().poll(&mut cx).is_pending());
            assert_eq!(engine.object_count(), before + 1);
            assert_eq!(engine.auth.live_handles(session), handles + 1);
            assert_eq!(
                engine.open_at(session, directory.handle, &fresh, Rights::READ),
                Err(FsError::Busy)
            );
            assert!(engine.stat(session, file.handle).is_ok());
            drop(future);
            assert_eq!(engine.object_count(), before);
            assert_eq!(engine.auth.live_handles(session), handles);
            assert_eq!(
                engine.open_at(session, directory.handle, &fresh, Rights::READ),
                Err(FsError::NotFound)
            );
        }
        let created =
            block_on(engine.create(session, root, &name("success"), ObjectKind::Directory))
                .unwrap();
        assert!(engine.stat(session, created.handle).is_ok());
    }

    #[test]
    fn creation_reserves_handle_capacity_and_checks_names_before_host_effects() {
        let (engine, session, root, _, _) = fixture();
        let calls = Cell::new(0);
        let before = engine.object_count();
        let make = async |name: Name| {
            engine
                .create_with_commit(
                    session,
                    root,
                    &name,
                    ObjectKind::File,
                    ObjectSubtype::PLAIN,
                    async |_| {
                        calls.set(calls.get() + 1);
                        Ok(())
                    },
                )
                .await
        };
        assert_eq!(block_on(make(name("file"))), Err(FsError::AlreadyExists));
        let mut opened = std::vec::Vec::new();
        loop {
            match engine.open_root(session, Rights::READ) {
                Ok(object) => opened.push(object.handle),
                Err(FsError::OutOfHandles) => break,
                other => panic!("unexpected open outcome: {other:?}"),
            }
        }
        assert_eq!(block_on(make(name("fresh"))), Err(FsError::OutOfHandles));
        assert_eq!(calls.get(), 0);
        assert_eq!(engine.object_count(), before);
        for handle in opened {
            engine.close(session, handle).unwrap();
        }
        let created = block_on(make(name("fresh"))).unwrap();
        assert_eq!(calls.get(), 1);
        assert!(engine.stat(session, created.handle).is_ok());
    }

    #[test]
    fn host_creation_refusal_leaves_no_native_object_or_name() {
        let (engine, session, root, _, _) = fixture();
        let before = engine.object_count();
        let handles = engine.auth.live_handles(session);
        assert_eq!(
            block_on(engine.create_with_commit(
                session,
                root,
                &name("new"),
                ObjectKind::File,
                ObjectSubtype::PLAIN,
                async |_| Err(FsError::Conflict)
            )),
            Err(FsError::Conflict)
        );
        assert_eq!(engine.object_count(), before);
        assert_eq!(engine.auth.live_handles(session), handles);
        assert_eq!(
            engine.open_at(session, root, &name("new"), Rights::READ),
            Err(FsError::NotFound)
        );
    }

    #[test]
    fn host_namespace_refusal_and_cancellation_do_not_publish_a_local_rename() {
        let (engine, session, root, directory, file) = fixture();
        assert_eq!(
            block_on(engine.rename_host_entry(
                session,
                root,
                &name("file"),
                &name("next"),
                file.stable_object,
                async || Err(FsError::Conflict)
            )),
            Err(FsError::Conflict)
        );
        let calls = Cell::new(0);
        {
            let old = name("file");
            let new = name("next");
            let future = engine.rename_host_entry(
                session,
                root,
                &old,
                &new,
                file.stable_object,
                async || {
                    calls.set(calls.get() + 1);
                    core::future::pending::<FsResult<()>>().await
                },
            );
            let mut future = core::pin::pin!(future);
            let mut cx = core::task::Context::from_waker(core::task::Waker::noop());
            assert!(future.as_mut().poll(&mut cx).is_pending());
            assert_eq!(
                engine.rename(session, root, &name("file"), &name("racing")),
                Err(FsError::Busy)
            );
        }
        assert_eq!(calls.get(), 1);
        assert_eq!(
            engine.open_at(session, root, &name("next"), Rights::READ),
            Err(FsError::NotFound)
        );
        assert!(engine.stat(session, file.handle).is_ok());
        // Failed preconditions never reach the host callback.
        assert_eq!(
            block_on(engine.rename_host_entry(
                session,
                root,
                &name("file"),
                &name("dir"),
                file.stable_object,
                async || {
                    calls.set(99);
                    Ok(())
                }
            )),
            Err(FsError::AlreadyExists)
        );
        assert_eq!(
            block_on(engine.rename_host_entry(
                session,
                root,
                &name("file"),
                &name("next"),
                directory.stable_object,
                async || {
                    calls.set(99);
                    Ok(())
                }
            )),
            Err(FsError::Conflict)
        );
        assert_eq!(calls.get(), 1);
        block_on(engine.rename_host_entry(
            session,
            root,
            &name("file"),
            &name("next"),
            file.stable_object,
            async || Ok(()),
        ))
        .unwrap();
        assert!(
            engine.stat(session, file.handle).is_ok(),
            "rename preserves the native edge generation"
        );
    }

    #[test]
    fn host_move_revalidates_overlapping_parent_proofs_and_retires_only_source_edge() {
        let (engine, session, root, directory, file) = fixture();
        block_on(engine.relocate_host_entry(
            session,
            root,
            &name("file"),
            directory.handle,
            &name("moved"),
            file.stable_object,
            async || Ok(()),
        ))
        .unwrap();
        assert_eq!(engine.stat(session, file.handle), Err(FsError::AccessLost));
        let moved = engine
            .open_at(session, directory.handle, &name("moved"), Rights::READ)
            .unwrap();
        assert_eq!(moved.stable_object, file.stable_object);
        let calls = Cell::new(0);
        assert_eq!(
            block_on(engine.relocate_host_entry(
                session,
                root,
                &name("dir"),
                directory.handle,
                &name("cycle"),
                directory.stable_object,
                async || {
                    calls.set(1);
                    Ok(())
                }
            )),
            Err(FsError::Cycle)
        );
        assert_eq!(calls.get(), 0);
    }

    #[test]
    fn host_removal_checks_emptiness_and_preserves_all_names_until_commit() {
        let (engine, session, root, directory, file) = fixture();
        block_on(engine.create(session, directory.handle, &name("child"), ObjectKind::File))
            .unwrap();
        let calls = Cell::new(0);
        assert_eq!(
            block_on(engine.remove_host_entry(
                session,
                root,
                &name("dir"),
                directory.stable_object,
                true,
                async || {
                    calls.set(1);
                    Ok(())
                }
            )),
            Err(FsError::NotEmpty)
        );
        assert_eq!(calls.get(), 0);
        assert_eq!(
            block_on(engine.remove_host_entry(
                session,
                root,
                &name("file"),
                file.stable_object,
                true,
                async || Err(FsError::Conflict)
            )),
            Err(FsError::Conflict)
        );
        assert!(engine.stat(session, file.handle).is_ok());
        block_on(engine.remove_host_entry(
            session,
            root,
            &name("dir"),
            directory.stable_object,
            false,
            async || Ok(()),
        ))
        .unwrap();
        assert!(engine.stat(session, directory.handle).is_err());
        assert!(engine.stat(session, file.handle).is_ok());
    }

    #[test]
    fn host_namespace_cannot_publish_locally_after_its_grant_is_revoked() {
        let (engine, issuer, root, directory, _) = fixture();
        block_on(engine.create(issuer, directory.handle, &name("child"), ObjectKind::File))
            .unwrap();
        let session = SessionId::new(2);
        block_on(engine.attach_session(session)).unwrap();
        let grant = engine
            .grant(issuer, directory.handle, session, Rights::HOME, None)
            .unwrap();
        let parent = engine.open_grant(session, grant, Rights::HOME).unwrap();
        let child = engine
            .open_at(session, parent.handle, &name("child"), Rights::READ)
            .unwrap();
        assert_eq!(
            block_on(engine.rename_host_entry(
                session,
                parent.handle,
                &name("child"),
                &name("next"),
                child.stable_object,
                async || { engine.revoke(issuer, grant).await }
            )),
            Err(FsError::AccessLost)
        );
        assert!(
            engine
                .open_at(issuer, directory.handle, &name("child"), Rights::READ)
                .is_ok()
        );
        assert_eq!(
            engine.open_at(issuer, directory.handle, &name("next"), Rights::READ),
            Err(FsError::NotFound)
        );
        let readonly = engine
            .open_at(issuer, root, &name("dir"), Rights::READ)
            .unwrap();
        assert_eq!(
            block_on(engine.rename_host_entry(
                issuer,
                readonly.handle,
                &name("child"),
                &name("next"),
                child.stable_object,
                async || panic!("read-only authority reached host mutation")
            )),
            Err(FsError::AccessDenied)
        );
    }
}
