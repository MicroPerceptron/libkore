//! Workspace import adapter over the ordinary capability filesystem engine.
//!
//! The importer writes beneath a system-owned container. The broker receives
//! no route until the manifest transaction has verified every byte and the
//! container has moved to its committed name. Runtime handles and grants are
//! never serialized; only the container, data subtree, and snapshot marker
//! are durable namespace facts.

use heapless::Vec;
use kore_workspace_sync::{
    BeginOutcome, CommitFailure, Entry, ImportStore, SnapshotHeader, bounds::MAX_DIRECTORIES,
};
use zerocopy::{FromBytes, IntoBytes};

use crate::{
    AuthorityKind, Engine, FsError, FsResult, GrantId, Handle, Name, ObjectKind, Revision, Rights,
    SessionId, StorageProvider,
};

const STAGING_NAME: &str = ".kore-import";
const CURRENT_NAME: &str = ".kore-workspace";
const DATA_NAME: &str = "data";
const SNAPSHOT_NAME: &str = "snapshot";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorkspaceCommit {
    pub snapshot: [u8; 32],
    pub broker_root: GrantId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LiveWorkspaceNode {
    pub path: kore_workspace_sync::CanonicalPath,
    pub stable_object: crate::StableObjectId,
    pub kind: kore_workspace_sync::EntryKind,
    pub len: u32,
    pub digest: [u8; 32],
    pub executable: bool,
}

pub struct LiveWorkspaceImport {
    pub commit: WorkspaceCommit,
    pub root_stable_object: crate::StableObjectId,
    pub nodes: Vec<LiveWorkspaceNode, { kore_workspace_sync::bounds::MAX_ENTRIES }>,
}

pub struct WorkspaceStore<'a, P: StorageProvider> {
    engine: &'a Engine<P>,
    importer: SessionId,
    broker: SessionId,
}

/// Synchronous handle retirement also runs when an awaited marker operation
/// is cancelled. These are importer-session handles, never persisted authority.
struct ImportHandles<'a, P: StorageProvider> {
    engine: &'a Engine<P>,
    session: SessionId,
    handles: Vec<Handle, 4>,
}

impl<'a, P: StorageProvider> ImportHandles<'a, P> {
    fn new(engine: &'a Engine<P>, session: SessionId) -> Self {
        Self {
            engine,
            session,
            handles: Vec::new(),
        }
    }

    fn keep(&mut self, opened: crate::OpenedObject) -> FsResult<crate::OpenedObject> {
        if self.handles.push(opened.handle).is_err() {
            let _ = self.engine.close(self.session, opened.handle);
            return Err(FsError::OutOfHandles);
        }
        Ok(opened)
    }
}

impl<P: StorageProvider> Drop for ImportHandles<'_, P> {
    fn drop(&mut self) {
        while let Some(handle) = self.handles.pop() {
            let _ = self.engine.close(self.session, handle);
        }
    }
}

impl<'a, P: StorageProvider> WorkspaceStore<'a, P> {
    pub const fn new(engine: &'a Engine<P>, importer: SessionId, broker: SessionId) -> Self {
        Self {
            engine,
            importer,
            broker,
        }
    }

    /// Publish validated live topology without materializing host file bytes.
    /// File records remain zero-length locally and are paired with an
    /// importer-private logical-length table by `FsService`; only a later COW
    /// write allocates provider storage.
    pub async fn import_live_topology(
        &mut self,
        manifest: &kore_workspace_sync::Manifest,
        policy: kore_workspace_sync::Policy,
    ) -> FsResult<LiveWorkspaceImport> {
        manifest.verify(policy).map_err(|_| FsError::Conflict)?;
        let snapshot = manifest.header();
        let mut staging = match self.begin(&snapshot).await? {
            BeginOutcome::AlreadyCommitted(commit) => {
                return self.catalog_committed_live(manifest, commit);
            }
            BeginOutcome::Staging(staging) => staging,
        };
        let root_stable_object = self.engine.stat(self.importer, staging.data)?.stable_object;
        let mut nodes = Vec::new();
        for entry in manifest.entries() {
            match entry.kind {
                kore_workspace_sync::EntryKind::Directory => {
                    if let Err(error) = self.create_directory(&mut staging, entry).await {
                        self.abort(staging).await;
                        return Err(error);
                    }
                    let handle = staging
                        .directories
                        .last()
                        .map(|(_, handle)| *handle)
                        .ok_or(FsError::Conflict)?;
                    let stable_object = self.engine.stat(self.importer, handle)?.stable_object;
                    nodes
                        .push(LiveWorkspaceNode {
                            path: entry.path.clone(),
                            stable_object,
                            kind: entry.kind,
                            len: 0,
                            digest: entry.digest,
                            executable: false,
                        })
                        .map_err(|_| FsError::OutOfObjects)?;
                }
                kore_workspace_sync::EntryKind::File => {
                    let file = match self.begin_file(&mut staging, entry).await {
                        Ok(file) => file,
                        Err(error) => {
                            self.abort(staging).await;
                            return Err(error);
                        }
                    };
                    if nodes
                        .push(LiveWorkspaceNode {
                            path: entry.path.clone(),
                            stable_object: file.stable_object,
                            kind: entry.kind,
                            len: entry.content_len,
                            digest: entry.digest,
                            executable: entry.flags.executable(),
                        })
                        .is_err()
                    {
                        let _ = self.engine.close(self.importer, file.handle);
                        self.abort(staging).await;
                        return Err(FsError::OutOfObjects);
                    }
                    if let Err(error) = self.engine.close(self.importer, file.handle) {
                        self.abort(staging).await;
                        return Err(error);
                    }
                }
            }
        }
        match self.commit(staging, &snapshot).await {
            Ok(commit) => Ok(LiveWorkspaceImport {
                commit,
                root_stable_object,
                nodes,
            }),
            Err(failure) => {
                let error = failure.error;
                self.abort(failure.staging).await;
                Err(error)
            }
        }
    }

    fn catalog_committed_live(
        &self,
        manifest: &kore_workspace_sync::Manifest,
        commit: WorkspaceCommit,
    ) -> FsResult<LiveWorkspaceImport> {
        let root = self
            .engine
            .open_grant(self.broker, commit.broker_root, Rights::HOME)?;
        let mut opened = Vec::<
            (kore_workspace_sync::CanonicalPath, Handle),
            { kore_workspace_sync::bounds::MAX_DIRECTORIES },
        >::new();
        let mut nodes = Vec::new();
        for entry in manifest.entries() {
            let parent = match entry.path.parent() {
                None => root.handle,
                Some(parent) => opened
                    .iter()
                    .find_map(|(path, handle)| (path == &parent).then_some(*handle))
                    .ok_or(FsError::Conflict)?,
            };
            let object = self.engine.open_at(
                self.broker,
                parent,
                &Name::from_bytes(entry.path.file_name())?,
                Rights::HOME,
            )?;
            if (entry.kind == kore_workspace_sync::EntryKind::File
                && object.kind != ObjectKind::File)
                || (entry.kind == kore_workspace_sync::EntryKind::Directory
                    && object.kind != ObjectKind::Directory)
            {
                let _ = self.engine.close(self.broker, object.handle);
                return Err(FsError::Conflict);
            }
            nodes
                .push(LiveWorkspaceNode {
                    path: entry.path.clone(),
                    stable_object: object.stable_object,
                    kind: entry.kind,
                    len: entry.content_len,
                    digest: entry.digest,
                    executable: entry.flags.executable(),
                })
                .map_err(|_| FsError::OutOfObjects)?;
            if entry.kind == kore_workspace_sync::EntryKind::Directory {
                opened
                    .push((entry.path.clone(), object.handle))
                    .map_err(|_| FsError::OutOfHandles)?;
            } else {
                self.engine.close(self.broker, object.handle)?;
            }
        }
        for (_, handle) in opened.into_iter().rev() {
            self.engine.close(self.broker, handle)?;
        }
        self.engine.close(self.broker, root.handle)?;
        let root_stable_object = root.stable_object;
        Ok(LiveWorkspaceImport {
            commit,
            root_stable_object,
            nodes,
        })
    }

    async fn current_header(&self) -> FsResult<Option<SnapshotHeader>> {
        let mut handles = ImportHandles::new(self.engine, self.importer);
        let root = handles.keep(self.engine.open_root(self.importer, Rights::ALL)?)?;
        let current = match self.engine.open_at(
            self.importer,
            root.handle,
            &Name::new(CURRENT_NAME)?,
            Rights::ALL,
        ) {
            Ok(current) => handles.keep(current)?,
            Err(FsError::NotFound) => return Ok(None),
            Err(error) => return Err(error),
        };
        let marker = handles.keep(
            self.engine
                .open_at(
                    self.importer,
                    current.handle,
                    &Name::new(SNAPSHOT_NAME)?,
                    Rights::READ,
                )
                .map_err(|error| {
                    if error == FsError::NotFound {
                        FsError::Conflict
                    } else {
                        error
                    }
                })?,
        )?;
        let mut bytes = [0u8; core::mem::size_of::<SnapshotHeader>()];
        let read = self
            .engine
            .read(self.importer, marker.handle, 0, &mut bytes)
            .await?;
        let stat = self.engine.stat(self.importer, marker.handle)?;
        let recorded = SnapshotHeader::read_from_bytes(&bytes).map_err(|_| FsError::Conflict)?;
        if read != bytes.len() || stat.len != bytes.len() as u64 || !recorded.shape_is_valid() {
            return Err(FsError::Conflict);
        }
        Ok(Some(recorded))
    }

    /// Record a verified live snapshot over the same native data root. The
    /// caller must have matched every topology record before using this seam.
    /// Staged replacement keeps the previous marker intact on provider failure.
    pub(crate) async fn record_live_snapshot(
        &self,
        manifest: &kore_workspace_sync::Manifest,
        data_root: crate::StableObjectId,
    ) -> FsResult<()> {
        manifest
            .verify(kore_workspace_sync::Policy::default())
            .map_err(|_| FsError::Conflict)?;
        let mut handles = ImportHandles::new(self.engine, self.importer);
        let root = handles.keep(self.engine.open_root(self.importer, Rights::ALL)?)?;
        let current = handles.keep(self.engine.open_at(
            self.importer,
            root.handle,
            &Name::new(CURRENT_NAME)?,
            Rights::ALL,
        )?)?;
        let data = handles.keep(self.engine.open_at(
            self.importer,
            current.handle,
            &Name::new(DATA_NAME)?,
            Rights::ALL,
        )?)?;
        if data.stable_object != data_root {
            return Err(FsError::Conflict);
        }
        let marker = handles.keep(self.engine.open_at(
            self.importer,
            current.handle,
            &Name::new(SNAPSHOT_NAME)?,
            Rights::ALL,
        )?)?;
        let mut bytes = [0u8; core::mem::size_of::<SnapshotHeader>()];
        if self
            .engine
            .read(self.importer, marker.handle, 0, &mut bytes)
            .await?
            != bytes.len()
            || marker.len != bytes.len() as u64
        {
            return Err(FsError::Conflict);
        }
        let previous = SnapshotHeader::read_from_bytes(&bytes).map_err(|_| FsError::Conflict)?;
        if !previous.shape_is_valid() {
            return Err(FsError::Conflict);
        }
        let next = manifest.header();
        if previous == *next {
            return Ok(());
        }
        self.engine
            .replace_host_file(
                self.importer,
                marker.handle,
                marker.revision,
                next.as_bytes(),
                async || Ok(()),
            )
            .await?;
        Ok(())
    }

    fn grant_current(&self, snapshot: [u8; 32]) -> FsResult<WorkspaceCommit> {
        let root = self.engine.open_root(self.importer, Rights::ALL)?;
        let current = match self.engine.open_at(
            self.importer,
            root.handle,
            &Name::new(CURRENT_NAME)?,
            Rights::ALL,
        ) {
            Ok(current) => current,
            Err(error) => {
                let _ = self.engine.close(self.importer, root.handle);
                return Err(error);
            }
        };
        let data = match self.engine.open_at(
            self.importer,
            current.handle,
            &Name::new(DATA_NAME)?,
            Rights::ALL,
        ) {
            Ok(data) => data,
            Err(error) => {
                let _ = self.engine.close(self.importer, current.handle);
                let _ = self.engine.close(self.importer, root.handle);
                return Err(error);
            }
        };
        let mut existing = None;
        for info in self.engine.grants(self.broker)? {
            if info.kind != AuthorityKind::Grant || info.rights != Rights::HOME {
                continue;
            }
            let Ok(opened) = self
                .engine
                .open_grant(self.broker, info.grant, Rights::HOME)
            else {
                continue;
            };
            let same_object = opened.stable_object == data.stable_object;
            let _ = self.engine.close(self.broker, opened.handle);
            if same_object {
                existing = Some(info.grant);
                break;
            }
        }
        let broker_root = existing.map_or_else(
            || {
                self.engine
                    .grant(self.importer, data.handle, self.broker, Rights::HOME, None)
            },
            Ok,
        );
        let _ = self.engine.close(self.importer, data.handle);
        let _ = self.engine.close(self.importer, current.handle);
        let _ = self.engine.close(self.importer, root.handle);
        broker_root.map(|broker_root| WorkspaceCommit {
            snapshot,
            broker_root,
        })
    }

    async fn existing(&self, snapshot: &SnapshotHeader) -> FsResult<Option<WorkspaceCommit>> {
        let Some(recorded) = self.current_header().await? else {
            return Ok(None);
        };
        if recorded != *snapshot {
            return Ok(None);
        }
        self.grant_current(snapshot.snapshot_id).map(Some)
    }

    /// Recover the committed workspace identity while minting fresh broker
    /// authority for this boot. Runtime grants are never read from disk.
    pub async fn recover_current(&self) -> FsResult<Option<WorkspaceCommit>> {
        let Some(recorded) = self.current_header().await? else {
            return Ok(None);
        };
        self.grant_current(recorded.snapshot_id).map(Some)
    }

    async fn detach_named(&self, root: Handle, name: &Name) -> FsResult<()> {
        match self.engine.open_at(self.importer, root, name, Rights::ALL) {
            Ok(opened) => {
                self.engine.close(self.importer, opened.handle)?;
                self.engine.detach_edge(self.importer, root, name).await
            }
            Err(FsError::NotFound) => Ok(()),
            Err(error) => Err(error),
        }
    }

    async fn reclaim_partial(
        &self,
        root: Handle,
        container: Option<Handle>,
        data: Option<Handle>,
        marker: Option<Handle>,
    ) {
        if let Some(marker) = marker {
            let _ = self.engine.close(self.importer, marker);
        }
        if let Some(data) = data {
            let _ = self.engine.close(self.importer, data);
        }
        if let Some(container) = container {
            let _ = self.engine.close(self.importer, container);
        }
        if let Ok(name) = Name::new(STAGING_NAME) {
            let _ = self.engine.detach_edge(self.importer, root, &name).await;
        }
        let _ = self.engine.close(self.importer, root);
    }
}

pub struct WorkspaceStaging {
    root: Handle,
    container: Handle,
    data: Handle,
    directories: Vec<(kore_workspace_sync::CanonicalPath, Handle), MAX_DIRECTORIES>,
    published: bool,
}

pub struct WorkspaceFile {
    handle: Handle,
    revision: Revision,
    stable_object: crate::StableObjectId,
}

impl<P: StorageProvider> ImportStore for WorkspaceStore<'_, P> {
    type Error = FsError;
    type Staging = WorkspaceStaging;
    type File = WorkspaceFile;
    type Commit = WorkspaceCommit;

    async fn begin(
        &mut self,
        snapshot: &SnapshotHeader,
    ) -> Result<BeginOutcome<Self::Staging, Self::Commit>, Self::Error> {
        if let Some(commit) = self.existing(snapshot).await? {
            return Ok(BeginOutcome::AlreadyCommitted(commit));
        }
        let root = self.engine.open_root(self.importer, Rights::ALL)?;
        let staging_name = Name::new(STAGING_NAME)?;
        if let Err(error) = self.detach_named(root.handle, &staging_name).await {
            let _ = self.engine.close(self.importer, root.handle);
            return Err(error);
        }
        let container = match self
            .engine
            .create(
                self.importer,
                root.handle,
                &staging_name,
                ObjectKind::Directory,
            )
            .await
        {
            Ok(container) => container,
            Err(error) => {
                self.reclaim_partial(root.handle, None, None, None).await;
                return Err(error);
            }
        };
        let data = match self
            .engine
            .create(
                self.importer,
                container.handle,
                &Name::new(DATA_NAME)?,
                ObjectKind::Directory,
            )
            .await
        {
            Ok(data) => data,
            Err(error) => {
                self.reclaim_partial(root.handle, Some(container.handle), None, None)
                    .await;
                return Err(error);
            }
        };
        let marker = match self
            .engine
            .create(
                self.importer,
                container.handle,
                &Name::new(SNAPSHOT_NAME)?,
                ObjectKind::File,
            )
            .await
        {
            Ok(marker) => marker,
            Err(error) => {
                self.reclaim_partial(root.handle, Some(container.handle), Some(data.handle), None)
                    .await;
                return Err(error);
            }
        };
        let marker_revision = match self
            .engine
            .write_at(
                self.importer,
                marker.handle,
                0,
                snapshot.as_bytes(),
                marker.revision,
            )
            .await
        {
            Ok(revision) => revision,
            Err(error) => {
                self.reclaim_partial(
                    root.handle,
                    Some(container.handle),
                    Some(data.handle),
                    Some(marker.handle),
                )
                .await;
                return Err(error);
            }
        };
        let marker_stat = match self.engine.stat(self.importer, marker.handle) {
            Ok(stat) => stat,
            Err(error) => {
                self.reclaim_partial(
                    root.handle,
                    Some(container.handle),
                    Some(data.handle),
                    Some(marker.handle),
                )
                .await;
                return Err(error);
            }
        };
        if marker_stat.revision != marker_revision
            || marker_stat.len != snapshot.as_bytes().len() as u64
        {
            self.reclaim_partial(
                root.handle,
                Some(container.handle),
                Some(data.handle),
                Some(marker.handle),
            )
            .await;
            return Err(FsError::Conflict);
        }
        if let Err(error) = self.engine.close(self.importer, marker.handle) {
            self.reclaim_partial(root.handle, Some(container.handle), Some(data.handle), None)
                .await;
            return Err(error);
        }
        Ok(BeginOutcome::Staging(WorkspaceStaging {
            root: root.handle,
            container: container.handle,
            data: data.handle,
            directories: Vec::new(),
            published: false,
        }))
    }

    async fn create_directory(
        &mut self,
        staging: &mut Self::Staging,
        entry: &Entry,
    ) -> Result<(), Self::Error> {
        let parent = parent_handle(staging, entry)?;
        let directory = self
            .engine
            .create(
                self.importer,
                parent,
                &Name::from_bytes(entry.path.file_name())?,
                ObjectKind::Directory,
            )
            .await?;
        staging
            .directories
            .push((entry.path.clone(), directory.handle))
            .map_err(|_| FsError::OutOfHandles)
    }

    async fn begin_file(
        &mut self,
        staging: &mut Self::Staging,
        entry: &Entry,
    ) -> Result<Self::File, Self::Error> {
        let file = self
            .engine
            .create(
                self.importer,
                parent_handle(staging, entry)?,
                &Name::from_bytes(entry.path.file_name())?,
                ObjectKind::File,
            )
            .await?;
        Ok(WorkspaceFile {
            handle: file.handle,
            revision: file.revision,
            stable_object: file.stable_object,
        })
    }

    async fn write_file(
        &mut self,
        _staging: &mut Self::Staging,
        file: &mut Self::File,
        offset: u32,
        bytes: &[u8],
    ) -> Result<(), Self::Error> {
        file.revision = self
            .engine
            .write_at(
                self.importer,
                file.handle,
                u64::from(offset),
                bytes,
                file.revision,
            )
            .await?;
        Ok(())
    }

    async fn finish_file(
        &mut self,
        _staging: &mut Self::Staging,
        file: Self::File,
        entry: &Entry,
    ) -> Result<(), Self::Error> {
        let stat = self.engine.stat(self.importer, file.handle)?;
        if stat.len != u64::from(entry.content_len) || stat.revision != file.revision {
            return Err(FsError::Conflict);
        }
        self.engine.close(self.importer, file.handle)
    }

    async fn commit(
        &mut self,
        mut staging: Self::Staging,
        snapshot: &SnapshotHeader,
    ) -> Result<Self::Commit, CommitFailure<Self::Error, Self::Staging>> {
        while let Some((_, handle)) = staging.directories.pop() {
            if let Err(error) = self.engine.close(self.importer, handle) {
                return Err(CommitFailure { error, staging });
            }
        }
        let previous = match self.recover_current().await {
            Ok(previous) => previous,
            Err(error) => return Err(CommitFailure { error, staging }),
        };
        let current_name = Name::new(CURRENT_NAME).expect("static valid name");
        if let Err(error) = self.detach_named(staging.root, &current_name).await {
            return Err(CommitFailure { error, staging });
        }
        if let Err(error) = self.engine.rename(
            self.importer,
            staging.root,
            &Name::new(STAGING_NAME).expect("static valid name"),
            &current_name,
        ) {
            return Err(CommitFailure { error, staging });
        }
        staging.published = true;
        // This is the old data root's exact native grant, resolved before
        // replacement. Retire it and its descendants before reusing the
        // publication name; a name alone never selects authority to revoke.
        if let Some(previous) = previous {
            if let Err(error) = self
                .engine
                .revoke(self.importer, previous.broker_root)
                .await
            {
                return Err(CommitFailure { error, staging });
            }
        }
        let broker_root = match self.engine.grant(
            self.importer,
            staging.data,
            self.broker,
            Rights::HOME,
            Some(Name::new("imported-workspace").expect("static valid name")),
        ) {
            Ok(grant) => grant,
            Err(error) => return Err(CommitFailure { error, staging }),
        };
        for handle in [staging.data, staging.container, staging.root] {
            if let Err(error) = self.engine.close(self.importer, handle) {
                let _ = self.engine.revoke(self.importer, broker_root).await;
                return Err(CommitFailure { error, staging });
            }
        }
        Ok(WorkspaceCommit {
            snapshot: snapshot.snapshot_id,
            broker_root,
        })
    }

    async fn abort(&mut self, mut staging: Self::Staging) {
        while let Some((_, handle)) = staging.directories.pop() {
            let _ = self.engine.close(self.importer, handle);
        }
        let _ = self.engine.close(self.importer, staging.data);
        let _ = self.engine.close(self.importer, staging.container);
        let name = Name::new(if staging.published {
            CURRENT_NAME
        } else {
            STAGING_NAME
        });
        if let Ok(name) = name {
            let _ = self
                .engine
                .detach_edge(self.importer, staging.root, &name)
                .await;
        }
        let _ = self.engine.close(self.importer, staging.root);
    }
}

fn parent_handle(staging: &WorkspaceStaging, entry: &Entry) -> FsResult<Handle> {
    match entry.path.parent() {
        None => Ok(staging.data),
        Some(parent) => staging
            .directories
            .iter()
            .find_map(|(path, handle)| (path == &parent).then_some(*handle))
            .ok_or(FsError::NotFound),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RamEngine, RamProvider};
    use kore_workspace_sync::{
        CanonicalPath, ImportPhase, ImportTransaction, ManifestBuilder, Policy,
    };

    #[test]
    fn commit_is_the_first_broker_route_and_retry_reuses_the_same_tree() {
        let engine = RamEngine::new(RamProvider::default());
        let importer = SessionId::new(0);
        let broker = SessionId::new(1);
        crate::block_on(engine.attach_session(importer)).unwrap();
        crate::block_on(engine.attach_session(broker)).unwrap();
        let bytes = b"workspace";
        let mut builder = ManifestBuilder::new([0; 32], Policy::default());
        builder
            .push(Entry::directory(
                CanonicalPath::new(b"src").unwrap(),
                [1; 16],
            ))
            .unwrap();
        builder
            .push(
                Entry::file(
                    CanonicalPath::new(b"src/main.rs").unwrap(),
                    true,
                    bytes.len() as u32,
                    *blake3::hash(bytes).as_bytes(),
                    [2; 16],
                )
                .unwrap(),
            )
            .unwrap();
        let manifest = builder.finish().unwrap();
        let mut tx = crate::block_on(ImportTransaction::begin(
            WorkspaceStore::new(&engine, importer, broker),
            manifest.clone(),
            Policy::default(),
        ))
        .unwrap();
        assert!(
            engine
                .grants(broker)
                .unwrap()
                .iter()
                .all(|grant| grant.kind == crate::AuthorityKind::PrivateRoot)
        );
        crate::block_on(tx.begin_next_file()).unwrap();
        crate::block_on(tx.write(0, bytes)).unwrap();
        crate::block_on(tx.finish_file()).unwrap();
        assert_eq!(tx.phase(), ImportPhase::Ready);
        let committed = crate::block_on(tx.commit()).unwrap();
        let broker_root = engine
            .open_grant(broker, committed.broker_root, Rights::READ_ONLY)
            .unwrap();
        let src = engine
            .open_at(
                broker,
                broker_root.handle,
                &Name::new("src").unwrap(),
                Rights::READ_ONLY,
            )
            .unwrap();
        let file = engine
            .open_at(
                broker,
                src.handle,
                &Name::new("main.rs").unwrap(),
                Rights::READ_ONLY,
            )
            .unwrap();
        let mut observed = [0; 9];
        assert_eq!(
            crate::block_on(engine.read(broker, file.handle, 0, &mut observed)).unwrap(),
            9
        );
        assert_eq!(&observed, bytes);

        let authority_edges = engine.authority_edge_count();
        let retry = crate::block_on(ImportTransaction::begin(
            WorkspaceStore::new(&engine, importer, broker),
            manifest,
            Policy::default(),
        ))
        .unwrap();
        assert_eq!(retry.phase(), ImportPhase::Committed);
        assert_eq!(engine.object_count(), 7, "retry did not duplicate the tree");
        assert_eq!(
            engine.authority_edge_count(),
            authority_edges,
            "retry reused the live broker grant"
        );
    }

    #[test]
    fn repeated_live_attach_recatalogs_the_committed_tree_without_copying_bytes() {
        let engine = RamEngine::new(RamProvider::default());
        let importer = SessionId::new(0);
        let broker = SessionId::new(1);
        crate::block_on(engine.attach_session(importer)).unwrap();
        crate::block_on(engine.attach_session(broker)).unwrap();
        let content = b"bytes remain on the host";
        let mut builder = ManifestBuilder::new([0; 32], Policy::default());
        builder
            .push(Entry::directory(
                CanonicalPath::new(b"src").unwrap(),
                [1; 16],
            ))
            .unwrap();
        builder
            .push(
                Entry::file(
                    CanonicalPath::new(b"src/lib.rs").unwrap(),
                    false,
                    content.len() as u32,
                    *blake3::hash(content).as_bytes(),
                    [2; 16],
                )
                .unwrap(),
            )
            .unwrap();
        let manifest = builder.finish().unwrap();
        let first = crate::block_on(
            WorkspaceStore::new(&engine, importer, broker)
                .import_live_topology(&manifest, Policy::default()),
        )
        .unwrap();
        let objects = engine.object_count();
        let second = crate::block_on(
            WorkspaceStore::new(&engine, importer, broker)
                .import_live_topology(&manifest, Policy::default()),
        )
        .unwrap();
        assert_eq!(second.root_stable_object, first.root_stable_object);
        assert_eq!(second.nodes, first.nodes);
        assert_eq!(second.nodes.len(), 2);
        assert_eq!(engine.object_count(), objects);
        let file = second
            .nodes
            .iter()
            .find(|node| node.kind == kore_workspace_sync::EntryKind::File)
            .unwrap();
        assert_eq!(file.len, content.len() as u32);
    }

    #[test]
    fn failed_digest_reclaims_private_staging_and_publishes_nothing() {
        let engine = RamEngine::new(RamProvider::default());
        let importer = SessionId::new(0);
        let broker = SessionId::new(1);
        crate::block_on(engine.attach_session(importer)).unwrap();
        crate::block_on(engine.attach_session(broker)).unwrap();
        let mut builder = ManifestBuilder::new([0; 32], Policy::default());
        builder
            .push(
                Entry::file(
                    CanonicalPath::new(b"bad").unwrap(),
                    false,
                    3,
                    *blake3::hash(b"one").as_bytes(),
                    [0; 16],
                )
                .unwrap(),
            )
            .unwrap();
        let mut tx = crate::block_on(ImportTransaction::begin(
            WorkspaceStore::new(&engine, importer, broker),
            builder.finish().unwrap(),
            Policy::default(),
        ))
        .unwrap();
        crate::block_on(tx.begin_next_file()).unwrap();
        crate::block_on(tx.write(0, b"two")).unwrap();
        assert!(matches!(
            crate::block_on(tx.finish_file()),
            Err(kore_workspace_sync::ImportError::DigestMismatch)
        ));
        assert!(
            engine
                .grants(broker)
                .unwrap()
                .iter()
                .all(|grant| grant.kind == crate::AuthorityKind::PrivateRoot)
        );
        let root = engine.open_root(importer, Rights::ALL).unwrap();
        assert!(matches!(
            engine.open_at(
                importer,
                root.handle,
                &Name::new(STAGING_NAME).unwrap(),
                Rights::READ
            ),
            Err(FsError::NotFound)
        ));
    }

    #[test]
    fn broker_derives_two_disjoint_agent_views() {
        let engine = RamEngine::new(RamProvider::default());
        let importer = SessionId::new(0);
        let broker = SessionId::new(1);
        let agent_a = SessionId::new(2);
        let agent_b = SessionId::new(3);
        for session in [importer, broker, agent_a, agent_b] {
            crate::block_on(engine.attach_session(session)).unwrap();
        }
        let mut builder = ManifestBuilder::new([0; 32], Policy::default());
        for (index, name) in ["agent-a", "agent-b"].into_iter().enumerate() {
            builder
                .push(Entry::directory(
                    CanonicalPath::new(name.as_bytes()).unwrap(),
                    [index as u8; 16],
                ))
                .unwrap();
        }
        let manifest = builder.finish().unwrap();
        let mut tx = crate::block_on(ImportTransaction::begin(
            WorkspaceStore::new(&engine, importer, broker),
            manifest,
            Policy::default(),
        ))
        .unwrap();
        let committed = crate::block_on(tx.commit()).unwrap();
        let broker_root = engine
            .open_grant(broker, committed.broker_root, Rights::HOME)
            .unwrap();
        let left = engine
            .open_at(
                broker,
                broker_root.handle,
                &Name::new("agent-a").unwrap(),
                Rights::READ.union(Rights::SHARE),
            )
            .unwrap();
        let right = engine
            .open_at(
                broker,
                broker_root.handle,
                &Name::new("agent-b").unwrap(),
                Rights::READ.union(Rights::SHARE),
            )
            .unwrap();
        let left_grant = engine
            .grant(broker, left.handle, agent_a, Rights::READ_ONLY, None)
            .unwrap();
        let right_grant = engine
            .grant(broker, right.handle, agent_b, Rights::READ_ONLY, None)
            .unwrap();
        let left_root = engine
            .open_grant(agent_a, left_grant, Rights::READ_ONLY)
            .unwrap();
        let right_root = engine
            .open_grant(agent_b, right_grant, Rights::READ_ONLY)
            .unwrap();
        assert_eq!(
            engine.open_at(
                agent_a,
                left_root.handle,
                &Name::new("agent-b").unwrap(),
                Rights::READ_ONLY,
            ),
            Err(FsError::NotFound)
        );
        assert_eq!(
            engine.open_at(
                agent_b,
                right_root.handle,
                &Name::new("agent-a").unwrap(),
                Rights::READ_ONLY,
            ),
            Err(FsError::NotFound)
        );
        for hidden in [STAGING_NAME, CURRENT_NAME, SNAPSHOT_NAME] {
            assert_eq!(
                engine.open_at(
                    agent_a,
                    left_root.handle,
                    &Name::new(hidden).unwrap(),
                    Rights::READ_ONLY,
                ),
                Err(FsError::NotFound)
            );
        }
    }

    #[test]
    fn committed_workspace_recovers_with_fresh_authority() {
        use kore_driver::block::{BlockChannel, CompletionMode, memory::MemoryBlockDevice};

        type Device = MemoryBlockDevice<327680, 8192>;
        type Volume = crate::persistence::PersistentFs<Device, 3, 8>;

        std::thread::Builder::new()
            .name("workspace-recovery".into())
            // This combined debug-build fixture holds import, publication and
            // restart futures with by-value block/engine owners. Their debug
            // temporaries need more stack than the single-phase mount tests.
            .stack_size(128 * 1024 * 1024)
            .spawn(|| {
                let channel =
                    BlockChannel::new(Device::new(512, 16).unwrap(), CompletionMode::Poll).unwrap();
                let data_start =
                    crate::persistence::bootstrap_data_start(channel.geometry().block_size)
                        .unwrap();
                let provider =
                    crate::provider::BlockProvider::new_with_data_start(channel, data_start)
                        .unwrap();
                let importer = SessionId::new(0);
                let broker = SessionId::new(1);
                let mut volume = crate::block_on(Volume::format(provider, importer)).unwrap();
                crate::block_on(volume.engine().attach_session(broker)).unwrap();
                let bytes = b"persistent workspace";
                let mut builder = ManifestBuilder::new([0; 32], Policy::default());
                builder
                    .push(
                        Entry::file(
                            CanonicalPath::new(b"state").unwrap(),
                            false,
                            bytes.len() as u32,
                            *blake3::hash(bytes).as_bytes(),
                            [4; 16],
                        )
                        .unwrap(),
                    )
                    .unwrap();
                let manifest = builder.finish().unwrap();
                let mut tx = crate::block_on(ImportTransaction::begin(
                    WorkspaceStore::new(volume.engine(), importer, broker),
                    manifest.clone(),
                    Policy::default(),
                ))
                .unwrap();
                crate::block_on(tx.begin_next_file()).unwrap();
                crate::block_on(tx.write(0, bytes)).unwrap();
                crate::block_on(tx.finish_file()).unwrap();
                crate::block_on(tx.commit()).unwrap();
                crate::block_on(volume.commit(importer)).unwrap();

                let recovered_importer = SessionId::new(7);
                let recovered_broker = SessionId::new(6);
                let volume = crate::block_on(volume.restart(recovered_importer)).unwrap();
                assert_eq!(
                    volume.engine().authority_edge_count(),
                    1,
                    "no boot-local grants or handles were serialized"
                );
                crate::block_on(volume.engine().attach_session(recovered_broker)).unwrap();
                let commit = crate::block_on(
                    WorkspaceStore::new(volume.engine(), recovered_importer, recovered_broker)
                        .recover_current(),
                )
                .unwrap()
                .expect("committed marker recovers");
                assert_eq!(commit.snapshot, manifest.snapshot_id());
                let root = volume
                    .engine()
                    .open_grant(recovered_broker, commit.broker_root, Rights::READ_ONLY)
                    .unwrap();
                let file = volume
                    .engine()
                    .open_at(
                        recovered_broker,
                        root.handle,
                        &Name::new("state").unwrap(),
                        Rights::READ_ONLY,
                    )
                    .unwrap();
                let mut observed = [0; 20];
                let count = crate::block_on(volume.engine().read(
                    recovered_broker,
                    file.handle,
                    0,
                    &mut observed,
                ))
                .unwrap();
                assert_eq!(&observed[..count], bytes);
            })
            .unwrap()
            .join()
            .unwrap();
    }
    #[test]
    fn live_snapshot_marker_failure_preserves_previous_identity_and_retires_handles() {
        let engine = RamEngine::new(RamProvider::default());
        let importer = SessionId::new(0);
        let broker = SessionId::new(1);
        crate::block_on(engine.attach_session(importer)).unwrap();
        crate::block_on(engine.attach_session(broker)).unwrap();
        let build = |parent, source| {
            let mut builder = ManifestBuilder::new(parent, Policy::default());
            builder
                .push(
                    Entry::file(
                        CanonicalPath::new(b"file").unwrap(),
                        false,
                        1,
                        *blake3::hash(b"x").as_bytes(),
                        source,
                    )
                    .unwrap(),
                )
                .unwrap();
            builder.finish().unwrap()
        };
        let original = build([0; 32], [1; 16]);
        let next = build(original.snapshot_id(), [2; 16]);
        let mut store = WorkspaceStore::new(&engine, importer, broker);
        let imported =
            crate::block_on(store.import_live_topology(&original, Policy::default())).unwrap();
        assert_eq!(
            crate::block_on(store.record_live_snapshot(&next, crate::StableObjectId::INVALID)),
            Err(FsError::Conflict)
        );
        for _ in 0..(crate::bounds::MAX_HANDLES_PER_SESSION * 2) {
            engine
                .provider()
                .inject(crate::provider::Fault::Write(crate::ProviderError::Media));
            assert_eq!(
                crate::block_on(store.record_live_snapshot(&next, imported.root_stable_object)),
                Err(FsError::ProviderFailure(crate::ProviderError::Media))
            );
            assert_eq!(
                crate::block_on(store.recover_current())
                    .unwrap()
                    .unwrap()
                    .snapshot,
                original.snapshot_id()
            );
            engine
                .provider()
                .inject(crate::provider::Fault::Read(crate::ProviderError::Media));
            assert_eq!(
                crate::block_on(store.recover_current()),
                Err(FsError::ProviderFailure(crate::ProviderError::Media))
            );
        }
        crate::block_on(store.record_live_snapshot(&next, imported.root_stable_object)).unwrap();
        let recovered = crate::block_on(store.recover_current()).unwrap().unwrap();
        assert_eq!(recovered.snapshot, next.snapshot_id());
        assert_eq!(recovered.broker_root, imported.commit.broker_root);
    }
    #[test]
    fn repeated_live_replacement_retires_previous_grants_with_bounded_cost() {
        let engine = RamEngine::new(RamProvider::default());
        let importer = SessionId::new(0);
        let broker = SessionId::new(1);
        crate::block_on(engine.attach_session(importer)).unwrap();
        crate::block_on(engine.attach_session(broker)).unwrap();
        let mut store = WorkspaceStore::new(&engine, importer, broker);
        let mut parent = [0; 32];
        let mut previous = None;
        let mut baseline = None;
        for generation in 1..=12 {
            let mut builder = ManifestBuilder::new(parent, Policy::default());
            builder
                .push(
                    Entry::file(
                        CanonicalPath::new(b"file").unwrap(),
                        false,
                        1,
                        *blake3::hash(b"x").as_bytes(),
                        [generation; 16],
                    )
                    .unwrap(),
                )
                .unwrap();
            let manifest = builder.finish().unwrap();
            let imported =
                crate::block_on(store.import_live_topology(&manifest, Policy::default())).unwrap();
            if let Some(grant) = previous {
                assert!(engine.open_grant(broker, grant, Rights::READ_ONLY).is_err());
            }
            let cost = (
                engine.object_count(),
                engine.authority_edge_count(),
                engine.live_handles(importer),
                engine.live_handles(broker),
                engine.provider().chunks_in_use(),
            );
            if let Some(baseline) = baseline {
                assert_eq!(cost, baseline);
            } else {
                baseline = Some(cost);
                std::println!(
                    "live replacement cost (objects, authority edges, importer handles, broker handles, RAM chunks): {cost:?}"
                );
            }
            assert_eq!(cost.2, 0);
            assert_eq!(cost.3, 0);
            previous = Some(imported.commit.broker_root);
            parent = manifest.snapshot_id();
        }
    }
}
