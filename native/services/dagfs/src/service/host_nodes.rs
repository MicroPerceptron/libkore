//! Service-private host topology. Stable IDs are metadata, never authority;
//! every client lookup revalidates its native session/handle first.

use super::*;
use heapless::Vec;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct HostNodeRecord {
    pub stable: crate::StableObjectId,
    pub path: kore_workspace_sync::CanonicalPath,
    pub kind: ObjectKind,
    pub len: u64,
    pub digest: [u8; 32],
    pub executable: bool,
    pub materialized: bool,
    pub revoked: bool,
}

impl<P: StorageProvider> FsService<P> {
    pub(crate) fn replace_host_nodes(
        &mut self,
        root: crate::StableObjectId,
        nodes: &[HostNodeRecord],
    ) -> FsResult<()> {
        if root == crate::StableObjectId::INVALID {
            return Err(FsError::Range);
        }
        Self::validate_host_nodes(nodes)?;
        if nodes.iter().any(|node| node.stable == root || node.revoked) {
            return Err(FsError::Conflict);
        }
        let mut replacement = Vec::<HostNodeRecord, { crate::bounds::MAX_OBJECTS }>::new();
        for old in self.host_nodes.iter() {
            if self.engine.contains_stable_object(old.stable)
                && nodes.iter().all(|node| node.stable != old.stable)
            {
                let mut retired = old.clone();
                retired.revoked = true;
                replacement
                    .push(retired)
                    .map_err(|_| FsError::OutOfObjects)?;
            }
        }
        for node in nodes {
            replacement
                .push(node.clone())
                .map_err(|_| FsError::OutOfObjects)?;
        }
        self.host_root = Some(root);
        self.host_nodes = replacement;
        Ok(())
    }

    pub(crate) fn install_host_node(&mut self, node: HostNodeRecord) -> FsResult<()> {
        Self::validate_host_nodes(core::slice::from_ref(&node))?;
        if self.host_root == Some(node.stable)
            || self.host_nodes.iter().any(|entry| {
                !entry.revoked && entry.path == node.path && entry.stable != node.stable
            })
        {
            return Err(FsError::Conflict);
        }
        if let Some(existing) = self
            .host_nodes
            .iter()
            .find(|entry| entry.stable == node.stable)
        {
            return if existing == &node {
                Ok(())
            } else {
                Err(FsError::Conflict)
            };
        }
        self.host_nodes
            .push(node)
            .map_err(|_| FsError::OutOfObjects)
    }

    /// Validate the complete candidate before publishing any part of it.
    fn validate_host_nodes(nodes: &[HostNodeRecord]) -> FsResult<()> {
        for (index, node) in nodes.iter().enumerate() {
            if node.stable == crate::StableObjectId::INVALID
                || node.len > kore_workspace_sync::bounds::MAX_FILE_BYTES
                || (node.kind == ObjectKind::Directory && (node.len != 0 || node.executable))
            {
                return Err(FsError::Range);
            }
            if nodes[..index].iter().any(|prior| {
                prior.stable == node.stable
                    || (!prior.revoked && !node.revoked && prior.path == node.path)
            }) {
                return Err(FsError::Conflict);
            }
        }
        Ok(())
    }

    pub(super) fn host_len(&self, stable: crate::StableObjectId) -> FsResult<Option<u64>> {
        let Some(entry) = self.host_nodes.iter().find(|entry| entry.stable == stable) else {
            return Ok(None);
        };
        if entry.revoked {
            return Err(FsError::AccessLost);
        }
        Ok((entry.kind == ObjectKind::File && !entry.materialized).then_some(entry.len))
    }

    pub(crate) fn host_node_for_handle(
        &self,
        session: SessionId,
        handle: Handle,
    ) -> FsResult<Option<HostNodeRecord>> {
        if self.host_root.is_none() {
            // No host namespace exists; let the ordinary engine validate
            // native-only writes without imposing host materialization rights.
            return Ok(None);
        }
        let stable = self.engine.stable_object_for_handle(session, handle)?;
        Ok(self
            .host_nodes
            .iter()
            .find(|entry| entry.stable == stable)
            .cloned())
    }

    pub(crate) fn host_node_at(
        &self,
        parent: &HostNodeRecord,
        name: &Name,
    ) -> Option<HostNodeRecord> {
        let path = Self::host_child_path(parent, name).ok()?;
        self.host_nodes
            .iter()
            .find(|entry| !entry.revoked && entry.path == path)
            .cloned()
    }

    pub(crate) fn host_path_under_handle(
        &self,
        session: SessionId,
        parent: Handle,
        name: &Name,
    ) -> FsResult<Option<kore_workspace_sync::CanonicalPath>> {
        if self.host_root.is_none() {
            return Ok(None);
        }
        let stable = self.engine.stable_object_for_handle(session, parent)?;
        if self.host_root == Some(stable) {
            return kore_workspace_sync::CanonicalPath::new(name.as_bytes())
                .map(Some)
                .map_err(|_| FsError::InvalidName);
        }
        let Some(parent) = self
            .host_nodes
            .iter()
            .find(|entry| entry.stable == stable && entry.kind == ObjectKind::Directory)
        else {
            return Ok(None);
        };
        if parent.revoked {
            return Err(FsError::AccessLost);
        }
        Self::host_child_path(parent, name).map(Some)
    }

    pub(crate) fn host_node_by_path(
        &self,
        path: &kore_workspace_sync::CanonicalPath,
    ) -> Option<HostNodeRecord> {
        self.host_nodes
            .iter()
            .find(|entry| !entry.revoked && &entry.path == path)
            .cloned()
    }

    pub(crate) fn host_file_bindings(
        &self,
    ) -> FsResult<
        Vec<
            (kore_workspace_sync::CanonicalPath, [u8; 32]),
            { kore_workspace_sync::bounds::MAX_ENTRIES },
        >,
    > {
        let mut bindings = Vec::new();
        for entry in self
            .host_nodes
            .iter()
            .filter(|entry| entry.kind == ObjectKind::File && !entry.revoked)
        {
            bindings
                .push((
                    entry.path.clone(),
                    crate::host_workspace::stable_live_object(entry.stable),
                ))
                .map_err(|_| FsError::OutOfObjects)?;
        }
        Ok(bindings)
    }

    pub(crate) fn host_child_path(
        parent: &HostNodeRecord,
        name: &Name,
    ) -> FsResult<kore_workspace_sync::CanonicalPath> {
        let mut bytes = [0u8; kore_workspace_sync::bounds::MAX_PATH_BYTES];
        let parent_bytes = parent.path.as_bytes();
        let needed = parent_bytes
            .len()
            .checked_add(1)
            .and_then(|len| len.checked_add(name.as_bytes().len()))
            .ok_or(FsError::Range)?;
        let out = bytes.get_mut(..needed).ok_or(FsError::Range)?;
        out[..parent_bytes.len()].copy_from_slice(parent_bytes);
        out[parent_bytes.len()] = b'/';
        out[parent_bytes.len() + 1..].copy_from_slice(name.as_bytes());
        kore_workspace_sync::CanonicalPath::new(out).map_err(|_| FsError::InvalidName)
    }

    pub(crate) fn update_host_node(&mut self, node: HostNodeRecord) -> FsResult<()> {
        Self::validate_host_nodes(core::slice::from_ref(&node))?;
        if self.host_nodes.iter().any(|entry| {
            entry.stable != node.stable
                && !entry.revoked
                && !node.revoked
                && entry.path == node.path
        }) {
            return Err(FsError::Conflict);
        }
        let existing = self
            .host_nodes
            .iter_mut()
            .find(|entry| entry.stable == node.stable)
            .ok_or(FsError::NotFound)?;
        *existing = node;
        Ok(())
    }

    pub(crate) fn retire_host_node(&mut self, stable: crate::StableObjectId) {
        for entry in self
            .host_nodes
            .iter_mut()
            .filter(|entry| entry.stable == stable)
        {
            entry.revoked = true;
        }
    }

    pub(crate) fn host_delete_changes(
        &self,
        root: &kore_workspace_sync::CanonicalPath,
    ) -> FsResult<Vec<kore_workspace_sync::Change, { kore_workspace_sync::bounds::MAX_CHANGES }>>
    {
        let mut changes = Vec::new();
        for entry in self
            .host_nodes
            .iter()
            .filter(|entry| !entry.revoked && entry.path.is_within(root))
        {
            changes
                .push(kore_workspace_sync::Change {
                    kind: kore_workspace_sync::ChangeKind::Delete,
                    path: entry.path.clone(),
                    target: None,
                    expected: entry.digest,
                    replacement: [0; 32],
                    replacement_len: 0,
                    executable: false,
                })
                .map_err(|_| FsError::Range)?;
        }
        Ok(changes)
    }

    pub(crate) fn retire_host_subtree(&mut self, root: &kore_workspace_sync::CanonicalPath) {
        for entry in self
            .host_nodes
            .iter_mut()
            .filter(|entry| entry.path.is_within(root))
        {
            entry.revoked = true;
        }
    }

    pub(crate) fn rename_host_subtree(
        &mut self,
        from: &kore_workspace_sync::CanonicalPath,
        to: &kore_workspace_sync::CanonicalPath,
    ) -> FsResult<()> {
        self.host_nodes = self.prepare_host_rename(from, to)?;
        Ok(())
    }

    pub(crate) fn prepare_host_rename(
        &self,
        from: &kore_workspace_sync::CanonicalPath,
        to: &kore_workspace_sync::CanonicalPath,
    ) -> FsResult<Vec<HostNodeRecord, { crate::bounds::MAX_OBJECTS }>> {
        // Prefix replacement can exceed either the path byte or depth bound,
        // or collide with another live path. Preflight the whole subtree so
        // refusal cannot publish only its first few names.
        let mut replacement = self.host_nodes.clone();
        for entry in replacement
            .iter_mut()
            .filter(|entry| !entry.revoked && entry.path.is_within(from))
        {
            let suffix = &entry.path.as_bytes()[from.as_bytes().len()..];
            let mut bytes = [0u8; kore_workspace_sync::bounds::MAX_PATH_BYTES];
            let len = to
                .as_bytes()
                .len()
                .checked_add(suffix.len())
                .ok_or(FsError::Range)?;
            let out = bytes.get_mut(..len).ok_or(FsError::Range)?;
            out[..to.as_bytes().len()].copy_from_slice(to.as_bytes());
            out[to.as_bytes().len()..].copy_from_slice(suffix);
            entry.path =
                kore_workspace_sync::CanonicalPath::new(out).map_err(|_| FsError::InvalidName)?;
        }
        Self::validate_host_nodes(replacement.as_slice())?;
        Ok(replacement)
    }

    pub(crate) fn prepare_host_create(
        &self,
        node: HostNodeRecord,
    ) -> FsResult<Vec<HostNodeRecord, { crate::bounds::MAX_OBJECTS }>> {
        if self.host_root == Some(node.stable) {
            return Err(FsError::Conflict);
        }
        let mut candidate = self.host_nodes.clone();
        candidate.push(node).map_err(|_| FsError::OutOfObjects)?;
        Self::validate_host_nodes(&candidate)?;
        Ok(candidate)
    }

    pub(crate) fn publish_host_nodes(
        &mut self,
        replacement: Vec<HostNodeRecord, { crate::bounds::MAX_OBJECTS }>,
    ) {
        // Only the workspace owner holds this prevalidated candidate, and it
        // retains an exclusive service borrow until the native commit ends.
        self.host_nodes = replacement;
    }

    fn host_handle(&self, session: SessionId, handle: Handle) -> FsResult<bool> {
        if self.host_root.is_none() {
            return Ok(false);
        }
        let stable = self.engine.stable_object_for_handle(session, handle)?;
        if let Some(node) = self.host_nodes.iter().find(|node| node.stable == stable) {
            if node.revoked {
                return Err(FsError::AccessLost);
            }
            return Ok(true);
        }
        Ok(self.host_root == Some(stable))
    }

    pub(crate) fn request_touches_host(
        &self,
        session: SessionId,
        request: &FsRequest,
    ) -> FsResult<bool> {
        let handles = match *request {
            FsRequest::Stat { handle }
            | FsRequest::List { handle, .. }
            | FsRequest::Read { handle, .. }
            | FsRequest::WriteAt { handle, .. }
            | FsRequest::Append { handle, .. }
            | FsRequest::Resize { handle, .. }
            | FsRequest::LendVersion { handle, .. }
            | FsRequest::LendExecutable { handle, .. }
            | FsRequest::DerivedInfo { handle, .. }
            | FsRequest::Search { handle, .. }
            | FsRequest::Grant { handle, .. }
            | FsRequest::GrantScoped { handle, .. }
            | FsRequest::GrantView { handle, .. }
            | FsRequest::BindView { handle, .. }
            | FsRequest::Sync { handle } => [Some(handle), None],
            FsRequest::OpenAt { parent, .. }
            | FsRequest::Create { parent, .. }
            | FsRequest::Rename { parent, .. }
            | FsRequest::Unlink { parent, .. }
            | FsRequest::DetachEdge { parent, .. } => [Some(parent), None],
            FsRequest::Relocate {
                src_parent,
                dst_parent,
                ..
            } => [Some(src_parent), Some(dst_parent)],
            FsRequest::Link { parent, child, .. } => [Some(parent), Some(child)],
            _ => [None, None],
        };
        let mut host = false;
        for handle in handles.into_iter().flatten() {
            host |= self.host_handle(session, handle)?;
        }
        Ok(host)
    }

    /// Keep topology placeholders from silently becoming native empty files.
    /// Live reads are intercepted by the mount before ordinary dispatch.
    /// Mutations and immutable snapshots need the COW/writeback composition
    /// before they can be admitted on a host route.
    pub(super) fn guard_host_request(
        &self,
        session: SessionId,
        request: &FsRequest,
    ) -> FsResult<()> {
        if !self.request_touches_host(session, request)? {
            return Ok(());
        }
        match *request {
            FsRequest::Create { .. }
            | FsRequest::WriteAt { .. }
            | FsRequest::Append { .. }
            | FsRequest::Resize { .. }
            | FsRequest::Rename { .. }
            | FsRequest::Relocate { .. }
            | FsRequest::Link { .. }
            | FsRequest::Unlink { .. }
            | FsRequest::DetachEdge { .. } => Err(FsError::AccessDenied),
            FsRequest::Read { handle, .. } => {
                let node = self
                    .host_node_for_handle(session, handle)?
                    .ok_or(FsError::IsDirectory)?;
                if node.materialized {
                    Ok(())
                } else {
                    Err(FsError::ProviderFailure(crate::ProviderError::Lost))
                }
            }
            FsRequest::LendVersion { handle, .. }
            | FsRequest::LendExecutable { handle, .. }
            | FsRequest::DerivedInfo { handle, .. } => {
                match self.host_node_for_handle(session, handle)? {
                    Some(node) if node.materialized => Ok(()),
                    _ => Err(FsError::Unsupported),
                }
            }
            _ => Ok(()),
        }
    }

    pub(crate) fn host_root_id(&self) -> Option<crate::StableObjectId> {
        self.host_root
    }

    pub(crate) fn host_topology_matches(&self, manifest: &kore_workspace_sync::Manifest) -> bool {
        self.host_root.is_some()
            && self.host_nodes.iter().filter(|node| !node.revoked).count()
                == manifest.entries().len()
            && manifest.entries().iter().all(|entry| {
                self.host_nodes.iter().any(|node| {
                    !node.revoked
                        && node.path == entry.path
                        && node.kind
                            == match entry.kind {
                                kore_workspace_sync::EntryKind::File => ObjectKind::File,
                                kore_workspace_sync::EntryKind::Directory => ObjectKind::Directory,
                            }
                        && node.len == u64::from(entry.content_len)
                        && node.digest == entry.digest
                        && node.executable == entry.flags.executable()
                })
            })
    }

    pub(crate) fn install_imported_workspace(&mut self, grant: GrantId) -> FsResult<()> {
        let admin = SessionId::new(crate::service::contract::ADMIN_VIA);
        let system = SessionId::new(0);
        if self.imported_workspace_view == Some(grant) {
            if let Some(source) = self.imported_workspace_source {
                let opened = self.engine.open_grant(system, source, Rights::HOME)?;
                let _ = self.engine.close(system, opened.handle);
                return Ok(());
            }
        }
        // Init's retained exec context resolves named grants through system
        // session zero. Delegate from the importer-published admin grant,
        // never by opening the hidden storage path through the system root.
        // Revoking a workspace generation then revokes these descendants too.
        let workspace = self.engine.open_grant(admin, grant, Rights::HOME)?;
        let source = self.engine.grant(
            admin,
            workspace.handle,
            system,
            Rights::HOME,
            Some(Name::new("imported-workspace")?),
        );
        let _ = self.engine.close(admin, workspace.handle);
        self.imported_workspace_source = Some(source?);
        self.imported_workspace_view = Some(grant);
        Ok(())
    }

    /// A revoked host object cannot fall through to the empty local staging
    /// file. Close an unsuccessful open so it does not consume a handle slot.
    pub(super) fn opened(
        &self,
        session: SessionId,
        object: crate::OpenedObject,
    ) -> FsResult<FsResponse> {
        let len = match self.host_len(object.stable_object) {
            Ok(len) => len.unwrap_or(object.len),
            Err(error) => {
                let _ = self.engine.close(session, object.handle);
                return Err(error);
            }
        };
        Ok(FsResponse::Opened {
            handle: object.handle,
            stable_object: object.stable_object,
            kind: object.kind,
            subtype: object.subtype,
            len,
            revision: object.revision,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ObjectSubtype, RamEngine, RamProvider, StableObjectId};
    use kore_workspace_sync::CanonicalPath;

    fn record(stable: StableObjectId, path: &[u8], kind: ObjectKind) -> HostNodeRecord {
        HostNodeRecord {
            stable,
            path: CanonicalPath::new(path).unwrap(),
            kind,
            len: if kind == ObjectKind::File { 9000 } else { 0 },
            digest: [7; 32],
            executable: false,
            materialized: false,
            revoked: false,
        }
    }

    fn fixture() -> (
        RamService,
        SessionId,
        crate::OpenedObject,
        crate::OpenedObject,
    ) {
        let mut fs = RamService::new(RamEngine::new(RamProvider::default()));
        let session = SessionId::new(0);
        assert!(fs.ensure_session(session));
        let root = fs.engine.open_root(session, Rights::ALL).unwrap();
        let file = block_on(fs.engine.create(
            session,
            root.handle,
            &Name::new("host").unwrap(),
            ObjectKind::File,
        ))
        .unwrap();
        fs.replace_host_nodes(
            root.stable_object,
            &[record(file.stable_object, b"host", ObjectKind::File)],
        )
        .unwrap();
        (fs, session, root, file)
    }

    #[test]
    fn host_nodes_overlay_length_without_materializing_or_changing_subtype() {
        let (mut fs, session, root, file) = fixture();
        assert_eq!(fs.engine.stat(session, file.handle).unwrap().len, 0);
        assert!(matches!(
            fs.dispatch(
                session,
                &FsRequest::Stat {
                    handle: file.handle
                },
                None
            ),
            FsResponse::Stat {
                len: 9000,
                subtype: ObjectSubtype::PLAIN,
                ..
            }
        ));
        assert!(matches!(
            fs.dispatch(
                session,
                &FsRequest::OpenAt {
                    parent: root.handle,
                    name: Name::new("host").unwrap(),
                    rights: Rights::READ
                },
                None
            ),
            FsResponse::Opened {
                len: 9000,
                subtype: ObjectSubtype::PLAIN,
                ..
            }
        ));
        // A bare frontend cannot accidentally answer EOF for a host file.
        assert_eq!(
            fs.dispatch(
                session,
                &FsRequest::Read {
                    handle: file.handle,
                    offset: 0,
                    window_offset: 0,
                    length: 1
                },
                None
            ),
            FsResponse::Refused(FsError::ProviderFailure(crate::ProviderError::Lost))
        );
    }

    #[test]
    fn host_nodes_revocation_survives_replacement_and_does_not_leak_open_handles() {
        let (mut fs, session, root, file) = fixture();
        fs.replace_host_nodes(root.stable_object, &[]).unwrap();
        assert!(
            fs.host_node_for_handle(session, file.handle)
                .unwrap()
                .unwrap()
                .revoked
        );
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
        for _ in 0..(crate::bounds::MAX_HANDLES_PER_SESSION * 2) {
            assert_eq!(
                fs.dispatch(
                    session,
                    &FsRequest::OpenAt {
                        parent: root.handle,
                        name: Name::new("host").unwrap(),
                        rights: Rights::READ
                    },
                    None
                ),
                FsResponse::Refused(FsError::AccessLost)
            );
        }
        assert!(fs.engine.open_root(session, Rights::READ).is_ok());
    }

    #[test]
    fn host_nodes_reject_duplicate_or_invalid_replacement_atomically() {
        let (mut fs, _, root, file) = fixture();
        let original = fs.host_nodes.clone();
        let node = record(file.stable_object, b"host", ObjectKind::File);
        assert_eq!(
            fs.replace_host_nodes(root.stable_object, &[node.clone(), node.clone()]),
            Err(FsError::Conflict)
        );
        let mut duplicate_path = node.clone();
        duplicate_path.stable = StableObjectId::new(file.stable_object.get() + 100).unwrap();
        assert_eq!(
            fs.replace_host_nodes(root.stable_object, &[node.clone(), duplicate_path]),
            Err(FsError::Conflict)
        );
        let mut invalid = node;
        invalid.len = kore_workspace_sync::bounds::MAX_FILE_BYTES + 1;
        assert_eq!(
            fs.replace_host_nodes(root.stable_object, &[invalid]),
            Err(FsError::Range)
        );
        assert_eq!(fs.host_nodes, original);
        assert_eq!(fs.host_root, Some(root.stable_object));
    }

    #[test]
    fn host_nodes_subtree_rename_preflights_depth_and_collisions() {
        let (mut fs, _, _, _) = fixture();
        fs.host_nodes.clear();
        fs.install_host_node(record(
            StableObjectId::new(30).unwrap(),
            b"a",
            ObjectKind::Directory,
        ))
        .unwrap();
        fs.install_host_node(record(
            StableObjectId::new(31).unwrap(),
            b"a/b",
            ObjectKind::File,
        ))
        .unwrap();
        let original = fs.host_nodes.clone();
        assert_eq!(
            fs.rename_host_subtree(
                &CanonicalPath::new(b"a").unwrap(),
                &CanonicalPath::new(b"1/2/3/4/5/6/7/8").unwrap()
            ),
            Err(FsError::InvalidName)
        );
        assert_eq!(fs.host_nodes, original);
        fs.install_host_node(record(
            StableObjectId::new(32).unwrap(),
            b"c",
            ObjectKind::Directory,
        ))
        .unwrap();
        let original = fs.host_nodes.clone();
        assert_eq!(
            fs.rename_host_subtree(
                &CanonicalPath::new(b"a").unwrap(),
                &CanonicalPath::new(b"c").unwrap()
            ),
            Err(FsError::Conflict)
        );
        assert_eq!(fs.host_nodes, original);
        fs.rename_host_subtree(
            &CanonicalPath::new(b"a").unwrap(),
            &CanonicalPath::new(b"d").unwrap(),
        )
        .unwrap();
        assert!(
            fs.host_node_by_path(&CanonicalPath::new(b"d/b").unwrap())
                .is_some()
        );
    }

    #[test]
    fn host_nodes_classification_preserves_native_write_only_authority() {
        let (mut fs, session, root, _) = fixture();
        let native = block_on(fs.engine.create(
            session,
            root.handle,
            &Name::new("native").unwrap(),
            ObjectKind::File,
        ))
        .unwrap();
        fs.engine.close(session, native.handle).unwrap();
        let write_only = fs
            .engine
            .open_at(
                session,
                root.handle,
                &Name::new("native").unwrap(),
                Rights::WRITE,
            )
            .unwrap();
        assert_eq!(
            fs.engine.stat(session, write_only.handle),
            Err(FsError::AccessDenied)
        );
        assert_eq!(
            fs.host_node_for_handle(session, write_only.handle),
            Ok(None)
        );
        fs.dispatch(
            session,
            &FsRequest::AttachWindow { slot: 0, bytes: 1 },
            None,
        );
        let mut bytes = [42];
        let mut window = DataWindow::new(&mut bytes);
        assert!(matches!(
            fs.dispatch(
                session,
                &FsRequest::WriteAt {
                    handle: write_only.handle,
                    offset: 0,
                    window_offset: 0,
                    length: 1,
                    expected: crate::ANY_REVISION
                },
                Some(&mut window)
            ),
            FsResponse::Committed { .. }
        ));
    }
}
