//! Trusted filesystem/importer client for the dedicated HostLink workspace face.
//!
//! This is the only guest component that can translate a kore filesystem
//! object into an opaque host-source request. Agent callers continue to hold
//! ordinary filesystem handles; no source correlation, route, bulk id, or
//! transport object crosses this boundary.

use core::mem::size_of;

use heapless::Vec;
use kore_hostlink::{BulkId, BulkRefV1, RequestId, WORKSPACE_REPLY};
use kore_rt::transport::Transport;
use kore_svc_hostlink::{ClientEvent, RingHostLink};
use kore_workspace_sync::{
    ATTACH_MAGIC, AttachRequestV1, AttachResponseV1, BlobStoredV1, CanonicalPath, Change,
    ChangeKind, ChangeSet, ChangeSetBuilder, DELTA_OP_REQUEST_MAGIC, DELTA_POLL_MAGIC,
    DeltaAssembler, DeltaOpRequestV1, DeltaPollV1, DeltaResponseV1, ENTRY_REQUEST_MAGIC, EntryKind,
    FORMAT_VERSION, LiveFile, LiveWorkspace, MAX_WRITE_TRANSACTION_BYTES, Manifest,
    ManifestBuilder, ManifestEntryRequestV1, ManifestEntryResponseV1, Policy, RANGE_REQUEST_MAGIC,
    RANGE_RESPONSE_MAGIC, RangeProof, RangeRequestV1, RangeResponseV1, TopologyState,
    TransactionResultV1, WORKSPACE_BLOB_CONTENT, WORKSPACE_TRANSACTION_CONTENT, WorkspaceErrorV1,
    WorkspaceMode, WorkspaceStatus, WriteTransaction,
    bounds::{MAX_ENTRIES, RANGE_BYTES},
    decode_delta_op, encode_write_transaction,
};
use zerocopy::byteorder::little_endian::{U16, U32, U64};
use zerocopy::{FromBytes, IntoBytes};

const RPC_POLL_LIMIT: usize = 4096;
const RANGE_BULK_MAX: usize = RANGE_BYTES + size_of::<RangeResponseV1>();
const DELTA_PAGE_MAX: usize = 1024;
const OBJECT_DOMAIN: &[u8] = b"kore.fs.host-workspace.object.v1\0";

#[allow(async_fn_in_trait)]
pub trait WorkspaceChannel {
    type Error;

    async fn send_workspace(
        &mut self,
        request: RequestId,
        payload: &[u8],
    ) -> Result<(), Self::Error>;
    async fn poll_workspace(&mut self) -> Result<Option<ClientEvent>, Self::Error>;
    async fn upload_workspace(
        &mut self,
        request: RequestId,
        bulk_id: BulkId,
        content_kind: u16,
        bytes: &[u8],
    ) -> Result<(), Self::Error>;
}

impl<C: WorkspaceChannel + ?Sized> WorkspaceChannel for &mut C {
    type Error = C::Error;

    async fn send_workspace(
        &mut self,
        request: RequestId,
        payload: &[u8],
    ) -> Result<(), Self::Error> {
        (**self).send_workspace(request, payload).await
    }

    async fn poll_workspace(&mut self) -> Result<Option<ClientEvent>, Self::Error> {
        (**self).poll_workspace().await
    }

    async fn upload_workspace(
        &mut self,
        request: RequestId,
        bulk_id: BulkId,
        content_kind: u16,
        bytes: &[u8],
    ) -> Result<(), Self::Error> {
        (**self)
            .upload_workspace(request, bulk_id, content_kind, bytes)
            .await
    }
}

impl<T: Transport> WorkspaceChannel for RingHostLink<'_, '_, T> {
    type Error = kore_svc_hostlink::ClientError;

    async fn send_workspace(
        &mut self,
        request: RequestId,
        payload: &[u8],
    ) -> Result<(), Self::Error> {
        RingHostLink::send_workspace(self, request, payload).await
    }

    async fn poll_workspace(&mut self) -> Result<Option<ClientEvent>, Self::Error> {
        self.poll().await
    }

    async fn upload_workspace(
        &mut self,
        request: RequestId,
        bulk_id: BulkId,
        content_kind: u16,
        bytes: &[u8],
    ) -> Result<(), Self::Error> {
        let reference = BulkRefV1 {
            bulk_id,
            total_len: U32::new(bytes.len() as u32),
            chunk_count: U16::new(
                bytes
                    .len()
                    .div_ceil(kore_svc_hostlink::FOUNDED_BULK_WIRE_CHUNK) as u16,
            ),
            content_kind: U16::new(content_kind),
            digest_blake3: *blake3::hash(bytes).as_bytes(),
        };
        let mut upload = self.begin_bulk(request, reference).await?;
        for fragment in bytes.chunks(kore_svc_hostlink::CLIENT_CHUNK_MAX) {
            self.send_bulk_chunk(&mut upload, fragment).await?;
        }
        self.finish_bulk(&mut upload).await
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum HostWorkspaceError<E> {
    Transport(E),
    Unavailable,
    Stale,
    Conflict,
    Refused,
    Protocol,
    Backpressure,
    Integrity,
    Bounds,
    Revoked,
}

pub struct HostWorkspace<C: WorkspaceChannel> {
    channel: C,
    policy: Policy,
    topology: TopologyState,
    live: LiveWorkspace,
    write_grant: Option<[u8; 16]>,
    next_request: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WritebackReceipt {
    pub transaction: [u8; 32],
    pub snapshot: [u8; 32],
}

impl<C: WorkspaceChannel> HostWorkspace<C> {
    /// Borrow the supervisor's channel slot across the handshake. Cancellation
    /// drops only a borrow, so the affine transport remains available to retry.
    pub(crate) async fn attach_in_slot(
        slot: &mut Option<C>,
        policy: Policy,
    ) -> Result<Self, HostWorkspaceError<C::Error>> {
        let attached = HostWorkspace::<&mut C>::attach(
            slot.as_mut().ok_or(HostWorkspaceError::Unavailable)?,
            policy,
        )
        .await?;
        let HostWorkspace {
            channel: _,
            policy,
            topology,
            live,
            write_grant,
            next_request,
        } = attached;
        Ok(Self {
            channel: slot
                .take()
                .expect("borrowed attachment retains its channel"),
            policy,
            topology,
            live,
            write_grant,
            next_request,
        })
    }

    /// Attach while returning the still-owned channel on failure. Supervisors
    /// use this form across a HostLink reconnect: a transient handshake race
    /// must not consume the only affine founded pair.
    pub async fn attach_recoverable(
        mut channel: C,
        policy: Policy,
    ) -> Result<Self, (C, HostWorkspaceError<C::Error>)> {
        match HostWorkspace::<&mut C>::attach(&mut channel, policy).await {
            Ok(attached) => {
                let HostWorkspace {
                    channel: _,
                    policy,
                    topology,
                    live,
                    write_grant,
                    next_request,
                } = attached;
                Ok(Self {
                    channel,
                    policy,
                    topology,
                    live,
                    write_grant,
                    next_request,
                })
            }
            Err(error) => Err((channel, error)),
        }
    }

    pub async fn attach(
        mut channel: C,
        policy: Policy,
    ) -> Result<Self, HostWorkspaceError<C::Error>> {
        let request = request_id(1).ok_or(HostWorkspaceError::Bounds)?;
        let attach = AttachRequestV1 {
            magic: ATTACH_MAGIC,
            version: U16::new(FORMAT_VERSION),
            mode: WorkspaceMode::Live as u8,
            flags: 0,
            policy_digest: policy.digest(),
            known_snapshot: [0; 32],
        };
        channel
            .send_workspace(request, attach.as_bytes())
            .await
            .map_err(HostWorkspaceError::Transport)?;
        let response = wait_control(&mut channel, request).await?;
        let response = AttachResponseV1::read_from_bytes(response.as_slice())
            .map_err(|_| HostWorkspaceError::Protocol)?;
        map_status(
            response
                .validate()
                .map_err(|_| HostWorkspaceError::Protocol)?,
        )?;
        if response.policy_digest != policy.digest() {
            return Err(HostWorkspaceError::Integrity);
        }

        let mut builder = ManifestBuilder::new(response.parent_snapshot, policy);
        let mut counter = 2u64;
        for index in 0..response.entry_count.get() {
            let id = request_id(counter).ok_or(HostWorkspaceError::Bounds)?;
            counter = counter.checked_add(1).ok_or(HostWorkspaceError::Bounds)?;
            let entry_request = ManifestEntryRequestV1 {
                magic: ENTRY_REQUEST_MAGIC,
                version: U16::new(FORMAT_VERSION),
                index: U16::new(index),
                snapshot: response.snapshot,
            };
            channel
                .send_workspace(id, entry_request.as_bytes())
                .await
                .map_err(HostWorkspaceError::Transport)?;
            let bytes = wait_control(&mut channel, id).await?;
            let entry = ManifestEntryResponseV1::read_from_bytes(bytes.as_slice())
                .map_err(|_| HostWorkspaceError::Protocol)?;
            let (status, entry) = entry.validate().map_err(|_| HostWorkspaceError::Protocol)?;
            map_status(status)?;
            builder
                .push(entry.ok_or(HostWorkspaceError::Protocol)?)
                .map_err(|_| HostWorkspaceError::Integrity)?;
        }
        let manifest = builder
            .finish()
            .map_err(|_| HostWorkspaceError::Integrity)?;
        let header = manifest.header();
        if header.snapshot_id != response.snapshot
            || header.root_digest != response.root_digest
            || header.policy_digest != response.policy_digest
            || header.entry_count != response.entry_count
            || header.directory_count != response.directory_count
            || header.total_bytes != response.total_bytes
        {
            return Err(HostWorkspaceError::Integrity);
        }
        let bindings = bindings(&manifest)?;
        let live = LiveWorkspace::attach(&manifest, policy, bindings.as_slice())
            .map_err(|_| HostWorkspaceError::Integrity)?;
        let topology = TopologyState::new(manifest, policy, response.delta_sequence.get())
            .map_err(|_| HostWorkspaceError::Integrity)?;
        let write_grant = (response.write_operations != 0).then_some(response.write_grant);
        Ok(Self {
            channel,
            policy,
            topology,
            live,
            write_grant,
            next_request: counter,
        })
    }

    #[must_use]
    pub const fn writeback_enabled(&self) -> bool {
        self.write_grant.is_some()
    }

    /// Commit one normalized filesystem mutation through the exact standing
    /// grant announced at attachment. File bytes are staged as a separate
    /// digest-addressed bulk object; the transaction metadata is idempotent
    /// and preconditioned on the currently attached snapshot.
    pub async fn writeback(
        &mut self,
        change: Change,
        content: Option<&[u8]>,
    ) -> Result<WritebackReceipt, HostWorkspaceError<C::Error>> {
        let mut builder = ChangeSetBuilder::new(self.topology.manifest().snapshot_id());
        builder
            .push(change)
            .map_err(|_| HostWorkspaceError::Bounds)?;
        self.writeback_changes(
            builder.finish().map_err(|_| HostWorkspaceError::Bounds)?,
            content,
        )
        .await
    }

    /// One bounded transaction may remove a whole subtree through explicit
    /// per-path deletions. Every descendant remains visible to the standing
    /// grant's exclusions and operation budget on the host.
    pub async fn writeback_changes(
        &mut self,
        changes: ChangeSet,
        content: Option<&[u8]>,
    ) -> Result<WritebackReceipt, HostWorkspaceError<C::Error>> {
        let grant = self.write_grant.ok_or(HostWorkspaceError::Refused)?;
        if changes.base != self.topology.manifest().snapshot_id()
            || changes.verify().is_err()
            || changes.changes().is_empty()
        {
            return Err(HostWorkspaceError::Protocol);
        }
        let file_changes = changes
            .changes()
            .iter()
            .filter(|change| matches!(change.kind, ChangeKind::CreateFile | ChangeKind::ModifyFile))
            .count();
        if (file_changes != 0) != content.is_some()
            || (file_changes != 0 && changes.changes().len() != 1)
        {
            return Err(HostWorkspaceError::Protocol);
        }
        if let Some(bytes) = content {
            let change = &changes.changes()[0];
            if bytes.len() != change.replacement_len as usize
                || blake3::hash(bytes).as_bytes() != &change.replacement
            {
                return Err(HostWorkspaceError::Integrity);
            }
        }
        let transaction = transaction_id(changes.base, changes.id, self.next_request);

        if let Some(bytes) = content.filter(|bytes| !bytes.is_empty()) {
            let request = self.request_id()?;
            let bulk_id = workspace_bulk_id(request, WORKSPACE_BLOB_CONTENT, bytes);
            self.channel
                .upload_workspace(request, bulk_id, WORKSPACE_BLOB_CONTENT, bytes)
                .await
                .map_err(HostWorkspaceError::Transport)?;
            let response = match wait_upload_terminal(&mut self.channel, request, bulk_id).await? {
                Some(response) => response,
                None => wait_control(&mut self.channel, request).await?,
            };
            let stored = BlobStoredV1::read_from_bytes(response.as_slice())
                .map_err(|_| HostWorkspaceError::Protocol)?;
            map_status(
                stored
                    .validate()
                    .map_err(|_| HostWorkspaceError::Protocol)?,
            )?;
            if stored.bulk_id != bulk_id.0
                || stored.digest != changes.changes()[0].replacement
                || stored.length.get() != bytes.len() as u32
            {
                return Err(HostWorkspaceError::Integrity);
            }
        }

        let metadata = WriteTransaction {
            id: transaction,
            grant,
            changes,
        };
        let mut encoded = alloc::boxed::Box::new([0u8; MAX_WRITE_TRANSACTION_BYTES]);
        let encoded_len = encode_write_transaction(&metadata, &mut encoded[..])
            .map_err(|_| HostWorkspaceError::Bounds)?;
        let request = self.request_id()?;
        let bulk_id = workspace_bulk_id(
            request,
            WORKSPACE_TRANSACTION_CONTENT,
            &encoded[..encoded_len],
        );
        self.channel
            .upload_workspace(
                request,
                bulk_id,
                WORKSPACE_TRANSACTION_CONTENT,
                &encoded[..encoded_len],
            )
            .await
            .map_err(HostWorkspaceError::Transport)?;
        let response = match wait_upload_terminal(&mut self.channel, request, bulk_id).await? {
            Some(response) => response,
            None => wait_control(&mut self.channel, request).await?,
        };
        let result = TransactionResultV1::read_from_bytes(response.as_slice())
            .map_err(|_| HostWorkspaceError::Protocol)?;
        map_status(
            result
                .validate()
                .map_err(|_| HostWorkspaceError::Protocol)?,
        )?;
        if result.transaction != transaction
            || result.changes != metadata.changes.id
            || result.applied.get() != metadata.changes.changes().len() as u16
        {
            return Err(HostWorkspaceError::Integrity);
        }
        Ok(WritebackReceipt {
            transaction,
            snapshot: result.snapshot,
        })
    }

    #[must_use]
    pub const fn manifest(&self) -> &Manifest {
        self.topology.manifest()
    }

    #[cfg(test)]
    pub(crate) fn channel_mut(&mut self) -> &mut C {
        &mut self.channel
    }

    pub fn into_channel(mut self) -> C {
        self.live.revoke();
        self.channel
    }

    pub async fn reattach(
        self,
        objects: &[(CanonicalPath, [u8; 32])],
    ) -> Result<Self, HostWorkspaceError<C::Error>> {
        let policy = self.policy;
        let channel = self.into_channel();
        let mut fresh = Self::attach(channel, policy).await?;
        fresh.bind_kore_objects(objects)?;
        Ok(fresh)
    }

    pub fn revoke(&mut self) {
        self.live.revoke();
    }

    /// Replace provisional manifest-derived identities with the stable kore
    /// object identities created by the filesystem transaction. Host source
    /// correlations remain inside `LiveWorkspace` and are not returned.
    pub fn bind_kore_objects(
        &mut self,
        objects: &[(CanonicalPath, [u8; 32])],
    ) -> Result<(), HostWorkspaceError<C::Error>> {
        let mut rebound = Vec::<LiveFile, MAX_ENTRIES>::new();
        for entry in self
            .topology
            .manifest()
            .entries()
            .iter()
            .filter(|entry| entry.kind == EntryKind::File)
        {
            let object = objects
                .iter()
                .find_map(|(path, object)| (path == &entry.path).then_some(*object))
                .ok_or(HostWorkspaceError::Integrity)?;
            rebound
                .push(
                    LiveFile::new(
                        object,
                        self.topology.manifest().snapshot_id(),
                        entry.source,
                        entry.digest,
                        entry.range_root,
                        entry.content_len,
                    )
                    .ok_or(HostWorkspaceError::Integrity)?,
                )
                .map_err(|_| HostWorkspaceError::Bounds)?;
        }
        self.live.revoke();
        self.live =
            LiveWorkspace::attach(self.topology.manifest(), self.policy, rebound.as_slice())
                .map_err(|_| HostWorkspaceError::Integrity)?;
        Ok(())
    }

    /// Read at most one 4 KiB proof range. A miss fetches and verifies the
    /// complete leaf; a hit performs no HostLink operation.
    pub async fn read(
        &mut self,
        object: [u8; 32],
        offset: u32,
        out: &mut [u8],
    ) -> Result<usize, HostWorkspaceError<C::Error>> {
        if out.is_empty() {
            return Ok(0);
        }
        let within = RANGE_BYTES - (offset as usize % RANGE_BYTES);
        let want = out.len().min(within).min(RANGE_BYTES);
        if let Some(count) = self
            .live
            .cached_read(object, offset, &mut out[..want])
            .map_err(map_live)?
        {
            return Ok(count);
        }
        let request = self.request_id()?;
        let key = self
            .live
            .begin_read(request_counter(request), object, offset, want as u16)
            .map_err(map_live)?;
        let range = RangeRequestV1 {
            magic: RANGE_REQUEST_MAGIC,
            version: U16::new(FORMAT_VERSION),
            flags: U16::ZERO,
            snapshot: key.snapshot,
            source: key.source,
            version_digest: key.version,
            range_root: key.range_root,
            file_len: U32::new(key.file_len),
            offset: U32::new(offset),
            length: U16::new(want as u16),
            retry: 0,
            reserved: 0,
        };
        self.channel
            .send_workspace(request, range.as_bytes())
            .await
            .map_err(HostWorkspaceError::Transport)?;
        let bytes = wait_bulk::<C, RANGE_BULK_MAX>(&mut self.channel, request).await?;
        if bytes.len() < size_of::<RangeResponseV1>() {
            return Err(HostWorkspaceError::Protocol);
        }
        let response = RangeResponseV1::read_from_bytes(&bytes[..size_of::<RangeResponseV1>()])
            .map_err(|_| HostWorkspaceError::Protocol)?;
        map_status(
            response
                .validate()
                .map_err(|_| HostWorkspaceError::Protocol)?,
        )?;
        if response.magic != RANGE_RESPONSE_MAGIC
            || response.snapshot != key.snapshot
            || response.source != key.source
            || response.version_digest != key.version
            || response.file_len.get() != key.file_len
            || response.range_index.get() != key.range_index
        {
            return Err(HostWorkspaceError::Integrity);
        }
        let proof = RangeProof::from_parts(
            response.file_len.get(),
            response.range_index.get(),
            response.range_count.get(),
            &response.siblings[..usize::from(response.proof_depth)],
        )
        .ok_or(HostWorkspaceError::Integrity)?;
        self.live
            .complete_read(
                request_counter(request),
                &proof,
                &bytes[size_of::<RangeResponseV1>()..],
            )
            .map_err(map_live)?;
        self.live
            .cached_read(object, offset, &mut out[..want])
            .map_err(map_live)?
            .ok_or(HostWorkspaceError::Integrity)
    }

    /// Apply at most one canonical host topology transaction. Watcher paths
    /// never enter this API; each operation is fetched as a bounded page and
    /// the complete delta identity is recomputed before namespace publication.
    pub async fn poll_delta(&mut self) -> Result<bool, HostWorkspaceError<C::Error>> {
        let request = self.request_id()?;
        let poll = DeltaPollV1 {
            magic: DELTA_POLL_MAGIC,
            version: U16::new(FORMAT_VERSION),
            flags: U16::ZERO,
            snapshot: self.topology.manifest().snapshot_id(),
            next_sequence: U64::new(self.topology.next_sequence()),
        };
        self.channel
            .send_workspace(request, poll.as_bytes())
            .await
            .map_err(HostWorkspaceError::Transport)?;
        let response = wait_control(&mut self.channel, request).await?;
        if let Ok(error) = WorkspaceErrorV1::read_from_bytes(response.as_slice()) {
            return match error.validate().map_err(|_| HostWorkspaceError::Protocol)? {
                WorkspaceStatus::Unavailable => Ok(false),
                status => map_status(status).map(|()| false),
            };
        }
        let response = DeltaResponseV1::read_from_bytes(response.as_slice())
            .map_err(|_| HostWorkspaceError::Protocol)?;
        map_status(
            response
                .validate()
                .map_err(|_| HostWorkspaceError::Protocol)?,
        )?;
        if response.parent != self.topology.manifest().snapshot_id()
            || response.sequence.get() != self.topology.next_sequence()
        {
            return Err(HostWorkspaceError::Stale);
        }
        let count = response.change_count.get();
        if count == 0 {
            return Err(HostWorkspaceError::Protocol);
        }
        let mut assembler = None;
        for index in 0..count {
            let id = self.request_id()?;
            let op = DeltaOpRequestV1 {
                magic: DELTA_OP_REQUEST_MAGIC,
                version: U16::new(FORMAT_VERSION),
                index: U16::new(index),
                sequence: response.sequence,
                delta_id: response.delta_id,
            };
            self.channel
                .send_workspace(id, op.as_bytes())
                .await
                .map_err(HostWorkspaceError::Transport)?;
            let bytes = wait_bulk::<C, DELTA_PAGE_MAX>(&mut self.channel, id).await?;
            let page =
                decode_delta_op(bytes.as_slice()).map_err(|_| HostWorkspaceError::Protocol)?;
            match assembler.as_mut() {
                None => {
                    assembler = Some(
                        DeltaAssembler::begin(page).map_err(|_| HostWorkspaceError::Protocol)?,
                    )
                }
                Some(assembler) => assembler
                    .push(page)
                    .map_err(|_| HostWorkspaceError::Protocol)?,
            }
        }
        let delta = assembler
            .ok_or(HostWorkspaceError::Protocol)?
            .finish()
            .map_err(|_| HostWorkspaceError::Integrity)?;
        if delta.id != response.delta_id || delta.next != response.next {
            return Err(HostWorkspaceError::Integrity);
        }
        self.topology
            .apply(&delta)
            .map_err(|_| HostWorkspaceError::Stale)?;
        self.live.revoke();
        let bindings = bindings(self.topology.manifest())?;
        self.live =
            LiveWorkspace::attach(self.topology.manifest(), self.policy, bindings.as_slice())
                .map_err(|_| HostWorkspaceError::Integrity)?;
        Ok(true)
    }

    fn request_id(&mut self) -> Result<RequestId, HostWorkspaceError<C::Error>> {
        let counter = self.next_request;
        self.next_request = counter.checked_add(1).ok_or(HostWorkspaceError::Bounds)?;
        request_id(counter).ok_or(HostWorkspaceError::Bounds)
    }
}

fn bindings<E>(manifest: &Manifest) -> Result<Vec<LiveFile, MAX_ENTRIES>, HostWorkspaceError<E>> {
    let mut bindings = Vec::new();
    for entry in manifest
        .entries()
        .iter()
        .filter(|entry| entry.kind == EntryKind::File)
    {
        let mut hash = blake3::Hasher::new();
        hash.update(OBJECT_DOMAIN);
        hash.update(&manifest.snapshot_id());
        hash.update(&(entry.path.as_bytes().len() as u16).to_le_bytes());
        hash.update(entry.path.as_bytes());
        hash.update(&entry.digest);
        let object = *hash.finalize().as_bytes();
        let binding = LiveFile::new(
            object,
            manifest.snapshot_id(),
            entry.source,
            entry.digest,
            entry.range_root,
            entry.content_len,
        )
        .ok_or(HostWorkspaceError::Integrity)?;
        bindings
            .push(binding)
            .map_err(|_| HostWorkspaceError::Bounds)?;
    }
    Ok(bindings)
}

async fn wait_control<C: WorkspaceChannel>(
    channel: &mut C,
    request: RequestId,
) -> Result<Vec<u8, 512>, HostWorkspaceError<C::Error>> {
    for _ in 0..RPC_POLL_LIMIT {
        match channel
            .poll_workspace()
            .await
            .map_err(HostWorkspaceError::Transport)?
        {
            None => yield_once().await,
            Some(ClientEvent::Control {
                kind,
                request: found,
                payload,
            }) if found == request && kind == WORKSPACE_REPLY => {
                return Vec::from_slice(payload.as_slice()).map_err(|_| HostWorkspaceError::Bounds);
            }
            Some(ClientEvent::Status(_)) => return Err(HostWorkspaceError::Unavailable),
            Some(_) => return Err(HostWorkspaceError::Protocol),
        }
    }
    Err(HostWorkspaceError::Backpressure)
}

/// Wait until the host has terminally accepted one guest-to-host bulk. The
/// reply travels on the control plane while the terminal bulk record may
/// arrive through the independent bulk plane, so retain one early workspace
/// reply instead of assuming cross-plane ordering.
async fn wait_upload_terminal<C: WorkspaceChannel>(
    channel: &mut C,
    request: RequestId,
    bulk_id: BulkId,
) -> Result<Option<Vec<u8, 512>>, HostWorkspaceError<C::Error>> {
    let mut early_reply = None;
    for _ in 0..RPC_POLL_LIMIT {
        match channel
            .poll_workspace()
            .await
            .map_err(HostWorkspaceError::Transport)?
        {
            None => yield_once().await,
            Some(ClientEvent::Control {
                kind: kore_hostlink::BULK_READY,
                request: found,
                payload,
            }) if found == request && payload.as_slice() == bulk_id.0.as_slice() => {
                return Ok(early_reply);
            }
            Some(ClientEvent::Control {
                kind: WORKSPACE_REPLY,
                request: found,
                payload,
            }) if found == request && early_reply.is_none() => {
                early_reply = Some(
                    Vec::from_slice(payload.as_slice()).map_err(|_| HostWorkspaceError::Bounds)?,
                );
            }
            Some(ClientEvent::Status(_)) => return Err(HostWorkspaceError::Unavailable),
            Some(_) => return Err(HostWorkspaceError::Protocol),
        }
    }
    Err(HostWorkspaceError::Backpressure)
}

async fn wait_bulk<C: WorkspaceChannel, const N: usize>(
    channel: &mut C,
    request: RequestId,
) -> Result<Vec<u8, N>, HostWorkspaceError<C::Error>> {
    let mut reference = None;
    let mut bytes = Vec::<u8, N>::new();
    for _ in 0..RPC_POLL_LIMIT {
        match channel
            .poll_workspace()
            .await
            .map_err(HostWorkspaceError::Transport)?
        {
            None => yield_once().await,
            Some(ClientEvent::BulkBegin {
                request: found,
                reference: found_ref,
            }) if found == request
                && found_ref.content_kind.get() == WORKSPACE_REPLY
                && found_ref.total_len.get() as usize <= N =>
            {
                if reference.replace(found_ref).is_some() {
                    return Err(HostWorkspaceError::Protocol);
                }
            }
            Some(ClientEvent::BulkChunk {
                request: found,
                bulk_id,
                offset,
                bytes: chunk,
            }) if found == request
                && reference.is_some_and(|reference| reference.bulk_id == bulk_id)
                && offset as usize == bytes.len() =>
            {
                bytes
                    .extend_from_slice(chunk.as_slice())
                    .map_err(|_| HostWorkspaceError::Bounds)?;
            }
            Some(ClientEvent::BulkReady {
                request: found,
                bulk_id,
            }) if found == request
                && reference.is_some_and(|reference| reference.bulk_id == bulk_id) =>
            {
                let reference = reference.ok_or(HostWorkspaceError::Protocol)?;
                if bytes.len() != reference.total_len.get() as usize
                    || blake3::hash(bytes.as_slice()).as_bytes() != &reference.digest_blake3
                {
                    return Err(HostWorkspaceError::Integrity);
                }
                return Ok(bytes);
            }
            Some(ClientEvent::Control {
                kind,
                request: found,
                payload,
            }) if found == request && kind == WORKSPACE_REPLY => {
                if let Ok(error) = WorkspaceErrorV1::read_from_bytes(payload.as_slice()) {
                    return Err(status_error(
                        error.validate().map_err(|_| HostWorkspaceError::Protocol)?,
                    ));
                }
                return Err(HostWorkspaceError::Protocol);
            }
            Some(ClientEvent::Status(_)) => return Err(HostWorkspaceError::Unavailable),
            Some(_) => return Err(HostWorkspaceError::Protocol),
        }
    }
    Err(HostWorkspaceError::Backpressure)
}

fn map_status<E>(status: WorkspaceStatus) -> Result<(), HostWorkspaceError<E>> {
    match status {
        WorkspaceStatus::Ok => Ok(()),
        status => Err(status_error(status)),
    }
}

fn status_error<E>(status: WorkspaceStatus) -> HostWorkspaceError<E> {
    match status {
        WorkspaceStatus::Ok => HostWorkspaceError::Protocol,
        WorkspaceStatus::Unavailable => HostWorkspaceError::Unavailable,
        WorkspaceStatus::Stale => HostWorkspaceError::Stale,
        WorkspaceStatus::Conflict | WorkspaceStatus::Partial => HostWorkspaceError::Conflict,
        WorkspaceStatus::Refused => HostWorkspaceError::Refused,
    }
}

fn map_live<E>(error: kore_workspace_sync::LiveError) -> HostWorkspaceError<E> {
    match error {
        kore_workspace_sync::LiveError::Revoked => HostWorkspaceError::Revoked,
        kore_workspace_sync::LiveError::Backpressure => HostWorkspaceError::Backpressure,
        kore_workspace_sync::LiveError::Cache(kore_workspace_sync::CacheError::Integrity) => {
            HostWorkspaceError::Integrity
        }
        kore_workspace_sync::LiveError::Range
        | kore_workspace_sync::LiveError::ArithmeticOverflow => HostWorkspaceError::Bounds,
        _ => HostWorkspaceError::Protocol,
    }
}

fn request_counter(request: RequestId) -> u64 {
    u64::from_le_bytes(
        request.0[..8]
            .try_into()
            .expect("RequestId has sixteen bytes"),
    )
}

fn transaction_id(base: [u8; 32], changes: [u8; 32], nonce: u64) -> [u8; 32] {
    let mut hash = blake3::Hasher::new();
    hash.update(b"kore.fs.workspace-transaction.v1\0");
    hash.update(&base);
    hash.update(&changes);
    hash.update(&nonce.to_le_bytes());
    let mut id = *hash.finalize().as_bytes();
    if id == [0; 32] {
        id[0] = 1;
    }
    id
}

fn workspace_bulk_id(request: RequestId, kind: u16, bytes: &[u8]) -> BulkId {
    let mut hash = blake3::Hasher::new();
    hash.update(b"kore.fs.workspace-bulk.v1\0");
    hash.update(&request.0);
    hash.update(&kind.to_le_bytes());
    hash.update(blake3::hash(bytes).as_bytes());
    let mut id = [0; 16];
    id.copy_from_slice(&hash.finalize().as_bytes()[..16]);
    if id == [0; 16] {
        id[0] = 1;
    }
    BulkId(id)
}

fn request_id(counter: u64) -> Option<RequestId> {
    if counter == 0 {
        return None;
    }
    let mut bytes = [0; 16];
    bytes[..8].copy_from_slice(&counter.to_le_bytes());
    bytes[8..].copy_from_slice(&(!counter).to_le_bytes());
    Some(RequestId(bytes))
}

#[must_use]
pub fn stable_live_object(stable: crate::StableObjectId) -> [u8; 32] {
    let mut hash = blake3::Hasher::new();
    hash.update(b"kore.fs.live-stable-object.v1\0");
    hash.update(&stable.get().to_le_bytes());
    *hash.finalize().as_bytes()
}

async fn yield_once() {
    let mut yielded = false;
    core::future::poll_fn(|cx| {
        if core::mem::replace(&mut yielded, true) {
            core::task::Poll::Ready(())
        } else {
            cx.waker().wake_by_ref();
            core::task::Poll::Pending
        }
    })
    .await
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::collections::VecDeque;

    use kore_hostlink::{BulkId, BulkRefV1};
    use kore_svc_hostlink::Payload;
    use kore_workspace_sync::{
        CanonicalPath, Entry, ManifestEntryResponseV1, build_range_proof, range_root,
    };

    pub(crate) struct FakeChannel {
        manifest: Manifest,
        content: std::vec::Vec<u8>,
        extra_contents: std::collections::BTreeMap<[u8; 16], std::vec::Vec<u8>>,
        events: VecDeque<ClientEvent>,
        range_requests: usize,
        write_grant: Option<[u8; 16]>,
        uploaded_blob: Option<std::vec::Vec<u8>>,
        pub(crate) transactions: usize,
        pub(crate) write_status: WorkspaceStatus,
        fail_next_send: bool,
        apply_writes: bool,
        pub(crate) lose_next_receipt: bool,
        pub(crate) pause_after_write: bool,
        pub(crate) pause_attach: bool,
        pub(crate) pause_attach_after_write: bool,
        pub(crate) pause_delta: bool,
    }

    impl FakeChannel {
        pub(crate) fn new() -> Self {
            let content = (0..(RANGE_BYTES + 31))
                .map(|index| index as u8)
                .collect::<std::vec::Vec<_>>();
            let mut builder = ManifestBuilder::new([0; 32], Policy::default());
            builder
                .push(Entry::directory(
                    CanonicalPath::new(b"src").unwrap(),
                    [6; 16],
                ))
                .unwrap();
            builder
                .push(
                    Entry::file_with_range_root(
                        CanonicalPath::new(b"src/main.rs").unwrap(),
                        false,
                        content.len() as u32,
                        *blake3::hash(&content).as_bytes(),
                        range_root(&content).unwrap(),
                        [7; 16],
                    )
                    .unwrap(),
                )
                .unwrap();
            Self {
                manifest: builder.finish().unwrap(),
                content,
                extra_contents: std::collections::BTreeMap::new(),
                events: VecDeque::new(),
                range_requests: 0,
                write_grant: None,
                uploaded_blob: None,
                transactions: 0,
                write_status: WorkspaceStatus::Ok,
                fail_next_send: false,
                apply_writes: false,
                lose_next_receipt: false,
                pause_after_write: false,
                pause_attach: false,
                pause_attach_after_write: false,
                pause_delta: false,
            }
        }

        pub(crate) fn with_writeback(mut self) -> Self {
            self.write_grant = Some([0x53; 16]);
            self
        }

        pub(crate) fn applying_writes(mut self) -> Self {
            self.apply_writes = true;
            self
        }
        pub(crate) fn content(&self) -> &[u8] {
            &self.content
        }
        pub(crate) fn has_path(&self, path: &[u8]) -> bool {
            self.manifest
                .entries()
                .iter()
                .any(|entry| entry.path.as_bytes() == path)
        }

        pub(crate) fn snapshot(&self) -> [u8; 32] {
            self.manifest.snapshot_id()
        }

        fn control(&mut self, request: RequestId, bytes: &[u8]) {
            self.events.push_back(ClientEvent::Control {
                kind: WORKSPACE_REPLY,
                request,
                payload: Payload::copy_from(bytes).unwrap(),
            });
        }
    }

    impl WorkspaceChannel for FakeChannel {
        type Error = ();

        async fn send_workspace(
            &mut self,
            request: RequestId,
            payload: &[u8],
        ) -> Result<(), Self::Error> {
            if core::mem::take(&mut self.fail_next_send) {
                return Err(());
            }
            match payload.get(..4) {
                Some(magic) if magic == DELTA_POLL_MAGIC && self.apply_writes => {
                    if self.pause_delta {
                        core::future::pending::<()>().await;
                    }
                    let poll = DeltaPollV1::read_from_bytes(payload).unwrap();
                    let response = WorkspaceErrorV1 {
                        magic: kore_workspace_sync::ERROR_RESPONSE_MAGIC,
                        version: U16::new(FORMAT_VERSION),
                        status: U16::new(if poll.snapshot == self.manifest.snapshot_id() {
                            WorkspaceStatus::Unavailable
                        } else {
                            WorkspaceStatus::Conflict
                        } as u16),
                        request_magic: DELTA_POLL_MAGIC,
                        reserved: U32::ZERO,
                    };
                    self.control(request, response.as_bytes());
                }
                Some(magic) if magic == ATTACH_MAGIC => {
                    if self.pause_attach || (self.pause_attach_after_write && self.transactions > 0)
                    {
                        core::future::pending::<()>().await;
                    }
                    let header = self.manifest.header();
                    let response = AttachResponseV1 {
                        magic: ATTACH_MAGIC,
                        version: U16::new(FORMAT_VERSION),
                        status: U16::new(WorkspaceStatus::Ok as u16),
                        snapshot: header.snapshot_id,
                        parent_snapshot: header.parent_snapshot,
                        root_digest: header.root_digest,
                        policy_digest: header.policy_digest,
                        entry_count: header.entry_count,
                        directory_count: header.directory_count,
                        total_bytes: header.total_bytes,
                        delta_sequence: U64::new(1),
                        write_grant: self.write_grant.unwrap_or([0; 16]),
                        write_operations: self
                            .write_grant
                            .map_or(0, |_| kore_workspace_sync::GrantOperations::ALL.bits()),
                        reserved0: 0,
                        write_max_operations: U16::new(self.write_grant.map_or(0, |_| 64)),
                        write_max_bytes: U64::new(
                            self.write_grant
                                .map_or(0, |_| kore_workspace_sync::bounds::MAX_OVERLAY_BYTES),
                        ),
                        write_expires_at: U64::ZERO,
                    };
                    self.control(request, response.as_bytes());
                }
                Some(magic) if magic == ENTRY_REQUEST_MAGIC => {
                    let query = ManifestEntryRequestV1::read_from_bytes(payload).unwrap();
                    let response = ManifestEntryResponseV1::from_entry(
                        self.manifest.snapshot_id(),
                        query.index.get(),
                        &self.manifest.entries()[usize::from(query.index.get())],
                    )
                    .unwrap();
                    self.control(request, response.as_bytes());
                }
                Some(magic) if magic == RANGE_REQUEST_MAGIC => {
                    self.range_requests += 1;
                    let query = RangeRequestV1::read_from_bytes(payload).unwrap();
                    let index = (query.offset.get() as usize / RANGE_BYTES) as u16;
                    let content = self
                        .extra_contents
                        .get(&query.source)
                        .unwrap_or(&self.content);
                    let proof = build_range_proof(content, index).unwrap();
                    let start = usize::from(index) * RANGE_BYTES;
                    let end = (start + RANGE_BYTES).min(content.len());
                    let bulk_id = BulkId(request.0);
                    let mut siblings = [[0; 32]; kore_workspace_sync::MAX_RANGE_PROOF_DEPTH];
                    siblings[..proof.siblings().len()].copy_from_slice(proof.siblings());
                    let response = RangeResponseV1 {
                        magic: RANGE_RESPONSE_MAGIC,
                        version: U16::new(FORMAT_VERSION),
                        status: U16::new(WorkspaceStatus::Ok as u16),
                        snapshot: query.snapshot,
                        source: query.source,
                        version_digest: query.version_digest,
                        bulk_id: bulk_id.0,
                        file_len: query.file_len,
                        range_index: U16::new(index),
                        range_count: U16::new(proof.leaf_count),
                        proof_depth: proof.siblings().len() as u8,
                        reserved: [0; 3],
                        siblings,
                    };
                    let mut bulk = std::vec::Vec::new();
                    bulk.extend_from_slice(response.as_bytes());
                    bulk.extend_from_slice(&content[start..end]);
                    self.events.push_back(ClientEvent::BulkBegin {
                        request,
                        reference: BulkRefV1 {
                            bulk_id,
                            total_len: U32::new(bulk.len() as u32),
                            chunk_count: U16::new(1),
                            content_kind: U16::new(WORKSPACE_REPLY),
                            digest_blake3: *blake3::hash(&bulk).as_bytes(),
                        },
                    });
                    for (offset, chunk) in bulk.chunks(512).enumerate() {
                        self.events.push_back(ClientEvent::BulkChunk {
                            request,
                            bulk_id,
                            offset: (offset * 512) as u32,
                            bytes: Payload::copy_from(chunk).unwrap(),
                        });
                    }
                    self.events
                        .push_back(ClientEvent::BulkReady { request, bulk_id });
                }
                _ => return Err(()),
            }
            Ok(())
        }

        async fn poll_workspace(&mut self) -> Result<Option<ClientEvent>, Self::Error> {
            Ok(self.events.pop_front())
        }

        async fn upload_workspace(
            &mut self,
            request: RequestId,
            bulk_id: BulkId,
            content_kind: u16,
            bytes: &[u8],
        ) -> Result<(), Self::Error> {
            match content_kind {
                WORKSPACE_BLOB_CONTENT => {
                    self.uploaded_blob = Some(bytes.to_vec());
                    self.events.push_back(ClientEvent::Control {
                        kind: kore_hostlink::BULK_READY,
                        request,
                        payload: Payload::copy_from(&bulk_id.0).unwrap(),
                    });
                    let stored = BlobStoredV1 {
                        magic: kore_workspace_sync::BLOB_STORED_MAGIC,
                        version: U16::new(FORMAT_VERSION),
                        status: U16::new(WorkspaceStatus::Ok as u16),
                        bulk_id: bulk_id.0,
                        digest: *blake3::hash(bytes).as_bytes(),
                        length: U32::new(bytes.len() as u32),
                        reserved: U32::ZERO,
                    };
                    self.control(request, stored.as_bytes());
                }
                WORKSPACE_TRANSACTION_CONTENT => {
                    let transaction = kore_workspace_sync::decode_write_transaction(bytes).unwrap();
                    assert_eq!(Some(transaction.grant), self.write_grant);
                    assert_eq!(transaction.changes.base, self.manifest.snapshot_id());
                    if let Some(change) = transaction.changes.changes().first()
                        && matches!(change.kind, ChangeKind::CreateFile | ChangeKind::ModifyFile)
                    {
                        let blob = if change.replacement_len == 0 {
                            &[][..]
                        } else {
                            self.uploaded_blob.as_deref().unwrap_or(&[])
                        };
                        assert_eq!(blob.len(), change.replacement_len as usize);
                        assert_eq!(blake3::hash(blob).as_bytes(), &change.replacement);
                    }
                    if self.apply_writes && self.write_status == WorkspaceStatus::Ok {
                        let changes = transaction.changes.changes();
                        for change in changes {
                            if matches!(
                                change.kind,
                                ChangeKind::CreateFile | ChangeKind::CreateDirectory
                            ) {
                                assert!(
                                    !self
                                        .manifest
                                        .entries()
                                        .iter()
                                        .any(|entry| entry.path == change.path)
                                );
                                assert_eq!(change.expected, [0; 32]);
                                continue;
                            }
                            let old = self
                                .manifest
                                .entries()
                                .iter()
                                .find(|entry| entry.path == change.path)
                                .unwrap();
                            assert_eq!(old.digest, change.expected);
                            if change.kind == ChangeKind::ModifyFile {
                                let bytes = if change.replacement_len == 0 {
                                    std::vec::Vec::new()
                                } else {
                                    self.uploaded_blob.clone().unwrap()
                                };
                                if old.source == [7; 16] {
                                    self.content = bytes;
                                } else {
                                    self.extra_contents.insert(old.source, bytes);
                                }
                            }
                            if change.kind == ChangeKind::Delete && old.kind == EntryKind::Directory
                            {
                                assert!(
                                    self.manifest
                                        .entries()
                                        .iter()
                                        .filter(|entry| entry.path.is_within(&change.path))
                                        .all(|entry| changes
                                            .iter()
                                            .any(|change| change.kind == ChangeKind::Delete
                                                && change.path == entry.path)),
                                    "host rmdir requires explicit child deletions"
                                );
                            }
                        }
                        let mut builder =
                            ManifestBuilder::new(self.manifest.snapshot_id(), Policy::default());
                        for entry in self.manifest.entries() {
                            if changes.iter().any(|change| {
                                change.kind == ChangeKind::Delete && entry.path == change.path
                            }) {
                                continue;
                            }
                            let mut next = entry.clone();
                            for change in changes {
                                match change.kind {
                                    ChangeKind::ModifyFile if entry.path == change.path => {
                                        let content = self
                                            .extra_contents
                                            .get(&entry.source)
                                            .unwrap_or(&self.content);
                                        next = Entry::file_with_range_root(
                                            entry.path.clone(),
                                            change.executable,
                                            content.len() as u32,
                                            change.replacement,
                                            range_root(content).unwrap(),
                                            entry.source,
                                        )
                                        .unwrap();
                                    }
                                    ChangeKind::Rename if entry.path.is_within(&change.path) => {
                                        let mut path =
                                            change.target.as_ref().unwrap().as_bytes().to_vec();
                                        path.extend_from_slice(
                                            &entry.path.as_bytes()[change.path.as_bytes().len()..],
                                        );
                                        next.path = CanonicalPath::new(&path).unwrap();
                                    }
                                    ChangeKind::ModifyFile
                                    | ChangeKind::Rename
                                    | ChangeKind::Delete
                                    | ChangeKind::CreateFile
                                    | ChangeKind::CreateDirectory => {}
                                }
                            }
                            builder.push(next).unwrap();
                        }
                        for change in changes.iter().filter(|change| {
                            matches!(
                                change.kind,
                                ChangeKind::CreateFile | ChangeKind::CreateDirectory
                            )
                        }) {
                            let mut hash = blake3::Hasher::new();
                            hash.update(&transaction.id);
                            hash.update(change.path.as_bytes());
                            let source: [u8; 16] =
                                hash.finalize().as_bytes()[..16].try_into().unwrap();
                            let entry = if change.kind == ChangeKind::CreateDirectory {
                                Entry::directory(change.path.clone(), source)
                            } else {
                                let bytes = if change.replacement_len == 0 {
                                    std::vec::Vec::new()
                                } else {
                                    self.uploaded_blob.clone().unwrap()
                                };
                                let entry = Entry::file_with_range_root(
                                    change.path.clone(),
                                    change.executable,
                                    change.replacement_len,
                                    change.replacement,
                                    range_root(&bytes).unwrap(),
                                    source,
                                )
                                .unwrap();
                                self.extra_contents.insert(source, bytes);
                                entry
                            };
                            builder.push(entry).unwrap();
                        }
                        self.manifest = builder.finish().unwrap();
                    }
                    self.transactions += 1;
                    if self.pause_after_write {
                        core::future::pending::<()>().await;
                    }
                    if core::mem::take(&mut self.lose_next_receipt) {
                        return Err(()); // Host applied; the guest did not receive its receipt.
                    }

                    let result = TransactionResultV1 {
                        magic: kore_workspace_sync::TRANSACTION_RESULT_MAGIC,
                        version: U16::new(FORMAT_VERSION),
                        status: U16::new(self.write_status as u16),
                        transaction: transaction.id,
                        changes: transaction.changes.id,
                        snapshot: if self.apply_writes {
                            self.manifest.snapshot_id()
                        } else {
                            [9; 32]
                        },
                        applied: U16::new(transaction.changes.changes().len() as u16),
                        reserved: U16::ZERO,
                    };
                    self.control(request, result.as_bytes());
                    self.events.push_back(ClientEvent::Control {
                        kind: kore_hostlink::BULK_READY,
                        request,
                        payload: Payload::copy_from(&bulk_id.0).unwrap(),
                    });
                }
                _ => return Err(()),
            }
            Ok(())
        }
    }

    #[test]
    fn ordinary_object_read_fetches_one_verified_leaf_then_hits_cache_and_revokes() {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let channel = FakeChannel::new();
                let manifest = channel.manifest.clone();
                let object = bindings::<()>(&manifest).unwrap()[0].kore_object;
                let mut live =
                    crate::block_on(HostWorkspace::attach(channel, Policy::default())).unwrap();
                let mut first = [0; 32];
                assert_eq!(
                    crate::block_on(live.read(object, 16, &mut first)).unwrap(),
                    32
                );
                assert_eq!(&first, &live.channel.content[16..48]);
                let mut second = [0; 16];
                assert_eq!(
                    crate::block_on(live.read(object, 24, &mut second)).unwrap(),
                    16
                );
                assert_eq!(live.channel.range_requests, 1);
                live.revoke();
                assert_eq!(
                    crate::block_on(live.read(object, 0, &mut second)),
                    Err(HostWorkspaceError::Revoked)
                );
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn cancelled_attachment_keeps_the_supervisors_affine_channel_slot() {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                use core::future::Future;
                let mut channel = FakeChannel::new();
                channel.pause_attach = true;
                let mut slot = Some(channel);
                let mut future = std::boxed::Box::pin(HostWorkspace::attach_in_slot(
                    &mut slot,
                    Policy::default(),
                ));
                let mut cx = core::task::Context::from_waker(core::task::Waker::noop());
                assert!(future.as_mut().poll(&mut cx).is_pending());
                drop(future);
                slot.as_mut()
                    .expect("cancelled handshake retains ownership")
                    .pause_attach = false;
                let attached =
                    crate::block_on(HostWorkspace::attach_in_slot(&mut slot, Policy::default()))
                        .unwrap();
                assert!(slot.is_none());
                assert_eq!(attached.manifest().entries().len(), 2);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn failed_attach_returns_the_affine_channel_for_reconnect() {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let mut channel = FakeChannel::new();
                channel.fail_next_send = true;
                let (returned, error) = match crate::block_on(HostWorkspace::attach_recoverable(
                    channel,
                    Policy::default(),
                )) {
                    Ok(_) => panic!("failed attach must return the channel"),
                    Err(failure) => failure,
                };
                assert_eq!(error, HostWorkspaceError::Transport(()));
                assert!(!returned.fail_next_send);
                assert!(
                    crate::block_on(HostWorkspace::attach_recoverable(
                        returned,
                        Policy::default()
                    ))
                    .is_ok()
                );
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn standing_grant_sends_preconditioned_content_and_transaction_bulks() {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let channel = FakeChannel::new().with_writeback();
                let manifest = channel.manifest.clone();
                let entry = manifest
                    .entries()
                    .iter()
                    .find(|entry| entry.kind == EntryKind::File)
                    .unwrap();
                let mut live =
                    crate::block_on(HostWorkspace::attach(channel, Policy::default())).unwrap();
                let replacement = b"updated through ordinary fs authority";
                let receipt = crate::block_on(live.writeback(
                    Change {
                        kind: ChangeKind::ModifyFile,
                        path: entry.path.clone(),
                        target: None,
                        expected: entry.digest,
                        replacement: *blake3::hash(replacement).as_bytes(),
                        replacement_len: replacement.len() as u32,
                        executable: false,
                    },
                    Some(replacement),
                ))
                .unwrap();
                assert_eq!(receipt.snapshot, [9; 32]);
                assert_eq!(live.channel.transactions, 1);
                assert_eq!(
                    live.channel.uploaded_blob.as_deref(),
                    Some(replacement.as_slice())
                );
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
