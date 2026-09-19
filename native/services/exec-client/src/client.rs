//! `client` — the exec-manager client seam.
//!
//! The task manager's data plane is its request/reply ring. Every capability
//! input crosses in the same atomic invocation transaction that publishes the
//! request, so a caller cannot expose a partial launch context.

use kore_abi::cap::{ResourceKind, Rights};
use kore_abi::device::{DeviceCensus, DeviceCensusPage};
use kore_abi::session::{SlotRef, Status};

use kore_rt::cap::{
    ChanRecvToken, ChanSendToken, ReturnedCapability, StashedLease, StashedView, TaskToken, verbs,
};
use kore_rt::chan::{CapRx, ChanTx, ExactTransferError, RecvError, SendError};
use kore_rt::executor::Doorbell;
use kore_rt::svc::{
    AffineBundleCallError, CallError, CallReply, ClientEndpoints, GrantPin, InvocationAuxiliary,
    MAX_INVOCATION_AUXILIARIES, ReturnCapability, SvcClient, client_rings,
};
use kore_rt::transport::Transport;

use crate::wire::{SHARED_WAKE_RIGHTS, SpawnSpec, TaskRefusal, TaskRequest, TaskResponse};
use crate::wire::ExecProtocol;

mod mint;
pub use mint::ExecClientMint;

/// Inert task-local executable metadata. Enumerating or copying this record
/// grants no authority; only the affine capability retained in the namespace
/// row can authorize a launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutableContext {
    None,
    /// The launcher supplies selector-specific status diagnostics (input or
    /// processor) through the bounded context prefix; no new page authority.
    InputSnapshot,
    /// The launcher serializes its current command metadata into ProcessIo's
    /// immutable context prefix. The child receives no namespace capability.
    CommandSnapshot,
    /// The bounded PCI function census (`kore_abi::device::DeviceCensus`),
    /// pulled by the launcher through exec from the monitor — the
    /// read-only observation contract behind `lspci`.  Descriptions only.
    DeviceCensus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecutableMetadata {
    pub local_name: kore_abi::image::ImageName,
    pub display_name: kore_abi::image::ImageName,
    pub usage: &'static [u8],
    pub summary: &'static [u8],
    pub context: ExecutableContext,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NamespaceError {
    Duplicate,
    Full,
    Missing,
}

struct CapabilityBinding<C> {
    metadata: ExecutableMetadata,
    capability: C,
}

/// A bounded task-owned mapping from inert names to affine authority.
///
/// The generic form makes namespace semantics host-testable without faking a
/// kernel capability. Production instantiates it with
/// [`ExecutableCapability`], which is neither `Copy` nor `Clone`.
pub struct CapabilityNamespace<C, const N: usize> {
    rows: [Option<CapabilityBinding<C>>; N],
}

impl<C, const N: usize> CapabilityNamespace<C, N> {
    pub fn new() -> Self {
        Self {
            rows: [const { None }; N],
        }
    }

    /// Bind one local name, returning the affine capability unchanged on
    /// refusal.
    pub fn bind(
        &mut self,
        metadata: ExecutableMetadata,
        capability: C,
    ) -> Result<(), (NamespaceError, C)> {
        if self
            .rows
            .iter()
            .flatten()
            .any(|row| row.metadata.local_name == metadata.local_name)
        {
            return Err((NamespaceError::Duplicate, capability));
        }
        let Some(vacant) = self.rows.iter_mut().find(|row| row.is_none()) else {
            return Err((NamespaceError::Full, capability));
        };
        *vacant = Some(CapabilityBinding {
            metadata,
            capability,
        });
        Ok(())
    }

    pub fn resolve(&self, local_name: kore_abi::image::ImageName) -> Option<&C> {
        self.rows
            .iter()
            .flatten()
            .find_map(|row| (row.metadata.local_name == local_name).then_some(&row.capability))
    }

    /// Mutably borrow the one owned binding, for task-local policy such as a
    /// remaining invocation budget. This does not derive its authority.
    pub fn resolve_mut(&mut self, local_name: kore_abi::image::ImageName) -> Option<&mut C> {
        self.rows
            .iter_mut()
            .flatten()
            .find_map(|row| (row.metadata.local_name == local_name).then_some(&mut row.capability))
    }

    /// Remove local reachability and return the affine member to its owner.
    /// The caller must retain it until native retirement is confirmed; removing
    /// a name alone is not proof of kernel revocation.
    pub fn unbind(&mut self, local_name: kore_abi::image::ImageName) -> Option<C> {
        let row = self.rows.iter_mut().find(|row| {
            row.as_ref()
                .is_some_and(|row| row.metadata.local_name == local_name)
        })?;
        row.take().map(|row| row.capability)
    }

    /// Copy only the inert metadata for one local binding. The affine
    /// capability remains owned by its namespace row.
    pub fn describe(&self, local_name: kore_abi::image::ImageName) -> Option<ExecutableMetadata> {
        self.rows
            .iter()
            .flatten()
            .find_map(|row| (row.metadata.local_name == local_name).then_some(row.metadata))
    }

    /// Rename metadata in place. The authority object is not derived,
    /// transferred, or retargeted.
    pub fn rename(
        &mut self,
        old: kore_abi::image::ImageName,
        new: kore_abi::image::ImageName,
    ) -> Result<(), NamespaceError> {
        if self
            .rows
            .iter()
            .flatten()
            .any(|row| row.metadata.local_name == new)
        {
            return Err(NamespaceError::Duplicate);
        }
        let row = self
            .rows
            .iter_mut()
            .flatten()
            .find(|row| row.metadata.local_name == old)
            .ok_or(NamespaceError::Missing)?;
        row.metadata.local_name = new;
        Ok(())
    }

    /// Enumerate metadata only. No borrow of the affine capability escapes.
    pub fn metadata(&self) -> impl Iterator<Item = ExecutableMetadata> + '_ {
        self.rows.iter().flatten().map(|row| row.metadata)
    }
}

impl<C, const N: usize> Default for CapabilityNamespace<C, N> {
    fn default() -> Self {
        Self::new()
    }
}

pub type ExecutableNamespace<const N: usize> = CapabilityNamespace<ExecutableCapability, N>;

/// Installation failure at the privileged name-resolution boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstallError {
    Call(CallError),
    Refused(TaskRefusal),
}

/// Typed reply of [`RingExec::seed_library`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LibrarySeedOutcome {
    /// The library is in the durable store (`lib/<name>`).
    pub seeded: bool,
    /// The system package manifest (`pkgs/kore/manifest`) stands.
    pub manifest: bool,
    /// Why not, when `seeded` is false (`kore_abi::svc::LibraryFailure`).
    pub failure: u8,
}

/// Typed disposition of one durable-store executable seed.
///
/// `failure` names the stage `SeedApp` stopped at
/// (`kore_abi::svc::LibraryFailure`, which the executable path shares and
/// extends above 63). Without it an `app-store: seeded N/M` verdict short of
/// M says only that something went wrong, for every member, with no stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AppSeedOutcome {
    pub seeded: bool,
    pub failure: u8,
}

/// Typed disposition of a durable-store executable probe.
///
/// `linked` proves the revision reached ImageService validation;
/// `already_resident` additionally proves digest identity selected an
/// existing executable peer before any new image generation was registered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TreeProbeOutcome {
    pub linked: bool,
    pub already_resident: bool,
}

/// Affine client-side authority for one independently writable exec face.
///
/// The ring lessor, complete reply-grant receive bank, and invocation-context
/// sends stay grouped until a parent explicitly delegates them to one child
/// session. The numeric `face` is lifecycle correlation for the privileged
/// owner; it cannot authorize invocation.
pub struct MintedExecClientFace {
    pub face: u8,
    pub ring: StashedLease,
    /// A private descendant of the service's shared request doorbell. The
    /// endpoint wakes the generic host; pair identity remains attached to the
    /// ring and cannot be selected by this capability.
    pub kick: ChanSendToken,
    pub grants: [ChanRecvToken; crate::wire::EXEC_REPLY_GRANTS],
    pub proof_tx: ChanSendToken,
    pub process_tx: ChanSendToken,
    /// Invocation transport and, only when MintClient requested nonempty
    /// filesystem rights, the registered default private-view lineage.
    /// NONE leaves this context unbound; its ACCESS bit alone grants no files.
    pub fs_view_tx: ChanSendToken,
    pub auxiliary_tx: [ChanSendToken; MAX_INVOCATION_AUXILIARIES],
}

/// The task-owned half of a dynamically minted exec face after it crosses a
/// post-founding configuration channel. The privileged parent retains only
/// the inert numeric face id needed to retire the service-owned peer.
pub struct DelegatedExecClientFace {
    pub ring: StashedView,
    pub kick: ChanSendToken,
    pub grants: [ChanRecvToken; crate::wire::EXEC_REPLY_GRANTS],
    pub proof_tx: ChanSendToken,
    pub process_tx: ChanSendToken,
    pub fs_view_tx: ChanSendToken,
    pub auxiliary_tx: [ChanSendToken; MAX_INVOCATION_AUXILIARIES],
}

/// Affine backing retained inside a dynamically connected [`RingExec`]. The
/// client rings borrow the mapped bytes for the task lifetime; this owner is
/// what keeps both that mapping and every endpoint live for exactly as long as
/// the client value.
struct OwnedExecClientFace {
    _ring: StashedView,
    _kick: ChanSendToken,
    _grants: [ChanRecvToken; crate::wire::EXEC_REPLY_GRANTS],
}

/// A bounded face-package transfer failed. Once any earlier member moved, the
/// parent must retire `face`; local owners that did not move retire normally.
/// No partial package is ever exposed as a usable [`DelegatedExecClientFace`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FacePublishError {
    pub face: u8,
    pub error: SendError,
    pub after_transfer: bool,
    /// Zero-based member of the fixed package that refused. This is inert
    /// diagnostics only; package order never selects authority.
    pub member: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceReceiveError {
    Receive(RecvError),
    Malformed,
}

/// Pace a refused face-package member without turning scheduler latency into a
/// protocol failure. The caller supplies one absolute deadline for the whole
/// package; this interval merely gives the receiver time to consume the prior
/// member and re-post its single landing.
const FACE_TRANSFER_RETRY_NS: u64 = 100_000;

async fn await_face_receiver(deadline_ns: u64) -> Result<(), SendError> {
    let now = kore_rt::executor::clock_ns().await;
    if now >= deadline_ns {
        return Err(SendError::WouldBlock);
    }
    let _ = kore_rt::executor::next_wake_for(
        FACE_TRANSFER_RETRY_NS.min(deadline_ns.saturating_sub(now)),
    )
    .await;
    Ok(())
}

async fn publish_view<T: Transport>(
    tx: &ChanTx<'_, T>,
    source: SlotRef,
    len: u64,
    rights: Rights,
    deadline_ns: u64,
) -> Result<(), SendError> {
    loop {
        match tx.transfer_view(source, 0, len, rights).await {
            Ok(()) => return Ok(()),
            Err(SendError::WouldBlock) => await_face_receiver(deadline_ns).await?,
            Err(error) => return Err(error),
        }
    }
}

async fn publish_derived_send<T: Transport>(
    tx: &ChanTx<'_, T>,
    source: &ChanSendToken,
    rights: Rights,
    deadline_ns: u64,
) -> Result<(), SendError> {
    let mut child = verbs::derive_owned_chan_send_with_rights(source.slot(), rights)
        .ok_or(SendError::Refused(Status::Denied))?;
    loop {
        match tx.transfer_exact(child).await {
            Ok(()) => return Ok(()),
            Err(ExactTransferError::NotMoved {
                error: SendError::WouldBlock,
                capability,
            }) => {
                child = capability;
                if let Err(error) = await_face_receiver(deadline_ns).await {
                    drop(child);
                    return Err(error);
                }
            }
            Err(ExactTransferError::NotMoved { error, capability }) => {
                drop(capability);
                return Err(error);
            }
            Err(ExactTransferError::AfterTransfer(error)) => return Err(error),
        }
    }
}

async fn publish_derived_recv<T: Transport>(
    tx: &ChanTx<'_, T>,
    source: &ChanRecvToken,
    rights: Rights,
    deadline_ns: u64,
) -> Result<(), SendError> {
    let mut child = verbs::derive_owned_chan_recv_with_rights(source.slot(), rights)
        .ok_or(SendError::Refused(Status::Denied))?;
    loop {
        match tx.transfer_exact(child).await {
            Ok(()) => return Ok(()),
            Err(ExactTransferError::NotMoved {
                error: SendError::WouldBlock,
                capability,
            }) => {
                child = capability;
                if let Err(error) = await_face_receiver(deadline_ns).await {
                    drop(child);
                    return Err(error);
                }
            }
            Err(ExactTransferError::NotMoved { error, capability }) => {
                drop(capability);
                return Err(error);
            }
            Err(ExactTransferError::AfterTransfer(error)) => return Err(error),
        }
    }
}

impl MintedExecClientFace {
    /// Publish the complete client package over one parent-owned
    /// configuration channel. The parent retains the revocation pivots and
    /// the child receives attenuated descendants; the child cannot observe a
    /// usable face until it has received and validated the fixed-length
    /// package.
    pub async fn publish_until<T: Transport>(
        &self,
        transport: &T,
        session_tx: &ChanSendToken,
        deadline_ns: u64,
    ) -> Result<u8, FacePublishError> {
        let face = self.face;
        let tx = ChanTx::new(transport, session_tx.slot());
        let mut published = 0u8;

        macro_rules! send_one {
            ($operation:expr) => {
                match $operation.await {
                    Ok(()) => published = published.saturating_add(1),
                    Err(error) => {
                        return Err(FacePublishError {
                            face,
                            error,
                            after_transfer: published != 0,
                            member: published,
                        });
                    }
                }
            };
        }

        let ring_slot = self.ring.slot().ok_or(FacePublishError {
            face,
            error: SendError::Refused(Status::Denied),
            after_transfer: false,
            member: 0,
        })?;
        send_one!(publish_view(
            &tx,
            ring_slot,
            kore_abi::task::FACE_RING_STRIDE,
            Rights::ACCESS | Rights::MODIFY | Rights::REVOKE,
            deadline_ns,
        ));
        send_one!(publish_derived_send(
            &tx,
            &self.kick,
            Rights::ACCESS | Rights::REVOKE,
            deadline_ns,
        ));
        for grant in &self.grants {
            send_one!(publish_derived_recv(
                &tx,
                grant,
                Rights::ACCESS | Rights::REVOKE,
                deadline_ns,
            ));
        }
        send_one!(publish_derived_send(
            &tx,
            &self.proof_tx,
            Rights::ACCESS | Rights::REVOKE,
            deadline_ns,
        ));
        send_one!(publish_derived_send(
            &tx,
            &self.process_tx,
            Rights::ACCESS | Rights::REVOKE,
            deadline_ns,
        ));
        send_one!(publish_derived_send(
            &tx,
            &self.fs_view_tx,
            kore_svc_dagfs::FsViewCapability::DELEGABLE_RIGHTS,
            deadline_ns,
        ));
        for auxiliary in &self.auxiliary_tx {
            send_one!(publish_derived_send(
                &tx,
                auxiliary,
                Rights::ACCESS | Rights::REVOKE,
                deadline_ns,
            ));
        }
        Ok(face)
    }
}

impl DelegatedExecClientFace {
    /// Receive and authenticate one complete fixed-shape face package. The
    /// numeric service row never crosses this boundary; it remains with the
    /// parent that is responsible for retirement.
    pub async fn receive<T: Transport>(rx: &mut CapRx<'_, T>) -> Result<Self, FaceReceiveError> {
        async fn next<T: Transport>(
            rx: &mut CapRx<'_, T>,
        ) -> Result<ReturnedCapability, FaceReceiveError> {
            rx.recv().await.map_err(FaceReceiveError::Receive)
        }

        let ring = next(rx)
            .await?
            .try_into_stashed_view(Rights::ACCESS | Rights::MODIFY | Rights::REVOKE)
            .map_err(|_| FaceReceiveError::Malformed)?;
        let kick = next(rx)
            .await?
            .try_into_chan_send(Rights::ACCESS | Rights::REVOKE)
            .map_err(|_| FaceReceiveError::Malformed)?;

        let mut grants = [const { None }; crate::wire::EXEC_REPLY_GRANTS];
        for destination in &mut grants {
            *destination = Some(
                next(rx)
                    .await?
                    .try_into_chan_recv(Rights::ACCESS | Rights::REVOKE)
                    .map_err(|_| FaceReceiveError::Malformed)?,
            );
        }
        let proof_tx = next(rx)
            .await?
            .try_into_chan_send(Rights::ACCESS | Rights::REVOKE)
            .map_err(|_| FaceReceiveError::Malformed)?;
        let process_tx = next(rx)
            .await?
            .try_into_chan_send(Rights::ACCESS | Rights::REVOKE)
            .map_err(|_| FaceReceiveError::Malformed)?;
        let fs_view_tx = next(rx)
            .await?
            .try_into_chan_send(kore_svc_dagfs::FsViewCapability::DELEGABLE_RIGHTS)
            .map_err(|_| FaceReceiveError::Malformed)?;
        let mut auxiliary_tx = [const { None }; MAX_INVOCATION_AUXILIARIES];
        for destination in &mut auxiliary_tx {
            *destination = Some(
                next(rx)
                    .await?
                    .try_into_chan_send(Rights::ACCESS | Rights::REVOKE)
                    .map_err(|_| FaceReceiveError::Malformed)?,
            );
        }

        Ok(Self {
            ring,
            kick,
            grants: grants.map(Option::unwrap),
            proof_tx,
            process_tx,
            fs_view_tx,
            auxiliary_tx: auxiliary_tx.map(Option::unwrap),
        })
    }
}

/// A failed atomic invocation, partitioned by kernel-proven input custody.
#[derive(Debug)]
pub enum InvokeError<A, B, C> {
    NotMoved {
        error: CallError,
        first: A,
        second: B,
        third: C,
        auxiliaries: [Option<InvocationAuxiliary>; MAX_INVOCATION_AUXILIARIES],
    },
    AfterTransfer(CallError),
}

/// Affine authority and transport state for one immutable image generation.
///
/// `endpoint` is the actual executable capability: every successful invoke
/// re-authenticates its live `ACCESS` endpoint in the same session that moves
/// the invocation inputs. The remaining fields are inert transport custody
/// bound to the same private generation peer; none can authorize a launch in
/// the absence of `endpoint`.
enum ExecutableEndpoint {
    Owned(ChanSendToken),
    Held { slot: SlotRef, rights: Rights },
}

/// A generic invocation-context send endpoint. Bootstrap owns the returned
/// endpoints dynamically; an ordinary founded task receives attenuated leaves
/// in fixed holdings. Both forms name the same authority at the session seam.
enum InvocationEndpoint {
    Owned(ChanSendToken),
    Held(SlotRef),
}

impl InvocationEndpoint {
    fn slot(&self) -> SlotRef {
        match self {
            Self::Owned(endpoint) => endpoint.slot(),
            Self::Held(slot) => *slot,
        }
    }
}

impl ExecutableEndpoint {
    fn slot(&self) -> SlotRef {
        match self {
            Self::Owned(endpoint) => endpoint.slot(),
            Self::Held { slot, .. } => *slot,
        }
    }

    fn rights(&self) -> Rights {
        match self {
            Self::Owned(endpoint) => endpoint.rights(),
            Self::Held { rights, .. } => *rights,
        }
    }
}

pub struct ExecutableCapability {
    endpoint: ExecutableEndpoint,
    pub segments: u8,
    /// Authenticated generation profile metadata. It selects only which
    /// caller-supplied context may be delegated; it is never launch authority.
    pub profile: u32,
}

impl ExecutableCapability {
    /// Minimum useful execute-only authority. `REVOKE` permits the affine
    /// holder to retire its own descendant; absence of `DERIVE` makes further
    /// delegation structurally impossible.
    pub const EXECUTE_ONLY_RIGHTS: Rights = Rights::ACCESS.union(Rights::REVOKE);

    /// Exact rights of the authority-bearing executable endpoint.
    pub fn rights(&self) -> Rights {
        self.endpoint.rights()
    }

    /// Reconstitute a typed executable capability from a delivered endpoint
    /// and inert validated image metadata. An arbitrary send endpoint cannot
    /// forge execution: only a peer rooted at exec's generation row can
    /// satisfy the invocation witness and admission receive.
    pub fn from_endpoint(
        endpoint: ChanSendToken,
        segments: u8,
        profile: u32,
    ) -> Result<Self, ChanSendToken> {
        if segments == 0 || !endpoint.rights().contains(Rights::ACCESS) {
            return Err(endpoint);
        }
        Ok(Self {
            endpoint: ExecutableEndpoint::Owned(endpoint),
            segments,
            profile,
        })
    }

    /// Possess an endpoint delivered into a founded task's fixed holdings.
    /// The role/index is inert; the live held `ChanSend` record is checked
    /// here and re-authenticated by the invocation transaction.
    pub fn from_held(slot: SlotRef, segments: u8, profile: u32) -> Option<Self> {
        let rights =
            kore_rt::cap::verbs::capability_rights(slot, kore_abi::cap::ResourceKind::ChanSend)?;
        (segments != 0 && rights.contains(Rights::ACCESS)).then_some(Self {
            endpoint: ExecutableEndpoint::Held { slot, rights },
            segments,
            profile,
        })
    }

    /// Consume the wrapper for ordinary affine transfer into a founded task.
    pub fn into_endpoint(self) -> Result<ChanSendToken, Self> {
        let Self {
            endpoint,
            segments,
            profile,
        } = self;
        match endpoint {
            ExecutableEndpoint::Owned(endpoint) => Ok(endpoint),
            endpoint @ ExecutableEndpoint::Held { .. } => Err(Self {
                endpoint,
                segments,
                profile,
            }),
        }
    }

    /// Derive an attenuated descendant. Execute-only holders fail here
    /// because the kernel authenticates `DERIVE` on the held source.
    pub fn derive(&self, rights: Rights) -> Option<Self> {
        if !rights.contains(Rights::ACCESS)
            || !kore_rt::cap::CHANNEL_ENDPOINT_RIGHTS.contains(rights)
        {
            return None;
        }
        let endpoint =
            kore_rt::cap::verbs::derive_owned_chan_send_with_rights(self.endpoint.slot(), rights)?;
        Some(Self {
            endpoint: ExecutableEndpoint::Owned(endpoint),
            segments: self.segments,
            profile: self.profile,
        })
    }

    /// Derive the ordinary non-delegable authority installed in a child
    /// task's local namespace.
    pub fn derive_execute_only(&self) -> Option<Self> {
        self.derive(Self::EXECUTE_ONLY_RIGHTS)
    }

    /// Publish only execution authority directly from this retained pivot.
    /// The atomic derive/send retains no temporary endpoint across a retry.
    pub async fn transfer_execute_only<T: Transport>(
        &self,
        tx: &kore_rt::chan::ChanTx<'_, T>,
    ) -> Result<(), kore_rt::chan::SendError> {
        tx.transfer_derived_send(self.endpoint.slot(), Self::EXECUTE_ONLY_RIGHTS)
            .await
    }

    /// Explicitly revoke/retire this local executable-authority subtree.
    pub fn try_retire(self) -> Result<kore_rt::cap::RetireOutcome, Self> {
        let Self {
            endpoint,
            segments,
            profile,
        } = self;
        match endpoint {
            ExecutableEndpoint::Owned(endpoint) => match endpoint.try_retire() {
                Ok(outcome) => Ok(outcome),
                Err(endpoint) => Err(Self {
                    endpoint: ExecutableEndpoint::Owned(endpoint),
                    segments,
                    profile,
                }),
            },
            ExecutableEndpoint::Held { slot, rights } => {
                let outcome = kore_rt::cap::verbs::retire_slot_outcome(slot);
                if !outcome.is_closed() {
                    Err(Self {
                        endpoint: ExecutableEndpoint::Held { slot, rights },
                        segments,
                        profile,
                    })
                } else {
                    Ok(outcome)
                }
            }
        }
    }
}

/// One successfully founded task and the affine lifecycle authority that
/// makes it the caller's child. The numeric id is diagnostic; `handle` is the
/// ownership.
#[derive(Debug)]
pub struct SpawnedTask {
    /// Hash-derived task id carried in the data reply.
    pub id: u64,
    /// Validated linked-image segment count returned by exec; zero for an
    /// unlinked founding.
    pub segments: u8,
    /// Exec's process-table snapshot after admitting this child.
    pub snapshot: kore_abi::monitor::TelemetrySnapshot,
    /// The filesystem posture captured in the same launch, for the caller to
    /// publish into the child's process page.
    pub filesystem: kore_abi::monitor::FilesystemSnapshot,
    /// The frame-service facts captured in the same launch (empty unless
    /// the image's profile asked), published the same way.
    pub net: kore_abi::monitor::NetSnapshot,
    /// Child handle returned on grant channel 0 (`MODIFY` is the explicit
    /// START authority). Ordinary invokes receive exact
    /// `ACCESS | MODIFY | REVOKE`; the explicit delegable invoke additionally
    /// receives `DERIVE` so the controller can attenuate a lifecycle observer.
    pub handle: TaskToken,
    /// Optional compositor response sender returned only by the exact
    /// filesystem-window or power-window launch contract. It is a sibling of
    /// the service response send and targets the same receive endpoint installed as the child's
    /// sole executor doorbell.
    pub shared_wake: Option<ChanSendToken>,
}

/// The closed answer vocabulary of an owned spawn round.
#[derive(Debug)]
pub enum SpawnOutcome {
    /// Founding and handle transfer both committed.
    Spawned(SpawnedTask),
    /// Exec refused before transferring ownership.
    Refused(TaskRefusal),
}

/// Exact bootstrap operation whose reply remains live after a bounded wait.
///
/// `TaskReqWire` is intentionally not semantic data and does not implement
/// equality. The typed client therefore retains this compact operation
/// identity and refuses a mismatched resume before touching the transport or
/// reply ring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TimedOutBootstrapRequest {
    SeedApp {
        target: kore_abi::image::ImageName,
    },
    SeedLibrary {
        target: kore_abi::image::ImageName,
    },
    TreeLoadProbe {
        target: kore_abi::image::ImageName,
    },
    TreeAcquire {
        target: kore_abi::image::ImageName,
    },
    Install {
        target: kore_abi::image::ImageName,
        profile: u32,
        include_context: bool,
        from_tree: bool,
    },
}

/// One bounded observation slice for INIT's audited bootstrap operations.
///
/// This is operation policy, not a generic protocol default: every operation
/// below retains a typed identity across expiry and INIT owns the whole-request
/// deadline. Ordinary exec calls stay unbounded and can never strand their
/// pair merely because an unaudited caller discarded `Timeout`.
const BOOTSTRAP_REPLY_BUDGET_NS: u64 = 5_000_000_000;

const fn invoke_result_grants(return_shared_wake: bool) -> u8 {
    1 + return_shared_wake as u8
}

/// The ring implementation of the generic, atomic invocation protocol.
pub struct RingExec<'w, 't, T: Transport> {
    pair: SvcClient<'w, 't, T, ExecProtocol>,
    reply_wait: Option<Doorbell>,
    proof_tx: Option<InvocationEndpoint>,
    process_tx: Option<InvocationEndpoint>,
    fs_view_tx: Option<InvocationEndpoint>,
    auxiliary_tx: [Option<InvocationEndpoint>; MAX_INVOCATION_AUXILIARIES],
    owned_face: Option<OwnedExecClientFace>,
    timed_out_bootstrap: Option<TimedOutBootstrapRequest>,
    discarded_wake: Option<ReturnedCapability>,
}

/// Retain a discarded launch's returned prefix until retirement confirms
/// closure. A failed attempt never erases the only owner or permits a new Invoke.
fn retire_discarded<C>(owner: &mut Option<C>, retire: impl FnOnce(&mut C) -> bool) -> bool {
    if owner.as_mut().is_some_and(|owner| !retire(owner)) {
        return false;
    }
    *owner = None;
    true
}

fn finish_invoke_reply<T: Transport>(
    pair: &mut SvcClient<'_, '_, T, ExecProtocol>,
    mut reply: CallReply<TaskResponse>,
    handle_rights: Rights,
    return_shared_wake: bool,
    discarded_wake: &mut Option<ReturnedCapability>,
) -> Result<SpawnOutcome, CallError> {
    let task_index = kore_abi::svc::TaskReqWire::invoke_task_grant(return_shared_wake);
    let wake_index = kore_abi::svc::TaskReqWire::INVOKE_WAKE_GRANT;
    match reply.response {
        TaskResponse::Spawned {
            id,
            segments,
            snapshot,
            filesystem,
            net,
        } if reply.grants == invoke_result_grants(return_shared_wake)
            && id == 0
            && reply.delivered.iter().all(Option::is_none)
            && reply.retained[1 + usize::from(return_shared_wake)..]
                .iter()
                .all(Option::is_none) =>
        {
            let Some(returned) = reply.retained[task_index].take() else {
                pair.poison_retained(reply.retained);
                return Err(CallError::Wire);
            };
            let shared_wake = if return_shared_wake {
                let Some(wake) = reply.retained[wake_index].take() else {
                    reply.retained[task_index] = Some(returned);
                    pair.poison_retained(reply.retained);
                    return Err(CallError::Wire);
                };
                if !wake.is_kind_with_exact_rights(ResourceKind::ChanSend, SHARED_WAKE_RIGHTS) {
                    reply.retained[task_index] = Some(returned);
                    reply.retained[wake_index] = Some(wake);
                    pair.poison_retained(reply.retained);
                    return Err(CallError::Wire);
                }
                match wake.try_into_chan_send(SHARED_WAKE_RIGHTS) {
                    Ok(wake) => Some(wake),
                    Err(wake) => {
                        reply.retained[task_index] = Some(returned);
                        reply.retained[wake_index] = Some(wake);
                        pair.poison_retained(reply.retained);
                        return Err(CallError::Wire);
                    }
                }
            } else {
                None
            };
            match verbs::claim_direct_returned_task(returned, handle_rights) {
                Ok(handle) => Ok(SpawnOutcome::Spawned(SpawnedTask {
                    id: handle.id(),
                    segments,
                    snapshot,
                    filesystem,
                    net,
                    handle,
                    shared_wake,
                })),
                Err((returned, _)) => {
                    reply.retained[task_index] = Some(returned);
                    if let Some(wake) = shared_wake {
                        reply.retained[wake_index] = Some(wake.into_returned_capability());
                    }
                    pair.poison_retained(reply.retained);
                    Err(CallError::Wire)
                }
            }
        }
        TaskResponse::Refused(TaskRefusal::Unfounded)
            if reply.grants == u8::from(return_shared_wake)
                && reply.delivered.iter().all(Option::is_none)
                && reply.retained[usize::from(return_shared_wake)..]
                    .iter()
                    .all(Option::is_none) =>
        {
            if return_shared_wake {
                let Some(wake) = reply.retained[wake_index].as_ref() else {
                    pair.poison_retained(reply.retained);
                    return Err(CallError::Wire);
                };
                if !wake.is_kind_with_exact_rights(ResourceKind::ChanSend, SHARED_WAKE_RIGHTS)
                    || discarded_wake.is_some()
                {
                    pair.poison_retained(reply.retained);
                    return Err(CallError::Wire);
                }
                *discarded_wake = reply.retained[wake_index].take();
            }
            if retire_discarded(discarded_wake, ReturnedCapability::try_retire) {
                Ok(SpawnOutcome::Refused(TaskRefusal::Unfounded))
            } else {
                Err(CallError::Timeout)
            }
        }
        TaskResponse::Refused(refusal)
            if crate::wire::is_spawn_refusal(refusal)
                && reply.grants == 0
                && reply.delivered.iter().all(Option::is_none)
                && reply.retained.iter().all(Option::is_none) =>
        {
            Ok(SpawnOutcome::Refused(refusal))
        }
        _ => {
            pair.poison_retained(reply.retained);
            Err(CallError::Wire)
        }
    }
}

impl<'w, 't, T: Transport> RingExec<'w, 't, T> {
    fn ensure_no_discarded_wake(&self) -> Result<(), InstallError> {
        if self.discarded_wake.is_some() {
            Err(InstallError::Call(CallError::Outstanding))
        } else {
            Ok(())
        }
    }

    /// The held exec face relays monitor-owned inert data; no authority moves.
    pub async fn processor_status(
        &mut self,
        first: u16,
    ) -> Option<kore_abi::processor_status::ProcessorSnapshot> {
        if !self.is_idle() {
            return None;
        }
        let mut out = kore_abi::processor_status::ProcessorSnapshot::default();
        out.first.set(first);
        while (out.count.get() as usize) < out.rows.len() {
            let next = first.checked_add(out.count.get())?;
            let reply = self
                .pair
                .call(&TaskRequest::ProcessorStatus { first: next })
                .await
                .ok()?;
            let TaskResponse::ProcessorStatus(page) = reply.response else {
                return None;
            };
            if !page.validate() || page.first.get() != next {
                return None;
            }
            if out.count.get() == 0 {
                out.started_ns = page.started_ns;
                out.total = page.total;
            }
            if out.total != page.total {
                return None;
            }
            let start = out.count.get() as usize;
            let count = page.count.get() as usize;
            if start.checked_add(count)? > out.rows.len() {
                return None;
            }
            out.rows[start..start + count].copy_from_slice(&page.rows[..count]);
            out.count.set((start + count) as u16);
            out.finished_ns = page.finished_ns;
            if u32::from(next) + count as u32 == out.total.get() {
                break;
            }
            if count == 0 {
                return None;
            }
        }
        out.available.set(1);
        out.validate().then_some(out)
    }
    /// Assemble an exec client from its exclusive pair and return send end.
    pub fn new(pair: SvcClient<'w, 't, T, ExecProtocol>) -> Self {
        Self {
            pair,
            reply_wait: None,
            proof_tx: None,
            process_tx: None,
            fs_view_tx: None,
            auxiliary_tx: [const { None }; MAX_INVOCATION_AUXILIARIES],
            owned_face: None,
            timed_out_bootstrap: None,
            discarded_wake: None,
        }
    }

    fn finish_bootstrap_wait(
        &mut self,
        request: TimedOutBootstrapRequest,
        result: Result<CallReply<TaskResponse>, CallError>,
    ) -> Result<CallReply<TaskResponse>, InstallError> {
        match result {
            Ok(reply) => {
                self.timed_out_bootstrap = None;
                Ok(reply)
            }
            Err(CallError::Timeout) => {
                self.timed_out_bootstrap = Some(request);
                Err(InstallError::Call(CallError::Timeout))
            }
            Err(error) => Err(InstallError::Call(error)),
        }
    }

    async fn resume_timed_out_bootstrap(
        &mut self,
        request: TimedOutBootstrapRequest,
    ) -> Result<CallReply<TaskResponse>, InstallError> {
        self.ensure_no_discarded_wake()?;
        match self.timed_out_bootstrap {
            Some(outstanding) if outstanding == request => {}
            Some(_) => return Err(InstallError::Call(CallError::Outstanding)),
            None => return Err(InstallError::Call(CallError::NoOutstanding)),
        }
        let result = self.pair.resume_bounded(BOOTSTRAP_REPLY_BUDGET_NS).await;
        self.finish_bootstrap_wait(request, result)
    }

    /// Connect a post-founding delegated face without fabricating held boot
    /// slots. The receiver maps a detached nondelegable view at its bounded
    /// map cursor; ownership of the view and endpoints remains inside the
    /// returned client.
    pub fn from_delegated(
        transport: &'t T,
        face: DelegatedExecClientFace,
    ) -> Option<RingExec<'static, 't, T>> {
        let DelegatedExecClientFace {
            ring,
            kick,
            grants,
            proof_tx,
            process_tx,
            fs_view_tx,
            auxiliary_tx,
        } = face;
        let mapped = verbs::map_stashed_view(&ring)?;
        if mapped.len() != kore_abi::task::FACE_RING_STRIDE as usize {
            return None;
        }
        // The detached view owner retained below keeps the mapping live for
        // the client lifetime. The service owns only the opposite SPSC halves.
        let (base, len) = {
            // SAFETY: the authenticated detached view spans the complete
            // exclusive client ring and remains owned by `owned_face`.
            let bytes = unsafe { &mut *mapped.as_ptr() };
            (bytes.as_mut_ptr(), bytes.len())
        };
        let window: &'static mut [u8] = unsafe { core::slice::from_raw_parts_mut(base, len) };
        let rings =
            client_rings::<ExecProtocol>(window, kore_rt::svc::REQ_CAP, kore_rt::svc::REP_CAP)
                .ok()?;
        let mut pins = [None; kore_rt::service::MAX_GRANTS];
        for (index, grant) in grants.iter().enumerate() {
            pins[index] = Some(GrantPin::retained(grant.slot()));
        }
        let pair = SvcClient::new(
            transport,
            rings,
            ClientEndpoints {
                kick: kick.slot(),
                grants: pins,
            },
        );
        Some(RingExec {
            pair,
            reply_wait: None,
            proof_tx: Some(InvocationEndpoint::Owned(proof_tx)),
            process_tx: Some(InvocationEndpoint::Owned(process_tx)),
            fs_view_tx: Some(InvocationEndpoint::Owned(fs_view_tx)),
            auxiliary_tx: auxiliary_tx.map(|endpoint| Some(InvocationEndpoint::Owned(endpoint))),
            owned_face: Some(OwnedExecClientFace {
                _ring: ring,
                _kick: kick,
                _grants: grants,
            }),
            timed_out_bootstrap: None,
            discarded_wake: None,
        })
    }

    /// Fuse each held-capability convey with a park on `reply_wait`.
    ///
    /// Executor-owned clients opt into the L4 send-and-wait crossing. A
    /// hand-driven bootstrap client keeps the ordinary asynchronous transfer
    /// by omitting this capability; it owns different doorbell settlement.
    pub fn with_reply_wait(mut self, reply_wait: Doorbell) -> Self {
        self.reply_wait = Some(reply_wait);
        self
    }

    /// Bind the fixed proof/ProcessIo/FS lanes and bounded generic auxiliary
    /// endpoints delivered in this task's bootstrap holdings. The caller
    /// supplies slots, never ambient names.
    pub fn with_held_invocation_context(
        mut self,
        proof_tx: SlotRef,
        process_tx: SlotRef,
        fs_view_tx: SlotRef,
        auxiliary_tx: [SlotRef; MAX_INVOCATION_AUXILIARIES],
    ) -> Option<Self> {
        for slot in [proof_tx, process_tx, fs_view_tx]
            .into_iter()
            .chain(auxiliary_tx)
        {
            let rights = verbs::capability_rights(slot, kore_abi::cap::ResourceKind::ChanSend)?;
            if !rights.contains(Rights::ACCESS) {
                return None;
            }
        }
        self.proof_tx = Some(InvocationEndpoint::Held(proof_tx));
        self.process_tx = Some(InvocationEndpoint::Held(process_tx));
        self.fs_view_tx = Some(InvocationEndpoint::Held(fs_view_tx));
        self.auxiliary_tx = auxiliary_tx.map(|slot| Some(InvocationEndpoint::Held(slot)));
        Some(self)
    }

    /// Remove the bootstrap-owned invocation context for affine transfer into
    /// the ordinary client's holdings. This is available only before the
    /// client has been converted to fixed held endpoints.
    pub fn take_invocation_context(
        &mut self,
    ) -> Option<(
        ChanSendToken,
        ChanSendToken,
        ChanSendToken,
        [ChanSendToken; MAX_INVOCATION_AUXILIARIES],
    )> {
        let auxiliaries = core::array::from_fn(|index| self.auxiliary_tx[index].take());
        match (
            self.proof_tx.take(),
            self.process_tx.take(),
            self.fs_view_tx.take(),
        ) {
            (
                Some(InvocationEndpoint::Owned(proof)),
                Some(InvocationEndpoint::Owned(process)),
                Some(InvocationEndpoint::Owned(fs_view)),
            ) if auxiliaries
                .iter()
                .all(|entry| matches!(entry, Some(InvocationEndpoint::Owned(_)))) =>
            {
                Some((
                    proof,
                    process,
                    fs_view,
                    auxiliaries.map(|entry| match entry {
                        Some(InvocationEndpoint::Owned(endpoint)) => endpoint,
                        _ => unreachable!(),
                    }),
                ))
            }
            (proof, process, fs_view) => {
                self.proof_tx = proof;
                self.process_tx = process;
                self.fs_view_tx = fs_view;
                self.auxiliary_tx = auxiliaries;
                None
            }
        }
    }

    /// Derive one affine filesystem/cwd input for an ordinary invocation.
    ///
    /// The retained context endpoint is the caller's reusable authority. Each
    /// invoke moves only a fresh descendant, so a successful launch cannot
    /// consume or retarget the caller's cwd binding. The session still
    /// re-authenticates this descendant against the service-owned view lineage
    /// in the same atomic commit that moves the other invocation members.
    pub fn derive_filesystem_invocation(&self) -> Option<ChanSendToken> {
        self.derive_filesystem_invocation_outcome().ok()
    }

    /// Preserve the cleanup owner when native derivation has no proved outcome.
    /// Callers that retain invocation inputs must retain this uncertainty too.
    pub fn derive_filesystem_invocation_outcome(
        &self,
    ) -> Result<ChanSendToken, verbs::EndpointDeriveError> {
        let fs_view = self.fs_view_tx.as_ref().ok_or(
            verbs::EndpointDeriveError::Refused(verbs::EndpointDeriveRefusal::Unavailable),
        )?;
        verbs::derive_owned_chan_send_outcome(
            fs_view.slot(),
            kore_svc_dagfs::FsViewCapability::INVOCATION_DELEGATE_RIGHTS,
        )
    }

    /// Whether the shell-owned pair can publish a new operation.
    ///
    /// `false` includes both a resumable outstanding operation and a poisoned
    /// pair; the next mutating operation remains responsible for distinguishing
    /// those states without racing a second publication onto the ring.
    pub fn is_idle(&self) -> bool {
        self.discarded_wake.is_none() && self.pair.is_idle()
    }

    /// Drive one caller-owned construction to a complete minted face. The
    /// construction remains with the caller if this future is cancelled or
    /// returns an error. Resident callers use its bounded `observe` operation
    /// and abort/cleanup continuation directly.
    pub async fn mint_client_face(
        &mut self,
        construction: &mut ExecClientMint,
    ) -> Result<MintedExecClientFace, InstallError> {
        loop {
            if construction.aborting() || construction.closed() {
                return Err(InstallError::Call(CallError::Wiring));
            }
            construction.observe(self, None).await?;
            if let Some(face) = construction.take_ready() {
                return Ok(face);
            }
        }
    }

    /// Begin retirement once; resumption consumes only that exact request.
    pub async fn observe_retire_client_face(
        &mut self,
        face: u8,
        resume: bool,
        budget_ns: u64,
    ) -> Result<u64, InstallError> {
        self.begin_retire_client_face(face, resume, Some(budget_ns))
            .await
    }

    pub async fn resume_retire_client_face(&mut self, face: u8) -> Result<u64, InstallError> {
        self.begin_retire_client_face(face, true, None).await
    }

    pub async fn begin_retire_client_face(
        &mut self,
        face: u8,
        resume: bool,
        budget_ns: Option<u64>,
    ) -> Result<u64, InstallError> {
        match self
            .client_retirement_observation(TaskRequest::RetireClient { face }, resume, budget_ns)
            .await?
        {
            TaskResponse::ClientRetiring {
                face: got,
                retirement,
            } if got == face && retirement != 0 => Ok(retirement),
            _ => Err(InstallError::Call(CallError::Wire)),
        }
    }

    /// `false` means resources remain owned by exec. A suspended observation
    /// must resume exactly; a completed pending reply permits another query
    /// with the same service-issued token, never a new RetireClient.
    pub async fn confirm_client_retirement(
        &mut self,
        face: u8,
        retirement: u64,
        resume: bool,
        budget_ns: Option<u64>,
    ) -> Result<bool, InstallError> {
        if retirement == 0 {
            return Err(InstallError::Call(CallError::Wire));
        }
        match self
            .client_retirement_observation(
                TaskRequest::ConfirmClientRetired { face, retirement },
                resume,
                budget_ns,
            )
            .await?
        {
            TaskResponse::ClientRetiring {
                face: got,
                retirement: token,
            } if got == face && token == retirement && token != 0 => Ok(false),
            TaskResponse::ClientRetirementConfirmed {
                face: got,
                retirement: token,
            } if got == face && token == retirement && token != 0 => Ok(true),
            _ => Err(InstallError::Call(CallError::Wire)),
        }
    }

    async fn client_retirement_observation(
        &mut self,
        request: TaskRequest,
        resume: bool,
        budget_ns: Option<u64>,
    ) -> Result<TaskResponse, InstallError> {
        self.ensure_no_discarded_wake()?;
        let reply = match (resume, budget_ns) {
            (false, None) => self.pair.call(&request).await,
            (true, None) => self.pair.resume_exact(&request).await,
            (false, Some(budget)) => self.pair.call_bounded(&request, budget).await,
            (true, Some(budget)) => self.pair.resume_exact_bounded(&request, budget).await,
        }
        .map_err(InstallError::Call)?;
        if reply.grants != 0
            || reply.delivered.iter().any(Option::is_some)
            || reply.retained.iter().any(Option::is_some)
        {
            return Err(InstallError::Call(CallError::Wire));
        }
        match reply.response {
            TaskResponse::Refused(refusal) => Err(InstallError::Refused(refusal)),
            response => Ok(response),
        }
    }

    /// The whole PCI function census (`kore_abi::device::DeviceCensus`),
    /// page by page, for a launcher stamping `ExecutableContext::DeviceCensus`.
    /// All-or-nothing: a page that disagrees with the first one's header, or
    /// a census that never completes, yields no census rather than a hole.
    pub async fn device_census(&mut self) -> Result<DeviceCensus, InstallError> {
        let mut census = DeviceCensus::EMPTY;
        let first = self.device_census_page(0).await?;
        if !census.insert_page(&first) {
            return Err(InstallError::Call(CallError::Wire));
        }
        for page in 1..census.page_count() {
            let page = self.device_census_page(page as u8).await?;
            if !census.insert_page(&page) {
                return Err(InstallError::Call(CallError::Wire));
            }
        }
        if census.validate() && census.complete() {
            Ok(census)
        } else {
            Err(InstallError::Call(CallError::Wire))
        }
    }

    async fn device_census_page(&mut self, page: u8) -> Result<DeviceCensusPage, InstallError> {
        self.ensure_no_discarded_wake()?;
        let reply = self
            .pair
            .call(&TaskRequest::DeviceCensus { page })
            .await
            .map_err(InstallError::Call)?;
        match reply.response {
            TaskResponse::DeviceCensus(got)
                if got.page == page
                    && reply.grants == 0
                    && reply.delivered.iter().all(Option::is_none)
                    && reply.retained.iter().all(Option::is_none) =>
            {
                Ok(got)
            }
            TaskResponse::Refused(refusal)
                if reply.grants == 0
                    && reply.delivered.iter().all(Option::is_none)
                    && reply.retained.iter().all(Option::is_none) =>
            {
                Err(InstallError::Refused(refusal))
            }
            _ => Err(InstallError::Call(CallError::Wire)),
        }
    }

    /// Irreversibly close this service instance's bootstrap-only global-name
    /// resolver before handing the SPSC pair to an ordinary task.
    /// Copy one already-registered generation into the durable executable
    /// store. `Ok(false)` means no store could be reached — a machine with no
    /// disk, which still boots on the archive it already has.
    pub async fn seed_app(
        &mut self,
        target: kore_abi::image::ImageName,
    ) -> Result<AppSeedOutcome, InstallError> {
        self.ensure_no_discarded_wake()?;
        let request = TimedOutBootstrapRequest::SeedApp { target };
        let result = self
            .pair
            .call_bounded(&TaskRequest::SeedApp { target }, BOOTSTRAP_REPLY_BUDGET_NS)
            .await;
        let reply = self.finish_bootstrap_wait(request, result)?;
        self.finish_seed_app_reply(reply)
    }

    /// Continue only the exact [`Self::seed_app`] request already published
    /// on this pair. A timeout leaves that request live; publishing an archive
    /// install as a fallback before collecting its reply would alias the
    /// uncorrelated response stream.
    pub async fn resume_seed_app(
        &mut self,
        target: kore_abi::image::ImageName,
    ) -> Result<AppSeedOutcome, InstallError> {
        let reply = self
            .resume_timed_out_bootstrap(TimedOutBootstrapRequest::SeedApp { target })
            .await?;
        self.finish_seed_app_reply(reply)
    }

    /// Copy one archive library into the durable library directory
    /// (`lib/`).  `seeded == false` is the no-store posture, as for
    /// [`Self::seed_app`]; `manifest` reports whether the system package
    /// manifest stands, which exec writes once every executable and library
    /// has seeded — so it is normally true only on the last library's reply.
    pub async fn seed_library(
        &mut self,
        target: kore_abi::image::ImageName,
    ) -> Result<LibrarySeedOutcome, InstallError> {
        self.ensure_no_discarded_wake()?;
        let request = TimedOutBootstrapRequest::SeedLibrary { target };
        let result = self
            .pair
            .call_bounded(
                &TaskRequest::SeedLibrary { target },
                BOOTSTRAP_REPLY_BUDGET_NS,
            )
            .await;
        let reply = self.finish_bootstrap_wait(request, result)?;
        self.finish_seed_library_reply(reply)
    }

    /// Continue only the exact [`Self::seed_library`] request already
    /// published on this pair (the [`Self::resume_seed_app`] discipline).
    pub async fn resume_seed_library(
        &mut self,
        target: kore_abi::image::ImageName,
    ) -> Result<LibrarySeedOutcome, InstallError> {
        let reply = self
            .resume_timed_out_bootstrap(TimedOutBootstrapRequest::SeedLibrary { target })
            .await?;
        self.finish_seed_library_reply(reply)
    }

    fn finish_seed_library_reply(
        &mut self,
        reply: CallReply<TaskResponse>,
    ) -> Result<LibrarySeedOutcome, InstallError> {
        match reply.response {
            TaskResponse::LibrarySeeded {
                seeded,
                manifest,
                failure,
            } if reply.grants == 0
                && reply.delivered.iter().all(Option::is_none)
                && reply.retained.iter().all(Option::is_none) =>
            {
                Ok(LibrarySeedOutcome {
                    seeded,
                    manifest,
                    failure,
                })
            }
            TaskResponse::Refused(refusal)
                if reply.grants == 0
                    && reply.delivered.iter().all(Option::is_none)
                    && reply.retained.iter().all(Option::is_none) =>
            {
                Err(InstallError::Refused(refusal))
            }
            _ => Err(InstallError::Call(CallError::Wire)),
        }
    }

    fn finish_seed_app_reply(
        &mut self,
        reply: CallReply<TaskResponse>,
    ) -> Result<AppSeedOutcome, InstallError> {
        match reply.response {
            TaskResponse::AppSeeded { seeded, failure }
                if reply.grants == 0
                    && reply.delivered.iter().all(Option::is_none)
                    && reply.retained.iter().all(Option::is_none) =>
            {
                Ok(AppSeedOutcome { seeded, failure })
            }
            TaskResponse::Refused(refusal)
                if reply.grants == 0
                    && reply.delivered.iter().all(Option::is_none)
                    && reply.retained.iter().all(Option::is_none) =>
            {
                Err(InstallError::Refused(refusal))
            }
            _ => {
                self.pair.poison();
                Err(InstallError::Call(CallError::Wire))
            }
        }
    }

    /// Load-and-discard probe of the durable store: whether `target` resolves
    /// from `app/` and links. Creates no peer or launch authority — a pure
    /// witness that the store read-back path (the seed's mirror) works.
    /// A false `linked` bit means the bytes were absent or did not link. The
    /// stronger `already_resident` bit proves this call consumed no image or
    /// exec generation row.
    pub async fn tree_load_probe(
        &mut self,
        target: kore_abi::image::ImageName,
    ) -> Result<TreeProbeOutcome, InstallError> {
        self.ensure_no_discarded_wake()?;
        let request = TimedOutBootstrapRequest::TreeLoadProbe { target };
        let result = self
            .pair
            .call_bounded(
                &TaskRequest::TreeLoadProbe { target },
                BOOTSTRAP_REPLY_BUDGET_NS,
            )
            .await;
        let reply = self.finish_bootstrap_wait(request, result)?;
        self.finish_tree_load_probe_reply(reply)
    }

    /// Continue only the exact [`Self::tree_load_probe`] request already
    /// published on this pair.
    pub async fn resume_tree_load_probe(
        &mut self,
        target: kore_abi::image::ImageName,
    ) -> Result<TreeProbeOutcome, InstallError> {
        let reply = self
            .resume_timed_out_bootstrap(TimedOutBootstrapRequest::TreeLoadProbe { target })
            .await?;
        self.finish_tree_load_probe_reply(reply)
    }

    fn finish_tree_load_probe_reply(
        &mut self,
        reply: CallReply<TaskResponse>,
    ) -> Result<TreeProbeOutcome, InstallError> {
        match reply.response {
            TaskResponse::TreeProbed {
                linked,
                already_resident,
            } if reply.grants == 0
                && reply.delivered.iter().all(Option::is_none)
                && reply.retained.iter().all(Option::is_none) =>
            {
                Ok(TreeProbeOutcome {
                    linked,
                    already_resident,
                })
            }
            TaskResponse::Refused(refusal)
                if reply.grants == 0
                    && reply.delivered.iter().all(Option::is_none)
                    && reply.retained.iter().all(Option::is_none) =>
            {
                Err(InstallError::Refused(refusal))
            }
            _ => {
                self.pair.poison();
                Err(InstallError::Call(CallError::Wire))
            }
        }
    }

    pub async fn seal_installer(&mut self) -> Result<(), InstallError> {
        self.ensure_no_discarded_wake()?;
        let reply = self
            .pair
            .call(&TaskRequest::SealInstaller)
            .await
            .map_err(InstallError::Call)?;
        match reply.response {
            TaskResponse::InstallerSealed
                if reply.grants == 0
                    && reply.delivered.iter().all(Option::is_none)
                    && reply.retained.iter().all(Option::is_none) =>
            {
                Ok(())
            }
            TaskResponse::Refused(refusal)
                if reply.grants == 0
                    && reply.delivered.iter().all(Option::is_none)
                    && reply.retained.iter().all(Option::is_none) =>
            {
                Err(InstallError::Refused(refusal))
            }
            _ => {
                self.pair.poison_retained(reply.retained);
                Err(InstallError::Call(CallError::Wire))
            }
        }
    }

    /// Retire the newest immutable generation selected at the privileged
    /// installation boundary. This operation is unavailable after sealing;
    /// ordinary executable holders possess no equivalent name-based control
    /// surface.
    pub async fn retire_generation(
        &mut self,
        target: kore_abi::image::ImageName,
    ) -> Result<(), InstallError> {
        self.ensure_no_discarded_wake()?;
        let reply = self
            .pair
            .call(&TaskRequest::RetireGeneration { target })
            .await
            .map_err(InstallError::Call)?;
        match reply.response {
            TaskResponse::GenerationRetired
                if reply.grants == 0
                    && reply.delivered.iter().all(Option::is_none)
                    && reply.retained.iter().all(Option::is_none) =>
            {
                Ok(())
            }
            TaskResponse::Refused(refusal)
                if reply.grants == 0
                    && reply.delivered.iter().all(Option::is_none)
                    && reply.retained.iter().all(Option::is_none) =>
            {
                Err(InstallError::Refused(refusal))
            }
            _ => Err(InstallError::Call(CallError::Wire)),
        }
    }

    /// Resolve one privileged image name from the boot archive, then return an
    /// ordinary generation-bound executable capability whose invoke wire has no
    /// name or numeric selector.
    pub async fn install(
        &mut self,
        target: kore_abi::image::ImageName,
        profile: u32,
    ) -> Result<ExecutableCapability, InstallError> {
        self.install_inner(target, profile, false).await
    }

    /// Continue only the exact archive [`Self::install`] request already
    /// published on this pair. This method never publishes a replacement.
    pub async fn resume_install(
        &mut self,
        target: kore_abi::image::ImageName,
        profile: u32,
    ) -> Result<ExecutableCapability, InstallError> {
        self.resume_install_inner(target, profile, false).await
    }

    /// Resolve one privileged image name from the durable executable store
    /// (`app/`): exec loads the immutable filesystem revision on demand and
    /// registers it, shadowing any archive entry of the same name. A
    /// tree-resident executable carries no archive authentication, so it always
    /// installs with launch profile zero.
    pub async fn install_from_tree(
        &mut self,
        target: kore_abi::image::ImageName,
    ) -> Result<ExecutableCapability, InstallError> {
        self.install_inner(target, 0, true).await
    }

    /// Continue only the exact [`Self::install_from_tree`] request already
    /// published on this pair. This method never publishes a replacement.
    pub async fn resume_install_from_tree(
        &mut self,
        target: kore_abi::image::ImageName,
    ) -> Result<ExecutableCapability, InstallError> {
        self.resume_install_inner(target, 0, true).await
    }

    /// Acquire one durable-store revision as an execute capability — the
    /// `app/<name>` command spelling, callable on ANY exec client face
    /// (seal-exempt; the service refuses bootstrap-prefixed names, and no
    /// global name is bound or shadowed).  The reply's profile is the
    /// granted generation's own launch metadata: an unchanged revision
    /// dedups onto the resident, archive-authenticated generation and keeps
    /// its profile; a genuinely fresh tree revision arrives at profile 0.
    pub async fn tree_acquire(
        &mut self,
        target: kore_abi::image::ImageName,
    ) -> Result<ExecutableCapability, InstallError> {
        self.ensure_no_discarded_wake()?;
        let request = TimedOutBootstrapRequest::TreeAcquire { target };
        let result = self
            .pair
            .call_bounded(
                &TaskRequest::TreeAcquire { target },
                BOOTSTRAP_REPLY_BUDGET_NS,
            )
            .await;
        let reply = self.finish_bootstrap_wait(request, result)?;
        self.finish_tree_acquire_reply(reply)
    }

    /// Continue only the exact [`Self::tree_acquire`] request already
    /// published on this pair.
    pub async fn resume_tree_acquire(
        &mut self,
        target: kore_abi::image::ImageName,
    ) -> Result<ExecutableCapability, InstallError> {
        let reply = self
            .resume_timed_out_bootstrap(TimedOutBootstrapRequest::TreeAcquire { target })
            .await?;
        self.finish_tree_acquire_reply(reply)
    }

    fn finish_tree_acquire_reply(
        &mut self,
        mut reply: CallReply<TaskResponse>,
    ) -> Result<ExecutableCapability, InstallError> {
        match reply.response {
            TaskResponse::Refused(refusal)
                if reply.grants == 0
                    && reply.delivered.iter().all(Option::is_none)
                    && reply.retained.iter().all(Option::is_none) =>
            {
                return Err(InstallError::Refused(refusal));
            }
            TaskResponse::TreeAcquired { .. }
                if reply.grants == 1
                    && reply.delivered.iter().all(Option::is_none)
                    && reply.retained[1..].iter().all(Option::is_none) => {}
            _ => {
                self.pair.poison_retained(reply.retained);
                return Err(InstallError::Call(CallError::Wire));
            }
        }
        let (segments, profile) = match reply.response {
            TaskResponse::TreeAcquired { segments, profile } => (segments, profile),
            _ => unreachable!(),
        };
        let Some(returned) = reply.retained[0].take() else {
            self.pair.poison_retained(reply.retained);
            return Err(InstallError::Call(CallError::Wire));
        };
        let endpoint = match returned.try_into_chan_send(Rights::ACCESS) {
            Ok(endpoint) => endpoint,
            Err(returned) => {
                reply.retained[0] = Some(returned);
                self.pair.poison_retained(reply.retained);
                return Err(InstallError::Call(CallError::Wire));
            }
        };
        Ok(ExecutableCapability {
            endpoint: ExecutableEndpoint::Owned(endpoint),
            segments,
            profile,
        })
    }

    fn install_request(
        &self,
        target: kore_abi::image::ImageName,
        profile: u32,
        from_tree: bool,
    ) -> Result<(TaskRequest, bool), InstallError> {
        let context_empty = self.proof_tx.is_none()
            && self.process_tx.is_none()
            && self.fs_view_tx.is_none()
            && self.auxiliary_tx.iter().all(Option::is_none);
        let context_complete = self.proof_tx.is_some()
            && self.process_tx.is_some()
            && self.fs_view_tx.is_some()
            && self.auxiliary_tx.iter().all(Option::is_some);
        if !context_empty && !context_complete {
            return Err(InstallError::Call(CallError::Poisoned));
        }
        Ok((
            TaskRequest::Install {
                target,
                profile,
                include_context: context_empty,
                from_tree,
            },
            context_empty,
        ))
    }

    async fn install_inner(
        &mut self,
        target: kore_abi::image::ImageName,
        profile: u32,
        from_tree: bool,
    ) -> Result<ExecutableCapability, InstallError> {
        self.ensure_no_discarded_wake()?;
        let (request, include_context) = self.install_request(target, profile, from_tree)?;
        let identity = TimedOutBootstrapRequest::Install {
            target,
            profile,
            include_context,
            from_tree,
        };
        let result = self
            .pair
            .call_bounded(&request, BOOTSTRAP_REPLY_BUDGET_NS)
            .await;
        let reply = self.finish_bootstrap_wait(identity, result)?;
        self.finish_install_reply(reply, profile, include_context)
    }

    async fn resume_install_inner(
        &mut self,
        target: kore_abi::image::ImageName,
        profile: u32,
        from_tree: bool,
    ) -> Result<ExecutableCapability, InstallError> {
        let (_, include_context) = self.install_request(target, profile, from_tree)?;
        let identity = TimedOutBootstrapRequest::Install {
            target,
            profile,
            include_context,
            from_tree,
        };
        let reply = self.resume_timed_out_bootstrap(identity).await?;
        self.finish_install_reply(reply, profile, include_context)
    }

    fn finish_install_reply(
        &mut self,
        mut reply: CallReply<TaskResponse>,
        profile: u32,
        include_context: bool,
    ) -> Result<ExecutableCapability, InstallError> {
        let expected_grants = if include_context {
            4 + MAX_INVOCATION_AUXILIARIES
        } else {
            1
        };
        match reply.response {
            TaskResponse::Refused(refusal)
                if reply.grants == 0
                    && reply.delivered.iter().all(Option::is_none)
                    && reply.retained.iter().all(Option::is_none) =>
            {
                return Err(InstallError::Refused(refusal));
            }
            TaskResponse::Installed { .. }
                if reply.grants == expected_grants as u8
                    && reply.delivered.iter().all(Option::is_none)
                    && reply.retained[expected_grants..]
                        .iter()
                        .all(Option::is_none) => {}
            _ => {
                self.pair.poison_retained(reply.retained);
                return Err(InstallError::Call(CallError::Wire));
            }
        }
        let segments = match reply.response {
            TaskResponse::Installed { segments } => segments,
            _ => unreachable!(),
        };

        macro_rules! take_returned {
            ($index:expr) => {{
                let Some(returned) = reply.retained[$index].take() else {
                    self.pair.poison_retained(reply.retained);
                    return Err(InstallError::Call(CallError::Wire));
                };
                returned
            }};
        }

        if include_context {
            let returned = take_returned!(0);
            let proof_tx = match returned.try_into_chan_send(Rights::ACCESS) {
                Ok(endpoint) => endpoint,
                Err(returned) => {
                    reply.retained[0] = Some(returned);
                    self.pair.poison_retained(reply.retained);
                    return Err(InstallError::Call(CallError::Wire));
                }
            };
            let returned = take_returned!(1);
            let process_tx = match returned.try_into_chan_send(Rights::ACCESS) {
                Ok(endpoint) => endpoint,
                Err(returned) => {
                    reply.retained[1] = Some(returned);
                    self.pair.poison_retained(reply.retained);
                    return Err(InstallError::Call(CallError::Wire));
                }
            };
            let returned = take_returned!(2);
            let fs_view_tx = match returned
                .try_into_chan_send(kore_svc_dagfs::FsViewCapability::DELEGABLE_RIGHTS)
            {
                Ok(endpoint) => endpoint,
                Err(returned) => {
                    reply.retained[2] = Some(returned);
                    self.pair.poison_retained(reply.retained);
                    return Err(InstallError::Call(CallError::Wire));
                }
            };
            let mut auxiliary_tx = [const { None }; MAX_INVOCATION_AUXILIARIES];
            for (index, destination) in auxiliary_tx.iter_mut().enumerate() {
                let returned = take_returned!(3 + index);
                match returned.try_into_chan_send(Rights::ACCESS) {
                    Ok(endpoint) => *destination = Some(endpoint),
                    Err(returned) => {
                        reply.retained[3 + index] = Some(returned);
                        self.pair.poison_retained(reply.retained);
                        return Err(InstallError::Call(CallError::Wire));
                    }
                }
            }
            self.proof_tx = Some(InvocationEndpoint::Owned(proof_tx));
            self.process_tx = Some(InvocationEndpoint::Owned(process_tx));
            self.fs_view_tx = Some(InvocationEndpoint::Owned(fs_view_tx));
            self.auxiliary_tx =
                auxiliary_tx.map(|endpoint| Some(InvocationEndpoint::Owned(endpoint.unwrap())));
        }
        let endpoint_index = if include_context {
            3 + MAX_INVOCATION_AUXILIARIES
        } else {
            0
        };
        let returned = take_returned!(endpoint_index);
        let endpoint = match returned.try_into_chan_send(Rights::ACCESS) {
            Ok(endpoint) => endpoint,
            Err(returned) => {
                reply.retained[endpoint_index] = Some(returned);
                self.pair.poison_retained(reply.retained);
                return Err(InstallError::Call(CallError::Wire));
            }
        };
        Ok(ExecutableCapability {
            endpoint: ExecutableEndpoint::Owned(endpoint),
            segments,
            profile,
        })
    }

    /// Invoke one borrowed generation capability over this task's fixed
    /// generic exec context. No name or generation index crosses the wire.
    pub async fn invoke<A: ReturnCapability, B: ReturnCapability, C: ReturnCapability>(
        &mut self,
        executable: &ExecutableCapability,
        spec: SpawnSpec,
        first: A,
        second: B,
        third: C,
    ) -> Result<SpawnOutcome, InvokeError<A, B, C>> {
        self.invoke_with_auxiliaries(
            executable,
            spec,
            first,
            second,
            third,
            [const { None }; MAX_INVOCATION_AUXILIARIES],
        )
        .await
    }

    /// Invoke with a bounded set of explicitly delegated auxiliary
    /// capabilities. Every live member moves with supply, ProcessIo, cwd,
    /// admission, and result custody in the same kernel session.
    pub async fn invoke_with_auxiliaries<
        A: ReturnCapability,
        B: ReturnCapability,
        C: ReturnCapability,
    >(
        &mut self,
        executable: &ExecutableCapability,
        spec: SpawnSpec,
        first: A,
        second: B,
        third: C,
        auxiliaries: [Option<InvocationAuxiliary>; MAX_INVOCATION_AUXILIARIES],
    ) -> Result<SpawnOutcome, InvokeError<A, B, C>> {
        self.invoke_with_handle_policy(
            executable,
            spec,
            first,
            second,
            third,
            auxiliaries,
            false,
            None,
        )
        .await
    }

    /// Invoke with a parent handle that may attenuate lifecycle observers into
    /// sibling workers. This authority is opt-in, the service gates it against
    /// the executable's authenticated launch profile, and ordinary invocation
    /// remains non-delegable.
    pub async fn invoke_delegable_with_auxiliaries<
        A: ReturnCapability,
        B: ReturnCapability,
        C: ReturnCapability,
    >(
        &mut self,
        executable: &ExecutableCapability,
        spec: SpawnSpec,
        first: A,
        second: B,
        third: C,
        auxiliaries: [Option<InvocationAuxiliary>; MAX_INVOCATION_AUXILIARIES],
    ) -> Result<SpawnOutcome, InvokeError<A, B, C>> {
        self.invoke_with_handle_policy(
            executable,
            spec,
            first,
            second,
            third,
            auxiliaries,
            true,
            None,
        )
        .await
    }

    /// Observe a newly committed invocation until the supplied executor deadline. Timeout
    /// preserves the original request; only its owner may resume it below.
    pub async fn invoke_with_auxiliaries_until<
        A: ReturnCapability,
        B: ReturnCapability,
        C: ReturnCapability,
    >(
        &mut self,
        executable: &ExecutableCapability,
        spec: SpawnSpec,
        first: A,
        second: B,
        third: C,
        auxiliaries: [Option<InvocationAuxiliary>; MAX_INVOCATION_AUXILIARIES],
        deadline_ns: u64,
    ) -> Result<SpawnOutcome, InvokeError<A, B, C>> {
        self.invoke_with_handle_policy(
            executable,
            spec,
            first,
            second,
            third,
            auxiliaries,
            false,
            Some(deadline_ns),
        )
        .await
    }

    /// Recover the sole pending invocation on the same held pair used to
    /// submit it. Call only from that invocation's retained lifecycle owner.
    /// Rights and reply shape come from the original encoded request; a
    /// different operation is refused before any transport or ring progress.
    pub async fn resume_invoke_bounded(
        &mut self,
        budget_ns: u64,
    ) -> Result<SpawnOutcome, CallError> {
        if self.discarded_wake.is_some() {
            return if retire_discarded(&mut self.discarded_wake, ReturnedCapability::try_retire) {
                Ok(SpawnOutcome::Refused(TaskRefusal::Unfounded))
            } else {
                Err(CallError::Timeout)
            };
        }
        let Some(request) = self.pair.outstanding_request().copied() else {
            // The generic owner also retries retained late-grant cleanup when
            // poisoned. Absence of correlation must not bypass that custody.
            self.pair.resume_bounded(budget_ns).await?;
            unreachable!("an idle or poisoned pair cannot complete a reply");
        };
        let Some(TaskRequest::Invoke(spec)) = crate::wire::decode_req(&request) else {
            return Err(CallError::Outstanding);
        };
        let reply = self
            .pair
            .resume_exact_bounded(&TaskRequest::Invoke(spec), budget_ns)
            .await?;
        finish_invoke_reply(
            &mut self.pair,
            reply,
            if spec.delegable_handle {
                crate::wire::DELEGABLE_HANDLE_RIGHTS
            } else {
                crate::wire::HANDLE_RIGHTS
            },
            spec.return_shared_wake,
            &mut self.discarded_wake,
        )
    }

    async fn invoke_with_handle_policy<
        A: ReturnCapability,
        B: ReturnCapability,
        C: ReturnCapability,
    >(
        &mut self,
        executable: &ExecutableCapability,
        mut spec: SpawnSpec,
        first: A,
        second: B,
        third: C,
        auxiliaries: [Option<InvocationAuxiliary>; MAX_INVOCATION_AUXILIARIES],
        delegable_handle: bool,
        deadline_ns: Option<u64>,
    ) -> Result<SpawnOutcome, InvokeError<A, B, C>> {
        if self.discarded_wake.is_some() {
            return Err(InvokeError::NotMoved {
                error: CallError::Outstanding,
                first,
                second,
                third,
                auxiliaries,
            });
        }
        if spec.boot_write_idx & kore_abi::svc::BOOT_FROM_SUPPLY == 0 {
            return Err(InvokeError::NotMoved {
                error: CallError::Refused(Status::Malformed),
                first,
                second,
                third,
                auxiliaries,
            });
        }
        spec.return_handle = true;
        spec.linked = true;
        spec.delegable_handle = delegable_handle;
        let (Some(proof_tx), Some(process_tx), Some(fs_view_tx)) =
            (&self.proof_tx, &self.process_tx, &self.fs_view_tx)
        else {
            return Err(InvokeError::NotMoved {
                error: CallError::Poisoned,
                first,
                second,
                third,
                auxiliaries,
            });
        };
        let auxiliary_tx = core::array::from_fn(|index| {
            self.auxiliary_tx[index]
                .as_ref()
                .map(InvocationEndpoint::slot)
        });
        if auxiliary_tx.iter().any(Option::is_none) {
            return Err(InvokeError::NotMoved {
                error: CallError::Poisoned,
                first,
                second,
                third,
                auxiliaries,
            });
        }
        let auxiliary_tx = auxiliary_tx.map(Option::unwrap);
        match self.pair.begin_invocation_bundle(
            &TaskRequest::Invoke(spec),
            first,
            kore_svc_mm::SUPPLY_RIGHTS,
            executable.endpoint.slot(),
            second,
            kore_svc_mm::SUPPLY_RIGHTS,
            process_tx.slot(),
            proof_tx.slot(),
            third,
            // The shell-owned source carries DERIVE only so the
            // rendezvous can mint this leaf. Exec receives the structural
            // cwd authority attenuated to ACCESS|REVOKE and cannot
            // delegate a wider view onward.
            kore_svc_dagfs::FsViewCapability::INVOCATION_RIGHTS,
            fs_view_tx.slot(),
            auxiliaries,
            auxiliary_tx,
        ) {
            Ok(()) => {}
            Err(AffineBundleCallError::NotMoved {
                error,
                first,
                second,
                third,
                auxiliaries,
            }) => {
                return Err(InvokeError::NotMoved {
                    error,
                    first,
                    second,
                    third,
                    auxiliaries,
                });
            }
            Err(AffineBundleCallError::AfterTransfer(error)) => {
                return Err(InvokeError::AfterTransfer(error));
            }
        };
        let reply = match deadline_ns {
            Some(deadline) => self.pair.resume_until(deadline).await,
            None => self.pair.resume().await,
        }
        .map_err(InvokeError::AfterTransfer)?;
        let handle_rights = if delegable_handle {
            crate::wire::DELEGABLE_HANDLE_RIGHTS
        } else {
            crate::wire::HANDLE_RIGHTS
        };
        finish_invoke_reply(
            &mut self.pair,
            reply,
            handle_rights,
            spec.return_shared_wake,
            &mut self.discarded_wake,
        )
        .map_err(InvokeError::AfterTransfer)
    }
}

#[cfg(test)]
mod tests {
    use core::cell::Cell;

    use super::*;
    use kore_abi::session::Committed;
    use kore_rt::{
        service::MAX_GRANTS,
        session::SessionBuilder,
        svc::{REP_CAP, REQ_CAP, ServeRings, SvcProtocol, split_window, window_bytes},
        transport as rt_transport,
    };

    struct RecordingTransport {
        commits: Cell<usize>,
        succeed: Cell<bool>,
    }

    impl RecordingTransport {
        const fn refusing() -> Self {
            Self {
                commits: Cell::new(0),
                succeed: Cell::new(false),
            }
        }

        fn complete(&self) -> Result<Committed, ()> {
            self.commits.set(self.commits.get() + 1);
            if self.succeed.get() {
                Ok(Committed::from_raw(Status::Ok.encode(0)))
            } else {
                Err(())
            }
        }
    }

    impl rt_transport::Transport for RecordingTransport {
        type Error = ();

        async fn commit(&self, _session: &SessionBuilder) -> Result<Committed, Self::Error> {
            self.complete()
        }

        fn commit_now(&self, _session: &SessionBuilder) -> Result<Committed, Self::Error> {
            self.complete()
        }
    }

    #[repr(align(64))]
    struct ExecRingWindow([u8; window_bytes::<ExecProtocol>(REQ_CAP, REP_CAP)]);

    fn ring_exec<'w, 't>(
        transport: &'t RecordingTransport,
        window: &'w mut ExecRingWindow,
    ) -> (
        RingExec<'w, 't, RecordingTransport>,
        ServeRings<'w, ExecProtocol>,
    ) {
        let (client, serve) =
            split_window::<ExecProtocol>(&mut window.0, REQ_CAP, REP_CAP).expect("exec rings");
        let pair = SvcClient::new(
            transport,
            client,
            ClientEndpoints {
                kick: SlotRef::from_wire(0x30),
                grants: [None; MAX_GRANTS],
            },
        );
        (RingExec::new(pair), serve)
    }

    fn run<F: core::future::Future>(future: F) -> F::Output {
        // Bounded reply observations read the executor's event-driven clock,
        // even when the mocked transport refuses before the future can park.
        // Adopt a dummy already-standing doorbell so these host tests exercise
        // that production waker contract without issuing a host-side park.
        let park = kore_rt::transport::TrapTransport::new();
        let doorbell = kore_rt::executor::Doorbell {
            chan: 0,
            recv: SlotRef::from_wire(0x31),
            land: SlotRef::from_wire(0x32),
            recv_tag: 0x33,
        };
        kore_rt::Executor::from_standing(&park, doorbell).run(future)
    }

    #[test]
    fn unfounded_cleanup_preserves_the_same_owner_until_retirement_confirms() {
        struct Owner<'a> {
            attempts: usize,
            drops: &'a Cell<usize>,
        }
        impl Drop for Owner<'_> {
            fn drop(&mut self) {
                self.drops.set(self.drops.get() + 1);
            }
        }
        let drops = Cell::new(0);
        let mut owner = Some(Owner {
            attempts: 0,
            drops: &drops,
        });
        for attempts in 1..=2 {
            assert!(!retire_discarded(&mut owner, |owner| {
                owner.attempts += 1;
                false
            }));
            assert_eq!(owner.as_ref().unwrap().attempts, attempts);
            assert_eq!(drops.get(), 0);
        }
        assert!(retire_discarded(&mut owner, |owner| {
            assert_eq!(owner.attempts, 2);
            true
        }));
        assert!(owner.is_none());
        assert_eq!(drops.get(), 1);
        assert!(retire_discarded(&mut owner, |_| panic!(
            "no remaining owner"
        )));
        assert_eq!(drops.get(), 1);
    }

    #[test]
    fn unfounded_without_wake_finishes_without_claiming_a_child() {
        let transport = RecordingTransport::refusing();
        let mut window = ExecRingWindow([0; window_bytes::<ExecProtocol>(REQ_CAP, REP_CAP)]);
        let (mut exec, mut serve) = ring_exec(&transport, &mut window);
        let reply = CallReply {
            response: TaskResponse::Refused(TaskRefusal::Unfounded),
            delivered: [None; MAX_GRANTS],
            retained: [const { None }; MAX_GRANTS],
            grants: 0,
        };
        assert!(matches!(
            finish_invoke_reply(
                &mut exec.pair,
                reply,
                crate::wire::HANDLE_RIGHTS,
                false,
                &mut exec.discarded_wake
            ),
            Ok(SpawnOutcome::Refused(TaskRefusal::Unfounded))
        ));
        assert!(exec.is_idle());
        assert_eq!(transport.commits.get(), 0);
        assert!(serve.req.pop().is_none());
        assert!(matches!(
            run(exec.resume_invoke_bounded(1)),
            Err(CallError::NoOutstanding)
        ));
    }

    #[test]
    fn unfounded_missing_wake_or_wrong_grant_count_poison_the_pair() {
        for (shared_wake, grants) in [(true, 1), (true, 0), (true, 2), (false, 1)] {
            let transport = RecordingTransport::refusing();
            let mut window = ExecRingWindow([0; window_bytes::<ExecProtocol>(REQ_CAP, REP_CAP)]);
            let (mut exec, mut serve) = ring_exec(&transport, &mut window);
            let reply = CallReply {
                response: TaskResponse::Refused(TaskRefusal::Unfounded),
                delivered: [None; MAX_GRANTS],
                retained: [const { None }; MAX_GRANTS],
                grants,
            };
            assert!(matches!(
                finish_invoke_reply(
                    &mut exec.pair,
                    reply,
                    crate::wire::HANDLE_RIGHTS,
                    shared_wake,
                    &mut exec.discarded_wake
                ),
                Err(CallError::Wire)
            ));
            assert!(!exec.is_idle());
            assert!(matches!(
                run(exec.resume_invoke_bounded(1)),
                Err(CallError::Poisoned)
            ));
            assert_eq!(transport.commits.get(), 0);
            assert!(serve.req.pop().is_none());
        }
    }

    #[test]
    fn shared_wake_reply_contract_is_two_results_with_nondelegable_rights() {
        assert_eq!(invoke_result_grants(false), 1);
        assert_eq!(invoke_result_grants(true), 2);
        assert_eq!(SHARED_WAKE_RIGHTS, Rights::ACCESS | Rights::REVOKE);
        assert!(!SHARED_WAKE_RIGHTS.contains(Rights::DERIVE));
    }

    #[test]
    fn malformed_two_result_reply_without_affine_owners_poisoned() {
        let transport = RecordingTransport::refusing();
        let mut window = ExecRingWindow([0; window_bytes::<ExecProtocol>(REQ_CAP, REP_CAP)]);
        let (mut exec, _) = ring_exec(&transport, &mut window);
        let reply = CallReply {
            response: TaskResponse::Spawned {
                id: 0,
                segments: 1,
                snapshot: kore_abi::monitor::TelemetrySnapshot::EMPTY,
                filesystem: kore_abi::monitor::FilesystemSnapshot::EMPTY,
                net: kore_abi::monitor::NetSnapshot::EMPTY,
            },
            delivered: [None; MAX_GRANTS],
            retained: [const { None }; MAX_GRANTS],
            grants: invoke_result_grants(true),
        };

        assert!(matches!(
            finish_invoke_reply(
                &mut exec.pair,
                reply,
                crate::wire::HANDLE_RIGHTS,
                true,
                &mut exec.discarded_wake
            ),
            Err(CallError::Wire)
        ));
        assert!(matches!(
            run(exec.pair.call(&TaskRequest::SealInstaller)),
            Err(CallError::Poisoned)
        ));
    }

    #[test]
    fn typed_bootstrap_resume_rejects_a_different_verb_or_name_before_progress() {
        let transport = RecordingTransport::refusing();
        let mut window = ExecRingWindow([0; window_bytes::<ExecProtocol>(REQ_CAP, REP_CAP)]);
        let (mut exec, mut serve) = ring_exec(&transport, &mut window);
        let hello = kore_abi::image::ImageName::new("hello").unwrap();
        let ls = kore_abi::image::ImageName::new("ls").unwrap();

        assert_eq!(
            run(exec.seed_app(hello)),
            Err(InstallError::Call(CallError::Transport))
        );
        assert_eq!(transport.commits.get(), 1);
        let request = serve.req.pop().expect("published seed request");
        assert!(matches!(
            <ExecProtocol as SvcProtocol>::decode_req(&request),
            Some(TaskRequest::SeedApp { target }) if target == hello
        ));

        // Model the bounded collector returning Timeout after publication. The
        // real collector performs exactly this state transition; the fake
        // transport above gives the host test a deterministic published pair.
        assert!(matches!(
            exec.finish_bootstrap_wait(
                TimedOutBootstrapRequest::SeedApp { target: hello },
                Err(CallError::Timeout),
            ),
            Err(InstallError::Call(CallError::Timeout))
        ));
        let reply =
            <ExecProtocol as SvcProtocol>::encode_rep(&TaskResponse::AppSeeded { seeded: true, failure: 0 }, 0);
        assert!(serve.rep.push(reply).is_ok());

        assert_eq!(
            run(exec.resume_seed_app(ls)),
            Err(InstallError::Call(CallError::Outstanding))
        );
        assert_eq!(
            run(exec.resume_install(hello, 0)).map(|_| ()),
            Err(InstallError::Call(CallError::Outstanding))
        );
        assert_eq!(
            transport.commits.get(),
            1,
            "a mismatched typed resume must not touch the transport"
        );

        transport.succeed.set(true);
        assert_eq!(
            run(exec.resume_seed_app(hello)),
            Ok(AppSeedOutcome { seeded: true, failure: 0 })
        );
        assert_eq!(transport.commits.get(), 2);
        assert!(exec.timed_out_bootstrap.is_none());
    }

    #[test]
    fn invoke_recovery_cannot_consume_another_operations_late_reply() {
        let transport = RecordingTransport::refusing();
        let mut window = ExecRingWindow([0; window_bytes::<ExecProtocol>(REQ_CAP, REP_CAP)]);
        let (mut exec, mut serve) = ring_exec(&transport, &mut window);
        let target = kore_abi::image::ImageName::new("hello").unwrap();
        assert!(matches!(
            run(exec.resume_invoke_bounded(1)),
            Err(CallError::NoOutstanding)
        ));
        assert_eq!(transport.commits.get(), 0);
        assert_eq!(
            run(exec.seed_app(target)),
            Err(InstallError::Call(CallError::Transport))
        );
        let original = serve.req.pop().unwrap();
        serve
            .rep
            .push(ExecProtocol::encode_rep(
                &TaskResponse::AppSeeded { seeded: true, failure: 0 },
                0,
            ))
            .unwrap();
        let commits = transport.commits.get();
        for _ in 0..2 {
            assert!(matches!(
                run(exec.resume_invoke_bounded(1)),
                Err(CallError::Outstanding)
            ));
            assert_eq!(exec.pair.outstanding_request(), Some(&original));
            assert_eq!(transport.commits.get(), commits);
            assert!(serve.req.pop().is_none());
        }
        // The original owner still gets its exact queued reply.
        transport.succeed.set(true);
        let reply = run(exec
            .pair
            .resume_exact_bounded(&TaskRequest::SeedApp { target }, 1))
        .unwrap();
        assert!(matches!(
            reply.response,
            TaskResponse::AppSeeded { seeded: true, failure: 0 }
        ));
        assert!(exec.is_idle());
        assert!(serve.req.pop().is_none());
    }

    #[test]
    fn invoke_recovery_preserves_poison_and_does_not_claim_a_missing_child() {
        let transport = RecordingTransport::refusing();
        let mut window = ExecRingWindow([0; window_bytes::<ExecProtocol>(REQ_CAP, REP_CAP)]);
        let (mut exec, mut serve) = ring_exec(&transport, &mut window);
        exec.pair.poison();
        assert!(matches!(
            run(exec.resume_invoke_bounded(1)),
            Err(CallError::Poisoned)
        ));
        assert!(!exec.is_idle());
        assert_eq!(transport.commits.get(), 0);
        assert!(serve.req.pop().is_none());
    }

    #[test]
    fn typed_seed_timeout_state_survives_two_slices_without_republishing() {
        let transport = RecordingTransport::refusing();
        let mut window = ExecRingWindow([0; window_bytes::<ExecProtocol>(REQ_CAP, REP_CAP)]);
        let (mut exec, mut serve) = ring_exec(&transport, &mut window);
        let hello = kore_abi::image::ImageName::new("hello").unwrap();
        let request = TimedOutBootstrapRequest::SeedApp { target: hello };

        assert_eq!(
            run(exec.seed_app(hello)),
            Err(InstallError::Call(CallError::Transport))
        );
        assert_eq!(transport.commits.get(), 1);
        let published = serve.req.pop().expect("published seed request");
        assert!(matches!(
            <ExecProtocol as SvcProtocol>::decode_req(&published),
            Some(TaskRequest::SeedApp { target }) if target == hello
        ));

        for _ in 0..2 {
            assert!(matches!(
                exec.finish_bootstrap_wait(request, Err(CallError::Timeout)),
                Err(InstallError::Call(CallError::Timeout))
            ));
            assert_eq!(exec.timed_out_bootstrap, Some(request));
            assert!(
                serve.req.pop().is_none(),
                "a typed timeout must leave the original request outstanding"
            );
        }

        let reply =
            <ExecProtocol as SvcProtocol>::encode_rep(&TaskResponse::AppSeeded { seeded: true, failure: 0 }, 0);
        assert!(serve.rep.push(reply).is_ok());
        transport.succeed.set(true);

        assert_eq!(
            run(exec.resume_seed_app(hello)),
            Ok(AppSeedOutcome { seeded: true, failure: 0 })
        );
        assert_eq!(transport.commits.get(), 2);
        assert!(serve.req.pop().is_none());
        assert!(exec.timed_out_bootstrap.is_none());
    }

    #[test]
    fn install_resume_identity_includes_context_profile_and_source() {
        let target = kore_abi::image::ImageName::new("hello").unwrap();
        let archive = TimedOutBootstrapRequest::Install {
            target,
            profile: 7,
            include_context: true,
            from_tree: false,
        };
        for different in [
            TimedOutBootstrapRequest::Install {
                target,
                profile: 8,
                include_context: true,
                from_tree: false,
            },
            TimedOutBootstrapRequest::Install {
                target,
                profile: 7,
                include_context: false,
                from_tree: false,
            },
            TimedOutBootstrapRequest::Install {
                target,
                profile: 7,
                include_context: true,
                from_tree: true,
            },
        ] {
            assert_ne!(archive, different);
        }
    }

    #[test]
    fn task_local_names_are_metadata_not_authority() {
        fn name(value: &str) -> kore_abi::image::ImageName {
            kore_abi::image::ImageName::new(value).unwrap()
        }
        let metadata = |local: &str, display: &str| ExecutableMetadata {
            local_name: name(local),
            display_name: name(display),
            usage: b"tool [args]",
            summary: b"inert help",
            context: ExecutableContext::None,
        };

        let mut first = CapabilityNamespace::<u8, 2>::new();
        let mut second = CapabilityNamespace::<u8, 2>::new();
        first.bind(metadata("tool", "generation-a"), 11).unwrap();
        second.bind(metadata("tool", "generation-b"), 29).unwrap();
        assert_eq!(first.resolve(name("tool")), Some(&11));
        assert_eq!(second.resolve(name("tool")), Some(&29));

        first.rename(name("tool"), name("renamed")).unwrap();
        assert_eq!(first.resolve(name("tool")), None);
        assert_eq!(first.resolve(name("renamed")), Some(&11));
        let listed: alloc::vec::Vec<_> = first.metadata().collect();
        assert_eq!(listed, [metadata("renamed", "generation-a")]);
        assert_eq!(
            first.describe(name("renamed")),
            Some(metadata("renamed", "generation-a"))
        );
    }

    #[test]
    fn namespace_refusals_return_the_unmoved_affine_member() {
        let metadata = |local: &str| ExecutableMetadata {
            local_name: kore_abi::image::ImageName::new(local).unwrap(),
            display_name: kore_abi::image::ImageName::new(local).unwrap(),
            usage: b"tool",
            summary: b"inert help",
            context: ExecutableContext::None,
        };
        let mut namespace = CapabilityNamespace::<u8, 1>::new();
        namespace.bind(metadata("one"), 7).unwrap();
        assert_eq!(
            namespace.bind(metadata("one"), 8),
            Err((NamespaceError::Duplicate, 8))
        );
        assert_eq!(
            namespace.bind(metadata("two"), 9),
            Err((NamespaceError::Full, 9))
        );
    }

    #[test]
    fn namespace_unbind_returns_affine_custody_without_retiring_it() {
        struct Witness<'a> {
            uses: u8,
            dropped: &'a core::cell::Cell<u8>,
        }
        impl Drop for Witness<'_> {
            fn drop(&mut self) {
                self.dropped.set(self.dropped.get() + 1);
            }
        }
        let dropped = core::cell::Cell::new(0);
        let name = kore_abi::image::ImageName::new("read").unwrap();
        let renamed = kore_abi::image::ImageName::new("inspect").unwrap();
        let metadata = ExecutableMetadata {
            local_name: name,
            display_name: name,
            usage: b"read path",
            summary: b"read a scoped file",
            context: ExecutableContext::None,
        };
        let mut namespace = CapabilityNamespace::<Witness<'_>, 1>::new();
        assert!(
            namespace
                .bind(
                    metadata,
                    Witness {
                        uses: 2,
                        dropped: &dropped
                    }
                )
                .is_ok()
        );
        namespace.resolve_mut(name).unwrap().uses -= 1;
        namespace.rename(name, renamed).unwrap();
        assert!(namespace.unbind(name).is_none());
        assert_eq!(dropped.get(), 0);
        let witness = namespace.unbind(renamed).unwrap();
        assert_eq!(witness.uses, 1);
        assert_eq!(dropped.get(), 0);
        assert!(namespace.resolve(renamed).is_none());
        assert!(namespace.describe(renamed).is_none());
        assert_eq!(namespace.metadata().count(), 0);
        assert!(namespace.unbind(renamed).is_none());
        // The vacated row is reusable, while the removed capability remains
        // owned by the caller for asynchronous native revocation retry.
        assert!(
            namespace
                .bind(
                    metadata,
                    Witness {
                        uses: 3,
                        dropped: &dropped
                    }
                )
                .is_ok()
        );
        drop(namespace);
        assert_eq!(dropped.get(), 1);
        drop(witness);
        assert_eq!(dropped.get(), 2);
    }

    #[test]
    fn execute_only_authority_cannot_authorize_delegation() {
        assert!(
            ExecutableCapability::EXECUTE_ONLY_RIGHTS.contains(Rights::ACCESS),
            "execute authority must remain invocable"
        );
        assert!(
            !ExecutableCapability::EXECUTE_ONLY_RIGHTS.contains(Rights::DERIVE),
            "the child namespace receives no delegation authority"
        );
        assert!(
            kore_rt::cap::CHANNEL_ENDPOINT_RIGHTS
                .contains(ExecutableCapability::EXECUTE_ONLY_RIGHTS | Rights::DERIVE),
            "a privileged parent may explicitly retain the distinct delegation right"
        );
    }

}
