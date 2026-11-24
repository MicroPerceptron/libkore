//! `client` — the accel client face for peers (`res/USERLAND.md` §4).
//!
//! [`RingAccel`] is the born-async ring + channel proxy over the pair's
//! shared window, generic over the transport seam and fixed to the
//! backend-free [`AccelWire`] protocol marker — a client never imports a
//! hardware backend type to place a call, and byte identity across
//! backends holds by construction (`wire.rs`'s delegation).
//!
//! Three flows matter beyond the plain typed calls:
//!
//! - **Attach** rides the return leg: the caller derives a writable view
//!   of its own staging backing and [`AccelClient::attach`] transfers it
//!   with the request (the swap/image registrar shape).  Attach once.
//! - **Signal** collects the reply's one retained grant, possesses it as
//!   a [`ChanRecvToken`], and hands it back; the caller parks a
//!   [`ChanRx`](kore_rt::chan::ChanRx) on it and treats every
//!   [`COOKIE_ACCEL_DONE`](crate::COOKIE_ACCEL_DONE) notice as "poll
//!   your events" — the poll stays the truth, so a missed notice loses
//!   nothing.
//! - **Submit** answers at the acceptance boundary: an
//!   [`AccelResponse::SubmitIndeterminate`] still names the event that
//!   owns the submission's resources, and the caller must poll/destroy
//!   it like any other.
//!
//! The resumable [`Execution`] lifecycle returns infrastructure errors and
//! retains its continuation. The legacy plain request path **panics** (the
//! executor's transport-failure posture): the closed [`AccelRefusal`]
//! vocabulary has no honest infra arm, and inventing one would let a
//! torn pair masquerade as a device refusal.

mod imports;
pub use imports::{BufferImport, ImportOutcome};

mod buffer_execution;
pub use buffer_execution::{
    BufferExecution, BufferExecutionActivity, BufferExecutionError, ExecutionBuffer, LaunchBuilder,
};

mod handoff;
pub use handoff::{BufferHandoff, HandoffError, HandoffOutcome, HandoffProgress, ImportTarget};

mod execution;
pub use execution::{
    ArtifactExecution, Execution, ExecutionError, ExecutionLedger, ExecutionOutcome,
    ExecutionProgress, ExecutionSet, ExecutionSubmission, LaunchExecution, OrderedExecution,
    OrderedLaunch, ProgramReuse, stage_ordered_executions, stage_ordered_launches,
};

mod objects;
pub use objects::{
    BatchKind, BufferBatch, BufferBatchError, BufferBatchProgress, BufferObjects, ObjectBatch,
    ObjectBatchError, ObjectBatchProgress, ProgramBatch, ProgramObjects,
};

use kore_abi::cap::Rights;
use kore_abi::session::SlotRef;
use kore_rt::cap::{ChanRecvToken, ReturnedCapability};
use kore_rt::svc::{ReturnCallError, ReturnCapability, SvcClient};
use kore_rt::transport::Transport;

#[allow(deprecated)]
use kore_driver::accel::Dispatch;
use kore_driver::accel::Workgroups;

use crate::service::{AccelBindings, AccelRequest, AccelResponse, STAGING_RIGHTS};
use crate::wire::AccelWire;

/// The **client** side of the accel protocol.  Replies are data; the
/// capability movements (staging/import in, completion or lease out) are the
/// explicit methods.
#[allow(async_fn_in_trait)] // clients are generic over the impl (§4)
pub trait AccelClient {
    /// Issue one plain request (anything but `Attach`/`AttachOwned`/`ProvisionMemory`/`ImportBuffer`/`ExportBuffer`/`AdmitPeer`/`Signal`).
    async fn request(&mut self, req: AccelRequest) -> AccelResponse;

    /// Admit one deployed legacy dispatch under a relative `timeout_ns`.
    ///
    /// Compatibility-only: scalar is the historical artifact-defined sentinel;
    /// every non-scalar value retains its exact grid/group/SIMD contract and a
    /// backend that cannot prove it rejects it. New clients should call
    /// [`submit_artifact_defined`](Self::submit_artifact_defined) or
    /// [`submit_workgroups`](Self::submit_workgroups).
    /// The reply names the event at the *acceptance boundary*:
    /// [`AccelResponse::Submitted`], or [`AccelResponse::SubmitIndeterminate`]
    /// which still names the event that owns the submission's resources —
    /// either way the caller polls/destroys it like any other.
    #[allow(deprecated)]
    #[deprecated(
        note = "legacy OP_SUBMIT compatibility only; use submit_artifact_defined or submit_workgroups"
    )]
    async fn submit(
        &mut self,
        queue: u64,
        program: u64,
        bindings: AccelBindings,
        dispatch: Dispatch,
        timeout_ns: u64,
    ) -> AccelResponse {
        self.request(AccelRequest::Submit {
            queue,
            program,
            timeout_ns,
            bindings,
            dispatch,
        })
        .await
    }

    /// Admit one artifact-defined execution. This encodes the deployed scalar
    /// OP13 sentinel so old and new services agree on the same bytes.
    async fn submit_artifact_defined(
        &mut self,
        queue: u64,
        program: u64,
        bindings: AccelBindings,
        timeout_ns: u64,
    ) -> AccelResponse {
        #[allow(deprecated)]
        self.request(AccelRequest::Submit {
            queue,
            program,
            timeout_ns,
            bindings,
            dispatch: Dispatch::scalar(),
        })
        .await
    }

    /// Admit one execution with explicit logical workgroup geometry.
    ///
    /// This uses the additive `SubmitLaunch` opcode. A backend that cannot
    /// honor the requested geometry rejects it; neither the service nor the
    /// core contract silently substitutes artifact-defined execution. Clients
    /// can discover support through `EXPLICIT_WORKGROUPS` in the
    /// `Query` response.
    async fn submit_workgroups(
        &mut self,
        queue: u64,
        program: u64,
        bindings: AccelBindings,
        workgroups: Workgroups,
        timeout_ns: u64,
    ) -> AccelResponse {
        self.request(AccelRequest::SubmitLaunch {
            queue,
            program,
            timeout_ns,
            bindings,
            launch: kore_driver::accel::LaunchShape::Workgroups(workgroups),
        })
        .await
    }

    /// Transfer the staging window and adopt it
    /// ([`AccelRequest::Attach`]): `staging` is a view (or lease) of the
    /// caller's own backing, attenuated to [`STAGING_RIGHTS`] — both
    /// sides write it.  On refusal before the transfer moved, the
    /// capability comes back to the caller.
    async fn attach<C: ReturnCapability>(
        &mut self,
        staging: C,
    ) -> Result<AccelResponse, ReturnCallError<C>>;

    /// Transfer a parent lease for reclaimable staging. Retain a client view
    /// before transfer, and cease using that view before requesting detach.
    async fn attach_owned<C: ReturnCapability>(
        &mut self,
        staging: C,
    ) -> Result<AccelResponse, ReturnCallError<C>>;

    /// Retire owned staging, including after device loss. A Busy reply retains
    /// cleanup custody; retry without accessing the old staging view.
    async fn detach_staging(&mut self) -> AccelResponse {
        self.request(AccelRequest::DetachStaging).await
    }

    /// Transfer owned supply capacity funding this pair's byte budgets. The
    /// increments may arrive any time, repeatedly; they die with the pair, so
    /// each process generation funds itself afresh. After transfer, even
    /// refusal may consume the supplied capability; ReturnCallError
    /// distinguishes failures before ownership moved. Unsupported providers
    /// refuse the new opcode.
    async fn provision_memory<C: ReturnCapability>(
        &mut self,
        supply: C,
        buffer_bytes: u64,
        program_bytes: u64,
    ) -> Result<AccelResponse, ReturnCallError<C>>;

    /// Mint this pair's completion channel ([`AccelRequest::Signal`]):
    /// the possessed recv end arrives beside the reply, or `None` when
    /// the service refused (polling remains lawful).
    async fn signal(&mut self) -> (AccelResponse, Option<ChanRecvToken>);
}

/// The ring + channel proxy (`res/USERLAND.md` §5.6): request pushed,
/// doorbell kicked, reply awaited; `Signal`'s retained grant lands in a
/// runtime-rotated owned slot and is possessed here.
pub struct RingAccel<'w, 't, T: Transport> {
    /// The pair's call machine.
    pub client: SvcClient<'w, 't, T, AccelWire>,
    /// The return leg's send end (`role::SVC_RETURN`, the odd index).
    pub return_tx: SlotRef,
    /// The founder-published return-landing correlation value the
    /// `Attach` request announces (`role::SVC_RETURN`, the even index).
    pub return_land: SlotRef,
}

/// Where an accelerator-profile process finds its private recyclable face.
///
/// Window applications already occupy `SVC_RING@0`, `SVC_SEND@0`, and
/// `SVC_GRANT@0`, so the accelerator face's request kick and grant receiver
/// use index 1. `SVC_RING` is the crowded namespace: the private filesystem
/// face holds 0 and 1 (control ring and data window), the GPU face 2 and an
/// engine's mm face 3, so this face's control ring is `SVC_RING@4`. The
/// filesystem face's return leg is `SVC_RETURN` 0/1 and the GPU face's 2/3,
/// so this face's is 4/5. A program may hold any of
/// these faces beside the others (a streamed-model engine holds the
/// filesystem and NPU faces together).
pub mod process_face {
    /// Accelerator request doorbell (`role::SVC_SEND`).
    pub const REQUEST_KICK: u16 = 1;
    /// Accelerator request/reply ring backing (`role::SVC_RING`).
    pub const CONTROL_RING: u16 = 4;
    /// Receiver for accelerator reply grants (`role::SVC_GRANT`).
    pub const GRANT_RECV: u16 = 1;
    /// Client-to-service retained return sender (`role::SVC_RETURN`).
    pub const RETURN_SEND: u16 = 4;
    /// Correlation datum for the service's retained return landing.
    pub const RETURN_LAND: u16 = 5;

    /// The NPU face's coordinates as one value.
    pub const SLOTS: super::FaceSlots = super::FaceSlots {
        request_kick: REQUEST_KICK,
        control_ring: CONTROL_RING,
        grant_recv: GRANT_RECV,
        return_send: RETURN_SEND,
        return_land: RETURN_LAND,
    };
}

/// Where a `PROFILE_GPU` process finds its private recyclable face on the
/// GPU row: the next index in each role namespace after the NPU face, and
/// the next two `SVC_RETURN` coordinates.  A process carrying both bits
/// holds both faces with no coordinate shared.
pub mod gpu_face {
    pub const REQUEST_KICK: u16 = 2;
    pub const CONTROL_RING: u16 = 2;
    pub const GRANT_RECV: u16 = 2;
    pub const RETURN_SEND: u16 = 2;
    pub const RETURN_LAND: u16 = 3;

    /// The GPU face's coordinates as one value.
    pub const SLOTS: super::FaceSlots = super::FaceSlots {
        request_kick: REQUEST_KICK,
        control_ring: CONTROL_RING,
        grant_recv: GRANT_RECV,
        return_send: RETURN_SEND,
        return_land: RETURN_LAND,
    };
}

/// The boot-block coordinates of one recyclable accelerator face.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaceSlots {
    pub request_kick: u16,
    pub control_ring: u16,
    pub grant_recv: u16,
    pub return_send: u16,
    pub return_land: u16,
}

const _: () = {
    assert!(process_face::REQUEST_KICK != gpu_face::REQUEST_KICK);
    assert!(process_face::CONTROL_RING != gpu_face::CONTROL_RING);
    assert!(process_face::GRANT_RECV != gpu_face::GRANT_RECV);
    assert!(process_face::RETURN_SEND != gpu_face::RETURN_SEND);
    assert!(process_face::RETURN_LAND != gpu_face::RETURN_LAND);
    assert!(process_face::RETURN_LAND != gpu_face::RETURN_SEND);
    assert!(gpu_face::RETURN_LAND != process_face::RETURN_SEND);
};

/// Attach the recyclable NPU accelerator face from a profiled process boot
/// block ([`process_face`]).
#[cfg(target_os = "none")]
pub fn attach_process_face<'t>(
    rt: &kore_rt::entry::Rt,
    transport: &'t kore_rt::transport::TrapTransport,
) -> Option<ProcessAccel<'t>> {
    attach_process_face_at(rt, transport, &process_face::SLOTS)
}

/// Attach the recyclable accelerator face at `slots` — the NPU face
/// ([`process_face::SLOTS`]) or the GPU face ([`gpu_face::SLOTS`]).  `None`
/// when the launch conveyed no face there (the profile bit was absent, or a
/// best-effort GPU admission found the row parked).
#[cfg(target_os = "none")]
pub fn attach_process_face_at<'t>(
    rt: &kore_rt::entry::Rt,
    transport: &'t kore_rt::transport::TrapTransport,
    slots: &FaceSlots,
) -> Option<ProcessAccel<'t>> {
    use kore_abi::task::role;
    use kore_rt::{
        cap::verbs,
        service::MAX_GRANTS,
        svc::{ClientEndpoints, GrantPin, client_rings},
    };

    let boot = rt.bootstrap();
    let ring = verbs::take_lease(boot.handle(role::SVC_RING, slots.control_ring)?)?;
    let window = verbs::map_held_rw(ring.slot())?;
    // SAFETY: this process is the sole client-side attachment over the
    // task-lifetime ring lease delivered by exec.
    let window: &'static mut [u8] =
        unsafe { core::slice::from_raw_parts_mut(window.cast::<u8>().as_ptr(), window.len()) };
    let rings = client_rings::<AccelWire>(
        window,
        <AccelWire as kore_rt::svc::SvcProtocol>::REQUEST_CAPACITY,
        <AccelWire as kore_rt::svc::SvcProtocol>::RESPONSE_CAPACITY,
    )
    .ok()?;
    let kick = boot.handle(role::SVC_SEND, slots.request_kick)?;
    let grant = boot.handle(role::SVC_GRANT, slots.grant_recv)?;
    let return_tx = boot.handle(role::SVC_RETURN, slots.return_send)?;
    let return_land = SlotRef::from_wire(boot.datum(role::SVC_RETURN, slots.return_land)?);
    let mut grants = [const { None }; MAX_GRANTS];
    grants[0] = Some(GrantPin::retained(grant));
    // exec admits an accelerator face only beside the launch's executor
    // doorbell, the bell this service's reply notice rings (as for the mm
    // face). Fusing the request kick with the park on it makes a call one
    // kernel session instead of a kick, an executor park and a re-post,
    // and lets the scheduler hand the CPU straight to the service and back
    // when both share one.
    Some(ProcessAccel {
        client: RingAccel {
            client: SvcClient::new(transport, rings, ClientEndpoints { kick, grants })
                .with_reply_wait(rt.doorbell()?),
            return_tx,
            return_land,
        },
        _ring: ring,
    })
}

/// An application accelerator client plus the affine ring lease that keeps
/// its mapped control window alive.
#[cfg(target_os = "none")]
pub struct ProcessAccel<'t> {
    pub client: RingAccel<'static, 't, kore_rt::transport::TrapTransport>,
    _ring: kore_rt::cap::LeaseToken,
}

/// Import completion carries either a buffer or authenticated lease custody.
#[derive(Debug)]
pub enum ImportReply {
    Imported {
        buffer: u64,
    },
    Returned {
        refusal: crate::service::AccelRefusal,
        lease: kore_rt::cap::StashedMemory,
    },
    Refused(crate::service::AccelRefusal),
}

/// Export returns an allocation lease or scoped pool loan after native detachment.
#[derive(Debug)]
pub enum ExportReply {
    Returned(kore_rt::cap::StashedMemory),
    Refused(crate::service::AccelRefusal),
}

impl<T: Transport> RingAccel<'_, '_, T> {
    /// Copy the retained device-failure report through bounded control replies.
    /// Best effort: never displace another outstanding call or replace the
    /// original failure. A timeout leaves its request owned by this client.
    pub async fn read_failure_report(&mut self, out: &mut [u8], budget_ns: u64) -> usize {
        self.read_text_report(
            out,
            budget_ns,
            crate::service::FAILURE_REPORT_BYTES,
            |offset| AccelRequest::ReadFailure { offset },
            || false,
        )
        .await.0
    }

    /// Copy the provider's current telemetry (clock, completions, completion
    /// wakes) through bounded control replies: the first page snapshots it.
    /// Best effort, with [`Self::read_failure_report`]'s rules.
    pub async fn read_telemetry_report(&mut self, out: &mut [u8], budget_ns: u64) -> usize {
        self.read_text_report(
            out,
            budget_ns,
            crate::service::TELEMETRY_REPORT_BYTES,
            |offset| AccelRequest::ReadTelemetry { offset },
            || false,
        )
        .await.0
    }

    /// The same bounded telemetry read, observing Stop between pages. The
    /// caller must still check `is_idle` before any subsequent GPU request:
    /// a timed-out page keeps its exact reply pending.
    pub async fn read_telemetry_report_checked(
        &mut self,
        out: &mut [u8],
        budget_ns: u64,
        should_stop: impl FnMut() -> bool,
    ) -> (usize, bool) {
        self.read_text_report(
            out,
            budget_ns,
            crate::service::TELEMETRY_REPORT_BYTES,
            |offset| AccelRequest::ReadTelemetry { offset },
            should_stop,
        )
        .await
    }

    async fn read_text_report(
        &mut self,
        out: &mut [u8],
        budget_ns: u64,
        report_bytes: usize,
        page: impl Fn(u64) -> AccelRequest,
        mut should_stop: impl FnMut() -> bool,
    ) -> (usize, bool) {
        if !self.client.is_idle() {
            return (0, false);
        }
        let capacity = out.len().min(report_bytes);
        let pages = capacity.div_ceil(kore_abi::svc::AccelRepWire::FAILURE_CHUNK_BYTES);
        let mut used = 0;
        for _ in 0..pages {
            if should_stop() {
                break;
            }
            let request = page(used as u64);
            let Ok(reply) = self.client.call_bounded(&request, budget_ns).await else {
                break;
            };
            if reply.grants != 0 {
                self.client.poison_retained(reply.retained);
                break;
            }
            let AccelResponse::Failure { offset, len, bytes } = reply.response else {
                break;
            };
            let Some((next, complete)) = copy_text_page(out, capacity, used, offset, len, &bytes)
            else {
                break;
            };
            used = next;
            if complete || used == capacity {
                return (used, complete);
            }
        }
        (used, false)
    }

    fn export_reply(
        &mut self,
        buffer: u64,
        mut reply: kore_rt::svc::CallReply<AccelResponse>,
    ) -> Result<ExportReply, kore_rt::svc::CallError> {
        match reply.response {
            AccelResponse::Refused(refusal) if reply.grants == 0 => {
                return Ok(ExportReply::Refused(refusal));
            }
            AccelResponse::BufferExported { buffer: returned }
                if returned == buffer && reply.grants == 1 =>
            {
                if let Some(owner) = reply.retained[0].take() {
                    match owner.try_into_memory(crate::service::PROVISION_RIGHTS) {
                        Ok(lease) => return Ok(ExportReply::Returned(lease)),
                        Err(owner) => reply.retained[0] = Some(owner),
                    }
                }
            }
            _ => {}
        }
        self.client.poison_retained(reply.retained);
        Err(kore_rt::svc::CallError::Wire)
    }

    /// Detach a buffer and receive its allocation lease or scoped route. After publication the
    /// buffer may already be gone even if observation is cancelled: resume
    /// this exact request, without resubmitting or touching the old buffer.
    pub async fn export_buffer(
        &mut self,
        buffer: u64,
    ) -> Result<ExportReply, kore_rt::svc::CallError> {
        let reply = self
            .client
            .call(&AccelRequest::ExportBuffer { buffer })
            .await?;
        self.export_reply(buffer, reply)
    }

    pub async fn resume_export_buffer(
        &mut self,
        buffer: u64,
        budget_ns: u64,
    ) -> Result<ExportReply, kore_rt::svc::CallError> {
        let reply = self
            .client
            .resume_exact_bounded(&AccelRequest::ExportBuffer { buffer }, budget_ns)
            .await?;
        self.export_reply(buffer, reply)
    }

    /// Advance the same export across cancelled or timed-out observations.
    /// The caller retains the buffer ID and reserves this face until custody
    /// is returned or containment is established. A late reply is consumed
    /// before any new publication, and returned grants are authenticated once.
    pub async fn advance_export_buffer(
        &mut self,
        buffer: u64,
        budget_ns: u64,
    ) -> Result<ExportReply, kore_rt::svc::CallError> {
        use kore_rt::svc::CallError;
        let request = AccelRequest::ExportBuffer { buffer };
        let reply = match self.client.resume_exact_bounded(&request, budget_ns).await {
            Err(CallError::NoOutstanding) => self.client.call_bounded(&request, budget_ns).await,
            result => result,
        }?;
        self.export_reply(buffer, reply)
    }

    fn import_reply(
        &mut self,
        mut reply: kore_rt::svc::CallReply<AccelResponse>,
    ) -> Result<ImportReply, kore_rt::svc::CallError> {
        match reply.response {
            AccelResponse::BufferMade { buffer } if buffer != 0 && reply.grants == 0 => {
                return Ok(ImportReply::Imported { buffer });
            }
            AccelResponse::Refused(refusal) if reply.grants == 0 => {
                return Ok(ImportReply::Refused(refusal));
            }
            AccelResponse::ImportReturned { refusal } if reply.grants == 1 => {
                if let Some(owner) = reply.retained[0].take() {
                    match owner.try_into_memory(crate::service::PROVISION_RIGHTS) {
                        Ok(lease) => return Ok(ImportReply::Returned { refusal, lease }),
                        Err(owner) => reply.retained[0] = Some(owner),
                    }
                }
            }
            _ => {}
        }
        self.client.poison_retained(reply.retained);
        Err(kore_rt::svc::CallError::Wire)
    }

    /// Transfer a whole allocation lease or scoped route as an imported Shared buffer. Stop accessing
    /// client aliases before transfer and until device completion establishes
    /// visibility. ReturnCallError distinguishes unmoved from transferred custody.
    pub async fn import_buffer<C: ReturnCapability>(
        &mut self,
        ctx: u64,
        bytes: u64,
        align: u64,
        usage: u32,
        lease: C,
    ) -> Result<ImportReply, ReturnCallError<C>> {
        let request = AccelRequest::ImportBuffer {
            ctx,
            bytes,
            align,
            usage,
            announced: self.return_land,
        };
        let reply = self
            .client
            .call_with_return(
                &request,
                self.return_tx,
                lease,
                crate::service::PROVISION_RIGHTS,
            )
            .await?;
        self.import_reply(reply)
            .map_err(ReturnCallError::AfterTransfer)
    }

    /// Cancellation-safe import: keep the source in caller-owned state until
    /// the synchronous transfer boundary. `Some` means custody never moved;
    /// `None` requires resuming this exact request, never sending a new lease.
    /// The exact transfer preserves the lease's possessed rights unchanged.
    pub async fn import_buffer_retained<C: ReturnCapability>(
        &mut self,
        ctx: u64,
        bytes: u64,
        align: u64,
        usage: u32,
        lease: &mut Option<C>,
    ) -> Result<ImportReply, kore_rt::svc::RetainedReturnCallError> {
        let request = AccelRequest::ImportBuffer {
            ctx,
            bytes,
            align,
            usage,
            announced: self.return_land,
        };
        let reply = self
            .client
            .call_with_exact_retained_return(&request, self.return_tx, lease)
            .await?;
        self.import_reply(reply)
            .map_err(kore_rt::svc::RetainedReturnCallError::AfterTransfer)
    }

    /// Advance one import whose source custody lives in caller-owned state.
    /// Retain `source` and the immutable target across cancelled futures.
    /// This first resumes the exact request; it publishes only when there is
    /// no outstanding request and the source has not moved. `None` after an
    /// uncertain transfer never authorizes a second publication.
    pub async fn advance_retained_import<C: ReturnCapability>(
        &mut self,
        target: ImportTarget,
        source: &mut Option<C>,
        budget_ns: u64,
    ) -> Result<ImportReply, kore_rt::svc::RetainedReturnCallError> {
        use kore_rt::svc::{CallError, RetainedReturnCallError};
        let ImportTarget {
            context,
            bytes,
            alignment,
            usage,
        } = target;
        let request = AccelRequest::ImportBuffer {
            ctx: context,
            bytes,
            align: alignment,
            usage,
            announced: self.return_land,
        };
        let reply = match self.client.resume_exact_bounded(&request, budget_ns).await {
            Err(CallError::NoOutstanding) if source.is_some() => {
                self.begin_import_buffer(context, bytes, alignment, usage, source)
                    .await?;
                self.client.resume_exact_bounded(&request, budget_ns).await
            }
            reply => reply,
        }
        .map_err(|error| {
            if source.is_some() {
                RetainedReturnCallError::NotMoved(error)
            } else {
                RetainedReturnCallError::AfterTransfer(error)
            }
        })?;
        self.import_reply(reply)
            .map_err(RetainedReturnCallError::AfterTransfer)
    }

    /// Publish a retained import, leaving reply collection to exact resumption.
    /// Success consumes source custody and leaves the request owned by this
    /// pair even if the caller stops observing it immediately afterwards.
    pub async fn begin_import_buffer<C: ReturnCapability>(
        &mut self,
        ctx: u64,
        bytes: u64,
        align: u64,
        usage: u32,
        lease: &mut Option<C>,
    ) -> Result<(), kore_rt::svc::RetainedReturnCallError> {
        let request = AccelRequest::ImportBuffer {
            ctx,
            bytes,
            align,
            usage,
            announced: self.return_land,
        };
        self.client
            .begin_with_exact_retained_return(&request, self.return_tx, lease)
            .await
    }

    /// Observe the same import after cancellation or an uncertain response.
    /// A timeout retains correlation and is not permission to reuse storage.
    /// A different pending request is refused without consuming its reply.
    pub async fn resume_import_buffer(
        &mut self,
        ctx: u64,
        bytes: u64,
        align: u64,
        usage: u32,
        budget_ns: u64,
    ) -> Result<ImportReply, kore_rt::svc::CallError> {
        let request = AccelRequest::ImportBuffer {
            ctx,
            bytes,
            align,
            usage,
            announced: self.return_land,
        };
        let reply = self
            .client
            .resume_exact_bounded(&request, budget_ns)
            .await?;
        self.import_reply(reply)
    }

    /// Convey an owned component on the founder administration pair.
    /// `PeerAccepted` acknowledges custody only; poll `TakePeerAdmission`
    /// after the complete bundle to learn the host-authored peer identity.
    pub async fn admit_component<C: ReturnCapability>(
        &mut self,
        part: kore_rt::svc::PeerBundlePart,
        capability: C,
        rights: Rights,
    ) -> Result<AccelResponse, ReturnCallError<C>> {
        let request = AccelRequest::AdmitPeer {
            part,
            announced: self.return_land,
        };
        let reply = self
            .client
            .call_with_return(&request, self.return_tx, capability, rights)
            .await?;
        Ok(reply.response)
    }

    /// Derive and convey atomically while retaining the source lineage.
    /// AfterTransfer means custody moved: resume this operation instead of
    /// deriving and publishing a second component.
    pub async fn admit_component_staged<F>(
        &mut self,
        part: kore_rt::svc::PeerBundlePart,
        rights: Rights,
        stage: F,
    ) -> Result<AccelResponse, kore_rt::svc::StagedReturnError>
    where
        F: FnOnce(
            kore_rt::svc::ReturnDeriver<'_>,
        ) -> Result<kore_rt::svc::StagedReturn, kore_rt::svc::ReturnDeriveError>,
    {
        let request = AccelRequest::AdmitPeer {
            part,
            announced: self.return_land,
        };
        let reply = self
            .client
            .call_with_staged_return(&request, self.return_tx, rights, stage)
            .await?;
        Ok(reply.response)
    }

    /// Resume only this component after an uncertain transport result.
    /// A different pending request is left untouched for its original owner.
    pub async fn resume_component(
        &mut self,
        part: kore_rt::svc::PeerBundlePart,
    ) -> Result<AccelResponse, kore_rt::svc::CallError> {
        let request = AccelRequest::AdmitPeer {
            part,
            announced: self.return_land,
        };
        self.client
            .resume_exact(&request)
            .await
            .map(|reply| reply.response)
    }

    /// Consume the host-authored identity of the most recently assembled
    /// recyclable face.
    pub async fn take_peer_admission(&mut self) -> Result<AccelResponse, kore_rt::svc::CallError> {
        self.client
            .call(&AccelRequest::TakePeerAdmission)
            .await
            .map(|reply| reply.response)
    }

    /// Carry one exact administration request forward under `observation`.
    /// Maintenance observes in slices, so a device operation on another
    /// accelerator pair cannot park exec's entire serve loop. Expiry retains
    /// the exact request: the same peer row must carry it to its reply before
    /// any new request uses this pair.
    async fn peer_control(
        &mut self,
        request: AccelRequest,
        observation: kore_rt::svc::Observation,
    ) -> Result<AccelResponse, kore_rt::svc::CallError> {
        use kore_rt::svc::CallError;
        let reply = self.client.call_exact(&request, observation).await?;
        if reply.grants != 0 {
            self.client.poison_retained(reply.retained);
            return Err(CallError::Wire);
        }
        Ok(reply.response)
    }

    pub async fn take_peer_admission_observed(
        &mut self,
        observation: kore_rt::svc::Observation,
    ) -> Result<AccelResponse, kore_rt::svc::CallError> {
        self.peer_control(AccelRequest::TakePeerAdmission, observation)
            .await
    }

    pub async fn retire_peer_observed(
        &mut self,
        peer: kore_rt::svc::PeerId,
        observation: kore_rt::svc::Observation,
    ) -> Result<AccelResponse, kore_rt::svc::CallError> {
        self.peer_control(AccelRequest::RetirePeer { peer }, observation)
            .await
    }

    pub async fn confirm_retired_observed(
        &mut self,
        peer: kore_rt::svc::PeerId,
        observation: kore_rt::svc::Observation,
    ) -> Result<AccelResponse, kore_rt::svc::CallError> {
        self.peer_control(AccelRequest::ConfirmRetired { peer }, observation)
            .await
    }

    /// Request retirement of one exact recyclable peer generation.
    pub async fn retire_peer(
        &mut self,
        peer: kore_rt::svc::PeerId,
    ) -> Result<AccelResponse, kore_rt::svc::CallError> {
        self.client
            .call(&AccelRequest::RetirePeer { peer })
            .await
            .map(|reply| reply.response)
    }

    /// Fence remote retirement before local ring/channel custody is reclaimed.
    pub async fn confirm_retired(
        &mut self,
        peer: kore_rt::svc::PeerId,
    ) -> Result<AccelResponse, kore_rt::svc::CallError> {
        self.client
            .call(&AccelRequest::ConfirmRetired { peer })
            .await
            .map(|reply| reply.response)
    }

    async fn call(&mut self, req: &AccelRequest) -> (AccelResponse, Option<ReturnedCapability>) {
        let mut reply = self
            .client
            .call(req)
            .await
            .expect("accel request path failed (torn rings/transport — a broken pair)");
        let delivered = (reply.grants > 0)
            .then(|| reply.retained.get_mut(0).and_then(Option::take))
            .flatten();
        (reply.response, delivered)
    }
}

/// A zero-length page at the exact next offset is the provider's EOF. A
/// nonempty page cut by the caller's buffer is never a complete report.
fn copy_text_page(
    out: &mut [u8],
    capacity: usize,
    used: usize,
    offset: u64,
    len: u8,
    bytes: &[u8; kore_abi::svc::AccelRepWire::FAILURE_CHUNK_BYTES],
) -> Option<(usize, bool)> {
    let len = usize::from(len);
    if offset != used as u64 || used > capacity || capacity > out.len() || len > bytes.len() {
        return None;
    }
    if len == 0 {
        return Some((used, true));
    }
    let take = len.min(capacity - used);
    out[used..used + take].copy_from_slice(&bytes[..take]);
    Some((used + take, take == len && len < bytes.len()))
}

#[cfg(test)]
mod text_report_tests {
    use super::copy_text_page;
    use kore_abi::svc::AccelRepWire;

    const PAGE: usize = AccelRepWire::FAILURE_CHUNK_BYTES;

    #[test]
    fn exact_page_multiples_complete_only_at_the_matching_empty_page() {
        let bytes = [b'x'; PAGE];
        for pages in 1..=1024 / PAGE {
            let mut out = [0u8; 1024];
            let mut used = 0;
            for _ in 0..pages {
                let (next, complete) =
                    copy_text_page(&mut out, 1024, used, used as u64, PAGE as u8, &bytes)
                        .expect("matching full page");
                assert!(!complete);
                used = next;
            }
            assert_eq!(used, pages * PAGE);
            assert!(out[..used].iter().all(|byte| *byte == b'x'));
            assert_eq!(
                copy_text_page(&mut out, 1024, used, used as u64, 0, &bytes),
                Some((used, true))
            );
            assert_eq!(
                copy_text_page(&mut out, 1024, used, (used + 1) as u64, 0, &bytes),
                None
            );
        }
    }

    #[test]
    fn a_short_final_page_completes_but_a_cut_page_does_not() {
        let bytes = [b'y'; PAGE];
        let mut out = [0u8; 1024];
        assert_eq!(
            copy_text_page(&mut out, 1024, 0, 0, 7, &bytes),
            Some((7, true))
        );
        assert_eq!(&out[..7], b"yyyyyyy");
        assert_eq!(copy_text_page(&mut out, 1024, 7, 8, 0, &bytes), None);

        let mut small = [0u8; 12];
        assert_eq!(
            copy_text_page(&mut small, 12, 0, 0, 20, &bytes),
            Some((12, false))
        );
        assert_eq!(&small, b"yyyyyyyyyyyy");
        assert_eq!(
            copy_text_page(&mut small, 12, 12, 12, PAGE as u8, &bytes),
            Some((12, false))
        );
    }
}

impl<T: Transport> AccelClient for RingAccel<'_, '_, T> {
    async fn request(&mut self, req: AccelRequest) -> AccelResponse {
        let (response, delivered) = self.call(&req).await;
        // No plain verb may carry a grant; a delivery here is a protocol
        // break and dropping the owner revokes it (fail-closed, no
        // authority leaks on an unexpected grant).
        drop(delivered);
        response
    }

    async fn attach<C: ReturnCapability>(
        &mut self,
        staging: C,
    ) -> Result<AccelResponse, ReturnCallError<C>> {
        let request = AccelRequest::Attach {
            announced: self.return_land,
        };
        let reply = self
            .client
            .call_with_return(&request, self.return_tx, staging, STAGING_RIGHTS)
            .await?;
        Ok(reply.response)
    }

    async fn attach_owned<C: ReturnCapability>(
        &mut self,
        staging: C,
    ) -> Result<AccelResponse, ReturnCallError<C>> {
        let request = AccelRequest::AttachOwned {
            announced: self.return_land,
        };
        let reply = self
            .client
            .call_with_return(
                &request,
                self.return_tx,
                staging,
                crate::service::PROVISION_RIGHTS,
            )
            .await?;
        Ok(reply.response)
    }

    async fn provision_memory<C: ReturnCapability>(
        &mut self,
        supply: C,
        buffer_bytes: u64,
        program_bytes: u64,
    ) -> Result<AccelResponse, ReturnCallError<C>> {
        let request = AccelRequest::ProvisionMemory {
            announced: self.return_land,
            buffer_bytes,
            program_bytes,
        };
        let reply = self
            .client
            .call_with_return(
                &request,
                self.return_tx,
                supply,
                crate::service::PROVISION_RIGHTS,
            )
            .await?;
        Ok(reply.response)
    }

    async fn signal(&mut self) -> (AccelResponse, Option<ChanRecvToken>) {
        let (response, delivered) = self.call(&AccelRequest::Signal).await;
        let Some(owner) = delivered else {
            if matches!(response, AccelResponse::Signaled) {
                self.client.poison();
                return (
                    AccelResponse::Refused(crate::service::AccelRefusal::Internal),
                    None,
                );
            }
            return (response, None);
        };
        match owner.try_into_chan_recv(Rights::ACCESS) {
            Ok(recv) => (response, Some(recv)),
            Err(owner) => {
                // This heterogeneous bank must not accept the reply tag as
                // evidence of capability kind. Preserve malformed custody in
                // generic poison cleanup, just like the import result path.
                let mut retained = core::array::from_fn(|_| None);
                retained[0] = Some(owner);
                self.client.poison_retained(retained);
                (
                    AccelResponse::Refused(crate::service::AccelRefusal::Internal),
                    None,
                )
            }
        }
    }
}

/// Build one submission's binding list — a thin re-export point so
/// protocol users need not import the service module for the one
/// constructor a call needs.
pub fn bindings(list: &[crate::service::AccelBinding]) -> Option<AccelBindings> {
    AccelBindings::new(list)
}
