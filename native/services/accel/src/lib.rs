//! # kore-svc-accel — the accelerator class protocol and its backend host
//!
//! The **public face of every accelerator** — the Intel NPU
//! (`drivers/ivpu`), the Intel Xe GPU (`drivers/xe`), a CPU-SIMD
//! executor (`services/accel-spmd`), and later providers — as one
//! `AccelRequest` vocabulary over the ordinary
//! kore pair rings, hosted by one generic engine over one backend seam.
//!
//! ## The protocol is virtio-accel 1.0, re-carried
//!
//! The vocabulary mirrors the house **virtio-accel protocol 1.0**
//! (`MicroPerceptron/virtio-accel`, frozen) object model one-to-one:
//! contexts isolate accelerator state; buffers, programs, and execution
//! queues live inside them; a submission binds buffers to a program's
//! slots and returns an **event**; events are polled, cancelled, and
//! destroyed.  Object ids are opaque, kind-tagged, and generational
//! (`virtio-accel-device`'s tables — a stale or cross-kind id never
//! resolves).  What is deliberately **not** carried is the virtio frame
//! encoding: between two kore-native tasks the pair rings, the return
//! leg, and the grant channels already move payloads and authority, so
//! the wire here is an ordinary [`kore_abi::svc`] record pair
//! ([`AccelReqWire`](kore_abi::svc::AccelReqWire) /
//! [`AccelRepWire`](kore_abi::svc::AccelRepWire)) and the descriptor
//! chains wait for the hypervisor-facing device, where a *guest* is the
//! peer and wire frames are the real interoperability boundary.
//!
//! ## Two planes
//!
//! - **Control** rides the rings: every verb is a small fixed record;
//!   replies are data.  The one authority-bearing reply is
//!   [`AccelRequest::Signal`](service::AccelRequest::Signal)'s, whose
//!   completion-channel recv end crosses as a retained grant.
//! - **Bulk bytes** (artifacts, tensor data) ride the **staging
//!   window**: one client-supplied region, transferred once over the
//!   return leg ([`AccelRequest::Attach`](service::AccelRequest::Attach),
//!   the swap service's shape exactly) and addressed by offset ever
//!   after.  Never a delivered view per operation — a receiver's map
//!   burns an irreversible span of its map-window cursor.
//!
//! Completion is **channel-signalled, poll-authoritative**: `Submit`
//! answers with the event id at the *acceptance boundary* (an
//! indeterminate admission still returns the event that owns the
//! submission's resources); when work finishes, the service posts one
//! [`COOKIE_ACCEL_DONE`] notice per drained batch on the pair's signal
//! channel, and a dropped notice loses nothing because
//! [`AccelRequest::PollEvent`](service::AccelRequest::PollEvent) is the
//! re-readable truth (the `COOKIE_REPORT` doctrine).
//!
//! ## The launch geometry
//!
//! Deployed `OP_SUBMIT` bytes remain compatible: its scalar sentinel is
//! artifact-defined, while a non-scalar record stays a deprecated
//! [`Submit`](service::AccelRequest::Submit) until a backend proves
//! its exact grid/group/SIMD contract or rejects it. A client that explicitly
//! chooses a portable workgroup count and local size
//! uses [`AccelClient::submit_workgroups`](client::AccelClient::submit_workgroups),
//! whose additive wire opcode carries [`Workgroups`](contract::Workgroups).
//! [`dispatch`] lowers a kore [`SpmdCtx`](kore_rt::spmd::SpmdCtx) tile grid to
//! that shape. SIMD/subgroup width remains a provider decision. A backend
//! whose artifact fixes the shape rejects explicit workgroups through the
//! seam's default `submit_with_launch`; neither dialect silently discards
//! geometry.
//!
//! ## The backend seam
//!
//! The lifecycle seam is the published
//! [`contract::Accelerator`](virtio_accel_core::Accelerator), which the
//! upstream conformance suite gates. Kore hosts add
//! [`contract::NativeAccelerator`] for native launch and health hooks. A backend
//! is an endowment choice, never a protocol change (`res/USERLAND.md`
//! §5.9 criterion 5): each hardware row instantiates
//! [`AccelService<A>`](service::AccelService) with its own endowment and
//! backend, and the frozen suite in `virtio-accel-conformance` is the
//! acceptance harness for every one of them.  The intended mappings:
//!
//! - **iVPU** (`drivers/ivpu`, the next slice): `Context` = an on-die
//!   SMMU stream/context install; `Buffer` = a DMA-pool carve mapped
//!   into its stage-1; `Program` = a VPUX ELF
//!   ([`FORMAT_VPUX_ELF`]) through `ModelBlob`
//!   parse→accept→place→relocate, with `bind_io`/`seal` per submission;
//!   `Queue` = a `JobQueue` ring + doorbell; `Event` = a job/fence slot
//!   probed by `complete.rs` — "completion is a probe, not a loop" *is*
//!   `poll_event`'s contract, and the JOB_DONE IRQ relay worker drives
//!   [`service::AccelService::pump_completions`].
//! - **Xe** (`drivers/xe`): `Context` = a PPGTT VM; `Buffer` = a BO placed
//!   by the row's own allocation owner; `Queue` = a native GuC CCS0
//!   context/ring; `Event` = a job fence. Vulkan reaches this same seam from
//!   a Mesa hypervisor guest over the virtio-accel transport, not from a
//!   driver-resident Vulkan backend.
//! - **AMD GPU** (`drivers/amdgpu`): a structural skeleton only — the
//!   per-IP-version layout for GC 11.x and up, with no implementation. Older
//!   AMD parts and Vulkan both take the guest route.
//! - **CPU-SIMD candidate** (`services/accel-spmd`): registered kernels over
//!   host buffers; submission only admits work. Its tests drive
//!   `SpmdBackend::run_pending` before polling terminal state, while a founded
//!   owning serve loop remains future work.
//!
//! ## What this crate is not
//!
//! Not a compute-graph placer — accelerator placement belongs to the
//! userland compute scheduler (`res/MANIFOLD.md` scope ruling), and this
//! service only executes what one client hands one device.  Not a boot
//! row — the first founded instantiation is the ivpu backend's, which
//! founds degraded on NPU-less machines (the swap-nvme posture); this
//! crate carries the protocol, the host, and the conformance gate.

#![no_std]

extern crate alloc;

#[cfg(test)]
extern crate std;

pub mod client;
pub mod dispatch;
pub mod service;
pub mod wire;

/// The frozen accelerator contracts this protocol re-carries — the
/// backend seam every accelerator row implements
/// ([`Accelerator`](virtio_accel_core::Accelerator)) and its
/// descriptor/validation vocabulary.
pub use kore_driver::accel as contract;
/// The access declaration a client's imported buffer makes for a binding.
pub use virtio_accel_core::AccessMode;

/// Signal-channel notice cookie: **"your accel events have news"** — at
/// most one per drained completion batch, never one per event (the
/// `COOKIE_REPORT` bare-notice doctrine: the poll is the truth, the
/// notice only wakes).  The payload word carries how many events turned
/// terminal in the batch, as a hint.
pub const COOKIE_ACCEL_DONE: u64 = 0x6B72_5F61_636C_444E; // "kr_aclDN"

pub use client::{FaceSlots, gpu_face, process_face};
#[cfg(target_os = "none")]
pub use client::{ProcessAccel, attach_process_face, attach_process_face_at};

// The cookie must collide neither with the kernel-reserved notice cookie
// (a service cookie equal to it would make the signal send unsendable —
// a puzzling runtime refusal where this is a build error) nor with the
// `kr_svc*` run the serve transport already speaks.
const _: () = {
    assert!(COOKIE_ACCEL_DONE != kore_abi::session::NOTICE_COOKIE_PAGER_FAULT);
    assert!(COOKIE_ACCEL_DONE != kore_rt::svc::COOKIE_REQUEST);
    assert!(COOKIE_ACCEL_DONE != kore_rt::svc::COOKIE_REPLY);
    assert!(COOKIE_ACCEL_DONE != kore_rt::svc::COOKIE_REPORT);
    assert!(COOKIE_ACCEL_DONE != kore_rt::svc::COOKIE_READY);
    assert!(COOKIE_ACCEL_DONE != kore_rt::svc::COOKIE_SERVING);
    assert!(COOKIE_ACCEL_DONE != kore_rt::svc::COOKIE_REFUSED);
    assert!(COOKIE_ACCEL_DONE != kore_rt::svc::COOKIE_SHELL);
};

/// The artifact-format registry — the `d` word of
/// [`AccelRequest::LoadProgram`](service::AccelRequest::LoadProgram).
/// Formats are opaque to the protocol ([`contract::ArtifactFormat`]);
/// this registry only keeps kore's backends from colliding on a number.
/// A backend refuses formats it does not speak with
/// [`AccelRefusal::Unsupported`](service::AccelRefusal::Unsupported).
/// For this format, all twelve target words must be zero: the artifact's own
/// ELF architecture/ABI metadata supplies compatibility requirements. Unknown
/// nonzero envelope encodings are incompatible, not advisory hints.
pub const FORMAT_VPUX_ELF: contract::ArtifactFormat = match contract::ArtifactFormat::new(1) {
    Some(format) => format,
    None => unreachable!(),
};

/// Sealed Intel Xe native-kernel package (`KXENATV`), generation and ABI
/// checked by the native Xe provider before it becomes resident.
pub const FORMAT_XE_NATIVE: contract::ArtifactFormat = kore_driver::accel::FORMAT_XE_NATIVE;
