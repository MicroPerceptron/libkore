//! # kore-svc-accel-spmd — the CPU backend for the accel protocol
//!
//! `crates/abi/src/svc.rs` describes the accel request wire form as "one
//! vocabulary for every backend (**NPU / GPU / CPU-SIMD**)". The NPU and GPU
//! have device-side implementations in tree; this crate supplies the CPU
//! backend candidate. It is not yet a founded service row. Its compute
//! resource is the machine's own cores, reached through [`kore_rt::spmd`]'s
//! region launch.
//!
//! ## Why the CPU backend is worth its own tenancy
//!
//! Not because a CPU computes faster than an NPU. The accel protocol targets
//! one client vocabulary across providers, and that target needs a CPU
//! implementation. The host tests here drive this backend through context →
//! buffer → program → submit → poll, while `kore-svc-accel` separately gates
//! the generic service and wire. That is a backend-contract witness, not yet
//! proof of a founded CPU service, common executable artifacts, scheduler
//! placement, or production portability across CPU, GPU, and NPU.
//! `res/AXIOM_INTEGRATION.md` defines the ownership boundary: `.ax` lowering
//! and CPU partition compilation belong to `ax`/`ax-exec`; this crate is the
//! provider tenancy and queue adapter. Its `KSPM` built-ins are bootstrap
//! witnesses, not a parallel graph or kernel catalog.
//!
//! ## R8: this backend is the `CPU-SIMD` half of one geometry
//!
//! `res/SPMD.md` R8 settles where the matrix backend lives: the SPMD launch
//! geometry is *both* a direct CPU API and a device-portable one. Legacy scalar
//! `OP_SUBMIT` is artifact-defined; a non-scalar compatibility record is
//! accepted only when its SIMD width equals the registered kernel's compiled
//! lanes. The additive explicit-launch operation carries only workgroup count
//! and local size, where SIMD remains purely a provider compilation/scheduling
//! choice.
//!
//! [`SpmdBackend`] implements both the deprecated exact compatibility seam and
//! `submit_with_launch`: artifact-defined execution derives its extent from the
//! output, while explicit workgroups must describe exactly the output the
//! registered kernel can hold.
//!
//! ## Two structural findings this row surfaced
//!
//! **A program need not be a compiled artifact.** [`ArtifactRef::target`] is
//! twelve *opaque* words whose "schema belongs to the artifact format, not
//! this crate" — so a format whose schema says the words name a kernel this
//! build already carries is conformant, and needs no CPU compiler to exist.
//! See [`kernel`].
//!
//! **`submit` and SPMD's one-shot launch did not compose — R9 fixes the
//! mechanism.** The `Accelerator` trait forbids `submit` from waiting for
//! execution, and SPMD's only reusable launch (`dispatch_slices`) blocked
//! on its join.  `res/SPMD.md` R9 lands the resident region — the founded-
//! once engine with a non-blocking `submit`, a completion word (`poll`),
//! and a `retire` — which IS the shape this row wants (and the device rows
//! already have).  So the compute still lives in
//! [`SpmdBackend::run_pending`] — admit, drive, observe — but that one
//! function becomes a resident client the day this crate is founded as a
//! service: `found_resident` once, `submit` per launch, `poll` for
//! completion, no gang re-founded per submit.  It is deliberately NOT
//! rewired yet: with no serve loop driving it, a resident path would be
//! metal code no gate exercises (the metal-real rule), so the rewire lands
//! with the service that drives it.  Until then `run_pending` keeps the
//! `dispatch_slices` path, which produces identical bytes.  See [`backend`].
//!
//! ## What is not here
//!
//! A `SystemService` row. This crate is the backend and its kernels; founding
//! it as a boot service (manifest row, endowment, founding order, archive
//! measurement) is its own slice — and it is the slice that turns
//! [`SpmdBackend::run_pending`] into an R9 resident client (above), because
//! only then is that path metal-driven and provable.
//!
//! [`ArtifactRef::target`]: virtio_accel_core::ArtifactRef

#![no_std]
#![feature(portable_simd)]

pub mod backend;
pub mod kernel;

pub use backend::{
    FORMAT_REGISTERED, REGISTERED_MANIFEST, REGISTERED_RESIDENT_BYTES, SpmdBackend, SpmdBuffer,
    SpmdContext, SpmdEvent, SpmdProgram, SpmdQueue,
};
pub use kernel::{BUILTIN, IDENTITY_MAGIC, Kernel, TileCtx};

#[cfg(test)]
extern crate std;
