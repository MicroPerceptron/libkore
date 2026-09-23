//! # kore-svc-image — the image service
//!
//! **The lessor of code** (`res/IMAGE.md`): ELF validated at `Register`
//! through the kore-ld gate, per-segment extents carved **in place**,
//! text leases consumed into executable-only successors, lease caps granted at
//! `Acquire`. Privileged control may resolve names; ordinary loader faces are
//! bound to one immutable generation and serve its entry through name-free
//! `AcquireEntry`, with dependency lookup confined to the frozen validated
//! closure.
//!
//! One concern per module (`res/USERLAND.md` §1.2 rule 4; `main.rs` —
//! the process entry — arrives with the founder's spawn, #96/#298):
//!
//! - [`service`] — the serve step: [`ImageService`] over one possessed
//!   store plus its legacy X-mint/profile role, the [`ImageRequest`]
//!   vocabulary, shadowing + retire.
//! - [`client`] — the client face: the async [`ImageClient`] (§4
//!   pinned), and the [`RingImage`] ring + channel proxy.
//! - [`wire`] — the fail-closed enum ⇄ wire codecs over
//!   [`kore_abi::svc`]'s frozen forms (§5.3).
//!
//! The production founder endows this service with the committed image
//! archive and the existing profile's X-mint. Metal startup maps and validates
//! the complete archive, reserves every persistent segment slot through
//! `kore-rt`, and beacons only after each member is registered. Request-side `Register`
//! transfers a lease into a runtime-owned persistent landing; the ring carries
//! only the pair's published return-leg name, never raw capability authority.

#![no_std]
#![feature(allocator_api)]

// The host model's fallible image table lives behind `heap`: the startup
// linker links this crate's CLIENT seam into a task with no allocator,
// and a non-generic host fn's codegen would otherwise demand one.
#[cfg(feature = "heap")]
extern crate alloc;
#[cfg(test)]
extern crate std;

pub mod client;
pub mod service;
pub mod wire;

pub use client::{
    FaceAuthority, ImageClient, ImageGrants, ImageRegistrar, RingImage, RingRegistrar,
};

// The **protocol** names — the vocabularies, the service type, its books,
// and the per-acquisition segment ceiling — exist in both builds: they are
// what a client of the image wire speaks, and the startup linker is exactly
// such a client (`SvcProtocol`'s wire/host split).
pub use service::{ImageRequest, ImageResponse, ImageService, ImageStats, MAX_IMAGE_SEGS};

// The **host** name is heap-only, like the host model it resolves for:
// [`ImageEndowment`] is the typed role bundle `SystemService::bind`
// consumes, and the fallible image table behind it lives behind `heap`.
// Re-exporting it unconditionally would make the no-allocator build fail
// on a name it has no business needing.
#[cfg(feature = "heap")]
pub use service::ImageEndowment;
