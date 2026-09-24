//! # kore-ld — the kore linker library (`res/IMAGE.md` §5; epic #291, P2 #293)
//!
//! Parse → plan → apply over ELF64 images, shaped by three constraints:
//!
//! - **no_std, no-alloc**: every buffer is fixed or caller-provided; the
//!   image service's serve step and the startup stub both run inside
//!   allocation-free contexts.
//! - **Semi-trusted input**: `check`/`parse` face bytes an adversary may
//!   have authored.  Malformed, truncated, overlapping, or lying
//!   headers *refuse* with a typed error — they never panic, never
//!   over-read, and never let a hostile image reach a lease.
//! - **Capability-shaped apply**: relocation writes go through the
//!   caller's *own mapped, writable segment slices* — a `&mut [u8]` is
//!   the authority.  There is no raw-address mapper, no ambient write,
//!   and text is never a write target (`DT_TEXTREL` refuses upstream).
//!
//! Ancestry: the parse layer descends from kraph's `klink` (the same
//! `elf`-crate substrate); the apply layer is new — klink's classical
//! half (raw-address `MemoryMapper`, RWX protection constants,
//! protect-before-relocate ordering, `#[cfg(target_arch)]` reloc
//! collapse) is exactly what `res/IMAGE.md` retires.
//!
//! Consumers: the **image service** calls [`check`] at `Register` (the
//! validation gate + the [`ModuleDescriptor`](kore_abi::image::ModuleDescriptor)
//! carve); the **startup stub** calls [`apply`] per module after mapping
//! its shares (map-RW → relocate → RELRO-seal — the seal itself is the
//! stub's verb, not kore-ld's).
//!
//! Arch dispatch is the [`reloc::RelocArch`] trait seam — one engine,
//! three data impls, selected by type instantiation.  All three arches
//! compile everywhere (the service validates any-arch images; the host
//! test suite exercises all three tables); nothing here is `#[cfg]`-ed.
//!
//! ## The unsafe charter
//!
//! Every input-facing module (`parse`/`check`/`apply`/`link`/`reloc`)
//! is `unsafe`-free under the crate-wide `deny` — hostile bytes never
//! meet a raw pointer.  The one audited exception is [`store`] (Phase
//! A, `res/USERLAND.md` §7): the caller-arena capacity plumbing
//! (`core::alloc::Allocator` is an unsafe-to-implement contract, and a
//! fixed-capacity table over an allocator block is raw-pointer
//! bookkeeping by nature).  It never touches image bytes — capacity,
//! not content.

#![no_std]
#![deny(unsafe_code)]
#![feature(allocator_api)]

#[cfg(test)]
extern crate std;

pub mod apply;
pub mod check;
pub mod functions;
pub mod late;
pub mod link;
pub mod parse;
pub mod place;
pub mod reloc;
pub mod stage;
#[allow(unsafe_code)]
pub mod store;
pub mod stub;
