//! # kore-svc-device — the device ownership broker (`res/DEVICE_SERVICE.md` §10)
//!
//! The userland face over the machine's device inventory: a `SystemService`
//! holding the sole `role::ROOT_COMPLEX` endowment, whose whole vocabulary
//! is *"give me a device matching this selector"* and *"here it is back"*.
//!
//! ## Why this exists
//!
//! Enumeration is mechanism and belongs to the kernel. Deciding **which
//! driver owns which function** is policy and does not — but that is
//! exactly what the founding loop used to do: resolve a class triple, rank
//! candidates by reading xHCI `PORTSC` and `CAPLENGTH` in ring 0, and size
//! a DMA pool through a driver crate the kernel linked. Every new device
//! class widened the kernel, which is the opposite of what a microkernel
//! with userland drivers is for.
//!
//! This service is where that policy went. The kernel keeps enumeration,
//! IOMMU binding and capability minting; the broker reads what firmware
//! described ([`verbs::device_record`](kore_rt::cap::verbs::device_record))
//! and hands one function to one driver.
//!
//! ## What it is *not*
//!
//! Not a god-object `DeviceService` (§1's ruling, which stands). Its
//! vocabulary is ownership arbitration plus read-only inventory observation —
//! and never `Read`/`Write`. A block read, a packet send and a display flip are
//! genuinely different operations, so each class keeps its own `SystemService`
//! vocabulary and this protocol never learns about any of them. `Inspect`
//! describes why an otherwise identical `Claim` would not match; it grants no
//! device authority and never consults the claim ledger.
//!
//! Not a nameserver either (§6). A driver does not look a device up by
//! name in an ambient directory: it holds a *capability* to this broker,
//! minted at its own founding, and a row that was founded without one can
//! claim nothing at all.
//!
//! It is also **off both hot paths**: the broker acts once per device, at
//! driver startup. No IRQ, no completion, and no byte of I/O ever re-enters
//! it (§3's control/data split).
//!
//! ## Entitlement — where it actually lives
//!
//! A claim carries the claimant's own selector, taken verbatim from its
//! `SystemService::DEVICES` declaration. That is not the broker trusting a
//! wire field: the declaration is compiled into the claimant's image, the
//! manifest generator cites it, and the boot digest gate seals the image —
//! so a row cannot ask for a class it did not declare without shipping a
//! different binary, which changes its measured digest. The manifest is
//! still the single source of truth for who may own what; only the
//! *matching* moved to ring 3.
//! Product-specific PCI rows additionally declare an exact vendor/device
//! identity; class-generic rows spell out `PciIdentitySelector::ANY`. The
//! broker applies that identity before reserving the affine row, because the
//! current index-only release cannot lawfully undo a wrong product claim.
//!
//! Two consequences worth stating rather than discovering:
//!
//! - **The client face is the entitlement.** Founding decides which rows
//!   get one. A row with no broker face is not refused at claim time — it
//!   has nothing to ask with.
//! - **This holds for *measured* rows.** When a post-boot spawned driver
//!   (not a manifest row, not digest-sealed) becomes a claimant, the
//!   structural caller identity has to become load-bearing, and the shape
//!   it grows into is the input router's per-endpoint stamp
//!   (`res/USERLAND.md` §5.8). That is deliberately not built ahead of its
//!   first real claimant.
//!
//! ## Known limitation — release needs authority proof
//!
//! The current broker deliberately refuses [`DeviceRequest::Release`]: an
//! index is replayable data and cannot prove that the delivered device grant
//! and all activity below it are dead. Clearing the ledger from that integer
//! would permit two live siblings to drive one controller. A broker-owned
//! ancestor is not sufficient either: revocation invalidates remote tags and
//! quiesces DMA, but physical BAR/IRQ/port holdings are reaped in the holder's
//! task. Reclaim therefore needs kernel-enforced cross-holder teardown or an
//! affine receipt produced only after child-first local teardown. That is not
//! faked here: the broker keeps the entry quarantined rather than handing out
//! a device somebody may still be driving.
//!
//! One concern per module (`res/USERLAND.md` §1.2 rule 4):
//!
//! - [`service`] — the endowment, the vocabulary, the ledger, and the serve step;
//! - [`client`] — the client seam a driver claims through;
//! - [`wire`] — the fail-closed codecs over [`kore_abi::svc`]'s frozen forms.

#![no_std]

#[cfg(test)]
extern crate std;

pub mod client;
pub mod service;
pub mod wire;

pub use client::{DeviceClient, RingDevice};
pub use service::{
    DeviceBroker, DeviceEndowment, DeviceRefusal, DeviceRequest, DeviceResponse, MAX_CLAIMS,
};
