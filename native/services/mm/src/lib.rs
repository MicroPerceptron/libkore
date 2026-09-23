//! # kore-svc-mm — the memory-manager service
//!
//! **The lessor**: the service endowed the memory supply,
//! granting extent windows out of it and taking them back — the first
//! [`SystemService`](kore_rt::service::SystemService) pillar, and the one
//! the default heap grows through.
//!
//! One concern per module (`res/USERLAND.md` §1.2 rule 4; `main.rs` —
//! the process entry — arrives with the founder's spawn, #96/#298):
//!
//! - [`service`] — the serve step: [`MemoryManager`] over one possessed
//!   [`SupplyToken`](kore_rt::cap::SupplyToken) plus one possessed
//!   [`RegionToken`](kore_rt::cap::RegionToken) (the child-supply mint,
//!   slice D), the [`MmRequest`] vocabulary, refusal-fail-closed carves,
//!   and the per-pair supply quota.
//! - [`client`] — the client face: the peer seam — the async
//!   [`MmClient`] (§4) with its [`RingMm`] proxy, whose
//!   [`supply`](MmClient::supply) is what a launcher mints child memory
//!   through.
//! - [`wire`] — the fail-closed enum ⇄ wire codecs over
//!   [`kore_abi::svc`]'s frozen forms (§5.3).

#![no_std]

extern crate alloc;
#[cfg(test)]
extern crate std;

pub mod client;
pub mod heap;
pub mod service;
pub mod wire;

pub use client::{MmClient, RingMm, SupplyError, process_face};
#[cfg(target_os = "none")]
pub use client::{ProcessMm, attach_process_face};
pub use service::{
    FREE_SLOTS, MemoryManager, MmEndowment, MmRefusal, MmRequest, MmResponse, MmStats, PAGE_BYTES,
    SUPPLY_QUOTA_BYTES, SUPPLY_RIGHTS,
};
