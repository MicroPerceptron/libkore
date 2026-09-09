//! # kore-svc-output — the output service
//!
//! The generic byte stream **from one client to a backend provider**
//! (`res/USERLAND.md` §5.9): one [`SystemService`](kore_rt::service::SystemService)
//! pillar, deliberately smaller than a TTY — the base contract knows no
//! cursor, screen, escape sequence, or line discipline; bytes are opaque.
//!
//! Two planes (§2): the **control plane** is the ordinary svc request
//! path ([`OutputRequest`]/[`OutputResponse`] over [`wire`]'s frozen
//! forms) — attach/detach one shared queue, flush, status.  The **data
//! plane** is a bounded `ring::spsc<OutChunk>`: the client produces
//! chunks into a window it carved and delivered into this service's
//! arena (the `role::SVC_RETURN` channel — the same client→service
//! transfer mm's `Release` rides); the backend consumes to the sink.
//! Backpressure is lossless: a full queue makes the *client* wait,
//! never a silent drop.
//!
//! The backend is a seam ([`service::Backend`]): the v1 metal backend
//! is a 16550 over a mapped PCI-serial BAR ([`service::UartBackend`],
//! built on `kore-uart`'s register logic); a loopback sink serves
//! tests.  A terminal-emulator or remote-console backend swaps in at
//! the same seam without a protocol change (§6.5).

#![no_std]

#[cfg(test)]
extern crate std;

pub mod client;
pub mod service;
pub mod wire;

pub use client::{OutputClient, RingOutput};
pub use service::{
    Backend, OutputEndowment, OutputRefusal, OutputRequest, OutputResponse, OutputStatus,
    UartBackend,
};

/// The production output service type: a 16550-family PCI-serial backend
/// over `kore-uart` (`res/USERLAND.md` §5.9's swappable UART tier).  The generic
/// [`service::OutputService<B>`] stays crate-internal so the manifest's
/// citation (`kore_svc_output::OutputService`) names one concrete service.
pub type OutputService = service::OutputService<UartBackend>;
