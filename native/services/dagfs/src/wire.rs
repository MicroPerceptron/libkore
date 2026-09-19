//! The fail-closed codecs over `kore_abi::svc`'s frozen filesystem forms.
//!
//! The codec is the client's as much as the service's — the same bytes are
//! encoded on one side and decoded on the other — so it lives in the
//! process-side leaf ([`kore_dagfs::wire`]) and the service consumes it from
//! there. This module exists so the serving host and the router keep their
//! `crate::wire::…` spelling.

pub use kore_dagfs::wire::*;
