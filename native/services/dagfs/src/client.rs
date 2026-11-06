//! The typed filesystem clients.
//!
//! [`RingFs`] (the pair's SPSC rings, on metal) and [`FsClient`] (the
//! synchronous host form over a [`Transport`]) live in the process-side leaf
//! ([`kore_dagfs::client`]): an application holds a client without linking
//! this service. The service re-exports them so its host suites and the
//! serving contract keep one spelling.

pub use kore_dagfs::client::*;
