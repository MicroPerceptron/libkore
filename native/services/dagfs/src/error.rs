//! Compatibility home for the shared filesystem refusal contract.
//!
//! The values live in `kore-dagfs` so a process can classify a refusal
//! without linking this service's provider and device dependencies.

pub use kore_dagfs::{FsError, FsResult, ProviderError};
