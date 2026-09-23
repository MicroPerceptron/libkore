//! The exec **protocol and client face**, separated from the spawn engine.
//!
//! A caller names a protocol to size its ring window and encode its requests.
//! It has no business linking the service that answers them — and it used to
//! have no choice: the only `SvcProtocol` witness for exec was `TaskManager`,
//! the 13k-line spawn engine, whose crate cites about twenty sibling services
//! in its founder manifest. Naming a request pulled the whole system in.
//!
//! The split was measured before it was made: `service` referenced `client`
//! zero times and `client` referenced `service` fourteen times, all of them
//! either shared wire vocabulary or the protocol witness itself. Moving the
//! vocabulary beside the codec and giving the protocol its own type took that
//! to zero, which is what made this crate a file move rather than a rewrite.
//!
//! `kore-svc-exec` depends on this crate and re-exports it, so every existing
//! `kore_svc_exec::client::…` path keeps working. It keeps its own
//! `SvcProtocol` impl — `SystemService: SvcProtocol` requires one of every
//! service — and that impl delegates here, so there is exactly one encoding,
//! one grant policy and one reply-matching rule.

#![no_std]
#![feature(allocator_api)]

extern crate alloc;
#[cfg(test)]
extern crate std;

pub mod client;
pub mod wire;

pub use client::{
    AppSeedOutcome, CapabilityNamespace, DelegatedExecClientFace, ExecClientMint,
    ExecutableCapability, ExecutableContext, ExecutableMetadata, ExecutableNamespace, InstallError,
    InvokeError, LibrarySeedOutcome, MintedExecClientFace, NamespaceError, RingExec, SpawnOutcome,
    SpawnedTask, TreeProbeOutcome,
};
pub use wire::{
    DELEGABLE_HANDLE_RIGHTS, EXEC_REPLY_GRANTS, ExecProtocol, HANDLE_RIGHTS, SHARED_WAKE_RIGHTS,
    SpawnSpec, TaskRefusal, TaskRequest, TaskResponse,
};
