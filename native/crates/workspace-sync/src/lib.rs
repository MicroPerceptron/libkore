//! Transport-neutral workspace snapshot and export transactions.
//!
//! A host path is accepted only by a host adapter.  This crate starts at the
//! first authority-neutral representation: a canonical relative path, a
//! bounded manifest, and content bytes whose digest is committed by that
//! manifest.  The same state machines are used by offline images and by a
//! future HostLink adapter; transport never decides filesystem policy.

#![no_std]

#[cfg(test)]
extern crate std;

pub mod bounds;
pub mod changes;
pub mod delta;
pub mod delta_wire;
pub mod grant;
pub mod live;
pub mod manifest;
pub mod merkle;
pub mod path;
pub mod rpc;
pub mod transaction;
pub mod wire;
pub mod write_wire;

pub use bounds::{FORMAT_VERSION, MAX_RANGE_PROOF_DEPTH};
pub use changes::{Change, ChangeKind, ChangeSet, ChangeSetBuilder, ChangeSetError};
pub use delta::{DeltaError, DeltaOp, TopologyDelta, TopologyState};
pub use delta_wire::{
    DELTA_BULK_MAGIC, DELTA_OP_BULK_MAGIC, DeltaAssembleError, DeltaAssembler, DeltaOpPage,
    DeltaWireError, MAX_DELTA_WIRE_BYTES, decode_delta, decode_delta_op, encode_delta,
    encode_delta_op,
};
pub use grant::{GrantError, GrantOperations, StandingGrant};
pub use live::{
    CacheError, CacheKey, LiveError, LiveFile, LiveWorkspace, TransactionOutcome,
    TransactionRecord, VerifiedRangeCache,
};
pub use manifest::{Entry, Manifest, ManifestBuilder, ManifestError, Policy};
pub use merkle::{RangeProof, build_range_proof, range_root, verify_range};
pub use path::{CanonicalPath, PathError};
pub use rpc::{
    ATTACH_MAGIC, AttachRequestV1, AttachResponseV1, BLOB_STORED_MAGIC, BlobStoredV1,
    DELTA_OP_REQUEST_MAGIC, DELTA_POLL_MAGIC, DELTA_RESPONSE_MAGIC, DeltaOpRequestV1, DeltaPollV1,
    DeltaResponseV1, ENTRY_REQUEST_MAGIC, ENTRY_RESPONSE_MAGIC, ERROR_RESPONSE_MAGIC,
    ManifestEntryRequestV1, ManifestEntryResponseV1, RANGE_REQUEST_MAGIC, RANGE_RESPONSE_MAGIC,
    RangeRequestV1, RangeResponseV1, RpcError, TRANSACTION_RESULT_MAGIC, TransactionResultV1,
    WorkspaceErrorV1, WorkspaceMode, WorkspaceStatus,
};
pub use transaction::{
    BeginOutcome, CommitFailure, DriveError, ImportError, ImportPhase, ImportSource, ImportStore,
    ImportTransaction, StagedFile, import_from,
};
pub use wire::{
    ContentDigest, EntryFlags, EntryKind, SnapshotHeader, SnapshotId, SourceCorrelation,
};
pub use write_wire::{
    MAX_WRITE_TRANSACTION_BYTES, WORKSPACE_BLOB_CONTENT, WORKSPACE_TRANSACTION_CONTENT,
    WRITE_TRANSACTION_MAGIC, WriteTransaction, WriteWireError, decode_write_transaction,
    encode_write_transaction,
};
