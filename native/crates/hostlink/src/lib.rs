//! Bounded, transport-independent HostLink contracts (#538).
//!
//! This crate owns the byte contract and the state that must agree across
//! every HostLink transport.  A VirtIO serial port, a PCI shared-memory
//! window, and host-side Unix sockets all feed the same [`FrameDecoder`] and
//! [`RouteState`].  Consequently a transport cannot reinterpret a route id,
//! silently relax replay checks, or grow its own unbounded request table.
//!
//! The crate is `no_std`, allocation-free, and contains no global state or
//! locks.  Callers own one route state per capability-established mailbox and
//! supply the storage used to reassemble a frame.

#![no_std]

#[cfg(test)]
extern crate std;

mod agent;
mod authority;
mod broker;
mod frontend;
mod frontend_update;
mod protocol;
mod router;
mod state;
mod stream;
mod transport;
mod wire;

pub use agent::*;
pub use authority::*;
pub use broker::*;
pub use frontend::*;
pub use frontend_update::*;
pub use protocol::*;
pub use router::{BindError, BoundRoute, ProcessError, RouteTable};
pub use state::{
    AdmitError, CreditError, Direction, RequestError, RouteError, RouteLifecycle, RouteRole,
    RouteState, SequenceDisposition,
};
pub use stream::{
    BulkFrameDecoder, ControlFrameDecoder, DecodeError, DecodeProgress, FrameDecoder, FramedDecoder,
};
pub use transport::{HostLinkTransport, Plane, PortError};
pub use wire::{
    BULK_CHUNK_MAX, BULK_CHUNKS_MAX, BULK_FRAME_MAX, BULK_OBJECT_MAX, BulkId, BulkRefV1,
    CONTROL_FRAME_MAX, ENVELOPE_LEN, EnvelopeFields, EnvelopeV1, FAMILY_LINK,
    FOUNDED_BULK_WIRE_CHUNK, HOSTLINK_MAGIC, INLINE_MAX, LinkEpoch, RequestId, RouteId,
    SCHEMA_MAJOR_V1, TaskGeneration, derive_link_epoch, derive_shmem_epoch,
};
