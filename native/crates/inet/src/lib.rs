//! # kore-inet — the internet stack above the vif
//!
//! The design is `res/INET_STACK.md`; this crate is its first slice.
//!
//! # What this crate is not
//!
//! It is not a driver, and it never sees one. Everything the three backends in
//! `drivers/net` disagree about — a descriptor ring against a bulk IN pipe,
//! MSI-X against a transfer event, hardware address filtering against
//! [`Steering::SOFTWARE`] — is resolved *below* the vif, and that resolution is
//! already witnessed by the frame program. A stack written against the vif is
//! compatible with all three the day it compiles, and with the fourth before it
//! exists.
//!
//! It is also not a service. The same library links into a client task, into a
//! row, or into a host test, and which one it is doing is a composition choice
//! made by whoever instantiates [`Ifnet`] (`res/INET_STACK.md` §4).
//!
//! # The seam
//!
//! One contract — [`Ifnet`] — with an associated [`TxSlot`] carrying the
//! per-interface encoding, so the engine above stays source-neutral and
//! selection is type instantiation rather than a branch:
//!
//! ```ignore
//! pub type MetalStack<'w> = IpStack<VifIfnet<'w>>;
//! #[cfg(test)]
//! pub type TestStack = IpStack<MockIfnet>;
//! ```
//!
//! [`MockIfnet`] is not a courtesy. It is what lets every protocol above run
//! under `make test` with no QEMU and no hardware, the way `drivers/net/dot11`
//! tests association against an in-memory AP double — and it is what makes the
//! other impl safe to change.
//!
//! [`Steering::SOFTWARE`]: kore_driver::net::Steering::SOFTWARE

#![no_std]

pub mod flow;
pub mod ifnet;
pub mod netinet;
pub mod row;
pub mod smol;
pub mod socket;

pub use flow::{
    FaceRefusal, Flow, FlowCap, FlowFrame, FlowInstall, FlowRefusal, FlowTx, InstallRefusal,
    NoFrameFace, PortRange, Powers, RecordingInstall, Scope,
};
pub use ifnet::{MOCK_SCRATCH_BYTES, Ifnet, IfnetError, MockIfnet, TxSlot, VifIfnet};
pub use netinet::{Event, IcmpEvent, IpStack, OutputError, ping::Ping};
pub use row::{Identity, InetRow, OpenGrant, OpenRefusal, PortTable};
pub use smol::{SmolDevice, SocketBudget, SocketsExhausted};
pub use socket::{FlowFace, LocalFlow, SocketRefusal, TcpListener, TcpStream, UdpSocket};
