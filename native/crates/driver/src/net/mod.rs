//! **The network class contract** — hardware-neutral L2 frame vocabulary, the
//! vif model, and the [`NicBackend`](backend::NicBackend) seam.
//!
//! The sibling of [`block`](crate::block): a driver *class* contract, living
//! in the substrate beside the [`seam`](crate::seam) it is written against, so
//! a hardware driver under `drivers/` can implement it without pulling the
//! runtime in.
//!
//! # The one idea here that is not Linux's
//!
//! A network service on Linux hands out **sockets**, and the kernel
//! demultiplexes one frame stream above the driver. Isolation between
//! mutually-distrusting users is then bolted on afterwards — macvlan, SR-IOV
//! functions, an `AF_PACKET` tap.
//!
//! Here the primitive is a **vif**: a capability-scoped virtual interface,
//! `(policy-approved filter, queue pair, address, generation)`. The
//! demultiplex *is* the interface rather than something that happens above
//! it, which buys three things a socket layer cannot:
//!
//! - **Isolation by construction.** Each vif's rings live in a separate
//!   client window ([`window`]), and the classifier only writes a frame into
//!   vifs whose filter accepts it. A client holds no handle to another vif's
//!   ring, so it cannot reach one — no policy check to forget.
//! - **Filters are authority, not assertion.** A client does not receive the
//!   filter it asks for; it receives what its [`VifPolicy`](policy::VifPolicy)
//!   permits, or a refusal naming the power it lacked.
//! - **Identity survives reuse.** A [`VifId`](vif::VifId) carries a
//!   generation, so an identifier held across a close stops resolving instead
//!   of addressing the slot's next tenant.
//!
//! # Layers
//!
//! | module | what it fixes |
//! |---|---|
//! | [`addr`] | `MacAddr`, `EtherType` — the wire scalars |
//! | [`frame`] | the Ethernet II header, and the ring's `FrameDesc` |
//! | [`filter`] | what a vif accepts, and the matching that decides it |
//! | [`policy`] | what a client's authority permits it to ask for |
//! | [`vif`] | vif identity and the generation-checked table |
//! | [`window`] | the shared-window layout both sides compile in |
//! | [`backend`] | the hardware seam, and the transmit mapping discipline |

pub mod addr;
pub mod backend;
pub mod filter;
pub mod frame;
pub mod policy;
pub mod vif;
pub mod window;

pub use addr::{EtherType, MacAddr};
pub use backend::{
    LinkState, MapFault, NicBackend, NicCounters, NicHost, PacketFilter, Steering,
    StillOutstanding, TxCompletion, TxCookie, TxError, TxFrame, TxStatus, VifMapping,
};
pub use filter::{EtherTypeMatch, FrameFilter, MulticastMatch};
pub use frame::{FRAME_MAX, FRAME_MIN, FrameDesc, FrameHead};
pub use policy::{PolicyRefusal, VifPolicy};
pub use vif::{MAX_VIFS, VifId, VifRefusal, VifTable};
pub use window::{
    FRAME_AREA_BYTES, MAX_RX_FANOUT, RX_DEPTH, SLOT_BYTES, TX_DEPTH, WINDOW_BYTES, rx_frame_at,
    rx_slot_of, tx_frame_at, tx_slot_of,
};
