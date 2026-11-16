//! A driver's face with the service: what both sides agree is where.
//!
//! A face is two pages exec supplies when it launches a program with the
//! virtual-device profile: the request rings, then the register block. The
//! driver has both mapped from its first instruction; the service is handed
//! the second when the face is admitted.

use kore_rt::svc::{REP_CAP, REQ_CAP, window_bytes};
use kore_vdev::regs;

#[cfg(feature = "host")]
use crate::service::VdevService;

/// Where the register block is in a driver's face: the page after the
/// request rings.
pub const REGISTERS_AT: u64 = 0x1000;

/// A driver's face: the rings' page and the register block's.
pub const PEER_BYTES: u64 = REGISTERS_AT + regs::BYTES;

// The rings fit the first page, so the block starts on the second.
const _: () = assert!(window_bytes::<VdevService>(REQ_CAP, REP_CAP) as u64 <= REGISTERS_AT);

/// The service, to a crate that speaks to it and does not host it: the name
/// its protocol hangs on, which nothing can make a value of. The service's
/// own build has the real one (`service::VdevService`).
#[cfg(not(feature = "host"))]
#[derive(Debug)]
pub enum VdevService {}
