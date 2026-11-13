//! The request vocabulary and its crossing shape.
//!
//! Two kinds of asker share one ring layout. The administration pair
//! (exec's) conveys the components of a driver's face and retires it when
//! the program ends; it never asks for a device. A driver's own face asks
//! for a device and lends it memory; it cannot administer. Which is which
//! is the pair a request arrived on, never a byte of the request.

use kore_abi::session::SlotRef;
use kore_abi::svc::{VdevRepWire, VdevReqWire};
use kore_rt::svc::{PeerBundlePart, PeerId, SvcProtocol};

use crate::VdevService;

/// What a face may ask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VdevRequest {
    /// Administration: one component of a driver's face, delivered on the
    /// administration pair's return leg.
    AdmitPeer {
        part: PeerBundlePart,
        announced: SlotRef,
    },
    /// Administration: the verdict on the face whose last component was
    /// conveyed.
    TakeAdmission,
    /// Administration: the program holding this face has ended.
    RetirePeer { peer: PeerId },
    /// Administration: has that face gone?
    ConfirmRetired { peer: PeerId },
    /// A driver ends its own face.
    ClosePeer,
    /// A driver asks for the device model with this register-block id.
    Open { device: u32 },
    /// A driver lends the memory its device addresses, delivered on the
    /// face's return leg.
    Lend { announced: SlotRef },
    /// A driver gives its device up and takes its memory back.
    Close,
}

/// What a face is told.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VdevResponse {
    Accepted,
    Refused(VdevRefusal),
    PeerAdmitted {
        peer: PeerId,
    },
    /// The device addresses this many 4 KiB pages from here on.
    Lent {
        pages: u32,
    },
}

/// Why a request was not done.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VdevRefusal {
    /// The request is not this face's to make.
    Denied,
    /// The admission result is not published yet.
    Pending,
    /// An earlier admission must be settled first.
    Conflict,
    /// The service cannot do this now.
    Unavailable,
    /// No device model has that id.
    NoDevice,
    /// The device is not in a state for this.
    Busy,
    /// Every device the service hosts at once is taken.
    Limit,
    /// What was lent is not memory a device can address.
    Memory,
}

#[cfg(feature = "host")]
impl From<crate::host::Refusal> for VdevRefusal {
    fn from(refusal: crate::host::Refusal) -> Self {
        use crate::host::Refusal;
        match refusal {
            Refusal::NoDevice => Self::NoDevice,
            Refusal::Busy => Self::Busy,
            Refusal::Limit => Self::Limit,
            Refusal::Unavailable => Self::Unavailable,
            Refusal::Memory => Self::Memory,
        }
    }
}

const PART_RING: u8 = 0;
const PART_KICK: u8 = 1;
const PART_RETURN: u8 = 2;
const PART_GRANT: u8 = 0x80;

/// Lower a request (total).
pub fn encode_req(req: &VdevRequest) -> VdevReqWire {
    let mut wire = VdevReqWire {
        op: 0,
        reserved: [0; 7],
    };
    wire.op = match *req {
        VdevRequest::AdmitPeer { part, announced } => {
            wire.reserved[0] = match part {
                PeerBundlePart::Ring => PART_RING,
                PeerBundlePart::Kick => PART_KICK,
                PeerBundlePart::ReturnRecv => PART_RETURN,
                PeerBundlePart::Grant(index) => PART_GRANT | index,
            };
            wire.reserved[1..5].copy_from_slice(&announced.index().to_le_bytes());
            VdevReqWire::OP_ADMIT_PEER
        }
        VdevRequest::TakeAdmission => VdevReqWire::OP_TAKE_ADMISSION,
        VdevRequest::RetirePeer { peer } => {
            wire.reserved[..2].copy_from_slice(&peer.arrival().to_le_bytes());
            VdevReqWire::OP_RETIRE_PEER
        }
        VdevRequest::ConfirmRetired { peer } => {
            wire.reserved[..2].copy_from_slice(&peer.arrival().to_le_bytes());
            VdevReqWire::OP_CONFIRM_RETIRED
        }
        VdevRequest::ClosePeer => VdevReqWire::OP_CLOSE_PEER,
        VdevRequest::Open { device } => {
            wire.reserved[..4].copy_from_slice(&device.to_le_bytes());
            VdevReqWire::OP_OPEN
        }
        VdevRequest::Lend { announced } => {
            wire.reserved[..4].copy_from_slice(&announced.index().to_le_bytes());
            VdevReqWire::OP_LEND
        }
        VdevRequest::Close => VdevReqWire::OP_CLOSE,
    };
    wire
}

fn word(bytes: &[u8; 7], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

fn decode_peer(bytes: &[u8; 7]) -> Option<PeerId> {
    if bytes[2..].iter().any(|byte| *byte != 0) {
        return None;
    }
    PeerId::from_arrival(u16::from_le_bytes([bytes[0], bytes[1]]))
}

/// Lift a wire request, fail-closed: only each opcode's declared payload is
/// accepted, and every byte beyond it must be zero.
pub fn decode_req(w: &VdevReqWire) -> Option<VdevRequest> {
    let empty = w.reserved == [0; 7];
    Some(match w.op {
        VdevReqWire::OP_ADMIT_PEER if w.reserved[5..] == [0; 2] => VdevRequest::AdmitPeer {
            part: match w.reserved[0] {
                PART_RING => PeerBundlePart::Ring,
                PART_KICK => PeerBundlePart::Kick,
                PART_RETURN => PeerBundlePart::ReturnRecv,
                value if value & PART_GRANT != 0 => PeerBundlePart::Grant(value & !PART_GRANT),
                _ => return None,
            },
            announced: SlotRef::from_wire(word(&w.reserved, 1)),
        },
        VdevReqWire::OP_TAKE_ADMISSION if empty => VdevRequest::TakeAdmission,
        VdevReqWire::OP_RETIRE_PEER => VdevRequest::RetirePeer {
            peer: decode_peer(&w.reserved)?,
        },
        VdevReqWire::OP_CONFIRM_RETIRED => VdevRequest::ConfirmRetired {
            peer: decode_peer(&w.reserved)?,
        },
        VdevReqWire::OP_CLOSE_PEER if empty => VdevRequest::ClosePeer,
        VdevReqWire::OP_OPEN if w.reserved[4..] == [0; 3] => VdevRequest::Open {
            device: word(&w.reserved, 0),
        },
        VdevReqWire::OP_LEND if w.reserved[4..] == [0; 3] => VdevRequest::Lend {
            announced: SlotRef::from_wire(word(&w.reserved, 0)),
        },
        VdevReqWire::OP_CLOSE if empty => VdevRequest::Close,
        _ => return None,
    })
}

/// Lower a response (total). The service grants nothing, so `grants` is
/// written zero whatever was asked.
pub fn encode_rep(rep: &VdevResponse, _grants: u8) -> VdevRepWire {
    let (tag, refusal, reserved) = match *rep {
        VdevResponse::Accepted => (VdevRepWire::TAG_ACCEPTED, 0, [0; 5]),
        VdevResponse::PeerAdmitted { peer } => {
            let arrival = peer.arrival().to_le_bytes();
            (
                VdevRepWire::TAG_PEER_ADMITTED,
                0,
                [arrival[0], arrival[1], 0, 0, 0],
            )
        }
        VdevResponse::Lent { pages } => {
            let pages = pages.to_le_bytes();
            (
                VdevRepWire::TAG_LENT,
                0,
                [pages[0], pages[1], pages[2], pages[3], 0],
            )
        }
        VdevResponse::Refused(refusal) => (
            VdevRepWire::TAG_REFUSED,
            match refusal {
                VdevRefusal::Denied => VdevRepWire::REFUSE_DENIED,
                VdevRefusal::Pending => VdevRepWire::REFUSE_PENDING,
                VdevRefusal::Conflict => VdevRepWire::REFUSE_CONFLICT,
                VdevRefusal::Unavailable => VdevRepWire::REFUSE_UNAVAILABLE,
                VdevRefusal::NoDevice => VdevRepWire::REFUSE_NO_DEVICE,
                VdevRefusal::Busy => VdevRepWire::REFUSE_BUSY,
                VdevRefusal::Limit => VdevRepWire::REFUSE_LIMIT,
                VdevRefusal::Memory => VdevRepWire::REFUSE_MEMORY,
            },
            [0; 5],
        ),
    };
    VdevRepWire {
        tag,
        refusal,
        grants: 0,
        reserved,
    }
}

/// Lift a wire reply, fail-closed: a non-zero `grants`, a refusal byte on
/// a success tag and a set reserved byte each refuse.
pub fn decode_rep(w: &VdevRepWire) -> Option<(VdevResponse, u8)> {
    if w.grants != 0 {
        return None;
    }
    let response = match (w.tag, w.refusal) {
        (VdevRepWire::TAG_ACCEPTED, 0) if w.reserved == [0; 5] => VdevResponse::Accepted,
        (VdevRepWire::TAG_PEER_ADMITTED, 0) if w.reserved[2..] == [0; 3] => {
            VdevResponse::PeerAdmitted {
                peer: PeerId::from_arrival(u16::from_le_bytes([w.reserved[0], w.reserved[1]]))?,
            }
        }
        (VdevRepWire::TAG_LENT, 0) if w.reserved[4] == 0 => VdevResponse::Lent {
            pages: u32::from_le_bytes([w.reserved[0], w.reserved[1], w.reserved[2], w.reserved[3]]),
        },
        (VdevRepWire::TAG_REFUSED, refusal) if w.reserved == [0; 5] => {
            VdevResponse::Refused(match refusal {
                VdevRepWire::REFUSE_DENIED => VdevRefusal::Denied,
                VdevRepWire::REFUSE_PENDING => VdevRefusal::Pending,
                VdevRepWire::REFUSE_CONFLICT => VdevRefusal::Conflict,
                VdevRepWire::REFUSE_UNAVAILABLE => VdevRefusal::Unavailable,
                VdevRepWire::REFUSE_NO_DEVICE => VdevRefusal::NoDevice,
                VdevRepWire::REFUSE_BUSY => VdevRefusal::Busy,
                VdevRepWire::REFUSE_LIMIT => VdevRefusal::Limit,
                VdevRepWire::REFUSE_MEMORY => VdevRefusal::Memory,
                _ => return None,
            })
        }
        _ => return None,
    };
    Some((response, 0))
}

impl SvcProtocol for VdevService {
    type Request = VdevRequest;
    type Response = VdevResponse;
    type ReqWire = VdevReqWire;
    type RepWire = VdevRepWire;

    /// The service mints no authority: structural, not a convention.
    const REPLY_GRANTS: usize = 0;
    /// Pair zero is the founder's control face; pair one is exec's, which
    /// administers the driver faces.
    const SERVE_PAIRS: usize = 2;

    fn reply_matches(request: &VdevReqWire, response: &VdevResponse, grants: u8) -> bool {
        grants == 0
            && match (decode_req(request), response) {
                (Some(VdevRequest::TakeAdmission), VdevResponse::PeerAdmitted { .. }) => true,
                (Some(VdevRequest::Lend { .. }), VdevResponse::Lent { .. }) => true,
                (Some(VdevRequest::Lend { .. }), VdevResponse::Accepted) => false,
                (Some(_), VdevResponse::Accepted | VdevResponse::Refused(_)) => true,
                _ => false,
            }
    }

    fn encode_req(req: &VdevRequest) -> VdevReqWire {
        encode_req(req)
    }
    fn decode_req(wire: &VdevReqWire) -> Option<VdevRequest> {
        decode_req(wire)
    }
    fn encode_rep(rep: &VdevResponse, grants: u8) -> VdevRepWire {
        encode_rep(rep, grants)
    }
    fn decode_rep(wire: &VdevRepWire) -> Option<(VdevResponse, u8)> {
        decode_rep(wire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peer() -> PeerId {
        PeerId::from_arrival(PeerId::SLOT_CAPACITY as u16).expect("the first recyclable pair")
    }

    const REQUESTS: [fn() -> VdevRequest; 11] = [
        || VdevRequest::AdmitPeer {
            part: PeerBundlePart::Ring,
            announced: SlotRef::from_wire(27),
        },
        || VdevRequest::AdmitPeer {
            part: PeerBundlePart::Kick,
            announced: SlotRef::from_wire(28),
        },
        || VdevRequest::AdmitPeer {
            part: PeerBundlePart::ReturnRecv,
            announced: SlotRef::from_wire(u32::MAX),
        },
        || VdevRequest::AdmitPeer {
            part: PeerBundlePart::Grant(3),
            announced: SlotRef::from_wire(0),
        },
        || VdevRequest::TakeAdmission,
        || VdevRequest::RetirePeer { peer: peer() },
        || VdevRequest::ConfirmRetired { peer: peer() },
        || VdevRequest::ClosePeer,
        || VdevRequest::Open {
            device: 0x6c63_6361,
        },
        || VdevRequest::Lend {
            announced: SlotRef::from_wire(0x0102_0304),
        },
        || VdevRequest::Close,
    ];

    const RESPONSES: [fn() -> VdevResponse; 12] = [
        || VdevResponse::Accepted,
        || VdevResponse::PeerAdmitted { peer: peer() },
        || VdevResponse::Lent { pages: 1 },
        || VdevResponse::Lent { pages: u32::MAX },
        || VdevResponse::Refused(VdevRefusal::Denied),
        || VdevResponse::Refused(VdevRefusal::Pending),
        || VdevResponse::Refused(VdevRefusal::Conflict),
        || VdevResponse::Refused(VdevRefusal::Unavailable),
        || VdevResponse::Refused(VdevRefusal::NoDevice),
        || VdevResponse::Refused(VdevRefusal::Busy),
        || VdevResponse::Refused(VdevRefusal::Limit),
        || VdevResponse::Refused(VdevRefusal::Memory),
    ];

    #[test]
    fn every_request_and_response_round_trips() {
        for request in REQUESTS {
            let request = request();
            assert_eq!(decode_req(&encode_req(&request)), Some(request));
        }
        for response in RESPONSES {
            let response = response();
            assert_eq!(decode_rep(&encode_rep(&response, 0)), Some((response, 0)));
            // Whatever was asked, nothing is granted.
            assert_eq!(encode_rep(&response, 3).grants, 0);
        }
    }

    /// A zeroed record, an unknown opcode and a byte set beyond an opcode's
    /// payload each refuse: nothing is guessed.
    #[test]
    fn a_request_decodes_only_from_exactly_its_own_bytes() {
        let zero = VdevReqWire {
            op: 0,
            reserved: [0; 7],
        };
        assert_eq!(decode_req(&zero), None, "a zeroed record asks nothing");
        assert_eq!(decode_req(&VdevReqWire { op: 9, ..zero }), None);
        assert_eq!(decode_req(&VdevReqWire { op: 255, ..zero }), None);

        for request in REQUESTS {
            let request = request();
            let wire = encode_req(&request);
            // How many leading bytes of the payload this opcode owns.
            let owned = match request {
                VdevRequest::AdmitPeer { .. } => 5,
                VdevRequest::RetirePeer { .. } | VdevRequest::ConfirmRetired { .. } => 2,
                VdevRequest::Open { .. } | VdevRequest::Lend { .. } => 4,
                VdevRequest::TakeAdmission | VdevRequest::ClosePeer | VdevRequest::Close => 0,
            };
            for at in owned..7 {
                let mut stray = wire;
                stray.reserved[at] = 1;
                assert_eq!(decode_req(&stray), None, "{request:?}, byte {at}");
            }
        }
        // An unknown part of a face is not a part.
        let mut part = encode_req(&REQUESTS[0]());
        part.reserved[0] = 3;
        assert_eq!(decode_req(&part), None);
        // A fixed pair is not a face exec can retire.
        let mut fixed = encode_req(&VdevRequest::RetirePeer { peer: peer() });
        fixed.reserved[..2].copy_from_slice(&1u16.to_le_bytes());
        assert_eq!(decode_req(&fixed), None);
    }

    #[test]
    fn a_reply_decodes_only_from_exactly_its_own_bytes() {
        let zero = VdevRepWire {
            tag: 0,
            refusal: 0,
            grants: 0,
            reserved: [0; 5],
        };
        assert_eq!(decode_rep(&zero), None);
        for response in RESPONSES {
            let response = response();
            let wire = encode_rep(&response, 0);
            assert_eq!(decode_rep(&VdevRepWire { grants: 1, ..wire }), None);
            if !matches!(response, VdevResponse::Refused(_)) {
                let refused = VdevRepWire { refusal: 1, ..wire };
                assert_eq!(decode_rep(&refused), None, "{response:?}");
            }
            let owned = match response {
                VdevResponse::PeerAdmitted { .. } => 2,
                VdevResponse::Lent { .. } => 4,
                VdevResponse::Accepted | VdevResponse::Refused(_) => 0,
            };
            for at in owned..5 {
                let mut stray = wire;
                stray.reserved[at] = 1;
                assert_eq!(decode_rep(&stray), None, "{response:?}, byte {at}");
            }
        }
        let unknown = VdevRepWire {
            tag: VdevRepWire::TAG_REFUSED,
            refusal: 9,
            ..zero
        };
        assert_eq!(decode_rep(&unknown), None);
    }

    /// A reply is believed only if it answers what was asked.
    #[test]
    fn a_reply_must_answer_its_request() {
        let matches = |request: VdevRequest, response: VdevResponse| {
            <VdevService as SvcProtocol>::reply_matches(&encode_req(&request), &response, 0)
        };
        let lend = VdevRequest::Lend {
            announced: SlotRef::from_wire(1),
        };
        assert!(matches(lend, VdevResponse::Lent { pages: 4 }));
        assert!(!matches(lend, VdevResponse::Accepted));
        assert!(matches(lend, VdevResponse::Refused(VdevRefusal::Busy)));
        assert!(!matches(lend, VdevResponse::PeerAdmitted { peer: peer() }));
        assert!(matches(
            VdevRequest::TakeAdmission,
            VdevResponse::PeerAdmitted { peer: peer() }
        ));
        assert!(!matches(
            VdevRequest::Close,
            VdevResponse::Lent { pages: 4 }
        ));
        assert!(matches(VdevRequest::Close, VdevResponse::Accepted));
        assert!(!<VdevService as SvcProtocol>::reply_matches(
            &encode_req(&VdevRequest::Close),
            &VdevResponse::Accepted,
            1
        ));
    }
}
