//! `wire` — the mm enum ⇄ wire codecs (`res/USERLAND.md` §5.3).
//!
//! The zerocopy wire forms live in [`kore_abi::svc`] (the frozen
//! contract); this module is the convenience bridge: encode is total,
//! decode is **fail-closed** — an out-of-vocabulary op, refusal, DMA
//! class, or attribute pattern decodes to `None`, never a guess.  The
//! [`SvcProtocol`] impl is what plugs [`MemoryManager`] into the generic
//! request path (rings + doorbells + grant channels).

use kore_abi::mem::{AllocProps, DmaClass, MemAttrs, NumaNode};
use kore_abi::session::SlotRef;
use kore_abi::svc::{MmRepWire, MmReqWire};

use kore_rt::svc::{PeerBundlePart, PeerId, SvcProtocol};

use crate::service::{MemoryManager, MmRefusal, MmRequest, MmResponse};

/// Lower a request to its wire form (total).
pub fn encode_req(req: &MmRequest) -> MmReqWire {
    match *req {
        MmRequest::Alloc(p) => MmReqWire {
            op: MmReqWire::OP_ALLOC,
            contiguous: p.contiguous as u8,
            numa: p.numa.0.into(),
            slot: 0.into(),
            size: p.size.into(),
            align: p.align.into(),
            attrs: p.attrs.bits().into(),
            dma: p.dma as u8,
            reserved: [0; 5],
        },
        MmRequest::Release { slot } => MmReqWire {
            op: MmReqWire::OP_RELEASE,
            contiguous: 0,
            numa: 0.into(),
            slot: slot.index().into(),
            size: 0.into(),
            align: 0.into(),
            attrs: 0.into(),
            dma: 0,
            reserved: [0; 5],
        },
        MmRequest::Supply { size, align } => MmReqWire {
            op: MmReqWire::OP_SUPPLY,
            contiguous: 0,
            numa: 0.into(),
            slot: 0.into(),
            size: size.into(),
            align: align.into(),
            attrs: 0.into(),
            dma: 0,
            reserved: [0; 5],
        },
        MmRequest::Stats => bare(MmReqWire::OP_STATS),
        MmRequest::AdmitPeer { part, announced } => MmReqWire {
            contiguous: encode_part(part),
            slot: announced.index().into(),
            ..bare(MmReqWire::OP_ADMIT_PEER)
        },
        MmRequest::TakeAdmission => bare(MmReqWire::OP_TAKE_ADMISSION),
        MmRequest::RetirePeer { peer } => MmReqWire {
            numa: peer.arrival().into(),
            ..bare(MmReqWire::OP_RETIRE_PEER)
        },
        MmRequest::ConfirmRetired { peer } => MmReqWire {
            numa: peer.arrival().into(),
            ..bare(MmReqWire::OP_CONFIRM_RETIRED)
        },
        MmRequest::ClosePeer => bare(MmReqWire::OP_CLOSE_PEER),
    }
}

/// An `op` with every argument field zero — the shape each administration
/// request starts from, and exactly `OP_STATS`.
fn bare(op: u8) -> MmReqWire {
    MmReqWire {
        op,
        contiguous: 0,
        numa: 0.into(),
        slot: 0.into(),
        size: 0.into(),
        align: 0.into(),
        attrs: 0.into(),
        dma: 0,
        reserved: [0; 5],
    }
}

/// The recyclable-peer part codes every face protocol shares (power's and
/// accel's wires use the same four).
const fn encode_part(part: PeerBundlePart) -> u8 {
    match part {
        PeerBundlePart::Ring => 0,
        PeerBundlePart::Kick => 1,
        PeerBundlePart::ReturnRecv => 2,
        PeerBundlePart::Grant(index) => 0x80 | index,
    }
}

fn decode_part(code: u8) -> Option<PeerBundlePart> {
    Some(match code {
        0 => PeerBundlePart::Ring,
        1 => PeerBundlePart::Kick,
        2 => PeerBundlePart::ReturnRecv,
        value if value & 0x80 != 0 => PeerBundlePart::Grant(value & 0x7f),
        _ => return None,
    })
}

/// Lift a wire request — fail-closed.
pub fn decode_req(w: &MmReqWire) -> Option<MmRequest> {
    if w.reserved != [0; 5] {
        return None;
    }
    match w.op {
        MmReqWire::OP_ALLOC if w.slot.get() == 0 => Some(MmRequest::Alloc(AllocProps {
            size: w.size.get(),
            align: w.align.get(),
            numa: NumaNode(w.numa.get()),
            dma: decode_dma(w.dma)?,
            attrs: MemAttrs::from_bits(w.attrs.get())?,
            contiguous: match w.contiguous {
                0 => false,
                1 => true,
                _ => return None,
            },
        })),
        MmReqWire::OP_RELEASE
            if w.contiguous == 0
                && w.numa.get() == 0
                && w.size.get() == 0
                && w.align.get() == 0
                && w.attrs.get() == 0
                && w.dma == 0 =>
        {
            Some(MmRequest::Release {
                slot: SlotRef::from_wire(w.slot.get()),
            })
        }
        // A supply is CPU write-back R/W by nature: `Alloc`'s placement
        // vocabulary does not apply, and a set bit refuses rather than
        // acquiring accidental meaning (the reserved-byte discipline).
        MmReqWire::OP_SUPPLY
            if w.contiguous == 0
                && w.numa.get() == 0
                && w.slot.get() == 0
                && w.attrs.get() == 0
                && w.dma == 0 =>
        {
            Some(MmRequest::Supply {
                size: w.size.get(),
                align: w.align.get(),
            })
        }
        MmReqWire::OP_STATS if *w == bare(MmReqWire::OP_STATS) => Some(MmRequest::Stats),
        MmReqWire::OP_ADMIT_PEER
            if MmReqWire {
                contiguous: 0,
                slot: 0.into(),
                ..*w
            } == bare(MmReqWire::OP_ADMIT_PEER) =>
        {
            Some(MmRequest::AdmitPeer {
                part: decode_part(w.contiguous)?,
                announced: SlotRef::from_wire(w.slot.get()),
            })
        }
        MmReqWire::OP_TAKE_ADMISSION if *w == bare(MmReqWire::OP_TAKE_ADMISSION) => {
            Some(MmRequest::TakeAdmission)
        }
        MmReqWire::OP_RETIRE_PEER | MmReqWire::OP_CONFIRM_RETIRED
            if MmReqWire {
                numa: 0.into(),
                ..*w
            } == bare(w.op) =>
        {
            let peer = PeerId::from_arrival(w.numa.get())?;
            Some(if w.op == MmReqWire::OP_RETIRE_PEER {
                MmRequest::RetirePeer { peer }
            } else {
                MmRequest::ConfirmRetired { peer }
            })
        }
        MmReqWire::OP_CLOSE_PEER if *w == bare(MmReqWire::OP_CLOSE_PEER) => {
            Some(MmRequest::ClosePeer)
        }
        _ => None,
    }
}

/// Lower a response + the effected grant count (total).
pub fn encode_rep(rep: &MmResponse, grants: u8) -> MmRepWire {
    let (tag, refusal, memory) = match rep {
        MmResponse::Granted => (
            MmRepWire::TAG_GRANTED,
            0,
            kore_abi::monitor::MemorySnapshot::EMPTY,
        ),
        MmResponse::Released => (
            MmRepWire::TAG_RELEASED,
            0,
            kore_abi::monitor::MemorySnapshot::EMPTY,
        ),
        MmResponse::Stats(memory) => (MmRepWire::TAG_STATS, 0, *memory),
        MmResponse::Accepted => (
            MmRepWire::TAG_ACCEPTED,
            0,
            kore_abi::monitor::MemorySnapshot::EMPTY,
        ),
        MmResponse::PeerAdmitted { .. } => (
            MmRepWire::TAG_PEER_ADMITTED,
            0,
            kore_abi::monitor::MemorySnapshot::EMPTY,
        ),
        MmResponse::Refused(r) => (
            MmRepWire::TAG_REFUSED,
            match r {
                MmRefusal::BadProps => MmRepWire::REFUSE_BAD_PROPS,
                MmRefusal::Exhausted => MmRepWire::REFUSE_EXHAUSTED,
                MmRefusal::BadReturn => MmRepWire::REFUSE_BAD_RETURN,
                MmRefusal::Denied => MmRepWire::REFUSE_DENIED,
                MmRefusal::Quota => MmRepWire::REFUSE_QUOTA,
                MmRefusal::Pending => MmRepWire::REFUSE_PENDING,
            },
            kore_abi::monitor::MemorySnapshot::EMPTY,
        ),
    };
    let mut reserved = [0; 5];
    if let MmResponse::PeerAdmitted { peer } = rep {
        reserved[..2].copy_from_slice(&peer.arrival().to_le_bytes());
    }
    MmRepWire {
        tag,
        refusal,
        grants,
        reserved,
        memory,
    }
}

/// Lift a wire reply + its grant count — fail-closed.
pub fn decode_rep(w: &MmRepWire) -> Option<(MmResponse, u8)> {
    let admitted = w.tag == MmRepWire::TAG_PEER_ADMITTED;
    if w.reserved[if admitted { 2 } else { 0 }..] != [0; 5][if admitted { 2 } else { 0 }..] {
        return None;
    }
    let rep = match (w.tag, w.refusal) {
        (MmRepWire::TAG_ACCEPTED, 0)
            if w.grants == 0 && w.memory == kore_abi::monitor::MemorySnapshot::EMPTY =>
        {
            MmResponse::Accepted
        }
        (MmRepWire::TAG_PEER_ADMITTED, 0)
            if w.grants == 0 && w.memory == kore_abi::monitor::MemorySnapshot::EMPTY =>
        {
            MmResponse::PeerAdmitted {
                peer: PeerId::from_arrival(u16::from_le_bytes([w.reserved[0], w.reserved[1]]))?,
            }
        }
        (MmRepWire::TAG_GRANTED, 0)
            if w.grants == 1 && w.memory == kore_abi::monitor::MemorySnapshot::EMPTY =>
        {
            MmResponse::Granted
        }
        (MmRepWire::TAG_RELEASED, 0)
            if w.grants == 0 && w.memory == kore_abi::monitor::MemorySnapshot::EMPTY =>
        {
            MmResponse::Released
        }
        (MmRepWire::TAG_STATS, 0)
            if w.grants == 0 && w.memory.validate() && w.memory.available() =>
        {
            MmResponse::Stats(w.memory)
        }
        (MmRepWire::TAG_REFUSED, r)
            if w.grants == 0 && w.memory == kore_abi::monitor::MemorySnapshot::EMPTY =>
        {
            MmResponse::Refused(match r {
                MmRepWire::REFUSE_BAD_PROPS => MmRefusal::BadProps,
                MmRepWire::REFUSE_EXHAUSTED => MmRefusal::Exhausted,
                MmRepWire::REFUSE_BAD_RETURN => MmRefusal::BadReturn,
                MmRepWire::REFUSE_DENIED => MmRefusal::Denied,
                MmRepWire::REFUSE_QUOTA => MmRefusal::Quota,
                MmRepWire::REFUSE_PENDING => MmRefusal::Pending,
                _ => return None,
            })
        }
        _ => return None,
    };
    Some((rep, w.grants))
}

fn decode_dma(b: u8) -> Option<DmaClass> {
    Some(match b {
        x if x == DmaClass::None as u8 => DmaClass::None,
        x if x == DmaClass::Dma32 as u8 => DmaClass::Dma32,
        x if x == DmaClass::Dma64 as u8 => DmaClass::Dma64,
        _ => return None,
    })
}

impl SvcProtocol for MemoryManager {
    type Request = MmRequest;
    type Response = MmResponse;
    type ReqWire = MmReqWire;
    type RepWire = MmRepWire;

    /// One grant-carrying transfer per conforming reply: the `Alloc`
    /// extent view or the `Supply` lease root (`Release`/refusals carry
    /// none).
    const REPLY_GRANTS: usize = 1;

    fn request_grants(request: &MmReqWire) -> usize {
        usize::from(matches!(
            request.op,
            MmReqWire::OP_ALLOC | MmReqWire::OP_SUPPLY
        ))
    }

    /// Four serving pairs (slice D): pair 0 the founder-era protocol
    /// pair, pair 1 the FIRST SHELL's supply face (the `shell_client`
    /// prefab, conveyed through the pre-land bundle — where `run`'s
    /// `OP_SUPPLY` arrives), pair 2 INIT's face (delivered at its
    /// founding round; quiescent until the respawn slices), and pair 3
    /// exec's face — its provider snapshots and the **administration** of
    /// per-launch faces, whose components arrive on its return leg (the
    /// only fixed pair with one). Pairs past these are per-launch faces
    /// (recyclable peers, one per engine launch), each metered on its own
    /// lane. Supply quota is reclaimed through the separate
    /// kernel-authenticated lineage-completion reports, and the byte limit
    /// comes from the founder's validated boot policy.
    const SERVE_PAIRS: usize = 4;

    /// Both grant flavors outlive the call (an `Alloc` view backs a heap
    /// until released; a `Supply` root is conveyed onward at a spawn), so
    /// the channel is **retained**: the runtime rotates an owned landing
    /// per delivery and the client takes possession through the typed
    /// claims ([`RingMm`](crate::client::RingMm)).  Deliberately untyped
    /// (`grant_policy` stays `None`): two kinds lawfully cross —
    /// `LeasedRegion` views and `MemLease` roots — and each taker
    /// kind-checks at possession.
    fn retain_grant(index: usize) -> bool {
        index == 0
    }

    fn reply_matches(request: &MmReqWire, response: &MmResponse, grants: u8) -> bool {
        match (request.op, response) {
            (MmReqWire::OP_ALLOC | MmReqWire::OP_SUPPLY, MmResponse::Granted) => grants == 1,
            (MmReqWire::OP_ALLOC | MmReqWire::OP_SUPPLY, MmResponse::Refused(_))
            | (MmReqWire::OP_RELEASE, MmResponse::Released)
            | (MmReqWire::OP_RELEASE, MmResponse::Refused(_))
            | (MmReqWire::OP_STATS, MmResponse::Stats(_))
            | (MmReqWire::OP_TAKE_ADMISSION, MmResponse::PeerAdmitted { .. })
            | (
                MmReqWire::OP_ADMIT_PEER
                | MmReqWire::OP_TAKE_ADMISSION
                | MmReqWire::OP_RETIRE_PEER
                | MmReqWire::OP_CONFIRM_RETIRED
                | MmReqWire::OP_CLOSE_PEER,
                MmResponse::Refused(_),
            )
            | (
                MmReqWire::OP_ADMIT_PEER
                | MmReqWire::OP_RETIRE_PEER
                | MmReqWire::OP_CONFIRM_RETIRED
                | MmReqWire::OP_CLOSE_PEER,
                MmResponse::Accepted,
            ) => grants == 0,
            _ => false,
        }
    }

    fn encode_req(req: &MmRequest) -> MmReqWire {
        encode_req(req)
    }
    fn decode_req(wire: &MmReqWire) -> Option<MmRequest> {
        decode_req(wire)
    }
    fn encode_rep(rep: &MmResponse, grants: u8) -> MmRepWire {
        encode_rep(rep, grants)
    }
    fn decode_rep(wire: &MmRepWire) -> Option<(MmResponse, u8)> {
        decode_rep(wire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn props() -> AllocProps {
        AllocProps {
            size: 12_288,
            align: 8192,
            numa: NumaNode(3),
            dma: DmaClass::Dma32,
            attrs: MemAttrs::WRITE_BACK | MemAttrs::READ | MemAttrs::WRITE,
            contiguous: true,
        }
    }

    /// Every request/response round-trips through its wire form.
    #[test]
    fn round_trips() {
        for req in [
            MmRequest::Alloc(props()),
            MmRequest::Release {
                slot: SlotRef::from_wire(9),
            },
            MmRequest::Supply {
                size: 0x10_0000,
                align: 0,
            },
            MmRequest::Stats,
        ] {
            let back = decode_req(&encode_req(&req)).expect("decodes");
            // MmRequest has no PartialEq (AllocProps doesn't) — compare
            // through the wire form, which is exact bytes.
            assert_eq!(encode_req(&back), encode_req(&req));
        }
        for rep in [
            MmResponse::Granted,
            MmResponse::Released,
            MmResponse::Stats(kore_abi::monitor::MemorySnapshot::new(
                1, 2, 3, 4, 5, 6, 7, 8,
            )),
            MmResponse::Refused(MmRefusal::BadProps),
            MmResponse::Refused(MmRefusal::Exhausted),
            MmResponse::Refused(MmRefusal::BadReturn),
            MmResponse::Refused(MmRefusal::Denied),
            MmResponse::Refused(MmRefusal::Quota),
        ] {
            let grants = u8::from(matches!(rep, MmResponse::Granted));
            let (back, grants) = decode_rep(&encode_rep(&rep, grants)).expect("decodes");
            assert_eq!(back, rep);
            assert_eq!(grants, u8::from(matches!(rep, MmResponse::Granted)));
        }
    }

    /// Decode is fail-closed: unknown op/tag/refusal/class bytes refuse.
    #[test]
    fn decode_fails_closed() {
        let mut w = encode_req(&MmRequest::Alloc(props()));
        w.op = 0;
        assert!(decode_req(&w).is_none(), "zero op (unseeded) refuses");
        w.op = 77;
        assert!(decode_req(&w).is_none());
        let mut w = encode_req(&MmRequest::Alloc(props()));
        w.dma = 9;
        assert!(decode_req(&w).is_none(), "unknown DMA class refuses");
        let mut w = encode_req(&MmRequest::Alloc(props()));
        w.attrs = 0x8000.into();
        assert!(decode_req(&w).is_none(), "reserved attr bits refuse");
        let mut w = encode_req(&MmRequest::Alloc(props()));
        w.contiguous = 2;
        assert!(decode_req(&w).is_none(), "non-boolean contiguous refuses");
        let mut w = encode_req(&MmRequest::Alloc(props()));
        w.slot = 1.into();
        assert!(decode_req(&w).is_none(), "an alloc cannot smuggle a slot");
        let mut w = encode_req(&MmRequest::Release {
            slot: SlotRef::from_wire(9),
        });
        w.size = 1.into();
        assert!(decode_req(&w).is_none(), "release has no size vocabulary");
        let mut w = encode_req(&MmRequest::Stats);
        w.reserved[4] = 1;
        assert!(decode_req(&w).is_none(), "reserved request bytes refuse");
        // A supply speaks only size/align: any placement byte refuses
        // rather than acquiring accidental meaning.
        for poke in [
            |w: &mut MmReqWire| w.contiguous = 1,
            |w: &mut MmReqWire| w.numa = 2.into(),
            |w: &mut MmReqWire| w.slot = 7.into(),
            |w: &mut MmReqWire| w.attrs = 1.into(),
            |w: &mut MmReqWire| w.dma = 1,
        ] {
            let mut w = encode_req(&MmRequest::Supply {
                size: 0x10_0000,
                align: 0x1000,
            });
            poke(&mut w);
            assert!(
                decode_req(&w).is_none(),
                "a placement byte on a supply refuses"
            );
        }

        let mut r = encode_rep(&MmResponse::Granted, 1);
        r.tag = 0;
        assert!(decode_rep(&r).is_none());
        let mut r = encode_rep(&MmResponse::Refused(MmRefusal::BadProps), 0);
        r.refusal = 99;
        assert!(decode_rep(&r).is_none(), "unknown refusal refuses");
        let mut r = encode_rep(&MmResponse::Granted, 0);
        r.refusal = 1;
        assert!(
            decode_rep(&r).is_none(),
            "a refusal byte on a grant refuses"
        );
        assert!(
            decode_rep(&encode_rep(&MmResponse::Granted, 0)).is_none(),
            "Granted requires exactly one delivered capability"
        );
        assert!(
            decode_rep(&encode_rep(&MmResponse::Released, 1)).is_none(),
            "Released carries no grant"
        );
    }

    /// Every administration request round-trips, and each rejects a stray
    /// bit in a field its opcode does not carry.
    #[test]
    fn administration_requests_round_trip_and_refuse_stray_fields() {
        let peer = PeerId::from_arrival(0x0105).expect("a generation-1 face");
        for req in [
            MmRequest::AdmitPeer {
                part: PeerBundlePart::Ring,
                announced: SlotRef::from_wire(41),
            },
            MmRequest::AdmitPeer {
                part: PeerBundlePart::Grant(0),
                announced: SlotRef::from_wire(42),
            },
            MmRequest::TakeAdmission,
            MmRequest::RetirePeer { peer },
            MmRequest::ConfirmRetired { peer },
            MmRequest::ClosePeer,
        ] {
            let wire = encode_req(&req);
            let back = decode_req(&wire).expect("canonical encoding decodes");
            assert_eq!(encode_req(&back), wire);
            let mut stray = wire;
            stray.size = 1.into();
            assert!(decode_req(&stray).is_none(), "{req:?} carries no size");
        }
        // A fixed pair's index is not a face arrival.
        let mut fixed = encode_req(&MmRequest::RetirePeer { peer });
        fixed.numa = 3.into();
        assert!(decode_req(&fixed).is_none());
    }

    #[test]
    fn peer_admitted_carries_its_arrival_and_nothing_else() {
        let peer = PeerId::from_arrival(0x0206).unwrap();
        let wire = encode_rep(&MmResponse::PeerAdmitted { peer }, 0);
        assert_eq!(
            decode_rep(&wire),
            Some((MmResponse::PeerAdmitted { peer }, 0))
        );
        let mut stray = wire;
        stray.reserved[4] = 1;
        assert!(decode_rep(&stray).is_none());
        let mut accepted = encode_rep(&MmResponse::Accepted, 0);
        assert_eq!(decode_rep(&accepted), Some((MmResponse::Accepted, 0)));
        accepted.reserved[0] = 1;
        assert!(
            decode_rep(&accepted).is_none(),
            "only PeerAdmitted carries bytes"
        );
    }
}
