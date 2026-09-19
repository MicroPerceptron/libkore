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

use kore_rt::svc::SvcProtocol;

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
        MmRequest::Stats => MmReqWire {
            op: MmReqWire::OP_STATS,
            contiguous: 0,
            numa: 0.into(),
            slot: 0.into(),
            size: 0.into(),
            align: 0.into(),
            attrs: 0.into(),
            dma: 0,
            reserved: [0; 5],
        },
    }
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
        MmReqWire::OP_STATS
            if w.contiguous == 0
                && w.numa.get() == 0
                && w.slot.get() == 0
                && w.size.get() == 0
                && w.align.get() == 0
                && w.attrs.get() == 0
                && w.dma == 0 =>
        {
            Some(MmRequest::Stats)
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
        MmResponse::Refused(r) => (
            MmRepWire::TAG_REFUSED,
            match r {
                MmRefusal::BadProps => MmRepWire::REFUSE_BAD_PROPS,
                MmRefusal::Exhausted => MmRepWire::REFUSE_EXHAUSTED,
                MmRefusal::BadReturn => MmRepWire::REFUSE_BAD_RETURN,
                MmRefusal::Denied => MmRepWire::REFUSE_DENIED,
                MmRefusal::Quota => MmRepWire::REFUSE_QUOTA,
            },
            kore_abi::monitor::MemorySnapshot::EMPTY,
        ),
    };
    MmRepWire {
        tag,
        refusal,
        grants,
        reserved: [0; 5],
        memory,
    }
}

/// Lift a wire reply + its grant count — fail-closed.
pub fn decode_rep(w: &MmRepWire) -> Option<(MmResponse, u8)> {
    if w.reserved != [0; 5] {
        return None;
    }
    let rep = match (w.tag, w.refusal) {
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
    /// exec's provider-snapshot face. Pairs 1–3
    /// are post-less — no return legs, so `Release` is structurally
    /// impossible on them. Supply quota is reclaimed through the separate
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
            | (MmReqWire::OP_STATS, MmResponse::Stats(_)) => grants == 0,
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
}
