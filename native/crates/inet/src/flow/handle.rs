//! **[`Flow`]** — Tier 1, the working face of a [`FlowCap`].
//!
//! `res/INET_STACK.md` §6.2 names three tiers: Tier 0 is the window ABI, Tier
//! 1 is "zero-copy and borrow-based", and Tier 2 is socket-shaped and written
//! **in terms of Tier 1 with no privileged access**. This is Tier 1. The
//! [`Ifnet`] seam below already reserves a slot and publishes it; what it does
//! not do is scope those operations to *one authorized flow*, because the seam
//! is deliberately authority-free. A [`Flow`] is that join: an authority and
//! the interface it is enforced on, which the row's own rule makes a one-to-one
//! pairing (`OpenRefusal::VifAlreadyBound` — a holder wanting a second flow
//! opens a second vif).
//!
//! Not to be confused with `kore_svc_inet::client::Flow`, which is the *report*
//! a mint sends back over the wire — a proto and two endpoints, describing what
//! was granted. This is the thing the grant is exercised through, and the two
//! live in different crates because one crosses a ring and the other never
//! leaves the holder.
//!
//! # It borrows the capability; it does not consume it
//!
//! [`FlowCap`] is affine, and the three durations its own documentation names
//! are "borrow for one request, hold for a flow, store for a process". A face
//! is the first of those. Consuming the cap would make the handle itself the
//! authority — a *transfer*, which is what sending a capability to another
//! task means, not what using it locally means — and a holder could then
//! neither [`derive`](FlowCap::derive) a narrower child nor
//! [`revoke`](FlowCap::revoke) while a face existed, because the cap would be
//! somewhere else.
//!
//! Borrowing makes the type system state the two properties that matter:
//!
//! - **A face cannot outlive its authority.** `Flow<'a, I>` holds `&'a
//!   FlowCap`, so the cap is borrowed for exactly as long as the face lives,
//!   and `revoke(self)` — which takes the cap by value — will not compile
//!   while one is outstanding. Revocation cannot race a face; it waits for it.
//! - **A face cannot exceed its authority.** Everything it will transmit under
//!   is read out of the cap at [`bind`](Flow::bind) and nothing the caller
//!   passes can widen it.
//!
//! # Reserve forward, and only forward
//!
//! [`FrameDesc`](kore_driver::net::FrameDesc) carries `len`, `flags`,
//! `ethertype` and `reserved` and **no offset** (`res/INET_MIGRATION.md`
//! §5.8), so a frame starts at slot offset zero and headers are reserved
//! *forward*. [`Flow::reserve`] therefore takes the header length the caller
//! is about to write, and [`FlowTx::write`] fills what follows it. That is the
//! seam's discipline, unchanged — this tier adds authority to it and takes
//! nothing away.
//!
//! # Nothing is allocated and nothing is copied
//!
//! [`Flow::reserve`] hands out the ring slot itself: the `&mut [u8]` a caller
//! writes into *is* the memory the service will read, and [`Flow::drain`]
//! presents the receive slot borrowed rather than staged. Contrast
//! [`IpStack::input`](crate::netinet::IpStack::input), which copies one frame
//! into its own buffer because its protocol handlers need `&mut` on the
//! interface the frame borrows. A Tier 1 holder has no such handler, so it
//! pays nothing.
//!
//! # The receive side is the filter's, not a runtime check
//!
//! "The demultiplex *is* the interface" (`crates/driver/src/net/mod.rs`). The
//! classifier runs [`FrameFilter::accepts_frame`] — the L2 half *and* the flow
//! half — and only then writes a frame into this vif's window. The filter it
//! runs is the one [`FlowCap::frame_filter`] produced and the row installed
//! inside the mint. So a frame in the window is, by construction, a frame this
//! authority admits, and [`Flow::drain`] re-checks nothing: a second predicate
//! here would be a second source of truth about the same question, and the one
//! that is actually load-bearing is the one a hostile peer cannot reach.
//!
//! # Where the guarantee lives, and why not here
//!
//! Both directions are enforced, and neither is enforced by this module.
//!
//! Receive is the classifier's: `FrameFilter::accepts_frame` decides what
//! reaches a vif at all, so a face drains what was already admitted and
//! contains no predicate that could disagree. Transmit is
//! `services/net`'s `validate_tx` (`services/net/src/service/mod.rs:1556`),
//! which checks a sane length, an in-range slot, a declared EtherType that
//! both matches the bytes and is one `vif.filter.ethertype` accepts,
//! `head.src == vif.mac`, and — since the slice that closed the asymmetry —
//! the flow half, asked with the same `accepts_flow` the receive side asks,
//! against the same filter.
//!
//! Two consequences this module lives with deliberately:
//!
//! - **[`FlowTx::commit`] does not promise the frame belongs to the flow.**
//!   It promises the frame is well-formed, that its length is one the link
//!   can carry, and that its EtherType is the authority's. Whether the bytes
//!   are inside the flow is decided below, and a rejected descriptor is
//!   reported out of band — the service stalls that vif's own ring and
//!   counts a refusal — so `commit` returning `Ok` is not a claim that the
//!   frame left.
//! - **No check belongs here.** Adding one would put enforcement *above* the
//!   seam, where a client that skips the handle and writes the window record
//!   directly — the whole point of Tier 0 being speakable — walks straight
//!   past it. Worse, it would read as the enforcement and quietly become the
//!   only one.
//!
//! One claim nearby is still unbacked, and is named rather than left to be
//! discovered: `VifPolicy::authorize_source` and `source_mac` have no caller
//! outside their own tests, so the policy's per-frame claim about source
//! addresses is one the datapath does not make. `Powers::spoof` is read by
//! no enforcement site for the same reason. Both are fail-closed — a holder
//! *with* the power cannot forge either, because `validate_tx` pins
//! `head.src` unconditionally — so the gap is an unwired model, not an open
//! door.
//!
//! `the_face_itself_holds_no_transmit_gate` records the division as a test
//! rather than as prose.
//!
//! # Revocation is a refusal
//!
//! A cap outlives its vif's close and "stops resolving at its next use". For a
//! face, binding *is* that next use: [`Flow::bind`] takes the vif table's live
//! identifier and refuses [`FaceRefusal::Stale`] when the authority no longer
//! resolves against it. Checking once is not a weakening — a face holds the
//! interface exclusively for its whole life, so there is no second moment at
//! which a holder could learn anything new, and the enforcement that survives
//! a lying holder is the filter one layer down, which the frame row replaced
//! when it bumped the generation.
//!
//! # Where this deviates from the §6.2 sketch, and why
//!
//! | sketch | here | why |
//! |---|---|---|
//! | `flow.reserve()?` | `flow.reserve(headers)?` | no offset in `FrameDesc`, so headers are reserved forward and only the sender knows its own encapsulation (§2.2) |
//! | `tx.commit();` | `tx.commit()?` | both frame bounds are enforced where the total is known, and a refusal is not a panic |
//! | `for pkt in flow.drain()` | `flow.drain(\|pkt\| …)` | each frame borrows the receive slot, so an iterator would have to *lend*, which `Iterator` cannot express. The alternatives are a copy per frame or a callback; the callback is the one that keeps Tier 1's promise, and it is the posture [`Ifnet::poll`] already has |
//!
//! [`FrameFilter::accepts_frame`]: kore_driver::net::FrameFilter::accepts_frame

use kore_driver::net::frame::HEADER_LEN;
use kore_driver::net::{EtherType, RX_DEPTH, VifId};

use crate::ifnet::{Ifnet, IfnetError, TxSlot};

use super::cap::FlowCap;

/// Why a face could not be bound, or a frame not placed through one.
///
/// Distinct from [`FlowRefusal`](super::FlowRefusal), which answers the mint's
/// question — *may this authority exist* — and from [`IfnetError`], which
/// answers the seam's — *can this interface carry these bytes*. This answers
/// the third one: *may this holder do this, now*.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceRefusal {
    /// The authority no longer resolves against the vif's live identifier, so
    /// the slot it names belongs to someone else or to nobody.
    ///
    /// The one refusal that is a *revocation* rather than a misuse: a holder
    /// that sees this has not done anything wrong, it has outlived its grant.
    Stale,
    /// S0 — the authority covers nothing, so there is no direct datapath to
    /// face.
    ///
    /// The design working rather than a gap: a holder granted no scope keeps
    /// the row on its datapath (`res/INET_STACK.md` §4.4), and a Tier 1 face
    /// is precisely the thing that bypasses the row.
    Mediated,
    /// The authority's filter names no single protocol, so there is no one
    /// EtherType this face could publish under.
    ///
    /// Unreachable for an authority this crate mints —
    /// [`FlowCap::frame_filter`] always names IPv4 — and refused rather than
    /// assumed, so a filter that is later widened to a capture power cannot
    /// have a protocol silently picked for it here.
    NoProtocol,
    /// The transmit ring has no slot.
    ///
    /// Backpressure, not a failure: the holder has learned the service has not
    /// caught up, which is a scheduling fact. The seam says the same thing by
    /// answering `None`.
    NoSlot,
    /// The frame the caller described is not one the slot or the link can
    /// carry, with the seam's own reason.
    Frame(IfnetError),
}

/// **The capability's working face** — one authority, one interface.
///
/// Obtained from a [`FlowCap`] by [`bind`](Self::bind), which **borrows** it.
/// Not constructible any other way, so there is no route to a face that is not
/// a route through an authority.
///
/// The borrow is the whole of the scoping argument, and the two doctests below
/// are what hold it rather than the prose. Scoped use is lawful, and the
/// authority outlives the face:
///
/// ```
/// use kore_inet::{Flow, FlowCap, MOCK_SCRATCH_BYTES, MockIfnet, Powers, Scope};
/// use kore_driver::net::flow::{FlowMatch, IpProto, Ipv4Addr};
/// use kore_driver::net::{MacAddr, VifId};
///
/// let vif = VifId { index: 0, generation: 1 };
/// let mut scratch = [0u8; MOCK_SCRATCH_BYTES];
/// let mut net = MockIfnet::new(&mut scratch, MacAddr([2, 0, 0, 0, 0, 1])).unwrap();
/// let cap = FlowCap::mint(
///     vif,
///     Scope::Flow(FlowMatch::connection(
///         IpProto::UDP,
///         (Ipv4Addr([10, 0, 2, 15]), 5000),
///         (Ipv4Addr([10, 0, 2, 2]), 53),
///     )),
///     Powers::NONE,
/// )
/// .unwrap();
/// let face = Flow::bind(&cap, &mut net, vif).unwrap();
/// assert_eq!(face.vif(), vif);
/// drop(face);
/// cap.revoke(); // the face is gone, so the authority can be given up
/// ```
///
/// Revoking *while* a face is outstanding is not, because
/// [`FlowCap::revoke`] takes the capability by value and the face still has it
/// borrowed — so a revocation cannot race a face, it waits for one:
///
/// ```compile_fail
/// use kore_inet::{Flow, FlowCap, MOCK_SCRATCH_BYTES, MockIfnet, Powers, Scope};
/// use kore_driver::net::flow::{FlowMatch, IpProto, Ipv4Addr};
/// use kore_driver::net::{MacAddr, VifId};
///
/// let vif = VifId { index: 0, generation: 1 };
/// let mut scratch = [0u8; MOCK_SCRATCH_BYTES];
/// let mut net = MockIfnet::new(&mut scratch, MacAddr([2, 0, 0, 0, 0, 1])).unwrap();
/// let cap = FlowCap::mint(
///     vif,
///     Scope::Flow(FlowMatch::connection(
///         IpProto::UDP,
///         (Ipv4Addr([10, 0, 2, 15]), 5000),
///         (Ipv4Addr([10, 0, 2, 2]), 53),
///     )),
///     Powers::NONE,
/// )
/// .unwrap();
/// let face = Flow::bind(&cap, &mut net, vif).unwrap();
/// cap.revoke(); // the face still borrows it — a revocation cannot race one
/// let _ = face.vif();
/// ```
pub struct Flow<'a, I: Ifnet> {
    cap: &'a FlowCap,
    net: &'a mut I,
    /// Read out of the authority's own filter at bind, never chosen by a
    /// caller — see [`Self::ethertype`].
    ethertype: EtherType,
}

impl<'a, I: Ifnet> Flow<'a, I> {
    /// Bind `cap` to the interface it is enforced on.
    ///
    /// `live` is the vif table's current identifier for the slot, supplied by
    /// whoever knows it — the founded client's device face, or the row in a
    /// test. A holder could of course pass its own cap's identifier back and
    /// defeat this check; it would gain nothing, because this is the local,
    /// fail-fast half of revocation and the half that actually holds is the
    /// filter the frame row replaced when it bumped the generation. Checking
    /// here is what turns "the window silently stopped delivering" into a
    /// refusal naming why.
    pub fn bind(cap: &'a FlowCap, net: &'a mut I, live: VifId) -> Result<Self, FaceRefusal> {
        if !cap.resolves_against(live) {
            return Err(FaceRefusal::Stale);
        }
        // The *same* filter the row installed, read back rather than rebuilt:
        // what this face may transmit is then exactly what the vif admits on
        // receive, and the two cannot drift apart. `None` is S0.
        let filter = cap.frame_filter().ok_or(FaceRefusal::Mediated)?;
        let ethertype = match filter.ethertype.listed() {
            [t] => *t,
            _ => return Err(FaceRefusal::NoProtocol),
        };
        Ok(Self { cap, net, ethertype })
    }

    /// The authority this face exercises.
    pub fn cap(&self) -> &FlowCap {
        self.cap
    }

    /// The vif this face is bound to, generation included.
    pub fn vif(&self) -> VifId {
        self.cap.vif()
    }

    /// The EtherType every frame this face publishes carries.
    ///
    /// Derived, never passed in: a holder of an IPv4 flow cannot publish an
    /// ARP or a peer-link frame through its face, because the value comes out
    /// of the filter the row installed on its behalf.
    pub fn ethertype(&self) -> EtherType {
        self.ethertype
    }

    /// The interface.
    pub fn ifnet(&self) -> &I {
        self.net
    }

    /// The interface, mutably.
    ///
    /// Not a back door. A holder owns its window before it binds a face —
    /// authority here is the *filter* on the receive side and the service's
    /// validation on the transmit side, never the absence of a `&mut`. What
    /// withholding this would buy is nothing; what it would cost is forcing a
    /// holder to drop its face whenever it wants an operation the face does
    /// not yet cover, such as the IPv4 encapsulation
    /// [`netinet::ip::output`](crate::netinet::ip) already owns.
    pub fn ifnet_mut(&mut self) -> &mut I {
        self.net
    }

    /// Reserve one transmit slot with `headers` bytes held at the front.
    ///
    /// The reservation is forward and it is structural: the payload region is
    /// reachable only through [`FlowTx::write`], which hands back what follows
    /// the reservation, so a caller cannot write a payload at offset zero and
    /// then discover it has nowhere to put its headers.
    ///
    /// Zero copy: the region handed to the closure is the ring slot.
    pub fn reserve(&mut self, headers: usize) -> Result<FlowTx<'_, I>, FaceRefusal> {
        let ethertype = self.ethertype;
        let mut slot = self.net.reserve().ok_or(FaceRefusal::NoSlot)?;
        // Taken and dropped: the reservation is recorded in the slot, and a
        // reservation the slot cannot hold is refused here rather than at
        // `write`, where a caller would already have built its headers.
        if slot.reserve_headers(headers).is_none() {
            return Err(FaceRefusal::Frame(IfnetError::TooLong));
        }
        Ok(FlowTx {
            slot,
            headers,
            payload: 0,
            ethertype,
        })
    }

    /// Present each frame the classifier admitted to this vif.
    ///
    /// `f` is called once per frame, with the receive slot **borrowed** —
    /// nothing is staged and nothing is copied. Returns whether more may be
    /// waiting, so one sweep is bounded and the caller comes back: the
    /// drive-cycle posture [`Ifnet::poll`] already uses rather than draining
    /// to exhaustion under an arrival rate nobody controls.
    ///
    /// Every frame is yielded. The flow predicate ran one layer down, in the
    /// classifier, before any of these bytes reached the window; re-running it
    /// here would second-guess the only check a hostile peer cannot reach.
    pub fn drain<F>(&mut self, mut f: F) -> bool
    where
        F: FnMut(FlowFrame<'_>),
    {
        // The ring's own depth bounds the sweep — a frame count that exists in
        // the ABI, not a number chosen here.
        for _ in 0..RX_DEPTH {
            let more = self
                .net
                .poll(|frame, ethertype| f(FlowFrame { frame, ethertype }));
            if !more {
                return false;
            }
        }
        true
    }
}

/// One reserved transmit slot, scoped to the flow that reserved it.
///
/// Consumed by [`commit`](Self::commit), so a frame is published exactly once;
/// dropped without it, the slot is released and nothing was published, because
/// the seam's slot is borrowed from the interface rather than owned.
pub struct FlowTx<'s, I: Ifnet + 's> {
    slot: I::TxSlot<'s>,
    headers: usize,
    payload: usize,
    ethertype: EtherType,
}

impl<I: Ifnet> FlowTx<'_, I> {
    /// The reserved header region, to fill in place.
    ///
    /// Disjoint from the payload region by construction, so the order the two
    /// are written in does not matter.
    pub fn headers(&mut self) -> &mut [u8] {
        self.slot.headers_mut()
    }

    /// Bytes held at the front for headers.
    pub fn reserved(&self) -> usize {
        self.headers
    }

    /// Write the payload, in place, after the reserved headers.
    ///
    /// `f` receives the ring slot's own bytes and returns how many of them it
    /// wrote; that count is the frame's payload length, so a header length and
    /// a frame length cannot disagree.
    ///
    /// A closure claiming more than it was handed is **refused**, not clamped:
    /// committing the claim would publish slot bytes the caller never wrote,
    /// which is a disclosure rather than a truncation.
    pub fn write<F>(&mut self, f: F) -> Result<(), FaceRefusal>
    where
        F: FnOnce(&mut [u8]) -> usize,
    {
        // `reserve_headers` is the only route to the payload region and is
        // idempotent in its length, so asking again re-borrows the same region
        // rather than reserving a second one.
        let region = self
            .slot
            .reserve_headers(self.headers)
            .ok_or(FaceRefusal::Frame(IfnetError::TooLong))?;
        let room = region.len();
        let written = f(region);
        if written > room {
            return Err(FaceRefusal::Frame(IfnetError::TooLong));
        }
        self.payload = written;
        Ok(())
    }

    /// Publish the frame.
    ///
    /// The EtherType is the face's, and the length is the reservation plus
    /// what [`write`](Self::write) reported — neither is a caller's assertion,
    /// which is why both frame bounds can be enforced here against a number
    /// that is known to be true.
    ///
    /// **This does not say the frame belongs to the flow.** `validate_tx` one
    /// layer down checks the length, the declared-against-actual EtherType
    /// against the vif's filter, and the source MAC against the vif's — and
    /// not `filter.flow`. So the addresses and ports inside the bytes are
    /// unconstrained on transmit today, and refusing them here would only
    /// constrain holders polite enough to use this type. See the module
    /// documentation, and `the_face_itself_holds_no_transmit_gate`.
    pub fn commit(self) -> Result<(), FaceRefusal> {
        self.slot
            .commit(self.payload, self.ethertype)
            .map_err(FaceRefusal::Frame)
    }
}

/// One received frame, borrowed from the receive slot.
///
/// A view, narrowed as a layer strips a header — the receive mirror
/// `res/INET_STACK.md` §3 describes, and the reason nothing here owns bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlowFrame<'f> {
    frame: &'f [u8],
    ethertype: EtherType,
}

impl<'f> FlowFrame<'f> {
    /// The whole frame, from the Ethernet header.
    pub fn frame(&self) -> &'f [u8] {
        self.frame
    }

    /// The EtherType the descriptor declared.
    pub fn ethertype(&self) -> EtherType {
        self.ethertype
    }

    /// The frame after its Ethernet header — what the flow's own layer reads.
    ///
    /// Empty for a frame shorter than its own header, which no [`Ifnet`] in
    /// this crate can produce (both validate a received length against
    /// `FRAME_MIN` first). Empty rather than a panic, because a third impl
    /// that got this wrong should surface as a frame nobody can parse and not
    /// as a dead holder.
    pub fn payload(&self) -> &'f [u8] {
        self.frame.get(HEADER_LEN..).unwrap_or(&[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::{Powers, Scope};
    use crate::ifnet::{MOCK_SCRATCH_BYTES, MockIfnet};
    use crate::netinet::ip::IPV4_HEADER_LEN;
    use kore_driver::net::flow::{FlowMatch, IpProto, Ipv4Addr};
    use kore_driver::net::{FRAME_MAX, FrameFilter, FrameHead, MacAddr};

    const STATION: MacAddr = MacAddr([0x02, 0, 0, 0, 0, 0x01]);
    const GATEWAY: MacAddr = MacAddr([0x02, 0, 0, 0, 0, 0x02]);
    const ME: Ipv4Addr = Ipv4Addr([10, 0, 2, 15]);
    const PEER_A: Ipv4Addr = Ipv4Addr([93, 184, 216, 34]);
    const PEER_B: Ipv4Addr = Ipv4Addr([142, 250, 72, 46]);

    /// A UDP header: two ports, a length and a checksum.
    const UDP_HEADER_LEN: usize = 8;

    /// Ethernet + IPv4 + UDP, the encapsulation a flow holder reserves.
    const RESERVE: usize = HEADER_LEN + IPV4_HEADER_LEN + UDP_HEADER_LEN;

    /// A window must be 64-byte aligned, which a bare stack array is not.
    #[repr(align(64))]
    struct Aligned<const N: usize>([u8; N]);

    fn vif_a() -> VifId {
        VifId {
            index: 0,
            generation: 1,
        }
    }

    fn vif_b() -> VifId {
        VifId {
            index: 1,
            generation: 1,
        }
    }

    /// The authority a client connecting outward holds.
    fn cap(vif: VifId, local_port: u16, peer: (Ipv4Addr, u16)) -> FlowCap {
        FlowCap::mint(
            vif,
            Scope::Flow(FlowMatch::connection(IpProto::UDP, (ME, local_port), peer)),
            Powers::NONE,
        )
        .expect("S1 is mintable")
    }

    /// The smallest IPv4 datagram a flow can classify: a header, a protocol
    /// and a port pair, framed as a peer would put it on the wire.
    fn ip_frame(src: (Ipv4Addr, u16), dst: (Ipv4Addr, u16), out: &mut [u8]) -> usize {
        FrameHead {
            dst: STATION,
            src: GATEWAY,
            ethertype: EtherType::IPV4,
        }
        .write(out)
        .expect("a header fits");
        // The header, plus the four bytes of port pair a flow matches on.
        let len = IPV4_HEADER_LEN + 4;
        let ip = &mut out[HEADER_LEN..HEADER_LEN + len];
        ip[0] = 0x45; // version 4, IHL 5
        ip[2..4].copy_from_slice(&(len as u16).to_be_bytes());
        ip[8] = 64; // TTL
        ip[9] = IpProto::UDP.0;
        ip[12..16].copy_from_slice(&src.0.0);
        ip[16..20].copy_from_slice(&dst.0.0);
        ip[IPV4_HEADER_LEN..IPV4_HEADER_LEN + 2].copy_from_slice(&src.1.to_be_bytes());
        ip[IPV4_HEADER_LEN + 2..IPV4_HEADER_LEN + 4].copy_from_slice(&dst.1.to_be_bytes());
        HEADER_LEN + len
    }

    /// Deliver through the **shipping** classifier predicate, as
    /// `services/net` does — never through test logic, so what passes here is
    /// what passes in the row.
    fn deliver(net: &mut MockIfnet<'_>, filter: &FrameFilter, frame: &[u8]) -> bool {
        let head = FrameHead::parse(frame).expect("well formed");
        if !filter.accepts_frame(&head, &frame[HEADER_LEN..], STATION) {
            return false;
        }
        net.inject(frame, head.ethertype).expect("injects");
        true
    }

    // ── transmit ────────────────────────────────────────────────────────

    /// **The zero-copy witness.** The address the caller's closure wrote at is
    /// the address the interface published from — not an equal copy of those
    /// bytes, the same bytes. Nothing between the holder and the ring.
    #[test]
    fn a_flow_sends_the_bytes_the_caller_wrote_in_place() {
        let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut net = MockIfnet::new(&mut scratch.0, STATION).expect("scratch");
        let cap = cap(vif_a(), 5000, (PEER_A, 53));
        let body = b"tier-one";

        let mut wrote_at = 0usize;
        {
            let mut flow = Flow::bind(&cap, &mut net, vif_a()).expect("the authority is live");
            assert_eq!(
                flow.ethertype(),
                EtherType::IPV4,
                "the face's protocol comes from the authority, not the caller"
            );

            let mut tx = flow.reserve(RESERVE).expect("an empty ring has a slot");
            tx.write(|buf| {
                wrote_at = buf.as_ptr() as usize;
                buf[..body.len()].copy_from_slice(body);
                body.len()
            })
            .expect("writes");

            // The headers the holder owns, filled in place after the payload
            // — the two regions are disjoint, so the order is free.
            FrameHead {
                dst: GATEWAY,
                src: STATION,
                ethertype: EtherType::IPV4,
            }
            .write(tx.headers())
            .expect("the reservation covers an Ethernet header");

            tx.commit().expect("commits");
        }

        let (sent, ethertype) = net.last_sent().expect("a frame was published");
        assert_eq!(sent.len(), RESERVE + body.len(), "headers and payload");
        assert_eq!(ethertype, EtherType::IPV4);
        assert_eq!(
            sent.as_ptr() as usize + RESERVE,
            wrote_at,
            "the caller wrote into the ring slot itself — nothing was copied"
        );
        assert_eq!(&sent[RESERVE..], body);
        let head = FrameHead::parse(sent).expect("a well-formed header");
        assert_eq!(head.src, STATION);
    }

    /// The reservation invariant, through the face: the payload begins after
    /// the reserved headers, and writing it does not reach into them.
    #[test]
    fn the_payload_begins_after_the_reserved_headers() {
        let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut net = MockIfnet::new(&mut scratch.0, STATION).expect("scratch");
        let cap = cap(vif_a(), 5000, (PEER_A, 53));
        let mut flow = Flow::bind(&cap, &mut net, vif_a()).expect("live");

        let mut tx = flow.reserve(RESERVE).expect("slot");
        assert_eq!(tx.reserved(), RESERVE);
        assert_eq!(tx.headers().len(), RESERVE, "held at the front, unwritten");

        tx.write(|buf| {
            assert_eq!(
                buf.len(),
                FRAME_MAX - RESERVE,
                "the payload is the rest of the frame"
            );
            buf.fill(0x11);
            buf.len()
        })
        .expect("writes");

        assert!(
            tx.headers().iter().all(|&b| b == 0),
            "writing the payload did not reach into the header region"
        );
    }

    /// Both frame bounds hold. The oversize case is caught **earlier** than
    /// the seam's own commit check, because a payload length here is reported
    /// by the closure that wrote it rather than asserted by the caller — so a
    /// frame longer than the slot cannot even be described.
    #[test]
    fn both_frame_bounds_are_enforced() {
        let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut net = MockIfnet::new(&mut scratch.0, STATION).expect("scratch");
        let cap = cap(vif_a(), 5000, (PEER_A, 53));

        {
            let mut flow = Flow::bind(&cap, &mut net, vif_a()).expect("live");

            let mut tx = flow.reserve(HEADER_LEN).expect("slot");
            assert_eq!(
                tx.write(|buf| buf.len() + 1),
                Err(FaceRefusal::Frame(IfnetError::TooLong)),
                "a claim beyond the region is refused, never published"
            );
            drop(tx);

            // Shorter than an Ethernet header: refused where the total is
            // known, which is the commit.
            let mut tx = flow.reserve(4).expect("slot");
            tx.write(|_| 0).expect("writes nothing");
            assert_eq!(
                tx.commit(),
                Err(FaceRefusal::Frame(IfnetError::TooShort)),
                "a runt is refused at commit"
            );

            // A reservation no slot can hold is refused before a caller has
            // built anything into it.
            assert_eq!(
                flow.reserve(FRAME_MAX + 1).err(),
                Some(FaceRefusal::Frame(IfnetError::TooLong))
            );
        }

        assert_eq!(net.sent_count(), 0, "no refusal published a frame");
    }

    /// The ring is the backpressure, and the face reports it as the fact it
    /// is rather than as a failure.
    #[test]
    fn a_full_ring_refuses_a_reservation() {
        let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut net = MockIfnet::new(&mut scratch.0, STATION).expect("scratch");
        net.set_tx_full(true);
        let cap = cap(vif_a(), 5000, (PEER_A, 53));
        let mut flow = Flow::bind(&cap, &mut net, vif_a()).expect("live");
        assert_eq!(flow.reserve(RESERVE).err(), Some(FaceRefusal::NoSlot));
    }

    /// **The face adds no gate of its own, and that is deliberate.** A face
    /// will publish a datagram sourced outside its authority, because the
    /// face is not where the question is asked.
    ///
    /// It cannot be. A client that bypasses this type and writes the window
    /// record directly reaches the same ring, so a predicate here would
    /// bound only the well-behaved. The gate is `services/net`'s
    /// `validate_tx`, below the seam, where no client can go around it —
    /// and since the slice that taught that gate the flow dimension, an
    /// out-of-flow frame is refused there.
    ///
    /// What this pins is the division: the face is a convenience over a
    /// ring, the authority is the filter the row installed, and the two are
    /// not the same object. Were a check added here it would read as the
    /// enforcement and quietly become the only one.
    #[test]
    fn the_face_itself_holds_no_transmit_gate() {
        let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut net = MockIfnet::new(&mut scratch.0, STATION).expect("scratch");
        let cap = cap(vif_a(), 5000, (PEER_A, 53));
        let filter = cap.frame_filter().expect("S1 enforces a filter");

        // A datagram from a local port this authority never bound, to a peer
        // it was never minted for.
        let mut forged = [0u8; 64];
        let len = ip_frame((ME, 31337), (PEER_B, 9), &mut forged);
        let datagram = &forged[HEADER_LEN..len];
        assert!(
            !filter.accepts_flow(EtherType::IPV4, datagram),
            "inbound, this authority's own filter rejects these bytes"
        );

        {
            let mut flow = Flow::bind(&cap, &mut net, vif_a()).expect("live");
            let mut tx = flow.reserve(HEADER_LEN).expect("slot");
            tx.write(|buf| {
                buf[..datagram.len()].copy_from_slice(datagram);
                datagram.len()
            })
            .expect("writes");
            // The source MAC the transmit gate *does* pin, so this frame
            // passes every check that gate actually makes.
            FrameHead {
                dst: GATEWAY,
                src: STATION,
                ethertype: EtherType::IPV4,
            }
            .write(tx.headers())
            .expect("fits");
            assert!(
                tx.commit().is_ok(),
                "the face publishes it; the gate that refuses it is below"
            );
        }

        let (sent, ethertype) = net.last_sent().expect("published");
        let head = FrameHead::parse(sent).expect("well formed");
        assert!(
            filter.ethertype.accepts(ethertype) && head.src == STATION,
            "the two dimensions transmit does enforce are both satisfied"
        );
        assert!(
            !filter.accepts_flow(ethertype, &sent[HEADER_LEN..]),
            "and the authority's own filter is what refuses it, one layer down"
        );
    }

    // ── receive ─────────────────────────────────────────────────────────

    /// **The scoping witness.** A face drains what the classifier admitted,
    /// and the classifier is what keeps another conversation out — so a face
    /// bound to A's authority never sees B's datagram, and the face itself
    /// contains no predicate that could disagree.
    #[test]
    fn a_face_cannot_reach_a_flow_it_was_not_minted_for() {
        let mut scratch_a = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut scratch_b = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        // Both clients answer to the *same* station address — the shape
        // `AddressMode::Shared` forces, where an L2 filter provably cannot
        // tell them apart.
        let mut net_a = MockIfnet::new(&mut scratch_a.0, STATION).expect("scratch");
        let mut net_b = MockIfnet::new(&mut scratch_b.0, STATION).expect("scratch");

        let cap_a = cap(vif_a(), 5000, (PEER_A, 53));
        let cap_b = cap(vif_b(), 5001, (PEER_B, 53));
        let filter_a = cap_a.frame_filter().expect("S1 enforces a filter");
        let filter_b = cap_b.frame_filter().expect("S1 enforces a filter");

        let mut bytes = [0u8; 64];
        let len = ip_frame((PEER_A, 53), (ME, 5000), &mut bytes);
        assert!(deliver(&mut net_a, &filter_a, &bytes[..len]), "A's own flow");
        assert!(
            !deliver(&mut net_b, &filter_b, &bytes[..len]),
            "B's filter rejected it — same station address, another flow"
        );

        let mut seen = 0usize;
        let mut payload_at = 0usize;
        {
            let mut flow = Flow::bind(&cap_a, &mut net_a, vif_a()).expect("live");
            flow.drain(|pkt| {
                seen += 1;
                payload_at = pkt.payload().as_ptr() as usize;
                assert_eq!(pkt.ethertype(), EtherType::IPV4);
                assert_eq!(pkt.frame().len(), len);
                assert_eq!(
                    pkt.payload().len(),
                    len - HEADER_LEN,
                    "the payload is the frame past its Ethernet header"
                );
            });
        }
        assert_eq!(seen, 1, "A drained its own datagram");
        assert_ne!(payload_at, 0, "the frame was borrowed from the slot");

        let mut seen_b = 0usize;
        {
            let mut flow = Flow::bind(&cap_b, &mut net_b, vif_b()).expect("live");
            flow.drain(|_| seen_b += 1);
        }
        assert_eq!(seen_b, 0, "B saw nothing of A's conversation");

        // And the authority itself refuses to face another holder's vif, so
        // the two halves of the isolation agree.
        assert_eq!(
            Flow::bind(&cap_a, &mut net_b, vif_b()).err(),
            Some(FaceRefusal::Stale),
            "A's authority does not resolve against B's vif"
        );
    }

    /// An idle interface drains nothing, so an empty sweep is distinguishable
    /// from a delivered frame.
    #[test]
    fn an_idle_face_drains_nothing() {
        let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut net = MockIfnet::new(&mut scratch.0, STATION).expect("scratch");
        let cap = cap(vif_a(), 5000, (PEER_A, 53));
        let mut flow = Flow::bind(&cap, &mut net, vif_a()).expect("live");
        let mut called = false;
        assert!(!flow.drain(|_| called = true));
        assert!(!called);
    }

    // ── the authority ───────────────────────────────────────────────────

    /// Revocation is the generation tag, and it surfaces as a refusal naming
    /// itself — not a panic, and not a face that quietly transmits into a
    /// window its grant no longer covers.
    #[test]
    fn a_revoked_generation_refuses_a_face() {
        let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut net = MockIfnet::new(&mut scratch.0, STATION).expect("scratch");
        let cap = cap(vif_a(), 5000, (PEER_A, 53));

        assert!(Flow::bind(&cap, &mut net, vif_a()).is_ok());

        let next_tenant = VifId {
            index: 0,
            generation: 2,
        };
        assert_eq!(
            Flow::bind(&cap, &mut net, next_tenant).err(),
            Some(FaceRefusal::Stale),
            "the slot's next tenant is not this authority's"
        );
        let elsewhere = VifId {
            index: 4,
            generation: 1,
        };
        assert_eq!(
            Flow::bind(&cap, &mut net, elsewhere).err(),
            Some(FaceRefusal::Stale),
            "another slot entirely"
        );
    }

    /// S0 has no face: it grants nothing, so there is no direct datapath to
    /// scope, and the row stays on that holder's path. A gap in the ladder
    /// would show up here as a face that transmitted with no authority.
    #[test]
    fn the_mediated_grant_has_no_face() {
        let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut net = MockIfnet::new(&mut scratch.0, STATION).expect("scratch");
        let cap = FlowCap::mediated(vif_a());
        assert_eq!(
            Flow::bind(&cap, &mut net, vif_a()).err(),
            Some(FaceRefusal::Mediated)
        );
    }

    /// A face is scoped *use*, not transfer: it borrows the authority, so the
    /// authority survives it and can still be derived from and revoked.
    #[test]
    fn a_face_borrows_its_authority_and_the_authority_survives() {
        let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut net = MockIfnet::new(&mut scratch.0, STATION).expect("scratch");
        let cap = cap(vif_a(), 5000, (PEER_A, 53));
        {
            let flow = Flow::bind(&cap, &mut net, vif_a()).expect("live");
            assert_eq!(flow.vif(), vif_a());
            assert_eq!(flow.cap().powers(), Powers::NONE);
        }
        // The authority outlived its face, so it can still be attenuated —
        // which a face that had *consumed* it could not offer.
        assert!(cap.derive(Scope::Mediated, Powers::NONE).is_ok());
        cap.revoke();
    }
}
