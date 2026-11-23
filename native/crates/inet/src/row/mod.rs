//! **The arbitration engine** — the half of the inet row that must be exactly
//! one party, and nothing else.
//!
//! `res/INET_STACK.md` §3's split, made concrete. The row owns the two things
//! that are singular — the machine's [`Identity`] and the [`PortTable`] — and
//! turns a request into a [`FlowCap`]. It is **not** on the datapath: once a
//! cap is minted, the holder transmits and receives through its own vif and
//! never calls here again.
//!
//! # Why this is a library and not a service
//!
//! The same reason the rest of `crates/inet` is: an engine that owns no
//! transport can be driven by a founded row, by a test, or by a client that
//! happens to hold both halves. `services/inet` is the thin composition that
//! gives this a boot row and a serving pair; everything decided here is
//! decided the same way in all three.
//!
//! # Minting is where the checking happens
//!
//! [`InetRow::open_flow`] is the one place that asks: is there an identity at
//! all, is this port free, may this caller bind it. The capability it returns
//! *is* that answer. Nothing downstream re-checks, because nothing downstream
//! could learn anything the mint did not already establish.
//!
//! # Minting is also where the enforcement is installed
//!
//! A capability whose enforcement is missing is worse than a refusal: the
//! holder believes it is isolated and is not. So a mint is not two steps with
//! a window between them — the [`FrameFilter`] that
//! [`FlowCap::frame_filter`] produces is installed through the
//! [`FlowInstall`] seam *inside* the mint, and an install that refuses unwinds
//! the port binding and returns the refusal. A `FlowCap` leaves this module
//! only once the predicate backing it is running.
//!
//! The one authority that installs nothing is S0 ([`Scope::Mediated`]), whose
//! `frame_filter` is `None` — because it grants no scope, so there is nothing
//! to enforce and the row stays on that holder's datapath. That is the design
//! working, not a gap.
//!
//! [`FrameFilter`]: kore_driver::net::FrameFilter

pub mod identity;
pub mod ports;

pub use identity::{Identity, IdentityRefusal};
pub use ports::{BindRefusal, PortTable};

use kore_driver::net::flow::{EndpointMatch, FlowMatch, IpProto, Ipv4Addr, ProtoMatch};
use kore_driver::net::{MAX_VIFS, MacAddr, VifId};

use crate::flow::{FlowCap, FlowInstall, InstallRefusal, Powers, Scope};

/// Why a flow could not be opened.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpenRefusal {
    /// The row has no address yet, so no flow can name a local end.
    NoIdentity,
    /// The local port could not be bound.
    Port(BindRefusal),
    /// The peer is off-link and no gateway is configured, so the flow could
    /// never route.
    Unreachable,
    /// The vif already enforces a flow this row minted.
    ///
    /// A vif carries exactly one [`FrameFilter`](kore_driver::net::FrameFilter),
    /// so a second flow installed on it would silently stop enforcing the
    /// first — leaving that holder believing it is listening for something it
    /// will never receive. Refused here, where it can be read, rather than
    /// discovered later as "the connection stopped working". A holder wanting
    /// a second flow opens a second vif.
    VifAlreadyBound,
    /// The caller asked for raw L2 (S3) and its grant does not carry
    /// [`OpenGrant::raw_l2`].
    ///
    /// Named separately from every other refusal because it is the one an
    /// operator answers by widening a *composition's* grant rather than by
    /// changing anything the caller did — and because nothing in this tree
    /// sets it, so a refusal here is the expected answer rather than a
    /// misconfiguration (`res/INET_STACK.md` §9).
    RawL2NotGranted,
    /// A raw-L2 authority this row minted already covers that station
    /// address, on another vif.
    ///
    /// Two S3 holders at one station address are not two isolated holders:
    /// each one's filter admits every frame addressed to the station, so
    /// each receives the other's traffic in full, without either having
    /// asked for [`Powers::promiscuous`](crate::flow::Powers::promiscuous).
    /// That is a mutual escape of the vif boundary, and it is the half of
    /// the shared-addressing hazard this row can see and therefore must
    /// refuse.
    StationAlreadyRaw,
    /// The authority was minted but its enforcement could not be installed,
    /// so the mint was unwound and nothing was granted.
    ///
    /// Carries the specific power or resource that was missing, because
    /// "could not install" alone leaves an operator unable to tell an unwired
    /// row from a policy that needs widening.
    Install(InstallRefusal),
}

/// What a caller is permitted to ask of the row.
///
/// Passed in by whatever composes this engine — a founded row derives it from
/// the arriving pair's authority, a test states it directly. Keeping it a
/// parameter is what stops this module from inventing a policy of its own.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OpenGrant {
    /// May bind a port below [`ports::FIRST_UNPRIVILEGED_PORT`].
    pub privileged_ports: bool,
    /// May mint **S3** — raw L2 at a station address
    /// ([`InetRow::open_interface`]).
    ///
    /// Default-off, and a separate field rather than a power, because it is
    /// a statement about the *composition* and not about the caller: S3 is
    /// an isolated grant only where the frame row gives each vif its own
    /// station address, and `crates/inet` names no service and so cannot
    /// learn the addressing mode for itself (`res/INET_STACK.md` §2.1).
    /// Whoever composes this engine knows, and says so here.
    ///
    /// **Nothing in this tree sets it.** §9's claimant is a hypervisor guest
    /// wanting raw L2, and `services/hypervisor` has no frame path of its
    /// own yet, so the mechanism exists and has no holder — which is stated
    /// here rather than disguised by a caller invented to exercise it.
    pub raw_l2: bool,
    /// The powers any minted capability may carry.
    pub powers: Powers,
}

/// The singular half of the inet row.
#[derive(Debug)]
pub struct InetRow {
    identity: Option<Identity>,
    ports: PortTable,
    /// What each vif slot currently enforces on this row's behalf.
    ///
    /// Stored rather than derived because not every mint binds a port —
    /// [`open_station`](InetRow::open_station) binds none — so the port table
    /// cannot answer this. A fixed array at the frame row's own ceiling
    /// ([`MAX_VIFS`]): policy state at an architectural bound, which is the
    /// shape that stays heapless.
    installed: [Option<Installed>; MAX_VIFS],
}

/// One vif's entry in the row's enforcement record.
///
/// The station address rides beside the identifier rather than in a second
/// array: the two are written and cleared together by the same mint and the
/// same close, and a parallel array is how they would eventually disagree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Installed {
    /// The tenant whose filter is installed — generation included, so a
    /// slot's next tenant inherits nothing.
    vif: VifId,
    /// The station address, for an **S3** entry only. `None` for every rung
    /// below, which names no L2 address of its own.
    station: Option<MacAddr>,
}

impl Default for InetRow {
    fn default() -> Self {
        Self::new()
    }
}

impl InetRow {
    /// A row with no identity and nothing bound.
    ///
    /// The honest starting state: before DHCP or a static configuration, this
    /// machine has no address, and every flow request is refused with
    /// [`OpenRefusal::NoIdentity`] rather than served against a guess.
    pub const fn new() -> Self {
        Self {
            identity: None,
            ports: PortTable::new(),
            installed: [const { None }; MAX_VIFS],
        }
    }

    /// The machine's identity, once it has one.
    pub fn identity(&self) -> Option<&Identity> {
        self.identity.as_ref()
    }

    /// Adopt an identity — from DHCP, or from a static configuration.
    pub fn set_identity(&mut self, identity: Identity) {
        self.identity = Some(identity);
    }

    /// The port table, for a composition that must report it.
    pub fn ports(&self) -> &PortTable {
        &self.ports
    }

    /// Release every port `vif` holds, leaving the enforcement installation
    /// alone.
    ///
    /// [`close`](Self::close) is the full retirement and is async because the
    /// filter half has to ask the installation whose vif it is. A vif that is
    /// being torn down by its **owner** needs only this half, and needs it
    /// from a synchronous path.
    ///
    /// **The ownership question is still asked — by the caller, before it
    /// calls.** `close` learned that the hard way (its own comment records
    /// why the check moved ahead of the release), so this is deliberately not
    /// a convenience that skips it: `services/net`'s `close_vif` refuses a
    /// vif it does not hold before reaching here.
    pub fn release_ports(&mut self, vif: VifId) -> usize {
        self.ports.release_all(vif)
    }

    /// Open a connected flow: both ends pinned.
    ///
    /// `local_port` of `None` draws an ephemeral port, which is what a client
    /// connecting outward wants. The enforcing filter is installed through
    /// `install` before the capability is returned; see [`Self::mint`].
    pub async fn open_flow(
        &mut self,
        vif: VifId,
        grant: OpenGrant,
        proto: IpProto,
        local_port: Option<u16>,
        peer: (Ipv4Addr, u16),
        install: &mut impl FlowInstall,
    ) -> Result<FlowCap, OpenRefusal> {
        let identity = *self.identity.as_ref().ok_or(OpenRefusal::NoIdentity)?;
        if identity.next_hop(peer.0).is_none() {
            return Err(OpenRefusal::Unreachable);
        }
        let slot = self.free_slot(vif)?;
        let port = match local_port {
            Some(p) => self
                .ports
                .bind(proto, p, vif, grant.privileged_ports)
                .map_err(OpenRefusal::Port)?,
            None => self
                .ports
                .bind_ephemeral(proto, vif)
                .map_err(OpenRefusal::Port)?,
        };
        let scope = Scope::Flow(FlowMatch::connection(
            proto,
            (identity.addr(), port),
            peer,
        ));
        self.mint(vif, slot, scope, grant.powers, Some((proto, port)), install)
            .await
    }

    /// Open a listening flow: a local endpoint, any peer.
    pub async fn open_listener(
        &mut self,
        vif: VifId,
        grant: OpenGrant,
        proto: IpProto,
        local_port: u16,
        install: &mut impl FlowInstall,
    ) -> Result<FlowCap, OpenRefusal> {
        let identity = *self.identity.as_ref().ok_or(OpenRefusal::NoIdentity)?;
        let slot = self.free_slot(vif)?;
        let port = self
            .ports
            .bind(proto, local_port, vif, grant.privileged_ports)
            .map_err(OpenRefusal::Port)?;
        let scope = Scope::Flow(FlowMatch::listener(proto, (identity.addr(), port)));
        self.mint(vif, slot, scope, grant.powers, Some((proto, port)), install)
            .await
    }

    /// The slot `vif` records its installed flow in — and the check that it
    /// is free to take one.
    ///
    /// Returning the index rather than a bare `Ok` is what lets the mint
    /// record the install without a second lookup that could be out of range:
    /// the bound is validated once, here, and carried.
    fn free_slot(&self, vif: VifId) -> Result<usize, OpenRefusal> {
        let index = usize::from(vif.index);
        let slot = self
            .installed
            .get(index)
            .ok_or(OpenRefusal::Install(InstallRefusal::NoVif))?;
        match slot {
            // A *stale* identity in the slot is not a conflict: the vif that
            // held it is gone, and its filter went with it when the frame row
            // bumped the generation. Only a live tenant blocks.
            Some(held) if held.vif == vif => Err(OpenRefusal::VifAlreadyBound),
            _ => Ok(index),
        }
    }

    /// **Mint, then install — and unwind the mint if the install refuses.**
    ///
    /// The whole fail-closed argument of this module lives here. A capability
    /// is handed back only once the predicate that enforces it is running, so
    /// there is no window in which a holder has an authority nothing is
    /// checking. If the install refuses, the port this mint bound is released
    /// and the refusal names what was missing — the caller is left exactly as
    /// it was before it asked.
    ///
    /// S0 is not a branch around this. [`FlowCap::frame_filter`] answers
    /// `None` for [`Scope::Mediated`], and `None` means there is nothing to
    /// install because nothing was granted; the same code path produces it.
    ///
    /// `bound` is the port this mint drew, so the unwind releases exactly
    /// that one and never another flow's — the row never frees a binding it
    /// did not just make.
    async fn mint(
        &mut self,
        vif: VifId,
        slot: usize,
        scope: Scope,
        powers: Powers,
        bound: Option<(IpProto, u16)>,
        install: &mut impl FlowInstall,
    ) -> Result<FlowCap, OpenRefusal> {
        // Read off the scope rather than passed in, so the record cannot
        // disagree with the authority it is recording — the one way a
        // station-collision check could be made to pass for the wrong vif.
        let station = scope.station();
        // The mint cannot fail for a rung this engine builds, but the result
        // is propagated rather than unwrapped: if the ladder ever grows, the
        // refusal should surface here instead of panicking in a row.
        let cap = match FlowCap::mint(vif, scope, powers) {
            Ok(cap) => cap,
            Err(_) => {
                self.unwind(vif, bound);
                return Err(OpenRefusal::Port(BindRefusal::Reserved));
            }
        };
        let Some(filter) = cap.frame_filter() else {
            // S0: nothing was granted, so there is nothing to enforce and
            // nothing to record. The row stays on this holder's datapath.
            return Ok(cap);
        };
        if let Err(refusal) = install.install(vif, filter).await {
            self.unwind(vif, bound);
            // The capability is dropped here rather than returned — it is
            // affine, so no copy of it survives this frame, and the holder
            // never sees an authority nothing is checking.
            return Err(OpenRefusal::Install(refusal));
        }
        self.installed[slot] = Some(Installed { vif, station });
        Ok(cap)
    }

    /// Give back exactly what a failed mint had spent.
    fn unwind(&mut self, vif: VifId, bound: Option<(IpProto, u16)>) {
        if let Some((proto, port)) = bound {
            self.ports.release(proto, port, vif);
        }
    }

    /// Mint the S0 authority: a grant that covers nothing, for a caller the
    /// row will keep serving on the datapath.
    ///
    /// No port is bound, because nothing is being reserved. This is what
    /// makes S0 the same operation as every other rung rather than a branch
    /// around them.
    pub fn open_mediated(&self, vif: VifId) -> FlowCap {
        FlowCap::mediated(vif)
    }

    /// A flow covering every protocol at this station's address — what a
    /// bootstrap client needs before it has an identity to be specific about.
    ///
    /// Deliberately narrow in the one dimension that matters: it names *this*
    /// station's address, so it is not a capture power.
    pub async fn open_station(
        &mut self,
        vif: VifId,
        grant: OpenGrant,
        install: &mut impl FlowInstall,
    ) -> Result<FlowCap, OpenRefusal> {
        let identity = *self.identity.as_ref().ok_or(OpenRefusal::NoIdentity)?;
        let slot = self.free_slot(vif)?;
        let scope = Scope::Flow(FlowMatch {
            proto: ProtoMatch::Any,
            local: EndpointMatch::Addr(identity.addr()),
            peer: EndpointMatch::Any,
        });
        // No port is bound, so there is none to give back on a refused
        // install — the unwind is empty, not absent.
        self.mint(vif, slot, scope, grant.powers, None, install).await
    }

    /// **Open S3** — raw L2 at `station`, on this vif.
    ///
    /// The strongest stratum below a dedicated hardware queue, so what it
    /// does *not* grant is the part worth stating:
    ///
    /// - **It is not the device.** The filter it installs admits frames
    ///   addressed to `station`, plus broadcast. Another vif's unicast is
    ///   out of reach — provided the frame row gave that vif a different
    ///   station address, which is the precondition `grant.raw_l2` asserts.
    /// - **It is not a promiscuous tap.** Without
    ///   [`Powers::promiscuous`](crate::flow::Powers::promiscuous) the
    ///   filter carries no multicast group and no promiscuous bit, so a
    ///   holder hears nothing it was not addressed.
    /// - **It is not a licence to forge.** Without
    ///   [`Powers::spoof`](crate::flow::Powers::spoof) the holder may put
    ///   only `station` in a frame's source address, and the frame row
    ///   refuses anything else on every frame regardless.
    /// - **It reserves no port.** An interface grant is below the port
    ///   space, so it binds nothing in the [`PortTable`] and is minted
    ///   without an [`Identity`] — a guest needs raw L2 *before* it has an
    ///   address, and refusing it for want of one would be refusing it for
    ///   the wrong reason. The corollary is that S3 and the port table do
    ///   not arbitrate against each other: an S3 holder is reached at L2 and
    ///   an S1 holder's port binding says nothing about it.
    ///
    /// `station` must be the address the frame row assigned this vif. The
    /// row cannot check that — it holds no vif table — but the frame row
    /// enforces it on every transmit, so a mismatch surfaces as a holder
    /// that can receive and cannot send.
    pub async fn open_interface(
        &mut self,
        vif: VifId,
        grant: OpenGrant,
        station: MacAddr,
        install: &mut impl FlowInstall,
    ) -> Result<FlowCap, OpenRefusal> {
        if !grant.raw_l2 {
            return Err(OpenRefusal::RawL2NotGranted);
        }
        let slot = self.free_slot(vif)?;
        // The half of the shared-addressing hazard this row can see. Two S3
        // holders at one station address each receive the other's traffic in
        // full — the vif boundary dissolved, with neither having asked for a
        // power. Checked before the mint so nothing is spent on a refusal.
        if self.station_is_raw(vif, station) {
            return Err(OpenRefusal::StationAlreadyRaw);
        }
        // No port is bound, so the unwind on a refused install is empty
        // rather than absent — the same `None` `open_station` passes.
        self.mint(
            vif,
            slot,
            Scope::Interface { station },
            grant.powers,
            None,
            install,
        )
        .await
    }

    /// Whether some **other** live vif already carries a raw-L2 authority of
    /// this row's at `station`.
    ///
    /// The vif being opened is excluded rather than special-cased away: a
    /// re-mint on the same slot is already refused by
    /// [`free_slot`](Self::free_slot), and counting it here would turn that
    /// into the wrong refusal.
    fn station_is_raw(&self, vif: VifId, station: MacAddr) -> bool {
        self.installed
            .iter()
            .flatten()
            .any(|held| held.vif != vif && held.station == Some(station))
    }

    /// Release everything a vif holds. What a close runs.
    ///
    /// Two releases, not one. The ports go back to the table, and the filter
    /// enforcing this vif's flow is replaced by the empty predicate — because
    /// an authority that has been given up must stop being enforced, and a
    /// vif left carrying its old filter would keep receiving traffic for a
    /// flow its holder no longer holds.
    ///
    /// Returns the ports released, and whether the release of the enforcement
    /// was itself refused. A refused release is reported rather than
    /// swallowed: it means the frame row is still running a predicate this
    /// row believes it has revoked, which is the one outcome a caller must
    /// not be able to mistake for a clean close.
    pub async fn close(
        &mut self,
        vif: VifId,
        install: &mut impl FlowInstall,
    ) -> (usize, Result<(), InstallRefusal>) {
        // Asked before either table moves. A caller naming a vif it does not
        // hold must free nothing: the port release below is unconditional and
        // the enforcement release further down is the only other ownership
        // question asked, so without this the ports of a vif belonging to
        // another holder are gone before anything checks whose they were.
        if let Err(refusal) = install.held(vif) {
            return (0, Err(refusal));
        }
        let released = self.ports.release_all(vif);
        let Some(slot) = self.installed.get_mut(usize::from(vif.index)) else {
            return (released, Ok(()));
        };
        if slot.map(|held| held.vif) != Some(vif) {
            // Nothing of this row's is installed on this vif — a close with
            // no enforcement to revoke, which is the S0 holder's close and a
            // stale identifier's alike.
            return (released, Ok(()));
        }
        match install.release(vif).await {
            Ok(()) => {
                *slot = None;
                (released, Ok(()))
            }
            // The record is KEPT on a refused release: this row still
            // believes a filter of its own is installed there, so a later
            // mint on the same vif must not quietly overwrite an authority
            // nobody confirmed was revoked.
            Err(refusal) => (released, Err(refusal)),
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::RecordingInstall;
    use kore_driver::net::policy::PolicyRefusal;
    use kore_driver::net::{EtherType, FrameFilter};

    const ME: Ipv4Addr = Ipv4Addr([10, 0, 2, 15]);
    const GW: Ipv4Addr = Ipv4Addr([10, 0, 2, 2]);
    const OFFLINK: Ipv4Addr = Ipv4Addr([142, 250, 72, 46]);

    /// Drive one of the row's futures to its answer.
    ///
    /// Bounded, not a spin: the install double never parks, so one poll is
    /// always enough, and an exhaustion here would be a real finding — a row
    /// operation that wanted an executor a host test cannot give it.
    fn run<T>(f: impl core::future::Future<Output = T>) -> T {
        kore_rt::try_block_on(f, 4).expect("the row's futures settle without an executor")
    }

    fn vif(index: u16) -> VifId {
        VifId {
            index,
            generation: 1,
        }
    }

    fn configured() -> InetRow {
        let mut row = InetRow::new();
        row.set_identity(Identity::new(ME, 24, Some(GW)).expect("valid"));
        row
    }

    fn grant() -> OpenGrant {
        OpenGrant::default()
    }

    /// Open a connected flow — the shape most tests below want when the
    /// install itself is not what is being pinned.
    fn open(
        row: &mut InetRow,
        install: &mut RecordingInstall,
        v: VifId,
        g: OpenGrant,
        proto: IpProto,
        local_port: Option<u16>,
        peer: (Ipv4Addr, u16),
    ) -> Result<FlowCap, OpenRefusal> {
        run(row.open_flow(v, g, proto, local_port, peer, install))
    }

    fn listen(
        row: &mut InetRow,
        install: &mut RecordingInstall,
        v: VifId,
        g: OpenGrant,
        proto: IpProto,
        local_port: u16,
    ) -> Result<FlowCap, OpenRefusal> {
        run(row.open_listener(v, g, proto, local_port, install))
    }

    /// **TCP/8443 and UDP/8443 are two ports.** The table arbitrates the
    /// transport's port space, and a port's identity is the pair.
    ///
    /// Without the protocol a UDP listener is refused `Taken` because a TCP
    /// one holds "the" number — a refusal with no authority behind it,
    /// which is a denial of service between two clients that never
    /// contended for anything. `FlowMatch` has always carried the protocol
    /// and the classifier has always matched on it; the table that hands
    /// the endpoint out was the one place it was missing.
    #[test]
    fn a_tcp_port_and_a_udp_port_of_the_same_number_do_not_contend() {
        let mut install = RecordingInstall::new();
        let mut row = configured();

        assert!(
            listen(&mut row, &mut install, vif(0), grant(), IpProto::TCP, 8443).is_ok(),
            "TCP/8443 is free"
        );
        assert!(
            listen(&mut row, &mut install, vif(1), grant(), IpProto::UDP, 8443).is_ok(),
            "and UDP/8443 is a different port, so it is free too"
        );

        assert_eq!(row.ports().holder(IpProto::TCP, 8443), Some(vif(0)));
        assert_eq!(row.ports().holder(IpProto::UDP, 8443), Some(vif(1)));

        // The contention that is real still is.
        assert!(
            matches!(
                listen(&mut row, &mut install, vif(2), grant(), IpProto::TCP, 8443),
                Err(OpenRefusal::Port(BindRefusal::Taken))
            ),
            "a second TCP/8443 is the same port twice"
        );
    }

    /// A release frees the port it names and not its namesake.
    #[test]
    fn closing_one_protocols_port_leaves_the_others_bound() {
        let mut install = RecordingInstall::new();
        let mut row = configured();

        listen(&mut row, &mut install, vif(0), grant(), IpProto::TCP, 8443).expect("TCP/8443");
        listen(&mut row, &mut install, vif(1), grant(), IpProto::UDP, 8443).expect("UDP/8443");

        let (freed, _) = run(row.close(vif(0), &mut install));
        assert_eq!(freed, 1, "the close frees exactly the one port that vif held");

        assert_eq!(row.ports().holder(IpProto::TCP, 8443), None, "released");
        assert_eq!(
            row.ports().holder(IpProto::UDP, 8443),
            Some(vif(1)),
            "and the other protocol's port is untouched"
        );
    }

    // ── identity ────────────────────────────────────────────────────────

    /// A machine with no address refuses every flow rather than serving one
    /// against a guess.
    #[test]
    fn without_an_identity_nothing_opens() {
        let mut row = InetRow::new();
        let mut install = RecordingInstall::new();
        assert!(row.identity().is_none());
        assert_eq!(
            open(
                &mut row,
                &mut install,
                vif(0),
                grant(),
                IpProto::TCP,
                None,
                (GW, 80)
            ),
            Err(OpenRefusal::NoIdentity)
        );
        assert_eq!(
            listen(&mut row, &mut install, vif(0), grant(), IpProto::TCP, 8080),
            Err(OpenRefusal::NoIdentity)
        );
        assert_eq!(
            install.attempts(),
            0,
            "a refusal taken before the mint installs nothing"
        );
    }

    /// Addresses that cannot be a station are refused at configuration, not
    /// at first use.
    #[test]
    fn an_unusable_address_is_refused() {
        for bad in [
            Ipv4Addr::UNSPECIFIED,
            Ipv4Addr::BROADCAST,
            Ipv4Addr([224, 0, 0, 1]),
            Ipv4Addr([127, 0, 0, 1]),
        ] {
            assert_eq!(
                Identity::new(bad, 24, None).unwrap_err(),
                IdentityRefusal::NotAStation,
                "{bad:?}"
            );
        }
        assert_eq!(
            Identity::new(ME, 33, None).unwrap_err(),
            IdentityRefusal::BadPrefix
        );
    }

    /// A gateway off its own prefix could never route, so it is refused where
    /// it is configured rather than discovered as a timeout.
    #[test]
    fn an_off_link_gateway_is_refused() {
        assert_eq!(
            Identity::new(ME, 24, Some(OFFLINK)).unwrap_err(),
            IdentityRefusal::GatewayOffLink
        );
        assert!(Identity::new(ME, 24, Some(GW)).is_ok());
    }

    /// On-link goes direct; off-link goes via the gateway; with no gateway,
    /// off-link is unreachable and says so.
    #[test]
    fn next_hop_follows_the_prefix() {
        let id = Identity::new(ME, 24, Some(GW)).expect("valid");
        assert_eq!(id.next_hop(GW), Some(GW), "on-link is its own next hop");
        assert_eq!(id.next_hop(OFFLINK), Some(GW), "off-link goes via the gateway");

        let stranded = Identity::new(ME, 24, None).expect("valid");
        assert_eq!(stranded.next_hop(Ipv4Addr([10, 0, 2, 99])), Some(Ipv4Addr([10, 0, 2, 99])));
        assert_eq!(
            stranded.next_hop(OFFLINK),
            None,
            "no gateway is unreachable, not a guess"
        );
    }

    /// An unreachable peer is refused at open, so a caller learns it here and
    /// not from a connection that never completes.
    #[test]
    fn an_unreachable_peer_is_refused_at_open() {
        let mut row = InetRow::new();
        let mut install = RecordingInstall::new();
        row.set_identity(Identity::new(ME, 24, None).expect("valid"));
        assert_eq!(
            open(
                &mut row,
                &mut install,
                vif(0),
                grant(),
                IpProto::TCP,
                None,
                (OFFLINK, 443)
            ),
            Err(OpenRefusal::Unreachable)
        );
    }

    // ── the port space ──────────────────────────────────────────────────

    /// The property the row exists for: two vifs cannot both hold one port.
    #[test]
    fn two_vifs_cannot_hold_one_port() {
        let mut row = configured();
        let mut install = RecordingInstall::new();
        assert!(listen(&mut row, &mut install, vif(0), grant(), IpProto::TCP, 8080).is_ok());
        assert_eq!(
            listen(&mut row, &mut install, vif(1), grant(), IpProto::TCP, 8080),
            Err(OpenRefusal::Port(BindRefusal::Taken)),
            "the second asker is refused, not silently co-bound"
        );
        assert_eq!(row.ports().holder(IpProto::TCP, 8080), Some(vif(0)));
    }

    /// A privileged port is a grant, not a status — and refused by name
    /// without it.
    #[test]
    fn a_privileged_port_needs_the_grant() {
        let mut row = configured();
        let mut install = RecordingInstall::new();
        assert_eq!(
            listen(&mut row, &mut install, vif(0), grant(), IpProto::TCP, 80),
            Err(OpenRefusal::Port(BindRefusal::Privileged))
        );
        let privileged = OpenGrant {
            privileged_ports: true,
            ..grant()
        };
        assert!(listen(&mut row, &mut install, vif(0), privileged, IpProto::TCP, 80).is_ok());
    }

    /// Port zero is the wire's wildcard and never a binding.
    #[test]
    fn port_zero_is_never_bound() {
        let mut row = configured();
        let mut install = RecordingInstall::new();
        let privileged = OpenGrant {
            privileged_ports: true,
            ..grant()
        };
        assert_eq!(
            listen(&mut row, &mut install, vif(0), privileged, IpProto::TCP, 0),
            Err(OpenRefusal::Port(BindRefusal::Reserved))
        );
    }

    /// An outward connection draws an ephemeral port, and two draw different
    /// ones.
    #[test]
    fn ephemeral_ports_do_not_collide() {
        let mut row = configured();
        let mut install = RecordingInstall::new();
        let a = open(
            &mut row,
            &mut install,
            vif(0),
            grant(),
            IpProto::TCP,
            None,
            (OFFLINK, 443),
        )
        .expect("opens");
        let b = open(
            &mut row,
            &mut install,
            vif(1),
            grant(),
            IpProto::TCP,
            None,
            (OFFLINK, 443),
        )
        .expect("opens");

        let (pa, pb) = (local_port(&a), local_port(&b));
        assert_ne!(pa, pb, "two outward flows got different local ports");
        for p in [pa, pb] {
            assert!((ports::EPHEMERAL_FIRST..=ports::EPHEMERAL_LAST).contains(&p));
        }
        assert_eq!(row.ports().len(), 2);
    }

    /// A close frees exactly that vif's bindings and nobody else's.
    #[test]
    fn closing_frees_only_that_vifs_ports() {
        let mut row = configured();
        let mut install = RecordingInstall::new();
        listen(&mut row, &mut install, vif(0), grant(), IpProto::TCP, 8080).expect("opens");
        listen(&mut row, &mut install, vif(1), grant(), IpProto::TCP, 9090).expect("opens");

        assert_eq!(run(row.close(vif(0), &mut install)), (1, Ok(())));
        assert_eq!(row.ports().holder(IpProto::TCP, 8080), None);
        assert_eq!(
            row.ports().holder(IpProto::TCP, 9090),
            Some(vif(1)),
            "another vif's binding survives"
        );
    }

    /// One client cannot free another's port — that would be a denial of
    /// service with no capability involved.
    #[test]
    fn a_non_holder_cannot_release_a_port() {
        let mut row = configured();
        let mut install = RecordingInstall::new();
        listen(&mut row, &mut install, vif(0), grant(), IpProto::TCP, 8080).expect("opens");
        assert_eq!(run(row.close(vif(1), &mut install)), (0, Ok(())));
        assert_eq!(row.ports().holder(IpProto::TCP, 8080), Some(vif(0)));
    }

    /// A reused slot does not inherit the previous tenant's bindings, because
    /// the generation is part of the identifier the table keys on.
    #[test]
    fn a_reused_slot_inherits_nothing() {
        let mut row = configured();
        let mut install = RecordingInstall::new();
        let first = VifId {
            index: 2,
            generation: 1,
        };
        let next_tenant = VifId {
            index: 2,
            generation: 2,
        };
        listen(&mut row, &mut install, first, grant(), IpProto::TCP, 8080).expect("opens");

        assert_eq!(
            listen(&mut row, &mut install, next_tenant, grant(), IpProto::TCP, 8080),
            Err(OpenRefusal::Port(BindRefusal::Taken)),
            "the slot's next tenant does not inherit the binding"
        );
        assert_eq!(
            run(row.close(next_tenant, &mut install)),
            (0, Ok(())),
            "nor may it free it"
        );
        assert_eq!(run(row.close(first, &mut install)), (1, Ok(())));
    }

    // ── what the mint produces ──────────────────────────────────────────

    /// The minted authority covers exactly the flow that was asked for.
    #[test]
    fn the_minted_cap_covers_the_opened_flow() {
        use kore_driver::net::flow::FiveTuple;

        let mut row = configured();
        let mut install = RecordingInstall::new();
        let cap = open(
            &mut row,
            &mut install,
            vif(0),
            grant(),
            IpProto::TCP,
            Some(5000),
            (OFFLINK, 443),
        )
        .expect("opens");

        let mine = FiveTuple {
            proto: IpProto::TCP,
            src: OFFLINK,
            dst: ME,
            ports: Some((443, 5000)),
        };
        assert!(cap.admits(&mine));

        let stranger = FiveTuple {
            src: Ipv4Addr([1, 1, 1, 1]),
            ..mine
        };
        assert!(!cap.admits(&stranger), "another peer is not this flow");
    }

    /// S0 binds no port, because nothing is being reserved — the same
    /// operation as every other rung, not a branch around them.
    #[test]
    fn the_mediated_grant_reserves_nothing() {
        let row = configured();
        let cap = row.open_mediated(vif(0));
        assert!(cap.scope().is_mediated());
        assert!(row.ports().is_empty());
    }

    // ── S3, and the two walls in front of it ────────────────────────────

    const STATION: MacAddr = MacAddr([0x02, 0, 0, 0, 0, 0x01]);
    const NEIGHBOUR: MacAddr = MacAddr([0x02, 0, 0, 0, 0, 0x02]);

    /// The grant a composer that has established per-vif addressing hands in.
    fn raw_grant() -> OpenGrant {
        OpenGrant {
            raw_l2: true,
            ..OpenGrant::default()
        }
    }

    /// **The first wall.** S3 is mintable in the ladder and still refused
    /// here, because being in the model is not being reachable: the grant
    /// that admits it is default-off and nothing in this tree sets it.
    #[test]
    fn raw_l2_needs_its_grant() {
        let mut row = configured();
        let mut install = RecordingInstall::new();
        assert_eq!(
            run(row.open_interface(vif(0), grant(), STATION, &mut install)),
            Err(OpenRefusal::RawL2NotGranted)
        );
        assert_eq!(
            install.attempts(),
            0,
            "refused before anything was installed"
        );
        assert!(
            run(row.open_interface(vif(0), raw_grant(), STATION, &mut install)).is_ok(),
            "and admitted once the composition says so"
        );
    }

    /// **The second wall.** Two raw-L2 holders at one station address are
    /// not two holders — each one's filter admits every frame addressed to
    /// the station, so each receives the other's traffic entire, neither
    /// having asked for a power. The row can see that collision, so it
    /// refuses it.
    #[test]
    fn two_raw_holders_may_not_share_one_station() {
        let mut row = configured();
        let mut install = RecordingInstall::new();

        assert!(run(row.open_interface(vif(0), raw_grant(), STATION, &mut install)).is_ok());
        assert_eq!(
            run(row.open_interface(vif(1), raw_grant(), STATION, &mut install)),
            Err(OpenRefusal::StationAlreadyRaw),
            "a second raw holder at one address is a mutual escape"
        );
        assert!(
            install.filter(vif(1)).is_none(),
            "and nothing was installed for it"
        );
        assert!(
            run(row.open_interface(vif(1), raw_grant(), NEIGHBOUR, &mut install)).is_ok(),
            "two stations are two grants"
        );

        // The collision is held by the *live* record, so closing the first
        // holder frees its station for the next one.
        let mut row = configured();
        assert!(run(row.open_interface(vif(0), raw_grant(), STATION, &mut install)).is_ok());
        assert_eq!(run(row.close(vif(0), &mut install)), (0, Ok(())));
        assert!(
            run(row.open_interface(vif(2), raw_grant(), STATION, &mut install)).is_ok(),
            "a released station is available again"
        );
    }

    /// S3 needs no [`Identity`] and reserves no port: a guest wants raw L2
    /// *before* it has an address, and an interface grant is below the port
    /// space rather than inside it.
    #[test]
    fn raw_l2_needs_neither_an_address_nor_a_port() {
        let mut row = InetRow::new();
        assert!(row.identity().is_none());
        let mut install = RecordingInstall::new();

        let cap = run(row.open_interface(vif(0), raw_grant(), STATION, &mut install))
            .expect("an interface grant names no IPv4 address to need one");
        assert_eq!(cap.scope().station(), Some(STATION));
        assert!(row.ports().is_empty(), "nothing was reserved");
    }

    /// The mint installs S3's own filter, and a refused install unwinds it
    /// exactly as every other rung's does — including freeing the station,
    /// so a failed raw mint does not lock the address out.
    #[test]
    fn a_refused_raw_install_leaves_the_station_free() {
        let mut row = configured();
        let mut install =
            RecordingInstall::refusing(InstallRefusal::Policy(PolicyRefusal::EtherType));

        // The frame row's own refusal for this filter: `EtherTypeMatch::Any`
        // is a capture power there, so an S3 does not install under an
        // ordinary client's policy. That is the third wall, and it is not
        // this row's.
        assert_eq!(
            run(row.open_interface(vif(0), raw_grant(), STATION, &mut install)),
            Err(OpenRefusal::Install(InstallRefusal::Policy(
                PolicyRefusal::EtherType
            )))
        );

        install.set_refusal(None);
        assert!(
            run(row.open_interface(vif(1), raw_grant(), STATION, &mut install)).is_ok(),
            "the station was never recorded, so it was never held"
        );
    }

    // ── the mint installs its own enforcement ───────────────────────────

    /// **The property this slice exists for.** A successful mint leaves the
    /// capability's own filter installed on the holder's vif, so the
    /// classifier one layer down is from that moment running the predicate
    /// the authority describes.
    ///
    /// Without it a `FlowCap` is a description: the holder believes it is
    /// isolated while the vif keeps whatever filter it was opened with, and
    /// nothing stops another client's datagrams arriving beside its own.
    #[test]
    fn a_successful_mint_installs_the_capabilitys_own_filter() {
        let mut row = configured();
        let mut install = RecordingInstall::new();
        let cap = open(
            &mut row,
            &mut install,
            vif(3),
            grant(),
            IpProto::UDP,
            Some(5000),
            (OFFLINK, 443),
        )
        .expect("opens");

        assert_eq!(
            install.filter(vif(3)),
            cap.frame_filter(),
            "what was installed is the capability's own filter, not an approximation"
        );
        let installed = install.filter(vif(3)).expect("a filter was installed");
        assert_eq!(
            installed.ethertype,
            FrameFilter::protocol(EtherType::IPV4).ethertype
        );
        assert!(
            installed.flow.is_some(),
            "and it carries the flow dimension, which is what separates two IP clients"
        );
        assert!(
            install.filter(vif(4)).is_none(),
            "and only on the vif the authority names"
        );
    }

    /// A listener's mint installs too — the enforcement is a property of the
    /// mint, not of which verb reached it.
    #[test]
    fn a_listener_installs_its_filter_too() {
        let mut row = configured();
        let mut install = RecordingInstall::new();
        let cap =
            listen(&mut row, &mut install, vif(1), grant(), IpProto::UDP, 8080).expect("opens");
        assert_eq!(install.filter(vif(1)), cap.frame_filter());
    }

    /// **Fail closed.** If the enforcement cannot be installed, no capability
    /// is handed back and the port the mint drew is given up — the caller is
    /// left exactly as it was before it asked.
    ///
    /// The alternative is the one outcome worse than a refusal: a holder with
    /// an authority nothing is checking, plus a port nobody can rebind.
    #[test]
    fn an_install_that_refuses_unwinds_the_whole_mint() {
        let mut row = configured();
        let mut install = RecordingInstall::refusing(InstallRefusal::NoFrameFace);

        assert_eq!(
            listen(&mut row, &mut install, vif(0), grant(), IpProto::TCP, 8080),
            Err(OpenRefusal::Install(InstallRefusal::NoFrameFace)),
            "the refusal names what was missing, not a bare failure"
        );
        assert_eq!(
            row.ports().holder(IpProto::TCP, 8080),
            None,
            "the port the mint drew went back"
        );
        assert!(row.ports().is_empty());

        // And the binding is genuinely free again: the same port opens once
        // the face is there.
        install.set_refusal(None);
        assert!(listen(&mut row, &mut install, vif(0), grant(), IpProto::TCP, 8080).is_ok());
        assert_eq!(row.ports().holder(IpProto::TCP, 8080), Some(vif(0)));
    }

    /// An ephemeral port spent on a refused install goes back too, so a row
    /// whose face is down does not leak its ephemeral range away one failed
    /// connection at a time.
    #[test]
    fn a_refused_install_does_not_leak_an_ephemeral_port() {
        let mut row = configured();
        let mut install = RecordingInstall::refusing(InstallRefusal::Lost);
        for _ in 0..8 {
            assert_eq!(
                open(
                    &mut row,
                    &mut install,
                    vif(0),
                    grant(),
                    IpProto::TCP,
                    None,
                    (OFFLINK, 443)
                ),
                Err(OpenRefusal::Install(InstallRefusal::Lost))
            );
        }
        assert!(
            row.ports().is_empty(),
            "eight refused connections spent no port between them"
        );
    }

    /// The refusal carries the frame row's own per-dimension answer, so an
    /// operator can tell an unwired row from a policy that needs widening.
    ///
    /// A single "install failed" would leave exactly the two causes with
    /// different fixes indistinguishable.
    #[test]
    fn an_install_refusal_names_the_dimension() {
        let mut row = configured();
        for refusal in [
            InstallRefusal::NoFrameFace,
            InstallRefusal::Policy(PolicyRefusal::FlowPeer),
            InstallRefusal::NotOwner,
            InstallRefusal::NoVif,
            InstallRefusal::Lost,
        ] {
            let mut install = RecordingInstall::refusing(refusal);
            assert_eq!(
                listen(&mut row, &mut install, vif(0), grant(), IpProto::TCP, 9000),
                Err(OpenRefusal::Install(refusal)),
                "{refusal:?} reaches the caller intact"
            );
        }
    }

    /// **S0 installs nothing, and that is the design working.** A grant that
    /// covers nothing has no predicate to enforce, so the row stays on that
    /// holder's datapath — reached by the same code path as every other rung,
    /// not by a branch around it.
    #[test]
    fn the_mediated_grant_installs_nothing() {
        let row = configured();
        let install = RecordingInstall::new();
        let cap = row.open_mediated(vif(0));
        assert!(cap.frame_filter().is_none(), "S0 has no filter to install");
        assert_eq!(
            install.attempts(),
            0,
            "and the row does not consult the face to learn that"
        );
        assert!(install.is_closed(vif(0)));
    }

    /// A vif carries one filter, so a second flow on it would silently stop
    /// enforcing the first. Refused by name instead.
    ///
    /// The failure this prevents is the one the policy module calls out: a
    /// client that believes it is listening for something it will never
    /// receive, discovered much later as "the protocol doesn't work".
    #[test]
    fn a_second_flow_on_one_vif_is_refused_rather_than_overwriting_the_first() {
        let mut row = configured();
        let mut install = RecordingInstall::new();
        let first =
            listen(&mut row, &mut install, vif(0), grant(), IpProto::TCP, 8080).expect("opens");

        assert_eq!(
            listen(&mut row, &mut install, vif(0), grant(), IpProto::TCP, 8081),
            Err(OpenRefusal::VifAlreadyBound)
        );
        assert_eq!(
            install.filter(vif(0)),
            first.frame_filter(),
            "the first flow's enforcement is untouched"
        );
        assert_eq!(
            row.ports().holder(IpProto::TCP, 8081),
            None,
            "and the refused mint drew no port"
        );
    }

    /// A vif slot whose tenant was replaced may take a fresh flow: the
    /// generation is the revocation, so a stale record is not a conflict.
    #[test]
    fn a_new_tenant_of_a_vif_slot_may_take_a_flow() {
        let mut row = configured();
        let mut install = RecordingInstall::new();
        let first = VifId {
            index: 5,
            generation: 1,
        };
        let next = VifId {
            index: 5,
            generation: 2,
        };
        listen(&mut row, &mut install, first, grant(), IpProto::TCP, 8080).expect("opens");
        assert!(
            listen(&mut row, &mut install, next, grant(), IpProto::TCP, 8081).is_ok(),
            "the slot's next tenant is not blocked by the previous one's record"
        );
        assert!(
            install.filter(first).is_none(),
            "and the old authority does not resolve against the new identity"
        );
    }

    // ── close releases the enforcement ──────────────────────────────────

    /// **Close releases the enforcement, not only the ports.** A holder that
    /// gives up its flow must stop receiving that flow's traffic; a vif left
    /// carrying its old filter would keep delivering for an authority nobody
    /// holds.
    #[test]
    fn a_close_releases_the_installed_filter() {
        let mut row = configured();
        let mut install = RecordingInstall::new();
        listen(&mut row, &mut install, vif(2), grant(), IpProto::UDP, 8080).expect("opens");
        assert!(!install.is_closed(vif(2)));

        assert_eq!(run(row.close(vif(2), &mut install)), (1, Ok(())));
        assert_eq!(
            install.filter(vif(2)),
            Some(FrameFilter::CLOSED),
            "the vif enforces the empty predicate, not its pre-flow filter"
        );
        assert!(install.is_closed(vif(2)));
    }

    /// A close that touched no flow of this row's consults the face not at
    /// all — including an S0 holder's close, and a stale identifier's.
    #[test]
    fn a_close_with_nothing_installed_releases_nothing() {
        let mut row = configured();
        let mut install = RecordingInstall::new();
        assert_eq!(run(row.close(vif(6), &mut install)), (0, Ok(())));
        assert_eq!(
            install.attempts(),
            0,
            "no authority of this row's was there to revoke"
        );
    }

    /// A refused release is reported, and the row keeps believing its filter
    /// is installed.
    ///
    /// The alternative — forgetting the record on a refused release — would
    /// let the next mint on that vif overwrite an authority nobody confirmed
    /// was revoked, which is precisely the silent overwrite
    /// `VifAlreadyBound` exists to prevent.
    #[test]
    fn a_refused_release_is_reported_and_the_record_is_kept() {
        let mut row = configured();
        let mut install = RecordingInstall::new();
        listen(&mut row, &mut install, vif(2), grant(), IpProto::UDP, 8080).expect("opens");

        install.set_refusal(Some(InstallRefusal::Lost));
        assert_eq!(
            run(row.close(vif(2), &mut install)),
            (1, Err(InstallRefusal::Lost)),
            "the caller is told the revocation did not land"
        );

        install.set_refusal(None);
        assert_eq!(
            listen(&mut row, &mut install, vif(2), grant(), IpProto::UDP, 9999),
            Err(OpenRefusal::VifAlreadyBound),
            "the unconfirmed authority still blocks the slot"
        );
    }

    fn local_port(cap: &FlowCap) -> u16 {
        match cap.scope() {
            Scope::Flow(f) => match f.local {
                EndpointMatch::AddrPort { port, .. } => port,
                _ => panic!("a connected flow pins its local port"),
            },
            other => panic!("expected a flow scope, got {other:?}"),
        }
    }
}
