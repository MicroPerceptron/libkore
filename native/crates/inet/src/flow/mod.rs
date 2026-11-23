//! **The flow mint** — the strata, and the affine authority that carries one.
//!
//! The capability half of `res/INET_STACK.md` §4.4. A client does not ask to
//! bypass the row; it holds authority the row already decided to grant, and
//! the row is consulted once per connection instead of once per packet.
//!
//! | module | what it fixes |
//! |---|---|
//! | [`scope`] | the four rungs, the two orthogonal powers, and containment |
//! | [`cap`] | [`FlowCap`] — affine, derivable, revocable by generation |
//! | [`install`] | the seam that makes a minted authority *enforced* |
//! | [`handle`] | [`Flow`] — Tier 1, the authority's working face |

pub mod cap;
pub mod handle;
pub mod install;
pub mod scope;

pub use cap::{FlowCap, FlowRefusal};
pub use handle::{FaceRefusal, Flow, FlowFrame, FlowTx};
pub use install::{FlowInstall, InstallRefusal, NoFrameFace, RecordingInstall};
pub use scope::{PortRange, Powers, Scope};

#[cfg(test)]
mod tests {
    use super::*;
    use kore_driver::net::flow::{
        EndpointMatch, FiveTuple, FlowMatch, IpProto, Ipv4Addr, ProtoMatch,
    };
    use kore_driver::net::{EtherTypeMatch, MacAddr, MulticastMatch, VifId};

    const LOCAL: Ipv4Addr = Ipv4Addr([10, 0, 2, 15]);
    const PEER: Ipv4Addr = Ipv4Addr([10, 0, 2, 2]);
    const OTHER: Ipv4Addr = Ipv4Addr([10, 0, 2, 99]);

    const STATION: MacAddr = MacAddr([0x02, 0, 0, 0, 0, 0x01]);
    const NEIGHBOUR: MacAddr = MacAddr([0x02, 0, 0, 0, 0, 0x02]);

    /// The S3 grant used throughout: raw L2 at one station.
    fn raw() -> Scope {
        Scope::Interface { station: STATION }
    }

    fn vif() -> VifId {
        VifId {
            index: 3,
            generation: 7,
        }
    }

    fn conn() -> FlowMatch {
        FlowMatch::connection(IpProto::TCP, (LOCAL, 443), (PEER, 44000))
    }

    fn tuple(src: Ipv4Addr, dst: Ipv4Addr, sport: u16, dport: u16) -> FiveTuple {
        FiveTuple {
            proto: IpProto::TCP,
            src,
            dst,
            ports: Some((sport, dport)),
        }
    }

    // ── S0 ──────────────────────────────────────────────────────────────

    /// S0 is a grant that covers nothing — which is what keeps the row on the
    /// datapath without a second code path existing.
    #[test]
    fn the_mediated_grant_admits_nothing() {
        let cap = FlowCap::mediated(vif());
        assert!(cap.scope().is_mediated());
        assert_eq!(cap.scope().rung(), 0);
        assert_eq!(cap.powers(), Powers::NONE);
        assert!(
            !cap.admits(&tuple(PEER, LOCAL, 44000, 443)),
            "no traffic is inside a grant of zero scope"
        );
    }

    /// Nothing can be derived *out* of nothing: S0 is the floor, so a holder
    /// cannot bootstrap itself into a flow.
    #[test]
    fn nothing_can_be_derived_from_the_mediated_grant() {
        let cap = FlowCap::mediated(vif());
        assert_eq!(
            cap.derive(Scope::Flow(conn()), Powers::NONE).unwrap_err(),
            FlowRefusal::WiderScope
        );
        assert_eq!(
            cap.derive(raw(), Powers::NONE).unwrap_err(),
            FlowRefusal::WiderScope
        );
        assert!(
            cap.derive(Scope::Mediated, Powers::NONE).is_ok(),
            "deriving nothing from nothing is lawful and useless"
        );
    }

    // ── attenuation ─────────────────────────────────────────────────────

    /// The headline property: a derivation may narrow and may never widen,
    /// in either dimension, and the refusal says which one exceeded.
    #[test]
    fn attenuation_never_widens() {
        let parent = FlowCap::mint(
            vif(),
            Scope::Flow(conn()),
            Powers {
                spoof: true,
                promiscuous: false,
            },
        )
        .expect("S1 is mintable");

        // Narrowing the scope to nothing is lawful.
        assert!(parent.derive(Scope::Mediated, Powers::NONE).is_ok());

        // Widening the scope is refused by name.
        assert_eq!(
            parent.derive(raw(), Powers::NONE).unwrap_err(),
            FlowRefusal::WiderScope
        );

        // A different flow is not narrower — it is elsewhere.
        let elsewhere = FlowMatch::connection(IpProto::TCP, (LOCAL, 443), (OTHER, 44000));
        assert_eq!(
            parent
                .derive(Scope::Flow(elsewhere), Powers::NONE)
                .unwrap_err(),
            FlowRefusal::WiderScope
        );

        // Dropping a power is lawful; acquiring one is not.
        assert!(
            parent
                .derive(Scope::Flow(conn()), Powers::NONE)
                .is_ok()
        );
        assert_eq!(
            parent
                .derive(
                    Scope::Flow(conn()),
                    Powers {
                        spoof: true,
                        promiscuous: true,
                    }
                )
                .unwrap_err(),
            FlowRefusal::WiderPowers,
            "promiscuous was never the parent's to give"
        );
    }

    /// A widened flow *dimension* is caught too — the containment is the
    /// vocabulary's, not a shallow equality.
    #[test]
    fn a_widened_flow_dimension_is_refused() {
        let parent =
            FlowCap::mint(vif(), Scope::Flow(conn()), Powers::NONE).expect("mints");

        let any_peer = FlowMatch {
            peer: EndpointMatch::Any,
            ..conn()
        };
        assert_eq!(
            parent.derive(Scope::Flow(any_peer), Powers::NONE).unwrap_err(),
            FlowRefusal::WiderScope
        );

        let any_proto = FlowMatch {
            proto: ProtoMatch::Any,
            ..conn()
        };
        assert_eq!(
            parent
                .derive(Scope::Flow(any_proto), Powers::NONE)
                .unwrap_err(),
            FlowRefusal::WiderScope
        );
    }

    // ── the rungs ───────────────────────────────────────────────────────

    /// S2 is in the ladder so containment is total, and refused at the mint
    /// so nothing depends on an untested rung. S3 is not: it is built, and
    /// the wall in front of it is the row's grant rather than the ladder's.
    #[test]
    fn the_modelled_rung_is_not_mintable_and_s3_is() {
        assert_eq!(
            FlowCap::mint(
                vif(),
                Scope::Range {
                    addr: LOCAL,
                    ports: PortRange::new(8000, 8999).expect("ordered"),
                },
                Powers::NONE,
            )
            .unwrap_err(),
            FlowRefusal::RungNotMintable
        );
        assert!(FlowCap::mint(vif(), Scope::Mediated, Powers::NONE).is_ok());
        assert!(FlowCap::mint(vif(), Scope::Flow(conn()), Powers::NONE).is_ok());
        assert!(FlowCap::mint(vif(), raw(), Powers::NONE).is_ok());
    }

    /// The ladder contains strictly, in the one direction.
    #[test]
    fn the_ladder_contains_strictly() {
        let range = Scope::Range {
            addr: LOCAL,
            ports: PortRange::new(400, 500).expect("ordered"),
        };
        let inside = Scope::Flow(FlowMatch::connection(
            IpProto::TCP,
            (LOCAL, 443),
            (PEER, 44000),
        ));
        let outside = Scope::Flow(FlowMatch::connection(
            IpProto::TCP,
            (LOCAL, 9000),
            (PEER, 44000),
        ));

        assert!(inside.within(&range), "443 is in 400..=500");
        assert!(!outside.within(&range), "9000 is not");
        assert!(!range.within(&inside), "a range is wider than one flow");

        assert!(range.within(&raw()));
        assert!(inside.within(&raw()));
        assert!(!raw().within(&range));

        assert!(Scope::Mediated.within(&inside));
        assert!(!inside.within(&Scope::Mediated));

        assert_eq!(
            (
                Scope::Mediated.rung(),
                inside.rung(),
                range.rung(),
                raw().rung()
            ),
            (0, 1, 2, 3)
        );
    }

    /// An inverted port range is refused rather than normalised — swapping
    /// the bounds would grant a range nobody checked.
    #[test]
    fn an_inverted_port_range_is_refused() {
        assert!(PortRange::new(500, 400).is_none());
        let r = PortRange::new(400, 500).expect("ordered");
        assert!(r.contains(400) && r.contains(500) && !r.contains(399));
        assert!(PortRange::one(443).within(&r));
        assert!(!r.within(&PortRange::one(443)));
    }

    // ── S3 ──────────────────────────────────────────────────────────────

    /// **What S3 installs.** Raw L2 means every protocol, and it means
    /// nothing else: the addressing half is exactly a plain client's, so the
    /// grant is wide at L3 and unchanged at L2.
    ///
    /// Every field is asserted rather than the filter compared whole,
    /// because a field that silently acquired a wider default is precisely
    /// the failure this rung cannot afford, and an equality against a
    /// hand-built expected value would be rewritten alongside it.
    #[test]
    fn the_interface_grant_installs_raw_l2_and_nothing_wider() {
        let cap = FlowCap::mint(vif(), raw(), Powers::NONE).expect("S3 mints");
        let f = cap.frame_filter().expect("S3 has a filter to install");

        assert_eq!(
            f.ethertype,
            EtherTypeMatch::Any,
            "raw L2 is every protocol — the one dimension S3 widens"
        );
        assert_eq!(f.flow, None, "and no L3 predicate, which is what raw means");
        assert!(f.unicast, "frames addressed to this station");
        assert!(f.broadcast, "broadcast reaches every station including this one");
        assert_eq!(
            f.multicast,
            MulticastMatch::None,
            "a group it never joined is traffic it was not addressed"
        );
        assert!(!f.promiscuous, "and no power it was not granted");
    }

    /// **What S2 installs: nothing.** A rung with no expressible enforcement
    /// is kept shut rather than approximated.
    ///
    /// `EndpointMatch` is `AddrPort | Addr | Any`, so a span of ports cannot
    /// be stated, and the nearest filter that can — the whole address, any
    /// protocol, any peer — admits traffic the grant does not. An earlier
    /// spelling emitted exactly that under a comment claiming the
    /// classifier would narrow it further; `FrameFilter::flow` *is* that
    /// narrowing, so nothing did. The port outside the span below is the
    /// datagram it wrongly admitted.
    ///
    /// Decided by [`FrameFilter::accepts_flow`], the shipping predicate.
    #[test]
    fn the_range_rung_installs_no_enforcement_rather_than_a_wider_one() {
        use kore_driver::net::flow::{FiveTuple, IpProto};
        use kore_driver::net::{EtherType, FrameHead};

        let s2 = Scope::Range {
            addr: LOCAL,
            ports: PortRange::new(400, 500).expect("ordered"),
        };
        assert!(!s2.is_mintable(), "no holder can reach this arm today");

        let cap = FlowCap::at_unmintable_rung(vif(), s2, Powers::NONE);
        let f = cap.frame_filter().expect("a scope below S3 still answers");

        assert!(
            f.admits_nothing(),
            "closed, because every expressible filter here is wider than the grant"
        );

        // The grant's own boundary, for the tuple the old filter admitted.
        let outside = FiveTuple {
            proto: IpProto::TCP,
            src: PEER,
            dst: LOCAL,
            ports: Some((9999, 8080)),
        };
        assert!(
            !cap.admits(&outside),
            "8080 is outside 400..=500, so the authority refuses it"
        );

        // And the installed filter refuses the frame carrying it, decided by
        // `FrameFilter::accepts` — the shipping predicate, not a double.
        let head = FrameHead {
            dst: STATION,
            src: NEIGHBOUR,
            ethertype: EtherType::IPV4,
        };
        assert!(
            !f.accepts(&head, STATION),
            "a closed filter admits no frame at all, so it cannot admit this one"
        );
    }

    /// **The boundary.** An S3 filter is evaluated against the vif's *own*
    /// station address, so another station's unicast is outside it — the
    /// whole of the difference between raw L2 on a vif and raw L2 on the
    /// device.
    ///
    /// Decided by [`FrameFilter::accepts`], the shipping predicate, so a
    /// pass here is a pass of the contract rather than of an agreeing
    /// double.
    #[test]
    fn an_interface_grant_does_not_reach_another_station() {
        use kore_driver::net::{EtherType, FrameHead};

        let cap = FlowCap::mint(vif(), raw(), Powers::NONE).expect("S3 mints");
        let f = cap.frame_filter().expect("installs");

        let head = |dst: MacAddr, t: EtherType| FrameHead {
            dst,
            src: NEIGHBOUR,
            ethertype: t,
        };

        // Every protocol, at its own station: the grant.
        for t in [EtherType::IPV4, EtherType::ARP, EtherType::KORE_PEER] {
            assert!(f.accepts(&head(STATION, t), STATION), "{t:?} at own station");
        }
        // The same protocols, at a neighbour's: not the grant.
        for t in [EtherType::IPV4, EtherType::ARP, EtherType::KORE_PEER] {
            assert!(
                !f.accepts(&head(NEIGHBOUR, t), STATION),
                "{t:?} addressed to another station"
            );
        }
        assert!(
            f.accepts(&head(MacAddr::BROADCAST, EtherType::ARP), STATION),
            "broadcast is addressed to this station too"
        );
    }

    /// **Without `promiscuous`, an S3 holder cannot overhear.** The power is
    /// orthogonal to the ladder, so the strongest rung below hardware still
    /// does not imply it — and holding it is what moves both the promiscuous
    /// bit and the multicast match, together, because they are the same
    /// authority spelled at two addresses.
    #[test]
    fn an_interface_grant_overhears_only_with_the_power() {
        use kore_driver::net::{EtherType, FrameHead};

        let deaf = FlowCap::mint(vif(), raw(), Powers::NONE).expect("mints");
        let tap = FlowCap::mint(
            vif(),
            raw(),
            Powers {
                spoof: false,
                promiscuous: true,
            },
        )
        .expect("mints");

        let deaf_f = deaf.frame_filter().expect("installs");
        let tap_f = tap.frame_filter().expect("installs");

        let group = MacAddr([0x01, 0x80, 0xc2, 0, 0, 0x03]);
        let head = |dst: MacAddr| FrameHead {
            dst,
            src: NEIGHBOUR,
            ethertype: EtherType::IPV4,
        };

        assert!(!deaf_f.accepts(&head(NEIGHBOUR), STATION));
        assert!(!deaf_f.accepts(&head(group), STATION));
        assert!(tap_f.accepts(&head(NEIGHBOUR), STATION), "the power widens it");
        assert!(tap_f.accepts(&head(group), STATION));

        // And the power is still not a protocol grant below S3: an S1 holder
        // that somehow held it reaches other stations, never other protocols.
        assert_eq!(deaf_f.promiscuous, false);
        assert_eq!(tap_f.promiscuous, true);
        assert_eq!(tap_f.multicast, MulticastMatch::All);
    }

    /// **Without `spoof`, an S3 holder cannot forge.** It may speak as the
    /// station its grant names and as nothing else; the power, not the rung,
    /// is what lifts that.
    ///
    /// The enforcement is the frame row's `validate_tx`, which is stricter
    /// still — it refuses any source but the vif's own address, consulting
    /// no grant — so this is the authority agreeing with the wire, never
    /// overriding it.
    #[test]
    fn an_interface_grant_forges_only_with_the_power() {
        let honest = FlowCap::mint(vif(), raw(), Powers::NONE).expect("mints");
        assert!(honest.may_source(STATION, STATION));
        assert!(
            !honest.may_source(NEIGHBOUR, STATION),
            "raw L2 is not a licence to be someone else"
        );

        let forger = FlowCap::mint(
            vif(),
            raw(),
            Powers {
                spoof: true,
                promiscuous: false,
            },
        )
        .expect("mints");
        assert!(forger.may_source(NEIGHBOUR, STATION));

        // Below S3 the question falls back to the vif's assigned address,
        // because no rung there names an L2 address of its own.
        let s1 = FlowCap::mint(vif(), Scope::Flow(conn()), Powers::NONE).expect("mints");
        assert_eq!(s1.scope().station(), None);
        assert!(s1.may_source(STATION, STATION));
        assert!(!s1.may_source(NEIGHBOUR, STATION));
    }

    /// **Derivation from S3 narrows and never widens**, in either dimension
    /// — including the one S3 adds: a child may not re-point the station.
    ///
    /// S2 between S3 and S1 stays modelled, so the ladder is crossed rather
    /// than walked: S3 derives straight to S1, and a child landing on S2 is
    /// refused for the rung rather than for its breadth.
    #[test]
    fn derivation_from_the_interface_grant_only_narrows() {
        let parent = FlowCap::mint(
            vif(),
            raw(),
            Powers {
                spoof: true,
                promiscuous: true,
            },
        )
        .expect("S3 mints");

        // Down the ladder, across the modelled rung.
        let child = parent
            .derive(Scope::Flow(conn()), Powers::NONE)
            .expect("S3 narrows to S1 while S2 stays modelled");
        assert_eq!(child.scope().rung(), 1);
        assert_eq!(child.powers(), Powers::NONE);
        assert_eq!(child.vif(), parent.vif(), "and stays on the parent's vif");
        assert!(parent.derive(Scope::Mediated, Powers::NONE).is_ok());

        // The unbuilt rung is refused for being unbuilt, not for being wide.
        assert_eq!(
            parent
                .derive(
                    Scope::Range {
                        addr: LOCAL,
                        ports: PortRange::new(400, 500).expect("ordered"),
                    },
                    Powers::NONE,
                )
                .unwrap_err(),
            FlowRefusal::RungNotMintable
        );

        // Sideways is not narrower: another station was never this
        // authority's to hand on.
        assert_eq!(
            parent
                .derive(
                    Scope::Interface {
                        station: NEIGHBOUR
                    },
                    Powers::NONE,
                )
                .unwrap_err(),
            FlowRefusal::WiderScope
        );
        assert!(
            parent.derive(raw(), Powers::NONE).is_ok(),
            "its own station is not a widening"
        );

        // Powers drop and never grow, at this rung as at every other.
        let narrowed = parent
            .derive(
                raw(),
                Powers {
                    spoof: false,
                    promiscuous: true,
                },
            )
            .expect("dropping one is lawful");
        assert_eq!(
            narrowed
                .derive(
                    raw(),
                    Powers {
                        spoof: true,
                        promiscuous: true,
                    },
                )
                .unwrap_err(),
            FlowRefusal::WiderPowers,
            "a dropped power does not come back"
        );
        // And the dropped one is gone from the enforcement, not just the
        // record: the child's filter is narrower than its parent's.
        assert!(!narrowed.may_source(NEIGHBOUR, STATION));
    }

    /// Containment at S3 is not a free pass in both directions: an interface
    /// grant contains every rung below it and is contained by no rung at all.
    #[test]
    fn the_interface_rung_is_the_ceiling_of_the_ladder() {
        let s1 = Scope::Flow(conn());
        let s2 = Scope::Range {
            addr: LOCAL,
            ports: PortRange::new(400, 500).expect("ordered"),
        };

        for narrower in [Scope::Mediated, s1, s2] {
            assert!(narrower.within(&raw()), "{narrower:?} is inside S3");
            assert!(!raw().within(&narrower), "S3 is inside nothing below it");
        }
        assert!(raw().within(&raw()));
        assert!(
            !raw().within(&Scope::Interface {
                station: NEIGHBOUR
            }),
            "two stations are two grants, not one"
        );
    }

    // ── revocation ──────────────────────────────────────────────────────

    /// Revocation is the generation tag: a cap outlives its vif's close, and
    /// stops resolving at its next use rather than needing to be found.
    #[test]
    fn a_stale_generation_stops_resolving() {
        let cap = FlowCap::mint(vif(), Scope::Flow(conn()), Powers::NONE).expect("mints");

        assert!(cap.resolves_against(vif()), "the live identifier");

        let reused = VifId {
            index: 3,
            generation: 8,
        };
        assert!(
            !cap.resolves_against(reused),
            "the slot's next tenant is not this authority's"
        );

        let elsewhere = VifId {
            index: 4,
            generation: 7,
        };
        assert!(!cap.resolves_against(elsewhere), "another slot entirely");
    }

    /// A derived child inherits its parent's vif binding, so revoking the vif
    /// stops the whole derivation subtree at once.
    #[test]
    fn a_derived_child_shares_the_revocation() {
        let parent = FlowCap::mint(vif(), Scope::Flow(conn()), Powers::NONE).expect("mints");
        let child = parent.derive(Scope::Mediated, Powers::NONE).expect("narrows");

        assert_eq!(child.vif(), parent.vif());
        let reused = VifId {
            index: 3,
            generation: 8,
        };
        assert!(!child.resolves_against(reused));
    }

    // ── admission ───────────────────────────────────────────────────────

    /// What the authority is actually *for*: it admits its own flow and
    /// nothing else, which is the same predicate the classifier enforces one
    /// layer down.
    #[test]
    fn an_authority_admits_only_its_own_flow() {
        let cap = FlowCap::mint(vif(), Scope::Flow(conn()), Powers::NONE).expect("mints");

        assert!(cap.admits(&tuple(PEER, LOCAL, 44000, 443)));
        assert!(
            !cap.admits(&tuple(OTHER, LOCAL, 44000, 443)),
            "same local endpoint, another peer"
        );
        assert!(
            !cap.admits(&tuple(PEER, LOCAL, 44001, 443)),
            "another peer port"
        );
        assert!(
            !cap.admits(&tuple(PEER, LOCAL, 44000, 80)),
            "another local port"
        );
    }
}
