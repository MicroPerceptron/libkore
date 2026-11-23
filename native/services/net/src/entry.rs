//! **The row's entry ladder** — claim a controller, bring up whatever is
//! plugged into it, prove the datapath once, then serve.
//!
//! Every rung that can fail on a machine we do not own reports rather than
//! refuses: a box with no second controller, no adaptor, or no carrier still
//! founds this row and answers every verb
//! [`NetRefusal::NotReady`](crate::NetRefusal::NotReady), with the readiness
//! note ([`crate::note`]) naming which. Only *wiring* faults NACK — a row
//! founded off the chain, or without the broker face it must claim through.

use kore_abi::session::Notice;
use kore_abi::task::role;
use kore_driver::net::filter::EtherTypeMatch;
use kore_driver::net::window::WINDOW_BYTES;
use kore_driver::net::window::{
    FRAME_AREA_BYTES, RX_RING_AT, TX_AREA_AT, TX_DEPTH, rx_frame_at, rx_slot_of, tx_frame_at,
    tx_slot_of,
};
use kore_driver::net::{EtherType, FrameDesc, FrameFilter, MacAddr};
use kore_ring::spsc;
use kore_rt::ChanTx;
use kore_rt::entry::{Rt, SERVE_SCRATCH_SLOTS, refuse_with, serve_scratch_base};
use kore_rt::service::{Bundle, DeviceSelector, MAX_GRANTS, SystemService};
use kore_rt::svc::{
    COOKIE_SERVING, INET_DNS_READY, INET_ICMP_RT_READY, INET_PING_WAN_READY, NET_FRAME_RT_READY,
    NET_LOAD_READY, NET_MT7925_FW_PINNED, NET_MT7925_PATCH_RESIDENT,
    MT7925_LADDER_D0, MT7925_LADDER_D1_MCU, MT7925_LADDER_D1_RING, MT7925_LADDER_FW_ALIVE,
    MT7925_LADDER_PATCH, MT7925_LADDER_RAM_SENT,
    NET_PCI_FRAME_READY,
    NET_VIF_ISOLATION_READY,
    NET_WIFI_FW_ALIVE, REQ_CAP,
    ServeEndpoints, ServeHost, serve_rings,
};

use crate::arp::{self, Heard};
use crate::metal::{
    E1000eConfig, E1000eHost, E1000eNetBinding, EcmHost, FrameHost, MetalHost, Mt7925Config,
    Mt7925Host, Mt7925NetBinding, NetEndowment, NetRow, RadioHost, Rtw89Config, Rtw89Host,
    Rtw89NetBinding, XhciNetBinding,
};
// Spelled through its module: `metal` re-exports both radio compositions and
// each names a BAR, so a bare `REGS_BAR` is ambiguous — and would be worse if
// it were not, since it would silently resolve to one part's window.
use crate::metal::wifi::REGS_BAR;
use crate::service::PAIRS;
use crate::{NET_EXEC_PAIR, NET_REPLY_GRANTS, NetService, exec_pair_policy, note, pair_policy};

/// The bring-up ladder's NACK rung codes — wiring faults only, sharing the
/// numbering every other driver row uses so there is one decoder ring.
mod rung {
    /// 2 — endowment resolve (pool, channel space).
    pub const ENDOWMENT: u64 = kore_rt::entry::RUNG_SERVICE_ENDOWMENT;
    /// 5 — the serve ladder (ring window, halves, kick, return post).
    pub const SERVE_SETUP: u64 = kore_rt::entry::RUNG_SERVICE_SERVE_SETUP;
    /// 17 — the broker's client face is missing from the boot block: the row
    /// was founded off the chain, or before the broker.
    pub const DEVICE_FACE: u64 = 17;
}

/// Nanoseconds the founding self-test will spend waiting for its round
/// trip — and, on a network where nobody answers for QEMU's gateway, for
/// the first station to be heard from so it can be asked instead.
///
/// The founder gives a row 120 seconds to reach readiness, and this test runs
/// *inside* that budget — so a dead wire must cost a note, never the row. A
/// SLIRP answer is a millisecond-scale event. On metal the wait is for the
/// far end to say something: a host whose link just came up announces its
/// address within a second or two (macOS and Linux both do, on the link
/// transition the adaptor's own bring-up causes), and five seconds covers
/// that with margin while staying more than an order inside the founder's
/// patience.
const SELF_TEST_BUDGET_NS: u64 = 5_000_000_000;

/// The clock-less fallback: sweeps spent waiting when this boot stamped no
/// calibration page, so [`SELF_TEST_BUDGET_NS`] cannot be read.
const SELF_TEST_SWEEPS: u32 = 200_000;

/// Frames the founding self-test's load burst pushes after the single
/// round trip: **one full transmit ring**, so the burst is exactly what a
/// client can have queued, the ring wraps under it, and the service's
/// in-flight cap (`MAX_TX_INFLIGHT`) is crossed several times over. Every
/// one is a unicast ARP request to whoever answered the first — the
/// gateway on QEMU, the real one on metal — so the load costs the network
/// one station's worth of unicast, never a broadcast storm.
const LOAD_BURST: u32 = TX_DEPTH;

/// The row's declared selectors, by index into `NetRow::DEVICES` — the
/// manifest's order, cited rather than re-typed. The ladder tries the PCI
/// NIC first: it is the adaptor itself, needs no port walk, and a machine
/// that has one should not spend the boot chain probing hubs.
const SELECTOR_XHCI: usize = 0;
/// See [`SELECTOR_XHCI`].
const SELECTOR_PCI: usize = 1;
/// See [`SELECTOR_XHCI`]: the Wi-Fi function, tried after the wired NIC.
const SELECTOR_WIFI: usize = 2;
/// See [`SELECTOR_XHCI`]: the **other** Wi-Fi function — the MediaTek part.
///
/// Its own selector because [`SELECTOR_WIFI`] demands the exact RTL8922AE
/// identity, as every selector here demands an exact one. The two parts
/// share a class and nothing else: BAR2 against BAR0.
const SELECTOR_MT7925: usize = 3;
const _: () = {
    let devices = <NetRow as SystemService>::DEVICES;
    assert!(devices.len() == 4, "four selectors, cited by index below");
    assert!(
        matches!(devices[SELECTOR_XHCI].selector, DeviceSelector::Pci(p) if p.class == 0x0c),
        "index 0 is the serial-bus (xHCI) class"
    );
    assert!(
        matches!(devices[SELECTOR_PCI].selector, DeviceSelector::Pci(p) if p.class == 0x02 && p.subclass == 0x00),
        "index 1 is the Ethernet controller class"
    );
    assert!(
        matches!(devices[SELECTOR_MT7925].selector, DeviceSelector::Pci(p) if p.class == 0x02 && p.subclass == 0x80),
        "index 3 is the other network controller (other) class — the MediaTek part"
    );
    assert!(
        matches!(devices[SELECTOR_WIFI].selector, DeviceSelector::Pci(p) if p.class == 0x02 && p.subclass == 0x80),
        "index 2 is the other-network-controller class (the Wi-Fi function)"
    );
};

/// Run the **frame service row**.
pub async fn run(rt: Rt) -> ! {
    let boot = rt.bootstrap();
    let Ok(endowment) = NetEndowment::resolve(boot) else {
        refuse_with(&rt, rung::ENDOWMENT);
    };
    let mut row = NetRow::bind(endowment);
    let transport = kore_rt::entry::transport();

    // ── Claim a controller and find the adaptor on it ───────────────────
    let Some(pool) = row.take_pool() else {
        refuse_with(&rt, rung::ENDOWMENT);
    };
    let Some(radio_pool) = row.take_radio_pool() else {
        refuse_with(&rt, rung::ENDOWMENT);
    };
    let channels = row.take_channels();
    // The founder-delivered Wi-Fi firmware module (`role::FIRMWARE_STORE`
    // coordinate 2), mapped WHOLE and read-only — the exact bytes `make
    // fetch-realtek-firmware` staged, the ivpu row's posture. Absent (no
    // module in this boot) is `None`; a claimed radio then refuses at its
    // own rung. The mapped span is page-rounded; the ladder recovers the
    // exact blob by the pinned length before the digest check. The mapping
    // outlives the possession token: `map_held` installs it into this task's
    // space for the task's life.
    let firmware: Option<&'static [u8]> = row
        .take_firmware()
        .and_then(|store| kore_rt::cap::verbs::map_held(store.slot()))
        // SAFETY: `map_held` installed the module's whole span R/O over this
        // task's address space for its life; the firmware bytes are immutable.
        .map(|win| unsafe { win.as_ref() });
    // The MT7925's two modules, mapped on the same terms as the Realtek
    // one above: page-rounded spans, read-only for this task's life, and
    // the pinned length recovers the exact blob before the digest check.
    let mut map_module = |store: Option<kore_rt::cap::ImageStoreToken>| {
        store
            .and_then(|s| kore_rt::cap::verbs::map_held(s.slot()))
            // SAFETY: as above — `map_held` installed the module's whole
            // span R/O over this task's address space for its life, and a
            // boot module's bytes are immutable.
            .map(|win| unsafe { win.as_ref() })
    };
    let mt7925_patch: Option<&'static [u8]> = map_module(row.take_mt7925_patch());
    let mt7925_ram: Option<&'static [u8]> = map_module(row.take_mt7925_ram());
    let (host, mut claim_note) =
        match claim_controller(
            transport,
            boot,
            pool,
            radio_pool,
            firmware,
            mt7925_patch,
            mt7925_ram,
        )
        .await
        {
            Ok(host) => {
                // The bound frame host says what it earned — three outcomes
                // on USB, not two (`NicBound::founding_note`), always live
                // on PCI — and the row prints whichever rung the walk
                // reports rather than re-deriving it. A controller with
                // nothing plugged in beside a bound radio is the ordinary
                // radio-only machine, not a degradation: the radio is the
                // network device, and `status net` says it lends no frames.
                let note = match host.founding_note() {
                    note::NO_ADAPTOR if host.is_wifi() => 0,
                    note => note,
                };
                (host, note)
            }
            Err(ClaimFailed::Wiring) => refuse_with(&rt, rung::DEVICE_FACE),
            Err(ClaimFailed::Note(n)) => (MetalHost::IDLE, n),
        };
    let has_route = host.has_route();
    let port = host.detail();
    let pci = host.is_pci();
    let wifi = host.is_wifi();
    let wifi_cap = host.wifi_phy_cap();
    let mt7925_d0 = host.mt7925_revision().is_some();
    let mt7925_ring = host.mt7925_ring_ready();
    let mt7925_mcu = host.mt7925_mcu_answered();
    let mt7925_rung = host.mt7925_bring_up_rung();
    let mt7925_fw_pinned = host.mt7925_firmware_pinned();
    let mt7925_patch = host.mt7925_patch();
    let mt7925_ram = host.mt7925_ram();
    let mt7925_fw_alive = host.mt7925_fw_alive();
    let mt7925_phy = host.mt7925_phy();
    let mt7925_detail = host.mt7925_bring_up_detail();
    let mt7925_evidence = host.mt7925_bring_up_evidence();
    // Pair 0's authority for every pair the runtime can serve, except the
    // one the founder hands to exec, which may open nothing.
    let mut policies = [pair_policy(); PAIRS];
    policies[NET_EXEC_PAIR] = exec_pair_policy();
    let mut svc = NetService::host(host, channels, policies);

    // The port the adaptor answered on rides the note's detail bits, so a
    // photo of a serial-less machine says *where* it was found — the only way
    // to tell one controller's ports from another's after the fact.
    let mut witness = 0u64;
    if wifi {
        // The radio lends no frame path at D1, so no datapath test can say
        // anything about it. What *is* proven is the bind itself: firmware
        // verified, downloaded and answering — which no lower rung lets
        // through — and the answer naming a PHY with at least one stream
        // each way, which a firmware echoing zeros would not. Derived from
        // the host the claim ladder bound and from what its firmware said,
        // never declared.
        match wifi_cap {
            Some(cap) if cap.rx_nss >= 1 && cap.tx_nss >= 1 => witness |= NET_WIFI_FW_ALIVE,
            _ => claim_note = note::WIFI_BRING_UP_BASE + u64::from(kore_rtw89::Error::NoAck.rung()),
        }
    }
    if mt7925_d0 {
        witness |= crate::metal::mt7925::climb(
            mt7925_ring,
            mt7925_mcu,
            mt7925_patch,
            mt7925_ram.is_some(),
            mt7925_fw_alive.is_some(),
            mt7925_phy.is_some(),
        );
        // **About the image, not the part**, so it is not a step and
        // hangs off `mt7925_d0` only because a row with no MT7925 has
        // nothing to say about the MT7925's firmware — not because the
        // part did anything to earn it.
        if mt7925_fw_pinned {
            witness |= NET_MT7925_FW_PINNED;
        }
    }
    if claim_note == 0 && svc.ready() {
        // A frame host is bound — the wired NIC, or an adaptor beside the
        // radio — so the datapath is proven on it, whatever else the boot
        // bound.
        let (bits, failed) = self_test(&mut svc).await;
        witness |= bits;
        if failed != 0 {
            claim_note = failed;
        }
        // Derived from the round trip and from which rung of the claim
        // ladder earned it, never declared: a PCI host that proved nothing
        // attests nothing, and a USB host can never carry this bit.
        if pci && bits & NET_FRAME_RT_READY != 0 {
            witness |= NET_PCI_FRAME_READY;
        }
    }
    // **A part that reached D1 and stopped says where.**
    //
    // After the self-test, so a machine carrying both a frame host and the
    // radio still reports the datapath's own verdict first — that one is
    // about moving frames, and this one is about a radio that cannot yet.
    // Before the port composition, so the detail bits compose as usual.
    //
    // Read off `MetalHost`, which reads `self.radio`. The first attempt at
    // this wrote the note onto the bound part instead, where
    // `founding_note` would have found it if `founding_note` looked at
    // radios. It does not, so the row reported nothing, twice.
    if claim_note == 0 {
        claim_note = note::mt7925(mt7925_d0, mt7925_rung, mt7925_detail, mt7925_evidence);
    }
    if claim_note != 0 && port != 0 {
        claim_note |= u64::from(port) << 8;
    }

    // The steady park arms the held MSI route beside the doorbell — which is
    // also what earns the boot gate's `-remapped-msi-irq` witness at the
    // beacon on a route-holding machine. Derived from the observation, never
    // declared, so a routeless row cannot claim it.
    // The row's transient reservation is the VA band minted pair windows
    // map into (`service::peers`). Geometry reserves address space only —
    // each admitted client supplies its own backing — so a row whose boot
    // stamped no geometry simply refuses admission rather than founding
    // degraded: the mint is not a datapath, and no witness depends on it.
    if let Some(window) = rt
        .geom()
        .and_then(|geom| geom.window_of(kore_abi::task::geom::TRANSIENT))
    {
        svc.configure_peer_map(window.base());
    }
    row.adopt(svc);
    if has_route {
        kore_rt::executor::enable_irq_parks().await;
    }

    // ── The serve ladder ────────────────────────────────────────────────
    let Some(view) = boot.handle(role::SVC_RING, 0) else {
        refuse_with(&rt, rung::SERVE_SETUP);
    };
    let Some(leased) = kore_rt::cap::verbs::take_view(view) else {
        refuse_with(&rt, rung::SERVE_SETUP);
    };
    let Some(window) = kore_rt::cap::verbs::map_view(leased) else {
        refuse_with(&rt, rung::SERVE_SETUP);
    };
    // SAFETY: `map_view` installed its reported `[base, +len)` R/W-USER in
    // this task's own space over the pair's leased backing, page-aligned and
    // live for the task's whole life; this task is the pair's sole
    // service-side attacher (the client half lives in another space and
    // touches it only through the SPSC protocol).
    let window: &'static mut [u8] =
        unsafe { core::slice::from_raw_parts_mut(window.cast::<u8>().as_ptr(), window.len()) };
    let Ok(rings) = serve_rings::<NetRow>(window, REQ_CAP, kore_rt::svc::REP_CAP) else {
        refuse_with(&rt, rung::SERVE_SETUP);
    };
    let Some(kick) = boot.handle(role::SVC_SEND, 0) else {
        refuse_with(&rt, rung::SERVE_SETUP);
    };
    // One reply grant: the `Signal` verb's retained receive doorbell.
    let mut grant_tx = [None; MAX_GRANTS];
    let Some(g0) = boot.handle(role::SVC_GRANT, 0) else {
        refuse_with(&rt, rung::SERVE_SETUP);
    };
    grant_tx[0] = Some(g0);
    let Some(scratch_base) = serve_scratch_base(&rt) else {
        refuse_with(&rt, rung::SERVE_SETUP);
    };
    let mut scratch = [None; MAX_GRANTS];
    for (k, sc) in scratch.iter_mut().enumerate().take(SERVE_SCRATCH_SLOTS) {
        *sc = Some(kore_abi::session::SlotRef::from_wire(
            scratch_base + k as u32,
        ));
    }
    let ep = ServeEndpoints {
        kick,
        grant_tx,
        scratch,
        // The return leg every `OpenVif` window rides in on.
        return_post: boot
            .handle(role::SVC_RETURN, 0)
            .zip(boot.handle(role::SVC_RETURN, 1)),
    };
    let Some(beacon) = boot.handle(role::PARENT_SEND, 0) else {
        kore_rt::entry::refuse();
    };
    let mut host = ServeHost::new(row, transport, rings, ep);

    // ── Pair 1: exec's ──────────────────────────────────────────────────
    // The row's FINAL pair belongs to the resident exec manager (the
    // founder's exec-provider class, `exec_client = true`): the same
    // window / kick / grant / return shape as pair 0, at the runtime's
    // pair-`k` handles (`SVC_RING@k`, `SVC_SEND@k`, `SVC_GRANT@(k * grants)`
    // as the bank's base, `SVC_RETURN@2k / @2k+1`), served under
    // `exec_pair_policy` — deny — so exec's two read-only verbs are all it
    // can ever ask.  Its return post stands beside pair 0's below, before
    // the beacon, so a client that sees `COOKIE_SERVING` never wins a
    // transfer against an un-stood post on either pair.
    {
        let k = NET_EXEC_PAIR as u16;
        let Some(view) = boot.handle(role::SVC_RING, k) else {
            refuse_with(&rt, rung::SERVE_SETUP);
        };
        let Some(leased) = kore_rt::cap::verbs::take_view(view) else {
            refuse_with(&rt, rung::SERVE_SETUP);
        };
        let Some(window) = kore_rt::cap::verbs::map_view(leased) else {
            refuse_with(&rt, rung::SERVE_SETUP);
        };
        // SAFETY: as for pair 0's window above — this task's own `map_view`
        // of pair 1's leased backing, whose sole other toucher is exec,
        // through the SPSC protocol.
        let window: &'static mut [u8] =
            unsafe { core::slice::from_raw_parts_mut(window.cast::<u8>().as_ptr(), window.len()) };
        let Ok(rings) = serve_rings::<NetRow>(window, REQ_CAP, kore_rt::svc::REP_CAP) else {
            refuse_with(&rt, rung::SERVE_SETUP);
        };
        let Some(kick) = boot.handle(role::SVC_SEND, k) else {
            refuse_with(&rt, rung::SERVE_SETUP);
        };
        let mut grant_tx = [None; MAX_GRANTS];
        let Some(g0) = boot.handle(role::SVC_GRANT, k * NET_REPLY_GRANTS as u16) else {
            refuse_with(&rt, rung::SERVE_SETUP);
        };
        grant_tx[0] = Some(g0);
        let ep = ServeEndpoints {
            kick,
            grant_tx,
            // The derive scratch is shared: the serve loop is sequential,
            // and a scratch slot lives only inside one request's session.
            scratch,
            return_post: boot
                .handle(role::SVC_RETURN, 2 * k)
                .zip(boot.handle(role::SVC_RETURN, 2 * k + 1)),
        };
        if !host.attach_pair(NET_EXEC_PAIR, rings, ep) {
            refuse_with(&rt, rung::SERVE_SETUP);
        }
    }

    // The return post stands BEFORE the serving beacon (the board law: a
    // client that sees `COOKIE_SERVING` can never win a return send against
    // an un-stood post).
    if !host.stand_return_post().await {
        refuse_with(&rt, rung::SERVE_SETUP);
    }
    let _ = ChanTx::new(transport, beacon)
        .send(Notice {
            cookie: COOKIE_SERVING,
            payload: claim_note | witness,
        })
        .await;

    loop {
        // Total drain of every pending request (§3.1) — a client's `OpenVif`
        // must be visible to the sweep below in the same wake it arrived.
        while host.serve_next().await {}
        // The kernel's terminal ask (#809 §L), checked between wakes: halt
        // and tear down the hosted xHC; the row keeps serving.
        if let Some(budget_ns) = kore_rt::executor::take_quiesce_ask().await {
            let _ = host
                .service_mut()
                .quiesce_devices(kore_rt::executor::quiesce_deadline(budget_ns))
                .await;
        }
        // Then the data plane: one bounded sweep, and one wake notice per
        // pair that received something. The poll is the truth and the notice
        // only wakes, so a refused notice costs a wake and never a frame.
        // A link edge the sweep derived goes to every pair holding a
        // doorbell the same way; the level rides `Query`.
        let mut service_deadline = None;
        if let Some(svc) = host.service_mut().service_mut() {
            let report = svc.drive();
            service_deadline = report.service_deadline;
            if report.wake != 0 {
                let _ = svc.pump_notices(transport, report.wake).await;
            }
            if report.link_changed {
                let _ = svc.pump_link_notices(transport).await;
            }
        }
        match service_deadline {
            Some(deadline) => {
                let _ = kore_rt::executor::next_wake_until(deadline).await;
            }
            None => {
                let _ = kore_rt::executor::next_wake().await;
            }
        }
    }
}

/// Why the claim step yielded no bound controller.
enum ClaimFailed {
    /// The broker face is not in the boot block — a founding fault, and the
    /// one claim failure that NACKs the row.
    Wiring,
    /// A hardware or broker fact, reported as the readiness note.
    Note(u64),
}

/// **Claim the network hosts from the device broker**, PCI first.
///
/// Rung A asks every live root for the exact 82574L identity; a function
/// that decodes and names the generation is bound as the frame host and
/// the ladder ends — a wired NIC is the whole answer. Otherwise rung A′
/// binds the Wi-Fi function, over its own supply, for its facts and its
/// firmware-alive proof, and rung B — the xHCI walk below — runs **beside**
/// it rather than after it: the radio lends no frames at #388 D1, so a
/// ladder that stopped at the first device to bind would let a radio
/// shadow the Ethernet adaptor plugged in next to it. Both `NoMatch` and
/// `Taken` on a rung are honest reasons to move on, and only a broker
/// refusal ends the claim.
///
/// The broker's ledger hands each function out at most once, keyed by the
/// claiming pair, so this row and the input router necessarily get
/// *different* xHCI controllers — that part is by design. Which one each
/// gets is not: the input router founds first and takes the first
/// controller with a connected port, so on a machine with one xHC and no
/// PCI NIC this row claims nothing and founds degraded. Moving the adaptor
/// is the operator's fix; the note says so rather than leaving them to guess.
async fn claim_controller(
    transport: &kore_rt::transport::TrapTransport,
    boot: &kore_abi::task::Bootstrap<'_>,
    pool: kore_rt::cap::SupplyToken,
    radio_pool: kore_rt::cap::SupplyToken,
    firmware: Option<&'static [u8]>,
    mt7925_patch: Option<&'static [u8]>,
    mt7925_ram: Option<&'static [u8]>,
) -> Result<MetalHost, ClaimFailed> {
    use kore_driver::host::{DeviceProbe, IntakeError};
    use kore_svc_device::client::{Claimed, ClaimedDevice, FaceLayout, MultiBroker};

    // The faces sit past this row's own two serving pairs, and past any
    // declared pair it provides (#1202) — the same derivation the founder
    // stamps the bank with.
    let Some(mut broker) = MultiBroker::open(
        transport,
        boot,
        FaceLayout::for_bootstrap(
            boot,
            <NetRow as kore_rt::svc::SvcProtocol>::SERVE_PAIRS,
            <NetRow as kore_rt::svc::SvcProtocol>::BOOT_REPLY_GRANTS,
        ),
    ) else {
        return Err(ClaimFailed::Wiring);
    };

    // ── Rung A: the PCI NIC ─────────────────────────────────────────────
    //
    // The pool is this row's one DMA supply and the broker cannot un-claim,
    // so the function is probed **pool-free** first (`res/DEVICE_SERVICE.md`
    // §10): only a BAR that decodes a controller whose identity names a
    // generation this core drives is allowed to spend it. A claimed function
    // that fails either check is dropped — quarantined for the boot, which
    // beats a second owner — and the ladder descends to USB.
    let pci = <NetRow as SystemService>::DEVICES[SELECTOR_PCI].selector;
    match broker.claim_first(pci).await {
        Ok(Claimed {
            device,
            vendor,
            device_id,
            ..
        }) => {
            let ClaimedDevice::Pci(token) = device else {
                return Err(ClaimFailed::Note(note::DENIED));
            };
            let product = kore_e1000e::Product::from_pci(vendor, device_id);
            let decoded = match DeviceProbe::open(token, 0) {
                Ok(probe) => {
                    let present =
                        kore_e1000e::regs::controller_present(probe.regs(), probe.regs().len());
                    match (present, product) {
                        (true, Some(product)) => probe.finish().ok().map(|token| (token, product)),
                        _ => None,
                    }
                }
                Err(_) => None,
            };
            if let Some((token, product)) = decoded {
                return match kore_driver::host::BoundDevice::<E1000eNetBinding>::claim_with_config(
                    token,
                    pool,
                    E1000eConfig { product },
                ) {
                    Ok(device) => Ok(MetalHost::new(
                        FrameHost::Pci(E1000eHost::bound(device)),
                        None,
                    )),
                    // The core's own rung, carried out through the intake so
                    // a serial-less photo says where the ladder stopped.
                    Err(IntakeError::ClassBindRefused { code, .. }) => {
                        Err(ClaimFailed::Note(note::PCI_BRING_UP_BASE + u64::from(code)))
                    }
                    Err(_) => Err(ClaimFailed::Note(note::BIND_REFUSED)),
                };
            }
        }
        Err(kore_svc_device::DeviceRefusal::NoMatch)
        | Err(kore_svc_device::DeviceRefusal::Taken) => {}
        Err(_) => return Err(ClaimFailed::Note(note::DENIED)),
    }

    // ── Rung A′: the Wi-Fi function ──────────────────────────────────────
    //
    // The same shape: the exact RTL8922AE identity, probed pool-free on
    // BAR2 (the register window; BAR0 is port I/O the kernel never maps)
    // before its supply is spent — its own supply, so binding it takes
    // nothing from the frame host rung B may still bind. The firmware the
    // boot carried rides the config; a boot without it reaches the ladder's
    // own rung rather than a made-up blob. A bound radio is #388 D1's proof
    // and lends no frames yet, which the row reports honestly; a radio that
    // is present and refuses its ladder is the fault to fix first, and its
    // rung is the note.
    let mut radio = None;
    // At most one radio binds, and whichever does consumes the supply — so
    // it is held in an `Option` and taken, rather than moved into an arm the
    // compiler cannot prove is the only one that runs.
    let mut radio_pool = Some(radio_pool);
    let wifi = <NetRow as SystemService>::DEVICES[SELECTOR_WIFI].selector;
    match broker.claim_first(wifi).await {
        Ok(Claimed {
            device,
            vendor,
            device_id,
            ..
        }) => {
            let ClaimedDevice::Pci(token) = device else {
                return Err(ClaimFailed::Note(note::DENIED));
            };
            // Which part is in the socket decides everything below, and it
            // is decided before the token is spent: `DeviceProbe::open`
            // consumes it, so a probe for the wrong part would leave nothing
            // to bind the right one with.
            if let Some(product) = kore_rtw89::Product::from_pci(vendor, device_id) {
                let decoded = match DeviceProbe::open(token, REGS_BAR) {
                    Ok(probe) => {
                        let present =
                            kore_rtw89::regs::controller_present(probe.regs(), probe.regs().len());
                        if present { probe.finish().ok() } else { None }
                    }
                    Err(_) => None,
                };
                if let Some(token) = decoded
                    && let Some(pool) = radio_pool.take()
                {
                    match kore_driver::host::BoundDevice::<Rtw89NetBinding>::claim_with_config(
                        token,
                        pool,
                        Rtw89Config { product, firmware },
                    ) {
                        Ok(device) => radio = Some(RadioHost::Rtw89(Rtw89Host::bound(device))),
                        Err(IntakeError::ClassBindRefused { code, .. }) => {
                            return Err(ClaimFailed::Note(
                                note::WIFI_BRING_UP_BASE + u64::from(code),
                            ));
                        }
                        Err(_) => return Err(ClaimFailed::Note(note::BIND_REFUSED)),
                    }
                }
            }
        }
        Err(kore_svc_device::DeviceRefusal::NoMatch)
        | Err(kore_svc_device::DeviceRefusal::Taken) => {}
        Err(_) => return Err(ClaimFailed::Note(note::DENIED)),
    }

    // ── The MediaTek radio, on its own selector ─────────────────────────
    //
    // Tried only if the Realtek one did not bind: the lab has one machine
    // and one radio, and a claim for a part that is not there costs a
    // broker round trip and nothing else.
    //
    // This needs its own `claim_first` because `SELECTOR_WIFI` demands the
    // exact RTL8922AE identity — `14c3:0717` never met it, which is why an
    // arm hung off that claim never ran.
    if radio.is_none() {
        let mt = <NetRow as SystemService>::DEVICES[SELECTOR_MT7925].selector;
        match broker.claim_first(mt).await {
            Ok(Claimed {
                device,
                vendor,
                device_id,
                ..
            }) => {
                let ClaimedDevice::Pci(token) = device else {
                    return Err(ClaimFailed::Note(note::DENIED));
                };
                // The selector already pinned the identity; this re-reads it
                // from the claim so the core decides which part it is from
                // the function it was actually handed, not from a selector
                // it cannot see.
                if let Some(product) = kore_mt7925::Product::from_pci(vendor, device_id)
                    && let Some(pool) = radio_pool.take()
                {
                    // No pre-probe: D0's identity read *is* the "do these
                    // registers decode" question, it runs inside `prepare`
                    // before any DMA is activated, and it is stronger than a
                    // presence check — it drives the L1 remap window and
                    // compares against the value this silicon is known to
                    // answer (`res/MT7925_EVIDENCE.md` §11).
                    match kore_driver::host::BoundDevice::<Mt7925NetBinding>::claim_with_config(
                        token,
                        pool,
                        Mt7925Config {
                            product,
                            patch: mt7925_patch,
                            ram: mt7925_ram,
                        },
                    ) {
                        Ok(device) => radio = Some(RadioHost::Mt7925(Mt7925Host::bound(device))),
                        Err(IntakeError::ClassBindRefused { code, .. }) => {
                            return Err(ClaimFailed::Note(
                                note::MT7925_BRING_UP_BASE + u64::from(code),
                            ));
                        }
                        Err(_) => return Err(ClaimFailed::Note(note::BIND_REFUSED)),
                    }
                }
            }
            Err(kore_svc_device::DeviceRefusal::NoMatch)
            | Err(kore_svc_device::DeviceRefusal::Taken) => {}
            Err(_) => return Err(ClaimFailed::Note(note::DENIED)),
        }
    }

    // ── Rung B: an Ethernet adaptor behind an xHCI controller ───────────
    //
    // Beside the radio, not instead of it. No second controller to claim
    // is the radio-only laptop's ordinary posture (and the claim-less
    // machine's degraded one); any other refusal is reported as before.
    match claim_xhci(&mut broker, pool).await {
        Ok(device) => Ok(MetalHost::new(
            FrameHost::Usb(EcmHost::bound(device)),
            radio,
        )),
        Err(ClaimFailed::Note(note::NO_MATCH)) if radio.is_some() => {
            Ok(MetalHost::new(FrameHost::None, radio))
        }
        Err(refused) => Err(refused),
    }
}

/// **Claim an xHCI controller from the device broker** and bring up the
/// adaptor plugged into it.
async fn claim_xhci(
    broker: &mut kore_svc_device::client::MultiBroker<
        'static,
        '_,
        kore_rt::transport::TrapTransport,
    >,
    pool: kore_rt::cap::SupplyToken,
) -> Result<kore_driver::host::BoundDevice<XhciNetBinding>, ClaimFailed> {
    use kore_driver::host::DeviceProbe;
    use kore_svc_device::DeviceClient;
    use kore_svc_device::client::{Claimed, ClaimedDevice};

    // The selector this row declares — cited from its own measured image,
    // never a number written here.
    let selector = <NetRow as SystemService>::DEVICES[SELECTOR_XHCI].selector;

    // Prefer a controller with something plugged into it; keep the first that
    // merely decodes as the fallback, so a machine whose adaptor is behind a
    // hub that has not settled still gets a controller rather than nothing.
    let mut fallback: Option<Claimed> = None;
    let mut selected: Option<Claimed> = None;
    'faces: for k in 0..kore_abi::device::MAX_ROOT_COMPLEXES {
        let Some(face) = broker.face(k) else {
            continue;
        };
        let mut claims = 0usize;
        loop {
            match face
                .claim(kore_svc_device::DeviceRequest::Claim(selector))
                .await
            {
                Ok(claimed) => {
                    claims += 1;
                    let Claimed {
                        index,
                        device,
                        diag,
                        revision,
                        vendor,
                        device_id,
                    } = claimed;
                    let ClaimedDevice::Pci(token) = device else {
                        return Err(ClaimFailed::Note(note::DENIED));
                    };
                    let Ok(probe) = DeviceProbe::open(token, 0) else {
                        if claims == kore_svc_device::MAX_CLAIMS {
                            continue 'faces;
                        }
                        continue;
                    };
                    let ports = kore_xhci::regs::connected_ports(probe.regs(), probe.regs().len());
                    let connected = ports.is_some_and(|(connected, _)| connected > 0);
                    let decodes = ports.is_some();
                    let Ok(token) = probe.finish() else {
                        return Err(ClaimFailed::Note(note::NO_CANDIDATE));
                    };
                    let claimed = Claimed {
                        index,
                        device: ClaimedDevice::Pci(token),
                        diag,
                        revision,
                        vendor,
                        device_id,
                    };
                    if connected {
                        selected = Some(claimed);
                        break 'faces;
                    }
                    if decodes && fallback.is_none() {
                        fallback = Some(claimed);
                    } else {
                        // The temporary BAR is already retired; dropping the
                        // root reaps it here. The broker row stays
                        // quarantined — fail-closed beats a second owner.
                        drop(claimed);
                    }
                    if claims == kore_svc_device::MAX_CLAIMS {
                        continue 'faces;
                    }
                }
                Err(kore_svc_device::DeviceRefusal::Taken) => continue 'faces,
                Err(kore_svc_device::DeviceRefusal::NoMatch) => continue 'faces,
                Err(_) => return Err(ClaimFailed::Note(note::DENIED)),
            }
        }
    }
    let selected = match selected.or_else(|| fallback.take()) {
        Some(claimed) => claimed,
        None => return Err(ClaimFailed::Note(note::NO_MATCH)),
    };
    drop(fallback);
    let access = kore_xhci::pointer_access_for_pci(selected.vendor, selected.device_id);
    let Some(token) = selected.device.pci() else {
        return Err(ClaimFailed::Note(note::DENIED));
    };
    kore_driver::host::BoundDevice::claim_with_config(token, pool, access)
        .map_err(|_| ClaimFailed::Note(note::BIND_REFUSED))
}

// ── the founding self-test ─────────────────────────────────────────────

/// SLIRP's gateway and the address QEMU hands the guest — the two numbers a
/// user-mode network answers ARP for, which is why they live here beside
/// the self-test and not in the datapath. On metal nobody answers for the
/// gateway, and the test asks instead the first station it hears from for
/// that station's own address ([`Heard::Request`]) — every host answers
/// that, whatever subnet the asker claims to be on.
const SLIRP_GATEWAY: [u8; 4] = [10, 0, 2, 2];
const SLIRP_GUEST: [u8; 4] = [10, 0, 2, 15];
/// SLIRP's resolver, which forwards to the host's.
const SLIRP_DNS: [u8; 4] = [10, 0, 2, 3];
/// SLIRP hands out a /24.
const SLIRP_PREFIX_LEN: u8 = 24;
/// The name the milestone is written in terms of.
const PING_NAME: &str = "google.com";
/// Identifies this row's echoes among any others on the link.
const ECHO_IDENT: u16 = 0x6b6f;
/// The second echo session's identifier — the resolved WAN host's.
///
/// Distinct from the gateway's so each session absorbs only its own replies
/// when both are fed every ICMP event.
const ECHO_IDENT_WAN: u16 = 0x6b70;
/// How long DHCP gets before the probes start regardless.
///
/// Two of the five seconds. A SLIRP lease is a millisecond-scale event and a
/// real server answers in well under this; past it the self-test proceeds on
/// its own scaffolding so a network with no DHCP still earns what it can.
const DHCP_BUDGET_US: u64 = 2_000_000;
/// The seed this row's DHCP transaction identifiers derive from.
const DHCP_XID_SEED: u32 = 0x6b6f_7265;
/// Probe cadence. Each session sends exactly one probe, so this only bounds
/// how soon a retry could follow; it is never reached.
const ECHO_INTERVAL_US: u64 = 1_000_000;
/// A probe's deadline — the whole self-test budget, so a slow reply inside
/// the budget is still credited rather than expired a moment before it lands.
const ECHO_TIMEOUT_US: u64 = IP_SELF_TEST_BUDGET_NS / 1_000;
const DNS_XID: u16 = 0x6b64;
const DNS_LOCAL_PORT: u16 = 49152;
/// Longer than the frame stage's: a name resolution crosses SLIRP to the
/// host's resolver and back, and the WAN echo leaves the machine.
const IP_SELF_TEST_BUDGET_NS: u64 = 5_000_000_000;
const IP_SELF_TEST_SWEEPS: u32 = 200_000;

/// **Prove the datapath once, before the beacon.**
///
/// Opens two vifs over two windows — one filtered to ARP, one to the peer
/// EtherType — transmits an ARP request out of the first, and waits for
/// the reply. What that proves, end to end on real silicon: the client's
/// own slot reaching hardware zero-copy, the wire, the receive path back
/// (the bulk IN pipe and its transfer event on USB; the descriptor ring
/// and its done-bit write-back on PCI), the interrupt route, the
/// classifier, and the vif's ring. And what the second vif proves by
/// staying **empty**: the classifier delivers to the vifs whose filter
/// accepts a frame and to no others.
///
/// The request asks for QEMU's gateway, which SLIRP answers. On a real
/// network nobody does — so the first station heard *asking* for anything
/// is asked, once, for its own address, which every host answers. The
/// round trip is then the same proof on either.
///
/// Then, on a path that answered, **the load burst** ([`load_burst`]): a
/// full ring of unicast requests to the responder, drained as a client
/// would drain them. The isolation half is judged *after* the burst, so
/// the empty vif is empty under pressure and not merely at rest.
///
/// Returns `(witness bits, failing note)`. A machine with no carrier — or
/// one where nothing spoke within the budget — gets a note and serves: the
/// diskless posture, one class over. A path that answers one probe but
/// not the burst keeps the round-trip witness, lacks the load one, and
/// serves: the counters say the rest.
async fn self_test(svc: &mut NetService<MetalHost>) -> (u64, u64) {
    let Some(info) = svc.info() else {
        return (0, note::NO_ADAPTOR);
    };
    let mac = info.mac;

    let mut base = core::ptr::null_mut();
    let infra = svc.open_vif(0, FrameFilter::protocol(EtherType::ARP), |host, _| {
        let window = host.carve_own_window()?;
        crate::client::format_window(window.bytes)
            .map_err(|_| crate::service::ShareFault::NoView)?;
        base = window.bytes.as_mut_ptr();
        Ok(window)
    });
    let Ok((infra, _)) = infra else {
        return (0, note::NO_SELF_VIF);
    };
    let peer = svc.open_vif(0, FrameFilter::protocol(EtherType::KORE_PEER), |host, _| {
        let window = host.carve_own_window()?;
        crate::client::format_window(window.bytes)
            .map_err(|_| crate::service::ShareFault::NoView)?;
        Ok(window)
    });
    let Ok((peer, _)) = peer else {
        // The first vif is already open; close it so the row starts serving
        // against an empty table rather than one carrying a half-built test.
        let _ = svc.close_vif(0, infra);
        return (0, note::NO_SELF_VIF);
    };

    // SAFETY: `base` addresses this vif's whole window, carved from this
    // row's own pool and live for the row's life. The service holds the other
    // halves of the same rings; the two sides touch the window only through
    // the SPSC protocol and write disjoint slot subregions, which is the
    // discipline `spsc::split` sanctions for exactly this arrangement. The
    // three slices are disjoint subregions of that one window.
    let (tx_ring, rx_ring, frames) = unsafe {
        (
            core::slice::from_raw_parts_mut(base, RX_RING_AT),
            core::slice::from_raw_parts_mut(base.add(RX_RING_AT), TX_AREA_AT - RX_RING_AT),
            core::slice::from_raw_parts_mut(base.add(TX_AREA_AT), FRAME_AREA_BYTES),
        )
    };
    let (Ok(mut tx), Ok(mut rx)) = (
        spsc::producer::<FrameDesc>(tx_ring),
        spsc::consumer::<FrameDesc>(rx_ring),
    ) else {
        let _ = svc.close_vif(0, infra);
        let _ = svc.close_vif(0, peer);
        return (0, note::NO_SELF_VIF);
    };

    // A client's two cursors: the n-th push owns transmit slot `n mod
    // TX_DEPTH` and the n-th pop receive slot `n mod RX_DEPTH` — the same
    // derivation the service makes, so neither side ever reads a slot
    // number out of a record.
    let mut pushed: u64 = 0;
    let mut popped: u64 = 0;
    let request = arp::request(mac, SLIRP_GUEST, MacAddr::BROADCAST, SLIRP_GATEWAY);
    if !push_frame(&mut tx, frames, &mut pushed, &request) {
        let _ = svc.close_vif(0, infra);
        let _ = svc.close_vif(0, peer);
        return (0, note::NO_SELF_VIF);
    }

    // The round trip. Time-bounded when this boot has a clock,
    // sweep-bounded when it does not. Both bounds are present because
    // either alone has a failure mode: a clock-less boot would spin forever,
    // and a sweep count alone is a duration nobody can predict across TCG,
    // KVM and metal. Every reply is popped as a client would pop it; the
    // first station heard asking is asked, once, for its own address.
    let deadline = uptime_ns().map(|now| now.saturating_add(SELF_TEST_BUDGET_NS));
    let mut responder = None;
    let mut asked = false;
    'wait: for _ in 0..SELF_TEST_SWEEPS {
        let _ = svc.drive();
        while let Some(heard) = pop_frame(&mut rx, frames, &mut popped) {
            match heard {
                Heard::Reply { station, ip } => {
                    responder = Some((station, ip));
                    break 'wait;
                }
                Heard::Request { station, ip } if !asked => {
                    let probe = arp::request(mac, SLIRP_GUEST, station, ip);
                    asked = push_frame(&mut tx, frames, &mut pushed, &probe);
                }
                Heard::Request { .. } | Heard::Other => {}
            }
        }
        if let (Some(deadline), Some(now)) = (deadline, uptime_ns())
            && now >= deadline
        {
            break;
        }
        kore_driver::host::relax();
    }

    let mut witness = 0;
    let mut loaded = false;
    if let Some((station, ip)) = responder {
        witness |= NET_FRAME_RT_READY;
        loaded = load_burst(
            svc,
            mac,
            station,
            ip,
            &mut tx,
            &mut rx,
            frames,
            &mut pushed,
            &mut popped,
        );
    }
    let infra_counters = svc.counters(infra).unwrap_or_default();
    let peer_counters = svc.counters(peer).unwrap_or_default();
    if witness & NET_FRAME_RT_READY != 0 {
        // The isolation half is only testimony if the frames really did
        // land in one vif and not the other, and neither dropped anything
        // on the way — a full ring would make an empty peer vif prove
        // nothing. Judged after the burst, so it held under load.
        let isolated = peer_counters.rx_frames == 0
            && peer_counters.rx_dropped == 0
            && infra_counters.rx_dropped == 0;
        if isolated {
            witness |= NET_VIF_ISOLATION_READY;
        }
        // The load half: every frame this test pushed the device reported
        // sent, none refused or failed, every burst reply landed, nothing
        // dropped, and the other vif still empty. Derived from the counters
        // the sweeps kept, never from the loop having ended.
        if loaded
            && isolated
            && infra_counters.tx_frames == pushed
            && infra_counters.tx_failed == 0
            && infra_counters.tx_refused == 0
        {
            witness |= NET_LOAD_READY;
        }
    }

    // The self-test's vifs are its own; close them so a client opening one
    // starts against an empty table. The windows are pool buffers, so nothing
    // is retired — the pool never frees.
    let _ = svc.close_vif(0, infra);
    let _ = svc.close_vif(0, peer);

    // ── above L2 ──────────────────────────────────────────────────────
    //
    // Only once the frame path is proven: a stack that cannot get a frame
    // to the wire tells you nothing about the stack. Its own vif, because
    // the frame stage's filter is ARP-only and IP needs both.
    if witness & NET_FRAME_RT_READY != 0 {
        witness |= ip_self_test(svc, mac).await;
    }
    // Drain whatever the close parked, so the row serves with clean books.
    let _ = svc.drive();

    let failing = if witness & NET_FRAME_RT_READY != 0 {
        0
    } else {
        note::NO_FRAME_PATH
    };
    (witness, failing)
}

/// The stack, on the wire: ARP a gateway, echo it, resolve a name, echo
/// what the name resolved to.
///
/// Everything above L2 this row can prove about itself. It runs the real
/// [`IpStack`] over a real vif rather than hand-building frames the way the
/// frame stage does, because the thing under test *is* the engine — a
/// hand-rolled echo would prove the wire works twice and the stack never.
///
/// Three claims, earned separately and reported separately, because a boot
/// that gets two of them has said something true and should say it:
///
///   * the gateway answered an echo — ARP, IPv4 and ICMP end to end;
///   * a name resolved — UDP, the binding and the resolver;
///   * the address that name gave answered an echo — a host past this link.
///
/// Addresses are SLIRP's, which is what the gate boots against. On a
/// machine whose network is not SLIRP the ARP goes unanswered and the
/// budget expires: no witness, no note, and the row serves. That is the
/// same posture the frame stage takes and for the same reason — this is
/// testimony, not a requirement.
async fn ip_self_test(svc: &mut NetService<MetalHost>, mac: MacAddr) -> u64 {
    use kore_driver::net::flow::Ipv4Addr;
    use kore_inet::netinet::Event;
    use kore_inet::netinet::ping::{self, Ping};
    use kore_inet::{Identity, IpStack, VifIfnet};

    let mut filter = FrameFilter::protocol(EtherType::IPV4);
    // ARP beside IPv4: the stack cannot send a datagram to an address whose
    // hardware address it has not learned, and it learns that here.
    let Some(both) = EtherTypeMatch::only(&[EtherType::IPV4, EtherType::ARP]) else {
        return 0;
    };
    filter.ethertype = both;

    let mut base = core::ptr::null_mut();
    let opened = svc.open_vif(0, filter, |host, _| {
        let window = host.carve_own_window()?;
        crate::client::format_window(window.bytes)
            .map_err(|_| crate::service::ShareFault::NoView)?;
        base = window.bytes.as_mut_ptr();
        Ok(window)
    });
    let Ok((vif, _)) = opened else {
        return 0;
    };

    // SAFETY: as the frame stage above — this vif's whole window, carved
    // from this row's own pool, live for the row's life, and touched by the
    // two sides only through the SPSC protocol over disjoint subregions.
    let window = unsafe { core::slice::from_raw_parts_mut(base, WINDOW_BYTES) };

    let mut station: Option<kore_inet::netinet::dhcp::Lease> = None;
    let witness = {
        let Ok(net) = VifIfnet::attach(window, mac) else {
            let _ = svc.close_vif(0, vif);
            return 0;
        };
        let Ok(mut identity) = Identity::new(
            Ipv4Addr(SLIRP_GUEST),
            SLIRP_PREFIX_LEN,
            Some(Ipv4Addr(SLIRP_GATEWAY)),
        ) else {
            let _ = svc.close_vif(0, vif);
            return 0;
        };
        identity.set_dns(Some(Ipv4Addr(SLIRP_DNS)));
        let mut stack = IpStack::new(net, identity);

        let mut witness = 0u64;
        let mut echoed = false;
        let mut resolved: Option<Ipv4Addr> = None;
        let mut wan_echoed = false;
        let mut asked_name = false;
        // **The station's address, learned rather than assumed.**
        //
        // The identity above is this self-test's own scaffolding — the
        // addresses the gate's SLIRP hands out, so the probes below have
        // somewhere to go on a machine whose DHCP says nothing. It is NOT
        // the row's: a station that guessed its address would be a second
        // claimant on the wire.
        //
        // So DHCP goes first, and a lease — the only thing that makes an
        // address *this machine's* — is published to the row. Until one
        // lands, `InetQuery` answers "no address", which is the honest
        // answer and the one a client must handle.
        let mut leased: Option<kore_inet::netinet::dhcp::Lease> = None;

        // One probe to each target, tracked by the shared echo session rather
        // than by a pair of booleans: the sequence discipline, the deadline
        // and the duplicate rule are then the same ones every other caller of
        // `ping` gets, and a reply this row credits is a reply any of them
        // would have credited.
        //
        // The two sessions carry distinct identifiers so that each absorbs
        // only its own replies — which is what lets both be fed every ICMP
        // event without either one claiming the other's.
        let mut gateway = Ping::new(
            Ipv4Addr(SLIRP_GATEWAY),
            ECHO_IDENT,
            ECHO_INTERVAL_US,
            ECHO_TIMEOUT_US,
            Some(1),
        );
        let mut wan: Option<Ping> = None;

        let mut adopted: Option<kore_inet::netinet::dhcp::Lease> = None;
        let started_us = uptime_ns().map(|ns| ns / 1_000).unwrap_or(0);
        let deadline = uptime_ns().map(|now| now.saturating_add(IP_SELF_TEST_BUDGET_NS));
        for sweep in 0..IP_SELF_TEST_SWEEPS {
            let now_us = uptime_ns()
                .map(|ns| ns / 1_000)
                .unwrap_or((sweep as u64) * 1_000);
            let now_ms = now_us / 1_000;

            // DHCP gets the first part of the budget to itself: an address
            // learned after a probe has gone out would change this station's
            // source mid-flight, and the reply would arrive for an address
            // nobody sent from.
            let dhcp_due = leased.is_none() && now_us.saturating_sub(started_us) < DHCP_BUDGET_US;
            if dhcp_due {
                let _ = stack.drive_dhcp(now_ms, DHCP_XID_SEED);
            }

            // The gateway's hardware address first: an echo with nowhere to
            // send it is not a test of ICMP. `Pending` while ARP resolves is
            // not a transmission, so the probe stays due.
            if !dhcp_due {
                let _ = gateway.send_due(&mut stack, now_us);
            }
            if echoed && !asked_name {
                asked_name = stack
                    .resolve_name(PING_NAME, DNS_XID, DNS_LOCAL_PORT, now_ms)
                    .unwrap_or(false);
            }
            if let Some(session) = wan.as_mut() {
                let _ = session.send_due(&mut stack, now_us);
            }

            // The row's own sweep moves frames between the controller and
            // this vif's rings; the stack's drains what landed.
            let _ = svc.drive();
            stack.input(now_ms, |event| match event {
                Event::Icmp(icmp) => {
                    // The station address is checked here rather than in the
                    // session, which reports who answered and leaves the
                    // judgement to the caller: this witness is specifically
                    // that *these* peers answered.
                    if let Some(ping::Heard::Reply(reply)) = gateway.absorb(icmp, now_us)
                        && reply.from == Ipv4Addr(SLIRP_GATEWAY)
                    {
                        echoed = true;
                    }
                    if let Some(session) = wan.as_mut()
                        && let Some(ping::Heard::Reply(reply)) = session.absorb(icmp, now_us)
                        && Some(reply.from) == resolved
                    {
                        wan_echoed = true;
                    }
                }
                Event::Resolved { addr } => resolved = Some(addr),
                Event::Leased(lease) => leased = Some(lease),
                _ => {}
            });

            // Adopted once, as soon as it lands: this station's own stack
            // starts sending from the leased address, and the row publishes
            // it so `InetQuery` can answer for the machine.
            if let Some(lease) = leased.take_if(|_| adopted.is_none())
                && let Ok(mut id) =
                    Identity::new(lease.addr, lease.prefix_len, lease.gateway)
            {
                id.set_dns(lease.dns);
                stack.set_identity(id);
                adopted = Some(lease);
            }

            // The resolved address is only known after a sweep has run, so the
            // session that probes it is opened here rather than above.
            if let Some(addr) = resolved
                && wan.is_none()
            {
                wan = Some(Ping::new(
                    addr,
                    ECHO_IDENT_WAN,
                    ECHO_INTERVAL_US,
                    ECHO_TIMEOUT_US,
                    Some(1),
                ));
            }

            gateway.expire(now_us);
            if let Some(session) = wan.as_mut() {
                session.expire(now_us);
            }

            if wan_echoed {
                break;
            }
            if let (Some(deadline), Some(now)) = (deadline, uptime_ns())
                && now >= deadline
            {
                break;
            }
            kore_driver::host::relax();
        }

        station = adopted;
        if echoed {
            witness |= INET_ICMP_RT_READY;
        }
        if resolved.is_some() {
            witness |= INET_DNS_READY;
        }
        // The milestone is the conjunction, not the last step alone: an echo
        // answered by an address nobody resolved would be a different claim.
        if echoed && resolved.is_some() && wan_echoed {
            witness |= INET_PING_WAN_READY;
        }
        witness
    };

    let _ = svc.close_vif(0, vif);
    let _ = svc.drive();
    // Published after the vif closes, so the row carries the address whether
    // or not the probes above earned anything: a lease is a fact about the
    // machine, and the witnesses are testimony about a datapath.
    if let Some(lease) = station
        && let Ok(mut id) = Identity::new(lease.addr, lease.prefix_len, lease.gateway)
    {
        id.set_dns(lease.dns);
        svc.configure(id);
    }
    witness
}

/// Push one frame as a client does: into the transmit slot the push count
/// owns, then the record. `false` when the ring has no room — the slot
/// would still be the device's, so nothing is written.
///
/// The ring having room means the record that last used the slot was
/// popped, and the service pops only on completion, so the bytes are free
/// to rewrite exactly when the push can succeed.
fn push_frame(
    tx: &mut spsc::Producer<'_, FrameDesc>,
    frames: &mut [u8],
    pushed: &mut u64,
    frame: &[u8],
) -> bool {
    if tx.capacity_left() == 0 {
        return false;
    }
    let Some(at) = tx_frame_at(tx_slot_of(*pushed)) else {
        return false;
    };
    frames[at..at + frame.len()].copy_from_slice(frame);
    if tx
        .push(FrameDesc::new(frame.len() as u16, EtherType::ARP))
        .is_err()
    {
        return false;
    }
    *pushed += 1;
    true
}

/// Pop one receive record as a client does, and say what it was. `None`
/// when the ring is empty. A record whose slot or length cannot be read is
/// consumed and reported as noise rather than trusted.
fn pop_frame(
    rx: &mut spsc::Consumer<'_, FrameDesc>,
    frames: &[u8],
    popped: &mut u64,
) -> Option<Heard> {
    let desc = rx.pop()?;
    let slot = rx_slot_of(*popped);
    *popped += 1;
    let heard = rx_frame_at(slot)
        .and_then(|at| frames.get(at..at + usize::from(desc.len)))
        .map_or(Heard::Other, arp::heard);
    Some(heard)
}

/// Nanoseconds since boot off the zero-trap clock page, or `None` when this
/// boot stamped no calibration page — in which case the caller falls back to
/// its sweep budget rather than treating a missing clock as time having
/// stopped.
fn uptime_ns() -> Option<u64> {
    kore_rt::clock::uptime()
        .map(kore_abi::clock::Uptime::as_nanos)
        .ok()
        .filter(|ns| *ns != 0)
}

/// **Load the path that just answered**: one full transmit ring of unicast
/// ARP requests to the station that replied, for the address it replied
/// for, pushed as the ring makes room and drained as the replies land — a
/// client streaming, not a test poking.
///
/// What this proves that one frame cannot: descriptors batched behind one
/// doorbell per sweep, completions reaped in ring order across the ring's
/// wrap, the receive ring refilled under a burst, moderated interrupts
/// reaping whole batches, and the client's slot discipline (a slot is
/// rewritten only once its record has been popped, which only a completion
/// does). Returns whether the whole burst went out and every reply — a
/// reply *from the responder*, not any frame — came back; the caller
/// judges the witness from the counters.
#[allow(clippy::too_many_arguments)] // the client's whole state, borrowed
fn load_burst(
    svc: &mut NetService<MetalHost>,
    mac: MacAddr,
    responder: MacAddr,
    ip: [u8; 4],
    tx: &mut spsc::Producer<'_, FrameDesc>,
    rx: &mut spsc::Consumer<'_, FrameDesc>,
    frames: &mut [u8],
    pushed: &mut u64,
    popped: &mut u64,
) -> bool {
    let request = arp::request(mac, SLIRP_GUEST, responder, ip);
    let deadline = uptime_ns().map(|now| now.saturating_add(SELF_TEST_BUDGET_NS));
    let mut sent = 0u32;
    let mut replies = 0u32;
    for _ in 0..SELF_TEST_SWEEPS {
        while sent < LOAD_BURST && push_frame(tx, frames, pushed, &request) {
            sent += 1;
        }
        let _ = svc.drive();
        while let Some(heard) = pop_frame(rx, frames, popped) {
            if matches!(heard, Heard::Reply { station, .. } if station == responder) {
                replies += 1;
            }
        }
        // RX is polled after TX completions in a sweep. The final reply
        // can therefore arrive before its transmit completion is reaped.
        // A client slot is released only by that reap: wait for the ring
        // to drain before the caller judges the successful-send counters.
        if sent == LOAD_BURST && replies >= LOAD_BURST && tx.capacity_left() == tx.capacity() {
            break;
        }
        if let (Some(deadline), Some(now)) = (deadline, uptime_ns())
            && now >= deadline
        {
            break;
        }
        kore_driver::host::relax();
    }
    sent == LOAD_BURST && replies >= LOAD_BURST && tx.capacity_left() == tx.capacity()
}
