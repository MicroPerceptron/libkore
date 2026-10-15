//! **The doorbell star** (`res/SPMD.md` §3) — the wake fabric a region
//! provisions so its barrier's park tier works at all.
//!
//! A parked futex waiter is woken by a notice into its doorbell, and a
//! doorbell is a capability: the waker must HOLD a send into the
//! sleeper's channel.  Workers founded bare have neither — no recv to
//! park on, no sends to wake with — which is why the pre-star barrier
//! could only spin until preempted.  A full mesh would cost N² sends;
//! the star costs 2N + 1:
//!
//! - every worker is endowed **its own recv** (`role::DOORBELL, 0` —
//!   the park tier engages) and **one send into the hub**
//!   (`role::DOORBELL,` [`RELAY`](crate::futex::RELAY));
//! - the **hub** — one extra gate-held thread — is endowed a send into
//!   every sibling, named by band index exactly as
//!   [`futex::kick`](crate::futex) looks them up, so a hub wake IS an
//!   ordinary `wake_all` from the one thread that can reach everyone;
//! - the **parent** keeps the minted send of every channel, so the gate
//!   release kicks every parked child directly — the launch never
//!   depends on the hub it is launching.
//!
//! A leader's `wake_all` thus reaches a parked member in two hops:
//! leader → hub (the futex relay fallback) → member.  The hub itself
//! parks on the region's gate word — any notice wakes it, it kicks all
//! peers unconditionally (spurious wakes are every waiter's documented
//! normal), and its [`HUB_SLICE`] bounds a lost notice at the cost the
//! pre-star world paid every time.
//!
//! # Degrade, never refuse
//!
//! Provisioning is all-or-nothing but the launch is not: no
//! `CHANNEL_SPACE` grant, a refused mint, or a failed hub founding all
//! degrade — no star (exactly the pre-star behavior, parks bounded by
//! the futex's own slice), never a refused launch.  Teardown retires
//! the channel roots, and T6 takes every derived end — the workers'
//! recvs and relay sends — with them.

use core::time::Duration;

#[cfg(target_os = "none")]
use core::sync::atomic::Ordering;

/// The hub's sentinel worker id — never a dense region index, so a
/// worker body that switches on its id cannot mistake the relay for a
/// tile peer.
#[cfg_attr(not(target_os = "none"), allow(dead_code))]
pub(crate) const HUB: u32 = crate::spmd::RELAY_HUB;

/// The hub's self-heal slice: a parked hub re-checks the gate at this
/// cadence even if every notice were lost — the quantum-scale bound the
/// pre-star world lived under, now the fallback rather than the path.
#[cfg_attr(not(target_os = "none"), allow(dead_code))]
const HUB_SLICE: Duration = Duration::from_millis(64);

/// The `(role::DOORBELL, index)` a hub send for band `b` is named under
/// — the one arithmetic the star and the futex's `kick` must agree on
/// (`PEER_BASE + band`), kept pure so the host suite can pin it.
// Host builds keep the declaration (and its pin test); only the metal
// star consumes it.
#[cfg_attr(not(target_os = "none"), allow(dead_code))]
pub(crate) const fn hub_peer_index(band: u32) -> u16 {
    crate::futex::PEER_BASE + band as u16
}

/// The star a launched region carries — absent when provisioning
/// degraded (or off metal, where nothing parks).  Affine: [`finish`]
/// (metal) is the one teardown, and it consumes the value.
///
/// [`finish`]: Star::finish
pub(crate) struct Star {
    #[cfg(target_os = "none")]
    hub: Option<crate::thread::Thread>,
    /// Borrowed name of the affine join landing owned by the enclosing
    /// `Spmd`. `Star` never releases it and cannot outlive that region.
    #[cfg(target_os = "none")]
    join_report: kore_abi::session::SlotRef,
    #[cfg(target_os = "none")]
    wires: Wires,
}

/// The minted forest: the per-worker channels (indexed by DENSE worker
/// id), the hub channel, and the claimed space witness.
#[cfg(target_os = "none")]
struct Wires {
    /// Held for the region's life; a witness, not a reservation — it
    /// drops at finish (retiring it would cut every channel the process
    /// ever mints, not just this region's).
    _space: crate::cap::ChannelSpaceToken,
    roots: [Option<crate::cap::ChannelRoot>; crate::thread::MAX_THREADS as usize],
    parent_sends: [Option<crate::cap::ChanSendToken>; crate::thread::MAX_THREADS as usize],
    hub_root: crate::cap::ChannelRoot,
    hub_send: crate::cap::ChanSendToken,
    // The hub and a resident submitter are mutually exclusive consumers.
    hub_recv: Option<crate::cap::ChanRecvToken>,
    completion: Option<
        core::cell::RefCell<crate::chan::NoticeRx<'static, crate::transport::TrapTransport>>,
    >,
    /// The founder's own doorbell send — its self-kick — when the founder
    /// is thread 0, endowed to every worker as peer 0 of the kick mesh so a
    /// resident job's last worker can kick the launcher.
    founder_kick: Option<kore_abi::session::SlotRef>,
    workers: usize,
}

/// The staged half — minted BEFORE any founding, consumed into a
/// [`Star`] once the hub ticket exists (or into a hubless star on the
/// ruled degrade).
#[cfg(target_os = "none")]
pub(crate) struct StarPlan {
    wires: Wires,
}

/// Mint the region's whole forest, all-or-nothing: `workers` channels
/// plus the hub's.  `None` — no `CHANNEL_SPACE` boot grant, dynamic
/// arena slots exhausted, a refused mint — releases everything minted
/// so far and the launch proceeds starless.
#[cfg(target_os = "none")]
pub(crate) fn provision(workers: usize) -> Option<StarPlan> {
    use crate::cap::verbs;
    if workers == 0 || workers > crate::thread::MAX_THREADS as usize {
        return None;
    }
    let rt = crate::entry::thread_context()?;
    let space_slot = rt
        .bootstrap()
        .handle(kore_abi::task::role::CHANNEL_SPACE, 0)?;
    let space = verbs::claim_channel_space(space_slot)?;
    let mut roots: [Option<crate::cap::ChannelRoot>; crate::thread::MAX_THREADS as usize] =
        [const { None }; crate::thread::MAX_THREADS as usize];
    let mut parent_sends: [Option<crate::cap::ChanSendToken>; crate::thread::MAX_THREADS as usize] =
        [const { None }; crate::thread::MAX_THREADS as usize];
    let mut minted = |roots: &mut [Option<crate::cap::ChannelRoot>],
                      sends: &mut [Option<crate::cap::ChanSendToken>]| {
        // All-or-nothing rollback: retire the leaves, then the pivots.
        for s in sends.iter_mut() {
            if let Some(s) = s.take() {
                let _ = s.try_retire();
            }
        }
        for r in roots.iter_mut() {
            if let Some(r) = r.take() {
                let _ = r.try_retire();
            }
        }
    };
    for w in 0..workers {
        let Ok(pair) = verbs::mint_channel(&space) else {
            minted(&mut roots, &mut parent_sends);
            return None;
        };
        // The parent's own recv is surplus — the WORKER's recv is a
        // fresh root-derive at founding — so it retires now rather than
        // sitting as a second consumer on the child's doorbell.
        let _ = pair.recv.try_retire();
        roots[w] = Some(pair.root);
        parent_sends[w] = Some(pair.send);
    }
    let Ok(hub_pair) = verbs::mint_channel(&space) else {
        minted(&mut roots, &mut parent_sends);
        return None;
    };
    // The founder joins the mesh as peer 0 through its self-kick send
    // (`DOORBELL@PEER_BASE`).  Only thread 0 holds a send into its OWN
    // doorbell — a spawned thread's `DOORBELL@PEER_BASE` names thread 0's —
    // so a region founded elsewhere carries no founder kick and its
    // launcher rides the park's slice (`res/SPMD.md` §14.2).
    let founder_kick = if crate::entry::geometry::shard_index() == Some(0) {
        rt.bootstrap()
            .handle(kore_abi::task::role::DOORBELL, crate::futex::PEER_BASE)
    } else {
        None
    };

    Some(StarPlan {
        wires: Wires {
            _space: space,
            roots,
            parent_sends,
            hub_root: hub_pair.root,
            hub_send: hub_pair.send,
            hub_recv: Some(hub_pair.recv),
            completion: None,
            founder_kick,
            workers,
        },
    })
}

/// Descriptor invocations one meshed worker costs its gang session in a
/// chunk of `chunk`: two views, the boot delegate and the Found, plus its
/// endowment of `chunk + 2` caps (own recv, hub send, the founder's send,
/// and a peer send per chunk sibling).
const fn meshed_worker_invocations(chunk: usize) -> usize {
    4 + chunk + 2
}

/// The widest chunk one FULLY MESHED gang session can carry — derived
/// from the descriptor's bounds, never chosen: a chunk of `n` costs
/// `n·(n + 6)` invocations against [`MAX_INVOCATIONS`], and no session
/// founds more than [`MAX_SESSION_FOUNDS`] (seven, at today's 107 and 12).
/// A wider region founds in several sessions (see [`spmd::launch`]): the
/// mesh spans a session's chunk, and a cross-chunk waiter is the hub's to
/// reach — the relay hop the mesh exists to keep OFF the common release
/// path.
///
/// [`MAX_INVOCATIONS`]: kore_abi::session::MAX_INVOCATIONS
/// [`MAX_SESSION_FOUNDS`]: kore_abi::session::MAX_SESSION_FOUNDS
/// [`spmd::launch`]: crate::spmd
pub(crate) const MESH_CEILING: usize = {
    let mut chunk = kore_abi::session::MAX_SESSION_FOUNDS;
    while chunk > 1
        && chunk * meshed_worker_invocations(chunk) > kore_abi::session::MAX_INVOCATIONS
    {
        chunk -= 1;
    }
    chunk
};

#[cfg(target_os = "none")]
impl StarPlan {
    /// Worker `w`'s endowments — **the intra-chunk kick mesh**: its own
    /// recv (→ `DOORBELL@0`, which the stamped landing name at
    /// `DOORBELL@1` completes into a working park), the hub send
    /// (→ `DOORBELL@`[`RELAY`]), and one send into every CHUNK
    /// SIBLING's doorbell, named `DOORBELL@PEER_BASE+band` — the exact
    /// `(role, index)` the futex peer lookup computes, so a barrier
    /// leader's `wake_all` reaches a parked sibling in ONE hop.
    ///
    /// The two-hop relay (leader → hub → member) is real but slow on
    /// metal: the hub is unpinned and parked, so every release of a
    /// parked member paid two wake latencies — the w≤4 terminal bench
    /// regressed from µs to ~½·PARK_SLICE averages exactly this way.
    /// The hub now serves only what the mesh cannot name: cross-chunk
    /// waiters and sends the caller lacks.
    ///
    /// `bands` pairs each chunk member's DENSE worker id with its
    /// claimed band slot (known only post-claim, which is why the
    /// caller builds endows after its claims loop, mirroring
    /// [`hub_endow`](Self::hub_endow)).  Writes into `out`, returns the
    /// filled count.
    ///
    /// [`RELAY`]: crate::futex::RELAY
    pub(crate) fn worker_endow_mesh(
        &self,
        w: usize,
        bands: &[(usize, u32)],
        out: &mut [crate::thread::WorkerCap],
    ) -> Option<usize> {
        let root = self.wires.roots.get(w)?.as_ref()?;
        if out.len() < 2 + bands.len() {
            return None;
        }
        out[0] = crate::thread::WorkerCap {
            role: kore_abi::task::role::DOORBELL,
            index: 0,
            subject: root.slot(),
            kind: kore_abi::cap::ResourceKind::ChanRecv as u16 as u8,
            rights: crate::cap::CHANNEL_ENDPOINT_RIGHTS,
        };
        out[1] = crate::thread::WorkerCap {
            role: kore_abi::task::role::DOORBELL,
            index: crate::futex::RELAY,
            subject: self.wires.hub_root.slot(),
            kind: kore_abi::cap::ResourceKind::ChanSend as u16 as u8,
            rights: crate::cap::CHANNEL_ENDPOINT_RIGHTS,
        };
        let mut n = 2;
        // The founder as peer 0 (`DOORBELL@PEER_BASE`): a send derived from
        // its self-kick, with the notice rights that send carries — the
        // same derivation `spawn_with_doorbell` makes for `kick_parent`.
        if let Some(kick) = self.wires.founder_kick {
            out[n] = crate::thread::WorkerCap {
                role: kore_abi::task::role::DOORBELL,
                index: crate::futex::PEER_BASE,
                subject: kick,
                kind: kore_abi::cap::ResourceKind::ChanSend as u16 as u8,
                rights: crate::cap::verbs::NOTICE_RIGHTS,
            };
            n += 1;
        }
        for &(j, band) in bands {
            if j == w {
                continue;
            }
            let peer_root = self.wires.roots.get(j)?.as_ref()?;
            out[n] = crate::thread::WorkerCap {
                role: kore_abi::task::role::DOORBELL,
                index: hub_peer_index(band),
                subject: peer_root.slot(),
                kind: kore_abi::cap::ResourceKind::ChanSend as u16 as u8,
                rights: crate::cap::CHANNEL_ENDPOINT_RIGHTS,
            };
            n += 1;
        }
        Some(n)
    }

    /// The hub's endowments: its own recv, then one send per founded
    /// worker named by the worker's BAND index (`bands` pairs the dense
    /// worker id with its band slot) — the exact `(role, index)` the
    /// futex peer lookup computes, which is what makes a hub wake an
    /// ordinary `kick`.
    pub(crate) fn hub_endow(
        &self,
        bands: &[(usize, u32)],
        out: &mut [crate::thread::WorkerCap],
    ) -> Option<usize> {
        if out.len() < 1 + bands.len() {
            return None;
        }
        out[0] = crate::thread::WorkerCap {
            role: kore_abi::task::role::DOORBELL,
            index: 0,
            subject: self.wires.hub_root.slot(),
            kind: kore_abi::cap::ResourceKind::ChanRecv as u16 as u8,
            rights: crate::cap::CHANNEL_ENDPOINT_RIGHTS,
        };
        let mut n = 1;
        for &(w, band) in bands {
            let root = self.wires.roots.get(w)?.as_ref()?;
            out[n] = crate::thread::WorkerCap {
                role: kore_abi::task::role::DOORBELL,
                index: hub_peer_index(band),
                subject: root.slot(),
                kind: kore_abi::cap::ResourceKind::ChanSend as u16 as u8,
                rights: crate::cap::CHANNEL_ENDPOINT_RIGHTS,
            };
            n += 1;
        }
        Some(n)
    }

    /// Seal the plan into the region's star, with (or, on the ruled
    /// degrade, without) its hub ticket.
    pub(crate) fn into_star(
        mut self,
        hub: Option<crate::thread::Thread>,
        join_report: kore_abi::session::SlotRef,
    ) -> Star {
        if hub.is_some() {
            if let Some(recv) = self.wires.hub_recv.take() {
                let _ = recv.try_retire();
            }
        } else if let Some(recv) = self.wires.hub_recv.as_ref() {
            self.wires.completion = Some(core::cell::RefCell::new(crate::chan::NoticeRx::new(
                crate::entry::transport(),
                recv.slot(),
            )));
        }
        Star {
            hub,
            join_report,
            wires: self.wires,
        }
    }
}

#[cfg(target_os = "none")]
impl Star {
    /// One observer owns the alternate post across its wait. Concurrent
    /// observers fall back to the bounded futex wait; cancellation leaves the
    /// standing landing with this region, never donated to another channel.
    pub(crate) fn completion_receiver(
        &self,
    ) -> Option<
        core::cell::RefMut<'_, crate::chan::NoticeRx<'static, crate::transport::TrapTransport>>,
    > {
        self.wires.completion.as_ref()?.try_borrow_mut().ok()
    }

    /// Kick every worker (and the hub) directly over the parent-held
    /// sends — the gate release, which must never depend on the hub it
    /// just launched.
    pub(crate) fn kick_all(&self) {
        // One session for the whole gang: n notices, one crossing.  The
        // terminal bench priced the one-post-per-worker form at ~40 µs a
        // worker, the workers waking one after another in that rhythm.  A
        // session is atomic, so a send whose post is not standing (that
        // worker is already awake and its post consumed) refuses the whole
        // commit; then the posts go one by one, as before, so every parked
        // worker still gets its kick.  The word is the truth either way.
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};
        let sends = self.wires.parent_sends.iter().flatten();
        let mut batch = SessionBuilder::new();
        let mut staged = 0usize;
        let mut live = 0usize;
        for s in sends.clone() {
            live += 1;
            if batch
                .notice(
                    s.slot(),
                    crate::futex::COOKIE_FUTEX,
                    u64::from(crate::futex::RELAY),
                )
                .is_ok()
            {
                staged += 1;
            }
        }
        let batched = live != 0
            && staged == live
            && TrapTransport::new()
                .commit_now(&batch)
                .is_ok_and(|c| c.status() == kore_abi::session::Status::Ok);
        if !batched {
            for s in sends {
                let _ = crate::cap::verbs::post_notice(
                    s.slot(),
                    crate::futex::COOKIE_FUTEX,
                    u64::from(crate::futex::RELAY),
                );
            }
        }
        // The hub post reaches a consumer only when a hub thread was spawned
        // (it holds the sole recv). A hubless resident retains that recv for
        // reverse completion; kicking the submitter on its own submit is wasteful.
        if self.hub.is_some() {
            let _ = crate::cap::verbs::post_notice(
                self.wires.hub_send.slot(),
                crate::futex::COOKIE_FUTEX,
                u64::from(crate::futex::RELAY),
            );
        }
    }

    /// The one teardown: wake the hub off its park, join it to a terminal
    /// observation, and retire the forest — sends first
    /// (leaves), then the roots, whose T6 cascade takes every derived
    /// end (the workers' recvs and relay sends) with them.  The caller
    /// has already flipped the gate off `RUN`, so the hub's loop is
    /// exiting; a worker that parks during the wind-down is bounded by
    /// the futex slice, never stranded.
    ///
    /// A failed lifecycle observation returns the complete affine star and
    /// its cause. The channel forest cannot be retired until the hub is known
    /// terminal: doing so would strand a live relay, while retrying a
    /// permanent transport/refusal error internally would wedge the caller.
    /// The enclosing [`Spmd`](crate::spmd::Spmd) retains this owner and exposes
    /// the same explicit retry contract as a worker join.
    pub(crate) fn finish(mut self) -> Result<(), (Self, crate::cap::TaskWaitError)> {
        // Wake the hub off its park so its loop can see the gate and exit —
        // only if one was spawned; the resident's receiver belongs to its
        // submitter and requires no retirement wake.
        if self.hub.is_some() {
            let _ = crate::cap::verbs::post_notice(
                self.wires.hub_send.slot(),
                crate::futex::COOKIE_FUTEX,
                0,
            );
        }
        if let Some(hub) = self.hub.take() {
            if let Err(error) = crate::thread::join_reserved_at(hub, self.join_report) {
                let (hub, cause) = error.into_parts();
                self.hub = Some(hub);
                return Err((self, cause));
            }
        }
        let Star {
            hub: _,
            join_report: _,
            wires,
        } = self;
        let Wires {
            _space,
            mut roots,
            mut parent_sends,
            hub_root,
            hub_send,
            hub_recv,
            completion,
            // A borrowed name in the founder's own table: nothing to retire.
            founder_kick: _,
            workers: _,
        } = wires;
        for s in parent_sends.iter_mut() {
            if let Some(s) = s.take() {
                let _ = s.try_retire();
            }
        }
        let _ = hub_send.try_retire();
        for r in roots.iter_mut() {
            if let Some(r) = r.take() {
                let _ = r.try_retire();
            }
        }
        if let Some(recv) = hub_recv {
            let _ = recv.try_retire();
        }
        if hub_root.try_retire().is_ok() {
            if let Some(rx) = completion {
                rx.into_inner().close_after_root_revoked();
            }
        }
        // An unproven root close leaves any armed landing quarantined by
        // NoticeRx/ReturnedCapability, just like the existing channel owners.
        Ok(())
    }
}

#[cfg(not(target_os = "none"))]
impl Star {
    pub(crate) fn completion_receiver(
        &self,
    ) -> Option<
        core::cell::RefMut<'_, crate::chan::NoticeRx<'static, crate::transport::TrapTransport>>,
    > {
        None
    }

    /// Off metal nothing was provisioned, so the teardown is nothing too
    /// — kept so the join path reads identically on every target.
    pub(crate) fn finish(self) -> Result<(), (Self, crate::cap::TaskWaitError)> {
        Ok(())
    }

    /// Off metal there are no sends to post — kept so the resident
    /// region's submit/retire read identically on every target
    /// ([`ResidentRegion`](crate::spmd::ResidentRegion), R9).
    pub(crate) fn kick_all(&self) {}
}

/// The hub body: park on the region's gate word, kick every held peer
/// on every wake, exit when the gate leaves `RUN`.  Entered through the
/// launch gate like any worker (an aborted launch exits it cleanly);
/// founded WITHOUT `SPMD_WORKER` — a parked relay must not occupy a
/// class-best pin nor count in a reservation batch.
#[cfg(target_os = "none")]
pub(crate) fn hub_main(ctx: &'static crate::spmd::SpmdCtx, _hub: u32) {
    while ctx.gate_word().load(Ordering::SeqCst) == crate::spmd::RUN {
        // Kick first: a worker may have parked between the previous
        // sweep and this wake, and a spurious extra notice is every
        // waiter's documented normal.
        kick_peers();
        let _ = crate::futex::wait_quiet(ctx.gate_word(), crate::spmd::RUN, Some(HUB_SLICE));
    }
    // One last sweep: a worker that parked while the region wound down
    // must still observe the flip.
    kick_peers();
}

/// Post one notice into every sibling doorbell this thread's boot block
/// names — the hub's whole vocabulary.
#[cfg(target_os = "none")]
fn kick_peers() {
    let Some(rt) = crate::entry::thread_context() else {
        return;
    };
    for band in 0..crate::thread::MAX_THREADS {
        if let Some(send) = rt
            .bootstrap()
            .handle(kore_abi::task::role::DOORBELL, hub_peer_index(band))
        {
            let _ =
                crate::cap::verbs::post_notice(send, crate::futex::COOKIE_FUTEX, u64::from(band));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The hub's send naming and the futex's peer lookup are one
    /// arithmetic; the relay slot sits exactly one past the last band
    /// peer.
    #[test]
    fn hub_peer_naming_matches_the_futex_mesh() {
        assert_eq!(hub_peer_index(0), 2, "band 0 is the first mesh peer");
        assert_eq!(
            hub_peer_index(crate::thread::MAX_THREADS - 1) + 1,
            crate::futex::RELAY,
            "the relay slot is one past the last band peer"
        );
        assert_eq!(HUB, u32::MAX, "the hub id can never be a dense worker id");
    }

    /// The ceiling is the descriptor's arithmetic: the widest meshed
    /// chunk fits one session, and one worker more would not.
    #[test]
    fn the_mesh_ceiling_is_the_widest_session_the_descriptor_carries() {
        use kore_abi::session::{MAX_INVOCATIONS, MAX_SESSION_FOUNDS};
        let cost = |chunk: usize| chunk * meshed_worker_invocations(chunk);
        assert!((1..=MAX_SESSION_FOUNDS).contains(&MESH_CEILING));
        assert!(cost(MESH_CEILING) <= MAX_INVOCATIONS);
        assert!(MESH_CEILING == MAX_SESSION_FOUNDS || cost(MESH_CEILING + 1) > MAX_INVOCATIONS);
    }
}
