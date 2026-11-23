//! **The row's own mint** (`res/PEER_LINK.md` P1) — bounded, affine assembly
//! of one frame client's serving pair.
//!
//! # Why this exists at all
//!
//! A frame client cannot be founder-wired. Both static routes are refused at
//! infrastructure build, and neither refusal is an oversight:
//!
//! - `shell_client = true` on this row trips the core-topology rule in
//!   `services/exec/src/founder/mod.rs` — a chained driver row may not also
//!   carry a core role's authority, because the uniform claimant round owns
//!   its wiring and a double role would need two rounds' worth;
//! - naming this row in another row's `clients` list trips
//!   `declared_client_topology_supported`, which refuses any *chained*
//!   provider.
//!
//! So the only face a client can hold on this row is one the row mints at
//! runtime, onto pair 0, which is exactly what `wire.rs`'s `NET_SERVE_PAIRS`
//! note means by "needs no founder edit and no change to this number".
//!
//! # What is this row's, and what is the runtime's
//!
//! Almost none of this is this row's. The assembly, the generational
//! identity, the mapping and the bank are [`kore_rt`]'s — the same engine
//! `services/accel` drives, instantiated here rather than reimplemented, so
//! a bundle that assembles correctly there assembles correctly here. This
//! module owns three things only: who may ask (the administration gate), how
//! a runtime refusal reads as a [`NetRefusal`], and the fact that a verdict
//! is collected separately from the request that earned it.
//!
//! # The two-step, and why it is two
//!
//! The mint completes *after* the reply to the bundle's last component, so
//! the verdict cannot ride that reply. [`NetRequest::TakePeerAdmission`]
//! collects it. A caller that conflated the two would read `PeerAccepted` as
//! "I have a face" and use an identity that does not exist yet.

use kore_rt::service::{PeerClose, Serve};
use kore_rt::svc::{
    PeerAdmissionRefusal, PeerBundleAssembler, PeerBundlePart, PeerId, PeerMapPlan,
};

use super::{NetRefusal as E, NetRequest as Q, NetResponse as R};

/// The VA band at the base of this row's TRANSIENT reservation that minted
/// pair windows are mapped into.
///
/// Geometry reserves address space only; the *backing* for each window comes
/// from the client that asked, which is what keeps a client's ring its own
/// lease rather than a charge against this row's pool.
const RING_RESERVE: u64 = 2 * 1024 * 1024;

/// One minted pair's ring window — [`RING_RESERVE`] split evenly across the
/// runtime's pair roster.
///
/// It therefore **shrinks when that roster grows**, exactly as the accel
/// row's does, and for the same reason: admission requires
/// `ring_bytes <= plan.stride`. A client leases this constant and never a
/// literal of its own, so a roster change refuses at the lease rather than
/// silently mapping short.
pub const PEER_RING_BYTES: u64 = RING_RESERVE / super::PAIRS as u64;

const _: () = {
    // `PeerMapPlan::address` refuses a stride that is not page aligned.
    assert!(
        PEER_RING_BYTES != 0 && PEER_RING_BYTES % 4096 == 0,
        "a peer ring window must be a non-zero page multiple"
    );
};

/// The map plan for a row whose TRANSIENT reservation begins at `base`.
pub(super) fn peer_map(base: u64) -> PeerMapPlan {
    PeerMapPlan {
        base,
        stride: PEER_RING_BYTES,
    }
}

/// This row's half of peer admission.
pub(super) struct Peers {
    assembler: PeerBundleAssembler,
    /// `None` until the row's transient geometry is known; admission refuses
    /// [`NetRefusal::Unsupported`] until then rather than inventing a base.
    pub map: Option<PeerMapPlan>,
    /// A complete bundle is with the runtime and its verdict has not landed.
    ///
    /// `pub(super)` so the row's own tests can stand at the one boundary
    /// they cannot otherwise reach — the runtime's callback, which only the
    /// host drives.
    pub(super) pending: bool,
    /// A landed verdict nobody has collected.
    result: Option<Result<PeerId, E>>,
}

impl Peers {
    pub const fn new() -> Self {
        Self {
            assembler: PeerBundleAssembler::new(crate::wire::NET_REPLY_GRANTS, true),
            map: None,
            pending: false,
            result: None,
        }
    }

    /// Record the runtime's verdict.
    ///
    /// `accepted` is the host's own judgement on the identity it chose — a
    /// peer seated in a cell a fixed pair already occupies is refused here,
    /// because `pair_index` would otherwise fold the two onto one session.
    ///
    /// Returns whether the admission stands, which is what the caller uses to
    /// decide whether to bind the face.
    pub fn admitted(
        &mut self,
        result: Result<PeerId, PeerAdmissionRefusal>,
        accepted: bool,
    ) -> bool {
        if !self.pending {
            return false;
        }
        self.pending = false;
        self.result = Some(match result {
            Ok(peer) if accepted => Ok(peer),
            Ok(_) => Err(E::Busy),
            Err(PeerAdmissionRefusal::Capacity) => Err(E::Exhausted),
            Err(PeerAdmissionRefusal::NoScratchSlots) => Err(E::Exhausted),
            Err(PeerAdmissionRefusal::MapFailed) => Err(E::NoMapping),
            Err(_) => Err(E::BadPeerBundle),
        });
        accepted
    }

    /// Serve one peer verb, or `None` for a verb that is not this module's.
    pub fn handle(&mut self, req: &Q, sv: &mut Serve<'_, R>) -> Option<R> {
        Some(match req {
            Q::AdmitPeer { part, announced } => {
                if !crate::wire::is_admin_pair(sv.via()) {
                    return Some(R::Refused(E::PeerDenied));
                }
                // A `Ring` is the boundary between bundles, so one arriving
                // while the assembler still owns components proves the
                // previous assembly was abandoned — its asker spent its
                // bounded re-ask and gave up. Retire what it left, and refuse
                // the replacement while any of those owners is still closing.
                //
                // Without this the leftovers are permanent: the next asker's
                // matching part answers `Duplicate`, the bundle never
                // completes, no verdict is ever published, and every later
                // client that needs a face on this row refuses for the rest
                // of the boot.
                //
                // Behind the administration gate and checked before anything
                // else touches the assembler: a via this row does not trust
                // to assemble must never retire what is being assembled.
                if matches!(part, PeerBundlePart::Ring)
                    && !self.assembler.is_empty()
                    && !self.assembler.retire_incomplete()
                {
                    return Some(R::Refused(E::Busy));
                }
                if self.pending || self.result.is_some() {
                    return Some(R::Refused(E::Busy));
                }
                let Some(map) = self.map else {
                    return Some(R::Refused(E::Unsupported));
                };
                let Some(mut input) = sv.returned_input(*announced) else {
                    return Some(R::Refused(E::BadPeerBundle));
                };
                let Some(cap) = input.capability() else {
                    return Some(R::Refused(E::BadPeerBundle));
                };
                // Validated before the owner is taken, so a refusal leaves
                // the capability with its sender rather than consuming it.
                if self.assembler.accepts(*part, cap).is_err() {
                    return Some(R::Refused(E::BadPeerBundle));
                }
                let owner = input.take().expect("validated owner remains present");
                if let Some(bundle) = self
                    .assembler
                    .push(*part, owner)
                    .expect("same synchronous validation")
                {
                    self.pending = true;
                    sv.admit_peer(bundle, map);
                }
                R::PeerAccepted
            }
            Q::TakePeerAdmission => {
                if !crate::wire::is_admin_pair(sv.via()) {
                    return Some(R::Refused(E::PeerDenied));
                }
                match self.result.take() {
                    Some(Ok(peer)) => R::PeerAdmitted { peer },
                    Some(Err(error)) => R::Refused(error),
                    // No verdict, and the assembler still owns components:
                    // this bundle never completed, so `admit_peer` was never
                    // called and no verdict can EVER be published. `Busy`
                    // here would be a lie the asker waits on forever.
                    None if !self.assembler.is_empty() => {
                        let _ = self.assembler.retire_incomplete();
                        R::Refused(E::BadPeerBundle)
                    }
                    // An EMPTY assembler means the bundle completed and the
                    // host has not drained its admission action yet: a
                    // verdict IS coming, and settling here would orphan the
                    // face it is about to admit.
                    None => R::Refused(E::Busy),
                }
            }
            Q::ClosePeer => {
                // Only a minted face may retire itself. A fixed pair is the
                // founder's, and this row closing one would retire authority
                // it never minted.
                if PeerId::from_arrival(sv.via()).is_none() {
                    return Some(R::Refused(E::PeerDenied));
                }
                sv.close_peer(PeerClose::Orderly);
                R::PeerClosing
            }
            _ => return None,
        })
    }
}
