//! **The install seam** — where a minted authority stops being a description
//! and becomes the predicate the classifier runs.
//!
//! [`FlowCap::frame_filter`](super::FlowCap::frame_filter) produces the
//! [`FrameFilter`] that *enforces* a flow. Producing it is not installing it:
//! until the frame row's classifier is running that filter on the holder's
//! vif, the capability is a claim about what the holder may receive and
//! nothing is stopping another client's datagrams from arriving beside it.
//! This trait is the step that closes that gap, and the row calls it as part
//! of the mint rather than after it.
//!
//! # Why a seam and not a call
//!
//! `crates/inet` names no service, for the same reason it names no driver
//! (`res/INET_STACK.md` §2.1). A founded row installs over a client face onto
//! `services/net`; a host test installs into a recording double; a client that
//! happens to hold both halves installs directly. The engine is written once,
//! against this contract, so what a test proves is what a founded row does —
//! the `StorageProvider` split `services/dagfs` already uses for exactly this
//! reason.
//!
//! # Why it is async
//!
//! The one production impl reaches another service over a ring. An impl that
//! must park cannot be hosted by a synchronous contract, and discovering that
//! after two impls and a test suite exist is how a committed seam gets broken.
//! The host double never actually parks — its futures are `Ready` on first
//! poll — but the seam is shaped for the one that will
//! (`services/dagfs/src/provider/mod.rs` states the same rule for storage).
//!
//! # The refusal vocabulary is the frame row's, lifted
//!
//! Each variant names the specific power or resource that was missing, and
//! [`InstallRefusal::Policy`] carries the frame row's own per-dimension
//! [`PolicyRefusal`] rather than collapsing it: an operator who cannot tell a
//! refused EtherType from a refused peer endpoint cannot tell which grant to
//! widen.

use kore_driver::net::policy::PolicyRefusal;
use kore_driver::net::{FrameFilter, VifId};

/// Why a flow's enforcement could not be installed.
///
/// Deliberately **not** a transport vocabulary: a torn ring and a refused
/// EtherType are different facts, and a caller that cannot tell them apart
/// cannot tell a misconfiguration from a broken row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstallRefusal {
    /// No face onto the frame row was wired, so there is nothing to install
    /// the filter on.
    ///
    /// The composition's fault, never the caller's — and it is named rather
    /// than folded into a generic failure because it is the one refusal an
    /// operator fixes in the boot manifest rather than in a policy.
    NoFrameFace,
    /// The frame row's policy refused the filter, naming the dimension that
    /// exceeded.
    Policy(PolicyRefusal),
    /// The identifier names no live vif.
    NoVif,
    /// The vif is live but belongs to another holder.
    NotOwner,
    /// The face's transport failed, so whether the filter was installed is
    /// unknown — which is why it is distinct from every refusal above, all of
    /// which mean "nothing changed".
    Lost,
}

/// **Install the receive filter that enforces a flow.**
///
/// One method, because releasing an authority *is* installing the empty
/// predicate ([`FrameFilter::CLOSED`]) — there is no second operation to get
/// wrong, and no impl can release in a way that differs from how another one
/// does.
#[allow(async_fn_in_trait)] // the engine is generic over the impl, never `dyn`
pub trait FlowInstall {
    /// Make `filter` the receive predicate on `vif`.
    ///
    /// Must be **total**: on `Ok` the filter is installed and from that moment
    /// the classifier runs it; on `Err` nothing changed and the vif keeps the
    /// filter it had — except for [`InstallRefusal::Lost`], which says the
    /// answer is unknown and is why that variant exists.
    async fn install(&mut self, vif: VifId, filter: FrameFilter) -> Result<(), InstallRefusal>;

    /// Whether `vif` is this caller's to act on.
    ///
    /// Asked **before** a close touches anything, and that ordering is the
    /// whole point: a close mutates two tables — the port table and the
    /// installed filter — and a caller naming a vif it does not hold must
    /// change neither. [`install`](Self::install) answers the same question,
    /// but only once the ports are already gone.
    ///
    /// Not async: ownership is a table the impl already holds, never a round
    /// trip. An impl with no notion of a caller answers for the face itself.
    fn held(&self, vif: VifId) -> Result<(), InstallRefusal>;

    /// Stop enforcing anything on `vif`.
    ///
    /// Not a hook an impl overrides: a released vif accepts nothing, and
    /// routing it through [`install`](Self::install) is what makes that one
    /// statement rather than one per impl.
    async fn release(&mut self, vif: VifId) -> Result<(), InstallRefusal> {
        self.install(vif, FrameFilter::CLOSED).await
    }
}

/// **The impl for a composition with no face onto the frame row**: refuses
/// every install, by name.
///
/// Not a stub and not a no-op. A no-op would let a row mint capabilities whose
/// enforcement never happened and report success — the one outcome worse than
/// refusing, because the holder then believes it is isolated and is not. This
/// refuses [`InstallRefusal::NoFrameFace`], so an unwired row says exactly
/// what it is missing instead of lying about what it did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NoFrameFace;

impl FlowInstall for NoFrameFace {
    async fn install(&mut self, _vif: VifId, _filter: FrameFilter) -> Result<(), InstallRefusal> {
        Err(InstallRefusal::NoFrameFace)
    }

    /// Nothing is held through a face that does not exist — the same answer
    /// `install` gives, for the same reason.
    fn held(&self, _vif: VifId) -> Result<(), InstallRefusal> {
        Err(InstallRefusal::NoFrameFace)
    }
}

/// Vifs one [`RecordingInstall`] remembers — the frame row's own ceiling, so
/// the double cannot hold a vif set the service could not.
pub const MAX_RECORDED_VIFS: usize = kore_driver::net::MAX_VIFS;

/// **The host double** — the reason every property of the mint is provable
/// without a frame row, a founder, or QEMU.
///
/// The `MockIfnet` pattern one layer down: a double that speaks the shipping
/// contract, so a test exercises the real engine rather than a parallel one.
/// It records what was installed on each vif and can be told to refuse, which
/// is what lets the unwind path be tested at all — an install that cannot fail
/// leaves the row's rollback unexercised.
#[derive(Debug)]
pub struct RecordingInstall {
    /// Per vif index: the filter currently installed, if any.
    installed: [Option<(VifId, FrameFilter)>; MAX_RECORDED_VIFS],
    /// Refuse every install with this, instead of recording it.
    refuse: Option<InstallRefusal>,
    /// Installs attempted, refusals included.
    attempts: usize,
}

impl Default for RecordingInstall {
    fn default() -> Self {
        Self::new()
    }
}

impl RecordingInstall {
    /// A double that accepts every install.
    pub const fn new() -> Self {
        Self {
            installed: [const { None }; MAX_RECORDED_VIFS],
            refuse: None,
            attempts: 0,
        }
    }

    /// A double that refuses every install with `refusal` — the shape that
    /// exercises a caller's unwind.
    pub const fn refusing(refusal: InstallRefusal) -> Self {
        Self {
            installed: [const { None }; MAX_RECORDED_VIFS],
            refuse: Some(refusal),
            attempts: 0,
        }
    }

    /// Start or stop refusing, so one test can cover both sides of a
    /// transition.
    pub fn set_refusal(&mut self, refusal: Option<InstallRefusal>) {
        self.refuse = refusal;
    }

    /// What is installed on `vif`, or `None` if nothing is — including when
    /// the slot's live tenant is a different generation, because an authority
    /// does not survive its vif.
    pub fn filter(&self, vif: VifId) -> Option<FrameFilter> {
        self.installed
            .get(usize::from(vif.index))?
            .filter(|(id, _)| *id == vif)
            .map(|(_, f)| f)
    }

    /// Whether `vif` currently enforces nothing — either never installed, or
    /// released.
    pub fn is_closed(&self, vif: VifId) -> bool {
        self.filter(vif).is_none_or(|f| f == FrameFilter::CLOSED)
    }

    /// Installs attempted, refusals included.
    pub fn attempts(&self) -> usize {
        self.attempts
    }
}

impl FlowInstall for RecordingInstall {
    async fn install(&mut self, vif: VifId, filter: FrameFilter) -> Result<(), InstallRefusal> {
        self.attempts += 1;
        if let Some(refusal) = self.refuse {
            return Err(refusal);
        }
        let slot = self
            .installed
            .get_mut(usize::from(vif.index))
            .ok_or(InstallRefusal::NoVif)?;
        *slot = Some((vif, filter));
        Ok(())
    }

    /// The double has no caller identity to check against, so every vif it
    /// can name is held. Its [`refuse`](Self::set_refusal) models an install
    /// that fails, not an ownership it denies — keeping the two separable is
    /// what lets one test cover a refused *release* on a vif the caller does
    /// own.
    fn held(&self, _vif: VifId) -> Result<(), InstallRefusal> {
        Ok(())
    }
}
