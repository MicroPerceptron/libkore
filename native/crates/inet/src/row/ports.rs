//! **The port space** — the other thing exactly one party may arbitrate.
//!
//! Two tasks that both believe they hold port 80 is not a race the datapath
//! can resolve, because by the time a segment arrives there is no record of
//! who asked first. The row keeps that record.
//!
//! Heapless and design-bounded, like every other table in this tree: a fixed
//! array sized by [`MAX_BOUND_PORTS`], searched linearly. The bound is small
//! enough that linear is faster than anything cleverer, and a bound that is
//! visible is a bound an operator can reason about.

use kore_driver::net::VifId;
use kore_driver::net::flow::IpProto;

/// Ports the row may hold bound at once.
///
/// An architectural constant, not a tuning knob: it sizes an inline array, so
/// raising it is a deliberate act with a memory cost attached.
pub const MAX_BOUND_PORTS: usize = 64;

/// Below this, a port is privileged and needs a separate grant.
///
/// The convention every IP host shares. Keeping it as a *grant* rather than a
/// uid check is the kore half: the authority to bind :80 is something a task
/// holds, not something it is.
pub const FIRST_UNPRIVILEGED_PORT: u16 = 1024;

/// The range an unnamed bind draws from (IANA dynamic/private).
pub const EPHEMERAL_FIRST: u16 = 49152;
/// The last ephemeral port.
pub const EPHEMERAL_LAST: u16 = 65535;

/// Why a bind was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindRefusal {
    /// Another vif holds this port.
    Taken,
    /// A privileged port without the grant to bind one.
    Privileged,
    /// Port zero is the wildcard on the wire and never a binding.
    Reserved,
    /// Every ephemeral port is spoken for.
    NoEphemeral,
    /// The table is full.
    Exhausted,
}

#[derive(Clone, Copy, Debug)]
struct Binding {
    /// TCP/443 and UDP/443 are two ports, not one.
    ///
    /// Without this the table arbitrates a space that does not exist: a
    /// UDP listener would refuse because a TCP one holds "the" port, and a
    /// release by either would free the other's. The transport number is
    /// part of a local endpoint's identity everywhere else in the stack —
    /// `FlowMatch` carries it, the classifier matches on it — and the one
    /// place it was missing is the one place that hands the endpoint out.
    proto: IpProto,
    port: u16,
    owner: VifId,
}

/// Who holds which local port.
#[derive(Debug)]
pub struct PortTable {
    bound: [Option<Binding>; MAX_BOUND_PORTS],
    /// Where the next ephemeral search starts, so repeated binds do not
    /// re-scan the same taken prefix and do not hand out the same number
    /// twice in a row after a release.
    next_ephemeral: u16,
}

impl Default for PortTable {
    fn default() -> Self {
        Self::new()
    }
}

impl PortTable {
    /// An empty table.
    pub const fn new() -> Self {
        Self {
            bound: [None; MAX_BOUND_PORTS],
            next_ephemeral: EPHEMERAL_FIRST,
        }
    }

    /// Who holds `port`, if anyone.
    pub fn holder(&self, proto: IpProto, port: u16) -> Option<VifId> {
        self.bound
            .iter()
            .flatten()
            .find(|b| b.proto == proto && b.port == port)
            .map(|b| b.owner)
    }

    /// Ports currently bound.
    pub fn len(&self) -> usize {
        self.bound.iter().flatten().count()
    }

    /// Whether nothing is bound.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bind `port` to `owner`.
    ///
    /// `may_bind_privileged` is the grant, passed in rather than inferred:
    /// this table arbitrates, it does not decide policy.
    pub fn bind(
        &mut self,
        proto: IpProto,
        port: u16,
        owner: VifId,
        may_bind_privileged: bool,
    ) -> Result<u16, BindRefusal> {
        if port == 0 {
            return Err(BindRefusal::Reserved);
        }
        if port < FIRST_UNPRIVILEGED_PORT && !may_bind_privileged {
            return Err(BindRefusal::Privileged);
        }
        if self.holder(proto, port).is_some() {
            return Err(BindRefusal::Taken);
        }
        let slot = self
            .bound
            .iter_mut()
            .find(|s| s.is_none())
            .ok_or(BindRefusal::Exhausted)?;
        *slot = Some(Binding { proto, port, owner });
        Ok(port)
    }

    /// Bind any free ephemeral port to `owner`.
    ///
    /// Scans the whole range once from a rotating cursor, so a caller cannot
    /// be starved by a neighbour that happens to hold the first few.
    pub fn bind_ephemeral(&mut self, proto: IpProto, owner: VifId) -> Result<u16, BindRefusal> {
        let span = (EPHEMERAL_LAST - EPHEMERAL_FIRST) as u32 + 1;
        for step in 0..span {
            let offset = (self.next_ephemeral - EPHEMERAL_FIRST) as u32 + step;
            let port = EPHEMERAL_FIRST + (offset % span) as u16;
            if self.holder(proto, port).is_none() {
                let got = self.bind(proto, port, owner, false)?;
                self.next_ephemeral = if got == EPHEMERAL_LAST {
                    EPHEMERAL_FIRST
                } else {
                    got + 1
                };
                return Ok(got);
            }
        }
        Err(BindRefusal::NoEphemeral)
    }

    /// Release `port`, but only to the vif that holds it.
    ///
    /// Returns whether anything was released. A release by a non-holder is a
    /// no-op rather than an error, and deliberately not a success either:
    /// letting one client free another's port would be a denial of service
    /// with no capability involved.
    pub fn release(&mut self, proto: IpProto, port: u16, owner: VifId) -> bool {
        for slot in self.bound.iter_mut() {
            if let Some(b) = slot
                && b.proto == proto
                && b.port == port
                && b.owner == owner
            {
                *slot = None;
                return true;
            }
        }
        false
    }

    /// Release everything a vif holds.
    ///
    /// What a close runs. Generation is part of [`VifId`], so a slot's next
    /// tenant never inherits the previous one's bindings.
    pub fn release_all(&mut self, owner: VifId) -> usize {
        let mut freed = 0;
        for slot in self.bound.iter_mut() {
            if let Some(b) = slot
                && b.owner == owner
            {
                *slot = None;
                freed += 1;
            }
        }
        freed
    }
}
