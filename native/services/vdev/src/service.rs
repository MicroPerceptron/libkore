//! The service: the device table, fed from the runtime.
//!
//! # Whose face is whose
//!
//! Pair zero is the founder's. Pair one is exec's, the *administration*
//! pair: exec builds a face for each program it launches with the
//! virtual-device profile, conveys its components here one request at a
//! time ([`VdevRequest::AdmitPeer`]), and retires it when the program ends
//! ([`VdevRequest::RetirePeer`]). Every other pair is such a face, a
//! driver's.
//!
//! A driver's face is two pages exec supplies: the request rings, then the
//! register block ([`REGISTERS_AT`]). The driver has both mapped before its
//! first instruction; the service is handed the second when the face is
//! admitted, and seats the driver over it.
//!
//! # What wakes the service
//!
//! A request on any pair, or a doorbell: a bare notice on a driver's kick.
//! Either way the service then looks at every open device
//! ([`Devices::service`]), so a doorbell needs no request and no reply. A
//! device whose budget ran out with chains still waiting is looked at again
//! at once, without waiting for another doorbell.

use kore_abi::task::geom;
use kore_rt::service::{PeerClose, ReturnMode, Serve, StartFailure, SystemService};
use kore_rt::svc::{
    PeerAdmissionRefusal, PeerBundleAssembler, PeerBundlePart, PeerId, PeerMapPlan, REP_CAP,
    REQ_CAP, window_bytes,
};
use kore_vdev::mem::Window;
use kore_vdev::regs;

pub use crate::face::{PEER_BYTES, REGISTERS_AT};
use crate::host::{Devices, MAX_SEATS};
pub use crate::wire::{VdevRefusal, VdevRequest, VdevResponse};

/// The service is endowed with nothing: it holds no device and no
/// authority over the machine, only its own heap and the faces exec
/// conveys to it.
#[derive(Debug, kore_rt::Bundle)]
#[bundle(name = "vdev")]
pub struct VdevEndowment;

/// The pair exec administers driver faces on.
const ADMIN_VIA: u16 = 1;

/// What keeps a lent window mapped in this task: the capability the driver
/// delivered. Dropping it revokes the view, and the mapping with it.
#[cfg(target_os = "none")]
type Lent = kore_rt::cap::ReturnedCapability;
/// Off metal nothing is ever lent: there is no arena to map from.
#[cfg(not(target_os = "none"))]
type Lent = core::convert::Infallible;

/// The virtual-device service.
#[derive(Debug)]
pub struct VdevService {
    devices: Devices<Lent>,
    /// A device ran out of budget with chains waiting: look again at once.
    pending: bool,
    peer_assembler: PeerBundleAssembler,
    peer_map: Option<PeerMapPlan>,
    last_admission: Option<Result<PeerId, PeerAdmissionRefusal>>,
}

impl VdevService {
    /// The seat of the driver a request arrived from, if it is a driver's
    /// face and still the occupant of its slot.
    fn seat_of(&self, via: u16) -> Option<usize> {
        let slot = PeerId::from_arrival(via)?.slot();
        (self.devices.arrival(slot) == Some(via)).then_some(slot)
    }

    /// The register block inside a face's tail: `aux` is the face after its
    /// rings.
    fn registers_in(aux: &'static mut [u8]) -> Option<Window> {
        let rings = window_bytes::<Self>(REQ_CAP, REP_CAP) as u64;
        let skip = usize::try_from(REGISTERS_AT - rings).ok()?;
        let block = aux.get_mut(skip..)?;
        if (block.len() as u64) < regs::BYTES {
            return None;
        }
        let base = core::ptr::NonNull::new(block.as_mut_ptr())?;
        if base.as_ptr() as usize % REGISTERS_AT as usize != 0 {
            return None;
        }
        // SAFETY: the host mapped the face read-write at a page boundary and
        // keeps it mapped until it has called `peer_detaching`, where the
        // seat and with it every copy of this view is dropped. The slice
        // this was made from is not used again, so nothing in this task
        // holds a reference into the page while the driver writes it.
        Some(unsafe { Window::new(base, regs::BYTES as usize) })
    }

    fn lend(&mut self, seat: usize, lent: Option<Lent>) -> VdevResponse {
        let Some(owner) = lent else {
            // Nothing arrived on the return leg to lend.
            return VdevResponse::Refused(VdevRefusal::Memory);
        };
        match map_lent(owner) {
            Some((memory, owner)) => match self.devices.lend(seat, memory, owner) {
                Ok(pages) => VdevResponse::Lent { pages },
                Err(refusal) => VdevResponse::Refused(refusal.into()),
            },
            None => VdevResponse::Refused(VdevRefusal::Memory),
        }
    }
}

/// Map what a driver lent. The owner comes back with the window so the two
/// are kept, and dropped, together. `None` drops the owner, which gives the
/// memory back.
#[cfg(target_os = "none")]
fn map_lent(owner: Lent) -> Option<(Window, Lent)> {
    use kore_rt::cap::verbs;
    let view = verbs::take_view(owner.slot()?)?;
    // A sink mapping: should the driver die with its memory lent, the
    // device's next look reads a sink page, which the transport takes for
    // what it is (a status of zero is a reset, an index of zero is an empty
    // queue), instead of faulting on a hole before the face is retired.
    let mapped = verbs::map_view_sink(view)?;
    let base = mapped.cast::<u8>();
    if base.as_ptr() as usize % core::mem::align_of::<u64>() != 0 {
        return None;
    }
    // SAFETY: the kernel installed `[base, +len)` read-write in this task
    // over the lent view and keeps it there while `owner` lives. The window
    // is handed to the device table with `owner`, which drops every view
    // before it drops the owner. The driver writes the same bytes at will,
    // which is what `Window` is for: no reference is ever formed into them.
    Some((unsafe { Window::new(base, mapped.len()) }, owner))
}

#[cfg(not(target_os = "none"))]
fn map_lent(owner: Lent) -> Option<(Window, Lent)> {
    match owner {}
}

/// Wake the driver parked on its device's interrupt count.
#[cfg(target_os = "none")]
fn interrupt(registers: &Window) {
    // The word's address, to name it to the kernel: never dereferenced.
    let word = registers
        .base()
        .as_ptr()
        .wrapping_add(regs::INTERRUPTS as usize);
    // The count moved before this. A driver not yet parked reads it again
    // before it parks; one that is parked is woken here. A wake the kernel
    // refuses is a lost kick, which the driver's deadline covers.
    if let Some((subject, offset)) = kore_rt::cap::mapped::locate(word) {
        let _ = kore_rt::cap::verbs::wake_word(subject, offset, u32::MAX);
    }
}

#[cfg(not(target_os = "none"))]
fn interrupt(_registers: &Window) {}

impl SystemService for VdevService {
    const NAME: &'static str = "vdev";
    type Endowment = VdevEndowment;

    /// Both return legs deliver here: the administration pair's carries a
    /// face's components, a driver's carries the memory it lends.
    const RETURN_MODE: ReturnMode = ReturnMode::RetainedAny;

    fn bind(_endowment: VdevEndowment) -> Self {
        Self {
            devices: Devices::new(),
            pending: false,
            peer_assembler: PeerBundleAssembler::new(0, true),
            peer_map: None,
            last_admission: None,
        }
    }

    /// One face for each program that drives a device, and many programs
    /// may run at once: the peer table grows from the service's heap.
    fn peer_arena(&self) -> Option<&'static (dyn core::alloc::Allocator + Sync)> {
        static GLOBAL: alloc::alloc::Global = alloc::alloc::Global;
        Some(&GLOBAL)
    }

    fn start_on_metal(&mut self, rt: &kore_rt::entry::Rt) -> Result<(), StartFailure> {
        // Where driver faces are mapped: one stride a slot, in the
        // transient window, for every slot the runtime can name.
        let window = rt
            .geom()
            .and_then(|geometry| geometry.window_of(geom::TRANSIENT))
            .filter(|window| {
                window.base() != 0
                    && window.base() & 0xfff == 0
                    && window.len() >= PEER_BYTES * MAX_SEATS as u64
            })
            .ok_or(StartFailure::new(1))?;
        self.peer_map = Some(PeerMapPlan {
            base: window.base(),
            stride: PEER_BYTES,
        });
        Ok(())
    }

    /// Runs on every wake, a doorbell included: look at every open device.
    fn poll_background(&mut self) {
        self.pending = self.devices.service(|_, registers| interrupt(registers));
    }

    /// A device with chains its budget did not reach is due now.
    fn next_deadline(&self) -> Option<u64> {
        self.pending.then_some(0)
    }

    fn drive_deadlines(&mut self, _now_ns: u64) -> u16 {
        // The look itself is `poll_background`'s, on the same pass.
        0
    }

    fn drive_host<T: kore_rt::transport::Transport>(
        host: kore_rt::svc::ServeHost<'_, '_, T, Self>,
    ) -> impl core::future::Future<Output = core::convert::Infallible> {
        async move {
            host.run_deadline_driven().await;
        }
    }

    fn serve(&mut self, req: VdevRequest, sv: &mut Serve<'_, VdevResponse>) {
        let resp = match req {
            VdevRequest::AdmitPeer { part, announced } => {
                // A `Ring` begins a face. One arriving while the assembler
                // still owns components means the last face was abandoned
                // by an asker that gave up: retire what it left, and refuse
                // the new one while any of those owners is still closing.
                // Administration only, and before anything else touches the
                // assembler.
                let abandoned = sv.via() == ADMIN_VIA
                    && matches!(part, PeerBundlePart::Ring)
                    && !self.peer_assembler.is_empty()
                    && !self.peer_assembler.retire_incomplete();
                if sv.via() != ADMIN_VIA || self.last_admission.is_some() || abandoned {
                    VdevResponse::Refused(VdevRefusal::Conflict)
                } else if let (Some(map), Some(mut input)) =
                    (self.peer_map, sv.returned_input(announced))
                {
                    let accepted = input
                        .capability()
                        .and_then(|capability| self.peer_assembler.accepts(part, capability).ok())
                        .is_some();
                    if !accepted {
                        VdevResponse::Refused(VdevRefusal::Denied)
                    } else {
                        let capability = input.take().expect("validated peer component remains");
                        let bundle = self
                            .peer_assembler
                            .push(part, capability)
                            .expect("synchronous peer validation remains stable");
                        if let Some(bundle) = bundle {
                            sv.admit_peer(bundle, map);
                        }
                        VdevResponse::Accepted
                    }
                } else {
                    VdevResponse::Refused(VdevRefusal::Unavailable)
                }
            }
            VdevRequest::TakeAdmission => {
                if sv.via() != ADMIN_VIA {
                    VdevResponse::Refused(VdevRefusal::Denied)
                } else {
                    match self.last_admission.take() {
                        Some(Ok(peer)) => VdevResponse::PeerAdmitted { peer },
                        Some(Err(_)) => VdevResponse::Refused(VdevRefusal::Unavailable),
                        // No verdict and components still held: the face
                        // never completed, so no verdict will ever come.
                        // `Pending` would be a wait with no end.
                        None if !self.peer_assembler.is_empty() => {
                            let _ = self.peer_assembler.retire_incomplete();
                            VdevResponse::Refused(VdevRefusal::Conflict)
                        }
                        // Complete, and the host has not admitted it yet.
                        None => VdevResponse::Refused(VdevRefusal::Pending),
                    }
                }
            }
            VdevRequest::RetirePeer { peer } => {
                if sv.via() == ADMIN_VIA
                    && self.devices.arrival(peer.slot()) == Some(peer.arrival())
                {
                    sv.retire_peer(peer, PeerClose::Orderly);
                    VdevResponse::Accepted
                } else {
                    VdevResponse::Refused(VdevRefusal::Denied)
                }
            }
            VdevRequest::ConfirmRetired { peer } => {
                if sv.via() == ADMIN_VIA
                    && self.devices.arrival(peer.slot()) != Some(peer.arrival())
                {
                    VdevResponse::Accepted
                } else {
                    VdevResponse::Refused(VdevRefusal::Conflict)
                }
            }
            VdevRequest::ClosePeer => {
                if self.seat_of(sv.via()).is_some() {
                    sv.close_peer(PeerClose::Orderly);
                    VdevResponse::Accepted
                } else {
                    VdevResponse::Refused(VdevRefusal::Denied)
                }
            }
            VdevRequest::Open { device } => match self.seat_of(sv.via()) {
                Some(seat) => match self.devices.open_device(seat, device) {
                    Ok(()) => VdevResponse::Accepted,
                    Err(refusal) => VdevResponse::Refused(refusal.into()),
                },
                None => VdevResponse::Refused(VdevRefusal::Denied),
            },
            VdevRequest::Lend { announced } => {
                // Taken whatever the answer: a capability lent to a request
                // that is refused is dropped, which gives it back.
                let lent = sv
                    .returned_input(announced)
                    .and_then(|mut input| input.take());
                match self.seat_of(sv.via()) {
                    Some(seat) => self.lend(seat, lent.and_then(lent_owner)),
                    None => VdevResponse::Refused(VdevRefusal::Denied),
                }
            }
            VdevRequest::Close => match self.seat_of(sv.via()) {
                Some(seat) => match self.devices.close_device(seat) {
                    Ok(()) => VdevResponse::Accepted,
                    Err(refusal) => VdevResponse::Refused(refusal.into()),
                },
                None => VdevResponse::Refused(VdevRefusal::Denied),
            },
        };
        sv.reply(resp);
    }

    /// The face's tail is its register block: seat the driver over it.
    fn attach_peer_aux(&mut self, peer: PeerId, aux: &'static mut [u8]) -> bool {
        match Self::registers_in(aux) {
            Some(registers) => self.devices.seat(peer.slot(), peer.arrival(), registers),
            None => false,
        }
    }

    /// The host is about to unmap the face: nothing here may touch its
    /// register block, or the memory its driver lent, after this.
    fn peer_detaching(&mut self, peer: PeerId) {
        if self.devices.arrival(peer.slot()) == Some(peer.arrival()) {
            self.devices.unseat(peer.slot());
        }
    }

    fn peer_admitted(&mut self, result: Result<PeerId, PeerAdmissionRefusal>) -> bool {
        if let Ok(peer) = result
            && self.devices.arrival(peer.slot()) != Some(peer.arrival())
        {
            // Admitted with no seat: the face's tail was never attached.
            self.last_admission = Some(Err(PeerAdmissionRefusal::PolicyCollision));
            return false;
        }
        self.last_admission = Some(result);
        true
    }

    fn peer_closed(&mut self, via: u16, _reason: PeerClose) -> bool {
        // `peer_detaching` has run for a face that was seated; this covers
        // one that closed before it was.
        if let Some(seat) = self.seat_of(via) {
            self.devices.unseat(seat);
        }
        true
    }
}

/// The capability a driver delivered, as what keeps its window mapped.
#[cfg(target_os = "none")]
fn lent_owner(capability: kore_rt::cap::ReturnedCapability) -> Option<Lent> {
    Some(capability)
}

/// Off metal a delivered capability maps nothing: it is dropped, and the
/// request is answered as one that lent no memory.
#[cfg(not(target_os = "none"))]
fn lent_owner(_capability: kore_rt::cap::ReturnedCapability) -> Option<Lent> {
    None
}
