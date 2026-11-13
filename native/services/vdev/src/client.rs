//! The driver's side of the service: the face a program is launched with,
//! and exec's administration of such faces.
//!
//! A driver does four things with its face, and only the first two are
//! requests:
//!
//! 1. [`open`](ProcessVdev::open) a device model. The register block, which
//!    the driver has had mapped since it started, is written before the
//!    reply.
//! 2. [`lend`](ProcessVdev::lend) the memory its queue and buffers are in.
//!    The driver keeps the lease and its own mapping; the service receives
//!    a view of exactly the bytes named.
//! 3. Ring the [`doorbell`](ProcessVdev::doorbell): a notice, with no reply.
//! 4. [`wait`](ProcessVdev::wait) for the interrupt: a park on the register
//!    block's interrupt count, which any thread of the driver may do.
//!
//! Everything else a driver does with its device it does in the memory the
//! two share, through `kore_vdev::driver` or by hand.
//!
//! # A driver with more than one thread
//!
//! Requests belong to the thread the program was launched on: the face's
//! rings have one producer, and the replies wake that thread. The doorbell
//! and the interrupt are any thread's, but a capability is a thread's own, so
//! a thread that rings or parks must have been founded holding its share of
//! the face ([`THREAD_SHARES`], [`ThreadVdev`]). The register block itself is
//! mapped once, by the launched thread, and every thread of the program sees
//! that mapping.

use kore_abi::cap::{ResourceKind, Rights};
use kore_abi::session::SlotRef;
use kore_abi::task::role;
use kore_rt::svc::{
    CallError, PeerBundlePart, PeerId, ReturnDeriveError, ReturnDeriver, StagedReturn,
    StagedReturnError, SvcClient,
};
use kore_rt::thread_controller::Handed;
use kore_rt::transport::Transport;

use crate::VdevService;
use crate::wire::{VdevRefusal, VdevRequest, VdevResponse};

/// Where a program launched with the virtual-device profile finds its face
/// in its own boot block.
///
/// The layout exec stamps and the program binds, in one place. Indices
/// 0 to 5 of these role namespaces belong to the filesystem, window,
/// memory, accelerator and power faces a launch may also carry; exec pins
/// the disjointness where it sees every face.
pub mod process_face {
    /// The doorbell, and the request kick (`role::SVC_SEND`).
    pub const REQUEST_KICK: u16 = 8;
    /// The face itself: request rings, then the register block
    /// (`role::SVC_RING`).
    pub const CONTROL_RING: u16 = 8;
    /// The return leg a driver lends memory on (`role::SVC_RETURN`).
    pub const RETURN_SEND: u16 = 8;
    /// Correlation datum for the service's landing on that leg
    /// (`role::SVC_RETURN`).
    pub const RETURN_LAND: u16 = 9;
}

/// The cookie of a doorbell notice: `"vdevbell"`. The service looks at its
/// devices on any wake, so the cookie only says what the notice was for.
pub const COOKIE_DOORBELL: u64 = u64::from_le_bytes(*b"vdevbell");

/// Rights of the view of lent memory the service receives: it reads and
/// writes, and can pass it nowhere.
pub const LENT_RIGHTS: Rights = Rights::ACCESS.union(Rights::MODIFY);
/// The driver's staged view carries `DERIVE` as well, so the rendezvous can
/// attenuate it to [`LENT_RIGHTS`].
#[cfg(target_os = "none")]
const LENT_STAGED_RIGHTS: Rights = LENT_RIGHTS.union(Rights::DERIVE);

/// What a thread of a driver holds of the face, beyond the thread the
/// program was launched on: the face's memory, to park on the interrupt
/// count in it, and the kick, to ring the doorbell. It reads and notifies;
/// it maps nothing and writes nothing through either.
///
/// A program hands these on to the threads it founds
/// (`kore_rt::thread::share_endowment` for one it founds itself,
/// `Controller::hand_on` for the threads a controller founds).
pub const THREAD_SHARES: [Handed; 2] = [
    Handed {
        role: role::SVC_RING,
        index: process_face::CONTROL_RING,
        kind: ResourceKind::MemLease,
        rights: Rights::ACCESS,
    },
    Handed {
        role: role::SVC_SEND,
        index: process_face::REQUEST_KICK,
        kind: ResourceKind::ChanSend,
        rights: Rights::ACCESS,
    },
];

/// How a wait for the interrupt ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Waited {
    /// The interrupt count moved, to this.
    Interrupt(u32),
    /// The deadline passed with the count unmoved.
    Deadline,
    /// The caller's own condition ended the wait.
    Stopped,
    /// This thread cannot wait on the device: its share of the face is
    /// gone, or the kernel will not park it. Waiting again would not help.
    Gone,
}

/// The longest one park of an open-ended wait: the kernel's word wait is
/// never unbounded, so a wait with no deadline is a loop of bounded parks
/// and each re-reads the count. A quarter of a second, kore-rt's figure for
/// the same loop (`futex`, `event`): long enough that a waiter is not paced
/// by it, short enough that the caller's condition, which no wake
/// announces, is seen soon after it changes.
#[cfg(target_os = "none")]
const PARK_SLICE_NS: u64 = 250_000_000;

/// The ring client over the generic [`SvcClient`].
pub struct RingVdev<'w, 't, T: Transport>(pub SvcClient<'w, 't, T, VdevService>);

impl<T: Transport> RingVdev<'_, '_, T> {
    /// One request. A call that never reached the service answers
    /// [`VdevRefusal::Unavailable`]: to a driver the two are the same, a
    /// device it cannot have now.
    pub async fn request(&mut self, request: VdevRequest) -> VdevResponse {
        match self.0.call(&request).await {
            Ok(reply) => reply.response,
            Err(_) => VdevResponse::Refused(VdevRefusal::Unavailable),
        }
    }
}

/// Exec's fixed administration face, with its retained return leg.
pub struct RingVdevAdmin<'w, 't, T: Transport> {
    pub client: RingVdev<'w, 't, T>,
    return_tx: SlotRef,
    return_land: SlotRef,
}

impl<'w, 't, T: Transport> RingVdevAdmin<'w, 't, T> {
    pub const fn new(
        client: RingVdev<'w, 't, T>,
        return_tx: SlotRef,
        return_land: SlotRef,
    ) -> Self {
        Self {
            client,
            return_tx,
            return_land,
        }
    }

    /// Convey one component of a driver's face.
    pub async fn admit_component_staged<F>(
        &mut self,
        part: PeerBundlePart,
        rights: Rights,
        stage: F,
    ) -> Result<VdevResponse, StagedReturnError>
    where
        F: FnOnce(ReturnDeriver<'_>) -> Result<StagedReturn, ReturnDeriveError>,
    {
        let request = VdevRequest::AdmitPeer {
            part,
            announced: self.return_land,
        };
        self.client
            .0
            .call_with_staged_return(&request, self.return_tx, rights, stage)
            .await
            .map(|reply| reply.response)
    }

    pub async fn take_admission(&mut self) -> Result<VdevResponse, CallError> {
        self.call(VdevRequest::TakeAdmission).await
    }

    pub async fn retire_peer(&mut self, peer: PeerId) -> Result<VdevResponse, CallError> {
        self.call(VdevRequest::RetirePeer { peer }).await
    }

    pub async fn confirm_retired(&mut self, peer: PeerId) -> Result<VdevResponse, CallError> {
        self.call(VdevRequest::ConfirmRetired { peer }).await
    }

    async fn call(&mut self, request: VdevRequest) -> Result<VdevResponse, CallError> {
        self.client
            .0
            .call(&request)
            .await
            .map(|reply| reply.response)
    }
}

#[cfg(target_os = "none")]
pub use metal::{ProcessVdev, ThreadVdev, attach_process_face, attach_thread_face};

#[cfg(target_os = "none")]
mod metal {
    use super::*;
    use kore_rt::cap::verbs::{WordWait, WordWaitError};
    use kore_rt::cap::{LeaseToken, verbs};
    use kore_rt::service::MAX_GRANTS;
    use kore_rt::svc::{ClientEndpoints, REP_CAP, REQ_CAP, ReturnLeaseSource, client_rings};
    use kore_rt::transport::TrapTransport;
    use kore_vdev::mem::{Memory, Window};
    use kore_vdev::regs;

    use crate::face::{PEER_BYTES, REGISTERS_AT};

    // A notice needs exactly what a thread's share of the kick carries.
    const _: () = assert!(THREAD_SHARES[1].rights.bits() == verbs::NOTICE_RIGHTS.bits());

    /// One thread's part of the face: the doorbell and the interrupt.
    ///
    /// Both name capabilities in the calling thread's own arena, so a value
    /// of this is its thread's and is attached on the thread that uses it.
    #[derive(Clone, Copy, Debug)]
    pub struct ThreadVdev {
        kick: SlotRef,
        /// The interrupt count, named to the kernel: this thread's holding
        /// of the face's memory and the count's offset in it.
        interrupts: (SlotRef, u32),
    }

    /// Attach the calling thread's part of the face from its own boot
    /// block. `None` on a thread founded without it.
    pub fn attach_thread_face(rt: &kore_rt::entry::Rt) -> Option<ThreadVdev> {
        let boot = rt.bootstrap();
        Some(ThreadVdev {
            kick: boot.handle(role::SVC_SEND, process_face::REQUEST_KICK)?,
            interrupts: (
                boot.handle(role::SVC_RING, process_face::CONTROL_RING)?,
                u32::try_from(REGISTERS_AT + regs::INTERRUPTS).ok()?,
            ),
        })
    }

    impl ThreadVdev {
        /// Ring the doorbell. `false` when the notice was not taken: the
        /// service's doorbell is full, which means it has a wake standing
        /// and will look anyway.
        pub fn doorbell(&self) -> bool {
            verbs::post_notice(self.kick, COOKIE_DOORBELL, 0)
        }

        /// Park until the interrupt count is no longer `seen`, or until
        /// `deadline_ns` (absolute monotonic nanoseconds): one park, and
        /// what ended it. The caller reads the count from the register
        /// block afterwards, whatever this answers.
        ///
        /// The park compares in the kernel, so an interrupt raised between
        /// the caller's read of `seen` and this call is not lost.
        pub fn park(&self, seen: u32, deadline_ns: u64) -> Result<WordWait, WordWaitError> {
            let (subject, offset) = self.interrupts;
            verbs::wait_word(subject, offset, seen, deadline_ns)
        }

        /// Wait until the interrupt count in `registers` is no longer
        /// `seen`: until `deadline_ns` (absolute monotonic nanoseconds)
        /// when one is given, and for as long as `go_on` answers `true`,
        /// which is asked before every park.
        ///
        /// Every wait is a park in the kernel; nothing here spins. A park
        /// the kernel has no room for now is a short cede and another look.
        pub fn wait(
            &self,
            registers: Window,
            seen: u32,
            deadline_ns: Option<u64>,
            mut go_on: impl FnMut() -> bool,
        ) -> Waited {
            loop {
                match registers.load_u32(regs::INTERRUPTS) {
                    Ok(count) if count != seen => return Waited::Interrupt(count),
                    Ok(_) => {}
                    Err(_) => return Waited::Gone,
                }
                if !go_on() {
                    return Waited::Stopped;
                }
                let Some(now) = kore_rt::clock::deadline_base_nanos() else {
                    // No way to name an instant, and the kernel takes no
                    // unbounded park.
                    return Waited::Gone;
                };
                let slice = now.saturating_add(PARK_SLICE_NS);
                let bound = match deadline_ns {
                    Some(at) if now >= at => return Waited::Deadline,
                    Some(at) => at.min(slice),
                    None => slice,
                };
                match self.park(seen, bound) {
                    Ok(WordWait::Changed | WordWait::Deadline) => {}
                    Ok(WordWait::WouldBlock) => {
                        if !kore_rt::pace::cede() {
                            return Waited::Gone;
                        }
                    }
                    Ok(WordWait::Revoked) | Err(_) => return Waited::Gone,
                }
            }
        }
    }

    /// A program's face with the service: its request rings, the register
    /// block of its device, the leg it lends memory on, and the launched
    /// thread's own doorbell and interrupt.
    pub struct ProcessVdev<'t> {
        client: RingVdev<'static, 't, TrapTransport>,
        registers: Window,
        thread: ThreadVdev,
        return_tx: SlotRef,
        return_land: SlotRef,
        /// Keeps the face mapped for the life of the program.
        _ring: LeaseToken,
    }

    /// Attach the face from a profiled program's boot block. `None` when the
    /// program was launched without one: there is no device to open.
    pub fn attach_process_face<'t>(
        rt: &kore_rt::entry::Rt,
        transport: &'t TrapTransport,
    ) -> Option<ProcessVdev<'t>> {
        let boot = rt.bootstrap();
        let thread = attach_thread_face(rt)?;
        let ring = verbs::take_lease(boot.handle(role::SVC_RING, process_face::CONTROL_RING)?)?;
        let window = verbs::map_held_rw(ring.slot())?;
        if (window.len() as u64) < PEER_BYTES {
            return None;
        }
        let base = window.cast::<u8>();
        // SAFETY: the first page is the face's rings, and this is the
        // program's sole client attachment over the task-lifetime lease exec
        // delivered. The slice stops short of the register block, so no
        // reference covers what the device writes.
        let rings: &'static mut [u8] =
            unsafe { core::slice::from_raw_parts_mut(base.as_ptr(), REGISTERS_AT as usize) };
        let rings = client_rings::<VdevService>(rings, REQ_CAP, REP_CAP).ok()?;
        // SAFETY: the second page of the same mapping, page-aligned, mapped
        // read-write for as long as `ring` is held, which is as long as
        // this value lives. The service writes it concurrently, which is
        // what a `Window` is for.
        let registers =
            unsafe { Window::new(base.add(REGISTERS_AT as usize), regs::BYTES as usize) };
        Some(ProcessVdev {
            client: RingVdev(SvcClient::new(
                transport,
                rings,
                ClientEndpoints {
                    kick: thread.kick,
                    grants: [const { None }; MAX_GRANTS],
                },
            )),
            registers,
            thread,
            return_tx: boot.handle(role::SVC_RETURN, process_face::RETURN_SEND)?,
            return_land: SlotRef::from_wire(
                boot.datum(role::SVC_RETURN, process_face::RETURN_LAND)? as u32,
            ),
            _ring: ring,
        })
    }

    impl ProcessVdev<'_> {
        /// The register block of this program's device. It says a device is
        /// there once [`open`](Self::open) has been accepted. The mapping is
        /// the process's: every thread of the program may use it for as long
        /// as this value lives.
        pub const fn registers(&self) -> Window {
            self.registers
        }

        /// The launched thread's own doorbell and interrupt.
        pub const fn thread(&self) -> ThreadVdev {
            self.thread
        }

        /// Open the device model with this register-block id.
        pub async fn open(&mut self, device: u32) -> VdevResponse {
            self.client.request(VdevRequest::Open { device }).await
        }

        /// Lend the first `bytes` of `source`: device addresses are offsets
        /// into them from here on. The caller keeps the lease and its own
        /// mapping of it.
        pub async fn lend<S: ReturnLeaseSource>(&mut self, source: &S, bytes: u64) -> VdevResponse {
            let request = VdevRequest::Lend {
                announced: self.return_land,
            };
            let reply = self
                .client
                .0
                .call_with_staged_return(&request, self.return_tx, LENT_RIGHTS, |derive| {
                    derive.lease_window(source, LENT_STAGED_RIGHTS, 0, bytes)
                })
                .await;
            match reply {
                Ok(reply) => reply.response,
                Err(_) => VdevResponse::Refused(VdevRefusal::Unavailable),
            }
        }

        /// Give the device up, and take back the memory lent to it.
        pub async fn close(&mut self) -> VdevResponse {
            self.client.request(VdevRequest::Close).await
        }

        /// Ring the doorbell ([`ThreadVdev::doorbell`]).
        pub fn doorbell(&self) -> bool {
            self.thread.doorbell()
        }

        /// The interrupt count now.
        pub fn interrupts(&self) -> u32 {
            // The block is a page of a mapping this value keeps: the load
            // cannot fault.
            self.registers.load_u32(regs::INTERRUPTS).unwrap_or(0)
        }

        /// Wait until the interrupt count is no longer `seen`, or until
        /// `deadline_ns` (absolute monotonic nanoseconds), and answer the
        /// count. `seen` again means no interrupt came: the deadline
        /// passed, or this thread cannot wait on the device
        /// ([`ThreadVdev::wait`]).
        pub fn wait(&self, seen: u32, deadline_ns: u64) -> u32 {
            match self
                .thread
                .wait(self.registers, seen, Some(deadline_ns), || true)
            {
                Waited::Interrupt(count) => count,
                Waited::Deadline | Waited::Stopped | Waited::Gone => seen,
            }
        }
    }
}
