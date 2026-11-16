//! A virtual device behind a descriptor.
//!
//! A program launched with Kore's virtual-device profile has a face with the
//! virtual-device service: a device with no hardware behind it, for a driver
//! to be written against (Kore's `res/VIRTUAL_DEVICE.md`). This module is
//! C's spelling for holding one, in the shape Linux's UIO gives a userspace
//! driver:
//!
//! - **`open`** a name of the device model. The service starts the model and
//!   writes its register block. One descriptor at a time: the face is one
//!   device's.
//! - **`mmap` at offset 0** answers the register block, one page the program
//!   and the device both write.
//! - **`mmap` at offset one page** answers fresh, zeroed memory the driver
//!   owns and lends the device: a device address is an offset into it. One
//!   such region at a time, and it cannot be unmapped while the device is
//!   open, because the device may be using it.
//! - **`read`** of one 32-bit word waits for an interrupt and answers the
//!   interrupt count. Each interrupt is read once.
//! - **`write`** of one 32-bit word rings the doorbell. The word is not read.
//! - **`poll`** says whether an interrupt is waiting to be read, with a
//!   timeout.
//! - **`close`** gives the device up. The service drops the model and its
//!   view of the lent memory, which stays the program's until it is unmapped.
//!
//! # Whose capability does what
//!
//! Capabilities are per thread. Thread 0 was launched holding the face, so
//! everything that is a request to the service (open, lend, close) is posted
//! to it and waited for, as a mapping of the memory supply is. The interrupt
//! and the doorbell are the hot path and never cross a thread: this library
//! founds the program's threads holding their own share of the face
//! ([`shares`]), so `read` parks the calling thread in the kernel on the
//! interrupt count and `write` posts its notice directly. A thread founded
//! without a share is refused with `ENXIO`, whoever opened the descriptor.
//!
//! The request rules, the refusals and the state a descriptor's words keep
//! are plain data, so the host suite exercises the code the target runs; the
//! face is the metal half.

use core::ffi::c_int;
use core::sync::atomic::{AtomicU32, Ordering};

use kore_std::vdev::client::process_face;
use kore_std::vdev::wire::VdevRefusal;

use crate::fd::errno;
use crate::mmap::{MAP_FIXED, MAP_SHARED, PAGE, PROT_READ, PROT_WRITE, pages};

/// The face's instance in `role::SVC_RING`: the coordinate a device name
/// stands for.
pub(crate) const FACE: u16 = process_face::CONTROL_RING;
/// The face's kick in `role::SVC_SEND`: what a thread rings the doorbell on.
pub(crate) const KICK: u16 = process_face::REQUEST_KICK;

/// The virtual accelerator's register-block id, `"accl"`.
pub(crate) const MODEL_ACCEL: u32 = u32::from_le_bytes(*b"accl");

/// `mmap` offsets on a device descriptor, UIO's convention: map `N` is at
/// offset `N` pages.
pub(crate) const MAP_REGISTERS: i64 = 0;
pub(crate) const MAP_MEMORY: i64 = PAGE as i64;

/// What `read` answers and `write` takes: one 32-bit word.
pub(crate) const WORD: usize = core::mem::size_of::<u32>();

/// What an `mmap` on a device descriptor asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Map {
    /// The register block.
    Registers,
    /// Memory to lend the device: `asked` whole pages' bytes.
    Memory { asked: usize },
}

impl Map {
    /// What `mmap`'s arguments ask of a device, or the `errno` that refuses
    /// them.
    ///
    /// `EINVAL`: no length, an offset that is neither map, a register
    /// mapping longer than the block, or anything but `MAP_SHARED` (the
    /// device sees every store, so there is no private copy to make).
    /// `ENOTSUP`: `MAP_FIXED`, and any protection but read or read-write.
    /// `ENOMEM`: a length that does not round to pages.
    pub(crate) fn of(len: usize, prot: c_int, flags: c_int, offset: i64) -> Result<Self, c_int> {
        if len == 0 {
            return Err(errno::EINVAL);
        }
        if flags & MAP_FIXED != 0 {
            return Err(errno::ENOTSUP);
        }
        if flags != MAP_SHARED {
            return Err(errno::EINVAL);
        }
        if prot != PROT_READ && prot != PROT_WRITE && prot != PROT_READ | PROT_WRITE {
            return Err(errno::ENOTSUP);
        }
        match offset {
            MAP_REGISTERS if len <= PAGE => Ok(Self::Registers),
            MAP_MEMORY => pages(len)
                .map(|asked| Self::Memory { asked })
                .ok_or(errno::ENOMEM),
            _ => Err(errno::EINVAL),
        }
    }

    /// Whether the mapping needs the descriptor opened for writing: any
    /// writable one, and lent memory always, since lending is an act and the
    /// driver's queue lives there.
    pub(crate) fn writes(self, prot: c_int) -> bool {
        matches!(self, Self::Memory { .. }) || prot & PROT_WRITE != 0
    }
}

/// The `errno` for a request the service refused.
pub(crate) fn refused(refusal: VdevRefusal) -> c_int {
    match refusal {
        // No model has that id.
        VdevRefusal::NoDevice => errno::ENXIO,
        // The device is not in a state for this: already open, or holding a
        // queue in the memory it has.
        VdevRefusal::Busy => errno::EBUSY,
        // Every device the service hosts at once is taken: ask again when
        // another driver closes.
        VdevRefusal::Limit => errno::EAGAIN,
        // What was lent is not memory a device can address.
        VdevRefusal::Memory => errno::EINVAL,
        // This face may not ask that.
        VdevRefusal::Denied => errno::EACCES,
        // The service did not answer, or answered out of turn.
        VdevRefusal::Unavailable | VdevRefusal::Pending | VdevRefusal::Conflict => errno::EIO,
    }
}

/// The open device's words: which opening this is, and the interrupt count
/// last read.
///
/// The generation is odd while a device is open. A blocked `read` belongs to
/// the opening it began under, and ends with `EBADF` when that opening does.
pub(crate) struct Opening {
    generation: AtomicU32,
    seen: AtomicU32,
}

impl Opening {
    pub(crate) const fn new() -> Self {
        Self {
            generation: AtomicU32::new(0),
            seen: AtomicU32::new(0),
        }
    }

    /// The opening now, if a device is open.
    pub(crate) fn current(&self) -> Option<u32> {
        let generation = self.generation.load(Ordering::Acquire);
        (generation & 1 != 0).then_some(generation)
    }

    /// Whether `generation` is still the opening.
    pub(crate) fn is(&self, generation: u32) -> bool {
        self.generation.load(Ordering::Acquire) == generation
    }

    /// A device opened with its interrupt count at `count`: interrupts
    /// before now are not this opening's to read. `false` when one is open.
    pub(crate) fn begin(&self, count: u32) -> bool {
        let generation = self.generation.load(Ordering::Acquire);
        if generation & 1 != 0 {
            return false;
        }
        self.seen.store(count, Ordering::Release);
        self.generation
            .store(generation.wrapping_add(1), Ordering::Release);
        true
    }

    /// The device closed. `false` when none was open.
    pub(crate) fn end(&self) -> bool {
        let generation = self.generation.load(Ordering::Acquire);
        if generation & 1 == 0 {
            return false;
        }
        self.generation
            .store(generation.wrapping_add(1), Ordering::Release);
        true
    }

    /// The interrupt count last read.
    pub(crate) fn seen(&self) -> u32 {
        self.seen.load(Ordering::Acquire)
    }

    /// Take the interrupt that moved the count from `seen` to `count`.
    /// `false` when another reader took it first.
    pub(crate) fn take(&self, seen: u32, count: u32) -> bool {
        self.seen
            .compare_exchange(seen, count, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }
}

#[cfg(target_os = "none")]
pub(crate) use metal::{
    Host, close, holds_kick, map, open, posted, read, readable, ring, shares, unmap, wait_readable,
};

/// Off Kore no thread holds a face.
#[cfg(not(target_os = "none"))]
pub(crate) fn holds_kick() -> bool {
    false
}

#[cfg(target_os = "none")]
mod metal {
    extern crate alloc;

    use core::cell::UnsafeCell;
    use core::ffi::c_int;
    use core::ptr::NonNull;
    use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
    use core::time::Duration;

    use kore_std::abi::cap::{ResourceKind, Rights};
    use kore_std::abi::task::role;
    use kore_std::runtime::cap::{MappedLeaseView, RetireOutcome, StashedLease, verbs};
    use kore_std::runtime::entry::{self, Rt};
    use kore_std::runtime::futex::{self, WaitOutcome};
    use kore_std::runtime::pace;
    use kore_std::runtime::thread::{self, WorkerCap};
    use kore_std::runtime::transport::TrapTransport;
    use kore_std::vdev::client::{
        ProcessVdev, THREAD_SHARES, ThreadVdev, Waited, attach_process_face, attach_thread_face,
    };
    use kore_std::vdev::mem::{Memory as _, Window};
    use kore_std::vdev::regs;
    use kore_std::vdev::wire::VdevResponse;

    use super::{FACE, KICK, Map, Opening, PAGE, refused};
    use crate::fd::errno;
    use crate::lock::Lock;

    /// The device this program has open, if it has one.
    static OPENING: Opening = Opening::new();

    /// The register block's address in this process, once thread 0 has
    /// attached the face; zero before. The mapping is thread 0's and lasts
    /// the program.
    static REGISTERS: AtomicUsize = AtomicUsize::new(0);

    /// The lent memory's address while the program holds a mapping of it;
    /// zero otherwise.
    static MEMORY: AtomicUsize = AtomicUsize::new(0);

    /// The register block, to read the interrupt count through.
    fn registers() -> Option<Window> {
        let base = NonNull::new(REGISTERS.load(Ordering::Acquire) as *mut u8)?;
        // SAFETY: thread 0 published the address of the face's second page,
        // which it keeps mapped read-write for the life of the process. The
        // device writes it concurrently, which is what a `Window` is for.
        Some(unsafe { Window::new(base, regs::BYTES as usize) })
    }

    /// The calling thread's own part of the face.
    fn face() -> Option<ThreadVdev> {
        attach_thread_face(&entry::thread_context()?)
    }

    /// Whether the calling thread holds the face's kick with the right a
    /// notice needs.
    pub(crate) fn holds_kick() -> bool {
        entry::thread_context()
            .and_then(|rt| rt.bootstrap().handle(role::SVC_SEND, KICK))
            .and_then(|kick| verbs::capability_rights(kick, ResourceKind::ChanSend))
            .is_some_and(|rights| rights.contains(verbs::NOTICE_RIGHTS))
    }

    /// The endowment that founds a thread holding its share of the face:
    /// what thread 0 gives the main thread, with `DERIVE` so the main thread
    /// can found the thread controller holding it too. Empty for a launch
    /// without a face.
    pub(crate) fn shares() -> impl Iterator<Item = WorkerCap> {
        THREAD_SHARES.iter().filter_map(|share| {
            thread::share_endowment(
                share.role,
                share.index,
                share.kind,
                share.rights | Rights::DERIVE,
            )
        })
    }

    // The table's row names the same coordinate the shares do.
    const _: () = assert!(THREAD_SHARES[0].index == FACE && THREAD_SHARES[1].index == KICK);

    // ── The threads' half ───────────────────────────────────────────────

    /// `read`: wait for an interrupt of this opening and take it.
    ///
    /// `EBADF`: no device is open, or it closed while this thread waited.
    /// `ENXIO`: the calling thread holds no share of the face. `EIO`: the
    /// thread cannot wait on the device (its share is gone, or the kernel
    /// will not park it).
    pub(crate) fn read() -> Result<u32, c_int> {
        let generation = OPENING.current().ok_or(errno::EBADF)?;
        let registers = registers().ok_or(errno::EBADF)?;
        let face = face().ok_or(errno::ENXIO)?;
        loop {
            let seen = OPENING.seen();
            match face.wait(registers, seen, None, || OPENING.is(generation)) {
                // Another reader may have taken this interrupt: then wait
                // for the next.
                Waited::Interrupt(count) => {
                    if OPENING.take(seen, count) {
                        return Ok(count);
                    }
                }
                Waited::Stopped => return Err(errno::EBADF),
                Waited::Deadline | Waited::Gone => return Err(errno::EIO),
            }
        }
    }

    /// `write`: ring the doorbell. A notice the service's doorbell had no
    /// room for is not an error: it has a wake standing and looks anyway.
    ///
    /// `EBADF`: no device is open. `ENXIO`: the calling thread holds no
    /// share of the face.
    pub(crate) fn ring() -> Result<(), c_int> {
        OPENING.current().ok_or(errno::EBADF)?;
        let _ = face().ok_or(errno::ENXIO)?.doorbell();
        Ok(())
    }

    /// Whether an interrupt is waiting to be read. `None` when no device is
    /// open.
    pub(crate) fn readable() -> Option<bool> {
        OPENING.current()?;
        let count = registers()?.load_u32(regs::INTERRUPTS).ok()?;
        Some(count != OPENING.seen())
    }

    /// `poll`'s wait: until an interrupt is waiting to be read or
    /// `deadline_ns` passes. `Ok(true)`: one is waiting. `Ok(false)`: the
    /// deadline passed. `Err`: as [`read`], and the caller reports it on the
    /// descriptor.
    pub(crate) fn wait_readable(deadline_ns: Option<u64>) -> Result<bool, c_int> {
        let generation = OPENING.current().ok_or(errno::EBADF)?;
        let registers = registers().ok_or(errno::EBADF)?;
        let face = face().ok_or(errno::ENXIO)?;
        match face.wait(registers, OPENING.seen(), deadline_ns, || {
            OPENING.is(generation)
        }) {
            Waited::Interrupt(_) => Ok(true),
            Waited::Deadline => Ok(false),
            Waited::Stopped => Err(errno::EBADF),
            Waited::Gone => Err(errno::EIO),
        }
    }

    /// `open`: have the service start the model with this id.
    ///
    /// `EBUSY`: this program has a device open. `ENXIO`: the launch carried
    /// no face, the program has no host to ask, or no model has that id.
    /// `ENOTSUP`: asked from thread 0 itself, which cannot wait for its own
    /// answer. Otherwise as the service refuses ([`refused`]).
    pub(crate) fn open(model: u32) -> Result<(), c_int> {
        ask(Op::Open { model }).map(|_| ())
    }

    /// `close`: give the device up. Nothing is reported: a device that could
    /// not be closed ends with the program.
    pub(crate) fn close() {
        let _ = ask(Op::Close);
    }

    /// `mmap` on the open device.
    ///
    /// `EBADF`: no device is open. `EBUSY`: the program already holds a
    /// mapping of device memory. `ENOMEM`: the supply or the link window has
    /// no room for the region, or the kernel refused the map. Otherwise as
    /// the service refuses the loan ([`refused`]).
    pub(crate) fn map(map: Map) -> Result<*mut u8, c_int> {
        OPENING.current().ok_or(errno::EBADF)?;
        match map {
            Map::Registers => NonNull::new(REGISTERS.load(Ordering::Acquire) as *mut u8)
                .map(NonNull::as_ptr)
                .ok_or(errno::EBADF),
            Map::Memory { asked } => ask(Op::Lend { asked }).map(|base| base as *mut u8),
        }
    }

    /// `munmap` of a device mapping. `None` when `base` is not one, and the
    /// memory supply's rules apply.
    ///
    /// The register block is the process's for its life: unmapping it is
    /// accepted and changes nothing. `EBUSY`: the lent memory, while the
    /// device is open. `EINVAL`: `len` is not the mapping's length.
    /// `EAGAIN`: the kernel did not confirm the unmap.
    pub(crate) fn unmap(base: usize, len: usize) -> Option<Result<(), c_int>> {
        if base == 0 {
            return None;
        }
        if base == REGISTERS.load(Ordering::Acquire) {
            return Some(if len == 0 || len > PAGE {
                Err(errno::EINVAL)
            } else {
                Ok(())
            });
        }
        if base != MEMORY.load(Ordering::Acquire) {
            return None;
        }
        Some(ask(Op::Release { base, len }).map(|_| ()))
    }

    // ── Reaching thread 0 ───────────────────────────────────────────────

    /// One request of thread 0.
    #[derive(Clone, Copy)]
    enum Op {
        Open { model: u32 },
        Close,
        Lend { asked: usize },
        Release { base: usize, len: usize },
    }

    // The cell's states. A caller owns it at IDLE (under [`CALLERS`]) and at
    // SERVED, thread 0 at POSTED.
    const IDLE: u32 = 0;
    const POSTED: u32 = 1;
    const SERVED: u32 = 2;

    /// One request at a time: the caller holding [`CALLERS`] writes `op` and
    /// posts, thread 0 writes `answer` and serves.
    struct Call {
        state: AtomicU32,
        op: UnsafeCell<Op>,
        answer: UnsafeCell<Result<usize, c_int>>,
    }
    // SAFETY: `op` and `answer` are touched only by the side `state` hands
    // the cell to.
    unsafe impl Sync for Call {}
    static CALL: Call = Call {
        state: AtomicU32::new(IDLE),
        op: UnsafeCell::new(Op::Close),
        answer: UnsafeCell::new(Ok(0)),
    };

    /// Callers take turns at the cell.
    static CALLERS: Lock = Lock::new();

    /// How long a caller waits before it re-reads the cell: thread 0 kicks
    /// the main thread when it has answered, and holds no send into a
    /// pthread's doorbell, which rides this slice (the mapping cell's rule).
    const SLICE: Duration = Duration::from_nanos(pace::CEDE_NS);

    /// Whether a request waits for thread 0.
    pub(crate) fn posted() -> bool {
        CALL.state.load(Ordering::Acquire) == POSTED
    }

    /// Have thread 0 carry `op` out, and wait for its answer.
    fn ask(op: Op) -> Result<usize, c_int> {
        if crate::program::on_host() {
            // Thread 0 answers these between its own turns: it cannot wait
            // here for an answer only it can give.
            return Err(errno::ENOTSUP);
        }
        if !crate::program::hosted() {
            return Err(errno::ENXIO);
        }
        CALLERS.acquire();
        // SAFETY: IDLE under the lock hands the cell to this caller.
        unsafe { *CALL.op.get() = op };
        CALL.state.store(POSTED, Ordering::SeqCst);
        // Refused only when no post stood, and then thread 0 re-reads the
        // cell before it parks.
        let _ = thread::kick_parent();
        loop {
            match CALL.state.load(Ordering::Acquire) {
                SERVED => break,
                state => {
                    if futex::wait(&CALL.state, state, Some(SLICE)) == WaitOutcome::Unsupported {
                        // No doorbell to park on: the deadline park any
                        // thread can.
                        let _ = pace::cede();
                    }
                }
            }
        }
        // SAFETY: SERVED hands the cell back to this caller.
        let answer = unsafe { *CALL.answer.get() };
        CALL.state.store(IDLE, Ordering::Release);
        CALLERS.release();
        answer
    }

    // ── Thread 0's half ─────────────────────────────────────────────────

    /// The memory lent to the device, or kept from the last loan.
    enum Lent {
        /// Nothing carved.
        Nothing,
        /// Unmapped. Its lease and address serve the next loan of its
        /// length: the supply's cursors and the window's only advance.
        Spare {
            lease: StashedLease,
            base: usize,
            len: usize,
        },
        /// Mapped in this process. `lent` while the device has it.
        Mapped {
            lease: StashedLease,
            view: MappedLeaseView,
            base: usize,
            len: usize,
            lent: bool,
        },
    }

    /// Thread 0's side of the program's device: the face, attached on first
    /// use, and the memory lent through it.
    pub(crate) struct Host<'r, 't> {
        rt: &'r Rt,
        transport: &'t TrapTransport,
        face: Option<ProcessVdev<'t>>,
        /// The launch carried no face: every later request is refused the
        /// same way without asking again.
        absent: bool,
        memory: Lent,
    }

    impl<'r, 't> Host<'r, 't> {
        pub(crate) fn new(rt: &'r Rt, transport: &'t TrapTransport) -> Self {
            Self {
                rt,
                transport,
                face: None,
                absent: false,
                memory: Lent::Nothing,
            }
        }

        /// Answer the posted request, if there is one. `true` when one was
        /// answered. Dropped mid-request when the program ends first; the
        /// caller it leaves waiting is stopped with the rest of the program.
        pub(crate) async fn serve(&mut self) -> bool {
            if !posted() {
                return false;
            }
            // SAFETY: POSTED hands the cell to thread 0 until it stores
            // SERVED.
            let op = unsafe { *CALL.op.get() };
            let answer = self.perform(op).await;
            // SAFETY: as above.
            unsafe { *CALL.answer.get() = answer };
            // SeqCst, as the futex contract asks of the store before a wake.
            CALL.state.store(SERVED, Ordering::SeqCst);
            futex::wake(&CALL.state);
            true
        }

        fn face(&mut self) -> Result<&mut ProcessVdev<'t>, c_int> {
            if self.absent {
                return Err(errno::ENXIO);
            }
            if self.face.is_none() {
                match attach_process_face(self.rt, self.transport) {
                    Some(face) => {
                        REGISTERS
                            .store(face.registers().base().as_ptr() as usize, Ordering::Release);
                        self.face = Some(face);
                    }
                    None => {
                        self.absent = true;
                        return Err(errno::ENXIO);
                    }
                }
            }
            self.face.as_mut().ok_or(errno::ENXIO)
        }

        async fn perform(&mut self, op: Op) -> Result<usize, c_int> {
            match op {
                Op::Open { model } => self.open(model).await,
                Op::Close => self.close().await,
                Op::Lend { asked } => self.lend(asked).await,
                Op::Release { base, len } => self.release(base, len),
            }
        }

        async fn open(&mut self, model: u32) -> Result<usize, c_int> {
            if OPENING.current().is_some() {
                return Err(errno::EBUSY);
            }
            let face = self.face()?;
            match face.open(model).await {
                VdevResponse::Accepted => {
                    // Interrupts from before this opening are not its to
                    // read.
                    OPENING.begin(face.interrupts());
                    Ok(0)
                }
                VdevResponse::Refused(refusal) => Err(refused(refusal)),
                _ => Err(errno::EIO),
            }
        }

        async fn close(&mut self) -> Result<usize, c_int> {
            if !OPENING.end() {
                return Ok(0);
            }
            let face = self.face()?;
            // The service drops the model and its view of what was lent.
            // Only its word for that frees the memory to be unmapped: a
            // close it did not answer leaves the loan standing, and the
            // face ends with the program.
            let closed = matches!(face.close().await, VdevResponse::Accepted);
            if closed && let Lent::Mapped { lent, .. } = &mut self.memory {
                *lent = false;
            }
            Ok(0)
        }

        async fn lend(&mut self, asked: usize) -> Result<usize, c_int> {
            if OPENING.current().is_none() {
                return Err(errno::EBADF);
            }
            // A mapping the program was never given (a loan the service
            // refused, whose unmap the kernel did not confirm) is retired
            // before anything else is tried.
            if MEMORY.load(Ordering::Acquire) == 0 {
                self.retire_unpublished();
            }
            let (lease, base, len) = match core::mem::replace(&mut self.memory, Lent::Nothing) {
                Lent::Spare { lease, base, len } if len == asked => (lease, base, len),
                mapped @ Lent::Mapped { .. } => {
                    self.memory = mapped;
                    return Err(errno::EBUSY);
                }
                // A spare of another length is given up for good, and
                // nothing carved is nothing to reuse.
                Lent::Spare { .. } | Lent::Nothing => {
                    let lease = kore_std::runtime::heap::reserve_from_top(asked as u64)
                        .ok_or(errno::ENOMEM)?;
                    let len = lease
                        .slot()
                        .and_then(verbs::lease_geometry)
                        .and_then(|(_, len)| usize::try_from(len).ok())
                        .ok_or(errno::ENOMEM)?;
                    let base = crate::mmap::place(len).ok_or(errno::ENOMEM)?;
                    (lease, base, len)
                }
            };
            let Some(view) = crate::mmap::map_zeroed(&lease, base, len, true) else {
                self.memory = Lent::Spare { lease, base, len };
                return Err(errno::ENOMEM);
            };
            let face = match self.face() {
                Ok(face) => face,
                Err(code) => {
                    self.memory = Lent::Mapped {
                        lease,
                        view,
                        base,
                        len,
                        lent: false,
                    };
                    self.retire_unpublished();
                    return Err(code);
                }
            };
            let answer = face.lend(&lease, len as u64).await;
            let lent = matches!(
                answer,
                VdevResponse::Lent { pages } if pages as usize == len / PAGE
            );
            self.memory = Lent::Mapped {
                lease,
                view,
                base,
                len,
                lent,
            };
            if lent {
                MEMORY.store(base, Ordering::Release);
                return Ok(base);
            }
            self.retire_unpublished();
            Err(match answer {
                VdevResponse::Refused(refusal) => refused(refusal),
                _ => errno::EIO,
            })
        }

        /// Unmap a mapping the program does not hold and the device does not
        /// have. One the kernel does not confirm stays as it is, to be tried
        /// again.
        fn retire_unpublished(&mut self) {
            self.memory = match core::mem::replace(&mut self.memory, Lent::Nothing) {
                Lent::Mapped {
                    lease,
                    mut view,
                    base,
                    len,
                    lent: false,
                } => match view.try_retire_outcome() {
                    RetireOutcome::Retired | RetireOutcome::AlreadyDead => {
                        Lent::Spare { lease, base, len }
                    }
                    RetireOutcome::Retryable => Lent::Mapped {
                        lease,
                        view,
                        base,
                        len,
                        lent: false,
                    },
                },
                other => other,
            };
        }

        fn release(&mut self, at: usize, asked: usize) -> Result<usize, c_int> {
            match &self.memory {
                Lent::Mapped { base, .. } if *base != at => return Err(errno::EINVAL),
                Lent::Mapped { lent: true, .. } => return Err(errno::EBUSY),
                Lent::Mapped { len, .. } => {
                    if crate::mmap::pages(asked) != Some(*len) {
                        return Err(errno::EINVAL);
                    }
                }
                Lent::Spare { .. } | Lent::Nothing => return Err(errno::EINVAL),
            }
            // The program's mapping ends here whether or not the kernel
            // confirms it at once: the address is no longer its to use.
            MEMORY.store(0, Ordering::Release);
            self.retire_unpublished();
            match self.memory {
                Lent::Mapped { base, .. } => {
                    // Still mapped: the program may ask again.
                    MEMORY.store(base, Ordering::Release);
                    Err(errno::EAGAIN)
                }
                _ => Ok(0),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RW: c_int = PROT_READ | PROT_WRITE;
    const MAP_PRIVATE: c_int = crate::mmap::MAP_PRIVATE;

    #[test]
    fn offset_zero_is_the_register_block_and_one_page_is_memory() {
        assert_eq!(Map::of(PAGE, RW, MAP_SHARED, 0), Ok(Map::Registers));
        // A shorter length is still the one page.
        assert_eq!(Map::of(0x58, PROT_READ, MAP_SHARED, 0), Ok(Map::Registers));
        assert_eq!(
            Map::of(1 << 20, RW, MAP_SHARED, MAP_MEMORY),
            Ok(Map::Memory { asked: 1 << 20 })
        );
        // Lent memory rounds up to whole pages, as any mapping does.
        assert_eq!(
            Map::of(PAGE + 1, RW, MAP_SHARED, MAP_MEMORY),
            Ok(Map::Memory { asked: 2 * PAGE })
        );
    }

    #[test]
    fn a_device_mapping_is_refused_by_cause() {
        assert_eq!(Map::of(0, RW, MAP_SHARED, 0), Err(errno::EINVAL));
        // The block is one page.
        assert_eq!(Map::of(PAGE + 1, RW, MAP_SHARED, 0), Err(errno::EINVAL));
        // No third map, and no offset inside one.
        for offset in [-1, 1, 2 * PAGE as i64, MAP_MEMORY + 1] {
            assert_eq!(
                Map::of(PAGE, RW, MAP_SHARED, offset),
                Err(errno::EINVAL),
                "{offset}"
            );
        }
        // The device sees every store: there is no private copy.
        assert_eq!(Map::of(PAGE, RW, MAP_PRIVATE, 0), Err(errno::EINVAL));
        assert_eq!(
            Map::of(PAGE, RW, MAP_SHARED | MAP_PRIVATE, 0),
            Err(errno::EINVAL)
        );
        assert_eq!(
            Map::of(PAGE, RW, MAP_SHARED | MAP_FIXED, 0),
            Err(errno::ENOTSUP)
        );
        // PROT_NONE and PROT_EXEC.
        for prot in [0, 4, RW | 4] {
            assert_eq!(
                Map::of(PAGE, prot, MAP_SHARED, 0),
                Err(errno::ENOTSUP),
                "{prot}"
            );
        }
        assert_eq!(
            Map::of(usize::MAX, RW, MAP_SHARED, MAP_MEMORY),
            Err(errno::ENOMEM)
        );
    }

    #[test]
    fn lent_memory_and_a_writable_block_need_a_writable_descriptor() {
        assert!(!Map::Registers.writes(PROT_READ));
        assert!(Map::Registers.writes(RW));
        assert!(Map::Registers.writes(PROT_WRITE));
        // Lending is an act, whatever protection the mapping asks.
        assert!(Map::Memory { asked: PAGE }.writes(PROT_READ));
        assert!(Map::Memory { asked: PAGE }.writes(RW));
    }

    #[test]
    fn every_refusal_has_an_errno() {
        assert_eq!(refused(VdevRefusal::NoDevice), errno::ENXIO);
        assert_eq!(refused(VdevRefusal::Busy), errno::EBUSY);
        assert_eq!(refused(VdevRefusal::Limit), errno::EAGAIN);
        assert_eq!(refused(VdevRefusal::Memory), errno::EINVAL);
        assert_eq!(refused(VdevRefusal::Denied), errno::EACCES);
        for refusal in [
            VdevRefusal::Unavailable,
            VdevRefusal::Pending,
            VdevRefusal::Conflict,
        ] {
            assert_eq!(refused(refusal), errno::EIO, "{refusal:?}");
        }
    }

    #[test]
    fn one_device_is_open_at_a_time_and_each_opening_is_its_own() {
        let opening = Opening::new();
        assert_eq!(opening.current(), None);
        assert!(!opening.end(), "nothing was open");
        assert!(opening.begin(7));
        let first = opening.current().expect("open");
        assert!(opening.is(first));
        assert_eq!(
            opening.seen(),
            7,
            "interrupts before the opening are not its"
        );
        assert!(!opening.begin(9), "a second device");
        assert_eq!(opening.seen(), 7, "a refused opening changes nothing");
        assert!(opening.end());
        assert_eq!(opening.current(), None);
        assert!(!opening.is(first), "a reader of the first opening ends");
        assert!(opening.begin(0));
        let second = opening.current().expect("open again");
        assert_ne!(first, second);
        assert!(!opening.is(first));
    }

    #[test]
    fn an_interrupt_is_read_once() {
        let opening = Opening::new();
        assert!(opening.begin(0));
        // Two readers saw the count at zero and woke on the same interrupt.
        assert!(opening.take(0, 1));
        assert!(!opening.take(0, 1), "the second waits for the next");
        assert_eq!(opening.seen(), 1);
        // The count wraps.
        assert!(opening.take(1, u32::MAX));
        assert!(opening.take(u32::MAX, 0));
        assert_eq!(opening.seen(), 0);
    }

    #[test]
    fn the_names_coordinates_are_the_faces() {
        assert_eq!(MODEL_ACCEL, 0x6c63_6361);
        assert_eq!(MAP_MEMORY, 0x1000);
        assert_eq!(WORD, 4);
        let shares = kore_std::vdev::client::THREAD_SHARES;
        assert_eq!(shares[0].index, FACE);
        assert_eq!(shares[1].index, KICK);
    }
}
