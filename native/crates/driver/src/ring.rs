//! The **shared ring mechanics** — a single-writer producer ring
//! ([`SubRing`]) and a device-writer, validity-bit-polled consumer ring
//! ([`CompRing`]).  The register pokes (doorbells, completion
//! acknowledgements) stay caller-supplied closures; only the *index and
//! ordering discipline* — which is identical across NVMe and xHCI — lives
//! here.
//!
//! Ordering (easy for a unit double to forgive, load-bearing on hardware): a staged
//! entry is published to the device with a **release** fence before the
//! doorbell ([`SubRing::ring`]); a completion read takes an **acquire**
//! fence after a matching validity bit ([`CompRing::poll`]), so the rest of
//! the entry is observed coherently.

use core::marker::PhantomData;
use core::sync::atomic::{Ordering, fence};

use kore_abi::device::IovaAddr;

use crate::seam::Dma;

/// A **submission-ring entry** — the producer stages these.
pub trait SqEntry: Copy {
    /// Stamp the producer cycle bit into the entry before it is published.
    ///
    /// NVMe SQEs carry no in-entry validity bit — the SQ tail doorbell is
    /// the sole publish signal — so the default is a no-op.  xHCI TRBs carry
    /// a **cycle** bit the controller polls to know a TRB is ready; an xHCI
    /// entry sets it here (matching the producer cycle state the ring holds
    /// at stage time).
    fn stamp_cycle(&mut self, _cycle: bool) {}
}

/// A **completion-ring entry** — the device is the single writer and flips a
/// per-entry validity bit on each wrap; the reader polls it.
pub trait CqEntry: Copy {
    /// Whether this entry has been published for the expected `phase`
    /// (NVMe: the CQE status **Phase** bit; xHCI: the event TRB **Cycle**
    /// bit — the identical single-writer handshake).
    fn valid(&self, phase: bool) -> bool;
}

/// How a producer ring wraps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wrap {
    /// A plain circular array — all `capacity` slots usable, the next index
    /// is `(i + 1) % capacity`, no cycle state (NVMe SQ: the tail doorbell
    /// is the publish signal, entries carry no validity bit).
    Modulo,
    /// The last slot holds a **Link** entry the driver installs at init
    /// (pointing back to slot 0, its toggle-cycle bit set): `capacity - 1`
    /// slots are usable; reaching the link slot hops to 0 and flips the
    /// producer cycle state (xHCI command / transfer rings).
    LinkTrb,
}

/// A **producer ring** over a device-visible buffer `B`, entries `E`.  One
/// writer; the caller tracks in-flight count (the cookie/cid layer above
/// enforces "don't overrun").
pub struct SubRing<E: SqEntry, B: Dma> {
    buf: B,
    capacity: u16,
    tail: u16,
    cycle: bool,
    wrap: Wrap,
    _e: PhantomData<E>,
}

impl<E: SqEntry, B: Dma> SubRing<E, B> {
    /// A fresh ring over `buf`, `capacity` slots of `size_of::<E>()` bytes.
    /// Producer cycle state starts `true` (xHCI PCS = 1); `tail`/`head`
    /// at 0.
    pub fn new(buf: B, capacity: u16, wrap: Wrap) -> Self {
        Self {
            buf,
            capacity,
            tail: 0,
            cycle: true,
            wrap,
            _e: PhantomData,
        }
    }

    /// Start over at slot 0 with the producer cycle set — the state a device
    /// expects after its queue is created again (an NVMe controller reset).
    /// Entries past the tail are never read, so the slots stay as they are.
    pub fn reset(&mut self) {
        self.tail = 0;
        self.cycle = true;
    }

    /// Usable slots — `capacity` (Modulo) or `capacity - 1` (LinkTrb, the
    /// last slot is the Link entry).
    pub fn usable(&self) -> u16 {
        match self.wrap {
            Wrap::Modulo => self.capacity,
            Wrap::LinkTrb => self.capacity - 1,
        }
    }

    /// The producer enqueue index.
    pub fn tail(&self) -> u16 {
        self.tail
    }

    /// The current producer cycle state (LinkTrb rings; always the initial
    /// value for Modulo, which does not use it).
    pub fn cycle(&self) -> bool {
        self.cycle
    }

    /// The ring's base device address (what `ASQ` / `CRCR` point at).
    pub fn iova(&self) -> IovaAddr {
        self.buf.iova()
    }

    /// The CPU-visible ring base (for installing a Link entry at init).
    pub fn cpu(&self) -> *mut u8 {
        self.buf.cpu()
    }

    /// The backing buffer (for the Link-entry install / ring inspection).
    pub fn buf(&self) -> &B {
        &self.buf
    }

    /// The device address of slot `i` — `base + i * size_of::<E>()`.  xHCI
    /// matches a command completion by the device address of its command
    /// TRB; this computes it.
    pub fn slot_iova(&self, i: u16) -> IovaAddr {
        IovaAddr(self.buf.iova().0 + (i as u64) * core::mem::size_of::<E>() as u64)
    }

    /// **Stage** one entry at the tail: stamp the cycle bit, write it
    /// volatile, advance (hopping the Link slot + toggling cycle for
    /// LinkTrb).  Returns the slot it landed in.  Does **not** ring the
    /// doorbell — batch, then [`ring`](Self::ring) once.
    pub fn stage(&mut self, mut entry: E) -> u16 {
        let slot = self.tail;
        entry.stamp_cycle(self.cycle);
        // SAFETY: `buf` is `capacity * size_of::<E>()` device-visible bytes;
        // `slot < capacity`; the entry is naturally aligned within the page.
        unsafe {
            (self.buf.cpu() as *mut E)
                .add(slot as usize)
                .write_volatile(entry);
        }
        self.advance();
        slot
    }

    fn advance(&mut self) {
        match self.wrap {
            Wrap::Modulo => {
                self.tail = (self.tail + 1) % self.capacity;
            }
            Wrap::LinkTrb => {
                self.tail += 1;
                if self.tail == self.capacity - 1 {
                    // Reached the pre-installed Link entry.  Re-stamp its
                    // cycle bit to the PCS the data TRBs on this lap used
                    // (so the controller, arriving with a matching consumer
                    // cycle, *follows* the link), then hop back to slot 0 and
                    // flip the producer cycle for the next lap.  The Link's
                    // other fields (segment pointer, toggle-cycle) survive —
                    // `stamp_cycle` touches only the cycle bit.
                    // SAFETY: the Link slot is `capacity - 1 < capacity`,
                    // installed at init; `buf` covers it.
                    unsafe {
                        let link = (self.buf.cpu() as *mut E).add(self.tail as usize);
                        let mut e = link.read_volatile();
                        e.stamp_cycle(self.cycle);
                        link.write_volatile(e);
                    }
                    self.tail = 0;
                    self.cycle = !self.cycle;
                }
            }
        }
    }

    /// **Publish** the staged batch: a release fence pairs every staged
    /// write with the doorbell, then the caller's closure pokes the register
    /// (NVMe writes the tail index to the SQ doorbell; xHCI writes the
    /// endpoint target to the doorbell array — the closure receives the
    /// current tail either way, and may ignore it).
    pub fn ring(&self, doorbell: impl FnOnce(u16)) {
        fence(Ordering::Release);
        doorbell(self.tail);
    }
}

/// A **consumer ring** over a device-visible buffer `B`, entries `E`.  The
/// device is the single writer; a per-entry validity bit flips on each wrap.
/// A plain modulo ring — both the NVMe CQ and a **single-segment** xHCI
/// event ring wrap this way (the event ring carries no Link entries; the
/// ERST defines the segment, and one segment is spec-legal).
pub struct CompRing<E: CqEntry, B: Dma> {
    buf: B,
    capacity: u16,
    head: u16,
    phase: bool,
    _e: PhantomData<E>,
}

impl<E: CqEntry, B: Dma> CompRing<E, B> {
    /// A fresh ring over `buf`, `capacity` slots.  Expected phase starts
    /// `true` (the device's first pass writes validity = 1).
    pub fn new(buf: B, capacity: u16) -> Self {
        Self {
            buf,
            capacity,
            head: 0,
            phase: true,
            _e: PhantomData,
        }
    }

    /// Start over at slot 0 expecting phase `true`, with every slot zeroed —
    /// the state a device expects after its queue is created again. The
    /// zeroing matters: a slot left from before would carry a validity bit
    /// that reads as a fresh completion on the next pass.
    pub fn reset(&mut self) {
        let bytes = self.capacity as usize * size_of::<E>();
        // SAFETY: `buf` is this ring's own device-visible buffer of at least
        // `capacity` entries; the device writes it only while its queue
        // exists, and the caller resets only once the controller dropped it.
        unsafe { core::ptr::write_bytes(self.buf.cpu(), 0, bytes.min(self.buf.len())) };
        core::sync::atomic::fence(core::sync::atomic::Ordering::Release);
        self.head = 0;
        self.phase = true;
    }

    /// The consumer dequeue index.
    pub fn head(&self) -> u16 {
        self.head
    }

    /// Slots in the ring.
    pub fn capacity(&self) -> u16 {
        self.capacity
    }

    /// The expected validity phase for the head slot.
    pub fn phase(&self) -> bool {
        self.phase
    }

    /// The ring's base device address (what `ACQ` / `ERSTBA` point at).
    pub fn iova(&self) -> IovaAddr {
        self.buf.iova()
    }

    /// The CPU-visible ring base.
    pub fn cpu(&self) -> *mut u8 {
        self.buf.cpu()
    }

    /// The device address of the current head slot — xHCI writes this to
    /// ERDP (with EHB) to acknowledge a drain.
    pub fn head_iova(&self) -> IovaAddr {
        IovaAddr(self.buf.iova().0 + (self.head as u64) * core::mem::size_of::<E>() as u64)
    }

    /// **Poll one completion** (non-blocking): read the head slot; if it
    /// carries the expected phase the device has published it — take an
    /// acquire fence, advance the head, flip the expected phase on wrap, and
    /// return it.  `None` when the head slot is still stale.
    ///
    /// The caller acknowledges (NVMe: ring the CQ head doorbell; xHCI: write
    /// ERDP) — per-completion or once-per-drain, its choice.
    pub fn poll(&mut self) -> Option<E> {
        // SAFETY: `buf` is `capacity * size_of::<E>()` device-visible bytes;
        // `head < capacity`; the entry is naturally aligned.
        let entry = unsafe {
            (self.buf.cpu() as *const E)
                .add(self.head as usize)
                .read_volatile()
        };
        if !entry.valid(self.phase) {
            return None;
        }
        // The validity bit matched: the device published this entry — observe
        // the rest of it coherently.
        fence(Ordering::Acquire);
        self.head = (self.head + 1) % self.capacity;
        if self.head == 0 {
            self.phase = !self.phase;
        }
        Some(entry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::boxed::Box;
    use alloc::vec;

    /// A host-memory DMA buffer — real backing so ring writes/reads
    /// round-trip; the IOVA is a chosen base so the `slot_iova` math is
    /// checkable.
    struct HostBuf {
        mem: Box<[u8]>,
        iova: u64,
    }
    impl HostBuf {
        fn new(bytes: usize, iova: u64) -> Self {
            Self {
                mem: vec![0u8; bytes].into_boxed_slice(),
                iova,
            }
        }
    }
    impl Dma for HostBuf {
        fn cpu(&self) -> *mut u8 {
            self.mem.as_ptr() as *mut u8
        }
        fn iova(&self) -> IovaAddr {
            IovaAddr(self.iova)
        }
        fn len(&self) -> usize {
            self.mem.len()
        }
    }

    /// A 16-byte entry carrying a payload and a cycle bit — stands in for a
    /// TRB (validity in the control dword).
    #[derive(Clone, Copy, Default)]
    #[repr(C)]
    struct Trb {
        payload: u32,
        _rsvd: [u32; 2],
        control: u32,
    }
    impl SqEntry for Trb {
        fn stamp_cycle(&mut self, cycle: bool) {
            self.control = (self.control & !1) | (cycle as u32);
        }
    }
    impl CqEntry for Trb {
        fn valid(&self, phase: bool) -> bool {
            (self.control & 1) == (phase as u32)
        }
    }

    #[test]
    fn modulo_wraps_without_touching_cycle() {
        let buf = HostBuf::new(4 * 16, 0x2000);
        let mut ring: SubRing<Trb, _> = SubRing::new(buf, 4, Wrap::Modulo);
        assert_eq!(ring.usable(), 4);
        for i in 0..4 {
            assert_eq!(
                ring.stage(Trb {
                    payload: i,
                    ..Default::default()
                }),
                i as u16
            );
        }
        // Wrapped back to 0; Modulo never toggles cycle.
        assert_eq!(ring.tail(), 0);
        assert!(ring.cycle());
        assert_eq!(ring.slot_iova(2), IovaAddr(0x2000 + 2 * 16));
    }

    #[test]
    fn linktrb_hops_the_link_restamps_its_cycle_and_toggles() {
        // capacity 4 → slots 0..2 usable, slot 3 is the Link.
        let buf = HostBuf::new(4 * 16, 0x3000);
        // Pre-install a Link at slot 3: a marker in the high bits (stands in
        // for the segment pointer + toggle-cycle bit) and cycle deliberately
        // 0, so the re-stamp to the current PCS (1) is observable.
        const MARKER: u32 = 0xAB << 8;
        unsafe {
            (buf.cpu() as *mut Trb).add(3).write(Trb {
                control: MARKER,
                ..Default::default()
            });
        }
        let mut ring: SubRing<Trb, _> = SubRing::new(buf, 4, Wrap::LinkTrb);
        assert_eq!(ring.usable(), 3);
        assert!(ring.cycle(), "PCS starts 1");

        // Stage the 3 usable slots; the last advance hits the Link.
        assert_eq!(ring.stage(Trb::default()), 0);
        assert_eq!(ring.stage(Trb::default()), 1);
        assert_eq!(ring.stage(Trb::default()), 2);
        assert_eq!(ring.tail(), 0, "hopped the Link back to 0");
        assert!(!ring.cycle(), "cycle toggled on the wrap");

        // The Link's cycle bit was re-stamped to the pre-toggle PCS (1) so
        // the controller follows it; the marker (its other fields) survived.
        let link = unsafe { (ring.cpu() as *const Trb).add(3).read() };
        assert_eq!(
            link.control & 1,
            1,
            "Link cycle re-stamped to the lap's PCS"
        );
        assert_eq!(link.control & !1, MARKER, "Link's other fields preserved");

        // Next lap stamps the toggled cycle (0) into the data entry.
        assert_eq!(ring.stage(Trb::default()), 0);
        let written = unsafe { (ring.cpu() as *const Trb).read() };
        assert_eq!(
            written.control & 1,
            0,
            "data TRB stamped with the toggled cycle"
        );
    }

    #[test]
    fn compring_polls_by_phase_and_flips_on_wrap() {
        let buf = HostBuf::new(2 * 16, 0x4000);
        let cpu = buf.cpu();
        let mut ring: CompRing<Trb, _> = CompRing::new(buf, 2);
        // Nothing published yet (all-zero control → cycle 0 ≠ expected 1).
        assert!(ring.poll().is_none());

        // Device publishes slot 0 with cycle 1.
        unsafe {
            (cpu as *mut Trb).write(Trb {
                payload: 7,
                control: 1,
                ..Default::default()
            });
        }
        let e = ring.poll().expect("slot 0 valid");
        assert_eq!(e.payload, 7);
        assert_eq!(ring.head(), 1);
        assert!(ring.phase(), "no wrap yet");

        // Device publishes slot 1 with cycle 1, then the reader wraps.
        unsafe {
            (cpu as *mut Trb).add(1).write(Trb {
                payload: 9,
                control: 1,
                ..Default::default()
            });
        }
        let e = ring.poll().expect("slot 1 valid");
        assert_eq!(e.payload, 9);
        assert_eq!(ring.head(), 0, "wrapped");
        assert!(!ring.phase(), "expected phase flipped on wrap");

        // Slot 0 still holds the old cycle-1 entry, but we now expect 0 → stale.
        assert!(
            ring.poll().is_none(),
            "old entry is stale after the phase flip"
        );
    }

    #[test]
    fn ring_doorbell_passes_the_tail() {
        let buf = HostBuf::new(4 * 16, 0x5000);
        let mut ring: SubRing<Trb, _> = SubRing::new(buf, 4, Wrap::Modulo);
        ring.stage(Trb::default());
        ring.stage(Trb::default());
        let mut seen = None;
        ring.ring(|tail| seen = Some(tail));
        assert_eq!(seen, Some(2), "doorbell closure receives the current tail");
    }
}
