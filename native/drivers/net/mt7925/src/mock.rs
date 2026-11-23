//! **`MockMt7925`** — an MT7925's register plane in process memory.
//!
//! `res/NETWORK_RESTART.md` §3's sixth rule: *the mock is the development
//! environment.* One machine, no serial port, one flash per experiment, and
//! no QEMU model of this part — so every fact established here is a boot not
//! spent.
//!
//! What it models is the thing that can actually be wrong: **the address
//! translation**. A plain byte array would let a wrong `route` pass, because
//! a wrong BAR0 offset still reads *some* word. So this double keeps the
//! chip's address space, not BAR0's, and reaches it the way the part does:
//!
//! - offsets below [`DIRECT_LIMIT`] are chip addresses already;
//! - a write to [`MT_HIF_REMAP_L1`] moves the L1 window, and a read through
//!   `MT_HIF_REMAP_BASE_L1 + offset` resolves to `(window << 16) | offset`;
//! - a write to [`MT_HIF_REMAP_L2`] parks a full address that the L2 window
//!   then resolves to.
//!
//! A driver that computes the wrong window, or forgets to program it, reads
//! the **wrong chip address** here and the test fails — which is exactly what
//! would happen on metal, and what a flat register file would hide.

use core::cell::{Cell, RefCell};

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use kore_abi::device::IovaAddr;
use kore_driver::host_double::{HostBuf, HostPool};
use kore_driver::seam::{Dma, DmaPool, Regs};

use crate::regs::mt7925::*;
use crate::regs::{DIRECT_LIMIT, RegMap, DEFAULT_FIXED_MAP};

/// An MT7925's register plane, addressed in chip space.
pub struct MockMt7925 {
    /// Chip-space address -> value. Sparse, because the space is 32 bits
    /// wide and the ladder touches a handful of words.
    cells: RefCell<BTreeMap<u32, u32>>,
    /// The current L1 window base, as [`MT_HIF_REMAP_L1`] holds it.
    l1: RefCell<u32>,
    /// The address parked in [`MT_HIF_REMAP_L2`].
    l2: RefCell<u32>,
    /// Fault switch for the D1b quiet rung: an engine that never goes idle.
    ///
    /// `res/NETWORK_RESTART.md` §3's sixth rule: each rung needs a switch
    /// that makes exactly that rung refuse, or its timeout path is code
    /// nothing ever executes.
    pub busy_stuck: Cell<bool>,
    /// Fault switch for the D1b receive rung: an engine that takes nothing.
    ///
    /// When clear, the mock advances `dma_idx` to whatever `cpu_idx` was
    /// written — the engine consuming what it was handed. When set, it
    /// leaves `dma_idx` where it was.
    pub rx_stuck: Cell<bool>,
    /// Fault switch for the ownership rung: a part that never hands over.
    ///
    /// `res/NETWORK_RESTART.md` §3's sixth rule wants each rung to have a
    /// switch that makes exactly that rung refuse — otherwise a timeout path
    /// is code nothing ever executes.
    pub own_stuck: Cell<bool>,
    /// **Opt-in**: a transmit queue whose engine consumes what it is handed.
    ///
    /// Default **off**, and that default is the point. On a *receive* queue
    /// `dma_idx` advances only on data actually received, which an earlier
    /// version of this mock got wrong and earned a witness against a
    /// fiction. On a *transmit* queue it is the engine's consumer index and
    /// does advance as descriptors are read — `mt76_dma_tx_cleanup`
    /// (`dma.c`) frees entries up to it, which is a citation rather than a
    /// belief.
    ///
    /// It is still a model, so it stays off unless a test says otherwise
    /// and says why. Nothing passes here by accident.
    pub tx_consumes: Cell<bool>,
    /// Every chip address read, in order — so a test can assert that a
    /// ladder reached the register it claims to have reached.
    pub reads: RefCell<Vec<u32>>,
    /// Every chip-space write, in order, as `(address, value)`.
    ///
    /// The reads log answers "did it get there"; this answers the question
    /// a refusal path raises, which is the opposite one: **did it touch the
    /// part at all?** A rung that refuses after half-programming a bank
    /// leaves the engine holding a value nobody intended, and that is
    /// invisible to a test that only checks the returned error.
    pub writes: RefCell<Vec<(u32, u32)>>,
}

impl MockMt7925 {
    /// A part that answers with its own identity and nothing else set.
    pub fn new() -> Self {
        let this = Self {
            cells: RefCell::new(BTreeMap::new()),
            l1: RefCell::new(0),
            l2: RefCell::new(0),
            own_stuck: Cell::new(false),
            busy_stuck: Cell::new(false),
            tx_consumes: Cell::new(false),
            rx_stuck: Cell::new(false),
            reads: RefCell::new(Vec::new()),
            writes: RefCell::new(Vec::new()),
        };
        this.poke(MT_HW_CHIPID, 0x7925);
        // A cold part is the firmware's, not the driver's: `OWN_SYNC` reads
        // set until a `CLR_OWN` strobe is answered.
        this.poke(MT_CONN_ON_LPCTL, PCIE_LPCR_HOST_OWN_SYNC);
        this.poke(MT_HW_REV, 0x00);
        this
    }

    /// The cell a chip address is stored under.
    ///
    /// A directly addressable register has two lawful chip names — its own,
    /// and the fixed block whose BAR0 window covers it — and this mock
    /// stores one cell per *offset*, under whichever name `resolve` picks.
    /// A test that wants to look at the cell a driver wrote must ask for it
    /// rather than assume, which is what this answers.
    pub fn cell_of(&self, chip: u32) -> u32 {
        self.resolve(bar_offset(chip))
    }

    /// Set a chip-space register directly, bypassing every window.
    pub fn poke(&self, addr: u32, val: u32) {
        self.cells.borrow_mut().insert(addr, val);
    }

    /// Read a chip-space register directly, bypassing every window.
    pub fn peek(&self, addr: u32) -> u32 {
        self.cells.borrow().get(&addr).copied().unwrap_or(0)
    }

    /// `MT_CONN_ON_LPCTL` is **not** a plain read/write cell.
    ///
    /// `SET_OWN` and `CLR_OWN` are write strobes; `OWN_SYNC` is read-only
    /// status that the *part* drives in response to them
    /// (`mt792x_core.c:1045-1051` writes one and then polls the other). A
    /// flat register file would let the driver's own `CLR_OWN` write appear
    /// to clear `OWN_SYNC`, so a driver that never waited would pass — which
    /// is precisely the bug this double exists to catch.
    fn write_lpctl(&self, val: u32) {
        let mut cur = self.peek(MT_CONN_ON_LPCTL);
        // The strobes are recorded so a test can assert the driver asked.
        cur = (cur & !(PCIE_LPCR_HOST_SET_OWN | PCIE_LPCR_HOST_CLR_OWN))
            | (val & (PCIE_LPCR_HOST_SET_OWN | PCIE_LPCR_HOST_CLR_OWN));
        if !self.own_stuck.get() {
            if val & PCIE_LPCR_HOST_CLR_OWN != 0 {
                cur &= !PCIE_LPCR_HOST_OWN_SYNC;
            }
            if val & PCIE_LPCR_HOST_SET_OWN != 0 {
                cur |= PCIE_LPCR_HOST_OWN_SYNC;
            }
        }
        self.poke(MT_CONN_ON_LPCTL, cur);
    }

    /// Resolve a BAR0 offset to the chip address it currently reaches.
    ///
    /// This is the inverse of `regs::route`, and deliberately written
    /// independently of it: a translation checked against its own inverse
    /// proves nothing, so this walks the same upstream table from the other
    /// end and the test asserts the round trip.
    fn resolve(&self, off: u32) -> u32 {
        if off == MT_HIF_REMAP_L1 || off == MT_HIF_REMAP_L2 {
            return off;
        }
        // The L1 window: `MT_HIF_REMAP_BASE_L1 .. +64 KiB`.
        if (MT_HIF_REMAP_BASE_L1..MT_HIF_REMAP_BASE_L1 + 0x1_0000).contains(&off) {
            let window = *self.l1.borrow();
            let base = (window & MT_HIF_REMAP_L1_MASK) >> MT_HIF_REMAP_L1_MASK.trailing_zeros();
            // The L2 case: the window sits at L2's base and an address was
            // parked, so the very first word of the window is that address.
            if base == MT_HIF_REMAP_BASE_L2 >> 16 && off == MT_HIF_REMAP_BASE_L1 {
                return *self.l2.borrow();
            }
            return (base << 16) | (off - MT_HIF_REMAP_BASE_L1);
        }
        // A statically mapped block, searched from the BAR0 side.
        // **Exclusive** upper bound, where `regs::route` uses upstream's
        // inclusive one. That is not an inconsistency to be reconciled: the
        // inclusive bound makes block N's one-past-end offset collide with
        // block N+1's base, so the backward direction is genuinely ambiguous
        // at those boundaries and only an exclusive bound has a single
        // answer. `regs::tests::upstream_blocks_overlap_by_one_word` pins the
        // collision with its concrete addresses.
        for &RegMap { phys, maps, size } in DEFAULT_FIXED_MAP {
            if off >= maps && off - maps < size {
                return phys + (off - maps);
            }
        }
        // Below the direct limit and in no window: a chip address already.
        debug_assert!(off < DIRECT_LIMIT, "offset outside BAR0");
        off
    }
}

/// The BAR offset `route` sends a directly addressable chip address to.
fn bar_offset(chip: u32) -> u32 {
    match crate::regs::route(chip) {
        crate::regs::Route::Direct(off) | crate::regs::Route::Fixed(off) => off,
        _ => u32::MAX,
    }
}

/// The queue whose `cpu_idx` doorbell lives at this BAR offset.
fn queue_of_cpu_idx(off: u32) -> Option<crate::dma::Queue> {
    [
        crate::dma::Queue::McuWm,
        crate::dma::Queue::Fwdl,
        crate::dma::Queue::EventWm,
    ]
    .into_iter()
    .find(|q| bar_offset(q.regs().cpu_idx()) == off)
}

impl MockMt7925 {
    /// The engine's side of `MT_WFDMA0_GLO_CFG`.
    ///
    /// The driver writes the configuration and the two enable bits; the
    /// **busy** bits are the engine's and are never the driver's to set. A
    /// real part raises them while it walks a ring and drops them when it
    /// stops, so an idle part reads them clear — unless `busy_stuck` says
    /// otherwise, which is how the timeout path is reached by a test.
    fn write_glo_cfg(&self, addr: u32, val: u32) {
        let busy = if self.busy_stuck.get() {
            crate::dma::GLO_CFG_BUSY
        } else {
            0
        };
        self.poke(addr, (val & !crate::dma::GLO_CFG_BUSY) | busy);
    }

    /// A doorbell: the driver's producer index, and **nothing else**.
    ///
    /// `dma_idx` deliberately does not follow. It is the engine's position
    /// — for a receive queue, how far it has *written received data*
    /// (`mt76_dma_rx_process`, `dma.c:1014-1025`), and for a transmit
    /// queue, how far it has consumed. Neither moves because the driver
    /// posted empty buffers.
    ///
    /// **An earlier version of this mock advanced `dma_idx` to whatever
    /// `cpu_idx` advertised**, which made a witness pass here and fail on
    /// metal — the model agreeing with the belief it was written from
    /// instead of checking it. `rx_stuck` is kept for the rung that will
    /// legitimately read a moved index (D1c, where a response arrives), so
    /// its timeout path still has a switch.
    fn kick_queue(&self, queue: crate::dma::Queue, addr: u32, val: u32) {
        self.poke(addr, val);
        // The one exception, opt-in and cited: a transmit engine consuming
        // what it was handed. See [`Self::tx_consumes`].
        if self.tx_consumes.get() && !matches!(queue, crate::dma::Queue::EventWm) {
            self.poke(self.cell_of(queue.regs().dma_idx()), val);
        }
    }
}

impl Default for MockMt7925 {
    fn default() -> Self {
        Self::new()
    }
}

impl Regs for MockMt7925 {
    fn read32(&self, off: usize) -> u32 {
        let off = off as u32;
        if off == MT_HIF_REMAP_L1 {
            return *self.l1.borrow();
        }
        if off == MT_HIF_REMAP_L2 {
            return *self.l2.borrow();
        }
        let addr = self.resolve(off);
        self.reads.borrow_mut().push(addr);
        self.peek(addr)
    }

    fn write32(&self, off: usize, val: u32) {
        let off = off as u32;
        if off == MT_HIF_REMAP_L1 {
            *self.l1.borrow_mut() = val;
            return;
        }
        if off == MT_HIF_REMAP_L2 {
            *self.l2.borrow_mut() = val;
            return;
        }
        // The engine's own registers are matched on the **BAR offset**, before
        // `resolve` picks a chip name for the cell. A directly addressable
        // register has two lawful chip names — its own and the fixed block
        // whose window covers it — and the driver writes the offset, so that
        // is what the model must compare.
        if off == bar_offset(MT_WFDMA0_GLO_CFG) {
            let addr = self.resolve(off);
            self.writes.borrow_mut().push((addr, val));
            self.write_glo_cfg(addr, val);
            return;
        }
        if let Some(bank) = queue_of_cpu_idx(off) {
            let addr = self.resolve(off);
            self.writes.borrow_mut().push((addr, val));
            self.kick_queue(bank, addr, val);
            return;
        }
        let addr = self.resolve(off);
        self.writes.borrow_mut().push((addr, val));
        if addr == MT_CONN_ON_LPCTL {
            self.write_lpctl(val);
            return;
        }
        self.poke(addr, val);
    }
}

/// A DMA buffer whose **device** address is inside this part's reach.
///
/// [`HostBuf`] reports its own host pointer as the IOVA, and on a 64-bit
/// host that is above 4 GiB — which [`crate::dma::addressable`] refuses, so
/// a test over `HostBuf` alone could not build a descriptor at all. The
/// shared host double keeps doing the allocation; this supplies the
/// translation, exactly as `rtw89`'s `IommuBuf` does
/// (`drivers/net/rtw89/src/mock.rs:55`).
///
/// It exists because the tests used to build rings out of **stack arrays**,
/// and a stack array is not memory a device writes behind your back. That
/// is precisely why a non-volatile completion poll looked correct here for
/// as long as it did.
pub struct ReachableBuf {
    inner: HostBuf,
    iova: u64,
    /// The length the buffer **reports**, which is what the caller asked
    /// for and not the page the host double rounded it up to. A pool slice
    /// of sixteen bytes is sixteen bytes to every bounds check that matters;
    /// reporting the page instead would quietly widen every one of them.
    len: usize,
}

impl Dma for ReachableBuf {
    fn cpu(&self) -> *mut u8 {
        self.inner.cpu()
    }
    fn iova(&self) -> IovaAddr {
        IovaAddr(self.iova)
    }
    fn len(&self) -> usize {
        self.len
    }
}

/// The device address the first buffer is handed: 1 GiB, which is
/// `NARROW_DMA_IOVA_BASE` — where the host really does place a pool for a
/// 32-bit master.
pub const REACHABLE_IOVA_BASE: u64 = 0x4000_0000;

/// A bump allocator over [`HostPool`] handing out reachable addresses.
pub struct ReachablePool {
    inner: HostPool,
    next: Cell<u64>,
}

impl Default for ReachablePool {
    fn default() -> Self {
        Self::new()
    }
}

impl ReachablePool {
    pub const fn new() -> Self {
        Self {
            inner: HostPool,
            next: Cell::new(REACHABLE_IOVA_BASE),
        }
    }

    /// Allocate `bytes`, rounded up by the host pool's own page rule.
    pub fn alloc(&self, bytes: usize) -> ReachableBuf {
        let inner = self.inner.alloc(bytes).expect("the host pool allocates");
        let iova = self.next.get();
        // The bump steps by the **backing**, so two buffers never share a
        // device address even though each reports only what was asked for.
        self.next.set(iova + inner.len() as u64);
        ReachableBuf {
            inner,
            iova,
            len: bytes,
        }
    }
}
