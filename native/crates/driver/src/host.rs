//! # `host` — the userland driver-process host (feature `service`)
//!
//! The generic **driver-process skeleton** (#101): it turns a task's
//! capability endowment — a device ([`role::DEVICE`](kore_abi::task::role::DEVICE))
//! and a DMA pool ([`role::DMA_POOL`](kore_abi::task::role::DMA_POOL)) — into the
//! [`seam`](crate::seam) a driver core brings up over, derives the device's
//! interrupt route, and tears it all down on drop. The production xHCI input
//! task is its first metal instance; NVMe, NIC, and IVPU adapters remain
//! future work. The cores (`kore-nvme`/`kore-xhci`) are generic over
//! [`Regs`]/[`DmaPool`], so one class-specific [`DeviceBinding`] can consume
//! the affine [`DeviceResources`], bring up its core, and return those
//! resources to the common host for containment at teardown.
//!
//! **Where it lives — the dependency graph.**  `kore-rt` and `kore-driver` both depend
//! only on `kore-abi`, not each other; `kore-rt` cannot name the seam traits and a
//! `kore-rt`-free `kore-driver` cannot name kore-rt's caps.  This module is the one place that
//! legitimately sees both — an *optional* `kore-rt` dep, off by default so the
//! kernel never pulls kore-rt (the discipline `kore-nvme`'s manifest already states).
//!
//! **Capability boundary.**  The device-cap verbs are the only route from an
//! endowed [`DeviceToken`] and [`SupplyToken`] to [`CapRegs`] and [`CapPool`].
//! A metal build must provide each verb honestly; an unavailable verb returns
//! its typed refusal rather than substituting host memory or a synthetic
//! device.  Driver-core host tests exercise the generic [`Regs`]/[`DmaPool`]
//! contracts separately.

#[cfg(test)]
#[path = "../../collections/src/testing.rs"]
mod test_arena;

extern crate alloc;

use alloc::rc::Rc;
use alloc::vec::Vec;
use core::alloc::Allocator;
#[cfg(test)]
use core::cell::Cell;
use core::cell::RefCell;
use core::marker::PhantomData;
use kore_collections::AtomicBitVec;

use kore_abi::device::IovaAddr;
use kore_abi::task::{Bootstrap, Profile, RoleSlot, role};

use kore_rt::cap::{
    BarWindow, DeviceToken, DmaWindow, FabricRegistrationToken, PlatformToken, QuiesceFault,
    Quiesced, RouteToken, SupplyToken, verbs,
};
use kore_rt::service::{
    BindError, Bundle, DeviceClass, PciBarMask, PciIdentitySelector, PciInterruptRequirement,
    PciMsixBlock, PciSelector, QuiesceSweep,
};

use crate::seam::{
    Dma, DmaPool, DmaReleaseFailure, Regs, Regs64, RegsNarrow, RegsRegion, RegsWindowError,
};

/// Page granularity — DMA buffers are page-aligned (rings and list pages
/// require 4 KiB alignment; the BAR window is page-aligned by the intake).
const PAGE: u64 = 4096;

/// Base of a driver's DMA device-address window.  The driver owns its IOMMU
/// domain, so it allocates its own IOVAs — mm is never told about device
/// addresses (epic #21: mm hands out physical-region tokens; the domain owner
/// decides where the device sees them).  Chosen well clear of the low
/// addresses firmware and legacy devices use.
///
/// Public because a driver that also maps *client* windows into the same
/// domain (the frame service's per-vif transmit windows) must place them
/// where this pool cannot reach, and that check belongs in a `const`
/// assertion rather than in a comment.
pub const DMA_IOVA_BASE: u64 = 0x1_0000_0000;

/// The two concrete device-root species accepted by the common driver host.
/// The enum is private so class bindings continue to see only realized
/// register/interrupt/DMA seams and cannot branch on bus provenance.
enum HostDeviceToken {
    Pci(DeviceToken),
    Platform(PlatformToken),
}

impl HostDeviceToken {
    fn map_registers(&self, index: u8) -> Result<BarWindow, verbs::BarMapFault> {
        match self {
            Self::Pci(device) => verbs::map_bar(device, index),
            Self::Platform(device) if index == 0 => verbs::map_platform_mmio(device),
            Self::Platform(_) => Err(verbs::BarMapFault::Geometry),
        }
    }

    fn establish_interrupts(&self, block: InterruptBlock) -> Option<RouteToken> {
        match self {
            Self::Pci(device) => {
                verbs::establish_interrupts(device, block.first_entry, block.count, block.affinity)
            }
            // A firmware-described line has immutable trigger/polarity and is
            // intrinsically a single route. Validation rejects other shapes.
            Self::Platform(device) if block == InterruptBlock::SINGLE => {
                verbs::establish_line(device)
            }
            Self::Platform(_) => None,
        }
    }

    fn dma_window(
        &self,
        pool: SupplyToken,
        len: u64,
        iova: u64,
    ) -> Result<(DmaWindow, Option<SupplyToken>), verbs::DmaMapFault> {
        match self {
            Self::Pci(device) => verbs::dma_window(pool, device, 0, len, iova),
            Self::Platform(device) => verbs::dma_window_platform(pool, device, 0, len, iova),
        }
    }

    fn import_window(
        &self,
        supply: DmaImport,
        iova: u64,
        device_write: bool,
    ) -> Result<DmaWindow, verbs::DmaMapFault> {
        match self {
            Self::Pci(device) => match supply {
                DmaImport::Lease(lease) => {
                    verbs::import_dma_window(lease, device, iova, device_write)
                }
                DmaImport::Route { route, bytes } => {
                    verbs::import_route_dma_window(route, device, iova, bytes, device_write)
                }
            },
            Self::Platform(_) => Err(verbs::DmaMapFault::Bounds),
        }
    }

    fn activate_window(&self, window: &DmaWindow) -> bool {
        match self {
            Self::Pci(device) => verbs::activate_device_window(device, window),
            Self::Platform(device) => verbs::activate_platform_window(device, window),
        }
    }

    /// The terminal counterpart of [`activate_window`](Self::activate_window):
    /// ask the kernel to prove this function quiet at the fabric.  The
    /// enforced set is the PCI class bindings (#809 §L.6); a platform root
    /// answers `Unenforced` until a rendezvous exists for it.
    fn prove_quiet(&self) -> Result<Quiesced, QuiesceFault> {
        match self {
            Self::Pci(device) => verbs::quiesce_device(device),
            Self::Platform(_) => Err(QuiesceFault::Unenforced),
        }
    }
}

/// Architectural PCI BAR count. Class declarations are validated against
/// this bound before any authority is claimed or mapping is attempted.
pub const PCI_BAR_COUNT: usize = 6;

/// Which PCI message-signalled source a device class can lawfully consume.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterruptSourcePolicy {
    /// Prefer MSI-X, but admit an MSI-only function through the same required
    /// interrupt-remapping transform.
    PreferMsiX,
    /// Require a kernel-mediated MSI-X table. An MSI-only function refuses
    /// instead of changing the device protocol's source semantics.
    RequireMsiX,
}

/// One consecutive MSI/MSI-X block requested by a device class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InterruptBlock {
    /// First MSI-X table entry, or zero for an MSI message block.
    pub first_entry: u16,
    /// Consecutive messages delivered through the resulting route.
    pub count: u16,
    /// Optional target CPU; `None` delegates placement to the kernel.
    pub affinity: Option<u32>,
    /// Whether an ordinary MSI source is semantically admissible.
    pub source: InterruptSourcePolicy,
}

impl InterruptBlock {
    /// One unpinned vector beginning at table entry zero.
    pub const SINGLE: Self = Self {
        first_entry: 0,
        count: 1,
        affinity: None,
        source: InterruptSourcePolicy::PreferMsiX,
    };

    /// One unpinned vector at MSI-X table entry zero, with no MSI fallback.
    pub const MSIX_SINGLE: Self = Self {
        first_entry: 0,
        count: 1,
        affinity: None,
        source: InterruptSourcePolicy::RequireMsiX,
    };

    /// Two unpinned consecutive MSI-X entries beginning at zero, with no MSI
    /// fallback.
    ///
    /// This is the smallest block for protocols that reserve entry zero for a
    /// control plane and entry one for a distinct completion plane. A single
    /// aggregate MSI message cannot preserve that assignment.
    pub const MSIX_PAIR: Self = Self {
        first_entry: 0,
        count: 2,
        affinity: None,
        source: InterruptSourcePolicy::RequireMsiX,
    };
}

/// Whether a class can operate without its declared interrupt block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterruptPolicy {
    /// This class does not consume an interrupt route.
    None,
    /// Prefer the declared block, but permit a polling-capable class to bind
    /// when the platform cannot realize it.
    Optional(InterruptBlock),
    /// Refuse the binding unless the declared block can be realized.
    Required(InterruptBlock),
}

/// Why a device intake refused.
///
/// Public since the claim path ([`BoundDevice::claim`]) has no `Bundle`
/// to fold it into: a claiming driver reads the refusal itself and decides
/// whether to release the device and ask for another candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntakeError {
    /// The boot block named no handle at this role/index.
    MissingRole { role: u16, index: u16 },
    /// The handle was present but did not possess as the expected kind.
    BadHandle { role: u16, index: u16 },
    /// The class declared a BAR index outside PCI geometry.
    InvalidBar(u8),
    /// The class declared one BAR index twice.
    DuplicateBar(u8),
    /// A declared BAR could not be mapped.
    BarUnavailable(u8),
    /// A declared BAR's map refused, stage-coded: `code[2:0]` the
    /// [`BarMapFault`](kore_rt::cap::verbs::BarMapFault) stage (1 slot
    /// exhaustion, 2/3 claim build/commit, 4/5 map build/commit, 6
    /// geometry), `code >> 3` the kernel status for the commit stages.
    BarMap { bar: u8, code: u8 },
    /// The declared vector block is not a non-zero power of two, or
    /// overflows the table.
    InvalidInterruptBlock,
    /// A required interrupt route could not be established.
    InterruptUnavailable,
    /// The DMA window could not be prepared or its completed graph could not
    /// be activated in the device's domain.
    DmaUnavailable,
    /// The DMA window refused, stage-coded: `code[2:0]` the
    /// [`DmaMapFault`](kore_rt::cap::verbs::DmaMapFault) stage (1 bounds,
    /// 2 slot exhaustion, 3 carve, 4 CPU-view map, 5 geometry, 6 device
    /// publish), `code >> 3` the kernel status for the commit stages.
    DmaMap { code: u8 },
    /// The class's own bring-up refused the realized resources — the
    /// controller is present and mappable but did not come up.
    ///
    /// The refusal a **claiming** driver acts on: it is the honest answer
    /// to "this candidate is not the controller I want", so a driver that
    /// sees it releases the device and asks the broker for the next
    /// match rather than failing its whole founding.
    ///
    /// Carries the class's own account of the refusal rather than discarding
    /// it. The two lifecycle halves fail for unrelated reasons — `Prepare`
    /// runs against a controller that cannot yet fetch DMA (reset handshake,
    /// queue allocation), `Start` against a live one (identify, namespace
    /// geometry) — and `code` is whatever [`DeviceBinding::refusal_code`]
    /// chose. On hardware that only reproduces on metal this is the whole
    /// diagnosis, so the seam must not flatten it.
    ClassBindRefused {
        /// Which class hook refused.
        at: ClassStage,
        /// The class's own refusal code; `0` when the class supplies none.
        code: u8,
    },
}

/// Which half of a class lifecycle refused, for [`IntakeError::ClassBindRefused`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClassStage {
    /// [`DeviceBinding::prepare`] — hardware cannot yet fetch DMA.
    Prepare,
    /// [`DeviceBinding::start`] — the DMA window is live.
    Start,
}

impl IntakeError {
    /// Stable, allocation-free hardware diagnostic. Low byte is the intake
    /// class; upper bytes retain the class-specific stage/status detail.
    pub const fn diagnostic_code(self) -> u64 {
        let (class, detail) = match self {
            Self::MissingRole { role, index } => (1, (role as u64) | ((index as u64) << 16)),
            Self::BadHandle { role, index } => (2, (role as u64) | ((index as u64) << 16)),
            Self::InvalidBar(bar) => (3, bar as u64),
            Self::DuplicateBar(bar) => (4, bar as u64),
            Self::BarUnavailable(bar) => (5, bar as u64),
            Self::BarMap { bar, code } => (6, (bar as u64) | ((code as u64) << 8)),
            Self::InvalidInterruptBlock => (7, 0),
            Self::InterruptUnavailable => (8, 0),
            Self::DmaUnavailable => (9, 0),
            Self::DmaMap { code } => (10, code as u64),
            Self::ClassBindRefused { at, code } => {
                let stage = match at {
                    ClassStage::Prepare => 1,
                    ClassStage::Start => 2,
                };
                (11, stage | ((code as u64) << 8))
            }
        };
        class | (detail << 8)
    }

    fn bind_error(self) -> BindError {
        match self {
            Self::MissingRole { role, index } => BindError::MissingRole { role, index },
            Self::BadHandle { role, index } => BindError::BadHandle { role, index },
            Self::InvalidBar(_)
            | Self::DuplicateBar(_)
            | Self::BarUnavailable(_)
            | Self::BarMap { .. }
            | Self::InvalidInterruptBlock
            | Self::InterruptUnavailable
            | Self::ClassBindRefused { .. } => BindError::BadHandle {
                role: role::DEVICE,
                index: 0,
            },
            Self::DmaUnavailable | Self::DmaMap { .. } => BindError::BadHandle {
                role: role::DMA_POOL,
                index: 0,
            },
        }
    }
}

/// Child-capability teardown outcomes, in dependency order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TeardownReport {
    /// Whether the userland interrupt-route capability was revoked, or no
    /// route existed.  This is an authority-containment result, not a claim
    /// that every physical endpoint/IRTE was reclaimed: kernel teardown
    /// deliberately quarantines those identities when hardware completion is
    /// uncertain while still closing the task's route authority.
    pub route: bool,
    /// Whether the device-visible DMA window was revoked.
    pub dma: bool,
    /// One bit per mapped BAR; set means the revoke completed.
    pub bars: u8,
    /// One bit per BAR the class declared and the intake mapped.
    pub expected_bars: u8,
}

impl TeardownReport {
    /// Whether every child declared by the binding was revoked.
    #[must_use]
    pub const fn complete(self) -> bool {
        self.route && self.dma && self.bars == self.expected_bars
    }
}

/// What a class did to its **own** controller in [`DeviceBinding::quiesce`]
/// — the class's claim, never a substitute for the fabric proof the host
/// mints after it.  `Ok(())` is deliberately not an option: a class that
/// stopped nothing must say so by name, so a stub reads as a declaration
/// rather than an accident, and a false `Stopped` still cannot forge quiet
/// (the kernel reads Bus Master Enable back, not the class).
#[must_use]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClassQuiesce {
    /// The class-owned engine was stopped: xHCI halted, NVMe `CC.SHN`
    /// complete, GuC / NPU reset or powered down.
    Stopped,
    /// The class held no started engine.  A *posture*, lawful only when
    /// `Bound` structurally holds none (parts still unbooted, a ladder never
    /// taken) — not a way to leave one running.
    NothingToStop,
}

/// Result of the class hook, the kernel's fabric proof, and capability
/// retirement — three facts from three authorities, kept apart.  Capability
/// revocation still runs when either of the first two refuses: containing a
/// device that may remain live is safer than retaining a silently
/// half-torn-down authority tree.
#[derive(Debug, PartialEq, Eq)]
pub struct BindingTeardown<E> {
    /// The class's account of its own engine.
    pub quiesce: Result<ClassQuiesce, E>,
    /// The kernel's proof that the function is quiet at the fabric (#809 §L)
    /// — interrupts masked, Bus Master Enable read back clear or the
    /// requester bound to Abort, the DMA epoch quarantined.  Minted by the
    /// host on the common resource set after the class has unbound, so no
    /// class can skip it.
    pub fabric: Result<Quiesced, QuiesceFault>,
    /// Result of the common host's unconditional capability containment.
    pub resources: TeardownReport,
}

impl<E> BindingTeardown<E> {
    /// Fold this outcome into a row's [`QuiesceSweep`].
    pub fn record(&self, sweep: &mut QuiesceSweep) {
        sweep.note(
            matches!(self.quiesce, Ok(ClassQuiesce::Stopped)),
            self.fabric.as_ref().ok().map(Quiesced::verdict),
            self.resources.complete(),
        );
    }
}

/// The [`DeviceClass`] citation for a binding class — `Binding` with the
/// class's own declared BAR mask, so the founder can assert it against the
/// row's selector and the two can never drift (#809 §L.5).
pub const fn device_class<B: DeviceBinding>() -> DeviceClass {
    DeviceClass::Binding {
        bars: <B::Pci as PciBindingContract>::BARS,
    }
}

/// Result of a temporary class probe that asks to reuse both affine roots
/// after complete child-first teardown.  A *non-terminal* transition: it
/// runs the class hook and containment only — no fabric proof, because the
/// recovered device is about to be offered to another candidate and must
/// not leave this boot quarantined.
#[derive(Debug)]
pub struct BindingRecovery<E> {
    /// The class's account of its own engine.
    pub quiesce: Result<ClassQuiesce, E>,
    /// Result of retiring the common host's derived children.
    pub resources: TeardownReport,
    /// The exact claimed device, recoverable only after complete closure.
    pub device: Option<DeviceToken>,
    /// The parent DMA supply, recoverable when it owned the mapped window.
    pub pool: Option<SupplyToken>,
}

/// **Volatile MMIO** over a mapped BAR window — the userland/real
/// [`Regs`] impl.  Every access is `read_volatile` / `write_volatile` at
/// `base + off` (never elided, reordered, or coalesced); the base is
/// page-aligned, so the core's naturally-aligned register offsets are sound.
pub struct CapRegs {
    bar: Option<BarWindow>,
    base: usize,
    len: u64,
}

impl CapRegs {
    /// Borrow the capability-backed BAR owner for a typed peer-aperture
    /// request (`verbs::peer_window`). Register access remains on `Regs`;
    /// this narrower escape is lifecycle-only and exposes no raw physical
    /// address.
    pub fn bar_window(&self) -> &BarWindow {
        self.bar.as_ref().expect("retired register window")
    }

    /// Adopt any capability-mapped volatile register aperture. PCI probes and
    /// firmware-described platform devices share this representation; the
    /// mapping token retains the authority that produced the window.
    pub fn from_window(window: BarWindow) -> Self {
        let base = window.base() as usize;
        let len = window.len();
        Self {
            bar: Some(window),
            base,
            len,
        }
    }

    /// Borrow an already capability-mapped aperture in this process.
    ///
    /// # Safety
    ///
    /// `base..base+len` must remain mapped for this view's lifetime. The
    /// process must retain the unique owning [`BarWindow`] and retire it only
    /// after every worker using this view has stopped.
    pub unsafe fn from_process_mapping(base: *mut u8, len: u64) -> Self {
        Self {
            bar: None,
            base: base as usize,
            len,
        }
    }

    #[inline]
    fn ptr(&self, off: usize, width: usize, align: usize) -> *mut u8 {
        assert!(off.is_multiple_of(align), "unaligned register access");
        let end = off.checked_add(width).expect("register offset overflow");
        assert!(
            end as u64 <= self.len,
            "register access past the BAR window"
        );
        // SAFETY: release-build bounds and alignment checks above establish
        // that `base + off` is a valid 32-bit MMIO location in the mapped BAR.
        unsafe { (self.base as *mut u8).add(off) }
    }

    fn retire(mut self) -> bool {
        self.bar.take().is_some_and(BarWindow::retire)
    }
}

impl Drop for CapRegs {
    fn drop(&mut self) {
        if let Some(bar) = self.bar.take() {
            let _ = bar.retire();
        }
    }
}

impl Regs for CapRegs {
    #[inline]
    fn read32(&self, off: usize) -> u32 {
        // SAFETY: MMIO read at an in-bounds, naturally-aligned offset.
        unsafe {
            (self.ptr(
                off,
                core::mem::size_of::<u32>(),
                core::mem::align_of::<u32>(),
            ) as *const u32)
                .read_volatile()
        }
    }

    #[inline]
    fn write32(&self, off: usize, val: u32) {
        // SAFETY: MMIO write at an in-bounds, naturally-aligned offset.
        unsafe {
            (self.ptr(
                off,
                core::mem::size_of::<u32>(),
                core::mem::align_of::<u32>(),
            ) as *mut u32)
                .write_volatile(val)
        }
    }
}

impl Regs64 for CapRegs {
    #[inline]
    fn read64_once(&self, off: usize) -> u64 {
        // SAFETY: the strong trait call selects a hardware-defined 64-bit
        // location; `ptr` proves mapped bounds and natural alignment.
        unsafe {
            (self.ptr(
                off,
                core::mem::size_of::<u64>(),
                core::mem::align_of::<u64>(),
            ) as *const u64)
                .read_volatile()
        }
    }

    #[inline]
    fn write64_once(&self, off: usize, val: u64) {
        // SAFETY: as above, one volatile u64 store is the required MMIO bus
        // transaction, not the `Regs` compatibility pair.
        unsafe {
            (self.ptr(
                off,
                core::mem::size_of::<u64>(),
                core::mem::align_of::<u64>(),
            ) as *mut u64)
                .write_volatile(val)
        }
    }
}

impl CapRegs {
    /// Mapped BAR byte length, used to bound class-specific discovery walks
    /// over device-reported register geometry.
    pub fn len(&self) -> usize {
        usize::try_from(self.len).unwrap_or(usize::MAX)
    }

    /// Base of this process mapping, for a bounded worker-local borrowed view.
    pub fn base(&self) -> *mut u8 {
        self.base as *mut u8
    }
}

/// The non-atomic shared owner underlying one binding thread's register views.
///
/// Kept generic so its affine rejoin rule and register forwarding can be
/// exercised without fabricating a hardware BAR capability in host tests.
struct SharedRegsOwner<R> {
    regs: Rc<R>,
}

impl<R> SharedRegsOwner<R> {
    fn new(regs: R) -> Self {
        Self {
            regs: Rc::new(regs),
        }
    }

    fn view(&self, region: RegsRegion) -> SharedRegsView<R> {
        SharedRegsView {
            regs: Rc::clone(&self.regs),
            region,
        }
    }

    fn try_rejoin(self) -> Result<R, Self> {
        match Rc::try_unwrap(self.regs) {
            Ok(regs) => Ok(regs),
            Err(regs) => Err(Self { regs }),
        }
    }
}

struct SharedRegsView<R> {
    regs: Rc<R>,
    region: RegsRegion,
}

impl<R> Clone for SharedRegsView<R> {
    fn clone(&self) -> Self {
        Self {
            regs: Rc::clone(&self.regs),
            region: self.region,
        }
    }
}

impl<R: Regs> Regs for SharedRegsView<R> {
    #[inline]
    fn read32(&self, off: usize) -> u32 {
        self.regs.read32(self.region.offset(
            off,
            core::mem::size_of::<u32>(),
            core::mem::align_of::<u32>(),
        ))
    }

    #[inline]
    fn write32(&self, off: usize, val: u32) {
        self.regs.write32(
            self.region.offset(
                off,
                core::mem::size_of::<u32>(),
                core::mem::align_of::<u32>(),
            ),
            val,
        );
    }
}

impl<R: Regs64> Regs64 for SharedRegsView<R> {
    #[inline]
    fn read64_once(&self, off: usize) -> u64 {
        self.regs.read64_once(self.region.offset(
            off,
            core::mem::size_of::<u64>(),
            core::mem::align_of::<u64>(),
        ))
    }

    #[inline]
    fn write64_once(&self, off: usize, val: u64) {
        self.regs.write64_once(
            self.region.offset(
                off,
                core::mem::size_of::<u64>(),
                core::mem::align_of::<u64>(),
            ),
            val,
        );
    }
}

/// One affine BAR owner that can issue checked, shareable register subviews.
///
/// `SharedBar` is intentionally not `Clone`: there remains exactly one handle
/// that can recover the underlying [`CapRegs`] for resource restoration. The
/// views use [`Rc`], not atomic reference counting, which preserves
/// [`BarWindow`]'s originating-thread ownership instead of implying that its
/// thread-bound capability slot may migrate through the process.
#[must_use = "dropping the shared BAR retires it after the last view is gone"]
pub struct SharedBar {
    inner: SharedRegsOwner<CapRegs>,
}

impl SharedBar {
    /// Convert the sole affine register capability into a shared-view owner.
    pub fn new(regs: CapRegs) -> Self {
        Self {
            inner: SharedRegsOwner::new(regs),
        }
    }

    /// Mapped BAR byte length.
    pub fn len(&self) -> usize {
        self.inner.regs.len()
    }

    /// Whether the mapped BAR is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Create a checked view over `[base, base + len)` in this BAR.
    ///
    /// The parent-relative base is consumed as construction geometry and is
    /// not exposed by the resulting view. All register calls are relative to
    /// byte zero of that view.
    pub fn view(&self, base: usize, len: usize) -> Result<SharedRegs, RegsWindowError> {
        let region = RegsRegion::new(self.len(), base, len)?;
        Ok(SharedRegs {
            inner: self.inner.view(region),
        })
    }

    /// Recover the affine BAR capability only after every view is gone.
    ///
    /// On failure the unchanged owner is returned, so a controller can drop
    /// its remaining views and retry without losing teardown authority.
    pub fn try_rejoin(self) -> Result<CapRegs, Self> {
        match self.inner.try_rejoin() {
            Ok(regs) => Ok(regs),
            Err(inner) => Err(Self { inner }),
        }
    }
}

/// A cloneable, checked register view into one thread-bound [`SharedBar`].
///
/// The view carries no independent BAR capability and exposes no parent BAR
/// offset. Cloning changes a non-atomic setup-time reference count; register
/// accesses add only checked rebasing before the underlying volatile access.
pub struct SharedRegs {
    inner: SharedRegsView<CapRegs>,
}

impl Clone for SharedRegs {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl SharedRegs {
    /// View-local byte length.
    pub fn len(&self) -> usize {
        self.inner.region.len()
    }

    /// Whether the view is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.region.is_empty()
    }
}

impl Regs for SharedRegs {
    #[inline]
    fn read32(&self, off: usize) -> u32 {
        self.inner.read32(off)
    }

    #[inline]
    fn write32(&self, off: usize, val: u32) {
        self.inner.write32(off, val);
    }
}

impl Regs64 for SharedRegs {
    #[inline]
    fn read64_once(&self, off: usize) -> u64 {
        self.inner.read64_once(off)
    }

    #[inline]
    fn write64_once(&self, off: usize, val: u64) {
        self.inner.write64_once(off, val);
    }
}

impl RegsNarrow for CapRegs {
    #[inline]
    fn read8(&self, off: usize) -> u8 {
        // SAFETY: MMIO read at an in-bounds offset; a byte is always aligned.
        unsafe { (self.ptr(off, 1, 1) as *const u8).read_volatile() }
    }

    #[inline]
    fn write8(&self, off: usize, val: u8) {
        // SAFETY: MMIO write at an in-bounds offset; a byte is always aligned.
        unsafe { self.ptr(off, 1, 1).write_volatile(val) }
    }

    #[inline]
    fn read16(&self, off: usize) -> u16 {
        // SAFETY: MMIO read at an in-bounds, naturally-aligned offset.
        unsafe {
            (self.ptr(
                off,
                core::mem::size_of::<u16>(),
                core::mem::align_of::<u16>(),
            ) as *const u16)
                .read_volatile()
        }
    }

    #[inline]
    fn write16(&self, off: usize, val: u16) {
        // SAFETY: MMIO write at an in-bounds, naturally-aligned offset.
        unsafe {
            (self.ptr(
                off,
                core::mem::size_of::<u16>(),
                core::mem::align_of::<u16>(),
            ) as *mut u16)
                .write_volatile(val)
        }
    }
}

impl<R: RegsNarrow> RegsNarrow for SharedRegsView<R> {
    #[inline]
    fn read8(&self, off: usize) -> u8 {
        self.regs.read8(self.region.offset(off, 1, 1))
    }

    #[inline]
    fn write8(&self, off: usize, val: u8) {
        self.regs.write8(self.region.offset(off, 1, 1), val);
    }

    #[inline]
    fn read16(&self, off: usize) -> u16 {
        self.regs.read16(self.region.offset(
            off,
            core::mem::size_of::<u16>(),
            core::mem::align_of::<u16>(),
        ))
    }

    #[inline]
    fn write16(&self, off: usize, val: u16) {
        self.regs.write16(
            self.region.offset(
                off,
                core::mem::size_of::<u16>(),
                core::mem::align_of::<u16>(),
            ),
            val,
        );
    }
}

impl RegsNarrow for SharedRegs {
    #[inline]
    fn read8(&self, off: usize) -> u8 {
        self.inner.read8(off)
    }

    #[inline]
    fn write8(&self, off: usize, val: u8) {
        self.inner.write8(off, val);
    }

    #[inline]
    fn read16(&self, off: usize) -> u16 {
        self.inner.read16(off)
    }

    #[inline]
    fn write16(&self, off: usize, val: u16) {
        self.inner.write16(off, val);
    }
}

/// A temporary, pool-free device-register probe.
///
/// Candidate selection may inspect a BAR before choosing which controller
/// receives the task's sole DMA supply. The probe owns the device token and
/// its one mapped child; [`finish`](Self::finish) returns the root only after
/// the child revoke proves no local BAR claim survives.
pub struct DeviceProbe {
    regs: Option<CapRegs>,
    device: Option<DeviceToken>,
    bar: u8,
}

/// A persistent, pool-free PCI register owner.
///
/// Small PIO controllers such as an I2C master need a BAR and an interrupt
/// route, but no device-visible memory.  Forcing them through
/// [`DeviceResources`] would manufacture a DMA requirement that their
/// protocol neither needs nor permits.  This owner is the corresponding
/// affine intake: one claimed device root, one mapped BAR child, and one
/// required MSI-family or explicitly requested firmware-line route, retired
/// child-first on every exit.
pub struct RegisterDevice {
    route: Option<RouteToken>,
    regs: Option<CapRegs>,
    device: Option<DeviceToken>,
    bar: u8,
}

/// A persistent, pool-free platform register owner.
///
/// This is the non-PCI sibling of [`RegisterDevice`]: one exact platform
/// root, its sole firmware-declared MMIO child, and its firmware line.  No BAR
/// number, MSI vocabulary, config-space assumption, or DMA supply enters the
/// intake.
pub struct PlatformRegisterDevice {
    route: Option<RouteToken>,
    regs: Option<CapRegs>,
    device: Option<PlatformToken>,
}

/// Observable containment result for [`RegisterDevice::teardown`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RegisterTeardown {
    /// The interrupt route was revoked.
    pub route: bool,
    /// The BAR mapping was revoked.
    pub bar: bool,
}

impl RegisterTeardown {
    /// Whether every derived child was retired.
    #[must_use]
    pub const fn complete(self) -> bool {
        self.route && self.bar
    }
}

impl RegisterDevice {
    /// Realize a BAR plus required interrupt route over an already-claimed
    /// PCI device, without asking for or constructing a DMA window.
    pub fn claim(
        device: DeviceToken,
        bar: u8,
        interrupt: InterruptBlock,
    ) -> Result<Self, IntakeError> {
        if usize::from(bar) >= PCI_BAR_COUNT {
            return Err(IntakeError::InvalidBar(bar));
        }
        if interrupt.count == 0
            || !interrupt.count.is_power_of_two()
            || interrupt.first_entry.checked_add(interrupt.count).is_none()
        {
            return Err(IntakeError::InvalidInterruptBlock);
        }

        Self::claim_with_route(device, bar, |device| {
            establish_interrupt_block(device, interrupt)
        })
    }

    /// Realize a PIO controller's BAR and native PCI completion line without
    /// a DMA supply or MSI requirement. The kernel resolves the initialized
    /// firmware route for this exact held device; no GSI or electrical
    /// configuration is accepted from the driver, and no fallback is implied.
    /// This PIO path requires the device's DMA lifecycle to remain inactive.
    pub fn claim_line(device: DeviceToken, bar: u8) -> Result<Self, IntakeError> {
        if usize::from(bar) >= PCI_BAR_COUNT {
            return Err(IntakeError::InvalidBar(bar));
        }
        Self::claim_with_route(device, bar, verbs::establish_pci_line)
    }

    fn claim_with_route(
        device: DeviceToken,
        bar: u8,
        establish: impl FnOnce(&DeviceToken) -> Option<RouteToken>,
    ) -> Result<Self, IntakeError> {
        let window = verbs::map_bar(&device, bar).map_err(|fault| IntakeError::BarMap {
            bar,
            code: fault.code(),
        })?;
        let regs = CapRegs::from_window(window);
        let Some(route) = establish(&device) else {
            drop(regs);
            return Err(IntakeError::InterruptUnavailable);
        };
        Ok(Self {
            route: Some(route),
            regs: Some(regs),
            device: Some(device),
            bar,
        })
    }

    /// The mapped register seam.
    pub fn regs(&self) -> &CapRegs {
        self.regs.as_ref().expect("register device already retired")
    }

    /// The route that paces this register device.
    pub fn route(&self) -> &RouteToken {
        self.route
            .as_ref()
            .expect("register device already retired")
    }

    /// Retire route then BAR and report both outcomes.
    pub fn teardown(mut self) -> RegisterTeardown {
        let route = self.route.take().is_some_and(verbs::release_route);
        let bar = self.regs.take().is_some_and(CapRegs::retire);
        drop(self.device.take());
        RegisterTeardown { route, bar }
    }

    /// The mapped BAR index, for diagnostics.
    pub const fn bar(&self) -> u8 {
        self.bar
    }
}

impl PlatformRegisterDevice {
    /// Realize the platform device's declared MMIO window and completion line.
    pub fn claim(device: PlatformToken) -> Result<Self, IntakeError> {
        let window = verbs::map_platform_mmio(&device).map_err(|fault| IntakeError::BarMap {
            bar: 0,
            code: fault.code(),
        })?;
        let regs = CapRegs::from_window(window);
        let Some(route) = verbs::establish_line(&device) else {
            drop(regs);
            return Err(IntakeError::InterruptUnavailable);
        };
        Ok(Self {
            route: Some(route),
            regs: Some(regs),
            device: Some(device),
        })
    }

    /// The mapped platform register seam.
    pub fn regs(&self) -> &CapRegs {
        self.regs
            .as_ref()
            .expect("platform register device retired")
    }

    /// The controller-completion route.
    pub fn route(&self) -> &RouteToken {
        self.route
            .as_ref()
            .expect("platform register device retired")
    }

    /// Retire route then MMIO mapping and root.
    pub fn teardown(mut self) -> RegisterTeardown {
        let route = self.route.take().is_some_and(verbs::release_route);
        let bar = self.regs.take().is_some_and(CapRegs::retire);
        drop(self.device.take());
        RegisterTeardown { route, bar }
    }
}

impl Drop for PlatformRegisterDevice {
    fn drop(&mut self) {
        if let Some(route) = self.route.take() {
            let _ = verbs::release_route(route);
        }
        if let Some(regs) = self.regs.take() {
            let _ = regs.retire();
        }
        drop(self.device.take());
    }
}

impl Drop for RegisterDevice {
    fn drop(&mut self) {
        if let Some(route) = self.route.take() {
            let _ = verbs::release_route(route);
        }
        if let Some(regs) = self.regs.take() {
            let _ = regs.retire();
        }
        drop(self.device.take());
    }
}

impl DeviceProbe {
    /// Map one BAR for class discovery without consuming a DMA pool.
    ///
    /// A mapping refusal consumes the root. `map_bar` can fail after deriving
    /// the BAR child, and its cleanup may itself be retryable; returning the
    /// root as reusable authority would therefore claim more teardown than
    /// the session proved. Dropping it here quarantines the broker row and
    /// lets this owning task reap whatever local subtree remains.
    pub fn open(device: DeviceToken, bar: u8) -> Result<Self, IntakeError> {
        let window = match verbs::map_bar(&device, bar) {
            Ok(window) => window,
            Err(fault) => {
                return Err(IntakeError::BarMap {
                    bar,
                    code: fault.code(),
                });
            }
        };
        Ok(Self {
            regs: Some(CapRegs::from_window(window)),
            device: Some(device),
            bar,
        })
    }

    /// The mapped register seam.
    pub fn regs(&self) -> &CapRegs {
        self.regs.as_ref().expect("device probe already finished")
    }

    /// Retire the temporary BAR child and recover the device root.
    ///
    /// A refused revoke consumes the probe and lets its device-root drop
    /// contain the local subtree. Continuing the candidate scan in that state
    /// could collide with a still-live BAR claim, so failure is explicit.
    pub fn finish(mut self) -> Result<DeviceToken, IntakeError> {
        let regs = self.regs.take().expect("device probe already finished");
        if !regs.retire() {
            return Err(IntakeError::BarUnavailable(self.bar));
        }
        Ok(self.device.take().expect("device probe lost its root"))
    }
}

impl Drop for DeviceProbe {
    fn drop(&mut self) {
        if let Some(regs) = self.regs.take() {
            let _ = regs.retire();
        }
    }
}

/// Pool-free register probe for a firmware-described platform device.
/// It mirrors [`DeviceProbe`] but maps the device's one exact MMIO aperture
/// instead of accepting a caller-selected PCI BAR number.
pub struct PlatformDeviceProbe {
    regs: Option<CapRegs>,
    device: Option<PlatformToken>,
}

impl PlatformDeviceProbe {
    /// Map the platform device's firmware-owned register aperture.
    pub fn open(device: PlatformToken) -> Result<Self, IntakeError> {
        let window = verbs::map_platform_mmio(&device).map_err(|fault| IntakeError::BarMap {
            bar: 0,
            code: fault.code(),
        })?;
        Ok(Self {
            regs: Some(CapRegs::from_window(window)),
            device: Some(device),
        })
    }

    /// The mapped register seam.
    pub fn regs(&self) -> &CapRegs {
        self.regs.as_ref().expect("platform probe already finished")
    }

    /// Retire the temporary MMIO child and recover the platform root.
    pub fn finish(mut self) -> Result<PlatformToken, IntakeError> {
        let regs = self.regs.take().expect("platform probe already finished");
        if !regs.retire() {
            return Err(IntakeError::BarUnavailable(0));
        }
        Ok(self.device.take().expect("platform probe lost its root"))
    }
}

impl Drop for PlatformDeviceProbe {
    fn drop(&mut self) {
        if let Some(regs) = self.regs.take() {
            let _ = regs.retire();
        }
    }
}

/// A DMA buffer handed out by [`CapPool`] — an exclusive allocation in the pool's
/// retained mapped window (a CPU pointer plus the device [`IovaAddr`] over the
/// *same* frames, zero-copy). The enclosing [`DeviceResources`] owns the window.
/// The allocation claim survives ordinary drop; only explicit quiescent release
/// makes its pages reusable.
pub struct CapBuf {
    cpu: *mut u8,
    iova: IovaAddr,
    len: usize,
    backing: CapBacking,
}

enum CapBacking {
    Pool {
        frames: Rc<AtomicBitVec<dyn Allocator>>,
        first_frame: usize,
    },
    Imported(Rc<RefCell<ImportedAllocation>>),
}

/// The resource shell and buffer jointly retain this allocation. Only an
/// explicitly quiescent buffer release hands it back for retirement at the
/// service boundary, after all stage-1 cleanup in that operation has returned.
struct ImportedAllocation {
    released: bool,
    return_requested: bool,
    returned: Option<kore_rt::cap::StashedMemory>,
    window: Option<DmaWindow>,
}

/// Validated import authority. A route lends mappings; it never becomes supply
/// capacity or gains a MemLease's physical release authority.
pub enum DmaImport {
    Lease(SupplyToken),
    Route {
        route: kore_rt::cap::FabricRouteToken,
        bytes: u64,
    },
}

impl DmaImport {
    pub fn claim(
        owner: kore_rt::cap::ReturnedCapability,
    ) -> Result<Self, kore_rt::cap::ReturnedCapability> {
        let required = kore_abi::cap::Rights::ACCESS
            | kore_abi::cap::Rights::MODIFY
            | kore_abi::cap::Rights::DERIVE
            | kore_abi::cap::Rights::REVOKE;
        if owner.is_lease_with(required) {
            return verbs::claim_returned_supply(owner).map(Self::Lease);
        }
        let route = owner.try_into_fabric_route(required)?;
        match route.describe() {
            Ok(info) if info.operation == 1 && info.access & 3 == 3 => Ok(Self::Route {
                route,
                bytes: info.bytes,
            }),
            _ => Err(route.into_returned_capability()),
        }
    }
    pub fn len(&self) -> u64 {
        match self {
            Self::Lease(lease) => lease.len(),
            Self::Route { bytes, .. } => *bytes,
        }
    }
    pub fn into_returned_capability(self) -> Result<kore_rt::cap::ReturnedCapability, Self> {
        match self {
            Self::Lease(lease) => lease.into_returned_capability().map_err(Self::Lease),
            Self::Route { route, .. } => Ok(route.into_returned_capability()),
        }
    }
}

/// Affine custody of a detached buffer awaiting the native serving boundary.
pub enum BufferReturn {
    Imported(ImportedReturn),
    Scoped(kore_rt::cap::StashedMemory),
}

pub struct ImportedReturn {
    owner: Rc<RefCell<ImportedAllocation>>,
}

impl BufferReturn {
    pub fn take(self) -> Result<kore_rt::cap::StashedMemory, Self> {
        match self {
            Self::Scoped(memory) => Ok(memory),
            Self::Imported(ticket) => {
                let returned = ticket.owner.borrow_mut().returned.take();
                match returned {
                    Some(memory) => Ok(memory),
                    None => Err(Self::Imported(ticket)),
                }
            }
        }
    }
}

/// Prepared before backend detachment. No mapping is transferred during
/// preparation; both scoped capabilities remain local until finish succeeds.
pub struct BufferExportPlan(ExportPlan);
enum ExportPlan {
    Imported(Rc<RefCell<ImportedAllocation>>),
    Pool {
        grant: kore_rt::cap::FabricRouteToken,
        root: kore_rt::cap::FabricRouteToken,
        frames: Rc<AtomicBitVec<dyn Allocator>>,
        first_frame: usize,
        bytes: usize,
    },
}

struct PoolLoan {
    buffer: CapBuf,
    root: Option<kore_rt::cap::FabricRouteToken>,
}

#[derive(Debug)]
pub enum BufferExportFault {
    InvalidBuffer,
    Capacity,
    Route(verbs::FabricRouteFault),
}

/// Reuse follows confirmed descendant retirement, never local handle drop.
fn retire_completed_loans<T, A: Allocator>(
    loans: &mut Vec<T, A>,
    mut retired: impl FnMut(&mut T) -> Result<bool, ()>,
    mut release: impl FnMut(T),
) -> bool {
    let mut index = 0;
    while index < loans.len() {
        match retired(&mut loans[index]) {
            Ok(true) => release(loans.swap_remove(index)),
            Ok(false) => index += 1,
            Err(()) => return false,
        }
    }
    true
}

/// Retire only backend-released allocations, retaining every refused owner.
/// All backend stage-1 cleanup has returned before this boundary is entered.
fn retire_imports_and_resume<T, A: Allocator>(
    imports: &mut Vec<T, A>,
    device_idle: bool,
    released: impl Fn(&T) -> bool,
    mut retire: impl FnMut(&T) -> bool,
    activate: impl FnOnce() -> bool,
) -> bool {
    // Exact IOVA retirement closes the entire requester's DMA latch. Keep
    // every detached owner until the backend proves its admissions quiescent.
    if !device_idle {
        return true;
    }
    let mut changed = false;
    let mut index = 0;
    while index < imports.len() {
        if !released(&imports[index]) {
            index += 1;
            continue;
        }
        if !retire(&imports[index]) {
            return false;
        }
        imports.remove(index);
        changed = true;
    }
    !changed || activate()
}

impl ImportedAllocation {
    fn retire(&mut self) -> bool {
        // The named IOVA child closes first. Keep registration and CPU-backed
        // allocation custody until device access is proved retired.
        if let Some(window) = self.window.as_mut() {
            if !window.try_retire_device().is_closed() {
                return false;
            }
        }
        if let Some(window) = self.window.take() {
            if self.return_requested {
                match window.try_into_memory() {
                    Ok(lease) => self.returned = Some(lease),
                    Err(window) => {
                        self.window = Some(window);
                        return false;
                    }
                }
            } else if let Err(window) = window.try_retire() {
                self.window = Some(window);
                return false;
            }
        }
        true
    }
}

impl Drop for ImportedAllocation {
    fn drop(&mut self) {
        if !self.retire() {
            core::mem::forget(self.window.take());
        }
    }
}

impl Dma for CapBuf {
    fn can_return_to_owner(&self) -> bool {
        matches!(self.backing, CapBacking::Imported(_))
    }

    fn cpu(&self) -> *mut u8 {
        self.cpu
    }

    fn iova(&self) -> IovaAddr {
        self.iova
    }

    fn len(&self) -> usize {
        self.len
    }
}

/// Page allocator over host-retained DMA windows. Allocation and explicit
/// quiescent release use the same bitmap claims as `PhysRegion`; no per-buffer
/// mapping or kernel round trip is needed. Ordinary drops keep pages claimed,
/// so an ambiguous device failure cannot turn quarantine into reuse.
pub struct CapPool {
    regions: Rc<RefCell<Vec<PoolRegion>>>,
    imports: Rc<RefCell<Vec<Rc<RefCell<ImportedAllocation>>>>>,
}

struct PoolRegion {
    cpu: *mut u8,
    iova: u64,
    len: u64,
    frames: Rc<AtomicBitVec<dyn Allocator>>,
}

impl PoolRegion {
    fn new(cpu: *mut u8, iova: u64, len: u64, allocator: &'static dyn Allocator) -> Option<Self> {
        if cpu.is_null()
            || iova % PAGE != 0
            || (cpu as usize as u64) % PAGE != 0
            || len == 0
            || len % PAGE != 0
        {
            return None;
        }
        iova.checked_add(len)?;
        (cpu as usize).checked_add(usize::try_from(len).ok()?)?;
        let pages = usize::try_from(len / PAGE).ok()?;
        let frames = AtomicBitVec::try_new_in(pages, allocator).ok()?;
        frames.release_run(0, pages);
        Some(Self {
            cpu,
            iova,
            len,
            frames: Rc::try_new(frames).ok()?,
        })
    }

    fn allocate(&self, bytes: usize, alignment: usize) -> Option<CapBuf> {
        let alignment = u64::try_from(alignment).ok()?;
        if alignment < PAGE || !alignment.is_power_of_two() {
            return None;
        }
        let mask = alignment - 1;
        if ((self.cpu as usize as u64) ^ self.iova) & mask != 0 {
            return None;
        }
        let size = u64::try_from(bytes).ok()?.max(1).checked_add(PAGE - 1)? & !(PAGE - 1);
        let pages = usize::try_from(size / PAGE).ok()?;
        let mut start = 0;
        loop {
            let first = self.frames.find_first_set_from(start)?;
            let candidate = self.iova.checked_add((first as u64).checked_mul(PAGE)?)?;
            let aligned = candidate.checked_add(mask)? & !mask;
            let off = aligned.checked_sub(self.iova)?;
            if off.checked_add(size)? > self.len {
                return None;
            }
            let first_frame = usize::try_from(off / PAGE).ok()?;
            if self.frames.try_claim_run(first_frame, pages).is_ok() {
                // SAFETY: the claimed extent is exclusive and within the live
                // window. Reused allocations must satisfy DmaPool's zeroing
                // contract just like their first allocation.
                let cpu = unsafe { self.cpu.add(off as usize) };
                unsafe { core::ptr::write_bytes(cpu, 0, size as usize) };
                return Some(CapBuf {
                    cpu,
                    iova: IovaAddr(aligned),
                    len: size as usize,
                    backing: CapBacking::Pool {
                        frames: Rc::clone(&self.frames),
                        first_frame,
                    },
                });
            }
            start = first_frame.checked_add(1)?;
        }
    }
}

impl CapPool {
    fn new(cpu: *mut u8, iova: u64, len: u64, allocator: &'static dyn Allocator) -> Option<Self> {
        let region = PoolRegion::new(cpu, iova, len, allocator)?;
        let mut regions = Vec::new();
        regions.try_reserve(1).ok()?;
        regions.push(region);
        Some(Self {
            regions: Rc::try_new(RefCell::new(regions)).ok()?,
            imports: Rc::try_new(RefCell::new(Vec::new())).ok()?,
        })
    }

    fn allocate(&self, bytes: usize, alignment: usize) -> Option<CapBuf> {
        self.regions
            .borrow()
            .iter()
            .find_map(|region| region.allocate(bytes, alignment))
    }

    /// Total admitted capacity across the retained DMA windows.
    pub fn capacity_bytes(&self) -> u64 {
        self.regions.borrow().iter().map(|region| region.len).sum()
    }
}

impl DmaPool for CapPool {
    type Buf = CapBuf;

    fn alloc(&self, bytes: usize) -> Option<CapBuf> {
        self.allocate(bytes, PAGE as usize)
    }

    fn alloc_aligned(&self, bytes: usize, alignment: usize) -> Option<CapBuf> {
        self.allocate(bytes, alignment)
    }

    fn owns(&self, buffer: &CapBuf) -> bool {
        match &buffer.backing {
            CapBacking::Pool { frames, .. } => self
                .regions
                .borrow()
                .iter()
                .any(|region| Rc::ptr_eq(&region.frames, frames)),
            CapBacking::Imported(owner) => {
                self.imports
                    .borrow()
                    .iter()
                    .any(|entry| Rc::ptr_eq(entry, owner))
                    && {
                        let allocation = owner.borrow();
                        allocation.window.is_some() && !allocation.released
                    }
            }
        }
    }

    unsafe fn release_quiesced(&self, buffer: CapBuf) -> Result<(), DmaReleaseFailure<CapBuf>> {
        match &buffer.backing {
            CapBacking::Pool {
                frames,
                first_frame,
            } => {
                let regions = self.regions.borrow();
                let Some(region) = regions
                    .iter()
                    .find(|region| Rc::ptr_eq(&region.frames, frames))
                else {
                    return Err(DmaReleaseFailure::Refused(buffer));
                };
                region
                    .frames
                    .release_run(*first_frame, buffer.len / PAGE as usize);
                Ok(())
            }
            CapBacking::Imported(owner) => {
                let imports = self.imports.borrow();
                if !imports.iter().any(|entry| Rc::ptr_eq(entry, owner)) {
                    return Err(DmaReleaseFailure::Refused(buffer));
                };
                // Stage-1 cleanup can continue for sibling buffers in this
                // same operation. Do not close BME mid-cleanup: hand the
                // allocation to the resource owner for boundary retirement.
                if owner.borrow().released {
                    return Err(DmaReleaseFailure::Refused(buffer));
                }
                owner.borrow_mut().released = true;
                Ok(())
            }
        }
    }
}

/// DMA exists in exactly two host-visible states. Preparation may allocate and
/// program device pointers but cannot start hardware; only the generic host can
/// consume the prepared state into the active one after kernel publication.
pub enum DmaPrepared {}

/// Proof carried by a resource set whose complete DMA graph was published to
/// the device domain before the class start hook ran.
pub enum DmaActive {}

/// The affine capability owner behind a movable [`CapPool`] allocator. Keeping
/// the tokens here lets an owning controller take the allocator while the generic
/// host still controls publication and final containment.
struct DmaOwner {
    window: Option<DmaWindow>,
    extensions: Vec<(DmaWindow, Option<SupplyToken>)>,
    regions: Rc<RefCell<Vec<PoolRegion>>>,
    imports: Rc<RefCell<Vec<Rc<RefCell<ImportedAllocation>>>>>,
    loans: RefCell<Vec<PoolLoan>>,
    /// A founder-delivered pool remains the parent of the derived DMA
    /// window. A provisioner-returned sublease is itself consumed as the
    /// window owner, so there is no separate parent token to retain.
    _pool: Option<SupplyToken>,
    device: Option<HostDeviceToken>,
}

impl DmaOwner {
    fn activate(&mut self) -> bool {
        let device = self.device.as_ref().expect("retired DMA device owner");
        let window = self.window.as_ref().expect("retired DMA owner");
        device.activate_window(window)
    }

    /// Remote descendants retain the claim. Only a proven close/retire permits
    /// bitmap reuse; Busy is ordinary outstanding custody, not device loss.
    fn retire_pool_loans(&self) -> bool {
        retire_completed_loans(
            &mut self.loans.borrow_mut(),
            |loan| {
                match loan.root.as_ref().unwrap().try_close() {
                    Err(verbs::FabricRouteFault::Busy | verbs::FabricRouteFault::Commit(4 | 6)) => {
                        return Ok(false);
                    }
                    Err(_) => return Err(()),
                    Ok(()) => {}
                }
                if let Err(root) = loan.root.take().unwrap().try_retire() {
                    loan.root = Some(root);
                    return Ok(false);
                }
                Ok(true)
            },
            |loan| {
                if let CapBacking::Pool {
                    frames,
                    first_frame,
                } = &loan.buffer.backing
                {
                    frames.release_run(*first_frame, loan.buffer.len / PAGE as usize);
                }
            },
        )
    }

    /// A refused mapping/route close must retain the entire custody chain.
    fn retire_window(window: DmaWindow) -> bool {
        match window.try_retire() {
            Ok(_) => true,
            Err(window) => {
                core::mem::forget(window);
                false
            }
        }
    }

    fn retire_primary(&mut self) -> bool {
        let retired = self
            .window
            .take()
            .is_some_and(|window| Self::retire_window(window));
        if !retired {
            core::mem::forget(self._pool.take());
        }
        retired
    }

    fn retire_extensions(&mut self) -> bool {
        let mut complete = self.retire_pool_loans() && self.loans.borrow().is_empty();
        self.imports.borrow_mut().retain(|owner| {
            let retired = owner.borrow_mut().retire();
            complete &= retired;
            !retired
        });
        for (window, parent) in self.extensions.drain(..) {
            let retired = Self::retire_window(window);
            complete &= retired;
            if !retired {
                core::mem::forget(parent);
            }
        }
        complete
    }

    fn retire(mut self) -> bool {
        let extensions = self.retire_extensions();
        self.retire_primary() && extensions
    }

    /// Retire the DMA child while preserving its roots only after that
    /// retirement completed. A failed revoke keeps neither root reusable.
    fn recover(mut self) -> (bool, Option<DeviceToken>, Option<SupplyToken>) {
        let extensions = self.retire_extensions();
        let retired = self.retire_primary() && extensions;
        if !retired {
            return (false, None, None);
        }
        match self.device.take() {
            Some(HostDeviceToken::Pci(device)) => (true, Some(device), self._pool.take()),
            // Platform roots are not part of the temporary PCI-candidate
            // recovery protocol. Contain one rather than pretending the
            // caller can safely reuse a capability of a different species.
            Some(HostDeviceToken::Platform(device)) => {
                drop(device);
                (true, None, None)
            }
            None => (false, None, None),
        }
    }
}

impl Drop for DmaOwner {
    fn drop(&mut self) {
        let _ = self.retire_extensions();
        if self.window.is_some() {
            let _ = self.retire_primary();
        }
    }
}

/// The common, bounded resource set produced from a class declaration. It owns
/// every derived capability child but no driver core. A class may temporarily
/// move route/BAR/pool seams into an owning controller during preparation, but the
/// resource set retains the DMA capability owner so the host alone controls
/// activation; [`DeviceBinding::unbind`] reconstructs it for containment.
pub struct DeviceResources<S = DmaActive> {
    // Declaration order is the bind-failure/implicit-drop fallback order.
    route: Option<RouteToken>,
    expected_route: bool,
    dma: Option<DmaOwner>,
    pool: Option<CapPool>,
    bars: [Option<CapRegs>; PCI_BAR_COUNT],
    expected_bars: u8,
    _state: PhantomData<S>,
}

impl<S> DeviceResources<S> {
    /// Borrow a mapped BAR from the class-declared set.
    pub fn bar(&self, index: u8) -> Option<&CapRegs> {
        self.bars.get(index as usize)?.as_ref()
    }

    /// Move one mapped BAR into a controller-owned seam. `None` means either
    /// the BAR was not declared or the class already took it.
    pub fn take_bar(&mut self, index: u8) -> Option<CapRegs> {
        self.bars.get_mut(index as usize)?.take()
    }

    /// Move one mapped BAR into an originating-thread shared-view owner.
    pub fn take_shared_bar(&mut self, index: u8) -> Option<SharedBar> {
        self.take_bar(index).map(SharedBar::new)
    }

    /// Restore a controller-owned BAR before returning the resource set from
    /// [`DeviceBinding::unbind`]. A wrong declaration or double restore is a
    /// class implementation bug, not a hardware refusal.
    pub fn restore_bar(&mut self, index: u8, bar: CapRegs) {
        let bit = 1u8
            .checked_shl(index as u32)
            .expect("BAR index outside declaration bitmap");
        assert!(self.expected_bars & bit != 0, "restored undeclared BAR");
        let slot = self
            .bars
            .get_mut(index as usize)
            .expect("restored BAR outside PCI geometry");
        assert!(slot.is_none(), "restored an occupied BAR slot");
        *slot = Some(bar);
    }

    /// Restore a shared BAR only after every register view has been dropped.
    ///
    /// Outstanding views return the unchanged affine owner and leave the
    /// resource slot untouched, allowing the class to quiesce those users and
    /// retry before [`DeviceBinding::unbind`] returns.
    pub fn restore_shared_bar(&mut self, index: u8, bar: SharedBar) -> Result<(), SharedBar> {
        let bar = bar.try_rejoin()?;
        self.restore_bar(index, bar);
        Ok(())
    }

    /// Borrow the binding's device-visible DMA pool while it remains in the
    /// common resource set.
    pub fn pool(&self) -> Option<&CapPool> {
        self.pool.as_ref()
    }

    /// Move the DMA pool into a controller-owned seam.
    pub fn take_pool(&mut self) -> Option<CapPool> {
        self.pool.take()
    }

    /// Restore a controller-owned DMA pool before returning the resource set
    /// from [`DeviceBinding::unbind`].
    pub fn restore_pool(&mut self, pool: CapPool) {
        assert!(self.pool.is_none(), "restored an occupied DMA pool slot");
        self.pool = Some(pool);
    }

    /// Borrow the PCI device capability the DMA graph was published against.
    ///
    /// **Lent, never moved**: the resource set owns it for the binding's whole
    /// life, and containment needs it. What a borrow is for is naming this
    /// device as the *target* of a further domain install — a client's window
    /// shared into the same IOMMU domain the driver's own pool lives in
    /// ([`verbs::share_window`](kore_rt::cap::verbs::share_window)). That is
    /// no new authority: the driver task already holds the device, and until
    /// now simply had no way to say so.
    ///
    /// A platform-device binding has no PCI token and returns `None`; the
    /// private host enum remains unobservable to class implementations.
    pub fn device(&self) -> Option<&DeviceToken> {
        match self.dma.as_ref()?.device.as_ref()? {
            HostDeviceToken::Pci(device) => Some(device),
            HostDeviceToken::Platform(_) => None,
        }
    }

    /// Borrow the realized interrupt route, when the class policy has one.
    pub fn route(&self) -> Option<&RouteToken> {
        self.route.as_ref()
    }

    /// Move the realized interrupt route into a controller-owned IRQ seam.
    ///
    /// The resource set remembers whether intake actually established a route,
    /// so teardown cannot report a false green if the class fails to restore
    /// the affine token during [`DeviceBinding::unbind`].
    pub fn take_route(&mut self) -> Option<RouteToken> {
        self.route.take()
    }

    /// Restore a controller-owned interrupt route before returning the resource
    /// set from [`DeviceBinding::unbind`].
    ///
    /// Restoring into a binding for which intake established no route, or
    /// restoring twice, is a class implementation bug.
    pub fn restore_route(&mut self, route: RouteToken) {
        assert!(
            self.expected_route,
            "restored an undeclared/unrealized route"
        );
        assert!(self.route.is_none(), "restored an occupied route slot");
        self.route = Some(route);
    }

    fn retire_parts(&mut self) -> TeardownReport {
        let route = match (self.expected_route, self.route.take()) {
            (false, None) => true,
            (true, Some(route)) => verbs::release_route(route),
            // A class moved the affine route into its controller and failed to
            // reconstruct the common owner. Do not call that containment.
            (true, None) => false,
            // No lawful path can populate an unrealized route slot, but still
            // retire the capability if an internal bug reaches this state.
            (false, Some(route)) => {
                let _ = verbs::release_route(route);
                false
            }
        };
        let dma = self.dma.take().map(DmaOwner::retire).unwrap_or(false);
        let mut bars = 0u8;
        for (index, bar) in self.bars.iter_mut().enumerate() {
            let bar = bar.take();
            if bar.is_some_and(CapRegs::retire) {
                bars |= 1 << index;
            }
        }
        TeardownReport {
            route,
            dma,
            bars,
            expected_bars: self.expected_bars,
        }
    }
}

impl DeviceResources<DmaPrepared> {
    /// Publish the complete prepared DMA window, consuming the only state from
    /// which a class may allocate/program pointers into the state accepted by
    /// its start hook. On refusal, the prepared owner rides back for cleanup.
    fn activate(mut self) -> Result<DeviceResources<DmaActive>, Self> {
        let Some(dma) = self.dma.as_mut() else {
            return Err(self);
        };
        if !dma.activate() {
            return Err(self);
        }
        Ok(self.into_active())
    }

    fn into_active(mut self) -> DeviceResources<DmaActive> {
        DeviceResources {
            route: self.route.take(),
            expected_route: self.expected_route,
            dma: self.dma.take(),
            pool: self.pool.take(),
            bars: core::array::from_fn(|index| self.bars[index].take()),
            expected_bars: self.expected_bars,
            _state: PhantomData,
        }
    }
}

impl DeviceResources<DmaActive> {
    /// Retire backend-released imports, then restore surviving windows before
    /// serving more work. Only a
    /// healthy driver epoch may call this; terminal teardown never reactivates
    /// the requester. `device_idle` must reflect all backend admissions, not
    /// only the released buffer. False defers retirement without dropping any
    /// owner. A false return requires containment of the entire epoch.
    pub fn resume_retired_dma(&self, device_idle: bool) -> bool {
        let Some(dma) = self.dma.as_ref() else {
            return false;
        };
        let mut imports = dma.imports.borrow_mut();
        let healthy = retire_imports_and_resume(
            &mut imports,
            device_idle,
            |owner| owner.borrow().released,
            |owner| owner.borrow_mut().retire(),
            || match (dma.device.as_ref(), dma.window.as_ref()) {
                (Some(device), Some(window)) => device.activate_window(window),
                _ => false,
            },
        );
        healthy && dma.retire_pool_loans()
    }

    /// Reserve loan custody before the backend removes its stage-1 mapping.
    /// Call only through the backend's guarded detach preparation callback.
    pub fn prepare_buffer_export(
        &self,
        buffer: &CapBuf,
    ) -> Result<BufferExportPlan, BufferExportFault> {
        let dma = self.dma.as_ref().ok_or(BufferExportFault::InvalidBuffer)?;
        match &buffer.backing {
            CapBacking::Imported(owner) => {
                if !dma
                    .imports
                    .borrow()
                    .iter()
                    .any(|entry| Rc::ptr_eq(entry, owner))
                    || owner.borrow().released
                {
                    return Err(BufferExportFault::InvalidBuffer);
                }
                Ok(BufferExportPlan(ExportPlan::Imported(Rc::clone(owner))))
            }
            CapBacking::Pool {
                frames,
                first_frame,
            } => {
                if !dma
                    .regions
                    .borrow()
                    .iter()
                    .any(|region| Rc::ptr_eq(&region.frames, frames))
                {
                    return Err(BufferExportFault::InvalidBuffer);
                }
                dma.loans
                    .borrow_mut()
                    .try_reserve(1)
                    .map_err(|_| BufferExportFault::Capacity)?;
                let end = buffer
                    .iova
                    .0
                    .checked_add(buffer.len as u64)
                    .ok_or(BufferExportFault::InvalidBuffer)?;
                let window = dma
                    .window
                    .iter()
                    .chain(dma.extensions.iter().map(|entry| &entry.0))
                    .find(|window| {
                        buffer.iova.0 >= window.iova()
                            && window
                                .iova()
                                .checked_add(window.len())
                                .is_some_and(|limit| end <= limit)
                    })
                    .ok_or(BufferExportFault::InvalidBuffer)?;
                let request = kore_abi::session::FabricRouteScopeReq {
                    offset: (buffer.iova.0 - window.iova()).into(),
                    bytes: (buffer.len as u64).into(),
                    access: 7,
                    reserved: [0; 7],
                };
                let root = window
                    .fabric_route()
                    .ok_or(BufferExportFault::InvalidBuffer)?
                    .scope(&request)
                    .map_err(BufferExportFault::Route)?;
                let grant = root
                    .scope(&kore_abi::session::FabricRouteScopeReq {
                        offset: 0.into(),
                        ..request
                    })
                    .map_err(BufferExportFault::Route)?;
                Ok(BufferExportPlan(ExportPlan::Pool {
                    grant,
                    root,
                    frames: Rc::clone(frames),
                    first_frame: *first_frame,
                    bytes: buffer.len,
                }))
            }
        }
    }

    /// Commit a prepared export after backend in-flight exclusion and stage-1
    /// invalidation. No kernel publication or allocation occurs in this step.
    ///
    /// # Safety
    /// The backend must have quiesced and detached this exact buffer.
    pub unsafe fn finish_buffer_export(
        &self,
        buffer: CapBuf,
        plan: BufferExportPlan,
    ) -> Result<BufferReturn, CapBuf> {
        let Some(dma) = self.dma.as_ref() else {
            return Err(buffer);
        };
        match (plan.0, &buffer.backing) {
            (ExportPlan::Imported(expected), CapBacking::Imported(owner))
                if Rc::ptr_eq(&expected, owner)
                    && !owner.borrow().released
                    && dma
                        .imports
                        .borrow()
                        .iter()
                        .any(|entry| Rc::ptr_eq(entry, owner)) =>
            {
                let ticket = ImportedReturn {
                    owner: Rc::clone(owner),
                };
                let mut allocation = owner.borrow_mut();
                allocation.return_requested = true;
                allocation.released = true;
                Ok(BufferReturn::Imported(ticket))
            }
            (
                ExportPlan::Pool {
                    grant,
                    root,
                    frames,
                    first_frame,
                    bytes,
                },
                CapBacking::Pool {
                    frames: actual,
                    first_frame: actual_first,
                },
            ) if Rc::ptr_eq(&frames, actual)
                && first_frame == *actual_first
                && bytes == buffer.len
                && dma
                    .regions
                    .borrow()
                    .iter()
                    .any(|region| Rc::ptr_eq(&region.frames, actual)) =>
            {
                let mut loans = dma.loans.borrow_mut();
                if loans.len() == loans.capacity() {
                    return Err(buffer);
                }
                loans.push(PoolLoan {
                    buffer,
                    root: Some(root),
                });
                Ok(BufferReturn::Scoped(kore_rt::cap::StashedMemory::Route(
                    grant,
                )))
            }
            _ => Err(buffer),
        }
    }

    /// Borrow the registration covering an exact range in a published DMA
    /// window, plus its domain-relative offset. This does not prove an
    /// allocation is live or authorize a transfer; the buffer owner must retain
    /// that separate authority. Failed, unpublished growth is never returned.
    pub fn fabric_registration(
        &self,
        iova: u64,
        bytes: u64,
    ) -> Option<(&FabricRegistrationToken, u64)> {
        let end = iova.checked_add(bytes)?;
        if bytes == 0 {
            return None;
        }
        let dma = self.dma.as_ref()?;
        if !dma.regions.borrow().iter().any(|region| {
            iova >= region.iova
                && region
                    .iova
                    .checked_add(region.len)
                    .is_some_and(|limit| end <= limit)
        }) {
            return None;
        }
        let covers = |window: &DmaWindow| {
            iova >= window.iova()
                && window
                    .iova()
                    .checked_add(window.len())
                    .is_some_and(|limit| end <= limit)
        };
        if let Some(window) = dma.window.as_ref().filter(|window| covers(window)) {
            return Some((window.fabric_registration()?, iova - window.iova()));
        }
        dma.extensions.iter().find_map(|(window, _)| {
            covers(window)
                .then(|| {
                    window
                        .fabric_registration()
                        .map(|reg| (reg, iova - window.iova()))
                })
                .flatten()
        })
    }

    /// Add a supplied extent to this device's allocator, even when the class
    /// has moved the pool into its controller. The domain owner chooses an IOVA
    /// after every retained window; MM supplies backing, never device addresses.
    ///
    /// The class must call this at a quiescent boundary. Any refusal requires
    /// containment before restarting: mapping/activation can fail after a kernel
    /// session has changed device translation state. No new allocations become
    /// visible until their affine window owner is retained and activated.
    pub fn grow_dma(&mut self, supply: SupplyToken) -> Result<u64, IntakeError> {
        let dma = self.dma.as_mut().ok_or(IntakeError::DmaUnavailable)?;
        let device = dma.device.as_ref().ok_or(IntakeError::DmaUnavailable)?;
        let len = supply.len();
        let mut regions = dma.regions.borrow_mut();
        let end = dma
            .window
            .iter()
            .chain(dma.extensions.iter().map(|(window, _)| window))
            .map(|window| window.iova() + window.len())
            .max()
            .ok_or(IntakeError::DmaUnavailable)?;
        let end = dma.imports.borrow().iter().fold(end, |end, owner| {
            owner
                .borrow()
                .window
                .as_ref()
                .map_or(end, |window| end.max(window.iova() + window.len()))
        });
        let iova = end
            .checked_add((2 << 20) - 1)
            .ok_or(IntakeError::DmaUnavailable)?
            & !((2 << 20) - 1);
        iova.checked_add(len).ok_or(IntakeError::DmaUnavailable)?;
        regions
            .try_reserve(1)
            .map_err(|_| IntakeError::DmaUnavailable)?;
        dma.extensions
            .try_reserve(1)
            .map_err(|_| IntakeError::DmaUnavailable)?;
        let (window, parent) = device
            .dma_window(supply, len, iova)
            .map_err(|fault| IntakeError::DmaMap { code: fault.code() })?;
        // Retain the full owner before any later fallible step. On error it
        // remains in the containment graph, even though no allocator can use it.
        dma.extensions.push((window, parent));
        let (window, _) = dma.extensions.last_mut().unwrap();
        let region = PoolRegion::new(
            window.cpu(),
            window.iova(),
            window.len(),
            &alloc::alloc::Global,
        )
        .ok_or(IntakeError::DmaUnavailable)?;
        if !device.activate_window(window) {
            return Err(IntakeError::DmaUnavailable);
        }
        regions.push(region);
        Ok(len)
    }

    /// Map an authenticated whole returned lease as one allocation. Unlike
    /// grow_dma, this preserves its bytes and never adds allocator capacity.
    /// `device_write` is the consumer's admitted access, not the CPU lease's
    /// mapping rights. Any publication failure requires device-epoch containment.
    pub fn import_dma(
        &mut self,
        supply: DmaImport,
        device_write: bool,
    ) -> Result<CapBuf, IntakeError> {
        if matches!(&supply, DmaImport::Lease(lease) if !lease.is_returned()) {
            return Err(IntakeError::DmaUnavailable);
        }
        let dma = self.dma.as_mut().ok_or(IntakeError::DmaUnavailable)?;
        let device = dma.device.as_ref().ok_or(IntakeError::DmaUnavailable)?;
        let len = supply.len();
        let bytes = usize::try_from(len).map_err(|_| IntakeError::DmaUnavailable)?;
        if len == 0 || len % PAGE != 0 {
            return Err(IntakeError::DmaUnavailable);
        }
        let mut imports = dma.imports.borrow_mut();
        if imports.len() >= 256 {
            return Err(IntakeError::DmaUnavailable);
        }
        imports
            .try_reserve(1)
            .map_err(|_| IntakeError::DmaUnavailable)?;
        // Allocate the retention cell before publishing any device mapping.
        let owner = Rc::try_new(RefCell::new(ImportedAllocation {
            released: false,
            return_requested: false,
            returned: None,
            window: None,
        }))
        .map_err(|_| IntakeError::DmaUnavailable)?;
        let mut end = dma
            .window
            .iter()
            .chain(dma.extensions.iter().map(|entry| &entry.0))
            .map(|window| window.iova() + window.len())
            .max()
            .ok_or(IntakeError::DmaUnavailable)?;
        for imported in imports.iter() {
            if let Some(window) = imported.borrow().window.as_ref() {
                end = end.max(
                    window
                        .iova()
                        .checked_add(window.len())
                        .ok_or(IntakeError::DmaUnavailable)?,
                );
            }
        }
        let iova = end
            .checked_add((2 << 20) - 1)
            .ok_or(IntakeError::DmaUnavailable)?
            & !((2 << 20) - 1);
        iova.checked_add(len).ok_or(IntakeError::DmaUnavailable)?;
        let window = device
            .import_window(supply, iova, device_write)
            .map_err(|fault| IntakeError::DmaMap { code: fault.code() })?;
        let cpu = window.cpu();
        owner.borrow_mut().window = Some(window);
        imports.push(Rc::clone(&owner));
        {
            let retained = owner.borrow();
            if !device.activate_window(retained.window.as_ref().unwrap()) {
                return Err(IntakeError::DmaUnavailable);
            }
        }
        Ok(CapBuf {
            cpu,
            iova: IovaAddr(iova),
            len: bytes,
            backing: CapBacking::Imported(owner),
        })
    }

    /// Ask the kernel to prove this device quiet at the fabric (#809 §L).
    /// Runs on the common resource set the host owns — the same structural
    /// trick that keeps DMA activation with the host — so a class cannot
    /// reach past it; and *before* [`retire`](Self::retire), so the latch
    /// closes while every translation still stands (the rendezvous rule).
    fn prove_quiet(&self) -> Result<Quiesced, QuiesceFault> {
        match self.dma.as_ref().and_then(|dma| dma.device.as_ref()) {
            Some(device) => device.prove_quiet(),
            None => Err(QuiesceFault::Unenforced),
        }
    }

    /// Retire every derived capability child after the class has quiesced.
    /// Route first closes new interrupt delivery, DMA second disables/removes
    /// device access, then BAR mappings disappear.
    fn retire(mut self) -> TeardownReport {
        self.retire_parts()
    }

    /// Retire every derived child and recover the device plus retained DMA
    /// parent only when the complete declared graph closed.
    fn recover(mut self) -> (TeardownReport, Option<DeviceToken>, Option<SupplyToken>) {
        let route = match (self.expected_route, self.route.take()) {
            (false, None) => true,
            (true, Some(route)) => verbs::release_route(route),
            (true, None) => false,
            (false, Some(route)) => {
                let _ = verbs::release_route(route);
                false
            }
        };
        let (dma, mut device, mut pool) = self
            .dma
            .take()
            .map(DmaOwner::recover)
            .unwrap_or((false, None, None));
        let mut bars = 0u8;
        for (index, bar) in self.bars.iter_mut().enumerate() {
            if bar.take().is_some_and(CapRegs::retire) {
                bars |= 1 << index;
            }
        }
        let report = TeardownReport {
            route,
            dma,
            bars,
            expected_bars: self.expected_bars,
        };
        if !report.complete() {
            drop(device.take());
            drop(pool.take());
        }
        (report, device, pool)
    }
}

impl<S> Drop for DeviceResources<S> {
    fn drop(&mut self) {
        // Bind/prepare/start failures cannot report containment, but they must
        // still retire every child the class left in the common owner.
        let _ = self.retire_parts();
    }
}

/// Reusable PCI resource declaration shared by broker selection and intake.
///
/// A provisioner and the eventual class owner may live in different crates
/// (block and filesystem are the current example). Naming one contract type
/// prevents their BAR/interrupt geometry from becoming adjacent handwritten
/// constants that can drift after the broker's claim becomes irreversible.
pub trait PciBindingContract {
    /// Exact BAR indices this class requires. The closed mask vocabulary makes
    /// duplicate and out-of-range declarations unrepresentable.
    const BARS: PciBarMask;
    /// Exact consecutive vector block, including whether absence is lawful.
    const INTERRUPTS: InterruptPolicy;
}

/// Class-specific composition contract. The host owns capability intake,
/// policy realization, and final containment. The implementation consumes the
/// complete resource set, may dismantle it into a controller-owned shape, and
/// must reconstruct it after quiescing and dropping its internal children.
pub trait DeviceBinding: Sized {
    /// The single PCI declaration used by both claim policy and intake.
    type Pci: PciBindingContract;
    /// Non-authority configuration chosen from immutable device metadata.
    /// Founded devices use the default; dynamic claims may pass an exact
    /// class-owned policy without smuggling metadata into a capability.
    type Config: Default;
    /// Class state after every DMA pointer is programmed but before the host
    /// publishes the window or permits hardware to run.
    type Prepared;
    /// Class-specific live driver state, including every consumed resource.
    type Bound;
    /// Class-specific bind and quiesce failure.
    type Error;

    /// Prepare the class while the generic host retains the DMA capability
    /// owner. A class may move route/BAR/pool seams out of `resources`, but it must
    /// leave hardware unable to fetch from any programmed pointer.
    fn prepare(
        resources: &mut DeviceResources<DmaPrepared>,
        config: Self::Config,
    ) -> Result<Self::Prepared, Self::Error>;
    /// Start the already-prepared class. This hook cannot be called until the
    /// generic host has converted the resource owner to [`DmaActive`].
    fn start(
        prepared: Self::Prepared,
        resources: DeviceResources<DmaActive>,
    ) -> Result<Self::Bound, Self::Error>;
    /// Stop class-owned device activity **first** — before the host proves
    /// the function quiet at the fabric and revokes derived authority.  The
    /// host runs this while Bus Master Enable is still open and interrupts
    /// still route, so a controller shutdown that needs either (NVMe
    /// `CC.SHN`, an xHCI `HCRST` that wants a live function) can complete;
    /// what follows makes both impossible.  Answer with what actually
    /// happened: [`ClassQuiesce::Stopped`], or [`ClassQuiesce::NothingToStop`]
    /// only when `Bound` structurally holds no started engine.
    fn quiesce(bound: &mut Self::Bound) -> Result<ClassQuiesce, Self::Error>;
    /// Drop class-internal children and reconstruct the complete common set.
    /// This is infallible by contract: containment cannot depend on successful
    /// hardware I/O after quiesce has already been attempted.
    fn unbind(bound: Self::Bound) -> DeviceResources<DmaActive>;

    /// Name why this class refused, as a small code carried out through
    /// [`IntakeError::ClassBindRefused`].
    ///
    /// `Self::Error` cannot cross this seam — [`BoundDevice::claim`] is
    /// generic over every class — so a class that wants its refusal to survive
    /// to a boot log reduces it here. Defaulted to `0` ("unclassified"): a
    /// class with nothing to add is unaffected, and no existing binding
    /// changes.
    fn refusal_code(_error: &Self::Error) -> u8 {
        0
    }
}

/// Which part of an interrupt-capable PCI contract a measured claim requires.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PciClaimPosture {
    /// Admit only a function capable of realizing the contract's route.
    RouteCapable,
    /// Admit without a route; legal only for `None`/`Optional` contracts whose
    /// owner has an explicit polling posture.
    PollingFallback,
}

/// Build the measured PCI selector for one concrete [`DeviceBinding`].
///
/// BAR and DMA requirements are derived from the same declaration the intake
/// realizes; callers provide only identity/class policy and which declared
/// posture to admit. Because service rows use this in a `const` declaration, a
/// weaker or contradictory request is a compile-time failure rather than a
/// post-claim resource mismatch.
pub const fn pci_selector_for_binding<B: DeviceBinding>(
    class: u8,
    subclass: u8,
    progif: u8,
    identity: PciIdentitySelector,
    posture: PciClaimPosture,
) -> PciSelector {
    pci_selector_for_contract::<B::Pci>(class, subclass, progif, identity, posture)
}

/// Build a selector directly from a shared [`PciBindingContract`].
///
/// This is the provisioner form: the task that claims a device can cite the
/// same contract type the downstream [`DeviceBinding`] names without depending
/// on that downstream service's implementation type.
pub const fn pci_selector_for_contract<C: PciBindingContract>(
    class: u8,
    subclass: u8,
    progif: u8,
    identity: PciIdentitySelector,
    posture: PciClaimPosture,
) -> PciSelector {
    let bars = C::BARS;
    let interrupts = selector_interrupt_requirement(C::INTERRUPTS, posture);
    PciSelector {
        class,
        subclass,
        progif,
        identity,
        bars,
        dma: true,
        interrupts,
    }
}

const fn selector_interrupt_requirement(
    binding: InterruptPolicy,
    posture: PciClaimPosture,
) -> PciInterruptRequirement {
    match (binding, posture) {
        (InterruptPolicy::None, PciClaimPosture::PollingFallback) => PciInterruptRequirement::None,
        (InterruptPolicy::Optional(block), PciClaimPosture::PollingFallback) => {
            validate_interrupt_block(block);
            PciInterruptRequirement::None
        }
        (InterruptPolicy::Required(_), PciClaimPosture::PollingFallback) => {
            panic!("required interrupt binding has no polling fallback")
        }
        (InterruptPolicy::None, PciClaimPosture::RouteCapable) => {
            panic!("route-capable claim requires an interrupt binding")
        }
        (
            InterruptPolicy::Optional(block) | InterruptPolicy::Required(block),
            PciClaimPosture::RouteCapable,
        ) => {
            validate_interrupt_block(block);
            match block.source {
                InterruptSourcePolicy::PreferMsiX => {
                    assert!(
                        block.first_entry == 0 && block.count == 1,
                        "MSI-family inventory proves only one message at entry zero"
                    );
                    PciInterruptRequirement::MsiFamily
                }
                InterruptSourcePolicy::RequireMsiX => {
                    match PciMsixBlock::new(block.first_entry, block.count) {
                        Some(block) => PciInterruptRequirement::Msix(block),
                        None => panic!("strict MSI-X binding has invalid geometry"),
                    }
                }
            }
        }
    }
}

const fn validate_interrupt_block(block: InterruptBlock) {
    assert!(
        block.count != 0
            && block.count.is_power_of_two()
            && block.first_entry.checked_add(block.count).is_some(),
        "PCI binding declares invalid interrupt geometry"
    );
}

#[derive(Debug, PartialEq, Eq)]
enum LifecycleError<E> {
    Prepare(E),
    Activation,
    Start(E),
}

/// One generic lifecycle engine: class preparation, host-owned DMA
/// publication, then class start. Keeping it generic over the resource states
/// makes the ordering independently testable while the concrete instantiation
/// carries [`DeviceResources<DmaPrepared>`] → [`DeviceResources<DmaActive>`].
fn drive_lifecycle<R, P, A, B, E>(
    mut resources: R,
    prepare: impl FnOnce(&mut R) -> Result<P, E>,
    activate: impl FnOnce(R) -> Result<A, R>,
    start: impl FnOnce(P, A) -> Result<B, E>,
) -> Result<B, LifecycleError<E>> {
    let prepared = prepare(&mut resources).map_err(LifecycleError::Prepare)?;
    let active = match activate(resources) {
        Ok(active) => active,
        Err(resources) => {
            // The class-owned prepared graph must die before its common
            // capability owner closes the DMA/BAR windows beneath it.
            drop(prepared);
            drop(resources);
            return Err(LifecycleError::Activation);
        }
    };
    start(prepared, active).map_err(LifecycleError::Start)
}

fn bind_resources<B: DeviceBinding>(
    resources: DeviceResources<DmaPrepared>,
    config: B::Config,
) -> Result<B::Bound, IntakeError> {
    drive_lifecycle(
        resources,
        |resources| B::prepare(resources, config),
        DeviceResources::activate,
        B::start,
    )
    .map_err(|error| match error {
        LifecycleError::Prepare(error) => IntakeError::ClassBindRefused {
            at: ClassStage::Prepare,
            code: B::refusal_code(&error),
        },
        LifecycleError::Start(error) => IntakeError::ClassBindRefused {
            at: ClassStage::Start,
            code: B::refusal_code(&error),
        },
        LifecycleError::Activation => IntakeError::DmaUnavailable,
    })
}

/// **The route-parked pacer** — the production completion wait: between
/// polls of a completion queue, park the task on its held MSI-family route
/// (`WAIT_IRQ | WAIT_SLEEP`,
/// [`wait_interrupt_until`](verbs::wait_interrupt_until)) bounded by one
/// absolute deadline.  The CPU is yielded while the command is in flight,
/// and the wake is the device's own vector — the interrupt rides *in* the
/// completion path, not beside it.
///
/// A doorbell kick cannot wake this park (it is a bare IRQ+sleep wait) —
/// deliberate: the caller's operation is synchronous, its client is the one
/// waiting, and the standing doorbell post keeps any concurrent kick's
/// notice for the serve loop's next drain.  Every non-`Fired` outcome —
/// deadline, revoke, a park the transport refused — returns `false` rather
/// than retrying: a wait that cannot park must not become a spin (the input
/// probe's rule), and the caller's final poll still catches a completion
/// that landed without a delivered interrupt.
///
/// Lives beside the intake that realizes the route because this is the one
/// module that sees both the capability verbs and the block-crate wait seam;
/// `kore-svc-swap-nvme` and the filesystem's direct NVMe backend compose the
/// same pacer.
pub struct RoutePacer<'r> {
    route: &'r RouteToken,
    deadline_ns: u64,
    parks: u64,
    fired: u64,
}

impl<'r> RoutePacer<'r> {
    /// A pacer over `route`, bounded by the absolute `deadline_ns`.
    pub fn new(route: &'r RouteToken, deadline_ns: u64) -> Self {
        Self {
            route,
            deadline_ns,
            parks: 0,
            fired: 0,
        }
    }

    /// Parks issued (the wait's denominator).
    pub fn parks(&self) -> u64 {
        self.parks
    }

    /// Parks the route resolved as real fires — the MSI-family delivery witness.
    pub fn fired(&self) -> u64 {
        self.fired
    }
}

impl crate::block::CompletionPacer for RoutePacer<'_> {
    fn pace(&mut self) -> bool {
        use kore_rt::cap::verbs::InterruptWait;
        self.parks += 1;
        match verbs::wait_interrupt_until(self.route, self.deadline_ns) {
            Ok(InterruptWait::Fired) => {
                self.fired += 1;
                true
            }
            Ok(InterruptWait::Deadline | InterruptWait::Revoked) | Err(_) => false,
        }
    }
}

/// **The clock-bounded polled pacer** — the completion wait for a machine
/// that realized no interrupt route but *does* have a readable clock.
///
/// It polls, like [`SpinPacer`](crate::block::SpinPacer) does, but its budget is
/// an **absolute deadline** read from the same clock [`RoutePacer`] uses
/// rather than a count of spin hints. That distinction is the whole reason
/// this exists:
///
/// A spin count does not bound time — it bounds *retired instructions*. Its
/// wall-clock meaning moves with core frequency, SMT contention, and whether
/// the hypervisor descheduled the vCPU mid-loop, so the tolerance it
/// expresses is not a duration anyone chose; it is whatever N spin hints
/// happened to cost on that machine at that moment. That is tolerable for a
/// pacing hint and *not* tolerable as the input to a liveness decision:
/// `crate::block::BlockProvider` treats an expired wait as evidence to declare
/// its channel lost, so a device that was merely slow could be answered by
/// marking a live queue dead. Evidence of a stuck device has to be elapsed
/// time.
///
/// One monitored pause is capped by [`PROBE_PAUSE`] and by the time left to
/// the caller's deadline.  This replaces the former batch of 64 spin hints
/// when the machine can monitor; when it cannot, the remaining 63 hints keep
/// the old fallback cadence.  A spin batch had no duration semantics and
/// could move substantially with core frequency or vCPU scheduling, while
/// simply replacing every hint with a 4 us pause would have stretched one
/// batch to 256 us.  One capped pause preserves the single clock-check cadence
/// and cannot deliberately sleep past a short remaining budget.
///
/// A clock that stops answering mid-wait returns `false` rather than
/// degrading into an unbounded spin — the caller's final poll still runs, and
/// a vanished clock page is a fault to surface, not to absorb.
pub struct DeadlinePacer {
    deadline_ns: u64,
    polls: u64,
    /// How the pacer reads the clock.  A function pointer rather than a
    /// direct call so the deadline arithmetic is exercisable off-metal: the
    /// host has no clock page, so a pacer hard-wired to
    /// [`kore_rt::clock::uptime`] could only ever be tested in its
    /// clock-unavailable posture, and the comparison that decides whether a
    /// disk is declared lost would ship unverified.
    now_ns: fn() -> Option<u64>,
}

/// The production clock read: uptime nanoseconds, or `None` when the
/// founder-published page is unreadable.
///
/// Public because it is also what a *driver* bring-up needs: every
/// `crate::wait::Budget` a service takes out over real hardware is built from
/// this exact reading, and a service hand-rolling `uptime().ok().map(…)` at
/// each call site would be a second copy of the "which clock, and what does
/// unreadable mean" decision — the two would drift, and the fallback posture
/// (`Budget`'s spin backstop) is chosen by whether this returns `None`.
pub fn clock_now_ns() -> Option<u64> {
    kore_rt::clock::uptime().ok().map(|now| now.as_nanos())
}

/// How long one relaxation between register probes may sleep the core.
///
/// The same 4 µs cadence [`kore_rt::pace`] and the filesystem completion
/// monitor use. Short enough that a handshake which completes early is not
/// held past its readiness by more than one pause; long enough that the pause
/// is worth taking at all.
pub const PROBE_PAUSE: core::time::Duration = kore_rt::pace::PROBE_PAUSE;

/// Spin hints between deadline clock reads when a monitored pause is
/// unavailable.  Kept at the historical cadence so an unsupported monitor
/// does not multiply MMIO probes and clock-page reads by 64.
const DEADLINE_FALLBACK_SPINS: u32 = 64;

/// **The relaxation a bring-up handshake issues between register probes** —
/// the monitored replacement for a bare `core::hint::spin_loop`.
///
/// This is a **liveness** change, not a power one. [`DeadlinePacer`]'s doc
/// already states the principle for the *budget*: a spin count bounds retired
/// instructions, not time, so its wall-clock meaning moves with core
/// frequency, SMT contention, and whether a hypervisor descheduled the vCPU
/// mid-loop. The pause hint between probes had never been converted, so every
/// bring-up handshake still relaxed by an amount no one chose — and bootstrap,
/// where these run, is exactly where that non-determinism is least observable
/// and most expensive.
///
/// A monitored pause returns because the hardware makes it, which is what
/// forces the caller's loop to re-check and lets it give up and report. A raw
/// spin offers no such guarantee: it is faster per iteration and has no
/// liveness property at all.
///
/// Degrades to exactly one `core::hint::spin_loop` where the machine has no
/// address monitor or no readable clock ([`kore_rt::hint::pause_until`]), so
/// it is never a behaviour change a caller must reason about — and callers
/// keep their own `Budget`, which is still what bounds the handshake.
pub fn relax() {
    let _ = kore_rt::hint::pause_until(PROBE_PAUSE);
}

impl DeadlinePacer {
    /// A pacer bounded by the absolute `deadline_ns`, with no route to park
    /// on.
    pub fn new(deadline_ns: u64) -> Self {
        Self {
            deadline_ns,
            polls: 0,
            now_ns: clock_now_ns,
        }
    }

    /// A pacer bounded by `deadline_ns`, **or `None` when this machine's
    /// clock cannot answer** — in which case no time bound is expressible and
    /// the caller must fall back to `crate::block::SpinPacer`.
    ///
    /// [`new`](Self::new) is correct for a caller that computed `deadline_ns`
    /// from *this* clock, because a successful read already proved it answers
    /// — which is what `kore-svc-dagfs` does. This constructor exists for a
    /// caller whose deadline came from somewhere else, and it closes a real
    /// trap: `kore_rt::executor::clock_ns` answers from a kernel millisecond
    /// anchor when the founder-published page is uncalibrated, so a deadline
    /// derived from it looks perfectly valid on a machine where this pacer's
    /// own clock returns `None`. Handed that deadline, `pace` would refuse on
    /// its first turn — one poll, then give up, which is *worse* than the spin
    /// budget it replaced rather than better.
    ///
    /// The two clocks are otherwise the same timebase (`clock_ns` derives from
    /// `clock::monotonic_nanos`, which is this pacer's clock exactly), so a
    /// deadline computed from either is comparable here to within the
    /// millisecond anchor's granularity — 0.1% of a one-second budget, and
    /// always in the generous direction.
    ///
    /// Asking the pacer rather than making each caller test the clock itself
    /// keeps the "may I use this" question with the thing that owns the
    /// answer; a caller cannot get it wrong by testing the wrong clock.
    pub fn try_new(deadline_ns: u64) -> Option<Self> {
        clock_now_ns()?;
        Some(Self::new(deadline_ns))
    }

    /// A pacer over an explicit clock — the seam the tests drive.
    pub fn with_clock(deadline_ns: u64, now_ns: fn() -> Option<u64>) -> Self {
        Self {
            deadline_ns,
            polls: 0,
            now_ns,
        }
    }

    /// Poll opportunities issued — the route-less wait's denominator.
    pub fn polls(&self) -> u64 {
        self.polls
    }

    /// **The bound alone**, for a caller whose wait has already happened.
    ///
    /// [`pace`](crate::block::CompletionPacer::pace) is two things: a wait (a
    /// monitored pause, or its fallback spin batch) and the bound (the clock
    /// comparison). A caller that slept
    /// its core on the completion's own cache line
    /// (`kore_nvme::NvmeBlockDevice::completion_word` under
    /// [`kore_rt::hint::wait_on`]) has already bought the wait — and a better
    /// one, since a store ends it immediately where the fallback spin batch
    /// only notices at its end. Waiting again would pay for the wait twice
    /// and hand back the power the monitored sleep just saved.
    ///
    /// It still needs the bound, and needs it on the **same** counter: the
    /// deadline is what makes this tier's expiry mean elapsed time, and
    /// `polls` is the denominator a route-less machine is read by. So both
    /// entry points count the turn; only one buys it.
    ///
    /// Not a second `CompletionPacer` impl, and deliberately not the trait
    /// method: a pacer whose `pace` did not wait would be a trap for every
    /// other caller, which is precisely the loop this one is nested inside.
    ///
    /// [`kore_rt::hint::wait_on`]: ../../kore_rt/hint/fn.wait_on.html
    pub fn pace_after_wait(&mut self) -> bool {
        self.within_deadline()
    }

    /// The clock comparison both entry points end on.
    fn within_deadline(&mut self) -> bool {
        self.polls += 1;
        match (self.now_ns)() {
            Some(now) => now < self.deadline_ns,
            // The clock answered when the deadline was computed and does not
            // now. Refuse rather than silently becoming unbounded.
            None => false,
        }
    }
}

impl crate::block::CompletionPacer for DeadlinePacer {
    fn pace(&mut self) -> bool {
        self.polls += 1;
        let Some(now) = (self.now_ns)() else {
            return false;
        };
        if now >= self.deadline_ns {
            return false;
        }

        // Sleep no longer than the caller can still afford.  The monitor is
        // a hint and may return early; the second clock read remains the
        // actual bound, exactly as the post-spin read was before.
        let left = self.deadline_ns - now;
        let cap = PROBE_PAUSE.min(core::time::Duration::from_nanos(left));
        if matches!(
            kore_rt::hint::pause_until(cap),
            kore_rt::hint::WaitEnd::Spun
        ) {
            // `pause_until` already issued the first fallback spin.
            for _ in 1..DEADLINE_FALLBACK_SPINS {
                core::hint::spin_loop();
            }
        }
        matches!((self.now_ns)(), Some(after) if after < self.deadline_ns)
    }
}

/// One class's bound owner. `Option` exists only so explicit/implicit teardown
/// can move the class-owned affine state exactly once; callers can never
/// observe an empty live binding.
pub struct BoundDevice<B: DeviceBinding> {
    bound: Option<B::Bound>,
    _class: PhantomData<B>,
}

impl<B: DeviceBinding> BoundDevice<B> {
    /// Bind a device the task **claimed** rather than was founded with
    /// (`res/DEVICE_SERVICE.md` §10) — realize the class declaration over
    /// the granted token, then run the class bring-up.
    ///
    /// The same two steps `Bundle::resolve` runs, reached without a
    /// `Bootstrap`: a claimed device is the identical authority, and
    /// nothing downstream of this — BAR claims, IOVA windows, MSI routes,
    /// the teardown order — can tell which road it arrived by.
    ///
    /// On failure the caller still holds nothing: `realize_resources`
    /// derived children are dropped with the resource set, and the device
    /// token was consumed. The broker cannot authenticate revocation from an
    /// integer index, so that function remains quarantined for this boot; the
    /// driver may scan a later candidate but cannot release or retry this one.
    pub fn claim(device: DeviceToken, pool: SupplyToken) -> Result<Self, IntakeError> {
        Self::claim_with_config(device, pool, B::Config::default())
    }

    /// Bind a claimed device with class-owned, non-authority configuration
    /// derived from the broker's immutable inventory metadata.
    pub fn claim_with_config(
        device: DeviceToken,
        pool: SupplyToken,
        config: B::Config,
    ) -> Result<Self, IntakeError> {
        let resources = realize_resources::<B>(device, pool)?;
        let bound = bind_resources::<B>(resources, config)?;
        Ok(Self {
            bound: Some(bound),
            _class: PhantomData,
        })
    }

    /// Bind a firmware-described ACPI/DT platform device through the same
    /// class lifecycle as PCI. Register aperture zero denotes the device's
    /// one exact firmware MMIO window; the class otherwise remains bus-neutral.
    pub fn claim_platform(device: PlatformToken, pool: SupplyToken) -> Result<Self, IntakeError> {
        let resources = realize_platform_resources::<B>(device, pool)?;
        let bound = bind_resources::<B>(resources, B::Config::default())?;
        Ok(Self {
            bound: Some(bound),
            _class: PhantomData,
        })
    }

    /// Borrow the complete class-owned binding.
    pub fn parts(&self) -> &B::Bound {
        self.bound.as_ref().expect("bound device already retired")
    }

    /// Mutably borrow the complete class-owned binding.
    pub fn parts_mut(&mut self) -> &mut B::Bound {
        self.bound.as_mut().expect("bound device already retired")
    }

    /// The terminal sequence, host-fixed: class stop → unbind → the kernel's
    /// fabric proof → unconditional authority containment, with all three
    /// outcomes returned to the owning service.  Order is the guarantee —
    /// the class stops its engine while the function is still live, the
    /// kernel then closes the latch while every translation still stands,
    /// and revocation runs last whatever the first two said.
    pub fn teardown(mut self) -> BindingTeardown<B::Error> {
        let mut bound = self.bound.take().expect("bound device already retired");
        let quiesce = B::quiesce(&mut bound);
        let resources = B::unbind(bound);
        let fabric = resources.prove_quiet();
        let resources = resources.retire();
        BindingTeardown {
            quiesce,
            fabric,
            resources,
        }
    }

    /// Stop a temporary class probe, retire every derived child, and recover
    /// its exact device plus parent DMA supply only after proven closure.
    pub fn teardown_recover(mut self) -> BindingRecovery<B::Error> {
        let mut bound = self.bound.take().expect("bound device already retired");
        let quiesce = B::quiesce(&mut bound);
        let (resources, mut device, mut pool) = B::unbind(bound).recover();
        if quiesce.is_err() || !resources.complete() {
            drop(device.take());
            drop(pool.take());
        }
        BindingRecovery {
            quiesce,
            resources,
            device,
            pool,
        }
    }
}

impl<B: DeviceBinding> Drop for BoundDevice<B> {
    fn drop(&mut self) {
        let Some(mut bound) = self.bound.take() else {
            return;
        };
        // Implicit Rust drop cannot report results, but it still runs the
        // whole terminal sequence — class hook, fabric proof, containment.
        // External task reap is a separate kernel sweep and cannot execute
        // userland quiesce code.
        let _ = B::quiesce(&mut bound);
        let resources = B::unbind(bound);
        let _ = resources.prove_quiet();
        let _ = resources.retire();
    }
}

/// Generic driver-task endowment. The profile stays the settled direct
/// DEVICE + DMA_POOL pair; only the class declaration controls what is derived
/// from those roots.
pub struct DeviceEndowment<B: DeviceBinding> {
    /// The single affine owner of the class state and derived resources.
    pub device: BoundDevice<B>,
}

impl<B: DeviceBinding> Bundle for DeviceEndowment<B> {
    const PROFILE: Profile = Profile::new(
        "driver",
        &[RoleSlot::at(role::DEVICE), RoleSlot::at(role::DMA_POOL)],
    );

    fn resolve(boot: &Bootstrap<'_>) -> Result<Self, BindError> {
        let resources = resolve_resources::<B>(boot).map_err(IntakeError::bind_error)?;
        let bound = bind_resources::<B>(resources, B::Config::default())
            .map_err(IntakeError::bind_error)?;
        Ok(Self {
            device: BoundDevice {
                bound: Some(bound),
                _class: PhantomData,
            },
        })
    }
}

fn resolve_resources<B: DeviceBinding>(
    boot: &Bootstrap<'_>,
) -> Result<DeviceResources<DmaPrepared>, IntakeError> {
    let dev_slot = boot
        .handle(role::DEVICE, 0)
        .ok_or(IntakeError::MissingRole {
            role: role::DEVICE,
            index: 0,
        })?;
    let device = verbs::claim_device(dev_slot).ok_or(IntakeError::BadHandle {
        role: role::DEVICE,
        index: 0,
    })?;
    let pool_slot = boot
        .handle(role::DMA_POOL, 0)
        .ok_or(IntakeError::MissingRole {
            role: role::DMA_POOL,
            index: 0,
        })?;
    let pool = verbs::claim_supply(pool_slot).ok_or(IntakeError::BadHandle {
        role: role::DMA_POOL,
        index: 0,
    })?;
    realize_resources::<B>(device, pool)
}

/// Realize a class declaration over an **already-held** device and DMA
/// supply — map its BAR set, derive its interrupt route, and open its DMA
/// window.
///
/// The intake proper.  It is split from the boot-block lookup above
/// because where the device comes from is not the host's business: a row
/// founded holding one and a row that **claimed** one from the device
/// broker (`res/DEVICE_SERVICE.md` §10) realize it identically, and the
/// class declaration is what decides the shape either way.
///
/// Public because a claiming driver has no `Bootstrap` handle to resolve
/// from — it has a token the broker granted it, which is the same
/// authority arriving by a different road.
pub fn realize_resources<B: DeviceBinding>(
    device: DeviceToken,
    pool: SupplyToken,
) -> Result<DeviceResources<DmaPrepared>, IntakeError> {
    realize_host_resources::<B>(HostDeviceToken::Pci(device), pool)
}

/// Realize one firmware-described platform device through the common driver
/// lifecycle. Kept separate from [`realize_resources`] at the public boundary
/// so callers cannot accidentally erase the capability species themselves.
pub fn realize_platform_resources<B: DeviceBinding>(
    device: PlatformToken,
    pool: SupplyToken,
) -> Result<DeviceResources<DmaPrepared>, IntakeError> {
    realize_host_resources::<B>(HostDeviceToken::Platform(device), pool)
}

fn realize_host_resources<B: DeviceBinding>(
    device: HostDeviceToken,
    pool: SupplyToken,
) -> Result<DeviceResources<DmaPrepared>, IntakeError> {
    let (expected_bars, interrupt) = validate_declaration::<B>()?;

    if matches!(&device, HostDeviceToken::Platform(_)) {
        if expected_bars != 1 {
            return Err(IntakeError::InvalidBar(expected_bars));
        }
        if interrupt.is_some_and(|block| block != InterruptBlock::SINGLE) {
            return Err(IntakeError::InvalidInterruptBlock);
        }
    }

    let mut bars = [const { None }; PCI_BAR_COUNT];
    for index in 0..PCI_BAR_COUNT as u8 {
        if expected_bars & (1 << index) == 0 {
            continue;
        }
        let bar = device
            .map_registers(index)
            .map_err(|fault| IntakeError::BarMap {
                bar: index,
                code: fault.code(),
            })?;
        bars[index as usize] = Some(CapRegs::from_window(bar));
    }

    let route = match (B::Pci::INTERRUPTS, interrupt) {
        (InterruptPolicy::None, _) => None,
        (InterruptPolicy::Optional(_), Some(block)) => device.establish_interrupts(block),
        (InterruptPolicy::Required(_), Some(block)) => Some(
            device
                .establish_interrupts(block)
                .ok_or(IntakeError::InterruptUnavailable)?,
        ),
        _ => return Err(IntakeError::InvalidInterruptBlock),
    };

    let len = pool.len();
    let (window, pool) = device
        .dma_window(pool, len, DMA_IOVA_BASE)
        .map_err(|fault| IntakeError::DmaMap { code: fault.code() })?;
    let cap_pool = CapPool::new(
        window.cpu(),
        window.iova(),
        window.len(),
        &alloc::alloc::Global,
    )
    .ok_or(IntakeError::DmaUnavailable)?;

    let expected_route = route.is_some();
    Ok(DeviceResources {
        route,
        expected_route,
        dma: Some(DmaOwner {
            window: Some(window),
            extensions: Vec::new(),
            regions: Rc::clone(&cap_pool.regions),
            imports: Rc::clone(&cap_pool.imports),
            loans: RefCell::new(Vec::new()),
            _pool: pool,
            device: Some(device),
        }),
        pool: Some(cap_pool),
        bars,
        expected_bars,
        _state: PhantomData,
    })
}

fn establish_interrupt_block(device: &DeviceToken, block: InterruptBlock) -> Option<RouteToken> {
    match block.source {
        InterruptSourcePolicy::PreferMsiX => {
            verbs::establish_interrupts(device, block.first_entry, block.count, block.affinity)
        }
        InterruptSourcePolicy::RequireMsiX => {
            verbs::establish_msix_interrupts(device, block.first_entry, block.count, block.affinity)
        }
    }
}

fn validate_declaration<B: DeviceBinding>() -> Result<(u8, Option<InterruptBlock>), IntakeError> {
    let expected_bars = B::Pci::BARS.bits();
    let interrupt = match B::Pci::INTERRUPTS {
        InterruptPolicy::None => None,
        InterruptPolicy::Optional(block) | InterruptPolicy::Required(block) => {
            if block.count == 0
                || !block.count.is_power_of_two()
                || block.first_entry.checked_add(block.count).is_none()
            {
                return Err(IntakeError::InvalidInterruptBlock);
            }
            Some(block)
        }
    };

    Ok((expected_bars, interrupt))
}

#[cfg(test)]
mod tests {
    #[test]
    fn remote_loan_release_requires_confirmed_retirement_and_preserves_busy_siblings() {
        let arena = test_arena::TestArena::<4096>::new();
        let mut loans = Vec::new_in(&arena);
        loans.extend([(1, false), (2, true), (3, false)]);
        let mut released = Vec::new_in(&arena);
        assert!(retire_completed_loans(
            &mut loans,
            |loan| Ok(loan.1),
            |loan| released.push(loan.0)
        ));
        assert_eq!(released, [2]);
        assert_eq!(loans, [(1, false), (3, false)]);
        assert!(!retire_completed_loans(
            &mut loans,
            |_| Err(()),
            |_| panic!("indeterminate custody must not release")
        ));
        assert_eq!(loans.len(), 2);
        loans[0].1 = true;
        assert!(retire_completed_loans(
            &mut loans,
            |loan| Ok(loan.1),
            |loan| released.push(loan.0)
        ));
        assert_eq!(released, [2, 1]);
        assert_eq!(loans, [(3, false)]);
    }

    #[test]
    fn lent_pool_run_cannot_be_reallocated_until_the_retirement_gate_opens() {
        #[repr(align(4096))]
        struct Backing([u8; 8192]);
        let mut backing = Backing([0; 8192]);
        static CLAIMS: test_arena::TestArena<4096> = test_arena::TestArena::new();
        let arena = test_arena::TestArena::<4096>::new();
        let pool = CapPool::new(backing.0.as_mut_ptr(), DMA_IOVA_BASE, 8192, &CLAIMS).unwrap();
        let loan = pool.alloc(4096).unwrap();
        let sibling = pool.alloc(4096).unwrap();
        let address = loan.iova();
        let mut loans = Vec::new_in(&arena);
        loans.push(loan);
        assert!(pool.alloc(4096).is_none());
        assert!(retire_completed_loans(
            &mut loans,
            |_| Ok(false),
            |_| panic!("busy loan")
        ));
        assert!(pool.alloc(4096).is_none());
        assert!(!retire_completed_loans(
            &mut loans,
            |_| Err(()),
            |_| panic!("quarantined loan")
        ));
        assert!(pool.alloc(4096).is_none());
        assert!(retire_completed_loans(
            &mut loans,
            |_| Ok(true),
            |buffer| {
                // Host-only retirement model: no hardware or external pointer exists.
                assert!(unsafe { pool.release_quiesced(buffer) }.is_ok());
            }
        ));
        let replacement = pool.alloc(4096).unwrap();
        assert_eq!(replacement.iova(), address);
        assert_ne!(replacement.iova(), sibling.iova());
        assert!(unsafe { pool.release_quiesced(replacement) }.is_ok());
        assert!(unsafe { pool.release_quiesced(sibling) }.is_ok());
    }

    #[test]
    fn imported_retirement_defers_live_owners_and_resumes_after_the_batch() {
        let arena = test_arena::TestArena::<4096>::new();
        let mut owners = Vec::new_in(&arena);
        owners.extend([false, true, true]);
        let retired = core::cell::Cell::new(0);
        assert!(super::retire_imports_and_resume(
            &mut owners,
            false,
            |released| *released,
            |_| panic!("unrelated device work excludes outer retirement"),
            || panic!("deferred retirement must not touch activation")
        ));
        assert_eq!(
            owners,
            [false, true, true],
            "deferred backing remains in native custody"
        );
        assert!(super::retire_imports_and_resume(
            &mut owners,
            true,
            |released| *released,
            |_| {
                retired.set(retired.get() + 1);
                true
            },
            || {
                assert_eq!(retired.get(), 2);
                true
            }
        ));
        assert_eq!(owners, [false], "live imports retain their mapping owners");
        assert!(super::retire_imports_and_resume(
            &mut owners,
            true,
            |released| *released,
            |_| panic!("live import must not retire"),
            || panic!("no retirement means no activation")
        ));
        owners.push(true);
        assert!(!super::retire_imports_and_resume(
            &mut owners,
            true,
            |released| *released,
            |_| false,
            || panic!("refused retirement requires containment")
        ));
        assert_eq!(
            owners,
            [false, true],
            "refused retirement preserves the exact owner"
        );
        assert!(
            !super::retire_imports_and_resume(
                &mut owners,
                true,
                |released| *released,
                |_| true,
                || false
            ),
            "failed activation cannot acknowledge success"
        );
        assert_eq!(owners, [false]);
    }

    use super::*;

    #[test]
    fn borrowed_process_register_view_carries_bounds_without_ownership() {
        let mut words = [0u64; 2];
        // SAFETY: `words` remains live for the view and provides a naturally
        // aligned, 16-byte test aperture.
        let regs = unsafe { CapRegs::from_process_mapping(words.as_mut_ptr().cast::<u8>(), 16) };
        regs.write32(4, 0xa5a5_5a5a);
        assert_eq!(regs.read32(4), 0xa5a5_5a5a);
        assert_eq!(regs.base(), words.as_mut_ptr().cast::<u8>());
        assert_eq!(regs.len(), 16);
        drop(regs);
        assert_eq!((words[0] >> 32) as u32, 0xa5a5_5a5a);
    }

    #[test]
    fn intake_diagnostics_keep_the_exact_class_and_detail() {
        assert_eq!(
            IntakeError::MissingRole {
                role: 0x1234,
                index: 0xabcd,
            }
            .diagnostic_code(),
            1 | ((0x1234u64 | (0xabcdu64 << 16)) << 8),
        );
        assert_eq!(
            IntakeError::BarMap { bar: 2, code: 37 }.diagnostic_code(),
            6 | ((2u64 | (37u64 << 8)) << 8),
        );
        assert_eq!(
            IntakeError::ClassBindRefused {
                at: ClassStage::Start,
                code: 91,
            }
            .diagnostic_code(),
            11 | ((2u64 | (91u64 << 8)) << 8),
        );
    }

    struct TraceRegisterFile {
        last_read32: Cell<Option<usize>>,
        last_write32: Cell<Option<(usize, u32)>>,
        last_read64: Cell<Option<usize>>,
        last_write64: Cell<Option<(usize, u64)>>,
    }

    impl TraceRegisterFile {
        fn new() -> Self {
            Self {
                last_read32: Cell::new(None),
                last_write32: Cell::new(None),
                last_read64: Cell::new(None),
                last_write64: Cell::new(None),
            }
        }
    }

    impl Regs for TraceRegisterFile {
        fn read32(&self, off: usize) -> u32 {
            self.last_read32.set(Some(off));
            0x1234_5678
        }

        fn write32(&self, off: usize, val: u32) {
            self.last_write32.set(Some((off, val)));
        }
    }

    impl Regs64 for TraceRegisterFile {
        fn read64_once(&self, off: usize) -> u64 {
            self.last_read64.set(Some(off));
            0x0123_4567_89ab_cdef
        }

        fn write64_once(&self, off: usize, val: u64) {
            self.last_write64.set(Some((off, val)));
        }
    }

    /// The pacer's budget is elapsed time, and the arithmetic that decides
    /// it has to be right: an expired completion wait is what
    /// `crate::block::BlockProvider` turns into `mark_lost`, so a wrong
    /// comparison here declares a live disk dead.
    #[test]
    fn a_deadline_pacer_waits_until_the_deadline_and_not_past_it() {
        use crate::block::CompletionPacer as _;
        use core::sync::atomic::{AtomicU64, Ordering};

        // A clock the test advances, reachable from a fn pointer. Owned by
        // this test alone, so the shared static cannot race another.
        static NOW: AtomicU64 = AtomicU64::new(0);
        fn now() -> Option<u64> {
            Some(NOW.load(Ordering::Relaxed))
        }

        NOW.store(0, Ordering::Relaxed);
        let mut pacer = DeadlinePacer::with_clock(1_000, now);
        assert!(pacer.pace(), "before the deadline the wait continues");
        assert_eq!(pacer.polls(), 1);

        NOW.store(999, Ordering::Relaxed);
        assert!(pacer.pace(), "one nanosecond short is still inside");

        NOW.store(1_000, Ordering::Relaxed);
        assert!(!pacer.pace(), "at the deadline the budget is spent");
        NOW.store(5_000, Ordering::Relaxed);
        assert!(!pacer.pace(), "and it stays spent");
        assert_eq!(pacer.polls(), 4, "every opportunity is counted");
    }

    /// A deadline already in the past must not buy even one wait — the
    /// caller's final poll is what catches a completion that already landed.
    #[test]
    fn a_deadline_already_passed_refuses_immediately() {
        use crate::block::CompletionPacer as _;
        fn now() -> Option<u64> {
            Some(10_000)
        }
        let mut pacer = DeadlinePacer::with_clock(1_000, now);
        assert!(!pacer.pace());
    }

    /// The bound-only entry must reach the SAME verdict as `pace` on the
    /// same clock, and count into the same denominator.
    ///
    /// It exists so a turn that already slept on the completion's cache line
    /// does not also spin — but if the two entry points could disagree about
    /// the deadline, a monitored wait would be bounded differently from a
    /// spun one, and `deadline_waits` would stop meaning one thing.
    #[test]
    fn the_bound_only_entry_agrees_with_pace_and_shares_its_denominator() {
        use crate::block::CompletionPacer as _;
        use core::sync::atomic::{AtomicU64, Ordering};

        static NOW: AtomicU64 = AtomicU64::new(0);
        fn now() -> Option<u64> {
            Some(NOW.load(Ordering::Relaxed))
        }

        NOW.store(0, Ordering::Relaxed);
        let mut monitored = DeadlinePacer::with_clock(1_000, now);
        let mut spun = DeadlinePacer::with_clock(1_000, now);
        for at in [0, 999, 1_000, 5_000] {
            NOW.store(at, Ordering::Relaxed);
            assert_eq!(
                monitored.pace_after_wait(),
                spun.pace(),
                "the two entry points disagreed about the deadline at {at}"
            );
        }
        assert_eq!(monitored.polls(), 4, "a waited turn is still a turn");
        assert_eq!(monitored.polls(), spun.polls());
    }

    /// A stopped clock refuses through the bound-only entry too.
    ///
    /// This is the path that would hang hardest if it got this wrong: a
    /// monitored sleep returns on its own microsecond cap, so a caller told
    /// "keep going" forever would spin through the monitor at full speed with
    /// no bound at all.
    #[test]
    fn the_bound_only_entry_refuses_a_stopped_clock() {
        fn no_clock() -> Option<u64> {
            None
        }
        let mut pacer = DeadlinePacer::with_clock(u64::MAX, no_clock);
        assert!(!pacer.pace_after_wait());
        assert_eq!(pacer.polls(), 1);
    }

    /// **`try_new` refuses when its own clock cannot answer**, which is the
    /// whole reason it exists rather than being spelled `new`.
    ///
    /// The trap it closes: a caller's deadline may come from
    /// `kore_rt::executor::clock_ns`, which keeps answering off a kernel
    /// millisecond anchor when the founder-published clock page is
    /// uncalibrated. Such a deadline looks perfectly valid while THIS pacer's
    /// clock reads `None`, and a pacer built on it refuses on its first turn —
    /// one poll, then give up. That is worse than the spin budget it would
    /// have replaced, so on such a machine the tier must not be selected at
    /// all.
    ///
    /// The host has no clock page, so this asserts the refusing side directly.
    /// The accepting side is covered by the `with_clock` tests here, which
    /// drive the same comparison over an injected clock.
    #[test]
    fn a_pacer_refuses_to_exist_without_a_clock_to_bound_it() {
        assert!(
            DeadlinePacer::try_new(1_000).is_none(),
            "offered a time bound on a machine with no time source"
        );
    }

    /// A clock that stops answering mid-wait refuses rather than degrading
    /// into an unbounded spin. The alternative — treating an unreadable clock
    /// as "keep going" — is a hang inside a filesystem request.
    #[test]
    fn a_clock_that_stops_answering_refuses_rather_than_spinning_forever() {
        use crate::block::CompletionPacer as _;
        fn no_clock() -> Option<u64> {
            None
        }
        let mut pacer = DeadlinePacer::with_clock(u64::MAX, no_clock);
        assert!(!pacer.pace());
        assert_eq!(pacer.polls(), 1);
    }

    #[test]
    fn shared_bar_views_rebase_xe_registers_and_gsm_without_widening_authority() {
        const XE_REGS_LEN: usize = 4 * 1024 * 1024;
        const GSM_BASE: usize = 8 * 1024 * 1024;
        const BAR0_LEN: usize = 16 * 1024 * 1024;

        let owner = SharedRegsOwner::new(TraceRegisterFile::new());
        let ordinary = owner.view(RegsRegion::new(BAR0_LEN, 0, XE_REGS_LEN).unwrap());
        let gsm = owner.view(RegsRegion::new(BAR0_LEN, GSM_BASE, BAR0_LEN - GSM_BASE).unwrap());

        ordinary.write32(0x200, 0xa5a5_5a5a);
        assert_eq!(owner.regs.last_write32.get(), Some((0x200, 0xa5a5_5a5a)));
        gsm.write64_once(0x80, 0xfeed_face_cafe_beef);
        assert_eq!(
            owner.regs.last_write64.get(),
            Some((GSM_BASE + 0x80, 0xfeed_face_cafe_beef))
        );
        assert_eq!(gsm.read64_once(0x88), 0x0123_4567_89ab_cdef);
        assert_eq!(owner.regs.last_read64.get(), Some(GSM_BASE + 0x88));

        drop((ordinary, gsm));
        assert!(owner.try_rejoin().is_ok());
    }

    #[test]
    fn shared_bar_rejoin_waits_for_every_view_and_clone() {
        let owner = SharedRegsOwner::new(TraceRegisterFile::new());
        let view = owner.view(RegsRegion::new(64, 16, 24).unwrap());
        let clone = view.clone();

        let owner = match owner.try_rejoin() {
            Ok(_) => panic!("rejoined while register views were live"),
            Err(owner) => owner,
        };
        view.write32(0, 7);
        assert_eq!(owner.regs.last_write32.get(), Some((16, 7)));

        drop(view);
        let owner = match owner.try_rejoin() {
            Ok(_) => panic!("rejoined while a cloned register view was live"),
            Err(owner) => owner,
        };
        clone.write64_once(8, 9);
        assert_eq!(owner.regs.last_write64.get(), Some((24, 9)));

        drop(clone);
        let regs = owner
            .try_rejoin()
            .unwrap_or_else(|_| panic!("all views were gone"));
        assert_eq!(regs.last_write32.get(), Some((16, 7)));
        assert_eq!(regs.last_write64.get(), Some((24, 9)));
    }

    #[test]
    fn cap_pool_honors_stronger_congruent_alignment() {
        use alloc::alloc::{Layout, alloc_zeroed, dealloc};

        let layout = Layout::from_size_align(64 * 1024, 16 * 1024).unwrap();
        // SAFETY: the layout is non-zero and valid; the allocation is released
        // after every non-owning CapBuf and the pool have gone out of use.
        let cpu = unsafe { alloc_zeroed(layout) };
        assert!(!cpu.is_null());
        static ARENA: test_arena::TestArena<4096> = test_arena::TestArena::new();
        let pool = CapPool::new(cpu, DMA_IOVA_BASE, layout.size() as u64, &ARENA).unwrap();
        let first = pool.alloc(4096).unwrap();
        let aligned = pool.alloc_aligned(4096, 8192).unwrap();
        assert_eq!(first.iova().0, DMA_IOVA_BASE);
        assert_eq!(aligned.iova().0 & 0x1fff, 0);
        assert_eq!(aligned.cpu() as usize & 0x1fff, 0);
        drop((first, aligned, pool));
        // SAFETY: `cpu` came from this exact layout and no view is live now.
        unsafe { dealloc(cpu, layout) };
    }

    #[test]
    fn cap_pool_reuses_only_quiesced_claims_and_zeros_recycled_bytes() {
        use alloc::alloc::{Layout, alloc_zeroed, dealloc};
        static ARENA: test_arena::TestArena<4096> = test_arena::TestArena::new();
        let layout = Layout::from_size_align(4 * 4096, 4096).unwrap();
        // SAFETY: the window is retained until all test views are dead.
        let cpu = unsafe { alloc_zeroed(layout) };
        assert!(!cpu.is_null());
        let pool = CapPool::new(cpu, DMA_IOVA_BASE, layout.size() as u64, &ARENA).unwrap();
        let a = pool.alloc(4096).unwrap();
        let b = pool.alloc(4096).unwrap();
        let c = pool.alloc(8192).unwrap();
        assert!(pool.alloc(4096).is_none());
        let reclaimed = b.iova();
        // SAFETY: no hardware sees these test buffers.
        unsafe { core::ptr::write_bytes(b.cpu(), 0xa5, b.len()) };
        drop(a);
        assert!(pool.alloc(4096).is_none(), "drop must preserve quarantine");
        // SAFETY: the test owns the only reference and has no device mappings.
        assert!(unsafe { pool.release_quiesced(b) }.is_ok());
        let reused = pool.alloc(4096).unwrap();
        assert_eq!(reused.iova(), reclaimed);
        assert!(
            unsafe { core::slice::from_raw_parts(reused.cpu(), reused.len()) }
                .iter()
                .all(|byte| *byte == 0)
        );
        // A different allocator must not accept this pool's buffer identity.
        let other = CapPool::new(cpu, DMA_IOVA_BASE, layout.size() as u64, &ARENA).unwrap();
        let DmaReleaseFailure::Refused(reused) =
            unsafe { other.release_quiesced(reused) }.unwrap_err()
        else {
            panic!("foreign allocation must refuse, not claim unsupported recycling");
        };
        assert!(pool.alloc(4096).is_none());
        assert!(unsafe { pool.release_quiesced(reused) }.is_ok());
        drop((c, pool, other));
        unsafe { dealloc(cpu, layout) };
    }

    #[test]
    fn cap_pool_growth_preserves_existing_allocations_and_releases_by_region() {
        use alloc::alloc::{Layout, alloc_zeroed, dealloc};
        static ARENA: test_arena::TestArena<4096> = test_arena::TestArena::new();
        let layout = Layout::from_size_align(4 * 4096, 4096).unwrap();
        let cpu = unsafe { alloc_zeroed(layout) };
        assert!(!cpu.is_null());
        let pool = CapPool::new(cpu, DMA_IOVA_BASE, 8192, &ARENA).unwrap();
        let original = pool.alloc(8192).unwrap();
        assert!(pool.alloc(4096).is_none());
        // Model the owner's publication of a second disjoint retained window.
        let extension = PoolRegion::new(
            unsafe { cpu.add(8192) },
            DMA_IOVA_BASE + (2 << 20),
            8192,
            &ARENA,
        )
        .unwrap();
        pool.regions.borrow_mut().push(extension);
        assert_eq!(pool.capacity_bytes(), 16384);
        let added = pool.alloc(8192).unwrap();
        assert_eq!(original.iova().0, DMA_IOVA_BASE);
        assert_eq!(added.iova().0, DMA_IOVA_BASE + (2 << 20));
        assert!(unsafe { pool.release_quiesced(original) }.is_ok());
        assert!(unsafe { pool.release_quiesced(added) }.is_ok());
        assert!(
            pool.alloc(12288).is_none(),
            "one buffer cannot span disjoint windows"
        );
        assert_eq!(pool.alloc(8192).unwrap().iova().0, DMA_IOVA_BASE);
        drop(pool);
        unsafe { dealloc(cpu, layout) };
    }

    #[test]
    fn cap_pool_refuses_incongruent_cpu_and_iova_alignment() {
        use alloc::alloc::{Layout, alloc_zeroed, dealloc};

        let layout = Layout::from_size_align(16 * 1024, 16 * 1024).unwrap();
        // SAFETY: as in the preceding test.
        let cpu = unsafe { alloc_zeroed(layout) };
        assert!(!cpu.is_null());
        static ARENA: test_arena::TestArena<4096> = test_arena::TestArena::new();
        let pool = CapPool::new(cpu, DMA_IOVA_BASE + 4096, layout.size() as u64, &ARENA).unwrap();
        assert!(pool.alloc_aligned(4096, 8192).is_none());
        drop(pool);
        // SAFETY: `cpu` came from this exact layout and no view is live now.
        unsafe { dealloc(cpu, layout) };
    }

    macro_rules! declaration {
        ($name:ident, $bars:expr, $interrupts:expr) => {
            struct $name;
            impl PciBindingContract for $name {
                const BARS: PciBarMask = $bars;
                const INTERRUPTS: InterruptPolicy = $interrupts;
            }
            impl DeviceBinding for $name {
                type Pci = Self;
                type Config = ();
                type Prepared = ();
                type Bound = ();
                type Error = ();
                fn prepare(_: &mut DeviceResources<DmaPrepared>, (): ()) -> Result<(), ()> {
                    Ok(())
                }
                fn start(_: (), _: DeviceResources<DmaActive>) -> Result<(), ()> {
                    Ok(())
                }
                fn quiesce(_: &mut ()) -> Result<ClassQuiesce, ()> {
                    Ok(ClassQuiesce::NothingToStop)
                }
                fn unbind(_: ()) -> DeviceResources<DmaActive> {
                    unreachable!("declaration tests never bind")
                }
            }
        };
    }

    declaration!(
        StrictPair,
        PciBarMask::BAR0.union(PciBarMask::BAR4),
        InterruptPolicy::Required(InterruptBlock::MSIX_PAIR)
    );

    /// A bindable class that records what the host asked of it.  The
    /// crate is `no_std`, so the log is one atomic word: each event shifts
    /// in a nibble (`1` = class quiesce, `2` = unbind), and the order is
    /// the word's value.
    struct Recording;
    struct RecordingBound {
        refuse: bool,
        resources: DeviceResources<DmaActive>,
    }
    static SEQUENCE: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
    fn record(event: u32) {
        use core::sync::atomic::Ordering;
        let prior = SEQUENCE.load(Ordering::Acquire);
        SEQUENCE.store((prior << 4) | event, Ordering::Release);
    }
    impl PciBindingContract for Recording {
        // Never validated: the sequence test constructs its `Bound` directly.
        const BARS: PciBarMask = PciBarMask::BAR0;
        const INTERRUPTS: InterruptPolicy = InterruptPolicy::None;
    }
    impl DeviceBinding for Recording {
        type Pci = Self;
        type Config = ();
        type Prepared = ();
        type Bound = RecordingBound;
        type Error = &'static str;
        fn prepare(_: &mut DeviceResources<DmaPrepared>, (): ()) -> Result<(), &'static str> {
            unreachable!("the sequence test constructs its Bound directly")
        }
        fn start(_: (), _: DeviceResources<DmaActive>) -> Result<Self::Bound, &'static str> {
            unreachable!("the sequence test constructs its Bound directly")
        }
        fn quiesce(bound: &mut Self::Bound) -> Result<ClassQuiesce, &'static str> {
            record(1);
            if bound.refuse {
                Err("engine wedged")
            } else {
                Ok(ClassQuiesce::Stopped)
            }
        }
        fn unbind(bound: Self::Bound) -> DeviceResources<DmaActive> {
            record(2);
            bound.resources
        }
    }

    fn empty_resources() -> DeviceResources<DmaActive> {
        DeviceResources {
            route: None,
            expected_route: false,
            dma: None,
            pool: None,
            bars: [const { None }; PCI_BAR_COUNT],
            expected_bars: 0,
            _state: PhantomData,
        }
    }

    /// #809 §L: the host owns the terminal order — class stop, then unbind,
    /// then the fabric proof, then containment — and the last two run even
    /// when the class refuses.  Off the metal the fabric proof is
    /// `Unenforced`, which is exactly the point: it is the *host* that asked,
    /// on a resource set the class no longer holds.
    #[test]
    fn teardown_runs_class_then_fabric_then_containment_even_on_refusal() {
        for refuse in [false, true] {
            SEQUENCE.store(0, core::sync::atomic::Ordering::Release);
            let bound = BoundDevice::<Recording> {
                bound: Some(RecordingBound {
                    refuse,
                    resources: empty_resources(),
                }),
                _class: PhantomData,
            };
            let outcome = bound.teardown();
            assert_eq!(
                SEQUENCE.load(core::sync::atomic::Ordering::Acquire),
                0x12,
                "the class stops its engine before it loses the resource set"
            );
            assert_eq!(
                outcome.quiesce,
                if refuse {
                    Err("engine wedged")
                } else {
                    Ok(ClassQuiesce::Stopped)
                }
            );
            assert_eq!(
                outcome.fabric,
                Err(QuiesceFault::Unenforced),
                "the proof was asked for regardless of the class verdict"
            );
            // Containment ran: an empty declaration retires completely except
            // for the DMA owner it never had — the report says so honestly.
            assert!(outcome.resources.route);
            assert!(!outcome.resources.dma, "no DMA owner was ever realized");
            assert_eq!(outcome.resources.bars, 0);
        }
    }
    declaration!(
        FamilyPair,
        PciBarMask::BAR0,
        InterruptPolicy::Required(InterruptBlock {
            first_entry: 0,
            count: 2,
            affinity: None,
            source: InterruptSourcePolicy::PreferMsiX,
        })
    );
    declaration!(
        NonPowerOfTwoInterruptBlock,
        PciBarMask::BAR0,
        InterruptPolicy::Required(InterruptBlock {
            first_entry: 0,
            count: 3,
            affinity: None,
            source: InterruptSourcePolicy::PreferMsiX,
        })
    );
    declaration!(
        EmptyInterruptBlock,
        PciBarMask::BAR0,
        InterruptPolicy::Required(InterruptBlock {
            first_entry: 0,
            count: 0,
            affinity: None,
            source: InterruptSourcePolicy::PreferMsiX,
        })
    );
    declaration!(
        LastVector,
        PciBarMask::BAR5,
        InterruptPolicy::Required(InterruptBlock {
            first_entry: u16::MAX,
            count: 1,
            affinity: Some(0),
            source: InterruptSourcePolicy::RequireMsiX,
        })
    );

    #[test]
    fn interrupt_declaration_rejects_an_empty_block() {
        assert_eq!(
            validate_declaration::<EmptyInterruptBlock>().err(),
            Some(IntakeError::InvalidInterruptBlock)
        );
    }

    #[test]
    fn interrupt_declaration_rejects_a_non_power_of_two_block() {
        assert_eq!(
            validate_declaration::<NonPowerOfTwoInterruptBlock>().err(),
            Some(IntakeError::InvalidInterruptBlock)
        );
    }

    #[test]
    fn interrupt_block_rejects_an_unrepresentable_exclusive_end() {
        assert_eq!(
            validate_declaration::<LastVector>().err(),
            Some(IntakeError::InvalidInterruptBlock)
        );
    }

    #[test]
    fn strict_blocks_name_the_dedicated_msix_source_contract() {
        assert_eq!(InterruptBlock::MSIX_SINGLE.first_entry, 0);
        assert_eq!(InterruptBlock::MSIX_SINGLE.count, 1);
        assert_eq!(InterruptBlock::MSIX_SINGLE.affinity, None);
        assert_eq!(
            InterruptBlock::MSIX_SINGLE.source,
            InterruptSourcePolicy::RequireMsiX
        );
        assert_eq!(InterruptBlock::MSIX_PAIR.first_entry, 0);
        assert_eq!(InterruptBlock::MSIX_PAIR.count, 2);
        assert_eq!(InterruptBlock::MSIX_PAIR.affinity, None);
        assert_eq!(
            InterruptBlock::MSIX_PAIR.source,
            InterruptSourcePolicy::RequireMsiX
        );
        assert_eq!(
            kore_rt::service::PciMsixBlock::ENTRIES_0_1.first_entry(),
            InterruptBlock::MSIX_PAIR.first_entry
        );
        assert_eq!(
            kore_rt::service::PciMsixBlock::ENTRIES_0_1.count(),
            InterruptBlock::MSIX_PAIR.count
        );
    }

    #[test]
    fn typed_pci_selector_derives_binding_resources() {
        let selector = pci_selector_for_binding::<StrictPair>(
            0x03,
            0x02,
            0x00,
            PciIdentitySelector::exact(0x8086, 0x56a0),
            PciClaimPosture::RouteCapable,
        );
        assert_eq!(selector.bars, PciBarMask::BAR0.union(PciBarMask::BAR4));
        assert!(selector.dma);
        assert_eq!(
            selector.interrupts,
            PciInterruptRequirement::Msix(PciMsixBlock::ENTRIES_0_1)
        );
    }

    #[test]
    #[should_panic(expected = "no polling fallback")]
    fn strict_pci_selector_cannot_weaken_to_polling() {
        let _ = pci_selector_for_binding::<StrictPair>(
            0x03,
            0x02,
            0x00,
            PciIdentitySelector::exact(0x8086, 0x56a0),
            PciClaimPosture::PollingFallback,
        );
    }

    #[test]
    #[should_panic(expected = "only one message")]
    fn msi_family_selector_cannot_compress_a_multi_message_binding() {
        let _ = pci_selector_for_binding::<FamilyPair>(
            0x0c,
            0x03,
            0x30,
            PciIdentitySelector::ANY,
            PciClaimPosture::RouteCapable,
        );
    }

    #[test]
    fn moved_route_must_be_restored_for_truthful_teardown() {
        // Model the state after a class moved an intake-established affine
        // route into its controller but failed to put it back during unbind.
        // A RouteToken cannot be fabricated in a host test (deliberately), so
        // `expected_route` is the retained intake witness under test.
        let mut resources = DeviceResources::<DmaActive> {
            route: None,
            expected_route: true,
            dma: None,
            pool: None,
            bars: [const { None }; PCI_BAR_COUNT],
            expected_bars: 0,
            _state: PhantomData,
        };
        let report = resources.retire_parts();
        assert!(!report.route);
    }

    #[test]
    fn generic_host_orders_prepare_activate_start() {
        let phase = Cell::new(0u8);
        let bound = drive_lifecycle(
            (),
            |_| {
                assert_eq!(phase.replace(1), 0, "prepare is first");
                Ok::<_, ()>(11u8)
            },
            |_| {
                assert_eq!(phase.replace(2), 1, "activation follows prepare");
                Ok::<_, ()>(22u8)
            },
            |prepared, active| {
                assert_eq!(phase.replace(3), 2, "start follows activation");
                Ok::<_, ()>((prepared, active))
            },
        )
        .expect("lifecycle");

        assert_eq!(bound, (11, 22));
        assert_eq!(phase.get(), 3);
    }

    #[test]
    fn activation_refusal_never_reaches_start() {
        struct DropStep<'a> {
            phase: &'a Cell<u8>,
            expected: u8,
            next: u8,
        }

        impl Drop for DropStep<'_> {
            fn drop(&mut self) {
                assert_eq!(self.phase.replace(self.next), self.expected);
            }
        }

        let phase = Cell::new(0u8);
        let result = drive_lifecycle(
            DropStep {
                phase: &phase,
                expected: 3,
                next: 4,
            },
            |_| {
                phase.set(1);
                Ok::<_, ()>(DropStep {
                    phase: &phase,
                    expected: 2,
                    next: 3,
                })
            },
            |resources| {
                assert_eq!(phase.replace(2), 1);
                Err::<(), _>(resources)
            },
            |_, _| -> Result<(), ()> { panic!("start ran after activation refusal") },
        );

        assert_eq!(result, Err(LifecycleError::Activation));
        assert_eq!(phase.get(), 4, "prepared state drops before resources");
    }
}
