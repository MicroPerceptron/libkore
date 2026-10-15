//! The **platform seam** — the two small traits a driver core is generic
//! over, the kore-native replacement for a HAL's `MappedBar` / `DmaBuffer` /
//! `DmaContext`.
//!
//! Everything hardware-facing lives behind these, so the *same core* can be
//! exercised in three deliberately different contexts:
//!
//! - **kernel first-light** (a boot-test): [`Regs`] over `Mem::map_mmio_raw`,
//!   [`Dma`] over `Mem::lease_from` + `share_into_device`; this is a
//!   kernel-internal integration test, never the userland acceptance path.
//! - **unit tests**: `cfg(test)` hardware doubles prove pure controller logic
//!   and are not a deployable backend.
//! - **userland driver task**: `host::CapRegs` and `host::CapPool` are backed
//!   by affine BAR and DMA windows produced through real sessions. This is the
//!   production composition.
//!
//! This is the `spdk_env` idea — factor the platform behind a replaceable
//! seam — but as Rust traits, and far narrower than SPDK's because kore
//! already owns the DMA and MMIO capabilities.

use kore_abi::device::IovaAddr;

/// **Volatile MMIO register access** over a device BAR window.  Distinct
/// from a plain memory window (`kore_rt::cap::Window`): controller registers
/// have read/write side effects, so every access is `read_volatile` /
/// `write_volatile` — the impl must not elide, reorder, or coalesce them.
///
/// Offsets are byte offsets into the BAR; the core only ever touches
/// naturally-aligned register offsets.
pub trait Regs {
    /// Read a 32-bit register.
    fn read32(&self, off: usize) -> u32;
    /// Write a 32-bit register.
    fn write32(&self, off: usize, val: u32);

    /// Read a 64-bit register.  Default: two 32-bit reads, low then high
    /// (the NVMe-legal access for CAP/ASQ/ACQ on 32-bit-only BARs); an
    /// impl over a 64-bit-capable window overrides with one access.
    fn read64(&self, off: usize) -> u64 {
        (self.read32(off) as u64) | ((self.read32(off + 4) as u64) << 32)
    }
    /// Write a 64-bit register (low then high, per above).
    fn write64(&self, off: usize, val: u64) {
        self.write32(off, val as u32);
        self.write32(off + 4, (val >> 32) as u32);
    }
}

/// MMIO windows that can perform one naturally aligned 64-bit transaction.
///
/// This is deliberately a separate contract from [`Regs::write64`]. Several
/// controller ABIs (notably NVMe) define a low-then-high pair of 32-bit
/// accesses, while Intel's GGTT requires one `writeq`-shaped PTE store: two
/// writes can expose a torn translation to hardware. A mechanism that needs an
/// indivisible bus transaction takes this stronger trait and cannot silently
/// fall back to the compatibility default.
pub trait Regs64: Regs {
    /// Read one aligned 64-bit MMIO quantity in one volatile transaction.
    fn read64_once(&self, off: usize) -> u64;
    /// Write one aligned 64-bit MMIO quantity in one volatile transaction.
    fn write64_once(&self, off: usize, val: u64);
}

/// MMIO windows whose registers are also addressable as bytes and halfwords.
///
/// A separate contract from [`Regs`] for the same reason [`Regs64`] is:
/// a narrow access cannot be synthesized from a 32-bit read-modify-write
/// without changing what the device sees. The Realtek BE MAC keeps
/// write-one-to-clear counters and self-clearing triggers in the *other*
/// bytes of the dwords its byte-wide control registers sit in
/// (`R_BE_H2CREG_CTRL`, `R_BE_UDM1 + 1`), so a dword RMW would acknowledge
/// or fire what the driver never addressed. A core that programs such a
/// part takes this trait and cannot silently fall back to a wider access;
/// a core that never needs it (every other driver in the tree) is not
/// asked to implement it.
///
/// Offsets are byte offsets into the BAR, naturally aligned for the width.
pub trait RegsNarrow: Regs {
    /// Read one byte.
    fn read8(&self, off: usize) -> u8;
    /// Write one byte.
    fn write8(&self, off: usize, val: u8);
    /// Read one naturally aligned 16-bit register.
    fn read16(&self, off: usize) -> u16;
    /// Write one naturally aligned 16-bit register.
    fn write16(&self, off: usize, val: u16);
}

/// A bounded borrowed subwindow of one affine register-window owner.
///
/// The view holds no capability and cannot outlive its parent. It lets a
/// driver split one BAR into typed regions (for example Xe BAR0 register MMIO
/// and the GSM aperture at BAR0 + 8 MiB) without duplicating BAR authority or
/// rebasing offsets by hand at every callsite.
#[derive(Clone, Copy, Debug)]
pub struct RegsWindow<'a, R: ?Sized> {
    regs: &'a R,
    region: RegsRegion,
}

/// Invalid geometry for a checked register subwindow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegsWindowError {
    /// `base + len` overflowed or exceeded the mapped parent.
    OutOfRange,
    /// The subwindow base cannot preserve the strongest supported register
    /// alignment (64-bit). This also satisfies every 32-bit-only user.
    MisalignedBase,
}

/// Checked geometry shared by borrowed and lifecycle-owned register views.
///
/// The parent-relative base deliberately has no accessor: callers choose the
/// layout once while constructing a view, then operate only in that view's
/// local coordinate space. This keeps BAR rebasing out of controller cores.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RegsRegion {
    base: usize,
    len: usize,
}

impl RegsRegion {
    /// Validate `[base, base + len)` against one mapped parent.
    pub(crate) const fn new(
        parent_len: usize,
        base: usize,
        len: usize,
    ) -> Result<Self, RegsWindowError> {
        if !base.is_multiple_of(core::mem::align_of::<u64>()) {
            return Err(RegsWindowError::MisalignedBase);
        }
        let end = match base.checked_add(len) {
            Some(end) => end,
            None => return Err(RegsWindowError::OutOfRange),
        };
        if end > parent_len {
            return Err(RegsWindowError::OutOfRange);
        }
        Ok(Self { base, len })
    }

    pub(crate) const fn len(self) -> usize {
        self.len
    }

    pub(crate) const fn is_empty(self) -> bool {
        self.len == 0
    }

    pub(crate) fn offset(&self, off: usize, width: usize, align: usize) -> usize {
        assert!(
            off.is_multiple_of(align),
            "unaligned subwindow register access"
        );
        let end = off.checked_add(width).expect("register offset overflow");
        assert!(end <= self.len, "register access past the subwindow");
        self.base + off
    }
}

impl<'a, R: ?Sized> RegsWindow<'a, R> {
    /// Borrow `[base, base + len)` from a parent of `parent_len` bytes.
    pub const fn new(
        regs: &'a R,
        parent_len: usize,
        base: usize,
        len: usize,
    ) -> Result<Self, RegsWindowError> {
        let region = match RegsRegion::new(parent_len, base, len) {
            Ok(region) => region,
            Err(error) => return Err(error),
        };
        Ok(Self { regs, region })
    }

    /// Subwindow length.
    pub const fn len(self) -> usize {
        self.region.len()
    }

    /// Whether the subwindow is empty.
    pub const fn is_empty(self) -> bool {
        self.region.is_empty()
    }

    fn offset(&self, off: usize, width: usize, align: usize) -> usize {
        self.region.offset(off, width, align)
    }
}

impl<R: Regs + ?Sized> Regs for RegsWindow<'_, R> {
    #[inline]
    fn read32(&self, off: usize) -> u32 {
        self.regs.read32(self.offset(
            off,
            core::mem::size_of::<u32>(),
            core::mem::align_of::<u32>(),
        ))
    }

    #[inline]
    fn write32(&self, off: usize, val: u32) {
        self.regs.write32(
            self.offset(
                off,
                core::mem::size_of::<u32>(),
                core::mem::align_of::<u32>(),
            ),
            val,
        );
    }
}

impl<R: Regs64 + ?Sized> Regs64 for RegsWindow<'_, R> {
    #[inline]
    fn read64_once(&self, off: usize) -> u64 {
        self.regs.read64_once(self.offset(
            off,
            core::mem::size_of::<u64>(),
            core::mem::align_of::<u64>(),
        ))
    }

    #[inline]
    fn write64_once(&self, off: usize, val: u64) {
        self.regs.write64_once(
            self.offset(
                off,
                core::mem::size_of::<u64>(),
                core::mem::align_of::<u64>(),
            ),
            val,
        );
    }
}

impl<R: RegsNarrow + ?Sized> RegsNarrow for RegsWindow<'_, R> {
    #[inline]
    fn read8(&self, off: usize) -> u8 {
        self.regs.read8(self.offset(off, 1, 1))
    }

    #[inline]
    fn write8(&self, off: usize, val: u8) {
        self.regs.write8(self.offset(off, 1, 1), val);
    }

    #[inline]
    fn read16(&self, off: usize) -> u16 {
        self.regs.read16(self.offset(
            off,
            core::mem::size_of::<u16>(),
            core::mem::align_of::<u16>(),
        ))
    }

    #[inline]
    fn write16(&self, off: usize, val: u16) {
        self.regs.write16(
            self.offset(
                off,
                core::mem::size_of::<u16>(),
                core::mem::align_of::<u16>(),
            ),
            val,
        );
    }
}

/// A **device-visible DMA buffer** — the RUNG-2 lease's two windows,
/// abstracted: a CPU-side pointer the driver writes/reads through, and the
/// [`IovaAddr`] the device sees for the same frames (typed — "no phys in
/// userland" is a type, never a raw `u64`).
///
/// # The contiguity contract
///
/// [`iova`](Dma::iova)`..iova + len` is **one contiguous device-visible
/// window**.  This holds per seam impl for different reasons, and each
/// must uphold it:
/// - IOMMU-backed (userland / bound devices): the IOVA window is
///   contiguous *by construction* even over scattered physical frames —
///   that is what the IOMMU leaf provides (RUNG 2).
/// - identity/unbound (the kernel boot-test): `iova == phys`, and the
///   backing lease is contiguous (`MemLease` refuses scattered).
///
/// Ring and descriptor-table math leans on this: within one buffer, byte
/// `n` is at `iova + n`.  A future backing that cannot promise it must
/// model scatter explicitly — the assumption is a stated contract here,
/// never silent.
///
/// # Safety
///
/// [`cpu`](Dma::cpu) must be valid for [`len`](Dma::len) bytes for the
/// buffer's whole life, and [`iova`](Dma::iova) must name the *same*
/// frames in the device's domain (zero-copy — no bounce buffer).
/// Device-concurrent scalar fields must go through [`DmaShared`]; bulk regions
/// use [`DmaRead`] / [`DmaWrite`] only while their protocol grants the CPU
/// ownership. Neither access seam supplies cache maintenance or a device
/// memory barrier: those remain explicit at each publication handoff.
pub trait Dma {
    /// CPU-visible base of the buffer.
    fn cpu(&self) -> *mut u8;
    /// Device-visible base of the *same* frames — one contiguous window
    /// (see the contract above).
    fn iova(&self) -> IovaAddr;
    /// Buffer length in bytes.
    fn len(&self) -> usize;

    /// Whether the native owner can recover independent allocation custody
    /// after quiescent detachment. This fact neither authorizes nor performs
    /// a return; ordinary pool slices keep the conservative default.
    fn can_return_to_owner(&self) -> bool {
        false
    }

    /// Whether the buffer is empty (the clippy convention pair).
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A structured write that would leave the buffer — refused, never clamped.
///
/// Carries enough to name the fault precisely, because the usual cause is a
/// mis-transcribed layout constant (an offset copied from the wrong `#define`),
/// and a bare "too small" hides which field was wrong.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutOfRange {
    /// Byte offset the write started at.
    pub offset: usize,
    /// Length the write wanted.
    pub len: usize,
    /// The buffer's actual length.
    pub buffer: usize,
}

/// A scalar shared-field access that could not be performed exactly.
///
/// Unlike [`DmaRead`] / [`DmaWrite`], a device-concurrent field is one typed
/// volatile transaction. Such a transaction may not be synthesized from
/// byte copies: in addition to the ordinary range proof it requires the
/// effective CPU address to be naturally aligned for its width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SharedDmaError {
    /// The complete scalar would not fit in the DMA buffer.
    Range(OutOfRange),
    /// `cpu + offset` was not naturally aligned for this scalar width.
    Misaligned {
        /// Byte offset requested within the DMA buffer.
        offset: usize,
        /// Width of the scalar transaction.
        width: usize,
        /// Required effective-address alignment.
        alignment: usize,
    },
}

impl From<OutOfRange> for SharedDmaError {
    fn from(error: OutOfRange) -> Self {
        Self::Range(error)
    }
}

/// A bounded volatile view of device-concurrent scalar fields in DMA memory.
///
/// This seam is deliberately a borrowed wrapper rather than another blanket
/// extension trait: callsites must opt into the stronger semantics visibly.
/// Use it only for ownership/status/index/fence words that either side may
/// observe while the other runs. Bulk descriptors and payloads remain plain
/// [`DmaRead`] / [`DmaWrite`] copies, published or acquired by one of these
/// control fields plus the caller's explicit ordering fence.
///
/// Every operation is exactly one naturally aligned `read_volatile` or
/// `write_volatile`, with explicit little-endian conversion. Volatile prevents
/// the compiler from eliding, inventing, or merging the access; it is **not**
/// a cache-coherency operation, an atomicity guarantee, or a device/CPU memory
/// barrier. Non-coherent users must clean/invalidate the relevant cache lines,
/// and producer/consumer protocols must retain their release/acquire fences.
#[derive(Clone, Copy, Debug)]
pub struct DmaShared<'a, D: ?Sized> {
    dma: &'a D,
}

impl<'a, D: Dma + ?Sized> DmaShared<'a, D> {
    /// Borrow the shared scalar fields of `dma`.
    pub const fn new(dma: &'a D) -> Self {
        Self { dma }
    }

    fn checked_ptr(
        &self,
        offset: usize,
        width: usize,
        alignment: usize,
    ) -> Result<*mut u8, SharedDmaError> {
        let end = offset.checked_add(width).ok_or_else(|| {
            SharedDmaError::Range(OutOfRange {
                offset,
                len: width,
                buffer: self.dma.len(),
            })
        })?;
        if end > self.dma.len() {
            return Err(SharedDmaError::Range(OutOfRange {
                offset,
                len: width,
                buffer: self.dma.len(),
            }));
        }
        // SAFETY: `Dma::cpu` is valid for `Dma::len` bytes and the complete
        // non-empty scalar range was proved to lie inside that allocation.
        let ptr = unsafe { self.dma.cpu().add(offset) };
        if !(ptr as usize).is_multiple_of(alignment) {
            return Err(SharedDmaError::Misaligned {
                offset,
                width,
                alignment,
            });
        }
        Ok(ptr)
    }

    /// Validate the bounds and effective natural alignment of a shared `u32`
    /// without touching its possibly uninitialized value.
    ///
    /// Constructors use this to reject malformed DMA geometry before taking a
    /// ring slot or publishing any device-visible state. The [`Dma`] contract
    /// keeps the CPU base and length stable, so a successful check remains
    /// valid for this buffer's life.
    #[inline]
    pub fn validate_u32_at(&self, offset: usize) -> Result<(), SharedDmaError> {
        self.checked_ptr(offset, 4, core::mem::align_of::<u32>())?;
        Ok(())
    }

    /// Volatile-load one shared byte.
    #[inline]
    pub fn read_u8_at(&self, offset: usize) -> Result<u8, SharedDmaError> {
        let ptr = self.checked_ptr(offset, 1, 1)?;
        // SAFETY: `checked_ptr` proved one valid byte; every byte is aligned.
        Ok(unsafe { ptr.read_volatile() })
    }

    /// Volatile-load one naturally aligned little-endian `u16`.
    #[inline]
    pub fn read_u16_at(&self, offset: usize) -> Result<u16, SharedDmaError> {
        let ptr = self.checked_ptr(offset, 2, core::mem::align_of::<u16>())?;
        // SAFETY: the complete field is in-bounds and naturally aligned.
        Ok(u16::from_le(unsafe { ptr.cast::<u16>().read_volatile() }))
    }

    /// Volatile-load one naturally aligned little-endian `u32`.
    #[inline]
    pub fn read_u32_at(&self, offset: usize) -> Result<u32, SharedDmaError> {
        let ptr = self.checked_ptr(offset, 4, core::mem::align_of::<u32>())?;
        // SAFETY: the complete field is in-bounds and naturally aligned.
        Ok(u32::from_le(unsafe { ptr.cast::<u32>().read_volatile() }))
    }

    /// Volatile-load one naturally aligned little-endian `u64`.
    #[inline]
    pub fn read_u64_at(&self, offset: usize) -> Result<u64, SharedDmaError> {
        let ptr = self.checked_ptr(offset, 8, core::mem::align_of::<u64>())?;
        // SAFETY: the complete field is in-bounds and naturally aligned.
        Ok(u64::from_le(unsafe { ptr.cast::<u64>().read_volatile() }))
    }

    /// Volatile-store one shared byte.
    #[inline]
    pub fn write_u8_at(&self, offset: usize, value: u8) -> Result<(), SharedDmaError> {
        let ptr = self.checked_ptr(offset, 1, 1)?;
        // SAFETY: `checked_ptr` proved one valid byte; every byte is aligned.
        unsafe { ptr.write_volatile(value) };
        Ok(())
    }

    /// Volatile-store one naturally aligned little-endian `u16`.
    #[inline]
    pub fn write_u16_at(&self, offset: usize, value: u16) -> Result<(), SharedDmaError> {
        let ptr = self.checked_ptr(offset, 2, core::mem::align_of::<u16>())?;
        // SAFETY: the complete field is in-bounds and naturally aligned.
        unsafe { ptr.cast::<u16>().write_volatile(value.to_le()) };
        Ok(())
    }

    /// Volatile-store one naturally aligned little-endian `u32`.
    #[inline]
    pub fn write_u32_at(&self, offset: usize, value: u32) -> Result<(), SharedDmaError> {
        let ptr = self.checked_ptr(offset, 4, core::mem::align_of::<u32>())?;
        // SAFETY: the complete field is in-bounds and naturally aligned.
        unsafe { ptr.cast::<u32>().write_volatile(value.to_le()) };
        Ok(())
    }

    /// Volatile-store one naturally aligned little-endian `u64`.
    #[inline]
    pub fn write_u64_at(&self, offset: usize, value: u64) -> Result<(), SharedDmaError> {
        let ptr = self.checked_ptr(offset, 8, core::mem::align_of::<u64>())?;
        // SAFETY: the complete field is in-bounds and naturally aligned.
        unsafe { ptr.cast::<u64>().write_volatile(value.to_le()) };
        Ok(())
    }
}

/// **Structured writes into a device-visible buffer** — bounds-checked,
/// little-endian, and *unaligned-safe*.
///
/// Device memory layouts are packed C structs, so a field's natural alignment
/// and its offset routinely disagree: the VPU's `vpu_boot_params` is
/// `#pragma pack(push, 4)`, which puts `u64` fields at 4-aligned but **not**
/// 8-aligned offsets (`shared_region_base` at 268, `crit_tracing_buff_addr` at
/// 836). Writing those through a `*mut u64` is undefined behaviour — it may
/// fault or tear. Every write here goes through `write_unaligned` on bytes, so
/// an offset that came straight out of the vendor header is always safe to use.
///
/// Endianness is **explicit** rather than host-native: the device reads these
/// bytes with its own CPU, and the driver core must encode identically whether
/// it is compiled for x86_64, aarch64, or riscv64.
///
/// # Ordering
///
/// These are plain (non-volatile) writes, matching the `wmb()`-then-doorbell
/// shape of the C drivers. Ordering against the hardware is the **caller's**
/// job: issue a release fence after the last field and before the doorbell /
/// boot handshake that tells the device to read the region. Do not rely on
/// these calls for ordering.
///
/// Blanket-implemented for every [`Dma`] buffer; there is nothing to write.
pub trait DmaWrite: Dma {
    /// Resolve `off..off + len` to a CPU pointer, or refuse.
    ///
    /// The one bounds check the rest of the trait is built on. `len == 0` is
    /// accepted at any in-bounds offset (including `off == self.len()`).
    fn checked_ptr(&self, off: usize, len: usize) -> Result<*mut u8, OutOfRange> {
        let end = off.checked_add(len).ok_or(OutOfRange {
            offset: off,
            len,
            buffer: self.len(),
        })?;
        if end > self.len() {
            return Err(OutOfRange {
                offset: off,
                len,
                buffer: self.len(),
            });
        }
        // SAFETY: `cpu()` is valid for `len()` bytes for the buffer's whole
        // life (the `Dma` contract) and `off <= off + len <= len()` was just
        // checked, so the offset stays inside that one allocation.
        Ok(unsafe { self.cpu().add(off) })
    }

    /// Fill `off..off + len` with `byte` (the `memset` a device region needs
    /// before a partial field write, so stale bytes never reach the device).
    fn fill_at(&self, off: usize, len: usize, byte: u8) -> Result<(), OutOfRange> {
        let dst = self.checked_ptr(off, len)?;
        // SAFETY: `checked_ptr` returned a pointer valid for `len` bytes.
        unsafe { core::ptr::write_bytes(dst, byte, len) };
        Ok(())
    }

    /// Copy `src` to `off`.
    fn write_bytes_at(&self, off: usize, src: &[u8]) -> Result<(), OutOfRange> {
        let dst = self.checked_ptr(off, src.len())?;
        // SAFETY: `checked_ptr` returned a pointer valid for `src.len()`
        // bytes, and `src` is a distinct borrow — the regions cannot overlap.
        unsafe { core::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
        Ok(())
    }

    /// Write a little-endian `u16` at `off` — any alignment.
    fn write_u16_at(&self, off: usize, val: u16) -> Result<(), OutOfRange> {
        self.write_bytes_at(off, &val.to_le_bytes())
    }

    /// Write a little-endian `u32` at `off` — any alignment.
    fn write_u32_at(&self, off: usize, val: u32) -> Result<(), OutOfRange> {
        self.write_bytes_at(off, &val.to_le_bytes())
    }

    /// Write a little-endian `u64` at `off` — any alignment (see the trait
    /// docs: packed device layouts routinely place these off natural alignment).
    fn write_u64_at(&self, off: usize, val: u64) -> Result<(), OutOfRange> {
        self.write_bytes_at(off, &val.to_le_bytes())
    }

    /// Write a little-endian `i32` at `off` — the signed IRQ-redirect fields.
    fn write_i32_at(&self, off: usize, val: i32) -> Result<(), OutOfRange> {
        self.write_bytes_at(off, &val.to_le_bytes())
    }

    /// Write a single byte at `off`.
    fn write_u8_at(&self, off: usize, val: u8) -> Result<(), OutOfRange> {
        self.write_bytes_at(off, &[val])
    }
}

impl<T: Dma + ?Sized> DmaWrite for T {}

/// **Structured reads out of a device-visible buffer** — the mirror of
/// [`DmaWrite`], with the same bounds checking, unaligned safety, and explicit
/// little-endian decoding.
///
/// This is the side that reads what the *device* wrote, so the bounds check is
/// doing real work: an address the firmware put in a descriptor is untrusted
/// input, and resolving it against the region is the difference between a
/// refusal and a wild read.
///
/// Blanket-implemented for every [`Dma`] buffer.
pub trait DmaRead: Dma {
    /// Resolve `off..off + len` to a readable pointer, or refuse.
    fn checked_read_ptr(&self, off: usize, len: usize) -> Result<*const u8, OutOfRange> {
        let end = off.checked_add(len).ok_or(OutOfRange {
            offset: off,
            len,
            buffer: self.len(),
        })?;
        if end > self.len() {
            return Err(OutOfRange {
                offset: off,
                len,
                buffer: self.len(),
            });
        }
        // SAFETY: `cpu()` is valid for `len()` bytes for the buffer's whole
        // life (the `Dma` contract) and the range was just checked.
        Ok(unsafe { self.cpu().add(off) as *const u8 })
    }

    /// Read a byte at `off`.
    fn read_u8_at(&self, off: usize) -> Result<u8, OutOfRange> {
        let src = self.checked_read_ptr(off, 1)?;
        // SAFETY: `checked_read_ptr` returned a pointer valid for 1 byte.
        Ok(unsafe { src.read() })
    }

    /// Read a little-endian `u16` at `off` — any alignment.
    fn read_u16_at(&self, off: usize) -> Result<u16, OutOfRange> {
        let mut b = [0u8; 2];
        self.read_bytes_at(off, &mut b)?;
        Ok(u16::from_le_bytes(b))
    }

    /// Read a little-endian `u32` at `off` — any alignment.
    fn read_u32_at(&self, off: usize) -> Result<u32, OutOfRange> {
        let mut b = [0u8; 4];
        self.read_bytes_at(off, &mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    /// Read a little-endian `u64` at `off` — any alignment.
    fn read_u64_at(&self, off: usize) -> Result<u64, OutOfRange> {
        let mut b = [0u8; 8];
        self.read_bytes_at(off, &mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    /// Copy `dst.len()` bytes from `off` into `dst`.
    fn read_bytes_at(&self, off: usize, dst: &mut [u8]) -> Result<(), OutOfRange> {
        let src = self.checked_read_ptr(off, dst.len())?;
        // SAFETY: `checked_read_ptr` returned a pointer valid for `dst.len()`
        // bytes, and `dst` is a distinct exclusive borrow — no overlap.
        unsafe { core::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), dst.len()) };
        Ok(())
    }
}

impl<T: Dma + ?Sized> DmaRead for T {}

/// A DMA target its engine handed back with the proof that retiring it needs
/// no device idleness: no command it popped still names the target, the
/// completion of every one that did has been reaped, and no entry not yet
/// popped can reach it any more.
///
/// Only an engine mints one ([`StorEngine::release_target`]). It is the one
/// input per-window DMA retirement takes, in place of the whole-requester
/// quiescence an exact retirement otherwise waits for. The proof is the
/// driver's own correctness: the kernel's live retirement completes the
/// window's invalidation before the memory can move on, so a proof that
/// were wrong costs a recorded IOMMU fault, not a stray write.
///
/// [`StorEngine::release_target`]: crate::stor::engine::StorEngine::release_target
#[must_use = "a quiet target is retired, not dropped"]
pub struct Quiet<T>(T);

impl<T> Quiet<T> {
    pub(crate) fn witness(target: T) -> Self {
        Self(target)
    }

    /// The same memory in another wrapper — a queue's buffer type to the
    /// allocation beneath it.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Quiet<U> {
        Quiet(f(self.0))
    }

    pub fn get(&self) -> &T {
        &self.0
    }

    /// Drop the proof, keeping the target.
    pub fn into_inner(self) -> T {
        self.0
    }
}

/// An allocation returned by a DMA release attempt. These states differ in
/// whether ordinary buffer destruction is known to be safe.
#[derive(Debug)]
#[must_use = "a refused DMA release retains its allocation owner"]
pub enum DmaReleaseFailure<B> {
    /// The pool has no recycling operation. Device access is already quiescent;
    /// ordinary destruction is permitted, but no capacity reuse was promised.
    NotReclaimed(B),
    /// Retirement was refused. Preserve the owner for retry or containment;
    /// dropping it must not turn an unfinished mapping teardown into reuse.
    Refused(B),
}

/// A **provider of device-visible DMA buffers** — one per driver, handing
/// out ring memory, identify/descriptor buffers, and list pages.  Bounded
/// and fallible: exhaustion returns `None` (an `AllocFailed` refusal
/// upstream, never a panic).
pub trait DmaPool {
    /// The concrete buffer type this pool hands out.
    type Buf: Dma;

    /// Allocate a device-visible buffer of at least `bytes`, page-aligned
    /// (rings and list pages require 4 KiB alignment), zeroed.  `None`
    /// when the pool is exhausted.
    fn alloc(&self, bytes: usize) -> Option<Self::Buf>;

    /// Whether this exact live allocation belongs to this pool's device
    /// lifetime. Import adoption must authenticate provenance before mapping
    /// it; an address falling inside a window is insufficient. Pools that do
    /// not expose adoption conservatively refuse this projection.
    fn owns(&self, _buffer: &Self::Buf) -> bool {
        false
    }

    /// Return an allocation after all device references and translations have
    /// been quiesced. Pools without reclamation return `NotReclaimed` with the
    /// unchanged buffer. Failed retirement instead returns `Refused`; callers
    /// must retain that owner for retry or containment, never discard the error.
    /// Dropping a buffer is deliberately not a reclamation operation.
    ///
    /// # Safety
    /// The caller must prove no device can access this allocation, including
    /// through stale translations or an outstanding command. The allocation
    /// must not be reused after successful release.
    unsafe fn release_quiesced(
        &self,
        buffer: Self::Buf,
    ) -> Result<(), DmaReleaseFailure<Self::Buf>> {
        Err(DmaReleaseFailure::NotReclaimed(buffer))
    }

    /// Allocate with a stronger CPU-and-device address alignment. The base
    /// contract guarantees 4 KiB; pools that can honor larger hardware pages
    /// override this method. The default fails closed instead of returning a
    /// buffer whose address violates a queue/list-page encoding.
    fn alloc_aligned(&self, bytes: usize, alignment: usize) -> Option<Self::Buf> {
        if alignment != 4096 {
            return None;
        }
        self.alloc(bytes)
    }
}

#[cfg(test)]
mod tests {
    use alloc::boxed::Box;
    use alloc::vec;
    use core::cell::Cell;

    use super::*;

    struct RegisterFile {
        words: [Cell<u64>; 8],
        wide_reads: Cell<usize>,
        wide_writes: Cell<usize>,
    }

    impl RegisterFile {
        fn new() -> Self {
            Self {
                words: core::array::from_fn(|_| Cell::new(0)),
                wide_reads: Cell::new(0),
                wide_writes: Cell::new(0),
            }
        }
    }

    impl Regs for RegisterFile {
        fn read32(&self, off: usize) -> u32 {
            let word = self.words[off / 8].get();
            if off.is_multiple_of(8) {
                word as u32
            } else {
                (word >> 32) as u32
            }
        }

        fn write32(&self, off: usize, val: u32) {
            let slot = &self.words[off / 8];
            let prior = slot.get();
            if off.is_multiple_of(8) {
                slot.set((prior & !u32::MAX as u64) | val as u64);
            } else {
                slot.set((prior & u32::MAX as u64) | ((val as u64) << 32));
            }
        }
    }

    impl Regs64 for RegisterFile {
        fn read64_once(&self, off: usize) -> u64 {
            self.wide_reads.set(self.wide_reads.get() + 1);
            self.words[off / 8].get()
        }

        fn write64_once(&self, off: usize, val: u64) {
            self.wide_writes.set(self.wide_writes.get() + 1);
            self.words[off / 8].set(val);
        }
    }

    #[test]
    fn register_subwindow_rebases_and_preserves_one_wide_transaction() {
        let regs = RegisterFile::new();
        let view = RegsWindow::new(&regs, 64, 16, 24).unwrap();

        view.write64_once(8, 0x0123_4567_89ab_cdef);
        assert_eq!(regs.words[3].get(), 0x0123_4567_89ab_cdef);
        assert_eq!(view.read64_once(8), 0x0123_4567_89ab_cdef);
        assert_eq!(regs.wide_writes.get(), 1);
        assert_eq!(regs.wide_reads.get(), 1);

        assert!(matches!(
            RegsWindow::new(&regs, 64, 4, 8),
            Err(RegsWindowError::MisalignedBase)
        ));
        assert!(matches!(
            RegsWindow::new(&regs, 64, 56, 16),
            Err(RegsWindowError::OutOfRange)
        ));
    }

    #[test]
    fn checked_regions_admit_xe_registers_and_gsm_but_not_absent_apertures() {
        const XE_REGS_LEN: usize = 4 * 1024 * 1024;
        const GSM_BASE: usize = 8 * 1024 * 1024;
        const BAR0_LEN: usize = 16 * 1024 * 1024;

        let ordinary = RegsRegion::new(BAR0_LEN, 0, XE_REGS_LEN).unwrap();
        let gsm = RegsRegion::new(BAR0_LEN, GSM_BASE, BAR0_LEN - GSM_BASE).unwrap();
        assert_eq!(ordinary.len(), XE_REGS_LEN);
        assert_eq!(gsm.len(), BAR0_LEN - GSM_BASE);
        assert!(!ordinary.is_empty());
        assert!(!gsm.is_empty());

        assert!(matches!(
            RegsRegion::new(GSM_BASE, GSM_BASE, 8),
            Err(RegsWindowError::OutOfRange)
        ));
        assert!(matches!(
            RegsRegion::new(usize::MAX, usize::MAX - 7, 16),
            Err(RegsWindowError::OutOfRange)
        ));
    }

    /// A host-memory DMA buffer, deliberately **odd-length and offset by one**
    /// inside its backing so nothing in the test accidentally sits on a
    /// convenient alignment.
    struct HostBuf {
        mem: Box<[u8]>,
    }
    impl HostBuf {
        fn new(bytes: usize) -> Self {
            Self {
                mem: vec![0u8; bytes].into_boxed_slice(),
            }
        }
        fn bytes(&self) -> &[u8] {
            &self.mem
        }
    }
    impl Dma for HostBuf {
        fn cpu(&self) -> *mut u8 {
            self.mem.as_ptr() as *mut u8
        }
        fn iova(&self) -> IovaAddr {
            IovaAddr(0x1000)
        }
        fn len(&self) -> usize {
            self.mem.len()
        }
    }

    /// The reason this trait exists: a `u64` at an offset that is 4-aligned
    /// but **not** 8-aligned — exactly where `#pragma pack(4)` puts
    /// `vpu_boot_params::shared_region_base` (268). A `*mut u64` write here
    /// would be UB; this must land byte-for-byte, little-endian.
    #[test]
    fn u64_writes_survive_a_four_but_not_eight_aligned_offset() {
        let buf = HostBuf::new(64);
        const OFF: usize = 12; // 12 % 4 == 0, 12 % 8 == 4
        assert_eq!(buf.write_u64_at(OFF, 0x0102_0304_0506_0708), Ok(()));
        assert_eq!(
            &buf.bytes()[OFF..OFF + 8],
            &[0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01],
            "little-endian, unaligned, no tearing"
        );
        // Neighbours untouched — the write did not spill.
        assert!(buf.bytes()[..OFF].iter().all(|&b| b == 0));
        assert!(buf.bytes()[OFF + 8..].iter().all(|&b| b == 0));
    }

    /// Every width encodes little-endian regardless of host byte order, and a
    /// fully unaligned (odd) offset is fine.
    #[test]
    fn widths_encode_little_endian_at_any_offset() {
        let buf = HostBuf::new(32);
        buf.write_u32_at(1, 0xAABB_CCDD).unwrap();
        assert_eq!(&buf.bytes()[1..5], &[0xDD, 0xCC, 0xBB, 0xAA]);

        buf.write_i32_at(9, -2).unwrap(); // two's complement 0xFFFF_FFFE
        assert_eq!(&buf.bytes()[9..13], &[0xFE, 0xFF, 0xFF, 0xFF]);

        buf.write_u8_at(20, 0x5A).unwrap();
        assert_eq!(buf.bytes()[20], 0x5A);
    }

    /// Shared control words are one typed volatile transaction, so unlike the
    /// packed-copy seam they require the *effective address* to be naturally
    /// aligned. They retain the same exact bounds and little-endian contract.
    #[test]
    fn shared_scalars_are_aligned_bounded_and_little_endian() {
        let buf = HostBuf::new(64);
        let base = buf.cpu() as usize;
        let aligned = base.next_multiple_of(8) - base;
        let shared = DmaShared::new(&buf);

        assert_eq!(shared.write_u64_at(aligned, 0x0102_0304_0506_0708), Ok(()));
        assert_eq!(
            &buf.bytes()[aligned..aligned + 8],
            &[0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]
        );
        assert_eq!(shared.read_u64_at(aligned), Ok(0x0102_0304_0506_0708));

        let misaligned = aligned + 1;
        assert_eq!(
            shared.write_u64_at(misaligned, u64::MAX),
            Err(SharedDmaError::Misaligned {
                offset: misaligned,
                width: 8,
                alignment: 8,
            })
        );
        assert_eq!(
            shared.read_u64_at(60),
            Err(SharedDmaError::Range(OutOfRange {
                offset: 60,
                len: 8,
                buffer: 64,
            }))
        );
        assert!(matches!(
            shared.write_u32_at(usize::MAX - 1, 1),
            Err(SharedDmaError::Range(_))
        ));

        // Refused transactions are all-or-nothing; neither the misaligned nor
        // the straddling access disturbed its neighbours.
        assert!(buf.bytes()[aligned + 8..].iter().all(|&b| b == 0));
    }

    /// Out-of-range writes are **refused, not clamped**, and the refusal names
    /// the offset — a mis-transcribed layout constant must be identifiable.
    #[test]
    fn out_of_range_writes_are_refused_and_leave_the_buffer_clean() {
        let buf = HostBuf::new(16);

        // Straddling the end.
        assert_eq!(
            buf.write_u64_at(12, u64::MAX),
            Err(OutOfRange {
                offset: 12,
                len: 8,
                buffer: 16
            })
        );
        // Starting past the end.
        assert!(buf.write_u32_at(64, 1).is_err());
        // Overflowing the offset arithmetic must refuse, never wrap.
        assert!(buf.write_u32_at(usize::MAX - 1, 1).is_err());
        assert!(buf.fill_at(8, usize::MAX, 0xff).is_err());

        // Nothing partial was written by any of the refusals.
        assert!(buf.bytes().iter().all(|&b| b == 0), "refusals wrote bytes");
    }

    /// `fill_at` is the `memset` a device region needs before a partial field
    /// write, and a zero-length write at the exact end is legal.
    #[test]
    fn fill_covers_a_subrange_and_empty_writes_are_legal() {
        let buf = HostBuf::new(16);
        buf.fill_at(4, 8, 0xff).unwrap();
        assert_eq!(
            buf.bytes(),
            &[
                0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0
            ]
        );
        // A zero-length write at the boundary is in range, not off the end.
        assert_eq!(buf.write_bytes_at(16, &[]), Ok(()));
        assert!(buf.write_bytes_at(17, &[]).is_err());
    }
}
