//! The eager relocation engine — the stub-side apply half
//! (`res/IMAGE.md` §5, §7), generic over [`RelocArch`].
//!
//! ## Contract
//!
//! The caller (the startup stub) has already mapped a module's shares
//! and copied its data template; what it hands the engine is exactly the
//! authority the writes need — **its own writable segment slice** —
//! plus read views of the link inputs.  There is no address arithmetic
//! the caller didn't authorize by constructing the views:
//!
//! - `ModuleImage<'a>` — one loaded module: chosen `base`, the
//!   [`ModuleDescriptor`], read slices
//!   for rodata-resident link inputs (symtab/strtab/gnu-hash/rela ranges
//!   resolve inside them), and the **writable private data slice** (the
//!   only write target; a reloc whose `r_offset` lands outside it
//!   refuses — text/rodata are structurally unwritable here).
//! - `SymbolScope<'a>` — the resolution set: the module list in load
//!   order (dependency post-order for init running).  Lookup =
//!   GNU-hash per module, strong-before-weak across the scope
//!   (`-Bsymbolic-functions` has already bound most intra-module
//!   references at link time).
//! - `TlsLayout` — the static-block builder's yield: per-module
//!   `tp`-relative base offsets computed from
//!   [`RelocArch::TLS`] (V1: positive,
//!   after the TCB, alignment-rounded; V2: negative, blocks below `tp`),
//!   plus the total block size/align the stub allocates per thread.
//!   Built once for the whole closed load set (IE/LE legality).
//!
//! `apply_module::<A, _>(module, scope, tls) -> Result<Applied, ApplyError>`
//! walks `.rela.dyn`, `DT_RELR`, then `.rela.plt`, computing per
//! [`RelocKind`]:
//!
//! | kind | value |
//! |---|---|
//! | `Relative` | `base + addend` |
//! | `GlobDat`/`JumpSlot` | `S` |
//! | `Abs64` | `S + addend` |
//! | `TpRel` | `tls.tp_off(module, sym_off) + addend` |
//!
//! — all checked arithmetic (overflow refuses, never wraps), all writes
//! little-endian u64 through the data slice.  After apply, the stub runs
//! `DT_INIT_ARRAY`s (post-order) and seals RELRO; both are stub verbs —
//! the engine only *reports* the RELRO window
//! ([`Applied::relro`]).
//!
//! Failure is total: one bad relocation refuses the module (no partial
//! GOT — a half-linked module is unrepresentable to the caller, the
//! bounded-session discipline applied to linking).

use kore_abi::image::{MAX_SEGMENTS, ModuleDescriptor, seg};
use zerocopy::FromBytes;
use zerocopy::byteorder::little_endian::{I64 as LeI64, U16 as LeU16, U32 as LeU32, U64 as LeU64};

use crate::reloc::{RelocArch, RelocKind, TlsRules};

/// Why apply refused.  Diagnostics carry the relocation index — the
/// klink convention, kept.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApplyError {
    /// A relocation refused classification (already caught at `check`;
    /// here for defense in depth).
    Reloc(crate::reloc::RelocRefusal, usize),
    /// `r_offset` (or a RELR target) lands outside the writable data
    /// slice — the one memory-safety gate of the engine.
    WriteOutOfRange(usize),
    /// A needed symbol resolved nowhere in scope (strong or weak).
    MissingSymbol(usize),
    /// Checked arithmetic overflowed (a lying addend/base).
    Overflow(usize),
    /// A TpRel reloc against a module absent from the TLS layout.
    Tls(usize),
    /// Malformed link input (rela/symtab/strtab/hash range lies outside
    /// the module's mapped views).
    Input,
}

/// Scope ceiling for **one load set** — a program plus the shared objects it
/// pulls in, not the image table as a whole.
///
/// This used to say it mirrored the image service's total catalogue ceiling,
/// which was never true and drifted anyway; that catalogue is now
/// allocator-backed. The bound a load set must actually respect is its own
/// link, and every link is already held far below this by
/// `MAX_LINK_MODULES <= MAX_MODULE_RECORDS` (16) in `link.rs`. So this is
/// deliberately left at 32 while the image budget grows: it is a per-process
/// ceiling that scales with dependency depth, not with how many images the
/// machine ships, and raising it would change nothing observable.
pub const MAX_MODULES: usize = 32;

// ELF64 symbol facts the resolver reads (numbers per the gABI).
const SHN_UNDEF: u16 = 0;
const SHN_ABS: u16 = 0xfff1;
const STB_GLOBAL: u8 = 1;
const STB_WEAK: u8 = 2;
const RELA_ENTRY_SIZE: usize = 24;
const SYM_ENTRY_SIZE: u64 = 24;

/// One `Elf64_Rela`, read as frozen little-endian bytes (zerocopy — no
/// offset math against hostile input).
#[derive(FromBytes, zerocopy::KnownLayout, zerocopy::Immutable)]
#[repr(C)]
struct Rela64 {
    r_offset: LeU64,
    r_info: LeU64,
    r_addend: LeI64,
}

/// One `Elf64_Sym`, likewise.
#[derive(FromBytes, zerocopy::KnownLayout, zerocopy::Immutable)]
#[repr(C)]
struct Sym64 {
    st_name: LeU32,
    st_info: u8,
    st_other: u8,
    st_shndx: LeU16,
    st_value: LeU64,
    st_size: LeU64,
}

// The wire shapes are ABI: a drifted layout would misread every image.
const _: () = {
    use core::mem::size_of;
    assert!(size_of::<Rela64>() == RELA_ENTRY_SIZE);
    assert!(size_of::<Sym64>() == SYM_ENTRY_SIZE as usize);
};

/// One loaded module's *read* half: the chosen base, the descriptor it
/// was acquired with, and read views of its mapped segments.  `ro`
/// parallels `descriptor.segs`; only [`seg::KIND_TEXT`] /
/// [`seg::KIND_RODATA`] entries are ever consulted (link inputs are
/// rodata-resident by the §7 layout — an input pointing into the data
/// segment refuses [`ApplyError::Input`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModuleView<'a> {
    /// The absolute load base this task chose for the module.
    pub base: u64,
    /// The acquired descriptor (image-relative geometry).
    pub descriptor: &'a ModuleDescriptor,
    /// Mapped read views, parallel to `descriptor.segs`.  A view may be
    /// the exact `filesz` bytes or the page-rounded mapping; reads are
    /// bounded by the smaller of the two.
    pub ro: [Option<&'a [u8]>; MAX_SEGMENTS],
}

/// The resolution set: every module of the closed load set, in load
/// order (dependency post-order).  Lookup walks GNU hash per module in
/// this order — first strong definition wins; failing that, the first
/// weak one ([`apply_module`]'s contract).
#[derive(Clone, Copy, Debug)]
pub struct SymbolScope<'a> {
    /// The load set.  Index positions here are the module indices the
    /// TLS layout and `ModuleImage::scope_index` speak.
    pub modules: &'a [ModuleView<'a>],
}

/// The module being applied: its read half plus the one write authority
/// the engine accepts — the caller's own private data copy.
#[derive(Debug)]
pub struct ModuleImage<'a> {
    /// This module's index in the [`SymbolScope`] (and so in the
    /// [`TlsLayout`]) — the self-reference TLS relocations with symbol
    /// index 0 resolve against.
    pub scope_index: usize,
    /// The read half (same content as the scope's entry for this module).
    pub view: ModuleView<'a>,
    /// The writable private data region: one union containing every
    /// [`seg::KIND_DATA`] template at its declared offset, with bss and
    /// inter-segment gaps zeroed. Relocations are still range-checked
    /// against an individual declared data segment; this slice is the
    /// only write authority.
    pub data: &'a mut [u8],
    /// The `PT_GNU_RELRO` window, image-relative `(offset, len)` — from
    /// [`crate::check::CheckedImage::relro`]; it does not ride the wire
    /// descriptor.  `None` = nothing to seal.
    pub relro: Option<(u64, u64)>,
}

/// What [`apply_module`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Applied {
    /// Relocation writes performed (rela.dyn + RELR targets + rela.plt).
    pub applied: usize,
    /// The RELRO window to seal, **absolute** `(start, len)` (the
    /// module's chosen base applied) — the stub's `unmap + re-map RO`
    /// range.  `None` when the image declared no `PT_GNU_RELRO`.
    pub relro: Option<(u64, u64)>,
}

/// A resolved definition: which module defines the symbol and the raw
/// `st_value`/`st_shndx` (absolute address derivation differs between
/// address and TLS uses, so both facts travel).
#[derive(Clone, Copy, Debug)]
struct Resolved {
    module: usize,
    st_value: u64,
    st_shndx: u16,
}

/// What a scope lookup yielded ([`resolve_for`]): a definition, or the
/// gABI's weak-undefined outcome — a **weak** reference that stayed
/// undefined across the whole scope resolves to absolute zero instead of
/// refusing (address semantics only; a TLS use of it still refuses —
/// there is no block for offset zero to mean).
#[derive(Clone, Copy, Debug)]
enum Resolution {
    Def(Resolved),
    WeakUndef,
}

/// The static-TLS layout of the closed load set (`res/IMAGE.md` §7):
/// per-module `tp`-relative block bases plus the total block the stub
/// allocates per thread.  Built once, before any module applies — IE/LE
/// legality is exactly "the whole set is known first".
///
/// Capacity is the **caller's** (Phase A, `res/USERLAND.md` §7): the
/// per-module tables are allocated from `Alloc` — arena-backed in the
/// stub — sized *exactly* to the load set, never to the [`MAX_MODULES`]
/// ceiling (which remains the semantic bound the build refuses past).
#[derive(Debug)]
pub struct TlsLayout<Alloc: core::alloc::Allocator> {
    /// Per-module block base, `tp`-relative (V1 positive, V2 negative);
    /// `None` = module has no TLS.
    bases: crate::store::Slots<Option<i64>, Alloc>,
    /// Per-module block size (bounds `tp_off`'s `sym_off`).
    sizes: crate::store::Slots<u64, Alloc>,
    /// Modules considered (the scope length at build).
    len: usize,
    /// Total static block size: V1 = bytes above `tp` (TCB included);
    /// V2 = bytes below `tp`.
    pub total: u64,
    /// The strictest block alignment (≥ 1); `tp` itself must satisfy it.
    pub align: u64,
}

impl<Alloc: core::alloc::Allocator + Copy> TlsLayout<Alloc> {
    /// Lay out static TLS for the load set, in scope order, with the
    /// tables carved from `alloc` (exact-size — the load set's length).
    ///
    /// V1 (aarch64 `tcb = 16`, riscv64 `tcb = 0`): blocks grow upward
    /// from `tp`; the first base is `align_up(tcb, align₁)`, each next
    /// `align_up(prev_end, alignᵢ)`; offsets are positive.
    ///
    /// V2 (x86_64): blocks grow downward; `off₁ = align_up(size₁,
    /// align₁)`, `offᵢ = align_up(offᵢ₋₁ + sizeᵢ, alignᵢ)`, block `i`
    /// based at `tp − offᵢ`; offsets are negative.
    ///
    /// All arithmetic checked — a lying `TlsDesc` refuses, never wraps;
    /// an exhausted arena refuses [`ApplyError::Input`], never panics.
    pub fn build<A: RelocArch>(
        modules: &[ModuleView<'_>],
        alloc: Alloc,
    ) -> Result<Self, ApplyError> {
        if modules.len() > MAX_MODULES {
            return Err(ApplyError::Input);
        }
        let mut layout = TlsLayout {
            bases: crate::store::Slots::new_from(modules.len(), alloc, |_| None)
                .map_err(|_| ApplyError::Input)?,
            sizes: crate::store::Slots::new_from(modules.len(), alloc, |_| 0)
                .map_err(|_| ApplyError::Input)?,
            len: modules.len(),
            total: 0,
            align: 1,
        };
        // V1: cursor = end of the last placed block (starts past the header
        // reserved at `tp`).  That header is the linker's to size, because
        // the linker resolves every TPREL offset that reads it, so it is
        // widened to hold the runtime's per-thread block
        // ([`RUNTIME_TCB_BYTES`]) — the psABI minimum is a floor, not a
        // ceiling, and a static block placed above it is unaffected.
        // V2: cursor = distance of the deepest block below `tp`; there the
        // runtime block extends the TCB *upward*, away from the blocks, so
        // it does not enter this layout at all.
        let mut cursor: u64 = match A::TLS {
            TlsRules::V1 { tcb } => tcb.max(kore_abi::image::RUNTIME_TCB_BYTES),
            TlsRules::V2 => 0,
        };
        for (i, m) in modules.iter().enumerate() {
            let size = m.descriptor.tls.memsz.get();
            if size == 0 {
                continue;
            }
            let align = normalize_align(m.descriptor.tls.align.get())?;
            let slot = layout
                .bases
                .as_mut_slice()
                .get_mut(i)
                .ok_or(ApplyError::Input)?;
            match A::TLS {
                TlsRules::V1 { .. } => {
                    let base = align_up(cursor, align).ok_or(ApplyError::Input)?;
                    *slot = Some(i64::try_from(base).map_err(|_| ApplyError::Input)?);
                    cursor = base.checked_add(size).ok_or(ApplyError::Input)?;
                }
                TlsRules::V2 => {
                    let off = cursor
                        .checked_add(size)
                        .and_then(|v| align_up(v, align))
                        .ok_or(ApplyError::Input)?;
                    let off_i = i64::try_from(off).map_err(|_| ApplyError::Input)?;
                    *slot = Some(off_i.checked_neg().ok_or(ApplyError::Input)?);
                    cursor = off;
                }
            }
            *layout
                .sizes
                .as_mut_slice()
                .get_mut(i)
                .ok_or(ApplyError::Input)? = size;
            layout.align = layout.align.max(align);
        }
        layout.total = cursor;
        Ok(layout)
    }

    /// The `tp`-relative offset of `sym_off` inside `module`'s block;
    /// `None` when the module has no block or the offset escapes it —
    /// the caller maps that onto [`ApplyError::Tls`].
    pub fn tp_off(&self, module: usize, sym_off: u64) -> Option<i64> {
        if module >= self.len {
            return None;
        }
        let base = (*self.bases.as_slice().get(module)?)?;
        let size = *self.sizes.as_slice().get(module)?;
        // A symbol offset beyond the block is a lie about the template.
        if sym_off > size {
            return None;
        }
        base.checked_add(i64::try_from(sym_off).ok()?)
    }
}

/// Eagerly relocate one module (the module contract): `.rela.dyn`, then
/// `DT_RELR`, then `.rela.plt`, total-failure semantics.  The
/// relocation index in errors counts rela entries and RELR words in
/// walk order.
pub fn apply_module<A: RelocArch, Alloc: core::alloc::Allocator + Copy>(
    module: &mut ModuleImage<'_>,
    scope: &SymbolScope<'_>,
    tls: &TlsLayout<Alloc>,
) -> Result<Applied, ApplyError> {
    let d = module.view.descriptor;
    // Defense in depth: a corrupt or foreign descriptor refuses before
    // any byte moves (the service validated content; this is the cheap
    // consumer re-check).
    if !d.validate() || d.machine.get() != A::MACHINE {
        return Err(ApplyError::Input);
    }

    // The one write authority: one private region spanning every writable
    // LOAD, while target validation below remains per declared segment.
    let data_bounds = data_bounds(d)?;
    match data_bounds {
        Some((_, len)) => {
            if u64::try_from(module.data.len()).ok() != Some(len) {
                return Err(ApplyError::Input);
            }
        }
        None => {
            if !module.data.is_empty() {
                return Err(ApplyError::Input);
            }
        }
    }

    let view = &module.view;
    let data = &mut *module.data;
    let mut idx: usize = 0;
    let mut applied: usize = 0;

    walk_rela::<A, _>(
        view,
        data,
        d,
        scope,
        tls,
        module.scope_index,
        d.dynz.rela_off.get(),
        d.dynz.rela_sz.get(),
        &mut idx,
        &mut applied,
    )?;
    walk_relr(
        view,
        data,
        d,
        d.dynz.relr_off.get(),
        d.dynz.relr_sz.get(),
        &mut idx,
        &mut applied,
    )?;
    walk_rela::<A, _>(
        view,
        data,
        d,
        scope,
        tls,
        module.scope_index,
        d.dynz.jmprel_off.get(),
        d.dynz.jmprel_sz.get(),
        &mut idx,
        &mut applied,
    )?;

    let relro = match module.relro {
        None => None,
        Some((off, len)) => Some((view.base.checked_add(off).ok_or(ApplyError::Input)?, len)),
    };

    Ok(Applied { applied, relro })
}

/// Walk one RELA table (`.rela.dyn` or `.rela.plt`).
#[allow(clippy::too_many_arguments)]
fn walk_rela<A: RelocArch, Alloc: core::alloc::Allocator + Copy>(
    view: &ModuleView<'_>,
    data: &mut [u8],
    descriptor: &ModuleDescriptor,
    scope: &SymbolScope<'_>,
    tls: &TlsLayout<Alloc>,
    self_index: usize,
    off: u64,
    sz: u64,
    idx: &mut usize,
    applied: &mut usize,
) -> Result<(), ApplyError> {
    if sz == 0 {
        return Ok(());
    }
    if sz % (RELA_ENTRY_SIZE as u64) != 0 {
        return Err(ApplyError::Input);
    }
    let table = ro_slice(view, off, sz)?;
    for entry in table.chunks_exact(RELA_ENTRY_SIZE) {
        let rela = Rela64::read_from_bytes(entry).map_err(|_| ApplyError::Input)?;
        let r_info = rela.r_info.get();
        // ELF64 r_info = sym << 32 | type; the mask/shift casts are exact.
        let r_type = (r_info & 0xffff_ffff) as u32;
        let r_sym = (r_info >> 32) as u32;
        let addend = rela.r_addend.get();

        // Defense in depth behind `check`: the classification gate runs
        // again here, so a foreign image applied directly still refuses.
        let kind = A::classify(r_type).map_err(|r| ApplyError::Reloc(r, *idx))?;

        let value: u64 = match kind {
            RelocKind::Relative => view
                .base
                .checked_add_signed(addend)
                .ok_or(ApplyError::Overflow(*idx))?,
            // GOT/PLT slots take the symbol address, addend ignored
            // (the module-doc table; identical under `-z now`).  A weak
            // reference nothing defines takes S = 0 (the gABI outcome).
            RelocKind::GlobDat | RelocKind::JumpSlot => {
                match resolve_for(scope, view, r_sym, *idx)? {
                    Resolution::Def(sym) => absolute(scope, &sym, *idx)?,
                    Resolution::WeakUndef => 0,
                }
            }
            RelocKind::Abs64 => {
                let s = match resolve_for(scope, view, r_sym, *idx)? {
                    Resolution::Def(sym) => absolute(scope, &sym, *idx)?,
                    Resolution::WeakUndef => 0,
                };
                s.checked_add_signed(addend)
                    .ok_or(ApplyError::Overflow(*idx))?
            }
            RelocKind::TpRel => {
                // Symbol 0 = a module-local TLS reference: the offset is
                // the addend against this module's own block.  A weak-
                // undefined TLS symbol refuses — a zero tp-offset would
                // alias a real block, not stand for "absent".
                let (m, sym_off) = if r_sym == 0 {
                    (self_index, 0)
                } else {
                    match resolve_for(scope, view, r_sym, *idx)? {
                        Resolution::Def(sym) => (sym.module, sym.st_value),
                        Resolution::WeakUndef => return Err(ApplyError::Tls(*idx)),
                    }
                };
                let tp = tls.tp_off(m, sym_off).ok_or(ApplyError::Tls(*idx))?;
                let v = tp.checked_add(addend).ok_or(ApplyError::Overflow(*idx))?;
                // The slot stores the two's-complement image of the
                // (possibly negative) tp offset — a bit cast, not math.
                v as u64
            }
        };

        write_word(data, descriptor, rela.r_offset.get(), value, *idx)?;
        *applied = applied.checked_add(1).ok_or(ApplyError::Input)?;
        *idx = idx.checked_add(1).ok_or(ApplyError::Input)?;
    }
    Ok(())
}

/// Decode + apply one `DT_RELR` table: an even entry relocates its own
/// address and re-anchors the cursor; an odd entry is a 63-bit bitmap
/// over the words following the anchor.  Targets take `base + in-place
/// value` (RELR keeps the addend in the slot).
fn walk_relr(
    view: &ModuleView<'_>,
    data: &mut [u8],
    descriptor: &ModuleDescriptor,
    off: u64,
    sz: u64,
    idx: &mut usize,
    applied: &mut usize,
) -> Result<(), ApplyError> {
    if sz == 0 {
        return Ok(());
    }
    if sz % 8 != 0 {
        return Err(ApplyError::Input);
    }
    let table = ro_slice(view, off, sz)?;
    let mut anchor: u64 = 0;
    for word in table.chunks_exact(8) {
        let entry = word
            .try_into()
            .map(u64::from_le_bytes)
            .map_err(|_| ApplyError::Input)?;
        if entry & 1 == 0 {
            relr_apply(view.base, data, descriptor, entry, *idx)?;
            *applied = applied.checked_add(1).ok_or(ApplyError::Input)?;
            anchor = entry.checked_add(8).ok_or(ApplyError::Overflow(*idx))?;
        } else {
            let mut bits = entry >> 1;
            let mut at = anchor;
            while bits != 0 {
                if bits & 1 == 1 {
                    relr_apply(view.base, data, descriptor, at, *idx)?;
                    *applied = applied.checked_add(1).ok_or(ApplyError::Input)?;
                }
                bits >>= 1;
                at = at.checked_add(8).ok_or(ApplyError::Overflow(*idx))?;
            }
            anchor = anchor
                .checked_add(63 * 8)
                .ok_or(ApplyError::Overflow(*idx))?;
        }
        *idx = idx.checked_add(1).ok_or(ApplyError::Input)?;
    }
    Ok(())
}

/// One RELR target: read the in-place addend, add the base, write back.
fn relr_apply(
    base: u64,
    data: &mut [u8],
    descriptor: &ModuleDescriptor,
    target: u64,
    idx: usize,
) -> Result<(), ApplyError> {
    let cur = read_word(data, descriptor, target, idx)?;
    let value = base.checked_add(cur).ok_or(ApplyError::Overflow(idx))?;
    write_word(data, descriptor, target, value, idx)
}

/// Bounding allocation geometry for all data templates. This span is only
/// allocation geometry; [`data_range`] separately requires each write to fit
/// one explicitly declared data segment.
/// The writable data union a module needs: image-relative `(start, len)`
/// spanning every [`seg::KIND_DATA`] segment (bss included), or `None`
/// for a module with no writable segment.  The length
/// [`apply_module`] demands of [`ModuleImage::data`], and the span a late
/// loader (`crate::late`) maps writable beside the module's code.
pub fn data_bounds(d: &ModuleDescriptor) -> Result<Option<(u64, u64)>, ApplyError> {
    let mut start: Option<u64> = None;
    let mut end: u64 = 0;
    for s in d.live_segs().iter().filter(|s| s.kind == seg::KIND_DATA) {
        start = Some(start.map_or(s.vaddr.get(), |at| at.min(s.vaddr.get())));
        end = end.max(
            s.vaddr
                .get()
                .checked_add(s.memsz.get())
                .ok_or(ApplyError::Input)?,
        );
    }
    match start {
        Some(start) => Ok(Some((
            start,
            end.checked_sub(start).ok_or(ApplyError::Input)?,
        ))),
        None => Ok(None),
    }
}

/// Translate an image-relative target into the writable data union — the
/// engine's one memory-safety gate. The complete word must fit one declared
/// data segment; a target in a gap, text, or rodata refuses.
fn data_range(
    data_len: usize,
    descriptor: &ModuleDescriptor,
    target: u64,
    idx: usize,
) -> Result<(usize, usize), ApplyError> {
    let target_end = target
        .checked_add(8)
        .ok_or(ApplyError::WriteOutOfRange(idx))?;
    let declared = descriptor
        .live_segs()
        .iter()
        .filter(|s| s.kind == seg::KIND_DATA)
        .any(|s| {
            s.vaddr
                .get()
                .checked_add(s.memsz.get())
                .is_some_and(|end| target >= s.vaddr.get() && target_end <= end)
        });
    if !declared {
        return Err(ApplyError::WriteOutOfRange(idx));
    }
    let (start, _) = data_bounds(descriptor)
        .map_err(|_| ApplyError::WriteOutOfRange(idx))?
        .ok_or(ApplyError::WriteOutOfRange(idx))?;
    let rel = target
        .checked_sub(start)
        .ok_or(ApplyError::WriteOutOfRange(idx))?;
    let end = rel.checked_add(8).ok_or(ApplyError::WriteOutOfRange(idx))?;
    let rel = usize::try_from(rel).map_err(|_| ApplyError::WriteOutOfRange(idx))?;
    let end = usize::try_from(end).map_err(|_| ApplyError::WriteOutOfRange(idx))?;
    if end > data_len {
        return Err(ApplyError::WriteOutOfRange(idx));
    }
    Ok((rel, end))
}

/// Little-endian u64 store through the data slice.
fn write_word(
    data: &mut [u8],
    descriptor: &ModuleDescriptor,
    target: u64,
    value: u64,
    idx: usize,
) -> Result<(), ApplyError> {
    let (a, b) = data_range(data.len(), descriptor, target, idx)?;
    data.get_mut(a..b)
        .ok_or(ApplyError::WriteOutOfRange(idx))?
        .copy_from_slice(&value.to_le_bytes());
    Ok(())
}

/// Little-endian u64 load from the data slice (the RELR addend read).
fn read_word(
    data: &[u8],
    descriptor: &ModuleDescriptor,
    target: u64,
    idx: usize,
) -> Result<u64, ApplyError> {
    let (a, b) = data_range(data.len(), descriptor, target, idx)?;
    data.get(a..b)
        .and_then(|s| s.try_into().ok())
        .map(u64::from_le_bytes)
        .ok_or(ApplyError::WriteOutOfRange(idx))
}

// ============================================================================
// Symbol resolution — GNU hash across the scope
// ============================================================================

/// Resolve the applying module's symbol `r_sym` across the scope:
/// read its name from the module's own symtab/strtab, then GNU-hash
/// walk every scope module in order — first strong definition wins,
/// else the first weak one.  Nothing found: a **weak** undefined
/// reference yields [`Resolution::WeakUndef`] (S = 0, the gABI rule);
/// a strong one refuses [`ApplyError::MissingSymbol`] — the link fails
/// closed.
fn resolve_for(
    scope: &SymbolScope<'_>,
    view: &ModuleView<'_>,
    r_sym: u32,
    idx: usize,
) -> Result<Resolution, ApplyError> {
    // Symbol 0 is the reserved undefined symbol — an address-bearing
    // relocation against it names nothing resolvable.
    if r_sym == 0 {
        return Err(ApplyError::MissingSymbol(idx));
    }
    let sym = read_sym(view, r_sym)?;
    let name = sym_name(view, sym.st_name.get())?;
    let h = gnu_hash(name);

    let mut first_weak: Option<Resolved> = None;
    for (m_idx, m) in scope.modules.iter().enumerate() {
        if let Some((resolved, binding)) = lookup_in(m, m_idx, name, h)? {
            if binding == STB_GLOBAL {
                return Ok(Resolution::Def(resolved));
            }
            if binding == STB_WEAK && first_weak.is_none() {
                first_weak = Some(resolved);
            }
        }
    }
    if let Some(w) = first_weak {
        return Ok(Resolution::Def(w));
    }
    // The weak-undef gate reads the *referencing* symbol: only a weak
    // reference that is itself undefined earns the zero resolution.
    if sym.st_info >> 4 == STB_WEAK && sym.st_shndx.get() == SHN_UNDEF {
        return Ok(Resolution::WeakUndef);
    }
    Err(ApplyError::MissingSymbol(idx))
}

/// Look one exported symbol up in one module — the `dlsym` half of
/// late binding (`res/IMAGE.md` §8).  GNU-hash walk of `view`'s own
/// tables: the first *defining* candidate wins (global before weak is
/// the table's own order); `None` when the module does not define
/// `name`; a module without `DT_GNU_HASH` or with lying tables refuses
/// [`ApplyError::Input`].  The address is absolute: `SHN_ABS` stands
/// alone, everything else rebases on `view.base`.
pub fn lookup_symbol(view: &ModuleView<'_>, name: &[u8]) -> Result<Option<u64>, ApplyError> {
    let h = gnu_hash(name);
    match lookup_in(view, 0, name, h)? {
        Some((found, _binding)) => {
            if found.st_shndx == SHN_ABS {
                return Ok(Some(found.st_value));
            }
            view.base
                .checked_add(found.st_value)
                .map(Some)
                .ok_or(ApplyError::Overflow(0))
        }
        None => Ok(None),
    }
}

/// A definition's absolute address: `SHN_ABS` values stand alone;
/// everything else rebases on the defining module's chosen base.
fn absolute(scope: &SymbolScope<'_>, sym: &Resolved, idx: usize) -> Result<u64, ApplyError> {
    if sym.st_shndx == SHN_ABS {
        return Ok(sym.st_value);
    }
    let m = scope.modules.get(sym.module).ok_or(ApplyError::Input)?;
    m.base
        .checked_add(sym.st_value)
        .ok_or(ApplyError::Overflow(idx))
}

/// GNU-hash lookup of `name` (pre-hashed `h`) in one module.  Returns
/// the first *defining* candidate (`st_shndx != SHN_UNDEF`, binding
/// global or weak) or `None`; malformed tables refuse
/// [`ApplyError::Input`].  The bloom filter is honored when present —
/// a word of zeros hides every name, exactly as on-disk.
fn lookup_in(
    m: &ModuleView<'_>,
    m_idx: usize,
    name: &[u8],
    h: u32,
) -> Result<Option<(Resolved, u8)>, ApplyError> {
    let gh_off = m.descriptor.dynz.gnu_hash_off.get();
    // Buckets + chains are mandatory (`res/IMAGE.md` §7): a module
    // without `DT_GNU_HASH` cannot participate in resolution.
    if gh_off == 0 {
        return Err(ApplyError::Input);
    }
    let hash = ro_tail(m, gh_off)?;

    let nbuckets = u32_at(hash, 0)?;
    let symoffset = u32_at(hash, 4)?;
    let bloom_size = u32_at(hash, 8)?;
    let bloom_shift = u32_at(hash, 12)?;
    // Zero buckets divides by zero; symoffset 0 would hash the reserved
    // null symbol — both mark a lying table.
    if nbuckets == 0 || symoffset == 0 {
        return Err(ApplyError::Input);
    }

    // Bloom gate (64-bit words).
    if bloom_size != 0 {
        let word_idx = u64::from(h / 64) % u64::from(bloom_size);
        let word_off = word_idx
            .checked_mul(8)
            .and_then(|o| o.checked_add(16))
            .ok_or(ApplyError::Input)?;
        let word = u64_at(hash, word_off)?;
        let bit_a = 1u64 << (h % 64);
        // A lying shift (≥ 32) must not panic the shift op; the spec's
        // shift is always < 32, so an out-of-range one just selects bit 0.
        let bit_b = 1u64 << (h.checked_shr(bloom_shift).unwrap_or(0) % 64);
        if word & bit_a == 0 || word & bit_b == 0 {
            return Ok(None);
        }
    }

    let buckets_off = u64::from(bloom_size)
        .checked_mul(8)
        .and_then(|o| o.checked_add(16))
        .ok_or(ApplyError::Input)?;
    let chains_off = u64::from(nbuckets)
        .checked_mul(4)
        .and_then(|o| o.checked_add(buckets_off))
        .ok_or(ApplyError::Input)?;

    let bucket_slot = u64::from(h % nbuckets)
        .checked_mul(4)
        .and_then(|o| o.checked_add(buckets_off))
        .ok_or(ApplyError::Input)?;
    let first = u32_at(hash, bucket_slot)?;
    if first == 0 {
        return Ok(None);
    }
    if first < symoffset {
        return Err(ApplyError::Input);
    }

    let mut sym_idx = first;
    loop {
        let chain_slot = u64::from(sym_idx.checked_sub(symoffset).ok_or(ApplyError::Input)?)
            .checked_mul(4)
            .and_then(|o| o.checked_add(chains_off))
            .ok_or(ApplyError::Input)?;
        let chain = u32_at(hash, chain_slot)?;
        // Hash match ignores the low (end-of-chain) bit.
        if chain | 1 == h | 1 {
            let sym = read_sym(m, sym_idx)?;
            if sym.st_shndx.get() != SHN_UNDEF && sym_name_matches(m, sym.st_name.get(), name)? {
                let binding = sym.st_info >> 4;
                if binding == STB_GLOBAL || binding == STB_WEAK {
                    return Ok(Some((
                        Resolved {
                            module: m_idx,
                            st_value: sym.st_value.get(),
                            st_shndx: sym.st_shndx.get(),
                        },
                        binding,
                    )));
                }
            }
        }
        if chain & 1 == 1 {
            return Ok(None);
        }
        // The chain walk is bounded: the index grows monotonically and
        // every read above refuses once it escapes the hash view.
        sym_idx = sym_idx.checked_add(1).ok_or(ApplyError::Input)?;
    }
}

/// Read symbol `idx` from the module's dynsym.
fn read_sym(m: &ModuleView<'_>, idx: u32) -> Result<Sym64, ApplyError> {
    let symtab_off = m.descriptor.dynz.symtab_off.get();
    if symtab_off == 0 {
        return Err(ApplyError::Input);
    }
    let off = u64::from(idx)
        .checked_mul(SYM_ENTRY_SIZE)
        .and_then(|o| o.checked_add(symtab_off))
        .ok_or(ApplyError::Input)?;
    let bytes = ro_slice(m, off, SYM_ENTRY_SIZE)?;
    Sym64::read_from_bytes(bytes).map_err(|_| ApplyError::Input)
}

/// The NUL-terminated name at `st_name`, bounded by `strtab_sz`.
fn sym_name<'a>(m: &ModuleView<'a>, st_name: u32) -> Result<&'a [u8], ApplyError> {
    let strtab_off = m.descriptor.dynz.strtab_off.get();
    let strtab_sz = m.descriptor.dynz.strtab_sz.get();
    let avail = strtab_sz
        .checked_sub(u64::from(st_name))
        .ok_or(ApplyError::Input)?;
    let start = strtab_off
        .checked_add(u64::from(st_name))
        .ok_or(ApplyError::Input)?;
    let tail = ro_slice(m, start, avail)?;
    let len = tail.iter().position(|&b| b == 0).ok_or(ApplyError::Input)?;
    tail.get(..len).ok_or(ApplyError::Input)
}

/// Whether the module's string at `st_name` equals `needle` (bounded;
/// a too-short or unterminated tail is a mismatch, not a read).
fn sym_name_matches(m: &ModuleView<'_>, st_name: u32, needle: &[u8]) -> Result<bool, ApplyError> {
    let strtab_sz = m.descriptor.dynz.strtab_sz.get();
    // st_name at or past the table end is a lying symbol.
    let avail = strtab_sz
        .checked_sub(u64::from(st_name))
        .ok_or(ApplyError::Input)?;
    let need = u64::try_from(needle.len())
        .ok()
        .and_then(|n| n.checked_add(1))
        .ok_or(ApplyError::Input)?;
    if avail < need {
        return Ok(false);
    }
    let start = m
        .descriptor
        .dynz
        .strtab_off
        .get()
        .checked_add(u64::from(st_name))
        .ok_or(ApplyError::Input)?;
    let s = ro_slice(m, start, need)?;
    Ok(s.get(..needle.len()) == Some(needle) && s.get(needle.len()) == Some(&0))
}

/// The GNU hash function (dj2 33-multiplier, as the format pins it).
/// The arithmetic is *defined* modulo 2³² — wrapping here is the
/// algorithm, not an unchecked overflow.
fn gnu_hash(name: &[u8]) -> u32 {
    let mut h: u32 = 5381;
    for &b in name {
        h = h.wrapping_mul(33).wrapping_add(u32::from(b));
    }
    h
}

// ============================================================================
// Read-view plumbing
// ============================================================================

/// Resolve an image-relative range to bytes inside one of the module's
/// text/rodata read views.  Reads are capped at the segment's `filesz`
/// AND the view's actual length; a range no view contains refuses
/// [`ApplyError::Input`].
fn ro_slice<'a>(m: &ModuleView<'a>, off: u64, len: u64) -> Result<&'a [u8], ApplyError> {
    for (i, s) in m.descriptor.live_segs().iter().enumerate() {
        if s.kind != seg::KIND_TEXT && s.kind != seg::KIND_RODATA {
            continue;
        }
        let Some(bytes) = m.ro.get(i).copied().flatten() else {
            continue;
        };
        let Some(rel) = off.checked_sub(s.vaddr.get()) else {
            continue;
        };
        let end = rel.checked_add(len).ok_or(ApplyError::Input)?;
        let avail = s
            .filesz
            .get()
            .min(u64::try_from(bytes.len()).map_err(|_| ApplyError::Input)?);
        if end > avail {
            continue;
        }
        let a = usize::try_from(rel).map_err(|_| ApplyError::Input)?;
        let b = usize::try_from(end).map_err(|_| ApplyError::Input)?;
        return bytes.get(a..b).ok_or(ApplyError::Input);
    }
    Err(ApplyError::Input)
}

/// Resolve an image-relative offset to the remainder of its containing
/// text/rodata view (the unsized `DT_GNU_HASH` read path).
fn ro_tail<'a>(m: &ModuleView<'a>, off: u64) -> Result<&'a [u8], ApplyError> {
    for (i, s) in m.descriptor.live_segs().iter().enumerate() {
        if s.kind != seg::KIND_TEXT && s.kind != seg::KIND_RODATA {
            continue;
        }
        let Some(bytes) = m.ro.get(i).copied().flatten() else {
            continue;
        };
        let Some(rel) = off.checked_sub(s.vaddr.get()) else {
            continue;
        };
        let avail = s
            .filesz
            .get()
            .min(u64::try_from(bytes.len()).map_err(|_| ApplyError::Input)?);
        if rel >= avail {
            continue;
        }
        let a = usize::try_from(rel).map_err(|_| ApplyError::Input)?;
        let b = usize::try_from(avail).map_err(|_| ApplyError::Input)?;
        return bytes.get(a..b).ok_or(ApplyError::Input);
    }
    Err(ApplyError::Input)
}

/// Bounded little-endian u32 read at `off` in a resolved view.
fn u32_at(bytes: &[u8], off: u64) -> Result<u32, ApplyError> {
    let a = usize::try_from(off).map_err(|_| ApplyError::Input)?;
    let b = a.checked_add(4).ok_or(ApplyError::Input)?;
    bytes
        .get(a..b)
        .and_then(|s| s.try_into().ok())
        .map(u32::from_le_bytes)
        .ok_or(ApplyError::Input)
}

/// Bounded little-endian u64 read at `off` in a resolved view.
fn u64_at(bytes: &[u8], off: u64) -> Result<u64, ApplyError> {
    let a = usize::try_from(off).map_err(|_| ApplyError::Input)?;
    let b = a.checked_add(8).ok_or(ApplyError::Input)?;
    bytes
        .get(a..b)
        .and_then(|s| s.try_into().ok())
        .map(u64::from_le_bytes)
        .ok_or(ApplyError::Input)
}

/// Normalize a TLS alignment: zero means byte-aligned; anything else
/// must be a power of two.
fn normalize_align(align: u64) -> Result<u64, ApplyError> {
    if align == 0 {
        return Ok(1);
    }
    if !align.is_power_of_two() {
        return Err(ApplyError::Input);
    }
    Ok(align)
}

/// Round up to a power-of-two alignment; `None` on wrap.
fn align_up(v: u64, align: u64) -> Option<u64> {
    let mask = align.checked_sub(1)?;
    v.checked_add(mask).map(|x| x & !mask)
}
