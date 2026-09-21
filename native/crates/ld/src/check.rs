//! The `Register` gate + carve planner (`res/IMAGE.md` §2, §3): what the
//! image service runs before an image may become leases.
//!
//! ## Contract
//!
//! `check_image::<A: RelocArch>(bytes) -> Result<CheckedImage, CheckError>`:
//!
//! 1. [`crate::parse`] the image for `A::MACHINE` under whichever
//!    addressing convention its header declares (aligned, bounded,
//!    in-bounds — every [`ParseError`] refuses here).
//! 2. **Flag gate**: `DT_FLAGS`/`DT_FLAGS_1` must carry `BIND_NOW`;
//!    `DF_TEXTREL` (or `DT_TEXTREL`) refuses.  Relative images only —
//!    an absolute one has no `.dynamic` to ask.
//! 3. **Reloc gate**: walk `.rela.dyn` + `.rela.plt` (+ `DT_RELR`
//!    presence); every entry must classify via `A::classify` — one
//!    reject-list or unknown type refuses the whole image.  Relative
//!    images only, for the same reason
//!    ([`CheckError::Reloc`]).  A TpRel reloc in an image with no
//!    `PT_TLS` refuses.
//! 4. **Carve plan**: retain every bounded `PT_LOAD` plus the optional
//!    `PT_TLS`, classified as text R|X, rodata R, data template R→private
//!    copy, or TLS init. Kinds may repeat (full RELRO commonly emits two
//!    adjacent writable LOADs); a segment whose flags fit no class (e.g.
//!    W|X) refuses ([`CheckError::Segment`]).
//! 5. Emit the frozen [`ModuleDescriptor`] (image-relative,
//!    `validate()`-clean by construction) + the carve plan
//!    ([`CheckedImage::carve`]): the page-aligned `(file_off, len,
//!    class)` windows the service turns into per-segment extents in
//!    place.
//!
//! The descriptor is the *promise* the stub later links against, so
//! `check` is the single place where a hostile image is caught — the
//! stub re-checks only cheap invariants (defense in depth), never
//! re-derives trust.

use elf::abi::{DF_1_NOW, DF_BIND_NOW, DF_TEXTREL, DT_NEEDED, DT_NULL};
use kore_abi::image::{IMAGE_ABI, ImageName, MAX_DEPENDENCIES, ModuleDescriptor, kind, seg};

use crate::parse::{
    Addressing, DynRange, DynStart, LoadSegment, PAGE_SIZE, ParseError, ParsedImage, parse_any,
};
use crate::reloc::{RelocArch, RelocKind, RelocRefusal};

/// One carve window: a page-aligned byte range of the registered image
/// that becomes one per-segment extent (in place, no copy).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CarveSeg {
    /// Byte offset into the registered image (page-aligned).
    pub file_off: u64,
    /// Window length (page-multiple; the lease's backing size).
    pub len: u64,
    /// The [`kore_abi::image::seg`] kind this window backs.
    pub kind: u8,
}

/// The `Register` gate's yield: the frozen descriptor + the carve plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckedImage {
    /// The image-relative promise (reply payload / `.kore.ld` body).
    pub descriptor: ModuleDescriptor,
    /// The in-place carve windows, one per live descriptor segment (same
    /// order as `descriptor.segs`).
    pub carve: [Option<CarveSeg>; kore_abi::image::MAX_SEGMENTS],
    /// The `PT_GNU_RELRO` window, image-relative `(offset, len)`.  Not
    /// on the wire (`DynDesc` is frozen); it rides the checked view so
    /// the apply side can report the seal range ([`crate::apply::Applied`]).
    pub relro: Option<(u64, u64)>,
    /// Direct `DT_NEEDED` names copied from the validated dynamic/string
    /// tables. ImageService expands and freezes their transitive generation
    /// closure at privileged installation.
    pub needed: [Option<ImageName>; MAX_DEPENDENCIES],
}

/// Why `Register` refused an image.  Maps onto
/// [`ImageRefusal::BadImage`](kore_abi::image::ImageRefusal) at the service
/// boundary; the variant is the diagnostic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckError {
    /// The parse layer refused.
    Parse(ParseError),
    /// `BIND_NOW` missing, or text relocations declared.
    Flags,
    /// A relocation refused classification (reject-list or unknown).
    Reloc(RelocRefusal),
    /// A TLS reloc with no `PT_TLS`, or TLS geometry out of bounds.
    Tls,
    /// A load segment fits no carve class (e.g. `W|X`), or classes
    /// repeat where they must not.
    Segment,
    /// RELRO is not an exact page range inside the private data union.
    Relro,
    /// The dynamic section promises a range the segments don't contain.
    Range,
}

/// The `Register` gate (contract steps 1–5): parse → flag gate → reloc
/// gate → carve plan → descriptor.  One pass, total refusal — a hostile
/// image never yields a partial [`CheckedImage`].
///
/// ## Both conventions come through here
///
/// The gate used to be `ET_DYN`-only, which left every fixed-address
/// image — the linker stub and all eight founded rows — reachable solely
/// through `parse_static_image` and a founder-written `.kore.ld`. So
/// there was no way for a `STATIC_EXECUTABLE` to acquire a descriptor
/// *and* the carve windows that back its segment leases, and the
/// degenerate static case of `res/IMAGE.md` §4 could not be assembled at
/// all.
///
/// Admitting `ET_EXEC` relaxes no gate. The flag and relocation gates
/// ask questions about dynamic linking, and are **skipped** for an
/// absolute image because there is nothing to ask them of: no
/// `.dynamic`, so no `BIND_NOW` to demand (demanding it would refuse
/// every static image for lacking a section it is defined by not
/// having), and no relocation table to classify. Everything that
/// survives — the carve classes, the W^X refusal, page congruence, the
/// entry-inside-a-segment proof, the segment-count bound — applies to
/// both and is applied to both.
pub fn check_image<A: RelocArch>(bytes: &[u8]) -> Result<CheckedImage, CheckError> {
    let (img, addressing) = parse_any(bytes, A::MACHINE).map_err(CheckError::Parse)?;

    if addressing == Addressing::Relative {
        // -- Flag gate -----------------------------------------------------
        // Eager binding is mandatory (`-z now`): either flags word may
        // carry it.  Text relocations refuse in both spellings (tag and
        // flag) — text is an immutable shared lease, there is nothing to
        // relocate.
        let bind_now = img.dynamic.flags & DF_BIND_NOW as u64 != 0
            || img.dynamic.flags1 & DF_1_NOW as u64 != 0;
        if !bind_now {
            return Err(CheckError::Flags);
        }
        if img.dynamic.textrel || img.dynamic.flags & DF_TEXTREL as u64 != 0 {
            return Err(CheckError::Flags);
        }

        // -- Reloc gate ----------------------------------------------------
        reloc_gate::<A>(bytes, &img)?;
    }

    // -- Carve plan + descriptor -------------------------------------------
    let needed = needed_names(bytes, &img)?;
    carve(&img, addressing, needed)
}

/// Copy the already parse-validated `DT_NEEDED` names into a bounded inert
/// table. Offsets are rechecked against the resolved string table so this
/// layer remains independently fail-closed.
fn needed_names(
    bytes: &[u8],
    img: &ParsedImage<'_>,
) -> Result<[Option<ImageName>; MAX_DEPENDENCIES], CheckError> {
    let mut out = [None; MAX_DEPENDENCIES];
    if img.dynamic.needed_count == 0 {
        return Ok(out);
    }
    let dynamic = table_bytes(bytes, img.dynamic.dynamic)?;
    let strtab = table_bytes(bytes, img.dynamic.strtab.ok_or(CheckError::Range)?)?;
    let mut count = 0usize;
    for entry in dynamic.chunks_exact(16) {
        let tag = i64::from_le_bytes(entry[..8].try_into().map_err(|_| CheckError::Range)?);
        if tag == DT_NULL {
            break;
        }
        if tag != DT_NEEDED {
            continue;
        }
        let off = usize::try_from(u64::from_le_bytes(
            entry[8..].try_into().map_err(|_| CheckError::Range)?,
        ))
        .map_err(|_| CheckError::Range)?;
        let tail = strtab.get(off..).ok_or(CheckError::Range)?;
        let len = tail
            .iter()
            .position(|byte| *byte == 0)
            .ok_or(CheckError::Range)?;
        let name = core::str::from_utf8(&tail[..len])
            .ok()
            .and_then(ImageName::new)
            .ok_or(CheckError::Range)?;
        let slot = out.get_mut(count).ok_or(CheckError::Range)?;
        *slot = Some(name);
        count += 1;
    }
    if count != usize::from(img.dynamic.needed_count) {
        return Err(CheckError::Range);
    }
    Ok(out)
}

/// Walk `.rela.dyn` + `.rela.plt`: every `r_type` must classify under
/// `A`, and a TLS relocation without a `PT_TLS` template refuses (there
/// is no block for its offset to land in).  `DT_RELR` needs no walk
/// here — its entries are all base-relative by construction; its shape
/// (word-multiple size) was the parse layer's gate.
fn reloc_gate<A: RelocArch>(bytes: &[u8], img: &ParsedImage<'_>) -> Result<(), CheckError> {
    for range in [img.dynamic.rela, img.dynamic.jmprel].into_iter().flatten() {
        let table = table_bytes(bytes, range)?;
        for entry in table.chunks_exact(24) {
            // r_info = sym << 32 | type (ELF64); the mask cast is exact.
            let r_info = entry
                .get(8..16)
                .and_then(|s| s.try_into().ok())
                .map(u64::from_le_bytes)
                .ok_or(CheckError::Range)?;
            let r_type = (r_info & 0xffff_ffff) as u32;
            let kind = A::classify(r_type).map_err(CheckError::Reloc)?;
            if kind == RelocKind::TpRel && img.tls.is_none() {
                return Err(CheckError::Tls);
            }
        }
    }
    Ok(())
}

/// Re-slice a parse-resolved table range (the parse layer proved it in
/// bounds; the re-check keeps this layer honest on its own).
fn table_bytes(bytes: &[u8], range: DynRange) -> Result<&[u8], CheckError> {
    let start = usize::try_from(range.file_off).map_err(|_| CheckError::Range)?;
    let len = usize::try_from(range.size).map_err(|_| CheckError::Range)?;
    let end = start.checked_add(len).ok_or(CheckError::Range)?;
    bytes.get(start..end).ok_or(CheckError::Range)
}

/// Carve class of one load segment (contract step 4).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    Text,
    Rodata,
    Data,
}

fn class_of(s: &LoadSegment<'_>) -> Result<Class, CheckError> {
    match (s.flags.write, s.flags.exec) {
        // W|X refuses outright — no segment may be both executable and
        // writable, independent of how many LOADs share a class.
        (true, true) => Err(CheckError::Segment),
        (false, true) => Ok(Class::Text),
        (true, false) => Ok(Class::Data),
        (false, false) if s.flags.read => Ok(Class::Rodata),
        // A no-access segment carries nothing mappable.
        (false, false) => Err(CheckError::Segment),
    }
}

/// Fold the load segments into bounded `res/IMAGE.md` §2 entries and
/// emit the descriptor + carve windows, in canonical kind order
/// (text, rodata, data, TLS — `descriptor.segs` and `carve` stay
/// parallel).
fn carve(
    img: &ParsedImage<'_>,
    addressing: Addressing,
    needed: [Option<ImageName>; MAX_DEPENDENCIES],
) -> Result<CheckedImage, CheckError> {
    // Validate the complete load set before emitting any descriptor.  Kinds
    // deliberately may repeat: lld's full-RELRO layout has one writable LOAD
    // for RELRO and another for ordinary data/bss.
    for s in img.segments.iter() {
        let _ = class_of(s)?;
    }

    // **What the descriptor's addresses are relative to.**  For a PIE
    // that is the lowest load vaddr — everything is stated as an offset
    // from it, because per-task ASLR makes an absolute meaningless.  For
    // a fixed-address image it is nothing: its link address is its
    // runtime address, so subtracting a base would turn the one number
    // that is already final into an offset from a base no consumer will
    // ever add back.
    let base = match addressing {
        // The engine maps lease windows, and a lease is page-granular.  The
        // load base is therefore the lowest LOAD rounded outward, not the raw
        // `p_vaddr`: a legal ELF LOAD may begin part-way through its first
        // page (`p_offset % page == p_vaddr % page`).
        Addressing::Relative => align_down(
            img.segments
                .iter()
                .map(|s| s.vaddr)
                .min()
                .ok_or(CheckError::Segment)?,
            PAGE_SIZE,
        ),
        Addressing::Absolute => 0,
    };

    let mut d = ModuleDescriptor::empty();
    d.abi.set(IMAGE_ABI);
    d.machine.set(img.machine);
    d.needed_count.set(img.dynamic.needed_count);

    // Entry decides the image kind; an executable's entry must land in a
    // load segment (a promise outside the image refuses).
    if img.entry != 0 {
        let in_a_segment = img.segments.iter().any(|s| {
            class_of(s) == Ok(Class::Text)
                && img.entry >= s.vaddr
                && s.vaddr
                    .checked_add(s.memsz)
                    .is_some_and(|end| img.entry < end)
        });
        if !in_a_segment {
            return Err(CheckError::Range);
        }
        // Entry decides *executable vs library*; the addressing decides
        // which executable convention.  Both are recorded, because the
        // consumer reads the kind before the address and the two answers
        // are not derivable from each other.
        d.kind = match addressing {
            Addressing::Relative => kind::EXECUTABLE,
            Addressing::Absolute => kind::STATIC_EXECUTABLE,
        };
        d.entry
            .set(img.entry.checked_sub(base).ok_or(CheckError::Range)?);
    } else if addressing == Addressing::Absolute {
        // `parse_static_image` proves the entry lies inside an
        // executable load, so a zero here means it reached this gate
        // some other way.  Refused rather than filed as a library: a
        // fixed-address image with nothing to enter is not one.
        return Err(CheckError::Range);
    } else {
        d.kind = kind::LIBRARY;
    }

    // Segment table + carve windows, canonical kind order.
    let mut carve = [None; kore_abi::image::MAX_SEGMENTS];
    let mut n: usize = 0;
    for (class, sk, access) in [
        (Class::Text, seg::KIND_TEXT, seg::ACCESS_R | seg::ACCESS_X),
        (Class::Rodata, seg::KIND_RODATA, seg::ACCESS_R),
        (Class::Data, seg::KIND_DATA, seg::ACCESS_R),
    ] {
        for s in img.segments.iter() {
            if class_of(s)? != class {
                continue;
            }
            let entry = d.segs.get_mut(n).ok_or(CheckError::Segment)?;
            let raw_mem_end = s.vaddr.checked_add(s.memsz).ok_or(CheckError::Segment)?;
            // Normalize the consumer-facing mapping outward to pages.  The
            // source lease starts at `align_down(p_offset)` and is installed at
            // `align_down(p_vaddr)`; congruence, already proved by the parser,
            // keeps every byte at the address the ELF declared.  Carry the prefix
            // in `filesz` and the rounded tail in `memsz` so the engine copies and
            // maps the complete window rather than asking the kernel for an
            // unaligned leaf.
            let map_va = align_down(s.vaddr, PAGE_SIZE);
            let map_end = align_up(raw_mem_end, PAGE_SIZE).ok_or(CheckError::Segment)?;
            let file_start = align_down(s.file_off, PAGE_SIZE);
            let file_end = s
                .file_off
                .checked_add(s.filesz)
                .and_then(|end| align_up(end, PAGE_SIZE))
                .ok_or(CheckError::Segment)?;
            let prefix = s.vaddr.checked_sub(map_va).ok_or(CheckError::Segment)?;
            entry
                .vaddr
                .set(map_va.checked_sub(base).ok_or(CheckError::Segment)?);
            entry
                .memsz
                .set(map_end.checked_sub(map_va).ok_or(CheckError::Segment)?);
            entry
                .filesz
                .set(prefix.checked_add(s.filesz).ok_or(CheckError::Segment)?);
            entry.kind = sk;
            entry.access = access;
            // The in-place window rounds `p_offset..p_offset+p_filesz`
            // outward. `p_offset` itself need not be page-aligned.
            let file_span = file_end
                .checked_sub(file_start)
                .ok_or(CheckError::Segment)?;
            let map_span = map_end.checked_sub(map_va).ok_or(CheckError::Segment)?;
            *carve.get_mut(n).ok_or(CheckError::Segment)? = Some(CarveSeg {
                file_off: file_start,
                // Immutable mappings install `memsz` whole, while data only
                // reads the file prefix. Back the larger shape so the same
                // lease contract is sound for both without reading padding
                // past the module bytes.  A bss-only LOAD therefore gets a
                // zero source lease and participates in the same data union.
                len: file_span.max(map_span),
                kind: sk,
            });
            n = n.checked_add(1).ok_or(CheckError::Segment)?;
        }
    }

    // lld owes the data class no adjacency: whether the RELRO LOAD's
    // page-rounded end abuts the next RW LOAD is decided by section sizes
    // (a build-root path-length shift flipped it and made one build's
    // kore-svc-swap unbootable while its twin booted), and the descriptor
    // contract requires the data union hole-free — the engine realizes it
    // as ONE private mapping and must not hand out memory the descriptor
    // never declared.  Normalize at the one producer: grow the earlier
    // data window's zero-fill tail (`memsz` past `filesz` is `.bss` by
    // the `seg::KIND_DATA` contract) until it abuts its successor, so the
    // page the linker skipped becomes DECLARED zeroed private data rather
    // than a refusal.  Real overlaps are still refused — the parser's
    // pairwise sweep ran before the fold, and `validate()`'s disjointness
    // sweep runs after; bridging only ever closes a gap.
    {
        let mut order: [usize; kore_abi::image::MAX_SEGMENTS] = [0; kore_abi::image::MAX_SEGMENTS];
        let mut m = 0usize;
        for i in 0..n {
            if d.segs[i].kind == seg::KIND_DATA {
                order[m] = i;
                m += 1;
            }
        }
        // Appended in load order, but bridge by ascending vaddr — the
        // ELF's phdr order is a convention, not a proof.
        order[..m].sort_unstable_by_key(|&i| d.segs[i].vaddr.get());
        // Bridgeable gap ceiling.  lld's congruence rounding opens at most
        // alignment-sized holes (64 KiB covers every max-page-size the
        // supported targets use); anything wider is not a linker artifact
        // but a layout lying about its shape — left unbridged for
        // `validate()` to refuse, so a hostile ELF cannot declare data at
        // two far-apart addresses and have the bridge mint the span
        // between them as a mappable bss.
        const MAX_DATA_BRIDGE: u64 = 16 * PAGE_SIZE;
        for w in 0..m.saturating_sub(1) {
            let (a, b) = (order[w], order[w + 1]);
            let a_end = d.segs[a]
                .vaddr
                .get()
                .checked_add(d.segs[a].memsz.get())
                .ok_or(CheckError::Segment)?;
            let b_va = d.segs[b].vaddr.get();
            if a_end < b_va
                && b_va.checked_sub(a_end).ok_or(CheckError::Segment)? <= MAX_DATA_BRIDGE
            {
                let grown = b_va
                    .checked_sub(d.segs[a].vaddr.get())
                    .ok_or(CheckError::Segment)?;
                d.segs[a].memsz.set(grown);
                // The parallel carve window backs the larger of the file
                // and mapped shapes; keep it in step with the grown map.
                if let Some(Some(c)) = carve.get_mut(a) {
                    c.len = c.len.max(grown);
                }
            }
        }
    }

    if let Some(tls) = &img.tls {
        // TLS geometry sanity: the alignment must be a power of two (the
        // layout builder rounds by it) and the init image must sit at or
        // above the image base.
        if tls.align != 0 && !tls.align.is_power_of_two() {
            return Err(CheckError::Tls);
        }
        let rel = tls.vaddr.checked_sub(base).ok_or(CheckError::Tls)?;
        // Pure .tbss has no bytes to grant. Its file offset need not be page
        // congruent (LLD can place it between LOADs); carry only TLS geometry.
        // Inventing a file-backed lease here would misrepresent an empty init.
        if !tls.init.is_empty() {
            // Page congruence: the init image reaches the consumer through a
            // page-rounded lease window, and the stub recovers its in-window
            // offset as `vaddr % PAGE` (the standard ELF mapping invariant
            // `p_offset ≡ p_vaddr (mod page)`).  An incongruent nonempty PT_TLS is
            // lying about where its bytes live — refused here, at the one
            // gate hostile images pass (`crates/ld/src/link.rs` trusts it).
            if tls.file_off % PAGE_SIZE != tls.vaddr % PAGE_SIZE {
                return Err(CheckError::Tls);
            }
            let entry = d.segs.get_mut(n).ok_or(CheckError::Segment)?;
            entry.vaddr.set(rel);
            entry.memsz.set(tls.memsz);
            entry.filesz.set(tls.init.len() as u64);
            entry.kind = seg::KIND_TLS;
            entry.access = seg::ACCESS_R;
            // The TLS init image is not page-aligned in the file (it lives
            // inside the data segment's span) — the window rounds outward.
            let start = align_down(tls.file_off, PAGE_SIZE);
            let end = tls
                .file_off
                .checked_add(tls.init.len() as u64)
                .and_then(|e| align_up(e, PAGE_SIZE))
                .ok_or(CheckError::Tls)?;
            *carve.get_mut(n).ok_or(CheckError::Segment)? = Some(CarveSeg {
                file_off: start,
                len: end.checked_sub(start).ok_or(CheckError::Tls)?,
                kind: seg::KIND_TLS,
            });
            n = n.checked_add(1).ok_or(CheckError::Segment)?;
        }

        d.tls.memsz.set(tls.memsz);
        d.tls.align.set(tls.align);
        d.tls.init_sz.set(tls.init.len() as u64);
    }
    d.seg_count = n as u8;

    // Dynamic geometry, image-relative (`DynDesc`'s space).
    let dy = &img.dynamic;
    set_range(
        &mut d.dynz.dynamic_off,
        &mut d.dynz.dynamic_sz,
        Some(dy.dynamic),
        base,
    )?;
    set_range(&mut d.dynz.rela_off, &mut d.dynz.rela_sz, dy.rela, base)?;
    set_range(&mut d.dynz.relr_off, &mut d.dynz.relr_sz, dy.relr, base)?;
    set_range(
        &mut d.dynz.jmprel_off,
        &mut d.dynz.jmprel_sz,
        dy.jmprel,
        base,
    )?;
    set_range(
        &mut d.dynz.strtab_off,
        &mut d.dynz.strtab_sz,
        dy.strtab,
        base,
    )?;
    set_range(
        &mut d.dynz.init_array_off,
        &mut d.dynz.init_array_sz,
        dy.init_array,
        base,
    )?;
    set_range(
        &mut d.dynz.fini_array_off,
        &mut d.dynz.fini_array_sz,
        dy.fini_array,
        base,
    )?;
    set_start(&mut d.dynz.symtab_off, dy.symtab, base)?;
    set_start(&mut d.dynz.gnu_hash_off, dy.gnu_hash, base)?;
    d.dynz.flags.set(dy.flags);

    // RELRO, image-relative — on the wire too (the pre-merge ABI
    // amendment: the stub learns its seal window from the Acquire
    // descriptor, `DynDesc::relro_*`), and on the checked view for the
    // service side.
    let relro = match img.relro {
        None => None,
        Some((vaddr, len)) => Some((vaddr.checked_sub(base).ok_or(CheckError::Range)?, len)),
    };
    if let Some((off, sz)) = relro {
        d.dynz.relro_off.set(off);
        d.dynz.relro_sz.set(sz);
    }

    // The contract's own promise is also enforced in release builds: this
    // includes data-union contiguity and pairwise mapping disjointness.
    if !d.validate() {
        return Err(CheckError::Segment);
    }

    // Shared startup/late admission: sealing may remove write permission only
    // from complete pages in this module's validated private data union.
    if let Some((off, len)) = relro {
        let (start, size) = crate::apply::data_bounds(&d)
            .map_err(|_| CheckError::Relro)?
            .ok_or(CheckError::Relro)?;
        let end = off.checked_add(len).ok_or(CheckError::Relro)?;
        let data_end = start.checked_add(size).ok_or(CheckError::Relro)?;
        if len == 0 || off % PAGE_SIZE != 0 || len % PAGE_SIZE != 0 || off < start || end > data_end
        {
            return Err(CheckError::Relro);
        }
    }

    Ok(CheckedImage {
        descriptor: d,
        carve,
        relro,
        needed,
    })
}

/// Stamp an image-relative `(off, sz)` pair from a resolved range
/// (`(0, 0)` when absent — `DynDesc`'s vacant encoding).
fn set_range(
    off: &mut zerocopy::byteorder::little_endian::U64,
    sz: &mut zerocopy::byteorder::little_endian::U64,
    range: Option<DynRange>,
    base: u64,
) -> Result<(), CheckError> {
    if let Some(r) = range {
        off.set(r.vaddr.checked_sub(base).ok_or(CheckError::Range)?);
        sz.set(r.size);
    }
    Ok(())
}

/// Stamp an image-relative unsized offset (zero when absent).
fn set_start(
    off: &mut zerocopy::byteorder::little_endian::U64,
    start: Option<DynStart>,
    base: u64,
) -> Result<(), CheckError> {
    if let Some(s) = start {
        off.set(s.vaddr.checked_sub(base).ok_or(CheckError::Range)?);
    }
    Ok(())
}

/// Round up to an alignment (power of two); `None` on wrap.
fn align_up(v: u64, align: u64) -> Option<u64> {
    let mask = align.checked_sub(1)?;
    v.checked_add(mask).map(|x| x & !mask)
}

/// Round down to an alignment (power of two, non-zero).
fn align_down(v: u64, align: u64) -> u64 {
    v & !(align.saturating_sub(1))
}
