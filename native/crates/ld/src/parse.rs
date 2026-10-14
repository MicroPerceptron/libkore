//! ELF64 parsing over the `elf` crate — the klink-descended layer
//! (`res/IMAGE.md` §5): typed, bounds-checked views of a semi-trusted
//! image, refusing malformed input with [`ParseError`], never panicking
//! or over-reading.
//!
//! ## Contract (what [`crate::check`] consumes)
//!
//! - `parse_image(bytes, expected_machine) -> Result<ParsedImage, ParseError>`
//!   — validate `ElfBytes` (ELF64, little-endian, machine match,
//!   **`ET_DYN` only** — `ET_EXEC` refuses: every kore image is PIE),
//!   collect `PT_LOAD` segments (bounded by [`MAX_LOAD_SEGMENTS`]),
//!   `PT_TLS`, and `PT_DYNAMIC`, each with `p_filesz <= p_memsz`, file
//!   ranges inside `bytes`, and per-segment flag extraction.
//! - `walk_dynamic(...)` — the `DynamicInfo` gather: every `DT_*` tag
//!   `res/IMAGE.md` §7 lists, with vaddr→file-offset resolution through
//!   the collected load segments (a vaddr that resolves into no segment
//!   refuses — a lying `.dynamic` cannot point the reader outside the
//!   image).
//!
//! Segment slices borrow from the input image — zero-copy throughout;
//! the only fixed buffers are the `heapless::Vec` segment tables.

use elf::ElfBytes;
use elf::abi::{
    DT_FINI_ARRAY, DT_FINI_ARRAYSZ, DT_FLAGS, DT_FLAGS_1, DT_GNU_HASH, DT_INIT_ARRAY,
    DT_INIT_ARRAYSZ, DT_JMPREL, DT_NEEDED, DT_NULL, DT_PLTREL, DT_PLTRELSZ, DT_RELA, DT_RELAENT,
    DT_RELASZ, DT_STRSZ, DT_STRTAB, DT_SYMENT, DT_SYMTAB, DT_TEXTREL, ET_DYN, ET_EXEC, PF_R, PF_W,
    PF_X, PT_DYNAMIC, PT_GNU_RELRO, PT_LOAD, PT_TLS,
};
use elf::endian::AnyEndian;
use elf::file::Class;
use heapless::Vec as HVec;

/// Load-segment ceiling (klink's bound, kept: a legitimate PIE under
/// `-z separate-code` carries 3–5).
pub const MAX_LOAD_SEGMENTS: usize = 8;

/// The carve granule (`res/IMAGE.md` §2): segment boundaries must sit on
/// 4 KiB pages or the in-place extent carve cannot honor them.
pub const PAGE_SIZE: u64 = 4096;

// Packed-relative-relocation tags (`-z pack-relative-relocs`).  The `elf`
// crate's abi module predates them; numbers are the generic ABI's
// (glibc `elf.h`): RELRSZ 35, RELR 36, RELRENT 37.
const DT_RELRSZ: i64 = 35;
const DT_RELR: i64 = 36;
const DT_RELRENT: i64 = 37;

/// ELF64 fixed record sizes — the shapes the size-carrying `DT_*` tags
/// must divide by (a table that doesn't is lying about its contents).
const DYN_ENTRY_SIZE: u64 = 16;
const RELA_ENTRY_SIZE: u64 = 24;
const SYM_ENTRY_SIZE: u64 = 24;

/// Why an image refused to parse.  One vocabulary for the whole layer —
/// [`crate::check`] maps it onto `ImageRefusal::BadImage` with the
/// variant preserved for diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParseError {
    /// Not an ELF the `elf` crate accepts (magic/header damage).
    Header,
    /// Not ELF64 little-endian.
    Class,
    /// `e_machine` differs from the arch this image was registered for.
    Machine,
    /// Not `ET_DYN` — kore images are PIE/shared-object only.
    NotPie,
    /// Not `ET_EXEC` — the bootstrap loader accepts only static executables.
    NotExecutable,
    /// Program headers missing or unreadable.
    ProgramHeaders,
    /// A segment's file range escapes the image bytes.
    OutOfBounds,
    /// `p_filesz > p_memsz`.
    FileExceedsMem,
    /// More than [`MAX_LOAD_SEGMENTS`] load segments.
    TooManySegments,
    /// No load segments at all.
    NoSegments,
    /// A `PT_LOAD` is not page-aligned (`-z separate-code` images always
    /// are; misalignment breaks the in-place carve).
    Misaligned,
    /// The dynamic section is missing, malformed, or a tag's vaddr
    /// resolves into no load segment.
    Dynamic,
    /// Overlapping load segments (a lying layout).
    Overlap,
    /// A bootstrap segment requests write and execute simultaneously.
    WritableExecutable,
    /// The executable entry does not lie in an executable load segment.
    Entry,
}

/// One `PT_LOAD`'s access flags (`p_flags`, decomposed).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegFlags {
    /// `PF_R`.
    pub read: bool,
    /// `PF_W`.
    pub write: bool,
    /// `PF_X`.
    pub exec: bool,
}

/// One validated `PT_LOAD`: geometry in both spaces (link vaddr and file
/// offset) plus the borrowed file bytes.  Every field has been bounds-
/// checked against the image; `bytes.len() == filesz`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoadSegment<'a> {
    /// Link-time virtual address (page-aligned — validated).
    pub vaddr: u64,
    /// In-memory size (`>= filesz`).
    pub memsz: u64,
    /// File-backed size.
    pub filesz: u64,
    /// File offset (page-aligned — validated; the in-place carve anchor).
    pub file_off: u64,
    /// `p_align` as declared (informational; the carve uses [`PAGE_SIZE`]).
    pub align: u64,
    /// Access flags.
    pub flags: SegFlags,
    /// The segment's file bytes, borrowed from the image.
    pub bytes: &'a [u8],
}

/// The `PT_TLS` initialization template (klink's shape, kept).  Present
/// only when `p_memsz > 0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TlsTemplate<'a> {
    /// Link-time virtual address of the init image.
    pub vaddr: u64,
    /// Per-thread block size (`p_memsz`; the `memsz − filesz` tail is tbss).
    pub memsz: u64,
    /// Required block alignment (`p_align`).
    pub align: u64,
    /// File offset of the init image.
    pub file_off: u64,
    /// The init-image bytes, borrowed from the image (`len == p_filesz`).
    pub init: &'a [u8],
}

/// A sized dynamic range, held in **both** spaces: the tag's link vaddr
/// (what the descriptor mirrors — image-relative once `check` subtracts
/// the base) and the resolved file offset (what the service-side walk
/// reads through).  Resolution proved `[vaddr, vaddr+size)` sits inside
/// one load segment's file-backed span — a lying tag never escapes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DynRange {
    /// The tag's virtual address.
    pub vaddr: u64,
    /// Resolved file offset of `vaddr`.
    pub file_off: u64,
    /// The range's byte size.
    pub size: u64,
}

/// An unsized dynamic pointer (`DT_SYMTAB`, `DT_GNU_HASH` carry no size
/// tag): the vaddr, its resolved file offset, and the bytes *available*
/// from there to the containing segment's file-backed end — the hard
/// ceiling any later indexed read must stay under.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DynStart {
    /// The tag's virtual address.
    pub vaddr: u64,
    /// Resolved file offset of `vaddr`.
    pub file_off: u64,
    /// Bytes available to the containing segment's file end.
    pub avail: u64,
}

/// The gathered `.dynamic` facts — every tag `res/IMAGE.md` §7 consumes,
/// shape-validated (paired tags both present, sizes entry-aligned,
/// `DT_SYMENT == 24`, `DT_PLTREL == DT_RELA`) and range-resolved.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DynamicInfo {
    /// `PT_DYNAMIC` itself (the stub walks `DT_NEEDED` from it).
    pub dynamic: DynRange,
    /// `DT_RELA`/`DT_RELASZ` (`.rela.dyn`); `None` when absent or empty.
    pub rela: Option<DynRange>,
    /// `DT_RELR`/`DT_RELRSZ` (packed relative).
    pub relr: Option<DynRange>,
    /// `DT_JMPREL`/`DT_PLTRELSZ` (`.rela.plt`; eager under `-z now`).
    pub jmprel: Option<DynRange>,
    /// `DT_SYMTAB` (unsized; `DT_SYMENT` validated `== 24`, not carried).
    pub symtab: Option<DynStart>,
    /// `DT_STRTAB`/`DT_STRSZ`.
    pub strtab: Option<DynRange>,
    /// `DT_GNU_HASH` (unsized).
    pub gnu_hash: Option<DynStart>,
    /// `DT_INIT_ARRAY`/`DT_INIT_ARRAYSZ`.
    pub init_array: Option<DynRange>,
    /// `DT_FINI_ARRAY`/`DT_FINI_ARRAYSZ`.
    pub fini_array: Option<DynRange>,
    /// `DT_FLAGS` (zero when absent).
    pub flags: u64,
    /// `DT_FLAGS_1` (zero when absent).
    pub flags1: u64,
    /// `DT_NEEDED` entry count (each name offset validated `< strsz`).
    pub needed_count: u16,
    /// A `DT_TEXTREL` tag was present (the flag-form rides [`Self::flags`]).
    pub textrel: bool,
}

/// The parsed, validated image view [`crate::check`] consumes: entry +
/// machine, the bounded load-segment table, the TLS template, the
/// dynamic facts, and the `PT_GNU_RELRO` window (`(vaddr, len)`) — the
/// RELRO window rides here and [`crate::check::CheckedImage`], never the
/// frozen wire descriptor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParsedImage<'a> {
    /// `e_entry` (raw link vaddr; zero = library).
    pub entry: u64,
    /// `e_machine` (equals the `expected_machine` argument — validated).
    pub machine: u16,
    /// The `PT_LOAD` table, in file order.
    pub segments: HVec<LoadSegment<'a>, MAX_LOAD_SEGMENTS>,
    /// The `PT_TLS` template (`None` when absent or zero-sized).
    pub tls: Option<TlsTemplate<'a>>,
    /// The `.dynamic` gather.
    pub dynamic: DynamicInfo,
    /// `PT_GNU_RELRO` as `(vaddr, memsz)`, link space.
    pub relro: Option<(u64, u64)>,
}

/// A validated static bootstrap executable.
///
/// Unlike [`ParsedImage`], this carries no dynamic-link metadata: the boot
/// founder accepts `ET_EXEC` images with no relocation step and maps the
/// validated load segments at their declared addresses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StaticImage<'a> {
    /// `e_entry`, proven to lie in an executable load segment.
    pub entry: u64,
    /// `e_machine`, equal to the expected machine.
    pub machine: u16,
    /// Page-separable, non-W+X load segments.
    pub segments: HVec<LoadSegment<'a>, MAX_LOAD_SEGMENTS>,
}

/// Parse the static `ET_EXEC` subset accepted by the bootstrap founder.
///
/// This is deliberately distinct from [`parse_image`]: the two conventions
/// have different gates, and neither is relaxed to accommodate the other.
/// [`parse_any`] is what picks between them on the header byte, so a caller
/// takes an image rather than a convention. The parser still faces
/// semi-trusted bytes and
/// therefore validates every range, page geometry, permission, overlap, and
/// entry-point property before the kernel allocates or maps anything.
pub fn parse_static_image(
    bytes: &[u8],
    expected_machine: u16,
) -> Result<StaticImage<'_>, ParseError> {
    let elf = ElfBytes::<AnyEndian>::minimal_parse(bytes).map_err(|_| ParseError::Header)?;
    if elf.ehdr.class != Class::ELF64 || elf.ehdr.endianness != AnyEndian::Little {
        return Err(ParseError::Class);
    }
    if elf.ehdr.e_machine != expected_machine {
        return Err(ParseError::Machine);
    }
    if elf.ehdr.e_type != ET_EXEC {
        return Err(ParseError::NotExecutable);
    }
    let Some(phdrs) = elf.segments() else {
        return Err(ParseError::ProgramHeaders);
    };

    let mut segments: HVec<LoadSegment<'_>, MAX_LOAD_SEGMENTS> = HVec::new();
    for ph in phdrs.iter() {
        if ph.p_type != PT_LOAD || ph.p_memsz == 0 {
            continue;
        }
        if ph.p_filesz > ph.p_memsz {
            return Err(ParseError::FileExceedsMem);
        }
        // A copied segment may begin within a page, but file and memory page
        // offsets must agree. The founder maps the rounded page span, so
        // incongruence would move bytes relative to their link addresses.
        if ph.p_vaddr % PAGE_SIZE != ph.p_offset % PAGE_SIZE {
            return Err(ParseError::Misaligned);
        }
        ph.p_vaddr
            .checked_add(ph.p_memsz)
            .ok_or(ParseError::OutOfBounds)?;
        let segment_bytes =
            subslice(bytes, ph.p_offset, ph.p_filesz).ok_or(ParseError::OutOfBounds)?;
        let flags = SegFlags {
            read: ph.p_flags & PF_R != 0,
            write: ph.p_flags & PF_W != 0,
            exec: ph.p_flags & PF_X != 0,
        };
        if flags.write && flags.exec {
            return Err(ParseError::WritableExecutable);
        }
        segments
            .push(LoadSegment {
                vaddr: ph.p_vaddr,
                memsz: ph.p_memsz,
                filesz: ph.p_filesz,
                file_off: ph.p_offset,
                align: ph.p_align,
                flags,
                bytes: segment_bytes,
            })
            .map_err(|_| ParseError::TooManySegments)?;
    }
    if segments.is_empty() {
        return Err(ParseError::NoSegments);
    }

    // Permissions are installed page-wise. Rounded load spans must therefore
    // be disjoint; two segments sharing a page would force a permission union.
    for (i, a) in segments.iter().enumerate() {
        let (a_start, a_end) = rounded_page_span(a)?;
        for b in segments
            .iter()
            .skip(i.checked_add(1).ok_or(ParseError::OutOfBounds)?)
        {
            let (b_start, b_end) = rounded_page_span(b)?;
            if a_start < b_end && b_start < a_end {
                return Err(ParseError::Overlap);
            }
        }
    }

    let entry = elf.ehdr.e_entry;
    let entry_is_executable = segments.iter().any(|seg| {
        seg.flags.exec
            && seg
                .vaddr
                .checked_add(seg.memsz)
                .is_some_and(|end| entry >= seg.vaddr && entry < end)
    });
    if !entry_is_executable {
        return Err(ParseError::Entry);
    }

    Ok(StaticImage {
        entry,
        machine: elf.ehdr.e_machine,
        segments,
    })
}

/// Which addressing convention an image declares — the one fact the two
/// parsers disagree about, and the only one a consumer of either has to
/// carry forward.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Addressing {
    /// `ET_DYN` — addresses are image-relative; a consumer picks a base.
    Relative,
    /// `ET_EXEC` — addresses are final.  There is no base.
    Absolute,
}

impl<'a> StaticImage<'a> {
    /// The static image in the shape [`crate::check`] consumes.
    ///
    /// Every dynamic-link fact is **absent**, not defaulted to something
    /// plausible: a fixed-address executable has no `.dynamic`, so there
    /// are no relocations to apply, no symbols to resolve, no
    /// `DT_NEEDED` to walk and no RELRO to seal.  A zeroed
    /// [`DynamicInfo`] says exactly that, and every downstream consumer
    /// is already driven off those fields rather than off the kind — so
    /// the static case falls out of the same code instead of forking it.
    pub fn as_parsed(&self) -> ParsedImage<'a> {
        ParsedImage {
            entry: self.entry,
            machine: self.machine,
            segments: self.segments.clone(),
            tls: None,
            dynamic: DynamicInfo::default(),
            relro: None,
        }
    }
}

/// Parse an image under **either** convention, reporting which one it
/// declared.
///
/// The two parsers exist because the conventions are genuinely different
/// — [`parse_image`] demands a well-formed `.dynamic` and refuses
/// `ET_EXEC`; [`parse_static_image`] demands an entry inside an
/// executable load and refuses `ET_DYN` — and this is the one place that
/// knows both. Dispatching on `e_type` here rather than at each caller
/// keeps "which conventions exist" a fact of the parse layer: a consumer
/// asks for an image and is told how to read the addresses it got back.
///
/// It is deliberately not a *third* parser. Neither gate is relaxed; the
/// header byte selects which one runs, and an image that satisfies
/// neither refuses with that parser's own error.
pub fn parse_any(
    bytes: &[u8],
    expected_machine: u16,
) -> Result<(ParsedImage<'_>, Addressing), ParseError> {
    let elf = ElfBytes::<AnyEndian>::minimal_parse(bytes).map_err(|_| ParseError::Header)?;
    match elf.ehdr.e_type {
        ET_EXEC => {
            let s = parse_static_image(bytes, expected_machine)?;
            Ok((s.as_parsed(), Addressing::Absolute))
        }
        // Not `ET_DYN` explicitly: `parse_image` owns that refusal
        // (`NotPie`), and duplicating it here would give the same
        // malformed image two different errors depending on the door.
        _ => Ok((parse_image(bytes, expected_machine)?, Addressing::Relative)),
    }
}

/// Page-rounded virtual span occupied by one static load segment.
fn rounded_page_span(seg: &LoadSegment<'_>) -> Result<(u64, u64), ParseError> {
    let start = seg.vaddr & !(PAGE_SIZE - 1);
    let raw_end = seg
        .vaddr
        .checked_add(seg.memsz)
        .ok_or(ParseError::OutOfBounds)?;
    let end = raw_end
        .checked_add(PAGE_SIZE - 1)
        .ok_or(ParseError::OutOfBounds)?
        & !(PAGE_SIZE - 1);
    Ok((start, end))
}

/// Parse and validate a semi-trusted ELF64 image (the module contract's
/// entry point): ELF64/LE/`expected_machine`/`ET_DYN` gates, bounded and
/// page-aligned load segments, no overlaps, and the full
/// [`DynamicInfo`] walk.  Refuses with a typed [`ParseError`]; never
/// panics, never reads outside `bytes`.
pub fn parse_image(bytes: &[u8], expected_machine: u16) -> Result<ParsedImage<'_>, ParseError> {
    let elf = ElfBytes::<AnyEndian>::minimal_parse(bytes).map_err(|_| ParseError::Header)?;

    if elf.ehdr.class != Class::ELF64 {
        return Err(ParseError::Class);
    }
    if elf.ehdr.endianness != AnyEndian::Little {
        return Err(ParseError::Class);
    }
    if elf.ehdr.e_machine != expected_machine {
        return Err(ParseError::Machine);
    }
    // ET_DYN only: every kore image is PIE/shared-object.  ET_EXEC is a
    // deliberate refusal, not a supported legacy path (the klink delta).
    if elf.ehdr.e_type != ET_DYN {
        return Err(ParseError::NotPie);
    }

    let Some(phdrs) = elf.segments() else {
        return Err(ParseError::ProgramHeaders);
    };

    let mut segments: HVec<LoadSegment<'_>, MAX_LOAD_SEGMENTS> = HVec::new();
    let mut tls: Option<TlsTemplate<'_>> = None;
    let mut dynamic_ph: Option<DynRange> = None;
    let mut relro: Option<(u64, u64)> = None;

    for ph in phdrs.iter() {
        match ph.p_type {
            PT_LOAD => {
                if ph.p_filesz > ph.p_memsz {
                    return Err(ParseError::FileExceedsMem);
                }
                // Zero-extent loads carry nothing; skipping them keeps
                // the carve honest (a lease of nothing is meaningless).
                if ph.p_memsz == 0 {
                    continue;
                }
                // Page alignment on BOTH addresses: the in-place carve
                // turns file windows into extents at these boundaries.
                if ph.p_vaddr % PAGE_SIZE != 0 || ph.p_offset % PAGE_SIZE != 0 {
                    return Err(ParseError::Misaligned);
                }
                // The vaddr extent must not wrap.
                ph.p_vaddr
                    .checked_add(ph.p_memsz)
                    .ok_or(ParseError::OutOfBounds)?;
                let bytes =
                    subslice(bytes, ph.p_offset, ph.p_filesz).ok_or(ParseError::OutOfBounds)?;
                segments
                    .push(LoadSegment {
                        vaddr: ph.p_vaddr,
                        memsz: ph.p_memsz,
                        filesz: ph.p_filesz,
                        file_off: ph.p_offset,
                        align: ph.p_align,
                        flags: SegFlags {
                            read: ph.p_flags & PF_R != 0,
                            write: ph.p_flags & PF_W != 0,
                            exec: ph.p_flags & PF_X != 0,
                        },
                        bytes,
                    })
                    .map_err(|_| ParseError::TooManySegments)?;
            }
            PT_TLS => {
                // At most one TLS template is legal.
                if tls.is_some() {
                    return Err(ParseError::ProgramHeaders);
                }
                if ph.p_filesz > ph.p_memsz {
                    return Err(ParseError::FileExceedsMem);
                }
                let init =
                    subslice(bytes, ph.p_offset, ph.p_filesz).ok_or(ParseError::OutOfBounds)?;
                if ph.p_memsz > 0 {
                    tls = Some(TlsTemplate {
                        vaddr: ph.p_vaddr,
                        memsz: ph.p_memsz,
                        align: ph.p_align,
                        file_off: ph.p_offset,
                        init,
                    });
                }
            }
            PT_DYNAMIC => {
                if dynamic_ph.is_some() {
                    return Err(ParseError::ProgramHeaders);
                }
                subslice(bytes, ph.p_offset, ph.p_filesz).ok_or(ParseError::OutOfBounds)?;
                dynamic_ph = Some(DynRange {
                    vaddr: ph.p_vaddr,
                    file_off: ph.p_offset,
                    size: ph.p_filesz,
                });
            }
            PT_GNU_RELRO => {
                if relro.is_some() {
                    return Err(ParseError::ProgramHeaders);
                }
                relro = Some((ph.p_vaddr, ph.p_memsz));
            }
            _ => {}
        }
    }

    if segments.is_empty() {
        return Err(ParseError::NoSegments);
    }

    // Overlap gate: load extents (memsz space) must be pairwise disjoint —
    // an overlapping layout is lying about what its bytes are, and the
    // carve would double-lease them.  ≤8 segments, so pairwise is fine.
    for (i, a) in segments.iter().enumerate() {
        // Ends were proven non-wrapping at collection.
        let a_end = a
            .vaddr
            .checked_add(a.memsz)
            .ok_or(ParseError::OutOfBounds)?;
        for b in segments
            .iter()
            .skip(i.checked_add(1).ok_or(ParseError::OutOfBounds)?)
        {
            let b_end = b
                .vaddr
                .checked_add(b.memsz)
                .ok_or(ParseError::OutOfBounds)?;
            if a.vaddr < b_end && b.vaddr < a_end {
                return Err(ParseError::Overlap);
            }
        }
    }

    // An ET_DYN image without PT_DYNAMIC has nothing to link and nothing
    // to validate `BIND_NOW` against — refused, not defaulted.
    let dynamic_ph = dynamic_ph.ok_or(ParseError::Dynamic)?;
    let dynamic = walk_dynamic(bytes, &segments, dynamic_ph)?;

    Ok(ParsedImage {
        entry: elf.ehdr.e_entry,
        machine: elf.ehdr.e_machine,
        segments,
        tls,
        dynamic,
        relro,
    })
}

/// The raw first-pass tag accumulation (options only; shape rules apply
/// after the walk so a missing pair is a *typed* refusal, not a default).
#[derive(Clone, Copy, Debug, Default)]
struct RawDyn {
    rela: Option<u64>,
    relasz: Option<u64>,
    relaent: Option<u64>,
    relr: Option<u64>,
    relrsz: Option<u64>,
    relrent: Option<u64>,
    jmprel: Option<u64>,
    pltrelsz: Option<u64>,
    pltrel: Option<u64>,
    symtab: Option<u64>,
    syment: Option<u64>,
    strtab: Option<u64>,
    strsz: Option<u64>,
    gnu_hash: Option<u64>,
    init_array: Option<u64>,
    init_arraysz: Option<u64>,
    fini_array: Option<u64>,
    fini_arraysz: Option<u64>,
    flags: Option<u64>,
    flags1: Option<u64>,
    textrel: bool,
}

/// Gather [`DynamicInfo`] from `PT_DYNAMIC`: accumulate the §7 tag set,
/// enforce the pairing/shape rules, and resolve every vaddr range
/// through the load segments (a range no segment's file-backed span
/// contains refuses [`ParseError::Dynamic`] — a lying `.dynamic` cannot
/// point the reader outside the image).
fn walk_dynamic<'a>(
    bytes: &'a [u8],
    segments: &HVec<LoadSegment<'a>, MAX_LOAD_SEGMENTS>,
    dyn_ph: DynRange,
) -> Result<DynamicInfo, ParseError> {
    let table = subslice(bytes, dyn_ph.file_off, dyn_ph.size).ok_or(ParseError::Dynamic)?;
    if dyn_ph.size % DYN_ENTRY_SIZE != 0 {
        return Err(ParseError::Dynamic);
    }

    let mut raw = RawDyn::default();
    let mut needed_raw: u64 = 0;
    let mut terminated = false;
    for entry in table.chunks_exact(DYN_ENTRY_SIZE as usize) {
        let (tag, val) = dyn_entry(entry)?;
        if tag == DT_NULL {
            terminated = true;
            break;
        }
        match tag {
            DT_RELA => raw.rela = Some(val),
            DT_RELASZ => raw.relasz = Some(val),
            DT_RELAENT => raw.relaent = Some(val),
            DT_RELR => raw.relr = Some(val),
            DT_RELRSZ => raw.relrsz = Some(val),
            DT_RELRENT => raw.relrent = Some(val),
            DT_JMPREL => raw.jmprel = Some(val),
            DT_PLTRELSZ => raw.pltrelsz = Some(val),
            DT_PLTREL => raw.pltrel = Some(val),
            DT_SYMTAB => raw.symtab = Some(val),
            DT_SYMENT => raw.syment = Some(val),
            DT_STRTAB => raw.strtab = Some(val),
            DT_STRSZ => raw.strsz = Some(val),
            DT_GNU_HASH => raw.gnu_hash = Some(val),
            DT_INIT_ARRAY => raw.init_array = Some(val),
            DT_INIT_ARRAYSZ => raw.init_arraysz = Some(val),
            DT_FINI_ARRAY => raw.fini_array = Some(val),
            DT_FINI_ARRAYSZ => raw.fini_arraysz = Some(val),
            DT_FLAGS => raw.flags = Some(val),
            DT_FLAGS_1 => raw.flags1 = Some(val),
            DT_TEXTREL => raw.textrel = true,
            DT_NEEDED => {
                needed_raw = needed_raw.checked_add(1).ok_or(ParseError::Dynamic)?;
            }
            _ => {}
        }
    }
    // A dynamic section that runs off its segment end without DT_NULL is
    // truncated or lying about its extent.
    if !terminated {
        return Err(ParseError::Dynamic);
    }

    // Shape + resolution.  Paired tags must both appear; fixed entry
    // sizes must match ELF64; every range must land inside a segment.
    let rela = sized_range(segments, raw.rela, raw.relasz, RELA_ENTRY_SIZE)?;
    if let Some(ent) = raw.relaent
        && ent != RELA_ENTRY_SIZE
    {
        return Err(ParseError::Dynamic);
    }
    let relr = sized_range(segments, raw.relr, raw.relrsz, 8)?;
    if let Some(ent) = raw.relrent
        && ent != 8
    {
        return Err(ParseError::Dynamic);
    }
    // JMPREL needs its size AND the table-kind tag; only RELA tables are
    // legal on the three ELF64 ABIs we serve (REL is an ELF32 shape).
    let jmprel = match (raw.jmprel, raw.pltrelsz, raw.pltrel) {
        (None, None, None) => None,
        (Some(vaddr), Some(sz), Some(kind)) => {
            if kind != DT_RELA as u64 {
                return Err(ParseError::Dynamic);
            }
            sized_range(segments, Some(vaddr), Some(sz), RELA_ENTRY_SIZE)?
        }
        // PLTREL with no table (or any other partial combination) lies.
        (None, None, Some(_)) => None,
        _ => return Err(ParseError::Dynamic),
    };
    let symtab = match raw.symtab {
        None => None,
        Some(vaddr) => {
            // DT_SYMENT is validated here and not carried (DynDesc's
            // documented posture): ELF64 symbols are exactly 24 bytes.
            if raw.syment != Some(SYM_ENTRY_SIZE) {
                return Err(ParseError::Dynamic);
            }
            Some(unsized_start(segments, vaddr)?)
        }
    };
    let strtab = match (raw.strtab, raw.strsz) {
        (None, None) => None,
        (Some(vaddr), Some(sz)) => {
            let file_off = resolve_range(segments, vaddr, sz)?;
            Some(DynRange {
                vaddr,
                file_off,
                size: sz,
            })
        }
        _ => return Err(ParseError::Dynamic),
    };
    let gnu_hash = match raw.gnu_hash {
        None => None,
        Some(vaddr) => {
            let start = unsized_start(segments, vaddr)?;
            // The fixed header (nbuckets/symoffset/bloom_size/shift)
            // must at least fit.
            if start.avail < 16 {
                return Err(ParseError::Dynamic);
            }
            Some(start)
        }
    };
    let init_array = sized_range(segments, raw.init_array, raw.init_arraysz, 8)?;
    let fini_array = sized_range(segments, raw.fini_array, raw.fini_arraysz, 8)?;

    // DT_NEEDED names must sit inside the string table; a second pass
    // validates each offset now that strsz is known.
    let needed_count = u16::try_from(needed_raw).map_err(|_| ParseError::Dynamic)?;
    if needed_count > 0 {
        let strsz = strtab.ok_or(ParseError::Dynamic)?.size;
        for entry in table.chunks_exact(DYN_ENTRY_SIZE as usize) {
            let (tag, val) = dyn_entry(entry)?;
            if tag == DT_NULL {
                break;
            }
            if tag == DT_NEEDED && val >= strsz {
                return Err(ParseError::Dynamic);
            }
        }
    }

    Ok(DynamicInfo {
        dynamic: dyn_ph,
        rela,
        relr,
        jmprel,
        symtab,
        strtab,
        gnu_hash,
        init_array,
        fini_array,
        flags: raw.flags.unwrap_or(0),
        flags1: raw.flags1.unwrap_or(0),
        needed_count,
        textrel: raw.textrel,
    })
}

/// One 16-byte `.dynamic` entry: `(d_tag, d_val)`, little-endian.
fn dyn_entry(entry: &[u8]) -> Result<(i64, u64), ParseError> {
    let tag = entry
        .get(0..8)
        .and_then(|s| s.try_into().ok())
        .map(i64::from_le_bytes)
        .ok_or(ParseError::Dynamic)?;
    let val = entry
        .get(8..16)
        .and_then(|s| s.try_into().ok())
        .map(u64::from_le_bytes)
        .ok_or(ParseError::Dynamic)?;
    Ok((tag, val))
}

/// Fold a `(vaddr, size)` tag pair into a resolved [`DynRange`]:
/// both-or-neither, `size % ent == 0`, and the range inside a segment.
/// An empty table (`size == 0`) folds to `None`.
fn sized_range(
    segments: &HVec<LoadSegment<'_>, MAX_LOAD_SEGMENTS>,
    vaddr: Option<u64>,
    size: Option<u64>,
    ent: u64,
) -> Result<Option<DynRange>, ParseError> {
    match (vaddr, size) {
        (None, None) => Ok(None),
        (Some(_), Some(0)) => Ok(None),
        (Some(vaddr), Some(size)) => {
            if size % ent != 0 {
                return Err(ParseError::Dynamic);
            }
            let file_off = resolve_range(segments, vaddr, size)?;
            Ok(Some(DynRange {
                vaddr,
                file_off,
                size,
            }))
        }
        _ => Err(ParseError::Dynamic),
    }
}

/// Resolve a sized vaddr range to its file offset: the whole range must
/// sit inside ONE load segment's file-backed span (link inputs never
/// straddle segments; a range that escapes every segment refuses).
fn resolve_range(
    segments: &HVec<LoadSegment<'_>, MAX_LOAD_SEGMENTS>,
    vaddr: u64,
    size: u64,
) -> Result<u64, ParseError> {
    for seg in segments.iter() {
        let Some(rel) = vaddr.checked_sub(seg.vaddr) else {
            continue;
        };
        let end = rel.checked_add(size).ok_or(ParseError::Dynamic)?;
        if end > seg.filesz {
            continue;
        }
        return seg.file_off.checked_add(rel).ok_or(ParseError::Dynamic);
    }
    Err(ParseError::Dynamic)
}

/// Resolve an unsized vaddr to `(file_off, avail)` — the bytes from it
/// to the containing segment's file-backed end (the ceiling later
/// indexed reads are bounded by).
fn unsized_start(
    segments: &HVec<LoadSegment<'_>, MAX_LOAD_SEGMENTS>,
    vaddr: u64,
) -> Result<DynStart, ParseError> {
    for seg in segments.iter() {
        let Some(rel) = vaddr.checked_sub(seg.vaddr) else {
            continue;
        };
        if rel >= seg.filesz {
            continue;
        }
        let file_off = seg.file_off.checked_add(rel).ok_or(ParseError::Dynamic)?;
        let avail = seg.filesz.checked_sub(rel).ok_or(ParseError::Dynamic)?;
        return Ok(DynStart {
            vaddr,
            file_off,
            avail,
        });
    }
    Err(ParseError::Dynamic)
}

/// Checked image subslice: `bytes[off .. off+len]`, `None` on any
/// overflow or escape (the single bounds gate every file read uses).
fn subslice(bytes: &[u8], off: u64, len: u64) -> Option<&[u8]> {
    let start = usize::try_from(off).ok()?;
    let end = start.checked_add(usize::try_from(len).ok()?)?;
    bytes.get(start..end)
}
