//! `check_image` gate suite: happy paths on all three ISAs, then the
//! adversarial corpus — every named refusal must be the *specific*
//! typed error, and no input (including every truncation length) may
//! panic.

mod common;

use common::*;
use kore_abi::image::{IMAGE_ABI, ImageName, kind, seg};
use kore_ld::check::{CheckError, check_image};
use kore_ld::parse::{ParseError, parse_static_image};
use kore_ld::reloc::{AArch64, RelocRefusal, Riscv64, X86_64};

// ---------------------------------------------------------------------------
// Happy paths
// ---------------------------------------------------------------------------

fn happy_fixture(arch: &ArchRelocs) -> Fixture {
    let mut f = Fixture::new(arch.machine);
    f.entry = TEXT_VADDR;
    f.syms = vec![Sym::global("alpha", TEXT_VADDR + 0x10)];
    f.relas = vec![Rela {
        offset: DATA_VADDR + 0x100,
        rtype: arch.relative,
        sym: 0,
        addend: 0x40,
    }];
    f.plt_relas = vec![Rela {
        offset: DATA_VADDR + 0x108,
        rtype: arch.jump_slot,
        sym: 1,
        addend: 0,
    }];
    f.needed = vec!["libkore".into(), "libm".into()];
    f.relro = Some((DATA_VADDR, 0x200));
    f.bss = 0x800;
    f
}

fn assert_happy(arch: &ArchRelocs, checked: &kore_ld::check::CheckedImage) {
    let d = &checked.descriptor;
    assert!(d.validate());
    assert_eq!(d.abi.get(), IMAGE_ABI);
    assert_eq!(d.kind, kind::EXECUTABLE);
    assert_eq!(d.machine.get(), arch.machine);
    assert_eq!(d.needed_count.get(), 2);
    assert_eq!(
        checked.needed[0].as_ref().and_then(ImageName::as_str),
        Some("libkore")
    );
    assert_eq!(
        checked.needed[1].as_ref().and_then(ImageName::as_str),
        Some("libm")
    );
    assert!(checked.needed[2..].iter().all(Option::is_none));
    assert_eq!(d.entry.get(), TEXT_VADDR);

    // Canonical kind order: text, rodata, data.
    assert_eq!(d.seg_count, 3);
    assert_eq!(d.segs[0].kind, seg::KIND_TEXT);
    assert_eq!(d.segs[0].vaddr.get(), TEXT_VADDR);
    assert_eq!(d.segs[0].access, seg::ACCESS_R | seg::ACCESS_X);
    assert_eq!(d.segs[1].kind, seg::KIND_RODATA);
    assert_eq!(d.segs[1].vaddr.get(), RO_VADDR);
    assert_eq!(d.segs[1].access, seg::ACCESS_R);
    assert_eq!(d.segs[2].kind, seg::KIND_DATA);
    assert_eq!(d.segs[2].vaddr.get(), DATA_VADDR);
    assert_eq!(d.segs[2].memsz.get(), 2 * PAGE);
    assert_eq!(d.segs[2].filesz.get(), PAGE);
    assert_eq!(d.segs[2].access, seg::ACCESS_R);

    // Carve windows: parallel to segs, page-aligned, and large enough for
    // both the file prefix and the complete mapped shape.
    let text = checked.carve[0].unwrap();
    assert_eq!(
        (text.file_off, text.len, text.kind),
        (TEXT_VADDR, PAGE, seg::KIND_TEXT)
    );
    let ro = checked.carve[1].unwrap();
    assert_eq!((ro.file_off, ro.len, ro.kind), (0, PAGE, seg::KIND_RODATA));
    let data = checked.carve[2].unwrap();
    assert_eq!(
        (data.file_off, data.len, data.kind),
        (DATA_VADDR, 2 * PAGE, seg::KIND_DATA)
    );
    assert!(checked.carve[3].is_none());
    for (segment, carve) in d.live_segs().iter().zip(&checked.carve) {
        assert_eq!(
            segment.lease_len(),
            carve.map(|window| window.len),
            "the frozen descriptor derives the exact granted lease span"
        );
    }

    // Dynamic geometry, image-relative.
    assert_eq!(d.dynz.dynamic_off.get(), DYNAMIC_VADDR);
    assert_eq!(d.dynz.rela_off.get(), RELA_OFF);
    assert_eq!(d.dynz.rela_sz.get(), 24);
    assert_eq!(d.dynz.jmprel_off.get(), PLT_RELA_OFF);
    assert_eq!(d.dynz.jmprel_sz.get(), 24);
    assert_eq!(d.dynz.symtab_off.get(), DYNSYM_OFF);
    assert_eq!(d.dynz.strtab_off.get(), DYNSTR_OFF);
    assert_eq!(d.dynz.gnu_hash_off.get(), GNU_HASH_OFF);
    assert_eq!(d.dynz.flags.get(), DF_BIND_NOW);

    // RELRO rides the checked view, image-relative.
    assert_eq!(checked.relro, Some((DATA_VADDR, 0x200)));
}

#[test]
fn happy_check_x86_64() {
    let img = happy_fixture(&X86_RELOCS).build();
    let checked = check_image::<X86_64>(&img).unwrap();
    assert_happy(&X86_RELOCS, &checked);
}

#[test]
fn happy_check_aarch64() {
    let img = happy_fixture(&AARCH64_RELOCS).build();
    let checked = check_image::<AArch64>(&img).unwrap();
    assert_happy(&AARCH64_RELOCS, &checked);
}

#[test]
fn happy_check_riscv64() {
    let img = happy_fixture(&RISCV_RELOCS).build();
    let checked = check_image::<Riscv64>(&img).unwrap();
    assert_happy(&RISCV_RELOCS, &checked);
}

/// A pure-bss LOAD (`filesz = 0`, writable — what real linkers emit after
/// RELRO) remains a distinct data template. The linker maps all contiguous
/// data templates through one private union, but each keeps its own exact
/// geometry and source-lease provenance.
#[test]
fn bss_load_remains_a_distinct_data_template() {
    let mut f = happy_fixture(&X86_RELOCS);
    f.bss = 0; // a page-clean data end, so the tail is contiguous
    f.bss_tail = 0x40;
    let checked = check_image::<X86_64>(&f.build()).unwrap();
    let d = &checked.descriptor;
    assert_eq!(d.seg_count, 4);
    assert_eq!(d.segs[2].kind, seg::KIND_DATA);
    assert_eq!(d.segs[2].memsz.get(), PAGE);
    assert_eq!(d.segs[2].filesz.get(), PAGE);
    assert_eq!(d.segs[3].kind, seg::KIND_DATA);
    assert_eq!(d.segs[3].memsz.get(), PAGE);
    assert_eq!(d.segs[3].filesz.get(), 0);
    assert_eq!(checked.carve[3].unwrap().len, PAGE);
}

/// A small hole between data LOADs is lld's congruence rounding, not a
/// malformed layout: whether the RELRO template's page-rounded end abuts
/// the next writable LOAD is decided by section sizes (a build-root
/// path-length shift flipped it on a real kore-svc-swap).  The checker
/// bridges it into the earlier template's DECLARED zero-fill tail, so
/// the union stays hole-free without refusing a bootable image.
#[test]
fn bss_tail_small_gap_bridges_into_declared_zero_fill() {
    let mut f = happy_fixture(&X86_RELOCS);
    f.bss = 0;
    f.bss_tail = 0x40;
    f.bss_tail_gap_pages = 1; // one page past the template's rounded end
    let checked = check_image::<X86_64>(&f.build()).unwrap();
    let d = &checked.descriptor;
    assert_eq!(d.segs[2].kind, seg::KIND_DATA);
    // The template grew across the gap page: map doubled + the gap page,
    // file prefix untouched — the bridge is bss, never file bytes.
    assert_eq!(d.segs[2].memsz.get(), 2 * PAGE);
    assert_eq!(d.segs[2].filesz.get(), PAGE);
    // The tail keeps its own exact geometry, now abutting the template.
    assert_eq!(d.segs[3].kind, seg::KIND_DATA);
    assert_eq!(d.segs[3].memsz.get(), PAGE);
    assert_eq!(
        d.segs[2].vaddr.get() + d.segs[2].memsz.get(),
        d.segs[3].vaddr.get()
    );
    // The carve lease backs the grown mapped shape.
    assert_eq!(checked.carve[2].unwrap().len, 2 * PAGE);
}

/// A wide hole is not a linker artifact — it is a layout lying about its
/// shape (or a hostile ELF minting a giant bss between two far-apart
/// declarations).  Beyond the bridge ceiling the union stays holed and
/// the contract refuses, exactly as before the bridge existed.
#[test]
fn bss_tail_wide_gap_refuses() {
    let mut f = happy_fixture(&X86_RELOCS);
    f.bss = 0;
    f.bss_tail = 0x40;
    f.bss_tail_gap_pages = 17; // one past the 16-page bridge ceiling
    let img = f.build();
    assert_eq!(check_image::<X86_64>(&img), Err(CheckError::Segment));
}

#[test]
fn library_kind_when_no_entry() {
    let mut f = Fixture::new(EM_X86_64);
    f.entry = 0;
    let checked = check_image::<X86_64>(&f.build()).unwrap();
    assert_eq!(checked.descriptor.kind, kind::LIBRARY);
    assert_eq!(checked.descriptor.entry.get(), 0);
}

#[test]
fn tls_geometry_carried() {
    let mut f = Fixture::new(EM_X86_64);
    f.tls = Some(TlsSpec {
        memsz: 0x30,
        align: 0x10,
        filesz: 0x10,
    });
    let checked = check_image::<X86_64>(&f.build()).unwrap();
    let d = &checked.descriptor;
    assert_eq!(d.tls.memsz.get(), 0x30);
    assert_eq!(d.tls.align.get(), 0x10);
    assert_eq!(d.tls.init_sz.get(), 0x10);
    // The TLS seg entry follows data in canonical order.
    assert_eq!(d.seg_count, 4);
    assert_eq!(d.segs[3].kind, seg::KIND_TLS);
    assert_eq!(d.segs[3].vaddr.get(), TLS_VADDR);
    // The carve window rounds outward to pages around the init image.
    let tls = checked.carve[3].unwrap();
    assert_eq!(
        (tls.file_off, tls.len, tls.kind),
        (DATA_VADDR, PAGE, seg::KIND_TLS)
    );
    assert_eq!(d.segs[3].lease_len(), Some(tls.len));
}

#[test]
fn incongruent_tls_file_offset_refuses() {
    // The stub recovers the init image's in-lease offset as `vaddr %
    // page`, so `Register` pins the standard mapping invariant
    // `p_offset ≡ p_vaddr (mod page)` on PT_TLS — an incongruent header
    // is lying about where its bytes live.
    let mut f = Fixture::new(EM_X86_64);
    f.tls = Some(TlsSpec {
        memsz: 0x10,
        align: 8,
        filesz: 8,
    });
    let mut img = f.build();
    // PT_TLS is phdr 4 (after RO, TEXT, DATA, DYNAMIC); shift its file
    // offset by 8 while the vaddr stays put.
    poke64(&mut img, PH_DYNAMIC + 1, P_OFFSET, TLS_VADDR + 8);
    assert_eq!(check_image::<X86_64>(&img).unwrap_err(), CheckError::Tls);
}

#[test]
fn init_fini_arrays_carried() {
    let mut f = Fixture::new(EM_X86_64);
    f.init_fini = true;
    let d = check_image::<X86_64>(&f.build()).unwrap().descriptor;
    assert_eq!(d.dynz.init_array_off.get(), DATA_VADDR + 0x400);
    assert_eq!(d.dynz.init_array_sz.get(), 16);
    assert_eq!(d.dynz.fini_array_off.get(), DATA_VADDR + 0x440);
    assert_eq!(d.dynz.fini_array_sz.get(), 8);
}

#[test]
fn bind_now_via_flags1_accepts() {
    let mut f = Fixture::new(EM_X86_64);
    f.flags = 0;
    f.flags1 = DF_1_NOW;
    assert!(check_image::<X86_64>(&f.build()).is_ok());
}

#[test]
fn nonzero_link_base_rebases() {
    // Same image shifted to link base 0x40000: the emitted descriptor
    // must be identical to the base-0 one (image-relative throughout).
    let base0 = check_image::<X86_64>(&happy_fixture(&X86_RELOCS).build()).unwrap();
    let mut f = happy_fixture(&X86_RELOCS);
    f.vaddr_base = 0x40000;
    let shifted = check_image::<X86_64>(&f.build()).unwrap();
    assert_eq!(base0.descriptor, shifted.descriptor);
    assert_eq!(base0.relro, shifted.relro);
}

// ---------------------------------------------------------------------------
// Adversarial corpus — each case must produce its specific typed error.
// ---------------------------------------------------------------------------

#[test]
fn truncation_fuzz_never_panics() {
    // Every prefix of a valid image must refuse with a typed error —
    // never parse, never panic.  catch_unwind proves the no-panic claim.
    let img = happy_fixture(&X86_RELOCS).build();
    for len in 0..img.len() {
        let slice = &img[..len];
        let result = std::panic::catch_unwind(|| check_image::<X86_64>(slice));
        let inner = result.unwrap_or_else(|_| panic!("panicked at truncation length {len}"));
        assert!(inner.is_err(), "truncated image parsed at length {len}");
    }
}

#[test]
fn filesz_exceeds_memsz_refuses() {
    let mut img = happy_fixture(&X86_RELOCS).build();
    poke64(&mut img, PH_DATA, P_MEMSZ, 0x800); // filesz stays 0x1000
    assert_eq!(
        check_image::<X86_64>(&img).unwrap_err(),
        CheckError::Parse(ParseError::FileExceedsMem)
    );
}

#[test]
fn out_of_bounds_p_offset_refuses() {
    let mut img = happy_fixture(&X86_RELOCS).build();
    poke64(&mut img, PH_TEXT, P_OFFSET, 0x10_0000);
    assert_eq!(
        check_image::<X86_64>(&img).unwrap_err(),
        CheckError::Parse(ParseError::OutOfBounds)
    );
}

#[test]
fn overlapping_loads_refuse() {
    let mut img = happy_fixture(&X86_RELOCS).build();
    // Slide the text load onto the RO load's page (still aligned).
    poke64(&mut img, PH_TEXT, P_VADDR, RO_VADDR);
    assert_eq!(
        check_image::<X86_64>(&img).unwrap_err(),
        CheckError::Parse(ParseError::Overlap)
    );
}

#[test]
fn unaligned_vaddr_refuses() {
    let mut img = happy_fixture(&X86_RELOCS).build();
    poke64(&mut img, PH_TEXT, P_VADDR, TEXT_VADDR + 0x10);
    assert_eq!(
        check_image::<X86_64>(&img).unwrap_err(),
        CheckError::Parse(ParseError::Misaligned)
    );
}

#[test]
fn unaligned_p_offset_refuses() {
    let mut img = happy_fixture(&X86_RELOCS).build();
    poke64(&mut img, PH_TEXT, P_OFFSET, TEXT_VADDR + 0x10);
    assert_eq!(
        check_image::<X86_64>(&img).unwrap_err(),
        CheckError::Parse(ParseError::Misaligned)
    );
}

/// **`ET_EXEC` checks, as the absolute kind.**
///
/// It used to refuse with `NotPie`, which left a fixed-address image
/// with no way to obtain a descriptor *and* the carve windows backing
/// its segment leases — the two halves `res/IMAGE.md` §4's degenerate
/// static case needs at once.
///
/// The convention is what separates the two, and a link base is what
/// makes it visible: [`nonzero_link_base_rebases`] proves an `ET_DYN`
/// descriptor is *identical* at any link base, because every address is
/// stated as an offset from it. The absolute descriptor is the exact
/// opposite — the shift rides through untouched, which is the whole
/// reason a consumer reads `kind` before an address.
#[test]
fn et_exec_checks_as_the_absolute_kind() {
    const BASE: u64 = 0x40000;

    let mut f = happy_fixture(&X86_RELOCS);
    f.vaddr_base = BASE;
    f.e_type = ET_EXEC;
    let checked = check_image::<X86_64>(&f.build()).expect("a static executable checks");

    let mut g = happy_fixture(&X86_RELOCS);
    g.vaddr_base = BASE;
    let relative = check_image::<X86_64>(&g.build()).expect("the PIE control checks");

    assert_eq!(checked.descriptor.kind, kind::STATIC_EXECUTABLE);
    assert!(kind::is_absolute(checked.descriptor.kind));
    assert_eq!(relative.descriptor.kind, kind::EXECUTABLE);

    assert_eq!(
        checked.descriptor.entry.get(),
        relative.descriptor.entry.get() + BASE,
        "absolute: the entry is the address, not an offset from a base"
    );
    assert_eq!(
        checked.descriptor.segs[0].vaddr.get(),
        relative.descriptor.segs[0].vaddr.get() + BASE,
        "segments keep their link addresses"
    );

    // The carve windows are half the point — a descriptor alone cannot
    // back a segment lease, and until now a static image could obtain
    // neither.
    assert!(
        checked.carve[0].is_some(),
        "the text segment has a carve window"
    );
    // No dynamic facts, because there is no `.dynamic`: nothing to
    // relocate, nothing to walk, nothing to seal.
    assert_eq!(checked.descriptor.needed_count.get(), 0);
    assert_eq!(checked.relro, None);
}

/// An `ET_EXEC` whose entry is not inside an executable load is refused
/// by the static parser, not filed as a library — the gate the absolute
/// path keeps that the relative one expresses differently.
#[test]
fn et_exec_with_a_bad_entry_still_refuses() {
    let mut f = happy_fixture(&X86_RELOCS);
    f.e_type = ET_EXEC;
    f.entry = DATA_VADDR;
    assert_eq!(
        check_image::<X86_64>(&f.build()).unwrap_err(),
        CheckError::Parse(ParseError::Entry)
    );
}

#[test]
fn static_boot_executable_uses_the_typed_segment_parser() {
    let mut f = happy_fixture(&X86_RELOCS);
    f.e_type = ET_EXEC;
    let bytes = f.build();
    let image = parse_static_image(&bytes, EM_X86_64).unwrap();
    assert_eq!(image.entry, TEXT_VADDR);
    assert_eq!(image.segments.len(), 3);
    assert!(image.segments.iter().any(|seg| seg.flags.exec));
    assert!(
        image
            .segments
            .iter()
            .all(|seg| !(seg.flags.write && seg.flags.exec))
    );
}

#[test]
fn static_boot_executable_refuses_writable_text() {
    let mut f = happy_fixture(&X86_RELOCS);
    f.e_type = ET_EXEC;
    let mut image = f.build();
    poke32(&mut image, PH_TEXT, P_FLAGS, PF_R | PF_W | PF_X);
    assert_eq!(
        parse_static_image(&image, EM_X86_64).unwrap_err(),
        ParseError::WritableExecutable
    );
}

#[test]
fn static_boot_executable_refuses_an_entry_outside_text() {
    let mut f = happy_fixture(&X86_RELOCS);
    f.e_type = ET_EXEC;
    f.entry = DATA_VADDR;
    assert_eq!(
        parse_static_image(&f.build(), EM_X86_64).unwrap_err(),
        ParseError::Entry
    );
}

#[test]
fn wrong_machine_refuses() {
    let img = happy_fixture(&X86_RELOCS).build();
    assert_eq!(
        check_image::<AArch64>(&img).unwrap_err(),
        CheckError::Parse(ParseError::Machine)
    );
}

#[test]
fn missing_bind_now_refuses() {
    let mut f = happy_fixture(&X86_RELOCS);
    f.flags = 0;
    f.flags1 = 0;
    assert_eq!(
        check_image::<X86_64>(&f.build()).unwrap_err(),
        CheckError::Flags
    );
}

#[test]
fn textrel_flag_refuses() {
    let mut f = happy_fixture(&X86_RELOCS);
    f.flags = DF_BIND_NOW | DF_TEXTREL;
    assert_eq!(
        check_image::<X86_64>(&f.build()).unwrap_err(),
        CheckError::Flags
    );
}

#[test]
fn textrel_tag_refuses() {
    let mut f = happy_fixture(&X86_RELOCS);
    f.textrel_tag = true;
    assert_eq!(
        check_image::<X86_64>(&f.build()).unwrap_err(),
        CheckError::Flags
    );
}

/// One test per rejected relocation type per arch (the §7 reject list):
/// each must refuse as `Reloc(Rejected(t))`, in both rela tables.
#[test]
fn reject_list_relocs_refuse_per_arch() {
    fn run<A: kore_ld::reloc::RelocArch>(arch: &ArchRelocs) {
        for &t in &arch.rejected {
            // In .rela.dyn …
            let mut f = Fixture::new(arch.machine);
            f.relas = vec![Rela {
                offset: DATA_VADDR + 0x100,
                rtype: t,
                sym: 0,
                addend: 0,
            }];
            assert_eq!(
                check_image::<A>(&f.build()).unwrap_err(),
                CheckError::Reloc(RelocRefusal::Rejected(t)),
                "machine {} type {t} (.rela.dyn)",
                arch.machine,
            );
            // … and in .rela.plt.
            let mut f = Fixture::new(arch.machine);
            f.plt_relas = vec![Rela {
                offset: DATA_VADDR + 0x100,
                rtype: t,
                sym: 0,
                addend: 0,
            }];
            assert_eq!(
                check_image::<A>(&f.build()).unwrap_err(),
                CheckError::Reloc(RelocRefusal::Rejected(t)),
                "machine {} type {t} (.rela.plt)",
                arch.machine,
            );
        }
    }
    run::<X86_64>(&X86_RELOCS);
    run::<AArch64>(&AARCH64_RELOCS);
    run::<Riscv64>(&RISCV_RELOCS);
}

#[test]
fn unknown_reloc_refuses() {
    let mut f = Fixture::new(EM_X86_64);
    f.relas = vec![Rela {
        offset: DATA_VADDR + 0x100,
        rtype: 9999,
        sym: 0,
        addend: 0,
    }];
    assert_eq!(
        check_image::<X86_64>(&f.build()).unwrap_err(),
        CheckError::Reloc(RelocRefusal::Unknown(9999))
    );
}

#[test]
fn tprel_without_pt_tls_refuses() {
    for (arch, tprel) in [
        (&X86_RELOCS, X86_RELOCS.tprel),
        (&AARCH64_RELOCS, AARCH64_RELOCS.tprel),
        (&RISCV_RELOCS, RISCV_RELOCS.tprel),
    ] {
        let mut f = Fixture::new(arch.machine);
        f.relas = vec![Rela {
            offset: DATA_VADDR + 0x100,
            rtype: tprel,
            sym: 0,
            addend: 0,
        }];
        let img = f.build();
        let err = match arch.machine {
            EM_X86_64 => check_image::<X86_64>(&img).unwrap_err(),
            EM_AARCH64 => check_image::<AArch64>(&img).unwrap_err(),
            _ => check_image::<Riscv64>(&img).unwrap_err(),
        };
        assert_eq!(err, CheckError::Tls, "machine {}", arch.machine);
    }
}

#[test]
fn dynamic_tag_outside_segments_refuses() {
    // DT_STRTAB pointing where no load segment lives.
    let mut f = happy_fixture(&X86_RELOCS);
    f.strtab_vaddr_override = Some(0x8000);
    assert_eq!(
        check_image::<X86_64>(&f.build()).unwrap_err(),
        CheckError::Parse(ParseError::Dynamic)
    );
}

#[test]
fn rela_range_escaping_segment_refuses() {
    // DT_RELA so close to the RO segment's file end that the table
    // crosses it — the range resolution must refuse.
    let mut f = happy_fixture(&X86_RELOCS);
    f.rela_vaddr_override = Some(PAGE - 8);
    assert_eq!(
        check_image::<X86_64>(&f.build()).unwrap_err(),
        CheckError::Parse(ParseError::Dynamic)
    );
}

#[test]
fn wx_segment_refuses() {
    let mut img = happy_fixture(&X86_RELOCS).build();
    poke32(&mut img, PH_DATA, P_FLAGS, PF_R | PF_W | PF_X);
    assert_eq!(
        check_image::<X86_64>(&img).unwrap_err(),
        CheckError::Segment
    );
}

#[test]
fn executable_entry_in_data_refuses() {
    // Repeated data LOADs are legal, but an executable entry may still land
    // only in an executable LOAD.
    let mut img = happy_fixture(&X86_RELOCS).build();
    poke32(&mut img, PH_TEXT, P_FLAGS, PF_R | PF_W);
    assert_eq!(check_image::<X86_64>(&img).unwrap_err(), CheckError::Range);
}

#[test]
fn no_load_segments_refuses() {
    let mut img = happy_fixture(&X86_RELOCS).build();
    poke32(&mut img, PH_RO, P_TYPE, PT_NULL);
    poke32(&mut img, PH_TEXT, P_TYPE, PT_NULL);
    poke32(&mut img, PH_DATA, P_TYPE, PT_NULL);
    assert_eq!(
        check_image::<X86_64>(&img).unwrap_err(),
        CheckError::Parse(ParseError::NoSegments)
    );
}

#[test]
fn too_many_load_segments_refuses() {
    let mut f = Fixture::new(EM_X86_64);
    f.extra_loads = 6; // 3 + 6 = 9 > MAX_LOAD_SEGMENTS (8)
    assert_eq!(
        check_image::<X86_64>(&f.build()).unwrap_err(),
        CheckError::Parse(ParseError::TooManySegments)
    );
}

#[test]
fn missing_pt_dynamic_refuses() {
    let mut img = happy_fixture(&X86_RELOCS).build();
    poke32(&mut img, PH_DYNAMIC, P_TYPE, PT_NULL);
    assert_eq!(
        check_image::<X86_64>(&img).unwrap_err(),
        CheckError::Parse(ParseError::Dynamic)
    );
}

#[test]
fn entry_outside_segments_refuses() {
    let mut f = happy_fixture(&X86_RELOCS);
    f.entry = 0x9_0000; // no load segment there
    assert_eq!(
        check_image::<X86_64>(&f.build()).unwrap_err(),
        CheckError::Range
    );
}

#[test]
fn pure_tbss_needs_no_file_congruence_or_initialization_lease() {
    for machine in [EM_X86_64, EM_AARCH64, EM_RISCV] {
        let mut fixture = Fixture::new(machine);
        fixture.entry = TEXT_VADDR;
        fixture.tls = Some(TlsSpec {
            memsz: 0x101a,
            filesz: 0,
            align: 8,
        });
        let mut image = fixture.build();
        poke64(&mut image, PH_DYNAMIC + 1, P_OFFSET, TLS_VADDR + 8);
        let checked = match machine {
            EM_X86_64 => check_image::<X86_64>(&image),
            EM_AARCH64 => check_image::<AArch64>(&image),
            _ => check_image::<Riscv64>(&image),
        }
        .unwrap();
        assert!(checked.descriptor.validate());
        assert_eq!(checked.descriptor.tls.memsz.get(), 0x101a);
        assert_eq!(checked.descriptor.tls.init_sz.get(), 0);
        assert!(
            checked
                .descriptor
                .live_segs()
                .iter()
                .all(|s| s.kind != seg::KIND_TLS)
        );
        assert!(
            checked
                .carve
                .iter()
                .flatten()
                .all(|s| s.kind != seg::KIND_TLS && s.len != 0)
        );
    }
}
