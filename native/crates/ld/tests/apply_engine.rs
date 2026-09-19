//! Apply-engine suite: value computation against hand-computed
//! expectations, cross-module GNU-hash resolution, RELR decode, the
//! refusal set (WriteOutOfRange / Overflow / MissingSymbol / Tls /
//! Reloc), and the psABI TLS layout numbers for all three variants.

/// A leaked caller arena — the Phase A capacity seam's test face
/// (`res/USERLAND.md` §7: the tables are arena-carved, caller-sized).
fn arena() -> &'static kore_ld::store::LinkArena<'static> {
    let buf = Box::leak(Box::new([core::mem::MaybeUninit::<u8>::uninit(); 1 << 16]));
    Box::leak(Box::new(kore_ld::store::LinkArena::new(buf)))
}

mod common;

use common::*;
use kore_abi::image::{IMAGE_ABI, MAX_SEGMENTS, ModuleDescriptor, kind, seg};
use kore_ld::apply::{
    Applied, ApplyError, ModuleImage, ModuleView, SymbolScope, TlsLayout, apply_module,
    lookup_symbol,
};
use kore_ld::check::{CheckedImage, check_image};
use kore_ld::reloc::{AArch64, RelocRefusal, Riscv64, X86_64};

// ---------------------------------------------------------------------------
// Plumbing: build ModuleViews / data copies from a checked fixture.
// The fixtures link at base 0, so image-relative offsets equal file
// offsets and the raw image doubles as the mapped segments.
// ---------------------------------------------------------------------------

fn view_of<'a>(img: &'a [u8], d: &'a ModuleDescriptor, base: u64) -> ModuleView<'a> {
    let mut ro = [None; MAX_SEGMENTS];
    for i in 0..d.seg_count as usize {
        let s = &d.segs[i];
        if s.kind == seg::KIND_TEXT || s.kind == seg::KIND_RODATA {
            let a = s.vaddr.get() as usize;
            let b = a + s.filesz.get() as usize;
            ro[i] = Some(&img[a..b]);
        }
    }
    ModuleView {
        base,
        descriptor: d,
        ro,
    }
}

/// The consumer's private copy of the data template: filesz bytes from
/// the image, zero-extended to memsz (the .bss tail).
fn data_copy(img: &[u8], d: &ModuleDescriptor) -> Vec<u8> {
    for i in 0..d.seg_count as usize {
        let s = &d.segs[i];
        if s.kind == seg::KIND_DATA {
            let a = s.vaddr.get() as usize;
            let mut v = img[a..a + s.filesz.get() as usize].to_vec();
            v.resize(s.memsz.get() as usize, 0);
            return v;
        }
    }
    Vec::new()
}

/// Raw u64 poke into a built image (post-`check` corruption for the
/// defense-in-depth cases).
fn write_raw64(img: &mut [u8], off: usize, v: u64) {
    img[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

const BASE_A: u64 = 0x7f00_0000;
const BASE_B: u64 = 0x0100_0000;
const BASE_C: u64 = 0x0300_0000;

/// One-module apply: check the fixture, map views at `base`, run the
/// engine, return (Applied, data).
fn apply_single<A: kore_ld::reloc::RelocArch>(
    img: &[u8],
    base: u64,
) -> Result<(Applied, Vec<u8>), ApplyError> {
    let checked: CheckedImage = check_image::<A>(img).expect("fixture must check");
    let d = checked.descriptor;
    let views = [view_of(img, &d, base)];
    let scope = SymbolScope { modules: &views };
    let tls = TlsLayout::build::<A>(&views, arena()).unwrap();
    let mut data = data_copy(img, &d);
    let mut module = ModuleImage {
        scope_index: 0,
        view: view_of(img, &d, base),
        data: &mut data,
        relro: checked.relro,
    };
    let applied = apply_module::<A, _>(&mut module, &scope, &tls)?;
    Ok((applied, data))
}

// ---------------------------------------------------------------------------
// Value computation (hand-computed expectations)
// ---------------------------------------------------------------------------

#[test]
fn relative_value() {
    let mut f = Fixture::new(EM_X86_64);
    f.relas = vec![Rela {
        offset: DATA_VADDR + 0x100,
        rtype: 8,
        sym: 0,
        addend: 0x123,
    }];
    f.relro = Some((DATA_VADDR, 0x200));
    let (applied, data) = apply_single::<X86_64>(&f.build(), BASE_A).unwrap();
    // B + A = 0x7f00_0000 + 0x123.
    assert_eq!(read64(&data, 0x100), BASE_A + 0x123);
    assert_eq!(applied.applied, 1);
    // RELRO reported absolute: base + image-relative window.
    assert_eq!(applied.relro, Some((BASE_A + DATA_VADDR, 0x200)));
}

#[test]
fn glob_dat_cross_module() {
    // A imports beta_fn; B defines it at 0x1020.  The GOT slot takes S
    // (no addend): BASE_B + 0x1020.
    let mut fa = Fixture::new(EM_X86_64);
    fa.syms = vec![Sym::import("beta_fn")];
    fa.relas = vec![Rela {
        offset: DATA_VADDR + 0x100,
        rtype: 6,
        sym: 1,
        addend: 0,
    }];
    let img_a = fa.build();
    let mut fb = Fixture::new(EM_X86_64);
    fb.syms = vec![Sym::global("beta_fn", TEXT_VADDR + 0x20)];
    let img_b = fb.build();

    let da = check_image::<X86_64>(&img_a).unwrap().descriptor;
    let db = check_image::<X86_64>(&img_b).unwrap().descriptor;
    let views = [view_of(&img_a, &da, BASE_A), view_of(&img_b, &db, BASE_B)];
    let scope = SymbolScope { modules: &views };
    let tls = TlsLayout::build::<X86_64>(&views, arena()).unwrap();
    let mut data = data_copy(&img_a, &da);
    let mut module = ModuleImage {
        scope_index: 0,
        view: view_of(&img_a, &da, BASE_A),
        data: &mut data,
        relro: None,
    };
    apply_module::<X86_64, _>(&mut module, &scope, &tls).unwrap();
    assert_eq!(read64(&data, 0x100), BASE_B + TEXT_VADDR + 0x20);
}

#[test]
fn jump_slot_via_plt() {
    // Same shape through .rela.plt — identical semantics under -z now.
    let mut fa = Fixture::new(EM_X86_64);
    fa.syms = vec![Sym::import("beta_fn")];
    fa.plt_relas = vec![Rela {
        offset: DATA_VADDR + 0x108,
        rtype: 7,
        sym: 1,
        addend: 0,
    }];
    let img_a = fa.build();
    let mut fb = Fixture::new(EM_X86_64);
    fb.syms = vec![Sym::global("beta_fn", TEXT_VADDR + 0x40)];
    let img_b = fb.build();

    let da = check_image::<X86_64>(&img_a).unwrap().descriptor;
    let db = check_image::<X86_64>(&img_b).unwrap().descriptor;
    let views = [view_of(&img_a, &da, BASE_A), view_of(&img_b, &db, BASE_B)];
    let scope = SymbolScope { modules: &views };
    let tls = TlsLayout::build::<X86_64>(&views, arena()).unwrap();
    let mut data = data_copy(&img_a, &da);
    let mut module = ModuleImage {
        scope_index: 0,
        view: view_of(&img_a, &da, BASE_A),
        data: &mut data,
        relro: None,
    };
    apply_module::<X86_64, _>(&mut module, &scope, &tls).unwrap();
    assert_eq!(read64(&data, 0x108), BASE_B + TEXT_VADDR + 0x40);
}

#[test]
fn abs64_value_with_addend() {
    // S + A: the module's own definition plus the addend.
    let mut f = Fixture::new(EM_X86_64);
    f.syms = vec![Sym::global("obj", DATA_VADDR + 0x300)];
    f.relas = vec![Rela {
        offset: DATA_VADDR + 0x100,
        rtype: 1,
        sym: 1,
        addend: 0x10,
    }];
    let (_, data) = apply_single::<X86_64>(&f.build(), BASE_A).unwrap();
    assert_eq!(read64(&data, 0x100), BASE_A + DATA_VADDR + 0x300 + 0x10);
}

#[test]
fn engine_is_arch_generic() {
    // The same semantics through the aarch64 and riscv64 tables.
    for (machine, relative) in [(EM_AARCH64, 1027u32), (EM_RISCV, 3u32)] {
        let mut f = Fixture::new(machine);
        f.relas = vec![Rela {
            offset: DATA_VADDR + 0x100,
            rtype: relative,
            sym: 0,
            addend: 0x55,
        }];
        let img = f.build();
        let (applied, data) = match machine {
            EM_AARCH64 => apply_single::<AArch64>(&img, BASE_A).unwrap(),
            _ => apply_single::<Riscv64>(&img, BASE_A).unwrap(),
        };
        assert_eq!(read64(&data, 0x100), BASE_A + 0x55, "machine {machine}");
        assert_eq!(applied.applied, 1);
    }
}

// ---------------------------------------------------------------------------
// Resolution order
// ---------------------------------------------------------------------------

#[test]
fn strong_preferred_over_earlier_weak() {
    // B (earlier in scope) defines `dual` weak; C defines it strong.
    // Strong wins across the whole scope, position notwithstanding.
    let mut fa = Fixture::new(EM_X86_64);
    fa.syms = vec![Sym::import("dual")];
    fa.relas = vec![Rela {
        offset: DATA_VADDR + 0x100,
        rtype: 6,
        sym: 1,
        addend: 0,
    }];
    let img_a = fa.build();
    let mut fb = Fixture::new(EM_X86_64);
    fb.syms = vec![Sym::weak("dual", TEXT_VADDR + 0x30)];
    let img_b = fb.build();
    let mut fc = Fixture::new(EM_X86_64);
    fc.syms = vec![Sym::global("dual", TEXT_VADDR + 0x40)];
    let img_c = fc.build();

    let da = check_image::<X86_64>(&img_a).unwrap().descriptor;
    let db = check_image::<X86_64>(&img_b).unwrap().descriptor;
    let dc = check_image::<X86_64>(&img_c).unwrap().descriptor;
    let views = [
        view_of(&img_a, &da, BASE_A),
        view_of(&img_b, &db, BASE_B),
        view_of(&img_c, &dc, BASE_C),
    ];
    let scope = SymbolScope { modules: &views };
    let tls = TlsLayout::build::<X86_64>(&views, arena()).unwrap();
    let mut data = data_copy(&img_a, &da);
    let mut module = ModuleImage {
        scope_index: 0,
        view: view_of(&img_a, &da, BASE_A),
        data: &mut data,
        relro: None,
    };
    apply_module::<X86_64, _>(&mut module, &scope, &tls).unwrap();
    assert_eq!(read64(&data, 0x100), BASE_C + TEXT_VADDR + 0x40);
}

#[test]
fn first_weak_wins_when_no_strong() {
    let mut fa = Fixture::new(EM_X86_64);
    fa.syms = vec![Sym::import("soft")];
    fa.relas = vec![Rela {
        offset: DATA_VADDR + 0x100,
        rtype: 6,
        sym: 1,
        addend: 0,
    }];
    let img_a = fa.build();
    let mut fb = Fixture::new(EM_X86_64);
    fb.syms = vec![Sym::weak("soft", TEXT_VADDR + 0x50)];
    let img_b = fb.build();
    let mut fc = Fixture::new(EM_X86_64);
    fc.syms = vec![Sym::weak("soft", TEXT_VADDR + 0x60)];
    let img_c = fc.build();

    let da = check_image::<X86_64>(&img_a).unwrap().descriptor;
    let db = check_image::<X86_64>(&img_b).unwrap().descriptor;
    let dc = check_image::<X86_64>(&img_c).unwrap().descriptor;
    let views = [
        view_of(&img_a, &da, BASE_A),
        view_of(&img_b, &db, BASE_B),
        view_of(&img_c, &dc, BASE_C),
    ];
    let scope = SymbolScope { modules: &views };
    let tls = TlsLayout::build::<X86_64>(&views, arena()).unwrap();
    let mut data = data_copy(&img_a, &da);
    let mut module = ModuleImage {
        scope_index: 0,
        view: view_of(&img_a, &da, BASE_A),
        data: &mut data,
        relro: None,
    };
    apply_module::<X86_64, _>(&mut module, &scope, &tls).unwrap();
    assert_eq!(read64(&data, 0x100), BASE_B + TEXT_VADDR + 0x50);
}

#[test]
fn missing_symbol_refuses() {
    let mut f = Fixture::new(EM_X86_64);
    f.syms = vec![Sym::import("ghost")];
    f.relas = vec![Rela {
        offset: DATA_VADDR + 0x100,
        rtype: 6,
        sym: 1,
        addend: 0,
    }];
    assert_eq!(
        apply_single::<X86_64>(&f.build(), BASE_A).unwrap_err(),
        ApplyError::MissingSymbol(0)
    );
}

#[test]
fn weak_undef_resolves_to_zero() {
    // The gABI rule: a *weak* reference nothing in scope defines takes
    // S = 0 — GOT slots read zero, Abs64 keeps its addend (0 + A).
    let mut f = Fixture::new(EM_X86_64);
    f.syms = vec![Sym::weak_import("maybe")];
    f.relas = vec![
        Rela {
            offset: DATA_VADDR + 0x100,
            rtype: 6,
            sym: 1,
            addend: 0,
        },
        Rela {
            offset: DATA_VADDR + 0x108,
            rtype: 1,
            sym: 1,
            addend: 5,
        },
    ];
    let mut img = f.build();
    // Seed the slots nonzero to prove the writes happened.
    write_raw64(&mut img, (DATA_VADDR + 0x100) as usize, 0xdead);
    write_raw64(&mut img, (DATA_VADDR + 0x108) as usize, 0xdead);
    let (applied, data) = apply_single::<X86_64>(&img, BASE_A).unwrap();
    assert_eq!(applied.applied, 2);
    assert_eq!(read64(&data, 0x100), 0);
    assert_eq!(read64(&data, 0x108), 5);
}

#[test]
fn weak_undef_tls_still_refuses() {
    // A weak-undefined *TLS* symbol earns no zero: a zero tp-offset
    // would alias a real block, so the TpRel path fails closed.
    let mut f = Fixture::new(EM_X86_64);
    f.tls = Some(TlsSpec {
        memsz: 0x10,
        align: 8,
        filesz: 0,
    });
    f.syms = vec![Sym::weak_import("tmaybe")];
    f.relas = vec![Rela {
        offset: DATA_VADDR + 0x100,
        rtype: 18,
        sym: 1,
        addend: 0,
    }];
    assert_eq!(
        apply_single::<X86_64>(&f.build(), BASE_A).unwrap_err(),
        ApplyError::Tls(0)
    );
}

#[test]
fn zero_bloom_hides_definitions() {
    // The bloom filter is honored: a zeroed bloom word makes B's
    // definition invisible, so resolution refuses.
    let mut fa = Fixture::new(EM_X86_64);
    fa.syms = vec![Sym::import("beta_fn")];
    fa.relas = vec![Rela {
        offset: DATA_VADDR + 0x100,
        rtype: 6,
        sym: 1,
        addend: 0,
    }];
    let img_a = fa.build();
    let mut fb = Fixture::new(EM_X86_64);
    fb.syms = vec![Sym::global("beta_fn", TEXT_VADDR + 0x20)];
    fb.zero_bloom = true;
    let img_b = fb.build();

    let da = check_image::<X86_64>(&img_a).unwrap().descriptor;
    let db = check_image::<X86_64>(&img_b).unwrap().descriptor;
    let views = [view_of(&img_a, &da, BASE_A), view_of(&img_b, &db, BASE_B)];
    let scope = SymbolScope { modules: &views };
    let tls = TlsLayout::build::<X86_64>(&views, arena()).unwrap();
    let mut data = data_copy(&img_a, &da);
    let mut module = ModuleImage {
        scope_index: 0,
        view: view_of(&img_a, &da, BASE_A),
        data: &mut data,
        relro: None,
    };
    assert_eq!(
        apply_module::<X86_64, _>(&mut module, &scope, &tls).unwrap_err(),
        ApplyError::MissingSymbol(0)
    );
}

// ---------------------------------------------------------------------------
// RELR
// ---------------------------------------------------------------------------

#[test]
fn relr_decode_hand_built_bitmap() {
    // Encoding (hand-derived):
    //   word 0 = 0x2100 (even)          → relocate 0x2100; anchor = 0x2108
    //   word 1 = ((1|1<<2) << 1) | 1    → bitmap: bit0 → 0x2108,
    //                                     bit2 → 0x2118 (0x2110 skipped)
    let mut f = Fixture::new(EM_X86_64);
    f.relr = vec![DATA_VADDR + 0x100, ((1 | 1 << 2) << 1) | 1];
    let mut img = f.build();
    // Seed the in-place addends RELR consumes.
    write_raw64(&mut img, (DATA_VADDR + 0x100) as usize, 0x111);
    write_raw64(&mut img, (DATA_VADDR + 0x108) as usize, 0x222);
    write_raw64(&mut img, (DATA_VADDR + 0x110) as usize, 0xdead);
    write_raw64(&mut img, (DATA_VADDR + 0x118) as usize, 0x333);

    let (applied, data) = apply_single::<X86_64>(&img, BASE_A).unwrap();
    assert_eq!(applied.applied, 3);
    assert_eq!(read64(&data, 0x100), BASE_A + 0x111);
    assert_eq!(read64(&data, 0x108), BASE_A + 0x222);
    // The skipped word keeps its raw value — no base added.
    assert_eq!(read64(&data, 0x110), 0xdead);
    assert_eq!(read64(&data, 0x118), BASE_A + 0x333);
}

#[test]
fn applied_counts_all_tables() {
    // 1 (.rela.dyn) + 3 (RELR: anchor + two bitmap bits) + 1 (.rela.plt).
    let mut f = Fixture::new(EM_X86_64);
    f.relas = vec![Rela {
        offset: DATA_VADDR + 0x200,
        rtype: 8,
        sym: 0,
        addend: 1,
    }];
    f.plt_relas = vec![Rela {
        offset: DATA_VADDR + 0x208,
        rtype: 8,
        sym: 0,
        addend: 2,
    }];
    f.relr = vec![DATA_VADDR + 0x100, ((1 | 1 << 2) << 1) | 1];
    let (applied, _) = apply_single::<X86_64>(&f.build(), BASE_A).unwrap();
    assert_eq!(applied.applied, 5);
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

#[test]
fn write_out_of_range_refuses() {
    // Into rodata (below the data window) …
    let mut f = Fixture::new(EM_X86_64);
    f.relas = vec![Rela {
        offset: 0x500,
        rtype: 8,
        sym: 0,
        addend: 0,
    }];
    assert_eq!(
        apply_single::<X86_64>(&f.build(), BASE_A).unwrap_err(),
        ApplyError::WriteOutOfRange(0)
    );
    // … and just past the data segment's memsz.
    let mut f = Fixture::new(EM_X86_64);
    f.relas = vec![Rela {
        offset: DATA_VADDR + PAGE,
        rtype: 8,
        sym: 0,
        addend: 0,
    }];
    assert_eq!(
        apply_single::<X86_64>(&f.build(), BASE_A).unwrap_err(),
        ApplyError::WriteOutOfRange(0)
    );
}

#[test]
fn overflow_refuses() {
    // base + i64::MIN wraps below zero.
    let mut f = Fixture::new(EM_X86_64);
    f.relas = vec![Rela {
        offset: DATA_VADDR + 0x100,
        rtype: 8,
        sym: 0,
        addend: i64::MIN,
    }];
    assert_eq!(
        apply_single::<X86_64>(&f.build(), BASE_A).unwrap_err(),
        ApplyError::Overflow(0)
    );
    // S + i64::MIN likewise (the addend-games case for Abs64).
    let mut f = Fixture::new(EM_X86_64);
    f.syms = vec![Sym::global("obj", DATA_VADDR + 0x300)];
    f.relas = vec![Rela {
        offset: DATA_VADDR + 0x100,
        rtype: 1,
        sym: 1,
        addend: i64::MIN,
    }];
    assert_eq!(
        apply_single::<X86_64>(&f.build(), BASE_A).unwrap_err(),
        ApplyError::Overflow(0)
    );
}

/// Check the pristine image, corrupt the rela `r_info` afterwards, then
/// apply against the corrupted mapping — the post-`check` tamper shape
/// the engine's own gates must catch.
fn apply_tampered(new_r_info: u64) -> ApplyError {
    let mut f = Fixture::new(EM_X86_64);
    f.relas = vec![Rela {
        offset: DATA_VADDR + 0x100,
        rtype: 8,
        sym: 0,
        addend: 0,
    }];
    let mut img = f.build();
    let checked = check_image::<X86_64>(&img).unwrap();
    write_raw64(&mut img, RELA_OFF as usize + 8, new_r_info);
    let d = checked.descriptor;
    let views = [view_of(&img, &d, BASE_A)];
    let scope = SymbolScope { modules: &views };
    let tls = TlsLayout::build::<X86_64>(&views, arena()).unwrap();
    let mut data = data_copy(&img, &d);
    let mut module = ModuleImage {
        scope_index: 0,
        view: view_of(&img, &d, BASE_A),
        data: &mut data,
        relro: checked.relro,
    };
    apply_module::<X86_64, _>(&mut module, &scope, &tls).unwrap_err()
}

#[test]
fn rejected_reloc_defense_in_depth() {
    // A checked image whose rela table is corrupted afterwards (type →
    // COPY): the engine's own classification gate must still refuse.
    assert_eq!(
        apply_tampered(5), // r_info: sym 0, type COPY
        ApplyError::Reloc(RelocRefusal::Rejected(5), 0)
    );
}

#[test]
fn tprel_without_layout_refuses() {
    // A no-TLS image whose rela type is corrupted to TPOFF64 after
    // check: the layout has no block for this module → Tls(0).
    assert_eq!(apply_tampered(18), ApplyError::Tls(0)); // sym 0, TPOFF64
}

#[test]
fn machine_mismatch_refuses_at_apply() {
    // The consumer-side cross-check: an x86_64 descriptor applied
    // through the aarch64 engine refuses before any byte moves.
    let f = Fixture::new(EM_X86_64);
    let img = f.build();
    let checked = check_image::<X86_64>(&img).unwrap();
    let d = checked.descriptor;
    let views = [view_of(&img, &d, BASE_A)];
    let scope = SymbolScope { modules: &views };
    let tls = TlsLayout::build::<AArch64>(&views, arena()).unwrap();
    let mut data = data_copy(&img, &d);
    let mut module = ModuleImage {
        scope_index: 0,
        view: view_of(&img, &d, BASE_A),
        data: &mut data,
        relro: None,
    };
    assert_eq!(
        apply_module::<AArch64, _>(&mut module, &scope, &tls).unwrap_err(),
        ApplyError::Input
    );
}

// ---------------------------------------------------------------------------
// TLS layout (psABI hand-derivations in comments)
// ---------------------------------------------------------------------------

/// A bare descriptor carrying only TLS geometry (layout-building needs
/// nothing else).
fn tls_descriptor(memsz: u64, align: u64) -> ModuleDescriptor {
    let mut d = ModuleDescriptor::empty();
    d.abi.set(IMAGE_ABI);
    d.kind = kind::LIBRARY;
    d.tls.memsz.set(memsz);
    d.tls.align.set(align);
    d
}

fn tls_views(descs: &[ModuleDescriptor]) -> Vec<ModuleView<'_>> {
    descs
        .iter()
        .map(|d| ModuleView {
            base: 0,
            descriptor: d,
            ro: [None; MAX_SEGMENTS],
        })
        .collect()
}

#[test]
fn tls_layout_v2_x86_64() {
    // Variant 2 (x86-64 psABI §3.4.6): blocks below tp, offsets
    // negative.  Sizes 0x18/align 8 and 0x30/align 16:
    //   off₁ = align_up(0x18, 8)         = 0x18 → base₁ = −0x18
    //   off₂ = align_up(0x18 + 0x30, 16) = align_up(0x48, 16) = 0x50
    //                                          → base₂ = −0x50
    //   total = 0x50, align = 16.
    let descs = [tls_descriptor(0x18, 8), tls_descriptor(0x30, 16)];
    let views = tls_views(&descs);
    let tls = TlsLayout::build::<X86_64>(&views, arena()).unwrap();
    assert_eq!(tls.tp_off(0, 0), Some(-0x18));
    assert_eq!(tls.tp_off(0, 0x8), Some(-0x10));
    assert_eq!(tls.tp_off(1, 0), Some(-0x50));
    assert_eq!(tls.tp_off(1, 0x10), Some(-0x40));
    assert_eq!(tls.total, 0x50);
    assert_eq!(tls.align, 16);
}

#[test]
fn tls_layout_v1_aarch64() {
    // Variant 1: blocks above tp after the reserved header, offsets
    // positive.  The header is `max(psABI tcb, RUNTIME_TCB_BYTES)` — the
    // AAELF64 minimum of 16 is a floor, and the runtime's per-thread block
    // lives at tp, so the reserve is a page:
    //   base₁ = align_up(0x1000, 8)  = 0x1000; end = 0x1018
    //   base₂ = align_up(0x1018, 16) = 0x1020; end = 0x1050
    //   total = 0x1050 (header included), align = 16.
    let reserve = kore_abi::image::RUNTIME_TCB_BYTES;
    let descs = [tls_descriptor(0x18, 8), tls_descriptor(0x30, 16)];
    let views = tls_views(&descs);
    let tls = TlsLayout::build::<AArch64>(&views, arena()).unwrap();
    assert_eq!(tls.tp_off(0, 0), Some(reserve as i64));
    assert_eq!(tls.tp_off(0, 0x8), Some(reserve as i64 + 0x8));
    assert_eq!(tls.tp_off(1, 0), Some(reserve as i64 + 0x20));
    assert_eq!(tls.tp_off(1, 0x10), Some(reserve as i64 + 0x30));
    assert_eq!(tls.total, reserve + 0x50);
    assert_eq!(tls.align, 16);
}

#[test]
fn tls_layout_v1_riscv64() {
    // Variant 1 with a psABI tcb of 0 (tp points one past the TCB): the
    // runtime reserve is what the header becomes, so riscv64 and aarch64
    // place identically once both are widened past their own minima.
    //   base₁ = align_up(0x1000, 8)  = 0x1000; end = 0x1018
    //   base₂ = align_up(0x1018, 16) = 0x1020; end = 0x1050
    let reserve = kore_abi::image::RUNTIME_TCB_BYTES;
    let descs = [tls_descriptor(0x18, 8), tls_descriptor(0x30, 16)];
    let views = tls_views(&descs);
    let tls = TlsLayout::build::<Riscv64>(&views, arena()).unwrap();
    assert_eq!(tls.tp_off(0, 0), Some(reserve as i64));
    assert_eq!(tls.tp_off(0, 0x8), Some(reserve as i64 + 0x8));
    assert_eq!(tls.tp_off(1, 0), Some(reserve as i64 + 0x20));
    assert_eq!(tls.tp_off(1, 0x10), Some(reserve as i64 + 0x30));
    assert_eq!(tls.total, reserve + 0x50);
    assert_eq!(tls.align, 16);
}

#[test]
fn tls_layout_gaps_and_bounds() {
    // A TLS-less module gets no block; offsets beyond a block refuse.
    let descs = [tls_descriptor(0, 0), tls_descriptor(0x18, 8)];
    let views = tls_views(&descs);
    let tls = TlsLayout::build::<X86_64>(&views, arena()).unwrap();
    assert_eq!(tls.tp_off(0, 0), None);
    assert_eq!(tls.tp_off(1, 0x19), None); // past the block
    assert_eq!(tls.tp_off(2, 0), None); // past the scope
}

#[test]
fn tprel_end_to_end_x86_64() {
    // Single module, TLS 0x30/align 16 (variant 2):
    //   off₁ = align_up(0x30, 16) = 0x30 → block base = tp − 0x30.
    //   `tvar` at block offset 8 → slot = −0x30 + 8 = −0x28.
    //   sym-0 form with addend 0x10 → slot = −0x30 + 0x10 = −0x20.
    let mut f = Fixture::new(EM_X86_64);
    f.tls = Some(TlsSpec {
        memsz: 0x30,
        align: 16,
        filesz: 0,
    });
    f.syms = vec![Sym {
        name: "tvar".into(),
        value: 8,
        size: 8,
        binding: STB_GLOBAL,
        shndx: 1,
    }];
    f.relas = vec![
        Rela {
            offset: DATA_VADDR + 0x100,
            rtype: 18,
            sym: 1,
            addend: 0,
        },
        Rela {
            offset: DATA_VADDR + 0x108,
            rtype: 18,
            sym: 0,
            addend: 0x10,
        },
    ];
    let (applied, data) = apply_single::<X86_64>(&f.build(), BASE_A).unwrap();
    assert_eq!(applied.applied, 2);
    assert_eq!(read64(&data, 0x100), (-0x28i64) as u64);
    assert_eq!(read64(&data, 0x108), (-0x20i64) as u64);
}

#[test]
fn tprel_cross_module_aarch64() {
    // A imports TLS `tv` from B (variant 1).  Scope [A, B]; the reserved
    // header at `tp` is `RUNTIME_TCB_BYTES` (the runtime's per-thread block
    // lives there), so the first block starts at the reserve rather than at
    // the psABI's 16.
    let mut fa = Fixture::new(EM_AARCH64);
    fa.syms = vec![Sym::import("tv")];
    fa.relas = vec![Rela {
        offset: DATA_VADDR + 0x100,
        rtype: 1030,
        sym: 1,
        addend: 0,
    }];
    // A TpRel reloc requires PT_TLS in its own image at check time; the
    // psABI allows an exe referencing only foreign TLS, but the §7 gate
    // is stricter — give A a tiny block of its own (size 8, align 8),
    // which lands at the reserve and pushes B's to reserve + 8.
    // Expected slot: reserve + 8 + 4.
    fa.tls = Some(TlsSpec {
        memsz: 8,
        align: 8,
        filesz: 0,
    });
    let img_a = fa.build();
    let mut fb = Fixture::new(EM_AARCH64);
    fb.syms = vec![Sym {
        name: "tv".into(),
        value: 4,
        size: 4,
        binding: STB_GLOBAL,
        shndx: 1,
    }];
    fb.tls = Some(TlsSpec {
        memsz: 0x10,
        align: 8,
        filesz: 0,
    });
    let img_b = fb.build();

    let da = check_image::<AArch64>(&img_a).unwrap().descriptor;
    let db = check_image::<AArch64>(&img_b).unwrap().descriptor;
    let views = [view_of(&img_a, &da, BASE_A), view_of(&img_b, &db, BASE_B)];
    let scope = SymbolScope { modules: &views };
    let tls = TlsLayout::build::<AArch64>(&views, arena()).unwrap();
    let mut data = data_copy(&img_a, &da);
    let mut module = ModuleImage {
        scope_index: 0,
        view: view_of(&img_a, &da, BASE_A),
        data: &mut data,
        relro: None,
    };
    apply_module::<AArch64, _>(&mut module, &scope, &tls).unwrap();
    assert_eq!(
        read64(&data, 0x100),
        kore_abi::image::RUNTIME_TCB_BYTES + 0x8 + 4
    );
}

// ---------------------------------------------------------------------------
// Late binding's `dlsym` half: one module, one name, an absolute address.
// ---------------------------------------------------------------------------

#[test]
fn lookup_symbol_rebases_a_definition_and_misses_honestly() {
    let mut f = Fixture::new(EM_X86_64);
    f.syms = vec![
        Sym::global("luaopen_demo", TEXT_VADDR + 0x40),
        Sym::weak("weak_only", TEXT_VADDR + 0x80),
        Sym::import("lua_pushinteger"),
    ];
    let img = f.build();
    let checked = check_image::<X86_64>(&img).unwrap();
    let d = checked.descriptor;
    let view = view_of(&img, &d, BASE_A);
    assert_eq!(
        lookup_symbol(&view, b"luaopen_demo"),
        Ok(Some(BASE_A + TEXT_VADDR + 0x40))
    );
    assert_eq!(
        lookup_symbol(&view, b"weak_only"),
        Ok(Some(BASE_A + TEXT_VADDR + 0x80))
    );
    // An import defines nothing; an unknown name is simply absent.
    assert_eq!(lookup_symbol(&view, b"lua_pushinteger"), Ok(None));
    assert_eq!(lookup_symbol(&view, b"nothing"), Ok(None));
}

#[test]
fn lookup_symbol_refuses_a_module_without_gnu_hash() {
    let f = Fixture::new(EM_X86_64);
    let img = f.build();
    let checked = check_image::<X86_64>(&img).unwrap();
    let mut d = checked.descriptor;
    d.dynz.gnu_hash_off = 0.into();
    let view = view_of(&img, &d, BASE_A);
    assert_eq!(lookup_symbol(&view, b"anything"), Err(ApplyError::Input));
}
