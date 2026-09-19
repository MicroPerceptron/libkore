//! The late-binding plan (`kore_ld::late`): what a checked library maps
//! to when a task links it into its own space after start, and the shapes
//! it refuses rather than guesses.

mod common;

use common::*;
use kore_abi::image::seg;
use kore_ld::check::check_image;
use kore_ld::late::{LateError, LateLayout};
use kore_ld::reloc::X86_64;

const NO_LIMIT: u64 = u64::MAX;

fn library() -> Fixture {
    let mut f = Fixture::new(EM_X86_64);
    f.syms = vec![Sym::global("alpha", TEXT_VADDR + 0x10)];
    f.relas = vec![Rela {
        offset: DATA_VADDR + 0x100,
        rtype: X86_RELOCS.relative,
        sym: 0,
        addend: 0x40,
    }];
    f
}

#[test]
fn a_library_plans_one_code_run_and_one_data_span_above_it() {
    let img = library().build();
    let checked = check_image::<X86_64>(&img).unwrap();
    let plan = LateLayout::plan(&checked, NO_LIMIT, NO_LIMIT).unwrap();
    // The fixture's headers+rodata sit at 0, text at TEXT_VADDR: one run.
    assert_eq!(plan.code_vaddr, 0);
    assert_eq!(plan.code_len, DATA_VADDR);
    let copies: Vec<_> = plan.code_copies.iter().flatten().collect();
    assert_eq!(copies.len(), 2);
    assert!(copies.iter().any(|c| c.dst == TEXT_VADDR));
    // Every copy lands inside the run and names bytes inside the image.
    for c in &copies {
        assert!(c.dst + c.len <= plan.code_len);
        assert!(c.file_off + c.len <= img.len() as u64);
    }
    let data = plan.data.expect("the fixture has a writable segment");
    assert_eq!(data.vaddr, DATA_VADDR);
    assert!(data.vaddr >= plan.code_vaddr + plan.code_len);
    assert_eq!(data.len % 4096, 0);
    assert!(data.bounds_len <= data.len);
    let template = data
        .copies
        .iter()
        .flatten()
        .next()
        .expect("a template copy");
    assert_eq!(template.dst, 0);
    assert_eq!(plan.span, data.vaddr + data.len);
    // The data segment is the only writable one, and it is the last thing placed.
    assert!(
        checked
            .descriptor
            .live_segs()
            .iter()
            .filter(|s| s.kind == seg::KIND_DATA)
            .count()
            == 1
    );
}

#[test]
fn bss_widens_the_data_span_but_not_its_template() {
    let mut f = library();
    f.bss = 0x800;
    let img = f.build();
    let checked = check_image::<X86_64>(&img).unwrap();
    let plan = LateLayout::plan(&checked, NO_LIMIT, NO_LIMIT).unwrap();
    let data = plan.data.unwrap();
    // The check gate normalizes a segment's memsz outward to pages, so the
    // union already ends on a page; the template stays the file's bytes.
    assert_eq!(data.bounds_len, 0x2000);
    assert_eq!(data.len, 0x2000);
    let template = data.copies.iter().flatten().next().unwrap();
    assert_eq!(template.len, 0x1000);
}

#[test]
fn an_executable_is_refused() {
    let mut f = library();
    f.entry = TEXT_VADDR;
    let checked = check_image::<X86_64>(&f.build()).unwrap();
    assert_eq!(
        LateLayout::plan(&checked, NO_LIMIT, NO_LIMIT),
        Err(LateError::NotLibrary)
    );
}

#[test]
fn a_tls_carrying_library_is_refused() {
    let mut f = library();
    f.tls = Some(TlsSpec {
        memsz: 0x10,
        align: 8,
        filesz: 0x10,
    });
    let checked = check_image::<X86_64>(&f.build()).unwrap();
    assert_eq!(
        LateLayout::plan(&checked, NO_LIMIT, NO_LIMIT),
        Err(LateError::Tls)
    );
}

#[test]
fn the_callers_budgets_bound_the_plan() {
    let checked = check_image::<X86_64>(&library().build()).unwrap();
    assert_eq!(
        LateLayout::plan(&checked, DATA_VADDR - 1, NO_LIMIT),
        Err(LateError::TooLarge)
    );
    assert_eq!(
        LateLayout::plan(&checked, NO_LIMIT, IMAGE_LEN as u64 - 1),
        Err(LateError::TooLarge)
    );
    assert!(LateLayout::plan(&checked, DATA_VADDR, NO_LIMIT).is_ok());
}
