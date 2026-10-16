//! Regime validation of the built userland images (`res/IMAGE.md` §7;
//! epic #291, P4 #295): the closed loop behind `make user-verify`.
//!
//! `KORE_USER_IMAGES` names the `user/` cargo target dir; for each
//! `<spec>/release/{libdemo.so,libpci_names.so,hello,time,sleep,status,lspci,help,window,clear,lua,liblua_demo.so,kore-editor}` present this asserts,
//! by raw ELF
//! inspection (the `elf` crate — the same substrate `kore_ld::parse` trusts):
//!
//! - `ET_DYN`, correct `e_machine`;
//! - every `PT_LOAD` page-aligned (vaddr *and* offset), page-rounded
//!   non-overlapping, `p_filesz ≤ p_memsz`, and never `W+X`;
//! - `DT_FLAGS`/`DT_FLAGS_1` carry `BIND_NOW`/`NOW`; no `DT_TEXTREL`;
//! - every `.rela.*` entry classifies into the eager set via
//!   [`kore_ld::reloc::RelocArch::classify`] — zero reject-list/unknown
//!   types (and no non-empty `SHT_REL` section at all);
//! - `libdemo.so` carries `PT_TLS` and ≥1 TPREL-class reloc;
//! - `hello` names `libdemo.so`, while `status` and `lspci` name
//!   `libpci_names.so` through `DT_NEEDED`, proving both shared-library
//!   boundaries survive the build.
//!
//! `KORE_USER_IMAGE_ARCH`, when set, closes the walk to exactly one of
//! `x86_64`, `aarch64`, or `riscv64`. Production packaging uses that
//! selector so a partial build for another ISA cannot become a prerequisite
//! merely because its target directory exists. `KORE_USER_IMAGE_PROFILE`
//! further selects the exact baseline/v2/v3/v4 directory produced by that
//! build, so a stale partial sibling cannot poison a production install. With
//! no selectors every ISA and profile present is checked, as in
//! `make user-verify`.
//!
//! `KORE_USER_IMAGES` unset ⇒ skip: `make test` stays hermetic. An arch whose
//! spec dir is absent is skipped (a gated-out ISA), but a set root with *no*
//! selected images found fails — never silently green.

use std::path::{Path, PathBuf};

use elf::ElfBytes;
use elf::abi;
use elf::endian::AnyEndian;
use kore_ld::reloc::{AArch64, RelocArch, RelocKind, Riscv64, X86_64};

const PAGE: u64 = 4096;

/// Per-image census of eager relocation semantics — the printed
/// evidence that the §7 set (and nothing else) is what the toolchain
/// emitted.
#[derive(Default)]
struct Census {
    relative: usize,
    glob_dat: usize,
    jump_slot: usize,
    abs64: usize,
    tp_rel: usize,
}

impl Census {
    fn bump(&mut self, kind: RelocKind) {
        match kind {
            RelocKind::Relative => self.relative += 1,
            RelocKind::GlobDat => self.glob_dat += 1,
            RelocKind::JumpSlot => self.jump_slot += 1,
            RelocKind::Abs64 => self.abs64 += 1,
            RelocKind::TpRel => self.tp_rel += 1,
        }
    }
}

fn flags_str(p_flags: u32) -> String {
    let mut s = String::new();
    s.push(if p_flags & abi::PF_R != 0 { 'R' } else { '-' });
    s.push(if p_flags & abi::PF_W != 0 { 'W' } else { '-' });
    s.push(if p_flags & abi::PF_X != 0 { 'X' } else { '-' });
    s
}

fn verify_image<A: RelocArch>(path: &Path, kind: ImageKind) {
    let tag = path.display();
    let data = std::fs::read(path).unwrap_or_else(|e| panic!("{tag}: unreadable: {e}"));
    let elf = ElfBytes::<AnyEndian>::minimal_parse(&data)
        .unwrap_or_else(|e| panic!("{tag}: not parseable ELF: {e}"));

    // Identity: ET_DYN only (kore images are PIE/shared-object), right ISA.
    assert_eq!(elf.ehdr.e_type, abi::ET_DYN, "{tag}: not ET_DYN");
    assert_eq!(elf.ehdr.e_machine, A::MACHINE, "{tag}: wrong e_machine");

    // Load geometry: the §2 in-place carve's preconditions.
    let segs = elf
        .segments()
        .unwrap_or_else(|| panic!("{tag}: no program headers"));
    let mut loads: Vec<(u64, u64, u32)> = Vec::new();
    let mut has_tls = false;
    for ph in segs.iter() {
        match ph.p_type {
            abi::PT_TLS => has_tls = true,
            abi::PT_GNU_STACK => {
                assert_eq!(ph.p_flags & abi::PF_X, 0, "{tag}: executable stack");
            }
            abi::PT_LOAD => {
                assert_eq!(ph.p_vaddr % PAGE, 0, "{tag}: PT_LOAD vaddr off-page");
                assert_eq!(ph.p_offset % PAGE, 0, "{tag}: PT_LOAD offset off-page");
                assert!(ph.p_filesz <= ph.p_memsz, "{tag}: filesz > memsz");
                let wx = abi::PF_W | abi::PF_X;
                assert_ne!(ph.p_flags & wx, wx, "{tag}: W+X segment");
                loads.push((ph.p_vaddr, ph.p_memsz, ph.p_flags));
            }
            _ => {}
        }
    }
    assert!(!loads.is_empty(), "{tag}: no PT_LOAD");
    loads.sort_unstable();
    for w in loads.windows(2) {
        let end = (w[0].0 + w[0].1).next_multiple_of(PAGE);
        assert!(end <= w[1].0, "{tag}: PT_LOADs overlap (page-rounded)");
    }

    // Dynamic flags: eager binding declared, text relocations refused.
    let dynamic = elf
        .dynamic()
        .unwrap_or_else(|e| panic!("{tag}: dynamic parse: {e}"))
        .unwrap_or_else(|| panic!("{tag}: no PT_DYNAMIC"));
    let (mut flags, mut flags1, mut textrel) = (0u64, 0u64, false);
    let mut needed: Vec<u64> = Vec::new();
    for d in dynamic.iter() {
        match d.d_tag {
            abi::DT_FLAGS => flags = d.d_val(),
            abi::DT_FLAGS_1 => flags1 = d.d_val(),
            abi::DT_TEXTREL => textrel = true,
            abi::DT_NEEDED => needed.push(d.d_val()),
            _ => {}
        }
    }
    // The `elf` crate types DF_* like d_tag (i64); the values are small
    // positive bit flags, so the cast is lossless.
    assert!(
        flags & abi::DF_BIND_NOW as u64 != 0 || flags1 & abi::DF_1_NOW as u64 != 0,
        "{tag}: neither DF_BIND_NOW nor DF_1_NOW"
    );
    assert!(
        !textrel && flags & abi::DF_TEXTREL as u64 == 0,
        "{tag}: text relocations declared"
    );

    // Relocation gate: everything present classifies eager; REL never
    // appears (all three ISAs are RELA-only under the regime).
    let shdrs = elf
        .section_headers()
        .unwrap_or_else(|| panic!("{tag}: no section headers"));
    let mut census = Census::default();
    for sh in shdrs.iter() {
        assert!(
            sh.sh_type != abi::SHT_REL || sh.sh_size == 0,
            "{tag}: non-empty SHT_REL section"
        );
        if sh.sh_type != abi::SHT_RELA {
            continue;
        }
        for rela in elf
            .section_data_as_relas(&sh)
            .unwrap_or_else(|e| panic!("{tag}: rela parse: {e}"))
        {
            match A::classify(rela.r_type) {
                Ok(kind) => census.bump(kind),
                Err(refusal) => panic!("{tag}: non-eager relocation: {refusal:?}"),
            }
        }
    }

    if matches!(kind, ImageKind::DemoLibrary) {
        assert!(has_tls, "{tag}: libdemo.so must carry PT_TLS");
        assert!(census.tp_rel >= 1, "{tag}: no TPREL-class reloc");
    }
    if let ImageKind::ExportsApi(symbol) | ImageKind::LateModule { opener: symbol, .. } = kind {
        let (symbols, strings) = elf
            .dynamic_symbol_table()
            .unwrap_or_else(|e| panic!("{tag}: dynsym parse: {e}"))
            .unwrap_or_else(|| panic!("{tag}: no .dynsym"));
        let mut defined = false;
        for sym in symbols.iter() {
            let name = strings
                .get(sym.st_name as usize)
                .unwrap_or_else(|e| panic!("{tag}: dynsym strtab: {e}"));
            if name == symbol {
                assert_ne!(sym.st_shndx, abi::SHN_UNDEF, "{tag}: {symbol} is undefined");
                defined = true;
            }
            if let ImageKind::LateModule { host_prefix, .. } = kind
                && sym.st_shndx == abi::SHN_UNDEF
                && !name.is_empty()
            {
                assert!(
                    name.starts_with(host_prefix),
                    "{tag}: undefined {name} is not a {host_prefix}* host import"
                );
            }
        }
        assert!(defined, "{tag}: .dynsym does not define {symbol}");
        if matches!(kind, ImageKind::LateModule { .. }) {
            assert!(!has_tls, "{tag}: a late module may not carry PT_TLS");
            let checked = kore_ld::check::check_image::<A>(&data)
                .unwrap_or_else(|e| panic!("{tag}: late module check: {e:?}"));
            assert!(
                checked.relro.is_some(),
                "{tag}: native witness must exercise RELRO sealing"
            );
            kore_ld::late::LateLayout::plan(&checked, u64::MAX, u64::MAX)
                .unwrap_or_else(|e| panic!("{tag}: late protection layout: {e:?}"));
        }
    }
    if let ImageKind::NeedsLibrary(required) = kind {
        let dynstr_hdr = elf
            .section_header_by_name(".dynstr")
            .unwrap_or_else(|e| panic!("{tag}: shdr walk: {e}"))
            .unwrap_or_else(|| panic!("{tag}: no .dynstr"));
        let dynstr = elf
            .section_data_as_strtab(&dynstr_hdr)
            .unwrap_or_else(|e| panic!("{tag}: dynstr parse: {e}"));
        let names: Vec<&str> = needed
            .iter()
            .map(|&off| {
                dynstr
                    .get(off as usize)
                    .unwrap_or_else(|e| panic!("{tag}: DT_NEEDED strtab: {e}"))
            })
            .collect();
        assert!(
            names.contains(&required),
            "{tag}: DT_NEEDED missing {required} (got {names:?})"
        );
    }

    // Exercise the actual registration gate too: structural ELF inspection
    // alone missed a pure-.tbss image that the image service then refused.
    if !matches!(kind, ImageKind::LateModule { .. }) {
        kore_ld::check::check_image::<A>(&data)
            .unwrap_or_else(|error| panic!("{tag}: image registration refuses: {error:?}"));
    }

    // Evidence for the human reading `make user-verify`.
    for (vaddr, memsz, p_flags) in &loads {
        println!("  load {:#08x}+{:#x} {}", vaddr, memsz, flags_str(*p_flags));
    }
    println!(
        "  relocs: relative={} glob_dat={} jump_slot={} abs64={} tp_rel={}",
        census.relative, census.glob_dat, census.jump_slot, census.abs64, census.tp_rel
    );
}

#[derive(Clone, Copy)]
enum ImageKind {
    DemoLibrary,
    SharedLibrary,
    NeedsLibrary(&'static str),
    Executable,
    /// An executable that exports a C API for late-linked modules: the
    /// named symbol must be defined in `.dynsym`.
    ExportsApi(&'static str),
    /// A late-linked module (`res/IMAGE.md` §8): no PT_TLS, the named
    /// opener defined, and every undefined dynamic symbol carrying the
    /// host prefix — it binds to the host at `require`, never at load.
    LateModule {
        opener: &'static str,
        host_prefix: &'static str,
    },
}

/// Verify one exact profile directory. Aggregate callers may treat absence as
/// a skipped profile; production callers require the directory they built.
fn check_variant<A: RelocArch>(root: &Path, variant: &str, required: bool) -> usize {
    let dir = root.join(variant).join("release");
    if !dir.is_dir() {
        assert!(
            !required,
            "built_images: required profile {variant} not built"
        );
        println!("built_images: {variant} not built — skipped");
        return 0;
    }
    for (name, kind) in [
        ("libdemo.so", ImageKind::DemoLibrary),
        ("libpci_names.so", ImageKind::SharedLibrary),
        ("hello", ImageKind::NeedsLibrary("libdemo.so")),
        ("time", ImageKind::Executable),
        ("sleep", ImageKind::Executable),
        ("status", ImageKind::NeedsLibrary("libpci_names.so")),
        ("lspci", ImageKind::NeedsLibrary("libpci_names.so")),
        ("help", ImageKind::Executable),
        ("window", ImageKind::Executable),
        ("clear", ImageKind::Executable),
        ("inference", ImageKind::Executable),
        ("lua", ImageKind::ExportsApi("lua_pushcclosure")),
        (
            "liblua_demo.so",
            ImageKind::LateModule {
                opener: "luaopen_lua_demo",
                host_prefix: "lua_",
            },
        ),
        ("kore-terminal", ImageKind::Executable),
        ("kore-editor", ImageKind::Executable),
        ("kore-aftershock", ImageKind::Executable),
        ("kore-skylilt", ImageKind::Executable),
    ] {
        println!("built_images: {variant}/{name}");
        verify_image::<A>(&dir.join(name), kind);
    }
    17
}

fn profile_belongs_to(profile: &str, spec: &str) -> bool {
    profile == spec
        || ["-v2", "-v3", "-v4"]
            .iter()
            .any(|suffix| profile == format!("{spec}{suffix}"))
}

fn check_arch<A: RelocArch>(root: &Path, spec: &str, selected: Option<&str>) -> usize {
    if let Some(profile) = selected {
        assert!(
            profile_belongs_to(profile, spec),
            "KORE_USER_IMAGE_PROFILE={profile:?} does not belong to {spec}"
        );
        return check_variant::<A>(root, profile, true);
    }
    [
        String::from(spec),
        format!("{spec}-v2"),
        format!("{spec}-v3"),
        format!("{spec}-v4"),
    ]
    .iter()
    .map(|variant| check_variant::<A>(root, variant, false))
    .sum()
}

#[test]
fn built_images_satisfy_regime() {
    let Some(root) = std::env::var_os("KORE_USER_IMAGES") else {
        println!("built_images: KORE_USER_IMAGES unset — skipped (run `make user-verify`)");
        return;
    };
    let root = PathBuf::from(root);
    let requested = std::env::var("KORE_USER_IMAGE_ARCH").ok();
    let selected = std::env::var("KORE_USER_IMAGE_PROFILE").ok();
    assert!(
        requested.is_some() || selected.is_none(),
        "KORE_USER_IMAGE_PROFILE requires KORE_USER_IMAGE_ARCH"
    );
    let n = match requested.as_deref() {
        None => {
            check_arch::<X86_64>(&root, "x86_64-unknown-kore-user", None)
                + check_arch::<AArch64>(&root, "aarch64-unknown-kore-user", None)
                + check_arch::<Riscv64>(&root, "riscv64gc-unknown-kore-user", None)
        }
        Some("x86_64") => {
            check_arch::<X86_64>(&root, "x86_64-unknown-kore-user", selected.as_deref())
        }
        Some("aarch64") => {
            check_arch::<AArch64>(&root, "aarch64-unknown-kore-user", selected.as_deref())
        }
        Some("riscv64") => {
            check_arch::<Riscv64>(&root, "riscv64gc-unknown-kore-user", selected.as_deref())
        }
        Some(other) => panic!(
            "KORE_USER_IMAGE_ARCH={other:?} is unsupported; expected x86_64, aarch64, or riscv64"
        ),
    };
    assert!(
        n > 0,
        "KORE_USER_IMAGES={} holds no built images",
        root.display()
    );
}
