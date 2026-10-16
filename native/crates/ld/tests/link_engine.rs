//! Link-engine suite over a minimal recording world — the engine's pure
//! parts in isolation (graph walk/dedup/ordering, placement, TLS block
//! math, the passive page, `relocate_self`), no kore-rt anywhere.  The mock
//! world serves `check_image`-validated fixtures by name; leases are
//! `(image, segment)` indices, windows are leaked slices — the affine
//! shape without the capability plumbing (that layer is kore-rt's e2e).

/// A leaked caller arena — the Phase A capacity seam's test face
/// (`res/USERLAND.md` §7: the tables are arena-carved, caller-sized).
fn arena() -> &'static kore_ld::store::LinkArena<'static> {
    let buf = Box::leak(Box::new([core::mem::MaybeUninit::<u8>::uninit(); 1 << 16]));
    Box::leak(Box::new(kore_ld::store::LinkArena::new(buf)))
}

mod common;

use common::*;
use kore_abi::image::{IMAGE_ABI, ImageName, MAX_SEGMENTS, ModuleDescriptor, ModulePage, seg};
use kore_ld::check::{CheckedImage, check_image};
use kore_ld::link::{
    LinkCaps, LinkError, LinkWorld, LinkedProgram, MAX_LINK_MODULES, SelfRelocError, link_program,
    relocate_self,
};
use kore_ld::reloc::{AArch64, RelocArch, Riscv64, X86_64};
use zerocopy::FromBytes;

// ---------------------------------------------------------------------------
// The recording world
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WErr {
    /// No image of that name registered (the Acquire refusal).
    Unknown,
}

#[derive(Clone, Debug, PartialEq)]
enum Op {
    Base { len: u64, align: u64, va: u64 },
    Map { va: u64, len: u64, access: u8 },
    Read { len: u64 },
    Alloc { va: u64, len: u64 },
    Seal { va: u64, len: u64 },
    SetTp(u64),
}

struct TestWorld {
    /// name → (leaked image bytes, its checked view).
    images: Vec<(ImageName, &'static [u8], CheckedImage)>,
    /// Every acquire, in order (the dedup evidence).
    acquires: Vec<ImageName>,
    ops: Vec<Op>,
    /// Private allocations for post-link inspection: (va, len, ptr).
    allocs: Vec<(u64, usize, *mut u8)>,
    template: Option<(Vec<u8>, u64, u64)>,
    next_base: u64,
    /// A fixed home for the passive page, when the test names one.
    page_at: Option<u64>,
    /// Test knob: withhold segment `.1` of image `.0`'s leases.
    withhold: Option<(ImageName, usize)>,
    /// What `placed_high_water` answers; `None` = a world that does not
    /// record placement (the default, so every existing test stamps 0).
    placed_high_water: Option<u64>,
}

impl TestWorld {
    fn new() -> Self {
        TestWorld {
            images: Vec::new(),
            acquires: Vec::new(),
            ops: Vec::new(),
            allocs: Vec::new(),
            template: None,
            next_base: 0x1000_0000,
            page_at: None,
            withhold: None,
            placed_high_water: None,
        }
    }

    /// How many times the engine asked the world to place something.
    fn base_calls(&self) -> usize {
        self.ops
            .iter()
            .filter(|op| matches!(op, Op::Base { .. }))
            .count()
    }

    /// Whether a mapping was installed at exactly `va`.
    fn mapped_at(&self, va: u64) -> bool {
        self.ops
            .iter()
            .any(|op| matches!(op, Op::Map { va: v, .. } if *v == va))
    }

    fn register<A: RelocArch>(&mut self, name: &str, fixture: &Fixture) {
        self.register_bytes::<A>(name, fixture.build());
    }

    fn register_bytes<A: RelocArch>(&mut self, name: &str, bytes: Vec<u8>) {
        let bytes: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        let checked = check_image::<A>(bytes).expect("fixture must check");
        self.images
            .push((ImageName::new(name).unwrap(), bytes, checked));
    }

    /// The bytes a lease's carve window backs (the in-place carve: a
    /// page-rounded file window of the registered image).
    fn carve_bytes(&self, (img, k): (usize, usize)) -> &'static [u8] {
        let (_, bytes, checked) = &self.images[img];
        let c = checked.carve[k].expect("live segment has a carve window");
        let a = c.file_off as usize;
        let b = (c.file_off + c.len).min(bytes.len() as u64) as usize;
        &bytes[a..b]
    }

    /// The `i`-th `choose_base` result (call order: module loads in BFS
    /// order, then the TLS block when present, then the passive page).
    fn base(&self, i: usize) -> u64 {
        self.ops
            .iter()
            .filter_map(|op| match op {
                Op::Base { va, .. } => Some(*va),
                _ => None,
            })
            .nth(i)
            .expect("base was chosen")
    }

    /// Read one little-endian word out of a private allocation.
    fn word_at(&self, va: u64) -> u64 {
        let mut out = [0u8; 8];
        self.bytes_at(va, &mut out);
        u64::from_le_bytes(out)
    }

    /// Copy bytes out of a private allocation (post-link inspection —
    /// the engine's window handles are dead by then).
    fn bytes_at(&self, va: u64, out: &mut [u8]) {
        let (base, len, ptr) = self
            .allocs
            .iter()
            .copied()
            .find(|&(b, l, _)| va >= b && va + out.len() as u64 <= b + l as u64)
            .expect("va lands in a private allocation");
        let off = (va - base) as usize;
        assert!(off + out.len() <= len);
        // SAFETY: in-bounds within a leaked, initialized allocation; the
        // engine's &mut window over it was dropped with the link.
        unsafe {
            core::ptr::copy_nonoverlapping(ptr.add(off), out.as_mut_ptr(), out.len());
        }
    }
}

impl LinkWorld for TestWorld {
    type Error = WErr;
    type Lease = (usize, usize);
    type Ro = &'static [u8];
    type Rw = RawWin;

    fn placed_high_water(&self) -> Option<u64> {
        self.placed_high_water
    }

    fn acquire(
        &mut self,
        name: &ImageName,
    ) -> Result<(ModuleDescriptor, [Option<(usize, usize)>; MAX_SEGMENTS]), WErr> {
        self.acquires.push(*name);
        let (img, entry) = self
            .images
            .iter()
            .enumerate()
            .find(|(_, (n, ..))| n == name)
            .ok_or(WErr::Unknown)?;
        let d = entry.2.descriptor;
        let mut leases = [None; MAX_SEGMENTS];
        for k in 0..d.seg_count as usize {
            if self.withhold == Some((*name, k)) {
                continue;
            }
            leases[k] = Some((img, k));
        }
        Ok((d, leases))
    }

    fn module_page_base(&mut self, len: u64) -> Result<u64, WErr> {
        match self.page_at {
            Some(va) => {
                self.ops.push(Op::Base {
                    len,
                    align: len,
                    va,
                });
                Ok(va)
            }
            None => self.choose_base(len, len),
        }
    }

    fn choose_base(&mut self, len: u64, align: u64) -> Result<u64, WErr> {
        let align = align.max(0x1000);
        let va = self.next_base.next_multiple_of(align);
        self.next_base = va + len.next_multiple_of(0x1000) + 0x1_0000;
        self.ops.push(Op::Base { len, align, va });
        Ok(va)
    }

    fn map_image(
        &mut self,
        lease: (usize, usize),
        va: u64,
        access: u8,
    ) -> Result<&'static [u8], WErr> {
        assert_eq!(
            access & seg::ACCESS_W,
            0,
            "the engine never asks W on a share"
        );
        let bytes = self.carve_bytes(lease);
        self.ops.push(Op::Map {
            va,
            len: bytes.len() as u64,
            access,
        });
        Ok(bytes)
    }

    fn read_segment(&mut self, lease: (usize, usize)) -> Result<&'static [u8], WErr> {
        let bytes = self.carve_bytes(lease);
        self.ops.push(Op::Read {
            len: bytes.len() as u64,
        });
        Ok(bytes)
    }

    fn alloc_private(&mut self, va: u64, len: u64) -> Result<RawWin, WErr> {
        let (win, ptr) = RawWin::leak_zeroed(len as usize);
        self.allocs.push((va, len as usize, ptr));
        self.ops.push(Op::Alloc { va, len });
        Ok(win)
    }

    fn retain_tls_template(&mut self, bytes: &[u8], tp: u64, align: u64) -> Result<(), WErr> {
        assert!(self.template.is_none());
        self.template = Some((bytes.to_vec(), tp, align));
        Ok(())
    }

    fn seal_ro(&mut self, va: u64, len: u64) -> Result<(), WErr> {
        self.ops.push(Op::Seal { va, len });
        Ok(())
    }

    fn set_tp(&mut self, tp: u64) -> Result<(), WErr> {
        self.ops.push(Op::SetTp(tp));
        Ok(())
    }

    fn lease_id(&self, &(img, k): &(usize, usize)) -> u32 {
        (img * MAX_SEGMENTS + k + 1) as u32
    }
}

fn name(s: &str) -> ImageName {
    ImageName::new(s).unwrap()
}

// ---------------------------------------------------------------------------
// The diamond fixtures (link base 0; per-arch reloc numbers)
// ---------------------------------------------------------------------------

/// exe → liba, libb; both → libc.  Cross-image function refs (GLOB_DAT /
/// JUMP_SLOT), a cross-image data ref (Abs64 + addend), one RELATIVE,
/// init/fini arrays populated via RELATIVE relocs, RELRO on every module.
fn diamond(world: &mut TestWorld, arch: &ArchRelocs) {
    let (mut exe, mut liba, mut libb, mut libc) = (
        Fixture::new(arch.machine),
        Fixture::new(arch.machine),
        Fixture::new(arch.machine),
        Fixture::new(arch.machine),
    );

    exe.entry = TEXT_VADDR;
    exe.needed = vec!["liba".into(), "libb".into()];
    exe.syms = vec![
        Sym::import("a_fn"),
        Sym::import("b_fn"),
        Sym::import("c_data"),
    ];
    exe.relas = vec![
        Rela {
            offset: DATA_VADDR + 0x100,
            rtype: arch.glob_dat,
            sym: 1,
            addend: 0,
        },
        Rela {
            offset: DATA_VADDR + 0x108,
            rtype: arch.glob_dat,
            sym: 2,
            addend: 0,
        },
        Rela {
            offset: DATA_VADDR + 0x110,
            rtype: arch.abs64,
            sym: 3,
            addend: 4,
        },
        Rela {
            offset: DATA_VADDR + 0x118,
            rtype: arch.relative,
            sym: 0,
            addend: 0x123,
        },
        // Init/fini entries become absolute through RELATIVE relocs.
        Rela {
            offset: DATA_VADDR + 0x400,
            rtype: arch.relative,
            sym: 0,
            addend: (TEXT_VADDR + 0x10) as i64,
        },
        Rela {
            offset: DATA_VADDR + 0x408,
            rtype: arch.relative,
            sym: 0,
            addend: (TEXT_VADDR + 0x20) as i64,
        },
        Rela {
            offset: DATA_VADDR + 0x440,
            rtype: arch.relative,
            sym: 0,
            addend: (TEXT_VADDR + 0x30) as i64,
        },
    ];
    exe.fini_count = 2;
    exe.relas.push(Rela {
        offset: DATA_VADDR + 0x448,
        rtype: arch.relative,
        sym: 0,
        addend: (TEXT_VADDR + 0x38) as i64,
    });
    exe.init_fini = true;
    exe.relro = Some((DATA_VADDR, 0x1000));

    liba.needed = vec!["libc".into()];
    liba.syms = vec![Sym::global("a_fn", TEXT_VADDR + 0x10), Sym::import("c_fn")];
    liba.plt_relas = vec![Rela {
        offset: DATA_VADDR + 0x100,
        rtype: arch.jump_slot,
        sym: 2,
        addend: 0,
    }];
    liba.relas = vec![
        Rela {
            offset: DATA_VADDR + 0x400,
            rtype: arch.relative,
            sym: 0,
            addend: (TEXT_VADDR + 0x40) as i64,
        },
        Rela {
            offset: DATA_VADDR + 0x408,
            rtype: arch.relative,
            sym: 0,
            addend: (TEXT_VADDR + 0x50) as i64,
        },
        Rela {
            offset: DATA_VADDR + 0x440,
            rtype: arch.relative,
            sym: 0,
            addend: (TEXT_VADDR + 0x60) as i64,
        },
    ];
    liba.init_fini = true;
    liba.relro = Some((DATA_VADDR, 0x1000));

    libb.needed = vec!["libc".into()];
    libb.syms = vec![
        Sym::global("b_fn", TEXT_VADDR + 0x20),
        Sym::import("c_data"),
    ];
    libb.relas = vec![
        Rela {
            offset: DATA_VADDR + 0x100,
            rtype: arch.glob_dat,
            sym: 2,
            addend: 0,
        },
        Rela {
            offset: DATA_VADDR + 0x400,
            rtype: arch.relative,
            sym: 0,
            addend: (TEXT_VADDR + 0x44) as i64,
        },
        Rela {
            offset: DATA_VADDR + 0x408,
            rtype: arch.relative,
            sym: 0,
            addend: (TEXT_VADDR + 0x54) as i64,
        },
        Rela {
            offset: DATA_VADDR + 0x440,
            rtype: arch.relative,
            sym: 0,
            addend: (TEXT_VADDR + 0x64) as i64,
        },
    ];
    libb.init_fini = true;
    libb.relro = Some((DATA_VADDR, 0x1000));

    libc.syms = vec![
        Sym::global("c_fn", TEXT_VADDR + 0x40),
        Sym::global("c_data", DATA_VADDR + 0x300),
    ];
    libc.relas = vec![
        Rela {
            offset: DATA_VADDR + 0x400,
            rtype: arch.relative,
            sym: 0,
            addend: (TEXT_VADDR + 0x70) as i64,
        },
        Rela {
            offset: DATA_VADDR + 0x408,
            rtype: arch.relative,
            sym: 0,
            addend: (TEXT_VADDR + 0x80) as i64,
        },
        Rela {
            offset: DATA_VADDR + 0x440,
            rtype: arch.relative,
            sym: 0,
            addend: (TEXT_VADDR + 0x90) as i64,
        },
    ];
    libc.init_fini = true;
    libc.relro = Some((DATA_VADDR, 0x1000));

    match arch.machine {
        EM_X86_64 => {
            world.register::<X86_64>("exe", &exe);
            world.register::<X86_64>("liba", &liba);
            world.register::<X86_64>("libb", &libb);
            world.register::<X86_64>("libc", &libc);
        }
        EM_AARCH64 => {
            world.register::<AArch64>("exe", &exe);
            world.register::<AArch64>("liba", &liba);
            world.register::<AArch64>("libb", &libb);
            world.register::<AArch64>("libc", &libc);
        }
        _ => {
            world.register::<Riscv64>("exe", &exe);
            world.register::<Riscv64>("liba", &liba);
            world.register::<Riscv64>("libb", &libb);
            world.register::<Riscv64>("libc", &libc);
        }
    }
}

fn link_diamond<A: RelocArch>(arch: &ArchRelocs) -> (TestWorld, LinkedProgram) {
    let mut world = TestWorld::new();
    diamond(&mut world, arch);
    let prog = link_program::<A, _, _>(&mut world, &name("exe"), arena(), LinkCaps::DEFAULT)
        .expect("the diamond links");
    (world, prog)
}

// ---------------------------------------------------------------------------
// The diamond: dedup, placement, values, ordering, the passive page
// ---------------------------------------------------------------------------

#[test]
fn diamond_links_once_per_name_and_resolves_across_modules() {
    let (world, prog) = link_diamond::<X86_64>(&X86_RELOCS);

    // Dedup: four acquires, one per name — libc exactly once although
    // two modules need it.
    assert_eq!(world.acquires.len(), 4);
    for nm in ["exe", "liba", "libb", "libc"] {
        assert_eq!(
            world.acquires.iter().filter(|n| **n == name(nm)).count(),
            1,
            "{nm} acquired exactly once"
        );
    }

    // BFS placement order: exe, liba, libb, libc.
    let (be, ba, bb, bc) = (world.base(0), world.base(1), world.base(2), world.base(3));
    assert_eq!(prog.module_count, 4);
    assert_eq!(prog.entry, be + TEXT_VADDR);

    // Cross-module GOT/data words, checked as actual bytes in the
    // private copies (the fixtures' link vaddrs are hand-known).
    assert_eq!(
        world.word_at(be + DATA_VADDR + 0x100),
        ba + TEXT_VADDR + 0x10,
        "exe→a_fn"
    );
    assert_eq!(
        world.word_at(be + DATA_VADDR + 0x108),
        bb + TEXT_VADDR + 0x20,
        "exe→b_fn"
    );
    assert_eq!(
        world.word_at(be + DATA_VADDR + 0x110),
        bc + DATA_VADDR + 0x300 + 4,
        "exe→c_data+4"
    );
    assert_eq!(
        world.word_at(be + DATA_VADDR + 0x118),
        be + 0x123,
        "exe relative"
    );
    assert_eq!(
        world.word_at(ba + DATA_VADDR + 0x100),
        bc + TEXT_VADDR + 0x40,
        "liba→c_fn (plt)"
    );
    assert_eq!(
        world.word_at(bb + DATA_VADDR + 0x100),
        bc + DATA_VADDR + 0x300,
        "libb→c_data"
    );

    // Init order: dependency-before-dependent — libc, liba, libb, exe;
    // fini is the exact reverse.
    let init: Vec<u64> = prog.init.iter().copied().collect();
    assert_eq!(
        init,
        vec![
            bc + TEXT_VADDR + 0x70,
            bc + TEXT_VADDR + 0x80,
            ba + TEXT_VADDR + 0x40,
            ba + TEXT_VADDR + 0x50,
            bb + TEXT_VADDR + 0x44,
            bb + TEXT_VADDR + 0x54,
            be + TEXT_VADDR + 0x10,
            be + TEXT_VADDR + 0x20,
        ]
    );
    let fini: Vec<u64> = prog.fini.iter().copied().collect();
    assert_eq!(
        fini,
        vec![
            be + TEXT_VADDR + 0x38,
            be + TEXT_VADDR + 0x30,
            bb + TEXT_VADDR + 0x64,
            ba + TEXT_VADDR + 0x60,
            bc + TEXT_VADDR + 0x90,
        ]
    );

    // No TLS in this diamond: no TP was installed.
    assert_eq!(prog.tp, 0);
    assert!(!world.ops.iter().any(|op| matches!(op, Op::SetTp(_))));
}

#[test]
fn diamond_maps_wx_clean_and_seals_after_relocation() {
    let (world, prog) = link_diamond::<X86_64>(&X86_RELOCS);
    let be = world.base(0);

    // Every share request: text R|X, rodata R — never W (the world
    // asserts too); private allocations never overlap a shared mapping.
    let mut text_maps = 0;
    for op in &world.ops {
        if let Op::Map { access, .. } = op {
            assert!(access & seg::ACCESS_W == 0);
            assert!(*access == seg::ACCESS_R || *access == (seg::ACCESS_R | seg::ACCESS_X));
            if access & seg::ACCESS_X != 0 {
                text_maps += 1;
            }
        }
    }
    assert_eq!(text_maps, 4, "one executable mapping per module");
    for &(ava, alen, _) in &world.allocs {
        for op in &world.ops {
            if let Op::Map { va, len, .. } = op {
                assert!(
                    ava + alen as u64 <= *va || va + len <= ava,
                    "private memory never overlaps a shared mapping"
                );
            }
        }
    }

    // Seal ordering: every RELRO seal comes after every module mapping
    // and allocation (relocation runs strictly between), and by seal
    // time the sealed window already holds the final GOT — checked by
    // reading the exe's GOT word (inside its RELRO span) after link and
    // knowing the world snapshots nothing: bytes are only written by
    // the engine, which cannot write after its windows died.
    let first_seal = world
        .ops
        .iter()
        .position(|op| matches!(op, Op::Seal { .. }))
        .expect("RELRO seals happened");
    let last_module_setup = world
        .ops
        .iter()
        .rposition(|op| matches!(op, Op::Map { .. } | Op::Read { .. }))
        .expect("mappings happened");
    assert!(
        last_module_setup < first_seal,
        "all module mapping precedes the first seal (map-RW → relocate → seal)"
    );
    // Four RELRO seals (one per module, at base+DATA_VADDR) + the
    // passive-page seal.
    let seals: Vec<(u64, u64)> = world
        .ops
        .iter()
        .filter_map(|op| match op {
            Op::Seal { va, len } => Some((*va, *len)),
            _ => None,
        })
        .collect();
    assert_eq!(seals.len(), 5);
    assert!(
        seals.contains(&(be + DATA_VADDR, 0x1000)),
        "the exe's RELRO window"
    );
    assert_eq!(
        *seals.last().unwrap(),
        (prog.module_page, PAGE),
        "the page seal is last"
    );
}

#[test]
fn diamond_writes_the_passive_page() {
    let (world, prog) = link_diamond::<X86_64>(&X86_RELOCS);
    let (be, bc) = (world.base(0), world.base(3));

    // Decode through the frozen wire type rather than duplicating its size
    // or record stride in the test.
    let mut bytes = [0u8; core::mem::size_of::<ModulePage>()];
    world.bytes_at(prog.module_page, &mut bytes);
    let page = ModulePage::read_from_bytes(&bytes).unwrap();
    assert_eq!(page.abi.get(), IMAGE_ABI, "IMAGE_ABI stamp");
    assert_eq!(page.count.get(), 4);

    let r0 = &page.records[0];
    assert_eq!(r0.name_hash.get(), name("exe").fnv1a());
    assert_eq!(r0.base.get(), be);
    assert_eq!(r0.entry.get(), be + TEXT_VADDR);
    assert_eq!(r0.seg_count, 3, "text + rodata + data recorded");
    assert_eq!(r0.kind, kore_abi::image::kind::EXECUTABLE);
    // Segment 0 (text): placed vaddr, page-rounded size, R|X, provenance.
    let s0 = &r0.segs[0];
    assert_eq!(s0.vaddr.get(), be + TEXT_VADDR);
    assert_eq!(s0.size.get(), PAGE);
    assert_ne!(s0.lease_id.get(), 0, "lease provenance rode along");
    assert_eq!(s0.kind, seg::KIND_TEXT);
    assert_eq!(s0.access, seg::ACCESS_R | seg::ACCESS_X);
    // The data record reports the installed private copy: R|W.
    let s2 = &r0.segs[2];
    assert_eq!(s2.kind, seg::KIND_DATA);
    assert_eq!(s2.access, seg::ACCESS_R | seg::ACCESS_W);

    let r3 = &page.records[3];
    assert_eq!(r3.name_hash.get(), name("libc").fnv1a());
    assert_eq!(r3.base.get(), bc);
    assert_eq!(r3.entry.get(), 0, "libraries have no entry");
    assert_eq!(r3.kind, kore_abi::image::kind::LIBRARY);
}

#[test]
fn diamond_links_on_aarch64_and_riscv64_tables() {
    let (world, prog) = link_diamond::<AArch64>(&AARCH64_RELOCS);
    let (be, ba, bc) = (world.base(0), world.base(1), world.base(3));
    assert_eq!(prog.module_count, 4);
    assert_eq!(
        world.word_at(be + DATA_VADDR + 0x100),
        ba + TEXT_VADDR + 0x10
    );
    assert_eq!(
        world.word_at(ba + DATA_VADDR + 0x100),
        bc + TEXT_VADDR + 0x40
    );

    let (world, prog) = link_diamond::<Riscv64>(&RISCV_RELOCS);
    let (be, ba, bc) = (world.base(0), world.base(1), world.base(3));
    assert_eq!(prog.module_count, 4);
    assert_eq!(
        world.word_at(be + DATA_VADDR + 0x100),
        ba + TEXT_VADDR + 0x10
    );
    assert_eq!(
        world.word_at(ba + DATA_VADDR + 0x100),
        bc + TEXT_VADDR + 0x40
    );
}

// ---------------------------------------------------------------------------
// TLS: block construction + TP per variant (hand-derived psABI numbers)
// ---------------------------------------------------------------------------

/// exe (TLS 0x18 align 8, init 8 bytes) → libtls (TLS 0x10 align 16,
/// init 8 bytes, exporting `ctls` at block offset 8).  The exe reads its
/// own block (sym-0 + addend 8) and the library's (`ctls` import).
fn tls_pair(world: &mut TestWorld, arch: &ArchRelocs) {
    let mut exe = Fixture::new(arch.machine);
    exe.entry = TEXT_VADDR;
    exe.needed = vec!["libtls".into()];
    exe.tls = Some(TlsSpec {
        memsz: 0x18,
        align: 8,
        filesz: 8,
    });
    exe.syms = vec![Sym::import("ctls")];
    exe.relas = vec![
        Rela {
            offset: DATA_VADDR + 0x120,
            rtype: arch.tprel,
            sym: 0,
            addend: 8,
        },
        Rela {
            offset: DATA_VADDR + 0x128,
            rtype: arch.tprel,
            sym: 1,
            addend: 0,
        },
    ];
    let mut exe_img = exe.build();
    exe_img[TLS_VADDR as usize..TLS_VADDR as usize + 8]
        .copy_from_slice(&[0xE0, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7]);

    let mut lib = Fixture::new(arch.machine);
    lib.tls = Some(TlsSpec {
        memsz: 0x10,
        align: 16,
        filesz: 8,
    });
    lib.syms = vec![Sym {
        name: "ctls".into(),
        value: 8,
        size: 8,
        binding: STB_GLOBAL,
        shndx: 1,
    }];
    let mut lib_img = lib.build();
    lib_img[TLS_VADDR as usize..TLS_VADDR as usize + 8]
        .copy_from_slice(&[0xC0, 0xC1, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7]);

    match arch.machine {
        EM_X86_64 => {
            world.register_bytes::<X86_64>("exe", exe_img);
            world.register_bytes::<X86_64>("libtls", lib_img);
        }
        EM_AARCH64 => {
            world.register_bytes::<AArch64>("exe", exe_img);
            world.register_bytes::<AArch64>("libtls", lib_img);
        }
        _ => {
            world.register_bytes::<Riscv64>("exe", exe_img);
            world.register_bytes::<Riscv64>("libtls", lib_img);
        }
    }
}

#[test]
fn tls_block_v2_x86_64() {
    // Variant 2: blocks below tp.  exe off = −align_up(0x18, 8) = −0x18;
    // lib off = −align_up(0x18 + 0x10, 16) = −0x30; total 0x30, align 16.
    // The window is align_up(0x30, 16) plus the runtime block reserved at
    // tp (whose first eight bytes are the psABI self-pointer); tp sits
    // 0x30 above its base, and the blocks below it are unaffected by the
    // reserve.
    let mut world = TestWorld::new();
    tls_pair(&mut world, &X86_RELOCS);
    let prog =
        link_program::<X86_64, _, _>(&mut world, &name("exe"), arena(), LinkCaps::DEFAULT).unwrap();

    let be = world.base(0);
    // Base call order: exe, libtls, the TLS block, the passive page.
    let tls_base = world.base(2);
    assert_eq!(prog.tp, tls_base + 0x30);
    assert_eq!(
        prog.tp % 16,
        0,
        "tp satisfies the strictest block alignment"
    );
    assert_eq!(prog.tls_total, 0x30);
    assert_eq!(prog.tls_align, 16);
    assert!(world.ops.contains(&Op::SetTp(prog.tp)));
    assert!(world.ops.contains(&Op::Alloc {
        va: tls_base,
        len: 0x30 + kore_abi::image::RUNTIME_TCB_BYTES
    }));

    // Slots hold tp-relative offsets: own var −0x18 + 8; ctls −0x30 + 8.
    assert_eq!(world.word_at(be + DATA_VADDR + 0x120), (-0x10i64) as u64);
    assert_eq!(world.word_at(be + DATA_VADDR + 0x128), (-0x28i64) as u64);

    // Init images landed at their layout slots; %fs:0 = tp itself.
    let mut got = [0u8; 8];
    world.bytes_at(prog.tp - 0x18, &mut got);
    assert_eq!(got, [0xE0, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7]);
    world.bytes_at(prog.tp - 0x30, &mut got);
    assert_eq!(got, [0xC0, 0xC1, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7]);
    assert_eq!(world.word_at(prog.tp), prog.tp, "the TCB self-pointer");
}

#[test]
fn tls_block_v1_aarch64() {
    // Variant 1: the header reserved at tp is RUNTIME_TCB_BYTES (the
    // runtime's per-thread block lives there), so the exe block sits at
    // align_up(reserve, 8) = reserve, end reserve+0x18; lib at
    // align_up(reserve+0x18, 16) = reserve+0x20, end reserve+0x30;
    // total reserve+0x30; tp = base.
    let reserve = kore_abi::image::RUNTIME_TCB_BYTES;
    let mut world = TestWorld::new();
    tls_pair(&mut world, &AARCH64_RELOCS);
    let prog = link_program::<AArch64, _, _>(&mut world, &name("exe"), arena(), LinkCaps::DEFAULT)
        .unwrap();

    let be = world.base(0);
    let tls_base = world.base(2);
    assert_eq!(prog.tp, tls_base);
    assert_eq!(prog.tls_total, reserve + 0x30);
    assert!(world.ops.contains(&Op::Alloc {
        va: tls_base,
        len: reserve + 0x30
    }));
    assert_eq!(world.word_at(be + DATA_VADDR + 0x120), reserve + 8);
    assert_eq!(world.word_at(be + DATA_VADDR + 0x128), reserve + 0x20 + 8);
    let mut got = [0u8; 8];
    world.bytes_at(prog.tp + reserve, &mut got);
    assert_eq!(got, [0xE0, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7]);
    world.bytes_at(prog.tp + reserve + 0x20, &mut got);
    assert_eq!(got, [0xC0, 0xC1, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7]);
    // The reserve stays zeroed for the runtime to stamp (no DTV here).
    world.bytes_at(prog.tp, &mut got);
    assert_eq!(got, [0; 8]);
}

#[test]
fn tls_block_v1_riscv64() {
    // Variant 1: riscv64's psABI reserves nothing at tp, but the runtime
    // block does, so the header is RUNTIME_TCB_BYTES here too — exe block
    // at the reserve, end +0x18; lib at align_up(+0x18, 16) = +0x20, end
    // +0x30; total reserve+0x30; tp = base.
    let reserve = kore_abi::image::RUNTIME_TCB_BYTES;
    let mut world = TestWorld::new();
    tls_pair(&mut world, &RISCV_RELOCS);
    let prog = link_program::<Riscv64, _, _>(&mut world, &name("exe"), arena(), LinkCaps::DEFAULT)
        .unwrap();

    let be = world.base(0);
    assert_eq!(prog.tp, world.base(2));
    assert_eq!(prog.tls_total, reserve + 0x30);
    assert_eq!(world.word_at(be + DATA_VADDR + 0x120), reserve + 8);
    assert_eq!(world.word_at(be + DATA_VADDR + 0x128), reserve + 0x20 + 8);
    let mut got = [0u8; 8];
    world.bytes_at(prog.tp + reserve, &mut got);
    assert_eq!(got, [0xE0, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7]);
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

#[test]
fn unknown_needed_name_fails_closed() {
    let mut world = TestWorld::new();
    let mut exe = Fixture::new(EM_X86_64);
    exe.entry = TEXT_VADDR;
    exe.needed = vec!["libghost".into()];
    world.register::<X86_64>("exe", &exe);
    assert_eq!(
        link_program::<X86_64, _, _>(&mut world, &name("exe"), arena(), LinkCaps::DEFAULT)
            .unwrap_err(),
        LinkError::World(WErr::Unknown)
    );
}

#[test]
fn dependency_cycle_refuses() {
    let mut world = TestWorld::new();
    let mut exe = Fixture::new(EM_X86_64);
    exe.entry = TEXT_VADDR;
    exe.needed = vec!["liba".into()];
    let mut liba = Fixture::new(EM_X86_64);
    liba.needed = vec!["libb".into()];
    let mut libb = Fixture::new(EM_X86_64);
    libb.needed = vec!["liba".into()];
    world.register::<X86_64>("exe", &exe);
    world.register::<X86_64>("liba", &liba);
    world.register::<X86_64>("libb", &libb);
    assert_eq!(
        link_program::<X86_64, _, _>(&mut world, &name("exe"), arena(), LinkCaps::DEFAULT)
            .unwrap_err(),
        LinkError::DependencyCycle
    );
    // Dedup still held: three acquires, no loop.
    assert_eq!(world.acquires.len(), 3);
}

#[test]
fn overbound_graph_refuses() {
    // A chain one longer than the table: exe → c00 → … → c15.
    let mut world = TestWorld::new();
    let mut exe = Fixture::new(EM_X86_64);
    exe.entry = TEXT_VADDR;
    exe.needed = vec!["c00".into()];
    world.register::<X86_64>("exe", &exe);
    for i in 0..MAX_LINK_MODULES {
        let mut lib = Fixture::new(EM_X86_64);
        if i + 1 < MAX_LINK_MODULES {
            lib.needed = vec![format!("c{:02}", i + 1)];
        }
        world.register::<X86_64>(&format!("c{i:02}"), &lib);
    }
    assert_eq!(
        link_program::<X86_64, _, _>(&mut world, &name("exe"), arena(), LinkCaps::DEFAULT)
            .unwrap_err(),
        LinkError::TooManyModules
    );
}

#[test]
fn kind_mismatch_refuses() {
    // A DT_NEEDED name resolving to an *executable* image is a foreign
    // graph; so is linking a library as the program.
    let mut world = TestWorld::new();
    let mut exe = Fixture::new(EM_X86_64);
    exe.entry = TEXT_VADDR;
    exe.needed = vec!["liba".into()];
    let mut fake = Fixture::new(EM_X86_64);
    fake.entry = TEXT_VADDR; // an exe registered under a library's name
    world.register::<X86_64>("exe", &exe);
    world.register::<X86_64>("liba", &fake);
    assert_eq!(
        link_program::<X86_64, _, _>(&mut world, &name("exe"), arena(), LinkCaps::DEFAULT)
            .unwrap_err(),
        LinkError::BadModule(1)
    );

    let mut world = TestWorld::new();
    let lib = Fixture::new(EM_X86_64);
    world.register::<X86_64>("lib", &lib);
    assert_eq!(
        link_program::<X86_64, _, _>(&mut world, &name("lib"), arena(), LinkCaps::DEFAULT)
            .unwrap_err(),
        LinkError::BadModule(0)
    );
}

#[test]
fn missing_lease_refuses() {
    let mut world = TestWorld::new();
    let mut exe = Fixture::new(EM_X86_64);
    exe.entry = TEXT_VADDR;
    world.register::<X86_64>("exe", &exe);
    world.withhold = Some((name("exe"), 0));
    assert_eq!(
        link_program::<X86_64, _, _>(&mut world, &name("exe"), arena(), LinkCaps::DEFAULT)
            .unwrap_err(),
        LinkError::MissingLease(0)
    );
}

#[test]
fn unresolved_strong_symbol_refuses_the_link() {
    let mut world = TestWorld::new();
    let mut exe = Fixture::new(EM_X86_64);
    exe.entry = TEXT_VADDR;
    exe.syms = vec![Sym::import("ghost")];
    exe.relas = vec![Rela {
        offset: DATA_VADDR + 0x100,
        rtype: X86_RELOCS.glob_dat,
        sym: 1,
        addend: 0,
    }];
    world.register::<X86_64>("exe", &exe);
    assert!(matches!(
        link_program::<X86_64, _, _>(&mut world, &name("exe"), arena(), LinkCaps::DEFAULT)
            .unwrap_err(),
        LinkError::Apply(kore_ld::apply::ApplyError::MissingSymbol(_), 0)
    ));
}

#[test]
fn single_static_module_is_first_class() {
    // The degenerate no-DT_NEEDED case: one module, one record, entry
    // rebased, no TLS, no TP.
    let mut world = TestWorld::new();
    let mut exe = Fixture::new(EM_X86_64);
    exe.entry = TEXT_VADDR + 0x8;
    exe.relas = vec![Rela {
        offset: DATA_VADDR + 0x100,
        rtype: X86_RELOCS.relative,
        sym: 0,
        addend: 1,
    }];
    world.register::<X86_64>("exe", &exe);
    let prog =
        link_program::<X86_64, _, _>(&mut world, &name("exe"), arena(), LinkCaps::DEFAULT).unwrap();
    assert_eq!(prog.module_count, 1);
    assert_eq!(prog.entry, world.base(0) + TEXT_VADDR + 0x8);
    assert_eq!(
        world.word_at(world.base(0) + DATA_VADDR + 0x100),
        world.base(0) + 1
    );
    assert_eq!(prog.tp, 0);
    assert!(prog.init.is_empty());
}

#[test]
fn initializer_in_writable_data_refuses_before_finalization() {
    let mut world = TestWorld::new();
    let mut exe = Fixture::new(EM_X86_64);
    exe.entry = TEXT_VADDR;
    exe.init_fini = true;
    exe.relas = vec![Rela {
        offset: DATA_VADDR + 0x400,
        rtype: X86_RELOCS.relative,
        sym: 0,
        addend: (DATA_VADDR + 0x100) as i64,
    }];
    world.register::<X86_64>("exe", &exe);
    assert!(matches!(
        link_program::<X86_64, _, _>(&mut world, &name("exe"), arena(), LinkCaps::DEFAULT),
        Err(LinkError::Function(
            kore_ld::functions::FunctionError::Target(0),
            0
        ))
    ));
    assert!(
        !world
            .ops
            .iter()
            .any(|op| matches!(op, Op::Seal { .. } | Op::SetTp(_)))
    );
}

/// **The absolute case places itself.**
///
/// `kind::STATIC_EXECUTABLE` is every founded row and the linker stub:
/// its link address IS its runtime address, so the engine must not ask
/// the world where to put it.  Getting this wrong is not a refusal — a
/// world answers a perfectly legitimate address, the segments map, the
/// seal succeeds, and the image sits somewhere its own relocationless
/// code does not believe it is.  Nothing downstream distinguishes that
/// from a correct placement, which is why the assertion is on the
/// *absence* of the call rather than on the resulting base.
#[test]
fn an_absolute_module_places_itself_without_asking_the_world() {
    let mut world = TestWorld::new();
    let mut exe = Fixture::new(EM_X86_64);
    // A real fixed-address image, through the real gate — no relocations
    // (a static executable has none by construction) and no world knob:
    // `check_image` produces this descriptor, so the test exercises the
    // path the founder and the image service take.
    exe.e_type = ET_EXEC;
    exe.entry = TEXT_VADDR + 0x8;
    world.register::<X86_64>("exe", &exe);

    let prog = link_program::<X86_64, _, _>(&mut world, &name("exe"), arena(), LinkCaps::DEFAULT)
        .expect("an absolute executable links");

    assert_eq!(prog.module_count, 1);
    // The entry rides through unchanged — an address, not an offset.
    assert_eq!(prog.entry, TEXT_VADDR + 0x8);
    // Exactly one placement was requested: the passive module page,
    // which is genuinely the world's to choose.  Two would mean the
    // module's base came from the world.
    assert_eq!(
        world.base_calls(),
        1,
        "only the passive page is placed, never the module"
    );
    // ...and the text landed at the descriptor's own address.
    assert!(
        world.mapped_at(TEXT_VADDR),
        "text maps at its link address, not at a chosen base"
    );
}

/// The same descriptor is still refused where it cannot belong: an
/// absolute executable reached through `DT_NEEDED` is a foreign graph.
/// The role admits two *executable* conventions, not two roles.
#[test]
fn an_absolute_module_is_still_refused_in_a_library_slot() {
    let mut world = TestWorld::new();
    let mut exe = Fixture::new(EM_X86_64);
    exe.entry = TEXT_VADDR;
    exe.needed = vec!["dep".into()];
    let mut dep = Fixture::new(EM_X86_64);
    dep.e_type = ET_EXEC;
    dep.entry = TEXT_VADDR;
    world.register::<X86_64>("exe", &exe);
    world.register::<X86_64>("dep", &dep);

    assert_eq!(
        link_program::<X86_64, _, _>(&mut world, &name("exe"), arena(), LinkCaps::DEFAULT)
            .unwrap_err(),
        LinkError::BadModule(1),
        "a DT_NEEDED name admits only a library"
    );
}

// ---------------------------------------------------------------------------
// relocate_self (the P7 crt0 seam)
// ---------------------------------------------------------------------------

fn rela_entry(offset: u64, rtype: u32, sym: u32, addend: i64) -> [u8; 24] {
    let mut e = [0u8; 24];
    e[0..8].copy_from_slice(&offset.to_le_bytes());
    e[8..16].copy_from_slice(&((u64::from(sym) << 32) | u64::from(rtype)).to_le_bytes());
    e[16..24].copy_from_slice(&addend.to_le_bytes());
    e
}

#[test]
fn relocate_self_applies_relative_entries() {
    let base = 0x7f00_0000u64;
    let mut table = Vec::new();
    table.extend_from_slice(&rela_entry(0x10, 8, 0, 0x123));
    table.extend_from_slice(&rela_entry(0x20, 8, 0, -0x8));
    let mut image = [0u8; 0x40];
    assert_eq!(relocate_self::<X86_64>(base, &table, &mut image), Ok(2));
    assert_eq!(read64(&image, 0x10), base + 0x123);
    assert_eq!(read64(&image, 0x20), base - 8);
    // Untouched words stay zero.
    assert_eq!(read64(&image, 0x18), 0);

    // The aarch64 and riscv64 tables drive the same engine.
    let t = rela_entry(0x8, 1027, 0, 0x40).to_vec();
    let mut image = [0u8; 0x20];
    assert_eq!(relocate_self::<AArch64>(base, &t, &mut image), Ok(1));
    assert_eq!(read64(&image, 0x8), base + 0x40);
    let t = rela_entry(0x8, 3, 0, 0x40).to_vec();
    let mut image = [0u8; 0x20];
    assert_eq!(relocate_self::<Riscv64>(base, &t, &mut image), Ok(1));
    assert_eq!(read64(&image, 0x8), base + 0x40);
}

#[test]
fn relocate_self_refuses_everything_else() {
    let mut image = [0u8; 0x40];
    // A non-RELATIVE type (GLOB_DAT) refuses before any write.
    let t = rela_entry(0x10, 6, 0, 0).to_vec();
    assert_eq!(
        relocate_self::<X86_64>(0, &t, &mut image),
        Err(SelfRelocError::Reloc(6, 0))
    );
    // A RELATIVE naming a symbol is malformed.
    let t = rela_entry(0x10, 8, 1, 0).to_vec();
    assert_eq!(
        relocate_self::<X86_64>(0, &t, &mut image),
        Err(SelfRelocError::Reloc(8, 0))
    );
    // A ragged table refuses whole.
    assert_eq!(
        relocate_self::<X86_64>(0, &[0u8; 23], &mut image),
        Err(SelfRelocError::Shape)
    );
    // An escaping offset refuses; nothing before it survives partially —
    // the caller treats any error as fatal (the stub cannot start).
    let t = rela_entry(0x39, 8, 0, 0).to_vec();
    assert_eq!(
        relocate_self::<X86_64>(0, &t, &mut image),
        Err(SelfRelocError::Bounds(0))
    );
    // base + addend overflow refuses.
    let t = rela_entry(0x10, 8, 0, -1).to_vec();
    assert_eq!(
        relocate_self::<X86_64>(0, &t, &mut image),
        Err(SelfRelocError::Overflow(0))
    );
    assert_eq!(read64(&image, 0x10), 0, "refusals wrote nothing");
}

// ---------------------------------------------------------------------------
// The passive page's home is the world's to name (the stub names the
// LINK window's last page so a late loader can read it back).
// ---------------------------------------------------------------------------

#[test]
fn the_passive_page_lands_where_the_world_names_it() {
    let mut world = TestWorld::new();
    world.page_at = Some(0x5fff_f000);
    diamond(&mut world, &X86_RELOCS);
    let prog = link_program::<X86_64, _, _>(&mut world, &name("exe"), arena(), LinkCaps::DEFAULT)
        .expect("the diamond links");
    assert_eq!(prog.module_page, 0x5fff_f000);
    // No module was placed on top of it.
    for op in &world.ops {
        if let Op::Base { va, len, .. } = op {
            if *va != 0x5fff_f000 {
                assert!(va + len <= 0x5fff_f000 || *va > 0x5fff_f000);
            }
        }
    }
}

#[test]
fn retained_tls_template_is_pristine_for_all_architectures() {
    for machine in [EM_X86_64, EM_AARCH64, EM_RISCV] {
        let mut world = TestWorld::new();
        match machine {
            EM_X86_64 => {
                tls_pair(&mut world, &X86_RELOCS);
                link_program::<X86_64, _, _>(&mut world, &name("exe"), arena(), LinkCaps::DEFAULT)
                    .unwrap();
            }
            EM_AARCH64 => {
                tls_pair(&mut world, &AARCH64_RELOCS);
                link_program::<AArch64, _, _>(&mut world, &name("exe"), arena(), LinkCaps::DEFAULT)
                    .unwrap();
            }
            _ => {
                tls_pair(&mut world, &RISCV_RELOCS);
                link_program::<Riscv64, _, _>(&mut world, &name("exe"), arena(), LinkCaps::DEFAULT)
                    .unwrap();
            }
        }
        let (bytes, tp, align) = world.template.unwrap();
        assert!(align.is_power_of_two());
        assert!(
            bytes[tp as usize..tp as usize + kore_abi::image::RUNTIME_TCB_BYTES as usize]
                .iter()
                .all(|b| *b == 0)
        );
        assert!(bytes.iter().any(|b| *b != 0));
    }
}

#[test]
fn pure_tbss_allocates_zeroed_tls_without_a_source_lease_on_every_architecture() {
    for machine in [EM_X86_64, EM_AARCH64, EM_RISCV] {
        let mut world = TestWorld::new();
        let mut image = Fixture::new(machine);
        image.entry = TEXT_VADDR;
        image.tls = Some(TlsSpec {
            memsz: 0x101a,
            filesz: 0,
            align: 8,
        });
        match machine {
            EM_X86_64 => {
                world.register::<X86_64>("exe", &image);
                link_program::<X86_64, _, _>(&mut world, &name("exe"), arena(), LinkCaps::DEFAULT)
                    .unwrap();
            }
            EM_AARCH64 => {
                world.register::<AArch64>("exe", &image);
                link_program::<AArch64, _, _>(&mut world, &name("exe"), arena(), LinkCaps::DEFAULT)
                    .unwrap();
            }
            _ => {
                world.register::<Riscv64>("exe", &image);
                link_program::<Riscv64, _, _>(&mut world, &name("exe"), arena(), LinkCaps::DEFAULT)
                    .unwrap();
            }
        }
        let (bytes, _, _) = world.template.unwrap();
        assert!(bytes.len() >= 0x101a);
        assert!(bytes.iter().all(|byte| *byte == 0));
    }
}

// ---------------------------------------------------------------------------
// The built images, through the engine the startup stub actually runs
// ---------------------------------------------------------------------------

/// The startup stub's caller arena, to the byte.
///
/// `user/runtime/ld-stub` links with a 4 KiB stack-local arena, not this
/// suite's generous 64 KiB one. A fixture that links here and refuses there
/// is the failure this whole test exists to catch *before* a QEMU run, so
/// the capacity has to match the stub rather than the suite.
fn stub_arena() -> &'static kore_ld::store::LinkArena<'static> {
    let buf = Box::leak(Box::new([core::mem::MaybeUninit::<u8>::uninit(); 4096]));
    Box::leak(Box::new(kore_ld::store::LinkArena::new(buf)))
}

/// The caps `user/runtime/ld-stub` passes: a program is itself plus at most
/// one library, with a single init array.
const STUB_CAPS: LinkCaps = LinkCaps {
    max_modules: 2,
    max_inits: 1,
};

/// **Every built executable must link under the stub's own caps.**
///
/// `built_images.rs` asserts the *regime* — segment shape, flags, relocation
/// classes. It does not link, so an image can satisfy every rule it checks
/// and still be refused by `link_program` at boot. That refusal is close to
/// invisible on metal: `refuse_link` keeps the structured diagnostic on the
/// readiness-NACK path, so an exec-spawned foreground program exits with a
/// bare `ExitCode::LINK_FAILED` (`-1`) and the typed `LinkError` is dropped.
///
/// This closes that gap on the host, where the error survives.
///
/// `KORE_USER_IMAGES` unset ⇒ skip, so `make test` stays hermetic.
#[test]
fn built_executables_link_under_the_stub_caps() {
    let Some(root) = std::env::var_os("KORE_USER_IMAGES") else {
        return;
    };
    let dir = std::path::Path::new(&root)
        .join("x86_64-unknown-kore-user")
        .join("release");
    if !dir.is_dir() {
        return;
    }

    // Collect every image present — libraries too, since a `DT_NEEDED`
    // consumer cannot link without the library it names.
    let mut images: Vec<(String, Vec<u8>)> = Vec::new();
    let mut executables = Vec::new();
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .expect("image dir is readable")
        .filter_map(Result::ok)
        .collect();
    entries.sort_by_key(std::fs::DirEntry::file_name);
    for entry in entries {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(stem) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        // Cargo leaves `.d` files and build dirs beside the images.
        if stem.contains('.') && !stem.ends_with(".so") {
            continue;
        }
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        if check_image::<X86_64>(&bytes).is_err() || ImageName::new(stem).is_none() {
            continue;
        }
        if !stem.ends_with(".so") {
            executables.push(stem.to_string());
        }
        images.push((stem.to_string(), bytes));
    }

    // `built_images.rs`'s rule: a set root that selected nothing is a broken
    // harness, not a pass. Silent green here would hide exactly the class of
    // failure this test exists to catch.
    assert!(
        !executables.is_empty(),
        "KORE_USER_IMAGES={} selected no linkable executables",
        dir.display()
    );
    println!(
        "linking {} built executables: {}",
        executables.len(),
        executables.join(", ")
    );

    // A fresh world per link: the engine consumes leases, so a world that
    // already linked one program is not a clean stand-in for the founding of
    // the next.
    let mut refused = Vec::new();
    for exe in &executables {
        let mut world = TestWorld::new();
        for (stem, bytes) in &images {
            world.register_bytes::<X86_64>(stem, bytes.clone());
        }
        if let Err(error) =
            link_program::<X86_64, _, _>(&mut world, &name(exe), stub_arena(), STUB_CAPS)
        {
            refused.push(format!("{exe}: {error:?}"));
        }
    }

    assert!(
        refused.is_empty(),
        "these built images link on the host but would exit LINK_FAILED (-1) on \
         metal with no diagnostic:\n  {}",
        refused.join("\n  ")
    );
}

// ---------------------------------------------------------------------------
// The placement high water stamped into the passive page
// ---------------------------------------------------------------------------

/// A world that reports where its placement ended gets that mark stamped
/// into the page, as pages below the page — and it reads back as the
/// address a late placer must start above.
#[test]
fn a_reported_placement_cursor_is_stamped_below_the_page() {
    let mut world = TestWorld::new();
    diamond(&mut world, &X86_RELOCS);
    let page_va = 0x5fff_f000;
    world.page_at = Some(page_va);
    world.placed_high_water = Some(page_va - 3 * PAGE);
    let prog = link_program::<X86_64, _, _>(&mut world, &name("exe"), arena(), LinkCaps::DEFAULT)
        .expect("links");
    assert_eq!(prog.module_page, page_va);

    let mut bytes = [0u8; core::mem::size_of::<ModulePage>()];
    world.bytes_at(prog.module_page, &mut bytes);
    let page = ModulePage::read_from_bytes(&bytes).unwrap();
    assert_eq!(page.placed.get(), 3);
    assert_eq!(
        page.placed_high_water(prog.module_page),
        Some(page_va - 3 * PAGE)
    );
}

/// A world that does not record placement stamps nothing, and the page
/// says *unrecorded* rather than answering an address.
#[test]
fn a_world_without_placement_stamps_unrecorded() {
    let (world, prog) = link_diamond::<X86_64>(&X86_RELOCS);
    let mut bytes = [0u8; core::mem::size_of::<ModulePage>()];
    world.bytes_at(prog.module_page, &mut bytes);
    let page = ModulePage::read_from_bytes(&bytes).unwrap();
    assert_eq!(page.placed.get(), 0);
    assert_eq!(page.placed_high_water(prog.module_page), None);
}

/// A cursor the engine cannot represent — above the page, or off a page
/// boundary — refuses the link. Stamping a floor a later placer would
/// trust is the one thing worse than not stamping.
#[test]
fn an_unrepresentable_placement_cursor_refuses_the_link() {
    let page_va = 0x5fff_f000;
    for bad in [page_va + PAGE, page_va - PAGE / 2] {
        let mut world = TestWorld::new();
        diamond(&mut world, &X86_RELOCS);
        world.page_at = Some(page_va);
        world.placed_high_water = Some(bad);
        let err =
            link_program::<X86_64, _, _>(&mut world, &name("exe"), arena(), LinkCaps::DEFAULT)
                .expect_err("a lying placement cursor refuses");
        assert!(matches!(err, LinkError::Layout(_)), "{err:?}");
    }
}
