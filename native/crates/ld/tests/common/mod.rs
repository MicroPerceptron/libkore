//! Synthetic-ELF fixture builder for the kore-ld host suite (std allowed in
//! tests).  Assembles minimal but *structurally honest* ELF64 `ET_DYN`
//! images — ehdr + phdrs + `.dynamic` + `.dynsym`/`.dynstr` + GNU hash +
//! `.rela.dyn`/`.rela.plt`/`.relr.dyn` — parameterized by arch, so the
//! same builder drives the happy paths, the adversarial corpus (tests
//! corrupt the emitted bytes at known offsets), and the apply-engine
//! fixtures.
//!
//! Fixed layout (link base 0 unless `vaddr_base` shifts it):
//!
//! ```text
//! file/vaddr 0x0000..0x1000  R    ehdr, phdrs, .dynstr, .dynsym,
//!                                 .gnu.hash, .rela.dyn, .rela.plt, .relr
//! file/vaddr 0x1000..0x2000  R|X  text (0x90 fill)
//! file/vaddr 0x2000..0x3000  R|W  .dynamic @ 0x2000, data/GOT after,
//!                                 TLS init image @ 0x2800 when present
//! ```

#![allow(dead_code)]

// ---- ELF constants the builder needs (kept local: the builder is an
// ---- independent implementation, not a re-export of the code under test).
pub const ET_DYN: u16 = 3;
pub const ET_EXEC: u16 = 2;
pub const PT_NULL: u32 = 0;
pub const PT_LOAD: u32 = 1;
pub const PT_DYNAMIC: u32 = 2;
pub const PT_TLS: u32 = 7;
pub const PT_GNU_RELRO: u32 = 0x6474_e552;
pub const PF_X: u32 = 1;
pub const PF_W: u32 = 2;
pub const PF_R: u32 = 4;

pub const DT_NULL: i64 = 0;
pub const DT_NEEDED: i64 = 1;
pub const DT_PLTRELSZ: i64 = 2;
pub const DT_STRTAB: i64 = 5;
pub const DT_SYMTAB: i64 = 6;
pub const DT_RELA: i64 = 7;
pub const DT_RELASZ: i64 = 8;
pub const DT_RELAENT: i64 = 9;
pub const DT_STRSZ: i64 = 10;
pub const DT_SYMENT: i64 = 11;
pub const DT_INIT_ARRAY: i64 = 25;
pub const DT_FINI_ARRAY: i64 = 26;
pub const DT_INIT_ARRAYSZ: i64 = 27;
pub const DT_FINI_ARRAYSZ: i64 = 28;
pub const DT_PLTREL: i64 = 20;
pub const DT_TEXTREL: i64 = 22;
pub const DT_JMPREL: i64 = 23;
pub const DT_FLAGS: i64 = 30;
pub const DT_RELRSZ: i64 = 35;
pub const DT_RELR: i64 = 36;
pub const DT_RELRENT: i64 = 37;
pub const DT_GNU_HASH: i64 = 0x6fff_fef5;
pub const DT_FLAGS_1: i64 = 0x6fff_fffb;

pub const DF_TEXTREL: u64 = 0x4;
pub const DF_BIND_NOW: u64 = 0x8;
pub const DF_1_NOW: u64 = 0x1;

pub const STB_GLOBAL: u8 = 1;
pub const STB_WEAK: u8 = 2;
pub const SHN_UNDEF: u16 = 0;

pub const EM_X86_64: u16 = 62;
pub const EM_AARCH64: u16 = 183;
pub const EM_RISCV: u16 = 243;

// ---- Layout anchors (see the module doc).
pub const PAGE: u64 = 0x1000;
pub const RO_VADDR: u64 = 0x0000;
pub const TEXT_VADDR: u64 = 0x1000;
pub const DATA_VADDR: u64 = 0x2000;
pub const DYNAMIC_VADDR: u64 = 0x2000;
pub const TLS_VADDR: u64 = 0x2800;
pub const IMAGE_LEN: usize = 0x3000;

pub const DYNSTR_OFF: u64 = 0x300;
pub const DYNSYM_OFF: u64 = 0x400;
pub const GNU_HASH_OFF: u64 = 0x580;
pub const RELA_OFF: u64 = 0x680;
pub const PLT_RELA_OFF: u64 = 0x800;
pub const RELR_OFF: u64 = 0x8c0;

pub const EHDR_SIZE: usize = 64;
pub const PHDR_SIZE: usize = 56;
pub const PHOFF: usize = 64;

/// Byte offset of phdr `i`'s start (for post-build corruption).
pub fn phdr_off(i: usize) -> usize {
    PHOFF + i * PHDR_SIZE
}

/// Phdr indices in the emitted table (fixed emission order).
pub const PH_RO: usize = 0;
pub const PH_TEXT: usize = 1;
pub const PH_DATA: usize = 2;
pub const PH_DYNAMIC: usize = 3;
// PT_TLS / PT_GNU_RELRO / extra loads follow, in that order, when present.

/// One relocation the builder emits.
#[derive(Clone, Copy, Debug)]
pub struct Rela {
    pub offset: u64,
    pub rtype: u32,
    pub sym: u32,
    pub addend: i64,
}

/// One dynsym entry after the reserved null symbol (dynsym index =
/// position in `Fixture::syms` + 1).
#[derive(Clone, Debug)]
pub struct Sym {
    pub name: String,
    pub value: u64,
    pub size: u64,
    /// `st_info` = binding << 4 (type nibble left 0 — the resolver
    /// doesn't consult it).
    pub binding: u8,
    /// `SHN_UNDEF` = import; anything else = definition.
    pub shndx: u16,
}

impl Sym {
    pub fn global(name: &str, value: u64) -> Self {
        Sym {
            name: name.into(),
            value,
            size: 8,
            binding: STB_GLOBAL,
            shndx: 1,
        }
    }
    pub fn weak(name: &str, value: u64) -> Self {
        Sym {
            name: name.into(),
            value,
            size: 8,
            binding: STB_WEAK,
            shndx: 1,
        }
    }
    pub fn import(name: &str) -> Self {
        Sym {
            name: name.into(),
            value: 0,
            size: 0,
            binding: STB_GLOBAL,
            shndx: SHN_UNDEF,
        }
    }
    pub fn weak_import(name: &str) -> Self {
        Sym {
            name: name.into(),
            value: 0,
            size: 0,
            binding: STB_WEAK,
            shndx: SHN_UNDEF,
        }
    }
}

/// TLS geometry knob.
#[derive(Clone, Copy, Debug)]
pub struct TlsSpec {
    pub memsz: u64,
    pub align: u64,
    pub filesz: u64,
}

/// The parameterized image builder.
#[derive(Clone, Debug)]
pub struct Fixture {
    pub machine: u16,
    pub e_type: u16,
    /// Zero = library; nonzero = executable entry (link vaddr).
    pub entry: u64,
    /// `DT_FLAGS` value (always emitted).
    pub flags: u64,
    /// `DT_FLAGS_1` value (emitted when nonzero).
    pub flags1: u64,
    /// Emit a bare `DT_TEXTREL` tag.
    pub textrel_tag: bool,
    pub relas: Vec<Rela>,
    pub plt_relas: Vec<Rela>,
    /// Raw RELR words (already encoded).
    pub relr: Vec<u64>,
    pub syms: Vec<Sym>,
    pub needed: Vec<String>,
    pub tls: Option<TlsSpec>,
    /// `PT_GNU_RELRO` as `(vaddr, memsz)`.
    pub relro: Option<(u64, u64)>,
    /// Extra `.bss` on the RW load (`p_memsz = 0x1000 + bss`).
    pub bss: u64,
    /// Emit a separate **filesz-0 writable LOAD** of `bss_tail` bytes
    /// right after the data load — the pure-bss tail real linkers emit
    /// (folded into the data segment's consumer memsz by the checker,
    /// never a carve class of its own).  `0` = absent.
    pub bss_tail: u64,
    /// Extra page distance between the data load's rounded end and the
    /// tail's vaddr (0 = contiguous; >0 = the non-contiguous malformed
    /// shape the checker refuses).
    pub bss_tail_gap_pages: u64,
    /// Emit an all-zero bloom word (hides every symbol from GNU-hash
    /// lookups that honor the filter).
    pub zero_bloom: bool,
    /// Emit `DT_INIT_ARRAY`/`DT_FINI_ARRAY` at fixed data offsets.
    pub init_fini: bool,
    /// Shift every vaddr (loads, dynamic tags, entry, relro) by this
    /// page-aligned amount — the nonzero-link-base variant.
    pub vaddr_base: u64,
    /// Extra degenerate PT_LOADs appended (for the segment-count bound).
    pub extra_loads: usize,
    /// Override the emitted `DT_STRTAB` vaddr (a lying tag).
    pub strtab_vaddr_override: Option<u64>,
    /// Override the emitted `DT_RELA` vaddr (a lying tag).
    pub rela_vaddr_override: Option<u64>,
}

impl Default for Fixture {
    fn default() -> Self {
        Fixture {
            machine: EM_X86_64,
            e_type: ET_DYN,
            entry: 0,
            flags: DF_BIND_NOW,
            flags1: 0,
            textrel_tag: false,
            relas: Vec::new(),
            plt_relas: Vec::new(),
            relr: Vec::new(),
            syms: Vec::new(),
            needed: Vec::new(),
            tls: None,
            relro: None,
            bss: 0,
            bss_tail: 0,
            bss_tail_gap_pages: 0,
            zero_bloom: false,
            init_fini: false,
            vaddr_base: 0,
            extra_loads: 0,
            strtab_vaddr_override: None,
            rela_vaddr_override: None,
        }
    }
}

fn w16(buf: &mut [u8], off: usize, v: u16) {
    buf[off..off + 2].copy_from_slice(&v.to_le_bytes());
}
fn w32(buf: &mut [u8], off: usize, v: u32) {
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
}
fn w64(buf: &mut [u8], off: usize, v: u64) {
    buf[off..off + 8].copy_from_slice(&v.to_le_bytes());
}
fn wi64(buf: &mut [u8], off: usize, v: i64) {
    buf[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

/// The GNU hash function — the builder's independent copy (djb2/33).
pub fn gnu_hash(name: &[u8]) -> u32 {
    let mut h: u32 = 5381;
    for &b in name {
        h = h.wrapping_mul(33).wrapping_add(u32::from(b));
    }
    h
}

impl Fixture {
    pub fn new(machine: u16) -> Self {
        Fixture {
            machine,
            ..Default::default()
        }
    }

    /// Assemble the image bytes.
    pub fn build(&self) -> Vec<u8> {
        let vb = self.vaddr_base;
        let mut img = vec![0u8; IMAGE_LEN];

        // ---- .dynstr: NUL, then every sym + needed name.
        let mut dynstr = vec![0u8];
        let name_off = |dynstr: &mut Vec<u8>, name: &str| -> u64 {
            let off = dynstr.len() as u64;
            dynstr.extend_from_slice(name.as_bytes());
            dynstr.push(0);
            off
        };
        let sym_name_offs: Vec<u64> = self
            .syms
            .iter()
            .map(|s| name_off(&mut dynstr, &s.name))
            .collect();
        let needed_offs: Vec<u64> = self
            .needed
            .iter()
            .map(|n| name_off(&mut dynstr, n))
            .collect();
        assert!(dynstr.len() <= 0x100, "dynstr overflows its window");
        let dynstr_len = dynstr.len() as u64;
        img[DYNSTR_OFF as usize..DYNSTR_OFF as usize + dynstr.len()].copy_from_slice(&dynstr);

        // ---- .dynsym: null entry + syms.
        let nsyms = self.syms.len();
        assert!((nsyms + 1) * 24 <= 0x180, "dynsym overflows its window");
        for (i, s) in self.syms.iter().enumerate() {
            let off = DYNSYM_OFF as usize + (i + 1) * 24;
            w32(&mut img, off, sym_name_offs[i] as u32);
            img[off + 4] = s.binding << 4;
            img[off + 5] = 0;
            w16(&mut img, off + 6, s.shndx);
            w64(&mut img, off + 8, s.value);
            w64(&mut img, off + 16, s.size);
        }

        // ---- .gnu.hash: nbuckets=1, symoffset=1, bloom_size=1,
        // bloom_shift=6; all syms (index 1..) hashed into bucket 0.
        {
            let off = GNU_HASH_OFF as usize;
            w32(&mut img, off, 1); // nbuckets
            w32(&mut img, off + 4, 1); // symoffset
            w32(&mut img, off + 8, 1); // bloom_size
            w32(&mut img, off + 12, 6); // bloom_shift
            let bloom: u64 = if self.zero_bloom { 0 } else { u64::MAX };
            w64(&mut img, off + 16, bloom);
            // bucket[0]: first hashed sym index, or 0 when none.
            w32(&mut img, off + 24, if nsyms == 0 { 0 } else { 1 });
            for (i, s) in self.syms.iter().enumerate() {
                let h = gnu_hash(s.name.as_bytes());
                let last = i == nsyms - 1;
                let chain = (h & !1) | u32::from(last);
                w32(&mut img, off + 28 + i * 4, chain);
            }
            assert!(28 + nsyms * 4 <= 0x100, "gnu.hash overflows its window");
        }

        // ---- rela tables + relr.
        let write_relas = |img: &mut Vec<u8>, base_off: u64, relas: &[Rela], cap: usize| {
            assert!(relas.len() * 24 <= cap, "rela table overflows its window");
            // (offsets shift with the link base like every other vaddr)
            for (i, r) in relas.iter().enumerate() {
                let off = base_off as usize + i * 24;
                w64(img, off, r.offset.wrapping_add(vb));
                w64(img, off + 8, (u64::from(r.sym) << 32) | u64::from(r.rtype));
                wi64(img, off + 16, r.addend);
            }
        };
        write_relas(&mut img, RELA_OFF, &self.relas, 0x180);
        write_relas(&mut img, PLT_RELA_OFF, &self.plt_relas, 0xc0);
        assert!(self.relr.len() * 8 <= 0x40, "relr overflows its window");
        for (i, w) in self.relr.iter().enumerate() {
            w64(&mut img, RELR_OFF as usize + i * 8, *w);
        }

        // ---- text fill.
        for b in &mut img[TEXT_VADDR as usize..(TEXT_VADDR + PAGE) as usize] {
            *b = 0x90;
        }

        // ---- .dynamic (at file 0x2000).
        let mut dynamic: Vec<(i64, u64)> = Vec::new();
        dynamic.push((DT_FLAGS, self.flags));
        if self.flags1 != 0 {
            dynamic.push((DT_FLAGS_1, self.flags1));
        }
        if self.textrel_tag {
            dynamic.push((DT_TEXTREL, 0));
        }
        for off in &needed_offs {
            dynamic.push((DT_NEEDED, *off));
        }
        dynamic.push((
            DT_STRTAB,
            self.strtab_vaddr_override.unwrap_or(DYNSTR_OFF + vb),
        ));
        dynamic.push((DT_STRSZ, dynstr_len));
        dynamic.push((DT_SYMTAB, DYNSYM_OFF + vb));
        dynamic.push((DT_SYMENT, 24));
        dynamic.push((DT_GNU_HASH, GNU_HASH_OFF + vb));
        if !self.relas.is_empty() {
            dynamic.push((DT_RELA, self.rela_vaddr_override.unwrap_or(RELA_OFF + vb)));
            dynamic.push((DT_RELASZ, self.relas.len() as u64 * 24));
            dynamic.push((DT_RELAENT, 24));
        }
        if !self.plt_relas.is_empty() {
            dynamic.push((DT_JMPREL, PLT_RELA_OFF + vb));
            dynamic.push((DT_PLTRELSZ, self.plt_relas.len() as u64 * 24));
            dynamic.push((DT_PLTREL, DT_RELA as u64));
        }
        if !self.relr.is_empty() {
            dynamic.push((DT_RELR, RELR_OFF + vb));
            dynamic.push((DT_RELRSZ, self.relr.len() as u64 * 8));
            dynamic.push((DT_RELRENT, 8));
        }
        if self.init_fini {
            dynamic.push((DT_INIT_ARRAY, DATA_VADDR + 0x400 + vb));
            dynamic.push((DT_INIT_ARRAYSZ, 16));
            dynamic.push((DT_FINI_ARRAY, DATA_VADDR + 0x440 + vb));
            dynamic.push((DT_FINI_ARRAYSZ, 8));
        }
        dynamic.push((DT_NULL, 0));
        let dyn_len = dynamic.len() * 16;
        assert!(dyn_len <= 0x200, "dynamic overflows its window");
        for (i, (tag, val)) in dynamic.iter().enumerate() {
            let off = DYNAMIC_VADDR as usize + i * 16;
            wi64(&mut img, off, *tag);
            w64(&mut img, off + 8, *val);
        }

        // ---- ehdr.
        img[0..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
        img[4] = 2; // ELFCLASS64
        img[5] = 1; // ELFDATA2LSB
        img[6] = 1; // EV_CURRENT
        w16(&mut img, 16, self.e_type);
        w16(&mut img, 18, self.machine);
        w32(&mut img, 20, 1); // e_version
        w64(
            &mut img,
            24,
            if self.entry == 0 { 0 } else { self.entry + vb },
        );
        w64(&mut img, 32, PHOFF as u64); // e_phoff
        w64(&mut img, 40, 0); // e_shoff
        w16(&mut img, 52, EHDR_SIZE as u16);
        w16(&mut img, 54, PHDR_SIZE as u16);
        let phnum = 4
            + usize::from(self.tls.is_some())
            + usize::from(self.relro.is_some())
            + usize::from(self.bss_tail != 0)
            + self.extra_loads;
        w16(&mut img, 56, phnum as u16);

        // ---- phdrs.
        let ph = |img: &mut Vec<u8>,
                  i: usize,
                  p_type: u32,
                  p_flags: u32,
                  p_off: u64,
                  p_vaddr: u64,
                  p_filesz: u64,
                  p_memsz: u64,
                  p_align: u64| {
            let off = phdr_off(i);
            w32(img, off, p_type);
            w32(img, off + 4, p_flags);
            w64(img, off + 8, p_off);
            w64(img, off + 16, p_vaddr);
            w64(img, off + 24, p_vaddr); // p_paddr
            w64(img, off + 32, p_filesz);
            w64(img, off + 40, p_memsz);
            w64(img, off + 48, p_align);
        };
        ph(
            &mut img,
            PH_RO,
            PT_LOAD,
            PF_R,
            0,
            RO_VADDR + vb,
            PAGE,
            PAGE,
            PAGE,
        );
        ph(
            &mut img,
            PH_TEXT,
            PT_LOAD,
            PF_R | PF_X,
            TEXT_VADDR,
            TEXT_VADDR + vb,
            PAGE,
            PAGE,
            PAGE,
        );
        ph(
            &mut img,
            PH_DATA,
            PT_LOAD,
            PF_R | PF_W,
            DATA_VADDR,
            DATA_VADDR + vb,
            PAGE,
            PAGE + self.bss,
            PAGE,
        );
        ph(
            &mut img,
            PH_DYNAMIC,
            PT_DYNAMIC,
            PF_R | PF_W,
            DYNAMIC_VADDR,
            DYNAMIC_VADDR + vb,
            dyn_len as u64,
            dyn_len as u64,
            8,
        );
        let mut next = PH_DYNAMIC + 1;
        if let Some(t) = self.tls {
            assert!(t.filesz <= t.memsz && TLS_VADDR + t.filesz <= DATA_VADDR + PAGE);
            ph(
                &mut img,
                next,
                PT_TLS,
                PF_R,
                TLS_VADDR,
                TLS_VADDR + vb,
                t.filesz,
                t.memsz,
                t.align,
            );
            next += 1;
        }
        if let Some((va, len)) = self.relro {
            ph(&mut img, next, PT_GNU_RELRO, PF_R, va, va + vb, len, len, 1);
            next += 1;
        }
        if self.bss_tail != 0 {
            // The pure-bss tail: writable, byte-less, page-aligned and
            // contiguous with the data load's page-rounded end (the
            // shape the checker's fold accepts — the consumer's
            // zero-fill runs straight on from the template).
            let va = (DATA_VADDR + PAGE + self.bss).next_multiple_of(PAGE)
                + self.bss_tail_gap_pages * PAGE;
            ph(
                &mut img,
                next,
                PT_LOAD,
                PF_R | PF_W,
                0,
                va + vb,
                0,
                self.bss_tail,
                PAGE,
            );
            next += 1;
        }
        for i in 0..self.extra_loads {
            // Degenerate but valid loads far away: file-empty, one page.
            let va = 0x10_0000 + (i as u64) * PAGE;
            ph(&mut img, next, PT_LOAD, PF_R, 0, va + vb, 0, PAGE, PAGE);
            next += 1;
        }
        assert_eq!(next, phnum);
        assert!(
            PHOFF + phnum * PHDR_SIZE <= DYNSTR_OFF as usize,
            "phdr table overflows into the dynstr window"
        );

        img
    }
}

/// A leaked private-memory window for mock link-worlds: `AsRef`/`AsMut`
/// materialize a reference only for the duration of each call, so the
/// world's own raw-pointer inspection of the same bytes (op-time
/// snapshots, post-link asserts) never overlaps a live `&mut` — the
/// aliasing discipline of kore-rt's mock `Window`, test-sized.
pub struct RawWin {
    ptr: *mut u8,
    len: usize,
}

impl RawWin {
    /// Leak a zeroed `len`-byte allocation; the paired raw pointer is
    /// the world's inspection handle.
    pub fn leak_zeroed(len: usize) -> (Self, *mut u8) {
        let ptr = Box::into_raw(vec![0u8; len].into_boxed_slice()) as *mut u8;
        (RawWin { ptr, len }, ptr)
    }
}

impl AsRef<[u8]> for RawWin {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: leaked, in-bounds, exclusively the engine's to write.
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl AsMut<[u8]> for RawWin {
    fn as_mut(&mut self) -> &mut [u8] {
        // SAFETY: as above; the borrow lives only for this call.
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

// ---- Post-build corruption helpers (phdr field offsets).
pub const P_TYPE: usize = 0;
pub const P_FLAGS: usize = 4;
pub const P_OFFSET: usize = 8;
pub const P_VADDR: usize = 16;
pub const P_FILESZ: usize = 32;
pub const P_MEMSZ: usize = 40;

/// Overwrite a u64 phdr field in a built image.
pub fn poke64(img: &mut [u8], phdr: usize, field: usize, v: u64) {
    let off = phdr_off(phdr) + field;
    img[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

/// Overwrite a u32 phdr field in a built image.
pub fn poke32(img: &mut [u8], phdr: usize, field: usize, v: u32) {
    let off = phdr_off(phdr) + field;
    img[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

/// Read a little-endian u64 out of a byte slice (assertion helper).
pub fn read64(bytes: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap())
}

// ---- Per-arch relocation numbers (the §7 table, test-side copy).
pub struct ArchRelocs {
    pub machine: u16,
    pub relative: u32,
    pub glob_dat: u32,
    pub jump_slot: u32,
    pub abs64: u32,
    pub tprel: u32,
    pub rejected: [u32; 5],
}

pub const X86_RELOCS: ArchRelocs = ArchRelocs {
    machine: EM_X86_64,
    relative: 8,
    glob_dat: 6,
    jump_slot: 7,
    abs64: 1,
    tprel: 18,
    rejected: [5, 16, 17, 36, 37],
};
pub const AARCH64_RELOCS: ArchRelocs = ArchRelocs {
    machine: EM_AARCH64,
    relative: 1027,
    glob_dat: 1025,
    jump_slot: 1026,
    abs64: 257,
    tprel: 1030,
    rejected: [1024, 1028, 1029, 1031, 1032],
};
pub const RISCV_RELOCS: ArchRelocs = ArchRelocs {
    machine: EM_RISCV,
    relative: 3,
    glob_dat: 2, // R_RISCV_64 doubles as the GOT-slot form
    jump_slot: 5,
    abs64: 2,
    tprel: 11,
    rejected: [4, 7, 9, 12, 58],
};
