//! **The firmware set, read before a byte of it moves.**
//!
//! `res/MT7925_EVIDENCE.md` §5 records the fact that shapes this module: the
//! MT7925 takes **two blobs, not one**, against the RTL8922AE's single
//! container. They are not two of the same thing either — the ROM patch
//! carries a **header** with big-endian fields, and the RAM code carries a
//! **trailer** with little-endian ones. Two formats, two parsers, one rung.
//!
//! Nothing here touches the device. Parsing is separated from downloading on
//! purpose: a blob that is the wrong shape must be refused before the part is
//! powered, and a parser that only runs with hardware attached is a parser
//! nobody tests. Every structure below is checked against the real
//! distribution bytes in this crate's tests.
//!
//! Field layouts are transcribed from `mt76_connac_mcu.h` at the pin in
//! `tools/mt7925.h` (`mt76_connac2_patch_hdr`, `mt76_connac2_patch_sec`,
//! `mt76_connac2_fw_trailer`, `mt76_connac2_fw_region`).
//!
//! # The image is longer than its regions, and that is correct
//!
//! The RAM code is 1,392,088 bytes and its five regions claim 1,391,680 of
//! them; the region table and the trailer account for 236 more. **172 bytes
//! are left over**, and for a while that gap was the reason to distrust this
//! parser. It is not a misparse — `res/MT7925_EVIDENCE.md` §14 dumps it:
//! between the last region's payload and the region table sits a build-info
//! block fenced by sixteen `'#'` bytes at each end, carrying a length word
//! and three length-tagged records (the build tag, a `CMD_VER` string and a
//! 16-byte binary field).
//!
//! **Upstream admits it by construction.** `mt76_connac2_fw_check`
//! (`mt76_connac_mcu.c:3048-3077`) checks `region_len > avail - len` — a
//! `<=` over the sum, never an `==` — so slack between the payloads and the
//! table is legal and is simply never read. [`RamCode::slack_bytes`] reports
//! it rather than hiding it, because a parser that silently tolerates an
//! unexplained remainder is a parser that cannot tell slack from a
//! truncation.
//!
//! # What actually crosses into the device is less than the file
//!
//! Two deductions, and both change a number the plan carries:
//!
//! - the RAM code's **last region is `FW_FEATURE_NON_DL`** (and
//!   `FW_TYPE_CLC`): 303,936 bytes of country-location data that
//!   `mt76_connac_mcu_send_ram_firmware` skips with `goto next`
//!   (`mt76_connac_mcu.c:3101`) while still advancing the image offset past
//!   it. [`RamRegion::is_downloaded`] is that bit, and
//!   [`RamCode::download_bytes`] is the sum that respects it;
//! - the patch's sections cover 187,968 of its 188,192 bytes — the first 224
//!   are the header and the two section records.
//!
//! So **1,275,712 bytes cross the WFDMA ring**, not the 1,580,280 that
//! `res/NETWORK_RESTART.md` §4 derives the DMA carve from. The larger figure
//! is the correct size of the *files*; it is the wrong size for the
//! *transfer*.

/// The firmware pin, read out of `firmware/manifest.toml` **at compile
/// time**, so the size and digest exist in exactly one place and a re-pin is
/// one edit reviewed as one diff.
///
/// Generalised over the entry name, because this part takes two blobs and a
/// second copy of the parser is a second place for them to disagree. The
/// shape is `drivers/net/rtw89/src/fw.rs`'s, which pins one.
mod manifest {
    const TEXT: &str = include_str!("../firmware/manifest.toml");

    const fn find(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
        let mut i = from;
        while i + needle.len() <= hay.len() {
            let mut j = 0;
            while j < needle.len() && hay[i + j] == needle[j] {
                j += 1;
            }
            if j == needle.len() {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    /// The byte index just past `key   = ` on the first line of `entry`
    /// that starts with `key`.
    const fn value_of(entry: &[u8], key: &[u8]) -> usize {
        let text = TEXT.as_bytes();
        let Some(table) = find(text, entry, 0) else {
            panic!("manifest.toml: no such [firmware.*] entry");
        };
        let mut line = table;
        loop {
            let Some(nl) = find(text, b"\n", line) else {
                panic!("manifest.toml: entry lacks the key");
            };
            line = nl + 1;
            if line + key.len() <= text.len() {
                let mut j = 0;
                while j < key.len() && text[line + j] == key[j] {
                    j += 1;
                }
                if j == key.len() {
                    let mut i = line + key.len();
                    while text[i] == b' ' || text[i] == b'=' {
                        i += 1;
                    }
                    return i;
                }
            }
            if text[line] == b'[' {
                panic!("manifest.toml: entry lacks the key");
            }
        }
    }

    /// `size = <decimal>`.
    pub const fn size(entry: &[u8]) -> usize {
        let text = TEXT.as_bytes();
        let mut i = value_of(entry, b"size");
        let mut n: usize = 0;
        let mut digits = 0;
        while i < text.len() && text[i].is_ascii_digit() {
            n = n * 10 + (text[i] - b'0') as usize;
            i += 1;
            digits += 1;
        }
        assert!(digits > 0, "manifest.toml: size is not a decimal");
        n
    }

    const fn hex(c: u8) -> u8 {
        match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            b'A'..=b'F' => c - b'A' + 10,
            _ => panic!("manifest.toml: blake3 is not 64 hex digits"),
        }
    }

    /// `blake3 = "<64 hex>"`.
    pub const fn digest(entry: &[u8]) -> [u8; 32] {
        let text = TEXT.as_bytes();
        let i = value_of(entry, b"blake3");
        assert!(text[i] == b'"', "manifest.toml: blake3 is not quoted");
        let mut out = [0u8; 32];
        let mut k = 0;
        while k < 32 {
            out[k] = (hex(text[i + 1 + 2 * k]) << 4) | hex(text[i + 2 + 2 * k]);
            k += 1;
        }
        assert!(
            text[i + 65] == b'"',
            "manifest.toml: blake3 is not 64 hex digits"
        );
        out
    }
}

/// The ROM patch's pinned length (`[firmware.mt7925_patch_mcu] size`).
pub const PATCH_PINNED_SIZE: usize = manifest::size(b"[firmware.mt7925_patch_mcu]");
/// …and its BLAKE3 digest.
pub const PATCH_PINNED_DIGEST: [u8; 32] = manifest::digest(b"[firmware.mt7925_patch_mcu]");
/// The N9 RAM code's pinned length (`[firmware.mt7925_ram_code] size`).
pub const RAM_PINNED_SIZE: usize = manifest::size(b"[firmware.mt7925_ram_code]");
/// …and its BLAKE3 digest.
pub const RAM_PINNED_DIGEST: [u8; 32] = manifest::digest(b"[firmware.mt7925_ram_code]");

const _: () = assert!(
    PATCH_PINNED_SIZE != RAM_PINNED_SIZE,
    "the two entries resolved to the same blob: the parser found one entry twice"
);

/// Take the pinned prefix of a mapped boot module, or refuse.
///
/// **The module is page-rounded and the blob is not**, so the length comes
/// from the manifest and never from the mapping — a digest over the padding
/// would never match, and a digest over whatever the mapping happens to be
/// long would not be a pin at all.
pub fn pinned<'a>(module: &'a [u8], size: usize, want: &[u8; 32]) -> Result<&'a [u8], FwError> {
    let Some(blob) = module.get(..size) else {
        return Err(FwError::Truncated);
    };
    let got = blake3::hash(blob);
    if got.as_bytes() != want {
        return Err(FwError::DigestMismatch);
    }
    Ok(blob)
}
/// What a firmware image could not be read as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FwError {
    /// The module is shorter than the length the manifest pins.
    ///
    /// Distinct from [`Self::TooShort`], which is a blob whose own header
    /// claims more than the blob holds: this one is the *boot module* not
    /// carrying the bytes the pin says it should, which is a provisioning
    /// fault rather than a parse one.
    Truncated,
    /// The module carries the pinned number of bytes and they are not the
    /// pinned bytes.
    DigestMismatch,
    /// The image is shorter than the structure it must contain.
    TooShort,
    /// The region count is larger than this parser's bound.
    TooManyRegions,
    /// A region's extent falls outside the image.
    RegionOutOfRange,
    /// A patch section is not the informational type the loader requires.
    BadSectionType,
}

/// The most regions either blob may declare.
///
/// The distribution's own images use **2** (patch) and **5** (RAM code), so
/// this is a bound with room rather than a fit. It exists because
/// `n_region` is a byte read out of a file: an unbounded loop over it is a
/// file deciding how long this driver runs.
pub const MAX_REGIONS: usize = 16;

// ── the ROM patch: a header, big-endian ──────────────────────────────────

/// Bytes of [`PatchHeader`] on the wire.
pub const PATCH_HDR_BYTES: usize = 96;
/// Bytes of one [`PatchSection`] on the wire.
pub const PATCH_SEC_BYTES: usize = 64;

/// `PATCH_SEC_TYPE_INFO` after `PATCH_SEC_TYPE_MASK` — the only section type
/// the loader accepts (`mt76_connac_mcu.c`'s `load_patch`).
const PATCH_SEC_TYPE_MASK: u32 = 0x0000_ffff;
const PATCH_SEC_TYPE_INFO: u32 = 0x2;

// ── the download-mode word, and the two fields that compose it ───────────
//
// `init_download` takes a `mode` the host computes from the blob's own
// bytes: a *patch section* composes it from `info.sec_key_idx`
// (`mt76_connac2_get_data_mode`, `mt76_connac_mcu.c:3283-3307`), a *RAM
// region* from `feature_set` (`mt76_connac_mcu_gen_dl_mode`,
// `mt76_connac_mcu.h:1927-1944`). Two different fields, one word — which is
// why [`PatchSection::data_mode`] and [`RamRegion::dl_mode`] are separate
// functions rather than one helper over "the mode field".

/// `DL_MODE_ENCRYPT` — `mt76_connac_mcu.h:17`.
pub const DL_MODE_ENCRYPT: u32 = 1 << 0;
/// `DL_MODE_KEY_IDX` — `mt76_connac_mcu.h:18`, `GENMASK(2, 1)`.
pub const DL_MODE_KEY_IDX: u32 = 0b110;
/// `DL_MODE_RESET_SEC_IV` — `mt76_connac_mcu.h:19`.
pub const DL_MODE_RESET_SEC_IV: u32 = 1 << 3;
/// `DL_MODE_WORKING_PDA_CR4` — `mt76_connac_mcu.h:20`. The WA image's bit;
/// this part has no WA blob, so nothing here ever sets it.
pub const DL_MODE_WORKING_PDA_CR4: u32 = 1 << 4;
/// `DL_CONFIG_ENCRY_MODE_SEL` — `mt76_connac_mcu.h:22`.
pub const DL_CONFIG_ENCRY_MODE_SEL: u32 = 1 << 6;
/// `DL_MODE_NEED_RSP` — `mt76_connac_mcu.h:23`. **Always set** by both
/// composers, which is the fact that makes the download a request/response
/// ladder rather than a stream of writes.
pub const DL_MODE_NEED_RSP: u32 = 1 << 31;

/// `PATCH_SEC_ENC_TYPE_MASK` — `mt76_connac_mcu.h:33`, `GENMASK(31, 24)`.
const PATCH_SEC_ENC_TYPE_MASK: u32 = 0xff00_0000;
/// `PATCH_SEC_ENC_TYPE_PLAIN` — `mt76_connac_mcu.h:34`.
const PATCH_SEC_ENC_TYPE_PLAIN: u32 = 0x00;
/// `PATCH_SEC_ENC_TYPE_AES` — `mt76_connac_mcu.h:35`.
const PATCH_SEC_ENC_TYPE_AES: u32 = 0x01;
/// `PATCH_SEC_ENC_TYPE_SCRAMBLE` — `mt76_connac_mcu.h:36`.
const PATCH_SEC_ENC_TYPE_SCRAMBLE: u32 = 0x02;
/// `PATCH_SEC_ENC_AES_KEY_MASK` — `mt76_connac_mcu.h:38`, `GENMASK(7, 0)`.
const PATCH_SEC_ENC_AES_KEY_MASK: u32 = 0xff;
/// `PATCH_SEC_NOT_SUPPORT` — `mt76_connac_mcu.h:32`; a section whose key
/// index reads all-ones asks for the plain mode word and nothing else.
const PATCH_SEC_NOT_SUPPORT: u32 = 0xffff_ffff;

/// One downloadable piece of the ROM patch.
///
/// Five fields where upstream's `mt76_connac2_patch_sec` has a `size` and an
/// `info` union that *also* carries a length. Both are kept, and the
/// distinction is not cosmetic: `mt76_connac2_patch_check` validates
/// `sec->offs` against `sec->info.len` (`mt76_connac_mcu.c:3325-3331`) and
/// `load_patch` downloads `info.len` bytes (`:3384-3392`), while `size` is
/// never read. They agree on the distribution's image — both `0x9a80` and
/// both `0x243c0` — and a parser that collapsed them would not notice the
/// day they stop.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct PatchSection {
    /// `sec->offs` — offset of the payload within the image.
    pub offs: u32,
    /// `sec->size` — the section record's own length field. **Not** what is
    /// downloaded; see [`PatchSection::len`].
    pub size: u32,
    /// `sec->info.addr` — where it lands in the part's address space.
    pub addr: u32,
    /// `sec->info.len` — the byte count `load_patch` actually sends.
    pub len: u32,
    /// `sec->info.sec_key_idx` — the encryption descriptor the mode word is
    /// composed from. It is **not** a mode word itself; the composition is
    /// [`PatchSection::data_mode`].
    pub sec_key_idx: u32,
}

impl PatchSection {
    /// The `mode` word `init_download` takes for this section.
    ///
    /// `mt76_connac2_get_data_mode` (`mt76_connac_mcu.c:3283-3307`), for a
    /// connac2 part. The distribution's two sections both carry
    /// `sec_key_idx = 0x01000000` — encryption type **AES**, key index 0 —
    /// so both ask for `NEED_RSP | ENCRYPT | RESET_SEC_IV`.
    ///
    /// An unrecognised encryption type is **not** a refusal here, matching
    /// upstream, which logs and returns the mode composed so far: the part
    /// is the authority on whether it can decrypt what it was handed, and a
    /// host-side guess about an unknown type would refuse images that work.
    pub const fn data_mode(self) -> u32 {
        let mode = DL_MODE_NEED_RSP;
        if self.sec_key_idx == PATCH_SEC_NOT_SUPPORT {
            return mode;
        }
        let info = self.sec_key_idx;
        match (info & PATCH_SEC_ENC_TYPE_MASK) >> PATCH_SEC_ENC_TYPE_MASK.trailing_zeros() {
            PATCH_SEC_ENC_TYPE_PLAIN => mode,
            PATCH_SEC_ENC_TYPE_AES => {
                let key = (info & PATCH_SEC_ENC_AES_KEY_MASK)
                    << DL_MODE_KEY_IDX.trailing_zeros();
                mode | DL_MODE_ENCRYPT | (key & DL_MODE_KEY_IDX) | DL_MODE_RESET_SEC_IV
            }
            PATCH_SEC_ENC_TYPE_SCRAMBLE => {
                mode | DL_MODE_ENCRYPT | DL_CONFIG_ENCRY_MODE_SEL | DL_MODE_RESET_SEC_IV
            }
            _ => mode,
        }
    }
}

/// The ROM patch's header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PatchHeader {
    /// `build_date`, verbatim — 16 bytes, not NUL-terminated.
    pub build_date: [u8; 16],
    /// `platform`, 4 bytes (`ALPS` on the distribution's image).
    pub platform: [u8; 4],
    /// The version word `mt7925e` prints as `HW/SW Version`.
    pub hw_sw_ver: u32,
    /// How many sections follow the header.
    pub n_region: u32,
}

/// A parsed ROM patch: its header and its sections.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Patch {
    /// The header.
    pub header: PatchHeader,
    sections: [PatchSection; MAX_REGIONS],
    count: usize,
}

impl Patch {
    /// The sections, in file order.
    pub fn sections(&self) -> &[PatchSection] {
        &self.sections[..self.count]
    }

    /// Bytes this blob sends to the part.
    ///
    /// **Not the file's length.** The distribution's patch is 188,192 bytes
    /// and its two sections cover 187,968 of them; the first 224 are the
    /// 96-byte header and the two 64-byte section records, which are read by
    /// the host and never sent.
    pub fn download_bytes(&self) -> u64 {
        self.sections().iter().map(|s| u64::from(s.len)).sum()
    }

    /// Read a ROM patch image.
    ///
    /// Every section's extent is checked against the image here rather than
    /// at download time: the loop that moves bytes should not also be the
    /// loop that decides whether the bytes exist.
    pub fn parse(image: &[u8]) -> Result<Self, FwError> {
        if image.len() < PATCH_HDR_BYTES {
            return Err(FwError::TooShort);
        }
        let be = |at: usize| -> u32 {
            u32::from_be_bytes([image[at], image[at + 1], image[at + 2], image[at + 3]])
        };
        let mut build_date = [0u8; 16];
        build_date.copy_from_slice(&image[0..16]);
        let mut platform = [0u8; 4];
        platform.copy_from_slice(&image[16..20]);
        let header = PatchHeader {
            build_date,
            platform,
            hw_sw_ver: be(20),
            // `desc.n_region` — 12 bytes into the `desc` block at 32.
            n_region: be(44),
        };
        let count = header.n_region as usize;
        if count > MAX_REGIONS {
            return Err(FwError::TooManyRegions);
        }
        let mut sections = [PatchSection::default(); MAX_REGIONS];
        for (i, slot) in sections.iter_mut().take(count).enumerate() {
            let at = PATCH_HDR_BYTES + i * PATCH_SEC_BYTES;
            if image.len() < at + PATCH_SEC_BYTES {
                return Err(FwError::TooShort);
            }
            let ty = be(at);
            if ty & PATCH_SEC_TYPE_MASK != PATCH_SEC_TYPE_INFO {
                return Err(FwError::BadSectionType);
            }
            let sec = PatchSection {
                offs: be(at + 4),
                size: be(at + 8),
                addr: be(at + 12),
                len: be(at + 16),
                sec_key_idx: be(at + 20),
            };
            // Both lengths are checked, though only `len` is downloaded:
            // upstream's `patch_check` validates `info.len`
            // (`mt76_connac_mcu.c:3329`) and never looks at `size`, so a
            // disagreement between them is a shape this parser has not seen
            // and should not quietly pick a winner for.
            let reach = |n: u32| (sec.offs as usize).checked_add(n as usize);
            if reach(sec.len).is_none_or(|e| e > image.len())
                || reach(sec.size).is_none_or(|e| e > image.len())
            {
                return Err(FwError::RegionOutOfRange);
            }
            *slot = sec;
        }
        Ok(Self {
            header,
            sections,
            count,
        })
    }
}

// ── the RAM code: a trailer, little-endian ───────────────────────────────

/// Bytes of [`RamTrailer`] on the wire.
pub const RAM_TRAILER_BYTES: usize = 36;
/// Bytes of one [`RamRegion`] on the wire.
pub const RAM_REGION_BYTES: usize = 40;

// ── the region feature-set byte, and the region type ────────────────────

/// `FW_FEATURE_SET_ENCRYPT` — `mt76_connac_mcu.h:11`.
pub const FW_FEATURE_SET_ENCRYPT: u8 = 1 << 0;
/// `FW_FEATURE_SET_KEY_IDX` — `mt76_connac_mcu.h:12`, `GENMASK(2, 1)`.
pub const FW_FEATURE_SET_KEY_IDX: u8 = 0b110;
/// `FW_FEATURE_ENCRY_MODE` — `mt76_connac_mcu.h:13`.
pub const FW_FEATURE_ENCRY_MODE: u8 = 1 << 4;
/// `FW_FEATURE_OVERRIDE_ADDR` — `mt76_connac_mcu.h:14`. The region whose
/// address becomes `FW_START_REQ`'s entry point.
pub const FW_FEATURE_OVERRIDE_ADDR: u8 = 1 << 5;
/// `FW_FEATURE_NON_DL` — `mt76_connac_mcu.h:15`. **The bit that makes the
/// file bigger than the transfer**: a region carrying it is skipped by
/// `mt76_connac_mcu_send_ram_firmware` (`mt76_connac_mcu.c:3101`) while the
/// image cursor still advances past its payload.
pub const FW_FEATURE_NON_DL: u8 = 1 << 6;

/// `FW_TYPE_DEFAULT` — `mt76_connac_mcu.h:47`.
pub const FW_TYPE_DEFAULT: u8 = 0;
/// `FW_TYPE_CLC` — `mt76_connac_mcu.h:48`. Country-location-control data,
/// which `mt7925_load_clc` reads out of the image itself rather than
/// downloading (`mt7925/mcu.c:941-953`). On the distribution's blob this is
/// one region of **303,936 bytes** — 21.8% of the file, and none of the
/// transfer.
pub const FW_TYPE_CLC: u8 = 2;

/// One downloadable piece of the RAM code.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct RamRegion {
    /// Where it lands in the part's address space.
    pub addr: u32,
    /// Payload length.
    pub len: u32,
    /// Feature-set byte, which selects whether this region is wanted.
    pub feature_set: u8,
    /// Region type.
    pub kind: u8,
    /// Where this region's payload starts in the image.
    ///
    /// **Not a field of `mt76_connac2_fw_region`** — it is the running sum
    /// of every preceding region's `len`, which upstream recomputes inside
    /// the download loop (`offset += len` at `mt76_connac_mcu.c:3121`,
    /// reached by `goto next` so that a skipped region still advances it).
    /// Recording it at parse time is what lets the download loop be a loop
    /// over *regions* rather than a loop that also does arithmetic it could
    /// get wrong.
    pub offset: u32,
}

impl RamRegion {
    /// Whether this region's bytes cross into the device at all.
    ///
    /// `FW_FEATURE_NON_DL` is the one bit that decides it. A region that
    /// answers `false` still occupies its extent in the image — see
    /// [`RamRegion::offset`].
    pub const fn is_downloaded(self) -> bool {
        self.feature_set & FW_FEATURE_NON_DL == 0
    }

    /// Whether this region is the country-location-control blob.
    pub const fn is_clc(self) -> bool {
        !self.is_downloaded() && self.kind == FW_TYPE_CLC
    }

    /// Whether this region's address is the firmware entry point.
    pub const fn is_override(self) -> bool {
        self.feature_set & FW_FEATURE_OVERRIDE_ADDR != 0
    }

    /// The `mode` word `init_download` takes for this region.
    ///
    /// `mt76_connac_mcu_gen_dl_mode` (`mt76_connac_mcu.h:1927-1944`) for a
    /// connac2 part with `is_wa == false` — this part ships no WA image, so
    /// `DL_MODE_WORKING_PDA_CR4` is unreachable and is deliberately not a
    /// parameter here. All four downloaded regions of the distribution's
    /// blob carry `FW_FEATURE_SET_ENCRYPT` with key index 0, so all four ask
    /// for the same `NEED_RSP | ENCRYPT | RESET_SEC_IV`.
    pub const fn dl_mode(self) -> u32 {
        let mut mode = DL_MODE_NEED_RSP;
        if self.feature_set & FW_FEATURE_SET_ENCRYPT != 0 {
            mode |= DL_MODE_ENCRYPT | DL_MODE_RESET_SEC_IV;
        }
        if self.feature_set & FW_FEATURE_ENCRY_MODE != 0 {
            mode |= DL_CONFIG_ENCRY_MODE_SEL;
        }
        let key = ((self.feature_set & FW_FEATURE_SET_KEY_IDX)
            >> FW_FEATURE_SET_KEY_IDX.trailing_zeros()) as u32;
        mode | ((key << DL_MODE_KEY_IDX.trailing_zeros()) & DL_MODE_KEY_IDX)
    }
}

/// The RAM code's trailer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RamTrailer {
    /// The chip this image is for (`0x18` on the distribution's MT7925
    /// image).
    pub chip_id: u8,
    /// ECO revision.
    pub eco_code: u8,
    /// How many regions precede the trailer.
    pub n_region: u8,
    /// Image format version.
    pub format_ver: u8,
    /// Format flags.
    pub format_flag: u8,
    /// `fw_ver`, verbatim — 10 bytes. The distribution's image carries
    /// `____000000`, four literal underscores, which `mt7925e` prints as-is
    /// (`res/MT7925_EVIDENCE.md` §11). Not a parse failure: the field is a
    /// placeholder in this build.
    pub fw_ver: [u8; 10],
    /// `build_date`, verbatim — 15 bytes.
    pub build_date: [u8; 15],
}

/// A parsed RAM code image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RamCode {
    /// The trailer.
    pub trailer: RamTrailer,
    regions: [RamRegion; MAX_REGIONS],
    count: usize,
    slack: u32,
}

impl RamCode {
    /// The regions, in file order.
    pub fn regions(&self) -> &[RamRegion] {
        &self.regions[..self.count]
    }

    /// Total payload bytes across every region — **including** the ones the
    /// download skips. This is the image cursor's final value, not the
    /// transfer's size; [`RamCode::download_bytes`] is the latter.
    pub fn payload_bytes(&self) -> u64 {
        self.regions().iter().map(|r| u64::from(r.len)).sum()
    }

    /// Bytes this blob sends to the part — every region but the
    /// `FW_FEATURE_NON_DL` ones.
    ///
    /// On the distribution's image this is **1,087,744** against a
    /// `payload_bytes` of 1,391,680 and a file of 1,392,088: the CLC region
    /// is 303,936 bytes that are read out of the image by the host and never
    /// handed to the part.
    pub fn download_bytes(&self) -> u64 {
        self.regions()
            .iter()
            .filter(|r| r.is_downloaded())
            .map(|r| u64::from(r.len))
            .sum()
    }

    /// Bytes of the image that lie between the last region's payload and the
    /// region table — claimed by nothing the download reads.
    ///
    /// **172 on the distribution's blob**, and the reason this accessor
    /// exists rather than the parser simply tolerating the gap: that number
    /// was an unexplained discrepancy until `res/MT7925_EVIDENCE.md` §14
    /// dumped it, and the way to keep it explained is to keep it reportable.
    /// Upstream's own `mt76_connac2_fw_check` admits any value here
    /// (`mt76_connac_mcu.c:3064-3073` compares the running sum with `<=`),
    /// so a non-zero answer is legal — it is simply not *silent*.
    pub const fn slack_bytes(&self) -> u32 {
        self.slack
    }

    /// The entry point `FW_START_REQ` is given, if any region claims one.
    ///
    /// `mt76_connac_mcu_send_ram_firmware` keeps the **last**
    /// `FW_FEATURE_OVERRIDE_ADDR` region's address and passes it with
    /// `FW_START_OVERRIDE` (`mt76_connac_mcu.c:3104-3105,3124-3126`); with
    /// no such region the start command takes address 0 and no option. The
    /// distribution's image overrides to `0x90d000`.
    pub fn override_addr(&self) -> Option<u32> {
        self.regions()
            .iter()
            .filter(|r| r.is_downloaded() && r.is_override())
            .map(|r| r.addr)
            .next_back()
    }

    /// Read a RAM code image.
    ///
    /// The trailer is at the **end**, and the region table immediately
    /// precedes it, so the image is read backwards from its own length.
    pub fn parse(image: &[u8]) -> Result<Self, FwError> {
        if image.len() < RAM_TRAILER_BYTES {
            return Err(FwError::TooShort);
        }
        let t = image.len() - RAM_TRAILER_BYTES;
        let mut fw_ver = [0u8; 10];
        fw_ver.copy_from_slice(&image[t + 7..t + 17]);
        let mut build_date = [0u8; 15];
        build_date.copy_from_slice(&image[t + 17..t + 32]);
        let trailer = RamTrailer {
            chip_id: image[t],
            eco_code: image[t + 1],
            n_region: image[t + 2],
            format_ver: image[t + 3],
            format_flag: image[t + 4],
            fw_ver,
            build_date,
        };
        let count = trailer.n_region as usize;
        if count > MAX_REGIONS {
            return Err(FwError::TooManyRegions);
        }
        let table = count * RAM_REGION_BYTES;
        let Some(base) = t.checked_sub(table) else {
            return Err(FwError::TooShort);
        };
        let le = |at: usize| -> u32 {
            u32::from_le_bytes([image[at], image[at + 1], image[at + 2], image[at + 3]])
        };
        let mut regions = [RamRegion::default(); MAX_REGIONS];
        let mut consumed = 0usize;
        for (i, slot) in regions.iter_mut().take(count).enumerate() {
            let at = base + i * RAM_REGION_BYTES;
            let r = RamRegion {
                addr: le(at + 16),
                len: le(at + 20),
                feature_set: image[at + 24],
                kind: image[at + 25],
                // The cursor *before* this region — upstream's `offset`
                // entering the loop body, which is what `data + offset`
                // points at when the region is sent.
                offset: consumed as u32,
            };
            consumed = consumed.saturating_add(r.len as usize);
            // Checked inside the loop rather than after it: `offset` is only
            // meaningful while the running sum is still inside the image,
            // and a `u32` cursor over a `usize` sum must not be allowed to
            // wrap before the bound is tested.
            if consumed > base {
                return Err(FwError::RegionOutOfRange);
            }
            *slot = r;
        }
        Ok(Self {
            trailer,
            regions,
            count,
            // Everything between the last payload and the region table.
            // `consumed <= base` is established above, so this cannot
            // underflow; it is a `u32` because the image itself is.
            slack: (base - consumed) as u32,
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    /// **The pin reads the manifest**, and reads a different entry for each
    /// blob.
    ///
    /// A const parser that silently found the first entry twice would give
    /// both blobs the patch's size and digest, and the RAM code would then
    /// fail to verify on hardware for a reason no refusal could name. The
    /// sizes are also asserted against the figures `manifest.toml` records
    /// in prose, so a re-pin that edits one and not the other is caught
    /// here rather than on the machine.
    #[test]
    fn both_blobs_are_pinned_from_their_own_entry() {
        assert_eq!(super::PATCH_PINNED_SIZE, 188_192);
        assert_eq!(super::RAM_PINNED_SIZE, 1_392_088);
        assert_ne!(super::PATCH_PINNED_DIGEST, super::RAM_PINNED_DIGEST);
        // The first bytes of each, from the manifest's own hex.
        assert_eq!(super::PATCH_PINNED_DIGEST[0], 0x9a);
        assert_eq!(super::RAM_PINNED_DIGEST[0], 0x22);
    }

    /// A module that is not the pinned blob is refused, and the two ways it
    /// can fail are told apart: too few bytes is a provisioning fault, the
    /// right number of wrong bytes is a trust one.
    #[test]
    fn an_unpinned_module_is_refused_by_which_way_it_failed() {
        let short = [0u8; 8];
        assert_eq!(
            super::pinned(&short, super::PATCH_PINNED_SIZE, &super::PATCH_PINNED_DIGEST),
            Err(super::FwError::Truncated)
        );
        let wrong = alloc::vec![0u8; super::PATCH_PINNED_SIZE];
        assert_eq!(
            super::pinned(&wrong, super::PATCH_PINNED_SIZE, &super::PATCH_PINNED_DIGEST),
            Err(super::FwError::DigestMismatch)
        );
    }

    use super::*;

    /// The ROM patch's own values, read off the distribution image
    /// (`linux-firmware 20260910`) and cross-checked against what `mt7925e`
    /// printed on the lab's machine: `HW/SW Version: 0x8a108a10, Build Time:
    /// 20260813113015a` (`res/MT7925_EVIDENCE.md` §11).
    const REAL_HW_SW_VER: u32 = 0x8a10_8a10;
    const REAL_PATCH_BUILD: &[u8; 16] = b"20260813113015a\n";
    const REAL_PATCH_REGIONS: u32 = 2;
    /// `info.sec_key_idx` on both of the distribution's sections: encryption
    /// type `0x01` (AES) in `GENMASK(31, 24)`, key index 0.
    const REAL_PATCH_SEC_KEY: u32 = 0x0100_0000;
    /// …and the RAM code's.
    const REAL_CHIP_ID: u8 = 0x18;
    const REAL_RAM_REGIONS: u8 = 5;
    const REAL_FW_VER: &[u8; 10] = b"____000000";

    /// A patch image with `n` sections, each `size` bytes of payload.
    fn patch_image(buf: &mut [u8], n: u32, size: u32, ty: u32) -> usize {
        buf[0..16].copy_from_slice(REAL_PATCH_BUILD);
        buf[16..20].copy_from_slice(b"ALPS");
        buf[20..24].copy_from_slice(&REAL_HW_SW_VER.to_be_bytes());
        buf[44..48].copy_from_slice(&n.to_be_bytes());
        let body = PATCH_HDR_BYTES + (n as usize) * PATCH_SEC_BYTES;
        for i in 0..n as usize {
            let at = PATCH_HDR_BYTES + i * PATCH_SEC_BYTES;
            buf[at..at + 4].copy_from_slice(&ty.to_be_bytes());
            let offs = (body + i * size as usize) as u32;
            buf[at + 4..at + 8].copy_from_slice(&offs.to_be_bytes());
            buf[at + 8..at + 12].copy_from_slice(&size.to_be_bytes());
            buf[at + 12..at + 16].copy_from_slice(&0x0090_0000u32.to_be_bytes());
            // `info.len` — the field the download actually reads; the
            // distribution's image carries it equal to `size`.
            buf[at + 16..at + 20].copy_from_slice(&size.to_be_bytes());
            // `info.sec_key_idx` — AES, key index 0, which is what both of
            // the distribution's sections carry.
            buf[at + 20..at + 24].copy_from_slice(&REAL_PATCH_SEC_KEY.to_be_bytes());
        }
        body + (n as usize) * size as usize
    }

    /// The header reads back what the silicon's own driver printed.
    #[test]
    fn a_patch_header_reads_the_distributions_values() {
        let mut buf = [0u8; 512];
        let len = patch_image(&mut buf, REAL_PATCH_REGIONS, 32, 0x0003_0002);
        let p = Patch::parse(&buf[..len]).expect("the distribution's shape");
        assert_eq!(p.header.hw_sw_ver, REAL_HW_SW_VER, "what mt7925e printed");
        assert_eq!(&p.header.build_date, REAL_PATCH_BUILD);
        assert_eq!(&p.header.platform, b"ALPS");
        assert_eq!(p.header.n_region, REAL_PATCH_REGIONS);
        assert_eq!(p.sections().len(), 2);
        assert_eq!(p.sections()[0].addr, 0x0090_0000);
    }

    /// A section claiming payload the image does not hold is refused here,
    /// not discovered by the loop that would have copied it.
    #[test]
    fn a_patch_section_past_the_image_is_refused() {
        let mut buf = [0u8; 512];
        let len = patch_image(&mut buf, 1, 32, 0x0003_0002);
        // Enlarge the section past the image's end — `info.len`, which is
        // the field the download reads.
        let at = PATCH_HDR_BYTES + 16;
        buf[at..at + 4].copy_from_slice(&9999u32.to_be_bytes());
        assert_eq!(
            Patch::parse(&buf[..len]),
            Err(FwError::RegionOutOfRange)
        );
    }

    /// Only the informational section type is accepted, as upstream's loader
    /// requires.
    #[test]
    fn a_patch_section_of_another_type_is_refused() {
        let mut buf = [0u8; 512];
        let len = patch_image(&mut buf, 1, 32, 0x0003_0007);
        assert_eq!(Patch::parse(&buf[..len]), Err(FwError::BadSectionType));
    }

    /// `n_region` is a number read out of a file; it does not get to decide
    /// how long this driver loops.
    #[test]
    fn an_absurd_region_count_is_refused() {
        let mut buf = [0u8; 512];
        patch_image(&mut buf, 1, 32, 0x0003_0002);
        buf[44..48].copy_from_slice(&9999u32.to_be_bytes());
        assert_eq!(Patch::parse(&buf), Err(FwError::TooManyRegions));
        let mut ram = [0u8; 256];
        ram[256 - RAM_TRAILER_BYTES + 2] = 200;
        assert_eq!(RamCode::parse(&ram), Err(FwError::TooManyRegions));
    }

    /// A RAM image with `n` regions of `size` bytes each.
    fn ram_image(buf: &mut [u8], n: u8, size: u32) -> usize {
        ram_image_with(buf, n, size, 0, &[])
    }

    /// A RAM image with `n` regions of `size` bytes each, `slack` unclaimed
    /// bytes between the last payload and the region table, and per-region
    /// `(feature_set, type)` overrides for the leading `over.len()` regions.
    pub(crate) fn ram_image_with(
        buf: &mut [u8],
        n: u8,
        size: u32,
        slack: usize,
        over: &[(u8, u8)],
    ) -> usize {
        let payload = (n as usize) * size as usize + slack;
        let table = (n as usize) * RAM_REGION_BYTES;
        let len = payload + table + RAM_TRAILER_BYTES;
        for i in 0..n as usize {
            let at = payload + i * RAM_REGION_BYTES;
            buf[at + 16..at + 20].copy_from_slice(&0x0090_d000u32.to_le_bytes());
            buf[at + 20..at + 24].copy_from_slice(&size.to_le_bytes());
            let (feature, kind) = over.get(i).copied().unwrap_or((0x21, 0));
            buf[at + 24] = feature;
            buf[at + 25] = kind;
        }
        let t = payload + table;
        buf[t] = REAL_CHIP_ID;
        buf[t + 2] = n;
        buf[t + 3] = 2;
        buf[t + 4] = 1;
        buf[t + 7..t + 17].copy_from_slice(REAL_FW_VER);
        buf[t + 17..t + 32].copy_from_slice(b"20260813113118\0");
        len
    }

    /// The trailer reads back what the silicon's own driver printed —
    /// underscores and all.
    #[test]
    fn a_ram_trailer_reads_the_distributions_values() {
        let mut buf = [0u8; 1024];
        let len = ram_image(&mut buf, REAL_RAM_REGIONS, 64);
        let r = RamCode::parse(&buf[..len]).expect("the distribution's shape");
        assert_eq!(r.trailer.chip_id, REAL_CHIP_ID);
        assert_eq!(r.trailer.n_region, REAL_RAM_REGIONS);
        assert_eq!(
            &r.trailer.fw_ver, REAL_FW_VER,
            "a placeholder field, not a parse failure"
        );
        assert_eq!(&r.trailer.build_date[..14], b"20260813113118");
        assert_eq!(r.regions().len(), 5);
        assert_eq!(r.payload_bytes(), 5 * 64);
    }

    /// Regions cannot together claim more than the image holds ahead of
    /// their own table.
    #[test]
    fn ram_regions_cannot_claim_more_than_the_image() {
        let mut buf = [0u8; 1024];
        let len = ram_image(&mut buf, 2, 64);
        // Re-stamp the first region's length past the payload area.
        let payload = 2 * 64;
        buf[payload + 20..payload + 24].copy_from_slice(&9999u32.to_le_bytes());
        assert_eq!(
            RamCode::parse(&buf[..len]),
            Err(FwError::RegionOutOfRange)
        );
    }

    /// Both formats refuse a truncated image rather than reading past it.
    #[test]
    fn a_truncated_image_is_refused() {
        assert_eq!(Patch::parse(&[0u8; 4]), Err(FwError::TooShort));
        assert_eq!(RamCode::parse(&[0u8; 4]), Err(FwError::TooShort));
    }

    // ── the distribution's own geometry ──────────────────────────────────
    //
    // The five rows below are the region table of
    // `WIFI_RAM_CODE_MT7925_1_1.bin` as it ships in `linux-firmware
    // 20260910` (sha256 `23ff53b4…`, `res/MT7925_EVIDENCE.md` §5′), read
    // field by field out of the file. An image built from them is not the
    // real firmware — the payloads are zeroes — but it has the real
    // *shape*, which is the thing this parser can be wrong about.

    /// `(addr, len, feature_set, type)` for each of the five regions.
    const REAL_RAM_TABLE: [(u32, u32, u8, u8); 5] = [
        (0x0090_d000, 77_200, 0x21, 0),
        (0x0221_2800, 382_928, 0x01, 0),
        (0x0040_4000, 32_720, 0x01, 0),
        (0xe002_6c00, 594_896, 0x01, 0),
        // The CLC blob: NON_DL, and `type == FW_TYPE_CLC`.
        (0x0000_0000, 303_936, 0x40, 2),
    ];
    /// The file's own length.
    const REAL_RAM_FILE_BYTES: usize = 1_392_088;
    /// The gap between the last payload and the region table.
    const REAL_RAM_SLACK: u32 = 172;

    /// Build an image with the distribution's exact geometry.
    fn real_ram_image() -> alloc::vec::Vec<u8> {
        let payload: usize = REAL_RAM_TABLE.iter().map(|r| r.1 as usize).sum();
        let table = REAL_RAM_TABLE.len() * RAM_REGION_BYTES;
        let mut buf =
            alloc::vec![0u8; payload + REAL_RAM_SLACK as usize + table + RAM_TRAILER_BYTES];
        // The fence the real image carries across the slack, so a reader of
        // this test sees what §14 found rather than a run of zeroes.
        let gap = payload;
        buf[gap..gap + 16].fill(b'#');
        let fence_end = gap + REAL_RAM_SLACK as usize;
        buf[fence_end - 16..fence_end].fill(b'#');
        let base = gap + REAL_RAM_SLACK as usize;
        for (i, &(addr, len, feature, kind)) in REAL_RAM_TABLE.iter().enumerate() {
            let at = base + i * RAM_REGION_BYTES;
            buf[at + 16..at + 20].copy_from_slice(&addr.to_le_bytes());
            buf[at + 20..at + 24].copy_from_slice(&len.to_le_bytes());
            buf[at + 24] = feature;
            buf[at + 25] = kind;
        }
        let t = base + table;
        buf[t] = REAL_CHIP_ID;
        buf[t + 2] = REAL_RAM_TABLE.len() as u8;
        buf[t + 3] = 2;
        buf[t + 4] = 1;
        buf[t + 7..t + 17].copy_from_slice(REAL_FW_VER);
        buf[t + 17..t + 32].copy_from_slice(b"20260813113118\0");
        buf
    }

    /// **The 172 bytes, accounted for.**
    ///
    /// The file is 1,392,088 bytes; the five regions claim 1,391,680; the
    /// table and trailer take 236 more. The remaining 172 are the build-info
    /// block `res/MT7925_EVIDENCE.md` §14 dumps, and this test is what keeps
    /// the number from drifting back into being a mystery: the geometry adds
    /// up exactly, and the slack is reported rather than absorbed.
    #[test]
    fn the_distributions_image_accounts_for_every_byte() {
        let image = real_ram_image();
        assert_eq!(image.len(), REAL_RAM_FILE_BYTES, "the distribution's size");
        let r = RamCode::parse(&image).expect("the distribution's shape");
        assert_eq!(r.regions().len(), 5);
        assert_eq!(r.payload_bytes(), 1_391_680);
        assert_eq!(r.slack_bytes(), REAL_RAM_SLACK);
        // Sum to the file, with nothing left over and nothing double-counted.
        let table = (r.regions().len() * RAM_REGION_BYTES) as u64;
        assert_eq!(
            r.payload_bytes() + u64::from(r.slack_bytes()) + table + RAM_TRAILER_BYTES as u64,
            REAL_RAM_FILE_BYTES as u64,
        );
    }

    /// **The transfer is smaller than the file**, and by exactly the CLC
    /// region.
    #[test]
    fn the_clc_region_is_read_but_never_downloaded() {
        let image = real_ram_image();
        let r = RamCode::parse(&image).expect("the distribution's shape");
        let clc = r.regions()[4];
        assert!(!clc.is_downloaded(), "FW_FEATURE_NON_DL");
        assert!(clc.is_clc(), "FW_TYPE_CLC");
        assert_eq!(clc.len, 303_936);
        assert_eq!(r.download_bytes(), 1_087_744);
        assert_eq!(
            r.payload_bytes() - r.download_bytes(),
            u64::from(clc.len),
            "the CLC region is the whole of the difference"
        );
        // …and its payload still occupies its place in the image, which is
        // what makes the following regions' offsets right.
        assert_eq!(clc.offset, 1_087_744);
    }

    /// Each region's payload starts where every earlier region's ended,
    /// skipped ones included.
    #[test]
    fn region_offsets_are_the_running_sum() {
        let image = real_ram_image();
        let r = RamCode::parse(&image).expect("the distribution's shape");
        let mut want = 0u32;
        for region in r.regions() {
            assert_eq!(region.offset, want, "region at {:#x}", region.addr);
            want += region.len;
        }
        assert_eq!(u64::from(want), r.payload_bytes());
    }

    /// The entry point is the overriding region's address, and only a
    /// downloaded region may supply it.
    #[test]
    fn the_entry_point_comes_from_the_overriding_region() {
        let image = real_ram_image();
        let r = RamCode::parse(&image).expect("the distribution's shape");
        assert_eq!(r.override_addr(), Some(0x0090_d000));
        // No region claiming the bit means no override and no option word —
        // `FW_START_REQ` then takes address 0.
        let mut plain = [0u8; 1024];
        let len = ram_image_with(&mut plain, 2, 64, 0, &[(0x01, 0), (0x01, 0)]);
        let p = RamCode::parse(&plain[..len]).expect("shape");
        assert_eq!(p.override_addr(), None);
    }

    /// The mode word each region asks for, composed from its feature byte.
    #[test]
    fn the_download_mode_is_composed_from_the_feature_byte() {
        let image = real_ram_image();
        let r = RamCode::parse(&image).expect("the distribution's shape");
        // All four downloaded regions are AES-encrypted with key index 0.
        for region in r.regions().iter().filter(|r| r.is_downloaded()) {
            assert_eq!(
                region.dl_mode(),
                DL_MODE_NEED_RSP | DL_MODE_ENCRYPT | DL_MODE_RESET_SEC_IV,
                "region at {:#x}",
                region.addr
            );
        }
        // A plain region asks only for the response.
        let plain = RamRegion {
            feature_set: 0,
            ..RamRegion::default()
        };
        assert_eq!(plain.dl_mode(), DL_MODE_NEED_RSP);
        // The key index rides `GENMASK(2, 1)` into `DL_MODE_KEY_IDX`, which
        // is the same two bit positions — a coincidence worth pinning,
        // because the composition is a shift in both directions.
        let keyed = RamRegion {
            feature_set: FW_FEATURE_SET_ENCRYPT | (0b11 << 1),
            ..RamRegion::default()
        };
        assert_eq!(keyed.dl_mode() & DL_MODE_KEY_IDX, 0b11 << 1);
        // And the alternate-cipher bit selects the other mode.
        let scrambled = RamRegion {
            feature_set: FW_FEATURE_SET_ENCRYPT | FW_FEATURE_ENCRY_MODE,
            ..RamRegion::default()
        };
        assert_ne!(scrambled.dl_mode() & DL_CONFIG_ENCRY_MODE_SEL, 0);
    }

    /// Slack is admitted — upstream's check is a `<=` — but it is never
    /// confused with payload.
    #[test]
    fn slack_between_the_payloads_and_the_table_is_admitted_and_reported() {
        let mut buf = [0u8; 1024];
        let len = ram_image_with(&mut buf, 2, 64, 172, &[]);
        let r = RamCode::parse(&buf[..len]).expect("slack is legal");
        assert_eq!(r.slack_bytes(), 172);
        assert_eq!(r.payload_bytes(), 128);
        // A blob with no gap reports none, so a non-zero answer is a fact
        // about the image rather than a constant this parser carries.
        let len = ram_image_with(&mut buf, 2, 64, 0, &[]);
        let r = RamCode::parse(&buf[..len]).expect("no slack is legal too");
        assert_eq!(r.slack_bytes(), 0);
    }

    /// The patch's two length fields are both read, and both bound the
    /// extent check.
    #[test]
    fn a_patch_section_carries_its_own_length_and_its_key_index() {
        let mut buf = [0u8; 512];
        let len = patch_image(&mut buf, REAL_PATCH_REGIONS, 32, 0x0003_0002);
        let p = Patch::parse(&buf[..len]).expect("the distribution's shape");
        let sec = p.sections()[0];
        assert_eq!(sec.len, 32, "info.len — what the download sends");
        assert_eq!(sec.size, 32, "sec.size — the record's own field");
        assert_eq!(sec.sec_key_idx, REAL_PATCH_SEC_KEY);
        assert_eq!(
            sec.data_mode(),
            DL_MODE_NEED_RSP | DL_MODE_ENCRYPT | DL_MODE_RESET_SEC_IV,
            "AES with key index 0"
        );
        assert_eq!(p.download_bytes(), 64);
        // `size` is checked too, even though nothing downloads it: a record
        // whose two lengths disagree past the image's end is refused.
        let at = PATCH_HDR_BYTES + 8;
        buf[at..at + 4].copy_from_slice(&9999u32.to_be_bytes());
        assert_eq!(Patch::parse(&buf[..len]), Err(FwError::RegionOutOfRange));
    }

    /// A section that declares no encryption support takes the bare mode.
    #[test]
    fn an_unsupported_key_index_asks_only_for_a_response() {
        let sec = PatchSection {
            sec_key_idx: 0xffff_ffff,
            ..PatchSection::default()
        };
        assert_eq!(sec.data_mode(), DL_MODE_NEED_RSP);
    }
}
