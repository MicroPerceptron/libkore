//! **The firmware** — the pinned trust decision, the container the blob
//! is, and the download plan the loader executes.
//!
//! Three layers, each refusing before the next reads a byte:
//!
//! 1. [`FirmwareImage::verified`] — the **only** constructor: the blob's
//!    BLAKE3 digest against [`PINNED_DIGEST`], the compiled citation of
//!    `firmware/manifest.toml`. Holding an image proves the bytes are the
//!    pinned firmware; nothing below parses an unverified byte.
//! 2. [`FirmwareImage::suit`] — the multi-firmware (MFW) wrapper
//!    (`rtw89_mfw_recognize`, fw.c:592-655): the `NORMAL` image whose cut
//!    is the highest at or below the chip's, or the whole blob when the
//!    wrapper is absent (legacy).
//! 3. [`Suit::plan`] — the inner header, version 0 or 1 by the header's
//!    own word (`rtw89_fw_hdr_parser`, fw.c:536-552), its section table
//!    and the security section's key pool, into the exact byte ranges the
//!    download pushes and the header packet the firmware is primed with.
//!
//! The dossier's open question 2 — whether the 8922A's file carries a
//! version 0 or version 1 inner header — is answered at runtime the way
//! upstream answers it: both are implemented, the file decides.

use crate::regs::be::*;
use crate::regs::{get, replace};

mod elements;
pub mod phy;
pub use elements::{Element, Elements};

/// A 256-bit BLAKE3 digest pinning a known-good firmware blob.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Digest(pub [u8; 32]);

/// The trust root, consumed rather than re-typed: the manifest is parsed
/// at compile time, so the size and digest exist in exactly one place and
/// a re-pin is one edit reviewed as one diff.
mod manifest {
    const TEXT: &str = include_str!("../firmware/manifest.toml");
    const ENTRY: &str = "[firmware.rtw8922a_fw]";

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

    /// The byte index just past `key   = ` on the first line of the entry
    /// that starts with `key`.
    const fn value_of(key: &[u8]) -> usize {
        let text = TEXT.as_bytes();
        let Some(table) = find(text, ENTRY.as_bytes(), 0) else {
            panic!("manifest.toml: no [firmware.rtw8922a_fw] entry");
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
    pub const fn size() -> usize {
        let text = TEXT.as_bytes();
        let mut i = value_of(b"size");
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
    pub const fn digest() -> [u8; 32] {
        let text = TEXT.as_bytes();
        let i = value_of(b"blake3");
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

/// The pinned blob's exact byte length (`firmware/manifest.toml`,
/// `[firmware.rtw8922a_fw] size`). The mapped boot module is page-rounded;
/// the ladder slices the blob to this length before hashing.
pub const PINNED_SIZE: usize = manifest::size();
/// …and its BLAKE3 digest (`blake3` in the same entry).
pub const PINNED_DIGEST: Digest = Digest(manifest::digest());

/// Why the firmware was refused. Every variant reaches the row as the one
/// bad-firmware rung; the variant is for the host tests and the log.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FwError {
    /// Not the pinned blob.
    DigestMismatch,
    /// A structure the container claims runs past its end.
    Truncated,
    /// The MFW wrapper lists no image.
    NoEntries,
    /// No `NORMAL` image at or below the chip's cut.
    NoImageForCut,
    /// The inner header's version word is neither 0 nor 1.
    BadHeaderVersion,
    /// More sections than `FWDL_SECTION_MAX_NUM`.
    TooManySections,
    /// The dynamic header's own length disagrees with the header's.
    BadDynamicHeader,
    /// The sections do not end exactly at the image's end
    /// (`fw_end != bin`, the hard error at fw.c:217 / fw.c:517).
    BadSize,
    /// The formatted MSS key pool is malformed (signature, remap-table
    /// format, or its raw-key offset).
    BadMssPool,
    /// Secure boot selects a key index the section does not carry.
    BadMssIndex,
}

/// The firmware's version words (`rtw89_fw_update_ver`: w1 and w2, the
/// same layout in both header versions — fw.h:635-639 / fw.h:588-592).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Version {
    /// `w1[7:0]`.
    pub major: u8,
    /// `w1[15:8]`.
    pub minor: u8,
    /// `w1[23:16]`.
    pub sub: u8,
    /// `w1[31:24]`.
    pub idx: u8,
    /// `w2`.
    pub commit_id: u32,
}

/// What the efuse says about secure boot, as the section parser needs it
/// (`struct rtw89_fw_secure`, core.h:4691-4700). Read by
/// [`crate::efuse::read_secure_boot`]; [`Self::OFF`] is a part whose
/// `sb_cryp_sel` word is unprogrammed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SecureBoot {
    /// Whether the WCPU verifies the image against a key.
    pub secure_boot: bool,
    /// The key-management selector the section must match.
    pub sb_sel_mgn: u32,
    /// `RTW89_FW_MSS_DEV_TYPE_*` (already the index form).
    pub mss_dev_type: u8,
    /// Customer index.
    pub mss_cust_idx: u8,
    /// Key number.
    pub mss_key_num: u8,
    /// The version-0 key index (unused on this part; kept for the exact
    /// upstream arithmetic).
    pub mss_idx: u8,
}

impl SecureBoot {
    /// Secure boot off: every security section is measured and skipped
    /// over, no key is copied.
    pub const OFF: Self = Self {
        secure_boot: false,
        sb_sel_mgn: 0,
        mss_dev_type: 0,
        mss_cust_idx: 0,
        mss_key_num: 0,
        mss_idx: 0,
    };
}

/// `RTW89_FW_NORMAL` (`enum rtw89_fw_type`, core.h:4581).
const FW_TYPE_NORMAL: u8 = 1;
/// `struct rtw89_mfw_hdr` without its entries: 16 bytes (fw.h:3915-3927).
const MFW_HDR_BYTES: usize = 16;
/// `struct rtw89_mfw_info`: 16 bytes (fw.h:3905-3913).
const MFW_INFO_BYTES: usize = 16;
/// `struct rtw89_fw_hdr` without sections: 8 dwords (fw.h:570-581).
const HDR_V0_BYTES: usize = 32;
/// `struct rtw89_fw_hdr_v1` without sections: 12 dwords (fw.h:619-633).
const HDR_V1_BYTES: usize = 48;
/// One section entry, both versions: 4 dwords.
const SECTION_BYTES: usize = 16;
/// `struct rtw89_fw_mss_pool_hdr` without its remap table: 32 bytes
/// (fw.h:661-674).
const MSS_POOL_HDR_BYTES: usize = 32;
/// `mss_signature[]` (fw.c:41): "MSSKPOOL".
const MSS_SIGNATURE: [u8; 8] = *b"MSSKPOOL";
/// Bytes a section's word-0 address keeps on a version-0 header
/// (fw.c:191, `& 0x1fffffff`).
const V0_DL_ADDR_MASK: u32 = 0x1FFF_FFFF;
/// The version-0 8852B override is not this part's (fw.c:203-204).
const _: () = ();

/// The most sections a header may declare (`FWDL_SECTION_MAX_NUM`).
pub const MAX_SECTIONS: usize = FWDL_SECTION_MAX_NUM as usize;

#[inline]
fn le32(bytes: &[u8], at: usize) -> Result<u32, FwError> {
    let b = bytes.get(at..at + 4).ok_or(FwError::Truncated)?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

#[inline]
fn le16(bytes: &[u8], at: usize) -> Result<u16, FwError> {
    let b = bytes.get(at..at + 2).ok_or(FwError::Truncated)?;
    Ok(u16::from_le_bytes([b[0], b[1]]))
}

/// The verified blob.
#[derive(Clone, Copy)]
pub struct FirmwareImage<'a> {
    blob: &'a [u8],
}

impl<'a> FirmwareImage<'a> {
    /// Verify `blob` against `expected` — the only constructor. The digest
    /// is public, so a plain compare is fine; anything that does not match
    /// is refused before the container is read.
    pub fn verified(blob: &'a [u8], expected: &Digest) -> Result<Self, FwError> {
        let actual = blake3::hash(blob);
        if actual.as_bytes() != &expected.0 {
            return Err(FwError::DigestMismatch);
        }
        Ok(Self { blob })
    }

    /// The `NORMAL` image for a chip of cut `cv`
    /// (`rtw89_mfw_recognize`): among the wrapper's `NORMAL` entries with
    /// `mp == 0`, the highest `cv` at or below the chip's. A blob without
    /// the `0xFF` signature byte is a legacy single image.
    pub fn suit(&self, cv: u8) -> Result<Suit<'a>, FwError> {
        let blob = self.blob;
        if blob.len() < MFW_HDR_BYTES || blob[0] != RTW89_MFW_SIG as u8 {
            return Suit::new(blob);
        }
        let fw_nr = blob[1] as usize;
        if fw_nr == 0 {
            return Err(FwError::NoEntries);
        }
        // `&mfw_hdr->info[fw_nr] > mfw + mfw_len` is the refusal; the table
        // may end exactly at the blob's end.
        if MFW_HDR_BYTES + fw_nr * MFW_INFO_BYTES > blob.len() {
            return Err(FwError::Truncated);
        }
        let mut best: Option<(u8, usize)> = None;
        for i in 0..fw_nr {
            let at = MFW_HDR_BYTES + i * MFW_INFO_BYTES;
            let (entry_cv, entry_type, mp) = (blob[at], blob[at + 1], blob[at + 2]);
            if entry_type != FW_TYPE_NORMAL || mp != 0 || entry_cv > cv {
                continue;
            }
            if best.is_none_or(|(best_cv, _)| best_cv < entry_cv) {
                best = Some((entry_cv, at));
            }
        }
        let (_, at) = best.ok_or(FwError::NoImageForCut)?;
        let shift = le32(blob, at + 4)? as usize;
        let size = le32(blob, at + 8)? as usize;
        let end = shift.checked_add(size).ok_or(FwError::Truncated)?;
        let data = blob.get(shift..end).ok_or(FwError::Truncated)?;
        Suit::new(data)
    }
}

/// One image inside the container: its bytes and the header version its
/// word 3 declares.
#[derive(Clone, Copy)]
pub struct Suit<'a> {
    data: &'a [u8],
    hdr_ver: u8,
    version: Version,
}

impl<'a> Suit<'a> {
    fn new(data: &'a [u8]) -> Result<Self, FwError> {
        if data.is_empty() {
            return Err(FwError::NoEntries);
        }
        let w1 = le32(data, 4)?;
        let w2 = le32(data, 8)?;
        let w3 = le32(data, 12)?;
        Ok(Self {
            data,
            hdr_ver: get(w3, FW_HDR_W3_HDR_VER) as u8,
            version: Version {
                major: get(w1, FW_HDR_W1_MAJOR_VERSION) as u8,
                minor: get(w1, FW_HDR_W1_MINOR_VERSION) as u8,
                sub: get(w1, FW_HDR_W1_SUBVERSION) as u8,
                idx: get(w1, FW_HDR_W1_SUBINDEX) as u8,
                commit_id: w2,
            },
        })
    }

    /// The inner header's version: 0 or 1.
    pub fn header_version(&self) -> u8 {
        self.hdr_ver
    }

    /// The image's version words.
    pub fn version(&self) -> Version {
        self.version
    }

    /// Bytes of the image.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the image is empty (never: `new` refuses one).
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Parse the header and section table into a download plan
    /// (`rtw89_fw_hdr_parser_v0` / `_v1`, fw.c:141-218 / 438-521).
    pub fn plan(&self, secure: &SecureBoot) -> Result<Plan<'a>, FwError> {
        let fw = self.data;
        let w6 = le32(fw, 24)?;
        let w7 = le32(fw, 28)?;
        let (base, hdr_len_word, hdr_len_mask, sec_num_mask, dyn_mask) = match self.hdr_ver {
            0 => (
                HDR_V0_BYTES,
                le32(fw, 12)?,
                FW_HDR_W3_LEN,
                FW_HDR_W6_SEC_NUM,
                FW_HDR_W7_DYN_HDR,
            ),
            1 => (
                HDR_V1_BYTES,
                le32(fw, 20)?,
                FW_HDR_V1_W5_HDR_SIZE,
                FW_HDR_V1_W6_SEC_NUM,
                FW_HDR_V1_W7_DYN_HDR,
            ),
            _ => return Err(FwError::BadHeaderVersion),
        };
        let section_num = get(w6, sec_num_mask) as usize;
        if section_num > MAX_SECTIONS {
            return Err(FwError::TooManySections);
        }
        // Only a version-1 header carries the DSP-checksum flag; version 0
        // never adds the per-key checksum.
        let dsp_checksum = self.hdr_ver == 1 && w6 & FW_HDR_V1_W6_DSP_CHKSUM != 0;
        let base_hdr_len = base + section_num * SECTION_BYTES;
        let hdr_len = if w7 & dyn_mask != 0 {
            let hdr_len = get(hdr_len_word, hdr_len_mask) as usize;
            let dynamic_len = hdr_len
                .checked_sub(base_hdr_len)
                .ok_or(FwError::BadDynamicHeader)?;
            // `struct rtw89_fw_dynhdr_hdr { __le32 hdr_len; … }` at the
            // base header's end must agree (fw.c:462 / fw.c:166).
            if le32(fw, base_hdr_len)? as usize != dynamic_len {
                return Err(FwError::BadDynamicHeader);
            }
            hdr_len
        } else {
            base_hdr_len
        };
        let header = fw.get(..base_hdr_len).ok_or(FwError::Truncated)?;
        if fw.len() < hdr_len {
            return Err(FwError::Truncated);
        }

        let mut sections = [Section::EMPTY; MAX_SECTIONS];
        let mut secure_section_exist = false;
        let mut bin = hdr_len;
        for (i, section) in sections.iter_mut().enumerate().take(section_num) {
            let at = base + i * SECTION_BYTES;
            let w0 = le32(fw, at)?;
            let w1 = le32(fw, at + 4)?;
            let w2 = le32(fw, at + 8)?;
            // The two versions' word-1 layouts are identical (fw.h:561-565
            // / 608-612); word 2's MSSC field is the whole word on
            // version 0 and the low byte on version 1.
            let kind = get(w1, FWSECTION_HDR_W1_SECTIONTYPE) as u8;
            let mut len = get(w1, FWSECTION_HDR_W1_SEC_SIZE) as usize;
            if w1 & FWSECTION_HDR_W1_CHECKSUM != 0 {
                len += FWDL_SECTION_CHKSUM_LEN as usize;
            }
            let dl_addr = if self.hdr_ver == 0 {
                w0 & V0_DL_ADDR_MASK
            } else {
                w0
            };
            let data = fw
                .get(bin..bin.checked_add(len).ok_or(FwError::Truncated)?)
                .ok_or(FwError::Truncated)?;
            *section = Section {
                dl_addr,
                data,
                redl: w1 & FWSECTION_HDR_W1_REDL != 0,
                ignore: false,
                key: None,
                len_override: 0,
            };
            let mssc_len = if kind == FWDL_SECURITY_SECTION_TYPE as u8 {
                let mssc = if self.hdr_ver == 0 {
                    get(w2, FWSECTION_HDR_W2_MSSC)
                } else {
                    get(w2, FWSECTION_HDR_V1_W2_MSSC)
                };
                parse_security_section(
                    fw,
                    bin,
                    section,
                    mssc,
                    dsp_checksum,
                    secure,
                    &mut secure_section_exist,
                )?
            } else {
                0
            };
            bin = bin
                .checked_add(len)
                .and_then(|b| b.checked_add(mssc_len))
                .ok_or(FwError::Truncated)?;
        }
        if bin != fw.len() {
            return Err(FwError::BadSize);
        }
        Ok(Plan {
            hdr_ver: self.hdr_ver,
            version: self.version,
            header,
            sections,
            section_num: section_num as u8,
            dsp_checksum,
        })
    }
}

/// `__parse_security_section` (fw.c:399-437): measure the MSSC blob that
/// trails the section (so the next section is found), and — under secure
/// boot — locate the key the WCPU will verify against. Returns the MSSC
/// length.
fn parse_security_section<'a>(
    fw: &'a [u8],
    content_at: usize,
    section: &mut Section<'a>,
    mssc: u32,
    dsp_checksum: bool,
    secure: &SecureBoot,
    secure_section_exist: &mut bool,
) -> Result<usize, FwError> {
    let len = section.data.len();
    if mssc & FORMATTED_MSSC_MASK == FORMATTED_MSSC {
        // `__parse_formatted_mssc` (fw.c:275-364).
        let hdr = content_at + len;
        let signature = fw
            .get(hdr..hdr + MSS_SIGNATURE.len())
            .ok_or(FwError::Truncated)?;
        if signature != MSS_SIGNATURE {
            return Err(FwError::BadMssPool);
        }
        let key_raw_offset = le32(fw, hdr + 12)? as usize;
        let defen = *fw.get(hdr + 16).ok_or(FwError::Truncated)?;
        let rmpfmt = *fw.get(hdr + 20).ok_or(FwError::Truncated)?;
        let mssdev_max = *fw.get(hdr + 21).ok_or(FwError::Truncated)? as usize;
        let keypair_num = le16(fw, hdr + 22)? as usize;
        let msscust_max = le16(fw, hdr + 24)? as usize;
        let msskey_num_max = le16(fw, hdr + 26)? as usize;
        if rmpfmt != MSS_POOL_RMP_TBL_BITMASK as u8 {
            return Err(FwError::BadMssPool);
        }
        let mut rmp_tbl_size = (msskey_num_max * msscust_max * mssdev_max) >> 3;
        if defen != 0 {
            rmp_tbl_size += FWDL_MSS_POOL_DEFKEYSETS_SIZE as usize;
        }
        if rmp_tbl_size + MSS_POOL_HDR_BYTES != key_raw_offset {
            return Err(FwError::BadMssPool);
        }
        // `section_content->key_sign_len.v`: `__le16` after 60 pad bytes.
        let mut key_sign_len = (le16(fw, content_at + 60)? >> 2) as usize;
        if key_sign_len == 0 {
            key_sign_len = FWDL_SECURITY_SIGLEN as usize;
        }
        if dsp_checksum {
            key_sign_len += FWDL_SECURITY_CHKSUM_LEN as usize;
        }
        let mssc_len = MSS_POOL_HDR_BYTES + rmp_tbl_size + keypair_num * key_sign_len;
        fw.get(hdr..hdr + mssc_len).ok_or(FwError::Truncated)?;

        if secure.secure_boot {
            // `section_content->sb_sel_ver.v`: `__le32` after 58 pad bytes.
            let sb_sel_ver = le32(fw, content_at + 58)?;
            if sb_sel_ver != 0 && sb_sel_ver != secure.sb_sel_mgn {
                section.ignore = true;
                return Ok(mssc_len);
            }
            let rmp_tbl = &fw[hdr + MSS_POOL_HDR_BYTES..hdr + MSS_POOL_HDR_BYTES + rmp_tbl_size];
            let Some(key_idx) =
                mssc_key_idx(secure, rmp_tbl, defen != 0, msskey_num_max, msscust_max)
            else {
                section.ignore = true;
                return Ok(mssc_len);
            };
            let key_at = hdr + key_raw_offset + key_sign_len * key_idx;
            section.key = Some(
                fw.get(key_at..key_at + key_sign_len)
                    .ok_or(FwError::Truncated)?,
            );
        }
        // `out:` — only the first security section is downloaded.
        if *secure_section_exist {
            section.ignore = true;
        } else {
            *secure_section_exist = true;
        }
        Ok(mssc_len)
    } else {
        let mut mssc_len = mssc as usize * FWDL_SECURITY_SIGLEN as usize;
        if dsp_checksum {
            mssc_len += mssc as usize * FWDL_SECURITY_CHKSUM_LEN as usize;
        }
        fw.get(content_at + len..content_at + len + mssc_len)
            .ok_or(FwError::Truncated)?;
        if secure.secure_boot {
            if secure.mss_idx as u32 >= mssc {
                return Err(FwError::BadMssIndex);
            }
            let key_at = content_at + len + secure.mss_idx as usize * FWDL_SECURITY_SIGLEN as usize;
            section.key = Some(
                fw.get(key_at..key_at + FWDL_SECURITY_SIGLEN as usize)
                    .ok_or(FwError::Truncated)?,
            );
        }
        *secure_section_exist = true;
        Ok(mssc_len)
    }
    // `__check_secure_blacklist` only warns (fw.c:433-434); it changes
    // nothing about what is downloaded and is not reproduced.
}

/// `__get_mssc_key_idx` (fw.c:232-273): the key this part selects out of
/// the pool's remap table, or `None` where upstream ignores the section.
fn mssc_key_idx(
    secure: &SecureBoot,
    rmp_tbl: &[u8],
    defen: bool,
    msskey_num_max: usize,
    msscust_max: usize,
) -> Option<usize> {
    let dev = secure.mss_dev_type as usize;
    let cust = secure.mss_cust_idx as usize;
    let key = secure.mss_key_num as usize;
    let mss_sel_idx = if secure.mss_dev_type == RTW89_FW_MSS_DEV_TYPE_FWSEC_DEF as u8 {
        if !defen {
            return None;
        }
        cust * msskey_num_max + key
    } else {
        let base = if defen {
            (FWDL_MSS_POOL_DEFKEYSETS_SIZE as usize) << 3
        } else {
            0
        };
        base + dev * msskey_num_max * msscust_max + cust * msskey_num_max + key
    };
    let sel_byte = mss_sel_idx >> 3;
    let sel_bit = (mss_sel_idx & 0x7) as u32;
    let byte = *rmp_tbl.get(sel_byte)?;
    if byte & (1 << sel_bit) == 0 {
        return None;
    }
    let mut key_idx = (byte & ((1u8 << sel_bit) - 1)).count_ones() as usize;
    for &b in &rmp_tbl[..sel_byte] {
        key_idx += b.count_ones() as usize;
    }
    Some(key_idx)
}

/// One section of the image, resolved to bytes.
#[derive(Clone, Copy, Debug)]
pub struct Section<'a> {
    /// Where the WCPU loads it (`FWSECTION_HDR_W0_DL_ADDR`).
    pub dl_addr: u32,
    /// The bytes to push (the checksum trailer included when present).
    pub data: &'a [u8],
    /// `FWSECTION_HDR_W1_REDL`.
    pub redl: bool,
    /// Not downloaded: a second security section, or one whose key the
    /// part cannot use.
    pub ignore: bool,
    /// Under secure boot, the key copied over the tail of the section's
    /// last packet (`__rtw89_fw_download_main`, fw.c:1655-1657).
    pub key: Option<&'a [u8]>,
    /// A shorter length to push instead of `data.len()` (version 0 only;
    /// zero means none).
    pub len_override: u32,
}

impl Section<'_> {
    const EMPTY: Self = Self {
        dl_addr: 0,
        data: &[],
        redl: false,
        ignore: false,
        key: None,
        len_override: 0,
    };

    /// Bytes the download pushes for this section (`residue_len` at
    /// fw.c:1616-1630).
    pub fn push_len(&self) -> usize {
        let len = self.data.len();
        let over = self.len_override as usize;
        if over != 0 && over <= len { over } else { len }
    }

    /// Whether the key rides in the section's last packet
    /// (`copy_key`, fw.c:1632-1640): only when the whole section fits one
    /// packet and is at least the key's length.
    pub fn copies_key(&self) -> Option<&[u8]> {
        let key = self.key?;
        let residue = self.push_len();
        if residue > FWDL_SECTION_PER_PKT_LEN as usize || self.data.len() < key.len() {
            None
        } else {
            Some(key)
        }
    }
}

/// The download plan: the header packet to prime the firmware with, and
/// the sections to push after it.
#[derive(Clone, Copy)]
pub struct Plan<'a> {
    hdr_ver: u8,
    version: Version,
    /// The base header: the fixed words and the section table, without
    /// the dynamic header (`info->hdr_len - info->dynamic_hdr_len`).
    header: &'a [u8],
    sections: [Section<'a>; MAX_SECTIONS],
    section_num: u8,
    dsp_checksum: bool,
}

/// The longest header packet: a version-1 header with every section.
pub const HEADER_PACKET_MAX: usize = HDR_V1_BYTES + MAX_SECTIONS * SECTION_BYTES;

impl<'a> Plan<'a> {
    /// The image's version words.
    pub fn version(&self) -> Version {
        self.version
    }

    /// The inner header's version.
    pub fn header_version(&self) -> u8 {
        self.hdr_ver
    }

    /// Whether the header asks for the per-key DSP checksum.
    pub fn dsp_checksum(&self) -> bool {
        self.dsp_checksum
    }

    /// The sections, in download order (ignored ones included, flagged).
    pub fn sections(&self) -> &[Section<'a>] {
        &self.sections[..self.section_num as usize]
    }

    /// Bytes the whole download pushes after the header packet.
    pub fn payload_len(&self) -> usize {
        self.sections()
            .iter()
            .filter(|s| !s.ignore)
            .map(Section::push_len)
            .sum()
    }

    /// Write the header packet's payload into `out` and return its length
    /// (`__rtw89_fw_download_hdr` with the version's tweak, fw.c:1474-1568):
    /// the base header with the per-packet size stamped into word 7 and,
    /// on version 1, the ignored sections compacted out of the table.
    pub fn header_packet(&self, out: &mut [u8]) -> Result<usize, FwError> {
        let len = self.header.len();
        let dst = out.get_mut(..len).ok_or(FwError::Truncated)?;
        dst.copy_from_slice(self.header);
        let stamp = |dst: &mut [u8], at: usize, mask: u32, val: u32| {
            let w = u32::from_le_bytes([dst[at], dst[at + 1], dst[at + 2], dst[at + 3]]);
            dst[at..at + 4].copy_from_slice(&replace(w, mask, val).to_le_bytes());
        };
        match self.hdr_ver {
            0 => {
                stamp(dst, 28, FW_HDR_W7_PART_SIZE, FWDL_SECTION_PER_PKT_LEN);
                for (i, s) in self.sections().iter().enumerate() {
                    if s.len_override != 0 {
                        let at = HDR_V0_BYTES + i * SECTION_BYTES + 4;
                        stamp(dst, at, FWSECTION_HDR_W1_SEC_SIZE, s.len_override);
                    }
                }
                Ok(len)
            }
            1 => {
                stamp(dst, 28, FW_HDR_V1_W7_PART_SIZE, FWDL_SECTION_PER_PKT_LEN);
                let mut kept = 0usize;
                for (i, s) in self.sections().iter().enumerate() {
                    if s.ignore {
                        continue;
                    }
                    if kept != i {
                        let (src_at, dst_at) = (
                            HDR_V1_BYTES + i * SECTION_BYTES,
                            HDR_V1_BYTES + kept * SECTION_BYTES,
                        );
                        dst.copy_within(src_at..src_at + SECTION_BYTES, dst_at);
                    }
                    kept += 1;
                }
                stamp(dst, 24, FW_HDR_V1_W6_SEC_NUM, kept as u32);
                Ok(HDR_V1_BYTES + kept * SECTION_BYTES)
            }
            _ => Err(FwError::BadHeaderVersion),
        }
    }
}

#[cfg(test)]
pub(crate) mod build {
    //! Container builders for the host tests: a version-0 or version-1
    //! image wrapped in an MFW table, byte-exact to the upstream structs.

    use alloc::vec::Vec;

    use super::*;

    /// One section to build.
    pub struct Sec {
        pub dl_addr: u32,
        pub kind: u8,
        pub data: Vec<u8>,
        pub checksum: bool,
        /// Word 2 (MSSC) verbatim.
        pub w2: u32,
        /// Bytes appended after the section (the MSSC blob).
        pub trailer: Vec<u8>,
    }

    impl Sec {
        pub fn plain(dl_addr: u32, data: Vec<u8>) -> Self {
            Self {
                dl_addr,
                kind: 0,
                data,
                checksum: false,
                w2: 0,
                trailer: Vec::new(),
            }
        }
    }

    /// An inner image with the given header version.
    pub fn image(hdr_ver: u8, version: Version, dsp_checksum: bool, secs: &[Sec]) -> Vec<u8> {
        let base = if hdr_ver == 0 {
            HDR_V0_BYTES
        } else {
            HDR_V1_BYTES
        };
        let mut out = Vec::new();
        let mut words = [0u32; 12];
        words[1] = (version.major as u32)
            | (version.minor as u32) << 8
            | (version.sub as u32) << 16
            | (version.idx as u32) << 24;
        words[2] = version.commit_id;
        words[3] = (hdr_ver as u32) << 24;
        words[6] = (secs.len() as u32) << 8 | if dsp_checksum { 1 << 24 } else { 0 };
        words[7] = 0;
        for w in &words[..base / 4] {
            out.extend_from_slice(&w.to_le_bytes());
        }
        for s in secs {
            let mut w1 = (s.data.len() as u32) | (s.kind as u32) << 24;
            if s.checksum {
                w1 |= FWSECTION_HDR_W1_CHECKSUM;
            }
            out.extend_from_slice(&s.dl_addr.to_le_bytes());
            out.extend_from_slice(&w1.to_le_bytes());
            out.extend_from_slice(&s.w2.to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
        }
        for s in secs {
            out.extend_from_slice(&s.data);
            if s.checksum {
                out.extend_from_slice(&[0xC5; FWDL_SECTION_CHKSUM_LEN as usize]);
            }
            out.extend_from_slice(&s.trailer);
        }
        out
    }

    /// `images`: `(cv, type, mp, bytes)`.
    pub fn mfw(images: &[(u8, u8, u8, Vec<u8>)]) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(RTW89_MFW_SIG as u8);
        out.push(images.len() as u8);
        out.extend_from_slice(&[0; 2]);
        out.extend_from_slice(&[1, 2, 3, 4]);
        out.extend_from_slice(&[0; 8]);
        let mut shift = MFW_HDR_BYTES + images.len() * MFW_INFO_BYTES;
        for (cv, kind, mp, bytes) in images {
            out.push(*cv);
            out.push(*kind);
            out.push(*mp);
            out.push(0);
            out.extend_from_slice(&(shift as u32).to_le_bytes());
            out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(&[0; 4]);
            shift += bytes.len();
        }
        for (_, _, _, bytes) in images {
            out.extend_from_slice(bytes);
        }
        out
    }

    /// A formatted MSS key pool: `keys` 512-byte keys, one device type,
    /// one customer, `msskey_num_max` slots, remap table bits `present`.
    pub fn mss_pool(keys: &[[u8; 512]], msskey_num_max: u16, present: &[u8]) -> Vec<u8> {
        let rmp_tbl_size = ((msskey_num_max as usize * 1 * 1) >> 3).max(present.len());
        let mut out = Vec::new();
        out.extend_from_slice(&MSS_SIGNATURE);
        out.extend_from_slice(&(MSS_POOL_HDR_BYTES as u32).to_le_bytes()); // rmp_tbl_offset
        out.extend_from_slice(&((MSS_POOL_HDR_BYTES + rmp_tbl_size) as u32).to_le_bytes());
        out.push(0); // defen
        out.extend_from_slice(&[0; 3]);
        out.push(MSS_POOL_RMP_TBL_BITMASK as u8);
        out.push(1); // mssdev_max
        out.extend_from_slice(&(keys.len() as u16).to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes()); // msscust_max
        out.extend_from_slice(&msskey_num_max.to_le_bytes());
        out.extend_from_slice(&[0; 4]);
        let mut tbl = present.to_vec();
        tbl.resize(rmp_tbl_size, 0);
        out.extend_from_slice(&tbl);
        for k in keys {
            out.extend_from_slice(k);
        }
        out
    }

    pub fn digest_of(blob: &[u8]) -> Digest {
        Digest(*blake3::hash(blob).as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::build::*;
    use super::*;

    fn version() -> Version {
        Version {
            major: 0,
            minor: 35,
            sub: 54,
            idx: 0,
            commit_id: 0xDEAD_BEEF,
        }
    }

    #[test]
    fn the_manifest_is_the_pinned_size_and_digest() {
        // The recorded pin, spelled out here so an edit to `manifest.toml`
        // that this crate did not intend fails the suite rather than
        // silently changing which blob the metal will accept:
        // `rtw89/rtw8922a_fw-4.bin` at linux-firmware fb0889c0.
        assert_eq!(PINNED_SIZE, 1_849_226);
        assert_eq!(
            PINNED_DIGEST,
            Digest([
                0x5e, 0xe6, 0x18, 0xf7, 0x94, 0x06, 0xb2, 0xde, 0x15, 0xee, 0x41, 0xae, 0x18, 0x18,
                0xad, 0x9e, 0xf4, 0xc5, 0xca, 0x0b, 0xa5, 0x48, 0x09, 0xa2, 0xa6, 0x54, 0x60, 0x68,
                0x45, 0xae, 0x55, 0x5a,
            ])
        );
        // A digest is not a size check: the empty blob, and any other blob,
        // is refused before the container is read.
        assert_eq!(
            FirmwareImage::verified(&[], &PINNED_DIGEST).err(),
            Some(FwError::DigestMismatch)
        );
    }

    #[test]
    fn the_wrapper_selects_the_highest_cut_at_or_below_the_chips() {
        let img = |cv: u8| {
            image(
                1,
                Version {
                    idx: cv,
                    ..version()
                },
                false,
                &[Sec::plain(0, vec![cv; 8])],
            )
        };
        let blob = mfw(&[
            (2, FW_TYPE_NORMAL, 0, img(2)),
            (0, FW_TYPE_NORMAL, 0, img(0)),
            (1, FW_TYPE_NORMAL, 1, img(9)), // mp: never selected
            (1, 3, 0, img(7)),              // WOWLAN: never selected
            (1, FW_TYPE_NORMAL, 0, img(1)),
        ]);
        let fw = FirmwareImage::verified(&blob, &digest_of(&blob)).unwrap();
        assert_eq!(fw.suit(0).unwrap().version().idx, 0);
        assert_eq!(fw.suit(1).unwrap().version().idx, 1);
        assert_eq!(fw.suit(5).unwrap().version().idx, 2);
        let legacy = img(3);
        let fw = FirmwareImage::verified(&legacy, &digest_of(&legacy)).unwrap();
        assert_eq!(
            fw.suit(0).unwrap().version().idx,
            3,
            "no signature: the whole blob"
        );
        let none = mfw(&[(3, FW_TYPE_NORMAL, 0, img(3))]);
        let fw = FirmwareImage::verified(&none, &digest_of(&none)).unwrap();
        assert_eq!(fw.suit(1).err(), Some(FwError::NoImageForCut));
        let mut wrong = blob.clone();
        wrong[100] ^= 1;
        assert_eq!(
            FirmwareImage::verified(&wrong, &digest_of(&blob)).err(),
            Some(FwError::DigestMismatch)
        );
    }

    #[test]
    fn a_version_1_plan_measures_sections_keys_and_the_header_packet() {
        let pool = mss_pool(&[[0xA1; 512], [0xB2; 512]], 8, &[0b0000_0110]);
        let mut sec_content = vec![0u8; 600];
        // `sb_sel_ver` (4 bytes at 58) and `key_sign_len` (2 bytes at 60)
        // overlap in upstream's union by design: 0x08000100's top half is
        // 0x0800, a 512-byte key.
        sec_content[58..62].copy_from_slice(&0x0800_0100u32.to_le_bytes());
        let secs = [
            Sec::plain(0x1000, vec![1; 3000]),
            Sec {
                dl_addr: 0x2000,
                kind: FWDL_SECURITY_SECTION_TYPE as u8,
                data: sec_content.clone(),
                checksum: true,
                w2: FORMATTED_MSSC,
                trailer: pool.clone(),
            },
            Sec {
                dl_addr: 0x3000,
                kind: FWDL_SECURITY_SECTION_TYPE as u8,
                data: sec_content.clone(),
                checksum: false,
                w2: FORMATTED_MSSC,
                trailer: pool.clone(),
            },
            Sec::plain(0x4000, vec![4; 10]),
        ];
        let img = image(1, version(), false, &secs);
        let blob = mfw(&[(0, FW_TYPE_NORMAL, 0, img)]);
        let fw = FirmwareImage::verified(&blob, &digest_of(&blob)).unwrap();
        let suit = fw.suit(0).unwrap();
        assert_eq!(suit.header_version(), 1);
        assert_eq!(suit.version(), version());

        // Secure boot off: both security sections measured, the second
        // ignored, no key.
        let plan = suit.plan(&SecureBoot::OFF).unwrap();
        let s = plan.sections();
        assert_eq!(s.len(), 4);
        assert_eq!(s[0].data.len(), 3000);
        assert_eq!(
            s[1].data.len(),
            608,
            "the checksum trailer rides with the section"
        );
        assert!(!s[1].ignore && s[1].key.is_none());
        assert!(s[2].ignore);
        assert_eq!(s[3].data, &[4; 10]);
        assert_eq!(plan.payload_len(), 3000 + 608 + 10);
        let mut out = [0u8; HEADER_PACKET_MAX];
        let n = plan.header_packet(&mut out).unwrap();
        assert_eq!(
            n,
            HDR_V1_BYTES + 3 * SECTION_BYTES,
            "one section compacted out"
        );
        let w6 = u32::from_le_bytes([out[24], out[25], out[26], out[27]]);
        assert_eq!(get(w6, FW_HDR_V1_W6_SEC_NUM), 3);
        let w7 = u32::from_le_bytes([out[28], out[29], out[30], out[31]]);
        assert_eq!(get(w7, FW_HDR_V1_W7_PART_SIZE), 2020);
        let third = HDR_V1_BYTES + 2 * SECTION_BYTES;
        assert_eq!(
            u32::from_le_bytes([out[third], out[third + 1], out[third + 2], out[third + 3]]),
            0x4000,
            "the fourth section moved up"
        );

        // Secure boot on, key slot 1 present in the remap table (bit 1):
        // key index = popcount of lower bits = 0 → the first key.
        let on = SecureBoot {
            secure_boot: true,
            sb_sel_mgn: 0x0800_0100,
            mss_dev_type: 0,
            mss_cust_idx: 0,
            mss_key_num: 1,
            mss_idx: 0,
        };
        let plan = suit.plan(&on).unwrap();
        let key = plan.sections()[1].key.expect("a key");
        assert_eq!(key.len(), 512);
        assert!(key.iter().all(|&b| b == 0xA1));
        assert_eq!(plan.sections()[1].copies_key().map(<[u8]>::len), Some(512));
        // A section shorter than its key is pushed without the copy
        // (upstream warns and skips, fw.c:1632-1640).
        let short = Section {
            data: &[0; 100],
            ..plan.sections()[1]
        };
        assert!(short.copies_key().is_none());
        // Key slot 3 is absent: the section is ignored, not refused.
        let absent = SecureBoot {
            mss_key_num: 3,
            ..on
        };
        let plan = suit.plan(&absent).unwrap();
        assert!(plan.sections()[1].ignore);
        // The pool's signature is checked before anything else in it.
        let mut bad = blob.clone();
        let at = bad.windows(8).position(|w| w == b"MSSKPOOL").unwrap();
        bad[at] = b'X';
        let fw = FirmwareImage::verified(&bad, &digest_of(&bad)).unwrap();
        assert_eq!(
            fw.suit(0).unwrap().plan(&SecureBoot::OFF).err(),
            Some(FwError::BadMssPool)
        );
    }

    #[test]
    fn a_version_0_plan_masks_the_address_and_keeps_the_table_whole() {
        let secs = [
            Sec::plain(0xE000_1000, vec![7; 2021]),
            Sec {
                dl_addr: 0x2000,
                kind: FWDL_SECURITY_SECTION_TYPE as u8,
                data: vec![9; 64],
                checksum: false,
                w2: 2,
                trailer: vec![0xEE; 2 * 512],
            },
        ];
        let img = image(0, version(), false, &secs);
        let fw = FirmwareImage::verified(&img, &digest_of(&img)).unwrap();
        let suit = fw.suit(0).unwrap();
        assert_eq!(suit.header_version(), 0);
        let plan = suit.plan(&SecureBoot::OFF).unwrap();
        assert_eq!(plan.sections()[0].dl_addr, 0x0000_1000);
        assert_eq!(plan.sections()[1].data.len(), 64);
        assert_eq!(plan.payload_len(), 2021 + 64);
        let mut out = [0u8; HEADER_PACKET_MAX];
        assert_eq!(
            plan.header_packet(&mut out).unwrap(),
            HDR_V0_BYTES + 2 * SECTION_BYTES
        );
        let on = SecureBoot {
            secure_boot: true,
            mss_idx: 1,
            ..SecureBoot::OFF
        };
        let key = suit.plan(&on).unwrap().sections()[1].key.unwrap();
        assert!(key.iter().all(|&b| b == 0xEE) && key.len() == 512);
        let over = SecureBoot { mss_idx: 2, ..on };
        assert_eq!(suit.plan(&over).err(), Some(FwError::BadMssIndex));
    }

    #[test]
    fn a_container_that_lies_about_its_size_is_refused() {
        let secs = [Sec::plain(0, vec![1; 100])];
        let mut img = image(1, version(), false, &secs);
        img.push(0); // one byte past the last section
        let fw = FirmwareImage::verified(&img, &digest_of(&img)).unwrap();
        assert_eq!(
            fw.suit(0).unwrap().plan(&SecureBoot::OFF).err(),
            Some(FwError::BadSize)
        );
        let mut short: Vec<u8> = image(1, version(), false, &secs);
        short.truncate(short.len() - 1);
        let fw = FirmwareImage::verified(&short, &digest_of(&short)).unwrap();
        assert_eq!(
            fw.suit(0).unwrap().plan(&SecureBoot::OFF).err(),
            Some(FwError::Truncated)
        );
        let mut v9 = image(1, version(), false, &secs);
        v9[15] = 9;
        let fw = FirmwareImage::verified(&v9, &digest_of(&v9)).unwrap();
        assert_eq!(
            fw.suit(0).unwrap().plan(&SecureBoot::OFF).err(),
            Some(FwError::BadHeaderVersion)
        );
    }
}
