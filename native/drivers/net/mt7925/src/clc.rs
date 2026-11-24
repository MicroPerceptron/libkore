//! **M2-regd, CLC** — the vendor's per-country power limits, read out of
//! the RAM code's `FW_TYPE_CLC` region and handed to the part as
//! `SET_POWER_LIMIT` tag 3.
//!
//! Upstream splits this in two. `mt7925_load_clc` (`mt7925/mcu.c:943`)
//! walks the region once and keeps, per table index, the first record whose
//! encapsulation matches the part's efuse (`MT_EE_HW_TYPE & 3`).
//! `__mt7925_mcu_set_clc` (`:3818`) then sends one command per rule of that
//! record naming the country in force. This module does both, without
//! copying: a [`Clc`] borrows the image, and each [`ClcCommand`] borrows its
//! segment.
//!
//! **CLC limits power; it never permits.** What may be transmitted at all
//! is `kore-dot11::regulatory`'s answer. A CLC rule for a country is the
//! vendor's tighter cap inside that answer.

/// `MT792x_CLC_POWER` and `MT792x_CLC_POWER_EXT` (`mt792x.h:90-91`): the
/// two tables `mt7925_mcu_set_clc` sends. Index 2 (`BE_CTRL`) and 3
/// (`REGD`) are kept by upstream but never sent from here, so they are not
/// selected.
pub const CLC_TABLES: usize = 2;

/// `sizeof(struct mt7925_clc)` up to `data` (`mt7925/mt7925.h:226`).
const RECORD_HEAD: usize = 16;
/// `sizeof(struct mt7925_clc_segment)` (`:206`).
const SEGMENT_BYTES: usize = 16;
/// `sizeof(struct mt7925_clc_rule)` (`:198`).
const RULE_BYTES: usize = 8;
/// The request header `__mt7925_mcu_set_clc` builds in front of a segment:
/// `rsv[4]`, tag, len, ver, pad, size, idx, env, acpi, pad, alpha2, type,
/// `rsvd[64]`.
pub const CLC_HEADER_BYTES: usize = 84;

/// Why the CLC region could not be read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClcError {
    /// A record's length is shorter than its own header or runs past the
    /// region (`"Invalid CLC record"`, `mcu.c:995-1005`).
    BadRecord { at: usize },
    /// A rule names a segment the record does not carry, or a segment
    /// points outside its record.
    BadSegment { table: u8, seg_idx: u8 },
}

/// One selected table: `idx`, `ver`, and the bytes behind its header.
#[derive(Clone, Copy, Debug)]
struct Table<'a> {
    idx: u8,
    ver: u8,
    nr_seg: u8,
    data: &'a [u8],
}

/// The CLC tables this part takes, chosen from the region by encapsulation.
#[derive(Clone, Copy, Debug)]
pub struct Clc<'a> {
    tables: [Option<Table<'a>>; CLC_TABLES],
}

impl<'a> Clc<'a> {
    /// Walk the region and keep the first record of each table whose
    /// encapsulation is `encap` (`MT_EE_HW_TYPE_ENCAP`, two bits).
    ///
    /// Upstream stops at the first record whose index is past its table
    /// array; records for tables 2 and 3 are walked past, and the walk
    /// still validates them, because a malformed record anywhere means the
    /// region is not the one the pin vouched for.
    pub fn select(region: &'a [u8], encap: u8) -> Result<Self, ClcError> {
        let mut tables = [None; CLC_TABLES];
        let mut at = 0;
        while region.len() - at >= RECORD_HEAD {
            let len = u32::from_le_bytes([region[at], region[at + 1], region[at + 2], region[at + 3]]) as usize;
            if len < RECORD_HEAD || len > region.len() - at {
                return Err(ClcError::BadRecord { at });
            }
            let (idx, ver) = (region[at + 4], region[at + 5]);
            // `t0`: nr_country, type, nr_seg.
            let (typ, nr_seg) = (region[at + 7], region[at + 8]);
            if (idx as usize) < CLC_TABLES && tables[idx as usize].is_none() && typ & 0b11 == encap & 0b11 {
                tables[idx as usize] = Some(Table { idx, ver, nr_seg, data: &region[at + RECORD_HEAD..at + len] });
            }
            at += len;
        }
        Ok(Self { tables })
    }

    /// Whether table `idx` was found for this encapsulation.
    pub fn has_table(&self, idx: usize) -> bool {
        self.tables.get(idx).is_some_and(Option::is_some)
    }

    /// The commands for `alpha2`, table 0's rules then table 1's, each in
    /// the order the record lists them, in `env` (`enum environment_cap`;
    /// 0 is "any", as the capture shows).
    ///
    /// **Returns how many it produced through `each`**, and zero means the
    /// image has no rule for this country, which is upstream's `-ENOENT`,
    /// answered there by retrying with `00`. That retry is the caller's,
    /// because the caller knows which country was in force.
    pub fn for_country(
        &self,
        alpha2: [u8; 2],
        env: u8,
        mut each: impl FnMut(ClcCommand<'a>) -> bool,
    ) -> Result<usize, ClcError> {
        let mut sent = 0;
        for table in self.tables.iter().flatten() {
            let data = table.data;
            let rules_at = SEGMENT_BYTES * table.nr_seg as usize;
            let bad = ClcError::BadSegment { table: table.idx, seg_idx: 0 };
            // `last_pos = data + le32(data + 4)`: the first segment's
            // offset, which is where the rules end.
            let last = data
                .get(4..8)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
                .ok_or(bad)?;
            let mut pos = rules_at;
            while pos < last {
                let rule = data.get(pos..pos + RULE_BYTES).ok_or(bad)?;
                pos += RULE_BYTES;
                if rule[0] != alpha2[0] || rule[1] != alpha2[1] {
                    continue;
                }
                let seg_idx = rule[4];
                let bad = ClcError::BadSegment { table: table.idx, seg_idx };
                let seg_at = (seg_idx as usize).checked_sub(1).ok_or(bad)? * SEGMENT_BYTES;
                let seg = data.get(seg_at..seg_at + SEGMENT_BYTES).ok_or(bad)?;
                let off = u32::from_le_bytes([seg[4], seg[5], seg[6], seg[7]]) as usize;
                let len = u32::from_le_bytes([seg[8], seg[9], seg[10], seg[11]]) as usize;
                let payload = data.get(off..off.checked_add(len).ok_or(bad)?).ok_or(bad)?;
                let size = u16::try_from(len).map_err(|_| bad)?;
                let mut header = [0u8; CLC_HEADER_BYTES];
                header[4..6].copy_from_slice(&3u16.to_le_bytes());
                header[6..8].copy_from_slice(&((CLC_HEADER_BYTES as u16 - 4).wrapping_add(size)).to_le_bytes());
                header[8] = table.ver;
                header[10..12].copy_from_slice(&size.to_le_bytes());
                header[12] = table.idx;
                header[13] = env;
                header[16..18].copy_from_slice(&alpha2);
                header[18..20].copy_from_slice(&rule[2..4]);
                sent += 1;
                if !each(ClcCommand { header, payload }) {
                    return Ok(sent);
                }
            }
        }
        Ok(sent)
    }
}

/// One `SET_POWER_LIMIT` tag 3 command: the request header, and the segment
/// it carries, borrowed from the image.
#[derive(Clone, Copy, Debug)]
pub struct ClcCommand<'a> {
    pub header: [u8; CLC_HEADER_BYTES],
    pub payload: &'a [u8],
}

impl ClcCommand<'_> {
    /// Bytes the command body occupies (header + segment).
    pub fn len(&self) -> usize {
        CLC_HEADER_BYTES + self.payload.len()
    }
    pub fn is_empty(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// A record: `len, idx, ver, t0 {nr_country, type, nr_seg, rsv[7]}`,
    /// then `nr_seg` segments, then rules, then segment bytes.
    fn record(idx: u8, ver: u8, typ: u8, rules: &[([u8; 2], [u8; 2], u8)], segs: &[&[u8]]) -> Vec<u8> {
        let rules_at = SEGMENT_BYTES * segs.len();
        let seg_data_at = rules_at + RULE_BYTES * rules.len();
        let mut data = alloc::vec![0u8; seg_data_at];
        let mut off = seg_data_at;
        for (i, s) in segs.iter().enumerate() {
            let at = i * SEGMENT_BYTES;
            data[at] = i as u8 + 1;
            data[at + 4..at + 8].copy_from_slice(&(off as u32).to_le_bytes());
            data[at + 8..at + 12].copy_from_slice(&(s.len() as u32).to_le_bytes());
            off += s.len();
        }
        for (i, (a2, ty, seg)) in rules.iter().enumerate() {
            let at = rules_at + i * RULE_BYTES;
            data[at..at + 2].copy_from_slice(a2);
            data[at + 2..at + 4].copy_from_slice(ty);
            data[at + 4] = *seg;
        }
        for s in segs {
            data.extend_from_slice(s);
        }
        let mut r = alloc::vec![0u8; RECORD_HEAD];
        r[0..4].copy_from_slice(&((RECORD_HEAD + data.len()) as u32).to_le_bytes());
        r[4] = idx;
        r[5] = ver;
        r[6] = 0xff;
        r[7] = typ;
        r[8] = segs.len() as u8;
        r.extend(data);
        r
    }

    fn collect(clc: &Clc<'_>, a2: [u8; 2]) -> Vec<([u8; CLC_HEADER_BYTES], Vec<u8>)> {
        let mut v = Vec::new();
        clc.for_country(a2, 0, |c| {
            v.push((c.header, c.payload.to_vec()));
            true
        })
        .unwrap();
        v
    }

    /// The header `__mt7925_mcu_set_clc` builds, byte for byte against the
    /// capture's first `US` command (2026-11-20, `len=564`: a 432-byte
    /// segment): tag 3, len 512, ver 2, size 432, idx 0, env 0, `US`, `-D`.
    #[test]
    fn the_header_is_the_captured_one() {
        let seg = alloc::vec![0x5au8; 432];
        let region = record(0, 2, 0, &[(*b"US", *b"-D", 1)], &[&seg]);
        let clc = Clc::select(&region, 0).unwrap();
        let got = collect(&clc, *b"US");
        assert_eq!(got.len(), 1);
        let mut want = [0u8; CLC_HEADER_BYTES];
        want[..20].copy_from_slice(&[
            0, 0, 0, 0, 0x03, 0x00, 0x00, 0x02, 0x02, 0x00, 0xb0, 0x01, 0x00, 0x00, 0x00, 0x00, b'U', b'S', b'-',
            b'D',
        ]);
        assert_eq!(got[0].0, want);
        assert_eq!(got[0].1, seg);
        // 48-byte descriptor + header + segment is the captured length.
        assert_eq!(48 + CLC_HEADER_BYTES + seg.len(), 564);
    }

    /// **The encapsulation decides the table**, first match wins, and a
    /// record for an unsent table (2, `BE_CTRL`) is walked past.
    #[test]
    fn encapsulation_selects_and_first_wins() {
        let mut region = record(0, 2, 1, &[(*b"US", *b"-D", 1)], &[&[1]]);
        region.extend(record(0, 2, 0, &[(*b"US", *b"-D", 1)], &[&[2]]));
        region.extend(record(0, 2, 0, &[(*b"US", *b"-D", 1)], &[&[3]]));
        region.extend(record(2, 2, 0, &[], &[]));
        region.extend(record(1, 2, 0, &[(*b"US", *b"0C", 1)], &[&[4]]));
        let clc = Clc::select(&region, 0).unwrap();
        let got: Vec<u8> = collect(&clc, *b"US").into_iter().map(|(_, p)| p[0]).collect();
        assert_eq!(got, [2, 4], "encap 0's first table 0, then table 1");
        let clc1 = Clc::select(&region, 1).unwrap();
        assert_eq!(collect(&clc1, *b"US").len(), 1, "encap 1 has no table 1");
    }

    /// Only the country's own rules are sent, in record order; a country
    /// with none produces zero, upstream's `-ENOENT`.
    #[test]
    fn only_this_countrys_rules() {
        let region = record(
            0,
            2,
            0,
            &[(*b"00", *b"-D", 1), (*b"US", *b"-D", 2), (*b"UY", *b"-D", 1), (*b"US", *b"0C", 3), (*b"GS", *b"-D", 1)],
            &[&[0xa], &[0xb], &[0xc]],
        );
        let clc = Clc::select(&region, 0).unwrap();
        let us: Vec<u8> = collect(&clc, *b"US").into_iter().map(|(_, p)| p[0]).collect();
        assert_eq!(us, [0xb, 0xc]);
        assert_eq!(clc.for_country(*b"ZZ", 0, |_| true), Ok(0));
        // The environment is the caller's and rides in byte 13.
        let mut env = 0;
        clc.for_country(*b"US", 1, |c| {
            env = c.header[13];
            false
        })
        .unwrap();
        assert_eq!(env, 1);
    }

    /// A record that lies about its length, or a rule naming a segment
    /// that is not there, refuses instead of reading past the region.
    #[test]
    fn malformed_regions_refuse() {
        let mut region = record(0, 2, 0, &[(*b"US", *b"-D", 1)], &[&[1]]);
        let lie = region.len() as u32 + 1;
        region[0..4].copy_from_slice(&lie.to_le_bytes());
        assert_eq!(Clc::select(&region, 0).err(), Some(ClcError::BadRecord { at: 0 }));
        let region = record(0, 2, 0, &[(*b"US", *b"-D", 9)], &[&[1]]);
        let clc = Clc::select(&region, 0).unwrap();
        assert!(matches!(clc.for_country(*b"US", 0, |_| true), Err(ClcError::BadSegment { seg_idx: 9, .. })));
        let region = record(0, 2, 0, &[(*b"US", *b"-D", 0)], &[&[1]]);
        let clc = Clc::select(&region, 0).unwrap();
        assert!(matches!(clc.for_country(*b"US", 0, |_| true), Err(ClcError::BadSegment { seg_idx: 0, .. })));
    }

    /// **Against the real blob** (`vendor/firmware/…`, not in git, so
    /// `#[ignore]`d; run with `cargo test -p kore-mt7925 -- --ignored`).
    /// The sixteen `US` commands the capture shows, by length, for
    /// encapsulation 0: table 0's eight, then table 1's eight.
    #[test]
    #[ignore = "needs the pinned RAM blob in vendor/firmware"]
    fn the_real_image_gives_the_captured_lengths() {
        extern crate std;
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../vendor/firmware/mediatek/mt7925/WIFI_RAM_CODE_MT7925_1_1.bin");
        let image = std::fs::read(path).expect("the pinned RAM blob");
        let ram = crate::fw::RamCode::parse(&image).unwrap();
        let r = ram.regions().iter().find(|r| r.is_clc()).unwrap();
        let region = &image[r.offset as usize..(r.offset + r.len) as usize];
        let clc = Clc::select(region, 0).unwrap();
        let mut lens = Vec::new();
        clc.for_country(*b"US", 0, |c| {
            lens.push(48 + c.len());
            true
        })
        .unwrap();
        assert_eq!(lens, [564, 1020, 1012, 612, 1046, 1046, 1022, 982, 636, 1068, 1028, 588, 934, 958, 1006, 966]);
        let mut world = 0;
        clc.for_country(*b"00", 0, |_| {
            world += 1;
            true
        })
        .unwrap();
        assert_eq!(world, 12, "the world fallback exists: six rules per table");
    }
}
