//! Borrowed PHY table plans from the digest-verified firmware container.
//! Framing/selection facts: Linux v6.17 fw.h:4070, fw.c:994, phy.c:1529.
//! A complete program is validated before its first selected register is
//! exposed. These values are register facts, never MMIO or channel authority.
use super::{Element, FirmwareImage};

const MAX_RECORDS: usize = 8192;
const MAX_HEADLINES: usize = 256;
/// Three firmware pages of 500 packed RF writes (fw.h:4141-4142).
pub const RF_WORDS_MAX: usize = 1500;
const TARGET_MASK: u32 = 0x0fff_ffff;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TableError {
    Missing,
    Duplicate,
    Shape,
    Selection,
    Program,
    Register,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RfPath {
    A,
    B,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TableKind {
    Baseband,
    Radio(RfPath),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Register {
    pub address: u32,
    pub value: u32,
}

#[derive(Clone, Copy)]
pub struct RegTable<'a> {
    bytes: &'a [u8],
    headline: usize,
    target: u32,
    kind: TableKind,
}
/// Gain entries are indexed calibration data, not MMIO register writes.
#[derive(Clone, Copy)]
pub struct GainTable<'a>(&'a [u8]);
impl GainTable<'_> {
    pub fn len(&self) -> usize {
        self.0.len() / 8
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}
pub struct PhyTables<'a> {
    pub(crate) baseband: RegTable<'a>,
    pub(crate) gain: GainTable<'a>,
    /// Firmware slot order, which need not equal physical path order.
    pub(crate) radio: [RegTable<'a>; 2],
}
impl<'a> FirmwareImage<'a> {
    pub fn phy_tables(&self, rfe: u8, cut: u8) -> Result<PhyTables<'a>, TableError> {
        let mut bb = None;
        let mut gain = None;
        let mut radio = [None, None];
        let mut paths = 0;
        for e in self.elements().map_err(|_| TableError::Shape)? {
            match e.id {
                2 => {
                    if bb.is_some() {
                        return Err(TableError::Duplicate);
                    }
                    bb = Some(RegTable::new(e, rfe, cut, TableKind::Baseband)?);
                }
                3 => {
                    if gain.is_some() {
                        return Err(TableError::Duplicate);
                    }
                    shape(e.contents)?;
                    for b in e.contents.chunks_exact(8) {
                        let a = record(b).address;
                        let [kind, path_bw, band, cfg] = a.to_le_bytes();
                        let valid_kind = match cfg {
                            0 => kind <= 2,
                            1 => matches!(kind, 0 | 0x10 | 0x20 | 0x30 | 0x31),
                            2 => kind <= 1, // bypass, ignored by BE
                            3 => kind <= 3,
                            4 => rfe < 50 && kind <= 5, // eFEM-only, ignored for this RFE
                            _ => false,
                        };
                        if path_bw & 15 >= 2 || path_bw >> 4 >= 2 || band >= 12 || !valid_kind {
                            return Err(TableError::Register);
                        }
                    }
                    gain = Some(GainTable(e.contents));
                }
                4 | 5 => {
                    let slot = e.private[0] as usize;
                    if slot >= 2 {
                        return Err(TableError::Shape);
                    }
                    let bit = 1 << (e.id - 4);
                    if radio[slot].is_some() || paths & bit != 0 {
                        return Err(TableError::Duplicate);
                    }
                    paths |= bit;
                    let path = if e.id == 4 { RfPath::A } else { RfPath::B };
                    radio[slot] = Some(RegTable::new(e, rfe, cut, TableKind::Radio(path))?);
                }
                _ => {}
            }
        }
        let tables = PhyTables {
            baseband: bb.ok_or(TableError::Missing)?,
            gain: gain.ok_or(TableError::Missing)?,
            radio: [
                radio[0].ok_or(TableError::Missing)?,
                radio[1].ok_or(TableError::Missing)?,
            ],
        };
        for table in &tables.radio {
            if table.records().filter(|r| r.address >= 0x100).count() > RF_WORDS_MAX {
                return Err(TableError::Shape);
            }
        }
        Ok(tables)
    }
}
impl<'a> PhyTables<'a> {
    pub fn baseband(&self) -> RegTable<'a> {
        self.baseband
    }
    pub fn gain(&self) -> GainTable<'a> {
        self.gain
    }
    pub fn radio(&self) -> &[RegTable<'a>; 2] {
        &self.radio
    }
}
fn shape(bytes: &[u8]) -> Result<(), TableError> {
    if bytes.is_empty() || bytes.len() % 8 != 0 || bytes.len() / 8 > MAX_RECORDS {
        Err(TableError::Shape)
    } else {
        Ok(())
    }
}
fn record(bytes: &[u8]) -> Register {
    Register {
        address: u32::from_le_bytes(bytes[..4].try_into().unwrap()),
        value: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
    }
}
impl<'a> RegTable<'a> {
    fn new(e: Element<'a>, rfe: u8, cut: u8, kind: TableKind) -> Result<Self, TableError> {
        shape(e.contents)?;
        let headline = e
            .contents
            .chunks_exact(8)
            .take_while(|b| record(b).address >> 28 == 15)
            .count();
        if headline > MAX_HEADLINES || headline * 8 == e.contents.len() {
            return Err(TableError::Shape);
        }
        let mut best = None;
        let mut duplicate = false;
        // The reference's four priorities: exact; same RFE/any cut;
        // highest cut for the RFE; highest cut for wildcard RFE.
        for bytes in e.contents[..headline * 8].chunks_exact(8) {
            let target = record(bytes).address & TARGET_MASK;
            if target & !0x00ff_00ff != 0 {
                return Err(TableError::Selection);
            }
            let table_rfe = (target >> 16) as u8;
            let table_cut = target as u8;
            let rank = if target == (u32::from(rfe) << 16 | u32::from(cut)) {
                0x400
            } else if target == (u32::from(rfe) << 16 | 0xff) {
                0x300
            } else if table_rfe == rfe {
                0x200 + u32::from(table_cut)
            } else if table_rfe == 0xff {
                0x100 + u32::from(table_cut)
            } else {
                continue;
            };
            match best {
                Some((old, _)) if old > rank => {}
                Some((old, _)) if old == rank => duplicate = true,
                _ => {
                    best = Some((rank, target));
                    duplicate = false;
                }
            }
        }
        if duplicate {
            return Err(TableError::Selection);
        }
        let target = if headline == 0 {
            0
        } else {
            best.ok_or(TableError::Selection)?.1
        };
        let result = Self {
            bytes: e.contents,
            headline,
            target,
            kind,
        };
        let mut branch = false;
        let mut check = false;
        let mut matched = false;
        let mut otherwise = false;
        let mut candidate = 0;
        let mut writes = 0;
        for bytes in e.contents[headline * 8..].chunks_exact(8) {
            let r = record(bytes);
            match r.address >> 28 {
                8 | 9 => {
                    if r.value != 0 || r.address & TARGET_MASK & !0x00ff_00ff != 0 {
                        return Err(TableError::Program);
                    }
                    if check
                        || (r.address >> 28 == 8 && branch)
                        || (r.address >> 28 == 9 && (!branch || otherwise))
                        || headline == 0
                    {
                        return Err(TableError::Program);
                    }
                    if r.address >> 28 == 8 {
                        matched = false;
                    }
                    branch = true;
                    check = true;
                    candidate = r.address & TARGET_MASK;
                }
                4 => {
                    if !check || r.address != 0x4000_0000 || r.value != 0 {
                        return Err(TableError::Program);
                    }
                    if matched && candidate == target {
                        return Err(TableError::Program);
                    }
                    matched |= candidate == target;
                    check = false;
                }
                10 => {
                    if r.address != 0xa000_0000 || r.value != 0 {
                        return Err(TableError::Program);
                    }
                    if !branch || check || otherwise || !matched {
                        return Err(TableError::Program);
                    }
                    otherwise = true;
                }
                11 => {
                    if r.address != 0xb000_0000 || r.value != 0 {
                        return Err(TableError::Program);
                    }
                    if !branch || check || !matched {
                        return Err(TableError::Program);
                    }
                    branch = false;
                    otherwise = false;
                }
                0 => {
                    if check {
                        return Err(TableError::Program);
                    }
                    writes += 1;
                    match kind {
                        TableKind::Baseband if (0xf9..=0xfe).contains(&r.address) => {}
                        TableKind::Baseband if r.address & 3 == 0 && r.address < 0x1_0000 => {}
                        TableKind::Radio(_)
                            if r.address & !0x1_00ff == 0 && r.value <= 0xf_ffff => {}
                        _ => return Err(TableError::Register),
                    }
                }
                _ => return Err(TableError::Program),
            }
        }
        if branch || check || writes == 0 {
            return Err(TableError::Program);
        }
        Ok(result)
    }
    pub fn kind(&self) -> TableKind {
        self.kind
    }
    pub fn selected_target(&self) -> u32 {
        self.target
    }
    pub fn records(&self) -> Records<'a> {
        Records {
            bytes: self.bytes[self.headline * 8..].chunks_exact(8),
            target: self.target,
            candidate: 0,
            selected: true,
            matched: false,
        }
    }
}
pub struct Records<'a> {
    bytes: core::slice::ChunksExact<'a, u8>,
    target: u32,
    candidate: u32,
    selected: bool,
    matched: bool,
}
impl Iterator for Records<'_> {
    type Item = Register;
    fn next(&mut self) -> Option<Self::Item> {
        for bytes in self.bytes.by_ref() {
            let r = record(bytes);
            match r.address >> 28 {
                8 | 9 => self.candidate = r.address & TARGET_MASK,
                4 => {
                    self.selected = !self.matched && self.candidate == self.target;
                    self.matched |= self.selected;
                }
                10 => self.selected = false,
                11 => {
                    self.selected = true;
                    self.matched = false;
                }
                _ if self.selected => return Some(r),
                _ => {}
            }
        }
        None
    }
}
impl core::iter::FusedIterator for Records<'_> {}

#[cfg(test)]
mod tests {
    use super::*;
    fn plan<'a>(
        pairs: &[(u32, u32)],
        bytes: &'a mut [u8],
        rfe: u8,
        cut: u8,
        kind: TableKind,
    ) -> Result<RegTable<'a>, TableError> {
        for (pair, out) in pairs.iter().zip(bytes.chunks_exact_mut(8)) {
            out[..4].copy_from_slice(&pair.0.to_le_bytes());
            out[4..].copy_from_slice(&pair.1.to_le_bytes());
        }
        RegTable::new(
            Element {
                id: 2,
                version: [0; 4],
                private: [0; 8],
                contents: &bytes[..pairs.len() * 8],
            },
            rfe,
            cut,
            kind,
        )
    }
    #[test]
    fn branch_selection_is_whole_program_and_restores_unconditional_writes() {
        let pairs = [
            (0xf0010000, 0),
            (0xf0010002, 1),
            (0x80010000, 0),
            (0x40000000, 0),
            (0x1000, 1),
            (0x90010002, 0),
            (0x40000000, 0),
            (0x1000, 2),
            (0xa0000000, 0),
            (0xb0000000, 0),
            (0x1004, 3),
        ];
        let mut bytes = [0; 128];
        let p = plan(&pairs, &mut bytes, 1, 2, TableKind::Baseband).unwrap();
        let mut records = p.records();
        assert_eq!(
            records.next(),
            Some(Register {
                address: 0x1000,
                value: 2
            })
        );
        assert_eq!(
            records.next(),
            Some(Register {
                address: 0x1004,
                value: 3
            })
        );
        assert_eq!(records.next(), None);
        assert_eq!(records.next(), None);
        let mut bad = pairs;
        bad[4].0 = 0x10001; // invalid in an UNSELECTED arm
        assert!(matches!(
            plan(&bad, &mut bytes, 1, 2, TableKind::Baseband),
            Err(TableError::Register)
        ));
        let mut bad = pairs;
        bad[8].1 = 1;
        assert!(matches!(
            plan(&bad, &mut bytes, 1, 2, TableKind::Baseband),
            Err(TableError::Program)
        ));
        let mut bad = pairs;
        bad[5].0 = 0x90010000; // selected target absent from group
        assert!(matches!(
            plan(&bad, &mut bytes, 1, 2, TableKind::Baseband),
            Err(TableError::Program)
        ));
    }
    #[test]
    fn headline_priorities_and_ambiguity_do_not_depend_on_order() {
        let mut bytes = [0; 64];
        let mut pairs = [
            (0xf0010000, 0),
            (0xf0010002, 1),
            (0xf00100ff, 2),
            (0xf0ff0003, 3),
            (0x1000, 42),
        ];
        for _ in 0..4 {
            assert_eq!(
                plan(&pairs, &mut bytes, 1, 2, TableKind::Baseband)
                    .unwrap()
                    .selected_target(),
                0x10002
            );
            assert_eq!(
                plan(&pairs, &mut bytes, 1, 1, TableKind::Baseband)
                    .unwrap()
                    .selected_target(),
                0x100ff
            );
            assert_eq!(
                plan(&pairs, &mut bytes, 7, 1, TableKind::Baseband)
                    .unwrap()
                    .selected_target(),
                0xff0003
            );
            pairs[..4].rotate_left(1);
        }
        let p = [(0xf0010000, 0), (0xf0010002, 1), (0x1000, 42)];
        assert_eq!(
            plan(&p, &mut bytes, 1, 1, TableKind::Baseband)
                .unwrap()
                .selected_target(),
            0x10002
        );
        let p = [(0xf0010002, 0), (0xf0010002, 1), (0x1000, 42)];
        assert!(matches!(
            plan(&p, &mut bytes, 1, 2, TableKind::Baseband),
            Err(TableError::Selection)
        ));
    }
    #[test]
    fn invalid_programs_and_register_shapes_never_expose_a_prefix() {
        let mut bytes = [0; 64];
        for invalid in [
            0x70000000, 0xf0010000, 0x80010000, 0x90010000, 0x40000000, 0xa0000000, 0xb0000000,
            0x10001,
        ] {
            assert!(
                plan(
                    &[(0x1000, 42), (invalid, 0)],
                    &mut bytes,
                    1,
                    2,
                    TableKind::Baseband
                )
                .is_err()
            );
        }
        for (addr, val) in [(0x10100, 0), (0x20000, 0), (0x10012, 0x100000)] {
            assert!(
                plan(
                    &[(addr, val)],
                    &mut bytes,
                    1,
                    2,
                    TableKind::Radio(RfPath::A)
                )
                .is_err()
            );
        }
        // These are physical RF-v2 addresses, not the legacy RF delay opcodes.
        assert_eq!(
            plan(&[(0xfe, 1)], &mut bytes, 1, 2, TableKind::Radio(RfPath::A))
                .unwrap()
                .records()
                .count(),
            1
        );
        let p = plan(&[(0x1000, 42)], &mut bytes, 1, 2, TableKind::Baseband).unwrap();
        for len in 0..8 {
            assert!(
                RegTable::new(
                    Element {
                        id: 2,
                        version: [0; 4],
                        private: [0; 8],
                        contents: &p.bytes[..len]
                    },
                    1,
                    2,
                    TableKind::Baseband
                )
                .is_err()
            );
        }
    }
    const fn container() -> [u8; 224] {
        let mut b = [0; 224];
        b[0] = 0xff;
        b[1] = 1;
        b[20] = 32;
        let mut i = 0;
        while i < 4 {
            let at = 32 + i * 48;
            b[at] = i as u8 + 2;
            b[at + 4] = 8;
            if i == 0 {
                b[at + 33] = 0x10;
            }
            if i == 2 {
                b[at + 24] = 1;
            }
            i += 1;
        }
        b
    }
    fn parsed(b: &[u8]) -> Result<PhyTables<'_>, TableError> {
        FirmwareImage::verified(b, &super::super::Digest(*blake3::hash(b).as_bytes()))
            .unwrap()
            .phy_tables(1, 2)
    }
    #[test]
    fn physical_paths_and_firmware_slots_are_independent_and_unique() {
        let mut b = container();
        let p = parsed(&b).unwrap();
        assert_eq!(p.radio[0].kind(), TableKind::Radio(RfPath::B));
        assert_eq!(p.radio[1].kind(), TableKind::Radio(RfPath::A));
        b[32 + 3 * 48 + 24] = 1;
        assert!(matches!(parsed(&b), Err(TableError::Duplicate)));
        b = container();
        b[32 + 3 * 48] = 4;
        assert!(matches!(parsed(&b), Err(TableError::Duplicate)));
        b = container();
        b[32 + 3 * 48] = 9;
        assert!(matches!(parsed(&b), Err(TableError::Missing)));
        b = container();
        b[32 + 2 * 48 + 24] = 2;
        assert!(matches!(parsed(&b), Err(TableError::Shape)));
        b = container();
        b[32 + 48 + 33] = 2; // gain path C, not a physical path here
        assert!(matches!(parsed(&b), Err(TableError::Register)));
    }
    #[test]
    fn rf_notification_limit_is_checked_before_execution() {
        for count in [RF_WORDS_MAX, RF_WORDS_MAX + 1] {
            let mut b = [0u8; 16384];
            b[0] = 0xff;
            b[1] = 1;
            b[20] = 32;
            let mut at = 32;
            for id in 2..=5 {
                let rows = if id == 4 { count } else { 1 };
                b[at] = id;
                b[at + 4..at + 8].copy_from_slice(&((rows * 8) as u32).to_le_bytes());
                b[at + 24] = u8::from(id == 5);
                for i in 0..rows {
                    let address = if id >= 4 { 0x10000u32 } else { 0 };
                    b[at + 32 + i * 8..at + 36 + i * 8].copy_from_slice(&address.to_le_bytes());
                }
                at = (at + 32 + rows * 8 + 15) & !15;
            }
            let result = parsed(&b[..at]);
            if count == RF_WORDS_MAX {
                assert!(result.is_ok());
            } else {
                assert!(matches!(result, Err(TableError::Shape)));
            }
        }
    }

    #[test]
    fn bounded_mutation_walk_never_panics_or_exposes_unbounded_records() {
        let base = container();
        for i in 0..base.len() {
            for value in [0, 1, 0x7f, 0xff] {
                let mut b = base;
                b[i] = value;
                if let Ok(p) = parsed(&b) {
                    assert!(p.baseband.records().count() <= MAX_RECORDS);
                    for r in &p.radio {
                        assert!(r.records().count() <= MAX_RECORDS);
                    }
                }
            }
        }
    }
}
