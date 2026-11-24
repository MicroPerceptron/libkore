//! Firmware element framing (Linux v6.17 fw.h `rtw89_fw_element_hdr`,
//! fw.c `rtw89_fw_recognize_elements`). Contents are outside the MFW suits,
//! aligned to 16 bytes; the 32-byte header is excluded from `size`.
use super::{FirmwareImage, FwError, MFW_HDR_BYTES, MFW_INFO_BYTES, Suit, le32};

#[derive(Clone, Copy)]
pub struct Element<'a> {
    pub id: u32,
    pub version: [u8; 4],
    pub private: [u8; 8],
    pub contents: &'a [u8],
}
#[derive(Clone, Copy)]
pub struct Elements<'a> {
    bytes: &'a [u8],
    offset: usize,
}
impl<'a> FirmwareImage<'a> {
    /// Validate every MFW extent and every trailing element before exposing
    /// one element. A valid BB-MCU prefix with a truncated tail is refused.
    pub fn elements(&self) -> Result<Elements<'a>, FwError> {
        let b = self.blob;
        if b.len() < MFW_HDR_BYTES || b[0] != 0xff {
            return Err(FwError::NoEntries);
        }
        let count = b[1] as usize;
        if count == 0 {
            return Err(FwError::NoEntries);
        }
        let table_end = MFW_HDR_BYTES + count * MFW_INFO_BYTES;
        if table_end > b.len() {
            return Err(FwError::Truncated);
        }
        let mut end = table_end;
        for i in 0..count {
            let at = MFW_HDR_BYTES + i * MFW_INFO_BYTES;
            let start = le32(b, at + 4)? as usize;
            let size = le32(b, at + 8)? as usize;
            let limit = start.checked_add(size).ok_or(FwError::Truncated)?;
            if start < table_end || limit > b.len() {
                return Err(FwError::BadSize);
            }
            end = end.max(limit);
        }
        let offset = align(end)?.min(b.len());
        let elements = Elements { bytes: b, offset };
        let mut cursor = offset;
        while cursor < b.len() {
            let (_, next) = take(b, cursor)?;
            cursor = next;
        }
        Ok(elements)
    }

    /// The highest BB-MCU0 cut no newer than the silicon. Unlike upstream's
    /// first-match walk this does not depend on records being sorted. Duplicate
    /// records at the selected cut are ambiguous and refused.
    pub fn bbmcu0(&self, cut: u8) -> Result<Suit<'a>, FwError> {
        let mut best: Option<Element<'a>> = None;
        let mut duplicate = false;
        for e in self.elements()? {
            if e.id != 0 || e.private[0] > cut {
                continue;
            }
            match best {
                Some(old) if old.private[0] == e.private[0] => duplicate = true,
                Some(old) if old.private[0] > e.private[0] => {}
                _ => {
                    best = Some(e);
                    duplicate = false;
                }
            }
        }
        if duplicate {
            return Err(FwError::BadSize);
        }
        Suit::new(best.ok_or(FwError::NoImageForCut)?.contents)
    }
}
fn align(n: usize) -> Result<usize, FwError> {
    Ok(n.checked_add(15).ok_or(FwError::Truncated)? & !15)
}
fn take(b: &[u8], at: usize) -> Result<(Element<'_>, usize), FwError> {
    let head_end = at.checked_add(32).ok_or(FwError::Truncated)?;
    let h = b.get(at..head_end).ok_or(FwError::Truncated)?;
    let end = head_end
        .checked_add(le32(h, 4)? as usize)
        .ok_or(FwError::Truncated)?;
    let contents = b.get(head_end..end).ok_or(FwError::Truncated)?;
    let mut version = [0; 4];
    version.copy_from_slice(&h[8..12]);
    let mut private = [0; 8];
    private.copy_from_slice(&h[24..32]);
    Ok((
        Element {
            id: le32(h, 0)?,
            version,
            private,
            contents,
        },
        align(end)?.min(b.len()),
    ))
}
impl<'a> Iterator for Elements<'a> {
    type Item = Element<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.offset == self.bytes.len() {
            return None;
        }
        let (e, next) = take(self.bytes, self.offset).ok()?;
        self.offset = next;
        Some(e)
    }
}
impl core::iter::FusedIterator for Elements<'_> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fw::Digest;
    fn fixture() -> [u8; 144] {
        let mut b = [0; 144];
        b[0] = 0xff;
        b[1] = 1;
        b[20..24].copy_from_slice(&32u32.to_le_bytes());
        b[24..28].copy_from_slice(&16u32.to_le_bytes());
        // Two 16-byte suits, each with a 32-byte element header.
        for (at, cv) in [(48, 1), (96, 2)] {
            b[at + 4..at + 8].copy_from_slice(&16u32.to_le_bytes());
            b[at + 24] = cv;
            b[at + 36] = cv; // suit version.major
        }
        b
    }
    fn verified(b: &[u8]) -> FirmwareImage<'_> {
        FirmwareImage::verified(b, &Digest(*blake3::hash(b).as_bytes())).unwrap()
    }
    #[test]
    fn selects_cut_and_validates_the_entire_tail() {
        let b = fixture();
        let fw = verified(&b);
        assert_eq!(fw.elements().unwrap().count(), 2);
        assert_eq!(fw.bbmcu0(1).unwrap().version().major, 1);
        assert_eq!(fw.bbmcu0(2).unwrap().version().major, 2);
        assert!(fw.bbmcu0(0).is_err());
        for len in 97..144 {
            assert!(verified(&b[..len]).elements().is_err());
        }
        let mut b = b;
        b[100..104].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(verified(&b).bbmcu0(1).is_err());
    }
    #[test]
    fn duplicate_selected_cut_and_invalid_extents_are_refused() {
        let mut b = fixture();
        b[120] = 1;
        assert!(verified(&b).bbmcu0(1).is_err());
        // Ambiguity at an older, unselected cut must not depend on ordering.
        let mut three = [0; 192];
        three[..144].copy_from_slice(&b);
        three[144..].copy_from_slice(&fixture()[96..]);
        assert_eq!(verified(&three).bbmcu0(2).unwrap().version().major, 2);
        assert!(verified(&three).bbmcu0(1).is_err());
        three[48..96].copy_from_slice(&fixture()[96..]);
        three[144..].copy_from_slice(&b[48..96]);
        assert_eq!(verified(&three).bbmcu0(2).unwrap().version().major, 2);
        for (at, value) in [(20, 0), (20, u32::MAX), (24, u32::MAX)] {
            let mut b = fixture();
            b[at..at + 4].copy_from_slice(&value.to_le_bytes());
            assert!(verified(&b).elements().is_err());
        }
    }
    #[test]
    fn unknown_element_and_unaligned_final_payload_remain_borrowed() {
        let mut b = fixture();
        b[96..100].copy_from_slice(&999u32.to_le_bytes());
        b[100..104].copy_from_slice(&3u32.to_le_bytes());
        let fw = verified(&b[..131]);
        let e = fw.elements().unwrap().nth(1).unwrap();
        assert_eq!(e.id, 999);
        assert_eq!(e.contents.as_ptr(), b[128..].as_ptr());
        assert_eq!(e.contents.len(), 3);
    }
}
