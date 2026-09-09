//! Bounded, metadata-only SFNT reader maintained with the extraction worker.
//!
//! Header checks are derived from ttf-parser 0.25.1's `RawFace`, `head`,
//! `hhea` and `maxp` readers; see ../TTF_PROVENANCE.md and ../LICENSE-TTF-MIT.
//! This is not a shaping/outline parser or a claim to validate a whole font.
//! No optional table is interpreted, no recursion or allocation is possible.
#![forbid(unsafe_code)]

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Error {
    Malformed,
    MemberLimit,
}

fn range(bytes: &[u8], offset: usize, length: usize) -> Result<&[u8], Error> {
    bytes
        .get(offset..offset.checked_add(length).ok_or(Error::Malformed)?)
        .ok_or(Error::Malformed)
}

fn u16_at(bytes: &[u8], offset: usize) -> Result<u16, Error> {
    Ok(u16::from_be_bytes(
        range(bytes, offset, 2)?.try_into().unwrap(),
    ))
}

fn u32_at(bytes: &[u8], offset: usize) -> Result<u32, Error> {
    Ok(u32::from_be_bytes(
        range(bytes, offset, 4)?.try_into().unwrap(),
    ))
}

fn sfnt_magic(bytes: &[u8], offset: usize) -> Result<(), Error> {
    match range(bytes, offset, 4)? {
        [0, 1, 0, 0] | b"true" | b"OTTO" => Ok(()),
        _ => Err(Error::Malformed),
    }
}

/// Inspect the first face, preserving the adapter's existing collection policy.
/// The same declared member budget bounds both collection and table directories.
/// All offsets are file-relative, including tables shared by collection faces.
pub(crate) fn glyph_count(bytes: &[u8], member_limit: u16) -> Result<u16, Error> {
    let face = if range(bytes, 0, 4)? == b"ttcf" {
        if !matches!(u32_at(bytes, 4)?, 0x0001_0000 | 0x0002_0000) {
            return Err(Error::Malformed);
        }
        let faces = u32_at(bytes, 8)?;
        if faces == 0 {
            return Err(Error::Malformed);
        }
        if faces > u32::from(member_limit) {
            return Err(Error::MemberLimit);
        }
        let directory_end = 12 + faces as usize * 4;
        range(bytes, 0, directory_end)?;
        let face = usize::try_from(u32_at(bytes, 12)?).map_err(|_| Error::Malformed)?;
        if face < directory_end {
            return Err(Error::Malformed);
        }
        face
    } else {
        0
    };
    sfnt_magic(bytes, face)?;
    let header = range(bytes, face, 12)?;
    let count = u16_at(header, 4)?;
    if count > member_limit {
        return Err(Error::MemberLimit);
    }
    let records = range(bytes, face + 12, usize::from(count) * 16)?;
    let mut head = None;
    let mut hhea = None;
    let mut maxp = None;
    let mut previous = None;
    for record in records.chunks_exact(16) {
        let tag = u32_at(record, 0)?;
        // SFNT requires a sorted, unique directory. Fail closed on ambiguous
        // required tables rather than selecting a duplicate by accident.
        if previous.is_some_and(|old| old >= tag) {
            return Err(Error::Malformed);
        }
        previous = Some(tag);
        let offset = usize::try_from(u32_at(record, 8)?).map_err(|_| Error::Malformed)?;
        let length = usize::try_from(u32_at(record, 12)?).map_err(|_| Error::Malformed)?;
        let table = range(bytes, offset, length)?;
        match &record[..4] {
            b"head" => head = Some(table),
            b"hhea" => hhea = Some(table),
            b"maxp" => maxp = Some(table),
            _ => {}
        }
    }
    let head = head.ok_or(Error::Malformed)?;
    range(head, 0, 54)?;
    if !(16..=16384).contains(&u16_at(head, 18)?) || u16_at(head, 50)? > 1 {
        return Err(Error::Malformed);
    }
    // Preserve the previous face reader's required hhea table, even though
    // glyph-count extraction does not consume its metrics.
    range(hhea.ok_or(Error::Malformed)?, 0, 36)?;
    let maxp = maxp.ok_or(Error::Malformed)?;
    if !matches!(u32_at(maxp, 0)?, 0x0000_5000 | 0x0001_0000) {
        return Err(Error::Malformed);
    }
    match u16_at(maxp, 4)? {
        0 => Err(Error::Malformed),
        glyphs => Ok(glyphs),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Three sorted required tables; padding belongs to each table's own range.
    fn fixture() -> [u8; 156] {
        let mut font = [0; 156];
        font[..4].copy_from_slice(&[0, 1, 0, 0]);
        font[4..6].copy_from_slice(&3u16.to_be_bytes());
        for (index, (tag, offset, length)) in [
            (*b"head", 60u32, 54u32),
            (*b"hhea", 114, 36),
            (*b"maxp", 150, 6),
        ]
        .into_iter()
        .enumerate()
        {
            let at = 12 + index * 16;
            font[at..at + 4].copy_from_slice(&tag);
            font[at + 8..at + 12].copy_from_slice(&offset.to_be_bytes());
            font[at + 12..at + 16].copy_from_slice(&length.to_be_bytes());
        }
        font[78..80].copy_from_slice(&1000u16.to_be_bytes());
        font[150..154].copy_from_slice(&0x0001_0000u32.to_be_bytes());
        font[154..].copy_from_slice(&3u16.to_be_bytes());
        font
    }

    #[test]
    fn sfnt_flavors_preserve_exact_glyph_counts() {
        for magic in [[0, 1, 0, 0], *b"true", *b"OTTO"] {
            let mut font = fixture();
            font[..4].copy_from_slice(&magic);
            assert_eq!(glyph_count(&font, 64), Ok(3));
        }
    }

    #[test]
    fn first_collection_face_uses_file_relative_table_offsets() {
        let mut collection = [0; 172];
        collection[..4].copy_from_slice(b"ttcf");
        collection[4..8].copy_from_slice(&0x0001_0000u32.to_be_bytes());
        collection[8..12].copy_from_slice(&1u32.to_be_bytes());
        collection[12..16].copy_from_slice(&16u32.to_be_bytes());
        collection[16..].copy_from_slice(&fixture());
        for (index, offset) in [76u32, 130, 166].into_iter().enumerate() {
            collection[36 + index * 16..40 + index * 16].copy_from_slice(&offset.to_be_bytes());
        }
        assert_eq!(glyph_count(&collection, 64), Ok(3));
        collection[16..20].copy_from_slice(b"ttcf");
        assert_eq!(glyph_count(&collection, 64), Err(Error::Malformed));
    }

    #[test]
    fn truncation_invalid_required_headers_and_ambiguous_directories_fail_closed() {
        let font = fixture();
        for end in 0..font.len() {
            assert!(glyph_count(&font[..end], 64).is_err());
        }
        for (at, bytes) in [
            (78, &[0, 0][..]),
            (110, &[0, 2]),
            (150, &[0, 0, 0, 0]),
            (154, &[0, 0]),
            (12, b"hhea"),
            (20, &[255; 4]),
            (24, &[255; 4]),
        ] {
            let mut bad = font;
            bad[at..at + bytes.len()].copy_from_slice(bytes);
            assert_eq!(glyph_count(&bad, 64), Err(Error::Malformed));
        }
        assert_eq!(glyph_count(&font, 2), Err(Error::MemberLimit));
    }
}
