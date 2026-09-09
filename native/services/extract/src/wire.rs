//! A compact, explicitly versioned transport envelope for normalized output.
//!
//! This is not the stable filesystem ABI. It is the bounded worker-output
//! framing used by the extraction pipeline. Parser-specific layouts never
//! cross it. Every section in [`NormalizedResult`] is represented explicitly;
//! a worker can therefore never appear to succeed after silently dropping a
//! section the parser produced.

use heapless::{String, Vec};

use crate::{
    registry::ParserId,
    schema::{
        CacheIdentity, Confidence, Diagnostic, DiagnosticCode, EmbeddedRange, Fact, FactKind,
        FactValue, InputIdentity, MAX_DIAGNOSTICS, MAX_EMBEDDED, MAX_FACTS, MAX_KEY_BYTES,
        MAX_REFERENCES, MAX_SYMBOLS, MAX_TEXT_BYTES, MAX_TEXT_SPANS, MAX_THUMBNAILS,
        MAX_VALUE_BYTES, NormalizedResult, ParserProvenance, Reference, SCHEMA_VERSION,
        SourceRange, Symbol, TextSpan, Thumbnail,
    },
};

pub const RESULT_WIRE_MAGIC: u32 = u32::from_le_bytes(*b"KEXR");
pub const RESULT_WIRE_VERSION: u16 = 2;
const HEADER_BYTES: usize = 120;
const FACT_HEAD_BYTES: usize = 28;
const RANGE_TEXT_HEAD_BYTES: usize = 20;
const THUMBNAIL_BYTES: usize = 20;
const DIAGNOSTIC_BYTES: usize = 24;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncodeError {
    Short,
    TooManyRecords,
    ValueTooLong,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeError {
    Short,
    Magic,
    Version,
    Reserved,
    Count,
    Kind,
    Value,
    Utf8,
    Range,
    Trailing,
}

fn put(dst: &mut [u8], at: &mut usize, bytes: &[u8]) -> Result<(), EncodeError> {
    let end = at.checked_add(bytes.len()).ok_or(EncodeError::Short)?;
    let slot = dst.get_mut(*at..end).ok_or(EncodeError::Short)?;
    slot.copy_from_slice(bytes);
    *at = end;
    Ok(())
}

fn take<'a>(src: &'a [u8], at: &mut usize, len: usize) -> Result<&'a [u8], DecodeError> {
    let end = at.checked_add(len).ok_or(DecodeError::Short)?;
    let bytes = src.get(*at..end).ok_or(DecodeError::Short)?;
    *at = end;
    Ok(bytes)
}

fn u16_at(src: &[u8], at: &mut usize) -> Result<u16, DecodeError> {
    Ok(u16::from_le_bytes(take(src, at, 2)?.try_into().unwrap()))
}

fn u32_at(src: &[u8], at: &mut usize) -> Result<u32, DecodeError> {
    Ok(u32::from_le_bytes(take(src, at, 4)?.try_into().unwrap()))
}

fn u64_at(src: &[u8], at: &mut usize) -> Result<u64, DecodeError> {
    Ok(u64::from_le_bytes(take(src, at, 8)?.try_into().unwrap()))
}

fn put_range_text(
    dst: &mut [u8],
    at: &mut usize,
    source: SourceRange,
    bytes: &[u8],
) -> Result<(), EncodeError> {
    if bytes.len() > u16::MAX as usize {
        return Err(EncodeError::ValueTooLong);
    }
    put(dst, at, &source.offset.to_le_bytes())?;
    put(dst, at, &source.len.to_le_bytes())?;
    put(dst, at, &(bytes.len() as u16).to_le_bytes())?;
    put(dst, at, &[0, 0])?;
    put(dst, at, bytes)
}

fn count(value: usize, maximum: usize) -> Result<u16, EncodeError> {
    if value > maximum || value > u16::MAX as usize {
        Err(EncodeError::TooManyRecords)
    } else {
        Ok(value as u16)
    }
}

/// Encode every normalized section. Records are emitted in schema order so
/// the same result has one deterministic byte representation.
pub fn encode_result(result: &NormalizedResult, dst: &mut [u8]) -> Result<usize, EncodeError> {
    let counts = [
        count(result.facts.len(), MAX_FACTS)?,
        count(result.text.len(), MAX_TEXT_SPANS)?,
        count(result.embedded.len(), MAX_EMBEDDED)?,
        count(result.references.len(), MAX_REFERENCES)?,
        count(result.symbols.len(), MAX_SYMBOLS)?,
        count(result.thumbnails.len(), MAX_THUMBNAILS)?,
        count(result.diagnostics.len(), MAX_DIAGNOSTICS)?,
    ];
    let mut at = 0;
    let p = result.provenance;
    put(dst, &mut at, &RESULT_WIRE_MAGIC.to_le_bytes())?;
    put(dst, &mut at, &RESULT_WIRE_VERSION.to_le_bytes())?;
    put(dst, &mut at, &p.schema_version.to_le_bytes())?;
    put(dst, &mut at, &p.input.source.to_le_bytes())?;
    put(dst, &mut at, &p.input.version.to_le_bytes())?;
    put(dst, &mut at, &p.input.len.to_le_bytes())?;
    put(dst, &mut at, &p.input.digest)?;
    put(dst, &mut at, &p.parser.0.to_le_bytes())?;
    put(dst, &mut at, &p.parser_version.to_le_bytes())?;
    put(dst, &mut at, &[u8::from(p.deterministic), 0, 0, 0])?;
    put(dst, &mut at, &p.cache.0)?;
    for count in counts {
        put(dst, &mut at, &count.to_le_bytes())?;
    }
    debug_assert_eq!(at, HEADER_BYTES);

    for fact in &result.facts {
        let mut scalar = [0u8; 8];
        let (tag, value): (u8, &[u8]) = match &fact.value {
            FactValue::Unsigned(value) => {
                scalar = value.to_le_bytes();
                (1, &scalar)
            }
            FactValue::Signed(value) => {
                scalar = value.to_le_bytes();
                (2, &scalar)
            }
            FactValue::Bool(value) => {
                scalar[0] = u8::from(*value);
                (3, &scalar[..1])
            }
            FactValue::Utf8(value) => (4, value.as_bytes()),
            FactValue::Bytes(value) => (5, value.as_slice()),
        };
        if fact.key.len() > u8::MAX as usize || value.len() > u16::MAX as usize {
            return Err(EncodeError::ValueTooLong);
        }
        let source = fact.source.unwrap_or(SourceRange::new(0, 0));
        let head_at = at;
        put(
            dst,
            &mut at,
            &[
                fact.kind as u8,
                tag,
                fact.key.len() as u8,
                u8::from(fact.source.is_some()),
            ],
        )?;
        put(dst, &mut at, &fact.confidence.get().to_le_bytes())?;
        put(dst, &mut at, &(value.len() as u16).to_le_bytes())?;
        put(dst, &mut at, &source.offset.to_le_bytes())?;
        put(dst, &mut at, &source.len.to_le_bytes())?;
        put(dst, &mut at, &[0, 0, 0, 0])?;
        debug_assert_eq!(at - head_at, FACT_HEAD_BYTES);
        put(dst, &mut at, fact.key.as_bytes())?;
        put(dst, &mut at, value)?;
    }
    for span in &result.text {
        put_range_text(dst, &mut at, span.source, span.text.as_bytes())?;
    }
    for embedded in &result.embedded {
        put_range_text(
            dst,
            &mut at,
            embedded.source,
            embedded.media_type.as_bytes(),
        )?;
    }
    for reference in &result.references {
        put_range_text(dst, &mut at, reference.source, reference.target.as_bytes())?;
    }
    for symbol in &result.symbols {
        put_range_text(dst, &mut at, symbol.source, symbol.name.as_bytes())?;
    }
    for thumbnail in &result.thumbnails {
        put(dst, &mut at, &thumbnail.output_offset.to_le_bytes())?;
        put(dst, &mut at, &thumbnail.output_len.to_le_bytes())?;
        put(dst, &mut at, &thumbnail.width.to_le_bytes())?;
        put(dst, &mut at, &thumbnail.height.to_le_bytes())?;
        put(dst, &mut at, &thumbnail.format.to_le_bytes())?;
        put(dst, &mut at, &[0, 0])?;
    }
    for diagnostic in &result.diagnostics {
        let source = diagnostic.source.unwrap_or(SourceRange::new(0, 0));
        put(dst, &mut at, &(diagnostic.code as u16).to_le_bytes())?;
        put(dst, &mut at, &[u8::from(diagnostic.source.is_some())])?;
        put(dst, &mut at, &[0; 5])?;
        put(dst, &mut at, &source.offset.to_le_bytes())?;
        put(dst, &mut at, &source.len.to_le_bytes())?;
    }
    Ok(at)
}

fn fact_kind(raw: u8) -> Option<FactKind> {
    Some(match raw {
        0 => FactKind::MediaType,
        1 => FactKind::Title,
        2 => FactKind::Author,
        3 => FactKind::Dimensions,
        4 => FactKind::EntryCount,
        5 => FactKind::UncompressedBytes,
        6 => FactKind::FontFamily,
        7 => FactKind::GlyphCount,
        8 => FactKind::JsonMemberCount,
        9 => FactKind::DocumentSpineCount,
        10 => FactKind::Custom,
        11 => FactKind::CsvRowCount,
        12 => FactKind::CsvColumnCount,
        13 => FactKind::TextLineCount,
        _ => return None,
    })
}

fn diagnostic_code(raw: u16) -> Option<DiagnosticCode> {
    Some(match raw {
        0 => DiagnosticCode::Truncated,
        1 => DiagnosticCode::Unsupported,
        2 => DiagnosticCode::Malformed,
        3 => DiagnosticCode::LimitReached,
        4 => DiagnosticCode::Ambiguous,
        _ => return None,
    })
}

fn take_range_text<'a>(
    src: &'a [u8],
    at: &mut usize,
    input_len: u64,
    maximum: usize,
) -> Result<(SourceRange, &'a str), DecodeError> {
    let head = take(src, at, RANGE_TEXT_HEAD_BYTES)?;
    let source = SourceRange::new(
        u64::from_le_bytes(head[0..8].try_into().unwrap()),
        u64::from_le_bytes(head[8..16].try_into().unwrap()),
    );
    let len = usize::from(u16::from_le_bytes(head[16..18].try_into().unwrap()));
    if !source.within(input_len) || len > maximum || head[18..].iter().any(|byte| *byte != 0) {
        return Err(DecodeError::Range);
    }
    core::str::from_utf8(take(src, at, len)?)
        .map(|text| (source, text))
        .map_err(|_| DecodeError::Utf8)
}

/// Decode exactly one envelope. Unknown versions and non-zero reserve bytes
/// fail closed; trailing bytes are allowed only when they are zeroed output
/// window capacity.
pub fn decode_result(src: &[u8]) -> Result<NormalizedResult, DecodeError> {
    let mut at = 0;
    if u32_at(src, &mut at)? != RESULT_WIRE_MAGIC {
        return Err(DecodeError::Magic);
    }
    if u16_at(src, &mut at)? != RESULT_WIRE_VERSION {
        return Err(DecodeError::Version);
    }
    let schema_version = u16_at(src, &mut at)?;
    if schema_version != SCHEMA_VERSION {
        return Err(DecodeError::Version);
    }
    let input = InputIdentity {
        source: u64_at(src, &mut at)?,
        version: u64_at(src, &mut at)?,
        len: u64_at(src, &mut at)?,
        digest: take(src, &mut at, 32)?.try_into().unwrap(),
    };
    let parser = ParserId(u16_at(src, &mut at)?);
    let parser_version = u32_at(src, &mut at)?;
    let flags = take(src, &mut at, 4)?;
    if flags[0] > 1 || flags[1..].iter().any(|byte| *byte != 0) {
        return Err(DecodeError::Reserved);
    }
    let cache = CacheIdentity(take(src, &mut at, 32)?.try_into().unwrap());
    let counts = core::array::from_fn::<_, 7, _>(|_| u16_at(src, &mut at));
    let [
        facts,
        text,
        embedded,
        references,
        symbols,
        thumbnails,
        diagnostics,
    ] = counts;
    let counts = [
        facts?,
        text?,
        embedded?,
        references?,
        symbols?,
        thumbnails?,
        diagnostics?,
    ];
    if usize::from(counts[0]) > MAX_FACTS
        || usize::from(counts[1]) > MAX_TEXT_SPANS
        || usize::from(counts[2]) > MAX_EMBEDDED
        || usize::from(counts[3]) > MAX_REFERENCES
        || usize::from(counts[4]) > MAX_SYMBOLS
        || usize::from(counts[5]) > MAX_THUMBNAILS
        || usize::from(counts[6]) > MAX_DIAGNOSTICS
    {
        return Err(DecodeError::Count);
    }
    let mut result = NormalizedResult::empty(ParserProvenance {
        input,
        parser,
        parser_version,
        schema_version,
        deterministic: flags[0] == 1,
        cache,
    });
    for _ in 0..counts[0] {
        let head = take(src, &mut at, FACT_HEAD_BYTES)?;
        let kind = fact_kind(head[0]).ok_or(DecodeError::Kind)?;
        let tag = head[1];
        let key_len = usize::from(head[2]);
        let has_source = match head[3] {
            0 => false,
            1 => true,
            _ => return Err(DecodeError::Reserved),
        };
        let confidence = Confidence::new(u16::from_le_bytes(head[4..6].try_into().unwrap()))
            .ok_or(DecodeError::Value)?;
        let value_len = usize::from(u16::from_le_bytes(head[6..8].try_into().unwrap()));
        let source = SourceRange::new(
            u64::from_le_bytes(head[8..16].try_into().unwrap()),
            u64::from_le_bytes(head[16..24].try_into().unwrap()),
        );
        if head[24..].iter().any(|byte| *byte != 0)
            || key_len > MAX_KEY_BYTES
            || value_len > MAX_VALUE_BYTES
        {
            return Err(DecodeError::Reserved);
        }
        let key =
            core::str::from_utf8(take(src, &mut at, key_len)?).map_err(|_| DecodeError::Utf8)?;
        let key = String::try_from(key).map_err(|_| DecodeError::Value)?;
        let bytes = take(src, &mut at, value_len)?;
        let value = match tag {
            1 if bytes.len() == 8 => {
                FactValue::Unsigned(u64::from_le_bytes(bytes.try_into().unwrap()))
            }
            2 if bytes.len() == 8 => {
                FactValue::Signed(i64::from_le_bytes(bytes.try_into().unwrap()))
            }
            3 if bytes.len() == 1 && bytes[0] <= 1 => FactValue::Bool(bytes[0] == 1),
            4 => FactValue::Utf8(
                String::try_from(core::str::from_utf8(bytes).map_err(|_| DecodeError::Utf8)?)
                    .map_err(|_| DecodeError::Value)?,
            ),
            5 => FactValue::Bytes(Vec::from_slice(bytes).map_err(|_| DecodeError::Value)?),
            _ => return Err(DecodeError::Value),
        };
        let source = has_source.then_some(source);
        if source.is_some_and(|range| !range.within(input.len)) {
            return Err(DecodeError::Range);
        }
        result
            .facts
            .push(Fact {
                kind,
                key,
                value,
                confidence,
                source,
            })
            .map_err(|_| DecodeError::Count)?;
    }
    for _ in 0..counts[1] {
        let (source, text) = take_range_text(src, &mut at, input.len, MAX_TEXT_BYTES)?;
        result
            .text
            .push(TextSpan {
                source,
                text: String::try_from(text).map_err(|_| DecodeError::Value)?,
            })
            .map_err(|_| DecodeError::Count)?;
    }
    for _ in 0..counts[2] {
        let (source, media_type) = take_range_text(src, &mut at, input.len, MAX_VALUE_BYTES)?;
        result
            .embedded
            .push(EmbeddedRange {
                source,
                media_type: String::try_from(media_type).map_err(|_| DecodeError::Value)?,
            })
            .map_err(|_| DecodeError::Count)?;
    }
    for _ in 0..counts[3] {
        let (source, target) = take_range_text(src, &mut at, input.len, MAX_VALUE_BYTES)?;
        result
            .references
            .push(Reference {
                source,
                target: String::try_from(target).map_err(|_| DecodeError::Value)?,
            })
            .map_err(|_| DecodeError::Count)?;
    }
    for _ in 0..counts[4] {
        let (source, name) = take_range_text(src, &mut at, input.len, MAX_VALUE_BYTES)?;
        result
            .symbols
            .push(Symbol {
                source,
                name: String::try_from(name).map_err(|_| DecodeError::Value)?,
            })
            .map_err(|_| DecodeError::Count)?;
    }
    for _ in 0..counts[5] {
        let record = take(src, &mut at, THUMBNAIL_BYTES)?;
        if record[18..].iter().any(|byte| *byte != 0) {
            return Err(DecodeError::Reserved);
        }
        let thumbnail = Thumbnail {
            output_offset: u32::from_le_bytes(record[0..4].try_into().unwrap()),
            output_len: u32::from_le_bytes(record[4..8].try_into().unwrap()),
            width: u32::from_le_bytes(record[8..12].try_into().unwrap()),
            height: u32::from_le_bytes(record[12..16].try_into().unwrap()),
            format: u16::from_le_bytes(record[16..18].try_into().unwrap()),
        };
        result
            .thumbnails
            .push(thumbnail)
            .map_err(|_| DecodeError::Count)?;
    }
    for _ in 0..counts[6] {
        let record = take(src, &mut at, DIAGNOSTIC_BYTES)?;
        let code = diagnostic_code(u16::from_le_bytes(record[0..2].try_into().unwrap()))
            .ok_or(DecodeError::Kind)?;
        let has_source = match record[2] {
            0 => false,
            1 => true,
            _ => return Err(DecodeError::Reserved),
        };
        if record[3..8].iter().any(|byte| *byte != 0) {
            return Err(DecodeError::Reserved);
        }
        let source = SourceRange::new(
            u64::from_le_bytes(record[8..16].try_into().unwrap()),
            u64::from_le_bytes(record[16..24].try_into().unwrap()),
        );
        if has_source && !source.within(input.len) {
            return Err(DecodeError::Range);
        }
        result
            .diagnostics
            .push(Diagnostic {
                code,
                source: has_source.then_some(source),
            })
            .map_err(|_| DecodeError::Count)?;
    }
    if src[at..].iter().any(|byte| *byte != 0) {
        return Err(DecodeError::Trailing);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{ParserProvenance, ResultBuilder};

    fn fixture() -> NormalizedResult {
        let input = InputIdentity::of(7, 3, br#"{"a":1}"#);
        let mut out = ResultBuilder::new(ParserProvenance::deterministic(input, ParserId(1), 4));
        out.fact(Fact {
            kind: FactKind::JsonMemberCount,
            key: String::try_from("root_members").unwrap(),
            value: FactValue::Unsigned(1),
            confidence: Confidence::CERTAIN,
            source: Some(SourceRange::new(0, input.len)),
        })
        .unwrap();
        out.finish()
    }

    #[test]
    fn round_trip_is_deterministic_and_zero_tail_tolerant() {
        let expected = fixture();
        let mut bytes = [0u8; 4096];
        let used = encode_result(&expected, &mut bytes).unwrap();
        assert!(used < bytes.len());
        assert_eq!(decode_result(&bytes).unwrap(), expected);
        let mut second = [0u8; 4096];
        assert_eq!(encode_result(&expected, &mut second).unwrap(), used);
        assert_eq!(bytes, second);
    }

    #[test]
    fn corruption_and_nonzero_tail_fail_closed() {
        let mut bytes = [0u8; 4096];
        let used = encode_result(&fixture(), &mut bytes).unwrap();
        bytes[0] ^= 1;
        assert_eq!(decode_result(&bytes), Err(DecodeError::Magic));
        bytes[0] ^= 1;
        bytes[used] = 1;
        assert_eq!(decode_result(&bytes), Err(DecodeError::Trailing));
    }

    #[test]
    fn every_section_round_trips_without_loss() {
        let mut result = fixture();
        let input = result.provenance.input;
        result
            .text
            .push(TextSpan {
                source: SourceRange::new(0, 4),
                text: String::try_from("text").unwrap(),
            })
            .unwrap();
        result
            .embedded
            .push(EmbeddedRange {
                source: SourceRange::new(0, input.len),
                media_type: String::try_from("application/json").unwrap(),
            })
            .unwrap();
        result
            .references
            .push(Reference {
                source: SourceRange::new(1, 2),
                target: String::try_from("urn:test").unwrap(),
            })
            .unwrap();
        result
            .symbols
            .push(Symbol {
                source: SourceRange::new(2, 1),
                name: String::try_from("symbol").unwrap(),
            })
            .unwrap();
        result
            .thumbnails
            .push(Thumbnail {
                output_offset: 128,
                output_len: 64,
                width: 8,
                height: 8,
                format: 1,
            })
            .unwrap();
        result
            .diagnostics
            .push(Diagnostic {
                code: DiagnosticCode::Malformed,
                source: Some(SourceRange::new(0, 1)),
            })
            .unwrap();
        let mut bytes = [0u8; 4096];
        let used = encode_result(&result, &mut bytes).unwrap();
        assert_eq!(decode_result(&bytes[..used]).unwrap(), result);
    }
}
