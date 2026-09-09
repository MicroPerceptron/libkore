//! Initial bounded pure-Rust format adapters.

use core_json::{ConstStack, Deserializer, Type};
use heapless::{String, Vec};
use tar_no_std::TarArchiveRef;
use tinybmp::RawBmp;
use tinyzip::{Archive, Compression};
use ttf_parser::Face;
use xmlparser::{ElementEnd, Token, Tokenizer};

use crate::{
    registry::{BMP, CSV, EPUB, PLAIN_TEXT, RUST_JSON, Registry, TAR, TTF, XML},
    schema::{
        Confidence, Fact, FactKind, FactValue, InputIdentity, NormalizedResult, ParserProvenance,
        ResultBuilder, SchemaError, SourceRange, TextSpan,
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdapterError {
    WrongFormat,
    Malformed,
    Truncated,
    InputLimit,
    RecursionLimit,
    MemberLimit,
    ExpansionLimit,
    DimensionLimit,
    OutputLimit,
}

impl From<SchemaError> for AdapterError {
    fn from(_: SchemaError) -> Self {
        Self::OutputLimit
    }
}

fn declaration(id: crate::registry::ParserId) -> &'static crate::registry::ParserDeclaration {
    Registry::get(id).expect("built-in adapter declaration")
}

fn start(input: InputIdentity, parser: crate::registry::ParserId) -> ResultBuilder {
    let declaration = declaration(parser);
    ResultBuilder::new(ParserProvenance::deterministic(
        input,
        parser,
        declaration.version,
    ))
}

fn text<const N: usize>(value: &str) -> Result<String<N>, AdapterError> {
    String::try_from(value).map_err(|_| AdapterError::OutputLimit)
}

fn media(builder: &mut ResultBuilder, value: &str) -> Result<(), AdapterError> {
    builder.fact(Fact {
        kind: FactKind::MediaType,
        key: text("media_type")?,
        value: FactValue::Utf8(text(value)?),
        confidence: Confidence::CERTAIN,
        source: None,
    })?;
    Ok(())
}

fn unsigned(
    builder: &mut ResultBuilder,
    kind: FactKind,
    key: &str,
    value: u64,
    source: Option<SourceRange>,
) -> Result<(), AdapterError> {
    builder.fact(Fact {
        kind,
        key: text(key)?,
        value: FactValue::Unsigned(value),
        confidence: Confidence::CERTAIN,
        source,
    })?;
    Ok(())
}

fn check_input(
    id: InputIdentity,
    bytes: &[u8],
    parser: crate::registry::ParserId,
) -> Result<(), AdapterError> {
    let limit = declaration(parser).limits.max_input;
    if id.len != bytes.len() as u64
        || id.len > limit
        || *blake3::hash(bytes).as_bytes() != id.digest
    {
        return Err(AdapterError::InputLimit);
    }
    Ok(())
}

fn rust_json_root_members(bytes: &[u8]) -> Result<u64, AdapterError> {
    let mut deserializer =
        Deserializer::<_, ConstStack<4>>::new(bytes).map_err(|_| AdapterError::Malformed)?;
    let mut value = deserializer.value().map_err(|_| AdapterError::Malformed)?;
    match value.kind().map_err(|_| AdapterError::Malformed)? {
        Type::Object => {
            let mut count = 0u64;
            let mut fields = value.fields().map_err(|_| AdapterError::Malformed)?;
            while let Some(field) = fields.next() {
                let field = field.map_err(|_| AdapterError::Malformed)?;
                drop(field.value());
                count = count.checked_add(1).ok_or(AdapterError::OutputLimit)?;
            }
            Ok(count)
        }
        Type::Array => {
            let mut count = 0u64;
            let mut items = value.iterate().map_err(|_| AdapterError::Malformed)?;
            while let Some(item) = items.next() {
                drop(item.map_err(|_| AdapterError::Malformed)?);
                count = count.checked_add(1).ok_or(AdapterError::OutputLimit)?;
            }
            Ok(count)
        }
        _ => Err(AdapterError::WrongFormat),
    }
}

pub fn parse_json_rust(
    input: InputIdentity,
    bytes: &[u8],
) -> Result<NormalizedResult, AdapterError> {
    check_input(input, bytes, RUST_JSON)?;
    let members = rust_json_root_members(bytes)?;
    let mut builder = start(input, RUST_JSON);
    media(&mut builder, "application/json")?;
    unsigned(
        &mut builder,
        FactKind::JsonMemberCount,
        "root_members",
        members,
        Some(SourceRange::new(0, input.len)),
    )?;
    Ok(builder.finish())
}

/// Parse a bounded UTF-8 comma-separated table.
///
/// Quoted fields and doubled quote escapes are accepted; embedded newlines in
/// quoted fields are counted as field bytes rather than rows. The adapter
/// computes only structural facts and never evaluates spreadsheet formulas.
pub fn parse_csv(input: InputIdentity, bytes: &[u8]) -> Result<NormalizedResult, AdapterError> {
    check_input(input, bytes, CSV)?;
    core::str::from_utf8(bytes).map_err(|_| AdapterError::Malformed)?;
    if bytes.is_empty()
        || bytes.iter().any(|byte| {
            *byte == 0 || (*byte < b' ' && !matches!(*byte, b'\t' | b'\r' | b'\n')) || *byte == 0x7f
        })
    {
        return Err(AdapterError::WrongFormat);
    }

    let limits = declaration(CSV).limits;
    let mut quoted = false;
    let mut after_quote = false;
    let mut field_start = true;
    let mut columns = 1u64;
    let mut expected_columns = 0u64;
    let mut rows = 0u64;
    let mut index = 0usize;
    let mut row_has_bytes = false;
    while index < bytes.len() {
        let byte = bytes[index];
        if quoted {
            if byte == b'"' {
                if bytes.get(index + 1) == Some(&b'"') {
                    index += 2;
                    row_has_bytes = true;
                    continue;
                }
                quoted = false;
                after_quote = true;
            } else {
                row_has_bytes = true;
            }
            index += 1;
            continue;
        }
        if after_quote && !matches!(byte, b',' | b'\r' | b'\n') {
            return Err(AdapterError::Malformed);
        }
        match byte {
            b'"' if field_start => {
                quoted = true;
                field_start = false;
                row_has_bytes = true;
            }
            b'"' => return Err(AdapterError::Malformed),
            b',' => {
                columns = columns.checked_add(1).ok_or(AdapterError::MemberLimit)?;
                if columns > u64::from(limits.max_members) {
                    return Err(AdapterError::MemberLimit);
                }
                field_start = true;
                after_quote = false;
                row_has_bytes = true;
            }
            b'\r' => {
                if bytes.get(index + 1) != Some(&b'\n') {
                    return Err(AdapterError::Malformed);
                }
                index += 1;
                finish_csv_row(
                    columns,
                    &mut expected_columns,
                    &mut rows,
                    limits.max_members,
                )?;
                columns = 1;
                field_start = true;
                after_quote = false;
                row_has_bytes = false;
            }
            b'\n' => {
                finish_csv_row(
                    columns,
                    &mut expected_columns,
                    &mut rows,
                    limits.max_members,
                )?;
                columns = 1;
                field_start = true;
                after_quote = false;
                row_has_bytes = false;
            }
            _ => {
                if after_quote {
                    return Err(AdapterError::Malformed);
                }
                field_start = false;
                row_has_bytes = true;
            }
        }
        index += 1;
    }
    if quoted {
        return Err(AdapterError::Truncated);
    }
    if row_has_bytes || columns != 1 || after_quote {
        finish_csv_row(
            columns,
            &mut expected_columns,
            &mut rows,
            limits.max_members,
        )?;
    }
    if rows == 0 || expected_columns < 2 {
        return Err(AdapterError::WrongFormat);
    }
    let mut builder = start(input, CSV);
    media(&mut builder, "text/csv")?;
    unsigned(&mut builder, FactKind::CsvRowCount, "rows", rows, None)?;
    unsigned(
        &mut builder,
        FactKind::CsvColumnCount,
        "columns",
        expected_columns,
        None,
    )?;
    Ok(builder.finish())
}

fn finish_csv_row(
    columns: u64,
    expected: &mut u64,
    rows: &mut u64,
    limit: u16,
) -> Result<(), AdapterError> {
    if *expected == 0 {
        *expected = columns;
    } else if *expected != columns {
        return Err(AdapterError::Malformed);
    }
    *rows = rows.checked_add(1).ok_or(AdapterError::MemberLimit)?;
    if *rows > u64::from(limit) {
        return Err(AdapterError::MemberLimit);
    }
    Ok(())
}

/// Parse bounded plain UTF-8 into parser-neutral text spans.
pub fn parse_plain_text(
    input: InputIdentity,
    bytes: &[u8],
) -> Result<NormalizedResult, AdapterError> {
    check_input(input, bytes, PLAIN_TEXT)?;
    let value = core::str::from_utf8(bytes).map_err(|_| AdapterError::Malformed)?;
    if bytes.is_empty()
        || bytes.iter().any(|byte| {
            *byte == 0 || (*byte < b' ' && !matches!(*byte, b'\t' | b'\r' | b'\n')) || *byte == 0x7f
        })
    {
        return Err(AdapterError::WrongFormat);
    }
    let limits = declaration(PLAIN_TEXT).limits;
    let mut builder = start(input, PLAIN_TEXT);
    media(&mut builder, "text/plain; charset=utf-8")?;
    let mut lines = 0u64;
    let mut offset = 0usize;
    while offset < value.len() {
        let remainder = &value[offset..];
        let newline = remainder.find('\n');
        let end = newline.map_or(value.len(), |at| offset + at);
        let mut line_end = end;
        if line_end > offset && bytes[line_end - 1] == b'\r' {
            line_end -= 1;
        }
        let line = &value[offset..line_end];
        lines = lines.checked_add(1).ok_or(AdapterError::MemberLimit)?;
        if lines > u64::from(limits.max_members) {
            return Err(AdapterError::MemberLimit);
        }
        builder.text(TextSpan {
            source: SourceRange::new(offset as u64, (line_end - offset) as u64),
            text: text(line)?,
        })?;
        offset = if newline.is_some() {
            end + 1
        } else {
            value.len()
        };
    }
    unsigned(&mut builder, FactKind::TextLineCount, "lines", lines, None)?;
    Ok(builder.finish())
}

#[cfg(feature = "c-json")]
unsafe extern "C" {
    fn kore_jsmn_summary(
        bytes: *const u8,
        len: usize,
        root_members: *mut u32,
        token_count: *mut u32,
    ) -> i32;
}

#[cfg(feature = "c-json")]
pub fn parse_json_c(input: InputIdentity, bytes: &[u8]) -> Result<NormalizedResult, AdapterError> {
    check_input(input, bytes, C_JSON)?;
    let mut members = 0u32;
    let mut tokens = 0u32;
    // SAFETY: the adapter passes one immutable slice for the call duration and
    // two initialized out-pointers. The C wrapper retains none of them and
    // writes only these two `u32` values.
    let status =
        unsafe { kore_jsmn_summary(bytes.as_ptr(), bytes.len(), &mut members, &mut tokens) };
    if status != 0 || tokens == 0 {
        return Err(match status {
            -1 => AdapterError::OutputLimit,
            -3 => AdapterError::Truncated,
            _ => AdapterError::Malformed,
        });
    }
    let mut builder = start(input, C_JSON);
    media(&mut builder, "application/json")?;
    unsigned(
        &mut builder,
        FactKind::JsonMemberCount,
        "root_members",
        u64::from(members),
        Some(SourceRange::new(0, input.len)),
    )?;
    Ok(builder.finish())
}

pub fn parse_tar(input: InputIdentity, bytes: &[u8]) -> Result<NormalizedResult, AdapterError> {
    check_input(input, bytes, TAR)?;
    let archive = TarArchiveRef::new(bytes).map_err(|_| AdapterError::Malformed)?;
    let limits = declaration(TAR).limits;
    let mut members = 0u64;
    let mut expanded = 0u64;
    for entry in archive.entries() {
        members = members.checked_add(1).ok_or(AdapterError::MemberLimit)?;
        if members > u64::from(limits.max_members) {
            return Err(AdapterError::MemberLimit);
        }
        expanded = expanded
            .checked_add(entry.data().len() as u64)
            .ok_or(AdapterError::ExpansionLimit)?;
        if expanded
            > input
                .len
                .saturating_mul(u64::from(limits.max_expansion_ratio))
        {
            return Err(AdapterError::ExpansionLimit);
        }
    }
    let mut builder = start(input, TAR);
    media(&mut builder, "application/x-tar")?;
    unsigned(&mut builder, FactKind::EntryCount, "entries", members, None)?;
    unsigned(
        &mut builder,
        FactKind::UncompressedBytes,
        "uncompressed_bytes",
        expanded,
        None,
    )?;
    Ok(builder.finish())
}

pub fn parse_bmp(input: InputIdentity, bytes: &[u8]) -> Result<NormalizedResult, AdapterError> {
    check_input(input, bytes, BMP)?;
    let bmp = RawBmp::from_slice(bytes).map_err(|_| AdapterError::Malformed)?;
    let size = bmp.header().image_size;
    let limit = declaration(BMP).limits.max_dimension;
    if size.width == 0 || size.height == 0 || size.width > limit || size.height > limit {
        return Err(AdapterError::DimensionLimit);
    }
    let mut builder = start(input, BMP);
    media(&mut builder, "image/bmp")?;
    unsigned(
        &mut builder,
        FactKind::Dimensions,
        "width",
        u64::from(size.width),
        None,
    )?;
    unsigned(
        &mut builder,
        FactKind::Dimensions,
        "height",
        u64::from(size.height),
        None,
    )?;
    Ok(builder.finish())
}

pub fn parse_ttf(input: InputIdentity, bytes: &[u8]) -> Result<NormalizedResult, AdapterError> {
    check_input(input, bytes, TTF)?;
    let face = Face::parse(bytes, 0).map_err(|_| AdapterError::Malformed)?;
    let mut builder = start(input, TTF);
    media(&mut builder, "font/ttf")?;
    unsigned(
        &mut builder,
        FactKind::GlyphCount,
        "glyphs",
        u64::from(face.number_of_glyphs()),
        None,
    )?;
    Ok(builder.finish())
}

pub fn parse_epub(input: InputIdentity, bytes: &[u8]) -> Result<NormalizedResult, AdapterError> {
    check_input(input, bytes, EPUB)?;
    let archive = Archive::open(bytes).map_err(|_| AdapterError::Malformed)?;
    let limits = declaration(EPUB).limits;
    if archive.entry_count() > u64::from(limits.max_members) {
        return Err(AdapterError::MemberLimit);
    }
    let mut expanded = 0u64;
    let mut saw_mimetype = false;
    let mut spine = 0u64;
    let mut seen_paths: Vec<[u8; 256], 64> = Vec::new();
    let mut opf_scratch = [0u8; 4096];
    for entry in archive.entries() {
        let entry = entry.map_err(|_| AdapterError::Malformed)?;
        expanded = expanded
            .checked_add(entry.uncompressed_size())
            .ok_or(AdapterError::ExpansionLimit)?;
        if expanded
            > input
                .len
                .saturating_mul(u64::from(limits.max_expansion_ratio))
        {
            return Err(AdapterError::ExpansionLimit);
        }
        let mut path_buf = [0u8; 256];
        let path = entry
            .read_path(&mut path_buf)
            .map_err(|_| AdapterError::MemberLimit)?;
        let mut path_key = [0u8; 256];
        path_key[..path.len()].copy_from_slice(path);
        if seen_paths.iter().any(|seen| seen == &path_key) {
            return Err(AdapterError::RecursionLimit);
        }
        seen_paths
            .push(path_key)
            .map_err(|_| AdapterError::MemberLimit)?;
        if path == b"mimetype" {
            if entry.compression().map_err(|_| AdapterError::Malformed)? != Compression::Stored {
                return Err(AdapterError::WrongFormat);
            }
            let mut value = [0u8; 32];
            let value = entry
                .read_to_slice(&mut value)
                .map_err(|_| AdapterError::Malformed)?;
            saw_mimetype = value == b"application/epub+zip";
        }
        if path.ends_with(b".opf") {
            if entry.compression().map_err(|_| AdapterError::Malformed)? != Compression::Stored {
                return Err(AdapterError::WrongFormat);
            }
            let xml = entry
                .read_to_slice(&mut opf_scratch)
                .map_err(|_| AdapterError::OutputLimit)?;
            // Container XML is subject to the same balanced-nesting and
            // recursion bound as a standalone XML input. Tokenization alone
            // would otherwise accept a deeply recursive or mismatched OPF
            // while still extracting a plausible spine count.
            let embedded = InputIdentity::of(input.source, input.version, xml);
            let _ = parse_xml(embedded, xml)?;
            let xml = core::str::from_utf8(xml).map_err(|_| AdapterError::Malformed)?;
            for token in Tokenizer::from(xml) {
                match token.map_err(|_| AdapterError::Malformed)? {
                    Token::ElementStart { local, .. } if local.as_str() == "itemref" => {
                        spine = spine.checked_add(1).ok_or(AdapterError::MemberLimit)?;
                    }
                    Token::ElementEnd {
                        end: ElementEnd::Close(_, _),
                        ..
                    } if spine > u64::from(limits.max_members) => {
                        return Err(AdapterError::MemberLimit);
                    }
                    _ => {}
                }
            }
        }
    }
    if !saw_mimetype {
        return Err(AdapterError::WrongFormat);
    }
    let mut builder = start(input, EPUB);
    media(&mut builder, "application/epub+zip")?;
    unsigned(
        &mut builder,
        FactKind::EntryCount,
        "entries",
        archive.entry_count(),
        None,
    )?;
    unsigned(
        &mut builder,
        FactKind::DocumentSpineCount,
        "spine_items",
        spine,
        None,
    )?;
    Ok(builder.finish())
}

/// Parse standalone XML while independently enforcing balanced element
/// nesting. `xmlparser` is intentionally only the tokenizer; the adapter owns
/// the recursion and structural policy that crosses the worker boundary.
pub fn parse_xml(input: InputIdentity, bytes: &[u8]) -> Result<NormalizedResult, AdapterError> {
    check_input(input, bytes, XML)?;
    let text = core::str::from_utf8(bytes).map_err(|_| AdapterError::Malformed)?;
    let limits = declaration(XML).limits;
    let mut stack: Vec<String<64>, 16> = Vec::new();
    let mut pending: Option<String<64>> = None;
    let mut elements = 0u64;
    let mut roots = 0u16;
    for token in Tokenizer::from(text) {
        match token.map_err(|_| AdapterError::Malformed)? {
            Token::ElementStart { local, .. } => {
                if stack.len() >= usize::from(limits.max_recursion) {
                    return Err(AdapterError::RecursionLimit);
                }
                if stack.is_empty() {
                    roots = roots.checked_add(1).ok_or(AdapterError::Malformed)?;
                    if roots > 1 {
                        return Err(AdapterError::Malformed);
                    }
                }
                elements = elements.checked_add(1).ok_or(AdapterError::MemberLimit)?;
                if elements > u64::from(limits.max_members) {
                    return Err(AdapterError::MemberLimit);
                }
                pending =
                    Some(String::try_from(local.as_str()).map_err(|_| AdapterError::OutputLimit)?);
            }
            Token::ElementEnd {
                end: ElementEnd::Open,
                ..
            } => {
                let name = pending.take().ok_or(AdapterError::Malformed)?;
                stack.push(name).map_err(|_| AdapterError::RecursionLimit)?;
            }
            Token::ElementEnd {
                end: ElementEnd::Empty,
                ..
            } => {
                pending.take().ok_or(AdapterError::Malformed)?;
            }
            Token::ElementEnd {
                end: ElementEnd::Close(_, local),
                ..
            } => {
                if pending.is_some() {
                    return Err(AdapterError::Malformed);
                }
                let expected = stack.pop().ok_or(AdapterError::Malformed)?;
                if expected.as_str() != local.as_str() {
                    return Err(AdapterError::Malformed);
                }
            }
            _ => {}
        }
    }
    if roots != 1 || pending.is_some() || !stack.is_empty() {
        return Err(AdapterError::Malformed);
    }
    let mut builder = start(input, XML);
    media(&mut builder, "application/xml")?;
    unsigned(
        &mut builder,
        FactKind::EntryCount,
        "elements",
        elements,
        None,
    )?;
    Ok(builder.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec::Vec as StdVec;

    fn identity(bytes: &[u8]) -> InputIdentity {
        InputIdentity::of(1, 1, bytes)
    }

    #[test]
    fn rust_json_worker_produces_a_valid_normalized_fixture() {
        let bytes = b"{\"alpha\":1,\"nested\":[true,null]}";
        let rust = parse_json_rust(identity(bytes), bytes).unwrap();
        Registry::validate(&rust, identity(bytes), 4096).unwrap();
    }

    #[test]
    fn csv_handles_quotes_and_refuses_ragged_or_truncated_rows() {
        let bytes = b"name,note\r\nalpha,\"one,two\"\r\nbeta,\"said \"\"hi\"\"\"\r\n";
        let result = parse_csv(identity(bytes), bytes).unwrap();
        Registry::validate(&result, identity(bytes), 4096).unwrap();
        assert!(result.facts.iter().any(|fact| {
            fact.kind == FactKind::CsvRowCount && fact.value == FactValue::Unsigned(3)
        }));
        assert!(result.facts.iter().any(|fact| {
            fact.kind == FactKind::CsvColumnCount && fact.value == FactValue::Unsigned(2)
        }));
        for malformed in [
            &b"a,b\n1\n"[..],
            &b"a,b\n\"unterminated,2"[..],
            &b"a,b\n\"closed\"junk,2\n"[..],
        ] {
            assert!(parse_csv(identity(malformed), malformed).is_err());
        }
    }

    #[test]
    fn plain_text_emits_bounded_source_ranges_and_rejects_binary() {
        let bytes = b"alpha\r\nbeta\n";
        let result = parse_plain_text(identity(bytes), bytes).unwrap();
        Registry::validate(&result, identity(bytes), 4096).unwrap();
        assert_eq!(result.text.len(), 2);
        assert_eq!(result.text[0].text.as_str(), "alpha");
        assert_eq!(result.text[1].text.as_str(), "beta");
        assert_eq!(
            parse_plain_text(identity(b"a\0b"), b"a\0b"),
            Err(AdapterError::WrongFormat)
        );
        let oversized = [b'x'; crate::schema::MAX_TEXT_BYTES + 1];
        assert_eq!(
            parse_plain_text(identity(&oversized), &oversized),
            Err(AdapterError::OutputLimit)
        );
    }

    #[test]
    fn standalone_xml_is_balanced_bounded_and_normalized() {
        let bytes = br#"<?xml version="1.0"?><root><item/><item></item></root>"#;
        let result = parse_xml(identity(bytes), bytes).unwrap();
        Registry::validate(&result, identity(bytes), 4096).unwrap();
        for malformed in [
            &b"<root><item></root>"[..],
            &b"<a/><b/>"[..],
            &b"<root>"[..],
        ] {
            assert!(parse_xml(identity(malformed), malformed).is_err());
        }
        let deep = b"<a><b><c><d><e><f><g><h><i><j><k><l><m><n><o><p><q/></p></o></n></m></l></k></j></i></h></g></f></e></d></c></b></a>";
        assert_eq!(
            parse_xml(identity(deep), deep),
            Err(AdapterError::RecursionLimit)
        );
    }

    #[test]
    fn malformed_truncated_and_recursive_json_are_bounded_refusals() {
        for bytes in [&b"{\"a\":"[..], &b"{bad}"[..]] {
            assert!(parse_json_rust(identity(bytes), bytes).is_err());
        }
        let deep = b"[[[[[[[[[[[[[[[[[0]]]]]]]]]]]]]]]]]";
        assert!(parse_json_rust(identity(deep), deep).is_err());
    }

    fn tar_one(name: &[u8], data: &[u8]) -> StdVec<u8> {
        let mut out = std::vec![0u8; 2048];
        out[..name.len()].copy_from_slice(name);
        out[100..108].copy_from_slice(b"0000644\0");
        out[108..116].copy_from_slice(b"0000000\0");
        out[116..124].copy_from_slice(b"0000000\0");
        let size = std::format!("{:011o}\0", data.len());
        out[124..136].copy_from_slice(size.as_bytes());
        out[136..148].copy_from_slice(b"00000000000\0");
        out[148..156].fill(b' ');
        out[156] = b'0';
        out[257..263].copy_from_slice(b"ustar\0");
        out[263..265].copy_from_slice(b"00");
        let checksum: u32 = out[..512].iter().map(|byte| u32::from(*byte)).sum();
        let sum = std::format!("{:06o}\0 ", checksum);
        out[148..156].copy_from_slice(sum.as_bytes());
        out[512..512 + data.len()].copy_from_slice(data);
        out
    }

    #[test]
    fn tar_and_bmp_produce_valid_facts_and_reject_absurd_geometry() {
        let tar = tar_one(b"a.txt", b"hello");
        let result = parse_tar(identity(&tar), &tar).unwrap();
        Registry::validate(&result, identity(&tar), 4096).unwrap();

        let mut bmp = std::vec![0u8; 58];
        bmp[0..2].copy_from_slice(b"BM");
        bmp[2..6].copy_from_slice(&(58u32).to_le_bytes());
        bmp[10..14].copy_from_slice(&(54u32).to_le_bytes());
        bmp[14..18].copy_from_slice(&(40u32).to_le_bytes());
        bmp[18..22].copy_from_slice(&(1i32).to_le_bytes());
        bmp[22..26].copy_from_slice(&(1i32).to_le_bytes());
        bmp[26..28].copy_from_slice(&(1u16).to_le_bytes());
        bmp[28..30].copy_from_slice(&(24u16).to_le_bytes());
        bmp[34..38].copy_from_slice(&(4u32).to_le_bytes());
        let result = parse_bmp(identity(&bmp), &bmp).unwrap();
        Registry::validate(&result, identity(&bmp), 4096).unwrap();

        bmp[18..22].copy_from_slice(&(9000i32).to_le_bytes());
        assert!(matches!(
            parse_bmp(identity(&bmp), &bmp),
            Err(AdapterError::Malformed | AdapterError::DimensionLimit)
        ));
    }

    const DEMO_TTF: &[u8] = &[
        0x00, 0x01, 0x00, 0x00, 0x00, 0x07, 0x00, 0x40, 0x00, 0x02, 0x00, 0x30, 0x63, 0x6d, 0x61,
        0x70, 0x00, 0x09, 0x00, 0x76, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x2c, 0x67, 0x6c,
        0x79, 0x66, 0xf1, 0xcb, 0x66, 0x98, 0x00, 0x00, 0x01, 0x34, 0x00, 0x00, 0x00, 0x5c, 0x68,
        0x65, 0x61, 0x64, 0xf2, 0x35, 0xdd, 0xf8, 0x00, 0x00, 0x00, 0x7c, 0x00, 0x00, 0x00, 0x36,
        0x68, 0x68, 0x65, 0x61, 0x06, 0x61, 0x00, 0xca, 0x00, 0x00, 0x00, 0xb4, 0x00, 0x00, 0x00,
        0x24, 0x68, 0x6d, 0x74, 0x78, 0x04, 0x74, 0x00, 0x6a, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x00,
        0x00, 0x08, 0x6c, 0x6f, 0x63, 0x61, 0x00, 0x2e, 0x00, 0x14, 0x00, 0x00, 0x01, 0x2c, 0x00,
        0x00, 0x00, 0x06, 0x6d, 0x61, 0x78, 0x70, 0x00, 0x05, 0x00, 0x0b, 0x00, 0x00, 0x00, 0xd8,
        0x00, 0x00, 0x00, 0x20, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0xf5, 0x9c, 0x29,
        0x44, 0x5f, 0x0f, 0x3c, 0xf5, 0x00, 0x02, 0x03, 0xe8, 0x00, 0x00, 0x00, 0x00, 0xb4, 0x92,
        0xf4, 0x00, 0x00, 0x00, 0x00, 0x00, 0xdc, 0x2f, 0xa6, 0x5c, 0x00, 0x06, 0x00, 0x00, 0x02,
        0x58, 0x02, 0xbc, 0x00, 0x00, 0x00, 0x03, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x01, 0x00, 0x00, 0x04, 0x00, 0xfe, 0x70, 0x00, 0x00, 0x02, 0x58, 0x00, 0x06, 0xff,
        0xff, 0x02, 0x58, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x0b, 0x00,
        0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x58, 0x00, 0x64, 0x02, 0x1c, 0x00,
        0x06, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x04,
        0x00, 0x20, 0x00, 0x00, 0x00, 0x04, 0x00, 0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x41, 0xff,
        0xff, 0x00, 0x00, 0x00, 0x41, 0xff, 0xff, 0xff, 0xc0, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x14, 0x00, 0x2e, 0x00, 0x00, 0x00, 0x02, 0x00, 0x64, 0x00, 0x00, 0x02,
        0x58, 0x02, 0xbc, 0x00, 0x03, 0x00, 0x07, 0x00, 0x00, 0x33, 0x11, 0x21, 0x11, 0x25, 0x21,
        0x11, 0x21, 0x64, 0x01, 0xf4, 0xfe, 0x34, 0x01, 0xa4, 0xfe, 0x5c, 0x02, 0xbc, 0xfd, 0x44,
        0x28, 0x02, 0x6c, 0x00, 0x02, 0x00, 0x06, 0x00, 0x00, 0x02, 0x1d, 0x02, 0x90, 0x00, 0x02,
        0x00, 0x0a, 0x00, 0x00, 0x13, 0x33, 0x03, 0x01, 0x13, 0x33, 0x13, 0x23, 0x27, 0x23, 0x07,
        0xad, 0xc4, 0x63, 0xfe, 0xf8, 0xda, 0x60, 0xdd, 0x59, 0x3e, 0xef, 0x42, 0x01, 0x0b, 0x01,
        0x40, 0xfd, 0xb5, 0x02, 0x90, 0xfd, 0x70, 0xc8, 0xc8, 0x00,
    ];

    #[test]
    fn font_produces_normalized_glyph_fact() {
        let result = parse_ttf(identity(DEMO_TTF), DEMO_TTF).unwrap();
        Registry::validate(&result, identity(DEMO_TTF), 4096).unwrap();
        assert!(
            result
                .facts
                .iter()
                .any(|fact| fact.kind == FactKind::GlyphCount)
        );
    }

    fn stored_zip(entries: &[(&[u8], &[u8], u32)]) -> StdVec<u8> {
        let mut out = StdVec::new();
        let mut central = StdVec::new();
        for (name, data, advertised_uncompressed) in entries {
            let offset = out.len() as u32;
            out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
            out.extend_from_slice(&20u16.to_le_bytes());
            out.extend_from_slice(&0x0800u16.to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(&[0u8; 8]);
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&advertised_uncompressed.to_le_bytes());
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(name);
            out.extend_from_slice(data);

            central.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
            central.extend_from_slice(&20u16.to_le_bytes());
            central.extend_from_slice(&20u16.to_le_bytes());
            central.extend_from_slice(&0x0800u16.to_le_bytes());
            central.extend_from_slice(&0u16.to_le_bytes());
            central.extend_from_slice(&[0u8; 8]);
            central.extend_from_slice(&(data.len() as u32).to_le_bytes());
            central.extend_from_slice(&advertised_uncompressed.to_le_bytes());
            central.extend_from_slice(&(name.len() as u16).to_le_bytes());
            central.extend_from_slice(&[0u8; 12]);
            central.extend_from_slice(&offset.to_le_bytes());
            central.extend_from_slice(name);
        }
        let central_offset = out.len() as u32;
        let central_len = central.len() as u32;
        out.extend_from_slice(&central);
        out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
        out.extend_from_slice(&[0u8; 4]);
        out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        out.extend_from_slice(&central_len.to_le_bytes());
        out.extend_from_slice(&central_offset.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out
    }

    #[test]
    fn epub_document_container_produces_facts_and_bomb_is_refused() {
        let opf = b"<package><spine><itemref/><itemref/></spine></package>";
        let epub = stored_zip(&[
            (b"mimetype", b"application/epub+zip", 20),
            (b"OEBPS/content.opf", opf, opf.len() as u32),
        ]);
        let result = parse_epub(identity(&epub), &epub).unwrap();
        Registry::validate(&result, identity(&epub), 4096).unwrap();
        assert!(
            result
                .facts
                .iter()
                .any(|fact| fact.kind == FactKind::DocumentSpineCount)
        );

        let bomb = stored_zip(&[
            (b"mimetype", b"application/epub+zip", 20),
            (b"OEBPS/content.opf", opf, 1_000_000),
        ]);
        assert_eq!(
            parse_epub(identity(&bomb), &bomb),
            Err(AdapterError::ExpansionLimit)
        );

        let duplicate = stored_zip(&[
            (b"mimetype", b"application/epub+zip", 20),
            (b"OEBPS/content.opf", opf, opf.len() as u32),
            (b"OEBPS/content.opf", opf, opf.len() as u32),
        ]);
        assert_eq!(
            parse_epub(identity(&duplicate), &duplicate),
            Err(AdapterError::RecursionLimit)
        );

        let deep = b"<package><a><b><c><d><e><f><g><h><i><j><k><l><m><n><o><p><q/></p></o></n></m></l></k></j></i></h></g></f></e></d></c></b></a></package>";
        let recursive = stored_zip(&[
            (b"mimetype", b"application/epub+zip", 20),
            (b"OEBPS/content.opf", deep, deep.len() as u32),
        ]);
        assert_eq!(
            parse_epub(identity(&recursive), &recursive),
            Err(AdapterError::RecursionLimit)
        );
    }

    /// Checked-in deterministic mutation corpus. This complements upstream
    /// coverage-guided fuzzing and keeps every accepted adapter plus its
    /// normalized conversion on the ordinary host-test gate.
    #[test]
    fn mutation_corpus_never_escapes_adapter_or_validator_bounds() {
        fn exercise(
            seed: &[u8],
            parser: fn(InputIdentity, &[u8]) -> Result<NormalizedResult, AdapterError>,
        ) {
            for index in 0..seed.len().min(96) {
                for replacement in [0u8, 0xff, seed[index] ^ 0x80] {
                    let mut mutated = StdVec::from(seed);
                    mutated[index] = replacement;
                    let input = identity(&mutated);
                    if let Ok(result) = parser(input, &mutated) {
                        Registry::validate(&result, input, 64 * 1024)
                            .expect("accepted mutation must normalize validly");
                    }
                }
            }
            for end in 0..seed.len().min(96) {
                let truncated = &seed[..end];
                let input = identity(truncated);
                if let Ok(result) = parser(input, truncated) {
                    Registry::validate(&result, input, 64 * 1024)
                        .expect("accepted truncation must normalize validly");
                }
            }
        }

        let json = br#"{"alpha":1,"nested":[true,null]}"#;
        exercise(json, parse_json_rust);
        exercise(b"a,b\n1,2\n", parse_csv);
        exercise(b"ordinary text\nsecond line\n", parse_plain_text);

        let tar = tar_one(b"hello.txt", b"hello");
        exercise(&tar, parse_tar);

        let mut bmp = [0u8; 58];
        bmp[0..2].copy_from_slice(b"BM");
        bmp[2..6].copy_from_slice(&58u32.to_le_bytes());
        bmp[10..14].copy_from_slice(&54u32.to_le_bytes());
        bmp[14..18].copy_from_slice(&40u32.to_le_bytes());
        bmp[18..22].copy_from_slice(&1i32.to_le_bytes());
        bmp[22..26].copy_from_slice(&1i32.to_le_bytes());
        bmp[26..28].copy_from_slice(&1u16.to_le_bytes());
        bmp[28..30].copy_from_slice(&24u16.to_le_bytes());
        bmp[34..38].copy_from_slice(&4u32.to_le_bytes());
        exercise(&bmp, parse_bmp);
        exercise(DEMO_TTF, parse_ttf);

        let opf = b"<package><spine><itemref/></spine></package>";
        let epub = stored_zip(&[
            (b"mimetype", b"application/epub+zip", 20),
            (b"OEBPS/content.opf", opf, opf.len() as u32),
        ]);
        exercise(&epub, parse_epub);
    }

    #[test]
    #[ignore = "manual vendor benchmark; run with --release --ignored --nocapture"]
    fn benchmark_vendor_adapters() {
        use std::time::Instant;

        fn bench(
            name: &str,
            bytes: &[u8],
            parser: fn(InputIdentity, &[u8]) -> Result<NormalizedResult, AdapterError>,
        ) {
            const RUNS: u32 = 20_000;
            let input = identity(bytes);
            let start = Instant::now();
            for _ in 0..RUNS {
                std::hint::black_box(parser(input, std::hint::black_box(bytes))).unwrap();
            }
            let ns = start.elapsed().as_nanos() / u128::from(RUNS);
            std::println!("extract-bench {name}: {ns} ns/op ({} bytes)", bytes.len());
        }

        let json = br#"{"alpha":1,"nested":[true,null]}"#;
        bench("core-json", json, parse_json_rust);
        let xml = br#"<?xml version="1.0"?><root><item/><item/></root>"#;
        bench("xmlparser", xml, parse_xml);
        let tar = tar_one(b"hello.txt", b"hello");
        bench("tar-no-std", &tar, parse_tar);
        bench("ttf-parser", DEMO_TTF, parse_ttf);
        let mut bmp = [0u8; 58];
        bmp[0..2].copy_from_slice(b"BM");
        bmp[2..6].copy_from_slice(&58u32.to_le_bytes());
        bmp[10..14].copy_from_slice(&54u32.to_le_bytes());
        bmp[14..18].copy_from_slice(&40u32.to_le_bytes());
        bmp[18..22].copy_from_slice(&1i32.to_le_bytes());
        bmp[22..26].copy_from_slice(&1i32.to_le_bytes());
        bmp[26..28].copy_from_slice(&1u16.to_le_bytes());
        bmp[28..30].copy_from_slice(&24u16.to_le_bytes());
        bmp[34..38].copy_from_slice(&4u32.to_le_bytes());
        bench("tinybmp", &bmp, parse_bmp);
        let opf = b"<package><spine><itemref/></spine></package>";
        let epub = stored_zip(&[
            (b"mimetype", b"application/epub+zip", 20),
            (b"OEBPS/content.opf", opf, opf.len() as u32),
        ]);
        bench("tinyzip+xmlparser", &epub, parse_epub);
    }
}
