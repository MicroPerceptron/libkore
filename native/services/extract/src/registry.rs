//! Versioned parser registry, bounded content sniffing, and result validation.

use crate::schema::{FactKind, InputIdentity, NormalizedResult, ParserProvenance, SCHEMA_VERSION};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ParserId(pub u16);

pub const RUST_JSON: ParserId = ParserId(1);
pub const C_JSON: ParserId = ParserId(2);
pub const TAR: ParserId = ParserId(3);
pub const BMP: ParserId = ParserId(4);
pub const TTF: ParserId = ParserId(5);
pub const EPUB: ParserId = ParserId(6);
pub const XML: ParserId = ParserId(7);
pub const CSV: ParserId = ParserId(8);
pub const PLAIN_TEXT: ParserId = ParserId(9);

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Format {
    Json,
    Tar,
    Bmp,
    Ttf,
    Epub,
    Xml,
    Csv,
    PlainText,
}

impl Format {
    pub const fn bit(self) -> u32 {
        1u32 << self as u8
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessPattern {
    Sequential,
    Random,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParserLimits {
    pub max_input: u64,
    pub max_scratch: u64,
    pub max_output: u64,
    pub max_recursion: u16,
    pub max_members: u16,
    pub max_expansion_ratio: u16,
    pub max_dimension: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnsafeSurface {
    Forbidden,
    AuditedRust,
    ConfinedC,
}

/// Audited vendor metadata carried by each exact parser declaration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VendorRecord {
    pub package: &'static str,
    pub version: &'static str,
    pub source: &'static str,
    pub license: &'static str,
    pub no_std: bool,
    pub allocation: &'static str,
    pub unsafe_surface: UnsafeSurface,
    pub fuzz_evidence: &'static str,
    pub benchmark: &'static str,
    pub risks: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Signature {
    Json,
    Magic { offset: usize, bytes: &'static [u8] },
    Tar,
    Epub,
    Xml,
    Csv,
    PlainText,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParserDeclaration {
    pub id: ParserId,
    pub version: u32,
    pub format: Format,
    pub access: AccessPattern,
    pub deterministic: bool,
    pub fact_kinds: u64,
    pub limits: ParserLimits,
    pub vendor: VendorRecord,
    signature: Signature,
    priority: u8,
}

const COMMON_LIMITS: ParserLimits = ParserLimits {
    max_input: 1024 * 1024,
    max_scratch: 64 * 1024,
    max_output: 64 * 1024,
    max_recursion: 16,
    max_members: 64,
    max_expansion_ratio: 32,
    max_dimension: 8192,
};

const RUST_SAFE: UnsafeSurface = UnsafeSurface::Forbidden;

const fn vendor(
    package: &'static str,
    version: &'static str,
    source: &'static str,
    license: &'static str,
    allocation: &'static str,
    fuzz_evidence: &'static str,
    benchmark: &'static str,
    risks: &'static str,
) -> VendorRecord {
    VendorRecord {
        package,
        version,
        source,
        license,
        no_std: true,
        allocation,
        unsafe_surface: RUST_SAFE,
        fuzz_evidence,
        benchmark,
        risks,
    }
}

const STRUCTURED_FACTS: u64 = FactKind::MediaType.bit()
    | FactKind::JsonMemberCount.bit()
    | FactKind::Title.bit()
    | FactKind::Author.bit();

/// Historical declarations remain accepted for provenance validation and
/// cache replay. Selection chooses only the greatest registered version for a
/// parser lineage, so upgrading an implementation does not create a false
/// content-signature ambiguity.
pub static DECLARATIONS: [ParserDeclaration; 10] = [
    ParserDeclaration {
        id: RUST_JSON,
        version: 1,
        format: Format::Json,
        access: AccessPattern::Sequential,
        deterministic: true,
        fact_kinds: STRUCTURED_FACTS,
        limits: COMMON_LIMITS,
        vendor: vendor(
            "core-json",
            "0.4.0",
            "crates.io/core-json@0.4.0",
            "MIT",
            "no allocator; caller-selected const stack",
            "upstream randomized serde_json differential corpus",
            "kore host fixture benchmark recorded in res/EXTRACT_VENDOR_AUDIT.md",
            "historical parser contract retained for cache/provenance validation",
        ),
        signature: Signature::Json,
        priority: 0,
    },
    ParserDeclaration {
        id: RUST_JSON,
        version: 2,
        format: Format::Json,
        access: AccessPattern::Sequential,
        deterministic: true,
        fact_kinds: STRUCTURED_FACTS,
        limits: COMMON_LIMITS,
        vendor: vendor(
            "core-json",
            "0.4.0",
            "crates.io/core-json@0.4.0",
            "MIT",
            "no allocator; caller-selected const stack",
            "upstream randomized serde_json differential corpus",
            "kore host fixture benchmark recorded in res/EXTRACT_VENDOR_AUDIT.md",
            "MSRV 1.89; dynamic Value is reparsed rather than stored",
        ),
        signature: Signature::Json,
        priority: 0,
    },
    ParserDeclaration {
        id: C_JSON,
        version: 1,
        format: Format::Json,
        access: AccessPattern::Sequential,
        deterministic: true,
        fact_kinds: STRUCTURED_FACTS,
        limits: COMMON_LIMITS,
        vendor: VendorRecord {
            package: "jsmn",
            version: "1.1.0 / upstream commit 25647e692c7906b96ffd2b05ca54c097948e879c",
            source: "github.com/zserge/jsmn",
            license: "MIT",
            no_std: true,
            allocation: "C89, libc-free, caller token array",
            unsafe_surface: UnsafeSurface::ConfinedC,
            fuzz_evidence: "upstream tests plus kore differential/fuzz corpus",
            benchmark: "kore host fixture benchmark recorded in res/EXTRACT_VENDOR_AUDIT.md",
            risks: "foreign pointer/length ABI; accepted only in isolated worker",
        },
        signature: Signature::Json,
        priority: 1,
    },
    ParserDeclaration {
        id: TAR,
        version: 1,
        format: Format::Tar,
        access: AccessPattern::Sequential,
        deterministic: true,
        fact_kinds: FactKind::MediaType.bit()
            | FactKind::EntryCount.bit()
            | FactKind::UncompressedBytes.bit(),
        limits: COMMON_LIMITS,
        vendor: VendorRecord {
            package: "tar-no-std",
            version: "0.4.2",
            source: "crates.io/tar-no-std@0.4.2",
            license: "MIT",
            no_std: true,
            allocation: "zero allocation in TarArchiveRef",
            unsafe_surface: UnsafeSurface::AuditedRust,
            fuzz_evidence: "kore malformed/mutation/archive-bomb corpus",
            benchmark: "kore host fixture benchmark recorded in res/EXTRACT_VENDOR_AUDIT.md",
            risks: "five pointer/layout unsafe blocks; basic ustar only; no GNU longname",
        },
        signature: Signature::Tar,
        priority: 0,
    },
    ParserDeclaration {
        id: BMP,
        version: 1,
        format: Format::Bmp,
        access: AccessPattern::Random,
        deterministic: true,
        fact_kinds: FactKind::MediaType.bit() | FactKind::Dimensions.bit(),
        limits: COMMON_LIMITS,
        vendor: vendor(
            "tinybmp",
            "0.7.0",
            "crates.io/tinybmp@0.7.0",
            "MIT OR Apache-2.0",
            "zero-copy and no allocator",
            "embedded-graphics fixture suite; kore truncation/dimension corpus",
            "kore host fixture benchmark recorded in res/EXTRACT_VENDOR_AUDIT.md",
            "BMP only; compressed modes are explicitly bounded/refused",
        ),
        signature: Signature::Magic {
            offset: 0,
            bytes: b"BM",
        },
        priority: 0,
    },
    ParserDeclaration {
        id: TTF,
        version: 1,
        format: Format::Ttf,
        access: AccessPattern::Random,
        deterministic: true,
        fact_kinds: FactKind::MediaType.bit()
            | FactKind::FontFamily.bit()
            | FactKind::GlyphCount.bit(),
        limits: COMMON_LIMITS,
        vendor: vendor(
            "ttf-parser",
            "0.25.1",
            "crates.io/ttf-parser@0.25.1",
            "MIT OR Apache-2.0",
            "zero allocation; default features disabled",
            "upstream testing-tools corpus; library forbids unsafe",
            "kore host fixture benchmark recorded in res/EXTRACT_VENDOR_AUDIT.md",
            "RUSTSEC-2026-0192 unmaintained; confined worker, replacement tracked",
        ),
        signature: Signature::Magic {
            offset: 0,
            bytes: &[0x00, 0x01, 0x00, 0x00],
        },
        priority: 0,
    },
    ParserDeclaration {
        id: EPUB,
        version: 1,
        format: Format::Epub,
        access: AccessPattern::Random,
        deterministic: true,
        fact_kinds: FactKind::MediaType.bit()
            | FactKind::EntryCount.bit()
            | FactKind::Title.bit()
            | FactKind::Author.bit()
            | FactKind::DocumentSpineCount.bit(),
        limits: COMMON_LIMITS,
        vendor: vendor(
            "tinyzip+xmlparser",
            "tinyzip 0.4.0; xmlparser 0.13.6",
            "crates.io/tinyzip@0.4.0 + crates.io/xmlparser@0.13.6",
            "MIT; MIT OR Apache-2.0",
            "zero allocation; default features disabled",
            "both upstream corpora plus kore EPUB/ZIP/XML regression corpus",
            "kore host fixture benchmark recorded in res/EXTRACT_VENDOR_AUDIT.md",
            "initial adapter accepts stored metadata entries; deflate is separately bounded",
        ),
        signature: Signature::Epub,
        priority: 0,
    },
    ParserDeclaration {
        id: XML,
        version: 1,
        format: Format::Xml,
        access: AccessPattern::Sequential,
        deterministic: true,
        fact_kinds: FactKind::MediaType.bit() | FactKind::EntryCount.bit(),
        limits: COMMON_LIMITS,
        vendor: vendor(
            "xmlparser",
            "0.13.6",
            "crates.io/xmlparser@0.13.6",
            "MIT OR Apache-2.0",
            "zero allocation; tokenizer borrows input",
            "upstream corpus plus kore nesting/mutation corpus",
            "kore host fixture benchmark recorded in res/EXTRACT_VENDOR_AUDIT.md",
            "tokenizer-level API; adapter independently validates balanced nesting",
        ),
        signature: Signature::Xml,
        priority: 0,
    },
    ParserDeclaration {
        id: CSV,
        version: 1,
        format: Format::Csv,
        access: AccessPattern::Sequential,
        deterministic: true,
        fact_kinds: FactKind::MediaType.bit()
            | FactKind::CsvRowCount.bit()
            | FactKind::CsvColumnCount.bit(),
        limits: COMMON_LIMITS,
        vendor: vendor(
            "kore-bounded-csv",
            "1",
            "in-tree adapter",
            "Apache-2.0",
            "zero allocation; bounded state machine",
            "kore quoted-field, malformed, mutation, and limit corpus",
            "kore host fixture benchmark recorded in res/EXTRACT_VENDOR_AUDIT.md",
            "UTF-8 comma-separated dialect; formula evaluation is deliberately absent",
        ),
        signature: Signature::Csv,
        priority: 0,
    },
    ParserDeclaration {
        id: PLAIN_TEXT,
        version: 1,
        format: Format::PlainText,
        access: AccessPattern::Sequential,
        deterministic: true,
        // Plain-text owners may publish bounded application-defined tokens
        // beside the parser's media/line facts.  The filesystem validates the
        // exact immutable input identity and this declaration; `Custom` does
        // not grant a producer authority to invent another parser's facts.
        fact_kinds: FactKind::MediaType.bit()
            | FactKind::TextLineCount.bit()
            | FactKind::Custom.bit(),
        limits: COMMON_LIMITS,
        vendor: vendor(
            "kore-bounded-text",
            "1",
            "in-tree adapter",
            "Apache-2.0",
            "zero allocation; UTF-8 slices copied into bounded spans",
            "kore malformed UTF-8, control-byte, truncation, and limit corpus",
            "kore host fixture benchmark recorded in res/EXTRACT_VENDOR_AUDIT.md",
            "plain UTF-8 only; binary and structured signatures are refused",
        ),
        signature: Signature::PlainText,
        priority: 0,
    },
];

/// Caller selection policy. Extensions are bounded hints and are never
/// consulted unless content already matched a parser declaration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParserPolicy {
    pub allowed_formats: u32,
    pub allowed_parsers: u64,
    pub preferred: Option<ParserId>,
    extension: [u8; 8],
    extension_len: u8,
}

impl ParserPolicy {
    pub const ALL: Self = Self {
        allowed_formats: u32::MAX,
        allowed_parsers: u64::MAX,
        preferred: None,
        extension: [0; 8],
        extension_len: 0,
    };

    pub fn with_extension(mut self, extension: &[u8]) -> Self {
        let len = extension.len().min(self.extension.len());
        self.extension[..len].copy_from_slice(&extension[..len]);
        self.extension_len = len as u8;
        self
    }

    /// Prefer one policy-permitted parser when several declarations match the
    /// same content. The preference never bypasses signature or format gates.
    pub const fn prefer(mut self, parser: ParserId) -> Self {
        self.preferred = Some(parser);
        self
    }

    fn permits(self, parser: ParserDeclaration) -> bool {
        self.allowed_formats & parser.format.bit() != 0
            && parser.id.0 < 64
            && self.allowed_parsers & (1u64 << parser.id.0) != 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Selection {
    pub parser: &'static ParserDeclaration,
    /// Number of policy-permitted signature matches before deterministic
    /// tie-breaking.
    pub ambiguity: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidationError {
    UnknownParser,
    Provenance,
    SchemaVersion,
    CacheIdentity,
    InputTooLarge,
    UndeclaredFact,
    SourceRange,
    OutputRange,
    Dimensions,
}

pub struct Registry;

impl Registry {
    pub const fn declarations() -> &'static [ParserDeclaration] {
        &DECLARATIONS
    }

    pub fn get(id: ParserId) -> Option<&'static ParserDeclaration> {
        DECLARATIONS
            .iter()
            .filter(|parser| parser.id == id)
            .max_by_key(|parser| parser.version)
    }

    /// Resolve one exact historical parser contract for validating durable
    /// provenance produced before a worker upgrade.
    pub fn get_version(id: ParserId, version: u32) -> Option<&'static ParserDeclaration> {
        DECLARATIONS
            .iter()
            .find(|parser| parser.id == id && parser.version == version)
    }

    pub fn select(prefix: &[u8], policy: ParserPolicy) -> Option<Selection> {
        let mut selected: Option<&ParserDeclaration> = None;
        let mut ambiguity = 0u8;
        for parser in &DECLARATIONS {
            if Self::get(parser.id).is_none_or(|current| current.version != parser.version) {
                continue;
            }
            if !policy.permits(*parser) || !matches_signature(parser.signature, prefix) {
                continue;
            }
            ambiguity = ambiguity.saturating_add(1);
            let prefer = policy.preferred == Some(parser.id);
            let replace = selected.is_none_or(|current| {
                (prefer && policy.preferred != Some(current.id))
                    || (prefer == (policy.preferred == Some(current.id))
                        && parser.priority < current.priority)
            });
            if replace {
                selected = Some(parser);
            }
        }
        selected.map(|parser| Selection { parser, ambiguity })
    }

    pub fn validate(
        result: &NormalizedResult,
        expected: InputIdentity,
        output_window_bytes: u64,
    ) -> Result<(), ValidationError> {
        let parser = Self::get_version(result.provenance.parser, result.provenance.parser_version)
            .ok_or(ValidationError::UnknownParser)?;
        if result.provenance.input != expected
            || result.provenance.parser_version != parser.version
            || result.provenance.deterministic != parser.deterministic
        {
            return Err(ValidationError::Provenance);
        }
        if result.provenance.schema_version != SCHEMA_VERSION {
            return Err(ValidationError::SchemaVersion);
        }
        if expected.len > parser.limits.max_input {
            return Err(ValidationError::InputTooLarge);
        }
        if parser.deterministic
            && result.provenance.cache
                != ParserProvenance::deterministic(expected, parser.id, parser.version).cache
        {
            return Err(ValidationError::CacheIdentity);
        }
        for fact in &result.facts {
            if parser.fact_kinds & fact.kind.bit() == 0 {
                return Err(ValidationError::UndeclaredFact);
            }
            if fact.source.is_some_and(|range| !range.within(expected.len)) {
                return Err(ValidationError::SourceRange);
            }
        }
        for range in result
            .text
            .iter()
            .map(|v| v.source)
            .chain(result.embedded.iter().map(|v| v.source))
            .chain(result.references.iter().map(|v| v.source))
            .chain(result.symbols.iter().map(|v| v.source))
            .chain(result.diagnostics.iter().filter_map(|v| v.source))
        {
            if !range.within(expected.len) {
                return Err(ValidationError::SourceRange);
            }
        }
        let output_limit = output_window_bytes.min(parser.limits.max_output);
        for thumb in &result.thumbnails {
            let end = u64::from(thumb.output_offset)
                .checked_add(u64::from(thumb.output_len))
                .ok_or(ValidationError::OutputRange)?;
            if end > output_limit {
                return Err(ValidationError::OutputRange);
            }
            if thumb.width == 0
                || thumb.height == 0
                || thumb.width > parser.limits.max_dimension
                || thumb.height > parser.limits.max_dimension
            {
                return Err(ValidationError::Dimensions);
            }
        }
        Ok(())
    }
}

fn matches_signature(signature: Signature, prefix: &[u8]) -> bool {
    match signature {
        Signature::Json => prefix
            .iter()
            .copied()
            .find(|byte| !byte.is_ascii_whitespace())
            .is_some_and(|byte| byte == b'{' || byte == b'['),
        Signature::Magic { offset, bytes } => prefix
            .get(offset..offset.saturating_add(bytes.len()))
            .is_some_and(|candidate| candidate == bytes),
        Signature::Tar => prefix.get(257..262) == Some(&b"ustar"[..]),
        Signature::Epub => {
            prefix.starts_with(b"PK\x03\x04")
                && prefix
                    .windows(b"application/epub+zip".len())
                    .any(|window| window == b"application/epub+zip")
        }
        Signature::Xml => {
            prefix
                .iter()
                .copied()
                .find(|byte| !byte.is_ascii_whitespace())
                == Some(b'<')
        }
        Signature::Csv => looks_like_csv(prefix),
        Signature::PlainText => looks_like_plain_text(prefix) && !looks_like_csv(prefix),
    }
}

fn looks_like_plain_text(prefix: &[u8]) -> bool {
    if prefix.is_empty()
        || prefix.iter().any(|byte| {
            *byte == 0 || (*byte < b' ' && !matches!(*byte, b'\t' | b'\r' | b'\n')) || *byte == 0x7f
        })
        || core::str::from_utf8(prefix).is_err()
    {
        return false;
    }
    let Some(first) = prefix
        .iter()
        .copied()
        .find(|byte| !byte.is_ascii_whitespace())
    else {
        return false;
    };
    !matches!(first, b'{' | b'[' | b'<')
        && !prefix.starts_with(b"BM")
        && !prefix.starts_with(b"PK\x03\x04")
        && prefix.get(257..262) != Some(&b"ustar"[..])
}

fn looks_like_csv(prefix: &[u8]) -> bool {
    if !looks_like_plain_text(prefix) {
        return false;
    }
    let mut rows = 0u8;
    let mut expected = 0u16;
    for line in prefix.split(|byte| *byte == b'\n').take(3) {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if line.is_empty() {
            continue;
        }
        let columns = 1u16.saturating_add(line.iter().filter(|byte| **byte == b',').count() as u16);
        if columns < 2 || (expected != 0 && columns != expected) {
            return false;
        }
        expected = columns;
        rows = rows.saturating_add(1);
    }
    rows >= 2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{Confidence, Fact, FactValue, ResultBuilder, SourceRange};
    use heapless::String;

    #[test]
    fn ambiguous_json_is_deterministic_and_policy_can_choose_c() {
        let selection = Registry::select(b" \n {\"a\":1}", ParserPolicy::ALL).unwrap();
        assert_eq!(selection.parser.id, RUST_JSON);
        assert_eq!(selection.ambiguity, 2);
        let selection = Registry::select(
            b"{\"a\":1}",
            ParserPolicy {
                preferred: Some(C_JSON),
                ..ParserPolicy::ALL
            },
        )
        .unwrap();
        assert_eq!(selection.parser.id, C_JSON);
    }

    #[test]
    fn extension_is_only_a_hint_never_format_proof() {
        let selected = Registry::select(
            b"definitely not json",
            ParserPolicy::ALL.with_extension(b"json"),
        )
        .unwrap();
        assert_eq!(selected.parser.id, PLAIN_TEXT);
        assert_ne!(selected.parser.id, RUST_JSON);
    }

    #[test]
    fn csv_and_plain_text_are_selected_by_content_without_stealing_structured_inputs() {
        assert_eq!(
            Registry::select(b"a,b\n1,2\n", ParserPolicy::ALL)
                .unwrap()
                .parser
                .id,
            CSV
        );
        assert_eq!(
            Registry::select(b"ordinary prose\nsecond line\n", ParserPolicy::ALL)
                .unwrap()
                .parser
                .id,
            PLAIN_TEXT
        );
        assert_eq!(
            Registry::select(b"{\"a\":1}", ParserPolicy::ALL)
                .unwrap()
                .parser
                .id,
            RUST_JSON
        );
    }

    #[test]
    fn xml_is_selected_from_content_not_its_extension() {
        let selected = Registry::select(
            b"<?xml version=\"1.0\"?><root/>",
            ParserPolicy::ALL.with_extension(b"bin"),
        )
        .unwrap();
        assert_eq!(selected.parser.id, XML);
        assert_eq!(selected.ambiguity, 1);
    }

    #[test]
    fn validator_rejects_wrong_provenance_kind_and_range() {
        let input = InputIdentity::of(9, 3, b"{\"a\":1}");
        let provenance = ParserProvenance::deterministic(
            input,
            RUST_JSON,
            Registry::get(RUST_JSON).unwrap().version,
        );
        let mut builder = ResultBuilder::new(provenance);
        builder
            .fact(Fact {
                kind: FactKind::JsonMemberCount,
                key: String::try_from("members").unwrap(),
                value: FactValue::Unsigned(1),
                confidence: Confidence::CERTAIN,
                source: Some(SourceRange::new(0, input.len)),
            })
            .unwrap();
        let valid = builder.finish();
        assert_eq!(Registry::validate(&valid, input, 4096), Ok(()));

        let mut bad = valid.clone();
        bad.provenance.input.version += 1;
        assert_eq!(
            Registry::validate(&bad, input, 4096),
            Err(ValidationError::Provenance)
        );
        let mut bad = valid.clone();
        bad.facts[0].kind = FactKind::GlyphCount;
        assert_eq!(
            Registry::validate(&bad, input, 4096),
            Err(ValidationError::UndeclaredFact)
        );
        let mut bad = valid;
        bad.facts[0].source = Some(SourceRange::new(input.len, 1));
        assert_eq!(
            Registry::validate(&bad, input, 4096),
            Err(ValidationError::SourceRange)
        );
    }
}
