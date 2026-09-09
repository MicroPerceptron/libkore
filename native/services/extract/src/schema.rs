//! Parser-neutral normalized extraction records.

use heapless::{String, Vec};

use crate::registry::ParserId;

pub const SCHEMA_VERSION: u16 = 1;
pub const MAX_FACTS: usize = 64;
pub const MAX_TEXT_SPANS: usize = 32;
pub const MAX_EMBEDDED: usize = 16;
pub const MAX_REFERENCES: usize = 16;
pub const MAX_SYMBOLS: usize = 32;
pub const MAX_THUMBNAILS: usize = 4;
pub const MAX_DIAGNOSTICS: usize = 16;
pub const MAX_KEY_BYTES: usize = 32;
pub const MAX_VALUE_BYTES: usize = 96;
pub const MAX_TEXT_BYTES: usize = 256;

/// Identity of one immutable source version.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InputIdentity {
    /// Consumer-local opaque source identity. It conveys no authority.
    pub source: u64,
    /// Immutable version or snapshot generation.
    pub version: u64,
    pub len: u64,
    pub digest: [u8; 32],
}

impl InputIdentity {
    pub fn of(source: u64, version: u64, bytes: &[u8]) -> Self {
        Self {
            source,
            version,
            len: bytes.len() as u64,
            digest: *blake3::hash(bytes).as_bytes(),
        }
    }
}

/// Half-open range in the immutable input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SourceRange {
    pub offset: u64,
    pub len: u64,
}

impl SourceRange {
    pub const fn new(offset: u64, len: u64) -> Self {
        Self { offset, len }
    }

    pub const fn end(self) -> Option<u64> {
        self.offset.checked_add(self.len)
    }

    pub const fn within(self, len: u64) -> bool {
        matches!(self.end(), Some(end) if end <= len)
    }
}

/// Fixed-point confidence in ten-thousandths.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Confidence(u16);

impl Confidence {
    pub const CERTAIN: Self = Self(10_000);

    pub const fn new(value: u16) -> Option<Self> {
        if value <= 10_000 {
            Some(Self(value))
        } else {
            None
        }
    }

    pub const fn get(self) -> u16 {
        self.0
    }
}

/// Stable normalized fact categories. Parser-specific structures never cross
/// the adapter boundary.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FactKind {
    MediaType,
    Title,
    Author,
    Dimensions,
    EntryCount,
    UncompressedBytes,
    FontFamily,
    GlyphCount,
    JsonMemberCount,
    DocumentSpineCount,
    Custom,
    CsvRowCount,
    CsvColumnCount,
    TextLineCount,
}

impl FactKind {
    pub const fn bit(self) -> u64 {
        1u64 << self as u8
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FactValue {
    Unsigned(u64),
    Signed(i64),
    Bool(bool),
    Utf8(String<MAX_VALUE_BYTES>),
    Bytes(Vec<u8, MAX_VALUE_BYTES>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fact {
    pub kind: FactKind,
    pub key: String<MAX_KEY_BYTES>,
    pub value: FactValue,
    pub confidence: Confidence,
    pub source: Option<SourceRange>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextSpan {
    pub source: SourceRange,
    pub text: String<MAX_TEXT_BYTES>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmbeddedRange {
    pub source: SourceRange,
    pub media_type: String<MAX_VALUE_BYTES>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reference {
    pub source: SourceRange,
    pub target: String<MAX_VALUE_BYTES>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Symbol {
    pub source: SourceRange,
    pub name: String<MAX_VALUE_BYTES>,
}

/// A thumbnail lives in the bounded worker output window; only its descriptor
/// crosses the normalized contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Thumbnail {
    pub output_offset: u32,
    pub output_len: u32,
    pub width: u32,
    pub height: u32,
    pub format: u16,
}

#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiagnosticCode {
    Truncated,
    Unsupported,
    Malformed,
    LimitReached,
    Ambiguous,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    pub code: DiagnosticCode,
    pub source: Option<SourceRange>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CacheIdentity(pub [u8; 32]);

/// Provenance repeated on every normalized result envelope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParserProvenance {
    pub input: InputIdentity,
    pub parser: ParserId,
    pub parser_version: u32,
    pub schema_version: u16,
    pub deterministic: bool,
    pub cache: CacheIdentity,
}

impl ParserProvenance {
    pub fn deterministic(input: InputIdentity, parser: ParserId, parser_version: u32) -> Self {
        let mut hash = blake3::Hasher::new();
        hash.update(&input.digest);
        hash.update(&input.version.to_le_bytes());
        hash.update(&parser.0.to_le_bytes());
        hash.update(&parser_version.to_le_bytes());
        hash.update(&SCHEMA_VERSION.to_le_bytes());
        Self {
            input,
            parser,
            parser_version,
            schema_version: SCHEMA_VERSION,
            deterministic: true,
            cache: CacheIdentity(*hash.finalize().as_bytes()),
        }
    }
}

/// Complete parser-neutral output. Every collection is compile-time bounded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormalizedResult {
    pub provenance: ParserProvenance,
    pub facts: Vec<Fact, MAX_FACTS>,
    pub text: Vec<TextSpan, MAX_TEXT_SPANS>,
    pub embedded: Vec<EmbeddedRange, MAX_EMBEDDED>,
    pub references: Vec<Reference, MAX_REFERENCES>,
    pub symbols: Vec<Symbol, MAX_SYMBOLS>,
    pub thumbnails: Vec<Thumbnail, MAX_THUMBNAILS>,
    pub diagnostics: Vec<Diagnostic, MAX_DIAGNOSTICS>,
}

impl NormalizedResult {
    pub fn empty(provenance: ParserProvenance) -> Self {
        Self {
            provenance,
            facts: Vec::new(),
            text: Vec::new(),
            embedded: Vec::new(),
            references: Vec::new(),
            symbols: Vec::new(),
            thumbnails: Vec::new(),
            diagnostics: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchemaError {
    Full,
}

/// Adapter helper that exposes only normalized bounded append operations.
pub struct ResultBuilder {
    result: NormalizedResult,
}

impl ResultBuilder {
    pub fn new(provenance: ParserProvenance) -> Self {
        Self {
            result: NormalizedResult::empty(provenance),
        }
    }

    pub fn fact(&mut self, fact: Fact) -> Result<(), SchemaError> {
        self.result.facts.push(fact).map_err(|_| SchemaError::Full)
    }

    pub fn text(&mut self, span: TextSpan) -> Result<(), SchemaError> {
        self.result.text.push(span).map_err(|_| SchemaError::Full)
    }

    pub fn embedded(&mut self, range: EmbeddedRange) -> Result<(), SchemaError> {
        self.result
            .embedded
            .push(range)
            .map_err(|_| SchemaError::Full)
    }

    pub fn reference(&mut self, reference: Reference) -> Result<(), SchemaError> {
        self.result
            .references
            .push(reference)
            .map_err(|_| SchemaError::Full)
    }

    pub fn symbol(&mut self, symbol: Symbol) -> Result<(), SchemaError> {
        self.result
            .symbols
            .push(symbol)
            .map_err(|_| SchemaError::Full)
    }

    pub fn thumbnail(&mut self, thumbnail: Thumbnail) -> Result<(), SchemaError> {
        self.result
            .thumbnails
            .push(thumbnail)
            .map_err(|_| SchemaError::Full)
    }

    pub fn diagnostic(&mut self, diagnostic: Diagnostic) -> Result<(), SchemaError> {
        self.result
            .diagnostics
            .push(diagnostic)
            .map_err(|_| SchemaError::Full)
    }

    pub fn finish(self) -> NormalizedResult {
        self.result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_identity_changes_with_input_or_parser_version() {
        let a = InputIdentity::of(1, 7, b"same");
        let b = InputIdentity::of(1, 8, b"same");
        let p1 = ParserProvenance::deterministic(a, ParserId(2), 1);
        let p2 = ParserProvenance::deterministic(a, ParserId(2), 2);
        let p3 = ParserProvenance::deterministic(b, ParserId(2), 1);
        assert_ne!(p1.cache, p2.cache);
        assert_ne!(p1.cache, p3.cache);
        assert_eq!(
            p1,
            ParserProvenance::deterministic(a, ParserId(2), 1),
            "deterministic identity is reproducible"
        );
    }

    #[test]
    fn range_arithmetic_fails_closed_on_overflow() {
        assert!(SourceRange::new(4, 3).within(7));
        assert!(!SourceRange::new(4, 4).within(7));
        assert!(!SourceRange::new(u64::MAX, 2).within(u64::MAX));
    }
}
