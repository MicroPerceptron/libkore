//! The search query vocabulary a client speaks: terms, planes, freshness,
//! cursors, scores, and the hit record a page delivers.
//!
//! Only the *data* half of search lives here. Candidate generation, the
//! rebuildable index seam (`QueryIndex`), and the page assembly stay in the
//! storage service, where the caller's live namespace view is walked — an
//! index scores an already-visible object and can never mint one.

use heapless::String;

use crate::{
    bounds::{MAX_DIR_ENTRIES, MAX_DIRECTORIES, MAX_LIST_PAGE, MAX_NAME_LEN, MAX_PATH_DEPTH},
    error::FsResult,
    name::Name,
    object::{ObjectId, ObjectKind, Revision, StableObjectId},
};

pub const MAX_SEARCH_SCAN: usize = MAX_DIRECTORIES * MAX_DIR_ENTRIES;
pub const MAX_SEARCH_PAGE: usize = MAX_LIST_PAGE;
pub const MAX_SEARCH_SNIPPET: usize = 64;
/// Worst-case candidate plus normalized-result items evaluated by one query.
pub const MAX_SEARCH_WORK: u32 = (MAX_SEARCH_SCAN
    * (1 + kore_svc_extract::schema::MAX_FACTS
        + kore_svc_extract::schema::MAX_TEXT_SPANS
        + kore_svc_extract::schema::MAX_REFERENCES
        + kore_svc_extract::schema::MAX_SYMBOLS)) as u32;

/// Bounded UTF-8 query text. Unlike a namespace [`Name`], search text may
/// contain `/`, `.` and spaces; it grants no traversal meaning.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct SearchTerm {
    len: u8,
    bytes: [u8; MAX_NAME_LEN],
}

impl SearchTerm {
    pub fn new(value: &str) -> FsResult<Self> {
        Self::from_bytes(value.as_bytes())
    }

    pub fn from_bytes(bytes: &[u8]) -> FsResult<Self> {
        if bytes.is_empty()
            || bytes.len() > MAX_NAME_LEN
            || bytes.contains(&0)
            || core::str::from_utf8(bytes).is_err()
        {
            return Err(crate::error::FsError::InvalidName);
        }
        let mut stored = [0; MAX_NAME_LEN];
        stored[..bytes.len()].copy_from_slice(bytes);
        Ok(Self {
            len: bytes.len() as u8,
            bytes: stored,
        })
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len as usize]
    }

    pub fn as_str(&self) -> &str {
        core::str::from_utf8(self.as_bytes()).unwrap_or("")
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }
}

impl core::fmt::Debug for SearchTerm {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "SearchTerm({:?})", self.as_str())
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchPlane {
    FactKey = 1,
    FullText = 2,
    Relationship = 3,
    /// Exact bounded token coverage; not a stable ranking ABI or embedding
    /// model. Hash collisions cannot manufacture a hit.
    Semantic = 4,
}

impl SearchPlane {
    pub const fn from_wire(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::FactKey),
            2 => Some(Self::FullText),
            3 => Some(Self::Relationship),
            4 => Some(Self::Semantic),
            _ => None,
        }
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FreshnessPolicy {
    StructuralOnly = 0,
    CurrentOnly = 1,
    /// The native index is synchronous and therefore reports zero lag; this
    /// is the stable policy seam for a future asynchronous provider.
    BoundedStale = 2,
}

impl FreshnessPolicy {
    pub const fn from_wire(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::StructuralOnly),
            1 => Some(Self::CurrentOnly),
            2 => Some(Self::BoundedStale),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchContent {
    pub plane: SearchPlane,
    pub term: SearchTerm,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchQuery {
    pub name_contains: Option<Name>,
    pub kind: Option<ObjectKind>,
    pub min_len: u64,
    pub max_len: u64,
    pub content: Option<SearchContent>,
    pub freshness: FreshnessPolicy,
    /// Must be zero unless `freshness` is [`FreshnessPolicy::BoundedStale`].
    pub max_index_lag: u32,
    /// Zero selects [`MAX_SEARCH_WORK`].
    pub max_work: u32,
}

impl SearchQuery {
    pub const ALL: Self = Self {
        name_contains: None,
        kind: None,
        min_len: 0,
        max_len: u64::MAX,
        content: None,
        freshness: FreshnessPolicy::StructuralOnly,
        max_index_lag: 0,
        max_work: 0,
    };

    pub fn matches_structural(&self, name: &Name, kind: ObjectKind, len: u64) -> bool {
        let name_matches = self.name_contains.is_none_or(|needle| {
            let needle = needle.as_bytes();
            needle.is_empty()
                || name
                    .as_bytes()
                    .windows(needle.len())
                    .any(|candidate| candidate == needle)
        });
        name_matches
            && self.kind.is_none_or(|wanted| wanted == kind)
            && len >= self.min_len
            && len <= self.max_len
    }

    pub fn work_limit(self) -> u32 {
        if self.max_work == 0 {
            MAX_SEARCH_WORK
        } else {
            self.max_work.min(MAX_SEARCH_WORK)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchCursor {
    pub observation: u64,
    pub position: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct SearchScore(u16);

impl SearchScore {
    pub const NONE: Self = Self(0);
    pub const EXACT: Self = Self(10_000);

    pub const fn new(value: u16) -> Self {
        Self(if value > 10_000 { 10_000 } else { value })
    }

    pub const fn get(self) -> u16 {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SearchHit {
    pub path: heapless::Vec<Name, MAX_PATH_DEPTH>,
    pub object: ObjectId,
    pub stable_object: StableObjectId,
    pub kind: ObjectKind,
    pub len: u64,
    pub revision: Revision,
    pub score: SearchScore,
    pub snippet: String<MAX_SEARCH_SNIPPET>,
}
