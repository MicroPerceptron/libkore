//! Capability-safe bounded structural, text, relationship, and semantic query.
//!
//! Query execution is split at a trait seam. The filesystem owns authority
//! and generates candidates only by walking the caller's live namespace view.
//! A [`QueryIndex`] may score that already-visible object, but it cannot mint
//! candidates, handles, identity, or access.

use heapless::String;

use crate::{
    error::FsResult,
    object::{ObjectId, Revision},
};

pub use kore_dagfs::search::{
    FreshnessPolicy, MAX_SEARCH_PAGE, MAX_SEARCH_SCAN, MAX_SEARCH_SNIPPET, MAX_SEARCH_WORK,
    SearchContent, SearchCursor, SearchHit, SearchPlane, SearchQuery, SearchScore, SearchTerm,
};
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexMatch {
    pub matched: bool,
    pub score: SearchScore,
    pub snippet: String<MAX_SEARCH_SNIPPET>,
    pub work: u32,
    /// Generation of this visible object's rebuildable result shard.
    pub generation: u64,
}

impl IndexMatch {
    pub fn miss(work: u32) -> Self {
        Self {
            matched: false,
            score: SearchScore::NONE,
            snippet: String::new(),
            work,
            generation: 0,
        }
    }
}

/// Rebuildable index seam. There is intentionally no enumeration method:
/// implementations can score a proven candidate, never create one.
pub trait QueryIndex {
    fn evaluate(
        &self,
        object: ObjectId,
        revision: Revision,
        query: SearchContent,
    ) -> FsResult<IndexMatch>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SearchPage {
    pub hits: heapless::Vec<SearchHit, MAX_SEARCH_PAGE>,
    pub observation: u64,
    pub snapshot: u64,
    pub index_generation: u64,
    pub freshness: FreshnessPolicy,
    pub next: Option<SearchCursor>,
    pub scanned: u32,
    pub work: u32,
}
