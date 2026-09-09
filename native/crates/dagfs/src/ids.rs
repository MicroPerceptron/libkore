//! The wire-visible identities of the filesystem protocol.
//!
//! Every value here is **data, never authority**: a right set, a session
//! index, a grant id, a handle, a version lease, a quota. Each is meaningful
//! only to the service that minted it and only on the session it was minted
//! for — the service re-derives the owning session from the pair that
//! delivered a request and revalidates the bounded route on every use, so
//! copying these bits to another pair buys nothing.
//!
//! They live in this leaf rather than in the storage service because a
//! process must be able to hold, encode, and compare them without linking the
//! engine, the providers, or the device stack. The service reconstructs its
//! arena keys from the `*_bits` forms; nothing here reaches into an arena.

use kore_collections::SlotId as SlotKey;

use crate::{
    bounds::{MAX_DIR_ENTRIES, MAX_DIRECTORIES, MAX_HANDLES_PER_SESSION, MAX_OBJECTS},
    error::{FsError, FsResult},
};

/// Storage and handle budget visible through one authority edge.
///
/// The limits travel with the authority edge and can only narrow when that
/// edge is delegated. Usage is deliberately *not* cached here: the engine
/// derives object, containment-edge, and byte usage from the live graph rooted
/// at the edge, while handle usage comes from the live generational handle
/// arena. This keeps quota accounting correct across detach/reclaim and after
/// durable recovery without a second mutable ledger that could drift from the
/// namespace.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AuthorityQuota {
    max_objects: u32,
    max_edges: u32,
    max_bytes: u64,
    max_handles: u32,
}

impl AuthorityQuota {
    /// The engine-wide structural ceiling. A private root starts here; an
    /// explicit task-view grant normally asks for a smaller budget.
    pub const MAX: Self = Self {
        max_objects: MAX_OBJECTS as u32,
        max_edges: (MAX_DIRECTORIES * MAX_DIR_ENTRIES) as u32,
        // The engine is provider-generic: a private root cannot inherit the
        // RAM composition's byte ceiling when the selected provider is a
        // block device. Provider limits remain the physical upper bound;
        // delegated task views choose their own finite budget explicitly.
        max_bytes: u64::MAX,
        max_handles: MAX_HANDLES_PER_SESSION as u32,
    };

    /// Construct an explicit bounded quota.
    pub const fn new(max_objects: u32, max_edges: u32, max_bytes: u64, max_handles: u32) -> Self {
        Self {
            max_objects,
            max_edges,
            max_bytes,
            max_handles,
        }
    }

    #[inline]
    pub const fn max_objects(self) -> u32 {
        self.max_objects
    }

    #[inline]
    pub const fn max_edges(self) -> u32 {
        self.max_edges
    }

    #[inline]
    pub const fn max_bytes(self) -> u64 {
        self.max_bytes
    }

    #[inline]
    pub const fn max_handles(self) -> u32 {
        self.max_handles
    }

    /// Narrow this quota. Like rights attenuation, asking for more than the
    /// source edge carries is a refusal rather than a silent clamp.
    pub fn attenuate(self, requested: Self) -> FsResult<Self> {
        if requested.max_objects > self.max_objects
            || requested.max_edges > self.max_edges
            || requested.max_bytes > self.max_bytes
            || requested.max_handles > self.max_handles
        {
            return Err(FsError::AccessDenied);
        }
        Ok(requested)
    }
}

/// What a route permits.  Absence of a bit is authoritative.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Rights(u8);

impl Rights {
    /// No authority at all.
    pub const NONE: Self = Self(0);
    /// List a directory's entries and read a file's bytes and metadata.
    pub const READ: Self = Self(1 << 0);
    /// Change a file's bytes or length.
    pub const WRITE: Self = Self(1 << 1);
    /// Create, rename, link, move, or unlink beneath a directory.
    pub const MUTATE_NAMESPACE: Self = Self(1 << 2);
    /// Create an attenuated authority edge for another session.
    pub const SHARE: Self = Self(1 << 3);
    /// Nominate an immutable file revision as source material for an
    /// executable binding.
    ///
    /// This does not create executable mappings: it is the filesystem-side
    /// authority edge that lets a trusted launcher freeze a source revision
    /// for ImageService's validation path. It controls installation into the
    /// shared executable namespace, not private generated code: a task may
    /// consume its own full-lifecycle memory lease into immutable executable
    /// memory without gaining authority over any filesystem revision.
    pub const BIND_EXECUTABLE: Self = Self(1 << 4);

    /// Everything — what a session's own private root carries.  This is the
    /// service-held root authority; the faces handed to ordinary tasks are
    /// attenuated derivations of it ([`HOME`]), never this in the raw.
    pub const ALL: Self = Self(
        Self::READ.0
            | Self::WRITE.0
            | Self::MUTATE_NAMESPACE.0
            | Self::SHARE.0
            | Self::BIND_EXECUTABLE.0,
    );

    /// A writable home **face** — read, write, restructure, and re-share the
    /// task's own tree — but WITHOUT [`BIND_EXECUTABLE`].  This is what a task
    /// actually holds: it can do everything to its files except nominate a
    /// revision for the system executable namespace, which stays the install
    /// authority's privilege (`ALL`, held by exec's system-source view).
    pub const HOME: Self =
        Self(Self::READ.0 | Self::WRITE.0 | Self::MUTATE_NAMESPACE.0 | Self::SHARE.0);

    /// Read-only: the recommended default for sharing.
    pub const READ_ONLY: Self = Self::READ;

    #[inline]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Build from raw bits, discarding any bit that is not a defined right.
    #[inline]
    pub const fn from_bits_truncating(bits: u8) -> Self {
        Self(bits & Self::ALL.0)
    }

    /// Build from raw bits, refusing any bit that is not a defined right.
    ///
    /// **This is the one the wire codec uses.** Truncating an unknown bit
    /// would silently accept a request from a peer speaking a protocol this
    /// service does not implement, and answer it as though the unknown right
    /// had not been asked for — a fail-*open* reading of a field whose whole
    /// job is to bound authority.
    #[inline]
    pub const fn from_bits_exact(bits: u8) -> Option<Self> {
        if bits & !Self::ALL.0 != 0 {
            return None;
        }
        Some(Self(bits))
    }

    /// `true` if every right in `other` is present here.
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// The rights held by both — the only way authority is ever combined, so
    /// that a transition cannot add a right.
    #[inline]
    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// `true` if this conveys no more than `other`.
    #[inline]
    pub const fn is_subset_of(self, other: Self) -> bool {
        other.contains(self)
    }

    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Narrow to `requested`, refusing if the request is not already held.
    ///
    /// Deliberately a refusal rather than a silent clamp: a client that asks
    /// for `WRITE` on a read-only route has made a mistake it should hear
    /// about, not receive a read-only handle it thinks it can write.
    pub fn attenuate(self, requested: Self) -> FsResult<Self> {
        if !self.contains(requested) {
            return Err(FsError::AccessDenied);
        }
        Ok(self.intersection(requested))
    }

    /// Require `needed`, or refuse.
    #[inline]
    pub fn require(self, needed: Self) -> FsResult<()> {
        if self.contains(needed) {
            Ok(())
        } else {
            Err(FsError::AccessDenied)
        }
    }
}

/// Which client a request belongs to.
///
/// On metal this is derived from the unforgeable pair authority that
/// delivered the request and is never read from a wire field; the host test
/// harness supplies the same trusted input directly.
#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord, Hash)]
pub struct SessionId(u16);

impl SessionId {
    #[inline]
    pub const fn new(index: u16) -> Self {
        Self(index)
    }

    #[inline]
    pub const fn index(self) -> u16 {
        self.0
    }
}

/// An authority edge's identity.
///
/// Opaque and generational.  Not bearer authority: opening one checks that
/// the caller is the edge's holder, and revoking one checks that the caller
/// is its holder or its issuer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GrantId(SlotKey);

impl GrantId {
    #[inline]
    pub const fn to_bits(self) -> u64 {
        self.0.to_bits()
    }

    #[inline]
    pub const fn from_bits(bits: u64) -> Self {
        Self(SlotKey::from_bits(bits))
    }
}

/// An open handle's identity — session-local in effect, because the record it
/// names records its owner.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Handle(SlotKey);

impl Handle {
    #[inline]
    pub const fn to_bits(self) -> u64 {
        self.0.to_bits()
    }

    #[inline]
    pub const fn from_bits(bits: u64) -> Self {
        Self(SlotKey::from_bits(bits))
    }
}

/// Opaque, generational access to one immutable source version.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VersionLease(SlotKey);

impl VersionLease {
    pub const INVALID: Self = Self(SlotKey::INVALID);

    /// Diagnostic/wire encoding. Possessing these bits is insufficient: the
    /// serving host also checks the structurally stamped connection owner.
    pub const fn to_bits(self) -> u64 {
        self.0.to_bits()
    }

    pub const fn from_bits(bits: u64) -> Self {
        Self(SlotKey::from_bits(bits))
    }
}
