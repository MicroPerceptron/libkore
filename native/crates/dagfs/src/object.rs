use kore_collections::SlotId as SlotKey;

/// Volume-stable object identity. Unlike [`ObjectId`], this value survives a
/// service restart and is safe to serialize because it conveys no authority.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash, PartialOrd, Ord)]
pub struct StableObjectId(u64);

impl StableObjectId {
    pub const INVALID: Self = Self(0);

    pub const fn new(value: u64) -> Option<Self> {
        if value == 0 { None } else { Some(Self(value)) }
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

/// A file or directory object's service-local identity.
///
/// Possessing this naming value conveys no authority. Reaching the object
/// always requires a live, pair-bound route; fabricated or stale bits fail
/// when the service resolves the generational slot.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash, PartialOrd, Ord)]
pub struct ObjectId(SlotKey);

impl ObjectId {
    /// The never-valid object id.
    pub const INVALID: Self = Self(SlotKey::INVALID);

    /// Pack for wire or diagnostic use. A fabricated word fails closed when
    /// resolved, exactly as a fabricated generational slot id does.
    #[inline]
    pub const fn to_bits(self) -> u64 {
        self.0.to_bits()
    }

    #[inline]
    pub const fn from_bits(bits: u64) -> Self {
        Self(SlotKey::from_bits(bits))
    }
}

/// What an object is — its **class**, which governs the operations that are
/// valid on it: a directory is listed and holds names, a file is read and
/// written. POSIX projections such as symlinks or device nodes do not leak
/// into this native core; a second name for an object is a `Link` edge.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ObjectKind {
    File,
    Directory,
}

impl ObjectKind {
    #[inline]
    pub const fn is_dir(self) -> bool {
        matches!(self, ObjectKind::Directory)
    }

    #[inline]
    pub const fn is_file(self) -> bool {
        matches!(self, ObjectKind::File)
    }
}

/// A file's monotonically increasing committed revision.
pub type Revision = u64;

/// The unconditional-write sentinel for a request's expected revision.
/// Live revisions begin above zero.
pub const ANY_REVISION: Revision = 0;

/// What an object *contains* — one byte of **data, never authority**, set
/// at creation and immutable for the object's life.
///
/// The class ([`ObjectKind`]) says what may be done to an object; the
/// subtype says how its bytes are meant to be read, so a consumer need not
/// infer it from a name suffix or a magic number. It gates nothing:
/// `Rights::BIND_EXECUTABLE`, the executable version lease, and the image
/// service's descriptor validation remain the only executable authority,
/// and an [`EXECUTABLE_IMAGE`](Self::EXECUTABLE_IMAGE) file is opened,
/// read, and written exactly as a plain one is. Directories are always
/// [`PLAIN`](Self::PLAIN). Unknown values are carried, not refused: a newer
/// writer's subtype is data an older reader passes through intact.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash, PartialOrd, Ord)]
pub struct ObjectSubtype(u8);

impl ObjectSubtype {
    /// No particular interpretation — every object before subtypes existed.
    pub const PLAIN: Self = Self(0);
    /// An executable image the store installer wrote: what exec's seeder
    /// re-adopts by digest, and what `inspect` reports.
    pub const EXECUTABLE_IMAGE: Self = Self(1);
    /// A package manifest (`kore_abi::pkg::PackageManifest`): the typed
    /// record at the root of one installed `.pkg` subtree (`res/PKG.md`).
    /// Data, like every subtype -- an installer reads it to decide which
    /// Link edges to project into `apps/` and `libs/`; it grants nothing.
    pub const PACKAGE_MANIFEST: Self = Self(2);

    #[inline]
    pub const fn to_bits(self) -> u8 {
        self.0
    }

    /// Total: every byte is a subtype, known or not.
    #[inline]
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    #[inline]
    pub const fn is_plain(self) -> bool {
        self.0 == 0
    }
}

#[cfg(test)]
mod subtype_tests {
    use super::*;

    #[test]
    fn object_subtype_bits_round_trip_including_unknown_values() {
        for bits in [0u8, 1, 2, 0x7f, u8::MAX] {
            assert_eq!(ObjectSubtype::from_bits(bits).to_bits(), bits);
        }
        assert!(ObjectSubtype::PLAIN.is_plain());
        assert!(!ObjectSubtype::EXECUTABLE_IMAGE.is_plain());
        assert_eq!(ObjectSubtype::default(), ObjectSubtype::PLAIN);
    }
}
