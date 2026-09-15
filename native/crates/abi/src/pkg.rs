//! The **package manifest** — the typed record at the root of one `.pkg`,
//! kore's native distribution unit (`res/PKG.md`).
//!
//! A package is a DAG subtree, `pkgs/<name>/{manifest, app/, lib/}`: the
//! manifest names the members the subtree carries, and installing the
//! package means projecting each member into the matching system view —
//! an `App` member becomes a Link edge under `apps/`, a `Lib` member a Link
//! edge under `libs/` — so the same immutable object is reachable by two
//! names and no byte is copied.  Uninstalling detaches the edges; the
//! subtree stays the package's own until it is dropped.  Versions are
//! object versions: the manifest of version *n+1* is a new version of the
//! same store object, so a rollback is a `ReadVersion` of the old one.
//!
//! This record is DATA.  It carries no authority: the installer's grant
//! over `apps/` and `libs/` (`sys-app`, `sys-lib`) is what projects an
//! edge, and a member's `EXECUTABLE_IMAGE` stamp plus exec's
//! `BIND_EXECUTABLE` right are what make it runnable.  A manifest that
//! names a member the subtree does not carry is an incomplete package,
//! refused at install, never a dangling name in a view.
//!
//! Frozen, little-endian, `Unaligned` so it is read straight out of a
//! `ReadVersion` reply; additive evolution claims `reserved` and bumps
//! [`PACKAGE_ABI`].

use zerocopy::byteorder::{LittleEndian, U16, U32};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

/// Layout version.  A zeroed record reads `abi == 0` and is *not a
/// manifest*; a reader refuses rather than guessing.
pub const PACKAGE_ABI: u16 = 1;
/// Members one manifest may declare.  The bootstrap archive today carries
/// 33 executables and 3 libraries (`BOOTSTRAP_EXECUTABLES` /
/// `BOOTSTRAP_LIBRARIES` in exec), and the system package `kore` must list
/// every one of them; 48 leaves a third free before the record grows a
/// second page.  `size_of::<PackageManifest>()` is pinned below.
pub const MAX_PACKAGE_MEMBERS: usize = 48;
/// A member or package name: the store's own name bound
/// ([`FS_NAME_BYTES`](crate::svc::FS_NAME_BYTES)), zero-padded, so a
/// manifest name is always a legal DagFS name and never needs clamping.
pub const PACKAGE_NAME_BYTES: usize = crate::svc::FS_NAME_BYTES;

/// Which system view a member projects into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum MemberKind {
    /// `pkgs/<pkg>/app/<name>` → `apps/…/<name>` (the executable's own
    /// projection rules decide the directory beneath `apps/`).
    App = 1,
    /// `pkgs/<pkg>/lib/<name>` → `libs/<name>`.
    Lib = 2,
}

impl MemberKind {
    /// Total decode: an unknown kind is `None`, and a manifest holding one
    /// fails [`PackageManifest::validate`] — a newer writer's kind is not
    /// something an older installer may guess about.
    pub const fn from_bits(bits: u8) -> Option<Self> {
        match bits {
            1 => Some(Self::App),
            2 => Some(Self::Lib),
            _ => None,
        }
    }

    pub const fn to_bits(self) -> u8 {
        self as u8
    }
}

/// One declared member.  40 bytes, frozen.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct PackageMember {
    /// [`MemberKind`] bits.
    pub kind: u8,
    /// Name length in bytes (1..=[`PACKAGE_NAME_BYTES`]).
    pub name_len: u8,
    pub reserved: [u8; 2],
    /// The launch-profile bits an `App` member was built against
    /// (`kore_abi::founding` profile vocabulary); 0 for a `Lib` member.
    /// Descriptive: exec's launcher bundle, authenticated from the image
    /// archive, remains the profile authority.
    pub profile: U32<LittleEndian>,
    /// The member's name in its subtree directory, zero-padded.
    pub name: [u8; PACKAGE_NAME_BYTES],
}

impl PackageMember {
    /// A zeroed member: the padding rows past `member_count`.
    pub const EMPTY: Self = Self {
        kind: 0,
        name_len: 0,
        reserved: [0; 2],
        profile: U32::ZERO,
        name: [0; PACKAGE_NAME_BYTES],
    };

    /// Build one member; `None` if the name is empty or too long.
    pub fn new(kind: MemberKind, name: &[u8], profile: u32) -> Option<Self> {
        if name.is_empty() || name.len() > PACKAGE_NAME_BYTES {
            return None;
        }
        let mut stored = [0u8; PACKAGE_NAME_BYTES];
        stored[..name.len()].copy_from_slice(name);
        Some(Self {
            kind: kind.to_bits(),
            name_len: name.len() as u8,
            reserved: [0; 2],
            profile: U32::new(profile),
            name: stored,
        })
    }

    pub fn kind(&self) -> Option<MemberKind> {
        MemberKind::from_bits(self.kind)
    }

    /// The name bytes, or `None` when `name_len` is out of range.
    pub fn name(&self) -> Option<&[u8]> {
        let len = usize::from(self.name_len);
        (1..=PACKAGE_NAME_BYTES)
            .contains(&len)
            .then(|| &self.name[..len])
    }
}

/// The manifest record.  Frozen; `size_of` pinned below.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct PackageManifest {
    /// [`PACKAGE_ABI`].
    pub abi: U16<LittleEndian>,
    /// Live rows in `members`.
    pub member_count: U16<LittleEndian>,
    /// The package's own version counter, monotone per package name.  The
    /// store's object version is the authority for *which bytes*; this is
    /// the number a human and an installer compare.
    pub version: U32<LittleEndian>,
    /// Package name length (1..=[`PACKAGE_NAME_BYTES`]).
    pub name_len: u8,
    pub reserved: [u8; 7],
    /// The package name: its directory beneath `pkgs/`, zero-padded.
    pub name: [u8; PACKAGE_NAME_BYTES],
    pub members: [PackageMember; MAX_PACKAGE_MEMBERS],
}

/// Why a record is not a manifest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ManifestError {
    /// `abi` is not [`PACKAGE_ABI`] (0 = zeroed, never written).
    Abi(u16),
    /// `member_count` exceeds [`MAX_PACKAGE_MEMBERS`].
    Count(u16),
    /// The package name length is out of range.
    Name,
    /// Member `index` has an unknown kind or an out-of-range name.
    Member(u16),
    /// Two live members share a kind and a name.
    Duplicate(u16),
    /// A row past `member_count` is not zeroed: the record was not written
    /// by this contract.
    Tail(u16),
}

impl PackageManifest {
    /// A manifest with no members.  `None` if the name is out of range.
    pub fn new(name: &[u8], version: u32) -> Option<Self> {
        if name.is_empty() || name.len() > PACKAGE_NAME_BYTES {
            return None;
        }
        let mut stored = [0u8; PACKAGE_NAME_BYTES];
        stored[..name.len()].copy_from_slice(name);
        Some(Self {
            abi: U16::new(PACKAGE_ABI),
            member_count: U16::ZERO,
            version: U32::new(version),
            name_len: name.len() as u8,
            reserved: [0; 7],
            name: stored,
            members: [PackageMember::EMPTY; MAX_PACKAGE_MEMBERS],
        })
    }

    /// Append a member; `Err(())` when the manifest is full or the member
    /// duplicates a live `(kind, name)`.
    pub fn push(&mut self, member: PackageMember) -> Result<(), ()> {
        let count = usize::from(self.member_count.get());
        if count >= MAX_PACKAGE_MEMBERS {
            return Err(());
        }
        if self
            .members()
            .iter()
            .any(|live| live.kind == member.kind && live.name() == member.name())
        {
            return Err(());
        }
        self.members[count] = member;
        self.member_count = U16::new((count + 1) as u16);
        Ok(())
    }

    /// The package name, or `None` when the length is out of range.
    pub fn name(&self) -> Option<&[u8]> {
        let len = usize::from(self.name_len);
        (1..=PACKAGE_NAME_BYTES)
            .contains(&len)
            .then(|| &self.name[..len])
    }

    /// The live members (clamped to the record; `validate` is what refuses
    /// an over-long count).
    pub fn members(&self) -> &[PackageMember] {
        let count = usize::from(self.member_count.get()).min(MAX_PACKAGE_MEMBERS);
        &self.members[..count]
    }

    /// Everything a reader must check before trusting a member row.
    pub fn validate(&self) -> Result<(), ManifestError> {
        let abi = self.abi.get();
        if abi != PACKAGE_ABI {
            return Err(ManifestError::Abi(abi));
        }
        let count = self.member_count.get();
        if usize::from(count) > MAX_PACKAGE_MEMBERS {
            return Err(ManifestError::Count(count));
        }
        if self.name().is_none() {
            return Err(ManifestError::Name);
        }
        let live = usize::from(count);
        for (index, member) in self.members[..live].iter().enumerate() {
            if member.kind().is_none() || member.name().is_none() {
                return Err(ManifestError::Member(index as u16));
            }
            if self.members[..index]
                .iter()
                .any(|earlier| earlier.kind == member.kind && earlier.name() == member.name())
            {
                return Err(ManifestError::Duplicate(index as u16));
            }
        }
        if let Some(index) = self.members[live..]
            .iter()
            .position(|tail| *tail != PackageMember::EMPTY)
        {
            return Err(ManifestError::Tail((live + index) as u16));
        }
        Ok(())
    }
}

const _: () = assert!(core::mem::size_of::<PackageMember>() == 8 + PACKAGE_NAME_BYTES);
/// 16-byte header + 32-byte name + 48 × 40-byte members = 1968 bytes: one
/// `ReadVersion` transfer (`MAX_TRANSFER` = 4096) reads a whole manifest.
const _: () = assert!(
    core::mem::size_of::<PackageManifest>()
        == 16 + PACKAGE_NAME_BYTES + MAX_PACKAGE_MEMBERS * (8 + PACKAGE_NAME_BYTES)
);

#[cfg(test)]
mod tests {
    use super::*;

    fn system_package() -> PackageManifest {
        let mut manifest = PackageManifest::new(b"kore", 1).unwrap();
        manifest
            .push(PackageMember::new(MemberKind::App, b"lua", 0b1011).unwrap())
            .unwrap();
        manifest
            .push(PackageMember::new(MemberKind::Lib, b"lua_demo.so", 0).unwrap())
            .unwrap();
        manifest
    }

    #[test]
    fn a_manifest_round_trips_through_its_bytes_and_validates() {
        let manifest = system_package();
        let bytes = manifest.as_bytes();
        assert_eq!(bytes.len(), 1968);
        let read = PackageManifest::read_from_bytes(bytes).unwrap();
        assert_eq!(read, manifest);
        assert_eq!(read.validate(), Ok(()));
        assert_eq!(read.name(), Some(&b"kore"[..]));
        let members = read.members();
        assert_eq!(members.len(), 2);
        assert_eq!(members[0].kind(), Some(MemberKind::App));
        assert_eq!(members[0].name(), Some(&b"lua"[..]));
        assert_eq!(members[0].profile.get(), 0b1011);
        assert_eq!(members[1].kind(), Some(MemberKind::Lib));
        assert_eq!(members[1].name(), Some(&b"lua_demo.so"[..]));
    }

    #[test]
    fn a_zeroed_record_is_not_a_manifest() {
        let zero = PackageManifest::read_from_bytes(&[0u8; 1968]).unwrap();
        assert_eq!(zero.validate(), Err(ManifestError::Abi(0)));
    }

    #[test]
    fn every_malformed_row_is_named() {
        let mut over = system_package();
        over.member_count = U16::new(MAX_PACKAGE_MEMBERS as u16 + 1);
        assert_eq!(
            over.validate(),
            Err(ManifestError::Count(MAX_PACKAGE_MEMBERS as u16 + 1))
        );

        let mut unnamed = system_package();
        unnamed.name_len = 0;
        assert_eq!(unnamed.validate(), Err(ManifestError::Name));

        let mut kind = system_package();
        kind.members[1].kind = 9;
        assert_eq!(kind.validate(), Err(ManifestError::Member(1)));

        let mut long = system_package();
        long.members[0].name_len = PACKAGE_NAME_BYTES as u8 + 1;
        assert_eq!(long.validate(), Err(ManifestError::Member(0)));

        let mut twice = system_package();
        twice.members[2] = twice.members[0];
        twice.member_count = U16::new(3);
        assert_eq!(twice.validate(), Err(ManifestError::Duplicate(2)));

        let mut dirty = system_package();
        dirty.members[5].name_len = 1;
        assert_eq!(dirty.validate(), Err(ManifestError::Tail(5)));
    }

    #[test]
    fn push_refuses_duplicates_and_the_forty_ninth_member() {
        let mut manifest = system_package();
        assert!(
            manifest
                .push(PackageMember::new(MemberKind::App, b"lua", 0).unwrap())
                .is_err(),
            "same kind and name"
        );
        assert!(
            manifest
                .push(PackageMember::new(MemberKind::Lib, b"lua", 0).unwrap())
                .is_ok(),
            "a library may share an app's name"
        );
        while manifest.members().len() < MAX_PACKAGE_MEMBERS {
            let n = manifest.members().len() as u8;
            let name = [b'm', b'0' + n / 10, b'0' + n % 10];
            manifest
                .push(PackageMember::new(MemberKind::Lib, &name, 0).unwrap())
                .unwrap();
        }
        assert!(
            manifest
                .push(PackageMember::new(MemberKind::Lib, b"one-more", 0).unwrap())
                .is_err()
        );
        assert_eq!(manifest.validate(), Ok(()));
    }

    #[test]
    fn names_obey_the_store_bound() {
        assert!(PackageMember::new(MemberKind::Lib, b"", 0).is_none());
        assert!(PackageMember::new(MemberKind::Lib, &[b'x'; PACKAGE_NAME_BYTES], 0).is_some());
        assert!(PackageMember::new(MemberKind::Lib, &[b'x'; PACKAGE_NAME_BYTES + 1], 0).is_none());
        assert!(PackageManifest::new(b"", 0).is_none());
        assert_eq!(MemberKind::from_bits(0), None);
    }
}
