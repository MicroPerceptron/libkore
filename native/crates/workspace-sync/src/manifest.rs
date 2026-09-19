//! Canonical snapshot manifests and their root identity.

use heapless::Vec;
use zerocopy::byteorder::little_endian::{U16, U32, U64};

use crate::{
    bounds::{
        FORMAT_VERSION, MAX_CHILDREN_PER_DIRECTORY, MAX_DIRECTORIES, MAX_ENTRIES, MAX_FILE_BYTES,
        MAX_SNAPSHOT_BYTES, POLICY_VERSION,
    },
    path::{CanonicalPath, PathError},
    wire::{
        ContentDigest, ENTRY_MAGIC, EntryFlags, EntryHeader, EntryKind, SNAPSHOT_MAGIC,
        SnapshotHeader, SnapshotId, SourceCorrelation,
    },
};

const POLICY_DOMAIN: &[u8] = b"kore.workspace.policy.v1\0";
const ROOT_DOMAIN: &[u8] = b"kore.workspace.root.v1\0";
const SNAPSHOT_DOMAIN: &[u8] = b"kore.workspace.snapshot.v1\0";
const DIRECTORY_DOMAIN: &[u8] = b"kore.workspace.directory.v1\0";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Policy {
    pub version: u16,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            version: POLICY_VERSION,
        }
    }
}

impl Policy {
    pub fn digest(self) -> ContentDigest {
        let mut hash = blake3::Hasher::new();
        hash.update(POLICY_DOMAIN);
        hash.update(&self.version.to_le_bytes());
        hash.update(&(crate::bounds::MAX_COMPONENT_BYTES as u16).to_le_bytes());
        hash.update(&(crate::bounds::MAX_PATH_DEPTH as u16).to_le_bytes());
        hash.update(&(MAX_ENTRIES as u16).to_le_bytes());
        hash.update(&(MAX_DIRECTORIES as u16).to_le_bytes());
        hash.update(&(MAX_CHILDREN_PER_DIRECTORY as u16).to_le_bytes());
        hash.update(&MAX_FILE_BYTES.to_le_bytes());
        hash.update(&MAX_SNAPSHOT_BYTES.to_le_bytes());
        *hash.finalize().as_bytes()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Entry {
    pub path: CanonicalPath,
    pub kind: EntryKind,
    pub flags: EntryFlags,
    pub content_len: u32,
    pub digest: ContentDigest,
    pub range_root: ContentDigest,
    pub source: SourceCorrelation,
}

impl Entry {
    pub fn directory(path: CanonicalPath, source: SourceCorrelation) -> Self {
        Self {
            path,
            kind: EntryKind::Directory,
            flags: EntryFlags::default(),
            content_len: 0,
            digest: *blake3::hash(DIRECTORY_DOMAIN).as_bytes(),
            range_root: [0; 32],
            source,
        }
    }

    pub fn file(
        path: CanonicalPath,
        executable: bool,
        content_len: u32,
        digest: ContentDigest,
        source: SourceCorrelation,
    ) -> Result<Self, ManifestError> {
        Self::file_with_range_root(path, executable, content_len, digest, digest, source)
    }

    pub fn file_with_range_root(
        path: CanonicalPath,
        executable: bool,
        content_len: u32,
        digest: ContentDigest,
        range_root: ContentDigest,
        source: SourceCorrelation,
    ) -> Result<Self, ManifestError> {
        if u64::from(content_len) > MAX_FILE_BYTES {
            return Err(ManifestError::FileTooLarge);
        }
        Ok(Self {
            path,
            kind: EntryKind::File,
            flags: EntryFlags::new(executable),
            content_len,
            digest,
            range_root,
            source,
        })
    }

    pub fn header(&self, index: u16) -> EntryHeader {
        EntryHeader {
            magic: ENTRY_MAGIC,
            version: U16::new(FORMAT_VERSION),
            index: U16::new(index),
            kind: self.kind as u8,
            flags: self.flags.bits(),
            depth: self.path.depth() as u8,
            reserved0: 0,
            path_len: U16::new(self.path.as_bytes().len() as u16),
            reserved1: U16::ZERO,
            content_len: U32::new(self.content_len),
            digest: self.digest,
            range_root: self.range_root,
            source: self.source,
        }
    }

    pub fn from_wire(header: &EntryHeader, path: &[u8]) -> Result<Self, ManifestError> {
        if !header.shape_is_valid()
            || usize::from(header.path_len.get()) != path.len()
            || usize::from(header.depth) > crate::bounds::MAX_PATH_DEPTH
        {
            return Err(ManifestError::Malformed);
        }
        let path = CanonicalPath::new(path).map_err(ManifestError::Path)?;
        if path.depth() != usize::from(header.depth) {
            return Err(ManifestError::Malformed);
        }
        let kind = EntryKind::from_wire(header.kind).ok_or(ManifestError::Malformed)?;
        let flags = EntryFlags::from_bits(header.flags).ok_or(ManifestError::Malformed)?;
        let entry = Self {
            path,
            kind,
            flags,
            content_len: header.content_len.get(),
            digest: header.digest,
            range_root: header.range_root,
            source: header.source,
        };
        entry.validate_shape()?;
        Ok(entry)
    }

    fn validate_shape(&self) -> Result<(), ManifestError> {
        match self.kind {
            EntryKind::Directory => {
                if self.content_len != 0
                    || self.flags.executable()
                    || self.digest != *blake3::hash(DIRECTORY_DOMAIN).as_bytes()
                    || self.range_root != [0; 32]
                {
                    return Err(ManifestError::Malformed);
                }
            }
            EntryKind::File if u64::from(self.content_len) > MAX_FILE_BYTES => {
                return Err(ManifestError::FileTooLarge);
            }
            EntryKind::File => {}
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ManifestError {
    Path(PathError),
    Full,
    TooManyDirectories,
    FileTooLarge,
    SnapshotTooLarge,
    DuplicatePath,
    PortableCollision,
    MissingParent,
    ParentNotDirectory,
    Malformed,
    DigestMismatch,
    WrongPolicy,
}

#[derive(Clone, Debug)]
pub struct Manifest {
    header: SnapshotHeader,
    entries: Vec<Entry, MAX_ENTRIES>,
}

impl Manifest {
    pub fn header(&self) -> &SnapshotHeader {
        &self.header
    }

    pub fn entries(&self) -> &[Entry] {
        self.entries.as_slice()
    }

    pub fn snapshot_id(&self) -> SnapshotId {
        self.header.snapshot_id
    }

    pub fn verify(&self, policy: Policy) -> Result<(), ManifestError> {
        if !self.header.shape_is_valid()
            || self.header.policy_version.get() != policy.version
            || self.header.policy_digest != policy.digest()
            || usize::from(self.header.entry_count.get()) != self.entries.len()
        {
            return Err(ManifestError::WrongPolicy);
        }
        validate_entries(self.entries.as_slice())?;
        let (directories, total, root) = root_digest(self.entries.as_slice(), policy.digest());
        if self.header.directory_count.get() != directories
            || self.header.total_bytes.get() != total
            || self.header.root_digest != root
            || self.header.snapshot_id
                != snapshot_id(root, self.header.parent_snapshot, policy.digest())
        {
            return Err(ManifestError::DigestMismatch);
        }
        Ok(())
    }
}

pub struct ManifestBuilder {
    parent: SnapshotId,
    policy: Policy,
    entries: Vec<Entry, MAX_ENTRIES>,
}

impl ManifestBuilder {
    pub const fn new(parent: SnapshotId, policy: Policy) -> Self {
        Self {
            parent,
            policy,
            entries: Vec::new(),
        }
    }

    pub fn push(&mut self, entry: Entry) -> Result<(), ManifestError> {
        entry.validate_shape()?;
        self.entries.push(entry).map_err(|_| ManifestError::Full)
    }

    pub fn finish(mut self) -> Result<Manifest, ManifestError> {
        self.entries
            .as_mut_slice()
            .sort_unstable_by(|left, right| left.path.cmp(&right.path));
        validate_entries(self.entries.as_slice())?;
        let policy_digest = self.policy.digest();
        let (directories, total, root_digest) = root_digest(self.entries.as_slice(), policy_digest);
        let snapshot_id = snapshot_id(root_digest, self.parent, policy_digest);
        let header = SnapshotHeader {
            magic: SNAPSHOT_MAGIC,
            version: U16::new(FORMAT_VERSION),
            policy_version: U16::new(self.policy.version),
            entry_count: U16::new(self.entries.len() as u16),
            directory_count: U16::new(directories),
            reserved: U32::ZERO,
            total_bytes: U64::new(total),
            parent_snapshot: self.parent,
            policy_digest,
            root_digest,
            snapshot_id,
        };
        Ok(Manifest {
            header,
            entries: self.entries,
        })
    }
}

fn validate_entries(entries: &[Entry]) -> Result<(), ManifestError> {
    let mut directories = 0usize;
    let mut total = 0u64;
    for (index, entry) in entries.iter().enumerate() {
        entry.validate_shape()?;
        if entry.kind == EntryKind::Directory {
            directories += 1;
            if directories > MAX_DIRECTORIES {
                return Err(ManifestError::TooManyDirectories);
            }
        } else {
            total = total
                .checked_add(u64::from(entry.content_len))
                .ok_or(ManifestError::SnapshotTooLarge)?;
            if total > MAX_SNAPSHOT_BYTES {
                return Err(ManifestError::SnapshotTooLarge);
            }
        }
        for earlier in &entries[..index] {
            if earlier.path == entry.path {
                return Err(ManifestError::DuplicatePath);
            }
            if earlier.path.portable_eq(&entry.path) {
                return Err(ManifestError::PortableCollision);
            }
        }
        if let Some(parent) = entry.path.parent() {
            match entries.binary_search_by(|candidate| candidate.path.cmp(&parent)) {
                Ok(parent_index) if entries[parent_index].kind == EntryKind::Directory => {}
                Ok(_) => return Err(ManifestError::ParentNotDirectory),
                Err(_) => return Err(ManifestError::MissingParent),
            }
        }
        let siblings = entries
            .iter()
            .filter(|candidate| candidate.path.parent() == entry.path.parent())
            .count();
        if siblings > MAX_CHILDREN_PER_DIRECTORY {
            return Err(ManifestError::Full);
        }
    }
    Ok(())
}

fn root_digest(entries: &[Entry], policy_digest: ContentDigest) -> (u16, u64, ContentDigest) {
    let directories = entries
        .iter()
        .filter(|entry| entry.kind == EntryKind::Directory)
        .count() as u16;
    let total = entries
        .iter()
        .map(|entry| u64::from(entry.content_len))
        .sum::<u64>();
    let mut hash = blake3::Hasher::new();
    hash.update(ROOT_DOMAIN);
    hash.update(&policy_digest);
    hash.update(&(entries.len() as u16).to_le_bytes());
    hash.update(&directories.to_le_bytes());
    hash.update(&total.to_le_bytes());
    for (index, entry) in entries.iter().enumerate() {
        hash.update(&(index as u16).to_le_bytes());
        hash.update(&[
            entry.kind as u8,
            entry.flags.bits(),
            entry.path.depth() as u8,
        ]);
        hash.update(&(entry.path.as_bytes().len() as u16).to_le_bytes());
        hash.update(&entry.content_len.to_le_bytes());
        hash.update(&entry.digest);
        hash.update(&entry.range_root);
        // `source` is an importer-private, session-scoped correlation. It is
        // deliberately excluded from the canonical namespace identity: a
        // reconnect may mint new adapter object IDs without changing the
        // kore snapshot represented by the same paths, metadata, and bytes.
        hash.update(entry.path.as_bytes());
    }
    (directories, total, *hash.finalize().as_bytes())
}

fn snapshot_id(root: ContentDigest, parent: SnapshotId, policy: ContentDigest) -> SnapshotId {
    let mut hash = blake3::Hasher::new();
    hash.update(SNAPSHOT_DOMAIN);
    hash.update(&root);
    hash.update(&parent);
    hash.update(&policy);
    *hash.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(value: &str) -> CanonicalPath {
        CanonicalPath::new(value.as_bytes()).unwrap()
    }

    #[test]
    fn canonical_order_makes_snapshot_identity_deterministic() {
        let file = Entry::file(
            path("src/lib.rs"),
            false,
            4,
            *blake3::hash(b"kore").as_bytes(),
            [2; 16],
        )
        .unwrap();
        let dir = Entry::directory(path("src"), [1; 16]);
        let build = |reverse| {
            let mut builder = ManifestBuilder::new([0; 32], Policy::default());
            if reverse {
                builder.push(file.clone()).unwrap();
                builder.push(dir.clone()).unwrap();
            } else {
                builder.push(dir.clone()).unwrap();
                builder.push(file.clone()).unwrap();
            }
            builder.finish().unwrap()
        };
        let left = build(false);
        let right = build(true);
        assert_eq!(left.snapshot_id(), right.snapshot_id());
        assert_eq!(left.entries(), right.entries());
        left.verify(Policy::default()).unwrap();
    }

    #[test]
    fn session_source_correlations_do_not_change_namespace_identity() {
        let build = |source| {
            let mut builder = ManifestBuilder::new([0; 32], Policy::default());
            builder
                .push(
                    Entry::file(
                        path("file"),
                        false,
                        4,
                        *blake3::hash(b"kore").as_bytes(),
                        source,
                    )
                    .unwrap(),
                )
                .unwrap();
            builder.finish().unwrap()
        };
        let first = build([1; 16]);
        let reconnected = build([2; 16]);
        assert_eq!(first.snapshot_id(), reconnected.snapshot_id());
        assert_ne!(first.entries()[0].source, reconnected.entries()[0].source);
    }

    #[test]
    fn parent_and_portable_collisions_fail_closed() {
        let mut missing = ManifestBuilder::new([0; 32], Policy::default());
        missing
            .push(
                Entry::file(
                    path("src/lib.rs"),
                    false,
                    0,
                    *blake3::hash(b"").as_bytes(),
                    [0; 16],
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(missing.finish().unwrap_err(), ManifestError::MissingParent);

        let mut collision = ManifestBuilder::new([0; 32], Policy::default());
        collision
            .push(Entry::directory(path("Src"), [0; 16]))
            .unwrap();
        collision
            .push(Entry::directory(path("src"), [1; 16]))
            .unwrap();
        assert_eq!(
            collision.finish().unwrap_err(),
            ManifestError::PortableCollision
        );
    }

    #[test]
    fn entry_wire_round_trip_is_exact() {
        let entry = Entry::file(
            path("main.rs"),
            true,
            3,
            *blake3::hash(b"abc").as_bytes(),
            [9; 16],
        )
        .unwrap();
        let head = entry.header(7);
        assert_eq!(Entry::from_wire(&head, entry.path.as_bytes()), Ok(entry));
    }

    #[test]
    fn a_directory_cannot_exceed_the_frozen_sibling_bound() {
        let mut builder = ManifestBuilder::new([0; 32], Policy::default());
        for index in 0..=MAX_CHILDREN_PER_DIRECTORY {
            let name = std::format!("f{index:02}");
            builder
                .push(
                    Entry::file(
                        path(&name),
                        false,
                        0,
                        *blake3::hash(b"").as_bytes(),
                        [index as u8; 16],
                    )
                    .unwrap(),
                )
                .unwrap();
        }
        assert_eq!(builder.finish().unwrap_err(), ManifestError::Full);
    }
}
