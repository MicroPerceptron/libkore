//! Atomic import state machine over a storage contract.

use crate::{
    bounds::MAX_CHUNK_BYTES,
    manifest::{Entry, Manifest, ManifestError, Policy},
    path::CanonicalPath,
    wire::{ContentDigest, EntryKind, SnapshotHeader},
};

#[allow(async_fn_in_trait)]
pub trait ImportStore {
    type Error;
    type Staging;
    type File;
    type Commit: Clone;

    async fn begin(
        &mut self,
        snapshot: &SnapshotHeader,
    ) -> Result<BeginOutcome<Self::Staging, Self::Commit>, Self::Error>;
    async fn create_directory(
        &mut self,
        staging: &mut Self::Staging,
        entry: &Entry,
    ) -> Result<(), Self::Error>;
    async fn begin_file(
        &mut self,
        staging: &mut Self::Staging,
        entry: &Entry,
    ) -> Result<Self::File, Self::Error>;
    async fn write_file(
        &mut self,
        staging: &mut Self::Staging,
        file: &mut Self::File,
        offset: u32,
        bytes: &[u8],
    ) -> Result<(), Self::Error>;
    async fn finish_file(
        &mut self,
        staging: &mut Self::Staging,
        file: Self::File,
        entry: &Entry,
    ) -> Result<(), Self::Error>;
    async fn commit(
        &mut self,
        staging: Self::Staging,
        snapshot: &SnapshotHeader,
    ) -> Result<Self::Commit, CommitFailure<Self::Error, Self::Staging>>;
    async fn abort(&mut self, staging: Self::Staging);
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BeginOutcome<S, C> {
    Staging(S),
    AlreadyCommitted(C),
}

/// A failed publication returns the still-private staging owner. Cleanup is
/// therefore an obligation carried by the type, not a best effort after the
/// only handle was lost.
#[derive(Debug, Eq, PartialEq)]
pub struct CommitFailure<E, T> {
    pub error: E,
    pub staging: T,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImportPhase {
    Contents,
    Ready,
    Committed,
    Aborted,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StagedFile {
    pub manifest_index: u16,
    pub path: CanonicalPath,
    pub content_len: u32,
    pub digest: ContentDigest,
}

#[derive(Debug, Eq, PartialEq)]
pub enum ImportError<E> {
    Manifest(ManifestError),
    Store(E),
    WrongPhase,
    FileAlreadyOpen,
    NoFileOpen,
    WrongOffset,
    ChunkTooLarge,
    LengthOverflow,
    DigestMismatch,
    ContentIncomplete,
}

#[allow(async_fn_in_trait)]
pub trait ImportSource {
    type Error;

    /// Read the exact next bytes of one manifest file. Returning zero before
    /// `content_len` is a refusal, never an implicit end-of-file success.
    async fn read(
        &mut self,
        file: &StagedFile,
        offset: u32,
        destination: &mut [u8],
    ) -> Result<usize, Self::Error>;
}

#[derive(Debug, Eq, PartialEq)]
pub enum DriveError<Store, Source> {
    Import(ImportError<Store>),
    Source(Source),
    ZeroProgress,
}

/// Drive one canonical manifest from any bounded byte source into any
/// capability store. Offline artifacts and HostLink implement the two traits;
/// neither transport gets a second filesystem policy engine.
pub async fn import_from<S: ImportStore, R: ImportSource>(
    store: S,
    manifest: Manifest,
    policy: Policy,
    source: &mut R,
) -> Result<(S, S::Commit), DriveError<S::Error, R::Error>> {
    let mut transaction = ImportTransaction::begin(store, manifest, policy)
        .await
        .map_err(DriveError::Import)?;
    if transaction.phase() == ImportPhase::Committed {
        let commit = transaction.commit().await.map_err(DriveError::Import)?;
        let store = transaction
            .into_store()
            .ok_or(DriveError::Import(ImportError::WrongPhase))?;
        return Ok((store, commit));
    }
    let mut chunk = [0u8; MAX_CHUNK_BYTES];
    while let Some(file) = transaction.next_file() {
        let opened = match transaction.begin_next_file().await {
            Ok(opened) => opened,
            Err(error) => {
                transaction.abort().await;
                return Err(DriveError::Import(error));
            }
        };
        debug_assert_eq!(opened, file);
        let mut offset = 0u32;
        while offset < file.content_len {
            let remaining = usize::try_from(file.content_len - offset)
                .unwrap_or(MAX_CHUNK_BYTES)
                .min(MAX_CHUNK_BYTES);
            let read = match source.read(&file, offset, &mut chunk[..remaining]).await {
                Ok(read) => read,
                Err(error) => {
                    transaction.abort().await;
                    return Err(DriveError::Source(error));
                }
            };
            if read == 0 || read > remaining {
                let _ = transaction.abort().await;
                return Err(DriveError::ZeroProgress);
            }
            if let Err(error) = transaction.write(offset, &chunk[..read]).await {
                transaction.abort().await;
                return Err(DriveError::Import(error));
            }
            offset = offset
                .checked_add(
                    u32::try_from(read)
                        .map_err(|_| DriveError::Import(ImportError::LengthOverflow))?,
                )
                .ok_or(DriveError::Import(ImportError::LengthOverflow))?;
        }
        if let Err(error) = transaction.finish_file().await {
            // Digest failure already consumes staging; abort is idempotent in
            // that phase and still returns the store owner.
            transaction.abort().await;
            return Err(DriveError::Import(error));
        }
    }
    let commit = match transaction.commit().await {
        Ok(commit) => commit,
        Err(error) => {
            transaction.abort().await;
            return Err(DriveError::Import(error));
        }
    };
    let store = transaction
        .into_store()
        .ok_or(DriveError::Import(ImportError::WrongPhase))?;
    Ok((store, commit))
}

struct ActiveFile<F> {
    manifest_index: usize,
    file: F,
    offset: u32,
    hasher: blake3::Hasher,
}

pub struct ImportTransaction<S: ImportStore> {
    store: S,
    manifest: Manifest,
    staging: Option<S::Staging>,
    prior_commit: Option<S::Commit>,
    active: Option<ActiveFile<S::File>>,
    next_index: usize,
    phase: ImportPhase,
}

impl<S: ImportStore> ImportTransaction<S> {
    pub async fn begin(
        mut store: S,
        manifest: Manifest,
        policy: Policy,
    ) -> Result<Self, ImportError<S::Error>> {
        manifest.verify(policy).map_err(ImportError::Manifest)?;
        let mut staging = match store
            .begin(manifest.header())
            .await
            .map_err(ImportError::Store)?
        {
            BeginOutcome::Staging(staging) => staging,
            BeginOutcome::AlreadyCommitted(commit) => {
                return Ok(Self {
                    store,
                    manifest,
                    staging: None,
                    prior_commit: Some(commit),
                    active: None,
                    next_index: 0,
                    phase: ImportPhase::Committed,
                });
            }
        };
        for entry in manifest.entries() {
            if entry.kind == EntryKind::Directory
                && let Err(error) = store.create_directory(&mut staging, entry).await
            {
                store.abort(staging).await;
                return Err(ImportError::Store(error));
            }
        }
        let phase = if manifest
            .entries()
            .iter()
            .any(|entry| entry.kind == EntryKind::File)
        {
            ImportPhase::Contents
        } else {
            ImportPhase::Ready
        };
        Ok(Self {
            store,
            manifest,
            staging: Some(staging),
            prior_commit: None,
            active: None,
            next_index: 0,
            phase,
        })
    }

    pub const fn phase(&self) -> ImportPhase {
        self.phase
    }

    pub fn next_file(&self) -> Option<StagedFile> {
        if self.phase != ImportPhase::Contents || self.active.is_some() {
            return None;
        }
        let (index, entry) = self
            .manifest
            .entries()
            .iter()
            .enumerate()
            .skip(self.next_index)
            .find(|(_, entry)| entry.kind == EntryKind::File)?;
        Some(StagedFile {
            manifest_index: index as u16,
            path: entry.path.clone(),
            content_len: entry.content_len,
            digest: entry.digest,
        })
    }

    pub async fn begin_next_file(&mut self) -> Result<StagedFile, ImportError<S::Error>> {
        if self.phase != ImportPhase::Contents {
            return Err(ImportError::WrongPhase);
        }
        if self.active.is_some() {
            return Err(ImportError::FileAlreadyOpen);
        }
        let staged = self.next_file().ok_or(ImportError::ContentIncomplete)?;
        let index = usize::from(staged.manifest_index);
        let entry = &self.manifest.entries()[index];
        let file = self
            .store
            .begin_file(self.staging.as_mut().ok_or(ImportError::WrongPhase)?, entry)
            .await
            .map_err(ImportError::Store)?;
        self.active = Some(ActiveFile {
            manifest_index: index,
            file,
            offset: 0,
            hasher: blake3::Hasher::new(),
        });
        Ok(staged)
    }

    pub async fn write(&mut self, offset: u32, bytes: &[u8]) -> Result<(), ImportError<S::Error>> {
        if bytes.is_empty() || bytes.len() > MAX_CHUNK_BYTES {
            return Err(ImportError::ChunkTooLarge);
        }
        let active = self.active.as_mut().ok_or(ImportError::NoFileOpen)?;
        if active.offset != offset {
            return Err(ImportError::WrongOffset);
        }
        let entry = &self.manifest.entries()[active.manifest_index];
        let next = offset
            .checked_add(u32::try_from(bytes.len()).map_err(|_| ImportError::LengthOverflow)?)
            .ok_or(ImportError::LengthOverflow)?;
        if next > entry.content_len {
            return Err(ImportError::LengthOverflow);
        }
        self.store
            .write_file(
                self.staging.as_mut().ok_or(ImportError::WrongPhase)?,
                &mut active.file,
                offset,
                bytes,
            )
            .await
            .map_err(ImportError::Store)?;
        active.hasher.update(bytes);
        active.offset = next;
        Ok(())
    }

    pub async fn finish_file(&mut self) -> Result<(), ImportError<S::Error>> {
        let active = self.active.take().ok_or(ImportError::NoFileOpen)?;
        let entry = &self.manifest.entries()[active.manifest_index];
        if active.offset != entry.content_len {
            self.active = Some(active);
            return Err(ImportError::ContentIncomplete);
        }
        if active.hasher.finalize().as_bytes() != &entry.digest {
            if let Some(staging) = self.staging.take() {
                self.store.abort(staging).await;
            }
            self.phase = ImportPhase::Aborted;
            return Err(ImportError::DigestMismatch);
        }
        self.store
            .finish_file(
                self.staging.as_mut().ok_or(ImportError::WrongPhase)?,
                active.file,
                entry,
            )
            .await
            .map_err(ImportError::Store)?;
        self.next_index = active.manifest_index + 1;
        if self.next_file().is_none() {
            self.phase = ImportPhase::Ready;
        }
        Ok(())
    }

    pub async fn commit(&mut self) -> Result<S::Commit, ImportError<S::Error>> {
        if self.phase == ImportPhase::Committed {
            return self.prior_commit.clone().ok_or(ImportError::WrongPhase);
        }
        if self.phase != ImportPhase::Ready || self.active.is_some() {
            return Err(ImportError::WrongPhase);
        }
        let staging = self.staging.take().ok_or(ImportError::WrongPhase)?;
        let result = match self.store.commit(staging, self.manifest.header()).await {
            Ok(commit) => commit,
            Err(failure) => {
                self.staging = Some(failure.staging);
                return Err(ImportError::Store(failure.error));
            }
        };
        self.phase = ImportPhase::Committed;
        self.prior_commit = Some(result.clone());
        Ok(result)
    }

    pub async fn abort(mut self) -> S {
        if let Some(staging) = self.staging.take() {
            self.store.abort(staging).await;
        }
        self.phase = ImportPhase::Aborted;
        self.store
    }

    pub fn into_store(self) -> Option<S> {
        matches!(self.phase, ImportPhase::Committed | ImportPhase::Aborted).then_some(self.store)
    }
}

#[cfg(test)]
mod tests {
    use core::{future::Future, pin::pin, task::Context};
    use proptest::prelude::*;
    use std::{collections::BTreeMap, vec::Vec};

    use super::*;
    use crate::{Entry, ManifestBuilder};

    fn run<F: Future>(future: F) -> F::Output {
        let mut future = pin!(future);
        let waker = std::task::Waker::noop();
        let mut context = Context::from_waker(waker);
        for _ in 0..64 {
            if let core::task::Poll::Ready(output) = future.as_mut().poll(&mut context) {
                return output;
            }
        }
        panic!("ready-only test store parked")
    }

    #[derive(Default)]
    struct Store {
        staged: BTreeMap<std::string::String, Vec<u8>>,
        committed: bool,
        aborted: bool,
    }

    impl ImportStore for Store {
        type Error = ();
        type Staging = ();
        type File = std::string::String;
        type Commit = ();

        async fn begin(&mut self, _: &SnapshotHeader) -> Result<BeginOutcome<(), ()>, ()> {
            Ok(BeginOutcome::Staging(()))
        }
        async fn create_directory(&mut self, _: &mut (), entry: &Entry) -> Result<(), ()> {
            self.staged.insert(entry.path.as_str().into(), Vec::new());
            Ok(())
        }
        async fn begin_file(&mut self, _: &mut (), entry: &Entry) -> Result<Self::File, ()> {
            self.staged.insert(entry.path.as_str().into(), Vec::new());
            Ok(entry.path.as_str().into())
        }
        async fn write_file(
            &mut self,
            _: &mut (),
            file: &mut Self::File,
            _: u32,
            bytes: &[u8],
        ) -> Result<(), ()> {
            self.staged.get_mut(file).unwrap().extend_from_slice(bytes);
            Ok(())
        }
        async fn finish_file(&mut self, _: &mut (), _: Self::File, _: &Entry) -> Result<(), ()> {
            Ok(())
        }
        async fn commit(&mut self, _: (), _: &SnapshotHeader) -> Result<(), CommitFailure<(), ()>> {
            self.committed = true;
            Ok(())
        }
        async fn abort(&mut self, _: ()) {
            self.aborted = true;
        }
    }

    fn fixture() -> (Manifest, [u8; 5]) {
        let bytes = *b"hello";
        let mut builder = ManifestBuilder::new([0; 32], Policy::default());
        builder
            .push(Entry::directory(
                CanonicalPath::new(b"src").unwrap(),
                [1; 16],
            ))
            .unwrap();
        builder
            .push(
                Entry::file(
                    CanonicalPath::new(b"src/lib.rs").unwrap(),
                    false,
                    bytes.len() as u32,
                    *blake3::hash(&bytes).as_bytes(),
                    [2; 16],
                )
                .unwrap(),
            )
            .unwrap();
        (builder.finish().unwrap(), bytes)
    }

    #[test]
    fn import_is_private_until_exact_digest_commit() {
        let (manifest, bytes) = fixture();
        let mut tx = run(ImportTransaction::begin(
            Store::default(),
            manifest,
            Policy::default(),
        ))
        .unwrap();
        assert_eq!(
            run(tx.begin_next_file()).unwrap().path.as_str(),
            "src/lib.rs"
        );
        run(tx.write(0, &bytes[..2])).unwrap();
        assert_eq!(run(tx.write(0, &bytes[2..])), Err(ImportError::WrongOffset));
        run(tx.write(2, &bytes[2..])).unwrap();
        run(tx.finish_file()).unwrap();
        assert_eq!(tx.phase(), ImportPhase::Ready);
        run(tx.commit()).unwrap();
        let store = tx.into_store().unwrap();
        assert!(store.committed);
        assert_eq!(store.staged["src/lib.rs"], bytes);
    }

    #[test]
    fn digest_mismatch_never_becomes_ready() {
        let (manifest, mut bytes) = fixture();
        bytes[0] ^= 1;
        let mut tx = run(ImportTransaction::begin(
            Store::default(),
            manifest,
            Policy::default(),
        ))
        .unwrap();
        run(tx.begin_next_file()).unwrap();
        run(tx.write(0, &bytes)).unwrap();
        assert_eq!(run(tx.finish_file()), Err(ImportError::DigestMismatch));
        assert_eq!(tx.phase(), ImportPhase::Aborted);
        let store = tx.into_store().expect("aborted store remains recoverable");
        assert!(store.aborted);
        assert!(!store.committed);
    }

    #[test]
    fn failed_commit_returns_staging_for_explicit_abort() {
        #[derive(Default)]
        struct FailingStore {
            aborted: bool,
        }

        impl ImportStore for FailingStore {
            type Error = u8;
            type Staging = u8;
            type File = ();
            type Commit = ();

            async fn begin(&mut self, _: &SnapshotHeader) -> Result<BeginOutcome<u8, ()>, u8> {
                Ok(BeginOutcome::Staging(7))
            }
            async fn create_directory(&mut self, _: &mut u8, _: &Entry) -> Result<(), u8> {
                Ok(())
            }
            async fn begin_file(&mut self, _: &mut u8, _: &Entry) -> Result<(), u8> {
                Ok(())
            }
            async fn write_file(
                &mut self,
                _: &mut u8,
                _: &mut (),
                _: u32,
                _: &[u8],
            ) -> Result<(), u8> {
                Ok(())
            }
            async fn finish_file(&mut self, _: &mut u8, _: (), _: &Entry) -> Result<(), u8> {
                Ok(())
            }
            async fn commit(
                &mut self,
                staging: u8,
                _: &SnapshotHeader,
            ) -> Result<(), CommitFailure<u8, u8>> {
                Err(CommitFailure { error: 9, staging })
            }
            async fn abort(&mut self, staging: u8) {
                assert_eq!(staging, 7);
                self.aborted = true;
            }
        }

        let mut builder = ManifestBuilder::new([0; 32], Policy::default());
        builder
            .push(Entry::directory(
                CanonicalPath::new(b"empty").unwrap(),
                [0; 16],
            ))
            .unwrap();
        let manifest = builder.finish().unwrap();
        let mut tx = run(ImportTransaction::begin(
            FailingStore::default(),
            manifest,
            Policy::default(),
        ))
        .unwrap();
        assert_eq!(run(tx.commit()), Err(ImportError::Store(9)));
        assert!(run(tx.abort()).aborted);
    }

    struct FailingSource;

    impl ImportSource for FailingSource {
        type Error = u8;

        async fn read(
            &mut self,
            _: &StagedFile,
            _: u32,
            _: &mut [u8],
        ) -> Result<usize, Self::Error> {
            Err(7)
        }
    }

    #[test]
    fn the_generic_driver_reclaims_staging_on_source_failure() {
        let (manifest, _) = fixture();
        let result = run(import_from(
            Store::default(),
            manifest,
            Policy::default(),
            &mut FailingSource,
        ));
        assert!(matches!(result, Err(DriveError::Source(7))));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn arbitrary_fragmentation_commits_but_replay_never_advances(
            bytes in proptest::collection::vec(any::<u8>(), 1..=16_384),
            fragment_sizes in proptest::collection::vec(1usize..=4_096, 1..=32),
        ) {
            let mut builder = ManifestBuilder::new([0; 32], Policy::default());
            builder
                .push(
                    Entry::file(
                        CanonicalPath::new(b"payload").unwrap(),
                        false,
                        bytes.len() as u32,
                        *blake3::hash(&bytes).as_bytes(),
                        [7; 16],
                    )
                    .unwrap(),
                )
                .unwrap();
            let manifest = builder.finish().unwrap();
            let mut transaction = run(ImportTransaction::begin(
                Store::default(),
                manifest,
                Policy::default(),
            ))
            .unwrap();
            run(transaction.begin_next_file()).unwrap();

            let mut offset = 0usize;
            let mut fragment = 0usize;
            while offset < bytes.len() {
                let count = fragment_sizes[fragment % fragment_sizes.len()]
                    .min(bytes.len() - offset);
                run(transaction.write(offset as u32, &bytes[offset..offset + count])).unwrap();
                prop_assert_eq!(
                    run(transaction.write(offset as u32, &bytes[offset..offset + count])),
                    Err(ImportError::WrongOffset),
                );
                offset += count;
                fragment += 1;
            }
            run(transaction.finish_file()).unwrap();
            run(transaction.commit()).unwrap();
            let store = transaction.into_store().unwrap();
            prop_assert!(store.committed);
            prop_assert_eq!(store.staged["payload"].as_slice(), bytes.as_slice());
        }
    }
}
