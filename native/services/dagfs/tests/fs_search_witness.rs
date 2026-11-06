//! Init's `fs-search` boot witness on a volume someone has lived in.
//!
//! On metal the witness shares a persistent store with whatever its owner
//! has saved, while every CI volume is born empty — so a defect that only
//! ordinary content can reach surfaces first as a machine that will not
//! found its shell. These scenarios run the witness's exact sequence, and
//! `find reindex`'s walk and per-file step, through the service's real
//! dispatch over a founded process's data window, committing the producer's
//! own batch ([`encode_batch`]) — the bytes a `find`, `write` or editor save
//! hands this store.

mod common;

use common::name;
use kore_dagfs::{Refusal, index::encode_batch};
use kore_svc_dagfs::{
    ANY_REVISION, DataWindow, FreshnessPolicy, FsError, FsRequest, FsResponse, Handle, ObjectKind,
    ObjectSubtype, RamService, Rights, SearchContent, SearchPlane, SearchQuery, SearchTerm,
    SessionId, block_on, bounds::MAX_TRANSFER, read_dir_entry, read_search_hit,
    read_version_identity,
};
use kore_svc_extract::{
    Confidence, Fact, FactKind, FactValue, WORKER_REPORT_BYTES, encode_result, registry::PLAIN_TEXT,
};
use kore_svc_extract_worker_rust::parse_plain_text;

/// exec's `FS_DATA_BYTES`: the data window every founded process face gets.
const WINDOW_BYTES: usize = 0x1_0000;

/// The witness's fixture directory and the one file in it
/// (`user/services/init`).
const WITNESS: &str = "kore-fs-reindex-witness";
const PROBE: &str = "probe";

/// Half a page of meeting notes: twenty short lines, more distinct words
/// than the producer keeps as tokens. Nothing about it is unusual — which
/// is the point.
const NOTES: &str = "\
Weekly sync notes for the storage and search working group.
Attendees reviewed durable batches, cold restarts and recovery witnesses.
Open question: does reindex finish before the first shell can answer?
Action items include profiling completion latency under sustained load.
Someone proposed caching directory listings between consecutive pages.
We agreed the index stays auxiliary; authoritative writes always win.
Measurements suggest boot spends most of its time extracting images.
Next steps: tighten budgets, document refusals, rehearse failure drills.
Reminder: archive older transcripts monthly and keep summaries handy.
Questions about quotas, grants, leases and revocation went unresolved.
The editor saves documents atomically through one held transaction.
Compositor work continues separately; display timing is unaffected.
Celebrate small wins: fewer flakes, clearer logs, calmer reviews.
Volunteers will pair on tracing the scheduler's placement decisions.
Budget owners should confirm hardware orders before the quarter closes.
Lab machines need firmware updates; schedule maintenance windows early.
Keyboard layouts, pointer acceleration and fonts deserve a usability pass.
Translation strings remain untouched until terminology settles further.
Security review covers capability attenuation, sealing and audit trails.
Thanks everyone; minutes circulate tomorrow morning for corrections.
";

/// One founded process's filesystem face, host-side: a session on the real
/// service and the data window exec would have delivered it.
struct Face {
    service: RamService,
    session: SessionId,
    window: Vec<u8>,
}

impl Face {
    fn new() -> Self {
        let mut face = Self {
            service: RamService::default(),
            session: SessionId::new(0),
            window: vec![0; WINDOW_BYTES],
        };
        block_on(face.service.engine().attach_session(face.session)).expect("attach session");
        let attached = face.call(FsRequest::AttachWindow {
            slot: 1,
            bytes: WINDOW_BYTES as u32,
        });
        assert!(
            matches!(attached, FsResponse::WindowAttached { .. }),
            "{attached:?}"
        );
        face
    }

    fn call(&mut self, request: FsRequest) -> FsResponse {
        let mut window = DataWindow::new(&mut self.window);
        self.service
            .dispatch(self.session, &request, Some(&mut window))
    }

    fn handle(response: FsResponse) -> Handle {
        match response {
            FsResponse::Opened { handle, .. } => handle,
            other => panic!("expected a view, got {other:?}"),
        }
    }

    fn root(&mut self) -> Handle {
        Self::handle(self.call(FsRequest::OpenRoot {
            rights: Rights::ALL,
        }))
    }

    fn create(&mut self, parent: Handle, leaf: &str, kind: ObjectKind) -> Handle {
        Self::handle(self.call(FsRequest::Create {
            parent,
            name: name(leaf),
            kind,
            subtype: ObjectSubtype::PLAIN,
        }))
    }

    fn close(&mut self, handle: Handle) {
        assert_eq!(self.call(FsRequest::Close { handle }), FsResponse::Done);
    }

    fn write(&mut self, file: Handle, bytes: &[u8]) {
        self.window[..bytes.len()].copy_from_slice(bytes);
        let written = self.call(FsRequest::WriteAt {
            handle: file,
            offset: 0,
            window_offset: 0,
            length: bytes.len() as u32,
            expected: ANY_REVISION,
        });
        assert!(
            matches!(written, FsResponse::Committed { .. }),
            "{written:?}"
        );
    }

    /// `kore_dagfs::index::publish`: lend the version visible now, take the
    /// bytes just written or read it back when it fits one transfer, commit
    /// the producer's batch, and release the lease on every path.
    fn publish(&mut self, file: Handle, supplied: Option<&[u8]>) -> Result<(), FsError> {
        let lease = match self.call(FsRequest::LendVersion {
            handle: file,
            window_offset: 0,
        }) {
            FsResponse::VersionLent { lease } => lease,
            FsResponse::Refused(error) => return Err(error),
            other => panic!("lend answered {other:?}"),
        };
        let identity = read_version_identity(&self.window, 0).expect("identity");
        let mut source = [0u8; MAX_TRANSFER];
        let staged = match supplied {
            Some(bytes) => Ok(Some(bytes)),
            None if identity.len <= MAX_TRANSFER as u64 => {
                let len = identity.len as usize;
                match self.call(FsRequest::ReadVersion {
                    lease,
                    offset: 0,
                    window_offset: 0,
                    length: len as u32,
                }) {
                    FsResponse::Transferred { count } if count as usize == len => {
                        source[..len].copy_from_slice(&self.window[..len]);
                        Ok(Some(&source[..len]))
                    }
                    FsResponse::Refused(error) => Err(error),
                    other => panic!("read answered {other:?}"),
                }
            }
            None => Ok(None),
        };
        let committed = staged
            .and_then(|bytes| encode_batch(identity, bytes, &mut self.window))
            .and_then(|length| {
                match self.call(FsRequest::CommitDerived {
                    lease,
                    window_offset: 0,
                    length: length as u32,
                    output_window_bytes: 0,
                }) {
                    FsResponse::DerivedCommitted { .. } => Ok(()),
                    FsResponse::Refused(error) => Err(error),
                    other => panic!("commit answered {other:?}"),
                }
            });
        assert_eq!(
            self.call(FsRequest::ReleaseVersion { lease }),
            FsResponse::Done
        );
        committed
    }

    /// `find reindex [in <dir>]`: walk from `from` — the view root, or the
    /// directory `in` names — and publish every file without a current
    /// plain-text batch. Unsupported content stays unindexed, a full index
    /// ends the walk as an incomplete one, and any other refusal is the
    /// command's — exactly what `find` prints and exits with. `from` stays
    /// the caller's; the walk closes what it opens.
    fn reindex(&mut self, from: Handle) -> Result<u64, FsError> {
        let mut pending = vec![(from, false)];
        let mut indexed = 0;
        while let Some((directory, opened)) = pending.pop() {
            let mut cursor = 0;
            loop {
                let FsResponse::Listed { entries, next } = self.call(FsRequest::List {
                    handle: directory,
                    cursor,
                    window_offset: 0,
                }) else {
                    panic!("the directory did not list");
                };
                let page: Vec<_> = (0..entries as usize)
                    .map(|index| read_dir_entry(&self.window, 0, index).expect("entry"))
                    .collect();
                for entry in page {
                    let child = Self::handle(self.call(FsRequest::OpenAt {
                        parent: directory,
                        name: entry.name,
                        rights: Rights::READ,
                    }));
                    if entry.kind == ObjectKind::Directory {
                        pending.push((child, true));
                        continue;
                    }
                    let current = self.call(FsRequest::DerivedInfo {
                        handle: child,
                        parser: PLAIN_TEXT,
                    });
                    let published = if current == FsResponse::Refused(FsError::NotFound) {
                        self.publish(child, None)
                    } else {
                        Err(FsError::AlreadyExists)
                    };
                    self.close(child);
                    match published {
                        Ok(()) => indexed += 1,
                        Err(FsError::AlreadyExists | FsError::Unsupported) => {}
                        Err(error) if Refusal::Fs(error).is_capacity_exhausted() => {
                            return Err(FsError::IndexIncomplete);
                        }
                        Err(error) => return Err(error),
                    }
                }
                match next {
                    Some(next) => cursor = next,
                    None => break,
                }
            }
            if opened {
                self.close(directory);
            }
        }
        Ok(indexed)
    }

    /// One `CurrentOnly` full-text page beneath `anchor`, as the paths `find`
    /// prints, or the refusal it prints instead.
    fn find(
        &mut self,
        anchor: Handle,
        named: Option<&str>,
        term: &str,
    ) -> Result<Vec<String>, FsError> {
        let query = SearchQuery {
            name_contains: named.map(name),
            content: Some(SearchContent {
                plane: SearchPlane::FullText,
                term: SearchTerm::new(term).unwrap(),
            }),
            freshness: FreshnessPolicy::CurrentOnly,
            ..SearchQuery::ALL
        };
        match self.call(FsRequest::Search {
            handle: anchor,
            query,
            cursor: None,
            window_offset: 0,
        }) {
            FsResponse::Searched {
                hits, next: None, ..
            } => Ok((0..hits as usize)
                .map(|index| {
                    let hit = read_search_hit(&self.window, 0, index).expect("hit");
                    hit.path
                        .iter()
                        .map(|component| component.as_str())
                        .collect::<Vec<_>>()
                        .join("/")
                })
                .collect()),
            FsResponse::Refused(error) => Err(error),
            other => panic!("search answered {other:?}"),
        }
    }

    /// Init's witness, after its first three launches: `make-directory`
    /// makes the fixture, `create` publishes the empty probe's batch, and
    /// `write` lands one unsupported byte whose batch is refused as
    /// unsupported (`fs: wrote 1 bytes (index incomplete)`). Yields the
    /// fixture: the scope of the witness's `find reindex in`.
    fn witness_fixture(&mut self, root: Handle) -> Handle {
        let fixture = self.create(root, WITNESS, ObjectKind::Directory);
        let probe = self.create(fixture, PROBE, ObjectKind::File);
        assert_eq!(self.publish(probe, Some(&[])), Ok(()));
        self.write(probe, b"\xff");
        assert_eq!(
            self.publish(probe, Some(b"\xff")),
            Err(FsError::Unsupported)
        );
        self.close(probe);
        fixture
    }
}

/// The mechanism behind Biggertron's first-shell failure (rung 590191663:
/// `find: out of range` where the witness expects `find: search index
/// incomplete`). Any text file without a plain-text batch — half a page of
/// notes suffices — was published by the witness's `find reindex` like every
/// other. The producer encoded that batch into the whole data window — past
/// the one 4 KiB #476 batch the store keeps per version — and `CommitDerived`
/// refused it `Range`, a refusal the reindex walk rightly does not swallow.
/// The page must instead index within the store's bound, and stay findable.
#[test]
fn a_page_of_notes_on_the_volume_indexes_and_stays_findable() {
    let mut face = Face::new();
    let root = face.root();
    let documents = face.create(root, "Documents", ObjectKind::Directory);
    let notes = face.create(documents, "notes", ObjectKind::File);
    face.write(notes, NOTES.as_bytes());
    face.close(notes);

    assert_eq!(
        face.reindex(root),
        Ok(1),
        "the page is indexed, never `Range`"
    );
    assert_eq!(
        face.find(documents, None, "answer"),
        Ok(vec!["notes".to_string()]),
        "full-text search reads the page's whole text"
    );
    assert_eq!(
        face.find(documents, None, "corrections"),
        Ok(vec!["notes".to_string()]),
        "the last line's words are as findable as the first's"
    );
}

/// The witness's verdict is a fact about its own fixture. `find reindex in`
/// walks the fixture directory alone — it lends and reads the probe and
/// nothing else the volume holds — and `CurrentOnly` judges only the probe.
/// The witness it replaced reindexed the whole view at boot, so any one file
/// there the store could not lend, read or commit stopped the boot.
#[test]
fn the_witness_reindexes_and_judges_only_its_fixture() {
    let mut face = Face::new();
    let root = face.root();
    let documents = face.create(root, "Documents", ObjectKind::Directory);
    let page = face.create(documents, "page", ObjectKind::File);
    face.write(page, b"the answer is forty two");
    assert_eq!(face.publish(page, Some(b"the answer is forty two")), Ok(()));
    face.close(page);
    // Written before this volume had an index, so never published.
    let legacy = face.create(documents, "legacy", ObjectKind::File);
    face.write(legacy, NOTES.as_bytes());

    let fixture = face.witness_fixture(root);
    assert_eq!(face.reindex(fixture), Ok(0), "the probe stays unsupported");
    assert_eq!(
        face.find(fixture, None, "answer"),
        Err(FsError::IndexIncomplete)
    );
    assert_eq!(
        face.call(FsRequest::DerivedInfo {
            handle: legacy,
            parser: PLAIN_TEXT,
        }),
        FsResponse::Refused(FsError::NotFound),
        "the walk never left the fixture"
    );
    // What the witness used to do instead: the whole view, `legacy` with it.
    assert_eq!(face.reindex(root), Ok(1));
}

/// The store's side of the same bound. A batch past the one the store keeps
/// per version — what any producer that encodes into its whole window sends
/// — is answered as a version this index cannot hold (`Unsupported`), which
/// every producer reports as `(index incomplete)` and `find reindex` skips.
/// It was `Range`, which failed the write the batch describes after that
/// write had committed.
#[test]
fn an_oversized_batch_is_answered_as_an_unindexable_version() {
    let mut face = Face::new();
    let root = face.root();
    let notes = face.create(root, "notes", ObjectKind::File);
    face.write(notes, NOTES.as_bytes());
    let FsResponse::VersionLent { lease } = face.call(FsRequest::LendVersion {
        handle: notes,
        window_offset: 0,
    }) else {
        panic!("the version was not lent");
    };
    let identity = read_version_identity(&face.window, 0).expect("identity");
    // The parser's spans plus a token fact per distinct word: the batch the
    // plain-text producer published before it bounded it.
    let mut result = parse_plain_text(identity, NOTES.as_bytes()).unwrap();
    for word in NOTES.split(|byte: char| !byte.is_ascii_alphanumeric()) {
        if word.is_empty() || result.facts.iter().any(|fact| fact.key.as_str() == word) {
            continue;
        }
        let fact = Fact {
            kind: FactKind::Custom,
            key: word.try_into().unwrap(),
            value: FactValue::Utf8(word.try_into().unwrap()),
            confidence: Confidence::CERTAIN,
            source: None,
        };
        if result.facts.push(fact).is_err() {
            break;
        }
    }
    let length = encode_result(&result, &mut face.window).unwrap();
    assert!(length > WORKER_REPORT_BYTES);

    assert_eq!(
        face.call(FsRequest::CommitDerived {
            lease,
            window_offset: 0,
            length: length as u32,
            output_window_bytes: 0,
        }),
        FsResponse::Refused(FsError::Unsupported)
    );
    assert_eq!(
        face.call(FsRequest::ReleaseVersion { lease }),
        FsResponse::Done
    );
    assert_eq!(
        face.call(FsRequest::DerivedInfo {
            handle: notes,
            parser: PLAIN_TEXT,
        }),
        FsResponse::Refused(FsError::NotFound),
        "the version stays unindexed"
    );
}
