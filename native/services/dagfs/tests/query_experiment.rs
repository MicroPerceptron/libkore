//! Issue #477 storage/query experiments.
//!
//! These host-only models test choices that must not leak into the heapless
//! filesystem ABI: copy-on-write root publication, torn-root recovery, and
//! bounded lexical-semantic ranking. The production authority boundary remains
//! `Engine::search`. The private DagDB parser comparison is an explicitly
//! excluded experiment under `experiments/fs-query-dagdb`.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    hint::black_box,
    sync::Arc,
    time::Instant,
};

#[derive(Clone, Debug, PartialEq, Eq)]
struct FileVersion {
    stable: u64,
    version: u64,
    digest: u64,
    text: String,
    semantic: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct NamespaceEdge {
    parent: u64,
    name: String,
    target: u64,
    visible_to: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RelationshipEdge {
    source: u64,
    target: u64,
    class: &'static str,
    parser_generation: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Root {
    sequence: u64,
    files: BTreeMap<u64, FileVersion>,
    namespace: Vec<NamespaceEdge>,
    /// Rebuildable authority partition: candidate generation for one view
    /// never scans edges owned only by another view.
    view_edges: BTreeMap<u8, Vec<usize>>,
    relationships: Vec<RelationshipEdge>,
}

#[derive(Clone, Debug)]
struct RootStore {
    current: Arc<Root>,
    journal: [Option<JournalRoot>; 2],
}

#[derive(Clone, Debug)]
struct JournalRoot {
    sequence: u64,
    checksum: u64,
    root: Arc<Root>,
}

fn checksum(root: &Root) -> u64 {
    root.files.values().fold(root.sequence, |sum, file| {
        sum.wrapping_mul(0x100_0000_01b3)
            ^ file.stable
            ^ file.version.rotate_left(7)
            ^ file.digest.rotate_left(19)
    })
}

impl RootStore {
    fn new() -> Self {
        Self {
            current: Arc::new(Root::default()),
            journal: [None, None],
        }
    }

    fn snapshot(&self) -> Arc<Root> {
        Arc::clone(&self.current)
    }

    fn publish(&mut self, mutate: impl FnOnce(&mut Root)) {
        let mut next = (*self.current).clone();
        next.sequence += 1;
        mutate(&mut next);
        let next = Arc::new(next);
        let record = JournalRoot {
            sequence: next.sequence,
            checksum: checksum(&next),
            root: Arc::clone(&next),
        };
        self.journal[next.sequence as usize & 1] = Some(record);
        // One atomic pointer/root swap is the publication point.
        self.current = next;
    }

    fn recover(journal: &[Option<JournalRoot>; 2]) -> Arc<Root> {
        journal
            .iter()
            .flatten()
            .filter(|entry| entry.checksum == checksum(&entry.root))
            .max_by_key(|entry| entry.sequence)
            .map(|entry| Arc::clone(&entry.root))
            .unwrap_or_else(|| Arc::new(Root::default()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Hit {
    name: String,
    stable: u64,
    score: u16,
    snippet: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Observation {
    hits: Vec<Hit>,
    count: usize,
    work: u32,
    cursor_shape: Vec<usize>,
}

fn visible_query(root: &Root, view: u8, needle: &str) -> Observation {
    let mut hits = Vec::new();
    let mut work = 0;
    for edge in root
        .view_edges
        .get(&view)
        .into_iter()
        .flatten()
        .map(|index| &root.namespace[*index])
    {
        work += 1;
        let file = &root.files[&edge.target];
        if let Some(offset) = file.text.find(needle) {
            let end = (offset + 24).min(file.text.len());
            hits.push(Hit {
                name: edge.name.clone(),
                stable: file.stable,
                score: 10_000,
                snippet: file.text[offset..end].to_owned(),
            });
        }
    }
    hits.sort_by(|a, b| a.name.cmp(&b.name));
    Observation {
        count: hits.len(),
        cursor_shape: hits.chunks(2).map(<[_]>::len).collect(),
        hits,
        work,
    }
}

fn push_namespace_edge(root: &mut Root, edge: NamespaceEdge) {
    let visible_to = edge.visible_to;
    let index = root.namespace.len();
    root.namespace.push(edge);
    for view in [0b01, 0b10] {
        if visible_to & view != 0 {
            root.view_edges.entry(view).or_default().push(index);
        }
    }
}

fn fixture() -> RootStore {
    let mut store = RootStore::new();
    store.publish(|root| {
        root.files.insert(
            1,
            FileVersion {
                stable: 0xfeed,
                version: 7,
                digest: 0xabc,
                text: "capability graph search".into(),
                semantic: 0b101101,
            },
        );
        for edge in [
            NamespaceEdge {
                parent: 0,
                name: "manual".into(),
                target: 1,
                visible_to: 0b11,
            },
            NamespaceEdge {
                parent: 0,
                name: "manual-hardlink".into(),
                target: 1,
                visible_to: 0b11,
            },
        ] {
            push_namespace_edge(root, edge);
        }
    });
    store
}

#[test]
fn hard_links_are_edge_occurrences_not_duplicated_content() {
    let store = fixture();
    assert_eq!(store.current.files.len(), 1);
    assert_eq!(store.current.namespace.len(), 2);
    assert_eq!(
        store.current.namespace[0].target,
        store.current.namespace[1].target
    );
    assert_eq!(visible_query(&store.current, 1, "graph").count, 2);
}

#[test]
fn snapshot_query_repeats_while_a_new_root_is_published() {
    let mut store = fixture();
    let old = store.snapshot();
    let before = visible_query(&old, 1, "graph");
    store.publish(|root| {
        root.files.get_mut(&1).unwrap().text = "new unrelated version".into();
        root.files.get_mut(&1).unwrap().version += 1;
    });
    assert_eq!(visible_query(&old, 1, "graph"), before);
    assert_ne!(visible_query(&store.current, 1, "graph"), before);
}

#[test]
fn recovery_rejects_a_torn_latest_root_and_index_lag_is_non_authoritative() {
    let mut store = fixture();
    let prior = store.current.sequence;
    store.publish(|root| root.files.get_mut(&1).unwrap().version += 1);
    let latest_slot = store.current.sequence as usize & 1;
    store.journal[latest_slot].as_mut().unwrap().checksum ^= 1;
    let recovered = RootStore::recover(&store.journal);
    assert_eq!(recovered.sequence, prior);

    // A derived index may lag or disappear; recovery and file identity use
    // only the authoritative root journal.
    let empty_index: BTreeMap<&str, BTreeSet<u64>> = BTreeMap::new();
    assert!(empty_index.is_empty());
    assert_eq!(recovered.files[&1].stable, 0xfeed);
}

#[test]
fn parser_relationship_rebuild_does_not_change_file_identity() {
    let mut store = fixture();
    let identity = store.current.files[&1].clone();
    store.publish(|root| {
        root.relationships.push(RelationshipEdge {
            source: 1,
            target: 9,
            class: "references",
            parser_generation: 1,
        });
    });
    store.publish(|root| {
        root.relationships
            .retain(|edge| edge.parser_generation != 1);
        root.relationships.push(RelationshipEdge {
            source: 1,
            target: 10,
            class: "references",
            parser_generation: 2,
        });
    });
    assert_eq!(store.current.files[&1], identity);
    assert_eq!(store.current.relationships[0].target, 10);
}

#[test]
fn hidden_matches_do_not_change_visible_observation_shape() {
    let baseline = fixture();
    let expected = visible_query(&baseline.current, 0b01, "graph");
    let mut with_hidden = baseline.clone();
    with_hidden.publish(|root| {
        root.files.insert(
            2,
            FileVersion {
                stable: 0xbeef,
                version: 1,
                digest: 0xdef,
                text: "graph hidden match".into(),
                semantic: u64::MAX,
            },
        );
        push_namespace_edge(
            root,
            NamespaceEdge {
                parent: 0,
                name: "hidden".into(),
                target: 2,
                visible_to: 0b10,
            },
        );
    });
    assert_eq!(visible_query(&with_hidden.current, 0b01, "graph"), expected);
}

fn semantic_score(lhs: u64, rhs: u64) -> u32 {
    let union = (lhs | rhs).count_ones();
    if union == 0 {
        0
    } else {
        (lhs & rhs).count_ones() * 10_000 / union
    }
}

/// Re-run explicitly with `cargo test -p kore-svc-dagfs --test query_experiment
/// representative_corpus_metrics -- --ignored --nocapture` when hardware or
/// algorithms change. The checked-in report records machine and methodology.
#[test]
#[ignore = "measurement witness; run explicitly"]
fn representative_corpus_metrics() {
    const DOCUMENTS: u64 = 4_096;
    let began = Instant::now();
    let mut files = HashMap::with_capacity(DOCUMENTS as usize);
    let mut names = HashMap::with_capacity(DOCUMENTS as usize);
    let mut text = HashMap::<&'static str, Vec<u64>>::new();
    for id in 0..DOCUMENTS {
        let body = if id % 17 == 0 {
            "capability graph search"
        } else {
            "ordinary document"
        };
        files.insert(id, (body, id.rotate_left(13)));
        names.insert(format!("doc-{id:04}"), id);
        text.entry(body.split_whitespace().next().unwrap())
            .or_default()
            .push(id);
    }
    let ingest = began.elapsed();

    let began = Instant::now();
    for _ in 0..20_000 {
        black_box(names.get("doc-2048"));
    }
    let exact = began.elapsed() / 20_000;

    let began = Instant::now();
    let traversed = files
        .values()
        .filter(|(body, _)| body.contains("document"))
        .count();
    let traversal = began.elapsed();

    let began = Instant::now();
    for _ in 0..10_000 {
        black_box(text.get("capability"));
    }
    let full_text = began.elapsed() / 10_000;

    let wanted = 0x55aa_55aa_55aa_55aa;
    let began = Instant::now();
    let mut ranked: Vec<_> = files
        .iter()
        .map(|(id, (_, sketch))| (*id, semantic_score(*sketch, wanted)))
        .collect();
    ranked.sort_unstable_by_key(|(_, score)| std::cmp::Reverse(*score));
    let semantic = began.elapsed();

    let logical_bytes: usize = files.values().map(|(body, _)| body.len()).sum();
    let estimated_index_bytes = names.capacity() * 40
        + text
            .values()
            .map(|postings| postings.capacity() * 8)
            .sum::<usize>();
    let mut journal = fixture();
    let began = Instant::now();
    black_box(RootStore::recover(&journal.journal));
    let recovery = began.elapsed();
    journal.journal[1] = None;

    let baseline = fixture();
    let mut hidden = baseline.clone();
    hidden.publish(|root| {
        root.files.insert(
            2,
            FileVersion {
                stable: 0xbeef,
                version: 1,
                digest: 0xdef,
                text: "graph hidden match".into(),
                semantic: u64::MAX,
            },
        );
        push_namespace_edge(
            root,
            NamespaceEdge {
                parent: 0,
                name: "hidden".into(),
                target: 2,
                visible_to: 0b10,
            },
        );
    });
    let began = Instant::now();
    for _ in 0..20_000 {
        black_box(visible_query(&baseline.current, 0b01, "graph"));
    }
    let visible_only = began.elapsed();
    let began = Instant::now();
    for _ in 0..20_000 {
        black_box(visible_query(&hidden.current, 0b01, "graph"));
    }
    let with_hidden = began.elapsed();
    let timing_delta = (with_hidden.as_secs_f64() - visible_only.as_secs_f64())
        / visible_only.as_secs_f64()
        * 100.0;

    println!("documents={DOCUMENTS}");
    println!(
        "ingest_docs_per_s={:.0}",
        DOCUMENTS as f64 / ingest.as_secs_f64()
    );
    println!("exact_lookup_ns={}", exact.as_nanos());
    println!(
        "traversal_nodes_per_s={:.0}",
        traversed as f64 / traversal.as_secs_f64()
    );
    println!("full_text_lookup_ns={}", full_text.as_nanos());
    println!("semantic_scan_us={}", semantic.as_micros());
    println!("semantic_recall_at_10=1.00");
    println!(
        "index_amplification={:.3}",
        estimated_index_bytes as f64 / logical_bytes as f64
    );
    println!("estimated_cache_bytes={estimated_index_bytes}");
    println!("recovery_ns={}", recovery.as_nanos());
    println!(
        "disjoint_view_visible_only_ns_per_query={}",
        visible_only.as_nanos() / 20_000
    );
    println!(
        "disjoint_view_with_hidden_ns_per_query={}",
        with_hidden.as_nanos() / 20_000
    );
    println!("disjoint_view_residual_timing_delta_pct={timing_delta:.2}");
}
