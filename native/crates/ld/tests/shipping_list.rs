//! Every service crate must say how it reaches metal — or say that it doesn't.
//!
//! A `services/*` crate is a workspace member, so `make check-*` compiles it
//! and `make test` runs its unit tests. Neither asks the question that
//! actually matters: **does anything found it?** A service that is superseded,
//! or built ahead of its wiring, therefore keeps every gate green while being
//! unreachable on the metal target — it looks alive because it compiles.
//!
//! The declaration already exists: a founded service carries
//! `services/<name>/boot.toml`, and `res/boot.toml` names the image the boot
//! chain measures and starts. What was missing is anything requiring a crate
//! to be in that set or to admit it is not. This test is that requirement,
//! so an unshipped service is a written-down fact with a reason rather than
//! a silence nobody can see.
//!
//! Adding a service crate therefore means one of two things:
//!
//! - it ships — give it `services/<name>/boot.toml` and list its image in
//!   `res/boot.toml`; or
//! - it does not ship yet — add it to [`UNSHIPPED`] with the reason, which is
//!   a line in a review rather than a discovery six months later.
//!
//! It lives beside `built_images`, which validates the shipped images
//! themselves, because this validates the *list* of them — and because the
//! kernel package is `#![no_main]`, so an integration test cannot live at the
//! repo root where the tree it reads begins.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Service crates that deliberately do not appear in the boot set, each with
/// the reason it is out.
///
/// This is a *declaration*, not an exemption: an entry here is a claim that
/// the crate's absence from boot is intended, and the reason is what a
/// reviewer checks. Entries should leave — by being wired up or by being
/// deleted — rather than accumulate.
const UNSHIPPED: &[(&str, &str)] = &[
    (
        "exec-client",
        "library, not a service: the exec protocol and caller face, split out of \
         kore-svc-exec so a client can link them without the spawn engine that \
         answers them (and without the ~20 sibling services its founder manifest \
         cites). Nothing founds it; kore-svc-exec and every exec caller link it",
    ),
    (
        "display",
        "backend crate: owns the physical display claim and aperture; linked \
         into user/services/compositor, whose founded image is kore-svc-compositor",
    ),
    // Reached on demand instead of at boot: the broker and its parsers are
    // packed into the image archive and founded per job by the worker
    // supervisor, so they are shipped without being in the boot set.
    ("extract", "archive-resident; founded per job by the broker"),
    (
        "extract-worker-c",
        "archive-resident; founded per job by the broker",
    ),
    (
        "extract-worker-rust",
        "archive-resident; founded per job by the broker",
    ),
    // Genuinely unreachable today. Recorded here so the state is visible in
    // review instead of being inferred from a missing file.
    (
        "output",
        "superseded: kore-svc-compositor serves OutReqWire/OutChunk; \
         nothing founds kore-svc-output",
    ),
    (
        "accel-ivpu",
        "backend crate: the Lunar Lake NPU `Accelerator` impl linked into \
         user/services/accel; the founded image is kore-svc-accel (services/accel), \
         not a boot row of its own",
    ),
    (
        "accel-spmd",
        "backend crate: the CPU-SIMD `Accelerator` impl over kore-rt's SPMD \
         region launch, the third tenant crates/abi/src/svc.rs declares. No boot \
         row yet — its compute step (`SpmdBackend::run_pending`) needs a serve \
         loop to drive it, and that lands with the row",
    ),
    (
        "swap-nvme",
        "not in USER_SERVICES and not founded; \
         res/x86_64/kore-svc-swap-nvme.elf is a stale committed artifact",
    ),
];

fn repo_root() -> PathBuf {
    // `crates/ld` → the tree. Resolved from the manifest dir rather than the
    // cwd so the test does not depend on where cargo was invoked.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/ld sits two levels below the repo root")
        .to_path_buf()
}

/// Every directory under `services/` that is a crate.
fn service_crates(root: &Path) -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    let dir = root.join("services");
    for entry in std::fs::read_dir(&dir).expect("services/ is readable") {
        let entry = entry.expect("a readable services/ entry");
        if !entry.file_type().expect("a file type").is_dir() {
            continue;
        }
        if !entry.path().join("Cargo.toml").is_file() {
            continue;
        }
        found.insert(entry.file_name().to_string_lossy().into_owned());
    }
    assert!(!found.is_empty(), "no service crates found under services/");
    found
}

/// Services carrying the per-service boot manifest — the founded set as the
/// crates themselves declare it.
fn declares_boot_manifest(root: &Path, name: &str) -> bool {
    root.join("services").join(name).join("boot.toml").is_file()
}

/// Whether `res/boot.toml` names this service's image, which is what the boot
/// chain measures and starts.
fn named_in_boot_manifest(boot_toml: &str, name: &str) -> bool {
    boot_toml.contains(&format!("\"kore-svc-{name}\""))
}

/// Service-shaped committed artifacts that intentionally are not founded.
/// Keep this narrower than [`UNSHIPPED`]: most backend/library crates have no
/// standalone ELF at all. An entry here is a temporary packaging exception
/// whose stale bytes would otherwise look like a bootable service.
const NON_FOUNDED_SERVICE_ARTIFACTS: &[(&str, &str)] = &[(
    "kore-svc-swap-nvme",
    "legacy claimant image retained while its standalone founder row is unwired",
)];

fn founded_image_names(boot_toml: &str) -> BTreeSet<String> {
    boot_toml
        .lines()
        .map(|line| line.trim().trim_end_matches(','))
        .filter_map(|line| line.strip_prefix('"')?.strip_suffix('"'))
        .filter(|name| name.starts_with("kore-svc-"))
        .map(str::to_owned)
        .collect()
}

/// Whether a founded image's fragment founds on `arch`: an `arch = [...]`
/// line lists the ISAs, and an undeclared one means every ISA (the manifest
/// vocabulary in res/boot.toml's header).  Read from the fragment beside the
/// crate, the same file `tools/boot_manifest.py --images-for` reads.
fn founds_on(root: &Path, image: &str, arch: &str) -> bool {
    let name = image.strip_prefix("kore-svc-").unwrap_or(image);
    let Ok(fragment) = std::fs::read_to_string(root.join("services").join(name).join("boot.toml"))
    else {
        return true;
    };
    let Some(list) = fragment
        .lines()
        .map(str::trim)
        .filter(|line| !line.starts_with('#'))
        .find_map(|line| line.strip_prefix("arch")?.trim_start().strip_prefix('='))
    else {
        return true;
    };
    list.split('"')
        .skip(1)
        .step_by(2)
        .any(|target| target == arch)
}

fn service_artifacts(root: &Path, arch: &str) -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    let dir = root.join("res").join(arch);
    for entry in std::fs::read_dir(&dir)
        .unwrap_or_else(|error| panic!("{} is unreadable: {error}", dir.display()))
    {
        let entry = entry.expect("a readable service artifact entry");
        if !entry.file_type().expect("an artifact file type").is_file() {
            continue;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let Some(stem) = name
            .strip_prefix("kore-svc-")
            .and_then(|stem| stem.strip_suffix(".elf"))
        else {
            continue;
        };
        found.insert(format!("kore-svc-{stem}"));
    }
    found
}

#[test]
fn every_service_crate_is_founded_or_declared_unshipped() {
    let root = repo_root();
    let unshipped: BTreeSet<&str> = UNSHIPPED.iter().map(|(n, _)| *n).collect();

    let mut unaccounted = Vec::new();
    for name in service_crates(&root) {
        let founded = declares_boot_manifest(&root, &name);
        if founded || unshipped.contains(name.as_str()) {
            continue;
        }
        unaccounted.push(name);
    }

    assert!(
        unaccounted.is_empty(),
        "these service crates are neither founded nor declared unshipped: {unaccounted:?}\n\
         Give each one `services/<name>/boot.toml` plus an image in res/boot.toml, \
         or add it to UNSHIPPED in this test with the reason it stays out."
    );
}

/// The two halves of "founded" must agree: a per-service manifest without an
/// image in `res/boot.toml` builds a service the boot chain never starts.
#[test]
fn a_services_boot_manifest_matches_the_boot_chains_image_list() {
    let root = repo_root();
    let boot_toml = std::fs::read_to_string(root.join("res/boot.toml")).expect("res/boot.toml");

    let mut disagreed = Vec::new();
    for name in service_crates(&root) {
        let declares = declares_boot_manifest(&root, &name);
        let named = named_in_boot_manifest(&boot_toml, &name);
        if declares != named {
            disagreed.push((name, declares, named));
        }
    }

    assert!(
        disagreed.is_empty(),
        "service boot.toml and res/boot.toml disagree (name, has services/<n>/boot.toml, \
         named in res/boot.toml): {disagreed:?}"
    );
}

/// Each architecture's committed service-shaped ELF set must match the
/// founded image list **for that architecture**: a row whose fragment
/// declares `arch = [...]` founds only there, so its artifact is expected on
/// those ISAs and is a stray everywhere else.  This catches renamed services
/// whose old binary remains in `res/<arch>`, new rows whose measured artifact
/// was never added, and a silicon-bound row's image left behind on an ISA
/// that never founds it.
#[test]
fn committed_service_artifacts_match_the_founded_set() {
    let root = repo_root();
    let boot_toml = std::fs::read_to_string(root.join("res/boot.toml")).expect("res/boot.toml");
    let every_arch = founded_image_names(&boot_toml);
    let allowed: BTreeSet<&str> = NON_FOUNDED_SERVICE_ARTIFACTS
        .iter()
        .map(|(name, _)| *name)
        .collect();

    for arch in ["x86_64", "aarch64"] {
        let founded: BTreeSet<String> = every_arch
            .iter()
            .filter(|image| founds_on(&root, image, arch))
            .cloned()
            .collect();
        assert!(
            founded.len() + 2 >= every_arch.len(),
            "res/{arch}: the arch filter dropped more rows than the manifest gates — \
             the fragment parse is wrong, not the manifest"
        );
        let actual = service_artifacts(&root, arch);
        let missing: Vec<&String> = founded.difference(&actual).collect();
        let stray: Vec<&String> = actual
            .difference(&founded)
            .filter(|name| !allowed.contains(name.as_str()))
            .collect();
        assert!(
            missing.is_empty() && stray.is_empty(),
            "res/{arch} service artifacts disagree with the founded set: \
             missing={missing:?}, stray={stray:?}. Remove renamed/stale ELFs or \
             add the founded row and its measured artifact."
        );
    }

    for (name, reason) in NON_FOUNDED_SERVICE_ARTIFACTS {
        assert!(
            !reason.trim().is_empty(),
            "artifact exception `{name}` has no reason"
        );
        assert!(
            !every_arch.contains(*name),
            "artifact exception `{name}` is founded now; remove the exception"
        );
    }
}

/// The arch column parses the way the generator reads it: the two Intel
/// accelerator rows found on x86_64 only, every other row everywhere.
#[test]
fn the_arch_column_gates_exactly_the_silicon_bound_rows() {
    let root = repo_root();
    assert!(founds_on(&root, "kore-svc-accel", "x86_64"));
    assert!(!founds_on(&root, "kore-svc-accel", "aarch64"));
    assert!(!founds_on(&root, "kore-svc-accel-xe", "aarch64"));
    assert!(founds_on(&root, "kore-svc-mm", "aarch64"));
    assert!(founds_on(&root, "kore-svc-device", "riscv64"));
}

/// Modules the bootloader carries that are **not** service rows, so the
/// cross-check below expects them beyond `res/boot.toml`'s `services` list.
///
/// The linker stub is deliberately absent: `iso-boot-x86` appends its
/// `module_string` at ISO time rather than declaring it in the checked-in
/// config, so a copy here would be loaded twice.
const NON_ROW_MODULES: &[&str] = &["kore-image-archive"];

/// Every founded row's image must be a module the **bootloader actually
/// carries**, and every service module it carries must be a founded row.
///
/// This is the third question in "how does this reach metal?", and the one
/// nothing was asking. `res/boot.toml` says which rows found; `USER_SERVICES` stages
/// their ELFs into the ISO; but Limine only loads what `boot/limine.conf`
/// names, and the founder resolves a row to its module **by name**
/// (`boot_modules().find(|m| m.cmdline() == name)`). A row present in the
/// manifest and absent from the config compiles, tests, measures and stages
/// perfectly — and then the machine boots DEGRADED with
/// `<name> module not shipped by the bootloader`, which is a QEMU round to
/// discover. The frame row cost exactly that.
#[test]
fn every_founded_row_is_carried_by_the_bootloader() {
    let root = repo_root();
    let boot_toml = std::fs::read_to_string(root.join("res/boot.toml")).expect("res/boot.toml");
    let limine = std::fs::read_to_string(root.join("boot/limine.conf")).expect("boot/limine.conf");

    // What the bootloader is told to load.
    let carried: BTreeSet<&str> = limine
        .lines()
        .filter_map(|line| line.trim().strip_prefix("module_string:"))
        .map(str::trim)
        .collect();
    assert!(
        !carried.is_empty(),
        "boot/limine.conf declares no modules at all — the parse is wrong, not the config"
    );

    // What the boot chain founds. Read from the `services` list rather than
    // the crate tree: it is the founder's own ordered source of truth, and it
    // is what the module names must match.
    let founded: BTreeSet<&str> = boot_toml
        .lines()
        .map(|line| line.trim().trim_end_matches(','))
        .filter_map(|line| line.strip_prefix('"')?.strip_suffix('"'))
        .filter(|name| name.starts_with("kore-svc-"))
        .collect();
    assert!(
        founded.len() > 5,
        "res/boot.toml yielded only {} founded rows — the parse is wrong, not the manifest",
        founded.len()
    );

    let missing: Vec<&&str> = founded.iter().filter(|n| !carried.contains(*n)).collect();
    assert!(
        missing.is_empty(),
        "these rows are founded by res/boot.toml but no bootloader module carries them: \
         {missing:?}\nAdd a `module_path` / `module_string` pair to boot/limine.conf — \
         without it the row founds FAILED and the boot goes DEGRADED."
    );

    let stray: Vec<&&str> = carried
        .iter()
        .filter(|n| n.starts_with("kore-svc-"))
        .filter(|n| !founded.contains(*n))
        .collect();
    assert!(
        stray.is_empty(),
        "boot/limine.conf carries service modules that res/boot.toml does not found: \
         {stray:?}\nDrop them, or add the row to the `services` list."
    );

    for extra in carried.iter().filter(|n| !n.starts_with("kore-svc-")) {
        assert!(
            NON_ROW_MODULES.contains(extra),
            "boot/limine.conf carries `{extra}`, which is neither a founded row nor a \
             declared non-row module — add it to NON_ROW_MODULES with its reason"
        );
    }
}

/// An `UNSHIPPED` entry that no longer names a service crate is stale — the
/// crate was wired up or deleted and the declaration outlived it.
#[test]
fn no_unshipped_entry_outlives_its_crate() {
    let root = repo_root();
    let crates = service_crates(&root);

    for (name, reason) in UNSHIPPED {
        assert!(
            crates.contains(*name),
            "UNSHIPPED names `{name}` ({reason}) but services/{name} does not exist — \
             drop the entry"
        );
        assert!(
            !declares_boot_manifest(&root, name),
            "`{name}` is declared UNSHIPPED but carries services/{name}/boot.toml — \
             it is founded now, so drop the entry"
        );
        assert!(
            !reason.trim().is_empty(),
            "UNSHIPPED entry `{name}` has no reason"
        );
    }
}
