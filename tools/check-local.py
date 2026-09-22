#!/usr/bin/env python3
"""Prove path/Git owner mixing is detected and a complete local patch composes."""
import argparse
import importlib.util
import json
import os
import shutil
from pathlib import Path
import subprocess
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def distribution(snapshot: Path):
    # Snapshot the current source into a local Git distribution, including
    # uncommitted alpha edits. Consumers of a Git package can override its
    # native owners; a path dependency itself is not a patchable Git source.
    snapshot.mkdir(parents=True)
    files = subprocess.check_output(
        ["git", "ls-files", "-c", "-o", "--exclude-standard", "-z", "--",
         "Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "README.md", "LICENSE",
         ".cargo", "src", "docs", "tests/consumer", "native", "crates/ffi", "crates/entry", "crates/wasi", "extern"], cwd=ROOT,
    ).decode().split("\0")
    for name in sorted(set(files)):
        if not name or not (ROOT / name).is_file():
            continue
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    subprocess.run(["git", "init", "-q", str(snapshot)], check=True)
    subprocess.run(["git", "add", "."], cwd=snapshot, check=True)
    subprocess.run(["git", "-c", "user.name=libkore test", "-c", "user.email=test@localhost",
                    "-c", "commit.gpgsign=false", "commit", "-qm", "consumer fixture"],
                   cwd=snapshot, check=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=snapshot).decode().strip()
    source = snapshot.as_uri()
    return source, revision


def check(tree: Path):
    tree = tree.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="libkore-consumer-") as temporary:
        work = Path(temporary)
        (work / "src").mkdir()
        source, revision = distribution(work / "distribution")
        dependencies = {
            "sdk": {"package": "kore-std", "git": source, "rev": revision, "features": ["exec"]},
            "ffi": {"package": "kore-ffi", "git": source, "rev": revision},
            "entry": {"package": "kore-entry", "git": source, "rev": revision},
            "kore-abi": {"path": str(tree / "crates/abi")},
            "kore-rt": {"path": str(tree / "crates/rt"), "default-features": False},
            "kore-exec-client": {"path": str(tree / "services/exec-client")},
            "kore-dagfs": {"path": str(tree / "crates/dagfs"), "features": ["runtime"]},
        }
        manifest = '[package]\nname="libkore-local-consumer"\nversion="0.0.0"\nedition="2024"\n[workspace]\n'
        for name, row in dependencies.items():
            manifest += f"[dependencies.{name}]\n"
            manifest += "".join(f"{key} = {json.dumps(value)}\n" for key, value in row.items())
        (work / "Cargo.toml").write_text(manifest)
        (work / "Cargo.lock").write_bytes((ROOT / "Cargo.lock").read_bytes())
        (work / "src/lib.rs").write_text('''
pub fn exec_identity(cap: kore_exec_client::ExecutableCapability) -> sdk::exec::ExecutableCapability { cap }
pub async fn delegate_executable<T: sdk::runtime::transport::Transport>(
    cap: &sdk::exec::ExecutableCapability,
    tx: &kore_rt::chan::ChanTx<'_, T>,
    rights: kore_abi::cap::Rights,
) -> Result<(), sdk::runtime::chan::SendError> {
    cap.transfer_derived(tx, rights).await
}
pub fn crash_identity(record: kore_abi::crash::CrashRecord) -> sdk::crash::CrashRecord { record }
pub fn task_identity(id: kore_abi::task::TaskIdentity) -> sdk::abi::task::TaskIdentity { id }
pub fn abi_identity(slot: kore_abi::session::SlotRef) -> sdk::abi::session::SlotRef { slot }
pub fn runtime_identity(rt: kore_rt::entry::Rt) -> sdk::runtime::entry::Rt { rt }
pub fn entry_runtime_identity(rt: kore_rt::entry::Rt) -> entry::runtime::entry::Rt { rt }
pub fn entry_exit_identity(code: kore_rt::ExitCode) -> entry::startup::ExitCode { code }
pub fn fs_identity<'w, 't>(fs: kore_dagfs::Fs<'w, 't>) -> sdk::fs::Fs<'w, 't> { fs }
pub fn view_identity(view: kore_dagfs::View) -> sdk::fs::View { view }
pub fn foreign_task_tokens(batch: &mut ffi::KoreTaskBatch, slot: usize) -> Option<kore_rt::executor::SpawnToken> { batch.take(slot) }
pub fn foreign_owner<'w, 't>(fs: kore_dagfs::Fs<'w, 't>) -> Result<ffi::Session<'w, 't>, kore_dagfs::Fs<'w, 't>> { ffi::Session::new(fs) }
''')
        # Keep the distribution lock untouched; this consumer has its own lock.
        toolchain = tomllib.loads((ROOT / "rust-toolchain.toml").read_text())["toolchain"]["channel"]
        env = dict(os.environ, RUSTUP_TOOLCHAIN=toolchain, CARGO_NET_OFFLINE="false")
        env.setdefault("CARGO_TARGET_DIR", str(ROOT / "target/local-consumer"))

        def metadata(*config):
            return json.loads(subprocess.check_output(
                ["cargo", *config, "metadata", "--format-version", "1"],
                cwd=work, env=env,
            ))

        mixed = module("check-owners").violations(metadata())
        if not any("kore-abi: multiple owners" in error for error in mixed):
            raise RuntimeError("negative control did not detect mixed native owners")
        config = work / "local.toml"
        config.write_text(module("local-kore").patches(tree, source))
        args = ["--config", str(config)]
        errors = module("check-owners").violations(metadata(*args))
        if errors:
            raise RuntimeError("\n".join(errors))
        subprocess.run(["cargo", *args, "check", "--locked"], cwd=work, env=env, check=True)
        print("local composition: duplicate-owner negative control and native type identities passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path, nargs="?")
    parser.add_argument("--snapshot-only", type=Path,
                        help="retain a local Git distribution for native executable integration checks")
    args = parser.parse_args()
    if args.snapshot_only:
        source, revision = distribution(args.snapshot_only.resolve())
        print(json.dumps({"source": source, "revision": revision}))
    elif args.checkout:
        check(args.checkout)
    else:
        parser.error("a checkout or --snapshot-only is required")
