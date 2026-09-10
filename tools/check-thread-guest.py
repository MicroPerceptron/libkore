#!/usr/bin/env python3
"""Run the foreign thread lifecycle witness in an isolated, clean Kore worktree.

Uses the existing founded hello/runtest gate and restores tracked worktree files
on completion. Build caches and the gate's test disk are retained for inspection.
Never use the concurrently edited primary checkout.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def run(tree):
    tree = tree.resolve(strict=True)
    if not (tree / '.git').is_file():
        raise SystemExit('supply an isolated git worktree, not the primary checkout')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=tree).strip():
        raise SystemExit('Kore worktree must be clean before guest validation')
    if subprocess.run(['pgrep', '-x', 'qemu-system-x86'], stdout=subprocess.DEVNULL).returncode != 1:
        raise SystemExit('another QEMU is running or process inspection failed; serialize guest gates')
    spec = importlib.util.spec_from_file_location('local_kore', ROOT / 'tools/local-kore.py')
    local = importlib.util.module_from_spec(spec); spec.loader.exec_module(local)
    with tempfile.TemporaryDirectory(prefix='libkore-thread-guest-') as temporary:
        snapshot = Path(temporary) / 'library'; snapshot.mkdir()
        files = subprocess.check_output(['git', 'ls-files', '-c', '-o', '--exclude-standard', '-z'], cwd=ROOT).decode().split('\0')
        for name in sorted(set(files)):
            if not name or not (ROOT / name).is_file(): continue
            destination = snapshot / name; destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
        subprocess.run(['git', 'init', '-q'], cwd=snapshot, check=True)
        subprocess.run(['git', 'add', '.'], cwd=snapshot, check=True)
        subprocess.run(['git', '-c', 'user.name=libkore test', '-c', 'user.email=test@localhost', '-c', 'commit.gpgsign=false', 'commit', '-qm', 'guest fixture'], cwd=snapshot, check=True)
        source = snapshot.as_uri()
        hello = tree / 'user/apps/system/hello'
        injected = hello / 'src/foreign_thread_guest.rs'
        config = tree / '.cargo/config.toml'
        config_existed = config.exists()
        config_bytes = config.read_bytes() if config_existed else b''
        try:
            with (hello / 'Cargo.toml').open('a') as out:
                out.write(f'\nkore-ffi = {{ git = {json.dumps(source)}, features = ["c-stdlib"] }}\n')
                out.write('kore-std = { package = "kore-std", git = '+json.dumps(source)+', features = ["heap"] }\n')
            main = hello / 'src/main.rs'
            text = main.read_text()
            hook = "if status == ANSWER && !thread_tls_witness() {\n        BAD_TLS"
            replacement = "if status == ANSWER && !foreign_thread_guest::run() {\n        ExitCode::app(70 + foreign_thread_guest::stage())"
            if hook not in text: raise RuntimeError('hello entry hook changed')
            text = text.replace(hook, replacement)
            main.write_text(text + '\nmod foreign_thread_guest;\n')
            # The original allocation-free hello has profile 0. The actual FFI
            # controller allocates through the normal endowed process heap.
            catalog = tree / 'services/exec/src/lib.rs'
            catalog_text = catalog.read_text()
            old_profile = 'name: "hello",\n        contract: AppContract::system("org.kore.system.hello"),\n        profile: 0,'
            if old_profile not in catalog_text: raise RuntimeError('hello founding profile changed')
            catalog.write_text(catalog_text.replace(old_profile, old_profile[:-2] + 'HEAP,', 1))
            makefile = tree / 'Makefile'
            make_text = makefile.read_text()
            if 'hello=$(2)/hello' not in make_text: raise RuntimeError('hello archive profile changed')
            makefile.write_text(make_text.replace('hello=$(2)/hello', 'hello@heap=$(2)/hello', 1))
            shutil.copyfile(ROOT / 'tests/guest/thread_lifecycle.rs', injected)
            config.parent.mkdir(exist_ok=True)
            config.write_bytes(config_bytes + b'\n' + local.patches(tree, source).encode())
            for target in (tree / 'targets').glob('*-unknown-kore-user.json'):
                data = json.loads(target.read_text()); data['tls-model'] = 'initial-exec'
                target.write_text(json.dumps(data, indent=2) + '\n')
            # Cargo must materialize the test-only Git dependency before locked make.
            subprocess.run(['cargo', 'metadata', '--format-version=1'], cwd=tree / 'user', stdout=subprocess.DEVNULL, check=True)
            metadata = json.loads(subprocess.check_output(['cargo', 'metadata', '--format-version=1', '--locked'], cwd=tree / 'user'))
            nodes = {node['id']: node for node in metadata['resolve']['nodes']}
            reached = set()
            pending = [package['id'] for package in metadata['packages'] if package['name'] == 'hello']
            while pending:
                identity = pending.pop()
                if identity in reached: continue
                reached.add(identity)
                pending.extend(nodes[identity]['dependencies'])
            metadata['resolve']['nodes'] = [nodes[identity] for identity in reached]
            subprocess.run(['python3', str(ROOT / 'tools/check-owners.py')], input=json.dumps(metadata).encode(), check=True)
            # The fixture changes hello's founding profile; do not reuse a
            # production store carrying its allocation-free authorization.
            disk = 'target/' + Path(temporary).name + '.qcow2'
            for target in ['user-x86', 'boot-digests', 'boot-manifest', 'runtest-x86']:
                subprocess.run(['make', target, 'MODE=release', 'NVME_IMG=' + disk, 'USER_BUILD_STD=-Z build-std=core,alloc,compiler_builtins -Z build-std-features=compiler-builtins-mem'], cwd=tree, check=True)
        finally:
            injected.unlink(missing_ok=True)
            if config_existed: config.write_bytes(config_bytes)
            else:
                config.unlink(missing_ok=True)
                try: config.parent.rmdir()
                except OSError: pass
            # Initial cleanliness made every tracked edit attributable to this run.
            changed = subprocess.check_output(['git', 'diff', '--name-only', '-z'], cwd=tree).decode().split('\0')
            changed = [name for name in changed if name]
            if changed: subprocess.run(['git', 'restore', '--worktree', '--', *changed], cwd=tree, check=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkout', type=Path)
    run(parser.parse_args().checkout)
