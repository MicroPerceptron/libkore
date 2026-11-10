#!/usr/bin/env python3
"""Link libkore.so for every Kore ISA and check the image a C program binds.

The image is a distribution's one-line cdylib over kore-c (Kore's is
user/runtime/libkore), linked with the contract kore-c documents; this builds
that crate in a temporary workspace, as check-entry.py builds its consumers.
The library is what an independent C program on Kore links with -lkore, so the
checks are the ones its loader and its consumers rely on:

- a shared object with no entry point: Kore's image gate classifies a module by
  its entry, and a library must never register as a program;
- the soname consumers record, nothing it needs, nothing left undefined;
- static initial-exec TLS only (Kore's startup linker resolves no dynamic TLS);
- exactly the declared ABI exported: every function kore.h and kore/program.h
  declare, the C names kore/stdlib.h and kore/pthread.h alias, and the C
  library's two lifecycle entry points; no other symbol.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
ARCHES = ('x86_64', 'aarch64', 'riscv64gc')
HEADERS = ('extern/c/include/kore.h', 'extern/c/include/kore/program.h')
# kore-ffi's `c-stdlib` names: allocation, the pthread lifecycle, and the
# compiler ABI's thread-exit destructor hooks.
C_STDLIB = {'malloc', 'calloc', 'realloc', 'free', 'pthread_create', 'pthread_join',
            'pthread_detach', 'pthread_cancel', 'pthread_key_create', 'pthread_key_delete',
            'pthread_setspecific', 'pthread_getspecific', 'pthread_setcancelstate',
            'pthread_setcanceltype', 'pthread_testcancel', 'pthread_exit',
            '__cxa_thread_atexit', '__cxa_thread_atexit_impl'}
# The C library's start and exit call these when a program links libkore.
LIFECYCLE = {'__kore_program_start', '__kore_program_exit'}
DYNAMIC_TLS = re.compile(r'TLSDESC|DTPMOD|DTPOFF|TLSGD|TLSLD|TLS_DTP')


def readelf(*args):
    return subprocess.check_output([os.environ.get('LLVM_READELF', 'llvm-readelf'), '-W', *args],
                                   text=True)


def declared():
    names = set()
    for header in HEADERS:
        names |= set(re.findall(r'\b(kore_[a-z][a-z0-9_]*)\s*\(', (ROOT / header).read_text()))
    if not names:
        raise SystemExit('no foreign functions found in the generated headers')
    return names


# The shared object as a distribution builds it: a cdylib named `kore` that
# names kore-c, linked with the soname and no undefined symbol.
WRAPPER_MANIFEST = ('[package]\nname = "libkore-so"\nversion = "0.0.0"\nedition = "2024"\n'
                    '[lib]\nname = "kore"\ncrate-type = ["cdylib"]\n[workspace]\n'
                    '[dependencies]\nkore-c = {{ path = {path} }}\n')
WRAPPER_SOURCE = '#![no_std]\n// The runtime and its exports come from kore-c.\nuse kore_c as _;\n'
LINK_CONTRACT = ['-C', 'link-arg=--soname=libkore.so', '-C', 'link-arg=--no-undefined']


def build(arch):
    channel = tomllib.loads((ROOT / 'rust-toolchain.toml').read_text())['toolchain']['channel']
    target_dir = ROOT / 'target/program-consumer'
    env = dict(os.environ, RUSTUP_TOOLCHAIN=channel, CARGO_TARGET_DIR=str(target_dir),
               CARGO_INCREMENTAL='0')
    with tempfile.TemporaryDirectory(prefix='libkore-so-') as temporary:
        work = Path(temporary)
        (work / 'src').mkdir()
        (work / 'Cargo.toml').write_text(
            WRAPPER_MANIFEST.format(path=json.dumps(str(ROOT / 'crates/c'))))
        (work / 'src/lib.rs').write_text(WRAPPER_SOURCE)
        # Reuse the reviewed dependency versions while Cargo adds this fixture.
        (work / 'Cargo.lock').write_bytes((ROOT / 'Cargo.lock').read_bytes())
        subprocess.run([
            'cargo', 'rustc', '--release', '-Zjson-target-spec',
            '--target', str(ROOT / f'targets/{arch}-unknown-kore-user.json'),
            '-Zbuild-std=core,alloc,compiler_builtins',
            '-Zbuild-std-features=compiler-builtins-mem', '--', *LINK_CONTRACT,
        ], cwd=work, env=env, check=True)
    return target_dir / f'{arch}-unknown-kore-user/release/libkore.so'


def check(arch, expected):
    image = build(arch)
    header = readelf('--file-header', str(image))
    if not re.search(r'Type:\s+DYN', header):
        raise SystemExit(f'{arch}: libkore.so is not a shared object')
    if not re.search(r'Entry point address:\s+0x0\b', header):
        raise SystemExit(f'{arch}: libkore.so has an entry point; it would register as a program')
    dynamic = readelf('--dynamic', str(image))
    if 'Library soname: [libkore.so]' not in dynamic:
        raise SystemExit(f'{arch}: libkore.so does not carry its soname')
    if needed := re.findall(r'\(NEEDED\)\s+Shared library: \[([^\]]+)\]', dynamic):
        raise SystemExit(f'{arch}: libkore.so needs other libraries: {needed}')
    exported, undefined = set(), set()
    for line in readelf('--dyn-syms', str(image)).splitlines():
        fields = line.split()
        if len(fields) < 8 or not fields[0].rstrip(':').isdigit():
            continue
        kind, binding, section, name = fields[3], fields[4], fields[6], fields[7]
        if section == 'UND':
            if name:
                undefined.add(name)
        elif binding in ('GLOBAL', 'WEAK') and kind in ('FUNC', 'OBJECT', 'TLS'):
            exported.add(name.split('@')[0])
    if undefined:
        raise SystemExit(f'{arch}: undefined symbols: {sorted(undefined)}')
    if missing := expected - exported:
        raise SystemExit(f'{arch}: declared but not exported: {sorted(missing)}')
    if extra := exported - expected:
        raise SystemExit(f'{arch}: exported outside the declared ABI: {sorted(extra)}')
    relocations = readelf('--relocations', str(image))
    if found := sorted(set(DYNAMIC_TLS.findall(relocations))):
        raise SystemExit(f'{arch}: dynamic TLS relocations {found}; static initial-exec TLS required')
    print(f'libkore.so: {arch}: {len(exported)} exported symbols, the declared ABI exactly', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--arch', choices=ARCHES)
    args = parser.parse_args()
    expected = declared() | C_STDLIB | LIFECYCLE
    for arch in [args.arch] if args.arch else ARCHES:
        check(arch, expected)
