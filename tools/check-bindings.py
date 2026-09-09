#!/usr/bin/env python3
"""Check generated declarations and compile C/Zig consumers for each Kore ISA."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    ('x86_64', 'x86_64-none-elf', 'x86_64-freestanding'),
    ('aarch64', 'aarch64-none-elf', 'aarch64-freestanding'),
    ('riscv64', 'riscv64-none-elf', 'riscv64-freestanding'),
]

if __name__ == '__main__':
    zig = os.environ.get('ZIG', 'zig')
    if subprocess.check_output([zig, 'version'], text=True).strip() != '0.16.0':
        raise SystemExit('Zig 0.16.0 is required')
    subprocess.run(['python3', 'tools/generate-bindings.py', '--check'], cwd=ROOT, check=True)
    destination = ROOT / 'target/foreign'
    destination.mkdir(parents=True, exist_ok=True)
    for arch, c_target, zig_target in TARGETS:
        subprocess.run([os.environ.get('CLANG', 'clang'), '-target', c_target, '-ffreestanding',
                        '-std=c11', '-Werror', '-fsyntax-only', '-I', 'extern/c/include',
                        'crates/ffi/tests/consumer.c', 'crates/ffi/tests/alloc.c'], cwd=ROOT, check=True)
        translated = destination / f'{arch}-c.zig'
        with translated.open('w') as output:
            subprocess.run([zig, 'translate-c', '-target', zig_target, 'extern/c/include/kore.h'],
                           cwd=ROOT, stdout=output, check=True)
        subprocess.run([zig, 'build-obj', '-target', zig_target, '-O', 'ReleaseFast',
                        '--dep', 'kore', '-Mroot=crates/ffi/tests/consumer.zig',
                        '--dep', 'kore-c', '-Mkore=extern/zig/src/root.zig',
                        f'-Mkore-c={translated}', f'-femit-bin={destination / (arch + ".o")}'],
                       cwd=ROOT, check=True)
        print(f'C/Zig declarations and ABI assertions: {arch}', flush=True)
    subprocess.run([zig, 'build', 'check'], cwd=ROOT, check=True)
