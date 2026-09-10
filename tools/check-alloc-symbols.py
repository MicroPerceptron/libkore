#!/usr/bin/env python3
"""Prove c-stdlib emits real linker symbols for all supported Kore targets."""
import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = {'malloc', 'calloc', 'realloc', 'free', 'pthread_create', 'pthread_join', 'pthread_detach',
           '__cxa_thread_atexit', '__cxa_thread_atexit_impl', 'pthread_cancel', 'pthread_key_create', 'pthread_key_delete', 'pthread_setspecific', 'pthread_getspecific', 'pthread_setcancelstate', 'pthread_setcanceltype', 'pthread_testcancel', 'pthread_exit',
           'kore_futex_available', 'kore_futex_wait', 'kore_futex_wake', 'kore_futex_wake_all'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arch', choices=['x86_64', 'aarch64', 'riscv64gc'])
    args = parser.parse_args()
    destination = ROOT / 'target/allocator-symbols'
    destination.mkdir(parents=True, exist_ok=True)
    for arch in ([args.arch] if args.arch else ['x86_64', 'aarch64', 'riscv64gc']):
        object_file = destination / f'{arch}.o'
        subprocess.run([
            'cargo', 'rustc', '--locked', '-p', 'kore-ffi', '--lib', '--all-features',
            '--target', f'targets/{arch}-unknown-kore-user.json',
            '-Z', 'build-std=core,alloc,compiler_builtins',
            '-Z', 'build-std-features=compiler-builtins-mem',
            '--', f'--emit=obj={object_file}',
        ], cwd=ROOT, check=True)
        output = subprocess.check_output([
            os.environ.get('LLVM_NM', 'llvm-nm'), '--defined-only', '--extern-only',
            '--format=posix', str(object_file),
        ], text=True)
        defined = {line.split()[0] for line in output.splitlines() if line.strip()}
        undefined = subprocess.check_output([
            os.environ.get('LLVM_NM', 'llvm-nm'), '--undefined-only', '--extern-only',
            '--format=posix', str(object_file),
        ], text=True)
        if '__tls_get_addr' in undefined:
            raise SystemExit(f'{arch}: dynamic TLS resolver dependency; static initial-exec TLS required')
        if missing := SYMBOLS - defined:
            raise SystemExit(f'{arch}: missing C allocation/thread/futex symbols: {sorted(missing)}')
        print(f'freestanding C allocation/thread/futex symbols: {arch}', flush=True)
