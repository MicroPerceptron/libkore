#!/usr/bin/env python3
"""Generate the C ABI and Zig namespaces from Rust, or check checked-in drift."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def generate(check=False):
    generator = os.environ.get('CBINDGEN', 'cbindgen')
    if subprocess.check_output([generator, '--version'], text=True).strip() != 'cbindgen 0.29.4':
        raise SystemExit('cbindgen 0.29.4 is required')
    with tempfile.TemporaryDirectory() as directory:
        header = Path(directory) / 'kore.h'
        subprocess.run([generator, '--config', 'crates/ffi/cbindgen.toml', '--crate', 'kore-ffi',
                        '--output', str(header)], cwd=ROOT, check=True)
        source = header.read_text()
    # cbindgen is the declaration authority. This step only groups exported
    # names; it never reproduces signatures or guesses C/Rust type layouts.
    functions = sorted(set(re.findall(r'\b(kore_[a-z][a-z0-9_]*)\s*\(', source)))
    if not functions:
        raise SystemExit('no foreign functions found in the generated header')
    groups = {}
    for function in functions:
        _, group, name = function.split('_', 2)
        groups.setdefault(group, []).append((name, function))
    zig = '// Generated from kore.h. Signatures come from Zig translate-c at build time.\n'
    zig += 'pub const abi = @import("kore-c");\n'
    for group, names in sorted(groups.items()):
        zig += f'pub const {group} = struct {{\n'
        for name, function in names:
            zig += f'    pub const @"{name}" = abi.{function};\n'
        zig += '};\n'
    zig_tool = os.environ.get('ZIG', 'zig')
    if subprocess.check_output([zig_tool, 'version'], text=True).strip() != '0.16.0':
        raise SystemExit('Zig 0.16.0 is required')
    zig = subprocess.check_output([zig_tool, 'fmt', '--stdin'], input=zig, text=True)
    for relative, data in [('extern/c/include/kore.h', source), ('extern/zig/src/root.zig', zig)]:
        path = ROOT / relative
        if check:
            if not path.exists() or path.read_text() != data:
                raise SystemExit(f'{relative} is stale; run make bindings')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    generate(parser.parse_args().check)
