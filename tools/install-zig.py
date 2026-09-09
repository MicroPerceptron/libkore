#!/usr/bin/env python3
"""Install checksum-pinned official Zig 0.15.2 into a caller-selected directory."""
import argparse
import hashlib
from pathlib import Path
import platform
import tarfile
import tempfile
import urllib.request

PINS = {
    ('Linux', 'x86_64'): ('x86_64-linux', '02aa270f183da276e5b5920b1dac44a63f1a49e55050ebde3aecc9eb82f93239'),
    ('Darwin', 'arm64'): ('aarch64-macos', '3cc2bab367e185cdfb27501c4b30b1b0653c28d9f73df8dc91488e66ece5fa6b'),
}
VERSION = '0.15.2'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    arch, expected = PINS[(platform.system(), platform.machine())]
    folder = f'zig-{arch}-{VERSION}'
    destination = args.destination.resolve()
    if (destination / folder).exists():
        raise SystemExit(f'refusing to replace {destination / folder}')
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile() as archive:
        digest = hashlib.sha256()
        with urllib.request.urlopen(f'https://ziglang.org/download/{VERSION}/{folder}.tar.xz', timeout=60) as response:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                archive.write(chunk)
        if digest.hexdigest() != expected:
            raise SystemExit('Zig archive checksum mismatch')
        archive.seek(0)
        with tarfile.open(fileobj=archive, mode='r:xz') as contents:
            contents.extractall(destination, filter='data')
    print(destination / folder / 'zig')
