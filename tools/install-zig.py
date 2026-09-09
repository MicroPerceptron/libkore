#!/usr/bin/env python3
"""Install checksum-pinned official Zig 0.16.0 into a caller-selected directory."""
import argparse
import hashlib
from pathlib import Path
import platform
import tarfile
import tempfile
import urllib.request

PINS = {
    ('Linux', 'x86_64'): ('x86_64-linux', '70e49664a74374b48b51e6f3fdfbf437f6395d42509050588bd49abe52ba3d00'),
    ('Darwin', 'arm64'): ('aarch64-macos', 'b23d70deaa879b5c2d486ed3316f7eaa53e84acf6fc9cc747de152450d401489'),
}
VERSION = '0.16.0'

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
