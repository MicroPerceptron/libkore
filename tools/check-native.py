#!/usr/bin/env python3
"""Check the native export against its recorded source bytes and workspace."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def violations(native: Path) -> list[str]:
    record = json.loads((native / "provenance.json").read_text())
    errors = []
    expected = record["files"]
    actual = set()
    for directory, dirs, files in os.walk(native):
        relative = Path(directory).relative_to(native)
        if relative == Path("."):
            dirs[:] = [name for name in dirs if name != "target"]
        for name in files:
            path = relative / name
            if str(path) not in ("Cargo.toml", "Cargo.lock", "provenance.json"):
                actual.add(str(path))
    if actual != set(expected):
        errors.append(f"native file set changed: {sorted(actual ^ set(expected))}")
    for path, digest in expected.items():
        file = native / path
        if (Path(path).is_absolute() or ".." in Path(path).parts
                or not file.resolve().is_relative_to(native.resolve())
                or file.is_symlink() or not file.is_file()
                or hashlib.sha256(file.read_bytes()).hexdigest() != digest):
            errors.append(f"native source differs: {path}")
    if hashlib.sha256((native / "Cargo.toml").read_bytes()).hexdigest() != record["workspace_sha256"]:
        errors.append("native workspace differs")
    return errors


if __name__ == "__main__":
    errors = violations(ROOT / "native")
    if errors:
        raise SystemExit("\n".join(errors))
    print("native export: exact recorded source bytes; no untracked implementation files")
