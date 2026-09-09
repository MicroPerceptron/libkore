#!/usr/bin/env python3
"""Print a Cargo --config patch for one local Kore checkout; change no manifests."""
import argparse
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def patches(tree: Path, source: str | None = None) -> str:
    tree = tree.resolve(strict=True)
    manifest = tomllib.loads((ROOT / "Cargo.toml").read_text())
    owners = manifest["package"]["metadata"]["kore"]["owner-paths"]
    source = source or manifest["package"]["repository"] + ".git"
    lines = [f"[patch.{json.dumps(source)}]"]
    for name, relative in sorted(owners.items()):
        path = (tree / relative).resolve(strict=True)
        if not path.is_relative_to(tree):
            raise ValueError(f"{name}: owner path escapes the checkout")
        actual = tomllib.loads((path / "Cargo.toml").read_text())["package"]["name"]
        if actual != name:
            raise ValueError(f"{path}: expected {name}, found {actual}")
        lines.append(f"{name} = {{ path = {json.dumps(str(path))} }}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--source", help="Git URL of the libkore dependency being overridden")
    args = parser.parse_args()
    try:
        print(patches(args.checkout, args.source), end="")
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f"local-kore: {error}\n")
