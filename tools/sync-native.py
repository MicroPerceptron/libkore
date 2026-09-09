#!/usr/bin/env python3
"""Export the exact committed native dependency closure from a Kore checkout."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import posixpath
import subprocess
import tarfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def export(checkout: Path, revision: str, check: bool = False):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(checkout), *args])

    revision = git("rev-parse", revision + "^{commit}").decode().strip()
    config = tomllib.loads((ROOT / "Cargo.toml").read_text())
    pending = list(config["package"]["metadata"]["kore"]["owner-paths"].values())
    packages = {}
    while pending:
        path = pending.pop()
        if path in packages:
            continue
        if path.startswith("../") or PurePosixPath(path).is_absolute():
            raise ValueError("dependency path escapes Kore")
        manifest = tomllib.loads(git("show", f"{revision}:{path}/Cargo.toml").decode())
        packages[path] = manifest["package"]["name"]

        def visit(table):
            for key, value in table.items():
                if key in ("dependencies", "dev-dependencies", "build-dependencies"):
                    for dependency in value.values():
                        if isinstance(dependency, dict) and "path" in dependency:
                            pending.append(posixpath.normpath(path + "/" + dependency["path"]))
                        elif isinstance(dependency, dict) and "git" in dependency:
                            raise ValueError("external Git dependency needs an explicit export policy")
                        elif isinstance(dependency, dict) and dependency.get("workspace"):
                            raise ValueError("workspace-inherited dependency needs an explicit export policy")
                elif isinstance(value, dict):
                    visit(value)
        visit(manifest)

    native = ROOT / "native"
    buffers = {}
    hashes = {}
    archive = git("archive", revision, "--", *sorted(packages), "LICENSE")
    with tarfile.open(fileobj=io.BytesIO(archive)) as contents:
        for member in contents:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("archive path escapes the export")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"unsupported source entry: {path}")
            data = contents.extractfile(member).read()
            buffers[str(path)] = (data, 0o755 if member.mode & 0o111 else 0o644)
            hashes[str(path)] = hashlib.sha256(data).hexdigest()
    workspace = '[workspace]\nresolver = "3"\nmembers = [\n'
    workspace += "".join(f"    {json.dumps(path)},\n" for path in sorted(packages)) + ']\n'
    record = {
        "repository": "https://github.com/MicroPerceptron/kore",
        "revision": revision,
        "packages": dict(sorted(packages.items())),
        "workspace_sha256": hashlib.sha256(workspace.encode()).hexdigest(),
        "files": dict(sorted(hashes.items())),
    }
    # Finish reading and validating the replacement before touching an export.
    previous = None
    if (native / "provenance.json").exists():
        subprocess.run(["python3", str(ROOT / "tools/check-native.py")], check=True)
        previous = json.loads((native / "provenance.json").read_text())
    elif native.exists() and any(native.iterdir()):
        raise ValueError("refusing to overwrite an unrecorded native directory")
    if check:
        if previous != record:
            raise ValueError("native export does not match the requested upstream revision")
        print(f"native export matches upstream commit {revision}")
        return
    native.mkdir(exist_ok=True)
    if previous:
        for path in set(previous["files"]) - set(buffers):
            (native / path).unlink()
    for path, (data, mode) in buffers.items():
        destination = native / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        destination.chmod(mode)
    (native / "Cargo.toml").write_text(workspace)
    (native / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"exported {len(packages)} native packages / {len(hashes)} files from {revision}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--check", action="store_true", help="compare to upstream without changing files")
    args = parser.parse_args()
    export(args.checkout.resolve(strict=True), args.revision, args.check)
