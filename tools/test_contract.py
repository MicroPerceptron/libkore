"""Adversarial checks for the distribution and local-integration boundaries."""
import importlib.util
from pathlib import Path
import re
import hashlib
import json
import tempfile
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("owners", ROOT / "tools/check-owners.py")
owners = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owners)


def graph(*rows):
    return {
        "packages": [{"name": name, "id": identity} for name, identity in rows],
        "resolve": {"nodes": [{"id": identity} for _, identity in rows]},
    }


class DistributionContract(unittest.TestCase):
    def test_export_and_patch_roots_cannot_drift(self):
        manifest = tomllib.loads((ROOT / "Cargo.toml").read_text())
        paths = manifest["package"]["metadata"]["kore"]["owner-paths"]
        # Executable-only owners live in the entry facade, not in reusable
        # kore-std (which must not acquire lang items or a panic handler).
        native_roots = {}
        for relative in [".", *manifest["workspace"]["members"]]:
            package_dir = (ROOT / relative).resolve()
            package = tomllib.loads((package_dir / "Cargo.toml").read_text())
            for name, dep in package.get("dependencies", {}).items():
                if not isinstance(dep, dict) or "path" not in dep:
                    continue
                dependency = (package_dir / dep["path"]).resolve()
                if dependency.is_relative_to(ROOT / "native"):
                    name = dep.get("package", name)
                    path = str(dependency.relative_to(ROOT / "native"))
                    self.assertEqual(native_roots.setdefault(name, path), path)
        self.assertEqual(native_roots, paths)
        record = json.loads((ROOT / "native/provenance.json").read_text())
        self.assertTrue(re.fullmatch(r"[0-9a-f]{40}", record["revision"]))
        for name, path in paths.items():
            self.assertEqual(record["packages"][path], name)
        self.assertTrue(all(not Path(path).is_absolute() for path in paths.values()))

    def test_user_targets_preserve_frames_for_crash_capture(self):
        for arch in ('x86_64', 'aarch64', 'riscv64gc'):
            target = json.loads((ROOT / f'targets/{arch}-unknown-kore-user.json').read_text())
            self.assertEqual(target['frame-pointer'], 'always', arch)
            self.assertEqual(target['panic-strategy'], 'abort', arch)

    def test_native_drift_is_detected(self):
        spec = importlib.util.spec_from_file_location("native", ROOT / "tools/check-native.py")
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        with tempfile.TemporaryDirectory() as temporary:
            tree = Path(temporary)
            (tree / "Cargo.toml").write_text("workspace")
            (tree / "lib.rs").write_text("original")
            record = {"workspace_sha256": hashlib.sha256(b"workspace").hexdigest(),
                      "files": {"lib.rs": hashlib.sha256(b"original").hexdigest()}}
            (tree / "provenance.json").write_text(json.dumps(record))
            self.assertEqual(native.violations(tree), [])
            (tree / "lib.rs").write_text("modified")
            self.assertTrue(native.violations(tree))
            (tree / "lib.rs").write_text("original")
            (tree / "hidden.rs").write_text("parallel owner")
            self.assertTrue(native.violations(tree))

    def test_distribution_needs_no_cross_repository_git_fetch(self):
        for relative in ("Cargo.lock", "native/Cargo.lock"):
            lock = tomllib.loads((ROOT / relative).read_text())
            self.assertFalse(any(package.get("source", "").startswith("git+")
                                 for package in lock["package"]), relative)

    def test_same_version_from_two_sources_is_still_two_owners(self):
        errors = owners.violations(graph(
            ("kore-std", "facade"), ("kore-abi", "path#0.1.4"),
            ("kore-abi", "git#0.1.4"),
        ))
        self.assertTrue(any("kore-abi" in error for error in errors))

    def test_kernel_and_vacuous_success_are_rejected(self):
        self.assertTrue(owners.violations(graph()))
        self.assertTrue(owners.violations(graph(("kore-std", "std"), ("kore", "kernel"))))

    def test_coherent_local_or_git_graph_is_valid(self):
        for source in ("path", "git"):
            self.assertEqual(owners.violations(graph(
                ("kore-std", "facade"), ("kore-abi", source + "abi"),
                ("kore-rt", source + "rt"),
            )), [])


if __name__ == "__main__":
    unittest.main()
