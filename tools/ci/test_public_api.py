"""Mutation tests for the accepted public API snapshot contract."""

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import public_api


class PublicApiSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cls.source = cls.root / "tools/ci/public-api"
        cls.manifest = json.loads((cls.source / "manifest.json").read_text())

    def check_rejected(self, mutate_manifest=None, mutate_files=None):
        with tempfile.TemporaryDirectory() as directory:
            snapshot_dir = Path(directory) / "public-api"
            shutil.copytree(self.source, snapshot_dir)
            manifest = copy.deepcopy(self.manifest)
            if mutate_manifest is not None:
                mutate_manifest(manifest)
            if mutate_files is not None:
                mutate_files(snapshot_dir, manifest)
            with self.assertRaises(ValueError):
                public_api.validate(self.root, snapshot_dir, manifest)

    def test_current_snapshots_pass(self):
        public_api.validate(self.root, self.source, self.manifest)

    def test_library_discovery_excludes_binary_only_cli(self):
        libraries = public_api.discover_libraries(self.root)
        self.assertEqual(len(libraries), 8)
        self.assertNotIn("alea-cli", libraries)

    def test_removed_package_fails(self):
        self.check_rejected(lambda manifest: manifest["packages"].pop("alea-smc"))

    def test_wrong_tool_fails(self):
        self.check_rejected(lambda manifest: manifest.__setitem__("tool", "cargo-public-api 0"))

    def test_changed_policy_fails(self):
        self.check_rejected(
            lambda manifest: manifest["policy"].__setitem__("default_features", True)
        )

    def test_removed_feature_variant_fails(self):
        self.check_rejected(
            lambda manifest: manifest["variants"].pop("alea-mcmc+simd")
        )

    def test_changed_feature_variant_fails(self):
        self.check_rejected(
            lambda manifest: manifest["variants"]["alea-mcmc+simd"].__setitem__(
                "features", ["alea-mcmc/faer"]
            )
        )

    def test_changed_snapshot_fails_digest(self):
        def change_file(snapshot_dir, manifest):
            path = snapshot_dir / manifest["packages"]["alea-core"]["file"]
            path.write_text(path.read_text() + "pub mod invented\n")

        self.check_rejected(mutate_files=change_file)

    def test_changed_count_fails(self):
        self.check_rejected(
            lambda manifest: manifest["packages"]["alea-core"].__setitem__("items", 0)
        )

    def test_missing_snapshot_fails(self):
        def remove_file(snapshot_dir, manifest):
            (snapshot_dir / manifest["packages"]["alea-core"]["file"]).unlink()

        self.check_rejected(mutate_files=remove_file)

    def test_stale_snapshot_fails(self):
        self.check_rejected(
            mutate_files=lambda snapshot_dir, _: (snapshot_dir / "stale.txt").write_text("stale\n")
        )

    def test_normalization_rejects_blank_lines(self):
        def add_blank(snapshot_dir, manifest):
            package = "alea-core"
            path = snapshot_dir / manifest["packages"][package]["file"]
            content = path.read_text() + "\n"
            path.write_text(content)
            manifest["packages"][package]["sha256"] = public_api.digest(content)
            manifest["packages"][package]["items"] = len(content.splitlines())

        self.check_rejected(mutate_files=add_blank)


if __name__ == "__main__":
    unittest.main()
