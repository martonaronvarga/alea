"""First-publication bootstrap cannot hide a different or self baseline."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from initial import validate

ROOT = Path(__file__).resolve().parents[2]


class InitialReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "current"
        self.baseline = Path(self.temporary.name) / "baseline"
        (self.root / "crates").mkdir(parents=True)
        shutil.copyfile(ROOT / "crates/Cargo.toml", self.root / "crates/Cargo.toml")
        shutil.copytree(ROOT / "tools/ci/public-api", self.root / "tools/ci/public-api")
        shutil.copyfile(ROOT / "tools/ci/api-review.json", self.root / "tools/ci/api-review.json")
        snapshots = self.baseline / "tools/ci/public-api"
        snapshots.mkdir(parents=True)
        review = json.loads((self.root / "tools/ci/api-review.json").read_text())
        for name, entry in review["packages"].items():
            after = set((self.root / "tools/ci/public-api" / f"{name}.txt").read_text().splitlines())
            before = (after - set(entry["added_lines"])) | set(entry["removed_lines"])
            (snapshots / f"{name}.txt").write_text("\n".join(sorted(before)) + "\n")

    def test_actual_review_endpoints_pass(self):
        validate(self.root, self.baseline)

    def test_duplicate_rendered_declarations_use_the_review_set_semantics(self):
        p = self.baseline / "tools/ci/public-api/alea-core.txt"
        with p.open("a") as stream:
            stream.write(p.read_text().splitlines()[0] + "\n")
        validate(self.root, self.baseline)

    def test_self_baseline_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "independent"):
            validate(self.root, self.root)

    def test_different_baseline_is_rejected(self):
        with (self.baseline / "tools/ci/public-api/alea-core.txt").open("a") as stream:
            stream.write("pub struct alea_core::Unreviewed\n")
        with self.assertRaisesRegex(ValueError, "not the reviewed"):
            validate(self.root, self.baseline)

    def test_initial_exception_cannot_apply_to_subsequent_versions(self):
        p = self.root / "crates/Cargo.toml"
        p.write_text(p.read_text().replace('version = "0.1.0"', 'version = "0.1.1"', 1))
        with self.assertRaisesRegex(ValueError, "only for 0.1.0"):
            validate(self.root, self.baseline)


if __name__ == "__main__":
    unittest.main()
