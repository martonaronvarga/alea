"""Compatibility classification and downstream/shallow-checkout mutation gates."""
import json
from pathlib import Path
import tempfile
import unittest

import api_compatibility as api


class CompatibilityTests(unittest.TestCase):
    def test_removal_is_breaking(self):
        result = api.difference(["pub struct x::A"], [])
        self.assertEqual(result["classification"], "potentially_breaking")

    def test_signature_change_is_breaking(self):
        result = api.difference(["pub fn x::f() -> u8"], ["pub fn x::f() -> u64"])
        self.assertEqual(result["classification"], "potentially_breaking")

    def test_variant_addition_needs_review(self):
        before = ["pub enum x::E"]
        unchanged = api.difference(before, before)
        added = api.difference(before, before + ["pub x::E::New"])
        self.assertEqual(added["classification"], "potentially_breaking")
        self.assertNotEqual(unchanged["delta"], added["delta"])

    def test_new_standalone_type_is_additive(self):
        result = api.difference([], ["pub struct x::A"])
        self.assertEqual(result["classification"], "additive")

    def test_reordering_is_unchanged(self):
        self.assertEqual(api.difference(["a", "b"], ["b", "a"])["classification"], "unchanged")

    def test_review_is_verifiable_without_historical_git_objects_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshots = root / "tools/ci/public-api"
            snapshots.mkdir(parents=True)
            snapshot = snapshots / "alea-example.txt"
            content = "pub struct alea_example::A\n"
            snapshot.write_text(content)
            review = {
                "schema_version": 1,
                "base_revision": "historical-provenance-only",
                "decision": "compatible",
                "rationale": "reviewed standalone addition",
                "packages": {"alea-example": api.difference([], content.splitlines())},
            }
            (snapshots.parent / "api-review.json").write_text(json.dumps(review))
            api.check(root)
            snapshot.write_text(content + "pub fn alea_example::unreviewed()\n")
            with self.assertRaises(ValueError):
                api.check(root)

    def test_breaking_delta_cannot_be_adjudicated_as_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshots = root / "tools/ci/public-api"
            snapshots.mkdir(parents=True)
            (snapshots / "alea-example.txt").write_text("")
            review = {
                "schema_version": 1,
                "base_revision": "historical-provenance-only",
                "decision": "compatible",
                "rationale": "incorrectly labeled removal",
                "packages": {"alea-example": api.difference(["pub struct alea_example::A"], [])},
            }
            (snapshots.parent / "api-review.json").write_text(json.dumps(review))
            with self.assertRaises(ValueError):
                api.check(root)


if __name__ == "__main__":
    unittest.main()
