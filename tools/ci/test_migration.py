"""Mutation tests for inventory enforcement; never modify the actual source tree."""
import copy
import json
from pathlib import Path
import unittest

import migration


class MigrationGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cls.ledger = json.loads((cls.root / "tools/ci/migration-ledger.json").read_text())
        cls.features = json.loads((cls.root / "tools/ci/migration-features.json").read_text())

    def check_rejected(self, mutate):
        ledger, features = copy.deepcopy(self.ledger), copy.deepcopy(self.features)
        mutate(ledger, features)
        with self.assertRaises(ValueError):
            migration.validate(self.root, ledger, features)

    def test_current_inventory_passes(self):
        migration.validate(self.root, self.ledger, self.features)

    def test_deleted_source_fails(self):
        self.check_rejected(lambda ledger, _: ledger["files"].pop())

    def test_duplicate_source_fails(self):
        self.check_rejected(lambda ledger, _: ledger["files"].__setitem__(0, ledger["files"][1]))

    def test_deleted_baseline_test_fails(self):
        self.check_rejected(lambda ledger, _: next(e for e in ledger["files"] if e.get("tests"))["tests"].pop())

    def test_preserved_source_hash_mismatch_fails(self):
        self.check_rejected(lambda ledger, _: next(e for e in ledger["files"] if "preserved_git_blob" in e).__setitem__("preserved_git_blob", "0" * 40))

    def test_removed_feature_fails(self):
        self.check_rejected(lambda _, features: features["features"].pop())

    def test_missing_feature_test_fails(self):
        self.check_rejected(lambda _, features: features["features"][0]["checks"][0].__setitem__("name", "this_test_does_not_exist"))

    def test_implemented_feature_without_tests_fails(self):
        self.check_rejected(lambda _, features: features["features"][0].__setitem__("checks", []))

    def test_research_cannot_be_claimed_as_implemented_without_tests(self):
        self.check_rejected(lambda _, features: next(f for f in features["features"] if f["status"] == "preserved_research").__setitem__("status", "migrated"))


if __name__ == "__main__":
    unittest.main()
