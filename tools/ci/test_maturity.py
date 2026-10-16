"""Mutation tests for the M13 maturity inventory."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import maturity


class MaturityGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cls.ledger = json.loads((cls.root / "tools/ci/maturity-ledger.json").read_text())

    def check_rejected(self, mutate):
        ledger = copy.deepcopy(self.ledger)
        mutate(ledger)
        with self.assertRaises(ValueError):
            maturity.validate(self.root, ledger)

    def test_current_inventory_passes(self):
        maturity.validate(self.root, self.ledger)

    def test_unlisted_public_module_fails(self):
        self.check_rejected(lambda ledger: ledger["source_modules"].pop())

    def test_duplicate_public_module_fails(self):
        self.check_rejected(
            lambda ledger: ledger["source_modules"].__setitem__(
                1, copy.deepcopy(ledger["source_modules"][0])
            )
        )

    def test_unlisted_api_namespace_fails(self):
        self.check_rejected(lambda ledger: ledger["api_namespaces"].pop())

    def test_duplicate_api_namespace_fails(self):
        self.check_rejected(
            lambda ledger: ledger["api_namespaces"].__setitem__(
                1, copy.deepcopy(ledger["api_namespaces"][0])
            )
        )

    def test_api_namespace_must_match_crate(self):
        self.check_rejected(
            lambda ledger: ledger["api_namespaces"][0].__setitem__(
                "path", "wrong::reference"
            )
        )

    def test_unknown_profile_fails(self):
        self.check_rejected(
            lambda ledger: ledger["source_modules"][0].__setitem__("profile", "unknown")
        )

    def test_open_gate_blocks_accidental_promotion(self):
        def promote(ledger):
            entry = next(item for item in ledger["source_modules"] if "promotion_gate" in item)
            entry["profile"] = "validated_component_reference"

        self.check_rejected(promote)

    def test_research_entry_requires_promotion_gate(self):
        def remove_gate(ledger):
            entry = next(item for item in ledger["source_modules"] if "promotion_gate" in item)
            del entry["promotion_gate"]

        self.check_rejected(remove_gate)

    def test_release_claim_fails_before_release_gate(self):
        self.check_rejected(
            lambda ledger: ledger["profiles"]["validated_component_reference"].__setitem__(
                "maturity", "S3"
            )
        )

    def test_unknown_maturity_fails(self):
        self.check_rejected(
            lambda ledger: ledger["profiles"]["validated_component_reference"].__setitem__(
                "maturity", "S9"
            )
        )

    def test_design_contract_cannot_be_adopted(self):
        self.check_rejected(
            lambda ledger: ledger["profiles"]["design_contract"].__setitem__(
                "adoption", "opt_in"
            )
        )

    def test_approximate_target_cannot_look_exact(self):
        self.check_rejected(
            lambda ledger: ledger["profiles"]["research_approximate"].__setitem__(
                "adoption", "opt_in"
            )
        )

    def test_unlisted_feature_fails(self):
        self.check_rejected(lambda ledger: ledger["cargo_features"].pop())

    def test_duplicate_feature_fails(self):
        self.check_rejected(
            lambda ledger: ledger["cargo_features"].__setitem__(
                1, copy.deepcopy(ledger["cargo_features"][0])
            )
        )

    def test_unlisted_cli_command_fails(self):
        self.check_rejected(lambda ledger: ledger["cli_commands"].pop())

    def test_invented_cli_source_fails(self):
        self.check_rejected(
            lambda ledger: ledger["cli_commands"][0].__setitem__("path", "src/missing.rs")
        )

    def test_restricted_visibility_is_not_a_public_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "crates/alea-test/src"
            source.mkdir(parents=True)
            (source / "lib.rs").write_text(
                "pub(crate) fn crate_only() {}\npub(super) struct ParentOnly;\n"
            )
            self.assertEqual(maturity.discover_public_modules(root), set())
            (source / "lib.rs").write_text("pub fn exported() {}\n")
            self.assertEqual(
                maturity.discover_public_modules(root),
                {"crates/alea-test/src/lib.rs"},
            )

    def test_exported_macro_is_a_public_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "crates/alea-test/src"
            source.mkdir(parents=True)
            (source / "lib.rs").write_text(
                "#[macro_export]\nmacro_rules! exported { () => {}; }\n"
            )
            self.assertEqual(
                maturity.discover_public_modules(root),
                {"crates/alea-test/src/lib.rs"},
            )


if __name__ == "__main__":
    unittest.main()
