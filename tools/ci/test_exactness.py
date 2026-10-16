"""Mutation tests for the M13 implemented-map law inventory."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import exactness


class ExactnessGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cls.ledger = json.loads((cls.root / "tools/ci/exactness-ledger.json").read_text())

    def check_rejected(self, mutate):
        ledger = copy.deepcopy(self.ledger)
        mutate(ledger)
        with self.assertRaises(ValueError):
            exactness.validate(self.root, ledger)

    def test_current_inventory_passes(self):
        counts = exactness.validate(self.root, self.ledger)
        self.assertEqual(
            counts,
            {"approximate_target": 1, "exact_target": 11, "exactness_pending": 1},
        )

    def test_unlisted_transition_fails(self):
        self.check_rejected(lambda ledger: ledger["transitions"].pop())

    def test_duplicate_transition_fails(self):
        self.check_rejected(
            lambda ledger: ledger["transitions"].__setitem__(
                1, copy.deepcopy(ledger["transitions"][0])
            )
        )

    def test_exact_transition_cannot_have_open_law(self):
        def open_law(ledger):
            entry = next(item for item in ledger["transitions"] if item["guarantee"] == "exact_target")
            entry["laws"]["support"]["status"] = "open"

        self.check_rejected(open_law)

    def test_pending_transition_must_name_open_law(self):
        def close_laws(ledger):
            entry = next(
                item for item in ledger["transitions"] if item["guarantee"] == "exactness_pending"
            )
            for law in entry["laws"].values():
                law["status"] = "closed"

        self.check_rejected(close_laws)

    def test_approximate_transition_must_expose_omitted_correction(self):
        def invent_correction(ledger):
            entry = next(
                item for item in ledger["transitions"] if item["guarantee"] == "approximate_target"
            )
            entry["laws"]["selection"]["status"] = "closed"

        self.check_rejected(invent_correction)

    def test_research_transition_requires_promotion_gate(self):
        def remove_gate(ledger):
            entry = next(item for item in ledger["transitions"] if item["maturity"] == "S1")
            del entry["promotion_gate"]

        self.check_rejected(remove_gate)

    def test_evidence_symbol_must_exist(self):
        self.check_rejected(
            lambda ledger: ledger["transitions"][0]["evidence"][0].__setitem__(
                "symbol", "invented_test_symbol"
            )
        )

    def test_discovery_deduplicates_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "tools/ci/public-api"
            snapshot.mkdir(parents=True)
            (snapshot / "alea-mcmc.txt").write_text(
                "impl<T> alea_mcmc::MarkovChain for alea_mcmc::hmc::Hmc<'_, T>\n"
                "impl<T> alea_mcmc::MarkovChain for alea_mcmc::hmc::Hmc<'_, T>\n"
            )
            self.assertEqual(
                exactness.discover_transitions(root), {"alea_mcmc::hmc::Hmc"}
            )


if __name__ == "__main__":
    unittest.main()
