#!/usr/bin/env python3
"""Offline contracts for the checked M4F rank/history measurement artifact."""

import hashlib
import json
from pathlib import Path
import statistics
import unittest


ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "docs/benchmarks/m4f-budgets-2026-10-14.json"


class BudgetArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = json.loads(ARTIFACT.read_text())

    def test_fixed_matrix_and_raw_samples_are_complete(self):
        expected = {
            f"d{dimension}_h{history}_r{rank}"
            for dimension, history, ranks in (
                (64, 40, (1, 2, 4)),
                (64, 160, (2, 4, 8)),
                (256, 40, (2, 4, 8)),
                (256, 160, (2, 4, 8)),
            )
            for rank in ranks
        }
        self.assertEqual(set(self.artifact["results"]), expected)
        self.assertEqual(self.artifact["measured_rounds"], 5)
        for result in self.artifact["results"].values():
            self.assertEqual(len(result["samples"]), 5)
            self.assertEqual(
                result["minimum_ring_bytes"],
                2 * result["dimension"] * result["history"] * 8,
            )
            walls = [sample["wall_seconds"] for sample in result["samples"]]
            self.assertAlmostEqual(result["wall_mean_seconds"], statistics.mean(walls))
            self.assertAlmostEqual(
                result["per_fit_mean_seconds"],
                statistics.mean(walls) / result["repetitions"],
            )

    def test_recorded_source_hashes_still_match(self):
        for relative, expected in self.artifact["source_sha256"].items():
            actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, relative)


if __name__ == "__main__":
    unittest.main()
