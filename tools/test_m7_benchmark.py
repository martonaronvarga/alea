#!/usr/bin/env python3
"""Offline contracts for the checked M7 benchmark artifact."""

import csv
import hashlib
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "docs/benchmarks/m7-fisher-nuts-2026-10-14.csv"
R2P_ARTIFACT = ROOT / "docs/benchmarks/m7-r2p-2026-10-15.csv"


class M7BenchmarkArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with ARTIFACT.open(newline="") as stream:
            cls.rows = list(csv.DictReader(stream))

    def test_matrix_and_diagnostics_are_complete(self):
        self.assertEqual(len(self.rows), 20)
        expected = {
            (target, method, str(observable))
            for target in ("funnel", "stochastic_volatility")
            for method in ("fisher_nuts", "fisher_walnuts")
            for observable in range(5)
        }
        self.assertEqual(
            {(row["target"], row["method"], row["observable"]) for row in self.rows},
            expected,
        )
        for row in self.rows:
            numeric = [float(value) for key, value in row.items() if key not in {"target", "method"}]
            self.assertTrue(all(math.isfinite(value) for value in numeric))
            self.assertLessEqual(float(row["rhat"]), 1.01)
            self.assertGreaterEqual(float(row["bulk_ess"]), 400.0)
            self.assertGreaterEqual(float(row["tail_ess"]), 400.0)
            self.assertAlmostEqual(
                float(row["ess_per_second"]),
                float(row["bulk_ess"]) / float(row["total_seconds"]),
            )
            self.assertAlmostEqual(
                float(row["ess_per_call"]),
                float(row["bulk_ess"]) / int(row["total_calls"]),
            )

    def test_artifact_is_the_reviewed_run(self):
        self.assertEqual(
            hashlib.sha256(ARTIFACT.read_bytes()).hexdigest(),
            "583e6c8dc39cb01c244d46f238a64ba7c5b623a47bbcebf4dc25432823f65bce",
        )


class M7R2PBenchmarkArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with R2P_ARTIFACT.open(newline="") as stream:
            cls.rows = list(csv.DictReader(stream))

    def test_matrix_and_diagnostics_are_complete(self):
        self.assertEqual(len(self.rows), 30)
        expected = {
            (target, method, str(observable))
            for target in ("funnel", "stochastic_volatility")
            for method in ("fisher_nuts", "fisher_walnuts", "fisher_walnuts_r2p")
            for observable in range(5)
        }
        self.assertEqual(
            {(row["target"], row["method"], row["observable"]) for row in self.rows},
            expected,
        )
        for row in self.rows:
            numeric = [float(value) for key, value in row.items() if key not in {"target", "method"}]
            self.assertTrue(all(math.isfinite(value) for value in numeric))
            self.assertLessEqual(float(row["rhat"]), 1.01)
            self.assertGreaterEqual(float(row["bulk_ess"]), 400.0)
            self.assertGreaterEqual(float(row["tail_ess"]), 400.0)
            self.assertAlmostEqual(
                float(row["ess_per_second"]),
                float(row["bulk_ess"]) / float(row["total_seconds"]),
            )
            self.assertAlmostEqual(
                float(row["ess_per_call"]),
                float(row["bulk_ess"]) / int(row["total_calls"]),
            )

    def test_artifact_is_the_reviewed_run(self):
        self.assertEqual(
            hashlib.sha256(R2P_ARTIFACT.read_bytes()).hexdigest(),
            "253696a66cd2be9a803aa1f20c38e1aab5c8b43a0a8a5f03fa985e55cc4279b5",
        )


if __name__ == "__main__":
    unittest.main()
