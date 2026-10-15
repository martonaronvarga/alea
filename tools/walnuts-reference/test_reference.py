#!/usr/bin/env python3
"""Offline provenance and mathematical checks for the Walnutpie trace."""

import csv
import hashlib
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures/walnutpie-trace.csv"
WALNUTPIE_REV = "5710b0334ba69cf0caa35fc02a2338b61d756953"
NUTS_RS_REV = "a762aae513bf7bb5ccb27ffc9a938a923fc837ec"


class ReferenceFixtureTests(unittest.TestCase):
    def test_trace_shape_density_and_digest(self):
        with FIXTURE.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 16)
        self.assertEqual({int(row["seed"]) for row in rows}, {17, 31})
        for seed in (17, 31):
            self.assertEqual(
                [int(row["draw"]) for row in rows if int(row["seed"]) == seed],
                list(range(8)),
            )
        for row in rows:
            q0, q1 = float(row["q0"]), float(row["q1"])
            self.assertAlmostEqual(float(row["logp"]), -0.5 * (q0 * q0 + q1 * q1))
        self.assertEqual(
            hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
            "4fdaf9751c982c4cb96a516b16ad8c790d87c59d4871d742fa1939e129b3e259",
        )

    def test_runner_pins_immutable_revisions(self):
        runner = (HERE / "run.py").read_text()
        self.assertIn(WALNUTPIE_REV, runner)
        self.assertIn(NUTS_RS_REV, runner)
        self.assertNotIn("origin/main", runner)


if __name__ == "__main__":
    unittest.main()
