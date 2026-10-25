#!/usr/bin/env python3
"""Offline contracts for the checked M4F rank/history measurement artifact."""

import copy
import hashlib
import json
from pathlib import Path
import statistics
import tomllib
import unittest


ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "docs/benchmarks/m4f-budgets-2026-10-14.json"
FISHER_SOURCE = "crates/alea-mcmc/src/adapt/fisher.rs"
FISHER_ARCHIVE = ROOT / "docs/benchmarks/m4f-measured-sources" / FISHER_SOURCE


def validate_fisher_extension(measured, current):
    """Only the reviewed native-NUTS module/export extend this historical file."""
    expected = measured.replace("mod weighted;", "mod nuts;\nmod weighted;")
    expected = expected.replace("use rand::Rng;", "pub use nuts::{FisherNutsWarmup, FisherNutsWarmupError, FisherNutsWarmupResult};\nuse rand::Rng;")
    if current != expected:
        raise ValueError("unreviewed post-measurement Fisher estimator/controller change")


def validate_lock_extension(measured, current):
    """Admit only M14 edges and the pinned, measured NUTS SIMD/normal packages."""
    expected = copy.deepcopy(measured)
    packages = {p["name"]: p for p in expected["package"]}
    for name, additions in (
        ("alea-distributions", ["allocation-counter", "statrs"]),
        ("alea-math", ["pulp"]),
        ("alea-mcmc", ["pulp", "rand_distr 0.6.0"]),
    ):
        packages[name]["dependencies"] = sorted(packages[name]["dependencies"] + additions)
    packages["pulp"].update(
        version="0.22.3",
        checksum="046aa45b989642ec2e4717c8e72d677b13edd831a4d3b6cf37d9a3e54912496a",
    )
    packages["pulp-wasm-simd-flag"].update(
        version="0.1.1",
        checksum="1d8f70e07b9c3962945a74e59ca1c511bba65b6419468acc217c457d93f3c740",
    )
    packages["faer"]["dependencies"] = [
        "rand_distr 0.5.1" if name == "rand_distr" else name
        for name in packages["faer"]["dependencies"]
    ]
    expected["package"].append({
        "name": "rand_distr", "version": "0.6.0",
        "source": "registry+https://github.com/rust-lang/crates.io-index",
        "checksum": "4d431c2703ccf129de4d45253c03f49ebb22b97d6ad79ee3ecfc7e3f4862c1d8",
        "dependencies": ["num-traits", "rand 0.10.1"],
    })
    expected["package"].sort(key=lambda p: p["name"])
    if current != expected:
        raise ValueError("unreviewed post-measurement Cargo resolution change")


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
            # Preserve the measured lock verbatim. M14 adds only two already
            # pinned dependencies to distributions; it did not remeasure Fisher.
            path = (ARTIFACT.with_suffix(".lock") if relative == "crates/Cargo.lock"
                    else FISHER_ARCHIVE if relative == FISHER_SOURCE else ROOT / relative)
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(actual, expected, relative)

    def test_only_the_reviewed_native_nuts_module_extends_measured_fisher(self):
        measured, current = FISHER_ARCHIVE.read_text(), (ROOT / FISHER_SOURCE).read_text()
        validate_fisher_extension(measured, current)
        changed = current.replace("0.8", "0.9")
        self.assertNotEqual(changed, current)
        with self.assertRaises(ValueError):
            validate_fisher_extension(measured, changed)

    def test_only_reviewed_lock_changes_differ_from_measured_inputs(self):
        measured = tomllib.loads(ARTIFACT.with_suffix(".lock").read_text())
        current = tomllib.loads((ROOT / "crates/Cargo.lock").read_text())
        validate_lock_extension(measured, current)
        # Preserve rejection of an unrelated version/checksum/edge change.
        for field, value in (("version", "0.25.0"), ("checksum", "0" * 64),
                             ("dependencies", [])):
            changed = copy.deepcopy(current)
            next(p for p in changed["package"] if p["name"] == "faer")[field] = value
            with self.assertRaises(ValueError):
                validate_lock_extension(measured, changed)


if __name__ == "__main__":
    unittest.main()
