"""Regression checks for the executed, source-bound M14 evidence bundle."""
import csv
import hashlib
import json
from pathlib import Path
import unittest
import tomllib
import copy
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "docs/benchmarks/m14"
RELEASE_SOURCES = {"crates/Cargo.toml", "crates/alea-core/src/transform/layout.rs"}
ARCHIVED_SOURCES = RELEASE_SOURCES | {
    "crates/Cargo.lock", "crates/alea-core/src/model/assembly.rs",
    "crates/alea-distributions/src/modeling.rs",
}


def validate(root, evidence):
    if evidence["schema_version"] != 1:
        raise ValueError("unsupported M14 evidence schema")
    for section in ("sources", "artifacts"):
        for name, expected in evidence[section].items():
            # Historical measurements bind their original bytes. Later model
            # source export has its own executed tests, not M14 calibration.
            path = (root / "docs/benchmarks/m14/downstream-conformance.lock"
                    if section == "sources" and name == "tools/downstream-conformance/Cargo.lock"
                    else root / "docs/benchmarks/m14/measured-sources" / name
                    if section == "sources" and name in ARCHIVED_SOURCES
                    else root / name)
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"stale M14 {section}: {name}")
            if section == "sources" and name in RELEASE_SOURCES:
                validate_release_source(name, path.read_text(), (root / name).read_text())
    measured = tomllib.loads((root / "docs/benchmarks/m14/downstream-conformance.lock").read_text())
    current = tomllib.loads((root / "tools/downstream-conformance/Cargo.lock").read_text())
    validate_lock_review(measured, current)
    calibration = json.loads((root / "docs/benchmarks/m14/calibration.json").read_text())
    rows = calibration["hierarchical_sbc"]
    expected = {(name, i) for name in ("hierarchical logistic", "hierarchical DDM") for i in range(4)}
    if {(row["model"], row["coordinate"]) for row in rows} != expected or len(rows) != 8:
        raise ValueError("hierarchical calibration inventory mismatch")
    for row in rows:
        bins = row["bins"]
        if len(bins) != 4 or sum(bins) != 32 or any(n < 0 for n in bins):
            raise ValueError("invalid hierarchical calibration ranks")
        chi = sum((n - 8) ** 2 / 8 for n in bins)
        if chi != row["chi_square"] or chi >= calibration["chi_square_limit"]:
            raise ValueError("failed or altered hierarchical calibration gate")
    normal = calibration["normal_sbc"]
    if sum(normal["bins"]) != 64 or normal["chi_square"] >= calibration["chi_square_limit"]:
        raise ValueError("failed normal calibration gate")
    with (root / "docs/benchmarks/m14/distribution-cost.csv").open() as handle:
        costs = list(csv.DictReader(handle))
    names = {"normal", "student_t", "lognormal", "exponential", "gamma", "beta", "bernoulli", "binomial", "poisson", "mvn2", "dirichlet3", "categorical", "multinomial", "lkj2", "mixture"}
    if {row["law"] for row in costs} != names or len(costs) != len(names):
        raise ValueError("distribution cost inventory mismatch")
    if any(int(row["iterations"]) != 200_000 or float(row["nanoseconds_per_value_and_score"]) <= 0 for row in costs):
        raise ValueError("invalid cost record")


def validate_lock_review(measured, current):
    expected = copy.deepcopy(measured)
    review = json.loads((ARTIFACTS / "reviewed-performance-lock.json").read_text())
    if type(review["schema_version"]) is not int or review["schema_version"] != 1:
        raise ValueError("unsupported post-M14 lock review")
    packages = {p["name"]: p for p in expected["package"]}
    for name, additions in review["dependency_additions"].items():
        packages[name]["dependencies"] = sorted(packages[name]["dependencies"] + additions)
    expected["package"].extend(review["added_packages"])
    expected["package"].sort(key=lambda p: p["name"])
    if current != expected:
        raise ValueError("post-M14 lock changed beyond reviewed runtime/performance dependencies")


def validate_release_source(name, measured, current):
    """Preserve measured bytes; admit reviewed metadata and pinned performance edges."""
    if name == "crates/Cargo.toml":
        expected = copy.deepcopy(tomllib.loads(measured))
        expected["workspace"]["package"].update(
            repository="https://github.com/martonaronvarga/alea",
            homepage="https://github.com/martonaronvarga/alea",
            license="MIT OR Apache-2.0",
        )
        for member in expected["workspace"]["members"]:
            expected["workspace"]["dependencies"][member]["version"] = "0.1.0"
        expected["workspace"]["dependencies"].update(
            pulp={"version": "=0.22.3", "default-features": False, "features": ["std", "x86-v3"]},
            rand_distr="=0.6.0",
        )
        if tomllib.loads(current) != expected:
            raise ValueError("unreviewed post-M14 manifest change")
    else:
        def code(source):
            stripped = "\n".join(line for line in source.splitlines()
                                 if not line.lstrip().startswith("///")
                                 and line.strip() != "#![deny(missing_docs)]")
            # Preserve tokens, strings and semantics while ignoring rustfmt's
            # line wrapping after documentation expands inline fields.
            return subprocess.check_output(
                ["rustfmt", "--edition", "2024", "--emit", "stdout"],
                input=stripped.encode(),
            )
        if code(measured) != code(current):
            raise ValueError("unreviewed post-M14 layout implementation change")


class EvidenceTests(unittest.TestCase):
    def test_post_m14_release_review_rejects_numerical_or_dependency_changes(self):
        for name in RELEASE_SOURCES:
            measured = (ARTIFACTS / "measured-sources" / name).read_text()
            current = (ROOT / name).read_text()
            changed = (current.replace('rand = "0.10.1"', 'rand = "0.11"')
                       if name.endswith(".toml") else current.replace("pub fn transform", "pub fn altered"))
            self.assertNotEqual(changed, current)
            with self.assertRaises(ValueError):
                validate_release_source(name, measured, changed)

    def test_executed_bundle_sources_and_artifacts_are_preserved(self):
        validate(ROOT, json.loads((ARTIFACTS / "evidence.json").read_text()))

    def test_unreviewed_post_m14_package_drift_is_rejected(self):
        measured = tomllib.loads((ARTIFACTS / "downstream-conformance.lock").read_text())
        current = tomllib.loads((ROOT / "tools/downstream-conformance/Cargo.lock").read_text())
        current["package"][0]["version"] = "999.0.0"
        with self.assertRaises(ValueError):
            validate_lock_review(measured, current)

    def test_altered_source_digest_is_rejected(self):
        evidence = json.loads((ARTIFACTS / "evidence.json").read_text())
        first = next(iter(evidence["sources"]))
        evidence["sources"][first] = "0" * 64
        with self.assertRaises(ValueError):
            validate(ROOT, evidence)


if __name__ == "__main__":
    unittest.main()
