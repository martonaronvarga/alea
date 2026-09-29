"""Guard migration destinations and explicit test mappings without Git history.

This is a source-inventory guard, NOT a substitute for executing Cargo/Miri tests.
Dormant research tests are deliberately counted separately from compiled coverage.
"""
import hashlib
import json
from pathlib import Path
import re


# Immutable baseline cardinalities: changing a disposition is allowed, silently
# deleting old files/tests from the ledger is not. New tests belong to destinations.
BASELINE_FILES = 118
BASELINE_TESTS = 188
BASELINE_FEATURES = frozenset("""
aligned-storage euclidean-metrics polar-normal-rng gaussian-density transactional-targets
parameter-transforms composed-models enzyme-and-opaque-vjp wiener4 wiener5 wiener7
wiener-numerics-and-batches cubature-boundary hamiltonian-phases fixed-hmc
derivative-free-rwmh dual-averaging-warmup aligned-draw-collection classical-diagnostics
streaming-runner latent-ddm grid-and-linear-spline ddm-cse-likelihood-research
affine-batch-and-observation-research wiener-surrogate-placeholder foreign-layout-research
particle-contracts nuts-and-policy-placeholders unused-extension-contracts
profiling-and-reference-tools research-data-analysis
""".split())


def require_test(root, replacement):
    path = root / replacement["path"]
    if not path.is_file():
        raise ValueError(f"missing test file: {replacement['path']}")
    content = path.read_text()
    if not re.search(r"\bfn\s+" + re.escape(replacement["name"]) + r"\s*\(", content):
        raise ValueError(f"missing replacement test: {replacement}")


def validate(root, ledger, inventory):
    sources = [entry["source"] for entry in ledger["files"]]
    if len(sources) != BASELINE_FILES or len(set(sources)) != BASELINE_FILES:
        raise ValueError("baseline source inventory is incomplete or duplicated")
    old_tests = [(entry["source"], test["old"]) for entry in ledger["files"] for test in entry.get("tests", [])]
    if len(old_tests) != BASELINE_TESTS or len(set(old_tests)) != BASELINE_TESTS:
        raise ValueError("baseline test inventory is incomplete or duplicated")
    counts = {}
    for entry in ledger["files"]:
        for target in entry["targets"]:
            # Local roadmap/research notes are intentionally excluded from commits.
            if target.startswith("docs/"):
                continue
            if not (root / target).exists():
                raise ValueError(f"missing migration destination: {target}")
        if "preserved_git_blob" in entry:
            content = (root / entry["targets"][0]).read_bytes()
            blob = b"blob " + str(len(content)).encode() + b"\0" + content
            if hashlib.sha1(blob).hexdigest() != entry["preserved_git_blob"]:
                raise ValueError(f"research source changed: {entry['source']}; update its disposition explicitly")
        for test in entry.get("tests", []):
            status = test["disposition"]
            if status not in {"retained", "replaced_contract", "dormant_preserved", "removed_scaffolding"}:
                raise ValueError(f"unknown test disposition: {status}")
            if status == "removed_scaffolding":
                if (entry["source"], test["old"]) != ("crates/data_prep/src/lib.rs", "it_works"):
                    raise ValueError("a behavioral test cannot be retired as scaffolding")
            elif not test.get("replacements"):
                raise ValueError(f"test has no replacement: {entry['source']}::{test['old']}")
            counts[status] = counts.get(status, 0) + 1
            for replacement in test.get("replacements", []):
                require_test(root, replacement)
    ids = [feature["id"] for feature in inventory["features"]]
    if len(ids) != len(set(ids)) or not BASELINE_FEATURES.issubset(ids):
        raise ValueError("feature inventory is incomplete or duplicated")
    for feature in inventory["features"]:
        if not feature["sources"] or not set(feature["sources"]).issubset(sources):
            raise ValueError(f"feature has unregistered source: {feature['id']}")
        if not feature["contract"] or not feature["targets"]:
            raise ValueError(f"feature lacks a contract or destination: {feature['id']}")
        for path in feature["targets"]:
            if not (root / path).is_file():
                raise ValueError(f"missing feature destination: {path}")
        status = feature["status"]
        if status in {"migrated", "ported_from_research"}:
            if not feature["checks"] or any(not test["path"].startswith("crates/alea-") for test in feature["checks"]):
                raise ValueError(f"implemented feature lacks canonical regression tests: {feature['id']}")
        elif status in {"preserved_research", "placeholder_only"}:
            if not feature.get("remaining"):
                raise ValueError(f"unfinished feature lacks explicit remaining work: {feature['id']}")
        elif status != "tooling":
            raise ValueError(f"unknown feature status: {status}")
        for test in feature["checks"]:
            require_test(root, test)
    return counts


def check(root):
    ledger = json.loads((root / "tools/ci/migration-ledger.json").read_text())
    inventory = json.loads((root / "tools/ci/migration-features.json").read_text())
    counts = validate(root, ledger, inventory)
    print(f"migration inventory: {len(ledger['files'])} paths; test dispositions: {counts}")
    print(f"feature inventory: {len(inventory['features'])} capability groups with explicit status and obligations")


if __name__ == "__main__":
    check(Path(__file__).resolve().parents[2])
