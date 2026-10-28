"""Validate the implemented-map law inventory for public MCMC transitions."""

from __future__ import annotations

import json
from pathlib import Path
import re


GUARANTEES = {"exact_target", "approximate_target", "exactness_pending"}
MATURITIES = {"S1", "S2"}
LAW_FIELDS = (
    "refresh",
    "support",
    "reversibility",
    "volume_or_density",
    "selection",
    "failure",
)
LAW_STATUSES = {"closed", "open", "not_applicable", "omitted_by_design"}
CHAIN_IMPL = re.compile(
    r"MarkovChain for "
    r"(alea_mcmc::(?:[a-z_][a-z0-9_]*::)+[A-Z][A-Za-z0-9_]*)"
)


def discover_transitions(root: Path) -> set[str]:
    snapshot = root / "tools/ci/public-api/alea-mcmc.txt"
    return set(CHAIN_IMPL.findall(snapshot.read_text()))


def discover_proposals(root: Path) -> set[str]:
    """Sealed integrators and public phase/batch proposal entry points."""
    text = (root / "tools/ci/public-api/alea-mcmc.txt").read_text()
    integrators = set(re.findall(r"SymmetricSplitting for (alea_mcmc::integrator::[A-Za-z0-9_]+)", text))
    methods = set(re.findall(r"pub fn (alea_mcmc::(?:hamiltonian::Phase|batch::BatchedLeapfrog))<[^\n]*?::(step_with|step|advance)[<(]", text))
    return integrators | {f"{owner}::{method}" for owner, method in methods}


def validate_laws(entry: dict) -> None:
    laws = entry["laws"]
    if not isinstance(laws, dict) or tuple(laws) != LAW_FIELDS:
        raise ValueError(f"{entry['path']}: laws must appear in canonical order {LAW_FIELDS}")
    for name, law in laws.items():
        if not isinstance(law, dict) or set(law) != {"status", "claim"}:
            raise ValueError(f"{entry['path']}: invalid {name} law")
        if law["status"] not in LAW_STATUSES:
            raise ValueError(f"{entry['path']}: unknown {name} status {law['status']!r}")
        if not isinstance(law["claim"], str) or not law["claim"].strip():
            raise ValueError(f"{entry['path']}: empty {name} claim")

    statuses = {law["status"] for law in laws.values()}
    guarantee = entry["guarantee"]
    if guarantee == "exact_target" and statuses - {"closed", "not_applicable"}:
        raise ValueError(f"{entry['path']}: exact-target transition has an open/omitted law")
    if guarantee == "exactness_pending" and "open" not in statuses:
        raise ValueError(f"{entry['path']}: exactness-pending transition has no open law")
    if guarantee == "approximate_target":
        if laws["selection"]["status"] != "omitted_by_design":
            raise ValueError(f"{entry['path']}: approximate transition must expose omitted correction")
        if "open" in statuses:
            raise ValueError(f"{entry['path']}: declared approximation cannot hide an open law")


def validate_evidence(root: Path, entry: dict) -> None:
    evidence = entry["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise ValueError(f"{entry['path']}: transition has no executable evidence")
    identities = []
    for item in evidence:
        if not isinstance(item, dict) or set(item) != {"path", "symbol"}:
            raise ValueError(f"{entry['path']}: invalid evidence record {item!r}")
        path = item["path"]
        symbol = item["symbol"]
        if not isinstance(path, str) or not isinstance(symbol, str) or not path or not symbol:
            raise ValueError(f"{entry['path']}: empty evidence identity")
        source = root / path
        if not source.is_file():
            raise ValueError(f"{entry['path']}: missing evidence file {path}")
        if symbol not in source.read_text():
            raise ValueError(f"{entry['path']}: missing evidence symbol {symbol!r} in {path}")
        identities.append((path, symbol))
    if identities != sorted(identities) or len(identities) != len(set(identities)):
        raise ValueError(f"{entry['path']}: evidence must be unique and sorted")


def validate(root: Path, ledger: dict) -> dict[str, int]:
    if set(ledger) != {"schema_version", "transitions", "proposals", "feature_effects"} or ledger.get("schema_version") != 2:
        raise ValueError("unsupported exactness-ledger schema")
    transitions = ledger["transitions"]
    if not isinstance(transitions, list):
        raise ValueError("exactness transitions must be a list")

    maturity_ledger = json.loads((root / "tools/ci/maturity-ledger.json").read_text())
    profiles = maturity_ledger["profiles"]
    modules = {entry["path"]: entry for entry in maturity_ledger["source_modules"]}
    required = {
        "path",
        "source",
        "guarantee",
        "maturity",
        "laws",
        "evidence",
    }
    recorded = []
    counts = {guarantee: 0 for guarantee in sorted(GUARANTEES)}
    for entry in transitions:
        if not isinstance(entry, dict):
            raise ValueError(f"transition entry must be an object: {entry!r}")
        permitted = required | {"promotion_gate"}
        if set(entry) - permitted or not required.issubset(entry):
            raise ValueError(f"transition entry has invalid shape: {entry}")
        if any(
            not isinstance(entry[field], str) or not entry[field]
            for field in ("path", "source", "guarantee", "maturity")
        ):
            raise ValueError(f"transition entry has an invalid identity: {entry}")
        if entry["guarantee"] not in GUARANTEES:
            raise ValueError(f"{entry['path']}: unknown guarantee")
        if entry["maturity"] not in MATURITIES:
            raise ValueError(f"{entry['path']}: unsupported transition maturity")
        source_entry = modules.get(entry["source"])
        if source_entry is None:
            raise ValueError(f"{entry['path']}: source is absent from maturity inventory")
        inherited = profiles[source_entry["profile"]]["maturity"]
        if entry["maturity"] != inherited:
            raise ValueError(
                f"{entry['path']}: maturity {entry['maturity']} disagrees with {inherited} source"
            )
        gate = entry.get("promotion_gate")
        if entry["maturity"] == "S1":
            if not isinstance(gate, str) or not gate.strip():
                raise ValueError(f"{entry['path']}: research transition lacks promotion gate")
        elif gate is not None:
            raise ValueError(f"{entry['path']}: validated transition retains an open gate")
        validate_laws(entry)
        validate_evidence(root, entry)
        recorded.append(entry["path"])
        counts[entry["guarantee"]] += 1

    if recorded != sorted(recorded) or len(recorded) != len(set(recorded)):
        raise ValueError("transition inventory must be unique and sorted")
    discovered = discover_transitions(root)
    recorded_set = set(recorded)
    if recorded_set != discovered:
        missing = sorted(discovered - recorded_set)
        stale = sorted(recorded_set - discovered)
        raise ValueError(f"transition inventory mismatch; missing={missing}, stale={stale}")
    proposals = ledger["proposals"]
    if {entry["path"] for entry in proposals} != discover_proposals(root):
        raise ValueError("standalone proposal/integrator inventory mismatch")
    if len(proposals) != len({entry["path"] for entry in proposals}):
        raise ValueError("duplicate standalone proposal law")
    for entry in proposals:
        if entry["guarantee"] != "proposal_only":
            raise ValueError("standalone integration cannot claim a corrected transition")
        validate_laws(entry)
        validate_evidence(root, entry)
        if entry["laws"]["selection"]["status"] != "not_applicable":
            raise ValueError("standalone proposal must require its caller's correction")
    features = ledger["feature_effects"]
    import tomllib
    cargo = tomllib.loads((root / "crates/alea-mcmc/Cargo.toml").read_text())
    if set(features) != set(cargo["features"]):
        raise ValueError("feature-specific exactness inventory mismatch")
    for name, entry in features.items():
        if not entry.get("law") or not entry.get("evidence"):
            raise ValueError(f"{name}: missing feature map law/evidence")
        validate_evidence(root, dict(entry, path=name))
    # Feature-only new kernels cannot hide outside the default API inventory.
    implemented = set()
    for path in (root / "crates/alea-mcmc/src").rglob("*.rs"):
        implemented.update(re.findall(r"impl[^;{]*?MarkovChain\s+for\s+(\w+)", path.read_text()))
    if implemented != {path.rsplit("::", 1)[1] for path in discovered}:
        raise ValueError("feature/source MarkovChain inventory differs from audited transitions")
    return counts


def check(root: Path) -> None:
    ledger = json.loads((root / "tools/ci/exactness-ledger.json").read_text())
    counts = validate(root, ledger)
    print(f"exactness inventory: transitions={len(ledger['transitions'])}; guarantees={counts}")


if __name__ == "__main__":
    check(Path(__file__).resolve().parents[2])
