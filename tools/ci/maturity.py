"""Enforce Alea's public maturity, adoption, and guarantee inventory.

The ledger combines source-module classification with explicit public namespace
boundaries: every file containing an unrestricted ``pub`` declaration inherits
one profile, while named facades expose narrower adoption contracts.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import tomllib


MATURITY_ORDER = {f"S{level}": level for level in range(5)}
ADOPTIONS = {"reference", "recommended", "opt_in", "approximate", "not_exposed"}
GUARANTEES = {
    "component",
    "utility",
    "exact_target",
    "approximate_target",
    "exactness_pending",
    "contract_only",
    "mixed",
}
PUBLIC_DECLARATION = re.compile(
    r"(?m)^[ \t]*pub[ \t]+"
    r"(?:(?:async|unsafe|const)[ \t]+)*"
    r"(?:extern[ \t]+(?:\"[^\"]+\"[ \t]+)?)?"
    r"(?:fn|struct|enum|union|trait|type|mod|use|static|const|macro)[ \t]"
)
PUBLIC_MACRO = re.compile(r"(?m)^[ \t]*#\[macro_export\]")


def load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def discover_public_modules(root: Path) -> set[str]:
    modules = set()
    for path in root.glob("crates/alea-*/src/**/*.rs"):
        source = path.read_text()
        if PUBLIC_DECLARATION.search(source) or PUBLIC_MACRO.search(source):
            modules.add(path.relative_to(root).as_posix())
    return modules


def discover_features(root: Path) -> set[tuple[str, str]]:
    features = set()
    for manifest in root.glob("crates/alea-*/Cargo.toml"):
        document = load_toml(manifest)
        crate = document["package"]["name"]
        features.update((crate, name) for name in document.get("features", {}))
    return features


def discover_api_namespaces(root: Path) -> set[tuple[str, str]]:
    """Discover top-level ``reference`` facades from accepted API snapshots."""
    namespaces = set()
    snapshot_root = root / "tools/ci/public-api"
    for snapshot in snapshot_root.glob("alea-*.txt"):
        crate = snapshot.stem
        expected = f"pub mod {crate.replace('-', '_')}::reference"
        if expected in snapshot.read_text().splitlines():
            namespaces.add((crate, expected.removeprefix("pub mod ")))
    return namespaces


def discover_cli_commands(root: Path) -> set[tuple[str, str, str]]:
    commands = set()
    for manifest in root.glob("crates/alea-*/Cargo.toml"):
        document = load_toml(manifest)
        package = document["package"]
        crate = package["name"]
        bins = document.get("bin", [])
        for binary in bins:
            commands.add((crate, binary["name"], binary["path"]))
        if package.get("autobins", True):
            source = manifest.parent / "src"
            main = source / "main.rs"
            explicit_paths = {binary["path"] for binary in bins}
            if main.is_file() and "src/main.rs" not in explicit_paths:
                commands.add((crate, crate, "src/main.rs"))
            for path in source.glob("bin/*.rs"):
                relative = path.relative_to(manifest.parent).as_posix()
                commands.add((crate, path.stem, relative))
    return commands


def validate_profile(name: str, profile: dict, allow_release_candidates: bool) -> None:
    required = {"maturity", "adoption", "guarantee", "rationale"}
    if set(profile) != required:
        raise ValueError(f"profile {name!r} must contain exactly {sorted(required)}")
    maturity = profile["maturity"]
    adoption = profile["adoption"]
    guarantee = profile["guarantee"]
    if maturity not in MATURITY_ORDER:
        raise ValueError(f"profile {name!r} has unknown maturity {maturity!r}")
    if adoption not in ADOPTIONS:
        raise ValueError(f"profile {name!r} has unknown adoption {adoption!r}")
    if guarantee not in GUARANTEES:
        raise ValueError(f"profile {name!r} has unknown guarantee {guarantee!r}")
    if not isinstance(profile["rationale"], str) or not profile["rationale"].strip():
        raise ValueError(f"profile {name!r} lacks a rationale")
    if maturity == "S0" and adoption != "not_exposed":
        raise ValueError(f"design-only profile {name!r} must be not_exposed")
    if maturity == "S1" and adoption in {"reference", "recommended"}:
        raise ValueError(f"research profile {name!r} cannot be a reference or recommendation")
    if maturity in {"S3", "S4"} and not allow_release_candidates:
        raise ValueError(f"profile {name!r} claims release maturity before the release gate")
    if guarantee == "approximate_target" and adoption != "approximate":
        raise ValueError(f"approximate target profile {name!r} must use approximate adoption")
    if adoption == "approximate" and guarantee != "approximate_target":
        raise ValueError(f"approximate adoption profile {name!r} must name its target guarantee")
    if guarantee == "exactness_pending" and maturity != "S1":
        raise ValueError(f"exactness-pending profile {name!r} must remain S1")
    if guarantee == "contract_only" and maturity != "S0":
        raise ValueError(f"contract-only profile {name!r} must remain S0")


def validate_entries(entries: list[dict], profiles: dict, identity_fields: tuple[str, ...], label: str) -> None:
    required = set(identity_fields) | {"profile", "summary"}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"{label} entry must be an object: {entry!r}")
        permitted = required | {"promotion_gate"}
        if set(entry) - permitted or not required.issubset(entry):
            raise ValueError(f"{label} entry has an invalid shape: {entry}")
        if any(not isinstance(entry[field], str) or not entry[field] for field in identity_fields):
            raise ValueError(f"{label} entry has an invalid identity: {entry}")
        profile_name = entry["profile"]
        if profile_name not in profiles:
            raise ValueError(f"{label} entry references unknown profile {profile_name!r}")
        if not isinstance(entry["summary"], str) or not entry["summary"].strip():
            raise ValueError(f"{label} entry lacks a summary: {entry}")
        gate = entry.get("promotion_gate")
        maturity = MATURITY_ORDER[profiles[profile_name]["maturity"]]
        if maturity <= 1 and gate is None:
            raise ValueError(f"{label} research/design entry lacks a promotion gate: {entry}")
        if gate is not None:
            if not isinstance(gate, str) or not gate.strip():
                raise ValueError(f"{label} entry has an empty promotion gate: {entry}")
            if maturity > 1:
                raise ValueError(f"{label} entry crossed an open promotion gate: {entry}")
    identities = [tuple(entry[field] for field in identity_fields) for entry in entries]
    if len(identities) != len(set(identities)):
        raise ValueError(f"{label} inventory contains duplicate identities")
    if identities != sorted(identities):
        raise ValueError(f"{label} inventory must remain sorted for reviewable diffs")


def validate(root: Path, ledger: dict) -> dict[str, Counter]:
    expected_sections = {
        "schema_version",
        "policy",
        "profiles",
        "source_modules",
        "api_namespaces",
        "cargo_features",
        "cli_commands",
    }
    if set(ledger) != expected_sections:
        raise ValueError(f"maturity ledger must contain exactly {sorted(expected_sections)}")
    if ledger.get("schema_version") != 1:
        raise ValueError("unsupported maturity-ledger schema")
    policy = ledger.get("policy")
    if not isinstance(policy, dict) or set(policy) != {"allow_release_candidates"}:
        raise ValueError("ledger policy must explicitly control release-candidate claims")
    allow_release_candidates = policy["allow_release_candidates"]
    if not isinstance(allow_release_candidates, bool):
        raise ValueError("allow_release_candidates must be boolean")

    profiles = ledger.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("maturity ledger has no profiles")
    for name, profile in profiles.items():
        validate_profile(name, profile, allow_release_candidates)

    modules = ledger.get("source_modules")
    namespaces = ledger.get("api_namespaces")
    features = ledger.get("cargo_features")
    commands = ledger.get("cli_commands")
    if not all(
        isinstance(entries, list) for entries in (modules, namespaces, features, commands)
    ):
        raise ValueError("maturity ledger inventories must be lists")
    validate_entries(modules, profiles, ("path",), "source module")
    validate_entries(namespaces, profiles, ("crate", "path"), "API namespace")
    validate_entries(features, profiles, ("crate", "feature"), "Cargo feature")
    validate_entries(commands, profiles, ("crate", "name", "path"), "CLI command")

    recorded_modules = {entry["path"] for entry in modules}
    discovered_modules = discover_public_modules(root)
    if recorded_modules != discovered_modules:
        missing = sorted(discovered_modules - recorded_modules)
        stale = sorted(recorded_modules - discovered_modules)
        raise ValueError(f"public module inventory mismatch; missing={missing}, stale={stale}")

    recorded_namespaces = {(entry["crate"], entry["path"]) for entry in namespaces}
    discovered_namespaces = discover_api_namespaces(root)
    if recorded_namespaces != discovered_namespaces:
        missing = sorted(discovered_namespaces - recorded_namespaces)
        stale = sorted(recorded_namespaces - discovered_namespaces)
        raise ValueError(f"API namespace inventory mismatch; missing={missing}, stale={stale}")
    for crate, path in recorded_namespaces:
        expected = f"{crate.replace('-', '_')}::reference"
        if path != expected:
            raise ValueError(f"API namespace {path!r} does not match crate {crate!r}")

    recorded_features = {(entry["crate"], entry["feature"]) for entry in features}
    discovered_features = discover_features(root)
    if recorded_features != discovered_features:
        missing = sorted(discovered_features - recorded_features)
        stale = sorted(recorded_features - discovered_features)
        raise ValueError(f"Cargo feature inventory mismatch; missing={missing}, stale={stale}")

    recorded_commands = {(entry["crate"], entry["name"], entry["path"]) for entry in commands}
    discovered_commands = discover_cli_commands(root)
    if recorded_commands != discovered_commands:
        missing = sorted(discovered_commands - recorded_commands)
        stale = sorted(recorded_commands - discovered_commands)
        raise ValueError(f"CLI command inventory mismatch; missing={missing}, stale={stale}")
    for crate, name, path in recorded_commands:
        command_path = root / "crates" / crate / path
        if not command_path.is_file():
            raise ValueError(f"CLI command {name!r} has missing source {command_path}")

    return {
        "modules": Counter(profiles[entry["profile"]]["maturity"] for entry in modules),
        "namespaces": Counter(
            profiles[entry["profile"]]["maturity"] for entry in namespaces
        ),
        "features": Counter(profiles[entry["profile"]]["maturity"] for entry in features),
        "commands": Counter(profiles[entry["profile"]]["maturity"] for entry in commands),
    }


def check(root: Path) -> None:
    ledger = json.loads((root / "tools/ci/maturity-ledger.json").read_text())
    counts = validate(root, ledger)
    print(
        "maturity inventory: "
        f"modules={dict(sorted(counts['modules'].items()))}; "
        f"namespaces={dict(sorted(counts['namespaces'].items()))}; "
        f"features={dict(sorted(counts['features'].items()))}; "
        f"commands={dict(sorted(counts['commands'].items()))}"
    )


if __name__ == "__main__":
    check(Path(__file__).resolve().parents[2])
