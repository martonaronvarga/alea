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
MATURITY_NAMESPACES = {"approximate", "prelude"}
ROOT_ITEM = re.compile(
    r"^(?:#\[[^\]]+\] )?pub "
    r"(?:fn|struct|enum|union|trait|type|use|static|const|macro) "
    r"([a-z_]+(?:::[A-Za-z_][A-Za-z0-9_]*)+)"
)


def load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def discover_public_modules(root: Path) -> set[str]:
    modules = set()
    for path in root.glob("crates/*/src/**/*.rs"):
        source = path.read_text()
        if PUBLIC_DECLARATION.search(source) or PUBLIC_MACRO.search(source):
            modules.add(path.relative_to(root).as_posix())
    return modules


def discover_features(root: Path) -> set[tuple[str, str]]:
    features = set()
    for manifest in root.glob("crates/*/Cargo.toml"):
        document = load_toml(manifest)
        crate = document["package"]["name"]
        features.update((crate, name) for name in document.get("features", {}))
    return features


def discover_api_namespaces(root: Path) -> set[tuple[str, str]]:
    """Discover named maturity facades from accepted API snapshots."""
    namespaces = set()
    snapshot_root = root / "tools/ci/public-api"
    for snapshot in snapshot_root.glob("alea*.txt"):
        crate = snapshot.stem.split("+", 1)[0]
        crate_path = crate.replace("-", "_")
        lines = set(snapshot.read_text().splitlines())
        for namespace in MATURITY_NAMESPACES:
            expected = f"pub mod {crate_path}::{namespace}"
            if expected in lines:
                namespaces.add((crate, expected.removeprefix("pub mod ")))
    return namespaces


def discover_root_items(root: Path) -> set[tuple[str, str]]:
    """Discover non-module items exported directly from each crate root."""
    items = set()
    snapshot_root = root / "tools/ci/public-api"
    for snapshot in snapshot_root.glob("alea*.txt"):
        crate = snapshot.stem.split("+", 1)[0]
        crate_path = crate.replace("-", "_")
        for line in snapshot.read_text().splitlines():
            match = ROOT_ITEM.match(line)
            if match is None:
                continue
            segments = match.group(1).split("::")
            if len(segments) == 2 and segments[0] == crate_path:
                items.add((crate, segments[1]))
    return items


def discover_api_items(root: Path) -> set[tuple[str, str]]:
    """Every rendered declaration/impl is individually classified, including aliases."""
    return {
        (snapshot.stem.split("+", 1)[0], line)
        for snapshot in (root / "tools/ci/public-api").glob("alea*.txt")
        for line in snapshot.read_text().splitlines()
    }


def discover_cli_commands(root: Path) -> set[tuple[str, str, str]]:
    commands = set()
    for manifest in root.glob("crates/*/Cargo.toml"):
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
    if maturity == "S4":
        raise ValueError(f"profile {name!r} claims stable publication before a stable-release policy review")
    if maturity == "S3" and not allow_release_candidates:
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
        "root_items",
        "api_items",
        "cargo_features",
        "cli_commands",
    }
    if set(ledger) != expected_sections:
        raise ValueError(f"maturity ledger must contain exactly {sorted(expected_sections)}")
    if ledger.get("schema_version") != 3:
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
    root_items = ledger.get("root_items")
    api_items = ledger.get("api_items")
    features = ledger.get("cargo_features")
    commands = ledger.get("cli_commands")
    if not all(
        isinstance(entries, list)
        for entries in (modules, namespaces, root_items, api_items, features, commands)
    ):
        raise ValueError("maturity ledger inventories must be lists")
    validate_entries(modules, profiles, ("path",), "source module")
    validate_entries(namespaces, profiles, ("crate", "path"), "API namespace")
    validate_entries(root_items, profiles, ("crate", "item"), "root API item")
    # Compact item records still require an explicit profile and research gate.
    expanded = [dict(entry, summary="item-level public API classification") for entry in api_items]
    validate_entries(expanded, profiles, ("crate", "item"), "public API item")
    recorded_items = {(entry["crate"], entry["item"]) for entry in api_items}
    if recorded_items != discover_api_items(root):
        missing = sorted(discover_api_items(root) - recorded_items)
        stale = sorted(recorded_items - discover_api_items(root))
        raise ValueError(f"item-level API inventory mismatch; missing={missing[:5]}, stale={stale[:5]}")
    validate_entries(features, profiles, ("crate", "feature"), "Cargo feature")
    validate_entries(commands, profiles, ("crate", "name", "path"), "CLI command")

    # Release qualification belongs to the frozen artifact review. Import paths,
    # including convenience preludes and deprecated aliases, are not releases.
    for entries in (modules, root_items, api_items, features, commands, namespaces):
        if any(profiles[e["profile"]]["maturity"] == "S3" for e in entries):
            raise ValueError("release qualification is recorded separately from import paths")

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
    namespace_entries = {
        (entry["crate"], entry["path"]): entry for entry in namespaces
    }
    for crate, path in recorded_namespaces:
        crate_path, separator, namespace = path.partition("::")
        if (
            not separator
            or crate_path != crate.replace("-", "_")
            or namespace not in MATURITY_NAMESPACES
        ):
            raise ValueError(f"API namespace {path!r} does not match crate {crate!r}")
        profile = profiles[namespace_entries[(crate, path)]["profile"]]
        if namespace == "prelude" and profile["adoption"] != "reference":
            raise ValueError(f"baseline import namespace {path!r} must use reference adoption")
        if namespace == "approximate" and (
            profile["adoption"] != "approximate"
            or profile["guarantee"] != "approximate_target"
        ):
            raise ValueError(
                f"approximate namespace {path!r} must expose an approximate target"
            )

    recorded_root_items = {
        (entry["crate"], entry["item"]) for entry in root_items
    }
    discovered_root_items = discover_root_items(root)
    if recorded_root_items != discovered_root_items:
        missing = sorted(discovered_root_items - recorded_root_items)
        stale = sorted(recorded_root_items - discovered_root_items)
        raise ValueError(f"root API item inventory mismatch; missing={missing}, stale={stale}")

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
        "items": Counter(profiles[entry["profile"]]["maturity"] for entry in api_items),
        "modules": Counter(profiles[entry["profile"]]["maturity"] for entry in modules),
        "namespaces": Counter(
            profiles[entry["profile"]]["maturity"] for entry in namespaces
        ),
        "root_items": Counter(
            profiles[entry["profile"]]["maturity"] for entry in root_items
        ),
        "features": Counter(profiles[entry["profile"]]["maturity"] for entry in features),
        "commands": Counter(profiles[entry["profile"]]["maturity"] for entry in commands),
    }


def check(root: Path) -> None:
    ledger = json.loads((root / "tools/ci/maturity-ledger.json").read_text())
    counts = validate(root, ledger)
    print(
        "maturity inventory: "
        f"items={dict(sorted(counts['items'].items()))}; "
        f"modules={dict(sorted(counts['modules'].items()))}; "
        f"namespaces={dict(sorted(counts['namespaces'].items()))}; "
        f"root_items={dict(sorted(counts['root_items'].items()))}; "
        f"features={dict(sorted(counts['features'].items()))}; "
        f"commands={dict(sorted(counts['commands'].items()))}"
    )


if __name__ == "__main__":
    check(Path(__file__).resolve().parents[2])
