"""Validate and reproduce Alea's accepted public API snapshot matrix."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib


SCHEMA_VERSION = 3
TOOL = "cargo-public-api 0.51.0"
POLICY = {
    "default_features": False,
    "omit": ["blanket-impls", "auto-trait-impls"],
    "parameter_names": True,
}
FEATURE_VARIANTS = {
    "alea-core+tracing": {"package": "alea-core", "features": ["alea-core/tracing"], "same_as": "alea-core"},
    "alea-math+tracing": {"package": "alea-math", "features": ["alea-math/tracing"], "same_as": "alea-math"},
    "alea-mcmc+tracing": {"package": "alea-mcmc", "features": ["alea-mcmc/tracing"], "same_as": "alea-mcmc"},
    "alea-distributions+tracing": {"package": "alea-distributions", "features": ["alea-distributions/tracing"], "same_as": "alea-distributions"},
    "alea-autodiff+tracing": {"package": "alea-autodiff", "features": ["alea-autodiff/tracing"], "same_as": "alea-autodiff"},
    "alea-ffi+tracing": {"package": "alea-ffi", "features": ["alea-ffi/tracing"], "same_as": "alea-ffi"},
    "alea+tracing": {"package": "alea", "features": ["alea/tracing"], "same_as": "alea"},
    "alea-runtime+tracing": {"package": "alea-runtime", "features": ["alea-runtime/tracing"], "same_as": "alea-runtime"},
    "alea+simd": {"package": "alea", "features": ["alea/simd"], "same_as": "alea"},
    "alea-distributions+branch-hints": {
        "package": "alea-distributions",
        "features": ["alea-distributions/branch-hints"],
        "same_as": "alea-distributions",
    },
    "alea-autodiff+std-autodiff": {
        "package": "alea-autodiff",
        "features": ["alea-autodiff/std-autodiff"],
        "same_as": "alea-autodiff",
    },
    "alea-distributions+simd": {
        "package": "alea-distributions",
        "features": ["alea-distributions/simd"],
        "same_as": "alea-distributions",
    },
    "alea-math+simd": {
        "package": "alea-math",
        "features": ["alea-math/simd"],
        "same_as": "alea-math",
    },
    "alea-mcmc+simd": {
        "package": "alea-mcmc",
        "features": ["alea-mcmc/simd"],
        "same_as": "alea-mcmc",
    },
}
FEATURE_SNAPSHOTS = {
    "alea+analysis": {"package": "alea", "features": ["alea/analysis"]},
    "alea-runtime+analysis": {"package": "alea-runtime", "features": ["alea-runtime/analysis"]},
    "alea+distributions": {"package": "alea", "features": ["alea/distributions"]},
    "alea+autodiff": {"package": "alea", "features": ["alea/autodiff"]},
    "alea+faer": {"package": "alea", "features": ["alea/faer"]},
    "alea+distributions+autodiff+faer": {"package": "alea", "features": ["alea/distributions", "alea/autodiff", "alea/faer"]},
    "alea-distributions+faer": {"package": "alea-distributions", "features": ["alea-distributions/faer"]},
    "alea-math+faer": {"package": "alea-math", "features": ["alea-math/faer"]},
    "alea-mcmc+faer": {"package": "alea-mcmc", "features": ["alea-mcmc/faer"]},
}


def load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def discover_libraries(root: Path) -> list[str]:
    workspace = load_toml(root / "crates/Cargo.toml")
    libraries = []
    for member in workspace["workspace"]["members"]:
        crate = root / "crates" / member
        if (crate / "src/lib.rs").is_file():
            libraries.append(load_toml(crate / "Cargo.toml")["package"]["name"])
    return sorted(libraries)


def normalize_snapshot(output: str) -> str:
    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    if not lines:
        raise ValueError("cargo-public-api produced an empty snapshot")
    return "\n".join(lines) + "\n"


def command_output(command: list[str], root: Path) -> str:
    result = subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"command failed ({' '.join(command)}): {detail}")
    return result.stdout.strip()


def installed_versions(root: Path) -> tuple[str, str]:
    tool = command_output(["cargo", "public-api", "--version"], root)
    rustc = command_output(["rustc", "--version"], root)
    return tool, rustc


def render_snapshot(root: Path, package: str, features: tuple[str, ...] = ()) -> str:
    command = [
        "cargo",
        "public-api",
        "--manifest-path",
        "crates/Cargo.toml",
        "--package",
        package,
        "--no-default-features",
        "--omit",
        "blanket-impls",
        "--omit",
        "auto-trait-impls",
        "--color",
        "never",
    ]
    if features:
        command.extend(["--features", ",".join(features)])
    return normalize_snapshot(command_output(command, root))


def digest(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def load_manifest(snapshot_dir: Path) -> dict:
    try:
        return json.loads((snapshot_dir / "manifest.json").read_text())
    except FileNotFoundError as error:
        raise ValueError("public API snapshot manifest is missing") from error


def validate(root: Path, snapshot_dir: Path, manifest: dict) -> None:
    required = {"schema_version", "tool", "rustc", "policy", "packages", "variants", "feature_packages"}
    if set(manifest) != required:
        raise ValueError(f"public API manifest must contain exactly {sorted(required)}")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported public API snapshot schema")
    if manifest["tool"] != TOOL:
        raise ValueError(f"public API snapshot must use {TOOL}")
    if manifest["policy"] != POLICY:
        raise ValueError("public API snapshot policy changed without a schema update")
    if not isinstance(manifest["rustc"], str) or not manifest["rustc"].startswith("rustc "):
        raise ValueError("public API snapshot lacks its rustc provenance")

    libraries = discover_libraries(root)
    packages = manifest["packages"]
    if not isinstance(packages, dict) or sorted(packages) != libraries:
        raise ValueError(
            "public API package inventory mismatch; "
            f"expected={libraries}, recorded={sorted(packages) if isinstance(packages, dict) else packages}"
        )

    expected_files = {"manifest.json"}
    for package in libraries:
        entry = packages[package]
        if set(entry) != {"file", "items", "sha256"}:
            raise ValueError(f"invalid public API manifest entry for {package}")
        expected_file = f"{package}.txt"
        if entry["file"] != expected_file:
            raise ValueError(f"noncanonical public API filename for {package}")
        path = snapshot_dir / expected_file
        try:
            content = path.read_text()
        except FileNotFoundError as error:
            raise ValueError(f"public API snapshot is missing for {package}") from error
        if normalize_snapshot(content) != content:
            raise ValueError(f"public API snapshot is not normalized for {package}")
        if entry["items"] != len(content.splitlines()):
            raise ValueError(f"public API item count mismatch for {package}")
        if entry["sha256"] != digest(content):
            raise ValueError(f"public API digest mismatch for {package}")
        expected_files.add(expected_file)

    extras = manifest["feature_packages"]
    if set(extras) != set(FEATURE_SNAPSHOTS):
        raise ValueError("feature-specific public API inventory mismatch")
    for name, entry in extras.items():
        if set(entry) != {"file", "items", "sha256"} or entry["file"] != f"{name}.txt":
            raise ValueError("invalid feature-specific API record")
        content = (snapshot_dir / entry["file"]).read_text()
        if normalize_snapshot(content) != content or digest(content) != entry["sha256"] or len(content.splitlines()) != entry["items"]:
            raise ValueError(f"{name}: feature-specific API digest/count mismatch")
        expected_files.add(entry["file"])

    variants = manifest["variants"]
    if variants != FEATURE_VARIANTS:
        raise ValueError(
            "public API feature matrix changed without a schema/policy review; "
            f"expected={FEATURE_VARIANTS}, recorded={variants}"
        )

    actual_files = {path.name for path in snapshot_dir.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise ValueError(
            "public API snapshot directory mismatch; "
            f"missing={sorted(expected_files - actual_files)}, "
            f"stale={sorted(actual_files - expected_files)}"
        )


def build_manifest(libraries: list[str], snapshots: dict[str, str], rustc: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "rustc": rustc,
        "policy": POLICY,
        "packages": {
            package: {
                "file": f"{package}.txt",
                "items": len(snapshots[package].splitlines()),
                "sha256": digest(snapshots[package]),
            }
            for package in libraries
        },
        "variants": FEATURE_VARIANTS,
        "feature_packages": {},
    }


def verify_api_neutral_variants(root: Path, snapshots: dict[str, str]) -> None:
    changed = []
    for name, variant in FEATURE_VARIANTS.items():
        expected = snapshots[variant["same_as"]]
        actual = render_snapshot(root, variant["package"], tuple(variant["features"]))
        if actual != expected:
            changed.append(name)
            diff = difflib.unified_diff(
                expected.splitlines(),
                actual.splitlines(),
                fromfile=f"accepted/{variant['same_as']}",
                tofile=f"current/{name}",
                lineterm="",
            )
            print("\n".join(diff), file=sys.stderr)
    if changed:
        raise ValueError(f"supported feature API is not API-neutral for {changed}")


def bless(root: Path, snapshot_dir: Path) -> None:
    tool, rustc = installed_versions(root)
    if tool != TOOL:
        raise ValueError(f"expected {TOOL}, found {tool}")
    libraries = discover_libraries(root)
    snapshots = {package: render_snapshot(root, package) for package in libraries}
    verify_api_neutral_variants(root, snapshots)
    manifest = build_manifest(libraries, snapshots, rustc)
    for name, spec in FEATURE_SNAPSHOTS.items():
        content = render_snapshot(root, spec["package"], tuple(spec["features"]))
        snapshots[name] = content
        manifest["feature_packages"][name] = {"file": f"{name}.txt", "items": len(content.splitlines()), "sha256": digest(content)}
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for package, content in snapshots.items():
        (snapshot_dir / f"{package}.txt").write_text(content)
    (snapshot_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    validate(root, snapshot_dir, manifest)
    print(
        f"blessed public API snapshots: {len(libraries)} crates, "
        f"{sum(v['items'] for v in manifest['packages'].values())} items, "
        f"{len(FEATURE_VARIANTS)} API-neutral feature variants"
    )


def check(root: Path, snapshot_dir: Path) -> None:
    manifest = load_manifest(snapshot_dir)
    validate(root, snapshot_dir, manifest)
    tool, rustc = installed_versions(root)
    if tool != manifest["tool"] or rustc != manifest["rustc"]:
        raise ValueError(
            "public API extraction provenance changed; "
            f"expected=({manifest['tool']}, {manifest['rustc']}), found=({tool}, {rustc})"
        )
    changed = []
    for package in discover_libraries(root):
        expected = (snapshot_dir / manifest["packages"][package]["file"]).read_text()
        actual = render_snapshot(root, package)
        if actual != expected:
            changed.append(package)
            diff = difflib.unified_diff(
                expected.splitlines(),
                actual.splitlines(),
                fromfile=f"accepted/{package}",
                tofile=f"current/{package}",
                lineterm="",
            )
            print("\n".join(diff), file=sys.stderr)
    for name, variant in manifest["variants"].items():
        package = variant["package"]
        expected = (snapshot_dir / manifest["packages"][variant["same_as"]]["file"]).read_text()
        actual = render_snapshot(root, package, tuple(variant["features"]))
        if actual != expected:
            changed.append(name)
            diff = difflib.unified_diff(
                expected.splitlines(),
                actual.splitlines(),
                fromfile=f"accepted/{variant['same_as']}",
                tofile=f"current/{name}",
                lineterm="",
            )
            print("\n".join(diff), file=sys.stderr)
    for name, spec in FEATURE_SNAPSHOTS.items():
        expected = (snapshot_dir / manifest["feature_packages"][name]["file"]).read_text()
        actual = render_snapshot(root, spec["package"], tuple(spec["features"]))
        if actual != expected:
            changed.append(name)
            print("\n".join(difflib.unified_diff(expected.splitlines(), actual.splitlines(), fromfile=f"accepted/{name}", tofile=f"current/{name}", lineterm="")), file=sys.stderr)
    if changed:
        raise ValueError(f"public API changed for {changed}; review, then run --bless deliberately")
    total = sum(entry["items"] for entry in manifest["packages"].values())
    print(
        f"public API snapshot: {len(manifest['packages'])} crates, {total} items, "
        f"{len(manifest['variants'])} API-neutral feature variants; no changes"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate", action="store_true", help="validate committed snapshot integrity")
    mode.add_argument("--check", action="store_true", help="regenerate with pinned tools and compare")
    mode.add_argument("--bless", action="store_true", help="replace snapshots after deliberate API review")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    snapshot_dir = root / "tools/ci/public-api"
    if args.bless:
        bless(root, snapshot_dir)
    elif args.check:
        check(root, snapshot_dir)
    else:
        manifest = load_manifest(snapshot_dir)
        validate(root, snapshot_dir, manifest)
        total = sum(entry["items"] for entry in manifest["packages"].values())
        print(
            f"public API snapshot integrity: {len(manifest['packages'])} crates, "
            f"{total} items, {len(manifest['variants'])} feature variants"
        )


if __name__ == "__main__":
    main()
