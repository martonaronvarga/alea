"""Validate and reproduce Alea's accepted default-feature public API snapshots."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib


SCHEMA_VERSION = 1
TOOL = "cargo-public-api 0.51.0"
POLICY = {
    "default_features": False,
    "omit": ["blanket-impls", "auto-trait-impls"],
    "parameter_names": False,
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


def render_snapshot(root: Path, package: str) -> str:
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
    return normalize_snapshot(command_output(command, root))


def digest(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def load_manifest(snapshot_dir: Path) -> dict:
    try:
        return json.loads((snapshot_dir / "manifest.json").read_text())
    except FileNotFoundError as error:
        raise ValueError("public API snapshot manifest is missing") from error


def validate(root: Path, snapshot_dir: Path, manifest: dict) -> None:
    required = {"schema_version", "tool", "rustc", "policy", "packages"}
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
    }


def bless(root: Path, snapshot_dir: Path) -> None:
    tool, rustc = installed_versions(root)
    if tool != TOOL:
        raise ValueError(f"expected {TOOL}, found {tool}")
    libraries = discover_libraries(root)
    snapshots = {package: render_snapshot(root, package) for package in libraries}
    manifest = build_manifest(libraries, snapshots, rustc)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for package, content in snapshots.items():
        (snapshot_dir / f"{package}.txt").write_text(content)
    (snapshot_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    validate(root, snapshot_dir, manifest)
    print(f"blessed public API snapshots: {len(libraries)} crates, {sum(v['items'] for v in manifest['packages'].values())} items")


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
    if changed:
        raise ValueError(f"public API changed for {changed}; review, then run --bless deliberately")
    total = sum(entry["items"] for entry in manifest["packages"].values())
    print(f"public API snapshot: {len(manifest['packages'])} crates, {total} items; no changes")


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
        print(f"public API snapshot integrity: {len(manifest['packages'])} crates, {total} items")


if __name__ == "__main__":
    main()
