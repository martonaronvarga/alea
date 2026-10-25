#!/usr/bin/env python3
"""Check package preparation; candidate mode additionally requires licensing.

This check never grants S3/S4. Platform execution, signing, package verification
and the real candidate cycle are separate recorded release evidence.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tomllib

from licensing import check_graph, required_notices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    metadata = json.loads(subprocess.check_output([
        "cargo", "metadata", "--manifest-path", str(root / "crates/Cargo.toml"),
        "--no-deps", "--locked", "--format-version", "1",
    ], text=True))
    policy = (root / "docs/releases.md").read_text()
    errors, blocked = [], []
    try:
        check_graph(metadata["packages"])
    except ValueError as error:
        errors.append(str(error))
    for package in metadata["packages"]:
        name = package["name"]
        for field in ("description", "repository", "homepage", "readme", "rust_version", "categories", "keywords"):
            if not package.get(field): errors.append(f"{name}: missing {field}")
        manifest = Path(package["manifest_path"])
        readme = manifest.parent / package["readme"]
        if not readme.is_file(): errors.append(f"{name}: README missing")
        if not (package.get("license") or package.get("license_file")):
            blocked.append(f"{name}: maintainer license selection pending")
        for filename, expected in required_notices(root, name).items():
            path = manifest.parent / filename
            if not path.is_file() or path.read_bytes() != expected:
                errors.append(f"{name}: canonical license/notice missing or changed: {filename}")
        for dep in package["dependencies"]:
            if dep.get("path") and dep["kind"] != "dev" and dep["req"] == "*":
                errors.append(f"{name}: local dependency {dep['name']} has no registry version")
        for feature in package["features"]:
            if feature != "default" and f"`{feature}`" not in policy:
                errors.append(f"{name}: undocumented feature {feature}")
        with manifest.open("rb") as handle:
            raw = tomllib.load(handle)
        if raw["package"].get("publish", {}).get("workspace") is not True:
            errors.append(f"{name}: independent publication needs explicit release review")
    print(json.dumps({"schema_version": 1, "packages": len(metadata["packages"]),
                      "preparation_errors": errors, "license_blockers": blocked,
                      "release_claim": "metadata verified separately from artifact publication"}, indent=2))
    if errors or (args.candidate and blocked):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
