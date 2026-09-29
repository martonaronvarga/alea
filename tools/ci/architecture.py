"""Enforce Alea's production crate boundaries, including optional/build edges.

Dev dependencies may cross layers for integration tests; production edges may not.
Uses only Python's standard library and Cargo's resolved workspace metadata.
"""
import json
from pathlib import Path
import subprocess


ALLOWED = {
    "alea-math": set(),
    "alea-core": {"alea-math"},
    "alea-autodiff": {"alea-core"},
    "alea-distributions": {"alea-core", "alea-math", "alea-ffi"},
    "alea-mcmc": {"alea-core", "alea-math"},
    "alea-smc": {"alea-core", "alea-math"},
    "alea-ffi": set(),
    "alea-runtime": {"alea-mcmc", "alea-math"},
    "alea-cli": {"alea-core", "alea-math", "alea-autodiff", "alea-distributions", "alea-mcmc", "alea-runtime", "alea-smc"},
}


def check(metadata):
    members = {p["name"]: p for p in metadata["packages"] if p["id"] in metadata["workspace_members"]}
    if set(members) != set(ALLOWED):
        raise ValueError(f"unexpected workspace members: {set(members) ^ set(ALLOWED)}")
    root = Path(metadata["workspace_root"])
    for name, package in members.items():
        if Path(package["manifest_path"]) != root / name / "Cargo.toml":
            raise ValueError(f"{name}: noncanonical crate path")
        for dep in package["dependencies"]:
            if dep["kind"] == "dev":
                continue
            if dep.get("path"):
                if dep["name"] not in ALLOWED[name]:
                    raise ValueError(f"forbidden production edge: {name} -> {dep['name']}")
                if dep.get("rename"):
                    raise ValueError(f"compatibility dependency alias: {name} -> {dep['name']}")


if __name__ == "__main__":
    metadata = json.loads(subprocess.check_output([
        "cargo", "metadata", "--manifest-path", "crates/Cargo.toml",
        "--no-deps", "--format-version", "1", "--locked",
    ]))
    check(metadata)
    print("nine canonical crates; production boundaries and dependency names verified")
