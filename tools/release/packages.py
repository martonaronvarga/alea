#!/usr/bin/env python3
"""Prepare all .crate source packages and build a consumer from their contents.

Pre-alpha crates are unpublished and have cyclic dev fixtures. Omit package-local
locks and patch registry edges only in the independent test consumer. The source
release still includes the measured workspace lock. This does not publish crates
or claim that their registry dependencies already exist.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from licensing import check_archive, check_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new artifact verification directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads(subprocess.check_output([
        "cargo", "metadata", "--manifest-path", "crates/Cargo.toml", "--no-deps",
        "--locked", "--format-version", "1",
    ], cwd=root, text=True))
    check_graph(metadata["packages"])
    subprocess.run(["cargo", "package", "--manifest-path", "crates/Cargo.toml",
                    "--workspace", "--allow-dirty", "--no-verify", "--exclude-lockfile", "--offline"],
                   cwd=root, check=True)
    unpacked = output / "consumer-source"
    (unpacked / "crates").mkdir(parents=True)
    example = unpacked / "tools/release/examples/reference.rs"
    example.parent.mkdir(parents=True)
    shutil.copyfile(root / "tools/release/examples/reference.rs", example)
    shutil.copyfile(root / "tools/release/examples/workflow.rs", example.with_name("workflow.rs"))
    entries = []
    for package in metadata["packages"]:
        stem = package["name"] + "-" + package["version"]
        source = Path(metadata["target_directory"]) / "package" / (stem + ".crate")
        destination = output / source.name
        shutil.copyfile(source, destination)
        with tarfile.open(destination, "r:gz") as archive:
            check_archive(root, package, archive)
            members = archive.getmembers()
            if any(not m.isfile() or not m.name.startswith(stem + "/") for m in members):
                raise ValueError("unexpected cargo package layout")
            archive.extractall(unpacked / "crates", filter="data")
        (unpacked / "crates" / stem).rename(unpacked / "crates" / package["name"])
        entries.append({"package": package["name"], "version": package["version"],
                        "license": package["license"], "notices_verified": True,
                        "artifact": destination.name,
                        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
    subprocess.run([sys.executable, str(root / "tools/release/fresh.py"), str(unpacked), "--packaged"],
                   cwd=root, check=True)
    subprocess.run([sys.executable, str(root / "tools/release/fresh.py"), str(unpacked), "--packaged",
                    "--workflow", "--output", str(output / "workflow")], cwd=root, check=True)
    (output / "packages.json").write_text(json.dumps({"schema_version": 1, "packages": entries,
                                                    "consumer_verified": True}, indent=2) + "\n")
    print(f"verified {len(entries)} source packages and fresh reference consumer")


if __name__ == "__main__":
    main()
