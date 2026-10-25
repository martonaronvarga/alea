#!/usr/bin/env python3
"""Verify, archive or restore immutable benchmark payloads without rerunning them.

The catalog stays in Git. Stores keep the same repository-relative paths; they
may be copied to a backup host without changing report or bundle identities.
Existing files are verified and never overwritten. This tool never deletes files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "docs/benchmarks/payloads.json"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def entries(catalog: Path) -> list[dict]:
    value = json.loads(catalog.read_text())
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("unsupported evidence catalog")
    rows = value["files"]
    names = []
    for row in rows:
        name = row["path"]
        path = PurePosixPath(name)
        if (path.is_absolute() or ".." in path.parts or path.as_posix() != name
                or path.parts[:2] != ("docs", "benchmarks")
                or not (path.name.endswith(".tar.gz") or ".tar.gz.part" in path.name)
                or type(row["bytes"]) is not int or row["bytes"] < 0
                or len(row["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in row["sha256"])):
            raise ValueError(f"invalid evidence catalog entry: {name}")
        names.append(name)
    if names != sorted(set(names)):
        raise ValueError("evidence catalog paths must be sorted and unique")
    return rows


def location(root: Path, row: dict) -> Path:
    root = root.resolve()
    path = root / row["path"]
    if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != root):
        raise ValueError(f"evidence symlink refused: {path}")
    if not path.resolve().is_relative_to(root):
        raise ValueError("evidence path escapes its store")
    return path


def verify(root: Path, row: dict) -> None:
    path = location(root, row)
    if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
        raise ValueError(f"evidence size or checksum mismatch: {path}")


def copy_verified(source: Path, destination: Path, row: dict) -> None:
    verify(source, row)
    target = location(destination, row)
    if target.exists():
        verify(destination, row)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves existing evidence, including concurrent writers.
    output = target.open("xb")
    try:
        with output:
            with location(source, row).open("rb") as stream:
                shutil.copyfileobj(stream, output)
        verify(destination, row)
    except BaseException:
        target.unlink()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("catalog", "verify", "archive", "restore"))
    parser.add_argument("--store", type=Path, help="external directory preserving catalog paths")
    parser.add_argument("--prefix", default="docs/benchmarks/", help="catalog path prefix to select")
    args = parser.parse_args()
    rows = entries(CATALOG)
    rows = [r for r in rows if r["path"].startswith(args.prefix)]
    if not rows:
        parser.error("no payloads selected")
    if args.command in ("archive", "restore"):
        if args.store is None:
            parser.error("archive/restore requires --store")
        if args.store.resolve().is_relative_to(ROOT):
            parser.error("evidence store must be outside the checkout")
    for row in rows:
        if args.command == "verify":
            verify(args.store or ROOT, row)
        elif args.command == "archive":
            copy_verified(ROOT, args.store, row)
        elif args.command == "restore":
            copy_verified(args.store, ROOT, row)
    print(f"{args.command}: {len(rows)} payloads, {sum(r['bytes'] for r in rows):,} bytes")
    if args.command == "catalog":
        print("Catalog validated; payload bytes were not read. Use verify for content checks.")


if __name__ == "__main__":
    main()
