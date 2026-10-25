#!/usr/bin/env python3
"""Verify source digests, optional trusted SSH signatures and every tar member."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile


def verify_archive(path: Path) -> int:
    """Reject duplicate/unsafe names, links, unexpected entries and hash changes."""
    with tarfile.open(path, "r:gz") as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)):
            raise ValueError("duplicate source archive members")
        for member in members:
            parts = PurePosixPath(member.name).parts
            if (not member.isfile() or not parts or parts[0] != "alea"
                    or any(p in {"..", ".agent", ".git", ".codex"} for p in parts)
                    or member.name.startswith("/") or member.size > 256 * 1024 * 1024
                    or member.uid != 0 or member.gid != 0 or member.mtime != 0
                    or member.mode not in {0o644, 0o755}):
                raise ValueError(f"invalid source member: {member.name}")
        manifest_member = tar.getmember("alea/SOURCE-MANIFEST.json")
        if manifest_member.size > 16 * 1024 * 1024:
            raise ValueError("oversized source manifest")
        with tar.extractfile(manifest_member) as stream:
            manifest = json.load(stream)
        if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
            raise ValueError("unsupported source manifest schema")
        expected = {"alea/SOURCE-MANIFEST.json"}
        for entry in manifest["files"]:
            name = "alea/" + entry["path"]
            if name in expected:
                raise ValueError("duplicate source manifest entry")
            expected.add(name)
            member = tar.getmember(name)
            with tar.extractfile(member) as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if member.mode != entry["mode"] or digest != entry["sha256"]:
                raise ValueError(f"source content/mode differs: {name}")
        if set(names) != expected:
            raise ValueError("source manifest coverage differs from archive")
    return len(manifest["files"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--allowed-signers", type=Path)
    parser.add_argument("--identity")
    args = parser.parse_args()
    if bool(args.allowed_signers) != bool(args.identity):
        parser.error("trusted allowed-signers and identity must be supplied together")
    digest_path = args.archive.with_name(args.archive.name + ".sha256")
    digest_text = digest_path.read_text()
    if args.allowed_signers:
        with digest_path.open("rb") as digest:
            subprocess.run(["ssh-keygen", "-Y", "verify", "-f", str(args.allowed_signers),
                            "-I", args.identity, "-n", "alea-source-v1", "-s",
                            str(digest_path) + ".sig"], stdin=digest, check=True)
    match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)\n", digest_text)
    with args.archive.open("rb") as archive:
        sha = hashlib.file_digest(archive, "sha256").hexdigest()
    if not match or match[1] != sha or match[2] != args.archive.name:
        raise ValueError("archive digest or filename mismatch")
    count = verify_archive(args.archive)
    print(json.dumps({"verified_source_files": count, "sha256": sha,
                      "trusted_signature": args.allowed_signers is not None}))


if __name__ == "__main__":
    main()
