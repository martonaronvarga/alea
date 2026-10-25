#!/usr/bin/env python3
"""Create a deterministic source archive; optionally sign its digest with SSH.

No network, Git mutation, key generation or publication is performed. Signing
requires an explicitly supplied maintainer identity. Verify with an independently
trusted allowed-signers file, never one shipped by the artifact being verified.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
FILES = {"README.md", "flake.nix", "flake.lock", ".gitmodules",
         ".gitignore", ".gitattributes",
         "LICENSE", "LICENSE-MIT", "LICENSE-APACHE", "LICENSE-GPL", "NOTICE",
         "LICENSING.md", "THIRD_PARTY_NOTICES.md", "CONTRIBUTING.md"}
EXCLUDED = {"target", "__pycache__", ".git", ".agent", ".codex", "AGENTS.md", "SECURITY.md", "design.md", "roadmap.md", "models.md", "archive", "research", "data"}

DOCS = {
    "README.md", "installation.md", "getting-started.md", "modeling.md",
    "sampling.md", "diagnostics.md", "sessions.md", "distributions.md",
    "wiener.md", "troubleshooting.md", "releases.md",
    "enzyme-native-models.md", "metric-contract.md",
}
TOOLS = {"ci", "release", "documentation", "downstream-conformance", "compile-fail", "enzyme-conformance"}


def allowed_source_path(relative: PurePosixPath) -> bool:
    """Only build inputs, executable checks, licenses, and the user guide ship."""
    parts = relative.parts
    if (relative.is_absolute() or not parts or ".." in parts
            or any(p in EXCLUDED for p in parts)):
        return False
    if str(relative) in {"tools/release/observe.py", "tools/release/examples/probe.rs"}:
        return False
    if len(parts) == 1:
        return str(relative) in FILES
    if parts[0] == "docs":
        return len(parts) == 2 and parts[1] in DOCS
    if parts[0] == "tools":
        return parts[1] in TOOLS
    if parts[0] == ".github":
        return len(parts) == 3 and parts[1] == "workflows" and parts[2] in {"ci.yml", "release-validation.yml"}
    return parts[0] in {"crates", "nix"}


def source_files(root: Path) -> list[Path]:
    """Include tracked and new source, expanding pinned submodules explicitly."""
    manifest_path = root / "SOURCE-MANIFEST.json"
    if manifest_path.is_file():
        # A verified unpacked source artifact has no Git metadata. Its sealed
        # input manifest is sufficient to reproduce the exact archive bytes.
        manifest = json.loads(manifest_path.read_text())
        if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
            raise ValueError("unsupported source manifest")
        files = []
        for entry in manifest["files"]:
            relative = PurePosixPath(entry["path"])
            if (relative.is_absolute() or ".." in relative.parts or not relative.parts
                    or relative.as_posix() != entry["path"]
                    or not allowed_source_path(relative)):
                raise ValueError("unsafe source manifest path")
            path = root / relative
            if (not path.resolve().is_relative_to(root.resolve()) or path.is_symlink()
                    or any(parent.is_symlink() for parent in path.parents if parent != root)):
                raise ValueError("source manifest symlink escape")
            with path.open("rb") as stream:
                sha = hashlib.file_digest(stream, "sha256").hexdigest()
            mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
            if sha != entry["sha256"] or mode != entry["mode"]:
                raise ValueError("unpacked source content or mode differs")
            files.append(path)
        if len(files) != len(set(files)):
            raise ValueError("duplicate source manifest paths")
        return sorted(files)
    raw = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root
    )
    names = set()
    for entry in raw.decode().split("\0"):
        if not entry:
            continue
        relative = Path(entry)
        if not allowed_source_path(PurePosixPath(relative.as_posix())):
            continue
        p = root / entry
        if p.is_symlink():
            raise ValueError(f"source symlinks require explicit review: {entry}")
        if p.is_dir():  # Gitlink: git ls-files does not recurse into submodules.
            sub = subprocess.check_output(["git", "ls-files", "-z"], cwd=p)
            names.update(p / s for s in sub.decode().split("\0") if s)
        else:
            names.add(p)
    result = []
    for p in sorted(names):
        relative = p.relative_to(root)
        if not allowed_source_path(PurePosixPath(relative.as_posix())):
            continue
        if any(s in EXCLUDED for s in relative.parts):
            continue
        if p.is_symlink():
            raise ValueError(f"source symlinks require explicit review: {relative}")
        if not p.is_file():
            raise ValueError(f"missing source input: {relative}")
        result.append(p)
    return result


def archive(root: Path, destination: Path) -> str:
    """Write once, with normalized tar metadata and gzip headers."""
    paths = source_files(root)
    entries = []
    with destination.open("xb") as output:
        with gzip.GzipFile(fileobj=output, filename="", mode="wb", mtime=0, compresslevel=9) as zipped:
            with tarfile.open(fileobj=zipped, mode="w", format=tarfile.USTAR_FORMAT) as tar:
                for path in paths:
                    relative = path.relative_to(root).as_posix()
                    content = path.read_bytes()
                    mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
                    entry = tarfile.TarInfo("alea/" + relative)
                    entry.size, entry.mode = len(content), mode
                    tar.addfile(entry, io.BytesIO(content))
                    entries.append({"path": relative, "mode": mode,
                                    "sha256": hashlib.sha256(content).hexdigest()})
                manifest = json.dumps({"schema_version": 1, "files": entries}, sort_keys=True, indent=2).encode() + b"\n"
                entry = tarfile.TarInfo("alea/SOURCE-MANIFEST.json")
                entry.size, entry.mode = len(manifest), 0o644
                tar.addfile(entry, io.BytesIO(manifest))
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new .tar.gz path")
    parser.add_argument("--signing-key", type=Path, help="maintainer SSH private key or public key backed by ssh-agent")
    args = parser.parse_args()
    digest = args.output.with_name(args.output.name + ".sha256")
    signature = digest.with_name(digest.name + ".sig")
    if any(p.exists() for p in (args.output, digest, signature)):
        parser.error("artifact, digest and signature paths must all be new")
    sha = archive(ROOT, args.output)
    with digest.open("x") as output:
        output.write(f"{sha}  {args.output.name}\n")
    if args.signing_key:
        subprocess.run(["ssh-keygen", "-Y", "sign", "-f", str(args.signing_key),
                        "-n", "alea-source-v1", str(digest)], check=True)
    print(json.dumps({"archive": str(args.output), "sha256": sha,
                      "signed": args.signing_key is not None}))


if __name__ == "__main__":
    main()
