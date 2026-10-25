"""Artifact format, reproducibility and exclusion contracts."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest

from source import archive, source_files
from verify import verify_archive


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        for name in ("crates/alea-core/src/lib.rs", ".agent/secret", "tools/release/file.py"):
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(name)

    def test_reproducible_bytes_and_manifest_cover_every_source_entry(self):
        one, two = self.root / "one.tar.gz", self.root / "two.tar.gz"
        self.assertEqual(archive(self.root, one), archive(self.root, two))
        with tarfile.open(one) as tar:
            names = tar.getnames()
            self.assertFalse(any(".agent" in n for n in names))
            manifest = json.load(tar.extractfile("alea/SOURCE-MANIFEST.json"))
            self.assertEqual(set(names), {"alea/" + e["path"] for e in manifest["files"]} | {"alea/SOURCE-MANIFEST.json"})
            for entry in manifest["files"]:
                self.assertEqual(entry["sha256"], hashlib.sha256(tar.extractfile("alea/" + entry["path"]).read()).hexdigest())
        with self.assertRaises(FileExistsError):
            archive(self.root, one)
        self.assertEqual(verify_archive(one), 2)

    def test_local_notes_datasets_and_history_never_ship(self):
        excluded = ["AGENTS.md", "design.md", "roadmap.md", "models.md", "SECURITY.md",
                    "research/calculation.rs", "archive/old.rs", "data/observations.csv",
                    "docs/benchmarks/report.json", "docs/settled-plan.md",
                    "tools/stan-reference/probe.cpp", "tools/release/observe.py",
                    "tools/release/examples/probe.rs", "crates/alea/design.md",
                    "crates/alea/tests/data/private.csv"]
        for name in excluded:
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("local-only")
        self.assertFalse(set(excluded) & {p.relative_to(self.root).as_posix() for p in source_files(self.root)})

    def test_unpacked_manifest_cannot_reintroduce_excluded_material(self):
        content = b"local-only"
        for name in ["design.md", "docs/benchmarks/results.json", "research/notes.rs", "AGENTS.md"]:
            (self.root / "SOURCE-MANIFEST.json").write_text(json.dumps({
                "schema_version": 1,
                "files": [{"path": name, "mode": 0o644,
                           "sha256": hashlib.sha256(content).hexdigest()}],
            }))
            with self.assertRaisesRegex(ValueError, "unsafe source manifest"):
                source_files(self.root)

    def test_source_content_changes_digest(self):
        before = archive(self.root, self.root / "one.tar.gz")
        (self.root / "crates/alea-core/src/lib.rs").write_text("changed")
        self.assertNotEqual(before, archive(self.root, self.root / "two.tar.gz"))

    def test_unpacked_source_reproduces_archive_without_git(self):
        original = self.root / "original.tar.gz"
        digest = archive(self.root, original)
        with tarfile.open(original) as tar:
            tar.extractall(self.root, filter="data")
        self.assertEqual(digest, archive(self.root / "alea", self.root / "reproduced.tar.gz"))
        (self.root / "alea/crates/alea-core/src/lib.rs").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "differs"):
            archive(self.root / "alea", self.root / "tampered.tar.gz")

    def test_manifest_cannot_override_agent_state_exclusion(self):
        (self.root / "SOURCE-MANIFEST.json").write_text(json.dumps({
            "schema_version": 1,
            "files": [{"path": ".agent/secret", "mode": 0o644,
                       "sha256": hashlib.sha256(b".agent/secret").hexdigest()}],
        }))
        with self.assertRaisesRegex(ValueError, "unsafe source manifest"):
            source_files(self.root)

    def test_symlink_input_is_rejected(self):
        (self.root / "crates/link").symlink_to("../.agent/secret")
        with self.assertRaisesRegex(ValueError, "symlink"):
            source_files(self.root)

    def test_unlisted_and_traversal_members_are_rejected(self):
        original = self.root / "original.tar.gz"
        archive(self.root, original)
        for name in ("alea/extra", "alea/../escape"):
            changed = self.root / "changed.tar.gz"
            with tarfile.open(original) as source, tarfile.open(changed, "w:gz") as target:
                for member in source.getmembers():
                    with source.extractfile(member) as stream:
                        target.addfile(member, stream)
                member = tarfile.TarInfo(name)
                member.size, member.mode = 4, 0o644
                target.addfile(member, io.BytesIO(b"oops"))
            with self.assertRaises((ValueError, KeyError)):
                verify_archive(changed)

    def test_signature_roundtrip_and_tampered_digest(self):
        key = self.root / "test-key"
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True)
        allowed = self.root / "allowed-signers"
        allowed.write_text("test@example.invalid " + key.with_suffix(".pub").read_text())
        digest = self.root / "digest"
        digest.write_text("test artifact digest\n")
        subprocess.run(["ssh-keygen", "-q", "-Y", "sign", "-f", str(key),
                        "-n", "alea-source-v1", str(digest)], check=True)
        def verify():
            with digest.open("rb") as stream:
                return subprocess.run(["ssh-keygen", "-Y", "verify", "-f", str(allowed),
                                       "-I", "test@example.invalid", "-n", "alea-source-v1", "-s",
                                       str(digest) + ".sig"], stdin=stream, capture_output=True).returncode
        self.assertEqual(verify(), 0)
        digest.write_text("changed digest\n")
        self.assertNotEqual(verify(), 0)


if __name__ == "__main__":
    unittest.main()
