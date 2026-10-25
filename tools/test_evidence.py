"""Retention, tamper detection and containment for external evidence stores."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from evidence import copy_verified, entries, verify


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source, self.store = self.root / "source", self.root / "store"
        self.row = {"path": "docs/benchmarks/run/evidence.tar.gz.part001",
                    "bytes": 8, "sha256": hashlib.sha256(b"evidence").hexdigest()}
        path = self.source / self.row["path"]
        path.parent.mkdir(parents=True)
        path.write_bytes(b"evidence")

    def test_roundtrip_and_existing_identical_payload(self):
        copy_verified(self.source, self.store, self.row)
        copy_verified(self.source, self.store, self.row)
        copy_verified(self.store, self.root / "restored", self.row)
        verify(self.root / "restored", self.row)

    def test_corrupt_source_is_not_copied(self):
        (self.source / self.row["path"]).write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "checksum"):
            copy_verified(self.source, self.store, self.row)
        self.assertFalse(self.store.exists())

    def test_existing_different_evidence_is_preserved(self):
        copy_verified(self.source, self.store, self.row)
        target = self.store / self.row["path"]
        target.write_bytes(b"keep me")
        with self.assertRaises(ValueError):
            copy_verified(self.source, self.store, self.row)
        self.assertEqual(target.read_bytes(), b"keep me")

    def test_store_symlink_is_refused(self):
        self.store.mkdir()
        (self.store / "docs").symlink_to(self.source / "docs", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            copy_verified(self.source, self.store, self.row)

    def test_catalog_rejects_traversal_duplicates_and_boolean_version(self):
        catalog = self.root / "catalog.json"
        for value in (
            {"schema_version": True, "files": [self.row]},
            {"schema_version": 1, "files": [self.row, self.row]},
            {"schema_version": 1, "files": [{**self.row, "path": "docs/benchmarks/../../escape.tar.gz"}]},
        ):
            catalog.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                entries(catalog)


if __name__ == "__main__":
    unittest.main()
