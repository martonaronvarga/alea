"""License-boundary and actual redistributed-notice regressions."""
import copy
import io
from pathlib import Path
import tarfile
import unittest

from licensing import check_archive, check_graph, required_notices

ROOT = Path(__file__).resolve().parents[2]


class LicensingTests(unittest.TestCase):
    def graph(self):
        return [
            {"name": "alea-core", "license": "MIT OR Apache-2.0", "dependencies": []},
            {"name": "alea-ffi", "license": "GPL-3.0-or-later", "dependencies": []},
        ]

    def test_native_package_cannot_be_mislabelled_permissive(self):
        packages = self.graph()
        check_graph(packages)
        packages[1]["license"] = "MIT OR Apache-2.0"
        with self.assertRaisesRegex(ValueError, "alea-ffi"):
            check_graph(packages)

    def test_new_transitive_native_edge_requires_review_but_dev_fixture_does_not(self):
        packages = self.graph()
        packages[0]["dependencies"] = [{"name": "alea-ffi", "kind": "dev"}]
        check_graph(packages)
        packages = copy.deepcopy(packages)
        packages[0]["dependencies"][0]["kind"] = None
        with self.assertRaisesRegex(ValueError, "production native dependency"):
            check_graph(packages)

    def test_packaged_license_and_notices_must_be_complete_and_unmodified(self):
        package = {"name": "alea-ffi", "version": "0.1.0"}
        notices = required_notices(ROOT, package["name"])
        def check(files):
            data = io.BytesIO()
            with tarfile.open(fileobj=data, mode="w") as archive:
                for filename, content in files.items():
                    member = tarfile.TarInfo("alea-ffi-0.1.0/" + filename)
                    member.size = len(content)
                    archive.addfile(member, io.BytesIO(content))
            data.seek(0)
            with tarfile.open(fileobj=data) as archive:
                check_archive(ROOT, package, archive)
        check(notices)
        missing = dict(notices)
        del missing["THIRD_PARTY_NOTICES.md"]
        with self.assertRaisesRegex(ValueError, "missing"):
            check(missing)
        changed = dict(notices, LICENSE=b"not the GPL text")
        with self.assertRaisesRegex(ValueError, "differs"):
            check(changed)


if __name__ == "__main__":
    unittest.main()
