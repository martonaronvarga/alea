"""Artifact format, reproducibility and exclusion contracts."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest

from source import archive, source_files, check_source_links
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
                    "crates/alea/tests/data/private.csv",
                    "crates/alea-cli/src/bin/bench.rs", "crates/alea-mcmc/examples/hard_nuts.rs",
                    "tools/native-benchmark/stan/linear", "tools/native-benchmark/results/raw.json",
                    "tools/native-benchmark/scripts/private.json", "tools/wiener-validation/failures.jsonl",
                    "tools/wiener-modeling/data/empirical.csv"]
        for name in excluded:
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("local-only")
        self.assertFalse(set(excluded) & {p.relative_to(self.root).as_posix() for p in source_files(self.root)})

    def test_verification_source_and_locks_ship_without_generated_artifacts(self):
        included = ['docs/native-model-performance.md','docs/kernel-optimization.md',
                    'tools/bench-features.sh',
                    'tools/native-benchmark/PINNED.json','tools/native-benchmark/REPORT.md',
                    'tools/native-benchmark/stan/linear.stan','tools/native-benchmark/scripts/replay.py',
                    'tools/wiener-validation/report.py','tools/wiener-validation/test_report.py']
        for tool in ['native-benchmark','wiener-validation','wiener-modeling']:
            included += [f'tools/{tool}/{p}' for p in ['Cargo.toml','Cargo.lock','README.md','src/main.rs']]
        for name in included:
            p = self.root/name
            p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text('verification source')
        paths = {p.relative_to(self.root).as_posix() for p in source_files(self.root)}
        self.assertTrue(set(included).issubset(paths))

    def test_relative_links_use_shipped_manifest_not_checkout_existence(self):
        contents = {'docs/README.md': b'[guide](modeling.md#model) [tool](../tools/check)\n'
                    b'[web](https://example.invalid/missing) [heading](#local)\n'
                    b'[guide][ref]\n[ref]: modeling.md\n```md\n[example](omitted.md)\n```\n',
                    'docs/modeling.md': b'guide', 'tools/check/main.py': b'check',
                    'crates/ffi/vendor/README.md': b'[upstream wiki](missing.md)'}
        check_source_links(contents)
        for target in [b'missing.md', b'../.agent/secret', b'modeling%20notes.md']:
            with self.subTest(target=target), self.assertRaisesRegex(ValueError,'omitted files'):
                check_source_links(dict(contents, **{'docs/README.md': b'[bad]('+target+b')'}))

    def test_untracked_feature_runner_survives_real_ignore_policy_and_git_free_help(self):
        repository = Path(__file__).resolve().parents[2]
        ignore = (repository / '.gitignore').read_bytes()
        (self.root / '.gitignore').write_bytes(ignore)
        runner = self.root / 'tools/bench-features.sh'
        original = repository / 'tools/bench-features.sh'
        runner.write_bytes(original.read_bytes())
        runner.chmod(original.stat().st_mode & 0o777)
        doc = self.root / 'docs/README.md'
        doc.parent.mkdir()
        doc.write_text('[feature runner](../tools/bench-features.sh)\n')
        # Unlike a force-added file, an ordinary new source file must be
        # discoverable under the checkout's REAL ignore rules before staging.
        self.assertEqual(subprocess.check_output(
            ['git', 'ls-files', '--cached', '--', 'tools/bench-features.sh'],
            cwd=self.root), b'')
        self.assertIn(runner, source_files(self.root))

        # Reproduce the old ignored/untracked failure. Checkout existence alone
        # cannot satisfy the archive's documentation/runner closure.
        (self.root / '.gitignore').write_bytes(ignore + b'\n/tools/*.sh\n')
        self.assertNotIn(runner, source_files(self.root))
        omitted = self.root / 'omitted.tar.gz'
        with self.assertRaisesRegex(ValueError, 'omitted files'):
            archive(self.root, omitted)
        self.assertFalse(omitted.exists())
        (self.root / '.gitignore').write_bytes(ignore)

        output = self.root / 'source.tar.gz'
        archive(self.root, output)
        verify_archive(output)
        with tarfile.open(output) as tar:
            tar.extractall(self.root, filter='data')
        unpacked = self.root / 'alea'
        restored = unpacked / 'tools/bench-features.sh'
        self.assertFalse((unpacked / '.git').exists())
        self.assertIn(restored, source_files(unpacked))
        self.assertEqual(restored.read_bytes(), original.read_bytes())
        self.assertEqual(restored.stat().st_mode & 0o111, 0o111)
        bash = shutil.which('bash')
        self.assertIsNotNone(bash)
        # --help must work before invoking Cargo, building, or admitting timings.
        result = subprocess.run([bash, str(restored), '--help'], cwd=unpacked,
                                env={'PATH': ''}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Usage: tools/bench-features.sh', result.stdout)
        self.assertIn('Slurm compute allocation', result.stdout)
        denied = subprocess.run([bash, str(restored)], cwd=unpacked,
                                env={'PATH': ''}, capture_output=True, text=True)
        self.assertEqual(denied.returncode, 2, denied.stderr)
        self.assertIn('Timing requires a Slurm compute node', denied.stderr)
        self.assertFalse((unpacked / 'target').exists())

    def test_archive_rejects_broken_links_and_git_free_recheck_uses_exact_manifest(self):
        doc = self.root/'docs/README.md'
        doc.parent.mkdir()
        doc.write_text('[private](../.agent/secret)')
        output = self.root/'broken.tar.gz'
        with self.assertRaisesRegex(ValueError,'omitted files'): archive(self.root,output)
        self.assertFalse(output.exists())
        doc.write_text('[source](../crates/alea-core/src/lib.rs)')
        archive(self.root,output)
        with tarfile.open(output) as tar: tar.extractall(self.root,filter='data')
        unpacked = self.root/'alea'
        paths = source_files(unpacked)
        check_source_links({p.relative_to(unpacked).as_posix(): p.read_bytes() for p in paths})

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
