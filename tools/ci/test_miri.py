"""Exact float regressions must execute, without disabling perturbed Miri gates."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class MiriSelectionTests(unittest.TestCase):
    def commands(self, scope):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            log = directory / "calls.jsonl"
            cargo = directory / "cargo"
            cargo.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "with open(os.environ['ALEA_TEST_CARGO_CALLS'], 'a') as output:\n"
                "    output.write(json.dumps({'args':sys.argv[1:], 'flags':os.environ['MIRIFLAGS']})+'\\n')\n"
            )
            cargo.chmod(0o755)
            result = subprocess.run(
                ["bash", str(ROOT / "tools/ci/miri.sh"), scope],
                env={**os.environ, "PATH":str(directory) + os.pathsep + os.environ["PATH"],
                     "ALEA_TEST_CARGO_CALLS":str(log)},
                text=True, capture_output=True, check=False,
            )
            calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
            return result, calls

    def test_exact_regressions_are_run_separately_from_perturbed_contracts(self):
        exact = {
            "storage":"numeric::finite_tests::logistic_primitives_preserve_small_log_probabilities",
            "targets":"rejects_precision_collapse_with_deterministic_transcendentals",
        }
        sources = {
            "storage":ROOT / "crates/alea-math/src/numeric.rs",
            "targets":ROOT / "crates/alea-core/tests/transforms.rs",
        }
        for scope, name in exact.items():
            with self.subTest(scope=scope):
                result, calls = self.commands(scope)
                self.assertEqual(result.returncode, 0, result.stderr)
                selected = [call for call in calls if name in call["args"]]
                self.assertEqual(len(selected), 1)
                self.assertIn("-Zmiri-deterministic-floats", selected[0]["flags"])
                self.assertIn("--exact", selected[0]["args"])
                self.assertIn("--ignored", selected[0]["args"])
                self.assertIn("fn " + name.split("::")[-1] + "(", sources[scope].read_text())
                perturbed = [call for call in calls if call not in selected]
                self.assertTrue(perturbed)
                for call in perturbed:
                    self.assertNotIn("-Zmiri-deterministic-floats", call["flags"])
                    self.assertIn("-Zmiri-strict-provenance", call["flags"])
                    self.assertIn("-Zmiri-symbolic-alignment-check", call["flags"])

    def test_all_ignored_selections_keep_explicit_deterministic_floats(self):
        result, calls = self.commands("all")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertGreaterEqual(len(calls), 20)
        for call in calls:
            if "--ignored" in call["args"]:
                self.assertIn("-Zmiri-deterministic-floats", call["flags"])
                self.assertIn("--exact", call["args"])

    def test_unknown_scope_fails_without_running_cargo(self):
        result, calls = self.commands("not-a-scope")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
