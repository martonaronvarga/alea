"""Contract checks for timing artifact publication, independent of performance."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("fisher_cost", Path(__file__).with_name("fisher-cost.py"))
cost = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cost)


class CostTests(unittest.TestCase):
    def run_main(self, output, measure):
        with (
            patch.object(sys, "argv", ["fisher-cost", sys.executable, str(output), "--runs", "2", "--warmups", "1"]),
            patch.object(cost.shutil, "which", return_value="time"),
            patch.object(cost.platform, "platform", return_value="Linux-test"),
            patch.object(cost.subprocess, "check_output", side_effect=["GNU time\n", "rustc test\n"]),
            patch.object(cost, "measure", side_effect=measure),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            cost.main()

    def test_round_order_warmup_exclusion_and_raw_samples(self):
        calls = []

        def measure(command, _timer):
            calls.append(command[1:])
            return {"wall_seconds": len(calls), "user_seconds": 0.1, "system_seconds": 0.0, "peak_rss_kib": 100}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            self.run_main(output, measure)
            artifact = json.loads(output.read_text())
        forward = [list(arguments) for _, arguments in cost.WORKLOADS]
        self.assertEqual(calls, forward + forward[::-1] + forward)
        for name, result in artifact["results"].items():
            self.assertEqual([sample["round"] for sample in result["samples"]], [0, 1], name)
            self.assertTrue(all(sample["wall_seconds"] > 5 for sample in result["samples"]))
        self.assertEqual(len(artifact["binary_sha256"]), 64)

    def test_failed_child_does_not_publish_success_artifact(self):
        def failure(command, _timer):
            raise subprocess.CalledProcessError(3, command)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            with self.assertRaises(subprocess.CalledProcessError):
                self.run_main(output, failure)
            self.assertFalse(output.exists())

    def test_measure_parses_each_process_resource_record(self):
        def completed(command, **_kwargs):
            Path(command[command.index("-o") + 1]).write_text("0.12 0.03 4567\n")

        with (
            patch.object(cost.subprocess, "run", side_effect=completed),
            patch.object(cost.time, "perf_counter", side_effect=[5.0, 5.25]),
        ):
            sample = cost.measure(["binary", "fit"], "time")
        self.assertEqual(sample, {"wall_seconds": 0.25, "user_seconds": 0.12, "system_seconds": 0.03, "peak_rss_kib": 4567})

    def test_existing_artifact_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            output.write_text("previous run\n")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.run_main(output, lambda *_args: self.fail("must not measure"))
            self.assertEqual(output.read_text(), "previous run\n")

    def test_changed_binary_prevents_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            with (
                patch.object(cost, "fingerprint", side_effect=["before"] * (len(cost.SOURCES) + 1) + ["after"]),
                self.assertRaisesRegex(RuntimeError, "changed during measurement"),
            ):
                self.run_main(output, lambda *_args: {"wall_seconds": 1.0})
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
