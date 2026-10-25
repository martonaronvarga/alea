"""CI failure diagnostics preserve exit status and cannot inject commands."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

from run import annotation, run


class RunnerTests(unittest.TestCase):
    def test_workflow_command_data_is_escaped(self):
        self.assertEqual(annotation("100%\0\r\n::error::another"),
                         "100%25%0D%0A::error::another")

    def test_failed_process_retains_log_and_status_with_bounded_annotation(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "command.log"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = run([sys.executable, "-c",
                              "print('start');print('x'*20000);print('actual failure');exit(7)"], log)
            self.assertEqual(status, 7)
            self.assertIn("start", log.read_text())
            self.assertIn("actual failure", log.read_text())
            line = next(s for s in output.getvalue().splitlines() if s.startswith("::error"))
            self.assertLess(len(line), 15000)
            self.assertIn("actual failure", line)


if __name__ == "__main__":
    unittest.main()
