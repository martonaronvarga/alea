"""Completion notification and deadline handling, independent of sampler behavior."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alea_bench.process import isolated

class CompletionTests(unittest.TestCase):
    def test_wait_is_blocking_and_elapsed_time_is_not_rounded(self):
        process=unittest.mock.Mock(pid=123)
        process.wait.return_value=0;process.poll.return_value=0
        with patch('alea_bench.process.subprocess.Popen',return_value=process),patch('alea_bench.process.os.pidfd_open',return_value=99),patch('alea_bench.process.os.close'),patch('alea_bench.process.select.select',return_value=([99],[],[])),patch('alea_bench.process.time.perf_counter_ns',side_effect=[1000,1235567]):
            result=isolated(['worker'],cwd='.',env={},log=None,timeout=1)
        process.wait.assert_called_once_with()
        self.assertEqual(result['wall_nanoseconds'],1234567)
        self.assertEqual(result['wall_seconds'],.001234567)
    def test_real_child_deadline_is_preserved(self):
        with tempfile.TemporaryFile() as log:
            result=isolated([sys.executable,'-c','import time; time.sleep(30)'],cwd=Path.cwd(),env=os.environ.copy(),log=log,timeout=.02)
        self.assertTrue(result['timed_out']);self.assertLess(result['returncode'],0)
    def test_real_short_child_exits_normally(self):
        with tempfile.TemporaryFile() as log:
            result=isolated([sys.executable,'-c','pass'],cwd=Path.cwd(),env=os.environ.copy(),log=log,timeout=10)
        self.assertFalse(result['timed_out']);self.assertEqual(result['returncode'],0)
        self.assertGreater(result['wall_nanoseconds'],0)
