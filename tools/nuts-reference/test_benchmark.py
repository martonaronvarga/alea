"""Timing denominators must include every chain and discarded integration work."""
import csv
from pathlib import Path
import tempfile
import unittest
import numpy as np
from benchmark import aggregate


class TimingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)/'timing.csv'
        self.rows = [(t,s,1.,2.,100) for t in range(4) for s in range(501,505)]
        self.draws = np.zeros((4,4,8192,8))
        self.draws[:,:,:,6] = 3
        self.diagnostics = [dict(target=t,bulk_ess=400+o,passed=True) for t in range(4) for o in range(5)]

    def write(self):
        with self.path.open('w') as out:
            writer=csv.writer(out)
            writer.writerow(('target','seed','warmup_seconds','sampling_seconds','warmup_steps'))
            writer.writerows(self.rows)

    def test_all_chains_and_warmup_work_in_denominator(self):
        self.write()
        row=aggregate(self.path,self.draws,self.diagnostics)[0]
        self.assertEqual(row['total_seconds'],12.)
        self.assertEqual(row['min_bulk_ess_per_sampling_second'],50.)
        self.assertEqual(row['min_bulk_ess_per_total_second'],400/12)
        self.assertEqual(row['min_bulk_ess_per_total_integrator_step'],400/(400+4*8192*3))

    def test_incomplete_timings_rejected(self):
        self.rows.pop()
        self.write()
        with self.assertRaises(ValueError):aggregate(self.path,self.draws,self.diagnostics)

    def test_nonfinite_timing_rejected(self):
        self.rows[0]=(0,501,float('inf'),2.,100)
        self.write()
        with self.assertRaises(ValueError):aggregate(self.path,self.draws,self.diagnostics)

    def test_failed_diagnostic_preserved(self):
        self.diagnostics[0]['passed']=False
        self.write()
        self.assertFalse(aggregate(self.path,self.draws,self.diagnostics)[0]['diagnostics_pass'])


if __name__=='__main__':unittest.main()
