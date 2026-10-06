"""Artifact integrity and failure-preservation tests for the fixed M4F protocol."""
import copy
import csv
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('efficiency', Path(__file__).with_name('m4f-efficiency.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        with (ROOT / 'crates/alea-mcmc/tests/fixtures/blackjax-sampling.csv').open() as source:
            self.refs = {int(r['observable']): (float(r['mean']), float(r['mcse']))
                         for r in csv.DictReader(line for line in source if not line.startswith('#'))
                         if r['target'] == 'logistic' and r['mass'] == 'identity' and int(r['observable']) < 5}
        # Analyzer contracts must run in a clean checkout without local benchmark
        # artifacts. These synthetic rows are not sampling evidence.
        self.rows = [dict(target=t, method=m, seed=str(s), observable=str(o),
                          mean=str(self.refs[o][0] if t == 'Logistic' else (0, 0, 1, 1, 0)[o]),
                          variance='1', ess='100', mcse='0.1', warmup_calls='100',
                          total_calls='1000', total_seconds='1', warmup_divergences='0',
                          retained_divergences='0', rank='0', step='0.1')
                     for t in module.TARGETS for m in module.METHODS
                     for s in module.SEEDS for o in range(5)]

    def test_complete_synthetic_artifact_passes_all_moment_gates(self):
        result = module.analyze(self.rows, self.refs)
        self.assertEqual(len(result['gates']), 320)
        self.assertEqual(len(result['summary']), 32)
        self.assertFalse(result['failures'])

    def test_missing_or_duplicate_outcomes_are_not_silently_omitted(self):
        for rows in [self.rows[:-1], self.rows + [self.rows[0]]]:
            with self.assertRaises(ValueError):
                module.analyze(rows, self.refs)

    def test_nonfinite_or_invalid_cost_is_rejected(self):
        for field, value in [('mean', 'nan'), ('ess', '-1'), ('total_seconds', '0')]:
            rows = copy.deepcopy(self.rows)
            rows[0][field] = value
            with self.assertRaises(ValueError):
                module.analyze(rows, self.refs)

    def test_failed_moment_gate_is_retained(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            if row['target'] == 'Correlated' and row['method'] == 'covariance':
                row['mean'] = '1000'
        result = module.analyze(rows, self.refs)
        self.assertEqual(len(result['failures']), 10)

    def test_inconsistent_diagnostics_and_chain_metadata_are_rejected(self):
        for field, value in [('total_calls', '1001'), ('retained_divergences', '-1'),
                             ('rank', '0.5'), ('mcse', '0.2'), ('variance', '-1')]:
            rows = copy.deepcopy(self.rows)
            rows[0][field] = value
            with self.assertRaises(ValueError):
                module.analyze(rows, self.refs)


if __name__ == '__main__':
    unittest.main()
