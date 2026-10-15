"""Run with the pinned ArviZ reference environment."""
import json
import unittest

import numpy as np

from modern_diagnostics import diagnostic_record, validate_draws


class DiagnosticContracts(unittest.TestCase):
    @staticmethod
    def fixture():
        keys = [(t, m, s) for t in ('Correlated', 'Logistic', 'Banana', 'Funnel')
                for m in ('covariance', 'fisher_diagonal', 'rank1_w20_h40', 'rank2_w20_h40',
                          'rank2_w80_h160', 'weighted_dual', 'weighted_rm', 'weighted_adam')
                for s in (*range(3101, 3105), *range(4101, 4105))]
        raw = np.broadcast_to(np.arange(1280).reshape(256, 1, 5), (256, 8192, 5))
        rows = [dict(target=t, method=m, seed=str(s), observable=str(o), mean=str(i*5+o),
                     variance='0', total_calls='100', warmup_calls='10', total_seconds='1',
                     warmup_divergences='0', retained_divergences='0', rank='0', step='0.1')
                for i, (t, m, s) in enumerate(keys) for o in range(5)]
        return rows, raw

    def test_complete_draw_metadata_binding(self):
        rows, raw = self.fixture()
        keys, _ = validate_draws(rows, raw)
        self.assertEqual(len(keys), 256)
        for field, value, message in (('mean', '9999', 'mean mismatch'),
                                      ('variance', '1', 'variance mismatch'),
                                      ('total_calls', '99', 'inconsistent')):
            changed = [dict(row) for row in rows]
            changed[1][field] = value
            with self.assertRaisesRegex(ValueError, message):
                validate_draws(changed, raw)
        with self.assertRaisesRegex(ValueError, 'metadata'):
            validate_draws(rows + rows[:1], raw)
        with self.assertRaisesRegex(ValueError, 'mean mismatch'):
            validate_draws(rows, raw[::-1])

    def test_gate_boundaries_are_inclusive(self):
        self.assertTrue(diagnostic_record(1.01, 400, 400)['passed'])
        for values in ((1.010001, 400, 400), (1, 399.9, 400), (1, 400, 399.9)):
            self.assertFalse(diagnostic_record(*values)['passed'])

    def test_undefined_diagnostics_are_preserved_as_failures(self):
        for values in ((float('nan'), 500, 500), (1, float('inf'), 500), (1, 500, float('nan'))):
            result = diagnostic_record(*values)
            self.assertFalse(result['passed'])
            self.assertTrue(result['undefined'])
            json.dumps(result, allow_nan=False)

    def test_incomplete_metadata_is_rejected_before_draw_processing(self):
        with self.assertRaisesRegex(ValueError, 'metadata'):
            validate_draws([], np.empty((0,)))


if __name__ == '__main__':
    unittest.main()
