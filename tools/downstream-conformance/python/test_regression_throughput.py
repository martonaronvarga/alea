import unittest
import json
from benchmark_regression_throughput import decode_trial

class PoolRecords(unittest.TestCase):
    def test_ids_and_bounds_are_checked_before_qualification(self):
        good=dict(execution="parallel",fits=3,threads=2,max_live_models=2,pool_seconds=.01,workload_seconds=.1,
                  runs=[dict(fit_id=i) for i in range(3)])
        self.assertEqual(decode_trial(json.dumps(good),3,2),good)
        for change in [dict(execution="sequential"),dict(fits=2),dict(threads=4),dict(max_live_models=3),
                       dict(runs=[dict(fit_id=0)]*3),dict(runs=good['runs'][::-1]),
                       dict(workload_seconds=-1)]:
            with self.assertRaises(ValueError):
                decode_trial(json.dumps(dict(good,**change)),3,2)

    def test_malformed_records_are_infrastructure_failures(self):
        for raw in ['[]', 'null', '{"fits":3,"threads":2,"max_live_models":2,"runs":[null]}']:
            with self.assertRaises(ValueError):
                decode_trial(raw,3,2)
