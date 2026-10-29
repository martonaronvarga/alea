"""Prevent failed or numerically disagreeing controls from acquiring speed ratios."""
import copy
import unittest
from benchmark_regression import eligible_pair

class BenchmarkGates(unittest.TestCase):
    def test_only_matching_valid_results_are_eligible(self):
        baseline={"termination":"ScoreTolerance","value":-2.,"gradient":[1.,2.],"estimate_free":[0.,1.],"final_gradient":[1e-10,0.],"final_value":-1.,"score_tolerance":1e-8}
        self.assertTrue(eligible_pair(baseline,baseline))
        for key,value in [("termination","LineSearchFailed"),("termination","EvaluationLimit"),("value",-3.),("final_value",-3.),("final_gradient",[1.,0.]),("final_gradient",[float("nan"),0.]),("final_gradient",[0.]),("score_tolerance",1e-6),("gradient",[1.]),("gradient",[1.,3.]),("estimate_free",[0.]),("estimate_free",[0.,2.])]:
            bad=copy.deepcopy(baseline);bad[key]=value;self.assertFalse(eligible_pair(baseline,bad))

class RevisionGates(unittest.TestCase):
    def setUp(self):
        self.row = dict(termination="ScoreTolerance", value=-2., gradient=[1., 2.],
                        estimate_free=[0., 1.], final_gradient=[1e-10, 0.],
                        final_value=-1., score_tolerance=1e-8, iterations=10,
                        evaluations=15, line_search_failure="None",
                        construction_seconds=1., evaluation_seconds=2.,
                        fit_seconds=3., process_seconds=5., peak_rss_kib=100)

    def test_failure_or_missing_repeat_disables_all_ratios(self):
        from benchmark_regression import revision_comparison
        for before, after in [([], []), ([self.row], []),
                              ([self.row], [dict(error="timeout")]),
                              ([self.row], [dict(self.row, termination="LineSearchFailed")])]:
            self.assertEqual(revision_comparison(before, after), {"eligible": False})

    def test_identity_is_distinct_from_qualification_and_timing(self):
        from benchmark_regression import revision_comparison
        other = dict(self.row, fit_seconds=1.5)
        comparison = revision_comparison([self.row], [other])
        self.assertTrue(comparison["eligible"])
        self.assertTrue(comparison["paired_outputs_identical"])
        self.assertEqual(comparison["fit_seconds_before_over_after"], 2.)
        other["iterations"] = 11
        comparison = revision_comparison([self.row], [other])
        self.assertTrue(comparison["eligible"])
        self.assertFalse(comparison["paired_outputs_identical"])

    def test_malformed_output_and_wrong_jobs_are_infrastructure_failures(self):
        import json
        from benchmark_regression import decode_benchmark_output
        job = dict(family="normal", rows=10, mode="batch")
        payload = dict(self.row, **job)
        self.assertEqual(decode_benchmark_output(json.dumps(payload), job), payload)
        with self.assertRaises(ValueError):
            decode_benchmark_output(json.dumps(payload), dict(job, revision="timed"))
        for raw in ["[]", "null", "{}", '{"value":NaN}', "invalid",
                    json.dumps(dict(payload, rows=100))]:
            with self.assertRaises(ValueError):
                decode_benchmark_output(raw, job)

class AttributionGates(unittest.TestCase):
    def test_fraction_retains_observer_residual_and_rejects_missing_work(self):
        from benchmark_regression import attribution_summary
        row = dict(target_calls=10, evaluations=10, target_seconds=0.75, fit_seconds=1.)
        result = attribution_summary([row])
        self.assertTrue(result['attribution_valid'])
        self.assertEqual(result['target_fraction_median'], 0.75)
        self.assertEqual(result['residual_seconds_median'], 0.25)
        for bad in [dict(row, target_calls=9), dict(row, target_seconds=None),
                    dict(row, target_seconds=2.), dict(row, target_seconds=float('nan')),
                    dict(row, fit_seconds=0.)]:
            self.assertEqual(attribution_summary([bad]), {'attribution_valid': False})

class PlannedJobs(unittest.TestCase):
    def test_targeted_runs_preserve_complete_denominators_and_randomization(self):
        from benchmark_regression import planned_revision_jobs
        args = (['normal'], [1000000], 15, ['before', 'after'], 271828)
        jobs = planned_revision_jobs(*args)
        self.assertEqual(jobs, planned_revision_jobs(*args))
        self.assertEqual(len(jobs), 60)
        keys = {(r['family'],r['rows'],r['repeat'],r['revision'],r['mode']) for r in jobs}
        expected = {('normal',1000000,i,v,m) for i in range(15)
                    for v in ['before','after'] for m in ['batch','scalar']}
        self.assertEqual(keys, expected)

    def test_width_is_part_of_the_declared_cohort(self):
        import json
        from benchmark_regression import planned_revision_jobs, decode_benchmark_output
        jobs = planned_revision_jobs(['normal'], [1000], 3, ['before','after'], 7, columns=32)
        self.assertEqual(len(jobs), 12)
        self.assertTrue(all(j['columns'] == 32 for j in jobs))
        row = RevisionGates(); row.setUp()
        payload = dict(row.row, family='normal', mode='batch', rows=1000, columns=8)
        with self.assertRaises(ValueError):
            decode_benchmark_output(json.dumps(payload), jobs[0])
