"""Independent dense reference derivatives, including the Gaussian log scale."""
import unittest
try:
    import numpy as np
    import scipy
except ImportError:
    np = None
from check_regression_benchmark import evaluate, reference


@unittest.skipIf(np is None, 'optional NumPy/SciPy analysis environment')
class DenseReference(unittest.TestCase):
    def test_scores_match_central_differences_in_every_coordinate(self):
        x = np.array([[1., -.7, .2], [1., .4, -.9], [1., .8, .6], [1., -.1, -.4]])
        for family, y in [('normal', [1., 2., -1., .5]), ('bernoulli', [0., 1., 0., 1.]), ('poisson', [1., 3., 0., 2.])]:
            q = np.array([.1, -.2, .3]+([-.4] if family == 'normal' else []))
            _, gradient = evaluate(family, x, np.array(y), q)
            for j, score in enumerate(gradient):
                delta = np.zeros(len(q)); delta[j] = 1e-5
                finite_difference = (evaluate(family,x,np.array(y),q+delta)[0]-evaluate(family,x,np.array(y),q-delta)[0])/2e-5
                self.assertAlmostEqual(float(score), finite_difference, places=7)

    def test_normal_reference_has_zero_score_and_mle_variance(self):
        x = np.column_stack([np.ones(9), np.arange(9)])
        y = np.array([1., 3., 2., 4., 3., 6., 5., 8., 7.])
        q, success, _ = reference('normal', x, y)
        self.assertTrue(success)
        self.assertLess(np.max(np.abs(evaluate('normal',x,y,q)[1])), 1e-12)
        self.assertAlmostEqual(np.exp(2*q[-1]), np.mean((y-x@q[:2])**2))


class ReferenceCohorts(unittest.TestCase):
    def setUp(self):
        self.job = dict(family='normal', mode='batch', rows=1000, columns=32, repeat=0)

    def test_revision_binds_binary_and_planned_denominator(self):
        from check_regression_benchmark import select_cohort
        before = dict(self.job, revision='before')
        after = dict(self.job, revision='after')
        report = dict(binaries={'before':'a','after':'b'}, planned_runs=[before,after], runs=[after,before])
        self.assertEqual(select_cohort(report,'a','before'),([before],1))
        with self.assertRaises(ValueError):
            select_cohort(report,'b','before')
        with self.assertRaises(ValueError):
            select_cohort(report,'a','timed')

    def test_duplicates_cannot_replace_missing_jobs(self):
        from check_regression_benchmark import select_cohort
        other = dict(self.job, repeat=1)
        report = dict(binary_sha256='a',planned_runs=[self.job,other],runs=[self.job,self.job])
        with self.assertRaises(ValueError):
            select_cohort(report,'a')
        report['runs']=[self.job,other]
        self.assertEqual(select_cohort(report,'a'),([self.job,other],2))

    def test_missing_and_unplanned_jobs_reject_even_at_equal_counts(self):
        from check_regression_benchmark import select_cohort
        other = dict(self.job, repeat=1)
        report = dict(binary_sha256='a', planned_runs=[self.job,other], runs=[self.job])
        with self.assertRaises(ValueError):
            select_cohort(report,'a')
        report['runs']=[self.job,dict(other,columns=8)]
        with self.assertRaises(ValueError):
            select_cohort(report,'a')
