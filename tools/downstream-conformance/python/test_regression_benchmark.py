"""Prevent failed or numerically disagreeing controls from acquiring speed ratios."""
import copy
import unittest
from benchmark_regression import eligible_pair

class BenchmarkGates(unittest.TestCase):
    def test_only_matching_valid_results_are_eligible(self):
        baseline={"termination":"ScoreTolerance","value":-2.,"gradient":[1.,2.],"estimate_free":[0.,1.]}
        self.assertTrue(eligible_pair(baseline,baseline))
        for key,value in [("termination","LineSearchFailed"),("termination","EvaluationLimit"),("value",-3.),("gradient",[1.]),("gradient",[1.,3.]),("estimate_free",[0.]),("estimate_free",[0.,2.])]:
            bad=copy.deepcopy(baseline);bad[key]=value;self.assertFalse(eligible_pair(baseline,bad))
