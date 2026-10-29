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
