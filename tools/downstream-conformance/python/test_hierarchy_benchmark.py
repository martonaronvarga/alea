import unittest
import math
from benchmark_hierarchy import reference, check

class HierarchyReference(unittest.TestCase):
    def test_analytic_scores_match_independent_central_differences(self):
        groups=[(3,12),(10,12)]
        for q in [[.1,-.7,-.2,.3],[-2.,.2,1.5,-1.],[30.,-1.,-2.,-3.]]:
            value,gradient=reference(q,groups)
            for j in range(4):
                a=q.copy();b=q.copy();a[j]+=1e-5;b[j]-=1e-5
                fd=(reference(a,groups)[0]-reference(b,groups)[0])/2e-5
                self.assertAlmostEqual(gradient[j]/(1+abs(fd)),fd/(1+abs(fd)),places=7)
            initial_value,initial_gradient=reference([.1,-.7,-.2,.3],groups)
            row=dict(trials=12,groups=groups,value=initial_value,gradient=initial_gradient,estimate=q,final_value=value,final_gradient=gradient)
            self.assertTrue(check(row)[0])
            for bad in [dict(final_gradient=gradient[:3]),dict(estimate=q[:3]),dict(value=initial_value+1),dict(final_value=float('nan'))]:
                self.assertFalse(check(dict(row,**bad))[0])

    def test_empty_groups_reduce_to_normalized_priors(self):
        q=[0.,-.7,0.,0.]
        value,gradient=reference(q,[(0,0),(0,0)])
        self.assertEqual(gradient,[0.,0.,0.,0.])
        expected=-2*math.log(2*math.pi)-math.log(.7)-math.log(.3)
        self.assertAlmostEqual(value,expected,places=14)
