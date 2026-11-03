import copy
import unittest
from regression_compare import compatible


class RegressionComparisonTests(unittest.TestCase):
    def test_scores_and_gradient_checksum_must_remain_exact(self):
        a = dict(evaluations=100, logp=-2., checksum=-200., gradient_checksum=5., gradient=[1., 2.])
        self.assertTrue(compatible(a,a))
        self.assertFalse(compatible(a,{}))
        for key,value in [('gradient',[1. + 1e-12,2.]), ('gradient_checksum',5.+1e-12),
                          ('evaluations',99), ('logp',float('nan'))]:
            b = copy.deepcopy(a); b[key]=value
            self.assertFalse(compatible(a,b))

    def test_density_rounding_is_tolerance_checked_not_assumed_exact(self):
        a = dict(evaluations=100, logp=-2., checksum=-200., gradient_checksum=5., gradient=[1., 2.])
        self.assertTrue(compatible(a,{**a,'logp':-2.+1e-15}))
        self.assertFalse(compatible(a,{**a,'logp':-1.}))

    def test_reassociated_scores_require_explicit_numerical_policy(self):
        a = dict(evaluations=100, logp=-2., checksum=-200., gradient_checksum=5., gradient=[1., 2.])
        b = {**a, 'gradient':[1.+1e-12,2.], 'gradient_checksum':5.+1e-12}
        self.assertFalse(compatible(a,b))
        self.assertTrue(compatible(a,b,'numerical'))
        self.assertFalse(compatible(a,{**b,'gradient':[1.1,2.]},'numerical'))


if __name__ == '__main__': unittest.main()
