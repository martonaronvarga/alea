"""Independent reference checks on analytically tractable likelihoods."""
from decimal import Decimal as D, localcontext
import unittest
import check_regression_fits as reference

class ReferenceTests(unittest.TestCase):
    def test_glm_reference_matches_analytic_intercept_optima(self):
        with localcontext() as ctx:
            ctx.prec=60
            # x=1 is a duplicate intercept: only a+b is identified.
            for family,sy,expected in [('logistic',D(3),D(3).ln()),('poisson',D(6),D('1.5').ln())]:
                rows=[(D(1),D(0),D(4),sy,D(0),D(0))]
                q=reference.reference(family,rows)
                self.assertLess(abs(q[0]+q[1]-expected),D('1e-24'))
                self.assertLess(max(abs(v) for v in reference.evaluate(family,rows,q)[1]),D('1e-24'))

    def test_gaussian_reference_and_scores_match_closed_form(self):
        with localcontext() as ctx:
            ctx.prec=60
            rows=[(D(-1),D(0),D(2),D(0),D(2),D(0)),(D(1),D(0),D(2),D(4),D(10),D(0))]
            q=reference.reference('gaussian',rows)
            self.assertEqual(q,[D(1),D(1),D(0)])
            self.assertEqual(reference.evaluate('gaussian',rows,q)[1],[D(0),D(0),D(0)])

    def test_false_optimizer_success_is_rejected(self):
        r={'family':'logistic','scenario':'ordinary','rows':33,'position':[0.,0.],
           'gradient':[1.,0.],'value':-20.,'score_tolerance':1e-8,'termination':'ScoreTolerance'}
        with self.assertRaises(AssertionError): reference.assess(r)

if __name__=='__main__': unittest.main()
