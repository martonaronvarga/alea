"""Optional python-flint tests for pointwise certificate semantics."""
import unittest
from flint import arb,ctx
from wiener7_reference import certify
class Certificate(unittest.TestCase):
    def setUp(self):ctx.prec=128
    def test_exact_candidate_is_certified(self):
        result=certify([arb(1)]*8,[1.]*8)
        self.assertEqual(result['status'],'certified')
    def test_wrong_candidate_is_rejected(self):
        result=certify([arb(1)]*8,[1.01]*8)
        self.assertEqual(result['status'],'rejected')
    def test_overlap_is_not_a_rejection_or_certificate(self):
        result=certify([arb(1,.1)]*8,[1.1]*8,absolute='.05',relative='0')
        self.assertEqual(result['status'],'unresolved')
    def test_nonfinite_candidate_is_explicit(self):
        self.assertEqual(certify([arb(1)]*8,[float('nan')]*8)['status'],'nonfinite_candidate')
    def test_invalid_tolerance_is_explicit(self):
        with self.assertRaises(ValueError):certify([arb(1)]*8,[1.]*8,absolute='nan')
    def test_missing_enclosure_is_invalid(self):
        with self.assertRaises(ValueError):certify([], [1.]*8)

class VarianceCancellationEnvelope(unittest.TestCase):
    def test_large_drift_variance_slab_bounds_density_and_derivatives(self):
        from flint import acb
        from wiener7_reference import envelopes,point
        ctx.prec=160
        a,v,s=map(arb,[1.6,.5,350.])
        qlo,qhi=map(arb,['.02','.67'])
        c=a*a*qlo*qlo/16
        cutoff=c/(1+s*c.sqrt())
        bound=envelopes(a,v,s,qlo,qhi,cutoff,cutoff,0,True)
        for q in [qlo,(qlo+qhi)/2,qhi]:
            for t in [cutoff/4,cutoff/2,cutoff]:
                for component in range(8):
                    value=point(acb(t),acb(q),a,v,s,arb('.5'),arb(1),True,8,False,component).real
                    self.assertTrue(abs(value)<=bound[component],(q,t,component,value,bound[component]))

if __name__=='__main__':unittest.main()
