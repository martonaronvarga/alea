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

class CombinedImages(unittest.TestCase):
    def test_combined_images_match_independent_uncombined_wiener5_series(self):
        from flint import acb
        from wiener7_reference import point
        from generate_wiener_balls import evaluate
        for lower in [False,True]:
            for t in [1e-5,.001,.03]:
                reference,_,_=evaluate(t,1.6,0.,.3,.5,350.,lower)
                a,beta,v,sd,time=map(arb,[1.6,.3,.5,350.,t])
                q=beta if lower else 1-beta
                effective_v=-v if lower else v
                values=[point(acb(time),acb(q),a,effective_v,sd,arb('.25'),arb('.75'),lower,8,False,i).real for i in range(8)]
                actual=[values[0].log()]+[x/values[0] for x in values[1:]]
                expected=reference+[reference[3]*arb('.25')*(1 if lower else -1),reference[2]*arb('.75')]
                for x,y in zip(actual,expected):self.assertTrue(abs(x-y)<arb('1e-20')*(1+abs(y)))

class BoundaryFluxReference(unittest.TestCase):
    def test_time_flux_reference_overlaps_independent_saved_integrals(self):
        import csv
        from pathlib import Path
        from wiener7_reference import evaluate
        with Path(__file__).with_name('wiener7_nested_balls.csv').open() as stream:rows=list(csv.reader(stream))
        x=list(map(float,rows[3]))
        values,_=evaluate(x[:8]+[bool(x[8])],goal='1e-12',max_precision=96)
        for i,value in enumerate(values):self.assertTrue(value.overlaps(arb(x[11+2*i],x[12+2*i])))

class AnalyticStartReference(unittest.TestCase):
    def test_start_antiderivative_enclosures_overlap_direct_complex_integrals(self):
        from flint import acb
        from wiener7_reference import point
        from wiener7_start_reference import start_point
        ctx.prec=160
        for time,sd,width in [(.03,.5,.1),(.285,350.,.63),(.001,350.,.63),(.03+.003j,.5,.1)]:
            a,v,m,s,h=map(arb,[1.6,.5,.34,sd,width]);t=acb(time)
            for lower in [False,True]:
                for component in range(8):
                    actual=start_point(t,m,h,a,v,s,lower,2,False,component,arb('.5'))
                    def f(x,analytic):return point(t,acb(m)+h*x,a,v,s,x,arb('.5'),lower,2,analytic,component)
                    expected=acb.integral(f,arb('-.5'),arb('.5'),abs_tol=arb('1e-24'),rel_tol=arb('1e-22'),eval_limit=4000)
                    self.assertTrue(actual.is_finite() and expected.is_finite() and actual.overlaps(expected),(time,lower,component))

class GaussianDriftCdfReference(unittest.TestCase):
    def test_cdf_time_average_overlaps_independent_continuous_integrals(self):
        import csv
        from pathlib import Path
        from wiener5_cdf_reference import evaluate
        with Path(__file__).with_name('wiener7_nested_balls.csv').open() as stream:rows=list(csv.reader(stream))[1:]
        for row in rows:
            x=list(map(float,row))
            values,_=evaluate(x[:8]+[bool(x[8])],goal='1e-14')
            for i,value in enumerate(values):self.assertTrue(value.overlaps(arb(x[11+2*i],x[12+2*i])))

    def test_cdf_time_derivative_matches_independent_large_variance_density(self):
        from flint import acb
        from wiener5_cdf_reference import point,tails
        from generate_wiener_balls import evaluate
        for lower in [False,True]:
            for t in [.001,.03,.3]:
                reference,_,_=evaluate(t,1.6,0.,.3,.5,350.,lower)
                a,beta,v,sd,time=map(arb,[1.6,.3,.5,350.,t])
                q=beta if lower else 1-beta
                effective_v=-v if lower else v
                derivative=point(time,a,acb(q),effective_v,sd,lower,8,2).real
                error=tails(time,a,effective_v,sd,q,q,8)[2]
                self.assertTrue((-derivative+arb(0,error.upper())).overlaps(reference[0].exp()))

if __name__=='__main__':unittest.main()
