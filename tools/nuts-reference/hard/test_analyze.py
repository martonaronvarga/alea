import tempfile
import unittest
from pathlib import Path
from analyze import clean,load
from models import np,CASES,DRAWS,SEEDS,generative,data

class Contracts(unittest.TestCase):
    def test_incomplete_draws_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'draws.csv';p.write_text('header\n1,2,3\n')
            with self.assertRaises(ValueError):load(p,0)

    def test_duplicate_draws_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'draws.csv'
            a=np.zeros((4*DRAWS,CASES[0][1]+4));a[:,3]=1
            a[:,:2]=np.array([(s,i) for s in SEEDS for i in range(DRAWS)])
            a[1,:2]=a[0,:2]
            np.savetxt(p,a,delimiter=',',header='header')
            with self.assertRaises(ValueError):load(p,0)

    def test_generative_inverse(self):
        rng=np.random.default_rng(77)
        for case in (0,1,2,4):
            z=rng.normal(size=(2,3,CASES[case][1]));q=z.copy()
            if case<=1:
                rho=.95 if case==0 else .9
                for i in range(1,q.shape[-1]):q[...,i]=rho*q[...,i-1]+np.sqrt(1-rho*rho)*z[...,i]
            elif case==2:q[...,1::2]+=.8*(q[...,::2]**2-1)
            else:q[...,0]*=3;q[...,1:]*=np.exp(.5*q[...,0,None])
            np.testing.assert_allclose(generative(case,q),z,atol=2e-14)

    def test_undefined_diagnostics_remain_null(self):
        self.assertEqual(clean(dict(ess=float('nan'),passed=False)),dict(ess=None,passed=False))

    def test_logistic_data_shape_and_labels(self):
        dataset=data(3)
        self.assertEqual(np.asarray(dataset['X']).shape,(512,32))
        self.assertEqual(len(dataset['y']),512)
        self.assertEqual(set(dataset['y']),{0,1})

if __name__=='__main__':unittest.main()
