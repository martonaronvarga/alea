"""Executed per-model AD oracle. Run with ALEA_THIRD_AD_LIBRARY set after build.
The earnings regression has an independent closed-form potential Hessian/tensor.
"""
import ctypes as c,itertools,json,math,os,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
@unittest.skipUnless(os.environ.get('ALEA_THIRD_AD_LIBRARY'),'requires compiled earnings third-AD helper')
class ThirdAd(unittest.TestCase):
 def setUp(self):
  self.lib=c.CDLL(os.environ['ALEA_THIRD_AD_LIBRARY'])
  self.lib.alea_third_abi.restype=c.c_uint
  self.lib.alea_third_new.argtypes=[c.c_char_p,c.c_void_p,c.c_size_t];self.lib.alea_third_new.restype=c.c_void_p
  self.lib.alea_third_drop.argtypes=[c.c_void_p]
  self.lib.alea_third_dimension.argtypes=[c.c_void_p];self.lib.alea_third_dimension.restype=c.c_size_t
  self.lib.alea_third_eval.argtypes=[c.c_void_p,c.POINTER(c.c_double),c.c_size_t,c.POINTER(c.c_double),c.c_size_t,c.POINTER(c.c_double),c.c_size_t,c.c_void_p,c.c_size_t]
  self.error=c.create_string_buffer(4096)
  self.data=json.loads((ROOT/'tools/bench/corpus/pdb-v2/earnings-earn_height/data.json').read_text())
  self.handle=self.lib.alea_third_new(os.fsencode(ROOT/'tools/bench/corpus/pdb-v2/earnings-earn_height/data.json'),self.error,4096)
  self.assertTrue(self.handle,self.error.value.decode());self.assertEqual(self.lib.alea_third_abi(),1);self.assertEqual(self.lib.alea_third_dimension(self.handle),3)
 def tearDown(self):self.lib.alea_third_drop(self.handle)
 def evaluate(self,q,nh=9):
  h=(c.c_double*9)(*([123.]*9));t=(c.c_double*27)(*([123.]*27));point=(c.c_double*3)(*q)
  status=self.lib.alea_third_eval(self.handle,point,3,h,nh,t,27,self.error,4096)
  return status,list(h),list(t)
 def test_independent_analytic_hessian_third_and_all_permutations(self):
  for q in [[.1,1.6,.97],[10000.,400.,9.],[0.,0.,0.]]:
   status,h,t=self.evaluate(q);self.assertEqual(status,0,self.error.value.decode())
   e=math.exp(-2*q[2]);x=[[1.,v] for v in self.data['height']]
   residual=[y-q[0]-q[1]*v for y,v in zip(self.data['earn'],self.data['height'])]
   xx=[[math.fsum(row[i]*row[j] for row in x)*e for j in range(2)] for i in range(2)]
   xr=[math.fsum(row[i]*r for row,r in zip(x,residual))*e for i in range(2)]
   rr=math.fsum(r*r for r in residual)*e
   expected=[xx[0][0],xx[0][1],2*xr[0],xx[1][0],xx[1][1],2*xr[1],2*xr[0],2*xr[1],2*rr]
   for actual,want in zip(h,expected):self.assertLessEqual(abs(actual-want),3e-12*(1+abs(want)))
   for k,i,j in itertools.product(range(3),repeat=3):
    indices=[k,i,j];count=indices.count(2)
    want=0. if count==0 else (-2*xx[[n for n in indices if n!=2][0]][[n for n in indices if n!=2][1]] if count==1 else (-4*xr[next(n for n in indices if n!=2)] if count==2 else -4*rr))
    actual=t[k*9+i*3+j];self.assertLessEqual(abs(actual-want),4e-12*(1+abs(want)))
    for a,b,d in itertools.permutations(indices):self.assertLessEqual(abs(actual-t[a*9+b*3+d]),1e-10*(1+abs(actual)))
 def test_bad_shape_publishes_no_output(self):
  status,h,t=self.evaluate([0.,0.,0.],8);self.assertEqual(status,1);self.assertEqual(h,[123.]*9);self.assertEqual(t,[123.]*27)
if __name__=='__main__':unittest.main()
