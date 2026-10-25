"""Independent80-digit fixed-point roots for the Newton generalized-leapfrog test.
Target U=q^2/2, metric G=exp(q); input rows are fixed and never tuned to Rust.
"""
from decimal import Decimal,localcontext
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def generate():
 rows=[]
 with localcontext() as context:
  context.prec=80
  for a,b,step in [('0.2','-0.35','0.04'),('-0.5','0.7','0.1'),('0.8','-1.1','0.2')]:
   q,p,h=map(Decimal,(a,b,step));half=p
   for _ in range(200):half=p+h/2*(-q-Decimal('.5')+Decimal('.5')*(-q).exp()*half*half)
   end=q
   for _ in range(200):end=q+h/2*half*((-q).exp()+(-end).exp())
   momentum=half+h/2*(-end-Decimal('.5')+Decimal('.5')*(-end).exp()*half*half)
   rows.append(','.join(map(str,(q,p,h,end,momentum,half))))
 return '\n'.join(rows)+'\n'
if __name__=='__main__':
 path=ROOT/'crates/alea-mcmc/tests/fixtures/rmhmc/exponential-decimal.csv'
 if path.read_text()!=generate():raise SystemExit('independent Decimal fixture drift')
 print('independent implicit roots verified')
