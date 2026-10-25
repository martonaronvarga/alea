"""Independent80-digit exponential-metric roots and bounded solver comparisons."""
from decimal import Decimal as D,localcontext
import csv,io,itertools,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def force(q,p):return -q-D('.5')+D('.5')*(-q).exp()*p*p
def scaled(a,b):return abs(a-b)/(1+max(abs(a),abs(b)))
def solve(q,p,h,method,tol,budget):
 half=p;res=D('Infinity');mi=pi=0
 for _ in range(budget):
  mi+=1;rhs=p+h/2*force(q,half);res=scaled(half,rhs)
  if method=='newton':
   if res<=tol:break
   jac=1-h/2*(-q).exp()*half
   if jac==0:return 'Momentum',q,p,half,mi,pi,res
   half-=(half-rhs)/jac
  else:half=rhs
 else:
  if method=='newton':return 'Momentum',q,p,half,mi,pi,res
 if res>tol:return 'Momentum',q,p,half,mi,pi,res
 end=q
 for _ in range(budget):
  pi+=1;rhs=q+h/2*half*((-q).exp()+(-end).exp());res=scaled(end,rhs)
  if method=='newton':
   if res<=tol:break
   jac=1+h/2*half*(-end).exp()
   if jac==0:return 'Position',q,p,half,mi,pi,res
   end-=(end-rhs)/jac
  else:end=rhs
 else:
  if method=='newton':return 'Position',q,p,half,mi,pi,res
 if res>tol:return 'Position',q,p,half,mi,pi,res
 momentum=half+h/2*force(end,half)
 return 'Success',end,momentum,half,mi,pi,res
def bisect(f,a,b):
 fa=f(a)
 for _ in range(280):
  m=(a+b)/2
  if (f(m)>0)==(fa>0):a=m
  else:b=m
 return (a+b)/2
def datasets():
 ladder=[];roots=[]
 with localcontext() as c:
  c.prec=80
  cases=[('.2','-.35','.04'),('-.5','.7','.1'),('.8','-1.1','.2')]
  for raw in cases:
   q,p,h=map(D,raw)
   for method,tol,budget in itertools.product(['newton','picard'],map(D,['1e-2','1e-6','1e-10','1e-12']),[1,4,100]):
    stage,end,momentum,half,mi,pi,res=solve(q,p,h,method,tol,budget)
    ladder.append(dict(q=q,p=p,h=h,method=method,tolerance=tol,budget=budget,stage=stage,end=end,momentum=momentum,half=half,momentum_iterations=mi,position_iterations=pi,residual=res))
   A=h/4*(-q).exp();B=p-h/2*(q+D('.5'));discriminant=(1-4*A*B).sqrt()
   for branch,half in [('near',2*B/(1+discriminant)),('far',(1+discriminant)/(2*A))]:
    f=lambda v:v-q-h/2*half*((-q).exp()+(-v).exp())
    if half>0:intervals=[(D(-100),D(100),'unique')]
    else:
     critical=(-h*half/2).ln()
     intervals=[(D(-100),critical,'left'),(critical,D(100),'right')] if f(critical)<0 else []
    for a,b,label in intervals:
     end=bisect(f,a,b);momentum=half+h/2*force(end,half)
     roots.append(dict(q=q,p=p,h=h,branch=branch+'-'+label,half=half,end=end,momentum=momentum))
 return {'solver-ladder.csv':ladder,'multiple-roots.csv':roots}
if __name__=='__main__':
 for name,rows in datasets().items():
  out=io.StringIO(newline='');writer=csv.DictWriter(out,list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
  path=ROOT/'crates/alea-mcmc/tests/fixtures/rmhmc'/name
  if '--check' in sys.argv:assert path.read_text()==out.getvalue(),name+' drift'
  else:path.write_text(out.getvalue())
  print(name,len(rows),'independent cases')
