"""Independent harmonic-map ATLAS fixtures from Appendix E.2.
Uses the closed 2x2 leapfrog matrix and explicit auxiliary variates, never Alea.
"""
import csv,io,itertools,math,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def flow(z,h):
 q,p=z;a=1-h*h/2
 return (a*q+h*p,(-h+h*h*h/4)*q+a*p)
def energy(z):return sum(x*x for x in z)/2
def flip(z):return z[0],-z[1]
def trace(z,h,cap):
 states=[];current=z;distance=0
 for i in range(cap):
  states.append(current);current=flow(current,h);next_distance=(current[0]-z[0])**2
  if abs(energy(current)-energy(z))>10 or next_distance<=distance:return i+1,states
  distance=next_distance
 return cap,states
def select(length,offset,n):
 lo=max(1,math.floor(length*offset))
 return 1/(length-lo) if lo<=n<length else 0.
def logdensity(h,baseline):
 sigma=math.log(1.2);mu=math.log(min(.5,.5*baseline))-.5*sigma*sigma
 return -math.log(h*sigma*math.sqrt(2*math.pi))-(math.log(h)-mu)**2/(2*sigma*sigma)
def masslog(x):return math.log(x) if x>0 else -math.inf
def one(q,p,h,cap,offset,index,local,u1,u2):
 z=(q,p);H=energy(z);L,history=trace(z,h,cap);work=[L,0,0];hvps=0;stage='Preliminary';reason='None';alpha=0;new=z;F=R=-math.inf
 failure=L<3
 if not failure:
  lo=max(1,math.floor(L*offset));n=lo+math.floor(index*(L-lo));qs=select(L,offset,n)
  candidate=flip(history[n]);reverse,_=trace(candidate,h,cap);work[1]+=reverse;qr=select(reverse,offset,n) if reverse>=3 else 0
  F=-H+masslog(qs);R=-energy(candidate)+masslog(qr);a1=min(1.,math.exp(min(0.,R-F)))
  alpha=a1
  if u1<a1:return dict(qnew=candidate[0],accepted=1,stage=stage,reason=reason,alpha=alpha,forward=work[0],reverse=work[1],ghost=0,hvps=0,log_forward=F,log_reverse=R)
  if not qr:return dict(qnew=q,accepted=0,stage=stage,reason='SubUTurn',alpha=alpha,forward=work[0],reverse=work[1],ghost=0,hvps=0,log_forward=F,log_reverse=R)
 else:
  empirical=[3,min(4,cap),min(5,cap)];n=empirical[math.floor(index*3)]
 stage='Failure' if failure else 'Delayed';hvps=14
 count=math.floor(n*h/local)
 if failure:count=max(3,min(cap,count))
 if count<1 or count>cap:
  return dict(qnew=q,accepted=0,stage=stage,reason='WorkLimit',alpha=0.,forward=work[0],reverse=work[1],ghost=0,hvps=hvps,log_forward=0.,log_reverse=-math.inf)
 proposal=z
 for _ in range(count):proposal=flow(proposal,local)
 proposal=flip(proposal);work[0]+=count
 ghost_length,ghost_states=trace(proposal,h,cap);work[2]+=ghost_length
 support=ghost_length<3 if failure else ghost_length>=3 and select(ghost_length,offset,n)>0
 if failure:
  qn=empirical.count(n)/3;F=-H+masslog(qn)+logdensity(local,h);R=-energy(proposal)+masslog(qn)+logdensity(local,h)
 else:
  F=-H+masslog(qs)+math.log1p(-a1)+logdensity(local,h)
  R=-math.inf
  if support:
   ghost=flip(ghost_states[n]);reverse_length,_=trace(ghost,h,cap);work[2]+=reverse_length
   qghost=select(ghost_length,offset,n);qreverse=select(reverse_length,offset,n) if reverse_length>=3 else 0
   support=qreverse>0
   if support:
    a_reverse=math.exp(min(0.,energy(proposal)-energy(ghost)+masslog(qreverse)-masslog(qghost)))
    R=-energy(proposal)+masslog(qghost)+masslog(1-a_reverse)+logdensity(local,h)
 if not support:
  return dict(qnew=q,accepted=0,stage=stage,reason='ReverseBranch',alpha=0.,forward=work[0],reverse=work[1],ghost=work[2],hvps=hvps,log_forward=F,log_reverse=-math.inf)
 hvps=28;alpha=math.exp(min(0.,R-F));u=u1 if failure else u2;accepted=u<alpha
 if not accepted:reason='Metropolis'
 return dict(qnew=proposal[0] if accepted else q,accepted=int(accepted),stage=stage,reason=reason,alpha=alpha,forward=work[0],reverse=work[1],ghost=work[2],hvps=hvps,log_forward=F,log_reverse=R)
def generate():
 rows=[]
 for q,p,h,cap,local,u1,u2 in itertools.product([.2,2.],[-.4,1.3],[.2,.8,1.5,3.], [3,8,16],[.01,.3,1.2],[0.,.9999],[0.,.9999]):
  offset=.47;index=.38;r=one(q,p,h,cap,offset,index,local,u1,u2)
  F,R=r['log_forward'],r['log_reverse']
  # Compare both probability fluxes without tiny-tail underflow.
  if math.isfinite(F) and math.isfinite(R):
   forward=F+min(0.,R-F);reverse=R+min(0.,F-R)
   assert abs(forward-reverse)<1e-10
  rows.append(dict(q=q,p=p,baseline=h,cap=cap,offset=offset,index=index,local_step=local,u1=u1,u2=u2,**r))
 assert {'Preliminary','Delayed','Failure'}<={r['stage'] for r in rows}
 assert {'SubUTurn','ReverseBranch','WorkLimit','Metropolis'}<={r['reason'] for r in rows}
 assert any(r['stage']=='Delayed' and r['alpha']==0 and r['reason']=='Metropolis' for r in rows)
 out=io.StringIO(newline='');w=csv.DictWriter(out,list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 return out.getvalue()
if __name__=='__main__':
 import sys
 path=ROOT/'crates/alea-mcmc/tests/fixtures/atlas/harmonic-proposals.csv';data=generate()
 if '--check' in sys.argv:
  assert path.read_text()==data,'independent ATLAS fixture drift'
 else:path.write_text(data)
 print('independent ATLAS cases',len(data.splitlines())-1)
