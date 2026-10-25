"""Deterministic corpus construction; --check verifies the reviewed data bytes."""
import argparse
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent


def build():
    cases=[]
    def add(name,kind,d,rows,data,partition,step,initial=None,family=None,provenance='Alea controlled formula v1'):
        cases.append(dict(id=name,kind=kind,dimension=d,rows=rows,data=data,partition=partition,step=step,
                          initial=initial or [.1*(-1 if i%2 else 1) for i in range(d)],
                          family=family or name,measure='Lebesgue on unconstrained q; parameter Jacobians included once; parameter-independent constants omitted except Wiener likelihood',provenance=provenance))
    add('gaussian_tune',0,2,0,[1.,1.],'tuning',.25,family='controlled')
    add('correlated_heldout',1,4,0,[.8],'heldout',.15,family='controlled')
    add('ill_conditioned_heldout',0,4,0,[.03,.2,1.,5.],'heldout',.01,family='pathological')
    add('banana_tune',2,2,0,[.4],'tuning',.2,family='pathological')
    add('funnel_heldout',3,5,0,[3.],'heldout',.05,family='pathological')
    data=[]
    for i in range(16):
        row=[1.,((17*i)%31)/15.-1.,((13*i)%29)/14.-1.]
        data.extend(row+[float((7*i)%17<9)])
    add('logistic_heldout',4,3,16,data,'heldout',.15,family='regression')
    add('hierarchical_normal',5,4,8,[v for i in range(8) for v in (i%2,(-.6 if i%2 else .3)+.05*i)],'heldout',.08,initial=[.1,-.7,.1,-.1],family='hierarchical')
    add('stochastic_volatility',6,8,8,[.85,.3]+[.3*math.sin(i)+.5 for i in range(8)],'heldout',.07,family='latent')
    add('latent_gaussian',7,4,4,[-.2,.4,.7,-.5],'heldout',.2,family='latent')
    data=[]
    for i in range(8):
        rt=.5+.08*i;t=rt-.15
        base=math.pi/(1.2**2)*sum(k*math.sin(k*math.pi*.5)*math.exp(-k*k*math.pi**2*t/(2*1.2**2)) for k in range(1,100))
        data.extend((rt,1. if i%3 else -1.,math.log(base)))
    add('wiener_drift',8,1,8,data,'heldout',.15,family='wiener',provenance='Wiener4 alpha=1.2,tau=.15,beta=.5; drift inferred; independent spectral zero-drift density')
    add('hierarchical_ddm_boundary',9,4,12,[v for i in range(12) for v in (i%2,float(i%3!=0))],'heldout',.1,initial=[.1,-.7,.1,-.1],family='wiener')
    vendor=json.loads((HERE/'corpus/vendor/eight_schools.json').read_text())
    data=[v for pair in zip(vendor['y'],vendor['sigma']) for v in pair]
    source=json.loads((HERE/'corpus/vendor/provenance.json').read_text())
    for kind,name,partition in [(10,'eight_schools_centered','tuning'),(11,'eight_schools_noncentered','heldout')]:
        add(name,kind,10,8,data,partition,.12,initial=[0.,1.]+[0.]*8,family='posteriordb',provenance=f"posteriordb {source['revision']} {name}; log(tau) Jacobian included")
    return dict(schema_version=1,id='reference-corpus-v1',cases=cases,vendor=source)

if __name__=='__main__':
    args=argparse.ArgumentParser();args.add_argument('--check',action='store_true');options=args.parse_args()
    content=json.dumps(build(),indent=2,allow_nan=False)+'\n';path=HERE/'corpus/v1.json'
    if options.check:
        if path.read_text()!=content: raise SystemExit('corpus differs from reviewed deterministic generation')
    else: path.write_text(content)
