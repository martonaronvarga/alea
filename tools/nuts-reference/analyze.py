"""Evaluate the predeclared M5 gates without selecting chains or dropping failures."""
import argparse
import csv
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path
import arviz as az
import numpy as np

BACKENDS=('alea','stan','blackjax','nuts-rs')

def load(path):
    data=np.loadtxt(path,delimiter=',',skiprows=1)
    if data.shape!=(4*4*8192,8) or not np.isfinite(data).all():
        raise ValueError('invalid reference draw shape or values')
    expected=np.array([(t,s,d) for t in range(4) for s in range(501,505) for d in range(8192)])
    if not np.array_equal(data[:,:3],expected):
        raise ValueError('missing, duplicate, or reordered reference rows')
    if not np.isin(data[:,5],[0,1]).all() or not ((data[:,6]>=1)&(data[:,6]<=255)&(data[:,6]%1==0)).all():
        raise ValueError('invalid divergence or edge metadata')
    return data.reshape(4,4,8192,8)

def summarize(data,backend):
    results=[]
    for t in range(4):
        q=data[t,:,:,3:5]
        x,y=q[:,:,0],q[:,:,1]
        if t==0:y=(y-0.6*x)/0.8
        if t==1:y=y-0.4*(x*x-1)
        if t==3:y=y*np.exp(-0.5*x)
        obs=np.stack((x,y,x*x,y*y,x*y),axis=-1)
        for o in range(5):
            v=obs[:,:,o]
            rhat=float(az.rhat(v,method='rank'));bulk=float(az.ess(v,method='bulk'));tail=float(az.ess(v,method='tail'))
            mcse=max(float(v.reshape(-1,b).mean(axis=1).std(ddof=1)/np.sqrt(v.size/b)) for b in (256,512))
            mcse=max(mcse,float(v.mean(axis=1).std(ddof=1)/2))
            passed=bool(np.isfinite([rhat,bulk,tail,mcse]).all() and rhat<=1.01 and min(bulk,tail)>=400)
            results.append(dict(backend=backend,target=t,observable=o,mean=float(v.mean()),mcse=mcse,rhat=rhat,
                                bulk_ess=bulk,tail_ess=tail,passed=passed,divergences=int(data[t,:,:,5].sum()),
                                mean_steps=float(data[t,:,:,6].mean())))
    return results

def serializable(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:serializable(v) for k,v in value.items()}
    if isinstance(value,list):return [serializable(v) for v in value]
    return value

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for backend in BACKENDS:parser.add_argument('--'+backend,type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fixture',type=Path)
    args=parser.parse_args()
    if args.output.exists() or (args.fixture and args.fixture.exists()):raise FileExistsError('choose new output paths')
    paths={b:getattr(args,b.replace('-','_')) for b in BACKENDS}
    results=[r for b,p in paths.items() for r in summarize(load(p),b)]
    indexed={(r['backend'],r['target'],r['observable']):r for r in results}
    comparisons=[]
    for t in range(4):
        for o in range(5):
            a=indexed['alea',t,o]
            for b in BACKENDS[1:]:
                ref=indexed[b,t,o];error=abs(a['mean']-ref['mean']);limit=6*math.hypot(a['mcse'],ref['mcse'])+.03
                comparisons.append(dict(target=t,observable=o,reference=b,error=error,limit=limit,passed=math.isfinite(error) and error<=limit))
            if t!=2:
                for b in BACKENDS:
                    r=indexed[b,t,o];error=abs(r['mean']-[0,0,1,1,0][o]);limit=6*r['mcse']+.03
                    comparisons.append(dict(target=t,observable=o,reference=b+'-analytic',error=error,limit=limit,passed=math.isfinite(error) and error<=limit))
    failed=sum(not r['passed'] for r in results+comparisons)
    artifact=dict(protocol='M5 four-target v1',versions={p:version(p) for p in ('arviz','numpy','scipy')},
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (*paths.values(),Path(__file__))},
                  results=results,comparisons=comparisons,failed=failed)
    with args.output.open('x') as out:json.dump(serializable(artifact),out,indent=2,allow_nan=False)
    if args.fixture:
        with args.fixture.open('x',newline='') as out:
            writer=csv.DictWriter(out,fieldnames=list(results[0]));writer.writeheader();writer.writerows(serializable(results))
    print(f'{len(results)} diagnostic and {len(comparisons)} moment gates; {failed} failures')
    if failed:raise SystemExit(1)

if __name__=='__main__':main()
