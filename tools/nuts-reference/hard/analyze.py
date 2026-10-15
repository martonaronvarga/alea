"""Fail-closed multi-coordinate diagnostics; no chain or divergence filtering."""
import csv
import math
from pathlib import Path
import arviz as az
from models import np,CASES,DRAWS,DEPTH,SEEDS,generative

def clean(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list):return [clean(v) for v in value]
    return value

def load(path,case):
    d=CASES[case][1];a=np.loadtxt(path,delimiter=',',skiprows=1)
    if a.shape!=(4*DRAWS,d+4) or not np.isfinite(a).all():raise ValueError('incomplete or nonfinite draws')
    expected=np.array([(s,i) for s in SEEDS for i in range(DRAWS)])
    if not np.array_equal(a[:,:2],expected):raise ValueError('duplicate/missing/reordered draws')
    if not np.isin(a[:,2],[0,1]).all():raise ValueError('invalid divergence flags')
    if not ((a[:,3]>=1)&(a[:,3]<2**DEPTH)&(a[:,3]%1==0)).all():raise ValueError('invalid step counts')
    return a.reshape(4,DRAWS,d+4)

def mcse(v):
    batches=v.reshape(-1,128).mean(axis=1)
    return max(float(batches.std(ddof=1)/np.sqrt(batches.size)),float(v.mean(axis=1).std(ddof=1)/2))

def summarize(path,timing,case):
    a=load(path,case);q=a[:,:,4:];z=generative(case,q)
    observables=[(f'q{i}',q[:,:,i]) for i in range(q.shape[-1])]
    observables.append(('mean_square_radius',np.mean(q*q,axis=-1)))
    if case!=3:observables.extend((f'z{i}',z[:,:,i]) for i in range(z.shape[-1]))
    diagnostics=[]
    for name,v in observables:
        rhat=float(az.rhat(v,method='rank'));bulk=float(az.ess(v,method='bulk'));tail=float(az.ess(v,method='tail'))
        passed=bool(np.isfinite([rhat,bulk,tail]).all() and rhat<=1.01 and min(bulk,tail)>=400)
        diagnostics.append(dict(observable=name,rhat=rhat,bulk_ess=bulk,tail_ess=tail,passed=passed))
    moments=[]
    if case!=3:
        for i in range(z.shape[-1]):
            for power in [1,2]:
                v=z[:,:,i]**power;mean=float(v.mean());se=mcse(v);limit=6*se+.05;truth=float(power==2)
                moments.append(dict(coordinate=i,power=power,mean=mean,mcse=se,limit=limit,
                                    passed=math.isfinite(mean) and math.isfinite(se) and abs(mean-truth)<=limit))
    with Path(timing).open() as source:rows=list(csv.DictReader(source))
    if [int(r['seed']) for r in rows]!=list(SEEDS):raise ValueError('incomplete timing rows')
    warm=sum(float(r['warmup_seconds']) for r in rows);sample=sum(float(r['sampling_seconds']) for r in rows)
    warmsteps=sum(int(r['warmup_steps']) for r in rows)
    if not (math.isfinite(warm) and math.isfinite(sample) and warm>0 and sample>0 and warmsteps>0):raise ValueError('invalid timing data')
    finite=all(np.isfinite([r['rhat'],r['bulk_ess'],r['tail_ess']]).all() for r in diagnostics)
    ess=min(r['bulk_ess'] for r in diagnostics) if finite else float('nan')
    return dict(warmup_seconds=warm,sampling_seconds=sample,total_seconds=warm+sample,
        max_rhat=max(r['rhat'] for r in diagnostics) if finite else float('nan'),
        min_bulk_ess=ess,min_tail_ess=min(r['tail_ess'] for r in diagnostics) if finite else float('nan'),
        ess_per_total_second=ess/(warm+sample),ess_per_sampling_second=ess/sample,
        ess_per_total_step=ess/(warmsteps+a[:,:,3].sum()),
        divergences=int(a[:,:,2].sum()),saturated_draws=int((a[:,:,3]==2**DEPTH-1).sum()),
        retained_steps=int(a[:,:,3].sum()),warmup_steps=warmsteps,
        actual_target_calls=sum(int(r['setup_calls'])+int(r['warmup_calls'])+int(r['sampling_calls']) for r in rows) if 'setup_calls' in rows[0] else None,
        diagnostics=diagnostics,diagnostics_pass=all(r['passed'] for r in diagnostics),
        analytic_moments=moments,analytic_moments_pass=all(r['passed'] for r in moments) if moments else None,
        raw_means=[dict(mean=float(q[:,:,i].mean()),mcse=mcse(q[:,:,i])) for i in range(q.shape[-1])])
