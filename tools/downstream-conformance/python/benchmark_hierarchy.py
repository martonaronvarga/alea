"""Profile equivalent hierarchical logistic representations; retain every search outcome."""
import argparse
from decimal import Decimal as D, DecimalException, localcontext
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import time
from benchmark_regression import digest, source_fingerprints


def reference(position,groups):
    """Independent full transformed posterior and score, using Decimal elementary functions."""
    with localcontext() as context:
        context.prec=70
        q=[D.from_float(x) for x in position]
        log2pi=(2*D('3.141592653589793238462643383279502884197169399375105820974944592307816')).ln()
        scales=[D.from_float(.7),D.from_float(.3),D(1),D(1)]; locations=[D(0),D.from_float(-.7),D(0),D(0)]
        value=sum(-((x-m)/s)**2/2-s.ln()-log2pi/2 for x,m,s in zip(q,locations,scales))
        g=[-(x-m)/s**2 for x,m,s in zip(q,locations,scales)]
        tau=q[1].exp()
        for j,(events,trials) in enumerate(groups):
            eta=q[0]+tau*q[j+2]
            # Both tails calculated directly, avoiding k-n*p cancellation.
            e=(-abs(eta)).exp(); small=(1+e).ln()
            p,r=(1/(1+e),e/(1+e)) if eta>=0 else (e/(1+e),1/(1+e))
            logp,logr=(-small,-eta-small) if eta>=0 else (eta-small,-small)
            value+=D(events)*logp+D(trials-events)*logr
            score=D(events)*r-D(trials-events)*p
            g[0]+=score; g[1]+=score*tau*q[j+2]; g[j+2]+=score*tau
        return float(value),[float(x) for x in g]


def check(row):
    errors=[]
    if len(row['estimate'])!=4 or any(len(row[k])!=4 for k in ['gradient','final_gradient']):
        return False, None
    for q,v,g in [([.1,-.7,-.2,.3],row['value'],row['gradient']),
                  (row['estimate'],row['final_value'],row['final_gradient'])]:
        if not all(math.isfinite(x) for x in [*q,v,*g]): return False,None
        value,score=reference(q,row['groups'])
        errors.append(max([abs(value-v)]+[abs(x-y) for x,y in zip(score,g)]))
    return max(errors)<=1e-11*max(1,2*row['trials']),max(errors)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('binary',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args()
    planned=[dict(trials=n,mode=m,repeat=r) for n in [12,1000,10000] for r in range(3) for m in ['scalar','counts']]
    random.Random(271828).shuffle(planned)
    report=dict(protocol='hierarchy-counts-v1',binary_sha256=digest(args.binary),source_files=source_fingerprints(),
                source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                toolchain=subprocess.check_output(['rustc','-Vv'],text=True),planned_runs=planned,runs=[],comparisons=[])
    with args.output.with_suffix('.plan.json').open('x') as f: json.dump(report,f,indent=2)
    with args.output.open('x') as output,args.output.with_suffix('.jsonl').open('x') as raw:
        try:
            for job in planned:
                started=time.monotonic();row=dict(job)
                try:
                    process=subprocess.run([str(args.binary),job['mode'],str(job['trials']),str(max(10,min(5000,1000000//job['trials'])))],capture_output=True,text=True,timeout=300)
                    row.update(returncode=process.returncode,stdout=process.stdout,stderr=process.stderr)
                    if process.returncode: raise ValueError('process failure')
                    payload=json.loads(process.stdout)
                    if any(payload[k]!=job[k] for k in ['mode','trials']): raise ValueError('job mismatch')
                    row.update(payload)
                    row['reference_passed'],row['reference_error']=check(row)
                    row['attribution_valid']=row['target_calls']==row['evaluations'] and 0<=row['target_seconds']<=row['fit_seconds']
                except (ValueError,KeyError,TypeError,DecimalException,OSError,subprocess.TimeoutExpired) as e: row['error']=str(e)
                row['process_seconds']=time.monotonic()-started
                report['runs'].append(row);raw.write(json.dumps(row,allow_nan=False)+'\n');raw.flush()
            for n in [12,1000,10000]:
                cohorts={m:[r for r in report['runs'] if r['trials']==n and r['mode']==m] for m in ['scalar','counts']}
                numeric=all(len(v)==3 and all('error' not in r and r['reference_passed'] for r in v) for v in cohorts.values())
                result=dict(trials=n,numerical_checks_passed=numeric)
                if numeric:
                    for key in ['construction_seconds','evaluation_seconds']:
                        result[key+'_scalar_over_counts']=statistics.median(r[key] for r in cohorts['scalar'])/statistics.median(r[key] for r in cohorts['counts'])
                    result['score_stops']={m:sum(r['termination']=='ScoreTolerance' for r in v) for m,v in cohorts.items()}
                    allrows=cohorts['scalar']+cohorts['counts']
                    qualified=all(r['termination']=='ScoreTolerance' and max(map(abs,r['final_gradient']))<=r['score_tolerance'] for r in allrows)
                    qualified=qualified and all(all(math.isclose(a,b,rel_tol=1e-5,abs_tol=1e-6) for a,b in zip(allrows[0]['estimate'],r['estimate'])) for r in allrows)
                    result['fit_qualified']=qualified
                    result['target_fractions']={m:statistics.median(r['target_seconds']/r['fit_seconds'] for r in v) for m,v in cohorts.items() if all(r['attribution_valid'] for r in v)}
                    if qualified:
                        result['fit_seconds_scalar_over_counts']=statistics.median(r['fit_seconds'] for r in cohorts['scalar'])/statistics.median(r['fit_seconds'] for r in cohorts['counts'])
                report['comparisons'].append(result)
        finally:
            json.dump(report,output,indent=2,allow_nan=False);output.write('\n')
    print(json.dumps(report['comparisons'],indent=2))

if __name__=='__main__':main()
