"""Bounded repeated-fit workload with caller-owned pools and retained input IDs.

Cold fits construct their own model; reuse builds once per fixed worker partition.
Each fit evaluates the initial point twice and runs
L-BFGS. No likelihood-internal parallelism or external BLAS pool is used.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
from benchmark_regression import benchmark_command, decode_benchmark_output, digest, eligible_pair, source_fingerprints


def decode_trial(stdout, fits, threads, reuse=False):
    payload = json.loads(stdout)
    if not isinstance(payload, dict):
        raise ValueError('expected throughput object')
    if payload.get('fits') != fits or payload.get('threads') != threads or payload.get('max_live_models') != min(fits,threads):
        raise ValueError('pool configuration mismatch')
    if payload.get('execution') != ('sequential' if threads == 1 else 'parallel'):
        raise ValueError('execution policy mismatch')
    if payload.get('reuse_model') != reuse or payload.get('model_build_attempts') != (min(fits,threads) if reuse else fits):
        raise ValueError('model construction policy mismatch')
    rows = payload['runs']
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError('expected fit records')
    if len(rows) != fits or [r.get('fit_id') for r in rows] != list(range(fits)):
        raise ValueError('missing, duplicate or reordered fit IDs')
    if any(not math.isfinite(payload[k]) or payload[k] < 0 for k in ['pool_seconds','workload_seconds']):
        raise ValueError('invalid timing')
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('binary',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--columns',type=int,default=8)
    p.add_argument('--rows',type=int,nargs='+',default=[1000,10000])
    p.add_argument('--threads',type=int,nargs='+',default=[1,2,4])
    p.add_argument('--construction',nargs='+',choices=['cold','reuse'],default=['cold'])
    p.add_argument('--fits',type=int,default=12)
    p.add_argument('--runs',type=int,default=3)
    args = p.parse_args()
    if min([args.columns,args.fits,args.runs,*args.rows,*args.threads]) < 1 or len(set(args.rows)) != len(args.rows) or len(set(args.threads)) != len(args.threads) or len(set(args.construction)) != len(args.construction):
        p.error('positive counts and unique rows/threads required')
    trials = [dict(family=f,rows=n,threads=t,round=r,construction=c) for f in ['normal','bernoulli','poisson'] for n in args.rows for r in range(args.runs) for t in args.threads for c in args.construction]
    random.Random(271828).shuffle(trials)
    planned = [dict(family=t['family'],mode='batch',rows=t['rows'],columns=args.columns,repeat=i*args.fits+j,
                    trial_id=i,fit_id=j,threads=t['threads'],construction=t['construction'],score_tolerance_per_observation=1e-8)
               for i,t in enumerate(trials) for j in range(args.fits)]
    report = dict(protocol='alea-regression-throughput-v2',binary_sha256=digest(args.binary),
                  source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  source_files=source_fingerprints(),affinity=sorted(os.sched_getaffinity(0)),
                  toolchain=subprocess.check_output(['rustc','-Vv'],text=True),
                  planned_runs=planned,runs=[],trials=[])
    with args.output.with_suffix('.plan.json').open('x') as plan:
        json.dump(report,plan,indent=2); plan.write('\n')
    with args.output.open('x') as output, args.output.with_suffix('.jsonl').open('x') as raw:
        try:
            for i,trial in enumerate(trials):
                jobs = planned[i*args.fits:(i+1)*args.fits]
                command = benchmark_command(args.binary,trial['family'],'batch',trial['rows'],args.columns,score_tolerance=1e-8)
                command[4]='1'
                command.extend(['--fits',str(args.fits),'--threads',str(trial['threads'])])
                reuse=trial['construction']=='reuse'
                if reuse: command.append('--reuse-model')
                started=time.monotonic(); result=dict(trial,trial_id=i,command=command)
                try:
                    process=subprocess.run(command,capture_output=True,text=True,timeout=300)
                    result.update(returncode=process.returncode,stdout=process.stdout,stderr=process.stderr)
                    if process.returncode: raise ValueError('process failure')
                    payload=decode_trial(process.stdout,args.fits,trial['threads'],reuse)
                    rows=[]
                    for job,row in zip(jobs,payload['runs']):
                        if 'error' not in row:
                            decode_benchmark_output(json.dumps(row),job)
                        rows.append(dict(job,**row))
                    result.update(pool_seconds=payload['pool_seconds'],workload_seconds=payload['workload_seconds'],model_build_attempts=payload['model_build_attempts'])
                except (OSError,ValueError,KeyError,TypeError,subprocess.TimeoutExpired) as error:
                    result['error']=str(error)
                    rows=[dict(job,error=str(error)) for job in jobs]
                result['process_seconds']=time.monotonic()-started
                qualified=all('error' not in r and eligible_pair(rows[0],r) for r in rows)
                result['qualified']=qualified
                result['qualified_fits']=args.fits if qualified else 0
                result['qualified_fits_per_process_second']=result['qualified_fits']/result['process_seconds']
                report['runs'].extend(rows); report['trials'].append(result)
                raw.write(json.dumps(dict(result,runs=rows),allow_nan=False)+'\n'); raw.flush()
        finally:
            json.dump(report,output,indent=2,allow_nan=False); output.write('\n')
    print(json.dumps(dict(planned=len(planned),completed=len(report['runs']),qualified_trials=sum(t['qualified'] for t in report['trials']),trials=len(trials))))


if __name__=='__main__': main()
