"""Run all harder-target CPU timings serially, then analyze all saved draws."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
from models import CASES
from analyze import clean,summarize

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BACKENDS=('alea','stan','blackjax','nuts-rs')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--stan',type=Path,required=True)
    args=p.parse_args();args.directory.mkdir(parents=True,exist_ok=False)
    binaries={'alea':ROOT/'crates/target/release/examples/hard_nuts',
              'nuts-rs':ROOT/'tools/nuts-reference/upstream/target/release/hard','stan':args.stan.resolve()}
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',STAN_NUM_THREADS='1',
        XDG_CACHE_HOME='/tmp/alea-reference-cache',MPLCONFIGDIR='/tmp/alea-reference-matplotlib')
    records=[]
    for r in range(3):
        for case,(name,_,_) in enumerate(CASES):
            start_index=(r+case)%4;order=BACKENDS[start_index:]+BACKENDS[:start_index]
            for backend in order:
                stem=args.directory/f'{r}-{name}-{backend}';raw=stem.with_suffix('.csv');timing=stem.with_suffix('.timing.csv')
                command=([str(binaries[backend])] if backend in ('alea','nuts-rs') else [sys.executable,str(HERE/f'{backend}_run.py')])
                command += [str(case),str(raw.resolve()),str(timing.resolve())]
                if backend=='stan':command.append(str(binaries['stan']))
                before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.perf_counter()
                rss=stem.with_suffix('.rss.txt')
                with stem.with_suffix('.log').open('x') as log:
                    result=subprocess.run(['/run/current-system/sw/bin/time','-f','%M','-o',str(rss.resolve()),*command],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                elapsed=time.perf_counter()-start;after=resource.getrusage(resource.RUSAGE_CHILDREN)
                record=dict(round=r,case=case,name=name,backend=backend,command=command,returncode=result.returncode,
                    process_seconds=elapsed,cpu_seconds=after.ru_utime-before.ru_utime+after.ru_stime-before.ru_stime,
                    process_peak_rss_kib=int(rss.read_text().splitlines()[-1]),
                    raw=str(raw),timing=str(timing))
                records.append(record)
                with (args.directory/'progress.jsonl').open('a') as out:out.write(json.dumps(record)+'\n')
                print(f'round {r} {name} {backend}: {elapsed:.3f}s exit={result.returncode}',flush=True)
    for record in records:
        if record['returncode']:continue
        try:
            record['metrics']=summarize(record['raw'],record['timing'],record['case'])
            record['draw_sha256']=hashlib.sha256(Path(record['raw']).read_bytes()).hexdigest()
            compile_path=Path(record['timing']).with_suffix('.compile.json')
            record['compile_seconds']=json.loads(compile_path.read_text())['seconds'] if compile_path.exists() else 0.
        except Exception as error:record['analysis_error']=str(error)
    comparisons=[]
    for r in range(3):
        for case in range(len(CASES)):
            group={rec['backend']:rec['metrics'] for rec in records if rec['round']==r and rec['case']==case and 'metrics' in rec}
            if 'alea' not in group:continue
            for backend in BACKENDS[1:]:
                if backend not in group:continue
                for i,(a,b) in enumerate(zip(group['alea']['raw_means'],group[backend]['raw_means'],strict=True)):
                    limit=6*math.hypot(a['mcse'],b['mcse'])+.05;error=abs(a['mean']-b['mean'])
                    comparisons.append(dict(round=r,case=case,reference=backend,coordinate=i,error=error,limit=limit,
                                            passed=math.isfinite(error) and math.isfinite(limit) and error<=limit))
    summary=[]
    for case in range(len(CASES)):
        for backend in BACKENDS:
            group=[r for r in records if r['case']==case and r['backend']==backend and 'metrics' in r]
            if len(group)!=3:summary.append(dict(case=case,backend=backend,complete=False));continue
            metrics={key:statistics.median(r['metrics'][key] for r in group) for key in
                ('warmup_seconds','sampling_seconds','total_seconds','max_rhat','min_bulk_ess','min_tail_ess','ess_per_total_second','divergences','saturated_draws')}
            metrics.update(case=case,backend=backend,complete=True,diagnostics_pass=all(r['metrics']['diagnostics_pass'] for r in group),
                process_seconds=statistics.median(r['process_seconds'] for r in group),compile_seconds=statistics.median(r['compile_seconds'] for r in group))
            summary.append(metrics)
    sources=list(HERE.glob('*.*'))+[ROOT/'crates/alea-mcmc/examples/hard_nuts.rs',ROOT/'tools/nuts-reference/upstream/src/bin/hard.rs']
    artifact=dict(protocol='hard CPU v1',platform=platform.platform(),cpu=Path('/proc/cpuinfo').read_text(),affinity=sorted(os.sched_getaffinity(0)),
        records=records,comparisons=comparisons,summary=summary,
        binary_sha256={str(v):hashlib.sha256(v.read_bytes()).hexdigest() for v in binaries.values()},
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    with (args.directory/'results.json').open('x') as out:json.dump(clean(artifact),out,indent=2,allow_nan=False)
    print(json.dumps(clean(summary),indent=2))

if __name__=='__main__':main()
