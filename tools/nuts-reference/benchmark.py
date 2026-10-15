"""Serial three-round M5 CPU benchmark; never times builds or diagnostic analysis."""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time

import analyze

ROOT = Path(__file__).resolve().parents[2]
BACKENDS = analyze.BACKENDS


def aggregate(timing, draws, diagnostics):
    with timing.open() as source:
        rows = list(csv.DictReader(source))
    if [(int(r['target']), int(r['seed'])) for r in rows] != [(t,s) for t in range(4) for s in range(501,505)]:
        raise ValueError('incomplete timing rows')
    result=[]
    for t in range(4):
        subset=rows[t*4:(t+1)*4]
        warm=sum(float(r['warmup_seconds']) for r in subset)
        sample=sum(float(r['sampling_seconds']) for r in subset)
        warm_steps=sum(int(r['warmup_steps']) for r in subset)
        if not (math.isfinite(warm) and math.isfinite(sample) and warm > 0 and sample > 0 and warm_steps > 0):
            raise ValueError('invalid timings')
        gates=[r for r in diagnostics if r['target']==t]
        ess=min(r['bulk_ess'] for r in gates)
        steps=int(draws[t,:,:,6].sum())
        result.append(dict(target=t, warmup_seconds=warm, sampling_seconds=sample,
            total_seconds=warm+sample, min_bulk_ess=ess,
            min_bulk_ess_per_sampling_second=ess/sample,
            min_bulk_ess_per_total_second=ess/(warm+sample),
            min_bulk_ess_per_total_integrator_step=ess/(warm_steps+steps),
            warmup_steps=warm_steps, sampling_steps=steps,
            divergences=int(draws[t,:,:,5].sum()), diagnostics_pass=all(r['passed'] for r in gates)))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--stan',type=Path,required=True)
    args=parser.parse_args()
    args.directory.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',STAN_NUM_THREADS='1',
               XDG_CACHE_HOME='/tmp/alea-reference-cache',MPLCONFIGDIR='/tmp/alea-reference-matplotlib')
    records=[]
    executables={'alea': ROOT/'crates/target/release/examples/nuts_reference',
                 'nuts-rs':ROOT/'tools/nuts-reference/upstream/target/release/alea-nuts-upstream-reference',
                 'stan':args.stan.resolve()}
    for round_id in range(3):
        # Rotate order to avoid always assigning one backend the coldest machine.
        order=BACKENDS[round_id:]+BACKENDS[:round_id]
        for backend in order:
            stem=args.directory/f'{round_id}-{backend}'
            output=stem.with_suffix('.csv'); timing=stem.with_suffix('.timing.csv')
            env['ALEA_BENCH_TIMINGS']=str(timing.resolve())
            if backend in ('alea','nuts-rs'):
                command=[str(executables[backend]),str(output)]
            elif backend=='stan':
                command=[sys.executable,str(ROOT/'tools/nuts-reference/stan_run.py'),str(executables['stan']),str(stem)+'-raw',str(output)]
            else:
                command=[sys.executable,str(ROOT/'tools/nuts-reference/benchmark_blackjax.py'),str(output)]
            start=time.perf_counter()
            subprocess.run(command,check=True,env=env,cwd=ROOT)
            elapsed=time.perf_counter()-start
            records.append(dict(round=round_id,backend=backend,process_seconds=elapsed,
                                draws=str(output),timing=str(timing)))
            print(f'{round_id} {backend}: {elapsed:.3f}s process time',flush=True)
    # Analysis is deliberately after ALL timed runs.
    for record in records:
        draws=analyze.load(Path(record['draws']))
        diagnostics=analyze.summarize(draws,record['backend'])
        record['targets']=aggregate(Path(record['timing']),draws,diagnostics)
        record['draw_sha256']=hashlib.sha256(Path(record['draws']).read_bytes()).hexdigest()
        compile_path=Path(record['timing']).with_suffix('.compile.json')
        record['jit_compile_seconds']=sum(r['seconds'] for r in json.loads(compile_path.read_text())) if compile_path.exists() else 0
    summaries=[]
    for backend in BACKENDS:
        runs=[r for r in records if r['backend']==backend]
        summaries.append(dict(backend=backend,
            process_seconds=statistics.median(r['process_seconds'] for r in runs),
            jit_compile_seconds=statistics.median(r['jit_compile_seconds'] for r in runs),
            targets=[{k:(all(r['targets'][t][k] for r in runs) if k=='diagnostics_pass' else
                         statistics.median(r['targets'][t][k] for r in runs)) for k in runs[0]['targets'][t]} for t in range(4)]))
    artifact=dict(protocol='M5 CPU timing v1; 3 serial rotated rounds; 4x(1024+8192) per target',
                  platform=platform.platform(),cpu=Path('/proc/cpuinfo').read_text(),
                  affinity=sorted(os.sched_getaffinity(0)),records=records,summary=summaries,
                  binary_sha256={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in executables.items()},
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [Path(__file__),ROOT/'tools/nuts-reference/benchmark_blackjax.py',ROOT/'tools/nuts-reference/stan_run.py',
                                ROOT/'crates/alea-mcmc/examples/nuts_reference.rs',ROOT/'tools/nuts-reference/upstream/src/main.rs']})
    with (args.directory/'results.json').open('x') as out:
        json.dump(analyze.serializable(artifact),out,indent=2,allow_nan=False)
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
