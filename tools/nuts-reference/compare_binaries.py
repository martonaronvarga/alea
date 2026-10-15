"""Paired serial timings of saved nuts_reference binaries, with draw hashes."""
import argparse
import csv
import hashlib
import json
import os
import resource
from pathlib import Path
import statistics
import subprocess
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('binaries',type=Path,nargs='+')
    parser.add_argument('--rounds',type=int,default=5)
    args=parser.parse_args()
    if args.rounds < 1 or len({p.name for p in args.binaries}) != len(args.binaries):
        parser.error('positive rounds and unique binary names required')
    args.directory.mkdir(parents=True,exist_ok=False)
    records=[]
    for r in range(args.rounds):
        order=args.binaries if r%2==0 else list(reversed(args.binaries))
        for binary in order:
            raw=args.directory/f'{r}-{binary.name}.csv'
            timing=args.directory/f'{r}-{binary.name}.timing.csv'
            env=os.environ.copy();env['ALEA_BENCH_TIMINGS']=str(timing.resolve())
            start=time.perf_counter()
            before=resource.getrusage(resource.RUSAGE_CHILDREN)
            subprocess.run([str(binary.resolve()),str(raw)],env=env,check=True)
            after=resource.getrusage(resource.RUSAGE_CHILDREN)
            elapsed=time.perf_counter()-start
            with timing.open() as source:rows=list(csv.DictReader(source))
            if [(int(row['target']),int(row['seed'])) for row in rows] != [(t,s) for t in range(4) for s in range(501,505)]:
                raise ValueError('incomplete timing rows')
            phases=[sum(float(row['warmup_seconds'])+float(row['sampling_seconds']) for row in rows[t*4:t*4+4]) for t in range(4)]
            record=dict(round=r,binary=binary.name,process_seconds=elapsed,phases=phases,
                        cpu_seconds=(after.ru_utime-before.ru_utime)+(after.ru_stime-before.ru_stime),
                        draw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
            records.append(record)
            print(json.dumps(record),flush=True)
    medians={b.name:[statistics.median(r['phases'][t] for r in records if r['binary']==b.name) for t in range(4)] for b in args.binaries}
    artifact=dict(affinity=sorted(os.sched_getaffinity(0)),records=records,medians=medians,
                  binary_sha256={str(b):hashlib.sha256(b.read_bytes()).hexdigest() for b in args.binaries})
    with (args.directory/'results.json').open('x') as out:json.dump(artifact,out,indent=2)
    print(json.dumps(medians,indent=2))


if __name__=='__main__':main()
