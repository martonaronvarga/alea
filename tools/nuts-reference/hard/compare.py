"""Paired serial hard-target binaries; preserve raw draws and require exact replay."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import time


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--cases', type=int, nargs='+', default=list(range(5)))
    args = parser.parse_args()
    if args.rounds < 1 or len(set(args.cases)) != len(args.cases) or any(c not in range(5) for c in args.cases):
        parser.error('positive rounds and unique cases in 0..4 required')
    binaries = {'baseline': args.baseline.resolve(), 'candidate': args.candidate.resolve()}
    hashes = {name: digest(path) for name, path in binaries.items()}
    args.directory.mkdir(parents=True, exist_ok=False)
    records = []
    for repeat in range(args.rounds):
        for case in args.cases:
            order = list(binaries) if (repeat + case) % 2 == 0 else list(reversed(binaries))
            for name in order:
                stem = args.directory / f'{repeat}-{case}-{name}'
                raw, timing = stem.with_suffix('.csv'), stem.with_suffix('.timing.csv')
                before = resource.getrusage(resource.RUSAGE_CHILDREN)
                start = time.perf_counter()
                with stem.with_suffix('.log').open('x') as log:
                    subprocess.run([str(binaries[name]), str(case), str(raw), str(timing)],
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                process = time.perf_counter() - start
                after = resource.getrusage(resource.RUSAGE_CHILDREN)
                with timing.open() as source:
                    rows = list(csv.DictReader(source))
                if [int(row['seed']) for row in rows] != list(range(701, 705)):
                    raise ValueError('incomplete timing rows')
                record = dict(round=repeat, case=case, binary=name, process_seconds=process,
                              cpu_seconds=after.ru_utime-before.ru_utime+after.ru_stime-before.ru_stime,
                              warmup_seconds=sum(float(row['warmup_seconds']) for row in rows),
                              sampling_seconds=sum(float(row['sampling_seconds']) for row in rows),
                              draw_sha256=digest(raw))
                record['total_seconds'] = record['warmup_seconds'] + record['sampling_seconds']
                records.append(record)
                with (args.directory / 'progress.jsonl').open('a') as out:
                    out.write(json.dumps(record) + '\n')
                print(json.dumps(record), flush=True)
    identical = all(len({r['draw_sha256'] for r in records if r['case'] == case}) == 1 for case in args.cases)
    summary = [dict(case=case, binary=name,
                    median=statistics.median(r['total_seconds'] for r in records if r['case'] == case and r['binary'] == name),
                    minimum=min(r['total_seconds'] for r in records if r['case'] == case and r['binary'] == name),
                    maximum=max(r['total_seconds'] for r in records if r['case'] == case and r['binary'] == name))
               for case in args.cases for name in binaries]
    artifact = dict(records=records, summary=summary, identical_draws=identical,
                    binaries={k: str(v) for k, v in binaries.items()}, binary_sha256=hashes,
                    affinity=sorted(os.sched_getaffinity(0)), cpu=Path('/proc/cpuinfo').read_text())
    with (args.directory / 'results.json').open('x') as out:
        json.dump(artifact, out, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))
    if not identical:
        raise SystemExit('draws changed: retain results and investigate before claiming exact replay')


if __name__ == '__main__':
    main()
