#!/usr/bin/env python3
"""Replay the original Gaussian rows and seed at a deterministic size grid."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--current', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = (ROOT/'data/linear_large.txt').read_text().splitlines()[1:]
    with (args.output/'records.jsonl').open('x') as stream:
        for n in [1000,2500,5000,7500,9000,9250,9500,9750,10000]:
            path = args.output/f'n{n}.txt'
            path.write_text(f'{n} 8\n'+'\n'.join(rows[:n])+'\n')
            for engine,binary in [('baseline',args.baseline),('current',args.current)]:
                p = subprocess.run([str(binary.resolve()),'linear',str(path.resolve())],
                    env={**os.environ,'ALEA_BENCH_TIMED':'1'},text=True,capture_output=True,check=False)
                record = dict(n=n,engine=engine,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
                stream.write(json.dumps(record)+'\n')
                stream.flush()
                print(n,engine,p.returncode,flush=True)


if __name__ == '__main__':
    main()
