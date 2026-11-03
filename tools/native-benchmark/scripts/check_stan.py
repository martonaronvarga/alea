#!/usr/bin/env python3
"""Compare native posteriors to CmdStan log_prob on shared free coordinates.

Uses the engines' production density/gradient APIs, not surrogate benchmark
densities. Subtracts each engine's density at zero to remove constant terms.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def close(a,b,tolerance):
    return math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tolerance*(1+abs(b))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--alea', type=Path, default=ROOT/'target/release/alea-stan-bench')
    parser.add_argument('--cases', nargs='+', choices=['mean','linear','linear_large','logistic','logistic_large'])
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    checks = []
    for case,model,dimension in [('mean','mean',2),('linear','linear',9),('linear_large','linear',9),
                                 ('logistic','logistic',8),('logistic_large','logistic',8)]:
        if args.cases is not None and case not in args.cases:
            continue
        points = [[0.]*dimension, [0.3]*dimension, [-1.5]*dimension]
        if model=='logistic':
            points += [[1000.]*dimension,[-1000.]*dimension]
        else:
            points += [[0.2]*(dimension-1)+[-20.]]
        native_file = args.output/f'{case}-native-points.json'
        stan_file = args.output/f'{case}-stan-points.json'
        native_file.write_text(json.dumps(points))
        stan_file.write_text(json.dumps(dict(params_r=points)))
        native = subprocess.run([str(args.alea.resolve()),model,str(ROOT/'data'/f'{case}.txt')],
            env={**os.environ,'ALEA_BENCH_PROBES':str(native_file.resolve())},text=True,capture_output=True,check=False)
        (args.output/f'{case}-alea.stdout').write_text(native.stdout)
        (args.output/f'{case}-alea.stderr').write_text(native.stderr)
        native.check_returncode()
        actual = json.loads(native.stdout)
        output = args.output/f'{case}-stan.csv'
        command = [str(ROOT/'stan'/model),'log_prob',f'unconstrained_params={stan_file.resolve()}',
            'jacobian=1','data',f'file={ROOT/"data"/f"{case}.json"}','output',f'file={output.resolve()}',
            f'profile_file={output.with_suffix(".profile.csv").resolve()}','sig_figs=17']
        stan = subprocess.run(command,text=True,capture_output=True,check=False)
        (args.output/f'{case}-stan.stdout').write_text(stan.stdout)
        (args.output/f'{case}-stan.stderr').write_text(stan.stderr)
        stan.check_returncode()
        with output.open() as stream:
            expected = list(csv.reader(line for line in stream if not line.startswith('#')))[1:]
        stan_values = [list(map(float,r)) for r in expected]
        for i,(a,b) in enumerate(zip(actual,stan_values,strict=True)):
            density_ok = close(a['logp']-actual[0]['logp'],b[0]-stan_values[0][0],2e-10)
            gradient_ok = all(close(x,y,3e-9) for x,y in zip(a['gradient'],b[1:],strict=True))
            checks.append(dict(case=case,point=i,density_ok=density_ok,gradient_ok=gradient_ok,
                maximum_absolute_gradient_error=max(abs(x-y) for x,y in zip(a['gradient'],b[1:],strict=True))))
    (args.output/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    assert all(c['density_ok'] and c['gradient_ok'] for c in checks),checks
    print(f'{len(checks)} shared-coordinate density and gradient checks passed')


if __name__ == '__main__':
    main()
