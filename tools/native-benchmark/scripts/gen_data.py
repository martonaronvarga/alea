#!/usr/bin/env python3
"""Fully deterministic, no nonstandard Python packages."""
from pathlib import Path
import json
import math
import random
import os

root = Path(__file__).resolve().parents[1]
data_dir = root / 'data'
data_dir.mkdir(exist_ok=True)
rng = random.Random(20261029)


def write(name, n, k, xs, ys, stan_data):
    with (data_dir / f'{name}.txt').open('w') as f:
        f.write(f'{n} {k}\n')
        for row, y in zip(xs, ys):
            f.write(' '.join([*(format(x, '.17g') for x in row), format(y, '.17g')]) + '\n')
    (data_dir / f'{name}.json').write_text(json.dumps(stan_data, separators=(',', ':')))

n = 1000
ys = [0.4 + 0.7 * math.sin(1.23 * i) + 0.15 * math.cos(0.51 * i) for i in range(n)]
write('mean', n, 0, [[] for _ in ys], ys, {'N': n, 'y': ys})

for name, n, k in [('linear', 1000, 8), ('logistic', 1000, 8), ('linear_large', 10000, 8), ('logistic_large', 10000, 8)]:
    xs = [[math.sin((i + 1) * (j + 1) * 0.017) + 0.5 * math.cos((i + 3) * (j + 2) * 0.019)
           for j in range(k)] for i in range(n)]
    beta = [(-1 if j % 2 else 1) * 0.3 / (1 + j / 4) for j in range(k)]
    eta = [sum(x * b for x, b in zip(row, beta)) for row in xs]
    if name.startswith('linear'):
        ys = [e + 0.4 * math.sin(i * 0.71) for i, e in enumerate(eta)]
    else:
        ys = [float(rng.random() < 1 / (1 + math.exp(-e))) for e in eta]
    write(name, n, k, xs, ys, {'N': n, 'K': k, 'X': xs,
                              'y': [int(y) for y in ys] if name.startswith('logistic') else ys})
write('funnel', 0, 8, [], [], {'K': 8})
# Reuse the retained PosteriorDB data, not another corpus/registry.
school_path = Path(os.environ.get('ALEA_SCHOOLS_DATA', root.parents[1]/'tools/bench/corpus/vendor/eight_schools.json'))
if school_path.is_file():
    schools = json.loads(school_path.read_text())
    for name in ['schools','schools_nc']:
        xs = [[s] for s in schools['sigma']]
        write(name,schools['J'],1,xs,schools['y'],{'N':schools['J'],'K':1,'X':xs,'y':schools['y']})
else:
    print('Schools data not installed; set ALEA_SCHOOLS_DATA to the existing PosteriorDB JSON. No replacement data generated.')
xs = []
for i in range(32):
    a = (i%7)/4-.75
    xs.append([a,.5*a+(i%11)/8-.625])
write('covariance',32,2,xs,[0.]*32,{'N':32,'K':2,'X':xs,'y':[0.]*32})
print(f'Created data in {data_dir}')

# Optional varied-coordinate controls, using the same dimensions/registry as
# the existing runner. Never replace the historical timing loop implicitly.
if state_dir := os.environ.get('ALEA_BENCH_STATE_DIR'):
    from bench import DIMENSIONS
    destination=Path(state_dir)
    destination.mkdir(parents=True,exist_ok=True)
    for case,dimension in DIMENSIONS.items():
        states=[[0.3*math.sin((i+1)*(j+1)*0.37) for j in range(dimension)] for i in range(10)]
        if case=='funnel':
            for i,state in enumerate(states): state[0]=(i-4.5)/2
        (destination/f'{case}.txt').write_text('\n'.join(' '.join(format(v,'.17g') for v in state) for state in states)+'\n')
