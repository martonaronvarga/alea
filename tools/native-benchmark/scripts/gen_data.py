#!/usr/bin/env python3
"""Fully deterministic, no nonstandard Python packages."""
from pathlib import Path
import json
import math
import random

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
print(f'Created data in {data_dir}')
