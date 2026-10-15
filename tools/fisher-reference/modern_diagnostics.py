"""Independent ArviZ bulk/tail ESS and rank R-hat for the fixed Fisher experiment.

CSV plus optional raw output of fisher_efficiency. Failures are preserved, never
filtered. R-hat <= 1.01 and bulk/tail ESS >= 400 for each ensemble/observable are
predeclared diagnostic gates, not proofs of convergence or efficiency superiority.
"""
import argparse
import csv
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path
import arviz as az
import numpy as np


def diagnostic_record(rhat, bulk, tail):
    """Undefined diagnostics fail the gate and remain JSON-serializable."""
    finite = all(math.isfinite(x) for x in (rhat, bulk, tail))
    return dict(rhat=rhat if math.isfinite(rhat) else None,
                bulk_ess=bulk if math.isfinite(bulk) else None,
                tail_ess=tail if math.isfinite(tail) else None,
                passed=finite and rhat <= 1.01 and min(bulk, tail) >= 400,
                undefined=not finite)


def validate_draws(rows, raw):
    """Bind binary chain order to every CSV observable before computing gates."""
    targets = ('Correlated', 'Logistic', 'Banana', 'Funnel')
    methods = ('covariance', 'fisher_diagonal', 'rank1_w20_h40', 'rank2_w20_h40',
               'rank2_w80_h160', 'weighted_dual', 'weighted_rm', 'weighted_adam')
    seeds = (*range(3101, 3105), *range(4101, 4105))
    expected = {(t, m, s, o) for t in targets for m in methods for s in seeds for o in range(5)}
    indexed = {(r['target'], r['method'], int(r['seed']), int(r['observable'])): r for r in rows}
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError('incomplete/duplicate/unexpected chain metadata')
    keys = [(r['target'], r['method'], int(r['seed'])) for r in rows if r['observable'] == '0']
    if raw.shape != (256, 8192, 5) or not np.isfinite(raw).all():
        raise ValueError('invalid retained draw shape or values')
    for key, values in zip(keys, raw):
        for obs in range(5):
            row = indexed[*key, obs]
            for field in ('warmup_calls', 'total_calls', 'total_seconds', 'warmup_divergences',
                          'retained_divergences', 'rank', 'step'):
                if row[field] != indexed[*key, 0][field]:
                    raise ValueError('inconsistent per-chain metadata')
            # Independent reduction order differs from the Rust online moments.
            if not np.isclose(values[:, obs].mean(), float(row['mean']), rtol=1e-10, atol=1e-12):
                raise ValueError('CSV/raw draw mean mismatch')
            if not np.isclose(values[:, obs].var(ddof=1), float(row['variance']), rtol=1e-10, atol=1e-12):
                raise ValueError('CSV/raw draw variance mismatch')
            if int(row['total_calls']) <= 0 or not math.isfinite(float(row['total_seconds'])) or float(row['total_seconds']) <= 0:
                raise ValueError('invalid total cost')
    return keys, indexed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('draws', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.csv.open() as source:
        rows = list(csv.DictReader(source))
    raw = np.fromfile(args.draws, dtype='<f8').reshape(256, 8192, 5)
    keys, indexed = validate_draws(rows, raw)
    chains = dict(zip(keys, raw))
    results = []
    for target, method in dict.fromkeys((t, m) for t, m, _ in keys):
        for first in (3101, 4101):
            samples = np.stack([chains[target, method, s] for s in range(first, first+4)])
            total_calls = sum(int(indexed[target, method, s, 0]['total_calls']) for s in range(first, first+4))
            total_seconds = sum(float(indexed[target, method, s, 0]['total_seconds']) for s in range(first, first+4))
            for obs in range(5):
                values = samples[:, :, obs]
                rhat = float(az.rhat(values, method='rank'))
                bulk = float(az.ess(values, method='bulk'))
                tail = float(az.ess(values, method='tail'))
                results.append(dict(target=target, method=method, ensemble=first, observable=obs,
                                    **diagnostic_record(rhat, bulk, tail),
                                    total_calls=total_calls, total_seconds=total_seconds,
                                    bulk_ess_per_total_call=bulk/total_calls if math.isfinite(bulk) else None,
                                    bulk_ess_per_total_second=bulk/total_seconds if math.isfinite(bulk) else None))
    artifact = dict(versions={name: version(name) for name in ('arviz', 'numpy', 'scipy')},
                    sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (args.csv, args.draws, Path(__file__))},
                    results=results, failed=sum(not r['passed'] for r in results))
    with args.output.open('x') as out:
        json.dump(artifact, out, indent=2, allow_nan=False)
    print(f"{len(results)} diagnostic gates; {artifact['failed']} failures")
    if artifact['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
