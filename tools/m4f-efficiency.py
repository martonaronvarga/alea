"""Check the predeclared M4F efficiency artifact; no fitted thresholds or filtering."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

METHODS = ("covariance", "fisher_diagonal", "rank1_w20_h40", "rank2_w20_h40",
           "rank2_w80_h160", "weighted_dual", "weighted_rm", "weighted_adam")
TARGETS = ("Correlated", "Logistic", "Banana", "Funnel")
SEEDS = (3101, 3102, 3103, 3104, 4101, 4102, 4103, 4104)


def analyze(rows, references):
    keyed = {}
    for row in rows:
        key = (row['target'], row['method'], int(row['seed']), int(row['observable']))
        if key in keyed:
            raise ValueError(f'duplicate {key}')
        numbers = {k: float(v) for k, v in row.items() if k not in ('target', 'method')}
        if not all(math.isfinite(v) for v in numbers.values()):
            raise ValueError(f'nonfinite {key}')
        if not (0 < numbers['ess'] <= 8192.000001 and numbers['mcse'] > 0
                and numbers['variance'] > 0 and numbers['step'] > 0
                and numbers['total_seconds'] > 0
                and 0 < numbers['warmup_calls'] < numbers['total_calls']):
            raise ValueError(f'invalid diagnostics {key}')
        for field in ('warmup_calls', 'total_calls', 'warmup_divergences',
                      'retained_divergences', 'rank'):
            if numbers[field] < 0 or not numbers[field].is_integer():
                raise ValueError(f'invalid counter {key}: {field}')
        if numbers['warmup_divergences'] > 1000 or numbers['retained_divergences'] > 8192 or numbers['rank'] > 2:
            raise ValueError(f'out-of-budget counter {key}')
        if not math.isclose(numbers['mcse'] ** 2 * numbers['ess'], numbers['variance'], rel_tol=1e-8):
            raise ValueError(f'inconsistent diagnostics {key}')
        keyed[key] = numbers
    expected = {(t, m, s, o) for t in TARGETS for m in METHODS for s in SEEDS for o in range(5)}
    if keyed.keys() != expected:
        raise ValueError('missing or unexpected scenarios')
    for target in TARGETS:
        for method in METHODS:
            for seed in SEEDS:
                first = keyed[target, method, seed, 0]
                for obs in range(1, 5):
                    current = keyed[target, method, seed, obs]
                    for field in ('warmup_calls', 'total_calls', 'total_seconds',
                                  'warmup_divergences', 'retained_divergences', 'rank', 'step'):
                        if current[field] != first[field]:
                            raise ValueError(f'inconsistent chain metadata: {target}, {method}, {seed}, {field}')
    failures, gates, summary = [], [], []
    for target in TARGETS:
        for method in METHODS:
            for seeds in (SEEDS[:4], SEEDS[4:]):
                for obs in range(5):
                    sample = [keyed[target, method, s, obs] for s in seeds]
                    mean = statistics.mean(r['mean'] for r in sample)
                    se = max(math.sqrt(sum(r['mcse']**2 for r in sample))/4,
                             statistics.stdev(r['mean'] for r in sample)/2)
                    ref, ref_se = references[obs] if target == 'Logistic' else ((0, 0, 1, 1, 0)[obs], 0)
                    tolerance = 6 * math.hypot(se, ref_se) + .03
                    gate = dict(target=target, method=method, ensemble=seeds[0], observable=obs,
                                mean=mean, reference=ref, mcse=se, tolerance=tolerance,
                                passed=abs(mean-ref) <= tolerance)
                    gates.append(gate)
                    if not gate['passed']:
                        failures.append(gate)
            # Sum per-chain ESS rather than joining chain boundaries. Minimum
            # across all five predeclared observables includes second moments.
            ess = min(sum(keyed[target, method, s, o]['ess'] for s in SEEDS) for o in range(5))
            sample = [keyed[target, method, s, 0] for s in SEEDS]
            calls = sum(r['total_calls'] for r in sample)
            seconds = sum(r['total_seconds'] for r in sample)
            summary.append(dict(target=target, method=method, min_observable_ess=ess,
                                ess_per_gradient=ess/calls, ess_per_second=ess/seconds,
                                total_seconds=seconds, total_calls=int(calls),
                                divergences=int(sum(r['retained_divergences'] for r in sample)),
                                ranks=[int(r['rank']) for r in sample]))
    return dict(gates=gates, failures=failures, summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with Path('crates/alea-mcmc/tests/fixtures/blackjax-sampling.csv').open() as source:
        refs = {int(r['observable']): (float(r['mean']), float(r['mcse']))
                for r in csv.DictReader(line for line in source if not line.startswith('#'))
                if r['target'] == 'logistic' and r['mass'] == 'identity' and int(r['observable']) < 5}
    if len(refs) != 5:
        raise ValueError('incomplete independent reference')
    with args.csv.open() as source:
        result = analyze(list(csv.DictReader(source)), refs)
    result['sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                        [args.csv, Path(__file__), Path('crates/target/release/examples/fisher_efficiency')]}
    with args.output.open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    for row in result['summary']:
        print(f"{row['target']:10} {row['method']:18} ESS/grad={row['ess_per_gradient']:.5f} "
              f"ESS/s={row['ess_per_second']:.1f} div={row['divergences']} ranks={row['ranks']}")
    print(f"{len(result['gates'])} moment gates; {len(result['failures'])} failures")
    if result['failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
