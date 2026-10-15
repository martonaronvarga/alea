"""Generate offline M6 fixtures with independently executed, Nix-pinned ArviZ."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import warnings

import arviz as az
import numpy as np
import scipy


def fixtures():
    rng = np.random.default_rng(20261013)
    iid = rng.normal(size=(4, 256))
    ar = rng.normal(size=(4, 256))
    for i in range(1, 256):
        ar[:, i] += 0.95 * ar[:, i - 1]
    anti = rng.normal(size=(4, 256))
    for i in range(1, 256):
        anti[:, i] -= 0.8 * anti[:, i - 1]
    cases = dict(iid=iid, correlated=ar, antithetic=anti,
                 shifted=iid + np.arange(4)[:, None] * 2,
                 scales=iid * np.array([0.1, 1, 2, 10])[:, None],
                 odd=rng.normal(size=(4, 33)), short=rng.normal(size=(2, 4)),
                 ties=rng.integers(-8, 9, size=(4, 128)).astype(float),
                 heavy=rng.standard_t(3, size=(4, 256)),
                 drift=iid + np.linspace(0, 4, 256),
                 constant=np.ones((4, 32)),
                 stuck=np.repeat(np.arange(4)[:, None], 32, axis=1))
    cases['ragged'] = [iid[0, :32], iid[1, :31]]
    bad = iid[:, :16].copy()
    bad[1, 3] = np.nan
    cases['nonfinite'] = bad
    draws, expected = io.StringIO(), io.StringIO()
    dw, ew = csv.writer(draws, lineterminator='\n'), csv.writer(expected, lineterminator='\n')
    dw.writerow(['case', 'chain', 'draw', 'value'])
    ew.writerow(['case', 'status', 'rhat', 'rank_rhat', 'folded_rhat', 'bulk_ess', 'tail_ess', 'mean_ess', 'mcse_mean', 'mean', 'ebfmi0'])
    for name, values in cases.items():
        for chain, row in enumerate(values):
            for draw, value in enumerate(row):
                dw.writerow([name, chain, draw, repr(float(value))])
        if name == 'ragged':
            ew.writerow([name, 'shape'] + [''] * 9)
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            reference = [az.rhat(values), az.rhat(values, method='z_scale'),
                         az.rhat(values, method='folded'), az.ess(values, method='bulk'),
                         az.ess(values, method='tail'), az.ess(values, method='mean'),
                         az.mcse(values, method='mean'), np.mean(values), az.bfmi(values)[0]]
        status = 'nonfinite' if name == 'nonfinite' else ('undefined' if name in ('constant', 'stuck') else 'ok')
        ew.writerow([name, status] + [repr(float(v)) for v in reference])
    return draws.getvalue(), expected.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    assert az.__version__ == '0.23.4', az.__version__
    root = Path(__file__).resolve().parents[2]
    destination = root / 'crates/alea-runtime/tests/fixtures'
    draws, expected = fixtures()
    files = {'modern-draws.csv': draws, 'modern-expected.csv': expected}
    metadata = dict(arviz=az.__version__, numpy=np.__version__, scipy=scipy.__version__,
                    seed=20261013, sha256={k: hashlib.sha256(v.encode()).hexdigest() for k, v in files.items()})
    files['modern-provenance.json'] = json.dumps(metadata, indent=2) + '\n'
    for name, content in files.items():
        path = destination / name
        if args.check:
            assert path.read_text() == content, f'stale fixture: {path}'
        else:
            destination.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    print(metadata)


if __name__ == '__main__':
    main()
