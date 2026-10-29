"""Check benchmark datasets and initial/final scores with independent SciPy fits.

Run in the analysis Nix shell. Dataset exports come from the measured executable;
no attempt is made to reproduce Rust's random-number generator in Python.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def evaluate(family, x, y, q):
    import numpy as np
    from scipy.special import expit, gammaln
    eta = x @ q[:x.shape[1]]
    if family == 'normal':
        residual = y-eta
        variance = np.exp(2*q[-1])
        value = np.sum(-q[-1]-.5*np.log(2*np.pi)-.5*residual**2/variance)
        score = np.r_[x.T @ (residual/variance), np.sum(residual**2/variance-1)]
    elif family == 'bernoulli':
        value = np.sum(np.where(y == 1, -np.logaddexp(0, -eta), -np.logaddexp(0, eta)))
        score = x.T @ (y-expit(eta))
    else:
        value = np.sum(y*eta-np.exp(eta)-gammaln(y+1))
        score = x.T @ (y-np.exp(eta))
    return float(value), score


def reference(family, x, y):
    import numpy as np
    from scipy import linalg, optimize
    from scipy.special import expit
    if family == 'normal':
        beta, _, rank, _ = linalg.lstsq(x, y, lapack_driver='gelsd')
        return np.r_[beta, .5*np.log(np.mean((y-x@beta)**2))], bool(rank == x.shape[1]), 'SciPy gelsd'
    def score(q):
        return evaluate(family, x, y, q)[1]
    def jacobian(q):
        eta = x@q
        mean = expit(eta) if family == 'bernoulli' else np.exp(eta)
        weight = mean*(1-mean) if family == 'bernoulli' else mean
        return -(x.T @ (weight[:,None]*x))
    result = optimize.root(score, np.zeros(x.shape[1]), jac=jacobian, method='hybr', options={'xtol':1e-10})
    return result.x, bool(result.success), str(result.message)


def select_cohort(report, binary_digest, revision=None):
    """Bind references to one measured binary and its complete declared cohort."""
    if revision is None:
        expected_digest = report.get('binary_sha256')
        planned, records = report['planned_runs'], report['runs']
    else:
        expected_digest = report.get('binaries', {}).get(revision)
        planned = [r for r in report['planned_runs'] if r.get('revision') == revision]
        records = [r for r in report['runs'] if r.get('revision') == revision]
    if expected_digest != binary_digest:
        raise ValueError('dataset exporter must be the measured executable')
    def key(row):
        return (row['family'], row['mode'], row['rows'], row.get('columns', 2), row['repeat'], row.get('revision'))
    expected, actual = [key(r) for r in planned], [key(r) for r in records]
    if not expected or len(set(expected)) != len(expected) or len(actual) != len(set(actual)) or set(expected) != set(actual):
        raise ValueError('reference cohort has missing, duplicate or unexpected jobs')
    return records, len(planned)


def main():
    import numpy as np
    import scipy
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('binary', type=Path)
    p.add_argument('report', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--revision', choices=['before', 'after', 'plain', 'counted', 'timed'])
    args = p.parse_args()
    report = json.loads(args.report.read_text())
    binary_digest = hashlib.sha256(args.binary.read_bytes()).hexdigest()
    records, planned = select_cohort(report, binary_digest, args.revision)
    results = []
    datasets = {}
    references = {}
    with tempfile.TemporaryDirectory() as temporary:
        for row in records:
            key = (row['rows'], row.get('columns', 2))
            if key not in datasets:
                path = Path(temporary)/f'{key[0]}-{key[1]}.json'
                subprocess.run([str(args.binary), 'normal', 'batch', str(key[0]), '1', 'plain', str(key[1]), str(path)], check=True, capture_output=True, timeout=300)
                datasets[key] = json.loads(path.read_text())
            data = datasets[key]
            item = {k: row[k] for k in ('family', 'mode', 'rows', 'repeat')}
            item['columns'] = key[1]
            if 'error' in row:
                results.append(dict(item, passed=False, error=row['error']))
                continue
            x = np.array(data['design']).reshape(key)
            y = np.array(data[row['family']], dtype=float)
            q = np.array(row['estimate_free'])
            initial = np.array([.1]+[-.2/j for j in range(1,key[1])]+([0.] if row['family']=='normal' else []))
            initial_value, initial_score = evaluate(row['family'], x, y, initial)
            value, score = evaluate(row['family'], x, y, q)
            reference_key = (row['family'], *key)
            if reference_key not in references:
                references[reference_key] = reference(row['family'], x, y)
            ref, success, message = references[reference_key]
            prediction_error = float(np.max(np.abs(x@(q[:key[1]]-ref[:key[1]]))))
            scale_error = abs(float(q[-1]-ref[-1])) if row['family']=='normal' else 0.
            score_error = max(float(np.max(np.abs(score-row['final_gradient']))), float(np.max(np.abs(initial_score-row['gradient']))))
            value_error = max(abs(value-row['final_value']), abs(initial_value-row['value']))
            passed = bool(success and prediction_error < 5e-6 and scale_error < 5e-6 and score_error <= 5e-11*key[0] and value_error <= 5e-10*key[0])
            results.append(dict(item, passed=passed, solver_success=success, solver_message=message, predictor_error=prediction_error, log_scale_error=scale_error, score_error=score_error, value_error=value_error, termination=row['termination']))
    manifest = dict(numpy=np.__version__, scipy=scipy.__version__, planned=planned, revision=args.revision, completed=len(results), binary_sha256=binary_digest, report_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest(), checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), results=results)
    with args.output.open('x') as output:
        json.dump(manifest, output, indent=2, allow_nan=False); output.write('\n')
    print(json.dumps(dict(planned=manifest['planned'], completed=len(results), passed=sum(r['passed'] for r in results))))
    if len(results) != manifest['planned'] or not all(r['passed'] for r in results):
        raise SystemExit('independent reference checks failed; all outcomes retained')


if __name__ == '__main__':
    main()
