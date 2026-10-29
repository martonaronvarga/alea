"""Run actual SciPy solvers against recorded ill-conditioned regression fits.

Use the optional analysis Nix shell. Outputs retain all solver statuses and check
predictions against the independent Decimal reference, never coefficient equality
alone. QR/SVD/coordinate recommendations were reviewed in R, statsmodels and Stan;
only the explicitly recorded SciPy solvers are executed by this tool.
"""
import argparse
from decimal import Decimal as D, localcontext
import hashlib
import json
from pathlib import Path
import platform
import check_regression_fits as decimal_reference


def solve(family, rows, coordinates):
    import numpy as np
    from scipy import linalg, optimize
    x = np.array([float(r[0]) for r in rows])
    count = np.array([float(r[2]) for r in rows])
    sy = np.array([float(r[3]) for r in rows])
    offset = np.array([float(r[1]) for r in rows])
    center = np.average(x, weights=count) if coordinates == 'standardized' else 0.
    scale = np.sqrt(np.average((x-center)**2, weights=count)) if coordinates == 'standardized' else 1.
    z = (x-center)/scale
    design = np.column_stack((np.ones_like(x),z))
    if family == 'gaussian':
        solution, _, rank, singular_values = linalg.lstsq(
            np.sqrt(count)[:,None]*design, (sy/count-offset)*np.sqrt(count),
            cond=1e-12, lapack_driver='gelsd')
        message = 'SciPy gelsd weighted least squares'
        solver_success = True
        singular_values = singular_values.tolist()
    else:
        def score(theta):
            eta = design @ theta + offset
            if family == 'logistic':
                from scipy.special import expit
                mean = expit(eta)
            else:
                mean = np.exp(eta)
            return design.T @ (sy-count*mean)
        def jacobian(theta):
            eta = design @ theta + offset
            if family == 'logistic':
                from scipy.special import expit
                mean = expit(eta); weight = count*mean*(1-mean)
            else:
                weight = count*np.exp(eta)
            return -(design.T @ (weight[:,None]*design))
        result = optimize.root(score, np.zeros(2), jac=jacobian, method='hybr', options={'xtol':1e-10})
        solution = result.x
        message, solver_success = result.message, bool(result.success)
        rank = int(np.linalg.matrix_rank(design)); singular_values = None
    # Map with Decimal before one final rounding, to separate mapping cancellation
    # from the solver's prediction accuracy.
    slope = D.from_float(float(solution[1]))/D.from_float(float(scale))
    intercept = D.from_float(float(solution[0]))-D.from_float(float(center))*slope
    q = [D.from_float(float(intercept)),D.from_float(float(slope))]
    if family == 'gaussian':
        n = sum(r[2] for r in rows)
        rss = sum(r[4]-2*(q[0]+q[1]*r[0]+r[1])*r[3]+r[2]*(q[0]+q[1]*r[0]+r[1])**2 for r in rows)
        q.append((rss/n).ln()/2)
    target = decimal_reference.reference(family,rows)
    error = max(abs(q[0]+q[1]*r[0]-target[0]-target[1]*r[0]) for r in rows)
    return {'solver_success':solver_success, 'solver_message':message,
            'rank':int(rank), 'singular_values':singular_values,
            'original_position':[str(v) for v in q], 'predictor_error':float(error),
            'prediction_check_passed':error<D('5e-6')}


def main():
    import numpy as np
    import scipy
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    records=[json.loads(line) for line in args.runs.read_text().splitlines()]
    representative={r['family']:r for r in records if r['scenario']=='ill_conditioned' and r['mode']=='batch' and r['round']==0 and r.get('coordinates','raw')=='raw'}
    output=[]
    with localcontext() as ctx:
        ctx.prec=60
        for family,record in sorted(representative.items()):
            rows=decimal_reference.groups(family,'ill_conditioned',record['rows'])
            for coordinates in ('raw','standardized'):
                item={'family':family,'coordinates':coordinates,**solve(family,rows,coordinates)}
                output.append(item)
    manifest={'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform(),
              'runs_sha256':hashlib.sha256(args.runs.read_bytes()).hexdigest(),
              'reference_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'decimal_script_sha256':hashlib.sha256(Path(decimal_reference.__file__).read_bytes()).hexdigest(),
              'planned':6, 'completed':len(output),
              'note':'Solver success, rank policy and prediction agreement are separate; raw controls are retained.'}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.output/'results.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))
    if len(output)!=6 or not all(r['prediction_check_passed'] for r in output if r['coordinates']=='standardized'):
        raise SystemExit('reference comparison incomplete or standardized prediction check failed')

if __name__=='__main__': main()
