"""Independent Decimal likelihood/Newton checks; retain every planned fit outcome.

No Alea implementation or derivative is used by the reference. Gaussian reference
normal equations are solved at 60 decimal digits, not used as a production solver.
Run --binary path/to/regression_fit --output NEW_DIRECTORY after a release build.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from decimal import Decimal as D, localcontext
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess


def groups(family, scenario, n):
    rows = defaultdict(lambda: [0, D(0), D(0), D(0)])
    factorials = {0: D(0)}
    for i in range(n):
        x = (i % 101) / 50. - 1. if scenario in ("tracing", "historical") else (i % 10) - 4.5 if scenario == "boundary" else float(i % 3) if scenario == "quasi" else float(i % 11) - 5.
        predictor = x * 1e4 if scenario == "scaled" else 1. + 1e-8 * x if scenario == "ill_conditioned" else 1. if scenario == "rank_deficient" else x
        offset = ((i % 3) - 1.) * .2 if scenario == "offset" else 0.
        if family == "gaussian":
            y = 1. + .5 * x if scenario == "perfect" else 1. + .5 * x + ((i * 17 % 19) - 9.) / 10.
        elif family == "logistic":
            y = i % 3 == 0 if scenario == "tracing" else (i * 19 % 31) / 31. < .5 + .2 * x if scenario == "historical" else x > 0 if scenario == "boundary" else x > 0 or i % 2 == 0 if scenario == "quasi" else (i * 19 % 31) / 31. < .5 + .08 * x
        else:
            y = 0 if scenario == "boundary" else i % 7 if scenario == "tracing" else i * 13 % 7
        y = D.from_float(y) if isinstance(y, float) else D(int(y))
        r = rows[D.from_float(predictor), D.from_float(offset)]
        r[0] += 1; r[1] += y; r[2] += y*y
        if family == "poisson":
            count = int(y)
            if count not in factorials:
                factorials[count] = sum((D(k).ln() for k in range(1, count+1)), D(0))
            r[3] += factorials[count]
    return [(x, off, D(count), sy, sy2, fact) for (x, off), (count, sy, sy2, fact) in sorted(rows.items())]


def evaluate(family, rows, q):
    value = D(0); gradient = [D(0)] * len(q)
    for x, off, count, sy, sy2, fact in rows:
        eta = q[0] + q[1]*x + off
        if family == "gaussian":
            invvar = (-2*q[2]).exp()
            residual = sy - count*eta
            rss = sy2 - 2*eta*sy + count*eta*eta
            # Independently rounded 60-digit constant log(2*pi).
            log2pi = D('1.837877066409345483560659472811235279722794947275566825634303')
            value += -count*(q[2] + log2pi/2) - rss*invvar/2
            score = residual*invvar
            gradient[2] += -count + rss*invvar
        elif family == "logistic":
            e = eta.exp(); p = e/(1+e)
            value += sy*eta - count*(1+e).ln(); score = sy-count*p
        else:
            e = eta.exp(); value += sy*eta-count*e-fact; score = sy-count*e
        gradient[0] += score; gradient[1] += score*x
    return value, gradient


def reference(family, rows):
    n = sum(r[2] for r in rows)
    center = (rows[0][0] + rows[-1][0])/2
    scale = max(abs(r[0]-center) for r in rows) or D(1)
    varying = any(r[0] != center for r in rows)
    if family == "gaussian":
        sx = sum(r[2]*r[0] for r in rows); sxx = sum(r[2]*r[0]*r[0] for r in rows)
        sy = sum(r[3]-r[2]*r[1] for r in rows)
        sxy = sum(r[0]*(r[3]-r[2]*r[1]) for r in rows)
        slope = (n*sxy-sx*sy)/(n*sxx-sx*sx) if varying else D(0)
        intercept = (sy-slope*sx)/n
        rss = sum(r[4]-2*(intercept+slope*r[0]+r[1])*r[3]+r[2]*(intercept+slope*r[0]+r[1])**2 for r in rows)
        return [intercept, slope, (rss/n).ln()/2]
    # Newton in a centered/scaled reference basis, with independent analytic Hessian.
    a = b = D(0)
    for _ in range(100):
        g0 = g1 = h00 = h01 = h11 = D(0)
        for x, off, count, sy, _, _ in rows:
            z = (x-center)/scale; e = (a+b*z+off).exp()
            p = e/(1+e) if family == "logistic" else e
            weight = count*p*(1-p) if family == "logistic" else count*p
            score = sy-count*p
            g0 += score; g1 += score*z; h00 += weight; h01 += weight*z; h11 += weight*z*z
        if max(abs(g0), abs(g1)) < D('1e-24'):
            return [a-b*center/scale, b/scale]
        if varying:
            det = h00*h11-h01*h01
            da, db = (g0*h11-g1*h01)/det, (g1*h00-g0*h01)/det
        else:
            da, db = g0/h00, D(0)
        current = evaluate(family, rows, [a-b*center/scale, b/scale])[0]
        step = D(1)
        for _ in range(100):
            aa, bb = a+step*da, b+step*db
            if evaluate(family, rows, [aa-bb*center/scale, bb/scale])[0] >= current:
                a, b = aa, bb
                break
            step /= 2
        else:
            raise ArithmeticError("reference Newton search did not improve")
    raise ArithmeticError("reference Newton did not converge")


def assess(record):
    family, scenario, n = record['family'], record['scenario'], record['rows']
    with localcontext() as ctx:
        ctx.prec = 60
        rows = groups(family, scenario, n)
        q = [D.from_float(x) for x in record['position']]
        value, score = evaluate(family, rows, q)
        scales = [D(1), D('1e-4')] + ([D(1)] if family == 'gaussian' else []) if scenario == 'scaled' else [D(1)]*len(q)
        scaled_score = max(abs(D.from_float(x)*s) for x,s in zip(record['gradient'],scales))
        criterion = scaled_score <= D.from_float(record['score_tolerance'])
        assert criterion == (record['termination'] == 'ScoreTolerance'), record
        result = {'numerical_criterion_met':criterion,
                  'score_evaluation_error':float(max(abs(a-D.from_float(b)) for a,b in zip(score,record['gradient']))),
                  'reference_value_at_candidate':str(value), 'reference_score_at_candidate':[str(x) for x in score]}
        if scenario in ('boundary','quasi','perfect'):
            result['reference_status'] = 'no_finite_joint_maximum'
            return result
        qstar = reference(family, rows); best, _ = evaluate(family, rows, qstar)
        predictor_error = max(abs(q[0]+q[1]*r[0]-qstar[0]-qstar[1]*r[0]) for r in rows)
        scale_error = abs(q[2]-qstar[2]) if family == 'gaussian' else D(0)
        result.update(reference_status='computed', reference_position=[str(x) for x in qstar],
                      predictor_error=float(predictor_error), log_scale_error=float(scale_error),
                      objective_gap=str(best-value), objective_ulp=math.ulp(record['value']),
                      gap_below_float_ulp=abs(best-value)<D.from_float(math.ulp(record['value'])),
                      candidate_accuracy_passed=predictor_error<D('5e-6') and scale_error<D('5e-6'))
        # Agreement tolerances are reference-comparison checks, not optimizer stopping rules.
        result['score_evaluation_passed'] = result['score_evaluation_error'] <= 5e-11*n
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=3)
    args=parser.parse_args()
    if args.rounds<1: parser.error('rounds must be positive')
    args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[3]
    source_names = ['crates/alea-core/src/model/assembly.rs','crates/alea-distributions/src/modeling/regression/diagnostics/assessment.rs','crates/alea-runtime/src/optimize.rs','crates/alea-math/src/quasi_newton.rs','crates/alea-distributions/src/modeling/regression.rs','tools/downstream-conformance/examples/regression_fit.rs',str(Path(__file__).relative_to(root))]
    manifest={'schema_version':1,'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
              'binary_sha256':hashlib.sha256(args.binary.read_bytes()).hexdigest(), 'host':platform.platform(),
              'source_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in source_names},
              'reference':'Python Decimal 60 digits; independent grouped likelihoods, scores and Newton equations',
              'acceptance':'predictor/log-scale error <5e-6; score evaluation error <=5e-11*n; optimizer criterion unchanged',
              'timing':'Rust Instant construction and fitting separately; peak process RSS; no reference-computation time included'}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.output/'source.patch').write_bytes(subprocess.check_output(['git','diff','HEAD','--binary'],cwd=root))
    cases=[(f,s,330,m) for f in ('gaussian','logistic','poisson') for s in ('ordinary','scaled','ill_conditioned','rank_deficient','offset') for m in ('batch','scalar')]
    cases += [(f,s,330,'batch') for f,s in [('gaussian','perfect'),('logistic','boundary'),('logistic','quasi'),('poisson','boundary')]]
    cases += [(f,'tracing',100000,'batch') for f in ('logistic','poisson')]
    jobs=[(r,*c) for r in range(args.rounds) for c in cases]; random.Random(20261029).shuffle(jobs)
    records=[]
    with (args.output/'runs.jsonl').open('x') as out:
        for round_index,f,s,n,m in jobs:
            p=subprocess.run([str(args.binary.resolve()),f,s,str(n),m],capture_output=True,text=True)
            item={'round':round_index,'family':f,'scenario':s,'rows':n,'mode':m,'exit_status':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
            if p.returncode==0:
                try:
                    item['fit']=json.loads(p.stdout); item['assessment']=assess(item['fit'])
                except (ValueError, ArithmeticError, AssertionError) as e:
                    item['reference_failure']=str(e)
            records.append(item);out.write(json.dumps(item)+'\n');out.flush()
    summary={'planned':len(jobs),'completed':sum(r['exit_status']==0 for r in records),
             'reference_failures':sum('reference_failure' in r for r in records),
             'required_accuracy_failures':sum(r['scenario'] not in ('boundary','quasi','perfect','ill_conditioned') and not r.get('assessment',{}).get('candidate_accuracy_passed',False) for r in records),
             'score_evaluation_failures':sum(r.get('assessment',{}).get('score_evaluation_passed') is False for r in records),
             'score_tolerance_met':sum(r.get('assessment',{}).get('numerical_criterion_met',False) for r in records),
             'accurate_candidates':sum(r.get('assessment',{}).get('candidate_accuracy_passed',False) for r in records),
             'all_outcomes_retained':True}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
    if summary['completed']!=summary['planned'] or summary['reference_failures'] or summary['required_accuracy_failures'] or summary['score_evaluation_failures']:
        raise SystemExit('incomplete/reference-failed cohort; see retained records')

if __name__=='__main__': main()
