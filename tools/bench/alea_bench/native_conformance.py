"""Pre-measurement cross-backend checks at shared, perturbed and tail points."""
import json
import math
import random
import subprocess
from decimal import Decimal, localcontext
from pathlib import Path

from .manifest import ROOT, write_new
from .pdb_external import Bridge
from .runner import environment


def points(initials):
    """Deterministic points chosen without consulting timing or sampler outcomes."""
    base = initials[0]
    rng = random.Random(482913)
    return [list(q) for q in initials] + [
        [x + scale*rng.uniform(-1., 1.) for x in base]
        for scale in (.001, .03, .3, 1., 3.)
    ]


def relative(a, b):
    if len(a) != len(b) or any(not math.isfinite(x) for x in [*a, *b]):
        return math.inf
    return max((abs(x-y)/(1.+abs(y)) for x, y in zip(a, b, strict=True)), default=0.)


def reference_location(bridge, case, metadata):
    from .pdb_report import reference_name
    path=ROOT/case['directory']/'reference-summary.json'
    if not path.exists():return None
    summaries=json.loads(path.read_text())
    names=[reference_name(n) for n in metadata['initial_parameter_names'].split(',')]
    if not all(n in summaries for n in names):return None
    values=[summaries[n]['mean'] for n in names]
    q=bridge.unconstrain(values)
    return q if all(math.isfinite(x) for x in q) else None


def reference(bridge, case, data, q):
    """Use an independent high-precision oracle for dogs' rounded probabilities.

    Its Stan program computes inv_logit then bernoulli; density cancellation
    reaches 2e-6 at one shared start although scores agree to 2e-15. Preserve
    the observed Stan value, and check the mathematical density at 80 digits.
    """
    if case != 'dogs-dogs_log':
        value, gradient = bridge.gradient(q)
        return value, gradient, None
    try:
        stan, stan_gradient = bridge.gradient(q)
    except ValueError:
        stan, stan_gradient = None, None
    if not (-100. < q[0] < 0. and 0. < q[1] < 100.):
        return -math.inf, [math.nan]*2, dict(stan_logp=stan)
    with localcontext() as context:
        context.prec = 80
        beta = list(map(Decimal.from_float, q)); total = Decimal(0); gradient = [Decimal(0), Decimal(0)]
        for dog in data['y']:
            avoid = shock = 0
            for y in dog:
                z = beta[0]*avoid+beta[1]*shock
                exponential = z.exp(); total += y*z-(1+exponential).ln()
                residual = y-exponential/(1+exponential)
                gradient[0] += avoid*residual; gradient[1] += shock*residual
                avoid += 1-y; shock += y
        score = list(map(float, gradient)); value = float(total)
    return value, score, dict(stan_logp=stan if stan is not None and math.isfinite(stan) else None,
        stan_score_relative_error=relative(stan_gradient,score) if stan_gradient is not None else None,
        reference='independent Decimal80 Bernoulli-logit, original uniform support; Stan value retained')


def validate_case(case, model, worker, python, output, runtime_environment=None):
    """Stan is the independent density/score oracle; check potential HVP signs too."""
    output.mkdir(parents=True, exist_ok=False)
    env = environment(ROOT)
    env.update(runtime_environment or {})
    data = ROOT/case['directory']/'data.json'
    values = json.loads(data.read_text())
    bridge = Bridge(model['builds']['bridge']['binary'], data)
    locations = points(model['metadata']['initials'])
    typical=reference_location(bridge,case,model['metadata'])
    if typical is not None:
        locations=[[x+scale*(1+abs(x))*(1 if i%2==0 else -1) for i,x in enumerate(typical)] for scale in (0.,-.01,.01)]+locations
    task = dict(library=model['builds']['bridge']['binary'], data=str(data),
                case=case['id'], points=locations, target_backend='enzyme')
    write_new(output/'probe.json', task)
    try:
        with (output/'probe.log').open('x') as log:
            subprocess.run([str(worker), 'native-probe', str(output/'probe.json'), str(output/'enzyme.json')],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
        rows = json.loads((output/'enzyme.json').read_text())['rows']
        if len(rows) != len(locations): raise ValueError('missing conformance points')
        checks = []; offset = None; admitted = 0
        for q, row in zip(locations, rows, strict=True):
            try:
                value, gradient, supplement = reference(bridge, case['id'], values, q)
                valid = math.isfinite(value) and all(math.isfinite(g) for g in gradient)
            except ValueError:
                valid = False; value = None; gradient = None
            if row['admitted'] != valid: raise ValueError('native/Stan support mismatch')
            if not valid:
                checks.append(dict(q=q, admitted=False)); continue
            admitted += 1
            delta = row['logp']-value
            if offset is None: offset = delta
            # Density differences use an absolute plus floating-point scale
            # allowance, not a tolerance proportional to a huge log-density.
            tolerance = 2e-7+2e-12*max(abs(value), abs(row['logp']))
            if abs(delta-offset) > tolerance: raise ValueError('native/Stan density difference mismatch')
            gradient_error = relative(row['gradient'], gradient)
            if gradient_error > 2e-6: raise ValueError(f'native/Stan score mismatch {gradient_error}')
            direction = [1. if i % 2 == 0 else -1. for i in range(bridge.d)]
            h = 2e-5
            _, plus, _ = reference(bridge,case['id'],values,[x+h*v for x, v in zip(q, direction, strict=True)])
            _, minus, _ = reference(bridge,case['id'],values,[x-h*v for x, v in zip(q, direction, strict=True)])
            numerical = [-(a-b)/(2*h) for a, b in zip(plus, minus, strict=True)]
            hvp_error = relative(row['hvp'], numerical)
            if row['hvp_error'] or hvp_error > 2e-4: raise ValueError(f'native potential HVP mismatch {hvp_error}')
            if bridge.d <= 10 and row['third_error']:raise ValueError('native dense derivative preflight: '+row['third_error'])
            if row.get('third_relative_error') is not None and row['third_relative_error'] > 2e-3:
                raise ValueError(f'native third derivative consistency mismatch {row["third_relative_error"]}')
            checks.append(dict(q=q, admitted=True, logp=value, gradient=gradient,
                                  gradient_error=gradient_error, hvp_error=hvp_error, supplement=supplement))
        if admitted < len(model['metadata']['initials']): raise ValueError('insufficient finite conformance points')
        write_new(output/'stan.json', dict(rows=checks, density_constant_offset=offset))
        # Run JAX in its pinned environment. Its translation and score are
        # checked independently against the same Stan oracle at all points.
        command = [str(python), '-m', 'alea_bench.native_conformance', str(output/'probe.json'), str(output/'jax.json')]
        with (output/'jax.log').open('x') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           env=env, check=True, timeout=300)
        return dict(status='validated', points=len(locations), admitted=admitted,
                    gradient_max_relative_error=max(r.get('gradient_error', 0.) for r in checks),
                    hvp_max_relative_error=max(r.get('hvp_error', 0.) for r in checks),
                    density_constant_offset=offset,third_relative_error=rows[0].get('third_relative_error'),
                    reference_mean_coordinate_checks=3 if typical is not None else 0)
    finally:
        bridge.close()


def validate_jax(task, output):
    import jax
    import jax.numpy as j
    import numpy as np
    from .native_jax import density
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_enable_compilation_cache', False)
    if jax.default_backend() != 'cpu': raise ValueError('conformance requires CPU')
    target = jax.jit(jax.value_and_grad(density(task['case'], json.loads(Path(task['data']).read_text()))))
    bridge = Bridge(task['library'], task['data']); rows = []; offset = None
    values = json.loads(Path(task['data']).read_text())
    try:
        for q in task['points']:
            value, gradient = target(j.asarray(q)); value = float(value); gradient = np.asarray(gradient).tolist()
            valid = math.isfinite(value) and all(math.isfinite(x) for x in gradient)
            try:
                expected, score, supplement = reference(bridge,task['case'],values,q)
                expected_valid = math.isfinite(expected) and all(math.isfinite(x) for x in score)
            except ValueError:
                expected_valid = False
            if valid != expected_valid: raise ValueError('JAX/Stan support mismatch')
            if not valid:
                rows.append(dict(admitted=False)); continue
            difference = value-expected
            if offset is None: offset = difference
            if abs(difference-offset) > 2e-7+2e-12*max(abs(value), abs(expected)): raise ValueError('JAX/Stan density difference mismatch')
            error = relative(gradient, score)
            if error > 2e-6: raise ValueError(f'JAX/Stan score mismatch {error}')
            rows.append(dict(admitted=True, logp=value, gradient=gradient, gradient_error=error, supplement=supplement))
        write_new(output, dict(status='validated', rows=rows, density_constant_offset=offset))
    finally:
        bridge.close()


if __name__ == '__main__':
    import sys
    validate_jax(json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]))
