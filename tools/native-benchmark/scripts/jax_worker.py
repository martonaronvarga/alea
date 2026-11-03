#!/usr/bin/env python3
"""Pinned CPU/f64 JAX reference; AD and NUTS are provided by JAX/BlackJAX.

Data are runtime arguments, not baked into an XLA executable. Every timed loop
consumes both values and ALL scores. Synchronization and compilation are explicit.
"""
import argparse
import csv
from importlib.metadata import version
import json
import os
from pathlib import Path
import time

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'true'


def density_for(family):
    import jax
    import jax.numpy as jnp
    from jax.scipy.stats import norm
    def density(q, x, y):
        if family == 'funnel':
            return norm.logpdf(q[0], 0., 3.) + norm.logpdf(q[1:], 0., jnp.exp(q[0] / 2.)).sum()
        if family == 'logistic':
            eta = x @ q
            return norm.logpdf(q, 0., 2.).sum() + jnp.where(y == 1, jax.nn.log_sigmoid(eta), jax.nn.log_sigmoid(-eta)).sum()
        beta, log_sigma = q[:-1], q[-1]
        sigma = jnp.exp(log_sigma)
        mean = beta[0] if family == 'mean' else x @ beta
        # Same restricted Normal prior as the native/CmdStan fixtures; no ln(2)
        # constant. Exactly one positive-chart Jacobian, no handwritten scores.
        return norm.logpdf(beta, 0., 2.).sum() + norm.logpdf(sigma, 0., 1.) + log_sigma + norm.logpdf(y, mean, sigma).sum()
    return density


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('family', choices=['mean', 'linear', 'logistic', 'funnel'])
    parser.add_argument('data', type=Path)
    parser.add_argument('mode', choices=['probes', 'evaluate', 'sampling'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--points', type=Path)
    parser.add_argument('--evaluations', type=int, default=10000)
    parser.add_argument('--chains', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    import jax
    import jax.numpy as jnp
    import numpy as np
    import blackjax
    versions = {name: version(name) for name in ('jax', 'jaxlib', 'blackjax', 'numpy', 'scipy')}
    if {k: versions[k] for k in ('jax', 'jaxlib', 'blackjax')} != dict(jax='0.9.2', jaxlib='0.9.2', blackjax='1.5'):
        raise ValueError(f'changed reference environment: {versions}')
    jax.config.update('jax_enable_compilation_cache', False)
    if not jax.config.x64_enabled or jax.default_backend() != 'cpu':
        raise ValueError('CPU float64 required')
    data = json.loads(args.data.read_text())
    dimension = 2 if args.family == 'mean' else data['K'] + int(args.family != 'logistic')
    x = jnp.asarray(data.get('X', []), dtype=jnp.float64)
    y = jnp.asarray(data.get('y', []), dtype=jnp.float64)
    q = jnp.zeros(dimension, dtype=jnp.float64)
    jax.block_until_ready((x, y, q))
    density = density_for(args.family)
    vg = jax.value_and_grad(density)
    setup_seconds = time.perf_counter() - started
    c = time.perf_counter()
    evaluate = jax.jit(vg).lower(q, x, y).compile()
    compile_seconds = time.perf_counter() - c
    result = dict(versions=versions, backend='cpu', dtype='float64', setup_seconds=setup_seconds)
    if args.mode == 'probes':
        points = json.loads(args.points.read_text())
        result['probes'] = []
        for point in points:
            value, gradient = jax.block_until_ready(evaluate(jnp.asarray(point, dtype=jnp.float64), x, y))
            result['probes'].append(dict(q=point, logp=float(value), gradient=np.asarray(gradient).tolist()))
    elif args.mode == 'evaluate':
        if args.evaluations < 10 or args.evaluations % 10:
            raise ValueError('fixed-work count must be positive multiple of ten')
        def loop(count, x, y, positions):
            def step(i, carry):
                # Every COMPLETE position is a runtime array input. Even an
                # external base position with an internal q0 update would let
                # XLA hoist unchanged coefficients/scale work across calls.
                point = positions[i]
                value, gradient = vg(point, x, y)
                # Sequential coordinate accumulation matches the other workers'
                # receipt. Retaining only the final gradient would let XLA remove
                # earlier reverse work and would not measure N gradient calls.
                total_gradient = carry[1]
                for j in range(dimension):
                    total_gradient = total_gradient + (j + 1) * gradient[j]
                return carry[0] + value, total_gradient
            return jax.lax.fori_loop(0, count, step, (jnp.float64(0.), jnp.float64(0.)))
        count = jnp.asarray(args.evaluations, dtype=jnp.int32)
        points = np.zeros((args.evaluations,dimension),dtype=np.float64)
        points[:,0]=(np.arange(args.evaluations)%10)*.01
        positions=jnp.asarray(points)
        jax.block_until_ready(positions)
        c = time.perf_counter()
        compiled = jax.jit(loop).lower(count, x, y, positions).compile()
        compile_seconds += time.perf_counter() - c
        (args.output / 'loop.hlo.txt').write_text(compiled.as_text())
        alternate = positions.at[:10].add(jnp.arange(dimension,dtype=jnp.float64)*.03+.07)
        actual = tuple(map(float,jax.block_until_ready(compiled(jnp.int32(10),x,y,alternate))))
        expected = [0.,0.]
        for i in range(10):
            v,g = jax.block_until_ready(evaluate(alternate[i],x,y))
            expected[0] += float(v)
            for j,score in enumerate(np.asarray(g)): expected[1] += (j+1)*float(score)
        if not np.allclose(actual,expected,rtol=3e-9,atol=3e-9):
            raise ValueError('compiled loop ignored or specialized the runtime coefficient state')
        result['runtime_position_check'] = True
        jax.block_until_ready(compiled(jnp.int32(10), x, y, positions))
        c = time.perf_counter()
        checksum, gradient_checksum = jax.block_until_ready(compiled(count, x, y, positions))
        seconds = time.perf_counter() - c
        last = q.at[0].set(.09)
        value, gradient = jax.block_until_ready(evaluate(last, x, y))
        # Separately expose Python dispatch + completion latency, not mixed into
        # the backend-resident loop or advertised as the same measurement scope.
        host_count = 100
        host_points=[q.at[0].set(i*.01) for i in range(10)]
        jax.block_until_ready(host_points)
        c = time.perf_counter()
        for i in range(host_count):
            jax.block_until_ready(evaluate(host_points[i%10], x, y))
        host_seconds = time.perf_counter() - c
        result.update(evaluations=args.evaluations, evaluation_seconds=seconds,
                      checksum=float(checksum), gradient_checksum=float(gradient_checksum),
                      logp=float(value), gradient=np.asarray(gradient).tolist(),
                      host_calls=host_count, host_dispatch_seconds=host_seconds)
    else:
        def warm(key, q, x, y):
            adapter = blackjax.window_adaptation(blackjax.nuts, lambda q: density(q, x, y),
                is_mass_matrix_diagonal=True, initial_step_size=.2,
                target_acceptance_rate=.8, max_num_doublings=10,
                adaptation_info_fn=lambda state, info, adaptation: (
                    info.is_divergent, info.num_integration_steps, info.num_trajectory_expansions))
            return adapter.run(key, q, num_steps=1000)
        def sample(state, key, step_size, mass, x, y):
            kernel = blackjax.nuts.build_kernel()
            def step(state, key):
                state, info = kernel(key, state, lambda q: density(q, x, y), step_size, mass, max_num_doublings=10)
                return state, (state.position, info.is_divergent, info.num_integration_steps,
                               info.num_trajectory_expansions, info.acceptance_rate)
            return jax.lax.scan(step, state, jax.random.split(key, 1000))
        c = time.perf_counter()
        warm_compiled = jax.jit(warm).lower(jax.random.key(17), q, x, y).compile()
        compile_seconds += time.perf_counter() - c
        sample_compiled = None
        phases, divergences, steps, depths, paths = [], 0, 0, [], []
        serialization_seconds = 0.
        for chain in range(args.chains):
            key, retained = jax.random.split(jax.random.key(17 + chain))
            c = time.perf_counter()
            (state, parameters), warm_info = jax.block_until_ready(warm_compiled(key, q, x, y))
            warm_seconds = time.perf_counter() - c
            if sample_compiled is None:
                c = time.perf_counter()
                sample_compiled = jax.jit(sample).lower(state, retained, parameters['step_size'], parameters['inverse_mass_matrix'], x, y).compile()
                compile_seconds += time.perf_counter() - c
            c = time.perf_counter()
            _, draws = jax.block_until_ready(sample_compiled(state, retained, parameters['step_size'], parameters['inverse_mass_matrix'], x, y))
            sampling_seconds = time.perf_counter() - c
            positions, div, leapfrogs, depth, acceptance = map(np.asarray, draws)
            if not np.isfinite(positions).all() or not np.isfinite(acceptance).all():
                raise ValueError('nonfinite retained state/statistics')
            c = time.perf_counter()
            path = args.output / f'chain-{chain}.csv'
            with path.open('x') as stream:
                writer = csv.writer(stream, lineterminator='\n')
                # Free-coordinate names allow the existing same-engine Alea
                # diagnostic reader to consume these without another transform.
                writer.writerow([f'q{j}' for j in range(dimension)])
                writer.writerows(positions.tolist())
            serialization_seconds += time.perf_counter() - c
            paths.append(str(path.resolve()))
            divergences += int(div.sum()); steps += int(leapfrogs.sum()); depths.extend(depth.tolist())
            phases.append(dict(chain=chain, warmup_seconds=warm_seconds, sampling_seconds=sampling_seconds,
                               warmup_divergences=int(np.asarray(warm_info[0]).sum()),
                               warmup_leapfrog_steps=int(np.asarray(warm_info[1]).sum()),
                               step_size=float(parameters['step_size'])))
            (args.output / 'progress.json').write_text(json.dumps(phases, indent=2, allow_nan=False)+'\n')
        result.update(completed=args.chains * 1000, chains=args.chains, dimension=dimension,
                      failures=[], phases=phases, retained_divergences=divergences,
                      retained_leapfrog_steps=steps, retained_mean_depth=float(np.mean(depths)),
                      retained_max_depth=int(max(depths)), final_step_sizes=[p['step_size'] for p in phases],
                      evaluations=None, evaluation_count_reason='not instrumented; never inferred from leapfrog counts',
                      serialization_seconds=serialization_seconds, csv_paths=paths,
                      warmup_sampling_seconds=sum(p['warmup_seconds']+p['sampling_seconds'] for p in phases))
    result['compile_seconds'] = compile_seconds
    result['worker_seconds'] = time.perf_counter() - started
    (args.output / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
