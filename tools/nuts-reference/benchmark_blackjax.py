"""Synchronized CPU BlackJAX/JAX timing, separate from compilation and CSV I/O."""
import csv
import json
import os
import sys
import time
from pathlib import Path

from blackjax_run import blackjax, jax, jnp, np, correlated, banana, logistic, funnel, VERSIONS, version


def main():
    output = Path(sys.argv[1])
    timing = Path(os.environ['ALEA_BENCH_TIMINGS'])
    for package, expected in VERSIONS.items():
        assert version(package) == expected
    assert jax.config.x64_enabled and jax.default_backend() == 'cpu'
    jax.config.update('jax_enable_compilation_cache', False)
    compilation = []
    with output.open('x') as out, timing.open('x') as times:
        out.write('target,seed,draw,q0,q1,divergent,steps,acceptance\n')
        writer = csv.writer(times)
        writer.writerow(('target','seed','warmup_seconds','sampling_seconds','warmup_steps'))
        for t, density in enumerate((correlated, banana, logistic, funnel)):
            kernel = blackjax.nuts(density, .06 if t == 3 else .15, jnp.ones(2), max_num_doublings=8)

            def run(state, keys):
                def step(state, key):
                    state, info = kernel.step(key, state)
                    return state, jnp.concatenate((state.position, jnp.array([
                        info.is_divergent, info.num_integration_steps, info.acceptance_rate])))
                return jax.lax.scan(step, state, keys)

            state = kernel.init(jnp.array([1., -1.]))
            keys = jax.random.split(jax.random.key(501), 9216)
            jax.block_until_ready((state, keys))
            start = time.perf_counter()
            warm = jax.jit(run).lower(state, keys[:1024]).compile()
            sample = jax.jit(run).lower(state, keys[1024:]).compile()
            compilation.append(dict(target=t, seconds=time.perf_counter()-start))
            for seed in range(501, 505):
                sign = -1. if seed % 2 == 0 else 1.
                state = kernel.init(jnp.array([sign, -sign]))
                keys = jax.random.split(jax.random.key(seed), 9216)
                warm_keys, sample_keys = keys[:1024], keys[1024:]
                jax.block_until_ready((state, warm_keys, sample_keys))
                start = time.perf_counter()
                state, warm_values = jax.block_until_ready(warm(state, warm_keys))
                warm_seconds = time.perf_counter()-start
                start = time.perf_counter()
                _, values = jax.block_until_ready(sample(state, sample_keys))
                sample_seconds = time.perf_counter()-start
                writer.writerow((t, seed, warm_seconds, sample_seconds, int(np.asarray(warm_values)[:,3].sum())))
                rows = np.column_stack((np.full(8192,t),np.full(8192,seed),np.arange(8192),np.asarray(values)))
                np.savetxt(out, rows, delimiter=',', fmt=['%d','%d','%d','%.17e','%.17e','%d','%d','%.17e'])
            print(f'BlackJAX/JAX benchmark target {t}', flush=True)
    with timing.with_suffix('.compile.json').open('x') as out:
        json.dump(compilation, out, indent=2)


if __name__ == '__main__':
    main()
