"""Executed independent HMC sampling references with batch-means uncertainty.

No Rust output is read. JAX float64/CPU and all packages are pinned by default.nix.
Four independent chains, rejected states included, no adaptation or thinning.
"""
import argparse
import csv
import io
import sys
from importlib.metadata import version
from pathlib import Path
from generate import VERSIONS, correlated, banana, logistic, funnel, rotated
import blackjax
import jax
import jax.numpy as jnp
import numpy as np

CHAINS, WARMUP, DRAWS = 4, 1024, 8192
HEADER = "target,mass,l00,l10,l11,step_size,steps,chains,warmup,draws,observable,mean,mcse,divergences"


def observables(name, q):
    x, y = q[0], q[1]
    if name == "correlated":
        y = (q[1] - 0.6*q[0])/0.8
    elif name == "banana":
        y = q[1] - 0.4*(q[0]**2 - 1.0)
    elif name == "funnel":
        y = q[1]*jnp.exp(-0.5*q[0])
    elif name == "rotated":
        x, y = (0.8*q[0]+0.6*q[1])/0.01, -0.6*q[0]+0.8*q[1]
    return jnp.array([x, y, x*x, y*y, x*y])


def generate():
    for package, expected in VERSIONS.items():
        assert version(package) == expected
    assert jax.config.x64_enabled and jax.default_backend() == "cpu"
    output = io.StringIO(newline="")
    output.write("# BlackJAX 1.5; JAX/jaxlib 0.9.2; CPU float64 JIT; seeds 301..304\n")
    output.write("# MCSE=max(nonoverlapping batch-means SE with batches 256,512); independent chains\n")
    output.write(HEADER + "\n")
    writer = csv.writer(output, lineterminator="\n")
    for name, density in [("correlated", correlated), ("banana", banana), ("logistic", logistic), ("funnel", funnel), ("rotated", rotated)]:
        cases = [("identity", [1.0, 0.0, 1.0]), ("diagonal", [2.0, 0.0, 0.5]), ("dense", [1.5, 0.4, 0.8])]
        if name == "rotated":
            a = np.sqrt(6400.36)
            cases = [("matched", [a, 4799.52/a, 100.0/a])]
        for mass, (a, b, c) in cases:
            factor = jnp.array([[a, 0.0], [b, c]])
            inverse = jnp.linalg.inv(factor @ factor.T)
            eps, steps = (0.06, 18) if name == "funnel" else (0.15, 9)
            sampler = blackjax.hmc(density, eps, inverse, steps)

            def run(seed):
                def step(state, key):
                    state, info = sampler.step(key, state)
                    values = jnp.concatenate([observables(name, state.position), jnp.array([info.is_accepted], dtype=jnp.float64)])
                    return state, (values, info.is_divergent)
                state = sampler.init(jnp.zeros(2))
                keys = jax.random.split(jax.random.key(seed), WARMUP+DRAWS)
                return jax.lax.scan(step, state, keys)[1]

            run = jax.jit(run)
            results = [run(seed) for seed in range(301, 301+CHAINS)]
            samples = np.stack([np.asarray(values)[WARMUP:] for values, _ in results])
            divergences = sum(int(np.asarray(flags).sum()) for _, flags in results)
            assert divergences == 0, (name, mass, divergences)
            assert np.isfinite(samples).all()
            mean = samples.mean(axis=(0, 1))
            errors = []
            for batch in [256, 512]:
                batches = samples.reshape(CHAINS * DRAWS // batch, batch, 6).mean(axis=1)
                errors.append(batches.std(axis=0, ddof=1) / np.sqrt(len(batches)))
            mcse = np.maximum(*errors)
            assert np.all(mcse > 0) and np.all(mcse < 0.2), (name, mass, mcse)
            for i in range(6):
                writer.writerow([name, mass, a, b, c, eps, steps, CHAINS, WARMUP, DRAWS, i, mean[i], mcse[i], divergences])
            print(f"generated {name}/{mass}", file=sys.stderr)
    return output.getvalue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    text = generate()
    if args.check:
        if args.check.read_text() != text:
            sys.exit("sampling fixture differs; review environment/numerical changes")
        print("78 sampling summary rows reproduce byte-for-byte")
    else:
        sys.stdout.write(text)
