"""Executed BlackJAX 1.5 M4 references: schedules, update traces and sampling.

Run with the Python closure from default.nix. No Rust output is read.
"""
import argparse
import csv
import io
import sys
from importlib.metadata import version
from pathlib import Path
from generate import VERSIONS
import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from blackjax.adaptation.window_adaptation import base, build_schedule


def csv_output(header):
    output = io.StringIO(newline="")
    output.write("# BlackJAX 1.5; JAX/jaxlib 0.9.2; CPU float64; independent executed reference\n")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(header.split(","))
    return output, writer


def schedules():
    output, writer = csv_output("length,index,slow,end")
    for length in [0, 1, 19, 20, 21, 149, 150, 151, 200, 1000]:
        for i, (slow, end) in enumerate(np.asarray(build_schedule(length))):
            writer.writerow([length, i, int(slow), int(end)])
    return output.getvalue()


def traces():
    output, writer = csv_output("kind,index,x,y,acceptance,step,c00,c01,c11")
    for diagonal in [True, False]:
        init, update, _ = base(diagonal)
        update = jax.jit(update)
        state = init(jnp.zeros(2), 0.3)
        for i, stage in enumerate(build_schedule(200)):
            # Deterministic, bounded input independent of any sampler's RNG.
            x, y = float(i % 13 - 6), float((i * 7) % 17 - 8)
            acceptance = [0.1, 0.5, 0.8, 0.0, 1.0, 0.95, 0.75][i % 7]
            state = update(state, stage, jnp.array([x, y]), acceptance)
            matrix = np.asarray(state.inverse_mass_matrix)
            values = [matrix[0], 0.0, matrix[1]] if diagonal else [matrix[0,0], matrix[0,1], matrix[1,1]]
            writer.writerow(["diagonal" if diagonal else "dense", i, x, y, acceptance, float(state.step_size), *values])
    return output.getvalue()


def density(q):
    # Rotated Gaussian with covariance eigenvalues 0.01 and 1 (condition 100).
    x, y = (0.8*q[0]+0.6*q[1])/0.1, -0.6*q[0]+0.8*q[1]
    return -0.5*(x*x+y*y)


def sampling():
    output, writer = csv_output("kind,chains,warmup,draws,steps,observable,mean,mcse,ess_per_gradient")
    chains, warmup, draws, steps = 4, 1000, 8192, 5
    for diagonal in [True, False]:
        adapt = blackjax.window_adaptation(blackjax.hmc, density,
            is_mass_matrix_diagonal=diagonal, initial_step_size=0.1,
            num_integration_steps=steps, progress_bar=False)

        @jax.jit
        def run(seed):
            key1, key2 = jax.random.split(jax.random.key(seed))
            (state, params), _ = adapt.run(key1, jnp.zeros(2), warmup)
            sampler = blackjax.hmc(density, **params)
            def step(state, key):
                state, info = sampler.step(key, state)
                q = state.position
                x, y = (0.8*q[0]+0.6*q[1])/0.1, -0.6*q[0]+0.8*q[1]
                return state, (jnp.array([x,y,x*x,y*y,x*y]), info.is_divergent)
            return jax.lax.scan(step, state, jax.random.split(key2, draws))[1]

        results = [run(seed) for seed in range(601, 601+chains)]
        samples = np.stack([np.asarray(x) for x, _ in results])
        assert not any(np.asarray(divergent).any() for _, divergent in results)
        errors = []
        for batch in [128, 256]:
            means = samples.reshape(chains*draws//batch,batch,5).mean(axis=1)
            errors.append(means.std(axis=0, ddof=1)/np.sqrt(len(means)))
        mcse = np.maximum(*errors)
        variance = samples.reshape(-1,5).var(axis=0, ddof=1)
        efficiency = variance/(mcse*mcse)/(chains*draws*steps)
        for i in range(5):
            writer.writerow(["diagonal" if diagonal else "dense", chains,warmup,draws,steps,i,samples.mean(axis=(0,1))[i],mcse[i],efficiency[i]])
        print(f"generated adapted {diagonal=}",file=sys.stderr)
    return output.getvalue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for package, expected in VERSIONS.items():
        assert version(package) == expected
    assert jax.config.x64_enabled and jax.default_backend() == "cpu"
    for name, generate in [("schedule",schedules), ("adaptation",traces), ("warmup-sampling",sampling)]:
        path = args.directory / f"blackjax-{name}.csv"
        text = generate()
        if args.check:
            assert path.read_text() == text, path
        else:
            path.write_text(text)
