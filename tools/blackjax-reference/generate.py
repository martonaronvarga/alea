"""Emit deterministic BlackJAX endpoint CSV to stdout, or check a saved fixture.

No Rust implementation or output is used to generate the reference values.
The optional Nix environment pins the complete Python dependency closure.
"""

import argparse
import csv
import io
import math
import os
import sys
from importlib.metadata import version
from pathlib import Path

# Set before importing JAX/BlackJAX, including on GPU-equipped hosts.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"

import jax
import jax.numpy as jnp
from blackjax.mcmc.integrators import new_integrator_state, velocity_verlet

VERSIONS = {"blackjax": "1.5", "jax": "0.9.2", "jaxlib": "0.9.2"}
HEADER = (
    "target,mass,step_size,steps,l00,l10,l11,inverse00,inverse01,inverse11,"
    "q0,q1,p0,p1,logp,g0,g1,energy,"
    "end_q0,end_q1,end_p0,end_p1,end_logp,end_g0,end_g1,end_energy"
)


def correlated(q):
    z = (q[1] - 0.6 * q[0]) / 0.8
    return -0.5 * (q[0] ** 2 + z**2)


def banana(q):
    z = q[1] - 0.4 * (q[0] ** 2 - 1.0)
    return -0.5 * (q[0] ** 2 + z**2)


def logistic(q):
    x = jnp.array([-1.5, -0.2, 0.7, 2.0])
    y = jnp.array([0.0, 1.0, 0.0, 1.0])
    eta = q[0] + q[1] * x
    return -0.5 * jnp.sum(q**2) + jnp.sum(y * eta - jax.nn.softplus(eta))


def funnel(q):
    # Centered funnel of width ONE; do not infer robustness for Neal's width 3.
    return -0.5 * q[0]**2 - 0.5 * q[0] - 0.5 * q[1]**2 * jnp.exp(-q[0])


def rotated(q):
    x = (0.8*q[0] + 0.6*q[1]) / 0.01
    y = -0.6*q[0] + 0.8*q[1]
    return -0.5 * (x*x + y*y)


def state_values(state, kinetic):
    values = [
        *state.position, *state.momentum, state.logdensity,
        *state.logdensity_grad, kinetic(state.momentum) - state.logdensity,
    ]
    assert all(value.dtype == jnp.float64 for value in values)
    assert all(math.isfinite(float(value)) for value in values)
    return [float(value) for value in values]


def generate(m3=False):
    for package, expected in VERSIONS.items():
        if version(package) != expected:
            raise RuntimeError(f"{package}: expected {expected}, got {version(package)}")
    assert jax.config.x64_enabled and jax.default_backend() == "cpu"
    output = io.StringIO(newline="")
    output.write("# BlackJAX 1.5; JAX/jaxlib 0.9.2; CPU; float64; no whole-step JIT\n")
    output.write("# Targets: logistic normal prior; funnel width=1; rotated condition=1e4\n" if m3 else "# Targets: correlated rho=0.6; banana bend=0.4; constants omitted\n")
    output.write(HEADER + "\n")
    writer = csv.writer(output, lineterminator="\n")
    targets = [("logistic", logistic), ("funnel", funnel), ("rotated", rotated)] if m3 else [("correlated", correlated), ("banana", banana)]
    for target_name, logp in targets:
        for mass_name, factor in [
            ("identity", [[1.0, 0.0], [0.0, 1.0]]),
            ("diagonal", [[2.0, 0.0], [0.0, 0.5]]),
            ("dense", [[1.5, 0.0], [0.4, 0.8]]),
        ]:
            lower = jnp.array(factor, dtype=jnp.float64)
            inverse = jnp.linalg.inv(lower @ lower.T)

            def kinetic(p):
                return 0.5 * (p @ inverse @ p)

            integrate = velocity_verlet(logp, kinetic)
            eps = 0.000625 if target_name == "rotated" else 0.125
            for step_size in [eps, -eps]:
                for count in [1, 5, 11]:
                    initial = new_integrator_state(
                        logp, jnp.array([0.7, -0.4]), jnp.array([0.3, 1.1])
                    )
                    state = initial
                    for _ in range(count):
                        state = integrate(state, step_size)
                    values = [
                        step_size, count, factor[0][0], factor[1][0], factor[1][1],
                        float(inverse[0, 0]), float(inverse[0, 1]), float(inverse[1, 1]),
                        *state_values(initial, kinetic), *state_values(state, kinetic),
                    ]
                    writer.writerow([target_name, mass_name, *values])
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, help="check saved CSV without modifying it")
    parser.add_argument("--m3", action="store_true", help="generate logistic/funnel/rotated references")
    args = parser.parse_args()
    generated = generate(args.m3)
    if args.check:
        # Bit-identical text is intentional here: regeneration check on the pinned
        # environment, not a portable floating-point comparison (Rust uses tolerance).
        if args.check.read_text() != generated:
            sys.exit(f"fixture differs: {args.check}; review numeric/environment changes")
        print(f"{54 if args.m3 else 36} BlackJAX endpoint fixtures reproduce byte-for-byte")
    else:
        sys.stdout.write(generated)


if __name__ == "__main__":
    main()
