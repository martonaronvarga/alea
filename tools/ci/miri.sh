#!/usr/bin/env bash
set -euo pipefail
export MIRIFLAGS='-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check'
args=(--manifest-path crates/Cargo.toml --locked)
cargo miri test "${args[@]}" -p alea-ffi --lib
cargo miri test "${args[@]}" -p alea-math --lib --test metrics --test interpolation --test low_rank
cargo miri test "${args[@]}" -p alea-core --test target --test transforms --test density --test capability
cargo miri test "${args[@]}" -p alea-autodiff --test models
cargo miri test "${args[@]}" -p alea-mcmc --lib --test hmc_chain --test hmc_validation --test hmc_stress --test phase_state --test rwmh --test adaptation --test warmup --test warmup_reference --test fisher --test integrators --test integrator_failures
cargo miri test "${args[@]}" -p alea-runtime --test streaming --test collection
cargo miri test "${args[@]}" -p alea-distributions --lib latent::tests
cargo miri test "${args[@]}" -p alea-distributions --test latent
# Exact replay is a different contract from numerical robustness under Miri's
# intentionally perturbed transcendental results. Keep both checks.
MIRIFLAGS="$MIRIFLAGS -Zmiri-deterministic-floats" cargo miri test "${args[@]}" -p alea-mcmc --test warmup warmup_freezes_is_deterministic_and_zero_iterations_consume_no_rng -- --ignored
