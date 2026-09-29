#!/usr/bin/env bash
set -euo pipefail
export MIRIFLAGS='-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check'
args=(--manifest-path crates/Cargo.toml --locked)
cargo miri test "${args[@]}" -p alea-ffi --lib
cargo miri test "${args[@]}" -p alea-math --lib --test metrics --test interpolation
cargo miri test "${args[@]}" -p alea-core --test target --test transforms --test density
cargo miri test "${args[@]}" -p alea-autodiff --test models
cargo miri test "${args[@]}" -p alea-mcmc --lib --test hmc_chain --test hmc_validation --test hmc_stress --test phase_state --test rwmh --test adaptation
cargo miri test "${args[@]}" -p alea-runtime --test streaming --test collection
cargo miri test "${args[@]}" -p alea-distributions --lib latent::tests
cargo miri test "${args[@]}" -p alea-distributions --test latent
