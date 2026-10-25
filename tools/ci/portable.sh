#!/usr/bin/env bash
set -euo pipefail
# Scalar reference surface, no Nix prerequisite. Integration-test fixtures use
# distributions/cubature and therefore require a C compiler on every host.
args=(--manifest-path crates/Cargo.toml --locked)
cargo check "${args[@]}" -p alea-core -p alea-math -p alea-mcmc -p alea-runtime --lib
cargo test "${args[@]}" -p alea-math --lib --test metrics --test interpolation
cargo test "${args[@]}" -p alea-core --test target --test transforms --test density --test capability --test transport
cargo test "${args[@]}" -p alea-mcmc --test hmc_chain --test nuts_contracts --test rwmh --test adaptation --test warmup --test nuts_warmup --test isokinetic_failures --test static_warmup
cargo test "${args[@]}" -p alea-runtime --test streaming --test collection --test session
RUSTDOCFLAGS='-D warnings' cargo doc "${args[@]}" -p alea-core -p alea-math -p alea-mcmc -p alea-runtime --no-deps
