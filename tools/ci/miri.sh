#!/usr/bin/env bash
set -euo pipefail
export MIRIFLAGS='-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check'
args=(--manifest-path crates/Cargo.toml --locked)
cargo miri test "${args[@]}" -p ffi --lib
cargo miri test "${args[@]}" -p memory --lib
cargo miri test "${args[@]}" -p kernels --test metrics --test target --test transforms --test models
cargo miri test "${args[@]}" -p runtime --lib
cargo miri test "${args[@]}" -p runtime --test hmc_chain --test hmc_validation --test phase_state --test rwmh_validation
