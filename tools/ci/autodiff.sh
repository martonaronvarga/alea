#!/usr/bin/env bash
set -euo pipefail
# Execute Enzyme lowering; cargo check alone is not an autodiff gate.
rustc --version --verbose
args=(--manifest-path crates/Cargo.toml --release --locked)
cargo test "${args[@]}" -p alea-autodiff --features std-autodiff --test autodiff --test models
cargo test "${args[@]}" -p alea-core --test transforms
cargo clippy "${args[@]}" -p alea-autodiff -p alea-cli --features alea-autodiff/std-autodiff,alea-cli/std-autodiff --all-targets -- -D warnings
cargo run "${args[@]}" -p alea-cli --features std-autodiff --bin alea-gradient-check -- all
