#!/usr/bin/env bash
set -euo pipefail
# Execute Enzyme lowering; cargo check alone is not an autodiff gate.
rustc --version --verbose
args=(--manifest-path crates/Cargo.toml --release --locked)
cargo test "${args[@]}" -p alea-autodiff --features std-autodiff --test autodiff --test models --test m14_models --test codegen
cargo test "${args[@]}" -p alea-core --test transforms
# A real downstream build script emits the primal and executes nested Enzyme AD.
cargo test --manifest-path tools/enzyme-conformance/Cargo.toml --release --locked
cargo clippy --manifest-path tools/enzyme-conformance/Cargo.toml --release --all-targets --locked -- -D warnings
cargo clippy "${args[@]}" -p alea-autodiff -p alea-cli --features alea-autodiff/std-autodiff,alea-cli/std-autodiff --all-targets -- -D warnings
cargo run "${args[@]}" -p alea-cli --features std-autodiff --bin alea-gradient-check -- all
