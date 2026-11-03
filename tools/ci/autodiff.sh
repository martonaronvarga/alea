#!/usr/bin/env bash
set -euo pipefail
# Execute Enzyme lowering; cargo check alone is not an autodiff gate.
rustc --version --verbose
args=(--manifest-path crates/Cargo.toml --release --locked)
cargo test "${args[@]}" -p alea-autodiff --features std-autodiff --test autodiff --test models --test m14_models --test codegen --test numerical
cargo test "${args[@]}" -p alea-core --test transforms
# A real downstream build script emits the primal and executes nested Enzyme AD.
cargo test --manifest-path tools/enzyme-conformance/Cargo.toml --release --locked
cargo test --manifest-path tools/enzyme-conformance/Cargo.toml --release --features faer --test information --locked
cargo test --manifest-path tools/enzyme-conformance/Cargo.toml --release --features faer --example gaussian_uncertainty --locked
cargo run --manifest-path tools/enzyme-conformance/Cargo.toml --release --features faer --example gaussian_uncertainty --locked | python3 tools/enzyme-conformance/python/gaussian_uncertainty.py
python3 -m unittest discover -s tools/enzyme-conformance/python
cargo clippy --manifest-path tools/enzyme-conformance/Cargo.toml --release --all-targets --locked -- -D warnings
cargo clippy --manifest-path tools/enzyme-conformance/Cargo.toml --release --features faer --all-targets --locked -- -D warnings
cargo clippy "${args[@]}" -p alea-autodiff -p alea-cli --features alea-autodiff/std-autodiff,alea-cli/std-autodiff --all-targets -- -D warnings
cargo run "${args[@]}" -p alea-cli --features std-autodiff --bin alea-gradient-check -- all
