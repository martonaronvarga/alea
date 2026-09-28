#!/usr/bin/env bash
set -euo pipefail
# Run inside .#autodiff. A cargo check alone never validates Enzyme lowering.
rustc --version --verbose
args=(--manifest-path crates/Cargo.toml -p kernels --release --features std-autodiff --locked)
cargo test "${args[@]}" --test autodiff --test transforms --test models
cargo clippy "${args[@]}" --all-targets -- -D warnings
cargo run "${args[@]}" --example gradient_check -- all
