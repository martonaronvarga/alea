#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root inside the selected lock-pinned Nix shell.
features=${1:-}
args=(--manifest-path crates/Cargo.toml --workspace --locked)
if [[ -n "$features" ]]; then args+=(--features "$features"); fi
cargo fmt --manifest-path crates/Cargo.toml --all --check
cargo check "${args[@]}" --all-targets
cargo test "${args[@]}"
cargo clippy "${args[@]}" --all-targets -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc "${args[@]}" --no-deps
