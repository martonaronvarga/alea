#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root inside the selected lock-pinned Nix shell.
features=${1:-}
args=(--manifest-path crates/Cargo.toml --workspace --locked)
python3 tools/ci/architecture.py
python3 tools/ci/migration.py
python3 tools/ci/test_migration.py
if [[ -n "$features" ]]; then args+=(--features "$features"); fi
cargo fmt --manifest-path crates/Cargo.toml --all --check
cargo check "${args[@]}" --all-targets
cargo test "${args[@]}"
cargo clippy "${args[@]}" --all-targets -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc "${args[@]}" --no-deps
