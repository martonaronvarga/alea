#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root inside the selected lock-pinned Nix shell.
features=${1:-}
scope=${2:-all}
if [[ "$scope" != all && "$scope" != workspace ]]; then
  echo "usage: $0 [features] [all|workspace]" >&2
  exit 2
fi
args=(--manifest-path crates/Cargo.toml --workspace --locked)
cargo fetch --manifest-path crates/Cargo.toml --locked
if [[ "$scope" == all ]]; then
  python3 tools/ci/architecture.py
  python3 tools/ci/test_run.py
  python3 tools/ci/maturity.py
  python3 tools/ci/test_maturity.py
  python3 tools/ci/public_api.py --validate
  python3 tools/ci/test_public_api.py
  python3 tools/ci/api_compatibility.py
  python3 tools/ci/test_api_compatibility.py
  python3 tools/ci/exactness.py
  python3 tools/ci/test_exactness.py
  python3 tools/release/metadata.py
  python3 -m unittest discover -s tools/release -p 'test_*.py'
  cargo fmt --manifest-path tools/downstream-conformance/Cargo.toml --all --check
  cargo fmt --manifest-path tools/compile-fail/Cargo.toml --all --check
  cargo fetch --manifest-path tools/downstream-conformance/Cargo.toml --locked
  cargo fetch --manifest-path tools/compile-fail/Cargo.toml --locked
  cargo test --manifest-path tools/downstream-conformance/Cargo.toml --locked --offline
  cargo test --manifest-path tools/downstream-conformance/Cargo.toml --features tracing --test tracing --locked --offline
  cargo test --manifest-path tools/downstream-conformance/Cargo.toml --features faer --test fit_assessment --locked --offline
  python3 -m unittest discover -s tools/downstream-conformance/python
  cargo test --manifest-path crates/Cargo.toml -p alea-runtime --features analysis --test analysis --locked --offline
  cargo test --manifest-path crates/Cargo.toml -p alea-runtime --features tracing,analysis --locked --offline
  cargo clippy --manifest-path crates/Cargo.toml -p alea-runtime --features analysis --all-targets --locked --offline -- -D warnings
  python3 tools/ci/compile_fail.py
  cargo clippy --manifest-path tools/downstream-conformance/Cargo.toml --all-targets --locked --offline -- -D warnings
  cargo clippy --manifest-path tools/compile-fail/Cargo.toml --lib --locked --offline -- -D warnings
  RUSTDOCFLAGS='-D warnings' cargo doc --manifest-path tools/downstream-conformance/Cargo.toml --locked --offline --no-deps
  RUSTDOCFLAGS='-D warnings' cargo doc --manifest-path tools/compile-fail/Cargo.toml --lib --locked --offline --no-deps
  cargo fmt --manifest-path tools/documentation/Cargo.toml --all --check
  cargo fetch --manifest-path tools/documentation/Cargo.toml --locked
  cargo test --manifest-path tools/documentation/Cargo.toml --doc --locked --offline
  cargo fmt --manifest-path crates/Cargo.toml --all --check
fi
if [[ -n "$features" ]]; then args+=(--features "$features"); fi
cargo check "${args[@]}" --all-targets
cargo test "${args[@]}"
cargo clippy "${args[@]}" --all-targets -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc "${args[@]}" --no-deps
if command -v cargo-public-api >/dev/null 2>&1; then
  python3 tools/ci/public_api.py --check
fi
