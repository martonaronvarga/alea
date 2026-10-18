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
if [[ "$scope" == all ]]; then
  python3 tools/ci/architecture.py
  python3 tools/ci/maturity.py
  python3 tools/ci/test_maturity.py
  python3 tools/ci/public_api.py --validate
  python3 tools/ci/test_public_api.py
  python3 tools/ci/exactness.py
  python3 tools/ci/test_exactness.py
  cargo fmt --manifest-path tools/downstream-conformance/Cargo.toml --all --check
  cargo test --manifest-path tools/downstream-conformance/Cargo.toml --locked --offline
  cargo clippy --manifest-path tools/downstream-conformance/Cargo.toml --all-targets --locked --offline -- -D warnings
  RUSTDOCFLAGS='-D warnings' cargo doc --manifest-path tools/downstream-conformance/Cargo.toml --locked --offline --no-deps
  python3 tools/fisher-reference/generate.py --check
  python3 tools/fisher-reference/upstream/adjudicate.py --check
  python3 tools/fisher-reference/upstream/test_adjudicate.py
  python3 tools/test_fisher_cost.py
  python3 tools/test_m4f_efficiency.py
  python3 tools/test_m4f_budgets.py
  python3 tools/walnuts-reference/test_reference.py
  python3 tools/test_m7_benchmark.py
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
