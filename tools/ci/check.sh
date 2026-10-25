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
  python3 tools/ci/compile_fail.py
  cargo clippy --manifest-path tools/downstream-conformance/Cargo.toml --all-targets --locked --offline -- -D warnings
  cargo clippy --manifest-path tools/compile-fail/Cargo.toml --lib --locked --offline -- -D warnings
  RUSTDOCFLAGS='-D warnings' cargo doc --manifest-path tools/downstream-conformance/Cargo.toml --locked --offline --no-deps
  RUSTDOCFLAGS='-D warnings' cargo doc --manifest-path tools/compile-fail/Cargo.toml --lib --locked --offline --no-deps
  python3 tools/atlas-reference/generate.py --check
  python3 tools/riemannian-reference/newton_fixture.py --check
  python3 tools/riemannian-reference/solver_ladder.py --check
  python3 tools/stan-adaptation-reference/check.py
  python3 tools/fisher-reference/generate.py --check
  python3 tools/fisher-reference/upstream/adjudicate.py --check
  python3 tools/fisher-reference/upstream/test_adjudicate.py
  python3 tools/test_fisher_cost.py
  python3 tools/test_m4f_efficiency.py
  python3 tools/test_m4f_budgets.py
  python3 tools/walnuts-reference/test_reference.py
  python3 tools/test_m7_benchmark.py
  python3 tools/test_m14.py
  cargo build --manifest-path crates/Cargo.toml -p alea-cli --bin alea-bench --locked --offline
  bench_target=$(cargo metadata --manifest-path crates/Cargo.toml --locked --no-deps --format-version 1 | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')
  ALEA_BENCH_TEST_BINARY="$bench_target/debug/alea-bench" ALEA_BENCH_REQUIRE_WORKER=1 python3 -W error::ResourceWarning -m unittest discover -s tools/bench/tests
  python3 tools/test_evidence.py
  python3 tools/evidence.py catalog
  # Full historical replay requires restored payloads, separately from routine
  # synthetic worker/parser contracts above. Never report a catalog as raw replay.
  if [[ "${ALEA_VERIFY_ARCHIVED_EVIDENCE:-0}" == 1 ]]; then
    python3 tools/test_m16.py
    python3 tools/test_m16_pdb.py
  fi
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
