#!/usr/bin/env bash
set -euo pipefail

CRATE="alea-mcmc"
BENCH="rwmh"

FEATURE_SETS=(
  ""
  "simd"
  "faer"
  "openblas"
  "simd faer"
  "simd openblas"
  "faer openblas"
  "simd faer openblas"
)

BASE_DIR="$(pwd)/target/criterion_features"
mkdir -p "$BASE_DIR"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

run_one() {
  local features="$1"

  local tag
  if [ -z "$features" ]; then
    tag="baseline"
  else
    tag="$(echo "$features" | tr ' ' '_')"
  fi

  local crit_dir="${BASE_DIR}/${tag}"
  mkdir -p "$crit_dir"

  export CRITERION_HOME="$crit_dir"

  echo "=== [$tag] ==="

  if [ -z "$features" ]; then
    cargo bench \
      --manifest-path crates/Cargo.toml --locked \
      -p "$CRATE" \
      --bench "$BENCH" \
      --no-default-features \
      -- \
      --output-format bencher \
      --quiet
  else
    cargo bench \
      --manifest-path crates/Cargo.toml --locked \
      -p "$CRATE" \
      --bench "$BENCH" \
      --no-default-features \
      --features "$features" \
      -- \
      --output-format bencher \
      --quiet
  fi
}

export -f run_one
export CRATE BENCH BASE_DIR

printf "%s\n" "${FEATURE_SETS[@]}" | \
  xargs -I{} bash -c 'run_one "$@"' _ "{}"

echo "Done."
