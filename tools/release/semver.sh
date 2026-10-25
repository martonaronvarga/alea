#!/usr/bin/env bash
set -euo pipefail
# A baseline must be an independently retained, unpacked accepted release source.
# Never substitute HEAD for a missing previous release.
baseline=${1:?usage: semver.sh ACCEPTED_RELEASE_SOURCE_DIRECTORY}
test -f "$baseline/crates/Cargo.toml"
command -v cargo-semver-checks >/dev/null
for package in alea-core alea-math alea-mcmc alea-runtime; do
  # Explicit selection is essential: cargo-semver-checks skips publish=false
  # packages by default and otherwise exits successfully without checking them.
  cargo semver-checks check-release --manifest-path "crates/$package/Cargo.toml" --package "$package" \
    --baseline-root "$baseline/crates/$package" --default-features
done
