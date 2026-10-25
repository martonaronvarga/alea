#!/usr/bin/env bash
set -euo pipefail
# A baseline must be an independently retained, unpacked accepted release source.
# Never substitute HEAD for a missing previous release.
initial=false
if [[ "${1:-}" == --initial ]]; then initial=true; shift; fi
baseline=${1:?usage: semver.sh [--initial] ACCEPTED_RELEASE_SOURCE_DIRECTORY}
test -f "$baseline/crates/Cargo.toml"
if [[ "$initial" == true ]]; then
  # The unpublished API may deliberately break. Verify both actual endpoints of
  # its accepted review instead of reporting a fabricated semver success.
  python3 tools/release/initial.py "$baseline"
  exit
fi
command -v cargo-semver-checks >/dev/null
for package in alea-core alea-math alea-mcmc alea-runtime; do
  # Explicit selection is essential: cargo-semver-checks skips publish=false
  # packages by default and otherwise exits successfully without checking them.
  cargo semver-checks check-release --manifest-path "crates/$package/Cargo.toml" --package "$package" \
    --baseline-root "$baseline/crates/$package" --default-features
done
