#!/usr/bin/env bash
set -euo pipefail
# Optional external ecosystem checks; no Python/R dependency enters native inference.
export ALEA_ANALYSIS_EXTERNAL=1
python3 -m unittest discover -s tools/downstream-conformance/python
cargo test --manifest-path tools/downstream-conformance/Cargo.toml --test analysis --locked
cargo test --manifest-path tools/downstream-conformance/Cargo.toml --features faer --test regression_assessment --locked
