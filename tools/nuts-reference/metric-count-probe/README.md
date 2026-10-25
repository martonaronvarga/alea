# NUTS metric-operation count

This original standalone consumer wraps Alea's public identity metric with
counters. It runs one fixed-step, fixed-geometry NUTS transition for each of nine
dimension/depth combinations, from the origin of a standard Gaussian. The tiny
step avoids turning, so complete trees have 1, 7 and 255 new gradient calls.
Initialization is excluded. Assertions now verify `velocity_calls = 1 + 2 * edges`
after endpoint velocity caching. The earlier review recorded `1 + 8 * edges`;
its output remains historical evidence of the pre-implementation sources.

This is an operation-count probe, not a timing harness. The wrapper may prevent
compiler optimization that a concrete identity metric would permit. It does not
measure nuts-rs or Stan and does not predict a walltime speedup.

From the repository root:

```sh
CARGO_TARGET_DIR=target/reference-review nix develop .#stable --command \
  cargo run --quiet --locked --manifest-path tools/nuts-reference/metric-count-probe/Cargo.toml
```

The original reviewed output is
`docs/benchmarks/nuts-reference-review-2026-10-22/metric-counts.csv`.
The conclusions and pinned source references are in
`docs/nuts-reference-review-2026-10-22.md`. This tool intentionally remains outside
the production workspace and adds no library dependencies or API.
