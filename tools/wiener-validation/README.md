# Wiener kernel validation

This tool replays the parameter grid used by the Stan Math Atlasz support run.
The full profile contains exactly 42,611,184 valid inputs, in the original order.
It uses the public `alea` facade and evaluates densities and fused parameter
scores independently. Finite output is a coverage check, not an accuracy proof.
Independent Arb/Acb reference tests live in the distributions test suite.

```sh
nix develop .#stable --command cargo build --release \
  --manifest-path tools/wiener-validation/Cargo.toml --locked
# Profile, shard index, shard count, per-shard limit (zero means all), new directory.
tools/wiener-validation/target/release/alea-wiener-validation \
  smoke 0 1 0 /tmp/wiener-smoke
# One shard of the full grid:
tools/wiener-validation/target/release/alea-wiener-validation \
  full 0 512 0 /tmp/wiener-full-0
```

Each directory contains a final summary, periodic progress, and every failing
input in JSON Lines format. Directories must be new. A limited or interrupted
run must not be reported as complete. Both evaluation errors and nonfinite
successful values count as failures. Unlike the historical runner, a failed
density does not skip the fused evaluation. The grid uses the upper response,
matching Stan's likelihood convention; separate reference tests cover reflection.

The axes and validity predicate reproduce
`stan_math/test/unit/math/rev/prob/wiener_combinatorial_grid_test.cpp` from the
local numerical research repository. The reaction time is formed as `tau + gap`
in binary64, as in that runner. Every valid case receives a zero-based index;
shard assignment is `index % shard_count`. Tests check the full denominator and
valid enumeration order. Timing comparisons require the same tool source,
Cargo lock, compiler, hardware, input order, and CPU affinity for both revisions.
Retain source and binary hashes, all failures, and incomplete outcomes.

## Timing worker

The `timing` binary accepts one comma-separated job per line:

```text
id,family,rt,alpha,tau,beta,delta,drift_sd,start_width,time_width,lower,precision,operation,samples
```

`family` is 4, 5 or 7; `lower` is 0 or 1. Operations are `density`, `fused`,
`batch_direct` and `batch_bucket`; batches support Wiener4/5 and contain 32
observations. Use positive `samples` (11 in the comparison protocol). The worker
retains failed jobs and reports the median and 95th percentile of batch-average
nanoseconds per observation, along with the evaluated values and scores. These
are not isolated-call latency quantiles. The optional `cdf` feature adds `cdf`
and `tail` operations for revisions exposing those APIs. The `simd` feature
requires the repository's nightly environment.

Aggregate completed or partial shards with denominator and failure-retention checks:

```sh
python tools/wiener-validation/report.py /path/to/shards/*/summary.json
python -m unittest discover -s tools/wiener-validation -p test_report.py
```

Missing shards produce an incomplete verdict. Duplicate shards, inconsistent
counts, and missing failure records are rejected. A finite-output pass is kept
separate from numerical accuracy validation.

To evaluate retained failures once each against another implementation revision:

```sh
tools/wiener-validation/target/release/alea-wiener-validation \
  replay /path/to/failures.jsonl /tmp/replayed-failures.jsonl
```

Replay preserves the input index, parameters and evaluation kind, and records
both successful values and errors. Its denominator is the supplied records;
it must not be described as a new complete support sweep.
