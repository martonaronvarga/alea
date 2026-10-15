# Fisher cost after the Cholesky/SVD repair

The [numerical repair](../fisher-noncommuting.md) now has a fresh systems-cost
baseline. This closes the outstanding remeasurement task, not M4F promotion.
The repaired implementation passes the existing accuracy gates; no numerical
policy or production code was changed during this measurement increment.

## Fixed protocol

`tools/fisher-cost.py` runs the existing `fisher_profile` executable on five
preselected workloads. There are two unrecorded warmup rounds and eight measured
rounds. Each round executes every workload serially, alternating forward/reverse
order. No compilation or profiler runs alongside the measurements. The fit and
warmup inputs, seed 7, rank cap 4, ridge, thresholds and budgets are inherited
unchanged from the [cost-v1 workloads](warmup-cost.md).

Wall time uses Python's monotonic performance counter around GNU time and the
child process. It includes process startup, driver setup and destruction.
GNU time records user/system CPU seconds and maximum resident set size for each
child separately. Peak RSS includes code, libraries and allocator effects; it
is not peak heap use, live metric storage or a cumulative maximum across runs.
CPU times have GNU time's coarse decimal resolution. Raw observations and
binary/source SHA-256 fingerprints are in [the JSON artifact](fisher-cost-2026-10-05.json).

The runner aborts on failed workloads, refuses to overwrite an existing artifact,
and checks for source/binary changes before publishing results. Five unit tests
cover warmup exclusion/round ordering, resource parsing, failure publication,
overwrite protection and changed-binary detection; normal CI runs these tests.
Timing itself is not a pass/fail CI threshold.

## Observations

Times below are batch means and sample standard deviations over eight runs.
The per-operation column divides the batch mean by its repetition count; it is
not an isolated kernel timing. RSS is the observed range, not a confidence bound.

| Workload | Batch wall time, mean ± SD | Amortized per operation | Peak RSS, KiB |
|---|---:|---:|---:|
| 50 fits, d64/history64 | 175.100 ± 2.110 ms | 3.502 ms | 4,548–4,608 |
| 20 fits, d256/history160 | 2,084.583 ± 25.499 ms | 104.229 ms | 15,688–15,740 |
| 10 covariance diagonal warmups, d64 | 17.238 ± 2.518 ms | 1.724 ms | 2,856–3,044 |
| 10 Fisher diagonal warmups, d64 | 32.188 ± 0.580 ms | 3.219 ms | 3,172–3,172 |
| 10 Fisher rank-cap-4 warmups, d64 | 158.912 ± 2.908 ms | 15.891 ms | 4,752–4,904 |

All warmups use the correlated Gaussian and 200 transitions of five leapfrog
steps. These are equal transition budgets, not equal schedules or gradient
budgets. Fit microbenchmarks use synthetic histories and must not be added to
warmup time, which already includes fitting on its own evolving histories.

The separately executed [actual-call CSV](warmup-calls-2026-10-05.csv) contains
all 36 cost-v1 cases. For correlated d64/seed 7, each single covariance,
Fisher-diagonal and rank-cap-4 warmup used respectively 1,011/1,005/1,005 fused
target calls and reported 20/15/10 warmup divergences. The final low-rank metric
had rank 1, not 4. These unsuccessful proposals remain included in all costs;
no retained draws or convergence claim is attached to this short warmup.

The current fit and low-rank warmup times are higher than the October 2 report.
There is no same-session pre-repair executable comparison here: host/session,
measurement wrapper and implementation all differ from that report. Do not
attribute the difference quantitatively to Cholesky/SVD. The old Massif heap
figures also cannot be compared to these native RSS measurements. No outliers
were dropped; the covariance batch includes a 23.072 ms observation.

Environment: Intel i5-8250U, Linux 7.2.3 x86-64/glibc 2.42, lock-pinned Nix
profiling shell, rustc 1.95.0, Faer 0.24.0. Built with release fat LTO and
`CARGO_PROFILE_RELEASE_DEBUG=1`; no SIMD/OpenBLAS/native-CPU feature was enabled.
`OPENBLAS_NUM_THREADS=1` was inherited but the OpenBLAS backend was not compiled.
The JSON environment describes measurement-time variables; the debug build
override was scoped to the build command. No affinity/frequency control or
dedicated-host isolation was applied. This is a local shared-machine baseline.

## Reproduction

```sh
nix develop .#profiling
CARGO_PROFILE_RELEASE_DEBUG=1 cargo build --manifest-path crates/Cargo.toml --locked \
  -p alea-mcmc --features faer --release --example fisher_profile
python3 tools/test_fisher_cost.py
python3 tools/fisher-cost.py crates/target/release/examples/fisher_profile /tmp/fisher-cost-new.json
```

Use a fresh output path. The script requires Linux, Python 3.11 or newer and GNU
time; all are supplied by the pinned environment. Rebuild immediately before
measurement: the runtime source fingerprints detect changes during a run, but
cannot prove an arbitrary supplied executable was built from those sources.

Retained-draw quality per total cost, non-Gaussian robustness, history/rank
sensitivity and independent ensembles remain open. These measurements justify
neither a Fisher default nor removal of correctness checks.
