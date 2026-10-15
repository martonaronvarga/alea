# Fisher fitting: profile-guided optimization (2026-10-02)

Historical timings: the [2026-10-05 numerical repair](../fisher-noncommuting.md)
replaced the middle eigensolve with Cholesky/SVD factors. Sequential execution
and reuse of the F eigensystem remain, but these cost measurements must be rerun
before describing the current fitter's performance. The
[October 5 remeasurement](fisher-cost-2026-10-05.md) supplies current timing and
native peak-RSS observations for five fixed workloads; it is not a paired
comparison against the old executable.

This increment changes systems execution, not the Fisher estimator, ridge, rank
selection, conditioning rejection, warmup schedule, or acceptance policy. It
does not establish an ESS improvement or a universally optimal implementation.

## Evidence and changes

The fixed cost-v1 correlated target exposed unnecessary work in cold fitting.
Tools: perf 6.19.11 (hardware cycles, instruction counts and DWARF stacks),
Hyperfine 1.20.0, Valgrind 3.26.0 Callgrind/Massif, and Criterion 0.5.1.
No kernel permissions were changed.

1. Baseline perf sampling attributed 57.49% of user cycles to
   `spindle::barrier::Barrier::wait_and_clear_while`, 9.22% to
   `wait_and_clear`, and 6.36% to `spindle::worker_loop`. A separate
   `RAYON_NUM_THREADS=1` experiment improved a 50-fit d64/history64 batch from
   420.3 ± 44.9 ms to 150.6 ± 5.2 ms (six runs). This implicated nested worker
   synchronization rather than the scalar target.
2. Fitting now passes `Par::Seq` explicitly to Faer products and eigensolves.
   It neither reads nor modifies process-global parallelism. A separate test
   binary disables global policy access, runs a real fit, and checks the policy
   remains disabled. Concurrent chains remain free to execute independently.
3. In the single-worker baseline profile, `pow` routines accounted for about
   12% of sampled cycles. Each eigenvalue power was recomputed for every entry
   in its eigenvector; the same Fisher matrix was also decomposed twice.
   The two powers now share one validated eigensystem, and each eigenvalue power
   is computed once per column. The same `powf` operation is retained; no
   reciprocal/square-root approximation, fast math, or relaxed check was added.

The final profile has no worker barriers or `pow` routine above its 1% display
threshold. Remaining leading costs include fit/basis work, tridiagonal
eigenvalue iterations, SIMD kernels and matrix products. Diagonal warmup was
profiled separately: log1p/exp/log are substantial costs. Its robust log-space
regularization remains unchanged; bypassing it needs a separate numerical study.

## Controlled comparison

[Hyperfine records](fisher-profile-2026-10-02.json) retain individual observations,
user/system time and command lines. Eight runs per command, two warmups, same
symbolized binaries throughout, no compilation or other profiling jobs during
timing. Results include process startup and fixed setup once per batch.

| Workload | Before (mean ± SD) | After (mean ± SD) | Observation |
|---|---:|---:|---|
| 50 fits, d64/history64 | 253.1 ± 14.1 ms | 103.8 ± 0.9 ms | ~2.44× faster |
| 10 complete low-rank warmups, d64 | 217.7 ± 42.9 ms | 104.8 ± 11.9 ms | ~2.08× faster |
| 20 fits, d256/history160 | 1326 ± 20 ms | 1379 ± 132 ms | no clear wall-time gain |

The intermediate sequential-only d64 fitting batch took 137.4 ± 0.9 ms in this
same experiment: eliminating repeated eigensolves/powers contributed an
additional ~1.32× improvement. Hyperfine's automatic overall ranking mixes
different workloads; only paired workloads above should be compared.

perf stat, three runs of 50 d64/history64 fits (user-space events):

| Metric | Before | After |
|---|---:|---:|
| Task CPU time | 797.48 ms | 110.91 ms |
| Cycles | 1,565,798,900 | 312,696,923 |
| Instructions | 1,013,882,437 | 649,097,193 |
| Cache misses | 671,733 | 589,045 |

Callgrind, three fits with `RAYON_NUM_THREADS=1` in both binaries, recorded
54,746,400 → 41,484,825 instructions (~24% fewer). These counts include setup;
they are instrumentation results, not timings or hardware instruction counts.
Massif on the same workload observed maximum heap use of 13,142,407 →
13,185,215 bytes: effectively unchanged, slightly higher. This optimization
does **not** establish lower memory use or allocation-free fitting.

The [complete 18-case Criterion rerun](warmup-cost-optimized-2026-10-02.csv)
retains 95% bootstrap intervals. The original [protocol](warmup-cost.md) and
original results are preserved. Unchanged covariance/diagonal controls also ran
faster than the earlier session, demonstrating machine/run variability. Do not
attribute all cross-session changes to the code; prefer the paired Hyperfine
comparison. Shared laptop, no affinity/frequency controls, no ESS measurement.

Sequential execution is deliberate for current chain-local fits, not a claim
that serial dense algebra wins at every dimension. The larger test shows the
tradeoff. A future configurable parallel fit needs independent crossover and
multi-chain throughput measurements, not a process-global toggle.

Validation: full Faer workspace check/tests/Clippy/rustdoc and offline independent
reference generation checks; targeted SIMD/Faer/OpenBLAS spectrum, upstream,
metric, warmup and controller suites; strict-Miri execution of the new fitting
policy regression. Fixture seeds and tolerances were not changed. This was not
a fresh Enzyme or full Miri-matrix qualification.

## Reproduction

Use the lightweight lock-pinned shell (it avoids the full research/R/Python stack):

```sh
nix develop .#profiling
CARGO_PROFILE_RELEASE_DEBUG=1 cargo build --manifest-path crates/Cargo.toml -p alea-mcmc --features faer --release --example fisher_profile
```

The driver supports `fit REPETITIONS DIMENSION HISTORY` and
`covariance|diagonal|lowrank REPETITIONS DIMENSION`. It uses the cost-v1 target,
initialization, seed, and budgets; fit input construction occurs once per process.
Errors terminate rather than dropping failed cases. A missing Faer feature
rejects `fit`/`lowrank` instead of substituting another method.

```sh
perf record -e cycles:u -F 499 --call-graph dwarf -o fit.perf -- crates/target/release/examples/fisher_profile fit 1000 64 64
perf report --stdio --no-children --call-graph none -i fit.perf
perf stat -r 3 -e task-clock,cycles,instructions,cache-misses -- crates/target/release/examples/fisher_profile fit 50 64 64
hyperfine --warmup 2 --runs 8 --export-json timings.json 'crates/target/release/examples/fisher_profile fit 50 64 64'
RAYON_NUM_THREADS=1 valgrind --tool=callgrind --callgrind-out-file=fit.callgrind crates/target/release/examples/fisher_profile fit 3 64 64
callgrind_annotate --auto=no fit.callgrind
RAYON_NUM_THREADS=1 valgrind --tool=massif --massif-out-file=fit.massif crates/target/release/examples/fisher_profile fit 3 64 64
ms_print fit.massif
cargo bench --manifest-path crates/Cargo.toml -p alea-mcmc --features faer --bench warmup -- _cost_v1 --noplot --save-baseline fisher-optimized
```

Copy the baseline executable before rebuilding to make paired comparisons.
Run timing, sampling and instrumentation separately. Profilers perturb execution;
never interpret Valgrind runtimes as native performance. Kernel restrictions may
prevent perf elsewhere; do not silently substitute another event or weaken host
security settings. Keep profiling dumps outside tracked source directories.

Environment: i5-8250U, Linux x86-64, rustc 1.95.0 (59807616e, LLVM 22.1.2),
Faer 0.24.0, release fat LTO, debug information level 1. No native-CPU flag,
SIMD/OpenBLAS feature or persistent global threading override was enabled.
Base commit remains `59ef802` plus the working-tree Fisher changes; it alone
is not the profiled source. SHA-256 fingerprints:

```text
aa3d91809211919be040d9eadb5746a6a441251157541149510a9870b62f1f23  before: crates/alea-math/src/fisher.rs
081eaf000579c32e4256e2e26f1e6c0e2844fd062b4a6ab1cd576c1ae643dd83  after: crates/alea-math/src/fisher.rs
adb1a0e07ca32e4a9cdf30b78b01f7e24175e2e3d5302249e02201bdf9a4348d  crates/alea-mcmc/examples/fisher_profile.rs
989fe994b3768c547519dabdfab58742828c0fd9541385fbbcb450337003b205  before executable
5c9fa3b46e5452366c8f3460a707795d5db93a053f107057fcaa1923c0051d79  after executable
```
