# Warmup cost baseline, version 1

This is a systems-cost experiment, not evidence of sampling efficiency or a
Fisher promotion gate by itself. No retained draws, ESS, convergence diagnostic,
peak-memory estimate, or cross-engine comparison is measured here.

Follow-up: [profile-guided fitting optimization](fisher-profiling.md) records
paired measurements, CPU/heap profiles, limitations and the unchanged-cost-v1 rerun.
The [October 5 report](fisher-cost-2026-10-05.md) records timing/RSS after the
Cholesky/SVD repair and regenerates all 36 actual-call cases.

## Fixed workload

`crates/alea-mcmc/benches/support/warmup.rs` is shared by Criterion, the
`warmup_cost` CSV example, and `tests/warmup_cost.rs`. The tests independently
check target quadratic forms/finite-difference gradients, actual evaluation
budgets, deterministic replay, and equivalence with/without instrumentation.

- Dimensions: 8 and 64. Isotropic precision `I`, or correlated precision
  `diag(1/4, 1, 4, ...) + 9 u u^T`, where `u_i = 1/sqrt(d)`.
- All positions start at `q_i = 0.25`, with identity initial geometry, initial
  step size 0.1, and 200 warmup transitions of five leapfrog steps each.
- Policies: covariance diagonal, Fisher diagonal, Fisher with output rank cap 4.
  Each uses its existing dual-averaging/search/metric schedule unchanged. These
  are equal transition budgets, **not equal adaptation schedules or gradient
  budgets**; covariance searches again at metric boundaries, Fisher does not.
- Timing uses seed 7, reset on every repetition; counter runs use seeds 7, 19,
  and 43. Identical seeds do not imply matched trajectories across policies.
  Cases and budgets were fixed before collecting the first timing results.

`warmup_cost_v1` times construction, owned storage, initial evaluation, search,
adaptation, transitions, cold metric fitting, and destruction. Immutable target
construction is outside timing. There is no counter wrapper in timed runs.
Results and final state are black-boxed. Each Criterion case uses 20 samples,
300 ms warmup, and a requested one-second measurement interval.

`fisher_fit_cost_v1` separately times one fit (including its allocations and
result destruction), dimensions 8/64/256, histories 16/64, ridge 1e-5, cutoff 2,
rank cap 4. Inputs are fixed seed-7 independent uniform coordinates on [-1,1)
with scores from the correlated target above, **not posterior samples**.
History construction and diagonal scale estimation are outside timing. The
result rank does not limit the joint-subspace dimension that gets factored.
This microbenchmark must not be added to total warmup time: warmup already
includes its own fitting work on different histories.

## Reproduction and counters

From the repository root:

```sh
nix develop .#stable --command cargo test --manifest-path crates/Cargo.toml -p alea-mcmc --features faer --test warmup_cost
nix develop .#stable --command cargo run --manifest-path crates/Cargo.toml -p alea-mcmc --features faer --release --example warmup_cost > warmup-calls.csv
nix develop .#stable --command cargo bench --manifest-path crates/Cargo.toml -p alea-mcmc --features faer --bench warmup -- _cost_v1 --noplot
```

Without `faer`, the low-rank policy and fitting benchmarks are omitted, not
substituted with a diagonal implementation. Criterion stores raw measurements
and estimates under `crates/target/criterion/`.

CSV schema 1 records scenario identifiers, actual fused backend entries
(including construction and search), attempted macrosteps, search probes,
metric updates, accepted transitions, divergences, final rank, step size, and
log density. `metric_updates` is the controller's geometry-update count, **not**
the number of spectral factorizations. `integration_attempts` is not a gradient
counter: attempts can fail before reaching the target. Numerical divergences
remain in the records. A controller error aborts the runner with a nonzero exit;
check the exit status before using a potentially partial CSV. The tests are
bounded workload checks, not a universal equality between attempts and calls.

## Interpretation limits

### Local observation, 2026-10-02

Measured on Intel i5-8250U (4 cores/8 logical CPUs), Linux x86-64, Nix `stable`,
rustc 1.95.0 (59807616e, LLVM 22.1.2), `alea-mcmc/faer`, release/bench fat LTO.
No SIMD/OpenBLAS/native-CPU flags or explicit thread/affinity/frequency controls
were set. Faer's default feature/thread settings were retained. This is a single
local run on a shared machine, not a dedicated-host performance qualification.

[Timing estimates](warmup-cost-2026-10-02.csv) retain Criterion's slope estimate
or mean (flat sampling), with its 95% bootstrap interval, in nanoseconds.
[Actual call records](warmup-calls-2026-10-02.csv) contain all 36 scenarios,
including warmup divergences. No failed run was omitted; all controllers returned.

| Target/dimension | Covariance diagonal | Fisher diagonal | Fisher rank ≤4 |
|---|---:|---:|---:|
| Isotropic/8 | 0.188 ms | 0.382 ms | 0.864 ms |
| Isotropic/64 | 1.040 ms | 2.374 ms | 28.341 ms |
| Correlated/8 | 0.202 ms | 0.421 ms | 1.002 ms |
| Correlated/64 | 1.222 ms | 2.714 ms | 26.828 ms |

The correlated/64 low-rank interval is 25.681–28.017 ms. Actual calls are
1009–1011 for covariance and 1005 for either Fisher policy on that target.
Cold fitting alone grows from approximately 2.26 ms (d=256, history=16) to
44.08 ms (history=64), despite an unchanged output rank cap of four. Variability
is substantial: the latter interval is 38.98–49.78 ms. Output rank zero on the
isotropic warmup also does not eliminate the cost of discovering that rank.
These observations justify profiling fitting overhead; they neither identify
the specific bottleneck nor show which policy yields better effective samples.

Source provenance: committed base `59ef802`, plus the working-tree Fisher
validation changes and workload files fingerprinted below. The base commit
alone does not reproduce this experiment; preserve these files with the result.

```text
7eca6ab6c18ac495e0a317626c8c7b76a0c625cd53a445274002634b3e7ba476  flake.lock
4083a163525dc85fa1782bd245a5690f3555a1e555955cacd295385e1b8e8020  crates/Cargo.lock
aa3d91809211919be040d9eadb5746a6a441251157541149510a9870b62f1f23  crates/alea-math/src/fisher.rs
3ed0b2ba958bbf5220a9623bbfd632d182d4fd1d058f63fbbb4458abe5c0c2e6  crates/alea-mcmc/src/adapt/fisher.rs
aa86aebcdb5ec8830cbb9f617f1a51613704f3270e071c590b02a8472cb128ed  crates/alea-mcmc/benches/warmup.rs
c0c1b991dce3dbd63300be0bcefd4ad28cfaedbb545d7a1cc985c80733c010f4  crates/alea-mcmc/benches/support/warmup.rs
2611449fc1aa23bead2f09dfe1689f4bd4abef9902878a1f12b66e845147ca4e  crates/alea-mcmc/examples/warmup_cost.rs
```

Warmup divergences are visible, not grounds for silently dropping cases. A
successful finite run says nothing about typical-set recovery. Fixed-length
Gaussian HMC can also resonate at an adapted trajectory length.

Before promoting Fisher, add retained-draw quality and total-time evidence on
predeclared non-Gaussian targets, independent diagnostics, and memory accounting.
Do not optimize away validation or select a production default from these costs.
