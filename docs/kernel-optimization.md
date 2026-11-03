# Kernel optimization: measured work, preserved mathematics

This repository-wide kernel inventory follows the
[native-model case study](native-model-performance.md). It distinguishes source
triage from profiling and profiling from a qualified optimization. It does **not**
claim every kernel has been exhaustively profiled, made SIMD, or proven optimal.

## Method and references

The [Rust Performance Book](https://nnethercote.github.io/perf-book/) informs this
pass: measure release builds, inspect hot code, preserve bounds checks through
slice/iterator structure, reuse memory, and benchmark inlining rather than adding
attributes everywhere. Consult its [profiling](https://nnethercote.github.io/perf-book/profiling.html),
[build configuration](https://nnethercote.github.io/perf-book/build-configuration.html),
[inlining](https://nnethercote.github.io/perf-book/inlining.html),
[bounds checks](https://nnethercote.github.io/perf-book/bounds-checks.html), and
[iterators](https://nnethercote.github.io/perf-book/iterators.html) chapters.
Existing fat LTO and single codegen unit are held fixed. We do not enable
fast-math, force a host-only ISA globally, or replace panic unwinding with abort;
the latter would break Alea's error/panic recovery contracts.

Relevant primary numerical research:

- Shibata and Petrogalli, [SLEEF](https://arxiv.org/abs/2001.09258) (2020):
  portable vector intrinsics, range reduction, distinct accuracy objectives and
  explicit special-input handling. This informs the kernel structure, not a
  claim that Alea contains SLEEF or inherits its accuracy guarantees.
- Anderson et al., [Correctly Rounded Functions For Vector Applications](https://arxiv.org/abs/2605.15547)
  (May 2026): most results concern single precision; double-precision experiments
  use accurate vector main paths and selective slower fallbacks. Their correct-
  rounding guarantees cannot be transferred to an unrelated f64 polynomial.
- Park and Nagarakatte, [Fast Trigonometric Functions using the RLIBM Approach](https://arxiv.org/abs/2510.13426)
  (2025): range-reduction error matters, and the reported representation coverage
  is up to 32 bits. This is not authorization to replace f64 scientific kernels
  with those approximations or low-precision arithmetic.

“Avoid scalar work” means eliminate unnecessary independent scalar work. It
does not mean discard compensated residuals, reorder a certified turning test,
change a recurrence, or vectorize a tiny loop despite measured dispatch overhead.

## Native regression execution plans

Native Gaussian, Bernoulli-logit and Poisson-log descriptors now retain an
immutable `RegressionPlan` beside their prepared design. Existing public modeling
helpers still select `RegressionExecution::Auto`. Pulp CPU selection is cached at
preparation; one dispatch branch remains per fused evaluation. Explicit `Scalar`
executes the same kernel with scalar lanes. Neither mode replaces distributions,
analytic derivatives, transforms or the sampler.

This is an inspectable dispatch foundation, **not a measured cost-based planner**.
No threaded-observation backend, Faer substitution, autotuner or
performance improvement is established by caching dispatch. Plans report actual
dimensions, selected lane width, four-row tiles, broadcast eligibility and packed
element storage. The byte count excludes original data, allocator overhead,
family constants and per-execution score buffers. It is not peak RSS or a total
memory estimate. Selected lane width does not mean every operation uses that
width; scalar tails and representation-limit fallbacks remain.

Automatic dispatch bypasses SIMD only when N is smaller than the four-row tile
and p is smaller than the available coefficient packet. There is then no full
vector packet on either execution axis. The bypass excludes widths larger than
the tile, which have distinct existing normal fallbacks, and explicit scalar
requests are unchanged. `is_scalar_for_shape()` exposes the decision. This is a
structural eligibility rule, not a fitted performance threshold or a new sum
policy. Raw tests compare it bit-for-bit with the original freshly dispatched
kernels, including domain/overflow errors and extreme normal scales.

The plan is fixed for a descriptor's lifetime. Consuming data rebinding preserves
the requested policy and rebuilds data-dependent packing, including transitions
between unit-column broadcast and offset designs. Failed rebinding retains the
original descriptor, source and plan. Generated Rust source remains a separate
execution path: its capabilities and floating-point behavior are not inferred
from native-plan metadata.

### Correctness gates

`tests/regression_execution.rs` exercises both policies across all three families,
ten row sizes (1 through 10,000), six column counts including zero, complete tiles
and tails. It compares values/scores, central finite differences for each backend,
repeatability and zero retained Rust allocations. Additional tests cover plan
inspection, packing changes, policy retention, shape/domain/overflow failures,
compiled-output staging, recovery, logit tail scores and underflowed Poisson
rates. An internal control compares cached automatic dispatch bit-for-bit with
the previous freshly detected dispatch for the same kernels. Agreement with a
forced backend is not itself an independent mathematical oracle; retain the
existing model-density, Decimal, finite-difference and chart/Jacobian tests.

### Candidate benchmark protocol

Run the default matrix with the pinned stable environment:

```sh
nix develop .#stable --command cargo bench --manifest-path crates/Cargo.toml \
  -p alea-distributions --bench regression_execution --locked
```

The matrix covers all three families, N=3/1,000/10,000 and p=0/1/4/16/64. Separate
`build` groups include descriptor preparation/destruction, data-sharing increments,
packed allocation and Poisson constants, but exclude original dataset/builder
construction. `evaluate` groups time fused value/gradient calls with retained
scratch, excluding construction and serialization. Their throughput unit is one
complete evaluation, not one observation, leapfrog step or retained draw. Both
candidates pass a value/score comparison before evaluation timing. Inputs are a
fixed deterministic synthetic design; this is not the historical Stan dataset.

For a small harness smoke run only:

```sh
nix develop .#stable --command env ALEA_REGRESSION_PLAN_QUICK=1 \
  cargo bench --manifest-path crates/Cargo.toml -p alea-distributions \
  --bench regression_execution --locked
```

Before deriving selection thresholds, record source/toolchain/CPU/flags and raw
Criterion estimates, alternate candidate order across independent pinned-CPU
processes, quantify uncertainty and test held-out shapes. Account for packing
amortization, workspace memory and concurrent chains. Measure selection regret
against the fastest **numerically admitted** candidate rather than assuming wider
SIMD wins. Scheduling must not choose a reduction order from task completion
order. Hardware timing stays outside correctness-CI pass/fail decisions.

These microbenchmarks do not compare old/new Alea or Stan, establish ESS/sec, or
qualify a new inference default. Threading and automatic cost-based selection
remain subsequent measured increments.

### Repeated, source-sealed candidate campaigns

`tools/native-benchmark/scripts/cpu_plan.py` runs the same Criterion executable,
not another evaluator. It creates a deterministic source archive using the
release source policy, compiles with the locked stable command, copies the exact
binary and records source/binary hashes, compiler, CPU, flags and thread settings.
Compilation is separate from measurement. Source and binary checks bracket each
repetition. Output directories must be new and outside the checkout.

```sh
nix develop .#stable --command python3 tools/native-benchmark/scripts/cpu_plan.py \
  --output /tmp/alea-cpu-plan-calibration --cpu 2 --reps 7
nix develop .#stable --command python3 tools/native-benchmark/scripts/cpu_plan.py \
  --output /tmp/alea-cpu-plan-validation --cpu 2 --suite validation --reps 3
```

The calibration grid is the 45 family/shape combinations above. The disjoint
validation grid uses N=1/7/127/2,049 and p=2/7/31 (36 combinations). No threshold
is fitted by either command. `--suite smoke --reps 1` selects N=1,000, p=4 only.
Default measurement/warmup budgets are 0.12/0.03 seconds with 20 samples and 1,000
bootstrap resamples per candidate/phase. These short defaults are exploratory;
increase them and independently replicate before using a small timing difference.

Each repetition is a fresh process containing both candidates; their order
alternates between repetitions. `CRITERION_HOME` is unique per repetition, so
raw samples and estimates cannot be overwritten by later runs or confused with
an old baseline. No compilation/profiling should overlap measurement. Child
stdout/stderr are direct files and survive failure. Explicit case-start and
value/score receipts distinguish unstarted cases from failed attempts. Complete
earlier cases remain admissible if a later case fails; a missing candidate,
changed source/binary, missing sample, invalid work count or mismatched score
withholds the entire pair. The runner exits unsuccessfully on child failure or
an incomplete campaign even if some cases remain useful.

Reports use Criterion's **mean nanoseconds per call**, not whichever estimator
looks faster. Within-process 95% intervals remain beside each raw estimate.
Median, minimum, maximum and standard deviation summarize independent process
repetitions; those ranges are not confidence intervals for a population median.
The report separately gives evaluations/second, Auto/scalar ratios and Auto's
point-estimate regret against the faster admitted candidate. Construction plus
K evaluations is also compared for K=1/10/1,000. Construction still excludes
original dataset, builder and workspace construction; neither phase includes process startup
or serialization. Timing-noise-induced regret is not proof of a planner defect.

Replay derives eligibility from sealed per-run artifacts and numerical receipts,
not cached favorable flags. It prints a **new** report without editing the old one:

```sh
nix develop .#stable --command python3 tools/native-benchmark/scripts/cpu_plan.py \
  --output /tmp/alea-cpu-plan-calibration --replay
```

To compare a changed implementation against an earlier sealed cohort, use
`--compare-before /path/to/before --output /path/to/after`. Settings and stable
CPU identity must match. Comparison independently rebuilds both reports and
requires identical f64 density/score bits for both modes at each supplied state;
a disagreement withholds that pair. These are sequential cohorts, not interleaved
old/new binaries. Candidate-order alternation alone does not remove temporal
drift between cohorts. Keep scalar controls, timing ranges and this limitation
when interpreting a before/after difference.

Campaign schema 2 includes explicit case-start receipts; earlier diagnostic
schema-1 smoke artifacts require their archived reader. The hashes detect changed
artifacts; they are not signatures or proof against coordinated malicious edits.
This is CPU-kernel evidence on supplied states, not an inference benchmark,
cross-CPU qualification, threading experiment or authorization to relax sums.

### Case study: SIMD dispatch with no useful vector packet

The first repeated campaign found a real execution-selection deficit. With fewer
than four observations, the packed observation loop has no full tile. With fewer
coefficients than a SIMD packet, coefficient-score accumulation also has no full
packet. Nevertheless, Auto entered the SIMD specialization. In particular, the
normal specialization initialized and merged empty lane accumulators before doing
the scalar tails. Selecting an ISA is not the same as having useful SIMD work.
These source facts identify unnecessary specialization/bookkeeping; wall time
alone does not isolate its cost from target-feature-specific compiler codegen.

The fix is deliberately structural: choose the existing scalar specialization
only when **both** execution axes lack a complete packet. Keep the compensated
formulas, data, priors/parameter interpretation, score propagation, finite checks
and errors. No separate probability implementation, fast-math, parameter clipping,
reduction reassociation or sampler change is involved. Widths greater than the
four-row tile are excluded because their existing normal fallback uses different
divided arithmetic. The planner explains this choice via `is_scalar_for_shape()`.

The benchmark is a native primitive evaluation, not a complete posterior or
sampling run. The synthetic design and parameter states are defined in its Rust
source, without RNG; normal's constrained sigma is 1.25. Both candidate modes
already prepare the same packed design. Thus this experiment does **not** compare
packing against row-major execution or measure a packing crossover.

#### Repeated results

One i5-8250U host, AVX2, CPU 2, stable Rust 1.95.0/LLVM 22.1.2, pinned flake and
Cargo lock, no extra Rust flags, one thread. Compilation/profile jobs did not
overlap timing. Calibration used 45 shapes and seven independent processes per
version; disjoint validation used 36 shapes and three processes per version.
All 423 before/after shape/repetition pairs were admitted: both modes' density and
every score coordinate matched **exact f64 bits**, including signed zero. This
qualifies the supplied states, not arbitrary conditioning or posterior inference.

Times below are medians of process-level Criterion **mean ns/evaluation**.
Ratios are medians of matched updated/original repetitions, with full observed
ratio ranges, not confidence intervals. The original and updated cohorts were
sequential; small shifts can reflect temporal or code-layout effects.

| Family | N | p | Original ns | Updated ns | Updated/original, median (range) |
|---|---:|---:|---:|---:|---:|
| Normal | 3 | 0 | 90.661 | 69.834 | 0.772 (0.731–0.798) |
| Bernoulli-logit | 3 | 0 | 84.263 | 78.601 | 0.933 (0.884–0.960) |
| Poisson-log | 3 | 0 | 62.010 | 54.433 | 0.876 (0.868–0.906) |
| Normal | 3 | 1 | 118.361 | 104.433 | 0.889 (0.868–0.900) |
| Normal, held-out shape | 1 | 2 | 86.214 | 71.466 | 0.829 (0.806–0.877) |

Large-shape controls do not show a general gain. For N=1,000, p=4, Auto's original
and updated median microseconds/evaluation were normal 10.637/10.969, logit
37.432/37.523 and Poisson 21.574/21.733. Their paired updated/original ratios were
1.021 (0.963–1.077), 1.002 (0.981–1.028) and 1.000 (0.963–1.030), respectively.
For scale, updated Auto throughput medians at those shapes were approximately
91,165, 26,650 and 46,013 evaluations/second, not retained iterations/second.
These controls are not evidence that every non-target shape is unchanged in cost.

#### Interleaved follow-up and negative findings

The sequential matrix also showed slower control point estimates: normal
N=10,000/p=0 by 5.7%, and logit N=1,000/p=64 by 3.5%. They were retained, not
removed as outliers. A post-hoc follow-up ran the **same frozen binaries** in
alternating old/new process order for seven pairs, with 30 samples, 0.25-second
measurement and 0.06-second warmup. It selected one target and three controls;
this was not a new full-grid qualification. All 28 shape/repetition pairs passed
the same exact-bit checks for both modes.

| Auto case | Updated/original, paired median (full range) |
|---|---:|
| Normal N=3, p=0 | 0.769 (0.751–0.885) |
| Normal N=1,000, p=4 | 0.998 (0.942–1.239) |
| Normal N=10,000, p=0 | 1.010 (1.002–1.116) |
| Logit N=1,000, p=64 | 1.012 (0.976–1.153) |

The target improvement persists under interleaving. Controls are less decisive:
their absolute latencies differed substantially from the earlier cohorts, and
scalar controls were also variable (the logit control's updated/original range
extended to 1.777). These observations neither establish a consistent 3.5% logit
regression nor prove cost equivalence. Small control regressions and host-state
effects remain unresolved; there is no blanket speedup claim.

The selector is still not optimal. Updated Auto/scalar medians include 1.158 for
normal N=1/p=7 and 1.032 for normal N=3/p=4. These shapes have a coefficient SIMD
packet, so they deliberately remain outside this rule. A broader cost model
needs further family/shape/CPU evidence, not an expanded cutoff chosen to hide
these cases. No observation threading, Faer selection or GPU backend is added.

#### Reproduction and evidence identities

The external local evidence root is
`../alea-inference-evidence/cpu-plan-2026-11-02.WRLYGu/`. It retains original
`calibration`/`validation` and updated `after-calibration`/`after-validation`
directories, deterministic source archives, exact executables, build records,
per-repetition Criterion samples/intervals, numeric receipts and replayable reports.
The `confirmation` directory additionally retains its exact driver `confirm.py`,
protocol, copied binaries/source archives, stdout/stderr and raw estimates.
It is local evidence, not data shipped in a source release or a signed artifact.

```text
Original benchmark SHA-256:
46d3b42a0ba281769d130791aa5b5221a8419e0f82a2765a69dd84babfdb75c7
Original source archive SHA-256:
462bebc31f56df82aeffca98ffa348f8cae96d3468c8c6e7db638d48d5847181
Updated benchmark SHA-256:
e9cce99a5d3c41223a28b7c7e703c161518215880ee9cf865e9ed34866427e1e
Updated source archive SHA-256:
45ea73ed62a78607829196174c78bba6283f3d0c5289fdbc6d561fc42d85a6d1
Calibration comparison SHA-256:
91641638447236fbdff062239d6a4013a6ed915ab05492f7d6460e0d385eaaf1
Validation comparison SHA-256:
46bc6be04245626100cd51217212e160e1440afc85990f49ea691622a7a81695
Interleaved summary SHA-256:
457d1129cfe0b360fbc53b8c22e9830d32637d3386a6bf101944dee515b8ffc6
```

Use `cpu_plan.py --replay` for each cohort and `--compare-before` for the two
version comparisons. Replay reads raw sealed artifacts, not cached eligibility.
For the follow-up, the saved driver contains the exact filter, order, commands,
numeric checks and estimator. Keep its original protocol separate from the main
matrix. This findings section postdates the measured source snapshots; it does
not change which sources/binaries produced the observations.

The regression suite independently covers finite differences, tails, retained
allocation counts, metadata, rebinding and failure recovery. Additional raw tests
compare old dispatch with the structural rule across tiny shapes, huge predictors,
normal scales from MIN_POSITIVE through MAX and invalid/nonfinite scales. Retain
those tests when extending the planner. The lesson is to model execution axes
and useful work explicitly, measure legal implementations separately, and preserve
negative controls rather than equating “SIMD selected” with “fast.” None of this
establishes Stan superiority, ESS/sec, new NUTS behavior or cross-CPU qualification.

## Source-owner inventory and disposition

The table covers production numerical owners, including those outside the
distribution crate. A listed owner is not a claim that every function in it was
individually profiled. Cold orchestration, storage, schemas and CLI plumbing are
not numerical kernels.

| Owner | Existing structure / disposition in this pass |
|---|---|
| Math `numeric.rs` | Stable links, compensated sums and SIMD finite checks already exist. Large scaled norms now combine finite/max scans and vectorize normalization/squares, retaining ordered reduction and direct division. |
| Math `metric.rs` | Dense actions already have blocked/backend implementations; inverse-diagonal energy already uses SIMD. Immutable mass-diagonal roots/determinant are now cached and momentum products use SIMD. An explicit velocity-SIMD trial showed no gain and was removed. |
| Math `metric/{structured,low_rank}.rs` | Block/sparse/low-rank representations avoid dense materialization. No new approximation or forced dense conversion. Reusable/mutable geometry must invalidate any additional cache transactionally. |
| Math `matrix.rs`, `least_squares.rs` | Cholesky explicitly promises sequential residual subtraction; least squares uses scaled sequential Faer SVD. No fast-math rewrite or normal-equation substitution. |
| Math `affine.rs`, `quasi_newton.rs`, `fisher.rs` | Reusable actions and caller scratch already exist; Fisher uses validated shared scales and sequential Faer products. No new measured gain claimed. |
| Math `exact/` | BigInt/rational diagnostic proofs, not approximate floating-point kernels. SIMD approximation is not an equivalent replacement. |
| Math `random.rs` | Existing larger-vector normal generation is optimized. RNG stream/consumption is part of replay; no change made. |
| Math `interpolation.rs` | Binary-search interpolation with overflow-aware weights and explicit endpoint behavior. No new workload evidence supports rewriting it. |
| Core model assembly, target/capability, transforms and transport | Checked staged evaluation, shared forward/reverse state, explicit Jacobians. No removal of validation or approximation of constrained coordinates. Earlier profiles locate the large-regression cost inside arithmetic, not graph dispatch. |
| Distributions `modeling/regression{,/kernels}.rs` | Normal/logit were already tiled. Poisson now shares prepared predictors and coefficient-score SIMD, with precomputed factorial normalizers and bounded vector exponential. |
| Distributions `modeling.rs`, scalar laws/links/ordinal/jets | Native Normal now requests the existing density and complete score kernel once each, rather than requesting the score kernel separately for event and parameters. Other generic factor fusion needs separate evidence/API design. |
| Distributions scalar special functions | The profiled NIG density already omits unused derivative/Bessel-k0 work in release code. No speculative N=0 rewrite was retained. Other families are not thereby proven optimal. |
| Distributions `multivariate{,/families}.rs` | Validated factors and some normalizers are already cached. Repeated whitening between independent density/score APIs is a possible fusion opportunity; no inverse-matrix shortcut or unmeasured optimization is shipped. |
| Distributions `gaussian.rs` | Specialized standard-normal target retains its existing public fused evaluation. Included as an unchanged fixed-work control, not rewritten to manufacture a favorable comparison. |
| Distributions `directional.rs` | Surface/volume measures and Bingham normalizers already have construction-time work. Replacing robust sphere admission with a differently rounded norm could move support boundaries; not changed. |
| Distributions `mixture.rs` | Immutable log weights are now cached; the first 32 weighted component values are retained on the stack across the two log-sum-exp passes. All components retain original evaluation order and zero-weight skipping. |
| Distributions `latent.rs` | SoA latent mapping exposes independent work; full-domain exponentials and rounded parameter-boundary admission require separate qualification. No new speed claim. |
| Distributions Wiener4/5/7, series, analytic reductions, CDF and sampling | Existing analytic marginalization, paired endpoints, bounded refinement and reusable buckets remain. Mixed-lane branch/series workloads need their own accuracy-qualified corpus; forcing one lane's work on all lanes is not automatically faster. |
| MCMC phase/vector and NUTS turning/weights | Runtime SIMD and conservative sign fast paths already exist. Ordered uncertain fallback, stable weights and proposal laws are not reassociated or replaced. |
| MCMC adaptation, Fisher, implicit/research samplers | Freeze points, inverse/mass conventions, bounded solves and unclosed exactness obligations remain unchanged. A faster implementation must not silently change the method. |
| Runtime optimization and diagnostics | Trial/scratch reuse and FFT autocovariance already exist. Legacy diagnostic reductions are not substituted for modern rank/bulk/tail diagnostics. |
| Autodiff/source generation | Enzyme-generated derivatives are a separate execution path. No finite-difference fallback or claim that native SIMD changes generated code. |
| FFI cubature and vendored C backend | Rust owns validated callback/error boundaries; C owns integration workspaces. No vendor algorithm, tolerance or callback protocol changed. |
| SMC | Traits are not an implemented particle kernel; there is no missing SIMD implementation to advertise. |

Unchanged owners remain visible deliberately. This pass is not evidence of
exhaustive numerical review or best possible performance across all Alea APIs.

## Implemented changes and numerical contracts

### Poisson batches

`poisson_log_regression` retains the model
`sum(y_i eta_i - exp(eta_i) - ln Gamma(y_i+1))`, including its constant terms.
Counts greater than 2^53 still fail admission. Factorial terms were already
precomputed; the new work addresses predictors, exponentials and score storage.

Immutable four-row predictor tiles reuse the existing compensated dot/product-
residual calculation. Coefficient scores keep row order, separate sum/correction
arrays and transactional output. Shared batch scratch is reused after errors.
The common exponential polynomial is enabled on [-32,32] for Poisson; the
logistic specialization retains its old [-32,0] guard and arithmetic. All tails
use scalar exp, including subnormal rates and overflow. No rate is clipped and
no failing observation is dropped.

The positive-domain Decimal80 fixture is independent of the implementation. It
uses exact binary64 inputs, includes neighboring range-reduction points and
boundary/overflow inputs, and complements the existing negative fixture. Tests
require <=4 ULP over those fixtures and a 40,000-point signed grid. This is an
observed tested accuracy bound, **not** a correctly-rounded/full-domain proof.
Poisson density/scores can change by final rounding bits; compare models and
gradients independently rather than requiring old sampler trajectory identity.

### Norms and diagonal metrics

`scaled_norm` keeps the max-scaled formula and direct division. Replacing division
with a cached reciprocal can overflow on tiny scales even when every ratio is
finite; that shortcut is rejected. Vectors of length >=32 use runtime SIMD for
the combined finite/max scan and normalized squares. The final sum preserves
the original coordinate order. Short vectors keep the previous implementation.

`DiagonalMetric` owns immutable masses. Constructor validation is unchanged;
the ordered log-determinant is derived once after validation; square roots are
cached only for dimensions >=16.
Momentum sampling uses the cached roots and SIMD from length 16. Velocity retains
the old direct division loop: explicit SIMD produced no improvement over compiler
vectorization. No reciprocal substitution is made, including for subnormal masses.
Tiny dimensions retain the original short sqrt/multiply loop: an always-cached
trial regressed at dimension four. Restoring that arithmetic alone was not enough:
the enlarged public wrapper also stopped inlining in the measured binary. The
ISA dispatcher is therefore outlined behind a non-inlined large-vector helper,
and the thin wrapper explicitly inlined after checking the generated binary.
Outlining alone still left a measurable call overhead in a trial; this is a
profile-justified local hint, not a blanket inlining policy.
The large-vector root cache costs 8d bytes
and one extra allocation at construction;
log-determinant storage costs one f64. Warmed actions allocate nothing.

We did not blindly cache every other metric getter. Inverse-diagonal and low-rank
metrics have transactional adaptation setters. Moving expensive calculations to
every setter can add work when a quantity is never requested. Any new cache there
needs an invalidation policy and an amortization measurement, not copied code.

### Mixtures

Weight logs are immutable construction-time data. A 32-term stack cache eliminates
the second component-density call for that prefix. Remaining terms in a larger
mixture retain their original second evaluation, avoiding per-call heap storage
or an unbounded stack. Two-pass max-then-sum arithmetic and component ordering
are unchanged, rather than switching to a differently rounded online reduction.

The cache is local: `FiniteMixture<D>` retains Send/Sync whenever D supplies them.
No RefCell, shared mutex, thread-local scratch or sticky state is introduced.
Zero-weight components are still never evaluated. Infinite maxima do not hide
later component errors; NaNs are still numerical errors. The first cohort's
event/parameter-score paths reused cached weight logs. The derivative follow-up
below additionally reuses weighted component values within each score request;
separate public density/event/parameter requests still do not share scratch.

### Native Normal factors

The native factory previously called density, event score and parameter score.
The last two both invoked the complete scalar score kernel. A crate-private
fused request reuses the **existing** density and score kernels and their error
order, returning the three local scores together. Scalar distribution methods,
generic factor APIs and generated primal mathematics remain unchanged. Tests
compare bits and typed errors against the previous sequence over extreme events,
locations and scales, including infinities and the smallest positive subnormal.

## Qualification and measurements

The fixed-work runner `tools/native-benchmark/src/bin/kernel_audit.rs` calls public
production kernels; it contains no surrogate density/derivative/sampler.
The original `scripts/kernel_compare.py` cohort declares 15 case/size/work
combinations before running seven alternating pairs on CPU 2. Construction is separate from evaluation;
raw failures and partial summaries survive. Norm, metric, mixture and unchanged
control checksums are exact; Poisson has an explicit checksum tolerance and
separate finite-difference/row-factor tests. Checksum agreement alone is not a
gradient correctness test.

Measurements use the pinned stable shell, Rust 1.95, release opt-level 3, fat LTO,
one codegen unit, generic target CPU plus runtime-selected Pulp SIMD, and no
fast-math. Thread counts are one. CPU is Intel i5-8250U. User-space `perf stat`
counters are available in this session, but `perf record -e cycles:u --call-graph
dwarf` fails with EACCES. Callgrind supplies source instruction attribution, not
sampled cycles; separate perf-stat counts corroborate the Poisson work change.
Compilation, unit tests and profiling do not overlap wall-time experiments.
The earlier [native report](native-model-performance.md) retains the separate
Stan end-to-end/ESS comparisons; isolated gains here are not new ESS/sec claims.

### Fixed-work results

Seven alternating pairs, 210 successful processes and 105 qualified pairs. Times
are median **evaluation** milliseconds for the listed total calls, not per-call
latency. Gain is median of paired before/after ratios; brackets give the observed
pair range, **not** a confidence interval. Ratios of marginal medians can differ.

| Public production workload | Calls | Before ms | After ms | Paired gain [range] |
|---|---:|---:|---:|---:|
| Poisson, N=1,000, p=8 | 10,000 | 964.448 | 364.643 | 2.636 [2.554–2.769] |
| Poisson, N=10,000, p=8 | 1,000 | 957.143 | 359.428 | 2.666 [2.602–2.792] |
| Scaled norm, d=4 | 2,000,000 | 94.465 | 93.120 | 1.015 [1.008–1.027] |
| Scaled norm, d=32 | 200,000 | 26.853 | 22.247 | 1.211 [1.165–1.249] |
| Scaled norm, d=4,096 | 10,000 | 140.942 | 87.640 | 1.611 [1.582–1.615] |
| Diagonal velocity, d=4 | 2,000,000 | 40.349 | 40.302 | 1.002 [0.961–1.022] |
| Diagonal velocity, d=4,096 | 20,000 | 69.593 | 69.511 | 1.003 [0.938–1.023] |
| Diagonal momentum, d=4 | 2,000,000 | 34.117 | 33.357 | 1.004 [0.983–1.096] |
| Diagonal momentum, d=4,096 | 20,000 | 104.820 | 37.141 | 2.847 [2.677–4.496] |
| Cached diagonal log-det, d=4,096 | 20,000 | 700.214 | 0.204 | 3,535 [3,299–6,965] |
| Mixture, 8 components | 100,000 | 53.770 | 24.845 | 2.167 [2.129–2.183] |
| Mixture, 64 components | 20,000 | 81.652 | 48.218 | 1.693 [1.599–1.754] |
| 128 native Normal factors | 10,000 | 171.967 | 136.612 | 1.239 [1.006–1.426] |
| Standard-normal target, d=4,096, control | 10,000 | 143.555 | 140.086 | 1.014 [1.000–1.029] |
| NIG scalar density, control | 1,000,000 | 196.115 | 198.287 | 0.991 [0.984–1.044] |

Poisson throughput at N=1,000 is approximately 10,369 to 27,424 joint
density/gradient calls/sec. Its N=10,000 throughput is 1,045 to 2,782 calls/sec.
These include the native model and the same gradient-checksum consumption in
both versions. Controls' small changes are not credited to a mathematical
optimization. The Normal-factor and large-momentum ranges show scheduling/
frequency variability; all raw values and standard deviations are retained.

The large log-det ratio is an O(d) getter replaced by an O(1) getter, not an
inference speedup. Its very short after interval makes its ratio particularly
noisy. Caching moves work to construction. In the momentum d=4,096 runner,
median construction is 192.04→262.37 microseconds; Poisson N=10,000 construction
is 3.714→4.213 milliseconds. Mixture K=8 construction is 4.83→18.90 microseconds.
These are single construction measurements per process (allocator/dispatch
initialization included), not isolated constructor microbenchmarks. The norm
runner also constructs a diagonal metric even though its timed loop uses only
the norm; its construction number must not be attributed to the norm function.
Caching is justified for repeated actions, not assumed free for one-shot use.

The earlier always-cached metric trial lost about 14% at d=4. Restoring the
short arithmetic and then outlining dispatch still left a call overhead; binary
inspection showed the wrapper out-of-line. The final local inline hint removes
that symbol and restores the small-case timing. The failed trials are retained,
not selectively substituted into the final table. Explicit velocity SIMD was
removed because the original LLVM-vectorized division performed equivalently.

### Evidence and replay

Raw artifacts live outside the checkout at
`../alea-inference-evidence/kernel-audit-20261030-XawzUE/`.
`paired-qualified-v6/{manifest.json,records.jsonl,validation.jsonl,summary.json}`
records the final cohort, hashes, all timings and checks. Other `paired-*`
directories are intermediate experiments, not the final source qualification.
Earlier exploratory harness versions are not represented as sealed final runs.

The final kernel binary is `kernel-audit-after-v6`, SHA-256
`c5b33e0e7677170d67531bf7d39ecc9010a06a6bd7d2f970df0d7eebf7991ab4`;
before is `kernel-audit-before-v3`, SHA-256
`156eab922d201716ae09683d64db862721b76785b3fb8a89291724b4e53cfc79`.
The before source is the preceding native case study's complete source capsule
plus the byte-identical `kernel_audit-v3.rs` runner added as a bin. Both engines
use the same runner and compiler settings. No dependency upgrade is credited
as a kernel improvement.

The final native binary is `native-audit-after-v6`, SHA-256
`758db01253bea2def0601c8536dfef85bcb941e171c5fdc50ed84ec4be34270c`.
`native-replay-v6/` contains three alternating pairs of three fixed-evaluation
cases and six sampling cases: 54 successful processes. All 27 pairs pass the
**default exact** work/checksum/trajectory checks against `native-blocked-v1`.
No rounding-change opt-in is used. The funnel's unchanged trajectories still
have the earlier statistical-quality limitation; replay is not convergence.
Gaussian/logistic sampling ratios lie around one (large Gaussian 1.038;
large logistic 0.982). This follow-up does not claim a new end-to-end speedup
on those models or supersede the earlier multi-chain ESS/Stan measurements.
`stan-checks-v6/checks.json` records 22 successful shared-coordinate CmdStan
density/gradient checks on the final binary, including extreme predictors and
small sigma. The independent model tests and original-seed failure regression
also pass.

`kernel-source-v6.tar.gz` contains the production source, tests, fixtures,
standalone harness, input data and locks (485 members); SHA-256
`0bf694ce98451cc983a4abba292f457d4275966631c9cce27192dac8af0145c4`.
All 12 final run-manifest source/package/lock digests were verified against
archive contents. Before reconstruction uses the prior complete capsule
`native-blocked-source-complete.tar.gz`, SHA-256
`50f604841e7cc6aea2802c319fcb4013676252531e9dd816c57b0790e150c0be`,
with the identical saved runner. Saved binaries permit direct replay; the local
dirty checkout's HEAD alone does not identify either executable.

### Work attribution

Matched Valgrind 3.27.1 Callgrind runs count whole-process instructions, including
startup/construction, at the same work and dimensions. They are not wall-time
benchmarks. Files are `*-before-v3-matched.callgrind` and `*-after-v6.callgrind`.

| Workload | Calls | Before instructions | After instructions |
|---|---:|---:|---:|
| Poisson N=1,000, p=8 | 1,000 | 672,139,742 | 221,834,446 |
| Norm d=4,096 | 1,000 | 75,976,645 | 18,130,779 |
| Momentum d=4,096 | 1,000 | 17,586,726 | 6,663,856 |
| Mixture K=8 | 10,000 | 36,255,657 | 18,364,241 |
| Native Normal factors=128 | 1,000 | 116,842,576 | 101,336,869 |

The old Poisson profile attributes 40.81% of instructions to the inlined
`numeric.rs` work in the batch callback; scalar exponential entry/kernel adds
9.09% exclusive. Predictor/score arithmetic, compensation and scalar FMA
machinery are substantial work, not evidence that one opaque factor's dynamic
dispatch dominates. The new callback instead spends its hot work in the shared
packet predictor/score/exp kernels. Normal's duplicate-score request is a
separate source-confirmed problem, not an inferred explanation for every law.
Matched `perf stat` runs at 10,000 Poisson calls count 6,700,036,751→2,199,640,934
instructions and 827,883,916→155,432,614 branches. Files are
`poisson-before-v3-matched.perf-stat.csv` and `poisson-after-v6.perf-stat.csv`.
These provide a second attribution control; their profiled durations are not
inserted into timing ratios. Cycle sampling remains permission-denied.

## Tests and limits

- Norm tests compare scalar/SIMD bits for lengths 0..257, shifted slices,
  subnormals, enormous scales, zeros and nonfinite entries.
- Diagonal tests cover lengths 0..65, exact scalar products/divisions/log sums,
  invalid shapes before writes, cache cloning, Send/Sync and zero warmed allocations.
- Mixture tests cover cache boundaries 31/32/33/65, evaluation counts, zero weights,
  exact reduction results, allocation freedom, Send/Sync and error recovery.
- Poisson tests compare full tiles/tails against single-row factors, offsets,
  dimensions 0/1/3/8/9, central finite differences, extreme predictors, counts at
  2^53, overflow errors, underflowed rates, staged gradients and scratch reuse.
- Native Normal fusion tests require exact density/score bits and identical error
  variants relative to the old three-call sequence.

The remaining opportunity list is explicit: generic law value/score fusion,
full-domain vector transcendentals, multivariate solve fusion, wider mixed-lane
Wiener workloads and adaptation-cache tradeoffs. These need representative
profiles and their existing independent accuracy gates. We do not remove
compensation, substitute normal equations, weaken precision checks, clip inputs,
or assume a new paper's result applies to Alea's f64 implementation.

## Follow-up: reuse mixture terms in derivative requests

The first mixture optimization removed duplicate density work inside log_prob.
It did not eliminate the subsequent density requests used to form derivative
responsibilities. Profiling the production event/parameter-score paths and a
native ModelBuilder factor identified this remaining shared work. The fix is
private to `FiniteMixture`: `log_prob_terms` retains the existing max-then-sum
evaluation and exposes its bounded local prefix to the derivative caller.

Each nonzero component in the first 32 slots now has one density evaluation
per event-score or parameter-score request instead of two. Components outside
that prefix retain three evaluations (the two density passes and responsibility
pass). Component score calls and row/reduction order are unchanged. The cache
is still 32 f64s on the stack: no new heap buffer, lock, TLS, shared mutable
state, public API or distribution/derivative formula is introduced.

This deliberately is not a fused distribution-triple API. A native continuous
mixture factor still invokes density, event score and parameter score separately
and constructs its distribution during evaluation. For a mixture entirely inside
the cache, those three requests reduce component density calls from five to
three, not to one. Remaining construction, normalization and score work must be
included in the native-path measurement.

`tests/mixture_cache.rs` now has seven tests. New coverage includes exact call
counts across 1/2/8/31/32/33/65 components, zero warmed allocations, exact score
bits against the previous public-law sequence, underflowed responsibilities,
typed component failures/NaNs, zero weights and clean recovery. Native model
tests finite-difference all free coordinates with simplex weights and positive
scales, including transform Jacobians and staged failure gradients. Unsized
multivariate events exercise existing component scratch, exact parameter scores,
shape/NaN failures and recovery. The call-count test fails against the previous
implementation (two calls instead of one), then passes after the change.

The benchmark adds public event-score/parameter-score cases at K=2/8/32/64 and
native continuous mixture factors at K=8/64. It contains no surrogate mixture
density, derivative or sampler. `kernel_compare.py --suite base` retains the
original 15-case default; `--suite mixture-scores` selects ten additions and
`--suite all` declares the combined 25 cases. Failed runs remain excluded from
timing ratios, and every new case requires exact checksums.

### Follow-up measurements and provenance

Seven alternating pairs on CPU 2, with the same stable/release flags described
above: all 350 processes succeed and all 175 pairs pass exact checksum checks.
Times below are median evaluation milliseconds for the listed number of calls;
gain is the median paired ratio, with the observed pair range in brackets (not
a confidence interval). No build, test or profiler overlaps these wall timings.

| Production path | Components | Calls | Before ms | After ms | Gain [range] |
|---|---:|---:|---:|---:|---:|
| Event score | 2 | 100,000 | 18.694 | 15.633 | 1.185 [1.097–1.401] |
| Event score | 8 | 100,000 | 61.355 | 44.787 | 1.336 [1.229–1.608] |
| Event score | 32 | 20,000 | 48.974 | 32.640 | 1.459 [1.290–1.575] |
| Event score | 64 | 20,000 | 108.114 | 95.429 | 1.122 [0.787–1.205] |
| Parameter score | 2 | 100,000 | 24.637 | 20.773 | 1.154 [1.074–1.362] |
| Parameter score | 8 | 100,000 | 79.760 | 62.293 | 1.293 [1.234–1.318] |
| Parameter score | 32 | 20,000 | 61.917 | 47.508 | 1.276 [1.188–1.384] |
| Parameter score | 64 | 20,000 | 135.925 | 121.723 | 1.099 [1.082–1.153] |
| Native continuous factor | 8 | 10,000 | 27.561 | 23.000 | 1.195 [1.086–1.266] |
| Native continuous factor | 64 | 10,000 | 198.081 | 180.239 | 1.099 [1.077–1.110] |

Native factor throughput is approximately 362,834→434,786 joint value/gradient
calls/sec at K=8 and 50,484→55,482 at K=64. Each native benchmark evaluates one
continuous factor with a free event and component means, fixed uniform weights
and fixed scales. It constructs the law inside the public continuous factory
on every evaluation and includes the same gradient-checksum consumption in both
versions. It is not a full mixture-data fit or a proper-posterior/ESS benchmark.
The correctness tests additionally cover free simplex weights and positive scales.

Construction medians for the native graph are 27.18→27.23 microseconds at K=8
and 69.16→70.16 at K=64. There is no new persistent cache or constructor work in
the production change. All fifteen earlier controls are retained in the raw
summary: excluding the extremely short log-det getter, their median paired
ratios range 0.965–1.041, with ranges crossing one. No speedup is credited to
these unchanged controls. K=64 event score includes one slower pair; it is not
removed. Larger mixtures benefit less because only the first 32 slots are cached.

Matched Callgrind 3.27.1 whole-process instructions at K=8 and 10,000 calls:
event score 38,347,011→30,632,575; parameter score 49,837,122→41,722,595;
native factor 172,149,070→156,503,011. The baseline parameter-score profile has
substantial scalar Normal density/log work; the call-count regression identifies
the exact duplicate requests. Counts include startup/construction and are not
substituted for measured wall time. These results do not establish new NUTS
efficiency or cross-engine superiority.

Evidence: `../alea-inference-evidence/mixture-scores-20261030-hstjDw/`, containing
`paired/{manifest.json,records.jsonl,validation.jsonl,summary.json}` and the six
`*-{before,after}.callgrind` profiles. Replay the 25-case cohort with
`kernel_compare.py --suite all --before .../kernel-before --after .../kernel-after
--output /new/path --reps 7 --cpu 2`. The production runner is byte-identical in
the two source capsules; it only calls public Alea laws and model-building APIs.

SHA-256 identities:

- `kernel-before`: `a10a25e5da2ba671e45940534d9fd52aa3c83204ef0be6d55668d89c59500bc1`.
- `kernel-after`: `491c65de4b543d7435983bc42da8e0547c8cbc2234aff4cf398198c60340297e`.
- `source-before.tar.gz`: `0e5775b49d16a5a93aabcdf2bb65a2e41e7519ad876d2649b721eabf31498dc2`.
- `source-after.tar.gz`: `44d2b85e7881a7984a02e4a81e153605d206d80f0b2d991873ff1c5914bd045e`.

Both capsules cover 485 distinct source/test/fixture/data/lock paths (501 before
and 517 after archive entries, including identical repeated vendor entries).
The after source
matches all twelve source/package/lock digests in the run manifest. The full
distribution suite, seven mixture regressions, three native benchmark correctness
tests, eight Python protocol tests, Clippy, formatting and architecture checks
pass. The final release build reproduces the saved after binary byte-for-byte.
Earlier native/Stan cohorts and their sealed source are not overwritten.

## Review checklist for future kernels

1. Write the mathematical operation, parameterization, support, Jacobian and
   error/publication contract before changing the loop. Identify which sums may
   reorder and which scalar dependencies are essential.
2. Preserve a runnable baseline with complete source, locks, data and binary
   digests. Use the same runner and flags; dirty HEAD is not a source identity.
3. Separate fixed joint density/gradient throughput, constructor amortization,
   sampler work counts and multi-chain ESS/sec. Never infer one from another.
4. Count calls and inspect optimized instructions first. A high-level helper can
   recompute an entire derivative tuple; a generic implementation can already
   eliminate unused work. Source suspicion is not proof of emitted work.
5. Hoist only immutable shared work, or specify transactional cache invalidation.
   Report extra packed storage and constructor costs. Avoid scratch designs that
   silently lose Send/Sync or introduce a lock into every evaluation.
6. Test vector packets, scalar fallback, every tail, tiny sizes, subnormals,
   overflow, NaN and cancellation. Use independent references and gradients;
   checksum agreement alone does not establish mathematical correctness.
7. Benchmark small-size dispatch and inlining as well as large throughput.
   Inspect binary code when restoring source arithmetic does not restore speed.
   Reject a SIMD rewrite if LLVM already handles the simple loop as well.
8. Measure alternating pairs without competing builds/profilers, retain failures
   and rejected trials, and report spread. A bounded vector transcendental needs
   an explicit tested accuracy statement and scalar tails, not an inherited
   correctness claim from a paper. Do not trade away compensation or error
   propagation for an attractive timing.
