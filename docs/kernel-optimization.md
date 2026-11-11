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

### Generic scalar score requests and shared preparation

The provided `ParameterScore::log_prob_event_parameter_score` hook extends the
same principle to other continuous factors. Its default delegates density, event
score and parameter score in that order, so downstream laws require no new
implementation. Built-in scalar families specialize it to compute their complete
score vector once. Jet-based families retain the original density-only primal;
combining Jet<0> and Jet<5> arithmetic is a separate numerical change, not assumed
safe here. A Cauchy regression test counts two derivative calls through the old
separate requests and one through the combined request, and checks failure and
recovery behavior. This is a work-count result, not a measured wall-time gain.

Native scalar Normal factors additionally declare a pure preparation of their
scale's logarithm. Compilation groups identical source expressions (or bitwise
identical constants) by primitive type and preparation function. Each model's
workspace caches that scalar by input bits. Original factor evaluation order,
local score checks, compensated factor sums and reverse edges remain unchanged;
there is no cross-factor reduction reassociation or speculative callback
execution. Rebinding rebuilds the plan. This shares normalization, but does not
automatically turn arbitrary scalar plates into SIMD regression batches.

The ignored `regression_stage_profile` fixture in the existing kernel test module
isolates the actual packed predictor, exponential, log1p and compensated score
operations. Set `ALEA_PROFILE_STAGE` to `predictors`, `exp`, `exp_signed`, `log1p`
or `scores`; `ALEA_PROFILE_SCALAR=1` selects its scalar control. Row, column and
repetition-work sizes use `ALEA_PROFILE_ROWS`, `ALEA_PROFILE_COLUMNS` and
`ALEA_PROFILE_CALLS`. Run it in release mode on an allocated remote benchmark
node only. It compiles the stage choice outside the inner work loop and emits
dimensions, seconds and a checksum. These isolated controls are not additive
percentages of fused evaluation time: validate conclusions with the existing
whole-kernel runner and CPU profiler before changing production mathematics.

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

## Native `cargo bench` stage suite

The workspace is in `crates/`: run `cargo bench` there, or use
`cargo bench --manifest-path crates/Cargo.toml --workspace --benches` from the
checkout root. These are native Criterion benchmarks, not ports of likelihoods
or samplers. Library test harnesses are disabled for `cargo bench`, so Criterion
filters and baseline arguments do not accidentally go to libtest.
The shared cold Criterion configuration namespaces output by package and target,
so identical operation names in different executables cannot overwrite results.
This changes the result-directory layout; retain old baselines rather than
moving them into the new namespace and pretending they share an identity.

The source review found missing runtime, scalar-score, source-emission and FFI
boundaries, combined setup/evaluation scopes, and a handwritten logistic HMC
fixture. The HMC fixture now uses the public native Normal priors and
Bernoulli-logit regression. Its old timing identity is **not comparable** across
this fixture change; start a new baseline.

| Owning crate / target | Distinct native measurement scopes |
|---|---|
| math / `storage`, `primitives`, `quasi_newton` | Allocation/drop versus reusable storage; compensated reduction, finite scans, norms, transcendentals, metric velocity/energy, cached RNG fill, exact dot sign, rank-zero affine actions, interpolation, Cholesky, allocating/reused L-BFGS action |
| core / `transforms` | Chart constrain, chart pullback, complete transformed target |
| core / `model_pipeline` | Registration/drop, compile, target conversion, rebind, conditional rows, constrained/unconstrained fused evaluation; native/checked density-gradient boundaries and transactional point update |
| core / `tensor_tape` | Scalar graph, direct tensor VJP, assembled tensor factor, explicit compensated/Faer/BLAS GEMM policies |
| distributions / `scalar_scores` | Ten scalar families, Normal mixtures, multivariate Normal, sphere; separate density/event/parameter calls versus combined scalar scores, constructor versus retained vector law |
| distributions / `regression_execution`, `model_semantics`, `wiener7` | Native Normal/logit/Poisson preparation versus evaluation, scalar/Auto policy, semantic/opaque registration, Wiener quadrature |
| autodiff / `derivatives` | Rust source emission and finite-difference diagnostic; neither is compiler execution time |
| mcmc / `hmc`, `rwmh`, `warmup` | HMC/RWMH transitions; NUTS initialize/drop, fixed-state transition, evolving transitions; adaptation/update/metric-fit scopes; shared fixed-state harness for GHMC, DRGHMC, MAMS, explicitly approximate MCLMC, WALNUTS, Atlas, ChEES and analytic/Hessian-free force-gradient HMC |
| runtime / `workflow` | Session configuration, initialize+one draw, warmup+one draw, retained collection/discard, owned-record clone, bounded enqueue/dequeue, complete L-BFGS fit, modern diagnostics |
| runtime / `workflow` with `analysis` | Memory-only CSV header setup, one-row encode/hash/buffering, final flush/manifest after 128 prepared rows |
| Enzyme conformance / `native_compiled` | Same native/generated models, full value/gradient, generated HVP and Hessian/third derivative; pure compiler-AD polynomial control, native-backed source rules, retained/checkpointed scan controls and explicit route receipts |

Coverage is **not complete**. The existing runner's `cargo-inventory` action
enumerates every workspace source module and its reviewed benchmark associations:

```sh
python3 tools/native-benchmark/scripts/three_engine.py cargo-inventory \
  --output target/cargo-bench-inventory.json
```

An association is a workflow link, not proof that all functions are independently
timed. Unmapped implementations remain explicit `gap` entries. Declaration/test
modules and the re-export facade/trait-only SMC crate do not get invented work.
Remaining gaps include manifold/split/batched sampler paths, general prediction
and transport workflows, some specialized linear algebra and source adapters.
Private logistic/Poisson stage profiling remains in the existing ignored kernel
fixture; these internals are not exposed as new public APIs just for Criterion.
CPU provider features do not imply every operation uses that provider.

### Running and comparing

All actual timings belong on an allocated ELTE HPC **compute** node. Use the
existing Slurm submission helper; request one CPU for single-thread comparisons,
or an explicit matching CPU/thread budget for threaded comparisons. The generic
[feature script](../tools/bench-features.sh) builds every selected cell first,
then runs cells sequentially with distinct Criterion directories and preserved
build/run failures. It refuses local/login timings.
From the repository root, `tools/bench-features.sh --help` lists options without
requiring Cargo or starting a build. The same executable entry point is included
in source-only distributions; no separate benchmark runner installation is needed.
The measurement child is pinned to the first `N` allowed CPUs for the requested
thread budget; compilation retains the complete allocation. Both masks are
recorded, without changing governor or machine-wide thread settings.
Example inside an allocation:

```sh
tools/bench-features.sh --crate alea-distributions --bench scalar_scores \
  --output /scratch/alea/scalar-comparison --feature-set '' --feature-set faer \
  -- --sample-size 30 --warm-up-time 0.5 --measurement-time 2
```

For the whole workspace omit `--crate`/`--bench`. Select the pinned nightly
externally before requesting SIMD, branch hints or compiler AD. For actual
compiled derivatives, use the **same script** with
`--manifest tools/enzyme-conformance/Cargo.toml --bench native_compiled` in the
pinned autodiff environment, including its autodiff/bitcode flags and fat LTO.
This is a separate Cargo workspace and is not silently qualified by a facade
feature flag. `--threads N` sets library budgets; it does not make sequential
Faer kernels threaded.

For before/after comparisons use Criterion's `--save-baseline`/`--baseline`
directly in the same stable remote environment, with the same feature/compiler
flags and unchanged fixture. Feature-cell runs are sequential exploratory
comparisons, **not** alternating cross-version pairs. Rotate/reverse cell order
and repeat before ranking close results. Use the existing paired native/Stan
runner for full posterior comparisons and diagnostic-qualified ESS/sec.

For a feature run, the existing runner also consumes Criterion's raw estimates
and samples directly, including median confidence intervals, sample/iteration
counts, feature labels, per-cell failure status and all same-case ratios:

```sh
python3 tools/native-benchmark/scripts/three_engine.py cargo-report \
  --output /scratch/alea/scalar-comparison
# Writes /scratch/alea/scalar-comparison/summary.json; no additional measurements.
```

These are ratios of medians from sequential cells, not paired confidence
intervals. Throughput units must match. No overall "fastest Alea" is inferred
by pooling different operations or experimental/exact samplers. Sampler
initialization receipts expose their actual transitions; fixed transition
timings exclude the prepared chain/RNG, while initialization includes seed setup.
Approximate MCLMC remains explicitly biased; timing it is not qualification of
its stationary distribution or comparative ESS/sec.

Local/CI correctness-only smoke execution is explicitly untimed:

```sh
cargo test --manifest-path crates/Cargo.toml --workspace --benches -- --test
# Build only (does not execute a benchmark):
cargo bench --manifest-path crates/Cargo.toml --workspace --benches --no-run
```

### Baseline routes: inspect execution, not feature names

The existing Enzyme `native_compiled` benchmark emits versioned
`ALEA_BASELINE_ROUTE_V1` planning/preflight receipts. It inventories `models::all()`
without another fixture registry. `cargo-report` reads these from each cell's
`run.log` into `route_inventory`, including requested/resolved route, derivative
order, first-order admission route, arithmetic/provider labels, checkpoint flag,
status and reason. Older logs remain `not-recorded`; Cargo features never supply
missing execution evidence. Failed build/run cells remain in the report even
when no fixture receipt was reached.

The same command also reads `ALEA_FIXED_WORK_V1` receipts into `fixed_work` and
`fixed_work_unavailable`. The `native_compiled` preflight uses one shared bank
of five coordinate vectors for native and generated evaluation. The first call
is recorded separately from four steady calls. Each phase counts attempted and
completed public fused target calls. Compiler AD, admission helpers and provider
internals can perform additional work that these counters do not expose.

The harness checks that the generated fixture contains the exact native primal
body before assigning a common model/data identity. The receipts identify the
query, coordinate metadata, model and bound literals, state bank, algorithm,
arithmetic and provider. Their FNV-1a
identities support within-run consistency checks. The existing source-sealed
runner manifest supplies cryptographic provenance. The report rejects changed
identities within a fixture and across matching cells before producing ratios.
Unsupported and unexecuted routes remain explicit, and failed cells retain
completed first phases and partial steady-call counts.

`input_gradient_bytes` counts the logical f64 input-bank and gradient storage.
The direct tensor control measures `workspace_owned_bytes` during workspace
construction with the existing allocation counter. Immutable bindings and
process/provider allocations are outside that count. Targets without this
measurement report an unknown value and reason. These fields describe neither
peak RSS nor a compiler tape bound.

`tensor_tape` and `model_pipeline` use the same cold bank and phase helper while
retaining their independent exact-value controls and Criterion inputs. The
receipts normally describe untimed preflight work, separate from Criterion
iteration counts. In a Slurm compute allocation, ALEA_FIXED_WORK_TIMING enables
first/steady phase wall times through ALEA_PHASE_ELAPSED_V1 records. These times
include target calls, poisoning/admission and output copies, and exclude receipt
logging. The report retains these optional durations as phase_elapsed_nanoseconds.
Cargo build timing reports and process RSS records describe their own scopes.

The ELTE before receipt used one exclusive Xeon Gold 6240 node and seven
alternating source pairs. Each tree ran 96 native/source cases and 32 pinned
Enzyme cases, yielding 1,792 Criterion case observations. The receipt retains
952 first/steady phase records, 2,380 counted public preflight target calls,
and 5,600 unavailable route-phase rows. Model/data, coordinate, state-bank,
algorithm, arithmetic and provider identities matched for the measured cases.
The interrupted first capture is retained separately because its Enzyme logs
were incomplete. One process-resource record also contains NUL padding, leaving
27 of 28 process RSS observations available. Cold builds have one observation
per variant and different generated registry sizes, so their costs are recorded
without a normalized cold-build comparison. Route/admission differences and
compiler/provider storage limits remain explicit in the receipt. Read existing runner output with
`python3 tools/native-benchmark/scripts/three_engine.py cargo-report --output RUN_DIRECTORY`.

| Label | Current baseline meaning |
| --- | --- |
| N | Stable native analytic/tensor execution, not absence of differentiation |
| C | Scalar/direct Rust-source compiler reverse without a trusted first-order override; the polynomial is the current control |
| L | Tensor-generated primal plus compiler reverse, without the explicit first-order override |
| H | Existing hybrid/trusted first-order composition, including explicit tensor replay when flagged |

The current generator selects its first-order route model-wide; it cannot force
a pure C/L baseline for a fixture that selects H. Such requests are recorded as
`unsupported`, not silently renamed. Planning alone is `not-run`. A `supported`
receipt means the benchmark actually executed that untimed preflight; independent
answers and broader invalid-state qualification are still separate conformance
tests, not inferred from metadata. Duplicate/conflicting or malformed receipts
are errors rather than an opportunity to merge incompatible runs.

For example, vector Normal uses H first order and C for curvature, with H
first-order admission. A checkpointed tensor uses H first order and L for
curvature, again with H admission. A retained tensor normally uses L throughout.
The checkpoint flag describes the generated first-order path; it does **not**
mean checkpointed HVPs/third derivatives or a compiler-memory bound. Primal-only
source exports use the original C/L primal, not the first-order pullback; their
admission route is `none`. Value-only comparisons remain `not-run` because this
benchmark does not isolate their cost. The generic native model has no curvature
trait, even when a separate specialized target elsewhere provides one.

`fixture-default` arithmetic preserves each fixture's existing scalar f64 and
compensated kernels; it is not an equal-accuracy certificate. These fixtures use
Rust kernels, not explicitly selected Faer/BLAS calls. Provider feature availability
alone does not change that receipt. Requested/resolved policy fields describe
this bounded default inventory, not a completed backend/provider powerset.

Use the existing feature script in the pinned autodiff environment with `--smoke`
and `--manifest tools/enzyme-conformance/Cargo.toml --bench native_compiled` for
untimed preflight, then the same `cargo-report` action above. Metadata inspection
is cold work outside Criterion iteration closures. The benchmark **still requires
`PotentialHessianDerivative` even for its first-order comparisons**; inventorying
that limitation does not fix it or establish a first-order-only compiler API.
Matrix/mixed fixtures not selected by this benchmark retain `not-run` receipts,
even if separately exercised by existing conformance tests. No throughput or
speedup follows from an untimed route inventory.

Criterion validates fixtures before measurement. `iter_batched` excludes declared
setup and output destruction for consuming phases; `iter_batched_ref` also keeps
the NUTS chain/RNG destruction outside fixed-transition timing. IDs containing
`drop` deliberately include destruction. Reused evaluations exclude model/data
construction, buffers, compilation and serialization. NUTS fixed-state receipts
report real gradient calls, leapfrog steps and depth in a separate instrumented
pass using the existing counter helper; the timed target has no counter/clock.
Evolving transitions are not a fixed number of gradients. The scalar law and
gradient fixtures check numerical agreement before comparing implementations.

These costs must not be subtracted or summed to manufacture a sampler-overhead
estimate: separate microbenchmarks have different cache/residency conditions.
`warmup80_sample1` includes the required retained tail. CSV measurements use
memory buffers, not disk durability. Sampling phases still include their named
initialization/monitoring boundaries. Process startup, compilation, disk I/O,
four-chain diagnostics and ESS/sec remain separate end-to-end measurements in
the existing native/Stan suite. Allocation correctness tests and perf/Callgrind
profiles are separate passes, never overlapping timed runs on the same host.

### Why construction and evaluation must remain separate

The completed Atlasz `normal-batching` control is a concrete negative result.
Seven alternating same-binary scalar/vector pairs on `cn18-31` (KVM, one pinned
CPU, nightly 2026-06-23) passed the existing exact work-receipt gate. These are
independent **free-event Normal factors with fixed location/scale**, not Gaussian
regression or a shared-mean inference benchmark. Evaluation includes the same
value/gradient checksum consumption in both variants.

| Free events | Scalar evaluation µs/call | Batch evaluation µs/call | Median paired scalar/batch [observed range] |
|---:|---:|---:|---:|
| 3 | 0.1440 | 0.1883 | 0.762 [0.750–0.836] |
| 4 | 0.1789 | 0.2257 | 0.792 [0.788–0.796] |
| 5 | 0.2211 | 0.2823 | 0.783 [0.765–0.785] |
| 128 | 5.180 | 5.257 | 0.957 [0.919–1.039] |
| 4096 | 192.481 | 192.239 | 1.002 [0.874–1.028] |

Construction medians for 4096 events were 45.373 ms scalar versus 21.231 ms
batch; for 128, 0.250 ms versus 0.113 ms. Thus reduced graph construction does
**not** establish faster retained evaluation. The small-batch regression is not
hidden or used to justify an unmeasured dispatcher threshold. These results do
not rank posterior sampling or the separate fixed-observation/vector-location
path. No inference or ESS claim follows.

Raw records, numerical admission, binary/source receipts and summary remain in
`/users/usumusu/alea-score-increment-20261103/normal-batching` on Atlasz, job
2776558, independently of the newer Criterion study. The ranges are observed
seven-pair ranges, not confidence intervals or a fresh measurement of later
changes. This is the confound the new phase scopes are designed to expose.

## Logistic and funnel: profile the representation actually executed

The November 4 investigation compared the existing native targets with the
CmdStan 2.40 / Stan Math 5.4 full-posterior evaluator. Logistic uses
`bernoulli_logit_glm_lpmf` as its primary reference and the ordinary vectorized
form as a separate control. Funnel uses one scalar Normal prior and a vectorized
Normal conditional. Both include the same priors and complete gradients.

With explicit maintainer authorization, local diagnostic profiles used `perf`
7.2.10, user-space cycles at 997 Hz, DWARF stacks, a pinned CPU and release/native
CPU binaries on an i5-8250U. These are **not Atlasz timings**, confidence intervals
or ESS measurements. No lost samples were reported. The representative local
instruction counts were:

| Evaluation | Stable Alea | Pinned nightly + autodiff flag | Stan |
|---|---:|---:|---:|
| Logistic 1k | 170,481 | 295,050 | 73,687 |
| Logistic 10k | 1,666,708 | 2,909,479 | 723,755 |
| Funnel, eight latents | 4,907 | 5,118 | 2,810 |

Approximately 98% of logistic cycle samples were inside the fused evaluator;
graph-dispatch work was not the primary bottleneck. Annotated assembly identified
retained fixed-array iterator loops in polynomial evaluation and four-row score
updates, plus temporary arrays and spills. The nightly evaluator reserved 1,368
stack bytes versus stable's 776. This establishes a code-generation difference,
not that Enzyme itself caused it: compiler/stdlib/flag controls must be separated.
Compensated predictors and scores also perform more arithmetic than Stan's
ordinary Eigen reductions; the implementations have different rounding policies.

The candidate explicitly expands the fixed seven polynomial and four score-row
steps without changing arithmetic order, compensation, tail handling or failure
fallbacks. An initial local counter pass reduced nightly logistic instructions
from approximately 295k to 254k (1k) and 2.909m to 2.497m (10k). These are binary
diagnostics, **not an admitted wall-time speedup**. Atlasz before/after comparisons
and stable/other-family controls remain required before accepting a speed claim.

After the final fixed-map cleanup, the existing runner's new `profile` command
completed six matched local diagnostic variants (100,000 logistic1k evaluations
each; all stat/record/report exits zero). Instructions/evaluation were:

| Saved configuration | Instructions/evaluation |
|---|---:|
| Before stable | 170,480 |
| After stable | 168,268 |
| Before nightly, autodiff flag enabled | 295,049 |
| After nightly, autodiff flag enabled | 208,275 |
| After nightly, autodiff flag absent | 168,158 |
| Stan fused GLM | 73,907 |

The same-flag nightly candidate executes about 29.4% fewer instructions; stable
saves about 1.3%. On the same candidate source, enabling the compiler AD flag
adds about 23.9% instructions even though the target uses native analytic
derivatives. This isolates a flag-dependent code-generation cost, not the cost
of executing an Enzyme-generated gradient or a particular compiler pass. These
are single diagnostic counter passes, not seven-run wall-time estimates.

The suite previously injected `-Zautodiff=Enable` into every nightly build. It
now enables that flag per Enzyme feature cell only when the user has not supplied
an explicit AD flag. Explicit `RUSTFLAGS`/encoded flag controls are preserved,
and actual per-cell flags remain recorded. This prevents native cells from being
silently charged for an unnecessary compiler mode. It does not rewrite or
retroactively improve any historical measurement.

Funnel has a different bottleneck: about 51–57% of cycle samples landed in model
evaluation and 20–21% in the Normal preparation callback frame, which includes
its arithmetic. These percentages must not be described as pure dispatch cost.
Shared log-scale preparation already existed. The candidate retains that cache,
checks a shared dynamic input once per forward evaluation, uses fixed-size scalar
argument packets and groups reverse operands without changing their order or
alias multiplicity. Normal event/location scores share one sign-symmetric
division. The initial funnel instruction reduction was below 1%; no substantial
funnel speedup is established, and small-batch promotion remains disabled.

### Reuse the existing tools

`three_engine.py profile --matrix-plan PLAN --output OUTPUT --cases logistic
--evaluations 100000` profiles saved binaries, retaining counters, sample data,
reports and failures. It requires a compute allocation; `--local-profile` is an
explicit diagnostic-only exception. `--perf-command 'sudo -n /path/to/perf'`
supports installations requiring privileges for DWARF sampling without changing
system-wide perf policy. Instrumented execution times are never speed rankings.

Historical evaluation loops vary only coordinate zero. Optional whitespace state
banks now vary all coordinates, including nonzero funnel latents. Use
`ALEA_BENCH_STATE_DIR=DIR` with the existing data generator, then
`matrix-plan --state-bank-dir DIR`. Matrix banks contain ten finite complete
states so the existing ten-evaluation work receipts remain valid. Loading is
construction work; copying each state is timed in both engines. Banks are included
in existing input identity/replay checks; historical defaults are unchanged.

The Stan builder now uses a study-local `CMDSTAN_MAIN` driver. Make otherwise
reuses `main.o` when compiler flags change, potentially mixing Eigen alignment
ABIs with native-CPU model objects. The historical Atlasz sampling aborts are
retained as failures; a consistently rebuilt reference must pass before producing
new comparative ESS/sec results. This build isolation does not retrospectively
validate failed runs or prove their precise allocator failure mechanism.

## Compensated versus conventional f64 regression arithmetic

Native `NormalRegression`, `BernoulliLogitRegression`, and
`PoissonLogRegression` support an explicit per-descriptor
`with_arithmetic(RegressionArithmetic::Standard)` option. The default remains
`Compensated`. This is a runtime construction choice, not a Cargo feature that
could unexpectedly change another dependency's arithmetic through feature
unification. `with_execution` independently selects native automatic SIMD or
scalar execution; rebinding preserves both policies.

Standard specializes the existing fused row kernels: predictor products and
sums, lane likelihood/scale reductions, and coefficient-score accumulation use
ordinary f64 arithmetic. It does not enable compiler fast-math, introduce tail
cutoffs, change exp/log approximations, drop validation, clip parameters, or
change sampler arithmetic. It allocates no evaluation buffers. Existing Gaussian
square-root compression and common-predictor Poisson sufficient statistics remain
unchanged, including their compensated preparation/evaluation. Thus a compressed
Gaussian control is not a measurement of uncompensated observation scans.

This policy is not an accuracy-equivalent default replacement. Two executable
counterexamples in `tests/regression_execution.rs` use exact binary64 inputs:

- A predictor `1e16 + 1 - 1e16` becomes 0 instead of 1. For five positive logistic
  observations, log likelihood changes from approximately -1.5663 to -3.4657.
- At beta=0, predictors and likelihoods can agree while the coefficient score
  `1e16 + 1 - 1e16` becomes 0 instead of 1. Checking density alone misses this.

Regular-shape comparisons cover zero columns, scalar/SIMD tails, N=1 through
10,000, finite differences, deterministic repeated evaluation, zero allocations,
nonfinite/support/shape errors, rebinding and transactional recovery. Agreement
on these fixtures is empirical, not a uniform error bound or posterior-efficiency
qualification. Pointwise reporting retains the reference row arithmetic, as it
already does for compressed likelihoods. Source AD export rejects Standard
descriptors explicitly; it must not silently substitute a compensated primal or
claim qualification for generated higher derivatives.

The untimed stable correctness smoke completed all 72 policy/family/shape
combinations (ten probe states each). On that local test build, the largest
absolute differences from the compensated reference were approximately
1.74e-10 in log density and 5.19e-11 in a score coordinate. Compressed Gaussian
controls were identical because their algorithm was unchanged. These are
fixture-specific numerical observations, **not local timing results**, uniform
accuracy bounds, or evidence of unchanged NUTS efficiency.

Reuse the existing benchmark target on an allocated **remote compute node**:

```sh
ALEA_REGRESSION_ARITHMETIC=1 cargo bench --manifest-path crates/Cargo.toml \
  -p alea-distributions --bench regression_execution
```

The `regression_arithmetic` groups compare the same executable/kernel formulas
under both policies, at N=3/1,000/10,000 and p=1/8/32/65. The p=65 Gaussian case
is outside compression's supported width and measures the actual row kernel
without disabling a production optimization. Construction and validation
are outside timing. Ten all-coordinate probe states report maximum absolute
density/score differences against the compensated reference before measurement.
`ALEA_REGRESSION_PLAN_REVERSE=1` reverses policy order for alternating repetitions.
Criterion retains per-policy estimates; retain every repetition rather than
pooling samples across policies. Use the existing CPU-plan campaign separately
for construction/scalar-dispatch controls. No wall-time speedup is established
until the remote campaign completes; no change to defaults is justified by the
cancellation counterexamples or by a fast result on ordinary fixtures alone.

Tensor `MatrixKernel::Faer` and `MatrixKernel::Blas` already explicitly permit
uncompensated library reductions. The new regression option is a native SIMD
arithmetic ablation, **not** automatic regression lowering into either provider.
Comparing library GEMV/layout/threading costs is a distinct experiment; enabling
those Cargo features alone still does not change native regression providers.
