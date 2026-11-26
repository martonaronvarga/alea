# Native-model performance: a correctness and speed case study

This investigation concerns the public `ModelBuilder` → native regression factors
→ transformed target → `SessionBuilder` NUTS path, compared with CmdStan 2.40.0.
It is not a comparison of languages, nor a claim that a fixed number of draws
has equal statistical value. The benchmark remains a client of Alea's public
distributions and sampler: it contains no replacement likelihood, gradient, or NUTS.

The short diagnosis: there were two independent problems. A numerical proposal
failure was incorrectly promoted to a fatal backend failure. Separately, our
accurate batch arithmetic was implemented in a form that executed serial scalar
work where Stan's array/matrix kernels hoisted invariants and used SIMD. The
graph and session abstractions were not the dominant cost in these workloads.

## 1. What was actually compared

The old benchmark's hand-written Gaussian likelihoods fixed the residual scale.
The native normal-mean and Gaussian regression models estimate a positive sigma
with a half-normal prior. Those are different posteriors and dimensions; their
timing ratio cannot establish an abstraction penalty. The logistic models are
comparable mathematically, but the old implementation uses ordinary reductions
and rounded tail scores rather than the native compensated reductions and stable
scores. Even there, equal iterations do not establish equal computational work.

The matched models use beta ~ Normal(0,2), where 2 is standard deviation, not
variance. Gaussian observations have mean X beta and standard deviation sigma;
sigma ~ HalfNormal(0,1). The logistic likelihood uses BernoulliLogit(X beta).
X is supplied row-major with explicit intercept columns; no implicit intercept,
standardization, prior change, or rank-based model modification is introduced.

For Gaussian models the free coordinates are (beta, u), sigma=exp(u). Their
free-coordinate log posterior contains the likelihood, coefficient priors,
`-sigma²/2`, and **one** `+u` Jacobian. The half-normal normalization differs
from a lower-bounded normal declaration only by a parameter-independent constant.
Consequently, compare density differences from a shared reference point and
compare all gradient components directly. Never omit the Jacobian to match an
incorrect control. Normal-mean has one coefficient, regression eight.

The original archive has no historical Cargo.lock. Its seven supplied timings
cannot be reproduced as an exact compiler/source cohort. New measurements use
saved executables, original data bytes, pinned lockfiles, and source manifests;
they are not relabeled as the historical numbers. See the detailed earlier
[investigation report](../tools/native-benchmark/REPORT.md) for the immutable
repair-only and coefficient-SIMD cohorts.

## 2. The warmup failure was not a defective exponential chart

`TransformError::PrecisionLoss` is defined in
`crates/alea-core/src/error.rs`. All production cases are in
`Transform::constrain_impl` (`transform/mod.rs`) and `Transform::factor`
(`transform/matrix.rs`):

| Transformation | Precision-loss condition |
|---|---|
| Lower/upper bound | bound ± exp(u) rounds back to the bound |
| Interval | mapped sigmoid rounds at/beyond an endpoint |
| Ordered/positive ordered | an increment no longer increases the preceding value |
| Simplex | component/remaining stick <= 0, or stick fraction equals 1 |
| Unit vector | 1 + squared norm nonfinite, or last coordinate rounds to 1 |
| Correlation factor | (1−tanh(u))(1+tanh(u)) <= 0 |
| Positive Cholesky diagonal | constructed diagonal <= 0 |

The original Gaussian N=10,000 input and seed [17;32] fail on target evaluation
three: integration step one in warmup transition two, **not** a long NUTS tree.
The free sigma coordinate is −806.6677923703763; binary64 exp returns zero.
The reported index 0 is local to the sigma transform block; its global coordinate
is 8. The Jacobian u is finite and correctly included. Permitting sigma=0 would
break the declared parameter domain; clipping it would change the algorithm.

Zero initialization gives sigma=1. Initial step size .2 and disabled reasonable-
step search expose a very large initial force. The first proposal is rejected;
dual averaging's initial reference ln(10 epsilon) can increase the next trial
step before subsequent updates reduce it. This explains the excursion, not a
reason to suppress transform checks or change the benchmark's model.

`PhaseWorkspace::advance` recognized outer nonfinite value/gradient/position
errors as divergences, but a typed representability error wrapped in
`EvaluationError::Model` was considered fatal. Thus a recoverable bad proposal
aborted warmup. The fix adds default-fatal `is_numerical_error` to fused target
and constrained-model protocols and forwards it through transformed, compiled,
composed, summed, frozen-transport and profiling adapters. Only explicitly
recognized numerical failures opt in. Boxed native factor errors opt in only
for the concrete core `DistributionError::Numerical` variant.

Integration maps those errors to `TargetNumerical` divergence and uses existing
rejection/adaptation logic. Direct target evaluation and invalid initialization
retain typed errors. Shape, domain, observation, sampling-budget, workspace and
opaque foreign backend errors remain fatal. Nothing inspects error strings.
The live position/density/gradient cache is never committed after a failed step.

The original row-prefix sweep reproduced different manifestations of the same
classification defect: N=1,000 and 2,500 completed; N=5,000, 7,500 and 9,000
overflowed native density/score arithmetic; N=9,250, 9,500, 9,750 and 10,000
underflowed the positive chart. The repaired version completed all nine. These
sampled boundaries are not a universal monotone threshold across seeds or data.

## 3. Profile the actual implementation, including the reference

Release builds and fixed-work evaluations separate cost per target call from
the number of target calls. Hardware and software `perf` events are denied on
this host, so attribution uses Valgrind Callgrind plus emitted assembly. Those
instruction counts are not CPU cycles or a wall-time speedup.

The original scalar native logistic path used 768,450,249 instructions for 1,000
fixed evaluations, versus 364,000,468 in the old hand-written diagnostic control.
Approximately 98% of native instructions were batch arithmetic. Neither graph
dispatch nor session storage explained the gap. The predictor's accurate product
residual used f64::mul_add per coefficient; generic compilation emitted cached
ISA-dispatch helper calls inside the observation loop. These were real hardware
FMA dispatches, not evidence of software-emulated FMA. The coefficient SIMD
repair dispatched once per factor evaluation, retained the compensation, and
reduced the count to 516,417,128. Eight million out-of-line FMA helper calls
disappeared. Seven paired repeats preserved exact work and trajectory hashes.

That repair was useful but insufficient: it left the Gaussian scale divisions,
serial per-row predictors, scalar scale/density reductions, and scalar logit
exponentials. Source-level claims such as “analytic gradients” or “one batch
factor” did not imply an efficient inner loop.

### What CmdStan actually runs

Read the installed **2.40.0** Stan Math source, generated model C++, compiler
metadata and executed assembly, not an assumed idealized Stan implementation.
The benchmark CSV has empty stanc optimization flags. Its generated C++ calls
ordinary `normal_lpdf(y, multiply(X,beta), sigma)` and
`bernoulli_logit_lpmf(y, multiply(X,beta))`, not specialized GLM entry points.

Relevant Stan Math files beneath `stan/lib/stan_math/stan/math` are:

- `prim/prob/normal_lpdf.hpp`: computes inverse sigma once, vector scaled
  residuals and partials, and -N log(sigma) once for a shared scalar scale.
- `rev/fun/multiply.hpp`: compressed reverse-mode matrix multiplication with
  Eigen X beta and X-transpose times residual kernels, not a tape node per row.
- `prim/prob/bernoulli_logit_lpmf.hpp` and Eigen's packet math: array exponential
  and vector arithmetic. SSE2 packet exponential is present in the actual binary.
  `log1p` remains scalar in the observed path in both systems.

Stan's ±20 logit tail branches do not explain these fixed profiles: fixed
predictors have |eta|<.135. Conservative bounds over all 4,000 retained Stan
draws are <5.087 for N=1,000 and <2.518 for N=10,000. This says nothing about all
warmup states; it is enough to reject tail clipping as the ordinary-case cause.

Three fresh Stan `log_prob` profiles include JSON parsing and gradient CSV
writing, while native fixed profiles reduce outputs to a checksum. They are
useful for attribution, **not** a matched cross-engine throughput experiment.

## 4. Targeted kernel repairs

Production changes are local to
`alea-distributions/src/modeling/regression.rs` and its private
`regression/kernels.rs`; Poisson kernels and global `CompensatedSum` are excluded.
The numerical-classification repair is independent in core/MCMC adapters.

1. **Shared scale:** compute reciprocal sigma once for ordinary representable
   scales, multiply residuals and partials, and add the shared normalizer with
   one compensated product. Accumulate -z²/2, not sum(z²), because the latter
   can overflow when the former remains finite. When squared scale or squared
   reciprocal is nonnormal, retain the original division-based reference path.
   No input is clipped or newly rejected by that guard.
2. **Broadcast recognition:** immutable data with one all-unit column and zero
   offsets use a scalar predictor and direct compensated mean-score reduction.
   This is an automatic general factor optimization, not a benchmark-only
   likelihood or a changed model declaration.
3. **Four-observation tiles:** prepare a private [tile][column][lane] design copy
   at factor construction. SIMD predicts independent observations together;
   coefficient order, compensated additions and product FMA residuals remain.
   Coefficient scores keep row order and their separate heads/corrections.
   Gaussian density/scale reductions combine independent lane heads **and
   corrections separately**, avoiding loss of a tiny residual when large lane
   heads cancel. Nonfinite fast-path output retries the original ordered/divided
   Gaussian calculation rather than inventing a finite answer.
4. **Bounded vector exponential:** ordinary logit exp arguments in [-32,0] use
   split-ln(2) range reduction, exact powers of two and a degree-13 Taylor
   polynomial evaluated with vector FMA. No saturation or probability cutoff.
   Arguments outside that range—including subnormal tails and nonfinite values—
   use the existing scalar exp. `ln1p` and signed tail-stable score formulas stay.

The added packed design costs approximately another 8Np bytes plus tile padding
per factor. The unit-intercept specialization needs no packed copy. Construction
is a one-time O(Np) cost, not a free operation; report it separately. Target
evaluation reuses O(p) score scratch and small fixed stack tiles, with no heap
allocation per observation or per warmed evaluation.

This follow-up deliberately changes some rounded density/score results and can
change adaptive trajectories. It preserves the posterior, not cross-version
bitwise identity. Same-version repeats remain deterministic. The comparison
harness defaults to exact replay checks; an explicit
`--allow-trajectory-change` records which hashes/work counts differ and still
checks equal fixed evaluation counts and near-equal finite checksums. It never
turns a failed inference run into a timing sample.

## 5. Accuracy and failure tests are separate from timing

The private predictor tests compare scalar and runtime-selected SIMD outputs
bitwise against the original compensated/product-residual implementation over
dimensions 0..17, row counts 1..17, cancellation, tails and subnormal inputs.
Lane-merge tests retain a unit residual after cancelling 1e16 heads.

The exponential fixture is generated with Python Decimal at precision 80 from
the exact input binary64 values, not from the implementation's formula. It
covers range endpoints, multiples of ln(2), neighboring representable values,
and underflow tails. Tests require at most four ULP against those independent
values and a 40,000-point scalar-exp grid, and exact scalar fallback outside
the bounded interval. This is a measured accuracy contract on the tested domain,
not a proof of correct rounding for every f64 argument. The polynomial truncation
bound is distinct from total floating-point error.

Model-level tests compare batches of 4, 5, 8 and 17 observations, dimensions
0, 1, 3, 8 and 9, and single-row factor assembly. They test broadcast and general
designs, central finite differences, large logits, underflow tails, zero warmed
allocations, deterministic repeat evaluation, failure output staging, and scratch
recovery. Extreme Gaussian scales include the smallest subnormal through f64::MAX
with representable residuals, plus a finite quadratic for which sum(z²) overflows.

The benchmark's separate correctness suite checks posterior equations, priors,
Jacobian, finite differences, extreme free coordinates, and the original-seed
early-warmup failure. A production CmdStan `log_prob jacobian=1` comparison checks
22 shared-coordinate density differences and gradients, including N=10,000,
small sigma, and logits ±1,000. Core/MCMC tests separately verify numerical
proposal recovery **and** propagation of genuine backend/initialization errors.

## 6. Measurement protocol and reproducibility

The new cohort compares the immutable coefficient-SIMD binary
`native-simd-score-v1` with `native-blocked-v1`, not the historically unmatched
hand-written Gaussian. Fixed-work tests use 10,000 target calls; inference uses
1,000 warmup, 1,000 retained draws, step .2, target .8, diagonal adaptation and
maximum depth 10. Seven alternating native pairs run sequentially on CPU 2.
Separate cross-engine runs use three repetitions of four sequential chains.
Compilation, unit tests and profiling never overlap wall-time measurements.

Host: Intel i5-8250U x86_64. Rust 1.95 release uses fat LTO, one codegen unit,
generic target CPU and no fast-math; runtime-selected SIMD is still available.
Stan is CmdStan 2.40.0, GCC 16.2, -O3, no explicit target CPU or fast-math.
All relevant thread counts are one. Rust and installed Stan use different pinned
Nixpkgs trees/libc builds (glibc 2.42 versus 2.44); compiler/library effects are
not controlled away. `tools/native-benchmark/PINNED.json`, the Nix/Cargo locks,
run manifests and input hashes retain the concrete versions.

Native target timing includes parameter transforms, priors, likelihood, pullback
and Jacobian. Stan's profile region encloses its posterior model block but
excludes transforms/Jacobian outside that region. Those per-call figures are
scoped diagnostics, not perfectly symmetric microbenchmarks. End-to-end times
include process startup and parsing. Alea collects in memory; Stan writes retained
CSV and launches a process per sequential chain. The benchmark offers native CSV
serialization measurement, but it cannot make the default scopes identical.

Raw outputs, incomplete outcomes, errors, source hashes, adaptation, evaluation
counts, leapfrog counts, depths, divergences and multi-chain diagnostics are
retained. ESS/sec uses wall time and the minimum coordinate bulk ESS; retain
tail ESS and R-hat as separate validity checks. A finished process is not proof
of convergence. Funnel results that fail divergence/R-hat criteria must not be
used to advertise statistical throughput.

### New results

Seven alternating pairs, all 126 outcomes completed. Fixed-work evaluation is
10,000 full transformed target calls, excluding construction. Parentheses give
the **range of paired speedup ratios**, not a confidence interval:

| Fixed-work workload | Before seconds | After seconds | Median paired gain (range) |
|---|---:|---:|---:|
| Normal mean, N=1,000 | .281402 | .027406 | 10.27× (10.00–10.57) |
| Gaussian regression, N=1,000 | .622456 | .240578 | 2.64× (2.52–2.67) |
| Logistic regression, N=1,000 | .928894 | .651410 | 1.43× (1.35–1.48) |

These correspond to approximately 35,536 → 364,887, 16,065 → 41,567 and
10,765 → 15,351 target evaluations/sec respectively. Fixed checksums agree
within the predeclared bound; per-point model checks remain separate.

Single-chain inference includes startup, construction, warmup, sampling and
instrumentation. Median times and median **paired** ratios are distinct
statistics; dividing the two median times need not give the median paired ratio.

| Inference workload | Before seconds | After seconds | Median paired gain (range) |
|---|---:|---:|---:|
| Normal mean, N=1,000 | .227003 | .029887 | 7.58× (6.58–7.93) |
| Gaussian regression, N=1,000 | .848035 | .326936 | 2.66× (2.35–2.72) |
| Logistic regression, N=1,000 | 1.200251 | .965839 | 1.42× (1.22–1.48) |
| Gaussian regression, N=10,000 | 8.010964 | 2.998451 | 2.67× (2.59–2.81) |
| Logistic regression, N=10,000 | 11.243565 | 7.904998 | 1.42× (1.39–1.46) |
| Funnel, unchanged arithmetic control | .098136 | .095358 | 1.03× (.96–1.06) |

The funnel control is consistent with ordinary timing variation, not a claimed
optimization. One chain cannot supply the multi-chain diagnostic gate.
`work-comparisons.json` explicitly retains changed work and trajectory identity.
Do not attribute the entire inference ratio to the kernel ratio or combine these
times with a Stan cohort measured at a different host frequency.

Fresh fixed-logistic Callgrind work falls from 516,417,128 to 319,470,830
instructions per 1,000 evaluations (−38.1%). Approximately 95.45% remains in
the batch callback; scalar log1p is about 25.26% inclusive. There are no millions
of out-of-line FMA helpers in this new path. This confirms both the eliminated
work and a remaining source of cost; it does not imply a 38.1% wall-time gain.

The cross-engine cohort completed all 36 planned outcomes: three repetitions,
four sequential chains, six workloads, two engines. All 30 Gaussian/logistic
outcomes meet the stated gate (all R-hat <=1.01, bulk/tail ESS >=100, no retained
divergences). Median process wall seconds, observed timing ranges and minimum
bulk ESS/sec are:

| Workload | Alea seconds (range) | Stan seconds (range) | Alea / Stan min bulk ESS/sec |
|---|---:|---:|---:|
| Normal mean | .1086 (.1058–.1093) | .2665 (.2541–.2761) | 32,686 / 13,250 |
| Gaussian N=1,000 | 1.2873 (1.2869–1.2938) | 1.5788 (1.4919–1.6080) | 5,123 / 3,498 |
| Logistic N=1,000 | 3.3008 (3.2601–3.4106) | 3.4340 (3.4225–3.4419) | 1,774 / 1,561 |
| Gaussian N=10,000 | 11.2977 (11.2008–11.3012) | 13.7718 (13.5920–13.8811) | 515 / 408 |
| Logistic N=10,000 | 31.4135 (31.1896–32.0226) | 32.7463 (32.4139–33.1278) | 176 / 181 |

Thus Alea is faster in wall time on these five cases and ahead in minimum bulk
ESS/sec on four. Large logistic is **2.6% behind** on that statistical-throughput
metric despite being faster in wall time. This is not grounds to claim superior
statistical efficiency there. It also demonstrates why iteration counts alone
would have been misleading.

Per-call measurements and actual work explain the outcome. Times have the unequal
profile scopes described above; calls include warmup, whereas steps/depths below
are retained-only. Values are Alea / Stan:

| Workload | Calls | Scoped microseconds/call | Leapfrog steps | Mean depth (both max 3) |
|---|---:|---:|---:|---:|
| Normal mean | 29,576 / 38,295 | 2.933 / 3.651 | 14,384 / 15,750 | 1.879 / 1.972 |
| Gaussian N=1,000 | 52,304 / 58,275 | 23.531 / 22.730 | 26,512 / 25,764 | 2.840 / 2.776 |
| Logistic N=1,000 | 49,436 / 56,911 | 65.590 / 56.302 | 25,700 / 25,584 | 2.764 / 2.766 |
| Gaussian N=10,000 | 49,863 / 56,194 | 224.832 / 235.809 | 25,616 / 25,012 | 2.789 / 2.752 |
| Logistic N=10,000 | 47,286 / 54,794 | 662.248 / 588.112 | 24,196 / 24,072 | 2.699 / 2.660 |

Logistic target calls remain more expensive than Stan's scoped calls. Fewer
warmup/overall calls help Alea's end-to-end outcome. The claim is not that every
native gradient is now cheaper. All these retained runs have zero divergences;
large Gaussian still records four numerical warmup evaluations, one per chain.
Construction totals for four Alea chains are approximately .178, .726, .638,
3.365 and 3.545 milliseconds in workload order, far below inference time.

Final adapted Alea steps, by chain, are retained as follows (rounded here; full
precision is in JSONL):

| Workload | Final steps |
|---|---|
| Normal mean | .769, .922, .925, .921 |
| Gaussian N=1,000 | .676, .646, .584, .629 |
| Logistic N=1,000 | .723, .763, .687, .603 |
| Gaussian N=10,000 | .525, .497, .572, .571 |
| Logistic N=10,000 | .756, .667, .772, .756 |

All six funnel outcomes complete but fail the diagnostic gate: Alea has 32
retained divergences and maximum R-hat 1.080, Stan 18 and 1.052. Their apparent
ESS/sec is not an eligible favorable-performance result. The final native
retained maximum depth is 9 versus Stan 10 on this control.

The published kernel gains combine the shared-scale, broadcast, layout and
exponential changes. Profiling supports their mechanisms, but these tables do
not assign an independent causal wall-time percentage to each optimization.
Earlier immutable cohorts remain in the linked investigation report.

A separate, predeclared three-pair native comparison repeats **four chains** on
both saved binaries to measure before/after statistical throughput. All 30
outcomes pass the diagnostic gate. This is additional evidence, not a replacement
for the seven single-chain pairs above:

| Workload | Before / after wall seconds | Before / after min bulk ESS/sec | Paired wall gain (range) |
|---|---:|---:|---:|
| Normal mean | .7906 / .1041 | 4,700 / 34,104 | 7.65× (7.55–7.73) |
| Gaussian N=1,000 | 3.2605 / 1.2465 | 2,016 / 5,291 | 2.61× (2.56–2.62) |
| Logistic N=1,000 | 4.5493 / 3.2275 | 1,268 / 1,814 | 1.41× (1.36–1.43) |
| Gaussian N=10,000 | 30.4216 / 11.1030 | 222 / 524 | 2.74× (2.70–2.77) |
| Logistic N=10,000 | 44.1553 / 30.4998 | 138 / 181 | 1.45× (1.45–1.46) |

Same-build trajectory hashes repeat exactly across timing repetitions. Current
hashes and non-timing ESS/R-hat match the cross-engine cohort exactly. Before/after
hashes and work need not match: the altered reductions change adaptation and
draws, and the reported ESS/sec accounts for that difference. Timing ranges are
descriptive with only three pairs, not confidence bounds on general performance.

The follow-up CSV control (three repetitions, four chains, the three N=1,000
models) completes all 18 cross-engine outcomes. Median native serialization
costs are 3.712 ms for mean, 11.905 ms for Gaussian regression, and 11.277 ms
for logistic regression—approximately 3.27%, .93% and .34% of their CSV-inclusive
process medians. Thus serialization is not zero and becomes noticeable in the
accelerated mean model, but it cannot explain a multi-fold kernel gap. Stan
still writes richer output; this control does not claim identical I/O workloads.
Do not interpret differences between this cohort's wall times and the default
cohort as a causal estimate of CSV cost: use the explicitly measured serialization
phase instead.

### Evidence and replay

The external evidence root is `/home/usu/documents/dev/alea-inference-evidence`.
It is intentionally outside the source checkout; it is not an independent backup.
`native-blocked-evidence.sha256` seals the executables, source capsule, raw results,
plans/manifests, summaries, work comparisons, Stan checks and profiles; all 25
listed digests passed `sha256sum --check` at handoff.
The following immutable cohorts retain raw JSONL, plans and summaries:

- `native-blocked-paired`: seven alternating pairs, fixed-call and single-chain
  inference, 126 completed outcomes.
- `native-four-chain-blocked`: three repetitions, four chains, native/Stan,
  36 completed outcomes; six explicitly fail the funnel diagnostic gate.
- `native-blocked-four-chain-paired`: three alternating pairs, four chains,
  five regression/location cases, 30 diagnostically qualified outcomes.
- `native-blocked-serialization`: three repetitions, four chains, three small
  cases with native CSV enabled, 18 qualified outcomes.
- `native-blocked-sizes`: original row prefixes/seed, nine sizes, 18 outcomes;
  all nine final runs complete and all seven expected fatal-baseline failures
  remain recorded. This final sweep repeats the repaired boundary result using
  the shipped optimized kernels, not only the earlier repair-only build.
- `native-stan-correctness-blocked-final`: 22 production-engine density/gradient
  probes; `*-fixed-blocked-v1.callgrind`: mean 16,121,518, Gaussian 139,461,578,
  logistic 319,470,830 instructions for 1,000 evaluations. Cross-engine profile
  serialization scopes still differ; these totals are not head-to-head timings.

The measured after executable is `native-blocked-v1`, SHA-256
`aecbf0c0e42406f04c14e5bc8f17475c3b26b06bdfb54d0524eace99143eb919`.
A subsequent release rebuild from the final working sources reproduces it byte for
byte. The before executable is `native-simd-score-v1`, SHA-256
`91e609d4569df080403d4ec35d4d3449866adad89f521400a45b2444b89708be`.
Its previously sealed complete source capsule is retained, not overwritten.

The after overlay `native-blocked-source-complete.tar.gz` has SHA-256
`50f604841e7cc6aea2802c319fcb4013676252531e9dd816c57b0790e150c0be`.
It contains source/locks, fixture/generator, native benchmark instructions/pins,
and ignored example aliases as file content. All 362 run-manifest source hashes
and standalone Cargo.lock were verified from the archive. The one deliberate
script-version distinction is explicit: the cross-engine run started before the
paired runner gained multi-chain/subset options. Its exact `compare.py` bytes
are preserved in archive member `native-blocked-cross-engine-compare.py`, whose
hash matches that manifest; the ordinary path contains the final paired runner.
Production/model/Stan runner sources are unchanged between these cohorts.
Rebuilding still requires the pinned baseline checkout and package dependencies.
The baseline Git commit is `3984ba6c3773a6542d48019d61a1f07594889526`;
the revised build is that commit plus the digest-bound overlay, not a floating
`main` reference or an invented new commit.

Use `scripts/compare.py` for alternating saved binaries, adding
`--allow-trajectory-change` for this arithmetic change; add `--chains 4 --modes
sampling --cases mean linear logistic linear_large logistic_large --reps 3`
for the multi-chain native cohort. Use `scripts/bench.py --chains 4 --reps 3
--cpu 2 --timed` with the pinned CmdStan path for the cross-engine cohort, and
`--alea-csv --cases mean linear logistic` for the serialization control. Scripts
are relative to `tools/native-benchmark`; choose fresh result directories.

Verification executed successfully: affected core/distributions/MCMC default test
suites; nine distribution regression integration tests; six private score/kernel
tests; three benchmark model tests; five Python harness tests; the 22 CmdStan
probes; distribution Clippy with `-D warnings`; formatting and architecture
checks. The broad test run retains a pre-existing unused helper warning in
core's downstream conformance inclusion; it is not a new kernel warning.

## 7. How we avoid repeating this

The mistake was not “using native distributions.” It was treating mathematically
fused batch evaluation as though it were already a hardware-efficient kernel,
then comparing unmatched targets and iterations without a work budget. Accuracy
features were not wrong; their scalar implementation and missing performance
qualification made their cost surprising.

For each new batch factor, the implementation/review checklist is:

1. Write the constrained density, free-coordinate map and Jacobian explicitly.
   Test independent equations/gradients before a sampler can hide mistakes.
2. Include the native public construction path in benchmarks from the start.
   A hand-written target is an attribution control, never the advertised result.
3. Freeze models, data bytes, initial states, seeds, compiler flags, source and
   binaries. Different dimensions/priors are separate experiments.
4. Measure fixed-call target throughput **and** end-to-end inference. Always
   retain call counts, steps, depths, divergences and adaptation; separate work
   changes from faster work. Add multi-chain ESS/sec and validity criteria.
5. Inspect release assembly/profile evidence. Hoist shared invariants; dispatch
   once per kernel, not once per arithmetic product. Confirm packet execution
   instead of assuming the compiler vectorizes a compensated scalar loop.
6. Choose data layout deliberately. Account for preprocessing, extra copies,
   scratch, tails and small-N fallback. Batch APIs alone do not guarantee locality.
7. Preserve accuracy requirements through vectorization. Independent lane work
   is safer than casually reassociating a reduction. Test compensation heads and
   residuals, cancellation, signed zeros, subnormals and representability limits.
8. Test proposed-state failures through every adapter. Recover only explicitly
   numerical proposals; backend/shape/domain/init errors must remain observable.
   No broad “catch error and return -infinity” policy.
9. Use paired, alternating saved executables on a quiet pinned CPU. Treat
   profiling/instruction counts as attribution, not wall-time measurements.
   Preserve unsuccessful cases and previous evidence instead of overwriting it.
10. Make performance gates hardware-calibrated and statistical, while keeping
    correctness, work/replay checks and zero-allocation assertions deterministic.
    Document deliberate rounding/trajectory changes rather than disabling checks.

Speed is a property of the shipped native path under a specified numerical and
statistical contract. It is not inherited from Rust, analytic gradients, a batch
API, or an earlier benchmark with a simpler model.

## 8. Limits and follow-up boundaries

- These are small-dimensional dense regression models on one CPU architecture,
  one input corpus and one seeded multi-chain configuration. Repeating seeded
  runs measures timing spread, not uncertainty across independent posterior
  experiments. No universal “faster than Stan” conclusion follows.
- Compensated predictors/scores cost more arithmetic than ordinary Eigen sums.
  Scalar log1p remains substantial. Removing accuracy requirements or copying
  Stan's tail cutoffs is not an accepted next optimization.
- The packed data copy increases persistent memory; very wide designs and
  construction-dominated workloads need separate qualification. No BLAS,
  threading or automatic algorithm-selection claim is made.
- Runtime-selected SIMD and changed reductions are not cross-version or
  cross-architecture bitwise replay promises. Numerical checkpoints/model
  workflows must retain build/source identities.
- Vector exponential tests establish a bounded observed error on the specified
  fixtures/grid, not a full formal f64 error enclosure. Extreme tails remain
  scalar; other platforms need their own numerical and performance gates.
- Recoverable numerical proposal errors still occur during warmup. They are
  counted and rejected, not concealed. Reasonable-step search may improve the
  initial exploration but is not substituted for the original failure policy.
- Engine startup, CSV/storage, diagnostic timing, profile regions, compiler and
  libc differ. End-to-end statistical throughput is useful despite those
  differences; it must not be described as an isolated language/kernel advantage.
- The centered funnel remains a separate diagnostic stress control; completing
  it does not cure divergences or make a low-quality fit eligible for ESS claims.

## 9. Stabilization: test the adapters and the evidence pipeline

A follow-up implementation audit found three gaps that did not require another
kernel optimization. First, `EnzymeModel` and `EnzymeHigherOrderTarget` returned
`AutodiffError::NonFinite` but inherited the default-fatal numerical-error hook.
The native graph adapters already forwarded that hook. A finite Hamiltonian
proposal overflowing a generated Gaussian therefore aborted rather than diverged.
Both Enzyme adapters now classify only their owned `NonFinite` variant. Stable
callback-contract tests and a real generated native Gaussian test cover typed
direct errors, failed initialization, integration rejection, unchanged live state,
and scratch reuse. Explicit HVP failures remain errors. No derivative arithmetic,
precision tolerance, chart, clipping or adaptation rule changed.

Second, the reporting code counted diagnostic failures but still included their
bulk ESS/sec. The kernel runner also marked incompatible pairs unsuccessful only
in memory: replaying its raw successful-exit records could restore their timing
eligibility. These are evidence-pipeline bugs, not speed improvements. Summaries
now recompute complete, finite, multi-chain, all-coordinate diagnostic admission;
ESS/sec requires the declared R-hat/ESS/divergence gates. Computational timings
remain separately labelled. Append-only compatibility decisions govern paired
ratios and kernel timing eligibility, with unvalidated pairs withheld. Protocol
tests run the actual runner control flow with incompatible child results, then
reconstruct exactly the same partial summary from persisted journals. Corrected
historical derivations must be new artifacts, not replacements of sealed reports.

Third, source archives omitted verification projects and two case studies that
shipped documentation linked to. Explicit source allowlists now include those
tools and lockfiles but exclude generated binaries, datasets and raw capsules.
Creation and verification check first-party Markdown file-target closure against
the actual shipped manifest. Release CI builds/tests the included verification
tools from Git-free extracted source. This establishes current source usability,
not availability of the historical machine, binaries or private evidence corpus.

The recurring lesson is to test end-to-end contracts at every boundary: a correct
kernel can still acquire the wrong error policy through an adapter, a completed
run can still be statistically inadequate, and a correct in-memory verdict can
be lost by an incomplete journal. CI must exercise these paths without relying on
hardware timing thresholds. This stabilization increment makes no new numerical
range, speed, ESS/sec, general uncertainty or severe-model qualification claim.

## 10. Current-checkout comparison with Stan and JAX (2026-11-02)

This is a new CPU/float64 measurement cohort, not a reconstruction of the original
timings or a before/after optimization claim. Only benchmark instrumentation and
reference clients changed in this increment; Alea's probability, gradient and
NUTS implementations were not replaced. The comparison covers the six original
models, not arbitrary tensor programs, Poisson, HVPs, GPU or threaded execution.

### Source identity and protocol

The measured checkout has uncommitted implementation changes. Git HEAD
`3984ba6c3773a6542d48019d61a1f07594889526` alone does **not** identify it. The
source capsule SHA-256 is
`b758ced364930ca0a67f09a36d3e47497ce92e5d5062569888ef3baf1cc0742a`;
the Alea executable SHA-256 is
`44a44efdff43da6839b8c764fc31f2cba36d19f55df1a0b7bb25e06e9b8e4575`.
All inputs, reference executables, scripts, compiler identities and output hashes
are recorded under this local, retained evidence directory:

```text
/home/usu/documents/dev/alea-inference-evidence/three-engine-2026-11-02.Ytcnhd/campaign-v4
```

The host is an Intel i5-8250U, x86-64 Linux. All numerical workers are restricted
to logical CPU 2 with thread counts set to one. Its SMT sibling is not isolated;
this is a shared desktop with no fixed-clock claim. Rust is 1.95.0/LLVM 22.1.2,
release with fat LTO and one codegen unit. C++ is GCC 16.2.0 with `-O3` and no
explicit native-ISA flag. Alea's runtime SIMD dispatch and JAX's host code
generation therefore do not imply identical ISA policies. Nix/compiler manifests
and original build logs are retained, including reused binary provenance. Build
durations can be incremental and are not a clean Rust-versus-JIT comparison.

References are CmdStan/Stan Math **2.40.0** (not the root shell's 2.38), JAX and
jaxlib **0.9.2**, BlackJAX **1.5**, NumPy 2.4.2 and SciPy 1.17.1. The exact Nix
pins, Python closure and flake-lock digest are in `manifest.json`. JAX uses CPU
float64, disabled persistent compilation cache, explicit synchronization, and
disabled XLA fast math/Eigen threading, following its
[benchmarking guidance](https://docs.jax.dev/en/latest/benchmarking.html).

Data bytes and seed 20261029 are unchanged. Mean has two free coordinates;
Gaussian regression has eight coefficients plus log sigma; logistic regression
has eight coefficients. Sigma is `exp(u)` with one `+u` Jacobian and the same
restricted Normal prior. The shared half-normal normalization constant is omitted.
Before timing, **84 cross-engine checks passed**: CmdStan, JAX and Stan GLM
density/score probes, including small sigma and extreme logits, plus all-score
fixed-work checks. These bounded checks are not a full-range numerical proof.

### Full posterior value-and-gradient throughput

Each of seven independent processes per engine/model evaluates 10,000 positions,
changing the first free coordinate through 0, .01, ..., .09. Every value and every
gradient coordinate contributes to returned checksums. All **168 jobs** completed
and passed the receipts; no failed or incompatible job was dropped.

Median microseconds per complete posterior value-and-gradient call (lower better):

| Model | Alea native | Stan Math vectorized | Stan Math fused GLM | JAX compiled loop |
|---|---:|---:|---:|---:|
| Mean, N=1,000 | 2.60 | 3.56 | — | 4.66 |
| Gaussian, N=1,000 | 22.30 | 21.94 | 6.61 | 8.42 |
| Logistic, N=1,000 | 61.87 | 53.24 | 32.85 | 44.30 |
| Gaussian, N=10,000 | 213.28 | 344.25 | 102.06 | 79.93 |
| Logistic, N=10,000 | 615.01 | 741.75 | 462.61 | 438.69 |
| Centered funnel, K=8 | 1.14 | 0.50 | — | 1.01 |

The regression throughput gap is real in this cohort: Alea takes **3.38x / 1.88x**
the fused Stan Math time for small Gaussian/logistic models, and **2.09x / 1.33x**
for their large counterparts. It takes **2.65x / 1.40x / 2.67x / 1.40x** JAX's
time in the same order. This locates an evaluation-cost deficit independently of
NUTS trajectory counts; it does not, by itself, profile which current arithmetic
instruction or adapter causes that deficit. Comparing only with the original
Stan expression would hide the optimized-reference gap.

In thousands of full gradients/sec, Alea's median throughputs are
384.9 / 44.84 / 16.16 / 4.689 / 1.626 for the five regular models above;
the corresponding JAX rates are 214.5 / 118.8 / 22.58 / 12.51 / 2.280.
Per-process counts, times, medians, ranges and standard deviations are retained
in `records.jsonl` and `summary.json`. Observed min–max microseconds, in the same
five-model order, are:

| Engine | Mean | Gaussian 1k | Logistic 1k | Gaussian 10k | Logistic 10k |
|---|---:|---:|---:|---:|---:|
| Alea | 2.56–2.92 | 21.87–24.25 | 60.39–65.37 | 209.51–233.93 | 605.36–732.36 |
| Stan Math vectorized | 3.52–3.71 | 21.85–25.91 | 51.92–54.80 | 335.47–454.16 | 709.35–1,035.99 |
| Stan Math GLM | — | 6.47–6.94 | 32.11–34.68 | 100.22–109.11 | 452.26–593.66 |
| JAX | 4.49–6.63 | 8.18–9.25 | 43.56–57.24 | 77.73–81.19 | 434.33–542.75 |

The Stan fixed-work worker is a direct Stan Math reverse-AD client using
`normal_lpdf`, `normal_id_glm_lpdf`, `bernoulli_logit_lpmf` and
`bernoulli_logit_glm_lpmf`. It is **not** a generated model's optimized tape or
CmdStan process latency. Mean and funnel have no distinct GLM implementation;
the runner's duplicate control labels there are not a second algorithm. Actual
generated CmdStan executables were used in the separate inference attempts.
Different compensated versus ordinary reduction policies remain; numerical
agreement is tolerance-tested, not bitwise equality or identical tail arithmetic.

JAX receives data, loop count and **complete positions as runtime arguments**.
The loop synchronizes and consumes every gradient, while separate fields record
compilation and Python dispatch/completion time. The first partial campaign
(`campaign-v2`) instead constructed zero coefficients inside the loop, permitting
XLA specialization. It was interrupted, preserved and **excluded**. A build-only
`campaign-v3` still allowed invariant coefficients to be hoisted. The final v4
uses a runtime position matrix and verifies changed all-coordinate states against
independent AD calls. Optimized HLO is retained. This correction illustrates why
an apparently consumed scalar checksum alone does not establish equivalent work.

The v4 inference attempt completed all 154 jobs: 133 regular-model fits passed
the quality gates and 21 funnel fits failed them. However, its **parent process
timings and process ESS/sec are excluded**: Python's POSIX `wait(timeout=...)`
polls with sleeps of up to 50 ms, inflating short jobs and charging four waits to
Stan. Internal kernel timers and sampling diagnostics are unaffected. The launcher
now uses blocking wait with a watchdog; deadline and interrupt paths kill/reap the
worker. Its replacement `sampling-v5` passed preflight but was interrupted when
an unrelated Cargo build saturated the host (large JAX rose from ~32 to ~98 s).
Both attempts and partial outcomes remain available. They are not a clean,
completed three-engine workflow timing comparison.

## 11. Profile-driven regression kernel improvements (2026-11-02)

This increment changes the existing native kernels, not the benchmark's probability
laws, gradients, model complexity, transforms or NUTS algorithm. Public APIs and
workspace allocation counts are unchanged. The preceding baseline executable was
first checked against **all current Rust source hashes**; there was no intervening
production drift. Saved binaries, original data and release settings are paired.

### Where the time went

Fresh Callgrind profiles of 1,000 complete posterior evaluations locate the work
inside the fused likelihoods, not the graph/session frontend. The Gaussian profile
is dominated by compensated predictor/score arithmetic and inlined iteration;
the logistic profile additionally attributes **23.09% of instructions** to
`__log1p_fma`. These are instruction attributions, not sampled CPU-time percentages.
Hardware counter totals work on this host; `perf record cycles:u` is denied.

Inspection of the pinned Stan 2.40 `normal_id_glm_lpdf.hpp` and
`bernoulli_logit_glm_lpmf.hpp` shows vector expressions for residuals, squared
residual reduction and transpose-matrix score products, with invariant scale work
hoisted. Our existing compensated score kernel instead revisited the score
arrays for each row. Our exponential was vectorized, but `log1p` still executed
one scalar call per observation. An analytic gradient and a batch graph node
were therefore not sufficient to obtain efficient machine code.

The [Rust Performance Book's profiling](https://nnethercote.github.io/perf-book/profiling.html)
and [bounds-check guidance](https://nnethercote.github.io/perf-book/bounds-checks.html)
informed the local changes: identify the hot work, expose common extents, and
inspect/measure the result rather than remove safety checks. JAX's
[whole-function compilation](https://docs.jax.dev/en/latest/jit-compilation.html)
offers a useful fusion reference, not evidence that Alea needs another frontend
rewrite to fix this particular deficit.

### Implementation and numerical contract

In `regression.rs`, `Scores::add_tile` processes four observations while each
coefficient packet's head/correction remain live. It applies the **same Neumaier
operations in the same row order**. Equal, already-validated slice extents remove
redundant zip-length work without unsafe indexing. Normal, logit and Poisson full
tiles use it; observation tails retain the original path. No scratch heap is added.

In `kernels.rs`, `Design` caches whether bound offsets are zero. Predictor
initialization avoids compensated addition to zero and retains the first product's
FMA residual. Explicit `+0` preserves signed-zero behavior; rebinding recomputes
the flag. Successful predictors remain bit-identical, and nonrepresentable ones
still fail admission rather than being clipped.

`log1p_unit` computes the logistic kernel's bounded logarithm using
`t=x/(2+x)` and `2*(t+t³/3+...+t³⁷/37)`, with four independent Horner groups.
For `0<=x<=1`, the real-arithmetic omitted tail is below `1.5e-20`. This bound
does **not** bound floating-point rounding. Independent Decimal(400-digit)
fixtures and a 200,000-point grid instead test an observed four-ULP envelope.
Scalar execution, subnormal arguments and out-of-domain lanes retain libm.
Exponential tails, analytic scores and compensated likelihood reductions remain.
This is not fast-math, a tail cutoff or a correctly-rounded-logarithm theorem.
Log densities may change by a few ULPs; cross-version trajectory replay is not
promised. Existing density, finite-difference and error tolerances were not relaxed.

Recent [sufficient-statistics research](https://arxiv.org/html/2502.04990v1)
identifies an important larger Gaussian opportunity: reduce repeated `O(np)` work
to `O(p²)` after preparation. This increment does **not** implement that change.
Direct Gram subtraction can lose small residuals near a fit, especially at small
sigma; a stable representation and cancellation/gradient admission evidence are
required first. Nor do these changes make the remaining scalar reductions optimal.

### Reproducible before/after measurements

Seven alternating before/after process pairs per case, 10,000 complete calls per
process, CPU 2, one thread, Rust 1.95 release/fat LTO/one codegen unit; no additional
ISA flags. The competing build had finished before timing. The host/SMT sibling
are still not isolated. **All 84 jobs and 42 pairs qualified**: exact gradient
coordinates and gradient checksums, plus the unchanged density/checksum tolerance.
Every raw output is retained. Replay verifies input/output hashes and reconstructs
the saved summary exactly. Repetitions quantify timing spread, not new datasets.

| Model | Before median µs/call | After median µs/call | Paired after/before median [min–max] | After thousands gradients/sec |
|---|---:|---:|---:|---:|
| Gaussian, N=1,000 | 22.40 | 21.05 | .936 [.851–.979] | 47.50 |
| Logistic, N=1,000 | 61.97 | 40.24 | .648 [.579–.682] | 24.85 |
| Gaussian, N=10,000 | 218.65 | 199.86 | .919 [.912–.944] | 5.004 |
| Logistic, N=10,000 | 624.03 | 422.66 | .680 [.672–.687] | 2.366 |

Thus logistic evaluation is **1.54x / 1.47x faster** in this cohort. Gaussian
runtime falls only about 6–8%, not a comparable breakthrough. After-time ranges
in table order are 20.83–21.77, 39.78–42.21, 198.77–206.53 and 420.05–424.24 µs.
The unchanged mean/funnel controls have median paired ratios .995/.989 with
ranges spanning one; no consistent improvement is claimed for them.

Matched full-program Callgrind instruction counts (including cold setup) fall
from **139,159,492 to 127,155,460** for Gaussian and **319,239,132 to 200,437,073**
for logistic. Instrumented durations are excluded from the wall-time table.

A separate **single-cohort** four-chain native NUTS screen uses the original
1,000 warmup/1,000 retained configuration and saves all draw CSVs:

| N=10,000 | Before process s | After process s | Target calls before→after | Retained steps before→after |
|---|---:|---:|---:|---:|
| Gaussian | 11.69 | 11.08 | 49,863→49,863 | 25,616→25,616 |
| Logistic | 29.78 | 20.59 | 47,286→48,320 | 24,196→24,960 |

All four fits complete 4,000 retained draws, have zero retained divergences,
all-coordinate Rhat<=1.01, bulk/tail ESS>=100, and paired means within six combined
MCSE. Gaussian has the same trajectory hash and four recoverable warmup evaluation
errors before and after. Logistic's trajectory changes, uses more evaluations,
and still runs faster; minimum bulk ESS is 5,524→5,606. This screen supports a
real inference benefit, but is **not** seven-run inference uncertainty or a new
Stan/JAX ESS/sec ranking. The small Gaussian improvement leaves a substantial
gap to the previously measured fused Stan/JAX references; those references were
not remeasured in this optimization cohort, and no cross-cohort speedup over them
is claimed.

### Artifacts, checks and remaining limits

Evidence root:

```text
/home/usu/documents/dev/alea-inference-evidence/regression-opt-2026-11-02.30OZ4Y
```

`paired/` contains source, binaries, compiler/CPU/thread metadata, original data,
frozen driver, raw outputs and all repeated timings. `native-screen/` retains the
single-cohort draws, work counts and diagnostics. Root `*-before.callgrind` and
`*-final.callgrind` contain instruction attribution; intermediate candidates and
the denied sampling attempt are retained separately, not substituted for results.

```text
after source archive: 640fadf0847791c8906817e668e7a31399f8dbae847d6af49b732f9f5d0b040e
after executable:     97d620156bd18cbb845127e82813322078da00e91a79d8b5603a9183abad56af
paired manifest:      1764670fd0d3eb781ef2a8534800909688d9f42237c16e483c16067426bd02dd
screen manifest:      1ac8daf58324d8fd58b127b8734434bcdae8d28ce38f32ed5e27acbd7d0c08da
```

Use `scripts/regression_compare.py run --before ... --after ...
--baseline-manifest ... --data ... --output NEW_DIRECTORY` for a fresh campaign;
`report --output EXISTING_DIRECTORY` verifies/reconstructs it. Its `sample` mode
requires a verified paired capsule and explicitly labels a single-cohort screen.
The frozen `paired/driver.py` also replays its own cohort without checkout changes.

Validation includes release regression unit and integration tests, model-level
finite differences and the original numerical-failure regression, all-target
Clippy, zero-allocation tests, independent fixture regeneration and benchmark
protocol/source-policy tests. Raw replay and measured Rust source/archive equality
were checked. No public API or maturity promotion follows. Other architectures,
wide designs, full three-engine workflow timings, allocation/cycle attribution
and a stable sufficient-statistics Gaussian kernel remain separate work.

## 12. Removing the remaining Gaussian observation scan (2026-11-02)

### Diagnosis: the wrong repeated work, not a thin modeling interface

Section 11 improved the existing O(NP) kernels but did not close the Gaussian
gap. Its final Callgrind profile attributes 39.92% of instructions to inlined AVX
operations, 16.18% to inlined zip iteration, 9.29% to regression score work and
7.50% to the regression kernel. These source attributions overlap inlined
concepts; they are not wall-clock percentages or independent algorithm stages.
The compiled model's own logp/gradient frame accounts for only 0.70% self
instructions. The principal remaining cost is visiting every observation and
coefficient at every proposal, with compensated predictors and score accumulation.

The pinned Stan 2.40 `normal_id_glm_lpdf.hpp` also performs O(NP) work, but uses
Eigen residual and transpose-score expressions and ordinary reductions. Simply
removing Alea's compensation to imitate those reductions would trade accuracy
for speed. Adding more graph machinery would not remove the observation scans.
The algorithmic opportunity is that X, observations and offsets are fixed.

[Pichler, Jewson and Avalos-Pacheco (2025)](https://arxiv.org/html/2502.04990v1)
describe sufficient-statistic acceleration for probabilistic programs. That is
an established statistical identity, not a new inference method invented here.
The [Rust Performance Book](https://nnethercote.github.io/perf-book/profiling.html)
likewise motivates profiling the actual hot work before choosing an optimization.
Our implementation uses a square-root representation rather than evaluating
`y'y - 2 beta'X'y + beta'X'X beta`, whose subtraction is unsafe near a fit.

### Implementation

The private `regression/normal_compression.rs::CompressedNormal::new` performs
streaming Givens rotations of the augmented matrix A = [X, y-offset]. It retains
an upper-triangular R, not Q, an inverse, an estimated rank, or a ridge penalty.
In exact arithmetic A'A = R'R, including singular designs. With v = [-beta, 1]
and r = Rv, the existing Gaussian law is evaluated as

```
logp       = -N log(sigma) - N/2 log(2*pi) - ||r||²/(2 sigma²)
score_beta = R_X' r / sigma²
score_sigma= (||r||²/sigma² - N) / sigma
```

Rotations, transformed residuals, squares and score products use compensated
accumulation with product residuals. Preparation costs O(NP²) time/O(P²) extra
storage; each admitted evaluation costs O(P²), independent of N. The original
data and packed row kernel remain available. This changes neither the public
model call nor priors, parameter chart, Jacobian, model dimension or NUTS.
It works for arbitrary graph expressions supplied as coefficients, not only
the benchmark's direct parameters or its zero-valued coordinates.

`NormalRegression::{new,with_data}` prepares a fresh root for a binding, including
changed responses, offsets or design. Mutable score scratch remains execution
local. Conditional row reporting and generated scalar source still use original
rows; neither acquires fictitious compressed observation units. The existing
`RegressionPlan` describes CPU dispatch/storage for the row fallback, not the
additional root. Both Auto and Scalar use the same scalar compression arithmetic;
Scalar is still not a strict row-wise reduction-order option.

### Numerical admission and limits

The initial eligibility envelope is P<=64, N>=128 and N>=16(P+1). These are
conservative engineering limits, not a measured universal cost-model optimum.
The performance qualification below covers P=8 and the original mean case, not
every eligible shape or CPU. Failed/nonfinite/subnormal preparation leaves the
original kernel selected without changing the dataset.

Evaluation falls back to the original compensated row kernel for nonnormal
squared scale/reciprocal, nonnormal transformed residual norm, nonfinite
arithmetic, or a poor conditioning screen. That screen compares
`64*eps*N*||[X,y,offset]||_F*sqrt(1+||beta||²)` with `1e-8*||r||`.
The input norm includes observations and offsets separately, so their cancellation
cannot make the screen optimistic. This is a conservative heuristic, **not a
certified forward-error bound or a uniform arbitrary-conditioning guarantee**.
No RSS is clipped, no scale is bounded and no failed proposal is made successful
by replacing its density. The row kernel remains responsible for the result/error
after fallback. Scores are staged until the entire compressed result is finite.

Compression changes floating-point density **and score** rounding. The old
dispatch-only test still checks exact row-kernel bits, explicitly bypassing
compression in that test. New independent row-oracle tests use 3e-12 scaled
tolerances; the benchmark explicitly opts into the existing three-engine 3e-9
admission policy. Its default exact-score policy has not been weakened. There
is no cross-version trajectory replay promise or claim that the new root is
more accurate than the original row-wise law in every condition.

### Reproducible fixed-work results

Immutable evidence root:
`../alea-inference-evidence/normal-compression-2026-11-02.EZ9uzY`.
`paired/` contains seven alternating saved-executable pairs for all six original
cases, 10,000 complete posterior value/score calls each. All 84 attempts/42 pairs
qualified, with no failures or unstarted jobs. CPU2 affinity, one thread,
rustc 1.95/LLVM22, fat LTO and one codegen unit remain unchanged. This is a shared
i5-8250U desktop without frequency/SMT isolation; retain all outliers and ranges.
No own compilation, tests or profiler overlapped either timing campaign.

| Original case | Before median µs/call | After median µs/call | Paired after/before median [range] |
| --- | ---: | ---: | ---: |
| Mean N=1,000 | 2.895 | 0.526 | .1852 [.1769, .4511] |
| Gaussian N=1,000, P=8 | 21.829 | 2.295 | .1040 [.1016, .1244] |
| Gaussian N=10,000, P=8 | 206.991 | 2.390 | .01136 [.01078, .02312] |

The latter two median paired speedups are 9.61x and 88.04x. Median throughput
increases from approximately 45,810 to 435,740 calls/s, and from 4,831 to
418,389 calls/s. The median construction cost increases from .266 to 1.839 ms
and from 2.122 to 17.548 ms, respectively: roughly 80 additional evaluations
amortize preparation on these workloads. Few-evaluation applications can lose.
Logistic and funnel controls also remain in the archive; they do not establish
an algorithmic improvement from this Gaussian-only change.

`references/` separately rotates Alea, pinned Stan Math 2.40 fused GLM and pinned
JAX 0.9.2 CPU/f64 workers seven times on the same original data/runtime positions.
All 42 attempts/14 groups pass full density/score receipts. Compilation, imports,
host dispatch and preparation are outside the internal evaluation timers.

| Gaussian case | Alea µs/call | Stan fused GLM µs/call | JAX µs/call |
| --- | ---: | ---: | ---: |
| N=1,000, P=8 | 2.445 | 6.777 | 8.717 |
| N=10,000, P=8 | 2.258 | 104.820 | 80.862 |

Paired Stan/Alea ratios are 2.83 [2.70,3.04] and 45.30 [41.01,50.64]; JAX/Alea
ratios are 3.67 [3.40,4.41] and 35.73 [31.86,49.76]. These compare the supplied
implementations, not the engines' theoretical limits: Stan/JAX could also be
given a precompressed Gaussian model. Alea now recognizes that opportunity
internally without requiring benchmark-specific distributions or gradients.
These are **not CmdStan/JAX inference or ESS/sec rankings**.

Callgrind for 1,000 complete N=1,000 Gaussian evaluations falls from 127,155,460
to 28,953,265 instructions, including preparation. The new profile is retained
as `linear-candidate.callgrind`; instrumented time is excluded from wall results.

### Does the gain survive NUTS?

`native-sampling/` contains seven alternating before/after repetitions for each
Gaussian workload, four sequential chains per process, original seed/zero starts,
1,000 warmup and 1,000 retained draws, diagonal adaptation, .2 initial step,
.8 target and depth limit10. Every one of the 28 processes completes 4,000 draws
and passes the existing Rhat<=1.01, bulk/tail ESS>=100 and divergence gates.
These are timing repetitions of the same four streams, **not 28 independent
chain groups for estimating posterior uncertainty**.

| Gaussian workload | Before process s, median [range] | After process s, median [range] | Paired after/before median [range] | Minimum-coordinate bulk ESS/process s, before → after |
| --- | ---: | ---: | ---: | ---: |
| N=1,000 | .853 [.833,1.197] | .143 [.136,.195] | .1648 [.1529,.1756] | 7,729 → 40,854 |
| N=10,000 | 8.363 [7.520,10.569] | .191 [.179,.249] | .02360 [.02140,.03098] | 696 → 27,972 |

Those are approximately 6.07x/42.37x paired process speedups, and 5.29x/40.20x
minimum-coordinate ESS/sec ratios, on this host. Process time includes model
preparation, adaptation, sampling, CSV serialization and built-in diagnostics.
No timer overhead is subtracted. Each chain still constructs its own target;
sharing roots across chain factories is a possible separate improvement, not
part of these timings.

Rounding changes trajectories, adaptation and statistical work. N=1,000 target
calls change 52,304→51,610 and retained leapfrogs 26,512→25,864; N=10,000 changes
49,863→47,531 and 25,616→23,348. This small reduction in calls cannot explain
the runtime gain. Mean retained depths change 2.8395→2.7805 and 2.78925→2.62575;
maximum retained depth remains3. Final step sizes and trajectory hashes are in
each raw receipt. Minimum bulk ESS changes 6,595→5,823 and 5,820→5,355, so
iteration throughput alone would overstate the efficiency improvement.

Maximum Rhat stays <=1.00341. All coordinate mean differences are below six
combined MCSE (observed maxima2.853/2.171). There are no retained divergences;
the large workload still reports the same four **recoverable warmup evaluation
errors** per process, not zero numerical errors or a newly fixed transform.
Neither those errors nor the unchanged original warmup-error regression is
suppressed by Gaussian compression. A full repeated Stan/JAX inference/ESS
comparison has not been redone here.

### Evidence identities and replay

The measured after executable has SHA256
`59eb6de0cb9208281b8e79490567b1864d2940943495a8c032a84d9cbba81234`;
the source capsule has
`4d39877fa2660934a9b316abdf8a3946475fa01a7d5b7b32d01a90958990045e`.
All 335 archived crate Rust sources were compared byte-for-byte with the final
checkout. The source policy excludes profiling examples; it is not an archive
of every ignored file on disk. Paired/reference/sampling manifest hashes are
`847db2e19094c7cfa12bf3126d9c3bab2d3829fc1a84fe08879d3b0d1aa7009e`,
`504866c38d9262263dd03585c3066fd1343836deb6d73437a82be898ce4e646d`, and
`3672a98e2766ea826cd47042eb2e7c59bde8877503bbcf7aaf9eec5b045f4e8a`.

Each capsule contains frozen Python report drivers and input/output hashes.
Replay `paired/driver.py report --output .../paired`,
`references/gaussian_reference.py report --output .../references`, and
`native-sampling/driver.py report --output .../native-sampling`. The last returns
raw-rederived qualified rows, including every diagnostic and timing repetition.
Never overwrite a historical manifest or cached summary to admit a changed input.

### Validation and engineering lessons

Tests compare the compressed result directly with the original row kernel at
N=128/1,000/2,500/5,000/7,500/10,000, zero columns, offsets, dependent columns and
multiple scales. Near-exact fits, extreme coordinates/scales and nonfinite input
exercise fallback and staged outputs. Rebinding tests change X, y and offsets,
compare conditional row sums, check typed error propagation and recovery.
Existing finite-difference, posterior-equivalence, zero-allocation and original
warmup-underflow regressions remain separate from timings.
The final affected suite passes 101 unit tests (two existing timing experiments
ignored), 31 related integration tests, three native benchmark correctness tests,
22 fresh shared-coordinate CmdStan density/score checks, 37 benchmark protocol
tests, 12 source-policy tests, all-target distributions Clippy and scoped rustfmt.

The lesson is not “always vectorize” or “compensation is a bug.” First distinguish
slower evaluations from extra sampler work; then ask whether fixed data permit
removing work altogether. Preserve a trustworthy row oracle, count preparation,
retain original reporting units, qualify changed arithmetic explicitly, and
measure actual posterior trajectories as well as attractive fixed positions.
Wide designs, arbitrary-conditioning certification, cross-architecture tuning
and a new full three-engine inference comparison remain unqualified.

## 13. Beyond Gaussian: evaluation kernels and tape overhead

The next request explicitly covers **evaluation backends and kernels**, not
every sampler algorithm. This distinction matters: a faster gradient does not
establish faster HMC, NUTS, optimization or better statistical efficiency by
itself. Nor does a few-model benchmark establish globally optimal code for every
scalar law, tensor program or optional compiler backend.

### Profiles and targeted changes

The source-sealed follow-up is in
`../alea-inference-evidence/remaining-engines-2026-11-02.8dTBYT`.
Fresh Callgrind profiles of 1,000 original posterior calls attribute logistic
work primarily to SIMD arithmetic, iteration and the batch kernel; total work
is 200,435,563 instructions. A Poisson production-kernel control uses
168,660,964 instructions. The small funnel instead spends substantial work in
the scalar tape, parameter layout and Normal callbacks (8,895,749 instructions).
These are instruction counts, not sampled CPU-time percentages.

Implemented improvements span several layers:

- `regression/kernels.rs`: the logit sigmoid divisions and likelihood terms now
  execute in SIMD packets. Logit and Poisson likelihoods use lane-local Neumaier
  accumulation and merge heads **and corrections**, not an ordinary horizontal
  sum. Scalar tails and stable signed-logit formulas remain; no Stan tail cutoff
  or subtraction of rounded probabilities is introduced.
- The bounded exponential uses a 16-bin power table and degree-seven polynomial
  on a smaller residual interval, rather than a degree-thirteen dependency
  chain. Its real truncation bound is below 1.3e-18; that is **not** a floating-
  point rounding theorem. Independent Decimal80 fixtures now cover table bins,
  nearest-integer switching surfaces and their f64 neighbors (9,136 input rows
  across both signs), alongside dense grids and exact libm fallback tests.
- Predictor additions use error-free TwoSum, retaining product residuals and
  coefficient order. Nonfinite candidate lanes retry the original ordered
  compensated predictor, protecting representability-limit admission. This is
  not permission to replace compensated sums globally.
- `core/model.rs`: all-identity layouts are recognized once. They avoid redundant
  constrain/copy/pullback passes, while preserving transform input errors,
  checked evaluation, reentrancy, signed-zero behavior and staged output.
  Mixed/nonidentity layouts keep their original path.
- `distributions/scalar.rs`: native scalar Normal density and scores share the
  standardized residual and the authoritative density expression. The existing
  finite/extreme/error-order bitwise oracle remains unchanged and passes.
- `core/model/assembly/reverse.rs`: compile derivative routing once. Only edges
  to parameter-dependent expressions remain; repeated-input accumulation order
  stays unchanged. The original semantic/forward tape, all callback executions,
  local-score admission, source export and factor selection remain authoritative.
  Constant derivative pruning must **never** silently remove a primal error.
  A randomized old-interpreter oracle checks exact gradient bits; explicit tests
  check errors from constant-input callbacks. The extra O(edges) immutable plan
  is a construction/storage tradeoff, not a new modeling language or JIT.

The shared interpreter changes affect ordinary scalar graphs and tensor factors
embedded in them. They do not automatically rewrite independently generated
Enzyme source into the native SIMD kernels.

Gaussian's observation-independent sufficient statistics are not a template for
an equivalent O(P²) evaluation of arbitrary logistic/Poisson designs. For example,
the Poisson normalization still sums exponentials of parameter-dependent row
predictors; [Pichler et al., section 3.4](https://arxiv.org/html/2502.04990v1)
describe only a partial sufficient-statistics reduction there. This experiment
keeps the stable original formulas rather than subtracting large precomputed
linear terms solely to obtain an attractive complexity claim.

### Backend qualification boundaries

| Evaluation path | Evidence in this increment | What remains unqualified |
| --- | --- | --- |
| Native batch Normal/logit/Poisson | Model-level density/score/finite-difference, shape, tails, allocation and error-recovery tests; saved-binary regression controls | Every shape, ISA, layout or optimal scheduling threshold |
| Native scalar analytic tape | Exact reverse-interpreter oracle; callback-error and factor-selection tests; original funnel and Normal-factor controls | Global primitive fusion, shared dynamic Normal normalization across distinct factors, arbitrary callback optimization |
| Native tensor/control-flow interpreter | Matrix, broadcasting, aliasing, both-branch, scan, empty-result and failure-recovery tests; tensor-factor controls | General fusion, tiled/SIMD matrix backend, all tensor shapes and programs |
| Generated source / actual Enzyme | Executed derivatives through third order, native batches, matrices, lazy branches, zero-trip scans, error recovery and sampler consumers | Native-kernel/source performance parity, Gaussian root lowering, general primitive custom-VJP fusion |
| Other scalar/catalog/scientific kernels | Production-kernel controls where supplied; existing family-specific correctness contracts | A fresh catalog-wide Stan/JAX ranking, all multivariate/Wiener regimes or every optional linear-algebra backend |

The Enzyme suite was executed with its actual pinned compiler, not replaced by
`cargo check` or source-string inspection. The compiler still prints its existing
`freeing without malloc` diagnostics for the nested control-flow fixture; the
executed tests pass. Passing those fixtures is not arbitrary-program compiler
qualification. Independent source generation deliberately retains its existing
mathematical rules; a native optimization must not be advertised as a generated-
backend speedup without measuring that backend.

### Measured results and rejected experiments

The `paired/` capsule contains seven alternating repetitions of the six original
models, with 10,000 full posterior value/all-gradient calls per process. All 84
jobs and 42 pairs qualify against the declared numerical score policy (3e-9),
with no failures or missing jobs. This is not the earlier exact-score protocol:
the predictor and exponential arithmetic intentionally change rounding.

| Model | Before / after median µs per call | Paired after/before median [range] |
| --- | ---: | ---: |
| Normal mean | 0.365 / 0.331 | 0.910 [0.658, 1.546] |
| Gaussian, N=1,000 | 1.703 / 1.552 | 0.931 [0.781, 0.999] |
| Logistic, N=1,000 | 27.863 / 24.274 | 0.870 [0.789, 1.025] |
| Gaussian, N=10,000 | 1.639 / 1.515 | 0.933 [0.822, 1.084] |
| Logistic, N=10,000 | 310.062 / 258.555 | 0.842 [0.773, 0.861] |
| Funnel | 0.833 / 0.503 | 0.602 [0.321, 0.636] |

The `references/` capsule rotates Alea, pinned Stan Math 2.40 GLM and JAX CPU
f64 workers in seven groups per case. All 126 jobs and 42 groups qualify against
the original density/all-score tolerances; no failed or unstarted jobs. Median
paired speed ratios below are **reference time / Alea time**, not ratios of
unpaired medians. The measured candidate is the `paired/bin/after` snapshot;
later scalar-backend changes must not retroactively relabel this snapshot.

| Model | Stan / Alea median [range] | JAX / Alea median [range] |
| --- | ---: | ---: |
| Normal mean | 8.324 [2.956, 10.394] | 10.243 [3.747, 11.767] |
| Gaussian, N=1,000 | 2.879 [1.634, 3.234] | 3.964 [2.012, 4.578] |
| Logistic, N=1,000 | 0.961 [0.695, 1.018] | 1.279 [0.869, 1.363] |
| Gaussian, N=10,000 | 46.769 [35.788, 52.767] | 38.919 [28.008, 42.254] |
| Logistic, N=10,000 | 1.335 [1.233, 1.814] | 1.293 [1.233, 1.802] |
| Funnel | 0.566 [0.458, 0.655] | 1.388 [1.019, 1.615] |

Large logistic now wins every measured reference group. Small logistic remains
near parity with Stan, not a demonstrated consistent win. The funnel is still
about 1.77 times slower than Stan despite the native improvement. Gaussian
retains its earlier specialized sufficient-statistics advantage; references
could implement equivalent compression too. These are evaluation-throughput
results, not a fresh three-engine ESS/sec ranking.

The shared i5-8250U desktop has variable clocks and an unisolated SMT sibling.
CPU affinity is 2, thread counts are one, stable Rust is 1.95/LLVM 22.1.2, release
fat LTO/one codegen unit. Compilation, construction and profiling are outside
evaluation timers. No own builds/tests/profiles overlap timing campaigns.
Absolute times differ materially between cohorts: compare paired ratios within
each cohort, never the absolute times across cohorts. Ranges retain all outliers
and are descriptive spread, not confidence intervals or cross-machine promises.

An additional `score-trial/` tried branch-free TwoSum for coefficient-score
accumulation. All 18 jobs passed exact-score checks, but three-pair logit
after/before medians were 1.007 (N=1,000) and 1.016 (N=10,000). The change was
reverted: fewer apparent arithmetic operations did not deliver a repeatable
improvement. The original ordered Neumaier score accumulator remains shipping.
Keep this negative result: operation-count arguments are not measured speedups.

Explicit scalar regression dispatch exposed a separate backend mismatch: the
SIMD-oriented exponential polynomial executed lane by lane, with scalar FMA
support accounting for about 18% of the baseline scalar-Poisson profile's
instructions. Scalar (and vectors wider than the four-row tile) now use libm
exponential directly. This is backend selection, not relaxed accuracy: independent
Decimal fixtures, exact scalar/libm tests, extreme-rate admission, finite
differences and allocation/error-recovery tests remain required.

The final `backend-controls/` uses an identical benchmark-only harness overlay
on the saved Gaussian baseline and current production kernels. Seven alternating
pairs per shape produce 98 successful jobs/49 compatible pairs, no missing jobs.
Poisson checks use the existing checksum tolerance (relative 1e-12/absolute 1e-8);
Normal-factor and tensor controls require exact checksums. These are fixed-work
controls, not new posterior benchmarks or full-gradient equivalence proofs.

| Control | Before / after median µs per call | Paired before/after median [range] |
| --- | ---: | ---: |
| Auto Poisson, N=1,000 | 22.457 / 20.692 | 1.081 [1.067, 1.117] |
| Auto Poisson, N=10,000 | 219.307 / 205.370 | 1.073 [0.788, 1.079] |
| Scalar Poisson, N=1,000 | 90.771 / 54.571 | 1.665 [1.645, 1.734] |
| Scalar Poisson, N=10,000 | 930.247 / 562.465 | 1.642 [1.547, 1.850] |
| 128 scalar Normal factors | 9.840 / 5.367 | 1.834 [1.820, 1.905] |
| Tensor factor, 1,000 elements | 11.550 / 11.794 | 1.011 [0.910, 1.063] |
| Tensor factor, 10,000 elements | 111.393 / 106.280 | 1.035 [0.999, 1.111] |

Scalar Poisson and Normal-factor gains repeat across all pairs. The tensor
controls do **not** demonstrate a material consistent improvement: faster outer
tape routing does not remove the tensor interpreter's element work. Likewise,
Auto Poisson's large control has an outlier loss. Preserve those deficits rather
than extending the successful Gaussian/logit result to unmeasured engines.

Subsequent tensor-specific changes and measurements are documented in the
[tensor execution-plan follow-up](tensor-ir.md#execution-plan-follow-up-immutable-bindings-and-active-adjoints).
They supersede the lack of gain for those later tensor fixtures, not the
historical control results above, and do not imply generic fusion or compiler
backend performance parity.

`scalar-dispatch-confirmation/` then compares the final executable with the
reference-measured candidate: seven pairs of all six original models, all 84
jobs/42 pairs qualify with **exact gradient receipts**, no failures/missing jobs.
The final change is not advertised as a new Auto-model speedup: paired median
after/before ratios lie between 0.985 and 1.013. Reference results remain tied to
their actual measured snapshot, not silently transferred to the new executable.

Final Callgrind totals, including preparation, are 181,993,875 instructions for
1,000 logistic calls (baseline 200,435,563), and 5,766,962 for 1,000 funnel calls
(baseline 8,895,749). Scalar Poisson's 100-call profile falls from 75,721,709 to
57,634,601 instructions. Counts support the direction of fixed-work improvement,
not a CPU-time allocation proof or an inference/ESS claim.

The actual Enzyme tiny-model example executes seven alternating rounds of one
million calls per backend, CPU 2, no overlapping own workload. All 21 comparisons
pass their checksum assertion; the separate conformance suite checks all
derivative coordinates. Analytic-tape/generated median ns per call are
131.693/37.920 (polynomial), 367.338/127.853 (small regression), and
347.326/221.979 (small logistic). Paired tape/generated ratios are respectively
3.393 [3.226, 4.064], 2.828 [2.675, 3.074], and 1.604 [1.185, 1.750]. This uses
Rust 1.98 nightly `4429659e4`/LLVM 22.1.7 rather than stable: it is a descriptive
same-compiler backend comparison, **not before/after compiler optimization** or
the original large-data benchmark. The CSV and executable/source digests are
retained; no claim of generated native-batch SIMD parity follows.

### What to do differently next time

1. Preserve mathematical operations and error contracts independently of the
   selected execution strategy. Keep an independent scalar/old-interpreter
   oracle; pruning derivatives must not prune invalid primal computations.
2. Measure the actual backend. SIMD polynomials can be poor scalar algorithms;
   faster native callbacks do not imply faster tensor or source-generated code.
3. Attribute model evaluation separately from sampler work and preparation.
   An O(edges) reverse plan and an O(np²) Gaussian root both have cold costs.
4. Qualify density and every score, not only an aggregate checksum. Include
   extreme representability, aliases, constant-input errors, recovery and
   allocation tests before timing. Changed rounding needs an explicit policy.
5. Save unsuccessful experiments and noisy controls. Do not ship arithmetic
   rewrites on operation counts alone, or extrapolate one successful workload
   into “every engine is optimal.”

Remaining priorities are the Stan funnel gap (shared scalar normalization and
callback/forward overhead), native tensor fusion/vectorized primitives, and
measured source/native-kernel lowering parity. Wide graphs/designs, cold-plan
regret, non-x86 targets, catalog/scientific kernels and fresh multichain
three-engine ESS/sec qualification remain open. No numerical validation was
weakened and no benchmark-local probability law, derivative or sampler was added.

## 14. Applying the sufficient-statistics paper

Reviewed [Pichler, Jewson and Avalos-Pacheco (2025), arXiv v1](https://arxiv.org/html/2502.04990v1)
and the [reference implementation at commit e10fafd](https://github.com/jejewson/ProbabilisticProgrammingSufficientStatistics/tree/e10fafd9f13907997ac88ffb0372c25860dc2fdd).
The mathematical identities are implemented independently; the reference Stan
code is not copied into Alea or its benchmark. Existing square-root Gaussian
compression already implements the main data reduction in a cancellation-resistant
form, rather than subtracting separately rounded quadratic forms.

| Paper application | Alea applicability and implementation |
| --- | --- |
| Gaussian regression | Existing `normal_compression.rs`: prepared square root, original row fallback, estimated sigma and arbitrary graph coefficients/priors |
| Conditional mixed effects | Fixed group-indicator columns with coefficients `tau*z_group` already use that same Gaussian reduction; a new model-level test checks latent/scale scores and both positive-scale Jacobians against uncompressed chunks and finite differences |
| Poisson with one common predictor | New `poisson_statistics.rs`: recognize exactly equal design rows **and offsets**, prepare count/factorial sums, evaluate one rate; public native modeling API unchanged |
| General Poisson regression | The paper's partial `X' y` reduction was implemented, tested and measured, then removed from shipping execution because it did not consistently improve the existing fused value/gradient kernel |
| Marginalized Gaussian factor analysis / Woodbury | Not automatically applied to explicit latent-coordinate models; no new marginalized-covariance batch primitive is supplied by this increment |

For the common-predictor Poisson batch, with `eta = x' beta + offset`, the full
log mass and coefficient scores are

```text
logp = sum(y) * eta - N * exp(eta) - sum(log(y!))
score_beta = x * (sum(y) - N * exp(eta))
```

The factorial term is retained, unlike the reference posterior-only Stan model.
There is no conjugacy assumption, count rounding/clipping or prior restriction.
Preparation checks exact f64 bits, including offsets; it does not cluster nearly
equal rows. Storage keeps original rows/counts for reporting, source and fallback.
The initial eligibility envelope is N>=128 and P<=64, not a cost-optimal planner.
Preparation is O(NP), eligible value/all-score evaluation O(P), original storage
O(NP). Same-shape rebinding constructs fresh statistics, including response-only,
offset and design changes. The CPU plan still describes the original row dispatch,
not a newly exposed statistics/backend selection API.

All calculations remain compensated where reductions occur. The optimized path
declines |eta|>32, nonfinite intermediate results, and heuristic cancellation
screens for density and scores. These are optimization limits, **not model
constraints**. Declining retries the unchanged row evaluator; its typed success
or failure remains authoritative and caller gradients are staged. The screen is
an engineering safeguard, not a certified uniform rounding-error bound. Source
generation and pointwise rows deliberately remain unchanged; no generated-backend
or pointwise-reporting speedup follows from a native aggregate.

### The rejected partial reduction matters

The reference's general Poisson identity precomputes `X' y` but still evaluates
`sum(exp(X beta + offset))` and its design-weighted gradient. Alea already
precomputed factorials and fused predictors, values and scores. Replacing
`X'(y-rate)` with `X'y-X'rate` removes comparatively little work and introduces
cancellation near stationary points. The trial used compensation, preserved full
constants/offsets and retried rows near cancellation; its source and binary are
retained, not quietly substituted into the final measurements.

The seven-pair `paired/` trial has 84 successful jobs/42 compatible pairs. Paired
before/after medians [ranges] for distinct predictors were 0.957 [0.845,1.082]
(Auto, N=1,000), 1.027 [1.014,1.051] (Auto, N=10,000), 0.989 [0.981,1.081]
(Scalar, N=1,000) and 0.990 [0.949,0.995] (Scalar, N=10,000). Thus a modest
large-Auto gain did not justify default regressions elsewhere. The final source
retains the original row kernel byte-for-byte and prepares statistics only for
common predictors. This is evidence-based applicability, not a claim that the
paper's identity is mathematically wrong or never useful on other backends.

The new Gaussian mixed-effects test retains sampled `z`, `tau`, and `sigma`;
it does not integrate them out. Its augmented design may be rank deficient and
still needs no inverse or artificial ridge. This validates a supported small-group
composition, not sparse large-group performance: the existing P<=64 preparation
envelope still applies. A future marginalized factor-model implementation needs
an explicit covariance/latent semantics contract, stable Cholesky/solve-based
likelihood and independent derivatives; copying the reference's explicit inverse
or silently changing a user's sampled parameter space is not an acceptable
automatic optimization.

### Final saved-binary measurements

Evidence: `../alea-inference-evidence/sufficient-statistics-2026-11-02.LpcuyI`.
The historical production source is the prior increment's final archive
`bb6dbf5027f8201bac478677f9952639e828909d53428dc46387736c7e41a8d6`.
Both builds use an identical `kernel_audit.rs` harness overlay to add the common-
predictor case; no benchmark-local distribution, derivative or sampler is used.
Pinned stable profiling shell, Rust 1.95/LLVM 22.1.2, fat LTO/one codegen unit,
CPU 2, one thread, same shared i5-8250U limitations as section 13. No own build,
test or profile overlaps wall timing. Construction is measured separately.

The final seven alternating pairs per shape yield 84 successful jobs and 42
compatible pairs, zero failed/unstarted. All controls consume value and every
gradient coordinate; Poisson checksums use the existing relative 1e-12/absolute
1e-8 policy, supplemented by the separate full-coordinate correctness tests.
Raw stdout was reparsed and pair/summary qualification independently rederived.

| Production control, P=8 | Before / after median µs per call | Paired before/after median [range] |
| --- | ---: | ---: |
| Common predictors, N=1,000 | 20.086 / 0.219 | 91.907 [81.279,94.420] |
| Common predictors, N=10,000 | 208.204 / 0.209 | 995.680 [920.224,1036.290] |
| Distinct predictors, Auto, N=1,000 | 19.897 / 20.442 | 0.972 [0.959,0.982] |
| Distinct predictors, Auto, N=10,000 | 206.917 / 203.155 | 1.018 [0.924,1.025] |
| Distinct predictors, Scalar, N=1,000 | 55.103 / 56.066 | 0.983 [0.980,1.059] |
| Distinct predictors, Scalar, N=10,000 | 575.300 / 572.424 | 1.005 [0.979,1.064] |

Common-predictor construction medians are 0.319→0.334 ms and 3.012→3.138 ms.
At these repeated-evaluation shapes, the extra construction amortizes in about
one call; that is not an all-shape preparation guarantee. Evaluation speedups
are **only** for the common-predictor shapes, not arbitrary Poisson regressions.
The small Auto control loses about 2.8% consistently in this cohort. Its row
kernel is unchanged; Callgrind totals for 1,000 evaluations including construction
are 156,828,140→156,839,813 instructions (about 0.007% more). That does not explain
the timing difference or establish a hardware-level cause; minor control/code-
generation regret remains unresolved and is not disguised as an improvement.
Ranges include all observations, not confidence intervals. No fresh Stan/JAX
comparison, inference timing, ESS/sec or cross-ISA conclusion is drawn here.

Final source archive digest:
`5edb777b3ab8d8cd594b38f4365b546cf254a5d3c9e3fa51715e354b28749bc6`.
Its 339 archived crate Rust sources matched the checkout byte-for-byte when
sealed; later reporting/test additions are retained separately in the qualified
source archive, without changing production kernels. Excluded profiling/example
inputs are not thereby claimed covered. The final kernel binary
is `e1260351714088ef4798b8f2138a0a745717d1d461fdef8babc95ad7732d97ab`.
The rejected partial trial, baseline overlay, final binary, manifests and raw
journals remain separate. Documentation completed after measurement is not part
of the measured snapshot.

Validation passes 104 distributions unit tests (two pre-existing timing tests
ignored), 34 related integration tests including the new mixed-effects test,
three native benchmark correctness tests, 39 Python benchmark protocol tests,
all-target distributions Clippy and scoped formatting. Tests cover N=128/129/
1,000/10,000, zero columns, scalar/Auto, exact offset/design eligibility, counts
through 2^53, near-stationary cancellation, underflow/overflow, full normalizers,
finite differences, transactional failures, allocations and consuming rebinding.

## 15. Checking the November 3 source-level Stan assessment

The assessment's central priority is sound: qualify and accelerate ordinary model
composition before adding more samplers. Its performance hypotheses are not
measurements, however, and several inventory statements need correction against
the checkout. The [tensor follow-up](tensor-ir.md#dependency-safe-fusion-and-first-order-source-replay)
records the new implementation and its remaining regressions separately.

| Assessment claim | Evidence and correction |
|---|---|
| General scalar evaluation is interpreted | True for the native assembly/reverse plans. But `alea-autodiff/src/codegen.rs` and core `assembly/source.rs` already export AOT Rust for the public builder; compiled execution is not wholly absent. It requires source rules and the pinned compiler, not arbitrary Rust closures. |
| Tensor fusion is missing | Partly superseded: dependency-safe pointwise DAG/reduction schedules now exist, with separate native/AOT selection. Dense intermediates and the original reverse order remain; generic reverse fusion, liveness allocation and tuned GEMM are still gaps. |
| Numeric safeguards explain the Stan gap | A plausible contributor, not established by source inspection. Our fixed-work experiments demonstrate both useful large-DAG fusion and regressions in unrelated controls. Count evaluations and profile the same density before attributing a sampler wall-time ratio. |
| Pathfinder should be added | Already present as `adapt::multi_pathfinder`, with bounded L-BFGS paths, ELBO selection, raw-mixture or explicit pathwise PSIS weighting and typed reports. `tests/pathfinder_robust.rs` exercises it. It is an S1 initialization service, not evidence of Stan-equivalent end-user approximate-inference maturity. |
| Alea has only GPL licensing | Too broad. Six reusable inference crates are `MIT OR Apache-2.0`; the facade, distributions, FFI and CLI are GPL. See the package manifests and [licensing boundary](../LICENSING.md), including the actual linked dependency closure. |
| No native Laplace/ADVI workflow was verified | Do not equate observed-information covariance or Pathfinder's local Gaussian with a qualified standalone Laplace/ADVI posterior workflow. Those remain distinct evidence/consumer gaps. |
| Old seven-run wall times describe current performance | Incorrect. They predate the kernel/statistics changes and contain different Gaussian variance models. Current per-evaluation evidence and four-chain evidence are separate records above; neither establishes all-model Stan superiority. |
| The historical `PrecisionLoss` failure is unverified today | The review did not run it, but section 2 already records the original seed/data excursion, nine row-prefix sizes, the classifier defect and recovery after the fix. `tests/native_numerical.rs` separately keeps transform/initialization errors typed while treating opted-in proposal failures as divergences. This is not blanket recovery for arbitrary backend errors. |
| Numerical qualification remains a large gap | Correct as a broad production claim. Passing bounded graph/gradient/Enzyme fixtures does not establish arbitrary programs, all architectures, hard-posterior calibration or universal error bounds. |

Stan 2.40 exposes a user-facing multipath Pathfinder workflow, whereas the local
Alea implementation is explicitly an initialization service; this is a workflow
and validation comparison, not an absent-algorithm comparison. See
[CmdStan's Pathfinder configuration](https://mc-stan.org/docs/cmdstan-guide/pathfinder_config.html).
Likewise, [Stan Math's matrix reverse kernels](https://github.com/stan-dev/math/blob/develop/stan/math/rev/fun/multiply.hpp)
are a useful lowering reference, not proof that every Stan matrix expression
becomes an optimally fused kernel. A moving `develop` URL is not a version pin.

### Compiled execution without a new modeling language

Build on the existing Rust exporter rather than introduce a second model API.
Keep mathematical IR and backend scheduling separate: shapes, alias/index maps,
chart measure, selected factors, lazy regions and primitive derivative rules
belong to the former; blocking, dispatch, buffer reuse and source compilation to
the latter. Lower large native factors as whole kernels; never scalarize a
regression simply to pass it through a general tape. Opaque callbacks remain
native or fail source capability admission explicitly.

The next compiler gates are wider semantic-rule coverage and reliable
higher-order composition, not just generating more Rust. First-order source
checkpointing now uses explicit tensor VJPs and the existing native chart
pullbacks. HVPs/third derivatives still use the qualified compiler-owned reverse
path because differentiating the rematerialized pullback failed actual compiler
tests. A cache key must include graph/factor selection, shapes, frozen data,
charts, numerical policy and compiler/target identity. Data rebinding invalidates
that source specialization; no live target is mutated behind a sampler cache.

### Attribute the performance gap one layer at a time

Use the same saved parameter probes and complete score checks for these pairs:

| Pair | What it isolates |
|---|---|
| Direct native factor versus that factor registered in `ModelBuilder` | Factor dispatch, argument copies, chart and scalar reverse overhead |
| Scalar observation assembly versus typed batch lowering | Per-observation graph/dispatch cost and whether the lowering recovers the existing kernel |
| Tensor IR versus generated Rust for the same graph | Interpretation, intermediate traffic and compiler lowering; not a new probability law |
| Value-only versus fused value/gradient, with identical admission | Reverse propagation and gradient-storage cost |
| Retained versus replayed scan | Memory/work tradeoff, first/steady activity and replay overhead |
| Fixed probe replay versus complete NUTS | Additional evaluation count, tree/adaptation/output costs and statistical efficiency |

Collect CPU profiles and deterministic instruction counts separately from wall
cohorts. Record allocation counts, construction and steady-state work; preserve
controls and failures. Do not label changes in unchanged scalar controls as a
new scalar algorithm. Do not remove compensation or finite admission to make
two numerical policies look comparable.

### Reuse the existing PosteriorDB registry and qualify its evidence

Do not start another corpus, port collection or runner. The retained native-v3
experiment already contains 20 posteriors, six methods and three repetitions
(360 jobs), pinned to PosteriorDB revision
`91027f9cdd13b7e76283fec3f2898f59d4b83ec4`. The existing bundle loader and raw
archives remain authoritative. New `replay.py conformance` derives a separate
layered qualification summary from the original manifest/report without changing
their bytes or original policy. Under the explicit four-chain, both-coordinate-
space policy, 244 historical jobs qualify and 116 are sampling-unqualified.
All 20 archived payload parts (647,731,517 bytes) were checked and a local generic
capsule was restored through the original PosteriorDB loader. This is evidence
preservation/reclassification, **not** a fresh run of today's native kernels or
independent recomputation of all diagnostics/reference comparisons.

The old native ports include generated source/Enzyme pathways. Do not present
their timings as a benchmark of every current public native distribution.
Unsupported and unstarted cases remain explicit in other retained experiments;
an all-completed historical matrix does not establish universal expressiveness.
Capsule byte integrity also does not establish scientific admission.

[PosteriorDB](https://github.com/stan-dev/posteriordb) associates
model/data specifications with reference information where available; reference
draws and their provenance must be checked per posterior, not assumed to exist
for every entry. See its [database specification](https://github.com/stan-dev/posteriordb/blob/master/doc/DATABASE_CONTENT.md).

Each registered case should pin the PosteriorDB revision, posterior/model/data
IDs and bytes, notices, Alea commit, CmdStan/Stan Math/stanc and BridgeStan
revisions, compiler flags, CPU/ISA/thread policy and all seeds. Keep the Alea
public-builder model, the Stan source and an explicit constrained-coordinate
mapping together. Start with GLMs, centered/noncentered hierarchies and covariance
models; mark unsupported dynamical models as unsupported rather than silently
substitute another posterior.

Before timing, verify parameter ordering, dimensions, normalization constants,
charts and Jacobians at ordinary, posterior and boundary probes. Compare every
gradient coordinate with central finite differences and an independent Stan
evaluation. [BridgeStan's density/gradient interface](https://roualdes.us/bridgestan/latest/languages/julia.html)
exposes proportional-constant and Jacobian switches; choose them explicitly.
Different charts require a density/gradient change of variables, not a raw
comparison of unrelated unconstrained arrays. [CmdStan's gradient diagnostic](https://mc-stan.org/docs/cmdstan-guide/diagnose_config.html)
is an additional check, not a proof over all support.

Then measure two distinct protocols. Fixed-work replay compares construction,
value/gradient throughput, allocations and instruction counts without a sampler.
Repeated four-or-more-chain inference compares wall time, warmup/sample work,
bulk/tail ESS/sec, rank-normalized split R-hat, energy diagnostics, divergences
and depth saturation. Compare posterior summaries with reference uncertainty
and MCSE, not draw-by-draw RNG equality. Use functionals appropriate to the
posterior; a mean is not a suitable criterion when that moment does not exist.
[Stan's posterior analysis guide](https://mc-stan.org/docs/reference-manual/analysis.html)
describes the convergence/ESS distinction.

Separate compilation, startup, construction/preparation, warmup, retained draws
and serialization. Measure comparable persistence modes or report the CSV versus
memory asymmetry explicitly. Report repeated-run medians and uncertainty with
the chain/run as the replication unit; thousands of correlated draws are not
thousands of timing replicates. Failed, statistically inadmissible and unsupported
cases stay visible and are excluded from speed rankings, not from the registry.

The acceptance milestone is broad correct composition with competitive measured
throughput and ESS/sec. It is not a universal speed claim inferred from one
Gaussian compression, source-code resemblance to Stan, or the presence of an IR.
Passing pointwise target checks and R-hat/ESS thresholds is necessary evidence,
not research-grade qualification by itself: still require representative
reference-functional agreement with uncertainty, repeated-seed difficult
geometries, simulation-based calibration, documented unsupported cases, failure
recovery and independent reproduction. A missing criterion stays missing rather
than being inferred from a successful run. These criteria do not promote the
current experimental compiler or API maturity classifications.

## 16. Composition and feature costs (2026-11-17)

This study measures the current scalar and block implementations, including their
cold costs. It changes benchmark coverage and admission records, with no new
production optimization. The synthetic covariance fixture has 32 observations; the
centered/noncentered school fixtures have eight schools; Radon uses the existing
pinned PosteriorDB J=85/N=919 data. The scalar Radon control reuses the same
declarations and production Normal laws. All four retain their priors, constants,
charts and free-coordinate order. Five positive states qualify each scalar/block
pair. Criterion hot loops reuse state 0; fixed-work receipts evaluate state 0
first and states 1–4 as steady work. The separate named-source cohort cycles all
five states. Scalar `effect0...` names are explicit aliases for the school block's
`theta`/`z` ordering.

Measurements ran serially in exclusive ELTE `hpc2019` allocations on `cn18-42`,
a virtualized Xeon Gold 6240 node, CPU 0 affinity and one runtime/provider thread.
Stable Rust 1.95.0 and pinned nightly 2026-06-23 are separate cohorts; all use
`-Cembed-bitcode=yes -Ctarget-cpu=native`. The nightly AD control additionally uses
`-Zautodiff=Enable`. Builds and numerical tests precede timing; the saved
executables are hash-verified before reuse. Seven paired repetitions reverse
lane and representation order on alternating runs. Tables use medians; displayed
ranges are observed min–max over seven runs, not confidence intervals or posterior
uncertainty. Raw repetitions retain the spread where a table shows only medians.
Individual Criterion runs use 20 samples, 0.1s warmup and
0.3s measurement; their bootstrap intervals remain in the raw artifacts.

The first allocation, job 2781887, passed preparation and tests but omitted
`--bench` when invoking Criterion executables directly. It produced test-mode
outputs and no Criterion estimates. Its logs are preserved; the corrected timing
allocation reuses the same binaries. No result below is inferred from that
mistaken run. Gnuplot's loader failed on the cluster, so Criterion used its
Plotters fallback; the numerical and JSON sample outputs were unaffected.

### The named covariance source control

A separate same-node cohort uses the existing matrix runner and one pinned
nightly/Enzyme-enabled CLI binary for scalar, block and generated-source paths.
The existing Stan Math evaluator supplies independent numerical answers. All
variants replay the native five-state bank twice per ten-call cycle, with 100,000
full-density/all-gradient calls per repetition, seven rotated repetitions. State
copying and checksum accumulation remain in each loop. All 28 planned runs pass
raw-output replay admission; each Alea variant also passes 18 density and 90 score
comparisons, including scale tails and finite-difference probes. Maximum
normalized score disagreement against Stan is below 9.2e-16 on this bank.

| Covariance implementation | Median µs/call [observed range] | Setup µs [observed range] |
| --- | --- | --- |
| Native scalar | 4.354 [4.349–4.371] | 214.779 [197.217–222.316] |
| Native block | 6.429 [6.251–6.646] | 125.010 [123.737–127.655] |
| Generated first-order H | 3.194 [3.190–3.224] | 31.148 [29.888–32.704] |
| Stan Math vectorized | 3.087 [2.970–3.337] | 128.517 [123.923–137.893] |

The paired scalar/block ratio is 0.677 (0.655–0.697): block composition is slower
for this small covariance fixture. Generated first-order H is numerically
admitted, but reuses trusted analytic rules: emitted
`SOURCE_EXPLICIT_FIRST_ORDER=true`, checkpoint=false, tensor-score-validation=false.
It is not a pure compiler-reverse first-order result. Native/generated/Stan
accumulation orders differ, so their timings are numerical tradeoffs rather than
pure abstraction ratios. Neither this result nor construction amortization
establishes NUTS efficiency. The setup column includes state-bank loading and native model construction or the
source-target factory; input-file parsing happens earlier. Stan's setup builds
its input/observation arrays rather than a native executable, so those cold scopes
are not interchangeable. Source rendering and Rust compilation are separate
preparation costs. The original CLI source construction=0 placeholder was removed
for this study. Earlier cohort 2781895 retained that placeholder and is preserved
separately; the table uses rerun 2781898, with no pooling of cohorts. First-order
numeric/source flags are unchanged by the setup timer.

The updated CLI release-test/build stages took 66.28s/25.99s; dependencies were
copied from the prior source cohort. These are single cache-dependent build
observations, not clean compiler speed ratios. The captured Stan executable uses
GCC 8.3.0 (`.comment`), existing Stan Math 2.40.0, `-O3`, C++17 and
`-march=native`; its sealed binary is reused in the rerun.

### Native scalar/block lifecycle

All rows use the same full posterior and compensated policy within each pair; no
sufficient-statistics compression replaces observation rows. Ratios are
scalar/block: above 1 favors block. Times are µs.

| Stable default | Scalar hot µs | Block hot µs | Paired scalar/block |
| --- | --- | --- | --- |
| covariance,N=32 | 3.651 [3.565–3.681] | 5.678 [5.630–5.759] | 0.642 [0.632–0.647] |
| schools,N=8 | 1.367 [1.343–1.372] | 2.838 [2.830–2.891] | 0.475 [0.469–0.485] |
| schools_nc,N=8 | 1.514 [1.512–1.517] | 3.026 [3.016–3.055] | 0.500 [0.496–0.503] |
| radon,N=919 | 62.042 [59.503–67.017] | 68.587 [68.071–69.092] | 0.905 [0.861–0.980] |

Cold medians below are scalar/block pairs in µs. Returned objects are dropped in
the timed cold cases. Binding uses an already-built immutable definition; JSON
parsing is outside these Criterion cases. Full seven-run ranges remain in the
replayed artifact. These stages overlap with full construction and must not be
added together.

| Model | Full construction +drop | Bind snapshot +drop | Executable compile +drop | Context create +drop |
| --- | --- | --- | --- | --- |
| covariance | 118.436/28.376 | 8.423/0.591 | 19.290/1.649 | 15.056/8.308 |
| schools | 35.535/24.491 | 1.868/0.311 | 4.198/0.968 | 4.251/6.482 |
| schools_nc | 36.912/26.740 | 1.909/0.328 | 4.886/0.973 | 4.253/7.900 |
| radon | 3481.896/176.389 | 120.174/0.571 | 299.886/2.121 | 202.588/27.060 |

The Criterion first case includes setup-target destruction. Fixed-work phase
receipts exclude construction/destruction/logging but include gradient poisoning,
admission and output copies; their first/steady columns must not be pooled with
the uninstrumented hot-loop estimates. Steady receipts average four distinct
states per run; first receipts contain one cold call.

| Model | Criterion first +drop µs | Phase first µs/call | Phase steady µs/call |
| --- | --- | --- | --- |
| covariance | 35.213/14.251 | 11.113/40.547 | 4.375/7.016 |
| schools | 8.420/10.073 | 7.594/16.431 | 1.949/4.034 |
| schools_nc | 8.546/10.879 | 6.244/14.830 | 2.015/4.069 |
| radon | 496.187/124.996 | 81.982/108.492 | 61.951/80.290 |

Scalar covariance primal rendering: 80.295 [80.016–81.902]µs. Scalar radon primal
rendering: 1813.803 [1795.663–1847.534]µs. Rendering eligibility is not generated
derivative execution.

### Features, arithmetic and providers

The following ratios are reference/candidate, measured within the same paired run
index. Default Compensated Auto stays on its compatible reference provider;
feature availability alone does not select Faer/BLAS or prove SIMD use. Compiler
cohorts and AD compiler flags remain distinct axes.

| Candidate/reference | Covariance | Schools | Schools NC | Radon |
| --- | --- | --- | --- | --- |
| stable-faer/stable-default | 1.012 [0.996–1.028] | 1.004 [0.981–1.017] | 1.018 [1.004–1.023] | 0.997 [0.990–1.002] |
| stable-blas/stable-default | 1.006 [0.991–1.023] | 1.005 [0.993–1.014] | 1.005 [0.979–1.015] | 0.987 [0.937–0.999] |
| stable-both/stable-default | 1.001 [0.988–1.016] | 0.992 [0.987–1.018] | 0.999 [0.980–1.014] | 1.000 [0.991–1.009] |
| nightly-native/stable-default | 0.990 [0.983–1.017] | 0.987 [0.983–1.001] | 0.986 [0.974–0.994] | 0.973 [0.956–0.979] |
| nightly-simd/nightly-native | 0.995 [0.983–1.015] | 0.993 [0.957–1.005] | 1.007 [0.999–1.023] | 1.002 [0.986–1.027] |
| nightly-ad-native/nightly-native | 0.895 [0.885–0.899] | 0.988 [0.959–0.999] | 1.000 [0.981–1.018] | 0.679 [0.673–0.688] |

Arithmetic controls use the unchanged 1000×65 row kernels, outside Gaussian
compression eligibility. Standard changes the accuracy contract; these are policy
tradeoffs, not equal-accuracy abstraction wins. Existing cancellation
counterexamples remain applicable despite passing this deterministic state bank.

| Nightly SIMD | Compensated µs | Standard µs | Compensated/Standard |
| --- | --- | --- | --- |
| normal | 100.289 [99.017–101.848] | 37.504 [37.120–38.432] | 2.661 [2.644–2.716] |
| logit | 102.068 [101.343–103.241] | 50.876 [50.106–52.200] | 2.003 [1.949–2.055] |
| poisson | 96.815 [95.710–98.201] | 46.278 [45.538–47.478] | 2.100 [2.036–2.120] |

The existing 8×8 GEMM control requests each provider explicitly. Accelerated
providers use Standard; the default uses Compensated. No equal-accuracy claim
compares those policies, and these small-matrix costs do not establish thresholds
for other shapes. Source differentiation of Faer/BLAS remains unsupported.

| Stable both: requested matrix policy | Value +VJP µs |
| --- | --- |
| { arithmetic: Compensated, provider: Auto, cpu: Auto, threads: Sequential } | 3.360 [3.353–3.370] |
| { arithmetic: Standard, provider: Blas, cpu: Auto, threads: HostConfigured } | 0.908 [0.898–0.915] |
| { arithmetic: Standard, provider: Faer, cpu: Auto, threads: HostConfigured } | 0.816 [0.804–0.822] |

### Unchanged controls and derivative orders

These current-implementation costs preserve the scalar, logit, funnel, matvec and
retained/checkpointed scan controls. They are not a matched historical
before/after comparison. Generated costs include their declared score admission; L
tensor first-order controls retain native-score admission outside compiler AD. H
reuses trusted first-order rules. Higher derivatives use original compiler
reverse, including checkpointed fixtures; their higher reverse is not
checkpointed.

| Control | First/higher route | Native first µs | Generated first µs | Potential HVP µs | Potential third µs |
| --- | --- | --- | --- | --- | --- |
| polynomial_compiler_ad | C/C | 0.122 | 0.030 | 0.050 | 0.206 |
| normal129 | H/C | 0.319 | 0.227 | 14.173 | 210.438 |
| logit129 | H/C | 3.381 | 3.322 | 28.357 | 153.896 |
| poisson129 | H/C | 2.889 | 2.892 | 22.899 | 113.391 |
| normal_vector | H/C | 0.492 | 0.358 | 0.799 | 37.996 |
| normal_vector_observed | H/C | 5.116 | 4.510 | 11.064 | 42.939 |
| tensor_matvec | L/L | 0.285 | 0.284 | 0.486 | 1.478 |
| tensor_checkpoint_long | H/L | 36.821 | 14.029 | 19.866 | 56.918 |
| tensor_retained_long | L/L | 36.880 | 22.430 | 28.142 | 68.768 |

| Stable default control | µs/call |
| --- | --- |
| model_pipeline/scalar/128/unconstrained_value_gradient_reused | 4.733 [4.359–4.756] |
| model_pipeline/batch/128/unconstrained_value_gradient_reused | 2.772 [2.709–2.924] |
| funnel_pipeline/scalar/8/constrained_value_gradient_reused | 0.385 [0.384–0.398] |
| funnel_pipeline/batch/8/constrained_value_gradient_reused | 0.389 [0.385–0.393] |
| target_boundary/64/checked_density_only | 0.107 [0.106–0.109] |
| target_boundary/64/checked_gradient | 0.092 [0.092–0.109] |
| target_boundary/64/raw_gradient | 0.055 [0.054–0.057] |
| tensor_broadcast/128/value_vjp | 16.262 [16.239–16.278] |

N/C/L/H inventory retains unrun/unsupported cells rather than guessing routes.
Native curvature and block source remain unsupported. Scalar schools lacks a
Cauchy source rule. Scalar Radon rendering is eligible but its compiled
derivatives remain unqualified/unrun. Covariance H first-order is qualified by the
separate cohort; covariance higher derivatives are not exercised by that CLI
experiment. Value-only model timings are not inferred from fused first-order
calls.

### Memory and build scope

The process RSS below covers the entire benchmark executable, including untimed
fixture/preflight grids; it cannot be assigned to an individual small model.
Private scalar/block target workspaces, source-target heap storage and compiler
reverse tapes are unexposed and remain unknown. Known input+gradient payloads are
80 B covariance, 160 B schools/NC and 1424 B pinned Radon. They are not total
memory.

| Whole process | Peak RSS MiB |
| --- | --- |
| stable-default/model_pipeline | 25.633 [21.785–26.145] |
| stable-default/tensor_tape | 205.441 [202.621–207.109] |
| stable-default/regression_execution | 47.637 [47.523–47.762] |
| enzyme-controls/native_compiled | 119.895 [119.598–120.219] |

The existing direct tensor 128 ownership counter reports plan 1574 B, workspace
36533 B (budget 36834 B), and zero first/steady allocations for its measured
calls. These owned-byte counters exclude compiler/provider state and RSS.

Build observations are one serial build per lane with toolchain-shared caches.
They are neither seven-run compile distributions nor clean compiler speed ratios.
Compiler peak RSS is separate from executable peak RSS. Runtime executable
compilation/rendering is covered by the repeated lifecycle cases above.

| Lane | Observed build elapsed | Build peak RSS MiB |
| --- | --- | --- |
| enzyme-controls | 4:18.18 | 2086.6 |
| nightly-ad-native | 2:17.80 | 630.2 |
| nightly-native | 2:25.50 | 622.7 |
| nightly-simd | 1:16.91 | 621.4 |
| stable-blas | 1:16.45 | 612.6 |
| stable-both | 1:30.76 | 695.6 |
| stable-default | 2:44.39 | 616.3 |
| stable-faer | 2:44.66 | 1076.8 |

### Explicit Scalar/Auto controls

The existing 1000×4 CPU grid repeats descriptor construction from shared prepared
data, excluding fresh input allocation/packing. All policies here are Compensated.
Normal retains its eligible sufficient-statistics path, so its Scalar/Auto row is
not a forced Gaussian row-kernel SIMD comparison. The SIMD plan reports
V3/AVX2/FMA, four lanes; Auto availability does not prove each internal operation
uses those lanes. These are separate rows from 1000×65 arithmetic controls. Times
are µs; ratios Scalar/Auto.

| Lane/law | Build scalar/Auto µs | Scalar µs/call | Auto µs/call | Paired Scalar/Auto |
| --- | --- | --- | --- | --- |
| stable-default/normal | 306.989/307.118 | 0.238 [0.238–0.239] | 0.238 [0.238–0.239] | 1.000 [1.000–1.000] |
| stable-default/logit | 0.695/0.691 | 46.708 [46.691–46.892] | 22.572 [22.544–22.631] | 2.069 [2.063–2.080] |
| stable-default/poisson | 52.984/52.975 | 30.431 [30.383–30.479] | 18.251 [17.922–18.270] | 1.667 [1.664–1.697] |
| nightly-native/normal | 305.959/306.042 | 0.239 [0.239–0.251] | 0.239 [0.238–0.251] | 1.001 [1.000–1.013] |
| nightly-native/logit | 0.692/0.693 | 46.404 [46.354–47.585] | 22.684 [22.667–22.776] | 2.044 [2.036–2.099] |
| nightly-native/poisson | 53.184/53.177 | 31.043 [30.684–31.249] | 18.003 [17.877–18.312] | 1.721 [1.695–1.742] |
| nightly-simd/normal | 307.573/307.093 | 0.238 [0.238–0.239] | 0.238 [0.238–0.239] | 1.000 [1.000–1.000] |
| nightly-simd/logit | 0.694/0.695 | 46.527 [46.495–46.915] | 22.862 [22.832–24.421] | 2.035 [1.904–2.055] |
| nightly-simd/poisson | 53.191/53.172 | 31.414 [31.297–31.778] | 18.139 [18.100–18.158] | 1.730 [1.725–1.756] |

### Interpretation and reproducibility

The main campaign contains 56 finished cells, 4396 Criterion estimates with 20
samples each, 6328 timed fixed-work phase rows / 15820 counted boundary calls and
1764 direct-storage records. Its 16240 route rows comprise 644 supported, 4270
not-run and 11326 unsupported repeated fixture/order/configuration cells; these
are not 16240 independent capabilities. The separate CPU campaign has 21 cells /
252 estimates. The final covariance cohort admits 28/28 process runs with seven
positive construction measurements per variant. Raw-output replay checks numerical
admission again instead of trusting saved pass flags. Each frozen study’s source
hashes, all 22 frozen benchmark binary hashes and the pinned Radon input hashes
match before/after evidence. The regenerated covariance module is byte-identical
across the timer change. The main/CPU snapshot precedes the CLI factory timer; the
updated CLI snapshot preserves the generated numerical code (SHA256
`d9d6a93deb6a3a70cabef54f826aad604b940758a6d86a2481d7755e6799d9b8`).

Block composition cuts full construction sharply—about 19.7× for pinned Radon— but
reused evaluation regresses in all four named stable-native fixtures. School block
calls take about twice as long; Radon block reuse is about 10% slower. The
separate compensated SIMD controls improve logit/Poisson row throughput, while
Normal's eligible compressed path is unchanged. These gains do not transfer
automatically to composed observation maps. Standard arithmetic and accelerated
GEMM can be faster under their explicit contracts; passing these inputs does not
remove the cancellation/accuracy distinction. Enabling Faer/BLAS availability does
not broadly accelerate the default block models. The AD-enabled native compiler
control is slower for covariance and Radon, with a Radon paired ratio of 0.679
relative to the matching nightly-native control; no profiling here attributes that
difference to a particular compiler pass.

This task adds measurement/admission coverage rather than production performance
changes. The study supports neither a universal block speedup nor an inference
speedup from pointwise timings. Multichain posterior/ESS efficiency remains a
separate gate. Build observations have no repeated clean-compile uncertainty;
private model/compiler memory is unknown; conditional reporting, data parsing,
serialization and source-target setup use their stated scopes rather than a single
end-to-end total. All failed/unavailable/unrun cells remain in the raw records,
including the initial Criterion test-mode mistake and unsupported source
providers/orders. No existing prior, reduction policy or unsupported source rule
was replaced to fill a cell.

Completed raw study roots on ELTE, each with scripts, frozen inputs/build
receipts, raw result JSON/logs and replay outputs:

- Preparation/test-mode attempt: `/users/usumusu/alea-i44-cwwVtIgq`, job 2781887.
- Corrected seven-pair timing: `/users/usumusu/alea-i44-timing-ndrQtjq2`, job 2781896;
  `evidence.tar.gz` SHA256 `be54d8d85b63d36c1cfd35a19bb3ff18f8c59c284e61bbbd986c3e8fa5edb609`.
- Explicit CPU controls: `/users/usumusu/alea-i44-isa-iVbOYU2W`, job 2781897;
  `evidence.tar.gz` SHA256 `2183f255022fca57f854476942efe74c89f296e82c7400bdda957abed83b655a`.
- Original covariance cohort: `/users/usumusu/alea-i44-source-Xewql2U9`, job 2781895;
  `evidence.tar.gz` SHA256 `4fd73c9b3bc0f1cb987499218afd3f7633083158cefbc8764074b3e40aef3676`.
- Covariance with measured source setup: `/users/usumusu/alea-i44-source-factory-hsfNtLwY`, job 2781898;
  `evidence.tar.gz` SHA256 `868ef721896cfd981fe9b7203f8ceb77bb48ab1a90df9e523f767a0d7751e27a`.

Pinned Radon data/model SHA256:
`99c607bdd4e11d2693b27f8fd71d7e79adb6426caf013993b267983fdf598d4f` /
`db097f50d6f6158ceff725f520179ae74f2debe1ced50d899cac92787fd62861`.
The existing corpus index remains authoritative; no new posterior corpus is added.
The original source capsules and immutable executable copies remain at the study
roots. The corrected timing and CPU studies reuse the preparation binaries and
have no concurrent builds; dependencies keep all campaigns serial on the same
node. Reproduce native untimed admission with `cargo bench --manifest-path
crates/Cargo.toml -p alea-core --bench model_pipeline -- --test`; actual timings
require an allocated compute node. The existing `three_engine.py cargo-report`
replays Criterion/route/work records, and `matrix-report` re-admits the sealed
covariance process outputs. Ratios above are paired by repetition; the generic
cargo report's all-cell comparisons are not paired uncertainty estimates.

## 17. Four-chain posterior/efficiency qualification (2026-11-18)

The predeclared four-model study qualifies covariance and pinned Radon comparisons.
Centered schools fails for both engines; noncentered schools improves mixing but
still fails the zero-divergence policy in several repetitions. Both school
comparisons remain unranked. These are bounded current-workflow measurements,
with no production sampler or evaluator change and no universal performance claim.

The existing sampling runner used four chains, 1000 adaptive warmup and 1000
retained draws per chain, seven independent predeclared seed sets 17–23, zero free
initials, diagonal adaptation, target acceptance 0.8 and depth 10. Native uses the
block constructors qualified in I4.3b. Native seeds repeat the selected byte across
32 seed bytes and use streams 0–3; Stan uses integer seeds 17–23 and chain IDs 1–4.
These are independent engine trajectories, not common-random-number replay.

All builds and fresh 19 shared-coordinate density/gradient checks preceded timing
on exclusive ELTE cn18-42, virtualized Xeon Gold 6240, CPU 0/one runtime thread.
Rust 1.95.0 and GCC 8.3.0/CmdStan 2.40.0 use native CPU specialization; source/flags,
data, binaries and raw outputs are retained. The pinned Radon Stan body has only
an added posterior-profile wrapper; its original source is preserved separately.
No prior, coordinate chart or likelihood was replaced to enter the comparison.

Admission is R-hat<1.01, bulk/tail ESS>=400 in four chains, zero retained
divergences and depth saturations, and each chain E-BFMI>=0.3. The energy threshold
is a comparison warning gate, not a convergence theorem. Production rank/folded
diagnostics and estimand mean MCSE are reused. Scientific quantities include
original-coordinate means and second moments, recovered school theta/Radon alpha,
positive scales, and covariance entries/correlation. Second-moment MCSE is not
variance MCSE. Each native/reference estimand must agree within six combined
MCSEs, including the finite reference-chain uncertainty. Tail ESS checks local
mixing; reference quantile accuracy and SBC are not inferred from it.

### Qualified comparisons

Efficiency is the minimum bulk ESS across the declared scientific quantities
per full workflow wall second. All seven planned pairs must pass before ranking.
Ratios are native/Stan ESS/sec: above 1 favors native. Intervals are exploratory
95% percentile bootstrap intervals over seven independent paired repetitions
(10000 resamples), not guaranteed coverage for ESS estimators or proof of global
posterior exploration. Fresh Stan chains are a comparison reference with their
own uncertainty, not exact or long-run PosteriorDB ground truth.

| Case | Qualified pairs | Native workflow s | Stan workflow s | Paired ESS/sec ratio | 95% bootstrap interval |
| --- | --- | --- | --- | --- | --- |
| covariance | 7/7 | 0.376 [0.370–0.385] | 0.626 [0.612–0.633] | 1.530 [1.269–1.866] | [1.349–1.686] |
| schools | 0/7 | 0.625 [0.578–0.668] | 0.624 [0.519–0.699] | withheld | withheld |
| schools_nc | 2/7 | 0.338 [0.329–0.351] | 0.350 [0.338–0.367] | withheld | withheld |
| radon | 7/7 | 12.454 [12.354–12.542] | 20.539 [20.078–20.817] | 1.588 [1.386–2.086] | [1.459–1.723] |

Both workflows include diagnostics and output in the qualified denominator.
Native runs four chains in one process, collects draws and then writes CSV; Stan
runs four serial processes with streamed CSV, followed by the shared diagnostic
reader. Reference diagnostic-reader wall time is explicitly included. The native
legacy `seconds` field ends before diagnostics; it is not the qualified denominator.
The scopes and process schedules differ and are not an equivalent-adaptation or
pure kernel comparison. These results therefore do not contradict §16's slower
block hot evaluation or establish a block-versus-scalar inference speedup.

| Case | Engine | Scientific minimum ESS/workflow second |
| --- | --- | --- |
| covariance | alea | 6538.971 [5822.118–8062.728] |
| covariance | stan | 4320.613 [4198.887–4811.520] |
| radon | alea | 131.384 [121.103–133.436] |
| radon | stan | 82.575 [63.885–90.067] |

### Failures and mixing evidence

All 56 four-chain attempts completed (224 logical chains); there are no unstarted
or computationally incomplete sampling cells. Diagnostic completion is distinct
from qualification. Values below show extrema across all seven repetitions,
including failed ones. No failed reference is dropped to produce a favorable ratio.

| Case | Engine | Retained divergences, total | Max depth saturations/run | Min E-BFMI | Max scientific R-hat | Min scientific bulk/tail ESS |
| --- | --- | --- | --- | --- | --- | --- |
| covariance | alea | 0 | 0 | 0.910 | 1.0044 | 2206.1/2013.4 |
| covariance | stan | 0 | 0 | 0.910 | 1.0039 | 2651.0/2240.2 |
| schools | alea | 499 | 0 | 0.168 | 1.1078 | 28.6/9.3 |
| schools | stan | 1348 | 0 | 0.131 | 1.2935 | 13.2/5.6 |
| schools_nc | alea | 7 | 0 | 0.880 | 1.0036 | 1994.9/1438.3 |
| schools_nc | stan | 9 | 0 | 0.894 | 1.0036 | 2121.7/1338.4 |
| radon | alea | 0 | 0 | 0.624 | 1.0057 | 1498.9/1860.3 |
| radon | stan | 0 | 0 | 0.632 | 1.0078 | 1329.9/1964.1 |

Centered schools has poor mixing, low energy mixing and many retained
divergences in both engines. Noncentered schools clears the reported mixing/energy
gates but has divergences in four of seven repetitions per engine; only two paired
repetitions pass. Its improvement does not authorize weakening the zero-divergence
policy or publishing a favorable subset's efficiency ratio. Numerical trial
rejections/warnings, including Radon scale-underflow proposals, remain in native
error counters and Stan stderr. They are distinct from retained divergences and
do not alone establish a model or sampler defect.

Illustrative seed 17 scientific estimates below use the same original-coordinate
quantities. Each ±value is an estimated mean MCSE, not a posterior credible
interval. Admission uses all declared quantities in all seven pairs, not only
these examples.

| Case | Quantity | Native mean ±MCSE | Reference mean ±MCSE | Difference/combined MCSE |
| --- | --- | --- | --- | --- |
| covariance | mu0 | -0.045859 ±0.001577 | -0.045501 ±0.001562 | 0.162 |
| covariance | correlation | 0.529309 ±0.002154 | 0.530112 ±0.001948 | 0.276 |
| radon | beta | -0.661638 ±0.000785 | -0.663132 ±0.000794 | 1.339 |
| radon | sigma_alpha | 0.320267 ±0.001095 | 0.320229 ±0.001032 | 0.025 |

### Counted work, warmup and retained phases

These are actual counters, not gradients inferred from tree depth. Native counts
are fused full-target calls; Stan counts are posterior-profile autodiff calls.
Stan profile time can include primal work and excludes chart/Jacobian work outside
the region. These counter scopes are different; dividing them does not establish
an equal-work kernel speedup. Timers/instrumentation remain part of the run cost.

| Case | Engine | Actual calls, median [range] | Retained integrator steps, median |
| --- | --- | --- | --- |
| covariance | alea | 53394 [52626–55256] | 25478 |
| covariance | stan | 62234 [60631–63235] | 26400 |
| schools | alea | 166229 [150415–176644] | 70067 |
| schools | stan | 185164 [148723–215201] | 72101 |
| schools_nc | alea | 75108 [72106–78437] | 36052 |
| schools_nc | stan | 80928 [77502–86391] | 34498 |
| radon | alea | 172028 [170987–173683] | 59712 |
| radon | stan | 179640 [174898–181727] | 59920 |

The existing logical-chain phase control ran streams 0–3 separately for seed 17.
All 16 final points match the primary native runs bit-for-bit, and each four-chain
phase-call sum equals its primary total. This admits attribution for these
controls. They are separate single observations, not seven-run phase-speed claims;
their times are not subtracted from primary workflow times. Initialization is
counted with warmup. Statistically ineligible school controls remain cost-only.

| Seed17 native, four streams | Warmup calls | Retained calls | Warmup s | Retained s |
| --- | --- | --- | --- | --- |
| covariance | 28238 | 26040 | 0.173600 | 0.159104 |
| schools | 97879 | 78361 | 0.322140 | 0.257948 |
| schools_nc | 39901 | 35664 | 0.141762 | 0.125850 |
| radon | 113971 | 59712 | 7.826015 | 4.092895 |

Stan phase footers sum four chains and include their CSV/output scope. Separate
Stan target-call counts by phase are unexposed and remain unknown.

| Stan four-chain sum | Warmup seconds, median [range] | Retained seconds, median [range] |
| --- | --- | --- |
| covariance | 0.271 [0.267–0.275] | 0.297 [0.286–0.302] |
| schools | 0.267 [0.245–0.302] | 0.254 [0.199–0.343] |
| schools_nc | 0.118 [0.112–0.120] | 0.151 [0.140–0.164] |
| radon | 12.612 [12.112–12.899] | 7.387 [7.364–7.463] |

### Output attribution and limits

CSV collection, memory collection and discarded-position runs use the same seed 17
and native binary. All eight memory/discard controls preserve their primary CSV
run's per-chain trajectory hashes and evaluation counts. Discard disables retained
position collection but retains energy evidence and the runtime's bounded
diagnostic window; it is not allocation-free. It has no scientific-summary rank.

| Seed17 native | CSV session s | Memory session s | Discard session s | Post-collection CSV writing s |
| --- | --- | --- | --- | --- |
| covariance | 0.339461 | 0.338286 | 0.340486 | 0.006934 |
| schools | 0.596080 | 0.590032 | 0.593622 | 0.012701 |
| schools_nc | 0.273777 | 0.273849 | 0.272103 | 0.012807 |
| radon | 11.928335 | 11.989759 | 11.995189 | 0.099824 |

These are single matched attribution controls with no repeated output-cost
uncertainty; their small session differences are not ranked. CSV writing also has
seven primary observations in raw records. Native Radon post-collection writing
is about 0.10s against about 12s of session work in this control, so it does not
justify introducing a new output/concurrency system here. Stan streaming output
cannot be isolated by subtracting the native CSV timer.

Passing these gates is bounded evidence, not proof of convergence, broad posterior
accuracy or calibration. There is no source-sampling or higher-derivative support
promotion. No seeds, priors, adaptation targets, rejection policies or thresholds
were changed after outcomes. I4.6's reviewed generative/SBC protocol remains a
separate prerequisite; I5.1 is the next ready nonconditional task.

### Reproduction and preserved failures

Preparation job 2781899 failed before timing because an explicit block-selection
environment variable leaked into unrelated mean/regression unit tests. Its six
Shape failures and environment/logs are preserved. Corrected job 2781900 unsets
that selection during tests and enables it for the four named models afterward;
it completed 0:0 in 15m23s on cn18-42. All builds/preflights precede timing, and
source plus five measured-binary hashes match before/after checks. Local replay
matches the compute-node report exactly and re-admits raw process outputs/CSV,
phase/profile files and timing witnesses rather than cached pass/metrics fields.

The progress-only sampling log contains 396 NUL bytes. It is preserved unchanged;
its cause is unresolved, and it is not used for qualification, counts or timing.
All 56 canonical records and their separate raw outputs/timing witnesses are
complete, hash-verified and replayed independently. This logging limitation is
not silently repaired or used to erase any attempt.

Raw study root: `/users/usumusu/alea-i45-fBRjkf0v`; corrected attempt in `attempt2`.
The frozen source capsule retains the original Radon model plus its temporary
profile-only wrapper, pinned data, source manifest, exact build commands/flags,
protocol and scripts. Raw archives and compiled references remain there and in
the retained local evidence copies. Archive SHA256:

- `attempt2/evidence.tar.gz`: `24a2e1adaece13b1ffb1d9a4673eba8378130009cce25ca22b794836c7d46662`.
- `attempt2/stan-binaries.tar.gz`: `5eed74fa618353ad2dacab4db2b9b736d46760f5b2c86f0b5d3ae0ac89575b27`.
- `failed-preparation.tar.gz`: `bf4a7d2b45fd9c0a7f4a12edd424877a44a69e65ee89fd1c2dad35efdaa70907`.
- `attempt2/preflight-and-progress.tar.gz`: `9e4c2b602c0fb3a5d73caf58574958f5025cacd3d1c0b2d475eb694ef3e97c6c`.

Use the existing `bench.py --qualify` interface for a new allocated-node study,
with explicit pinned binaries/data and fresh output paths. Re-admit an archived
sampling directory with `replay.py stan <directory> <new-report.json>`; it does
not overwrite historical reports. The source artifact has no Git metadata;
the complete frozen source/data hashes, not a fabricated clean commit identity,
bind this experiment to the dirty source snapshot.

## 18. Residual-deficit profiling gate (I5.1, November 18)

I5.1 is incomplete. The I4.4 hot-reuse regressions motivate profiling, but do not
identify a storage-lifetime or reverse-dispatch optimization. No production change
or negative result closing I5 follows from this attempt.

The existing saved-binary `three_engine.py profile` command was run on exclusive
ELTE allocation 2782298, hpc2019 node cn18-42, comparing covariance scalar/block
representations in the same I4.5 native executable, with 100,000 requested
calls per invocation, CPU 0 and one library thread. Both `perf stat` and `perf
record` failed before starting the target: `/usr/bin/perf` delegates to missing
`perf_4.19` (`linux-perf-4.19 is not installed`). The allocation exited 1; its
source-script and binary SHA-256 checks passed afterward. The recorded child
wall times measure tool failure, not model execution. No new performance was
measured. No sysctl, privilege, or system-package changes were made.

The existing accounting boundaries remain relevant to the next attempt:

- Native tensor `execute::workspace_bytes` counts the full primal slots,
  separately planned adjoints, staged input-gradient storage and frame/cache
  metadata, recursively including region frames. Reverse reads the same admitted
  primals; they are not an additional independent residual allocation.
- Explicit source reverse has a separate residual layout and a complete selected
  forward/admission frame. Its costs cannot be assigned to the native executable
  merely because both implement the same objective.
- Regular broadcast index maps are compact metadata; explicit gathers retain
  their index payload. Existing tensor broadcast bench receipts distinguish plan
  ownership, workspace, frozen payload, source text and first/steady allocations.
  Those totals do not provide a disjoint primal/residual/index/admission breakdown.
- Native `evaluate` includes validation, selected forward execution and staged
  output copying. `value_and_vjp` additionally validates seeds, runs reverse and
  copies staged gradients. Subtracting their wall times would not independently
  measure reverse, copies or checks. These contributions remain unattributed.

Resume with a working compute-node profiler and the same saved comparison
controls; retain this failed attempt. Select a bounded problem only after
separating these costs. No storage reuse, fusion, admission-check removal,
provider change, generated-AD qualification or inference speed claim is made.

A subsequent tool repair resolved the missing executable without system-package
changes. Debian's archived `linux-perf-4.19` 4.19.316-1 and `libopencsd0` 0.10.1-1
were extracted into `/users/usumusu/alea-perf-4.19/extracted`. Both downloaded
packages matched the SHA-256 digests in ELTE's existing APT package index:
`3ac63de18c8dd9fa714cf0ef74447d0dfa3581539f3684876202990722a22b01`
and `eb9663c560804dcb5e6815516ee0aebb4b91588e2a4e481c166ce64caf684077`,
respectively. With the private library directory on `LD_LIBRARY_PATH`, the
executable reports `perf version 4.19.316`; no shared libraries remain missing.

Job 2782299 reran the same saved comparison through the existing profiler on
exclusive cn18-42. All four hardware-counter/sample invocations now reach
`perf_event_open`, but fail with error 13, Permission denied. An additional
`task-clock:u` software-event probe also fails. The node has
`kernel.perf_event_paranoid=3`; installing the tool does not grant event access.
Binary/script hashes still verify. This attempt supplies no model performance
or hotspot evidence. ELTE administration must permit user-process events or
provide an approved privileged profiler command before this route can proceed.
The private executable can be passed to the existing `--perf-command` option;
its binary SHA-256 is
`57052fceaa20c3e7a265e74541ef2343539eb9b8cbd6d7b93bfe6571d00f054d`.

### Authorized local diagnostic and selected follow-up

The maintainer subsequently deferred ELTE administrator access and explicitly
permitted local profiling. This exception applies to this diagnostic; it does
not pool laptop results with ELTE measurements or change benchmark defaults.
I5.1's bounded dispatch-selection condition is met by the following local
profile. No storage-lifetime change was justified; I5.2 is deferred. I5.3 can
investigate the specific reverse traversal below, with any performance admission
still subject to its separate controls and I5.4.

The current default native benchmark was rebuilt using stable Rust 1.98.1,
release fat LTO, one codegen unit and debug information. No production source was
changed. Nix supplied perf 6.19.11 from the repository's locked nixpkgs revision
`68d8aa3d661f0e6bd5862291b5bb263b2a6595c9`. The existing profiler's
`--local-profile` option ran three alternating scalar/block pairs, one million
covariance evaluations per cell, CPU 0, one library thread, user-only counters
and 997 Hz cycle samples with DWARF stacks. Unprivileged counting succeeded but
recording failed with EACCES; the successful cohort used local `sudo -n perf`
without changing kernel policy. All six cells' counter, record and report
commands passed; reports show zero lost samples. Counter/sample numerical
outputs match exactly within each representation; scalar/block final values,
gradients and aggregate checksums agree within relative tolerance 1e-9.
The native correctness suite also passed all 17 tests, including independent
score/finite-difference, admission and failure/recovery controls.

Local retired instructions per million evaluations were 37.513 billion for
scalar (observed 37.512–37.514 billion) and 40.585 billion for block
(40.585250–40.585252 billion). This establishes extra executed instructions in
this local control, not an ELTE cycle or wall-time speed estimate. Source/data
and executable hashes were checked unchanged after profiling. Raw perf files,
commands, numerical outputs and the 763-file source snapshot are retained
locally; this is not a publicly reproduced campaign.

The following are flat **self-sample** percentages of the complete block
process, with median and observed range across three runs. They are not
independently timed phases. Inlined checks/arithmetic remain charged to their
containing symbol; libc operations are not uniquely attributable to a caller.
Nested reverse-index symbols are a separate category, not included in the tensor
reverse row. Ranges are descriptive, not confidence intervals.

| Work visible in symbols | Median % | Observed range % |
| --- | ---: | ---: |
| Observation forward, including law construction, scores and local checks | 27.27 | 27.14–27.57 |
| Observation cached-score reverse | 5.64 | 5.45–5.69 |
| Assembly region-call forward boundary | 6.44 | 5.82–6.46 |
| Assembly region-call reverse boundary | 5.95 | 5.77–6.15 |
| Tensor forward interpreter | 7.29 | 6.53–7.43 |
| Tensor reverse interpreter and reverse-node symbols | 9.61 | 9.21–9.67 |
| Binary reverse generic index traversal | 6.16 | 5.71–6.66 |
| Outlined copy routines | 4.46 | 4.13–4.85 |
| Outlined clearing routines | 2.25 | 2.01–2.54 |
| Outlined finite-check routines | 4.90 | 4.60–4.95 |

**Selected bounded problem:** `tensor/kernels.rs::reverse_with` sends
Scalar/Identity and Identity/Scalar maps with **both operands active** through
`for_each_index`; existing contiguous special cases cover one-active broadcast
operands. The covariance predictor contains the relevant active vector/scalar
multiply and add. Their generic add/multiply traversal symbols together account
for 5.71–6.66% of block samples. Source-line reporting also locates samples in the
generic reverse loop. This justifies testing an ordered contiguous reverse loop
for these two maps, not replacing the scheduler or promising a 6% total speedup.
The retained scalar covariance model is the same-posterior numerical comparison
control; the existing generic reverse implementation is the future implementation
control. Preserve per-element operand order, zero-seed skipping, inactive
admission, shared destinations and generic-map fallbacks. No such optimization
is implemented in this report.

Memory attribution is deliberately narrower than total ownership: the following
are **code-derived f64 payloads**, not new peak-allocation measurements. At n=32,
each of the two selected binary regions has scalar+vector inputs (33 elements)
and a vector output (32). Shapes/maps and metadata remain additional costs.

| Category | Payload and boundary |
| --- | --- |
| Primal frame | 65 × 8 = 520 bytes per selected binary region; full admitted inputs/output retained |
| Reverse residuals | Binary reverse borrows those primals, no separate residual cache; the two observed-Normal blocks separately cache 2 × 32 × 2 × 8 = 1,024 bytes of parameter scores |
| Index payload | Scalar/Identity maps need no heap index table; inline map tags, shape axes and graph metadata still exist; no gather in the selected two regions |
| Admission | Native selected-forward checks read existing primal/score storage; no separate full admission f64 frame is introduced. Observation parameter-point scratch is 2 × 2 × 8 = 32 bytes. Source residual/admission layouts are separate and not measured here |
| Outer region staging | Per selected region: arguments 264, value 256, scores 264, adjoint 256 bytes, totaling 1,040; tensor adjoints/gradient and metadata are additional |

These categories must not be added as a total peak: residuals can overlap
primals, and staging, tensor adjoints, binding caches, allocator overhead and
other model regions have separate ownership. Existing I2.4/I4.4 workspace and
allocation receipts remain the authority for their measured totals. No
compiler/provider/stack/RSS bound, storage saving or removal of admission checks
follows from this dispatch selection. Full forward/reverse/copy/check timings
remain unseparated where the compiler inlines them; the symbol table exposes
that limitation rather than subtracting mismatched phase measurements.


## 19. Scoped reverse specialization (I5.3, November 18)

The I5.3 candidate reused contiguous loops for the I5.1-selected Scalar/Identity
and Identity/Scalar reverse cases when both operands were active and their
producer adjoint starts differed. I5.4 subsequently rejected this candidate and
restored the original kernel; see §20. Each destination retains its original row accumulation
order and zero-seed skipping; a scalar score starts with its existing adjoint.
Length-one aliases keep the original interleaved generic path. This changes only
native reverse dispatch: no scheduler, storage layout, source emitter, admission
check, provider policy, public API or derivative capability is replaced.

The expanded scalar-bit oracle passed on the original kernel before the change
and on the candidate (including the optimized scalar test run): all four
arithmetic operations, both broadcast orientations,
inactive/alias controls, extreme finite values, cancellation and signed-zero
seeds. An independent finite-difference/shared-consumer case checks preexisting
scores and scalar/length-one reshape aliases. Existing fused/unfused controls
confirm large shared-DAG parity and retain the untiled active scalar reduction.
Core unit/integration/property checks passed 159 tests; combined Faer/OpenBLAS
tensor/checkpoint checks passed 31; native model checks passed 17; production
observed-law checks passed 11. Actual pinned-toolchain compiled source/replay
checks passed 13, including independent derivatives and failure/recovery. These
are regression checks for existing capabilities, not broader generated-AD support.
Default core-library and combined-provider test Clippy checks and scoped format
checks passed.

Performance is **not measured** for this candidate. I5.4 owns matched before/after
cold/steady controls and total-cost admission; the local I5.1 hotspot fraction is
not a candidate speed estimate. ELTE SSH was unreachable during this task; no
local timings were substituted. Matched 763-file baseline/candidate snapshots
retain identical new tests, harnesses and data, differing only in the kernel
source, for the later allocated-node comparison. There is no storage or inference
speed claim, and no previously qualified source/measurement artifact was changed.


## 20. Paired total-cost admission: candidate rejected (I5.4)

The November 18 ELTE cohort rejects the I5.3 both-active scalar/vector two-pass
specialization. Review completed November 19. Stable covariance improved, but
five unchanged controls met the predeclared repeatable >5% regression rule.
Only the candidate kernel delta was reverted; the expanded correctness tests
remain. Production therefore retains the original generic both-active reverse
path and existing one-active contiguous shortcuts. I5.2 remains deferred: no
storage-lifetime deficit or memory-only tradeoff justified a change.

### Matched experiment and admission

Job 2783274 completed 0:0 in 1h29m08s on exclusive hpc2019 node cn18-42, a KVM
Xeon Gold 6240 host. Seven alternating baseline/candidate pairs used CPU 0,
one library thread, identical data and shared benchmark code. Both revisions
were built serially at one canonical source path, with only `kernels.rs` changed;
all builds and numerical preflights finished before timing. Copied executables
were frozen and verified before and after timing; no running source was edited.

Stable lanes used Rust 1.95.0; the pinned native/SIMD/AD lanes used
nightly-2026-06-23 (4429659e4), with native and AD-enabled flags distinguished.
All lanes used `-Cembed-bitcode=yes -Ctarget-cpu=native`; compiler-AD lanes
add `-Zautodiff=Enable`. Stable default/Faer/OpenBLAS/both, plain nightly,
nightly portable-SIMD, nightly AD-enabled native, and pinned source controls
were retained. Feature availability is not evidence of provider selection:
compensated Auto remains the reference; explicit Standard Faer/BLAS matrix
cases keep their different arithmetic guarantees. There is no cross-provider
or cross-accuracy speed ranking here.

The existing Criterion benches used 20 samples, 0.1s warmup and 0.3s requested
measurement. Stable default covered broad lifecycle and small/large controls;
other lanes used focused steady/provider controls plus broadcast cold stages.
The existing tensor-storage kernel suite supplied additional scalar, matvec,
lazy branch, shared DAG, retained/checkpointed scan and vector/reuse controls.
No timing runner, kernel formula, distribution or sampler was added.

All 112 planned cells finished: 5,124 Criterion estimates, 12,656 first/steady
work records describing 31,640 actual public fused calls, and 3,528 broadcast
storage records. All 23 kernel-suite cases completed seven compatible pairs
(322 child attempts); no failed, incompatible or unstarted pair was discarded.
Recorded work/query/data/coordinate/algorithm/route identities match across
revisions. Full-ID completeness was checked against the untimed fixture list,
resolving Criterion's abbreviated display names uniquely. Both revisions passed
159 core, 33 distribution and 47 actual pinned source/AD independent tests.
The complete native benchmark preflight also exercised 110 model, 261 tensor
and 12 regression cases in each applicable lane.

Ratios below are baseline/candidate, so values above one favor the candidate.
Intervals are exploratory 95% paired-repetition percentile bootstrap intervals
(10,000 resamples, fixed seed 5401); they do not establish simultaneous coverage
or independent-host reproducibility. Before timing, the >5% gate was fixed as
at least six of seven ratios below 1/1.05, median below that threshold, and
bootstrap upper endpoint below it. Cases failing the median threshold but not
that repeatability condition remain inconclusive, not regression-free.

### Primary stable-default block results

| Model | Baseline median µs | Candidate median µs | Paired ratio [interval] |
| --- | ---: | ---: | --- |
| covariance | 5.600 | 5.395 | 1.039 [1.028, 1.059] |
| schools | 2.819 | 2.816 | 1.001 [0.989, 1.028] |
| schools_nc | 3.046 | 2.905 | 1.051 [1.017, 1.069] |
| radon | 68.501 | 68.732 | 0.995 [0.984, 0.999] |

Covariance's median paired throughput gain is 3.9%, and noncentered schools'
is 5.1%. These are candidate measurements, not shipped improvements. Centered
schools is inconclusive; Radon has no favorable candidate gain. The original
scalar-versus-block abstraction deficit is not solved by this experiment.

### Blocking unchanged-control regressions

Absolute columns are medians of each arm; paired ratios are medians of the
within-repetition ratios, which need not equal a ratio of those arm medians.

| Lane / control | Baseline median µs/call | Candidate median µs/call | Paired ratio [interval] | Pairs beyond 5% |
| --- | ---: | ---: | --- | ---: |
| nightly-native / scalar Radon | 65.206 | 75.906 | 0.869 [0.845, 0.911] | 7/7 |
| nightly-simd / scalar Radon | 60.311 | 65.027 | 0.896 [0.849, 0.949] | 6/7 |
| nightly-simd / 8×8 compensated GEMM | 3.173 | 3.494 | 0.904 [0.900, 0.910] | 7/7 |
| enzyme-controls / 129-row Normal compiled first order (H) | 0.227 | 0.239 | 0.949 [0.949, 0.951] | 6/7 |
| stable default / scalar tensor, n=1000 | 21.020 | 22.150 | 0.950 [0.949, 0.951] | 7/7 |

The matching kernel scalar control also fails after including construction:
baseline/candidate total-cost ratio 0.950 [0.949, 0.951] at K=10,000 requested
steady calls. Its paired candidate latency increase is about 5.3%; the nightly
GEMM increase is about 10.7%. The scalar Radon and compiled Normal paths are
controls, not demonstrations that their arithmetic executed the new branch.
Whole-binary codegen/layout effects are possible; the mechanism was not isolated.
No claim is made that all measured slowdowns are direct index-loop overhead.

Across all lane/case pairs, 122 Criterion cases showed an exploratory improvement,
233 were inconclusive, seven had an inconclusive >5% median regression and four
met the blocking gate. The kernel suite had 13 improvements, eight inconclusive
cases, one inconclusive >5% median regression and one blocking regression. These
counts are not a weighted global score or evidence that unflagged cases cannot
regress. Favorable cases do not override the declared unchanged-control gate.

### Cold, first-call and total-cost scopes

The stable-default lifecycle medians below keep nested scopes separate.
`construct_compile_context_and_drop` already contains construction, compilation,
context creation and destruction; do not add bind/compile/context benchmarks
to it as though they were independent stages. `first_value_gradient` excludes
its batched setup but includes destruction, as in the existing I4.4 protocol.
Source-generation and native/compiler setup costs remain distinct; serial build
resource logs are not matched clean-build performance estimates.

| Block model | Construct/compile/context/drop µs, before → candidate | First-call benchmark µs, before → candidate | Five-state fixed-work batch µs, before → candidate |
| --- | ---: | ---: | ---: |
| covariance | 28.517 → 28.248 | 14.273 → 14.053 | 79.090 → 84.596 |
| schools | 24.782 → 24.390 | 10.069 → 10.000 | 34.027 → 41.106 |
| schools_nc | 26.858 → 26.581 | 10.844 → 10.732 | 33.821 → 33.084 |
| radon | 176.973 → 172.603 | 124.920 → 124.846 | 437.093 → 433.932 |

The five-state batch is exactly one first plus four steady fused calls; its
measured scope includes poisoning, admission and output copies and excludes
receipt logging. It uses changing states and is not interchangeable with
Criterion's fixed-state steady benchmark. Its variability does not support a
blanket cold/first-call improvement claim.

For every kernel case, total cost was also reconstructed per repetition as the
reported construction time plus the measured K-call loop. Construction includes
one initial admission evaluation, while K counts the requested steady calls;
process launch, serialization and compiler time are excluded. At retained scan
n=10000, K=100, paired steady/total ratios were 1.010/1.011; checkpointed scan
was 0.999/0.999. Shared-DAG n=100000, K=100 was 1.026/1.025, while vector n=100000,
K=10 was 0.999/0.999. The complete case-level distributions are retained rather
than treating these representative medians as universal wins. No break-even
claim is needed for a rejected candidate, and no nested lifecycle timings were
summed to manufacture an end-to-end figure.

### Memory, cleanup and evidence limits

All matched broadcast storage payloads are identical across revisions: plan
ownership, workspace budget/ownership, source lengths, frozen payload and
first/steady allocations. For regular 128×8 broadcasting, measured plan ownership
is 1,574 bytes and workspace ownership 36,533 bytes (budget 36,834); explicit
gather is 17,944/69,495 bytes (budget 69,814). Regular 65536×8 retains 1,574 bytes
of plan ownership and 18,350,773 workspace bytes. First and steady calls allocate
zero in these receipts. These are unchanged fixtures, not candidate storage
savings. Compiler/provider scratch and stack/process RSS are outside these
ownership totals. Saved per-process peak-RSS resource records include the entire
Criterion executable and all its setup; they are not per-fixture arena bounds.

I5.4 is complete by rejecting the candidate under its predeclared gate and
restoring the exact baseline kernel. The 19-added/2-removed I5.3 specialization
is absent from production; its tests remain useful regression coverage. No new
planner, runner, API, backend policy or source/derivative capability is promoted.
The pinned compiler remains optional, native stable remains first-class, and
source higher-order/checkpoint limits remain as previously qualified. This closes
the measured optimization branch negatively; it is not a general fastest-runtime,
posterior-efficiency or SBC claim.

The primary raw archive SHA-256 is
`a3ec4a5bd5c75e9cf936155ffeb2190c6b549e29833f7ff346d3cc1e3f39353e`.
The separately preserved 46 frozen executables, build metadata, generated source
and input snapshot archive has SHA-256
`a9090c245649558303a4a5a0431f43f1cdc5df0c68b065507be3b48fc5e3de1f`.
Both archives verify locally; every saved executable matches its recorded hash.
The existing kernel journal replays exactly, and the existing Criterion reader
reconstructs every planned case and work/storage receipt. This is private
retained evidence, not a claim of independent public reproduction or an offline
toolchain distribution. Remote study root:
`/users/usumusu/alea-i54-LWPIjuQc`. Job 2783275 only archived artifacts on cn18-14;
it supplies no additional timing host. An interrupted archive transfer was
resumed and hash-verified. A transient remote progress read showed NULs, but final
sealed build logs and all benchmark/preflight records contain none; the cause
of that progress display was not established.

## 21. I5.3 implementation and architecture investigation

The follow-up investigates the rejected candidate; it does not reinstate it.
Production retains the baseline kernel and the expanded I5.3 tests. Three
questions must be separated: mathematical correctness, direct broadcast-loop
cost, and effects of the change on other compiled paths.

### Correctness and what the original tests establish

No mathematical implementation defect was found in the two-pass candidate.
The scalar accumulator retains the original row order and initial adjoint;
vector destinations retain their original per-row additions. Zero seeds still
skip both destinations. Distinct producer starts denote disjoint live ranges
under the existing planner; reshape aliases share starts and retain the generic
interleaved fallback. The four existing arithmetic closures are pure. Required
forward values and local scores remain admitted before reverse execution.

The retained scalar-bit grid exercises all four binary operations, both active
broadcast orientations, inactive operands, aliases, cancellation and signed-zero
seeds. Related tests cover shared DAGs, gathers, lazy branches, failure/recovery
and independent finite differences. I5.4 additionally executed actual pinned
source/replay tests on both revisions. These are evidence for the I5.3 numerical
Done condition, not for profitability or compiler stability. I5.4's independent
performance rejection therefore remains valid.

### Exact executable evidence: integration matters

Disassembly of the frozen I5.4 executables shows substantial code growth despite
the small source delta. In `kernel_audit`, the enclosing tensor reverse function
grows from 74,300 to 90,907 bytes; one reverse-node interpreter grows from 23,336
to 28,137 bytes. The helper is forced inline and instantiated inside a shared,
already large interpreter. Two contiguous passes remove generic index traversal
but also revisit seeds and vector values; cheaper indexing alone does not imply
a cheaper complete executable.

The nightly SIMD GEMM control exposes an inlining boundary change. Its enclosing
`BoundProgram::reverse` has 5,849 instructions before and 373 after, with new
out-of-line calls to the enlarged reverse-node interpreters. The matrix forward
and reverse kernels themselves retain identical normalized instructions (291
and 567 respectively). This is a demonstrated compiler integration effect,
not proof that the matrix arithmetic became more expensive.

The scalar polynomial control uses scalar assembly `Multiply` operations and
never executes the changed tensor broadcast path. Its normalized scalar
`NativeModel::logp_grad` body has the same 2,295 instructions in both binaries,
but changes placement/alignment. The source Normal-129 control likewise has
identical normalized generated-gradient (392 instructions) and native
`NormalRegression::evaluate` (436 instructions) bodies at changed addresses.
Thus their measured regressions cannot be attributed to executing the extra
broadcast pass. Code placement and front-end interactions are plausible, but
cache misses, branch misses and cycle attribution were not established.
Normalization removes address-dependent displacements and LLVM suffixes; it
is not an independent proof that all referenced data are identical.

### No-admin CPU profiles and limits

A private gperftools 2.7 `libprofiler` successfully sampled the exact frozen
executables on allocated ELTE node cn18-42. It uses CPU-time timer signals rather
than restricted performance events, following the
[gperftools CPU profiler documentation](https://gperftools.github.io/gperftools/cpuprofile.html).
Job 2783277 completed all 12 baseline/candidate profiles. Sampling used 250 Hz
(4 ms period); Criterion cases requested eight seconds of profiling. Scalar
polynomial controls used 200,000 requested calls plus initial admission.

Profiles were symbolized with exact main executables and matching glibc 2.28
library/debug build IDs. Nearest exported symbols from stripped libraries were
initially misleading; those preliminary labels were replaced, and every graph
was regenerated. Collapsed stacks preserve full Rust names and inline frames;
checked totals equal the raw sample counts. Both whole-process and measured-loop
flamegraphs are retained. Criterion measured-loop graphs select stacks inside
`criterion::bencher::Bencher`; this matters for GEMM, where setup accounts for
about 14% of whole-process samples. Flat and inclusive percentages are different
scopes and must not be added together.

| Baseline measured loop | CPU attribution | Interpretation |
| --- | --- | --- |
| covariance block | Normal observation forward about 39% inclusive; `log` about 20% flat | Forward law preparation dominates the selected reverse indexing loop |
| covariance block | generic index traversal 6.1% flat | Removing only this work has a limited whole-call ceiling |
| scalar polynomial | scalar reverse 42%, scalar forward 30%, finite checks 11% flat | No execution of the changed broadcast kernel |
| compensated GEMM 8×8 | matrix forward 52%, reverse 37% flat | Matrix kernels dominate; wrapper/inlining effects still matter |
| source Normal-129 | regression evaluation 59% inclusive; `hypot` 23% flat | Qualified compressed regression and its numerical guards, not broadcast work |

These are statistical CPU-stack samples, not exact internal call counts or
per-function elapsed-time measurements. Inclusive frames overlap. Instrumented
profile durations are not new speed-ranking evidence. Hardware events remain
unavailable without administrator changes. An exact-binary Callgrind attempt
failed in its old AVX-512 instruction decoder before model execution; recompiling
for another ISA would no longer profile the same executable. The failed attempt
and the initial profiler/loader failures remain preserved.

### Architecture costs outside the I5.3 delta

The covariance block constructor's private `binary` helper creates a separate
lexical region for each binary operation. The current predictor therefore has
three model-to-region call boundaries plus a constant-data region. Two observed
law maps and their two sum regions bring the complete model to eight region
forward evaluations and seven dynamic region pullbacks per gradient call.
`assembly::blocks::Call` packs arguments, executes and admits the region, stages its
outputs and seeds, and copies/accumulates reverse scores. These boundaries are
useful for shared whole-value calls, but unnecessarily fine for this straight
three-operation predictor.

A second asymmetry is in law preparation. Generic `observations::forward`
constructs the Normal and requests its parameter scores for every row. The
existing Normal combined score kernel computes `ln(scale)` on each invocation.
For the two 32-row covariance observation maps, scales are shared scalars.
The scalar `NormalFactor` route instead explicitly declares `scalar_preparation`
and reuses its prepared log scale. This explains a real architectural opportunity
that a reverse-loop specialization cannot address. It predates I5.3 and belongs
to the block-observation/model-port increments, not to a newly incorrect density.

Any preparation improvement must retain law construction/error semantics,
parameter-only score admission, exclusive point-dependent workspaces and valid
recovery after failure. Generic user callbacks cannot be assumed pure and
silently memoized. Reuse an explicit preparation capability and the existing
law kernels; do not add a parallel Normal implementation, type-based cache or
second graph. The expensive `hypot` calls in compressed Normal regression serve
stable residual norms and conditioning screens. Deleting those safeguards or
replacing them with cancellation-prone Gram subtraction is not justified by
this profile.

### Counterfactual experiment design

A separate source snapshot tests three kernel variants: the restored baseline,
the original I5.3 inline candidate, and the same two-pass arithmetic in a
selectively outlined helper for both-active disjoint broadcasts only. Existing
one-active shortcuts and alias fallback are unchanged. The outlined form tests
the inlining interaction rather than replacing the arithmetic or broadly
outlining the interpreter. In this snapshot's SIMD tensor executable, reverse
wrapper sizes are 27,196 / 1,578 / 27,433 bytes respectively: selective outlining
restores the baseline inlining structure, although helper instantiations still
add code elsewhere. This observation alone establishes no speed improvement.

A cold construction switch also tests grouping covariance's three predictor
operations into one existing lexical region. Both architecture arms use exactly
the same executable. Arithmetic operations, laws, charts, observations and
per-node admission remain; no density, gradient or sampler is reimplemented.
The grouping removes three forward and two reverse region boundaries. Normal
scale preparation is deliberately unchanged, so any gain is not attributed to
an unimplemented law optimization.

The experiment freezes three source manifests, builds at one canonical path,
then completes all numerical preflights before timing. Each kernel variant passes
159 core tests and the 17 native CLI tests with grouping both off and on. Six
variant/architecture probe sets match full densities, gradients and errors on
seven valid/extreme-scale inputs. The probe agreement tolerance is 3e-10 relative
plus absolute; it is finite input evidence, not global bitwise equivalence.

The main study uses seven rotating three-arm repetitions, stable Rust 1.95 and
pinned nightly SIMD, identical native flags, one CPU/thread, the existing
Criterion benches and existing fixed-work kernel runner. Its grouped arms follow
the main arms within each repetition. A separate confirmation therefore repeats
the same-executable architecture comparison with alternating arm order. All
experiments use allocated cn18-42 compute resources; instrumentation runs after
uninstrumented timing. Results compare only within each new matched cohort,
not against old absolute I5.4 times. Exploratory 95% intervals use 10,000 paired
percentile-bootstrap resamples, seed 53019; they do not provide simultaneous or
independent-host coverage. These focused experiments do not replace the full
eight-lane I5.4 admission protocol.

### Results: outlining is insufficient; region grouping improves this fixture

All 56 main Criterion cells completed, yielding 1,442 estimates. Both ten-case
kernel comparisons completed seven pairs (280 compatible child attempts), and
raw journal replay exactly reconstructs their summaries. All 18 copied binaries
match their frozen hashes; the three 763-file source manifests differ only in
`kernels.rs`. The confirmation contributes another 14 cells and 182 estimates.
No failed timed cell was omitted.

Ratios are paired baseline/candidate throughput ratios; above one favors the
candidate. Arm latencies are separate medians, so their quotient need not equal
the median paired ratio.

| Main matched control | Baseline µs | Inline µs / ratio [interval] | Outlined µs / ratio [interval] |
| --- | ---: | --- | --- |
| stable covariance block | 5.587 | 5.350 / 1.046 [1.030, 1.056] | 5.367 / 1.048 [1.035, 1.055] |
| nightly SIMD compensated GEMM 8×8 | 3.186 | 3.512 / 0.908 [0.905, 0.911] | 3.530 / 0.901 [0.890, 0.905] |
| stable fixed-work scalar polynomial n=1000 | 21.049 / 21.061 | 22.160 / 0.949 [0.947, 0.951] | 22.179 / 0.950 [0.947, 0.951] |

The fixed-work baseline is taken from each runner's own paired cohort; the
inline and outlined runners' baseline medians are 21.049 and 21.061 µs
respectively (shown in the same order in the table). At n=1000, including
construction gives inline and outlined ratios 0.949 [0.947, 0.951] and 0.950 [0.947, 0.951]. Both retain the
scalar regression. Stable GEMM improves in these new candidate binaries, whereas
nightly SIMD GEMM regresses; scalar Radon also varies by compiler/variant. These
mixed controls are retained in full, not averaged into a favorable global score.

Selective outlining restores the reverse wrapper's inlining structure but
**does not fix the SIMD GEMM or scalar-polynomial regression**. Fresh SIMD matrix
leaf bodies are normalized-identical across all three variants (285 forward,
567 reverse instructions), yet their complete executable timings differ. The
observed inlining boundary change is therefore not a sufficient causal
explanation. Code placement, caller/control-path changes and microarchitectural
interactions remain unresolved without further isolation. Neither candidate is
qualified for promotion, and blanket `inline(never)` is not a demonstrated fix.

The more useful result is the independent architecture comparison using the
baseline kernel. Alternating-order confirmation measures:

| Covariance block scope | Separate regions µs | Grouped predictor µs | Paired ratio [interval] |
| --- | ---: | ---: | --- |
| full construction/compile/context/drop | 28.533 | 23.901 | 1.195 [1.186, 1.203] |
| bind snapshot | 0.596 | 0.574 | 1.036 [1.031, 1.056] |
| executable compile | 1.644 | 1.605 | 1.032 [1.023, 1.038] |
| context creation | 8.906 | 5.926 | 1.495 [1.417, 1.544] |
| first value/gradient benchmark | 13.836 | 12.072 | 1.146 [1.139, 1.150] |
| reused value/gradient | 5.571 | 4.930 | 1.129 [1.114, 1.138] |

The reused latency falls about 11.5%, corresponding to 12.9% more calls per
second by the paired ratio. All seven steady pairs favor grouping. The first
cohort gives 1.133 [1.122, 1.139]; grouping with the outlined kernel
gives 1.143 [1.125, 1.154] within that binary. The combined baseline-to-outlined-
plus-grouped ratio is 1.190, but this cross-binary result retains outlined
controls and is not a promotion argument. No nested lifecycle stages were added
together. The first-call benchmark retains its existing destruction scope.
Scalar covariance, which the switch does not change, has an inconclusive paired
steady ratio of 0.997 [0.985, 1.145] in the confirmation.

Two subsequent same-executable CPU profiles remain consistent with reduced
boundary overhead: `Call::forward` falls from about 6.8% to 4.1% of measured-loop
samples, while the Normal observation/log work remains prominent. Proportions
are normalized to different total runtimes, so their changes are not absolute
per-function time savings. Region grouping does not solve the whole abstraction
deficit: grouped block reuse at 4.930 µs still exceeds scalar reuse at 3.661 µs
in the confirmation. Shared law preparation remains a separate, unmeasured
opportunity.

### Disposition and preserved evidence

Keep the I5.3 candidate rejected. Prioritize a small production construction
change that groups the predictor through the existing region API, with a final
clean compiled-candidate control gate. The experiment's cold environment switch
is an investigative tool, not a proposed permanent API. Removing it changes
the compiled binary again, so even this same-executable architectural result
cannot waive final unchanged-control measurements. An explicit preparation
capability for shared observation parameters is the next architectural question;
its speedup has not been measured. A single-pass disjoint scalar/vector pullback
is another possible future experiment, not a demonstrated improvement here.
No new planner, law kernel, source capability, memory saving, posterior-efficiency
claim or production optimization follows from this review.

Job 2783773 completed all main timings and hash verification, then failed in the
optional profiling tail because its output directory was created relative to
the source tree rather than the study root. Job 2783775 was cancelled while
waiting on that failed dependency and ran no work. Recovery/confirmation job
2783776 completed 0:0 in 2m04s on cn18-42: alternating architecture timings,
then the two optional profiles and packaging. Original timings were not rerun,
replaced or pooled with recovery timings. The original failure and scripts are
retained. The two evidence archives verify locally:

- main counterfactual SHA-256:
  `ca9640d816d639b5365f6cbbd6a453a10cb854f68b5a6ebcd8241e79dc4fa217`;
- architecture confirmation SHA-256:
  `a3c28a84881c2687f69bb80f3b11b12181ffff31a931f6acca397f81f2de0158`.

Remote roots are `/users/usumusu/alea-i53-counterfactual-9NCdWMZp` and
`/users/usumusu/alea-i53-architecture-confirmation`. Frozen executables, source
snapshot, raw profiles, matching library symbols, 28 whole/measured-loop SVGs,
analysis scripts and replay outputs are retained privately. SVGs use the official
[FlameGraph implementation](https://github.com/brendangregg/FlameGraph), revision
`41fee1f99f9276008b7cd112fca19dc3ea84ac32`, and
[pprof](https://github.com/google/pprof) for symbolized stacks. Profile job
2783277, its twelve exact-I5.4-binary profiles, preliminary failed attempts and
Callgrind decoder failure are retained separately. None of the instrumented
profile durations is used as throughput evidence. Existing report-reader tests
pass 31 + 32 cases. Production Rust and all other starting worktree patches are
unchanged by this investigation; roadmap admission remains as recorded in §20.

## 22. I5.5 production-candidate admission: no optimization accepted

The construction change proposed in §21 was implemented and tested, but neither
final-source candidate qualified. The original covariance constructor is restored
byte for byte. Only two additional scalar-reference test states remain: zero
loading, which still requires its loading score, and mixed mean/scale/loading
directions. The I5.3 kernel also remains reverted. There is **no accepted runtime
improvement** from I5.5.

The measured covariance saving is real within each stated experiment. It is not
evidence that the complete compiled change is safe to accept: unrelated model
controls regress. Earlier language describing a “genuine improvement” should be
read as a local architectural cost saving, not a qualified repository-wide win.
There is no weighted aggregate performance score or inferred NUTS/ESS improvement.

### Scope and mathematical checks

Both candidates used the existing lexical-region API for fixed first-coordinate
data and the original add/multiply/add predictor. Offset, slope and location
remained scalar formal inputs; priors, scales, charts, conditional reporting and
all observation-law calls remained unchanged. Four old predictor/data calls
became one, reducing eight region forwards to five and seven pullbacks to five.
No distribution, analytic gradient, sampler, shared preparation, source capability
or public API was added. Required intermediate and local-score admission was
retained through the existing executor. Generic Normal scale preparation remains
unimplemented.

Candidate A placed the region construction directly in the existing constructor.
Candidate B isolated precisely that construction in a private `#[cold]`,
`#[inline(never)]` helper. It did not add padding or global compiler flags.
Neither mathematical correctness nor source capability was the rejection reason.
Both full qualification studies passed, per revision, 159 core, 33 distribution,
47 actual pinned source/AD and 17 native CLI tests, plus all benchmark numerical
preflights. The native reference tests now exercise the two added states.
Local native, optimized reference, related tensor/distribution, Clippy and
format checks also passed. No universal bitwise or new derivative-support claim
follows from these finite tests.

### Matched final-source studies

Both full studies reused the existing I5.4 eight-lane protocol and predeclared
>5% unchanged-control gate: seven alternating pairs, 20 Criterion samples,
0.1s warmup and 0.3s requested measurement, one selected CPU/library thread,
identical arithmetic/provider requests and inputs. Builds and numerical
preflights preceded timing; copied executables were frozen and verified before
and after timing. Source snapshots contained 763 files and differed only in
`tools/native-benchmark/src/conformance.rs`; both arms shared the expanded tests.
The 18 existing benchmark fixture files were sealed experiment inputs, not
additions to the release source allowlist. Pinned Radon input bytes were unchanged.

All timing ran on exclusive ELTE compute node cn18-26. The initially preferred
cn18-42 was occupied; its constraint was removed before the first study started.
No laptop or login-node timing was used. Candidate A job 2783780 completed 0:0
in 1h21m05s. Candidate B job 2783785 completed 0:0 in 1h00m04s. Preservation jobs
2783781 and 2783786 only packaged artifacts on cn18-14; they are not timing hosts.
The local calendar crossed into November 20 during qualification.

Each full study completed 112 cells / 5,124 Criterion estimates, 12,656 first/steady
work receipts representing 31,640 actual public calls, 3,528 storage receipts,
and 23 kernel cases / 322 compatible child attempts. Expected complete case IDs,
work/data/coordinate/query identities, source hashes and all 46 executable hashes
verify. The existing kernel journal replays exactly. All sealed preflight,
result and build logs contain zero NUL bytes; transient concurrent NFS progress
reads with NULs or missing early lines were recorded rather than normalized.

Candidate B reused the exact original baseline executables and build metadata
from A, with explicit origin records, matching source manifests, canonical
compiler path, toolchains and flags. Its baseline numerical preflights and
independent tests were rerun at the restored baseline source; B was then rebuilt
and tested before measurement. No old timing observations were pooled into the
new study. The canonical source directory was reused serially only after the
previous measurements and preservation finished. Historical source receipts
refer to their sealed snapshots, not the later live source directory.

### Local gain versus failed control admission

Ratios are paired baseline/candidate throughput ratios; values above one favor
the candidate. Intervals retain the §20 exploratory paired-bootstrap method
(10,000 resamples, seed 5401), without simultaneous or independent-host coverage.
Arm columns are separate medians; their quotient need not equal the median
within-pair ratio.

| Study / scope | Baseline µs | Candidate µs | Paired ratio [interval] |
| --- | ---: | ---: | --- |
| A / covariance reused value-gradient | 5.649 | 5.000 | 1.132 [1.126, 1.137] |
| A / full construction/compile/context/drop | 28.253 | 23.669 | 1.195 [1.183, 1.201] |
| B / covariance reused value-gradient | 5.676 | 5.035 | 1.125 [1.123, 1.135] |
| B / full construction/compile/context/drop | 28.451 | 23.811 | 1.198 [1.190, 1.205] |
| B / context creation | 8.333 | 5.885 | 1.414 [1.386, 1.479] |
| B / first value-gradient benchmark | 14.199 | 12.058 | 1.178 [1.173, 1.180] |

These are candidate measurements, not shipped gains. Nested lifecycle stages
were not summed; first-call measurements keep their existing destruction scope.
The target does not expose total workspace ownership, so fewer frames and faster
context creation are not promoted into a measured memory-saving claim.

A fails eight repeatable unchanged controls: SIMD scalar covariance, Radon,
centered/noncentered schools, constrained/unconstrained batch-128 reuse, plus
AD-enabled native scalar covariance and Radon. For example, SIMD scalar Radon
moves from 59.119 to 74.659 µs, paired ratio 0.790 [0.778, 0.836], with all seven
pairs beyond the 5% threshold. Four additional >5% median cases remain
inconclusive. Its kernel controls have no regression meeting that gate.

B removes those eight repeatable failures in its focused pilot, but **the final
full study still fails**:

| B unchanged control | Baseline µs | Candidate µs | Paired ratio [interval] | Pairs beyond 5% |
| --- | ---: | ---: | --- | ---: |
| stable default / funnel batch-8 constrained reuse | 0.391 | 0.412 | 0.9492 [0.9203, 0.9522] | 6/7 |

The paired median latency increase is about 5.4%. The gain interval's upper
endpoint, approximately 0.952180, is below the fixed 1/1.05 threshold. It therefore
meets the stated rule even though the boundary is close. The threshold was not
relaxed after seeing the result. Four additional >5% median cases—stable scalar
covariance/Radon and two transactional-boundary controls—remain inconclusive.
All 23 B kernel cases are inconclusive rather than demonstrated changes.
Because one repeatable failure already blocks admission, no additional
confirmation was used to rescue this final candidate.

### What the follow-up establishes and leaves unresolved

In both full studies, 16 of 23 executable pairs are byte-identical. Only the seven
`model_pipeline` lane executables change. Thus the construction-only delta avoids
the broad tensor/regression/source binary changes seen with I5.3, but it does
not reliably preserve the other models in the shared model benchmark executable.
For A's SIMD scalar `NativeModel::logp_grad`, normalized disassembly has the same
1,749 instructions and 8,464 bytes at addresses 0x12aa80 and 0x12b8b0. This
supports investigating code placement, callees and benchmark interactions;
it does not identify cache misses, branch effects or a complete causal mechanism.
Address normalization also does not prove referenced data/callees identical.
Cold construction isolation alone is not a generally qualified remedy.

The B pilot history is retained explicitly. Job 2783782 was cancelled after an
audit found that its narrower Cargo package selection changed transitive feature
unification (`libc`/`bitflags`, among others). This setup mistake made the comparison
unmatched; none of those timings was interpreted. Corrected job 2783783 used the
original package/target selection and matched all 175 dependency artifacts per
lane by features, profiles and artifact identities. It completed 42 cells /
1,050 estimates in stable default, SIMD and AD-enabled native lanes, with no
repeatable >5% failure but two inconclusive controls. Predeclared longer
confirmation job 2783784 used 40 samples, 0.2s warmup and 1s requested measurement
for seven alternating pairs: scalar Radon 0.988 [0.936, 1.025], and checked-gradient
boundary 1.002 [1.000, 1.042]. These permitted the final study; they did not waive
it or replace its failing funnel control. Cohorts were never pooled. There was
no further code variant after B.

Both candidates are rejected and the constructor is restored exactly. The
remaining problem is performance qualification under compiler/benchmark
interactions, not an identified density/gradient defect. Broader investigation
would need a separately scoped plan for that interaction; another unqualified
local saving is insufficient. No benchmark framework, parallel graph, permanent
experimental switch or superseded predictor implementation remains in production.

### Preserved evidence

The full studies' verified SHA-256 digests are:

| Study | Raw results | Frozen binaries/builds |
| --- | --- | --- |
| A | `c66830b8d92ecc739a27aa3c1c36910be6c36681602e6d20721d30ed3cc5a1fb` | `7d74d5488b7e89236cf01ca1f14589f2f761689bb64b632777340c7c73ad3a64` |
| B | `01ed41821fd8a09dc60d7a07a03b4a81c195f84000322b69851d23001c117f40` | `04a7f3f64702078c0edd4e793196fe24d279a26487f75ab4f241874e21ff54b8` |

Remote roots are `/users/usumusu/alea-i55-xNouQtHx` and its `candidate-b-full`,
`candidate-b`, `candidate-b-matched` and `candidate-b-confirm` subdirectories.
Frozen source, protocols, full case distributions, rejected variants, invalid
pilot, dependency audit and exact scripts remain private retained evidence.
This is not a public independent reproduction or offline toolchain distribution.

## I4.6 bounded Gaussian SBC protocol (predeclared 2026-11-20)

This scientific pilot uses the existing downstream `modeling::regression`
fixture, not the covariance, Radon or schools models. Independently generated
intercept and slope have N(0,1) priors; observations at x = [-1,0,1] have
independent N(intercept + slope*x, 0.7) errors. All distributions are proper and
already supported by `Normal::sample`. The generator is checked separately
against prior-predictive means/covariances and the existing independent density
and gradient oracle. Centered design makes the posterior coordinates independent:
variances are 0.49/3.49 and 0.49/2.49, with means sum(y)/3.49 and sum(x*y)/2.49.
An exact control uses these formulas and the existing Normal sampler.

Before looking at ranks, fix 512 datasets and 31 posterior draws per dataset.
Native draws each come from a separate independently seeded NUTS chain: zero
start, step 0.2, depth 7, 256 warmup transitions with the default adaptation,
then 32 frozen transitions, keeping only the final position. This avoids
within-chain rank dependence; finite warmup bias remains testable, not assumed
absent. Exact, ignore-data (prior draws), and wrong-scale (twice the exact
posterior standard deviation) controls each use 31 independent draws. No failed
dataset or chain is dropped. Report warmup and frozen divergences and depth hits.

Quantities, fixed in advance, are intercept, slope, joint log posterior, full
log likelihood, first-observation log likelihood, and remaining-two log
likelihood. Evaluate densities through existing native models, subtracting the
prior-only model for likelihoods. Count ranks below the truth, breaking exact
ties uniformly over 0..=number of ties using a separate random stream. All 32
rank positions map evenly into four bins. Independent streams have seeds
`0x14600000 + stream*1000000 + dataset*1000 + draw`, with streams 0 (generator),
1 (native), 2 (exact), 3 (ignore data), 4 (wrong scale), and 10..13 (rank ties).
The separate generator check uses seed 0x146fffff and 16384 replicates.

Use a simultaneous Hoeffding count band with family alpha 0.01 over all
4 methods * 6 quantities * 4 bins: half-width
sqrt(512/2 * ln(2*96/0.01)), about 50.25 counts (9.81 percentage points).
Under independent uniform ranks this bounds family false rejection by 1%,
without requiring independence between quantities or controls. Correct methods
must stay inside; ignore-data must be detected by at least one likelihood
quantity and wrong-scale by at least one parameter. All bins and failures are
reported before assertions. No threshold, seed or budget is tuned to outcomes.
For one specified bin, a deviation exceeding twice the band width has detection
probability at least 1 - 0.01/96 under the same independent-dataset assumptions;
smaller deviations can easily escape detection. This is a low-resolution pilot,
not a general power guarantee or broad calibration claim. Talts et al. (paper
10) motivates dependence handling; Modrak et al. (paper 27) motivates
data-dependent quantities, randomized ties and broken controls.

### Executed pilot and interpretation

Stable native Rust executed all 512 datasets and 15,872 independently warmed
chains without failed or dropped datasets. Four-bin counts, ordered low to high:

| Quantity | Native NUTS | Exact posterior | Ignore data | Wrong scale |
| --- | --- | --- | --- | --- |
| Intercept | 123,127,127,135 | 121,130,129,132 | 121,134,112,145 | 44,213,190,65 |
| Slope | 128,134,127,123 | 125,127,135,125 | 106,133,125,148 | 54,208,196,54 |
| Joint log posterior | 110,137,138,127 | 108,140,135,129 | 0,2,34,476 | 1,22,134,355 |
| Joint log likelihood | 115,125,135,137 | 124,111,142,135 | 0,2,38,472 | 4,38,145,325 |
| First log likelihood | 116,148,124,124 | 118,139,129,126 | 3,25,142,342 | 19,98,184,211 |
| Remaining log likelihood | 129,121,108,154 | 124,121,118,149 | 0,14,89,409 | 10,69,174,259 |

The simultaneous count half-width was 50.24780973188845 around 128. Neither
correct method rejected any quantity. Ignore-data rejected all three likelihood
quantities and the joint posterior, but neither parameter; wrong-scale rejected
all six. These controls demonstrate sensitivity to these particular large
errors. They do not establish power against small errors or other alternatives.

There were 58,962 divergent warmup transitions out of 4,063,232 (about 1.45%)
and one warmup depth hit. Frozen sampling had zero divergences and zero depth
hits across 507,904 transitions. Warmup divergences were retained, not used to
exclude chains or retune the protocol. Thus the result is compatible with
calibration at this pilot's coarse resolution despite adaptation excursions;
it does not establish convergence for arbitrary initial states or models.
Generator and exact draws share the production Normal sampler: the independent
moment and density/gradient checks reduce, but do not eliminate, common-mode
implementation risk. Historical hierarchical SBC tests are unchanged and do
not inherit this pilot's independence or multiplicity guarantees.

Reproduce the explicit (normally ignored) pilot with:

```sh
cargo test --release --manifest-path tools/downstream-conformance/Cargo.toml \
  --test modeling gaussian_sbc::bounded_regression_pilot -- \
  --ignored --exact --nocapture
```

Its JSON record names all methods/quantities, counts, rejection decisions,
uncertainty band and warmup/frozen failure diagnostics. Separate routine tests
check prior-predictive moments, ties and an independently derived conjugate
posterior density/gradient. No production implementation, derivative capability
or performance claim changes; performance was not measured.

## Performance admission for composed models

The optimization target is elapsed time to a validated statistical result for
a naturally expressed model. The primary compiler unit of work is a complete
value-and-gradient call, including the requested coordinate measure and
admission. For sampling, compare end-to-end time at prespecified precision for
named estimands after numerical and statistical validation. Keep fixed-work
throughput as engine evidence. The initial target for a comparable composed
route is within 20–30% of an equivalent specialized route on a declared
suite. Treat lower-complexity algorithms such as prepared Gaussian compression
as a separate comparison.

### Attribute work and elapsed time

For a retained context and one serial workload use
`T(N) = T_define + T_plan + T_codegen/build/load + T_bind/prepare + T_allocate
+ T_first + (N - 1) × T_steady + T_infrastructure + T_output + T_validation`,
with `N >= 1` and only the applicable terms included. A cache hit reports its
lookup and load costs separately. For a faster steady route with extra setup,
report the break-even calls from matched incremental setup and steady costs.
If the saving is indistinguishable from noise, leave the break-even unclear.

| Boundary | Measure or count | Attribution |
| --- | --- | --- |
| Model and binding | Definition, plan compilation, rebind, provider packing, immutable preparation and source/cache cold phases | Time lifecycle steps around their actual entry points and retain allocation, workspace and artifact sizes |
| Native value and gradient | Chart forward and pullback, scalar/model dispatch, region bridges and copies, tensor forward, reverse accumulation, law/provider calls, validation and publication | Use external fixed-coordinate banks for fused timing. Count calls, explicit copy/clear bytes, residual and replay work. Profile exact frozen binaries separately to attribute inclusive and self time |
| Compiled value and gradient | Emission, pinned Cargo build and load, preparation, first call, steady value and gradient | Record resolved derivative route and admission work. Use compiler/trusted/hybrid comparisons on the same request and input bank. Fused or inlined work may lack a separable stage time |
| Inference and reporting | Warmup, retained elapsed time, actual attempted target calls, transitions, metric adaptation, output and diagnostics | Use existing sampler and session counters. Measure target and non-target work on matched scopes. Report elapsed critical path and CPU work separately across parallel chains |

Use the existing model pipeline, native-compiled, tensor and fixed-work
reports. Preserve unclassified time and profiler overhead as separate
observations. Inclusive samples overlap. Subtract target elapsed time to
estimate infrastructure only where intervals are disjoint on the same
serial scope. In parallel work, summed chain or provider time can exceed
walltime. Call counts alone cannot explain time-to-precision when adaptation,
mixing or numerical admission changes.

### Admit changes by claim and affected workload

Before timing, record the change category, supported path and feature state,
tier, model and requested measure, budgets, primary outcomes, uncertainty
method, stopping and exclusion rules. Freeze a small CPU release workload set
drawn from existing fixtures. Include scalar, composed covariance and Radon,
Gaussian and logistic regression, a tensor-heavy workload and a representative
Wiener workload where supported. Keep actual four-chain workflows distinct from
single-call probes.

| Tier | Evidence | Decision |
| --- | --- | --- |
| A | Qualified scientific and inference workflows, including setup, warmup, output and prespecified precision | A material repeatable degradation blocks the affected release claim unless an explicit reviewed correctness tradeoff applies |
| B | Complete value and gradient, compiled routes, cold cost and memory | Require a predeclared budget for the affected route. Separate native, compiled, default and opt-in results |
| C | Isolated gather, scalar tape, small GEMM, VJP or compiler probes | Use for attribution on architecture changes. Apply strict predeclared guardrails when the patch targets the kernel or when the probe is a designated production proxy |

Numerical correctness and supported statistical semantics are required in
every tier. A necessary correctness fix can carry a reviewed cost and
remediation plan. An architecture or new capability increment can complete on
its opt-in scope with bounded resource use and documented costs. Default
promotion has a separate decision on the exact shipping build. A performance-
only patch needs a reproducible gain on its target and clearance of designated
controls. Established specialized paths retain strict controls for changes
that affect them.

Use practical review bands for latency increases wherever the uncertainty
resolves the size of the change. Up to 3% is generally tolerable for a
concrete architectural improvement. From 3% to 5%, investigate where a
representative workload or several controls move. Above 5% through 10%,
require explicit justification or mitigation. A repeatable unexplained
increase above 10% on a critical production workload blocks promotion.
Use tighter budgets where scientific throughput, resources or a public
performance contract requires them. A wide interval crossing a declared
limit leaves that decision open. Keep the recorded I5 and L4.2 results
under their original gates.

Keep a compact cost-exception entry in the existing report for any accepted
regression. Record baseline and candidate identities, affected path and tier,
absolute cost and interval, suspected mechanism and confidence, reason for
acceptance, mitigation owner or roadmap card, and review or expiry event.
Retain negative and inconclusive results without aggregating unweighted
fixtures into a single score.

The first scoped entry is L4.2's opt-in joint-tile implementation. The sealed
pre-L4.2/current comparison in ELTE job 2784125 confirmed 7.2% higher SIMD
Radon latency and 6.5% higher small compensated GEMM latency. Native Radon
and compiled Enzyme remained inconclusive at the 5% threshold after job
2784131. SIMD Radon is a model-level engine control and small GEMM is a
kernel diagnostic. The original study treated both as blocking controls.
Keep the implementation for its independently validated joint path and
measured scratch and latency benefits on eligible cases. Stable keeps the
materialized plan. L4.6 reviews this evidence and measures the exact proposed
default if a representative workload justifies promotion. Matching normalized
Radon instructions at altered addresses leave the cause unresolved.
[Joint-tile results](tensor-ir.md#joint-tiles-under-qualification).

### Distinguish execution, layout and measurement effects

Use allocated ELTE nodes and the existing paired workflows. Match input bank,
query, numerical policy, compiler and features, provider, affinity, threads
and controls. Preserve original, failed and unstarted attempts. Report
baseline/candidate speedup, absolute latency and uncertainty. Collect enough
pairs for the prespecified decision. Use a bounded confirmation when the
decision remains ambiguous.

Same-executable alternatives identify plan tradeoffs. A/A repetitions estimate
variation for one binary layout. Matched shipping builds measure release
behavior. Predeclared layout perturbations and exact-binary profiles can test
layout sensitivity independently. Matching normalized instructions at moved
addresses supports investigation and leaves causation open. Record requested
and observable clocks separately. Compare uninstrumented timings for release
admission. Keep altered layouts and different hosts as separate evidence strata.

Stop when the planned evidence resolves the stated decision or the
prespecified budget is exhausted. If a diagnostic control moves while
representative workflows clear their budgets, record the cost and end
attribution work when it no longer changes the decision. Use measured
fractions of total cost to bound likely benefit before investing in sampler,
kernel or derivative optimization. For an initial CPU release, require the
qualified model suite to meet independent value/score and reference-posterior
checks, useful latency and workspace budgets, allocation-free retained hot
paths where promised, and a real research workflow. Keep research-only
scheduling, curvature and acceleration on their separate entry gates.
