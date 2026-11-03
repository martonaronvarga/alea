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
`Transform::constrain_impl` (`transform.rs`) and `Transform::factor`
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
