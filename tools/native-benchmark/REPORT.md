# Native-model numerical and performance investigation

Baseline: Alea `3984ba6c3773a6542d48019d61a1f07594889526`. The revised
implementation is that source plus the retained patch and benchmark source
capsule; it has not been committed or pushed. This report distinguishes the
historical supplied timings from fresh same-host measurements.

Latest result: the [SIMD score-kernel follow-up](#follow-up-simd-score-kernel-optimization)
removes a measured native-kernel bottleneck, yielding 1.16–1.71x paired fit
speedups with exact replay. The earlier v3 tables isolate the numerical repair
and deliberately remain separate from these newer performance measurements.

## Numerical root cause

`TransformError::PrecisionLoss` is defined in
`crates/alea-core/src/error.rs`. Its seven production sites are:

| Function / chart | Exact rejection condition |
| --- | --- |
| `Transform::constrain_impl`, lower/upper | bound ± exp(u) == bound |
| Same, interval | sigmoid mapping rounds at or beyond an endpoint |
| Same, ordered/positive ordered | the next increment does not increase the previous value |
| Same, simplex | component or remaining stick <= 0, or stick fraction == 1 |
| Same, unit vector | 1 + squared norm is nonfinite, or final coordinate == 1 |
| `Transform::factor` in transform/matrix.rs, correlation | (1−tanh(u)) (1+tanh(u)) <= 0 |
| Same, positive Cholesky diagonal | constructed diagonal <= 0 |

The reported case is the **first row**, with lower bound zero. The parameter is
`sigma`, global free coordinate 8 following eight coefficients. The error index
is block-local, hence zero. At the third fused evaluation (initialization plus
one evaluation in each of the first two warmup transitions), its coordinate is
`−806.6677923703763`; binary64 `exp` returns zero. The transform correctly rejects
this unrepresentable positive scale. The Jacobian would be q_sigma, which is
finite; it is neither missing nor duplicated. This is not a long-tree failure:
the error is on integration step one of warmup transition two.

The original benchmark disables initial reasonable-step search and starts at
q=0, sigma=1, with step .2 and identity geometry. The first energy-increasing
proposal is rejected. Standard dual averaging uses an initial mu=ln(10*.2),
so its first zero-acceptance update can increase the next step before subsequent
updates reduce it. The second proposal reaches the coordinate above. This
explains why initialization/search policy exposes the problem, without making
the exponential chart, its Jacobian, or the boundary check incorrect.

The defect is **error classification across adapters**. TransformedTarget keeps
typed transform/model errors. Checked target evaluation wraps those as
`EvaluationError::Model`, while `PhaseWorkspace::advance` previously recognized
only outer nonfinite position/value/gradient variants as numerical divergence.
An unrepresentable proposed state therefore became a fatal backend error and
aborted adaptation. Genuine backend failures must continue to abort.

The repair adds default-fatal `is_numerical_error` methods to fused and constrained
target protocols. Native transform representability failures and explicit native
assembly numerical failures opt in. A boxed factor error opts in only if its
concrete type is the core DistributionError::Numerical variant; parameter,
observation, shape, sampling-budget and foreign errors remain fatal. This also
covers Normal-prior overflow at log sigma=400 or coefficients of magnitude
1e200, preserving the original typed error in direct evaluation. Model/sum/frozen-transport and profiling
adapters forward classification. The phase engine reports `TargetNumerical`
divergence on integration only, using existing rejection/subtree and adaptation
logic. No density, score, Jacobian, tolerance, parameter, step policy or NUTS
algorithm was replaced. Direct evaluation and invalid initialization still return
their original typed errors. Opaque backend, shape, domain, nonfinite-input and
workspace errors remain fatal. The classifier does not interpret error strings
or arbitrary nested foreign errors.

The original native archive SHA-256 is
`dd9ba5858cb1dc348dcecb6551588c9f6a71504a16325efe1c54fe0c1d65d555`.
Data were regenerated with its unchanged generator and seed 20261029; Alea uses
seed [17;32]. The earlier low-level benchmark archive also contains original data:
all five regression/location text files match the regenerated native inputs
byte for byte. The N=10000 Gaussian input SHA-256 is
`55bf42dac7de0e23e4138bb7b54bc55417400548903e243013847da8a15627ce`.
The native archive contains no historical Cargo.lock, so the old seven timing
runs' exact source/toolchain are unavailable. Fresh manifests retain actual input
bytes and interpreter/toolchain pins (CPython 3.13.12 in the pinned stable shell).

## Deterministic size sweep

Same eight-column original Gaussian design, first N original generated rows,
zero initialization and identical chain seed/policy:

| N | Baseline | Revised | First numerical failure |
| ---: | --- | --- | --- |
| 1,000 | complete | complete | none |
| 2,500 | complete | complete | none |
| 5,000 | fatal | complete | native numerical overflow, log sigma −401.9756847399835 |
| 7,500 | fatal | complete | native numerical overflow, −605.5458462949777 |
| 9,000 | fatal | complete | native numerical overflow, −726.988898892894 |
| 9,250 | fatal | complete | transform underflow, −747.1393964340903 |
| 9,500 | fatal | complete | transform underflow, −764.9002357438923 |
| 9,750 | fatal | complete | transform underflow, −788.7651011887035 |
| 10,000 | fatal | complete | transform underflow, −806.6677923703763 |

These are sampled boundaries, not a proof of a monotone threshold for other
data/seeds. Overflow before exp itself underflows is a separate representability
condition in the same proposal/error-classification path. All revised runs retain
1000 draws; failed proposals are counted rather than clipped or accepted.

## Correctness and validation

`src/correctness.rs` checks full normal priors, the restricted sigma prior, the
normal/logistic joint likelihood, all score components and the exp-chart Jacobian
against independent posterior equations at N=37/1000/2500/5000/7500/10000. Central
differences are a separate sampler-independent check. It also tests extreme
coordinates, unchanged caller gradients after raw target failure, and the actual
second-transition failure with the original large-model seed.

`scripts/check_stan.py` executes CmdStan 2.40.0 `log_prob jacobian=1` on 22 shared
free-coordinate points across all five regression/location cases. Densities agree
after subtracting parameter-independent constants; every gradient passes the
declared comparison tolerances. These include log sigma=−20 and logistic
coefficients ±1000. Both models use the same row orientation, coefficient priors,
initial values and dimension. Neither adds a half-normal ln(2) normalizing constant.

CmdStan's default central-difference `diagnose` at zero for N=10000 returns one
absolute residual 1.27485e−6 against its default 1e−6 threshold; the other eight
coordinates pass. That failed diagnostic is retained rather than relabeled a
pass. It is not evidence of a disagreement between the engines' analytic scores:
the independent equations, finite differences and direct cross-engine score
comparisons are separate evidence.

`crates/alea-mcmc/tests/native_numerical.rs` checks typed transform errors in direct
evaluation and initialization, integration divergence, cache/scratch recovery,
and continued propagation of opaque backend errors. Existing callback/scalar/
chunked regression, allocation-reuse and finite-difference tests remain unchanged.
Final core/distribution/MCMC execution: 75 suites, 492 passed, zero failed,
three existing ignored tests. The 18 runtime session tests, the
downstream profiling observer test, scoped Clippy and formatting pass. The public
API matrix was regenerated with its pinned compiler and reviewed for the two
default methods and divergence variant; all 14 neutral feature variants pass.

## Confirmed cost attribution

Linux perf recording is denied for both hardware cycles and software CPU-clock
events. Release Callgrind on N=1000 Gaussian regression records 8,415,526,999
instructions; approximately 98% are inside the fused regression callback.
Compensated arithmetic accounts for 46.25% of total instructions at numeric.rs
source attribution; the FMA residual implementation accounts for about 10%
inclusive. Row/zip/index and scalar arithmetic account for most of the remainder.
These are instruction counts, not sampled CPU-cycle percentages.

The callback already fuses likelihood and analytic scores. Its checked fixed
design and scratch are reused; there is one batch node, no node/allocation per
observation. Every row computes a compensated predictor with multiplication
residuals and compensated density/coefficient/scale reductions. The older
handwritten target uses ordinary sums and omits sigma entirely for Gaussian
models. Thus the original timing gap cannot be assigned to graph dispatch alone,
nor can the fixed-sigma Gaussian target establish a native-path regression.

The large-model public kernel control records 12891 evaluations: initialization
1, warmup 6246 edges and sampling 6644 edges. It has 8 warmup divergences, 5 metric
updates, no search probes or bounded-step updates, warmup maximum depth 8 and mean
2.532, final step .4904306781529189 and zero retained divergences. In one attributed
run, target evaluation occupies 9.171 of 9.185 warmup+sampling seconds. Construction
is below a millisecond in an ordinary release run. These phase observations are
not repetition-based speed estimates. They rule out repeated session construction
or tree infrastructure as the dominant cost for this workload.

A small compensated-sum rewrite was tried and removed: the preliminary timing
did not support keeping it. The numerical repair alone does not improve kernel
throughput. The subsequent SIMD score-kernel change is measured separately below;
none of the numerical-repair timings are relabeled as SIMD measurements.

## Measurement scope and evidence

Fresh timing tables and limitations below are populated from final preserved
cohorts. Early overlapping-build and profiling cohorts remain separate and are
not used to claim speed improvements. All raw evidence lives externally in
`../alea-inference-evidence/`, including the original archives, baseline checkout,
saved executables, source patches/capsules, size-sweep JSONL, Stan checks and
Callgrind output. No historical archive or user input was overwritten.

## Fresh measurements (final v3)

Intel Core i5-8250U, x86_64, one logical CPU (2), one thread per engine.
Rust 1.95.0, release opt-level 3, fat LTO, one codegen unit. Stan models use
GCC 16.2.0, -O3; ELF .comment confirms the compiler. Early v2 manifests' cxx
field is runtime GCC 15.2.0, not the model build compiler; v3 records the ELF
compiler separately. No target-cpu=native, fast-math or parallel-chain speed claim.

The final v3 includes boxed core distribution numerical-error classification.
All 18 native four-chain outcomes reproduce v2 trajectory hashes, evaluation/
leapfrog/error counts and frozen step sizes exactly. The matched baseline and
v3 binaries use identical Rust benchmark harness source.

Before/after: three alternating pairs on the final binaries, one chain,
1000 warmup + 1000 retained draws. Entries are median seconds [observed min,max],
not confidence intervals. Every commonly successful trajectory and work count
is identical. The original large-model failure recovers in 3/3 final repeats.
The preceding v2 repair also had seven paired repeats, with 7/7 large-model
recoveries; that independent cohort remains preserved. There is no demonstrated
general speedup.

| Workload | Native baseline | Native final v3 |
| --- | ---: | ---: |
| mean | 0.179 [0.176, 0.184] | 0.178 [0.177, 0.180] |
| linear | 0.963 [0.957, 0.966] | 0.956 [0.954, 0.964] |
| logistic | 1.165 [1.157, 1.168] | 1.168 [1.164, 1.173] |
| linear_large | failed 3/3 | 9.202 [9.159, 9.327] |
| logistic_large | 11.143 [11.074, 11.317] | 11.095 [11.087, 11.174] |
| funnel | 0.066 [0.065, 0.070] | 0.071 [0.067, 0.072] |

Fixed-work evaluation throughput, 10000 calls cycling through ten positions,
three alternating pairs. Counts/second use posterior evaluation time. The
checksums are identical.

| Workload (N=1000) | Before calls/s | After calls/s |
| --- | ---: | ---: |
| mean | 45472 | 45695 |
| linear | 14075 | 14000 |
| logistic | 10959 | 10794 |

Cross-engine final v3: four sequential chains per outcome, three timing repeats
per engine/workload, 1000 warmup + 1000 retained draws per chain. Times below
cover all four chains, so do not compare them directly with single-chain times.
ESS/sec is minimum bulk ESS over every parameter divided by total process time.

All 36 outcomes complete. Every Gaussian/logistic outcome passes R-hat<=1.01,
bulk/tail ESS>=100 and zero retained-divergence gates. Both funnels fail these
diagnostics: their apparent timing/ESS advantages are ineligible for favorable
inference-efficiency claims (†).

| Workload | Alea seconds [range] | Stan seconds [range] | Alea min ESS/s | Stan min ESS/s |
| --- | ---: | ---: | ---: | ---: |
| mean | 0.728 [0.708, 0.756] | 0.199 [0.192, 0.247] | 5104 | 17740 |
| linear | 3.912 [3.819, 4.093] | 1.016 [1.016, 1.155] | 1681 | 5436 |
| logistic | 4.735 [4.616, 4.879] | 2.474 [2.324, 2.545] | 1218 | 2166 |
| linear_large | 36.041 [36.018, 36.065] | 9.270 [9.257, 9.363] | 187 | 607 |
| logistic_large | 44.330 [43.969, 44.488] | 23.155 [22.210, 24.691] | 137 | 255 |
| funnel † | 0.283 [0.282, 0.289] | 0.430 [0.425, 0.439] | 174 | 141 |

Work and diagnostics are repeat-identical. Each A/S pair below is Alea / Stan.
Evaluations include warmup and initialization/search; edges/depth/divergences
are retained-phase statistics. Counts are measured, not inferred from depth.

| Workload | Posterior evaluations A/S | Retained edges A/S | Mean depth A/S (max A/S) | Max R-hat A/S | Retained divergences A/S |
| --- | ---: | ---: | ---: | ---: | ---: |
| mean | 28239/38295 | 12920/15750 | 1.820/1.972 (3/3) | 1.0011/1.0006 | 0/0 |
| linear | 52312/58275 | 26336/25764 | 2.824/2.776 (3/3) | 1.0036/1.0023 | 0/0 |
| logistic | 49172/56911 | 25504/25584 | 2.756/2.765 (3/3) | 1.0025/1.0030 | 0/0 |
| linear_large | 50183/56194 | 25852/25012 | 2.808/2.752 (3/3) | 1.0018/1.0022 | 0/0 |
| logistic_large | 47506/54794 | 24502/24072 | 2.707/2.660 (3/3) | 1.0020/1.0027 | 0/0 |
| funnel | 213904/271740 | 107104/185935 | 3.749/4.269 (9/10) | 1.0798/1.0515 | 32/18 |

Alea uses fewer evaluations than Stan in every Gaussian/logistic case. Extra
gradient calls do not explain its slower wall time. Four large-Gaussian chains
each encounter one recoverable transform failure during warmup; all other
cases record zero returned target errors. Energy-cutoff warmup divergences are
separate from returned target errors.

| Workload | Posterior calls/s A/S | Calls/process-second A/S |
| --- | ---: | ---: |
| mean | 39948/364433 | 38789/192430 |
| linear | 13540/67622 | 13373/57359 |
| logistic | 10476/24734 | 10385/23004 |
| linear_large | 1396/6287 | 1392/6062 |
| logistic_large | 1074/2405 | 1072/2366 |
| funnel | 1059436/2907896 | 754856/632020 |

Posterior throughput has an explicit scope asymmetry: Alea times the transformed
target, while Stan profiles its posterior block after parameter transforms.
Both observers add overhead, and none is subtracted. Standard deviations and
per-repetition values remain in summary.json/records.jsonl.

All normal/regression retained maximum depths are 3 in both engines. Final
frozen steps agree exactly with the preceding v2 trajectories. Stan values use
full-precision stepsize__ output, not rounded adaptation comments:

| Workload | Alea min–max step | Stan min–max step |
| --- | ---: | ---: |
| mean | 0.786329–1.066573 | 0.795275–0.857682 |
| linear | 0.630510–0.662041 | 0.655182–0.701722 |
| logistic | 0.649199–0.720689 | 0.673456–0.702201 |
| linear_large | 0.490431–0.558495 | 0.518009–0.600047 |
| logistic_large | 0.684318–0.809276 | 0.676265–0.859586 |
| funnel | 0.052866–0.093932 | 0.079881–0.223853 |

Serialization control (v2, three repeats/four chains, separate cohort): native
17-digit parameter CSV costs median 2.44 ms (mean), 8.24 ms (linear), 7.81 ms
(logistic). These are below 0.5% of process time and cannot explain 2–4x gaps.
Unpaired cohort wall differences are not treated as causal serialization gains.
Stan writes richer CSV; Alea diagnostics run inside its process, while Stan
analysis runs outside measured processes. Workflows remain explicitly unequal.

A real retention control runs the failing baseline large model first, then mean:
the runner records the failure, completes Stan and both subsequent mean jobs,
and produces 4 planned/attempted, 3 successful and 1 failed outcome. The failed
model has no eligible timing value.

## Earlier low-level logistic control

The earlier ZIP's original handwritten target was reused **externally**, with
only counters and a fixed-work driver added. Its probability/score code was not
rewritten and is not part of the native benchmark. It uses the identical
archived logistic data (SHA-256 a1620b259ea7d43732bd310a17d0d5ce15306a110618670b749e40672c48e4d0)
and the same current Alea sampler/seed. Seven alternating pairs on CPU 2 give:

| Logistic N=1000 control | Original low-level target | Native public target |
| --- | ---: | ---: |
| Fixed 10000-call evaluation seconds, median | 0.441 | 0.890 |
| Sampling process seconds [range] | 0.567 [0.555, 0.579] | 1.127 [1.115, 1.172] |
| Actual total evaluations | 12641 | 12521 |
| Retained leapfrog steps | 6668 | 6556 |

The native target takes about twice the time with slightly **less** trajectory
work. This independently supports a kernel-cost explanation. The low-level
target omits parameter-independent prior constants and uses ordinary sums and
a different tail-score calculation, so it is a diagnostic reference, not a
replacement for Alea's native distribution accuracy contracts. Instrumentation
also differs (an atomic call counter versus the native count/timer observer).
Fixed-sigma Gaussian comparisons are deliberately not used. This control uses
the saved v2 repair binary and is separate from final v3 cohorts; no timing is
relabeled as a measurement of another binary.

## Follow-up: SIMD score-kernel optimization

The performance defect is missed vectorization in an expensive inner reduction,
not that the distribution definitions are outside main.rs. The native batch
callback already fuses density and analytic gradients. Its scalar coefficient
loop updates an array of two-field CompensatedSum accumulators with a conditional
rounding correction at every observation/column. The earlier main.rs target
uses ordinary sums, no multiplication residuals, and a cheaper but less accurate
tail-score expression. The Gaussian control additionally fixes sigma, so it is
not the same posterior or trajectory workload.

The follow-up changes only the normal and Bernoulli-logit batch implementation
in `crates/alea-distributions/src/modeling/regression.rs`. Private `Scores`
stores heads and corrections separately; `Scores::add_scaled` uses existing
Pulp runtime dispatch to update independent coefficients with SIMD masks.
`NormalEval::with_simd` and `LogitEval::with_simd` dispatch once per batch
evaluation, not once per row. Normal's sigma accumulator stays scalar. There is
no SIMD reduction across observations: every coordinate retains its exact
observation/addition order and Neumaier correction. Predictor product residuals,
scalar distributions, Jacobians, validations, graph propagation, and NUTS are
unchanged. Existing reusable callback scratch still owns all working buffers;
no per-observation allocations are introduced. Unsupported CPUs use the portable
fallback. Pulp was already pinned in the workspace; no fast-math or global native
CPU compiler flag is enabled.

Fixed-work release Callgrind, 1000 evaluations of the same logistic points,
including process construction, records:

| Target | Instructions |
| --- | ---: |
| Original handwritten control | 364,000,468 |
| Native v3 before SIMD | 768,450,249 |
| Native SIMD score kernel | 516,417,128 |

The optimized native process executes 32.8% fewer instructions. Counts are not
CPU-cycle percentages. `logistic-fixed-{handwritten,before,simd-score}.callgrind`
retain the profiles. There is also avoidable ISA/code-generation overhead: the
baseline makes eight million out-of-line product-residual FMA helper calls in
this control (7999999 cached hardware calls plus the initializer). This is not
eight million software-emulated FMAs. Those hot-loop helper calls disappear in
the optimized ISA-specialized kernel; the fused residual arithmetic is retained.
Plain runtime-dispatched closures, forced-inline hints and
a branch-free TwoSum trial did not demonstrate sufficient benefit; those trials
were removed. The global CompensatedSum implementation is unchanged.

Seven alternating paired repeats on CPU 2 compare saved `native-final-v3`
(already numerically repaired) against `native-simd-score-v1`. All 126 outcomes
complete. Every evaluation checksum and successful sampling trajectory hash,
evaluation count, retained leapfrog count and divergence count agrees exactly.
Thus these gains are from cheaper evaluation, not fewer evaluations. Each fit
uses one chain, 1000 warmup and 1000 retained draws. Times include process
startup/parsing/construction/diagnostics; ranges are observed min/max, not
confidence intervals. Repeated seeds measure timing spread, not independent
Monte Carlo accuracy.

| Workload | Before seconds [range] | SIMD seconds [range] | Median paired speedup |
| --- | ---: | ---: | ---: |
| mean | 0.177 [0.176, 0.178] | 0.153 [0.152, 0.153] | 1.16x |
| linear | 0.995 [0.961, 1.063] | 0.579 [0.577, 0.584] | 1.71x |
| logistic | 1.161 [1.133, 1.166] | 0.810 [0.791, 0.820] | 1.43x |
| linear_large | 8.939 [8.883, 9.068] | 5.348 [5.306, 5.421] | 1.67x |
| logistic_large | 10.789 [10.734, 11.047] | 7.470 [7.444, 7.614] | 1.44x |
| funnel (unchanged kernel) | 0.0645 [0.0644, 0.0654] | 0.0645 [0.0638, 0.0647] | 1.01x |

Fixed 10000-call native evaluation medians are mean 0.220736 -> 0.190643 s,
linear 0.723794 -> 0.429734 s, and logistic 0.922204 -> 0.637964 s. Full
per-repeat throughput, standard deviations, hashes and paired ratios are in
`native-simd-score-paired/{records.jsonl,summary.json}` outside the checkout.

Correctness: two new private score tests compare scalar/fallback and dispatched
SIMD totals bit-for-bit against CompensatedSum at every prefix for dimensions
0..65, including tails, cancellation and subnormals. Overflow/NaN preserve output
sentinels and scratch is reusable afterward. Existing distribution integration
tests, independent model equations/finite differences/extreme-coordinate tests,
the original-seed warmup recovery test and both numerical propagation tests pass.
All 22 actual CmdStan shared-coordinate density/gradient checks pass again in
`native-stan-correctness-simd-score`. Scoped Clippy with warnings denied passes.
The final full core/distributions/MCMC run passes 494 tests in 75 suites, with
zero failures and three existing ignored tests; raw output is retained as
`native-simd-score-tests.log`. Python runner tests, formatting and the workspace
architecture guard pass.

Fresh cross-engine follow-up: four sequential chains per outcome, three repeats,
same pinned source/data/toolchains and process-scope asymmetries as above. All
36 outcomes complete; all 30 Gaussian/GLM outcomes pass the declared diagnostic
gates. Native trajectory hashes, calls, retained edges/depth/divergences, final
steps and non-timing R-hat/ESS summaries exactly match v3 in all 18 outcomes.
There are no extra or missing gradient evaluations behind the gains. The
following times cover **all four chains**, not the single-chain fits above.

| Workload | SIMD Alea seconds [range] | Stan seconds [range] | Alea min bulk ESS/s | Stan min bulk ESS/s |
| --- | ---: | ---: | ---: | ---: |
| mean | 0.595 [0.593, 0.611] | 0.191 [0.186, 0.199] | 6251 | 18452 |
| linear | 2.448 [2.445, 2.457] | 1.094 [1.085, 1.114] | 2685 | 5048 |
| logistic | 3.386 [3.375, 3.419] | 2.457 [2.453, 2.458] | 1704 | 2181 |
| linear_large | 22.041 [21.883, 23.043] | 9.797 [9.352, 10.000] | 307 | 574 |
| logistic_large | 31.012 [30.449, 31.114] | 22.452 [21.904, 22.458] | 196 | 263 |

Both centered funnels still fail diagnostic gates (native/Stan retained
divergences 32/18, max R-hat 1.080/1.052); their completed times are not eligible
for a favorable efficiency claim. Full profile throughput, diagnostics, work,
manifest and CSVs are in `native-four-chain-simd-score`. Alea still loses to Stan
on the five qualified cases; the SIMD change narrows that gap rather than proving
parity. Earlier main.rs Gaussian times still cannot establish a same-model
speed comparison.

The saved SIMD executable SHA-256 is
`91e609d4569df080403d4ec35d4d3449866adad89f521400a45b2444b89708be`.
A subsequent release rebuild produces this identical executable SHA-256.
Source overlay `native-simd-score-source-complete.tar.gz` has SHA-256
`f811a5e19510d8adbf4e921f4222630dc20fa4eced899d7686571edf18e1da55`.
All 360 source hashes in the four-chain manifest and its standalone Cargo.lock
were verified against that capsule. It includes ignored example source aliases
as file content; the initial `native-simd-score-source.tar.gz` reused the older
archive's narrower membership and is not the qualified complete overlay.
`native-core-and-simd-score.patch` retains tracked changes over the pinned commit;
dependencies/submodules still require the pinned checkout and package sources.
The final evidence seal is `native-simd-score-evidence-final.sha256`.

## Remaining limitations

- The v3 tables above measure the numerical repair, before the subsequent SIMD
  score-kernel optimization. That follow-up improves native throughput without
  removing compensation, but does not establish universal parity with Stan.
  Predictor dot products still retain multiplication and summation residuals;
  arbitrary-dimensional, ill-conditioned designs cannot be replaced with the
  earlier target's ordinary sums merely to match its instruction count.
- Extreme finite free coordinates can still produce typed transform or
  intermediate-density/score errors. They are not silently mapped to boundary
  parameters or fabricated finite answers. The default-fatal protocol does not
  infer that arbitrary foreign numerical/backend errors are recoverable, and
  target wrappers must forward explicit classification when appropriate.
- Centered funnel inference is inadequate in both engines at these budgets.
  More iterations alone do not establish efficient inference; the reported
  funnel ESS rates are unqualified diagnostic results, not speed claims.
- Workflows use different initial search/adaptation, RNG streams and process/
  storage/diagnostic boundaries. Instrumented evaluation timings have different
  transform scopes. Compilation is excluded, while startup/parsing are included
  in process times rather than isolated by a universal empty-process correction.
- One shared laptop and three four-chain timing repetitions provide observed
  spread, not confidence intervals or multi-host performance qualification.
  perf recording is unavailable; instruction attribution is from Callgrind.
  Allocation tests cover reused evaluations/transitions, not whole-process
  allocation counts or retained draw storage.
- Rust and CmdStan were built from different pinned Nixpkgs trees. The installed
  CmdStan tree has an exact content pin but no available upstream Git revision.
  Revised Alea is identified by baseline commit plus exact patch/source hashes;
  there is no invented committed revision for an uncommitted worktree.

The primary final cohorts are native-paired-v3 (54 outcomes, three expected
baseline failures) and native-four-chain-v3 (36 completed outcomes, six funnel
diagnostic failures). Earlier native-paired-quiet (126 outcomes, seven expected
baseline failures) and native-four-chain-final (36 complete v2 outcomes) remain
separate. Additional controls are native-serialization-final (18 complete outcomes),
native-failure-retention (three complete/one expected failure), and
lowlevel-diagnostic.jsonl (28 complete controls). Exploratory native-current-single
was stopped and is excluded; native-paired-final is also separate from the final
quiet cohort. The earlier v2 capsule is native-measured-source-final.tar.gz,
SHA-256 0f2a9a64f05b67f0bf00a62769890e8321c584b6fa2fe4322a2c24a615076967.
The earlier native-final-v2 binary SHA-256 is
d1eecdd3fb088cf0688632adb3963ee8798346b09ba56dc6d17375d96181f8c6.
The measured runner is preserved as native-four-chain-runner.py; later metadata
hardening records the actual ELF compiler rather than the runtime PATH compiler.

Final v3 source capsule: native-measured-source-v3.tar.gz, SHA-256
e3e513db21506732ca1174cedaafb375dce48da58ca7ec43cf41dbec3303326c.
Final v3 binary: native-final-v3, SHA-256
03d163ba178b6565dd3a8db04993c009babebc4ab033a93f22b99def773f5429.
The final patch is native-core-and-api-v3.patch and the matched baseline binary
is native-baseline-v3. Those source and executable artifacts are preserved.


