# Alea, nuts-rs and Stan: source review and numerical choices

Alea has avoidable trajectory overhead and a separate adaptation-efficiency gap.
The clearest engineering defects are repeated endpoint-velocity calculations,
full phase-vector copies when creating leaves, and an unnecessary dense matrix
inverse during covariance fitting. Those have concrete replacements. The excess
gradient calls require controlled adaptation and stopping-rule comparisons;
changing Rust syntax will not remove them.

This review compares the current Alea worktree with nuts-rs 0.18.3 at
[`a762aae`](https://github.com/pymc-devs/nuts-rs/tree/a762aae513bf7bb5ccb27ffc9a938a923fc837ec)
and Stan 2.38.0. The local nuts-rs checkout has three formatting-only changes;
the examined numerical behavior matches that revision. Stan source provenance
is retained in [the executed adaptation reference](../tools/stan-adaptation-reference/README.md).
[Review input hashes](benchmarks/nuts-reference-review-2026-10-22/receipt.json)
identify the uncommitted Alea sources actually examined. These are pinned
comparisons, not claims about every release or backend.

Primary modules are Alea's NUTS tree, turning arithmetic, native warmup and
covariance fitting. Supporting modules are phase/point storage, Euclidean
metrics, dual averaging and existing Fisher adaptation. Relevant regression
coverage includes the recursive tree oracle, numerical turning cases, typed
failure/panic rollback, allocation contracts, native warmup and executed Stan
adaptation fixtures. Research sampler exactness and a new broad benchmark matrix
are outside this review. No sampler implementation or default changes here.

**1. Cache endpoint velocities before changing turning mathematics.**

Every call to Alea's
[`TurnScratch::continues`](../crates/alea-mcmc/src/nuts.rs#L208) applies the metric
to both endpoint momenta. Every merge calls it three times, even after a previous
criterion reports a turn. This means six metric applications per merge. With a
dense mass, each application currently performs two triangular solves.

A fresh instrumented run at dimensions 2, 16 and 64 confirms the count:

| Complete leapfrog edges | New gradient calls | Metric velocity calls |
| ---: | ---: | ---: |
| 1 | 1 | 9 |
| 7 | 7 | 57 |
| 255 | 255 | 2,041 |

All nine runs end at their configured depth, without divergence. Construction
calls are excluded. The result is exactly `1 + 8 * edges` for these trajectories:
one initial energy application, two applications per edge for drift and endpoint
energy, and six per merge. This is an executed operation count, not a timing
comparison or a prediction of an eightfold speedup.
[Raw counts](benchmarks/nuts-reference-review-2026-10-22/metric-counts.csv),
[standalone consumer](../tools/nuts-reference/metric-count-probe/README.md).

Stan carries endpoint sharp momenta, `M^-1 p`, through tree construction.
nuts-rs stores velocity in its transformed phase points. Adopt that caching
principle: compute an endpoint velocity after its final momentum update and reuse
it for energy and turning. On this probe's metric, `1 + 2 * edges` applications
could serve those operations. The cache must be invalidated whenever momentum
or geometry changes; it cannot span a momentum refresh indiscriminately.
[Stan tree](https://github.com/stan-dev/stan/blob/v2.38.0/src/stan/mcmc/hmc/nuts/base_nuts.hpp),
[nuts-rs phase representation](https://github.com/pymc-devs/nuts-rs/blob/a762aae513bf7bb5ccb27ffc9a938a923fc837ec/src/dynamics/transformed_hamiltonian.rs).

Also elide the two duplicate cross checks when both children are leaves: all
three use the same momentum sum and endpoint pair. Consider short-circuiting
after a turn, as nuts-rs does, but explicitly review how skipping a metric call
affects observable typed failures. Preserve the generalized full-tree and cross
criteria for larger trees. nuts-rs uses displacement-based tests with different
zero-dot handling; their lower cost does not establish better sampling.

The existing native 16D Gaussian profile attributes 41.70% of sampled cycles to
turning arithmetic and 9.44% to `memmove`. This independently makes the first two
recommendations plausible priorities, although that profile precedes the newest
Enzyme integration and is not a profile of the counting wrapper.
[Profiling evidence](benchmarks/overhead-2026-10-22.md).

**2. Replace leaf vector copies with private reusable state handles.**

[`Tree::leaf`](../crates/alea-mcmc/src/nuts.rs#L182) clones a complete phase into
left endpoint, right endpoint and candidate, then copies momentum into `rho`.
Each phase contains position, gradient and momentum. This writes `10 * dimension`
floating-point entries per leaf, before other workspace copies. Reusing the
allocations avoids allocation cost; it still copies the arrays.

nuts-rs clones reference-counted state handles and recycles phase slots after
their last reference disappears. Its pool may allocate while reaching its high
water mark, so that is not a guarantee of allocation-free first transitions.
The useful idea is immutable shared snapshots with reuse.
[nuts-rs pool](https://github.com/pymc-devs/nuts-rs/blob/a762aae513bf7bb5ccb27ffc9a938a923fc837ec/src/dynamics/state.rs).

Choose a private, preallocated slot arena with lightweight endpoint/candidate
handles, or an equivalent safe pool. Keep Alea's iterative binary carries,
bounded `O(dimension * depth)` live storage and coherent public point snapshots.
The existing buffer exchanges already remove some copies; leaf copies remain.
There is no need to reproduce the upstream pool's unsafe implementation.

**3. Factor the learned inverse mass directly.**

Let `G = M^-1`. Alea's regularized position covariance estimates `G`, but its
[`dense fitting branch`](../crates/alea-mcmc/src/adapt/covariance.rs#L193) factors
`G`, solves against every basis vector to form `M`, symmetrizes `M`, and factors
`M` again. That adds an explicit inverse and another cubic factorization. The
inverse can also introduce enough error to spoil an otherwise usable SPD fit.

Choose a distinct representation for learned inverse mass, preserving the public
meaning of existing mass matrices. For `G = L L^T`, the factor supports:

| Operation | Formula |
| --- | --- |
| Velocity | `v = L (L^T p)` |
| Kinetic energy | `T = 0.5 * norm(L^T p)^2` |
| Momentum sampling | solve `L^T p = z`, with `z ~ N(0, I)` |
| Mass log determinant | `log det M = -2 * sum(log L_ii)` |

This avoids forming `M` and gives a nonnegative sum-of-squares expression for
energy. Retain checked finite arithmetic and SPD admission; a factor alone does
not prevent overflow on arbitrary inputs. If a cached full-`G` matrix-vector
product is faster at relevant dimensions, compare it with the factor form and
its rounding behavior explicitly.

Stan already stores and applies inverse mass directly. Its pinned dense momentum
sampler invokes an LLT factorization when sampling; do not copy that repeated
factorization. Combine its inverse-mass convention with an immutable cached
factor. Checkpoint representation and replay need deliberate versioning rather
than silently reinterpreting existing saved mass factors.
[Stan dense metric](https://github.com/stan-dev/stan/blob/v2.38.0/src/stan/mcmc/hmc/hamiltonians/dense_e_metric.hpp).

The same engineering principle applies to diagonal geometry. Alea currently
divides by each mass on every velocity operation and computes square roots during
momentum sampling. Learned `G_i` and cached square-root scales can avoid repeated
work. A generic public mass constructor still needs a policy for positive masses
whose reciprocals are unrepresentable; replacing every division by multiplication
without that review would weaken its numerical domain.

**4. Treat adaptation as the main explanation to test for excess gradients.**

The previous 20-target, three-repetition PosteriorDB cohort can be decomposed
without rerunning sampling. These are medians across targets; call ratios first
take the median across repetitions. ESS ratios divide the reported per-target
median minimum bulk ESS per retained gradient call.

| Alea method, relative to nuts-rs | Retained calls | Retained seconds/call | Retained ESS/call |
| --- | ---: | ---: | ---: |
| Covariance diagonal | 1.619 | 1.062 | 0.692 |
| Fisher diagonal | 1.369 | 1.070 | 0.751 |
| Fisher rank-4 | 1.047 | 1.062 | 1.328 |

Every target is included without a quality filter. Repetitions reuse declared
seed sets. Seconds/call includes sampler work and the common derivative backend;
it does not measure pure gradient latency. The columns have different aggregation
orders and must not be multiplied to reconstruct a median walltime ratio.
[Recomputed records](benchmarks/nuts-reference-review-2026-10-22/cohort.json),
[analysis](benchmarks/nuts-reference-review-2026-10-22/analyze.py).

nuts-rs diagonal adaptation uses both position and gradient moments by default.
Its coordinate inverse-mass estimate is `G_i = sqrt(Var(q_i) / Var(score_i))`,
with paired foreground/background estimators, frequent updates and clipped scale
ranges. Alea's default uses position covariance and discrete slow windows. These
are materially different workflows despite sharing a target acceptance of 0.8.
[nuts-rs estimator](https://github.com/pymc-devs/nuts-rs/blob/a762aae513bf7bb5ccb27ffc9a938a923fc837ec/src/math/cpu_math.rs#L667),
[adaptation schedule](https://github.com/pymc-devs/nuts-rs/blob/a762aae513bf7bb5ccb27ffc9a938a923fc837ec/src/adapt_strategy.rs).

There is a useful mathematical motivation for paired geometry. With position
covariance `C` and score covariance `F`, matching them through `G F G = C` gives
the diagonal square-root ratio above. A regular Gaussian has `F = C^-1`, so the
full-matrix solution recovers `G = C`. Beyond that setting, finite moments,
regularity and coordinate dependence matter; the diagonal rule does not solve
arbitrary rotated curvature. This derivation motivates an experiment, not a
theorem of faster mixing on every posterior.

Alea already implements Fisher diagonal and low-rank adaptation. Choose a
controlled comparison of its initialization, estimator windows, update frequency,
regularization and final step-size restart. The rank-4 results make that a strong
existing candidate. They do not isolate which lifecycle choice caused the gain.
Keep covariance warmup as the simple reference while making this comparison.

Do not copy every nuts-rs heuristic. Its diagonal collector excludes origin
repeats and uses a selected-index threshold on divergent draws. Removing repeats
generally changes Markov-state moment weighting to jump-chain weighting. Keep
repeats when estimating ordinary chain covariance unless the estimator's intended
measure and correction are explicit. Do not replace Alea's numerically hardened
Fisher factorization merely to match a less stable algebraic formulation.
[Collector](https://github.com/pymc-devs/nuts-rs/blob/a762aae513bf7bb5ccb27ffc9a938a923fc837ec/src/transform/adapt/diagonal.rs#L73).

**5. Keep ordinary whole-tree acceptance as the reference tuning statistic.**

nuts-rs switches from ordinary average Metropolis acceptance early in warmup to
a symmetric statistic later. For energy error `delta`, the latter is
`2 / (1 + exp(abs(delta)))`. It differs pointwise from `min(1, exp(-delta))`.
The same nominal acceptance target therefore does not imply the same update
sequence. This is not evidence that the symmetric statistic is invalid; paired
expectations can agree under appropriate reversible stationary proposals.
[Statistics](https://github.com/pymc-devs/nuts-rs/blob/a762aae513bf7bb5ccb27ffc9a938a923fc837ec/src/stepsize/dual_avg.rs#L130),
[early/late updates](https://github.com/pymc-devs/nuts-rs/blob/a762aae513bf7bb5ccb27ffc9a938a923fc837ec/src/stepsize/adapt.rs#L211).

Choose ordinary whole-tree acceptance, a one-sided energy-increase cutoff, and
explicitly frozen retained geometry as Alea's baseline. Keep the existing
Stan-like covariance windows and post-fit step search/restart. Executed fixtures
already document short-budget schedule differences, so this is not a claim of
bitwise identity with every Stan service. If symmetric tuning is compared, expose
it explicitly and evaluate it as an isolated change. A stable implementation uses
`t = exp(-abs(delta)); 2*t/(1+t)` to avoid an overflowing positive exponential.

Do not import nuts-rs's default step-size ceiling of pi as a mathematical rule:
stability depends on curvature, scaling and integrator. Retain Alea's
`ln(10) + ln(initial_step)` dual-averaging center, which avoids overflowing an
intermediate `10 * initial_step`.
[Alea dual averaging](../crates/alea-mcmc/src/adapt.rs#L111).

**6. Reduce scalar bookkeeping and add SIMD where the numerical contract permits.**

Alea updates its acceptance mean with two divisions per attempted edge. Stan and
nuts-rs accumulate a sum and count, then divide when reading the mean. Choose that
formulation, keeping failed attempts as zero and preserving the treatment of
finite cutoff edges and discarded subtrees. It changes rounding, so adaptation
traces must be checked; this review found no demonstrated acceptance-bounds bug.
[Alea accumulation](../crates/alea-mcmc/src/nuts.rs#L545),
[nuts-rs accumulator](https://github.com/pymc-devs/nuts-rs/blob/a762aae513bf7bb5ccb27ffc9a938a923fc837ec/src/stepsize/dual_avg.rs#L83).

Avoiding an RNG call when selection probability is exactly zero or one is another
small valid optimization. It changes seeded RNG consumption, so assess transition
semantics rather than demanding unchanged historical draw hashes.

nuts-rs uses stable runtime CPU dispatch through `pulp`, with vectorized kernels,
multiple accumulators and fused multiply-add operations. That is useful engineering
to study, but its reduction order is different from Alea's ordered scaled turning
dots. First remove repeated metric applications, copies and divisions. Then
evaluate runtime-dispatched coordinate updates and metric kernels. Preserve the
existing scaled turning fallback and subnormal behavior; changing near-zero sign
decisions through reassociation needs a separate numerical comparison.
[nuts-rs CPU kernels](https://github.com/pymc-devs/nuts-rs/blob/a762aae513bf7bb5ccb27ffc9a938a923fc837ec/src/math/util.rs),
[Alea turning](../crates/alea-mcmc/src/nuts/turning.rs#L9).

The existing slice/iterator loops, generic static dispatch and buffer reuse are
appropriate Rust. Checked complete gradient writes and transactional live caches
are useful guarantees. Keep them while reducing duplicate internal work; blanket
unchecked indexing or deleting finite checks is not justified by these findings.

**Implementation order and evidence needed.**

1. Cache phase velocities; eliminate duplicate leaf turning checks. Verify the
   recursive oracle, extreme-scale turning, error/panic rollback and allocation
   contracts, then repeat matched-work native and Enzyme measurements.
2. Introduce private reusable phase handles. Preserve the same transition law and
   bounded storage, and measure bytes copied and cheap-target time per edge.
3. Add inverse-mass factor geometry for learned diagonal/dense metrics. Check
   independent energy/velocity/momentum identities, ill-conditioned fits and
   frozen checkpoint compatibility before measuring fitting and trajectory cost.
4. Simplify acceptance accumulation and deterministic selection branches as
   separate changes. Check arithmetic bounds and document RNG/rounding effects.
5. Run a small adaptation ablation using the existing Fisher implementation,
   matching starts, budgets, transforms, acceptance statistic and stopping rule.
   Report final geometry/step sizes, divergences, depth, retained calls, ESS/call,
   ESS/second and reference errors together. Add SIMD only where profiles still
   show a meaningful kernel cost.

There are two useful performance experiments: fixed-geometry/fixed-step runs
isolate implementation overhead; independently adapted runs measure the complete
workflow. Compare the former per actual edge and target call, and the latter per
useful inference. Use native analytic, generated Enzyme and shared BridgeStan
targets separately. The newly measured Enzyme gradient gains do not establish
sampler parity with either mature library.

Existing end-to-end evidence already prevents a blanket claim that Alea is slower
than Stan: the broad cohort's Alea/CmdStan diagonal median process-time ratio is
0.6917, while retained time is 0.9922. Against nuts-rs those ratios are 2.184 and
1.83. These include different policies and process/output costs and have no
quality filter. They establish the practical gap to investigate, rather than
identifying a universally superior library or numerical formulation.
[Full workflow comparison](benchmarks/m16/workflow-completion.md).

Validation for this review: all nine operation-count cases execute successfully;
the standalone consumer passes formatting and Clippy with warnings denied. A
second analysis of the original cohort reproduces the saved JSON exactly. This
does not rerun the earlier statistical suites or establish a new timing result.
