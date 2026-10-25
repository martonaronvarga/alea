# Alea: settled product, architecture and implementation plan

Originally settled 2026-09-30; maturity review 2026-10-16. Canonical decisions
are linked from [roadmap.md](../roadmap.md).
Scope confirmed by the user: **a modeling framework with several distributions,
not only an inference engine**. Implementation status remains explicit; this
document does not claim that the future traits/algorithms already exist.

Execution-priority update: M4–M11 research implementations now exist, but most
advanced methods failed or have not cleared promotion. M13–M16 now complete the
boundary audit, modeling verticals, bounded reference sessions and immutable
experiment service. The next work is M17 per-crate release graduation. The
[maturity and promotion policy](maturity-and-promotion.md) defines S0–S4 and the
gates used below. The product scope is unchanged.

## 1. Product identity and limits

Alea is a Rust-native probabilistic modeling and inference laboratory: typed,
composable models; a curated distribution/scientific-kernel library; interchangeable
Monte Carlo components; and reproducible numerical/performance evidence.

Two equally supported entry points:

1. Native models assembled from typed parameters, distributions, factors, data
   plates and deterministic computations, lowered to an unconstrained target.
2. Existing analytic/compiled/foreign targets using the same fused interface.

The native model framework is not a demo layer or a deferred aspiration. DDM/Wiener
is a first-class application and performance workload. Enzyme/std::autodiff is the
compiled derivative route, with analytic primitives where appropriate; the sampler
does not depend on an AD implementation.

Near-term non-goals: a new textual language/parser, full Stan/PyMC distribution and
ecosystem parity, a replacement array/accelerator ecosystem, or a claim that one
sampler wins everywhere. A Rust embedded modeling API is in scope. Continuous
parameter samplers do not magically support discrete latent states; those require
marginalization or compatible future kernels.

No new crate split is needed: keep the nine enforced production boundaries.
Models/capabilities in core, distributions in distributions, numeric geometry in
math, inference mechanics/adaptation in MCMC, execution/monitoring in runtime,
AD/FFI adapters at their existing boundaries. Do not add empty public frameworks.

## 2. Target architecture: factor the transition, not just the integrator

| Component | Owns | Must not own |
|---|---|---|
| Target | fused logp/score; optional potential HVP/batch/structure | sampler state or global AD choice |
| Geometry | velocity, momentum law, metric determinants/actions | trajectory stopping |
| Dynamics | phase law and invariant extended measure | acceptance policy or warmup |
| Integrator | one numerical map and applicable map metadata | local adaptation or candidate selection |
| StepController | fixed/dyadic attempted-edge construction and reverse bookkeeping | hidden global warmup |
| Trajectory | fixed-length/tree construction and proposal-selection metadata | implicit acceptance assumptions |
| Correction | valid log acceptance/selection correction for its proposal family | pretending every map has unit Jacobian |
| Adaptation | initialization, estimators, schedules, scalar optimization | mutating a frozen production engine |
| Monitor | diagnostic summaries, progress/stop decisions | blocking chain hot paths |

Planned generic skeleton (design signatures, not current exports):

```rust,ignore
struct TransitionEngine<D, I, S, Tr, C> {
    dynamics: D,
    integrator: I,
    step_controller: S,
    trajectory: Tr,
    correction: C,
    // Chain-local live state, proposal and workspace are also owned here.
}

trait Dynamics {
    type State;
    type Workspace;
    type Error: std::error::Error + 'static;
    // Defines refresh, invariant density and admissible phase representation.
}
trait Integrator<D: Dynamics> {
    type MapInfo;
    // advance consumes a ValidPhase token and returns a CompleteEdge token;
    // failure cannot publish an intermediate stage.
}
trait StepController<D: Dynamics, I: Integrator<D>> {
    type Edge;
    type Workspace;
    // Builds an edge, including forward/reverse step-choice metadata.
}
trait Trajectory<D: Dynamics, I: Integrator<D>, S: StepController<D, I>> {
    type Proposal;
    type Workspace;
    // Returns a proposal plus sufficient selection/path metadata.
}
trait Correction<D: Dynamics> {
    type Proposal;
    type Guarantee; // ExactTarget or ApproximateTarget, sealed capability markers
    // Correction accepts only its compatible proposal family.
}
```

The engine's bounds tie `C::Proposal` to `Tr::Proposal`. Error types retain backend
sources; no boxing/dynamic dispatch is required. Workspace construction is cold;
`step` reuses it. Convenience constructors hide the generic arity. A marker records
a supported guarantee under stated assumptions; it is not a proof for arbitrary
user code or a claim of finite-run convergence.

First concrete composition: `CanonicalHamiltonian<Target,Metric> + Leapfrog +
FixedStep + FixedLength + CanonicalMetropolis`. Extract the existing HMC code into
these components; do not keep a duplicate legacy path. Preserve all old reference
tests, RNG order where the algorithm is unchanged, and the allocation budget.
Rename today's coefficient-only `Integrator` to `SymmetricSplitting` when the
dynamics-aware integrator lands; do not imply its current sealed trait is the
final extension API. Provide an external implementation test suite before opening
new integrator/dynamics implementations to downstream research code.

For NUTS, proposal selection itself contributes to correctness; a generic endpoint
MH accept/reject is not its replacement. WALNUTS requires a compatible controller,
trajectory and biased-progressive selection correction, not `WalnutsIntegrator`.
Unsupported combinations should lack implementations, not silently approximate.

Later isokinetic/MAMS uses a different dynamics and correction family. A
non-symplectic map is not automatically non-volume-preserving; derive the actual
Jacobian/proposal correction from that method, rather than infer it from the label.
Unadjusted MCLMC has an explicit approximate-kernel type and benchmark category.
Unadjusted *warmup* may eventually hand off a validated state to an exact kernel,
discarding all adaptation draws and rebuilding the production cache/geometry.

## 3. Initialization and continuous adaptation

The [September 2 correction](https://statmodeling.stat.columbia.edu/2026/09/02/confirmation-bias-in-software-testing-walnutpie-edition/)
reports a Walnutpie diagnostic-interface bug and sensitivity of score-based
initialization. It does not invalidate the Fisher objective, nor prove that every
earlier paper benchmark has the same bug. Alea will independently validate both
inference and diagnostics before accepting speed claims.

Implemented now: `MetricInitialization::Identity` is the default, with an explicitly
experimental `ClippedScore` option. Paired-score estimation remains independent.
Clipping to a huge numeric range only prevents some overflow; it does not solve
tail sensitivity. A regression checks extreme scores leave identity initialization
unchanged. Future comparisons include user-supplied validated geometry, multiple
bounded probes and log-space shrinkage toward identity. Account for every extra
probe; reject strategies whose gains disappear when initialization is timed.

The implemented Fisher estimator uses finite foreground/background windows and
freezes before retained draws. It is **not continuous diminishing adaptation**.
The next estimator supports paired weighted central moments. For either signal
x (position or score), let `delta=x_t-m_(t-1)` and validated `0<alpha_t<=1`:

```
m_t = m_(t-1) + alpha_t delta
C_t = (1-alpha_t) [C_(t-1) + alpha_t delta delta^T]
w2_t = (1-alpha_t)^2 w2_(t-1) + alpha_t^2.
```

Initialize the first observation with alpha=1, mean=x and scatter=0. `C_t` is
normalized weighted scatter; `C_t/(1-w2_t)` is its usual finite-weight covariance
correction when defined. Apply identical weights/normalization to positions and
scores. The diagonal Fisher ratio cancels a common normalization; ridge terms do
not, so define them in standardized coordinates before implementation. Both signals,
weight state and count must commit transactionally after finite checks. For low-rank
fits, use a bounded weighted sketch or explicitly windowed approximation; a finite
unweighted ring is not an exact infinite-history weighted estimator.

Candidate schedule: `alpha_t=(t+t0)^(-gamma)` for `1/2<gamma<=1`, t0>=1, after
the special first observation. Its weights diminish with divergent total weight
and finite squared-weight sum. Equivalently investigate
`alpha_t=1-exp(-1/tau_t)` with `tau_t` growing like `t^gamma` (use expm1 for
small arguments). Keep `WeightSchedule` independent from the estimator.

Those Robbins–Monro conditions are **not sufficient by themselves** to prove an
adaptive MCMC kernel ergodic: containment/stability and diminishing *transition
kernel* changes matter too. Hard rank thresholds can introduce discontinuities;
constant forgetting never becomes diminishing. Initial production policy remains
warmup-only adaptation followed by freeze. Endless adaptation needs its own target
assumptions, bounded controls, rank policy, theory and stress tests.

### Scalar step adaptation

Use `eta=log epsilon`, error `g=target_acceptance-observed_acceptance`.
Scalar optimizers minimize this root objective with negative feedback:

```
m_t = beta1 m_(t-1) + (1-beta1) g_t
v_t = beta2 v_(t-1) + (1-beta2) g_t^2
eta_(t+1) = eta_t - learning_rate_t * mhat_t/(sqrt(vhat_t)+epsilon_adam).
```

Compare dual averaging, Robbins–Monro and bias-corrected Adam with a validated
decaying learning rate. Adaptation sees a typed observation: finite acceptance,
numerical divergence, invalid statistic, or backend failure. A Metropolis rejection
is not necessarily zero acceptance probability. A numerical divergence supplies
zero acceptance by explicit policy; NaN/Inf statistics and backend failures return
errors without changing optimizer state. Step bounds, iteration overflow and
zero/underflow denominators have regression tests. Adam is not adopted merely
because it is fashionable or because another implementation reports a win.

## 4. Modeling and distributions are a delivery track

Current foundations: `ConstrainedModel`, `AnalyticModel`, `SumModel`,
`TransformedTarget`, `ParameterLayout`, opaque derivative primitives, Gaussian,
Wiener4/5/7 and latent/state-space contracts. Preserve all of them. The missing
work is a reusable distribution API and ergonomic typed model assembly, not another
unconstrained-density abstraction.

B1, distribution contracts and first tier:

- Separate `log_prob` (normalized density/PMF) from unnormalized factors and from
  RNG sampling capabilities. Separate support/event shape from observation batch shape.
- Checked parameter constructors; distinguish invalid parameters from legitimate
  out-of-support observations. Backend failure is never an ordinary support zero.
- Scalar Normal, Student-t, LogNormal, Exponential, Gamma, Beta, Bernoulli, Binomial,
  Poisson; preserve optimized existing Gaussian/Wiener methods. Names must distinguish
  a normalized distribution from today's unnormalized standard-Gaussian target.
- Each ships boundary/tail/normalization reference values, derivative tests with
  respect to continuous parameters, sampling tests where sampling is exposed, and
  zero-allocation evaluation benchmarks.

B2, typed assembly:

- Typed parameter handles/support descriptors mapped once to flat layout slots.
- Priors and likelihood factors, tuple/product composition, plate/reduction over
  data, deterministic transforms and hierarchical/shared parameter references.
- Lower to one compiled/analytic fused target and chain-local workspace. Exactly
  one transform Jacobian per parameter, not per use/factor. Typed views must not
  introduce mutable aliases of sampler state.
- Prior/posterior predictive sampling and generated quantities are separate from
  the gradient hot loop. Coordinate names/shape and optional block structure survive
  lowering for output and later structured geometry.
- Initial examples: normal regression, hierarchical logistic regression, hierarchical
  DDM. Compare generated targets with manually written analytic references.

B3 adds multivariate Normal, Dirichlet, categorical/multinomial and Cholesky-based
correlation priors, then stable finite-mixture marginalization. No broad catalogue
without per-distribution gates. A distribution need not offer every operation:
log-CDF/quantile/reparameterized sampling are optional explicit capabilities.

## 5. Concrete implementation slices and dependencies

| Slice | Current result | Remaining exit gate |
|---|---|---|
| A0–A3 reference mechanics/adaptation | S2 core/HMC/covariance; Fisher S1 | Preserve reference behavior; enforce public maturity and exact/approximate boundaries in M13. |
| A4 reference NUTS (M5) | S2 with independent Stan/BlackJAX/nuts-rs evidence | Turnkey model/session, SBC, portability, downstream and API-release gates. |
| A5 runtime/diagnostics (M6) | S2 bounded CPU runtime and modern diagnostics | M15 adds durable output/restart and bounded fault/memory tests; platform-wide soak and release compatibility remain. |
| A6 WALNUTS (M7) | Correct opt-in implementation; negative performance promotion | New predeclared held-out efficiency protocol only; preserve reverse construction. |
| A7 alternatives/geometry/integration (M8–M10) | Research/validated components; no default promotions | Method-specific re-entry gates in the maturity policy; no suite result promotes unrelated methods. |
| A8 Riemannian methods (M11) | S1 research | Numerical reverse-support/involution, volume/Jacobian, independent reference, derivative and total-cost gates. |
| B1/B2/B3 modeling | M14 completed in declared regression scope | Typed lowering, normalized tiers, predictions and three downstream/SBC verticals; see docs/m14-modeling.md. |
| V1/V2 validation | M16 completed; S2 opt-in experiment utility | Source-bound 252-job report, held-out corpus, three SBC/predictive verticals and multi-host tracking; one physical host measured, negative promotion. See docs/m16-experiments.md. |
| R1 public surface | M13 completed | Item/feature maturity, downstream/compile-fail laws, digest-bound API adjudication and exactness-pending RMHMC containment. |
| R2 session/runtime | M15 completed in bounded regression scope | Validated reference sessions, partial typed outcomes, bounded/durable output, frozen replay and resource/fault gates; see docs/m15-sessions.md. |
| R3 release | Pre-alpha, `publish = false` | M17 per-crate S3 cycle, semver/MSRV/platform/docs/artifact gates. |

Ordering from the current state is M17 (M13–M16 complete), with M12 SMC
beginning after shared B1 support/event/batch semantics settle. Small vertical
slices are preferred to a repository-wide framework rewrite. Do not implement the
nine-component abstraction ahead of extracting it from concrete validated uses;
do not retain pre-alpha public exposure merely for backwards compatibility.

Modern diagnostics precede automatic stopping and performance promotion. Publish
latest-only immutable summaries to a separate monitor; sampler workers never wait
for it. Minimum post-warmup draws and monitored observables are explicit. Log-density
R-hat alone is not a sufficient stopping criterion. Ragged-chain calculations must
have independent references or be rejected rather than silently truncated/reweighted.
Stein score/moment summaries are optional diagnostics only: integration-by-parts
boundary conditions and finite moments must hold for the target.

ChEES/GHMC, DR-G-HMC, ATLAS, MAMS, transports, sparse/block geometry, Pathfinder,
advanced splitting and RMHMC now have research implementations of differing scope.
None is a dependency of the native modeling framework, and none becomes stable by
being publicly exported. The current RMHMC proposal has an open finite-tolerance
reverse-support gate. Accelerator execution and the compiled target ABI remain
unimplemented; the ABI requires two real consumers before its surface is frozen.

## 6. Benchmark protocol and falsifiable success

V1 manifest records git/toolchain/dependency revisions, CPU/thread settings, target
data/hash/dimension/parameterization, seeds, initial states, derivative backend,
geometry/algorithm policies, stopping rule, warmup and retained budgets. Results
record cold setup, warmup and retained time, actual value/gradient/HVP calls, metric
actions, allocations/peak memory, divergences, acceptance/energy errors and trajectory
work. V2 adds independently validated bulk/tail ESS, rank-normalized R-hat and MCSE.
Never derive exact call counts from macrostep count after early failures.

Corpus tiers: (1) controlled Gaussian scales/correlations; (2) banana/funnel/logistic;
(3) hierarchical normal/logistic and stochastic volatility; (4) latent Gaussian and
Wiener/DDM; (5) selected versioned PosteriorDB models. Use separate tuning/development
and held-out reporting subsets. Freeze parameters and diagnostics before the held-out
comparison. Match model measures and count all initialization/adaptation costs.

The current product goal is a reliable, useful baseline to build on. Baseline
acceptance requires correct densities/gradients, useful posterior accuracy and
mixing, standard warmup, clear failure reporting and competitive resource costs
against mature implementations. Performance parity is an acceptable baseline;
comparable inference quality reached sooner is an improvement. A mandatory 1.5x
ESS/second advantage is not a baseline or milestone-completion requirement.
Evaluate representative models with sensible adaptation/parameterization, and
retain pathological cases as visible stress tests. Separate efficiency warnings
(such as depth saturation alone) from insufficient statistical quality.

M16's sealed `reference-v1` experiment retains its original strict superiority
gate and all failures. Its fixed-step, mostly identity-geometry run is not an
adapted production-baseline verdict. A new baseline protocol must fix controller
timing resolution, compare matched inference quality and report cold-start and
steady-state costs separately; do not reinterpret the old results by editing gates.

Research/default-replacement claims require mathematical/reference/statistical
gates first. Suggested thresholds for those stronger claims, to preregister in
their own experiment manifests:

- Performance: at least 1.5x median total-time efficiency on a declared held-out
  workload family versus the strongest selected baseline, with uncertainty intervals
  and no worse diagnostic-failure rate. Publish every target, including losses.
- Research usability: implement and validate one new metric and one trajectory in
  isolated modules without changes to target, unrelated algorithms or runtime.
- Modeling/domain utility: deliver the three B2 examples, posterior prediction and
  an independently validated hierarchical DDM workflow with documented costs.
- Validation utility: another user regenerates the cross-engine report from one
  manifest; injected wrong gradients, biased/stuck chains and diagnostic bugs fail CI.

After A6/B2/V2, require at least two demonstrated benefits (performance, research
usability, modeling/domain utility, reusable validation) before widening the advanced
algorithm catalogue. These are decision criteria, not current achievements. Correct
but uncompetitive configurations remain useful references; they are not advertised
as production defaults. No speedup from an external paper substitutes for this evidence.
