# Fisher geometry and composable trajectories

Decision record, 2026-09-30. Implementation status is in [the roadmap](../roadmap.md).
The final architecture/modeling scope is specified in [the settled plan](settled-plan.md).
This record supersedes the earlier Stan-first/RMHMC-next ordering. CPU independent
chains are the first execution target. None of the proposed dynamic kernels below
is implemented yet. Enzyme remains an optional derivative provider, not a sampler dependency.

## Implemented ownership boundaries

| Concern | Rust boundary | Status |
|---|---|---|
| Target | `LogDensityGradient`, optional `HessianVector`, `BatchLogDensityGradient` in core | implemented |
| Geometry | `EuclideanMetric` in math, identity/diagonal/dense/spectral low-rank | implemented |
| Integrator | sealed `Integrator`, `Leapfrog`, `TwoStage::{OMF,BCSS}` in MCMC | implemented |
| Trajectory | `Hmc<T,M,I>`, consuming `Phase::step_with` | fixed HMC implemented; NUTS/WALNUTS planned |
| Adaptation | `HmcWarmup<T,I>`, `FisherHmcWarmup<T,I>` | reference covariance; experimental Fisher |
| Execution | `MarkovChain`, runtime streaming/collection | synchronous CPU; independent parallel CPU next |

Both warmups accept an integrator through `new_with_integrator`; completion
returns an ordinary immutable-configuration chain. `HmcTransition::integration_steps`
counts attempted macrosteps, **not forces**. Successful leapfrog costs one new
fused evaluation per edge; two-stage costs two. Initial target evaluation and
search costs are separate. Early numerical failures can invoke fewer target calls.
Instrumentation must count actual backend invocations for ESS/gradient reports.

Only complete phase endpoints can be saved. Failed internal stages consume the
phase token and cannot escape as candidates. This is a validity boundary, not
merely an allocation optimization. No trait here requires dense matrix storage.
The sealed integrator interface currently covers separable constant geometry;
WALNUTS and nonseparable manifold dynamics need additional contracts/proofs.

HVPs are explicitly `Hessian(U) v`, with `U=-log pi`; fused gradients remain
`grad(log pi)`. The checked wrapper detects partial writes, rejects shape/nonfinite
inputs before output mutation, and preserves typed model errors. Batch shapes
carry chain-major or coordinate-major layout; lane failures are independent.
These capabilities do not promise that arbitrary transformed/Enzyme targets
already implement second derivatives or batch compilation.

## Fisher derivation: Alea's sign and mass conventions

Write `s(q)=grad log pi(q)`, `G=M^-1`, and `q=mu+A z`, where `G=A A^T`.
The transformed score is `A^T s(q)` and the standard normal score is `-z`.
The squared score mismatch is

```
J(A,mu) = E || A^T s(q) + A^-1(q-mu) ||^2.
```

Differentiate with respect to translation: `mu = E[q] + G E[s]`.
Let `x=q-E[q]`, `y=s-E[s]`, `C=E[xx^T]`, `F=E[yy^T]`. Substitution gives

```
J(G) = tr(G F) + tr(G^-1 C) + 2 E[x^T y].
dJ  = tr((F - G^-1 C G^-1) dG).
G F G = C.
G = F^-1/2 (F^1/2 C F^1/2)^1/2 F^-1/2.
```

The cross term is independent of G; no assumption that the finite-sample mean
score vanishes is needed. Positive definite regularized C and F give the unique
SPD solution. For a Gaussian, `F=Sigma^-1`, `C=Sigma`, hence `G=Sigma`,
**M=Sigma^-1**, not the other way around. This algebra supplies independent
Riccati residual tests in `alea-math/tests/low_rank.rs`.

Restricting G to a positive diagonal gives

```
J = sum_i (g_i F_ii + C_ii/g_i) + constant
g_i = sqrt(C_ii/F_ii)
sigma_i = sqrt(g_i) = (C_ii/F_ii)^(1/4).
```

Do not confuse coordinate standard deviations sigma with diagonal inverse masses
g. Standardized positions are `x_i/sigma_i`, standardized scores `sigma_i*y_i`.
Multiplying both by sigma is incorrect. The objective follows the
[Fisher-HMC paper](https://arxiv.org/html/2603.18845v1); the derivation above fixes
the conventions used by Alea rather than copying ambiguous pseudocode.

### Reduced solve and storage

Choose orthonormal Q spanning centered standardized position **and score** columns.
Compute reduced scatter matrices `C_r=X_r X_r^T + gamma I` and
`F_r=Y_r Y_r^T + gamma I`; solve the same SPD equation. With reduced eigensystem
`G_r=V diag(lambda) V^T`, use `U=Q V` for retained directions:

```
G = S [I + U diag(lambda-1) U^T] S
M = S^-1 [I + U diag(lambda^-1-1) U^T] S^-1
p = S^-1 [I + U diag(lambda^-1/2-1) U^T] z, z ~ N(0,I)
velocity = G p
log det M = -2 sum(log sigma) - sum(log lambda).
```

Positive eigenvalues smaller than one are required. A positive-only `D+VV^T`
API cannot express them. Construction checks orthonormality; immutable fields
preserve it. Column-major U and separate contiguous q/p/gradient/velocity arrays
allow contiguous loops. No unsafe code or transient operation allocation is added.

Baseline hardening restricts the supported spectrum: positive finite eigenvalues
alone do not suffice for the subtraction-based f64 implementation. The constructor
rejects `kappa * (rank * max_gram_error + 16*EPSILON*(dimension+rank)) > 1e-6`,
including nonfinite estimates, where kappa includes the unit residual spectrum.
This is a conservative supported-precision policy, not a certified error bound.
It prevents demonstrated zero/negative metric actions; it does not guarantee that
arbitrary finite scales/momenta cannot overflow. No eigenvalue is silently clamped.
See [regressions and remaining limits](baseline-hardening.md).

The implementation uses twice-reorthogonalized modified Gram–Schmidt, a relative
1e-10 dependence threshold, and `faer` reduced eigensolves. It rejects nonfinite
arithmetic/decomposition failures. Raw-scatter ridge gamma=1e-5 follows the
[pinned nuts-rs implementation](https://github.com/pymc-devs/nuts-rs/blob/a762aae513bf7bb5ccb27ffc9a938a923fc837ec/src/transform/adapt/low_rank.rs).
If using sample covariance instead, convert the ridge too: gamma/(n-1) for unbiased
covariance. Dividing just one side or retaining gamma unchanged changes the estimator.
Pinned upstream fit-prefix traces are now executed; whole-warmup equivalence
remains unclaimed. See the scoped comparison below.

Numerical implementation: both reduced regularized scatters are divided by one
common positive maximum entry magnitude before the nested square-root solve.
This leaves `G F G = C` unchanged, avoiding intermediate scale overflow/underflow;
it does not repair unrepresentable raw scatter or arbitrarily ill-conditioned data.
Diagonal updates form regularized scatter sums in log space and apply the existing
scale bounds before exponentiation. Neither change adds target evaluations or
allocations to observation updates.

[Independent mathematical fixtures](../tools/fisher-reference/README.md) use a
120-digit Decimal full 2x2 solve through the C-oriented formula and closed-form
principal square roots, plus directly centered batch diagonal traces. They cover
common observation scales 1e-100/1e100 and are checked byte-for-byte in CI. They
validate the stated estimator in those cases, not complete upstream warmup policy
or a general high-dimensional error bound.

The fitted-spectrum grid also checks 240 commuting-scatter cases in 4/16
dimensions: three orthonormal directions with analytically known regularized
eigenvalues, varying rank caps/cutoffs, coordinate scales 1e-8/1/1e8 and common
data factors 1e-100/1/1e100 (ridge scaled quadratically). It checks metric actions,
mass determinant and momentum-energy identity; an unsupported fitted spectrum
must report the conditioning error. This does not cover every noncommuting or
near-dependent subspace (`alea-math/tests/fisher_spectrum.rs`).

The [noncommuting fit grids](fisher-noncommuting.md) add 864 planted-solution
cases with two/four interacting directions. Their accuracy failures motivated
evaluating the middle square root through Cholesky/SVD factors rather than an
eigensolve of the formed sandwich product. This preserves `G F G = C` and the
existing rejection/truncation policies; it is not an arbitrary-conditioning
certificate. Cold-fit timings from before this repair are historical.

[Executed nuts-rs fit-prefix fixtures](../tools/fisher-reference/upstream/README.md)
cover 18 cases in 2/8/16 dimensions with upstream-computed diagonal scales supplied
to Alea. A separate 80-digit dense oracle adjudicates conditioning-sensitive
differences: two upstream matrices miss the preset 1e-8 accuracy threshold, while
Alea passes that threshold on all cases. The original direct comparison failures
are documented, not hidden by widening tolerances. This validates the low-rank
estimator on those inputs, not identical diagonal regularization, initialization,
translation, collector selection or warmup schedules. No sampler policy changed.

Keep eigenvalues outside [1/2,2], then cap rank by descending |log lambda|.
This is a documented truncation heuristic, not a proof of the best rank-r fit.
Hot storage/work is O(d r+d); fitting costs O(d k^2+k^3),
`k <= min(d,2n)`. Output rank does not bound intermediate storage/work. Current
paired rings cap n at 20/160. Sketching is future work, not an implied property.

### Implemented experimental warmup policy

1. Evaluate the initial fused target once. Initialize with identity geometry by
   default. `MetricInitialization::ClippedScore` retains the historical absolute-score
   rule as an explicit experiment, with masses clipped to [1e-20,1e20] and mass one
   at zero score. Initialization is independent of subsequent paired estimation.
   Zero warmup preserves the identity metric and RNG. See the settled plan for
   the verified September initialization warning and robust initialization experiments.
2. Find an acceptable initial step with bounded one-macrostep probes. Backend
   errors are errors; numerical divergences give zero acceptance.
3. Observe actual post-transition cached states/scores, including rejected repeats.
   Joint moment preparation commits both signals atomically. No additional AD call.
4. Foreground/background periods are 10 before 30% of warmup and 80 afterward.
   Reset at the switch. A foreground keeps one to two periods; raw low-rank pairs
   are bounded separately to the latest two periods.
5. Update diagonal scales each observation. Add matching ridge in coordinates
   standardized by the initial fallback scales; clamp sigma to [1e-10,1e10].
   Constant windows retain the initial scales rather than divide zero by zero.
6. Refit low-rank corrections on period boundaries **and immediately before
   freezing**, so final spectral corrections use the final coordinate scales.
7. Freeze geometry at floor(85% n); restart dual averaging for the remaining phase.
   Consume/discard all adaptation state and publish only the final fixed chain.

Controller contract tests (`alea-mcmc/src/adapt/fisher/tests.rs`) observe completed
updates through a private generic callback; public `run` supplies a no-op and
stores no observer. Tests independently recompute centered batch scatters from
retained positions/scores, including rejected repeats. Spectral schedule checks
fit chronologically retained pairs with the separately validated math fitter,
not the controller's history ring. They cover identity/clipped-score initialization,
short schedules, period switches and wraparound, final off-boundary refits,
dual-averaging restart and bit-exact frozen geometry. Backend failures at the
switch/freeze boundaries return errors without a chain. A counting target checks
fused evaluation accounting for the tested finite Gaussian trajectories; this is
not a general equality between attempted steps and calls after partial failures.
An allocation test verifies public diagonal `run` allocates nothing after
construction. Low-rank fitting remains an allocating cold path. These checks do
not establish general adaptation efficiency or perpetual-adaptation convergence.

Extreme-start tests distinguish numerical safety from successful initialization.
For unit Gaussians they check finite coherent state, bounded evaluation counts and
frozen production geometry from both signs of positions up to magnitude 1e150,
under identity and clipped-score initialization. At such log-density magnitudes,
floating-point Hamiltonian subtraction can lose small energy differences; a
successful return does not certify typical-set recovery or useful sampling.
Unrepresentable initial densities are rejected before adaptation; a stiff Gaussian
whose required step lies below the default search floor exercises typed bounded
search exhaustion. These tests do not justify making clipped-score initialization
the default or silently accepting an unresolved search.

This uses dual averaging, not nutpie/walnutpie Adam; it is not claimed to reproduce
their complete warmup. Short budgets may contain no low-rank fit or no meaningful
retuning. No convergence guarantee comes from completing an iteration budget.

## Splitting derivation and what to optimize

Use formal drift/kick generators A and B. For the palindromic product
`exp(lambda h B) exp(h A/2) exp((1-2lambda)h B) exp(h A/2) exp(lambda h B)`,
BCH cancels even powers and gives

```
log Phi = h(A+B) + h^3 { alpha [A,[A,B]] + beta [B,[A,B]] } + O(h^5)
alpha = (6 lambda - 1)/24
beta  = (-6 lambda^2 + 6 lambda - 1)/12.
```

These are operator-commutator conventions; translating to a shadow Hamiltonian
requires fixing the Poisson/Lie sign convention too. Minimizing
`alpha^2+beta^2` yields lambda≈0.1931833275037836 (OMF2). That is an asymptotic
error-norm objective, not ESS. Lambda=1/4 equals two leapfrog half-steps;
lambda=1/2 collapses to one full leapfrog with an unnecessary internal evaluation.

For a unit-variance oscillator in scaled coordinates and nu=h*omega, multiply

```
A(t) = [[1,t nu],[0,1]], B(t) = [[1,0],[-t nu,1]].
P(nu,lambda) = B(lambda) A(1/2) B(1-2lambda) A(1/2) B(lambda)
             = [[a,b],[c,a]], det P=1.
```

Stable oscillatory dynamics require |a|<1. Set `cos(theta)=a`,
`chi=sqrt(-b/c)`; after L macrosteps the mean energy error for independent standard
normal initial coordinates is `sin^2(L theta)*(chi-1/chi)^2/2`.
This derives a finite-interval minimax surrogate
`rho=(chi-1/chi)^2/2`, with unstable coefficients assigned infinite cost.
BCSS2's ~0.211781 coefficient targets a finite interval rather than infinitesimal h;
see [Blanes, Casas and Sanz-Serna](https://arxiv.org/abs/1405.3153).

NUTS must only see complete macrosteps, not these internal stages. Compare
`LF(h/2)` twice with `TwoStage(h)` once, and count actual gradients. Good energy
conservation or acceptance alone does not establish exploration. In particular,
`L theta` close to a multiple of 2pi can make fixed-length Gaussian HMC nearly
stationary while acceptance is excellent.

An initial four-macrostep Fisher/OMF regression hit that regime (aggregate mean
0.137 versus a fixed 0.10 tolerance). Its purpose is now isolated geometry/moment
validation with one macrostep and the same tolerance, plus a direct check that
the final inverse mass recovers the known correlated covariance. This does **not**
certify the four-step configuration. Randomized lengths/dynamic trajectories and
modern diagnostics are required before promoting a general-purpose policy.

### Planned joint geometry/step/coefficient adaptation

Use explicit epochs: (a) update Fisher moments/geometry; (b) estimate an upper
curvature range using potential HVPs and restarts, counting all HVPs; (c) choose a
bounded lambda by minimizing the oscillator envelope over a conservative nu
interval; (d) reinitialize/tune step size conditional on that fixed coefficient.
Repeat only during warmup. Last epoch freezes G and lambda, then tunes/fixes h.
Power iteration alone is a lower estimate of the largest eigenvalue, not a certified
upper stability bound; require safety margins and divergence-based fallback.
Do not update coefficients or h based on the current retained state.

Keep OMF/BCSS references. Adaptive2/s-AIA, exact Gaussian+residual rotations,
multiple-time-scale force splits and force-gradient methods are future experiments.
Each needs independent reversibility, determinant/volume, endpoint and stationary
sampling tests. Negative-coefficient high-order compositions are not the default.
Dense whitening can make drift O(d), but dense transforms/pullbacks may still cost
O(d^2) per target call; it does not erase that cost.

## Algorithm comparison and benchmark order

| Kernel | Mechanism / validity obligations | Alea priority |
|---|---|---|
| Stan-style NUTS | global h, reversible doubling and generalized turns, multinomial selection | first dynamic oracle |
| nutpie-style Fisher NUTS | same dynamic-kernel class with paired-score preconditioning; whole warmup policy matters | first geometry comparator |
| WALNUTS | within-orbit dyadic refinement plus its special reversible trajectory/selection construction | next candidate CPU default |
| ChEES-HMC | ensemble adaptation of static randomized length; freeze adaptation | first batch-oriented alternative |
| ATLAS | curvature/local-step machinery with valid extended transition corrections | experimental multiscale comparator |
| DR-G-HMC | persistent momentum and staged delayed-rejection acceptance ratios | experimental multiscale comparator |
| MCLMC | isokinetic/microcanonical dynamics; distinguish unadjusted discretization bias from adjusted variants | separate approximate-inference track |

Primary specifications: [WALNUTS](https://www.jmlr.org/papers/v27/25-1452.html),
[ChEES](https://proceedings.mlr.press/v130/hoffman21a.html),
[ATLAS](https://arxiv.org/abs/2410.21587),
[DR-G-HMC](https://proceedings.mlr.press/v258/turok25a.html),
[unadjusted microcanonical work](https://arxiv.org/abs/2412.08876),
[Metropolis-adjusted microcanonical work](https://arxiv.org/abs/2503.01707).
These are not interchangeable implementations of an arbitrary adaptive integrator.

Benchmark matrix, in order: diagonal/rotated Gaussian (including extreme scales),
logistic regression, banana/funnel, hierarchical normal/stochastic volatility,
then the project-specific Wiener targets. Compare LF+covariance, LF+Fisher diagonal,
LF+Fisher low rank before crossing integrator factors. Next compare reference NUTS,
Fisher NUTS and WALNUTS; then ChEES/GHMC. Pin external versions and distinct RNG streams.

Report bulk/tail ESS/sec, ESS/gradient, gradients/draw, HVPs/draw, divergences,
depth/edge counts, warmup time, retained time, cold-factorization time, peak memory
and diagnostic failure rate. Do not silently discard nonconverged models. Until
rank-normalized diagnostics exist, label batch-means approximations as such.
Predeclare MCSE tolerances, chain counts and budgets; publish failures as well as
wins. Paper-reported median speedups are motivation, not Alea performance claims.
