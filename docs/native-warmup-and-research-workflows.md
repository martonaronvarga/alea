# Native warmup and repaired research workflows

The reference session's NUTS policy now adapts with NUTS trees. The previous
policy tuned a short static HMC trajectory before switching to NUTS. That was a
poor initialization policy for the earnings/height and `sblri-blr` starts in the
sealed version-two PosteriorDB experiment. This change replaces the warmup policy;
stationary HMC/NUTS dynamics and frozen checkpoint encoding remain compatible.
Historical benchmark records retain their original source and policy. Version 3
manifests explicitly record the reference-warmup change relative to version 2.

`NutsWarmup` owns windowed covariance, reasonable-step search and dual averaging.
It uses the existing window schedule and covariance regularization. Search probes
fresh momentum at the same cached point and treats finite energy decreases as
acceptable. A sharply downhill trajectory can therefore reach a useful step
without being rejected by static HMC's symmetric absolute-energy cutoff. Tree
acceptance averages actual attempted edges; a late divergence does not erase the
valid observations preceding it. Bounds, search work, divergences, depth limits
and log-density extrema are reported. Cancellation publishes no partly adapted
chain. Observers see coherent live states, including repeats after rejection.
Zero warmup performs no search or RNG work.

Executed Stan 2.38 and BlackJAX 1.5 fixtures triangulate 400 step/covariance
recurrences with a largest scaled discrepancy of 3.44e-14. The 1711 schedule
observations retain four short-budget window-end differences. Alea keeps its
BlackJAX-compatible schedule; no whole-workflow equivalence is claimed. The
[reference harness](../tools/stan-adaptation-reference/README.md) records the
setup/reset scope and all differences.

`FisherNutsWarmup` changes the trajectories supplying adaptation observations.
Paired moments, fitting equations, regularization, low-rank selection and the
30% reset/85% geometry-freeze schedule retain their existing policy. Diagonal,
rank-capped, Robbins–Monro and Adam variants remain opt-in research components.

The new benchmark workflows use the following finite policies:

| Workflow | Pilot and adaptation | Retained policy |
| --- | --- | --- |
| Covariance/Fisher NUTS | Native NUTS throughout | Frozen NUTS |
| HMC/OMF/BCSS/GHMC/DRGHMC | 75% NUTS, 25% named-kernel step adaptation | Frozen geometry and step; DRGHMC adapts first-stage probability and charges ghosts |
| WALNUTS | Full native NUTS pilot | Existing deterministic or randomized refinement law |
| ChEES | Half NUTS, shared frozen diagonal affine map from cohort slow-window states, half shared ChEES | Frozen parameters; maximum 1024 macrosteps per trajectory |
| Affine transport | Half NUTS, slow-window states fit one bounded affine map, half transformed NUTS | Frozen map and NUTS |
| MAMS | 75% NUTS, frozen diagonal affine preconditioning, 25% corrected step adaptation to 0.8; duration from actual pilot latent scales | Frozen step and `ceil(duration/step)` macrosteps, capped at 1024 |
| MCLMC | Same preconditioning, centered signed local-error coefficient variance tuning | Explicit approximate target; refresh radius from actual centered pilot latent variances |
| Joint s-AIA NUTS | Native covariance NUTS; curvature/coefficient epochs at initial/slow-window boundaries | Coefficient frozen before final-fast tuning; dimensionless policy step capped at 3.8 |

MCLMC reports its omitted correction error without inventing an acceptance
probability. Refinement tests for this implemented symmetric integrator find
squared local error proportional to `h^6` at fixed integration count, both with
and without angular refreshment. The warmup target is centered correction-error variance 5e-4 per dimension per
transition; a scaled EW recurrence estimates the signed `error/h^3` coefficient
variance without overflowing `error²/h^6`. Numerical warmup attempts halve the step
and reset the estimator, preserving the live cache and persistent direction.
Their work is counted. Retained MCLMC errors stay explicit; there is no hidden
rejection or retry in its unadjusted sampling law. This is a practical error policy, not a finite-step bias guarantee.

Frozen affine targets expose successful nonfinite density/score and numerical
map-overflow outputs to the outer checked evaluator. It rejects them as numerical
evaluation failures while preserving valid cached points. Concrete backend and
structural errors retain their types. MAMS turns numerical proposal failures into
transactional rejected transitions; retained MCLMC propagates them.

## Pathfinder

The first robust L-BFGS trial scales its coordinate displacement before any
curvature history exists. This resolves sharply scaled initial scores within the
existing backtracking budget; it does not increase optimizer iterations.

Raw mixture ratios and generating-path PSIS are separate explicit modes. Both
robust workflows backtrack on support-zero candidates and retain numerical path
outcomes. Concrete backend and shape failures remain typed errors. Nonfinite
successful outputs fail that numerical path, whose work/outcome stays reported
while other paths survive. They never supply a selected cached point. A
negative-infinite proposal density contributes zero weight; it is never installed
as a valid cached sampling point. Actual target work includes failed paths, local
ELBO trials and proposal-pool draws.

PSIS uses the Zhang–Stephens generalized Pareto fit and Stan2.38 tail policy.
Small or tied tails are explicitly reported as unsmoothed. Reports include path
outcomes, attempted draws, zero-mass draws, weight ESS, tail length and fitted
Pareto shape. Executed Stan fixtures check weights and shape; support-zero and
backend-failure tests cover the numerical recovery boundary. This initializer
changes starting points and does not supply a stationary MCMC transition.

## Full ATLAS

`atlas_sampler::Atlas` implements preliminary NoUT, one delayed proposal and the
short-trajectory failure branch, using `constant_trajectory=2` and `probabilistic=1`. Baseline
metric and step are frozen. `AtlasWarmup` records a bounded empirical distribution
of preliminary lengths and freezes its 10–90% trimmed observations. Every forward,
reverse and ghost leg has a maximum 1024 steps. A larger delayed trajectory is an
explicit work-limit rejection. Actual work counts include every discarded leg, with separate forward, reverse
and ghost integration counters. DRGHMC similarly separates first, delayed and
ghost work in warmup and retained telemetry.

Local step distributions use 14 deterministic HVP calls at each endpoint. Their
lognormal width is `log(1.2)`. This replaces the author's random BFGS/power policy;
no unaccounted random curvature fit enters the proposal density. It is a local
policy estimate, not a global curvature certificate.

Acceptance follows [Appendix E.2 of the ATLAS paper](https://arxiv.org/html/2410.21587v1#A5.SS2).
For the delayed branch the mass at a phase and selected baseline length is
`exp(-H) * q(n|phase) * (1-alpha_first)`. The reverse/forward ratio includes these
actual selection densities, the rejection masses and both local-step densities.
The implementation at author revision
`a03c0608984a7720102230ffc19465af8868160b` instead subtracts preliminary Hastings
ratios in that term. Alea follows the paper's density equation. A balance fixture
with unequal selection/rejection masses tests this distinction; no bitwise author
trajectory-equivalence claim is made. Reverse branch/support checks are explicit.
Gaussian regressions exercise every branch, and measured transitions allocate no
storage. Wider PosteriorDB results remain descriptive evidence, not promotion.

## RMHMC

The new benchmark-only C++ helper instantiates the concrete Stan model and calls
`grad_hessian` for Hessians and third derivatives, including the unconstraining
Jacobian exactly once. Its owned Rust binding validates ABI/dimensions, serializes
calls, keeps code loaded through model destruction and preserves error text.
Independent earnings-regression formulas check every Hessian entry and all third
tensor permutations. No finite-difference fallback is allowed in version 3. The
smoke runs still rejected some AD tensors under the unchanged
64-times-machine-epsilon symmetry contract; passing the independent analytic
accuracy fixture does not imply satisfying that tighter floating-point consistency
requirement. These remain visible research failures.

The opt-in Newton profile uses analytic residual Jacobians, pivoted linear solves,
maximum 100 iterations and an actual-equation residual tolerance 1e-12. Full reverse
replay checks every phase boundary and intermediate half-momentum at 1e-8. Reports
charge Jacobians, linear systems, inverse actions, factorizations and replay work.
A corrupted-half-momentum test demonstrates that endpoint checks alone are
insufficient.

`ExactnessPendingRmhmc` retains that name and guarantee. Finite Newton termination
selects a numerical map, whose Jacobian is not automatically the exact implicit
map's symplectic Jacobian. A loose-tolerance counterexample for `G(q)=exp(q)` has
Jacobian determinant approximately 0.943, despite satisfying its residual policy.
This does not quantify the error at 1e-12; it disproves a general claim that finite
termination supplies unit volume. Reverse replay improves support checks but does
not supply the missing volume correction. The [explicit finite-map derivation](rmhmc-finite-map-analysis.md), independent
72-case solver/tolerance ladder, alternative-root fixtures and singular-system
regressions document this negative finding. The volume obligation remains open
and does not block the reference NUTS repair.

## Descriptive experiments

The version 3 protocol keeps 20 posteriors, four chains and three repetitions with
512 warmup plus 512 retained draws. The v3 corpus corrects the legacy `sblri-blr`
family tag to Gaussian linear regression while retaining identical
model/data/reference bytes. It has 28 configurations, including separate raw/PSIS
Pathfinder and full ATLAS. RMHMC keeps its smaller 32+64 budget and
maximum dimension 10, visibly unmatched. A separate six-target protocol uses
1000+2000 for earnings/height, `sblri-blr`, centered schools, both mixtures and seeds.
It includes baseline/Fisher, ATLAS, Stan, nuts-rs and BlackJAX. Preparation and
compilation finish before sampler timing; long runs belong on dusk or CI.
There is no aggregate score, promotion decision or 1.5x ESS/s gate.

The [verified pipeline smoke run](benchmarks/m16/posteriordb-v3-smoke/README.md)
contains 56 jobs on two posteriors: 51 completed and five failed. Both ChEES cases
and all four Alea raw/PSIS Pathfinder cases complete after the numerical recovery
repairs. Four RMHMC jobs fail strict symmetry checks and one CmdStan Pathfinder
initialization fails. Its 128+64 budget tests workflow execution and artifact
reproduction; it does not establish convergence or performance. The broad and
longer focused experiments use separate sealed records.

The first diagnostic pilot used the four original starts/seeds on earnings/height
and `sblri-blr`. All covariance/Fisher variants completed. For earnings/height,
covariance-diagonal NUTS's largest mean error was 0.041 reference SD (previously
about 49), with R-hat 1.007 and minimum bulk ESS 535. For `sblri-blr`, its spread error
was 0.043 and largest mean error 0.154 reference SD; Fisher-diagonal had R-hat 1.003
and minimum bulk ESS 2190. These pilots show the extreme initialization failures
are repaired; they are not a sealed timing comparison or a universal convergence
claim. Version 3 broad/focused results must be assessed independently.
