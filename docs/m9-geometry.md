# M9: frozen learned and structured geometry

M9 adds opt-in constant-coordinate mechanisms. It does not change Alea's default
covariance warmup or multinomial NUTS, and it does not introduce position-dependent
geometry. Training and stationary sampling have different owning types.

## Frozen affine target

`LowRankAffineMap` represents

```text
x = b + S [I + U diag(sqrt(lambda) - 1) U^T] z,
```

where `S` is positive diagonal and the columns of `U` are orthonormal. Its
log-Jacobian is `sum(log(S)) + 0.5 sum(log(lambda))`.
`FrozenTransportTarget` evaluates `log p(x) + log|det dx/dz|` and pulls the score
back with the exact transpose Jacobian. Forward, inverse and pullback operations
cost `O(d r + d)`, reuse caller or chain-local storage, and construct no dense
matrix. Constructor validation, analytic roundtrips, finite-difference pullbacks,
Jacobian values, allocation checks and retained Gaussian moments cover the measure
change. This is a frozen affine instance of the broader transport-HMC idea; it is
not presented as a neural flow or nonlinear NeuTra implementation.

## Warmup-only learning

`AffineTransportLearner` owns a bounded ring of warmup states. `observe` validates
before committing and allocates nothing. `finish(self)` consumes the learner,
estimates coordinate scales, performs deterministic orthogonalized covariance
power iterations, and returns only an immutable `LowRankAffineMap`.

`learn_affine_transport` observes the initial state and each completed HMC state,
including repeats after rejection. Its result contains the frozen map and the last
warmup point in latent coordinates. The retained target has no training method. A
regression snapshots every map coefficient, runs retained HMC, and requires bitwise
equality afterward. The learner is deliberately a bounded empirical affine
experiment, not a claim that finite single-chain history identifies nonlinear or
tail geometry.

## Structured metrics

`BlockDiagonalMetric` composes validated diagonal and dense blocks without a
global dense representation. `SparseCholeskyMetric` stores a canonical lower
factor in compressed sparse-column form and implements momentum generation,
triangular solves and log determinant in `O(nnz)` work. Both implement the existing
`EuclideanMetric` trait directly. Dense-oracle algebra, log-determinant and
zero-allocation tests cover both paths.

These are constant Euclidean metrics. Sparse symbolic factorization and
position-dependent sparse metrics remain outside M9.

## Optional multi-path initialization

`multi_pathfinder` is a bounded initialization service, never a transition kernel.
It runs independent L-BFGS paths, constructs positive-definite local Gaussian
approximations from inverse-Hessian actions, estimates local ELBOs by Monte Carlo,
and pools path draws using exact mixture importance ratios. Actual target calls
include optimization, ELBO draws and proposal-pool evaluation.

Published multi-path Pathfinder uses Pareto-smoothed importance resampling. This
initial Alea experiment intentionally uses **raw** mixture importance resampling
and exposes that fact in `PathfinderReport`; it does not imply PSIS behavior. The
independent `score_initialize` baseline uses bounded Armijo score ascent and the
same cost fields. Backend failures propagate rather than becoming rejected
candidates.

Primary references are the
[Pathfinder paper](https://jmlr.org/papers/v23/21-0889.html) and
[NeuTra HMC paper](https://proceedings.mlr.press/v108/hoffman20a.html).

## Held-out benchmark and decision

`alea-runtime/examples/m9_benchmark.rs` is a no-overwrite release harness. It uses
four fixed seeds, 600 total warmup iterations and 1,024 retained draws per chain on
an eight-dimensional held-out Gaussian with heterogeneous scales and a rank-one
covariance multiplier of 100. It compares Fisher rank-4 NUTS, Fisher rank-4
WALNUTS-D, and rank-4 learned-transport NUTS. Transport spends 399 transitions
learning and 200 final transformed warmup transitions. Initialization, warmup,
target calls and retained time are charged.

Local release result from `/tmp/alea-m9-benchmark-20261015-v1.csv`:

| Method | max R-hat | min bulk ESS/s | min bulk ESS/call | calls | seconds |
|---|---:|---:|---:|---:|---:|
| Fisher NUTS | 1.0029 | 21,092 | 0.0492 | 42,541 | 0.0993 |
| Fisher WALNUTS-D | 1.0025 | 15,177 | 0.00507 | 241,065 | 0.0806 |
| learned transport NUTS | 1.0094 | 12,283 | 0.00914 | 66,621 | 0.0496 |

Transport reduced calls relative to WALNUTS but did not beat either comparator on
minimum ESS/second and used 1.57 times the calls of Fisher NUTS. One transported
variance estimate was 0.856 with MCSE 0.039 versus the analytic value 1, matching
the weak minimum ESS and close-to-threshold R-hat. The first method pays process-
cold Faer setup, so millisecond timing is not a portable backend claim.

The separate initialization comparison on the same target and starts was:

| Initializer | seconds | calls | squared Mahalanobis distance |
|---|---:|---:|---:|
| raw-resampling Pathfinder experiment | 0.000399 | 508 | 7.53 |
| score ascent | 0.00000165 | 13 | 0.118 |

M9's implementation and evidence work is complete, but the promotion gate is
negative. Frozen transports, structured metrics and Pathfinder remain explicit
opt-ins. Broader targets, shared ensemble learning and PSIS are future experiments,
not inferred successes.

Reproduce from `crates/`:

```console
nix develop -c cargo run --release -p alea-runtime --features faer \
  --example m9_benchmark -- /tmp/new-m9-results.csv
```
