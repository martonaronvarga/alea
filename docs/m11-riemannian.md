# M11: Riemannian HMC and adaptive manifold experiments

M11 adds position-dependent geometry as an opt-in research path. It does not
change Alea's covariance-warmed Euclidean HMC/NUTS defaults. The implementation
follows the nonseparable Hamiltonian and generalized leapfrog in Girolami and
Calderhead's [RMHMC paper](https://arxiv.org/abs/0907.1100), the spectral map in
Betancourt's [SoftAbs paper](https://arxiv.org/abs/1212.4693), and the efficiency
objective in Wang et al.'s
[adaptive HMC/RMHMC paper](https://proceedings.mlr.press/v28/wang13e.html).

The 2026-10-16 promotion audit reopens the exact-target gate. The research
implementation is useful for studying RMHMC mechanics, but it does not yet perform
a per-proposal reverse/involution check after its finite-tolerance implicit solves.
It therefore remains S1 research even though bounded tests pass.

## Geometry and transition contract

`PositionMetric` evaluates a row-major positive-definite metric `G(q)` and all
coordinate derivatives `dG/dq_k` into caller-owned buffers. Checked evaluation
validates dimensions, finite values and symmetry, canonicalizing only symmetric
roundoff. Positive definiteness is established by the sampler's factorization.
Backend errors preserve their concrete type.

`Rmhmc` targets

```text
H(q, p) = -log pi(q) + 1/2 log det G(q) + 1/2 p^T G(q)^-1 p.
```

Its generalized leapfrog solves the momentum and position fixed-point equations
with a fixed iteration budget. Fixed work makes the forward numerical procedure
deterministic; it does **not** prove that the reverse solve selects the same
solution branch. The configured tolerance is a forward postcondition. A failed
postcondition rejects the proposal and records its stage, integration-step index,
iteration budget, scaled residual and tolerance. It is never retried until success.
Finite target failures become divergence events; concrete backend/factorization
failures remain typed.
Errors, panics and rejections do not publish proposal state.

Each attempted transition separately counts target calls, metric/derivative
evaluations, dense factorizations, triangular solves, inverse actions, dynamics
steps and both implicit iteration types. Constructor factorization is outside the
transition report and is recorded separately by the benchmark. Retained
transitions perform zero measured Rust heap allocations.

The current scalar dense factorization explicitly forms `G^-1`. This makes the
experimental path easy to audit, but it is not a claim that dense RMHMC is ready
for large dimension, BLAS, Faer or SIMD promotion.

### Open exactness gate

The ideal generalized leapfrog is designed to be reversible and volume preserving,
but a practical implicit equation can have no solution or multiple solutions.
Convergence to a small forward residual does not by itself establish numerical
reversibility. Alea currently has a low-dimensional signed-reversal regression and
a separate numerical unit-Jacobian check; neither enforces reverse support for each
proposal accepted by the production sampler.

Before exact-target promotion, the transition must perform a momentum-flipped
reverse/involution check (or implement another mathematically derived proposal
correction), reject failed reverse support transactionally, and validate the
volume/Jacobian assumption across nonlinear and multiple-solution cases. The
required distinction is described in Lelièvre, Santet and Stoltz,
[Unbiasing Hamiltonian Monte Carlo algorithms for a general Hamiltonian function](https://arxiv.org/abs/2303.15918),
and Brofos and Lederman,
[On Numerical Considerations for Riemannian Manifold Hamiltonian Monte Carlo](https://arxiv.org/abs/2111.09995).

## Metric families and derivatives

`LogisticFisherMetric` implements

```text
X^T diag(sigmoid(Xq) (1 - sigmoid(Xq))) X + prior precision
```

with its analytic coordinate derivatives. `HierarchicalNormalFisherMetric`
implements the expected information for `(mu, log sigma, theta_1, ..., theta_n)`
with explicit mean, log-scale and observation precisions. Both derivatives agree
with central finite differences in regression tests.

`SoftAbsMetric` maps every potential-Hessian eigenvalue with
`lambda coth(alpha lambda)` and evaluates the Fréchet derivative using spectral
divided differences. Small arguments use series expansions and large arguments
use their absolute-value limits. It requires an explicit dense potential Hessian
and all third derivatives: an HVP alone is mathematically insufficient. Jacobi
eigendecomposition is bounded and reports nonconvergence. Given the dense third-
derivative tensor, derivative reconstruction uses four matrix contractions per
coordinate, `O(d^4)` overall rather than an element-wise `O(d^5)` contraction.

`EnzymeManifoldModel` is the allocation-free typed boundary for generated score,
potential-HVP, Hessian and third-derivative callbacks. It deliberately keeps AD at
the scalar model boundary instead of differentiating validation and allocation
code. Analytic adapters are tested on stable; generated callbacks remain tied to
the separately pinned Enzyme toolchain.

The stationary gates cover a constant-metric Gaussian, a logistic posterior
against independent quadrature, and the hierarchical funnel's known
`v ~ Normal(0, 3)` and conditional standardized second moment under SoftAbs.
Generalized-leapfrog sampled reversal and numerical unit Jacobian are tested
separately. These are regression evidence, not a production-wide involution proof.

## Warmup-only Wang-style search

`ManifoldBayesTuner` reproduces the relevant bounded optimization pieces rather
than permitting continuing adaptation. Its observation is the paper's efficiency
objective

```text
||q' - q||^2 / sqrt(dynamics cost).
```

It searches at most 64 caller-supplied candidates with an RBF Gaussian process in
log step size and log trajectory time. Every candidate is observed before a
deterministic upper-confidence acquisition rule is used. Candidate means are
updated transactionally. The paper's diminishing adaptation probability is also
exposed as a reference function, but Alea never uses it for retained draws.

`RmhmcManifoldWarmup` can switch candidates without cold refactorization. Its
`finish` method consumes the warmup boundary, installs the selected fixed options,
and returns an ordinary `Rmhmc`; no GP state reaches stationary sampling. This is
a bounded finite-grid reproduction, not a bitwise copy of the paper's continuous
Bayesian optimizer.

## Executed comparison and decision

`alea-runtime/examples/m11_benchmark.rs` is a no-overwrite release harness. The
declared run used four chains, 500 warmup transitions and 2,048 retained draws on
the two-coefficient logistic posterior. All times include warmup. The same target
and seeds compare Euclidean NUTS, ChEES-HMC, fixed Fisher RMHMC and GP-adapted
Fisher RMHMC. The minimum bulk ESS and maximum rank/folded R-hat are across five
raw/quadratic observables.

Local artifact `/tmp/alea-m11-benchmark-20261015-v2.csv`, SHA-256
`ffa282458fa07afc8d9e88486d520dcaf32b033dff9208f1e4bad6ab3c52a4ba`:

| Method | max R-hat | min bulk ESS | total s | calls | min ESS/s | failures |
|---|---:|---:|---:|---:|---:|---:|
| NUTS | 1.00051 | 4,499 | 0.05371 | 54,745 | 83,764 | 0 |
| ChEES-HMC | 1.00133 | 3,013 | 0.00978 | 10,624 | 308,008 | 0 |
| fixed Fisher RMHMC | 1.00375 | 2,255 | 0.54134 | 81,540 | 4,166 | 0 |
| adapted Fisher RMHMC | 1.00045 | 4,495 | 0.64632 | 101,867 | 6,955 | 7 |

The tuner selected candidate 3 (`step=0.22`, `L=10`). It roughly doubled the
fixed RMHMC minimum ESS and improved minimum ESS per target call, so the adaptive
mechanism works on the declared target. It was nevertheless about 12 times slower
than NUTS and 66 times slower than ChEES in total time, while emitting seven
bounded implicit-solve rejections. Dense RMHMC, SoftAbs and the tuner therefore
remain research-only and are rejected for default promotion. The performance
result is an additional negative gate; it does not substitute for the open
numerical-reversibility gate. Fisher metrics remain useful explicit geometry
providers, not an implied sampler default.

Promotion can be reconsidered only after the reverse-support correction, generated
higher-derivative execution, broader stationary/SBC evidence and measured dense
factor/solve work are complete. SoftAbs additionally needs a practical dimension
ceiling or a replacement for its current `O(d^4)` derivative reconstruction.

Reproduce the comparison with:

```sh
nix develop .#stable -c cargo run --release --manifest-path crates/Cargo.toml \
  -p alea-runtime --example m11_benchmark -- /tmp/new-m11-results.csv
```
