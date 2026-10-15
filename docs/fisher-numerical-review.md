# Fisher numerics: mathematics and reference implementations

Review and bounded repair, 2026-10-05. This is a source comparison, not an
executed cross-library sampling benchmark. Fisher remains experimental.

## Mathematical contract

For centered, standardized position columns `X` and score columns `Y`, Alea
uses raw scatters `C = X X^T + gamma I`, `F = Y Y^T + gamma I`. The SPD inverse
mass `G` solves `G F G = C`. This is the regularized Fisher optimality condition;
the paper writes it using covariances, so translating its ridge to raw scatters
requires accounting for the covariance denominator. See Seyboldt, Carlson and
Carpenter, [§2.4 and Appendix C](https://arxiv.org/html/2603.18845v1).

Do not mix mass `M`, inverse mass `G`, and coordinate scales `S`: Alea stores
`G = S [I + U diag(lambda - 1) U^T] S`. In one dimension the inverse mass is
`sqrt(C/F)`; the coordinate scale is its square root. The derivation and
translation treatment are in [Fisher geometry](fisher-and-trajectories.md).

Our rescaling follows directly from that equation. For any common positive `a`,

```
X' = X/a, Y' = Y/a, gamma' = gamma/a^2
C' = C/a^2, F' = F/a^2
G F' G = C'  <=>  G F G = C.
```

Scaling the two scatters independently, or leaving the ridge unscaled, does not
have this property. In particular, adding jitter or clipping eigenvalues is a
change to the computed problem, not merely a rearrangement of this identity.

## Reference comparison

The following code revisions were inspected; they are not new dependencies.
Links are pinned rather than relying on moving default branches.

| Reference | Relevant behavior | Consequence for Alea |
|---|---|---|
| [Stan `covar_adaptation.hpp`](https://github.com/stan-dev/stan/blob/babdd0e67933e92355207a7cfe4c1dd07459e8c6/src/stan/mcmc/covar_adaptation.hpp) | Welford covariance, sample-count-dependent shrinkage toward `1e-3 I`, explicit nonfinite failure. | This is a covariance estimator, not the Fisher solve. Preserve explicit errors; do not substitute its ridge formula. |
| [nutpie's nuts-rs `low_rank.rs`](https://github.com/pymc-devs/nuts-rs/blob/a762aae513bf7bb5ccb27ffc9a938a923fc837ec/src/transform/adapt/low_rank.rs) | Joint position/score subspace via SVD and pivoted QR; raw scatters divided by gamma plus identity; nested eigensolves for the geometric mean. | Same regularized equation, different factorization. Existing pinned fit-prefix fixtures remain independently checked against Decimal, not accepted solely by upstream agreement. |
| [BlackJAX `metric_estimators.py`](https://github.com/blackjax-devs/blackjax/blob/6f0edc214bf3329989153d5a2aba00a2f86edf6c/blackjax/adaptation/metric_estimators.py) | `_spd_mean` floors intermediate eigenvalues at a spectrum-relative epsilon; rank selection scores `abs(lambda-1)`. | Flooring is an explicit repair policy. Do not import it into Alea's no-clipping contract. Alea's `abs(log(lambda))` truncation is also a distinct heuristic. Neither library is a mathematical oracle for the other. |
| [JAX `lax.linalg`](https://github.com/jax-ml/jax/blob/4435223ffe84464f6c73415ca34f45f8c1815b22/jax/_src/lax/linalg.py) | `eigh` symmetrizes input by default; lower-triangle decomposition without symmetrization is a different convention. | Record symmetry conventions in comparisons. This patch does not change Alea's lower-triangle convention. |
| [walnutpie diagonal adaptation](https://github.com/flatironinstitute/walnutpie/blob/5710b0334ba69cf0caa35fc02a2338b61d756953/include/walnutpie/adaptive_walnuts.hpp) and [trajectory selection](https://github.com/flatironinstitute/walnutpie/blob/5710b0334ba69cf0caa35fc02a2338b61d756953/include/walnutpie/walnuts.hpp) | Diagonal position/score variance ratio; macrostep refinement with a reverse step-selection check. | Diagonal Fisher geometry and locally adaptive trajectories are separate mechanisms. A successful smaller-step replay is not permission to retry production HMC until it succeeds. |

JAX comparisons must explicitly enable float64: its
[default dtype policy](https://docs.jax.dev/en/latest/101/default_dtypes.html)
otherwise permits materially different precision. Existing executed BlackJAX
endpoint fixtures pin release 1.5 and float64; the newer source inspection above
does not upgrade or regenerate those fixtures.

For trajectory changes, the [WALNUTS paper, §3 and Appendix B](https://arxiv.org/html/2506.18746v1)
treats local step selection and reversibility together. Its theory is not a
justification for ad hoc retries in a fixed-HMC kernel. Our funnel replay remains
diagnostic; this numerical patch makes no divergence-free sampling claim.

## Reproduced failure and repair

The previous implementation formed squared column norms and reduced Gram
matrices *before* common scatter normalization. A representable, well-conditioned
problem could therefore fail before reaching that safeguard. With rows
`+/- (4,4,0)*2^510`, scores `+/- (1,1,0)*2^510`, and ridge `2^1020`, the exact
active solution is `sqrt(13)`, with identity on its orthogonal complement.
The old fitter returned `FisherFitError::Numerical` in the new regression.

The fitter now chooses a power of two from the exponent of
`max(sqrt(gamma), max(abs(X)), max(abs(Y)))`, then scales both data matrices and
the ridge *before* any norm or Gram product. Including the square-root ridge
ensures the scale is normal even when gamma is the smallest positive subnormal.
The ridge is divided twice; `a*a` is never formed. Power-of-two division avoids
significand rounding when the scaled result is normal. A ridge that becomes zero
is rejected explicitly, not silently removed. Later common scatter normalization
and the existing Cholesky/SVD square-root repair remain in place.

This applies the general scale-before-squaring principle used by
[LAPACK's scaled sum-of-squares routines](https://www.netlib.org/lapack/explore-html/d8/d76/group__lassq.html).
It is not a port of DLASSQ: common normalization must also preserve the Fisher
ridge and both Gram matrices, rather than only a vector norm.

## Verification and limits

`nix develop .#stable --command bash tools/ci/check.sh faer` passed: workspace
check/tests/doctests, all 12 Fisher sampling tests, independent fixture checks,
architecture/migration gates, formatting, Clippy and rustdoc with warnings denied.
The `low_rank` and `fisher_noncommuting` integration suites also pass in release.
The SIMD/OpenBLAS matrix, Miri and Enzyme were not rerun for this safe, Faer-only
fitter change.

- Fourteen scalar-oracle fits use amplitudes `2^e`, with
  `e = -537,-535,-530,-510,0,500,510`, and two active position magnitudes.
  They cover the smallest positive ridge, subnormal products, and overflowing
  unscaled norms; check rank, active/complement velocity and mass log determinant.
- The noncommuting grid now has 1,440 fits (540 two-direction, 900 four-direction),
  adding common amplitudes `2^-510` and `2^510` to the previous 864 cases.
  Its original `1e-8` gates, coordinate-scale variation, caps, thresholds,
  ordering checks, momentum energies and independent planted solutions are intact.
- The existing unrepresentable-range rejection and independent Decimal/upstream
  fixtures still pass. No tolerance, seed, ridge default or rejection policy was
  loosened, and no dependencies or unsafe code were added.

This repair does not cure arbitrarily ill-conditioned input or recover information
lost before scaling (for example during centering/coordinate standardization).
Warmup's online moment accumulators are unchanged and retain their own finite
range checks; success in the batch fitter does not certify the same extreme
observations through the entire warmup pipeline.
Extreme *relative* dynamic range can still lose small components or fail a
decomposition. There is no general forward-error certificate. Symmetry policy,
basis rank decisions, nonlinear funnel geometry and retained-draw quality per
total cost remain separate questions. Existing timing artifacts predate the added
normalization pass; no new cost or speed claim is made.
