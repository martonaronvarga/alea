# Noncommuting Fisher fit validation

The bounded regression in
`crates/alea-math/tests/fisher_noncommuting.rs` plants a positive definite metric
`G`, chooses a positive definite score scatter `F`, and constructs `C = G F G`.
This supplies an independent expected solution without calling a matrix square
root or eigensolver in the oracle. The position and score scatters do not share
eigenvectors. These synthetic observations test the fitter algebra; they are
not samples and scores from an executed Markov chain.

## Construction and gates

The two-direction grid covers 540 fits in dimensions 4, 16 and 64. In the active
basis, `G` has eigenvalues `[0.25, 3]` or `[0.05, 5]`, and `F` is rotated by
45 degrees with condition number 4, 64 or 1024. Independent scalar Cholesky
factors generate paired positive/negative observations. The test recomputes their
scatter before fitting. It checks active projections, every ambient matrix entry,
mass log determinant and momentum energy, including rank caps 0 through 2 and
thresholds 1.1 and 10.

The four-direction extension covers 900 fits in dimensions 8 and 32. It uses
`G = diag(0.1, 0.4, 2, 6)` and
`F = diag(1, 2, 3, 4) + a v v^T`, where `v = [1, -2, 0.5, 3]` and
`a` is 0.5, 8 or 128. All four axes interact. Samples are formed directly from
the diagonal and outer-product factors, and their scatter is checked separately.
Both original and reversed row orders must recover the planted solution.
Rank caps 0 through 4 and thresholds 1.1, 3 and 20 exercise both truncation and
complete filtering. A complete Walsh basis checks every active and complementary
projection, so a small error in an active eigenvalue cannot hide among many
ambient coordinates. Determinants and unit-noise momentum energy on every basis
direction are checked independently as well.

Both grids subtract the fitter's ridge from the generated active scatters before
factoring. Adding the ridge back therefore gives exactly the intended `C` and `F`
in real arithmetic; in the complementary subspace both scatters equal the ridge,
and the solution is identity. Coordinate scales cycle through `1e-8`, `1`, `1e8`.
Observations share a factor of `2^-510`, `1e-100`, `1`, `1e100`, or `2^510`, and the ridge shares its
square. Position and score standardization use reciprocal coordinate factors.

Action errors use the existing `1e-8` normalized tolerance, scaled by the geometric
mean of the corresponding expected diagonal/eigenvalues. Log determinants use
`1e-8 * (1 + abs(expected))`. These gates must not be relaxed to accept an observed
failure. The samples' scatter checks use `1e-12` tolerances.

## Numerical repair — 2026-10-05

Both grids exposed accuracy loss in the previous nested eigensolve. For example,
the two-direction case with `G = diag(0.05, 5)`, `cond(F) = 1024`, dimension 4,
common factor `1e-100` and rank cap 1 recovered `0.05000000087390738` on the
contracting axis, exceeding its `5e-10` absolute tolerance. The four-direction
case with coupling 128, dimension 8 and the same common factor missed an
off-diagonal projection gate as well. Neither failure was a nonpositive output
metric; constructor conditioning checks alone cannot detect inaccurate fitting.

The fitter now factors the normalized `C = L L^T` and computes the SVD
`B = F^(1/2) L = U Sigma V^T`. Then
`sqrt(F^(1/2) C F^(1/2)) = U Sigma U^T`, so the original Fisher formula is
unchanged in real arithmetic. Recovering singular values from `B` avoids first
forming `B B^T`, which squares its condition number. Cholesky uses no dynamic
regularization, and failed factorizations or nonpositive/nonfinite singular values
return `FisherFitError::Numerical`. All new Faer calls explicitly use `Par::Seq`.
Common scatter normalization, rank selection and metric rejection remain intact.

All original 864 grid cases passed on the pinned stable/Faer toolchain at the original
tolerances. The replacement changes cold fitting work and allocation constants;
the older profiling timings describe the previous implementation. No speedup or
unchanged-cost claim is made for this numerical repair.

Follow-up: [five-workload cost remeasurement](benchmarks/fisher-cost-2026-10-05.md)
now records current timing, native peak RSS and separate actual-call evidence.
It does not isolate the repair's cost against a paired previous executable.

The subsequent [numerical reference review](fisher-numerical-review.md) adds
early power-of-two data/ridge normalization and extends these grids to 1,440
cases at the same tolerances. It covers subnormal ridge and unscaled scatter
overflow; the earlier timing artifacts predate this additional normalization.

## Reproduction and limits

Stable/Faer validation passed: workspace check/tests/doctests, the offline Decimal
fixture checks, architecture/migration gates, formatting, Clippy with warnings
denied, and rustdoc with warnings denied. This includes the existing upstream
fit-prefix, commuting-spectrum, thread-policy, Fisher controller and retained-draw
regressions. The initial Clippy run found two range-loop style issues in the
two-direction test; iterator replacements and the focused grid rerun passed.
SIMD/OpenBLAS, Miri, Enzyme and performance measurements were not rerun here.

```sh
nix develop .#stable --command cargo test --manifest-path crates/Cargo.toml \
  -p alea-math --features faer --test fisher_noncommuting
```

This extends the commuting-spectrum grid and the independent Decimal/upstream
fixtures. It does not certify arbitrary ill-conditioning, arbitrary spectra or
whole-warmup equivalence. The correction representation's existing conditioning
rejection remains a separate gate. M4F promotion still requires retained-draw
quality, memory and total-cost evidence, and broader robustness studies.
