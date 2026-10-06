# Independent Fisher fixtures

These fixtures validate Alea's specified estimators, not complete nuts-rs/nutpie
warmup equivalence. They do not call Alea or duplicate its reduced MGS/eigensolver
or streaming-Welford implementation.

A separate [executed nuts-rs fit reference](upstream/README.md) calls the pinned
upstream implementation. Keep that evidence distinct from this mathematical oracle.

Run from the repository root:

```sh
nix develop .#stable --command python3 tools/fisher-reference/generate.py
nix develop .#stable --command python3 tools/fisher-reference/generate.py --check
```

The generator uses Python 3.13.12 from the repository's lock-pinned Nix environment,
standard-library `decimal` at 120 decimal digits, and no third-party dependencies.
CI checks regenerated fixture bytes. Rust consumes checked-in decimal values as f64.

## Independent formulation

For centered, standardized observations, form full 2x2 scatter matrices C and F,
adding the same ridge to each. The oracle computes
`G = C^(1/2) [C^(1/2) F C^(1/2)]^(-1/2) C^(1/2)`.
The principal square root uses the closed 2x2 determinant/trace formula, not an
eigendecomposition. An independent residual check verifies `G F G = C` before
truncation. Analytical spectral projectors apply Alea's threshold/rank policy.
The matrix is finally rescaled by the coordinate scales; `logdet_mass = -logdet(G)`.

This is the SPD Fisher optimum described by the
[Fisher-HMC paper, affine-transform section](https://arxiv.org/html/2603.18845v1#S2.SS4).
The raw-scatter/ridge convention agrees with the
[pinned nuts-rs source](https://github.com/pymc-devs/nuts-rs/blob/a762aae513bf7bb5ccb27ffc9a938a923fc837ec/src/transform/adapt/low_rank.rs);
that implementation is not executed by this generator.

`fisher-dense.csv` covers noncommuting scatter matrices, non-unit scales, rank caps,
spectral filtering, rank-deficient data, and common observation factors 1e-100 and
1e100 with the ridge scaled quadratically. The latter preserve the same geometry
while testing nested-product underflow/overflow. It is a two-dimensional oracle,
not general high-dimensional rank-selection validation.

`fisher-diagonal.csv` recomputes centered batch scatter from retained observations
at every update across period-three foreground/background boundaries. It does not
use a Welford recurrence. Its schedule is Alea's existing specified schedule, not
an independently discovered or upstream-verified policy.

## Pinned artifacts (SHA-256)

```text
952b72db9f1b78fd99c3fd9732a97c84dc76e620a6092b1330efe181366c4fd3  generate.py
e0797d14fa06d16cf378859f4485ab1ddc6c6656bef3259abddfa65e81de64f5  crates/alea-math/tests/fixtures/fisher-dense.csv
251f45b16c172429d178626e82f7158a06785a46418b53cef71d76f90c21f782  crates/alea-mcmc/tests/fixtures/fisher-diagonal.csv
```

Input decimal rounding is separate from Rust arithmetic error. Dense comparisons
use a 1e-8 tolerance scaled by geometric diagonal magnitudes, diagonal update
comparisons use 2e-12 relative tolerance. These tolerances were set before executing
the Rust comparisons; existing sampling seeds/tolerances are unchanged.
