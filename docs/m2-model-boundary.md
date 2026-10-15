# M2: constrained model boundary and Enzyme

Implementation date: 2026-09-28. Source lives in `kernels::{transform, model,
autodiff, gradient_check}`. These APIs extend M1's target protocol; existing
samplers, storage and metric conventions are unchanged.

Status: **M2 complete**, with the explicit supported scope below. Local stable
and SIMD workspace format/check/test/Clippy/docs gates pass. The release Enzyme
gate passes 15 focused tests (4 AD, 5 models, 6 transforms), release Clippy with
warnings denied, and all six CLI diagnostics. The expanded strict-Miri suite
passes, including 9 new transform/model tests; allocation instrumentation and
proptest's native randomness/persistence are excluded there. HMC exponential/Beta
moments and allocation gates pass. `nix flake check --no-build`, Nix formatting,
Rust formatting and `git diff --check` pass. These are local x86-64 Linux results,
not a remote CI run or cross-platform/backend-matrix claim; Faer/OpenBLAS were
not rerun for M2.

## Mathematical contract

`ParameterLayout` computes immutable offsets and checked dimensions once.
`Transform` constructors reject invalid bounds, dimensions and size overflow.
Evaluation takes caller storage. `TransformedTarget` owns four initialized,
64-byte-aligned buffers per chain: constrained parameters, constrained gradients,
unconstrained gradients and shared transform scratch. There is no lock, dynamic
dispatch, uninitialized storage or new unsafe code.

One evaluation constrains parameters, accumulates log absolute Jacobians, calls
the constrained model once, then applies analytic reverse pullbacks including
the Jacobian gradient. Enzyme differentiates the pure constrained density;
validation and transform machinery stay outside AD IR. Fusion is at the target
call boundary, not one compiler-differentiated function.

Models must overwrite every derivative; missing/nonfinite outputs are rejected.
The adapter publishes its unconstrained gradient only after all checks pass.
Errors and panics leave caller gradients untouched. Create one adapter per chain:
it is intentionally not `Sync`. Models must preserve their dimension and function.

| Transform | Input / output dimensions | Convention |
|---|---|---|
| Identity; lower; upper; interval | N / N | exp for one-sided bounds; logistic for intervals |
| Ordered; positive ordered | N / N | free or positive first coordinate, positive exp increments |
| Simplex | K-1 / K | centered stick breaking; first K-1 coordinate measure |
| Unit vector | K-1 / K | stereographic chart; surface measure |
| Cholesky covariance | N(N+1)/2 / N² | positive exp diagonal |
| Covariance | N(N+1)/2 / N² | L Lᵀ; independent lower-triangle Jacobian |
| Cholesky correlation | N(N-1)/2 / N² | tanh partial correlations; positive diagonal/unit row norms |
| Correlation | N(N-1)/2 / N² | L Lᵀ; unit diagonal; independent strict-lower Jacobian |

Matrices are **row-major** and inputs pack the lower triangle row by row.
Supply constrained gradients of the full matrix: pullbacks add the two
off-diagonal contributions. Cholesky upper entries are constant zero.
Inverse matrices require exact symmetry and positive Cholesky pivots.
Correlation diagonals/row norms use explicit floating-point tolerances.

Simplex coordinates are not Stan 2.40's newer sum-to-zero-softmax chart.
The sphere chart is not Stan's redundant radial construction. It excludes the
north pole (surface measure zero) and can have difficult tail geometry there.
Neither choice claims identical trajectories to Stan. Reference:
[Stan's transforms](https://mc-stan.org/docs/reference-manual/transforms.html).

Boundary rounding is an error, never clamping. Full-matrix transforms check the
rounded L Lᵀ product with Cholesky, adding O(N³) work; prefer factor transforms
when the likelihood accepts them. Future optimizations must preserve support
checks. Empty identity/ordered blocks and singleton simplex/correlation blocks
work; spheres require K >= 2.

Raw numerical transform errors may leave partial outputs; shape/input-finiteness
errors precede writes. `pullback` requires a point accepted by `constrain` and
does not repeat support validation. Layout error indices are block-local.

## Enzyme and opaque primitives

`.#autodiff` uses prebuilt **nightly-2026-06-23**, rustc
`4429659e4745016bd3f26a4a421843edc7fbc422`, LLVM **22.1.7**, and matching
`enzyme-preview`. The manifest hash in `flake.nix` is
`sha256-/yl/30nnnHU8U//+i5usZZDCcBZ+QUMeQ+3uWw9lN0g=`.
The default/stable pins are unchanged; no Rust/LLVM source build is needed.

Use `#![feature(autodiff)]` and
`#[std::autodiff::autodiff_reverse(derivative, Duplicated, Active)]` on a pure
`fn(&[f64]) -> f64`. Pass its generated function to
`EnzymeModel::new(dimension, derivative)`, then wrap it with `TransformedTarget`.
Enzyme accumulates adjoints, so the adapter zeros them on **every** evaluation.
The feature exposes an adapter, not proof that arbitrary callbacks came from AD.
The tests actually compile and execute generated functions.

Release mode/fat LTO is the supported gate. LTO comes from the workspace profile,
not global `-Clto=fat`, which breaks procedural-macro dependencies. The shell sets
`-Zautodiff=Enable -Cembed-bitcode=yes`. Debug builds and arbitrary feature/backend
combinations are not claimed validated. See [Rust AD docs](https://doc.rust-lang.org/std/autodiff/index.html)
and [installation](https://rustc-dev-guide.rust-lang.org/autodiff/installation.html).

`OpaquePrimitive<N>` returns a fused value and analytic derivative array.
`PrimitiveModel` adapts it; `SumModel` composes same-coordinate densities without
allocation. `WienerPrimitive` uses the existing externally checked Wiener4 series
derivatives. Tests combine it with an Enzyme prior and check the complete
transformed target against finite differences. This is an **explicit boundary**,
not registration in Enzyme's internal derivative registry. Arbitrary nested opaque
calls inside differentiated Rust remain unsupported; use explicit chain rules
or this boundary, and never assume Enzyme can see through FFI/cubature.

## Verification and diagnostics

```sh
nix develop .#stable --command bash tools/ci/check.sh
nix develop .#autodiff --command bash tools/ci/autodiff.sh
nix develop --command bash tools/ci/miri.sh
nix develop .#stable --command cargo run --manifest-path crates/Cargo.toml -p kernels --example gradient_check -- all
nix develop .#stable --command cargo bench --manifest-path crates/Cargo.toml -p kernels --bench transforms
```

The diagnostic accepts `all`, `gaussian`, `correlated`, `logistic`, `banana`,
`funnel`, or `wiener`, plus an optional relative step. Invalid arguments,
evaluation failures and gradient disagreements exit nonzero. `check_gradient`
is the equivalent API for arbitrary targets. It intentionally allocates and is
never a sampler fallback. The CLI uses analytic fixtures; the separate Enzyme
gate compares executed generated derivatives with these references.

Tests cover all 13 transform families, numerical Jacobian determinants (Gram
determinants for the sphere), finite-difference pullbacks, seeded randomized
dimensions, proptest shrinking, malformed/nonfinite inputs, boundary collapse,
singleton cases, and allocation counters. Model tests cover one-call fusion,
alignment, complete gradients, dimension changes, panic/error recovery and
diagnostic failure detection. An HMC integration test checks exponential/Beta
moments through the new boundary.

Miri checks safe transforms and analytic models, not generated Enzyme machine code
or foreign C. Enzyme is tested natively. Allocation claims apply after constructing
storage and to the tested models, not arbitrary callbacks or AD-generated loops.

Criterion's `m2_fused_boundary` group exercises identity, positive, simplex and
matrix/factor paths. These are baselines, not evidence of optimal code generation
or superiority to Vec/Stan/JAX. Profile recomputation and validation before adding
SIMD or dropping checks. Initial benchmark execution overlapped other compilation;
its timing estimates are not suitable for performance comparisons.
