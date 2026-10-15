# Validated Euclidean mass metrics

Implemented in roadmap increment 3 (2026-09-25). This is the existing `Metric`
interface hardened at its boundaries, not yet the final target/state protocol.

## Meaning and construction

All built-in metrics represent the **momentum mass matrix** `M`:

- `apply_sqrt(z, p)` writes `p = L z`, where `M = L L^T`.
- `apply_inverse(p, v)` writes velocity `v = M^-1 p`.
- `log_det()` returns `ln det(M)`, not `ln det(L)`.

`DiagonalMetric::new` and `CholeskyFactor::new_lower` now return
`Result<_, MetricError>`. Callers must propagate or handle constructor failures;
there is no compatibility constructor that silently clamps invalid input.

```rust
use kernels::{
    buffer::OwnedBuffer,
    metric::{CholeskyFactor, DenseMetric, DiagonalMetric, Metric, MetricError},
};

fn example() -> Result<(), MetricError> {
    let diagonal = DiagonalMetric::new(OwnedBuffer::from_fn(2, |i| [4.0, 9.0][i]))?;
    let lower = OwnedBuffer::from_fn(4, |i| [2.0, 0.5, 0.0, 3.0][i]);
    let dense = DenseMetric::new(CholeskyFactor::new_lower(2, lower)?);
    let mut momentum = [0.0; 2];
    diagonal.try_apply_sqrt(&[1.0, 2.0], &mut momentum)?;
    dense.try_apply_inverse(&[6.0, 19.5], &mut momentum)?;
    Ok(())
}
```

Diagonal masses must be finite and strictly positive. Cholesky storage must have
exactly `dimension * dimension` entries in full column-major order, finite values,
strictly positive diagonal, and zero upper triangle. Signed zero above the
diagonal is accepted. A symmetric mass matrix is **not** a Cholesky factor; no
factorization is performed by this constructor. Dimensions and byte counts are
checked for overflow before indexing. Empty zero-dimensional metrics are valid
linear operators; samplers may impose stricter model-dimension requirements later.

Requiring an explicitly zero upper triangle is intentional: malformed full
matrices are rejected, not silently reinterpreted. Existing in-repository factor
generators already satisfy this requirement. Storage is moved into the metric,
retaining its allocation and 64-byte alignment; it is not copied or repaired.

These are structural guarantees, **not** a conditioning guarantee. Subnormal
positive values remain unchanged. Finite operands can still produce non-finite
results, and HMC must retain its trajectory checks. Regularization belongs in an
explicit adaptation policy, not a constructor.

## Operation boundaries

`try_apply_inverse` and `try_apply_sqrt` return `MetricError::VectorLength` if
either vector length differs from the metric dimension. Output remains untouched
on a dimension error. Both short and oversized buffers are rejected.

The existing infallible operations remain for dimensioned sampler workspaces;
they now assert both lengths **in release builds before any output mutation**.
They are safe APIs and must never rely on a debug-only assertion for FFI safety.
Neither API scans vectors for finiteness nor adds temporary numerical storage.
Built-in operations reuse the caller's output. An allocation-count regression and
updated end-to-end timings remain outstanding; this increment makes no speedup claim.

The OpenBLAS boundary additionally checks matrix storage and checked integer
conversion, uses unit vector stride, and skips the foreign call for dimension zero.
Each unsafe block records pointer extent, aliasing, and lifetime justification.
This satisfies the shape/leading-dimension obligations of
[BLAS DTRSV](https://www.netlib.org/lapack/explore-html/dd/dc3/group__trsv_ga7a7dcbb8745b4776ce13063ab031141f.html);
it does not audit the linked BLAS implementation or the other FFI modules.

### OpenBLAS installation and ABI

The `openblas` feature uses pkg-config to discover and link **LP64** OpenBLAS
(32-bit BLAS dimensions/strides, as required by `cblas` 0.4). A small C compile
check rejects headers declaring 64-bit BLAS integers. Nixpkgs' default OpenBLAS
on this host uses ILP64, so the `blas` and `full` shells now explicitly override
`blas64 = false`. Do not substitute an ILP64 library or unrelated header path.

No BLAS build dependency or link flag is activated for the default feature set.
On Linux, the opt-in shells set `LD_LIBRARY_PATH` for this OpenBLAS installation:
the upstream Fenix linker does not embed the Nix store library rpath. The default
and stable shells remain unchanged. Outside Nix, install matching LP64 development
headers/library and pkg-config metadata, and configure the runtime loader normally.

## Regression coverage

`crates/kernels/tests/metrics.rs` covers invalid diagonals, both non-finite factor
triangles, nonzero upper entries, shape/byte overflow, unchanged output on every
short/long dimension mismatch, empty operators, alignment retention, exact mass
conventions, and numerical comparison to independently formed `M = L L^T` across
SIMD/block boundaries. Tests deliberately offset output storage. Existing HMC
tests cover metric-aware momentum and nonidentity Gaussian sampling.

Run from the repository root:

```sh
nix develop --command cargo test --manifest-path crates/Cargo.toml -p kernels --test metrics --locked
nix develop --command cargo test --manifest-path crates/Cargo.toml -p kernels --test metrics --features simd --locked
nix develop --command cargo test --manifest-path crates/Cargo.toml -p kernels --test metrics --features faer --locked
nix develop .#blas --command cargo test --manifest-path crates/Cargo.toml -p kernels --test metrics --features openblas --locked
nix develop --command cargo test --manifest-path crates/Cargo.toml -p kernels --test metrics --release --locked
nix develop --command cargo test --manifest-path crates/Cargo.toml -p runtime --test hmc --locked
nix develop .#blas --command cargo test --manifest-path crates/Cargo.toml -p kernels --lib --test metrics --features simd,faer,openblas --locked
nix develop .#blas --command cargo test --manifest-path crates/Cargo.toml -p runtime --test hmc --features openblas --locked
nix develop --command env MIRIFLAGS='-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check' \
  cargo miri test --manifest-path crates/Cargo.toml -p kernels --test metrics --locked
```

### Validation record (x86-64 Linux, pinned toolchains)

- Default workspace: 84 tests passed, plus benchmark smoke tests and 5 doctests.
- Eight metric regressions passed separately on scalar, SIMD, Faer, and LP64
  OpenBLAS; the scalar release-mode run also passed.
- Combined SIMD/Faer/OpenBLAS: 58 kernel unit tests and 8 metric regressions passed;
  HMC with OpenBLAS: all 6 regressions passed.
- The C ABI guard rejected the installed ILP64 headers, as intended.
- All eight default-backend metric tests passed Miri with strict provenance and
  symbolic alignment checks, including the full matrix-reference size grid.
- Pinned stable 1.95.0 workspace/all-targets compilation passed.
- Workspace Clippy completed with existing warnings; docs, Rust/Nix formatting,
  and uncached `nix flake check --no-build` evaluation passed.

Foreign backend results are native tests, not Miri verification of foreign code.
No new end-to-end timing or allocation-count result is claimed here. Other
platforms, the full research shell, and Enzyme execution remain unvalidated.

Next: fallible fused target evaluation and atomic cached state, typed sampler
configuration/statistics, allocation counters, and the remaining FFI audits.
