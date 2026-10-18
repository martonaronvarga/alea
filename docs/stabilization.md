# M0 repository baseline

The supported baseline is Rust 1.95 (2024 edition), tested with the stable
toolchain from `flake.lock`. The default shell provides a same-date nightly,
Clippy, Miri, rustfmt, and rust-src. Workspace package metadata, dependencies,
and lint policy are centralized in `crates/Cargo.toml`; members opt in explicitly.
Unused direct dependencies and unused raw cubature bindings have been removed.

## Supported feature combinations

| Configuration | Toolchain / prerequisites |
|---|---|
| Default; Faer | Pinned stable |
| SIMD; distribution branch hints | Pinned nightly |
| OpenBLAS, including SIMD + Faer + OpenBLAS together | `nix develop .#blas`, LP64 ABI |
| `std-autodiff` | Pinned `.#autodiff` nightly, release profile/fat LTO; executed M2 gate |

Backend flags preserve the same metric API and mathematics; when combined,
dense operations select OpenBLAS, then Faer, then SIMD, then scalar. Features do
not remove default APIs. Unrepaired DDM modules remain uncompiled; only the tested
latent adapter is public.

M2 replaces the unavailable-backend diagnostic with an executed Enzyme CI job.
The separate upstream prebuilt compiler/plugin are content-pinned together;
see [the model boundary and validation record](m2-model-boundary.md). There is
no finite-difference fallback. Stable analytic models remain the default.

## Gates

Run the same checks used by `.github/workflows/ci.yml` from the root:

```sh
nix develop .#stable --command bash tools/ci/check.sh
nix develop .#stable --command bash tools/ci/check.sh alea-mcmc/faer workspace
nix develop --command bash tools/ci/check.sh alea-mcmc/simd,alea-distributions/simd,alea-distributions/branch-hints workspace
nix develop .#blas --command bash tools/ci/check.sh alea-mcmc/simd,alea-distributions/simd,alea-mcmc/faer,alea-mcmc/openblas workspace
nix develop --command bash tools/ci/miri.sh
nix develop .#autodiff --command bash tools/ci/autodiff.sh
```

Native jobs check formatting, all-target compilation, tests/doctests, Clippy with
warnings denied, and docs with warnings denied. Research-only Wiener helpers are
test-gated rather than suppressed globally. C builds compile only the bound
h-adaptive cubature entry point, with optimization sufficient for fortify checks.

CI uses read-only permissions, immutable action SHAs, recursive submodule checkout,
and the repository's locked Nix environments. No secrets or publishing steps are
needed. Adding workflow files does not configure GitHub branch-protection rules;
repository administrators must make these jobs required if desired.

Miri cannot execute arbitrary foreign C/BLAS functions. Its test selection and
the native foreign-library tests must remain complementary; native FFI execution
must never be represented as having run under Miri.

## Research archive

The [archive](../archive/README.md) preserves alternate Wiener sources, comparison
programs, lint output, and local benchmark binaries. Original paths and hashes
are recorded. Binaries are ignored, not deleted. No model data or active numerical
implementation was removed. Future compiled outputs should go under `target/`.
