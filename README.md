# Alea

A work-in-progress Rust library for programmable, typed Monte Carlo inference,
with Wiener/DDM modeling as its first application. Performance and numerical
correctness guide the [canonical engineering roadmap](roadmap.md).

## Current status

M0 (repository stabilization) and M1 (safe core) are complete;
see the [validation and completion record](docs/m0-m1-completion.md).

The default workspace includes tested Wiener likelihoods and gradients, a native
Gaussian target, random-walk Metropolis, experimental fixed-length Euclidean HMC,
and latent DDM adapters. HMC uses fused value/gradient evaluation and
metric-correct momentum; its full validation is unfinished.
Mass-metric constructors now return typed validation errors; see the
[metric contract and migration notes](docs/metric-contract.md).
The additive [target/state API](docs/target-state.md) provides fallible fused
evaluation and transactional, target-bound caches. The new
[`HmcChain`](docs/hmc-chain.md) uses these caches, validated settings, reusable
aligned scratch, and typed transition/divergence diagnostics. Legacy `Hmc` remains
available during migration.
The [HMC validation suite](docs/hmc-validation.md) covers scalar-reference
transitions, all-metric Gaussian/banana moments, and ill-conditioned Gaussian
regressions. [Executed, version-pinned BlackJAX fixtures](docs/blackjax-reference.md)
now check deterministic endpoints and signed reversal for all three mass types;
broader external statistical validation remains pending.
HMC uses a reusable [public phase integrator and snapshots](docs/hamiltonian-phase.md)
whose consuming ownership token prevents saving a partially failed step.
The [unsafe/FFI audit](docs/unsafe-audit.md) records callback panic containment,
initialized foreign output buffers, and native allocation-failure cleanup tests.

NUTS is a placeholder; SMC/particle-MCMC files are empty. These and policy sketches
are available only behind `runtime/experimental`, not in the supported API.
ChEES-HMC, RMHMC,
and the std::autodiff/Enzyme backend are planned, not implemented. Existing
`ess_bulk` and `split_rhat` helpers are legacy estimators, not modern
rank-normalized diagnostics; do not treat them as production convergence checks.

Only `runtime::ddm::latent` is currently compiled from the DDM adapters. The
other DDM sketches remain in the source directory, uncompiled, pending repair.
This is pre-alpha software, not yet a production inference engine.

## Workspace

- `crates/memory`: initialized 64-byte-aligned buffers, standalone Miri tests and benchmarks.
- `crates/kernels`: density/state traits, safe reusable buffers, metrics, Wiener kernels.
- `crates/ffi`: foreign numerical bindings, including cubature.
- `crates/runtime`: chain execution, MCMC, diagnostics, latent DDM adapters.
- `crates/app`: experiments and executable wiring.
- `crates/data_prep`: data preparation.

Crate names remain unchanged while the core contracts stabilize.

## Development checks

Enter `nix develop` for a pinned nightly with Clippy, Miri, rustfmt, rust-src, and
rust-analyzer. Use `nix develop .#stable` for stable-feature checks. No local
toolchain symlink is required. See [toolchain and storage notes](docs/toolchains-and-storage.md)
for Miri, SIMD benchmarks, optional shells, and the experimental Enzyme boundary.

Then run from `crates/`:

```sh
cargo fmt --all -- --check
cargo check --workspace --all-targets --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --all-targets --locked
cargo test --workspace --doc --locked
cargo doc --workspace --no-deps --locked
cargo miri test -p memory --lib --locked
```

The optional SIMD and autodiff paths have additional toolchain requirements; a
passing default build does not validate them. See the roadmap for milestone
gates, validation evidence, and the next implementation tasks.
The [M0 baseline guide](docs/stabilization.md) documents the supported feature
matrix, CI commands, research archive, and explicit unavailable-autodiff diagnostic.

## Guides and links

1. [polars](https://docs.rs/polars/latest/polars/)
2. [Rust FFI](https://jakegoulding.com/rust-ffi-omnibus/)
3. [Stan Math Wiki & Quickstart](https://github.com/stan-dev/math/wiki)
4. [SMTC (Particle Filter in C++)](https://github.com/awllee/smctc)
5. [Eigen C++](https://eigen.tuxfamily.org/dox/GettingStarted.html)

6. [Futhark Scan](https://futhark-book.readthedocs.io/en/latest/functional-parallel-programming.html#scan)
7. [Futhark C/Rust Backend](https://futhark.readthedocs.io/en/latest/c-api.html)

8. [Zig Guide](https://zig.guide/)
9. [Zig C Interop](https://ziglang.org/documentation/master/#C)

## Profiling

- Perf: `perf record -g ./your_binary && perf report` or open `perf.data` in `hotspot`
- Valgrind: `valgrind --tool=massif ./binary` to profile memory, view with `massif-visualizer`
- [hyperfine](https://github.com/sharkdp/hyperfine)
