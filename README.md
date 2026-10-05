# Alea

A pre-alpha Rust framework for typed probabilistic modeling and high-performance
Monte Carlo inference. Compose models from distributions and scientific likelihoods,
then compare inference methods through explicit numerical contracts.

The [canonical roadmap](roadmap.md) and [settled implementation plan](docs/settled-plan.md)
cover both modeling and inference. Alea is not merely a sampler wrapper.

## Direction and scope

- A Rust-native modeling framework: typed parameters, priors, likelihoods, transforms,
  vectorized observations, hierarchical composition and posterior prediction.
- A curated distribution library, alongside optimized scientific models such as
  Wiener/DDM. Each distribution earns support through numerical and derivative tests.
- Composable geometry, dynamics, integration, local step control, trajectories,
  correction, adaptation and diagnostics; exact and approximate kernels stay distinct.
- Reproducible cross-engine validation and end-to-end performance measurements,
  including initialization and warmup—not just iterations per second.

Models may also arrive through the fused target interface and future foreign bridges.
Native modeling and external targets are complementary entry points. Enzyme is an
optional compiled derivative backend; analytic derivatives remain first-class.

The near-term project will not build a separate textual modeling language, attempt
Stan's entire distribution/ecosystem coverage, replace JAX's accelerator stack, or
claim universal sampler superiority. Those limits do not exclude a substantial
Rust modeling API. Discrete observations are in scope; direct HMC updates of discrete
latent variables are not. No unvalidated diagnostic automatically declares convergence.

## Architecture

The [target architecture](docs/architecture.md) is implemented as nine crates.
This is a breaking replacement: no legacy crate aliases, mutable-state sampler
protocols or compatibility constructors are retained.

| Crate | Responsibility |
|---|---|
| `alea-core` | Density-only/fused target capabilities, transactional state, transforms, model composition |
| `alea-math` | Initialized 64-byte-aligned storage, Euclidean metrics, numerical primitives |
| `alea-autodiff` | Enzyme adapters and finite-difference validation |
| `alea-distributions` | Gaussian and Wiener/DDM models, analytic derivatives |
| `alea-mcmc` | Composable fixed HMC/RWMH, reference covariance and experimental Fisher warmup |
| `alea-smc` | Particle-kernel contracts; filtering/PMCMC are not implemented |
| `alea-ffi` | Audited synchronous foreign boundaries |
| `alea-runtime` | Allocation-free streaming, aligned draw collection, classical diagnostics |
| `alea-cli` | `alea`, `alea-rwmh` profiling loop and `alea-gradient-check` |

Core/math/MCMC production dependencies do not include `alea-ffi`; Wiener owns
its cubature dependency. Dependencies used only by tests may cross layers.
The CI architecture check enforces production boundaries, including optional edges.

## Status

M0–M3 gates have been completed: safe core, constrained transforms, executed
`std::autodiff`/Enzyme and fixed-length Euclidean HMC. The architecture migration
retains [M3's independent BlackJAX references](docs/m3-hmc-completion.md),
sampling checks and allocation regressions.

M4 [reference windowed adaptation](docs/m4-warmup.md) is implemented and validated.
Optional HVP/batch targets, spectral low-rank metrics and leapfrog/OMF2/BCSS2
composition are implemented. [Fisher warmup](docs/fisher-and-trajectories.md) is
experimental; upstream trace equivalence and broad efficiency gates remain open.

NUTS, WALNUTS, ChEES-HMC, RMHMC, parallel multi-chain execution,
modern ESS/R-hat and particle algorithms remain planned. RWMH transitions are
fixed-scale; explicit dual-averaging warmup retains the previous functionality.
Classical diagnostics are named honestly, not presented as modern bulk ESS/R-hat.
The [migration audit](docs/migration-audit.md) inventories retained behavior,
test replacements and dormant research sources; CI guards the inventory.
This is not yet a production inference engine.

## Development

`nix develop` supplies the pinned nightly, Clippy, Miri, rustfmt and rust-analyzer.
`nix develop .#stable` supports analytic models; `nix develop .#autodiff` provides
the separately pinned compiler and matching Enzyme plugin.

Run from the repository root:

```sh
nix develop .#stable --command bash tools/ci/check.sh
nix develop --command bash tools/ci/miri.sh
nix develop .#autodiff --command bash tools/ci/autodiff.sh
nix develop .#stable --command cargo run --manifest-path crates/Cargo.toml -p alea-cli --bin alea
nix develop .#stable --command cargo run --manifest-path crates/Cargo.toml -p alea-cli --bin alea-gradient-check -- all
```

The Nix default application is `alea`, not `matmod`. The cubature submodule now
lives at `crates/alea-ffi/vendor/cubature`; initialize it with
`git submodule update --init --recursive` after checking out the migrated tree.
See [architecture and API examples](docs/architecture.md) for the current interface.
Earlier milestone reports are historical evidence, not compatibility guarantees.
