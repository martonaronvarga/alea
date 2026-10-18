# Alea architecture — breaking migration and composable foundations, 2026-09-30

The roadmap's nine-crate split is the current implementation, not a facade over
the old workspace. There are no `kernels`, `runtime`, `memory`, `ffi`, `app` or
`data_prep` packages and no dependency aliases for them.

## Production boundaries

| Crate | Internal dependencies | Owns |
|---|---|---|
| alea-math | none | initialized aligned storage, mass matrices, numerical operations/RNG primitive |
| alea-core | alea-math | density-only/fused target capabilities, atomic caches, transforms, opaque derivative contract, state-space contracts |
| alea-autodiff | alea-core | Enzyme derivative adapter and finite-difference diagnostics |
| alea-ffi | none | checked cubature and audited C boundary |
| alea-distributions | alea-core, alea-math, alea-ffi | dimension-bound Gaussian, Wiener kernels/models/derivatives, latent DDM models |
| alea-mcmc | alea-core, alea-math | consuming Hamiltonian phases, HMC, derivative-free RWMH, explicit step-size warmup, typed transitions |
| alea-smc | none | particle weight/mutation/resampling interfaces; no algorithms yet |
| alea-runtime | alea-mcmc, alea-math | streaming, reproducible Rayon ensembles, bounded monitoring/Tokio subscriptions, modern and classical diagnostics |
| alea-cli | model, distribution, sampler and runtime crates | `alea`, `alea-rwmh`, `alea-gradient-check` |

The table lists internal **production** dependencies. Cargo dev dependencies cross
these layers to test real distributions against core contracts. They are not an
FFI dependency in a downstream core-only or sampler-only build. Optional OpenBLAS
is scoped to `alea-math`, with its existing LP64 ABI check. The broader cubature
crate is never a production dependency of math/core/autodiff/MCMC/runtime.

`tools/ci/architecture.py` checks canonical member names/paths and permitted
internal production/build edges, including optional dependencies. It rejects
local dependency aliases. `alea-smc` is a contract boundary, not a fabricated
particle-filter implementation; future algorithms must satisfy their milestones.

## One model and state path

`alea_core::target::LogDensityGradient` is the unconstrained fused, fallible model
contract. `ConstrainedModel` distinguishes constrained-coordinate derivatives.
`TransformedTarget` composes a model with the log-Jacobian and analytic pullback,
using reusable, chain-local aligned scratch. `PointState` exposes read-only cached
coordinates/density/gradient and publishes complete updates transactionally.

`Gaussian::new(dimension)` directly implements the target contract. The removed
`FusedAdapter` and split `LogDensity`/`GradLogDensity`/`FusedLogDensity` protocols are
not substitutes or aliases. `WienerModel<Wiener4/5/7, Data>` implements the
constrained model contract for AoS and SoA observations. Fixed-size numerical batch
operations remain available for benchmarking; the model boundary reports typed
shape and non-finite errors. `WienerPrimitive` lives with its distribution, while
the generic opaque-derivative contract remains independent in core.

`alea_math::metric::EuclideanMetric` has one fallible interface: `dimension`,
`sample_momentum`, `velocity`, `kinetic_energy`, `log_det`. Built-ins check shapes
before mutation. There is no parallel panicking public `apply_*` or `try_apply_*`
interface. Dense metrics store a validated column-major lower factor of the mass
matrix, not inverse mass; core transform matrix outputs remain row-major.

## One sampler and runner path

The validated target-bound implementation is now `alea_mcmc::Hmc`. The duplicate
mutable-state HMC, boolean `Kernel`, mutable `ChainState`, split integrator, old
runner, policy/NUTS placeholders and uncompiled DDM sketches were removed.
Tracked originals remain recoverable from Git; no compatibility layer or duplicate
compiled implementation remains. Research RMarkdown and C++ reference programs
were moved to `research/` and `tools/stan-reference/` rather than discarded.

`Rwmh` uses the fallible `alea_core::density::LogDensity` capability and
`DensityPoint`, a transactional position/density cache with no gradient storage.
HMC's `LogDensityGradient`/`PointState` retain their fused contract. These are
different numerical capabilities, not compatibility aliases. RWMH
uses immutable validated options and reusable proposal storage. Numerical failures
produce typed rejections; model errors propagate; failed evaluation/unwinding never
publishes partial state. Its metric is the **proposal covariance** before squared
step-size scaling. It computes no gradients and retains the old polar RNG's spare
normal variate. `adapt::warmup_rwmh` ports dual averaging as an explicit controller:
sampling never implicitly adapts. Zero warmup does nothing; `None` runs fixed-scale
burn-in. The controller is consumed when its averaged scale is installed. This
retains earlier functionality but does not implement M4's windowed mass adaptation.

Both chains implement `MarkovChain`, with associated error and transition types.
`alea_runtime::run_chain` streams every completed transition (including rejections)
to a callback. The callback may stop through `ControlFlow`; no draw buffer, logger,
async runtime or per-step allocation is imposed. Errors propagate before observation;
an observer panic leaves the already completed transition intact. This minimal
single-chain API remains separate from M6's richer ensemble runner.

`alea_runtime::draws::collect_draws` optionally allocates one 64-byte-aligned,
row-major buffer before advancing the chain/RNG. It includes rejected draws and
checks size overflow. `diagnostics::autocorrelation_ess` and
`classical_split_rhat` retain the old calculations under accurate names, rejecting
undefined inputs instead of reporting reassuring defaults. They are not rank-
normalized bulk/tail ESS or modern R-hat. See the [migration audit](migration-audit.md).

M6 adds `ensemble::run` with logical ChaCha12 streams and a caller-owned Rayon
pool, a versioned borrowed draw/transition schema, and cooperative cancellation
between transitions. A separate diagnostics thread reads latest-only immutable
snapshots of bounded contiguous windows; Tokio subscriptions expose cloned reports
without leaking locks. `diagnostics::modern` implements independently referenced
rank/folded R-hat, bulk/tail/mean ESS, mean MCSE and E-BFMI. Monitors reject
ragged/undefined inputs and stop only under an explicit minimum-draw policy.
See [M6 ownership, memory and mathematical contracts](m6-runtime.md).

```rust
use alea_distributions::Gaussian;
use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};
use alea_mcmc::{Hmc, HmcOptions};

let target = Gaussian::new(4);
let chain = Hmc::new(
    &target,
    OwnedBuffer::new(4),
    IdentityMetric::new(4),
    HmcOptions::new(0.1, 8)?,
)?;
// chain.step(&mut rng)? returns typed HmcTransition; point() is read-only.
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Geometry, integration and adaptation

`alea-core::capability` adds optional potential-HVP and explicit-layout batch
evaluation. They do not expand the minimum fused target protocol or make Enzyme
mandatory. `Gaussian` implements both capabilities. HVP means Hessian of **negative**
log density; batch lane identity is independent of chain-major/coordinate-major storage.

`LowRankDiagonalMetric` stores coordinate scales, orthonormal column-major
directions and positive eigenvalues of an inverse-mass correction. Eigenvalues
below one are supported. Runtime actions allocate nothing; reduced Fisher fitting
is a cold `alea-math/faer` capability. Core/MCMC still have no direct faer dependency.

`Hmc<T,M,I>` composes a target, metric and statically dispatched `Integrator`.
Reference leapfrog and validated OMF2/BCSS2 two-stage maps expose only complete
macrosteps through consuming `Phase::step_with`. Public counters use
`integration_steps`; a successful two-stage edge uses two fused evaluations.
No backwards-compatible `leapfrog_steps` alias is retained.

`HmcWarmup<T,I>` implements the M4 reference covariance schedule.
`FisherHmcWarmup<T,I>` is experimental paired-score adaptation. Both consume
their adapters and return a frozen HMC chain; neither is a `MarkovChain` while
adapting. Final low-rank fitting precedes geometry freeze. See
[the derivation and roadmap decisions](fisher-and-trajectories.md),
[M4 validation](m4-warmup.md), and the [future compiled-target ABI](compiled-target-abi.md).

## Reproduction commands

```sh
nix develop .#stable --command bash tools/ci/check.sh
nix develop .#stable --command bash tools/ci/check.sh alea-mcmc/faer workspace
nix develop --command bash tools/ci/check.sh alea-mcmc/simd,alea-distributions/simd,alea-distributions/branch-hints workspace
nix develop .#blas --command bash tools/ci/check.sh alea-mcmc/simd,alea-distributions/simd,alea-mcmc/faer,alea-mcmc/openblas workspace
nix develop --command bash tools/ci/miri.sh
nix develop .#autodiff --command bash tools/ci/autodiff.sh
nix develop .#stable --command cargo bench --manifest-path crates/Cargo.toml -p alea-mcmc --bench hmc --locked
```

The default `all` scope runs feature-independent policy, evidence and downstream
conformance gates as well as workspace checks. CI runs that scope once, then uses
the `workspace` scope for additional feature combinations.

The cubature Git submodule was relocated without changing upstream bytes or its
commit. Nix's default executable is now `alea`; the `matmod` wrapper/backend
environment variable was removed. Clippy/Miri/toolchain availability is unchanged.
Python's standard library is included in development shells for the architecture
gate. It is not a Rust runtime dependency.

Earlier milestone reports retain their recorded pre-migration paths and benchmark
labels as historical provenance. Use this document and README for current commands.
No performance claim follows merely from moving modules across crate boundaries.
