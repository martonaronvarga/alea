# Target-bound fixed-length HMC

Roadmap increment 5 (2026-09-25) adds `runtime::HmcChain`. It consumes the
[fallible target/state protocol](target-state.md), while legacy `Hmc` and mutable
`ChainState` remain available. This is experimental fixed-length Euclidean HMC,
not adapted HMC, NUTS, ChEES-HMC, or RMHMC.

Increment 7 extracts an [internal consuming phase/integrator boundary](hamiltonian-phase.md).
The public API, eight-buffer layout, and fixed-length transition policy remain unchanged.

## API and invariants

```rust
use kernels::{buffer::OwnedBuffer, dist::Gaussian, metric::IdentityMetric,
              target::FusedAdapter};
use rand::{rngs::SmallRng, SeedableRng};
use runtime::{HmcChain, HmcOptions};

let target = FusedAdapter::new(&Gaussian, 2);
let options = HmcOptions::new(0.1, 8)?;
let mut chain = HmcChain::new(&target, OwnedBuffer::new(2),
                            IdentityMetric::new(2), options)?;
let mut rng = SmallRng::seed_from_u64(42);
let transition = chain.step(&mut rng)?;
let position: &[f64] = chain.point().position();
```

`StepSize` is finite and positive; the leapfrog count is nonzero. Options are
validated before chain construction and immutable during sampling. HMC rejects
zero-dimensional targets and mismatched metrics before evaluation. The target
and metric must remain semantically fixed; dimension checks do not enforce
mathematical purity. `set_position` is an explicit transactional, one-evaluation
reset. Read-only point access cannot inject a stale or foreign target cache.

Construction evaluates once. Each completed `L`-step trajectory uses exactly `L`
fused evaluations; there is no starting refresh. Failed trajectories stop early.
Momentum is freshly sampled as `p = L z` for mass `M = L L^T`; kinetic energy is
`p^T M^-1 p / 2`. Momentum is discarded after each transition; only position and
its density/gradient cache persist. There is no persistent-momentum API here.

The live point, proposal point, and integration scratch are separate. Integration
uses the checked target evaluator; only acceptance swaps the complete proposal
into the live point. Rejection, evaluation errors, and unwinding preserve the live
cache. Proposal scratch may be dirty after failure and is reset on the next step.
RNG state is consumed by attempted transitions and is not rolled back.

## Failures and diagnostics

Types are in `runtime::mcmc::hmc_chain`:

- `HmcConfigError`: invalid step size/count or energy-error limit.
- `HmcError<E>`: dimension contracts, metric failures, or the original model
  error chain. Evaluation errors carry a one-based attempted leapfrog step
  (zero outside integration). These errors are not ordinary MH rejections.
- `HmcTransition`: acceptance, probability, attempted steps, finite energies/error
  when available, and optional `Divergence` reason.
- `Divergence`: non-finite position, momentum, density, incomplete/non-finite
  gradient, non-finite energy, or an excessive finite endpoint energy error.

The default cutoff rejects when `abs(H_proposal - H_initial) > 1000`; it can be
changed using `with_max_energy_error`. This **symmetric endpoint cutoff** is a
fixed-length design choice, not the one-sided stopping convention of a NUTS tree.
It is checked at the endpoint, not after every finite intermediate leapfrog step.
Divergent transitions have zero acceptance probability. Finite non-divergent
transitions use `exp(min(0, H_initial - H_proposal))`. Negative-infinite density is
a divergence here; models with boundaries should use unconstraining transforms
once those are implemented.

## Memory and performance contract

The chain owns eight dimension-sized, 64-byte-aligned numerical buffers: two for
each point, momentum, velocity, candidate position, and evaluation gradient scratch.
This excludes metric/model storage. No new unsafe code is used. Same-size
`PointState::clone_from` copies a coherent cache into existing storage; acceptance
swaps entire points without copying. The checked evaluator still fills gradients
with NaNs and scans inputs/outputs; successful proposal updates copy coordinates.
These passes are real costs, especially for cheap targets.

There are no sampler-owned steady-state allocations. User models, RNGs, and
metrics can allocate; this is not a blanket guarantee about every implementation
of those traits. The dedicated native counter measures complete Gaussian
two-dimensional transitions for identity/diagonal/dense metrics and an energy-divergence case,
after construction, with a positive allocation probe to check the counter works.
This Rust allocator counter does not observe allocation inside foreign libraries.

### Local transition baseline (2026-09-25)

Machine: Intel Core i5-8250U, x86-64 Linux. Pinned Rust 1.97.0-nightly
(2026-04-17), default features/scalar metric backend, release fat LTO, no explicit
native-CPU flags. Criterion: 30 samples, 500 ms warmup, 2 s measurement per case.
CPU frequency was not locked; no other validation/build jobs ran during timed
sampling, but ordinary desktop activity and thermal variation were uncontrolled.

Both cases use the same standard Gaussian, identity mass, step size 0.1, eight
leapfrog steps, and SmallRng seed 912. Construction is outside timing; the chain
advances throughout measurement. Legacy HMC performs nine fused evaluations per
transition and the checked path eight. Their safety/diagnostic contracts differ,
so this is an end-to-end comparison, not an isolated cost per validation scan.

Mean transition time, microseconds (95% bootstrap confidence intervals):

| Dimension | Legacy Hmc | Checked HmcChain |
|---|---:|---:|
| 8 | 1.080 [1.070, 1.089] | 1.012 [0.991, 1.036] |
| 64 | 6.728 [6.653, 6.821] | 6.281 [6.231, 6.332] |
| 1024 | 112.790 [107.341, 120.794] | 101.062 [100.324, 101.798] |

The checked path's mean elapsed time was approximately 6–10% lower in this run.
Several cases had outliers, especially the legacy 1024-dimensional case. Do not
extrapolate these results to other hardware, expensive targets, dense backends,
or effective samples per second. Retain checks until profiling identifies a
specific bottleneck and a replacement preserves the validated contracts.
The [machine-readable baseline](benchmarks/hmc-2026-09-25.csv) stores Criterion's
arithmetic-mean estimates and intervals in nanoseconds (not its console slope
estimates). The benchmark source is `crates/runtime/benches/hmc.rs`.

## Validation and reproduction

Tests cover exact fused-call counts, configuration/dimension failures, cache
coherence, model-error source retention, late failure and panic recovery,
non-finite/partial output rejection, ordinary rejection versus divergence,
metric-correct momentum, reversal for all three built-in mass types, and
second-order energy-error scaling. A seeded two-dimensional correlated Gaussian
checks empirical mean/covariance using a nontrivial dense mass (22,000 transitions,
2,000 discarded); it is not a general statistical certification.

```sh
nix develop --command cargo test --manifest-path crates/Cargo.toml --workspace --all-targets --locked
nix develop --command cargo test --manifest-path crates/Cargo.toml --workspace --doc --locked
nix develop --command cargo test --manifest-path crates/Cargo.toml -p runtime --lib --test hmc_chain --test hmc_chain_allocations --release --locked
nix develop --command cargo test --manifest-path crates/Cargo.toml -p runtime --lib --test hmc_chain --test hmc_chain_allocations --features simd --locked
nix develop --command cargo test --manifest-path crates/Cargo.toml -p runtime --lib --test hmc_chain --test hmc_chain_allocations --features faer --locked
nix develop .#blas --command cargo test --manifest-path crates/Cargo.toml -p runtime --lib --test hmc_chain --test hmc_chain_allocations --features simd,faer,openblas --locked
nix develop --command env MIRIFLAGS='-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check' cargo miri test --manifest-path crates/Cargo.toml -p runtime --lib --locked mcmc::hmc_chain
nix develop --command env MIRIFLAGS='-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check' cargo miri test --manifest-path crates/Cargo.toml -p runtime --test hmc_chain --locked
nix develop --command cargo bench --manifest-path crates/Cargo.toml -p runtime --bench hmc --locked -- --noplot
```

Miri uses the scalar backend and excludes the long statistical test and allocator
counter; native checks cover those. Foreign BLAS execution is not Miri-validated.

Validation record (x86-64 Linux, pinned toolchains): 109 default workspace tests
and 8 doctests passed, including benchmark smoke tests. The three HMC unit tests,
seven integration tests, and allocation regression passed in debug/release and
with SIMD, Faer, and combined SIMD/Faer/OpenBLAS features (backend precedence
selects OpenBLAS in the combined build). Strict Miri passed the three HMC unit
tests, six deterministic integration tests, and all 13 target/cache tests.
The long legacy latent statistical test was excluded from the focused Miri run;
it passes natively. Stable workspace/all-targets compilation, documentation,
formatting, and Clippy completed; existing unrelated lint warnings remain.

Increment 6 adds [all-metric Gaussian/banana moments, condition-1e8 Gaussian
validation, an independent scalar oracle, and allocation measurements through
dimension 129](hmc-validation.md). External reference fixtures, logistic/funnel
and rotated ill-conditioned cases, expensive-model benchmarks, arbitrary-size
backend allocation measurements, and CI enforcement remain open.
The Enzyme backend remains unvalidated; this work does not complete M1/M3.
