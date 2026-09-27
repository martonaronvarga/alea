# Alea: canonical engineering roadmap

Status: active, pre-alpha<br>
Last reviewed: 2026-09-24<br>
Last milestone update: 2026-09-27 — M0/M1 complete<br>
Scope: architecture, correctness, numerical validation, performance, and implementation order

This is the canonical roadmap for Alea. It replaces the earlier project plan. Keep it
current as design decisions are accepted and milestones are completed.

## Updating this document

For every material change:

1. Update the affected task and exit gate.
2. Record accepted design decisions in the decision log.
3. Link implementation, benchmark, or validation evidence.
4. Update `Last reviewed` after a whole-roadmap review.

Task markers are `[ ]` planned, `[~]` in progress, `[x]` complete with evidence, and
`[!]` blocked with the blocker stated. An algorithm is complete only when its correctness,
numerical, statistical, and performance gates pass.

## Implementation progress: 2026-09-27, increment 9 — M0 and M1 complete

M0's reproducible baseline and M1's safe-core exit gates now pass. The
[completion record](docs/m0-m1-completion.md) contains commands, exact counts,
scope boundaries and performance evidence.

- Centralized workspace metadata/dependencies/lints; warning-free four-way CI
  matrix; explicit experimental gates; recoverable archive; pinned toolchains.
- Audited all compiled repository-owned unsafe Rust. Cubature callbacks now
  initialize foreign scratch before forming references, contain panics until C
  returns, validate bounds/sizes/casts, and preserve typed error codes. Exact-match
  build-time C fixes repair foreign failure cleanup without modifying the submodule.
- Added acceptance-target/tree-depth newtypes, transform error vocabulary, and
  fallible legacy RWMH configuration/step/sampling boundaries.
- Published the consuming phase API and coherent persistent snapshots, including
  atomic updates, complete target-binding copies, and same-size allocation reuse.
- Added strict Miri callback/state tests, randomized shape/failure tests, a
  compile-fail ownership test, native C allocation-failure injection and allocation
  counters. Local ASan/UBSan checks of the C harness also pass.

Validation: default debug/release and Faer each pass 144 tests plus 6 doctests;
SIMD and combined OpenBLAS each pass 145 plus 6. All four feature matrix jobs pass
format/check/Clippy/docs with warnings denied. Strict Miri passes 64 tests, with
six documented native/statistical exclusions. Repeated target/HMC/phase operations
retain zero-allocation gates. No throughput improvement or remote CI run is claimed.

M2 transforms/Enzyme execution, M3's full external statistical/performance gates,
adaptation, NUTS tree selection, ChEES-HMC and RMHMC remain open. Legacy mutable
state APIs remain compatibility paths, not substitutes for the new atomic core.

## Implementation progress: 2026-09-27, increment 8 (historical)

The deterministic external endpoint subgate now has executed, version-pinned
BlackJAX evidence. Production HMC code and its public API are unchanged.

- BlackJAX 1.5/JAX 0.9.2 CPU-float64 execution generated 36 reference trajectories:
  correlated Gaussian/banana, identity/diagonal/dense mass, signed steps, and
  lengths 1/5/11. JAX supplies autodiff gradients independently of Rust analytics.
- Private Rust tests compare initial/endpoint position, momentum, log density,
  gradient, and Hamiltonian through the production phase integrator, then reverse
  each trajectory. Basis-vector checks make mass/inverse-mass conventions explicit.
- An optional content-pinned Nix Python environment and stdout generator support
  regeneration; the saved CSV reproduces byte-for-byte locally. Normal Rust tests
  require no Python, network access, or new dependencies.

See [provenance, case definitions, tolerances, and reproduction](docs/blackjax-reference.md).
This completes deterministic external endpoint comparison, not M3's broader
statistical gate. Logistic/funnel/rotated targets, external sampling comparisons,
CI/FFI audits, persistent phase/tree state, adaptation, and Enzyme remain open.
Validation: 124 workspace tests and 8 doctests pass. All 36 reference cases pass
debug/release, stable Rust, SIMD, Faer, combined OpenBLAS, and strict Miri through three unit
tests. Existing allocation regressions pass; Clippy completes with existing
warnings and formatting passes. No production-path performance change is claimed.

## Implementation progress: 2026-09-27, increment 7 (historical)

`HmcChain` now uses a reusable crate-private Hamiltonian phase/integrator boundary.
The public HMC API and transition policy are unchanged; legacy HMC is untouched.

- `PhaseWorkspace` retains the same six proposal/scratch buffers, plus two in the
  live point. Starting copies a coherent cache without another target evaluation.
- A borrowed, non-cloneable `Phase` token is consumed by leapfrog. Only success
  returns a token that can advance or commit; failure/unwinding leaves reusable
  but inaccessible partial scratch. Dropping a phase leaves the live point intact.
- Signed nonzero steps are validated outside the hot loop. The sampler keeps its
  positive-step configuration, endpoint cutoff, MH decision, and typed diagnostics.
- Test-only explicit momentum feeds the production integrator for reversal and
  order tests, and provides the entry point for upcoming external fixtures.

See [phase ownership, failure semantics, and validation](docs/hamiltonian-phase.md).
The independent transition oracle and allocation gates remain in place. External
BlackJAX fixtures, persistent phase/tree state, adaptation, and Enzyme remain open.
Validation: 121 workspace tests and 8 doctests pass. Runtime tests pass release,
SIMD, Faer, and combined OpenBLAS builds. Six phase tests plus ten deterministic
HMC integration tests pass strict Miri. Allocation counters remain zero through
dimension 129 in the tested paths. Stable compilation/docs/formatting pass;
Clippy completes with existing warnings.

The [post-extraction timing record](docs/hamiltonian-phase.md#local-timing-follow-up-2026-09-27)
and [raw estimates](docs/benchmarks/hmc-2026-09-27.csv) are retained. Small-case
intervals overlap with legacy; the 1024-dimensional checked path took about 9.6%
less time in this run. Uncontrolled cross-run timing changes are not credited to
the refactor.

## Implementation progress: 2026-09-25, increment 6 (historical)

The HMC validation gate now covers non-Gaussian and ill-conditioned analytic
targets. This increment adds test infrastructure, not a production algorithm or
new sampler; existing safety checks and transition semantics are unchanged.

- An independent scalar oracle compares 96 complete transitions, including dense
  mass actions, both energies, acceptance decisions, cache fields, and RNG use.
  Analytic fixture gradients are independently checked by central differences.
- Correlated Gaussian and banana stationary moments are checked with all three
  mass types, using two seeds per configuration. A Gaussian with covariance
  condition number `1e8` is tested with explicitly matched mass. Rejected states
  remain in moment calculations; no ESS/convergence certification is claimed.
- Unstable unmatched geometry and actual floating-point overflow regressions
  check typed divergence and exact live-cache preservation.
- Allocation tests now cover dimensions 7, 33, and 129 with identity, diagonal,
  and dense mass, requiring complete trajectories inside the measured region.

See [validation definitions, tolerances, and remaining reference gate](docs/hmc-validation.md).
The scalar oracle is not an executed Stan/BlackJAX reference: that gate, general
phase-state design, logistic/funnel targets, and rotated ill-conditioning remain
open. No NUTS/adaptation work starts in this increment.
Validation: 117 workspace tests and 8 doctests pass. The new suite passes release,
SIMD, Faer, and combined OpenBLAS builds; zero Rust allocations are observed in
the tested complete trajectories. Four deterministic tests pass strict Miri.
Stable compilation/formatting pass; Clippy completes with existing warnings.

## Implementation progress: 2026-09-25, increment 5 (historical)

The additive `runtime::HmcChain` now consumes the fallible target protocol and
target-bound caches. Legacy `Hmc` remains available and retains its necessary
starting refresh; the new path requires exactly `L` fused calls for a successful
`L`-step trajectory after one initialization call.

- Validated `StepSize`/`HmcOptions` reject invalid configuration before execution.
- A separate proposal and reusable aligned scratch keep the live point intact on
  rejection, model error, and unwinding. Acceptance swaps a complete validated cache.
- Model errors preserve their sources; typed transition information distinguishes
  ordinary rejection, non-finite trajectories, and excessive absolute endpoint
  energy error. This fixed-length cutoff is not a NUTS divergence convention.
- Tests cover call counts, late failure/recovery, cache coherence, metric-aware
  momentum, reversal for all three mass types, second-order energy error, and a
  seeded correlated-Gaussian/dense-mass sampling check.
- Native allocation counters and a cheap-gradient Criterion comparison provide
  performance evidence without claiming that checked evaluation is free.
  The [local baseline](docs/benchmarks/hmc-2026-09-25.csv) measured approximately
  6–10% less elapsed time for the new path across dimensions 8/64/1024; this is
  one uncontrolled-frequency run, not a general throughput/ESS guarantee.

See [HMC contract, validation, and performance](docs/hmc-chain.md). General phase
state, broader reference/statistical tests, adaptation, and Enzyme remain open;
this does not complete M1/M3 or start NUTS.
Validation: 109 workspace tests and 8 doctests pass. New HMC regressions pass
release, SIMD, Faer, and combined OpenBLAS builds; measured Rust allocations are
zero in the tested transition paths. Strict Miri passes 3 HMC unit tests,
6 deterministic transition tests, and 13 target/cache tests. Stable compilation,
docs, and formatting pass; Clippy retains existing warnings.

## Implementation progress: 2026-09-25, increment 4 (historical)

The fallible target protocol and target-bound transactional point cache are now
implemented in `kernels::target`. This is an additive M1 foundation; existing HMC
still uses the old state API and retains its necessary starting cache refresh.

- `LogDensityGradient` declares a fixed dimension and returns concrete model
  errors. `evaluate` validates shapes, finite inputs, finite log density, and a
  complete finite gradient around one fused call. Model error sources are retained.
- Gradient scratch is NaN-filled before evaluation to reject incomplete writes;
  its linear cost is explicit, not hidden behind a throughput claim.
- `FusedAdapter` provides a dimension-declared migration path for legacy fused
  models without invoking their split value/gradient APIs.
- `PointState` privately owns aligned position/gradient storage and borrows one
  target. `EvaluationWorkspace` lets updates commit all cache fields only after
  success; errors/unwinding leave the old point intact. No mutable legacy state
  trait or per-update target replacement is exposed.
- Regression tests cover randomized dimensions, non-finite and partial outputs,
  error-source preservation, panic recovery, buffer reuse, and fused call counts.
  A native allocation counter observes zero allocations/bytes after construction
  for successful Gaussian updates and the tested allocation-free failure paths.

See [target/state contract and migration](docs/target-state.md) for costs,
validation commands, purity requirements, and the precise remaining HMC work.
Validation: 97 workspace tests and 7 doctests pass; the 12 new correctness tests
also pass strict Miri and release builds. Allocation regressions pass in debug
and release. Stable compilation/docs/formatting pass; Clippy retains existing warnings.
Phase state, fallible integration, typed sampler configuration/statistics, and
removing the legacy HMC cache refresh remain open; this does not complete M1/M3.

## Implementation progress: 2026-09-25, increment 3 (historical)

The Euclidean metric boundary now rejects invalid input explicitly. This is a
bounded M1/M3 increment, not completion of either milestone.

- Diagonal mass and Cholesky-factor constructors return `MetricError`, replacing
  silent positive-value coercion and constructor panics. Matrix storage sizes and
  byte counts use checked arithmetic; all factor entries are validated.
- Factors are full column-major lower-triangular matrices with positive diagonal
  and zero upper triangle. Constructors retain aligned storage without copying.
  They do not regularize, factorize mass matrices, or certify conditioning.
- Fallible metric actions reject mismatched vector lengths without mutating output.
  Existing infallible actions also enforce lengths in release mode before writing.
- The BLAS boundary checks lengths/integer conversion, skips empty operations,
  and documents the unsafe-call invariants. Other FFI boundaries remain unaudited.
- Backend testing exposed missing BLAS linkage and an ILP64/LP64 mismatch in the
  Nix shell. The optional feature now discovers/links LP64 OpenBLAS with pkg-config
  and a compile-time header ABI guard; optional Nix shells select matching LP64.
- Backend-independent regression tests cover invalid values, overflow, dimension
  failures, mass conventions, and independent matrix-reference calculations across
  block boundaries. Existing callers are migrated to fallible construction.

The [metric contract](docs/metric-contract.md) records API migration, validation
commands, and limits. The default workspace passes 84 tests and 5 doctests;
metric tests pass on scalar/SIMD/Faer/LP64 OpenBLAS and in scalar release mode;
combined backend features and OpenBLAS HMC regressions pass. All eight metric tests
also pass strict-provenance/symbolic-alignment Miri on the default backend. Stable
workspace/all-targets compilation passes. Clippy runs with existing warnings.
Typed target evaluation, atomic state, sampler configuration,
allocation counters, CI, and general FFI coverage remain open.

## Implementation progress: 2026-09-24, increment 2 (historical)

Toolchain availability and explicit SIMD alignment are now implemented. This
supersedes increment 1's temporary choice of ordinary `Vec` backing; it was a
safety repair, not a claim that `Vec` was fastest.

- The default Nix shell directly provides a lock-pinned, same-date nightly with
  Clippy, Miri, rust-src, rustfmt, and rust-analyzer. No `.nix-rust` link is needed.
  Stable, BLAS, full research, and experimental autodiff shells are separate.
  Native-CPU, BLAS-link, and autodiff flags are no longer imposed globally.
- The experimental source build now has a dated Rust archive/hash and a
  Nixpkgs-release-pinned Enzyme source. It remains **unbuilt/unvalidated** in this
  increment; successful shell/toolchain evaluation is not an Enzyme correctness gate.
- `OwnedBuffer<T>` now wraps `AVec<T, ConstAlign<64>>` in the isolated `memory`
  crate, re-exported through its original `kernels::buffer` path. Allocations have
  alignment `max(64, align_of::<T>())`, remain tightly packed, and expose initialized
  values only. `from_fn` avoids initializing values twice. Generic drop and panic
  cleanup are tested; no new local unsafe block was introduced.
- All 6 storage tests pass Miri with strict provenance and symbolic alignment
  checks. This covers the wrapper's selected dependency APIs, not all unsafe code
  elsewhere in the repository. FFI audits remain open.
- Workspace Clippy now executes successfully with existing warnings; the new
  memory crate is checked separately with warnings denied, including its SIMD bench.
- A scalar-reference regression exercises SIMD metric operations on dimensions
  around lane/block boundaries and deliberately unaligned subslices. Base alignment
  does not justify aligned loads from arbitrary triangular subranges.
- A repeatable storage benchmark separates allocation/initialization from reused
  AXPY working sets, comparing ordinary Vec, aligned storage, and explicit SIMD.
  Results do not establish a globally fastest container or end-to-end HMC speedup.

Commands, toolchain versions, storage invariants, benchmark methodology, and
validation evidence live in [toolchains and storage](docs/toolchains-and-storage.md).
M0/M1 remain in progress: CI, lint cleanup, the remaining unsafe/FFI audits, typed
errors, and validated state/metric boundaries are not complete.

## Implementation progress: 2026-09-24, increment 1 (historical)

M0 and M1 are in progress. Targeted repairs to the existing HMC prototype also
address M3 defects; this does **not** complete M1 or M3 or authorize starting NUTS.

Delivered:

- Replaced the unsafe generic allocator with initialized `Vec<T>` storage in
  [buffer.rs](crates/kernels/src/buffer.rs). Growth initializes elements, shrinking
  drops them, and capacity is reused. Tests cover nontrivial destructors, zero-sized
  and over-aligned elements. There is no extra SIMD alignment guarantee.
- Wiener4/5 branch buckets now push initialized entries into a capacity-reserved
  `Vec`; invalid observations never expose uninitialized scratch entries.
  [Batch regressions](crates/kernels/tests/wiener_batches.rs) compare both strategies.
  Bucket evaluation still allocates per call; a chain-local workspace is pending.
- Repaired stale state/RNG imports and associated-type bounds across runtime,
  examples, and benchmarks. The Gaussian reference target now has native Rust
  scalar and fused implementations, without the unbuilt CXX Gaussian bridge.
- Reduced compiled DDM exports to `ddm::latent`. Other DDM sketches are preserved
  on disk, uncompiled; no experimental source/artifact cleanup was performed.
- Latent random walks now draw centered normal increments with the declared scale.
  Scale fields are private to preserve constructor validation. Zero scale is an
  exact point mass. Tests check mean, variance, fourth moment, and zero/invalid scales.
- HMC draws `p = L z` for `M = L L^T`, uses fused target evaluation in leapfrog,
  commits accepted position/value/gradient together, and rejects non-finite paths.
  Publicly mutable state forces one cache refresh at each transition start:
  `L + 1` fused calls for `L` leapfrog steps. Eliminating this extra evaluation
  requires M1's validated atomic state contract.
- [HMC regressions](crates/runtime/tests/hmc.rs) cover diagonal/dense momentum,
  fused call counts, cache consistency, non-finite rejection, forward/reverse
  recovery, second-order energy error, and fixed-seed Gaussian moments.
- Applied workspace rustfmt and replaced aspirational README layouts with current
  capabilities and explicit limitations.

Verification commands (run from `crates/`; default features unless stated):

```sh
cargo fmt --all -- --check
cargo check --workspace --all-targets --locked
cargo test --workspace --all-targets --locked
cargo test --workspace --doc --locked
cargo doc --workspace --no-deps --locked
cargo test -p runtime --test hmc --features faer --locked
```

All six commands passed locally with Rust 1.97.1: 73 unit/integration tests,
4 doctests, Criterion benchmark smoke tests, and all 6 HMC regressions with
`faer`. This increment adds 13 regression tests. Existing dead-code/unused-import
and vendored C signedness warnings remain; passing is not warning-free.

At the end of increment 1, Clippy and Miri were not installed (resolved in increment 2);
CI/toolchain pinning, FFI safety audit, typed evaluation/transition errors, validated
metric constructors, modern diagnostics, and allocation/ESS benchmarks remain open.
Default tests do not validate SIMD, OpenBLAS, or std::autodiff/Enzyme. Benchmark
smoke tests are not performance measurements; no speedup is claimed.

## Mission

Alea is a programmable, algebraic, strongly typed Monte Carlo library inspired by Stan,
JAX, BlackJAX, nuts-rs, and related systems. Its primary goal is trustworthy
high-performance inference.

```text
typed model and parameter transforms
                  |
                  v
fused unconstrained log-density and gradient
                  |
                  v
Euclidean Hamiltonian mechanics
                  |
                  v
HMC -> adaptation -> multinomial NUTS
                  |
                  v
multi-chain runtime and diagnostics
                  |
          +-------+-------+
          |               |
          v               v
     ChEES-HMC           RMHMC
                          |
                          v
             adaptive manifold methods
```

## Non-negotiable principles

### Correctness before feature count

- Memory, density, gradient, RNG, and integrator correctness block sampler work.
- Approximate targets must never masquerade as exact targets.
- Public diagnostics must implement their stated modern definitions.
- Invalid input and evaluation failures use typed errors, not booleans, `Option`, public
  assertions, or silent coercion.

### Performance is measured

- Measure allocations, target/gradient evaluations, and transition time.
- Measure bulk/tail ESS per second and per gradient evaluation.
- Keep scalar references beside SIMD, BLAS, and specialized paths.
- Profile before adding unsafe code, hand-written SIMD, or new backends.
- Never trade stationary-distribution correctness for iteration throughput.

### Hot paths are static and allocation-free

- Prefer concrete types/generics for models, metrics, and integrators.
- Use dynamic dispatch only at genuine orchestration/plugin boundaries.
- Allocate chain state and workspaces once and reuse them.
- Perform one fused value-and-gradient evaluation per new Hamiltonian position.

### Samplers use unconstrained coordinates

- Models may expose strongly typed constrained parameters.
- HMC-family samplers use a flat unconstrained `f64` vector.
- Transforms contribute log absolute Jacobian determinants.
- Transform dimensions and layouts are explicit and validated.

### Parallelism is deterministic

- Derive independent, reproducible RNG streams per chain.
- Use Rayon/scoped CPU threads for numerical work.
- Reserve Tokio for optional I/O, cancellation, or service control.
- Sequential and parallel runs with equal per-chain seeds produce equal chain results.

## Baseline review: 2026-09-24

### Strengths

- The Wiener/DDM numerical kernel is the repository's strongest component.
- `kernels` and `ffi` pass 59 tests: 57 kernel/Wiener and 2 FFI tests.
- Tests cover Stan reference values, finite-difference gradients, transforms, boundary
  symmetry, and fixed/adaptive quadrature agreement.
- The code already explores reusable buffers, SoA batches, static dispatch, Criterion,
  `thiserror`, and `tracing`.

### Initial health (before increment 1)

- `cargo fmt --all -- --check` fails.
- `cargo check --workspace --all-targets` fails with 24 errors.
- Clippy is absent from the selected toolchain.
- NUTS is only configuration and a shell type.
- Particle-filter and particle-MCMC implementation files are empty.
- DDM modules contain stale/broken exports.
- Generated Wiener alternatives and compiled experiments need archiving/removal.

### Blocking findings

#### P0: `OwnedBuffer<T>` is unsound

`crates/kernels/src/buffer.rs` exposes uninitialized allocations as initialized slices and
also mishandles generic alignment, growth initialization, and dropping nontrivial `T`.

Decision: replace the unsound allocator with initialized storage. Explicit SIMD base
alignment is now a user requirement: use the safe aligned-storage wrapper described
in increment 2, with Miri coverage and separate performance measurements. Do not infer
a speedup merely from stronger alignment.

#### P0: latent random-walk sampling disagrees with its density

`LatentRandomWalk::sample_next` generates positive uniform increments while
`log_transition` evaluates centered Gaussian increments.

Decision: use a tested normal sampler and property-test sample/transition consistency.

#### P1: dense-metric HMC samples incorrect momentum

The HMC path always draws independent standard-normal momentum. For kinetic energy
`p^T M^-1 p / 2`, momentum must follow `N(0, M)`.

Decision: make momentum sampling, velocity, and kinetic energy explicit metric operations.

#### P1: density and gradient evaluation are split

`FusedLogDensity` exists but is not the primary integrator/HMC contract, duplicating the
dominant computation and permitting stale cached state.

Decision: foundational model evaluation returns log density and writes its gradient in one
operation.

#### P1: diagnostics are mislabeled

The current `ess_bulk` is a simple single-chain autocorrelation estimate, not modern
rank-normalized bulk ESS. Split R-hat is neither rank-normalized nor folded.

Decision: do not expose those public names until modern cross-chain definitions are
implemented and reference-tested.

## Target architecture

The target crate organization is:

```text
alea-core          target protocol, transforms, state, errors
alea-math          metrics, linear algebra, numerical primitives
alea-autodiff      std::autodiff/Enzyme adapters and validation
alea-distributions distributions, including Wiener/DDM
alea-mcmc          HMC, NUTS, ChEES-HMC, RMHMC
alea-smc           particle filters and particle MCMC
alea-ffi           audited foreign boundaries only
alea-runtime       chains, parallelism, streaming output
alea-cli           binaries, configuration, user-facing context
```

This is not a flag-day rename. Move boundaries after core contracts stabilize. In
particular, numerical core crates must not depend on the broad FFI crate.

### Primary model contract

```rust
pub trait LogDensityGradient {
    type Error: std::error::Error + 'static;

    fn dimension(&self) -> usize;

    fn logp_grad(
        &self,
        position: &[f64],
        gradient: &mut [f64],
    ) -> Result<f64, Self::Error>;
}
```

Requirements:

- dimensions are validated at an outer boundary;
- evaluation is semantically pure;
- scratch storage is chain-local or explicitly synchronized;
- model-domain and non-finite failures are typed;
- analytic, Enzyme, and FFI gradients share the contract.

### State and transition model

```text
PointState = position + log density + gradient
PhasePoint = PointState + momentum + Hamiltonian energy
Workspace  = chain/kernel-owned reusable temporary buffers
```

Log density and gradient update atomically. Replace `Kernel::step(...) -> bool` with typed
transition information containing acceptance probability, energy/error, divergence,
leapfrog count, and tree depth where applicable.

### Euclidean metric contract

```rust
pub trait EuclideanMetric {
    fn dimension(&self) -> usize;
    fn sample_momentum(&self, z: &[f64], momentum: &mut [f64]);
    fn velocity(&self, momentum: &[f64], velocity: &mut [f64]);
    fn kinetic_energy(&self, momentum: &[f64], scratch: &mut [f64]) -> f64;
}
```

Type names and documentation must say whether stored matrices are mass, inverse mass,
covariance, or precision.

## Autodiff strategy

`std::autodiff`/Enzyme is the intended compiled AD path but remains an experimental
toolchain dependency. At review time it is nightly-only, requires fat LTO, does not support
differentiated `dyn Trait` calls, and is less reliable in debug builds.

Support these providers:

```text
AnalyticGradient      reference/performance-critical distributions
EnzymeGradient        default compiled Rust model path
FfiGradient           validated external implementations
FiniteDifference      tests and diagnostics only
```

Rules:

- Pin exact nightly and LLVM/Enzyme/Nix inputs.
- Test AD in release mode with fat LTO.
- Keep differentiated functions concrete, pure, and statically dispatched.
- Never differentiate through async/coroutine code.
- Give cubature, FFI, and specialized kernels explicit derivative rules when opaque to AD.
- Compare generated gradients with analytic, finite-difference, and external references.
- Keep stable builds usable via analytic/FFI gradients.
- For RMHMC, plan for Hessian-vector products and metric derivatives, not only Hessians.

## Milestones

### M0 — repository stabilization (complete)

Goal: a green, reproducible baseline.

- [x] Make the default workspace compile and format cleanly (increment 1).
- [x] Install/configure Clippy and inherited workspace lints; enforce warning-free CI.
- [x] Gate unfinished NUTS/SMC/policy behind `experimental`; DDM exports only latent adapters.
- [x] Centralize workspace metadata and dependency versions.
- [x] Make supported Cargo features additive; verify scalar, Faer, SIMD, and combined BLAS builds.
- [x] Archive alternatives and compiled benchmark artifacts with a SHA-256 recovery manifest.
- [x] Add CI for format, check, test, docs, feature combinations, and Miri.
- [x] Pin stable/nightly tools and the experimental Enzyme source inputs. Actual Enzyme
  compiler execution/adapter validation is M2; `std-autodiff` fails with an explicit diagnostic.

Completed locally on 2026-09-27: all four native CI matrix commands passed, including
Clippy/docs with warnings denied. See [stabilization evidence and commands](docs/stabilization.md).
The workflow is installed; this does not assert a remote GitHub Actions run or branch protection.

Exit gate: clean format, check, test, docs, and Clippy for the stable feature set;
experimental features fail clearly when prerequisites are absent.

### M1 — safe core and target protocol (complete)

Goal: trustworthy model evaluation and state.

- [x] Replace `OwnedBuffer<T>` with initialized aligned storage (increment 2,
  `memory` tests and Miri; increment 1's ordinary Vec backing was temporary).
- [x] Add strict Miri coverage to the supported Rust unsafe boundaries, including
  the production FFI trampoline; complement non-interpretable C/BLAS execution
  with documented native tests and an [unsafe audit](docs/unsafe-audit.md).
- [x] Implement fused `LogDensityGradient`. Dimension-aware fallible protocol,
  checked evaluation, and a legacy fused adapter implemented in increment 4;
  target-bound HMC consumer implemented in increment 5. Legacy callers remain;
  Enzyme adapters belong to M2.
- [x] Add typed evaluation, dimension, transform, metric, and sampler errors.
  Metric construction/vector-dimension errors implemented in increment 3;
  target evaluation/dimension errors in increment 4 and HMC errors in increment 5.
  Increment 9 adds typed RWMH configuration/shape/storage errors and transform
  error vocabulary; transform implementations/Jacobians are M2.
- [x] Add validated newtypes for step size, acceptance target, and tree depth.
  Step size implemented in increment 5; acceptance target/tree depth in increment 9.
- [x] Define atomic point/phase state and chain-local workspaces. Target-bound
  transactional point and reusable evaluation workspace implemented in increment 4;
  private HMC workspace implemented in increment 5 and extracted into a reusable
  consuming phase-token boundary in increment 7. Increment 9 publishes the phase
  API and reusable coherent endpoint snapshots; NUTS-specific tree assembly and
  selection are M5, not part of this state-storage prerequisite.
- [x] Fix/test the latent Gaussian transition (increment 1, latent unit tests).
- [x] Audit FFI panic containment, pointer invariants, casts, and callback lifetimes;
  fix discovered Rust boundary issues and C failure-path ownership leaks.

Exit gate: no unaudited unsafe code; Miri passes; repeated target evaluation allocates
nothing; randomized tests cover dimensions and non-finite failures.

Exit gate passed locally on 2026-09-27; see [completion evidence](docs/m0-m1-completion.md).
Foreign C/BLAS is natively validated, not claimed to execute under Miri.

### M2 — transforms and Enzyme backend

Goal: a Stan-like unconstrained model boundary.

- [ ] Scalar identity, lower, upper, and interval transforms.
- [ ] Positive, ordered, positive-ordered, simplex, and unit-vector transforms.
- [ ] Cholesky covariance/correlation transforms.
- [ ] Covariance/correlation matrix transforms after Cholesky validation.
- [ ] Flat parameter layout and dimension accounting.
- [ ] Fuse constrain + Jacobian + model evaluation.
- [ ] Analytic and Enzyme target adapters.
- [ ] Custom derivatives for opaque primitives.
- [ ] Finite-difference gradient diagnostic command.

Exit gate: transform round trips/Jacobians pass property tests; gradients agree on
Gaussian, correlated Gaussian, logistic, banana, funnel, and Wiener targets.

### M3 — correct Euclidean HMC

Goal: small, correct, allocation-free fixed-length HMC.

- [~] Identity, diagonal, and dense metrics: validated mass/factor constructors
  and dimension-safe operations implemented in increment 3; final protocol and
  allocation/performance gates remain open.
- [x] Correct metric-aware momentum generation (diagonal/dense regression tests).
- [~] Reversible velocity Verlet/leapfrog (new fallible path tested for reversal
  with identity/diagonal/dense masses and identity Gaussian energy-error order;
  reusable signed phase boundary implemented in increment 7; executed BlackJAX
  endpoint and signed-reversal fixtures added in increment 8).
- [~] Fixed-length HMC with Metropolis correction (`HmcChain` implements the new
  target contract; increment 6 adds an independent scalar transition oracle,
  all-metric correlated Gaussian/banana moments, and matched-mass condition-1e8
  Gaussian tests. Increment 8 adds deterministic BlackJAX references; external
  statistical comparisons and broader validation remain pending).
- [~] Divergence detection from non-finite evaluation/excess energy error.
  New HMC path reports typed reasons and a symmetric absolute endpoint threshold;
  increment 6 covers real momentum/position/density overflow and an unstable
  ill-conditioned target. Broader numerical stress tests remain.
- [~] Reusable integrator/proposal storage (implemented; native Gaussian transition
  allocation regressions added in increment 5 and extended through dimension 129
  across backends in increment 6; arbitrary model/size and foreign allocation gates remain).
- [x] Typed fixed-length HMC transition statistics (`HmcTransition`, increment 5).

Exit gate: forward/reverse recovery, expected energy-error convergence, correct Gaussian
moments/covariance for all metrics, zero allocations per leapfrog step, and agreement with
Stan/BlackJAX reference targets.

### M4 — step-size and metric adaptation

Goal: robust automatic warmup.

- [ ] Reasonable initial step-size search.
- [ ] Dual-averaging step-size adaptation.
- [ ] Welford online variance/covariance.
- [ ] Regularized diagonal metric adaptation.
- [ ] Opt-in dense metric adaptation.
- [ ] Stan-style initial fast, expanding slow, and final fast windows.
- [ ] Correct reinitialization after metric changes.
- [ ] Freeze adaptation before retained draws.

Exit gate: improved ESS per gradient on correlated targets without bias; schedules match
Stan/BlackJAX; fixed seeds yield deterministic adaptation.

### M5 — multinomial NUTS

Goal: modern dynamic HMC.

- [ ] Iterative, allocation-bounded trajectory construction.
- [ ] Random forward/backward doubling.
- [ ] Multinomial proposal selection in log space.
- [ ] Metric-aware generalized U-turn criterion.
- [ ] Divergence, U-turn, and maximum-depth termination.
- [ ] Full tree/energy/acceptance statistics.
- [ ] Integration with M4 adaptation.

Exit gate: moments and diagnostics agree with Stan, BlackJAX, and nuts-rs; no allocation
proportional to tree size; edge cases have reference tests.

### M6 — multi-chain runtime and diagnostics

Goal: deterministic parallel inference and trustworthy diagnostics.

- [ ] Reproducible independent per-chain RNG streams.
- [ ] Sequential and Rayon-parallel runners with stable ordering.
- [ ] Streaming draws and bounded-memory summaries.
- [ ] Rank-normalized split and folded R-hat.
- [ ] Cross-chain bulk/tail ESS and MCSE.
- [ ] E-BFMI, divergences, and tree-depth saturation.
- [ ] Stable draw/warmup/transition output schema.
- [ ] Cancellation/progress outside numerical hot loops.

Exit gate: sequential/parallel per-chain equality for equal seeds; diagnostics agree with
Stan/ArviZ fixtures; streaming memory is independent of retained draw count.

### M7 — ChEES-HMC

Goal: ensemble-adapted static HMC as a NUTS alternative.

- [ ] Shared chain-ensemble adaptation state.
- [ ] ChEES objective and trajectory-length adaptation.
- [ ] Joint warmup of step size and trajectory length.
- [ ] Randomized integration length.
- [ ] Optional difficult-direction criteria, benchmarked separately.
- [ ] Freeze adaptation before retained sampling.

Exit gate: reproduce published behavior; compare ESS/gradient and ESS/second against NUTS
and fixed HMC; ensemble scheduling remains deterministic.

### M8 — Riemannian HMC

Goal: position-dependent geometry after Euclidean mechanics are proven.

- [ ] Position-dependent metric contract.
- [ ] Metric log determinant, inverse solve/action, and derivatives.
- [ ] Generalized implicit leapfrog with bounded iterations and diagnostics.
- [ ] Experimental SoftAbs Hessian metric.
- [ ] Fisher metrics for selected model families.
- [ ] Analytic/Enzyme Hessian-vector and metric-derivative operations.
- [ ] Explicit events for failed implicit solves.

Exit gate: reversibility and stationary-distribution tests pass; published logistic and
hierarchical targets agree; costs separate derivatives, factorization, solves, and dynamics.

### M9 — adaptive manifold methods

Goal: evaluate Wang et al. and later adaptations on validated RMHMC.

- [ ] Independently reproduce relevant algorithms.
- [ ] Separate warmup-only adaptation from valid transition mechanics.
- [ ] Compare with Euclidean NUTS, ChEES-HMC, and base RMHMC.
- [ ] Promote only methods with correctness and measured efficiency evidence.

Exit gate: reproducible paper benchmarks, stationarity tests, and a written support/reject
decision for each method.

### M10 — SMC and particle MCMC

Goal: state-space inference independent of Hamiltonian internals.

- [ ] Stable log-weight normalization and particle ESS.
- [ ] Multinomial, stratified, systematic, and residual resampling.
- [ ] Bootstrap/auxiliary filter core with SoA particles.
- [ ] Deterministic parallel propagation/weighting.
- [ ] Conditional SMC, particle Gibbs, and PMMH.
- [ ] Reuse core model, RNG, diagnostic, and output infrastructure.

Exit gate: unbiased likelihood-estimator tests where applicable; agreement on analytically
solvable linear-Gaussian models; resampling passes distributional/boundary tests.

## Validation suite

| Target | Purpose |
|---|---|
| independent Gaussian | moments and initialization |
| correlated Gaussian | metric and covariance adaptation |
| ill-conditioned Gaussian | stability and preconditioning |
| banana | nonlinear geometry |
| Neal funnel | divergences and hierarchy |
| logistic regression | realistic smooth posterior |
| hierarchical normal | centered/non-centered behavior |
| Wiener/DDM | project-specific likelihood/gradient cost |
| RMHMC paper targets | position-dependent geometry |

Validation layers:

1. Unit tests for transforms, metrics, RNG, integrators, and tree operations.
2. Property tests for round trips, reversibility, dimensions, and resampling.
3. Derivative checks against analytic and finite-difference references.
4. Fixed-seed statistical moment/coverage tests.
5. Cross-implementation comparisons with Stan, BlackJAX, and nuts-rs.
6. Simulation-based calibration for end-to-end model APIs.
7. Criterion microbenchmarks and end-to-end ESS benchmarks.
8. Miri/sanitizers for unsafe and FFI boundaries.

## Performance scorecard

Track per model and dimension:

- time per log density and fused gradient;
- target/gradient evaluations and allocations per transition;
- metric factorization/solve and integrator overhead;
- peak memory per chain;
- bulk/tail ESS per second and ESS per gradient;
- divergence and maximum-depth rates;
- E-BFMI;
- warmup and retained-draw time separately.

Preserve inputs/toolchains. Compare backends only when semantics match. Never select a
backend from one dimension or microbenchmark. Keep symbols in benchmark builds; consider
PGO and native CPU flags only after the portable path is stable.

## Error, observability, and documentation policy

- Libraries use typed `thiserror`; applications may add `anyhow` context.
- Recoverable evaluation failures never panic; source chains are preserved.
- Libraries emit structured `tracing` events but install no subscriber.
- Disabled hot-loop tracing must not format or allocate.
- Every public item is documented; fallible and unsafe APIs include `# Errors`/`# Safety`.
- Core traits document semantic laws.
- The README describes supported reality, not aspirational layouts.
- Experimental algorithms and features are labeled explicitly.

## Deferred until M0-M6 are complete

- GPU/Futhark likelihoods and particle operations;
- PyO3/maturin bindings;
- variational inference and Gibbs/mixed inference;
- specialized Kalman/RTS methods;
- neural likelihood surrogates;
- factor graphs and belief propagation;
- distributed execution.

## Primary references

### Algorithms

- Stan MCMC/warmup: <https://mc-stan.org/docs/reference-manual/mcmc.html>
- Stan transforms: <https://mc-stan.org/docs/reference-manual/transforms.html>
- BlackJAX: <https://blackjax-devs.github.io/blackjax/>
- nuts-rs: <https://github.com/pymc-devs/nuts-rs>
- ChEES-HMC: <https://proceedings.mlr.press/v130/hoffman21a.html>
- Girolami-Calderhead RMHMC: <https://arxiv.org/abs/0907.1100>
- Wang et al. (2013): <https://proceedings.mlr.press/v28/wang13e.pdf>

### Autodiff and transformation architecture

- Rust `std::autodiff`: <https://doc.rust-lang.org/std/autodiff/index.html>
- Enzyme: <https://github.com/EnzymeAD/Enzyme>
- JAX transformations: <https://docs.jax.dev/en/latest/101/transformations.html>
- JAX custom derivatives: <https://docs.jax.dev/en/latest/301/custom-jvp-vjp.html>

### Rust engineering

- Rust API Guidelines: <https://rust-lang.github.io/api-guidelines/>
- Rustonomicon: <https://doc.rust-lang.org/nomicon/>
- Rust Performance Book: <https://nnethercote.github.io/perf-book/>
- Tokio: <https://tokio.rs/>
- `thiserror`: <https://docs.rs/thiserror/>
- `anyhow`: <https://docs.rs/anyhow/>

## Decision log

| Date | Decision | Status |
|---|---|---|
| 2026-09-24 | Make this file the canonical engineering roadmap. | accepted |
| 2026-09-24 | Prioritize safety and a green workspace before samplers. | accepted |
| 2026-09-24 | Use a fused unconstrained value/gradient contract. | core protocol increment 4; target-bound HMC consumer increment 5 |
| 2026-09-24 | Put `std::autodiff`/Enzyme behind a backend boundary. | proposed; validate M2 |
| 2026-09-24 | Implement HMC, adaptation, and NUTS before ChEES/RMHMC. | accepted |
| 2026-09-24 | Use deterministic CPU parallelism; reserve Tokio for I/O/control. | proposed; finalize M6 |
| 2026-09-24 | Use initialized `Vec<T>` backing as an immediate safety repair. | superseded by explicit alignment requirement, increment 2 |
| 2026-09-24 | Preserve broken DDM sketches uncompiled; export only repaired latent adapters. | implemented, increment 1 |
| 2026-09-24 | Refresh HMC fused caches per transition until mutable state is encapsulated. | retained for legacy Hmc; eliminated in target-bound HmcChain, increment 5 |
| 2026-09-24 | Guarantee 64-byte buffer base alignment via a safe AVec wrapper; isolate storage for Miri and benchmark before claiming speedups. | implemented, increment 2 |
| 2026-09-24 | Use matching lock-pinned nightly components for development/Miri; separate stable and experimental Enzyme shells. | implemented, increment 2 |
| 2026-09-25 | Reject invalid metric input with typed errors; require explicit zero upper triangle and retain small positive masses without implicit regularization. | implemented, increment 3 |
| 2026-09-25 | Match cblas i32 bindings to LP64 OpenBLAS, verify header integer width at build time, and scope native linking/runtime lookup to opt-in backends. | implemented, increment 3 |
| 2026-09-25 | Bind cached points to one semantically pure target; evaluate into reusable scratch and commit only complete finite results. | implemented, increment 4 |
| 2026-09-25 | Poison gradient scratch with NaNs to detect partial writes; retain the linear passes until end-to-end profiling justifies another checked protocol. | implemented, increment 4 |
| 2026-09-25 | Add a target-bound HMC chain alongside the mutable legacy kernel; commit a complete point only on acceptance and propagate model errors. | implemented, increment 5 |
| 2026-09-25 | Use a symmetric absolute endpoint energy-error cutoff for fixed-length HMC; keep it distinct from future dynamic-trajectory stopping rules. | implemented, increment 5 |
| 2026-09-25 | Separate deterministic scalar-oracle agreement, fixed-seed analytic moments, and version-pinned external reference validation; do not treat any one as a substitute for the others. | first two implemented, increment 6; external reference gate open |
| 2026-09-27 | Keep phase evolution internal and separate from trajectory selection; consume the borrowed phase token on a failed step so partial scratch cannot be committed. Preserve the existing buffer budget. | implemented, increment 7 |
| 2026-09-27 | Check in executed CPU-float64 BlackJAX endpoint references and a separately pinned regeneration environment; keep reference tools outside Rust runtime/build dependencies. Endpoint agreement does not imply statistical agreement. | implemented, increment 8 |
| 2026-09-27 | Enforce the supported feature matrix and unsafe-documentation policy; gate unsupported algorithms and fail explicitly for unimplemented autodiff. | implemented, M0 completion, increment 9 |
| 2026-09-27 | Publish consuming phase evolution plus coherent reusable endpoint snapshots; leave tree selection to M5 and preserve HmcChain's eight-buffer budget. | implemented, M1 completion, increment 9 |
| 2026-09-27 | Initialize foreign outputs before Rust references; catch callback panics and resume after C; require unsafe for raw callback contracts; repair C cleanup using guarded build-time patches. | implemented and audited, increment 9 |

## Immediate next actions

1. Begin M2 with parameter layout and scalar transforms; define dimension accounting,
   constrained/unconstrained domains and fused Jacobian accumulation. Add analytic
   round-trip/Jacobian tests before Enzyme integration.
2. Build and execute the separately pinned Enzyme compiler experiment, then implement
   the `std::autodiff` target adapter with analytic/finite-difference cross-checks.
   Do not remove the unsupported-feature diagnostic until that gate passes.
3. Extend the now-executed BlackJAX endpoint references to logistic/funnel and
   rotated ill-conditioned targets. Add external sampling comparisons with
   documented uncertainty, following the
   [reference gate checklist](docs/hmc-validation.md#remaining-reference-gate-and-implementation-order).
   Endpoint agreement and fixed-seed moments do not close the full statistical gate.
4. Use the transition allocation tests and Criterion baseline to profile checked
   evaluation/copy overhead; add expensive-model and larger dense-metric cases
   before changing safety checks or claiming SIMD/backend speedups.
5. Do not begin NUTS, ChEES-HMC, or RMHMC before preceding exit gates pass.
