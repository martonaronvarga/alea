# HMC numerical and sampling validation

Historical increment-6 record. Its remaining M3 reference gates are now closed;
see [M3 completion, uncertainty bounds and performance evidence](m3-hmc-completion.md).

Roadmap increment 6 (2026-09-25) expands the gates for the existing target-bound
`HmcChain`; it does not add a sampler or change its production algorithm. Tests
live in `crates/runtime/tests/hmc_validation.rs` and
`crates/runtime/tests/hmc_chain_allocations.rs`. They add no dependencies or
production-path allocations.

## Analytic targets

All densities omit additive constants. Write `z = whiten(q)` and
`logp(q) = -(z[0]^2 + z[1]^2)/2`:

| Target | Whitened coordinates | Known distribution |
|---|---|---|
| Correlated Gaussian | `(q0, (q1 - 0.6 q0)/0.8)` | covariance `[[1, 0.6], [0.6, 1]]` |
| Banana | `(q0, q1 - 0.4(q0² - 1))` | nonlinear image of independent standard normals; unit Jacobian determinant |
| Scaled Gaussian | `(q0/0.001, q1/10)` | covariance `diag(1e-6, 100)`, condition number `1e8` |

The analytic gradients are checked against central differences at four specified
points per target, including the origin and tails. Differences use step size
`1e-5` in each coordinate's natural scale; tolerance is
`1e-8 * (1 + abs(reference))`. This validates these test fixtures, not a general
gradient diagnostic or an Enzyme implementation.

## Independent scalar transition oracle

The reference uses fixed-size arrays, an explicit two-by-two inverse mass matrix,
and its own kick/drift loop. It does not call the production metric actions or
integrator. Target formulae are shared with the finite-difference-checked fixtures.
It replays the current Marsaglia-polar momentum/RNG convention independently so
complete transitions can be compared through the public chain API.

Coverage: correlated Gaussian and banana targets with identity, diagonal, and
nontrivial dense mass; the scaled target with matched diagonal mass; and a
deliberately unstable but finite Gaussian configuration exercising MH rejection.
Each of eight cases runs four consecutive transitions at each of 1, 5, and 11
leapfrog steps: 96 full transitions. The energy-error cutoff is set to `f64::MAX`
in this oracle test to isolate finite Metropolis behavior.

Checks include both Hamiltonians, energy error, acceptance probability/decision,
accepted position/density/gradient, rejected-cache preservation, and identical RNG
consumption. Numerical tolerance is `1e-11 * (1 + abs(reference))`; rejected cache
fields and RNG state are compared exactly. This detects integration/mass/cache
regressions but shares the target and normal-generation conventions. It is **not
an independent Stan/BlackJAX execution** or a proof of the transition kernel.

## Stationary-moment regressions

Correlated Gaussian and banana targets each run with identity, diagonal, and
dense masses. The scaled Gaussian uses matched mass `diag(1e6, 0.01)`: mass is
precision here, so `M^-1 * target_precision = I`. This is explicit fixture tuning,
not automatically learned adaptation or evidence that untuned HMC handles all
ill-conditioned problems.

Each of the seven configurations runs seeds 19 and 6539, with 1,000 discarded
transitions followed by 12,000 retained states per seed. Step size is 0.15 and
trajectory length is nine steps. Rejected states are included. Tests require no
divergences, more than 80% acceptance, and these per-seed whitened moments:

| Observable | Expected | Absolute tolerance |
|---|---:|---:|
| `z0`, `z1` | 0 | 0.08 |
| `z0²`, `z1²` | 1 | 0.12 |
| `z0*z1` | 0 | 0.08 |
| `z0⁴`, `z1⁴` | 3 | 0.65 |

These are fixed-seed regression tolerances, **not confidence intervals**, ESS or
convergence diagnostics. Fourteen chains cover 182,000 transitions including
warmup. A separate unstable identity-mass run on the scaled target must diverge
and preserve all live cache fields. Real finite-input overflow cases test typed
momentum, coordinate, and density divergences without model fault injection.

## Allocation coverage

The existing two-dimensional tests remain. Added dimensions 7, 33, and 129
exercise SIMD remainders and larger dense operations. Each dimension runs eight
complete ten-step Gaussian trajectories for all three mass types. Dense factors
are lower bidiagonal with diagonal 1.5 and subdiagonal 0.2. Initialization is
outside the measured region; no warmup transition hides a first-step allocation.
An assertion checks the full leapfrog count so an early numerical exit cannot
masquerade as a complete-trajectory allocation measurement.

Measured Rust allocations and bytes are zero for the tested scalar, SIMD, Faer,
and OpenBLAS configurations. The counter does not intercept allocation inside
foreign BLAS libraries. These finite dimensions do not prove allocation behavior
for every backend size, arbitrary model, or arbitrary trait implementation.

## Reproduction

```sh
nix develop --command cargo test --manifest-path crates/Cargo.toml -p runtime --test hmc_validation --test hmc_chain_allocations --locked
nix develop --command cargo test --manifest-path crates/Cargo.toml -p runtime --test hmc_validation --test hmc_chain_allocations --release --locked
nix develop --command cargo test --manifest-path crates/Cargo.toml -p runtime --test hmc_validation --test hmc_chain_allocations --features simd --locked
nix develop --command cargo test --manifest-path crates/Cargo.toml -p runtime --test hmc_validation --test hmc_chain_allocations --features faer --locked
nix develop .#blas --command cargo test --manifest-path crates/Cargo.toml -p runtime --test hmc_validation --test hmc_chain_allocations --features simd,faer,openblas --locked
nix develop --command env MIRIFLAGS='-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check' cargo miri test --manifest-path crates/Cargo.toml -p runtime --test hmc_validation --locked
```

The combined build selects OpenBLAS for dense operations. Strict Miri runs four
deterministic tests; three long sampling tests are intentionally native-only.
The allocator-counter binary is also native-only. No new unsafe code is added.

Validation record (x86-64 Linux, pinned Nix toolchains): 117 workspace tests and
8 doctests pass. All seven new numerical/sampling tests and both allocation
tests pass in default debug/release builds, SIMD, Faer, and combined
SIMD/Faer/OpenBLAS builds. Strict Miri passes four deterministic tests (three
sampling tests intentionally ignored). Stable workspace/all-targets compilation,
formatting, and Clippy complete; existing unrelated Clippy warnings remain.

## Remaining reference gate and implementation order

1. Implemented in increment 7: the [internal phase/integrator boundary](hamiltonian-phase.md)
   preserves private caches, signed deterministic integration, positive public HMC
   settings, and the allocation gates. Persistent phase/tree storage is still future work.
2. Implemented in increment 8: [executed BlackJAX 1.5/JAX 0.9.2 fixtures](blackjax-reference.md)
   record CPU-float64 endpoints from explicit initial `(q, p)`, package versions,
   target parameters, step size/count, inverse mass, density/gradient, and energies.
   The optional content-pinned Nix generator reproduces the saved CSV locally.
3. Implemented in increment 8: private tests check all 36 endpoints through the
   production integrator, including dense factors, signed reversal, and explicit
   mass/inverse-mass conversion checks. Statistical agreement is not inferred.
4. Add logistic/funnel and rotated ill-conditioned cases, then external likelihood/
   gradient references and sampling comparisons with documented uncertainty.
   Only close M3 when its full exit gate passes; adaptation/NUTS remain later work.

CI enforcement, remaining FFI audits, and the separate std::autodiff/Enzyme gate
remain open. This increment neither changes those priorities nor claims a new
performance improvement; the previous Criterion baseline remains historical.
