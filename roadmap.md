# Alea: canonical engineering roadmap

Status: active, pre-alpha<br>
Last reviewed: 2026-10-16<br>
Last milestone update: 2026-10-25 — M17 reference candidate cycle complete; frozen candidate reviewed; ordinary modules and preludes, stable publication open<br>
Last architecture update: 2026-10-16 — maturity/adoption states and per-crate graduation gates<br>
Scope: architecture, correctness, numerical validation, performance, and implementation order

This is the canonical roadmap for Alea. It replaces the earlier project plan. Keep it
current as design decisions are accepted and milestones are completed.

## Settled scope and execution order — reviewed 2026-10-16

Alea is a **typed probabilistic modeling framework and composable inference library**.
Multiple distributions, priors/likelihood composition, hierarchical models and
Wiener/DDM are first-class. We defer a separate textual language and ecosystem-wide
coverage; we do not defer native modeling indefinitely. See the
[settled plan](docs/settled-plan.md) for exact architectural seams, distribution
tiers, continuous-adaptation mathematics, implementation slices and success gates.

The central planned transition composition is `Dynamics + Integrator + StepController
+ Trajectory + Correction`, with target/geometry capabilities below it and adaptation
and stopping outside it. `Hmc<T,M,I>` is the current canonical implementation, not
the final universal interface. WALNUTS belongs to step control/trajectory/correction;
isokinetic dynamics do not masquerade as a different leapfrog coefficient.

M4–M11 have now produced reference or research implementations. The next executable
sequence is M17 per-crate release graduation. M13 boundary enforcement, M14 modeling
verticals, M15 reference sessions and M16 unified experiments are complete.
M12 particle inference begins after shared B1 support/event/batch semantics settle.
Advanced sampler retuning is not the default work queue; each negative promotion
decision has an explicit re-entry gate.

Identity initialization is the default; the single-score rule is experimental.
Retained draws still use frozen adaptation. Diminishing adaptation and automatic
stopping require separate theory, containment and diagnostic gates.

## Updating this document

For every material change:

1. Update the affected task and exit gate.
2. Record accepted design decisions in the decision log.
3. Link implementation, benchmark, or validation evidence.
4. Update `Last reviewed` after a whole-roadmap review.

Task markers are `[ ]` planned, `[~]` in progress, `[x]` complete with evidence, and
`[!]` blocked with the blocker stated. A checked implementation task means its
declared work/evidence is complete; it does not imply promotion. An algorithm is
validated/promoted only when the applicable maturity gates below pass.

## Maturity and promotion model — 2026-10-16

Implementation, validation, API stability and default selection are separate
decisions. Alea now uses the S0 design, S1 research, S2 validated, S3 release-
candidate and S4 stable states defined in the
[maturity and promotion policy](docs/maturity-and-promotion.md). Adoption is a
separate label: reference, recommended, opt-in, approximate, or not exposed.

No Alea crate is S4 today. The strongest baseline components are S2: core target
and state contracts, transforms, Euclidean metric algebra, fixed HMC/RWMH,
covariance warmup, multinomial NUTS, independent-chain runtime and modern
diagnostics. They are validated within their documented scope, but the workspace
is still `publish = false` and pre-alpha. The earlier frozen candidate completed
the S3 workflow/platform/compatibility cycle. Current imports use ordinary modules
and convenience preludes; crate-wide stable scope, publication and support remain
open.

Research milestones may close with negative promotion decisions. Their checked
implementation tasks remain historical evidence; their promotion gates remain
open until a new predeclared protocol addresses the recorded failure. Exact-target
claims additionally require the implemented floating-point proposal—not only the
ideal equations—to satisfy its reversibility, volume/Jacobian and support
assumptions. Approximate methods remain typed and reported separately.

## Current priority: M17 reference release-candidate consolidation

M13 classifies the complete public surface and explicitly contains RMHMC as
`ExactnessPendingRmhmc`; its numerical-map promotion obligation remains open.
M14 delivers normalized scalar/vector/discrete distributions, typed model assembly,
three downstream inference/prediction workflows and bounded calibration gates.
M15 composes these components into validated sessions with partial typed outcomes,
bounded/durable output and frozen-phase replay. M16 adds sealed source-bound
experiments, calibration and reproducible reports with a negative promotion review.
Preserve covariance/NUTS as the S2 numerical reference; research samplers retain
their promotion obligations. The frozen reference candidate has a
separate S3 review; none of these milestones establishes S4.

The 2026-10-23 [native parity report](docs/benchmarks/walltime-parity-2026-10-23/README.md)
records approximately comparable walltime for the explicit fast NUTS profile.
The [completed candidate review](docs/releases/2026-10-25-reference-candidate/README.md)
records native platform CI, downstream calibration/recovery, compatibility and
signed source/package evidence. Next obtain external application feedback and
prepare per-crate stable compatibility/publication decisions. The
[development sequence and workspace policy](docs/workspace.md) describe the order,
candidate retention and archive restoration. No default or maturity
change follows automatically from benchmark performance.

## Completed baseline hardening and remaining performance assertions

This completed sequence hardened Fisher/low-rank numerics and produced the negative
M4F promotion decision. Preserve the reference HMC/covariance policy, historical
failures and all regression evidence. Any new statistical performance assertion
requires a new predeclared protocol.

- [x] Reproduce f64 cancellation in accepted low-rank corrections; reject unsupported
  spectral/basis conditioning with a typed error instead of constructing singular
  or negative inverse-mass actions. No silent spectral clipping.
- [x] Deterministic rank/spectrum grids check positive momentum energy and determinant
  conventions; every metric action checks shape before output mutation.
- [x] Check foreground/background windows against centered batch moments, including
  failed observations at reset/swap boundaries; verify bounded raw history.
- [x] Reject mismatched dimensional reinterpretation of stored Fisher observations.
- [x] Enable the MCMC Faer feature in CI, not merely the underlying math feature.
- [x] Independent mathematical Fisher fixtures: 120-digit Decimal dense 2x2
  Riccati solutions (including rank caps/filtering) and batch-recomputed diagonal
  update traces. Generator/provenance in `tools/fisher-reference/`; CI checks bytes.
  Matched scatter scale tests and an extreme diagonal standardization grid exposed
  avoidable intermediate overflow/underflow, fixed by common scatter normalization
  and log-space regularization. Truly unrepresentable scatters still return errors.
- [x] Known two-direction Fisher geometry embedded in 8, 32 and 128 dimensions:
  verify actions on fitted and orthogonal directions, determinant, rank caps,
  eigenvalue filtering and observation-order invariance against scalar formulas.
- [x] Execute pinned nuts-rs low-rank fit-prefix traces (18 cases, dimensions
  2/8/16). Adjudicate reference roundoff with an independent 80-digit dense oracle;
  two upstream fixtures miss the original accuracy threshold, Alea passes all 18.
  Tooling/provenance and deliberate policy differences are documented in
  `tools/fisher-reference/upstream/README.md`; no tolerance widening or policy change.
- [x] Bounded end-to-end Fisher controller validation (2026-10-02): identity and
  clipped-score initialization, centered batch moments of retained draws including
  rejections, 10/80-period reset/swap/ring boundaries, rank-1/rank-2 updates and final
  nonboundary refits. Verify frozen geometry, final dual-averaging restart, cache
  coherence, target-call/report counts, typed backend failures, and allocation-free
  diagonal warmup after construction. Leapfrog/OMF and budgets through 401 are covered.
- [x] Extend fitted-spectrum validation to 240 commuting-scatter cases (d=4/16,
  coordinate scales 1e-8/1/1e8, common data factors 1e-100/1/1e100, rank caps and
  filtering), checking actions, determinant and momentum energy against scalar
  formulas. Unsupported fitted conditioning returns its typed error.
- [x] Gaussian tail-start safety for both initializers through |q|=1e150; verify
  finite coherent caches, bounded calls and frozen geometry. Reject unrepresentable
  initial density and exhaust unresolved-stiffness search with typed errors.
  These are safety checks, **not** typical-set recovery or convergence claims.
- [x] Matched warmup/fitting cost baseline (2026-10-02): shared tested Gaussian
  workloads, 36 actual-call records, 12 complete-warmup and six cold-fit timings.
  [Protocol/results](docs/benchmarks/warmup-cost.md) preserve divergences,
  budgets, uncertainty and provenance. Low-rank fitting overhead is material;
  profile it before optimizing. No ESS or production-default claim.
- [x] Profile-guided fitting optimization (2026-10-02): perf, Hyperfine,
  Callgrind/Massif and Criterion isolate nested worker synchronization and repeated
  eigensolve/power work. Explicit chain-local sequential kernels and shared
  eigensystems improve the paired d64 workloads without weakening numerical checks.
  [Evidence and large-fit limitations](docs/benchmarks/fisher-profiling.md);
  no clear d256/history160 wall-time gain and no memory/ESS improvement claim.
- [x] Bounded non-Gaussian retained-draw regression: covariance/Fisher diagonal
  and rank-capped warmup on banana, logistic and width-one funnel, with fixed
  seeds/budgets, analytic/pinned BlackJAX moments, cache/freeze checks and actual
  evaluation/divergence reporting. [Protocol and limits](docs/benchmarks/fisher-retained-draws.md).
  Final fitted ranks are zero in this corpus; it does not validate nonzero-rank
  non-Gaussian production geometry or establish efficiency/robustness superiority.
- [x] Nonzero-rank non-Gaussian extension: unit-determinant sheared banana with
  independently checked derivative pullback and analytic/reference observables.
  Same seeds, budgets and tolerances; each rank-capped chain must freeze at
  nonzero rank (observed rank two for all four). Cache/freeze/call checks pass.
  Seventeen retained divergences versus zero for both diagonal baselines prevent
  a robustness-superiority claim; rank two in d=2 is not a scalability result.
- [x] Diagnose the sheared-banana Fisher divergences by same-state/momentum/metric,
  fixed-time step refinement. All 17 default failures recover at half steps.
  Add validated opt-in `FisherOptions::with_target` with unchanged default 0.8;
  target 0.95 eliminates observed banana/sheared-banana retained divergences,
  but leaves 2/3 diagonal/rank-capped funnel divergences. Preserve the negative
  result and original gates in [the investigation](docs/benchmarks/fisher-divergences.md).
- [x] Bounded noncommuting fit hardening (2026-10-05): 324 two-direction and
  540 four-direction planted `G F G = C` cases cover rank/filter budgets,
  extreme common/coordinate scales, complementary actions and row reversal.
  Both grids exposed nested-square-root accuracy loss; a factor/SVD evaluation
  passes the original tolerances without clipping or changing the estimator.
  [Construction, repair and limits](docs/fisher-noncommuting.md). General
  ill-conditioning and promotion remain open; prior fit timings are historical.
- [x] Literature/reference-guided numerical scaling repair (2026-10-05): pin
  Stan, walnutpie, nuts-rs/nutpie, BlackJAX and JAX source comparisons; normalize
  centered standardized data and ridge together before squared norms/Gram
  products. Preserve the Fisher equation and explicit failure policy. Fourteen
  scalar-oracle edge cases and 1,440 noncommuting fits retain original tolerances.
  [Mathematics, references and limitations](docs/fisher-numerical-review.md).
- [x] Remeasure repaired Fisher costs (2026-10-05): five fixed fit/warmup
  workloads, eight serial observations each, per-process peak RSS, source/binary
  fingerprints and separate actual-call records. [Results and reproduction](docs/benchmarks/fisher-cost-2026-10-05.md).
  This is a current systems baseline, not a paired repair-cost comparison or
  retained-draw efficiency result. Measurement artifact contracts run in CI.
- [x] Funnel replay extension (2026-10-05): original plus disjoint four-seed
  ensembles, acceptance .8/.95 and Fisher rank caps 0/2. All 69 observed energy
  divergences recover at fixed-time half/quarter/eighth steps; factor-one replay
  reproduces each failure, and live caches/RNG/counters remain unchanged.
  [Curvature checks, errors and limitations](docs/benchmarks/funnel-replay.md).
  Half-step errors can still be large; failing frozen metrics all have rank zero.
  This is diagnostic evidence, not a retry algorithm or efficiency result.
- [x] Predeclared efficiency/robustness evidence and an M4F promotion review, including
  non-Gaussian geometry and broader noncommuting ill-conditioning. Start with
  retained-draw quality, memory and total cost beyond the warmup baseline;
  do not infer a production default
  or add Adam from passing bounded Gaussian tests. Fit-prefix/controller fixtures
  do not establish nuts-rs/nutpie whole-warmup equivalence (policies differ).
- [x] HVP/batch backend-failure and panic-isolation audit (2026-10-01): preserve
  preflight outputs, retain typed errors, validate independent batch lanes, clear
  stale results on reuse, and explicitly require discarding outputs after unwinding.
  Backend panics propagate; the wrappers do not promise per-lane panic isolation.
- [x] Complete-macrostep target-failure/property tests beyond the happy-path
  oracles: `crates/alea-mcmc/tests/integrator_failures.rs` injects backend errors,
  panics, missing/non-finite gradients and non-finite densities at every fused
  evaluation of three-step leapfrog/OMF/BCSS trajectories. Checks typed outcomes,
  bit-exact preservation of the live cache, restart/recovery, evaluation counts,
  allocation-free returned failures, and 128 seeded two-stage reversal cases.
  Covered by the Miri script; not an exhaustive arbitrary-metric failure proof.
- [ ] Revisit statistical performance assertions with a predeclared benchmark protocol;
  do not retune seeds or tolerances to make a backend pass.

Evidence and numerical limits: [baseline hardening](docs/baseline-hardening.md).

## Implementation progress: baseline hardening after increment 13

This increment adds no sampler algorithm, distribution or generic engine. It fixes
two validation failures, adds deterministic algebra/rollback coverage and closes a
feature-matrix coverage gap. Fisher remains experimental; the new floating-point
conditioning policy is conservative, not a proof of accuracy for arbitrary inputs.
Follow-up on 2026-10-01: five additional optional-capability failure tests and an
explicit panic/output-validity contract; no new backend or sampler algorithm.

## Implementation progress: 2026-09-30, increment 13 — reference warmup and Fisher foundations

[M4 evidence](docs/m4-warmup.md) records covariance warmup, executed BlackJAX
fixtures, sampling comparisons, allocation checks and benchmarks. M4 remains the
reference policy; it is not Alea's proposed final default.

Implemented optional potential-HVP and explicit-layout batch target capabilities;
spectral low-rank metrics; paired foreground/background Fisher moments; optional
`faer` reduced Fisher fitting; experimental consuming Fisher warmup; and statically
dispatched leapfrog/OMF2/BCSS2 complete macrosteps in fixed HMC and both warmups.
Existing implementations/tests are preserved; the historical migration audit
remains in documentation and Git history. Public
step counters now say `integration_steps`, not `leapfrog_steps`.

[The research/design record](docs/fisher-and-trajectories.md) specifies signs,
regularization, cold-path complexity, the Fisher derivation, BCH/oscillator
analysis, future joint adaptation and benchmark priorities. Fisher is **not yet
promotion-ready**: independent nuts-rs adaptation traces, pathological-target
efficiency comparisons and rank-budget sensitivity remain gates. No NUTS,
WALNUTS, adaptive-coefficient controller, exact-Gaussian split or accelerator
runtime is claimed implemented.

CPU independent chains come first. Batch layout/capability contracts are present
now; FSM/accelerator execution is deferred. The versioned compiled-target ABI is
a [design specification](docs/compiled-target-abi.md), not an implemented backend.

## Implementation progress: 2026-09-29, increment 12 — target architecture (historical)

The [architecture record](docs/architecture.md) is the current API/dependency guide.
At the user's direction, compatibility is not a requirement: the project is not in
use, so old paths and contracts are replaced rather than retained or deprecated.

- Implemented the nine `alea-*` workspace crates with explicit production edges.
  Aligned storage belongs to math; opaque Wiener derivatives belong to distributions;
  generic model composition remains in core. Core/math/MCMC have no cubature edge.
- Replaced split density adapters and duplicate HMC/state/kernel APIs with direct
  targets, coherent caches, one `Hmc`, and fallible `EuclideanMetric` operations.
- Rewrote RWMH against a derivative-free, fallible target capability and typed
  transitions. Restored explicit dual-averaging warmup, retained spare-normal
  generation, and kept stationary transitions allocation-free.
- Added streaming and aligned draw collection; retained classical diagnostic
  calculations under accurate names with invalid-input errors. Removed mutable
  compatibility APIs and placeholder sampler exports, not working functionality.
- Preserved dormant DDM/C++/inference sources separately as research. The
  [migration audit](docs/migration-audit.md) maps 118 old paths and 188 test
  definitions (including uncommitted M3 work). Its completed acceptance inventory
  was retired after current-surface architecture, maturity, API and exactness
  gates superseded it.
- Added a 31-group feature inventory with implementation status, behavioral
  obligations and test evidence for migration acceptance. Ported grid/spline
  utilities from research into math and added latent-adapter regressions. The
  migration-only mutation harness is now retained through Git history.
- Migrated tests, M3 fixtures, benchmarks, Enzyme tooling and the cubature submodule.
  Updated Nix binaries and CI, including a production dependency-boundary gate.

This is architecture implementation, not completion of M4–M9. Windowed warmup,
modern diagnostics, parallel chains, NUTS, ChEES-HMC and RMHMC retain their gates.
Older progress sections and reports below are historical, including compatibility
decisions now explicitly superseded by this breaking migration.

## Implementation progress: 2026-09-28, increment 11 — M3 complete (historical)

The [M3 completion record](docs/m3-hmc-completion.md) documents executed reference
provenance, statistical tolerances, allocation scope, benchmarks and reproduction.
The existing fixed-length HMC transition kernel and its public API are unchanged.

- Extended independently executed BlackJAX/JAX references to logistic, width-one
  centered funnel and rotated condition-1e4 Gaussian targets: 90 endpoints and
  signed reversals across identity/diagonal/dense masses. Retained earlier
  correlated/banana and diagonal condition-1e8 coverage.
- Added second-order energy-error convergence checks for all 15 target/metric
  pairs. Added numerical stress regressions for repeated rejection, bit-exact
  cache preservation and recovery.
- Compared actual BlackJAX HMC sampling with separate Rust RNG streams across
  13 configurations, four chains per implementation/configuration, 8,192 retained
  draws each. All 78 observable comparisons satisfy documented batch-means MCSE
  bounds; known whitened moments independently check both implementations.
- Extended zero-Rust-allocation per-leapfrog checks to non-Gaussian/stiff models.
  User callbacks and foreign BLAS malloc are explicitly outside the guarantee.
- Added steady-state Criterion workloads for all metric types, larger logistic
  regression and transformed Wiener evaluation, retaining checked/legacy baselines.

All four supported feature-matrix check/test/Clippy/rustdoc gates pass. Both new
BlackJAX artifacts regenerate byte-for-byte in the pinned environment. Full strict
Miri and benchmark execution are recorded in the completion record. Statistical
tests use conservative approximate MCSE, not an ESS/R-hat or mixing certificate.
No Stan sampling execution, cross-platform run or remote CI execution is claimed.
M4 adaptation is next; NUTS, ChEES-HMC and RMHMC remain unimplemented.

## Implementation progress: 2026-09-28, increment 10 — M2 complete (historical)

The [M2 model-boundary record](docs/m2-model-boundary.md) documents the public
contracts, coordinate measures, toolchain, reproducible checks and limitations.

- Implemented all scalar/vector/Cholesky/full-matrix transforms, immutable flat
  parameter layout, inverses, log-Jacobians and analytic reverse pullbacks.
- Added a fused, fallible constrained-model adapter with reusable aligned scratch,
  complete-gradient validation, no evaluation allocations in tested models, and
  atomic publication of gradients on error/panic paths.
- Executed `std::autodiff`/Enzyme derivatives with the separately content-pinned
  upstream nightly-2026-06-23 distribution and matching plugin. Default/stable
  toolchains are unchanged; no source compiler build is needed.
- Added explicit custom-derivative primitives and allocation-free model sums.
  Wiener uses an analytic opaque primitive composed with an Enzyme prior; this
  does not register arbitrary nested foreign calls inside differentiated IR.
- Added generic finite-difference diagnostics and the `gradient_check` command,
  proptest and numerical-Jacobian/pullback properties, HMC exponential/Beta moment
  integration, allocation tests, Miri coverage and an executed-Enzyme CI job.
- Added Criterion coverage for the fused model boundary. Initial timings overlapped
  compilation and are not used for speedup claims; Cholesky forms remain preferable
  when a likelihood can avoid materializing/validating a full matrix.

Enzyme tests compare Gaussian, correlated Gaussian, logistic, banana and funnel
derivatives with analytic references, plus finite differences of transformed targets.
Wiener hybrid derivatives are checked against finite differences and reuse the
existing externally checked series implementation. Coordinate conventions are
explicit: centered stick-breaking simplex, stereographic unit sphere, row-major
matrices, rejection of floating-point boundary collapse.

Validation: stable and SIMD workspace format/check/test/Clippy/docs gates pass;
the release Enzyme gate passes 15 focused tests (4 AD, 5 models, 6 transforms),
Clippy and all six CLI diagnostics. Strict Miri passes the expanded safety suite,
including 9 new transform/model tests with native-only allocation/property tests
excluded. The HMC integration and zero-allocation gates pass. Nix flake evaluation
and Nix/Rust formatting pass. No remote CI, cross-platform execution, or new
Faer/OpenBLAS run is claimed for M2.

## Implementation progress: 2026-09-27, increment 9 — M0 and M1 complete (historical)

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
- Wiener4/5 branch buckets use initialized aligned scratch with contiguous bucket
  offsets. Caller-owned workspaces and cached models avoid repeated allocation.
  [Batch regressions](crates/alea-distributions/tests/wiener_batches.rs) cover
  equivalence, invalid observations and allocation-free reuse.
  [Wiener notes](docs/wiener-research-notes.md) record tail fixes and measured costs.
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

Native typed models/distributions and external models lower to the same checked
target capabilities. Composable dynamics/geometry and corrected trajectory engines
consume those capabilities. Adaptation operates outside retained transitions;
runtime monitoring observes immutable summaries. The near-term dynamic sequence is
Fisher → reference NUTS → validated CPU diagnostics/runtime → WALNUTS. Modeling and
benchmark tracks proceed alongside this sequence; ChEES/isokinetic and manifold
methods follow their independent validity gates.

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

The implemented crate organization is:

```text
alea-core          target/model protocols, transforms, state, errors
alea-math          aligned storage, metrics, linear algebra, numerical primitives
alea-autodiff      std::autodiff/Enzyme adapters and validation
alea-distributions distributions, including Wiener/DDM
alea-mcmc          HMC, NUTS, ChEES-HMC, RMHMC
alea-smc           particle contracts; filters and particle MCMC planned
alea-ffi           audited foreign boundaries only
alea-runtime       streaming output; parallel chains planned
alea-cli           binaries, configuration, user-facing context
```

Implemented as a breaking migration in increment 12. No backward compatibility,
legacy crate aliases or duplicated sampler protocols are retained. Numerical core
crates do not depend on the broad FFI crate; the architecture CI gate checks this.
See [current boundaries and interfaces](docs/architecture.md).

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
    fn sample_momentum(&self, z: &[f64], momentum: &mut [f64]) -> Result<(), MetricError>;
    fn velocity(&self, momentum: &[f64], velocity: &mut [f64]) -> Result<(), MetricError>;
    fn kinetic_energy(&self, momentum: &[f64], scratch: &mut [f64]) -> Result<f64, MetricError>;
    fn log_det(&self) -> f64;
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

### M2 — transforms and Enzyme backend (complete)

Goal: a Stan-like unconstrained model boundary.

- [x] Scalar identity, lower, upper, and interval transforms.
- [x] Positive, ordered, positive-ordered, simplex, and unit-vector transforms.
- [x] Cholesky covariance/correlation transforms.
- [x] Covariance/correlation matrix transforms after Cholesky validation.
- [x] Flat parameter layout and dimension accounting.
- [x] Fuse constrain + Jacobian + model evaluation (analytic transform pullback,
  compiled AD of the constrained density, one fused model call).
- [x] Analytic and Enzyme target adapters.
- [x] Custom derivatives for opaque primitives through an explicit value/gradient
  boundary; native Enzyme registry/nested opaque calls are not part of this API.
- [x] Finite-difference gradient diagnostic command.

Exit gate: transform round trips/Jacobians pass property tests; gradients agree on
Gaussian, correlated Gaussian, logistic, banana, funnel, and Wiener targets.

Exit gate passed locally on 2026-09-28: [M2 record](docs/m2-model-boundary.md).

### M3 — correct Euclidean HMC (implemented; S2 component)

Goal: small, correct, allocation-free fixed-length HMC.

- [x] Identity, diagonal, and dense metrics: validated constructors, dimension-safe
  operations, external inverse-mass convention checks and measured workloads.
- [x] Correct metric-aware momentum generation (diagonal/dense regression tests).
- [x] Reversible velocity Verlet/leapfrog: 90 BlackJAX endpoints/reversals and
  second-order energy-error convergence for 15 target/metric pairs.
- [x] Fixed-length HMC with Metropolis correction: independent scalar oracle,
  all-metric Gaussian/banana moments, matched-mass condition-1e8 Gaussian and
  actual BlackJAX sampling comparisons with documented Monte Carlo uncertainty.
- [x] Divergence detection from non-finite evaluation/excess energy error:
  typed reasons, symmetric absolute endpoint threshold, true overflow tests and
  unstable logistic/funnel/rotated regressions preserve the live cache bit-exactly.
- [x] Reusable integrator/proposal storage: first-transition allocation checks
  through dimension 129 plus non-Gaussian per-leapfrog regressions, all metrics.
  Scope is library-owned Rust hot paths and tested targets; user callbacks and
  foreign allocator behavior require their own instrumentation and cannot be
  guaranteed by the Rust allocation counter.
- [x] Typed fixed-length HMC transition statistics (`HmcTransition`, increment 5).

Exit gate: forward/reverse recovery, expected energy-error convergence, correct Gaussian
moments/covariance for all metrics, zero allocations per leapfrog step, and agreement with
Stan/BlackJAX reference targets.

Exit gate passed locally on 2026-09-28 using BlackJAX:
[M3 completion record](docs/m3-hmc-completion.md). No external speed ranking or
universal mixing guarantee is implied. Modern diagnostics remain in M6.

### M4 — step-size and metric adaptation (implemented; S2 reference)

Goal: validated covariance-only reference warmup (implemented).

- [x] Reasonable initial step-size search, bounded failures and unchanged live cache.
- [x] Dual-averaging step-size adaptation.
- [x] Transactional Welford online variance/covariance.
- [x] Regularized diagonal metric adaptation.
- [x] Opt-in dense metric adaptation.
- [x] Stan-style initial fast, expanding slow, and final fast windows.
- [x] Correct reinitialization after metric changes.
- [x] Consuming warmup freezes adaptation before retained draws.

Exit gate: improved ESS per gradient on correlated targets without bias; schedules match
Stan/BlackJAX; fixed seeds yield deterministic adaptation.
Evidence: [M4 completion record](docs/m4-warmup.md). Native exact replay and Miri
deterministic-float replay are separate from ordinary perturbed-float Miri safety tests.

### M4F — Fisher geometry (reviewed; experimental, not promoted)

- [x] Optional potential-HVP and batch capabilities, typed failures and explicit layout.
- [x] Allocation-free spectral low-rank inverse-mass actions, including eigenvalues below one.
- [x] Paired cached state/score observations, transactional moments, bounded foreground/background history.
- [x] Separate identity/default and experimental clipped-score initialization;
  regularized diagonal Fisher estimates.
- [x] Opt-in reduced SPD Fisher solve with explicit raw-scatter regularization and rank cap.
- [x] Consuming fixed-HMC Fisher warmup with final geometry freeze; no post-warmup controller.
- [x] Composable leapfrog/OMF2/BCSS2 in fixed HMC and either warmup policy.
- [x] Executed, commit-pinned nuts-rs fit/update traces (not merely matching the paper);
  collector-policy and whole-warmup equivalence explicitly excluded.
- [x] Independent low-rank fixtures spanning degeneracy, extreme scales and rank truncation.
- [x] ESS/gradient and total time including warmup on Gaussian, logistic, banana and funnel targets.
- [x] Benchmarked rank/window budgets, peak memory and cold-factorization costs.
- [x] Promotion reviewed after the gates: rejected; covariance warmup remains the reference.
- [x] Weighted continuous moments and a validated diminishing schedule, initially warmup-only.
- [x] Typed acceptance/divergence/failure observations; decaying Adam and Robbins–Monro
  compared with dual averaging under a common log-step objective.
- [x] Extract canonical dynamics, fixed step control and canonical Metropolis from
  the current HMC without duplicating its working implementation or tests.

October 14 final review: the predeclared eight-method/four-target experiment covers
two independent ensembles and all 320 pooled moment gates, with batch-means
ESS/call and ESS/time including warmup. Independent modern diagnostics pass
319/320 gates; the retained Gaussian R-hat failure is 1.011216 > 1.01. A fixed
d64/d256 rank/history matrix now records cold fit time, history storage and GNU
time peak RSS. The review rejects promotion: the diagnostic gate failed,
covariance is usually more efficient, and RM/Adam increase banana divergences.
Keep Fisher experimental and retain covariance/dual averaging defaults. Evidence:
[M4F final promotion review](docs/benchmarks/m4f-gates-2026-10-14.md).

Full reduced fitting can cost O(d k^2 + k^3), k <= min(d, 2n). An output rank cap
does not bound intermediate decomposition size. Bound n as well; introduce a
sketch/randomized solver only with its own numerical and approximation gates.

### M5 — multinomial NUTS (implemented; S2 recommended kernel)

Goal: classical multinomial NUTS as a correctness oracle, not the endpoint.

- [x] Iterative, allocation-bounded trajectory construction.
- [x] Random forward/backward doubling.
- [x] Multinomial proposal selection in log space.
- [x] Metric-aware generalized U-turn criterion.
- [x] Divergence, U-turn, and maximum-depth termination.
- [x] Full tree/energy/acceptance statistics.
- [x] Frozen-chain integration with M4 reference and opt-in experimental M4F;
  this does not promote Fisher or implement NUTS-specific warmup adaptation.
- [x] One complete integrator macrostep per tree edge; never select internal stages.

Exit gate: moments and diagnostics agree with Stan, BlackJAX, and nuts-rs; no allocation
proportional to tree size; edge cases have reference tests.

October 9 completion: `Nuts<T,M,I>` reuses the coherent phase engine, with binary
carry tree construction and enhanced generalized turning checks. All 80 modern
diagnostic and 120 moment gates pass against executed pinned Stan, BlackJAX and
nuts-rs runs on Gaussian/logistic/banana/width-one funnel targets. A 768-case
recursive scalar oracle, stage-failure/panic tests and allocation measurements
cover tree mechanics; no transition allocation through depth 10. Reference
warmup outputs convert without target evaluation or RNG consumption. Covariance
defaults and M4F's experimental status remain unchanged. See `docs/nuts.md` and
`tools/nuts-reference/README.md` for conventions, reproduction and limitations.

### M6 — multi-chain runtime and diagnostics (implemented; S2 reference)

Goal: deterministic parallel inference and trustworthy diagnostics.

- [x] Reproducible independent per-chain RNG streams.
- [x] Sequential and Rayon-parallel runners with stable ordering.
- [x] Streaming draws and bounded-memory summaries.
- [x] Rank-normalized split and folded R-hat.
- [x] Cross-chain bulk/tail ESS and mean MCSE.
- [x] E-BFMI, divergences, and tree-depth saturation.
- [x] Stable draw/warmup/transition output schema.
- [x] Cancellation/progress outside numerical hot loops.
- [x] Latest-only nonblocking summaries and minimum-draw, opt-in stopping monitors;
  diagnostic fixtures include deliberately stuck, shifted and ragged chains.

Exit gate: sequential/parallel per-chain equality for equal seeds; diagnostics agree with
Stan/ArviZ fixtures; streaming memory is independent of retained draw count.

October14 completion: ChaCha12 logical streams, caller-owned Rayon pools, typed
versioned streaming records, fixed-capacity diagnostic windows and a separate
latest-only worker with Tokio subscriptions/cancellation. Fourteen independently
executed ArviZ0.23.4 fixture cases validate modern estimates and deliberate
failures. Real NUTS draw/statistic replay, multiple pool sizes, bounded live
allocation, shutdown races and cancellation/failure tests pass. Diagnostics are
explicitly window-local; ragged/nonfinite/undefined inputs cannot pass stopping.
Fixed-budget equality excludes scheduling-dependent stop/cancel lengths. Adaptive
warmup stays explicit; Fisher remains experimental. See [M6 runtime](docs/m6-runtime.md).

### M7 — WALNUTS (implemented; not promoted)

Goal: candidate production dynamic kernel after reference NUTS and CPU diagnostics.

- [x] Independently execute the JMLR reference and pin walnutpie/nuts-rs versions.
- [x] Dyadic microstep schedule, fixed simulated-time macroblocks and bounded work.
- [x] Implement the paper's forward/reverse construction and selection probabilities.
- [x] Test detailed-balance/reversibility edge cases, failed local refinement and overflow.
- [x] Reuse Euclidean geometry/target storage; do not pretend local adaptation is a
  drop-in reversible fixed-step integrator.
- [x] Benchmark funnels/stochastic volatility and total warmup cost against Fisher NUTS.

Exit: reference traces, calibrated moments, modern diagnostics and reproducible
ESS/sec improvement on the declared workload suite. No default change from a paper alone.

October 14 review: the implementation uses the paper's full within-macroblock
Hamiltonian range and deterministic forward/reverse support correction, with
bounded iterative `O(dimension * depth)` storage. The pinned Walnutpie trace,
independent scalar map/range tests, reverse-support/rollback cases, four-target
moments and all modern diagnostics pass. On the declared benchmark, minimum
ESS/sec is about 2% higher on the width-one funnel but 45% lower on the
50-dimensional stochastic-volatility target, with 5.6–5.8x as many target calls.
The performance exit gate therefore fails: WALNUTS remains explicit and opt-in;
no default changes. See [WALNUTS implementation and review](docs/walnuts.md).

October 15 completion: randomized two-point selection now implements the published
2/3 critical and 1/3 next-dyadic distribution, cumulative forward/reverse log-
probability weights in both orbit directions, bounded successor work and optional
orbit-level macro jitter. Reciprocal asymmetric-edge tests, allocation checks and
calibrated moments pass. The separately declared R2P run passes diagnostics and
reduces support stops, but is about 53% below Fisher NUTS minimum ESS/second on
stochastic volatility while using 7.2x the calls. The negative promotion decision
is unchanged. See [R2P completion evidence](docs/benchmarks/m7-r2p-protocol.md).

### M8 — ChEES-HMC, GHMC and multiscale alternatives (implemented; opt-in)

Goal: ensemble-adapted static HMC as a NUTS alternative.

- [x] Shared chain-ensemble adaptation state.
- [x] ChEES objective and trajectory-length adaptation.
- [x] Joint warmup of step size and trajectory length.
- [x] Randomized integration length.
- [x] Optional difficult-direction criteria, benchmarked separately.
- [x] Freeze adaptation before retained sampling.
- [x] Persistent-momentum GHMC with explicit momentum state and rejection convention.
- [x] DR-G-HMC acceptance ratios and staged-rejection reference tests.
- [x] ATLAS curvature/local-step experiments with bounded derivative work.
- [x] Separate isokinetic dynamics and MAMS correction, with an explicit approximation
  marker for unadjusted MCLMC; no canonical-energy acceptance shortcut.

Exit gate: reproduce published behavior; compare ESS/gradient and ESS/second against NUTS
and fixed HMC; ensemble scheduling remains deterministic.

October 15 completion: Algorithm 1 ChEES centering/path gradients, harmonic-mean
step adaptation, shared Halton scheduling, randomized frozen trajectories and
bitwise sequential/Rayon replay pass. Persistent GHMC, bounded two-stage DR-G-HMC,
bounded `H^2` ATLAS probes, exact isokinetic MAMS correction and explicitly biased
MCLMC are separate typed paths. On the declared four-target release run, ChEES had
3.1–4.1x the minimum ESS/gradient of NUTS; all maximum modern R-hats were below
1.004. The difficult-direction arm was mixed. Millisecond analytic-target timing
is not promoted as a broad speed claim, and every M8 kernel remains opt-in. See
[M8 implementation and evidence](docs/m8-alternatives.md).

### M9 — learned and structured geometry (implemented; not promoted)

- [x] Frozen invertible transport target with log-Jacobian/gradient tests.
- [x] Warmup-only learning; prove retained draws never train the transform.
- [x] Low-rank linear correction in transported coordinates.
- [x] Block/sparse metric implementations without requiring a dense trait representation.
- [x] Optional multi-path Pathfinder initialization; compare its full cost with score initialization.

Exit: invertibility/measure tests, held-out sampling diagnostics and end-to-end
benefit over Fisher + WALNUTS. Learned transforms are not necessary dependencies.

October 15 completion: immutable low-rank affine maps have exact inverse,
log-determinant and score-pullback tests; consuming bounded warmup learning leaves
no training interface in retained targets. Block-diagonal and sparse-Cholesky
metrics implement the existing constant Euclidean trait without dense global
storage. The bounded multi-path initializer uses L-BFGS local normals, Monte Carlo
ELBO selection and explicitly raw (not Pareto-smoothed) mixture importance
resampling. Exact transformed-Gaussian moments and held-out diagnostics pass, but
the declared rank-one Gaussian experiment does not establish an end-to-end
benefit: learned transport NUTS had lower minimum ESS/second than Fisher NUTS and
Fisher WALNUTS. Pathfinder used 508 target calls versus 13 for score ascent and
selected a worse point on this Gaussian. M9 closes with a negative promotion
decision; all additions remain optional. See
[M9 implementation and evidence](docs/m9-geometry.md).

### M10 — advanced integration and accelerator research (implemented; not promoted)

- [x] Joint geometry/step/coefficient warmup with epoch boundaries and final freeze.
- [x] HVP-based curvature estimates, conservative stability bounds and s-AIA experiments.
- [x] Exact Gaussian-plus-residual flow with model capability and independent oracle.
- [x] Cheap/expensive force splitting with separate evaluation counters.
- [x] HVP/force-gradient and Hessian-free methods only with complete validity tests.
- [x] Explicit scalar/coordinate-major layout adapters and batched FSM execution.
- [x] Benchmark LF/OMF2/BCSS2/adaptive2 at equal force budgets, not equal edge counts.

Exit: retained-sampling stationarity/reversibility and ESS/sec including launch,
layout conversion, warmup, derivative and factorization costs.

October 15 completion: joint warmup consumes covariance, step, HVP and coefficient
state before returning a fixed two-stage HMC kernel. Exact Gaussian/residual,
nested cheap/expensive, analytic force-gradient and explicitly nonsymplectic
Hessian-free kernels have reversal, volume, stationary-moment, counter and
allocation tests. Explicit layout conversion and transactional batched leapfrog
agree across chain/coordinate-major storage. The declared 32-force release run
passes modern retained diagnostics for OMF2 and adaptive2, but LF and BCSS2 expose
fixed-trajectory resonance (maximum R-hat 1.0259 and 1.2112). The suite-wide exit
gate is therefore negative and no M10 path is promoted. See
[M10 implementation and evidence](docs/m10-advanced-integration.md).

### M11 — Riemannian HMC and adaptive manifold methods (S1 research)

Goal: position-dependent geometry after Euclidean mechanics are proven.

- [x] Position-dependent metric contract.
- [x] Metric log determinant, inverse solve/action, and derivatives.
- [x] Research generalized implicit leapfrog with bounded forward iterations,
  residual diagnostics and transactional rejection.
- [x] Experimental SoftAbs Hessian metric.
- [x] Fisher metrics for selected model families.
- [x] Analytic/Enzyme Hessian-vector and metric-derivative operations.
- [x] Explicit events for failed implicit solves.
- [ ] Enforce numerical reverse support with a per-proposal momentum-flipped
  involution/reversibility check, or derive and implement another valid correction.
  A fixed forward residual does not establish that the reverse implicit solve
  selects the same branch.
- [ ] Validate the implemented proposal's volume/Jacobian assumption across
  nonlinear metrics, multiple/no-solution cases, tolerances and iteration budgets.
- [ ] Execute independent RMHMC references and broader stationarity/SBC suites on
  logistic, hierarchical and funnel-like targets.
- [ ] Validate generated second/third derivatives in the pinned Enzyme toolchain;
  analytic callbacks alone do not close the generated-derivative gate.
- [ ] Replace or bound explicit dense inverse/O(d^4) SoftAbs work and benchmark
  scalar/Faer/BLAS paths with derivative, factorization and solve costs separated.

Exit gate: the production numerical map enforces reverse support and its required
volume/Jacobian property; independent published targets and SBC agree; costs
separate derivatives, factorization, solves, reverse checks and dynamics. This
gate is open.

#### Adaptive manifold follow-up

Goal: evaluate Wang et al. and later adaptations on validated RMHMC.

- [x] Implement a bounded warmup-only Wang-style finite-grid GP search.
- [x] Separate warmup-only adaptation from valid transition mechanics.
- [x] Compare with Euclidean NUTS, ChEES-HMC, and base RMHMC.
- [x] Reject default promotion under the declared logistic comparison.
- [ ] Re-run adaptation evidence only after the underlying RMHMC transition clears
  exactness, using independently reproduced held-out paper targets and total cost.

Exit gate: reproducible paper benchmarks, stationarity tests, and a written support/reject
decision for each method.

October 16 review: the typed position-metric boundary, dense generalized leapfrog,
selected analytic Fisher families, bounded SoftAbs and higher-order callback
adapter are implemented as research paths. Existing one-dimensional reversal,
numerical-Jacobian, derivative, stationary, failure and allocation tests remain
useful but do not prove that every finite-tolerance implicit proposal is reversible:
the production transition has no reverse-branch/involution check. The exact-target
promotion gate is therefore open, consistent with the numerical issue described by
Lelièvre, Santet and Stoltz. A consumed finite-grid Wang-style GP tuner preserves
warmup-only adaptation. On the declared logistic comparison it improved fixed
RMHMC mixing, but remained about 12 times slower than NUTS and 66 times slower
than ChEES; all M11 samplers and adaptation remain S1 research opt-ins. See
[M11 implementation and decision](docs/m11-riemannian.md).

### M12 — SMC and particle MCMC (planned; S0 contracts only)

Goal: state-space inference independent of Hamiltonian internals.

- [ ] Stable log-weight normalization and particle ESS.
- [ ] Multinomial, stratified, systematic, and residual resampling.
- [ ] Bootstrap/auxiliary filter core with SoA particles.
- [ ] Deterministic parallel propagation/weighting.
- [ ] Conditional SMC, particle Gibbs, and PMMH.
- [ ] Reuse core model, RNG, diagnostic, and output infrastructure.

Exit gate: unbiased likelihood-estimator tests where applicable; agreement on analytically
solvable linear-Gaussian models; resampling passes distributional/boundary tests.

M12 remains independent of Hamiltonian research, but its public traits must be
derived from working vertical slices rather than stabilized in their current
three-trait placeholder form. Schedule its first filter after B1 event/batch/support
semantics are settled so particle and ordinary modeling APIs do not diverge.

### M13 — exactness audit and public maturity boundaries

Completed 2026-10-20. This is a pre-release boundary milestone, not sampler or
semver-stability promotion. Evidence: [API audit](docs/m13-api-boundary.md),
[map laws](docs/m13-exactness-audit.md), and the executable `tools/ci` inventories.

- [x] Individually classify every reachable declaration/implementation, all explicit
  features, root exports, facades and CLI binaries. The schema-v3 ledger covers
  4,729 distinct API lines, including conditional Faer exports; mixed adaptation
  and integration owners have explicit reference/research classifications.
- [x] Define the S2 reference boundary around targets, transforms, constant metrics,
  fixed HMC/RWMH, covariance warmup, NUTS and diagnostics. Research remains named
  and gated, approximate MCLMC has its own namespace, and defaults are unchanged.
- [x] Audit refresh, support, reversal, volume/density, selection and failure laws
  for all 13 chains, five standalone proposal/integrator entries and every MCMC
  feature. Source discovery rejects feature-only unaudited chains. Proposal-only
  components explicitly require their caller's compatible correction.
- [x] Contain RMHMC by renaming the executable type to `ExactnessPendingRmhmc`,
  with no old alias or reference export. Its S1 reversal/volume obligations remain
  open; M13 closes the relabeling option, not the M11 numerical exactness gate.
- [x] Enforce external target/metric/chain/diagnostic/sink conformance from the
  separately locked downstream package; integrators stay sealed and SMC stays S0.
- [x] Enforce compiler-negative ownership, freeze, incompatible integrator,
  exact/approximate, result evolution and typed-parameter-support contracts.
- [x] Pin eight default APIs (5,990 rendered entries), two Faer snapshots and five
  API-neutral variants. Digest-bound added/removed signatures record explicit
  pre-release compatibility adjudication. The selected reference owners completed
  constructor, must-use, non-exhaustive, standard-trait, source and rustdoc review.
- [x] Internalize ownerless error/probe modules. Remaining named modules have
  reference, cross-crate, implemented research or unfrozen design roles. No S3/S4
  or general Rust-language semver completeness claim is made.

Exit met: enforced maturity/adoption states, honest implemented-map claims,
downstream composition and reviewable compatibility diffs. Release graduation
remains M17.

### M14 — complete native modeling and distribution verticals

Completed 2026-10-20 within the explicit numerical and calibration scope in the
[modeling guide](docs/m14-modeling.md), [predeclared protocol](docs/m14-validation-protocol.md)
and [executed results](docs/benchmarks/m14-validation.md).

- [x] Separate normalized density/PMF, unnormalized factors, event/support/batch
  semantics, continuous parameter/event scores and optional bounded sampling.
- [x] Deliver checked Normal, Student-t, LogNormal, Exponential, Gamma, Beta,
  Bernoulli, Binomial and Poisson with values, boundaries/tails, normalization,
  derivative, RNG mean/variance, zero-allocation and measured cost gates.
- [x] Deliver support-typed handles, shared expression references, independent
  plates/reductions, deterministic computations, normalized priors/likelihoods,
  opaque VJPs and immutable lowering through one ParameterLayout/Jacobian boundary.
- [x] Preserve named constrained dimensions/coordinates and transform/block
  metadata in outputs; project geometry groups to unconstrained ranges, including
  dimension-reducing simplex/correlation transforms, outside numerical hot loops.
- [x] Add separate prior/posterior prediction and generated-quantity workflows.
- [x] Run normal regression, hierarchical logistic and hierarchical Wiener4 drift
  from downstream construction through covariance warmup/NUTS, diagnostics and
  prediction. Compare independent formulas/finite differences; execute Enzyme for
  supported elementary normal/logistic primals and analytic opaque DDM VJPs.
- [x] Deliver multivariate Normal, Dirichlet, categorical/multinomial, normalized
  LKJ-Cholesky and stable finite-mixture marginalization with per-law tests.

Exit met: fresh downstream code builds all three models, derivative/measure and
prediction gates pass, and bounded SBC passes for 64 conjugate normal datasets and
32 datasets per hierarchical model. All eight hierarchical coordinate histograms
pass, with zero retained divergences in those suites. These are weak-data regression
gates, not universal calibration or mixing claims. The failed five-step warmup
trial is preserved; the completed policy uses single-step warmup. Production DDM
prediction covers boundary events; an RT RNG is not advertised. Existing Gaussian,
Wiener and sampler fixtures remain intact.

### M15 — turnkey inference sessions and durable runtime semantics

Goal: one mature user workflow over the S2 reference components without hiding
the underlying numerical policy.

- [x] Add a validated session builder for model/target, initialization, covariance
  warmup, NUTS or fixed HMC/RWMH, chains, seed, retained budget, observables, sinks,
  diagnostics and cancellation.
- [x] Return a typed run outcome that distinguishes success, diagnostic warning,
  user cancellation, numerical exhaustion and backend failure while retaining all
  valid completed draws and provenance.
- [x] Define versioned owned output separate from borrowed hot-path records; support
  bounded streaming and non-overwriting durable sinks.
- [x] Specify checkpoint/restart for chain point, RNG, frozen geometry and run phase,
  or explicitly reject unsupported mid-warmup checkpoints. Verify replay scope.
- [x] Exercise backpressure, subscriber loss, observer panic, worker failure,
  cancellation races, shutdown/join and long-run bounded memory.
- [x] Make resource budgets explicit: chains, threads, tree depth, warmup windows,
  diagnostic windows, output storage and wall-clock cancellation.
- [x] Keep Rayon pools caller-owned and Tokio outside numerical loops; libraries
  emit structured events but never install a subscriber.

Exit: the same session API runs sequential/Rayon, streaming/collected and
cancelled/resumed reference inference with documented deterministic boundaries,
bounded resources and no hidden adaptation.

Completed 2026-10-20 in bounded regression scope. `alea_runtime::session` provides
chain-local target/model factories, explicit covariance/scale warmup, typed partial
outcomes, version-one owned output, fail-fast bounded channels and create-new durable
streams/checkpoints. Frozen replay saves exact metric factors, ChaCha12 word position
and the RWMH normal spare; mid-warmup and mixed-phase ensemble restart are explicitly
unsupported. Fifteen session tests plus storage/monitor tests cover parity, races,
fault retention, join and 1,000-versus-100,000-draw peak memory. Full stable CI,
pinned-nightly API reproduction and SIMD session tests pass. The external session
example saves/loads a cancelled native-model NUTS run and writes 96 resumed records.
Method, byte/resource policies and crash/replay limits: [docs/m15-sessions.md](docs/m15-sessions.md).
No sampler default or maturity graduation follows from these operational gates.

### M16 — reproducible validation, calibration and performance service

Goal: provide reproducible, descriptive evidence about inference quality and cost.
Current scope includes broad PosteriorDB coverage and implemented research
methods; no promotion or mandatory 1.5x ESS/s requirement defines M16.

- [x] Complete V1/V2 machine-readable manifests/results with source/toolchain/data
  hashes, model measure, policy, seeds, hardware/threading, actual work counters,
  cold/warmup/retained time, peak RSS and failure records.
- [x] Build `alea-bench` on the manifest library; subprocess isolation separates
  JIT/compiler/process startup from retained execution and prevents RSS contamination.
- [x] Version controlled, pathological and representative target corpora, including
  hierarchical, stochastic-volatility, latent Gaussian, Wiener/DDM and selected
  PosteriorDB models with tuning and held-out partitions.
- [x] Add simulation-based calibration and predictive validation for generative
  model verticals; preserve rank/ESS/R-hat as necessary, not sufficient, evidence.
- [x] Run matched Stan, BlackJAX/JAX, nuts-rs/nutpie and relevant reference tools
  under explicit differences in adaptation, precision, JIT and process accounting.
- [x] Add multi-host performance tracking with uncertainty/regression bands and
  separate correctness CI from noisy hardware alerts.
- [x] Preserve the original v1 promotion report and its immutable declared policy.
- [x] Complete the descriptive v2 PosteriorDB matrix, including research methods,
  high-resolution timing, empirical fidelity, diagnostics and all failures.
  Method: [docs/m16-posteriordb.md](docs/m16-posteriordb.md).

Exit: a second user can regenerate a complete report from one manifest; deliberately
wrong gradients, biased/stuck chains, schema drift and lost failures are detected;
no promotion relies on an untracked `/tmp` artifact.

Original v1 completed October 20. `alea-bench` accounts for 156 comparison jobs across 13 corpus
cases, four pinned engines and three timing repetitions, and 96 calibration jobs
across three verticals (252 total). All calibration jobs pass; 249 jobs complete and three retain nuts-rs
centered-eight-schools initialization failures. The 36 quality failures yield a
negative promotion decision. The checked compressed bundle includes raw draws,
sources, preparation/process logs and a report regenerated by both current and
archived readers. Evidence is from one physical host; multi-host tracking and
synthetic separation/regression tests do not establish cross-host speed. nuts-rs's
native diagonal adaptation is explicitly unmatched and excluded from matched
promotion. Method: [docs/m16-experiments.md](docs/m16-experiments.md); complete
evidence and commands: [docs/benchmarks/m16/README.md](docs/benchmarks/m16/README.md).

The descriptive v2 extension completed October 21 on dusk: 20 posteriors × 26
configurations × three repetitions × four chains, accounting for all 1,560 jobs
(1,267 completed, 260 failed, 33 unsupported). Technical completion does not imply
adequate inference; adverse diagnostics, reference discrepancies and all failures
remain visible. The complete segmented bundle includes raw traces, source/build
provenance and granular phase costs. Current and independently archived readers
reproduce identical reports and supplementary analyses; 35 worker contracts and
the original v1 reader pass. [Findings](docs/benchmarks/m16/posteriordb-v2/findings.md)
and [interactive report](docs/benchmarks/m16/posteriordb-v2/report.html) are checked
in with the immutable evidence. No promotion/1.5x gate or sampler default changes.

### M17 — release engineering and stable API graduation

Goal: graduate each crate independently from S2 through S3 to S4 rather than
declaring the whole workspace mature at once.

Release preparation implemented 2026-10-21; **reference S3 candidate cycle completed
2026-10-25, S4 graduation remains open**. The
[release policy](docs/releases.md), [migration guide](docs/migration.md), package
READMEs, metadata validation, independent source/package consumers, deterministic
source archives with SSH signing/verification, semantic compatibility wrapper and
portable CI workflow are in place. Local scalar stable/MSRV passes 398 workspace
tests, Clippy and rustdoc; the reference API inventory remains unchanged. Four
reference crates pass cargo-semver-checks against the independently sealed M16
pre-M17 source, and a deliberate removed API is rejected. All nine `.crate`
artifacts build a fresh reference consumer. Source archives reproduce byte for
byte, including from unpacked source without Git. Native ASan/UBSan, bounded
parser mutations and a 409,600-step/200-restart replay soak pass.
Strict Miri passes all six aligned-buffer tests; actionlint validates the release
workflow. These original local preparation results are supplemented by the exact
candidate's successful native runs in the
[candidate review](docs/releases/2026-10-25-reference-candidate/README.md).

Licensing is now declared per package: MIT OR Apache-2.0 for the six reusable Rust
libraries; GPL-3.0-or-later for ffi/distributions/cli. Full texts and upstream notices
ship in each package; dependency-graph and archive checks enforce the split.
Candidate `148c22c` passes all ten native platform/compiler lanes and the complete
feature/Enzyme/Miri, sanitizer/property/soak and artifact/semver gates. The full
independent workflow passes from all native lanes, actual packages and the signed
archived source, including 32 retained SBC datasets and exact 64+192 draw recovery.
The cycle repaired incomplete NUTS checkpoint policy replay, deliberately advancing
checkpoints to v2 while preserving draw-stream v1. All nine package hashes and the
source archive match independent CI output. That frozen source qualified as an S3 candidate; later pre-alpha APIs may change.
Implementations and research retain their existing classifications. No
publication, S4 or numerical-default change is claimed. The ABI stays design-only
because it has no two real consumers.

- [x] Complete Cargo metadata, licenses, feature documentation, MSRV policy,
  platform/CPU support matrix and reproducible signed source artifacts.
- [x] Enforce `missing_docs` and runnable fresh-project examples for the intended
  stable surface; document errors, panics, safety, determinism and resource costs.
- [x] Run Linux/macOS/Windows and supported-architecture CI, MSRV, stable/nightly
  features, Miri, native FFI sanitizers, fuzz/property suites and operational soaks.
- [x] Qualify the frozen reference candidate through the signed
  candidate, independent downstream calibration/recovery and multi-host observations.
- [ ] Stabilize pure-Rust core/math/reference-MCMC APIs first; graduate optional
  Enzyme, native FFI and research algorithms on their own evidence schedules.
- [ ] Implement the versioned compiled-target ABI only with two real consumers,
  ownership/panic/version negotiation tests and an explicit compatibility policy.
- [x] Publish migration guides, deprecation and bug/security reporting policies;
  run automated API/semver checks for every release candidate.
- [x] Complete at least one S3 release cycle without unresolved correctness,
  unsoundness, data-loss or schema-compatibility defects before any 1.0/S4 claim.

Exit: each published crate states its own S4 surface and compatibility window;
research APIs cannot be imported accidentally as stable defaults; fresh downstream
projects reproduce documented inference workflows and release artifacts.

## Validation suite

### B1/B2/B3 — native distributions and modeling

Completed through M14's declared scalar/vector/discrete/correlation/mixture and
three-model verticals. See [the guide](docs/m14-modeling.md) and
[results](docs/benchmarks/m14-validation.md). Optional CDF/quantile and broader
catalogue capabilities remain future, explicit additions. Discrete latent
variables are marginalized or require a compatible future kernel; they are not
continuous HMC parameters. Release stability remains a separate M17 gate.

### V1/V2 — comparable experiments and diagnostic validation

- [x] Versioned machine-readable experiment manifest/results, fixed target corpus,
  exact evaluation counters, separate warmup/retained timings and memory/allocation scope.
- [x] Independent modern-diagnostic fixtures and deliberate stuck/shifted/ragged/
  nonfinite failure cases exist for M6; M16 adds model-level SBC, predictive gates,
  source-bound manifests and explicit failed-job/schema/gradient fault tests.
- [x] Predeclared workload, uncertainty, failure-rate and promotion thresholds.
- [x] `alea-bench` CLI after the manifest/library contracts exist; do not expose
  sampler choices that have no implementation.

Exit: a second user reproduces a report from its manifest; held-out comparisons
include failures and use the same diagnostics across engines. Success criteria are
in [the settled plan](docs/settled-plan.md), not inferred from a favorable microbenchmark.

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
- Numerical reversibility checks for nonseparable HMC: <https://arxiv.org/abs/2303.15918>
- RMHMC finite-tolerance study: <https://arxiv.org/abs/2111.09995>
- Wang et al. (2013): <https://proceedings.mlr.press/v28/wang13e.pdf>
- Pathfinder: <https://jmlr.org/papers/v23/21-0889.html>
- Pareto-smoothed importance sampling: <https://www.jmlr.org/papers/v25/19-556.html>
- Simulation-based calibration: <https://mc-stan.org/docs/stan-users-guide/simulation-based-calibration.html>

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
| 2026-10-05 | Evaluate the Fisher middle square root through Cholesky/SVD factors to repair noncommuting accuracy loss; preserve the estimator, tolerances, sequential execution and rejection policy. Remeasure cold-fit costs. | implemented; 864 planted-solution cases |
| 2026-09-24 | Make this file the canonical engineering roadmap. | accepted |
| 2026-09-24 | Prioritize safety and a green workspace before samplers. | accepted |
| 2026-09-24 | Use a fused unconstrained value/gradient contract. | core protocol increment 4; target-bound HMC consumer increment 5 |
| 2026-09-24 | Put `std::autodiff`/Enzyme behind a backend boundary. | proposed; validate M2 |
| 2026-09-24 | Implement HMC, adaptation, and NUTS before ChEES/RMHMC. | accepted |
| 2026-09-24 | Use deterministic CPU parallelism; reserve Tokio for I/O/control. | implemented M6: caller-owned Rayon, dedicated diagnostics, Tokio subscriptions/control |
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
| 2026-09-25 | Separate deterministic scalar-oracle agreement, fixed-seed analytic moments, and version-pinned external reference validation; do not treat any one as a substitute for the others. | all three gates passed, M3 increment 11 |
| 2026-09-27 | Keep phase evolution internal and separate from trajectory selection; consume the borrowed phase token on a failed step so partial scratch cannot be committed. Preserve the existing buffer budget. | implemented, increment 7 |
| 2026-09-27 | Check in executed CPU-float64 BlackJAX endpoint references and a separately pinned regeneration environment; keep reference tools outside Rust runtime/build dependencies. Endpoint agreement does not imply statistical agreement. | implemented, increment 8 |
| 2026-09-27 | Enforce the supported feature matrix and unsafe-documentation policy; gate unsupported algorithms and fail explicitly for unimplemented autodiff. | implemented, M0 completion, increment 9 |
| 2026-09-27 | Publish consuming phase evolution plus coherent reusable endpoint snapshots; leave tree selection to M5 and preserve HmcChain's eight-buffer budget. | implemented, M1 completion, increment 9 |
| 2026-09-27 | Initialize foreign outputs before Rust references; catch callback panics and resume after C; require unsafe for raw callback contracts; repair C cleanup using guarded build-time patches. | implemented and audited, increment 9 |
| 2026-09-28 | Use validated flat transforms with analytic Jacobian pullbacks; differentiate pure constrained densities with Enzyme and compose opaque kernels at an explicit derivative boundary. | implemented and executed, M2 |
| 2026-09-28 | Use row-major matrix outputs, centered stick-breaking simplex and a nonredundant stereographic sphere chart; reject rounded support boundaries rather than clamp. | implemented; numerical Jacobian/property gates pass |
| 2026-09-28 | Replace the source-built AD experiment with content-pinned upstream nightly-2026-06-23 and its matching Enzyme component; require native release execution in CI. | implemented; default/stable pins unchanged |
| 2026-09-28 | Validate fixed HMC against independently executed BlackJAX trajectories and sampling chains using separate RNG streams and explicit batch-means uncertainty; preserve the checked kernel and endpoint divergence policy. | implemented and executed, M3 |
| 2026-09-28 | Scope zero-allocation guarantees to library-owned Rust hot paths and tested models; arbitrary callbacks and foreign malloc require separate evidence. Retain safety checks while establishing realistic workload baselines. | measured, M3 |
| 2026-09-29 | Replace the workspace with the target nine-crate architecture now; do not retain compatibility packages, adapters, mutable-state kernels or placeholder algorithms. | implemented, increment 12; supersedes additive migration decisions |
| 2026-09-29 | Keep model/state-space contracts in core, storage/metrics in math, Wiener and its FFI dependency in distributions, and orchestration outside sampler hot paths. Enforce production dependencies in CI. | implemented, increment 12 |
| 2026-09-29 | Breaking compatibility must preserve working behavior: retain derivative-free RWMH, explicit warmup, aligned collection, honest classical diagnostics, benchmarks and research sources. Audit test replacements in CI. | implemented, migration preservation audit |
| 2026-09-30 | Complete covariance warmup as a reproducible reference; add paired Fisher and spectral low-rank geometry behind independent promotion gates. | implemented M4; experimental M4F |
| 2026-09-30 | Expose complete symmetric macrosteps and optional potential-HVP/batch capabilities; distinguish macrostep counts from gradient calls. | implemented |
| 2026-09-30 | Separate metric initialization from estimation; identity is the default, single-score initialization explicitly experimental after the September tail-sensitivity report. | implemented and regression tested |
| 2026-09-30 | Keep native distributions and typed modeling first-class alongside the composable inference engine; defer a separate textual language, not modeling itself. | accepted; B1/B2/B3 and settled plan |
| 2026-09-30 | Extract dynamics, step control, trajectory and correction from the tested HMC; do not force WALNUTS or isokinetic dynamics into a coefficient-only integrator. | accepted; A3/A4/A6 |
| 2026-09-30 | Require independently validated diagnostics before automatic stopping; keep retained adaptation frozen until stronger adaptive-MCMC gates pass. | accepted; A2/V2/A5 |
| 2026-10-16 | Separate implementation, validation, API stability and default selection. Record maturity and adoption independently; a completed negative experiment does not promote a method. | accepted; maturity policy and M13–M17 |
| 2026-10-16 | Reopen the RMHMC exactness gate: a bounded forward implicit residual and sampled reversal test do not guarantee reverse-branch support for every proposal. Require a runtime involution check or another derived correction before exact-target promotion. | open M11/M13 gate |

## Immediate next actions

Use the breaking architecture and APIs in [docs/architecture.md](docs/architecture.md).
Do not reintroduce old crate paths, mutable-state compatibility APIs or duplicate
samplers while implementing subsequent milestones.

1. Obtain external application feedback on the completed S3 reference candidate,
   maintain its native/soak checks, and settle per-crate stable compatibility and
   publication decisions. Preserve explicit outcomes, resource budgets and frozen
   retained adaptation; a newly found release-blocking defect resets its cycle.
2. Preserve covariance warmup and multinomial NUTS as the S2 reference path.
   M4F, M7 and M8–M11 retain their recorded promotion gates. RMHMC remains
   `ExactnessPendingRmhmc`; do not call its M11 map exact.
3. Extend M16 corpora and physical-host observations only with new predeclared
   protocols. Preserve existing failures, thresholds and archived source identity.
4. Implement M12 particle inference using settled B1 support/event/batch semantics;
   derive traits from working filters rather than freezing placeholders.
5. Implement the compiled-target ABI only when two real consumers and
   version-negotiation tests justify it.
