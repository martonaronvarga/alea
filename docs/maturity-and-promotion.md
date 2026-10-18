# Alea maturity and promotion policy

Reviewed: 2026-10-16

This document separates three questions that earlier milestone reports sometimes
collapsed:

1. Is the mechanism implemented?
2. Under which assumptions has it been validated?
3. Is its public API stable, and should it be a default?

A completed research milestone can legitimately end in a negative promotion
decision. Passing moment tests does not make an API stable, and a stable component
does not become the default merely because it is mathematically valid. The
[canonical roadmap](../roadmap.md) owns scheduling; this document owns the common
maturity vocabulary and promotion gates.

## Maturity and adoption are separate

Every public component has one maturity state:

| State | Meaning |
|---|---|
| S0 design | A documented contract or preserved prototype; not an implemented user capability. |
| S1 research | Implemented enough to investigate, but one or more exactness, numerical, statistical, performance, API, or operational gates remain open. |
| S2 validated | Its declared scope has independent correctness evidence and explicit failure semantics. The API may still change before release. |
| S3 release candidate | Intended stable surface; end-to-end, portability, documentation, downstream-use, and compatibility gates pass. |
| S4 stable | Covered by Alea's published compatibility policy and release process. |

Adoption is recorded independently as `reference`, `recommended`, `opt-in`,
`approximate`, or `not exposed`. `Recommended` is a policy choice, not a theorem
that the method wins on every target. `Approximate` mechanisms never silently
enter an exact-target interface.

A component's advertised state is the minimum of its required evidence axes, not
an average. A fast implementation with an open invariance proof stays S1. An exact
but uncompetitive algorithm may reach S2 while remaining opt-in indefinitely.

No Alea crate is S4 today. All workspace crates are `publish = false`, the project
is pre-alpha, and neither semver compatibility nor the complete modeling workflow
has been declared stable.

## Evidence axes

Each promotion review records all of the following, using `not applicable` only
with a reason:

| Axis | Required evidence |
|---|---|
| Mathematics | Target measure, state space, proposal law, correction, adaptation boundary, and exact/approximate guarantee are stated. Preconditions are enforceable or checked. |
| Numerics | Conditioning domain, finite checks, convergence tolerances, failure atomicity, precision policy, and independent scalar/high-precision oracles are covered. |
| Statistics | Fixed-seed moments plus rank/folded R-hat, bulk/tail ESS, MCSE, divergences and method-specific diagnostics on declared observables. Failures remain in the report. |
| Performance | Total setup, warmup and retained time; actual derivative/metric calls; allocations and peak RSS; cold factorization/JIT/launch costs; uncertainty and hardware provenance. |
| API | Validated construction, typed errors, documented laws, ownership/freeze semantics, examples, compile-fail misuse tests, and an external-implementer test where extension is promised. |
| Operations | Deterministic seed identity, cancellation/failure behavior, bounded memory, output schema, restart policy, supported platforms/features, and reproducible artifacts. |
| Ecosystem | Representative models, interoperability boundary, release notes, MSRV/feature policy, and compatibility checks appropriate to the component. |

Benchmark evidence never repairs a failed mathematical or statistical gate. A
negative benchmark remains evidence and must not be erased by retuning the same
seeds, thresholds, or held-out workloads.

## Promotion ladder

### S0 to S1: research implementation

- Cite the algorithm and pin any executable reference implementation used.
- State coordinate, metric, density, Jacobian, RNG, and precision conventions.
- Give approximate algorithms an unmistakable type/API marker.
- Bound loops, storage and failure paths; do not publish partial live state.
- Add focused unit and property tests before performance claims.
- Label missing pieces honestly: an ATLAS curvature probe is not an ATLAS sampler.

### S1 to S2: validated component

- Pass an independent mathematical or executable-reference oracle.
- Establish the proposal's invariance requirements, including every numerical
  solve and branch used in production.
- Pass adversarial non-finite, backend-error, panic, rollback, shape and allocation
  tests over the supported domain.
- Pass a predeclared multi-chain statistical suite on more than toy Gaussians.
- Report full costs and losses against the strongest relevant Alea reference.
- Document supported assumptions and unsupported regimes in the public API.

### S2 to S3: release candidate

- Exercise the component through a complete user workflow rather than a bespoke
  benchmark example.
- Pass Linux/macOS/Windows and supported-architecture CI, MSRV, stable/nightly
  feature, Miri/sanitizer, rustdoc and downstream-build gates as applicable.
- Freeze names, trait laws, error taxonomy, feature behavior and serialization
  schemas; run an API/semver diff against the accepted baseline.
- Supply runnable examples, migration notes, resource limits and troubleshooting.
- Run independent held-out models and simulation-based calibration where the
  model is generative. R-hat/ESS are necessary diagnostics, not calibration proofs.
- Establish benchmark regression bands on multiple machines without converting
  noisy microbenchmarks into hard correctness failures.

### S3 to S4: stable

- Publish signed, reproducible release artifacts and complete crate metadata.
- Commit to a compatibility window, MSRV policy, security/bug reporting process,
  and deprecation policy.
- Complete at least one release-candidate cycle without unresolved correctness,
  data-loss, unsoundness, or schema-compatibility defects.
- Verify fresh-project documentation and downstream integration without repository
  internals, local fixtures, or undeclared toolchains.

### Stable or validated to recommended/default

- Predeclare the comparator, workload family and decision threshold.
- Show a meaningful aggregate reliability or total-cost benefit without a worse
  diagnostic-failure rate. Publish losing targets and uncertainty.
- Provide a bounded fallback and explain when users should choose another method.
- Keep a validated reference implementation after promotion.

Default changes require a separate decision even when every S2/S3 gate passes.

## Exact MCMC and adaptation gates

For an exact-target transition, Metropolis correction based on energy alone is
valid only under the proposal properties assumed by that correction. The review
must cover the implemented floating-point map, not only the ideal differential
equations:

- refresh law and extended target density;
- forward/reverse support and selection probability;
- involution or momentum-flipped reversibility of the actual proposal;
- volume preservation, or the correct Jacobian/proposal-density term;
- behavior when an implicit solve has zero, one, or multiple numerical solutions;
- rejection/error behavior that leaves one coherent live state;
- stationary tests sensitive to small persistent bias.

This is an open gate for the current RMHMC transition. Fixed-budget Picard work and
a small forward residual make work deterministic, but do not guarantee that the
reverse solve returns the original numerical branch. Before exact-target S2
promotion, Alea must implement and test a per-proposal reversibility/involution
check (or another derived correction), reject failed reverse support, and validate
volume/Jacobian assumptions across adversarial nonlinear metrics. This follows the
issue and remedy in Lelièvre, Santet and Stoltz,
[Unbiasing Hamiltonian Monte Carlo algorithms for a general Hamiltonian function](https://arxiv.org/abs/2303.15918),
and the tolerance study in
[On Numerical Considerations for Riemannian Manifold Hamiltonian Monte Carlo](https://arxiv.org/abs/2111.09995).

Finite warmup followed by a consumed freeze is reviewed as a nonadaptive retained
kernel. Continuing adaptation additionally requires diminishing transition-kernel
changes, containment/stability assumptions, discontinuous-rank handling, checkpoint
semantics, and dedicated stress evidence. A diminishing scalar weight by itself is
not enough.

## Current maturity ledger

These states describe the present code and declared scope, not the long-term value
of the underlying algorithms.

| Area | Current state | Adoption | Main reason it is not stable/default |
|---|---|---|---|
| Aligned initialized storage; fused target evaluation; transactional point state | S2 | reference | Public-law/downstream tests, portability and semver release gates remain. |
| Scalar/vector/matrix transforms and parameter layout | S2 | reference | Ergonomic typed modeling, metadata, compile-fail misuse coverage and end-to-end SBC remain. |
| Identity/diagonal/dense Euclidean metric algebra | S2 | reference | API freeze, downstream extension policy and multi-platform release evidence remain. |
| Block/sparse constant metrics and frozen affine map algebra | S2 component | opt-in | No mature factorization/ordering/adaptation workflow or broad end-to-end benefit yet. |
| Analytic targets and first-order Enzyme adapter | S2 in tested scope | analytic reference; Enzyme opt-in | Enzyme is pinned-nightly; generated higher-order coverage and stable compiler/tooling UX remain. |
| Gaussian and Wiener/DDM kernels/models | S2 in narrow scope | reference examples | No general normalized distribution capability or tiered modeling library. |
| Fixed Euclidean HMC and fixed-scale RWMH | S2 | HMC reference; RWMH baseline | No stable turnkey session/API compatibility promise or broad model-level calibration suite. |
| Covariance/dual-averaging warmup | S2 | reference/default warmup | Needs release-level workflow, portability and public configuration/diagnostic audit. |
| Multinomial NUTS | S2 | recommended dynamic kernel | Correctness is strong; stable workflow, SBC, portability, API and release gates remain. |
| Multi-chain runtime and modern diagnostics | S2 | reference | Persistence/restart, schema compatibility, broader observables, platform and operational soak gates remain. |
| Fisher warmup/low-rank fitted geometry (M4F) | S1 | opt-in | One predeclared modern diagnostic failed; broad efficiency is unfavorable; high-dimensional nonzero-rank evidence is incomplete. |
| WALNUTS-D/R2P (M7) | S2 algorithm, S1 policy | opt-in | Declared performance suites failed, especially stochastic volatility and target-call cost. |
| ChEES-HMC | S2 in identity/prewhitened scope | opt-in | Evidence is small/analytic; nonidentity semantics, realistic models and total-cost comparisons remain. |
| GHMC, DR-G-HMC and MAMS | S1 | opt-in | Narrow mechanics tests exist; broad diagnostics, tuning and held-out efficiency evidence do not. |
| ATLAS | S1 probe | opt-in research | Only a bounded curvature probe exists; extended-state sampler/correction is absent. |
| MCLMC | S1 approximate | approximate only | It deliberately omits exact correction and must never be presented as exact-target inference. |
| Learned affine warmup and raw multi-path initialization | S1 | opt-in | Declared benchmark was negative; nonlinear transport, PSIS and robust multimodal evidence are absent. |
| Advanced/split/force-gradient/batched integration | S1 sampler policy; some S2 kernels | opt-in | Equal-force suite had resonance failures; accelerator execution and representative split models are absent. |
| Position-metric capability and analytic Fisher metric formulas | S2 component | opt-in | Family coverage, derivative-generation UX and backend scaling remain limited. |
| RMHMC generalized leapfrog | S1 | research opt-in | Runtime reverse-branch/involution gate is absent; dense scalar performance and broad statistical evidence are insufficient. |
| SoftAbs and manifold GP tuner | S1 | research opt-in | Dense Hessian/third derivatives, O(d^4) reconstruction, narrow benchmark and RMHMC exactness dependency. |
| SMC traits | S0 | not exposed as inference | No particle filter, resampler, likelihood estimator or PMCMC implementation. |
| Cubature FFI | S2 in current native use | internal/opt-in | Cross-platform packaging, ABI/version and downstream native-build gates remain. |
| CLI binaries | S1 utilities | development | No complete model-to-diagnostics inference command, manifest, persistence or stable output contract. |
| Preserved `research/` and `archive/` prototypes | S0 | not exposed | Recovery/provenance material only; the historical migration record is not executed semantic evidence. |

## Machine-readable enforcement

[`tools/ci/maturity-ledger.json`](../tools/ci/maturity-ledger.json) is the
executable counterpart to this policy. It records an exact entry for every source
module containing an unrestricted public declaration, every named public API
namespace, every explicit Cargo feature and every Cargo binary. Each entry names a
reusable profile that fixes maturity, adoption and target-guarantee vocabulary.
Research entries also retain a concrete promotion blocker; CI rejects assigning a
validated profile while that blocker remains.

[`tools/ci/maturity.py`](../tools/ci/maturity.py) discovers the live workspace
surface instead of trusting ledger cardinalities. A new public module, named
reference namespace, feature or binary therefore fails CI until it is classified,
while a stale ledger entry fails after removal. It also prevents design-only
contracts from being adopted,
approximate-target mechanisms from using an exact-looking adoption state,
exactness-pending mechanisms from moving above S1, and S3/S4 claims before the
release-candidate policy is deliberately opened. Mutation tests exercise those
failure modes.

Source files still use module-level inheritance rather than final public-item
classification. The additive `reference` facades provide the first narrower path
boundary, while mixed root facades retain the lowest defensible state and an open
promotion gate. M13 still needs research/approximate path refinement,
compile-fail conformance cases and accepted compatibility
decisions before the inventory task can be marked complete.

External implementations of the current open target, Euclidean metric, Markov
chain and diagnostic boundaries, together with the borrowed draw sink, are now
compiled and law-tested by the separately locked
[`tools/downstream-conformance`](../tools/downstream-conformance) package. This is
downstream evidence for those specific contracts, not a promise that sealed
integrators or design-only particle traits are ready to open.

The first item-level compatibility baseline is recorded in
[`m13-api-boundary.md`](m13-api-boundary.md). The pinned nightly CI row regenerates
the current 6,089 reachable default-feature entries across all eight library crates
with `cargo-public-api`; every other row verifies the committed files, counts and
digests. The initial 5,057-entry inventory remains documented as the pre-boundary
baseline. This makes additions, removals and signature changes reviewable, but it
does not promote the recorded pre-alpha surface. Feature-specific snapshots and
path-level maturity refinement for mixed facades remain open.

[`tools/ci/exactness-ledger.json`](../tools/ci/exactness-ledger.json) separately
records the finite-precision laws of every public default-feature `MarkovChain`.
Its checker discovers transitions from the accepted API snapshot and requires
refresh, support, reversibility, volume/proposal-density, selection/correction and
failure claims plus executable evidence. An exact-target entry cannot carry an
open or deliberately omitted law; an approximate transition must expose its
omitted correction; an exactness-pending transition must retain an open law. The
current 13-transition review is summarized in
[`m13-exactness-audit.md`](m13-exactness-audit.md).

## Research re-entry gates

Negative promotion reviews stay closed until a new protocol addresses the recorded
failure rather than rerunning the same experiment.

### Fisher geometry (M4F)

- Pass all predeclared modern diagnostic gates on an independent held-out ensemble.
- Demonstrate nonzero low rank where rank is materially below dimension on several
  non-Gaussian targets; cover noncommuting and difficult spectra beyond planted grids.
- Compare covariance and Fisher under matched NUTS/ChEES policies, total warmup,
  retained time, actual calls, cold fit, peak RSS and failure rate.
- Establish a bounded high-dimensional fitting strategy; rank caps alone do not
  bound the current intermediate decomposition.
- Validate any whole-warmup compatibility claim against executed upstream behavior,
  while preserving deliberate Alea policy differences.

### WALNUTS (M7)

- Retain the proved full-range and reverse-support construction.
- Profile and reduce repeated target work without weakening that construction.
- Predeclare multiscale held-out models where local refinement should help, include
  total warmup, and beat covariance-NUTS or another strongest baseline as a family.
- Treat D and R2P as separate policies; one cannot promote the other by association.

### M8 alternatives

- ChEES: add realistic hierarchical, stochastic-volatility and Wiener/DDM targets,
  explicit whitening/nonidentity contracts, ensemble-size sensitivity and total
  wall-time evidence on multiple machines.
- GHMC/DR-G-HMC: independently verify persistent-momentum stationarity and delayed-
  rejection ratios beyond scalar cases; benchmark tuning cost and metastable targets.
- ATLAS: implement the complete extended proposal and correction before sampler
  claims; validate derivative budgets and reverse support.
- MAMS: expand exact-correction and stationary tests across dimensions/targets and
  compare with tuned HMC/NUTS. Keep MCLMC in an approximate-results category.

### M9 learned and structured geometry

- Separate algebra promotion from learner promotion: frozen affine and sparse/block
  actions need not wait for a successful learned policy.
- Add sparse ordering/factorization input contracts and realistic sparse models.
- Implement the published Pareto-smoothed multi-path resampling with Pareto-k
  diagnostics before calling the initializer Pathfinder-compatible.
- Use held-out nonlinear/multimodal targets, multiple starts and full target-call
  cost; require benefit over simpler covariance/Fisher initialization.
- Nonlinear flows require invertibility, log-determinant, pullback, freeze and
  generalization gates of their own.

### M10 advanced integration and batching

- Promote kernels independently rather than requiring a favorable average across
  unrelated methods, but preserve every existing failed configuration.
- Test randomized/static and dynamic trajectory policies to distinguish an
  integrator defect from fixed-length resonance.
- Validate split declarations against fused targets on representative models and
  count cheap, expensive, full and HVP work separately.
- Extend volume/reversal/Jacobian tests beyond low-dimensional numerical probes.
- Add a real accelerator backend and include transfer, compilation, launch, layout,
  padding, batch occupancy and device memory before accelerator performance claims.

### M11 Riemannian methods

- Add the per-proposal reverse/involution gate described above, with typed rejection
  events and no retry-until-success behavior.
- Test multiple-solution/no-solution implicit maps and adversarial metrics; quantify
  bias sensitivity to tolerance and iteration budget.
- Execute independent logistic and hierarchical RMHMC references and broader
  stationarity/SBC suites, including funnel-like geometries where RMHMC has a reason
  to repay its derivative cost.
- Replace explicit dense inverse work with measured factor/solve kernels; benchmark
  scalar, Faer and BLAS implementations without implying that SIMD fixes cubic work.
- Validate generated Hessian/third-derivative callbacks in the pinned Enzyme toolchain.
- Reduce SoftAbs derivative complexity or state a practical dimension ceiling;
  compare all derivative/factorization/solve costs with reparameterized Euclidean NUTS.
- Re-evaluate manifold adaptation only after the underlying RMHMC transition clears
  exactness, then use held-out targets and a predeclared total-cost decision.

## Stable surface required at each crate boundary

### `alea-math`

- Freeze storage alignment/initialization and metric conventions without exposing
  backend-specific representations as semantic requirements.
- Document trait laws and supply a reusable conformance suite for external metrics.
- Establish scalar/Faer/BLAS/SIMD equivalence domains, feature additivity, thread
  ownership and platform fallbacks.

### `alea-core`

- Freeze density/score/HVP/batch signs, layouts, cache validity and panic/error laws.
- Complete typed parameter metadata, event/batch/support semantics and lowering from
  model coordinates to one unconstrained target.
- Add downstream implementation and compile-fail aliasing/Jacobian misuse tests.

### `alea-distributions`

- Complete B1/B2/B3 capability tiers with checked parameters, normalized log
  probabilities, explicit factors and optional sampling/CDF capabilities.
- Validate boundaries, tails, normalization, derivatives and RNGs per distribution;
  no catalogue-wide maturity claim from Gaussian/Wiener alone.

### `alea-autodiff`

- Keep analytic/foreign targets independent of Enzyme.
- Version generated callback ABIs, compiler/plugin pins and derivative conventions.
- Execute first-, second- and third-order tests where advertised, with finite-
  difference/analytic references and actionable unsupported-operation errors.

### `alea-mcmc`

- Stabilize a small reference surface first: fixed HMC/RWMH, covariance warmup and
  multinomial NUTS. Move or clearly gate research exports before semver freeze.
- Encode exact versus approximate guarantees and warmup/frozen ownership in types
  where misuse would otherwise be possible.
- Add conformance suites before opening currently sealed extension traits. Extract
  broader dynamics/controller/trajectory/correction traits only from concrete uses.

### `alea-runtime`

- Provide one validated session builder for warmup, retained execution, chain seeds,
  observables, sinks, diagnostics and cancellation without hiding numerical policy.
- Define versioned owned output, checkpoint/restart and partial-run/failure semantics;
  keep hot-path borrowed records separate from persistent formats.
- Test bounded backpressure, subscriber loss, shutdown, panics, long runs and replay
  across sequential/Rayon execution and supported platforms.

### `alea-smc`

- Do not stabilize the current traits before implementing M12 vertical slices.
- Derive particle ownership, resampling, ancestry, likelihood and conditional-SMC
  contracts from working filters and PMCMC, then add conformance tests.

### `alea-ffi`

- Pin the native ABI and vendored source provenance, test build/link/runtime failures,
  and document allocator, callback, panic, thread and platform assumptions.
- Keep foreign dependencies out of core inference crates and provide pure-Rust or
  explicit unsupported-platform behavior where appropriate.

### `alea-cli`

- Build on stable library workflows; do not define a second inference policy.
- Add manifest-driven runs, non-overwriting outputs, resume/error reporting and a
  versioned machine-readable summary before calling it a user inference CLI.

## Release-wide gates

- Complete B1/B2 modeling and V1/V2 validation milestones.
- Maintain a machine-readable maturity ledger checked against public exports and
  Cargo features so a research item cannot appear stable accidentally.
- Add API snapshots and semver checks, `missing_docs` enforcement for stable modules,
  doctests/fresh-project examples, and crate metadata/licenses.
- Run fuzz/property tests on parsers/configuration/layout boundaries, Miri on unsafe
  Rust surfaces, native sanitizers on FFI, and long-running failure/cancellation soak tests.
- Publish reproducible target corpora and benchmark manifests with artifact hashes;
  distinguish correctness CI from hardware performance monitoring.
- Define supported platforms, CPU features, MSRV/toolchain policy, deterministic
  replay scope, deprecation policy and security/bug-reporting process.
- Complete one S3 release-candidate cycle before any S4/1.0 claim.
