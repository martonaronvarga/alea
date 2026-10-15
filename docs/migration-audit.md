# Architecture migration preservation audit

Date: 2026-09-29. Baseline: `50b66bb` plus the uncommitted M3 work present before
the nine-crate migration. This supplements [architecture.md](architecture.md)
and the canonical [roadmap](../roadmap.md); it does not reinstate old APIs.

## Inventory and enforcement

The machine-readable inventory is
[`tools/ci/migration-ledger.json`](../tools/ci/migration-ledger.json). It accounts
for **118 source/fixture paths**: 113 tracked deletions and five additional M3
paths that were uncommitted before migration. Each entry lists its destination or
an explicit scaffolding disposition. Split files list multiple destinations.

The baseline contains **188 test definitions**:

| Disposition | Count | Meaning |
|---|---:|---|
| Retained names | 155 | Migrated tests, including uncommitted M3 additions |
| Replaced contract tests | 10 | Explicit mappings to stronger/new-API assertions |
| Dormant research tests preserved | 22 | Uncompiled DDM tests, still not claimed as executed coverage |
| Removed scaffolding | 1 | `data_prep::it_works`, the generated `add(2, 2) == 4` test |

`tools/ci/migration.py` checks destination existence, replacement test definitions,
and Git-blob hashes of preserved research/reference sources. It does not require
old Git history (works with shallow CI clones). Local docs are not prerequisites
of the source gate. The inventory is a regression guard, **not** a semantic proof
or a substitute for executing native, backend, statistical, Enzyme and Miri tests.

### Feature-level continuation

`tools/ci/migration-features.json` adds 31 capability groups with old symbols,
source paths, new owners, behavioral obligations, regression tests and explicit
remaining work. Working algorithms, newly ported utilities, preserved research,
tooling and placeholder-only declarations have distinct statuses. CI rejects
missing baseline files/tests/features, duplicate entries, nonexistent evidence,
or an implementation claim without canonical regression tests. Nine mutation
tests exercise the guard itself; the original 118-path/188-test baseline is fixed.

The dormant `GridSpec`/`LinearSpline1D` functionality is now compiled as
`alea_math::interpolation::{UniformGrid,LinearSpline}`. It retains spacing,
binary-search interpolation and endpoint clamping, reuses input allocations,
and adds validated immutable inputs, NaN errors and overflow-safe evaluation.
Five native tests cover scalar-oracle agreement, extreme/subnormal data and
allocation behavior. Four deterministic tests also run under Miri. The original
approximation sketch remains intact: its NaN-returning likelihood grid is still
unimplemented, not promoted by the utility port.

Four additional latent-model regressions cover affine scalar/SoA agreement,
untouched output tails, shape-failure atomicity, direct Wiener agreement for both
boundaries, and generic context/time/RNG delegation. The existing
`map_into_parallel` entry point is explicitly documented and tested as serial;
no parallel implementation existed to preserve. These tests supplement the old
random-walk moments/zero-scale tests.

## Functionality recovered during review

The first migration draft removed more than compatibility. This review corrected
the following omissions before accepting the migration:

| Old implementation | Current owner and behavior |
|---|---|
| Density-only RWMH | `alea-core::density::{LogDensity,DensityPoint}` and `alea-mcmc::Rwmh`; no gradient evaluation/storage |
| Cached second polar normal | `alea-math::random::NormalGenerator`; explicit seeded recurrence/RNG-consumption regression |
| RWMH dual averaging | `alea-mcmc::adapt::{DualAveraging,warmup_rwmh}`; explicit warmup, fixed-scale burn-in, zero-warmup no-op, averaged scale frozen afterward |
| Draws and size validation | `alea-runtime::draws::collect_draws`; checked dimensions/overflow before RNG use, row-major 64-byte-aligned storage, rejected draws included |
| Acceptance and convergence calculations | `alea-runtime::diagnostics`; acceptance rate, accurately named autocorrelation ESS and classical split R-hat |
| RWMH profiling loop | `alea-rwmh --dim=N --steps=N [--dense]`; invalid numeric arguments report errors |
| Full RWMH benchmark decomposition | 49 cases: six sizes each for identity/dense transitions, density/copy/metric operations, plus no-density transition overhead |
| Backend benchmark automation | `tools/bench-features.sh`, eight feature combinations, no blanket warning suppression |
| Extreme Wiener example | `alea-distributions/examples/wiener_extreme.rs` |

The dual-averaging recurrence retains gamma=0.05, t0=10, kappa=0.75. Invalid
acceptance probabilities cannot mutate it. Extreme proposed scales are clamped
to a finite normal range, unlike the old overflow-prone helper. Tests compare
ordinary updates against the original recurrence and cover zero/fixed/adaptive
warmup and frozen sampling. This is **not** completion of windowed adaptation M4.

The classical diagnostics retain their original mathematical calculations for
valid ordinary inputs, but no longer return apparently healthy defaults for
missing, constant, mismatched or non-finite data. They are not rank-normalized
bulk/tail ESS or modern folded R-hat, and do not satisfy M6.

The runner's observer replaces mandatory hot-loop tracing with caller-controlled
observation outside the numerical transition. This preserves access to transitions
without imposing a logger, asynchronous runtime or formatting allocation.

## What changed deliberately

- Old `kernels`, `memory`, `runtime`, `ffi`, `app`, `data_prep` package paths are gone.
- The second HMC/integrator/boolean-kernel path is replaced by the validated HMC
  and consuming Hamiltonian phases. Its six tests map to evaluation-count/cache,
  rejection/configuration, momentum-mass, reversibility/order and stationary-moment
  tests. Mutation requiring an extra evaluation is replaced by transactional reset.
- The old fused-adapter test maps to direct Gaussian shape validation; fused call
  counting is retained separately in the target/HMC/model tests.
- The three RWMH tests map to validated construction, pre-RNG failures, explicit
  warmup/frozen sampling and aligned collection tests. Invalid mutable options can
  no longer be stored.
- Metric `dim`, panicking `apply_*` and duplicated `try_apply_*` become the single
  fallible `EuclideanMetric` interface. Numerical backend kernels remain present.
- Wiener family/target wrappers become direct `WienerModel` implementations and
  inherent numerical batch operations. Analytic series, quadrature, gradients,
  boundary/reflection behavior, SoA buckets, transforms and tests are retained.
- Wiener-specific opaque derivative composition moves out of core into distributions;
  generic model composition stays in core. Executed Enzyme tests retain the hybrid
  analytic/compiled derivative path.

## Research is preserved, not falsely activated

`research/ddm-prototypes/` preserves all six previously uncompiled DDM sketches,
including independent series/CSE likelihoods, interpolation, affine maps, FFI
layouts and their 22 tests. They had unresolved imports/types and invalid tests
before migration; a likelihood-grid placeholder returns NaN. They remain design
inputs pending repair and independent validation, not production implementations.
The supported latent DDM model and its two tests are compiled in distributions.

`research/cxx-prototype/` preserves the dormant Gaussian bridge and its unused
build script. `research/inference-prototypes/` preserves extension/proposal/family
contracts, NUTS configuration and text-only inference selection. Empty particle
filter/PMCMC files were not algorithms. No fake algorithm exports were reinstated.

The RMarkdown analysis and Stan C++ reference programs/binaries are retained.
The cubature submodule remains at commit
`4ba70a7d595a00cfb15b02763013e0088013adb7`, now under `alea-ffi/vendor/cubature`.
Checked-in BlackJAX fixture bytes are unchanged, including uncommitted M3 fixtures.

## Updating this record

When porting a research item, update its ledger disposition and destination, add
executable replacement tests, and only then retire the preserved source. When
changing a test contract, record its replacement rather than silently dropping
its inventory entry. Preserve the historical baseline and distinguish source
retention from executed coverage.

## Executed verification

Initial verification after restoring the missing functionality on the migrated paths:

- Stable and stable+Faer workspace gates: **173 passing tests/doctests each**,
  all-target checking, Clippy with warnings denied, rustdoc with warnings denied.
- Nightly SIMD/branch-hints and combined SIMD+Faer+LP64 OpenBLAS gates:
  **174 passing tests/doctests each**, the same lint/documentation checks.
- Strict-provenance/symbolic-alignment Miri: **83 passed, 9 explicitly ignored**
  (native statistical/allocation checks); includes density-only caches, RWMH,
  adaptation, draw collection and the migrated latent-model boundary.
- Compiled `std::autodiff`/Enzyme gate: **15 tests passed**, Clippy passed and all
  six gradient-check model cases executed successfully.
- Release-mode RWMH/adaptation/collection/streaming: **13 tests passed**.
- Both inventory/dependency gates pass; cubature commit and preserved fixture/
  research bytes verified. `git diff --check` was clean.
- `nix flake check --no-build` passed on an untracked-source-inclusive temporary
  snapshot for x86_64-linux. This is evaluation, **not** a Nix package build or
  verification on the three other advertised platforms. The `alea-rwmh` dense
  CLI smoke run also passed.

Benchmark reproduction uses `cargo bench -p alea-mcmc --bench hmc` and the restored
`--bench rwmh`, from `crates/Cargo.toml` in the stable Nix shell. Hot-path allocation
assertions pass; no universal speedup is inferred from the crate split.

The feature-inventory continuation added five interpolation and four latent-model
regression tests. The exact staged snapshot passed the stable workspace gate with
**182 tests/doctests**; the combined SIMD+Faer+OpenBLAS gate passed **183**. Both
included all-target checking, warning-denied Clippy/rustdoc and **9 inventory-guard
mutation tests**. All nine new Rust tests also passed in release mode. The eight
Miri-compatible new tests passed strict-provenance/symbolic-alignment Miri; the
allocation-counter test is intentionally native-only. This supplements, rather
than reruns, the earlier full Miri and Enzyme gates. Documentation and AI artifacts
are excluded from the implementation commit.

### Post-migration local baseline

Criterion baseline `architecture-20260929`, stable Rust 1.95.0, default scalar
features, release fat LTO (Cargo's default codegen-unit setting), Intel i5-8250U. No CPU affinity or
frequency control; measurements are a local baseline, not a cross-version speedup
claim. Large test jobs had finished before measurement. Values are Criterion
point estimates per transition; construction is outside the timed loop.

| Case | Microseconds |
|---|---:|
| HMC protocol, dimension 8 | 0.79360 |
| HMC protocol, dimension 64 | 4.4284 |
| HMC protocol, dimension 1024 | 74.635 |
| HMC Gaussian64, identity | 4.4105 |
| HMC Gaussian64, diagonal | 4.9529 |
| HMC Gaussian64, dense | 51.980 |
| HMC logistic1024x16 | 397.88 |
| HMC transformed Wiener4 | 4.2958 |
| Density-only RWMH64, identity | 1.1784 |
| Density-only RWMH64, dense | 2.4698 |

All 49 RWMH benchmark cases compile; only the two RWMH64 cases were timed in this
audit. Run the full suite/backend script for a comparative backend decision.
