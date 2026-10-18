# M13 public API boundary baseline

Reviewed: 2026-10-16

This report establishes the first reproducible item-level public API baseline for
Alea. It is inventory evidence, not a stability promise. The maturity policy in
[`maturity-and-promotion.md`](maturity-and-promotion.md) remains authoritative:
no crate is S3 or S4, and research mechanisms do not become supported merely
because their current signatures are recorded.

## Extraction contract

The project flake pins `cargo-public-api 0.51.0` and
`rustc 1.97.0-nightly (7af3402cd 2026-04-16)`. The extractor uses rustdoc JSON and
records the reachable default-feature API of every workspace library crate. It
omits blanket implementations and compiler-generated auto-trait implementations,
which are too noisy for the current review, but retains explicitly derived trait
implementations because standard traits are part of the intended M13 audit.
Function parameter names are omitted because changing a parameter binding is not a
Rust call-site compatibility break.

The accepted snapshots live under [`tools/ci/public-api`](../tools/ci/public-api).
Their manifest records the toolchain, extraction policy, per-crate item count and
SHA-256 digest. Stable and feature CI rows validate those committed artifacts.
The pinned nightly row also regenerates all eight APIs and rejects any difference.
A change is accepted only after reviewing its maturity and compatibility effect and
deliberately replacing the baseline.

```console
nix develop .#default --command python3 tools/ci/public_api.py --check
nix develop .#default --command python3 tools/ci/public_api.py --bless
```

`--bless` is not a formatter. Its diff is an API review artifact and must not be
run merely to make CI green.

## Initial baseline

| Crate | Reachable entries | Present boundary interpretation |
|---|---:|---|
| `alea-autodiff` | 75 | Mixed first-order and higher-order facade; S1 overall. |
| `alea-core` | 483 | Reference contracts mixed with research/design capabilities. |
| `alea-distributions` | 622 | Narrow validated Gaussian/Wiener surface, not a general distribution library. |
| `alea-ffi` | 76 | Current cubature boundary; packaging and ABI gates remain. |
| `alea-math` | 313 | Reference algebra mixed with experimental Fisher/low-rank fitting. |
| `alea-mcmc` | 3,144 | Broad validated, research, exactness-pending and approximate surface. |
| `alea-runtime` | 337 | Validated execution/diagnostic components without a stable session contract. |
| `alea-smc` | 7 | S0 particle trait design only. |
| **Total** | **5,057** | Inventory only; no release candidate is declared. |

The `alea-mcmc` cardinality confirms that stabilizing its current root facade would
accidentally freeze a large research surface. M13 should first define a small
reference boundary around fixed HMC/RWMH, covariance warmup, NUTS and their required
Hamiltonian/configuration types. WALNUTS, Fisher fitting, learned geometry, advanced
integration, M8 alternatives and RMHMC must remain explicitly research or opt-in.

## Additive reference namespace

The first boundary refinement adds `reference` facades to `alea-core`, `alea-math`,
`alea-mcmc` and `alea-runtime`. They compose the existing fused target and transform
contracts, constant Euclidean metrics, exact-target HMC/RWMH/NUTS kernels, finite
covariance warmup, draw collection and classical diagnostics. They are aliases of
the existing implementations, not forks with independent numerical behavior.

The accepted baseline now contains 6,089 rendered entries: 661 for `alea-core`,
459 for `alea-math`, 3,775 for `alea-mcmc`, and 414 for `alea-runtime`, with the
other crate counts unchanged. Of these, 267 entries have a named `reference` path.
The 1,032-line increase is larger because rustdoc expands reachable inherent and
trait implementations through each alias; it does not represent duplicate
implementations.

These S2 facades remain pre-release and make no semver promise. The MCMC reference
facade deliberately excludes Fisher fitting, WALNUTS policy, learned geometry,
advanced integration, M8 alternatives, approximate MCLMC and RMHMC. Root exports
remain available for compatibility during M13, but retain their more conservative
module-level maturity classifications.

## Limits and next review

This baseline uses no default features. Feature declarations themselves are already
classified by the maturity ledger, but feature-specific reachable APIs are not yet
separate snapshots. Before a feature becomes an S3 candidate, add its supported
feature combination to the API matrix and prove feature additivity.

The baseline detects additions, removals and rendered signature/implementation
changes. It does not decide whether a difference is semver-compatible, correctly
documented or mature. The next M13 review must:

- finish associating paths outside the new reference facades with explicit research
  or approximate boundaries;
- decide which root re-exports become internal or move behind deliberate modules;
- audit validated constructors, `#[must_use]`, `#[non_exhaustive]`, standard traits,
  error sources and rustdoc laws;
- keep downstream conformance and compile-fail contracts aligned with every public
  extension point proposed for freezing; and
- add feature-specific snapshots for every feature proposed as supported.

Until those reviews finish, API diffs are rejected for explicit human adjudication,
not because the recorded pre-alpha surface is promised unchanged.

The positive half of the conformance gate is enforced by the separately locked
[`tools/downstream-conformance`](../tools/downstream-conformance) package. It
implements target, metric, chain and diagnostic traits externally and composes the
borrowed runtime sink without private access. The reusable laws and explicit
exclusions are recorded in
[`m13-downstream-conformance.md`](m13-downstream-conformance.md). Its negative
complement, [`tools/compile-fail`](../tools/compile-fail), verifies that invalid
phase ownership, adaptation, integrator and exactness compositions fail at their
call sites. These gates are necessary but not sufficient for S3: the remaining
classification, constructor, documentation and feature-matrix reviews above still
prevent an API-freeze claim.
