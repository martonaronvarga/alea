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
Function parameter names are retained by the pinned renderer. They are review
metadata: changing a binding alone is not a Rust call-site compatibility break,
although the conservative textual adjudicator may request review of that delta.

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

The additive-facade baseline contained 6,089 rendered entries: 661 for `alea-core`,
459 for `alea-math`, 3,775 for `alea-mcmc`, and 414 for `alea-runtime`, with the
other crate counts unchanged. Of these, 267 entries had a named `reference` path.
The 1,032-line increase was larger because rustdoc expands reachable inherent and
trait implementations through each alias; it did not represent duplicate
implementations.

These S2 facades remain pre-release and make no semver promise. The MCMC reference
facade deliberately excludes Fisher fitting, WALNUTS policy, learned geometry,
advanced integration, M8 alternatives, approximate MCLMC and RMHMC.

## Root boundary cleanup

The reviewed pre-release break removes root aliases for ChEES, GHMC, MAMS,
WALNUTS, RMHMC, advanced integration, batched integration and their supporting
types. The implementations remain public through the explicitly classified named
modules. HMC, NUTS, RWMH and `MarkovChain` retain their ergonomic root aliases as
the validated reference path. Approximate MCLMC is additionally exposed through
`alea_mcmc::approximate`; its namespace is machine-classified as S1 approximate
and cannot inherit the exact-target reference profile.

The resulting default-feature baseline contains 5,294 entries, including 2,980
for `alea-mcmc`; every other crate count is unchanged. The 795 removed lines were
duplicate reachability through ambiguous root aliases, not removed algorithms.
The four reference facades still account for 267 rendered entries, and the new
approximate facade adds eight. The maturity ledger now separately inventories all
25 non-module crate-root items: 22 validated S2 items and the three S0 particle
contracts. A new root alias therefore cannot enter through an API-snapshot update
without an explicit maturity decision.

## Reference API hardening

The first item-level audit covers the types named by the four `reference` facades,
not every public research type in their defining modules. Validated transforms,
layouts, aligned buffers, Euclidean metrics, sampler options and emitted result
records now implement the ordinary comparison or cloning traits their semantics
support. Values whose construction or returned diagnostics would otherwise be
silently discarded are `#[must_use]`.

Emitted HMC, NUTS, RWMH and warmup records are `#[non_exhaustive]`, as are evolving
termination, numerical-rejection, adaptation, covariance and diagnostic errors.
Their private or non-exhaustive representation preserves validated construction
and allows additive diagnostics without forcing downstream exhaustive matches.
`DimensionError` deliberately remains externally constructible because downstream
target implementations must be able to report that contract failure.

The compiler-negative package now contains six downstream cases. In addition to
the ownership, adaptation, integrator and exactness contracts, it proves that a
NUTS termination match requires a wildcard and that an HMC transition cannot be
assembled with a struct literal. Fallible reference operations document their
error conditions, and wrapper errors retain their `source` chain through the
existing transparent/source annotations.

The reviewed hardening snapshot contains 5,464 rendered entries: 679 for
`alea-core`, 507 for `alea-math`, 3,074 for `alea-mcmc`, and 424 for
`alea-runtime`; the other crate counts remain unchanged. The 170 added entries are
derived standard-trait implementations repeated through direct and reference
paths, not new algorithms or new root exports.

## Named-module disposition

The first release-disposition review removed three compatibility-free module
paths. `alea_core::error` is now private and `TransformError` is owned by the
`transform` and `reference` surfaces. `alea_distributions::error` is private and
its checked Wiener `ProbError`/`Result` vocabulary is owned by `wiener`.
`alea_mcmc::atlas` is also private: it never implemented an ATLAS transition, and
its raw probe/result interface had no external consumer. Joint warmup now exposes
only the algebraically relevant `CurvatureProbeOptions` and error types through
`adapt`; unused local-step bounds were removed from that configuration.

This disposition reduces the current reviewed snapshot to 5,421 reachable
entries: 648 for `alea-core`, 621 for `alea-distributions`, 507 for `alea-math`,
3,063 for `alea-mcmc`, 424 for `alea-runtime`, and unchanged counts for the
other three library crates. The reduction removes module-path duplication and
the unused raw probe surface; it does not remove the transform, Wiener, or joint
warmup capabilities.

The remaining default-feature named modules are retained deliberately rather than
for pre-alpha compatibility. They either own types embedded in cross-crate public
signatures, supply a validated/reference workflow, or are the explicit opt-in path
to an implemented research kernel. The latter remain governed by S1 promotion
gates and are not part of the `reference` surface. State-space and particle
contracts remain pre-release design inputs for M14/M12 and are not candidates for
freezing in their current form.

## Completed boundary audit (2026-10-20)

M13 is complete as a pre-release boundary milestone. The manifest now records
5,990 rendered default-feature entries and two separately reviewed Faer surfaces
(math: 525; MCMC: 3,068). Five additional feature variants are checked as
API-neutral: three SIMD surfaces, distribution branch hints, and the generic
standard-autodiff adapter. Research features retain their adoption gates.

The schema-v3 maturity ledger individually classifies all 4,729 distinct rendered
declarations/implementations in the union, including aliases and feature-only
fitting methods. Repeated identical rustdoc lines are deduplicated, accounting for
the difference from the rendered counts. Mixed adaptation/integration owners are
refined explicitly: covariance/window/dual-averaging and the sealed reference
integrators remain S2; Fisher, joint/manifold learning, alternative kernels and
advanced selection components retain S1 gates. Source modules, every explicit
Cargo feature, all root items, named facades and CLI binaries are also enforced.

`api_compatibility.py` distinguishes additions from potentially breaking removals,
signature replacements, member and variant additions. Its accepted review records
both endpoint hashes and the actual added/removed signatures, so shallow checkouts
can validate the decision. The explicit decision is a pre-release break, covering
root cleanup, evolving records, removed implementation aliases and the renamed
`riemannian::ExactnessPendingRmhmc`. This is conservative adjudication rather than
a claim of complete Rust-language semver analysis. Every delta remains review-bound.

The RMHMC numerical map is unchanged. Its only executable type name now carries
the unresolved exactness obligation; no compatibility alias or reference facade
restores `Rmhmc`. Its S1 law entries retain the open reversal/volume gates.
Standalone phase steps, both sealed integrators and batched advancement now have
separate proposal-only laws: none imply a corrected invariant Markov transition.
Every MCMC feature has a law record, and source discovery rejects a feature-only
MarkovChain missing from the transition inventory.

The reference-owner audit covers validated construction, discarded-result
annotations, protected evolving records, supported common traits, error sources
and fallible-operation laws. No other named research owner is selected for stable
release by M13. The downstream positive laws, seven compiler-negative cases,
API/compatibility mutation tests and pinned extraction run are the executable
boundary evidence. M14's new normalized distribution/modeling owners are named-only
validated opt-ins; their separate protocol does not promote any research kernel.

S3/S4 graduation, full release-surface missing-doc enforcement and general
language-level semver tooling remain M17 gates. S1 research items remain accessible
through named modules; exactness auditing and milestone completion are not algorithm
promotion or stability promises.

## M15 session additions

M15 deliberately adds the pre-release S2 session facade and named session/storage
contracts, plus lossless frozen-state access and cancellable covariance warmup.
The runtime now directly depends on core's target/error contracts to support both
gradient models and density-only RWMH; no numerical algorithm moves into runtime.
The reviewed API delta, maturity inventory and snapshots include these additions.
Historical counts above remain the corresponding M13 audit steps; current counts
and hashes are authoritative in the snapshot manifest. No research sampler or
S3/S4 release promotion follows. See [session semantics](m15-sessions.md).

## M16 experiment command

M16 adds the opt-in S2 `alea-bench` command to the command inventory. It uses
private worker protocols and a versioned experiment library under `tools/bench/`;
the eight Rust library API snapshots and their existing compatibility review are
unchanged. Its contract/fault gates are documented in [m16-experiments.md](m16-experiments.md).
