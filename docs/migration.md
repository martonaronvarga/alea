# Pre-alpha migration and future deprecations

All current crates are unpublished and pre-alpha. The
[earlier candidate review](releases/2026-10-25-reference-candidate/README.md)
is evidence for its frozen source, not a promise that later pre-alpha APIs remain
compatible. Pin source revisions and retain the lockfile when upgrading.

The `reference` modules have been removed. Use normal module paths, such as
`alea_core::target::LogDensityGradient`, `alea_mcmc::Nuts` and
`alea_runtime::session::SessionBuilder`, or import a crate’s curated `prelude::*`.
There are no compatibility aliases. Experimental samplers remain in their named
modules and are marked in Rustdoc; a prelude is only an import convenience.

The M13 boundary intentionally removed ambiguous root/research/error aliases.
Import each error from its owning API. Use the explicitly named
`riemannian::ExactnessPendingRmhmc` only with its documented open exactness gates;
there is no compatibility alias. Approximate MCLMC requires explicit bias
acknowledgement and cannot be substituted into an exact-target interface.

M14 models distinguish constrained parameters, unconstrained target coordinates
and predictions. Build a named layout and apply its Jacobian once. Ordinary
log-density factors do not advertise normalized sampling capabilities.

M15 sessions use logical chain IDs and caller-owned execution pools. Durable draw
streams use schema1; current checkpoints use `ALEACP02` and implementation
`session/2`, preserving NUTS jitter and turning. V1 checkpoints explicitly reject
with `CheckpointError::Format`; retain their draws and start a new session from
the original model/configuration. Restart only
frozen checkpoints with the same model/build/backend/seed identity. Do not rename
an incompatible checkpoint or edit its provenance to force admission. Keep old
data and use the original pinned reader when no migration reader is available.

Future stable deprecations and compatibility windows follow the
[release policy](releases.md). Every candidate must include release notes with
the actual API/feature/schema diff and runnable before/after examples for breaking
changes. No deprecation window is retroactively promised for pre-alpha APIs.

Native NUTS warmup changes seeded trajectories. A session selecting NUTS now
uses NUTS transitions during adaptation instead of eight-step HMC. Starting with
the same seed and initial position therefore gives different warmup RNG
consumption, fitted geometry, step size and retained draws. Reproduce historical
results with their archived source and protocol. That warmup change did not
alter the frozen checkpoint container; the subsequent candidate-cycle repair
deliberately advances it to v2 to fix incomplete policy replay. Restart restores
the frozen metric, step, policy and RNG state. Adaptation state is not persisted;
no mid-warmup migration is supported. Numerical identity remains `numerics/3`,
and the separate owned-record provenance schema remains1.
