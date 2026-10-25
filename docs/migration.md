# Pre-alpha migration and future deprecations

All current crates are unpublished and below S3. Pin source revisions and retain
the lockfile when upgrading. The complete reviewed API changes are in
`tools/ci/api-review.json`; the inventory is not a complete Rust semver proof.

Use the explicit `alea_core::reference`, `alea_math::reference`,
`alea_mcmc::reference` and `alea_runtime::reference` entry points for the baseline.
Construct validated options, warm up explicitly, then consume the warmup result
to obtain a frozen retained chain. For end-to-end execution use `SessionBuilder`;
do not reproduce seed allocation or checkpoint parsing in application code.

The M13 boundary intentionally removed ambiguous root/research/error aliases.
Import each error from its owning API. Use the explicitly named
`riemannian::ExactnessPendingRmhmc` only with its documented open exactness gates;
there is no compatibility alias. Approximate MCLMC requires explicit bias
acknowledgement and cannot be substituted into an exact-target interface.

M14 models distinguish constrained parameters, unconstrained target coordinates
and predictions. Build a named layout and apply its Jacobian once. Ordinary
log-density factors do not advertise normalized sampling capabilities.

M15 sessions use logical chain IDs and caller-owned execution pools. Durable draw
streams/checkpoints use schema v1 and typed incompatibility errors. Restart only
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
results with their archived source and protocol. Frozen schema-v1 checkpoints
remain unchanged: adaptation state is not persisted, and restart still restores
the frozen metric, step and RNG state. No mid-warmup migration is supported.
