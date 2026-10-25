# Workspace and evidence maintenance

Keep source, tests, licenses, reports, source identities and checksums in Git.
Large immutable benchmark archives belong in a separate evidence store. They are
historical results, including failed and interrupted runs, rather than disposable
build output. The [payload catalog](benchmarks/payloads.json) records their exact
original paths, sizes and SHA-256 digests. Existing bundle manifests and reports
are unchanged. A source checkout alone does not contain these payload bytes.
Git attributes preserve measured files and corpus/fixture line endings verbatim;
do not normalize their recorded CSV or Stan bytes during formatting cleanup.

## Evidence storage and restoration

Before removing a local payload, copy it to an external store and verify it:

```sh
python3 tools/evidence.py archive --store /path/to/alea-evidence
python3 tools/evidence.py verify --store /path/to/alea-evidence
```

Stores preserve the catalog's relative paths. Back up the store and catalog
together to durable storage. A directory on the same disk is separation from the
checkout, not an independent backup. No public download endpoint is configured;
obtain the catalog-matching store from the maintainer. Never substitute a new
benchmark execution for an archived result.

Restore all payloads, or select one campaign, into their original ignored paths:

```sh
python3 tools/evidence.py restore --store /path/to/alea-evidence
python3 tools/evidence.py restore --store /path/to/alea-evidence --prefix docs/benchmarks/walltime-parity-2026-10-23/
python3 tools/evidence.py verify
```

The commands refuse checksum mismatches, symlinks and overwriting different
existing evidence. They never delete files. Restoring is sufficient for existing
bundle unpack commands and historical readers; no report rewrite is needed.
`catalog` checks the index structure only; `verify` reads every selected payload.

Routine CI runs synthetic benchmark/parser contracts and validates the catalog.
After restoring the historical v1/v2 payloads, run their complete independent
reader checks explicitly:

```sh
python3 tools/test_m16.py
python3 tools/test_m16_pdb.py
# Or include both in the full stable gate:
ALEA_VERIFY_ARCHIVED_EVIDENCE=1 nix develop .#stable --command bash tools/ci/check.sh
```

Historical reports and verification receipts attest to their recorded source,
not to the latest worktree. Candidate validation must use the candidate revision.

## Build and scratch retention

Use the normal `crates/target` cache for active development. Put experiment outputs
under root `target`, with a named campaign and a declared retention destination.
Avoid a new Cargo target directory for every investigation. For one-off CI and
profiling builds, `CARGO_INCREMENTAL=0` avoids retaining incremental caches;
leave incremental compilation available for normal edit/build cycles.

Before cleaning, check that no relevant builds or benchmarks are running. Cargo
target caches can be rebuilt; root `target` may also contain unique experiment
records and must be inspected before removal. Use `cargo clean --manifest-path
crates/Cargo.toml` for a complete workspace rebuild, or remove a specifically
identified obsolete cache. Never use a blanket `git clean -fdx` in a research
checkout. Seal source, raw results and failures before retiring an experiment.

Agent handoffs live in ignored `.agent/`; archive old scratch outside the checkout
and retain the current navigation files. Personal agent configuration belongs in
local Git excludes. Do not include agent artifacts in source commits or releases.

## Next development sequence

The next milestone is completion of **M17's reference release-candidate cycle**.
The performance work supports that effort; it does not complete platform or
compatibility validation. Approximate walltime parity was measured for an explicit
gradient-diagonal, jittered, position-turning NUTS profile. Constructor defaults
and research maturity remain as documented.

Suggested order, with indicative working-day budgets rather than release promises:

1. Consolidation: archive payloads, retire build caches, validate and push the
   accumulated baseline changes. Keep the checkout and source release compact.
2. Next 1–3 working days: obtain native Linux/macOS/Windows and architecture CI
   results for the pushed revision; fix portability and fresh-install failures.
   Use the existing release workflow and retain successful and failed logs.
3. Following 3–7 working days: exercise one external downstream workflow, including
   model construction, warmup, diagnostics, cancellation and checkpoint recovery.
   Document the explicit fast NUTS configuration and its diagnostic tradeoffs.
4. After those results: review the exact reference API/schema compatibility scope,
   seal and sign a candidate, and start its S3 feedback/soak cycle. Release timing
   depends on resolving correctness, data-loss and compatibility defects.

Keep RMHMC exactness, optional samplers and Enzyme compiler qualification on their
own schedules. Add targeted seeds/surgical or cheap-target performance experiments
when they answer a concrete regression question; a new full PosteriorDB run is
not the immediate prerequisite for release consolidation. See
[the release policy](releases.md) for the actual graduation requirements.
