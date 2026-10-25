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

The first **M17 reference candidate cycle is complete**. The
[candidate review](releases/2026-10-25-reference-candidate/README.md) binds successful
native CI, the independent downstream workflow, calibration/recovery, compatibility
checks and a signed reproducible source archive to the exact code revision. That review applies to the frozen source. Current APIs use ordinary module paths
and convenience preludes; constructor defaults and research maturity
retain their documented classifications; all crates remain unpublished.

The next work is field feedback and per-crate stable-release preparation:

1. Use the candidate in an external application and record the user's installation,
   modeling, diagnostics and recovery experience. The completed independent
   consumer was agent-executed; external-human feedback is still outstanding.
2. Maintain the frozen candidate baseline, run the existing scheduled soak and
   native gates, and diagnose concrete regressions. Correctness, data-loss,
   unsoundness or schema defects reset the affected cycle after repair.
3. Before S4, settle each crate's compatibility scope for existing research exports,
   confirm its support/deprecation commitments, and review actual publication.
   Do not lift `publish = false` or infer stable status from S3 alone.

There is no fixed publication date. Stable graduation depends on that evidence
and an explicit per-crate publication decision. Keep candidate artifacts outside
the checkout and back them up with their signed digest and evidence manifest.

Keep RMHMC exactness, optional samplers and Enzyme compiler qualification on their
own schedules. Add targeted seeds/surgical or cheap-target performance experiments
when they answer a concrete regression question; a new full PosteriorDB run is
not the immediate prerequisite for release consolidation. See
[the release policy](releases.md) for the actual graduation requirements.
