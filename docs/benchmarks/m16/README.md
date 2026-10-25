# M16 evidence and reproduction

## Descriptive PosteriorDB extension (v2)

The completed dusk run covers 20 posteriors, 26 baseline/research/upstream
configurations, three repetitions and four chains per job: **1,560 jobs**.
There are 1,267 completed executions, 260 failures and 33 unsupported combinations.
Execution status is separate from diagnostics and empirical fidelity; some
completed runs have severe discrepancies. No promotion score or quality eligibility
filter is computed.

Start with [findings](posteriordb-v2/findings.md) or the
[interactive report](posteriordb-v2/report.html). The complete immutable
[bundle](posteriordb-v2/README.md) includes 28 checksummed raw-evidence segments,
full reports, source/build seals, failure records and phase/timing tables. Native
JAX and BridgeStan callback timings are labelled separately. RMHMC remains
exactness-pending, MCLMC approximate, and ATLAS/Pathfinder composed workflows.

Both current and archived source readers reproduce the report and supplemental
analysis exactly. All 35 worker contracts and the original v1 reader pass. Check
the complete v2 evidence offline with:

```sh
python3 tools/test_m16_pdb.py
```

This scans all raw traces and can take several minutes. To restore the evidence
without rerunning samplers:

```sh
python3 tools/bench/alea-bench pdb unpack docs/benchmarks/m16/posteriordb-v2 target/m16-pdb-review
```

The [method and accounting guide](../../m16-posteriordb.md) describes budgets,
backend differences and reproduction. The v1 artifacts and their historical
decision below remain unchanged.

## Historical reference experiment (v1)

The predeclared `m16-reference-v1` protocol has 252 jobs: 13 corpus cases × four
engines × three timing repetitions, plus 32 independently generated datasets for
each of three Alea SBC/predictive verticals. Each job has four logical chains.
The checked report records every failure and scientific quality outcome.

The final run accounts for 249 completed jobs and three nuts-rs initialization
failures on centered eight schools. All 96 SBC/predictive jobs pass their sealed
gates. There are 216 quality passes and 36 failures overall: the ill-conditioned
Gaussian, funnel and centered eight-schools cases fail across engines. Held-out
cross-engine mean agreement passes, but diagnostics and analytic second moments
prevent promotion. The generated decision is **not eligible**; failure rates on
held-out cases are 20% for each engine under this fixed policy. These results do
not establish that the algorithms fail under a suitably tuned policy.

The compact review files are `manifest.json`, `report.json`, `report.md` and
`tracking.json`. `reference-v1.tar.gz` contains the complete bundle, including
compressed raw and native Stan draws, preparation/process logs, diagnostics,
source/preparation archives, plan/index, host details and result records. Engine
binaries and compiler caches are rebuildable inputs, rather than evidence stored
only in an ignored directory.

`verification.json` records archive/log checksums and successful pinned-stable,
standalone-wrapper and pinned-nightly API gates. Nineteen offline contract/fault
tests pass; library API extraction remains at 6,356 entries without changes.

To regenerate the complete report without Rust, JAX, Stan or network access:

```sh
nix develop .#stable
mkdir target/m16-review
tar -xzf docs/benchmarks/m16/reference-v1.tar.gz -C target/m16-review
python3 tools/bench/alea-bench verify target/m16-review/reference-v1
python3 tools/bench/alea-bench report target/m16-review/reference-v1 --output target/m16-review/regenerated.json
diff docs/benchmarks/m16/report.json target/m16-review/regenerated.json
python3 tools/test_m16.py
```

The last check also runs the Python implementation inside the archived source
snapshot as an independent reader. Correctness CI validates this evidence and
fault tests without imposing elapsed-time thresholds on CI hardware.

The current reader additionally rejects boolean schema-version aliases; this
post-measurement input hardening changes no valid record, threshold or result.
Both current and archived readers regenerate the same complete report.

To rerun all numerical jobs from the single manifest when source bytes match:

```sh
python3 tools/bench/alea-bench reproduce docs/benchmarks/m16/manifest.json target/m16-rerun
```

For an advanced worktree, extract `sources.tar.gz` from the bundle into a new
directory **outside an existing Git worktree**, then run the archived
`tools/bench/alea-bench reproduce ABSOLUTE_MANIFEST_PATH NEW_DIRECTORY` there in its
Nix stable shell. That source snapshot includes all benchmark code, compiled
native inputs, Cargo locks, Nix pins and the sealed ledger. Pinned upstream
nuts-rs is fetched separately by `prepare`. Builds happen outside job timings.
See [method, accounting and reproduction limits](../../m16-experiments.md).

The measurement is from one physical host. `track` accepts multiple complete
bundles and keeps hosts separate; its uncertainty bands describe timing
repetitions with fixed numerical seeds. nuts-rs has explicitly unmatched native
diagonal geometry, so its timings do not qualify matched-geometry promotion.
No research/default or S3/S4 promotion follows from milestone completion.

The initial incomplete attempt remains in `attempt-01-incomplete.tar.gz`, with
its original manifest and controller log alongside it. The analyzer rejected a
BlackJAX acceptance probability `1.0000000000000002`. It was interrupted before
finalization; its missing final index makes it ineligible for a complete report.
The final attempt permits only a bounded eight-epsilon acceptance roundoff
envelope, preserves original raw values, and leaves every scientific threshold,
seed, step and model datum unchanged. No measured result was re-sealed.
