# Native Alea / CmdStan benchmark

Production models use ModelBuilder, native normal priors and the native fused
normal/Bernoulli-logit regression factors. The target observer only counts and
times calls; it forwards values, gradients, typed errors and numerical-error
classification. It contains no distribution, derivative or sampler algorithm.
Independent equations in correctness.rs are test oracles and are never timed.

The original archive is reproduced here with a local Alea path dependency and
Cargo.lock. PINNED.json identifies the baseline commit, Rust Nixpkgs input and
installed CmdStan 2.40.0 package/source tree. Each timing manifest seals the
current commit **and working-tree patch**, every Rust/TOML/Python/Stan source,
binary, lockfile, data, CPU, compiler, flags, affinity and thread counts. A
working-tree run must retain those sources alongside its manifest; HEAD alone
does not identify it. The installed CmdStan Nixpkgs tree has no Git revision;
its exact NAR hash is recorded rather than inventing a revision.

| Case | Observations | Free coordinates |
| --- | ---: | ---: |
| mean | 1,000 | mu, log sigma (2) |
| linear | 1,000 | 8 coefficients, log sigma (9) |
| logistic | 1,000 | 8 coefficients |
| linear_large | 10,000 | 8 coefficients, log sigma (9) |
| logistic_large | 10,000 | 8 coefficients |
| funnel | — | v and 8 z coordinates |

Gaussian coefficients/location have N(0,2) priors and positive sigma has an
N(0,1) density restricted to sigma>0. Both engines omit the half-normal ln(2)
constant, so they define the same posterior. sigma=exp(q_sigma) adds q_sigma
once to the density and 1 to its score. Both designs are row-major inputs with
no implicit intercept/standardization/offset; Stan's JSON matrix preserves row
order. All free coordinates start at zero. The older fixed-sigma Gaussian
benchmark is a different posterior and is not a native-model timing baseline.

## Build and correctness checks

From the repository root, enter `nix develop .#stable`. Use the exact CmdStan
package in PINNED.json (or record a newly pinned package as a different cohort).

```sh
cargo build --release --locked --manifest-path tools/native-benchmark/Cargo.toml
python3 tools/native-benchmark/scripts/gen_data.py
make -C /nix/store/zdb9gq6d00i09k3dsdd1flab6mmgswsb-cmdstan-2.40.0/opt/cmdstan \
  CXX=/nix/store/5q6bdxkp7lc70gn9mnf4yiybrx8ry5xq-gcc-16.2.0/bin/g++ \
  PRECOMPILED_HEADERS=false \
  "$PWD/tools/native-benchmark/stan/mean" \
  "$PWD/tools/native-benchmark/stan/linear" \
  "$PWD/tools/native-benchmark/stan/logistic" \
  "$PWD/tools/native-benchmark/stan/funnel"
cargo test --locked --manifest-path tools/native-benchmark/Cargo.toml
python3 -m unittest discover -s tools/native-benchmark/scripts -p 'test_*.py'
python3 tools/native-benchmark/scripts/check_stan.py --output /tmp/native-density-check
```

`check_stan.py` compares densities modulo constants and all scores at 22 shared
free-coordinate points, including log sigma=-20 and logistic coefficients ±1000.
Rust tests separately check independent equations, central differences, extreme
failures and the original second-warmup-transition underflow. Core regression
tests additionally compare scalar, chunked and batch factors and allocation reuse.

The executed Stan builds used GCC 16.2.0 (confirmed by ELF .comment), while the
Rust shell's default C++ compiler is GCC 15.2.0. The explicit CXX above preserves
that distinction; the runner records ELF compiler comments separately from the
runtime PATH compiler. Do not infer the build compiler from a runtime version.

## Measurement

`scripts/kernel_compare.py --suite tensor` declares ten fixed-work shapes for
elementwise tensor factors, scalar-tape controls, matrix-vector products, both
lazy branch directions and scans. Use identical `kernel_audit.rs` harnesses for
saved before/after builds. All tensor controls require exact checksums; retain
failed/missing work and control regressions. See the
[tensor execution-plan report](../../docs/tensor-ir.md#execution-plan-follow-up-immutable-bindings-and-active-adjoints).

`--suite tensor-checkpoint` retains those ten controls and adds 1,025/10,000-trip
retained and opt-in checkpointed scans. Timing regressions are evidence, not a
reason to drop cases; checkpointing trades replay work for reverse storage and
is not the ordinary scan default. Historical baseline builds need the identical
harness and an explicitly archived API-only shim forwarding `checkpointed_scan`
to retained `scan`; never substitute a new distribution or gradient. See the
[checkpoint/kernel follow-up](../../docs/tensor-ir.md#checkpointed-scans-and-matrix-vector-kernels).

`--suite tensor-fusion` retains all fourteen checkpoint controls and adds shared
18-map pointwise DAGs at 1,000/10,000/100,000 elements. It requires identical
harnesses and exact checksums, not only successful exits. Numerical equivalence
is tested independently of these timing checksums. Preserve every control,
regression and failed/missing job; do not rank only eligible fusion cases. See
[fusion and source replay](../../docs/tensor-ir.md#dependency-safe-fusion-and-first-order-source-replay).

`scripts/kernel_compare.py --suite sufficient-statistics` compares common-
predictor Poisson batches with distinct-predictor Auto/Scalar controls at 1,000
and 10,000 observations. Build saved before/after executables with identical
`kernel_audit.rs` harnesses, then supply `--before`, `--after`, `--output`, and
`--reps 7`. Construction is separate; failed/partial jobs remain in the journal.
The [paper application case study](../../docs/native-model-performance.md#14-applying-the-sufficient-statistics-paper)
distinguishes the shipping common-predictor reduction from the rejected general
partial-statistics experiment, including the small-control performance regret.

The current beyond-Gaussian increment and its remaining backend deficits are
documented in [the case study](../../docs/native-model-performance.md#13-beyond-gaussian-evaluation-kernels-and-tape-overhead).
`kernel_compare.py --suite backends` adds explicit Auto/Scalar Poisson,
scalar-Normal-factor and tensor-factor controls. When comparing historical
production sources, build both with the identical `kernel_audit.rs` harness;
record the overlay separately rather than claiming an untouched historical
binary supported new cases. These controls do not replace full model-gradient
tests or establish tensor/compiler/native-backend performance parity.

### Targeted regression optimization comparisons

`scripts/regression_compare.py run` pairs saved native executables against a
verified baseline manifest and the original data. It alternates process order,
consumes every gradient coordinate, defaults to exact score receipts and checks
densities at the existing tolerance. It saves a source capsule, immutable input
copies, raw outcomes and a replayable summary; `report` verifies/rederives it.
`--score-policy numerical` explicitly selects the existing three-engine 3e-9
density/score tolerances for algorithmic reassociation. Its
`sample --input PAIRED_CAPSULE --reps 7` mode runs separately labelled four-chain
timing repetitions with the original seeds; repetitions are not extra independent
chains. `scripts/gaussian_reference.py` rechecks a saved paired candidate against
sealed Stan fused-GLM/JAX workers, retaining raw outcomes and hash-verified replay.
See the
[profile-driven results](../../docs/native-model-performance.md#11-profile-driven-regression-kernel-improvements-2026-11-02)
for the current bounded-SIMD logarithm's accuracy contract and remaining gaps.
The [Gaussian compression case study](../../docs/native-model-performance.md#12-removing-the-remaining-gaussian-observation-scan-2026-11-02)
records O(P²) evaluation, preparation costs, numerical fallback and fresh references.

### Current Alea / Stan / JAX comparison

`scripts/three_engine.py` provides a separate, source-sealed CPU/float64 campaign.
It keeps all six original datasets/posteriors and adds Stan's existing fused
`normal_id_glm`/`bernoulli_logit_glm` kernels as **separate** references, never a
silent replacement of the original vectorized Stan models. Alea remains native
ModelBuilder/distribution assembly. JAX uses native elementary/statistical
functions, automatic differentiation and BlackJAX NUTS; no sampler is rewritten.
Use `tools/blackjax-reference/default.nix` for pinned JAX/jaxlib0.9.2/BlackJAX1.5.

`prepare` requires explicit `--cmdstan`, `--python`, `--cxx`, `--output` paths and
builds/copies executables before measurement. It seals the actual source archive,
data, compiled C++ dependencies, compiler identities and frozen worker scripts.
Rust's recorded build duration can be incremental; it is not a clean-build/JIT
cost comparison. `run` uses those frozen inputs on one selected CPU, rotates
engine order and retains raw stdout/stderr/CSV and partial append-only outcomes.
`report` verifies input/output hashes and rederives metrics from raw child output;
write reconstructed summaries to a new file, not over historical evidence.
`--phases evaluate` or `--phases sampling` can seal a single-phase campaign;
the default measures both. Reusing a build still requires exact compiled-input
and toolchain matches. A watchdog enforces child deadlines while the parent uses
blocking `wait()`; POSIX `wait(timeout=...)` polls in up-to-50-ms sleeps and must
not be used to measure short process latency, particularly four-process Stan.

Before timing admission, full CmdStan and JAX density/score probes check shared
free coordinates, small sigma and extreme logits; Stan GLM probes and all-score
fixed-work checks are additional controls. Fixed-work mode performs 10,000 full
posterior value/gradient calls at ten changing positions. All score coordinates
contribute to a returned checksum. JAX receives data, count, and the complete
position matrix as runtime inputs; its compiled loop retains AD work for every
iteration. Passing only data and constructing zero coefficients inside the loop
would let XLA specialize away work. Passing a base position and updating only one
coordinate would still permit loop-invariant hoisting. A runtime-position check
changes every coordinate and compares the loop with independent AD calls.
JIT compilation and the
additional Python host-dispatch/completion latency test are separate from backend
loop time. This is a CPU backend-throughput comparison, not a GPU or Python-call
latency comparison. XLA fast math and Eigen threading are explicitly disabled.

NUTS uses four sequential chains, 1,000 warmup/1,000 retained draws, diagonal
adaptation, zero free initialization, .2 initial step and .8 target. Native engine
adaptation policies/RNGs remain different. Full fresh-process wall time includes
JAX imports/JIT and CSV; JAX additionally reports synchronized warmup/sampling,
compilation and serialization separately. All engines retain draw CSV, but
CmdStan writes richer statistics and uses four processes; Alea/JAX use one.
Alea's process also performs its built-in diagnostic calculation, whereas the
common external diagnostic readers for Stan/JAX are outside their measured
process walls. The fixed-work Stan reference is a direct Stan Math reverse-AD
client, not a generated model's optimized tape; actual CmdStan executables are
used for inference. These scopes must not be conflated.
Alea's existing diagnostic implementation evaluates every engine's coordinates.
JAX gradient counts remain unavailable rather than inferred from leapfrog steps.
Process ESS/sec comparisons require both engines' Rhat<=1.01, bulk/tailESS>=100,
no retained divergences and means within six combined MCSE for every coordinate.
Repeated seeds support timing spread, not independent extra posterior chains.
The funnel stays in the denominator as a negative control; failed quality gates
withhold comparisons, never turn a short failed run into a speedup.

For native CPU kernel selection rather than inference, use `scripts/cpu_plan.py`.
It runs the existing public-descriptor Criterion benchmark with independent
process repetitions, alternating scalar/Auto order, source/binary archives,
disjoint calibration/validation shapes and disk-replayable partial outcomes.
It reports evaluation throughput, selection regret and construction amortization.
Current construction receipts explicitly label repeated binding to shared
prepared data (`shared-prepared-data-v2`). They exclude fresh input allocation
and packing. The comparison reader rejects older private-pack construction
cohorts against this scope; historical raw reports remain unchanged.
It does not implement probabilities, fit thresholds or change production dispatch.
See [CPU-plan campaign methodology](../../docs/kernel-optimization.md#repeated-source-sealed-candidate-campaigns).

Compile before measurement and keep executables immutable during a run. Run
experiments serially on one permitted CPU with no concurrent builds/profilers.
Use a fresh output directory for each cohort; prior outcomes are never replaced.

```sh
python3 tools/native-benchmark/scripts/bench.py \
  --cmdstan /nix/store/zdb9gq6d00i09k3dsdd1flab6mmgswsb-cmdstan-2.40.0/opt/cmdstan \
  --results /tmp/native-four-chain --chains 4 --reps 7 --cpu 2 --timed
```

Seven repeats estimate timing spread; the four chains, rather than timing
repeats of identical seeded chains, support R-hat/ESS diagnostics. Both engines
use 1000 warmup/1000 retained draws, target .8, diagonal covariance adaptation,
step .2, depth 10, no jitter and zero free initial values. Alea explicitly
disables reasonable-step search to reproduce the original workflow. Stan has
its own initial search and adaptation; these are not identical policies.

JSONL records, stdout/stderr, Stan draw/profile CSVs, manifests and a summary
survive failures. Summaries retain attempted, failed and unstarted jobs. Failed
runs never enter speed ratios. Mean/max tree depth, retained leapfrogs and
divergences, final Alea step sizes, actual evaluation counts and evaluation time
are distinct fields. Stan's posterior profile counts autodiff calls; its time
excludes transforms/Jacobians outside the block and may include primal calls.
Alea's observer covers the entire transformed target and costs a clock read
and uncontended mutex per call. No observer overhead is subtracted.

ESS/sec uses the minimum bulk ESS across every parameter divided by the whole
fresh-process time, including all sequential chains. The same Alea modern
diagnostic implementation analyzes both engines. Single-chain diagnostics are
explicitly unavailable. Completion alone does not establish convergence.
ESS/sec is eligible only for complete multi-chain runs with finite diagnostics
for every coordinate, R-hat at most 1.01, bulk and tail ESS at least 100, and zero
retained divergences. Missing diagnostics and failed gates retain explicit reasons
and computational timings, but no eligible ESS/sec. Paired ESS/sec comparisons
require both engines to qualify. Computational work/timing ratios are labelled
separately and do not establish equivalent statistical efficiency.
CmdStan chains use ids 1..C and seed 17; Alea uses ChaCha12 streams 0..C-1 and
the byte seed [17;32]. Different engines do not share random trajectories.

### Interrupted runs and journal replay

The paired and kernel runners write append-only `records.jsonl` and
`validation.jsonl` journals. Pending or incompatible pairs never enter paired
timing ratios. Kernel timings require pair validation; individual completed-run
metrics remain in the raw records. Summaries include planned/unstarted attempts
and are refreshed after every attempt and compatibility decision, including before
an incompatible-pair exit. A crash can leave the derived summary behind the journal.
Rebuild it to a **new** file (never overwrite sealed historical results):

```sh
python3 tools/native-benchmark/scripts/replay.py kernel /tmp/kernel-run /tmp/kernel-rederived.json
python3 tools/native-benchmark/scripts/replay.py paired /tmp/native-pair /tmp/pair-rederived.json
python3 tools/native-benchmark/scripts/replay.py stan /tmp/native-four-chain /tmp/stan-rederived.json
```

Only a torn final JSONL record is ignored; malformed complete records, duplicate
identities and non-finite child metrics are rejected. Historical raw records stay
unchanged; a rederived summary applies the current qualification policy, not a
retroactive replacement of the original report. Kernel journals without recorded
compatibility decisions stay ineligible. CI runs correctness/protocol tests, not
hardware-specific timing thresholds or a CmdStan-dependent measurement campaign.

## Attribution and regression harness

```sh
ALEA_BENCH_EVALUATIONS=10000 tools/native-benchmark/target/release/alea-stan-bench linear tools/native-benchmark/data/linear.txt
ALEA_BENCH_KERNEL=1 tools/native-benchmark/target/release/alea-stan-bench linear tools/native-benchmark/data/linear_large.txt
python3 tools/native-benchmark/scripts/compare.py --baseline /path/to/baseline \
  --current /path/to/current --output /tmp/native-paired --reps 7 --cpu 2
python3 tools/native-benchmark/scripts/reproduce.py --baseline /path/to/baseline \
  --current /path/to/current --output /tmp/native-sizes
```

Evaluation mode performs fixed native calls without the sampler. Kernel mode
uses public NutsWarmup/run_observed and the same seed/stream and policy as the
session: it reports separate warmup/sampling seconds and evaluations, the full
warmup adaptation report and final point. It is an attribution control, not a
replacement sampler. Pair comparisons enforce identical trajectory hashes,
evaluation/leapfrog counts and divergences on jointly successful runs; fixed
evaluation comparisons enforce checksums. Hardware-dependent timing thresholds
are descriptive, not correctness gates. Baseline builds use the same harness
with `--no-default-features` to omit the new classifier forwarding method.

For a calibrated, isolated performance job, `compare.py --max-slowdown 1.1`
emits an alert and exits nonzero if a jointly successful median pair exceeds
that slowdown. The limit is opt-in and recorded before measurement; failure
times are never used and hardware timing alerts remain separate from correctness
CI. Unit tests verify the threshold and failed-fit exclusion.

When an accuracy-preserving kernel deliberately changes rounding, opt in to
`--allow-trajectory-change`: comparisons record each changed hash/work field,
retain fixed evaluation counts and validate finite checksums within a declared
tolerance. Default replay checking remains exact. Use `--chains 4 --modes sampling`
for paired before/after multi-chain ESS/sec, and `--cases` to predeclare a subset.
Never treat a changed trajectory or a single chain as equal statistical work.

Construction is separately timed. Kernel phase times exclude session/storage
and diagnostics. Optional `--alea-csv` measures a separate serialization cohort;
serialization seconds are reported. Stan always writes its richer CSV, while
Alea ordinarily collects/discards in memory. Alea multi-chain diagnostics run
inside its process; Stan CSV diagnostics run outside measured processes. Stan
also starts one process per chain while Alea executes all chains sequentially
in one process. These asymmetries remain explicit; a wall-time ratio is not a
pure kernel or statistical-efficiency comparison.

See REPORT.md for executed results, numerical limits and external raw evidence.
The subsequent shared-scale/blocked-kernel implementation and preventive review
checklist are in [the native-kernel case study](../../docs/native-model-performance.md).

The broader [kernel inventory and optimization guide](../../docs/kernel-optimization.md)
covers the follow-up Poisson, norm, metric, mixture and Normal-score changes.
Run its fixed-work controls separately from sampling:

```sh
cargo build --release --manifest-path tools/native-benchmark/Cargo.toml --bins
python3 tools/native-benchmark/scripts/kernel_compare.py \
  --before /path/to/saved/kernel_audit \
  --after tools/native-benchmark/target/release/kernel_audit \
  --output /tmp/kernel-paired --reps 7 --cpu 2
```

Use the actual saved binary paths; the usual current binary is
`tools/native-benchmark/target/release/kernel_audit`. The harness records hashes,
construction and evaluation times, partial failures and checksum qualification.
Run without concurrent compilation, profiling or other CPU-heavy work. These
fixed-work gains are not interchangeable with inference throughput or ESS/sec.
Execute timing campaigns on Dusk or another stable remote server, not a laptop
whose AC/battery policy can change. Transfer the same sealed source, binaries
and runtime closure; record the measurement host, CPU affinity, governor, load
and thread settings. Never pool laptop exploratory timings with remote results.
Correctness tests and compilation can remain local and separate from measurement.

`kernel_compare.py --suite base` preserves the original 15-case default.
`--suite mixture-scores` selects ten event-score, parameter-score and native
mixture-factor controls; `--suite all` runs both sets. New mixture cases require
the updated runner in both saved builds and exact checksum agreement. The native
factor uses public `modeling::continuous` and `FiniteMixture<Normal>` and includes
per-evaluation construction. No benchmark-local probability or derivative is used.

## Fresh conformance and numerical-feature studies

The existing executable additionally accepts `schools`, `schools_nc` and
`covariance`. The school models reuse the retained PosteriorDB eight-schools
JSON and its centered/noncentered parameter ordering and Cauchy prior. They do
not create another corpus. Set `ALEA_SCHOOLS_DATA` to that existing JSON when
running `scripts/gen_data.py` from a source artifact without the local corpus;
missing external data is reported, never replaced. Original six-case defaults
remain unchanged. The covariance control is a separate 32-observation synthetic
Cholesky model, not a PosteriorDB posterior or an LKJ/Wishart benchmark.

`simd`, `faer` and `openblas` forward the corresponding facade features. The
optional `enzyme` feature generates a frozen covariance target from the very
same public native model in `src/conformance.rs`, using the existing source
exporter. Runtime `ALEA_BENCH_BACKEND=enzyme` selects its **evaluation-only**
protocol. Changed frozen data and other models reject explicitly. The school
prior uses the existing opaque Cauchy factor, whose source export is unsupported;
do not replace it with a Normal prior to fill a timing cell.

The maintained `scripts/three_engine.py` runner also supports:

```sh
# Builds/tests are separate from timing; use the pinned nightly/BLAS environment.
python3 scripts/three_engine.py matrix-build --output /new/native-builds
# Under the pinned autodiff environment, with LP64 BLAS available:
python3 scripts/three_engine.py matrix-build --enzyme --output /new/enzyme-builds
# Alternatively build the matched 16-way matrix in one autodiff environment:
python3 scripts/three_engine.py matrix-build \
  --feature-axes simd faer openblas enzyme --output /new/matched-builds
python3 scripts/three_engine.py matrix-stan-build \
  --cmdstan /pinned/cmdstan --cxx /pinned/g++ \
  --cases schools schools_nc covariance --output /new/stan-build
python3 scripts/three_engine.py matrix --matrix-plan /sealed/plan.json --output /new/results
python3 scripts/three_engine.py matrix-report --output /new/results
```

Run these commands from this benchmark directory. A matrix plan declares
`cases`, positive `reps`, `evaluations` (a positive multiple of ten), `cpu` and
`variants`. Each variant declares a unique `name`, saved `binary`, feature and
compiler receipts, and optional `env` and applicable `cases`. The reference
variant is named `stan`; Stan Math variants use `kind: "stan_math"` and optional
`implementation: "glm"`. Other variants use the existing native executable
protocol. Declare unsupported combinations before measurement.

The builder writes a pending cohort receipt first and only completes it after
the final source-freeze check, sealing binaries, test logs and input identities.
Per-variant success cannot override a pending/rejected cohort. Preserve the
`build_receipt` and `source_sha256` fields when constructing plans from its
`variants.json`; relocate the receipt path with the saved build directory.

The runner reuses child-process isolation and the existing JSON/compatibility
helpers. It checks all gradient coordinates, central finite differences and
changing-state work receipts before admitting timing ratios. It rotates variant
order, preserves every planned outcome, and captures binaries/data by digest.
Replay re-admits raw process/probe/loop outputs rather than trusting journal pass
flags. Captured inputs allow replay after moving a sealed capsule; executable
permission is not needed for replay. These are fixed-work measurements, not
statistical efficiency or proof of every support point.

For inference, the existing `scripts/bench.py` accepts
`--minimum-ess-per-chain 100 --both-coordinate-spaces --alea-csv --timed`.
The stronger policy is opt-in so historical reports retain their original
thresholds. Compare mapped posterior functionals and independent references
separately, and only rank ESS/sec after all required gates pass.

Use `kernel_compare.py --suite matrix-providers` with `--before-env
ALEA_MATRIX_KERNEL=compensated` and `--after-env ALEA_MATRIX_KERNEL=faer` (or
`blas`) to select actual tensor forward-and-VJP providers in one saved build.
The kernel audit checks every value and score against compensated execution
before timing. Provider scratch and changed reduction semantics remain explicit.
Enabling a provider feature alone does not select it; `simd` is not a switch
disabling all existing `pulp` runtime dispatch. Portable SIMD requires nightly.
Match compiler revisions/flags when attributing a feature effect; the general
nightly and Enzyme shells currently pin different compilers.

All new performance measurements run on ELTE HPC (`ssh eltehpc`) through the
existing `elte-vpn` connection, inside a Slurm compute allocation. Do not time on
the login node, Dusk, or the variable-power laptop. Local correctness builds and
tests are permitted. Do not overlap builds/tests/profiles with timing.

### Full configuration comparison on HPC

The `suite` action reuses the existing builders, numerical preflight, evaluation
runner, kernel controls and four-chain inference runner. It defaults to all nine
models, seven repetitions, both evaluation and sampling, and the full powerset
of `simd faer openblas enzyme branch-hints tracing analysis`, in separate stable
and pinned Enzyme-nightly compiler cohorts. Unsupported builds remain visible;
they do not acquire timings. This is deliberately a long run, not a smoke test.

On the remote source checkout, with the two Rust toolchains, CmdStan 2.40 and
LP64 OpenBLAS installed, submit the existing suite through the same CLI:

```sh
python3 scripts/three_engine.py slurm-submit --output /new/submission \
  --cpus 18 --walltime 120:00:00 --job-command \
  python3 scripts/three_engine.py suite --output /new/results \
  --cmdstan /installed/cmdstan-2.40.0 --cxx /usr/bin/g++ --native-cpu
```

Submission requests one exclusive node by default; it never changes the host's
governor, global thread settings or VPN configuration. `--dry-run` writes the job
without submitting. The worker chooses a CPU from its actual allocation rather
than assuming CPU 0 is available. Scheduler identity, affinity and virtualization
are recorded automatically. `--native-cpu` specializes both Rust and Stan builds
on the allocated compute node. Evaluation latency controls use one CPU. Provider
controls compare one thread against the full allocated CPU budget; Faer currently
remains sequential, while BLAS may use its bounded library threads. Four-chain
inference compares one worker with up to four workers in both engines. Library
threads stay at one during chain parallelism to avoid nested oversubscription.
An 18-CPU allocation alone does **not** establish multicore scaling.

`report.md` ranks full posterior evaluations separately from qualified inference
ESS/sec. `summary.json` includes all pairwise evaluation ratios, observed ranges,
failed and unstarted measurements. `progress.json` retains stage outcomes. Use
`suite-report --output /existing/results` to regenerate reports. Existing saved
builds can be supplied with `--builds /saved/variants.json`; compiler identities
are retained rather than inferred from the runtime shell.

Native Auto and explicitly scalar regression execution are distinct rows even
when compiled features match. Scalar controls select the row-kernel policy, not
disable Gaussian compression. Provider controls explicitly select compensated,
Faer or BLAS tensor GEMMs. Source/Enzyme rows currently cover the frozen covariance
worker only; source-backed inference is explicitly unsupported, never silently
measured as native inference. The broader PosteriorDB worker remains a separate
existing suite; these nine models are not every Alea model or AD capability.

Build evidence is shared by many execution configurations. The runner enumerates
the union of input paths before hashing, verifies each file once per admission
boundary, and reuses those decisions across workloads. The cache is local to
that admission pass: replay rereads the files, and completion checks the live
inputs again. `evaluation/setup.json` exposes setup progress separately from
model timings. This avoids cohort-sized file reads inside a configuration loop
without bypassing numerical preflight or accepting stale build evidence.
For independent profiling jobs, `slurm-submit --exclude-nodes NODE ...` passes an
explicit Slurm exclusion. Verify the actual allocated node; an environment-only
exclusion must not be assumed to have taken effect.

`kernel_compare.py --suite normal-batching` compares scalar-factor assembly with
the native vector Normal operation at small, tail and larger shapes. For a
same-binary representation comparison, pass `--before-env ALEA_NORMAL_BATCH=0`
and `--after-env ALEA_NORMAL_BATCH=1` with the same `kernel_audit` executable on
both sides. The density, parameters and all active event scores are unchanged;
the existing exact-checksum admission remains in effect. This measures model
representation, not a new sampler, general automatic plate fusion, or ESS/sec.

Build identity and input checks are automatic bookkeeping. Archive/capsule
packaging below is optional, not a prerequisite for running or reading results.

## Reusing PosteriorDB evidence and local sealed capsules

Do not create another model registry or inference runner to classify an existing
experiment. The existing PosteriorDB manifest/report contract can be replayed by
the same summary tool:

```sh
python3 tools/native-benchmark/scripts/replay.py conformance /path/to/existing-run /tmp/conformance.json
python3 tools/native-benchmark/scripts/replay.py pack /path/to/evidence-tree /tmp/local-capsule
python3 tools/native-benchmark/scripts/replay.py unpack /tmp/local-capsule /tmp/restored-capsule
```

All destinations must be new. Capsules are local only; no upload, download service
or second experiment database is introduced. Pack the original source seals,
workload data, manifests, raw draws, journals, validation and failures together.
The deterministic gzip stream is segmented into 32 MiB parts, with SHA-256 and
byte counts for every part and every file. Restoring verifies all segments before
creating output, rejects symlinks, traversal, duplicate/unsealed members and
overwrites, and verifies all restored file bytes. Members are limited to 1 GiB
and capsules to 999 segments. A checksum establishes integrity, not correctness
or authenticity; use the existing source-bound experiment reader after restore.
File contents, not ownership/permissions, are sealed: restored files are ordinary
data files. Mark only trusted saved binaries executable before running them.
Existing PosteriorDB bundles can be included unchanged, then opened with their
original loader. Historical manifests and qualification policies stay unchanged.

Conformance replay preserves every planned case/method/repetition, including
unsupported, failed and unstarted jobs. It distinguishes recorded target
preflight failures, runtime failures, unavailable evidence and diagnostic
failures. Qualification requires a recorded reference/corrected/exact method guarantee,
at least four chains, finite defined constrained **and** unconstrained diagnostics,
R-hat strictly below 1.01, bulk/tail ESS at least 100 per chain and no retained
divergences. Only qualified jobs receive bulk ESS/process-second. Recorded domain
rejections remain explicit; a failed diagnostic alone cannot identify a numerical
bug, an expressive limitation or poor mixing. Cross-backend pointwise preflight
does not prove global posterior equivalence. Reference-moment agreement and SBC
remain separate scientific requirements, not inferred passing flags.

The existing `kernel_compare.py --suite tensor-storage` adds vector-input and
alias-separated lifetime controls to the entire `tensor-fusion` suite. Both saved
binaries must contain the identical updated harness. Performance admission is
still exact-checksum based, with complete-coordinate correctness tests separate
from timing. This is not a PosteriorDB or Stan inference comparison.

### Targeted perf profiles and varied parameter states

The existing runner can profile saved matrix variants without rebuilding them:

```sh
python3 scripts/three_engine.py profile --matrix-plan plan.json --output profiles \
  --cases logistic --evaluations 100000 --profile-variants CONFIGURATION
```

This collects hardware counters, DWARF cycle samples and text reports, retaining
failed commands with a nonzero final status. Run inside a compute allocation.
`--local-profile` is an explicit diagnostic-only exception, not permission to
publish laptop timings as server results. If required by system perf policy,
use `--perf-command 'sudo -n /path/to/perf'`; the tool does not change sysctls.
Keep compilation, correctness tests and other profiles off the measurement host
while collecting comparative timings. Perf-instrumented elapsed time is not a
benchmark ranking.

Historical evaluation varies coordinate zero only. For all-coordinate controls,
the existing data generator optionally writes state banks when
`ALEA_BENCH_STATE_DIR` is set. Generate them while preparing a **new** study, not
by changing an active study's inputs. Pass `matrix-plan --state-bank-dir DIR`;
each `<case>.txt` contains ten complete whitespace-delimited finite coordinate
rows. Both executables load outside evaluation timing, cycle through the same
rows and report their existing complete-gradient/checksum receipts. Matrix replay
includes the state banks in its existing input checks. Omitting the option keeps
the historical workload unchanged.

`matrix-build --feature-axes` with no names builds just the default configuration,
useful for compiler/flag controls. This does not select generated AD: the actual
execution backend is still explicit in the matrix plan. Stan model builds use a
study-local driver object so changed CPU flags cannot silently reuse a differently
compiled shared `main.o`.

The suite does not automatically enable compiler AD for every nightly/native
cell. `matrix-build` enables it only for Enzyme cells that need it, while retaining
explicit user `RUSTFLAGS` or encoded flags for controlled experiments. Inspect
the saved per-cell flags: enabling a Cargo feature, selecting generated gradients
and enabling a compiler mode are distinct configuration axes.
The autodiff Nix shell itself supplies an AD flag: explicitly override its
`RUSTFLAGS` when preparing native-only compiler controls.
