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

`kernel_compare.py --suite base` preserves the original 15-case default.
`--suite mixture-scores` selects ten event-score, parameter-score and native
mixture-factor controls; `--suite all` runs both sets. New mixture cases require
the updated runner in both saved builds and exact checksum agreement. The native
factor uses public `modeling::continuous` and `FiniteMixture<Normal>` and includes
per-evaluation construction. No benchmark-local probability or derivative is used.
