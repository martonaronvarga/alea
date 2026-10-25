# Descriptive PosteriorDB experiment

The current M16 goal is an informative account of inference quality and cost,
including the implemented research methods. Comparable inference quality and
competitive runtime are useful baseline results. No 1.5x ESS/s requirement,
promotion verdict, eligibility filter, or default change is computed here. The
original v1 experiment and its historical thresholds remain immutable.

The declared protocol is `tools/bench/protocols/posteriordb-v2.json`;
`tools/bench/schemas/v2.json` describes its structural contract. The Python reader
also checks source seals, seeds, planned denominators, raw chain completeness,
finite coordinates, diagnostic artifacts and immutable indexes. All failures,
timeouts and unsupported combinations remain records. A completed job means that
its expected draws were produced and analyzed, not that inference was adequate.

## Corpus and execution

Twenty model/data pairs from PosteriorDB commit
`91027f9cdd13b7e76283fec3f2898f59d4b83ec4` span rates, regressions, AR/ARMA,
GARCH, mixtures, centered/noncentered hierarchical models, radon, seeds, surgical
outcomes and a hidden Markov model. Original Stan/data bytes, upstream metadata,
licenses and available empirical reference draws are vendored under
`tools/bench/corpus/pdb-v2/`. This is a diverse subset, not complete PosteriorDB
coverage. Reference means and sample standard deviations summarize the supplied
reference draws; they are not exact posterior truth.

The plan contains 26 configurations, three repetitions and four chains per job:
1,560 planned jobs. Regular jobs use 512 warmup and 512 retained draws per chain.
Independent transition seeds are 18001–18004, 28001–28004 and 38001–38004. Chains
share declared finite initial positions across methods and repetitions. Finite
initial positions come from at most 100 uniform [-2,2] attempts per chain with
fixed initialization seeds. BlackJAX uses distinct warmup/retained PRNG subtrees. Stan Pathfinder uses a
SHA-256-tagged initialization seed distinct from its NUTS sampling seed.
Sampling is serial on one allowed CPU, with native
thread pools limited to one thread. Method ordering rotates across cases and
repetitions. Build, dependency fetch and preflight occur before measurement.

RMHMC uses a smaller explicit budget: dimension at most ten, 32 warmup and 64
retained draws per chain, and a 60-second deadline. Other jobs have a 120-second
deadline. Unsupported combinations keep their place in the matrix. These unequal
budgets preclude treating RMHMC timing or ESS as a directly matched race.

## Implemented configurations

| Configurations | Recorded workflow |
| --- | --- |
| `nuts_cov_diag`, `nuts_cov_dense` | Alea eight-step HMC covariance warmup, frozen conversion to NUTS |
| `nuts_fisher_diag`, `nuts_fisher_rank4` | Fisher diagonal or rank-cap-four metric, dual averaging, frozen NUTS |
| `nuts_fisher_rm`, `nuts_fisher_adam` | Fisher diagonal with explicit Robbins–Monro or Adam warmup step policy |
| `nuts_atlas_joint` | Implemented ATLAS curvature probe plus joint s-AIA metric/two-stage warmup, frozen NUTS |
| `nuts_pathfinder` | Four paths, forty iterations, raw importance initializer, covariance warmup and NUTS |
| `nuts_transport` | Fixed-step HMC map learning over half the warmup budget, frozen affine transport and covariance/NUTS |
| `hmc_cov`, `hmc_omf`, `hmc_bcss` | Adapted fixed-length HMC with leapfrog, OMF or BCSS two-stage integration |
| `ghmc`, `drghmc` | Covariance HMC warmup, persistent or delayed-rejection retained kernel |
| `mams`, `mclmc` | Corrected or explicitly approximate isokinetic integration; fixed step .08/sqrt(d), eight steps; d>=2 |
| `chees` | Four-chain shared ChEES warmup with identity geometry, frozen randomized HMC |
| `walnuts_d`, `walnuts_r2p` | Covariance warmup, deterministic or randomized-two-point dyadic WALNUTS |
| `rmhmc_softabs`, `rmhmc_manifold` | Exactness-pending SoftAbs RMHMC, fixed .02 step or tuner over .01/.02/.04 |
| `stan_nuts_diag`, `stan_nuts_dense` | Native CmdStan adapted NUTS |
| `stan_pathfinder_nuts` | Native four-path PSIS Pathfinder initializer, then adapted NUTS |
| `blackjax_nuts` | Native BlackJAX window adaptation and NUTS, with explicitly identified target backend |
| `nuts_rs` | Pinned nuts-rs native diagonal NUTS adaptation |

ATLAS here is a composed workflow using the component Alea implements, not a full
ATLAS sampler. Alea Pathfinder does not implement PSIS. RMHMC remains
exactness-pending; sampling diagnostics and derivative checks do not close its
open reversal/volume obligations. MCLMC remains approximate even when a short
run appears accurate. Fixed-policy research settings are exploratory, with no
post-result tuning or hidden fallback. Transport failures are exposed as typed
failures, rather than retrying at a smaller step. Target-specific splitting and
analytic force-gradient decompositions are outside this generic target adapter;
no decomposition or new numerical algorithm is invented for this benchmark.

## Target and derivative accounting

Alea and nuts-rs use the same compiled BridgeStan 2.7.0 / Stan 2.38.0 target with
`propto=true`, `jacobian=true` on unconstrained coordinates. Consequently this
experiment measures implemented sampling workflows against Stan AD targets; it
does not establish the speed of Alea's native modeling or likelihood evaluation.
The benchmark-only Rust package is isolated from the nine-crate production
workspace and does not change a public Rust library API.

Native JAX translations cover Rate 1/2, centered/noncentered eight schools, wells
dist100 and kidiq momhs. Preparation checks their gradients and density differences
at three points against Stan. The remaining fourteen BlackJAX targets use explicitly
labelled BridgeStan host callbacks through `custom_vjp`. Their callback/transfer
cost must not be presented as native JAX kernel speed. BlackJAX uses float64 CPU
JAX 0.9.2 / BlackJAX 1.5. JIT compilation caches are disabled between fresh jobs.

Before measurement, central differences check all initial gradients and one
alternating-sign HVP direction. An HVP limitation does not exclude gradient-only
methods. The hidden Markov model has finite checked gradients but nonfinite HVPs
at the declared initial point; its ATLAS workflow is unsupported, while full
Hessian RMHMC remains an observed attempt. SoftAbs uses AD Hessians; third
potential derivatives are explicit central differences of Hessians with
h=1e-4*(1+abs(q)). These approximations remain labelled. On several coupled models the numerical
third tensor fails SoftAbs's strict permutation-consistency check before retained
sampling. Such rows describe a derivative-backend contract failure, not an
executed RMHMC inference comparison. No tensor projection or post-result
relaxation of the check is applied.

Actual gradient/Hessian/HVP API calls are counted for shared-backend Rust jobs;
phase snapshots distinguish warmup and retained work. RMHMC exposes retained
factorization and triangular-solve counts. Native Stan total target callbacks,
native JAX gradient counts and upstream allocation counts are unavailable, not
inferred from leapfrog counts. Instrumentation and adaptation policies differ.

## Timing, diagnostics and reference comparisons

Fresh-process wall time uses `perf_counter_ns` with Linux pidfd exit readiness.
The final blocking wait only reaps a notified child; it does not poll with a
50-ms timeout. Deadlines kill the complete process group. Each job records wall
nanoseconds, peak RSS from its own GNU-time process, command, affinity, phases and
partial outputs. Instrument resolution is not a claim of nanosecond accuracy. Cold means a fresh
process; operating-system file/page caches are not flushed.

Native retained timings include kernel conversion and collecting owned positions,
while affine transport also maps positions back to target coordinates. They
exclude CSV writing and constrained outputs. Stan native sampling times include
its CSV and generated quantities. Stan's emitted phase timer has its own precision;
outer process timing remains high resolution. BlackJAX timings synchronize device
execution and separate JIT compilation. Worker cold phases start inside the
worker; interpreter/dynamic-loader startup remains in total process wall time.
There is no single interchangeable timing scope across engines. The Rust driver
runs four chains inside one worker; the CmdStan adapter launches a native process
per chain. Python driver startup, native subprocess startup and output conversion
are included in cold wall time. This is a workflow comparison rather than a
standalone numerical-kernel microbenchmark.

Constrained outputs always include the model parameters; transformed parameters
are also included when the full output count is at most 256. Generated quantities
are excluded from BridgeStan outputs and remain in native Stan CSV files. The
metadata records this choice and reports the number of reference coordinates
actually compared.

Both unconstrained and constrained coordinates retain rank/folded R-hat,
bulk/tail ESS, mean MCSE, empirical mean and sample SD. Undefined diagnostics stay
null with their reason while finite empirical means/spreads remain available.
Reference comparisons report signed mean error in reference posterior SD units
and current/reference SD ratios, alongside current-chain MCSE. These are
descriptive discrepancies without a significance or universal calibration claim.
Divergences and depth saturation remain observations, without an eligibility gate.

Reports include every raw job, completion denominators, median costs across
completed repetitions, worst observed diagnostics, reference discrepancies,
backend labels and failures. Medians condition on completion, explicitly beside
failure counts. A single physical host and three repetitions do not establish a
universal winner or cross-host speed. Inspect accuracy and diagnostics alongside
runtime; throughput from a stuck or biased chain has little practical meaning.

## Reproduction

The full dusk measurement completed on 2026-10-21 and is retained under
[`benchmarks/m16/posteriordb-v2`](benchmarks/m16/posteriordb-v2/README.md): 1,267
completed jobs, 260 failures and 33 unsupported jobs, accounting for all 1,560.
All raw/source/build seals verify; current and archived readers regenerate the
same report and supplementary analyses. Begin with the
[findings](benchmarks/m16/posteriordb-v2/findings.md) and inspect per-posterior
diagnostics/reference discrepancies alongside timing. Completion is an execution
outcome, not an inference adequacy claim. No sampler/default or maturity changes.

From a matching source tree inside the pinned stable Nix shell:

```sh
python3 tools/bench/alea-bench pdb experiment tools/bench/protocols/posteriordb-v2.json target/new-pdb-experiment
```

This fetches clean pinned BridgeStan/nuts-rs sources, resolves pinned CmdStan/JAX
environments, builds the isolated Rust worker and all models, records preparation
and derivative checks, seals a new manifest, runs every job and creates JSON,
Markdown, interactive HTML and CSV reports. Compilers never overlap numerical
measurement. All output directories are new; failed preparations retain logs.
Original corpus generation is `python3 tools/bench/pdb_corpus.py PINNED_PDB_CHECKOUT`.
The checked corpus is already vendored and does not require a fresh PDB download.

Stages can be invoked separately with `pdb prepare`, `pdb seal` and `pdb run`.
`pdb prepare NEW_DIRECTORY --models EXISTING_MODELS_DIRECTORY` reuses only models
whose source, build-log and executable hashes verify. `pdb verify RUN_DIRECTORY`
requires only standard-library Python and retained artifacts. `pdb report
RUN_DIRECTORY --output NEW_JSON` regenerates the machine-readable report without
an engine. Reports read references from the sealed source archive rather than
an advanced worktree.

`pdb pack RUN_DIRECTORY NEW_BUNDLE_DIRECTORY` archives complete raw draws,
failures, native outputs, model build logs, source snapshots, metadata and indexes
in checksummed archive segments of at most 32 MiB, alongside review reports.
`pdb unpack BUNDLE_DIRECTORY NEW_DIRECTORY` checks and restores the entire run.
An independent reader from the archived sources can regenerate its report. A new
numerical rerun gets a new manifest/build identity and new timings; bitwise timing
reproduction is not promised. Existing v1 archives and reports remain readable.
