# Immutable inference experiments (M16)

M16 supplies one manifest, execution and reporting contract for reference NUTS
experiments. It adds an S2 opt-in experiment utility, without promoting a sampler,
changing numerical defaults or closing the M11 RMHMC map-law obligation.

The primary modules are `tools/bench/alea_bench/{schema,manifest,runner,report}.py`
and `crates/alea-cli/src/bin/bench.rs`. Supporting modules prepare pinned engines,
define independent densities and generative observations, and adapt external
workers. Offline tests in `tools/bench/tests/` exercise the scientific and
provenance contracts. Existing numerical implementations, historical M4–M15
measurements and research promotion policies are outside this change.

## Frozen inputs and complete results

JSON schema version one implements V1 accounting and V2 diagnostics. A manifest
seals the actual worktree source bytes, dependency/toolchain pins, corpus/data,
unconstrained model measure, geometry, integrator policy, seeds, chain counts,
warmup and retained budgets, thresholds, exclusions and maturity/exactness ledger
content. Git HEAD is advisory because uncommitted source is hashed too. Each
result binds the entire manifest digest. A rewritten threshold seal cannot
reinterpret old jobs. Unknown versions/fields, duplicate JSON keys, nonfinite
numbers, malformed tasks and unsupported policies are rejected.

`prepare` fetches locked dependencies and builds engines before measurement,
preserving build commands, logs, toolchain and environment settings. The sealed
source archive includes the native hcubature inputs compiled by `build.rs` and
their upstream license; the unused pcubature implementation/table is excluded.
The plan is written before execution. Every planned job must appear in the final
index, including failed processes. Original failure messages and partial raw
outputs survive. Missing failures, draw rows or artifacts cannot improve a
report. SHA-256 detects drift; it is not an authenticity signature.

Each job runs in a fresh process group, pinned to one allowed CPU, with numerical
thread settings fixed to one. Its own GNU time measurement supplies peak RSS;
compiler/controller/preflight/diagnostic memory is excluded. A deadline kills the
worker process group and reaps the direct child. Output admission bounds expected
CSV sizes and checks actual output after completion; this is not an OS disk quota
or a hostile-code sandbox. The controller and machine may still experience
external load, so timing is observational rather than a hard real-time guarantee.
GNU time reports the maximum process RSS in the job, rather than a simultaneous
sum of every process's memory footprint.

Results separate build, setup/cold import, JIT, warmup and retained timings where
the engine exposes them. Retained times exclude serialization for Alea, BlackJAX
and nuts-rs; Stan's native phase timings include its CSV output. The common
efficiency denominator is full fresh-process wall time, including import, JIT,
sampling and serialization. This favors some deployment patterns over others;
warm retained-only comparisons must not silently reuse that label. All raw draws,
including repeated states and divergent transitions, remain in the denominator.

Alea and nuts-rs record actual target callback counts. Alea additionally counts
metric velocity/momentum operations. Integration steps, acceptance, divergences,
depth saturation and available energies/errors are recorded separately.
Unexposed upstream gradient/allocation/HVP counts are `null` with reasons, never
estimated from leapfrog steps. Peak RSS is not an allocation count. Energy scopes
differ: Alea reports maximum finite signed increase including zero, nuts-rs
reports maximum absolute error, Stan reports native `energy__`, and BlackJAX's
proposal energy is not substituted for the initial Hamiltonian. E-BFMI is null
when compatible energy series are unavailable.

Original acceptance values remain in raw CSV. The analyzer permits an eight-epsilon
upper roundoff envelope and caps only the aggregate probability at one; larger
overshoots, negative values and nonfinite probabilities fail. An initial incomplete
attempt exposed BlackJAX's one-ulp overshoot and is archived separately. The final
attempt uses the same scientific thresholds, seeds, steps and model data.

## Corpus and comparison policy

`tools/bench/corpus/v1.json` has 13 cases with explicit tuning/held-out partitions:
Gaussian/AR/ill-conditioned Gaussian, banana and funnel controls, logistic
regression, hierarchical normal, stochastic volatility, latent Gaussian, Wiener
drift, hierarchical DDM boundary events, and centered/noncentered eight schools.
Three tuning cases are distinguished from ten held-out cases. Data and initial
points are versioned. No target or threshold is tuned after observing the
held-out results.

All densities use the declared unconstrained Lebesgue measure. Parameter-dependent
Jacobian terms occur once. Parameter-independent constants may be omitted;
Wiener's precomputed fixed-shape base is retained by the independent oracle.
Wiener inference here varies drift with fixed boundary/start/nondecision time.
DDM calibration covers boundary events, not a full hierarchical reaction-time
model or all Wiener parameters.

The vendored PosteriorDB eight-schools sources and data come from
[the upstream repository](https://github.com/stan-dev/posteriordb/tree/91027f9cdd13b7e76283fec3f2898f59d4b83ec4),
with the pinned revision, original URLs, file digests and BSD license recorded in
`tools/bench/corpus/vendor/`. These are small selected models, not a claim to have
run the whole PosteriorDB suite.

The predeclared `reference-v1` protocol uses four chains, 256 discarded warmup
steps, 512 retained draws, depth eight and three timing repetitions. Alea,
BlackJAX 1.5/JAX 0.9.2 and CmdStan 2.38.0 use fixed identity geometry and step
sizes, float64 and multinomial NUTS with no adaptation/jitter. Their RNGs and
tree policies differ. JAX uses synchronized compiled scans and disables its
persistent compilation cache. Compilation and cold import are measured separately.

Pinned nuts-rs 0.18.3 at `a762aae513bf7bb5ccb27ffc9a938a923fc837ec`
requires positive tuning and initializes/adapts diagonal geometry. It uses a
fixed step and disables jitter, but its geometry is explicitly unmatched; its
timing cannot qualify a matched-geometry promotion. The Rust adapters share
analytic formulas, while JAX differentiates the independent Python density and
Stan differentiates a separate model. Alea calls the Wiener kernel; the other
Rust/JAX drift adapters use its fixed-shape algebraic identity. Expanded nuts-rs
statistics/position allocation is included in retained time. These differences
were declared before the measured run.

## Falsifiable scientific gates and tracking

Before each job, actual Alea density/gradient probes at three points are compared
with independent Python density values and central differences. The report uses
the existing validated rank-normalized R-hat, bulk/tail ESS and MCSE, plus depth,
divergence and nonfinite-energy gates. Analytic means and second moments are
checked where available. Cross-engine means must agree within the sealed MCSE
and absolute tolerance. Good R-hat/ESS alone cannot certify unbiased inference.

SBC generates independent prior truth and data for latent Gaussian, hierarchical
normal and hierarchical DDM boundary models, 32 datasets each. It checks four-bin
rank histograms, central 80% truth coverage, and independently generated held-out
posterior predictive PIT/coverage. Binary PIT is randomized using posterior event
probability. Failed datasets remain in the denominator; failed chain diagnostics
also prevent calibration qualification. Finite, correlated Monte Carlo ranks and
small dataset counts provide regression evidence, not universal calibration or
precise tail guarantees. Only Alea runs SBC in this protocol.

Promotion requires all held-out correctness, cross-engine and calibration gates,
no worse failure rate, and a bootstrap lower process-wall ESS-efficiency ratio at
least 1.5 on every matched held-out comparison. Failed quality pairs cannot enter
a favorable ratio. Reports retain open exactness laws and never mutate a ledger.
Three timing repetitions with fixed numerical seeds quantify timing variation,
not uncertainty over arbitrary sampler seeds or physical hosts.

The user subsequently clarified baseline readiness as the current goal:
comparable inference quality and competitive runtime are sufficient, without a
mandatory 1.5x superiority claim. The gate above describes the historical sealed
experiment; it does not define baseline readiness. The controller's timeout wait
also polls at intervals up to 50 ms, limiting resolution for the short native
jobs. New performance claims need corrected timing and a fresh adapted-baseline
protocol, with the old artifacts and thresholds preserved. Baseline and stronger
research criteria are distinguished in [the settled plan](settled-plan.md).

`track` accepts complete bundles from multiple machines, separates physical-host,
corpus/policy/case/engine/build strata, and emits bootstrap timing bands and
informational regressions against the first supplied build in each matching
stratum. A quality failure disables that timing band's eligibility. Host identity
uses CPU/architecture/system and a hashed node identity; cloned hostnames may
require distinct machine naming. The manual `performance.yml` workflow can
collect separate runners; persistent self-hosted runners are needed for stable
same-machine baselines. No remote run or workflow publication is implied by a
local run. Correctness CI tests contracts and artifact integrity without failing
on noisy elapsed-time thresholds. The checked M16 measurement covers one physical
host; synthetic two-host fault tests validate separation, not cross-host speed.

A tracking build identity hashes the engine binary, complete recorded sources and
toolchain pins. Python adapter changes therefore distinguish builds even when
the interpreter executable is unchanged.

## Reproduction

The checked `docs/benchmarks/m16/` bundle contains its manifest, complete raw and
native compressed draws, failures, phase records, process/build logs, diagnostics,
source/preparation archives, index and generated report. No promotion evidence
depends on an ignored target directory or `/tmp` file. See its README for exact
commands. Report regeneration needs only Python and the bundle; numerical reruns
also need the pinned engines and matching sources. A rerun produces new timings
and may produce machine-dependent numerical results; byte-identical timing is
not promised. Rebuilds use the archived source snapshot when the worktree has
advanced. Every new attempt writes a new directory instead of relabeling an old
measurement.


## Current descriptive extension

The user's current M16 scope is descriptive and informative, including the
implemented research samplers. The v2 [PosteriorDB method](m16-posteriordb.md)
uses adapted baselines, 20 posteriors and 26 configurations, fine process timing,
phase costs, fidelity and diagnostics with all failures retained. It computes no
promotion or 1.5x ESS/s requirement. The v1 report above remains historical and
immutable; its decision fields do not define the new report's goal.
