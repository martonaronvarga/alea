# Native-backend PosteriorDB observations

This experiment measures the current Alea implementation on the existing twenty
pinned PosteriorDB model/data pairs. It is a descriptive comparison, with every
failure, timeout and unsupported configuration retained. It does not cover the
entire PosteriorDB catalogue and does not select numerical defaults or graduate
research methods.

| Engine | Density and derivatives |
| --- | --- |
| Alea | Ahead-of-time native Rust primals, Enzyme gradients, HVPs and third derivatives |
| nuts-rs | The identical Enzyme target through `CpuLogpFunc` |
| CmdStan | Original compiled Stan programs and Stan AD |
| BlackJAX | Native float64 JAX translations; no BridgeStan sampling callbacks |

The native Alea/nuts-rs pair holds the target implementation and compiler constant
while comparing sampling **and adaptation**. A companion BridgeStan lane repeats
the Rust comparisons with Stan derivatives. Library-specific adaptation remains
different; neither comparison isolates a fixed-geometry transition kernel.

The Rust ports are explicit, auditable scalar source, assembled as a build-only
`ModelBuilder` custom primal in Stan's unconstrained coordinates. The benchmark
specializes that single factor to borrow the original coordinate array directly.
It does not copy coordinates into an opaque local argument array, which triggers
a type-analysis failure in the pinned Enzyme compiler for loop-heavy models.
The private benchmark template uses reverse-over-forward HVPs and
reverse-over-forward-over-forward third derivatives. These are real Enzyme AD,
with potential signs supplied at the adapter boundary. There is no numerical
derivative fallback. This is not an automatic Stan compiler, nor a comparison of
model-authoring convenience or the ordinary analytic operation tape.

BridgeStan remains loaded by Rust/JAX workers for coordinate metadata and
constraining saved output. It is not called by their native sampler gradients.
Loading and output conversion remain charged to process/output time.

## Conformance before measurement

Each native port is checked at the shared starts, deterministic perturbations and
tail points. Where base-parameter reference means are available, additional points
are obtained by unconstraining those means and perturbing them. These better
scaled points help expose Jacobian mistakes masked by huge gradients at arbitrary
regression starts. Missing reference-coordinate coverage is reported explicitly.
Each measured engine receives exactly the requested prefix of the prepared starts;
the two-chain pilot uses the same first two points for Rust, Stan and JAX. Requests
exceeding the sealed starting-point inventory are rejected before measurement.

Density differences allow a parameter-independent normalization offset, with
tolerance `2e-7 + 2e-12 * max(abs(logp_native), abs(logp_reference))`.
Coordinate score errors use `abs(error)/(1+abs(reference)) <= 2e-6`.
Potential HVPs use independent score differences in a signed alternating direction
and tolerance `2e-4`; dense third derivatives are checked against Hessian
differences at the first point, with tolerance `2e-3` and dimension at most ten.
Finite-difference computations belong exclusively to preflight verification.

Two model-specific details matter:

- Stan 2.38 uses ILR simplex coordinates. For the two-state HMM the logit is
  `sqrt(2)*q`, with the matching Jacobian up to a constant. The ports use that
  coordinate system rather than the older stick-breaking coordinate.
- The dogs program rounds `inv_logit` before evaluating a Bernoulli density.
  Its density loses approximately `2e-6` at one shared start, while scores agree
  to approximately `2e-15`. An independent 80-digit Decimal Bernoulli-logit oracle
  checks the native density and score; conformance artifacts preserve the actual
  Stan value and the discrepancy. Its uniform-prior support is unchanged.

Native conformance failures produce explicit unsupported jobs; no native row
silently switches to a different backend. Model/data bytes, generated primals,
dependency sources, compiler, worker/model binaries and conformance outputs are
bound into preparation and source manifests.

## Protocols and timing

| Protocol | Jobs | Purpose |
| --- | ---: | --- |
| Native pilot | 36 | Six models, six baseline configurations, two chains |
| Native broad | 360 | Twenty models, six configurations, three four-chain repetitions |
| Shared Stan | 240 | Twenty models, four Rust configurations, three repetitions |
| Parallel | 72 | Six models, six baseline configurations, two four-chain repetitions |
| Research | 420 | Twenty models, twenty-one optional configurations, one repetition |

Broad baseline budgets are 512 warmup and 1,024 retained draws per chain.
Research budgets are 256/512. Dense manifold methods use at most ten dimensions,
32 warmup and 64 retained draws per chain. These budgets are deliberately
different and remain visible in every result. The pilot uses 128/128.

Serial mode pins sequential chains to one allowed CPU. Parallel mode launches
one pinned process per chain on distinct CPUs, merges by logical chain identity,
then computes pooled diagnostics. Shared ChEES adaptation is not decomposed into
independent one-chain adaptations. Parallel measurements include child setup and
merging. The sum of individual chain RSS peaks is an upper bound, not measured
simultaneous aggregate memory; the outer GNU-time high-water mark is not that
aggregate either.

Process timing uses `perf_counter_ns` and Linux pidfd completion. Rust `Instant`
and synchronized JAX timers separate warmup, retained sampling and serialization.
JAX JIT time is recorded separately and already included in process walltime.
Rust retained timing includes owned position collection; CmdStan also includes
its CSV and generated quantities, so unequal output scopes remain explicit.

Native compilation is recorded as a batch worker build and may reuse incremental
caches. Reused Stan binaries retain their original source/build records. These
are not comparable per-model cold build measurements. Per-process JIT and total
workflow time are measured directly; no batch compilation cost is silently
allocated to individual models.

Reports include rank R-hat, bulk/tail ESS, MCSE, reference mean/spread discrepancies,
divergences, depth saturation, walltime, memory and derivative API counts.
Derivative capabilities have separate counts, without an invented exchange rate.
Missing upstream counters stay null. Paired tables show Alea/competitor ratios
with failures and diagnostics alongside them; there is no quality-based row
filter, universal winner, promotion score or required `1.5x` ESS/s improvement.

MAMS, ChEES, Fisher adaptations, WALNUTS, ATLAS, transport and Pathfinder remain
opt-in. MCLMC remains approximate. RMHMC remains exactness-pending even when its
third derivatives are exact AD. Native-backend performance does not resolve the
finite-solver proposal-law obligation.

## Reproduction

With hash-verified compiled reference models already available:

```console
python3 tools/bench/alea-bench native campaign target/new-native-campaign --models target/reference-models
```

The campaign builds the worker, checks all ports, executes the pilot, then runs
the broad native, shared-Stan, parallel and research protocols. Invalid pilot
ports stop automatic expansion. Sampler failures remain observations.

Individual stages are available as `native build`, `native conform`, and
`native run`. The latter accepts `--backend enzyme|bridgestan` and
`--execution serial|parallel`. All destinations are create-new. Existing
`pdb verify`, `pdb report`, `pdb pack` and `pdb unpack` commands verify and reproduce
each run offline. The separate native CI workflow preserves preparation failures
and result artifacts; configuring it does not constitute executed platform evidence.

## Recorded results

The twenty-model campaign accounts for 1,128 outcomes: 1,095 completed, 23 failed
and ten unsupported. All five evidence bundles reproduce identical reports with
current and archived readers, both on dusk and after local import.

Read the [combined outcome](benchmarks/m16/posteriordb-native-v3/outcome.md),
[complete descriptive findings](benchmarks/m16/posteriordb-native-v3/findings.md),
[equal-budget adaptation observations](benchmarks/m16/posteriordb-native-v3/adaptation.md)
or [interactive report index](benchmarks/m16/posteriordb-native-v3/index.html).
The bundles retain every raw and unsuccessful outcome. The observations use one
physical host and the declared timing scopes; no default or maturity change follows.
