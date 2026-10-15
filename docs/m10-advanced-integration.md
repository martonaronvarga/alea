# M10: advanced integration and batched execution

M10 adds opt-in Euclidean experiments. It does not change the default leapfrog,
windowed covariance warmup, NUTS or WALNUTS paths. Adaptation is owned and
consumed before sampling, and every retained kernel remains Metropolis corrected.

## Joint warmup and s-AIA

`JointHmcWarmup` adapts a covariance metric and dual-averaged step continuously,
but changes the two-stage coefficient only at explicit epoch boundaries: after
the initial search, after each metric window, and once at the final frozen step.
The returned `Hmc<_, WarmupMetric, TwoStage>` contains no covariance moments,
step controller, HVP workspace or coefficient selector.

`probe_metric_curvature` applies the generalized operator `M^-1 H` through the
existing metric and potential-HVP contracts. Power iteration remains an observed
lower spectral estimate. `CurvatureBound` therefore requires an explicit
inflation greater than one and calls its result a policy bound, not a certified
global Hessian bound.

The two-stage selector minimizes the published Gaussian energy-error bound
`rho_2(h,b)` on a bounded grid over `0 < h <= h_bar`, followed by a deterministic
golden-section search over the ME2--VV2 coefficient interval. It allocates
nothing and runs only during warmup. This follows the
[s-AIA formulation](https://arxiv.org/abs/2307.02096), including its
dimensionless stability interval and minimax criterion, but uses HVP curvature
with explicit inflation rather than claiming that a finite warmup certifies the
paper's fitted global frequency model.

## Split and force-gradient kernels

`GaussianResidual` declares

```text
U(q) = 0.5 sum_i precision_i (q_i - center_i)^2 + residual(q).
```

`GaussianSplitHmc` composes a residual half-kick, the exact diagonal harmonic
flow, and a residual half-kick. The capability is deliberately identity-mass;
other constant geometry must first be represented as a frozen coordinate
transform. Closed-orbit energy and signed reversal have independent scalar
oracles. The construction follows
[Split Hamiltonian Monte Carlo](https://arxiv.org/abs/1106.5941).

`SplitLogDensityGradient` exposes cheap and expensive score contributions whose
sum is the full score. `SplitForceHmc` applies the published nested leapfrog:
`L + 1` expensive evaluations, `L * (inner + 1)` cheap evaluations, and one full
endpoint density evaluation per attempted trajectory. Counters distinguish all
three. Checked wrappers validate shapes, finite inputs and complete outputs while
preserving concrete backend errors.

`ForceGradientHmc` implements the five-stage fourth-order composition

```text
B(h/6) A(h/2) [B(2h/3) + C(h^3/72)] A(h/2) B(h/6).
```

The analytic mode obtains `H M^-1 score` from the potential HVP contract. The
separate Hessian-free mode evaluates the force at its prescribed shifted
position. Tests require signed reversal, a numerical unit phase-volume Jacobian,
stationary Gaussian moments and exact evaluation counts for both modes. The
Hessian-free mode is explicitly described as reversible and volume preserving,
not symplectic, matching the limitations in the
[Hessian-free force-gradient paper](https://arxiv.org/abs/2403.10370).

All three samplers reuse aligned chain-local buffers. Allocation tests cover
retained transitions. Numerical rejection or a capability/backend error cannot
publish the proposal cache.

## Explicit batched layouts

`copy_batch_layout` converts between scalar chain-major and coordinate-major
storage without changing logical lane identity. `BatchedLeapfrog` is an
identity-mass, transactional kick/drift/evaluate/kick state machine over the
existing `BatchLogDensityGradient` capability. It accepts either physical layout,
reports batched launches and scalar-equivalent evaluations, allocates nothing in
`advance`, and publishes the entire batch only after every lane completes.
Cross-layout endpoints and signed reversal agree with scalar oracles.

This is a CPU execution contract suitable for future accelerator backends. It is
not a GPU runtime, implicit transpose, or claim that coordinate-major storage is
always faster.

## Equal-force benchmark and decision

`alea-runtime/examples/m10_benchmark.rs` is a no-overwrite release harness. The
final declared run uses four chains, 300 warmup iterations, 2,048 retained draws,
a heterogeneous 16-dimensional Gaussian, target acceptance 0.9 and exactly 32
force evaluations per retained transition. Leapfrog uses 32 macrosteps; OMF2,
BCSS2 and adaptive2 use 16. Launch, physical-layout conversion, one batched
launch, actual warmup, retained sampling, target calls and HVP calls are recorded.
An isolated HVP call and dense covariance factorization are also reported as
microcosts; they are not added again to total time because actual warmup already
contains derivative and factorization work.

Local release artifact `/tmp/alea-m10-benchmark-20261015-v4.csv`:

| Method | max R-hat | min bulk ESS | min bulk ESS/total s | calls | HVP calls | total s |
|---|---:|---:|---:|---:|---:|---:|
| LF | 1.0259 | 88.0 | 1,089 | 300,602 | 0 | 0.0808 |
| OMF2 | 1.0054 | 599.3 | 15,287 | 300,670 | 0 | 0.0392 |
| BCSS2 | 1.2112 | 13.9 | 358 | 300,674 | 0 | 0.0387 |
| adaptive2 | 1.0067 | 1,540.4 | 34,025 | 300,672 | 360 | 0.0453 |

The measured shared microcosts were 0.064 microseconds per layout conversion,
15.4 microseconds for construction plus one batched launch, 0.100 microseconds
per HVP, and 57.1 microseconds for the isolated dense factorization. Timings are
single-host analytic-target measurements, not portable backend claims.

OMF2 and adaptive2 pass the retained diagnostic gate in this run. The fixed
trajectory protocol exposes severe BCSS2 resonance and a weaker LF failure, so
the suite-wide stationarity exit gate is negative. Earlier 32-force/1,024-draw
and 30-force/2,048-draw pilots also failed and were not used to tune seeds or
erase the failure. No integrator or joint warmup is promoted to a default.

Reproduce from the repository root:

```console
nix develop .#stable --command cargo run --release \
  --manifest-path crates/Cargo.toml -p alea-runtime \
  --example m10_benchmark -- /tmp/new-m10-results.csv
```
