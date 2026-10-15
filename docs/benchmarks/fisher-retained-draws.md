# Fisher retained-draw validation

This is a bounded M4F correctness regression, not a promotion or performance
benchmark. The implementation is `crates/alea-mcmc/tests/fisher_sampling.rs`.
It changes no estimator, warmup policy, sampler defaults or target definitions.

## Fixed protocol

Compare reference covariance-diagonal warmup, Fisher-diagonal warmup and
Fisher warmup with a rank cap of two (requires `alea-mcmc/faer`). All use the
existing two-dimensional banana, logistic and width-one funnel targets in
`tests/support/targets.rs`. The funnel has a standard-normal log-scale variable;
it is **not** the more severe width-three Neal funnel.

Each configuration runs four chains with seeds 1001–1004, starts
`[-1,-1], [1,1], [-1,1], [1,-1]`, 1,000 warmup transitions, and 8,192 retained
transitions per chain. Initial geometry is identity, initial step size is 0.1,
and each trajectory has five leapfrog steps. Existing default step-size search,
dual averaging and each controller's own schedule are unchanged. Geometry and
step size are frozen before sampling. Seeds, budgets and moment tolerances were
chosen before the first run; backend differences must not be handled by tuning
them until a test passes.

Every transition, including a rejection or numerical divergence, contributes its
retained state. The test independently reevaluates the analytic target at that
state, checks the cached log density and score, and checks bit-exact cache
preservation after rejection. These extra validation evaluations bypass the
sampler's counter. Metric basis actions/log determinant and step size must be
unchanged after sampling. For nondivergent transitions, actual fused target calls
must equal five; divergent trajectories may terminate early.

Five observables are checked: `x`, `y`, `x²`, `y²`, `xy`. Banana and funnel use
their existing inverse generative transforms, giving independent standard-normal
coordinates and exact expectations `(0,0,1,1,0)`. Logistic uses its two parameter
coordinates. All three are also compared with the identity-metric rows of the
existing pinned `tests/fixtures/blackjax-sampling.csv` (BlackJAX 1.5, JAX/jaxlib
0.9.2, independent seeds 301–304). That reference uses **different** trajectory
settings and 1,024 burn-in transitions without metric adaptation. It is a
distributional check, not a trajectory replay or matched efficiency comparison.

The Monte Carlo standard error (MCSE) is the larger nonoverlapping batch-means
estimate from batches of 256 and 512. Batches stay inside chain boundaries;
incomplete batches are rejected rather than silently dropping draws. If `b_j`
are the equal-size batch means and there are `B` batches, the estimate is
`sqrt(sum((b_j - overall_mean)²) / (B * (B - 1)))`. Small deterministic tests
check this helper, including uncertainty across distinct chain means.

For every observable, require `0 < MCSE < 0.12`. Analytic moment differences
must be less than `6 * MCSE + 0.01`. Differences from the independently sampled
BlackJAX reference must be less than `6 * hypot(MCSE, reference_MCSE) + 0.01`.
These are deliberately broad regression gates, not a proof of unbiasedness or
convergence. Batch means can underestimate uncertainty when mixing is poor;
this is not a substitute for rank-normalized R-hat, bulk/tail ESS or MCSE studies.

## Observations

Stable Faer debug execution passed all moment/cache/evaluation-count gates.
The new suite also passed scalar and combined SIMD/Faer/OpenBLAS builds; the full
stable/Faer workspace checks, tests, Clippy and rustdoc passed.
The following totals are over four chains: 4,000 warmup transitions and 32,768
retained transitions for each row. The harness reports per-chain step sizes,
actual target calls, attempted steps, acceptances, divergences and frozen ranks
with `--nocapture`; divergences are not excluded from moment estimates.

| Target | Adaptation | Warmup divergences | Retained divergences | Frozen Fisher ranks |
|---|---|---:|---:|---|
| Banana | Covariance diagonal | 405 | 1 | N/A |
| Banana | Fisher diagonal | 299 | 20 | 0, 0, 0, 0 |
| Banana | Fisher rank cap 2 | 287 | 18 | 0, 0, 0, 0 |
| Logistic | Covariance diagonal | 68 | 0 | N/A |
| Logistic | Fisher diagonal | 18 | 0 | 0, 0, 0, 0 |
| Logistic | Fisher rank cap 2 | 18 | 0 | 0, 0, 0, 0 |
| Width-one funnel | Covariance diagonal | 148 | 13 | N/A |
| Width-one funnel | Fisher diagonal | 116 | 16 | 0, 0, 0, 0 |
| Width-one funnel | Fisher rank cap 2 | 102 | 8 | 0, 0, 0, 0 |

Every retained transition in this run reached all five evaluations, including
those rejected by the energy-error check: 163,840 fused evaluations per row.
This does not imply equal wall time, ESS, or warmup cost. Divergence counts are
reported findings, not new tuned acceptance thresholds for the test suite.

Importantly, the current spectral filter discarded all low-rank directions in
the **final** metric for these targets. Rank-capped warmup may still follow a
different path, but this does not validate retained sampling with a nonzero-rank
metric. No claim about ranks at intermediate fits follows from final-rank output.

## Remaining promotion gates

- Broader non-Gaussian retained sampling with a demonstrably nonzero frozen rank;
  the bounded sheared-banana extension below covers one such case. Retain the
  original rank-zero cases as coverage of the legitimate diagonal fallback.
- Severe funnels, broader noncommuting ill-conditioning, and explicit robustness
  criteria rather than inferring robustness from these five moments.
- Matched initialization, total warmup-plus-sampling cost, memory, bulk/tail
  ESS per second and per gradient, with uncertainty and convergence checks.

Fisher remains experimental. The observed banana divergences and zero-rank
coverage limit of the original corpus are reasons not to promote it from that
evidence alone.

## Nonzero-rank extension: sheared banana

The additional case preserves the existing budget, seeds, starting positions,
five-step trajectories and statistical tolerances. It changes only the target:
if `z` has the original banana law, define

```text
q0 = z0
q1 = 3*z0 + z1
z0 = q0
z1 = q1 - 3*q0
```

The linear shear has determinant one. Thus the density is `logp_banana(z)` and
the score is `(s0 - 3*s1, s1)` for the banana score `(s0,s1)` evaluated at `z`.
Equivalently, generate independent standard normals `(x,y)` and set
`q=(x, 3*x + y + 0.4*(x²-1))`. The inverse observables are again `(x,y,x²,y²,xy)`,
so both the analytic expectations and the original BlackJAX banana reference
apply. This is **not** a new independently executed BlackJAX sheared-target run.

The test checks log density and observables against that separate generative
formula on a 15-point grid, and compares the pulled-back score with central
differences of the expanded scalar density. The coefficient three and the
requirement that **each** rank-capped chain finish with rank greater than zero
were fixed before running the new sampler case. Neither spectral thresholds nor
sampler settings were changed to obtain rank. The existing signed-unit starts
are used in `q` coordinates; they are not transformed from the base starts.

Observed stable/Faer results:

| Adaptation | Warmup divergences | Retained divergences | Frozen Fisher ranks |
|---|---:|---:|---|
| Covariance diagonal | 234 | 0 | N/A |
| Fisher diagonal | 141 | 0 | 0, 0, 0, 0 |
| Fisher rank cap 2 | 293 | 17 | 2, 2, 2, 2 |

All three configurations passed the same five moment gates and cache/freeze/call
checks. Every retained transition used five evaluations (163,840 per row).
The extended suite passed scalar, Faer and combined SIMD/Faer/OpenBLAS builds,
along with the adjacent Fisher tests, covariance property test and Faer MCMC
all-target Clippy. The extension adds no production or unsafe code.
The rank-two means were `(0.00163, 0.00373, 1.03001, 1.00767, 0.01122)` with
MCSE `(0.00610, 0.00568, 0.01579, 0.01616, 0.00984)`.

This supplies bounded nonzero-rank non-Gaussian coverage, not a performance win:
the low-rank option had more divergences than the two diagonal baselines. Also,
rank two in dimension two is full rank, not evidence about scalability when
rank is much smaller than dimension. Broader geometry and efficiency gates stay
open; no production algorithm or default changed.

## Reproduction

For the follow-up fixed-time replay and opt-in 0.95 acceptance experiment, see
[Fisher divergence diagnosis](fisher-divergences.md). The original 0.8 baselines
above are preserved; funnel divergences are reduced but remain unresolved.

```sh
nix develop .#stable --command cargo test --manifest-path crates/Cargo.toml \
  -p alea-mcmc --features faer --test fisher_sampling -- --nocapture --test-threads=1
```

Omit `--features faer` for the scalar covariance/Fisher-diagonal cases. The
combined backend check uses `.#blas` with `--features faer,simd,openblas`.
The long sampling regressions are native-only (`cfg(not(miri))`), as are the
existing statistical sampling tests. No new unsafe code is introduced.
