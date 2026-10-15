# Width-one funnel: fixed-time divergence replay

## Protocol declared before execution

The bounded experiment extends the existing sheared-banana replay. It keeps
`WARMUP=1000`, `DRAWS=8192`, `L=5`, the four existing starts and original seeds
1001–1004. A second disjoint seed ensemble uses 2001–2004 with the same starts.
Fisher acceptance targets are 0.8 and 0.95; rank caps are 0 and 2. All 32 chains
use frozen geometry for retained draws. No seed, threshold or budget is tuned
after examining outcomes.

For each original divergent proposal, replay the saved position and RNG using
an exact copy of the metric at refinement factors 1, 2, 4 and 8. Step size is
divided by the factor and step count multiplied by it, keeping requested
integration time fixed. Factor 1 must reproduce the original divergence,
energy error and attempted steps. Successful refinements must finish all steps;
failed refinements remain in the report. There is no requirement that any
particular refinement succeeds or that errors decrease monotonically.

Replay uses a separate target-call counter. Check the original position/score/
density, metric signature, RNG continuation and evaluation counter after replay.
No retry enters production or alters which proposals the original chain retains.
Replay calls include one constructor evaluation and are reported separately.

The starting-point Hessian is that of potential `U=-logp`:

```text
H00 = 1 + y² exp(-x)/2
H01 = H10 = -y exp(-x)
H11 = exp(-x)
```

Its largest eigenvalue is computed by the scalar symmetric 2x2 formula. The
Hessian is independently checked against central differences of the score,
including neck coordinates down to x=-8. This is original-coordinate curvature
at the starting point, not mass-normalized curvature or a bound over the path.
It must not be interpreted as a universal leapfrog stability criterion.

## Reproduction

```sh
nix develop .#stable --command cargo test --manifest-path crates/Cargo.toml \
  -p alea-mcmc --features faer --test fisher_sampling funnel_ -- --nocapture
```

## Results — 2026-10-05

[Unfiltered replay output](funnel-replay-2026-10-05.txt) records every original
failure and all four replays. Each ensemble/configuration has 32,768 retained
transitions. All 69 failures were finite endpoint `EnergyErrorLimit` rejections.

| Target acceptance | Rank cap | Original ensemble failures | New ensemble failures | Remaining at 2× / 4× / 8× refinement |
|---|---:|---:|---:|---:|
| 0.8 | 0 | 16 | 13 | 0 / 0 / 0 |
| 0.8 | 2 | 17 | 10 | 0 / 0 / 0 |
| 0.95 | 0 | 2 | 4 | 0 / 0 / 0 |
| 0.95 | 2 | 2 | 5 | 0 / 0 / 0 |

Maximum absolute endpoint energy errors across both ensembles:

| Target acceptance | Rank cap | Original | Half steps | Quarter steps | Eighth steps |
|---|---:|---:|---:|---:|---:|
| 0.8 | 0 | 28,182.050 | 673.692 | 0.160074 | 0.031751 |
| 0.8 | 2 | 19,860.195 | 595.126 | 0.223943 | 0.027070 |
| 0.95 | 0 | 11,593.618 | 79.531 | 0.115278 | 0.025474 |
| 0.95 | 2 | 8,703.471 | 79.531 | 0.115278 | 0.025474 |

Halving steps removes the threshold violations in this corpus, but errors near
674 remain unacceptable for efficient Metropolis sampling. Quarter steps give
much smaller errors on these selected proposals. The test preserves the
diagnostic outcome report rather than adding a universal recovery assertion.
All successful replays executed their full step budget. The 276 diagnostic
replays used 5,451 fused evaluations, including 276 constructor evaluations;
these calls were isolated from live sampler counters and are not a proposed
production cost or policy.

Starting x coordinates ranged from -3.3035 to +2.9695; starting largest Hessian
eigenvalues ranged from 1.0001 to 27.2174. Failures do not all start deep in the
neck. Starting curvature alone does not describe the regions traversed by a
trajectory, so this experiment does not establish a curvature-based controller.
All replayed failing proposals had frozen metric rank zero, including those
whose warmup allowed rank two. This does not test nonzero-rank funnel geometry.

The original-seed rank-capped counts are now 17/2 for targets 0.8/0.95, whereas
the [earlier report](fisher-divergences.md) recorded 8/3 before the Cholesky/SVD
fitting repair. Small floating-point changes during warmup can change subsequent
chain trajectories; the historical observations are preserved, not silently
replaced. Diagonal counts remain 16/2. No seeds, tolerances or schedules changed.
The existing analytic/BlackJAX moment gates pass on the original ensemble.
The added ensemble here supplies replay evidence only, not a new moment or
mixing certificate. No convergence or efficiency superiority is inferred.

Validation: all 12 Faer sampling tests, the scalar curvature test, MCMC
all-target Clippy with warnings denied, and workspace formatting passed.
Native statistical tests exclude Miri. SIMD/OpenBLAS and the full workspace
matrix were not rerun for this test-only extension. Rust 1.95.0/Faer 0.24.0,
pinned stable shell; base commit `59ef802` plus working-tree changes. SHA-256:

```text
32d91373c1cf10918de605dda61854ad48096b57518bc5ad5c575c836582deaf  crates/alea-mcmc/tests/fisher_sampling.rs
a0b21f3f1b0afe9f383955fe2a24a9682e2f27db7660e0afff1ecb3415441db7  crates/alea-mcmc/tests/support/targets.rs
de38d5699050b8341c5ab337b7cd5ae170529ffe20c4e484d152d62a62092815  crates/alea-math/src/fisher.rs
```
