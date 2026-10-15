# Fisher divergence diagnosis and explicit acceptance control

These observations predate the Cholesky/SVD fitting repair. The subsequent
[funnel replay study](funnel-replay.md) adds a second seed ensemble and reports
current counts without replacing the historical results below.

The retained-draw corpus exposed 17 divergent proposals in 32,768 sheared-banana
draws after rank-capped Fisher warmup at target acceptance 0.8. This investigation
keeps that baseline intact. The production change is an opt-in
`FisherOptions::with_target(AcceptanceTarget)` builder, matching covariance
warmup's existing control. Default acceptance remains 0.8.

## Controlled integration replay

`divergent_fisher_proposals_recover_with_fixed_time_step_refinement` saves the
state and RNG before every transition in the existing four-chain ensemble. For
each divergent proposal it reconstructs exactly the same metric and replays with
the same momentum draw, first with `(epsilon/2, 2L)`, then `(epsilon/4, 4L)`.
Initial energies must match. The original chain and its RNG are never changed;
the replay is a diagnostic, **not** a production retry/acceptance algorithm.

All 17 observed failures were finite endpoint energy errors beyond the existing
1,000 threshold. Original errors ranged from about 1,019 to 2.7e31. Halving the
step at fixed integration time removed every divergence; maximum absolute error
was about 0.674. Quarter steps also removed every divergence, with maximum
absolute error about 0.136. Each refined path must complete its full step count.

This is direct evidence of coarse-step integration instability in these proposals,
not evidence that the fitted metric is invalid. The same metric, target, initial
state and momentum work under refinement. It does not prove the absence of every
metric/controller bug or diagnose all possible targets. No fitting equation,
rank threshold, cache rule or divergence threshold was changed.

## Explicit remedy and failed universal hypothesis

Increasing the target acceptance is a standard way to request smaller adapted
steps, with potential efficiency costs; it cannot guarantee that every region
of a varying-curvature target will be resolved. See the
[Stan HMC adaptation documentation](https://mc-stan.org/docs/reference-manual/mcmc.html).
Fisher previously hard-coded 0.8 and lacked this user control.

```rust
use alea_mcmc::adapt::FisherOptions;
use alea_mcmc::config::SamplerConfigError;

fn conservative_warmup() -> Result<FisherOptions, SamplerConfigError> {
    Ok(FisherOptions::new(1000).with_target(0.95.try_into()?))
}
```

The validated type rejects NaN, infinities and endpoints before options can be
constructed. Both the initial dual-averaging phase and the final fixed-geometry
restart use the configured target. Zero warmup still does no adaptation.
Search uses its existing criterion; the divergence limit remains unchanged.

We selected 0.95 before running the experiment, with exactly the original seeds,
starts, warmup/draw budgets, leapfrog count, fixtures and moment tolerances.
The initial hypothesis of **zero divergences on all three targets failed** on
the width-one funnel. We did not increase the target again or adjust seeds or
moment tolerances to conceal this failure. The funnel remains a moment/cache
regression with reported divergences, not a divergence-free assertion. The two
banana cases retain the successful zero-divergence check at 0.95.

Stable/Faer retained divergence counts, each out of 32,768 draws:

| Target | Fisher variant | Default 0.8 | Explicit 0.95 |
|---|---|---:|---:|
| Banana | Diagonal | 20 | 0 |
| Banana | Rank cap 2 | 18 | 0 |
| Sheared banana | Diagonal | 0 | 0 |
| Sheared banana | Rank cap 2 | 17 | 0 |
| Width-one funnel | Diagonal | 16 | 2 |
| Width-one funnel | Rank cap 2 | 8 | 3 |

Covariance-diagonal retained divergences at 0.95 were zero on all three targets.
Sheared-banana Fisher fits still froze at rank two in every chain. All cases
passed the unchanged five analytic/reference moment gates, coherent cache checks,
rejection preservation, frozen geometry and actual target-call checks. Sampling
still cost 163,840 fused evaluations per configuration in this run.

For the sheared-banana rank-capped case, frozen step sizes changed from
`(0.7685, 0.6836, 0.8123, 0.7192)` to `(0.5222, 0.5456, 0.5698, 0.5008)`.
Unlike the controlled replay, the acceptance experiment reruns all of warmup,
so it also changes learned geometry and subsequent chain paths. It is not a
comparison holding the metric fixed.

## Limits and next evidence

At fixed `L=5`, smaller steps shorten the integration time and can hurt mixing;
unchanged call counts and passing broad moment gates do not establish better
ESS/sec. Warmup can still diverge. A finite run with zero divergences does not
prove safety on future draws or independent ensembles. No defaults are promoted.
The [funnel follow-up](funnel-replay.md) now supplies bounded fixed-time replay
evidence; matched mixing/cost evaluation remains open. A local step controller
must eventually supply its own reversible
selection/correction proof rather than copying this diagnostic retry.

Run the diagnostic and opt-in regressions with:

```sh
nix develop .#stable --command cargo test --manifest-path crates/Cargo.toml \
  -p alea-mcmc --features faer --test fisher_sampling -- --nocapture
```

Private controller tests use a separate dual-averaging instance to verify wiring
before and after freeze, nondefault targets, zero warmup and unchanged
geometry/cache contracts. Full stable/Faer workspace checks, tests, Clippy and
rustdoc passed, as did scalar sampling and the combined SIMD/Faer/OpenBLAS library
and sampling suites. Both new configuration/controller tests passed strict Miri.
Native statistical tests intentionally remain outside Miri.
