# M4F completion experiment protocol — 2026-10-06

Declared before executing the new experiment. All runs use the current pinned
release/Faer CPU implementation, no adaptive retries, and the existing target
formulas/observables independently checked against BlackJAX.

- Four two-dimensional targets: correlated Gaussian, logistic, banana, width-one
  funnel. Five observables per target from `tests/support/targets.rs`.
- Two independent ensembles: seeds 3101–3104 and 4101–4104. Starts alternate
  `(-1,1)` and `(1,-1)`. No seed selection after seeing results.
- 1,000 warmup iterations and 8,192 retained draws, five leapfrog steps, initial
  step .1, acceptance target .8. No thinning or removal of divergences/repeats.
- Eight methods: covariance diagonal; existing windowed Fisher diagonal;
  Fisher rank 1/window 20/history 40, rank 2/window 20/history 40, rank 2/window
  80/history 160; diminishing weighted diagonal with dual averaging, RM, or Adam.
- Weighted alpha uses offset 1/exponent .75. RM rate 1; Adam rate .3, beta1 .9,
  beta2 .999, epsilon 1e-8; both learning-rate schedules use offset 1/exponent .6.
  Weighted ridge 1e-5 is on normalized scatters, window ridge on raw scatters.
- Record every scenario and observable, actual warmup and total fused calls,
  construction+warmup+retained-draw elapsed time, divergences, final rank and step.
  Rotate method order by seed; do not infer timing accuracy beyond this local run.
- Estimate per-chain batch-means ESS using batch lengths 64,128,256; retain the
  largest variance-of-mean estimate and cap ESS at N. This is a cost diagnostic,
  not M6 rank-normalized bulk/tail ESS or a convergence certificate.
- Check pooled means separately in each four-chain ensemble against analytic
  moments (Gaussian, banana, funnel) or the independent pinned logistic reference.
  Use six standard errors, taking the larger of within-chain batch uncertainty
  and between-chain uncertainty, plus the existing .03 absolute floor. Include
  reference MCSE for logistic. Report failures; do not relax the gate.
- Report ESS per total gradient call and ESS per total elapsed time, including
  warmup. Promotion means validated opt-in fixed-HMC functionality with documented
  tradeoffs, not universal superiority over covariance or zero divergences.

Rank/window sensitivity above is complemented by the existing d8/d64 warmup and
d64/d256 cold-fit workloads. Remeasure their current binary with native peak RSS;
these are not retained-draw memory or general high-dimensional efficiency claims.
If any prerequisite remains unverified, keep the corresponding roadmap item open.
