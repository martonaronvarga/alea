# Fixed-HMC Fisher efficiency experiment

Run from the repository root, using a new output path each time:

```sh
nix develop .#stable --command cargo run --manifest-path crates/Cargo.toml -p alea-mcmc --features faer --release --example fisher_efficiency -- /tmp/fisher-efficiency.csv
nix develop .#stable --command python3 tools/m4f-efficiency.py /tmp/fisher-efficiency.csv /tmp/fisher-efficiency.json
```

The fixed protocol uses correlated Gaussian, logistic, banana and width-one
funnel targets; seeds 3101–3104 and 4101–4104 in separate four-chain ensembles;
alternating (-1,1)/(1,-1) starts; 1,000 warmup iterations, 8,192 retained draws,
five leapfrog steps, initial step .1 and acceptance target .8. Target formulas
and five observables are shared with the independent BlackJAX regressions.
No thinning, divergence removal, seed selection or adaptive retries are used.

Eight methods compare covariance diagonal, windowed Fisher diagonal, rank caps
1/2 with period20/history40, rank2 with period80/history160, and weighted diagonal
with dual averaging, Robbins–Monro or Adam. Weighted moments use offset1/power.75;
RM rate1; Adam rate.3, beta1=.9, beta2=.999, epsilon=1e-8. RM/Adam learning rates
use offset1/power.6. Weighted ridge1e-5 acts on normalized scatter; windowed ridge
acts on raw scatter. These are intentionally different estimators.

Construction, warmup and sampling elapsed time and actual fused calls are
recorded. Methods rotate by seed. Batch-means ESS uses the largest variance of
the mean across batch sizes64/128/256 and the IID estimate. Per-chain ESS is
summed, then minimized over observables; chains are never concatenated.
ESS/call and ESS/second include warmup costs. These are local cost diagnostics,
not modern rank-normalized bulk/tail ESS or proof of convergence.

The 320 pooled-moment checks use the larger within-chain/between-chain standard
error, six standard errors plus .03, with independent logistic reference MCSE.
Failed checks remain in the JSON and cause a nonzero exit. CI tests analyzer
contracts using explicitly synthetic rows, not untracked measurement artifacts.

Initial execution passed all moment checks but covariance was usually more
efficient; RM/Adam produced substantially more banana divergences. Defaults
remain covariance/dual averaging; Fisher and its alternative controllers remain
experimental. Successful moment checks do not justify promotion. Rerun after
numerical changes: trajectory-sensitive draws and timings need not be identical.

For separate cold-fit/warmup timing and process peak RSS, build `fisher_profile`
in release and run `tools/fisher-cost.py` inside `.#profiling`. Its fixed workloads
cover d64 and d256; RSS includes the process and is not retained-draw heap usage.
