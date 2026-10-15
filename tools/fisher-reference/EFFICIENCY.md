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

## Independent modern diagnostics

Pass a second new path to `fisher_efficiency` to retain little-endian f64 draws:
`... --example fisher_efficiency -- /tmp/efficiency.csv /tmp/efficiency.bin`.
Serialization occurs outside the per-chain sampling timer. The binary has
256 chains in CSV encounter order, 8192 draws per chain, five observables per draw.

Use the pinned Python environment from `tools/blackjax-reference/default.nix`
(ArviZ 0.23.4) to run `tools/fisher-reference/modern_diagnostics.py CSV BIN OUTPUT`.
Set writable `XDG_CACHE_HOME` and `MPLCONFIGDIR` in restricted environments.
The analyzer verifies complete metadata and raw-draw means/variances, then checks
rank R-hat <= 1.01 and bulk/tail ESS >= 400 separately for each four-chain
ensemble and observable (320 gates). Undefined diagnostics are null-valued
failures, not discarded observations. Bulk ESS/total fused call and ESS/total
second include construction, warmup and retained sampling, but not serialization.
Run analyzer contracts with `python -m unittest discover -s tools/fisher-reference
-p test_modern_diagnostics.py` in that same environment.

The initial modern run fails one gate: correlated Gaussian, `rank2_w80_h160`,
seeds 3101–3104, observable 1 has R-hat 1.011216 (bulk ESS 670.617, tail ESS
1663.015). All other diagnostic gates pass. This artifact predates the subsequent
Fisher reduction optimization; do not treat it as validation of changed draws.
The preset threshold and experimental/default distinction remain unchanged.
The October 9 post-optimization rerun also passes all 320 moment checks and
319/320 modern gates, with the same single R-hat failure; artifacts are
`target/m4f-optimized-2026-10-09.{csv,bin}` and
`target/m4f-optimized-diagnostics-2026-10-09.json`.
