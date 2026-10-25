# Descriptive PosteriorDB benchmark

6 posteriors, 6 method configurations, 36 planned jobs. Status counts: `{'completed': 36}`.

Every failure and unsupported combination is retained. Diagnostics and accuracy are observations; this report makes no promotion, pass/fail or default-selection decision.

Cold wall time includes process startup, adaptation, sampling and serialization. Retained timings have engine-specific scopes; JAX host callbacks and native-JAX translations are labelled separately. RMHMC is exactness-pending; MCLMC is approximate; ATLAS/Pathfinder rows are labelled composed workflows.

| Posterior | Method | Completed/planned | Process s | Warmup s | Retained s | Worst R-hat | Min bulk ESS | Max mean error / ref SD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rate_2_data-Rate_2_model | nuts_cov_diag | 1/1 | 0.008698 | 0.0007367 | 0.0003377 | 1.01 | 173.7 | 0.03924 |
| Rate_2_data-Rate_2_model | nuts_cov_dense | 1/1 | 0.008321 | 0.0007361 | 0.0003517 | 0.9994 | 259.7 | 0.02986 |
| Rate_2_data-Rate_2_model | nuts_rs | 1/1 | 0.005887 | 0.0005372 | 0.0003469 | 1.023 | 296.9 | 0.09513 |
| Rate_2_data-Rate_2_model | stan_nuts_diag | 1/1 | 0.09104 | 0.002 | 0.002 | 1.027 | 245.5 | 0.1459 |
| Rate_2_data-Rate_2_model | stan_nuts_dense | 1/1 | 0.09516 | 0.003 | 0.002 | 1.027 | 245.5 | 0.1459 |
| Rate_2_data-Rate_2_model | blackjax_nuts | 1/1 | 5.402 | 0.006996 | 0.007246 | 1.014 | 196.8 | 0.08677 |
| garch-garch11 | nuts_cov_diag | 1/1 | 0.08772 | 0.0497 | 0.03045 | 0.9999 | 114.6 | 0.2581 |
| garch-garch11 | nuts_cov_dense | 1/1 | 0.07803 | 0.04768 | 0.0227 | 1.015 | 156.6 | 0.06976 |
| garch-garch11 | nuts_rs | 1/1 | 0.05869 | 0.02941 | 0.02388 | 1.032 | 106.7 | 0.09441 |
| garch-garch11 | stan_nuts_diag | 1/1 | 0.2595 | 0.092 | 0.074 | 1.033 | 92.54 | 0.04795 |
| garch-garch11 | stan_nuts_dense | 1/1 | 0.2602 | 0.092 | 0.075 | 1.033 | 92.54 | 0.04795 |
| garch-garch11 | blackjax_nuts | 1/1 | 6.97 | 0.06662 | 0.04526 | 1.012 | 107.8 | 0.0433 |
| earnings-earn_height | nuts_cov_diag | 1/1 | 0.6589 | 0.3236 | 0.3271 | 1.062 | 21.07 | 6.283 |
| earnings-earn_height | nuts_cov_dense | 1/1 | 0.6902 | 0.3107 | 0.3714 | 1.24 | 8.341 | 6.305 |
| earnings-earn_height | nuts_rs | 1/1 | 0.584 | 0.1088 | 0.4687 | 2.397 | 2.593 | 9.434 |
| earnings-earn_height | stan_nuts_diag | 1/1 | 1.965 | 0.606 | 1.266 | 1.193 | 8.175 | 6.335 |
| earnings-earn_height | stan_nuts_dense | 1/1 | 1.988 | 0.629 | 1.266 | 1.193 | 8.175 | 6.335 |
| earnings-earn_height | blackjax_nuts | 1/1 | 6.288 | 0.2239 | 0.2418 | 1.81 | 3.141 | 6.312 |
| eight_schools-eight_schools_noncentered | nuts_cov_diag | 1/1 | 0.01381 | 0.002894 | 0.001995 | 1.022 | 181.1 | 0.1299 |
| eight_schools-eight_schools_noncentered | nuts_cov_dense | 1/1 | 0.01558 | 0.003143 | 0.003502 | 1.037 | 259.4 | 0.09678 |
| eight_schools-eight_schools_noncentered | nuts_rs | 1/1 | 0.009839 | 0.00175 | 0.001272 | 1.037 | 134.6 | 0.166 |
| eight_schools-eight_schools_noncentered | stan_nuts_diag | 1/1 | 0.1645 | 0.008 | 0.006 | 1.114 | 17.94 | 0.3679 |
| eight_schools-eight_schools_noncentered | stan_nuts_dense | 1/1 | 0.1654 | 0.008 | 0.006 | 1.114 | 17.94 | 0.3679 |
| eight_schools-eight_schools_noncentered | blackjax_nuts | 1/1 | 6.002 | 0.02129 | 0.0248 | 1.047 | 89.35 | 0.1354 |
| normal_2-normal_mixture | nuts_cov_diag | 1/1 | 0.1585 | 0.07068 | 0.0797 | 1.038 | 253.2 | — |
| normal_2-normal_mixture | nuts_cov_dense | 1/1 | 0.113 | 0.05646 | 0.04868 | 1.034 | 234.3 | — |
| normal_2-normal_mixture | nuts_rs | 1/1 | 0.08822 | 0.04719 | 0.03558 | 1.012 | 227.9 | — |
| normal_2-normal_mixture | stan_nuts_diag | 1/1 | 0.5258 | 0.22 | 0.211 | 1.01 | 133.5 | — |
| normal_2-normal_mixture | stan_nuts_dense | 1/1 | 0.621 | 0.259 | 0.257 | 1.01 | 133.5 | — |
| normal_2-normal_mixture | blackjax_nuts | 1/1 | 6.183 | 0.04048 | 0.03279 | 1.003 | 248.9 | — |
| sblri-blr | nuts_cov_diag | 1/1 | 0.03002 | 0.01269 | 0.003691 | 1.056 | 268.4 | 0.1036 |
| sblri-blr | nuts_cov_dense | 1/1 | 0.02308 | 0.01135 | 0.003681 | 1.021 | 235 | 0.1335 |
| sblri-blr | nuts_rs | 1/1 | 0.01194 | 0.003607 | 0.002751 | 1.013 | 226.7 | 0.1004 |
| sblri-blr | stan_nuts_diag | 1/1 | 0.1343 | 0.013 | 0.014 | 1.061 | 17.91 | 0.5221 |
| sblri-blr | stan_nuts_dense | 1/1 | 0.1296 | 0.013 | 0.016 | 1.061 | 17.91 | 0.5221 |
| sblri-blr | blackjax_nuts | 1/1 | 6.063 | 0.02396 | 0.01794 | 1.009 | 169.3 | 0.09169 |

## Accounting and scope

- Descriptive native-backend observations; no promotion score, automatic selection or required ESS/s speedup.
- Twenty pinned PosteriorDB model/data pairs, not the entire PosteriorDB catalogue. All density ports must pass independent Stan density-difference, gradient and potential-HVP checks before native jobs; unavailable translations never fall back to callbacks.
- Alea and nuts-rs use the same ahead-of-time Enzyme Rust target in the native lane. This is both a native Rust workflow comparison and a shared-backend comparison of sampling plus adaptation. nuts-rs has no separate modeling compiler in this protocol. Shared-Stan companion runs hold BridgeStan constant instead.
- BlackJAX uses float64 CPU native JAX densities on every admitted case. CmdStan uses its compiled C++ model and Stan AD. Versions, compiler, source/data bytes, binaries, starts and seed identities are sealed. Seed labels do not imply equal random-number streams across engines.
- Cold process wall time uses perf_counter_ns and Linux pidfd completion. Rust Instant and synchronized JAX phase clocks separate warmup, retention and serialization; JIT time is recorded separately but already included in process wall time. Compilation outside jobs is recorded as batch preparation cost, with cache status explicit; it is not a comparable per-model cold compilation experiment.
- Native derivatives use no Stan calls. BridgeStan remains loaded for coordinate names and post-timing constrained output only; those costs appear in cold process/output time. Rust retained scopes include position collection; CmdStan includes native CSV/generated quantities. No pure-kernel claim is inferred from unequal output scopes.
- Serial mode pins all chains to one CPU and runs them sequentially. Parallel mode launches one worker per chain on distinct allowed CPUs, merges draws by logical chain identity, and computes pooled diagnostics afterward. Parallel end-to-end time includes worker setup and merging. The sum of individual chain RSS peaks is an upper bound, not simultaneous aggregate RSS.
- Every failed, timed-out, unsupported, divergent and repeated draw remains accounted for. Rank R-hat, bulk/tail ESS, MCSE and reference discrepancies accompany walltime, ESS/s and capability-specific API counts. Missing counters remain null; gradients, HVPs and third derivatives have no invented equal-cost conversion.
- Library-specific adaptation policies remain declared and may differ. Frozen-geometry kernel microbenchmarks are separate evidence; these workflows compare adaptation plus retained sampling.
- Research samplers remain opt-in. MCLMC is approximate; RMHMC is exactness-pending even with real Enzyme third derivatives. Dense manifold methods use separate dimension and iteration budgets; no default or maturity changes follow from this experiment.
- Stan 2.38 HMM simplex coordinates use the ILR transform (sqrt(2)*q for K=2), not the older stick-breaking coordinate. Dogs uses a stable native Bernoulli-logit expression; an independent Decimal80 oracle validates density/score because the original inv_logit-then-bernoulli Stan density loses about2e-6 at a shared start. The actual Stan value is preserved in conformance artifacts.
- Conformance includes shared starts, deterministic perturbations/tails and reference-mean-derived coordinates where all base parameters have reference summaries. Potential HVP signs are checked against independent score differences, and dense third derivatives against Hessian differences at the first finite point. Reference-free models retain that explicit coverage limitation.
