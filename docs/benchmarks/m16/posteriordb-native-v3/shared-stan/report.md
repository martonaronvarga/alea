# Descriptive PosteriorDB benchmark

20 posteriors, 4 method configurations, 240 planned jobs. Status counts: `{'completed': 240}`.

Every failure and unsupported combination is retained. Diagnostics and accuracy are observations; this report makes no promotion, pass/fail or default-selection decision.

Cold wall time includes process startup, adaptation, sampling and serialization. Retained timings have engine-specific scopes; JAX host callbacks and native-JAX translations are labelled separately. RMHMC is exactness-pending; MCLMC is approximate; ATLAS/Pathfinder rows are labelled composed workflows.

| Posterior | Method | Completed/planned | Process s | Warmup s | Retained s | Worst R-hat | Min bulk ESS | Max mean error / ref SD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rate_1_data-Rate_1_model | nuts_cov_diag | 3/3 | 0.04648 | 0.006411 | 0.005604 | 1.003 | 1483 | 0.03332 |
| Rate_1_data-Rate_1_model | nuts_fisher_diag | 3/3 | 0.04708 | 0.006207 | 0.005355 | 1.003 | 1448 | 0.0388 |
| Rate_1_data-Rate_1_model | nuts_fisher_rank4 | 3/3 | 0.04737 | 0.006818 | 0.005536 | 1.003 | 1448 | 0.0388 |
| Rate_1_data-Rate_1_model | nuts_rs | 3/3 | 0.01991 | 0.003269 | 0.005688 | 1.004 | 1593 | 0.002573 |
| Rate_2_data-Rate_2_model | nuts_cov_diag | 3/3 | 0.05966 | 0.009889 | 0.01219 | 1.002 | 2834 | 0.02734 |
| Rate_2_data-Rate_2_model | nuts_fisher_diag | 3/3 | 0.05807 | 0.009101 | 0.01038 | 1.002 | 3365 | 0.02029 |
| Rate_2_data-Rate_2_model | nuts_fisher_rank4 | 3/3 | 0.06184 | 0.01078 | 0.01231 | 1.002 | 3365 | 0.02029 |
| Rate_2_data-Rate_2_model | nuts_rs | 3/3 | 0.02963 | 0.005734 | 0.009634 | 1.002 | 4022 | 0.03746 |
| Rate_5_data-Rate_5_model | nuts_cov_diag | 3/3 | 0.0468 | 0.006558 | 0.00626 | 1.002 | 1532 | 0.007286 |
| Rate_5_data-Rate_5_model | nuts_fisher_diag | 3/3 | 0.1021 | 0.01334 | 0.01303 | 1.002 | 1371 | 0.01582 |
| Rate_5_data-Rate_5_model | nuts_fisher_rank4 | 3/3 | 0.08698 | 0.01277 | 0.008872 | 1.002 | 1371 | 0.01582 |
| Rate_5_data-Rate_5_model | nuts_rs | 3/3 | 0.02106 | 0.003546 | 0.006309 | 1.003 | 1648 | 0.01349 |
| arK-arK | nuts_cov_diag | 3/3 | 7.679 | 2.676 | 4.953 | 1.002 | 2409 | 0.02515 |
| arK-arK | nuts_fisher_diag | 3/3 | 6.902 | 2.329 | 4.562 | 1.003 | 1724 | 0.04589 |
| arK-arK | nuts_fisher_rank4 | 3/3 | 1.77 | 0.7755 | 0.9779 | 1.003 | 5200 | 0.02732 |
| arK-arK | nuts_rs | 3/3 | 4.991 | 1.729 | 3.24 | 1.005 | 1725 | 0.047 |
| arma-arma11 | nuts_cov_diag | 3/3 | 0.6372 | 0.2242 | 0.3394 | 1.002 | 3634 | 0.01992 |
| arma-arma11 | nuts_fisher_diag | 3/3 | 0.7257 | 0.2559 | 0.3277 | 1.003 | 3396 | 0.03707 |
| arma-arma11 | nuts_fisher_rank4 | 3/3 | 0.6821 | 0.3236 | 0.3044 | 1.002 | 3416 | 0.02475 |
| arma-arma11 | nuts_rs | 3/3 | 0.3797 | 0.145 | 0.2182 | 1.002 | 4556 | 0.02152 |
| garch-garch11 | nuts_cov_diag | 3/3 | 1.923 | 0.7406 | 1.201 | 1.003 | 1575 | 0.03812 |
| garch-garch11 | nuts_fisher_diag | 3/3 | 1.509 | 0.5117 | 0.9671 | 1.005 | 1367 | 0.04829 |
| garch-garch11 | nuts_fisher_rank4 | 3/3 | 1.092 | 0.3837 | 0.667 | 1.003 | 1717 | 0.02229 |
| garch-garch11 | nuts_rs | 3/3 | 1.2 | 0.4329 | 0.7501 | 1.005 | 1330 | 0.03366 |
| dogs-dogs_log | nuts_cov_diag | 3/3 | 4.721 | 1.711 | 2.968 | 1.003 | 959.1 | — |
| dogs-dogs_log | nuts_fisher_diag | 3/3 | 4.676 | 1.66 | 2.976 | 1.003 | 1114 | — |
| dogs-dogs_log | nuts_fisher_rank4 | 3/3 | 3.772 | 1.51 | 2.22 | 1.003 | 2465 | — |
| dogs-dogs_log | nuts_rs | 3/3 | 3.203 | 1.171 | 2.015 | 1.004 | 1098 | — |
| earnings-earn_height | nuts_cov_diag | 3/3 | 10.93 | 4.215 | 6.675 | 1.003 | 849.7 | 0.02834 |
| earnings-earn_height | nuts_fisher_diag | 3/3 | 5.649 | 1.626 | 3.977 | 1.015 | 498 | 0.04142 |
| earnings-earn_height | nuts_fisher_rank4 | 3/3 | 0.9203 | 0.4126 | 0.4655 | 1.002 | 3171 | 0.01767 |
| earnings-earn_height | nuts_rs | 3/3 | 4.265 | 1.365 | 2.883 | 1.011 | 504.4 | 0.07358 |
| earnings-logearn_interaction_z | nuts_cov_diag | 3/3 | 6.236 | 2.228 | 3.954 | 1.003 | 1915 | 0.03389 |
| earnings-logearn_interaction_z | nuts_fisher_diag | 3/3 | 5.116 | 1.773 | 3.167 | 1.004 | 1667 | 0.04399 |
| earnings-logearn_interaction_z | nuts_fisher_rank4 | 3/3 | 2.625 | 0.9334 | 1.639 | 1.003 | 3963 | 0.05269 |
| earnings-logearn_interaction_z | nuts_rs | 3/3 | 3.378 | 1.213 | 2.146 | 1.004 | 1095 | 0.0517 |
| eight_schools-eight_schools_centered | nuts_cov_diag | 3/3 | 0.3446 | 0.09497 | 0.09713 | 1.109 | 25.23 | — |
| eight_schools-eight_schools_centered | nuts_fisher_diag | 3/3 | 0.169 | 0.04099 | 0.07622 | 1.077 | 52.75 | — |
| eight_schools-eight_schools_centered | nuts_fisher_rank4 | 3/3 | 0.182 | 0.04835 | 0.06576 | 1.165 | 17.33 | — |
| eight_schools-eight_schools_centered | nuts_rs | 3/3 | 0.09801 | 0.02512 | 0.04613 | 1.181 | 15.89 | — |
| eight_schools-eight_schools_noncentered | nuts_cov_diag | 3/3 | 0.2683 | 0.06287 | 0.06314 | 1.004 | 2206 | 0.02151 |
| eight_schools-eight_schools_noncentered | nuts_fisher_diag | 3/3 | 0.211 | 0.04228 | 0.04885 | 1.004 | 2305 | 0.02453 |
| eight_schools-eight_schools_noncentered | nuts_fisher_rank4 | 3/3 | 0.28 | 0.05219 | 0.05446 | 1.004 | 1503 | 0.0225 |
| eight_schools-eight_schools_noncentered | nuts_rs | 3/3 | 0.09725 | 0.02245 | 0.03819 | 1.004 | 1881 | 0.03417 |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | nuts_cov_diag | 3/3 | 38.67 | 13.76 | 24.86 | 1.33 | 9.592 | — |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | nuts_fisher_diag | 3/3 | 28.16 | 9.626 | 19.08 | 1.32 | 9.765 | — |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | nuts_fisher_rank4 | 3/3 | 15.89 | 7.114 | 8.727 | 1.542 | 7.167 | — |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | nuts_rs | 3/3 | 17.76 | 6.549 | 11.19 | 1.383 | 8.757 | — |
| normal_2-normal_mixture | nuts_cov_diag | 3/3 | 5.61 | 2.012 | 3.558 | 1.531 | 7.157 | — |
| normal_2-normal_mixture | nuts_fisher_diag | 3/3 | 5.055 | 1.606 | 3.289 | 1.53 | 7.158 | — |
| normal_2-normal_mixture | nuts_fisher_rank4 | 3/3 | 4.925 | 1.749 | 3.222 | 1.53 | 7.158 | — |
| normal_2-normal_mixture | nuts_rs | 3/3 | 4.012 | 1.516 | 2.359 | 1.529 | 7.14 | — |
| sblri-blr | nuts_cov_diag | 3/3 | 0.2079 | 0.06616 | 0.09767 | 1.015 | 525.1 | 0.03964 |
| sblri-blr | nuts_fisher_diag | 3/3 | 0.226 | 0.07155 | 0.06165 | 1.003 | 4090 | 0.04041 |
| sblri-blr | nuts_fisher_rank4 | 3/3 | 0.1825 | 0.0754 | 0.06451 | 1.002 | 4343 | 0.02397 |
| sblri-blr | nuts_rs | 3/3 | 0.1127 | 0.03253 | 0.05416 | 1.003 | 3993 | 0.02258 |
| wells_data-wells_dist100_model | nuts_cov_diag | 3/3 | 3.055 | 1.049 | 1.928 | 1.005 | 1259 | 0.01068 |
| wells_data-wells_dist100_model | nuts_fisher_diag | 3/3 | 2.818 | 1 | 1.777 | 1.004 | 1000 | 0.0378 |
| wells_data-wells_dist100_model | nuts_fisher_rank4 | 3/3 | 1.841 | 0.6682 | 1.107 | 1.002 | 3648 | 0.006321 |
| wells_data-wells_dist100_model | nuts_rs | 3/3 | 2.129 | 0.7911 | 1.322 | 1.004 | 963.8 | 0.03145 |
| kidiq-kidscore_momhs | nuts_cov_diag | 3/3 | 0.9539 | 0.4368 | 0.4756 | 1.004 | 1495 | 0.02508 |
| kidiq-kidscore_momhs | nuts_fisher_diag | 3/3 | 0.6085 | 0.2216 | 0.3443 | 1.005 | 937.2 | 0.05885 |
| kidiq-kidscore_momhs | nuts_fisher_rank4 | 3/3 | 0.6665 | 0.1575 | 0.1846 | 1.001 | 4306 | 0.01474 |
| kidiq-kidscore_momhs | nuts_rs | 3/3 | 0.4571 | 0.1622 | 0.2784 | 1.004 | 1228 | 0.03876 |
| seeds_data-seeds_stanified_model | nuts_cov_diag | 3/3 | 0.7313 | 0.226 | 0.405 | 1.009 | 324.1 | — |
| seeds_data-seeds_stanified_model | nuts_fisher_diag | 3/3 | 0.5909 | 0.1614 | 0.3464 | 1.247 | 12.06 | — |
| seeds_data-seeds_stanified_model | nuts_fisher_rank4 | 3/3 | 0.5177 | 0.1966 | 0.2516 | 1.011 | 336.6 | — |
| seeds_data-seeds_stanified_model | nuts_rs | 3/3 | 0.4143 | 0.1363 | 0.2198 | 1.021 | 271 | — |
| surgical_data-surgical_model | nuts_cov_diag | 3/3 | 0.2533 | 0.06562 | 0.0986 | 1.02 | 224.7 | — |
| surgical_data-surgical_model | nuts_fisher_diag | 3/3 | 0.2173 | 0.04904 | 0.0811 | 1.005 | 795.1 | — |
| surgical_data-surgical_model | nuts_fisher_rank4 | 3/3 | 0.2086 | 0.06586 | 0.06915 | 1.005 | 775.2 | — |
| surgical_data-surgical_model | nuts_rs | 3/3 | 0.1829 | 0.04608 | 0.07732 | 1.007 | 581.8 | — |
| radon_mn-radon_variable_intercept_noncentered | nuts_cov_diag | 3/3 | 13.24 | 7.438 | 5.531 | 1.007 | 1455 | — |
| radon_mn-radon_variable_intercept_noncentered | nuts_fisher_diag | 3/3 | 7.816 | 2.827 | 4.701 | 1.006 | 915.5 | — |
| radon_mn-radon_variable_intercept_noncentered | nuts_fisher_rank4 | 3/3 | 8.143 | 3.522 | 4.35 | 1.008 | 1213 | — |
| radon_mn-radon_variable_intercept_noncentered | nuts_rs | 3/3 | 5.282 | 2.085 | 2.935 | 1.004 | 801.1 | — |
| hmm_example-hmm_example | nuts_cov_diag | 3/3 | 6.193 | 2.848 | 3.334 | 1.004 | 1936 | 0.02245 |
| hmm_example-hmm_example | nuts_fisher_diag | 3/3 | 4.051 | 1.535 | 2.48 | 1.004 | 1244 | 0.02105 |
| hmm_example-hmm_example | nuts_fisher_rank4 | 3/3 | 2.691 | 1.047 | 1.588 | 1.004 | 3499 | 0.01894 |
| hmm_example-hmm_example | nuts_rs | 3/3 | 2.859 | 1.082 | 1.789 | 1.006 | 1430 | 0.03112 |

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
