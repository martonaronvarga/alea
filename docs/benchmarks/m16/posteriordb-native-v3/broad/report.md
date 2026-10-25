# Descriptive PosteriorDB benchmark

20 posteriors, 6 method configurations, 360 planned jobs. Status counts: `{'completed': 360}`.

Every failure and unsupported combination is retained. Diagnostics and accuracy are observations; this report makes no promotion, pass/fail or default-selection decision.

Cold wall time includes process startup, adaptation, sampling and serialization. Retained timings have engine-specific scopes; JAX host callbacks and native-JAX translations are labelled separately. RMHMC is exactness-pending; MCLMC is approximate; ATLAS/Pathfinder rows are labelled composed workflows.

| Posterior | Method | Completed/planned | Process s | Warmup s | Retained s | Worst R-hat | Min bulk ESS | Max mean error / ref SD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rate_1_data-Rate_1_model | nuts_cov_diag | 3/3 | 0.04291 | 0.004792 | 0.002983 | 1.003 | 1403 | 0.02383 |
| Rate_1_data-Rate_1_model | nuts_cov_dense | 3/3 | 0.04296 | 0.00454 | 0.00332 | 1.002 | 1371 | 0.02199 |
| Rate_1_data-Rate_1_model | nuts_rs | 3/3 | 0.01684 | 0.002157 | 0.003542 | 1.004 | 1593 | 0.002573 |
| Rate_1_data-Rate_1_model | stan_nuts_diag | 3/3 | 0.3662 | 0.012 | 0.025 | 1.002 | 1314 | 0.0266 |
| Rate_1_data-Rate_1_model | stan_nuts_dense | 3/3 | 0.3674 | 0.012 | 0.028 | 1.002 | 1314 | 0.0266 |
| Rate_1_data-Rate_1_model | blackjax_nuts | 3/3 | 5.657 | 0.04314 | 0.06087 | 1.002 | 1474 | 0.03952 |
| Rate_2_data-Rate_2_model | nuts_cov_diag | 3/3 | 0.2846 | 0.01112 | 0.007915 | 1.003 | 2846 | 0.02411 |
| Rate_2_data-Rate_2_model | nuts_cov_dense | 3/3 | 0.05187 | 0.006649 | 0.007338 | 1.002 | 2570 | 0.01435 |
| Rate_2_data-Rate_2_model | nuts_rs | 3/3 | 0.0228 | 0.003264 | 0.005059 | 1.002 | 4022 | 0.03746 |
| Rate_2_data-Rate_2_model | stan_nuts_diag | 3/3 | 0.4981 | 0.021 | 0.041 | 1.003 | 3045 | 0.01825 |
| Rate_2_data-Rate_2_model | stan_nuts_dense | 3/3 | 0.5012 | 0.024 | 0.043 | 1.003 | 3040 | 0.01573 |
| Rate_2_data-Rate_2_model | blackjax_nuts | 3/3 | 5.627 | 0.04977 | 0.08126 | 1.002 | 2967 | 0.02282 |
| Rate_5_data-Rate_5_model | nuts_cov_diag | 3/3 | 0.3532 | 0.005242 | 0.003817 | 1.003 | 1535 | 0.01174 |
| Rate_5_data-Rate_5_model | nuts_cov_dense | 3/3 | 0.0664 | 0.004936 | 0.003676 | 1.003 | 1535 | 0.01174 |
| Rate_5_data-Rate_5_model | nuts_rs | 3/3 | 0.02271 | 0.002867 | 0.004637 | 1.003 | 1648 | 0.01349 |
| Rate_5_data-Rate_5_model | stan_nuts_diag | 3/3 | 0.3745 | 0.016 | 0.033 | 1.002 | 1532 | 0.02389 |
| Rate_5_data-Rate_5_model | stan_nuts_dense | 3/3 | 0.3737 | 0.016 | 0.035 | 1.002 | 1532 | 0.02389 |
| Rate_5_data-Rate_5_model | blackjax_nuts | 3/3 | 5.654 | 0.04098 | 0.05832 | 1.002 | 1505 | 0.06753 |
| arK-arK | nuts_cov_diag | 3/3 | 0.9118 | 0.3015 | 0.5623 | 1.003 | 2467 | 0.03194 |
| arK-arK | nuts_cov_dense | 3/3 | 0.2841 | 0.1128 | 0.1208 | 1.003 | 5124 | 0.02814 |
| arK-arK | nuts_rs | 3/3 | 0.5644 | 0.1913 | 0.3507 | 1.004 | 1672 | 0.02995 |
| arK-arK | stan_nuts_diag | 3/3 | 7.082 | 2.162 | 4.173 | 1.003 | 2439 | 0.02434 |
| arK-arK | stan_nuts_dense | 3/3 | 2.452 | 0.771 | 0.968 | 1.004 | 5126 | 0.02791 |
| arK-arK | blackjax_nuts | 3/3 | 8.041 | 0.441 | 0.8498 | 1.002 | 2492 | 0.04016 |
| arma-arma11 | nuts_cov_diag | 3/3 | 0.2826 | 0.07517 | 0.07244 | 1.003 | 3106 | 0.01909 |
| arma-arma11 | nuts_cov_dense | 3/3 | 1.06 | 0.0727 | 0.07027 | 1.001 | 3711 | 0.02357 |
| arma-arma11 | nuts_rs | 3/3 | 0.08927 | 0.0289 | 0.04364 | 1.002 | 4556 | 0.02152 |
| arma-arma11 | stan_nuts_diag | 3/3 | 1.04 | 0.199 | 0.339 | 1.002 | 3538 | 0.02256 |
| arma-arma11 | stan_nuts_dense | 3/3 | 1.071 | 0.206 | 0.342 | 1.003 | 3894 | 0.02911 |
| arma-arma11 | blackjax_nuts | 3/3 | 6.849 | 0.1633 | 0.2283 | 1.004 | 2679 | 0.02512 |
| garch-garch11 | nuts_cov_diag | 3/3 | 1.115 | 0.4066 | 0.6526 | 1.004 | 1864 | 0.04114 |
| garch-garch11 | nuts_cov_dense | 3/3 | 0.7536 | 0.2661 | 0.4358 | 1.002 | 1666 | 0.02044 |
| garch-garch11 | nuts_rs | 3/3 | 0.5655 | 0.2085 | 0.3419 | 1.005 | 1398 | 0.02294 |
| garch-garch11 | stan_nuts_diag | 3/3 | 2.124 | 0.572 | 1.035 | 1.003 | 1512 | 0.03157 |
| garch-garch11 | stan_nuts_dense | 3/3 | 1.698 | 0.474 | 0.683 | 1.004 | 1946 | 0.02792 |
| garch-garch11 | blackjax_nuts | 3/3 | 8.236 | 0.4358 | 0.8161 | 1.004 | 1677 | 0.03516 |
| dogs-dogs_log | nuts_cov_diag | 3/3 | 1.062 | 0.3792 | 0.6491 | 1.004 | 1215 | — |
| dogs-dogs_log | nuts_cov_dense | 3/3 | 0.8816 | 0.3219 | 0.5567 | 1.001 | 1992 | — |
| dogs-dogs_log | nuts_rs | 3/3 | 0.8484 | 0.3124 | 0.5166 | 1.004 | 1098 | — |
| dogs-dogs_log | stan_nuts_diag | 3/3 | 50.08 | 3.173 | 5.87 | 1.006 | 1125 | — |
| dogs-dogs_log | stan_nuts_dense | 3/3 | 44.51 | 2.886 | 5.425 | 1.003 | 2224 | — |
| dogs-dogs_log | blackjax_nuts | 3/3 | 6.449 | 0.2968 | 0.5239 | 1.003 | 1067 | — |
| earnings-earn_height | nuts_cov_diag | 3/3 | 4.336 | 1.62 | 2.598 | 1.003 | 989.8 | 0.04742 |
| earnings-earn_height | nuts_cov_dense | 3/3 | 1.138 | 0.8705 | 0.2012 | 1.003 | 3311 | 0.02365 |
| earnings-earn_height | nuts_rs | 3/3 | 1.413 | 0.4472 | 0.9446 | 1.01 | 475.4 | 0.03747 |
| earnings-earn_height | stan_nuts_diag | 3/3 | 9.587 | 3.328 | 5.761 | 1.005 | 961.3 | 0.02576 |
| earnings-earn_height | stan_nuts_dense | 3/3 | 2.887 | 1.902 | 0.561 | 1.002 | 3026 | 0.01953 |
| earnings-earn_height | blackjax_nuts | 3/3 | 9.099 | 1.294 | 2.1 | 1.004 | 890.9 | 0.03576 |
| earnings-logearn_interaction_z | nuts_cov_diag | 3/3 | 1.274 | 0.4298 | 0.8019 | 1.003 | 1853 | 0.02076 |
| earnings-logearn_interaction_z | nuts_cov_dense | 3/3 | 0.704 | 0.248 | 0.4091 | 1.002 | 4326 | 0.03411 |
| earnings-logearn_interaction_z | nuts_rs | 3/3 | 0.8222 | 0.2947 | 0.5099 | 1.004 | 1427 | 0.03627 |
| earnings-logearn_interaction_z | stan_nuts_diag | 3/3 | 4.888 | 1.449 | 2.88 | 1.004 | 1830 | 0.04778 |
| earnings-logearn_interaction_z | stan_nuts_dense | 3/3 | 2.937 | 0.868 | 1.509 | 1.002 | 3911 | 0.02183 |
| earnings-logearn_interaction_z | blackjax_nuts | 3/3 | 6.546 | 0.2516 | 0.4821 | 1.004 | 2074 | 0.03006 |
| eight_schools-eight_schools_centered | nuts_cov_diag | 3/3 | 0.1609 | 0.05549 | 0.05714 | 1.124 | 25.96 | — |
| eight_schools-eight_schools_centered | nuts_cov_dense | 3/3 | 0.1972 | 0.07066 | 0.07035 | 1.318 | 10.1 | — |
| eight_schools-eight_schools_centered | nuts_rs | 3/3 | 0.08416 | 0.01874 | 0.03781 | 1.091 | 33.75 | — |
| eight_schools-eight_schools_centered | stan_nuts_diag | 3/3 | 1.069 | 0.1 | 0.122 | 1.061 | 71.98 | — |
| eight_schools-eight_schools_centered | stan_nuts_dense | 3/3 | 1.222 | 0.139 | 0.156 | 1.052 | 98.45 | — |
| eight_schools-eight_schools_centered | blackjax_nuts | 3/3 | 6.511 | 0.2496 | 0.3076 | 1.158 | 17.65 | — |
| eight_schools-eight_schools_noncentered | nuts_cov_diag | 3/3 | 0.1305 | 0.02507 | 0.03756 | 1.003 | 1951 | 0.03235 |
| eight_schools-eight_schools_noncentered | nuts_cov_dense | 3/3 | 0.1899 | 0.04683 | 0.05184 | 1.003 | 1962 | 0.03484 |
| eight_schools-eight_schools_noncentered | nuts_rs | 3/3 | 0.08537 | 0.01427 | 0.02404 | 1.006 | 2027 | 0.03068 |
| eight_schools-eight_schools_noncentered | stan_nuts_diag | 3/3 | 2.053 | 0.088 | 0.18 | 1.005 | 1868 | 0.03012 |
| eight_schools-eight_schools_noncentered | stan_nuts_dense | 3/3 | 1.505 | 0.073 | 0.131 | 1.004 | 2401 | 0.02397 |
| eight_schools-eight_schools_noncentered | blackjax_nuts | 3/3 | 6.961 | 0.1441 | 0.2477 | 1.004 | 2382 | 0.02532 |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | nuts_cov_diag | 3/3 | 9.094 | 3.023 | 6.018 | 1.223 | 13.54 | — |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | nuts_cov_dense | 3/3 | 6.473 | 2.137 | 4.269 | 1.563 | 6.98 | — |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | nuts_rs | 3/3 | 4.882 | 1.705 | 3.151 | 1.35 | 9.607 | — |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | stan_nuts_diag | 3/3 | 40.84 | 12.58 | 27.53 | 1.173 | 15.95 | — |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | stan_nuts_dense | 3/3 | 21.86 | 8.2 | 13.54 | 1.623 | 6.58 | — |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | blackjax_nuts | 3/3 | 11.6 | 1.582 | 3.345 | 1.175 | 16.83 | — |
| normal_2-normal_mixture | nuts_cov_diag | 3/3 | 1.317 | 0.402 | 0.7167 | 1.529 | 7.133 | — |
| normal_2-normal_mixture | nuts_cov_dense | 3/3 | 1.247 | 0.4208 | 0.7869 | 1.53 | 7.162 | — |
| normal_2-normal_mixture | nuts_rs | 3/3 | 0.7722 | 0.2932 | 0.4629 | 1.529 | 7.14 | — |
| normal_2-normal_mixture | stan_nuts_diag | 3/3 | 5.414 | 1.777 | 3.195 | 1.53 | 7.17 | — |
| normal_2-normal_mixture | stan_nuts_dense | 3/3 | 5.652 | 1.836 | 3.318 | 1.531 | 7.154 | — |
| normal_2-normal_mixture | blackjax_nuts | 3/3 | 6.902 | 0.2704 | 0.4795 | 1.53 | 7.157 | — |
| sblri-blr | nuts_cov_diag | 3/3 | 0.1746 | 0.05289 | 0.07806 | 1.014 | 644.6 | 0.05791 |
| sblri-blr | nuts_cov_dense | 3/3 | 0.1872 | 0.05838 | 0.07949 | 1.007 | 650.6 | 0.02282 |
| sblri-blr | nuts_rs | 3/3 | 0.08048 | 0.02368 | 0.03702 | 1.003 | 3993 | 0.02258 |
| sblri-blr | stan_nuts_diag | 3/3 | 0.947 | 0.084 | 0.135 | 1.014 | 585.9 | 0.05749 |
| sblri-blr | stan_nuts_dense | 3/3 | 0.8677 | 0.086 | 0.134 | 1.01 | 672.3 | 0.03132 |
| sblri-blr | blackjax_nuts | 3/3 | 6.654 | 0.1474 | 0.2409 | 1.007 | 570.3 | 0.06755 |
| wells_data-wells_dist100_model | nuts_cov_diag | 3/3 | 5.27 | 1.864 | 3.402 | 1.004 | 1147 | 0.0378 |
| wells_data-wells_dist100_model | nuts_cov_dense | 3/3 | 3.773 | 1.458 | 2.269 | 1.003 | 3306 | 0.01277 |
| wells_data-wells_dist100_model | nuts_rs | 3/3 | 3.41 | 1.203 | 2.192 | 1.002 | 1085 | 0.02689 |
| wells_data-wells_dist100_model | stan_nuts_diag | 3/3 | 4.175 | 1.187 | 2.461 | 1.003 | 1134 | 0.0363 |
| wells_data-wells_dist100_model | stan_nuts_dense | 3/3 | 3 | 0.939 | 1.602 | 1.002 | 2896 | 0.007952 |
| wells_data-wells_dist100_model | blackjax_nuts | 3/3 | 8.939 | 0.9821 | 1.787 | 1.005 | 922.1 | 0.02883 |
| kidiq-kidscore_momhs | nuts_cov_diag | 3/3 | 0.609 | 0.1875 | 0.1718 | 1.003 | 1453 | 0.01561 |
| kidiq-kidscore_momhs | nuts_cov_dense | 3/3 | 0.4906 | 0.141 | 0.08482 | 1.004 | 3321 | 0.02586 |
| kidiq-kidscore_momhs | nuts_rs | 3/3 | 0.1651 | 0.05396 | 0.09584 | 1.004 | 1266 | 0.05574 |
| kidiq-kidscore_momhs | stan_nuts_diag | 3/3 | 1.556 | 0.402 | 0.53 | 1.004 | 1364 | 0.07929 |
| kidiq-kidscore_momhs | stan_nuts_dense | 3/3 | 1.136 | 0.339 | 0.281 | 1.001 | 2923 | 0.02983 |
| kidiq-kidscore_momhs | blackjax_nuts | 3/3 | 6.775 | 0.2646 | 0.3051 | 1.005 | 1519 | 0.02192 |
| seeds_data-seeds_stanified_model | nuts_cov_diag | 3/3 | 0.5451 | 0.131 | 0.2317 | 1.016 | 282.7 | — |
| seeds_data-seeds_stanified_model | nuts_cov_dense | 3/3 | 1.063 | 0.377 | 0.3213 | 1.004 | 489.4 | — |
| seeds_data-seeds_stanified_model | nuts_rs | 3/3 | 0.1967 | 0.04778 | 0.09097 | 1.02 | 236.2 | — |
| seeds_data-seeds_stanified_model | stan_nuts_diag | 3/3 | 2.515 | 0.24 | 0.406 | 1.021 | 374.5 | — |
| seeds_data-seeds_stanified_model | stan_nuts_dense | 3/3 | 3.213 | 0.444 | 0.549 | 1.008 | 400.5 | — |
| seeds_data-seeds_stanified_model | blackjax_nuts | 3/3 | 8.86 | 0.2991 | 0.5129 | 1.012 | 289.8 | — |
| surgical_data-surgical_model | nuts_cov_diag | 3/3 | 0.1576 | 0.02946 | 0.05184 | 1.015 | 258.7 | — |
| surgical_data-surgical_model | nuts_cov_dense | 3/3 | 0.2217 | 0.05392 | 0.0651 | 1.004 | 1669 | — |
| surgical_data-surgical_model | nuts_rs | 3/3 | 0.08485 | 0.01485 | 0.02315 | 1.004 | 988.4 | — |
| surgical_data-surgical_model | stan_nuts_diag | 3/3 | 2.173 | 0.088 | 0.17 | 1.045 | 109.2 | — |
| surgical_data-surgical_model | stan_nuts_dense | 3/3 | 2.309 | 0.105 | 0.202 | 1.003 | 1703 | — |
| surgical_data-surgical_model | blackjax_nuts | 3/3 | 7.156 | 0.1565 | 0.2618 | 1.006 | 898.2 | — |
| radon_mn-radon_variable_intercept_noncentered | nuts_cov_diag | 3/3 | 2.881 | 1.489 | 1.106 | 1.006 | 1453 | — |
| radon_mn-radon_variable_intercept_noncentered | nuts_cov_dense | 3/3 | 17.4 | 13.44 | 3.682 | 1.004 | 1598 | — |
| radon_mn-radon_variable_intercept_noncentered | nuts_rs | 3/3 | 1.145 | 0.3552 | 0.5349 | 1.004 | 801.1 | — |
| radon_mn-radon_variable_intercept_noncentered | stan_nuts_diag | 3/3 | 19.06 | 5.149 | 4.4 | 1.007 | 1677 | — |
| radon_mn-radon_variable_intercept_noncentered | stan_nuts_dense | 3/3 | 43.32 | 27.06 | 7.307 | 1.003 | 1513 | — |
| radon_mn-radon_variable_intercept_noncentered | blackjax_nuts | 3/3 | 12.07 | 1.805 | 1.472 | 1.006 | 1538 | — |
| hmm_example-hmm_example | nuts_cov_diag | 3/3 | 1.678 | 0.6702 | 0.9434 | 1.005 | 1906 | 0.02213 |
| hmm_example-hmm_example | nuts_cov_dense | 3/3 | 0.9795 | 0.4888 | 0.4396 | 1.002 | 3630 | 0.01638 |
| hmm_example-hmm_example | nuts_rs | 3/3 | 0.7976 | 0.281 | 0.4903 | 1.004 | 1522 | 0.01624 |
| hmm_example-hmm_example | stan_nuts_diag | 3/3 | 6.213 | 2.106 | 3.122 | 1.005 | 2203 | 0.02612 |
| hmm_example-hmm_example | stan_nuts_dense | 3/3 | 4.271 | 1.571 | 1.761 | 1.003 | 3332 | 0.02143 |
| hmm_example-hmm_example | blackjax_nuts | 3/3 | 8.716 | 0.5662 | 0.8006 | 1.004 | 2204 | 0.0303 |

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
