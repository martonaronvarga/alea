# Per-target workflow observations

Before/after entries compare sealed version2 with this experiment. They describe whole workflows, not an isolated cutoff effect. The separate regression/AR-K ablation and identical-phase one-step probes isolate those questions more closely.

Times are per-job medians. R-hat and ESS use unconstrained coordinates; mean/spread discrepancies use available empirical PosteriorDB reference parameters. A completed run is not a convergence certificate. Undefined quantities stay absent.

## earnings-earn_height

The pilot must learn coordinate scales before retention; difficult regression starts are directly checked in the ablation.

The baseline changed from historical HMC warmup to native NUTS warmup: completed 3/3 → 1/1; worst R-hat 4.567 → 3.083; minimum bulk ESS 4.272 → 5.127; maximum mean error/reference SD 48.97 → 6.248; maximum spread-ratio error 0.9999 → 0.8676.

| Workflow | Complete/planned | Process s | Retained s | Worst R-hat | Min bulk ESS | Mean error/ref SD | Spread-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | 1/1 | 2.233 | 0.6841 | 3.083 | 5.127 | 6.248 | 0.8676 |
| nuts_cov_dense | 1/1 | 2.48 | 0.9255 | 1.792 | 6.683 | 6.344 | 0.9738 |
| nuts_fisher_diag | 1/1 | 0.436 | 0.1973 | 1.146 | 43.64 | 0.1323 | 0.1267 |
| nuts_fisher_rank4 | 1/1 | 0.4392 | 0.1821 | 1.04 | 102.5 | 0.03692 | 0.04699 |
| nuts_fisher_rm | 1/1 | 1.206 | 0.4277 | 4.642 | 4.789 | 48.96 | 0.9999 |
| nuts_fisher_adam | 1/1 | 1.092 | 0.368 | 4.643 | 4.789 | 48.96 | 0.9999 |
| nuts_transport | 1/1 | 1.93 | 0.8574 | 3.119 | 5.116 | 6.359 | 0.9669 |
| hmc_cov | 1/1 | 1.047 | 0.05708 | 3.168 | 5.086 | 6.375 | 0.9402 |
| hmc_omf | 1/1 | 0.2834 | 0.1066 | 3.674 | 4.939 | 10.11 | 32.46 |
| hmc_bcss | 1/1 | 1.363 | 0.114 | 2.69 | 5.332 | 6.349 | 0.9903 |
| ghmc | 1/1 | 1.014 | 0.05362 | 3.887 | 4.892 | 6.372 | 0.9334 |
| drghmc | 1/1 | 1.095 | 0.06278 | 4.159 | 4.848 | 6.379 | 0.9173 |
| mams | 1/1 | 1.137 | 0.1189 | 2.888 | 5.203 | 6.372 | 0.9387 |
| mclmc | 1/1 | 1.041 | 0.05658 | 3.413 | 5.026 | 6.374 | 0.9355 |
| chees | 1/1 | 1.113 | 0.1993 | 3.162 | 5.086 | 6.337 | 0.9966 |
| walnuts_d | 1/1 | 3.017 | 0.2422 | 3.2 | 5.07 | 6.262 | 0.8985 |
| walnuts_r2p | 1/1 | 1.787 | 0.1565 | 3.173 | 5.123 | 6.269 | 0.909 |
| rmhmc_softabs | 0/1 | — | — | — | — | — | — |
| rmhmc_manifold | 0/1 | — | — | — | — | — | — |
| stan_nuts_diag | 1/1 | 2.603 | 1.22 | 2.599 | 5.391 | 6.365 | 0.996 |
| stan_nuts_dense | 1/1 | 2.555 | 1.195 | 2.599 | 5.391 | 6.365 | 0.996 |
| stan_pathfinder_nuts | 0/1 | — | — | — | — | — | — |
| blackjax_nuts | 1/1 | 47.01 | 14.09 | 2.394 | 5.641 | 6.346 | 0.958 |
| nuts_rs | 1/1 | 3.049 | 2.043 | 2.646 | 5.341 | 4.707 | 1.702 |
| nuts_saia_joint | 1/1 | 0.2823 | 0.1298 | 3.924 | 4.903 | 48.78 | 0.9999 |
| nuts_pathfinder_raw | 1/1 | 3.642 | 0.819 | 2.191 | 5.799 | 6.227 | 0.8583 |
| nuts_pathfinder_psis | 1/1 | 3.749 | 1.003 | 2.219 | 5.766 | 6.235 | 0.9128 |
| atlas_full | 1/1 | 3.91 | 2.058 | 3.096 | 5.112 | 6.166 | 0.7142 |

Unresolved outcomes: rmhmc_softabs: 1 failed/unsupported repetition(s); rmhmc_manifold: 1 failed/unsupported repetition(s); stan_pathfinder_nuts: 1 failed/unsupported repetition(s). See failures.csv for exact reasons.

## sblri-blr

The pilot must learn coordinate scales before retention; difficult regression starts are directly checked in the ablation.

The baseline changed from historical HMC warmup to native NUTS warmup: completed 3/3 → 1/1; worst R-hat 1.019 → 1.009; minimum bulk ESS 240.9 → 256.8; maximum mean error/reference SD 16.26 → 0.09509; maximum spread-ratio error 157.1 → 0.1631.

| Workflow | Complete/planned | Process s | Retained s | Worst R-hat | Min bulk ESS | Mean error/ref SD | Spread-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | 1/1 | 0.1098 | 0.004439 | 1.009 | 256.8 | 0.09509 | 0.1631 |
| nuts_cov_dense | 1/1 | 2.57 | 0.01736 | 1.015 | 282.5 | 0.08472 | 0.0911 |
| nuts_fisher_diag | 1/1 | 0.2218 | 0.007065 | 1.025 | 318.9 | 0.0774 | 0.1729 |
| nuts_fisher_rank4 | 1/1 | 0.4495 | 0.009284 | 1.063 | 238.4 | 0.1435 | 0.1128 |
| nuts_fisher_rm | 1/1 | 0.5258 | 0.1067 | 4.643 | 4.789 | 799.7 | 945.1 |
| nuts_fisher_adam | 1/1 | 0.5318 | 0.07572 | 4.616 | 4.792 | 794.9 | 949.2 |
| nuts_transport | 1/1 | 0.1063 | 0.03315 | 1.029 | 147.9 | 0.07809 | 0.1332 |
| hmc_cov | 1/1 | 0.05789 | 0.009755 | 1.093 | 33.74 | 0.1145 | 0.09751 |
| hmc_omf | 1/1 | 0.07181 | 0.01895 | 1.621 | 7.519 | 1.301 | 1.715 |
| hmc_bcss | 1/1 | 0.07709 | 0.01901 | 1.583 | 7.838 | 0.385 | 0.1072 |
| ghmc | 1/1 | 0.03975 | 0.007638 | 1.068 | 293.8 | 0.1014 | 0.09473 |
| drghmc | 1/1 | 0.06137 | 0.006135 | 1.056 | 154.3 | 0.04885 | 0.1388 |
| mams | 1/1 | 0.09779 | 0.03646 | 1.067 | 48.31 | 0.1349 | 0.1066 |
| mclmc | 1/1 | 0.0589 | 0.01366 | 1.046 | 248.2 | 0.06768 | 0.2341 |
| chees | 1/1 | 0.309 | 0.1203 | 1.049 | 206.2 | 0.09601 | 0.1355 |
| walnuts_d | 1/1 | 0.06078 | 0.02098 | 1.031 | 145.3 | 0.09758 | 0.1019 |
| walnuts_r2p | 1/1 | 0.07591 | 0.03209 | 1.014 | 156.8 | 0.1986 | 0.1164 |
| rmhmc_softabs | 0/1 | — | — | — | — | — | — |
| rmhmc_manifold | 0/1 | — | — | — | — | — | — |
| stan_nuts_diag | 1/1 | 0.2411 | 0.025 | 1.204 | 16.44 | 0.793 | 0.09842 |
| stan_nuts_dense | 1/1 | 0.1856 | 0.019 | 1.204 | 16.44 | 0.793 | 0.09842 |
| stan_pathfinder_nuts | 1/1 | 0.3951 | 0.03 | 1.256 | 16.06 | 0.5333 | 0.113 |
| blackjax_nuts | 1/1 | 12.44 | 0.6315 | 1.014 | 297.6 | 0.1052 | 0.08044 |
| nuts_rs | 1/1 | 0.01886 | 0.003671 | 1.03 | 296.6 | 0.1178 | 0.1136 |
| nuts_saia_joint | 1/1 | 0.1304 | 0.088 | 3.668 | 5.014 | 500 | 930.1 |
| nuts_pathfinder_raw | 1/1 | 1.126 | 0.01305 | 1.03 | 64.92 | 0.1391 | 0.1749 |
| nuts_pathfinder_psis | 1/1 | 0.1348 | 0.01125 | 1.076 | 39.08 | 0.07539 | 0.152 |
| atlas_full | 1/1 | 0.4186 | 0.08272 | 1.121 | 62.27 | 0.2421 | 0.1201 |

Unresolved outcomes: rmhmc_softabs: 1 failed/unsupported repetition(s); rmhmc_manifold: 1 failed/unsupported repetition(s). See failures.csv for exact reasons.
