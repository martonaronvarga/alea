# Per-target workflow observations

Before/after entries compare sealed version2 with this experiment. They describe whole workflows, not an isolated cutoff effect. The separate regression/AR-K ablation and identical-phase one-step probes isolate those questions more closely.

Times are per-job medians. R-hat and ESS use unconstrained coordinates; mean/spread discrepancies use available empirical PosteriorDB reference parameters. A completed run is not a convergence certificate. Undefined quantities stay absent.

## earnings-earn_height

The pilot must learn coordinate scales before retention; difficult regression starts are directly checked in the ablation.

The baseline changed from historical HMC warmup to native NUTS warmup: completed 3/3 → 3/3; worst R-hat 4.567 → 1.003; minimum bulk ESS 4.272 → 1805; maximum mean error/reference SD 48.97 → 0.01648; maximum spread-ratio error 0.9999 → 0.02451.

| Workflow | Complete/planned | Process s | Retained s | Worst R-hat | Min bulk ESS | Mean error/ref SD | Spread-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | 3/3 | 31.94 | 20.68 | 1.003 | 1805 | 0.01648 | 0.02451 |
| nuts_cov_dense | 3/3 | 5.583 | 1.484 | 1.001 | 7546 | 0.01486 | 0.02341 |
| nuts_fisher_diag | 3/3 | 12.31 | 8.247 | 1.005 | 1147 | 0.02587 | 0.02161 |
| nuts_fisher_rank4 | 3/3 | 1.473 | 0.7565 | 1.001 | 1.091e+04 | 0.009481 | 0.01323 |
| atlas_full | 3/3 | 29.67 | 20.6 | 1.036 | 291 | 0.04873 | 0.02645 |
| stan_nuts_diag | 3/3 | 19.19 | 12.31 | 1.003 | 2064 | 0.02056 | 0.0193 |
| stan_nuts_dense | 3/3 | 4.122 | 1.097 | 1.001 | 6943 | 0.02083 | 0.0102 |
| nuts_rs | 3/3 | 7.604 | 5.177 | 1.003 | 871.5 | 0.04441 | 0.01329 |
| blackjax_nuts | 3/3 | 248.5 | 156.7 | 1.004 | 1804 | 0.02405 | 0.0154 |

## eight_schools-eight_schools_centered

Centered hierarchical geometry can retain bottlenecks after covariance adaptation; inspect divergences and between-chain agreement alongside timing.

The baseline changed from historical HMC warmup to native NUTS warmup: completed 3/3 → 3/3; worst R-hat 1.115 → 1.169; minimum bulk ESS 34.66 → 16.06; maximum mean error/reference SD — → —; maximum spread-ratio error — → —.

| Workflow | Complete/planned | Process s | Retained s | Worst R-hat | Min bulk ESS | Mean error/ref SD | Spread-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | 3/3 | 0.4309 | 0.1896 | 1.169 | 16.06 | — | — |
| nuts_cov_dense | 3/3 | 1.275 | 0.7486 | 1.035 | 125.9 | — | — |
| nuts_fisher_diag | 3/3 | 0.9009 | 0.2454 | 1.109 | 25.24 | — | — |
| nuts_fisher_rank4 | 3/3 | 0.4889 | 0.2033 | 1.036 | 111.4 | — | — |
| atlas_full | 3/3 | 1.385 | 0.5494 | 1.078 | 35.33 | — | — |
| stan_nuts_diag | 3/3 | 2.189 | 0.256 | 1.028 | 195.7 | — | — |
| stan_nuts_dense | 3/3 | 2.519 | 0.408 | 1.042 | 94.79 | — | — |
| nuts_rs | 3/3 | 0.2108 | 0.0932 | 1.017 | 151.6 | — | — |
| blackjax_nuts | 3/3 | 8.667 | 0.7147 | 1.036 | 134 | — | — |

## low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse

Mixture occupancy and label behavior can dominate between-chain diagnostics. Faster transitions alone do not establish mode exploration.

The baseline changed from historical HMC warmup to native NUTS warmup: completed 3/3 → 3/3; worst R-hat 1.335 → 1.092; minimum bulk ESS 9.701 → 35.44; maximum mean error/reference SD — → —; maximum spread-ratio error — → —.

| Workflow | Complete/planned | Process s | Retained s | Worst R-hat | Min bulk ESS | Mean error/ref SD | Spread-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | 3/3 | 72.63 | 50.01 | 1.092 | 35.44 | — | — |
| nuts_cov_dense | 3/3 | 76.26 | 56.8 | 1.486 | 7.553 | — | — |
| nuts_fisher_diag | 3/3 | 55.15 | 36.58 | 1.224 | 13.05 | — | — |
| nuts_fisher_rank4 | 3/3 | 40.22 | 24.8 | 1.531 | 7.169 | — | — |
| atlas_full | 3/3 | 123.6 | 93.61 | 1.36 | 9.364 | — | — |
| stan_nuts_diag | 3/3 | 56.65 | 37.83 | 1.126 | 26.27 | — | — |
| stan_nuts_dense | 3/3 | 45.27 | 27.97 | 1.587 | 6.766 | — | — |
| nuts_rs | 3/3 | 40.71 | 26.66 | 1.158 | 19.43 | — | — |
| blackjax_nuts | 3/3 | 185.6 | 122.9 | 1.129 | 22.06 | — | — |

## normal_2-normal_mixture

Mixture occupancy and label behavior can dominate between-chain diagnostics. Faster transitions alone do not establish mode exploration.

The baseline changed from historical HMC warmup to native NUTS warmup: completed 3/3 → 3/3; worst R-hat 1.532 → 1.529; minimum bulk ESS 7.246 → 7.112; maximum mean error/reference SD — → —; maximum spread-ratio error — → —.

| Workflow | Complete/planned | Process s | Retained s | Worst R-hat | Min bulk ESS | Mean error/ref SD | Spread-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | 3/3 | 10.75 | 7.47 | 1.529 | 7.112 | — | — |
| nuts_cov_dense | 3/3 | 10.71 | 7.173 | 1.529 | 7.115 | — | — |
| nuts_fisher_diag | 3/3 | 8.759 | 5.562 | 1.529 | 7.099 | — | — |
| nuts_fisher_rank4 | 3/3 | 10.9 | 6.848 | 1.529 | 7.099 | — | — |
| atlas_full | 3/3 | 72.44 | 62.95 | 1.531 | 7.116 | — | — |
| stan_nuts_diag | 3/3 | 12.79 | 7.804 | 1.529 | 7.106 | — | — |
| stan_nuts_dense | 3/3 | 12.42 | 7.689 | 1.53 | 7.12 | — | — |
| nuts_rs | 3/3 | 7.92 | 4.826 | 1.529 | 7.097 | — | — |
| blackjax_nuts | 3/3 | 35.16 | 19.26 | 1.529 | 7.101 | — | — |

## sblri-blr

The pilot must learn coordinate scales before retention; difficult regression starts are directly checked in the ablation.

The baseline changed from historical HMC warmup to native NUTS warmup: completed 3/3 → 3/3; worst R-hat 1.019 → 1.003; minimum bulk ESS 240.9 → 1444; maximum mean error/reference SD 16.26 → 0.03522; maximum spread-ratio error 157.1 → 0.01689.

| Workflow | Complete/planned | Process s | Retained s | Worst R-hat | Min bulk ESS | Mean error/ref SD | Spread-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | 3/3 | 0.3775 | 0.1752 | 1.003 | 1444 | 0.03522 | 0.01689 |
| nuts_cov_dense | 3/3 | 0.5075 | 0.2551 | 1.004 | 1512 | 0.0447 | 0.01224 |
| nuts_fisher_diag | 3/3 | 0.3458 | 0.1263 | 1.002 | 8020 | 0.02653 | 0.01563 |
| nuts_fisher_rank4 | 3/3 | 0.3286 | 0.1175 | 1.001 | 7013 | 0.02986 | 0.02467 |
| atlas_full | 3/3 | 1.711 | 1.409 | 1.005 | 1701 | 0.03255 | 0.03704 |
| stan_nuts_diag | 3/3 | 1.451 | 0.214 | 1.003 | 1465 | 0.02461 | 0.0141 |
| stan_nuts_dense | 3/3 | 1.466 | 0.229 | 1.006 | 1545 | 0.0328 | 0.01489 |
| nuts_rs | 3/3 | 0.1743 | 0.08684 | 1.002 | 8511 | 0.01728 | 0.01892 |
| blackjax_nuts | 3/3 | 39.22 | 20.47 | 1.004 | 1549 | 0.02197 | 0.02025 |

## seeds_data-seeds_stanified_model

The declared parameterization, starts and budgets are held in the manifest. Inspect inference fidelity and total cost together.

The baseline changed from historical HMC warmup to native NUTS warmup: completed 3/3 → 3/3; worst R-hat 1.069 → 1.044; minimum bulk ESS 59.77 → 168.4; maximum mean error/reference SD — → —; maximum spread-ratio error — → —.

| Workflow | Complete/planned | Process s | Retained s | Worst R-hat | Min bulk ESS | Mean error/ref SD | Spread-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | 3/3 | 1.543 | 0.9009 | 1.044 | 168.4 | — | — |
| nuts_cov_dense | 3/3 | 4.444 | 2.241 | 1.027 | 142.6 | — | — |
| nuts_fisher_diag | 3/3 | 1.169 | 0.6477 | 1.019 | 189.4 | — | — |
| nuts_fisher_rank4 | 3/3 | 1.617 | 0.8085 | 1.01 | 518 | — | — |
| atlas_full | 3/3 | 3.119 | 2.163 | 1.016 | 154.7 | — | — |
| stan_nuts_diag | 3/3 | 4.749 | 0.815 | 1.055 | 85.3 | — | — |
| stan_nuts_dense | 3/3 | 4.989 | 0.725 | 1.009 | 633.3 | — | — |
| nuts_rs | 3/3 | 0.7116 | 0.3969 | 1.013 | 387.9 | — | — |
| blackjax_nuts | 3/3 | 150 | 88.62 | 1.006 | 473.7 | — | — |
