# Per-target workflow observations

Before/after entries compare sealed version2 with this experiment. They describe whole workflows, not an isolated cutoff effect. The separate regression/AR-K ablation and identical-phase one-step probes isolate those questions more closely.

Times are per-job medians. R-hat and ESS use unconstrained coordinates; mean/spread discrepancies use available empirical PosteriorDB reference parameters. A completed run is not a convergence certificate. Undefined quantities stay absent.

## Rate_1_data-Rate_1_model

The declared parameterization, starts and budgets are held in the manifest. Inspect inference fidelity and total cost together.

The baseline changed from historical HMC warmup to native NUTS warmup: completed 3/3 → 3/3; worst R-hat 1.006 → 1.004; minimum bulk ESS 623 → 761; maximum mean error/reference SD 0.0206 → 0.03954; maximum spread-ratio error 0.01062 → 0.005838.

| Workflow | Complete/planned | Process s | Retained s | Worst R-hat | Min bulk ESS | Mean error/ref SD | Spread-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | 3/3 | 0.07831 | 0.004378 | 1.004 | 761 | 0.03954 | 0.005838 |
| nuts_cov_dense | 3/3 | 0.08875 | 0.007831 | 1.004 | 780.8 | 0.03954 | 0.005838 |
| nuts_fisher_diag | 3/3 | 0.0793 | 0.004265 | 1.005 | 771.8 | 0.02801 | 0.005494 |
| nuts_fisher_rank4 | 3/3 | 0.07421 | 0.005725 | 1.005 | 771.8 | 0.02801 | 0.005494 |
| nuts_fisher_rm | 3/3 | 0.06132 | 0.004714 | 1.006 | 773.2 | 0.02636 | 0.02272 |
| nuts_fisher_adam | 3/3 | 0.05914 | 0.003197 | 1.005 | 914.1 | 0.0351 | 0.01622 |
| nuts_transport | 3/3 | 0.02494 | 0.004419 | 1.008 | 727.2 | 0.03699 | 0.01415 |
| hmc_cov | 3/3 | 0.03597 | 0.006356 | 1.027 | 6782 | 0.004621 | 0.01212 |
| hmc_omf | 3/3 | 0.06278 | 0.01765 | 1.21 | 15.01 | 0.0145 | 0.06745 |
| hmc_bcss | 3/3 | 0.04867 | 0.01381 | 1.038 | 120.5 | 0.01428 | 0.03328 |
| ghmc | 3/3 | 0.03587 | 0.007038 | 1.048 | 6782 | 0.006407 | 0.03174 |
| drghmc | 3/3 | 0.02817 | 0.006726 | 1.015 | 6782 | 0.00455 | 0.01382 |
| mams | 0/3 | — | — | — | — | — | — |
| mclmc | 0/3 | — | — | — | — | — | — |
| chees | 3/3 | 0.0206 | 0.001971 | 1.004 | 1305 | 0.039 | 0.02062 |
| walnuts_d | 3/3 | 0.03112 | 0.01874 | 1.004 | 3472 | 0.008789 | 0.00402 |
| walnuts_r2p | 3/3 | 0.1489 | 0.01832 | 1.008 | 3068 | 0.008094 | 0.02096 |
| rmhmc_softabs | 3/3 | 0.07529 | 0.0356 | 3.448 | 5.017 | 0.647 | 0.6351 |
| rmhmc_manifold | 3/3 | 0.05881 | 0.03421 | 3.396 | 5.011 | 0.5465 | 0.5256 |
| stan_nuts_diag | 3/3 | 0.3262 | 0.015 | 1.007 | 601.8 | 0.01398 | 0.0174 |
| stan_nuts_dense | 3/3 | 0.3215 | 0.014 | 1.007 | 601.8 | 0.01398 | 0.0174 |
| stan_pathfinder_nuts | 3/3 | 0.51 | 0.023 | 1.006 | 638 | 0.0181 | 0.006444 |
| blackjax_nuts | 3/3 | 7.171 | 0.03325 | 1.005 | 720.9 | 0.04718 | 0.0126 |
| nuts_rs | 3/3 | 0.01593 | 0.002893 | 1.005 | 788.6 | 0.006852 | 0.02268 |
| nuts_saia_joint | 3/3 | 0.02129 | 0.002794 | 1.055 | 2501 | 0.01735 | 0.06341 |
| nuts_pathfinder_raw | 3/3 | 0.05923 | 0.004978 | 1.006 | 767.1 | 0.007322 | 0.01222 |
| nuts_pathfinder_psis | 3/3 | 0.04487 | 0.003283 | 1.005 | 741.4 | 0.009531 | 0.01187 |
| atlas_full | 3/3 | 0.04534 | 0.02463 | 1.03 | 2528 | 0.01625 | 0.03209 |

Unresolved outcomes: mams: 3 failed/unsupported repetition(s); mclmc: 3 failed/unsupported repetition(s). See failures.csv for exact reasons.

## Rate_2_data-Rate_2_model

The declared parameterization, starts and budgets are held in the manifest. Inspect inference fidelity and total cost together.

The baseline changed from historical HMC warmup to native NUTS warmup: completed 3/3 → 3/3; worst R-hat 1.006 → 1.004; minimum bulk ESS 1068 → 1447; maximum mean error/reference SD 0.04803 → 0.02145; maximum spread-ratio error 0.02105 → 0.03303.

| Workflow | Complete/planned | Process s | Retained s | Worst R-hat | Min bulk ESS | Mean error/ref SD | Spread-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | 3/3 | 0.06262 | 0.008227 | 1.004 | 1447 | 0.02145 | 0.03303 |
| nuts_cov_dense | 3/3 | 0.0847 | 0.01159 | 1.003 | 1307 | 0.03008 | 0.02414 |
| nuts_fisher_diag | 3/3 | 0.4704 | 0.007906 | 1.005 | 1815 | 0.02097 | 0.0129 |
| nuts_fisher_rank4 | 3/3 | 0.1082 | 0.007081 | 1.005 | 1815 | 0.02097 | 0.0129 |
| nuts_fisher_rm | 3/3 | 0.2311 | 0.01069 | 1.002 | 1265 | 0.03813 | 0.0306 |
| nuts_fisher_adam | 3/3 | 0.1443 | 0.01169 | 1.01 | 1857 | 0.02376 | 0.02674 |
| nuts_transport | 3/3 | 0.1072 | 0.01531 | 1.008 | 1302 | 0.0285 | 0.01872 |
| hmc_cov | 3/3 | 0.03094 | 0.01103 | 1.021 | 171.8 | 0.09566 | 0.04104 |
| hmc_omf | 3/3 | 0.04269 | 0.01903 | 1.072 | 80.62 | 0.07861 | 0.05001 |
| hmc_bcss | 3/3 | 0.04292 | 0.02079 | 1.081 | 39.43 | 0.05703 | 0.04498 |
| ghmc | 3/3 | 0.03074 | 0.01074 | 1.044 | 83.02 | 0.07655 | 0.01282 |
| drghmc | 3/3 | 0.03421 | 0.01241 | 1.019 | 823.1 | 0.02797 | 0.03665 |
| mams | 3/3 | 0.02805 | 0.003716 | 1.006 | 685.1 | 0.03163 | 0.02235 |
| mclmc | 3/3 | 0.04178 | 0.01917 | 1.002 | 2089 | 0.01753 | 0.03079 |
| chees | 3/3 | 0.08784 | 0.003395 | 1.005 | 835.1 | 0.03724 | 0.02387 |
| walnuts_d | 3/3 | 0.06772 | 0.02585 | 1.003 | 1734 | 0.04131 | 0.01529 |
| walnuts_r2p | 3/3 | 0.05171 | 0.03551 | 1.005 | 1771 | 0.02975 | 0.01205 |
| rmhmc_softabs | 3/3 | 0.1491 | 0.07758 | 3.639 | 4.964 | 0.655 | 0.7786 |
| rmhmc_manifold | 3/3 | 0.1302 | 0.08547 | 3.815 | 4.928 | 0.5838 | 0.7579 |
| stan_nuts_diag | 3/3 | 0.4665 | 0.026 | 1.005 | 1419 | 0.02838 | 0.01297 |
| stan_nuts_dense | 3/3 | 0.5681 | 0.03 | 1.008 | 1504 | 0.02206 | 0.02622 |
| stan_pathfinder_nuts | 3/3 | 0.6683 | 0.038 | 1.004 | 1407 | 0.02802 | 0.0167 |
| blackjax_nuts | 3/3 | 5.874 | 0.04104 | 1.004 | 1514 | 0.03198 | 0.02877 |
| nuts_rs | 3/3 | 0.02096 | 0.004888 | 1.007 | 1971 | 0.0423 | 0.0073 |
| nuts_saia_joint | 3/3 | 0.02844 | 0.005544 | 1.071 | 2559 | 0.02614 | 0.06532 |
| nuts_pathfinder_raw | 3/3 | 0.05871 | 0.008303 | 1.005 | 1222 | 0.03867 | 0.02266 |
| nuts_pathfinder_psis | 3/3 | 0.05647 | 0.007355 | 1.006 | 1440 | 0.05135 | 0.03147 |
| atlas_full | 3/3 | 0.1688 | 0.04118 | 1.025 | 2094 | 0.0284 | 0.04073 |

## Rate_5_data-Rate_5_model

The declared parameterization, starts and budgets are held in the manifest. Inspect inference fidelity and total cost together.

The baseline changed from historical HMC warmup to native NUTS warmup: completed 3/3 → 3/3; worst R-hat 1.007 → 1.004; minimum bulk ESS 711.2 → 759.8; maximum mean error/reference SD 0.04305 → 0.004691; maximum spread-ratio error 0.01219 → 0.008835.

| Workflow | Complete/planned | Process s | Retained s | Worst R-hat | Min bulk ESS | Mean error/ref SD | Spread-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | 3/3 | 0.04321 | 0.003708 | 1.004 | 759.8 | 0.004691 | 0.008835 |
| nuts_cov_dense | 3/3 | 0.04729 | 0.005422 | 1.004 | 779.7 | 0.004691 | 0.01543 |
| nuts_fisher_diag | 3/3 | 0.1349 | 0.003469 | 1.006 | 688.7 | 0.02703 | 0.01137 |
| nuts_fisher_rank4 | 3/3 | 0.04509 | 0.003721 | 1.006 | 688.7 | 0.02703 | 0.01137 |
| nuts_fisher_rm | 3/3 | 0.04994 | 0.005641 | 1.003 | 671.9 | 0.009089 | 0.01181 |
| nuts_fisher_adam | 3/3 | 0.07157 | 0.004583 | 1.005 | 938.4 | 0.05567 | 0.01164 |
| nuts_transport | 3/3 | 0.02336 | 0.005524 | 1.011 | 686.4 | 0.04574 | 0.01576 |
| hmc_cov | 3/3 | 0.02889 | 0.008369 | 1.045 | 6782 | 0.002649 | 0.1188 |
| hmc_omf | 3/3 | 0.03573 | 0.0169 | 1.067 | 47.82 | 0.06329 | 0.02206 |
| hmc_bcss | 3/3 | 0.03128 | 0.01437 | 1.086 | 60.4 | 0.06147 | 0.04232 |
| ghmc | 3/3 | 0.02149 | 0.00696 | 1.047 | 6782 | 0.01639 | 0.0596 |
| drghmc | 3/3 | 0.02356 | 0.008026 | 1.034 | 6782 | 0.005306 | 0.05588 |
| mams | 0/3 | — | — | — | — | — | — |
| mclmc | 0/3 | — | — | — | — | — | — |
| chees | 3/3 | 0.01596 | 0.001411 | 1.004 | 1242 | 0.02716 | 0.01814 |
| walnuts_d | 3/3 | 0.04573 | 0.01308 | 1.003 | 2335 | 0.003681 | 0.003103 |
| walnuts_r2p | 3/3 | 0.06204 | 0.01619 | 1.001 | 2818 | 0.00595 | 0.009243 |
| rmhmc_softabs | 3/3 | 0.0466 | 0.02339 | 3.374 | 5.034 | 0.8912 | 1.158 |
| rmhmc_manifold | 3/3 | 0.03495 | 0.01964 | 3.628 | 4.95 | 0.6988 | 1.008 |
| stan_nuts_diag | 3/3 | 0.3813 | 0.021 | 1.004 | 796.2 | 0.08082 | 0.01335 |
| stan_nuts_dense | 3/3 | 0.3205 | 0.019 | 1.004 | 796.2 | 0.08082 | 0.01335 |
| stan_pathfinder_nuts | 3/3 | 0.2889 | 0.016 | 1.006 | 662.8 | 0.03576 | 0.005456 |
| blackjax_nuts | 3/3 | 7.813 | 1.227 | 1.009 | 717.9 | 0.05669 | 0.0191 |
| nuts_rs | 3/3 | 0.01468 | 0.003113 | 1.004 | 913.7 | 0.03246 | 0.003427 |
| nuts_saia_joint | 3/3 | 0.01981 | 0.003073 | 1.164 | 3110 | 0.009988 | 0.1524 |
| nuts_pathfinder_raw | 3/3 | 0.04516 | 0.003329 | 1.005 | 694.6 | 0.04512 | 0.02689 |
| nuts_pathfinder_psis | 3/3 | 0.2789 | 0.003524 | 1.011 | 701.6 | 0.04512 | 0.02318 |
| atlas_full | 3/3 | 0.04891 | 0.02756 | 1.012 | 2212 | 0.008134 | 0.02545 |

Unresolved outcomes: mams: 3 failed/unsupported repetition(s); mclmc: 3 failed/unsupported repetition(s). See failures.csv for exact reasons.

## arK-arK

AR-K separates startup geometry adaptation from retained trajectory cost; its starts also appear in the controlled ablation.

The baseline changed from historical HMC warmup to native NUTS warmup: completed 3/3 → 3/3; worst R-hat 1.007 → 1.005; minimum bulk ESS 989.2 → 1114; maximum mean error/reference SD 0.04259 → 0.04387; maximum spread-ratio error 0.05467 → 0.04399.

| Workflow | Complete/planned | Process s | Retained s | Worst R-hat | Min bulk ESS | Mean error/ref SD | Spread-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | 3/3 | 5.976 | 2.913 | 1.005 | 1114 | 0.04387 | 0.04399 |
| nuts_cov_dense | 3/3 | 2.084 | 0.6675 | 1.005 | 2243 | 0.03574 | 0.04026 |
| nuts_fisher_diag | 3/3 | 5.308 | 2.612 | 1.012 | 923.8 | 0.07707 | 0.03803 |
| nuts_fisher_rank4 | 3/3 | 1.637 | 0.5772 | 1.006 | 2205 | 0.03155 | 0.0345 |
| nuts_fisher_rm | 3/3 | 18.6 | 9.071 | 1.329 | 14.04 | 0.2806 | 0.1925 |
| nuts_fisher_adam | 3/3 | 10.8 | 4.408 | 2.36 | 5.856 | 0.5859 | 0.3709 |
| nuts_transport | 3/3 | 4.681 | 2.091 | 1.008 | 1462 | 0.03643 | 0.02764 |
| hmc_cov | 3/3 | 2.955 | 0.6606 | 1.011 | 327.5 | 0.08242 | 0.05576 |
| hmc_omf | 3/3 | 4.385 | 1.334 | 1.005 | 744.5 | 0.05497 | 0.04287 |
| hmc_bcss | 3/3 | 4.134 | 1.246 | 1.005 | 911.3 | 0.04581 | 0.04194 |
| ghmc | 3/3 | 3.261 | 0.7414 | 1.012 | 823 | 0.03438 | 0.05015 |
| drghmc | 3/3 | 3.053 | 0.7217 | 1.007 | 850.7 | 0.03026 | 0.05341 |
| mams | 3/3 | 3.739 | 0.8075 | 1.008 | 327.3 | 0.1001 | 0.05539 |
| mclmc | 3/3 | 4.681 | 1.054 | 1.022 | 292.4 | 0.0661 | 0.04073 |
| chees | 3/3 | 3.475 | 0.6971 | 1.044 | 86.04 | 0.09651 | 0.06718 |
| walnuts_d | 3/3 | 13.38 | 9.795 | 1.035 | 216.9 | 0.08943 | 0.06445 |
| walnuts_r2p | 3/3 | 12.55 | 9.291 | 1.016 | 421.1 | 0.06317 | 0.06355 |
| rmhmc_softabs | 0/3 | — | — | — | — | — | — |
| rmhmc_manifold | 0/3 | — | — | — | — | — | — |
| stan_nuts_diag | 3/3 | 6.912 | 2.864 | 1.008 | 1198 | 0.03659 | 0.04054 |
| stan_nuts_dense | 3/3 | 2.705 | 0.668 | 1.008 | 2461 | 0.04141 | 0.04558 |
| stan_pathfinder_nuts | 3/3 | 6.459 | 2.536 | 1.005 | 1185 | 0.05562 | 0.04332 |
| blackjax_nuts | 3/3 | 77.13 | 28.54 | 1.008 | 1084 | 0.04627 | 0.03043 |
| nuts_rs | 3/3 | 3.821 | 1.738 | 1.008 | 786.1 | 0.06606 | 0.04958 |
| nuts_saia_joint | 3/3 | 7.624 | 3.076 | 1.005 | 1048 | 0.02562 | 0.04464 |
| nuts_pathfinder_raw | 3/3 | 6.436 | 3.048 | 1.006 | 1193 | 0.03479 | 0.04328 |
| nuts_pathfinder_psis | 3/3 | 6.238 | 2.772 | 1.006 | 1234 | 0.03725 | 0.03307 |
| atlas_full | 3/3 | 10.76 | 5.236 | 1.031 | 579.7 | 0.04193 | 0.05838 |

Unresolved outcomes: rmhmc_softabs: 3 failed/unsupported repetition(s); rmhmc_manifold: 3 failed/unsupported repetition(s). See failures.csv for exact reasons.

## arma-arma11

The declared parameterization, starts and budgets are held in the manifest. Inspect inference fidelity and total cost together.

The baseline changed from historical HMC warmup to native NUTS warmup: completed 3/3 → 3/3; worst R-hat 1.007 → 1.003; minimum bulk ESS 1877 → 1629; maximum mean error/reference SD 0.03084 → 0.01805; maximum spread-ratio error 0.02157 → 0.02586.

| Workflow | Complete/planned | Process s | Retained s | Worst R-hat | Min bulk ESS | Mean error/ref SD | Spread-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | 3/3 | 0.513 | 0.2159 | 1.003 | 1629 | 0.01805 | 0.02586 |
| nuts_cov_dense | 3/3 | 0.5074 | 0.207 | 1.002 | 1688 | 0.02161 | 0.04305 |
| nuts_fisher_diag | 3/3 | 0.5618 | 0.1783 | 1.004 | 1813 | 0.02139 | 0.03194 |
| nuts_fisher_rank4 | 3/3 | 0.494 | 0.1721 | 1.006 | 1599 | 0.02284 | 0.02804 |
| nuts_fisher_rm | 3/3 | 11.38 | 4.946 | 1.723 | 6.287 | 31 | 52.68 |
| nuts_fisher_adam | 3/3 | 6.257 | 2.236 | 2.254 | 5.443 | 31.06 | 52.65 |
| nuts_transport | 3/3 | 0.5656 | 0.22 | 1.004 | 1680 | 0.01896 | 0.0373 |
| hmc_cov | 3/3 | 0.5599 | 0.2687 | 1.171 | 20.63 | 0.1284 | 0.071 |
| hmc_omf | 3/3 | 1.14 | 0.5943 | 1.039 | 200.6 | 0.03938 | 0.05851 |
| hmc_bcss | 3/3 | 0.929 | 0.5291 | 1.069 | 40.72 | 0.04597 | 0.03084 |
| ghmc | 3/3 | 0.582 | 0.2729 | 1.176 | 22.29 | 0.09952 | 0.02551 |
| drghmc | 3/3 | 0.6662 | 0.33 | 1.292 | 151.4 | 0.05062 | 0.1495 |
| mams | 3/3 | 0.2796 | 0.05881 | 1.003 | 717.7 | 0.02535 | 0.03263 |
| mclmc | 3/3 | 0.5486 | 0.2824 | 1.003 | 2154 | 0.03121 | 0.01893 |
| chees | 3/3 | 0.2356 | 0.0481 | 1.01 | 985.9 | 0.04987 | 0.03623 |
| walnuts_d | 3/3 | 1.006 | 0.7806 | 1.007 | 1465 | 0.02306 | 0.05006 |
| walnuts_r2p | 3/3 | 1.321 | 1.069 | 1.004 | 1663 | 0.03966 | 0.03352 |
| rmhmc_softabs | 0/3 | — | — | — | — | — | — |
| rmhmc_manifold | 0/3 | — | — | — | — | — | — |
| stan_nuts_diag | 3/3 | 0.7542 | 0.171 | 1.003 | 1715 | 0.03375 | 0.0251 |
| stan_nuts_dense | 3/3 | 0.8738 | 0.193 | 1.005 | 1826 | 0.04156 | 0.03097 |
| stan_pathfinder_nuts | 0/3 | — | — | — | — | — | — |
| blackjax_nuts | 3/3 | 16.31 | 3.8 | 1.005 | 1648 | 0.03295 | 0.03134 |
| nuts_rs | 3/3 | 0.2808 | 0.1147 | 1.004 | 2304 | 0.02213 | 0.02171 |
| nuts_saia_joint | 3/3 | 0.7771 | 0.2332 | 1.046 | 183.2 | 0.04185 | 0.03701 |
| nuts_pathfinder_raw | 0/3 | — | — | — | — | — | — |
| nuts_pathfinder_psis | 0/3 | — | — | — | — | — | — |
| atlas_full | 3/3 | 1.463 | 1 | 1.013 | 1110 | 0.02163 | 0.04732 |

Unresolved outcomes: rmhmc_softabs: 3 failed/unsupported repetition(s); rmhmc_manifold: 3 failed/unsupported repetition(s); stan_pathfinder_nuts: 3 failed/unsupported repetition(s); nuts_pathfinder_raw: 3 failed/unsupported repetition(s); nuts_pathfinder_psis: 3 failed/unsupported repetition(s). See failures.csv for exact reasons.

## dogs-dogs_log

The declared parameterization, starts and budgets are held in the manifest. Inspect inference fidelity and total cost together.

The baseline changed from historical HMC warmup to native NUTS warmup: completed 3/3 → 3/3; worst R-hat 1.013 → 1.009; minimum bulk ESS 436.6 → 566.9; maximum mean error/reference SD — → —; maximum spread-ratio error — → —.

| Workflow | Complete/planned | Process s | Retained s | Worst R-hat | Min bulk ESS | Mean error/ref SD | Spread-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | 3/3 | 3.37 | 1.449 | 1.009 | 566.9 | — | — |
| nuts_cov_dense | 3/3 | 2.93 | 1.19 | 1.007 | 1106 | — | — |
| nuts_fisher_diag | 3/3 | 3.436 | 1.435 | 1.012 | 459.3 | — | — |
| nuts_fisher_rank4 | 3/3 | 2.736 | 1.278 | 1.004 | 1425 | — | — |
| nuts_fisher_rm | 3/3 | 37.64 | 16.28 | 1.011 | 407.8 | — | — |
| nuts_fisher_adam | 3/3 | 21.27 | 7.06 | 1.018 | 361.1 | — | — |
| nuts_transport | 3/3 | 2.637 | 1.089 | 1.006 | 1037 | — | — |
| hmc_cov | 3/3 | 3.799 | 2.028 | 1.039 | 1329 | — | — |
| hmc_omf | 3/3 | 7.023 | 4.213 | 1.046 | 1222 | — | — |
| hmc_bcss | 3/3 | 7.711 | 4.447 | 1.043 | 398.3 | — | — |
| ghmc | 3/3 | 5.95 | 3.308 | 1.083 | 2864 | — | — |
| drghmc | 3/3 | 6.26 | 3.49 | 1.026 | 3001 | — | — |
| mams | 3/3 | 3.899 | 0.808 | 1.006 | 320.8 | — | — |
| mclmc | 0/3 | — | — | — | — | — | — |
| chees | 3/3 | 2.201 | 0.6098 | 1.015 | 403 | — | — |
| walnuts_d | 3/3 | 11.04 | 8.139 | 1.007 | 574.1 | — | — |
| walnuts_r2p | 3/3 | 14.9 | 8.518 | 1.011 | 698.6 | — | — |
| rmhmc_softabs | 0/3 | — | — | — | — | — | — |
| rmhmc_manifold | 0/3 | — | — | — | — | — | — |
| stan_nuts_diag | 3/3 | 38.82 | 3.275 | 1.011 | 512.9 | — | — |
| stan_nuts_dense | 3/3 | 36.91 | 3.076 | 1.005 | 997.3 | — | — |
| stan_pathfinder_nuts | 3/3 | 32.03 | 3.12 | 1.007 | 536.3 | — | — |
| blackjax_nuts | 3/3 | 16.28 | 4.783 | 1.018 | 508.7 | — | — |
| nuts_rs | 3/3 | 2.43 | 1.173 | 1.012 | 536.1 | — | — |
| nuts_saia_joint | 3/3 | 3.535 | 1.558 | 1.007 | 563.2 | — | — |
| nuts_pathfinder_raw | 3/3 | 4.816 | 1.969 | 1.008 | 558.6 | — | — |
| nuts_pathfinder_psis | 3/3 | 3.983 | 1.584 | 1.009 | 587.4 | — | — |
| atlas_full | 3/3 | 11.53 | 8.341 | 1.017 | 1078 | — | — |

Unresolved outcomes: mclmc: 3 failed/unsupported repetition(s); rmhmc_softabs: 3 failed/unsupported repetition(s); rmhmc_manifold: 3 failed/unsupported repetition(s). See failures.csv for exact reasons.

## earnings-earn_height

The pilot must learn coordinate scales before retention; difficult regression starts are directly checked in the ablation.

The baseline changed from historical HMC warmup to native NUTS warmup: completed 3/3 → 3/3; worst R-hat 4.567 → 1.007; minimum bulk ESS 4.272 → 534.5; maximum mean error/reference SD 48.97 → 0.04146; maximum spread-ratio error 0.9999 → 0.05819.

| Workflow | Complete/planned | Process s | Retained s | Worst R-hat | Min bulk ESS | Mean error/ref SD | Spread-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | 3/3 | 10.96 | 4.801 | 1.007 | 534.5 | 0.04146 | 0.05819 |
| nuts_cov_dense | 3/3 | 3.843 | 0.3715 | 1.004 | 1654 | 0.01715 | 0.01823 |
| nuts_fisher_diag | 3/3 | 5.213 | 2.374 | 1.016 | 318.9 | 0.0269 | 0.04967 |
| nuts_fisher_rank4 | 3/3 | 0.9449 | 0.2384 | 1.003 | 2149 | 0.0218 | 0.0242 |
| nuts_fisher_rm | 3/3 | 3.802 | 0.9492 | 4.567 | 4.272 | 48.96 | 0.9999 |
| nuts_fisher_adam | 3/3 | 5.415 | 0.8623 | 4.567 | 4.272 | 48.95 | 0.9999 |
| nuts_transport | 3/3 | 8.743 | 3.702 | 1.008 | 369.1 | 0.04793 | 0.02851 |
| hmc_cov | 3/3 | 4.452 | 0.623 | 1.138 | 22.6 | 0.2666 | 0.09031 |
| hmc_omf | 3/3 | 9.716 | 0.9873 | 3.492 | 4.46 | 6.176 | 0.722 |
| hmc_bcss | 3/3 | 5.229 | 0.8893 | 1.095 | 36.15 | 0.1685 | 0.08844 |
| ghmc | 3/3 | 3.376 | 0.4673 | 1.037 | 107.7 | 0.1185 | 0.0578 |
| drghmc | 3/3 | 3.499 | 0.4924 | 1.057 | 115.1 | 0.161 | 0.03745 |
| mams | 3/3 | 11.15 | 6.035 | 1.008 | 1219 | 0.03875 | 0.03369 |
| mclmc | 3/3 | 3.82 | 0.4997 | 1.037 | 105.9 | 0.121 | 0.1012 |
| chees | 3/3 | 3.527 | 0.4139 | 1.387 | 9.104 | 0.2898 | 0.05896 |
| walnuts_d | 3/3 | 6.088 | 1.804 | 1.145 | 20.12 | 0.1608 | 0.1208 |
| walnuts_r2p | 3/3 | 7.105 | 2.493 | 1.096 | 44.87 | 0.1696 | 0.1072 |
| rmhmc_softabs | 0/3 | — | — | — | — | — | — |
| rmhmc_manifold | 0/3 | — | — | — | — | — | — |
| stan_nuts_diag | 3/3 | 7.061 | 3.08 | 1.011 | 476.4 | 0.0159 | 0.02943 |
| stan_nuts_dense | 3/3 | 2.763 | 0.307 | 1.006 | 1669 | 0.02956 | 0.02219 |
| stan_pathfinder_nuts | 0/3 | — | — | — | — | — | — |
| blackjax_nuts | 3/3 | 110.6 | 45.86 | 1.007 | 521 | 0.04286 | 0.05653 |
| nuts_rs | 3/3 | 3.581 | 1.876 | 1.029 | 258.1 | 0.05724 | 0.0462 |
| nuts_saia_joint | 3/3 | 36.22 | 29.28 | 3.322 | 4.48 | 29.4 | 0.9999 |
| nuts_pathfinder_raw | 3/3 | 9.221 | 4.321 | 1.007 | 664.5 | 0.02726 | 0.03806 |
| nuts_pathfinder_psis | 3/3 | 9.354 | 4.12 | 1.009 | 655.2 | 0.06232 | 0.03223 |
| atlas_full | 3/3 | 17.63 | 6.805 | 1.105 | 39.59 | 0.04341 | 0.0682 |

Unresolved outcomes: rmhmc_softabs: 3 failed/unsupported repetition(s); rmhmc_manifold: 3 failed/unsupported repetition(s); stan_pathfinder_nuts: 3 failed/unsupported repetition(s). See failures.csv for exact reasons.

## earnings-logearn_interaction_z

The declared parameterization, starts and budgets are held in the manifest. Inspect inference fidelity and total cost together.

The baseline changed from historical HMC warmup to native NUTS warmup: completed 3/3 → 3/3; worst R-hat 1.005 → 1.005; minimum bulk ESS 1029 → 872.8; maximum mean error/reference SD 0.03289 → 0.04392; maximum spread-ratio error 0.0428 → 0.02775.

| Workflow | Complete/planned | Process s | Retained s | Worst R-hat | Min bulk ESS | Mean error/ref SD | Spread-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | 3/3 | 4.015 | 1.939 | 1.005 | 872.8 | 0.04392 | 0.02775 |
| nuts_cov_dense | 3/3 | 2.377 | 1.028 | 1.005 | 2059 | 0.03396 | 0.03581 |
| nuts_fisher_diag | 3/3 | 3.229 | 1.527 | 1.011 | 707.6 | 0.03845 | 0.0281 |
| nuts_fisher_rank4 | 3/3 | 1.854 | 0.7855 | 1.004 | 1601 | 0.04093 | 0.02671 |
| nuts_fisher_rm | 3/3 | 59.82 | 33.42 | 1.677 | 6.446 | 42.69 | 74.54 |
| nuts_fisher_adam | 3/3 | 44.96 | 22.46 | 1.606 | 6.778 | 83.61 | 148.3 |
| nuts_transport | 3/3 | 5.773 | 1.916 | 1.007 | 1220 | 0.04347 | 0.03076 |
| hmc_cov | 3/3 | 3.879 | 1.532 | 1.033 | 1287 | 0.05236 | 0.03022 |
| hmc_omf | 3/3 | 5.553 | 2.636 | 1.091 | 916 | 0.06409 | 0.09268 |
| hmc_bcss | 3/3 | 4.913 | 2.576 | 1.035 | 1711 | 0.04537 | 0.04068 |
| ghmc | 3/3 | 3.567 | 1.251 | 1.073 | 2221 | 0.03945 | 0.06231 |
| drghmc | 3/3 | 3.555 | 1.631 | 1.073 | 2212 | 0.03371 | 0.03985 |
| mams | 3/3 | 1.995 | 0.4513 | 1.018 | 249.3 | 0.09739 | 0.02601 |
| mclmc | 3/3 | 3.124 | 1.282 | 1.009 | 529.9 | 0.04613 | 0.02917 |
| chees | 3/3 | 1.658 | 0.3547 | 1.024 | 189.3 | 0.04409 | 0.0326 |
| walnuts_d | 3/3 | 7.908 | 5.74 | 1.007 | 465.6 | 0.0513 | 0.036 |
| walnuts_r2p | 3/3 | 8.936 | 7.106 | 1.014 | 568.3 | 0.03547 | 0.03019 |
| rmhmc_softabs | 0/3 | — | — | — | — | — | — |
| rmhmc_manifold | 0/3 | — | — | — | — | — | — |
| stan_nuts_diag | 3/3 | 4.254 | 1.735 | 1.006 | 974.2 | 0.06897 | 0.03029 |
| stan_nuts_dense | 3/3 | 2.8 | 0.964 | 1.007 | 1819 | 0.01635 | 0.03216 |
| stan_pathfinder_nuts | 3/3 | 3.741 | 1.503 | 1.006 | 997.8 | 0.03982 | 0.02481 |
| blackjax_nuts | 3/3 | 33.58 | 11.39 | 1.006 | 886.9 | 0.03332 | 0.04734 |
| nuts_rs | 3/3 | 2.77 | 1.276 | 1.004 | 717.4 | 0.05839 | 0.0356 |
| nuts_saia_joint | 3/3 | 5.984 | 2.069 | 1.006 | 777.4 | 0.06274 | 0.03891 |
| nuts_pathfinder_raw | 3/3 | 4.198 | 1.781 | 1.005 | 959.1 | 0.05989 | 0.04107 |
| nuts_pathfinder_psis | 3/3 | 4.526 | 1.963 | 1.006 | 1006 | 0.03213 | 0.03603 |
| atlas_full | 3/3 | 8.082 | 5.484 | 1.028 | 877.9 | 0.05569 | 0.04262 |

Unresolved outcomes: rmhmc_softabs: 3 failed/unsupported repetition(s); rmhmc_manifold: 3 failed/unsupported repetition(s). See failures.csv for exact reasons.

## eight_schools-eight_schools_centered

Centered hierarchical geometry can retain bottlenecks after covariance adaptation; inspect divergences and between-chain agreement alongside timing.

The baseline changed from historical HMC warmup to native NUTS warmup: completed 3/3 → 3/3; worst R-hat 1.115 → 1.063; minimum bulk ESS 34.66 → 70.08; maximum mean error/reference SD — → —; maximum spread-ratio error — → —.

| Workflow | Complete/planned | Process s | Retained s | Worst R-hat | Min bulk ESS | Mean error/ref SD | Spread-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | 3/3 | 0.2786 | 0.06727 | 1.063 | 70.08 | — | — |
| nuts_cov_dense | 3/3 | 0.6047 | 0.1556 | 1.048 | 97.53 | — | — |
| nuts_fisher_diag | 3/3 | 0.1639 | 0.05469 | 1.343 | 10.04 | — | — |
| nuts_fisher_rank4 | 3/3 | 0.1937 | 0.05276 | 1.231 | 13 | — | — |
| nuts_fisher_rm | 3/3 | 0.183 | 0.06228 | 1.067 | 43.21 | — | — |
| nuts_fisher_adam | 3/3 | 0.2452 | 0.06391 | 1.043 | 84.44 | — | — |
| nuts_transport | 3/3 | 0.2393 | 0.08068 | 1.055 | 88.48 | — | — |
| hmc_cov | 3/3 | 0.1294 | 0.01616 | 1.112 | 25.45 | — | — |
| hmc_omf | 3/3 | 0.1295 | 0.02933 | 1.05 | 65.54 | — | — |
| hmc_bcss | 3/3 | 0.5595 | 0.04504 | 1.255 | 11.88 | — | — |
| ghmc | 3/3 | 0.1964 | 0.02563 | 1.163 | 18.72 | — | — |
| drghmc | 3/3 | 0.1506 | 0.02306 | 1.103 | 46.67 | — | — |
| mams | 3/3 | 0.1299 | 0.01932 | 1.032 | 82.9 | — | — |
| mclmc | 3/3 | 0.1416 | 0.03832 | 1.131 | 24.51 | — | — |
| chees | 3/3 | 0.1009 | 0.0127 | 1.092 | 34.78 | — | — |
| walnuts_d | 3/3 | 0.3251 | 0.1595 | 1.14 | 22.52 | — | — |
| walnuts_r2p | 3/3 | 0.3959 | 0.2611 | 1.141 | 21.29 | — | — |
| rmhmc_softabs | 3/3 | 5.336 | 3.302 | 3.953 | 4.9 | — | — |
| rmhmc_manifold | 3/3 | 4.692 | 3.215 | 3.909 | 4.899 | — | — |
| stan_nuts_diag | 3/3 | 1.41 | 0.072 | 1.183 | 16.07 | — | — |
| stan_nuts_dense | 3/3 | 1.043 | 0.112 | 1.102 | 27.1 | — | — |
| stan_pathfinder_nuts | 3/3 | 0.8774 | 0.071 | 1.079 | 48.6 | — | — |
| blackjax_nuts | 3/3 | 7.618 | 0.1655 | 1.327 | 10.51 | — | — |
| nuts_rs | 3/3 | 0.09446 | 0.03718 | 1.273 | 12.02 | — | — |
| nuts_saia_joint | 3/3 | 0.3184 | 0.0767 | 1.039 | 109 | — | — |
| nuts_pathfinder_raw | 3/3 | 0.2544 | 0.07663 | 1.15 | 19.07 | — | — |
| nuts_pathfinder_psis | 3/3 | 0.2571 | 0.07271 | 1.074 | 47.28 | — | — |
| atlas_full | 3/3 | 0.3874 | 0.233 | 1.298 | 13.37 | — | — |

## eight_schools-eight_schools_noncentered

This uses the declared noncentered coordinates; do not infer that results transfer to the centered parameterization.

The baseline changed from historical HMC warmup to native NUTS warmup: completed 3/3 → 3/3; worst R-hat 1.006 → 1.005; minimum bulk ESS 1110 → 1167; maximum mean error/reference SD 0.04614 → 0.03332; maximum spread-ratio error 0.05051 → 0.04551.

| Workflow | Complete/planned | Process s | Retained s | Worst R-hat | Min bulk ESS | Mean error/ref SD | Spread-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | 3/3 | 0.1292 | 0.0266 | 1.005 | 1167 | 0.03332 | 0.04551 |
| nuts_cov_dense | 3/3 | 0.28 | 0.08937 | 1.008 | 1296 | 0.03487 | 0.04847 |
| nuts_fisher_diag | 3/3 | 0.1082 | 0.0246 | 1.006 | 1084 | 0.03575 | 0.02651 |
| nuts_fisher_rank4 | 3/3 | 0.1169 | 0.02509 | 1.008 | 1147 | 0.03559 | 0.05689 |
| nuts_fisher_rm | 3/3 | 0.2118 | 0.02878 | 1.01 | 834.4 | 0.02643 | 0.04805 |
| nuts_fisher_adam | 3/3 | 0.1042 | 0.02387 | 1.006 | 1031 | 0.03409 | 0.048 |
| nuts_transport | 3/3 | 0.1078 | 0.0389 | 1.004 | 948.8 | 0.03243 | 0.05404 |
| hmc_cov | 3/3 | 0.1343 | 0.0198 | 1.08 | 1286 | 0.04099 | 0.06469 |
| hmc_omf | 3/3 | 0.1456 | 0.04065 | 1.039 | 190.5 | 0.06205 | 0.05919 |
| hmc_bcss | 3/3 | 0.1026 | 0.03842 | 1.157 | 17.96 | 0.08544 | 0.07149 |
| ghmc | 3/3 | 0.1441 | 0.02266 | 1.086 | 925.3 | 0.03943 | 0.05589 |
| drghmc | 3/3 | 0.08065 | 0.02328 | 1.037 | 1105 | 0.0316 | 0.05953 |
| mams | 3/3 | 0.07284 | 0.009919 | 1.017 | 401.2 | 0.115 | 0.07901 |
| mclmc | 3/3 | 0.09538 | 0.03447 | 1.003 | 1136 | 0.03623 | 0.04782 |
| chees | 3/3 | 0.4125 | 0.02854 | 1.023 | 234.7 | 0.0697 | 0.04682 |
| walnuts_d | 3/3 | 0.1181 | 0.06021 | 1.01 | 867 | 0.04393 | 0.05621 |
| walnuts_r2p | 3/3 | 0.1589 | 0.0975 | 1.007 | 994.4 | 0.03551 | 0.05413 |
| rmhmc_softabs | 3/3 | 4.693 | 3.108 | 3.885 | 4.908 | 1.312 | 0.7363 |
| rmhmc_manifold | 3/3 | 5.317 | 3.434 | 3.841 | 4.916 | 1.357 | 0.7692 |
| stan_nuts_diag | 3/3 | 0.9853 | 0.052 | 1.008 | 1026 | 0.04322 | 0.0637 |
| stan_nuts_dense | 3/3 | 1.192 | 0.058 | 1.007 | 1136 | 0.04336 | 0.04683 |
| stan_pathfinder_nuts | 3/3 | 0.9221 | 0.053 | 1.006 | 1087 | 0.02621 | 0.05166 |
| blackjax_nuts | 3/3 | 6.14 | 0.1096 | 1.005 | 997.3 | 0.02635 | 0.0442 |
| nuts_rs | 3/3 | 0.06632 | 0.01959 | 1.01 | 808.3 | 0.03521 | 0.06125 |
| nuts_saia_joint | 3/3 | 0.1773 | 0.03027 | 1.009 | 1273 | 0.04428 | 0.07065 |
| nuts_pathfinder_raw | 3/3 | 0.1349 | 0.03297 | 1.007 | 1148 | 0.03337 | 0.04379 |
| nuts_pathfinder_psis | 3/3 | 0.2554 | 0.03407 | 1.006 | 1282 | 0.05267 | 0.06381 |
| atlas_full | 3/3 | 0.1912 | 0.09902 | 1.019 | 574.8 | 0.06274 | 0.1011 |

## garch-garch11

Trajectory and curvature work can be expensive relative to a single gradient. Actual derivative calls remain separate from integration counts.

The baseline changed from historical HMC warmup to native NUTS warmup: completed 3/3 → 3/3; worst R-hat 1.007 → 1.009; minimum bulk ESS 870.6 → 829.9; maximum mean error/reference SD 0.04457 → 0.03412; maximum spread-ratio error 0.05185 → 0.02951.

| Workflow | Complete/planned | Process s | Retained s | Worst R-hat | Min bulk ESS | Mean error/ref SD | Spread-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | 3/3 | 1.61 | 0.6545 | 1.009 | 829.9 | 0.03412 | 0.02951 |
| nuts_cov_dense | 3/3 | 1.44 | 0.4204 | 1.005 | 989.5 | 0.02892 | 0.03585 |
| nuts_fisher_diag | 3/3 | 1.324 | 0.6248 | 1.012 | 652.4 | 0.06466 | 0.03353 |
| nuts_fisher_rank4 | 3/3 | 0.9561 | 0.364 | 1.004 | 920.9 | 0.0279 | 0.03201 |
| nuts_fisher_rm | 3/3 | 1.811 | 0.7737 | 1.009 | 762.7 | 0.0774 | 0.02528 |
| nuts_fisher_adam | 3/3 | 1.304 | 0.4614 | 1.007 | 674.9 | 0.05477 | 0.03404 |
| nuts_transport | 3/3 | 1.291 | 0.457 | 1.006 | 812.4 | 0.03515 | 0.03881 |
| hmc_cov | 3/3 | 1.418 | 0.5706 | 1.007 | 2060 | 0.03499 | 0.02813 |
| hmc_omf | 3/3 | 2.166 | 0.9782 | 1.02 | 2976 | 0.01253 | 0.07215 |
| hmc_bcss | 3/3 | 2.18 | 0.993 | 1.011 | 3355 | 0.03119 | 0.04677 |
| ghmc | 3/3 | 1.502 | 0.4327 | 1.015 | 2174 | 0.0416 | 0.03476 |
| drghmc | 3/3 | 1.371 | 0.5223 | 1.008 | 2371 | 0.02796 | 0.02774 |
| mams | 3/3 | 0.8666 | 0.1943 | 1.016 | 258.5 | 0.0561 | 0.05229 |
| mclmc | 3/3 | 1.213 | 0.4518 | 1.004 | 638.7 | 0.05022 | 0.0277 |
| chees | 3/3 | 0.8473 | 0.2046 | 1.018 | 217.2 | 0.06074 | 0.03205 |
| walnuts_d | 3/3 | 2.261 | 1.41 | 1.013 | 350.5 | 0.05864 | 0.02068 |
| walnuts_r2p | 3/3 | 2.878 | 1.972 | 1.012 | 457.3 | 0.05306 | 0.04682 |
| rmhmc_softabs | 0/3 | — | — | — | — | — | — |
| rmhmc_manifold | 0/3 | — | — | — | — | — | — |
| stan_nuts_diag | 3/3 | 1.52 | 0.514 | 1.007 | 818.3 | 0.04669 | 0.04813 |
| stan_nuts_dense | 3/3 | 1.246 | 0.394 | 1.006 | 853.3 | 0.03706 | 0.04142 |
| stan_pathfinder_nuts | 3/3 | 1.472 | 0.496 | 1.009 | 925.9 | 0.05123 | 0.03182 |
| blackjax_nuts | 3/3 | 21.74 | 7.516 | 1.004 | 848.8 | 0.02965 | 0.0189 |
| nuts_rs | 3/3 | 0.8361 | 0.3915 | 1.01 | 628.9 | 0.02393 | 0.02728 |
| nuts_saia_joint | 3/3 | 1.532 | 0.6555 | 1.005 | 819.5 | 0.04605 | 0.05787 |
| nuts_pathfinder_raw | 3/3 | 1.613 | 0.6343 | 1.005 | 817.2 | 0.04138 | 0.03154 |
| nuts_pathfinder_psis | 3/3 | 1.556 | 0.6262 | 1.006 | 919.8 | 0.05095 | 0.03388 |
| atlas_full | 3/3 | 2.623 | 1.605 | 1.02 | 390.1 | 0.04053 | 0.04236 |

Unresolved outcomes: rmhmc_softabs: 3 failed/unsupported repetition(s); rmhmc_manifold: 3 failed/unsupported repetition(s). See failures.csv for exact reasons.

## hmm_example-hmm_example

The target backend performs marginalization; wall time includes that common backend cost, not a native Alea likelihood comparison.

The baseline changed from historical HMC warmup to native NUTS warmup: completed 3/3 → 3/3; worst R-hat 1.004 → 1.006; minimum bulk ESS 1061 → 965.2; maximum mean error/reference SD 0.03761 → 0.043; maximum spread-ratio error 0.03647 → 0.03612.

| Workflow | Complete/planned | Process s | Retained s | Worst R-hat | Min bulk ESS | Mean error/ref SD | Spread-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | 3/3 | 5.518 | 2.278 | 1.006 | 965.2 | 0.043 | 0.03612 |
| nuts_cov_dense | 3/3 | 3.208 | 1.012 | 1.003 | 1490 | 0.04514 | 0.03185 |
| nuts_fisher_diag | 3/3 | 3.088 | 1.273 | 1.013 | 559.9 | 0.03257 | 0.02886 |
| nuts_fisher_rank4 | 3/3 | 1.98 | 0.815 | 1.002 | 1951 | 0.03414 | 0.021 |
| nuts_fisher_rm | 3/3 | 55.65 | 26.64 | 1.014 | 319.5 | 0.046 | 0.02996 |
| nuts_fisher_adam | 3/3 | 38.37 | 13.45 | 1.011 | 469.6 | 0.06833 | 0.04452 |
| nuts_transport | 3/3 | 3.59 | 1.008 | 1.008 | 1532 | 0.03173 | 0.0349 |
| hmc_cov | 3/3 | 3.819 | 1.261 | 1.059 | 4066 | 0.01296 | 0.08173 |
| hmc_omf | 3/3 | 6.447 | 2.59 | 1.042 | 1888 | 0.04951 | 0.06421 |
| hmc_bcss | 3/3 | 6.323 | 2.59 | 1.171 | 2410 | 0.05271 | 0.1631 |
| ghmc | 3/3 | 3.93 | 1.22 | 1.205 | 3828 | 0.01019 | 0.1209 |
| drghmc | 3/3 | 4.746 | 1.48 | 1.032 | 5036 | 0.01269 | 0.07514 |
| mams | 3/3 | 2.895 | 0.5132 | 1.02 | 297.9 | 0.05109 | 0.05416 |
| mclmc | 3/3 | 4.029 | 1.404 | 1.004 | 628.2 | 0.02236 | 0.02139 |
| chees | 3/3 | 3.479 | 0.5269 | 1.021 | 144.5 | 0.06283 | 0.03106 |
| walnuts_d | 3/3 | 13.22 | 5.783 | 1.012 | 544.9 | 0.04546 | 0.03218 |
| walnuts_r2p | 3/3 | 12.12 | 8.952 | 1.015 | 664.9 | 0.05653 | 0.0416 |
| rmhmc_softabs | 0/3 | — | — | — | — | — | — |
| rmhmc_manifold | 0/3 | — | — | — | — | — | — |
| stan_nuts_diag | 3/3 | 6.858 | 2.355 | 1.007 | 1129 | 0.05811 | 0.04532 |
| stan_nuts_dense | 3/3 | 5.751 | 1.491 | 1.008 | 1650 | 0.02596 | 0.03246 |
| stan_pathfinder_nuts | 3/3 | 5.683 | 1.839 | 1.011 | 1087 | 0.04722 | 0.03188 |
| blackjax_nuts | 3/3 | 30.84 | 9.875 | 1.006 | 861.4 | 0.04195 | 0.05298 |
| nuts_rs | 3/3 | 2.061 | 0.9465 | 1.009 | 721.7 | 0.04381 | 0.03549 |
| nuts_saia_joint | 0/3 | — | — | — | — | — | — |
| nuts_pathfinder_raw | 3/3 | 5.914 | 1.939 | 1.008 | 976.7 | 0.05492 | 0.03474 |
| nuts_pathfinder_psis | 3/3 | 5.553 | 2.039 | 1.005 | 1011 | 0.02781 | 0.04808 |
| atlas_full | 0/3 | — | — | — | — | — | — |

Unresolved outcomes: rmhmc_softabs: 3 failed/unsupported repetition(s); rmhmc_manifold: 3 failed/unsupported repetition(s); nuts_saia_joint: 3 failed/unsupported repetition(s); atlas_full: 3 failed/unsupported repetition(s). See failures.csv for exact reasons.

## kidiq-kidscore_momhs

The pilot must learn coordinate scales before retention; difficult regression starts are directly checked in the ablation.

The baseline changed from historical HMC warmup to native NUTS warmup: completed 3/3 → 3/3; worst R-hat 1.007 → 1.006; minimum bulk ESS 703.5 → 621.3; maximum mean error/reference SD 0.03102 → 0.05215; maximum spread-ratio error 0.03976 → 0.07266.

| Workflow | Complete/planned | Process s | Retained s | Worst R-hat | Min bulk ESS | Mean error/ref SD | Spread-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | 3/3 | 0.6749 | 0.2453 | 1.006 | 621.3 | 0.05215 | 0.07266 |
| nuts_cov_dense | 3/3 | 0.4766 | 0.1158 | 1.01 | 1808 | 0.04471 | 0.03145 |
| nuts_fisher_diag | 3/3 | 0.3901 | 0.1663 | 1.012 | 453 | 0.0578 | 0.05571 |
| nuts_fisher_rank4 | 3/3 | 0.261 | 0.09451 | 1.003 | 1791 | 0.03631 | 0.04323 |
| nuts_fisher_rm | 3/3 | 3.986 | 2.474 | 4.565 | 4.277 | 37.73 | 42.11 |
| nuts_fisher_adam | 3/3 | 3.575 | 2.148 | 4.516 | 4.279 | 37.76 | 43.26 |
| nuts_transport | 3/3 | 0.4856 | 0.1096 | 1.004 | 1549 | 0.0402 | 0.05227 |
| hmc_cov | 3/3 | 0.5919 | 0.1749 | 1.232 | 3521 | 0.01867 | 0.0651 |
| hmc_omf | 3/3 | 0.9408 | 0.3427 | 1.111 | 4192 | 0.03147 | 0.08647 |
| hmc_bcss | 3/3 | 1.044 | 0.3789 | 1.245 | 1236 | 0.03656 | 0.07808 |
| ghmc | 3/3 | 0.5941 | 0.18 | 1.135 | 3213 | 0.02098 | 0.102 |
| drghmc | 3/3 | 0.7281 | 0.2133 | 1.018 | 3498 | 0.0148 | 0.05442 |
| mams | 3/3 | 0.4586 | 0.08757 | 1.01 | 430.2 | 0.04246 | 0.01825 |
| mclmc | 3/3 | 0.5972 | 0.1766 | 1.007 | 687.1 | 0.01362 | 0.04125 |
| chees | 3/3 | 0.374 | 0.05455 | 1.024 | 199.7 | 0.06399 | 0.03237 |
| walnuts_d | 3/3 | 0.9773 | 0.5446 | 1.026 | 216 | 0.03926 | 0.0239 |
| walnuts_r2p | 3/3 | 1.32 | 0.7932 | 1.019 | 290.2 | 0.07551 | 0.02353 |
| rmhmc_softabs | 0/3 | — | — | — | — | — | — |
| rmhmc_manifold | 0/3 | — | — | — | — | — | — |
| stan_nuts_diag | 3/3 | 0.9265 | 0.212 | 1.006 | 746.6 | 0.06168 | 0.02373 |
| stan_nuts_dense | 3/3 | 0.725 | 0.114 | 1.003 | 1259 | 0.0329 | 0.01962 |
| stan_pathfinder_nuts | 3/3 | 0.9814 | 0.203 | 1.007 | 663.4 | 0.04069 | 0.02762 |
| blackjax_nuts | 3/3 | 6.003 | 0.1462 | 1.009 | 620.9 | 0.03201 | 0.04122 |
| nuts_rs | 3/3 | 0.2719 | 0.1229 | 1.007 | 669.2 | 0.05743 | 0.05655 |
| nuts_saia_joint | 3/3 | 1.434 | 0.2203 | 1.007 | 736.5 | 0.02402 | 0.02995 |
| nuts_pathfinder_raw | 3/3 | 0.8928 | 0.2167 | 1.01 | 645.2 | 0.0183 | 0.01408 |
| nuts_pathfinder_psis | 3/3 | 0.866 | 0.2123 | 1.006 | 710.5 | 0.03922 | 0.02162 |
| atlas_full | 3/3 | 1.641 | 1.015 | 1.017 | 292.3 | 0.02046 | 0.02573 |

Unresolved outcomes: rmhmc_softabs: 3 failed/unsupported repetition(s); rmhmc_manifold: 3 failed/unsupported repetition(s). See failures.csv for exact reasons.

## low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse

Mixture occupancy and label behavior can dominate between-chain diagnostics. Faster transitions alone do not establish mode exploration.

The baseline changed from historical HMC warmup to native NUTS warmup: completed 3/3 → 3/3; worst R-hat 1.335 → 1.48; minimum bulk ESS 9.701 → 7.736; maximum mean error/reference SD — → —; maximum spread-ratio error — → —.

| Workflow | Complete/planned | Process s | Retained s | Worst R-hat | Min bulk ESS | Mean error/ref SD | Spread-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | 3/3 | 25.85 | 12.71 | 1.48 | 7.736 | — | — |
| nuts_cov_dense | 3/3 | 16.94 | 6.223 | 1.786 | 6.076 | — | — |
| nuts_fisher_diag | 3/3 | 21.88 | 10.87 | 1.459 | 8.153 | — | — |
| nuts_fisher_rank4 | 3/3 | 20.8 | 5.975 | 1.641 | 6.664 | — | — |
| nuts_fisher_rm | 3/3 | 111.8 | 48.61 | 1.687 | 6.473 | — | — |
| nuts_fisher_adam | 3/3 | 86.01 | 37.77 | 1.605 | 6.762 | — | — |
| nuts_transport | 3/3 | 51.29 | 21.34 | 1.405 | 8.56 | — | — |
| hmc_cov | 3/3 | 16.48 | 4.282 | 1.686 | 6.462 | — | — |
| hmc_omf | 3/3 | 23.81 | 9.308 | 1.486 | 7.735 | — | — |
| hmc_bcss | 3/3 | 24.84 | 9.092 | 1.548 | 7.115 | — | — |
| ghmc | 3/3 | 18.44 | 4.752 | 1.553 | 7.215 | — | — |
| drghmc | 3/3 | 18.36 | 5.925 | 1.589 | 6.865 | — | — |
| mams | 3/3 | 28.45 | 10.27 | 1.629 | 6.791 | — | — |
| mclmc | 3/3 | 16.22 | 4.616 | 1.565 | 7.118 | — | — |
| chees | 3/3 | 15.46 | 4.807 | 1.805 | 6.002 | — | — |
| walnuts_d | 3/3 | 39.56 | 22.63 | 1.741 | 6.464 | — | — |
| walnuts_r2p | 3/3 | 44.23 | 30.54 | 1.737 | 6.303 | — | — |
| rmhmc_softabs | 0/3 | — | — | — | — | — | — |
| rmhmc_manifold | 0/3 | — | — | — | — | — | — |
| stan_nuts_diag | 3/3 | 20.31 | 10.84 | 1.278 | 11.02 | — | — |
| stan_nuts_dense | 3/3 | 18.93 | 8.961 | 1.684 | 6.426 | — | — |
| stan_pathfinder_nuts | 3/3 | 23.12 | 9.782 | 1.428 | 8.241 | — | — |
| blackjax_nuts | 3/3 | 66.32 | 31.66 | 1.331 | 9.904 | — | — |
| nuts_rs | 3/3 | 14.07 | 6.783 | 1.566 | 7.036 | — | — |
| nuts_saia_joint | 3/3 | 44.41 | 22.87 | 1.279 | 11.24 | — | — |
| nuts_pathfinder_raw | 3/3 | 33.87 | 13.73 | 1.392 | 8.701 | — | — |
| nuts_pathfinder_psis | 3/3 | 25.52 | 11.57 | 1.35 | 9.52 | — | — |
| atlas_full | 3/3 | 40.29 | 25.45 | 1.58 | 7.064 | — | — |

Unresolved outcomes: rmhmc_softabs: 3 failed/unsupported repetition(s); rmhmc_manifold: 3 failed/unsupported repetition(s). See failures.csv for exact reasons.

## normal_2-normal_mixture

Mixture occupancy and label behavior can dominate between-chain diagnostics. Faster transitions alone do not establish mode exploration.

The baseline changed from historical HMC warmup to native NUTS warmup: completed 3/3 → 3/3; worst R-hat 1.532 → 1.53; minimum bulk ESS 7.246 → 7.231; maximum mean error/reference SD — → —; maximum spread-ratio error — → —.

| Workflow | Complete/planned | Process s | Retained s | Worst R-hat | Min bulk ESS | Mean error/ref SD | Spread-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | 3/3 | 5.185 | 2.058 | 1.53 | 7.231 | — | — |
| nuts_cov_dense | 3/3 | 5.815 | 2.105 | 1.534 | 7.249 | — | — |
| nuts_fisher_diag | 3/3 | 5.278 | 1.777 | 1.531 | 7.244 | — | — |
| nuts_fisher_rank4 | 3/3 | 4.973 | 1.632 | 1.531 | 7.244 | — | — |
| nuts_fisher_rm | 3/3 | 42.52 | 17.12 | 1.537 | 7.257 | — | — |
| nuts_fisher_adam | 3/3 | 20.57 | 7.429 | 1.533 | 7.262 | — | — |
| nuts_transport | 3/3 | 4.523 | 2.187 | 1.532 | 7.237 | — | — |
| hmc_cov | 3/3 | 5.471 | 3.154 | 1.752 | 6.983 | — | — |
| hmc_omf | 3/3 | 10.84 | 6.848 | 1.788 | 6.113 | — | — |
| hmc_bcss | 3/3 | 13.89 | 7.596 | 1.809 | 5.941 | — | — |
| ghmc | 3/3 | 7.662 | 4.594 | 1.545 | 7.346 | — | — |
| drghmc | 3/3 | 8.398 | 5.366 | 1.563 | 7.198 | — | — |
| mams | 3/3 | 2.331 | 0.4372 | 1.533 | 7.271 | — | — |
| mclmc | 3/3 | 7.154 | 4.217 | 1.53 | 7.215 | — | — |
| chees | 2/3 | 72.91 | 44.26 | 1.532 | 7.281 | — | — |
| walnuts_d | 3/3 | 11.16 | 8.849 | 1.535 | 7.241 | — | — |
| walnuts_r2p | 3/3 | 12.6 | 10.63 | 1.538 | 7.237 | — | — |
| rmhmc_softabs | 0/3 | — | — | — | — | — | — |
| rmhmc_manifold | 0/3 | — | — | — | — | — | — |
| stan_nuts_diag | 3/3 | 3.807 | 1.653 | 1.53 | 7.228 | — | — |
| stan_nuts_dense | 3/3 | 4.107 | 1.879 | 1.533 | 7.242 | — | — |
| stan_pathfinder_nuts | 3/3 | 4.462 | 2.118 | 1.533 | 7.228 | — | — |
| blackjax_nuts | 3/3 | 15.52 | 4.67 | 1.532 | 7.245 | — | — |
| nuts_rs | 3/3 | 2.502 | 1.087 | 1.531 | 7.222 | — | — |
| nuts_saia_joint | 3/3 | 4.428 | 1.56 | 1.732 | 7.197 | — | — |
| nuts_pathfinder_raw | 0/3 | — | — | — | — | — | — |
| nuts_pathfinder_psis | 0/3 | — | — | — | — | — | — |
| atlas_full | 3/3 | 19.15 | 13.97 | 1.539 | 7.212 | — | — |

Unresolved outcomes: chees: 1 failed/unsupported repetition(s); rmhmc_softabs: 3 failed/unsupported repetition(s); rmhmc_manifold: 3 failed/unsupported repetition(s); nuts_pathfinder_raw: 3 failed/unsupported repetition(s); nuts_pathfinder_psis: 3 failed/unsupported repetition(s). See failures.csv for exact reasons.

## radon_mn-radon_variable_intercept_noncentered

The declared parameterization, starts and budgets are held in the manifest. Inspect inference fidelity and total cost together.

The baseline changed from historical HMC warmup to native NUTS warmup: completed 3/3 → 3/3; worst R-hat 1.013 → 1.018; minimum bulk ESS 763.2 → 728.6; maximum mean error/reference SD — → —; maximum spread-ratio error — → —.

| Workflow | Complete/planned | Process s | Retained s | Worst R-hat | Min bulk ESS | Mean error/ref SD | Spread-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | 3/3 | 10.47 | 2.916 | 1.018 | 728.6 | — | — |
| nuts_cov_dense | 3/3 | 60.64 | 7.285 | 1.007 | 899 | — | — |
| nuts_fisher_diag | 3/3 | 5.44 | 2.406 | 1.01 | 437.2 | — | — |
| nuts_fisher_rank4 | 3/3 | 5.789 | 2.44 | 1.008 | 609.8 | — | — |
| nuts_fisher_rm | 3/3 | 79.46 | 36.88 | 1.061 | 61.69 | — | — |
| nuts_fisher_adam | 3/3 | 66.49 | 30.67 | 1.016 | 442.7 | — | — |
| nuts_transport | 3/3 | 13.34 | 3.356 | 1.011 | 771.3 | — | — |
| hmc_cov | 3/3 | 15.85 | 1.668 | 1.178 | 396.7 | — | — |
| hmc_omf | 3/3 | 18.29 | 2.843 | 1.049 | 469.4 | — | — |
| hmc_bcss | 3/3 | 19.51 | 3.083 | 1.043 | 77.97 | — | — |
| ghmc | 3/3 | 8.624 | 1.445 | 1.095 | 193.6 | — | — |
| drghmc | 3/3 | 8.482 | 1.696 | 1.067 | 492.8 | — | — |
| mams | 3/3 | 8.622 | 0.6277 | 1.03 | 180.9 | — | — |
| mclmc | 3/3 | 10.25 | 1.537 | 1.005 | 550.1 | — | — |
| chees | 3/3 | 7.475 | 0.7704 | 1.038 | 174.2 | — | — |
| walnuts_d | 3/3 | 23.43 | 11.45 | 1.012 | 434.5 | — | — |
| walnuts_r2p | 3/3 | 26.35 | 13.37 | 1.018 | 519.2 | — | — |
| rmhmc_softabs | 0/3 | — | — | — | — | — | — |
| rmhmc_manifold | 0/3 | — | — | — | — | — | — |
| stan_nuts_diag | 3/3 | 14.89 | 2.585 | 1.012 | 880.6 | — | — |
| stan_nuts_dense | 3/3 | 42.22 | 4.542 | 1.007 | 715.8 | — | — |
| stan_pathfinder_nuts | 3/3 | 15.71 | 2.508 | 1.016 | 710 | — | — |
| blackjax_nuts | 0/3 | — | — | — | — | — | — |
| nuts_rs | 3/3 | 3.557 | 1.505 | 1.014 | 393.3 | — | — |
| nuts_saia_joint | 3/3 | 22.17 | 3.226 | 1.013 | 439.2 | — | — |
| nuts_pathfinder_raw | 3/3 | 15.39 | 3.95 | 1.016 | 784.6 | — | — |
| nuts_pathfinder_psis | 3/3 | 17.72 | 3.856 | 1.019 | 777.5 | — | — |
| atlas_full | 3/3 | 19.73 | 8.153 | 1.022 | 546.7 | — | — |

Unresolved outcomes: rmhmc_softabs: 3 failed/unsupported repetition(s); rmhmc_manifold: 3 failed/unsupported repetition(s); blackjax_nuts: 3 failed/unsupported repetition(s). See failures.csv for exact reasons.

## sblri-blr

The pilot must learn coordinate scales before retention; difficult regression starts are directly checked in the ablation.

The baseline changed from historical HMC warmup to native NUTS warmup: completed 3/3 → 3/3; worst R-hat 1.019 → 1.017; minimum bulk ESS 240.9 → 331.1; maximum mean error/reference SD 16.26 → 0.1215; maximum spread-ratio error 157.1 → 0.04322.

| Workflow | Complete/planned | Process s | Retained s | Worst R-hat | Min bulk ESS | Mean error/ref SD | Spread-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | 3/3 | 0.156 | 0.04907 | 1.017 | 331.1 | 0.1215 | 0.04322 |
| nuts_cov_dense | 3/3 | 0.2207 | 0.08302 | 1.014 | 341.9 | 0.04775 | 0.04165 |
| nuts_fisher_diag | 3/3 | 0.1679 | 0.03866 | 1.008 | 1996 | 0.03946 | 0.04151 |
| nuts_fisher_rank4 | 3/3 | 0.1536 | 0.03436 | 1.005 | 2012 | 0.04868 | 0.04255 |
| nuts_fisher_rm | 3/3 | 1.163 | 0.6385 | 4.567 | 4.272 | 823.9 | 939 |
| nuts_fisher_adam | 3/3 | 1.158 | 0.7409 | 4.558 | 4.273 | 721.6 | 934.7 |
| nuts_transport | 3/3 | 0.1513 | 0.04522 | 1.005 | 2004 | 0.04927 | 0.03511 |
| hmc_cov | 3/3 | 0.1123 | 0.04206 | 1.072 | 48.44 | 0.1257 | 0.03977 |
| hmc_omf | 3/3 | 0.1916 | 0.08197 | 1.014 | 923.1 | 0.05712 | 0.03657 |
| hmc_bcss | 3/3 | 0.2444 | 0.1015 | 1.014 | 684.5 | 0.05422 | 0.04237 |
| ghmc | 3/3 | 0.1527 | 0.04897 | 1.026 | 182.3 | 0.05536 | 0.07323 |
| drghmc | 3/3 | 0.1447 | 0.05437 | 1.03 | 245.7 | 0.05401 | 0.02805 |
| mams | 3/3 | 0.09556 | 0.01608 | 1.004 | 488.8 | 0.04815 | 0.05708 |
| mclmc | 3/3 | 0.3456 | 0.06122 | 1.005 | 1545 | 0.04184 | 0.0289 |
| chees | 3/3 | 0.1416 | 0.01159 | 1.022 | 453.5 | 0.05168 | 0.02466 |
| walnuts_d | 3/3 | 0.4928 | 0.3948 | 1.021 | 238.1 | 0.03276 | 0.04173 |
| walnuts_r2p | 3/3 | 0.5209 | 0.4252 | 1.013 | 218.2 | 0.05044 | 0.0461 |
| rmhmc_softabs | 0/3 | — | — | — | — | — | — |
| rmhmc_manifold | 0/3 | — | — | — | — | — | — |
| stan_nuts_diag | 3/3 | 0.8125 | 0.074 | 1.012 | 301.1 | 0.06287 | 0.07231 |
| stan_nuts_dense | 3/3 | 0.7552 | 0.078 | 1.017 | 303.4 | 0.04336 | 0.06119 |
| stan_pathfinder_nuts | 3/3 | 0.7235 | 0.07 | 1.011 | 297.1 | 0.04592 | 0.05387 |
| blackjax_nuts | 3/3 | 20.78 | 6.013 | 1.045 | 127.8 | 0.1084 | 0.03795 |
| nuts_rs | 3/3 | 0.06794 | 0.02372 | 1.005 | 2085 | 0.02196 | 0.03493 |
| nuts_saia_joint | 3/3 | 0.2962 | 0.08065 | 1.02 | 397.6 | 0.08062 | 0.04481 |
| nuts_pathfinder_raw | 3/3 | 0.2293 | 0.05726 | 1.018 | 342.6 | 0.04864 | 0.06302 |
| nuts_pathfinder_psis | 3/3 | 0.2082 | 0.05714 | 1.016 | 316.3 | 0.05903 | 0.05625 |
| atlas_full | 3/3 | 0.5743 | 0.4124 | 1.029 | 337.2 | 0.06151 | 0.05978 |

Unresolved outcomes: rmhmc_softabs: 3 failed/unsupported repetition(s); rmhmc_manifold: 3 failed/unsupported repetition(s). See failures.csv for exact reasons.

## seeds_data-seeds_stanified_model

The declared parameterization, starts and budgets are held in the manifest. Inspect inference fidelity and total cost together.

The baseline changed from historical HMC warmup to native NUTS warmup: completed 3/3 → 3/3; worst R-hat 1.069 → 1.025; minimum bulk ESS 59.77 → 207.4; maximum mean error/reference SD — → —; maximum spread-ratio error — → —.

| Workflow | Complete/planned | Process s | Retained s | Worst R-hat | Min bulk ESS | Mean error/ref SD | Spread-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | 3/3 | 0.9258 | 0.3412 | 1.025 | 207.4 | — | — |
| nuts_cov_dense | 3/3 | 1.771 | 0.4796 | 1.014 | 280.5 | — | — |
| nuts_fisher_diag | 3/3 | 0.4623 | 0.1823 | 1.147 | 19.21 | — | — |
| nuts_fisher_rank4 | 3/3 | 0.6205 | 0.2587 | 1.023 | 144.9 | — | — |
| nuts_fisher_rm | 3/3 | 1.934 | 0.431 | 1.048 | 80.63 | — | — |
| nuts_fisher_adam | 3/3 | 1.566 | 0.1998 | 1.016 | 156.7 | — | — |
| nuts_transport | 3/3 | 0.5267 | 0.2193 | 1.021 | 139.5 | — | — |
| hmc_cov | 3/3 | 0.352 | 0.07322 | 1.028 | 156.2 | — | — |
| hmc_omf | 3/3 | 0.5829 | 0.1859 | 1.06 | 64.03 | — | — |
| hmc_bcss | 3/3 | 0.7776 | 0.1899 | 1.043 | 88.26 | — | — |
| ghmc | 3/3 | 0.5786 | 0.1267 | 1.051 | 123.3 | — | — |
| drghmc | 3/3 | 0.5861 | 0.1449 | 1.052 | 85.27 | — | — |
| mams | 3/3 | 0.4564 | 0.07802 | 1.055 | 93.05 | — | — |
| mclmc | 3/3 | 0.5632 | 0.1636 | 1.021 | 207.3 | — | — |
| chees | 3/3 | 0.4577 | 0.08883 | 1.096 | 34.96 | — | — |
| walnuts_d | 3/3 | 1.543 | 1.013 | 1.094 | 60.91 | — | — |
| walnuts_r2p | 3/3 | 1.24 | 0.8697 | 1.041 | 103.3 | — | — |
| rmhmc_softabs | 0/3 | — | — | — | — | — | — |
| rmhmc_manifold | 0/3 | — | — | — | — | — | — |
| stan_nuts_diag | 3/3 | 1.585 | 0.192 | 1.023 | 193.9 | — | — |
| stan_nuts_dense | 3/3 | 1.848 | 0.211 | 1.025 | 260.5 | — | — |
| stan_pathfinder_nuts | 3/3 | 1.787 | 0.205 | 1.058 | 83.07 | — | — |
| blackjax_nuts | 3/3 | 66.71 | 27.32 | 1.042 | 145.9 | — | — |
| nuts_rs | 3/3 | 0.4015 | 0.1631 | 1.031 | 172.6 | — | — |
| nuts_saia_joint | 3/3 | 0.6942 | 0.3109 | 1.044 | 126.9 | — | — |
| nuts_pathfinder_raw | 3/3 | 3.125 | 0.2228 | 1.032 | 155.6 | — | — |
| nuts_pathfinder_psis | 3/3 | 0.886 | 0.3079 | 1.033 | 132.2 | — | — |
| atlas_full | 3/3 | 0.6659 | 0.3635 | 1.055 | 57.8 | — | — |

Unresolved outcomes: rmhmc_softabs: 3 failed/unsupported repetition(s); rmhmc_manifold: 3 failed/unsupported repetition(s). See failures.csv for exact reasons.

## surgical_data-surgical_model

The declared parameterization, starts and budgets are held in the manifest. Inspect inference fidelity and total cost together.

The baseline changed from historical HMC warmup to native NUTS warmup: completed 3/3 → 3/3; worst R-hat 1.009 → 1.018; minimum bulk ESS 523.6 → 354.8; maximum mean error/reference SD — → —; maximum spread-ratio error — → —.

| Workflow | Complete/planned | Process s | Retained s | Worst R-hat | Min bulk ESS | Mean error/ref SD | Spread-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | 3/3 | 0.2002 | 0.05545 | 1.018 | 354.8 | — | — |
| nuts_cov_dense | 3/3 | 1.426 | 0.1354 | 1.006 | 878.5 | — | — |
| nuts_fisher_diag | 3/3 | 0.3432 | 0.04212 | 1.007 | 494.4 | — | — |
| nuts_fisher_rank4 | 3/3 | 0.2205 | 0.04199 | 1.008 | 505.7 | — | — |
| nuts_fisher_rm | 3/3 | 0.3381 | 0.07785 | 1.01 | 462 | — | — |
| nuts_fisher_adam | 3/3 | 0.1788 | 0.04686 | 1.018 | 434.2 | — | — |
| nuts_transport | 3/3 | 0.1517 | 0.05929 | 1.011 | 413.5 | — | — |
| hmc_cov | 3/3 | 0.1472 | 0.03993 | 1.032 | 329 | — | — |
| hmc_omf | 3/3 | 0.1989 | 0.07451 | 1.02 | 613.6 | — | — |
| hmc_bcss | 3/3 | 0.3109 | 0.06477 | 1.039 | 231 | — | — |
| ghmc | 3/3 | 0.1346 | 0.03467 | 1.025 | 222.7 | — | — |
| drghmc | 3/3 | 0.2037 | 0.04218 | 1.026 | 456.5 | — | — |
| mams | 3/3 | 0.4432 | 0.01758 | 1.015 | 269.9 | — | — |
| mclmc | 3/3 | 0.2695 | 0.0567 | 1.008 | 554.8 | — | — |
| chees | 3/3 | 0.09577 | 0.01587 | 1.022 | 156.6 | — | — |
| walnuts_d | 3/3 | 0.4047 | 0.1286 | 1.02 | 287.5 | — | — |
| walnuts_r2p | 3/3 | 0.3677 | 0.2095 | 1.017 | 224.9 | — | — |
| rmhmc_softabs | 0/3 | — | — | — | — | — | — |
| rmhmc_manifold | 0/3 | — | — | — | — | — | — |
| stan_nuts_diag | 3/3 | 2.006 | 0.11 | 1.157 | 21.8 | — | — |
| stan_nuts_dense | 3/3 | 2.254 | 0.144 | 1.008 | 916.7 | — | — |
| stan_pathfinder_nuts | 3/3 | 1.906 | 0.097 | 1.023 | 312.1 | — | — |
| blackjax_nuts | 3/3 | 31.76 | 12.67 | 1.01 | 420.4 | — | — |
| nuts_rs | 3/3 | 0.1025 | 0.02995 | 1.017 | 290.9 | — | — |
| nuts_saia_joint | 3/3 | 0.1504 | 0.04343 | 1.01 | 393.2 | — | — |
| nuts_pathfinder_raw | 3/3 | 0.326 | 0.1003 | 1.007 | 458.8 | — | — |
| nuts_pathfinder_psis | 3/3 | 0.2798 | 0.09348 | 1.017 | 554.2 | — | — |
| atlas_full | 3/3 | 0.2282 | 0.117 | 1.068 | 96.93 | — | — |

Unresolved outcomes: rmhmc_softabs: 3 failed/unsupported repetition(s); rmhmc_manifold: 3 failed/unsupported repetition(s). See failures.csv for exact reasons.

## wells_data-wells_dist100_model

The declared parameterization, starts and budgets are held in the manifest. Inspect inference fidelity and total cost together.

The baseline changed from historical HMC warmup to native NUTS warmup: completed 3/3 → 3/3; worst R-hat 1.008 → 1.009; minimum bulk ESS 432 → 557.4; maximum mean error/reference SD 0.02953 → 0.04081; maximum spread-ratio error 0.02831 → 0.01653.

| Workflow | Complete/planned | Process s | Retained s | Worst R-hat | Min bulk ESS | Mean error/ref SD | Spread-ratio error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | 3/3 | 2.101 | 0.9382 | 1.009 | 557.4 | 0.04081 | 0.01653 |
| nuts_cov_dense | 3/3 | 1.377 | 0.5781 | 1.007 | 1632 | 0.02561 | 0.01889 |
| nuts_fisher_diag | 3/3 | 2.076 | 0.9784 | 1.008 | 563.3 | 0.04614 | 0.03003 |
| nuts_fisher_rank4 | 3/3 | 1.482 | 0.6554 | 1.005 | 1909 | 0.01459 | 0.0281 |
| nuts_fisher_rm | 3/3 | 11.96 | 4.96 | 1.017 | 394.3 | 0.03511 | 0.052 |
| nuts_fisher_adam | 3/3 | 6.716 | 2.212 | 1.011 | 376.3 | 0.03068 | 0.0282 |
| nuts_transport | 3/3 | 1.497 | 0.6202 | 1.004 | 1389 | 0.04149 | 0.02724 |
| hmc_cov | 3/3 | 2.384 | 1.271 | 1.086 | 6782 | 0.01016 | 0.04805 |
| hmc_omf | 3/3 | 4.445 | 2.687 | 1.015 | 292.4 | 0.02624 | 0.03983 |
| hmc_bcss | 3/3 | 4.637 | 2.872 | 1.008 | 832 | 0.03606 | 0.03212 |
| ghmc | 3/3 | 2.648 | 1.401 | 1.204 | 6782 | 0.01091 | 0.07188 |
| drghmc | 3/3 | 2.747 | 1.492 | 1.085 | 6782 | 0.01486 | 0.05795 |
| mams | 3/3 | 1.364 | 0.4435 | 1.017 | 331.7 | 0.02976 | 0.0182 |
| mclmc | 3/3 | 2.536 | 1.359 | 1.004 | 931.8 | 0.03579 | 0.01228 |
| chees | 3/3 | 1.465 | 0.3664 | 1.007 | 567.6 | 0.06335 | 0.03245 |
| walnuts_d | 3/3 | 4.687 | 3.62 | 1.012 | 535.7 | 0.0191 | 0.01292 |
| walnuts_r2p | 3/3 | 6.347 | 4.969 | 1.007 | 582.1 | 0.04854 | 0.02368 |
| rmhmc_softabs | 3/3 | 44.31 | 30.16 | 4.246 | 4.845 | 12.17 | 15.54 |
| rmhmc_manifold | 3/3 | 43.21 | 29.39 | 4.268 | 4.833 | 10.08 | 13.85 |
| stan_nuts_diag | 3/3 | 2.615 | 1.155 | 1.007 | 547.4 | 0.0797 | 0.02968 |
| stan_nuts_dense | 3/3 | 2.288 | 0.848 | 1.004 | 1569 | 0.04313 | 0.02022 |
| stan_pathfinder_nuts | 3/3 | 2.788 | 1.149 | 1.01 | 511.9 | 0.03058 | 0.02867 |
| blackjax_nuts | 3/3 | 8.844 | 0.7665 | 1.02 | 499 | 0.0349 | 0.02383 |
| nuts_rs | 3/3 | 1.426 | 0.6688 | 1.007 | 541 | 0.04043 | 0.03212 |
| nuts_saia_joint | 3/3 | 2.333 | 0.9374 | 1.011 | 488.5 | 0.03723 | 0.04119 |
| nuts_pathfinder_raw | 3/3 | 2.293 | 0.9669 | 1.009 | 631.6 | 0.01873 | 0.01057 |
| nuts_pathfinder_psis | 3/3 | 2.319 | 0.9816 | 1.006 | 586 | 0.0289 | 0.03419 |
| atlas_full | 3/3 | 10.78 | 7.941 | 1.011 | 1265 | 0.02506 | 0.05485 |
