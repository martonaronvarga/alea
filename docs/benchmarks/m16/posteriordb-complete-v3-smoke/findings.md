# Native warmup and research workflow results

56 planned jobs: {'completed': 51, 'failed': 5}. 2 posteriors, 28 configurations, 1 repetitions, 4 chains; 128 warmup and 64 retained draws per chain.

Measured on one Intel(R) Core(TM) i7-7500U CPU @ 2.70GHz host. Jobs run serially on one allowed CPU. Compilation and preparation precede measurement; fresh-process wall time uses nanosecond clocks and pidfd completion.

Completion is a technical outcome. R-hat, ESS, posterior reference discrepancies, divergences and costs remain separate observations. Medians condition on completion with denominators shown. No promotion score, quality eligibility filter or default-selection decision is computed.

## Configuration coverage

| Configuration | Completed / planned | Failed | Unsupported | Worst R-hat | Smallest bulk ESS | Largest median max mean error / reference SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | 2/2 | 0 | 0 | 3.083 | 5.127 | 6.248 |
| nuts_cov_dense | 2/2 | 0 | 0 | 1.792 | 6.683 | 6.344 |
| nuts_fisher_diag | 2/2 | 0 | 0 | 1.146 | 43.64 | 0.1323 |
| nuts_fisher_rank4 | 2/2 | 0 | 0 | 1.063 | 102.5 | 0.1435 |
| nuts_fisher_rm | 2/2 | 0 | 0 | 4.643 | 4.789 | 799.7 |
| nuts_fisher_adam | 2/2 | 0 | 0 | 4.643 | 4.789 | 794.9 |
| nuts_transport | 2/2 | 0 | 0 | 3.119 | 5.116 | 6.359 |
| hmc_cov | 2/2 | 0 | 0 | 3.168 | 5.086 | 6.375 |
| hmc_omf | 2/2 | 0 | 0 | 3.674 | 4.939 | 10.11 |
| hmc_bcss | 2/2 | 0 | 0 | 2.69 | 5.332 | 6.349 |
| ghmc | 2/2 | 0 | 0 | 3.887 | 4.892 | 6.372 |
| drghmc | 2/2 | 0 | 0 | 4.159 | 4.848 | 6.379 |
| mams | 2/2 | 0 | 0 | 2.888 | 5.203 | 6.372 |
| mclmc | 2/2 | 0 | 0 | 3.413 | 5.026 | 6.374 |
| chees | 2/2 | 0 | 0 | 3.162 | 5.086 | 6.337 |
| walnuts_d | 2/2 | 0 | 0 | 3.2 | 5.07 | 6.262 |
| walnuts_r2p | 2/2 | 0 | 0 | 3.173 | 5.123 | 6.269 |
| rmhmc_softabs | 0/2 | 2 | 0 | — | — | — |
| rmhmc_manifold | 0/2 | 2 | 0 | — | — | — |
| stan_nuts_diag | 2/2 | 0 | 0 | 2.599 | 5.391 | 6.365 |
| stan_nuts_dense | 2/2 | 0 | 0 | 2.599 | 5.391 | 6.365 |
| stan_pathfinder_nuts | 1/2 | 1 | 0 | 1.256 | 16.06 | 0.5333 |
| blackjax_nuts | 2/2 | 0 | 0 | 2.394 | 5.641 | 6.346 |
| nuts_rs | 2/2 | 0 | 0 | 2.646 | 5.341 | 4.707 |
| nuts_saia_joint | 2/2 | 0 | 0 | 3.924 | 4.903 | 500 |
| nuts_pathfinder_raw | 2/2 | 0 | 0 | 2.191 | 5.799 | 6.227 |
| nuts_pathfinder_psis | 2/2 | 0 | 0 | 2.219 | 5.766 | 6.235 |
| atlas_full | 2/2 | 0 | 0 | 3.096 | 5.112 | 6.166 |

Worst values span different posteriors and budgets and do not define a ranking. RMHMC remains exactness-pending with its smaller 32+64 budget and dimension limit. MCLMC is approximate. ATLAS uses a full research proposal with deterministic curvature; Pathfinder rows include initialization plus native NUTS.

## Baseline costs alongside statistical results

Alea baseline means covariance-diagonal native NUTS warmup and frozen NUTS. Each ratio below uses per-posterior medians only where both workflows complete all declared repetitions. Counts state that availability condition; they do not indicate convergence. All cases and their diagnostic discrepancies remain in the full report.

| Comparator | Available paired posteriors | Median Alea / comparator process time | Median Alea / comparator retained time | Median Alea / comparator bulk ESS per process second |
| --- | ---: | ---: | ---: | ---: |
| CmdStan diagonal NUTS | 2/2 | 0.6566 | 0.3691 | 17.7 |
| CmdStan dense NUTS | 2/2 | 0.7328 | 0.403 | 13.74 |
| nuts-rs diagonal NUTS | 2/2 | 3.276 | 0.772 | 0.7297 |
| BlackJAX native JAX targets | 0/0 | — | — | — |
| BlackJAX BridgeStan callbacks | 2/2 | 0.02816 | 0.02779 | 58.44 |

A time ratio below one indicates less recorded time for Alea. An ESS/s ratio above one indicates greater recorded bulk ESS per second in those completed comparisons. Inspect the paired R-hat and reference errors in [baseline-pairs.csv](baseline-pairs.csv) before interpreting either. No ratio supplies a scientific success gate. Native retained scopes differ: Stan includes CSV/generated quantities; the Rust worker includes owned retained positions and kernel conversion but excludes serialization. Cold process times include startup, initialization, adaptation, sampling and output. BlackJAX callback timings include host transfers and do not establish native-JAX speed.

## Difficult starts

| Posterior | Method | Completed / repetitions | Worst R-hat | Smallest bulk ESS | Median largest mean error / reference SD | Median largest spread-ratio error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| earnings-earn_height | nuts_cov_diag | 1/1 | 3.083 | 5.127 | 6.248 | 0.8676 |
| earnings-earn_height | nuts_cov_dense | 1/1 | 1.792 | 6.683 | 6.344 | 0.9738 |
| earnings-earn_height | nuts_fisher_diag | 1/1 | 1.146 | 43.64 | 0.1323 | 0.1267 |
| earnings-earn_height | nuts_fisher_rank4 | 1/1 | 1.04 | 102.5 | 0.03692 | 0.04699 |
| earnings-earn_height | nuts_pathfinder_raw | 1/1 | 2.191 | 5.799 | 6.227 | 0.8583 |
| earnings-earn_height | nuts_pathfinder_psis | 1/1 | 2.219 | 5.766 | 6.235 | 0.9128 |
| earnings-earn_height | atlas_full | 1/1 | 3.096 | 5.112 | 6.166 | 0.7142 |
| earnings-earn_height | stan_nuts_diag | 1/1 | 2.599 | 5.391 | 6.365 | 0.996 |
| earnings-earn_height | nuts_rs | 1/1 | 2.646 | 5.341 | 4.707 | 1.702 |
| sblri-blr | nuts_cov_diag | 1/1 | 1.009 | 256.8 | 0.09509 | 0.1631 |
| sblri-blr | nuts_cov_dense | 1/1 | 1.015 | 282.5 | 0.08472 | 0.0911 |
| sblri-blr | nuts_fisher_diag | 1/1 | 1.025 | 318.9 | 0.0774 | 0.1729 |
| sblri-blr | nuts_fisher_rank4 | 1/1 | 1.063 | 238.4 | 0.1435 | 0.1128 |
| sblri-blr | nuts_pathfinder_raw | 1/1 | 1.03 | 64.92 | 0.1391 | 0.1749 |
| sblri-blr | nuts_pathfinder_psis | 1/1 | 1.076 | 39.08 | 0.07539 | 0.152 |
| sblri-blr | atlas_full | 1/1 | 1.121 | 62.27 | 0.2421 | 0.1201 |
| sblri-blr | stan_nuts_diag | 1/1 | 1.204 | 16.44 | 0.793 | 0.09842 |
| sblri-blr | nuts_rs | 1/1 | 1.03 | 296.6 | 0.1178 | 0.1136 |

The 128+64 smoke run and earlier diagnostic pilot are separate evidence. They are not pooled with this experiment. Empirical PosteriorDB reference summaries are not exact truth; the full coordinate file retains MCSE and reference spread for each comparison.

## Failures

- `rmhmc_softabs`: {'failed': 2}. Examples: `worker exit 1: position-metric backend failed: SoftAbs third derivative is inconsistent at (0, 0, 2)`; `worker exit 1: position-metric derivative 4 is asymmetric at (4, 2)`.
- `rmhmc_manifold`: {'failed': 2}. Examples: `worker exit 1: position-metric backend failed: SoftAbs third derivative is inconsistent at (0, 0, 2)`; `worker exit 1: position-metric derivative 4 is asymmetric at (4, 2)`.
- `stan_pathfinder_nuts`: {'failed': 1}. Examples: `worker exit 1: {'kind': 'CalledProcessError', 'message': "Command '['/home/usu/alea-bench-runs/m16-posteriordb-complete-20261022/source/target/m16b-models-01/earnings-earn_height/stan/model', 'pathfinder', 'num`.

All failures, timeouts, unsupported configurations and partial outputs remain in the denominator. A strict SoftAbs symmetry rejection is a backend-contract failure rather than an RMHMC inference timing. Finite-solver replay does not establish volume preservation. Successful nonfinite affine outputs now reach the checked evaluator as numerical failures; concrete backend errors remain typed. Do not replace these outcomes with retries or fallback policies after measurement.

## Evidence

[Interactive report](report.html), [all groups](summary.csv), [coordinate comparisons](reference-comparisons.csv), [phase costs](phase-costs.csv), [initialization paths](initialization-paths.csv), [failures](failures.csv). The complete bundle retains every result, raw draw, trace and source/preparation archive. Reproduce this supplementary analysis with `python3 analysis.py RUN_DIRECTORY OUTPUT_DIRECTORY`.

Manifest: `d19772433d242ee4e169703094ef982ee2d647c5222d2c17c567709d2eda4ebc`.
