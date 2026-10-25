# Native warmup and research workflow results

1680 planned jobs: {'completed': 1553, 'failed': 91, 'unsupported': 36}. 20 posteriors, 28 configurations, 3 repetitions, 4 chains; 512 warmup and 512 retained draws per chain.

Measured on one Intel(R) Core(TM) i7-7500U CPU @ 2.70GHz host. Jobs run serially on one allowed CPU. Compilation and preparation precede measurement; fresh-process wall time uses nanosecond clocks and pidfd completion.

Completion is a technical outcome. R-hat, ESS, posterior reference discrepancies, divergences and costs remain separate observations. Medians condition on completion with denominators shown. No promotion score, quality eligibility filter or default-selection decision is computed.

## Configuration coverage

| Configuration | Completed / planned | Failed | Unsupported | Worst R-hat | Smallest bulk ESS | Largest median max mean error / reference SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | 60/60 | 0 | 0 | 1.53 | 7.231 | 0.1215 |
| nuts_cov_dense | 60/60 | 0 | 0 | 1.786 | 6.076 | 0.04775 |
| nuts_fisher_diag | 60/60 | 0 | 0 | 1.531 | 7.244 | 0.07707 |
| nuts_fisher_rank4 | 60/60 | 0 | 0 | 1.641 | 6.664 | 0.04868 |
| nuts_fisher_rm | 60/60 | 0 | 0 | 4.567 | 4.272 | 823.9 |
| nuts_fisher_adam | 60/60 | 0 | 0 | 4.567 | 4.272 | 721.6 |
| nuts_transport | 60/60 | 0 | 0 | 1.532 | 7.237 | 0.04927 |
| hmc_cov | 60/60 | 0 | 0 | 1.752 | 6.462 | 0.2666 |
| hmc_omf | 60/60 | 0 | 0 | 3.492 | 4.46 | 6.176 |
| hmc_bcss | 60/60 | 0 | 0 | 1.809 | 5.941 | 0.1685 |
| ghmc | 60/60 | 0 | 0 | 1.553 | 7.215 | 0.1185 |
| drghmc | 60/60 | 0 | 0 | 1.589 | 6.865 | 0.161 |
| mams | 54/60 | 0 | 6 | 1.629 | 6.791 | 0.115 |
| mclmc | 51/60 | 3 | 6 | 1.565 | 7.118 | 0.121 |
| chees | 59/60 | 1 | 0 | 1.805 | 6.002 | 0.2898 |
| walnuts_d | 60/60 | 0 | 0 | 1.741 | 6.464 | 0.1608 |
| walnuts_r2p | 60/60 | 0 | 0 | 1.737 | 6.303 | 0.1696 |
| rmhmc_softabs | 18/60 | 33 | 9 | 4.246 | 4.845 | 12.17 |
| rmhmc_manifold | 18/60 | 33 | 9 | 4.268 | 4.833 | 10.08 |
| stan_nuts_diag | 60/60 | 0 | 0 | 1.53 | 7.228 | 0.08082 |
| stan_nuts_dense | 60/60 | 0 | 0 | 1.684 | 6.426 | 0.08082 |
| stan_pathfinder_nuts | 54/60 | 6 | 0 | 1.533 | 7.228 | 0.05562 |
| blackjax_nuts | 57/60 | 3 | 0 | 1.532 | 7.245 | 0.1084 |
| nuts_rs | 60/60 | 0 | 0 | 1.566 | 7.036 | 0.06606 |
| nuts_saia_joint | 57/60 | 0 | 3 | 3.322 | 4.48 | 29.4 |
| nuts_pathfinder_raw | 54/60 | 6 | 0 | 1.392 | 8.701 | 0.05989 |
| nuts_pathfinder_psis | 54/60 | 6 | 0 | 1.35 | 9.52 | 0.06232 |
| atlas_full | 57/60 | 0 | 3 | 1.58 | 7.064 | 0.06274 |

Worst values span different posteriors and budgets and do not define a ranking. RMHMC remains exactness-pending with its smaller 32+64 budget and dimension limit. MCLMC is approximate. ATLAS uses a full research proposal with deterministic curvature; Pathfinder rows include initialization plus native NUTS.

## Baseline costs alongside statistical results

Alea baseline means covariance-diagonal native NUTS warmup and frozen NUTS. Each ratio below uses per-posterior medians only where both workflows complete all declared repetitions. Counts state that availability condition; they do not indicate convergence. All cases and their diagnostic discrepancies remain in the full report.

| Comparator | Available paired posteriors | Median Alea / comparator process time | Median Alea / comparator retained time | Median Alea / comparator bulk ESS per process second |
| --- | ---: | ---: | ---: | ---: |
| CmdStan diagonal NUTS | 20/20 | 0.6917 | 0.9922 | 1.329 |
| CmdStan dense NUTS | 20/20 | 0.544 | 1.101 | 1.62 |
| nuts-rs diagonal NUTS | 20/20 | 2.184 | 1.83 | 0.5918 |
| BlackJAX native JAX targets | 6/6 | 0.02881 | 0.3246 | 69.39 |
| BlackJAX BridgeStan callbacks | 13/14 | 0.07748 | 0.1021 | 11 |

A time ratio below one indicates less recorded time for Alea. An ESS/s ratio above one indicates greater recorded bulk ESS per second in those completed comparisons. Inspect the paired R-hat and reference errors in [baseline-pairs.csv](baseline-pairs.csv) before interpreting either. No ratio supplies a scientific success gate. Native retained scopes differ: Stan includes CSV/generated quantities; the Rust worker includes owned retained positions and kernel conversion but excludes serialization. Cold process times include startup, initialization, adaptation, sampling and output. BlackJAX callback timings include host transfers and do not establish native-JAX speed.

## Difficult starts

| Posterior | Method | Completed / repetitions | Worst R-hat | Smallest bulk ESS | Median largest mean error / reference SD | Median largest spread-ratio error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| earnings-earn_height | nuts_cov_diag | 3/3 | 1.007 | 534.5 | 0.04146 | 0.05819 |
| earnings-earn_height | nuts_cov_dense | 3/3 | 1.004 | 1654 | 0.01715 | 0.01823 |
| earnings-earn_height | nuts_fisher_diag | 3/3 | 1.016 | 318.9 | 0.0269 | 0.04967 |
| earnings-earn_height | nuts_fisher_rank4 | 3/3 | 1.003 | 2149 | 0.0218 | 0.0242 |
| earnings-earn_height | nuts_pathfinder_raw | 3/3 | 1.007 | 664.5 | 0.02726 | 0.03806 |
| earnings-earn_height | nuts_pathfinder_psis | 3/3 | 1.009 | 655.2 | 0.06232 | 0.03223 |
| earnings-earn_height | atlas_full | 3/3 | 1.105 | 39.59 | 0.04341 | 0.0682 |
| earnings-earn_height | stan_nuts_diag | 3/3 | 1.011 | 476.4 | 0.0159 | 0.02943 |
| earnings-earn_height | nuts_rs | 3/3 | 1.029 | 258.1 | 0.05724 | 0.0462 |
| sblri-blr | nuts_cov_diag | 3/3 | 1.017 | 331.1 | 0.1215 | 0.04322 |
| sblri-blr | nuts_cov_dense | 3/3 | 1.014 | 341.9 | 0.04775 | 0.04165 |
| sblri-blr | nuts_fisher_diag | 3/3 | 1.008 | 1996 | 0.03946 | 0.04151 |
| sblri-blr | nuts_fisher_rank4 | 3/3 | 1.005 | 2012 | 0.04868 | 0.04255 |
| sblri-blr | nuts_pathfinder_raw | 3/3 | 1.018 | 342.6 | 0.04864 | 0.06302 |
| sblri-blr | nuts_pathfinder_psis | 3/3 | 1.016 | 316.3 | 0.05903 | 0.05625 |
| sblri-blr | atlas_full | 3/3 | 1.029 | 337.2 | 0.06151 | 0.05978 |
| sblri-blr | stan_nuts_diag | 3/3 | 1.012 | 301.1 | 0.06287 | 0.07231 |
| sblri-blr | nuts_rs | 3/3 | 1.005 | 2085 | 0.02196 | 0.03493 |

The 128+64 smoke run and earlier diagnostic pilot are separate evidence. They are not pooled with this experiment. Empirical PosteriorDB reference summaries are not exact truth; the full coordinate file retains MCSE and reference spread for each comparison.

## Failures

- `mams`: {'unsupported': 6}. Examples: `implemented isokinetic kernel requires at least two dimensions`.
- `mclmc`: {'unsupported': 6, 'failed': 3}. Examples: `implemented isokinetic kernel requires at least two dimensions`; `worker exit 1: isokinetic target evaluation failed at step 3: target returned a non-finite log density`.
- `chees`: {'failed': 1}. Examples: `worker exit 1: ChEES observation is not finite and valid`.
- `rmhmc_softabs`: {'failed': 33, 'unsupported': 9}. Examples: `worker exit 1: position-metric derivative 1 is asymmetric at (3, 1)`; `worker exit 1: position-metric derivative 0 is asymmetric at (2, 1)`.
- `rmhmc_manifold`: {'failed': 33, 'unsupported': 9}. Examples: `worker exit 1: position-metric derivative 1 is asymmetric at (3, 1)`; `worker exit 1: position-metric derivative 0 is asymmetric at (2, 1)`.
- `stan_pathfinder_nuts`: {'failed': 6}. Examples: `worker exit 1: {'kind': 'CalledProcessError', 'message': "Command '['/home/usu/alea-bench-runs/m16-posteriordb-complete-20261022/source/target/m16b-models-01/arma-arma11/stan/model', 'pathfinder', 'num_paths=4'`; `worker exit 1: {'kind': 'CalledProcessError', 'message': "Command '['/home/usu/alea-bench-runs/m16-posteriordb-complete-20261022/source/target/m16b-models-01/earnings-earn_height/stan/model', 'pathfinder', 'num`.
- `blackjax_nuts`: {'failed': 3}. Examples: `declared job deadline exhausted; process group killed and direct worker reaped`.
- `nuts_saia_joint`: {'unsupported': 3}. Examples: `HVP capability preflight: nonfinite HVP validation`.
- `nuts_pathfinder_raw`: {'failed': 6}. Examples: `worker exit 1: Pathfinder produced no finite local Gaussian approximation`.
- `nuts_pathfinder_psis`: {'failed': 6}. Examples: `worker exit 1: Pathfinder produced no finite local Gaussian approximation`.
- `atlas_full`: {'unsupported': 3}. Examples: `HVP capability preflight: nonfinite HVP validation`.

All failures, timeouts, unsupported configurations and partial outputs remain in the denominator. A strict SoftAbs symmetry rejection is a backend-contract failure rather than an RMHMC inference timing. Finite-solver replay does not establish volume preservation. Successful nonfinite affine outputs now reach the checked evaluator as numerical failures; concrete backend errors remain typed. Do not replace these outcomes with retries or fallback policies after measurement.

## Evidence

[Interactive report](report.html), [all groups](summary.csv), [coordinate comparisons](reference-comparisons.csv), [phase costs](phase-costs.csv), [initialization paths](initialization-paths.csv), [failures](failures.csv). The complete bundle retains every result, raw draw, trace and source/preparation archive. Reproduce this supplementary analysis with `python3 analysis.py RUN_DIRECTORY OUTPUT_DIRECTORY`.

Manifest: `b3f79774164d1e92befd4bd427c27a1cba384ac108bdc865044428e2470e2024`.
