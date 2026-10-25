# Native warmup and research workflow results

162 planned jobs: {'completed': 162}. 6 posteriors, 9 configurations, 3 repetitions, 4 chains; 1000 warmup and 2000 retained draws per chain.

Measured on one Intel(R) Core(TM) i7-7500U CPU @ 2.70GHz host. Jobs run serially on one allowed CPU. Compilation and preparation precede measurement; fresh-process wall time uses nanosecond clocks and pidfd completion.

Completion is a technical outcome. R-hat, ESS, posterior reference discrepancies, divergences and costs remain separate observations. Medians condition on completion with denominators shown. No promotion score, quality eligibility filter or default-selection decision is computed.

## Configuration coverage

| Configuration | Completed / planned | Failed | Unsupported | Worst R-hat | Smallest bulk ESS | Largest median max mean error / reference SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | 18/18 | 0 | 0 | 1.529 | 7.112 | 0.03522 |
| nuts_cov_dense | 18/18 | 0 | 0 | 1.529 | 7.115 | 0.0447 |
| nuts_fisher_diag | 18/18 | 0 | 0 | 1.529 | 7.099 | 0.02653 |
| nuts_fisher_rank4 | 18/18 | 0 | 0 | 1.531 | 7.099 | 0.02986 |
| atlas_full | 18/18 | 0 | 0 | 1.531 | 7.116 | 0.04873 |
| stan_nuts_diag | 18/18 | 0 | 0 | 1.529 | 7.106 | 0.02461 |
| stan_nuts_dense | 18/18 | 0 | 0 | 1.587 | 6.766 | 0.0328 |
| nuts_rs | 18/18 | 0 | 0 | 1.529 | 7.097 | 0.04441 |
| blackjax_nuts | 18/18 | 0 | 0 | 1.529 | 7.101 | 0.02405 |

Worst values span different posteriors and budgets and do not define a ranking. RMHMC remains exactness-pending with its smaller 32+64 budget and dimension limit. MCLMC is approximate. ATLAS uses a full research proposal with deterministic curvature; Pathfinder rows include initialization plus native NUTS.

## Baseline costs alongside statistical results

Alea baseline means covariance-diagonal native NUTS warmup and frozen NUTS. Each ratio below uses per-posterior medians only where both workflows complete all declared repetitions. Counts state that availability condition; they do not indicate convergence. All cases and their diagnostic discrepancies remain in the full report.

| Comparator | Available paired posteriors | Median Alea / comparator process time | Median Alea / comparator retained time | Median Alea / comparator bulk ESS per process second |
| --- | ---: | ---: | ---: | ---: |
| CmdStan diagonal NUTS | 6/6 | 0.6777 | 1.122 | 1.121 |
| CmdStan dense NUTS | 6/6 | 0.6619 | 1.127 | 1.646 |
| nuts-rs diagonal NUTS | 6/6 | 2.238 | 2.147 | 0.7105 |
| BlackJAX native JAX targets | 1/1 | 0.05122 | 0.243 | 7.75 |
| BlackJAX BridgeStan callbacks | 5/5 | 0.07707 | 0.08284 | 11.26 |

A time ratio below one indicates less recorded time for Alea. An ESS/s ratio above one indicates greater recorded bulk ESS per second in those completed comparisons. Inspect the paired R-hat and reference errors in [baseline-pairs.csv](baseline-pairs.csv) before interpreting either. No ratio supplies a scientific success gate. Native retained scopes differ: Stan includes CSV/generated quantities; the Rust worker includes owned retained positions and kernel conversion but excludes serialization. Cold process times include startup, initialization, adaptation, sampling and output. BlackJAX callback timings include host transfers and do not establish native-JAX speed.

## Difficult starts

| Posterior | Method | Completed / repetitions | Worst R-hat | Smallest bulk ESS | Median largest mean error / reference SD | Median largest spread-ratio error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| earnings-earn_height | nuts_cov_diag | 3/3 | 1.003 | 1805 | 0.01648 | 0.02451 |
| earnings-earn_height | nuts_cov_dense | 3/3 | 1.001 | 7546 | 0.01486 | 0.02341 |
| earnings-earn_height | nuts_fisher_diag | 3/3 | 1.005 | 1147 | 0.02587 | 0.02161 |
| earnings-earn_height | nuts_fisher_rank4 | 3/3 | 1.001 | 1.091e+04 | 0.009481 | 0.01323 |
| earnings-earn_height | atlas_full | 3/3 | 1.036 | 291 | 0.04873 | 0.02645 |
| earnings-earn_height | stan_nuts_diag | 3/3 | 1.003 | 2064 | 0.02056 | 0.0193 |
| earnings-earn_height | nuts_rs | 3/3 | 1.003 | 871.5 | 0.04441 | 0.01329 |
| sblri-blr | nuts_cov_diag | 3/3 | 1.003 | 1444 | 0.03522 | 0.01689 |
| sblri-blr | nuts_cov_dense | 3/3 | 1.004 | 1512 | 0.0447 | 0.01224 |
| sblri-blr | nuts_fisher_diag | 3/3 | 1.002 | 8020 | 0.02653 | 0.01563 |
| sblri-blr | nuts_fisher_rank4 | 3/3 | 1.001 | 7013 | 0.02986 | 0.02467 |
| sblri-blr | atlas_full | 3/3 | 1.005 | 1701 | 0.03255 | 0.03704 |
| sblri-blr | stan_nuts_diag | 3/3 | 1.003 | 1465 | 0.02461 | 0.0141 |
| sblri-blr | nuts_rs | 3/3 | 1.002 | 8511 | 0.01728 | 0.01892 |

The 128+64 smoke run and earlier diagnostic pilot are separate evidence. They are not pooled with this experiment. Empirical PosteriorDB reference summaries are not exact truth; the full coordinate file retains MCSE and reference spread for each comparison.

## Failures


All failures, timeouts, unsupported configurations and partial outputs remain in the denominator. A strict SoftAbs symmetry rejection is a backend-contract failure rather than an RMHMC inference timing. Finite-solver replay does not establish volume preservation. Finite-point score/output failures through frozen affine targets retain their declared typed error behavior. Do not replace these outcomes with retries or fallback policies after measurement.

## Evidence

[Interactive report](report.html), [all groups](summary.csv), [coordinate comparisons](reference-comparisons.csv), [phase costs](phase-costs.csv), [initialization paths](initialization-paths.csv), [failures](failures.csv). The complete bundle retains every result, raw draw, trace and source/preparation archive. Reproduce this supplementary analysis with `python3 analysis.py RUN_DIRECTORY OUTPUT_DIRECTORY`.

Manifest: `c0f59280be42166d617e7c23b65981f184574a90fb3c3ba7dee99e2ff7da5f56`.
