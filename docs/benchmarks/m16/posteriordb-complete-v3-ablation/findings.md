# Native warmup and research workflow results

18 planned jobs: {'completed': 18}. 3 posteriors, 2 configurations, 3 repetitions, 4 chains; 512 warmup and 512 retained draws per chain.

Measured on one Intel(R) Core(TM) i7-7500U CPU @ 2.70GHz host. Jobs run serially on one allowed CPU. Compilation and preparation precede measurement; fresh-process wall time uses nanosecond clocks and pidfd completion.

Completion is a technical outcome. R-hat, ESS, posterior reference discrepancies, divergences and costs remain separate observations. Medians condition on completion with denominators shown. No promotion score, quality eligibility filter or default-selection decision is computed.

## Configuration coverage

| Configuration | Completed / planned | Failed | Unsupported | Worst R-hat | Smallest bulk ESS | Largest median max mean error / reference SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_legacy_warmup | 9/9 | 0 | 0 | 4.567 | 4.272 | 48.97 |
| nuts_cov_diag | 9/9 | 0 | 0 | 1.017 | 331.1 | 0.1215 |

Worst values span different posteriors and budgets and do not define a ranking. RMHMC remains exactness-pending with its smaller 32+64 budget and dimension limit. MCLMC is approximate. ATLAS uses a full research proposal with deterministic curvature; Pathfinder rows include initialization plus native NUTS.

## Baseline costs alongside statistical results

Alea baseline means covariance-diagonal native NUTS warmup and frozen NUTS. Each ratio below uses per-posterior medians only where both workflows complete all declared repetitions. Counts state that availability condition; they do not indicate convergence. All cases and their diagnostic discrepancies remain in the full report.

| Comparator | Available paired posteriors | Median Alea / comparator process time | Median Alea / comparator retained time | Median Alea / comparator bulk ESS per process second |
| --- | ---: | ---: | ---: | ---: |
| CmdStan diagonal NUTS | 0/3 | — | — | — |
| CmdStan dense NUTS | 0/3 | — | — | — |
| nuts-rs diagonal NUTS | 0/3 | — | — | — |
| BlackJAX native JAX targets | 0/0 | — | — | — |
| BlackJAX BridgeStan callbacks | 0/3 | — | — | — |

A time ratio below one indicates less recorded time for Alea. An ESS/s ratio above one indicates greater recorded bulk ESS per second in those completed comparisons. Inspect the paired R-hat and reference errors in [baseline-pairs.csv](baseline-pairs.csv) before interpreting either. No ratio supplies a scientific success gate. Native retained scopes differ: Stan includes CSV/generated quantities; the Rust worker includes owned retained positions and kernel conversion but excludes serialization. Cold process times include startup, initialization, adaptation, sampling and output. BlackJAX callback timings include host transfers and do not establish native-JAX speed.

## Difficult starts

| Posterior | Method | Completed / repetitions | Worst R-hat | Smallest bulk ESS | Median largest mean error / reference SD | Median largest spread-ratio error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| earnings-earn_height | nuts_cov_diag | 3/3 | 1.007 | 534.5 | 0.04146 | 0.05819 |
| sblri-blr | nuts_cov_diag | 3/3 | 1.017 | 331.1 | 0.1215 | 0.04322 |

The 128+64 smoke run and earlier diagnostic pilot are separate evidence. They are not pooled with this experiment. Empirical PosteriorDB reference summaries are not exact truth; the full coordinate file retains MCSE and reference spread for each comparison.

## Failures


All failures, timeouts, unsupported configurations and partial outputs remain in the denominator. A strict SoftAbs symmetry rejection is a backend-contract failure rather than an RMHMC inference timing. Finite-solver replay does not establish volume preservation. Successful nonfinite affine outputs now reach the checked evaluator as numerical failures; concrete backend errors remain typed. Do not replace these outcomes with retries or fallback policies after measurement.

## Evidence

[Interactive report](report.html), [all groups](summary.csv), [coordinate comparisons](reference-comparisons.csv), [phase costs](phase-costs.csv), [initialization paths](initialization-paths.csv), [failures](failures.csv). The complete bundle retains every result, raw draw, trace and source/preparation archive. Reproduce this supplementary analysis with `python3 analysis.py RUN_DIRECTORY OUTPUT_DIRECTORY`.

Manifest: `49f22f61a3c76688b8f090ac064ae1b1dcd1dece5abf33678e9b722a287ab7ca`.
