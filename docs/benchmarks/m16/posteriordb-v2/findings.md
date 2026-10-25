# Descriptive PosteriorDB findings

1560 planned jobs: {'completed': 1267, 'failed': 260, 'unsupported': 33}. Twenty posteriors, 26 configurations, three repetitions, four chains per job. One Intel(R) Core(TM) i7-7500U CPU @ 2.70GHz host; one pinned CPU per job.

Completion is a technical outcome, not an adequacy gate. Reference discrepancies, R-hat, ESS, divergences and failures remain separate observations. All medians below condition on completion, with denominators shown. There is no aggregate promotion score.

## Configuration coverage and observations

| Configuration | Completed / 60 | Failed | Unsupported | Worst R-hat | Smallest bulk ESS | Largest median max mean error / ref SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | 60/60 | 0 | 0 | 4.567 | 4.272 | 48.97 |
| nuts_cov_dense | 60/60 | 0 | 0 | 4.567 | 4.272 | 48.97 |
| nuts_fisher_diag | 60/60 | 0 | 0 | 4.567 | 4.272 | 48.97 |
| nuts_fisher_rank4 | 60/60 | 0 | 0 | 4.361 | 4.294 | 48.97 |
| nuts_fisher_rm | 60/60 | 0 | 0 | 4.567 | 4.272 | 1037 |
| nuts_fisher_adam | 60/60 | 0 | 0 | 4.567 | 4.272 | 1044 |
| nuts_atlas_joint | 5/60 | 52 | 3 | 4.567 | 4.272 | 48.96 |
| nuts_pathfinder | 17/60 | 43 | 0 | 1.01 | 431.3 | 0.03377 |
| nuts_transport | 3/60 | 57 | 0 | 1.004 | 1007 | 0.03451 |
| hmc_cov | 60/60 | 0 | 0 | 4.567 | 4.272 | 501 |
| hmc_omf | 60/60 | 0 | 0 | 4.567 | 4.272 | 48.96 |
| hmc_bcss | 60/60 | 0 | 0 | 4.567 | 4.272 | 48.97 |
| ghmc | 60/60 | 0 | 0 | 4.567 | 4.272 | 131 |
| drghmc | 60/60 | 0 | 0 | 4.567 | 4.272 | 121.3 |
| mams | 51/60 | 3 | 6 | 1.221e+15 | 4.063 | 1071 |
| mclmc | 51/60 | 3 | 6 | 4.256 | 4.306 | 16.45 |
| chees | 51/60 | 9 | 0 | 1.532 | 7.229 | 0.2236 |
| walnuts_d | 60/60 | 0 | 0 | 4.567 | 4.272 | 537.5 |
| walnuts_r2p | 60/60 | 0 | 0 | 4.567 | 4.272 | 583.9 |
| rmhmc_softabs | 9/60 | 42 | 9 | 3.639 | 4.964 | 0.8912 |
| rmhmc_manifold | 9/60 | 42 | 9 | 3.702 | 4.941 | 0.7159 |
| stan_nuts_diag | 60/60 | 0 | 0 | 1.53 | 7.228 | 0.08082 |
| stan_nuts_dense | 60/60 | 0 | 0 | 1.684 | 6.426 | 0.08082 |
| stan_pathfinder_nuts | 54/60 | 6 | 0 | 1.533 | 7.228 | 0.05562 |
| blackjax_nuts | 57/60 | 3 | 0 | 1.532 | 7.245 | 0.1084 |
| nuts_rs | 60/60 | 0 | 0 | 1.566 | 7.036 | 0.06606 |

Worst diagnostics and discrepancies span very different posteriors and budgets; they are not a ranking. RMHMC has only 64 retained draws per chain and remains exactness-pending. MCLMC is approximate. ATLAS and Pathfinder identify composed workflows. See the full report for each posterior and repetition.

## Paired baseline costs

Alea below means covariance-diagonal warmup plus NUTS. Ratios use per-posterior medians where both configurations complete all three repetitions. Counts state the available denominator; failed comparisons are not hidden or treated as successful. Process ratios include driver/startup costs. Retained scopes differ across engines, and adaptation policies are not identical.

| Comparison | Paired posteriors | Median Alea / other process time | Median Alea / other retained time | Median Alea / other bulk ESS per process second |
| --- | ---: | ---: | ---: | ---: |
| CmdStan diagonal NUTS | 20 | 0.2679 | 1.194 | 2.587 |
| CmdStan dense NUTS | 20 | 0.2695 | 1.137 | 2.109 |
| nuts-rs diagonal NUTS | 20 | 1.395 | 1.818 | 0.6849 |
| BlackJAX native JAX targets | 6 | 0.01665 | 0.5087 | 66.34 |
| BlackJAX BridgeStan callbacks | 13 | 0.05419 | 0.09318 | 12.85 |

A process/retained time ratio below one means less recorded time for the Alea workflow. An ESS/s ratio above one means greater recorded bulk ESS per second on those completed cases; inspect the corresponding R-hat and reference discrepancies before interpreting it. There is no quality eligibility filter. Callback timings include Python/host transfers and do not establish native JAX performance.

## Failures and unsupported combinations

RMHMC rows reporting inconsistent SoftAbs third derivatives failed the finite-difference backend contract before sampling; they do not constitute RMHMC inference timings. ATLAS interval rejections and fixed transport-learning domain errors describe the declared warmup workflow. Native engine logs distinguish optimization failures from adapter/runtime errors.

- nuts_transport: 57 failed jobs. Examples: `worker exit 1: evaluation failed at integration step 2: target evaluation failed: target returned a non-finite log density; worker exit 1: evaluation failed at integration step 6: target evaluation failed: target returned a non-finite log density; worker exit 1: evaluation failed at integration step 5: target evaluation failed: target returned a non-finite log density`. Full per-job reasons are in failures.csv.
- nuts_atlas_joint: 52 failed jobs. Examples: `worker exit 1: dimensionless two-stage interval must be finite and strictly between zero and four`. Full per-job reasons are in failures.csv.
- nuts_pathfinder: 43 failed jobs. Examples: `worker exit 1: Pathfinder target evaluation failed on path 0: target returned a non-finite log density; worker exit 1: Pathfinder target evaluation failed on path 1: target returned a non-finite log density; worker exit 1: Pathfinder target evaluation failed on path 0: gradient entry 0 is missing or not finite`. Full per-job reasons are in failures.csv.
- rmhmc_softabs: 42 failed jobs. Examples: `worker exit 1: position-metric backend failed: SoftAbs third derivative is inconsistent at (0, 0, 6); worker exit 1: position-metric backend failed: SoftAbs third derivative is inconsistent at (0, 0, 1); worker exit 1: position-metric backend failed: SoftAbs third derivative is inconsistent at (0, 0, 2)`. Full per-job reasons are in failures.csv.
- rmhmc_manifold: 42 failed jobs. Examples: `worker exit 1: position-metric backend failed: SoftAbs third derivative is inconsistent at (0, 0, 6); worker exit 1: position-metric backend failed: SoftAbs third derivative is inconsistent at (0, 0, 1); worker exit 1: position-metric backend failed: SoftAbs third derivative is inconsistent at (0, 0, 2)`. Full per-job reasons are in failures.csv.
- chees: 9 failed jobs. Examples: `declared job deadline exhausted; process group killed and direct worker reaped`. Full per-job reasons are in failures.csv.
- stan_pathfinder_nuts: 6 failed jobs. Examples: `worker exit 1: {'kind': 'CalledProcessError', 'message': "Command '['/home/usu/alea-bench-runs/m16-posteriordb-v2-20261021/source/target/m16b-models-01/arma-arma11/stan/m; worker exit 1: {'kind': 'CalledProcessError', 'message': "Command '['/home/usu/alea-bench-runs/m16-posteriordb-v2-20261021/source/target/m16b-models-01/earnings-earn_heig; worker exit 1: {'kind': 'CalledProcessError', 'message': "Command '['/home/usu/alea-bench-runs/m16-posteriordb-v2-20261021/source/target/m16b-models-01/arma-arma11/stan/m`. Full per-job reasons are in failures.csv.
- mams: 3 failed jobs. Examples: `worker exit 1: isokinetic target evaluation failed at step 6: target returned a non-finite log density`. Full per-job reasons are in failures.csv.
- mclmc: 3 failed jobs. Examples: `worker exit 1: isokinetic target evaluation failed at step 6: target returned a non-finite log density; worker exit 1: isokinetic target evaluation failed at step 3: target returned a non-finite log density; worker exit 1: isokinetic target evaluation failed at step 8: target returned a non-finite log density`. Full per-job reasons are in failures.csv.
- blackjax_nuts: 3 failed jobs. Examples: `declared job deadline exhausted; process group killed and direct worker reaped`. Full per-job reasons are in failures.csv.
- Unsupported, 12 job(s): implemented isokinetic kernel requires at least two dimensions.
- Unsupported, 18 job(s): declared dense Hessian/third-derivative dimension budget.
- Unsupported, 3 job(s): HVP capability preflight: nonfinite HVP validation.

Full failure commands and partial outputs are in each job artifact. Timeouts preserve the declared budgets. Undefined diagnostics stay in the report; completion alone is not convergence.

## Review and reproduction

Open [the interactive report](report.html), inspect [the per-configuration CSV](summary.csv) or [coordinate reference comparisons](reference-comparisons.csv), and read [the method and timing scopes](../../../m16-posteriordb.md). The complete raw bundle includes every planned outcome and its source/preparation archives.
