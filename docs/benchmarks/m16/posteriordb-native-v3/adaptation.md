# Equal-budget adaptation observations

All four configurations complete 60/60 jobs: twenty posteriors, three repetitions, four chains, 512 warmup and 1,024 retained draws per chain. All use the same BridgeStan target backend. The table reports Alea / nuts-rs ratios, taking the median across posterior-specific median repetition ratios. Every ratio has 60/60 defined pairs; no diagnostic eligibility filter is applied.

| Alea warmup | Process walltime | Retained time | Total gradient calls | Bulk ESS/process second |
| --- | ---: | ---: | ---: | ---: |
| nuts_cov_diag | 1.845 | 1.668 | 1.57 | 0.6729 |
| nuts_fisher_diag | 1.405 | 1.386 | 1.364 | 0.7469 |
| nuts_fisher_rank4 | 1.279 | 1.017 | 1.062 | 0.9593 |

On this shared backend, rank-four Fisher records retained time and total gradient calls near nuts-rs, with additional workflow cost outside retained sampling. Its median bulk ESS per process second is also near parity. This is a descriptive observation under the declared history, rank and warmup policies; it does not establish general conditioning, an Enzyme speed result for this budget, or grounds to change the reference default.

Observed divergence totals are 971 for covariance diagonal, 915 for Fisher diagonal, 631 for Fisher rank-four and 758 for nuts-rs. Difficult posteriors and poor mixing remain in the table. The complete per-pair diagnostics and failures are retained in [shared-adaptation-pairs.csv](shared-adaptation-pairs.csv) and the [interactive shared-Stan report](shared-stan/report.html).

These observations make warmup and learned geometry useful places to investigate the remaining call-count gap. A matched frozen-metric/step-size comparison can distinguish tree work from adaptation outcomes. Fisher remains opt-in; this benchmark makes no promotion or default-selection decision.

See the [combined outcome](outcome.md), [numeric summary](shared-adaptation-summary.json) and [verification receipt](verification.json).
