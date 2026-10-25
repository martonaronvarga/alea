# What this campaign establishes

All 1,128 planned outcomes are accounted for: 1095 completed, 23 failed and 10 unsupported. The pilot, broad native, shared-Stan and parallel protocols completed all 708 jobs. Research completed 387/420, with 23 failures and 10 declared unsupported combinations. Both local and dusk verification reproduce every report with current and archived readers.

## Performance

The native backend helps, while nuts-rs remains the faster Rust workflow in this corpus. The following ratios compare Alea covariance-diagonal NUTS with each competitor across all 20 posteriors and three repetitions. They are medians of per-posterior median repetition ratios, with all 60 pairs defined and no diagnostic filter.

| Comparator | Alea / comparator process walltime | Alea / comparator retained time | Alea / comparator bulk ESS per process second |
| --- | ---: | ---: | ---: |
| nuts_rs | 2.02 | 1.71 | 0.6168 |
| stan_nuts_diag | 0.2234 | 0.2595 | 3.956 |
| stan_nuts_dense | 0.2974 | 0.4088 | 3.428 |
| blackjax_nuts | 0.09675 | 0.6436 | 9.611 |

In the paired Alea backend comparison, Enzyme / BridgeStan process time is 0.5256: the native workflow records about 1.9× less elapsed time. This includes adaptation and output; it is not an isolated AD kernel measurement. BlackJAX process time includes fresh JIT. CmdStan includes native CSV/generated quantities, while Rust retained timing includes owned position collection. The ratios therefore describe these workflows and output scopes.

With the same Enzyme target, Alea / nuts-rs total gradient calls are 1.559, retained gradient calls 1.589, and retained seconds per gradient call 1.014. The near parity per call points toward adaptation and trajectory length as the main remaining gap in this run. It does not establish causal attribution: the libraries learn different metrics, use different step-size/window policies and visit different states. The pinned nuts-rs default uses a gradient-informed diagonal estimate and step jitter; Alea covariance warmup uses its declared reference policy. [work-comparisons.csv](work-comparisons.csv) preserves the 60 paired observations.

## Statistical behavior and maturity

Alea baseline reference-mean errors are at most 0.0866 reference standard deviations among coordinates with available comparisons. That observation does not establish convergence: centered eight schools, both mixtures and some hierarchical targets retain poor mixing or divergences. The competing libraries also struggle on these targets at the stated budgets. Alea diagonal NUTS has more observed divergences than nuts-rs (1,210 versus 747 over 60 jobs); CmdStan diagonal records 1,042. Each method has the same broad iteration budget, with library-specific adaptation.

The practical baseline is executable across the 20-model corpus with a real native AD path and independently checked evidence. The ports remain explicit hand translations, so this does not establish a general Stan compiler or convenient automatic model export for every user model. One Linux host and complete jobs do not establish platform graduation, universal performance or production readiness. Public defaults and maturity/exactness classifications remain unchanged.

## Parallel scope

The common process-per-chain parallel lane completes all 72 jobs. Full logical CSV contents match the corresponding serial runs in 72/72 pairs. The table below reports serial / parallel end-to-end time: above one means less elapsed time in parallel. Values are medians across six posterior-specific medians, with two repetitions each.

| Method | Serial / parallel process seconds |
| --- | ---: |
| nuts_cov_diag | 0.9682 |
| nuts_cov_dense | 0.7138 |
| nuts_rs | 0.7528 |
| stan_nuts_diag | 1.427 |
| stan_nuts_dense | 1.104 |
| blackjax_nuts | 0.5696 |

The lane measures a common subprocess architecture on four logical CPUs, including worker startup and merging. It is not a comparison of each library’s fastest native ensemble API. Small jobs can be slower in parallel. Phase seconds sum chain work rather than elapsed parallel time; the process timer measures the elapsed cohort.

## Research findings

Fisher diagonal/rank-four, ChEES, full ATLAS, both Pathfinder variants, transport, joint s-AIA and both WALNUTS variants all execute on 20/20 models. Execution is distinct from useful inference: Fisher RM/Adam reach extreme reference discrepancies, and several static/isokinetic/transport/ATLAS workflows mix poorly under their finite budgets. Fisher rank-four and Pathfinder PSIS are useful follow-up candidates by the recorded diagnostics, without a claim that they should replace the reference baseline.

MAMS completes all 18 supported cases. MCLMC completes 17/18 supported cases and fails on dogs after a non-finite target evaluation; its approximate classification remains explicit. The four one-dimensional isokinetic combinations are declared unsupported.

Each RMHMC configuration completes 6/20 cases, fails on 11 and exceeds the declared dense dimension budget on three. The 22 failed manifold jobs report Hessian/third-derivative consistency or metric symmetry problems at encountered states. Passing finite preflight points does not guarantee derivative robustness along a trajectory. Completed RMHMC runs also have poor diagnostics at 32/64 budgets. There is no finite-difference fallback or weakened symmetry check, and finite solver reversal/volume obligations remain open.

## Engineering direction

For the baseline, the next useful attribution experiment holds the frozen metric and step size fixed, then compares tree work and trajectory lengths against nuts-rs. Warmup outcomes can then be compared separately, including gradient-informed diagonal estimation. This directly addresses the observed extra calls while preserving accuracy checks. Manifold derivative robustness needs a separate numerical investigation that reproduces the archived failure commands and inspects the encountered states before broader RMHMC claims. Research tuning can remain opt-in while the baseline and modeling path improve.

The full [findings](findings.md), [interactive report index](index.html), [paired summary](paired-summary.json), [failure inventory](failures.csv) and [verification receipt](verification.json) retain the evidence and scope behind these observations.
