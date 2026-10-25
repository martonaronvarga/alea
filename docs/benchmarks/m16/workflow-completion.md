# Baseline and research workflow findings

The native NUTS warmup repair resolves the large regression-start errors while preserving stationary HMC behavior and frozen checkpoint encoding. The final descriptive matrix covers 20 PosteriorDB targets and 28 configurations. No promotion score, automatic sampler choice or required 1.5× ESS/s improvement is computed.

Final broad coverage: **1,553 completed, 91 failed and 36 unsupported** out of 1,680 jobs. The longer 1000+2000 comparison completed all 162 jobs. Every failure and unsupported outcome remains available; successful process completion does not certify posterior exploration.

## What caused the regression failures

The original HMC warmup rejected large finite downhill moves under its absolute energy-error cutoff. Native NUTS warmup uses one-sided search and actual tree observations. A separate cutoff-only intervention leaves the historical eight-step HMC pilot in place and changes only its warmup cutoff policy. All 18 jobs complete, and all nine controls reproduce the earlier raw trajectories byte-for-byte.

Errors below are medians over three repetitions of the largest absolute coordinate mean discrepancy, scaled by the PosteriorDB reference SD.

| Original-start target | Historical mean error/ref SD | Cutoff-only repair | Native NUTS warmup |
| --- | ---: | ---: | ---: |
| earnings-earn_height | 48.97 | 0.1201 | 0.04146 |
| sblri-blr | 16.26 | 0.04098 | 0.1215 |
| arK-arK | 0.04259 | 0.04109 | 0.04387 |

The cutoff intervention removes most of the extreme error on both regressions. Native NUTS warmup adds a coherent pilot and improves earnings ESS further: the observed minimum bulk ESS rises from 4.27 historically to 190 with the cutoff intervention and 535 with native warmup. On sblri, native warmup repairs the extreme spread discrepancy (about 157 → 0.043 in the maximum spread-ratio error). The smaller mean-error differences between successful workflows are Monte Carlo observations, not a ranking. AR-K already estimates well and shows a warmup-versus-retained-cost tradeoff.

The cutoff experiment changes search and subsequent HMC warmup proposals together; it does not assign their separate causal shares. [Cutoff evidence](warmup-cutoff-v3/README.md), [whole-warmup ablation](posteriordb-complete-v3-ablation/per-target-findings.md), and [identical-phase probes](workflow-refinement-v3/README.md) retain the exact settings and all outcomes.

## Cost relative to mature libraries

These are medians across available per-target comparisons, conditional only on both workflows completing all repetitions. No inference-quality filter is applied. A time ratio below 1 means less recorded time for Alea; an ESS/s ratio above 1 means more recorded bulk ESS per process second. Inspect per-target diagnostics and reference discrepancies alongside each ratio.

| Cohort | Comparator | Available targets | Alea/comparator process time | Alea/comparator retained time | Alea/comparator bulk ESS/process second |
| --- | --- | ---: | ---: | ---: | ---: |
| Broad | CmdStan diagonal NUTS | 20 | 0.6917 | 0.9922 | 1.329 |
| Broad | CmdStan dense NUTS | 20 | 0.544 | 1.101 | 1.62 |
| Broad | nuts-rs diagonal NUTS | 20 | 2.184 | 1.83 | 0.5918 |
| Broad | BlackJAX native JAX targets | 6 | 0.02881 | 0.3246 | 69.39 |
| Broad | BlackJAX BridgeStan callbacks | 13 | 0.07748 | 0.1021 | 11 |
| Longer | CmdStan diagonal NUTS | 6 | 0.5826 | 1.031 | 1.138 |
| Longer | CmdStan dense NUTS | 6 | 0.5873 | 1.107 | 1.429 |
| Longer | nuts-rs diagonal NUTS | 6 | 2.105 | 2.026 | 0.4722 |
| Longer | BlackJAX native JAX targets | 1 | 0.04972 | 0.2653 | 8.291 |
| Longer | BlackJAX BridgeStan callbacks | 5 | 0.1286 | 0.132 | 8.042 |

The baseline is competitive with CmdStan in total workflow cost on this host, while nuts-rs remains a substantial performance reference. Cold process cost includes startup, initialization, adaptation and output; CmdStan launches separate chain processes. Retained scopes also differ: Stan includes native CSV/generated quantities; Rust timers include owned position collection but exclude serialization. Native-JAX and BridgeStan-callback BlackJAX timings are separate comparisons. These are workflow observations on one physical host, not universal kernel-speed or likelihood-speed claims.

## Where the research methods stand

| Workflow | Completed / planned | Failed | Unsupported |
| --- | ---: | ---: | ---: |
| nuts_cov_diag | 60/60 | 0 | 0 |
| nuts_cov_dense | 60/60 | 0 | 0 |
| nuts_fisher_diag | 60/60 | 0 | 0 |
| nuts_fisher_rank4 | 60/60 | 0 | 0 |
| nuts_fisher_rm | 60/60 | 0 | 0 |
| nuts_fisher_adam | 60/60 | 0 | 0 |
| nuts_transport | 60/60 | 0 | 0 |
| chees | 59/60 | 1 | 0 |
| mams | 54/60 | 0 | 6 |
| mclmc | 51/60 | 3 | 6 |
| ghmc | 60/60 | 0 | 0 |
| drghmc | 60/60 | 0 | 0 |
| walnuts_d | 60/60 | 0 | 0 |
| walnuts_r2p | 60/60 | 0 | 0 |
| nuts_saia_joint | 57/60 | 0 | 3 |
| atlas_full | 57/60 | 0 | 3 |
| nuts_pathfinder_raw | 54/60 | 6 | 0 |
| nuts_pathfinder_psis | 54/60 | 6 | 0 |
| rmhmc_softabs | 18/60 | 33 | 9 |
| rmhmc_manifold | 18/60 | 33 | 9 |

- Covariance NUTS and dual-averaged Fisher NUTS complete every broad job. Difficult mixture diagnostics remain imperfect for Alea and mature comparators. Fisher does not show a universal efficiency advantage; its Robbins–Monro/Adam variants still produce very poor regression estimates despite completing.
- Transport now completes 60/60 and ChEES 59/60, versus 51/60 and 49/60 before the numerical-output fix. The remaining ChEES failure is an invalid shared adaptation observation on the normal-mixture case. Geometry consistency and scheduler replay are tested, but universal robustness is not established.
- Corrected MAMS completes all 54 dimension-admitted jobs. Approximate MCLMC has three retained numerical failures on dogs-log; these stay failures rather than hidden rejection/retry. The 128-trial refinement study shows finite-step moment discrepancies shrinking on the analytic Gaussian/banana endpoints, with uncertainty and corrected-control fluctuations retained. MCLMC remains approximate.
- Static HMC/GHMC/DRGHMC, WALNUTS and joint s-AIA are finite, frozen workflows with actual work accounting. They offer no general advantage in this matrix; resonance and poor configurations remain visible. The full report contains OMF and BCSS as well.
- Full Atlas completes every HVP-admitted job. Its independent 576-case construction checks complement descriptive nonlinear runs. It follows the paper density rather than the pinned author Python expression; the 1024-step bound is per leg. It stays outside the reference facade.
- Raw and PSIS Pathfinder each fail to produce a finite local Gaussian on ARMA and the normal mixture in all three repetitions. Other paths and their diagnostics are retained. No raw fallback replaces failed PSIS silently.
- RMHMC uses actual Stan third AD, but strict symmetry checks still reject many targets. Its budget is explicitly unmatched at 32+64. More fundamentally, the accepted finite solver can pass tolerant replay and still change volume. The [derivation](../../rmhmc-finite-map-analysis.md) closes the investigation with a negative result; `ExactnessPendingRmhmc` remains the correct boundary.

## Further improvements supported by these results

Keep covariance/dual-averaged native NUTS as the practical baseline. Profile the remaining work and overhead against nuts-rs using both ESS per actual call and time per call, with the same inference diagnostics. Treat optional research tuning separately: stabilize or restrict the failing optimizer policies, improve shared adaptation observations and unadjusted numerical stability, and improve Pathfinder initialization on the recorded failures. Higher-order backend consistency and a valid finite-RMHMC proposal law remain independent research problems; they do not block baseline delivery.

## Evidence and scope

[Broad findings and all-target explanations](posteriordb-complete-v3/findings.md), [per-target before/after records](posteriordb-complete-v3/per-target-findings.md), [longer comparison](posteriordb-complete-v3-focused/findings.md), [interactive report](posteriordb-complete-v3/report.html), [bias/solver/probe evidence](workflow-refinement-v3/README.md), [disjoint phase-call counts](posteriordb-complete-v3/phase-work.csv), [DRGHMC/Atlas stage work](posteriordb-complete-v3/stage-work.csv), [counter verification](posteriordb-complete-v3/phase-work-verification.json), and [validation receipt](workflow-validation-v3/README.md) provide raw/source hashes and reproduction paths.

Fresh validation passed 440 workspace tests (one ignored), all-target checks, Clippy, rustdoc, affected SIMD/Faer/OpenBLAS contracts and 10 Miri tests. Bounded CI commands executed locally; hosted GitHub Actions was not dispatched. Executed Stan/BlackJAX references retain four short-budget schedule differences and do not establish whole-workflow equivalence. No historical v1/v2 or earlier-v3 measurement was rewritten.
