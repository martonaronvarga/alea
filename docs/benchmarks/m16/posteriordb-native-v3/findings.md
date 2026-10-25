# Native PosteriorDB findings

1128 planned jobs across five protocols: `{'completed': 1095, 'unsupported': 10, 'failed': 23}`. Twenty pinned PosteriorDB model/data pairs on one dusk host.

All 20 explicit Rust/Enzyme and float64 CPU JAX ports passed the declared independent conformance checks before measurement. Alea and nuts-rs share the same generated Enzyme target; CmdStan uses Stan AD; BlackJAX uses native JAX on every case. The shared-Stan companion holds BridgeStan constant across the Rust workflows.

Completion is a technical outcome. R-hat, ESS, divergences, reference discrepancies and failures remain separate observations. The tables impose no statistical eligibility filter or promotion threshold. Failed and unsupported rows remain in every inventory.

## Evidence by protocol

| Protocol | Planned | Completed | Failed | Unsupported | Report |
| --- | ---: | ---: | ---: | ---: | --- |
| pilot | 36 | 36 | 0 | 0 | [interactive](pilot/report.html), [CSV](pilot/summary.csv) |
| broad | 360 | 360 | 0 | 0 | [interactive](broad/report.html), [CSV](broad/summary.csv) |
| shared-stan | 240 | 240 | 0 | 0 | [interactive](shared-stan/report.html), [CSV](shared-stan/summary.csv) |
| parallel | 72 | 72 | 0 | 0 | [interactive](parallel/report.html), [CSV](parallel/summary.csv) |
| research | 420 | 387 | 23 | 10 | [interactive](research/report.html), [CSV](research/summary.csv) |

The pilot has two chains and 128/128 warmup/retained draws. Broad and shared-Stan use four chains and 512/1024 draws, with three repetitions. Parallel uses the same baseline budget on six models, with two repetitions. Research uses four chains and 256/512 draws, except dense manifold methods at 32/64 and dimension at most ten. These tracks are not pooled for efficiency comparisons.

## Native baseline coverage

| Method | Target intent | Complete/planned | Failed | Unsupported | Worst R-hat | Smallest bulk ESS | Divergences | Largest mean error / ref SD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | reference | 60/60 | 0 | 0 | 1.529 | 7.133 | 1210 | 0.08652 |
| nuts_cov_dense | reference | 60/60 | 0 | 0 | 1.563 | 6.98 | 988 | 0.05928 |
| nuts_rs | reference | 60/60 | 0 | 0 | 1.529 | 7.14 | 747 | 0.08154 |
| stan_nuts_diag | reference | 60/60 | 0 | 0 | 1.53 | 7.17 | 1042 | 0.08221 |
| stan_nuts_dense | reference | 60/60 | 0 | 0 | 1.623 | 6.58 | 652 | 0.04269 |
| blackjax_nuts | reference | 60/60 | 0 | 0 | 1.53 | 7.157 | 1068 | 0.08754 |

Extrema span different models. Inspect the per-posterior records for the corresponding timing and reference discrepancy.

## Paired native baseline costs

Ratios are Alea / competitor. Time below one means less recorded time for Alea; ESS/s above one means more recorded bulk ESS per second. Each table entry is the median across posterior-specific medians of defined repetition ratios. Medians condition only on completion and defined denominators; defined/planned counts remain explicit. The full pair inventory includes unsuccessful pairs.

| Alea method | Competitor | Defined process pairs/planned | Defined ESS/s pairs/planned | Process ratio | Retained ratio | Bulk ESS/process-second ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | nuts_rs | 60/60 | 60/60 | 2.02 | 1.71 | 0.6168 |
| nuts_cov_diag | stan_nuts_diag | 60/60 | 60/60 | 0.2234 | 0.2595 | 3.956 |
| nuts_cov_diag | stan_nuts_dense | 60/60 | 60/60 | 0.2974 | 0.4088 | 3.428 |
| nuts_cov_diag | blackjax_nuts | 60/60 | 60/60 | 0.09675 | 0.6436 | 9.611 |
| nuts_cov_dense | nuts_rs | 60/60 | 60/60 | 1.945 | 1.229 | 0.5831 |
| nuts_cov_dense | stan_nuts_diag | 60/60 | 60/60 | 0.1676 | 0.1929 | 6.714 |
| nuts_cov_dense | stan_nuts_dense | 60/60 | 60/60 | 0.2238 | 0.3085 | 4.576 |
| nuts_cov_dense | blackjax_nuts | 60/60 | 60/60 | 0.09186 | 0.3145 | 20.25 |

Retained scopes differ: Rust includes owned position collection, CmdStan includes native CSV/generated quantities, and JAX timings synchronize the device. Process walltime includes JAX JIT, startup and output. Native build cost is batch/incremental preparation, not a matched cold build per model.

## Shared Stan adaptation coverage

| Method | Target intent | Complete/planned | Failed | Unsupported | Worst R-hat | Smallest bulk ESS | Divergences | Largest mean error / ref SD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | reference | 60/60 | 0 | 0 | 1.531 | 7.157 | 971 | 0.09488 |
| nuts_fisher_diag | research_exact_target_intent | 60/60 | 0 | 0 | 1.53 | 7.158 | 915 | 0.07568 |
| nuts_fisher_rank4 | research_exact_target_intent | 60/60 | 0 | 0 | 1.542 | 7.158 | 631 | 0.07238 |
| nuts_rs | reference | 60/60 | 0 | 0 | 1.529 | 7.14 | 758 | 0.08344 |

## Native and shared-Stan paired backends

These pairs match model, configuration, repetition and declared budgets. The backend can change floating-point trajectories and call counts, so the workflow ratios are not isolated gradient microbenchmarks. Every outcome remains in [backend-pairs.csv](backend-pairs.csv).

## Parallel baseline

Four worker processes use four allowed logical CPUs. End-to-end time includes setup, output merge and compression. Phase seconds are sums across chain processes and therefore measure cumulative work, not the elapsed parallel critical path. Parallel throughput comparisons should use process walltime. Child RSS peak sums are upper bounds on simultaneous memory.

| Method | Target intent | Complete/planned | Failed | Unsupported | Worst R-hat | Smallest bulk ESS | Divergences | Largest mean error / ref SD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_cov_diag | reference | 12/12 | 0 | 0 | 1.529 | 7.133 | 5 | 0.08652 |
| nuts_cov_dense | reference | 12/12 | 0 | 0 | 1.529 | 7.189 | 5 | 0.05928 |
| nuts_rs | reference | 12/12 | 0 | 0 | 1.529 | 7.14 | 7 | 0.05929 |
| stan_nuts_diag | reference | 12/12 | 0 | 0 | 1.53 | 7.17 | 1 | 0.0668 |
| stan_nuts_dense | reference | 12/12 | 0 | 0 | 1.531 | 7.154 | 5 | 0.03409 |
| blackjax_nuts | reference | 12/12 | 0 | 0 | 1.53 | 7.157 | 3 | 0.08754 |

## Research coverage

These are opt-in algorithms and composed workflows under the stated finite budgets. MCLMC remains approximate; RMHMC remains exactness-pending, including when its higher derivatives come from real Enzyme AD. Short manifold budgets are not directly comparable with the baseline draw budget.

| Method | Target intent | Complete/planned | Failed | Unsupported | Worst R-hat | Smallest bulk ESS | Divergences | Largest mean error / ref SD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nuts_fisher_diag | research_exact_target_intent | 20/20 | 0 | 0 | 1.528 | 7.251 | 117 | 0.07469 |
| nuts_fisher_rank4 | research_exact_target_intent | 20/20 | 0 | 0 | 1.547 | 7.24 | 111 | 0.06157 |
| nuts_fisher_rm | research_exact_target_intent | 20/20 | 0 | 0 | 4.567 | 4.272 | 948 | 855.9 |
| nuts_fisher_adam | research_exact_target_intent | 20/20 | 0 | 0 | 4.567 | 4.272 | 1482 | 835.3 |
| nuts_transport | research_exact_target_intent | 20/20 | 0 | 0 | 2.175 | 5.238 | 252 | 6.267 |
| hmc_cov | research_exact_target_intent | 20/20 | 0 | 0 | 2.947 | 4.634 | 125 | 6.225 |
| hmc_omf | research_exact_target_intent | 20/20 | 0 | 0 | 3.392 | 4.492 | 110 | 22.34 |
| hmc_bcss | research_exact_target_intent | 20/20 | 0 | 0 | 2.94 | 4.656 | 135 | 3.752 |
| ghmc | research_exact_target_intent | 20/20 | 0 | 0 | 2.768 | 4.696 | 145 | 6.178 |
| drghmc | research_exact_target_intent | 20/20 | 0 | 0 | 2.317 | 5.065 | 251 | 6.19 |
| mams | research_exact_target_intent | 18/20 | 0 | 2 | 3.034 | 4.63 | 39 | 6.228 |
| mclmc | approximate | 17/20 | 1 | 2 | 3.155 | 4.536 | 0 | 6.235 |
| chees | research_exact_target_intent | 20/20 | 0 | 0 | 2.633 | 4.791 | 34 | 6.36 |
| walnuts_d | research_exact_target_intent | 20/20 | 0 | 0 | 1.666 | 6.455 | 0 | 0.1291 |
| walnuts_r2p | research_exact_target_intent | 20/20 | 0 | 0 | 1.711 | 6.36 | 0 | 0.1258 |
| rmhmc_softabs | exactness_pending | 6/20 | 11 | 3 | 3.97 | 4.894 | 0 | 12.3 |
| rmhmc_manifold | exactness_pending | 6/20 | 11 | 3 | 4.268 | 4.842 | 0 | 10.14 |
| nuts_saia_joint | research_exact_target_intent | 20/20 | 0 | 0 | 4.166 | 4.324 | 210 | 20.6 |
| nuts_pathfinder_raw | research_exact_target_intent | 20/20 | 0 | 0 | 1.53 | 7.302 | 119 | 0.119 |
| nuts_pathfinder_psis | research_exact_target_intent | 20/20 | 0 | 0 | 1.333 | 10.74 | 357 | 0.06331 |
| atlas_full | research_exact_target_intent | 20/20 | 0 | 0 | 1.967 | 5.556 | 0 | 2.947 |

## Reproduction and limitations

Each protocol directory contains its complete segmented evidence bundle, source/preparation archives, raw draws, failure artifacts and checksums. [verification.json](verification.json) records local current-reader and archived-reader equality after unpacking. The per-protocol comparisons and preparation costs retain their declared scopes. [failures.csv](failures.csv) contains the complete unsuccessful inventory.

Use `alea-bench pdb unpack BUNDLE OUTPUT`, then `alea-bench pdb report OUTPUT/posteriordb-v3 --output NEW_REPORT.json` to verify and regenerate a report offline. The method specification is [m16-native-backends.md](../../../m16-native-backends.md).

The explicit ports are hand translations, not an automatic Stan compiler or a model-authoring usability comparison. Twenty posteriors on one host do not cover all PosteriorDB models or establish a universal winner. Reference means/spreads and diagnostic estimates have their recorded scope and finite-sample uncertainty. This campaign does not change production defaults, API maturity or sampler exactness classifications.
