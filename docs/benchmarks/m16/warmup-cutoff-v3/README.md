# Warmup cutoff-only comparison

This source-sealed benchmark-only variant compares the historical eight-step HMC warmup at its absolute correction cutoff 1000 with the same warmup admitting finite downhill errors. Both arms freeze geometry/step and reconstruct ordinary retained NUTS with cutoff 1000; both charge the same extra target evaluation. It changes the cutoff for search and HMC warmup proposals together. It does not distinguish their separate contributions. No production source or API was changed.

All 18 planned four-chain jobs and their raw draws, failures, metrics and clocks are retained. The nine control trajectories reproduce the earlier historical-warmup ablation byte-for-byte. The process timer uses the existing nanosecond/pidfd harness; every job runs serially on one CPU after the full matrix and focused run. No concurrent compilation occurs.

| Target | Warmup cutoff | Complete/planned | Worst R-hat | Smallest bulk ESS | Median mean error/ref SD | Median spread error | Median process s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| earnings-earn_height | absolute 1000 | 3/3 | 4.567 | 4.2723 | 48.97 | 0.99989 | 1.3446 |
| earnings-earn_height | finite downhill allowed | 3/3 | 1.0239 | 189.91 | 0.12006 | 0.065197 | 3.932 |
| sblri-blr | absolute 1000 | 3/3 | 1.0194 | 240.88 | 16.264 | 157.12 | 0.20093 |
| sblri-blr | finite downhill allowed | 3/3 | 1.0231 | 286.32 | 0.040982 | 0.067719 | 0.17075 |
| arK-arK | absolute 1000 | 3/3 | 1.007 | 989.24 | 0.042589 | 0.054671 | 5.6972 |
| arK-arK | finite downhill allowed | 3/3 | 1.0061 | 1021.1 | 0.041087 | 0.038394 | 5.31 |

Changing only the cutoff policy removes most of the extreme regression-start errors: earnings mean error falls from 48.97 to 0.120 reference SD, and sblri from 16.264 to 0.041. Native NUTS warmup supplies an additional pilot change; its earnings minimum bulk ESS is 535 versus 190 here. These observations support the cutoff as a major cause, without assigning separate shares to search and subsequent warmup proposals.

These are finite four-chain comparisons against empirical references. Inspect R-hat, ESS and raw coordinate MCSE together. No universal convergence claim or speed gate is computed. Compare with the separate [native NUTS warmup ablation](../posteriordb-complete-v3-ablation/per-target-findings.md) and [identical-phase probes](../workflow-refinement-v3/README.md).

The exact [source patch](cutoff-worker.patch), [specification](cutoff-spec.json), [build proof](cutoff-build.json) and [raw checksums](checksums.json) preserve the intervention.
