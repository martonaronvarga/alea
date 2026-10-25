# Descriptive PosteriorDB benchmark

20 posteriors, 3 method configurations, 180 planned jobs. Status counts: `{'completed': 180}`.

Every failure and unsupported combination is retained. Diagnostics and accuracy are observations; this report makes no promotion, pass/fail or default-selection decision.

Cold wall time includes process startup, adaptation, sampling and serialization. Retained timings have engine-specific scopes; JAX host callbacks and native-JAX translations are labelled separately. RMHMC is exactness-pending; MCLMC is approximate; ATLAS/Pathfinder rows are labelled composed workflows.

| Posterior | Method | Completed/planned | Process s | Warmup s | Retained s | Worst R-hat | Min bulk ESS | Max mean error / ref SD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rate_1_data-Rate_1_model | nuts_cov_diag | 3/3 | 0.03309 | 0.001996 | 0.003288 | 1.003 | 1403 | 0.02383 |
| Rate_1_data-Rate_1_model | nuts_grad_diag_jitter | 3/3 | 0.01971 | 0.002448 | 0.003378 | 1.003 | 1714 | 0.02257 |
| Rate_1_data-Rate_1_model | nuts_rs | 3/3 | 0.01834 | 0.002417 | 0.003968 | 1.004 | 1593 | 0.002573 |
| Rate_2_data-Rate_2_model | nuts_cov_diag | 3/3 | 0.3487 | 0.004672 | 0.008682 | 1.003 | 2846 | 0.02411 |
| Rate_2_data-Rate_2_model | nuts_grad_diag_jitter | 3/3 | 0.03936 | 0.003408 | 0.004688 | 1.003 | 3750 | 0.02494 |
| Rate_2_data-Rate_2_model | nuts_rs | 3/3 | 0.02369 | 0.003448 | 0.005339 | 1.002 | 4022 | 0.03746 |
| Rate_5_data-Rate_5_model | nuts_cov_diag | 3/3 | 0.01978 | 0.002757 | 0.004118 | 1.003 | 1535 | 0.01174 |
| Rate_5_data-Rate_5_model | nuts_grad_diag_jitter | 3/3 | 0.01791 | 0.002143 | 0.003074 | 1.003 | 1578 | 0.0124 |
| Rate_5_data-Rate_5_model | nuts_rs | 3/3 | 0.01738 | 0.002295 | 0.003609 | 1.003 | 1648 | 0.01349 |
| arK-arK | nuts_cov_diag | 3/3 | 0.9016 | 0.3018 | 0.5751 | 1.003 | 2467 | 0.03194 |
| arK-arK | nuts_grad_diag_jitter | 3/3 | 0.6095 | 0.2137 | 0.372 | 1.004 | 2026 | 0.02001 |
| arK-arK | nuts_rs | 3/3 | 0.5633 | 0.1925 | 0.3485 | 1.004 | 1672 | 0.02995 |
| arma-arma11 | nuts_cov_diag | 3/3 | 0.1308 | 0.04258 | 0.06984 | 1.003 | 3106 | 0.01909 |
| arma-arma11 | nuts_grad_diag_jitter | 3/3 | 0.1422 | 0.0414 | 0.0461 | 1.002 | 5091 | 0.01897 |
| arma-arma11 | nuts_rs | 3/3 | 0.09223 | 0.03008 | 0.04293 | 1.002 | 4556 | 0.02152 |
| garch-garch11 | nuts_cov_diag | 3/3 | 0.977 | 0.3574 | 0.6005 | 1.004 | 1864 | 0.04114 |
| garch-garch11 | nuts_grad_diag_jitter | 3/3 | 0.6354 | 0.223 | 0.3932 | 1.005 | 1313 | 0.04059 |
| garch-garch11 | nuts_rs | 3/3 | 0.578 | 0.2085 | 0.3522 | 1.005 | 1398 | 0.02294 |
| dogs-dogs_log | nuts_cov_diag | 3/3 | 1.049 | 0.3572 | 0.6598 | 1.004 | 1215 | — |
| dogs-dogs_log | nuts_grad_diag_jitter | 3/3 | 0.7714 | 0.2789 | 0.4822 | 1.008 | 839.9 | — |
| dogs-dogs_log | nuts_rs | 3/3 | 0.7517 | 0.276 | 0.4586 | 1.004 | 1098 | — |
| earnings-earn_height | nuts_cov_diag | 3/3 | 4.3 | 1.613 | 2.594 | 1.003 | 989.8 | 0.04742 |
| earnings-earn_height | nuts_grad_diag_jitter | 3/3 | 1.793 | 0.5515 | 1.221 | 1.011 | 655.1 | 0.04816 |
| earnings-earn_height | nuts_rs | 3/3 | 1.402 | 0.4439 | 0.936 | 1.01 | 475.4 | 0.03747 |
| earnings-logearn_interaction_z | nuts_cov_diag | 3/3 | 1.277 | 0.4372 | 0.8365 | 1.003 | 1853 | 0.02076 |
| earnings-logearn_interaction_z | nuts_grad_diag_jitter | 3/3 | 0.9472 | 0.3324 | 0.5933 | 1.003 | 1476 | 0.03556 |
| earnings-logearn_interaction_z | nuts_rs | 3/3 | 0.8263 | 0.2942 | 0.5178 | 1.004 | 1427 | 0.03627 |
| eight_schools-eight_schools_centered | nuts_cov_diag | 3/3 | 0.1572 | 0.04812 | 0.05605 | 1.124 | 25.96 | — |
| eight_schools-eight_schools_centered | nuts_grad_diag_jitter | 3/3 | 0.1046 | 0.02528 | 0.04884 | 1.04 | 160 | — |
| eight_schools-eight_schools_centered | nuts_rs | 3/3 | 0.07844 | 0.01768 | 0.0336 | 1.091 | 33.75 | — |
| eight_schools-eight_schools_noncentered | nuts_cov_diag | 3/3 | 0.0919 | 0.01787 | 0.0354 | 1.003 | 1951 | 0.03235 |
| eight_schools-eight_schools_noncentered | nuts_grad_diag_jitter | 3/3 | 0.0745 | 0.01344 | 0.02268 | 1.004 | 1900 | 0.03568 |
| eight_schools-eight_schools_noncentered | nuts_rs | 3/3 | 0.0687 | 0.01193 | 0.02011 | 1.006 | 2027 | 0.03068 |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | nuts_cov_diag | 3/3 | 8.528 | 2.761 | 5.567 | 1.223 | 13.54 | — |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | nuts_grad_diag_jitter | 3/3 | 5.548 | 1.988 | 3.538 | 1.252 | 11.99 | — |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | nuts_rs | 3/3 | 4.231 | 1.548 | 2.668 | 1.35 | 9.607 | — |
| normal_2-normal_mixture | nuts_cov_diag | 3/3 | 1.255 | 0.4348 | 0.8002 | 1.529 | 7.133 | — |
| normal_2-normal_mixture | nuts_grad_diag_jitter | 3/3 | 0.769 | 0.2888 | 0.4557 | 1.531 | 7.147 | — |
| normal_2-normal_mixture | nuts_rs | 3/3 | 0.7887 | 0.3045 | 0.4783 | 1.529 | 7.14 | — |
| sblri-blr | nuts_cov_diag | 3/3 | 0.1565 | 0.05165 | 0.08645 | 1.014 | 644.6 | 0.05791 |
| sblri-blr | nuts_grad_diag_jitter | 3/3 | 0.08431 | 0.02536 | 0.03835 | 1.003 | 4413 | 0.03957 |
| sblri-blr | nuts_rs | 3/3 | 0.08661 | 0.02381 | 0.03902 | 1.003 | 3993 | 0.02258 |
| wells_data-wells_dist100_model | nuts_cov_diag | 3/3 | 4.65 | 1.652 | 2.98 | 1.004 | 1147 | 0.0378 |
| wells_data-wells_dist100_model | nuts_grad_diag_jitter | 3/3 | 3.772 | 1.378 | 2.375 | 1.004 | 1123 | 0.01771 |
| wells_data-wells_dist100_model | nuts_rs | 3/3 | 3.422 | 1.271 | 2.135 | 1.002 | 1085 | 0.02689 |
| kidiq-kidscore_momhs | nuts_cov_diag | 3/3 | 0.3751 | 0.1711 | 0.1845 | 1.003 | 1453 | 0.01561 |
| kidiq-kidscore_momhs | nuts_grad_diag_jitter | 3/3 | 0.1857 | 0.06089 | 0.1078 | 1.003 | 1446 | 0.02362 |
| kidiq-kidscore_momhs | nuts_rs | 3/3 | 0.1654 | 0.0539 | 0.09592 | 1.004 | 1266 | 0.05574 |
| seeds_data-seeds_stanified_model | nuts_cov_diag | 3/3 | 0.3304 | 0.08675 | 0.1667 | 1.016 | 282.7 | — |
| seeds_data-seeds_stanified_model | nuts_grad_diag_jitter | 3/3 | 0.2284 | 0.06049 | 0.1108 | 1.016 | 187.9 | — |
| seeds_data-seeds_stanified_model | nuts_rs | 3/3 | 0.1853 | 0.04447 | 0.08523 | 1.02 | 236.2 | — |
| surgical_data-surgical_model | nuts_cov_diag | 3/3 | 0.1215 | 0.02497 | 0.0454 | 1.015 | 258.7 | — |
| surgical_data-surgical_model | nuts_grad_diag_jitter | 3/3 | 0.1701 | 0.03022 | 0.04841 | 1.006 | 920.6 | — |
| surgical_data-surgical_model | nuts_rs | 3/3 | 0.08606 | 0.01583 | 0.02376 | 1.004 | 988.4 | — |
| radon_mn-radon_variable_intercept_noncentered | nuts_cov_diag | 3/3 | 2.753 | 1.417 | 1.076 | 1.006 | 1453 | — |
| radon_mn-radon_variable_intercept_noncentered | nuts_grad_diag_jitter | 3/3 | 1.242 | 0.3904 | 0.5883 | 1.005 | 901.7 | — |
| radon_mn-radon_variable_intercept_noncentered | nuts_rs | 3/3 | 1.207 | 0.3623 | 0.5481 | 1.004 | 801.1 | — |
| hmm_example-hmm_example | nuts_cov_diag | 3/3 | 1.636 | 0.6626 | 0.9451 | 1.005 | 1906 | 0.02213 |
| hmm_example-hmm_example | nuts_grad_diag_jitter | 3/3 | 0.8951 | 0.3163 | 0.5515 | 1.004 | 1407 | 0.05052 |
| hmm_example-hmm_example | nuts_rs | 3/3 | 0.8527 | 0.3126 | 0.5086 | 1.004 | 1522 | 0.01624 |

## Accounting and scope

- Descriptive native-backend observations; no promotion score, automatic selection or required ESS/s speedup.
- Twenty pinned PosteriorDB model/data pairs, not the entire PosteriorDB catalogue. All density ports must pass independent Stan density-difference, gradient and potential-HVP checks before native jobs; unavailable translations never fall back to callbacks.
- Alea and nuts-rs use the same ahead-of-time Enzyme Rust target in the native lane. This is both a native Rust workflow comparison and a shared-backend comparison of sampling plus adaptation. nuts-rs has no separate modeling compiler in this protocol. Shared-Stan companion runs hold BridgeStan constant instead.
- BlackJAX uses float64 CPU native JAX densities on every admitted case. CmdStan uses its compiled C++ model and Stan AD. Versions, compiler, source/data bytes, binaries, starts and seed identities are sealed. Seed labels do not imply equal random-number streams across engines.
- Cold process wall time uses perf_counter_ns and Linux pidfd completion. Rust Instant and synchronized JAX phase clocks separate warmup, retention and serialization; JIT time is recorded separately but already included in process wall time. Compilation outside jobs is recorded as batch preparation cost, with cache status explicit; it is not a comparable per-model cold compilation experiment.
- Native derivatives use no Stan calls. BridgeStan remains loaded for coordinate names and post-timing constrained output only; those costs appear in cold process/output time. Rust retained scopes include position collection; CmdStan includes native CSV/generated quantities. No pure-kernel claim is inferred from unequal output scopes.
- Serial mode pins all chains to one CPU and runs them sequentially. Parallel mode launches one worker per chain on distinct allowed CPUs, merges draws by logical chain identity, and computes pooled diagnostics afterward. Parallel end-to-end time includes worker setup and merging. The sum of individual chain RSS peaks is an upper bound, not simultaneous aggregate RSS.
- Every failed, timed-out, unsupported, divergent and repeated draw remains accounted for. Rank R-hat, bulk/tail ESS, MCSE and reference discrepancies accompany walltime, ESS/s and capability-specific API counts. Missing counters remain null; gradients, HVPs and third derivatives have no invented equal-cost conversion.
- Library-specific adaptation policies remain declared and may differ. Frozen-geometry kernel microbenchmarks are separate evidence; these workflows compare adaptation plus retained sampling.
- Research samplers remain opt-in. MCLMC is approximate; RMHMC is exactness-pending even with real Enzyme third derivatives. Dense manifold methods use separate dimension and iteration budgets; no default or maturity changes follow from this experiment.
- Stan 2.38 HMM simplex coordinates use the ILR transform (sqrt(2)*q for K=2), not the older stick-breaking coordinate. Dogs uses a stable native Bernoulli-logit expression; an independent Decimal80 oracle validates density/score because the original inv_logit-then-bernoulli Stan density loses about2e-6 at a shared start. The actual Stan value is preserved in conformance artifacts.
- Conformance includes shared starts, deterministic perturbations/tails and reference-mean-derived coordinates where all base parameters have reference summaries. Potential HVP signs are checked against independent score differences, and dense third derivatives against Hessian differences at the first finite point. Reference-free models retain that explicit coverage limitation.
- Performance repair follow-up. nuts_grad_diag_jitter explicitly selects paired position/score diagonal estimation, continuous dual averaging with late symmetric acceptance, and independent10%step jitter once per trajectory in both warmup and retention. The gradient final restart is explicitly disabled to follow the nuts-rs policy more closely. Covariance remains unchanged with zero jitter. Initial score floor and repeated live-state inclusion still differ; equal seed labels do not imply equal RNG streams.
- Warmup tracing is disabled by this sealed protocol; actual work counts, summaries and every retained outcome/diagnostic remain recorded. The three previous repair cohorts are preserved separately, including poor mixing and negative performance results. This is descriptive evidence, not automatic policy selection.
