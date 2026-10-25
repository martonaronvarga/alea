# Descriptive PosteriorDB benchmark

20 posteriors, 3 method configurations, 180 planned jobs. Status counts: `{'completed': 180}`.

Every failure and unsupported combination is retained. Diagnostics and accuracy are observations; this report makes no promotion, pass/fail or default-selection decision.

Cold wall time includes process startup, adaptation, sampling and serialization. Retained timings have engine-specific scopes; JAX host callbacks and native-JAX translations are labelled separately. RMHMC is exactness-pending; MCLMC is approximate; ATLAS/Pathfinder rows are labelled composed workflows.

| Posterior | Method | Completed/planned | Process s | Warmup s | Retained s | Worst R-hat | Min bulk ESS | Max mean error / ref SD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rate_1_data-Rate_1_model | nuts_cov_diag | 3/3 | 0.02256 | 0.002163 | 0.003491 | 1.003 | 1403 | 0.02383 |
| Rate_1_data-Rate_1_model | nuts_grad_diag | 3/3 | 0.01917 | 0.00205 | 0.003134 | 1.003 | 1490 | 0.02244 |
| Rate_1_data-Rate_1_model | nuts_rs | 3/3 | 0.01786 | 0.002238 | 0.00384 | 1.004 | 1593 | 0.002573 |
| Rate_2_data-Rate_2_model | nuts_cov_diag | 3/3 | 0.0274 | 0.00443 | 0.006129 | 1.003 | 2846 | 0.02411 |
| Rate_2_data-Rate_2_model | nuts_grad_diag | 3/3 | 0.02334 | 0.0031 | 0.004489 | 1.002 | 3505 | 0.01185 |
| Rate_2_data-Rate_2_model | nuts_rs | 3/3 | 0.02484 | 0.004012 | 0.005426 | 1.002 | 4022 | 0.03746 |
| Rate_5_data-Rate_5_model | nuts_cov_diag | 3/3 | 0.01813 | 0.001977 | 0.003611 | 1.003 | 1535 | 0.01174 |
| Rate_5_data-Rate_5_model | nuts_grad_diag | 3/3 | 0.01823 | 0.002083 | 0.003007 | 1.003 | 1770 | 0.02037 |
| Rate_5_data-Rate_5_model | nuts_rs | 3/3 | 0.01911 | 0.002481 | 0.004334 | 1.003 | 1648 | 0.01349 |
| arK-arK | nuts_cov_diag | 3/3 | 0.9086 | 0.2957 | 0.59 | 1.003 | 2467 | 0.03194 |
| arK-arK | nuts_grad_diag | 3/3 | 0.6257 | 0.2144 | 0.3828 | 1.003 | 2075 | 0.03092 |
| arK-arK | nuts_rs | 3/3 | 0.5795 | 0.1952 | 0.3573 | 1.004 | 1672 | 0.02995 |
| arma-arma11 | nuts_cov_diag | 3/3 | 0.1315 | 0.04342 | 0.06727 | 1.003 | 3106 | 0.01909 |
| arma-arma11 | nuts_grad_diag | 3/3 | 0.08841 | 0.03011 | 0.04085 | 1.001 | 4471 | 0.02936 |
| arma-arma11 | nuts_rs | 3/3 | 0.08749 | 0.02818 | 0.04197 | 1.002 | 4556 | 0.02152 |
| garch-garch11 | nuts_cov_diag | 3/3 | 0.9545 | 0.3375 | 0.603 | 1.004 | 1864 | 0.04114 |
| garch-garch11 | nuts_grad_diag | 3/3 | 0.6396 | 0.2229 | 0.3974 | 1.004 | 1429 | 0.0399 |
| garch-garch11 | nuts_rs | 3/3 | 0.5677 | 0.2055 | 0.3462 | 1.005 | 1398 | 0.02294 |
| dogs-dogs_log | nuts_cov_diag | 3/3 | 1.067 | 0.3739 | 0.7042 | 1.004 | 1215 | — |
| dogs-dogs_log | nuts_grad_diag | 3/3 | 0.7905 | 0.2875 | 0.4888 | 1.005 | 1176 | — |
| dogs-dogs_log | nuts_rs | 3/3 | 0.774 | 0.2759 | 0.4834 | 1.004 | 1098 | — |
| earnings-earn_height | nuts_cov_diag | 3/3 | 4.345 | 1.702 | 2.743 | 1.003 | 989.8 | 0.04742 |
| earnings-earn_height | nuts_grad_diag | 3/3 | 1.9 | 0.5743 | 1.308 | 1.008 | 482.9 | 0.02654 |
| earnings-earn_height | nuts_rs | 3/3 | 1.487 | 0.4716 | 0.9939 | 1.01 | 475.4 | 0.03747 |
| earnings-logearn_interaction_z | nuts_cov_diag | 3/3 | 1.294 | 0.4268 | 0.8476 | 1.003 | 1853 | 0.02076 |
| earnings-logearn_interaction_z | nuts_grad_diag | 3/3 | 0.9619 | 0.3368 | 0.6014 | 1.005 | 1594 | 0.04083 |
| earnings-logearn_interaction_z | nuts_rs | 3/3 | 0.855 | 0.299 | 0.5315 | 1.004 | 1427 | 0.03627 |
| eight_schools-eight_schools_centered | nuts_cov_diag | 3/3 | 0.1903 | 0.05831 | 0.09804 | 1.124 | 25.96 | — |
| eight_schools-eight_schools_centered | nuts_grad_diag | 3/3 | 0.09726 | 0.02376 | 0.04409 | 1.039 | 124.8 | — |
| eight_schools-eight_schools_centered | nuts_rs | 3/3 | 0.0904 | 0.02065 | 0.03936 | 1.091 | 33.75 | — |
| eight_schools-eight_schools_noncentered | nuts_cov_diag | 3/3 | 0.08962 | 0.0193 | 0.02943 | 1.003 | 1951 | 0.03235 |
| eight_schools-eight_schools_noncentered | nuts_grad_diag | 3/3 | 0.07555 | 0.01391 | 0.02284 | 1.004 | 2118 | 0.03015 |
| eight_schools-eight_schools_noncentered | nuts_rs | 3/3 | 0.07316 | 0.01186 | 0.02308 | 1.006 | 2027 | 0.03068 |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | nuts_cov_diag | 3/3 | 9.212 | 2.905 | 6.341 | 1.223 | 13.54 | — |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | nuts_grad_diag | 3/3 | 5.954 | 2.11 | 3.93 | 1.289 | 11.13 | — |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | nuts_rs | 3/3 | 4.996 | 1.816 | 3.157 | 1.35 | 9.607 | — |
| normal_2-normal_mixture | nuts_cov_diag | 3/3 | 1.199 | 0.4206 | 0.7496 | 1.529 | 7.133 | — |
| normal_2-normal_mixture | nuts_grad_diag | 3/3 | 0.8477 | 0.32 | 0.5092 | 1.535 | 7.135 | — |
| normal_2-normal_mixture | nuts_rs | 3/3 | 0.9313 | 0.3581 | 0.5546 | 1.529 | 7.14 | — |
| sblri-blr | nuts_cov_diag | 3/3 | 0.1589 | 0.05481 | 0.08154 | 1.014 | 644.6 | 0.05791 |
| sblri-blr | nuts_grad_diag | 3/3 | 0.0864 | 0.0245 | 0.03923 | 1.002 | 3872 | 0.02512 |
| sblri-blr | nuts_rs | 3/3 | 0.08112 | 0.02375 | 0.03689 | 1.003 | 3993 | 0.02258 |
| wells_data-wells_dist100_model | nuts_cov_diag | 3/3 | 4.91 | 1.709 | 3.183 | 1.004 | 1147 | 0.0378 |
| wells_data-wells_dist100_model | nuts_grad_diag | 3/3 | 4.059 | 1.559 | 2.481 | 1.006 | 1071 | 0.01931 |
| wells_data-wells_dist100_model | nuts_rs | 3/3 | 3.973 | 1.295 | 2.571 | 1.002 | 1085 | 0.02689 |
| kidiq-kidscore_momhs | nuts_cov_diag | 3/3 | 0.3439 | 0.1619 | 0.1628 | 1.003 | 1453 | 0.01561 |
| kidiq-kidscore_momhs | nuts_grad_diag | 3/3 | 0.1867 | 0.05927 | 0.1113 | 1.005 | 1233 | 0.02496 |
| kidiq-kidscore_momhs | nuts_rs | 3/3 | 0.1673 | 0.05283 | 0.09942 | 1.004 | 1266 | 0.05574 |
| seeds_data-seeds_stanified_model | nuts_cov_diag | 3/3 | 0.306 | 0.0868 | 0.1598 | 1.016 | 282.7 | — |
| seeds_data-seeds_stanified_model | nuts_grad_diag | 3/3 | 0.3066 | 0.06423 | 0.1042 | 1.528 | 7.421 | — |
| seeds_data-seeds_stanified_model | nuts_rs | 3/3 | 0.2067 | 0.05239 | 0.08956 | 1.02 | 236.2 | — |
| surgical_data-surgical_model | nuts_cov_diag | 3/3 | 0.1425 | 0.02426 | 0.04727 | 1.015 | 258.7 | — |
| surgical_data-surgical_model | nuts_grad_diag | 3/3 | 0.1661 | 0.02684 | 0.03484 | 1.014 | 285.1 | — |
| surgical_data-surgical_model | nuts_rs | 3/3 | 0.08898 | 0.01394 | 0.0243 | 1.004 | 988.4 | — |
| radon_mn-radon_variable_intercept_noncentered | nuts_cov_diag | 3/3 | 2.802 | 1.447 | 1.069 | 1.006 | 1453 | — |
| radon_mn-radon_variable_intercept_noncentered | nuts_grad_diag | 3/3 | 1.178 | 0.3763 | 0.5375 | 1.006 | 912.1 | — |
| radon_mn-radon_variable_intercept_noncentered | nuts_rs | 3/3 | 1.123 | 0.3533 | 0.5199 | 1.004 | 801.1 | — |
| hmm_example-hmm_example | nuts_cov_diag | 3/3 | 1.709 | 0.7398 | 0.9576 | 1.005 | 1906 | 0.02213 |
| hmm_example-hmm_example | nuts_grad_diag | 3/3 | 0.9092 | 0.3138 | 0.5603 | 1.004 | 1574 | 0.02471 |
| hmm_example-hmm_example | nuts_rs | 3/3 | 0.8451 | 0.2991 | 0.5187 | 1.004 | 1522 | 0.01624 |

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
- NUTS performance repair follow-up. Covariance NUTS remains the existing reference; nuts_grad_diag explicitly selects conventional paired position/score diagonal adaptation, growing background windows, and late symmetric acceptance. Its retained kernel uses the same enhanced generalized turning as covariance NUTS. This is an opt-in workflow, not an automatic default replacement.
- Detailed Alea warmup tracing is disabled by this sealed protocol. Prior v3 workflows traced Alea during timed warmup but did not trace nuts-rs. Final warmup summaries, failures, work counters and retained diagnostics remain recorded. The historical traced results remain immutable and are not a single-factor kernel comparison.
- Gradient-informed initialization clips abs(score) below at1 rather than nuts-rs1e-20. All live repeated states enter paired moments, unlike the nuts-rs accepted-state filter. nuts-rs retains its own turning, jitter and warmup policy. Equal seed labels do not force equal random streams or trajectories.
