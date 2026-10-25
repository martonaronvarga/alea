# Descriptive PosteriorDB benchmark

20 posteriors, 3 method configurations, 180 planned jobs. Status counts: `{'completed': 180}`.

Every failure and unsupported combination is retained. Diagnostics and accuracy are observations; this report makes no promotion, pass/fail or default-selection decision.

Cold wall time includes process startup, adaptation, sampling and serialization. Retained timings have engine-specific scopes; JAX host callbacks and native-JAX translations are labelled separately. RMHMC is exactness-pending; MCLMC is approximate; ATLAS/Pathfinder rows are labelled composed workflows.

| Posterior | Method | Completed/planned | Process s | Warmup s | Retained s | Worst R-hat | Min bulk ESS | Max mean error / ref SD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rate_1_data-Rate_1_model | nuts_cov_diag | 3/3 | 0.01854 | 0.002019 | 0.003037 | 1.003 | 1403 | 0.02383 |
| Rate_1_data-Rate_1_model | nuts_grad_diag | 3/3 | 0.01863 | 0.002117 | 0.003081 | 1.002 | 1279 | 0.01377 |
| Rate_1_data-Rate_1_model | nuts_rs | 3/3 | 0.02058 | 0.003175 | 0.004869 | 1.004 | 1593 | 0.002573 |
| Rate_2_data-Rate_2_model | nuts_cov_diag | 3/3 | 0.02656 | 0.003672 | 0.006516 | 1.003 | 2846 | 0.02411 |
| Rate_2_data-Rate_2_model | nuts_grad_diag | 3/3 | 0.02428 | 0.003209 | 0.005284 | 1.003 | 3685 | 0.02082 |
| Rate_2_data-Rate_2_model | nuts_rs | 3/3 | 0.02376 | 0.0033 | 0.006186 | 1.002 | 4022 | 0.03746 |
| Rate_5_data-Rate_5_model | nuts_cov_diag | 3/3 | 0.01798 | 0.002143 | 0.003201 | 1.003 | 1535 | 0.01174 |
| Rate_5_data-Rate_5_model | nuts_grad_diag | 3/3 | 0.02825 | 0.003391 | 0.005356 | 1.002 | 1450 | 0.02877 |
| Rate_5_data-Rate_5_model | nuts_rs | 3/3 | 0.02202 | 0.002822 | 0.004747 | 1.003 | 1648 | 0.01349 |
| arK-arK | nuts_cov_diag | 3/3 | 0.913 | 0.3074 | 0.5797 | 1.003 | 2467 | 0.03194 |
| arK-arK | nuts_grad_diag | 3/3 | 0.7287 | 0.2264 | 0.4762 | 1.004 | 1667 | 0.04817 |
| arK-arK | nuts_rs | 3/3 | 0.5632 | 0.1948 | 0.3457 | 1.004 | 1672 | 0.02995 |
| arma-arma11 | nuts_cov_diag | 3/3 | 0.1261 | 0.04476 | 0.06415 | 1.003 | 3106 | 0.01909 |
| arma-arma11 | nuts_grad_diag | 3/3 | 0.105 | 0.031 | 0.05571 | 1.001 | 3444 | 0.02275 |
| arma-arma11 | nuts_rs | 3/3 | 0.08695 | 0.02836 | 0.04192 | 1.002 | 4556 | 0.02152 |
| garch-garch11 | nuts_cov_diag | 3/3 | 1.036 | 0.3554 | 0.6601 | 1.004 | 1864 | 0.04114 |
| garch-garch11 | nuts_grad_diag | 3/3 | 0.7195 | 0.2339 | 0.4665 | 1.005 | 1323 | 0.04141 |
| garch-garch11 | nuts_rs | 3/3 | 0.5805 | 0.2064 | 0.3568 | 1.005 | 1398 | 0.02294 |
| dogs-dogs_log | nuts_cov_diag | 3/3 | 1.088 | 0.395 | 0.6748 | 1.004 | 1215 | — |
| dogs-dogs_log | nuts_grad_diag | 3/3 | 0.9902 | 0.3443 | 0.6265 | 1.003 | 1141 | — |
| dogs-dogs_log | nuts_rs | 3/3 | 1.001 | 0.3604 | 0.6232 | 1.004 | 1098 | — |
| earnings-earn_height | nuts_cov_diag | 3/3 | 4.345 | 1.662 | 2.673 | 1.003 | 989.8 | 0.04742 |
| earnings-earn_height | nuts_grad_diag | 3/3 | 2.013 | 0.5869 | 1.408 | 1.007 | 419.5 | 0.05195 |
| earnings-earn_height | nuts_rs | 3/3 | 1.403 | 0.448 | 0.9371 | 1.01 | 475.4 | 0.03747 |
| earnings-logearn_interaction_z | nuts_cov_diag | 3/3 | 1.3 | 0.4244 | 0.8204 | 1.003 | 1853 | 0.02076 |
| earnings-logearn_interaction_z | nuts_grad_diag | 3/3 | 1.066 | 0.3479 | 0.7042 | 1.003 | 1426 | 0.03554 |
| earnings-logearn_interaction_z | nuts_rs | 3/3 | 0.8173 | 0.2916 | 0.5153 | 1.004 | 1427 | 0.03627 |
| eight_schools-eight_schools_centered | nuts_cov_diag | 3/3 | 0.1587 | 0.04998 | 0.05869 | 1.124 | 25.96 | — |
| eight_schools-eight_schools_centered | nuts_grad_diag | 3/3 | 0.1291 | 0.02697 | 0.05531 | 1.028 | 113.4 | — |
| eight_schools-eight_schools_centered | nuts_rs | 3/3 | 0.08868 | 0.0195 | 0.03753 | 1.091 | 33.75 | — |
| eight_schools-eight_schools_noncentered | nuts_cov_diag | 3/3 | 0.09577 | 0.02136 | 0.03233 | 1.003 | 1951 | 0.03235 |
| eight_schools-eight_schools_noncentered | nuts_grad_diag | 3/3 | 0.0762 | 0.01429 | 0.02335 | 1.004 | 2258 | 0.02385 |
| eight_schools-eight_schools_noncentered | nuts_rs | 3/3 | 0.08437 | 0.01459 | 0.02447 | 1.006 | 2027 | 0.03068 |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | nuts_cov_diag | 3/3 | 8.529 | 2.748 | 5.795 | 1.223 | 13.54 | — |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | nuts_grad_diag | 3/3 | 6.515 | 2.115 | 4.378 | 1.306 | 10.79 | — |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | nuts_rs | 3/3 | 4.451 | 1.679 | 2.752 | 1.35 | 9.607 | — |
| normal_2-normal_mixture | nuts_cov_diag | 3/3 | 1.241 | 0.4309 | 0.7918 | 1.529 | 7.133 | — |
| normal_2-normal_mixture | nuts_grad_diag | 3/3 | 0.8748 | 0.3039 | 0.5536 | 1.53 | 7.136 | — |
| normal_2-normal_mixture | nuts_rs | 3/3 | 0.8795 | 0.3351 | 0.5252 | 1.529 | 7.14 | — |
| sblri-blr | nuts_cov_diag | 3/3 | 0.1603 | 0.05302 | 0.0796 | 1.014 | 644.6 | 0.05791 |
| sblri-blr | nuts_grad_diag | 3/3 | 0.09399 | 0.02511 | 0.04714 | 1.003 | 3857 | 0.03225 |
| sblri-blr | nuts_rs | 3/3 | 0.08314 | 0.02367 | 0.03854 | 1.003 | 3993 | 0.02258 |
| wells_data-wells_dist100_model | nuts_cov_diag | 3/3 | 6.067 | 2.089 | 3.955 | 1.004 | 1147 | 0.0378 |
| wells_data-wells_dist100_model | nuts_grad_diag | 3/3 | 4.285 | 1.39 | 2.888 | 1.006 | 1037 | 0.03077 |
| wells_data-wells_dist100_model | nuts_rs | 3/3 | 3.345 | 1.232 | 2.166 | 1.002 | 1085 | 0.02689 |
| kidiq-kidscore_momhs | nuts_cov_diag | 3/3 | 0.3468 | 0.1644 | 0.1574 | 1.003 | 1453 | 0.01561 |
| kidiq-kidscore_momhs | nuts_grad_diag | 3/3 | 0.204 | 0.06236 | 0.125 | 1.003 | 1169 | 0.02417 |
| kidiq-kidscore_momhs | nuts_rs | 3/3 | 0.1655 | 0.05383 | 0.09676 | 1.004 | 1266 | 0.05574 |
| seeds_data-seeds_stanified_model | nuts_cov_diag | 3/3 | 0.3313 | 0.08966 | 0.1805 | 1.016 | 282.7 | — |
| seeds_data-seeds_stanified_model | nuts_grad_diag | 3/3 | 0.287 | 0.06776 | 0.163 | 1.028 | 166.3 | — |
| seeds_data-seeds_stanified_model | nuts_rs | 3/3 | 0.1909 | 0.04737 | 0.08671 | 1.02 | 236.2 | — |
| surgical_data-surgical_model | nuts_cov_diag | 3/3 | 0.1421 | 0.02934 | 0.05037 | 1.015 | 258.7 | — |
| surgical_data-surgical_model | nuts_grad_diag | 3/3 | 0.111 | 0.02124 | 0.03187 | 1.025 | 194.2 | — |
| surgical_data-surgical_model | nuts_rs | 3/3 | 0.08958 | 0.01422 | 0.02361 | 1.004 | 988.4 | — |
| radon_mn-radon_variable_intercept_noncentered | nuts_cov_diag | 3/3 | 2.801 | 1.46 | 1.087 | 1.006 | 1453 | — |
| radon_mn-radon_variable_intercept_noncentered | nuts_grad_diag | 3/3 | 1.346 | 0.3916 | 0.6924 | 1.015 | 829.5 | — |
| radon_mn-radon_variable_intercept_noncentered | nuts_rs | 3/3 | 1.139 | 0.358 | 0.5315 | 1.004 | 801.1 | — |
| hmm_example-hmm_example | nuts_cov_diag | 3/3 | 2.15 | 0.8848 | 1.222 | 1.005 | 1906 | 0.02213 |
| hmm_example-hmm_example | nuts_grad_diag | 3/3 | 1.369 | 0.4203 | 0.8859 | 1.004 | 1270 | 0.02948 |
| hmm_example-hmm_example | nuts_rs | 3/3 | 1.051 | 0.3502 | 0.6639 | 1.004 | 1522 | 0.01624 |

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
- Follow-up to the initial parity cohort: gradient-informed warmup now restarts bounded search and dual averaging when geometry freezes, using the final fixed-geometry phase to tune retention. All20posteriors, original seeds, competitors and diagnostics are retained; the first cohort is preserved separately, including its seeds/surgical weaknesses.
