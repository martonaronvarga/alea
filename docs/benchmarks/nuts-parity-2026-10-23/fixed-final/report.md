# Descriptive PosteriorDB benchmark

20 posteriors, 3 method configurations, 180 planned jobs. Status counts: `{'completed': 180}`.

Every failure and unsupported combination is retained. Diagnostics and accuracy are observations; this report makes no promotion, pass/fail or default-selection decision.

Cold wall time includes process startup, adaptation, sampling and serialization. Retained timings have engine-specific scopes; JAX host callbacks and native-JAX translations are labelled separately. RMHMC is exactness-pending; MCLMC is approximate; ATLAS/Pathfinder rows are labelled composed workflows.

| Posterior | Method | Completed/planned | Process s | Warmup s | Retained s | Worst R-hat | Min bulk ESS | Max mean error / ref SD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rate_1_data-Rate_1_model | nuts_cov_diag | 3/3 | 0.01872 | 0.002138 | 0.003154 | 1.003 | 1403 | 0.02383 |
| Rate_1_data-Rate_1_model | nuts_grad_diag | 3/3 | 0.01903 | 0.002178 | 0.003048 | 1.003 | 1399 | 0.03441 |
| Rate_1_data-Rate_1_model | nuts_rs | 3/3 | 0.02044 | 0.002477 | 0.004005 | 1.004 | 1593 | 0.002573 |
| Rate_2_data-Rate_2_model | nuts_cov_diag | 3/3 | 0.02586 | 0.003589 | 0.006279 | 1.003 | 2846 | 0.02411 |
| Rate_2_data-Rate_2_model | nuts_grad_diag | 3/3 | 0.02434 | 0.003217 | 0.005177 | 1.001 | 3596 | 0.02038 |
| Rate_2_data-Rate_2_model | nuts_rs | 3/3 | 0.02306 | 0.003322 | 0.005026 | 1.002 | 4022 | 0.03746 |
| Rate_5_data-Rate_5_model | nuts_cov_diag | 3/3 | 0.01787 | 0.002001 | 0.003481 | 1.003 | 1535 | 0.01174 |
| Rate_5_data-Rate_5_model | nuts_grad_diag | 3/3 | 0.01815 | 0.001991 | 0.003171 | 1.004 | 1474 | 0.006733 |
| Rate_5_data-Rate_5_model | nuts_rs | 3/3 | 0.01714 | 0.002217 | 0.003555 | 1.003 | 1648 | 0.01349 |
| arK-arK | nuts_cov_diag | 3/3 | 0.8704 | 0.298 | 0.5483 | 1.003 | 2467 | 0.03194 |
| arK-arK | nuts_grad_diag | 3/3 | 0.7189 | 0.2274 | 0.4683 | 1.006 | 1822 | 0.03725 |
| arK-arK | nuts_rs | 3/3 | 0.5891 | 0.1965 | 0.3702 | 1.004 | 1672 | 0.02995 |
| arma-arma11 | nuts_cov_diag | 3/3 | 0.1291 | 0.04966 | 0.06509 | 1.003 | 3106 | 0.01909 |
| arma-arma11 | nuts_grad_diag | 3/3 | 0.1106 | 0.03225 | 0.05951 | 1.003 | 3487 | 0.02368 |
| arma-arma11 | nuts_rs | 3/3 | 0.09324 | 0.0308 | 0.04528 | 1.002 | 4556 | 0.02152 |
| garch-garch11 | nuts_cov_diag | 3/3 | 0.9792 | 0.3342 | 0.626 | 1.004 | 1864 | 0.04114 |
| garch-garch11 | nuts_grad_diag | 3/3 | 0.7446 | 0.2415 | 0.4772 | 1.003 | 1369 | 0.0332 |
| garch-garch11 | nuts_rs | 3/3 | 0.6428 | 0.2403 | 0.3832 | 1.005 | 1398 | 0.02294 |
| dogs-dogs_log | nuts_cov_diag | 3/3 | 1.415 | 0.5018 | 0.8152 | 1.004 | 1215 | — |
| dogs-dogs_log | nuts_grad_diag | 3/3 | 1.142 | 0.3728 | 0.7207 | 1.003 | 1121 | — |
| dogs-dogs_log | nuts_rs | 3/3 | 1.006 | 0.3727 | 0.6127 | 1.004 | 1098 | — |
| earnings-earn_height | nuts_cov_diag | 3/3 | 4.46 | 1.727 | 2.713 | 1.003 | 989.8 | 0.04742 |
| earnings-earn_height | nuts_grad_diag | 3/3 | 2.107 | 0.6091 | 1.517 | 1.011 | 538.8 | 0.02585 |
| earnings-earn_height | nuts_rs | 3/3 | 1.483 | 0.4876 | 0.9783 | 1.01 | 475.4 | 0.03747 |
| earnings-logearn_interaction_z | nuts_cov_diag | 3/3 | 1.319 | 0.4324 | 0.861 | 1.003 | 1853 | 0.02076 |
| earnings-logearn_interaction_z | nuts_grad_diag | 3/3 | 1.115 | 0.3581 | 0.7309 | 1.005 | 1601 | 0.04904 |
| earnings-logearn_interaction_z | nuts_rs | 3/3 | 0.8439 | 0.2974 | 0.5234 | 1.004 | 1427 | 0.03627 |
| eight_schools-eight_schools_centered | nuts_cov_diag | 3/3 | 0.1645 | 0.0526 | 0.07798 | 1.124 | 25.96 | — |
| eight_schools-eight_schools_centered | nuts_grad_diag | 3/3 | 0.1121 | 0.03168 | 0.05745 | 1.071 | 52.69 | — |
| eight_schools-eight_schools_centered | nuts_rs | 3/3 | 0.0843 | 0.01972 | 0.03743 | 1.091 | 33.75 | — |
| eight_schools-eight_schools_noncentered | nuts_cov_diag | 3/3 | 0.1312 | 0.02479 | 0.03212 | 1.003 | 1951 | 0.03235 |
| eight_schools-eight_schools_noncentered | nuts_grad_diag | 3/3 | 0.07779 | 0.01442 | 0.02402 | 1.004 | 2117 | 0.04721 |
| eight_schools-eight_schools_noncentered | nuts_rs | 3/3 | 0.07176 | 0.01314 | 0.02088 | 1.006 | 2027 | 0.03068 |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | nuts_cov_diag | 3/3 | 8.772 | 2.841 | 6.052 | 1.223 | 13.54 | — |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | nuts_grad_diag | 3/3 | 6.372 | 2.073 | 4.301 | 1.323 | 10.05 | — |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | nuts_rs | 3/3 | 4.358 | 1.58 | 2.759 | 1.35 | 9.607 | — |
| normal_2-normal_mixture | nuts_cov_diag | 3/3 | 1.217 | 0.4419 | 0.7561 | 1.529 | 7.133 | — |
| normal_2-normal_mixture | nuts_grad_diag | 3/3 | 1.123 | 0.3697 | 0.7319 | 1.529 | 7.162 | — |
| normal_2-normal_mixture | nuts_rs | 3/3 | 0.9462 | 0.3754 | 0.554 | 1.529 | 7.14 | — |
| sblri-blr | nuts_cov_diag | 3/3 | 0.1529 | 0.05157 | 0.0791 | 1.014 | 644.6 | 0.05791 |
| sblri-blr | nuts_grad_diag | 3/3 | 0.1094 | 0.02648 | 0.05543 | 1.003 | 4331 | 0.02523 |
| sblri-blr | nuts_rs | 3/3 | 0.08626 | 0.02356 | 0.04042 | 1.003 | 3993 | 0.02258 |
| wells_data-wells_dist100_model | nuts_cov_diag | 3/3 | 4.734 | 1.715 | 3.003 | 1.004 | 1147 | 0.0378 |
| wells_data-wells_dist100_model | nuts_grad_diag | 3/3 | 4.456 | 1.495 | 2.939 | 1.004 | 1071 | 0.02204 |
| wells_data-wells_dist100_model | nuts_rs | 3/3 | 3.88 | 1.31 | 2.444 | 1.002 | 1085 | 0.02689 |
| kidiq-kidscore_momhs | nuts_cov_diag | 3/3 | 0.3402 | 0.16 | 0.1615 | 1.003 | 1453 | 0.01561 |
| kidiq-kidscore_momhs | nuts_grad_diag | 3/3 | 0.2211 | 0.06573 | 0.1367 | 1.008 | 1004 | 0.04222 |
| kidiq-kidscore_momhs | nuts_rs | 3/3 | 0.1748 | 0.05509 | 0.1014 | 1.004 | 1266 | 0.05574 |
| seeds_data-seeds_stanified_model | nuts_cov_diag | 3/3 | 0.3089 | 0.08841 | 0.1618 | 1.016 | 282.7 | — |
| seeds_data-seeds_stanified_model | nuts_grad_diag | 3/3 | 0.3251 | 0.06343 | 0.1689 | 1.073 | 40.38 | — |
| seeds_data-seeds_stanified_model | nuts_rs | 3/3 | 0.185 | 0.04454 | 0.08517 | 1.02 | 236.2 | — |
| surgical_data-surgical_model | nuts_cov_diag | 3/3 | 0.1158 | 0.0246 | 0.04354 | 1.015 | 258.7 | — |
| surgical_data-surgical_model | nuts_grad_diag | 3/3 | 0.09896 | 0.01924 | 0.03123 | 1.004 | 1040 | — |
| surgical_data-surgical_model | nuts_rs | 3/3 | 0.08904 | 0.01404 | 0.02288 | 1.004 | 988.4 | — |
| radon_mn-radon_variable_intercept_noncentered | nuts_cov_diag | 3/3 | 2.741 | 1.412 | 1.069 | 1.006 | 1453 | — |
| radon_mn-radon_variable_intercept_noncentered | nuts_grad_diag | 3/3 | 1.647 | 0.4113 | 0.9856 | 1.01 | 1117 | — |
| radon_mn-radon_variable_intercept_noncentered | nuts_rs | 3/3 | 1.154 | 0.35 | 0.5303 | 1.004 | 801.1 | — |
| hmm_example-hmm_example | nuts_cov_diag | 3/3 | 1.655 | 0.6871 | 0.9396 | 1.005 | 1906 | 0.02213 |
| hmm_example-hmm_example | nuts_grad_diag | 3/3 | 1.02 | 0.332 | 0.6726 | 1.004 | 1226 | 0.03006 |
| hmm_example-hmm_example | nuts_rs | 3/3 | 0.7965 | 0.2901 | 0.4874 | 1.004 | 1522 | 0.01624 |

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
- Final fixed-geometry tuning uses ordinary mean tree acceptance, matching the covariance controller target convention. Symmetric acceptance is confined to the late phase with changing geometry. The initial and all-symmetric-restart cohorts remain separate evidence. No failed/poorly mixing posterior is removed.
