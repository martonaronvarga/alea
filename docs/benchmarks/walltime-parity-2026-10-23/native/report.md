# Descriptive PosteriorDB benchmark

20 posteriors, 3 method configurations, 180 planned jobs. Status counts: `{'completed': 180}`.

Every failure and unsupported combination is retained. Diagnostics and accuracy are observations; this report makes no promotion, pass/fail or default-selection decision.

Cold wall time includes process startup, adaptation, sampling and serialization. Retained timings have engine-specific scopes; JAX host callbacks and native-JAX translations are labelled separately. RMHMC is exactness-pending; MCLMC is approximate; ATLAS/Pathfinder rows are labelled composed workflows.

| Posterior | Method | Completed/planned | Process s | Warmup s | Retained s | Worst R-hat | Min bulk ESS | Max mean error / ref SD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rate_1_data-Rate_1_model | nuts_grad_diag_jitter | 3/3 | 0.01803 | 0.00196 | 0.003083 | 1.003 | 1714 | 0.02257 |
| Rate_1_data-Rate_1_model | nuts_grad_diag_position | 3/3 | 0.01842 | 0.002199 | 0.00321 | 1.002 | 1443 | 0.02857 |
| Rate_1_data-Rate_1_model | nuts_rs | 3/3 | 0.01704 | 0.00204 | 0.003594 | 1.004 | 1593 | 0.002573 |
| Rate_2_data-Rate_2_model | nuts_grad_diag_jitter | 3/3 | 0.02339 | 0.003081 | 0.004598 | 1.003 | 3750 | 0.02494 |
| Rate_2_data-Rate_2_model | nuts_grad_diag_position | 3/3 | 0.02384 | 0.003324 | 0.004846 | 1.003 | 4015 | 0.03346 |
| Rate_2_data-Rate_2_model | nuts_rs | 3/3 | 0.02263 | 0.003359 | 0.005136 | 1.002 | 4022 | 0.03746 |
| Rate_5_data-Rate_5_model | nuts_grad_diag_jitter | 3/3 | 0.01779 | 0.002089 | 0.003163 | 1.003 | 1578 | 0.0124 |
| Rate_5_data-Rate_5_model | nuts_grad_diag_position | 3/3 | 0.01771 | 0.002104 | 0.00321 | 1.004 | 1741 | 0.02381 |
| Rate_5_data-Rate_5_model | nuts_rs | 3/3 | 0.01736 | 0.002368 | 0.003709 | 1.003 | 1648 | 0.01349 |
| arK-arK | nuts_grad_diag_jitter | 3/3 | 0.6882 | 0.2254 | 0.4301 | 1.004 | 1963 | 0.01939 |
| arK-arK | nuts_grad_diag_position | 3/3 | 1.01 | 0.2172 | 0.3388 | 1.002 | 1613 | 0.03059 |
| arK-arK | nuts_rs | 3/3 | 0.5753 | 0.205 | 0.3479 | 1.004 | 1672 | 0.02995 |
| arma-arma11 | nuts_grad_diag_jitter | 3/3 | 0.08896 | 0.02766 | 0.04318 | 1.002 | 5091 | 0.01897 |
| arma-arma11 | nuts_grad_diag_position | 3/3 | 0.0882 | 0.02745 | 0.04141 | 1.003 | 4216 | 0.01808 |
| arma-arma11 | nuts_rs | 3/3 | 0.08724 | 0.02899 | 0.04182 | 1.002 | 4556 | 0.02152 |
| garch-garch11 | nuts_grad_diag_jitter | 3/3 | 0.9128 | 0.3253 | 0.5593 | 1.005 | 1313 | 0.04059 |
| garch-garch11 | nuts_grad_diag_position | 3/3 | 0.6611 | 0.242 | 0.3896 | 1.006 | 1226 | 0.01873 |
| garch-garch11 | nuts_rs | 3/3 | 0.6431 | 0.2474 | 0.3749 | 1.005 | 1398 | 0.02294 |
| dogs-dogs_log | nuts_grad_diag_jitter | 3/3 | 1.194 | 0.4421 | 0.7287 | 1.008 | 839.9 | — |
| dogs-dogs_log | nuts_grad_diag_position | 3/3 | 1.072 | 0.3689 | 0.6597 | 1.006 | 908.3 | — |
| dogs-dogs_log | nuts_rs | 3/3 | 1.054 | 0.3585 | 0.6738 | 1.004 | 1098 | — |
| earnings-earn_height | nuts_grad_diag_jitter | 3/3 | 1.841 | 0.5633 | 1.258 | 1.011 | 576.5 | 0.03068 |
| earnings-earn_height | nuts_grad_diag_position | 3/3 | 1.413 | 0.4485 | 0.9471 | 1.015 | 428.6 | 0.03777 |
| earnings-earn_height | nuts_rs | 3/3 | 1.591 | 0.5636 | 0.9823 | 1.01 | 475.4 | 0.03747 |
| earnings-logearn_interaction_z | nuts_grad_diag_jitter | 3/3 | 1.042 | 0.3555 | 0.6545 | 1.003 | 1476 | 0.03556 |
| earnings-logearn_interaction_z | nuts_grad_diag_position | 3/3 | 0.8359 | 0.2888 | 0.5187 | 1.003 | 1493 | 0.02962 |
| earnings-logearn_interaction_z | nuts_rs | 3/3 | 0.8399 | 0.302 | 0.5158 | 1.004 | 1427 | 0.03627 |
| eight_schools-eight_schools_centered | nuts_grad_diag_jitter | 3/3 | 0.1171 | 0.02597 | 0.05247 | 1.04 | 160 | — |
| eight_schools-eight_schools_centered | nuts_grad_diag_position | 3/3 | 0.1191 | 0.03308 | 0.04521 | 1.283 | 11.04 | — |
| eight_schools-eight_schools_centered | nuts_rs | 3/3 | 0.07875 | 0.01773 | 0.03345 | 1.091 | 33.75 | — |
| eight_schools-eight_schools_noncentered | nuts_grad_diag_jitter | 3/3 | 0.08749 | 0.01394 | 0.02844 | 1.004 | 1900 | 0.03568 |
| eight_schools-eight_schools_noncentered | nuts_grad_diag_position | 3/3 | 0.0769 | 0.01299 | 0.02422 | 1.004 | 1901 | 0.02334 |
| eight_schools-eight_schools_noncentered | nuts_rs | 3/3 | 0.07282 | 0.0126 | 0.02122 | 1.006 | 2027 | 0.03068 |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | nuts_grad_diag_jitter | 3/3 | 5.881 | 2.046 | 3.933 | 1.252 | 11.99 | — |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | nuts_grad_diag_position | 3/3 | 4.727 | 1.722 | 2.981 | 1.274 | 11.09 | — |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | nuts_rs | 3/3 | 4.96 | 1.791 | 3.073 | 1.35 | 9.607 | — |
| normal_2-normal_mixture | nuts_grad_diag_jitter | 3/3 | 0.8774 | 0.3251 | 0.5447 | 1.531 | 7.147 | — |
| normal_2-normal_mixture | nuts_grad_diag_position | 3/3 | 0.8132 | 0.3015 | 0.4941 | 1.529 | 7.14 | — |
| normal_2-normal_mixture | nuts_rs | 3/3 | 0.8064 | 0.3037 | 0.4855 | 1.529 | 7.14 | — |
| sblri-blr | nuts_grad_diag_jitter | 3/3 | 0.1032 | 0.02683 | 0.04424 | 1.003 | 4413 | 0.03957 |
| sblri-blr | nuts_grad_diag_position | 3/3 | 0.0883 | 0.0255 | 0.03879 | 1.002 | 4350 | 0.03336 |
| sblri-blr | nuts_rs | 3/3 | 0.0827 | 0.02367 | 0.03821 | 1.003 | 3993 | 0.02258 |
| wells_data-wells_dist100_model | nuts_grad_diag_jitter | 3/3 | 3.991 | 1.508 | 2.462 | 1.004 | 1123 | 0.01771 |
| wells_data-wells_dist100_model | nuts_grad_diag_position | 3/3 | 3.632 | 1.236 | 2.377 | 1.008 | 883.5 | 0.02361 |
| wells_data-wells_dist100_model | nuts_rs | 3/3 | 3.623 | 1.312 | 2.294 | 1.002 | 1085 | 0.02689 |
| kidiq-kidscore_momhs | nuts_grad_diag_jitter | 3/3 | 0.3308 | 0.1097 | 0.1924 | 1.003 | 1446 | 0.02362 |
| kidiq-kidscore_momhs | nuts_grad_diag_position | 3/3 | 0.1873 | 0.06211 | 0.1067 | 1.002 | 1268 | 0.02383 |
| kidiq-kidscore_momhs | nuts_rs | 3/3 | 0.1734 | 0.05462 | 0.1027 | 1.004 | 1266 | 0.05574 |
| seeds_data-seeds_stanified_model | nuts_grad_diag_jitter | 3/3 | 0.2128 | 0.05955 | 0.09383 | 1.539 | 7.254 | — |
| seeds_data-seeds_stanified_model | nuts_grad_diag_position | 3/3 | 0.2953 | 0.08127 | 0.12 | 1.013 | 314.3 | — |
| seeds_data-seeds_stanified_model | nuts_rs | 3/3 | 0.2856 | 0.06821 | 0.1273 | 1.02 | 236.2 | — |
| surgical_data-surgical_model | nuts_grad_diag_jitter | 3/3 | 0.1326 | 0.02516 | 0.04213 | 1.006 | 920.6 | — |
| surgical_data-surgical_model | nuts_grad_diag_position | 3/3 | 0.09538 | 0.01851 | 0.0272 | 1.012 | 658.8 | — |
| surgical_data-surgical_model | nuts_rs | 3/3 | 0.1023 | 0.01728 | 0.02743 | 1.004 | 988.4 | — |
| radon_mn-radon_variable_intercept_noncentered | nuts_grad_diag_jitter | 3/3 | 1.154 | 0.3571 | 0.5463 | 1.006 | 828.4 | — |
| radon_mn-radon_variable_intercept_noncentered | nuts_grad_diag_position | 3/3 | 1.127 | 0.3565 | 0.5205 | 1.007 | 710.8 | — |
| radon_mn-radon_variable_intercept_noncentered | nuts_rs | 3/3 | 1.171 | 0.3556 | 0.5464 | 1.004 | 801.1 | — |
| hmm_example-hmm_example | nuts_grad_diag_jitter | 3/3 | 0.8896 | 0.3186 | 0.5418 | 1.004 | 1407 | 0.05052 |
| hmm_example-hmm_example | nuts_grad_diag_position | 3/3 | 0.7998 | 0.2819 | 0.4934 | 1.003 | 1341 | 0.03273 |
| hmm_example-hmm_example | nuts_rs | 3/3 | 0.8155 | 0.3021 | 0.4866 | 1.004 | 1522 | 0.01624 |

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
- Walltime follow-up: both Alea profiles use runtime SIMD finite checks, noncontracting kick/drift arithmetic, owned point buffer exchange and cached norm turning certificates. nuts_grad_diag_position additionally uses the mass-whitened endpoint displacement criterion expressed as delta(q) dot canonical p; the other profile retains generalized momentum-sum turning. Both retain enhanced cross-subtree intervals and eager numerical admission. Same jitter/adaptation/starts/budgets as the preceding cohort. This descriptive protocol does not change the constructor default.
- Ordinary relative-energy weights share one leaf exponential and merge as positive linear masses; extreme energies retain logarithmic max/scaled sums. Tree carries swap fixed pointers. Identity/inverse-diagonal energies use nonnegative half-scaled SIMD reductions at16+coordinates, and Euclidean phase starts at16+coordinates use Rand Ziggurat normals. These are declared implementation/RNG arithmetic changes, not an assertion of old draw-byte identity.
