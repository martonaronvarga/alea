# Descriptive PosteriorDB benchmark

6 posteriors, 6 method configurations, 72 planned jobs. Status counts: `{'completed': 72}`.

Every failure and unsupported combination is retained. Diagnostics and accuracy are observations; this report makes no promotion, pass/fail or default-selection decision.

Cold wall time includes process startup, adaptation, sampling and serialization. Retained timings have engine-specific scopes; JAX host callbacks and native-JAX translations are labelled separately. RMHMC is exactness-pending; MCLMC is approximate; ATLAS/Pathfinder rows are labelled composed workflows.

| Posterior | Method | Completed/planned | Process s | Warmup s | Retained s | Worst R-hat | Min bulk ESS | Max mean error / ref SD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rate_2_data-Rate_2_model | nuts_cov_diag | 2/2 | 0.5546 | 0.01489 | 0.01408 | 1.003 | 3100 | 0.02389 |
| Rate_2_data-Rate_2_model | nuts_cov_dense | 2/2 | 0.4508 | 0.01285 | 0.01467 | 1.002 | 2570 | 0.01011 |
| Rate_2_data-Rate_2_model | nuts_rs | 2/2 | 0.2946 | 0.006222 | 0.01068 | 1.002 | 4024 | 0.03428 |
| Rate_2_data-Rate_2_model | stan_nuts_diag | 2/2 | 0.7315 | 0.058 | 0.0885 | 1.003 | 3045 | 0.01714 |
| Rate_2_data-Rate_2_model | stan_nuts_dense | 2/2 | 0.7333 | 0.05 | 0.0945 | 1.003 | 3040 | 0.01973 |
| Rate_2_data-Rate_2_model | blackjax_nuts | 2/2 | 10.91 | 0.1006 | 0.1571 | 1.002 | 2967 | 0.0234 |
| garch-garch11 | nuts_cov_diag | 2/2 | 0.8708 | 0.5358 | 0.9939 | 1.001 | 1932 | 0.04991 |
| garch-garch11 | nuts_cov_dense | 2/2 | 0.7028 | 0.4068 | 0.5363 | 1.002 | 1972 | 0.02037 |
| garch-garch11 | nuts_rs | 2/2 | 0.5537 | 0.3678 | 0.5465 | 1.005 | 1406 | 0.03849 |
| garch-garch11 | stan_nuts_diag | 2/2 | 1.66 | 1.274 | 2.003 | 1.003 | 1704 | 0.04542 |
| garch-garch11 | stan_nuts_dense | 2/2 | 1.343 | 0.8425 | 1.286 | 1.003 | 2100 | 0.02293 |
| garch-garch11 | blackjax_nuts | 2/2 | 14.4 | 0.8981 | 1.493 | 1.004 | 1974 | 0.03662 |
| earnings-earn_height | nuts_cov_diag | 2/2 | 2.575 | 3.246 | 5.165 | 1.003 | 989.8 | 0.06697 |
| earnings-earn_height | nuts_cov_dense | 2/2 | 0.931 | 1.685 | 0.3512 | 1.003 | 3608 | 0.02686 |
| earnings-earn_height | nuts_rs | 2/2 | 0.9596 | 0.8553 | 1.869 | 1.01 | 475.4 | 0.04717 |
| earnings-earn_height | stan_nuts_diag | 2/2 | 5.021 | 6.34 | 10.63 | 1.005 | 961.3 | 0.02564 |
| earnings-earn_height | stan_nuts_dense | 2/2 | 1.936 | 3.667 | 1.148 | 1.002 | 3026 | 0.01672 |
| earnings-earn_height | blackjax_nuts | 2/2 | 13.21 | 2.755 | 4.583 | 1.003 | 1031 | 0.03498 |
| eight_schools-eight_schools_noncentered | nuts_cov_diag | 2/2 | 0.8153 | 0.03949 | 0.06166 | 1.003 | 1951 | 0.03226 |
| eight_schools-eight_schools_noncentered | nuts_cov_dense | 2/2 | 0.8364 | 0.05975 | 0.0781 | 1.003 | 1962 | 0.02743 |
| eight_schools-eight_schools_noncentered | nuts_rs | 2/2 | 0.7 | 0.02263 | 0.0377 | 1.004 | 2027 | 0.02528 |
| eight_schools-eight_schools_noncentered | stan_nuts_diag | 2/2 | 1.569 | 0.123 | 0.225 | 1.005 | 1868 | 0.02527 |
| eight_schools-eight_schools_noncentered | stan_nuts_dense | 2/2 | 1.594 | 0.1545 | 0.2555 | 1.004 | 2401 | 0.02385 |
| eight_schools-eight_schools_noncentered | blackjax_nuts | 2/2 | 11.67 | 0.241 | 0.3758 | 1.004 | 2382 | 0.02813 |
| normal_2-normal_mixture | nuts_cov_diag | 2/2 | 0.8947 | 0.6494 | 1.162 | 1.529 | 7.133 | — |
| normal_2-normal_mixture | nuts_cov_dense | 2/2 | 1.023 | 0.7492 | 1.414 | 1.529 | 7.189 | — |
| normal_2-normal_mixture | nuts_rs | 2/2 | 0.5697 | 0.4845 | 0.7555 | 1.529 | 7.14 | — |
| normal_2-normal_mixture | stan_nuts_diag | 2/2 | 3.397 | 3.588 | 6.955 | 1.53 | 7.17 | — |
| normal_2-normal_mixture | stan_nuts_dense | 2/2 | 3.325 | 3.651 | 6.573 | 1.531 | 7.154 | — |
| normal_2-normal_mixture | blackjax_nuts | 2/2 | 12.14 | 0.5037 | 0.8889 | 1.53 | 7.157 | — |
| sblri-blr | nuts_cov_diag | 2/2 | 0.5978 | 0.1019 | 0.1409 | 1.014 | 661.3 | 0.07152 |
| sblri-blr | nuts_cov_dense | 2/2 | 0.5931 | 0.1061 | 0.1462 | 1.006 | 650.6 | 0.04105 |
| sblri-blr | nuts_rs | 2/2 | 0.4275 | 0.0413 | 0.06775 | 1.002 | 3993 | 0.03647 |
| sblri-blr | stan_nuts_diag | 2/2 | 1.074 | 0.1635 | 0.264 | 1.014 | 585.9 | 0.0609 |
| sblri-blr | stan_nuts_dense | 2/2 | 1.075 | 0.173 | 0.304 | 1.005 | 672.3 | 0.02651 |
| sblri-blr | blackjax_nuts | 2/2 | 12 | 0.2844 | 0.4681 | 1.007 | 593.5 | 0.07755 |

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
