# Descriptive PosteriorDB benchmark

2 posteriors, 28 method configurations, 56 planned jobs. Status counts: `{'completed': 51, 'failed': 5}`.

Every failure and unsupported combination is retained. Diagnostics and accuracy are observations; this report makes no promotion, pass/fail or default-selection decision.

Cold wall time includes process startup, adaptation, sampling and serialization. Retained timings have engine-specific scopes; JAX host callbacks and native-JAX translations are labelled separately. RMHMC is exactness-pending; MCLMC is approximate; ATLAS/Pathfinder rows are labelled composed workflows.

| Posterior | Method | Completed/planned | Process s | Warmup s | Retained s | Worst R-hat | Min bulk ESS | Max mean error / ref SD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| earnings-earn_height | nuts_cov_diag | 1/1 | 2.233 | 1.536 | 0.6841 | 3.083 | 5.127 | 6.248 |
| earnings-earn_height | nuts_cov_dense | 1/1 | 2.48 | 1.542 | 0.9255 | 1.792 | 6.683 | 6.344 |
| earnings-earn_height | nuts_fisher_diag | 1/1 | 0.436 | 0.2258 | 0.1973 | 1.146 | 43.64 | 0.1323 |
| earnings-earn_height | nuts_fisher_rank4 | 1/1 | 0.4392 | 0.1735 | 0.1821 | 1.04 | 102.5 | 0.03692 |
| earnings-earn_height | nuts_fisher_rm | 1/1 | 1.206 | 0.7164 | 0.4277 | 4.642 | 4.789 | 48.96 |
| earnings-earn_height | nuts_fisher_adam | 1/1 | 1.092 | 0.7104 | 0.368 | 4.643 | 4.789 | 48.96 |
| earnings-earn_height | nuts_transport | 1/1 | 1.93 | 1.066 | 0.8574 | 3.119 | 5.116 | 6.359 |
| earnings-earn_height | hmc_cov | 1/1 | 1.047 | 0.983 | 0.05708 | 3.168 | 5.086 | 6.375 |
| earnings-earn_height | hmc_omf | 1/1 | 0.2834 | 0.1702 | 0.1066 | 3.674 | 4.939 | 10.11 |
| earnings-earn_height | hmc_bcss | 1/1 | 1.363 | 1.243 | 0.114 | 2.69 | 5.332 | 6.349 |
| earnings-earn_height | ghmc | 1/1 | 1.014 | 0.9538 | 0.05362 | 3.887 | 4.892 | 6.372 |
| earnings-earn_height | drghmc | 1/1 | 1.095 | 1.025 | 0.06278 | 4.159 | 4.848 | 6.379 |
| earnings-earn_height | mams | 1/1 | 1.137 | 1.012 | 0.1189 | 2.888 | 5.203 | 6.372 |
| earnings-earn_height | mclmc | 1/1 | 1.041 | 0.9778 | 0.05658 | 3.413 | 5.026 | 6.374 |
| earnings-earn_height | chees | 1/1 | 1.113 | 0.9067 | 0.1993 | 3.162 | 5.086 | 6.337 |
| earnings-earn_height | walnuts_d | 1/1 | 3.017 | 2.769 | 0.2422 | 3.2 | 5.07 | 6.262 |
| earnings-earn_height | walnuts_r2p | 1/1 | 1.787 | 1.624 | 0.1565 | 3.173 | 5.123 | 6.269 |
| earnings-earn_height | rmhmc_softabs | 0/1 | — | — | — | — | — | — |
| earnings-earn_height | rmhmc_manifold | 0/1 | — | — | — | — | — | — |
| earnings-earn_height | stan_nuts_diag | 1/1 | 2.603 | 1.265 | 1.22 | 2.599 | 5.391 | 6.365 |
| earnings-earn_height | stan_nuts_dense | 1/1 | 2.555 | 1.241 | 1.195 | 2.599 | 5.391 | 6.365 |
| earnings-earn_height | stan_pathfinder_nuts | 0/1 | — | — | — | — | — | — |
| earnings-earn_height | blackjax_nuts | 1/1 | 47.01 | 26.53 | 14.09 | 2.394 | 5.641 | 6.346 |
| earnings-earn_height | nuts_rs | 1/1 | 3.049 | 0.988 | 2.043 | 2.646 | 5.341 | 4.707 |
| earnings-earn_height | nuts_saia_joint | 1/1 | 0.2823 | 0.1454 | 0.1298 | 3.924 | 4.903 | 48.78 |
| earnings-earn_height | nuts_pathfinder_raw | 1/1 | 3.642 | 2.554 | 0.819 | 2.191 | 5.799 | 6.227 |
| earnings-earn_height | nuts_pathfinder_psis | 1/1 | 3.749 | 2.45 | 1.003 | 2.219 | 5.766 | 6.235 |
| earnings-earn_height | atlas_full | 1/1 | 3.91 | 1.842 | 2.058 | 3.096 | 5.112 | 6.166 |
| sblri-blr | nuts_cov_diag | 1/1 | 0.1098 | 0.02099 | 0.004439 | 1.009 | 256.8 | 0.09509 |
| sblri-blr | nuts_cov_dense | 1/1 | 2.57 | 0.06108 | 0.01736 | 1.015 | 282.5 | 0.08472 |
| sblri-blr | nuts_fisher_diag | 1/1 | 0.2218 | 0.03569 | 0.007065 | 1.025 | 318.9 | 0.0774 |
| sblri-blr | nuts_fisher_rank4 | 1/1 | 0.4495 | 0.06567 | 0.009284 | 1.063 | 238.4 | 0.1435 |
| sblri-blr | nuts_fisher_rm | 1/1 | 0.5258 | 0.2187 | 0.1067 | 4.643 | 4.789 | 799.7 |
| sblri-blr | nuts_fisher_adam | 1/1 | 0.5318 | 0.2218 | 0.07572 | 4.616 | 4.792 | 794.9 |
| sblri-blr | nuts_transport | 1/1 | 0.1063 | 0.05893 | 0.03315 | 1.029 | 147.9 | 0.07809 |
| sblri-blr | hmc_cov | 1/1 | 0.05789 | 0.03303 | 0.009755 | 1.093 | 33.74 | 0.1145 |
| sblri-blr | hmc_omf | 1/1 | 0.07181 | 0.04102 | 0.01895 | 1.621 | 7.519 | 1.301 |
| sblri-blr | hmc_bcss | 1/1 | 0.07709 | 0.04606 | 0.01901 | 1.583 | 7.838 | 0.385 |
| sblri-blr | ghmc | 1/1 | 0.03975 | 0.02215 | 0.007638 | 1.068 | 293.8 | 0.1014 |
| sblri-blr | drghmc | 1/1 | 0.06137 | 0.01937 | 0.006135 | 1.056 | 154.3 | 0.04885 |
| sblri-blr | mams | 1/1 | 0.09779 | 0.05201 | 0.03646 | 1.067 | 48.31 | 0.1349 |
| sblri-blr | mclmc | 1/1 | 0.0589 | 0.0308 | 0.01366 | 1.046 | 248.2 | 0.06768 |
| sblri-blr | chees | 1/1 | 0.309 | 0.1726 | 0.1203 | 1.049 | 206.2 | 0.09601 |
| sblri-blr | walnuts_d | 1/1 | 0.06078 | 0.02797 | 0.02098 | 1.031 | 145.3 | 0.09758 |
| sblri-blr | walnuts_r2p | 1/1 | 0.07591 | 0.03121 | 0.03209 | 1.014 | 156.8 | 0.1986 |
| sblri-blr | rmhmc_softabs | 0/1 | — | — | — | — | — | — |
| sblri-blr | rmhmc_manifold | 0/1 | — | — | — | — | — | — |
| sblri-blr | stan_nuts_diag | 1/1 | 0.2411 | 0.036 | 0.025 | 1.204 | 16.44 | 0.793 |
| sblri-blr | stan_nuts_dense | 1/1 | 0.1856 | 0.028 | 0.019 | 1.204 | 16.44 | 0.793 |
| sblri-blr | stan_pathfinder_nuts | 1/1 | 0.3951 | 0.076 | 0.03 | 1.256 | 16.06 | 0.5333 |
| sblri-blr | blackjax_nuts | 1/1 | 12.44 | 3.679 | 0.6315 | 1.014 | 297.6 | 0.1052 |
| sblri-blr | nuts_rs | 1/1 | 0.01886 | 0.009121 | 0.003671 | 1.03 | 296.6 | 0.1178 |
| sblri-blr | nuts_saia_joint | 1/1 | 0.1304 | 0.03286 | 0.088 | 3.668 | 5.014 | 500 |
| sblri-blr | nuts_pathfinder_raw | 1/1 | 1.126 | 0.1 | 0.01305 | 1.03 | 64.92 | 0.1391 |
| sblri-blr | nuts_pathfinder_psis | 1/1 | 0.1348 | 0.05501 | 0.01125 | 1.076 | 39.08 | 0.07539 |
| sblri-blr | atlas_full | 1/1 | 0.4186 | 0.08401 | 0.08272 | 1.121 | 62.27 | 0.2421 |

## Accounting and scope

- Entirely descriptive: no promotion, pass/fail, speedup eligibility, default selection or maturity graduation is computed.
- Twenty declared PosteriorDB model/data pairs at 91027f9cdd13b7e76283fec3f2898f59d4b83ec4; original models/data/licenses and available reference draws are vendored and hashed. This is a diverse subset, not all PosteriorDB.
- All numerical jobs are serial on one allowed CPU; float64, three independent four-chain seed sets. Cold process wall time uses perf_counter_ns and kernel pidfd completion without 50-ms timeout polling.
- Build/compiler costs are recorded outside measurements. Cold process time includes loading, adaptation/initialization, sampling, generated outputs and serialization. Native worker retained timing excludes serialization and constraining outputs; Stan retained timing includes its native CSV and generated quantities.
- Rust Alea and nuts-rs use a shared compiled BridgeStan 2.7.0 / Stan 2.38.0 target backend with unconstrained Lebesgue measure, propto=true and jacobian=true. These runs compare implemented sampling workflows using Stan AD targets, rather than native Alea likelihood speed.
- BlackJAX 1.5 / JAX 0.9.2 uses independently checked native-JAX densities for Rate 1/2, centered/noncentered eight schools, wells dist100 and kidiq momhs. Other models use explicit BridgeStan custom-VJP host callbacks; callback and host-transfer overhead must not be presented as native-JAX kernel speed.
- Fisher variants cover diagonal, rank-four, Robbins-Monro and Adam step adaptation. Low-rank fitting may reject unsupported conditioning; no fallback or hidden tuning is used.
- Every failure, timeout, partial output and unsupported method/model combination remains in the planned denominator. Missing or undefined ESS/R-hat is reported, not silently filtered. Efficiency warnings are observations, not decision gates.
- Reference comparisons report mean discrepancy in reference posterior standard-deviation units and posterior spread ratios, with current-chain MCSE separately. Reference draws are empirical, not exact truth; no universal calibration or significance claim is made.
- Actual target/Hessian/HVP API counts and exposed factorization/solve costs are recorded. Internal Stan AD operation counts and upstream allocation counts are unavailable; no work count is inferred from integration steps.
- One physical host; timing repetitions do not establish cross-host performance or universal sampler superiority.
- Gradient and signed-direction HVP finite-difference preflight occurs before measurements. All six native-JAX translations are checked against Stan scores and density differences before measurements. A failed HVP capability excludes only the HVP-dependent ATLAS workflow; full-Hessian RMHMC failures remain observed attempts. Cold phase records start inside each worker; interpreter/dynamic-loader costs remain in fresh-process wall time. Native retained timers include sampler conversion and owned position collection; affine transport also includes mapping retained positions back to target coordinates.
- Warmup and retained BlackJAX sampling use separate subtrees from an initial split key. Native Stan Pathfinder initialization uses a SHA-256-tagged seed distinct from the chain sampling seed; both actual seeds appear in commands/logs. Alea Pathfinder jitters four starts by uniform [-.5,.5]; Stan uses the shared declared constrained initial file for its four paths.
- Version3 covariance and Fisher warmup use native NUTS trees, one-sided bounded searches and explicit acceptance adaptation, with live-state traces. Stationary kernels and historical version2 artifacts retain their declared semantics.
- Static HMC/OMF/BCSS/GHMC/DRGHMC and MAMS/MCLMC use75% native NUTS pilot plus25% kernel step tuning; WALNUTS uses a full NUTS pilot. DRGHMC adapts first-stage probability and counts reverse ghosts.
- ChEES uses half NUTS pilot and one shared frozen affine whitening from cohort slow-window states, then half shared ChEES adaptation, with at most1024 macrosteps per trajectory. Transport similarly learns only from NUTS slow-window live states before transformed NUTS adaptation.
- nuts_saia_joint is the curvature/s-AIA component in native NUTS warmup. The coefficient freezes before final fast step tuning and the oscillator interval is capped at3.8. The observed inflated curvature is a policy estimate, not a global spectral certificate.
- Pathfinder raw-mixture and generating-path PSIS are separate configurations. Numerical paths and support-zero draws are recorded; concrete backend errors propagate. PSIS weights and shape have an executed Stan2.38 fixture.
- atlas_full is the full preliminary/delayed/failure ATLAS construction: constant_trajectory2, probabilistic1, one delayed stage,1024 steps per leg, frozen baseline and trimmed empirical lengths. It uses deterministic14-call HVP curvature and Appendix E.2 selection densities. This deliberately differs from the pinned author Python delayed-ratio calculation; see method documentation.
- MAMS tunes the actual corrected acceptance to0.8 after frozen diagonal affine preconditioning. Approximate MCLMC tunes centered signed correction-error coefficient variance per dimension to5e-4, using verified h^6 scaling and measured pilot refresh scales. No exact-target claim applies to MCLMC.
- RMHMC remains exactness-pending. Per-model Stan grad_hessian supplies AD third derivatives, with no finite-difference fallback. Newton has at most100 iterations, residual1e-12 and full phase/half-momentum reverse replay at1e-8; all reverse work is charged. These checks do not close finite-solver volume preservation. Dimension<=10,32 warmup and64 draws per chain remain explicitly unmatched budgets.
- The same20 model/data/reference byte sets are retained; the v3 corpus corrects the legacy sblri-blr family tag to Gaussian linear regression.
- The frozen affine adapter exposes successful nonfinite numerical outputs to the outer checked evaluator; concrete backend and structural errors retain their types. Robust Pathfinder scales its first L-BFGS trial before curvature exists; the four-path/forty-iteration/twenty-backtrack budget remains unchanged.
- Completion revision: MAMS uses a pilot-estimated latent scale duration and adapts its macrostep count with the step, capped at1024; both freeze before retention.
- MCLMC adapts centered signed local-error coefficient variance, with explicitly counted warmup-only numerical recovery. Retained numerical errors remain failures. Refresh length is estimated from actual pilot latent variances.
- DRGHMC first/delayed/ghost and ATLAS forward/reverse/ghost integration work are reported separately. Pathfinder initialization time and calls are separate from warmup, with no double addition.
- Capability-specific bulk ESS per actual gradient/Hessian/HVP/third-AD API call is descriptive. Counts include failed proposals. Missing upstream counters remain null; derivative capabilities are not treated as equal-cost operations.
