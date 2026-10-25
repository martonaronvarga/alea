# Descriptive PosteriorDB benchmark

2 posteriors, 28 method configurations, 56 planned jobs. Status counts: `{'completed': 51, 'failed': 5}`.

Every failure and unsupported combination is retained. Diagnostics and accuracy are observations; this report makes no promotion, pass/fail or default-selection decision.

Cold wall time includes process startup, adaptation, sampling and serialization. Retained timings have engine-specific scopes; JAX host callbacks and native-JAX translations are labelled separately. RMHMC is exactness-pending; MCLMC is approximate; ATLAS/Pathfinder rows are labelled composed workflows.

| Posterior | Method | Completed/planned | Process s | Warmup s | Retained s | Worst R-hat | Min bulk ESS | Max mean error / ref SD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| earnings-earn_height | nuts_cov_diag | 1/1 | 2.21 | 1.523 | 0.6739 | 3.083 | 5.127 | 6.248 |
| earnings-earn_height | nuts_cov_dense | 1/1 | 2.464 | 1.533 | 0.9182 | 1.792 | 6.683 | 6.344 |
| earnings-earn_height | nuts_fisher_diag | 1/1 | 0.4439 | 0.2322 | 0.1991 | 1.146 | 43.64 | 0.1323 |
| earnings-earn_height | nuts_fisher_rank4 | 1/1 | 0.3312 | 0.139 | 0.1795 | 1.04 | 102.5 | 0.03692 |
| earnings-earn_height | nuts_fisher_rm | 1/1 | 1.155 | 0.7077 | 0.434 | 4.642 | 4.789 | 48.96 |
| earnings-earn_height | nuts_fisher_adam | 1/1 | 1.031 | 0.6679 | 0.3505 | 4.643 | 4.789 | 48.96 |
| earnings-earn_height | nuts_transport | 1/1 | 1.995 | 1.115 | 0.8735 | 3.119 | 5.116 | 6.359 |
| earnings-earn_height | hmc_cov | 1/1 | 1.026 | 0.9661 | 0.05318 | 3.168 | 5.086 | 6.375 |
| earnings-earn_height | hmc_omf | 1/1 | 0.2785 | 0.1657 | 0.1065 | 3.674 | 4.939 | 10.11 |
| earnings-earn_height | hmc_bcss | 1/1 | 1.368 | 1.254 | 0.1073 | 2.69 | 5.332 | 6.349 |
| earnings-earn_height | ghmc | 1/1 | 1.028 | 0.966 | 0.05587 | 3.887 | 4.892 | 6.372 |
| earnings-earn_height | drghmc | 1/1 | 1.025 | 0.9584 | 0.05991 | 4.159 | 4.848 | 6.379 |
| earnings-earn_height | mams | 1/1 | 1.012 | 0.9511 | 0.05471 | 4.145 | 4.861 | 6.373 |
| earnings-earn_height | mclmc | 1/1 | 1.02 | 0.9581 | 0.05515 | 3.773 | 4.927 | 6.375 |
| earnings-earn_height | chees | 1/1 | 1.117 | 0.9017 | 0.2088 | 3.162 | 5.086 | 6.337 |
| earnings-earn_height | walnuts_d | 1/1 | 1.654 | 1.518 | 0.1302 | 3.2 | 5.07 | 6.262 |
| earnings-earn_height | walnuts_r2p | 1/1 | 1.662 | 1.512 | 0.144 | 3.173 | 5.123 | 6.269 |
| earnings-earn_height | rmhmc_softabs | 0/1 | — | — | — | — | — | — |
| earnings-earn_height | rmhmc_manifold | 0/1 | — | — | — | — | — | — |
| earnings-earn_height | stan_nuts_diag | 1/1 | 2.537 | 1.225 | 1.193 | 2.599 | 5.391 | 6.365 |
| earnings-earn_height | stan_nuts_dense | 1/1 | 2.997 | 1.45 | 1.411 | 2.599 | 5.391 | 6.365 |
| earnings-earn_height | stan_pathfinder_nuts | 0/1 | — | — | — | — | — | — |
| earnings-earn_height | blackjax_nuts | 1/1 | 44.62 | 25.02 | 13.49 | 2.394 | 5.641 | 6.346 |
| earnings-earn_height | nuts_rs | 1/1 | 2.915 | 0.9996 | 1.902 | 2.646 | 5.341 | 4.707 |
| earnings-earn_height | nuts_saia_joint | 1/1 | 0.308 | 0.1703 | 0.1274 | 3.924 | 4.903 | 48.78 |
| earnings-earn_height | nuts_pathfinder_raw | 1/1 | 4.868 | 3.705 | 1.146 | 2.191 | 5.799 | 6.227 |
| earnings-earn_height | nuts_pathfinder_psis | 1/1 | 3.27 | 2.228 | 1.022 | 2.219 | 5.766 | 6.235 |
| earnings-earn_height | atlas_full | 1/1 | 4.456 | 2.184 | 2.251 | 3.096 | 5.112 | 6.166 |
| sblri-blr | nuts_cov_diag | 1/1 | 0.377 | 0.03716 | 0.005622 | 1.009 | 256.8 | 0.09509 |
| sblri-blr | nuts_cov_dense | 1/1 | 2.32 | 0.1146 | 0.03495 | 1.015 | 282.5 | 0.08472 |
| sblri-blr | nuts_fisher_diag | 1/1 | 0.2514 | 0.06257 | 0.009711 | 1.025 | 318.9 | 0.0774 |
| sblri-blr | nuts_fisher_rank4 | 1/1 | 0.06339 | 0.03703 | 0.008864 | 1.063 | 238.4 | 0.1435 |
| sblri-blr | nuts_fisher_rm | 1/1 | 2.336 | 0.4468 | 0.1585 | 4.643 | 4.789 | 799.7 |
| sblri-blr | nuts_fisher_adam | 1/1 | 0.2917 | 0.2032 | 0.06545 | 4.616 | 4.792 | 794.9 |
| sblri-blr | nuts_transport | 1/1 | 0.1107 | 0.06088 | 0.03646 | 1.029 | 147.9 | 0.07809 |
| sblri-blr | hmc_cov | 1/1 | 0.02573 | 0.01431 | 0.004807 | 1.093 | 33.74 | 0.1145 |
| sblri-blr | hmc_omf | 1/1 | 0.06601 | 0.03889 | 0.01845 | 1.621 | 7.519 | 1.301 |
| sblri-blr | hmc_bcss | 1/1 | 0.05373 | 0.03087 | 0.01542 | 1.583 | 7.838 | 0.385 |
| sblri-blr | ghmc | 1/1 | 0.02587 | 0.01426 | 0.004895 | 1.068 | 293.8 | 0.1014 |
| sblri-blr | drghmc | 1/1 | 0.04353 | 0.0278 | 0.008308 | 1.056 | 154.3 | 0.04885 |
| sblri-blr | mams | 1/1 | 0.04674 | 0.02771 | 0.008324 | 1.04 | 67.43 | 0.1381 |
| sblri-blr | mclmc | 1/1 | 0.03521 | 0.0193 | 0.007766 | 1.064 | 280.8 | 0.06198 |
| sblri-blr | chees | 1/1 | 0.1846 | 0.1121 | 0.06597 | 1.049 | 206.2 | 0.09601 |
| sblri-blr | walnuts_d | 1/1 | 0.1702 | 0.01521 | 0.01306 | 1.031 | 145.3 | 0.09758 |
| sblri-blr | walnuts_r2p | 1/1 | 0.03821 | 0.01528 | 0.01664 | 1.014 | 156.8 | 0.1986 |
| sblri-blr | rmhmc_softabs | 0/1 | — | — | — | — | — | — |
| sblri-blr | rmhmc_manifold | 0/1 | — | — | — | — | — | — |
| sblri-blr | stan_nuts_diag | 1/1 | 0.2091 | 0.031 | 0.021 | 1.204 | 16.44 | 0.793 |
| sblri-blr | stan_nuts_dense | 1/1 | 0.1942 | 0.034 | 0.021 | 1.204 | 16.44 | 0.793 |
| sblri-blr | stan_pathfinder_nuts | 1/1 | 0.2147 | 0.042 | 0.016 | 1.256 | 16.06 | 0.5333 |
| sblri-blr | blackjax_nuts | 1/1 | 10.57 | 3.031 | 0.6283 | 1.014 | 297.6 | 0.1052 |
| sblri-blr | nuts_rs | 1/1 | 0.02456 | 0.01202 | 0.005998 | 1.03 | 296.6 | 0.1178 |
| sblri-blr | nuts_saia_joint | 1/1 | 0.222 | 0.03735 | 0.07698 | 3.668 | 5.014 | 500 |
| sblri-blr | nuts_pathfinder_raw | 1/1 | 3.088 | 0.101 | 0.007632 | 1.03 | 64.92 | 0.1391 |
| sblri-blr | nuts_pathfinder_psis | 1/1 | 0.1283 | 0.06457 | 0.007461 | 1.076 | 39.08 | 0.07539 |
| sblri-blr | atlas_full | 1/1 | 0.2164 | 0.07632 | 0.06797 | 1.121 | 62.27 | 0.2421 |

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
- MAMS tunes the actual corrected acceptance to0.8 after frozen diagonal affine preconditioning. Approximate MCLMC tunes per-dimension squared correction error to5e-4 using measured h^6 local scaling at a fixed integration count, with refresh radius from standardized pilot scales. No exact-target claim applies to MCLMC.
- RMHMC remains exactness-pending. Per-model Stan grad_hessian supplies AD third derivatives, with no finite-difference fallback. Newton has at most100 iterations, residual1e-12 and full phase/half-momentum reverse replay at1e-8; all reverse work is charged. These checks do not close finite-solver volume preservation. Dimension<=10,32 warmup and64 draws per chain remain explicitly unmatched budgets.
- The same20 model/data/reference byte sets are retained; the v3 corpus corrects the legacy sblri-blr family tag to Gaussian linear regression.
- The frozen affine adapter preserves raw support zero so numerical/out-of-support proposals reject normally. Other backend and finite-point score errors remain typed. Robust Pathfinder scales its first L-BFGS trial before curvature exists; the four-path/forty-iteration/twenty-backtrack budget remains unchanged.
