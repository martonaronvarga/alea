# Descriptive PosteriorDB benchmark

6 posteriors, 9 method configurations, 162 planned jobs. Status counts: `{'completed': 162}`.

Every failure and unsupported combination is retained. Diagnostics and accuracy are observations; this report makes no promotion, pass/fail or default-selection decision.

Cold wall time includes process startup, adaptation, sampling and serialization. Retained timings have engine-specific scopes; JAX host callbacks and native-JAX translations are labelled separately. RMHMC is exactness-pending; MCLMC is approximate; ATLAS/Pathfinder rows are labelled composed workflows.

| Posterior | Method | Completed/planned | Process s | Warmup s | Retained s | Worst R-hat | Min bulk ESS | Max mean error / ref SD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| earnings-earn_height | nuts_cov_diag | 3/3 | 19.89 | 6.692 | 13.12 | 1.003 | 1805 | 0.01648 |
| earnings-earn_height | nuts_cov_dense | 3/3 | 3.656 | 2.613 | 0.9852 | 1.001 | 7546 | 0.01486 |
| earnings-earn_height | nuts_fisher_diag | 3/3 | 10.23 | 3.283 | 6.881 | 1.005 | 1147 | 0.02587 |
| earnings-earn_height | nuts_fisher_rank4 | 3/3 | 1.401 | 0.6011 | 0.7349 | 1.001 | 1.091e+04 | 0.009481 |
| earnings-earn_height | atlas_full | 3/3 | 29.17 | 8.617 | 19.85 | 1.036 | 291 | 0.04873 |
| earnings-earn_height | stan_nuts_diag | 3/3 | 17.75 | 5.629 | 11.31 | 1.003 | 2064 | 0.02056 |
| earnings-earn_height | stan_nuts_dense | 3/3 | 4.019 | 2.162 | 1.058 | 1.001 | 6943 | 0.02083 |
| earnings-earn_height | nuts_rs | 3/3 | 7.488 | 2.277 | 5.259 | 1.003 | 871.5 | 0.04441 |
| earnings-earn_height | blackjax_nuts | 3/3 | 258.1 | 86.84 | 158.4 | 1.004 | 1804 | 0.02405 |
| eight_schools-eight_schools_centered | nuts_cov_diag | 3/3 | 0.393 | 0.1287 | 0.1697 | 1.169 | 16.06 | — |
| eight_schools-eight_schools_centered | nuts_cov_dense | 3/3 | 1.358 | 0.3397 | 0.7974 | 1.035 | 125.9 | — |
| eight_schools-eight_schools_centered | nuts_fisher_diag | 3/3 | 0.4001 | 0.09306 | 0.1992 | 1.109 | 25.24 | — |
| eight_schools-eight_schools_centered | nuts_fisher_rank4 | 3/3 | 0.378 | 0.1105 | 0.1564 | 1.036 | 111.4 | — |
| eight_schools-eight_schools_centered | atlas_full | 3/3 | 0.6545 | 0.1512 | 0.4724 | 1.078 | 35.33 | — |
| eight_schools-eight_schools_centered | stan_nuts_diag | 3/3 | 1.926 | 0.158 | 0.277 | 1.028 | 195.7 | — |
| eight_schools-eight_schools_centered | stan_nuts_dense | 3/3 | 1.948 | 0.191 | 0.286 | 1.042 | 94.79 | — |
| eight_schools-eight_schools_centered | nuts_rs | 3/3 | 0.199 | 0.05092 | 0.09434 | 1.017 | 151.6 | — |
| eight_schools-eight_schools_centered | blackjax_nuts | 3/3 | 7.673 | 0.4624 | 0.6982 | 1.036 | 134 | — |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | nuts_cov_diag | 3/3 | 89.35 | 26.02 | 63.24 | 1.092 | 35.44 | — |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | nuts_cov_dense | 3/3 | 71.52 | 19.29 | 52.11 | 1.486 | 7.553 | — |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | nuts_fisher_diag | 3/3 | 52.83 | 17.36 | 35.38 | 1.224 | 13.05 | — |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | nuts_fisher_rank4 | 3/3 | 27.66 | 11.53 | 16.88 | 1.531 | 7.169 | — |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | atlas_full | 3/3 | 136.8 | 31.94 | 104.8 | 1.36 | 9.364 | — |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | stan_nuts_diag | 3/3 | 68.77 | 19.6 | 47.39 | 1.126 | 26.27 | — |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | stan_nuts_dense | 3/3 | 51.12 | 15.15 | 34.48 | 1.587 | 6.766 | — |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | nuts_rs | 3/3 | 35.72 | 13.14 | 23.06 | 1.158 | 19.43 | — |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | blackjax_nuts | 3/3 | 185 | 53.18 | 125.9 | 1.129 | 22.06 | — |
| normal_2-normal_mixture | nuts_cov_diag | 3/3 | 10.22 | 3.159 | 6.993 | 1.529 | 7.112 | — |
| normal_2-normal_mixture | nuts_cov_dense | 3/3 | 9.604 | 3.232 | 6.297 | 1.529 | 7.115 | — |
| normal_2-normal_mixture | nuts_fisher_diag | 3/3 | 7.782 | 2.835 | 4.862 | 1.529 | 7.099 | — |
| normal_2-normal_mixture | nuts_fisher_rank4 | 3/3 | 7.753 | 2.926 | 4.725 | 1.529 | 7.099 | — |
| normal_2-normal_mixture | atlas_full | 3/3 | 56.73 | 8.366 | 48.33 | 1.531 | 7.116 | — |
| normal_2-normal_mixture | stan_nuts_diag | 3/3 | 10.59 | 3.254 | 6.484 | 1.529 | 7.106 | — |
| normal_2-normal_mixture | stan_nuts_dense | 3/3 | 11.42 | 3.422 | 7.173 | 1.53 | 7.12 | — |
| normal_2-normal_mixture | nuts_rs | 3/3 | 7.006 | 2.532 | 4.504 | 1.529 | 7.097 | — |
| normal_2-normal_mixture | blackjax_nuts | 3/3 | 32.25 | 9.51 | 17.99 | 1.529 | 7.101 | — |
| sblri-blr | nuts_cov_diag | 3/3 | 0.6189 | 0.1832 | 0.2899 | 1.003 | 1444 | 0.03522 |
| sblri-blr | nuts_cov_dense | 3/3 | 0.8368 | 0.2629 | 0.4248 | 1.004 | 1512 | 0.0447 |
| sblri-blr | nuts_fisher_diag | 3/3 | 0.331 | 0.1178 | 0.1263 | 1.002 | 8020 | 0.02653 |
| sblri-blr | nuts_fisher_rank4 | 3/3 | 0.3763 | 0.1535 | 0.1192 | 1.001 | 7013 | 0.02986 |
| sblri-blr | atlas_full | 3/3 | 1.716 | 0.2554 | 1.427 | 1.005 | 1701 | 0.03255 |
| sblri-blr | stan_nuts_diag | 3/3 | 1.584 | 0.173 | 0.255 | 1.003 | 1465 | 0.02461 |
| sblri-blr | stan_nuts_dense | 3/3 | 1.442 | 0.139 | 0.227 | 1.006 | 1545 | 0.0328 |
| sblri-blr | nuts_rs | 3/3 | 0.1721 | 0.05114 | 0.0863 | 1.002 | 8511 | 0.01728 |
| sblri-blr | blackjax_nuts | 3/3 | 51.06 | 15.78 | 26.52 | 1.004 | 1549 | 0.02197 |
| seeds_data-seeds_stanified_model | nuts_cov_diag | 3/3 | 1.698 | 0.4975 | 0.9725 | 1.044 | 168.4 | — |
| seeds_data-seeds_stanified_model | nuts_cov_dense | 3/3 | 3.014 | 1.358 | 1.494 | 1.027 | 142.6 | — |
| seeds_data-seeds_stanified_model | nuts_fisher_diag | 3/3 | 1.079 | 0.3107 | 0.5935 | 1.019 | 189.4 | — |
| seeds_data-seeds_stanified_model | nuts_fisher_rank4 | 3/3 | 1.29 | 0.471 | 0.6176 | 1.01 | 518 | — |
| seeds_data-seeds_stanified_model | atlas_full | 3/3 | 2.008 | 0.4651 | 1.432 | 1.016 | 154.7 | — |
| seeds_data-seeds_stanified_model | stan_nuts_diag | 3/3 | 5.203 | 0.401 | 0.879 | 1.055 | 85.3 | — |
| seeds_data-seeds_stanified_model | stan_nuts_dense | 3/3 | 5.55 | 0.637 | 0.995 | 1.009 | 633.3 | — |
| seeds_data-seeds_stanified_model | nuts_rs | 3/3 | 1.253 | 0.3755 | 0.7138 | 1.013 | 387.9 | — |
| seeds_data-seeds_stanified_model | blackjax_nuts | 3/3 | 151.3 | 50.52 | 94.1 | 1.006 | 473.7 | — |

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
