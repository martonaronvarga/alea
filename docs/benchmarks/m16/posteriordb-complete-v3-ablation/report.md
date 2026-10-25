# Descriptive PosteriorDB benchmark

3 posteriors, 2 method configurations, 18 planned jobs. Status counts: `{'completed': 18}`.

Every failure and unsupported combination is retained. Diagnostics and accuracy are observations; this report makes no promotion, pass/fail or default-selection decision.

Cold wall time includes process startup, adaptation, sampling and serialization. Retained timings have engine-specific scopes; JAX host callbacks and native-JAX translations are labelled separately. RMHMC is exactness-pending; MCLMC is approximate; ATLAS/Pathfinder rows are labelled composed workflows.

| Posterior | Method | Completed/planned | Process s | Warmup s | Retained s | Worst R-hat | Min bulk ESS | Max mean error / ref SD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| arK-arK | nuts_legacy_warmup | 3/3 | 3.876 | 0.6884 | 3.248 | 1.007 | 989.2 | 0.04259 |
| arK-arK | nuts_cov_diag | 3/3 | 5.62 | 2.936 | 2.652 | 1.005 | 1114 | 0.04387 |
| earnings-earn_height | nuts_legacy_warmup | 3/3 | 1.049 | 0.5404 | 0.4974 | 4.567 | 4.272 | 48.97 |
| earnings-earn_height | nuts_cov_diag | 3/3 | 8.462 | 4.584 | 3.84 | 1.007 | 534.5 | 0.04146 |
| sblri-blr | nuts_legacy_warmup | 3/3 | 0.119 | 0.04034 | 0.06464 | 1.019 | 240.9 | 16.26 |
| sblri-blr | nuts_cov_diag | 3/3 | 0.2001 | 0.07065 | 0.0533 | 1.017 | 331.1 | 0.1215 |

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
- Controlled whole-warmup comparison at identical original starts and seeds: historical eight-step HMC covariance warmup versus native NUTS warmup. Retained NUTS law/budgets/target backend unchanged. This changes the whole warmup policy, not only its energy cutoff. Separate uncensored one-step probes isolate the cutoff at identical q,p,h.
