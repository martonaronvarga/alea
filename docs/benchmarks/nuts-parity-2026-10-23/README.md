# Alea / nuts-rs performance repair — 2026-10-23

Most of the native nuts-rs gap is closed through an explicit gradient-diagonal/jitter profile. It uses 5% more retained gradients, takes 9% more retained walltime and 11% more process walltime, and delivers 3% higher retained bulk ESS/s but 6% lower bulk ESS/process-second. Full walltime parity remains open. The unchanged covariance default is still 1.69x slower in process time in this cohort.

The largest remaining weaknesses are the cheap 64D kernel (2.31x nuts-rs time), seeds (0.50x retained bulk ESS/s), and surgical (0.44x). Seeds and surgical now have much closer diagnostics, but remain slower. These results support a competitive option to build on, not universal sampler equivalence.

Twenty pinned PosteriorDB posteriors, four sequential chains on one dusk CPU, 512 warmup and 1024 retained draws per chain, depth 8, three repetitions. Each of four cohorts accounts for 180 jobs (720 in total). Both libraries use the same fused native Rust/Enzyme target. This isolates sampling/adaptation workflows from AD-backend differences. Detailed warmup tracing is explicitly disabled for this comparison; historical traced campaigns remain unchanged.

Ratios are Alea / nuts-rs. Time and call ratios below 1 favor Alea; ESS/s ratios above 1 favor Alea. Each entry is the median across 20 posterior-specific medians of the three repetition ratios. There is no quality filter: failures, divergences and poor mixing remain visible. Three repetitions on one host do not establish statistical equivalence or universal superiority.

| Cohort | Alea workflow | Process time | Retained time | Total gradients | Retained gradients | Bulk ESS/process-second | Bulk ESS/retained-second |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| initial | nuts_cov_diag | 1.542 | 1.644 | 1.559 | 1.589 | 0.784 | 0.7857 |
| initial | nuts_grad_diag | 1.078 | 1.092 | 1.071 | 1.058 | 0.987 | 1.04 |
| restart | nuts_cov_diag | 1.645 | 1.698 | 1.559 | 1.589 | 0.7024 | 0.7206 |
| restart | nuts_grad_diag | 1.233 | 1.3 | 1.218 | 1.269 | 0.8111 | 0.772 |
| fixed-final | nuts_cov_diag | 1.565 | 1.623 | 1.559 | 1.589 | 0.6954 | 0.7797 |
| fixed-final | nuts_grad_diag | 1.204 | 1.327 | 1.285 | 1.361 | 0.8205 | 0.7782 |
| final | nuts_cov_diag | 1.688 | 1.726 | 1.559 | 1.589 | 0.6917 | 0.7047 |
| final | nuts_grad_diag_jitter | 1.105 | 1.091 | 1.059 | 1.052 | 0.9374 | 1.03 |

`initial` uses paired diagonal geometry with a continuous dual-averaging history. `restart` restarts tuning at metric freeze and uses symmetric acceptance throughout the final phase. `fixed-final` restarts at freeze and targets ordinary mean acceptance during that fixed-geometry phase, following the reference covariance controller. `final` uses continuous dual averaging with late symmetric acceptance and explicit 10% trajectory step jitter, following nuts-rs more closely. These are whole-policy comparisons; the earlier cohorts are preserved, including their weaknesses.

## Diagnosis and implemented corrections

- The preceding sealed native campaign recorded 1.559x total and 1.589x retained gradient calls for covariance Alea versus nuts-rs. Covariance windows and repeated dual-averaging restarts differ from nuts-rs’s paired position/score geometry and late symmetric statistic. Merely accelerating autodiff cannot remove this extra sampling work.

- `NutsWarmup::with_gradient_diagonal()` explicitly supplies conventional paired position/score diagonal adaptation, bounded score initialization, foreground/background windows and warmup-only symmetric acceptance. All live repeated states enter the estimates. The default final phase restarts search/dual averaging and uses ordinary tree acceptance; an explicit switch retains continuous late symmetric adaptation. Independent trajectory step jitter is also explicit, with zero jitter the default. Covariance remains the default; Fisher research defaults and guarantees are unchanged.

- Turning checks formerly repeated scale discovery and normalized divisions. A conservative rounding/underflow certificate now admits ordinary dot-product signs only when they agree with the original predicate; uncertain cases retain ordered normalization. Endpoint scales are cached and sum/scale/dot work is fused. First kick/drift read immutable phase caches directly, avoiding three coordinate copies per leaf. No unsafe code or approximate reciprocal was added.

- Benchmark warmup tracing previously charged Alea for detailed JSON observations while nuts-rs had no equivalent trace. The new sealed flag removes that uneven instrumentation in all four cohorts. Final summaries and actual derivative counters remain. Process timing still includes initialization, collection, constrained output and serialization; retained timing includes owned position collection.

The measured fast profile is explicit:

```rust
let options = NutsOptions::new(0.1, 8)?.with_step_size_jitter(0.1)?;
let (chain, report) = NutsWarmup::new(target, position, options, WarmupOptions::new(512))?
    .with_gradient_diagonal()?
    .with_gradient_final_restart(false)
    .run(&mut rng)?;
```

The final-restart switch defaults to true for the gradient option. The benchmark’s false setting follows the nuts-rs continuous schedule. Search/adaptation bounds apply to the base step; the actual step varies within the configured jitter width. These options affect warmup and retained trajectories as declared, so diagnostic review remains necessary.

## Controlled retained-kernel attribution

An analytic identity Gaussian holds step 0.3, depth 8 and geometry fixed, with 20,000 retained draws, five alternating process repetitions and CPU affinity. Each engine retains owned positions. The reference compiler CPU target is unchanged between the original and optimized Alea binaries. A separate native-CPU build applies the same flag to both libraries. These timings contain neither warmup tracing nor model compilation.

| Dimensions | Optimized/original Alea time | Optimized Alea/nuts-rs | Native-CPU Alea/nuts-rs | Work and first-coordinate moment bits unchanged |
| ---: | ---: | ---: | ---: | --- |
| 2 | 1.017 | 0.7075 | 0.7351 | True |
| 16 | 0.7888 | 1.328 | 1.366 | True |
| 64 | 0.6253 | 2.314 | 2.209 | True |

The remaining cheap-target overhead is real. Callgrind on the optimized 64D/1024 draw probe reports 137,576,936 instructions: 49.61% in the inlined NUTS step, 26.49% in turning, 9.91% in cached phase completion and 4.02% in memcpy. The nuts-rs run reports 48,096,548 instructions, with its runtime SIMD kernels prominent. These are instruction counts under instrumentation, not sampled CPU-cycle percentages. The primary controlled timings use the current API with width zero. The supplementary native-CPU flag experiment precedes the jitter API addition and isolates the same deterministic trajectory calculations. Alea retains enhanced cross-subtree turning and its existing validation/RNG policy. The native-CPU build alone does not close the gap. An attempted validation-scan fusion showed no gain and an 8.6% small-target regression; it was discarded.

All 60 current default covariance-workflow unconstrained draw CSVs are byte-identical after decompression to the preceding sealed native broad campaign, including all coordinates and transition statistics. The [current hash comparison](reference-replay-final.json) provides each pair. This is stronger coverage than the three Gaussian moment smoke checks; it still does not prove arbitrary-target equivalence.

## Per-posterior final results

| Posterior | Process ratio | Retained ratio | Gradient ratio | Bulk ESS/retained-second ratio | Alea/nuts-rs divergences | Alea/nuts-rs worst R-hat |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rate_1_data-Rate_1_model | 1.074 | 0.8513 | 1.001 | 1.159 | 0/0 | 1.003/1.004 |
| Rate_2_data-Rate_2_model | 1.359 | 0.881 | 0.9976 | 1.058 | 0/0 | 1.003/1.002 |
| Rate_5_data-Rate_5_model | 1.036 | 0.8239 | 1.004 | 1.122 | 0/0 | 1.003/1.003 |
| arK-arK | 1.108 | 1.107 | 1.071 | 0.9538 | 0/0 | 1.004/1.004 |
| arma-arma11 | 1.447 | 1.074 | 1.033 | 1.032 | 0/0 | 1.002/1.002 |
| dogs-dogs_log | 1.023 | 1.029 | 1.066 | 0.9605 | 694/662 | 1.008/1.004 |
| earnings-earn_height | 1.279 | 1.305 | 1.307 | 1.078 | 0/0 | 1.011/1.01 |
| earnings-logearn_interaction_z | 1.12 | 1.146 | 1.162 | 0.9328 | 0/0 | 1.003/1.004 |
| eight_schools-eight_schools_centered | 1.292 | 1.489 | 1.233 | 1.076 | 78/76 | 1.04/1.091 |
| eight_schools-eight_schools_noncentered | 1.097 | 1.122 | 1.007 | 0.8185 | 7/8 | 1.004/1.006 |
| garch-garch11 | 1.053 | 1.061 | 1.125 | 1.021 | 0/0 | 1.005/1.005 |
| hmm_example-hmm_example | 1.05 | 1.084 | 1.117 | 0.9148 | 0/0 | 1.004/1.004 |
| kidiq-kidscore_momhs | 1.136 | 1.145 | 1.133 | 0.8499 | 0/0 | 1.003/1.004 |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | 1.308 | 1.335 | 1.33 | 1.18 | 0/0 | 1.252/1.35 |
| normal_2-normal_mixture | 0.9762 | 0.9527 | 0.9969 | 1.05 | 0/0 | 1.531/1.529 |
| radon_mn-radon_variable_intercept_noncentered | 1.029 | 1.067 | 0.9944 | 1.057 | 0/0 | 1.005/1.004 |
| sblri-blr | 0.9735 | 0.9661 | 0.997 | 1.056 | 0/0 | 1.003/1.003 |
| seeds_data-seeds_stanified_model | 1.259 | 1.388 | 1.032 | 0.4972 | 1/0 | 1.016/1.02 |
| surgical_data-surgical_model | 1.806 | 1.928 | 1.038 | 0.4444 | 7/1 | 1.006/1.004 |
| wells_data-wells_dist100_model | 1.102 | 1.097 | 1.098 | 1.027 | 0/0 | 1.004/1.002 |

These diagnostics are observations, not an adequacy decision or a filter on the speed table. Completion means the software finished its budget. Multimodal, hierarchical and pathological posteriors still require model-specific diagnostic review and often longer warmup or reparameterization. The new diagonal option is explicit; neither this experiment nor a median ESS/s ratio makes it a universal default.

## Reproduction and evidence

The [standalone probe source](probe-source.tar.gz) preserves its original repository paths; [file hashes and instructions](probe-source.json) bind the source and lock. Extract it at an unpacked cohort source root. The pre-repair control uses the [preceding native broad source bundle](../m16/posteriordb-native-v3/broad/bundle.json); the current implementation uses `final/`. Build each separately with:

```sh
nix develop .#stable --command cargo build --release --locked \
  --manifest-path tools/nuts-reference/parity/Cargo.toml
taskset -c CPU tools/nuts-reference/parity/target/release/alea-nuts-parity alea 64 20000
taskset -c CPU tools/nuts-reference/parity/target/release/alea-nuts-parity nuts-rs 64 20000
```

Replace `CPU` with the same allowed CPU for both engines. Complete all builds and tests before timing; rotate engine order across repetitions. The raw observations preserve every repetition.


- Interactive reports: [initial](initial/report.html), [symmetric restart](restart/report.html), [ordinary final tuning](fixed-final/report.html), [final jittered comparison](final/report.html). Each directory includes the complete bounded, checksummed evidence bundle, source archive, preparation/conformance records, raw draws, all outcomes and an offline report. Current and archived readers reproduced each report exactly after import.
- [All aggregate ratios](results.json), [all per-case ratios](per-case.csv), [controlled observations](fixed-geometry.json), [binary/source provenance](fixed-provenance.json).
- Protocols: `tools/bench/protocols/posteriordb-native-nuts-parity{,-final-restart,-final,-jitter}-v3.json`; standalone probe and workload contract: `tools/nuts-reference/parity/`. Use the source archive belonging to each cohort when reproducing historical policies.
- Rust validation: 291 math/MCMC tests with Faer; 104 current stable NUTS tests including jitter, allocation and failure contracts; 109 nightly SIMD tests including jitter; runtime session/release/reference tests; zero-allocation and typed-failure tests; Clippy, rustdoc and pinned public-API checks. Python benchmark suite: 56 tests, two capability-dependent skips. Exact checks and their scopes are summarized in [verification.json](verification.json).

No default, release maturity, research-sampler guarantee, or historical result was changed.
