# Native NUTS walltime repair — 2026-10-23

The explicit position-turning profile reaches approximate walltime parity with
nuts-rs on the 20-model native PosteriorDB cohort: median retained time is
**0.991×**, and full fresh-process time is **1.021×**. Retained gradient calls are
**0.994×**. This closes the previously measured kernel/call-count gap; it does
not establish a win on every posterior or convergence on difficult mixtures.
The constructor still selects covariance adaptation and generalized turning.
Those defaults are available, but the covariance workflow was not rebenchmarked
in this follow-up. Near-parity results below use the explicit settings shown here.

```rust
let nuts = NutsOptions::new(0.1, 8)?
    .with_step_size_jitter(0.1)?
    .with_position_turning(true);
let (chain, report) = NutsWarmup::new(
    &target, initial, nuts, WarmupOptions::new(512),
)?
.with_gradient_diagonal()?
.with_gradient_final_restart(false)
.run(&mut rng)?;
```

## Native posterior results

All 180 jobs completed: 20 models × 3 methods × 3 repetitions; each job runs four
sequential chains with 512 warmup and 1024 retained draws, depth 8, on one dusk CPU.
Both libraries evaluate the same ahead-of-time Enzyme Rust density/score target.
All 20 native ports passed the existing independent Stan/gradient/HVP conformance
checks before measurement. Every retained draw and divergence remains available.

Each ratio is Alea/nuts-rs. First take the median of the three paired repetition
ratios for each posterior, then the median across all 20 posteriors. These are
unfiltered descriptive observations, not confidence intervals or a promotion gate.
Repetition labels share budgets and seed labels; the libraries use independent
RNG streams. Per-case absolute process medians are also retained in the CSV.
Process time includes worker setup, warmup, sampling and output; compilation and
common external diagnostics have separate scopes. The process clock uses
perf_counter_ns/pidfd completion, with no 50 ms polling.

| Metric | Generalized turning + gradient/jitter | Position turning + gradient/jitter |
|---|---:|---:|
| Fresh process walltime | 1.111 | 1.021 |
| Warmup walltime | 1.120 | 1.009 |
| Retained walltime | 1.117 | 0.991 |
| All gradient calls | 1.056 | 0.994 |
| Retained gradient calls | 1.052 | 0.994 |
| Minimum bulk ESS / process second | 0.956 | 0.937 |
| Minimum bulk ESS / retained second | 0.976 | 0.980 |

The position profile is faster in 7/20 process-time
medians and 12/20 retained-time medians. The median
ESS/s ratio remains below one: **0.937×**
including process costs and **0.980×**
during retention. The small remaining walltime gap and the stochastic efficiency
variation remain visible; inference efficiency is not uniformly superior.

The preceding explicit generalized/jitter cohort had process/retained ratios
1.105/1.091. That historical evidence is preserved in the
[previous report](../nuts-parity-2026-10-23/README.md). All 60 new nuts-rs draw CSVs
are byte-identical after decompression to that preceding cohort, including every
coordinate and transition statistic: [reference replay](reference-replay.json).
Its timings were freshly measured.

| Posterior | d | Process ratio | Retained ratio | Retained call ratio | Retained ESS/s ratio | Divergences Alea / nuts-rs | Worst R-hat Alea / nuts-rs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rate_1_data-Rate_1_model | 1 | 1.081 | 0.876 | 0.986 | 1.034 | 0 / 0 | 1.0020 / 1.0037 |
| Rate_2_data-Rate_2_model | 2 | 1.053 | 0.959 | 1.008 | 1.121 | 0 / 0 | 1.0027 / 1.0020 |
| Rate_5_data-Rate_5_model | 1 | 1.037 | 0.899 | 0.995 | 1.078 | 0 / 0 | 1.0036 / 1.0030 |
| arK-arK | 7 | 1.382 | 0.987 | 0.989 | 0.873 | 0 / 0 | 1.0021 / 1.0044 |
| arma-arma11 | 4 | 0.989 | 0.990 | 1.001 | 1.170 | 0 / 0 | 1.0025 / 1.0016 |
| garch-garch11 | 4 | 1.028 | 0.966 | 0.990 | 0.942 | 0 / 0 | 1.0057 / 1.0047 |
| dogs-dogs_log | 2 | 1.017 | 1.006 | 0.988 | 0.900 | 614 / 662 | 1.0060 / 1.0044 |
| earnings-earn_height | 3 | 0.946 | 0.960 | 0.998 | 1.048 | 0 / 0 | 1.0151 / 1.0102 |
| earnings-logearn_interaction_z | 5 | 0.995 | 0.982 | 0.987 | 1.095 | 0 / 0 | 1.0034 / 1.0036 |
| eight_schools-eight_schools_centered | 10 | 1.630 | 1.352 | 0.958 | 0.845 | 100 / 76 | 1.2834 / 1.0913 |
| eight_schools-eight_schools_noncentered | 10 | 1.056 | 1.121 | 0.998 | 0.802 | 13 / 8 | 1.0044 / 1.0062 |
| low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse | 5 | 1.033 | 1.011 | 0.988 | 1.057 | 0 / 0 | 1.2740 / 1.3504 |
| normal_2-normal_mixture | 3 | 1.008 | 1.018 | 0.993 | 0.982 | 0 / 0 | 1.5291 / 1.5289 |
| sblri-blr | 6 | 1.100 | 1.069 | 1.001 | 0.928 | 0 / 0 | 1.0021 / 1.0026 |
| wells_data-wells_dist100_model | 2 | 1.008 | 1.053 | 1.022 | 0.878 | 0 / 0 | 1.0077 / 1.0023 |
| kidiq-kidscore_momhs | 3 | 1.025 | 1.013 | 1.028 | 0.979 | 0 / 0 | 1.0025 / 1.0040 |
| seeds_data-seeds_stanified_model | 26 | 0.878 | 0.848 | 0.984 | 0.980 | 0 / 0 | 1.0131 / 1.0197 |
| surgical_data-surgical_model | 14 | 0.932 | 0.998 | 1.007 | 0.881 | 3 / 1 | 1.0118 / 1.0044 |
| radon_mn-radon_variable_intercept_noncentered | 89 | 0.962 | 0.953 | 0.961 | 0.999 | 0 / 0 | 1.0073 / 1.0044 |
| hmm_example-hmm_example | 4 | 0.967 | 0.991 | 0.997 | 0.908 | 0 / 0 | 1.0031 / 1.0038 |

Divergence counts sum all three repetitions; R-hat is the maximum across them.
Both mixtures remain poorly mixed. A favorable time ratio on such a row does not
establish adequate inference. The full generalized profile is also preserved;
it has 973 divergences, including 187 on seeds, versus 730 for position turning and
747 for nuts-rs. Different RNG paths can expose adaptation sensitivity.

- **seeds_data-seeds_stanified_model:** process/retained ratios 0.878/0.848; retained ESS/s 0.980; divergences 0 versus 0; worst R-hat 1.0131 versus 1.0197.
- **surgical_data-surgical_model:** process/retained ratios 0.932/0.998; retained ESS/s 0.881; divergences 3 versus 1; worst R-hat 1.0118 versus 1.0044.

## Fixed-kernel attribution

The analytic standard-normal probe holds identity geometry, step 0.3, depth 8 and
seed 43001 fixed, burns 33 transitions, then times 20,000 retained transitions.
Both libraries collect an owned position and accumulate the same two moments.
No model compiler, CSV or diagnostics runs inside retained timing. Ten rotated,
interleaved repetitions ran on dusk CPU 0 **after** the native campaign finished.
The CPU was an Intel i7-7500U; exact binary hashes, platform, clocks and raw 150
observations are in [fixed-protocol.json](fixed-protocol.json),
[fixed.json](fixed.json), [fixed-summary.json](fixed-summary.json) and
[compiler/build scope](fixed-build.json).

| Dimension | New position / current nuts-rs build | New position / older nuts-rs control | New position / previous Alea |
|---|---:|---:|---:|
| 2 | 0.798 | 0.837 | 1.126 |
| 16 | 0.875 | 0.882 | 0.671 |
| 64 | 0.930 | 0.968 | 0.417 |

The 64-dimensional kernel is about 2.4× faster than the previous Alea build and
faster than both nuts-rs control builds in this experiment. The 2-dimensional
case regresses about 13% against previous Alea while retaining an advantage over
both reference builds. The two reference builds differ by up to about 5% in
median runtime; their work counts and moments are identical. This control is
preserved rather than selecting the slower reference.

These are fixed-seed smoke observations, not posterior convergence claims. New
large-vector normal generation changes RNG paths, so before/after Alea sample
bytes are not required to match. Actual gradient/step counts and moments are
reported. The same probe binaries under local Callgrind execute 137,618,680 versus
54,291,425 instructions for 64 coordinates/1024 retained draws, a 60.5% reduction.
These are instruction counts, not CPU cycles; the profiling host is separately
recorded in [profiles/summary.json](profiles/summary.json).

## Implemented numerical and runtime changes

- Constant Euclidean geometry can use position turning without a transformed
  position buffer: for G=L L^T, q=Lz and p=L^-T r give delta(q) dot p = delta(z)
  dot r. Both endpoints and both enhanced cross intervals are checked. This
  follows the whitened original NUTS criterion used by the pinned nuts-rs
  Hamiltonian; see [Hoffman and Gelman, section 5](https://jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf).
  Canonical momentum is essential here; physical-coordinate velocity would give
  a different test. Generalized turning stays available.
- Cached infinity-norm bounds and Hölder's inequality certify ordinary dot signs
  without materializing/scanning temporary sums. All six enhanced dots share a
  SIMD pass. Every uncertain, cancelling, overflow or underflow case retains the
  checked ordered normalized fallback, including eager admission of all intervals.
- Pulp provides stable CPU dispatch for coordinate operations, finite checks,
  turning and norm admission. Kicks/drifts retain separate multiply/add; no FMA
  contracts the integrator. Finite failures retain the first invalid index.
- Checked owned point exchange removes a coordinate copy. Immutable endpoint
  norm admission shares the velocity finite pass; tree carries exchange pointers.
  Sampler transitions allocate nothing through complete 1023-edge trees in tested
  scalar/SIMD sizes and jitter/turning combinations.
- Ordinary relative-energy weights share one leaf exponential with acceptance,
  then merge using positive additions/divisions. Extreme values retain the stable
  logarithmic maximum/scaled sum, including candidate multiplicity.
- Nonnegative half-scaled kinetic terms use independent accumulators at 16 or more
  coordinates. This reduces accumulation error and avoids overflow of an unhalved
  norm whose final energy is representable. Euclidean phase starts at those sizes
  use rand_distr 0.6.0 Ziggurat normals; smaller starts retain the polar generator. The
  global polar helper and RWMH cached spare remain unchanged.

The new dependency uses are pinned Pulp 0.22.3 and rand_distr 0.6.0, with their
existing permissive licenses. No unsafe code is introduced in Alea's kernels.
No new research sampler or retained adaptation is introduced. Public behavior and
API details are in [docs/nuts.md](../../nuts.md). Checkpoint implementation
identity is now **numerics/3**: old numerics/2 checkpoints are rejected, while old
schema-one draw streams remain readable. See
[session compatibility](../../m15-sessions.md).

## Verification and reproduction

- 349 broad core/math/MCMC Faer tests/doctests and 134 focused nightly SIMD/Faer
  tests pass. Runtime 26 tests pass with one preexisting long-soak ignore.
- A strict-provenance Miri check of the 17-coordinate owned exchange, including
  error/panic rollback and runtime finite checks, passes.
- Typed backend errors/panics, owned-cache rollback, first-error indexing,
  non-FMA rounding, norm overflow, zero allocation, frozen warmup and conditional
  jitter/RNG replay pass. Independent recursive tree checks cover both criteria;
 20,000 exponent/cancellation cases plus 10,000 combined-certificate cases retain
  the original checked signs. Dense whitening has an independent algebra check.
- Clippy with warnings denied, rustdoc with warnings denied, public API 7105-entry
  matrix, maturity/exactness/compatibility inventories, architecture, downstream
  conformance and seven compile-fail contracts pass. Python benchmark 56 tests
  (two optional skips) and CI 55 tests pass.
- [Validation commands and logs](checks/commands.json) accompany these counts.
- The 180-job bundle is unpacked and verified with both current and archived
  readers, producing identical reports. [Verification](native/verification.json),
  [interactive report](native/report.html), [raw bundle index](native/bundle.json),
  [numeric summary](results.json), [per-case CSV](per-case.csv).

The final source capsule contains 626 checked files, including the standalone
probe source, pinned reference source and their locks:
[source.tar.gz](source.tar.gz), [source hashes](source.json). Extract it into a
fresh directory and use its lock-pinned stable shell to build
`tools/nuts-reference/parity/Cargo.toml` in release mode. Run
`alea-nuts-parity ENGINE DIM 20000`, with ENGINE=`alea`, `alea-position` or
`nuts-rs`, DIM=2/16/64. Alternate engines on a fixed CPU with no concurrent builds.
The earlier Alea source and probe reconstruction are linked in the previous
report; the old/new binary hashes in fixed-protocol.json bind the comparison.
Native source/preparation archives and every raw draw are inside the bounded
native bundle. Batch build cost is separate from per-process timing.

A preliminary source export failed the vendored-header symlink preflight before
execution. A subsequent partial run was stopped to include the checkpoint replay
guard in the final source: 42 completed jobs, one interrupted and 137 unstarted,
all preserved in [partial-02](partial-02/partial.json). They are not pooled into
the final 180-job comparisons. The sampler arithmetic was unchanged between that
partial attempt and the final cohort. No maturity promotion or change to constructor warmup/turning defaults is
inferred from this descriptive experiment.
