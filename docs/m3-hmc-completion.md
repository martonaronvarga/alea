# M3: fixed-length Euclidean HMC

This increment closes validation of the existing `HmcChain`, `Phase`, and metric
implementations. It does not introduce another sampler, change the transition
kernel, weaken safety checks, or claim adaptation/NUTS. The default HMC API is
fixed-length, fallible, target-bound and uses initialized chain-local storage.

## Validation record — 2026-09-28

- The four supported workspace matrices pass format, check, tests, Clippy with
  warnings denied and rustdoc with warnings denied: stable/default, stable/Faer
  plus experimental, nightly/SIMD plus branch hints, and combined SIMD/Faer/OpenBLAS
  plus experimental. Native tests consume the saved fixture files;
  no Python dependency enters the runtime.
- Stable release runtime library/integration tests: **51 passed**. This includes
  all deterministic references, sampling comparisons, energy-order, numerical
  stress, zero-allocation and transformed-target tests.
- Full `tools/ci/miri.sh` with strict provenance and symbolic alignment:
  **75 passed, 8 explicitly ignored**. Native FFI execution, allocation-counter
  instrumentation and long statistical/convergence sweeps retain their native
  gates. The new deterministic references and both stress tests run under Miri.
- Both new Python artifacts reproduce byte-for-byte: 54 added endpoints and
  78 sampling summaries. Native and release sampling comparisons pass without
  adjusting their preselected uncertainty bounds.
- Stress tests require bit-exact preservation of rejected caches. Independently
  recomputed densities/gradients instead use a `1e-12*(1+abs(expected))` bound:
  Miri correctly exposed unspecified low-bit precision in transcendental functions.

These are local x86-64 Linux results, not remote CI or cross-platform evidence.

## Independent deterministic reference

The content-pinned BlackJAX 1.5 / JAX 0.9.2 / CPU-float64 generator now supplies
54 additional trajectories, retaining the earlier 36. Five targets × three metrics
× two directions × three lengths give **90 endpoints and their reversals**.
Rust computes analytic gradients; JAX differentiates independently. Each test
checks inverse-mass conventions, initial/final positions, momenta, gradients,
log density and Hamiltonian with mixed tolerance `2e-12*(1+abs(reference))`.
Coverage assertions reject missing/duplicate configurations.

New targets:

- Logistic regression: standard-normal prior; x = [-1.5,-0.2,0.7,2],
  y = [0,1,0,1], predictor q0 + q1*x; stable softplus.
- Centered funnel: q0 ~ N(0,1), q1 | q0 ~ N(0,exp(q0)). Width is **one**,
  not Neal's more difficult width-three funnel.
- Rotated Gaussian: z0 = (0.8*q0+0.6*q1)/0.01,
  z1 = -0.6*q0+0.8*q1; independent standard-normal z gives condition number 1e4.
  Earlier diagonal condition-1e8 tests remain in place.

Metrics use the previous identity, diagonal factor [2,0.5], and dense lower
factor [[1.5,0],[0.4,0.8]]. Endpoint step sizes are ±0.125, except the stiff
rotated target uses ±0.000625. Lengths are 1,5,11. Separate fixed-duration
convergence tests cover all 15 target/metric pairs at 32,64,128 steps: halving
the step must reduce absolute energy error by a factor in [3.7,4.3].

## Independent sampling with uncertainty

`sampling.py` executes BlackJAX's actual endpoint-HMC kernel, not a hand-written
Python integrator and not Rust-produced expected values. JIT compiles a sequential
`lax.scan`; the complete pinned Python environment is unchanged.

There are 13 configurations: correlated Gaussian, banana, logistic and funnel
with all three metrics, plus the rotated Gaussian with matched dense mass equal
to its precision. Each implementation runs **four independent chains**, discards
1,024 transitions, retains 8,192, and includes every rejection. Thus each runs
52 chains / 479,232 transitions / 425,984 retained states. There is no thinning
or adaptation. BlackJAX seeds are 301–304; Rust SmallRng seeds are 901–904.
No cross-library random-stream equivalence is assumed.

Step size/count: 0.15/9 except funnel 0.06/18. Matched rotated mass is
[[6400.36,4799.52],[4799.52,3600.64]]. This is intentional fixture tuning, not
evidence that identity-mass HMC handles arbitrary ill-conditioning automatically.

Six observables: two coordinates, their squares and product, plus the acceptance
indicator. For Gaussian/banana/funnel/rotated targets the coordinates are whitened
to independent standard normals; logistic uses raw coefficients.

Nonoverlapping batch means estimate MCSE at batch lengths 256 and 512, respecting
chain boundaries. We use the larger estimate. Agreement requires
`abs(mean_rust-mean_reference) <= 6*hypot(mcse_rust,mcse_reference)` for all 78
summaries. MCSE must be positive, finite and below 0.2. Whitened moments must also
agree with [0,0,1,1,0] within six respective MCSEs in **both** implementations.
Every sampling transition must be nondivergent. No tolerances were adjusted
after viewing the comparisons; all configurations passed the initial run.

These are approximate Monte Carlo uncertainty gates with conservative multipliers,
not exact simultaneous confidence coverage, modern ESS/R-hat, or a mixing proof.
Finite burn-in and two batch scales do not certify arbitrary posterior geometries.
Modern diagnostics belong to M6. The reference is BlackJAX, not an executed Stan
sampling comparison; existing Wiener/Stan likelihood references are separate.

## Numerical and allocation checks

Stress regressions exercise unstable logistic, narrow-funnel and rotated-Gaussian
trajectories. Repeated divergence leaves position, gradient and log density
bit-identical; resetting restores the coherent cache. Existing true overflow,
model-error, panic, momentum covariance and MH-rejection tests remain required.

The original first-transition allocation checks cover all three metrics and
dimensions through 129. New checks measure 128 individual leapfrog steps for
logistic/funnel/rotated targets with every metric, including the first call.
No Rust allocations or bytes are observed. M2's transformed-model/Wiener/Enzyme
allocation tests remain applicable. This guarantee covers library-owned hot paths
and tested models: arbitrary user callbacks may allocate. The Rust counter cannot
observe malloc inside BLAS; optional foreign backends do not inherit a blanket
foreign-allocation guarantee. Default scalar/SIMD/Rust operations avoid that boundary.

The symmetric absolute **endpoint** energy-error cutoff is unchanged. BlackJAX
uses its own divergence policy; comparisons use nondivergent chains and therefore
do not claim policy identity on pathological trajectories. No numerical rejection
policy is a substitute for geometry-appropriate tuning.

## Reproduce

```sh
nix shell --impure --expr 'import ./tools/blackjax-reference' --command python3 tools/blackjax-reference/generate.py --m3 --check crates/runtime/tests/fixtures/blackjax-m3.csv
nix shell --impure --expr 'import ./tools/blackjax-reference' --command python3 tools/blackjax-reference/sampling.py --check crates/runtime/tests/fixtures/blackjax-sampling.csv
nix develop .#stable --command bash tools/ci/check.sh
nix develop .#stable --command bash tools/ci/check.sh kernels/faer,runtime/experimental
nix develop --command bash tools/ci/check.sh kernels/simd,kernels/branch-hints,memory/simd
nix develop .#blas --command bash tools/ci/check.sh kernels/simd,kernels/faer,kernels/openblas,runtime/experimental
nix develop --command bash tools/ci/miri.sh
nix develop .#stable --command cargo bench --manifest-path crates/Cargo.toml -p runtime --bench hmc --locked -- --save-baseline m3-20260928
```

Native CI consumes saved reference CSVs without Python. Explicit regeneration
checks require byte-identical output on the recorded environment. Miri exercises
deterministic reference trajectories and stress tests, not long Monte Carlo or
convergence sweeps, allocation instrumentation, JAX, or foreign BLAS.

## Performance scope

Criterion retains legacy/checked Gaussian dimensions 8,64,1024 and adds identity,
diagonal and dense 64-dimensional metrics; a 1,024-observation, 16-coefficient
logistic model; and a bounded transformed Wiener4 model. Setup is outside timing;
each transition must complete all eight steps with no divergence. These establish
throughput baselines, not a claim of superiority to Stan/BlackJAX or peak SIMD.

Executed sequentially after all other task-owned builds/tests finished, on an
Intel i5-8250U, x86-64 Linux, stable rustc 1.95.0 (LLVM 22.1.2), default features,
release optimization and workspace fat LTO. No CPU affinity, governor control or
background-process isolation was imposed. Criterion 0.5 reports these point
estimates and 95% intervals in **microseconds per eight-step transition**:

| Case | Point estimate | 95% interval |
|---|---:|---:|
| Legacy Gaussian, dimension 8 | 1.0779 | 1.0732–1.0841 |
| Checked Gaussian, dimension 8 | 1.0261 | 1.0210–1.0341 |
| Legacy Gaussian, dimension 64 | 6.4564 | 6.4194–6.5015 |
| Checked Gaussian, dimension 64 | 6.3967 | 6.3848–6.4105 |
| Legacy Gaussian, dimension 1024 | 102.48 | 102.26–102.75 |
| Checked Gaussian, dimension 1024 | 102.49 | 102.17–103.00 |
| Gaussian64 / identity | 6.4408 | 6.4170–6.4618 |
| Gaussian64 / diagonal | 6.4696 | 6.4549–6.4916 |
| Gaussian64 / dense | 83.012 | 82.876–83.188 |
| Logistic1024x16 / identity / fused model | 588.86 | 582.57–598.52 |
| Wiener4 / bounded transforms / identity | 6.2746 | 6.2294–6.3328 |

Protocol pairs use step size 0.1; workload cases use 0.02. Protocol groups use 30
samples, 0.5 s warmup and 2 s measurement; workload groups use 20 samples, 0.3 s
warmup and 1 s measurement. Raw Criterion artifacts live under ignored
`crates/target/criterion/`, baseline `m3-20260928`. The checked kernel is not slower
than legacy by a large factor in this run; small differences and overlapping
intervals must not be generalized into a throughput advantage. The transition
kernel itself did not change in M3. Gaussian, logistic and Wiener workload sizes
are different, so their timings do not compare algorithm efficiency or ESS/sec.

All eleven cases executed successfully. Outliers were reported, notably 6/20 in
the dense workload; repeat under controlled CPU conditions before optimizing.
The much larger dense and logistic costs identify workloads to profile, not proof
of a particular bottleneck. No unsafe, unchecked, specialized SIMD or compiler-flag
change is justified by this single baseline alone.

Primary references: [BlackJAX 1.5 HMC source](https://github.com/blackjax-devs/blackjax/blob/1.5/blackjax/mcmc/hmc.py),
[BlackJAX integrators](https://github.com/blackjax-devs/blackjax/blob/1.5/blackjax/mcmc/integrators.py).
