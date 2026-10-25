# Wiener kernels

Implemented and checked on 2026-10-25. Wiener4/5/7 retain their existing boundary
convention and analytic scores. Ordinary evaluation still starts with adaptive
natural-scale cubature; the tail repair is a rare path.

## Numerical robustness

Wiener7 retries zero, subnormal or non-finite density results with locally scaled
quadrature. Each region uses a shared log-density scale for the density and all
seven score integrals. Independent GL7/GL15 product rules estimate density and
score-ratio errors; regions are refined up to 100,000 inner evaluations. An
unresolved integral or unattainable precision returns a checked `NumericalError`.
The discrepancy estimate is not an accuracy certificate for arbitrary inputs.
Explicit fixed quadrature retains the selected order and uses log scaling if
needed; it remains a fixed-rule approximation.

Width derivatives on the scaled path integrate the inner score times the uniform
coordinate. This avoids subtracting large endpoint ratios. Non-decision-time
integration stops at the observation's support boundary. With both uniform widths
zero, log evaluation delegates directly to Wiener5 without an `exp`/`ln` round trip.
Checked fused evaluation requires finite scores as well as a finite log density.

The upstream starting-point derivative correction was merged in
[Stan Math PR 3324](https://github.com/stan-dev/math/pull/3324).
The full-model underflow work inspected here is
[commit c8b33c4](https://github.com/martonaronvarga/math/commit/c8b33c42cc5c222802579084e7096358a0a3423a),
following [issue 3329](https://github.com/stan-dev/math/issues/3329). Its scaled
fallback was absent from the upstream `develop` header inspected on this date.
The comparison below pins that underflow branch rather than an unspecified Stan
installation. Local Stan worktrees, including uncommitted research, are untouched.

Permanent tests include the three Stan underflow regressions with log densities
near −4,983/−4,998, finite-difference scores including non-decision time, the
zero-variability tail, and a 96-case finite-support grid. The ignored extension
checks 336 cases. Both boundaries, short decision times, drift ±10, small boundary
separations and all active variability combinations are covered. These are bounded
regressions, not a claim that every finite-support input can be resolved.

## Batch workspace and imports

`WienerBatchWorkspace` owns one initialized aligned buffer. A counting partition
places the seven branch/term-count buckets in contiguous ranges. Bucket evaluation
therefore avoids repeated full-buffer scans. An explicit
`batch_value_gradient_with_workspace` method supports caller-owned reuse;
`WienerModel::with_workspace` retains scratch in a chain-local `WienerBatchModel`.
It is available for Wiener4/5. Construction/growth can allocate; repeated valid
calls within capacity do not. Allocation-counter and strict-provenance Miri tests
also check reuse after an invalid observation.

```rust
use alea_core::prelude::*;
use alea_distributions::prelude::*;

let observations: WienerObservations = vec![
    WienerObservation { rt: 0.8, boundary: Boundary::Upper },
].into();
let model = WienerModel::new(
    Wiener5,
    observations.with_batch_strategy(BatchStrategy::BranchBuckets),
).with_workspace();
```

Normal owning-module imports remain available. The former `reference` modules
were removed; there are no compatibility aliases. Preludes collect common types,
while experimental methods stay explicitly marked in their owning modules.
See [migration](migration.md). Release reviews describe frozen artifacts rather
than assigning release status to an import path.

Wiener7 is split into adaptive, fixed, endpoint, scaled and test-only audit modules.
Parameters, evaluations, options and observations have separate private modules;
the public Wiener type paths stay under `alea_distributions::wiener`.

## Performance observations

The [raw Criterion samples and estimates](benchmarks/wiener-2026-10-25/criterion.json)
cover direct SoA and reusable buckets on mixed upper/lower observations at
128/1,000/10,000 rows. Reaction times are deterministic synthetic values from
0.35 to 1.85 seconds, with the parameters recorded in the benchmark source.
The final run used stable release builds and CPU 2 after other checks completed.
Medians below are nanoseconds per observation.

| Family | Rows | Direct SoA | Reused buckets | Bucket/direct |
|---|---:|---:|---:|---:|
| Wiener4 | 128 | 261.6 | 259.3 | 0.991 |
| Wiener4 | 1,000 | 275.1 | 247.5 | 0.900 |
| Wiener4 | 10,000 | 276.2 | 244.7 | 0.886 |
| Wiener5 | 128 | 327.1 | 304.7 | 0.932 |
| Wiener5 | 1,000 | 334.4 | 324.9 | 0.972 |
| Wiener5 | 10,000 | 320.5 | 321.1 | 1.002 |

`Direct` remains the default. Buckets are an explicit workload-dependent choice.
Separate Criterion cases exercise production `small_branch_fused` at k=2/3/5 and
`large_branch_fused` at k=4/5/6, without exposing benchmark-only public methods.

The [Stan comparison](benchmarks/wiener-2026-10-25/stan-comparison.json) retains all
140 observations from five alternating process pairs per case, plus independent
primal finite differences. Scalar log density, scalar fused score and batch target
work are timed separately. Compilation and setup are excluded. Correct batch
controls share Stan parameters and perform one reverse sweep over a scalar sum.
The vectorized API is tested separately.

| Family | Scalar logp Alea/Stan | Scalar fused | Direct batch | Bucket batch |
|---|---:|---:|---:|---:|
| Wiener4 | 1.755 | 0.262 | 0.156 | 0.152 |
| Wiener5 | 0.306 | 0.156 | 0.143 | 0.125 |
| Wiener7 | 0.371 | 0.203 | 0.011 | — |

These are single-host observations on these models. Precision policies differ:
Alea4/5 request 1e-12; Stan4 has a fixed density bound of 1e-6; Stan5 uses its
internal density policy with derivatives requested at 1e-12. Full-model derivative
precision is 1e-4. Passing value/score comparisons accompany eligible ratios, but
they do not establish matched-error performance. Wiener4's Alea logp wrapper still
computes the fused score internally, which explains work absent from a primal-only
comparison. No general language, model or sampler speedup is inferred.

The pinned Stan branch's vectorized Wiener5/7 calls with scalar shared parameters
return inconsistent gradients despite matching log densities. Alea and Stan's
scalar sum agree with independent finite differences: maximum scaled residuals
are below 3e-5 here. The vectorized controls fail by orders of magnitude; their
observations are retained and their timing eligibility is false. This is a
reproduced finding for the pinned branch, not a claim about every Stan release.

The [assembly inspection](benchmarks/wiener-2026-10-25/assembly.json) of the existing
nightly SIMD bucket path found packed arithmetic alongside scalar libm `exp`/`log`
references, with no vector-transcendental symbol. `StdFloat` intrinsics alone do
not establish vectorized transcendental work. SIMD remains optional; no new SIMD
kernel or transcendental speedup claim was added.

## Reproduce

```sh
nix develop .#stable --command cargo test --manifest-path crates/Cargo.toml -p alea-distributions
nix develop .#stable --command cargo test --manifest-path crates/Cargo.toml -p alea-distributions --test wiener_underflow -- --ignored
nix develop .#stable --command taskset -c 2 cargo bench --manifest-path crates/Cargo.toml -p alea-distributions --bench wiener7 -- 'wiener/(strategy|series)'
nix develop .#stable --command python3 tools/stan-reference/benchmark_wiener.py --stan-repo /path/to/math --cpu 2 --out /new/output
```

The comparison needs Eigen 3.4, Boost 1.87, Sundials 6.1.1 and built TBB 2020.3 in
that checkout's `lib` tree. Stan headers are extracted from the pinned Git commit;
the working tree's modified headers are never used. The driver seals source inputs and retains worker binaries before measurement,
then records compiler commands, feature flags, CPU affinity and actual
header/native-library and worker hashes.

Full reports, sealed source inputs, binaries and assembly are retained outside
the checkout in `../alea-wiener-evidence/`. The committed
[manifest](benchmarks/wiener-2026-10-25/manifest.json) identifies their hashes and
paths. Development attempts remain separate from the final observations.
