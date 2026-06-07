# Wiener Kernel TODO

Prioritized follow-up work for the Wiener distribution kernels.

## P0 Correctness / numerical robustness

1. Add a robust `Wiener7` underflow fallback.
   - Current `Wiener4` and `Wiener5` log-scale kernels stayed finite in an adversarial finite-support sweep.
   - `Wiener7` can still underflow inside support because it integrates natural-scale inner `Wiener5` densities.
   - Prefer a rare-path fallback: keep adaptive natural-scale integration on ordinary paths, retry with scaled/log fixed quadrature when the adaptive result has zero or non-finite density.

2. Add permanent finite-support stress tests.
   - Include the known difficult case around `alpha=0.2`, `beta=0.1`, `delta=-10`, `s_beta>0`, very small decision time.
   - Keep the sweep bounded enough for CI; use a larger ignored/stress test for local reports.

## P1 Performance

1. Benchmark direct SoA versus `BatchStrategy::BranchBuckets` on real workloads.
   - Branch buckets are runtime-selected because they are not always faster.
   - Watch large `Wiener5` SoA batches where scratch scans may dominate.

2. Add a reusable batch workspace.
   - Avoid allocating scratch buffers on each target evaluation.
   - Use one contiguous `OwnedBuffer` with bucket offsets if branch bucketing remains useful.

3. Continue SIMD work only where the assembly shows vectorized transcendental work.
   - `std::simd::StdFloat` gives vectorized `exp`/`ln` on nightly.
   - Keep scalar fallbacks and strict equivalence tests.

4. Keep function-level Criterion benchmarks for the series kernels.
   - Track `small_branch_fused`, `large_branch_fused`, and common fixed `k` cases separately.

## P2 API / organization

1. Split `wiener7.rs` further.
   - Suggested modules: adaptive cubature, fixed quadrature, endpoint gradients, reference/audit path.

2. Split `types.rs` when it grows again.
   - Suggested modules: params, eval, options, observations.

3. Compare against Stan Math in a reproducible benchmark harness.
   - Normalize by operation: scalar log density, scalar fused gradient, and batch target evaluation.
   - Record compiler flags, branch-hints/SIMD features, and data distributions.
