# Backend and Fisher profile follow-up

Local measurements, not universal hardware rankings or sampler promotion.
Covariance/dual-averaging defaults are unchanged. M5 is not implemented here.

## Isolated backend actions

`tools/backend-profile.py` builds all four backends using the same pinned nightly
compiler in `.#blas`, one feature at a time, then measures serially with two
discarded warmups and six retained rounds, alternating backend order. No native
CPU flags; BLAS/OMP thread counts are one. Construction is outside the inner
timer. Each repetition is one velocity action plus one momentum transform.

Median inner nanoseconds per pair (October 9 artifact):

| Backend | d8 | d64 | d256 |
|---|---:|---:|---:|
| Scalar | 300 | 10508 | 183076 |
| SIMD, repaired | 314 | 6161 | 63684 |
| Faer | 715 | 8543 | 63758 |
| OpenBLAS | 507 | 5687 | 42275 |

Dense dispatch prioritizes OpenBLAS, then Faer, then explicit SIMD, then scalar.
Low-rank metric actions remain scalar source loops, potentially autovectorized;
their feature rows are controls, not distinct low-rank backend implementations.
Cold Fisher decomposition uses Faer with its own SIMD-capable kernels.

The explicit SIMD implementation previously forced `mul_add` on a portable
compiler target without guaranteed hardware FMA. Disassembly exposed scalar
FMA dispatch. Separate multiply/add removes that obligation and preserves the
scalar arithmetic convention, although reduction orders still differ. No unsafe
CPU assumption, fast-math option, or process-global threading change was added.
SIMD tails/unaligned slices and independent metric action tests pass.

A separate direct old/new binary comparison (d256, 3000 pairs, two warmups/eight
runs, same compiler) measured 1.407 s versus 185.6 ms: 7.58x faster in this workload.
Raw artifact: `target/simd-paired-2026-10-09.json`. This is not a sampler speedup.

The original October 6 runner used Python `wait4`, whose child RSS inherited the
launcher high-water mark (every row reported 23088 KiB). Those memory values are
invalid benchmark evidence. The corrected runner uses GNU time's isolated child;
observed process peaks range 2544–5988 KiB, including construction and runtime.
These are not heap allocations or high-dimensional Fisher memory gates.
Raw versions, CPU, binary/source hashes, and rounds:
`target/backend-profile-2026-10-09/results.json` (local artifact).

## Fisher reductions

Baseline perf sampling put 28.3% of samples in the inlined fit frame; annotation
identified the modified Gram–Schmidt reductions as a hotspot. Four independent
accumulation chains replace serial sums, including both reorthogonalization
passes. This changes rounding, not the estimator or regularization.

Saved pre-change and post-change binaries use the same pinned stable compiler,
Faer, and release profile with debug information. Isolated Hyperfine: two warmups,
eight repetitions per command. Mean process-batch time:

| Workload | Before | After | Reduction |
|---|---:|---:|---:|
| 100 fits, d64/history64 | 371.1 ms | 327.5 ms | 11.7% |
| 10 fits, d256/history160 | 1048 ms | 840.8 ms | 19.8% |

Process setup is amortized, not subtracted. Commands are grouped, not randomized;
host drift remains possible. The earlier preliminary pass overlapped diagnostics
and is excluded. Raw final artifact: `target/fisher-dot-isolated-2026-10-09.json`.
The independent 1440-case noncommuting, 240-case spectrum, upstream-adjudicated
and extreme-scale math suites pass their unchanged tolerances.

## Promotion status

Independent ArviZ diagnostics of the pre-reduction raw draws pass 319/320 gates.
The failed correlated-Gaussian rank2/window80/history160 ensemble has R-hat
1.011216 > 1.01; bulk/tail ESS pass. Retain the failure and the threshold.
See `tools/fisher-reference/EFFICIENCY.md` for reproduction and limitations.
Modern diagnostics are offline reference tooling, not an implemented M6 runtime.
Neither these kernel timings nor the mathematical fixtures establish whole-chain
superiority over Stan, BlackJAX, walnutpie or nutpie. Those timed comparisons and
the broader rank/history memory gates remain outstanding.

The post-optimization 256-chain rerun also passes all 320 moment checks and
319/320 modern gates, with the same failed diagnostic values. Raw artifacts:
`target/m4f-optimized-2026-10-09.{csv,bin}` and
`target/m4f-optimized-diagnostics-2026-10-09.json`. Full Faer math tests, Fisher
sampling/continuous tests, formatting and all-target math/MCMC Clippy pass.
