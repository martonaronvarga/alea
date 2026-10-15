# Profile-guided NUTS optimization

This pass applies the requested [Rust optimization essay](https://gist.github.com/jFransham/369a86eff00e5f280ed25121454acec1#dont-use-inlinealways)
and [Rust Performance Book](https://nnethercote.github.io/perf-book/title-page.html)
to measured sampler work. It does **not** claim every possible optimization is
exhausted or that the results generalize to every Alea subsystem.

## Profile and changes

The saved release baseline's hardware-cycle profile attributed 19.94% to U-turn
checks, 10.62% to memmove, 8.57% to point/phase clone_from, and about 17.4% to
exp/log symbols. This includes CSV serialization, which was not optimized.
The implementation remains generic, safe Rust, with the same public API.

1. Merge inputs that are no longer observable exchange owned endpoint/candidate
   snapshots. This removes buffer copies without allocation, new lifetimes or
   interior mutability. All turning/error checks precede the exchanges.
2. Weight merging and selection share one max-shifted calculation. Ordinary
   merges need one exponential and no logarithms instead of separately computing
   the merged weight and log ratio. Gaps beyond 500 retain logarithmic selection:
   exp(-750) alone underflows, but a subtree multiplicity of 1e18 can rescue a
   representable probability. This is not a fast-math approximation.
3. U-turn momentum summation, finite checks and three scaling maxima share one
   pass. Dot-product ordering and robust positive scaling remain unchanged;
   both metric calls and all three enhanced turning checks remain in place.

No unchecked indexing, approximate transcendental functions, unconditional FMA,
relaxed floating-point semantics, new forced inlining or panic-abort policy.

## Measurements

Intel i5-8250U, one pinned logical CPU (2), portable release/fat-LTO builds.
Same four targets, seeds and draws as the earlier NUTS comparison. Preserved
before/after binaries run serially in alternating order. An initial five-round
four-variant experiment was noisy; a separate nine-round confirmation records
both process CPU time and phase wall time. All outliers are retained.

Median burn-in plus sampling seconds, summed over four chains:

| Target | Saved baseline | Optimized | Time reduction |
|---|---:|---:|---:|
| Correlated Gaussian | 0.20203 | 0.16592 | 17.9% |
| Banana | 0.21951 | 0.18101 | 17.5% |
| Logistic | 0.20899 | 0.18107 | 13.4% |
| Width-one funnel | 0.53237 | 0.41969 | 21.2% |

These are medians, not confidence bounds. Some individual runs slow substantially,
even in CPU time; machine-level variation remains. Do not compare absolute times
with another day's report to attribute a speedup. Use the paired saved baseline.

Hardware retired-user-instruction counts for a complete four-target execution,
including unchanged output formatting, provide separate evidence of reduced work:

| Cumulative variant | Instructions (billions) |
|---|---:|
| Baseline | 10.656 |
| Snapshot exchanges | 10.326 |
| Plus shared weight arithmetic | 9.729 |
| Plus fused turning scans | 8.952 |
| Optimized with codegen-units=1 | 9.082 |

The accepted source changes use **16.0% fewer instructions** in this experiment.
Instruction counts are one measurement per variant, not a hardware-independent
certificate. Cycle and branch-miss counts were noisy and are not used as proof
of a fixed latency gain.

`codegen-units=1` was tested without changing Cargo defaults. Its median phase
times were .17674/.18869/.18106/.44124 seconds: no convincing improvement over
the default optimized build, so it was **not adopted**.

Every raw CSV from all variants has the same SHA-256 as the original M5 draws:
`8fa5a86703a08e3f93e194b8528a671cd0cf3202c208786753eb17dcb52712da`.
Thus positions, step counts, divergence flags and reported acceptance match in
these runs, and their existing ESS/moment results carry over. This is evidence
for the workloads, not a promise of bitwise equivalence for every input.

## Validation and remaining strategies

The recursive tree oracle, new independent log-probability tests around the
fast-path boundary, extreme-weight tests, failure/panic recovery, full-tree
allocation checks, and Gaussian sampling across geometries/integrators pass.
Four-target moments still agree with the executed Stan/BlackJAX/nuts-rs fixture.
Faer all-target Clippy and strict Miri NUTS unit tests pass.
Combined SIMD/Faer/OpenBLAS unit, failure-contract and allocation tests also pass.

Strategy disposition:

| Strategy | Decision |
|---|---|
| Inlining / cold outlining | No blanket attributes. Profile individual call sites and code size before changing them. |
| Work reduction / loop fusion | Applied to measured merging and turning hot paths. |
| Allocation / memory reuse | Preserve zero-allocation transitions; remove copies through ownership exchanges. |
| Allocator replacement | Not useful for an allocation-free transition; cold fitting needs separate evidence. |
| LTO / codegen units | Fat LTO already enabled; one codegen unit did not help this benchmark. |
| Native CPU flags / SIMD | Keep portable defaults. Existing explicit SIMD remains; larger-dimension crossover needs measurement. |
| PGO / BOLT | Deployment-specific follow-up requiring representative training and held-out models; not measured here. |
| Cached endpoint velocities / packed tree storage | Potential larger-dimension follow-up; must preserve validity and typed metric failures. |
| Parallel chains | Separate runtime work; do not introduce global threading into small chain-local kernels. |
| Fast math / unchecked memory / abort on panic | Not acceptable substitutes for Alea's numerical and recovery contracts. |

The [inlining chapter](https://nnethercote.github.io/perf-book/inlining.html),
[iterator chapter](https://nnethercote.github.io/perf-book/iterators.html), and
[build configuration chapter](https://nnethercote.github.io/perf-book/build-configuration.html)
motivate testing these choices rather than treating them as universally faster.
The older essay contains dated API/architecture details; its prescriptions are
hypotheses, not replacements for current compiler output and measurements.

## Reproduction

Artifacts: `target/nuts-opt-2026-10-10/`, including saved binaries, baseline.perf,
`paired/results.json`, `confirmation/results.json`, all raw timing/draw CSVs,
and `stat-{baseline,swap,weights,turns,codegen1}.csv`. Intermediate variants are
cumulative. The `turns` executable is the final portable source candidate.

```sh
taskset -c 2 PYTHON tools/nuts-reference/compare_binaries.py NEW_DIRECTORY BASELINE_BINARY OPTIMIZED_BINARY --rounds 9
```

Run after all builds/tests finish. This tool records executable hashes and keeps
every raw repetition. `perf record -e cycles:u` identifies hot functions;
`perf stat -e instructions:u,cycles:u,branches:u,branch-misses:u` measures work.
See the earlier CPU benchmark for the target definitions and external-reference
limitations. High-dimensional speedups are not established by this 2D experiment.
