# CPU timing protocol v1

Declared before measurement, 2026-10-09. This benchmarks current NUTS on the
four existing two-dimensional M5 targets, not high-dimensional scaling or all
Alea algorithms. Four serial chains per target, seeds 501–504, 1024 discarded
iterations and 8192 retained iterations; step sizes .15/.06, depth 8, cutoff 1000.
The correctness protocol and version pins in README.md remain unchanged.

Three fresh-process repetitions, rotating backend order. Build Rust runners in
release mode with fat LTO; builds and diagnostic analysis are outside timing.
Use a single allowed CPU via taskset and set native thread counts to one.
No simultaneous build/test/profiling workload during measurements. Preserve all
draws and failures. Report medians and raw repetitions, not a best-case run.

Alea and nuts-rs time discarded and retained loops separately, storing retained
coordinates/stats in preallocated memory and writing CSV afterwards. Nuts-rs's
native draw expansion/allocation is included. Stan reports native warmup/sampling
times including its CSV writer (17 significant digits, warmup saved for counts).
Consequently Stan times are end-to-end native draw production, not isolated tree
kernel times. Process times include startup, setup, CSV and wrappers for everyone.
Stan's wrapper also combines its native CSVs into the common format.

BlackJAX/JAX CPU float64 lowers and compiles the warmup and retained scans before
timing them. Persistent compilation cache is disabled. Every phase is synchronized
with block_until_ready; host CSV conversion is outside phase timing. JIT compilation
is reported separately, plus full process wall time. JAX is the execution/AD
backend here, not a separate sampler. See
[JAX benchmarking guidance](https://docs.jax.dev/en/latest/benchmarking.html).

The first 1024 steps are **burn-in without adaptation** for Alea, Stan and BlackJAX;
nuts-rs uses native diagonal geometry adaptation with a fixed step. Its results
are a separately configured baseline, not a controlled kernel speedup comparison.
No automatic tuning or Fisher promotion is implied. Alea/nuts-rs use analytic
scores; Stan/JAX use AD. Tree stopping, RNGs and per-draw work differ.

Report minimum bulk ESS over all five predefined observables per target, divided
by summed four-chain sampling time and by sampling+burn-in time. Also report ESS
per total integrator step including burn-in. This is explicitly an edge-normalized
proxy, **not instrumented actual gradient-call counts**. Initial evaluations,
adaptation evaluations and divergences can make the latter different. All rank
Rhat/bulk/tail gates must remain visible. No dropping divergent draws.

Reproduction (use a new directory, after building both release runners):

```sh
taskset -c CPU PINNED_PYTHON tools/nuts-reference/benchmark.py target/nuts-benchmark-NEW --stan target/m5-stan-model/targets
```

The JSON records commands' input/output locations, binary/source hashes, CPU,
affinity, raw timings, diagnostic gates and aggregate medians. Cold AOT builds,
peak memory, GPU throughput, high-dimensional scaling and adaptive-policy parity
are not measured by this protocol.
