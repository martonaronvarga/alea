# M5 reference gate

Predeclared protocol: correlated Gaussian (rho .6), banana (bend .4), logistic
normal-prior posterior and centered width-one funnel, matching the existing
independent target definitions. Four chains, seeds501–504, alternating (1,-1)
and (-1,1) starts. Discard1024 initial draws; retain8192, without thinning or
divergence removal. Maximum depth8, cutoff1000. Alea/BlackJAX/Stan use fixed
identity geometry and steps .15 (first three) / .06 (funnel). Nuts-rs uses its
native1024-step geometry warmup and a fixed step of the same magnitude; this
is a distributional correctness comparison, NOT identical warmup or timing.

Pinned references: Stan2.38.0 from the workspace flake; BlackJAX1.5/JAX0.9.2 CPU
float64 from tools/blackjax-reference/default.nix; nuts-rs
a762aae513bf7bb5ccb27ffc9a938a923fc837ec. Upstream dependencies stay out of the
production workspace. No upstream trajectory code is altered.

Five observables: standardized/generative x,y,x²,y²,xy (logistic uses raw x,y).
For every backend/target/observable require rank Rhat<=1.01 and bulk/tail ESS>=400.
Compare Alea means against each reference with tolerance6*hypot(MCSEs)+.03;
also compare non-logistic observables with analytic0,0,1,1,0 using6*MCSE+.03.
MCSE is the maximum of pooled batch-mean SE at batch sizes256 and512 and the
between-chain mean SE. Failed/nonfinite diagnostics remain failures. Report all
divergences and edge counts; tree implementations and RNG streams differ, so
no bitwise/identical-trajectory or identical edge-count claim.

Separate deterministic tests use a recursive scalar oracle for the iterative
tree, with identical random tapes. Allocation tests force all edges through depth10.

## Reproduction

Use new output paths (all writers refuse to replace measured draws). From root:

```sh
nix develop .#stable --command cargo run --manifest-path crates/Cargo.toml -p alea-mcmc --release --example nuts_reference -- target/m5-alea.csv
nix develop .#stable --command python3 tools/nuts-reference/run_upstream.py target/m5-nuts-rs.csv
nix-build tools/blackjax-reference/default.nix --no-out-link
```

Use that pinned Python environment for `blackjax_run.py target/m5-blackjax.csv`
and `analyze.py` below. In restricted environments set `XDG_CACHE_HOME` and
`MPLCONFIGDIR` to writable temporary directories; do not replace HOME.

Build `nix build .#cmdStan --no-link --print-out-paths`, copy `targets.stan` into
a new target directory, then use the resulting installation's `opt/cmdstan`:
`make -C CMDSTAN/opt/cmdstan ABSOLUTE_MODEL_PATH PRECOMPILED_MODEL_HEADER=`.
The empty precompiled-header override is needed with immutable Nix installations.
Run `stan_run.py MODEL NEW_RAW_DIRECTORY target/m5-stan.csv`; it preserves all
original Stan CSVs. No adaptation, jitter, or sample filtering is enabled there.

```sh
python tools/nuts-reference/analyze.py --alea target/m5-alea.csv --stan target/m5-stan.csv --blackjax target/m5-blackjax.csv --nuts-rs target/m5-nuts-rs.csv --output target/m5-gates.json --fixture NEW_SUMMARY_FIXTURE.csv
```

The nuts-rs CSV acceptance column is -1 (unavailable through `draw`); it is not
used as an acceptance estimate. Its actual divergence and step counts are retained.
The JSON contains raw-file hashes and all failed gates. The committed summary
fixture is consumed offline by Rust integration tests; no external library or
network is required for ordinary CI. The raw-output tools are correctness
experiments by default. The separate [CPU timing protocol](BENCHMARK.md) and
`benchmark.py` enable sidecars via `ALEA_BENCH_TIMINGS`; BlackJAX uses a synchronized
runner. See the [CPU comparison](../../docs/benchmarks/nuts-cpu-2026-10-10.md) for
results and differences in geometry, AD and output timing.

## Harder CPU workloads

The [hard-target protocol](hard/PROTOCOL.md) extends the suite to dimensions
10–128, a 512-observation logistic model, and a centered width-three funnel.
It retains every coordinate and reports failed convergence gates without
filtering samples. The benchmark is serial; do not overlap builds or tests with
measurement. From the repository root, build the two Rust drivers:

```sh
nix develop .#stable --command cargo build --release --manifest-path crates/Cargo.toml -p alea-mcmc --example hard_nuts
nix develop .#stable --command env CARGO_PROFILE_RELEASE_LTO=fat cargo build --release --locked --manifest-path tools/nuts-reference/upstream/Cargo.toml --bin hard
```

The second command requires the clean pinned checkout installed by
`run_upstream.py`. Build `hard/targets.stan` using the CmdStan procedure above,
with a fresh model directory. In the pinned BlackJAX Python environment:

```sh
python tools/nuts-reference/hard/check_models.py crates/target/release/examples/hard_nuts
python -m unittest discover -s tools/nuts-reference/hard -p 'test_*.py'
taskset -c 2 python tools/nuts-reference/hard/run.py NEW_RESULTS_DIRECTORY --stan ABSOLUTE_MODEL_PATH
```

Select an available CPU for `taskset`. The runner currently uses GNU time at
`/run/current-system/sw/bin/time` (NixOS). It records all subprocess logs, native
Stan CSVs, phase/JIT/process timings, peak process RSS, hashes and diagnostics in
the new directory. Reused seeds are timing repetitions, not additional chains.
See [the harder-suite report](../../docs/benchmarks/nuts-hard-2026-10-11.md).
