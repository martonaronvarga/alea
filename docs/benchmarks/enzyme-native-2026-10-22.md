# Native Enzyme model integration

Typed native models now lower ahead of time to fused Rust primals and executed
Enzyme gradients, HVPs, Hessians and third derivatives. The generated targets run
through native NUTS warmup, curvature adaptation, force-gradient HMC and SoftAbs
RMHMC. The [usage guide](../enzyme-native-models.md) explains the build workflow
and boundaries.

## Descriptive gradient costs

Intel Core i5-8250U, logical CPU 0, portable release/fat-LTO builds, Rust
1.98.0-nightly with matching Enzyme from nightly-2026-06-23. Seven serial rounds
per model, one million calls per variant per round, alternating which variant
runs first. Each phase has 64 untimed warm calls. No concurrent agent builds or
tests ran during measurement. Every round is retained and checksum agreement is
checked. Timers cover gradient evaluations, including each target adapter's checks;
construction and serialization are outside the timer.

| Two-parameter model | Analytic tape ns/call | Fused Enzyme ns/call | Median paired speedup |
| --- | ---: | ---: | ---: |
| Coupled quartic | 178.86 | 31.30 | 5.73× |
| Normal regression, 3 observations | 383.66 | 100.93 | 3.81× |
| Logistic regression, 4 observations | 497.66 | 192.70 | 2.58× |

Times are per-variant medians; the speedup is the median of paired ratios. These
are deliberately small specialized models. The improvement combines model graph
specialization, removal of dynamic callbacks/reverse tape and compiled Enzyme AD.
It does not isolate Enzyme from those other changes or compare with handwritten
fused analytic derivatives. It is not a Stan/nuts-rs comparison, an ESS/s result,
or evidence of the same speedup for large datasets or entire inference workflows.
HVP/tensor timings are not measured in this experiment.

[All 21 observations](enzyme-native-2026-10-22.csv) and the
[pre-measurement settings, source hashes and executable identity](enzyme-native-2026-10-22.json)
bind this experiment. Raw CSV SHA-256:
`479488075681a93c64b770a28c7fbac157884a218fc01bb4b5a676908b479700`.

Rebuild and run after other builds/tests finish:

```console
nix develop .#autodiff --command env CARGO_TARGET_DIR=target/enzyme-integration \
  cargo build --manifest-path tools/enzyme-conformance/Cargo.toml --example profile --release --locked
taskset -c 0 target/enzyme-integration/release/examples/profile
```

## Correctness and integration

Eight executed downstream tests cover 17 emitted models: an independent coupled
quartic oracle with nonzero third derivatives, regression/logistic, every transform
kind, 3×3 covariance/correlation charts and a 16D Gaussian. They check potential
signs, tensor ordering, finite differences, zero Rust allocations, output reset,
shape/domain failures and rounded-singular matrix rejection. They also execute
the four sampler/adaptation paths named above, including SoftAbs metric finite
differences and reverse-checked RMHMC transitions.

Stable workspace validation passes 445 tests, zero failures, one ignored; the
additional zero-dimensional exporter test passes separately. Workspace all-target
check, affected stable and generated-consumer Clippy with warnings denied,
rustdoc with warnings denied and formatting pass. The existing Enzyme CI script
also executes successfully. The API inventory contains 6,946 entries with five
API-neutral feature variants; architecture, maturity, compatibility and exactness
checks pass. No sampler default or RMHMC exactness classification changes.

Opaque Wiener/custom callbacks still need an explicit differentiable Rust primal
before export. The Enzyme compiler remains pinned and experimental; the evidence
above establishes the tested native integration, not arbitrary-function coverage.
