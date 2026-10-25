# Generated native Enzyme targets

This downstream package constructs models through public Alea APIs in `build.rs`,
emits Rust into `OUT_DIR`, and executes generated gradients, HVPs, Hessians and
third derivatives through existing sampler capabilities.

```console
nix develop .#autodiff --command env CARGO_TARGET_DIR=target/enzyme-integration \
  cargo test --manifest-path tools/enzyme-conformance/Cargo.toml --release --locked
```

Release/fat LTO and the matching Enzyme compiler are required. Keep this target
directory separate from stable builds. See [the native-model guide](../../docs/enzyme-native-models.md)
for integration and [the measured gradient profile](../../docs/benchmarks/enzyme-native-2026-10-22.md)
for the bounded performance evidence. `examples/profile.rs` compares the analytic
modeling tape and generated gradients with alternating order and preserved rounds.
