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
for integration. Historical measured profiles are maintainer evidence, not bundled
source-archive results. `examples/profile.rs` compares the analytic
modeling tape and generated gradients with alternating order and preserved rounds.

Native Gaussian, Bernoulli-logit and Poisson-log regression batches now export
counted-loop primals and are included in actual third-order checks. Tensor source
is separately compiled/executed in `tests/tensor_source.rs`. Default
`tests/tensor_matrix.rs` and `tests/tensor_control.rs` execute nested AD with
independent derivatives, both branches, zero-trip scans, empty-result domain
errors and recovery. A raw zero-direction/linearity regression catches the
spurious third-derivative term exposed after the original compiler failure was
fixed. The diagnostic-only `tensor-ad` feature has been removed; these checks
are no longer opt-in. Arbitrary programs and statistical applications
still require their own qualification. See
[tensor contracts and limitations](../../docs/tensor-ir.md).
