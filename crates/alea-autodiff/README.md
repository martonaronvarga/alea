# alea-autodiff

Derivative adapters and Enzyme source generation for Alea models.

Analytic adapters connect supplied derivatives to the model interfaces. The
`codegen::enzyme_source` exporter turns a compiled model into Rust source with
gradients, Hessian-vector products, Hessians, and third derivatives.

The exporter and analytic adapters work on stable Rust. Compiling the generated
automatic derivatives requires the pinned nightly compiler and matching Enzyme
component:

```sh
nix develop .#autodiff --command cargo test \
  --manifest-path tools/enzyme-conformance/Cargo.toml --release --locked
```

See [compiling models with Enzyme](https://github.com/martonaronvarga/alea/blob/release/0.1.1/docs/enzyme-native-models.md) for model
construction, build scripts, derivative conventions, and supported operations.
Keep Enzyme and stable builds in separate Cargo target directories.

## License

MIT OR Apache-2.0, at your option; see [LICENSE-MIT](LICENSE-MIT) and
[LICENSE-APACHE](LICENSE-APACHE). This package has no production cubature dependency.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream attribution.
