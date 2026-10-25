# alea-autodiff

Optional compiled derivative adapters for Alea targets.

Pre-alpha, unpublished (`publish = false`). No S3/S4 or semver stability claim.
This crate graduates independently from the reference inference crates.

`codegen::enzyme_source` lowers a typed native model ahead of time to fused Rust
and Enzyme-generated gradients, HVPs, Hessians and third derivatives. It includes
the parameter transforms and their Jacobians and feeds the existing sampler
capabilities. See [the native-model guide](https://github.com/martonaronvarga/alea/blob/main/docs/enzyme-native-models.md)
and `tools/enzyme-conformance` for a complete build-script consumer. The generated
code requires the pinned Enzyme nightly; the exporter and analytic path work on
stable Rust.

See the [release policy](https://github.com/martonaronvarga/alea/blob/main/docs/releases.md)
for platform, MSRV, feature and compatibility contracts, and the
[maturity ledger](https://github.com/martonaronvarga/alea/blob/main/docs/maturity-and-promotion.md)
for scope and outstanding evidence.

## License

MIT OR Apache-2.0, at your option; see [LICENSE-MIT](LICENSE-MIT) and
[LICENSE-APACHE](LICENSE-APACHE). This package has no production cubature dependency.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream attribution.
