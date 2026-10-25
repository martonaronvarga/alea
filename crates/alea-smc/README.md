# alea-smc

Interfaces for particle weights, mutation, and resampling.

The crate defines contracts for composing particle algorithms. It does not supply
a particle filter or particle MCMC implementation. Use `alea-mcmc` and
`alea-runtime` for the [sampling workflows](https://github.com/martonaronvarga/alea/blob/release/0.1.1/docs/sampling.md) in the user guide.

## License

MIT OR Apache-2.0, at your option; see [LICENSE-MIT](LICENSE-MIT) and
[LICENSE-APACHE](LICENSE-APACHE). This package has no production cubature dependency.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream attribution.
