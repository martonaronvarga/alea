# Alea

Alea is a Rust library for Bayesian modeling and Monte Carlo inference. Build a
model from probability distributions or supply your own log density and gradient,
then sample with NUTS, Hamiltonian Monte Carlo, or other experimental samplers.

Alea provides parameter transforms, warmup, multiple chains, convergence
diagnostics, posterior prediction, and checkpoint recovery. Models and samplers
use ordinary Rust types and return errors that applications can inspect.

## Installation

Alea requires Rust 1.95 or later. Add the libraries you need to `Cargo.toml`:

```toml
[dependencies]
alea = { git = "https://github.com/martonaronvarga/alea", rev = "717053a49172a0179b8c759914ea8ac1fc9e8979" }
```

The default build uses stable Rust. Add `features = ["distributions"]` for
probability laws and Wiener models; that feature also requires a C compiler.
See [installation](docs/installation.md) for source builds and optional backends.
The crate is not currently distributed through crates.io.

## Documentation

Start with [your first model](docs/getting-started.md), or browse the
[user guide](docs/README.md).

- [Models and parameter constraints](docs/modeling.md)
- [Sampling and warmup](docs/sampling.md)
- [Diagnosing a fit](docs/diagnostics.md)
- [Saving and resuming a run](docs/sessions.md)
- [Probability distributions](docs/distributions.md) and [Wiener models](docs/wiener.md)

## Contributing and support

Bug reports, questions, and suggestions are welcome on the
[issue tracker](https://github.com/martonaronvarga/alea/issues).
See [Contributing](CONTRIBUTING.md) for building and testing changes.
Report security issues privately to [contact@martonaronvarga.dev](mailto:contact@martonaronvarga.dev).

## License

Alea uses GPL-3.0-or-later. Some internal components also offer MIT OR Apache-2.0.
See [Licensing](LICENSING.md) for package details and third-party notices.
