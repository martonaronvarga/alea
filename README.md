# Alea

Alea is a Rust library for probability computation and statistical inference.
Evaluate and sample probability distributions, build statistical models, and fit
them by likelihood optimization or Monte Carlo sampling. Models can use Alea's
distribution factors, differentiable expressions, or custom densities and gradients.

- **Probability computation:** continuous, discrete, multivariate, mixture, and
  directional distributions, including Wiener drift-diffusion likelihoods for
  reaction times and choices. Laws expose density evaluation, derivatives, and
  sampling according to their supported capabilities.
- **Statistical modeling and fitting:** constrained and named parameters,
  hierarchical models, batch regression likelihoods, and local L-BFGS optimization
  for maximum likelihood and posterior modes. Likelihood objectives and sampling
  densities handle parameter-transform Jacobians separately.
- **Monte Carlo inference:** NUTS, Hamiltonian Monte Carlo, and random-walk
  Metropolis, with covariance warmup, parallel chains, convergence diagnostics,
  posterior prediction, streaming output, and checkpoint recovery.
- **Analysis output:** named draws, sample statistics, and explicitly declared
  pointwise likelihoods and predictions. Source-checkout tools read the artifacts
  in Python and R and export NetCDF for downstream analysis with ArviZ.

Native model gradients work on stable Rust. Compiled Enzyme differentiation and
alternative numerical backends are optional. Experimental inference methods have
separate APIs and documented guarantees. Optimization returns a point estimate
and its stopping reason; it does not by itself establish a finite or global
maximum, or supply standard errors. See the guides for method-specific limits
and numerical validation.

## Installation

Alea requires Rust 1.95 or later. Add `alea` to `Cargo.toml`:

```toml
[dependencies]
alea = { git = "https://github.com/martonaronvarga/alea", branch = "main" }
```

Keep `Cargo.lock` with your application to preserve the resolved source revision.
The default build uses stable Rust. Add `features = ["distributions"]` for
probability laws and Wiener models; that feature also requires a C compiler.
See [installation](docs/installation.md) for source builds and optional backends.
Enable `analysis` for CSV/JSON sampling artifacts, or `faer` for additional matrix
operations and Gaussian least-squares assessment. Applications use the `alea`
crate; its internal workspace crates are implementation details.

## Documentation

Start with [your first model](docs/getting-started.md), or browse the
[user guide](docs/README.md).

- [Models, likelihoods and fitting](docs/modeling.md)
- [Maximum likelihood and posterior modes](docs/modeling.md#maximum-likelihood-and-posterior-modes)
- [Sampling and warmup](docs/sampling.md)
- [Diagnosing a fit](docs/diagnostics.md)
- [Saving and resuming a run](docs/sessions.md)
- [Exporting results to Python, R, and NetCDF](docs/sessions.md#export-a-named-sampling-table)
- [Probability distributions](docs/distributions.md) and [Wiener models](docs/wiener.md)

## Contributing and support

Bug reports, questions, and suggestions are welcome on the
[issue tracker](https://github.com/martonaronvarga/alea/issues).
See [Contributing](CONTRIBUTING.md) for building and testing changes.
Report security issues privately to [contact@martonaronvarga.dev](mailto:contact@martonaronvarga.dev).

## License

Alea uses GPL-3.0-or-later. Some internal components also offer MIT OR Apache-2.0.
See [Licensing](LICENSING.md) for package details and third-party notices.
