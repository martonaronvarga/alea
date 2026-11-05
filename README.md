# Alea

Alea is a framework for probability computation and statistical inference supporting Bayesian and frequentist philosophies, built around fast model evaluation and explicit numerical contracts. Describe a
statistical model in ordinary Rust, then reuse it for likelihood optimization,
posterior sampling, prediction, and analysis.

The goal is for composable models to benefit from optimized execution without
requiring model authors to write likelihood kernels or gradients by hand. Alea
combines native probability factors, differentiable model graphs, a tensor IR,
and an inference runtime. Custom densities and gradients are supported.

## Features today

- **Probability computation:** continuous, discrete, multivariate, mixture, and
  directional distributions, including Wiener drift-diffusion likelihoods for
  reaction times and choices. Laws expose density evaluation, derivatives, and
  sampling according to their supported capabilities.
- **Rust-native modeling:** `ModelBuilder`, the `model!` construction macro,
  named parameter vectors, constraints, scoped model components, and hierarchical
  models. Likelihood objectives and sampling densities handle parameter-transform Jacobians.
- **Optimized evaluation:** fused likelihood-and-gradient kernels for Gaussian,
  Bernoulli-logit, and Poisson regression; SIMD execution and specialized
  sufficient-statistics compression where applicable. Fixed-shape tensor regions
  support broadcasting, matrix products, lazy conditionals, scans, immutable data
  rebinding, restricted pointwise fusion, and opt-in checkpointed reverse passes.
- **Fitting:** local L-BFGS optimization for maximum likelihood and posterior
  modes, with named estimates and explicit stopping reasons. Optional tools
  provide observed-information calculations and fixed-design Gaussian
  coefficient intervals under their declared assumptions.
- **Monte Carlo inference:** NUTS, Hamiltonian Monte Carlo, and random-walk
  Metropolis, with covariance warmup, parallel chains, convergence diagnostics,
  posterior prediction, streaming output, and checkpoint recovery. Experimental samplers include RMHMC, Chees-HMC, WALNUTS, and more.
- **Analysis output:** named draws, sample statistics, and explicitly declared
  pointwise likelihoods and predictions. Source-checkout tools read the artifacts
  in Python and R and export NetCDF for downstream analysis with ArviZ.

Native analytic gradients and tensor VJPs work on stable Rust. Supported models
can also be exported as Rust source for ahead-of-time compilation and optional
Enzyme differentiation. Faer and LP64 OpenBLAS provide optional matrix kernels;
their explicit execution policies can change floating-point reduction order.

## State, performance, and goals

Alea is under active development. Its numerical kernels, modeling interfaces,
and research backends have different levels of qualification. It is not yet a
general-purpose replacement for Stan, JAX, or established statistical tools.

Performance is measured at both the evaluation and inference levels. The
[native-model case study](docs/native-model-performance.md) reports competitive
results on specific regression workloads, alongside slower cases, numerical
failures, fixes, and methodological limits. These are measurements of identified
source snapshots, not a universal speed ranking. Gradient throughput alone does
not establish effective sample size per second or trustworthy inference.

The development priorities are:

- **Fast composition:** bring ordinary user-built models closer to specialized
  kernel performance through dependency-aware scheduling, storage planning, and
  appropriate numerical backends in shared infrastructure.
- **Reliable mathematics:** preserve density, gradient, transform, and error
  semantics; expand model-level conformance and reference-posterior validation.
- **Reproducible inference:** make failures, diagnostics, computational work, and
  performance evidence inspectable, with useful end-to-end statistical workflows.

Important boundaries remain. Tensor fusion is currently restricted, and the IR
is not a general optimizing tensor compiler or GPU backend. Source-generated AD
supports declared primitives, not arbitrary Rust callbacks. First-order source
checkpointing has scoped tests; higher-order differentiation through that
checkpointed pullback remains unqualified. Pathfinder is available as an
initialization service, not a standalone approximate-posterior workflow.
Experimental inference methods retain their documented limits. Optimization
alone proves neither a finite/global maximum nor valid
uncertainty estimates. See the method guides for the actual contracts and limits.

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
Enable `analysis` for CSV/JSON sampling artifacts, `faer` for additional matrix
operations and Gaussian least-squares assessment, or `openblas` for opt-in LP64
BLAS-backed operations. Applications use the `alea` crate; its internal workspace
crates are implementation details.

## Documentation

Start with [your first model](docs/getting-started.md), or browse the
[user guide](docs/README.md).

- [Models, likelihoods and fitting](docs/modeling.md)
- [Maximum likelihood and posterior modes](docs/modeling.md#maximum-likelihood-and-posterior-modes)
- [Sampling and warmup](docs/sampling.md)
- [Diagnosing a fit](docs/diagnostics.md)
- [Saving and resuming a run](docs/sessions.md)
- [Subscribing to execution events](docs/tracing.md)
- [Exporting results to Python, R, and NetCDF](docs/sessions.md#export-a-named-sampling-table)
- [Probability distributions](docs/distributions.md) and [Wiener models](docs/wiener.md)
- [Tensor IR, fusion, data rebinding, and matrix kernels](docs/tensor-ir.md)
- [Compiled derivatives with Enzyme](docs/enzyme-native-models.md)
- [Native-model performance and numerical validation](docs/native-model-performance.md)
- [Compatibility, features, and supported builds](docs/releases.md)

## Contributing and support

Bug reports, questions, and suggestions are welcome on the
[issue tracker](https://github.com/martonaronvarga/alea/issues).
See [Contributing](CONTRIBUTING.md) for building and testing changes.
Report security issues privately to [contact@martonaronvarga.dev](mailto:contact@martonaronvarga.dev).

## License

The `alea` facade, distribution, FFI, and CLI packages use GPL-3.0-or-later.
Six reusable core, numerical, inference, runtime, and AD packages use
MIT OR Apache-2.0. See [Licensing](LICENSING.md) for the exact package split,
dependency obligations, and third-party notices.
