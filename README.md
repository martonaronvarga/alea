# Alea

**Statistical modeling, probability computation, and inference in Rust.**

Alea is a Rust framework for building and evaluating statistical models, optimizing likelihoods, and sampling posterior distributions.

Models are defined in ordinary Rust using named parameters, probability distributions, differentiable expressions, and custom numerical factors. The same model can be used for maximum likelihood estimation, posterior sampling, prediction, and analysis.

Alea emphasizes fast repeated model evaluation, explicit numerical contracts, and reproducible statistical computation.

> **Project status:** Alea is under active development. Its native inference and numerical components are usable, while the tensor compiler, compiled differentiation, and several inference algorithms have narrower experimental qualifications. It is not yet a drop-in replacement for Stan or JAX.

## Quick start

Requires Rust 1.95 or later.

Add Alea to your application:

```toml
[dependencies]
alea = {
    git = "https://github.com/martonaronvarga/alea",
    branch = "main",
    features = ["distributions"]
}
```

For reproducible applications, commit `Cargo.lock` or pin a specific Git revision.

### Define a statistical model

The following model estimates a normal location and positive scale from four observations:

```rust
use alea::{
    distributions::modeling::ModelBuilderExt,
    model::assembly::ModelBuilder,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = ModelBuilder::new();

    let mean = builder.real("mean")?;
    let scale = builder.positive("scale")?;

    // Normal prior for the location.
    builder.normal_prior("mean prior", mean, 0.0, 2.0)?;

    // Exponential prior for the positive scale.
    let scale_log_prior = builder.multiply(-1.0, scale)?;
    builder.prior("scale prior", scale_log_prior)?;

    // Observations conditional on mean and scale.
    builder.normal_observations(
        "measurements",
        &[1.2, 0.9, 1.4, 1.1],
        mean,
        &scale,
    )?;

    let model = builder.compile()?;

    // Use the compiled model for inference or optimization.
    let _ = model;

    Ok(())
}
```

Alea handles constrained parameter coordinates and the corresponding Jacobian adjustments for sampling. Its native reverse-mode implementation computes the log density and gradient without requiring Enzyme or nightly Rust.

The [getting-started guide](docs/getting-started.md) provides a complete NUTS sampling example.

## Features

### Probability and numerical computation

- Continuous, discrete, multivariate, mixture, and directional distributions.
- Wiener drift-diffusion densities for reaction-time and choice models.
- Analytic derivatives for supported distribution families.
- Numerically stable likelihoods and explicitly checked domain conditions.
- Optional Faer and OpenBLAS linear algebra backends.

### Statistical modeling

- Named scalar and vector parameters.
- Constraints and transformations, including positive, bounded, simplex, and matrix-valued domains where supported.
- Differentiable model graphs and composable hierarchical models.
- Native Gaussian, Bernoulli-logit, and Poisson regression factors.
- Custom probability factors with supplied derivatives.
- Explicit likelihood, prior, and prediction components.

### Inference and optimization

- No-U-Turn Sampling (NUTS).
- Hamiltonian Monte Carlo and random-walk Metropolis.
- Covariance adaptation, parallel chains, diagnostics, and checkpoint recovery.
- L-BFGS optimization for likelihood and posterior-mode objectives.
- Named sampling results, prediction, and analysis exports.
- Experimental inference algorithms like RMHMC or WALNUTS with separate documented contracts.

## Execution and automatic differentiation

Alea provides multiple execution strategies rather than requiring a single compiler backend.

| Backend | Differentiation | Requirements | Status |
|---|---|---|---|
| Native model executor | Analytic scalar reverse AD and block tensor VJPs | Stable Rust | Implemented |
| Specialized probability kernels | Fused log-density and gradient evaluation | Stable Rust | Implemented for supported laws |
| Ahead-of-time compiled source | Explicit first-order rules and Enzyme-generated derivatives | Optional pinned nightly toolchain | Qualified experimental subset |
| Tensor IR execution | Fixed-shape SSA, fusion, structured regions, analytic VJPs | Stable Rust | Experimental |

Alea's tensor IR supports fixed-shape operations, broadcasting, reductions, matrix multiplication, lazy conditionals, scans, and optional checkpointed reverse passes.

The tensor system is **not yet a general optimizing tensor compiler**. It does not provide a general GPU/TPU backend, dynamic tensor shapes, or universal source compilation of arbitrary Rust callbacks.

See [Tensor IR](docs/tensor-ir.md) and [Enzyme compilation](docs/enzyme-native-models.md).

## Performance and numerical correctness

Alea's numerical implementations are tested against independent mathematical references, finite differences, and selected external implementations.

Performance work examines both individual log-density/gradient evaluations and complete inference runs.

The [native-model performance study](docs/native-model-performance.md) compares selected regression and inference workloads with CmdStan and JAX, including cases where Alea is faster, slower, or numerically different.

These results apply to specified versions, models, hardware, compiler configurations, and numerical policies. ___They don't establish universal performance superiority___.

Alea treats invalid domains, nonfinite values, derivative failures, and parameter transformation errors as explicit numerical concerns.

## Installation and optional features

The default native build uses stable Rust.

Optional features include:

| Feature | Purpose |
|---|---|
| `distributions` | Probability laws, model factors, and Wiener models |
| `analysis` | Analysis and result-export functionality |
| `faer` | Additional native linear algebra functionality |
| `openblas` | Opt-in LP64 BLAS-backed numerical operations |
| `autodiff` / `std-autodiff` | Optional source generation and compiler differentiation components |

The distribution feature also requires a C compiler.

See [installation](docs/installation.md) for supported feature combinations, compiler configuration, and source builds.

## Documentation

**Getting started**
- [Your first model](docs/getting-started.md)
- [Model construction and likelihoods](docs/modeling.md)
- [Sampling and warmup](docs/sampling.md)
- [Optimization](docs/modeling.md#maximum-likelihood-and-posterior-modes)

**Analysis and numerical methods**
- [Diagnostics](docs/diagnostics.md)
- [Sampling sessions and recovery](docs/sessions.md)
- [Distribution reference](docs/distributions.md)
- [Wiener diffusion models](docs/wiener.md)

**Compiler and development**
- [Tensor IR and lowering](docs/tensor-ir.md)
- [Compiled differentiation with Enzyme](docs/enzyme-native-models.md)
- [Performance and numerical validation](docs/native-model-performance.md)
- [Feature and compatibility matrix](docs/compatibility.md)

## Development priorities

Alea's ongoing development concentrates on:

1. Making general model composition approach specialized numerical-kernel performance.
2. Improving tensor indexing, fusion, storage planning, and backend-aware lowering.
3. Expanding mathematical and derivative conformance across execution backends.
4. Strengthening end-to-end inference validation and statistical workflows.
5. Evaluating portable accelerator backends where measurements justify them.

Experimental capabilities are documented separately from supported paths.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development, testing, and contribution guidelines.

Issues and suggestions are welcome at the [GitHub issue tracker](https://github.com/martonaronvarga/alea/issues).

The canonical development repository is [git.martonaronvarga.dev/usu/alea](https://git.martonaronvarga.dev/usu/alea); GitHub hosts a mirror.

## License

Licensing varies across the workspace.

The facade, distribution, FFI, and CLI packages use GPL-3.0-or-later. Six reusable core, numerical, inference, runtime, and AD packages use MIT OR Apache-2.0.

See [LICENSING.md](LICENSING.md) for package-specific terms and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for dependency notices.
