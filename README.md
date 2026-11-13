# Alea

**Probability distributions, differentiable statistical models, and CPU inference in Rust.**

Alea is a modular Rust library for probability computation, statistical model construction, numerical optimization, and Monte Carlo inference. You can assemble models in ordinary Rust or supply your own log-density target to a sampler. There is no separate modeling language.

The primary execution path uses native Rust on the CPU: analytic distribution scores, reverse-mode model differentiation, specialized regression kernels, and reusable evaluation workspaces. Stable Rust is sufficient for this path. Compiler-generated differentiation through Enzyme is optional and qualified for a narrower subset of models and derivative operations.

> **Project status:** Alea is under active development, currently at version 0.1.1. APIs and numerical contracts are evolving. Native capabilities, experimental algorithms, and compiler differentiation have different qualification boundaries; availability of an API does not establish support for every model or derivative order.

## Quick start

Requires Rust 1.95 or later. The optional `distributions` feature also requires a C compiler to build the bundled cubature dependency.

Alea is installed from source. Add it to your application:

```toml
[dependencies.alea]
git = "https://github.com/martonaronvarga/alea"
branch = "main"
features = ["distributions"]
```

For reproducible applications, commit `Cargo.lock` or pin a specific Git revision.

This model combines normal observations with a normal prior on the mean and an exponential prior on a positive scale. It evaluates the sampling log density and its gradient in unconstrained coordinates:

```rust
use alea::{
    distributions::modeling::ModelBuilderExt,
    model::assembly::ModelBuilder,
    target::LogDensityGradient,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = ModelBuilder::new();
    let mean = builder.real("mean")?;
    let scale = builder.positive("scale")?;

    builder.normal_prior("mean prior", mean, 0.0, 2.0)?;
    let scale_log_prior = builder.multiply(-1.0, scale)?;
    builder.prior("scale prior", scale_log_prior)?;
    builder.normal_observations(
        "measurements",
        &[1.2, 0.9, 1.4, 1.1],
        mean,
        &scale,
    )?;

    let target = builder.compile()?.into_target()?;
    let mut gradient = [0.0; 2];
    // Coordinates are [mean, log(scale)]; here scale = 1.
    let logp = target.logp_grad(&[1.0, 0.0], &mut gradient)?;
    println!("log density: {logp}, gradient: {gradient:?}");
    Ok(())
}
```

The sampling target includes the Jacobian adjustment for the scale transformation. Native gradients require neither Enzyme nor nightly Rust. The [getting-started guide](docs/getting-started.md) provides a complete NUTS sampling example, including a custom target.

## What you can do today

### Probability and numerical computation

- Evaluate continuous, discrete, multivariate, mixture, and directional distributions, with analytic derivatives for supported families.
- Compute Wiener drift-diffusion densities for reaction-time and choice models.
- Use checked numerical routines, compensated arithmetic, and optional Faer or LP64 OpenBLAS matrix operations.

### Model construction

- Define named scalar and vector parameters, with positive, bounded, simplex, and matrix-domain transformations where supported.
- Compose likelihoods, priors, hierarchical models, and custom factors with supplied derivative rules.
- Use specialized Gaussian, Bernoulli-logit, and Poisson regression factors.
- Share fixed-shape vector and matrix expressions across observation blocks. Native Normal, Bernoulli-logit, and Poisson-log observation adapters accept shared predictors, including nonlinear predictors, and propagate cached scores through the same graph.
- Build fixed-shape tensor expressions with broadcasting, reductions, matrix multiplication, lazy conditionals, scans, and supported checkpointed reverse passes.

Model definitions, data bindings, compiled executables, and evaluation contexts are separate parts of the model lifecycle. Evaluation contexts own mutable workspaces so repeated evaluations and independent chains can reuse storage without sharing mutable chain state.

### Inference, optimization, and results

- Run NUTS, Hamiltonian Monte Carlo, or random-walk Metropolis through sampling sessions.
- Configure warmup, covariance adaptation, parallel chains, diagnostics, and checkpoint recovery.
- Optimize likelihood and posterior-mode objectives with L-BFGS.
- Work with named draws, explicit prediction and pointwise reporting components, and optional CSV/JSON analysis exports.

Objectives and sampling densities have different meanings. Maximum likelihood uses likelihood factors; declared-measure MAP uses likelihood and prior factors. A transformed sampling density additionally includes parameter-coordinate Jacobians. An optimizer reaching its numerical stopping condition does not by itself establish that a finite estimator exists or that uncertainty estimates are valid. See [optimization and posterior modes](docs/modeling.md#maximum-likelihood-and-posterior-modes).

## Execution and qualification boundaries

| Path | What it provides | Boundary |
|---|---|---|
| Native model executor | Scalar reverse-mode differentiation and block vector–Jacobian products | Stable Rust; supported operations and supplied rules |
| Specialized probability kernels | Fused density and score evaluation, including regression kernels | Supported laws and their explicit numerical policies |
| Source generation and compiled differentiation | Ahead-of-time source and explicitly requested derivative operations | Eligible model subset; Enzyme execution requires the matching pinned toolchain |

The fixed-shape tensor representation is part of native model execution and source lowering, not a general accelerator compiler. Alea does not currently provide a general GPU/TPU backend, dynamic tensor shapes, or universal differentiation of arbitrary Rust callbacks.

Native first-order support does not imply source compilation, Jacobian-vector products, Hessian-vector products, third derivatives, or replay support. In particular, the shared observation adapters above are qualified for native first-order evaluation; their source and higher-order paths are not available.

Additional samplers have separate experimental contracts. Examples include RMHMC, whose exactness qualification remains pending, WALNUTS, and explicitly approximate MCLMC. They are not interchangeable with the ordinary session samplers. The `alea-smc` package currently defines particle-operation traits; filtering and particle MCMC are not implemented. The CLI contains experiment and benchmark utilities rather than a general model frontend.

See [model construction](docs/modeling.md), [Tensor IR](docs/tensor-ir.md), [sampling](docs/sampling.md), and [Enzyme compilation](docs/enzyme-native-models.md) for operation-specific contracts.

## Installation and optional features

The facade has no default features. Its native model, sampler, and runtime interfaces work on stable Rust without Enzyme or BLAS; distribution implementations are opt-in.

| Feature | Purpose and requirements |
|---|---|
| `distributions` | Probability laws, model factors, and Wiener models; requires a C compiler |
| `analysis` | Structured analysis and CSV/JSON result exports |
| `tracing` | Runtime and enabled-component tracing events |
| `faer` | Optional native matrix operations and additional linear algebra routines |
| `openblas` | Optional LP64 BLAS operations; requires compatible OpenBLAS libraries |
| `autodiff` | Source generation and derivative-checking tools |
| `std-autodiff` | Enzyme compiler differentiation; requires the documented pinned nightly toolchain |
| `simd` | Additional SIMD paths; requires nightly Rust |
| `branch-hints` | Distribution branch hints; requires nightly Rust |

Enabling a matrix provider does not silently replace the default compensated matrix arithmetic. Standard arithmetic and provider/thread policies are explicit choices with separate numerical contracts.

See [installation](docs/installation.md) and [supported builds](docs/releases.md) for toolchains, feature combinations, and platform requirements.

## Performance and numerical validation

Alea uses independent mathematical references, finite differences, and selected external implementations to check supported numerical paths. Derivative claims are specific to the operation, backend, and derivative order that was exercised.

The [native-model performance study](docs/native-model-performance.md) records selected regression and inference comparisons with CmdStan and JAX, including correctness issues, repairs, and workloads where results differ. Its timings apply to the recorded models, hardware, source revisions, compiler settings, and numerical policies. They do not establish a universal speed ranking or equal statistical value for equal draw counts.

## Workspace

Most users start with the `alea` facade. The workspace also separates reusable components:

| Package | Role |
|---|---|
| `alea-math` | Numerical routines, matrix operations, metrics, and aligned storage |
| `alea-core` | Distribution and target contracts, transforms, and model execution |
| `alea-distributions` | Probability laws, observation/regression factors, and Wiener models |
| `alea-mcmc` | Markov kernels and sampler implementations |
| `alea-runtime` | Sessions, warmup, diagnostics, optimization, outputs, and recovery |
| `alea-autodiff` | Source generation, compiler differentiation, and gradient checks |
| `alea-smc` | Particle-operation contracts |
| `alea-ffi` | Native dependency bindings |
| `alea-cli` | Experiment and benchmark utilities |

## Documentation

- [Documentation index](docs/README.md)
- [Getting started](docs/getting-started.md)
- [Model construction and likelihoods](docs/modeling.md)
- [Hierarchical models](docs/hierarchical-models.md)
- [Sampling and warmup](docs/sampling.md)
- [Diagnostics](docs/diagnostics.md)
- [Sampling sessions and recovery](docs/sessions.md)
- [Distribution reference](docs/distributions.md)
- [Wiener diffusion models](docs/wiener.md)
- [Tensor IR and lowering](docs/tensor-ir.md)
- [Compiled differentiation with Enzyme](docs/enzyme-native-models.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development, testing, and contribution guidelines.

Issues and suggestions are welcome at the [GitHub issue tracker](https://github.com/martonaronvarga/alea/issues).

The canonical development repository is [git.martonaronvarga.dev/usu/alea](https://git.martonaronvarga.dev/usu/alea); GitHub hosts a mirror.

## License

Licensing varies across the workspace. The facade, distribution, FFI, and CLI packages use GPL-3.0-or-later. Six reusable core, numerical, inference, runtime, AD, and particle-contract packages use MIT OR Apache-2.0.

See [LICENSING.md](LICENSING.md) for package-specific terms and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for dependency notices.
