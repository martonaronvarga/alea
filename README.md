# Alea

**Bayesian modeling and inference in Rust.**

Use Alea when you want to build a statistical model and fit it from the same
Rust program. Describe your priors and likelihoods, then use posterior draws
to quantify uncertainty or optimize the model for a point estimate. Alea
aims to make models straightforward to express, efficient to fit on a CPU,
and easy to check with diagnostics. You can also bring your own log density
and gradient.

Alea is under active development. The example below runs on stable Rust.

## Fit your first model

Create a project with `cargo new my-fit` and add this to `Cargo.toml`.
You’ll need Rust 1.95 or later and a C compiler for the `distributions` feature.

```toml
[dependencies]
alea = { git = "https://github.com/martonaronvarga/alea", branch = "main", features = ["distributions"] }
```

Four measurements share an unknown mean. Give the mean a normal prior
centered at 0 with a standard deviation of 2, and assume measurement noise
with a standard deviation of 1.
Paste the program into `src/main.rs` and run `cargo run --release`.

```rust
use alea::{
    distributions::modeling::ModelBuilderExt,
    ensemble::Execution,
    model::assembly::ModelBuilder,
    sampling::adapt::MetricKind,
    session::{SessionBuilder, Status, Warmup, storage::Discard},
};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = ModelBuilder::new();
    let mean = builder.real("mean")?;
    builder.normal_prior("mean prior", mean, 0.0, 2.0)?;
    for (i, y) in [1.2, 0.9, 1.4, 1.1].into_iter().enumerate() {
        builder.normal_likelihood(&format!("measurement {i}"), y, mean, 1.0)?;
    }
    let model = Arc::new(builder.compile()?.into_executable());

    let session = SessionBuilder::new(
        "normal-mean-demo",
        1,
        vec![vec![-1.0], vec![0.0], vec![1.0], vec![2.0]],
    )
    .seed([17; 32])
    .warmup(Warmup::Covariance {
        iterations: 500,
        metric: MetricKind::Diagonal,
        target: 0.8.try_into()?,
        search: None,
    })
    .draws(1_000)
    .build()?;

    let fit = session.run(|_| model.target(), Execution::Sequential, &Discard)?;
    println!("NUTS status is {:?}", fit.status);
    if fit.status == Status::Success {
        let draws = fit.chains.iter().flat_map(|chain| &chain.draws);
        let sum: f64 = draws.clone().map(|draw| draw.position[0]).sum();
        println!("Posterior mean is {:.2}", sum / draws.count() as f64);
    }
    Ok(())
}
```

The posterior mean should be close to 1.08. Check the run status and
[diagnostics](docs/diagnostics.md) before interpreting a fit.

## More examples

- [Fit a hierarchical logistic model](crates/alea/examples/hierarchical_logistic.rs) with partial pooling across groups.
- [Supply your own log density and gradient](docs/getting-started.md) and use the same NUTS interface.
- [Fit a regression or find a posterior mode](docs/modeling.md#maximum-likelihood-and-posterior-modes).
- [Model reaction times and choices](docs/wiener.md) with the Wiener diffusion model.

See the [user guide](docs/README.md) for sampling, diagnostics and exporting draws,
or the [installation guide](docs/installation.md) for optional features.

## Contributing and license

[Contributing guide](CONTRIBUTING.md) · [Issues](https://github.com/martonaronvarga/alea/issues) · [Licenses](LICENSING.md)
