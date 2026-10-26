# alea-core

Model interfaces, parameter transforms, and typed model assembly for Alea.

Use `ModelBuilder` to name parameters and combine log-density factors. Implement
`LogDensityGradient` to connect an existing model to the samplers. Parameter
layouts handle constrained coordinates and their Jacobians.

```rust
use alea::model::assembly::ModelBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = ModelBuilder::new();
    let location = builder.real("location")?;
    let squared = builder.multiply(location.expr(), location.expr())?;
    let coefficient = builder.constant(-0.5)?;
    let log_density = builder.multiply(coefficient, squared)?;
    builder.factor(log_density)?;
    let target = builder.compile()?.into_target()?;
    let _ = target;
    Ok(())
}
```

This builds a standard-normal target, omitting a constant independent of the
parameter. See [models and constraints](https://github.com/martonaronvarga/alea/blob/release/0.1.1/docs/modeling.md) for priors, likelihoods,
positive parameters, and prediction. The [first example](https://github.com/martonaronvarga/alea/blob/release/0.1.1/docs/getting-started.md)
shows how to run a fit. Stable Rust 1.95 or later is sufficient.

## License

MIT OR Apache-2.0, at your option; see [LICENSE-MIT](LICENSE-MIT) and
[LICENSE-APACHE](LICENSE-APACHE). This package has no production cubature dependency.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream attribution.
