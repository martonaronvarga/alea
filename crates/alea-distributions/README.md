# alea-distributions

Probability laws, model factors, and Wiener reaction-time likelihoods for Alea.

The scalar and multivariate laws provide normalized log probabilities. Optional
traits supply sampling and derivatives. Constructors check parameter domains;
sampling uses a random-number generator supplied by the caller.

The catalogue covers continuous and count likelihoods, heavy-tailed and survival
models, ordered responses, vector and covariance priors, and directional data.
`modeling::continuous` and `modeling::observation` connect analytic distribution
scores to native models. Bingham's cached normalizer uses holonomic continuation;
the two-dimensional case uses a direct Bessel formula.

```rust
use alea::distribution::Distribution;
use alea::distributions::scalar::Normal;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let observation_model = Normal::new(0.0, 1.0)?;
    println!("log density = {}", observation_model.log_prob(&0.3)?);
    Ok(())
}
```

See [distributions](https://github.com/martonaronvarga/alea/blob/release/0.1.1/docs/distributions.md), [model factors](https://github.com/martonaronvarga/alea/blob/release/0.1.1/docs/modeling.md),
and [Wiener models](https://github.com/martonaronvarga/alea/blob/release/0.1.1/docs/wiener.md) for examples and parameter conventions.
Building this crate requires a C compiler for native cubature.

## License

GPL-3.0-or-later; see [LICENSE](LICENSE). This package includes or depends on
GPL-covered cubature; upstream terms remain intact.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream attribution.
