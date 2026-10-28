# Alea

Bayesian modeling and Monte Carlo inference in Rust. Build a model or supply a
log density and gradient, then run NUTS, Hamiltonian Monte Carlo, or Metropolis.
Alea provides parameter transforms, warmup, parallel chains, diagnostics,
posterior prediction, and checkpoint recovery.

```rust
use alea::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let kernel = SamplerConfig::Nuts(NutsOptions::new(0.2, 8)?);
    let warmup = Warmup::Covariance {
        iterations: 1_000,
        metric: alea::sampling::adapt::MetricKind::Diagonal,
        target: 0.8.try_into()?,
        search: None,
    };
    let _ = (kernel, warmup);
    Ok(())
}
```

See the [user guide](https://github.com/martonaronvarga/alea/blob/release/0.1.1/docs/README.md)
for complete programs and advice on interpreting a fit. Add the `distributions`
feature for probability laws and Wiener models; that feature requires a C compiler.
Other optional backends are described in the installation guide.

## License

GPL-3.0-or-later; see [LICENSE](https://github.com/martonaronvarga/alea/blob/release/0.1.1/LICENSE-GPL) and
[THIRD_PARTY_NOTICES.md](https://github.com/martonaronvarga/alea/blob/release/0.1.1/THIRD_PARTY_NOTICES.md).
