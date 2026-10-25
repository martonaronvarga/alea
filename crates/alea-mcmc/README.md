# alea-mcmc

Hamiltonian Monte Carlo, NUTS, random-walk Metropolis, and warmup for Alea.

Choose checked sampler options, supply a target, and retain a coherent chain
state after every completed transition. Warmup is explicit and ends before
retained sampling.

```rust
use alea::sampling::{adapt::MetricKind, NutsOptions};
use alea::session::{Kernel, Warmup};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let kernel = Kernel::Nuts(NutsOptions::new(0.2, 8)?);
let warmup = Warmup::Covariance {
    iterations: 1_000,
    metric: MetricKind::Diagonal,
    target: 0.8.try_into()?,
    search: None,
};
# let _ = (kernel, warmup);
# Ok(())
# }
```

Use these settings with an `alea-runtime` session, or construct a chain directly.
See [sampling](https://github.com/martonaronvarga/alea/blob/release/0.1.1/docs/sampling.md) for the choices and
[your first model](https://github.com/martonaronvarga/alea/blob/release/0.1.1/docs/getting-started.md) for a complete program.

Specialized methods are available in named modules. Read their numerical
assumptions before use. `ApproximateMclmc` and `ExactnessPendingRmhmc` have different
guarantees from ordinary corrected Euclidean HMC. Optional backends are described
in [installation](https://github.com/martonaronvarga/alea/blob/release/0.1.1/docs/installation.md).

## License

MIT OR Apache-2.0, at your option; see [LICENSE-MIT](LICENSE-MIT) and
[LICENSE-APACHE](LICENSE-APACHE). This package has no production cubature dependency.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream attribution.
