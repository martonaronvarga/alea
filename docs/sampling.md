# Sampling and warmup

A Monte Carlo chain produces a sequence of parameter values whose long-run
distribution is the target posterior. Nearby draws are usually correlated. Run
multiple chains, inspect diagnostics, and estimate the uncertainty of the summary
you actually need.

## Choose a sampler

| Sampler | Needs gradients? | When to consider it |
|---|---|---|
| NUTS | Yes | A starting point for differentiable continuous models |
| Fixed-trajectory HMC | Yes | When you want direct control over trajectory length |
| Random-walk Metropolis | No | When gradients are unavailable or a simple comparison is useful |

NUTS chooses its trajectory length while exploring. Its maximum tree depth limits
work per transition. Fixed HMC instead takes a chosen number of integration steps.
Metropolis makes local random-walk proposals, which can mix slowly when parameters
are strongly correlated.

A session selects the kernel with its corresponding checked options:

```rust
use alea::sampling::{HmcOptions, NutsOptions, RwmhOptions};
use alea::session::Kernel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let nuts = Kernel::Nuts(NutsOptions::new(0.2, 8)?);
    let hmc = Kernel::Hmc(HmcOptions::new(0.1, 10)?);
    let metropolis = Kernel::Rwmh(RwmhOptions::new(0.5)?);
    let _ = (nuts, hmc, metropolis);
    Ok(())
}
```

For NUTS, `0.2` is the initial step size and `8` is the maximum tree depth. A depth
of 8 allows up to 255 integration steps in the tree. For HMC, `0.1` is the step
size and `10` is the number of steps. These are illustrative starting values,
not settings that suit every model.

Use `Session::run` for a target with gradients. Use `Session::run_density` with a
`LogDensity` target for gradient-free Metropolis. The typed interfaces make that
difference explicit.

## Request warmup

Warmup adjusts numerical settings before collecting draws. Covariance warmup
learns a metric and tunes the step size toward a requested acceptance rate:

```rust
use alea::sampling::adapt::MetricKind;
use alea::session::Warmup;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let warmup = Warmup::Covariance {
        iterations: 1_000,
        metric: MetricKind::Diagonal,
        target: 0.8.try_into()?,
        search: None,
    };
    let _ = warmup;
    Ok(())
}
```

Pass this value to `.warmup(warmup)` on the session builder. Warmup draws are
discarded. Adaptation is frozen before any retained draw reaches a sink or the
collected output. Without an explicit warmup choice, the session starts sampling
with the supplied step size and geometry.

A diagonal metric learns a separate scale for each coordinate. A dense metric can
also represent linear correlations, at the cost of quadratic storage and more
expensive operations. Reparameterization is often more effective than increasing
the complexity of the metric.

A higher acceptance target usually leads to smaller integration steps. It can
help with divergences, but it does not repair a wrong gradient or a poorly
identified model. See [Troubleshooting](troubleshooting.md).

## Start several chains

Give `SessionBuilder` one unconstrained initial position per chain. Prefer
plausible, dispersed values to identical extreme values. A positive model
parameter is initialized on its log scale; an initial sampling coordinate of zero
therefore means a positive parameter value of one.

Use a reproducible 32-byte seed, such as `.seed([17; 32])`, and record it with your
model and data. Chain identities are derived from their input order, so switching
from sequential execution to a caller-owned parallel pool preserves each logical
chain's random stream under the same numerical build.

## Work with draws

A retained draw records the chain identity, draw index, position, and common
transition statistics. Rejections are represented by repeated positions. Keep
those repeats when computing means, intervals, or effective sample sizes.

Session positions are unconstrained coordinates. Convert them with the model's
layout before reporting constrained parameters. A raw log-scale draw is not a draw
of the corresponding positive scale.

## Other methods

Alea also exposes ChEES-HMC, persistent and delayed-rejection GHMC, MAMS, WALNUTS,
and specialized integration and geometry methods through their named modules.
Their API documentation states the applicable assumptions and limitations.

`ApproximateMclmc` omits a Metropolis correction and should not be interpreted as
an exact-target sampler. `ExactnessPendingRmhmc` has unresolved numerical-map
reversibility and volume obligations. These names describe material differences
in the methods, not alternative names for ordinary Euclidean HMC.
