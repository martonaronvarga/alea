# Wiener reaction-time models

The Wiener likelihood describes a noisy decision process that ends when it reaches
one of two boundaries. An observation contains both the elapsed reaction time and
which boundary was reached. These models are often called drift-diffusion models.

Alea provides analytic scores for the model parameters, including the variability
parameters of Wiener5 and Wiener7. Reaction times and non-decision time must use
the same unit; the examples use seconds. The diffusion variance is fixed at one.

## Evaluate one observation

```rust
use alea::distributions::wiener::{
    Boundary, Wiener4, Wiener4Params, WienerObservation, WienerOptions,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let observation = WienerObservation {
        rt: 0.8,
        boundary: Boundary::Upper,
    };
    let parameters = Wiener4Params::with_params(1.2, 0.2, 0.5, 0.7)?;
    let evaluation = Wiener4.try_fused(&observation, &parameters, WienerOptions::default())?;
    println!("log likelihood = {}", evaluation.log_prob);
    println!("drift derivative = {}", evaluation.grad.delta);
    Ok(())
}
```

The log likelihood is for the joint observation of this time and boundary.
Do not replace lower-boundary observations by upper-boundary observations without
also accounting for the model's boundary convention.

## Choose the model

| Parameter | Meaning | Domain |
|---|---|---|
| `alpha` | Boundary separation | Positive |
| `tau` | Non-decision time | Nonnegative; below the observed time |
| `beta` | Relative starting point | Strictly between 0 and 1 |
| `delta` | Drift toward the upper boundary | Finite real |
| `s_delta` | Standard deviation of normal drift variability | Nonnegative |
| `s_beta` | Full width of uniform starting-point variability | Nonnegative; interval stays inside (0, 1) |
| `s_tau` | Width of uniform non-decision-time variability | Nonnegative |

Wiener4 uses the first four parameters. Wiener5 adds `s_delta`. Wiener7 adds
`s_beta` and `s_tau`. The starting-point interval is centered on `beta`; the
non-decision-time interval runs from `tau` to `tau + s_tau`.
The two uniform widths are not standard deviations.

Begin with the simpler model when it answers the scientific question. Extra
variability parameters can be weakly identified and make inference more expensive.

## Evaluate a batch

`WienerModel` combines observations into a model with shared parameters. Reusing
its workspace avoids repeated scratch allocation:

```rust
use alea::distributions::wiener::{
    BatchStrategy, Boundary, Wiener5, WienerModel, WienerObservation, WienerObservations,
};
use alea::model::ConstrainedModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let observations: WienerObservations = vec![
        WienerObservation {
            rt: 0.8,
            boundary: Boundary::Upper,
        },
        WienerObservation {
            rt: 1.1,
            boundary: Boundary::Lower,
        },
    ]
    .into();
    let model = WienerModel::new(
        Wiener5,
        observations.with_batch_strategy(BatchStrategy::BranchBuckets),
    )
    .with_workspace();
    let mut gradient = [0.0; 5];
    let logp = model.logp_grad(&[1.2, 0.2, 0.5, 0.7, 0.3], &mut gradient)?;
    println!("batch log likelihood = {logp}");
    Ok(())
}
```

The input order is `alpha, tau, beta, delta, s_delta`. These are constrained model
parameters. For inference, put the model behind an appropriate parameter layout
or use the Wiener factor in a typed model; do not ask HMC to step directly across
invalid separation or starting-point values.

`Direct` is the default batch strategy. `BranchBuckets` can help some large
batches, but its benefit depends on the model and observations. The cached model
is intended for one chain at a time. Construct separate instances for parallel
chains.

## Precision and extreme times

The checked methods return errors for invalid inputs and unresolved numerical
work. Wiener7 ordinarily uses adaptive cubature. In extreme tails it can switch
to locally log-scaled quadrature, avoiding underflow of the likelihood before
taking its logarithm. A finite log likelihood does not by itself establish that
all derivatives are numerically resolved.

The tolerance is a numerical policy, not a guarantee of accuracy for every
possible parameter combination. Extremely tight tolerances can fail. Fixed
Gauss-Legendre quadrature is available when you deliberately want a fixed-order
approximation; it should be checked against adaptive evaluation on your data.

Alea does not expose an exact reaction-time random sampler for these models.
The likelihood and boundary-choice prediction capabilities should not be mistaken
for a complete generative reaction-time API.

## Boundary-adjacent observations

A valid parameter can still produce an extremely small density. The log density
can remain finite even when the density itself would underflow. Keep likelihoods
in log form, and use the checked `try_fused` interface to distinguish numerical
failure from an observation outside support.

Near either starting-point boundary, Alea retains both endpoint distances and
pairs image terms before subtracting them. This avoids losing a small distance
through reflection or cancellation. The near-boundary path also tightens its
series tolerance because a small denominator amplifies error in a score ratio.
The ordinary path remains a fused value-and-score evaluation.

The numerical precision option controls series truncation and integration work.
It is not a certified absolute bound on every log-density derivative. The
extreme-support regression table independently checks values and all five scores
at 432 parameter/boundary combinations using high-precision direct series.
A finite table validates those points; it does not certify interpolation between
them or all possible tails. Fixed quadrature remains an explicit approximation,
and adaptive quadrature error estimates are not interval proofs.
