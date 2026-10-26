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
at 648 parameter/boundary combinations using high-precision direct series.
A finite table validates those points; it does not certify interpolation between
them or all possible tails. Fixed quadrature remains an explicit approximation,
and adaptive quadrature error estimates are not interval proofs.

## Simulate reaction times and choices

The same parameter types can generate joint observations. Add `rand = "0.10"`
to your application's dependencies and give prediction its own random stream:

```rust
use alea::distributions::wiener::{Boundary, Wiener4, Wiener4Params};
use rand::{SeedableRng, rngs::SmallRng};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let parameters = Wiener4Params::with_params(1.2, 0.2, 0.5, 0.7)?;
    let mut rng = SmallRng::seed_from_u64(73001);
    let mut upper = 0;
    let mut times = Vec::with_capacity(1_000);
    for _ in 0..1_000 {
        let observation = Wiener4.sample(&parameters, &mut rng)?;
        upper += usize::from(observation.boundary == Boundary::Upper);
        times.push(observation.rt);
    }
    times.sort_by(f64::total_cmp);
    println!("upper choices: {upper}/1000");
    println!("median RT: {} seconds", times[500]);
    Ok(())
}
```

`Wiener5.sample` also draws a normally distributed drift for each trial.
`Wiener7.sample` additionally draws independent starting points uniformly over
`beta ± s_beta/2` and non-decision times uniformly over `tau..tau + s_tau`.
A zero width delegates without consuming extra random numbers. Do not draw this
trial variability again outside the generator.

The generator samples centered first exits and composes them until an original
boundary is reached. It uses no simulation time step. Each exit uses bounded
rejection sampling with alternating-series acceptance tests. It returns a
numerical error if its limits are exhausted, a comparison remains unresolved,
or the resulting time cannot be represented. Do not silently discard failed
simulations and report the remainder as an unconditional predictive sample.
If a very short decision time rounds away when added to `tau`, the generator
returns an error rather than a reaction time on the excluded support boundary.
Floating-point error is not a certified bound on the generated distribution.

For a prior predictive check, draw subject/population parameters from their
priors, then generate observations. For a posterior predictive check, use each
retained draw's parameters after checking the fit. Existing-subject predictions
condition on that subject's fitted effects; new-subject predictions require new
effects from the population distribution. Keep these effects distinct from the
trial variability already sampled by Wiener5 or Wiener7.

Compare boundary proportions, reaction-time quantiles within each boundary,
and condition or subject contrasts with the observed data. Keep lower-boundary
times and any slow responses; selectively removing them changes the model.
Simulation tests cover choice probabilities, joint time tails, mean decision
times, and drift calibration with an independently derived conjugate posterior.
These are bounded checks, not calibration of every hierarchical model or of all
seven parameters jointly.

## Check a drift fit independently

For Wiener4 with fixed `alpha`, `beta`, and `tau`, the drift-dependent part of
the likelihood has a particularly useful form:

\[
p(\text{observations}\mid\delta)\;\propto\;
\exp\!\left(\delta D-\tfrac12\delta^2 T\right).
\]

Here `T` is the sum of decision times `rt - tau`. Each upper-boundary observation
adds `alpha * (1 - beta)` to `D`; each lower-boundary observation adds
`-alpha * beta`. With a normal drift prior of mean `m` and variance `s²`, the
posterior variance is `1 / (1/s² + T)` and its mean is that variance times
`m/s² + D`.

This gives an independent way to check simulation and inference without treating
another Monte Carlo run as exact truth. A normal hierarchy of subject drifts also
has a multivariate normal posterior when its population scale and the other
Wiener parameters are fixed. The downstream workflow test uses this case: it
generates two subjects' joint observations, fits a noncentered hierarchy with
four NUTS chains, checks parameter and subject-drift diagnostics against analytic
posterior means, and generates predictions for existing and new subjects.

The simplification does not apply unchanged when estimating the other Wiener
parameters or adding drift variability. Those models still need their own
identifiability, posterior recovery, and predictive checks. In particular, a
successful drift-only calibration does not establish recovery of all seven
parameters together.

## Derivatives at zero variability

The uniform non-decision interval extends from `tau` to `tau + s_tau`. Its
right-hand log-likelihood derivative at `s_tau = 0` is half the `tau` derivative,
which is generally nonzero. In contrast, the starting-point interval is symmetric
around `beta`, so its width derivative is zero at `s_beta = 0`. Widths are
constrained to be nonnegative; these are one-sided derivatives on their boundary.
