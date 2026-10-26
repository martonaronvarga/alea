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

### Controlling series truncation

`try_fused_with_truncation` is an optional Wiener4/Wiener5 evaluator for work
that needs an explicit truncation budget. The tolerance is an absolute budget
for the log density and each parameter score, in `grad_array()` order.

```rust
use alea::distributions::wiener::{Boundary, Wiener4, Wiener4Params, WienerObservation};
let observation = WienerObservation { rt: 0.8, boundary: Boundary::Upper };
let (evaluation, error) = Wiener4.try_fused_with_truncation(
    &observation, &Wiener4Params::default(), 1e-9,
)?;
assert!(evaluation.log_prob.is_finite());
assert!(error.log_prob <= 1e-9);
assert!(error.scores.iter().all(|bound| *bound <= 1e-9));
# Ok::<(), alea::distributions::wiener::ProbError>(())
```

The report concerns truncation in real arithmetic. Floating-point rounding,
including errors in exponential and trigonometric functions, is excluded.
It is not an end-to-end numerical certificate. The ordinary evaluator keeps
its existing precision convention and avoids the extra planning work.

Both representations use a common term count for the density and its two
normalized derivatives. For a Gaussian moment sequence, the ratio of successive
terms decreases. Once that ratio is below one, the first omitted term divided
by one minus the ratio bounds the whole tail. The evaluator takes the smaller
of this geometric bound and a valid decreasing-function integral bound.
Spectral density and time-derivative tails additionally use
`|sin(nπw)| ≤ n|sin(πw)|`, which helps near absorbing boundaries. Image derivative
bounds include the time-dependent density prefactor, not just the exponential.

The density lower bound must be positive before converting these tails to
log-density and score errors. The search rejects evaluations that cannot meet
the budget within 128 harmonics or image pairs. Branch choice compares an
explicit work estimate of two exponentials per image pair and one spectral
harmonic. This is a term-work estimate; it does not establish minimum latency.

An independent Arb ball-arithmetic fixture checks values and all Wiener5 scores
across both boundaries, small and large times, endpoint distances down to
`1e-12`, and zero/nonzero drift variability. The oracle encloses its arithmetic
and omitted tails and increases precision until the resulting balls are narrow
enough. Rust comparisons include a separate floating-point allowance. This
fixture covers Wiener4/5 series; it does not certify Wiener7 quadrature.

Wiener7 differentiates its fixed unit-cube integrand with respect to both uniform
widths. This avoids subtracting endpoint terms of size `1 / width` to obtain a
small score. Fixed Gauss–Legendre gradients differentiate the selected rule,
including at order one. Adaptive integration still provides estimated
quadrature errors rather than rigorous integration enclosures.

For a Wiener7 error breakdown, use `try_fused_with_error_estimate` with adaptive
quadrature. It returns the evaluation and separate series, quadrature, and
combined estimates for the log density and all seven scores. Pointwise series
envelopes are integrated alongside the density and derivatives. Their numerical
integration errors are added to the series contribution before forming the
combined estimate. The density must remain separated from zero under that
estimate. The method returns an error if the requested combined budget is not
met. Log-density error uses an absolute tolerance; each score uses the tolerance
times the larger of one and the absolute score.

This diagnostic method does extra work and is useful for checking demanding
parameter regimes. It does not provide rigorous continuous cubature enclosures
or assess roundoff. Fixed quadrature is rejected because it has no embedded
error estimator. Ordinary adaptive evaluation now also checks the cubature's
returned component errors in log-density and score units before accepting its
natural-scale result; otherwise it tries the scaled adaptive path.

When only starting-point variability is active, drift variability is zero, and
normalized decision time is at least `0.15`, adaptive evaluation can average the
spectral series analytically. For interval center `m`, width `h`, and
`c = alpha * drift + i * n * π`, the averaged exponential is
`exp(c * m) * sinh(c * h / 2) / (c * h / 2)`. Its imaginary part gives the averaged
sine term. Differentiating the same expression supplies all seven scores. A
Taylor evaluation of `sinh(z) / z` and its derivative handles narrow intervals.
Gaussian-moment tail bounds control the integrated density and derivatives.

This path avoids numerical quadrature in its supported region. Large drift
variation across the interval, unsuccessful truncation checks, and other
variability configurations retain cubature. An explicitly requested fixed rule
still evaluates that rule. Gaussian drift variability instead uses the image-moment method described below.

### Cumulative and interval probabilities

Wiener4 provides `try_cdf`, `try_log_cdf`, `try_log_survival`, and
`try_log_interval_prob`. These describe a **joint boundary subdistribution**:
`cdf(rt, boundary)` is the probability of reaching that boundary by the stated
reaction time. Its limit is the probability of choosing that boundary, which
can be less than one. Likewise, the boundary survival probability is
`P(T > rt, B = boundary)`, rather than `1 - cdf(rt, boundary)`.

```rust
use alea::distributions::wiener::{Boundary, Wiener4, Wiener4Params, WienerObservation, WienerOptions};
let p = Wiener4Params::default();
let options = WienerOptions::new(1e-10);
let deadline = WienerObservation { rt: 0.8, boundary: Boundary::Upper };
let probability = Wiener4.try_cdf(&deadline, &p, options)?;
let log_interval = Wiener4.try_log_interval_prob(0.5, 0.8, Boundary::Upper, &p, options)?;
assert!((0.0..=1.0).contains(&probability));
assert!(log_interval <= 0.0);
# Ok::<(), alea::distributions::wiener::ProbError>(())
```

The evaluator uses integrated image terms at short times and an integrated
spectral tail at longer times. Mills ratios avoid underflow in normal tails.
Geometric remainder bounds control series truncation in log-probability units;
they exclude roundoff. The work limit is 128 terms. Unresolved cancellation,
very large scaled drift, and unsupported numerical regimes return an error.

Interval evaluation chooses a well-separated CDF or boundary-tail difference.
It tightens the inner truncation budget before subtraction. Very narrow
intervals can return a numerical error. No negative interval probability is
silently clamped to zero.

For Wiener7 with only nondecision-time variability, density-only adaptive
evaluation can use this interval probability divided by the time width. If
the difference is too narrow or the series cannot meet its numerical checks,
it retains cubature. The fused path uses analytic CDF parameter scores and an endpoint density for
the time-width score, with a separate cancellation check. Fixed
quadrature still evaluates the requested rule. The CDF functions do not yet
marginalize Gaussian drift or starting-point variability.

The short-time branch can integrate starting-point variability even when drift
variability is nonzero. It evaluates four Gaussian moments per image and
analytically differentiates their combination. A controlled Taylor branch
handles narrow intervals; a scaled normal-probability formula handles other
supported cases. Conditioning and truncation checks retain cubature when
needed. With both uniform widths active, supported start averages reduce
adaptive integration to one time dimension. Clipped time support retains the
ordinary evaluator. See [the numerical methods](wiener-numerics.md) for the
formulas, bounds, and scope of the independent ball-arithmetic checks.

`try_fused_cdf`, `try_fused_survival`, and `try_fused_interval` return the log
probability and analytic parameter scores in ordinary Wiener4 order. The
infinite-time CDF gives the choice probability and its scores. A zero CDF or
zero boundary tail has no finite log score and returns a support error.

For a rigorous check of a specific numerical result, the optional source tool
`tests/support/wiener7_reference.py` encloses the mathematical density and all
seven scores, increases ball precision when needed, and compares the exact
binary64 candidate with those enclosures. It reports certified, rejected, or
unresolved; reference work exhaustion is explicit. This pointwise certificate
covers all numerical error sources together, including floating-point and
special-function evaluation. It is not a global error theorem for every Rust
execution. Ordinary library evaluation does not invoke Python or ball arithmetic.
