# Probability distributions

The `distributions` feature provides normalized log densities, probability masses,
sampling, and derivatives. Constructors validate parameters, and optional traits
make the capabilities of each law explicit.

## Evaluate and sample

```rust
use alea::distribution::{ScalarEventScore, Distribution, Sample};
use alea::distributions::scalar::Normal;
use rand::{SeedableRng, rngs::SmallRng};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let law = Normal::new(2.0, 0.5)?;
    let observation = 2.3;
    println!("log density = {}", law.log_prob(&observation)?);
    println!(
        "derivative with respect to the observation = {}",
        law.event_score(observation)?
    );
    let mut rng = SmallRng::seed_from_u64(123);
    println!("simulated observation = {}", law.sample(&mut rng)?);
    Ok(())
}
```

`Normal::new` takes a mean and a standard deviation. Its `log_prob` includes the
normalizing constant. This differs from `alea::distributions::StandardNormalTarget`, which is
an unnormalized multivariate standard-normal target for inference examples.

Sampling uses a random-number generator supplied by you. Record its seed when
reproducible simulated data or predictions matter.

## Available laws

| Module | Laws | Constructor parameters, in order |
|---|---|---|
| `scalar` | Normal, Cauchy, Laplace, Logistic, Gumbel | Location, scale |
| `scalar` | StudentT | Degrees of freedom, location, scale |
| `scalar` | SkewNormal | Location, scale, skew |
| `scalar` | ExpModNormal | Normal location, normal scale, exponential rate |
| `scalar` | NormalInverseGaussian | Alpha, beta, location, delta; alpha > abs(beta) |
| `scalar` | LogNormal | Mean and standard deviation of log X |
| `scalar` | Exponential | Rate |
| `scalar` | Gamma, InverseGamma | Shape, rate; for InverseGamma this is the rate of 1/X |
| `scalar` | ChiSquared, InverseChiSquared | Degrees of freedom |
| `scalar` | ScaledInverseChiSquared | Degrees of freedom, scale; X = df * scale² / chi²(df) |
| `scalar` | FisherF | Numerator and denominator degrees of freedom |
| `scalar` | Weibull, LogLogistic | Shape, scale |
| `scalar` | Frechet | Location, scale, shape |
| `scalar` | GeneralizedExtremeValue | Location, scale, shape; zero shape is Gumbel |
| `scalar` | Pareto | Positive minimum, shape |
| `scalar` | ParetoType2 | Location, scale, shape; location zero gives Lomax |
| `scalar` | InverseGaussian | Mean, shape |
| `scalar` | Levy | Location, scale |
| `scalar` | HalfNormal, HalfCauchy, Rayleigh | Scale |
| `scalar` | Uniform | Lower, upper |
| `scalar` | Triangular | Lower, upper, mode |
| `scalar` | Pert | Lower, upper, mode, weight; weight 4 gives conventional beta-PERT |
| `scalar` | Beta | Alpha, beta |
| `scalar` | BetaProportion | Mean, precision; alpha = mean * precision |
| `scalar` | VonMises | Mean angle, concentration; angles in radians |
| `scalar` | Bernoulli, BernoulliLogit | Success probability or log odds; boolean events |
| `scalar` | Binomial, BinomialLogit | Trials, success probability or log odds |
| `scalar` | Poisson, PoissonLog | Rate or log rate |
| `scalar` | Geometric | Success probability; counts **failures**, starting at zero |
| `scalar` | NegativeBinomial | Shape, success probability; mean = shape * (1-p)/p |
| `scalar` | NegativeBinomialMean | Mean, inverse dispersion; variance = mean + mean²/dispersion |
| `scalar` | BetaBinomial | Trials, alpha, beta |
| `scalar` | BetaNegativeBinomial | Shape, alpha, beta |
| `scalar` | Hypergeometric | Population, successes in population, draws |
| `scalar` | DiscreteUniform | Inclusive signed integer bounds |
| `scalar` | PoissonBinomial | Vector of independent trial probabilities |
| `scalar` | Zipf | Finite support size, exponent; support starts at one |
| `scalar` | Zeta | Exponent greater than one; infinite positive-integer support |
| `scalar` | OrderedLogistic, OrderedProbit | Location, increasing cutpoints; zero-based categories |
| `multivariate` | MultivariateNormal | Mean vector, lower covariance Cholesky factor |
| `multivariate` | MultivariateStudentT | Degrees of freedom, mean vector, lower **scale** Cholesky factor |
| `multivariate` | Dirichlet, Categorical, Multinomial | Concentrations, probabilities, or trials and probabilities |
| `multivariate` | DirichletMultinomial | Trials, concentration vector |
| `multivariate` | Wishart, InverseWishart | Degrees of freedom, dimension, lower scale Cholesky factor |
| `multivariate` | Lkj, LkjCholesky | Dimension, concentration; different reference measures |
| `directional` | Bingham | Principal-axis concentrations; unit-vector events |
| `directional` | UniformSphere, UniformBall | Dimension; surface-area or volume density |
| `mixture` | FiniteMixture | Component laws and mixture weights |
| `wiener` | Wiener4, Wiener5, Wiener7 | Reaction time and boundary-choice likelihoods |

Consult each type's documentation for its constructor order and supported
sampling range. A rate is the reciprocal of a scale. Mixing up those conventions
can produce a plausible-looking but incorrect model.

## Discrete observations

Discrete laws use boolean or integer events. They provide probability masses,
not continuous densities:

```rust
use alea::distribution::Distribution;
use alea::distributions::scalar::{Bernoulli, Poisson};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let success = Bernoulli::new(0.7)?;
    assert!((success.log_prob(&true)?.exp() - 0.7).abs() < 1e-12);
    let counts = Poisson::new(3.0)?;
    println!("P(count = 2) = {}", counts.log_prob(&2)?.exp());
    Ok(())
}
```

There is no derivative with respect to a boolean or count observation. Continuous
parameters of those laws can still have scores. In models with latent categories,
marginalize the category when possible, for example with a finite mixture.

## Vector observations

Multivariate normal covariance is supplied as a full row-major lower Cholesky
matrix. This example uses independent coordinates with standard deviations 1 and 2:

```rust
use alea::distribution::Distribution;
use alea::distributions::multivariate::MultivariateNormal;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let law = MultivariateNormal::new(vec![0.0, 0.0], vec![1.0, 0.0, 0.0, 2.0])?;
    println!("log density = {}", law.log_prob(&[0.2, -0.5])?);
    Ok(())
}
```

Simplex scores are expressed in ambient coordinates; a simplex transform handles
the pullback to sampling coordinates. LKJ-Cholesky is a density on the free lower
Cholesky coordinates, with the diagonal determined by row norms. Its density is
not generally constant when its concentration parameter is 1.

## Support and numerical errors

An observation outside a law's support can have log probability negative infinity.
Invalid parameters, NaNs, dimension mismatches, and numerical failures return
errors. A density may have a well-defined boundary value while its derivative is
undefined, so successful evaluation does not imply every derivative is available.

Sampling has explicit work and storage limits. Poisson sampling supports rates
up to 30,000. Negative-binomial sampling uses a gamma–Poisson mixture and inherits
that limit on the realized rate; it can return a budget error even when its
log mass is available. Binomial and Multinomial sampling support at most one
million nontrivial trials. Hypergeometric sampling supports at most one million
draws. Degenerate Binomial laws and zero-rate Poisson laws return their fixed
outcome directly.

Zipf caches at most one million probabilities. PoissonBinomial caches a
log-space convolution for up to 4096 trials. Its probability-vector scores use
positive prefix/suffix convolutions, with temporary storage proportional to
`trials * (1 + min(count, trials-count))`. Zeta sampling can produce counts beyond the
exact binary64 integer range and reports those cases rather than rounding them.
VonMises accepts concentration up to 1e8. See the
[directional guide](directional-distributions.md) for Bingham's HGM domain and
error estimates. Sampling limits do not silently narrow a law's mathematical
support.

All named scalar laws provide parameter and event scores where continuous
parameters and differentiable events exist. Hypergeometric and DiscreteUniform
have only fixed integer parameters. UniformSphere and UniformBall have
no continuous distribution parameters. Multivariate and directional scores have
explicit coordinate conventions in their type documentation.

General CDF, quantile, and truncation interfaces are not provided for this entire
catalogue. The Wiener module has its own boundary-specific CDF and interval
methods. A normalized density and an RNG do not imply a censoring likelihood is
available.

To use these laws as priors or likelihoods, see [Models and constraints](modeling.md).


## Use a new law in a model

`modeling::continuous` includes the derivative of a scalar event, so it can be
used for a prior on a parameter. `modeling::observation` holds data fixed and
uses the law's parameter scores. The constructor closure receives parameter
values in the same order as the expression slice. Fixed parameters belong in
that closure rather than in the score slice.

```rust
use alea::distributions::modeling::{continuous, observation};
use alea::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = ModelBuilder::new();
    let mean = builder.positive("mean")?;
    let dispersion = builder.positive("dispersion")?;
    let scale = builder.constant(5.0)?;
    let prior = continuous(&mut builder, mean.expr(), &[scale], |p| {
        HalfNormal::new(p[0])
    })?;
    builder.factor(prior)?;
    let dispersion_prior = continuous(&mut builder, dispersion.expr(), &[scale], |p| {
        HalfNormal::new(p[0])
    })?;
    builder.factor(dispersion_prior)?;
    let likelihood = observation(
        &mut builder, 4_u64, &[mean.expr(), dispersion.expr()],
        |p| NegativeBinomialMean::new(p[0], p[1]),
    )?;
    builder.factor(likelihood)?;
    let target = builder.compile()?.into_target()?;
    let mut gradient = [0.0; 2];
    println!("log density = {}", target.logp_grad(&[1.0, 0.0], &mut gradient)?);
    Ok(())
}
```

These generic factors are analytic callbacks. They support ordinary model
execution and reverse scores, but need a custom scalar primal for Enzyme source
export. For repeated direct evaluations at fixed parameters, construct a law
once and reuse it; this is particularly useful for Bingham's cached normalizer.

## Reference conventions

The catalogue was compared with the documented families in
[rand_distr](https://docs.rs/rand_distr/latest/rand_distr/index.html) and the
[Stan function reference](https://mc-stan.org/docs/functions-reference/index.html).
Constructor order and parameterization can differ between libraries. Alea uses
normalized densities and masses with separate score and sampling capabilities;
it does not re-export rand_distr sampling types as likelihoods. Standardized
variants are obtained by supplying unit scale and zero location, rather than
adding aliases for the same law. `UniformSphere::new(2)` gives the unit circle;
`UniformBall::new(2)` gives the unit disc. Dimension three gives the sphere and
ball counterparts.

## Events and measures

`event_shape()` describes storage; `support()` describes the domain;
`reference_measure()` says what the density is relative to. These are different
properties. A four-component Dirichlet event stores four values, but its density
uses three independent simplex coordinates. A sphere law uses surface area;
a discrete law uses counting measure. A finite mixture requires a common event
shape and reference measure, even when its components have different domains.

`ScalarEventScore` supplies the derivative of log probability with respect to a
scalar observation. It is separate from `ParameterScore`, which differentiates
the law's parameters. Sampling returns an owned value that borrows the same event
type accepted by `log_prob`. Each law retains its own evaluation error type.

For a parameter-bound Wiener law, construct `Wiener4Law`, `Wiener5Law` or
`Wiener7Law` with parameters and `WienerOptions`. These implement `Distribution`,
`ParameterScore` and `Sample`. An event contains reaction time and response
boundary; its joint density is relative to time Lebesgue measure times boundary
counting measure. The lower-level Wiener evaluators remain available for fused
calculations and numerical diagnostics. A bound law validates its parameters and
keeps them immutable; evaluating a support zero returns negative infinity, while
invalid observations and numerical failures remain errors.
