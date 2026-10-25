# Probability distributions

The `distributions` feature provides normalized log densities, probability masses,
sampling, and derivatives. Constructors validate parameters, and optional traits
make the capabilities of each law explicit.

## Evaluate and sample

```rust
use alea::distribution::{ContinuousDistribution, Distribution, Sample};
use alea::distributions::scalar::Normal;
use rand::{rngs::SmallRng, SeedableRng};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let law = Normal::new(2.0, 0.5)?;
let observation = 2.3;
println!("log density = {}", law.log_prob(&observation)?);
println!("derivative with respect to the observation = {}", law.event_score(observation)?);
let mut rng = SmallRng::seed_from_u64(123);
println!("simulated observation = {}", law.sample(&mut rng)?);
# Ok(())
# }
```

`Normal::new` takes a mean and a standard deviation. Its `log_prob` includes the
normalizing constant. This differs from `alea::distributions::Gaussian`, which is
an unnormalized multivariate standard-normal target for inference examples.

Sampling uses a random-number generator supplied by you. Record its seed when
reproducible simulated data or predictions matter.

## Available laws

| Module | Laws | Parameter convention |
|---|---|---|
| `scalar` | Normal, Student-t, LogNormal | Location/scale; Student-t also takes degrees of freedom |
| `scalar` | Exponential, Gamma | Exponential rate; Gamma shape and rate |
| `scalar` | Beta, Bernoulli, Binomial, Poisson | Shape parameters, probability, trials, or rate as appropriate |
| `multivariate` | MultivariateNormal | Mean vector and lower Cholesky factor of covariance |
| `multivariate` | Dirichlet, Categorical, Multinomial | Concentrations or probabilities |
| `multivariate` | LkjCholesky | Dimension and concentration for a correlation Cholesky factor |
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

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let success = Bernoulli::new(0.7)?;
assert!((success.log_prob(&true)?.exp() - 0.7).abs() < 1e-12);
let counts = Poisson::new(3.0)?;
println!("P(count = 2) = {}", counts.log_prob(&2)?.exp());
# Ok(())
# }
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

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let law = MultivariateNormal::new(
    vec![0.0, 0.0],
    vec![1.0, 0.0,
         0.0, 2.0],
)?;
println!("log density = {}", law.log_prob(&[0.2, -0.5])?);
# Ok(())
# }
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

Random sampling uses bounded rejection work. Poisson sampling supports rates up
to 30,000; Binomial and Multinomial sampling support at most one million trials.
These sampling limits do not restrict their log-probability methods.

To use these laws as priors or likelihoods, see [Models and constraints](modeling.md).
