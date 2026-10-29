# Models and parameter constraints

A Bayesian model combines a prior with a likelihood. Alea's `ModelBuilder` lets
you express those terms using named parameters and distribution factors. It
computes derivatives and applies parameter transforms for you.

Enable the `distributions` feature on your `alea` dependency for the examples on
this page. See [Installation](installation.md) for the dependency and C compiler.

## A normal model with an unknown scale

Here both the population mean and measurement scale are unknown. We use a normal
prior on the mean and an exponential prior with rate 1 on the scale.

```rust
use alea::distributions::modeling::{normal, normal_observation};
use alea::{
    model::assembly::{AssemblyError, CompiledModel, ModelBuilder},
    target::LogDensityGradient,
};

fn model(observations: &[f64]) -> Result<CompiledModel, AssemblyError> {
    let mut builder = ModelBuilder::new();
    let mean = builder.real("mean")?;
    let scale = builder.positive("scale")?;

    let zero = builder.constant(0.0)?;
    let two = builder.constant(2.0)?;
    let mean_prior = normal(&mut builder, mean.expr(), zero, two)?;
    builder.factor(mean_prior)?;

    // log Exponential(scale | rate = 1) = -scale for scale > 0.
    let minus_one = builder.constant(-1.0)?;
    let scale_prior = builder.multiply(minus_one, scale.expr())?;
    builder.factor(scale_prior)?;

    for &observation in observations {
        let likelihood = normal_observation(&mut builder, observation, mean.expr(), &scale)?;
        builder.factor(likelihood)?;
    }
    builder.compile()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let compiled = model(&[1.2, 0.9, 1.4, 1.1])?;
    let layout = compiled.layout().clone();
    let target = compiled.into_target()?;
    let q = [1.0, 0.0];
    let mut gradient = [0.0; 2];
    let logp = target.logp_grad(&q, &mut gradient)?;

    let mut parameters = [0.0; 2];
    let mut scratch = vec![0.0; layout.scratch_dimension()];
    layout.constrain(&q, &mut parameters, &mut scratch)?;
    assert_eq!(parameters, [1.0, 1.0]);
    println!(
        "log posterior = {logp}; mean = {}; scale = {}",
        parameters[0], parameters[1]
    );
    Ok(())
}
```

`factor` adds a term to the total log density. Creating an expression alone does
not make it part of the model. The same parameter handle may be used in many
factors; their derivative contributions accumulate into that parameter.

Build one compiled model per chain. Its structure is fixed after `compile`, and
its scratch storage is reused during evaluation. A session's target factory is a
convenient place to create it.

## Understand the sampling coordinates

The scale must be positive, so Alea samples `log(scale)`. In the example,
`q = [1.0, 0.0]` means a mean of 1 and a scale of `exp(0) = 1`.

`into_target` adds the transform's Jacobian once. Do not add another Jacobian to
the prior or each observation. Use the saved `ParameterLayout` to convert stored
positions back to parameter units before plotting or reporting them.

| Declaration | Parameter values | Sampling coordinate |
|---|---|---|
| `builder.real("mean")` | Any finite real | The parameter itself |
| `builder.positive("scale")` | Positive | Logarithm |
| `builder.unit_interval("probability")` | Between 0 and 1 | Log odds |
| `builder.vector::<alea::model::assembly::DynamicDomain>(..., transform)` | Depends on the transform | Transform-specific coordinates |

Simplex and correlation transforms change dimension. A simplex with three
probabilities has two unconstrained coordinates. Do not assume that a stored
position has the same length as the corresponding constrained parameter vector.

## Logistic regression

Use a Bernoulli likelihood in log-odds coordinates to avoid manually taking logs
of probabilities close to zero or one:

```rust
use alea::distributions::modeling::{bernoulli_logit, normal};
use alea::model::assembly::ModelBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = ModelBuilder::new();
    let intercept = builder.real("intercept")?;
    let slope = builder.real("slope")?;
    let zero = builder.constant(0.0)?;
    let two = builder.constant(2.0)?;
    for parameter in [intercept.expr(), slope.expr()] {
        let prior = normal(&mut builder, parameter, zero, two)?;
        builder.factor(prior)?;
    }
    for (x, event) in [(-1.0, false), (0.0, false), (1.0, true), (2.0, true)] {
        let x = builder.constant(x)?;
        let effect = builder.multiply(slope.expr(), x)?;
        let log_odds = builder.add(intercept.expr(), effect)?;
        let likelihood = bernoulli_logit(&mut builder, event, log_odds)?;
        builder.factor(likelihood)?;
    }
    let model = builder.compile()?;
    assert_eq!(model.coordinates().len(), 2);
    Ok(())
}
```

The observed boolean is data. HMC and NUTS move continuous parameters such as the
intercept and slope; they do not differentiate a discrete category.

## Posterior prediction

Prediction asks what observations the fitted model would generate. It is separate
from fitting: draw parameter values from the posterior, then simulate new data.

```rust
use alea::distribution::Sample;
use alea::distributions::scalar::Normal;
use alea::{
    model::prediction::posterior_predict,
    transform::{ParameterLayout, Transform},
};
use rand::{SeedableRng, rngs::SmallRng};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let layout = ParameterLayout::new(vec![Transform::identity(1)?, Transform::positive(1)?])?;
    // For illustration. In a fit, use the positions collected from your chains.
    let positions = [[1.0, 0.0], [1.2, -0.2], [0.9, 0.1]];
    let mut rng = SmallRng::seed_from_u64(42);
    let predictions = posterior_predict(
        &layout,
        positions.iter().map(|q| q.as_slice()),
        &mut rng,
        |parameters, rng| Normal::new(parameters[0], parameters[1])?.sample(rng),
    )?;
    assert_eq!(predictions.generated.len(), 3);
    Ok(())
}
```

Use the layout saved from your compiled model in a real analysis. The predictive
helper constrains each position before calling your generator. It does not add
another density weight or Jacobian.

## Supply a custom density

Implement `LogDensityGradient` when you already have a log density and gradient,
as in [Your first model](getting-started.md). Every successful call must overwrite
the complete gradient buffer. Return a concrete error for an evaluation failure.
An impossible observation can have log density negative infinity, but it must not
be confused with NaN or an unexpected numerical failure.

For an opaque factor inside a built model, `ModelBuilder::custom` accepts a local
value-and-derivative callback. Its input expressions and derivative slots have the
same order. Incorrect custom derivatives can invalidate a fit, so check them
against finite differences before sampling. [Enzyme code generation](enzyme-native-models.md)
is also available for compatible models that need compiled automatic derivatives.

For a complete model with shared population parameters and named group effects,
see [Partial pooling with a hierarchical model](hierarchical-models.md).

`TargetSpace` holds the coordinate dimension and error type shared by a target's
capabilities. `LogDensity` evaluates only the log density; `LogDensityGradient`
evaluates the value and score together. Implementing both does not force
primal-only callers to calculate gradients. Higher derivative capabilities are
named for the potential `U = -log_density`, including `PotentialHessianVector`.

State-space models use the same time index for transition sampling and transition
density evaluation. Observation likelihoods also receive time, and all three
operations return typed errors. A particle proposal and an incremental importance
weight have separate contracts; a general importance weight includes proposal
corrections and cannot be replaced by the observation likelihood alone.

## Likelihood objectives and parameter coordinates

A likelihood can use the same parameter declarations and distribution factors as
a Bayesian model. Include only the observation factors when you want a likelihood
objective. `into_objective()` evaluates their sum after converting free
coordinates into the declared parameter domains. It applies the chain rule to
the gradient, without a density-change Jacobian.

```rust
use alea::distributions::modeling::normal_observation;
use alea::model::assembly::{AssemblyError, CompiledModel, ModelBuilder};
use alea::target::LogDensityGradient;

fn likelihood() -> Result<CompiledModel, AssemblyError> {
    let mut b = ModelBuilder::new();
    let mean = b.real("mean")?;
    let sigma = b.positive("sigma")?;
    for y in [-1.0, 0.0, 2.0] {
        let observation = normal_observation(&mut b, y, mean.expr(), &sigma)?;
        b.factor(observation)?;
    }
    b.compile()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let objective = likelihood()?.into_objective()?;
    let q = [0.3, 0.5_f64.ln()]; // mean, log(sigma)
    let mut score = [0.0; 2];
    let log_likelihood = objective.logp_grad(&q, &mut score)?;
    println!("log likelihood = {log_likelihood}, score = {score:?}");
    Ok(())
}
```

The returned value is the likelihood at `sigma = exp(q[1])`. This evaluates the
objective; a numerical optimization procedure is needed to find its maximum.
The named parameter metadata remains available through `objective.model()`.

`into_objective()` retains every factor already in the model. If you add prior
densities, it evaluates that posterior objective relative to the prior's declared
parameter measure. It does not automatically select likelihood factors or remove
priors. A MAP estimate depends on that reference measure.

For sampling, use `into_target()`. It adds the log-Jacobian and its gradient once,
so the density is expressed in the sampler's coordinates. For example, a positive
scale contributes `log(sigma)` to that sampling density. That term changes an
optimization objective and must not be added for MLE in the original parameters.
