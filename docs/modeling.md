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
    builder.prior("mean prior", mean_prior)?;

    // log Exponential(scale | rate = 1) = -scale for scale > 0.
    let minus_one = builder.constant(-1.0)?;
    let scale_prior = builder.multiply(minus_one, scale.expr())?;
    builder.prior("scale prior", scale_prior)?;

    for (i, &observation) in observations.iter().enumerate() {
        let likelihood = normal_observation(&mut builder, observation, mean.expr(), &scale)?;
        builder.likelihood(&format!("observation {i}"), likelihood)?;
    }
    builder.compile()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let compiled = model(&[1.2, 0.9, 1.4, 1.1])?;
    let layout = compiled.layout().clone();
    let target = compiled.select_posterior()?.into_target()?;
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

`prior` and `likelihood` add named terms to the total log density. `factor`
adds a generic term without assigning a statistical role. Creating an expression
alone does not make it part of the model. The same parameter handle may be used
in many factors; their derivative contributions accumulate into that parameter.

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
    for (name, parameter) in [
        ("intercept prior", intercept.expr()),
        ("slope prior", slope.expr()),
    ] {
        let prior = normal(&mut builder, parameter, zero, two)?;
        builder.prior(name, prior)?;
    }
    let observations = [(-1.0, false), (0.0, false), (1.0, true), (2.0, true)];
    for (i, (x, event)) in observations.into_iter().enumerate() {
        let x = builder.constant(x)?;
        let effect = builder.multiply(slope.expr(), x)?;
        let log_odds = builder.add(intercept.expr(), effect)?;
        let likelihood = bernoulli_logit(&mut builder, event, log_odds)?;
        builder.likelihood(&format!("observation {i}"), likelihood)?;
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
a Bayesian model. Declare observation terms with `likelihood`, prior densities
with `prior`, and arbitrary penalties with `potential`. Use `select_likelihood()`
to keep the observation terms explicitly. `into_objective()` evaluates their sum
after converting free coordinates into the declared parameter domains. It applies the chain rule to
the gradient, without a density-change Jacobian.

```rust
use alea::distributions::modeling::normal_observation;
use alea::model::assembly::{AssemblyError, CompiledModel, ModelBuilder};
use alea::target::LogDensityGradient;

fn likelihood() -> Result<CompiledModel, AssemblyError> {
    let mut b = ModelBuilder::new();
    let mean = b.real("mean")?;
    let sigma = b.positive("sigma")?;
    for (i, y) in [-1.0, 0.0, 2.0].into_iter().enumerate() {
        let observation = normal_observation(&mut b, y, mean.expr(), &sigma)?;
        b.likelihood(&format!("observation {i}"), observation)?;
    }
    b.compile()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let objective = likelihood()?.select_likelihood()?.into_objective()?;
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

`into_objective()` evaluates the active sum. Without a selection, it retains all
factors, including prior densities or penalties. It does not automatically select
likelihood factors or remove priors. A posterior objective is evaluated relative
to the prior's declared parameter measure. A MAP estimate depends on that reference measure.

For sampling, use `into_target()`. It adds the log-Jacobian and its gradient once,
so the density is expressed in the sampler's coordinates. For example, a positive
scale contributes `log(sigma)` to that sampling density. That term changes an
optimization objective and must not be added for MLE in the original parameters.

`select_posterior()` retains likelihoods and priors. Convert that view with
`into_objective()` to evaluate a mode objective in the declared parameter measure,
or with `into_target()` for sampling. Unclassified `factor` terms cause both
statistical selections to fail. An arbitrary `potential` is deliberately excluded
from a likelihood view and rejected by a posterior selection. Generic objectives
and sampling targets can still evaluate the complete sum with `select_all()`.
These declarations state your interpretation; they do not prove normalization or
posterior propriety.

Selection consumes the model without copying its data or callbacks. Recover it
with `objective.into_model()` or `target.into_model()` to select another view.
The original factors, parameter names and parameter coordinates remain available;
`factors()` lists all declarations and `selected_factors()` lists the active sum.
A parameter used only by a prior can become unidentified in the likelihood view.
A factor name can describe a joint data block and does not by itself define a
pointwise leave-out unit for cross-validation.

## Maximum likelihood and posterior modes

`alea::optimize::maximize` runs a bounded local L-BFGS search over a supplied
objective. Select the statistical factors first. Use `into_objective()` for MLE
or MAP in the original parameter measure; retain `into_target()` for sampling.

```rust
use alea::distributions::modeling::regression::{normal_regression, RegressionData};
use alea::model::{assembly::ModelBuilder, prediction::NamedOutput};
use alea::optimize::{
    maximize, LbfgsOptions, NamedFit, ObjectiveDescription, ObjectiveKind, OptimizationTermination,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut b = ModelBuilder::new();
    let mean = b.real("mean")?;
    let sigma = b.positive("sigma")?;
    // One explicit intercept column; no intercept is inserted automatically.
    let data = RegressionData::new(1, vec![1.0; 3], vec![0.0, 1.0, 2.0])?;
    let term = normal_regression(&mut b, &[mean.expr()], &sigma, data)?;
    b.likelihood("measurements", term)?;
    let model = b.compile()?;
    let parameters = NamedOutput::new(model.coordinates(), model.layout().clone(), ())?;
    let objective = model.select_likelihood()?.into_objective()?;
    let description = ObjectiveDescription {
        kind: ObjectiveKind::Likelihood,
        measure: "Lebesgue in mean and sigma".into(),
        model: "independent normal observations".into(),
        data: "example observations: 0, 1, 2".into(),
        source: "modeling guide".into(),
    };
    let options = LbfgsOptions::new(300, 3000)?.with_score_tolerance(1e-6)?;
    // Initial free coordinates: mean = 0, log(sigma) = 0.
    let result = maximize(&objective, &[0.0, 0.0], description, options)?;
    if result.termination != OptimizationTermination::ScoreTolerance {
        return Err(format!("search stopped: {:?}", result.termination).into());
    }
    let fit = NamedFit::new(parameters, result)?;
    let mean = fit.estimate("mean").ok_or("no mean estimate")?;
    let sigma = fit.estimate("sigma").ok_or("no scale estimate")?;
    println!("mean = {mean}, sigma = {sigma}");
    assert!((mean - 1.0).abs() < 1e-5);
    assert!((sigma - (2.0_f64 / 3.0).sqrt()).abs() < 1e-5);
    Ok(())
}
```

The scale estimate uses the maximum-likelihood divisor `n`, not the residual
variance divisor `n - p`. For MAP, declare priors, select `select_posterior()` and
identify the prior's parameter measure in the description. Objective descriptions
are caller declarations; they do not inspect or verify the model's factors.
Record actual source and data digests when saving research results.

Results retain the final objective value, free-coordinate gradient, complete
numerical policy, iterations and evaluation counts. `with_scales` declares
coordinate scales: with `z_i = q_i / scale_i`, the stopping condition is
`max_i |scale_i * gradient_i| <= tolerance`. The tolerance is absolute and does
not loosen when a constant is added to the objective. A score tolerance alone
cannot establish a maximum, identifiability, existence of a finite MLE or global
optimality. Check the statistical problem as well as the termination reason.

Iteration/evaluation limits, cancellation, numerical directions and failed line
searches have separate outcomes. `maximize_with_cancel` checks a callback between
calls; it cannot interrupt a model evaluation. Negative-infinity trial objectives
are backtracked over as explicit support rejections. Typed backend errors, NaN,
positive infinity and incomplete gradients stop with the original error and the
last valid candidate. An invalid initial evaluation has no candidate. A tiny
unrepresentable step is a search failure, not successful convergence.

This result is a point-estimation record. It has no chain/draw axes or invented
standard errors. The inverse-BFGS preconditioner is not a validated covariance
estimate. Pathfinder shares the numerical inverse-BFGS action but retains its
separate approximation selection and resampling workflow; see the
[Pathfinder reference](https://mc-stan.org/docs/reference-manual/pathfinder.html).

## Batch regression likelihoods

`alea::distributions::modeling::regression` supplies three analytic batch factors:

| Function | Response | Linear predictor |
| --- | --- | --- |
| `normal_regression` | Finite real values | Mean, with a shared positive residual standard deviation |
| `bernoulli_logit_regression` | Boolean outcomes | Log odds |
| `poisson_log_regression` | Unsigned integer counts, at most 2^53 | Log rate |

`RegressionData::new(columns, design, observations)` takes a row-major design.
Its columns correspond exactly to the coefficient expressions you supply. Include
a column of ones for an intercept. `with_offsets` adds an explicit offset to each
linear predictor; Poisson exposure `e > 0` is supplied as `ln(e)`, not `e`. Empty
data, shape mismatches and nonfinite inputs are rejected rather than dropped.
Zero columns permit an offset-only likelihood.

The factors retain one graph node per batch regardless of row count; dataset
storage still grows with the observations. Values, predictors and score reductions
use compensated summation, and each fused call allocates temporary score storage
proportional to the number of coefficients. Captured data cannot currently be
exported as an Enzyme primal; the analytic scores work on stable Rust.

Each returned factor is joint over its batch. Choose separate batches when you
need separately declared pointwise predictive units; attaching observation IDs
does not split an aggregated factor. Chunked batches produce the same mathematical
likelihood, with ordinary floating-point reduction differences.

`NamedFit` attaches validated parameter metadata and constrains the final candidate
without a Jacobian. `estimates()` follows the constrained names in `parameters()`;
`result()` retains the free-coordinate gradient, stopping condition and counters.
A simplex can therefore have more named values than free coordinates. No candidate
means no estimates. A failed projection returns the original numerical result.
Metadata must describe the objective you actually fitted; dimensions alone cannot
establish that identity.

These likelihood factors do not establish design rank or finite estimator
existence. Perfect Gaussian fits, separated binary outcomes and all-zero Poisson
counts need problem-specific assessment. A small score in such a problem is not
proof of a finite MLE. No standard errors are inferred from L-BFGS history.

## Pointwise likelihoods and predictive units

For predictive comparison, choose what an omitted unit means before evaluating
its likelihood. It could be one reaction time, all measurements from a subject,
or a joint multivariate observation. A joint density cannot be divided into
observation-level terms just by renaming it.

`PointwiseLikelihood` groups explicitly named likelihood factors. Each factor
must belong to exactly one unit, and observation identifiers must not overlap.
The output order follows the unit list, preserving alignment with your data.
Priors, arbitrary potentials and parameter-transform Jacobians are excluded.

```rust
use alea::distributions::modeling::normal_observation;
use alea::model::assembly::{ModelBuilder, PointwiseLikelihood, PredictiveUnit};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut b = ModelBuilder::new();
    let mean = b.real("mean")?;
    let sigma = b.positive("sigma")?;
    let mut units = Vec::new();
    for (i, y) in [1.2, 0.8].into_iter().enumerate() {
        let factor = normal_observation(&mut b, y, mean.expr(), &sigma)?;
        let name = format!("measurement {i}");
        b.likelihood(&name, factor)?;
        units.push(PredictiveUnit {
            name: name.clone(),
            factors: vec![name],
            observations: vec![format!("data row {i}")],
            conditioned_on: vec!["mean".into(), "sigma".into()],
            integrated_out: vec![],
        });
    }
    let pointwise = PointwiseLikelihood::new(b.compile()?, units)?;
    let mut terms = vec![0.0; pointwise.units().len()];
    let total = pointwise.evaluate(&[1.0, 0.5_f64.ln()], &mut terms)?;
    assert!((total - terms.iter().sum::<f64>()).abs() < 1e-12);
    for (unit, value) in pointwise.units().iter().zip(&terms) {
        println!("{}: {value}", unit.name);
    }
    Ok(())
}
```

To evaluate a joint subject-level likelihood, list that subject's factors and
observation identifiers together in one unit. Terms within each unit are summed
in their original declaration order. Regrouping can introduce small differences
between the sum of unit values and the total likelihood.

The conditioning fields describe the likelihood you supplied. For example,
`conditioned_on: vec!["subject effect".into()]` describes prediction conditional
on an existing effect; `integrated_out: vec!["subject effect".into()]` describes a
likelihood whose calculation already integrates that effect out. These are
different prediction questions. Setting a label performs no integration and does
not establish that units are statistically independent or suitable for LOO.

For targets built with `CompiledModel::into_target`, recover the model after
fitting with `target.into_model()` and pass it to `PointwiseLikelihood::new`. You can recover it again with `pointwise.into_model()`.
The evaluator reuses its storage and runs the forward graph once per coordinate
vector, without reverse propagation. Custom value/VJP callbacks still supply
both outputs under their existing contract. Invalid charts, nonfinite values
(including negative-infinite log likelihoods), or callback failures return errors
and leave the output untouched. Preserve those failures rather than replacing
them with a probability of zero.
