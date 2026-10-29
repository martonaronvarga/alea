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

Use `ModelBuilder::custom_with_scratch` for a callback that needs reusable working
storage. Supply its initial scratch and a callback taking `(inputs, scores, scratch)`.
The compiled model owns that storage and borrows it exclusively during evaluation.
Reset every scratch entry you read and write every local score on each successful
call. Scratch survives errors and unwinding; it must never change the mathematical
result according to previous calls. Scratch-backed nodes currently have no generated primal and return
`MissingPrimal` when source generation reaches them.

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
use compensated summation. Score storage, proportional to the number of
coefficients, is allocated at construction and reused without evaluation allocations. Captured data cannot currently be
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

## Check estimator existence separately

Numerical convergence and existence of an estimator answer different questions.
For example, a separated logistic model can produce a rounded zero score at a
large coefficient even though its likelihood has no finite maximum.

The regression diagnostics check proposed directions of the unconstrained
coefficient vector. For Bernoulli-logit likelihoods, a direction `d` is an
improving ray when every signed margin `(2*y_i - 1) * (X_i d)` is nonnegative
and at least one is positive. All-positive margins give complete separation;
zero margins give quasi-complete separation. For Poisson-log likelihoods, positive
counts must have `X_i d = 0`, zero counts must have `X_i d <= 0`, and at least one
zero-count predictor must decrease strictly. Along either checked ray the
likelihood improves for every finite starting coefficient, so no finite maximum
exists. Finite offsets do not change these conditions. See
[Albert and Anderson](https://doi.org/10.1093/biomet/71.1.1) and
[Santos Silva and Tenreyro](https://doi.org/10.1016/j.econlet.2010.02.020).

```rust
use alea::distributions::modeling::regression::{
    RegressionData, diagnostics::{DirectionAssessment, RecessionKind},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = RegressionData::new(1, vec![-1.0, 1.0], vec![false, true])?;
    match data.bernoulli_recession(&[1.0])? {
        DirectionAssessment::Witness(witness) => {
            assert_eq!(witness.kind(), RecessionKind::CompleteSeparation);
            println!("no finite logistic MLE along {:?}", witness.direction());
        }
        DirectionAssessment::NotWitness { .. } => {
            println!("this direction does not settle existence");
        }
    }
    Ok(())
}
```

`bernoulli_coordinate_recession()` and `poisson_coordinate_recession()` screen
only the positive and negative coordinate axes. A returned witness is checked;
`None` is unresolved, not evidence of overlap. A supplied combination of columns
may reveal separation that this simple screen misses. Dot-product signs use exact
dyadic integer arithmetic on the stored binary64 data and direction, including
cancelling or underflowing products. Every finite binary64 value is an integer
multiple of 2^-1074, so dot-product signs can be evaluated with integer sums
on the common product scale 2^-2148. This is an allocating diagnostic, not part
of hot likelihood evaluation. Keep the checked dataset's identity with a saved
direction and recheck it if the data change.

Use `bernoulli_mle_existence(options)` or `poisson_mle_existence(options)` to
search combinations of columns and assess the entire recession cone:

```rust
use alea::distributions::modeling::regression::{
    RegressionData, diagnostics::{ConeSearchOptions, MleExistence},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Neither coordinate axis separates these observations; a combined direction does.
    let data = RegressionData::new(
        2, vec![1.0, -0.5, -0.5, 1.0], vec![true, true],
    )?;
    match data.bernoulli_mle_existence(ConeSearchOptions::default())? {
        MleExistence::FiniteMaximum => println!("a finite maximum is attained"),
        MleExistence::NoFiniteMaximum(witness) => {
            println!("improving integer ray: {:?}", witness.direction().coefficients());
        }
        MleExistence::Unresolved => println!("the search exceeded its resource budget"),
    }
    Ok(())
}
```

The search forms `A*d >= 0` and `E*d = 0`. Bernoulli rows in `A` are the
signed design rows; Poisson rows in `A` are the negated zero-count design rows,
and `E` contains the positive-count rows. Strict improvement requires at least
one positive component of `A*d`. Since the cone is homogeneous, this is
equivalent to adding `sum(A*d) >= 1`: any improving ray can be rescaled to
satisfy that inequality. Flat design-nullspace directions cannot satisfy it.

Fourier–Motzkin elimination solves this linear feasibility problem with exact
integers derived from the stored binary64 values. Infeasibility establishes
`FiniteMaximum` for these two likelihood families, including designs without an
intercept. A finite maximum can have nonunique coefficients when the design has
a nullspace; it does not guarantee representable estimates, successful numerical
optimization, or valid standard errors. A feasible system yields an exact integer
ray after rational back substitution. Its `kind()` describes that ray's margins;
a ray with flat margins does not rule out another completely separating ray.
Do not round the returned integers to f64 before checking Poisson equalities.

Elimination can grow exponentially. `ConeSearchOptions` limits the input columns,
total input/generated rows, and integer sizes. The defaults allow 32 columns,
16,384 rows, and 8,192-bit coefficients; this is a diagnostic for bounded problems,
not an unrestricted large-design LP solver. Any exhausted limit returns
`Unresolved`, never `FiniteMaximum`. Increasing a budget can settle an unresolved
case, but a numerical score tolerance cannot override an improving-ray witness.
The independent test oracle uses exact rational vertex enumeration rather than
the production elimination algorithm. See also
[Konis's separation analysis](https://ora.ox.ac.uk/objects/uuid:8f9ee0d0-d78e-4101-9ab4-f9cbceed2a2a).

These checks assume independent unconstrained coefficients and a likelihood
without priors. Parameter restrictions, shared/nonlinear coefficient expressions
or a proper prior require a different assessment. They do not prevent using a
separated likelihood in a Bayesian model.

All-zero Poisson counts alone do not prove a boundary estimate. With an intercept,
a negative intercept direction is a witness. With no intercept and predictors
`[-1, 1]`, the log likelihood is `-exp(-beta)-exp(beta)`, which has its finite
maximum at zero. The diagnostic checks the actual design rather than flagging
zero responses automatically.

### Gaussian rank and residual scale

Enable Alea's `faer` feature for `gaussian_assessment`. It uses the existing
Faer thin-SVD backend, not normal equations, with an explicit relative singular-
value cutoff. The same least-squares calculation is available independently as
`alea::math::least_squares::solve`.

```rust
use alea::distributions::modeling::regression::RegressionData;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = RegressionData::new(
        2,
        vec![1.0, -1.0, 1.0, 0.0, 1.0, 1.0],
        vec![0.0, 2.0, 2.0],
    )?;
    let assessment = data.gaussian_assessment(1e-12, 1e-12)?;
    println!("coefficients: {:?}", assessment.least_squares.coefficients);
    println!("issues: {:?}", assessment.issues);
    assert!(assessment.issues.is_empty());
    assert!((assessment.scale_candidate.ok_or("unqualified scale")?
        - (2.0_f64 / 9.0).sqrt()).abs() < 1e-12);
    Ok(())
}
```

The first tolerance selects retained singular directions relative to the largest
singular value. Numerical rank depends on units and this policy; it is not an
exact rank certificate. Coefficients are the minimum-norm solution in the retained
subspace, including for wide or numerically rank-deficient designs.

The second tolerance flags a residual norm at or below that fraction of the
offset-adjusted response norm. Exact and near-perfect fits produce
`VanishingResidual`; no positive residual-scale candidate is then qualified.
Rank and residual issues can appear together. Otherwise `scale_candidate` is the
retained-fit residual norm divided by `sqrt(n)`, not `sqrt(n-p)`. Discarded singular
directions and other issues must still be assessed before calling it an MLE.
Residual degrees of freedom are reported separately; no standard errors or
covariance matrix are inferred. These numerical cutoffs are not confidence levels
or rigorous roundoff enclosures.

## Keeping fit evidence together

An optimizer outcome answers a numerical question: why did this search stop?
Regression assessment answers different questions: does a finite MLE exist under
these assumptions, and which design directions are numerically identified?
Keep both records, even when their conclusions differ.

```rust
use alea::{
    distributions::modeling::regression::{
        RegressionData, bernoulli_logit_regression,
        diagnostics::{ConeSearchOptions, RegressionEvidence, MleExistence},
    },
    model::{assembly::ModelBuilder, prediction::NamedOutput},
    optimize::{maximize, LbfgsOptions, NamedFit, ObjectiveDescription, ObjectiveKind},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = RegressionData::new(1, vec![1.0; 4], vec![false, true, true, true])?;
    let evidence = data.assess_bernoulli(&["intercept"], ConeSearchOptions::default())?;
    let mut builder = ModelBuilder::new();
    let intercept = builder.real("intercept")?;
    let likelihood = bernoulli_logit_regression(&mut builder, &[intercept.expr()], data)?;
    builder.likelihood("responses", likelihood)?;
    let model = builder.compile()?;
    let names = NamedOutput::new(model.coordinates(), model.layout().clone(), ())?;
    let objective = model.select_likelihood()?.into_objective()?;
    let result = maximize(&objective, &[0.0], ObjectiveDescription {
        kind: ObjectiveKind::Likelihood,
        measure: "unconstrained intercept".into(),
        model: "intercept-only logistic regression".into(),
        data: "four responses".into(),
        source: "example".into(),
    }, LbfgsOptions::new(100, 1000)?)?;
    let fit = NamedFit::new(names, result)?.with_assessment(evidence);
    println!("intercept: {:?}", fit.estimate("intercept"));
    println!("termination: {:?}", fit.result().termination);
    println!("search failure: {:?}", fit.result().line_search_failure);
    assert!(matches!(fit.assessment().evidence(),
        RegressionEvidence::BernoulliLogit(MleExistence::FiniteMaximum)));
    Ok(())
}
```

`with_assessment` attaches evidence; it does not inspect or validate an arbitrary
model. Here the same data and explicitly named coefficient column define both
the assessment and the likelihood. If you add priors, penalties, coefficient
constraints, shared/nonlinear coefficients or weights, these fixed-design MLE
assumptions no longer describe the fitted objective. Matching names alone is not
proof that the evidence belongs to a fit.

Use `assess_poisson` for Poisson-log models. With `faer`,
`assess_bernoulli_with_rank` and `assess_poisson_with_rank` additionally accept an
explicit relative singular-value cutoff. `assess_gaussian` accepts both rank and
residual tolerances and retains the Gaussian least-squares evidence.
`coefficient_names()` records design-column order; `search_options()` records the
exact-search budget. Missing `design_rank()` means rank was not assessed.
A finite maximum can be nonunique through a design nullspace. An unresolved
search remains unresolved, including when the optimizer meets its score tolerance.

`LineSearchFailed` retains either `BacktrackLimit` or `NoRepresentableStep` in
`line_search_failure`. The first means that no allowed trial was accepted; the
second means the next trial rounded to the current position. Neither proves
that the maximum or the score tolerance was reached. A large dataset or a large
constant in the objective can make strict objective improvements unresolvable
before a stringent score tolerance is met. Alea does not accept equal-valued
steps or silently loosen the tolerance in that situation.

The optimizer reuses scaled-score, trial-position and trial-gradient buffers within
one fit. An accepted trial swaps buffers with the retained candidate; rejected or
failed evaluations cannot overwrite that candidate. Every trial gradient is reset
to NaN so incomplete backend writes remain detectable. Rejected line-search trials
do not allocate optimizer working vectors. Secant history, result construction
and allocations inside a supplied model remain separate costs. This storage
policy preserves the objective and acceptance rules; it is not a guarantee of
lower fitting time.

### Stable predictors and coefficient coordinates

Linear predictors can lose accuracy when large products cancel. Compensating
only their sum cannot recover bits already lost in multiplication. Batch regression
factors retain the product residual with a fused multiply-add, following the
[TwoProductFMA construction of Ogita, Rump and Oishi](https://www.tuhh.de/ti3/paper/rump/OgRuOi05.pdf).
For scalar assembly, `ModelBuilder::linear_combination(inputs, coefficients, offset)`
uses the same accumulator; its analytic input derivatives are the supplied weights.
This node supports native analytic gradients; generated Rust primal lowering
returns `MissingPrimal`. Neither path supplies a certified roundoff enclosure.

A design can also make optimization difficult even when its scores are accurate.
For example, a predictor almost constant at one is strongly confounded with an
intercept. `RegressionData::standardize(intercept_column)` explicitly centers and
scales the other columns. It requires an existing column of exact ones and shares
the original observations and offsets. Constant columns become zero with scale
one; they remain in the design and coefficient map. No rank truncation, penalty,
observation deletion or automatic optimizer retry occurs.

For a non-intercept column, the map is

\[
z_j=(x_j-c_j)/s_j,\qquad
\theta_j=s_j\beta_j,\qquad
\theta_0=\beta_0+\sum_jc_j\beta_j.
\]

Here `c_j` is the population mean and `s_j` the population standard deviation.
Both coordinate systems express the same linear predictor in real arithmetic.
The returned map converts coefficients explicitly. Floating-point reconstruction
still has limits, especially for large cancelling original coefficients.

```rust
use alea::{
    distributions::modeling::regression::{RegressionData, poisson_log_regression},
    model::assembly::ModelBuilder,
    optimize::{maximize, LbfgsOptions, ObjectiveDescription, ObjectiveKind},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let original = RegressionData::new(
        2,
        vec![1.0, 1.0 - 2e-8, 1.0, 1.0 - 1e-8,
             1.0, 1.0 + 1e-8, 1.0, 1.0 + 2e-8],
        vec![1_u64, 2, 3, 4],
    )?;
    let (data, map) = original.standardize(0)?.into_parts();
    let mut builder = ModelBuilder::new();
    let intercept = builder.real("centered_intercept")?;
    let slope = builder.real("standardized_slope")?;
    let likelihood = poisson_log_regression(
        &mut builder, &[intercept.expr(), slope.expr()], data,
    )?;
    builder.likelihood("counts", likelihood)?;
    let objective = builder.compile()?.select_likelihood()?.into_objective()?;
    let result = maximize(&objective, &[0.0, 0.0], ObjectiveDescription {
        kind: ObjectiveKind::Likelihood,
        measure: "centered and scaled coefficients".into(),
        model: "Poisson regression".into(),
        data: "four counts".into(),
        source: "example".into(),
    }, LbfgsOptions::new(100, 1000)?)?;
    println!("termination: {:?}", result.termination);
    if let Some(candidate) = &result.candidate {
        let beta = map.to_original(&candidate.position)?;
        println!("original intercept and slope: {beta:?}");
    }
    Ok(())
}
```

The stopping condition above concerns the declared standardized coordinates.
Converting the final coefficients does not relabel the original optimizer gradient
or prove convergence in another norm. Keep its result and coordinate map together.
For MLE, composing the likelihood with this invertible map requires no density
Jacobian. Bayesian priors and coefficient constraints need their own explicit
transformation. Applying independent priors to the new coefficients generally
specifies a different model.

Centering and scaling address intercept correlation and unequal units; they do
not remove arbitrary correlations among several predictors or establish rank.
[Stan's regression guide](https://mc-stan.org/docs/stan-users-guide/regression.html)
also describes centering and a more general QR reparameterization.
[R's GLM fitter](https://raw.githubusercontent.com/wch/r-source/trunk/src/library/stats/R/glm.R)
uses iteratively reweighted least squares with QR, while
[statsmodels documents SVD and QR alternatives](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.fit.html).
Their rank cutoffs and convergence criteria differ; align those policies before
comparing coefficients or success flags.

### Validation and performance

The source checkout includes `regression_fit`, the independent 60-digit Decimal
check `tools/downstream-conformance/python/check_regression_fits.py`, and an
optional actual SciPy comparison in `check_regression_references.py`. Run the
latter in the analysis Nix shell. It records package versions and executes SciPy
SVD least squares and analytic-Jacobian root solvers. R, statsmodels and Stan
informed the design review; those engines were not executed in this comparison.

The paired cohort uses the same inputs, budgets and error thresholds in raw and
standardized coordinates. Its 216 planned executions completed. All 192
finite-reference evaluations passed both original-score and optimization-score
accuracy checks. All required candidate checks passed, including every
standardized ill-conditioned case. Maximum
Gaussian score error in the difficult design was below `1e-13`, compared with
about `5.6e-7` before product compensation. Standardized Poisson predictions
agreed with the independent reference within `4e-9`, below the unchanged `5e-6`
threshold. The six raw-coordinate Poisson prediction failures remain recorded
controls: generic L-BFGS does not automatically discover a design transformation.

SciPy's standardized fits passed the same prediction threshold for all three
families. Its raw-coordinate logistic and Poisson root solves did not. These
results demonstrate the importance of arithmetic and coordinate choice; they
do not rank inference libraries. The cohort contains repeated deterministic
executions, not independent simulated datasets or statistical calibration.

The optimizer's strict stopping rule was met in 78 of the 216 executions.
Candidate accuracy, numerical termination and finite-MLE existence remain separate.
Separated, boundary and perfect-fit cases retain their original typed outcomes.
The tools check original coefficients against the original design and independently
check optimization gradients on the recorded standardized design.

The separate scalar/batch benchmark completed all 54 planned runs at 10, 1,000
and 1,000,000 rows. All nine family/size comparisons passed their numerical and
agreement gates. Both paths used the original absolute score tolerance `n * 1e-7`,
not the default `1e-8`. The scalar graph expands each observation into elementary
model nodes; the batch graph uses one likelihood node. The table shows median
ratios of scalar time or memory to batch time or memory at one million rows,
from three fresh processes per path on one Linux host.

| Likelihood | Construction | Evaluation | Fitting | Peak process memory |
| --- | ---: | ---: | ---: | ---: |
| Gaussian | 5.59 | 2.91 | 3.22 | 8.27 |
| Bernoulli-logit | 7.08 | 1.54 | 1.51 | 7.45 |
| Poisson-log | 3.55 | 3.45 | 3.43 | 8.12 |

Poisson graph construction was slower in the batch path at ten rows and faster
at 1,000 rows on these inputs. This brackets a workload-specific crossover;
it does not identify a universal threshold. These are exploratory measurements,
not guaranteed speedups or comparisons with another library. More accurate
product accumulation adds work, and its cost depends on the processor's FMA
implementation.

At one million rows, batch graphs contained three or four expressions; scalar
graphs contained four to five million. Peak batch memory was about 57 MiB,
compared with 423–469 MiB for scalar graphs. Batch graph construction is bounded;
stored observations and design storage still grow with the data size.

Native graph factor-value sums also use the compensated accumulator. This fixed
the million-row scalar Gaussian search failure without changing its declared
stopping tolerance. These arithmetic improvements do not certify arbitrary
intermediate expressions, reverse accumulations or generated Enzyme code.

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

### Reusing batch evaluation storage

A separate before/after comparison measured the change from temporary batch score
vectors to evaluator-owned scratch. It used the same three likelihoods at 10,
1,000 and 1,000,000 rows, with seven fresh processes per revision and path:
252 planned runs, all completed. Runs were pinned to one CPU and revision/path
order was shuffled within each repetition. All 18 family/size/path comparisons
passed the existing numerical gates. Paired values, gradients, fitted parameters,
termination reasons and work counts were identical. Independent 60-digit Decimal
calculations also passed the existing initial-evaluation and fitted-candidate checks.

The following ratios are previous fitting time divided by revised fitting time
for the batch path; values above one favor reuse.

| Likelihood | 10 rows | 1,000 rows | 1,000,000 rows |
| --- | ---: | ---: | ---: |
| Gaussian | 0.98 | 0.99 | 1.00 |
| Bernoulli-logit | 0.99 | 1.01 | 1.02 |
| Poisson-log | 1.18 | 0.98 | 0.97 |

These exploratory measurements on one Intel i5-8250U host do not establish a
general fitting speedup. The verified benefit is allocation-free batch evaluation
with the same compensated arithmetic. At one million rows, scalar controls used
roughly 4–5% less peak process memory after callback storage moved out of graph
nodes; batch peak memory was essentially unchanged. Peak RSS is not an allocation
count. These are repeated deterministic fits, not statistical calibration or an
inference-engine comparison.

The existing `tools/downstream-conformance/python/benchmark_regression.py` accepts
`--baseline-binary` and `--baseline-revision` to compare two built revisions.
It seals a job plan before execution, records binary hashes and CPU affinity,
retains process records alongside the summary, and withholds timing ratios when
any repeated fit fails its numerical or agreement gates. Build both binaries with
the same toolchain and dependency lockfile; report setup, evaluation and fitting
separately. Baseline labels are supplied by the caller, so retain the build records
that establish which source produced each binary.

### Attributing fitting time

Pass `--attribution` to the regression benchmark runner to compare plain,
call-counted and timed target evaluation using one binary. The observer lives in
the downstream benchmark package, outside Alea's production crates. It forwards
the target's concrete error type, values and gradients without changing them.
Count-only observation reads no clock. Timed observation sums returned target-call
wall durations, including parameter transforms and graph evaluation. It includes
scheduling delays; it is not CPU time. Output serialization is outside fit time.

A pinned five-repetition run completed all 270 planned processes. All 36
observer/control comparisons passed the existing numerical gates and retained
identical numerical outputs and optimizer work counts. Independent 60-digit
reference checks passed. The table shows median target-call fractions for batch
fits, without subtracting a clock-overhead estimate.

| Likelihood | 10 rows | 1,000 rows | 1,000,000 rows |
| --- | ---: | ---: | ---: |
| Gaussian | 45.9% | 96.5% | 99.984% |
| Bernoulli-logit | 37.3% | 97.6% | 99.994% |
| Poisson-log | 32.7% | 95.6% | 99.984% |

The residual includes optimizer work, checks and observer bookkeeping. Small-fit
controls were too variable to establish a precise observer cost: some timed
medians were lower than plain medians. Their ratios remain in the report, rather
than being subtracted as a correction. These one-host measurements identify the
large-data target-evaluation bottleneck; they do not establish exact uninstrumented
fractions, a general speedup, or results for hierarchical models and samplers.

### Sharing likelihood and score arithmetic

Bernoulli-logit modeling evaluates the log probability and its log-odds score
through one distribution kernel. For response sign `s` (one for true, minus one
for false) and log odds `eta`, they are `-softplus(-s*eta)` and
`s*sigmoid(-s*eta)`. The elementary helper
`alea::math::numeric::softplus_and_sigmoid` shares one exponential between these
functions. It does not subtract a rounded probability from one, so representable
tail scores remain available. Analytic distribution jets use the same helper;
density-only evaluation keeps its separate path. Predictor products and score
reductions retain compensated accumulation.

Instruction profiling identified the duplicated exponential. On the measured
Bernoulli batch workload, fusion reduced total Callgrind instructions by about
13%, from 523 million to 455 million. Those are instruction counts under
instrumentation, not native execution times. Hardware and software perf sampling
were unavailable under the host's permissions; no system settings were changed.

A separate pinned native comparison completed all 252 planned processes and all
18 family/size/path comparisons passed their numerical gates. Numerical outputs
and optimizer work counts were identical. At one million rows, Bernoulli batch
evaluation was 1.20 times faster and fitting 1.21 times faster; scalar fitting was
1.09 times faster. Small and medium cases were mixed: Bernoulli batch fitting
ratios were 0.75 at ten rows and 1.13 at 1,000 rows (previous time divided by revised
time). Unchanged Gaussian and Poisson controls also varied. These seven-repetition,
one-host measurements support a workload-specific improvement, not a general
performance guarantee. Independent 60-digit checks and the separate 72-case
conditioning/existence cohort passed all required accuracy gates. The latter
retains known raw-coordinate Poisson failures and distinct stopping outcomes.

The compensated predictor can also be inlined into the batch score loop. This
compiler hint leaves the offset, multiplication residuals, addition compensation
and coefficient order unchanged. Whole-process Callgrind counts fell about 10%
for Gaussian and 8.5% for Bernoulli in the profiled two-predictor workloads.
A separate 252-process native comparison preserved all numerical outputs and
work counts, and the 72-case conditioning replay passed the required gates.
Million-row Bernoulli and Poisson batch fitting ratios were 1.06 and 1.07.
Gaussian was initially slower (0.88), so a separately retained 15-repetition
Gaussian replay measured it again: evaluation and fitting ratios were both 1.03,
with the unchanged scalar fitting control at 0.99. The initial result remains
recorded. These mixed one-host timings support a modest, workload-dependent
compiler optimization, not a general speed guarantee. The runner's `--families`
option supports such targeted replays while retaining the complete declared jobs.

Wider fixed-design comparisons can be requested with `--columns`, including the
intercept. For example, after building the release benchmark:

```sh
python tools/downstream-conformance/python/benchmark_regression.py \
  tools/downstream-conformance/target/release/examples/regression_benchmark \
  regression-32.json --columns 32 --rows 1000 10000 --runs 3
```

Widths other than two use bounded, independently generated predictor columns
with a fixed Rust seed. The original two-column observations are preserved.
The runner checks the reported width and retains process failures. The independent
`check_regression_benchmark.py` consumes this report and exports the actual dataset
from the same executable, verified by its hash. Run it in the analysis Nix shell;
it checks initial and final values and scores against NumPy/SciPy, and estimates
against SVD least squares or analytic-score root finding. Solver status, objective
accuracy, predictor agreement and Alea's stopping condition remain separate checks.

A one-host comparison at 2, 8 and 32 coefficients, 1,000 and 10,000 observations,
and three repetitions completed all 108 planned fits. All 18 scalar/batch cohorts
met their shared score and agreement gates. Independent checks passed 102 of 108:
the six Bernoulli fits at 32 coefficients and 1,000 observations had predictor error
`5.76e-6`, exceeding the `5e-6` reference gate despite satisfying the requested
`n * 1e-7` score tolerance. Initial and final score errors were below `5e-14` in
that case. A small score alone does not establish a requested prediction accuracy;
these six outcomes are retained as failed reference checks.

At 10,000 observations and 32 coefficients, all independent checks passed. Median
scalar time divided by batch time was 4.54 for Gaussian fitting, 4.32 for Bernoulli
and 4.98 for Poisson; peak resident memory ratios were about 7.2. These compare
batch factors with scalar expression graphs, not different inference engines or
versions of the compiler hint. Three repetitions on one host do not establish a
general speed guarantee. No quality-qualified speed claim is made for the failed
1,000-observation Bernoulli case.

The stopping-policy replay can be requested explicitly:

```sh
python tools/downstream-conformance/python/benchmark_regression.py \
  tools/downstream-conformance/target/release/examples/regression_benchmark \
  regression-32-tighter.json --columns 32 --rows 1000 --runs 3 \
  --score-tolerance-per-observation 1e-8
```

This sets the infinity-norm score threshold to `n * 1e-8`; it changes neither the
objective nor the independent accuracy gates. The runner records the requested
policy and rejects outputs reporting a different threshold. The default remains
`n * 1e-7` so that the original experiment remains reproducible.

A separate replay at 32 coefficients and 1,000 observations retained both policies
across all three families and both execution paths. The original policy reproduced
the six Bernoulli reference failures; the tighter policy passed all 18 checks.
Worst predictor errors were `7.4e-9` for Gaussian, `1.4e-7` for Bernoulli and
`1.2e-8` for Poisson. Each family took two more iterations and two more objective
evaluations, with scalar and batch paths retaining the same work counts.
This replay establishes an accuracy/work tradeoff, not a timing improvement.

A score threshold bounds a derivative, not the distance to the optimum. Locally,
parameter error also depends on inverse curvature; predictor error further depends
on the design matrix. Dividing the threshold by observation count does not remove
these dependencies. Use independent prediction checks and an explicitly chosen
stopping policy rather than interpreting `ScoreTolerance` as an accuracy guarantee.

A separate compiler-hint comparison used identical benchmark sources and production
code differing only by the private predictor's `#[inline]` attribute. It covered
8 and 32 coefficients, 1,000 and 10,000 observations, all three families, both
execution paths and three repetitions. All 144 fresh-process runs completed;
all 24 before/after cohorts had identical numerical outputs and optimizer work
counts. Independent SciPy checks passed for every run. Execution was randomized,
pinned to one CPU, and used `n * 1e-8` score tolerance throughout.

At 10,000 observations, median fitting ratios (without hint divided by with hint)
were:

| Coefficients | Gaussian | Bernoulli | Poisson |
| --- | ---: | ---: | ---: |
| 8 | 1.02 | 0.92 | 1.03 |
| 32 | 1.08 | 0.89 | 1.05 |

Across both observation counts, batch fitting ratios ranged from 0.89 to 1.26;
scalar controls ranged from 0.86 to 1.04. This mixed three-repetition evidence
does not establish a general wider-model speedup. The hint remains a small
compiler choice supported by earlier instruction profiles, not a reason to change
the numerical algorithm or introduce stronger forced-inlining rules.

For revision reports, pass `--revision before` or `--revision after` to
`check_regression_benchmark.py`, along with that revision's retained executable.
The checker verifies its binary hash and the complete declared set of logical
jobs before evaluating references. Missing, duplicate and unexpected jobs are
rejected even when total counts happen to match. Reference outcomes are recorded
separately for each revision; failures are not removed from the denominator.

### Running many independent fits

Independent fits can use separate model instances in a caller-owned Rayon pool.
Construct the model and its evaluation scratch inside each task, so concurrent
fits never share mutable evaluator state. Bound the pool size explicitly and
retain a stable input ID with every outcome. A faster batch does not make an
unsuccessful fit statistically usable.

The downstream benchmark exercises this composition without introducing a new
public scheduling API:

```sh
python tools/downstream-conformance/python/benchmark_regression_throughput.py \
  tools/downstream-conformance/target/release/examples/regression_benchmark \
  regression-throughput.json --columns 32 --rows 1000 10000 \
  --fits 12 --threads 1 2 4 --runs 3
```

Here one thread means direct sequential execution, with no pool. Two and four
threads use a locally constructed pool; global Rayon configuration is untouched.
Only task IDs are queued. At most `min(fits, threads)` model workspaces are live;
result storage grows with the number of fits. This is a workspace-count bound,
not a byte-level memory limit. Reported resident memory is process-wide, not
per-fit memory.

Each task builds the same deterministic regression likelihood independently,
evaluates its initial value and score twice, then runs L-BFGS with `n * 1e-8`
score tolerance. This workload measures repeated model construction and fitting,
not shared-data reuse, bootstrap resampling or a simulation study. Process-level
throughput includes executable startup, pool creation, computation and output.
Native pool/workload times are also retained. Typed task errors retain their IDs;
malformed output and process failures retain the full planned denominator.

A four-core, one-host comparison covered 8 and 32 coefficients, 1,000 and 10,000
observations, all three families, 12 fits per process and three repetitions.
All 108 processes and 1,296 fits completed. Numerical outputs and optimizer work
counts were identical across execution policies, and independent SciPy checks
passed for every fit. At 10,000 observations, process-throughput ratios relative
to direct sequential execution were:

| Coefficients | Family | Two workers | Four workers |
| --- | --- | ---: | ---: |
| 8 | Gaussian | 1.49 | 2.65 |
| 8 | Bernoulli | 1.93 | 2.97 |
| 8 | Poisson | 1.73 | 2.50 |
| 32 | Gaussian | 1.93 | 3.32 |
| 32 | Bernoulli | 1.94 | 3.55 |
| 32 | Poisson | 2.02 | 3.39 |

These are aggregate fit-throughput gains, not faster individual likelihoods or
single-fit latency guarantees. Small-job gains varied more, and three repetitions
on one host do not establish a universal crossover. An earlier cohort with a
one-worker-pool baseline is retained separately and was not pooled into this
comparison. Thread budgets must also account for any parallel work inside a
user-supplied target; these native regression targets use no nested BLAS pool.
