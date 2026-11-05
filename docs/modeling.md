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
use alea::distributions::modeling::ModelBuilderExt;
use alea::{
    model::assembly::{AssemblyError, CompiledModel},
    target::LogDensityGradient,
};

fn model(observations: &[f64]) -> Result<CompiledModel, AssemblyError> {
    alea::model! { |builder|
        parameters { mean: real(), scale: positive() }
        {
            builder.normal_prior("mean prior", &mean, 0.0, 2.0)?;
            // log Exponential(scale | rate = 1) = -scale for scale > 0.
            let scale_prior = builder.multiply(-1.0, &scale)?;
            builder.prior("scale prior", scale_prior)?;
            builder.normal_observations("observations", observations, &mean, &scale)?;
        }
    }
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

## Concise syntax, ordinary Rust

`model!` is a small construction macro, not a separate modeling language or AD
backend. `parameters { mean: real(), scale: positive() }` passes the binding
names to the existing typed builder methods. The following block is ordinary
Rust: use loops, conditionals, helper functions and `?`. The macro returns
`Result<CompiledModel, AssemblyError>` and compiles only after the block succeeds;
the block's final value must be `()`. It never infers a prior, adds a hidden
Jacobian, chooses an inference algorithm or runs a sampler. Use the builder
directly, or `model!(|builder| { ... })`, for explicit/custom declarations.

Arithmetic and Normal/Bernoulli-logit factor constructors accept `f64` literals,
`Expr`, and borrowed or owned scalar parameter handles. No `.expr()` or
`constant()` call is needed for those arguments. Use floating literals (`0.0`,
not a count `0`); discrete events remain typed `bool`/`u64`. An existing expression
is reused, while each literal inserts an ordinary constant node. To deliberately
share a constant, bind `let zero = builder.constant(0.0)?;` once. Arguments retain
builder ownership and their original charts; invalid literals/foreign handles
fail before the input list adds any nodes.

`ModelBuilderExt` (also in `alea::prelude`) provides named `normal_prior`,
`normal_likelihood`, `bernoulli_logit_likelihood`, and the three native
`*_regression_likelihood` methods. Each constructs the existing native factor and
registers its explicit role. Unknown support/truncation normalizers are not
inferred: `normal_prior` on a positive event does not add the half-normal `ln(2)`.
As with other fallible construction sequences, discard an unfinished builder on
error: helpers can leave unused expressions but never attach a failed named term.

`normal_observations` uses the fused shared-location/scale regression kernel for
a nonempty observation slice. Its graph size is independent of row count, though
owned data storage remains linear. It declares one **joint factor**, not per-row
leave-out units. An explicit native batch rule now provides conditional rows via
`CompiledModel::batch_likelihood()`, and a lazy source rule emits counted Rust
loops. Neither route changes the posterior measure or infers marginal leave-out
units. See [tensor regions and batch lowering](tensor-ir.md).

For a non-regression Normal observation map, use vector locations/scales directly:

```rust
use alea::{model::assembly::ModelBuilder, distributions::modeling::ModelBuilderExt};
let mut builder = ModelBuilder::new();
let means = builder.real_vector("means", &["a", "b", "c"])?;
let sigma = builder.positive("sigma")?;
builder.normal_vector_likelihood(
    "data", &[0.2, -0.4, 1.1], means.expressions(), &[sigma.expr()],
)?;
let model = builder.compile()?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Each location/scale slice has length one or the observation count. The native
`NormalBatch` primitive selects packet/scalar execution automatically, caches
shared scale logarithms, and reuses derivative storage. It keeps one graph node
for the joint density; conditional rows remain explicit through
`batch_likelihood()`. Fixed responses can be replaced with the existing consuming
`CompiledModel::rebind` and `NormalBatch::with_observations` interfaces.

For differentiable event vectors, `modeling::normal_batch` accepts event,
location and scale expression slices with the same broadcasting rule. It does
not infer priors or add Jacobians: parameter declarations still own those charts.
All local scalar scores are validated, including fixed-event scores. Broadcast
scores and likelihoods use row-ordered compensated sums; separate scalar graph
reductions need not be bit-identical. Arbitrary `plate` closures are **not**
automatically rewritten into vector factors. Existing `normal_observations`
continues to use compressed regression when all observations share one mean.

For regression, named vector declarations are convenient:

```rust
use alea::{distributions::modeling::ModelBuilderExt, model::assembly::ModelBuilder};
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut builder = ModelBuilder::new();
let beta = builder.real_vector("beta", &["intercept", "slope"])?;
for (i, &coefficient) in beta.expressions().iter().enumerate() {
    builder.normal_prior(&format!("beta prior {i}"), coefficient, 0.0, 2.0)?;
}
// Pass beta.expressions() and owned RegressionData to the appropriate
// *_regression_likelihood method. No intercept column is silently inserted.
# Ok(()) }
```

`positive_vector` and `unit_interval_vector` work analogously. Their dimension
and geometry-block labels default to the vector name; `vector` remains available
for explicit labels and simplex/matrix/custom charts. Macro declarations accept
arguments too: `beta: real_vector(&["intercept", "slope"])`. Construction helpers
do not claim faster evaluation or change batch kernels, mathematical validation,
factor selection or inference defaults.

## Reuse model functions with scoped names

Use ordinary Rust functions to compose models. A scope qualifies parameter and
factor names without creating another builder, copying a graph or changing the
parameter measure:

```rust
use alea::{
    distributions::modeling::ModelBuilderExt,
    model::assembly::{AssemblyError, Expr, ModelBuilder},
};

fn component(builder: &mut ModelBuilder, observations: &[f64]) -> Result<Expr, AssemblyError> {
    let mean = builder.real("mean")?;
    let scale = builder.positive("scale")?;
    builder.normal_prior("mean prior", mean, 0.0, 2.0)?;
    let prior = builder.multiply(-1.0, scale)?;
    builder.prior("scale prior", prior)?;
    builder.normal_observations("data", observations, mean, &scale)?;
    Ok(mean.expr())
}

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut builder = ModelBuilder::new();
let first = builder.with_scope("first", |b| component(b, &[0.2, 0.7]))?;
let second = builder.with_scope("second", |b| component(b, &[1.0, 1.2]))?;
let difference = builder.multiply(-1.0, second)?;
let difference = builder.add(first, difference)?;
let model = builder.compile()?;
assert_eq!(model.coordinates()[0].name, "first::mean");
assert!(model.graph().any(|node| node.expression == difference));
# Ok(())
# }
```

Nested scopes compose (`first::inner::mean`). Scope components must be nonempty
and cannot contain `::`; existing declarations still reject name collisions.
Dropping a scope restores its enclosing namespace, including on panic. It does
not roll back declarations or invalidate escaped handles. Named vector helpers
also scope their default dimension and geometry-block labels, avoiding accidental
merging between components. Explicit `vector`/`annotate` metadata stays explicit.

## Inspect mathematical operations and backend support

Native `normal` and `normal_regression` operations retain immutable typed
descriptors, rather than retaining only executable callbacks. Existing free
functions, named builder helpers and `model!` all use this same registration path.
`CompiledModel::graph()` exposes scalar dependencies, constants, factor roots
(including repeated declarations), current-view activity and optional primitive
metadata. Inspection is a cold operation: input-handle vectors are allocated;
captured datasets are borrowed, not copied. The retained analytic evaluator does
not construct graph snapshots or inspect signatures.

```rust
use alea::{
    distributions::modeling::{ModelBuilderExt, regression::NormalRegression},
    model::assembly::{ArgumentShape, ModelBuilder},
};
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut builder = ModelBuilder::new();
let mean = builder.real("mean")?;
let scale = builder.positive("scale")?;
builder.normal_observations("data", &[0.2, 0.7], mean, &scale)?;
let model = builder.compile()?;
let node = model.graph().find(|node| node.primitive.is_some()).unwrap();
let metadata = node.primitive.unwrap();
let regression = metadata.descriptor::<NormalRegression>().unwrap();
assert_eq!(regression.data().rows(), 2);
assert_eq!(metadata.signature().arguments[0].shape, ArgumentShape::Vector(1));
assert_eq!(metadata.signature().probability.as_ref().unwrap().batch, [2]);
assert!(model.source_capabilities().supports_rust_primal());
assert!(model.rust_primal()?.contains("for i in 0.."));
# Ok(())
# }
```

Logical argument shape, probability event shape and batch shape are distinct. A
normal regression has a coefficient-vector argument, scalar scale, scalar events
with one-dimensional Lebesgue measure, and a row batch. Its typed descriptor
preserves the original design, responses and offsets. It still lowers to **one
fused batch kernel**, not one scalar graph per observation. Scalar normals retain
their existing Rust primal; all three native regression families now have lazy,
counted-loop source lowering and explicit conditional batch rules.
`source_capabilities()` reports every missing active callback primal, while actual
export preserves `MissingPrimal`. Statistical selections retain inactive
descriptors but do not require their source primals.

For custom mathematical operations, implement `Primitive`: declare logical
arguments, retain immutable data, provide separate execution-local workspace and
delegate fused value/local-score evaluation to your implementation. Register with
`builder.primitive(operation)`. Registration validates argument counts, shape
overflow, names and handle ownership. Arbitrary existing `custom` callbacks remain
supported and are reported as opaque; their semantics are never guessed.

The scalar assembly interface is not a generative language. A primitive signature
is its author's contract, not proof of arbitrary
callback normalization or posterior propriety. Neither a batch axis nor a native
distribution descriptor invents a plate, leave-out partition, prior, truncation
normalizer, predictive generator or higher derivative. A separate
[tensor/control-flow IR](tensor-ir.md) supports fixed-shape blocks, explicit lazy
regions, fixed-trip scans and immutable rebinding. Registered primitives can be
rebound by consuming the compiled model, preserving its topology. Rust `if`/loops
in ordinary model functions still execute at construction time; staged control
flow uses explicit IR operations. Jacobian rules and selections are unchanged.

The opt-in `model_semantics` Criterion benchmark compares descriptor registration
against explicit opaque lowering of the **same native primitive and kernel**,
separately timing construction and fused evaluation at 1,000 and 10,000 rows:

```bash
nix develop .#stable --command cargo bench --manifest-path crates/Cargo.toml \
  -p alea-distributions --bench model_semantics --locked
```

This measures interface overhead, not a historical before/after speedup or
statistical efficiency. Correctness/allocation tests run independently of timings;
no machine-dependent wall-time assertion is a CI correctness gate.

### Local semantic-interface overhead check

On 2026-11-01, three sequential benchmark processes (forward, reverse, forward
order) on an x86-64 Intel i5-8250U produced the following microsecond timings.
Each cell is the median of three Criterion run-median estimates, with their
minimum–maximum in parentheses—not an across-process confidence interval.

| Operation | Rows | Opaque lowering, µs | Semantic registration, µs |
|---|---:|---:|---:|
| Construct and destroy | 1,000 | 7.099 (7.079–7.165) | 7.424 (7.094–7.608) |
| Construct and destroy | 10,000 | 60.520 (59.940–62.566) | 60.708 (60.066–60.745) |
| Fused value/score | 1,000 | 15.081 (14.984–15.112) | 15.275 (15.000–15.679) |
| Fused value/score | 10,000 | 150.068 (149.440–150.735) | 150.621 (149.211–151.713) |

Semantic evaluation point estimates are about 1.3% and 0.4% higher, respectively;
the process ranges overlap. These short laptop measurements do not establish
zero overhead or a general performance difference. Construction includes data
validation/preparation, graph/scratch allocation and destruction, but reuses
already-owned response/design arrays through `Arc`. Evaluation uses the same
native kernel with two coefficients and constrained sigma=1.1; it excludes
transforms, priors, inference, process startup and serialization. Both paths
retain four execution nodes. Independent tests require zero allocations during
retained evaluation and bit-identical values/scores.

Protocol: lock-pinned stable Rust 1.95.0/LLVM 22.1.2, workspace release profile
(opt-level 3, fat LTO), no extra `RUSTFLAGS`, one model-evaluation thread, Criterion
30 samples/case, 0.5-second warmup and 2-second measurement. This was a shared
desktop with the `powersave` governor, without affinity or frequency control.
The initial exploratory process is not pooled into the three-process summary.
Reproduce the named baselines using the benchmark command above, adding
`-- --noplot --save-baseline semantic-forward-1`; for the middle process set
`ALEA_SEMANTICS_REVERSE=1` and name it `semantic-reverse-1`, then run forward again
as `semantic-forward-2`. Criterion's local `estimates.json` files retain the
run estimates and within-process bootstrap intervals.

Benchmark source SHA-256:
`3108989662923b5b34d855ee4ea3b37870f09f4612f011a3a852c79042af2a90`.
This is a checkout-level overhead check, not a sealed historical benchmark,
Stan comparison, sampler qualification or speed improvement claim. Scalar-heavy
graphs and multi-chain construction overhead were not measured by this check.

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

Eligible tall Gaussian batches use prepared square-root sufficient statistics.
This also applies to conditional mixed effects expressed as fixed group-indicator
columns with deterministic coefficient expressions such as `tau * z_group`;
latent coordinates and their priors are not integrated out. Poisson batches with
exactly identical design rows **and offsets** use count sufficient statistics,
retaining the full log-factorial constants. Distinct predictors retain their
fused row kernel. Both optimizations retry original rows at numerical limits;
rebinding refreshes preparation, while pointwise reporting/source keep all rows.
These paths change floating-point rounding, not the model. See the
[sufficient-statistics case study](native-model-performance.md#14-applying-the-sufficient-statistics-paper).

The factors retain one graph node per batch regardless of row count; dataset
storage still grows with the observations. Values, predictors and score reductions
use compensated summation. Score storage, proportional to the number of
coefficients, is allocated at construction and reused without evaluation allocations.
The analytic scores work on stable Rust. Native descriptors also supply lazy
counted-loop Rust source rules; those are separate from native SIMD execution
and do not establish arbitrary-program compiler-derivative support.

Each returned factor is joint over its batch. Explicit native row rules can be
partitioned with `PointwiseLikelihood::from_batch(batch, units, rows)` without
scalarizing the sampling graph. `rows[i]` contains the original `BatchUnit`s
assigned to `units[i]`; every actual row must occur exactly once. Observation IDs,
grouping and conditioning remain caller declarations, not marginalization.
The result works with `AnalysisSpec::from_likelihood`. Batch-row reports emit
manifest version three with explicit factor/local-row indices and complete row
counts; whole-factor reports retain version two. The strict Python reader
validates both formats and rejects missing/overlapping rows, inconsistent counts
and ambiguous conditioning. Older readers reject version three rather than
silently interpreting repeated factor names. Chunked batches produce the same mathematical
likelihood, with ordinary floating-point reduction differences.

### Inspect or diagnose native CPU execution

The existing regression helper calls automatically prepare a fixed CPU plan.
The typed descriptor's `execution_plan()` reports the requested policy, selected
f64 lane count, row/column dimensions, four-row tile, additional packed-design
bytes and broadcast eligibility. Inspect descriptors through `model.graph()` as
shown above. A lane count is not a measure of SIMD utilization: tails, scalar
transcendentals and numerical fallbacks can still require scalar work.

For a controlled comparison, select scalar execution explicitly before registering
the descriptor. This uses the **same fused kernel**, not a second distribution or
a different posterior:

```rust
use alea::distributions::modeling::regression::{
    BernoulliLogitRegression, RegressionData, RegressionExecution,
};
use alea::model::assembly::ModelBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = ModelBuilder::new();
    let beta = builder.real("beta")?;
    let data = RegressionData::new(1, vec![1.0; 5], vec![true, false, true, false, true])?;
    let primitive = BernoulliLogitRegression::new(&[beta.expr()], data)?
        .with_execution(RegressionExecution::Scalar);
    assert_eq!(primitive.execution_plan().lanes(), 1);
    assert_eq!(primitive.execution_plan().workers(), 1);
    let term = builder.primitive(primitive)?;
    builder.likelihood("observations", term)?;
    let model = builder.compile()?;
    assert_eq!(model.expression_count(), 2);
    Ok(())
}
```

`Auto` selects scalar execution when there is neither a complete four-observation
tile nor a full coefficient packet for the available SIMD width. This structural
bypass applies only to widths up to the tile size, preserving wider normal
fallbacks. `is_scalar_for_shape()` identifies this decision. Otherwise it caches
Pulp's CPU selection at construction; it does not use fitted timing thresholds,
autotuning, observation threading or Faer kernel selection.
`Scalar` is a diagnostic backend, not a strict row-wise reduction policy or a
cross-ISA bitwise replay guarantee. Both default to compensated arithmetic,
with the same error checks and reusable score buffers. Native descriptors also
accept `.with_arithmetic(RegressionArithmetic::Standard)` to opt into conventional
f64 predictor, likelihood and score accumulation independently of CPU dispatch.
This can lose meaningful digits under cancellation; it is not accuracy-equivalent
to the default and has no established universal speed advantage. Gaussian and
Poisson sufficient-statistics algorithms remain unchanged. `with_data` preserves
both policies while rebuilding data-dependent packing/broadcast decisions.
Standard descriptors explicitly reject source AD export; other source backends
retain their existing rules. See the [arithmetic experiment](kernel-optimization.md#compensated-versus-conventional-f64-regression-arithmetic)
for numerical counterexamples and the remote benchmark procedure.
See [the kernel methodology](kernel-optimization.md#native-regression-execution-plans)
for separate construction/evaluation benchmarks and the remaining selection gates.

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

## Export a point-estimation report

With the optional `analysis` feature, `analysis::write_fit` exports a `NamedFit`
as one UTF-8 JSON document (`alea-fit`, version 1). This is separate from the
sampling-table format: it has no chain/draw axes, fabricated posterior samples,
covariance or standard errors. It retains the original objective kind and measure,
named constrained estimates, free-coordinate position and score, numerical policy,
work counters, exact termination and both line-search failure reasons. An absent
candidate remains null; a retained candidate after failure is not a successful fit.

```rust
use alea::{
    analysis::{write_fit, Provenance},
    optimize::NamedFit,
};
use std::collections::BTreeMap;

fn export(fit: &NamedFit) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let objective = &fit.result().objective;
    let provenance = Provenance {
        model: objective.model.clone(),
        data: objective.data.clone(),
        source: objective.source.clone(),
        method: "L-BFGS".into(),
        settings: BTreeMap::new(),
    };
    let mut bytes = Vec::new();
    // No assessment was attached: encode explicit JSON null.
    write_fit(&mut bytes, fit, &provenance, |_| Ok(Default::default()))?;
    Ok(bytes)
}
# fn main() {}
```

For `NamedFit<A>`, the last argument explicitly converts `&A` to a
`Result<serde_json::Value, AnalysisError>`. Choose and document the assessment's
own schema; the writer does not interpret arbitrary evidence or infer its match
to the fitted model. Separately retained backend error details can be included
there: `OptimizationResult` preserves the failure classification, not the original
typed error itself. Model/data/source identities must match the objective's
declaration; labels and matching dimensions do not verify content digests.

Metadata, counters, stopping evidence and assessment encoding are checked before
any write. Callback errors or panics leave the stream untouched. I/O or flush
failure may leave a partial document: discard it, and use an application-owned
temporary file plus atomic installation when durable replacement is required.
The report is not a checkpoint and does not reconstruct the chart or optimizer.
The source-checkout reader `tools/downstream-conformance/python/fit_analysis.py`
validates versions, dimensions, finite numbers, exact integer counters, duplicate
or unknown fields and stopping evidence, retaining failed outcomes and assessments.
It needs only Python's standard library and does not create an ArviZ posterior.

## Check likelihood curvature explicitly

With `faer`, `optimize::information::observed_information` computes numerical
observed-likelihood covariance for a `NamedFit` meeting its declared score
tolerance. Supply the same **no-Jacobian likelihood objective** through the
optional `PotentialHessianVector` capability. Ordinary compiled graph targets do
not automatically provide HVPs: an analytic provider or generated Enzyme
**objective** is required. There is no implicit finite-difference fallback, and
the L-BFGS history is not used as an uncertainty estimate.

```rust
use alea::{
    capability::PotentialHessianVector,
    math::information::InformationOptions,
    optimize::{NamedFit, information::{observed_information, ObservedInformation,
                                     ObservedInformationError}},
};

fn curvature<T: PotentialHessianVector>(
    objective: &T, fit: &NamedFit,
) -> Result<ObservedInformation, ObservedInformationError<T::Error>> {
    // Explicit numerical policy in the supplied free-coordinate units.
    let policy = InformationOptions {
        relative_eigenvalue_cutoff: 1e-12,
        relative_symmetry_tolerance: 64.0 * f64::EPSILON,
    };
    observed_information(objective, fit, policy)
}
# fn main() {}
```

The operation replays the candidate's value and score exactly, then uses one
checked HVP per free coordinate to obtain the negative likelihood Hessian.
Replay detects many incorrect-objective substitutions but cannot prove that
arbitrary callbacks or their Hessians describe the declared model. A posterior
fit, failed search, absent candidate or changed objective rejects explicitly.
Errors and panics leave the original fit unchanged; no partial covariance is
returned. No eigenvalue floor, ridge, pseudoinverse or automatic rank truncation
turns failed curvature into uncertainty.

Both triangles must meet the declared symmetry tolerance relative to the
matrix's maximum absolute entry, with no absolute unit-dependent floor;
admitted pairs are averaged (zero tolerance requires equality). Max-absolute
scaling precedes a sequential Faer symmetric eigensolve. All estimated eigenvalues
must be positive and their minimum/maximum ratio must exceed the chosen cutoff.
This is a numerical, unit-dependent full-rank policy, not an exact rank or error
certificate. Existing Cholesky factorization and triangular solves form covariance;
the implementation never calls a dense inverse routine.

`free_covariance()` is in the original free coordinates, not the optimizer's
scaled coordinates. `constrained_covariance().values` is a full row-major matrix
in constrained parameter metadata order. Its projection is the first-order delta
calculation `J I_free^-1 Jᵀ`, using the analytic chart Jacobian without a density
adjustment. A simplex may therefore produce a singular physical covariance with
more rows than free coordinates. This is not exact nonlinear uncertainty
propagation. Nonrepresentable division, solve or covariance output rejects;
the current implementation is not a full-range floating-point enclosure.

A small score and positive information do **not** establish a finite/global MLE,
regularity, empirical identifiability, confidence-interval coverage or model
adequacy. At an only approximately stationary point, chart curvature still
contains score-dependent terms. No intervals, residual-variance correction or
standard errors are inferred. Keep the original fit, existence/rank evidence and
statistical assumptions alongside this report; an explicit fit-export assessment
callback may encode the numerical report without promoting those assumptions.

For supplied analytic information matrices, the lower-level Rust API
`math::information::InformationMatrix` supports the same checked factorization
and covariance projection without Enzyme. A generated native Gaussian example
in `tools/enzyme-conformance/tests/information.rs` verifies unit physical variance
for an unknown positive location with known unit observation scale, excludes a
deliberately invalid prior and rejects substitution of the sampling target.
This is scoped numerical evidence, not a general calibration study.

The native unknown-mean/unknown-scale Gaussian check also compares all HVP entries
with the analytic Hessian and both covariance matrices with their MLE references.
It caught an initial entry-relative symmetry rule: cross-curvatures near `2e-8`
differed by about `2.4e-17` from cancellation, despite each agreeing with the
analytic value within `3e-16`. The final matrix-scaled rule admits that roundoff,
while regression tests retain exact-symmetry mode and reject larger asymmetry.

Reference conventions: [Stan's optimization Jacobian policy](https://mc-stan.org/docs/cmdstan-guide/optimize_config.html)
distinguishes constrained-measure optimization from transformed-density modes;
the spectral implementation uses [Faer's eigensolver](https://github.com/sarah-quinones/faer-rs),
with version 0.24.0 pinned by the workspace and generated-curvature test locks.

## Distinguish Gaussian ML scale from residual variance

For independent, homoskedastic Gaussian errors, fixed known offsets, a full-rank
fixed design with `p` unconstrained coefficient columns, and no weights, priors
or penalties, the likelihood fit estimates `sigma_ML² = RSS/n`. The intercept,
when supplied, counts as a coefficient. The residual variance estimate instead
uses `s² = RSS/(n-p)`; it requires `n > p`. The native fit and Gaussian
assessment deliberately retain the ML scale. Do not replace it with `s` inside
the likelihood or relabel it as an unbiased standard deviation.

At the Gaussian MLE the coefficient block of inverse observed information is
`sigma_ML² (XᵀX)⁻¹`. The usual residual-variance coefficient covariance estimate
is `s² (XᵀX)⁻¹`, or `n/(n-p)` times that coefficient block. This specific
correction is **not** a rule for all parameters: in particular, do not multiply
the joint information variance of `sigma` by the same factor. These distinctions
follow the [R linear-model summary convention](https://www.stat.ethz.ch/R-manual/R-devel/library/stats/html/summary.lm.html).

The downstream qualification consumer
`tools/enzyme-conformance/examples/gaussian_uncertainty.rs` exercises this
distinction using Alea's public native batch normal regression, the existing SVD
assessment, and generated HVPs of the equivalent scalar-native likelihood.
It does not implement another distribution, gradient or optimizer. Three dyadic
fixtures retain nonzero coefficient cross-covariance and known offsets while
rescaling predictor and response units. Each fixture checks numerical rank two,
six observations, four residual degrees of freedom, both fitted coefficients,
ML sigma, same-coordinate batch/generated value and score, the complete joint
covariance, and the separately corrected coefficient block. All generated HVP
columns are also checked against central differences of the native batch score
as a model-level diagnostic, not a derivative fallback. In the base fixture,
`RSS=12`, `sigma_ML²=2` and `s²=3`; these values must not be interchanged.
Separate tests retain rank-deficient, perfect-fit and zero-residual-df cases as
unqualified rather than returning manufactured uncertainty.

The standard-library checker independently solves these two-column fixtures
using exact `Fraction` arithmetic. Its optional SciPy mode executes
[LAPACK column-pivoted QR](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.qr.html)
and triangular solves, including unpivoting the covariance. Neither Alea's SVD
coefficient estimate nor this QR reference solves floating-point normal equations.
From the repository root:

```sh
nix develop .#autodiff --command cargo test \
  --manifest-path tools/enzyme-conformance/Cargo.toml --release \
  --features faer --example gaussian_uncertainty --locked
# Build the runnable consumer, then use the optional external analysis shell.
nix develop .#autodiff --command cargo build \
  --manifest-path tools/enzyme-conformance/Cargo.toml --release \
  --features faer --example gaussian_uncertainty --locked
set -o pipefail
tools/enzyme-conformance/target/release/examples/gaussian_uncertainty | \
  nix develop .#analysis --command python3 \
  tools/enzyme-conformance/python/gaussian_uncertainty.py --scipy
```

The checker requires all three case identities, records input/checker hashes and
external package versions, and rejects malformed, incomplete or numerically
inconsistent output. This is a deterministic qualification cohort, not a general
uncertainty API, random-design study or coverage experiment. It does not qualify
constrained coefficients, penalized/hierarchical fits, heteroskedasticity,
correlated errors or arbitrary conditioning. Neither an information matrix nor
this residual correction supplies a confidence level or a critical-value law;
The explicit Gaussian interval method below applies only under its stated
assumptions; broader application qualification remains open. No interval is
silently inferred from `NamedFit`.

## Request marginal Gaussian coefficient intervals

With `faer`, fixed-design data can explicitly request Student-t coefficient
intervals. This is a separate least-squares fit, not an interpretation of an
arbitrary `NamedFit` or a sampling posterior:

```rust
use alea::distributions::modeling::regression::{
    RegressionData, diagnostics::GaussianIntervalOptions,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = RegressionData::new(
        2,
        vec![1.,-2., 1.,-1., 1.,0., 1.,1., 1.,2., 1.,3.],
        vec![4.,0.25,2.5,1.75,-2.,0.25],
    )?;
    let result = data.gaussian_coefficient_intervals(GaussianIntervalOptions {
        confidence_level: 0.95,
        relative_rank_tolerance: 1e-12,
        relative_residual_tolerance: 1e-12,
        quantile_relative_tolerance: 1e-8,
    })?;
    assert_eq!(result.assessment.least_squares.residual_degrees_of_freedom, 4);
    assert!((result.residual_variance - 3.).abs() < 1e-12);
    // ML sigma is still sqrt(RSS/n), not the corrected residual standard error.
    assert!((result.assessment.scale_candidate.ok_or("missing ML scale")?
             - 2_f64.sqrt()).abs() < 1e-12);
    let intercept_interval = result.bounds[0]; // [lower, upper], design-column order
    assert!(intercept_interval[0] < 1.5 && intercept_interval[1] > 1.5);
    Ok(())
}
```

Calling this method declares independent homoskedastic Gaussian errors, a fixed
full-column-rank design and fixed known offsets, with unconstrained coefficients
and no priors, weights or penalties. The code cannot establish those statistical
assumptions from the stored numbers. The intercept, if present, counts in `p`.
The result uses `beta_j ± t_(n-p, 1-alpha/2) * SE_j`, consistent with
[R's Gaussian linear-model intervals](https://www.stat.ethz.ch/R-manual/R-devel/library/stats/html/confint.html).
These are **marginal**, not simultaneous, intervals; they are not prediction,
sigma, hierarchical, penalized or asymptotic-Wald intervals.

One sequential thin SVD supplies both coefficients and residual-scaled covariance
through `math::least_squares::solve_with_residual_covariance`. Residual scaling
precedes its Gram calculation, avoiding an overflowing unscaled inverse-design
covariance intermediate. A checked Cholesky on max-absolute-scaled output rejects
rounded covariance that fails numerical positive definiteness. No floating normal
equations, inverse routine, ridge, rank-truncated covariance or optimizer history
is used. The assessment-only API keeps its original truncated-SVD policy; the
interval path requires full numerical column rank and positive residual df.

The existing pinned Statrs Student-t implementation supplies critical values.
It evaluates the lower tail `(1-confidence_level)/2`, avoiding an upper-tail
probability that rounds to one, and checks survival-probability replay under the
explicit relative tolerance. This check is not an independent quantile accuracy
certificate. Independent SciPy references test residual df `1,4,9,29` and levels
`0.90,0.95,0.99`; the downstream QR consumer checks all endpoints for the three
unit-rescaled fixtures. Near-perfect fits, deficient rank, no coefficients/df,
invalid policy, failed quantile replay, nonfinite/underflowed covariance or
residual variance, and endpoints that overflow or collapse to the estimate are
errors. No partial interval report is returned, and original numerical errors
remain available through their typed source chain. Representation checks are
conservative, not a full-range guarantee.

A fixed-seed coverage smoke test uses Alea's native Normal sampler and public
interval API: `4096` replications for each `(n,p)=(6,1),(12,2),(32,3)`, two levels
`0.90,0.95`, and all coefficient margins. Its twelve predeclared checks use a
Hoeffding/union-bound count allowance of about `143.72`, with family error budget
`0.001`; failed fits stay in the planned denominator and must number zero.
This familywise **test rule** does not make the returned intervals simultaneous.
The initial run passes all twelve checks. This is scoped finite-precision/RNG
evidence, not proof of general numerical coverage, arbitrary conditioning or
validity after data-dependent model selection. Reproduce it with:

```sh
nix develop .#stable --command cargo test \
  --manifest-path tools/downstream-conformance/Cargo.toml \
  --features faer --test gaussian_intervals --release --locked -- --nocapture
```

## Run a fixed-effect Gaussian workflow

The public-API consumer
[`gaussian_workflow.rs`](../tools/downstream-conformance/src/gaussian_workflow.rs)
connects native batch likelihood fitting, the separate coefficient-interval
estimator above, new-data point forecasts and point-fit export. It is a small
fixed-effect example, not qualification of Gaussian hierarchies or all regression
applications. It adds no distribution, gradient, optimizer or inference API.
Run its six planned cases through the independent standard-library Python reader:

```sh
nix develop .#stable --command bash -euo pipefail -c '
  cargo run --quiet --manifest-path tools/downstream-conformance/Cargo.toml \
    --features faer --example gaussian_workflow --locked -- all |
    python3 tools/downstream-conformance/python/gaussian_workflow.py
'
```

For a single case, replace `all` with its name and add `--case NAME` to the reader.
Each output line is an existing standalone `alea-fit` version-1 report. Its caller
assessment uses `gaussian-workflow` version 1; the generic fit schema is unchanged.
The fixture contains six observations, an explicit intercept and slope column,
known offsets and an estimated positive sigma. Native optimization selects the
likelihood objective without a density Jacobian or priors. Provenance labels
identify this fixture/version; they are not cryptographic content hashes.

Three results remain deliberately separate:

- The native result retains its termination, counters, last valid candidate and
  original evaluation-error chain. Small score alone is not an existence,
  uniqueness or uncertainty certificate.
- Marginal coefficient intervals belong to a **separate fixed-design OLS fit**
  under the preceding assumptions. They can remain available after native budget
  exhaustion, cancellation or initial evaluation failure. They never replace the
  native candidate or turn its failure into a successful native fit. The ML
  scale uses `RSS/n`; coefficient covariance uses `RSS/(n-p)`.
- New-data forecasts are labelled `plug_in_conditional_mean`, not posterior
  predictions or prediction intervals. This example requires native score
  termination, admitted OLS intervals and agreement of physical coefficients and
  ML sigma with the OLS reference under the declared `1e-5*(1+abs(reference))`
  tolerance. Otherwise all three planned forecast rows remain present with null
  means. This conservative application policy is not a general forecast API or
  proof of optimizer convergence.

The planned controls are `qualified`, `budget`, `cancelled`, `initial_transform`,
`rank_deficient` and `perfect_fit`. In the checked run, both the qualified and
rank-deficient fits reach score tolerance, but only the qualified fit passes the
reference/interval policy. Rank deficiency therefore does not silently become
qualified uncertainty. The perfect-fit likelihood is unbounded as sigma tends
to zero; the native run retains its last valid candidate and subsequent transform
boundary error. No scale floor, dropped column, prior or repaired initialization
is introduced. Error-chain strings are caller evidence, not portable typed-error
serialization. Do not rely on exact optimizer trajectories for other fixtures.

The reader independently checks the dyadic fixture using exact rational algebra,
saved SciPy Student-t critical values, physical/free-coordinate alignment,
normalized likelihood, ML-versus-residual scale, interval endpoints and new-data
alignment. It rejects missing/duplicate cases, fabricated intervals, suppressed
failure evidence, posterior relabelling, blank/truncated JSON lines and unqualified
forecasts. Native integration tests exercise the real producer; Python mutation
tests exercise malformed reports. A failed write leaves an unusable partial
report: discard it rather than treating a valid prefix as completed output.
This downstream correctness gate supplies no new timing, ESS or Stan comparison.

## Run a native logistic/Poisson point-fit workflow

The checkout [GLM workflow](../tools/downstream-conformance/src/glm_workflow.rs)
connects the public native batch factors, likelihood optimization, named
finite-attainment/rank evidence, conditional means and `alea-fit` export. It uses
two independent real coefficients with fixed design/offsets and no priors,
weights, penalties or coefficient constraints. It does not implement a density,
gradient, optimizer or sampler. Reproduce all sixteen planned family/case pairs:

```sh
nix develop .#stable --command bash -euo pipefail -c '
  cargo run --quiet --manifest-path tools/downstream-conformance/Cargo.toml \
    --features faer --example glm_workflow --release --locked |
    python3 tools/downstream-conformance/python/glm_workflow.py
'
```

To inspect one pair, append e.g. `-- poisson zero_no_intercept` to the Rust command
and `--family poisson --case zero_no_intercept` to the reader. Faer supplies the
existing SVD rank assessment; the likelihoods and optimizer remain Alea's native
implementations. Each JSON line is a standalone version-1 point-fit report with
caller `glm-workflow` version-1 evidence, not a new production inference schema.

The ordinary fixture has two groups of four rows with design columns `[1,g]`.
Logistic responses have one then three successes, with offsets `0.25` and `-0.5`.
The independent saturated-group solution is
`beta0=log(1/3)-0.25`, `beta1=2*log(3)+0.75`. Poisson counts sum to four then eight;
exposures are two then one half, supplied as **log exposures**, not exposures.
The per-unit fitted rates are therefore `0.5` and `4`, with coefficients
`beta0=-log(2)`, `beta1=log(8)`. Predicted expected counts include the new row's
exposure; they are not per-unit rates unless that exposure is one.

Every family runs every control, retaining its numerical outcome:

| Case | Separate application outcome in the checked run |
|---|---|
| `qualified` | Score termination, finite maximum, numerical full rank; point forecasts available |
| `zero_no_intercept` | All-zero responses but finite maximum and full rank; point forecasts available |
| `budget` | Evaluation budget exhausted; initial candidate retained, forecasts withheld |
| `cancelled` | No candidate; no invented successful fit |
| `evaluation_failure` | Original evaluation-error chain, no candidate; forecasts withheld |
| `rank_deficient` | Finite maximum and small numerical score, but nonidentified coefficients |
| `boundary` | Small numerical score, yet an exact improving ray proves no finite maximum |
| `unresolved` | Exact search budget exhausted; existence remains unresolved despite the small score |

The no-intercept control supplies opposing feature rows `[1,0]`, `[-1,0]`,
`[0,1]`, `[0,-1]`; no column is an intercept. At all-zero responses, both families
have their finite unique maximum at zero coefficients. This deliberately checks
that zero outcomes alone do not diagnose a nonexistent estimator. In contrast,
the boundary fixture retains its intercept and has separated binary responses or
zero Poisson counts. The native exact-cone result, not the response label or
optimizer termination, distinguishes these geometries.

The reader checks the full planned cohort, normalized likelihood values,
independent analytic coefficients, offset orientation, typed responses and three
aligned new-data rows. Recession directions are exported as exact decimal integer
strings and independently replayed using rational arithmetic; they never round
through `f64`. Model-level density/central-difference score tests are separate from
the optimizer checks. Mutation tests reject fabricated uncertainty, promoted
failures, false witnesses, changed offsets/policies and missing/duplicate/partial
reports. Deterministic replay is checked within the same build, not promised
across versions or CPUs. Evaluation-error strings are caller evidence, not
portable typed-error serialization; export failures retain their original errors.

Forecast admission requires finite-attainment evidence, full numerical rank under
the declared `1e-12` cutoff, and native score termination under the `1e-6` policy.
This is an explicit application policy, not a generic interpretation of an
arbitrary `NamedFit`. Forecasts are **plug-in conditional means**, not posterior
draws or prediction intervals. All planned forecast rows remain present with null
means when admission fails. The original candidate, counters and termination are
never replaced by the analytic reference or a repaired model.

Uncertainty is explicitly `not_computed`: this workflow supplies no explicit
likelihood HVP to the checked observed-information API. It does not repurpose
L-BFGS history as covariance, silently finite-difference a Hessian, or advertise
Wald coverage. The native batch factors now provide source rules, but this
workflow still supplies no explicit derivative backend for that next uncertainty
step. These small point-fit fixtures do not complete general GLM application or
four-family qualification. No timing, ESS/second or Stan speed claim follows.

## Check estimator existence separately

### Typed observation maps and reusable data

For supported fixed-design models, describe the affine map, law and reduction
without choosing a CPU kernel:

```rust
use alea::{distributions::modeling::regression::RegressionData,
           model::assembly::ModelBuilder};
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut builder = ModelBuilder::new();
let beta = builder.real_vector("beta", &["intercept", "slope"])?;
let data = RegressionData::new(2, vec![1.0, -1.0, 1.0, 1.0], vec![false, true])?;
let map = data.clone().linear_map(beta.expressions())?.bernoulli_logit()?;
let likelihood = map.sum_log_prob(&mut builder)?;
builder.likelihood("responses", likelihood)?;
let model = builder.compile()?;
assert_eq!(model.expression_count(), 3); // Two coefficients and one native batch.
let replacement = data.with_observations(vec![true, false])?;
assert_eq!(replacement.design(), data.design());
# Ok(()) }
```

The typed map retains the existing native descriptor and selects its normal,
Bernoulli-logit or Poisson-log kernel on reduction. It is not an optimizer that
recognizes arbitrary scalar/tensor graphs, nor a transformation of Rust closures.
`plate` still expands scalar factors. No independence or prior is inferred.

Regression clones and `with_observations` share immutable predictor packs once
prepared. Offsets and dispatch remain per binding; standardized designs get a
fresh pack. Mutable score storage remains execution-local. Native row rules use
specialized empty derivative scratch rather than a second coefficient buffer.
These are ownership/allocation changes, not a measured wall-time speed claim.

### Native GLM curvature

`GlmLikelihood::bernoulli_logit(data)` and `GlmLikelihood::poisson_log(data)`
provide explicit likelihood targets on real coefficients in design-column order.
Their value/score calls delegate the existing native batch kernels. Potential
HVPs compute `Xᵀ W X v`, reusing prepared predictors, compensated scores and CPU
dispatch. They admit the native value/score first, even for a zero direction;
overflow/errors never become a plausible zero HVP. Successful retained calls
reuse scratch; outputs publish only after complete admission.

Fit and call `optimize::information::observed_information` with the **same**
target. Its exact replay check remains unchanged. The facade-only
`tools/downstream-conformance/tests/native_verticals.rs` exercises native fitting,
analytic reference covariance and explicit fit export for both families.
This standalone capability does not add HVPs to arbitrary composed graphs,
transformed coefficients, priors or nonlinear predictors. Finite attained MLE,
rank, regularity and interval coverage require separate qualification; the
older sixteen-case GLM workflow deliberately retains its uncomputed-uncertainty
status rather than being relabeled by the new bounded consumer.

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

### Errors and cancellation in a fit batch

Set the worker count to the smaller of the requested parallelism and the number
of model workspaces you can keep live. Build each model inside its task and keep
its input ID with the complete optimizer outcome. Collect individual results
rather than short-circuiting the batch on the first returned error.

Cancellation is cooperative. `maximize_with_cancel` checks before its initial
evaluation and between trials; it does not interrupt model construction or a
backend call already in progress. A cancelled fit may therefore have no candidate,
or may retain a complete accepted candidate. Neither cancellation nor a returned
backend error should be reported as convergence. Panics propagate separately.

The downstream `fit_batches` checks exercise this composition with different
Gaussian regression dimensions and observation counts. The design consists of
orthogonal group indicators, with equally many observations at each group mean
minus one and plus one. For `m` pairs per group, the residual sum of squares is

\[
\operatorname{RSS}(\beta)=2m\sum_j\bigl[(\beta_j-\mu_j)^2+1\bigr].
\]

Thus the independent MLE is `beta[j] = mu[j]` and `sigma = 1`. This also supplies
closed-form values and scores for checking every retained candidate, including
those left by errors or cancellation. The checks cover first-call and later-call
backend failures, dirty trial gradients, cancellation before evaluation and during
an in-flight call, stable input IDs, and workspace release during unwinding.
A four-worker request capped at two workspaces reaches two overlapping reservations
without exceeding the cap. Sequential and parallel searches retain identical
numerical outcomes and work counts for deterministic per-fit policies.

One larger ordinary fit stops at `LineSearchFailed` on the tested host despite
estimates within about `2e-9` of its analytic MLE. Its score remains above the
requested `2e-7` tolerance. That failure is retained in both execution policies;
accurate estimates do not turn an unsuccessful stopping condition into
`ScoreTolerance`. These are execution and numerical conformance checks, not a
claim of statistical convergence or a general-purpose batch scheduler.

### Reusing a compiled model for repeated fits

When the data and likelihood stay fixed, the same compiled objective can be
borrowed for several calls to `maximize`. Its evaluation scratch is reusable;
each optimization still starts from its supplied initial parameters with fresh
L-BFGS history. Concurrent fits need separate objectives. A retained objective
is not an interface for replacing observations or changing model parameters.

The throughput runner can compare cold construction with this reuse pattern:

```sh
python tools/downstream-conformance/python/benchmark_regression_throughput.py \
  tools/downstream-conformance/target/release/examples/regression_benchmark \
  regression-reuse.json --columns 32 --rows 1000 10000 \
  --fits 12 --threads 1 4 --construction cold reuse --runs 3
```

Both modes use identical fixed partitions of fit IDs. Cold mode constructs a
model for every fit; reuse mode constructs one inside each partition and evaluates
its fits sequentially. No objective crosses worker boundaries, and no locks or
thread-local model cache are introduced. For 12 fits, construction attempts fall
from 12 to one in direct execution, or to four with four partitions. All model
construction and pool startup remain in the process-throughput denominator.
Per-fit construction time is charged to the first fit using each model, then
reported as zero for its later uses; aggregate timings retain the complete cost.

A fresh cold/reuse comparison at 8 and 32 coefficients, 1,000 and 10,000
observations, all three families, one and four workers, and three repetitions
completed all 144 processes and 1,728 fits. Every fit passed independent SciPy
checks. Initial and final values, gradients, estimates, termination and optimizer
work counts were identical across construction policies and worker counts.
At 10,000 observations, process-throughput ratios (cold divided by reuse) were:

| Coefficients | Family | Direct sequential | Four workers |
| --- | --- | ---: | ---: |
| 8 | Gaussian | 1.02 | 1.07 |
| 8 | Bernoulli | 1.11 | 0.93 |
| 8 | Poisson | 1.14 | 1.18 |
| 32 | Gaussian | 1.09 | 1.10 |
| 32 | Bernoulli | 1.13 | 1.06 |
| 32 | Poisson | 1.15 | 1.16 |

Across both observation counts, ratios ranged from 0.92 to 1.21. The slower
Bernoulli result and the 1,000-observation Gaussian four-worker result remain
recorded. Avoiding construction is useful,
but these three-repetition, one-host timings do not establish a universal speedup.
The comparison concerns repeated optimization of the same likelihood, not
bootstrap samples, changing datasets, reuse of optimizer history, or faster
individual density evaluations. Earlier cold-construction cohorts remain separate.
