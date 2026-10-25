# Native distributions and model assembly

M14 adds an opt-in validated modeling path over the existing fused target and
reference covariance/NUTS components. No sampler default changes. The normalized
`Normal` law is distinct from the existing unnormalized dimension-bound `Gaussian`
target. Existing Gaussian, Wiener4/5/7, opaque primitives and analytic/Enzyme
adapters retain their implementations.

## Probability contracts and measures

`alea_core::distribution::Distribution` separates normalized density/PMF from
`ConstrainedModel` factors. Its event type and `EventShape` describe one event;
observation batches are separate slices. Discrete events use boolean or integer
types and cannot acquire continuous event derivatives. `ParameterScore`,
`ContinuousDistribution` and `Sample` are optional capabilities. A valid impossible
event returns negative infinity. NaNs, invalid parameters, shape mismatches and
numerical/backend failures remain errors. Continuous support describes its
interior; documented boundary density limits can be zero, finite or infinite.
Derivatives at singular boundaries are errors, even when the density is defined.

Tier 1 contains Normal (location/scale), Student-t (df/location/scale), LogNormal
(log location/scale), Exponential (rate), Gamma (shape/rate), Beta, Bernoulli,
Binomial and Poisson. Constructors check parameter domains; each implementation
documents score order. Sampling uses caller-owned RNGs. Gamma uses bounded
Marsaglia–Tsang rejection, Normal uses Box–Muller, Poisson sums independent
rate-at-most-30 inversions and Binomial uses Bernoulli trials. Rejection budgets
fail explicitly. Poisson sampling supports rates through 30,000; Binomial and
Multinomial sampling support at most one million trials. These sampling budgets
do not limit density evaluation. Finite arithmetic has a declared numerical
error path; extreme parameters do not have a universal accuracy guarantee.

Tier 2 supplies covariance-Cholesky MultivariateNormal, Dirichlet, Categorical,
Multinomial and LKJ-Cholesky. Vector/matrix events have explicit dimensions and
full row-major matrix storage. MultivariateNormal parameter scores use mean then
packed lower Cholesky order. Simplex probability scores are ambient derivatives;
the simplex transform supplies their tangent pullback. The LKJ density is relative
to free strict-lower Cholesky coordinates, with positive diagonal derived from row
norms. It includes its normalizer and the Cholesky measure, even at eta=1.
The derivation and exponent agree with the
[Stan Cholesky LKJ reference](https://mc-stan.org/docs/functions-reference/correlation_matrix_distributions.html).
Tests independently integrate its two-dimensional law and check correlation
second moments. These are small-dimensional regression gates, not arbitrary-size
or arbitrary-conditioning certificates.

`FiniteMixture` marginalizes a homogeneous finite component collection with
max-shifted log-sum-exp. Zero weights skip the component, all-zero support remains
negative infinity, and component failures propagate. Event scores and parameter
scores use responsibilities; parameter scores require interior mixture weights.
Sampling selects a component and then uses its generative capability. Discrete
latent categories are marginalized rather than differentiated by HMC.

## Typed assembly and one Jacobian

`ModelBuilder` declares `Parameter<Real>`, `Parameter<Positive>` and
`Parameter<UnitInterval>` handles. Each expression belongs to exactly one builder;
foreign handles are rejected. `vector` adds a single existing transform for a
simplex, vector or full matrix and names its constrained coordinates. Construction
expands independent plates and deterministic expression/reduction nodes once.
Repeated parameter references accumulate reverse derivatives into the same slot.
`normal`, `normal_observation`, `bernoulli_logit` and `wiener4` provide checked
normalized local factors; `factor` also accepts an explicit unnormalized potential.
`SumModel` remains the statically dispatched product-composition boundary.

`custom` permits an opaque deterministic/factor VJP. It must overwrite every local
derivative; missing/non-finite writes are rejected. Compiled plans use an immutable
operation tape, chain-owned reusable values/adjoints and local derivative storage.
Built-in operations use enum dispatch; custom factors use one dynamic callback per
node. Construction and outputs allocate, successful evaluations do not unless a
user callback allocates. Errors and panics never publish a partial caller gradient.
The owning `CompiledModel::into_target` applies one `ParameterLayout`, so each
transform contributes exactly one Jacobian independently of factor/plate use count.
This is an analytic reverse plan; Enzyme is optional, not an interpreter dependency.

The same compiled plan can now be exported ahead of time with
`alea_autodiff::codegen::enzyme_source`. Build scripts emit a fused Rust primal
and generated first-, second-, and third-order callbacks; the retained path has
no operation tape or dynamic factor dispatch. Normal and Bernoulli-logit helpers
supply Rust primals automatically. Plain opaque `custom`/Wiener nodes reject
export unless given a differentiable Rust primal with `custom_with_primal`.
See [native Enzyme models](enzyme-native-models.md) for the build workflow,
transform contracts, executed conformance scope and limitations.

Coordinate labels and optional block groups are cold metadata. `NamedOutput`
retains both constrained names and their layout, rejects a split transform group,
and projects geometry groups to **unconstrained** ranges. A three-element simplex
thus selects two geometry coordinates. Diagnostics of raw sampler positions must
be labeled as unconstrained; the log coordinate of `tau` is not a draw of `tau`.

## Predictions and examples

`prior_predict` invokes the caller's generative prior specification.
`posterior_predict` transforms completed posterior positions and invokes an
observation generator or deterministic generated quantity, with no density or
Jacobian weighting. Both return explicit prior/posterior provenance. They are
separate from transition evaluation and cannot mutate adaptation.

The separately locked external package builds normal regression, noncentered
hierarchical logistic and hierarchical Wiener4-drift models entirely through
public APIs. Each runs four-chain finite covariance warmup, frozen NUTS,
rank/folded R-hat and ESS, named blocks and posterior prediction:

```console
nix develop .#stable --command cargo run --manifest-path tools/downstream-conformance/Cargo.toml --locked --example modeling
nix develop .#stable --command cargo test --manifest-path tools/downstream-conformance/Cargo.toml --locked --test modeling -- --nocapture
```

DDM prediction generates the upper/lower boundary from its analytic first-passage
probability. RT simulation is not exposed as a production capability; no finite-step
Brownian approximation is called an exact RT RNG. The calibration suite's separate
spectral-tail inversion generates independent reference RTs with declared budgets.

## Executed evidence and limits

The [predeclared protocol](m14-validation-protocol.md) and
[results](benchmarks/m14-validation.md) bind the scope. Nine scalar-law gates cover
values, numerical normalization, event/parameter derivatives, RNG means/variances
and zero allocations. Vector/categorical/LKJ/mixture tests add their own measure,
normalization, score and sampling checks. Model tests cover foreign handles,
incomplete derivatives, shared parameters, simplex/correlation pullbacks, one
Jacobian, metadata projection and zero evaluation allocations. A compiler-negative
external case rejects a real-valued scale handle where a positive handle is required.

All three models agree with independently written formulas. Normal and logistic
lowerings also agree with executed pinned Enzyme gradients; Wiener remains an
opaque analytic VJP and is checked against an independent spectral density and
finite differences. The hierarchical posterior SBC suite uses 32 simulated
datasets per model and four coordinates, without seed selection or failed-dataset
removal. The normal conjugate check uses 64 datasets. These bounded suites do not
establish universal calibration, mixing or workflow robustness. Five-step warmup
failed explicitly on transform overflow; the completed protocol uses single-step
warmup and preserves that limitation. No S3 stability or research-sampler promotion
follows from completion of M14.
