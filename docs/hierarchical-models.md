# Partial pooling with a hierarchical model

Suppose three groups each perform twelve trials. They observe three, seven,
and ten events. Estimating each probability separately ignores what the groups
have in common. A hierarchical model shares information while allowing their
probabilities to differ.

The complete [hierarchical logistic example](../crates/alea/examples/hierarchical_logistic.rs)
uses only the public `alea` API. It builds the model, runs four chains, checks
parameters and predicted probabilities, and generates replicated event counts.
From a checkout, run:

```sh
cargo run --manifest-path crates/Cargo.toml -p alea \
  --features distributions --example hierarchical_logistic
```

To use it in your own project, copy the example into `src/main.rs`. Enable
`distributions` on `alea` and add `rand = { version = "0.10", features = ["small_rng"] }`.
See [Installation](installation.md) for the complete dependency declaration.

## Write down the model

For group `g`, let `p[g]` be its event probability:

```text
mu       ~ Normal(0, 0.7)
log_tau  ~ Normal(-0.7, 0.3)
z[g]     ~ Normal(0, 1)
tau       = exp(log_tau)
p[g]      = logistic(mu + tau * z[g])
y[g, i]  ~ Bernoulli(p[g])
```

`mu` is the shared log-odds location. `tau` describes between-group variation on
the log-odds scale. The prior on `log_tau` is fairly informative for illustration;
three groups provide little information about population variation. Assess prior
sensitivity before using this model for a substantive analysis.

The model samples standard-normal group effects `z`, then constructs each group's
log odds from `mu + tau * z[g]`. This is a noncentered parameterization. It often
helps when group-level effects are weakly identified. A centered formulation can
work better with strongly informative group data; compare both for your problem.

Here `log_tau` is an unconstrained real parameter with a normal prior. `tau` is
a derived expression. Do not add a Jacobian for this expression. If you instead
declare a positive `tau` and assign a density to `tau`, its change of coordinates
must be handled by the parameter transform, as explained in [Modeling](modeling.md).

Every trial in a group shares the same logit. The program therefore evaluates
one Bernoulli success term and one failure term, weighted by their observed
counts. This preserves the likelihood of the original trial sequence while
avoiding a graph node for every observation. Zero counts are skipped. It adds no
binomial combinatorial constant; predictions below still concern event counts.
Only combine trials that share the same logit expression throughout the parameter
domain. Matching fitted probabilities alone is not sufficient.

## Keep names and groups attached

The builder declares `mu`, `log_tau`, and a named vector `z` with labels `a`, `b`,
and `c`. Coordinate metadata records population and group parameter blocks. The
example reads these names back from the compiled model when reporting diagnostics.
Group labels describe the model; they do not change its geometry automatically.

Each chain constructs its own compiled model and reusable differentiation scratch.
Ordinary Rust expressions passed to the builder specify the graph; Alea computes
its reverse derivatives. Enzyme is not needed for this example.

## Check the fit before predicting

The example uses 1,000 discarded warmup transitions and 2,000 retained draws per
chain, with diagonal covariance adaptation and a target acceptance of 0.9. These
are example settings, not a universal fitting policy.

Before generating predictions, the program checks:

- Every chain completed its retained draws without a target failure.
- There were no retained divergences, numerical rejections, or maximum-depth terminations.
- Each chain has E-BFMI of at least 0.3.
- R-hat is at most 1.01 and bulk and tail ESS are at least 400 for every parameter,
  the derived scale, and each group probability.

These checks identify reasons to investigate a fit; passing them does not prove
that the model is appropriate or that all posterior regions were explored. ESS
requirements should follow the precision needed for the scientific estimands.
Inspect the reported E-BFMI and other diagnostics as well for real applications.

Keep chain identity when calculating diagnostics. The example transforms every
chain's draws into `tau` and `p[g]` separately, then summarizes those quantities.
Diagnostics on an unconstrained parameter alone need not describe the precision
of a nonlinear prediction.

## Interpret the predictions

Independent deterministic integration gives approximate posterior means:

| Quantity | Posterior mean |
|---|---:|
| `mu` | 0.1660 |
| `tau` | 0.5686 |
| `p[a]` | 0.4068 |
| `p[b]` | 0.5600 |
| `p[c]` | 0.6717 |

Finite Monte Carlo estimates vary. Compare their differences with their reported
MCSE, rather than expecting identical decimal values. The reference values come
from Gaussian-prior quadrature with increasing orders; this small example does
not establish calibration for arbitrary hierarchical models.

The observed proportions are 0.25, approximately 0.58, and approximately 0.83.
Partial pooling moves the extreme groups toward the shared population. The amount
of movement depends on the priors and data; it is not a correction applied after
sampling.

For each posterior draw, the program generates twelve new Bernoulli outcomes
per group and reports their average count. It uses a prediction seed separate
from the inference seed. Compare the full replicated count distribution with the
observed data in a real posterior predictive check. A mean alone can conceal a
poor fit.

These predictions concern another twelve trials in an **existing group**. To
predict a new group, draw a new standard-normal group effect for each posterior
draw, then use that draw's `mu` and `tau`. Reusing a fitted group's effect answers
a different question.

## Save an analysis

Record the model and data identity, source revision, dependency lock, chain seeds,
warmup settings, and predictions alongside the draws. The example's fixed identity
includes its event and trial counts. Change it when changing the model or data.
For durable output and restarts, follow [Sessions and recovery](sessions.md).

### Recover the same hierarchical fit

A checkpoint contains the sampler state, not the model or its data. Rebuild the
same model in the target factory when restarting. For the example above, keep the
group order, event counts, priors, parameter order, source revision, dependency
lock, and numerical backend fixed. The sampled coordinates are `mu`, `log_tau`,
and the three `z` values. Store those coordinates in the draw file; derive `tau`
and group probabilities after reading it.

Use the cancellation and durable-storage pattern in [Sessions and recovery](sessions.md)
with this model factory. Cancellation must happen after warmup to produce a frozen
checkpoint. Save the checkpoint for every chain you intend to resume, preserve
its logical chain ID, and restart with no warmup. Reusing a checkpoint with changed
data is a new analysis, not a continuation.

The requested draw count is the absolute end index. If a chain stops after 32
retained draws and the requested total is 96, recovery produces indices 32 through
95. Keep the first 32 records and write the remaining 64 to a new segment. Combine
segments by chain and absolute index, retaining rejected proposals as repeated
positions. Recalculate diagnostics on the combined chains before using their
predictions; a continuation's diagnostic summary covers only its new segment.

A downstream regression exercises this process with a two-group noncentered
logistic model, for both diagonal and dense covariance warmup. It rebuilds the
model, reads the checkpoint from disk, writes and reads a durable continuation,
and compares every draw and final checkpoint with an uninterrupted run in the
same build. This checks recovery mechanics, not posterior convergence or replay
across different builds and CPUs.

Structured parameter handles retain a domain marker, such as
`VectorParameter<Real>` or `VectorParameter<Simplex>`. Select that marker with
`builder.vector::<Real>(...)` or `builder.vector::<Simplex>(...)`; construction
checks that the supplied transform reaches the requested domain before changing
the builder. The handle's `transform()` retains ambient and free dimensions.
For a chart selected at runtime, use the explicit `DynamicDomain` marker and
inspect that validated transform rather than assuming a static domain.

## Repeated trials with identical logits

If a group has `k` observed events among `n` trials, its contribution to the
original trial-sequence log likelihood is

\[
k\log\operatorname{logistic}(\eta)
 +(n-k)\log\operatorname{logistic}(-\eta).
\]

Build the two Bernoulli terms with `bernoulli_logit`, multiply by their nonzero
counts, and add them as factors. This uses the existing stable tail formulas and
reverse derivatives. It also keeps the original likelihood constant; a normalized
binomial count likelihood instead includes `log choose(n, k)`. Choose the likelihood
that matches the recorded observation. If individual observations are predictive
units for model comparison, retain their individual reporting information even
when their total fitting likelihood is summarized.

A two-group benchmark compared expanded trials with this representation at 12,
1,000 and 10,000 trials per group. Two separately retained cohorts of 18 processes
each passed independent 70-digit checks of initial and final densities and scores.
On the tested host, target evaluation improved by about 2 times, 100 times and
over 1,000 times respectively. The count representation's graph size stays fixed
as the number of trials grows. These gains come from sufficient-count reduction;
they do not apply when logits differ between observations.

The attributed cohort met its declared score tolerance in both representations at
12 and 1,000 trials, with median mode-fitting ratios of about 1.3 and 42. At 10,000
trials neither representation met that tolerance; those six failed searches remain
recorded and have no qualified fitting-speed ratio. This is posterior-mode timing
in explicitly declared sampled coordinates, not a NUTS efficiency or cross-engine
performance comparison. Three repetitions on one host are not a speed guarantee.

## Qualify an unknown-scale Gaussian hierarchy

The checkout example
[`gaussian_hierarchy`](../tools/downstream-conformance/examples/gaussian_hierarchy.rs)
connects model assembly, session NUTS, four-chain diagnostics, existing/new-group
simulation and sampling-artifact export. Its
[shared consumer module](../tools/downstream-conformance/src/gaussian_hierarchy.rs)
uses existing public Alea APIs, not handwritten distributions, gradients or NUTS.
Unlike the [fixed-effect ML workflow](modeling.md#run-a-fixed-effect-gaussian-workflow),
this is Bayesian inference with an estimated between-group scale:

```text
mu       ~ Normal(0, 1)
tau      ~ HalfNormal(0, 1)
z[g]     ~ Normal(0, 1)
theta[g]  = mu + tau*z[g]
y[g, i]  ~ Normal(theta[g], 1)
```

The two observed groups are `[-1,-0.5,0]` and `[0.5,1,1.5]`. The residual
standard deviation is fixed at one; `tau` is not fixed. A positive parameter
transform supplies `tau=exp(q_tau)` and exactly one log-Jacobian term `q_tau`
in the sampling target. The native Normal prior plus `log(2)` supplies its
normalized half-normal prior. Model-level tests check the normalized density and
scores against independent formulas and central differences, separately from
sampling. This is not joint-random-effects MLE, marginal mixed-model ML or REML.

Reproduce the complete application check with:

```sh
nix develop .#stable --command bash -euo pipefail -c '
  cargo run --quiet --manifest-path tools/downstream-conformance/Cargo.toml \
    --example gaussian_hierarchy --release --locked -- complete |
    python3 tools/downstream-conformance/python/gaussian_hierarchy.py
'
```

Replace `complete` with `cancelled` or `initial_transform` for the empty-chain
controls. Each retains all four planned chains and their actual status/error
chains without fabricating posterior summaries. The initial-failure control uses
`q_tau=-1000`, preserving the original transform-boundary error rather than
repairing the initialization. Completed valid draws survive any subsequent native
failure in the producer's artifact; the qualification reader rejects incomplete
successful-case cohorts instead of selecting only successful chains.

The bundle embeds the existing version-2 sampling manifest and exact UTF-8 CSV
bytes, with their SHA-256 digest, alongside the session plan, native chain outcomes,
adapted step sizes and full-retained-window diagnostics. The shared Python reader
checks the same schema/digest/records from a seekable stream or files. The outer
`gaussian-hierarchy-workflow` version-1 wrapper is a checkout qualification format,
not a new production inference/restart format. Provenance labels identify the
fixture/version; they are not verified source hashes. Preserve the original
producer output if a later qualification step fails. Partial output is not valid.

### Existing groups are not new groups

An existing-group replication uses its fitted `theta[g]`. A new group requires a
fresh standard-normal effect at each posterior draw. Two observations from the
**same** new group share that effect but have independent observation noise; a
different new group receives another independent effect. This follows the
distinction between conditional and mixed replication in the
[Stan guide](https://mc-stan.org/docs/stan-users-guide/posterior-predictive-checks.html#mixed-predictive-replication-for-hierarchical-models).
These generated quantities are not plug-in ML forecasts. The two exported
pointwise likelihood units condition on existing fitted effects; the exporter
does not reinterpret them as integrated new-group predictive densities.

The reader checks this joint behavior, not only predictive means. Conditional on
one posterior draw, squared differences between same-new-group observations
have expectation `2`; differences between different-new-group observations have
expectation `2*(1+tau²)`. Reusing an existing fitted effect, resampling a separate
effect for every same-group observation, or reusing observation noise changes
these reference moments. Simulation uses Alea's native Normal sampler and a
prediction RNG stream separate from inference.

### Independent reference and fixed gates

At a fixed physical `tau`, let `V=tau²+1/3`. After integrating `mu` and both group
effects, the two group means have covariance `V*I + 11'`. The unnormalized marginal
posterior weight is

```text
exp(-0.5 * (tau² + 1.25/V - 0.25/(V*(V+2)))) / sqrt(V*(V+2)), tau >= 0.
```

Conditional Gaussian algebra gives moments of `mu` and `theta`; the independent
standard-library Python oracle integrates only physical `tau`, without a log-scale
Jacobian in that integral. Composite Simpson integration doubles from 6,144 to
12,288 panels on `[0,12]` and extends to 20,480 panels on `[0,20]`. All 15 normalized
reference moments must agree within `1e-9`. This is a refinement/domain-extension
check, not a rigorous integration-error bound or an independent sampler.

The protocol fixes four dispersed initial states, session seed `[83;32]`, 1,000
warmup and 2,000 retained draws per chain, initial step `0.1`, depth limit `8`,
diagonal covariance adaptation, target acceptance `0.9`, and no step-size search.
Prediction seeds are `83001+chain`. Before the first run, the gates were fixed to
R-hat below `1.05`, bulk/tail/raw-mean ESS above `200` for all 15 observables,
reference-moment error below six estimated MCSEs, no retained numerical rejections,
and at most 1% retained divergences and depth saturation. The reader independently
replays means and MCSEs from every exported draw; native diagnostic implementations
have separate reference tests. Gates and seeds are not searched after a failure.
The native session status is reported separately; passing these application gates
does not overwrite a native diagnostic warning (which may use stricter transition
gates or other observables).

The checked run retains all 8,000 draws, with zero retained divergences/depth
saturation, maximum R-hat `1.00193`, minimum bulk/tail ESS about `2367`/`1770`, and
maximum reference discrepancy `1.62` estimated MCSEs. Debug and release replay are
checked separately. Finite-chain diagnostics and estimated-MCSE gates are scoped
evidence, not a coverage guarantee or proof that all posterior modes were found.
This small, two-group, noncentered, fixed-residual-scale example does not resolve
the centered-eight-schools stress case, unknown residual-scale hierarchies, general
group counts/designs or the whole four-family application cohort. It supplies no
Stan timing comparison, ESS/second claim or production method-maturity promotion.
