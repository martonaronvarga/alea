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

The program represents each trial as a Bernoulli observation. Event ordering does
not affect the posterior in this model. A binomial likelihood for the counts
would add a data-only combinatorial constant; predictions below concern counts.

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
