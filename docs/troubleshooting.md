# Troubleshooting

Keep the model, data, initial values, seed, and compiler settings while you
investigate a problem. Change one thing at a time. A shorter reproducible fit is
usually more informative than a long run with several settings changed at once.

## The project does not build

Check `rustc --version`: Alea requires Rust 1.95 or later. Start with the default
features on stable Rust. `simd`, `branch-hints`, and Enzyme require their documented
nightly toolchains; enabling every feature on stable will fail.

If the failure mentions cubature or a C compiler, check whether you enabled
the `distributions` feature. Install a C toolchain and initialize submodules
when building from a Git checkout. The default build does not need that native dependency. See [Installation](installation.md).

## Initial evaluation fails

Check dimensions and the distinction between constrained parameters and sampling
coordinates. For a positive parameter, a sampling coordinate of zero means a
parameter value of one. A very large log-scale coordinate can overflow after
transformation.

Evaluate the model once at the proposed initial point before running warmup.
Inspect the original error. Do not replace an unexpected evaluation failure with
negative infinity merely to make the sampler continue.

## The posterior looks wrong

Check the log density against a hand calculation on a small dataset. Common
mistakes include an omitted prior, a missing factor, a rate used as a scale, a
repeated Jacobian, and a gradient with the wrong sign. Sampling uses derivatives
of the **log density**. A capability explicitly named for the potential uses the
opposite sign.

For a custom gradient, compare every coordinate against finite differences at
several points away from singular boundaries. A plausible trace does not validate
a derivative. Where possible, fit simulated data with known generating parameters
and compare a simple case with an analytic posterior.

## There are divergences

Look for strong correlations, weak identification, scales close to zero, and
posterior regions with rapidly changing curvature. Reparameterize a hierarchical
model when its centered form creates a narrow funnel. Check the gradient first,
then consider a higher warmup acceptance target or more warmup.

Do not discard divergent draws and report the remainder as though the fit were
unchanged. Treat the warning as information about the fit that needs explanation.

## NUTS keeps reaching its tree-depth limit

First check the model's scaling and parameterization. Raising the depth limit can
allow longer exploration, but the worst-case integration work grows exponentially
with depth. Increasing the limit alone does not guarantee useful sampling.

## R-hat or ESS is unavailable

Modern diagnostics require at least two equal-length chains and enough variation.
Short, constant, nonfinite, or ragged series may make a statistic undefined.
Alea reports an error or warning for these cases. After recovery, a session only
summarizes its new segment; load the saved prefix for a full-run diagnostic.

## A run used more memory than expected

Collected draws grow with chains × draws × dimension. Use `.collect(false)` with
a durable or application-owned sink when keeping all records in memory is not
necessary. Review the diagnostic window and resource limits as well. Dense metric
storage grows quadratically with dimension.

## Recovery refuses a checkpoint

Check the model/data identity, seed, dimensions, sampler settings, and build.
A checkpoint is not a portable interchange format between different numerical
implementations. A deliberate rejection of an incompatible checkpoint is safer
than silently reinterpreting its saved state. Keep the exact source, lock file,
and build settings with long-running analyses.

## Report a problem

Open an [issue](https://github.com/martonaronvarga/alea/issues) with a small runnable
example, Alea revision, Rust version, operating system, enabled features, input
shapes, and seed. Include the original error and any diagnostic warnings. Remove
private data and secrets. For a security-sensitive report, use
[contact@martonaronvarga.dev](mailto:contact@martonaronvarga.dev).
