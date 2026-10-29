# Alea user guide

Alea lets you evaluate probability distributions, describe statistical models in
Rust, optimize likelihoods and posterior modes, and draw posterior samples.
You choose the model, estimation method, and initial values. Alea provides
parameter transformations, derivatives, optimization, sampling, and result storage.

If this is your first visit, follow [Your first model](getting-started.md). It
fits a normal mean to four observations and explains each part of the program.

| I want to… | Read |
|---|---|
| Install Alea or select a numerical backend | [Installation](installation.md) |
| Run a complete example | [Your first model](getting-started.md) |
| Declare parameters, priors, and likelihoods | [Models and constraints](modeling.md) |
| Optimize a likelihood or posterior mode | [Point estimation](modeling.md#maximum-likelihood-and-posterior-modes) |
| Evaluate many regression observations together | [Batch likelihoods](modeling.md#batch-regression-likelihoods) |
| Fit group effects with partial pooling | [Hierarchical models](hierarchical-models.md) |
| Choose a sampler and configure warmup | [Sampling](sampling.md) |
| Interpret R-hat, ESS, and warnings | [Diagnostics](diagnostics.md) |
| Run chains in parallel or save a checkpoint | [Sessions and recovery](sessions.md) |
| Export draws, predictions, and pointwise likelihoods | [Analysis artifacts and interoperability](sessions.md#export-a-named-sampling-table) |
| Evaluate or sample from a distribution | [Distributions](distributions.md) |
| Model axes and unit-vector observations | [Directional distributions](directional-distributions.md) |
| Work with reaction times and choices | [Wiener models](wiener.md) |
| Understand Wiener numerical accuracy | [Wiener numerical methods](wiener-numerics.md) |
| Understand a failed or suspicious fit | [Troubleshooting](troubleshooting.md) |

## How the pieces fit together

A **model** describes factors and their derivatives. An **objective** selects the
terms to optimize, such as observation likelihoods or likelihoods and priors.
An **optimizer** searches for a local maximum and records its stopping reason.
A **sampling target** includes the change-of-variable Jacobians needed for
sampling in unconstrained coordinates. A **sampler** uses that target to propose
parameter values. A **session** runs one or more chains,
discards warmup, and collects or streams the draws. **Diagnostics** help you judge
whether those draws are useful for the quantities you want to estimate.

For a constrained parameter, such as a positive scale, Alea samples an
unconstrained coordinate and transforms it back to the parameter's natural units.
This distinction matters when interpreting stored draws and diagnostic labels.
The [modeling guide](modeling.md) shows both representations.

## Where to look next

The `alea` crate documentation describes individual types and methods. To build it from
a checkout:

```sh
cargo doc --manifest-path crates/Cargo.toml -p alea --features distributions --open
```

[Compatibility and supported platforms](releases.md) explains versioning, compiler
requirements, features, and numerical reproducibility. [Metric conventions](metric-contract.md) explains the mass and inverse-mass
representations used by the samplers.
