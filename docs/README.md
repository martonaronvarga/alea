# Alea user guide

Alea lets you describe a probability model in Rust and draw samples from its
posterior distribution. You choose the model and the initial values. Alea
provides the sampling algorithms, warmup, diagnostics, and storage.

If this is your first visit, follow [Your first model](getting-started.md). It
fits a normal mean to four observations and explains each part of the program.

| I want to… | Read |
|---|---|
| Install Alea or select a numerical backend | [Installation](installation.md) |
| Run a complete example | [Your first model](getting-started.md) |
| Declare parameters, priors, and likelihoods | [Models and constraints](modeling.md) |
| Choose a sampler and configure warmup | [Sampling](sampling.md) |
| Interpret R-hat, ESS, and warnings | [Diagnostics](diagnostics.md) |
| Run chains in parallel or save a checkpoint | [Sessions and recovery](sessions.md) |
| Evaluate or sample from a distribution | [Distributions](distributions.md) |
| Work with reaction times and choices | [Wiener models](wiener.md) |
| Understand a failed or suspicious fit | [Troubleshooting](troubleshooting.md) |

## How the pieces fit together

A **model** describes the log density and its derivatives. A **sampler** uses that
information to propose parameter values. A **session** runs one or more chains,
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
