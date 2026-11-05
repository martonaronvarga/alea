# Compiling derivatives with Enzyme

Enzyme can generate derivatives for a model compiled from Alea's model builder.
The source exporter writes a Rust module during your application's build. That
module provides a target with gradients, Hessian-vector products, a Hessian, and
third derivatives.

Use this path when you need compiled automatic derivatives and can use the pinned
nightly toolchain. Ordinary analytic models do not need Enzyme. The
[modeling guide](modeling.md) is a simpler place to start.

Built-in scalar/vector Normal and native Gaussian/logit/Poisson regression primitives provide
trusted first-order score rules. Source export composes those scores with the
graph and chart pullbacks instead of asking Enzyme to differentiate their scalar
reference loops. Regression rules call the existing native kernels, including
their execution policy, sufficient statistics and numerical fallbacks; they do
not contain a second probability implementation. Every local score is checked
before reverse propagation, even for a constant argument.

Generated `SOURCE_EXPLICIT_FIRST_ORDER` reports this composition, while
`SOURCE_CHECKPOINTED` specifically reports segmented tensor replay. Neither
qualifies higher-order differentiation of a native score rule: HVPs and third
derivatives still differentiate the original scalar primal with Enzyme.
Frozen native batch descriptors and mutable workspaces are initialized once per
calling thread and retained in function-local thread storage. First-use heap
allocation and that cache's memory are **not** included in the fixed source
float-arena count. Steady-state evaluation reuses the cache. Changing data
requires regenerating the module, as with other source-specialized constants.
No measured speedup follows merely from selecting this lowering.

## A complete small project

Create these three files in an application directory. The model is a
one-dimensional standard normal, with a constant normalizer omitted.

`Cargo.toml`:

```toml
[package]
name = "compiled-fit"
version = "0.1.0"
edition = "2024"

[dependencies]
alea = { git = "https://github.com/martonaronvarga/alea", tag = "v0.1.1", features = ["std-autodiff"] }

[build-dependencies]
alea = { git = "https://github.com/martonaronvarga/alea", tag = "v0.1.1", features = ["autodiff"] }

[profile.release]
lto = "fat"
```

`build.rs`:

```rust,ignore
use alea::model::assembly::ModelBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = ModelBuilder::new();
    let x = builder.real("x")?;
    let squared = builder.multiply(x.expr(), x.expr())?;
    let coefficient = builder.constant(-0.5)?;
    let log_density = builder.multiply(coefficient, squared)?;
    builder.factor(log_density)?;
    let model = builder.compile()?;

    let source = alea::autodiff::codegen::enzyme_source(&model)?;
    let output = std::env::var_os("OUT_DIR").ok_or("Cargo did not provide OUT_DIR")?;
    std::fs::write(
        std::path::PathBuf::from(output).join("posterior.rs"),
        source,
    )?;
    println!("cargo:rerun-if-changed=build.rs");
    Ok(())
}
```

`src/main.rs`:

```rust,ignore
#![feature(autodiff)]

mod posterior {
    include!(concat!(env!("OUT_DIR"), "/posterior.rs"));
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use alea::target::LogDensityGradient;
    let target = posterior::target();
    let mut gradient = [0.0; posterior::DIMENSION];
    let value = target.logp_grad(&[0.5], &mut gradient)?;
    println!("log density = {value}; derivative = {}", gradient[0]);
    Ok(())
}
```

From an Alea checkout, run the application with the compiler and Enzyme component
provided by its Nix environment:

```sh
nix develop .#autodiff --command env CARGO_TARGET_DIR=target/compiled-fit \
  cargo run --manifest-path /path/to/compiled-fit/Cargo.toml --release
```

The environment supplies the matching compiler flags. Fat LTO is required. Keep
this target directory separate from stable builds. The output for the example
is log density `-0.125` and derivative `-0.5`, up to floating-point formatting.

## Parameter transforms

Generated targets already operate in unconstrained sampling coordinates and
include each transform's log Jacobian once. Pass the generated target directly
to a sampler. Wrapping it in another parameter transform would apply the
Jacobian twice.

Simplex, unit-vector, and matrix charts can change dimension. The source exporter
retains their layout. Matrix admission checks remain outside automatic
differentiation; Enzyme differentiates the numerical transform itself.

## Supported model operations

The normal and Bernoulli-logit factor helpers provide their scalar Rust primals.
Native Gaussian, Bernoulli-logit and Poisson-log regression descriptors also emit
lazy, counted-loop primals; actual third-order conformance checks cover these
batch paths. `source_capabilities()` reports source availability, not universal
compiler support. [Tensor/control-flow source](tensor-ir.md) has default executed
matrix/branch/scan third-order regressions, including the zero-direction check
that caught a backend defect after compilation was repaired. The exporter keeps
typed float scratch explicit and supplies an explicit zero tangent for the HVP
direction. Arbitrary shapes/programs and statistical applications still require
their own qualification on the matching pinned compiler.
Custom callbacks need `custom_with_primal` to supply a matching differentiable
Rust expression. An opaque callback without such a primal returns
`AssemblyError::MissingPrimal` during export. This includes the Wiener factor.

Data constants and repeated observations are embedded in the generated source.
Changing them requires rebuilding the application. A supplied custom primal is
ordinary trusted Rust code: it must agree with the callback's value and domain,
and its operations must be supported by the matching Enzyme compiler.

## Derivative conventions

Curvature requests must pass the same finite value-and-score admission as a
gradient request. Even a zero-direction product rejects a singular score with a
numerical error. This stronger admission uses the actual gradient entry point,
not AD side outputs, and can add reverse work compared with a primal-only check.

The target gradient differentiates the log density. The higher derivative
capabilities differentiate the potential `U = -logp`. Hessians use row-major
storage, and the block starting at `k * dimension * dimension` contains the
Hessian's derivative with respect to coordinate `k`.

Dense Hessians and third derivative tensors become expensive as dimension grows.
Use a Hessian-vector product when an algorithm needs only directional curvature.
Generated derivatives do not change a sampler's numerical assumptions or make an
implicit transition exact by themselves.

## Compiled scalar objectives

Use `alea::autodiff::codegen::enzyme_objective_source(&model)` to generate a
composed scalar objective rather than a transformed sampling density. The emitted
module exports `value(q)` and `objective()`. The latter supplies fused scores,
potential Hessian-vector products, Hessians and third derivatives using the same
calling conventions as generated targets.

The objective includes all model factors and parameter charts, with no
log-Jacobian in its value or derivatives. Parameter-domain admission remains
checked, including rounded positive definiteness for matrix charts. Higher
derivatives include the curvature of nonlinear charts; they are not obtained by
only multiplying the physical Hessian by two Jacobians. The potential convention
is `U = -objective_value` for these derivatives.
