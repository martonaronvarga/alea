# Compiling derivatives with Enzyme

Enzyme can generate derivatives for a model compiled from Alea's model builder.
The source exporter writes a Rust module during your application's build. That
module provides a target with gradients, Hessian-vector products, a Hessian, and
third derivatives.

Use this path when you need compiled automatic derivatives and can use the pinned
nightly toolchain. Ordinary analytic models do not need Enzyme. The
[modeling guide](modeling.md) is a simpler place to start.

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
Custom callbacks need `custom_with_primal` to supply a matching differentiable
Rust expression. An opaque callback without such a primal returns
`AssemblyError::MissingPrimal` during export. This includes the Wiener factor.

Data constants and repeated observations are embedded in the generated source.
Changing them requires rebuilding the application. A supplied custom primal is
ordinary trusted Rust code: it must agree with the callback's value and domain,
and its operations must be supported by the matching Enzyme compiler.

## Derivative conventions

The target gradient differentiates the log density. The higher derivative
capabilities differentiate the potential `U = -logp`. Hessians use row-major
storage, and the block starting at `k * dimension * dimension` contains the
Hessian's derivative with respect to coordinate `k`.

Dense Hessians and third derivative tensors become expensive as dimension grows.
Use a Hessian-vector product when an algorithm needs only directional curvature.
Generated derivatives do not change a sampler's numerical assumptions or make an
implicit transition exact by themselves.
