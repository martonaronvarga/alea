# Native model compilation with Enzyme

`alea_autodiff::codegen::enzyme_source(&model)` compiles the existing typed model
graph to Rust source ahead of time. Enzyme differentiates the emitted numerical
primal on Alea's `nightly-2026-06-23` compiler with its matching Enzyme component.
The resulting target supplies fused log density/gradient, potential HVP, dense
potential Hessian and coordinate-major third derivatives. Samplers consume the
usual `LogDensityGradient`, `HessianVector` and `PotentialHessianDerivative`
interfaces; no sampler numerical policy changes.

## Build-script workflow

Define your model construction in a shared source file using `ModelBuilder` and
the ordinary Normal/Bernoulli-logit helpers. Add `alea-core`, `alea-autodiff` and
any model-building distribution crate as build dependencies. Add `alea-core` and
`alea-autodiff` as runtime dependencies. The exporter itself works on stable Rust;
the crate that includes generated derivatives requires the Enzyme nightly.

In `build.rs`, after constructing the `CompiledModel`:

```rust,ignore
let source = alea_autodiff::codegen::enzyme_source(&model)?;
let out = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
std::fs::write(out.join("posterior.rs"), source)?;
println!("cargo:rerun-if-changed=models.rs");
```

In the application crate:

```rust,ignore
#![feature(autodiff)]

mod posterior {
    include!(concat!(env!("OUT_DIR"), "/posterior.rs"));
}

// Already unconstrained, with Jacobians included. Pass directly to NUTS,
// force-gradient HMC, curvature adaptation or a SoftAbs metric.
fn main() {
    use alea_core::target::LogDensityGradient;
    let target = posterior::target();
    let q = [0.0; posterior::DIMENSION];
    let mut gradient = [0.0; posterior::DIMENSION];
    println!("{:?}", target.logp_grad(&q, &mut gradient));
}
```

Compile in release mode with fat LTO:

```console
nix develop .#autodiff --command env CARGO_TARGET_DIR=target/my-enzyme-app \
  cargo test --manifest-path PATH_TO_APPLICATION/Cargo.toml --release --locked
```

The consuming application's release profile needs `lto = "fat"`; the shell sets
`-Zautodiff=Enable -Cembed-bitcode=yes`. Keep its Cargo target directory separate
from stable builds. [The downstream consumer](../tools/enzyme-conformance/) has
a complete manifest, shared model definitions, build script and executed tests.

## Numerical and modeling contracts

Every transform is lowered to scalar Rust expressions, including dimension-changing
simplex/unit-vector and matrix charts. The generated primal includes exactly one
log Jacobian per block. Do not wrap the generated target in another transform.
The rounded covariance/correlation positive-definiteness admission check runs
outside AD using the original Rust transform implementation. Enzyme differentiates
the numerical chart rather than the admission factorization or Result/enum machinery.

`normal`, `normal_observation` and `bernoulli_logit` automatically supply their
scalar primals while retaining the original checked analytic VJPs for stable
evaluation. `custom_with_primal` accepts an explicit Rust function body with an
`x: &[f64; N]` input. That body must match the callback's value/domain and use
explicit expression inputs for captured data. It is trusted Rust application
source. Plain opaque callbacks, including the existing Wiener helper, return
`AssemblyError::MissingPrimal` during export. There is no hidden analytic or
finite-difference derivative fallback.

Data constants and expanded plates are specialized into the source. Changing
them requires regeneration and compilation; this is not runtime JIT compilation
or an exporter from arbitrary Stan programs. The source itself is inspectable.
Custom functions must also be supported by the pinned Enzyme compiler.

Gradients use reverse AD, HVPs use forward-over-reverse, and third contractions
use forward-over-forward-over-reverse. Hessians are row-major; the block beginning
at `k*d*d` is `dH/dq_k`. All higher outputs differentiate potential `U = -logp`,
including the negative sign and every transform/Jacobian term. Outputs and seeds
are reset on each call. Higher-order calls explicitly check scalar primal admission
because a derivative can be finite on a NaN-return branch. They therefore include
one additional scalar log-density evaluation. Bad shapes/inputs fail before
callbacks; failed raw derivative outputs must be discarded.

Successful evaluations allocate no Rust heap memory. Generated callbacks use O(d)
fixed-size stack scratch. Dense Hessian/third assembly uses O(d²) directional
third evaluations and caller-owned O(d³) tensor storage. Large dimensions therefore
need stack/compile-size consideration; this is not a matrix-free SoftAbs algorithm.

## Executed validation

`tools/enzyme-conformance` builds models through the public API in a real build
script and executes their generated machine code. It checks:

- An independent coupled quartic oracle with nonzero mixed third derivatives,
  potential signs, tensor block order and HVP products.
- Normal regression with a positive scale, logistic regression and every existing
  transform kind against the original analytic model, finite differences of
  independent gradients and differentiated Hessian checks, including chart centers.
- 3×3 covariance/correlation charts and a 16-dimensional Gaussian tensor check.
- Repeated-call seed/output reset, invalid inputs, support collapse and zero Rust
  heap allocations.
- Native NUTS warmup and retained sampling, joint curvature adaptation,
  force-gradient HMC, SoftAbs metric derivative finite differences, and Newton/
  reverse-checked RMHMC with nonzero metric derivatives and no allocations.

The existing Enzyme first-order tests and stable analytic-model tests remain
separate. `tools/ci/autodiff.sh` now runs the generated downstream suite as well.
These are bounded compiler/model conformance tests, not evidence of a universal
Enzyme speedup or a new PosteriorDB comparison. RMHMC remains explicitly
`ExactnessPendingRmhmc`; generated third derivatives do not change its finite-map
volume/reverse-support obligations.

The bounded [gradient-cost experiment](benchmarks/enzyme-native-2026-10-22.md)
records gradient speedups of 2.58–5.73× over the analytic modeling tape on
three small specialized targets. It does not establish end-to-end library speed.
