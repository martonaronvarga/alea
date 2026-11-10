# Compatibility and supported builds

Alea's public Rust interface is the `alea` crate. Use its modules or
`alea::prelude::*` in applications. The other workspace packages are implementation
components and are not the application-facing dependency interface.

## Versions and compatibility

Within the 0.1.x series, public Rust items, feature names, documented trait laws,
and the minimum supported Rust version remain compatible. A breaking API change
requires an incompatible release and migration instructions. After 1.0, breaking
API changes require a major version.

This promise includes APIs described as experimental. Their numerical assumptions
and limits still apply, but the label is not an exception to source compatibility.
A correction to numerical behavior can change calculated values without changing
an API. Such corrections are described in the release notes.

The current compatible series and its predecessor receive correctness and
security fixes for at least 12 months after the successor's publication.
Deprecations remain usable for at least two minor releases and 90 days; removal
requires an incompatible version. Urgent soundness or security repairs may need
a behavior change, which will be explained with the correction.

Send security-sensitive reports to [contact@martonaronvarga.dev](mailto:contact@martonaronvarga.dev).
For other bugs, use the [issue tracker](https://github.com/martonaronvarga/alea/issues).
Include the source version, platform, compiler, features, and a reproducer without
private data.

## Unreleased modeling ergonomics

Native regression descriptors now expose immutable CPU execution plans and an
explicit scalar diagnostic mode. Existing modeling helpers keep automatic Pulp
dispatch and the same fused compensated kernels. Data rebinding preserves the
requested policy and refreshes packing decisions. These are reviewed development
API additions, not a new precision policy, cost-based scheduler, observation
threading backend or speed claim. See [native CPU plans](modeling.md#inspect-or-diagnose-native-cpu-execution).

Automatic native regression dispatch additionally bypasses SIMD when neither an
observation tile nor a coefficient packet fits. The decision is inspectable and
leaves wider-than-tile normal fallbacks unchanged. It uses the same kernels and
preserves tested value/score bits and errors; it is not a learned timing cutoff.

Native normal and normal-regression factors now retain typed semantic descriptors
alongside the unchanged analytic execution tape. Graph inspection, explicit
source-capability reports, custom `Primitive` registration and scoped Rust model
composition are additive development interfaces. Scoped vector defaults qualify
their dimension/block labels; explicitly supplied annotations are unchanged.
Existing unscoped declarations, probability formulas, Jacobian boundaries and
opaque-callback source limitations are preserved. Descriptor registration adds cold storage
and construction work; it is not a speed claim or a general graph compiler.

The next development increment adds fixed-shape f64 tensor/control-flow regions,
immutable data bindings, consuming descriptor rebinding, and explicit conditional
row/source rules for native Normal, Gaussian, Bernoulli-logit and Poisson-log
regression. `AssemblyError::MissingPointwise` is a new enum variant; the added
trait methods/types and this matching impact require pre-release compatibility
review, not a waiver of the policy above. Topology and parameter charts cannot
change during rebinding. Workspace-explicit source export and a defaulted
`Primitive::source_workspace` hook additionally preserve typed mutable buffers
through compiler AD. These additions receive the same pre-release API review.
The original tensor compiler failure and a scan third-derivative defect now have
default executed regressions. Source emission is still not a promise that a
compiler AD backend supports every region or that tensor statistical workloads
are qualified. See [contracts and qualification](tensor-ir.md).

The development interface adds `alea::model!`, literal/handle scalar inputs, named
native distribution helpers and named vector conveniences. The explicit builder
uses one `Value` handle for scalar assembly and lexical tensor regions. Arithmetic
and Normal/Bernoulli-logit
constructor signatures are now generic over checked scalar inputs; code relying
on implicit `.into()` inference may need to pass the handle directly or supply a
type annotation. These signature changes require compatibility review before a
release; this is not a waiver of the compatible-series policy above. No version
or publication is implied by the development changes. See the
[modeling guide](modeling.md#concise-syntax-ordinary-rust) for examples and limits.

## Rust and platforms

The minimum supported Rust version is **1.95**. Patch releases preserve it.
Changes to this minimum require a minor release before 1.0; after 1.0 they require
notice for at least one minor release and 90 days.

The native test matrix covers Linux x86_64 and aarch64, macOS x86_64 and aarch64,
and Windows x86_64 MSVC, at the declared minimum and stable Rust. Other targets,
including wasm, musl, no_std, and GPU execution, have no support claim.

## Cargo features

The default build provides modeling interfaces, samplers, sessions, and diagnostics
on stable Rust. Select optional capabilities on the `alea` dependency:

| Feature | Requirements | Adds |
|---|---|---|
| `distributions` | A C compiler | Probability laws, model factors, and Wiener likelihoods |
| `autodiff` | Stable Rust | Derivative adapters and Enzyme source export |
| `analysis` | Stable Rust | Named sampling artifacts, point-estimation JSON, typed data arrays and predictive bindings |
| `tracing` | Stable Rust; application-owned subscriber | Opt-in structured execution events and spans; see [Tracing](tracing.md) |
| `faer` | Stable Rust | Optional matrix operations, geometry fitting, Gaussian rank/residual assessment, assumption-qualified Gaussian marginal intervals and explicit likelihood-information checks |
| `openblas` | LP64 OpenBLAS headers and libraries | BLAS-backed matrix operations |
| `simd` | Pinned nightly Rust | Portable SIMD kernels |
| `branch-hints` | Pinned nightly Rust | Additional Wiener probability hints; enables distributions |
| `std-autodiff` | Matching nightly Rust and Enzyme | Compiled automatic derivatives; enables autodiff |

Examples and toolchain commands are in [Installation](installation.md).

## Numerical reproducibility

Use the same model and data, source version, dependency lock, target semantics,
compiler options, and backend when reproducing an analysis. Logical chain identity
preserves seed assignment across sequential and parallel execution, but output
from different chains can be interleaved differently.

Checkpoint replay requires matching numerical identities and rechecks the saved
point. Bitwise agreement across arbitrary builds, CPUs, or backends is not
promised. See [saving and resuming a run](sessions.md).

Saved formats have explicit versions. Incompatible data is rejected rather than
reinterpreted. Checkpoints currently use `ALEACP02`; durable draw streams use
`ALEADW01`. A saved checkpoint is not a substitute for recording the model, data,
source, lock file, and run settings.

## Source archives

Source distributions contain the libraries, user documentation, build inputs,
licenses, and executable verification tools. Local datasets, experiment archives,
agent instructions, and private working notes are excluded.

Verification source includes the native Alea/Stan benchmark, Wiener coverage and
modeling tools, their Cargo locks, Stan model sources, and the two kernel case
studies. Generated executables, datasets and raw measurement capsules are not
bundled. Historical measurements describe their original sealed source, compiler
and machine; the current archive alone cannot replay that historical environment.
The included tools support new, separately identified verification runs. Source
creation and verification check first-party Markdown relative file targets against
the shipped manifest, including after extraction without Git metadata.

Enzyme adapters now classify their own non-finite evaluation errors as numerical
divergences during checked Hamiltonian integration, matching native models. Direct
evaluation, initialization and explicit curvature requests still return typed
errors. Dimension errors and unrelated backend failures remain fatal. This fixes
error-policy drift, not derivative arithmetic, and makes no new speed claim.

Verify a source archive's checksum and detached SSH signature before use. The
allowed-signers file must come from a source you already trust, independently of
the archive. The checksum detects changed bytes; the signature identifies the
signer. Neither replaces checking that the model and inference settings suit your
analysis.
