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
| `analysis` | Stable Rust | Named CSV/JSON sampling artifacts, typed data arrays and predictive bindings |
| `faer` | Stable Rust | Optional matrix operations, geometry fitting and Gaussian rank/residual assessment |
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

Verify a source archive's checksum and detached SSH signature before use. The
allowed-signers file must come from a source you already trust, independently of
the archive. The checksum detects changed bytes; the signature identifies the
signer. Neither replaces checking that the model and inference settings suit your
analysis.
