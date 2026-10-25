# Contributing to Alea

Bug reports, examples, documentation improvements, and numerical fixes are
welcome. For a substantial API or algorithm change, open an
[issue](https://github.com/martonaronvarga/alea/issues) first so we can discuss the
intended behavior and how to check it.

## Build and test

Install Rust 1.95 or later and a C compiler, then clone with submodules:

```sh
git clone --recurse-submodules https://github.com/martonaronvarga/alea.git
cd alea
cargo test --manifest-path crates/Cargo.toml --workspace --locked
```

If you use Nix, `nix develop .#stable` supplies the stable toolchain. The default
shell supplies the pinned nightly compiler, and `.#blas` and `.#autodiff` provide
the optional native environments.

Start with tests for the code you changed:

```sh
cargo test --manifest-path crates/Cargo.toml -p alea-mcmc --test nuts_contracts
cargo fmt --manifest-path crates/Cargo.toml --all -- --check
cargo clippy --manifest-path crates/Cargo.toml --workspace --all-targets --locked -- -D warnings
```

Before submitting, run the complete stable checks:

```sh
nix develop .#stable --command bash tools/ci/check.sh
```

Run optional backend checks when your change affects them. Keep nightly and
stable build outputs in separate target directories.

## Write useful examples

Explain the statistical model, units, parameter conventions, and what a result
means. Prefer a complete small program to a fragment that depends on repository
internals. Use `?` to preserve errors and show how to inspect diagnostic warnings.
The [user guide](docs/README.md) is organized around tasks rather than implementation
history.

The Rust examples in the guide and package READMEs are executable documentation:

```sh
cargo test --manifest-path tools/documentation/Cargo.toml --doc --locked
```

Keep examples deterministic where practical.

## Numerical and API changes

Include a reproducer for a bug and, where possible, an independent formula,
high-precision calculation, or established implementation for comparison.
Preserve checked construction, complete gradient writes, coherent state after
errors, and the boundary between warmup and retained sampling.

A change to a public API needs a reviewed API diff. A change to a saved format
needs a version decision and reader tests. Explain changes in numerical behavior,
including new limits or failure cases. Historical experiment records retain their
original inputs and thresholds; add new evidence rather than rewriting old results.

## Report problems

Include the Alea revision, Rust version, operating system, enabled features, seed,
and a small runnable example. Keep original errors and diagnostic warnings in the
report. Remove private data and credentials. Send security-sensitive reports to
[contact@martonaronvarga.dev](mailto:contact@martonaronvarga.dev).

## Licensing

Contributions follow the receiving package's license: MIT OR Apache-2.0 for the
Rust inference libraries, GPL-3.0-or-later for distributions, cubature, and CLI
packages. Preserve notices and identify the source and license of copied or adapted
material. See [LICENSING.md](LICENSING.md).
