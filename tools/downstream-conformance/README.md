# Alea downstream conformance

This standalone package compiles against Alea exactly as an external Rust package
does. It is intentionally not a member of the production workspace and cannot use
crate-private APIs or inherit workspace dependencies.

The reusable assertions exercise the documented laws for externally implementable
fused targets, constant Euclidean metrics, Markov chains, and diagnostic projections.
The integration test supplies independent implementations and composes a chain with
the runtime's higher-ranked borrowed draw sink. Shape failures also check the
documented no-mutation boundary.

Run the same gate used by CI:

```console
cargo test --manifest-path tools/downstream-conformance/Cargo.toml --locked --offline
cargo clippy --manifest-path tools/downstream-conformance/Cargo.toml --all-targets --locked --offline -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --manifest-path tools/downstream-conformance/Cargo.toml --locked --offline --no-deps
```

This package does not test sealed integrator internals or the design-only SMC traits.
Compiler-negative ownership and exact/approximate composition contracts live in
the separate [`tools/compile-fail`](../compile-fail) package.

The public modeling verticals are also built here, outside the production workspace.
`src/modeling.rs` owns the normal-regression and hierarchical logistic/Wiener4
constructions. `tests/modeling.rs` compares independent formulas, runs retained
inference/prediction, and executes the predeclared posterior calibration suites.
Run the three complete examples with:

```console
cargo run --manifest-path tools/downstream-conformance/Cargo.toml --locked --example modeling
```

See [the modeling guide](../../docs/m14-modeling.md) for support, measure,
Jacobian, metadata and prediction semantics and the bounded validation scope.

The normalized-distribution extension point has its own independent external
implementation and reusable value/measure fixture helper in `tests/distribution.rs`
and `assert_distribution_conformance`.

The M15 session example uses a chain-local native model through explicit covariance
warmup, cancelled NUTS, a create-new checkpoint and bounded durable resumed output:

```sh
cargo run --manifest-path tools/downstream-conformance/Cargo.toml --locked \
  --example session -- /tmp/alea-session-example-new
```

The path must not exist. The example asserts 96 new records and 128 total retained
draws; one chain deliberately returns a diagnostic warning.

The analysis round-trip test uses Python 3 from PATH (or `ALEA_PYTHON`) to read
artifacts with an independent standard-library implementation. Native Alea
applications do not need Python. Reader mutation tests run with:

```console
python3 -m unittest discover -s tools/downstream-conformance/python
```
