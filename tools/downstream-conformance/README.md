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
