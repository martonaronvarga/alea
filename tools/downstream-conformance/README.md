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


The optional external analysis gate exercises actual Gaussian and logistic fits,
DataTree/NetCDF and R round trips, a conjugate-normal PSIS-LOO oracle, and retained
Pareto-k warnings. Run it with the locked analysis environment:

```sh
nix develop .#analysis --command bash tools/ci/analysis.sh
```

The Python reader needs only the standard library. DataTree conversion needs the
packages pinned in `python/requirements-analysis.txt`; the R reader needs Python
and jsonlite. Native applications need none of these. See the
[analysis guide](../../docs/sessions.md#export-to-python-and-netcdf) for commands,
array bindings and partial-chain behavior. The short fitting fixtures verify the
export workflow; they are not convergence or comparative performance evidence.


`tests/optimization.rs` consumes `alea::optimize` through the public facade.
It checks Gaussian regression MLE coefficients, objective value and unknown scale
against analytic results, the distinct transformed-density mode, and a conjugate
posterior mode. Optimizer fault/budget/cancellation and nonlinear tests live in
runtime `tests/optimize.rs`; the shared inverse-BFGS action has an independent
dense-update oracle in math `tests/quasi_newton.rs`.

The fixed-design batch factors are checked against scalar assembly and finite
differences in `crates/alea-distributions/tests/regression.rs`. Named fitted
Gaussian/Bernoulli/Poisson examples use analytic coefficient solutions, including
explicit Poisson log exposures. A release-build scalar/batch benchmark is available:

```sh
cargo build --manifest-path tools/downstream-conformance/Cargo.toml --release \
  --example regression_benchmark --locked
python3 tools/downstream-conformance/python/benchmark_regression.py \
  tools/downstream-conformance/target/release/examples/regression_benchmark results-new.json
```

Run measurements without concurrent build/test jobs. The default plan has three
fresh-process runs per family/mode at 10, 1,000 and 1,000,000 rows. It records graph
expressions, construction, fused evaluation, local fitting, work counters, final
scores, exact binary/source hashes and Linux peak process RSS. Synthetic data and
per-observation score tolerances are fixed in the source. Ratios require matching
likelihood/gradient/estimates and the same satisfied stopping criterion; failed
searches remain in the report without a favorable ratio. RSS includes the entire
benchmark process, not just the graph. These are CPU measurements of two explicit
predictors on one host, not comparisons with other inference libraries.


`tests/regression_assessment.rs` checks likelihood recession witnesses and shows
why a small rounded score does not establish a finite MLE. With this package's
`faer` feature it also exercises Gaussian rank/residual assessment through the
facade. The optional analysis gate compares tall/wide/rank-deficient least squares
against NumPy's independent LAPACK-backed implementation:

```sh
nix develop .#analysis --command bash tools/ci/analysis.sh
```

Native witness and Faer tests need neither Python nor NumPy. The exact dot-sign
predicate has an independent Python Fraction fixture in math tests; Gaussian
rank and vanishing-residual warnings retain their explicit numerical tolerances.
Coordinate-direction screening is intentionally incomplete and never reports a
failed search as proof of overlap.
