# Generated native Enzyme targets

This downstream package constructs models through public Alea APIs in `build.rs`,
emits Rust into `OUT_DIR`, and executes generated gradients, HVPs, Hessians and
third derivatives through existing sampler capabilities.

```console
nix develop .#autodiff --command env CARGO_TARGET_DIR=target/enzyme-integration \
  cargo test --manifest-path tools/enzyme-conformance/Cargo.toml --release --locked
```

Release/fat LTO and the matching Enzyme compiler are required. Keep this target
directory separate from stable builds. See [the native-model guide](../../docs/enzyme-native-models.md)
for integration. Historical measured profiles are maintainer evidence, not bundled
source-archive results. `examples/profile.rs` compares the analytic
modeling tape and generated gradients with alternating order and preserved rounds.

Native Gaussian, Bernoulli-logit and Poisson-log regression batches now export
counted-loop primals and are included in actual third-order checks. Tensor source
is separately compiled/executed in `tests/tensor_source.rs`. Default
`tests/tensor_matrix.rs` and `tests/tensor_control.rs` execute nested AD with
independent derivatives, both branches, zero-trip scans, empty-result domain
errors and recovery. A raw zero-direction/linearity regression catches the
spurious third-derivative term exposed after the original compiler failure was
fixed. The diagnostic-only `tensor-ad` feature has been removed; these checks
are no longer opt-in. Arbitrary programs and statistical applications
still require their own qualification. See
[tensor contracts and limitations](../../docs/tensor-ir.md).

Explicit checkpoint fixtures also exercise first-order segmented source VJPs,
257-trip replay, mixed native priors/positive charts, matrix scores, empty selected
errors and first-versus-steady derivative activity. `SOURCE_CHECKPOINTED` reports
this first-order selection. HVPs/third derivatives retain the original compiler
reverse: forward AD of the rematerialized reverse failed pinned-compiler tests.
Do not infer higher-order tape bounds or a peak-memory win from source presence
or a zero-allocation test; the retained small compiler fixture also allocates zero.

First-order source adjoint storage excludes inactive/immutable ranges and shares
reshape aliases. Additional executed fixtures check immutable `sqrt(0)` values,
parameter-independent checkpoint outputs, alias finite differences and overflow
recovery. This is not liveness compaction of active intermediates or higher-order
checkpointing; declared scratch reductions are distinct from measured peak RSS.

Segment replay now retains validated body primals for its prepared reverse,
avoiding a second body evaluation per replayed trip. Byte-weighted segment sizing
accounts for that retained storage. A nonlinear vector-carry fixture checks
partial segments analytically and by finite differences. The remaining model/VJP
admission and boundary passes, and higher-order compiler limits, still apply.

The fixed-work control executable accepts a case and evaluation count:

```console
nix develop .#autodiff --command env CARGO_TARGET_DIR=target/enzyme-integration \
  cargo run --release --locked --manifest-path tools/enzyme-conformance/Cargo.toml \
  --example source_checkpoint -- long 20000
```

Cases are `checkpoint`, `long`, `retained`, `mixed`, `matrix`, and `routes`.
Output reports internal evaluation seconds and density/all-coordinate score
checksums. Preserve saved before/after binaries and alternate repeated runs;
compilation, startup and model construction are not part of this timer. These
controls are neither NUTS benchmarks nor ESS/sec measurements.

The higher-order follow-up replaces a differentiated chained finite-check iterator
with equivalent separate checks, but keeps HVP/third on the original compiler
reverse. Experimental forward AD of the explicit VJP both miscomputed negative-
branch higher derivatives in the conformance binary and failed mixed-factor
preprocessing, despite some standalone probes succeeding. Expanded native-score
difference/objective/singular-activity tests and a generator routing guard cover
the shipped fallback. See the [case study](../../docs/tensor-ir.md) for exact
counterexamples; successful compilation is not higher-order qualification.

Curvature admission also calls the actual value/gradient entry point, not just
the primal: a zero-direction HVP must not admit a singular score. The regression
checks typed numerical failures for zero/nonzero directions and valid recovery.
This replaces primal-only admission and can cost additional reverse work; it
does not change the sampling gradient path.

The existing `tensor_matrix` target also contains bounded direct-provider probes.
They call the checked `alea::math::gemm::try_gemm` adapter. Faer primal and
explicit-first source conformance run with the normal feature:

```console
nix develop .#autodiff --command cargo test --manifest-path tools/enzyme-conformance/Cargo.toml --release --locked --features faer --test tensor_matrix
```

Deliberate compiler/link-failure probes are opt-in cfgs on this test target only:

```console
nix develop .#autodiff --command cargo rustc --manifest-path tools/enzyme-conformance/Cargo.toml --release --locked --features faer --test tensor_matrix --message-format=json -- --cfg alea_probe_reverse
# Supply the LP64 OpenBLAS PKG_CONFIG_PATH and LD_LIBRARY_PATH from .#blas:
nix develop .#autodiff --command cargo rustc --manifest-path tools/enzyme-conformance/Cargo.toml --release --locked --features openblas --test tensor_matrix --message-format=json
# Add --cfg alea_probe_reverse to the previous command for BLAS reverse AD.
```

A successful `cargo rustc` build is not execution: run the test executable named
by its JSON compiler-artifact record with `provider_probe --nocapture`, retaining
the same loader path. Keep each probe's stdout/stderr and exit status separately.
The historical I3.2 Faer reverse probe crashed Enzyme type analysis; its BLAS
reverse probe could not link `cblas_dlascl`. Those failures remain preserved; the
request API migration does not qualify direct compiler reverse. Both primals pass independent rectangular
answers. Those negative results do not qualify direct compiler reverse. Separate
explicit-first source fixtures reuse checked provider calls for both operand
VJPs without differentiating through the provider. See [the tensor method specification](../../docs/tensor-ir.md)
for versions, call/ABI visibility and limits. No higher-order derivative or speed
claim follows; feature forwarding and compiler/curvature qualification remain separate.

The combined Faer/LP64-BLAS explicit-first qualification uses named `faer` and
`openblas` features. Both forward the same Alea dependency for build-time source generation and runtime execution; neither is
enabled by default:

```sh
matrix_pkg=$(nix develop .#blas --command printenv PKG_CONFIG_PATH)
matrix_lib=$(nix develop .#blas --command printenv LD_LIBRARY_PATH)
nix develop .#autodiff --command env \
  PKG_CONFIG_PATH="$matrix_pkg" LD_LIBRARY_PATH="$matrix_lib" \
  cargo test --manifest-path tools/enzyme-conformance/Cargo.toml --release --locked \
  --features faer,openblas --test tensor_matrix
```

Provider fixtures emit value/first entries only. HVP/third requests explicitly
reject with `UnqualifiedDerivative`; no silent compiler or reference fallback is
used. The ordinary compensated and mixed-factor compiler controls remain. These
tests do not qualify provider curvature, arbitrary programs or performance.

The existing feature runner can retain untimed release correctness cells, including
build/runtime failures, in its normal reports:

```sh
nix develop .#autodiff --command env \
  PKG_CONFIG_PATH="$matrix_pkg" LD_LIBRARY_PATH="$matrix_lib" \
  bash tools/bench-features.sh --smoke \
  --manifest tools/enzyme-conformance/Cargo.toml --test tensor_matrix \
  --output target/provider-correctness \
  --feature-set '' --feature-set faer --feature-set openblas \
  --feature-set faer,openblas
```

`--test` requires `--smoke` or `--no-run`; it never collects timings. An explicit
`CARGO_TARGET_DIR` may reuse Cargo's feature-keyed cache for correctness cells.
Provider route receipts follow successful independent density/objective VJP and
recovery checks and assert the selected kernel in emitted source (or call
elision for empty output axes). Feature flags
or a successful link alone do not qualify derivatives. The report retains build
and runtime failures, including unavailable features and missing loader paths.

The existing build checks capability resolution and export rejection independently
for each fixture, both scalar modes, all four orders and both curvature selections.
`OriginalCompilerReverse` cannot authorize provider curvature. Matrix/control tests
also exercise independent compensated higher derivatives and singular-score
admission in the combined-provider binary. First-only provider targets have
positive first-order and compile-fail HVP/third trait doctests:

```sh
nix develop .#autodiff --command env \
  PKG_CONFIG_PATH="$matrix_pkg" LD_LIBRARY_PATH="$matrix_lib" \
  cargo test --manifest-path tools/enzyme-conformance/Cargo.toml --release --locked \
  --features faer,openblas --doc
```
