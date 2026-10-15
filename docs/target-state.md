# Fused evaluation and transactional point state

Roadmap increment 4 (2026-09-25) adds `kernels::target`. It is additive: the old
density traits, mutable `ChainState`, and HMC API remain available and unchanged.

## Target contract

```rust
pub trait LogDensityGradient {
    type Error: std::error::Error + 'static;
    fn dimension(&self) -> usize;
    fn logp_grad(&self, position: &[f64], gradient: &mut [f64])
        -> Result<f64, Self::Error>;
}
```

The target operates in unconstrained coordinates. Its mathematical density and
dimension must stay fixed while a cached point borrows it. Interior mutability
may hold scratch or counters, but must not change evaluation results. A shared
reference alone cannot enforce mathematical purity; this is an explicit trait
contract. Models should keep scratch chain-local and overwrite every gradient
coordinate on success.

`evaluate(target, position, gradient)` wraps exactly one fused call with checks:

- Position and gradient lengths must equal the declared dimension.
- Input coordinates must be finite before model invocation.
- Returned log density and all gradient entries must be finite.
- Model errors retain their concrete type and `Error::source()` chain.

Before calling the model, the wrapper fills the gradient with NaNs. This prevents
a partial write from silently reusing old finite coordinates. The fill and output
scan are linear passes, deliberately included in the correctness baseline. There
is no unchecked fast path yet. Measure their end-to-end cost before changing this
policy; allocation freedom does not establish maximum throughput.

`EvaluationError` distinguishes shape mismatch, changed target dimension,
non-finite position, model failure, non-finite log density, and missing/non-finite
gradient. Negative-infinite log density is **not** a valid cached point. Future
sampler transitions must explicitly classify support failures, backend failures,
and numerical divergences rather than flattening them into a boolean.

The raw trait method leaves gradient contents unspecified on failure. `evaluate`
leaves output untouched for shape/input errors, but its scratch output can be dirty
after invoking the model. Model panics propagate; no panic-to-error conversion or
foreign unwind containment is supplied here.

## Cached point and workspace

`PointState<'a, T>` owns aligned position and gradient buffers and a finite log
density, and borrows one target for its lifetime. Its fields are private, getters
are read-only, and updates cannot supply a replacement target. It deliberately
does not implement the legacy mutable state traits.

Construction consumes the position buffer without copying and evaluates once.
`try_update(position, workspace)` evaluates into an `EvaluationWorkspace` first.
Only after success does it copy coordinates, swap gradient buffers, and commit
the log density. No fallible/user operation runs during commit. Failure or an
unwinding panic leaves all old point fields intact. Dirty workspace contents are
overwritten on the next evaluation, so recovery does not require allocating again.

Here “atomic” means coherent updates under Rust's exclusive mutable borrow, not
an atomic CPU operation or concurrent shared state. There is no momentum or phase
point yet. Empty, zero-dimensional targets are supported; sampler-specific model
restrictions will be validated separately.

An executable version of this example lives in the `PointState` rustdoc:

```rust
use kernels::{buffer::OwnedBuffer, dist::Gaussian, target::{
    EvaluationWorkspace, FusedAdapter, PointState,
}};

fn example() -> Result<(), Box<dyn std::error::Error>> {
    let target = FusedAdapter::new(&Gaussian, 2);
    let mut point = PointState::new(&target, OwnedBuffer::new(2))?;
    let mut workspace = EvaluationWorkspace::new(2);
    point.try_update(&[1.0, 2.0], &mut workspace)?;
    assert_eq!(point.log_density(), -2.5);
    assert_eq!(point.gradient(), &[-1.0, -2.0]);
    Ok(())
}
```

`FusedAdapter` borrows an existing `FusedLogDensity` with an explicit dimension.
It calls only the fused method, never separate value and gradient methods. It
checks shape even when called directly, but cannot recover domain/backend errors
discarded by a legacy implementation or infer its true model dimension. New
analytic and Enzyme targets should implement `LogDensityGradient` directly.

## Performance and validation

The successful update path copies coordinates and swaps gradient ownership; it
does not resize or allocate. The allocation regression measures 100 updates and
100 repetitions each of typed model, dimension, and input-finiteness errors. It
observes zero allocations and zero allocated bytes after construction for the
allocation-free Gaussian fixture. Arbitrary models and owned model errors may
allocate; no blanket claim is made about them.

The [allocation counter](https://docs.rs/allocation-counter/0.8.1/allocation_counter/)
is pinned as a dev dependency and referenced only by the dedicated native test
binary. Production code does not install a global allocator. The allocation test
is excluded under Miri; Miri runs the separate correctness suite instead.

Regression coverage includes randomized dimensions, complete cache agreement,
shape/input rejection before evaluation, partial gradient writes, concrete error
sources, all non-finite categories, target dimension changes, panic recovery,
legacy fused-only dispatch, dynamic trait dispatch, and empty targets. Buffer
addresses also verify coordinate-allocation retention and gradient-buffer reuse.

```sh
nix develop --command cargo test --manifest-path crates/Cargo.toml -p kernels --test target --test target_allocations --locked
nix develop --command cargo test --manifest-path crates/Cargo.toml -p kernels --doc --locked
nix develop --command env MIRIFLAGS='-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check' \
  cargo miri test --manifest-path crates/Cargo.toml -p kernels --test target --locked
```

### Validation record (x86-64 Linux, pinned toolchains)

- Default workspace: 97 tests, benchmark smoke tests, and 7 doctests passed.
- The 12 target/state regressions passed in debug, release, and strict Miri with
  symbolic alignment checks. This does not cover foreign execution or Enzyme.
- The native allocation regression passed in debug and release, with zero
  allocations/bytes in the measured update and failure paths.
- Stable workspace/all-targets compilation, workspace documentation, and Rust
  formatting passed. Clippy completed with existing warnings.

## HMC migration follow-up

The following increment-4 checklist is now implemented by the additive
[`HmcChain` API in increment 5](hmc-chain.md), with a local Criterion baseline.
Broad statistical/reference validation remains open. `PointState` also supports
cloning a complete cache; same-dimension `clone_from` reuses its allocations and
copies the target borrow together with all cached values. This is whole-point
replacement, not a way to change the target of `try_update`.

1. Bind HMC to one target and accept only validated cached points.
2. Introduce reusable phase/proposal storage with a fallible leapfrog boundary.
3. Preserve rejected/failed point caches, propagate backend errors, and report
   divergences through typed transition statistics.
4. Remove HMC's starting cache refresh only on this new path; require exactly
   `L` fused calls for `L` successful leapfrog steps after initialization.
5. Measure full transitions with allocation counters and Criterion, retaining
   metric momentum, reversibility, energy-order, and stationary-moment tests.

Legacy HMC still takes mutable `ChainState` and still refreshes its initial cache
on every transition (`L + 1` fused calls). The existing refresh must not be removed
just because a new immutable state API now exists alongside it.
