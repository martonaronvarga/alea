# Euclidean phase boundary

## M1 public API (2026-09-27)

`runtime::hamiltonian` is now public. `PhaseWorkspace`, `Phase`, `SignedStep`,
`PhaseError` and the new owned `PhaseState` expose reusable evolution without
exposing independent mutable position/gradient/density/momentum fields.

- `PhaseState::new` takes a checked `PointState` and finite matching momentum.
- `PhaseState::try_update` evaluates into caller-owned scratch, then commits every
  field together. Errors and unwinding preserve the entire old snapshot.
- `PhaseWorkspace::start_from_phase` restarts from cached `(q, logp, grad, p)`
  without evaluation or allocation; `start` still samples metric-correct momentum.
- `Phase::save_into` copies the complete phase and target binding into a same-sized
  preallocated snapshot. Dimension mismatch leaves the destination unchanged.
- Same-sized snapshot `clone_from` reuses all buffers. A different-sized clone
  may allocate; construction always allocates owned storage.
- A failed step consumes its validity token. A compile-fail doctest prevents
  reuse after moving it; runtime tests verify recovery after failures and panics.

The point-only commit method remains internal to `HmcChain`. Its eight-buffer
budget and transition policy are unchanged. Persistent snapshots add three
dimension-sized buffers each (position, gradient, momentum) only when requested.
No tree algorithm, endpoint-selection rule, persistent-momentum transition or
cached metric-dependent energy is implied by storing a phase. NUTS tree assembly
and selection remain M5; the coherent storage prerequisite is now available.

`runtime/tests/phase_state.rs` checks public snapshot semantics, target rebinding,
signed reversal, randomized shapes, non-finite values, model errors and panics.
`phase_allocations.rs` verifies zero allocations for restart/step/save/clone/update
through dimension 129. Both the original HMC allocation tests and BlackJAX endpoint
fixtures continue to exercise the same production integrator.

## Increment 7 record (historical)

Roadmap increment 7 (2026-09-27) extracts `runtime::hamiltonian` from `HmcChain`.
The module is private to the crate; the public HMC API is unchanged. The
`Divergence` enum is still available at `runtime::mcmc::hmc_chain::Divergence`
through a re-export. Legacy `Hmc`/`integrator::leapfrog_step` are unchanged.

## Ownership and validity

`PhaseWorkspace` owns a coherent point cache plus momentum, velocity, candidate
position, and evaluation scratch. Its fields are private. Starting a trajectory
copies a complete live point into existing storage, samples metric-correct
momentum, and returns a borrowed `Phase` token. It does not reevaluate the target.
Shape/target-dimension errors are rejected before consuming RNG.

`Phase::step(self, SignedStep)` consumes that token. Success returns a new token
after the complete kick/drift/evaluate/kick sequence. A model/metric error,
numerical divergence, or unwind does not return a token. Workspace may contain a
partial trajectory, but callers cannot read or commit it as a valid phase. They
must start again from a coherent point. The exclusive workspace borrow prevents
reusing scratch while a token is live.

`Phase::commit(self, live)` consumes a successful token and swaps the entire
validated point into the live state. Dropping a token without committing is a
rejection: no live field is changed. No cloning or mutable cache escape hatch is
available on the token. Momentum is still discarded after a transition; this is
not a persistent-momentum kernel or a stored NUTS tree node.

`Phase::energy(&mut self)` reads the phase and writes only velocity scratch. It
returns a typed error for non-finite Hamiltonian energy. A complete finite
position/momentum/gradient does not imply representable finite energy; HMC drops
the token and rejects on such an error. Energy checking itself does not consume
the token or mutate its position/momentum/cache.

## Boundary between integrator and sampler

The phase module owns momentum generation, signed leapfrog evolution, and
Hamiltonian evaluation. `HmcChain` retains fixed trajectory length, attempted-step
accounting, endpoint energy cutoff, Metropolis selection, and commit decisions.
Model error sources and the public error shape are preserved when internal
`PhaseError` values are mapped back to `HmcError`.

`SignedStep` accepts finite nonzero values in either direction, validated once
outside the integration loop. Public HMC options still accept only positive step
sizes. A test-only explicit-momentum constructor uses the same production
integrator, enabling deterministic endpoint/reference fixtures without exposing a
public momentum override or arbitrary cached gradients.

There are still eight dimension-sized aligned numerical buffers: the live
point's two plus the workspace's six. No new unsafe code, dependencies, dynamic
dispatch, per-step allocations, or full phase snapshots are introduced. The
existing NaN poisoning, coordinate copies, and finite-value checks remain.
Zero-allocation tests are performance evidence, not proof of unchanged throughput.

## Validation

Six module tests cover signed-step validation, all-metric reversal, second-order
energy-error scaling, initial dimension/momentum errors, partial-step model
errors/unwinding/final-kick overflow, and dropping/restarting a phase. The reversal
and order tests were moved from the HMC module and retained, not newly counted as
additional coverage. HMC retains its momentum-generation regression.

The existing [scalar transition oracle and stationary tests](hmc-validation.md)
exercise the refactored path unchanged. They check full RNG consumption and both
accepted/rejected state handling. Allocation tests still measure complete
trajectories, including dimensions 7, 33, and 129 across built-in mass types.

```sh
nix develop --command cargo test --manifest-path crates/Cargo.toml --workspace --all-targets --locked
nix develop --command cargo test --manifest-path crates/Cargo.toml --workspace --doc --locked
nix develop --command cargo test --manifest-path crates/Cargo.toml -p runtime --lib --tests --release --locked
nix develop --command cargo test --manifest-path crates/Cargo.toml -p runtime --lib --tests --features simd --locked
nix develop --command cargo test --manifest-path crates/Cargo.toml -p runtime --lib --tests --features faer --locked
nix develop .#blas --command cargo test --manifest-path crates/Cargo.toml -p runtime --lib --tests --features simd,faer,openblas --locked
nix develop --command env MIRIFLAGS='-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check' cargo miri test --manifest-path crates/Cargo.toml -p runtime --lib --locked hamiltonian
nix develop --command env MIRIFLAGS='-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check' cargo miri test --manifest-path crates/Cargo.toml -p runtime --test hmc_chain --test hmc_validation --locked
```

Miri uses scalar Rust code, ignores long statistical tests, and does not execute
the allocation-counter binary or foreign BLAS. See the canonical roadmap for the
recorded results and unfinished gates.

Recorded on x86-64 Linux with the pinned Nix toolchains: 121 workspace tests and
8 doctests pass. Runtime unit/integration tests pass in release and with SIMD,
Faer, and combined SIMD/Faer/OpenBLAS features. The Rust allocation counters
remain zero for the measured trajectories. Strict Miri passes six phase tests,
six deterministic HMC chain tests, and four deterministic validation tests.
Stable workspace/all-targets compilation, documentation, formatting, and Clippy
complete; existing unrelated lint warnings remain.

### Local timing follow-up (2026-09-27)

The existing `runtime/benches/hmc.rs` benchmark was rerun after extraction, with
the same Gaussian/identity-mass cases, epsilon 0.1, eight steps, 30 samples,
500 ms warmup, and 2 s measurements. Hardware: Intel Core i5-8250U, x86-64 Linux;
pinned nightly 1.97.0 (2026-04-17), default features, fat LTO, no explicit native
CPU flags. No validation/build jobs ran during sampling. CPU frequency and
desktop activity were not controlled.

Arithmetic-mean transition times in microseconds, with 95% bootstrap intervals:

| Dimension | Legacy Hmc | Refactored HmcChain |
|---|---:|---:|
| 8 | 0.694 [0.687, 0.706] | 0.682 [0.670, 0.695] |
| 64 | 4.410 [4.324, 4.509] | 4.366 [4.217, 4.637] |
| 1024 | 84.950 [83.765, 86.197] | 76.791 [75.628, 77.978] |

Intervals overlap at dimensions 8 and 64. At 1024 the checked path took about
9.6% less time than legacy in this run. Both paths timed substantially faster than
the September 25 baseline, including the unmodified legacy path; do not attribute
that cross-run change to this extraction. This is not an isolated pre/post
refactor experiment, proof of zero overhead, or an ESS/sec comparison.
The [raw mean estimates](benchmarks/hmc-2026-09-27.csv) retain nanosecond precision.

Reproduce with:

```sh
nix develop --command cargo bench --manifest-path crates/Cargo.toml -p runtime --bench hmc --locked -- --noplot
```

## External endpoint fixtures (implemented in increment 8)

The first item of the [reference checklist](hmc-validation.md#remaining-reference-gate-and-implementation-order)
is now implemented. [Executed BlackJAX 1.5 fixtures](blackjax-reference.md) now
exercise `start_with_momentum` and the production `Phase::step` from a private
child test module. They record explicit initial `(q,p)`, inverse mass, step/count,
log density/gradient, endpoint `(q,p)`, and energies. Random seeds are not compared
across libraries.

At increment 8, persistent phase state, CI/FFI work, and the std::autodiff/Enzyme
gate remained open. M1 now provides persistent phase storage and the FFI audit;
logistic/funnel/rotated-geometry validation, Enzyme, adaptation and NUTS remain
later work. The historical extraction alone did not complete M1/M3.
