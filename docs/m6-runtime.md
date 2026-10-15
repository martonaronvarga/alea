# M6: independent CPU chains and modern diagnostics

`alea-runtime` now separates numerical transitions, parallel execution, streaming
output, and diagnostic monitoring. Existing `run_chain`, `Draws` and honestly
named classical diagnostic functions are preserved. MCMC algorithms, covariance
warmup defaults and experimental Fisher status are unchanged.

## Ownership and execution

`ensemble::Chain<C,S>` owns one sampler, one ChaCha12 RNG and one streaming sink.
`chain_rng([u8;32], ChainId)` uses explicit independent stream identifiers rather
than scheduler-derived seeds or thread-local RNGs. Keep draws below the cipher's
2^64-block period per stream. Initialize RNG before adaptive warmup and transfer
that same advancing RNG with the frozen chain; never reseed after warmup.

`ensemble::run` uses `Execution::Sequential` or a borrowed, caller-owned Rayon
pool. No global thread pool configuration is mutated. Indexed parallel work
preserves input-order result/error aggregation. Every chain's draw/transition
sequence is identical under either policy for completed fixed-budget runs on
the same numerical backend. Cross-chain callback interleaving is intentionally
unspecified; order records by `(chain, phase, index)` when collecting externally.

CPU work belongs on Rayon, not Tokio executor threads. The runnable
`crates/alea-runtime/examples/monitored_nuts.rs` uses Tokio `spawn_blocking` to
host the synchronous runner and an async subscription task to display reports.
The library itself needs no active Tokio runtime for synchronous execution or
monitoring. Tokio supplies channels/control, not sampler scheduling.

```sh
nix develop .#stable --command cargo run --release --locked --manifest-path crates/Cargo.toml -p alea-runtime --example monitored_nuts
```

## Schema and lifecycle

Version1 `output::Draw` contains logical chain ID, explicit `Warmup`/`Sampling`,
zero-based phase-local index, borrowed committed position, original typed
transition and standardized `TransitionStats`. Rejected repeats are emitted.
Unsupported energy/divergence/tree statistics are absent, not invented zeros.
Metadata adapters cover NUTS, HMC, RWMH and user-defined transitions through
`DiagnosticTransition`. Serialization and draw storage are caller-owned sinks.

`RunOptions::warmup` is discarded fixed-geometry burn-in, **not an adaptive warmup
controller**. M4/M4F consuming warmup remains explicit before this runner. Run
indices and diagnostic windows start fresh on each invocation. Warmup records
are streamed but excluded from retained diagnostics. No log-density-only stopping.

Cancellation uses `tokio_util::sync::CancellationToken` before each transition;
there are no checks inside integrator steps. An in-flight transition completes
and is emitted. A sink can stop via `ControlFlow::Break`. Errors retain their
concrete sampler source and chain identity; all borrowed Rayon jobs join before
returning. Other chains may finish their chunk. Panics propagate without rolling
back committed transitions or RNG. No completion snapshot is guaranteed after a
panic; typed errors produce a terminal failed report. Stop/cancel lengths can
depend on scheduling and are not covered by fixed-budget equality guarantees.

## Bounded diagnostic worker and subscriptions

Each chain retains a fixed-capacity ring of selected coordinates and optional
energies. At reporting boundaries the coordinator publishes an immutable aligned
snapshot through an ArcSwap latest-value mailbox and a capacity-one try-notify
channel. The dedicated diagnostic thread is independent of the Rayon pool.
Sampling never waits for diagnostic calculation or subscribers. Snapshot creation
and copying do cost O(chains × window × observables) at each publication; choose
the reporting chunk accordingly. Per-draw ring updates allocate nothing.

Reports expose their exact half-open retained-draw ranges and all-run divergence/
depth counts. Modern estimates are **window-local**, not estimates for discarded
history. Snapshots with different per-chain ranges are rejected, not silently
truncated to a common length. Old pending snapshots may be overwritten; snapshots
contain contiguous raw windows, so skipped notifications never thin diagnostic
input. Raw retained history, pending/in-flight snapshots and diagnostic scratch
are bounded by the configured window, independent of total run length. User sinks
that collect all draws or subscribers retaining many report Arcs add their own
explicit memory cost. No unbounded queue is used.

`Subscription::latest` clones the current immutable report; `changed().await`
waits for the latest unseen report. Internal watch guards never escape the API.
Slow or dropped subscribers do not block sampler workers. `Monitor::finish`
drains the final mailbox value and joins; Drop also closes/joins, including while
unwinding. Stop publishing before finish. Each publisher/monitor belongs to one
run; do not multiplex unrelated runs through it. Finish can wait for the current
diagnostic computation, so async callers should join it on a blocking thread.

## Mathematics and stopping

`diagnostics::modern::summarize` provides pooled average-rank normal scores,
split/folded/max R-hat, cross-chain Geyer initial-positive/monotone bulk and tail
ESS, raw mean ESS and mean MCSE. Odd splits omit the middle draw. Tail ESS uses
pooled 5%/95% quantile indicators; mean MCSE uses raw mean ESS, **not bulk ESS**.
Autocovariances use zero-padded RustFFT transforms and the biased denominator n.
Rank and fold calculations preserve exact ties; scaling guards raw variance and
overflowing deviations. Undefined, constant, ragged and nonfinite inputs return
typed errors. Constant quantile indicators use the ArviZ ESS convention, while
constant observables still fail R-hat. These are scalar-observable diagnostics,
not multivariate convergence guarantees.

E-BFMI is mean squared consecutive energy difference divided by sample variance.
HMC/NUTS adapters report the initial Hamiltonian after fresh momentum sampling;
this is neither energy error nor log density. Any missing energy invalidates
window E-BFMI rather than bridging gaps. Divergences and maximum-depth saturation
remain separate counts; RWMH does not pretend to have Hamiltonian diagnostics.

Stopping is disabled unless a `StopPolicy` is supplied. It requires a minimum
number of post-warmup observations **within each diagnostic window**, bounds
rank/folded R-hat and both ESS values for every selected coordinate, optionally
limits absolute mean MCSE and requires E-BFMI, and refuses any observed divergence
or saturation. Undefined/ragged/failed reports cannot pass. A minimum larger than
the window simply never passes. Repeated checking and a rolling window do not
prove convergence and can bias stopped estimates; prefer fixed-budget runs for
confirmatory inference. This does not promote Fisher or repair difficult funnels.

Reference definitions and independent execution:
[Vehtari et al.](https://arxiv.org/abs/1903.08008),
[ArviZ fixtures and provenance](../tools/diagnostics-reference/README.md).
Concurrency contracts follow the canonical
[Rayon pool](https://docs.rs/rayon/latest/rayon/struct.ThreadPool.html) and
[Tokio watch](https://docs.rs/tokio/latest/tokio/sync/watch/index.html) APIs.

## Validation

The offline ArviZ0.23.4 suite fixes 14 cases before comparison, including stuck,
shifted, differently scaled, tied, odd, antithetic, ragged and nonfinite inputs.
It compares numerical estimates at 2e-9 tolerance; invalid cases remain failures.
Runtime tests cover stream equality across pool sizes/input order, real NUTS
draw/statistic equality, RNG preservation on invalid/pre-cancelled runs,
mid-chunk cancellation, observer stops, typed errors, panics, latest-only slow
subscribers, final-snapshot shutdown races, warmup exclusion, independent window
recalculation, stopping minimums and no per-draw ring allocation. A live-allocation
test compares 128 versus16,384 retained draws at the same memory budget.

Completion gates on October14: full stable `tools/ci/check.sh` passed (architecture,
migration, workspace tests, all-target Clippy and rustdoc). The monitored NUTS
release example completed four chains of256 discarded plus2048 retained draws,
delivered the final1024-draw window and shut down cleanly. Fixture regeneration
with `--check` matched exactly. Strict-provenance/symbolic-alignment Miri passed
the bounded-ring/missing-energy test. This is not a full Miri, Enzyme or native
BLAS feature-matrix claim. Alea skill guidance drove the separation of CPU work,
bounded messaging, explicit lifecycle and numerical reference validation.
Final runtime tests and all-target Clippy also passed after the equal-quantile
endpoint guard, negative stopping-policy checks and lockfile cleanup.
