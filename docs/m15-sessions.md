# Reference inference sessions (M15)

`alea_runtime::session::{SessionBuilder, Session}` composes the S2 covariance
warmup, frozen leapfrog HMC/multinomial NUTS and density-only RWMH components.
The session layer adds execution and persistence policy; it adds no transition
map, retained adaptation or convergence guarantee. The API remains pre-release S2.

The executable external example is
[`tools/downstream-conformance/examples/session.rs`](../tools/downstream-conformance/examples/session.rs):

```sh
cargo run --manifest-path tools/downstream-conformance/Cargo.toml \
  --example session -- /tmp/alea-session-example-new
```

The directory must be new. The example constructs a native model, performs
covariance warmup, cancels retained NUTS after 32 draws, saves/loads a checkpoint,
then streams the remaining 96 draws to a new durable file. It deliberately uses
one chain and reports unavailable multi-chain diagnostics as a warning.

## Configuration and execution

Supply a model/data/build identity, unconstrained dimension and one finite initial
point per chain. Logical IDs are the input-order indices, starting at zero. Set
`Kernel::{Hmc, Nuts, Rwmh}` with their existing validated option types, a retained
end index, seed, coordinate observables, diagnostics and resource limits. Defaults
are explicit in `SessionBuilder`: NUTS step 0.1/depth 8, 1,000 retained draws,
zero seed, collection enabled and **no warmup**. Adaptation requires an explicit
`Warmup::Covariance`, `Warmup::GradientDiagonal` (NUTS only), or
`Warmup::RwmhScale` selection.

Covariance warmup uses the existing initial/slow/final window schedule, diagonal
or dense covariance, a caller-selected acceptance target and optional bounded
step search. NUTS covariance warmup now uses native NUTS trees and bounded one-sided step
search; fixed HMC warmup uses its requested
trajectory length. RWMH uses identity proposal covariance and optional scale dual
averaging, without gradients. Warmup is consumed before retained sampling; it
never appears in sinks or retained diagnostics. Zero warmup consumes no warmup RNG.

The measured fast NUTS profile is explicit:

```rust,ignore
let kernel = Kernel::Nuts(NutsOptions::new(0.2, 8)?
    .with_step_size_jitter(0.1)?.with_position_turning(true));
let warmup = Warmup::GradientDiagonal {
    iterations: 512, target: 0.8.try_into()?, search: None, final_restart: false,
};
let session = builder.kernel(kernel).warmup(warmup).build()?;
```

This exposes the same frozen profile as the native parity experiment. It keeps
covariance as the reference option and does not guarantee better diagnostics on
every model. Jitter and turning are preserved through checkpoint/restart.

`Session::run` accepts a gradient target factory; `run_density` accepts a density-only
factory for RWMH. Each factory runs inside its chain worker, so its returned target
can contain local `RefCell` scratch and need not implement `Send` or `Sync`. The
factory is `Sync`; returned error types are `Send`. Factories return the target's
concrete error type, including model lowering errors when relevant. Target shape
is checked before constructing the sampler. Initialization failure remains a
chain-local typed outcome.

`Execution::Sequential` and `Execution::Parallel(&pool)` use the same API. The
caller creates and owns the Rayon pool; admission rejects pools larger than the
session's thread limit. Workers do not enter Tokio, install a tracing subscriber,
change process-global parallelism or detach threads. Structured sink callbacks
provide run identity and records; applications own logging/subscription policy.
Every worker joins before sink finalization and return, including on worker panic.
A failed chain does not silently cancel valid work in independent chains.

## Outcomes and diagnostics

`RunOutcome` carries provenance, the requested `RunPlan`, input-order chain results,
final diagnostics and any sink finalization failure. Each chain includes absolute
completion count, actual completed warmup count, optional collected new draws,
last undelivered draw, frozen checkpoint and original typed error. Outcomes classify:

| Status | Meaning |
| --- | --- |
| `Success` | Retained budget completed and configured final diagnostic gates passed. |
| `DiagnosticWarning` | Budget completed; undefined/failed R-hat/ESS, divergence, numerical rejection or maximum tree depth needs review. |
| `Cancelled` | User token or cooperative deadline observed before work completed. |
| `NumericalExhaustion` | Bounded warmup arithmetic/search failed, invalid numeric evaluation occurred, or consecutive numerical rejection limit was reached. |
| `BackendFailure` | Original target/backend error, contract/restart mismatch, worker/observer panic, sink loss or finalization error. |

Precedence is backend failure, numerical exhaustion, cancellation, diagnostic
warning, success. A backend error never becomes a numerical rejection. Divergences
are valid repeated/admitted chain states under the existing kernel contracts; the
session retains them and stops only at the explicit consecutive numerical rejection
limit (default 100). RWMH numerical rejection is a separate owned-record flag;
Hamiltonian-only statistics remain absent. The numerical streak survives restart.

Diagnostics use bounded contiguous retained windows, including rejected repeats.
Every configured coordinate must have defined R-hat, bulk ESS and tail ESS meeting
its thresholds; one chain or a short/constant/ragged window yields a warning.
Depth saturation and numerical events across this invocation also produce warnings.
These gates are screening, not a proof of convergence. No optional-stopping rule
is enabled. RunPlan stores observable names, initial points, starting indices,
adaptation, numerical thresholds, storage policy and resource budgets. Diagnostics
on restart cover **new draws only**, labeled with absolute retained ranges;
nonmatching chain ranges remain an explicit shape error. Earlier history and
warnings are not retrospectively reconstructed from a checkpoint.

## Owned output, backpressure and durability

`OwnedDraw` schema 1 is separate from borrowed `output::Draw<'_, T>`. It owns the
committed position and common transition statistics; its `(chain, index)` pair is
the stable key. Session sinks receive only retained sampling records. For collected
runs, all new valid draws stay in the outcome even if a later transition or sink
fails. In streaming runs, acknowledged draws belong to the sink; a completed draw
that the sink rejects or panics on is returned as `undelivered`. A sink error does
not roll back chain state or RNG. When collection is enabled, `undelivered` also
exists in collected output and must not be counted twice.

`ChannelSink::bounded(capacity)` uses a bounded standard channel and fail-fast
`try_send`. Full and disconnected queues are distinct errors; there is no blocking
backpressure, implicit drop or dependence on an async consumer running on the same
pool. The last undelivered record and checkpoint let an application reconcile the
prefix before continuing. A receiver controls its own lifetime. Custom observers
are synchronous; their own time/memory budgets remain the caller's responsibility.

`DurableSink::create_new(path, provenance, maximum_bytes)` uses `create_new`, never
truncation or reopening an old stream. It validates the run identity before any
target factory runs. File writes serialize complete frames under one mutex;
cross-chain interleaving is unspecified and per-chain order is preserved. An I/O
error poisons further writes. A finished durable sink cannot be reused for another
run; continuation uses a new segment. `finish` flushes and `sync_all`s file contents after
workers join; failure is retained in the outcome. There is no detached I/O worker.

The binary stream is `ALEADW01`, one framed provenance header, then framed draws.
A frame is little-endian `u64 payload_length`, payload and `u64 FNV-1a` checksum.
F64 values use their exact IEEE bits; optional values have checked 0/1 presence
tags. The checksum detects accidental corruption, not hostile modification. The
header records schema, target/build identity, implementation identifier, dimension
and 256-bit seed. Draw payloads record schema, chain, sampling-phase tag, absolute
index, position, all `TransitionStats` fields and numerical-rejection flag. Changing
the format requires a new schema/magic; unknown versions are rejected.

`DrawReader` reads one frame at a time with a caller-selected maximum frame length.
Clean EOF differs from truncation/checksum/format failure. Complete frames before
an interrupted tail remain readable. A write error may occur after some bytes
reached the filesystem; the application must reconcile keys before retrying a
returned undelivered record. No atomic exactly-once acknowledgement across a file
write and checkpoint is claimed. A power loss can truncate a tail. File-content
sync does not promise parent-directory entry persistence; applications needing that
crash guarantee must sync the directory using their platform/filesystem policy.
The full RunPlan is an outcome value, not silently encoded as arbitrary user data
in the draw stream; persist it alongside streams for experiment archival.

## Checkpoint and replay scope

A checkpoint exists only after geometry freezes, at a completed retained boundary
(including zero retained draws after warmup). Mid-warmup checkpointing is explicitly
unsupported. Cancelling/failed warmup returns no checkpoint and no partially adapted
sampler. Restart requires a complete input-order checkpoint set. If another chain
has not frozen, the full ensemble must restart from its original initialization;
there is no mixed adaptive/frozen restart mode. A complete prefix of logical IDs
starting at zero can continue as a smaller session. A loaded checkpoint with a
warmup phase tag is rejected. Warmup must be
restarted from the supplied initial point/seed when it has not frozen.

Checkpoint schema 2 uses `ALEACP02` and one checksummed frame containing provenance,
sampling phase, chain ID, absolute completion index, ChaCha12 word position, frozen
kernel settings (including NUTS jitter and position turning), frozen mass
representation, position/log density/gradient cache,
RWMH normal-generator spare and consecutive numerical rejection streak. Uniform
stream identity is the logical chain ID. Identity, diagonal masses and full
column-major lower mass factors retain their original encoding tags 0–2. Learned
inverse diagonal masses and lower inverse-mass factors use distinct tags 3–4.
Every representation is saved losslessly; restart never inverts, refits or
refactorizes it. HMC/NUTS momentum normals have no persistent spare. RWMH's cached
second polar variate **is** required, especially for odd dimensions.

`Checkpoint::write_new` never replaces an existing path and flushes/syncs contents.
`Checkpoint::read(path, maximum_bytes)` bounds total bytes, validates versions,
geometry, RNG range and caches, and rejects trailing/truncated/corrupt data. A
checkpoint is a point/RNG/policy continuation, not the draw history or an atomic
transaction with another file. Durable output and checkpoints should use fresh
segment paths; concatenate segments by `(chain,index)` after reconciling any
undelivered record.

The current implementation identity includes `session/2` and `numerics/3`, covering runtime SIMD,
relative-energy weight accumulation, high-dimensional kinetic reduction and
Ziggurat momentum generation. It rejects `numerics/2` checkpoints as well as
earlier implementations; draw stream schema remains unchanged.
Schema-one checkpoints omitted nondefault NUTS jitter/turning settings and could
silently reconstruct a different kernel. They are explicitly rejected with
`CheckpointError::Format`; restart those old runs from their original model/seed
instead of reinterpreting their bytes. Schema-two restart also rejects a requested
jitter or turning policy different from the frozen checkpoint.
Earlier implementation checkpoints are rejected. Earlier schema-one draw streams
remain readable for analysis; reading data does not authorize checkpoint replay.

`SessionBuilder::restart` accepts one checkpoint per logical input chain, consumes
no new warmup (requesting adaptation again is rejected), and uses the checkpoint's frozen step and geometry. The retained
budget is an **absolute end index**. Admission checks chain, identity, dimension,
seed and trajectory policy; the requested pre-warmup step may differ because the
frozen adapted value is authoritative. Reconstruction evaluates the saved point
once without RNG, then requires bit-identical density and gradient. It refuses a
mismatch instead of silently refreshing the target. The caller's model/data/build
identity must change whenever any mathematical or executable input changes; a
single matching point cannot prove equality of two arbitrary target functions.

For the same executable, floating-point environment, fixed deterministic target,
configuration and logical streams, completed fixed-budget draws match between
sequential/Rayon and streaming/collected execution. Checkpoint continuation matches
the uninterrupted suffix, including transition statistics and frozen endpoint.
Tests cover HMC, NUTS, odd-dimensional RWMH and diagonal/dense covariance warmup. No bitwise
promise spans compiler/CPU/libm/features/version changes or target side effects.
Target re-evaluation on restart is observable to counters/external backends. After
a failed attempted transition, RNG consumption is preserved rather than rolled
back; replay resumes from that actual consumed stream. Cancellation lengths and
cross-chain output order depend on scheduling. No thread count changes RNG identity.

## Resource and failure boundaries

`Resources` explicitly caps chain count, supplied pool threads, dimension, tree
depth, fixed-HMC steps, warmup transitions, diagnostic float values, collected
record/coordinate bytes and conservative sampler/covariance float storage. Products use
checked arithmetic before work; dense restart geometry has the same admission
budget as fresh dense warmup. Warmup windows follow the finite reference schedule;
step-search probe counts/bounds are explicit in `SearchOptions`. NUTS work can grow
as `2^depth`; its workspace grows linearly with depth and dimension.

Collection reserves space for the requested new draw records; the collection byte limit
includes records and coordinates, with allocator overhead additional. Streaming has
O(chains × (sampler workspace + dimension + window × observables)) live numerical
storage; it does not retain a hidden full draw history. Each in-flight record,
checkpoint and reader frame is additionally bounded by dimension/geometry/frame
budgets. Channel capacity and durable total bytes are separate explicit sink limits.
User targets and custom sinks own their scratch/allocation limits. Resource budgets
are admission policy, not a process RSS sandbox.

`wall_time` starts at run entry and polls the same safe boundaries as the user token.
It includes factory, warmup and observer time, but cannot interrupt user code,
an in-flight trajectory, bounded step search or filesystem syscall. A completed
last draw wins a racing cancellation; it remains completed. Worker and observer
unwinds are caught and classified; Rust's process panic hook still runs. Abort/OOM
and indefinitely blocking foreign/user code cannot be recovered by this library.

Regression gates are runtime `tests/session.rs`, storage unit tests, existing
ensemble/monitor shutdown tests, MCMC warmup/property/reference tests and the
separately locked downstream session example. They cover serial/Rayon and output
parity, cancellation races/replay, malformed checkpoints, sink loss/panic/write
failure, join/finalization, non-overwrite/byte budgets and streaming peak allocations
with 1,000 versus 100,000 draws at the same 64-draw diagnostic window. These are
bounded operational tests, not platform-wide durability certification or S3/S4
promotion.
