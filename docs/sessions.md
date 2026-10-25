# Sessions, storage, and recovery

A session owns the settings for a fit: model identity, initial points, random
seed, sampler, warmup, draw count, diagnostics, and resource limits. It can run
chains sequentially or on a thread pool supplied by your application.

## Run chains in parallel

Add `rayon = "1.12"` to your dependencies. This example samples a two-dimensional
standard normal target on two worker threads:

```rust
use alea::distributions::Gaussian;
use alea::sampling::{adapt::MetricKind, NutsOptions};
use alea::{
    ensemble::Execution,
    session::{storage::Discard, Kernel, SessionBuilder, Warmup},
};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build()?;
let session = SessionBuilder::new("standard-normal-2d-v1", 2, vec![vec![0.0; 2]; 4])
    .seed([17; 32])
    .draws(500)
    .kernel(Kernel::Nuts(NutsOptions::new(0.2, 8)?))
    .warmup(Warmup::Covariance {
        iterations: 300,
        metric: MetricKind::Diagonal,
        target: 0.8.try_into()?,
        search: None,
    })
    .build()?;
let result = session.run(|_| Ok(Gaussian::new(2)), Execution::Parallel(&pool), &Discard)?;
println!("Run status: {:?}", result.status);
# Ok(())
# }
```

The factory creates a separate target inside each chain worker. This supports
models with local reusable scratch that cannot be shared across threads. Keep the
pool alive until the run finishes. Alea does not install or resize a global pool.

Logical chain IDs come from the input order. A parallel run may interleave output
differently, while the per-chain random streams remain tied to those IDs.

## Choose where draws go

By default a session collects draws in each chain result. It also passes retained
draws to the supplied sink. A sink can observe, stream, or save them:

| Sink | Behavior |
|---|---|
| `Discard` | No external output; collection can still be enabled |
| `Observer` | Calls your synchronous callback for each retained draw |
| `DurableSink` | Writes a new checksummed draw file |
| `ChannelSink` | Sends to a bounded channel; a full queue is an explicit error |

Use `.collect(false)` when the sink owns your output and you do not need another
in-memory copy. Collection requires memory proportional to chains × draws ×
dimension. Diagnostic windows and sampler scratch have separate resource limits.

Sinks do not receive warmup draws. Rejected retained proposals are ordinary
records with repeated positions and must remain in the output.

## Save and resume a cancelled run

Add `tokio-util = { version = "0.7", features = ["rt"] }` to use a cancellation
token. The following program stops after 32 retained draws, saves a checkpoint,
and writes the remaining 96 draws to a new file:

```rust
use alea::distributions::Gaussian;
use alea::sampling::{adapt::MetricKind, NutsOptions};
use alea::{
    ensemble::Execution,
    session::{
        storage::{DrawReader, DurableSink, Observer},
        Checkpoint, Kernel, OwnedDraw, SessionBuilder, Warmup,
    },
};
use std::{fs::File, path::PathBuf};
use tokio_util::sync::CancellationToken;

fn settings() -> Result<SessionBuilder, Box<dyn std::error::Error>> {
    Ok(SessionBuilder::new("normal-1d;data=none;build=1", 1, vec![vec![0.0]])
        .seed([15; 32])
        .draws(128)
        .kernel(Kernel::Nuts(NutsOptions::new(0.1, 6)?)))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output = PathBuf::from("fit-output");
#   let output = std::env::temp_dir().join(format!("alea-guide-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos()));
    std::fs::create_dir(&output)?;
    let cancel = CancellationToken::new();
    let session = settings()?
        .warmup(Warmup::Covariance {
            iterations: 100,
            metric: MetricKind::Diagonal,
            target: 0.8.try_into()?,
            search: None,
        })
        .cancellation(cancel.clone())
        .build()?;
    let partial = session.run(
        |_| Ok(Gaussian::new(1)),
        Execution::Sequential,
        &Observer(|draw: &OwnedDraw| {
            if draw.index == 31 { cancel.cancel(); }
            Ok(())
        }),
    )?;
    let checkpoint = partial.chains[0].checkpoint.as_ref().ok_or("no checkpoint available")?;
    checkpoint.write_new(output.join("chain.checkpoint"), 65_536)?;

    let checkpoint = Checkpoint::read(output.join("chain.checkpoint"), 65_536)?;
    let resumed = settings()?.restart(vec![checkpoint]).collect(false).build()?;
    let sink = DurableSink::create_new(
        output.join("continuation.alea"), resumed.provenance(), 1_000_000,
    )?;
    let result = resumed.run(|_| Ok(Gaussian::new(1)), Execution::Sequential, &sink)?;
    assert_eq!(result.chains[0].completed, 128);

    let mut reader = DrawReader::new(File::open(output.join("continuation.alea"))?, 65_536)?;
    let mut count = 0;
    while reader.next_draw()?.is_some() { count += 1; }
    assert_eq!(count, 96);
    println!("Saved {count} continuation draws; status = {:?}", result.status);
#   drop(reader);
#   drop(sink);
#   std::fs::remove_dir_all(output)?;
    Ok(())
}
```

The output directory must not already exist. The example deliberately uses one
chain to keep the recovery steps short, so a diagnostic warning is expected.
Use several chains to assess a real fit.

`.draws(128)` is the absolute retained end index. On restart it means continue
until 128 total draws have completed, not generate another 128 draws. The fresh
settings use no warmup; restored frozen chains must not adapt again.

The prefix draws remain in `partial.chains[0].draws`. The continuation file holds
only the later segment. To analyze the full chain, combine the segments in draw
index order. Restart diagnostics describe the continuation segment alone.

## Respect the recovery boundary

Checkpoints store the frozen sampler state, point, geometry, and RNG position.
They do not resume a half-finished warmup. Use the same model identity, data,
implementation, target semantics, compiler options, dependencies, and numerical
backend. Recovery re-evaluates the saved point and rejects a cache mismatch.
Bitwise replay across arbitrary builds or CPUs is not promised.

Durable files and checkpoints use create-new paths. Continuing a run writes a new
segment rather than appending blindly to an existing file. A truncated tail does
not invalidate earlier complete frames, but the reader reports that truncation.
Checksums detect accidental corruption; they do not authenticate hostile input.

If a sink rejects a completed draw, the run returns it as `undelivered` together
with the continuing state. When collection is enabled, that draw is also in the
collected output. Reconcile records by `(chain, index)` before retrying output so
you do not count it twice. File sync does not guarantee that a new directory entry
survives power loss; applications needing that guarantee must also sync the
parent directory according to their platform's rules.
