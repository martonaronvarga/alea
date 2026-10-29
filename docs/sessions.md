# Sessions, storage, and recovery

A session owns the settings for a fit: model identity, initial points, random
seed, sampler, warmup, draw count, diagnostics, and resource limits. It can run
chains sequentially or on a thread pool supplied by your application.

## Run chains in parallel

Add `rayon = "1.12"` to your dependencies. This example samples a two-dimensional
standard normal target on two worker threads:

```rust
use alea::distributions::StandardNormalTarget;
use alea::sampling::{NutsOptions, adapt::MetricKind};
use alea::{
    ensemble::Execution,
    session::{SamplerConfig, SessionBuilder, Warmup, storage::Discard},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build()?;
    let session = SessionBuilder::new("standard-normal-2d-v1", 2, vec![vec![0.0; 2]; 4])
        .seed([17; 32])
        .draws(500)
        .sampler(SamplerConfig::Nuts(NutsOptions::new(0.2, 8)?))
        .warmup(Warmup::Covariance {
            iterations: 300,
            metric: MetricKind::Diagonal,
            target: 0.8.try_into()?,
            search: None,
        })
        .build()?;
    let result = session.run(
        |_| Ok(StandardNormalTarget::new(2)),
        Execution::Parallel(&pool),
        &Discard,
    )?;
    println!("Run status: {:?}", result.status);
    Ok(())
}
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
use alea::distributions::StandardNormalTarget;
use alea::sampling::{NutsOptions, adapt::MetricKind};
use alea::{
    ensemble::Execution,
    session::{
        Checkpoint, SamplerConfig, OwnedDraw, SessionBuilder, Warmup,
        storage::{DrawReader, DurableSink, Observer},
    },
};
use std::fs::File;
use tokio_util::sync::CancellationToken;

fn settings() -> Result<SessionBuilder, Box<dyn std::error::Error>> {
    Ok(
        SessionBuilder::new("normal-1d;data=none;build=1", 1, vec![vec![0.0]])
            .seed([15; 32])
            .draws(128)
            .sampler(SamplerConfig::Nuts(NutsOptions::new(0.1, 6)?)),
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output = std::env::temp_dir().join(format!(
        "alea-guide-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
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
        |_| Ok(StandardNormalTarget::new(1)),
        Execution::Sequential,
        &Observer(|draw: &OwnedDraw| {
            if draw.index == 31 {
                cancel.cancel();
            }
            Ok(())
        }),
    )?;
    let checkpoint = partial.chains[0]
        .checkpoint
        .as_ref()
        .ok_or("no checkpoint available")?;
    checkpoint.write_new(output.join("chain.checkpoint"), 65_536)?;

    let checkpoint = Checkpoint::read(output.join("chain.checkpoint"), 65_536)?;
    let resumed = settings()?
        .restart(vec![checkpoint])
        .collect(false)
        .build()?;
    let sink = DurableSink::create_new(
        output.join("continuation.alea"),
        resumed.provenance(),
        1_000_000,
    )?;
    let result = resumed.run(|_| Ok(StandardNormalTarget::new(1)), Execution::Sequential, &sink)?;
    assert_eq!(result.chains[0].completed, 128);

    let mut reader = DrawReader::new(File::open(output.join("continuation.alea"))?, 65_536)?;
    let mut count = 0;
    while reader.next_draw()?.is_some() {
        count += 1;
    }
    assert_eq!(count, 96);
    println!(
        "Saved {count} continuation draws; status = {:?}",
        result.status
    );
    Ok(())
}
```

The program saves its files in a new temporary directory. Print `output.display()`
to find them, or change `output` to a directory of your choice that does not already exist. The example deliberately uses one
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

## Export a named sampling table

Enable Alea's `analysis` feature to write a CSV table and a JSON manifest. This
format carries named physical parameters, optional free coordinates, transition
statistics, scalar predictions and pointwise likelihoods. It is separate from a
checkpoint: use checkpoints to resume a computation.

The following example writes to memory. Pass files or other `std::io::Write`
streams instead to save a fit without keeping all rows in memory.

```rust
use alea::analysis::{AnalysisSpec, AnalysisWriter, ChainStatus, Provenance};
use alea::distributions::modeling::normal;
use alea::math::buffer::OwnedBuffer;
use alea::model::{assembly::ModelBuilder, prediction::NamedOutput};
use alea::output::{ChainId, DiagnosticTransition, Draw, Phase, SCHEMA_VERSION};
use alea::sampling::{HmcOptions, Nuts, adapt::{HmcWarmup, WarmupOptions}, config::TreeDepth};
use rand::{SeedableRng, rngs::SmallRng};
use std::collections::BTreeMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut b = ModelBuilder::new();
    let mean = b.real("mean")?;
    let zero = b.constant(0.0)?;
    let one = b.constant(1.0)?;
    let prior = normal(&mut b, mean.expr(), zero, one)?;
    b.prior("mean prior", prior)?;
    let likelihood = normal(&mut b, one, mean.expr(), one)?;
    b.likelihood("measurement", likelihood)?;
    let model = b.compile()?;
    let parameters = NamedOutput::new(model.coordinates(), model.layout().clone(), ())?;
    let provenance = Provenance {
        model: "normal-mean-v1".into(),
        data: "one-observation:y=1".into(),
        source: "this-example-v1".into(),
        method: "NUTS".into(),
        settings: BTreeMap::from([("seed".into(), "17".into())]),
    };
    let mut spec = AnalysisSpec::new(parameters, provenance, vec![ChainId(0)]);
    spec.save_unconstrained = true;
    spec.predictions = vec!["expected measurement".into()];
    let mut writer = AnalysisWriter::new(Vec::new(), spec)?;
    let target = model.select_posterior()?.into_target()?;
    let mut rng = SmallRng::seed_from_u64(17);
    let (hmc, _) = HmcWarmup::new(
        &target, OwnedBuffer::from_fn(1, |_| 0.0),
        HmcOptions::new(0.1, 1)?, WarmupOptions::new(128),
    )?.run(&mut rng)?;
    let mut chain = Nuts::from_hmc(hmc, TreeDepth::try_from(6)?)?;
    for index in 0..64 {
        let transition = chain.step(&mut rng)?;
        let q = chain.point().position();
        let draw = Draw {
            schema_version: SCHEMA_VERSION, chain: ChainId(0),
            phase: Phase::Sampling, index, position: q,
            stats: transition.diagnostic_stats(), transition: &transition,
        };
        writer.write_draw(&draw, &[q[0]])?;
    }
    let mut manifest = Vec::new();
    let csv = writer.finish(&mut manifest, &[(ChainId(0), ChainStatus::Completed)])?;
    println!("CSV: {} bytes; manifest: {} bytes", csv.len(), manifest.len());
    Ok(())
}
```

For pointwise output, use `AnalysisSpec::from_likelihood` with a validated
[`PointwiseLikelihood`](modeling.md#pointwise-likelihoods-and-predictive-units).
It takes the exact named layout and predictive units from that model. Alternatively,
attach it to `spec.likelihood`; the writer rejects a different parameter schema.
Physical values use the model's charts without a density-change Jacobian.

Supply the full planned chain list before writing, and an outcome for every chain
at `finish`, including chains that never initialized. `Completed` describes the
computation, not convergence or successful predictive reporting. Interrupted and
failed outcomes require a reason. Warmup and sampling are distinct phases.
Within each chain and phase, indices are consecutive; the first may be nonzero,
so a resumed segment keeps its absolute indices. Chains may interleave.

After a valid logical key is admitted, an invalid numerical payload or pointwise
failure writes a `failure` row and returns the original error. That index is
consumed. Handle the error to retain a partial result, then seal the artifact with
explicit chain outcomes. Do not retry the same index. `write_failure` records
other unsuccessful attempts. Failed rows contain an explanation and absent
numeric fields, never invented parameter values or a probability of zero.
Invalid versions, unknown chains, duplicate indices and gaps are rejected before
writing. An I/O error or a callback panic poisons the stream; discard it.

### CSV and manifest contract

The manifest identifies `alea-analysis`, schema version `1`, kind `sampling`.
Its `csv.columns` list is the exact header order:

- `chain`, `phase`, `draw`, `kind`, `error` identify a successful draw or failure.
- `parameter:NAME` contains constrained coordinates. Free coordinates, when
  requested, are `unconstrained:0`, `unconstrained:1`, and so on.
- `stat:...` columns carry the common transition statistics. Unsupported optional
  statistics are empty; a supported false or zero remains `0`.
- `log_likelihood:UNIT` and `prediction:NAME` follow their declared metadata order.

Names and explanations are UTF-8. All fields are quoted; embedded quotes are
doubled, and records end with CRLF. Fields may contain commas, quotes and line
breaks. Floating-point values use round-trip decimal binary64 representations,
including negative zero. NaN and infinities are rejected in this version. Integer
chain IDs and draw indices retain their exact unsigned 64-bit values; do not
convert them through floating-point numbers.

The JSON manifest preserves parameter names, block/dimension/coordinate labels,
free-coordinate dimension, optional physical units, predictive-unit definitions,
model/data/source identities, method settings, per-phase ranges and failure
counts, and every planned chain's status. `spec.units` associates a physical unit
with an exact value-column name. For reproducible applications, use content digests
for model/data identities and record the actual source revision, build features,
toolchain, seeds and inference settings. Labels alone do not verify provenance.
Observed data remain identified by their source identity and predictive-unit
observation identifiers; this format does not embed a copy of the dataset.

The writer flushes CSV before emitting its SHA-256 digest and counts in the
manifest. Validate both files together. It performs no file creation, atomic
rename or filesystem synchronization on your behalf. A CSV alone or a partially
written manifest is not a sealed result. This sampling contract does not turn
optimization estimates into posterior draws or fabricate standard errors.

### Read and validate an artifact

The source checkout includes a reader requiring only Python's standard library:

```sh
python3 tools/downstream-conformance/python/analysis.py fit.json draws.csv
```

Add `--rectangular` to require completed chains with identical nonempty draw-index
coordinates and no failed rows. The reader checks the digest, schema, names,
counts, ranges, numeric values and statistic availability. It retains warmup,
partial chains and failure records. It never pads, truncates or silently drops
chains. In Python, `artifact.rectangular(chains=[...])` permits an explicitly
chosen completed subset; `artifact.records` preserves the full validated result.
The reader materializes records in memory; the native writer streams them.
