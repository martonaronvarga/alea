//! One session API for native models, bounded output and cancelled/resumed NUTS.
use alea_core::{
    model::{
        ModelError,
        assembly::{AssemblyError, CompiledModel, ModelBuilder},
    },
    target::LogDensityGradient,
};
use alea_mcmc::{NutsOptions, adapt::MetricKind};
use alea_runtime::{
    ensemble::Execution,
    session::SessionBuilder,
    session::{
        Checkpoint, Kernel, Status, Warmup,
        storage::{DrawReader, DurableSink, Observer},
    },
};
use std::{fs::File, path::Path};
use tokio_util::sync::CancellationToken;

fn model() -> CompiledModel {
    let mut b = ModelBuilder::new();
    let q = b.real("location").unwrap();
    let square = b.multiply(q.expr(), q.expr()).unwrap();
    let minus = b.constant(-0.5).unwrap();
    let logp = b.multiply(square, minus).unwrap();
    b.factor(logp).unwrap();
    b.compile().unwrap()
}
fn factory(
    _: alea_runtime::output::ChainId,
) -> Result<impl LogDensityGradient<Error = ModelError<AssemblyError>>, ModelError<AssemblyError>> {
    model().into_target()
}
fn spec() -> SessionBuilder {
    SessionBuilder::new(
        "standard-normal;session-example;build-v1",
        1,
        vec![vec![0.0]],
    )
    .seed([15; 32])
    .draws(128)
    .kernel(Kernel::Nuts(NutsOptions::new(0.1, 6).unwrap()))
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A caller-selected empty directory keeps every durable path create-new.
    let output = std::env::args()
        .nth(1)
        .ok_or("usage: session NEW_OUTPUT_DIRECTORY")?;
    std::fs::create_dir(&output)?;
    let output = Path::new(&output);
    let cancel = CancellationToken::new();
    let session = spec()
        .warmup(Warmup::Covariance {
            iterations: 100,
            metric: MetricKind::Diagonal,
            target: 0.8.try_into()?,
            search: None,
        })
        .cancellation(cancel.clone())
        .build()?;
    let partial = session.run(
        factory,
        Execution::Sequential,
        &Observer(|d: &alea_runtime::session::OwnedDraw| {
            if d.index == 31 {
                cancel.cancel();
            }
            Ok(())
        }),
    )?;
    assert_eq!(partial.status, Status::Cancelled);
    partial.chains[0]
        .checkpoint
        .as_ref()
        .ok_or("no frozen checkpoint")?
        .write_new(output.join("chain-0.checkpoint"), 65_536)?;
    let cp = Checkpoint::read(output.join("chain-0.checkpoint"), 65_536)?;
    let resumed = spec().restart(vec![cp]).collect(false).build()?;
    let sink = DurableSink::create_new(output.join("draws.alea"), resumed.provenance(), 1_000_000)?;
    let result = resumed.run(factory, Execution::Sequential, &sink)?;
    let mut reader = DrawReader::new(File::open(output.join("draws.alea"))?, 65_536)?;
    let mut count = 0;
    while reader.next_draw()?.is_some() {
        count += 1;
    }
    assert_eq!(count, 96);
    assert_eq!(result.chains[0].completed, 128);
    // One chain intentionally yields a diagnostic warning; this example tests
    // durable execution/replay, not a multi-chain convergence claim.
    println!(
        "{:?}: {} new durable draws; {} total",
        result.status, count, result.chains[0].completed
    );
    Ok(())
}
