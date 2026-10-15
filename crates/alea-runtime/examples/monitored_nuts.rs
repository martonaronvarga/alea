//! CPU chains on Rayon; asynchronous latest-only diagnostics on Tokio.
use alea_distributions::Gaussian;
use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};
use alea_mcmc::{Nuts, NutsOptions, NutsTransition};
use alea_runtime::{
    ensemble::{self, Chain, Execution, RunOptions, chain_rng},
    monitor::Monitor,
    output::{ChainId, Draw},
};
use std::{num::NonZeroUsize, ops::ControlFlow};
use tokio_util::sync::CancellationToken;

type Error = Box<dyn std::error::Error + Send + Sync>;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Error> {
    let cancel = CancellationToken::new();
    // None: observe only. Enabling stopping requires an explicit StopPolicy.
    let monitor = Monitor::start(None, cancel.clone())?;
    let publisher = monitor.publisher();
    let mut subscription = monitor.subscribe();
    let display = tokio::spawn(async move {
        while let Some(report) = subscription.changed().await {
            println!(
                "finished={} ranges={:?} diagnostics={:?}",
                report.finished,
                report.chains.iter().map(|c| &c.range).collect::<Vec<_>>(),
                report.diagnostics
            );
        }
    });
    // Never execute CPU sampling directly on a Tokio executor thread.
    let worker = tokio::task::spawn_blocking(move || -> Result<_, Error> {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
        let target = Gaussian::new(4);
        let mut chains = Vec::new();
        for i in 0..4 {
            let id = ChainId(i);
            chains.push(Chain {
                id,
                rng: chain_rng([42; 32], id),
                sampler: Nuts::new(
                    &target,
                    OwnedBuffer::new(4),
                    IdentityMetric::new(4),
                    NutsOptions::new(0.2, 7)?,
                )?,
                sink: |_: Draw<'_, NutsTransition>| ControlFlow::Continue(()),
            });
        }
        let options = RunOptions {
            warmup: 256,
            draws: 2048,
            chunk: NonZeroUsize::new(128).expect("positive constant"),
            window: NonZeroUsize::new(1024).expect("positive constant"),
            coordinates: vec![0, 1, 2, 3],
        };
        Ok(ensemble::run(
            &mut chains,
            &options,
            Execution::Parallel(&pool),
            &cancel,
            Some(&publisher),
        )?)
    });
    let result = worker.await;
    // Drain the final snapshot and join even if sampling returned an error.
    tokio::task::spawn_blocking(move || monitor.finish())
        .await?
        .map_err(|_| std::io::Error::other("diagnostic worker panicked"))?;
    display.await?;
    println!("{:?}", result??);
    Ok(())
}
