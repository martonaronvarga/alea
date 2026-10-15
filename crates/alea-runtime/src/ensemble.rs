//! Deterministic independent streams and bounded, chunked multi-chain execution.
use crate::{
    monitor::{Publisher, Snapshot, Window},
    output::{ChainId, DiagnosticTransition, Draw, Phase, SCHEMA_VERSION},
};
use alea_mcmc::MarkovChain;
use chacha20::ChaCha12Rng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::{num::NonZeroUsize, ops::ControlFlow};
use tokio_util::sync::CancellationToken;

/// ChaCha12 with an explicit 256-bit seed and disjoint 64-bit chain stream.
/// Keep each chain below 2^64 blocks. No scheduler/thread-local RNG is consulted.
pub fn chain_rng(seed: [u8; 32], id: ChainId) -> ChaCha12Rng {
    let mut rng = ChaCha12Rng::from_seed(seed);
    rng.set_stream(id.0);
    rng
}

/// A bound chain, its continuing RNG stream and its own streaming sink.
/// Construct RNG before adaptive warmup, then transfer it here without reseeding.
pub struct Chain<C, S> {
    /// Logical identity; must be unique in an ensemble.
    pub id: ChainId,
    /// Current coherent sampler state.
    pub sampler: C,
    /// Stream continues across warmup and retained sampling.
    pub rng: ChaCha12Rng,
    /// Per-chain sink; invoked in chain order, outside numerical transitions.
    pub sink: S,
}

/// Execution policy; Rayon global configuration is never changed.
#[derive(Clone, Copy)]
pub enum Execution<'a> {
    /// Chains advance sequentially within each reporting chunk.
    Sequential,
    /// Independent chains use the caller's bounded CPU pool.
    Parallel(&'a rayon::ThreadPool),
}

/// Fixed-budget plan. Discarded transitions do not adapt sampler settings.
#[derive(Debug, Clone)]
pub struct RunOptions {
    /// Discarded fixed-geometry burn-in; adaptive warmup is a separate operation.
    pub warmup: usize,
    /// Maximum retained draws per chain.
    pub draws: usize,
    /// Progress/snapshot interval; cancellation is checked before every transition.
    pub chunk: NonZeroUsize,
    /// Maximum contiguous retained draws used in monitoring, at least four.
    pub window: NonZeroUsize,
    /// Explicit coordinate observables; cannot be empty or contain duplicates.
    pub coordinates: Vec<usize>,
}

/// Why a run returned. Errors use [`RunError`] instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Termination {
    /// All requested draws completed.
    Completed,
    /// Cancellation (including optional diagnostic stopping) was observed.
    Cancelled,
    /// At least one streaming sink requested a stop.
    Observer,
}

/// Stable input-order completion counts, including partial cancelled runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainProgress {
    /// Chain identity.
    pub chain: ChainId,
    /// Completed discarded transitions.
    pub warmup: usize,
    /// Completed retained transitions.
    pub draws: usize,
}

/// Final control-plane result; samples remain with each caller-owned sink.
#[derive(Debug, Clone)]
pub struct RunReport {
    /// Explicit termination classification.
    pub termination: Termination,
    /// Input-order per-chain progress, not scheduler completion order.
    pub chains: Vec<ChainProgress>,
}

/// Setup or typed sampler failure. Completed transitions are never rolled back.
#[derive(Debug, thiserror::Error)]
pub enum RunError<E: std::error::Error + 'static> {
    /// Invalid IDs, monitored coordinates, dimensions or memory budget.
    #[error("invalid ensemble configuration")]
    Configuration,
    /// A sampler changed dimension or emitted a nonfinite position.
    #[error("chain {chain:?} violated its position contract")]
    Position {
        /// Failing chain.
        chain: ChainId,
    },
    /// Original sampler error and stable chain identity.
    #[error("chain {chain:?} failed: {source}")]
    Chain {
        /// Failing chain.
        chain: ChainId,
        /// Original error.
        #[source]
        source: E,
    },
}

/// Runs chains with O(chains × window × observables) monitoring memory.
/// Sinks may store all draws, but that is an explicit caller-owned memory cost.
/// Equal seeds/state produce identical per-chain draws under either execution
/// policy for completed fixed-budget runs. Cancellation/monitor stopping timing
/// is scheduler-dependent. Cross-chain sink interleaving is not ordered.
///
/// # Errors
/// Validates all configuration before advancing RNGs. Reports the first error
/// in input chain order after joining the current chunk. Other chains may have
/// completed their chunk. Panics propagate after Rayon joins borrowed work.
pub fn run<C, S>(
    chains: &mut [Chain<C, S>],
    options: &RunOptions,
    execution: Execution<'_>,
    cancel: &CancellationToken,
    publisher: Option<&Publisher>,
) -> Result<RunReport, RunError<C::Error>>
where
    C: MarkovChain + Send,
    C::Error: Send,
    C::Transition: DiagnosticTransition,
    S: for<'a> FnMut(Draw<'a, C::Transition>) -> ControlFlow<()> + Send,
{
    let dimension = chains.first().map_or(0, |c| c.sampler.position().len());
    let row_bytes = options
        .coordinates
        .len()
        .checked_mul(size_of::<f64>())
        .and_then(|n| n.checked_add(size_of::<Option<f64>>()));
    let valid_memory = chains
        .len()
        .checked_mul(options.window.get())
        .and_then(|n| row_bytes.and_then(|bytes| n.checked_mul(bytes)))
        .is_some_and(|n| n <= isize::MAX as usize);
    if chains.is_empty()
        || dimension == 0
        || options.window.get() < 4
        || !valid_memory
        || options.coordinates.is_empty()
        || options.coordinates.iter().any(|&i| i >= dimension)
        || options
            .coordinates
            .iter()
            .enumerate()
            .any(|(i, x)| options.coordinates[..i].contains(x))
        || chains.iter().enumerate().any(|(i, c)| {
            chains[..i].iter().any(|p| p.id == c.id)
                || c.sampler.position().len() != dimension
                || c.rng.get_stream() != c.id.0
        })
    {
        return Err(RunError::Configuration);
    }
    let mut windows: Vec<_> = chains
        .iter()
        .map(|c| Window::new(c.id, options.window.get(), options.coordinates.len()))
        .collect();
    let mut progress: Vec<_> = chains
        .iter()
        .map(|c| ChainProgress {
            chain: c.id,
            warmup: 0,
            draws: 0,
        })
        .collect();
    let mut reason = Termination::Completed;
    'phases: for (phase, budget) in [
        (Phase::Warmup, options.warmup),
        (Phase::Sampling, options.draws),
    ] {
        let mut offset = 0;
        while offset < budget {
            if cancel.is_cancelled() {
                reason = Termination::Cancelled;
                break 'phases;
            }
            let count = options.chunk.get().min(budget - offset);
            let work =
                |((chain, window), progress): (
                    (&mut Chain<C, S>, &mut Window),
                    &mut ChainProgress,
                )| {
                    for _ in 0..count {
                        if cancel.is_cancelled() {
                            return Ok(false);
                        }
                        if chain.sampler.position().len() != dimension {
                            return Err(RunError::Position { chain: chain.id });
                        }
                        let transition = chain.sampler.step(&mut chain.rng).map_err(|source| {
                            RunError::Chain {
                                chain: chain.id,
                                source,
                            }
                        })?;
                        let position = chain.sampler.position();
                        if position.len() != dimension || position.iter().any(|x| !x.is_finite()) {
                            return Err(RunError::Position { chain: chain.id });
                        }
                        let stats = transition.diagnostic_stats();
                        let index = match phase {
                            Phase::Warmup => {
                                let i = progress.warmup;
                                progress.warmup += 1;
                                i
                            }
                            Phase::Sampling => {
                                let i = progress.draws;
                                progress.draws += 1;
                                window.push(position, &options.coordinates, stats);
                                i
                            }
                        };
                        if (chain.sink)(Draw {
                            schema_version: SCHEMA_VERSION,
                            chain: chain.id,
                            phase,
                            index,
                            position,
                            transition: &transition,
                            stats,
                        })
                        .is_break()
                        {
                            return Ok(true);
                        }
                    }
                    Ok(false)
                };
            let results: Vec<_> = match execution {
                Execution::Sequential => chains
                    .iter_mut()
                    .zip(&mut windows)
                    .zip(&mut progress)
                    .map(work)
                    .collect(),
                Execution::Parallel(pool) => pool.install(|| {
                    chains
                        .par_iter_mut()
                        .zip(&mut windows)
                        .zip(&mut progress)
                        .map(work)
                        .collect()
                }),
            };
            if let Some(p) = publisher {
                p.publish(Snapshot::new(
                    &windows,
                    &progress,
                    &options.coordinates,
                    false,
                ));
            }
            for result in results {
                match result {
                    Ok(true) => reason = Termination::Observer,
                    Ok(false) => {}
                    Err(error) => {
                        if let Some(p) = publisher {
                            let mut snapshot =
                                Snapshot::new(&windows, &progress, &options.coordinates, true);
                            snapshot.failed = true;
                            p.publish(snapshot);
                        }
                        return Err(error);
                    }
                }
            }
            if reason == Termination::Observer {
                break 'phases;
            }
            offset += count;
        }
    }
    if reason == Termination::Completed
        && cancel.is_cancelled()
        && progress
            .iter()
            .any(|p| p.warmup != options.warmup || p.draws != options.draws)
    {
        reason = Termination::Cancelled;
    }
    if let Some(p) = publisher {
        p.publish(Snapshot::new(
            &windows,
            &progress,
            &options.coordinates,
            true,
        ));
    }
    Ok(RunReport {
        termination: reason,
        chains: progress,
    })
}
