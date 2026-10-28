//! Deterministic lockstep orchestration for shared ChEES warmup.

use alea_math::metric::IdentityMetric;
use alea_mcmc::{
    LogDensityGradient,
    adapt::{CheesConfigError, CheesEnsembleAdapter, CheesUpdate, FrozenCheesParameters},
    chees::CheesHmc,
    hmc::HmcError,
    integrator::SymmetricSplitting,
};
use rayon::prelude::*;

use crate::{
    ensemble::{Chain, Execution},
    output::ChainId,
};

/// Completed shared warmup work. No retained draws are produced here.
#[derive(Debug, Clone, Copy)]
pub struct CheesWarmupReport {
    /// Completed lockstep ensemble iterations.
    pub iterations: usize,
    /// Total complete integration macrosteps across all chains.
    pub integration_steps: usize,
    /// Last ensemble update, absent for zero warmup iterations.
    pub last_update: Option<CheesUpdate>,
}

/// Invalid scheduling, adaptation, or a stable-identity chain failure.
#[derive(Debug, thiserror::Error)]
pub enum CheesWarmupError<E: std::error::Error + 'static> {
    /// The ensemble is empty, IDs repeat, or RNG streams do not match IDs.
    #[error("invalid ChEES ensemble configuration")]
    Configuration,
    /// Shared-controller input or arithmetic failed.
    #[error(transparent)]
    Adaptation(#[from] CheesConfigError),
    /// A chain failed; all earlier/later chains in that lockstep may already have run.
    #[error("chain {chain:?} failed during shared ChEES warmup: {source}")]
    Chain {
        /// Stable logical chain identity.
        chain: ChainId,
        /// Concrete target/metric failure.
        #[source]
        source: HmcError<E>,
    },
    /// Total integration-step accounting overflowed.
    #[error("ChEES warmup integration-step counter overflow")]
    CounterOverflow,
}

/// Runs ensemble-adapted trajectories in deterministic lockstep.
///
/// One shared plan is applied to every chain, results are collected in input
/// order, and only then is the shared controller advanced. Equal starting state
/// and per-chain streams therefore produce identical parameters and positions
/// under sequential and Rayon execution. A failed lockstep does not update the
/// adapter, although other chains may already have completed that iteration.
/// Successful completion consumes the adapter and returns fixed parameters.
///
/// # Errors
/// Validates identities/streams before consuming schedules or RNG. Propagates the
/// first chain error in input order, adaptation errors, and counter overflow.
pub fn warmup<T, I, S>(
    chains: &mut [Chain<CheesHmc<'_, T, IdentityMetric, I>, S>],
    mut adapter: CheesEnsembleAdapter,
    iterations: usize,
    execution: Execution<'_>,
) -> Result<(FrozenCheesParameters, CheesWarmupReport), CheesWarmupError<T::Error>>
where
    T: LogDensityGradient + Sync + ?Sized,
    I: SymmetricSplitting + Send,
    T::Error: Send,
    S: Send,
{
    if chains.is_empty()
        || chains.iter().enumerate().any(|(index, chain)| {
            chain.rng.get_stream() != chain.id.0
                || chains[..index]
                    .iter()
                    .any(|previous| previous.id == chain.id)
        })
    {
        return Err(CheesWarmupError::Configuration);
    }
    let mut completed_steps = 0_usize;
    let mut last_update = None;
    for _ in 0..iterations {
        let plan = adapter.next_plan()?;
        let execute = |chain: &mut Chain<CheesHmc<'_, T, IdentityMetric, I>, S>| {
            chain
                .sampler
                .step_with_plan(plan, &mut chain.rng)
                .map_err(|source| CheesWarmupError::Chain {
                    chain: chain.id,
                    source,
                })
        };
        let transitions: Vec<_> = match execution {
            Execution::Sequential => chains.iter_mut().map(execute).collect(),
            Execution::Parallel(pool) => {
                pool.install(|| chains.par_iter_mut().map(execute).collect())
            }
        };
        let mut successful = Vec::with_capacity(chains.len());
        for transition in transitions {
            successful.push(transition?);
        }
        completed_steps = completed_steps
            .checked_add(
                plan.integration_steps
                    .checked_mul(chains.len())
                    .ok_or(CheesWarmupError::CounterOverflow)?,
            )
            .ok_or(CheesWarmupError::CounterOverflow)?;
        let observations: Vec<_> = chains
            .iter()
            .zip(&successful)
            .map(|(chain, transition)| chain.sampler.observation(transition))
            .collect();
        last_update = Some(adapter.update(&observations)?);
    }
    let frozen = adapter.finish()?;
    for chain in chains.iter_mut() {
        chain.sampler.install_frozen(frozen);
    }
    Ok((
        frozen,
        CheesWarmupReport {
            iterations,
            integration_steps: completed_steps,
            last_update,
        },
    ))
}
