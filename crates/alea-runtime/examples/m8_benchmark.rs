//! Reproducible M8 fixed-HMC/NUTS/ChEES benchmark; refuses to overwrite output.
#[path = "../../alea-mcmc/tests/support/targets.rs"]
mod targets;

use std::{
    fs::OpenOptions,
    io::{BufWriter, Write},
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

use alea_core::target::LogDensityGradient;
use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};
use alea_mcmc::{
    CheesHmc, CheesHmcOptions, HmcOptions, MarkovChain, Nuts,
    adapt::{CheesCriterion, CheesEnsembleAdapter, HmcWarmup, WarmupOptions},
};
use alea_runtime::{
    chees,
    diagnostics::modern,
    ensemble::{Chain, Execution, chain_rng},
    output::ChainId,
};

const WARMUP: usize = 500;
const DRAWS: usize = 2_048;
const CHAINS: usize = 4;

struct Counted {
    target: targets::Target,
    calls: AtomicUsize,
}

impl Counted {
    fn new(target: targets::Target) -> Self {
        Self {
            target,
            calls: AtomicUsize::new(0),
        }
    }
}

impl LogDensityGradient for Counted {
    type Error = std::convert::Infallible;

    fn dimension(&self) -> usize {
        2
    }

    fn logp_grad(&self, position: &[f64], gradient: &mut [f64]) -> Result<f64, Self::Error> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.target.logp_grad(position, gradient)
    }
}

fn initial(chain: usize) -> OwnedBuffer {
    let sign = if chain.is_multiple_of(2) { -1.0 } else { 1.0 };
    OwnedBuffer::from_fn(2, |coordinate| if coordinate == 0 { sign } else { -sign })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("provide a new output CSV")?;
    let mut output = BufWriter::new(OpenOptions::new().write(true).create_new(true).open(path)?);
    writeln!(
        output,
        "target,method,observable,mean,mcse,rhat,bulk_ess,tail_ess,warmup_seconds,total_seconds,gradient_calls,integration_steps,ess_per_second,ess_per_gradient"
    )?;
    for target_kind in [
        targets::Target::Correlated,
        targets::Target::Logistic,
        targets::Target::Banana,
        targets::Target::Funnel,
    ] {
        for method in ["fixed_hmc", "nuts", "chees", "chees_difficult"] {
            let target = Counted::new(target_kind);
            let mut samples: Vec<Vec<Vec<f64>>> = (0..CHAINS)
                .map(|_| (0..5).map(|_| Vec::with_capacity(DRAWS)).collect())
                .collect();
            let mut integration_steps = 0_usize;
            let started = Instant::now();
            let warmup_seconds;
            if method.starts_with("chees") {
                let criterion = if method == "chees_difficult" {
                    CheesCriterion::difficult_direction(&[1.0, 0.0])?
                } else {
                    CheesCriterion::Global
                };
                let adapter =
                    CheesEnsembleAdapter::new(2, 0.1.try_into()?, 1.0, 0.05, 4.0, criterion)?;
                let placeholder = adapter.clone().finish()?;
                let mut chains: Vec<_> = (0..CHAINS)
                    .map(|index| {
                        let id = ChainId(index as u64);
                        Ok::<_, Box<dyn std::error::Error>>(Chain {
                            id,
                            sampler: CheesHmc::new(
                                &target,
                                initial(index),
                                IdentityMetric::new(2),
                                CheesHmcOptions::new(placeholder),
                            )?,
                            rng: chain_rng([23; 32], id),
                            sink: (),
                        })
                    })
                    .collect::<Result<_, _>>()?;
                let (_, report) =
                    chees::warmup(&mut chains, adapter, WARMUP, Execution::Sequential)?;
                integration_steps += report.integration_steps;
                warmup_seconds = started.elapsed().as_secs_f64();
                for (chain_index, chain) in chains.iter_mut().enumerate() {
                    for _ in 0..DRAWS {
                        let transition = chain.sampler.step(&mut chain.rng)?;
                        integration_steps += transition.plan.integration_steps;
                        for (output, value) in samples[chain_index]
                            .iter_mut()
                            .zip(target_kind.observables(chain.sampler.position()))
                        {
                            output.push(value);
                        }
                    }
                }
            } else {
                let mut warmed = Vec::with_capacity(CHAINS);
                for chain_index in 0..CHAINS {
                    let id = ChainId(chain_index as u64);
                    let mut rng = chain_rng([23; 32], id);
                    let (chain, report) = HmcWarmup::new(
                        &target,
                        initial(chain_index),
                        HmcOptions::new(0.1, 10)?,
                        WarmupOptions::new(WARMUP),
                    )?
                    .run(&mut rng)?;
                    integration_steps += report.integration_attempts;
                    warmed.push((chain, rng));
                }
                warmup_seconds = started.elapsed().as_secs_f64();
                for (chain_index, (chain, mut rng)) in warmed.into_iter().enumerate() {
                    if method == "fixed_hmc" {
                        let mut chain = chain;
                        for _ in 0..DRAWS {
                            let transition = chain.step(&mut rng)?;
                            integration_steps += transition.integration_steps;
                            for (output, value) in samples[chain_index]
                                .iter_mut()
                                .zip(target_kind.observables(chain.position()))
                            {
                                output.push(value);
                            }
                        }
                    } else {
                        let mut chain = Nuts::from_hmc(chain, 8.try_into()?)?;
                        for _ in 0..DRAWS {
                            let transition = chain.step(&mut rng)?;
                            integration_steps += transition.integration_steps;
                            for (output, value) in samples[chain_index]
                                .iter_mut()
                                .zip(target_kind.observables(chain.position()))
                            {
                                output.push(value);
                            }
                        }
                    }
                }
            }
            let total_seconds = started.elapsed().as_secs_f64();
            let calls = target.calls.load(Ordering::Relaxed);
            for observable in 0..5 {
                let views: Vec<_> = samples
                    .iter()
                    .map(|chain| chain[observable].as_slice())
                    .collect();
                let summary = modern::summarize(&views)?;
                writeln!(
                    output,
                    "{target_kind:?},{method},{observable},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{warmup_seconds:.9},{total_seconds:.9},{calls},{integration_steps},{:.17e},{:.17e}",
                    summary.mean,
                    summary.mcse_mean,
                    summary.rhat,
                    summary.bulk_ess,
                    summary.tail_ess,
                    summary.bulk_ess / total_seconds,
                    summary.bulk_ess / calls as f64,
                )?;
            }
        }
    }
    output.flush()?;
    Ok(())
}
