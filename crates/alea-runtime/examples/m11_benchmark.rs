//! M11 same-process mechanism comparison; refuses to overwrite its CSV output.

use std::{
    fs::OpenOptions,
    io::{BufWriter, Write},
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

use alea_core::target::LogDensityGradient;
use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};
use alea_mcmc::nuts::NutsTermination;
use alea_mcmc::{
    CheesHmc, CheesHmcOptions, HmcOptions, LogisticFisherMetric, MarkovChain, Nuts, Rmhmc,
    RmhmcCosts, RmhmcOptions,
    adapt::{
        CheesCriterion, CheesEnsembleAdapter, HmcWarmup, ManifoldBayesTuner, ManifoldTunerOptions,
        RmhmcManifoldWarmup, WarmupOptions,
    },
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

struct Logistic {
    calls: AtomicUsize,
}

impl Logistic {
    fn new() -> Self {
        Self {
            calls: AtomicUsize::new(0),
        }
    }

    fn observables(q: &[f64]) -> [f64; 5] {
        [q[0], q[1], q[0] * q[0], q[1] * q[1], q[0] * q[1]]
    }
}

impl LogDensityGradient for Logistic {
    type Error = std::convert::Infallible;

    fn dimension(&self) -> usize {
        2
    }

    fn logp_grad(&self, q: &[f64], gradient: &mut [f64]) -> Result<f64, Self::Error> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let mut logp = -0.5 * (q[0] * q[0] + q[1] * q[1]);
        gradient[0] = -q[0];
        gradient[1] = -q[1];
        for (x, y) in [(-1.5, 0.0), (-0.2, 1.0), (0.7, 0.0), (2.0, 1.0)] {
            let eta = q[0] + q[1] * x;
            let exponential = (-eta.abs()).exp();
            let probability = if eta >= 0.0 {
                1.0 / (1.0 + exponential)
            } else {
                exponential / (1.0 + exponential)
            };
            logp += y * eta - eta.max(0.0) - exponential.ln_1p();
            gradient[0] += y - probability;
            gradient[1] += x * (y - probability);
        }
        Ok(logp)
    }
}

#[derive(Default)]
struct ResultRow {
    samples: Vec<Vec<Vec<f64>>>,
    warmup_seconds: f64,
    total_seconds: f64,
    calls: usize,
    costs: RmhmcCosts,
    cold_factorizations: usize,
    failures: usize,
    selected_candidate: Option<usize>,
}

fn initial(chain: usize) -> OwnedBuffer {
    let sign = if chain.is_multiple_of(2) { -1.0 } else { 1.0 };
    OwnedBuffer::from_fn(2, |coordinate| if coordinate == 0 { sign } else { -sign })
}

fn empty_samples() -> Vec<Vec<Vec<f64>>> {
    (0..CHAINS)
        .map(|_| (0..5).map(|_| Vec::with_capacity(DRAWS)).collect())
        .collect()
}

fn push_sample(samples: &mut [Vec<Vec<f64>>], chain: usize, position: &[f64]) {
    for (output, value) in samples[chain]
        .iter_mut()
        .zip(Logistic::observables(position))
    {
        output.push(value);
    }
}

fn add_costs(total: &mut RmhmcCosts, next: RmhmcCosts) {
    total.target_evaluations += next.target_evaluations;
    total.metric_derivative_evaluations += next.metric_derivative_evaluations;
    total.factorizations += next.factorizations;
    total.triangular_solves += next.triangular_solves;
    total.inverse_actions += next.inverse_actions;
    total.dynamics_steps += next.dynamics_steps;
    total.momentum_iterations += next.momentum_iterations;
    total.position_iterations += next.position_iterations;
}

fn fisher_metric() -> Result<LogisticFisherMetric, Box<dyn std::error::Error>> {
    let design = [1.0, -1.5, 1.0, -0.2, 1.0, 0.7, 1.0, 2.0];
    Ok(LogisticFisherMetric::new(
        4,
        2,
        OwnedBuffer::from_fn(design.len(), |index| design[index]),
        OwnedBuffer::from_fn(2, |_| 1.0),
    )?)
}

fn run_rmhmc(adaptive: bool) -> Result<ResultRow, Box<dyn std::error::Error>> {
    let target = Logistic::new();
    let metric = fisher_metric()?;
    let candidates = [
        RmhmcOptions::new(0.08, 6, 12, 1.0e-9)?,
        RmhmcOptions::new(0.12, 8, 12, 1.0e-9)?,
        RmhmcOptions::new(0.16, 10, 12, 1.0e-9)?,
        RmhmcOptions::new(0.22, 10, 12, 1.0e-9)?,
    ];
    let baseline = candidates[1];
    let started = Instant::now();
    let mut rngs: Vec<_> = (0..CHAINS)
        .map(|index| chain_rng([31; 32], ChainId(index as u64)))
        .collect();
    let mut costs = RmhmcCosts::default();
    let cold_factorizations = CHAINS;
    let mut failures = 0;
    let mut selected_candidate = None;
    let mut chains = if adaptive {
        let mut tuner = ManifoldBayesTuner::new(&candidates, ManifoldTunerOptions::default())?;
        let mut warmups: Vec<_> = (0..CHAINS)
            .map(|chain_index| {
                RmhmcManifoldWarmup::new(&target, &metric, initial(chain_index), candidates[0])
            })
            .collect::<Result<_, _>>()?;
        for _ in 0..WARMUP {
            let candidate = tuner.suggest()?;
            for chain_index in 0..CHAINS {
                let before = warmups[chain_index].position().to_vec();
                let transition =
                    warmups[chain_index].step(candidates[candidate], &mut rngs[chain_index])?;
                failures += usize::from(transition.termination.is_some());
                add_costs(&mut costs, transition.costs);
                tuner.observe(
                    candidate,
                    &before,
                    warmups[chain_index].position(),
                    candidates[candidate].integration_steps().get(),
                )?;
            }
        }
        let frozen = tuner.finish()?;
        selected_candidate = Some(frozen.candidate_index());
        warmups
            .into_iter()
            .map(|warmup| warmup.finish(frozen))
            .collect()
    } else {
        let mut chains: Vec<_> = (0..CHAINS)
            .map(|chain_index| Rmhmc::new(&target, &metric, initial(chain_index), baseline))
            .collect::<Result<_, _>>()?;
        for _ in 0..WARMUP {
            for chain_index in 0..CHAINS {
                let transition = chains[chain_index].step(&mut rngs[chain_index])?;
                failures += usize::from(transition.termination.is_some());
                add_costs(&mut costs, transition.costs);
            }
        }
        chains
    };
    let warmup_seconds = started.elapsed().as_secs_f64();
    let mut samples = empty_samples();
    for (chain_index, chain) in chains.iter_mut().enumerate() {
        for _ in 0..DRAWS {
            let transition = chain.step(&mut rngs[chain_index])?;
            failures += usize::from(transition.termination.is_some());
            add_costs(&mut costs, transition.costs);
            push_sample(&mut samples, chain_index, chain.position());
        }
    }
    Ok(ResultRow {
        samples,
        warmup_seconds,
        total_seconds: started.elapsed().as_secs_f64(),
        calls: target.calls.load(Ordering::Relaxed),
        costs,
        cold_factorizations,
        failures,
        selected_candidate,
    })
}

fn run_nuts() -> Result<ResultRow, Box<dyn std::error::Error>> {
    let target = Logistic::new();
    let started = Instant::now();
    let mut warmed = Vec::with_capacity(CHAINS);
    for chain_index in 0..CHAINS {
        let id = ChainId(chain_index as u64);
        let mut rng = chain_rng([31; 32], id);
        let (chain, _) = HmcWarmup::new(
            &target,
            initial(chain_index),
            HmcOptions::new(0.1, 10)?,
            WarmupOptions::new(WARMUP),
        )?
        .run(&mut rng)?;
        warmed.push((Nuts::from_hmc(chain, 8.try_into()?)?, rng));
    }
    let warmup_seconds = started.elapsed().as_secs_f64();
    let mut samples = empty_samples();
    let mut dynamics_steps = 0;
    let mut failures = 0;
    for (chain_index, (mut chain, mut rng)) in warmed.into_iter().enumerate() {
        for _ in 0..DRAWS {
            let transition = chain.step(&mut rng)?;
            dynamics_steps += transition.integration_steps;
            failures += usize::from(matches!(
                transition.termination,
                NutsTermination::Divergence(_)
            ));
            push_sample(&mut samples, chain_index, chain.position());
        }
    }
    Ok(ResultRow {
        samples,
        warmup_seconds,
        total_seconds: started.elapsed().as_secs_f64(),
        calls: target.calls.load(Ordering::Relaxed),
        costs: RmhmcCosts {
            dynamics_steps,
            ..RmhmcCosts::default()
        },
        failures,
        ..ResultRow::default()
    })
}

fn run_chees() -> Result<ResultRow, Box<dyn std::error::Error>> {
    let target = Logistic::new();
    let started = Instant::now();
    let adapter =
        CheesEnsembleAdapter::new(2, 0.1.try_into()?, 1.0, 0.05, 4.0, CheesCriterion::Global)?;
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
                rng: chain_rng([31; 32], id),
                sink: (),
            })
        })
        .collect::<Result<_, _>>()?;
    let (_, report) = chees::warmup(&mut chains, adapter, WARMUP, Execution::Sequential)?;
    let warmup_seconds = started.elapsed().as_secs_f64();
    let mut samples = empty_samples();
    let mut dynamics_steps = report.integration_steps;
    let mut failures = 0;
    for (chain_index, chain) in chains.iter_mut().enumerate() {
        for _ in 0..DRAWS {
            let transition = chain.sampler.step(&mut chain.rng)?;
            dynamics_steps += transition.plan.integration_steps;
            failures += usize::from(transition.divergence.is_some());
            push_sample(&mut samples, chain_index, chain.sampler.position());
        }
    }
    Ok(ResultRow {
        samples,
        warmup_seconds,
        total_seconds: started.elapsed().as_secs_f64(),
        calls: target.calls.load(Ordering::Relaxed),
        costs: RmhmcCosts {
            dynamics_steps,
            ..RmhmcCosts::default()
        },
        failures,
        ..ResultRow::default()
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("provide a new output CSV")?;
    let mut output = BufWriter::new(OpenOptions::new().write(true).create_new(true).open(path)?);
    writeln!(
        output,
        "method,observable,mean,mcse,rhat,bulk_ess,tail_ess,warmup_seconds,total_seconds,target_calls,metric_derivatives,factorizations,cold_factorizations,triangular_solves,inverse_actions,dynamics_steps,failures,selected_candidate,ess_per_second,ess_per_target_call"
    )?;
    for (method, result) in [
        ("nuts", run_nuts()?),
        ("chees", run_chees()?),
        ("base_rmhmc", run_rmhmc(false)?),
        ("adaptive_rmhmc", run_rmhmc(true)?),
    ] {
        for observable in 0..5 {
            let views: Vec<_> = result
                .samples
                .iter()
                .map(|chain| chain[observable].as_slice())
                .collect();
            let summary = modern::summarize(&views)?;
            writeln!(
                output,
                "{method},{observable},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.9},{:.9},{},{},{},{},{},{},{},{},{},{:.17e},{:.17e}",
                summary.mean,
                summary.mcse_mean,
                summary.rhat,
                summary.bulk_ess,
                summary.tail_ess,
                result.warmup_seconds,
                result.total_seconds,
                result.calls,
                result.costs.metric_derivative_evaluations,
                result.costs.factorizations,
                result.cold_factorizations,
                result.costs.triangular_solves,
                result.costs.inverse_actions,
                result.costs.dynamics_steps,
                result.failures,
                result
                    .selected_candidate
                    .map_or_else(|| "none".to_owned(), |index| index.to_string()),
                summary.bulk_ess / result.total_seconds,
                summary.bulk_ess / result.calls as f64,
            )?;
        }
    }
    output.flush()?;
    Ok(())
}
