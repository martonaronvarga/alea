//! Fixed M7 Fisher-NUTS versus WALNUTS-D/R2P benchmark; refuses to overwrite output.
#[cfg(feature = "faer")]
#[path = "support/walnuts_targets.rs"]
mod walnuts_targets;

#[cfg(not(feature = "faer"))]
fn main() {
    panic!("enable alea-runtime/faer");
}

#[cfg(feature = "faer")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use alea_math::buffer::OwnedBuffer;
    use alea_mcmc::{
        HmcOptions, Nuts, Walnuts, WalnutsMicrostepDistribution, WalnutsOptions,
        adapt::{FisherHmcWarmup, FisherOptions},
    };
    use alea_runtime::diagnostics::modern;
    use rand::{SeedableRng, rngs::SmallRng};
    use std::{
        fs::OpenOptions,
        io::{BufWriter, Write},
        time::Instant,
    };
    use walnuts_targets::{Kind, Target};

    const WARMUP: usize = 1_000;
    const DRAWS: usize = 4_096;
    const DEPTH: usize = 8;
    const SEEDS: [u64; 4] = [9_001, 9_002, 9_003, 9_004];

    #[derive(Default)]
    struct Work {
        macrosteps: usize,
        microsteps: usize,
        refinement_stops: usize,
    }

    impl Work {
        fn add_nuts(&mut self, steps: usize) -> Result<(), &'static str> {
            self.macrosteps = self
                .macrosteps
                .checked_add(steps)
                .ok_or("macrostep counter overflow")?;
            self.microsteps = self
                .microsteps
                .checked_add(steps)
                .ok_or("microstep counter overflow")?;
            Ok(())
        }

        fn add_walnuts(
            &mut self,
            transition: alea_mcmc::WalnutsTransition,
        ) -> Result<(), &'static str> {
            self.macrosteps = self
                .macrosteps
                .checked_add(transition.macro_steps)
                .ok_or("macrostep counter overflow")?;
            self.microsteps = self
                .microsteps
                .checked_add(transition.microsteps)
                .ok_or("microstep counter overflow")?;
            self.refinement_stops = self
                .refinement_stops
                .checked_add(usize::from(matches!(
                    transition.termination,
                    alea_mcmc::WalnutsTermination::RefinementLimit
                        | alea_mcmc::WalnutsTermination::ReverseSelection
                )))
                .ok_or("refinement-stop counter overflow")?;
            Ok(())
        }
    }

    let output = std::env::args().nth(1).ok_or("provide a new output CSV")?;
    let mut output = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)?,
    );
    writeln!(
        output,
        "target,method,observable,mean,mcse,rhat,bulk_ess,tail_ess,total_seconds,total_calls,ess_per_second,ess_per_call,macrosteps,microsteps,refinement_stops"
    )?;

    for kind in [Kind::Funnel, Kind::StochasticVolatility] {
        for method in ["fisher_nuts", "fisher_walnuts", "fisher_walnuts_r2p"] {
            let mut chains: Vec<Vec<Vec<f64>>> = (0..SEEDS.len())
                .map(|_| (0..5).map(|_| Vec::with_capacity(DRAWS)).collect())
                .collect();
            let mut total_seconds = 0.0;
            let mut total_calls = 0_usize;
            let mut work = Work::default();
            for (chain_index, seed) in SEEDS.into_iter().enumerate() {
                let target = Target::new(kind);
                let dimension = kind.dimension();
                let sign = if chain_index % 2 == 0 { -1.0 } else { 1.0 };
                let initial = OwnedBuffer::from_fn(dimension, |index| {
                    if matches!(kind, Kind::Funnel) {
                        if index == 0 { sign } else { -sign }
                    } else {
                        0.1 * sign
                    }
                });
                let mut rng = SmallRng::seed_from_u64(seed);
                let start = Instant::now();
                let fisher = FisherOptions::new(WARMUP).with_max_rank(dimension.min(8));
                let (hmc, _) =
                    FisherHmcWarmup::new(&target, initial, HmcOptions::new(0.1, 5)?, fisher)?
                        .run(&mut rng)?;
                if method == "fisher_nuts" {
                    let mut sampler = Nuts::from_hmc(hmc, DEPTH.try_into()?)?;
                    for _ in 0..DRAWS {
                        let info = sampler.step(&mut rng)?;
                        work.add_nuts(info.integration_steps)?;
                        for (output, value) in chains[chain_index]
                            .iter_mut()
                            .zip(kind.observables(sampler.point().position()))
                        {
                            output.push(value);
                        }
                    }
                } else {
                    let macro_step = 4.0 * hmc.options().step_size().value();
                    let mut options = WalnutsOptions::new(macro_step, DEPTH)?
                        .with_local_refinement(10, 1, 0.21)?;
                    if method == "fisher_walnuts_r2p" {
                        options = options
                            .with_microstep_distribution(
                                WalnutsMicrostepDistribution::RandomizedTwoPoint,
                            )?
                            .with_macro_step_jitter(0.2)?;
                    }
                    let mut sampler = Walnuts::from_hmc(hmc, options)?;
                    for _ in 0..DRAWS {
                        let info = sampler.step(&mut rng)?;
                        work.add_walnuts(info)?;
                        for (output, value) in chains[chain_index]
                            .iter_mut()
                            .zip(kind.observables(sampler.point().position()))
                        {
                            output.push(value);
                        }
                    }
                }
                total_seconds += start.elapsed().as_secs_f64();
                total_calls = total_calls
                    .checked_add(target.calls())
                    .ok_or("target-call counter overflow")?;
            }
            for observable in 0..5 {
                let views: Vec<_> = chains
                    .iter()
                    .map(|chain| chain[observable].as_slice())
                    .collect();
                let summary = modern::summarize(&views)?;
                writeln!(
                    output,
                    "{},{},{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.9},{},{:.17e},{:.17e},{},{},{}",
                    kind.name(),
                    method,
                    observable,
                    summary.mean,
                    summary.mcse_mean,
                    summary.rhat,
                    summary.bulk_ess,
                    summary.tail_ess,
                    total_seconds,
                    total_calls,
                    summary.bulk_ess / total_seconds,
                    summary.bulk_ess / total_calls as f64,
                    work.macrosteps,
                    work.microsteps,
                    work.refinement_stops,
                )?;
            }
        }
    }
    Ok(())
}
