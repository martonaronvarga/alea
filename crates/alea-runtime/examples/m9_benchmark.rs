//! End-to-end M9 benchmark; refuses to overwrite its CSV output.
#[cfg(not(feature = "faer"))]
fn main() {
    panic!("enable alea-runtime/faer");
}

#[cfg(feature = "faer")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::{
        convert::Infallible,
        fs::OpenOptions,
        io::{BufWriter, Write},
        sync::atomic::{AtomicUsize, Ordering},
        time::Instant,
    };

    use alea_core::{target::LogDensityGradient, transport::FrozenTransportTarget};
    use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};
    use alea_mcmc::{
        Hmc, HmcOptions, Nuts,
        adapt::{
            AffineTransportLearner, FisherHmcWarmup, FisherOptions, HmcWarmup,
            MultiPathfinderOptions, ScoreInitializationOptions, TransportLearningOptions,
            WarmupOptions, learn_affine_transport, multi_pathfinder, score_initialize,
        },
        walnuts::{Walnuts, WalnutsOptions},
    };
    use alea_runtime::diagnostics::modern;
    use rand::{SeedableRng, rngs::SmallRng};

    const DIMENSION: usize = 8;
    const WARMUP: usize = 600;
    const DRAWS: usize = 1_024;
    const DEPTH: usize = 8;
    const SEEDS: [u64; 4] = [41_001, 41_002, 41_003, 41_004];

    struct Target {
        calls: AtomicUsize,
    }
    impl Target {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
            }
        }
        fn scale(index: usize) -> f64 {
            0.5 + index as f64 / (DIMENSION - 1) as f64 * 1.5
        }
        fn direction(index: usize) -> f64 {
            (if index.is_multiple_of(2) { 1.0 } else { -1.0 }) / (DIMENSION as f64).sqrt()
        }
        fn observables(q: &[f64]) -> [f64; 5] {
            let standardized = (0..DIMENSION)
                .map(|i| Self::direction(i) * q[i] / Self::scale(i))
                .sum::<f64>()
                / 10.0;
            let contrast = (q[0] / Self::scale(0) + q[1] / Self::scale(1)) / 2.0_f64.sqrt();
            [
                standardized,
                contrast,
                standardized * standardized,
                contrast * contrast,
                standardized * contrast,
            ]
        }
    }
    impl LogDensityGradient for Target {
        type Error = Infallible;
        fn dimension(&self) -> usize {
            DIMENSION
        }
        fn logp_grad(&self, q: &[f64], gradient: &mut [f64]) -> Result<f64, Infallible> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let projection = (0..DIMENSION)
                .map(|i| Self::direction(i) * q[i] / Self::scale(i))
                .sum::<f64>();
            let coefficient = (0.01 - 1.0) * projection;
            let mut potential = 0.0;
            for i in 0..DIMENSION {
                let standardized = q[i] / Self::scale(i);
                potential += 0.5 * standardized * standardized;
                gradient[i] = -(standardized + coefficient * Self::direction(i)) / Self::scale(i);
            }
            potential += 0.5 * (0.01 - 1.0) * projection * projection;
            Ok(-potential)
        }
    }

    fn initial(chain: usize) -> OwnedBuffer {
        let sign = if chain.is_multiple_of(2) { -1.0 } else { 1.0 };
        OwnedBuffer::from_fn(DIMENSION, |i| sign * (0.3 + 0.02 * i as f64))
    }

    let path = std::env::args().nth(1).ok_or("provide a new output CSV")?;
    let mut output = BufWriter::new(OpenOptions::new().write(true).create_new(true).open(path)?);
    writeln!(
        output,
        "target,method,observable,mean,mcse,rhat,bulk_ess,tail_ess,warmup_seconds,total_seconds,target_calls,trajectory_steps,ess_per_second,ess_per_call"
    )?;
    for method in ["fisher_nuts", "fisher_walnuts", "transport_nuts"] {
        let mut samples: Vec<Vec<Vec<f64>>> = (0..SEEDS.len())
            .map(|_| (0..5).map(|_| Vec::with_capacity(DRAWS)).collect())
            .collect();
        let mut total_time = 0.0;
        let mut warmup_time = 0.0;
        let mut total_calls = 0;
        let mut trajectory_steps = 0;
        for (chain_index, seed) in SEEDS.into_iter().enumerate() {
            let target = Target::new();
            let mut rng = SmallRng::seed_from_u64(seed);
            let started = Instant::now();
            if method == "transport_nuts" {
                let chain = Hmc::new(
                    &target,
                    initial(chain_index),
                    IdentityMetric::new(DIMENSION),
                    HmcOptions::new(0.04, 8)?,
                )?;
                let mut learning_chain = chain;
                let learner = AffineTransportLearner::new(
                    DIMENSION,
                    TransportLearningOptions::new(400, 4)?
                        .with_power_iterations(20)?
                        .with_ridge(1e-5)?,
                )?;
                let learned = learn_affine_transport(&mut learning_chain, learner, 399, &mut rng)?;
                let transformed = FrozenTransportTarget::new(&target, learned.map)?;
                let (hmc, report) = HmcWarmup::new(
                    &transformed,
                    learned.latent_position,
                    HmcOptions::new(0.1, 8)?,
                    WarmupOptions::new(200),
                )?
                .run(&mut rng)?;
                trajectory_steps += report.integration_attempts + 399 * 8;
                warmup_time += started.elapsed().as_secs_f64();
                let mut sampler = Nuts::from_hmc(hmc, DEPTH.try_into()?)?;
                let mut target_position = vec![0.0; DIMENSION];
                for _ in 0..DRAWS {
                    let transition = sampler.step(&mut rng)?;
                    trajectory_steps += transition.integration_steps;
                    transformed.to_target(sampler.point().position(), &mut target_position)?;
                    for (series, value) in samples[chain_index]
                        .iter_mut()
                        .zip(Target::observables(&target_position))
                    {
                        series.push(value);
                    }
                }
            } else {
                let fisher = FisherOptions::new(WARMUP).with_max_rank(4);
                let (hmc, report) = FisherHmcWarmup::new(
                    &target,
                    initial(chain_index),
                    HmcOptions::new(0.1, 5)?,
                    fisher,
                )?
                .run(&mut rng)?;
                trajectory_steps += report.integration_attempts;
                warmup_time += started.elapsed().as_secs_f64();
                if method == "fisher_nuts" {
                    let mut sampler = Nuts::from_hmc(hmc, DEPTH.try_into()?)?;
                    for _ in 0..DRAWS {
                        let transition = sampler.step(&mut rng)?;
                        trajectory_steps += transition.integration_steps;
                        for (series, value) in samples[chain_index]
                            .iter_mut()
                            .zip(Target::observables(sampler.point().position()))
                        {
                            series.push(value);
                        }
                    }
                } else {
                    let macro_step = 4.0 * hmc.options().step_size().value();
                    let options = WalnutsOptions::new(macro_step, DEPTH)?
                        .with_local_refinement(10, 1, 0.21)?;
                    let mut sampler = Walnuts::from_hmc(hmc, options)?;
                    for _ in 0..DRAWS {
                        let transition = sampler.step(&mut rng)?;
                        trajectory_steps += transition.microsteps;
                        for (series, value) in samples[chain_index]
                            .iter_mut()
                            .zip(Target::observables(sampler.point().position()))
                        {
                            series.push(value);
                        }
                    }
                }
            }
            total_time += started.elapsed().as_secs_f64();
            total_calls += target.calls.load(Ordering::Relaxed);
        }
        for observable in 0..5 {
            let views = samples
                .iter()
                .map(|chain| chain[observable].as_slice())
                .collect::<Vec<_>>();
            let summary = modern::summarize(&views)?;
            writeln!(
                output,
                "rank_one_gaussian,{method},{observable},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{warmup_time:.9},{total_time:.9},{total_calls},{trajectory_steps},{:.17e},{:.17e}",
                summary.mean,
                summary.mcse_mean,
                summary.rhat,
                summary.bulk_ess,
                summary.tail_ess,
                summary.bulk_ess / total_time,
                summary.bulk_ess / total_calls as f64,
            )?;
        }
    }

    // Separate initialization-cost experiment: same target, starts and work
    // accounting; Pathfinder remains optional and is not hidden in sampler time.
    let initials = (0..4)
        .flat_map(|chain| initial(chain).iter().copied().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let target = Target::new();
    let mut rng = SmallRng::seed_from_u64(51_001);
    let started = Instant::now();
    let pathfinder = multi_pathfinder(
        &target,
        &initials,
        MultiPathfinderOptions::new(4, 12)?
            .with_elbo_draws(8)?
            .with_draws_per_path(8)?,
        &mut rng,
    )?;
    let pathfinder_seconds = started.elapsed().as_secs_f64();
    let target = Target::new();
    let started = Instant::now();
    let score = score_initialize(
        &target,
        &initials[..DIMENSION],
        ScoreInitializationOptions::new(12, 0.1, 20)?,
    )?;
    let score_seconds = started.elapsed().as_secs_f64();
    eprintln!(
        "initialization,method,seconds,target_calls,mahalanobis_sq\ninitialization,pathfinder,{pathfinder_seconds:.9},{},{:.17e}\ninitialization,score,{score_seconds:.9},{},{:.17e}",
        pathfinder.report.target_evaluations,
        -2.0 * Target::new().logp_grad(&pathfinder.position, &mut [0.0; DIMENSION])?,
        score.report.target_evaluations,
        -2.0 * Target::new().logp_grad(&score.position, &mut [0.0; DIMENSION])?,
    );
    output.flush()?;
    Ok(())
}
