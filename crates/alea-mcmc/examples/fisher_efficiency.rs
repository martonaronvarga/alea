//! Execute tools/fisher-reference/EFFICIENCY.md; refuses to overwrite an artifact.
#[cfg(not(feature = "faer"))]
fn main() {
    panic!("enable alea-mcmc/faer");
}

#[cfg(feature = "faer")]
#[path = "../tests/support/targets.rs"]
mod targets;

#[cfg(feature = "faer")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use alea_core::target::LogDensityGradient;
    use alea_math::{buffer::OwnedBuffer, metric::EuclideanMetric};
    use alea_mcmc::{
        Hmc, HmcOptions,
        adapt::{
            DiminishingSchedule, FisherHmcWarmup, FisherOptions, HmcWarmup, StepAdaptation,
            WarmupOptions,
        },
    };
    use rand::{SeedableRng, rngs::SmallRng};
    use std::{
        cell::Cell,
        convert::Infallible,
        fs::OpenOptions,
        io::{BufWriter, Write},
        time::Instant,
    };
    struct Counted(targets::Target, Cell<usize>);
    impl LogDensityGradient for Counted {
        type Error = Infallible;
        fn dimension(&self) -> usize {
            2
        }
        fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Infallible> {
            self.1.set(self.1.get() + 1);
            self.0.logp_grad(q, g)
        }
    }
    fn collect<M: EuclideanMetric>(
        chain: &mut Hmc<'_, Counted, M>,
        rng: &mut SmallRng,
        rows: &mut Vec<[f64; 5]>,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let mut divergences = 0;
        for _ in 0..8192 {
            let info = chain.step(rng)?;
            divergences += usize::from(info.divergence.is_some());
            rows.push(
                chain
                    .point()
                    .target()
                    .0
                    .observables(chain.point().position()),
            );
        }
        Ok(divergences)
    }
    let path = std::env::args()
        .nth(1)
        .ok_or("provide a new output CSV path")?;
    let mut out = BufWriter::new(OpenOptions::new().write(true).create_new(true).open(path)?);
    writeln!(
        out,
        "target,method,seed,observable,mean,variance,ess,mcse,warmup_calls,total_calls,total_seconds,warmup_divergences,retained_divergences,rank,step"
    )?;
    for target in [
        targets::Target::Correlated,
        targets::Target::Logistic,
        targets::Target::Banana,
        targets::Target::Funnel,
    ] {
        for (s, seed) in [3101, 3102, 3103, 3104, 4101, 4102, 4103, 4104]
            .into_iter()
            .enumerate()
        {
            for order in 0..8 {
                let method = (order + s) % 8;
                let name = [
                    "covariance",
                    "fisher_diagonal",
                    "rank1_w20_h40",
                    "rank2_w20_h40",
                    "rank2_w80_h160",
                    "weighted_dual",
                    "weighted_rm",
                    "weighted_adam",
                ][method];
                let counted = Counted(target, Cell::new(0));
                let mut rng = SmallRng::seed_from_u64(seed);
                let mut rows = Vec::with_capacity(8192);
                let start = Instant::now();
                let sign = if s % 2 == 0 { -1.0 } else { 1.0 };
                let q = OwnedBuffer::from_fn(2, |i| if i == 0 { sign } else { -sign });
                let hmc = HmcOptions::new(0.1, 5)?;
                let (report, warmup_calls, divergences, rank) = if method == 0 {
                    let (mut chain, report) =
                        HmcWarmup::new(&counted, q, hmc, WarmupOptions::new(1000))?
                            .run(&mut rng)?;
                    let calls = counted.1.get();
                    let divergences = collect(&mut chain, &mut rng, &mut rows)?;
                    (report, calls, divergences, 0)
                } else {
                    let mut options = FisherOptions::new(1000);
                    if (2..=4).contains(&method) {
                        options = options.with_max_rank(if method == 2 { 1 } else { 2 });
                        if method < 4 {
                            options = options.with_window_budget(20, 40)?;
                        }
                    }
                    if method >= 5 {
                        options =
                            options.with_weighted_moments(DiminishingSchedule::new(1.0, 0.75)?);
                    }
                    let decay = DiminishingSchedule::new(1.0, 0.6)?;
                    if method == 6 {
                        options = options
                            .with_step_adaptation(StepAdaptation::robbins_monro(1.0, decay)?);
                    }
                    if method == 7 {
                        options = options.with_step_adaptation(StepAdaptation::adam(
                            0.3, decay, 0.9, 0.999, 1e-8,
                        )?);
                    }
                    let (mut chain, report) =
                        FisherHmcWarmup::new(&counted, q, hmc, options)?.run(&mut rng)?;
                    let calls = counted.1.get();
                    let rank = chain.metric().rank();
                    let divergences = collect(&mut chain, &mut rng, &mut rows)?;
                    (report, calls, divergences, rank)
                };
                let elapsed = start.elapsed().as_secs_f64();
                for j in 0..5 {
                    let mean = rows.iter().map(|x| x[j]).sum::<f64>() / 8192.0;
                    let variance = rows.iter().map(|x| (x[j] - mean).powi(2)).sum::<f64>() / 8191.0;
                    let mut variance_mean = variance / 8192.0;
                    for batch in [64, 128, 256] {
                        let count = 8192 / batch;
                        let scatter = rows
                            .chunks_exact(batch)
                            .map(|xs| {
                                (xs.iter().map(|x| x[j]).sum::<f64>() / batch as f64 - mean).powi(2)
                            })
                            .sum::<f64>();
                        variance_mean = variance_mean.max(scatter / (count * (count - 1)) as f64);
                    }
                    let ess = variance / variance_mean;
                    if !ess.is_finite() || ess <= 0.0 {
                        return Err("invalid batch ESS".into());
                    }
                    writeln!(
                        out,
                        "{target:?},{name},{seed},{j},{mean:.17e},{variance:.17e},{ess:.17e},{:.17e},{warmup_calls},{},{elapsed:.9},{},{divergences},{rank},{:.17e}",
                        variance_mean.sqrt(),
                        counted.1.get(),
                        report.divergences,
                        report.step_size.value()
                    )?;
                }
            }
        }
    }
    out.flush()?;
    Ok(())
}
