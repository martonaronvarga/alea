//! Repeat fixed work for perf/Callgrind/hyperfine, without Criterion's analysis.
//! `fisher_profile fit <repetitions> <dimension> <history>` or
//! `fisher_profile <covariance|diagonal|lowrank> <repetitions> <dimension>`.
//! Use the same compiled binary for timing and profiling; no CSV in the loop.
#[allow(dead_code)]
#[path = "../benches/support/warmup.rs"]
mod workload;

use std::{error::Error, hint::black_box};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let mode = args.next().ok_or("missing workload")?;
    let repetitions: usize = args.next().ok_or("missing repetitions")?.parse()?;
    let dimension: usize = args.next().ok_or("missing dimension")?.parse()?;
    if repetitions == 0 || dimension < 4 {
        return Err("repetitions must be positive and dimension at least four".into());
    }
    if mode == "fit" {
        let history: usize = args.next().ok_or("missing history")?.parse()?;
        if history < 2 || args.next().is_some() {
            return Err("history must be at least two; unexpected arguments".into());
        }
        fit(repetitions, dimension, history)?;
    } else {
        let method = match mode.as_str() {
            "covariance" => workload::Method::Covariance,
            "diagonal" => workload::Method::FisherDiagonal,
            #[cfg(feature = "faer")]
            "lowrank" => workload::Method::FisherLowRank,
            _ => return Err("unknown workload (fit/lowrank require faer)".into()),
        };
        if args.next().is_some() {
            return Err("unexpected arguments".into());
        }
        let target = workload::Target::new(dimension, workload::Shape::Correlated);
        for _ in 0..repetitions {
            black_box(workload::run(black_box(&target), method, 7));
        }
    }
    Ok(())
}

#[cfg(feature = "faer")]
fn fit(repetitions: usize, dimension: usize, history: usize) -> Result<(), Box<dyn Error>> {
    use alea_core::target::LogDensityGradient;
    use alea_mcmc::adapt::FisherMetricAdapter;
    use rand::{RngExt, SeedableRng, rngs::SmallRng};

    // Same setup as fisher_fit_cost_v1; setup occurs once per process, not per fit.
    let target = workload::Target::new(dimension, workload::Shape::Correlated);
    let mut rng = SmallRng::seed_from_u64(7);
    let mut adapter = FisherMetricAdapter::new(dimension, history, history)?;
    let mut q = vec![0.0; dimension];
    let mut score = vec![0.0; dimension];
    for _ in 0..history {
        for x in &mut q {
            *x = rng.random_range(-1.0..1.0);
        }
        target.logp_grad(&q, &mut score)?;
        adapter.observe(&q, &score)?;
    }
    let mut scales = vec![0.0; dimension];
    adapter.scales_into(&vec![1.0; dimension], 1e-5, &mut scales)?;
    for _ in 0..repetitions {
        black_box(adapter.fit_low_rank(black_box(&scales), 1e-5, 2.0, 4)?);
    }
    Ok(())
}

#[cfg(not(feature = "faer"))]
fn fit(_: usize, _: usize, _: usize) -> Result<(), Box<dyn Error>> {
    Err("fit requires faer".into())
}
