#[path = "../../../hard/target.rs"]
mod target;
use nuts_rs::rand::{Rng, SeedableRng, rngs::SmallRng};
use nuts_rs::{
    Chain, CpuLogpFunc, CpuMath, CpuMathError, DiagNutsSettings, HasDims, LogpError, Settings,
    StepSizeAdaptMethod,
};
use std::{
    collections::HashMap,
    fs::OpenOptions,
    io::{BufWriter, Write},
    time::Instant,
};
use target::{DEPTH, DRAWS, Target, WARMUP};
#[derive(Debug)]
pub struct Error;
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "benchmark target error")
    }
}
impl std::error::Error for Error {}
impl LogpError for Error {
    fn is_recoverable(&self) -> bool {
        false
    }
}
impl HasDims for Target {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        HashMap::from([("parameter".into(), self.dimension as u64)])
    }
}
impl CpuLogpFunc for Target {
    type LogpError = Error;
    type ExpandedVector = Vec<f64>;
    type FlowParameters = ();
    fn dim(&self) -> usize {
        self.dimension
    }
    fn expand_vector<R: Rng + ?Sized>(
        &mut self,
        _: &mut R,
        q: &[f64],
    ) -> Result<Vec<f64>, CpuMathError> {
        Ok(q.to_vec())
    }
    fn logp(&mut self, q: &[f64], g: &mut [f64]) -> Result<f64, Error> {
        Ok(self.evaluate(q, g))
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let case: usize = args
        .get(1)
        .ok_or("provide case and new output paths")?
        .parse()?;
    let template = Target::new(case).ok_or("unknown case")?;
    let d = template.dimension;
    let mut out = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(args.get(2).ok_or("missing draws path")?)?,
    );
    let mut timing = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(args.get(3).ok_or("missing timings path")?)?,
    );
    write!(out, "seed,draw,divergent,steps")?;
    for i in 0..d {
        write!(out, ",q{i}")?;
    }
    writeln!(out)?;
    writeln!(timing, "seed,warmup_seconds,sampling_seconds,warmup_steps")?;
    for seed in 701..705 {
        let mut settings = DiagNutsSettings {
            num_tune: WARMUP as u64,
            num_draws: DRAWS as u64,
            maxdepth: DEPTH as u64,
            ..Default::default()
        };
        settings
            .adapt_options
            .step_size_settings
            .adapt_options
            .method = StepSizeAdaptMethod::Fixed(template.step);
        let mut sampler = settings.new_chain(
            0,
            CpuMath::new(Target::new(case).ok_or("unknown case")?),
            &mut SmallRng::seed_from_u64(seed),
        );
        let sign = if seed % 2 == 0 { -0.1 } else { 0.1 };
        sampler.set_position(
            &(0..d)
                .map(|i| if i % 2 == 0 { sign } else { -sign })
                .collect::<Vec<_>>(),
        )?;
        let mut positions = Vec::with_capacity(DRAWS * d);
        let mut stats = Vec::with_capacity(DRAWS);
        let start = Instant::now();
        let mut warm_steps = 0;
        for _ in 0..WARMUP {
            warm_steps += sampler.draw()?.1.num_steps;
        }
        let warm = start.elapsed().as_secs_f64();
        let start = Instant::now();
        for _ in 0..DRAWS {
            let (q, info) = sampler.draw()?;
            positions.extend_from_slice(&q);
            stats.push((usize::from(info.diverging), info.num_steps));
        }
        let sample = start.elapsed().as_secs_f64();
        writeln!(timing, "{seed},{warm},{sample},{warm_steps}")?;
        for (draw, (q, (div, steps))) in positions.chunks_exact(d).zip(stats).enumerate() {
            write!(out, "{seed},{draw},{div},{steps}")?;
            for x in q {
                write!(out, ",{x:.17e}")?;
            }
            writeln!(out)?;
        }
    }
    out.flush()?;
    timing.flush()?;
    Ok(())
}
