//! Higher-dimensional benchmark; timings exclude CSV formatting.
#[path = "../../../tools/nuts-reference/hard/target.rs"]
mod target;
use alea_core::target::LogDensityGradient;
use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};
use alea_mcmc::{Nuts, NutsOptions, nuts::NutsTermination};
use rand::{SeedableRng, rngs::SmallRng};
use std::{
    convert::Infallible,
    fs::OpenOptions,
    io::{BufWriter, Write},
    time::Instant,
};
use target::{DEPTH, DRAWS, Target, WARMUP};
impl LogDensityGradient for Target {
    type Error = Infallible;
    fn dimension(&self) -> usize {
        self.dimension
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Infallible> {
        Ok(self.evaluate(q, g))
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let case: usize = args
        .get(1)
        .ok_or("provide case, new draw path and new timing path")?
        .parse()?;
    let target = Target::new(case).ok_or("unknown benchmark case")?;
    if args.get(2).is_some_and(|s| s == "--probe") {
        for offset in [-0.4, 0.1, 0.9] {
            let q: Vec<_> = (0..target.dimension)
                .map(|i| offset + 0.03 * (i % 7) as f64)
                .collect();
            let mut g = vec![0.; q.len()];
            let lp = target.evaluate(&q, &mut g);
            print!("{lp:.17e}");
            for v in g {
                print!(",{v:.17e}");
            }
            println!();
        }
        return Ok(());
    }
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
            .open(args.get(3).ok_or("missing timing path")?)?,
    );
    write!(out, "seed,draw,divergent,steps")?;
    for i in 0..target.dimension {
        write!(out, ",q{i}")?;
    }
    writeln!(out)?;
    writeln!(
        timing,
        "seed,warmup_seconds,sampling_seconds,warmup_steps,setup_calls,warmup_calls,sampling_calls"
    )?;
    for seed in 701..705 {
        let before = target.calls.get();
        let sign = if seed % 2 == 0 { -0.1 } else { 0.1 };
        let mut chain = Nuts::new(
            &target,
            OwnedBuffer::from_fn(target.dimension, |i| if i % 2 == 0 { sign } else { -sign }),
            IdentityMetric::new(target.dimension),
            NutsOptions::new(target.step, DEPTH)?,
        )?;
        let setup = target.calls.get() - before;
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut positions = Vec::with_capacity(DRAWS * target.dimension);
        let mut stats = Vec::with_capacity(DRAWS);
        let before = target.calls.get();
        let start = Instant::now();
        let mut warm_steps = 0;
        for _ in 0..WARMUP {
            warm_steps += chain.step(&mut rng)?.integration_steps;
        }
        let warm = start.elapsed().as_secs_f64();
        let warm_calls = target.calls.get() - before;
        let before = target.calls.get();
        let start = Instant::now();
        for _ in 0..DRAWS {
            let info = chain.step(&mut rng)?;
            positions.extend_from_slice(chain.point().position());
            stats.push((
                usize::from(matches!(info.termination, NutsTermination::Divergence(_))),
                info.integration_steps,
            ));
        }
        let sample = start.elapsed().as_secs_f64();
        let calls = target.calls.get() - before;
        writeln!(
            timing,
            "{seed},{warm},{sample},{warm_steps},{setup},{warm_calls},{calls}"
        )?;
        for (draw, (q, (div, steps))) in positions
            .chunks_exact(target.dimension)
            .zip(stats)
            .enumerate()
        {
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
