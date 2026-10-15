//! Fixed M5 correctness experiment; raw coordinates, no thinning/rejection removal.
#[path = "../tests/support/targets.rs"]
mod targets;
use alea_math::{buffer::OwnedBuffer, metric::IdentityMetric};
use alea_mcmc::{Nuts, NutsOptions, nuts::NutsTermination};
use rand::{SeedableRng, rngs::SmallRng};
use std::{
    fs::OpenOptions,
    io::{BufWriter, Write},
    time::Instant,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("provide a new CSV path")?;
    let mut out = BufWriter::new(OpenOptions::new().write(true).create_new(true).open(path)?);
    writeln!(out, "target,seed,draw,q0,q1,divergent,steps,acceptance")?;
    let mut timing = std::env::var_os("ALEA_BENCH_TIMINGS")
        .map(|p| OpenOptions::new().write(true).create_new(true).open(p))
        .transpose()?;
    if let Some(file) = &mut timing {
        writeln!(
            file,
            "target,seed,warmup_seconds,sampling_seconds,warmup_steps"
        )?;
    }
    for (t, target) in [
        targets::Target::Correlated,
        targets::Target::Banana,
        targets::Target::Logistic,
        targets::Target::Funnel,
    ]
    .into_iter()
    .enumerate()
    {
        for seed in 501..505 {
            let sign = if seed % 2 == 0 { -1.0 } else { 1.0 };
            let mut chain = Nuts::new(
                &target,
                OwnedBuffer::from_fn(2, |i| if i == 0 { sign } else { -sign }),
                IdentityMetric::new(2),
                NutsOptions::new(if t == 3 { 0.06 } else { 0.15 }, 8)?,
            )?;
            let mut rng = SmallRng::seed_from_u64(seed);
            let mut rows = Vec::with_capacity(8192);
            let start = Instant::now();
            let mut warmup_steps = 0;
            for _ in 0..1024 {
                warmup_steps += chain.step(&mut rng)?.integration_steps;
            }
            let warmup_seconds = start.elapsed().as_secs_f64();
            let start = Instant::now();
            for _ in 0..8192 {
                let info = chain.step(&mut rng)?;
                rows.push((
                    [chain.point().position()[0], chain.point().position()[1]],
                    info,
                ));
            }
            let sampling_seconds = start.elapsed().as_secs_f64();
            if let Some(file) = &mut timing {
                writeln!(
                    file,
                    "{t},{seed},{warmup_seconds},{sampling_seconds},{warmup_steps}"
                )?;
            }
            for (draw, (q, info)) in rows.into_iter().enumerate() {
                writeln!(
                    out,
                    "{t},{seed},{},{:.17e},{:.17e},{},{},{:.17e}",
                    draw,
                    q[0],
                    q[1],
                    usize::from(matches!(info.termination, NutsTermination::Divergence(_))),
                    info.integration_steps,
                    info.acceptance_probability
                )?;
            }
        }
    }
    out.flush()?;
    Ok(())
}
