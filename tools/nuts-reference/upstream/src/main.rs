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
#[derive(Debug)]
struct Error;
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "reference target error")
    }
}
impl std::error::Error for Error {}
impl LogpError for Error {
    fn is_recoverable(&self) -> bool {
        false
    }
}
struct Target(usize);
impl HasDims for Target {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        HashMap::from([("unconstrained_parameter".into(), 2)])
    }
}
impl CpuLogpFunc for Target {
    type LogpError = Error;
    type ExpandedVector = Vec<f64>;
    type FlowParameters = ();
    fn dim(&self) -> usize {
        2
    }
    fn expand_vector<R: Rng + ?Sized>(
        &mut self,
        _: &mut R,
        q: &[f64],
    ) -> Result<Vec<f64>, CpuMathError> {
        Ok(q.to_vec())
    }
    fn logp(&mut self, q: &[f64], g: &mut [f64]) -> Result<f64, Error> {
        Ok(match self.0 {
            0 => {
                let z = (q[1] - 0.6 * q[0]) / 0.8;
                g[0] = -q[0] + 0.75 * z;
                g[1] = -z / 0.8;
                -0.5 * (q[0] * q[0] + z * z)
            }
            1 => {
                let z = q[1] - 0.4 * (q[0] * q[0] - 1.0);
                g[0] = -q[0] + 0.8 * q[0] * z;
                g[1] = -z;
                -0.5 * (q[0] * q[0] + z * z)
            }
            2 => {
                let mut lp = -0.5 * (q[0] * q[0] + q[1] * q[1]);
                g[0] = -q[0];
                g[1] = -q[1];
                for (x, y) in [(-1.5, 0.0), (-0.2, 1.0), (0.7, 0.0), (2.0, 1.0)] {
                    let eta = q[0] + q[1] * x;
                    let t = (-eta.abs()).exp();
                    let p = if eta >= 0.0 {
                        1.0 / (1.0 + t)
                    } else {
                        t / (1.0 + t)
                    };
                    lp += y * eta - eta.max(0.0) - t.ln_1p();
                    g[0] += y - p;
                    g[1] += x * (y - p);
                }
                lp
            }
            3 => {
                let prec = (-q[0]).exp();
                g[0] = -q[0] - 0.5 + 0.5 * q[1] * q[1] * prec;
                g[1] = -q[1] * prec;
                -0.5 * q[0] * q[0] - 0.5 * q[0] - 0.5 * q[1] * q[1] * prec
            }
            _ => return Err(Error),
        })
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("provide new output CSV")?;
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
    for t in 0..4 {
        for seed in 501..505 {
            let mut settings = DiagNutsSettings {
                num_tune: 1024,
                num_draws: 8192,
                maxdepth: 8,
                ..Default::default()
            };
            settings
                .adapt_options
                .step_size_settings
                .adapt_options
                .method = StepSizeAdaptMethod::Fixed(if t == 3 { 0.06 } else { 0.15 });
            let mut sampler = settings.new_chain(
                0,
                CpuMath::new(Target(t)),
                &mut SmallRng::seed_from_u64(seed),
            );
            let sign = if seed % 2 == 0 { -1.0 } else { 1.0 };
            sampler.set_position(&[sign, -sign])?;
            let mut rows = Vec::with_capacity(8192);
            let start = Instant::now();
            let mut warmup_steps = 0;
            for _ in 0..1024 {
                warmup_steps += sampler.draw()?.1.num_steps;
            }
            let warmup_seconds = start.elapsed().as_secs_f64();
            let start = Instant::now();
            for _ in 0..8192 {
                let (q, info) = sampler.draw()?;
                rows.push(([q[0], q[1]], info));
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
                    "{t},{seed},{},{:.17e},{:.17e},{},{},-1",
                    draw,
                    q[0],
                    q[1],
                    usize::from(info.diverging),
                    info.num_steps
                )?;
            }
        }
        eprintln!("nuts-rs completed target {t}");
    }
    out.flush()?;
    Ok(())
}
