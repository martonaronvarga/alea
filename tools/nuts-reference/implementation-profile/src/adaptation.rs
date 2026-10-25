//! All cases and diagnostics are retained, including divergent/undefined ones.
use super::*;
use alea_mcmc::{
    adapt::{FisherNutsWarmup, FisherOptions, NutsWarmup, NutsWarmupReport, WarmupOptions},
    nuts::NutsTermination,
};
use alea_runtime::diagnostics::modern::summarize;
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
};
#[path = "../../../../crates/alea-mcmc/tests/support/targets.rs"]
mod targets;

struct Counted(targets::Target, Cell<usize>);
impl LogDensityGradient for Counted {
    type Error = std::convert::Infallible;
    fn dimension(&self) -> usize {
        2
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Self::Error> {
        self.1.set(self.1.get() + 1);
        self.0.logp_grad(q, g)
    }
}
struct Chain {
    q: Vec<[f64; 2]>,
    warm_calls: usize,
    retained_calls: usize,
    warm_divergences: usize,
    divergences: usize,
    depth: usize,
    warm_ns: u128,
    retained_ns: u128,
    step: f64,
    rank: usize,
    g: [f64; 4],
}
fn retain<M: EuclideanMetric>(
    target: &Counted,
    mut chain: Nuts<'_, Counted, M>,
    report: NutsWarmupReport,
    warm_ns: u128,
    rank: usize,
    rng: &mut SmallRng,
) -> Result<Chain, Box<dyn Error>> {
    let mut g = [0.; 4];
    chain.metric().velocity(&[1., 0.], &mut g[..2])?;
    chain.metric().velocity(&[0., 1.], &mut g[2..])?;
    let step = chain.options().step_size().value();
    let warm_calls = target.1.get();
    let mut q = Vec::with_capacity(2048);
    let mut divergences = 0;
    let mut depth = 0;
    let start = Instant::now();
    for _ in 0..2048 {
        let info = chain.step(rng)?;
        divergences += usize::from(matches!(info.termination, NutsTermination::Divergence(_)));
        depth += usize::from(info.termination == NutsTermination::MaxDepth);
        q.push([chain.point().position()[0], chain.point().position()[1]]);
    }
    Ok(Chain {
        q,
        warm_calls,
        retained_calls: target.1.get() - warm_calls,
        warm_divergences: report.divergences,
        divergences,
        depth,
        warm_ns,
        retained_ns: start.elapsed().as_nanos(),
        step,
        rank,
        g,
    })
}

fn chain(target: &Counted, method: &str, seed: u64) -> Result<Chain, Box<dyn Error>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let nuts = NutsOptions::new(0.1, 8)?;
    let start = Instant::now();
    if method == "cov_diag" {
        let (chain, report) = NutsWarmup::new(
            target,
            OwnedBuffer::from_fn(2, |i| [0.25, -0.25][i]),
            nuts,
            WarmupOptions::new(512),
        )?
        .run(&mut rng)?;
        retain(
            target,
            chain,
            report,
            start.elapsed().as_nanos(),
            0,
            &mut rng,
        )
    } else {
        let period = if method.ends_with("40") { 40 } else { 80 };
        let rank = if method.starts_with("fisher_rank2") {
            2
        } else {
            0
        };
        let options = FisherOptions::new(512)
            .with_max_rank(rank)
            .with_window_budget(period, 160)?;
        let (chain, report) = FisherNutsWarmup::new(
            target,
            OwnedBuffer::from_fn(2, |i| [0.25, -0.25][i]),
            nuts,
            options,
        )?
        .run(&mut rng)?;
        let rank = chain.metric().rank();
        retain(
            target,
            chain,
            report,
            start.elapsed().as_nanos(),
            rank,
            &mut rng,
        )
    }
}

pub fn run(directory: &str) -> Result<(), Box<dyn Error>> {
    let directory = Path::new(directory);
    fs::create_dir(directory)?;
    let mut raw = BufWriter::new(File::create_new(directory.join("draws.csv"))?);
    let mut work = BufWriter::new(File::create_new(directory.join("work.csv"))?);
    let mut diagnostics = BufWriter::new(File::create_new(directory.join("diagnostics.csv"))?);
    writeln!(raw, "case,method,chain,index,q0,q1")?;
    writeln!(
        work,
        "case,method,chain,seed,status,error,warm_calls,retained_calls,warm_divergences,divergences,depth_saturations,warm_ns,retained_ns,step,rank,g00,g10,g01,g11"
    )?;
    writeln!(
        diagnostics,
        "case,method,observable,status,rhat,bulk_ess,tail_ess,mean,mcse_mean"
    )?;
    for name in ["correlated", "banana", "logistic", "funnel", "rotated"] {
        for method in [
            "cov_diag",
            "fisher_diag80",
            "fisher_diag40",
            "fisher_rank280",
            "fisher_rank240",
        ] {
            let mut observations: Vec<Vec<Vec<f64>>> = (0..5).map(|_| Vec::new()).collect();
            for index in 0..4 {
                let target = Counted(targets::Target::parse(name), Cell::new(0));
                let seed = 23001 + index;
                let result = match chain(&target, method, seed) {
                    Ok(result) => result,
                    Err(error) => {
                        let mut fields = vec![String::new(); 19];
                        fields[..6].clone_from_slice(&[
                            name.into(),
                            method.into(),
                            index.to_string(),
                            seed.to_string(),
                            "failed".into(),
                            format!("\"{}\"", error.to_string().replace('"', "\"\"")),
                        ]);
                        writeln!(work, "{}", fields.join(","))?;
                        continue;
                    }
                };
                writeln!(
                    work,
                    "{name},{method},{index},{seed},completed,,{},{},{},{},{},{},{},{},{},{},{},{},{}",
                    result.warm_calls,
                    result.retained_calls,
                    result.warm_divergences,
                    result.divergences,
                    result.depth,
                    result.warm_ns,
                    result.retained_ns,
                    result.step,
                    result.rank,
                    result.g[0],
                    result.g[1],
                    result.g[2],
                    result.g[3]
                )?;
                for (draw, q) in result.q.iter().enumerate() {
                    writeln!(raw, "{name},{method},{index},{draw},{},{}", q[0], q[1])?;
                }
                for (observable, chains) in observations.iter_mut().enumerate() {
                    chains.push(
                        result
                            .q
                            .iter()
                            .map(|q| target.0.observables(q)[observable])
                            .collect(),
                    );
                }
            }
            for (observable, chains) in observations.iter().enumerate() {
                if chains.len() != 4 {
                    writeln!(
                        diagnostics,
                        "{name},{method},{observable},incomplete_chains,,,,,"
                    )?;
                    continue;
                }
                let slices: Vec<_> = chains.iter().map(Vec::as_slice).collect();
                match summarize(&slices) {
                    Ok(s) => writeln!(
                        diagnostics,
                        "{name},{method},{observable},defined,{},{},{},{},{}",
                        s.rhat, s.bulk_ess, s.tail_ess, s.mean, s.mcse_mean
                    )?,
                    Err(error) => {
                        writeln!(diagnostics, "{name},{method},{observable},{error},,,,,")?
                    }
                }
            }
        }
    }
    raw.flush()?;
    work.flush()?;
    diagnostics.flush()?;
    Ok(())
}
