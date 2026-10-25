//! Experiment launcher and isolated numerical worker; not a new sampler API.
#[path = "../../../../tools/bench/targets.rs"]
mod targets;
use alea_core::target::LogDensityGradient;
use alea_distributions::wiener::{
    Boundary, Wiener4, Wiener4Params, WienerObservation, WienerOptions,
};
use alea_math::{
    buffer::OwnedBuffer,
    metric::{EuclideanMetric, IdentityMetric, MetricError},
};
use alea_mcmc::{Nuts, NutsOptions};
use alea_runtime::{diagnostics::modern, ensemble::chain_rng, output::ChainId};
use std::{
    cell::Cell,
    error::Error,
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    rc::Rc,
    time::Instant,
};
use targets::{Failure, Task};
struct Target<'a> {
    task: &'a Task,
    calls: Cell<usize>,
}
impl LogDensityGradient for Target<'_> {
    type Error = Failure;
    fn dimension(&self) -> usize {
        self.task.dimension
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Failure> {
        self.calls.set(self.calls.get() + 1);
        if self.task.kind != 8 {
            return match self.task.evaluate(q, g) {
                Err(e) if e.0 == "nonfinite model evaluation" => Ok(f64::NEG_INFINITY),
                other => other,
            };
        }
        let mut lp = -0.5 * q[0] * q[0];
        g[0] = -q[0];
        let params =
            Wiener4Params::with_params(1.2, 0.15, 0.5, q[0]).map_err(|e| Failure(e.to_string()))?;
        for row in self.task.data.chunks_exact(3) {
            let event = WienerObservation {
                rt: row[0],
                boundary: if row[1] > 0.0 {
                    Boundary::Upper
                } else {
                    Boundary::Lower
                },
            };
            let eval = Wiener4
                .try_fused(&event, &params, WienerOptions::default())
                .map_err(|e| Failure(e.to_string()))?;
            lp += eval.log_prob;
            g[0] += eval.grad_array()[3];
        }
        Ok(lp)
    }
}
#[derive(Default)]
struct Actions {
    velocity: Cell<usize>,
    momentum: Cell<usize>,
}
struct Metric {
    inner: IdentityMetric,
    actions: Rc<Actions>,
}
impl EuclideanMetric for Metric {
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }
    fn velocity(&self, p: &[f64], out: &mut [f64]) -> Result<(), MetricError> {
        self.actions.velocity.set(self.actions.velocity.get() + 1);
        self.inner.velocity(p, out)
    }
    fn sample_momentum(&self, z: &[f64], out: &mut [f64]) -> Result<(), MetricError> {
        self.actions.momentum.set(self.actions.momentum.get() + 1);
        self.inner.sample_momentum(z, out)
    }
    fn log_det(&self) -> f64 {
        self.inner.log_det()
    }
}
fn create(path: impl AsRef<Path>) -> Result<BufWriter<File>, Box<dyn Error>> {
    Ok(BufWriter::new(
        OpenOptions::new().write(true).create_new(true).open(path)?,
    ))
}
fn number(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.17e}")
    } else {
        "null".into()
    }
}
fn quoted(value: &str) -> String {
    format!(
        "\"{}\"",
        value
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t")
    )
}
fn worker(task: &Task, directory: &Path) -> Result<(), Box<dyn Error>> {
    let mut raw = create(directory.join("draws.csv"))?;
    let mut phases = create(directory.join("phases.csv"))?;
    write!(
        raw,
        "chain,draw,divergent,saturated,steps,acceptance,energy,energy_error"
    )?;
    for i in 0..task.dimension {
        write!(raw, ",q{i}")?;
    }
    writeln!(raw)?;
    writeln!(
        phases,
        "chain,setup_seconds,warmup_seconds,retained_seconds,setup_gradient_calls,warmup_gradient_calls,retained_gradient_calls,warmup_steps,retained_steps,velocity_actions,momentum_actions"
    )?;
    for (id, &seed) in task.seeds.iter().enumerate() {
        let start = Instant::now();
        let target = Target {
            task,
            calls: Cell::new(0),
        };
        let actions = Rc::new(Actions::default());
        let metric = Metric {
            inner: IdentityMetric::new(task.dimension),
            actions: Rc::clone(&actions),
        };
        let q = OwnedBuffer::from_fn(task.dimension, |i| task.initial[i]);
        let mut sampler = Nuts::new(&target, q, metric, NutsOptions::new(task.step, task.depth)?)?;
        let mut rng = chain_rng([u8::try_from(seed % 256)?; 32], ChainId(seed));
        let setup = start.elapsed().as_secs_f64();
        let initial = target.calls.get();
        let start = Instant::now();
        let mut warmup_steps = 0;
        for _ in 0..task.warmup {
            warmup_steps += sampler.step(&mut rng)?.integration_steps;
        }
        let warmup = start.elapsed().as_secs_f64();
        let before = target.calls.get();
        let start = Instant::now();
        let mut rows = Vec::with_capacity(task.draws);
        let mut retained_steps = 0;
        for draw in 0..task.draws {
            let info = sampler.step(&mut rng)?;
            retained_steps += info.integration_steps;
            rows.push((draw, sampler.point().position().to_vec(), info));
        }
        let retained = start.elapsed().as_secs_f64();
        writeln!(
            phases,
            "{id},{setup:.17e},{warmup:.17e},{retained:.17e},{initial},{},{},{warmup_steps},{retained_steps},{},{}",
            before - initial,
            target.calls.get() - before,
            actions.velocity.get(),
            actions.momentum.get()
        )?;
        // Output serialization is outside retained time; process wall includes it.
        for (draw, q, info) in rows {
            write!(
                raw,
                "{id},{draw},{},{},{},{:.17e},{},{}",
                usize::from(matches!(
                    info.termination,
                    alea_mcmc::nuts::NutsTermination::Divergence(_)
                )),
                usize::from(info.termination == alea_mcmc::nuts::NutsTermination::MaxDepth),
                info.integration_steps,
                info.acceptance_probability,
                info.initial_energy.map_or_else(String::new, number),
                info.max_energy_error.map_or_else(String::new, number)
            )?;
            for v in q {
                write!(raw, ",{v:.17e}")?;
            }
            writeln!(raw)?;
        }
    }
    raw.flush()?;
    phases.flush()?;
    let threads = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines().find_map(|l| {
                l.strip_prefix("Threads:")
                    .and_then(|v| v.trim().parse::<usize>().ok())
            })
        });
    let mut metadata = create(directory.join("engine.json"))?;
    writeln!(
        metadata,
        "{{\"versions\":{{\"alea\":\"{}\"}},\"jit_seconds\":0,\"cold_seconds\":null,\"actual_threads\":{}}}",
        env!("CARGO_PKG_VERSION"),
        threads.map_or_else(|| "null".into(), |n| n.to_string())
    )?;
    metadata.flush()?;
    Ok(())
}
fn probe(task: &Task, out: &Path) -> Result<(), Box<dyn Error>> {
    let target = Target {
        task,
        calls: Cell::new(0),
    };
    let mut g = vec![0.0; task.dimension];
    let lp = target.logp_grad(&task.initial, &mut g)?;
    let mut output = create(out)?;
    write!(
        output,
        "{{\"schema_version\":1,\"log_density\":{},\"gradient\":[",
        number(lp)
    )?;
    for (i, v) in g.iter().enumerate() {
        if i > 0 {
            write!(output, ",")?;
        }
        write!(output, "{}", number(*v))?;
    }
    writeln!(output, "]}}")?;
    output.flush()?;
    Ok(())
}
fn summaries(
    output: &mut impl Write,
    chains: &[Vec<Vec<f64>>],
    squared: bool,
) -> Result<(), Box<dyn Error>> {
    for i in 0..chains[0].len() {
        if i > 0 {
            write!(output, ",")?;
        }
        let columns: Vec<Vec<f64>> = chains
            .iter()
            .map(|c| {
                c[i].iter()
                    .map(|v| if squared { v * v } else { *v })
                    .collect()
            })
            .collect();
        let views: Vec<_> = columns.iter().map(Vec::as_slice).collect();
        match modern::summarize(&views) {
            Ok(s) => write!(
                output,
                "{{\"mean\":{},\"rhat\":{},\"bulk_ess\":{},\"tail_ess\":{},\"mcse\":{},\"error\":null}}",
                number(s.mean),
                number(s.rhat),
                number(s.bulk_ess),
                number(s.tail_ess),
                number(s.mcse_mean)
            )?,
            Err(e) => write!(
                output,
                "{{\"mean\":null,\"rhat\":null,\"bulk_ess\":null,\"tail_ess\":null,\"mcse\":null,\"error\":{}}}",
                quoted(&e.to_string())
            )?,
        }
    }
    Ok(())
}
fn analyze(task: &Task, path: &Path, out: &Path) -> Result<(), Box<dyn Error>> {
    let mut chains = vec![vec![Vec::with_capacity(task.draws); task.dimension]; task.seeds.len()];
    let mut energy = vec![Vec::new(); task.seeds.len()];
    let mut divergences = 0;
    let mut saturated = 0;
    let mut rows = 0;
    let mut steps = 0;
    let mut acceptance_total = 0.0;
    let mut acceptance_observations = 0_usize;
    let mut maximum_energy_error: Option<f64> = None;
    let mut nonfinite_energy_errors = 0_usize;
    let mut lines = BufReader::new(File::open(path)?).lines();
    let header = lines.next().ok_or("missing raw header")??;
    let expected = format!(
        "chain,draw,divergent,saturated,steps,acceptance,energy,energy_error,{}",
        (0..task.dimension)
            .map(|i| format!("q{i}"))
            .collect::<Vec<_>>()
            .join(",")
    );
    if header != expected {
        return Err("raw schema mismatch".into());
    }
    for line in lines {
        let line = line?;
        let fields: Vec<_> = line.split(',').collect();
        if fields.len() != 8 + task.dimension {
            return Err("raw row width".into());
        }
        let id: usize = fields[0].parse()?;
        let index: usize = fields[1].parse()?;
        if id >= chains.len() || index != chains[id][0].len() || index >= task.draws {
            return Err("raw chain/index mismatch".into());
        }
        let divergent: usize = fields[2].parse()?;
        let depth: usize = fields[3].parse()?;
        if divergent > 1 || depth > 1 {
            return Err("invalid transition flags".into());
        }
        divergences += divergent;
        saturated += depth;
        steps += fields[4].parse::<usize>()?;
        if !fields[5].is_empty() {
            let a: f64 = fields[5].parse()?;
            // Upstream log-sum reductions can round a probability one ulp
            // above one. Preserve the original CSV; permit only an explicit
            // floating-point envelope here, never arbitrary invalid values.
            if !a.is_finite() || !(0.0..=1.0 + 8.0 * f64::EPSILON).contains(&a) {
                return Err("invalid acceptance probability".into());
            }
            acceptance_total += a.min(1.0);
            acceptance_observations += 1;
        }
        if !fields[7].is_empty() {
            let error: f64 = fields[7].parse()?;
            if !error.is_finite() {
                nonfinite_energy_errors += 1;
            } else {
                maximum_energy_error =
                    Some(maximum_energy_error.map_or(error.abs(), |old| old.max(error.abs())));
            }
        }
        let e = if fields[6].is_empty() {
            None
        } else {
            Some(fields[6].parse::<f64>()?)
        };
        if e.is_some_and(|v| !v.is_finite()) {
            return Err("nonfinite energy".into());
        }
        energy[id].push(e);
        for i in 0..task.dimension {
            let v: f64 = fields[8 + i].parse()?;
            if !v.is_finite() {
                return Err("nonfinite coordinate".into());
            }
            chains[id][i].push(v);
        }
        rows += 1;
    }
    if chains.iter().any(|c| c[0].len() != task.draws) {
        return Err("lost/partial chain draws".into());
    }
    let mut output = create(out)?;
    write!(
        output,
        "{{\"schema_version\":1,\"draws\":{rows},\"divergences\":{divergences},\"depth_saturations\":{saturated},\"integration_steps\":{steps},\"coordinates\":["
    )?;
    summaries(&mut output, &chains, false)?;
    write!(output, "],\"second_moments\":[")?;
    summaries(&mut output, &chains, true)?;
    write!(
        output,
        "],\"nonfinite_energy_errors\":{nonfinite_energy_errors},\"acceptance_observations\":{acceptance_observations},\"mean_acceptance\":{},\"maximum_reported_energy_error\":{},\"ebfmi\":[",
        if acceptance_observations == 0 {
            "null".into()
        } else {
            number(acceptance_total / acceptance_observations as f64)
        },
        maximum_energy_error.map_or_else(|| "null".into(), number)
    )?;
    for (i, e) in energy.iter().enumerate() {
        if i > 0 {
            write!(output, ",")?;
        }
        let all: Option<Vec<f64>> = e.iter().copied().collect();
        let value = all.and_then(|v| modern::ebfmi(&v).ok());
        write!(output, "{}", value.map_or_else(|| "null".into(), number))?;
    }
    writeln!(output, "]}}")?;
    output.flush()?;
    Ok(())
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args
        .first()
        .is_some_and(|a| a == "--worker" || a == "--probe" || a == "--analyze")
    {
        let expected = if args[0] == "--analyze" { 4 } else { 3 };
        if args.len() != expected {
            return Err("invalid worker arguments".into());
        }
        let task = Task::read(Path::new(&args[1]))?;
        if args[0] == "--worker" {
            worker(&task, Path::new(&args[2]))
        } else if args[0] == "--probe" {
            probe(&task, Path::new(&args[2]))
        } else {
            analyze(&task, Path::new(&args[2]), Path::new(&args[3]))
        }
    } else {
        let root = std::env::var_os("ALEA_BENCH_ROOT")
            .map(PathBuf::from)
            .unwrap_or(std::env::current_dir()?);
        let status = std::process::Command::new("python3")
            .arg(root.join("tools/bench/alea-bench"))
            .args(args)
            .current_dir(root)
            .status()?;
        if status.success() {
            Ok(())
        } else {
            std::process::exit(status.code().unwrap_or(1))
        }
    }
}
