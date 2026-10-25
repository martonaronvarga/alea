//! Pinned upstream engine adapter for the shared M16 worker protocol.
#[path = "../../../../bench/targets.rs"]
mod targets;
use nuts_rs::rand::{Rng, SeedableRng, rngs::SmallRng};
use nuts_rs::{
    Chain, CpuLogpFunc, CpuMath, CpuMathError, DiagNutsSettings, HasDims, LogpError, Settings,
    StepSizeAdaptMethod,
};
use std::{
    cell::Cell,
    collections::HashMap,
    error::Error,
    fs::OpenOptions,
    io::{BufWriter, Write},
    path::Path,
    rc::Rc,
    time::Instant,
};
struct Target {
    task: targets::Task,
    calls: Rc<Cell<usize>>,
}
#[derive(Debug)]
struct Fault {
    message: String,
    numerical: bool,
}
impl std::fmt::Display for Fault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}
impl Error for Fault {}
impl LogpError for Fault {
    fn is_recoverable(&self) -> bool {
        self.numerical
    }
}
impl HasDims for Target {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        HashMap::from([("q".into(), self.task.dimension as u64)])
    }
}
impl CpuLogpFunc for Target {
    type LogpError = Fault;
    type ExpandedVector = Vec<f64>;
    type FlowParameters = ();
    fn dim(&self) -> usize {
        self.task.dimension
    }
    fn expand_vector<R: Rng + ?Sized>(
        &mut self,
        _: &mut R,
        q: &[f64],
    ) -> Result<Vec<f64>, CpuMathError> {
        Ok(q.to_vec())
    }
    fn logp(&mut self, q: &[f64], g: &mut [f64]) -> Result<f64, Fault> {
        self.calls.set(self.calls.get() + 1);
        self.task.evaluate(q, g).map_err(|e| Fault {
            message: e.to_string(),
            numerical: true,
        })
    }
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: m16 TASK NEW_JOB_DIRECTORY".into());
    }
    let task = targets::Task::read(Path::new(&args[0]))?;
    let directory = Path::new(&args[1]);
    let mut raw = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(directory.join("draws.csv"))?,
    );
    let mut phases = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(directory.join("phases.csv"))?,
    );
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
    for (id, seed) in task.seeds.iter().enumerate() {
        let start = Instant::now();
        let calls = Rc::new(Cell::new(0));
        // Native score initialization and diagonal warmup are declared in the
        // manifest; this upstream revision cannot construct zero-tuning NUTS.
        let mut settings = DiagNutsSettings {
            num_tune: task.warmup as u64,
            num_draws: task.draws as u64,
            maxdepth: task.depth as u64,
            ..Default::default()
        };
        settings.adapt_options.step_size_settings.jitter = None;
        settings
            .adapt_options
            .step_size_settings
            .adapt_options
            .method = StepSizeAdaptMethod::Fixed(task.step);
        let target = Target {
            task: task.clone(),
            calls: Rc::clone(&calls),
        };
        let mut sampler = settings.new_chain(
            id as u64,
            CpuMath::new(target),
            &mut SmallRng::seed_from_u64(*seed),
        );
        sampler.set_position(&task.initial)?;
        let setup = start.elapsed().as_secs_f64();
        let initial = calls.get();
        let start = Instant::now();
        let mut warmup_steps = 0;
        for _ in 0..task.warmup {
            warmup_steps += sampler.draw()?.1.num_steps;
        }
        let warmup = start.elapsed().as_secs_f64();
        let before = calls.get();
        let start = Instant::now();
        let mut rows = Vec::with_capacity(task.draws);
        let mut retained_steps = 0;
        for draw in 0..task.draws {
            let (q, _expanded, stats, info) = sampler.expanded_draw()?;
            if info.tuning {
                return Err("unexpected retained adaptation".into());
            }
            retained_steps += info.num_steps;
            rows.push((
                draw,
                q,
                stats.maxdepth_reached,
                stats.adapt.step_size.mean_tree_accept,
                stats.adapt.step_size.max_energy_error,
                info,
            ));
        }
        let retained = start.elapsed().as_secs_f64();
        writeln!(
            phases,
            "{id},{setup:.17e},{warmup:.17e},{retained:.17e},{initial},{},{},{warmup_steps},{retained_steps},,",
            before - initial,
            calls.get() - before
        )?;
        for (draw, q, saturated, acceptance, energy_error, info) in rows {
            write!(
                raw,
                "{id},{draw},{},{},{},{acceptance:.17e},,{energy_error:.17e}",
                usize::from(info.diverging),
                usize::from(saturated),
                info.num_steps
            )?;
            for v in q {
                write!(raw, ",{v:.17e}")?;
            }
            writeln!(raw)?;
        }
    }
    raw.flush()?;
    phases.flush()?;
    Ok(())
}
