//! End-to-end NUTS benchmark with Alea-native model assembly and likelihoods.
//! The observer forwards the native target; no hand-written density functions.
//!
//! Models are assembled using the public Alea main-branch API:
//! - ModelBuilder + distributions::modeling::normal for priors;
//! - regression::normal_regression for normal location/linear regression;
//! - regression::bernoulli_logit_regression for binary logistic regression;
//! - normal factors and expression graph for the centered Neal funnel.
#![cfg_attr(feature = "enzyme", feature(autodiff))]

mod conformance;
#[cfg(feature = "enzyme")]
pub mod generated_covariance {
    include!(concat!(env!("OUT_DIR"), "/covariance.rs"));
}

use alea::distributions::modeling::{
    normal,
    regression::{BernoulliLogitRegression, NormalRegression, RegressionData, RegressionExecution},
};
use alea::model::assembly::{AssemblyError, CompiledModel, ModelBuilder};
use alea::sampling::{NutsOptions, adapt::MetricKind};
use alea::target::{LogDensityGradient, TargetSpace};
use alea::{
    ensemble::Execution,
    session::{SamplerConfig, SessionBuilder, Warmup, storage::Discard},
};
use std::{
    env,
    error::Error,
    fs,
    sync::{Arc, Mutex},
    time::Instant,
};

#[cfg(test)]
mod correctness;

// Flat whitespace-delimited rows, shared with the Stan evaluator. Loading and
// validation are construction work; copying the next state is timed on both.
fn state_bank(dimension: usize) -> Result<Option<Vec<f64>>, Box<dyn Error>> {
    env::var("ALEA_BENCH_STATES")
        .ok()
        .map(|path| parse_state_bank(&fs::read_to_string(path)?, dimension))
        .transpose()
}
fn parse_state_bank(text: &str, dimension: usize) -> Result<Vec<f64>, Box<dyn Error>> {
    let values: Vec<f64> = text
        .split_whitespace()
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    if dimension == 0
        || values.is_empty()
        || !values.len().is_multiple_of(dimension)
        || values.iter().any(|v| !v.is_finite())
    {
        return Err("state bank requires complete finite coordinate rows".into());
    }
    Ok(values)
}

#[derive(Default)]
struct Metrics {
    calls: usize,
    evaluation_ns: u128,
    construction_ns: u128,
    errors: usize,
    first_error: Option<String>,
}
struct Observed<T> {
    target: T,
    metrics: Arc<Mutex<Metrics>>,
    timed: bool,
}
impl<T: TargetSpace> TargetSpace for Observed<T> {
    type Error = T::Error;
    fn dimension(&self) -> usize {
        self.target.dimension()
    }
}
impl<T: LogDensityGradient> LogDensityGradient for Observed<T> {
    #[cfg(feature = "classified")]
    fn is_numerical_error(&self, error: &Self::Error) -> bool {
        self.target.is_numerical_error(error)
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Self::Error> {
        let start = self.timed.then(Instant::now);
        let result = self.target.logp_grad(q, g);
        let elapsed = start.map_or(0, |s| s.elapsed().as_nanos());
        let mut m = self.metrics.lock().expect("metrics lock");
        m.calls += 1;
        m.evaluation_ns += elapsed;
        if let Err(e) = &result {
            m.errors += 1;
            if m.first_error.is_none() {
                m.first_error = Some(format!("call={} q={q:?} error={e:?}", m.calls));
            }
        }
        result
    }
}

#[derive(Clone)]
struct Input {
    n: usize,
    k: usize,
    x: Vec<f64>,
    y: Vec<f64>,
}

fn parse_input(path: &str) -> Result<Input, Box<dyn Error>> {
    let content = fs::read_to_string(path)?;
    let mut tokens = content.split_whitespace();
    let n: usize = tokens.next().ok_or("missing N")?.parse()?;
    let k: usize = tokens.next().ok_or("missing K")?.parse()?;
    let mut x = Vec::with_capacity(n.checked_mul(k).ok_or("N*K overflow")?);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        for _ in 0..k {
            let value: f64 = tokens.next().ok_or("missing X")?.parse()?;
            if !value.is_finite() {
                return Err("nonfinite X".into());
            }
            x.push(value);
        }
        let value: f64 = tokens.next().ok_or("missing y")?.parse()?;
        if !value.is_finite() {
            return Err("nonfinite y".into());
        }
        y.push(value);
    }
    if tokens.next().is_some() {
        return Err("extra input tokens".into());
    }
    Ok(Input { n, k, x, y })
}

fn build_model(name: &str, input: &Input) -> Result<CompiledModel, AssemblyError> {
    let execution = regression_execution(env::var("ALEA_BENCH_REGRESSION").ok().as_deref())?;
    build_model_with_execution(name, input, execution)
}

fn build_model_with_execution(
    name: &str,
    input: &Input,
    execution: RegressionExecution,
) -> Result<CompiledModel, AssemblyError> {
    if matches!(name, "schools" | "schools_nc" | "covariance") {
        return conformance::build(name, input.n, input.k, &input.x, &input.y);
    }
    let mut b = ModelBuilder::new();
    let zero = b.constant(0.0)?;
    let one = b.constant(1.0)?;
    let two = b.constant(2.0)?;

    match name {
        "mean" | "linear" | "logistic" => {
            let gaussian = name != "logistic";
            let columns = if name == "mean" { 1 } else { input.k };
            let mut coefficients = Vec::with_capacity(columns);
            for j in 0..columns {
                let label = if name == "mean" {
                    "mu".to_owned()
                } else {
                    format!("beta_{}", j + 1)
                };
                let beta = b.real(&label)?;
                let prior = normal(&mut b, beta.expr(), zero, two)?;
                b.prior(&format!("{label} prior"), prior)?;
                coefficients.push(beta.expr());
            }
            if gaussian {
                // Alea's native normal_regression is a fused batch with a positive
                // residual SD parameter. Stan has precisely the same free parameter.
                let sigma = b.positive("sigma")?;
                let sigma_prior = normal(&mut b, sigma.expr(), zero, one)?;
                b.prior("sigma prior", sigma_prior)?;
                let design = if name == "mean" {
                    vec![1.0; input.n]
                } else {
                    input.x.clone()
                };
                let data = RegressionData::new(columns, design, input.y.clone())
                    .map_err(|_| AssemblyError::Shape)?;
                let likelihood = b.primitive(
                    &[&coefficients[..], &[sigma.expr()]].concat(),
                    NormalRegression::new(data)?.with_execution(execution),
                )?;
                b.likelihood("normal observations", likelihood)?;
            } else {
                let observations = input.y.iter().map(|&y| y == 1.0).collect();
                let data = RegressionData::new(columns, input.x.clone(), observations)
                    .map_err(|_| AssemblyError::Shape)?;
                let likelihood = b.primitive(
                    &coefficients,
                    BernoulliLogitRegression::new(data)?.with_execution(execution),
                )?;
                b.likelihood("bernoulli observations", likelihood)?;
            }
        }
        "funnel" => {
            // Centered Neal funnel: v ~ N(0,3); z_j | v ~ N(0,exp(v/2)).
            // Normal scores and chain-rule derivatives are supplied by Alea.
            let v = b.real("v")?;
            let three = b.constant(3.0)?;
            let v_prior = normal(&mut b, v.expr(), zero, three)?;
            b.prior("v prior", v_prior)?;
            let half = b.constant(0.5)?;
            let log_sigma = b.multiply(v.expr(), half)?;
            let conditional_sigma = b.exp(log_sigma)?;
            for j in 0..input.k {
                let label = format!("z_{}", j + 1);
                let z = b.real(&label)?;
                let factor = normal(&mut b, z.expr(), zero, conditional_sigma)?;
                b.prior(&format!("{label} conditional"), factor)?;
            }
        }
        _ => return Err(AssemblyError::Shape),
    }
    b.compile()
}

fn regression_execution(value: Option<&str>) -> Result<RegressionExecution, AssemblyError> {
    match value {
        None | Some("auto") => Ok(RegressionExecution::Auto),
        Some("scalar") => Ok(RegressionExecution::Scalar),
        _ => Err(AssemblyError::Shape),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let process_start = Instant::now();
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        return Err("usage: alea-stan-bench <mean|linear|logistic|funnel|schools|schools_nc|covariance> <input.txt>".into());
    }
    let name = args[1].as_str();
    if ![
        "mean",
        "linear",
        "logistic",
        "funnel",
        "schools",
        "schools_nc",
        "covariance",
    ]
    .contains(&name)
    {
        return Err("unknown model".into());
    }
    let input = parse_input(&args[2])?;
    if (name == "funnel" && (input.n != 0 || input.k == 0))
        || (name == "mean" && (input.n == 0 || input.k != 0))
        || ((name == "linear" || name == "logistic") && (input.n == 0 || input.k == 0))
    {
        return Err("invalid dimensions for model".into());
    }
    if name == "logistic" && input.y.iter().any(|&y| y != 0.0 && y != 1.0) {
        return Err("logistic responses must be 0 or 1".into());
    }

    let dimension = match name {
        "mean" => 2,             // mu, log(sigma)
        "linear" => input.k + 1, // coefficients, log(sigma)
        "logistic" => input.k,
        "funnel" => input.k + 1,
        "schools" | "schools_nc" => input.n + 2,
        "covariance" => 5,
        _ => unreachable!(),
    };
    if env::var("ALEA_BENCH_BACKEND").as_deref() == Ok("enzyme") {
        #[cfg(feature = "enzyme")]
        {
            let (x, y) = conformance::covariance_data();
            if name != "covariance" || input.n != 32 || input.k != 2 || input.x != x || input.y != y
            {
                return Err("unsupported source model or changed frozen binding (schools Cauchy lacks a source rule)".into());
            }
            return evaluation_protocol(&generated_covariance::target(), dimension);
        }
        #[cfg(not(feature = "enzyme"))]
        return Err("Enzyme backend was not compiled".into());
    }
    if let Ok(path) = env::var("ALEA_BENCH_PROBES") {
        let probes: Vec<Vec<f64>> = serde_json::from_str(&fs::read_to_string(path)?)?;
        let target = build_model(name, &input)?
            .select_posterior()?
            .into_target()?;
        let mut records = Vec::new();
        for q in probes {
            let mut g = vec![0.; dimension];
            match alea::target::evaluate(&target, &q, &mut g) {
                Ok(value) => records.push(serde_json::json!({"q":q,"logp":value,"gradient":g})),
                Err(error) => records.push(serde_json::json!({"q":q,"error":format!("{error:?}")})),
            }
        }
        println!("{}", serde_json::json!(records));
        return Ok(());
    }
    if let Ok(paths) = env::var("ALEA_BENCH_STAN_CSVS") {
        let mut chains = Vec::new();
        for path in paths.split(':') {
            let text = fs::read_to_string(path)?;
            let mut rows = text
                .lines()
                .filter(|l| !l.starts_with('#') && !l.is_empty());
            let header: Vec<_> = rows
                .next()
                .ok_or("missing CSV header")?
                .split(',')
                .collect();
            let indices: Vec<_> = header
                .iter()
                .enumerate()
                .filter_map(|(i, s)| (!s.ends_with("__")).then_some(i))
                .collect();
            let mut values = vec![Vec::new(); indices.len()];
            for row in rows {
                let fields: Vec<_> = row.split(',').collect();
                for (column, &i) in values.iter_mut().zip(&indices) {
                    let value = fields[i].parse::<f64>()?;
                    column.push(if ["sigma", "tau", "s0", "s1"].contains(&header[i]) {
                        value.ln()
                    } else {
                        value
                    });
                }
            }
            chains.push(values);
        }
        let mut diagnostics = Vec::new();
        for j in 0..dimension {
            let refs: Vec<_> = chains.iter().map(|c| c[j].as_slice()).collect();
            let s = alea::diagnostics::modern::summarize(&refs)?;
            diagnostics.push(
                serde_json::json!({"rhat":s.rhat,"bulk_ess":s.bulk_ess,"tail_ess":s.tail_ess,"mean":s.mean,"mcse_mean":s.mcse_mean,"mean_ess":s.mean_ess}),
            );
        }
        let constrained_diagnostics = constrained_summaries(name, dimension, &chains);
        println!(
            "{}",
            serde_json::json!({"diagnostics":diagnostics,"constrained_diagnostics":constrained_diagnostics})
        );
        return Ok(());
    }
    if env::var_os("ALEA_BENCH_KERNEL").is_some() {
        return kernel_profile(name, &input, dimension);
    }
    if let Ok(count) = env::var("ALEA_BENCH_EVALUATIONS") {
        let count: usize = count.parse()?;
        if count == 0 {
            return Err("positive evaluations required".into());
        }
        let start = Instant::now();
        let states = state_bank(dimension)?;
        let target = build_model(name, &input)?
            .select_posterior()?
            .into_target()?;
        let construction = start.elapsed().as_secs_f64();
        let mut q = vec![0.; dimension];
        let mut g = vec![0.; dimension];
        let start = Instant::now();
        let mut checksum = 0.;
        let mut gradient_checksum = 0.;
        let mut value = 0.;
        for i in 0..count {
            if let Some(states) = &states {
                let offset = (i % (states.len() / dimension)) * dimension;
                q.copy_from_slice(&states[offset..offset + dimension]);
            } else {
                q[0] = (i % 10) as f64 * 0.01;
            }
            value = std::hint::black_box(target.logp_grad(std::hint::black_box(&q), &mut g)?);
            checksum += value;
            for (j, score) in g.iter().enumerate() {
                gradient_checksum += (j + 1) as f64 * score;
            }
            std::hint::black_box(&g);
        }
        println!(
            "{}",
            serde_json::json!({"evaluations":count,"evaluation_seconds":start.elapsed().as_secs_f64(),"construction_seconds":construction,"checksum":checksum,"gradient_checksum":gradient_checksum,"logp":value,"gradient":g,"q":q})
        );
        return Ok(());
    }
    let chains: usize = env::var("ALEA_BENCH_CHAINS")
        .ok()
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(1);
    let timed = env::var_os("ALEA_BENCH_TIMED").is_some();
    let chain_threads: usize = env::var("ALEA_BENCH_CHAIN_THREADS")
        .ok()
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(1);
    if chain_threads == 0 || chain_threads > chains || chain_threads > 16 {
        return Err("chain threads must be positive, no greater than chains or the session thread budget (16)".into());
    }
    let pool = (chain_threads > 1)
        .then(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(chain_threads)
                .build()
        })
        .transpose()?;
    let target_acceptance =
        acceptance_option(env::var("ALEA_BENCH_TARGET_ACCEPT").ok().as_deref())?;
    let metrics: Vec<_> = (0..chains)
        .map(|_| Arc::new(Mutex::new(Metrics::default())))
        .collect();
    let factory_metrics = metrics.clone();
    let session = SessionBuilder::new(
        format!("alea-stan-native:{name}:{}:{}", input.n, input.k),
        dimension,
        vec![vec![0.0; dimension]; chains],
    )
    .seed([17; 32])
    .draws(1_000)
    .sampler(SamplerConfig::Nuts(NutsOptions::new(0.2, 10)?))
    .warmup(Warmup::Covariance {
        iterations: 1_000,
        metric: MetricKind::Diagonal,
        target: target_acceptance.try_into()?,
        search: None,
    })
    .build()?;

    // Alea's session factory takes an independent target per chain.
    // All datasets and factors use the public model API; no custom scores.
    let outcome = session.run(
        move |chain| {
            let start = Instant::now();
            let model = build_model(name, &input).expect("validated benchmark model must assemble");
            let model = model
                .select_posterior()
                .expect("named prior/likelihood factors");
            let target = model.into_target().expect("valid parameter layout");
            let metrics = &factory_metrics[usize::try_from(chain.0).expect("bounded chain index")];
            metrics.lock().expect("metrics lock").construction_ns += start.elapsed().as_nanos();
            Ok(Observed {
                target,
                metrics: Arc::clone(metrics),
                timed,
            })
        },
        pool.as_ref()
            .map_or(Execution::Sequential, Execution::Parallel),
        &Discard,
    )?;
    let serialization = Instant::now();
    if let Ok(path) = env::var("ALEA_BENCH_CSV") {
        use std::io::{BufWriter, Write};
        let file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        let mut writer = BufWriter::new(file);
        writeln!(
            writer,
            "chain,draw,{}",
            (0..dimension)
                .map(|j| format!("q{j}"))
                .collect::<Vec<_>>()
                .join(",")
        )?;
        for chain in &outcome.chains {
            for draw in &chain.draws {
                write!(writer, "{},{}", chain.chain.0, draw.index)?;
                for q in &draw.position {
                    write!(writer, ",{q:.17e}")?;
                }
                writeln!(writer)?;
            }
        }
        writer.flush()?;
    }
    let serialization_seconds = serialization.elapsed().as_secs_f64();
    let elapsed = process_start.elapsed().as_secs_f64();
    let mut steps = 0;
    let mut depths = Vec::new();
    let mut divergences = 0;
    let mut step_sizes = Vec::new();
    let mut failures = Vec::new();
    use std::hash::{Hash, Hasher};
    let mut trajectory_hash = std::collections::hash_map::DefaultHasher::new();
    for chain in &outcome.chains {
        if let Some(failure) = &chain.failure {
            failures.push(format!("{failure:?}"));
        }
        for draw in &chain.draws {
            chain.chain.0.hash(&mut trajectory_hash);
            draw.index.hash(&mut trajectory_hash);
            for q in &draw.position {
                q.to_bits().hash(&mut trajectory_hash);
            }
            steps += draw.stats.integration_steps.unwrap_or(0);
            if let Some(depth) = draw.stats.tree_depth {
                depths.push(depth);
            }
            divergences += usize::from(draw.stats.divergent.unwrap_or(false));
        }
        if let Some(checkpoint) = &chain.checkpoint
            && let SamplerConfig::Nuts(options) = checkpoint.sampler()
        {
            step_sizes.push(options.step_size().value());
        }
    }
    let summaries: Vec<_> = (0..dimension).map(|j| {
        let values: Vec<Vec<f64>> = outcome.chains.iter().map(|c| c.draws.iter().map(|d| d.position[j]).collect()).collect();
        let refs: Vec<&[f64]> = values.iter().map(Vec::as_slice).collect();
        match alea::diagnostics::modern::summarize(&refs) {
            Ok(s) => serde_json::json!({"rhat":s.rhat,"bulk_ess":s.bulk_ess,"tail_ess":s.tail_ess,"mean":s.mean,"mcse_mean":s.mcse_mean,"mean_ess":s.mean_ess,"bulk_ess_per_process_second":s.bulk_ess/elapsed}),
            Err(e) => serde_json::json!({"error":format!("{e:?}")}),
        }
    }).collect();
    let coordinates: Vec<Vec<Vec<f64>>> = outcome
        .chains
        .iter()
        .map(|c| {
            (0..dimension)
                .map(|j| c.draws.iter().map(|d| d.position[j]).collect())
                .collect()
        })
        .collect();
    let constrained_diagnostics = constrained_summaries(name, dimension, &coordinates);
    let mut m = Metrics::default();
    for metrics in &metrics {
        let chain = metrics.lock().expect("metrics lock");
        m.calls += chain.calls;
        m.evaluation_ns += chain.evaluation_ns;
        m.construction_ns += chain.construction_ns;
        m.errors += chain.errors;
        if m.first_error.is_none() {
            m.first_error = chain.first_error.clone();
        }
    }
    let completed: usize = outcome.chains.iter().map(|c| c.completed).sum();
    println!(
        "{}",
        serde_json::json!({
            "completed":completed,"chains":chains,"chain_threads":chain_threads,"dimension":dimension,"status":format!("{:?}",outcome.status),"target_acceptance":target_acceptance,
        "serialization_seconds":serialization_seconds,
        "trajectory_hash":format!("{:016x}",trajectory_hash.finish()),
            "failures":failures,"seconds":elapsed,"construction_seconds":m.construction_ns as f64/1e9,
            "evaluations":m.calls,"evaluation_seconds":timed.then_some(m.evaluation_ns as f64/1e9),
            "evaluation_errors":m.errors,"first_error":m.first_error,"retained_leapfrog_steps":steps,
            "retained_mean_depth": if depths.is_empty(){0.} else {depths.iter().sum::<usize>() as f64/depths.len() as f64},
            "retained_max_depth":depths.iter().max(),"retained_divergences":divergences,
            "final_step_sizes":step_sizes,"diagnostics":summaries,"constrained_diagnostics":constrained_diagnostics,
            "warmup_completed":outcome.chains.iter().map(|c|c.warmup_completed).collect::<Vec<_>>()
        })
    );
    if !failures.is_empty() || completed != chains * 1_000 {
        return Err("incomplete run (see JSON record)".into());
    }
    Ok(())
}

fn acceptance_option(text: Option<&str>) -> Result<f64, Box<dyn Error>> {
    let value = text.map(str::parse::<f64>).transpose()?.unwrap_or(0.8);
    if !value.is_finite() || value <= 0. || value >= 1. {
        return Err("target acceptance must be finite and strictly between zero and one".into());
    }
    Ok(value)
}

fn constrained_summaries(
    name: &str,
    dimension: usize,
    chains: &[Vec<Vec<f64>>],
) -> Vec<serde_json::Value> {
    (0..dimension).map(|j| {
        let positive = match name {
            "mean" | "linear" => j+1==dimension,
            "schools" | "schools_nc" => j+1==dimension,
            "covariance" => j==2 || j==3,
            _ => false,
        };
        let values:Vec<Vec<f64>> = chains.iter().map(|c|c[j].iter().map(|&x|if positive {x.exp()} else {x}).collect()).collect();
        let refs:Vec<_> = values.iter().map(Vec::as_slice).collect();
        match alea::diagnostics::modern::summarize(&refs) {
            Ok(s) => serde_json::json!({"rhat":s.rhat,"bulk_ess":s.bulk_ess,"tail_ess":s.tail_ess,"mean":s.mean,"mcse_mean":s.mcse_mean}),
            Err(e) => serde_json::json!({"error":format!("{e:?}")}),
        }
    }).collect()
}

#[cfg(feature = "enzyme")]
fn evaluation_protocol<T: LogDensityGradient>(
    target: &T,
    dimension: usize,
) -> Result<(), Box<dyn Error>> {
    if let Ok(path) = env::var("ALEA_BENCH_PROBES") {
        let points: Vec<Vec<f64>> = serde_json::from_str(&fs::read_to_string(path)?)?;
        let records: Vec<_> = points
            .into_iter()
            .map(|q| {
                let mut g = vec![0.; dimension];
                match alea::target::evaluate(target, &q, &mut g) {
                    Ok(logp) => serde_json::json!({"q":q,"logp":logp,"gradient":g}),
                    Err(error) => serde_json::json!({"q":q,"error":format!("{error:?}")}),
                }
            })
            .collect();
        println!("{}", serde_json::json!(records));
        return Ok(());
    }
    let calls: usize = env::var("ALEA_BENCH_EVALUATIONS")
        .map_err(|_| "source backend currently supports evaluation, not this sampling workflow")?
        .parse()?;
    if calls == 0 {
        return Err("positive evaluations required".into());
    }
    let mut q = vec![0.; dimension];
    let states = state_bank(dimension)?;
    let mut g = vec![0.; dimension];
    let mut value = 0.;
    let mut checksum = 0.;
    let mut gradient_checksum = 0.;
    let start = Instant::now();
    for i in 0..calls {
        if let Some(states) = &states {
            let offset = (i % (states.len() / dimension)) * dimension;
            q.copy_from_slice(&states[offset..offset + dimension]);
        } else {
            q[0] = (i % 10) as f64 * 0.01;
        }
        value = target.logp_grad(std::hint::black_box(&q), &mut g)?;
        checksum += std::hint::black_box(value);
        gradient_checksum += g
            .iter()
            .enumerate()
            .map(|(j, g)| (j + 1) as f64 * g)
            .sum::<f64>();
        std::hint::black_box(&g);
    }
    println!(
        "{}",
        serde_json::json!({"evaluations":calls,"evaluation_seconds":start.elapsed().as_secs_f64(),"construction_seconds":0.,"construction_scope":"frozen source target; preparation charged at build time","logp":value,"gradient":g,"checksum":checksum,"gradient_checksum":gradient_checksum})
    );
    Ok(())
}

// A diagnostic control using the SAME public native target, seed/stream and
// native covariance warmup as SessionBuilder. No probability or NUTS code here.
fn kernel_profile(name: &str, input: &Input, dimension: usize) -> Result<(), Box<dyn Error>> {
    use alea::{
        math::buffer::OwnedBuffer,
        output::ChainId,
        sampling::adapt::{NutsWarmup, WarmupOptions},
    };
    let metrics = Arc::new(Mutex::new(Metrics::default()));
    let target = Observed {
        target: build_model(name, input)?
            .select_posterior()?
            .into_target()?,
        metrics: Arc::clone(&metrics),
        timed: true,
    };
    let mut rng = alea::ensemble::chain_rng([17; 32], ChainId(0));
    let warmup = NutsWarmup::new(
        &target,
        OwnedBuffer::new(dimension),
        NutsOptions::new(0.2, 10)?,
        WarmupOptions::new(1000)
            .with_target(
                acceptance_option(env::var("ALEA_BENCH_TARGET_ACCEPT").ok().as_deref())?
                    .try_into()?,
            )
            .with_search(None),
    )?;
    let start = Instant::now();
    let mut depths = Vec::new();
    let (mut chain, report) = warmup.run_observed(&mut rng, |observation| {
        depths.push(observation.transition.tree_depth)
    })?;
    let warmup_seconds = start.elapsed().as_secs_f64();
    let warmup_evaluations = metrics.lock().expect("metrics lock").calls;
    let start = Instant::now();
    let mut steps = 0;
    let mut divergences = 0;
    for _ in 0..1000 {
        let info = chain.step(&mut rng)?;
        steps += info.integration_steps;
        divergences += usize::from(matches!(
            info.termination,
            alea::sampling::nuts::NutsTermination::Divergence(_)
        ));
    }
    let sampling_seconds = start.elapsed().as_secs_f64();
    let m = metrics.lock().expect("metrics lock");
    println!(
        "{}",
        serde_json::json!({"warmup_report":format!("{report:?}"),"warmup_seconds":warmup_seconds,"sampling_seconds":sampling_seconds,"warmup_evaluations":warmup_evaluations,"sampling_evaluations":m.calls-warmup_evaluations,"evaluation_seconds":m.evaluation_ns as f64/1e9,"evaluation_errors":m.errors,"first_error":m.first_error,"warmup_mean_depth":depths.iter().sum::<usize>() as f64/depths.len() as f64,"warmup_max_depth":depths.iter().max(),"retained_leapfrog_steps":steps,"retained_divergences":divergences,"final_position":chain.point().position()})
    );
    Ok(())
}
