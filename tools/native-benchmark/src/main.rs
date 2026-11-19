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
use alea::model::{
    ConstrainedModel,
    assembly::{AssemblyError, ModelBuilder, NativeModel},
};
use alea::sampling::{NutsOptions, adapt::MetricKind};
use alea::target::{LogDensityGradient, TargetSpace};
use alea::{
    ensemble::Execution,
    session::{SamplerConfig, SessionBuilder, Warmup},
};
use std::{
    env,
    error::Error,
    fs,
    hash::{Hash, Hasher},
    sync::{Arc, Mutex},
    time::Instant,
};

#[cfg(test)]
use alea::session::storage::Discard;
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

fn build_model(name: &str, input: &Input) -> Result<NativeModel, AssemblyError> {
    let execution = regression_execution(env::var("ALEA_BENCH_REGRESSION").ok().as_deref())?;
    build_model_with_execution(name, input, execution)
}

fn build_model_with_execution(
    name: &str,
    input: &Input,
    execution: RegressionExecution,
) -> Result<NativeModel, AssemblyError> {
    if matches!(name, "schools" | "schools_nc" | "covariance") {
        return if composition_blocks(env::var("ALEA_BENCH_REPRESENTATION").ok().as_deref())? {
            conformance::native(name, input.n, input.k, &input.x, &input.y)
        } else {
            conformance::build(name, input.n, input.k, &input.x, &input.y)
        };
    }
    if env::var_os("ALEA_BENCH_REPRESENTATION").is_some() {
        return Err(AssemblyError::Shape);
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

fn composition_blocks(value: Option<&str>) -> Result<bool, AssemblyError> {
    match value {
        None | Some("block") => Ok(true),
        Some("scalar") => Ok(false),
        _ => Err(AssemblyError::Shape),
    }
}

fn probe_model(model: NativeModel, path: &str) -> Result<(), Box<dyn Error>> {
    let probes: Vec<Vec<f64>> = serde_json::from_str(&fs::read_to_string(path)?)?;
    let target = model.select_posterior()?.into_target()?;
    let mut records = Vec::new();
    for q in probes {
        let mut g = vec![0.; target.dimension()];
        match alea::target::evaluate(&target, &q, &mut g) {
            Ok(value) => records.push(serde_json::json!({"q":q,"logp":value,"gradient":g})),
            Err(error) => records.push(serde_json::json!({"q":q,"error":format!("{error:?}")})),
        }
    }
    println!("{}", serde_json::json!(records));
    Ok(())
}

struct RadonInput {
    j: usize,
    county: Vec<usize>,
    floor: Vec<f64>,
    y: Vec<f64>,
}
impl RadonInput {
    fn parse(data: &serde_json::Value) -> Result<Self, Box<dyn Error>> {
        let count = |name: &str| {
            data[name]
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or("invalid Radon dimension")
        };
        let j = count("J")?;
        let n = count("N")?;
        let county = data["county_idx"]
            .as_array()
            .ok_or("missing county indices")?
            .iter()
            .map(|i| {
                i.as_u64()
                    .and_then(|i| usize::try_from(i).ok())
                    .and_then(|i| i.checked_sub(1))
                    .ok_or("invalid one-based county index")
            })
            .collect::<Result<Vec<_>, _>>()?;
        let values = |name: &str| -> Result<Vec<f64>, Box<dyn Error>> {
            data[name]
                .as_array()
                .ok_or("missing Radon observations")?
                .iter()
                .map(|v| v.as_f64().ok_or_else(|| "invalid Radon observation".into()))
                .collect()
        };
        let floor = values("floor_measure")?;
        let y = values("log_radon")?;
        if y.len() != n {
            return Err("Radon N/observation mismatch".into());
        }
        Ok(Self {
            j,
            county,
            floor,
            y,
        })
    }
    fn build(&self) -> Result<NativeModel, AssemblyError> {
        if composition_blocks(env::var("ALEA_BENCH_REPRESENTATION").ok().as_deref())? {
            conformance::radon(self.j, &self.county, &self.floor, &self.y)
        } else {
            conformance::radon_scalar(self.j, &self.county, &self.floor, &self.y)
        }
    }
}
#[cfg(test)]
fn radon_input(data: &serde_json::Value) -> Result<NativeModel, Box<dyn Error>> {
    Ok(RadonInput::parse(data)?.build()?)
}

fn main() -> Result<(), Box<dyn Error>> {
    let process_start = Instant::now();
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        return Err("usage: alea-stan-bench <mean|linear|logistic|funnel|schools|schools_nc|covariance|radon> <input.txt>".into());
    }
    let name = args[1].as_str();
    if name == "radon" {
        if env::var("ALEA_BENCH_BACKEND").as_deref() == Ok("enzyme") {
            return Err("Radon block source export is unsupported".into());
        }
        let data = serde_json::from_str(&fs::read_to_string(&args[2])?)?;
        let input = RadonInput::parse(&data)?;
        let admitted = input.build()?;
        let dimension = admitted.dimension();
        if let Ok(path) = env::var("ALEA_BENCH_PROBES") {
            return probe_model(admitted, &path);
        }
        if let Ok(paths) = env::var("ALEA_BENCH_STAN_CSVS") {
            return stan_diagnostics(name, dimension, &paths);
        }
        if env::var_os("ALEA_BENCH_EVALUATIONS").is_some() {
            return Err("Radon fixed-work evaluation is unsupported by this CLI".into());
        }
        if env::var_os("ALEA_BENCH_KERNEL").is_some() {
            return kernel_profile(name, dimension, || input.build());
        }
        return sample_model(
            name,
            dimension,
            format!("radon:{}:{}", input.j, input.y.len()),
            process_start,
            || input.build(),
        );
    }
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
        if let Ok(representation) = env::var("ALEA_BENCH_REPRESENTATION")
            && composition_blocks(Some(&representation))?
        {
            return Err(
                "block source export is unsupported; retained source target is scalar".into(),
            );
        }
        #[cfg(feature = "enzyme")]
        {
            let (x, y) = conformance::covariance_data();
            if name != "covariance" || input.n != 32 || input.k != 2 || input.x != x || input.y != y
            {
                return Err("unsupported source model or changed frozen binding (schools Cauchy lacks a source rule)".into());
            }
            return evaluation_protocol(generated_covariance::target, dimension);
        }
        #[cfg(not(feature = "enzyme"))]
        return Err("Enzyme backend was not compiled".into());
    }
    if let Ok(path) = env::var("ALEA_BENCH_PROBES") {
        return probe_model(build_model(name, &input)?, &path);
    }
    if let Ok(paths) = env::var("ALEA_BENCH_STAN_CSVS") {
        return stan_diagnostics(name, dimension, &paths);
    }
    if env::var_os("ALEA_BENCH_KERNEL").is_some() {
        return kernel_profile(name, dimension, || build_model(name, &input));
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
    sample_model(
        name,
        dimension,
        format!("alea-stan-native:{name}:{}:{}", input.n, input.k),
        process_start,
        || build_model(name, &input),
    )
}

fn sample_model(
    name: &str,
    dimension: usize,
    identity: String,
    process_start: Instant,
    build: impl Fn() -> Result<NativeModel, AssemblyError> + Sync,
) -> Result<(), Box<dyn Error>> {
    let output_mode = sampling_output(
        env::var("ALEA_BENCH_OUTPUT").ok().as_deref(),
        env::var_os("ALEA_BENCH_CSV").is_some(),
    )?;
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
    let session = SessionBuilder::new(identity, dimension, vec![vec![0.0; dimension]; chains])
        .seed([seed_option()?; 32])
        .draws(1_000)
        .collect(output_mode != "discard")
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
    let evidence: Vec<_> = (0..chains)
        .map(|_| Mutex::new(RetainedEvidence::default()))
        .collect();
    let observer = alea::session::storage::Observer(|draw: &alea::session::OwnedDraw| {
        evidence[draw.chain.0 as usize]
            .lock()
            .expect("evidence lock")
            .record(draw);
        Ok(())
    });
    let sampling_start = Instant::now();
    let outcome = session.run(
        move |chain| {
            let start = Instant::now();
            let model = build().expect("validated benchmark model must assemble");
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
        &observer,
    )?;
    let session_seconds = sampling_start.elapsed().as_secs_f64();
    let evidence: Vec<_> = evidence
        .into_iter()
        .map(|e| e.into_inner().expect("evidence lock"))
        .collect();
    let energy_diagnostics: Vec<_> = evidence.iter().map(RetainedEvidence::energy).collect();
    let depth_saturations = if evidence.iter().any(|e| e.missing_depth) {
        None
    } else {
        Some(evidence.iter().map(|e| e.saturated).sum::<usize>())
    };
    let output_hashes: Vec<_> = evidence
        .iter()
        .map(|e| format!("{:016x}", e.hash.finish()))
        .collect();
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
    if output_mode == "discard" {
        steps = evidence.iter().map(|e| e.steps).sum();
        divergences = evidence.iter().map(|e| e.divergences).sum();
        depths = evidence
            .iter()
            .flat_map(|e| e.depths.iter().copied())
            .collect();
    }
    let completed: usize = outcome.chains.iter().map(|c| c.completed).sum();
    println!(
        "{}",
        serde_json::json!({
            "completed":completed,"chains":chains,"chain_threads":chain_threads,"dimension":dimension,"status":format!("{:?}",outcome.status),"target_acceptance":target_acceptance,
        "serialization_seconds":serialization_seconds,"session_seconds":session_seconds,"output_mode":output_mode,"output_scope":"discard disables retained-position collection; energy evidence and bounded runtime diagnostics remain","output_trajectory_hashes":output_hashes,"energy_diagnostics":energy_diagnostics,"retained_depth_saturations":depth_saturations,"scientific_quantities":if output_mode=="discard" {vec![]} else {scientific_summaries(name,dimension,&coordinates)},"final_positions":outcome.chains.iter().map(|c|c.checkpoint.as_ref().map(|p|p.position())).collect::<Vec<_>>(),
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
            "radon" => j+2>=dimension,
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
    factory: impl FnOnce() -> T,
    dimension: usize,
) -> Result<(), Box<dyn Error>> {
    if let Ok(path) = env::var("ALEA_BENCH_PROBES") {
        let target = factory();
        let points: Vec<Vec<f64>> = serde_json::from_str(&fs::read_to_string(path)?)?;
        let records: Vec<_> = points
            .into_iter()
            .map(|q| {
                let mut g = vec![0.; dimension];
                match alea::target::evaluate(&target, &q, &mut g) {
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
    let setup = Instant::now();
    let states = state_bank(dimension)?;
    let target = factory();
    let construction = setup.elapsed().as_secs_f64();
    let mut q = vec![0.; dimension];
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
        serde_json::json!({"evaluations":calls,"evaluation_seconds":start.elapsed().as_secs_f64(),"construction_seconds":construction,"construction_scope":"state bank loading and source target factory; excludes source generation and compilation","logp":value,"gradient":g,"checksum":checksum,"gradient_checksum":gradient_checksum})
    );
    Ok(())
}

// A diagnostic control using the SAME public native target, seed/stream and
// native covariance warmup as SessionBuilder. No probability or NUTS code here.
fn kernel_profile(
    name: &str,
    dimension: usize,
    build: impl Fn() -> Result<NativeModel, AssemblyError>,
) -> Result<(), Box<dyn Error>> {
    use alea::{
        math::buffer::OwnedBuffer,
        output::ChainId,
        sampling::adapt::{NutsWarmup, WarmupOptions},
    };
    let metrics = Arc::new(Mutex::new(Metrics::default()));
    let target = Observed {
        target: build()?.select_posterior()?.into_target()?,
        metrics: Arc::clone(&metrics),
        timed: true,
    };
    let chain_id: u64 = env::var("ALEA_BENCH_CHAIN")
        .ok()
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(0);
    if chain_id >= 16 {
        return Err("logical chain must be below16".into());
    }
    let mut rng = alea::ensemble::chain_rng([seed_option()?; 32], ChainId(chain_id));
    let start = Instant::now();
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
        serde_json::json!({"logical_chain":chain_id,"case":name,"phase_scope":"separate logical-chain diagnostic; initialization counted with warmup","warmup_report":format!("{report:?}"),"warmup_seconds":warmup_seconds,"sampling_seconds":sampling_seconds,"warmup_evaluations":warmup_evaluations,"sampling_evaluations":m.calls-warmup_evaluations,"evaluation_seconds":m.evaluation_ns as f64/1e9,"evaluation_errors":m.errors,"first_error":m.first_error,"warmup_mean_depth":depths.iter().sum::<usize>() as f64/depths.len() as f64,"warmup_max_depth":depths.iter().max(),"retained_leapfrog_steps":steps,"retained_divergences":divergences,"final_position":chain.point().position()})
    );
    Ok(())
}

fn seed_option() -> Result<u8, Box<dyn Error>> {
    Ok(env::var("ALEA_BENCH_SEED")
        .ok()
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(17))
}

fn sampling_output(value: Option<&str>, csv: bool) -> Result<&'static str, &'static str> {
    match (value, csv) {
        (None, false) | (Some("memory"), false) => Ok("memory"),
        (None | Some("csv"), true) => Ok("csv"),
        (Some("discard"), false) => Ok("discard"),
        _ => Err("output must be memory/discard without CSV, or csv with a create-new CSV path"),
    }
}

#[derive(Default)]
struct RetainedEvidence {
    hash: std::collections::hash_map::DefaultHasher,
    energies: Vec<f64>,
    missing_energy: bool,
    missing_depth: bool,
    steps: usize,
    divergences: usize,
    saturated: usize,
    depths: Vec<usize>,
}
impl RetainedEvidence {
    fn record(&mut self, draw: &alea::session::OwnedDraw) {
        draw.chain.0.hash(&mut self.hash);
        draw.index.hash(&mut self.hash);
        for q in &draw.position {
            q.to_bits().hash(&mut self.hash);
        }
        if let Some(e) = draw.stats.energy {
            self.energies.push(e);
        } else {
            self.missing_energy = true;
        }
        if let Some(d) = draw.stats.tree_depth {
            self.depths.push(d);
        }
        if let Some(s) = draw.stats.depth_saturated {
            self.saturated += usize::from(s);
        } else {
            self.missing_depth = true;
        }
        self.steps += draw.stats.integration_steps.unwrap_or(0);
        self.divergences += usize::from(draw.stats.divergent.unwrap_or(false));
    }
    fn energy(&self) -> serde_json::Value {
        if self.missing_energy {
            return serde_json::json!({"error":"missing transition energy"});
        }
        match alea::diagnostics::modern::ebfmi(&self.energies) {
            Ok(value) => serde_json::json!({"ebfmi":value,"transitions":self.energies.len()}),
            Err(error) => serde_json::json!({"error":format!("{error:?}")}),
        }
    }
}

// Reuse the production rank/folded ESS/MCSE routines on explicit scientific
// observables. Squares are second-moment estimands, not variance MCSE estimates.
fn scientific_summaries(
    name: &str,
    dimension: usize,
    chains: &[Vec<Vec<f64>>],
) -> Vec<serde_json::Value> {
    let positive = |j| match name {
        "mean" | "linear" | "schools" | "schools_nc" => j + 1 == dimension,
        "covariance" => j == 2 || j == 3,
        "radon" => j + 2 >= dimension,
        _ => false,
    };
    let labels: Vec<String> = match name {
        "schools" | "schools_nc" => (0..dimension)
            .map(|j| {
                if j + 2 < dimension {
                    format!("theta[{}]", j + 1)
                } else if j + 1 == dimension {
                    "tau".into()
                } else {
                    "mu".into()
                }
            })
            .collect(),
        "radon" => (0..dimension)
            .map(|j| {
                if j + 4 < dimension {
                    format!("alpha[{}]", j + 1)
                } else {
                    ["beta", "mu_alpha", "sigma_alpha", "sigma_y"][j + 4 - dimension].into()
                }
            })
            .collect(),
        "covariance" => ["mu0", "mu1", "s0", "s1", "loading"]
            .map(String::from)
            .to_vec(),
        _ => (0..dimension).map(|j| format!("parameter[{j}]")).collect(),
    };
    let series: Vec<Vec<Vec<f64>>> = (0..dimension)
        .map(|j| {
            chains
                .iter()
                .map(|c| {
                    c[j].iter()
                        .enumerate()
                        .map(|(i, &q)| {
                            if name == "schools_nc" && j + 2 < dimension {
                                c[dimension - 2][i] + c[dimension - 1][i].exp() * q
                            } else if name == "radon" && j + 4 < dimension {
                                c[dimension - 3][i] + c[dimension - 2][i].exp() * q
                            } else if positive(j) {
                                q.exp()
                            } else {
                                q
                            }
                        })
                        .collect()
                })
                .collect()
        })
        .collect();
    let mut observables = Vec::new();
    for (j, values) in series.iter().enumerate() {
        observables.push((labels[j].clone(), "mean", Some(j), values.clone()));
        observables.push((
            format!("{}:second_moment", labels[j]),
            "second_moment",
            Some(j),
            values
                .iter()
                .map(|c| c.iter().map(|v| v * v).collect())
                .collect(),
        ));
    }
    if name == "covariance" {
        for (label, kind) in [
            ("cov[0,0]", 0),
            ("cov[0,1]", 1),
            ("cov[1,1]", 2),
            ("correlation", 3),
        ] {
            let values = series[0]
                .iter()
                .enumerate()
                .map(|(c, x)| {
                    (0..x.len())
                        .map(|i| {
                            let (s0, s1, l) = (series[2][c][i], series[3][c][i], series[4][c][i]);
                            match kind {
                                0 => s0 * s0,
                                1 => s0 * l,
                                2 => l * l + s1 * s1,
                                _ => l / l.hypot(s1),
                            }
                        })
                        .collect()
                })
                .collect();
            observables.push((label.into(), "derived", None, values));
        }
    }
    observables.into_iter().map(|(name,kind,coordinate,values)| {
        let refs:Vec<_>=values.iter().map(Vec::as_slice).collect();
        match alea::diagnostics::modern::summarize(&refs) {
            Ok(s)=>serde_json::json!({"name":name,"kind":kind,"coordinate":coordinate,"mean":s.mean,"mcse_mean":s.mcse_mean,"mean_ess":s.mean_ess,"rhat":s.rhat,"rank_rhat":s.rank_rhat,"folded_rhat":s.folded_rhat,"bulk_ess":s.bulk_ess,"tail_ess":s.tail_ess}),
            Err(e)=>serde_json::json!({"name":name,"kind":kind,"coordinate":coordinate,"error":format!("{e:?}")}),
        }
    }).collect()
}

fn stan_diagnostics(name: &str, dimension: usize, paths: &str) -> Result<(), Box<dyn Error>> {
    let names: Option<Vec<String>> = match name {
        "covariance" => Some(
            ["mu0", "mu1", "s0", "s1", "loading"]
                .map(String::from)
                .to_vec(),
        ),
        "schools" | "schools_nc" => Some(
            (1..=dimension - 2)
                .map(|i| format!("effect.{i}"))
                .chain(["mu".into(), "tau".into()])
                .collect(),
        ),
        "radon" => Some(
            (1..=dimension - 4)
                .map(|i| format!("alpha_raw.{i}"))
                .chain(["beta", "mu_alpha", "sigma_alpha", "sigma_y"].map(String::from))
                .collect(),
        ),
        _ => None,
    };
    let mut chains = Vec::new();
    let mut energies = Vec::new();
    let mut saturated = Some(0_usize);
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
        let indices = if let Some(names) = &names {
            names
                .iter()
                .map(|n| {
                    header
                        .iter()
                        .position(|h| *h == n)
                        .ok_or("missing named Stan parameter")
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            header
                .iter()
                .enumerate()
                .filter_map(|(i, h)| (!h.ends_with("__")).then_some(i))
                .take(dimension)
                .collect()
        };
        if indices.len() != dimension {
            return Err("CSV coordinate dimension mismatch".into());
        }
        let e = header.iter().position(|h| *h == "energy__");
        let depth = header.iter().position(|h| *h == "treedepth__");
        let mut values = vec![Vec::new(); dimension];
        let mut energy = Vec::new();
        for row in rows {
            let fields = row.split(',').collect::<Vec<_>>();
            if fields.len() != header.len() {
                return Err("CSV row dimension mismatch".into());
            }
            for (j, &index) in indices.iter().enumerate() {
                let value = fields[index].parse::<f64>()?;
                let positive = match name {
                    "mean" | "linear" | "schools" | "schools_nc" => j + 1 == dimension,
                    "covariance" => j == 2 || j == 3,
                    "radon" => j + 2 >= dimension,
                    _ => false,
                };
                values[j].push(if positive { value.ln() } else { value });
            }
            if let Some(i) = e {
                energy.push(fields[i].parse::<f64>()?);
            }
            if let (Some(i), Some(count)) = (depth, &mut saturated) {
                *count += usize::from(fields[i].parse::<usize>()? >= 10);
            } else {
                saturated = None;
            }
        }
        let refs = energy.as_slice();
        energies.push(match alea::diagnostics::modern::ebfmi(refs) {
            Ok(v) => serde_json::json!({"ebfmi":v,"transitions":energy.len()}),
            Err(err) => serde_json::json!({"error":format!("{err:?}")}),
        });
        chains.push(values);
    }
    let diagnostics:Vec<_>=(0..dimension).map(|j| {
        let refs:Vec<_>=chains.iter().map(|c|c[j].as_slice()).collect();
        match alea::diagnostics::modern::summarize(&refs) {Ok(s)=>serde_json::json!({"rhat":s.rhat,"bulk_ess":s.bulk_ess,"tail_ess":s.tail_ess,"mean":s.mean,"mcse_mean":s.mcse_mean,"mean_ess":s.mean_ess}),Err(e)=>serde_json::json!({"error":format!("{e:?}")})}
    }).collect();
    println!(
        "{}",
        serde_json::json!({"diagnostics":diagnostics,"constrained_diagnostics":constrained_summaries(name,dimension,&chains),"scientific_quantities":scientific_summaries(name,dimension,&chains),"energy_diagnostics":energies,"retained_depth_saturations":saturated})
    );
    Ok(())
}
