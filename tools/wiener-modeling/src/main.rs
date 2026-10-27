//! Complete native-model fits; datasets and measurements are supplied externally.
use alea::{
    diagnostics::modern,
    distributions::{
        modeling::normal,
        wiener::{
            Boundary, ProbError, Wiener4, Wiener4Params, Wiener5, Wiener5Params, Wiener7,
            Wiener7Params, WienerObservation, WienerOptions,
        },
    },
    ensemble::Execution,
    model::{
        ModelError, TransformedTarget,
        assembly::{AssemblyError, CompiledModel, Expr, ModelBuilder},
    },
    sampling::{
        NutsOptions,
        adapt::{MetricKind, SearchOptions},
    },
    session::{Diagnostics, Kernel, SessionBuilder, Warmup, storage::Observer},
    target::{EvaluationError, LogDensityGradient},
};
use rand::{SeedableRng, rngs::SmallRng};
use serde::{Deserialize, Serialize};
use std::{
    error::Error,
    fs,
    io::{BufWriter, Write},
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};
type AnyError = Box<dyn Error>;
#[derive(Clone, Deserialize)]
struct Config {
    name: String,
    data: String,
    family: u8,
    hierarchy: bool,
    #[serde(default)]
    subject_scales: bool,
    time_only: bool,
    precision: f64,
    warmup: usize,
    draws: usize,
    seed: u8,
    threads: usize,
    seconds: u64,
    target_acceptance: Option<f64>,
    initial_step: Option<f64>,
    #[serde(default)]
    step_search: bool,
    maximum_step: Option<f64>,
}
#[derive(Clone)]
struct Trial {
    subject: usize,
    condition: f64,
    observation: WienerObservation,
}
fn load(path: &str) -> Result<Vec<Trial>, AnyError> {
    let text = fs::read_to_string(path)?;
    text.lines()
        .skip(1)
        .map(|line| {
            let x: Vec<_> = line.split(',').collect();
            if x.len() != 4 {
                return Err("expected subject,condition,rt,response".into());
            }
            if x[3].parse::<u8>()? > 1 {
                return Err("response must be zero or one".into());
            }
            Ok(Trial {
                subject: x[0].parse()?,
                condition: x[1].parse()?,
                observation: WienerObservation {
                    rt: x[2].parse()?,
                    boundary: if x[3].parse::<u8>()? == 1 {
                        Boundary::Upper
                    } else {
                        Boundary::Lower
                    },
                },
            })
        })
        .collect()
}
fn sigmoid(x: f64) -> f64 {
    if x >= 0. {
        1. / (1. + (-x).exp())
    } else {
        x.exp() / (1. + x.exp())
    }
}
fn logistic(b: &mut ModelBuilder, x: Expr, scale: f64) -> Result<Expr, AssemblyError> {
    b.custom(&[x], move |x, g| {
        let p = sigmoid(x[0]);
        g[0] = scale * p * (1. - p);
        Ok(scale * p)
    })
}
fn raw(b: &mut ModelBuilder, name: &str, mu: f64, sd: f64) -> Result<Expr, AssemblyError> {
    let q = b.real(name)?.expr();
    let m = b.constant(mu)?;
    let s = b.constant(sd)?;
    let prior = normal(b, q, m, s)?;
    b.factor(prior)?;
    Ok(q)
}
fn prior_layout(c: &Config, minimum: f64, subjects: usize) -> Vec<(String, f64, f64)> {
    let mut p = vec![
        ("log_alpha".into(), 1.5_f64.ln(), 0.5),
        ("tau_logit".into(), 0., 1.),
        ("beta_logit".into(), 0., 0.8),
        ("mean_drift".into(), 0., 1.5),
        ("conflict_effect".into(), 0., 1.),
    ];
    if c.family >= 5 && !c.time_only {
        p.push(("log_trial_drift_sd".into(), 0.5_f64.ln(), 0.7));
    }
    if c.family == 7 && !c.time_only {
        p.push(("start_width_logit".into(), (1_f64 / 3.).ln(), 1.));
    }
    if c.family == 7 {
        let ratio = (0.08 / (0.8 * minimum)).clamp(0.05, 0.8);
        p.push(("time_width_logit".into(), (ratio / (1. - ratio)).ln(), 1.));
    }
    if c.hierarchy {
        p.push(("log_subject_sd".into(), 0.6_f64.ln(), 0.5));
        for i in 0..subjects {
            p.push((format!("subject_z_{i}"), 0., 1.));
        }
    }
    if c.subject_scales {
        p.push(("log_alpha_subject_sd".into(), 0.3_f64.ln(), 0.5));
        for i in 0..subjects {
            p.push((format!("alpha_z_{i}"), 0., 1.));
        }
        p.push(("log_tau_subject_sd".into(), 0.5_f64.ln(), 0.5));
        for i in 0..subjects {
            p.push((format!("tau_z_{i}"), 0., 1.));
        }
    }
    p
}
fn subject_minima(data: &[Trial], subjects: usize) -> Vec<f64> {
    let mut values = vec![f64::INFINITY; subjects];
    for row in data {
        values[row.subject] = values[row.subject].min(row.observation.rt);
    }
    values
}
fn build(c: &Config, data: &[Trial]) -> Result<CompiledModel, AssemblyError> {
    let mut b = ModelBuilder::new();
    let minimum = data
        .iter()
        .map(|x| x.observation.rt)
        .fold(f64::INFINITY, f64::min);
    let subjects = 1 + data.iter().map(|x| x.subject).max().unwrap();
    let layout = prior_layout(c, minimum, subjects);
    let q: Vec<_> = layout
        .iter()
        .map(|(name, m, s)| raw(&mut b, name, *m, *s))
        .collect::<Result<_, _>>()?;
    let alpha = b.exp(q[0])?;
    let tau = logistic(&mut b, q[1], minimum)?;
    let center = logistic(&mut b, q[2], 1.)?;
    let mut next = 5;
    let zero = b.constant(0.)?;
    let sd = if c.family >= 5 && !c.time_only {
        let x = b.exp(q[next])?;
        next += 1;
        x
    } else {
        zero
    };
    let width = if c.family == 7 && !c.time_only {
        let x = logistic(&mut b, q[next], 0.8)?;
        next += 1;
        x
    } else {
        zero
    };
    let half = b.constant(0.5)?;
    let halfwidth = b.multiply(half, width)?;
    let negative = b.constant(-1.)?;
    let negwidth = b.multiply(negative, width)?;
    let one = b.constant(1.)?;
    let span = b.add(one, negwidth)?;
    let position = b.multiply(span, center)?;
    let beta = b.add(halfwidth, position)?;
    let st = if c.family == 7 {
        let x = logistic(&mut b, q[next], 0.8 * minimum)?;
        next += 1;
        x
    } else {
        zero
    };
    let mut drifts = vec![q[3]; subjects];
    if c.hierarchy {
        let scale = b.exp(q[next])?;
        next += 1;
        for (i, drift) in drifts.iter_mut().enumerate() {
            let deviation = b.multiply(scale, q[next + i])?;
            *drift = b.add(q[3], deviation)?;
        }
    }
    next += if c.hierarchy { subjects } else { 0 };
    let mut separations = vec![alpha; subjects];
    let mut delays = vec![tau; subjects];
    if c.subject_scales {
        let sigma = b.exp(q[next])?;
        next += 1;
        for (i, value) in separations.iter_mut().enumerate() {
            let delta = b.multiply(sigma, q[next + i])?;
            let log = b.add(q[0], delta)?;
            *value = b.exp(log)?;
        }
        next += subjects;
        let sigma = b.exp(q[next])?;
        next += 1;
        for (i, minimum) in subject_minima(data, subjects).into_iter().enumerate() {
            let delta = b.multiply(sigma, q[next + i])?;
            let logit = b.add(q[1], delta)?;
            delays[i] = logistic(&mut b, logit, minimum)?;
        }
    }
    let options = WienerOptions::new(c.precision);
    for trial in data {
        let condition = b.constant(trial.condition)?;
        let effect = b.multiply(condition, q[4])?;
        let drift = b.add(drifts[trial.subject], effect)?;
        let event = trial.observation;
        let family = c.family;
        let likelihood = b.custom(
            &[
                separations[trial.subject],
                delays[trial.subject],
                beta,
                drift,
                sd,
                width,
                st,
            ],
            move |x, g| {
                g.fill(0.);
                let result: Result<f64, ProbError> = match family {
                    4 => Wiener4
                        .try_fused(
                            &event,
                            &Wiener4Params::from_array([x[0], x[1], x[2], x[3]]),
                            options,
                        )
                        .map(|e| {
                            g[..4].copy_from_slice(&e.grad_array());
                            e.log_prob
                        }),
                    5 => Wiener5
                        .try_fused(
                            &event,
                            &Wiener5Params::from_array([x[0], x[1], x[2], x[3], x[4]]),
                            options,
                        )
                        .map(|e| {
                            g[..5].copy_from_slice(&e.grad_array());
                            e.log_prob
                        }),
                    _ => Wiener7Params::with_params(x[0], x[1], x[2], x[3], x[5], x[6], x[4])
                        .and_then(|p| Wiener7.try_fused(&event, &p, options))
                        .map(|e| {
                            g.copy_from_slice(&e.grad_array());
                            e.log_prob
                        }),
                };
                result.map_err(|e| {
                    eprintln!("failed_likelihood event={event:?} parameters={x:?} error={e:?}");
                    AssemblyError::Factor(Box::new(e))
                })
            },
        )?;
        b.factor(likelihood)?;
    }
    b.compile()
}
struct Counted {
    inner: TransformedTarget<CompiledModel>,
    counter: Arc<AtomicU64>,
    errors: Arc<AtomicU64>,
}
impl LogDensityGradient for Counted {
    type Error = ModelError<AssemblyError>;
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }
    fn logp_grad(&self, q: &[f64], g: &mut [f64]) -> Result<f64, Self::Error> {
        self.counter.fetch_add(1, Ordering::Relaxed);
        let r = self.inner.logp_grad(q, g);
        if r.is_err() {
            self.errors.fetch_add(1, Ordering::Relaxed);
            eprintln!("failed_target position={q:?} error={r:?}");
        }
        r
    }
}
/// Deterministic physical estimands, including the modeled condition contrast.
fn physical(c: &Config, q: &[f64], minimum: f64, subjects: usize, minima: &[f64]) -> Vec<f64> {
    let mut v = vec![
        q[0].exp(),
        minimum * sigmoid(q[1]),
        sigmoid(q[2]),
        q[3],
        q[4],
    ];
    let mut next = 5;
    let sd = if c.family >= 5 && !c.time_only {
        let x = q[next].exp();
        next += 1;
        x
    } else {
        0.
    };
    let sw = if c.family == 7 && !c.time_only {
        let x = 0.8 * sigmoid(q[next]);
        next += 1;
        x
    } else {
        0.
    };
    let st = if c.family == 7 {
        let x = 0.8 * minimum * sigmoid(q[next]);
        next += 1;
        x
    } else {
        0.
    };
    v[2] = sw / 2. + (1. - sw) * v[2];
    v.extend([sd, sw, st]);
    if c.hierarchy {
        let scale = q[next].exp();
        next += 1;
        v.push(scale);
        for i in 0..subjects {
            v.push(q[3] + scale * q[next + i]);
        }
    }
    next += if c.hierarchy { subjects } else { 0 };
    if c.subject_scales {
        v[1] = sigmoid(q[1]);
        let scale = q[next].exp();
        next += 1;
        v.push(scale);
        v.extend((0..subjects).map(|i| (q[0] + scale * q[next + i]).exp()));
        next += subjects;
        let scale = q[next].exp();
        next += 1;
        v.push(scale);
        v.extend((0..subjects).map(|i| minima[i] * sigmoid(q[1] + scale * q[next + i])));
    }
    v
}
#[derive(Serialize)]
struct Scalar {
    name: String,
    mean: f64,
    sd: f64,
    q025: f64,
    q50: f64,
    q975: f64,
    rhat: Option<f64>,
    bulk_ess: Option<f64>,
    tail_ess: Option<f64>,
    mcse: Option<f64>,
}
fn fit(c: &Config, data: &[Trial], out: &Path) -> Result<(), AnyError> {
    let config = c;
    if out.exists() {
        return Err("output directory already exists; keep original runs".into());
    }
    fs::create_dir_all(out)?;
    let all_start = Instant::now();
    let minimum = data
        .iter()
        .map(|x| x.observation.rt)
        .fold(f64::INFINITY, f64::min);
    let subjects = 1 + data.iter().map(|x| x.subject).max().unwrap();
    let minima = subject_minima(data, subjects);
    let layout = prior_layout(c, minimum, subjects);
    let dimension = layout.len();
    let starts: Vec<Vec<_>> = (0..4)
        .map(|chain| {
            layout
                .iter()
                .enumerate()
                .map(|(j, (_, mu, _))| mu + 0.15 * ((chain * dimension + j + 1) as f64).sin())
                .collect()
        })
        .collect();
    let counters: Vec<_> = (0..4).map(|_| Arc::new(AtomicU64::new(0))).collect();
    let errors: Vec<_> = (0..4).map(|_| Arc::new(AtomicU64::new(0))).collect();
    let first = std::sync::Mutex::new(vec![None; 4]);
    let run_start = Instant::now();
    let sink = Observer(|draw: &alea::session::OwnedDraw| {
        if draw.index == 0 {
            first.lock().unwrap()[draw.chain.0 as usize] = Some(run_start.elapsed().as_secs_f64());
        }
        if draw.index.is_multiple_of(100) {
            eprintln!(
                "chain={} retained={} target_evaluations={} seconds={:.3}",
                draw.chain.0,
                draw.index + 1,
                counters[draw.chain.0 as usize].load(Ordering::Relaxed),
                run_start.elapsed().as_secs_f64()
            );
        }
        Ok(())
    });
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(c.threads)
        .build()?;
    let result = SessionBuilder::new(&c.name, dimension, starts)
        .seed([c.seed; 32])
        .draws(c.draws)
        .kernel(Kernel::Nuts(NutsOptions::new(
            c.initial_step.unwrap_or(0.05),
            9,
        )?))
        .warmup(Warmup::Covariance {
            iterations: c.warmup,
            metric: MetricKind::Dense,
            target: c.target_acceptance.unwrap_or(0.9).try_into()?,
            search: if c.step_search {
                Some(SearchOptions::new(
                    1e-8,
                    c.maximum_step.unwrap_or(1e6),
                    80,
                    0.8,
                )?)
            } else {
                None
            },
        })
        .diagnostics(Diagnostics {
            window: c.draws,
            maximum_rhat: 1.01,
            minimum_ess: 400.,
        })
        .wall_time(Duration::from_secs(c.seconds))
        .build()?
        .run(
            |id| {
                let inner = build(c, data)
                    .map_err(|e| ModelError::Evaluation(EvaluationError::Model(e)))?
                    .into_target()?;
                Ok(Counted {
                    inner,
                    counter: Arc::clone(&counters[id.0 as usize]),
                    errors: Arc::clone(&errors[id.0 as usize]),
                })
            },
            Execution::Parallel(&pool),
            &sink,
        )?;
    let inference_seconds = run_start.elapsed().as_secs_f64();
    let save_start = Instant::now();
    let mut raw = BufWriter::new(fs::File::create(out.join("draws.csv"))?);
    writeln!(
        raw,
        "chain,draw,divergent,depth_saturated,steps,energy,acceptance,{}",
        layout
            .iter()
            .map(|x| x.0.as_str())
            .collect::<Vec<_>>()
            .join(",")
    )?;
    for chain in &result.chains {
        for d in &chain.draws {
            write!(
                raw,
                "{},{},{},{},{},{},{:0.17}",
                chain.chain.0,
                d.index,
                d.stats.divergent.unwrap_or(false),
                d.stats.depth_saturated.unwrap_or(false),
                d.stats.integration_steps.unwrap_or(0),
                d.stats.energy.unwrap_or(f64::NAN),
                d.stats.acceptance_probability
            )?;
            for x in &d.position {
                write!(raw, ",{x:0.17}")?;
            }
            writeln!(raw)?;
        }
    }
    raw.flush()?;
    let output_seconds = save_start.elapsed().as_secs_f64();
    let mut names = vec![
        "alpha",
        "tau",
        "beta",
        "population_drift",
        "conflict_effect",
        "trial_drift_sd",
        "start_width",
        "time_width",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<Vec<_>>();
    if c.hierarchy {
        names.push("subject_sd".into());
        names.extend((0..subjects).map(|i| format!("subject_drift_{i}")));
    }
    if c.subject_scales {
        names[0] = "population_median_alpha".into();
        names[1] = "population_tau_fraction".into();
        names.push("alpha_subject_sd".into());
        names.extend((0..subjects).map(|i| format!("subject_alpha_{i}")));
        names.push("tau_subject_sd".into());
        names.extend((0..subjects).map(|i| format!("subject_tau_{i}")));
    }
    let mut summaries = Vec::new();
    let mut checks = true;
    for (i, name) in names.into_iter().enumerate() {
        let chains: Vec<Vec<f64>> = result
            .chains
            .iter()
            .map(|c| {
                c.draws
                    .iter()
                    .map(|d| physical(config, &d.position, minimum, subjects, &minima)[i])
                    .collect()
            })
            .collect();
        let refs: Vec<_> = chains.iter().map(Vec::as_slice).collect();
        let summary = modern::summarize(&refs).ok();
        let mut values: Vec<f64> = chains.iter().flatten().copied().collect();
        values.sort_by(f64::total_cmp);
        if values.is_empty() {
            continue;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let sd = (values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / (values.len().max(2) - 1) as f64)
            .sqrt();
        if sd == 0. {
            continue;
        }
        checks &= summary
            .as_ref()
            .is_some_and(|s| s.rhat < 1.01 && s.bulk_ess >= 400. && s.tail_ess >= 400.);
        summaries.push(Scalar {
            name,
            mean,
            sd,
            q025: values[(0.025 * (values.len() - 1) as f64) as usize],
            q50: values[values.len() / 2],
            q975: values[(0.975 * (values.len() - 1) as f64) as usize],
            rhat: summary.as_ref().map(|s| s.rhat),
            bulk_ess: summary.as_ref().map(|s| s.bulk_ess),
            tail_ess: summary.as_ref().map(|s| s.tail_ess),
            mcse: summary.as_ref().map(|s| s.mcse_mean),
        });
    }
    let chains:Vec<_>=result.chains.iter().enumerate().map(|(i,chain)|{
        let energies:Vec<_>=chain.draws.iter().filter_map(|d|d.stats.energy).collect();
        let mean=energies.iter().sum::<f64>()/energies.len().max(1)as f64;
        let variance=energies.iter().map(|e|(e-mean).powi(2)).sum::<f64>();let jump=energies.windows(2).map(|e|(e[1]-e[0]).powi(2)).sum::<f64>();
        serde_json::json!({"chain":i,"completed":chain.completed,"warmup_completed":chain.warmup_completed,"failure":chain.failure.as_ref().map(|e|format!("{e:?}")),"cancelled":chain.cancelled,"numerical_rejections":chain.numerical_rejections,"target_evaluations":counters[i].load(Ordering::Relaxed),"target_errors":errors[i].load(Ordering::Relaxed),"first_draw_seconds":first.lock().unwrap()[i],"divergences":chain.draws.iter().filter(|d|d.stats.divergent==Some(true)).count(),"depth_saturations":chain.draws.iter().filter(|d|d.stats.depth_saturated==Some(true)).count(),"ebfmi":if variance>0.{Some(jump/variance*energies.len()as f64/(energies.len()-1)as f64)}else{None}})
    }).collect();
    checks &= result.chains.iter().all(|chain| {
        chain.failure.is_none()
            && !chain.cancelled
            && chain.completed == c.draws
            && chain.numerical_rejections == 0
            && chain
                .draws
                .iter()
                .all(|d| d.stats.divergent != Some(true) && d.stats.depth_saturated != Some(true))
    });
    let result = serde_json::json!({"name":c.name,"observations":data.len(),"subjects":subjects,"dimension":dimension,"inference_seconds":inference_seconds,"output_seconds":output_seconds,"total_seconds":all_start.elapsed().as_secs_f64(),"diagnostics_pass":checks,"chains":chains,"parameters":summaries});
    fs::write(
        out.join("summary.json"),
        serde_json::to_string_pretty(&result)? + "\n",
    )?;
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
fn evaluate(
    c: &Config,
    data: &[Trial],
    positions: &[Vec<f64>],
) -> Result<serde_json::Value, AnyError> {
    let setup = Instant::now();
    let target = build(c, data)?.into_target()?;
    let setup_seconds = setup.elapsed().as_secs_f64();
    let minimum = data
        .iter()
        .map(|x| x.observation.rt)
        .fold(f64::INFINITY, f64::min);
    let subjects = 1 + data.iter().map(|x| x.subject).max().unwrap();
    let minima = subject_minima(data, subjects);
    let mut evaluations = Vec::new();
    for q in positions {
        if q.len() != target.dimension() || q.iter().any(|x| !x.is_finite()) {
            return Err("invalid position vector".into());
        }
        let mut gradient = vec![0.; target.dimension()];
        let start = Instant::now();
        let lp = target.logp_grad(q, &mut gradient)?;
        evaluations.push(serde_json::json!({"position":q,"logp":lp,"gradient":gradient,"seconds":start.elapsed().as_secs_f64(),"physical":physical(c,q,minimum,subjects,&minima)}));
    }
    Ok(serde_json::json!({"setup_seconds":setup_seconds,"evaluations":evaluations}))
}
fn predict(
    c: &Config,
    data: &[Trial],
    positions: &[Vec<f64>],
) -> Result<serde_json::Value, AnyError> {
    let minimum = data
        .iter()
        .map(|x| x.observation.rt)
        .fold(f64::INFINITY, f64::min);
    let subjects = 1 + data.iter().map(|x| x.subject).max().unwrap();
    let minima = subject_minima(data, subjects);
    let mut rng = SmallRng::seed_from_u64(2026102703);
    let mut reports = Vec::new();
    for (replicate, q) in positions.iter().enumerate() {
        let p = physical(c, q, minimum, subjects, &minima);
        let mut outcomes = [Vec::new(), Vec::new()];
        for trial in data {
            let drift = if c.hierarchy {
                p[9 + trial.subject]
            } else {
                p[3]
            };
            let offset = 9 + subjects;
            let a = if c.subject_scales {
                p[offset + 1 + trial.subject]
            } else {
                p[0]
            };
            let tau = if c.subject_scales {
                p[offset + subjects + 2 + trial.subject]
            } else {
                p[1]
            };
            let parameters = Wiener7Params::with_params(
                a,
                tau,
                p[2],
                drift + p[4] * trial.condition,
                p[6],
                p[7],
                p[5],
            )?;
            let observation = Wiener7.sample(&parameters, &mut rng)?;
            outcomes[usize::from(trial.condition > 0.)].push(observation);
        }
        for (condition, observations) in outcomes.into_iter().enumerate() {
            let mut times: Vec<_> = observations.iter().map(|x| x.rt).collect();
            times.sort_by(f64::total_cmp);
            if times.is_empty() {
                continue;
            }
            reports.push(serde_json::json!({"replicate":replicate,"condition":condition,"n":times.len(),
                "upper_proportion":observations.iter().filter(|x|x.boundary==Boundary::Upper).count() as f64/times.len() as f64,
                "rt_q10":times[((times.len()-1)as f64*0.1)as usize],"rt_q50":times[times.len()/2],"rt_q90":times[((times.len()-1)as f64*0.9)as usize]}));
        }
    }
    Ok(serde_json::json!(reports))
}
fn generate(path: &Path, family: u8) -> Result<(), AnyError> {
    let mut rng = SmallRng::seed_from_u64(2026102701 + u64::from(family));
    let mut file = BufWriter::new(fs::File::create(path)?);
    writeln!(file, "subject,condition,rt,response")?;
    let subjects = if family == 4 { 8 } else { 1 };
    let trials = if family == 4 { 80 } else { 160 };
    for subject in 0..subjects {
        for trial in 0..trials {
            let condition = if trial % 2 == 0 { -0.5 } else { 0.5 };
            let drift = 0.6
                + 0.8 * condition
                + if family == 4 {
                    0.5 * (subject as f64 - 3.5) / 3.5
                } else {
                    0.
                };
            let p = Wiener7Params::with_params(
                1.5,
                0.25,
                0.5,
                drift,
                if family == 7 { 0.3 } else { 0. },
                if family == 7 { 0.1 } else { 0. },
                if family == 7 { 0.6 } else { 0. },
            )?;
            let event = Wiener7.sample(&p, &mut rng)?;
            writeln!(
                file,
                "{subject},{condition},{:0.17},{}",
                event.rt,
                u8::from(event.boundary == Boundary::Upper)
            )?;
        }
    }
    file.flush()?;
    Ok(())
}
fn main() -> Result<(), AnyError> {
    let args: Vec<_> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("generate") {
        return generate(Path::new(&args[2]), args[3].parse()?);
    }
    if args.len() < 4 {
        return Err(
            "usage: fit CONFIG OUTPUT | evaluate CONFIG POSITIONS OUTPUT | generate CSV FAMILY"
                .into(),
        );
    }
    let c: Config = serde_json::from_str(&fs::read_to_string(&args[2])?)?;
    let data = load(&c.data)?;
    if c.subject_scales && !c.hierarchy {
        return Err("subject_scales requires a hierarchy".into());
    }
    if data.is_empty()
        || data.iter().any(|x| {
            !x.observation.rt.is_finite() || x.observation.rt <= 0. || !x.condition.is_finite()
        })
    {
        return Err("invalid or empty observations".into());
    }
    match args[1].as_str() {
        "fit" => fit(&c, &data, Path::new(&args[3])),
        "evaluate" | "predict" => {
            let points: Vec<Vec<f64>> = serde_json::from_str(&fs::read_to_string(&args[3])?)?;
            let result = if args[1] == "predict" {
                predict(&c, &data, &points)?
            } else {
                evaluate(&c, &data, &points)?
            };
            fs::write(&args[4], serde_json::to_string_pretty(&result)? + "\n")?;
            Ok(())
        }
        _ => Err("unknown mode".into()),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_native_model_gradient_matches_density_differences() -> Result<(), AnyError> {
        let data: Vec<_> = (0..8)
            .map(|i| Trial {
                subject: i % 2,
                condition: if i % 2 == 0 { -0.5 } else { 0.5 },
                observation: WienerObservation {
                    rt: 0.55 + 0.1 * i as f64,
                    boundary: if i % 3 == 0 {
                        Boundary::Lower
                    } else {
                        Boundary::Upper
                    },
                },
            })
            .collect();
        for (family, time_only, hierarchy, subject_scales) in [
            (4, false, true, false),
            (4, false, true, true),
            (5, false, true, false),
            (7, true, false, false),
            (7, false, false, false),
        ] {
            let c = Config {
                name: "test".into(),
                data: String::new(),
                family,
                hierarchy,
                subject_scales,
                time_only,
                precision: 1e-8,
                warmup: 100,
                draws: 100,
                seed: 1,
                threads: 1,
                seconds: 60,
                target_acceptance: None,
                initial_step: None,
                step_search: false,
                maximum_step: None,
            };
            let target = build(&c, &data)?.into_target()?;
            let q: Vec<_> = prior_layout(&c, 0.55, 2)
                .iter()
                .map(|(_, m, _)| *m)
                .collect();
            let mut gradient = vec![0.; q.len()];
            target.logp_grad(&q, &mut gradient)?;
            for i in 0..q.len() {
                let mut plus = q.clone();
                let mut minus = q.clone();
                plus[i] += 1e-5;
                minus[i] -= 1e-5;
                let mut scratch = gradient.clone();
                let fd = (target.logp_grad(&plus, &mut scratch)?
                    - target.logp_grad(&minus, &mut scratch)?)
                    / 2e-5;
                assert!(
                    (fd - gradient[i]).abs() < 2e-4 * (1. + fd.abs()),
                    "family {family}, coordinate {i}: {} vs {fd}",
                    gradient[i]
                );
            }
        }
        Ok(())
    }
}
