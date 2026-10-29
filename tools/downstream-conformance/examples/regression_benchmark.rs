//! Deterministic synthetic likelihood benchmark; timings are not statistical qualification.
use alea::{
    distributions::{
        modeling::{bernoulli_logit, normal_observation, observation, regression::*},
        scalar::Poisson,
    },
    model::{
        ComposedObjective,
        assembly::{CompiledModel, ModelBuilder},
    },
    optimize::{LbfgsOptions, ObjectiveDescription, ObjectiveKind, maximize},
    target::LogDensityGradient,
};
use alea_downstream_conformance::profiling::EvaluationObserver;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use std::{hint::black_box, io::Write, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<_> = std::env::args().collect();
    let reuse = args
        .iter()
        .position(|x| x == "--reuse-model")
        .map(|i| args.remove(i))
        .is_some();
    let fits = take_count(&mut args, "--fits")?;
    let threads = take_count(&mut args, "--threads")?;
    if let Some(fits) = fits {
        let threads = threads.ok_or("--fits requires --threads")?;
        println!("{}", repeated_fits(&args, fits, threads, reuse)?);
    } else {
        if threads.is_some() || reuse {
            return Err("--threads and --reuse-model require --fits".into());
        }
        println!("{}", benchmark(args)?);
    }
    Ok(())
}

fn take_count(
    args: &mut Vec<String>,
    flag: &str,
) -> Result<Option<usize>, Box<dyn std::error::Error>> {
    let Some(index) = args.iter().position(|x| x == flag) else {
        return Ok(None);
    };
    let count = args
        .get(index + 1)
        .ok_or("missing count")?
        .parse::<usize>()?;
    if count == 0 {
        return Err("counts must be positive".into());
    }
    args.drain(index..index + 2);
    Ok(Some(count))
}

fn repeated_fits(
    args: &[String],
    fits: usize,
    threads: usize,
    reuse: bool,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    use rayon::prelude::*;
    let started = Instant::now();
    // Only IDs are queued. Models and evaluator scratch are constructed inside
    // workers, so at most min(fits, threads) model workspaces are live.
    let pool = if threads == 1 {
        None
    } else {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()?,
        )
    };
    let pool_seconds = if pool.is_some() {
        started.elapsed().as_secs_f64()
    } else {
        0.
    };
    let started = Instant::now();
    let lanes = fits.min(threads);
    // Identical fixed partitions in cold and reuse modes. The objective is
    // constructed inside the lane, never shared or sent between workers.
    let run_lane = |lane: usize| {
        let base = fits / lanes;
        let extra = fits % lanes;
        let start = lane * base + lane.min(extra);
        let end = start + base + usize::from(lane < extra);
        let prepared = reuse.then(|| prepare(args.to_vec()));
        (start..end)
            .map(|id| {
                let result = match &prepared {
                    Some(Ok(Some(model))) => model.run(if id == start {
                        model.construction_seconds
                    } else {
                        0.
                    }),
                    Some(Ok(None)) => Err("expected model, got dataset export".into()),
                    Some(Err(error)) => Err(error.to_string().into()),
                    None => benchmark(args.to_vec()),
                };
                match result {
                    Ok(output) => match serde_json::from_str::<serde_json::Value>(&output) {
                        Ok(mut row) if row.is_object() => {
                            row["fit_id"] = id.into();
                            row
                        }
                        Ok(_) => serde_json::json!({"fit_id":id,"error":"expected fit record"}),
                        Err(error) => serde_json::json!({"fit_id":id,"error":error.to_string()}),
                    },
                    Err(error) => serde_json::json!({"fit_id":id,"error":error.to_string()}),
                }
            })
            .collect::<Vec<_>>()
    };
    let chunks: Vec<_> = match pool {
        Some(pool) => pool.install(|| (0..lanes).into_par_iter().map(run_lane).collect()),
        None => (0..lanes).map(run_lane).collect(),
    };
    let runs: Vec<_> = chunks.into_iter().flatten().collect();
    let workload_seconds = started.elapsed().as_secs_f64();
    Ok(
        serde_json::json!({"execution":if threads == 1 {"sequential"} else {"parallel"},"fits":fits,"threads":threads,"max_live_models":fits.min(threads),
        "reuse_model":reuse,"model_build_attempts":if reuse {lanes} else {fits},
        "pool_seconds":pool_seconds,"workload_seconds":workload_seconds,"runs":runs}),
    )
}

struct Prepared {
    objective: ComposedObjective<CompiledModel>,
    family: String,
    mode: String,
    profile: String,
    n: usize,
    columns: usize,
    repetitions: usize,
    expressions: usize,
    construction_seconds: f64,
    score_tolerance_per_observation: f64,
}

fn benchmark(args: Vec<String>) -> Result<String, Box<dyn std::error::Error>> {
    match prepare(args)? {
        Some(model) => model.run(model.construction_seconds),
        None => Ok("null".into()),
    }
}

fn prepare(mut args: Vec<String>) -> Result<Option<Prepared>, Box<dyn std::error::Error>> {
    let mut score_tolerance_per_observation = 1e-7_f64;
    if args.len() >= 3 && args[args.len() - 2] == "--score-tolerance-per-observation" {
        score_tolerance_per_observation = args.pop().ok_or("missing tolerance")?.parse()?;
        args.pop();
    }
    if !(5..=8).contains(&args.len()) {
        return Err(
            "usage: regression_benchmark normal|bernoulli|poisson batch|scalar rows repetitions [plain|counted|timed] [columns] [dataset.json] [--score-tolerance-per-observation value]"
                .into(),
        );
    }
    let profile = args.get(5).map_or("plain", String::as_str);
    let columns = args.get(6).map_or(Ok(2), |x| x.parse::<usize>())?;
    let family = args[1].as_str();
    let mode = args[2].as_str();
    let n = args[3].parse::<usize>()?;
    let repetitions = args[4].parse::<usize>()?;
    if !["normal", "bernoulli", "poisson"].contains(&family)
        || !["batch", "scalar"].contains(&mode)
        || !["plain", "counted", "timed"].contains(&profile)
        || !score_tolerance_per_observation.is_finite()
        || score_tolerance_per_observation <= 0.
        || columns == 0
        || n.checked_mul(columns).is_none()
        || n == 0
        || repetitions == 0
    {
        return Err("invalid benchmark configuration".into());
    }
    let started = Instant::now();
    let mut b = ModelBuilder::new();
    let mut beta = Vec::with_capacity(columns);
    for j in 0..columns {
        let name = match j {
            0 => "intercept".to_owned(),
            1 => "slope".to_owned(),
            _ => format!("coefficient{j}"),
        };
        beta.push(b.real(&name)?.expr());
    }
    let sigma = if family == "normal" {
        Some(b.positive("sigma")?)
    } else {
        None
    };
    let mut rng = SmallRng::seed_from_u64(20261029);
    let mut design = Vec::with_capacity(n * columns);
    for i in 0..n {
        for j in 0..columns {
            design.push(if j == 0 {
                1.
            } else if columns == 2 {
                (i % 101) as f64 / 50. - 1.
            } else {
                rng.random_range(-1.0..1.0)
            });
        }
    }
    let means: Vec<_> = design
        .chunks_exact(columns)
        .map(|row| {
            1. + row
                .iter()
                .enumerate()
                .skip(1)
                .map(|(j, x)| x * (0.5 / j as f64))
                .sum::<f64>()
        })
        .collect();
    let real: Vec<_> = (0..n)
        .map(|i| means[i] + ((i * 17 % 19) as f64 - 9.) / 10.)
        .collect();
    let binary: Vec<_> = (0..n)
        .map(|i| {
            (i * 19 % 31) as f64 / 31.
                < if columns == 2 {
                    0.5 + 0.2 * design[2 * i + 1]
                } else {
                    alea::math::numeric::sigmoid(means[i] - 1.)
                }
        })
        .collect();
    let counts: Vec<_> = (0..n).map(|i| (i * 13 % 7) as u64).collect();
    if let Some(path) = args.get(7) {
        let mut output = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        writeln!(
            output,
            "{{\"columns\":{columns},\"rows\":{n},\"design\":{design:?},\"normal\":{real:?},\"bernoulli\":{binary:?},\"poisson\":{counts:?}}}"
        )?;
        return Ok(None);
    }
    if mode == "batch" {
        let f = match family {
            "normal" => normal_regression(
                &mut b,
                &beta,
                sigma.as_ref().unwrap(),
                RegressionData::new(columns, design, real)?,
            )?,
            "bernoulli" => bernoulli_logit_regression(
                &mut b,
                &beta,
                RegressionData::new(columns, design, binary)?,
            )?,
            _ => poisson_log_regression(
                &mut b,
                &beta,
                RegressionData::new(columns, design, counts)?,
            )?,
        };
        b.factor(f)?;
    } else {
        for i in 0..n {
            let mut eta = beta[0];
            for j in 1..columns {
                let predictor = b.constant(design[columns * i + j])?;
                let effect = b.multiply(predictor, beta[j])?;
                eta = b.add(eta, effect)?;
            }
            let f = match family {
                "normal" => normal_observation(&mut b, real[i], eta, sigma.as_ref().unwrap())?,
                "bernoulli" => bernoulli_logit(&mut b, binary[i], eta)?,
                _ => {
                    let rate = b.exp(eta)?;
                    observation::<Poisson, _, _>(&mut b, counts[i], &[rate], |x| {
                        Poisson::new(x[0])
                    })?
                }
            };
            b.factor(f)?;
        }
    }
    // Every factor is an observation likelihood, with no priors/penalties. The
    // benchmark uses the generic sum for both paths to avoid per-row name lookup.
    let model = b.compile()?;
    let expressions = model.expression_count();
    let objective = model.into_objective()?;
    let construction_seconds = started.elapsed().as_secs_f64();
    Ok(Some(Prepared {
        objective,
        family: family.into(),
        mode: mode.into(),
        profile: profile.into(),
        n,
        columns,
        repetitions,
        expressions,
        construction_seconds,
        score_tolerance_per_observation,
    }))
}

impl Prepared {
    fn run(&self, construction_seconds: f64) -> Result<String, Box<dyn std::error::Error>> {
        let Self {
            n,
            columns,
            repetitions,
            expressions,
            score_tolerance_per_observation,
            ..
        } = *self;
        let family = self.family.as_str();
        let mode = self.mode.as_str();
        let profile = self.profile.as_str();
        let objective = &self.objective;
        let mut q: Vec<_> = (0..columns)
            .map(|j| if j == 0 { 0.1 } else { -0.2 / j as f64 })
            .collect();
        if family == "normal" {
            q.push(0.);
        }
        let mut g = vec![0.; q.len()];
        let value = objective.logp_grad(&q, &mut g)?;
        let gradient = g.clone();
        let started = Instant::now();
        for _ in 0..repetitions {
            black_box(objective.logp_grad(black_box(&q), black_box(&mut g))?);
        }
        let evaluation_seconds = started.elapsed().as_secs_f64() / repetitions as f64;
        let counted = EvaluationObserver::<_, false>::new(objective);
        let timed = EvaluationObserver::<_, true>::new(objective);
        let target: &dyn LogDensityGradient<Error = _> = match profile {
            "counted" => &counted,
            "timed" => &timed,
            _ => objective,
        };
        let started = Instant::now();
        let result = maximize(
            target,
            &q,
            ObjectiveDescription {
                kind: ObjectiveKind::Likelihood,
                measure: "original parameters".into(),
                model: family.into(),
                data: format!("synthetic-v2-{n}-{columns}"),
                source: "regression_benchmark-v1".into(),
            },
            LbfgsOptions::new(100, 1000)?
                .with_score_tolerance(n as f64 * score_tolerance_per_observation)?,
        )?;
        let fit_seconds = started.elapsed().as_secs_f64();
        let target_calls = match profile {
            "counted" => Some(counted.calls()),
            "timed" => Some(timed.calls()),
            _ => None,
        };
        if let Some(calls) = target_calls {
            assert_eq!(calls, result.evaluations);
        }
        let target_calls = target_calls.map_or("null".into(), |x| x.to_string());
        let target_seconds = timed
            .elapsed()
            .filter(|_| profile == "timed")
            .map_or("null".into(), |x| x.as_secs_f64().to_string());
        let rss = std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("VmHWM:"))
                    .and_then(|l| l.split_whitespace().nth(1))
                    .and_then(|x| x.parse::<u64>().ok())
            });
        let rss = rss.map_or("null".into(), |x| x.to_string());
        let estimates = result.candidate.as_ref().map(|c| c.position.as_slice());
        let final_gradient = &result
            .candidate
            .as_ref()
            .ok_or("no benchmark candidate")?
            .gradient;
        let final_value = result
            .candidate
            .as_ref()
            .ok_or("no benchmark candidate")?
            .value;
        let score_tolerance = result.options.score_tolerance();
        let line_search_failure = result.line_search_failure;
        Ok(format!(
            "{{\"columns\":{columns},\"profile\":\"{profile}\",\"target_calls\":{target_calls},\"target_seconds\":{target_seconds},\"family\":\"{family}\",\"mode\":\"{mode}\",\"rows\":{n},\"repetitions\":{repetitions},\"expressions\":{expressions},\"construction_seconds\":{construction_seconds},\"evaluation_seconds\":{evaluation_seconds},\"fit_seconds\":{fit_seconds},\"peak_rss_kib\":{rss},\"value\":{value},\"gradient\":{gradient:?},\"final_gradient\":{final_gradient:?},\"final_value\":{final_value},\"score_tolerance\":{score_tolerance},\"line_search_failure\":\"{line_search_failure:?}\",\"termination\":\"{:?}\",\"evaluations\":{},\"iterations\":{},\"estimate_free\":{:?}}}",
            result.termination,
            result.evaluations,
            result.iterations,
            estimates.ok_or("no benchmark estimate")?
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments(family: &str) -> Vec<String> {
        ["benchmark", family, "batch", "100", "1", "plain", "8"]
            .map(str::to_owned)
            .to_vec()
    }

    #[test]
    fn pool_size_preserves_fit_identity_and_numerical_work() {
        let sequential = repeated_fits(&arguments("normal"), 4, 1, false).unwrap();
        let parallel = repeated_fits(&arguments("normal"), 4, 2, false).unwrap();
        assert_eq!(parallel["max_live_models"], 2);
        for id in 0..4 {
            assert_eq!(parallel["runs"][id]["fit_id"], id);
            for key in [
                "value",
                "gradient",
                "final_value",
                "final_gradient",
                "estimate_free",
                "termination",
                "iterations",
                "evaluations",
                "score_tolerance",
            ] {
                assert_eq!(sequential["runs"][id][key], parallel["runs"][id][key]);
                assert!(!parallel["runs"][id][key].is_null());
            }
        }
    }

    #[test]
    fn failures_retain_every_input_id() {
        let result = repeated_fits(&arguments("invalid"), 5, 2, false).unwrap();
        assert_eq!(result["runs"].as_array().unwrap().len(), 5);
        for id in 0..5 {
            assert_eq!(result["runs"][id]["fit_id"], id);
            assert!(result["runs"][id]["error"].is_string());
        }
    }
    #[test]
    fn worker_local_models_preserve_fits_and_avoid_reconstruction() {
        for family in ["normal", "bernoulli", "poisson"] {
            for threads in [1, 2, 4] {
                let cold = repeated_fits(&arguments(family), 6, threads, false).unwrap();
                let reused = repeated_fits(&arguments(family), 6, threads, true).unwrap();
                assert_eq!(cold["model_build_attempts"], 6);
                assert_eq!(reused["model_build_attempts"], threads);
                for id in 0..6 {
                    let mut a = cold["runs"][id].as_object().unwrap().clone();
                    let mut b = reused["runs"][id].as_object().unwrap().clone();
                    assert!(!a.contains_key("error") && !b.contains_key("error"));
                    for key in [
                        "construction_seconds",
                        "fit_seconds",
                        "evaluation_seconds",
                        "peak_rss_kib",
                    ] {
                        a.remove(key);
                        b.remove(key);
                    }
                    assert_eq!(a, b, "{family}, threads {threads}, id {id}");
                }
            }
        }
        let failed = repeated_fits(&arguments("invalid"), 5, 2, true).unwrap();
        assert_eq!(failed["model_build_attempts"], 2);
        for id in 0..5 {
            assert_eq!(failed["runs"][id]["fit_id"], id);
            assert!(failed["runs"][id]["error"].is_string());
        }
    }
    #[test]
    fn reused_evaluator_observation_starts_fresh_for_each_optimization() {
        let mut args = arguments("bernoulli");
        args[5] = "counted".into();
        let report = repeated_fits(&args, 4, 1, true).unwrap();
        for row in report["runs"].as_array().unwrap() {
            assert_eq!(row["target_calls"], row["evaluations"]);
            assert!(row["target_calls"].as_u64().unwrap() > 0);
        }
    }
}
