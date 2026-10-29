//! Deterministic synthetic likelihood benchmark; timings are not statistical qualification.
use alea::{
    distributions::{
        modeling::{bernoulli_logit, normal_observation, observation, regression::*},
        scalar::Poisson,
    },
    model::assembly::ModelBuilder,
    optimize::{LbfgsOptions, ObjectiveDescription, ObjectiveKind, maximize},
    target::LogDensityGradient,
};
use alea_downstream_conformance::profiling::EvaluationObserver;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use std::{hint::black_box, io::Write, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(5..=8).contains(&args.len()) {
        return Err(
            "usage: regression_benchmark normal|bernoulli|poisson batch|scalar rows repetitions [plain|counted|timed] [columns] [dataset.json]"
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
        return Ok(());
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
    let counted = EvaluationObserver::<_, false>::new(&objective);
    let timed = EvaluationObserver::<_, true>::new(&objective);
    let target: &dyn LogDensityGradient<Error = _> = match profile {
        "counted" => &counted,
        "timed" => &timed,
        _ => &objective,
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
        LbfgsOptions::new(100, 1000)?.with_score_tolerance(n as f64 * 1e-7)?,
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
    println!(
        "{{\"columns\":{columns},\"profile\":\"{profile}\",\"target_calls\":{target_calls},\"target_seconds\":{target_seconds},\"family\":\"{family}\",\"mode\":\"{mode}\",\"rows\":{n},\"repetitions\":{repetitions},\"expressions\":{expressions},\"construction_seconds\":{construction_seconds},\"evaluation_seconds\":{evaluation_seconds},\"fit_seconds\":{fit_seconds},\"peak_rss_kib\":{rss},\"value\":{value},\"gradient\":{gradient:?},\"final_gradient\":{final_gradient:?},\"final_value\":{final_value},\"score_tolerance\":{score_tolerance},\"line_search_failure\":\"{line_search_failure:?}\",\"termination\":\"{:?}\",\"evaluations\":{},\"iterations\":{},\"estimate_free\":{:?}}}",
        result.termination,
        result.evaluations,
        result.iterations,
        estimates.ok_or("no benchmark estimate")?
    );
    Ok(())
}
