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
use std::{hint::black_box, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err(
            "usage: regression_benchmark normal|bernoulli|poisson batch|scalar rows repetitions"
                .into(),
        );
    }
    let family = args[1].as_str();
    let mode = args[2].as_str();
    let n = args[3].parse::<usize>()?;
    let repetitions = args[4].parse::<usize>()?;
    if !["normal", "bernoulli", "poisson"].contains(&family)
        || !["batch", "scalar"].contains(&mode)
        || n == 0
        || repetitions == 0
    {
        return Err("invalid benchmark configuration".into());
    }
    let started = Instant::now();
    let mut b = ModelBuilder::new();
    let beta = [b.real("intercept")?.expr(), b.real("slope")?.expr()];
    let sigma = if family == "normal" {
        Some(b.positive("sigma")?)
    } else {
        None
    };
    let design: Vec<_> = (0..n)
        .flat_map(|i| [1., (i % 101) as f64 / 50. - 1.])
        .collect();
    let real: Vec<_> = (0..n)
        .map(|i| 1. + 0.5 * design[2 * i + 1] + ((i * 17 % 19) as f64 - 9.) / 10.)
        .collect();
    let binary: Vec<_> = (0..n)
        .map(|i| (i * 19 % 31) as f64 / 31. < 0.5 + 0.2 * design[2 * i + 1])
        .collect();
    let counts: Vec<_> = (0..n).map(|i| (i * 13 % 7) as u64).collect();
    if mode == "batch" {
        let f = match family {
            "normal" => normal_regression(
                &mut b,
                &beta,
                sigma.as_ref().unwrap(),
                RegressionData::new(2, design, real)?,
            )?,
            "bernoulli" => {
                bernoulli_logit_regression(&mut b, &beta, RegressionData::new(2, design, binary)?)?
            }
            _ => poisson_log_regression(&mut b, &beta, RegressionData::new(2, design, counts)?)?,
        };
        b.factor(f)?;
    } else {
        for i in 0..n {
            let predictor = b.constant(design[2 * i + 1])?;
            let effect = b.multiply(predictor, beta[1])?;
            let eta = b.add(beta[0], effect)?;
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
    let q = if family == "normal" {
        vec![0.1, -0.2, 0.]
    } else {
        vec![0.1, -0.2]
    };
    let mut g = vec![0.; q.len()];
    let value = objective.logp_grad(&q, &mut g)?;
    let gradient = g.clone();
    let started = Instant::now();
    for _ in 0..repetitions {
        black_box(objective.logp_grad(black_box(&q), black_box(&mut g))?);
    }
    let evaluation_seconds = started.elapsed().as_secs_f64() / repetitions as f64;
    let started = Instant::now();
    let result = maximize(
        &objective,
        &q,
        ObjectiveDescription {
            kind: ObjectiveKind::Likelihood,
            measure: "original parameters".into(),
            model: family.into(),
            data: format!("synthetic-v1-{n}"),
            source: "regression_benchmark-v1".into(),
        },
        LbfgsOptions::new(100, 1000)?.with_score_tolerance(n as f64 * 1e-7)?,
    )?;
    let fit_seconds = started.elapsed().as_secs_f64();
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
    println!(
        "{{\"family\":\"{family}\",\"mode\":\"{mode}\",\"rows\":{n},\"repetitions\":{repetitions},\"expressions\":{expressions},\"construction_seconds\":{construction_seconds},\"evaluation_seconds\":{evaluation_seconds},\"fit_seconds\":{fit_seconds},\"peak_rss_kib\":{rss},\"value\":{value},\"gradient\":{gradient:?},\"final_gradient\":{final_gradient:?},\"termination\":\"{:?}\",\"evaluations\":{},\"iterations\":{},\"estimate_free\":{:?}}}",
        result.termination,
        result.evaluations,
        result.iterations,
        estimates.ok_or("no benchmark estimate")?
    );
    Ok(())
}
