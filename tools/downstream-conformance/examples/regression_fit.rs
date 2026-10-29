//! Fixed-design fitting and separate numerical/statistical assessment.
use alea::{
    distributions::{
        modeling::{
            bernoulli_logit, normal_observation, observation,
            regression::{diagnostics::*, *},
        },
        scalar::Poisson,
    },
    model::{
        assembly::{CompiledModel, ModelBuilder},
        prediction::NamedOutput,
    },
    optimize::{
        LbfgsOptions, MaximizeError, NamedFit, ObjectiveDescription, ObjectiveKind, maximize,
    },
};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 && args.len() != 6 {
        return Err("usage: regression_fit gaussian|logistic|poisson ordinary|scaled|ill_conditioned|rank_deficient|offset|boundary|quasi|perfect|tracing|historical rows batch|scalar [raw|standardized]".into());
    }
    let (family, scenario, mode) = (args[1].as_str(), args[2].as_str(), args[4].as_str());
    let n: usize = args[3].parse()?;
    let coordinates = args.get(5).map_or("raw", String::as_str);
    if !["raw", "standardized"].contains(&coordinates) {
        return Err("invalid coefficient coordinates".into());
    }
    if n < 3
        || !["gaussian", "logistic", "poisson"].contains(&family)
        || !["batch", "scalar"].contains(&mode)
        || ![
            "ordinary",
            "scaled",
            "ill_conditioned",
            "rank_deficient",
            "offset",
            "boundary",
            "quasi",
            "perfect",
            "tracing",
            "historical",
        ]
        .contains(&scenario)
        || (scenario == "perfect" && family != "gaussian")
        || (scenario == "quasi" && family != "logistic")
        || (scenario == "boundary" && family == "gaussian")
        || (scenario == "tracing" && family == "gaussian")
    {
        return Err("invalid case".into());
    }
    let started = Instant::now();
    let mut design = Vec::with_capacity(2 * n);
    let mut real = Vec::with_capacity(n);
    let mut binary = Vec::with_capacity(n);
    let mut counts = Vec::with_capacity(n);
    let mut offsets = Vec::with_capacity(n);
    for i in 0..n {
        let x = if ["tracing", "historical"].contains(&scenario) {
            (i % 101) as f64 / 50. - 1.
        } else if scenario == "boundary" {
            (i % 10) as f64 - 4.5
        } else if scenario == "quasi" {
            (i % 3) as f64
        } else {
            (i % 11) as f64 - 5.
        };
        let predictor = match scenario {
            "scaled" => x * 1e4,
            "ill_conditioned" => 1. + 1e-8 * x,
            "rank_deficient" => 1.,
            _ => x,
        };
        design.extend([1., predictor]);
        real.push(if scenario == "perfect" {
            1. + 0.5 * x
        } else {
            1. + 0.5 * x + ((i * 17 % 19) as f64 - 9.) / 10.
        });
        binary.push(match scenario {
            "tracing" => i % 3 == 0,
            "historical" => (i * 19 % 31) as f64 / 31. < 0.5 + 0.2 * x,
            "boundary" => x > 0.,
            "quasi" => x > 0. || i % 2 == 0,
            _ => (i * 19 % 31) as f64 / 31. < 0.5 + 0.08 * x,
        });
        counts.push(if scenario == "boundary" {
            0
        } else if scenario == "tracing" {
            (i % 7) as u64
        } else {
            (i * 13 % 7) as u64
        });
        offsets.push(if scenario == "offset" {
            ((i % 3) as f64 - 1.) * 0.2
        } else {
            0.
        });
    }
    let coefficient_names = ["intercept", "slope"];
    let (rd, bd, pd) = (
        RegressionData::new(2, design.clone(), real.clone())?.with_offsets(offsets.clone())?,
        RegressionData::new(2, design.clone(), binary.clone())?.with_offsets(offsets.clone())?,
        RegressionData::new(2, design.clone(), counts.clone())?.with_offsets(offsets.clone())?,
    );
    let construction_start = Instant::now();
    let (work_rd, map) = if coordinates == "standardized" {
        let (data, map) = rd.standardize(0)?.into_parts();
        (data, Some(map))
    } else {
        (rd.clone(), None)
    };
    let work_bd = if map.is_some() {
        bd.standardize(0)?.into_parts().0
    } else {
        bd.clone()
    };
    let work_pd = if map.is_some() {
        pd.standardize(0)?.into_parts().0
    } else {
        pd.clone()
    };
    let model = compile(family, mode, &work_rd, &work_bd, &work_pd)?;
    let expressions = model.expression_count();
    let names = NamedOutput::new(model.coordinates(), model.layout().clone(), ())?;
    let objective = model.select_likelihood()?.into_objective()?;
    let construction_seconds = construction_start.elapsed().as_secs_f64();
    let mut options = LbfgsOptions::new(500, 5000)?;
    if scenario == "historical" {
        options = options.with_score_tolerance(n as f64 * 1e-7)?;
    }
    if scenario == "scaled" && coordinates == "raw" {
        options = options.with_scales(if family == "gaussian" {
            vec![1., 1e-4, 1.]
        } else {
            vec![1., 1e-4]
        })?;
    }
    let tolerance = options.score_tolerance();
    let mut initial = if family == "gaussian" {
        vec![0.1, -0.2, 0.]
    } else if scenario == "tracing" {
        vec![0., 0.]
    } else {
        vec![0.1, -0.2]
    };
    if scenario == "scaled" {
        initial[1] *= 1e-4;
    }
    if let Some(map) = &map {
        let transformed = map.to_standardized(&initial[..2])?;
        initial[..2].copy_from_slice(&transformed);
    }
    let fit_start = Instant::now();
    let outcome = maximize(
        &objective,
        &initial,
        ObjectiveDescription {
            kind: ObjectiveKind::Likelihood,
            measure: "unconstrained coefficients and positive Gaussian scale".into(),
            model: family.into(),
            data: format!("{scenario}-{n}"),
            source: "regression-fit-v1".into(),
        },
        options,
    );
    let (result, evaluation_failed) = match outcome {
        Ok(result) => (result, false),
        Err(MaximizeError::Evaluation(error)) => {
            eprintln!("{}", error.source);
            (*error.result, true)
        }
        Err(error) => return Err(error.into()),
    };
    let fit_seconds = fit_start.elapsed().as_secs_f64();
    let assessment_start = Instant::now();
    let evidence = match family {
        "logistic" => bd.assess_bernoulli(&coefficient_names, ConeSearchOptions::default())?,
        "poisson" => pd.assess_poisson(&coefficient_names, ConeSearchOptions::default())?,
        _ => {
            #[cfg(feature = "faer")]
            {
                rd.assess_gaussian(&coefficient_names, 1e-12, 1e-12)?
            }
            #[cfg(not(feature = "faer"))]
            {
                return Err("Gaussian assessment requires faer".into());
            }
        }
    };
    let assessment_seconds = assessment_start.elapsed().as_secs_f64();
    let fit = NamedFit::new(names, result)?.with_assessment(evidence);
    let r = fit.result();
    let c = r.candidate.as_ref().ok_or("no candidate")?;
    let original_position = if let Some(map) = &map {
        let mut position = map.to_original(&c.position[..2])?;
        position.extend_from_slice(&c.position[2..]);
        position
    } else {
        c.position.clone()
    };
    let mut original_gradient = vec![f64::NAN; original_position.len()];
    let original_value = if map.is_some() {
        use alea::target::LogDensityGradient;
        let original = compile(family, mode, &rd, &bd, &pd)?.into_objective()?;
        original.logp_grad(&original_position, &mut original_gradient)?
    } else {
        original_gradient.copy_from_slice(&c.gradient);
        c.value
    };
    let coefficient_centers = map
        .as_ref()
        .map_or_else(|| vec![0.; 2], |m| m.centers().to_vec());
    let coefficient_scales = map
        .as_ref()
        .map_or_else(|| vec![1.; 2], |m| m.scales().to_vec());
    let optimization_scales = r
        .options
        .scales()
        .map_or_else(|| vec![1.; r.dimension], <[f64]>::to_vec);
    let existence = match fit.assessment().evidence() {
        RegressionEvidence::BernoulliLogit(m) | RegressionEvidence::PoissonLog(m) => match m {
            MleExistence::FiniteMaximum => "finite",
            MleExistence::NoFiniteMaximum(_) => "no_finite_maximum",
            MleExistence::Unresolved => "unresolved",
        },
        #[cfg(feature = "faer")]
        RegressionEvidence::Gaussian(_) => "numerical_gaussian_assessment",
    };
    let rss = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|n| n.parse::<u64>().ok())
        })
        .map_or("null".into(), |n| n.to_string());
    println!(
        "{{\"family\":\"{family}\",\"scenario\":\"{scenario}\",\"mode\":\"{mode}\",\"coordinates\":\"{coordinates}\",\"coefficient_centers\":{coefficient_centers:?},\"coefficient_scales\":{coefficient_scales:?},\"optimization_position\":{:?},\"optimization_gradient\":{:?},\"optimization_scales\":{optimization_scales:?},\"optimization_value\":{},\"rows\":{n},\"expressions\":{expressions},\"position\":{:?},\"gradient\":{:?},\"value\":{},\"termination\":\"{:?}\",\"line_search_failure\":\"{:?}\",\"score_tolerance\":{tolerance},\"existence\":\"{existence}\",\"evaluation_failed\":{evaluation_failed},\"evaluations\":{},\"iterations\":{},\"construction_seconds\":{construction_seconds},\"fit_seconds\":{fit_seconds},\"assessment_seconds\":{assessment_seconds},\"total_seconds\":{},\"peak_rss_kib\":{rss}}}",
        c.position,
        c.gradient,
        c.value,
        original_position,
        original_gradient,
        original_value,
        r.termination,
        r.line_search_failure,
        r.evaluations,
        r.iterations,
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn compile(
    family: &str,
    mode: &str,
    rd: &RegressionData<f64>,
    bd: &RegressionData<bool>,
    pd: &RegressionData<u64>,
) -> Result<CompiledModel, Box<dyn std::error::Error>> {
    let design = rd.design();
    let offsets = rd.offsets();
    let n = rd.rows();
    let real = rd.observations();
    let binary = bd.observations();
    let counts = pd.observations();
    let mut b = ModelBuilder::new();
    let a = b.real("intercept")?;
    let slope = b.real("slope")?;
    let beta = [a.expr(), slope.expr()];
    let sigma = if family == "gaussian" {
        Some(b.positive("sigma")?)
    } else {
        None
    };
    if mode == "batch" {
        let f = match family {
            "gaussian" => normal_regression(&mut b, &beta, sigma.as_ref().unwrap(), rd.clone())?,
            "logistic" => bernoulli_logit_regression(&mut b, &beta, bd.clone())?,
            _ => poisson_log_regression(&mut b, &beta, pd.clone())?,
        };
        b.likelihood("observations", f)?;
    } else {
        for i in 0..n {
            let eta = b.linear_combination(&beta, &design[2 * i..2 * i + 2], offsets[i])?;
            let f = match family {
                "gaussian" => normal_observation(&mut b, real[i], eta, sigma.as_ref().unwrap())?,
                "logistic" => bernoulli_logit(&mut b, binary[i], eta)?,
                _ => {
                    let rate = b.exp(eta)?;
                    observation::<Poisson, _, _>(&mut b, counts[i], &[rate], |x| {
                        Poisson::new(x[0])
                    })?
                }
            };
            b.likelihood(&format!("row {i}"), f)?;
        }
    }
    Ok(b.compile()?)
}
