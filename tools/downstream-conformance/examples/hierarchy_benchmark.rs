//! Identical hierarchical logistic target, expanded or weighted trial likelihood.
use alea::{
    optimize::{LbfgsOptions, ObjectiveDescription, ObjectiveKind, maximize},
    target::LogDensityGradient,
};
use alea_downstream_conformance::{
    modeling::{Observation, hierarchy, hierarchy_counts},
    profiling::EvaluationObserver,
};
use std::{hint::black_box, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: hierarchy_benchmark scalar|counts trials_per_group repetitions".into());
    }
    let mode = &args[1];
    let n = args[2].parse::<usize>()?;
    let repetitions = args[3].parse::<usize>()?;
    if !["scalar", "counts"].contains(&mode.as_str()) || n == 0 || repetitions == 0 || n > 1_000_000
    {
        return Err("invalid workload".into());
    }
    let groups = [(n / 4, n), (3 * (n / 4), n)];
    let started = Instant::now();
    let model = if mode == "scalar" {
        let data: Vec<_> = groups
            .iter()
            .enumerate()
            .flat_map(|(group, &(events, trials))| {
                (0..trials).map(move |i| Observation::Logistic {
                    group,
                    event: i < events,
                })
            })
            .collect();
        hierarchy(&data)
    } else {
        hierarchy_counts(&groups)
    };
    let expressions = model.expression_count();
    let target = model.into_target()?;
    let construction_seconds = started.elapsed().as_secs_f64();
    let initial = [0.1, -0.7, -0.2, 0.3];
    let mut gradient = [0.; 4];
    let value = target.logp_grad(&initial, &mut gradient)?;
    let started = Instant::now();
    for _ in 0..repetitions {
        black_box(target.logp_grad(black_box(&initial), black_box(&mut gradient))?);
    }
    let evaluation_seconds = started.elapsed().as_secs_f64() / repetitions as f64;
    let observer = EvaluationObserver::<_, true>::new(&target);
    let started = Instant::now();
    let result = maximize(
        &observer,
        &initial,
        ObjectiveDescription {
            kind: ObjectiveKind::Posterior,
            measure: "Lebesgue in mu, log_tau, z[0], z[1]".into(),
            model: "two-group logistic hierarchy".into(),
            data: format!("events={},{};trials={n}", groups[0].0, groups[1].0),
            source: "hierarchy_benchmark-v1".into(),
        },
        LbfgsOptions::new(500, 5000)?.with_score_tolerance(1e-7)?,
    )?;
    let fit_seconds = started.elapsed().as_secs_f64();
    let c = result.candidate.as_ref().ok_or("no accepted candidate")?;
    println!(
        "{}",
        serde_json::json!({"mode":mode,"trials":n,"groups":groups,"repetitions":repetitions,
        "expressions":expressions,"construction_seconds":construction_seconds,"evaluation_seconds":evaluation_seconds,
        "fit_seconds":fit_seconds,"target_seconds":observer.elapsed().unwrap().as_secs_f64(),"target_calls":observer.calls(),
        "value":value,"gradient":gradient,"estimate":c.position,"final_value":c.value,"final_gradient":c.gradient,
        "termination":format!("{:?}",result.termination),"line_search_failure":format!("{:?}",result.line_search_failure),
        "iterations":result.iterations,"evaluations":result.evaluations,"score_tolerance":result.options.score_tolerance()})
    );
    Ok(())
}
