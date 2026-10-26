//! Fit partially pooled event probabilities with noncentered group effects.
use alea::{
    diagnostics::modern,
    distribution::Sample,
    distributions::{
        modeling::{bernoulli_logit, normal},
        scalar::Bernoulli,
    },
    ensemble::Execution,
    model::{
        ModelError,
        assembly::{AssemblyError, CompiledModel, ModelBuilder},
    },
    sampling::{NutsOptions, adapt::MetricKind},
    session::{Kernel, SessionBuilder, Warmup, storage::Discard},
    target::EvaluationError,
    transform::Transform,
};
use rand::{SeedableRng, rngs::SmallRng};

// (group label, observed events, number of trials). Counts are data, not parameters.
const GROUPS: [(&str, usize, usize); 3] = [("a", 3, 12), ("b", 7, 12), ("c", 10, 12)];
const DIMENSION: usize = 5;
const DRAWS: usize = 2_000;

fn model() -> Result<CompiledModel, AssemblyError> {
    let mut b = ModelBuilder::new();
    let mu = b.real("mu")?;
    let log_tau = b.real("log_tau")?;
    b.annotate(&mu, "population", "location", "population")?;
    b.annotate(&log_tau, "population", "log_scale", "population")?;
    let z = b.vector(
        "z",
        "group",
        &["a", "b", "c"],
        "effects",
        Transform::identity(3)?,
    )?;
    let zero = b.constant(0.0)?;
    let one = b.constant(1.0)?;
    let mu_sd = b.constant(0.7)?;
    let scale_location = b.constant(-0.7)?;
    let scale_sd = b.constant(0.3)?;
    let prior = normal(&mut b, mu.expr(), zero, mu_sd)?;
    b.factor(prior)?;
    let prior = normal(&mut b, log_tau.expr(), scale_location, scale_sd)?;
    b.factor(prior)?;
    let tau = b.exp(log_tau.expr())?;
    for ((_, events, trials), &zi) in GROUPS.iter().zip(z.expressions()) {
        let prior = normal(&mut b, zi, zero, one)?;
        b.factor(prior)?;
        let deviation = b.multiply(tau, zi)?;
        let log_odds = b.add(mu.expr(), deviation)?;
        for trial in 0..*trials {
            let likelihood = bernoulli_logit(&mut b, trial < *events, log_odds)?;
            b.factor(likelihood)?;
        }
    }
    b.compile()
}

fn logistic(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        x.exp() / (1.0 + x.exp())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let description = model()?;
    let names = description.coordinates();
    // log_tau is a real parameter; tau = exp(log_tau) is a derived quantity.
    // There is no density on tau here, so no extra Jacobian is needed.
    let session = SessionBuilder::new(
        "hierarchical-logistic;events=3,7,10;trials=12,12,12;model=1",
        DIMENSION,
        vec![
            vec![-0.4, -0.9, -0.2, 0.0, 0.2],
            vec![0.0, -0.7, 0.0, 0.0, 0.0],
            vec![0.4, -0.5, 0.2, 0.0, -0.2],
            vec![0.2, -0.8, -0.1, 0.1, 0.3],
        ],
    )
    .seed([29; 32])
    .draws(DRAWS)
    .kernel(Kernel::Nuts(NutsOptions::new(0.15, 9)?))
    .warmup(Warmup::Covariance {
        iterations: 1_000,
        metric: MetricKind::Diagonal,
        target: 0.9.try_into()?,
        search: None,
    })
    .build()?;
    let result = session.run(
        |_| {
            model()
                .map_err(|e| ModelError::Evaluation(EvaluationError::Model(e)))?
                .into_target()
        },
        Execution::Sequential,
        &Discard,
    )?;
    println!("Run status: {:?}", result.status);
    for chain in &result.chains {
        println!(
            "Chain {:?}: completed={}, failure={:?}, numerical_rejections={}",
            chain.chain, chain.completed, chain.failure, chain.numerical_rejections
        );
    }
    if result
        .chains
        .iter()
        .any(|c| c.failure.is_some() || c.completed != DRAWS)
    {
        return Err("Incomplete fit; inspect the chain outcomes".into());
    }
    let divergent = result
        .chains
        .iter()
        .flat_map(|c| &c.draws)
        .filter(|d| d.stats.divergent == Some(true))
        .count();
    let depth_saturated = result
        .chains
        .iter()
        .flat_map(|c| &c.draws)
        .filter(|d| d.stats.depth_saturated == Some(true))
        .count();
    println!("Retained divergences={divergent}, depth saturations={depth_saturated}");
    if divergent > 0
        || depth_saturated > 0
        || result.chains.iter().any(|c| c.numerical_rejections > 0)
    {
        return Err("Investigate transition warnings before using these predictions".into());
    }
    // Keep chains separate for diagnostics of both parameters and predictions.
    let mut acceptable = true;
    for chain in &result.chains {
        let energies = chain
            .draws
            .iter()
            .map(|d| d.stats.energy.ok_or("Missing NUTS energy"))
            .collect::<Result<Vec<_>, _>>()?;
        let ebfmi = modern::ebfmi(&energies)?;
        println!("Chain {:?}: E-BFMI={ebfmi:.3}", chain.chain);
        acceptable &= ebfmi >= 0.3;
    }
    for index in 0..DIMENSION + 1 + GROUPS.len() {
        let label = if index < DIMENSION {
            names[index].name.clone()
        } else if index == DIMENSION {
            "tau".to_owned()
        } else {
            format!("p[{}]", GROUPS[index - DIMENSION - 1].0)
        };
        let chains: Vec<Vec<f64>> = result
            .chains
            .iter()
            .map(|c| {
                c.draws
                    .iter()
                    .map(|d| {
                        let q = &d.position;
                        if index < DIMENSION {
                            q[index]
                        } else if index == DIMENSION {
                            q[1].exp()
                        } else {
                            logistic(q[0] + q[1].exp() * q[index - DIMENSION + 1])
                        }
                    })
                    .collect()
            })
            .collect();
        let refs: Vec<&[f64]> = chains.iter().map(Vec::as_slice).collect();
        let s = modern::summarize(&refs)?;
        println!(
            "{label}: mean={:.6}, MCSE={:.6}, Rhat={:.4}, bulk_ESS={:.0}, tail_ESS={:.0}",
            s.mean, s.mcse_mean, s.rhat, s.bulk_ess, s.tail_ess
        );
        acceptable &= s.rhat <= 1.01 && s.bulk_ess >= 400.0 && s.tail_ess >= 400.0;
    }
    if !acceptable {
        return Err("Diagnostics need investigation before using these predictions".into());
    }
    let mut rng = SmallRng::seed_from_u64(30);
    for (group, &(label, events, trials)) in GROUPS.iter().enumerate() {
        let mut predicted_events = 0;
        for q in result
            .chains
            .iter()
            .flat_map(|c| c.draws.iter().map(|d| &d.position))
        {
            let law = Bernoulli::new(logistic(q[0] + q[1].exp() * q[group + 2]))?;
            for _ in 0..trials {
                predicted_events += usize::from(law.sample(&mut rng)?);
            }
        }
        let replicates = result.chains.iter().map(|c| c.draws.len()).sum::<usize>();
        println!(
            "Group {label}: observed {events}/{trials}; posterior predictive mean events={:.3}",
            predicted_events as f64 / replicates as f64
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alea::target::LogDensityGradient;

    // Independent scalar posterior in the sampled coordinates, including constants.
    fn independent(q: &[f64]) -> f64 {
        fn gaussian(x: f64, mu: f64, sd: f64) -> f64 {
            -0.5 * ((x - mu) / sd).powi(2) - sd.ln() - 0.5 * std::f64::consts::TAU.ln()
        }
        let mut lp = gaussian(q[0], 0.0, 0.7) + gaussian(q[1], -0.7, 0.3);
        for (i, &(_, events, trials)) in GROUPS.iter().enumerate() {
            let z = q[i + 2];
            lp += gaussian(z, 0.0, 1.0);
            let eta = q[0] + q[1].exp() * z;
            let softplus = eta.max(0.0) + (-eta.abs()).exp().ln_1p();
            lp += events as f64 * eta - trials as f64 * softplus;
        }
        lp
    }

    #[test]
    fn density_and_gradients_match_independent_formula() -> Result<(), Box<dyn std::error::Error>> {
        let compiled = model()?;
        assert_eq!(compiled.coordinates()[4].name, "z[c]");
        let target = compiled.into_target()?;
        for q in [
            [0.0, -0.7, 0.0, 0.0, 0.0],
            [-1.2, -1.5, 0.3, -0.8, 1.1],
            [1.1, 0.2, -2.0, 0.5, 2.0],
        ] {
            let mut gradient = [0.0; DIMENSION];
            let value = target.logp_grad(&q, &mut gradient)?;
            assert!((value - independent(&q)).abs() < 1e-11);
            for i in 0..DIMENSION {
                let mut plus = q;
                let mut minus = q;
                plus[i] += 1e-5;
                minus[i] -= 1e-5;
                let fd = (independent(&plus) - independent(&minus)) / 2e-5;
                assert!(
                    (gradient[i] - fd).abs() < 1e-7 * (1.0 + fd.abs()),
                    "coordinate {i}"
                );
            }
        }
        Ok(())
    }
}
