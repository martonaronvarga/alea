//! Native hierarchy fit and joint prediction against an analytic posterior.
//! Fixed separation/start/nondecision/group scale isolate drift inference.
use alea::{
    diagnostics::modern,
    distributions::{
        modeling::{normal, wiener4},
        wiener::{Boundary, Wiener4, Wiener4Params, WienerObservation, WienerOptions},
    },
    ensemble::Execution,
    model::{
        ModelError,
        assembly::{AssemblyError, CompiledModel, ModelBuilder},
    },
    sampling::{NutsOptions, adapt::MetricKind},
    session::{Kernel, SessionBuilder, Warmup, storage::Discard},
    target::EvaluationError,
};
use rand::{SeedableRng, rngs::SmallRng};

fn model(data: &[(usize, WienerObservation)]) -> Result<CompiledModel, AssemblyError> {
    let mut b = ModelBuilder::new();
    let mu = b.real("population_drift")?;
    let z = [b.real("subject_a")?, b.real("subject_b")?];
    let zero = b.constant(0.)?;
    let one = b.constant(1.)?;
    let prior_sd = b.constant(0.7)?;
    let scale = b.constant(0.5)?;
    let prior = normal(&mut b, mu.expr(), zero, prior_sd)?;
    b.factor(prior)?;
    let mut drift = Vec::new();
    for subject in &z {
        let prior = normal(&mut b, subject.expr(), zero, one)?;
        b.factor(prior)?;
        let deviation = b.multiply(scale, subject.expr())?;
        drift.push(b.add(mu.expr(), deviation)?);
    }
    let alpha = b.constant(1.2)?;
    let tau = b.constant(0.15)?;
    let beta = b.constant(0.4)?;
    for &(group, observation) in data {
        let likelihood = wiener4(
            &mut b,
            observation,
            [alpha, tau, beta, drift[group]],
            WienerOptions::default(),
        )?;
        b.factor(likelihood)?;
    }
    b.compile()
}

#[test]
fn hierarchical_wiener_fit_matches_analytic_posterior_and_predicts_joint_events()
-> Result<(), Box<dyn std::error::Error>> {
    let mut generating = SmallRng::seed_from_u64(74001);
    let mut data = Vec::new();
    for (group, drift) in [-0.4, 0.8].into_iter().enumerate() {
        let params = Wiener4Params::with_params(1.2, 0.15, 0.4, drift)?;
        for _ in 0..24 {
            data.push((group, Wiener4.sample(&params, &mut generating)?));
        }
    }
    // Independently integrate both normal random effects. The Wiener drift law
    // has sufficient statistics D=sum(signed exit distance), T=sum(decision time).
    let (mut d, mut t) = ([0.; 2], [0.; 2]);
    for &(group, obs) in &data {
        d[group] += if obs.boundary == Boundary::Upper {
            1.2 * 0.6
        } else {
            -1.2 * 0.4
        };
        t[group] += obs.rt - 0.15;
    }
    let lam = t.map(|t| 1. + 0.25 * t);
    let precision = 1. / 0.49 + t[0] / lam[0] + t[1] / lam[1];
    let mu = (d[0] / lam[0] + d[1] / lam[1]) / precision;
    let subject = [0, 1].map(|g| (mu + 0.25 * d[g]) / lam[g]);
    let expected = [
        mu,
        (subject[0] - mu) / 0.5,
        (subject[1] - mu) / 0.5,
        subject[0],
        subject[1],
    ];

    let result = SessionBuilder::new(
        "wiener-hierarchy;fixed-a-beta-tau-scale;dataset=74001;v1",
        3,
        vec![
            vec![-0.2, 0., 0.],
            vec![0., 0.2, -0.2],
            vec![0.2, 0., 0.],
            vec![0., -0.2, 0.2],
        ],
    )
    .seed([74; 32])
    .draws(1500)
    .kernel(Kernel::Nuts(NutsOptions::new(0.15, 8)?))
    .warmup(Warmup::Covariance {
        iterations: 700,
        metric: MetricKind::Dense,
        target: 0.9.try_into()?,
        search: None,
    })
    .build()?
    .run(
        |_| {
            model(&data)
                .map_err(|e| ModelError::Evaluation(EvaluationError::Model(e)))?
                .into_target()
        },
        Execution::Sequential,
        &Discard,
    )?;
    for chain in &result.chains {
        assert!(chain.failure.is_none());
        assert_eq!(chain.completed, 1500);
        assert_eq!(chain.numerical_rejections, 0);
        assert!(
            chain
                .draws
                .iter()
                .all(|d| d.stats.divergent != Some(true) && d.stats.depth_saturated != Some(true))
        );
    }
    for (index, truth) in expected.into_iter().enumerate() {
        let chains: Vec<Vec<f64>> = result
            .chains
            .iter()
            .map(|c| {
                c.draws
                    .iter()
                    .map(|d| {
                        if index < 3 {
                            d.position[index]
                        } else {
                            d.position[0] + 0.5 * d.position[index - 2]
                        }
                    })
                    .collect()
            })
            .collect();
        let refs: Vec<_> = chains.iter().map(Vec::as_slice).collect();
        let summary = modern::summarize(&refs)?;
        assert!(summary.rhat < 1.03 && summary.bulk_ess > 200. && summary.tail_ess > 200.);
        assert!(
            (summary.mean - truth).abs() < 6. * summary.mcse_mean,
            "coordinate {index}: {} vs {truth}",
            summary.mean
        );
    }
    // Existing subjects use their fitted z. New subjects require a fresh z.
    let mut prediction = SmallRng::seed_from_u64(74002);
    use alea::distribution::Sample;
    let standard = alea::distributions::scalar::Normal::new(0., 1.)?;
    let (mut existing_upper, mut new_upper, mut times) = (0, 0, Vec::new());
    let (mut expected_choices, mut choice_variance) = ([0.; 2], [0.; 2]);
    fn probability(v: f64) -> f64 {
        let r = 2.4 * v;
        if r == 0. {
            0.4
        } else if r > 0. {
            (-(-0.4 * r).exp_m1()) / (-(-r).exp_m1())
        } else {
            (0.6 * r).exp() * (0.4 * r).exp_m1() / r.exp_m1()
        }
    }
    for draw in result.chains.iter().flat_map(|c| &c.draws) {
        let existing_drift = draw.position[0] + 0.5 * draw.position[1];
        let new_drift = draw.position[0] + 0.5 * standard.sample(&mut prediction)?;
        let existing = Wiener4.sample(
            &Wiener4Params::with_params(1.2, 0.15, 0.4, existing_drift)?,
            &mut prediction,
        )?;
        let new = Wiener4.sample(
            &Wiener4Params::with_params(1.2, 0.15, 0.4, new_drift)?,
            &mut prediction,
        )?;
        existing_upper += usize::from(existing.boundary == Boundary::Upper);
        new_upper += usize::from(new.boundary == Boundary::Upper);
        times.push((existing.rt, new.rt));
        for (i, v) in [existing_drift, new_drift].into_iter().enumerate() {
            let p = probability(v);
            expected_choices[i] += p;
            choice_variance[i] += p * (1. - p);
        }
    }
    assert!(times.iter().all(|&(a, b)| a > 0.15 && b > 0.15));
    assert!(existing_upper > 0 && existing_upper < times.len());
    assert!(new_upper > 0 && new_upper < times.len());
    for (i, actual) in [existing_upper, new_upper].into_iter().enumerate() {
        assert!((actual as f64 - expected_choices[i]).abs() < 6. * choice_variance[i].sqrt() + 3.);
    }
    Ok(())
}
