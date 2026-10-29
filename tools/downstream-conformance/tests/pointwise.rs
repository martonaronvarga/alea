//! One public-facade graph is fitted, then reused for aligned group likelihoods.
use alea::{
    distributions::modeling::{bernoulli_logit, normal},
    math::buffer::OwnedBuffer,
    model::assembly::{ModelBuilder, PointwiseLikelihood, PredictiveUnit},
    sampling::{
        HmcOptions, Nuts,
        adapt::{HmcWarmup, WarmupOptions},
        config::TreeDepth,
    },
};
use rand::{SeedableRng, rngs::SmallRng};

fn normal_log(x: f64) -> f64 {
    -0.5 * x * x - 0.5 * (2. * std::f64::consts::PI).ln()
}

#[test]
fn retained_hierarchical_logistic_draws_keep_conditional_predictive_units() {
    let mut b = ModelBuilder::new();
    let mu = b.real("mu").unwrap();
    let tau = b.positive("tau").unwrap();
    let z = [b.real("z0").unwrap(), b.real("z1").unwrap()];
    let zero = b.constant(0.).unwrap();
    let one = b.constant(1.).unwrap();
    let prior = normal(&mut b, mu.expr(), zero, one).unwrap();
    b.prior("mu prior", prior).unwrap();
    let minus = b.constant(-1.).unwrap();
    let prior = b.multiply(minus, tau.expr()).unwrap();
    b.prior("tau prior", prior).unwrap();
    let data = [[false, true, true], [false, false, true]];
    let mut units = Vec::new();
    for group in 0..2 {
        let prior = normal(&mut b, z[group].expr(), zero, one).unwrap();
        b.prior(&format!("z{group} prior"), prior).unwrap();
        let deviation = b.multiply(tau.expr(), z[group].expr()).unwrap();
        let effect = b.add(mu.expr(), deviation).unwrap();
        let mut names = Vec::new();
        let mut observations = Vec::new();
        for (row, &event) in data[group].iter().enumerate() {
            let factor = bernoulli_logit(&mut b, event, effect).unwrap();
            let name = format!("group {group}:row {row}");
            b.likelihood(&name, factor).unwrap();
            names.push(name.clone());
            observations.push(name);
        }
        units.push(PredictiveUnit {
            name: format!("group {group}"),
            factors: names,
            observations,
            conditioned_on: vec!["mu".into(), "tau".into(), format!("z{group}")],
            integrated_out: vec![],
        });
    }
    let target = b
        .compile()
        .unwrap()
        .select_posterior()
        .unwrap()
        .into_target()
        .unwrap();
    let mut rng = SmallRng::seed_from_u64(1402901);
    let mut retained: Vec<([f64; 4], f64)> = Vec::new();
    {
        let (hmc, _) = HmcWarmup::new(
            &target,
            OwnedBuffer::from_fn(4, |_| 0.),
            HmcOptions::new(0.08, 1).unwrap(),
            WarmupOptions::new(128),
        )
        .unwrap()
        .run(&mut rng)
        .unwrap();
        let mut nuts = Nuts::from_hmc(hmc, TreeDepth::try_from(6).unwrap()).unwrap();
        for _ in 0..32 {
            let _ = nuts.step(&mut rng).unwrap();
            retained.push((
                nuts.point().position().try_into().unwrap(),
                nuts.point().log_density(),
            ));
        }
    }
    // Reuse the exact owned graph; there is no second model implementation.
    let pointwise = PointwiseLikelihood::new(target.into_model(), units).unwrap();
    for (q, log_density) in retained {
        let mut output = [99.; 2];
        let total = pointwise.evaluate(&q, &mut output).unwrap();
        for group in 0..2 {
            let eta = q[0] + q[1].exp() * q[2 + group];
            let softplus = eta.max(0.) + (-eta.abs()).exp().ln_1p();
            let expected: f64 = data[group]
                .iter()
                .map(|&y| f64::from(y) * eta - softplus)
                .sum();
            assert!((output[group] - expected).abs() < 1e-12);
        }
        let prior = normal_log(q[0]) - q[1].exp() + normal_log(q[2]) + normal_log(q[3]);
        assert!((total + prior + q[1] - log_density).abs() < 1e-12);
        assert!((total - output.iter().sum::<f64>()).abs() < 1e-12);
    }
    assert_eq!(pointwise.units()[1].observations[2], "group 1:row 2");
    assert!(
        pointwise
            .units()
            .iter()
            .all(|unit| unit.integrated_out.is_empty())
    );
}
