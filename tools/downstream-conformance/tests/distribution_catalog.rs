use alea::{
    distributions::modeling::{continuous, observation},
    model::assembly::ModelBuilder,
    prelude::*,
    target::LogDensityGradient,
};
#[test]
fn generic_distribution_factors_compose_and_pull_back_scores()
-> Result<(), Box<dyn std::error::Error>> {
    let mut b = ModelBuilder::new();
    let scale = b.positive("scale")?;
    let mean = b.positive("mean")?;
    let dispersion = b.positive("dispersion")?;
    let prior_scale = b.constant(1.5)?;
    let prior = continuous(&mut b, scale.expr(), &[prior_scale], |p| {
        HalfNormal::new(p[0])
    })?;
    b.factor(prior)?;
    let likelihood = observation(&mut b, 4_u64, &[mean.expr(), dispersion.expr()], |p| {
        NegativeBinomialMean::new(p[0], p[1])
    })?;
    b.factor(likelihood)?;
    let target = b.compile()?.into_target()?;
    let q = [0.3, 0.8, -0.2];
    let mut g = [0.0; 3];
    let lp = target.logp_grad(&q, &mut g)?;
    let expected = HalfNormal::new(1.5)?.log_prob(&q[0].exp())?
        + NegativeBinomialMean::new(q[1].exp(), q[2].exp())?.log_prob(&4)?
        + q.iter().sum::<f64>();
    assert!((lp - expected).abs() < 1e-12);
    for i in 0..3 {
        let mut a = q;
        let mut c = q;
        let h = 1e-5;
        a[i] += h;
        c[i] -= h;
        let mut scratch = [0.0; 3];
        let fd =
            (target.logp_grad(&a, &mut scratch)? - target.logp_grad(&c, &mut scratch)?) / (2.0 * h);
        assert!((g[i] - fd).abs() < 1e-7);
    }
    Ok(())
}
#[test]
fn vector_observations_and_parameter_shape_failures_are_checked()
-> Result<(), Box<dyn std::error::Error>> {
    let mut b = ModelBuilder::new();
    let a = b.positive("concentration")?;
    let node = observation(
        &mut b,
        vec![0.2, 0.3, 0.5],
        &[a.expr(), a.expr(), a.expr()],
        |p| Dirichlet::new(p.to_vec()),
    )?;
    b.factor(node)?;
    let target = b.compile()?.into_target()?;
    let mut g = [0.0];
    assert!(target.logp_grad(&[0.2], &mut g).unwrap().is_finite());
    let mut b = ModelBuilder::new();
    let a = b.real("a")?;
    let node = observation(&mut b, 1_u64, &[a.expr()], |_| {
        NegativeBinomialMean::new(2.0, 3.0)
    })?;
    b.factor(node)?;
    let t = b.compile()?.into_target()?;
    assert!(t.logp_grad(&[0.0], &mut g).is_err());
    Ok(())
}
