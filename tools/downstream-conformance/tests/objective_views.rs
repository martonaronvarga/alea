//! External facade consumer: one model, explicit statistical views.
use alea::{
    distributions::modeling::{normal, normal_observation},
    model::assembly::{AssemblyError, ModelBuilder},
    target::LogDensityGradient,
};

#[test]
fn a_single_owned_model_supports_likelihood_and_posterior_evaluation() -> Result<(), AssemblyError>
{
    let mut b = ModelBuilder::new();
    let mean = b.real("mean")?;
    let sigma = b.positive("sigma")?;
    let zero = b.constant(0.)?;
    let one = b.constant(1.)?;
    let prior = normal(&mut b, mean.expr(), zero, one)?;
    b.prior("mean prior", prior)?;
    let prior = normal(&mut b, sigma.expr(), zero, one)?;
    b.prior("scale prior", prior)?;
    let normalizer = b.constant(std::f64::consts::LN_2)?;
    b.prior("half-normal normalization", normalizer)?;
    for (i, y) in [2., 3.].into_iter().enumerate() {
        let observation = normal_observation(&mut b, y, mean.expr(), &sigma)?;
        b.likelihood(&format!("observation {i}"), observation)?;
    }
    let objective = b.compile()?.select_likelihood()?.into_objective().unwrap();
    let q = [1., 2_f64.ln()];
    let mut g = [0.; 2];
    let likelihood = objective.logp_grad(&q, &mut g).unwrap();
    assert_eq!(g, [0.75, -0.75]);
    assert_eq!(objective.model().selected_factors().count(), 2);
    let posterior = objective
        .into_model()
        .select_posterior()?
        .into_objective()
        .unwrap();
    let posterior_value = posterior.logp_grad(&q, &mut g).unwrap();
    assert_eq!(g, [-0.25, -4.75]);
    let prior_value = -2.5 - (2. * std::f64::consts::PI).ln() + std::f64::consts::LN_2;
    assert!((posterior_value - likelihood - prior_value).abs() < 1e-13);
    let target = posterior.into_model().into_target().unwrap();
    assert_eq!(
        target.logp_grad(&q, &mut g).unwrap(),
        posterior_value + q[1]
    );
    assert_eq!(g, [-0.25, -3.75]);
    assert_eq!(target.model().coordinates()[1].name, "sigma");
    Ok(())
}
