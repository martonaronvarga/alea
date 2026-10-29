use alea::{
    distributions::modeling::{normal, normal_observation},
    model::assembly::{CompiledModel, ModelBuilder},
    optimize::{
        LbfgsOptions, ObjectiveDescription, ObjectiveKind, OptimizationTermination, maximize,
    },
};
fn description(kind: ObjectiveKind) -> ObjectiveDescription {
    ObjectiveDescription {
        kind,
        measure: "Lebesgue in original parameters".into(),
        model: "analytic regression".into(),
        data: "five observations".into(),
        source: "downstream conformance".into(),
    }
}
fn regression() -> CompiledModel {
    let mut b = ModelBuilder::new();
    let intercept = b.real("intercept").unwrap();
    let slope = b.real("slope").unwrap();
    let sigma = b.positive("sigma").unwrap();
    for (i, (x, e)) in [-2., -1., 0., 1., 2.]
        .into_iter()
        .zip([1., -2., 2., -2., 1.])
        .enumerate()
    {
        let x = b.constant(x).unwrap();
        let effect = b.multiply(x, slope.expr()).unwrap();
        let mu = b.add(intercept.expr(), effect).unwrap();
        let y = 1. + 2. * [-2., -1., 0., 1., 2.][i] + e;
        let term = normal_observation(&mut b, y, mu, &sigma).unwrap();
        b.likelihood(&format!("observation {i}"), term).unwrap();
    }
    b.compile().unwrap()
}
#[test]
fn gaussian_regression_mle_agrees_with_independent_coefficients_value_and_scale() {
    let objective = regression()
        .select_likelihood()
        .unwrap()
        .into_objective()
        .unwrap();
    for start in [[0., 0., 1.], [3., -1., 2.]] {
        let r = maximize(
            &objective,
            &start,
            description(ObjectiveKind::Likelihood),
            LbfgsOptions::new(300, 3000)
                .unwrap()
                .with_score_tolerance(2e-6)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            r.termination,
            OptimizationTermination::ScoreTolerance,
            "{r:?}"
        );
        let c = r.candidate.unwrap();
        assert!((c.position[0] - 1.).abs() < 2e-6);
        assert!((c.position[1] - 2.).abs() < 2e-6);
        assert!((c.position[2].exp().powi(2) - 14. / 5.).abs() < 2e-6);
        let expected = -2.5 * ((14_f64 / 5.).ln() + 1. + std::f64::consts::TAU.ln());
        assert!((c.value - expected).abs() < 1e-10);
        let mut physical = [0.; 3];
        let mut scratch = vec![0.; objective.layout().scratch_dimension()];
        objective
            .layout()
            .constrain_values(&c.position, &mut physical, &mut scratch)
            .unwrap();
        assert!((physical[2] - (14_f64 / 5.).sqrt()).abs() < 1e-6);
    }
    // The transformed sampling density has a different mode: variance RSS/(n-1).
    let target = regression()
        .select_likelihood()
        .unwrap()
        .into_target()
        .unwrap();
    let r = maximize(
        &target,
        &[0., 0., 1.],
        description(ObjectiveKind::Other),
        LbfgsOptions::new(300, 3000)
            .unwrap()
            .with_score_tolerance(2e-6)
            .unwrap(),
    )
    .unwrap();
    assert_eq!(r.termination, OptimizationTermination::ScoreTolerance);
    assert!((r.candidate.unwrap().position[2].exp().powi(2) - 14. / 4.).abs() < 2e-6);
}
#[test]
fn posterior_mode_uses_declared_measure_and_prior_selection() {
    let mut b = ModelBuilder::new();
    let mean = b.real("mean").unwrap();
    let zero = b.constant(0.).unwrap();
    let one = b.constant(1.).unwrap();
    let prior = normal(&mut b, mean.expr(), zero, one).unwrap();
    b.prior("mean prior", prior).unwrap();
    for (i, y) in [1., 2., 3.].into_iter().enumerate() {
        let y = b.constant(y).unwrap();
        let l = normal(&mut b, y, mean.expr(), one).unwrap();
        b.likelihood(&format!("y{i}"), l).unwrap();
    }
    let objective = b
        .compile()
        .unwrap()
        .select_posterior()
        .unwrap()
        .into_objective()
        .unwrap();
    let r = maximize(
        &objective,
        &[0.],
        description(ObjectiveKind::Posterior),
        LbfgsOptions::new(100, 1000).unwrap(),
    )
    .unwrap();
    assert_eq!(r.termination, OptimizationTermination::ScoreTolerance);
    assert!((r.candidate.unwrap().position[0] - 1.5).abs() < 1e-12);
}

#[test]
fn batch_fits_export_named_estimates_and_match_analytic_glm_solutions() {
    use alea::{
        distributions::modeling::regression::*, model::prediction::NamedOutput, optimize::NamedFit,
    };
    for family in 0..3 {
        let mut b = ModelBuilder::new();
        let a = b.real("intercept").unwrap();
        let slope = b.real("slope").unwrap();
        let beta = [a.expr(), slope.expr()];
        let (factor, start, expected) = match family {
            0 => {
                let sigma = b.positive("sigma").unwrap();
                let x = [-2., -1., 0., 1., 2.];
                let design = x.iter().flat_map(|&x| [1., x]).collect();
                let y = x
                    .iter()
                    .zip([1., -2., 2., -2., 1.])
                    .map(|(&x, e)| 1. + 2. * x + e)
                    .collect();
                (
                    normal_regression(
                        &mut b,
                        &beta,
                        &sigma,
                        RegressionData::new(2, design, y).unwrap(),
                    )
                    .unwrap(),
                    vec![0., 0., 1.],
                    vec![1., 2., (14_f64 / 5.).sqrt()],
                )
            }
            1 => {
                let design = [-1., -1., -1., -1., 1., 1., 1., 1.]
                    .iter()
                    .flat_map(|&x| [1., x])
                    .collect();
                (
                    bernoulli_logit_regression(
                        &mut b,
                        &beta,
                        RegressionData::new(
                            2,
                            design,
                            vec![false, false, false, true, true, true, true, false],
                        )
                        .unwrap(),
                    )
                    .unwrap(),
                    vec![0., 0.],
                    vec![0., 3_f64.ln()],
                )
            }
            _ => {
                let design = [-1., -1., -1., -1., 1., 1., 1., 1.]
                    .iter()
                    .flat_map(|&x| [1., x])
                    .collect();
                let data = RegressionData::new(2, design, vec![2, 4, 2, 4, 6, 8, 6, 8])
                    .unwrap()
                    .with_offsets(vec![2_f64.ln(); 8])
                    .unwrap();
                (
                    poisson_log_regression(&mut b, &beta, data).unwrap(),
                    vec![0., 0.],
                    vec![
                        0.5 * (1.5_f64.ln() + 3.5_f64.ln()),
                        0.5 * (3.5_f64 / 1.5).ln(),
                    ],
                )
            }
        };
        b.likelihood("observations", factor).unwrap();
        let model = b.compile().unwrap();
        assert_eq!(model.expression_count(), start.len() + 1);
        let parameters = NamedOutput::new(model.coordinates(), model.layout().clone(), ()).unwrap();
        let objective = model.select_likelihood().unwrap().into_objective().unwrap();
        let r = maximize(
            &objective,
            &start,
            description(ObjectiveKind::Likelihood),
            LbfgsOptions::new(500, 5000)
                .unwrap()
                .with_score_tolerance(3e-6)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            r.termination,
            OptimizationTermination::ScoreTolerance,
            "family={family}: {r:?}"
        );
        let fit = NamedFit::new(parameters, r).unwrap();
        for (a, b) in fit.estimates().unwrap().iter().zip(expected) {
            assert!((a - b).abs() < 3e-6, "family={family}: {a} != {b}");
        }
        assert_eq!(fit.estimate("intercept"), Some(fit.estimates().unwrap()[0]));
    }
}
