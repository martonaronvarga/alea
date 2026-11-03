//! Native distribution graph -> generated no-Jacobian HVP -> physical covariance.
#![cfg(feature = "faer")]
#[path = "../models.rs"]
mod models;
use alea::{
    capability::PotentialHessianVector,
    math::information::InformationOptions,
    model::prediction::NamedOutput,
    optimize::{
        LbfgsOptions, NamedFit, ObjectiveDescription, ObjectiveKind, OptimizationTermination,
        information::{ObservedInformationError, observed_information},
        maximize,
    },
};
use alea_enzyme_conformance::selected_likelihood;

#[test]
fn native_gaussian_likelihood_has_unit_physical_information_without_prior_or_jacobian() {
    let (_, model) = models::all()
        .unwrap()
        .into_iter()
        .find(|(name, _)| *name == "selected_likelihood")
        .unwrap();
    let names = NamedOutput::new(model.coordinates(), model.layout().clone(), ()).unwrap();
    let target = selected_likelihood::composition::objective();
    let result = maximize(
        &target,
        &[0.3],
        ObjectiveDescription {
            kind: ObjectiveKind::Likelihood,
            measure: "Lebesgue in positive normal location".into(),
            model: "selected native Gaussian location".into(),
            data: "single observation y=2, known sigma=1".into(),
            source: "generated native no-Jacobian objective".into(),
        },
        LbfgsOptions::new(200, 2000)
            .unwrap()
            .with_score_tolerance(1e-6)
            .unwrap(),
    )
    .unwrap();
    assert_eq!(result.termination, OptimizationTermination::ScoreTolerance);
    let fit = NamedFit::new(names, result).unwrap();
    let physical = fit.estimates().unwrap()[0];
    assert!((physical - 2.).abs() < 1e-6);
    let policy = InformationOptions {
        relative_eigenvalue_cutoff: 1e-12,
        relative_symmetry_tolerance: 64. * f64::EPSILON,
    };
    let information = observed_information(&target, &fit, policy).unwrap();
    assert!((information.free_covariance().as_slice()[0] - 0.25).abs() < 1e-6);
    assert!((information.constrained_covariance().values[0] - 1.).abs() < 1e-6);
    assert_eq!(
        information.constrained_covariance().coordinates()[0].name,
        "theta"
    );
    // This graph excludes an opaque, deliberately failing prior. Its sampling
    // adapter still adds a chart Jacobian and must not be substituted here.
    assert!(matches!(
        observed_information(&selected_likelihood::target(), &fit, policy),
        Err(ObservedInformationError::ObjectiveMismatch)
    ));
}

#[test]
fn native_unknown_location_and_scale_match_analytic_observed_information() {
    let (_, model) = models::all()
        .unwrap()
        .into_iter()
        .find(|(name, _)| *name == "likelihood_location_scale")
        .unwrap();
    let names = NamedOutput::new(model.coordinates(), model.layout().clone(), ()).unwrap();
    let target = alea_enzyme_conformance::likelihood_location_scale::composition::objective();
    let result = maximize(
        &target,
        &[0., 0.],
        ObjectiveDescription {
            kind: ObjectiveKind::Likelihood,
            measure: "Lebesgue in normal mean and sigma".into(),
            model: "native Gaussian location and scale".into(),
            data: "three observations -1,0.5,2".into(),
            source: "generated no-Jacobian objective".into(),
        },
        LbfgsOptions::new(300, 3000)
            .unwrap()
            .with_score_tolerance(1e-6)
            .unwrap(),
    )
    .unwrap();
    assert_eq!(result.termination, OptimizationTermination::ScoreTolerance);
    let fit = NamedFit::new(names, result).unwrap();
    assert!((fit.estimate("mu").unwrap() - 0.5).abs() < 1e-6);
    assert!((fit.estimate("sigma").unwrap() - 1.5_f64.sqrt()).abs() < 1e-6);
    let policy = InformationOptions {
        relative_eigenvalue_cutoff: 1e-12,
        relative_symmetry_tolerance: 64. * f64::EPSILON,
    };
    let q = &fit.result().candidate.as_ref().unwrap().position;
    let mut columns = [[0.; 2]; 2];
    target.potential_hvp(q, &[1., 0.], &mut columns[0]).unwrap();
    target.potential_hvp(q, &[0., 1.], &mut columns[1]).unwrap();
    let mu = fit.estimate("mu").unwrap();
    let sigma = fit.estimate("sigma").unwrap();
    let residual_sum = [-1., 0.5, 2.].iter().map(|y| y - mu).sum::<f64>();
    let rss = [-1., 0.5, 2.]
        .iter()
        .map(|y| (y - mu) * (y - mu))
        .sum::<f64>();
    let expected = [
        [3. / (sigma * sigma), 2. * residual_sum / (sigma * sigma)],
        [
            2. * residual_sum / (sigma * sigma),
            2. * rss / (sigma * sigma),
        ],
    ];
    for i in 0..2 {
        for j in 0..2 {
            assert!((columns[j][i] - expected[i][j]).abs() < 5e-13);
        }
    }
    // At the nearly stationary point these cross entries are ~2e-8, with
    // ~2.4e-17 asymmetry. Check accuracy against the actual analytic Hessian,
    // rather than imposing misleading entry-relative precision near zero.
    let information = observed_information(&target, &fit, policy).unwrap();
    for (&actual, expected) in
        information
            .free_covariance()
            .as_slice()
            .iter()
            .zip([0.5, 0., 0., 1. / 6.])
    {
        assert!((actual - expected).abs() < 1e-6);
    }
    for (&actual, expected) in information
        .constrained_covariance()
        .values
        .iter()
        .zip([0.5, 0., 0., 0.25])
    {
        assert!((actual - expected).abs() < 1e-6);
    }
}
