use alea_distributions::modeling::regression::{
    RegressionData,
    diagnostics::{DirectionAssessment, RecessionKind},
};
fn witness(
    a: DirectionAssessment,
) -> alea_distributions::modeling::regression::diagnostics::RecessionWitness {
    let DirectionAssessment::Witness(w) = a else {
        panic!("expected witness")
    };
    w
}
#[test]
fn complete_quasi_and_null_directions_are_distinguished_exactly() {
    let data = RegressionData::new(1, vec![-1., 1.], vec![false, true])
        .unwrap()
        .with_offsets(vec![1000., -1000.])
        .unwrap();
    let w = witness(data.bernoulli_recession(&[1.]).unwrap());
    assert_eq!(w.kind(), RecessionKind::CompleteSeparation);
    assert_eq!(w.improving_rows(), 2);
    assert_eq!(w.flat_rows(), 0);
    let data = RegressionData::new(
        2,
        vec![-1., -1., 1., 1., 1., -1., -1., 1.],
        vec![false, true, true, false],
    )
    .unwrap();
    let w = witness(data.bernoulli_recession(&[1., 1.]).unwrap());
    assert_eq!(w.kind(), RecessionKind::QuasiCompleteSeparation);
    assert_eq!(w.improving_rows(), 2);
    assert_eq!(w.flat_rows(), 2);
    assert!(matches!(
        data.bernoulli_recession(&[0., 0.]).unwrap(),
        DirectionAssessment::NotWitness {
            improving_rows: 0,
            ..
        }
    ));
    assert!(data.bernoulli_recession(&[f64::NAN, 1.]).is_err());
    assert!(data.bernoulli_recession(&[1.]).is_err());
}
#[test]
fn axis_screen_is_deliberately_incomplete_and_does_not_claim_overlap() {
    let data = RegressionData::new(
        2,
        vec![1., -0.5, -0.5, 1., -1., 0.5, 0.5, -1.],
        vec![true, true, false, false],
    )
    .unwrap();
    assert!(data.bernoulli_coordinate_recession().unwrap().is_none());
    assert_eq!(
        witness(data.bernoulli_recession(&[1., 1.]).unwrap()).kind(),
        RecessionKind::CompleteSeparation
    );
    let overlap = RegressionData::new(1, vec![1., 1.], vec![true, false]).unwrap();
    assert!(overlap.bernoulli_coordinate_recession().unwrap().is_none());
}
#[test]
fn poisson_positive_count_equalities_and_zero_count_boundary_are_required() {
    let data = RegressionData::new(2, vec![1., 0., 1., 1.], vec![2, 0]).unwrap();
    let w = witness(data.poisson_recession(&[0., -1.]).unwrap());
    assert_eq!(w.kind(), RecessionKind::PoissonBoundary);
    assert_eq!(w.direction(), &[0., -1.]);
    assert_eq!(w.flat_rows(), 1);
    assert!(matches!(
        data.poisson_recession(&[-1., 0.]).unwrap(),
        DirectionAssessment::NotWitness {
            violating_rows: 1,
            ..
        }
    ));
    assert!(
        RegressionData::new(1, vec![1., 1.], vec![0, 0])
            .unwrap()
            .poisson_coordinate_recession()
            .unwrap()
            .is_some()
    );
    // With no intercept and predictors -1,+1, all-zero responses have finite
    // maximizer beta=0: log L=-exp(-beta)-exp(beta). No automatic all-zero flag.
    let finite = RegressionData::new(1, vec![-1., 1.], vec![0, 0]).unwrap();
    assert!(finite.poisson_coordinate_recession().unwrap().is_none());
}
#[cfg(feature = "faer")]
#[test]
fn gaussian_regular_rank_deficient_and_vanishing_residuals_are_separate() {
    use alea_distributions::modeling::regression::diagnostics::GaussianIssue;
    let x = [-2., -1., 0., 1., 2.];
    let design: Vec<_> = x.iter().flat_map(|&x| [1., x]).collect();
    let offset = [10., 11., 12., 13., 14.];
    let y: Vec<_> = x
        .iter()
        .zip([1., -2., 2., -2., 1.])
        .zip(offset)
        .map(|((&x, e), o)| 1. + 2. * x + e + o)
        .collect();
    let assessment = RegressionData::new(2, design.clone(), y)
        .unwrap()
        .with_offsets(offset.to_vec())
        .unwrap()
        .gaussian_assessment(1e-12, 1e-12)
        .unwrap();
    assert!(assessment.issues.is_empty());
    assert!((assessment.scale_candidate.unwrap() - (14_f64 / 5.).sqrt()).abs() < 1e-12);
    assert_eq!(assessment.least_squares.residual_degrees_of_freedom, 3);
    let perfect = RegressionData::new(2, design, x.iter().map(|&x| 1. + 2. * x).collect())
        .unwrap()
        .gaussian_assessment(1e-12, 1e-12)
        .unwrap();
    assert!(perfect.issues.contains(&GaussianIssue::VanishingResidual));
    assert!(perfect.scale_candidate.is_none());
    let rank = RegressionData::new(2, vec![1., 1., 2., 2., 3., 3.], vec![2., 4., 7.])
        .unwrap()
        .gaussian_assessment(1e-12, 1e-12)
        .unwrap();
    assert_eq!(rank.issues, [GaussianIssue::NumericalRankDeficiency]);
    assert!(rank.scale_candidate.is_some());
    let both = RegressionData::new(2, vec![1., 1., 2., 2., 3., 3.], vec![2., 4., 6.])
        .unwrap()
        .gaussian_assessment(1e-12, 1e-12)
        .unwrap();
    assert_eq!(both.issues.len(), 2);
    assert!(both.scale_candidate.is_none());
    assert!(
        RegressionData::new(1, vec![1.], vec![f64::NAN])
            .unwrap()
            .gaussian_assessment(1e-12, 1e-12)
            .is_err()
    );
}

#[cfg(feature = "faer")]
#[test]
fn gaussian_subnormal_scale_is_reported_without_a_false_perfect_fit() {
    use alea_distributions::modeling::regression::diagnostics::GaussianIssue;
    let tiny = f64::from_bits(1);
    let stable = RegressionData::new(0, vec![], vec![tiny; 4])
        .unwrap()
        .gaussian_assessment(1e-12, 1e-12)
        .unwrap();
    assert_eq!(stable.scale_candidate, Some(tiny));
    assert!(stable.issues.is_empty());
    let underflow = RegressionData::new(0, vec![], vec![tiny, 0., 0., 0.])
        .unwrap()
        .gaussian_assessment(1e-12, 1e-12)
        .unwrap();
    assert_eq!(underflow.issues, [GaussianIssue::ScaleNotRepresentable]);
    assert!(underflow.scale_candidate.is_none());
}
