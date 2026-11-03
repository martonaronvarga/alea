#![cfg(feature = "faer")]

#[test]
fn fitting_does_not_access_process_global_parallelism() {
    // This integration-test binary contains only this test. Disabling global
    // access would panic in any high-level Faer operation using that policy.
    faer::disable_global_parallelism();
    let metric = alea_math::fisher::fit_low_rank(
        &[-2.0, 0.0, 2.0, 0.0, 0.0, -1.0, 0.0, 1.0],
        &[0.5, 0.0, -0.5, 0.0, 0.0, 1.0, 0.0, -1.0],
        &[1.0, 1.0],
        1e-5,
        2.0,
        2,
    )
    .unwrap();
    assert_eq!(metric.rank(), 1);
    let information = alea_math::information::InformationMatrix::new(
        alea_math::matrix::SquareMatrixRef::new(2, &[2., 0.5, 0.5, 1.]).unwrap(),
        alea_math::information::InformationOptions {
            relative_eigenvalue_cutoff: 1e-12,
            relative_symmetry_tolerance: 64. * f64::EPSILON,
        },
    )
    .unwrap();
    assert_eq!(information.covariance().unwrap().len(), 4);
    assert_eq!(information.project(1, &[1., 2.]).unwrap().len(), 1);
    let (_, covariance) = alea_math::least_squares::solve_with_residual_covariance(
        3,
        2,
        &[1., 0., 1., 1., 1., 2.],
        &[0., 2., 1.],
        1e-12,
        1,
    )
    .unwrap();
    assert_eq!(covariance.len(), 4);
    // Fitting must not silently re-enable or replace the caller's policy either.
    assert!(std::panic::catch_unwind(faer::get_global_parallelism).is_err());
}
