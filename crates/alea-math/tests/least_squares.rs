#![cfg(feature = "faer")]
use alea_math::least_squares::solve_with_residual_covariance;
use alea_math::least_squares::{LeastSquaresError, solve};
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 2e-12 * (1. + b.abs()), "{a} != {b}");
}
#[test]
fn tall_design_matches_analytic_coefficients_singular_values_and_residuals() {
    let x = [-2., -1., 0., 1., 2.];
    let design: Vec<_> = x.iter().flat_map(|&x| [1., x]).collect();
    let response: Vec<_> = x
        .iter()
        .zip([1., -2., 2., -2., 1.])
        .map(|(&x, e)| 1. + 2. * x + e)
        .collect();
    let fit = solve(5, 2, &design, &response, 1e-12).unwrap();
    close(fit.coefficients[0], 1.);
    close(fit.coefficients[1], 2.);
    close(fit.residual_norm, 14_f64.sqrt());
    assert_eq!(fit.numerical_rank, 2);
    assert_eq!(fit.residual_degrees_of_freedom, 3);
    close(fit.scaled_singular_values[0], 10_f64.sqrt() / 2.);
    close(fit.scaled_singular_values[1], 5_f64.sqrt() / 2.);
}
#[test]
fn wide_and_rank_deficient_solutions_are_minimum_norm_in_retained_space() {
    let fit = solve(2, 3, &[1., 0., 1., 0., 1., 1.], &[1., 2.], 1e-12).unwrap();
    for (a, b) in fit.coefficients.iter().zip([0., 1., 1.]) {
        close(*a, b);
    }
    assert_eq!(fit.numerical_rank, 2);
    close(fit.residual_norm, 0.);
    let fit = solve(3, 2, &[1., 1., 2., 2., 3., 3.], &[2., 4., 7.], 1e-12).unwrap();
    for &a in &fit.coefficients {
        close(a, 31. / 28.);
    }
    assert_eq!(fit.numerical_rank, 1);
    close(fit.residual_norm, (5_f64 / 14.).sqrt());
    let zero = solve(3, 2, &[0.; 6], &[1., 2., 3.], 1e-12).unwrap();
    assert_eq!(zero.numerical_rank, 0);
    assert_eq!(zero.coefficients, [0., 0.]);
    close(zero.residual_norm, 14_f64.sqrt());
}
#[test]
fn cutoff_is_explicit_and_scaling_does_not_square_the_condition_number() {
    let coarse = solve(2, 2, &[1., 0., 0., 1e-12], &[1., 1e-12], 1e-10).unwrap();
    assert_eq!(coarse.numerical_rank, 1);
    close(coarse.coefficients[1], 0.);
    let fine = solve(2, 2, &[1., 0., 0., 1e-12], &[1., 1e-12], 1e-14).unwrap();
    assert_eq!(fine.numerical_rank, 2);
    close(fine.coefficients[1], 1.);
    for scale in [1e-300, 1e300] {
        let f = solve(2, 2, &[scale, 0., 0., scale], &[scale, scale], 1e-12).unwrap();
        close(f.coefficients[0], 1.);
        close(f.coefficients[1], 1.);
    }
    let f = solve(2, 2, &[1e-200, 0., 0., 1e-200], &[1., 1.], 1e-12).unwrap();
    assert!((f.coefficients[0] / 1e200 - 1.).abs() < 1e-14);
}
#[test]
fn invalid_and_unrepresentable_problems_fail_without_repair() {
    assert_eq!(
        solve(0, 1, &[], &[], 1e-12).unwrap_err(),
        LeastSquaresError::Shape
    );
    assert!(solve(2, 1, &[1.], &[1., 2.], 1e-12).is_err());
    assert!(solve(1, 1, &[1.], &[f64::NAN], 1e-12).is_err());
    assert!(solve(1, 1, &[1.], &[1.], 0.).is_err());
    assert!(solve(1, 1, &[1e-200], &[1e200], 1e-12).is_err());
    let f = solve(2, 0, &[], &[3., 4.], 1e-12).unwrap();
    assert_eq!(f.numerical_rank, 0);
    close(f.residual_norm, 5.);
}

#[test]
fn residual_covariance_reuses_svd_and_rescales_before_squaring() {
    // Exact coefficients [1.5,-.75], RSS12 and df4; nonzero covariance.
    for scale in [1., 1e-250, 1e250] {
        let x: Vec<_> = (-2..=3).flat_map(|x| [scale, x as f64 * scale]).collect();
        let y: Vec<_> = (-2..=3)
            .zip([1., -2., 1., 1., -2., 1.])
            .map(|(x, e)| (1.5 - 0.75 * x as f64 + e) * scale)
            .collect();
        let original = solve(6, 2, &x, &y, 1e-12).unwrap();
        let (fit, covariance) = solve_with_residual_covariance(6, 2, &x, &y, 1e-12, 4).unwrap();
        assert_eq!(
            original
                .coefficients
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            fit.coefficients
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            original.residual_norm.to_bits(),
            fit.residual_norm.to_bits()
        );
        assert_eq!(original.scaled_singular_values, fit.scaled_singular_values);
        assert_eq!(fit.numerical_rank, 2);
        assert_eq!(fit.residual_degrees_of_freedom, 4);
        for (actual, expected) in
            covariance
                .into_iter()
                .zip([57. / 105., -9. / 105., -9. / 105., 18. / 105.])
        {
            close(actual, expected);
        }
    }
}

#[test]
fn covariance_never_truncates_rank_or_replaces_unrepresentable_values() {
    assert_eq!(
        solve_with_residual_covariance(3, 2, &[1.; 6], &[0., 1., 2.], 1e-12, 1).unwrap_err(),
        LeastSquaresError::RankDeficient
    );
    assert_eq!(
        solve_with_residual_covariance(1, 2, &[1., 2.], &[1.], 1e-12, 1).unwrap_err(),
        LeastSquaresError::RankDeficient
    );
    assert_eq!(
        solve_with_residual_covariance(1, 1, &[1.], &[1.], 1e-12, 0).unwrap_err(),
        LeastSquaresError::Shape
    );
    for scale in [1e-200, 1e200] {
        assert_eq!(
            solve_with_residual_covariance(3, 1, &[scale; 3], &[-1., 0., 1.], 1e-12, 2)
                .unwrap_err(),
            LeastSquaresError::Numerical
        );
    }
    assert!(matches!(
        solve_with_residual_covariance(3, 1, &[1.; 3], &[f64::NAN, 0., 1.], 1e-12, 2),
        Err(LeastSquaresError::NonFinite)
    ));
    let (_, empty) = solve_with_residual_covariance(2, 0, &[], &[3., 4.], 1e-12, 2).unwrap();
    assert!(empty.is_empty());
    let (_, zero) = solve_with_residual_covariance(2, 1, &[1., 1.], &[0., 0.], 1e-12, 1).unwrap();
    assert_eq!(zero, [0.]); // Genuine zero residual, not numerical repair.
    if let Ok(divisor) = usize::try_from((1_u64 << 53) + 1) {
        assert_eq!(
            solve_with_residual_covariance(1, 1, &[1.], &[1.], 1e-12, divisor).unwrap_err(),
            LeastSquaresError::Shape
        );
    }
}

#[test]
fn covariance_factor_errors_preserve_the_original_pivot_source() {
    let original = alea_math::matrix::MatrixError::NotPositiveDefinite { index: 1 };
    let error = LeastSquaresError::from(original);
    assert_eq!(error, LeastSquaresError::Covariance(original));
    assert_eq!(
        std::error::Error::source(&error)
            .unwrap()
            .downcast_ref::<alea_math::matrix::MatrixError>(),
        Some(&original)
    );
}
