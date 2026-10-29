#![cfg(feature = "faer")]
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
