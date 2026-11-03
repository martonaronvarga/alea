#![cfg(feature = "faer")]
use alea_distributions::modeling::regression::{
    RegressionData,
    diagnostics::{GaussianIntervalError, GaussianIntervalOptions},
};
use alea_math::least_squares::LeastSquaresError;
fn options(level: f64) -> GaussianIntervalOptions {
    GaussianIntervalOptions {
        confidence_level: level,
        relative_rank_tolerance: 1e-12,
        relative_residual_tolerance: 1e-12,
        quantile_relative_tolerance: 1e-8,
    }
}
fn data(scale: f64) -> RegressionData<f64> {
    let x: Vec<_> = (-2..=3).flat_map(|x| [1., x as f64]).collect();
    let offsets = vec![0.25, -0.5, 0.75, -1., 1.25, -1.5];
    let y = (-2..=3)
        .zip([1., -2., 1., 1., -2., 1.])
        .zip(&offsets)
        .map(|((x, e), o)| (1.5 - 0.75 * x as f64 + e + o) * scale)
        .collect();
    RegressionData::new(2, x, y)
        .unwrap()
        .with_offsets(offsets.into_iter().map(|o| o * scale).collect())
        .unwrap()
}
#[test]
fn student_intervals_use_residual_df_and_preserve_ml_scale_and_offsets() {
    // Independent SciPy t.isf(.025,4), also standard t-table reference.
    let critical = 2.7764451051977987;
    for scale in [0.25, 1., 4.] {
        let data = data(scale);
        let before = data.gaussian_assessment(1e-12, 1e-12).unwrap();
        let result = data.gaussian_coefficient_intervals(options(0.95)).unwrap();
        assert_eq!(result.options, options(0.95));
        assert_eq!(
            result.assessment.least_squares.residual_degrees_of_freedom,
            4
        );
        assert_eq!(result.assessment.least_squares.numerical_rank, 2);
        assert_eq!(result.assessment.scale_candidate, before.scale_candidate);
        assert!((result.residual_variance / (3. * scale * scale) - 1.).abs() < 1e-12);
        assert!((result.critical_value - critical).abs() < 2e-10);
        for (i, beta) in [1.5 * scale, -0.75 * scale].into_iter().enumerate() {
            let variance = [57. / 105., 18. / 105.][i] * scale * scale;
            assert!((result.covariance[3 * i] / variance - 1.).abs() < 1e-12);
            let width = critical * variance.sqrt();
            assert!((result.bounds[i][0] - (beta - width)).abs() < 1e-9);
            assert!((result.bounds[i][1] - (beta + width)).abs() < 1e-9);
        }
        assert!(result.covariance[1] < 0.);
        assert_eq!(
            result.covariance[1].to_bits(),
            result.covariance[2].to_bits()
        );
    }
}
#[test]
fn invalid_policy_and_unqualified_or_unrepresentable_fits_return_errors() {
    for bad in [f64::NAN, f64::INFINITY, -1., 0., 1., 2.] {
        for field in 0..4 {
            let mut o = options(0.95);
            match field {
                0 => o.confidence_level = bad,
                1 => o.relative_rank_tolerance = bad,
                2 => o.relative_residual_tolerance = bad,
                _ => o.quantile_relative_tolerance = bad,
            }
            assert!(matches!(
                data(1.).gaussian_coefficient_intervals(o),
                Err(GaussianIntervalError::Policy)
            ));
        }
    }
    let rank = RegressionData::new(2, vec![1.; 6], vec![0., 1., 2.]).unwrap();
    assert!(matches!(
        rank.gaussian_coefficient_intervals(options(0.95)),
        Err(GaussianIntervalError::LeastSquares(
            LeastSquaresError::RankDeficient
        ))
    ));
    for (p, x, y) in [
        (1, vec![1.; 3], vec![2.; 3]),
        (2, vec![1., 0., 1., 1.], vec![1., 2.]),
        (0, vec![], vec![1., 2.]),
    ] {
        assert!(matches!(
            RegressionData::new(p, x, y)
                .unwrap()
                .gaussian_coefficient_intervals(options(0.95)),
            Err(GaussianIntervalError::Unqualified)
        ));
    }
    for scale in [1e-200, 1e200] {
        assert!(
            data(scale)
                .gaussian_coefficient_intervals(options(0.95))
                .is_err()
        );
    }
    assert!(matches!(
        data(1.).gaussian_coefficient_intervals(options(f64::from_bits(1))),
        Err(GaussianIntervalError::Numerical)
    ));
    let error = rank
        .gaussian_coefficient_intervals(options(0.95))
        .unwrap_err();
    assert!(
        std::error::Error::source(&error)
            .unwrap()
            .downcast_ref::<LeastSquaresError>()
            .is_some()
    );
    // Invalid intervals must not mutate the supplied data or subsequent assessment.
    assert_eq!(
        rank.gaussian_assessment(1e-12, 1e-12)
            .unwrap()
            .least_squares
            .numerical_rank,
        1
    );
}

#[test]
fn critical_values_match_independent_scipy_references_over_residual_df() {
    // SciPy1.17.1 t.isf((1-level)/2,df), analysis Nix shell; not Statrs output.
    let references = [
        (
            1,
            [6.313751514675045, 12.706204736174694, 63.656741162871526],
        ),
        (
            4,
            [2.131846786326651, 2.7764451051977934, 4.604094871349992],
        ),
        (
            9,
            [1.8331129326562374, 2.262157162798205, 3.249835541592126],
        ),
        (
            29,
            [1.699127026533498, 2.045229642132703, 2.756385903670605],
        ),
    ];
    for (df, values) in references {
        let mut y = vec![0.; df + 1];
        y[0] = -1.;
        y[1] = 1.;
        let data = RegressionData::new(1, vec![1.; df + 1], y).unwrap();
        for (level, expected) in [0.9, 0.95, 0.99].into_iter().zip(values) {
            let result = data.gaussian_coefficient_intervals(options(level)).unwrap();
            assert!((result.critical_value - expected).abs() <= 2e-9 * (1. + expected));
        }
    }
}

#[test]
fn rounded_endpoint_collapse_and_quantile_replay_failure_are_not_successes() {
    let y = (0..64)
        .map(|i| 1e16 + if i % 2 == 0 { 2. } else { -2. })
        .collect();
    let data = RegressionData::new(1, vec![1.; 64], y).unwrap();
    let mut policy = options(0.95);
    policy.relative_residual_tolerance = 1e-20;
    assert!(matches!(
        data.gaussian_coefficient_intervals(policy),
        Err(GaussianIntervalError::Numerical)
    ));
    let mut exercised = false;
    use statrs::distribution::{ContinuousCDF, StudentsT};
    let law = StudentsT::new(0., 1., 4.).unwrap();
    for level in [0.9, 0.95, 0.99] {
        let tail = (1. - level) * 0.5;
        let error = ((law.sf(-law.inverse_cdf(tail)) - tail) / tail).abs();
        if error > 0. {
            let mut policy = options(level);
            policy.quantile_relative_tolerance = error * 0.5;
            assert!(matches!(
                crate::data(1.).gaussian_coefficient_intervals(policy),
                Err(GaussianIntervalError::Numerical)
            ));
            exercised = true;
        }
    }
    assert!(
        exercised,
        "pinned quantile backend must exercise nonzero replay error"
    );
}
