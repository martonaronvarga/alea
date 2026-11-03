#![cfg(feature = "faer")]
use alea_math::{
    information::{InformationError, InformationMatrix, InformationOptions},
    matrix::{RowMajor, SquareMatrixRef},
};
fn options() -> InformationOptions {
    InformationOptions {
        relative_eigenvalue_cutoff: 1e-12,
        relative_symmetry_tolerance: 64. * f64::EPSILON,
    }
}
fn factor(d: usize, a: &[f64]) -> Result<InformationMatrix, InformationError> {
    InformationMatrix::new(SquareMatrixRef::<RowMajor>::new(d, a).unwrap(), options())
}
fn close(a: f64, b: f64) {
    assert!(
        if b == 0. {
            a.abs() < 1e-13
        } else {
            (a / b - 1.).abs() < 2e-12
        },
        "{a:e} != {b:e}"
    );
}
#[test]
fn solves_match_independent_rational_inverse_and_projected_gram() {
    let a = [4., 2., 0., 2., 3., 1., 0., 1., 2.];
    let reference = [
        5. / 12.,
        -1. / 3.,
        1. / 6.,
        -1. / 3.,
        2. / 3.,
        -1. / 3.,
        1. / 6.,
        -1. / 3.,
        2. / 3.,
    ];
    let f = factor(3, &a).unwrap();
    let covariance = f.covariance().unwrap();
    for (&actual, expected) in covariance.iter().zip(reference) {
        close(actual, expected);
    }
    let j = [1., 0., 2., 0., -3., 1.];
    let projected = f.project(2, &j).unwrap();
    for i in 0..2 {
        for k in 0..2 {
            let mut expected = 0.;
            for p in 0..3 {
                for q in 0..3 {
                    expected += j[i * 3 + p] * reference[p * 3 + q] * j[k * 3 + q];
                }
            }
            close(projected[i * 2 + k], expected);
        }
    }
    assert_eq!(f.dimension(), 3);
    assert_eq!(f.options(), options());
    assert_eq!(f.scale(), 4.);
    assert!(f.reciprocal_condition() > options().relative_eigenvalue_cutoff);
    assert_eq!(f.project(2, &[0.; 6]).unwrap(), vec![0.; 4]);
    assert!(f.project(0, &[]).unwrap().is_empty());
}
#[test]
fn independent_two_by_two_grid_preserves_signs_and_full_rank() {
    for a in [0.25, 1., 4., 100.] {
        for b in [0.125, 2., 16.] {
            for rho in [-0.9, -0.5, 0., 0.5, 0.9] {
                let off = rho * f64::sqrt(a * b);
                let matrix = [a, off, off, b];
                let c = factor(2, &matrix).unwrap().covariance().unwrap();
                let det = a * b - off * off;
                for (&actual, expected) in c.iter().zip([b / det, -off / det, -off / det, a / det])
                {
                    close(actual, expected);
                }
            }
        }
    }
}
#[test]
fn global_scaling_keeps_representable_extreme_covariance() {
    for scale in [1e-300, 1., 1e300] {
        let f = factor(2, &[2. * scale, scale, scale, 2. * scale]).unwrap();
        for (actual, expected) in
            f.covariance()
                .unwrap()
                .into_iter()
                .zip([2. / 3., -1. / 3., -1. / 3., 2. / 3.])
        {
            close(actual * scale, expected);
        }
        close(f.reciprocal_condition(), 1. / 3.);
    }
    for scale in [f64::MIN_POSITIVE, f64::MAX] {
        close(
            factor(1, &[scale]).unwrap().covariance().unwrap()[0],
            1. / scale,
        );
    }
    assert!(matches!(
        factor(1, &[f64::from_bits(1)]).unwrap().covariance(),
        Err(InformationError::Numerical)
    ));
}
#[test]
fn invalid_singular_indefinite_asymmetric_and_ill_conditioned_are_not_repaired() {
    for matrix in [[1., 2., 2., 1.], [0., 0., 0., 0.], [-1., 0., 0., 1.]] {
        assert!(matches!(
            factor(2, &matrix),
            Err(InformationError::NotPositiveDefinite)
        ));
    }
    assert!(factor(2, &[1., 1., 1., 1.]).is_err());
    assert!(matches!(
        factor(2, &[1., 0., 0., 1e-14]),
        Err(InformationError::IllConditioned { .. })
    ));
    for scale in [1e-300, 1., 1e300] {
        assert!(matches!(
            factor(2, &[scale, scale * 1e-5, -scale * 1e-5, scale]),
            Err(InformationError::Asymmetric { .. })
        ));
    }
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            factor(2, &[1., value, 0., 1.]),
            Err(InformationError::NonFinite)
        ));
    }
    assert!(matches!(factor(0, &[]), Err(InformationError::Invalid)));
    for cutoff in [0., 1., -1., f64::NAN] {
        let mut o = options();
        o.relative_eigenvalue_cutoff = cutoff;
        assert!(InformationMatrix::new(SquareMatrixRef::new(1, &[1.]).unwrap(), o).is_err());
    }
    for tolerance in [-1., 1., f64::NAN] {
        let mut o = options();
        o.relative_symmetry_tolerance = tolerance;
        assert!(o.validate().is_err());
    }
}
#[test]
fn symmetry_averaging_is_explicit_and_projection_never_invents_zero_variance() {
    let matrix = [3., 1. + 2. * f64::EPSILON, 1., 2.];
    assert!(factor(2, &matrix).is_ok());
    let mut exact = options();
    exact.relative_symmetry_tolerance = 0.;
    assert!(matches!(
        InformationMatrix::new(SquareMatrixRef::new(2, &matrix).unwrap(), exact),
        Err(InformationError::Asymmetric { .. })
    ));
    let f = factor(1, &[1e300]).unwrap();
    assert!(matches!(
        f.project(1, &[1e-300]),
        Err(InformationError::Numerical)
    ));
    assert!(matches!(
        f.project(1, &[1e308]),
        Err(InformationError::Numerical)
    ));
    assert!(matches!(
        f.project(2, &[1.]),
        Err(InformationError::Invalid)
    ));
    assert!(matches!(
        f.project(usize::MAX, &[]),
        Err(InformationError::Invalid)
    ));
    assert!(matches!(
        f.project(1, &[f64::NAN]),
        Err(InformationError::NonFinite)
    ));
}

#[test]
fn near_zero_cross_curvature_uses_matrix_not_entry_relative_error() {
    // Captured generated Gaussian mean/log-scale HVPs. Each cross derivative
    // agrees with its independent analytic value within 3e-16, while their
    // entry-relative discrepancy is ~1e-9 due to cancellation near the MLE.
    let matrix = [
        2.0000000007227503,
        2.2114950981588777e-8,
        2.2114950957590583e-8,
        6.000000002168251,
    ];
    assert!(factor(2, &matrix).is_ok());
    let mut exact = options();
    exact.relative_symmetry_tolerance = 0.;
    assert!(matches!(
        InformationMatrix::new(SquareMatrixRef::new(2, &matrix).unwrap(), exact),
        Err(InformationError::Asymmetric { .. })
    ));
    let mut bad = matrix;
    bad[1] += 1e-9;
    assert!(matches!(
        factor(2, &bad),
        Err(InformationError::Asymmetric { .. })
    ));
}
