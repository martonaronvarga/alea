//! Regression for nested AD of mutable tensor intermediates (not scalarized).
use alea::{
    autodiff::autodiff::AutodiffError,
    capability::{PotentialHessianDerivative, PotentialHessianVector},
    target::LogDensityGradient,
};
use alea_enzyme_conformance::tensor_matrix;

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12 * (1. + expected.abs()),
        "{actual} != {expected}"
    );
}

#[test]
fn matrix_density_score_curvature_and_third_match_independent_quadratic() {
    let target = tensor_matrix::target();
    for [a, b] in [[0.3, -0.4], [0., 0.], [-10., 1.], [2., 0.5]] {
        let mut g = [41.; 2];
        let mut h = [41.; 4];
        let mut t = [41.; 8];
        let q = [a, b];
        close(
            target.logp_grad(&q, &mut g).unwrap(),
            -0.5 * ((a + 0.5 * b).powi(2) + (-0.5 * a + 1.5 * b).powi(2)),
        );
        close(g[0], -1.25 * a + 0.25 * b);
        close(g[1], 0.25 * a - 2.5 * b);
        target
            .potential_hessian_and_derivatives(&q, &mut h, &mut t)
            .unwrap();
        for (actual, expected) in h.into_iter().zip([1.25, -0.25, -0.25, 2.5]) {
            close(actual, expected);
        }
        for value in t {
            close(value, 0.);
        }
        target.potential_hvp(&q, &[0.4, -0.2], &mut g).unwrap();
        close(g[0], 0.55);
        close(g[1], -0.6);
    }
}

#[test]
fn invalid_tensor_states_remain_typed_errors_and_do_not_poison_scratch() {
    let target = tensor_matrix::target();
    let mut g = [41.; 2];
    let mut h = [41.; 4];
    let mut t = [41.; 8];
    for q in [[f64::NAN, 0.], [f64::MAX, f64::MAX]] {
        assert!(matches!(
            target.logp_grad(&q, &mut g),
            Err(AutodiffError::NonFinite)
        ));
        assert!(matches!(
            target.potential_hvp(&q, &[1., 0.], &mut g),
            Err(AutodiffError::NonFinite)
        ));
        assert!(matches!(
            target.potential_hessian_and_derivatives(&q, &mut h, &mut t),
            Err(AutodiffError::NonFinite)
        ));
        target.logp_grad(&[0.3, -0.4], &mut g).unwrap();
        target
            .potential_hessian_and_derivatives(&[0.3, -0.4], &mut h, &mut t)
            .unwrap();
        close(g[0], -0.475);
        close(h[0], 1.25);
    }
    let allocations = allocation_counter::measure(|| {
        for _ in 0..8 {
            target.logp_grad(&[0.3, -0.4], &mut g).unwrap();
            target
                .potential_hvp(&[0.3, -0.4], &[1., 0.], &mut g)
                .unwrap();
            target
                .potential_hessian_and_derivatives(&[0.3, -0.4], &mut h, &mut t)
                .unwrap();
        }
    });
    assert_eq!(allocations.count_total, 0);
}
