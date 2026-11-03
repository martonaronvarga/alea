//! Independent derivatives and directional laws for lazy branches and scans.
use alea::{
    autodiff::autodiff::AutodiffError, capability::PotentialHessianDerivative,
    target::LogDensityGradient,
};
use alea_enzyme_conformance::{tensor_control, tensor_empty, tensor_zero_scan};

#[track_caller]
fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 2e-11 * (1. + expected.abs()),
        "{actual} != {expected}"
    );
}

// Closed form of three steps c <- c*b + row, followed by log(abs(c)).
// This is an independent deterministic test oracle, not a model/AD backend.
fn reference([a, b]: [f64; 2], zero_trip: bool) -> (f64, [f64; 2], [f64; 4], [f64; 8]) {
    let (r, d, dd, ddd) = if zero_trip {
        (a, [1., 0.], [0.; 4], [0.; 8])
    } else {
        (
            a * b * b * b - 0.1 * b * b + 0.25 * b + 0.5,
            [b * b * b, 3. * a * b * b - 0.2 * b + 0.25],
            [0., 3. * b * b, 3. * b * b, 6. * a * b - 0.2],
            [0., 0., 0., 6. * b, 0., 6. * b, 6. * b, 6. * a],
        )
    };
    let score = d.map(|v| v / r);
    let mut h = [0.; 4];
    let mut third = [0.; 8];
    for i in 0..2 {
        for j in 0..2 {
            h[2 * i + j] = d[i] * d[j] / (r * r) - dd[2 * i + j] / r;
            for k in 0..2 {
                third[4 * k + 2 * i + j] =
                    (dd[2 * i + k] * d[j] + d[i] * dd[2 * j + k] + dd[2 * i + j] * d[k]) / (r * r)
                        - 2. * d[i] * d[j] * d[k] / (r * r * r)
                        - ddd[4 * k + 2 * i + j] / r;
            }
        }
    }
    (r.abs().ln(), score, h, third)
}

#[test]
fn scan_and_zero_trip_both_branches_match_closed_form_through_third_order() {
    fn check(target: impl PotentialHessianDerivative<Error = AutodiffError>, zero_trip: bool) {
        for q in [[-10., 1.], [2., 0.5], [-0.7, -0.57], [0.4, 0.53]] {
            let (lp, g, h, t) = reference(q, zero_trip);
            let mut score = [41.; 2];
            let mut matrix = [41.; 4];
            let mut third = [41.; 8];
            close(target.logp_grad(&q, &mut score).unwrap(), lp);
            target
                .potential_hessian_and_derivatives(&q, &mut matrix, &mut third)
                .unwrap();
            for (a, b) in score.into_iter().zip(g) {
                close(a, b);
            }
            for (a, b) in matrix.into_iter().zip(h) {
                close(a, b);
            }
            for (a, b) in third.into_iter().zip(t) {
                close(a, b);
            }
            let v = [0.3, -0.7];
            target.potential_hvp(&q, &v, &mut score).unwrap();
            for i in 0..2 {
                close(score[i], h[2 * i] * v[0] + h[2 * i + 1] * v[1]);
            }
        }
    }
    check(tensor_control::target(), false);
    check(tensor_zero_scan::target(), true);
}

#[test]
fn raw_nested_jvp_has_zero_direction_and_linearity_without_spurious_hessian_terms() {
    let q = [-0.7, -0.57];
    let v = [0.3, -0.7];
    let (_, _, h, t) = reference(q, false);
    for w in [[0., 0.], [1., 0.], [0., 1.], [0.4, -0.2]] {
        let mut hv = [41.; 2];
        let mut derivative = [41.; 2];
        tensor_control::hvp_position_jvp_for_test(&q, &v, &w, &mut hv, &mut derivative);
        for i in 0..2 {
            close(-hv[i], h[2 * i] * v[0] + h[2 * i + 1] * v[1]);
            let mut expected = 0.;
            for k in 0..2 {
                for j in 0..2 {
                    expected += w[k] * t[4 * k + 2 * i + j] * v[j];
                }
            }
            close(-derivative[i], expected);
        }
    }
}

#[test]
fn selected_domain_errors_survive_empty_outputs_and_zero_trip_loops_and_recover() {
    fn check<const D: usize>(
        target: impl PotentialHessianDerivative<Error = AutodiffError>,
        invalid: [f64; D],
        valid: [f64; D],
    ) {
        let mut g = vec![41.; D];
        let mut h = vec![41.; D * D];
        let mut t = vec![41.; D * D * D];
        assert!(matches!(
            target.logp_grad(&invalid, &mut g),
            Err(AutodiffError::NonFinite)
        ));
        assert!(matches!(
            target.potential_hvp(&invalid, &vec![1.; D], &mut g),
            Err(AutodiffError::NonFinite)
        ));
        assert!(matches!(
            target.potential_hessian_and_derivatives(&invalid, &mut h, &mut t),
            Err(AutodiffError::NonFinite)
        ));
        target.logp_grad(&valid, &mut g).unwrap();
        target
            .potential_hessian_and_derivatives(&valid, &mut h, &mut t)
            .unwrap();
        assert!(g.iter().chain(&h).chain(&t).all(|v| v.is_finite()));
    }
    check(tensor_control::target(), [-0.65, 1.], [0.3, -0.4]);
    check(tensor_zero_scan::target(), [0., 1.], [-1., 0.5]);
    check(tensor_empty::target(), [-1.], [0.]);
    let target = tensor_empty::target();
    let mut g = [41.];
    let mut h = [41.];
    let mut t = [41.];
    for q in [[0.], [1.], [10.]] {
        close(target.logp_grad(&q, &mut g).unwrap(), 0.);
        target
            .potential_hessian_and_derivatives(&q, &mut h, &mut t)
            .unwrap();
        assert_eq!((g, h, t), ([0.], [0.], [0.]));
    }
}
