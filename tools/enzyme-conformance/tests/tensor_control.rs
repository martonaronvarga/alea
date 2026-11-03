//! Independent derivatives and directional laws for lazy branches and scans.
use alea::{
    autodiff::autodiff::AutodiffError, capability::PotentialHessianDerivative,
    target::LogDensityGradient,
};
use alea_enzyme_conformance::{
    tensor_checkpoint, tensor_checkpoint_zero, tensor_control, tensor_empty, tensor_zero_scan,
};

#[test]
fn source_checkpoint_long_replay_and_mixed_native_prior_chart_match_independent_scores() {
    use alea_enzyme_conformance::{
        tensor_checkpoint_long as long, tensor_checkpoint_mixed as mixed,
    };
    const {
        assert!(long::SOURCE_CHECKPOINTED && mixed::SOURCE_CHECKPOINTED);
    }
    const {
        assert!(!tensor_control::SOURCE_CHECKPOINTED);
    }
    for q in [[-0.7, -0.57], [0.4, 0.53], [2., 1.]] {
        let [mut r, b] = q;
        let (mut da, mut db) = (1., 0.);
        for step in 0..257 {
            db = db * b + r;
            da *= b;
            r = r * b + [-0.1, 0.25, 0.5][step % 3];
        }
        let mut g = [41.; 2];
        let target = long::target();
        close(target.logp_grad(&q, &mut g).unwrap(), r.abs().ln());
        close(g[0], da / r);
        close(g[1], db / r);
        for j in 0..2 {
            let mut lo = q;
            let mut hi = q;
            lo[j] -= 1e-6;
            hi[j] += 1e-6;
            let fd = (long::log_density(&hi) - long::log_density(&lo)) / 2e-6;
            assert!((fd - g[j]).abs() < 2e-5 * (1. + g[j].abs()));
        }
    }
    for q in [[-0.7, -0.57], [0.4, 0.53], [0., 0.]] {
        let a: f64 = q[0];
        let a = a.exp();
        let (l, s, _, _) = reference([a, q[1]], false);
        let mut g = [41.; 2];
        let expected = l * l - 0.5 * a * a - 0.5 * (2. * std::f64::consts::PI).ln() + q[0];
        close(mixed::target().logp_grad(&q, &mut g).unwrap(), expected);
        close(g[0], (2. * l * s[0] - a) * a + 1.);
        close(g[1], 2. * l * s[1]);
        let mut h = [41.; 2];
        mixed::composition::objective()
            .logp_grad(&q, &mut h)
            .unwrap();
        close(h[0], g[0] - 1.);
        close(h[1], g[1]);
    }
    let mut g = [41.; 2];
    assert!(
        tensor_checkpoint_zero::target()
            .logp_grad(&[0., 0.5], &mut g)
            .is_err()
    );
    // Raw backend outputs are disposable after failure (as on the existing
    // compiler path); successful recovery must overwrite every coordinate.
    assert!(g.iter().all(|v| v.is_nan()));
    assert!(
        tensor_checkpoint_zero::target()
            .logp_grad(&[1., 0.5], &mut g)
            .is_ok()
    );
}

#[test]
fn source_checkpoint_gradient_reuses_explicit_workspace_without_heap_tape() {
    use alea_enzyme_conformance::{
        tensor_checkpoint_long as checkpoint, tensor_retained_long as retained,
    };
    let q = [0.4, 0.53];
    let mut expected = [0.; 2];
    let mut actual = [0.; 2];
    let baseline = retained::target();
    let candidate = checkpoint::target();
    let before = allocation_counter::measure(|| {
        baseline.logp_grad(&q, &mut expected).unwrap();
    });
    let after = allocation_counter::measure(|| {
        candidate.logp_grad(&q, &mut actual).unwrap();
    });
    for j in 0..2 {
        close(actual[j], expected[j]);
    }
    assert_eq!(after.count_total, 0);
    println!("source scan 257 retained {before:?}; checkpoint {after:?}");
}

#[test]
fn source_replay_preserves_first_activity_and_rejects_a_genuine_singular_score() {
    use alea_enzyme_conformance::tensor_checkpoint_activity as model;
    let target = model::target();
    let mut g = [41.];
    for gain in [0.25_f64, 0.5, 2.] {
        let middle = gain.sqrt() + gain;
        let value = middle.sqrt() + gain;
        let score = 0.5 * (0.5 / gain.sqrt() + 1.) / middle.sqrt() + 1.;
        close(target.logp_grad(&[gain], &mut g).unwrap(), value);
        close(g[0], score);
    }
    assert!(target.logp_grad(&[0.], &mut g).is_err());
    assert!(target.logp_grad(&[-1.], &mut g).is_err());
    assert!(target.logp_grad(&[0.5], &mut g).is_ok());
    assert!(g[0].is_finite());
}

#[test]
fn compact_source_adjoints_preserve_aliases_and_immutable_singular_primals() {
    use alea_enzyme_conformance::tensor_checkpoint_immutable as model;
    const {
        assert!(model::SOURCE_CHECKPOINTED);
    }
    let target = model::target();
    let mut g = [41.];
    for gain in [-2., 0., 0.5, 3.] {
        close(target.logp_grad(&[gain], &mut g).unwrap(), 6. * gain);
        close(g[0], 6.);
        let fd = (model::log_density(&[gain + 1e-6]) - model::log_density(&[gain - 1e-6])) / 2e-6;
        assert!((g[0] - fd).abs() < 2e-8);
    }
    // Compacting derivative storage does not bypass overflow admission.
    assert!(target.logp_grad(&[1e308], &mut g).is_err());
    assert!(target.logp_grad(&[0.5], &mut g).is_ok());
    close(g[0], 6.);
    let constant = alea_enzyme_conformance::tensor_checkpoint_constant::target();
    for gain in [-2., 0., 1e308] {
        close(constant.logp_grad(&[gain], &mut g).unwrap(), 6.);
        close(g[0], 0.);
    }
}

#[track_caller]
fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 2e-11 * (1. + expected.abs()),
        "{actual} != {expected}"
    );
}

#[test]
fn compact_source_scan_routes_rows_and_shared_captures_in_operand_order() {
    use alea_enzyme_conformance::tensor_checkpoint_routes as model;
    let target = model::target();
    for q in [
        [0.2, 0.1, -0.4, 0.5, 0.7],
        [-1., 2., 0., -0.5, -0.8],
        [1., 2., 3., 4., 0.],
    ] {
        let b = q[4];
        let (mut value, mut db) = (q[0], 0.);
        for row in &q[1..4] {
            db = db * b + value + row;
            value = value * b + row * b + 0.25;
        }
        let expected = [b * b * b, b * b * b, b * b, b, db];
        let mut g = [41.; 5];
        close(target.logp_grad(&q, &mut g).unwrap(), value);
        for j in 0..5 {
            close(g[j], expected[j]);
            let mut lo = q;
            let mut hi = q;
            lo[j] -= 1e-6;
            hi[j] += 1e-6;
            let fd = (model::log_density(&hi) - model::log_density(&lo)) / 2e-6;
            assert!((g[j] - fd).abs() < 2e-8 * (1. + g[j].abs()));
        }
    }
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
    check(tensor_checkpoint::target(), false);
    check(tensor_checkpoint_zero::target(), true);
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
    check(
        alea_enzyme_conformance::tensor_checkpoint_empty::target(),
        [-1.],
        [0.],
    );
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
