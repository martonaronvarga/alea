//! One-sided width derivatives follow the likelihood's uniform interval convention.
use alea_distributions::wiener::{
    Boundary, Quadrature, Wiener7, Wiener7Params, WienerObservation, WienerOptions,
};

#[test]
fn nondecision_width_right_score_is_half_the_location_score() {
    for boundary in [Boundary::Upper, Boundary::Lower] {
        for sw in [0., 0.2] {
            for quadrature in [
                Quadrature::Adaptive,
                Quadrature::FixedGaussLegendre { order: 25 },
            ] {
                let obs = WienerObservation { rt: 0.8, boundary };
                let p = Wiener7Params::with_params(1.2, 0.15, 0.4, 0.7, sw, 0., 0.3).unwrap();
                let options = WienerOptions::new(1e-9)
                    .with_inner_precision(1e-12)
                    .with_quadrature(quadrature);
                let e = Wiener7.try_fused(&obs, &p, options).unwrap();
                assert_eq!(e.grad.s_tau, 0.5 * e.grad.tau);
                let h = 1e-4;
                let value = |width| {
                    let p =
                        Wiener7Params::with_params(1.2, 0.15, 0.4, 0.7, sw, width, 0.3).unwrap();
                    Wiener7.try_log_prob(&obs, &p, options).unwrap().log_prob
                };
                let fd = (-3. * e.log_prob + 4. * value(h) - value(2. * h)) / (2. * h);
                assert!(
                    (e.grad.s_tau - fd).abs() < 1e-6 * (1. + fd.abs()),
                    "{quadrature:?} sw={sw} {boundary:?}: {} vs {fd}",
                    e.grad.s_tau
                );
                let unchecked = Wiener7.fused(&obs, &p, 1e-9);
                assert_eq!(unchecked.grad.s_tau, 0.5 * unchecked.grad.tau);
            }
        }
    }
}

#[test]
fn scaled_tail_retains_the_same_zero_width_limit() {
    let obs = WienerObservation {
        rt: 1e-6,
        boundary: Boundary::Upper,
    };
    let p = Wiener7Params::with_params(0.1, 0., 0.0001, -10., 1e-8, 0., 0.1).unwrap();
    let e = Wiener7
        .try_fused(&obs, &p, WienerOptions::new(1e-8))
        .unwrap();
    assert!(e.log_prob < -745.);
    assert_eq!(e.grad.s_tau, 0.5 * e.grad.tau);
}
