//! The independent oracle encloses its series and arithmetic with Arb balls.
//! Rust comparisons separately allow f64 roundoff; truncation is not roundoff.
use alea_distributions::wiener::{
    Boundary, Wiener4, Wiener4Params, Wiener5, Wiener5Params, WienerObservation,
};

#[test]
fn componentwise_evaluations_match_independent_ball_enclosures() {
    let mut count = 0;
    for line in include_str!("support/wiener_balls.csv").lines().skip(1) {
        let x: Vec<f64> = line.split(',').map(|s| s.parse().unwrap()).collect();
        let obs = WienerObservation {
            rt: x[0],
            boundary: if x[6] == 1. {
                Boundary::Lower
            } else {
                Boundary::Upper
            },
        };
        let p = Wiener5Params::from_array([x[1], x[2], x[3], x[4], x[5]]);
        let (e, r) = Wiener5.try_fused_with_truncation(&obs, &p, 1e-9).unwrap();
        assert!(r.log_prob <= 1e-9 && r.scores.iter().all(|v| *v <= 1e-9));
        for (i, (actual, tail)) in std::iter::once(e.log_prob)
            .chain(e.grad_array())
            .zip(std::iter::once(r.log_prob).chain(r.scores))
            .enumerate()
        {
            let reference = x[9 + 2 * i];
            let radius = x[10 + 2 * i];
            let roundoff_allowance = 2e-10 * (1. + reference.abs());
            assert!(
                (actual - reference).abs() <= radius + tail + roundoff_allowance,
                "row {count}, component {i}: {actual} vs {reference}, {r:?}"
            );
        }
        if x[5] == 0. {
            let p = Wiener4Params::from_array([x[1], x[2], x[3], x[4]]);
            let (four, _) = Wiener4.try_fused_with_truncation(&obs, &p, 1e-9).unwrap();
            assert!((four.log_prob - e.log_prob).abs() < 1e-9 * (1. + e.log_prob.abs()));
            for (a, b) in four.grad_array().into_iter().zip(e.grad_array()) {
                assert!((a - b).abs() < 1e-9 * (1. + b.abs()));
            }
        }
        count += 1;
    }
    assert_eq!(count, 288);
}

#[test]
fn invalid_accuracy_budgets_are_errors() {
    let obs = WienerObservation {
        rt: 1.,
        boundary: Boundary::Upper,
    };
    for tolerance in [0., -1., f64::NAN, f64::INFINITY] {
        assert!(
            Wiener4
                .try_fused_with_truncation(&obs, &Wiener4Params::default(), tolerance)
                .is_err()
        );
    }
}
