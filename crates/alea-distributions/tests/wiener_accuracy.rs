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

#[test]
fn wiener7_reports_separate_estimated_error_contributions() {
    use alea_distributions::wiener::{Wiener7, Wiener7Params, WienerOptions};
    for boundary in [Boundary::Upper, Boundary::Lower] {
        for (sw, st) in [(0., 0.), (0.2, 0.), (0., 0.1), (0.2, 0.1)] {
            let obs = WienerObservation { rt: 0.8, boundary };
            let p = Wiener7Params::with_params(1.2, 0.15, 0.4, 0.7, sw, st, 0.3).unwrap();
            let (e, r) = Wiener7
                .try_fused_with_error_estimate(&obs, &p, WienerOptions::new(1e-7))
                .unwrap();
            assert!(r.total_log_prob <= 1e-7);
            assert!(
                r.total_log_prob >= r.series_log_prob && r.total_log_prob >= r.quadrature_log_prob
            );
            for ((total, series), quad) in r
                .total_scores
                .iter()
                .zip(r.series_scores)
                .zip(r.quadrature_scores)
            {
                assert!(*total >= series && *total >= quad);
            }
            let reference = Wiener7
                .try_fused(&obs, &p, WienerOptions::new(1e-11))
                .unwrap();
            assert!((e.log_prob - reference.log_prob).abs() < 1e-7);
            for (a, b) in e.grad_array().into_iter().zip(reference.grad_array()) {
                assert!((a - b).abs() < 1e-6 * (1. + b.abs()));
            }
        }
    }
}

#[test]
fn uniform_start_marginalization_matches_rigorous_continuous_ball_integrals() {
    use alea_distributions::wiener::{Wiener7, Wiener7Params, WienerOptions};
    let mut count = 0;
    for line in include_str!("support/wiener7_balls.csv").lines().skip(1) {
        let x: Vec<f64> = line.split(',').map(|s| s.parse().unwrap()).collect();
        let obs = WienerObservation {
            rt: x[0],
            boundary: if x[5] == 1. {
                Boundary::Lower
            } else {
                Boundary::Upper
            },
        };
        let p = Wiener7Params::with_params(1., 0., x[1], x[2], x[4], 0., x[3]).unwrap();
        let e = Wiener7
            .try_fused(&obs, &p, WienerOptions::new(1e-10))
            .unwrap();
        for (i, actual) in std::iter::once(e.log_prob)
            .chain(e.grad_array())
            .enumerate()
        {
            let reference = x[8 + 2 * i];
            let radius = x[9 + 2 * i];
            assert!(
                (actual - reference).abs() <= radius + 2e-9 * (1. + reference.abs()),
                "row {count},component {i}: {actual} vs {reference}"
            );
        }
        count += 1;
    }
    assert_eq!(count, 48);
}
