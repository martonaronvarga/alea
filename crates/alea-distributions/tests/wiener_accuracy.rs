//! The independent oracle encloses its series and arithmetic with Arb balls.
//! Rust comparisons separately allow f64 roundoff; truncation is not roundoff.
use alea_distributions::wiener::{
    Boundary, Wiener4, Wiener4Params, Wiener5, Wiener5Params, WienerObservation,
};

#[test]
fn clipped_boundary_fluxes_match_independent_integrated_cdf_balls() {
    use alea_distributions::wiener::{Wiener7, Wiener7Params, WienerOptions};
    let mut count = 0;
    for line in include_str!("support/wiener7_flux_balls.csv")
        .lines()
        .skip(1)
    {
        let x: Vec<f64> = line.split(',').map(|s| s.parse().unwrap()).collect();
        let event = WienerObservation {
            rt: x[0],
            boundary: if x[7] == 1. {
                Boundary::Lower
            } else {
                Boundary::Upper
            },
        };
        let p = Wiener7Params::with_params(x[1], x[2], x[3], x[4], x[5], x[6], 0.).unwrap();
        let evaluation = Wiener7
            .try_fused(&event, &p, WienerOptions::new(1e-8))
            .unwrap();
        for (i, actual) in std::iter::once(evaluation.log_prob)
            .chain(evaluation.grad_array())
            .enumerate()
        {
            let reference = x[8 + 2 * i];
            assert!(
                (actual - reference).abs() <= x[9 + 2 * i] + 1e-8 * reference.abs().max(1.),
                "row {count}, component {i}: {actual} vs {reference}"
            );
        }
        count += 1;
    }
    assert_eq!(count, 6);
}

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

#[test]
fn nested_and_clipped_variability_match_rigorous_ball_integrals() {
    use alea_distributions::wiener::{Wiener7, Wiener7Params, WienerOptions};
    let mut count = 0;
    for line in include_str!("support/wiener7_nested_balls.csv")
        .lines()
        .skip(1)
    {
        let x: Vec<f64> = line.split(',').map(|s| s.parse().unwrap()).collect();
        let obs = WienerObservation {
            rt: x[0],
            boundary: if x[8] == 1. {
                Boundary::Lower
            } else {
                Boundary::Upper
            },
        };
        let p = Wiener7Params::with_params(x[1], x[2], x[3], x[4], x[6], x[7], x[5]).unwrap();
        let e = Wiener7
            .try_fused(&obs, &p, WienerOptions::new(1e-9))
            .unwrap();
        for (i, value) in std::iter::once(e.log_prob)
            .chain(e.grad_array())
            .enumerate()
        {
            let reference = x[11 + 2 * i];
            let radius = x[12 + 2 * i];
            assert!(
                (value - reference).abs() <= radius + 1e-9 * (1. + reference.abs()),
                "row {count},component {i}: {value} vs {reference}"
            );
        }
        if x[5] == 0. && x[6] == 0. && x[7] == x[0] - x[2] {
            let base = Wiener4Params::from_array([x[1], x[2], x[3], x[4]]);
            let cdf = Wiener4
                .try_fused_cdf(&obs, &base, WienerOptions::new(1e-11))
                .unwrap();
            assert!((cdf.log_prob - (x[11] + x[7].ln())).abs() < 1e-10);
            for (i, value) in cdf.grad_array().iter().enumerate() {
                assert!((*value - x[13 + 2 * i]).abs() < 1e-9 * (1. + value.abs()));
            }
        }
        count += 1;
    }
    assert_eq!(count, 8);
}

#[test]
fn cdf_scores_match_independent_continuous_ball_integrals() {
    use alea_distributions::wiener::WienerOptions;
    let mut count = 0;
    for line in include_str!("support/wiener_cdf_gradient_balls.csv")
        .lines()
        .skip(1)
    {
        let x: Vec<f64> = line.split(',').map(|s| s.parse().unwrap()).collect();
        let obs = WienerObservation {
            rt: x[0],
            boundary: if x[3] == 1. {
                Boundary::Lower
            } else {
                Boundary::Upper
            },
        };
        let p = Wiener4Params::from_array([1., 0., x[1], x[2]]);
        let result = Wiener4.try_fused_cdf(&obs, &p, WienerOptions::new(1e-10));
        let e = result.unwrap_or_else(|error| panic!("row {count}: {error}"));
        for (i, actual) in std::iter::once(e.log_prob)
            .chain(e.grad_array())
            .enumerate()
        {
            let reference = x[5 + 2 * i];
            assert!(
                (actual - reference).abs() <= x[6 + 2 * i] + 1e-9 * (1. + reference.abs()),
                "row {count}, component {i}: {actual} vs {reference}"
            );
        }
        count += 1;
    }
    assert_eq!(count, 24);
}

#[test]
fn complete_wiener7_grid_matches_ball_references() {
    use alea_distributions::wiener::{Wiener7, Wiener7Params, WienerOptions};
    let mut count = 0;
    for line in include_str!("support/wiener7_grid_balls.csv")
        .lines()
        .skip(1)
    {
        let mut fields = line.split(',');
        let group = fields.next().unwrap();
        let x: Vec<f64> = fields.map(|s| s.parse().unwrap()).collect();
        let obs = WienerObservation {
            rt: x[0],
            boundary: if x[8] == 1. {
                Boundary::Lower
            } else {
                Boundary::Upper
            },
        };
        let params = Wiener7Params::with_params(x[1], x[2], x[3], x[4], x[6], x[7], x[5]).unwrap();
        let options = WienerOptions::new(1e-9);
        let fused = Wiener7.try_fused(&obs, &params, options).unwrap();
        let density = Wiener7
            .try_log_prob(&obs, &params, options)
            .unwrap_or_else(|e| panic!("{group}: {params:?}, {obs:?}: {e}"));
        assert!(
            (density.log_prob - x[9]).abs() <= x[10] + 1e-9 * (1. + x[9].abs()),
            "{group} density"
        );
        for (i, actual) in std::iter::once(fused.log_prob)
            .chain(fused.grad_array())
            .enumerate()
        {
            let reference = x[9 + 2 * i];
            assert!(
                (actual - reference).abs() <= x[10 + 2 * i] + 1e-9 * (1. + reference.abs()),
                "row {count}, {group}, component {i}: {actual} vs {reference}"
            );
        }
        count += 1;
    }
    assert_eq!(count, 76);
}

#[test]
fn large_drift_variance_retains_density_and_score_precision() {
    use alea_distributions::wiener::WienerOptions;
    let mut count = 0;
    for line in include_str!("support/wiener_variance_balls.csv")
        .lines()
        .skip(1)
    {
        let x: Vec<f64> = line.split(',').map(|x| x.parse().unwrap()).collect();
        let event = WienerObservation {
            rt: x[0],
            boundary: if x[6] == 1. {
                Boundary::Lower
            } else {
                Boundary::Upper
            },
        };
        let params = Wiener5Params::from_array([x[1], x[2], x[3], x[4], x[5]]);
        let result = Wiener5
            .try_fused(&event, &params, WienerOptions::new(1e-12))
            .unwrap();
        for (i, value) in std::iter::once(result.log_prob)
            .chain(result.grad_array())
            .enumerate()
        {
            let reference = x[9 + 2 * i];
            assert!(
                (value - reference).abs() <= x[10 + 2 * i] + 2e-12 * reference.abs().max(1.),
                "row {count},component {i}: {value} vs {reference}"
            );
        }
        count += 1;
    }
    assert_eq!(count, 24);
}
