//! Density evaluation has its own truncation and does not require finite scores.
use alea_distributions::wiener::*;

#[test]
fn densities_match_independent_ball_references() {
    let mut count = 0;
    for table in [
        include_str!("support/wiener_balls.csv"),
        include_str!("support/wiener_variance_balls.csv"),
    ] {
        for line in table.lines().skip(1) {
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
            let value = Wiener5
                .try_log_prob(&obs, &p, WienerOptions::new(1e-12))
                .unwrap();
            let tolerance = x[10] + 1e-9 * x[9].abs().max(1.);
            assert!(
                (value.log_prob - x[9]).abs() <= tolerance,
                "{x:?}: {}",
                value.log_prob
            );
            assert!(value.grad_array().iter().all(|x| x.is_nan()));
            if x[5] == 0. {
                assert_eq!(
                    value.log_prob,
                    Wiener4
                        .try_log_prob(
                            &obs,
                            &Wiener4Params::from_array([x[1], x[2], x[3], x[4]]),
                            WienerOptions::new(1e-12)
                        )
                        .unwrap()
                        .log_prob
                );
            }
            count += 1;
        }
    }
    assert_eq!(count, 312);
}

#[test]
fn density_survives_a_score_overflow() {
    let obs = WienerObservation {
        rt: 1e-310,
        boundary: Boundary::Lower,
    };
    let p = Wiener4Params::from_array([1e-100, 0., 1e-55, 0.]);
    let value = Wiener4
        .try_log_prob(&obs, &p, WienerOptions::default())
        .unwrap();
    // All noncentral images are exponentially negligible at this time scale.
    let distance: f64 = 1e-155;
    let expected = distance.ln()
        - 0.5 * std::f64::consts::TAU.ln()
        - 1.5 * obs.rt.ln()
        - 0.5 * (distance * distance / obs.rt);
    assert!((value.log_prob - expected).abs() < 1e-10);
    assert!(value.grad_array().iter().all(|x| x.is_nan()));
    assert!(
        Wiener4
            .try_fused(&obs, &p, WienerOptions::default())
            .is_err()
    );
    let p5 = Wiener5Params::from_array([p.alpha, p.tau, p.beta, p.delta, 0.]);
    assert_eq!(
        value.log_prob,
        Wiener5
            .try_log_prob(&obs, &p5, WienerOptions::default())
            .unwrap()
            .log_prob
    );
}

#[test]
fn density_endpoints_match_extreme_support_lookup() {
    for line in include_str!("support/wiener_extremes.csv").lines().skip(1) {
        let cells: Vec<_> = line.split(',').collect();
        let x: Vec<f64> = cells[..12].iter().map(|s| s.parse().unwrap()).collect();
        let obs = WienerObservation {
            rt: x[0],
            boundary: if cells[12] == "lower" {
                Boundary::Lower
            } else {
                Boundary::Upper
            },
        };
        let p = Wiener5Params::from_array([x[1], x[2], x[3], x[4], x[5]]);
        let value = Wiener5
            .try_log_prob(&obs, &p, WienerOptions::default())
            .unwrap();
        assert!(
            (value.log_prob - x[6]).abs() <= 1e-8 + 64. * f64::EPSILON * x[6].abs(),
            "{line}: {}",
            value.log_prob
        );
    }
}
