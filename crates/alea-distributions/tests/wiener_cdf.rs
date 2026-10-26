use alea_distributions::wiener::{
    Boundary, Wiener4, Wiener4Params, WienerObservation, WienerOptions,
};
#[test]
fn boundary_probabilities_match_independent_ball_enclosures() {
    let mut count = 0;
    for line in include_str!("support/wiener_cdf_balls.csv").lines().skip(1) {
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
        let f = Wiener4
            .try_log_cdf(&obs, &p, WienerOptions::new(1e-12))
            .unwrap_or_else(|e| panic!("row {count}: {x:?}: {e:?}"));
        let h = Wiener4
            .try_log_survival(&obs, &p, WienerOptions::new(1e-12))
            .unwrap_or_else(|e| panic!("tail row {count}: {x:?}: {e:?}"));
        for (actual, index) in [(f, 6), (h, 8)] {
            assert!(
                (actual - x[index]).abs() <= x[index + 1] + 2e-10 * (1. + x[index].abs()),
                "row {count} input {x:?}: {actual} vs {}",
                x[index]
            );
        }
        count += 1;
    }
    assert_eq!(count, 400);
}

#[test]
fn cdf_time_derivative_is_the_density() {
    for boundary in [Boundary::Upper, Boundary::Lower] {
        for t in [0.01, 0.1, 0.15, 0.3, 1., 10.] {
            let p = Wiener4Params::from_array([1., 0.1, 0.4, 1.5]);
            let step = 1e-6 * t;
            let obs = |offset| WienerObservation {
                rt: 0.1 + t + offset,
                boundary,
            };
            let logf = |offset| {
                Wiener4
                    .try_log_cdf(&obs(offset), &p, WienerOptions::new(1e-12))
                    .unwrap()
            };
            let derivative = (logf(step) - logf(-step)) / (2. * step);
            let expected = (Wiener4.fused(&obs(0.), &p, 1e-13).log_prob - logf(0.)).exp();
            assert!(
                (derivative - expected).abs() < 1e-5 * expected.max(1e-8),
                "t={t}: {derivative} vs {expected}"
            );
        }
    }
}

#[test]
fn cdf_and_tail_have_boundary_subdistribution_semantics() {
    let p = Wiener4Params::from_array([1., 0.1, 0.3, 0.]);
    for (boundary, mass) in [(Boundary::Upper, 0.3), (Boundary::Lower, 0.7)] {
        let obs = |rt| WienerObservation { rt, boundary };
        assert_eq!(
            Wiener4
                .try_cdf(&obs(0.05), &p, WienerOptions::default())
                .unwrap(),
            0.
        );
        assert!(
            (Wiener4
                .try_log_survival(&obs(0.05), &p, WienerOptions::default())
                .unwrap()
                .exp()
                - mass)
                .abs()
                < 1e-15
        );
        assert!(
            (Wiener4
                .try_cdf(&obs(f64::INFINITY), &p, WienerOptions::default())
                .unwrap()
                - mass)
                .abs()
                < 1e-15
        );
        let f = Wiener4
            .try_cdf(&obs(0.4), &p, WienerOptions::default())
            .unwrap();
        let h = Wiener4
            .try_log_survival(&obs(0.4), &p, WienerOptions::default())
            .unwrap()
            .exp();
        assert!((f + h - mass).abs() < 1e-12);
    }
}

#[test]
fn cdf_differences_match_time_variability_integration() {
    use alea_distributions::wiener::{Wiener7, Wiener7Params};
    for boundary in [Boundary::Upper, Boundary::Lower] {
        for delta in [-5., 0., 5.] {
            for rt in [0.18, 0.5, 2.] {
                for width in [1e-9, 0.03, 0.2] {
                    let obs = WienerObservation { rt, boundary };
                    let p =
                        Wiener7Params::with_params(1., 0.15, 0.3, delta, 0., width, 0.).unwrap();
                    let value = Wiener7
                        .try_log_prob(&obs, &p, WienerOptions::new(1e-9))
                        .unwrap()
                        .log_prob;
                    let integrated = Wiener7
                        .try_fused(&obs, &p, WienerOptions::new(1e-11))
                        .unwrap();
                    assert!(
                        (value - integrated.log_prob).abs()
                            < 1e-8 * (1. + integrated.log_prob.abs()),
                        "rt={rt},width={width},delta={delta},{boundary:?}: {value} vs {}",
                        integrated.log_prob
                    );
                }
            }
        }
    }
}

#[test]
fn interval_probabilities_partition_boundary_mass() {
    let p = Wiener4Params::from_array([1., 0.1, 0.4, -1.]);
    for boundary in [Boundary::Upper, Boundary::Lower] {
        let options = WienerOptions::new(1e-10);
        let edges = [0., 0.15, 0.3, 1., f64::INFINITY];
        let sum: f64 = edges
            .windows(2)
            .map(|pair| {
                Wiener4
                    .try_log_interval_prob(pair[0], pair[1], boundary, &p, options)
                    .unwrap()
                    .exp()
            })
            .sum();
        let total = Wiener4
            .try_cdf(
                &WienerObservation {
                    rt: f64::INFINITY,
                    boundary,
                },
                &p,
                options,
            )
            .unwrap();
        assert!((sum - total).abs() < 1e-10);
        assert!(
            Wiener4
                .try_log_interval_prob(1., 0., boundary, &p, options)
                .is_err()
        );
    }
}
